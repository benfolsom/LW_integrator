"""GPU Green-cache orchestration controls that do not require a device."""

import numpy as np
import pytest

from core.pic import gpu_backend
from core.pic.gpu_backend import GPUBackend


class ArrayGPU(GPUBackend):
    xp = np
    dtype = "float32"
    name = "array_cache_control"

    def array(self, value, dtype=None):
        return np.asarray(value, dtype=dtype or self.dtype)

    def to_host(self, value):
        return np.asarray(value)

    def synchronize(self, *values):
        pass


def test_alternating_spacings_reuse_exact_spectra_and_bound_memory(monkeypatch):
    builds = []
    original = gpu_backend.green_mesh

    def measured(shape, spacing, component):
        builds.append((tuple(spacing), component))
        return original(shape, spacing, component)

    monkeypatch.setattr(gpu_backend, "green_mesh", measured)
    backend = ArrayGPU()
    shape = (4, 5, 6)
    spacings = ([0.001, 0.002, 0.003], [0.001, 0.002, 0.004])
    spectra = [backend._spectrum(shape, h, 0) for h in spacings]
    for h, spectrum in zip(spacings, spectra):
        assert backend._spectrum(shape, h, 0) is spectrum
    assert len(builds) == 2
    backend._spectrum(shape, [0.001, 0.002, 0.005], 0)
    assert len(backend._spectrum_cache) == 2
    assert (shape, tuple(spacings[0])) not in backend._spectrum_cache
    # Revisiting an evicted geometry recomputes exactly the same coefficients.
    rebuilt = backend._spectrum(shape, spacings[0], 0)
    np.testing.assert_array_equal(rebuilt, spectra[0])


def test_float32_canonical_geometry_reuses_subprecision_changes(monkeypatch):
    backend = ArrayGPU()
    backend.green_spacing_dtype = "float32"
    builds = []
    original = gpu_backend.green_mesh

    def measured(shape, spacing, component):
        builds.append(np.asarray(spacing).copy())
        assert np.asarray(spacing).dtype == np.float64
        return original(shape, spacing, component)

    monkeypatch.setattr(gpu_backend, "green_mesh", measured)
    h = np.array([0.000375, 0.000375, 0.001123046875])
    saved = h.copy()
    shape = (4, 4, 8)
    first = backend._spectrum(shape, h, 2)
    perturbed = h * (1 + 1e-12)
    assert backend._spectrum(shape, perturbed, 2) is first
    assert len(builds) == 1
    np.testing.assert_array_equal(h, saved)
    np.testing.assert_array_equal(builds[0], h.astype(np.float32).astype(float))
    # Resolving a new float32 spacing must build a new spectrum, never use a
    # tolerance-based or stale entry from a physically different geometry.
    different = h.astype(np.float32)
    different[2] = np.nextafter(different[2], np.float32(np.inf))
    assert backend._spectrum(shape, different, 2) is not first
    assert len(builds) == 2


def test_components_shapes_and_potential_are_distinct_cache_entries():
    backend = ArrayGPU()
    h = [0.001, 0.002, 0.003]
    charge = np.zeros((4, 5, 6), dtype=np.float32)
    charge[2, 2, 2] = 1e-12
    electric, phi = backend.solve(charge, h, potential=True)
    fresh_e, fresh_phi = ArrayGPU().solve(charge, h, potential=True)
    np.testing.assert_array_equal(electric, fresh_e)
    np.testing.assert_array_equal(phi, fresh_phi)
    key = (charge.shape, tuple(h))
    assert set(backend._spectrum_cache[key]) == {0, 1, 2, None}
    backend._spectrum((4, 5, 8), h, 0)
    assert len(backend._spectrum_cache) == 2


