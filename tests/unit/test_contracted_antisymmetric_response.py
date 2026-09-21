"""Tests for Bianchi-reduced and compiled analytical RFS contractions."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from core.antisymmetric_response_rfs import antisymmetric_response_rfs_native
from core.constants import C_MMNS
from core.contracted_antisymmetric_response_numba import (
    antisymmetric_response_charge_force_strict_serial,
    antisymmetric_response_rfs_strict_serial,
)
from core.magnetic_dipole import boost_rest_polarization

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0))
_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _derivative_compatible_partial(rng: np.random.Generator) -> np.ndarray:
    partial2_a = rng.normal(size=(4, 4, 4))
    partial2_a = 0.5 * (partial2_a + np.swapaxes(partial2_a, 0, 1))
    result = np.empty((4, 6))
    for derivative in range(4):
        for pair_index, (mu, nu) in enumerate(_PAIRS):
            result[derivative, pair_index] = (
                _SIGNS[mu] * partial2_a[derivative, mu, nu]
                - _SIGNS[nu] * partial2_a[derivative, nu, mu]
            )
    return result


@pytest.mark.parametrize("speed", (0.0, 0.5, 0.9, 0.99, 0.9999))
@pytest.mark.parametrize("seed", range(8))
def test_compiled_contractions_match_reference(speed: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    beta = speed * direction
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = C_MMNS * gamma * np.concatenate(((1.0,), beta))
    rest_spin = rng.normal(size=3)
    rest_spin /= np.linalg.norm(rest_spin)
    spin = boost_rest_polarization(rest_spin, beta)
    field = rng.normal(scale=2.0e-3, size=6)
    partial = _derivative_compatible_partial(rng) * 4.0e-4
    kwargs = dict(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        antisymmetric_response=field,
        charge_native=-0.8,
        mass_amu=0.000548579909,
        magnetic_moment_native=-1.7e-3,
        invariant_spin_native=0.5,
    )
    expected = antisymmetric_response_rfs_native(
        partial_antisymmetric_response=partial,
        **kwargs,
    )
    compiled_values = antisymmetric_response_rfs_strict_serial(
        velocity,
        spin,
        field,
        partial,
        kwargs["charge_native"],
        kwargs["mass_amu"],
        kwargs["magnetic_moment_native"],
        kwargs["invariant_spin_native"],
    )
    charge_force = antisymmetric_response_charge_force_strict_serial(
        velocity,
        field,
        kwargs["charge_native"],
    )
    np.testing.assert_allclose(
        charge_force,
        expected.charge_four_force,
        rtol=5.0e-15,
        atol=1.0e-18,
    )
    for actual, reference in zip(
        compiled_values,
        (
            expected.charge_four_force,
            expected.dipole_four_force,
            expected.total_four_force,
            expected.spin_rhs,
        ),
    ):
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=2.0e-14,
            atol=1.0e-18,
        )
