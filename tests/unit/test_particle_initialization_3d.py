"""Unit tests for the general 3D particle initializer."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.particle_initialization import create_particle_state, create_particle_state_3d

AMU_TO_MEV = 931.494


def _ke_to_pz(kinetic_energy_mev: float, mass_amu: float) -> float:
    """Match the relativistic momentum magnitude used by create_particle_state_3d."""
    rest = mass_amu * AMU_TO_MEV
    gamma = 1.0 + kinetic_energy_mev / rest
    beta = np.sqrt(max(0.0, 1.0 - 1.0 / gamma**2))
    return gamma * mass_amu * beta * C_MMNS


def test_z_axis_parity_matches_create_particle_state():
    starting_dist = 1000.0
    transv_dist = 0.25
    transv_mom = 1.5e-3
    ke_mev = 35.0
    mass_amu = 0.00054857990907
    stripped = 1.0
    pcount = 4
    charge_sign = -1.0

    starting_pz = _ke_to_pz(ke_mev, mass_amu)

    legacy, legacy_rest = create_particle_state(
        starting_distance=starting_dist,
        transv_momentum=transv_mom,
        starting_pz=starting_pz,
        stripped_ions=stripped,
        particle_mass_amu=mass_amu,
        transv_distance=transv_dist,
        particle_count=pcount,
        charge_sign=charge_sign,
    )
    state, rest = create_particle_state_3d(
        starting_position_mm=(0.0, transv_dist, starting_dist),
        momentum_axis=(0.0, 0.0, 1.0),
        kinetic_energy_mev=ke_mev,
        stripped_ions=stripped,
        particle_mass_amu=mass_amu,
        particle_count=pcount,
        charge_sign=charge_sign,
        transverse_distance_mm=transv_dist,
        transverse_momentum=transv_mom,
        # Match legacy convention: transverse offset/momentum along y for z-axis propagation.
        transverse_axes=((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    )

    assert rest == pytest.approx(legacy_rest)

    # Longitudinal momentum, total momentum magnitude, gamma, beta, charge, mass
    # must match. Note: legacy applies transverse position along y but transverse
    # momentum along x; the 3D initializer applies both along the first transverse
    # axis. We therefore compare invariant physics quantities rather than
    # individual transverse components.
    np.testing.assert_allclose(state["pz"], legacy["pz"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(state["Pt"], legacy["Pt"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(state["gamma"], legacy["gamma"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(state["q"], legacy["q"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(state["m"], legacy["m"], rtol=1e-6, atol=1e-12)

    # |beta| should match (beta magnitude is invariant under transverse rotation).
    legacy_beta_mag = np.sqrt(legacy["bx"] ** 2 + legacy["by"] ** 2 + legacy["bz"] ** 2)
    state_beta_mag = np.sqrt(state["bx"] ** 2 + state["by"] ** 2 + state["bz"] ** 2)
    np.testing.assert_allclose(state_beta_mag, legacy_beta_mag, rtol=1e-6, atol=1e-12)

    # Longitudinal velocity component along z must match.
    np.testing.assert_allclose(state["bz"], legacy["bz"], rtol=1e-6, atol=1e-12)

    assert state["count"] == legacy["count"]


def test_y_axis_driver_momentum_along_y():
    mass_amu = 197.0
    ke_mev = 0.5
    rest = mass_amu * AMU_TO_MEV
    expected_gamma = 1.0 + ke_mev / rest

    state, returned_rest = create_particle_state_3d(
        starting_position_mm=(0.0, 0.0, 0.0),
        momentum_axis=(0.0, 1.0, 0.0),
        kinetic_energy_mev=ke_mev,
        stripped_ions=1.0,
        particle_mass_amu=mass_amu,
        particle_count=3,
        charge_sign=+1.0,
    )

    assert returned_rest == pytest.approx(rest)

    np.testing.assert_allclose(state["gamma"], expected_gamma, rtol=1e-9)
    assert np.all(state["py"] > 0)
    np.testing.assert_allclose(state["px"], 0.0, atol=1e-15)
    np.testing.assert_allclose(state["pz"], 0.0, atol=1e-15)

    assert np.all(np.abs(state["by"]) > 0)
    np.testing.assert_allclose(state["bz"], 0.0, atol=1e-12)
    np.testing.assert_allclose(state["bx"], 0.0, atol=1e-12)


def test_auto_transverse_axes_orthonormal_and_offset_perpendicular_to_y():
    mass_amu = 197.0
    transv_dist = 0.5

    state, _ = create_particle_state_3d(
        starting_position_mm=(0.0, 0.0, 0.0),
        momentum_axis=(0.0, 1.0, 0.0),
        kinetic_energy_mev=0.5,
        stripped_ions=1.0,
        particle_mass_amu=mass_amu,
        particle_count=2,
        charge_sign=+1.0,
        transverse_distance_mm=transv_dist,
    )

    # All particles should be offset perpendicular to y: y stays at 0.
    np.testing.assert_allclose(state["y"], 0.0, atol=1e-15)
    # The transverse offset must lie in the x-z plane with magnitude transv_dist.
    radius = np.sqrt(state["x"] ** 2 + state["z"] ** 2)
    np.testing.assert_allclose(radius, transv_dist, rtol=1e-12)

    # Re-derive the auto axes via the same helper to check orthonormality.
    from core.particle_initialization import _orthonormal_transverse_axes

    n = np.array([0.0, 1.0, 0.0])
    u, v = _orthonormal_transverse_axes(n)
    np.testing.assert_allclose(np.dot(u, n), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.dot(v, n), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.dot(u, v), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-15)


def test_longitudinal_span_even_distribution():
    mass_amu = 197.0
    span = 10.0
    pcount = 4
    centroid = (1.0, 2.0, 3.0)

    state, _ = create_particle_state_3d(
        starting_position_mm=centroid,
        momentum_axis=(0.0, 0.0, 1.0),
        kinetic_energy_mev=0.5,
        stripped_ions=1.0,
        particle_mass_amu=mass_amu,
        particle_count=pcount,
        charge_sign=+1.0,
        longitudinal_span_mm=span,
    )

    # Momentum axis is z, so longitudinal spread appears in z only.
    expected_z = centroid[2] + np.linspace(-0.5, 0.5, pcount) * span
    np.testing.assert_allclose(state["z"], expected_z, rtol=1e-12)
    np.testing.assert_allclose(state["x"], centroid[0], atol=1e-15)
    np.testing.assert_allclose(state["y"], centroid[1], atol=1e-15)

    # Total range equals span.
    assert state["z"].max() - state["z"].min() == pytest.approx(span, rel=1e-12)


def test_state_dict_structure_matches_legacy_keys_and_shapes():
    mass_amu = 197.0
    pcount = 5

    legacy, _ = create_particle_state(
        starting_distance=10.0,
        transv_momentum=0.0,
        starting_pz=100.0,
        stripped_ions=1.0,
        particle_mass_amu=mass_amu,
        transv_distance=0.0,
        particle_count=pcount,
        charge_sign=+1.0,
    )
    state, _ = create_particle_state_3d(
        starting_position_mm=(0.0, 0.0, 10.0),
        momentum_axis=(0.0, 0.0, 1.0),
        kinetic_energy_mev=1.0,
        stripped_ions=1.0,
        particle_mass_amu=mass_amu,
        particle_count=pcount,
        charge_sign=+1.0,
    )

    assert set(state.keys()) == set(legacy.keys())

    per_particle_keys = [
        "x",
        "y",
        "z",
        "t",
        "px",
        "py",
        "pz",
        "Px",
        "Py",
        "Pz",
        "Pt",
        "bx",
        "by",
        "bz",
        "bdotx",
        "bdoty",
        "bdotz",
        "gamma",
        "q",
        "m",
        "char_time",
    ]
    for key in per_particle_keys:
        assert isinstance(state[key], np.ndarray), key
        assert state[key].shape == (pcount,), key

    assert state["count"] == pcount
    assert isinstance(state["rest_energy_mev"], float)
