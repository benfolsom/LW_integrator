from __future__ import annotations

import numpy as np
import pytest

from core import diagnostics
from core import equations
from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.integration_runner import retarded_integrator
from core.types import SimulationType, StartupMode


def _make_integrator_state() -> dict:
    gamma = 2.0
    beta_z = np.sqrt(1.0 - 1.0 / gamma**2)
    pz = gamma * ELECTRON_MASS_AMU * C_MMNS * beta_z
    return {
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "t": np.array([0.0]),
        "Px": np.array([0.0]),
        "Py": np.array([0.0]),
        "Pz": np.array([pz]),
        "Pt": np.array([gamma * ELECTRON_MASS_AMU * C_MMNS]),
        "gamma": np.array([gamma]),
        "bx": np.array([0.0]),
        "by": np.array([0.0]),
        "bz": np.array([beta_z]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "radiation_power": np.array([0.0]),
        "radiation_energy": np.array([0.0]),
        "radiation_energy_applied": np.array([0.0]),
        "q": np.array([ELEMENTARY_CHARGE]),
        "m": np.array([ELECTRON_MASS_AMU]),
        "char_time": np.array([1.0e-3]),
    }


def test_radiation_mode_defaults_to_off_and_rejects_legacy_bdot() -> None:
    assert equations._canonicalize_radiation_reaction_mode(None) == "off"
    assert equations._canonicalize_radiation_reaction_mode("none") == "off"
    assert (
        equations._canonicalize_radiation_reaction_mode("power-damping")
        == "power_matched_damping"
    )

    with pytest.raises(ValueError):
        equations._canonicalize_radiation_reaction_mode("legacy_bdot")


def test_diagnostic_only_matches_off_trajectory_state() -> None:
    common_kwargs = dict(
        steps=3,
        h_step=0.001,
        wall_z=100.0,
        aperture_radius=10.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_integrator_state(),
        init_driver=None,
        mean=0.0,
        cav_spacing=10.0,
        z_cutoff=50.0,
        startup_mode=StartupMode.COLD_START,
        use_numba=False,
    )

    off_traj, _, off_soa, _ = retarded_integrator(
        **common_kwargs, radiation_reaction_mode="off"
    )
    diagnostic_traj, _, diagnostic_soa, _ = retarded_integrator(
        **common_kwargs, radiation_reaction_mode="diagnostic_only"
    )

    keys = (
        "x",
        "y",
        "z",
        "Px",
        "Py",
        "Pz",
        "Pt",
        "gamma",
        "bx",
        "by",
        "bz",
        "radiation_power",
        "radiation_energy",
        "radiation_energy_applied",
    )
    for off_state, diagnostic_state in zip(off_traj, diagnostic_traj):
        for key in keys:
            np.testing.assert_allclose(off_state[key], diagnostic_state[key])

    assert off_soa is not None
    assert diagnostic_soa is not None
    np.testing.assert_allclose(off_soa.radiation_power, diagnostic_soa.radiation_power)
    np.testing.assert_allclose(
        off_soa.radiation_energy, diagnostic_soa.radiation_energy
    )
    np.testing.assert_allclose(
        off_soa.radiation_energy_applied,
        diagnostic_soa.radiation_energy_applied,
    )


def test_lienard_power_zero_for_unaccelerated_motion() -> None:
    power = equations._compute_lienard_radiated_power(
        ELEMENTARY_CHARGE,
        beta=(0.5, 0.0, 0.0),
        beta_dot_t=(0.0, 0.0, 0.0),
        gamma=1.2,
    )

    assert power == pytest.approx(0.0)


def test_lienard_power_uses_coordinate_time_beta_dot() -> None:
    gamma = 2.0
    beta = (0.5, 0.0, 0.0)
    beta_dot_t = (0.0, 3.0, 0.0)

    power = equations._compute_lienard_radiated_power(
        ELEMENTARY_CHARGE,
        beta=beta,
        beta_dot_t=beta_dot_t,
        gamma=gamma,
    )

    expected_transverse_term = 3.0**2 - (0.5 * 3.0) ** 2
    expected = (
        2.0
        * ELEMENTARY_CHARGE**2
        / (3.0 * C_MMNS)
        * gamma**6
        * expected_transverse_term
    )
    assert power == pytest.approx(expected)


def test_power_matched_damping_removes_energy_from_mechanical_momentum() -> None:
    gamma = 2.0
    mass = ELECTRON_MASS_AMU
    momentum_mag = mass * C_MMNS * (gamma**2 - 1.0) ** 0.5
    radiated_energy = 0.1 * mass * C_MMNS**2

    (new_px, new_py, new_pz), new_gamma, applied = (
        equations._apply_power_matched_radiation_damping(
            (momentum_mag, 0.0, 0.0),
            mass,
            gamma,
            radiated_energy,
        )
    )

    assert applied == pytest.approx(radiated_energy)
    assert new_gamma == pytest.approx(gamma - 0.1)
    assert new_py == pytest.approx(0.0)
    assert new_pz == pytest.approx(0.0)
    assert new_px < momentum_mag


def test_power_matched_damping_does_not_cross_rest_energy() -> None:
    gamma = 1.01
    mass = ELECTRON_MASS_AMU
    momentum_mag = mass * C_MMNS * (gamma**2 - 1.0) ** 0.5
    requested_energy = 10.0 * mass * C_MMNS**2

    (_, _, _), new_gamma, applied = equations._apply_power_matched_radiation_damping(
        (momentum_mag, 0.0, 0.0),
        mass,
        gamma,
        requested_energy,
    )

    assert new_gamma == pytest.approx(1.0)
    assert applied == pytest.approx((gamma - 1.0) * mass * C_MMNS**2)


def test_large_bdot_change_diagnostic_has_legacy_alias() -> None:
    trajectory = [
        {"bdotz": np.array([0.0, 0.0])},
        {"bdotz": np.array([2.0e-3, 5.0e-4])},
    ]

    expected = [(1, 0)]
    assert diagnostics.find_large_bdot_changes(trajectory) == expected
    assert diagnostics.find_radiation_reaction_activations(trajectory) == expected
