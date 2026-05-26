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
    assert equations._canonicalize_radiation_reaction_mode("medina") == "medina_lad"

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


def test_lienard_power_parallel_acceleration_uses_gamma6_scaling() -> None:
    gamma = 2.5
    beta_dot_t = (4.0, 0.0, 0.0)

    power = equations._compute_lienard_radiated_power(
        ELEMENTARY_CHARGE,
        beta=(0.6, 0.0, 0.0),
        beta_dot_t=beta_dot_t,
        gamma=gamma,
    )

    expected = (
        2.0
        * ELEMENTARY_CHARGE**2
        / (3.0 * C_MMNS)
        * gamma**6
        * beta_dot_t[0] ** 2
    )
    assert power == pytest.approx(expected)


def test_lienard_power_transverse_acceleration_uses_synchrotron_scaling() -> None:
    gamma = 4.0
    beta_x = np.sqrt(1.0 - 1.0 / gamma**2)
    beta_dot_t = (0.0, 5.0, 0.0)

    power = equations._compute_lienard_radiated_power(
        ELEMENTARY_CHARGE,
        beta=(beta_x, 0.0, 0.0),
        beta_dot_t=beta_dot_t,
        gamma=gamma,
    )

    expected = (
        2.0
        * ELEMENTARY_CHARGE**2
        / (3.0 * C_MMNS)
        * gamma**4
        * beta_dot_t[1] ** 2
    )
    assert power == pytest.approx(expected)


def test_lienard_power_converts_stored_bdot_to_coordinate_time() -> None:
    gamma = 3.0
    beta = (0.0, 0.0, np.sqrt(1.0 - 1.0 / gamma**2))
    beta_dot_t = (2.0, 0.0, 0.0)
    stored_bdot = tuple(component / C_MMNS for component in beta_dot_t)

    direct = equations._compute_lienard_radiated_power(
        ELEMENTARY_CHARGE,
        beta=beta,
        beta_dot_t=beta_dot_t,
        gamma=gamma,
    )
    converted = equations._compute_lienard_radiated_power(
        ELEMENTARY_CHARGE,
        beta=beta,
        beta_dot_t=tuple(component * C_MMNS for component in stored_bdot),
        gamma=gamma,
    )

    assert converted == pytest.approx(direct)


def test_integrated_lienard_power_converges_for_analytic_circular_motion() -> None:
    gamma = 8.0
    beta_mag = np.sqrt(1.0 - 1.0 / gamma**2)
    omega = 3.0
    duration = 0.2
    coefficient = 2.0 * ELEMENTARY_CHARGE**2 / (3.0 * C_MMNS)
    expected_power = coefficient * gamma**4 * (beta_mag * omega) ** 2
    expected_energy = expected_power * duration

    def integrated_energy(steps: int) -> float:
        dt = duration / steps
        total = 0.0
        for step in range(steps):
            time = (step + 0.5) * dt
            beta = (
                beta_mag * np.cos(omega * time),
                beta_mag * np.sin(omega * time),
                0.0,
            )
            beta_dot_t = (
                -beta_mag * omega * np.sin(omega * time),
                beta_mag * omega * np.cos(omega * time),
                0.0,
            )
            total += (
                equations._compute_lienard_radiated_power(
                    ELEMENTARY_CHARGE,
                    beta=beta,
                    beta_dot_t=beta_dot_t,
                    gamma=gamma,
                )
                * dt
            )
        return total

    assert integrated_energy(8) == pytest.approx(expected_energy, rel=1.0e-12)
    assert integrated_energy(64) == pytest.approx(expected_energy, rel=1.0e-12)


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


def test_medina_impulse_damps_velocity_for_transverse_force() -> None:
    impulse, capped = equations._compute_medina_radiation_reaction_impulse(
        external_force=(2.0, 0.0, 0.0),
        beta=(0.0, 0.0, 0.9),
        beta_dot_t=(3.0, 0.0, 0.0),
        gamma=4.0,
        dgamma_dt=0.0,
        mass=ELECTRON_MASS_AMU,
        charge=ELEMENTARY_CHARGE,
        coordinate_dt=1.0e-6,
        max_impulse_fraction=0.0,
    )

    assert capped is False
    assert impulse[0] == pytest.approx(0.0)
    assert impulse[1] == pytest.approx(0.0)
    assert impulse[2] < 0.0


def test_medina_impulse_cancels_ultrarelativistic_longitudinal_force() -> None:
    gamma = 1.0e12
    beta_z = np.sqrt(1.0 - 1.0 / gamma**2)
    force_z = 2.5e5

    impulse, capped = equations._compute_medina_radiation_reaction_impulse(
        external_force=(0.0, 0.0, force_z),
        beta=(0.0, 0.0, beta_z),
        beta_dot_t=(0.0, 0.0, force_z / (gamma**3 * ELECTRON_MASS_AMU * C_MMNS)),
        gamma=gamma,
        dgamma_dt=beta_z * force_z / (ELECTRON_MASS_AMU * C_MMNS),
        mass=ELECTRON_MASS_AMU,
        charge=ELEMENTARY_CHARGE,
        coordinate_dt=1.0e-3,
        max_impulse_fraction=0.25,
    )

    assert capped is False
    assert impulse == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-30)


def test_medina_impulse_applies_numerical_cap() -> None:
    impulse, capped = equations._compute_medina_radiation_reaction_impulse(
        external_force=(2.0e12, 0.0, 0.0),
        beta=(0.0, 0.0, 0.9),
        beta_dot_t=(0.0, 0.0, 0.0),
        gamma=4.0,
        dgamma_dt=0.0,
        mass=ELECTRON_MASS_AMU,
        charge=ELEMENTARY_CHARGE,
        coordinate_dt=1.0e-6,
        max_impulse_fraction=0.25,
    )

    assert capped is True
    assert np.linalg.norm(impulse) == pytest.approx(0.25 * 2.0e6)


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
