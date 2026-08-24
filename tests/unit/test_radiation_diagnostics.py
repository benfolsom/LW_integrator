from __future__ import annotations

import numpy as np
import pytest

from core import diagnostics
from core import equations
from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.integration_runner import retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.species import get_species
from core.types import (
    ExternalFieldConfig,
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    SimulationType,
    StartupMode,
)


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


def _empty_driver_state() -> dict:
    return {
        key: np.array([], dtype=float)
        for key in (
            "x",
            "y",
            "z",
            "t",
            "Px",
            "Py",
            "Pz",
            "Pt",
            "gamma",
            "bx",
            "by",
            "bz",
            "bdotx",
            "bdoty",
            "bdotz",
            "q",
            "m",
            "char_time",
        )
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

    off_traj, _, off_soa, _, *_ = retarded_integrator(
        **common_kwargs, radiation_reaction_mode="off"
    )
    diagnostic_traj, _, diagnostic_soa, _, *_ = retarded_integrator(
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
        2.0 * ELEMENTARY_CHARGE**2 / (3.0 * C_MMNS) * gamma**6 * beta_dot_t[0] ** 2
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
        2.0 * ELEMENTARY_CHARGE**2 / (3.0 * C_MMNS) * gamma**4 * beta_dot_t[1] ** 2
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
    force = (2.0, 0.0, 0.0)
    beta = (0.0, 0.0, 0.9)
    gamma = 1.0 / np.sqrt(1.0 - beta[2] ** 2)
    beta_dot_t, _ = equations._derive_relativistic_kinematics_from_force(
        force,
        beta,
        gamma,
        ELECTRON_MASS_AMU,
    )
    medina_result = equations.compute_medina_radiation_reaction(
        external_force=force,
        external_force_time_derivative=(0.0, 0.0, 0.0),
        beta=beta,
        acceleration=tuple(C_MMNS * value for value in beta_dot_t),
        gamma=gamma,
        mass=ELECTRON_MASS_AMU,
        charge=ELEMENTARY_CHARGE,
        coordinate_dt=1.0e-6,
    )
    impulse, capped = equations._cap_medina_radiation_reaction_impulse(
        impulse=medina_result.radiation_reaction_impulse,
        external_force=force,
        coordinate_dt=1.0e-6,
        max_impulse_fraction=0.0,
    )

    assert capped is False
    assert impulse[0] == pytest.approx(0.0)
    assert impulse[1] == pytest.approx(0.0)
    assert impulse[2] < 0.0


def test_medina_impulse_cancels_ultrarelativistic_longitudinal_force() -> None:
    gamma = 1.0e6
    beta_z = np.sqrt(1.0 - 1.0 / gamma**2)
    force_z = 2.5e5
    force = (0.0, 0.0, force_z)
    beta = (0.0, 0.0, beta_z)
    beta_dot_t, _ = equations._derive_relativistic_kinematics_from_force(
        force,
        beta,
        gamma,
        ELECTRON_MASS_AMU,
    )
    medina_result = equations.compute_medina_radiation_reaction(
        external_force=force,
        external_force_time_derivative=(0.0, 0.0, 0.0),
        beta=beta,
        acceleration=tuple(C_MMNS * value for value in beta_dot_t),
        gamma=gamma,
        mass=ELECTRON_MASS_AMU,
        charge=ELEMENTARY_CHARGE,
        coordinate_dt=1.0e-3,
    )
    impulse, capped = equations._cap_medina_radiation_reaction_impulse(
        impulse=medina_result.radiation_reaction_impulse,
        external_force=force,
        coordinate_dt=1.0e-3,
        max_impulse_fraction=0.25,
    )

    assert capped is False
    assert impulse == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-30)


def test_medina_impulse_applies_numerical_cap() -> None:
    impulse, capped = equations._cap_medina_radiation_reaction_impulse(
        impulse=(0.0, 0.0, -1.0e12),
        external_force=(2.0e12, 0.0, 0.0),
        coordinate_dt=1.0e-6,
        max_impulse_fraction=0.25,
    )

    assert capped is True
    assert np.linalg.norm(impulse) == pytest.approx(0.25 * 2.0e6)


@pytest.mark.parametrize("gamma_from_energy", [0.9999999999999855, 1.0])
def test_production_medina_projects_near_rest_driver_before_kernel(
    monkeypatch: pytest.MonkeyPatch,
    gamma_from_energy: float,
) -> None:
    """A rounded near-rest canonical energy must not erase finite momentum."""

    proton = get_species("proton")
    beta_squared = 1.3036849076105039e-12
    beta_x = float(np.sqrt(beta_squared))
    input_gamma = float(1.0 / np.sqrt(1.0 - beta_squared))
    mechanical_px = input_gamma * proton.mass_amu * C_MMNS * beta_x
    driver = _make_integrator_state()
    driver["m"] = np.array([proton.mass_amu])
    driver["q"] = np.array([ELEMENTARY_CHARGE])
    driver["Px"] = np.array([mechanical_px])
    driver["Py"] = np.array([0.0])
    driver["Pz"] = np.array([0.0])
    driver["Pt"] = np.array([gamma_from_energy * proton.mass_amu * C_MMNS])
    driver["gamma"] = np.array([input_gamma])
    driver["bx"] = np.array([beta_x])
    driver["by"] = np.array([0.0])
    driver["bz"] = np.array([0.0])

    real_compute = equations.compute_medina_radiation_reaction
    recorded: dict[str, object] = {}

    def recording_compute(**kwargs: object):
        recorded.update(kwargs)
        return real_compute(**kwargs)

    monkeypatch.setattr(
        equations,
        "compute_medina_radiation_reaction",
        recording_compute,
    )

    h_step = 1.0e-6
    result = equations.retarded_equations_of_motion(
        h_step,
        [driver],
        [],
        0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.COLD_START,
        self_consistency=SelfConsistencyConfig(enabled=False),
        radiation_reaction_mode="medina_lad",
    )

    expected_gamma = float(
        np.hypot(proton.mass_amu * C_MMNS, mechanical_px) / (proton.mass_amu * C_MMNS)
    )
    expected_beta_x = mechanical_px / (expected_gamma * proton.mass_amu * C_MMNS)
    expected_coordinate_dt = h_step * expected_gamma

    assert float(recorded["gamma"]) == expected_gamma
    assert tuple(recorded["beta"]) == pytest.approx(
        (expected_beta_x, 0.0, 0.0), rel=0.0, abs=1.0e-21
    )
    assert float(recorded["coordinate_dt"]) == expected_coordinate_dt
    assert result["Px"][0] == mechanical_px
    assert result["Px"][0] != 0.0
    assert result["gamma"][0] == expected_gamma
    assert result["bx"][0] == pytest.approx(expected_beta_x, rel=0.0, abs=1.0e-21)
    assert result["t"][0] == expected_coordinate_dt
    assert result["Pt"][0] == pytest.approx(
        expected_gamma * proton.mass_amu * C_MMNS,
        rel=0.0,
        abs=np.spacing(proton.mass_amu * C_MMNS),
    )
    assert result["mass_shell_projection_energy"][0] == pytest.approx(
        C_MMNS * proton.mass_amu * C_MMNS * (expected_gamma - gamma_from_energy),
        rel=2.0e-3,
    )
    assert result["medina_external_force_sample_time"][0] == pytest.approx(
        0.5 * expected_coordinate_dt
    )
    assert not bool(result["medina_force_derivative_ready"][0])
    assert not bool(result["medina_impulse_capped"][0])
    assert result["radiation_reaction_work"][0] == 0.0


def test_nonexact_medina_rescales_spatial_momentum_before_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regular non-exact Medina path retains its temporal-energy shell."""

    proton = get_species("proton")
    spatial_gamma = 1.5
    energy_gamma = 2.0
    mass_momentum = proton.mass_amu * C_MMNS
    initial_px = mass_momentum * np.sqrt((spatial_gamma - 1.0) * (spatial_gamma + 1.0))
    driver = _make_integrator_state()
    driver["m"] = np.array([proton.mass_amu])
    driver["q"] = np.array([ELEMENTARY_CHARGE])
    driver["Px"] = np.array([initial_px])
    driver["Py"] = np.array([0.0])
    driver["Pz"] = np.array([0.0])
    driver["Pt"] = np.array([energy_gamma * mass_momentum])
    driver["gamma"] = np.array([spatial_gamma])
    driver["bx"] = np.array([initial_px / (spatial_gamma * mass_momentum)])
    driver["by"] = np.array([0.0])
    driver["bz"] = np.array([0.0])

    real_compute = equations.compute_medina_radiation_reaction
    recorded: dict[str, object] = {}

    def recording_compute(**kwargs: object):
        recorded.update(kwargs)
        return real_compute(**kwargs)

    monkeypatch.setattr(
        equations,
        "compute_medina_radiation_reaction",
        recording_compute,
    )

    h_step = 1.0e-6
    result = equations.retarded_equations_of_motion(
        h_step,
        [driver],
        [],
        0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.COLD_START,
        self_consistency=SelfConsistencyConfig(enabled=False),
        radiation_reaction_mode="medina_lad",
    )

    expected_px = mass_momentum * np.sqrt((energy_gamma - 1.0) * (energy_gamma + 1.0))
    expected_beta_x = expected_px / (energy_gamma * mass_momentum)
    assert float(recorded["gamma"]) == energy_gamma
    assert tuple(recorded["beta"]) == pytest.approx(
        (expected_beta_x, 0.0, 0.0), rel=0.0, abs=2.0e-16
    )
    assert float(recorded["coordinate_dt"]) == h_step * energy_gamma
    assert result["Px"][0] == pytest.approx(expected_px, rel=2.0e-16)
    assert result["Pt"][0] == energy_gamma * mass_momentum
    assert result["gamma"][0] == energy_gamma
    assert result["mass_shell_projection_energy"][0] == 0.0
    assert result["radiation_reaction_work"][0] == 0.0


def test_nonexact_medina_uses_spatial_fallback_when_energy_has_no_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A moving energy shell cannot invent a direction from exactly zero p."""

    proton = get_species("proton")
    energy_gamma = 1.1
    mass_momentum = proton.mass_amu * C_MMNS
    driver = _make_integrator_state()
    driver["m"] = np.array([proton.mass_amu])
    driver["q"] = np.array([ELEMENTARY_CHARGE])
    driver["Px"] = np.array([0.0])
    driver["Py"] = np.array([0.0])
    driver["Pz"] = np.array([0.0])
    driver["Pt"] = np.array([energy_gamma * mass_momentum])
    driver["gamma"] = np.array([energy_gamma])
    driver["bx"] = np.array([0.0])
    driver["by"] = np.array([0.0])
    driver["bz"] = np.array([0.0])

    real_compute = equations.compute_medina_radiation_reaction
    recorded: dict[str, object] = {}

    def recording_compute(**kwargs: object):
        recorded.update(kwargs)
        return real_compute(**kwargs)

    monkeypatch.setattr(
        equations,
        "compute_medina_radiation_reaction",
        recording_compute,
    )

    result = equations.retarded_equations_of_motion(
        1.0e-6,
        [driver],
        [_empty_driver_state()],
        0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.COLD_START,
        self_consistency=SelfConsistencyConfig(enabled=False),
        radiation_reaction_mode="medina_lad",
    )

    assert float(recorded["gamma"]) == 1.0
    assert tuple(recorded["beta"]) == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(result["Px"], 0.0)
    np.testing.assert_array_equal(result["Py"], 0.0)
    np.testing.assert_array_equal(result["Pz"], 0.0)
    assert result["Pt"][0] == mass_momentum
    assert result["gamma"][0] == 1.0
    assert result["mass_shell_projection_energy"][0] == pytest.approx(
        (1.0 - energy_gamma) * proton.mass_amu * C_MMNS**2
    )
    assert result["radiation_reaction_work"][0] == 0.0


def test_exact_rfs_off_and_medina_share_near_rest_mass_shell_boundary() -> None:
    proton = get_species("proton")
    beta_squared = 1.3036849076105039e-12
    beta_x = float(np.sqrt(beta_squared))
    input_gamma = float(1.0 / np.sqrt(1.0 - beta_squared))
    mechanical_px = input_gamma * proton.mass_amu * C_MMNS * beta_x

    driver = _make_integrator_state()
    driver["m"] = np.array([proton.mass_amu])
    driver["q"] = np.array([ELEMENTARY_CHARGE])
    driver["Px"] = np.array([mechanical_px])
    driver["Py"] = np.array([0.0])
    driver["Pz"] = np.array([0.0])
    driver["Pt"] = np.array([0.9999999999999855 * proton.mass_amu * C_MMNS])
    driver["gamma"] = np.array([input_gamma])
    driver["bx"] = np.array([beta_x])
    driver["by"] = np.array([0.0])
    driver["bz"] = np.array([0.0])

    common = dict(
        h=1.0e-6,
        trajectory=[driver],
        trajectory_ext=[],
        index_traj=0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.INERTIAL_PREHISTORY,
        self_consistency=SelfConsistencyConfig(enabled=False),
        magnetic_dipole=MagneticDipoleConfig(enabled=True),
    )
    off_result = equations.retarded_equations_of_motion(
        **common,
        radiation_reaction_mode="off",
    )
    medina_result = equations.retarded_equations_of_motion(
        **common,
        radiation_reaction_mode="medina_lad",
    )

    for key in (
        "x",
        "y",
        "z",
        "t",
        "Px",
        "Py",
        "Pz",
        "Pt",
        "gamma",
        "bx",
        "by",
        "bz",
        "mass_shell_projection_energy",
    ):
        np.testing.assert_array_equal(off_result[key], medina_result[key])
    assert off_result["Px"][0] == mechanical_px
    assert off_result["Px"][0] != 0.0
    assert off_result["gamma"][0] > 1.0
    expected_gamma = float(
        np.hypot(proton.mass_amu * C_MMNS, mechanical_px) / (proton.mass_amu * C_MMNS)
    )
    assert off_result["gamma"][0] == expected_gamma
    assert not bool(medina_result["medina_force_derivative_ready"][0])
    assert medina_result["radiation_reaction_work"][0] == 0.0


def test_production_medina_endpoint_is_first_order_under_step_halving() -> None:
    field = ExternalFieldConfig(magnetic_field_native=(0.0, 1.0e7, 0.0))
    proper_duration_ns = 6.4e-5

    def run(intervals: int):
        positron = get_species("positron")
        rider = _make_integrator_state()
        gamma = float(rider["gamma"][0])
        beta_z = float(rider["bz"][0])
        rider["m"] = np.array([positron.mass_amu])
        rider["q"] = np.array([ELEMENTARY_CHARGE])
        rider["Pz"] = np.array([gamma * positron.mass_amu * C_MMNS * beta_z])
        rider["Pt"] = np.array([gamma * positron.mass_amu * C_MMNS])
        trajectory, _, soa, _, *_ = retarded_integrator(
            steps=intervals + 1,
            h_step=proper_duration_ns / intervals,
            wall_z=0.0,
            aperture_radius=1.0e9,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=rider,
            init_driver=_empty_driver_state(),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=1.0e9,
            startup_mode=StartupMode.INERTIAL_PREHISTORY,
            use_numba=False,
            external_field=field,
            radiation_reaction_mode="medina_lad",
            magnetic_dipole=MagneticDipoleConfig(
                enabled=True,
                rider=MagneticDipoleParticleConfig(species="positron"),
            ),
        )
        assert soa is not None
        np.testing.assert_array_equal(soa.medina_force_derivative_ready[:2, 0], False)
        np.testing.assert_array_equal(soa.medina_force_derivative_ready[2:, 0], True)
        assert not np.any(soa.medina_impulse_capped[:, 0])
        total_work = float(np.sum(soa.radiation_reaction_work[:, 0]))
        far_energy = np.asarray(soa.radiation_energy[:, 0], dtype=float)
        assert np.all(np.isfinite(far_energy))
        assert np.all(far_energy >= 0.0)
        assert float(np.sum(far_energy)) > 0.0
        assert total_work < 0.0
        final = trajectory[-1]
        return (
            np.asarray([final[key][0] for key in ("Px", "Py", "Pz")]),
            float(final["gamma"][0]),
            total_work,
            float(np.sum(soa.mass_shell_projection_energy[:, 0])),
        )

    coarse, medium, fine = (run(intervals) for intervals in (16, 32, 64))
    momentum_differences = (
        float(np.linalg.norm(coarse[0] - medium[0])),
        float(np.linalg.norm(medium[0] - fine[0])),
    )
    gamma_differences = (
        abs(coarse[1] - medium[1]),
        abs(medium[1] - fine[1]),
    )
    work_differences = (
        abs(coarse[2] - medium[2]),
        abs(medium[2] - fine[2]),
    )
    for coarse_difference, fine_difference in (
        momentum_differences,
        gamma_differences,
        work_differences,
    ):
        assert coarse_difference > 0.0
        assert fine_difference > 0.0
        assert 0.4 < fine_difference / coarse_difference < 0.6
    assert all(np.isfinite(run[3]) and run[3] > 0.0 for run in (coarse, medium, fine))
    assert 0.45 < medium[3] / coarse[3] < 0.55
    assert 0.45 < fine[3] / medium[3] < 0.55


def test_production_medina_primes_before_applying_complete_derivative() -> None:
    field = ExternalFieldConfig(magnetic_field_native=(0.0, 1.0e7, 0.0))
    common_kwargs = dict(
        steps=5,
        h_step=1.0e-6,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_integrator_state(),
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        use_numba=False,
        external_field=field,
    )

    off_trajectory, _, _, _, *_ = retarded_integrator(
        **common_kwargs,
        radiation_reaction_mode="off",
    )
    medina_trajectory, _, medina_soa, _, *_ = retarded_integrator(
        **common_kwargs,
        radiation_reaction_mode="medina_lad",
    )

    assert medina_soa is not None
    np.testing.assert_array_equal(
        medina_soa.medina_force_derivative_ready[:, 0],
        (False, False, True, True, True),
    )
    assert np.isnan(medina_soa.medina_external_force_sample_time[0, 0])
    assert np.all(np.isfinite(medina_soa.medina_external_force_sample_time[1:, 0]))
    # Row 1 is the unprimed first physical step: far radiation is diagnosed,
    # but no incomplete dF/dt=0 impulse is applied.  The ordinary non-exact
    # Medina boundary may rescale spatial momentum onto the temporal-energy
    # shell before sampling; that constraint operation is not RR work.
    assert medina_soa.radiation_energy[1, 0] > 0.0
    assert medina_soa.radiation_reaction_work[1, 0] == pytest.approx(0.0)
    assert medina_soa.radiation_energy_applied[1, 0] == 0.0
    assert medina_soa.mass_shell_projection_energy[1, 0] == 0.0
    medina_unprimed = medina_trajectory[1]
    off_unprimed = off_trajectory[1]
    np.testing.assert_array_equal(medina_unprimed["gamma"], off_unprimed["gamma"])
    np.testing.assert_array_equal(medina_unprimed["Pt"], off_unprimed["Pt"])
    mass = float(medina_unprimed["m"][0])
    gamma = float(medina_unprimed["gamma"][0])
    momentum = np.asarray(
        [medina_unprimed[key][0] for key in ("Px", "Py", "Pz")], dtype=float
    )
    expected_magnitude = mass * C_MMNS * np.sqrt((gamma - 1.0) * (gamma + 1.0))
    assert float(np.linalg.norm(momentum)) == pytest.approx(
        expected_magnitude,
        rel=2.0e-15,
    )

    assert np.all(medina_soa.radiation_reaction_work[2:, 0] < 0.0)
    assert not np.any(medina_soa.medina_impulse_capped[:, 0])
    balance = (
        medina_soa.radiation_reaction_work[2:, 0]
        + medina_soa.radiation_energy[2:, 0]
        + medina_soa.medina_cross_field_energy_change[2:, 0]
    )
    np.testing.assert_allclose(
        balance,
        0.0,
        rtol=0.0,
        atol=2.0e-6 * float(np.max(medina_soa.radiation_energy[2:, 0])),
    )


def test_production_medina_is_noop_for_neutral_particle() -> None:
    neutral_state = _make_integrator_state()
    neutral_state["q"] = np.array([0.0])
    _, _, medina_soa, _, *_ = retarded_integrator(
        steps=4,
        h_step=1.0e-6,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=neutral_state,
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        use_numba=False,
        external_field=ExternalFieldConfig(electric_field_native=(1.0e8, 0.0, 0.0)),
        radiation_reaction_mode="medina_lad",
    )

    assert medina_soa is not None
    assert not np.any(medina_soa.medina_force_derivative_ready)
    assert not np.any(medina_soa.medina_impulse_capped)
    assert np.all(np.isnan(medina_soa.medina_external_force_sample_time))
    np.testing.assert_array_equal(medina_soa.radiation_reaction_work, 0.0)
    np.testing.assert_array_equal(medina_soa.radiation_power, 0.0)
    np.testing.assert_array_equal(medina_soa.radiation_energy_applied, 0.0)


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
