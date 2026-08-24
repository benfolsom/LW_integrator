"""Integration-seam tests for the coupled RFS magnetic-dipole model."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

import core.equations as equations
import core.rfs as rfs
from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.external_fields import (
    ExternalFieldConfig,
    NATIVE_FORCE_UNIT_NEWTON,
    electric_field_v_per_m_to_native,
)
from core.integration_runner import retarded_integrator
from core.magnetic_dipole import HBAR_NATIVE, boost_rest_polarization
from core.medina_radiation_reaction import MedinaRadiationReactionResult
from core.rfs import electromagnetic_field_tensor_native, minkowski_dot
from core.self_consistency import SelfConsistencyConfig
from core.species import get_species
from core.types import (
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    SimulationType,
    StartupMode,
)


def _particle_state(
    species_name: str,
    *,
    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    beta: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    species = get_species(species_name)
    beta_vector = np.asarray(beta, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - float(beta_vector @ beta_vector))
    momentum = gamma * species.mass_amu * C_MMNS * beta_vector
    charge = float(species.charge_e) * ELEMENTARY_CHARGE
    return {
        "x": np.array([position_mm[0]], dtype=float),
        "y": np.array([position_mm[1]], dtype=float),
        "z": np.array([position_mm[2]], dtype=float),
        "t": np.array([0.0]),
        "Px": np.array([momentum[0]]),
        "Py": np.array([momentum[1]]),
        "Pz": np.array([momentum[2]]),
        "Pt": np.array([gamma * species.mass_amu * C_MMNS]),
        "gamma": np.array([gamma]),
        "bx": np.array([beta_vector[0]]),
        "by": np.array([beta_vector[1]]),
        "bz": np.array([beta_vector[2]]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        # q remains the legacy single-particle value. The explicit fields are
        # included so this test exercises source/observer charge separation.
        "q": np.array([charge]),
        "q_species": np.array([charge]),
        "q_observer": np.array([charge]),
        "q_source": np.array([charge]),
        "m": np.array([species.mass_amu]),
        "m_species": np.array([species.mass_amu]),
        "char_time": np.array([0.0]),
    }


def _empty_state() -> dict[str, np.ndarray]:
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
            "q_species",
            "q_observer",
            "q_source",
            "m",
            "m_species",
            "char_time",
        )
    }


def _rfs_config(
    *, rider_polarization: float = 1.0, rider_species: str = "neutron"
) -> MagneticDipoleConfig:
    return MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=True,
        stern_gerlach_force_enabled=True,
        rider=MagneticDipoleParticleConfig(
            species=rider_species,
            rest_spin=(0.0, 0.0, 1.0),
            polarization=rider_polarization,
        ),
        driver=MagneticDipoleParticleConfig(
            species="proton",
            rest_spin=(0.0, 0.0, 1.0),
            polarization=0.0,
        ),
    )


def _run(
    *,
    rider: dict[str, np.ndarray],
    driver: dict[str, np.ndarray] | None,
    magnetic_dipole: MagneticDipoleConfig,
    steps: int = 2,
    h_step: float = 1.0e-3,
    startup_mode: StartupMode = StartupMode.COLD_START,
    radiation_reaction_mode: str = "off",
    external_field: ExternalFieldConfig | None = None,
) -> Any:
    return retarded_integrator(
        steps=steps,
        h_step=h_step,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=rider,
        init_driver=driver,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=startup_mode,
        radiation_reaction_mode=radiation_reaction_mode,
        external_field=external_field,
        magnetic_dipole=magnetic_dipole,
        self_consistency=SelfConsistencyConfig(enabled=False),
        use_numba=False,
    )


def test_neutral_neutron_gets_signed_rfs_impulse_in_vacuum_axial_gradient() -> None:
    neutron = get_species("neutron")
    assert neutron.magnetic_moment_j_t is not None
    gradient_t_m = 1.0e12
    h_step_ns = 1.0e-3
    field = ExternalFieldConfig(
        magnetic_field_gradient_t_per_m=(
            (-0.5 * gradient_t_m, 0.0, 0.0),
            (0.0, -0.5 * gradient_t_m, 0.0),
            (0.0, 0.0, gradient_t_m),
        )
    )

    trajectory, _, soa, _, *_ = _run(
        rider=_particle_state("neutron"),
        driver=_empty_state(),
        magnetic_dipole=_rfs_config(),
        h_step=h_step_ns,
        external_field=field,
    )

    assert soa is not None
    expected_force_newton = neutron.magnetic_moment_j_t * gradient_t_m
    expected_pz = expected_force_newton / NATIVE_FORCE_UNIT_NEWTON * h_step_ns
    assert float(soa.Pz[-1, 0]) == pytest.approx(expected_pz, rel=2.0e-15)
    assert float(soa.Pz[-1, 0]) < 0.0
    np.testing.assert_array_equal(soa.Px, 0.0)
    np.testing.assert_array_equal(soa.Py, 0.0)
    np.testing.assert_array_equal(soa.spin_x, 0.0)
    np.testing.assert_array_equal(soa.spin_y, 0.0)
    np.testing.assert_array_equal(soa.spin_z, 1.0)

    final_state = trajectory[-1]
    beta = np.array([final_state[axis][0] for axis in ("bx", "by", "bz")], dtype=float)
    gamma = float(final_state["gamma"][0])
    spin_four = boost_rest_polarization((0.0, 0.0, 1.0), beta)
    four_velocity = C_MMNS * np.concatenate(([gamma], gamma * beta))
    spin_magnitude = neutron.spin_quantum_number * HBAR_NATIVE

    # The stored rest spin has no spatial precession at B=0, but its boosted
    # time component must become nonzero once the gradient accelerates it.
    assert spin_four[0] < 0.0
    assert minkowski_dot(four_velocity, spin_four) / C_MMNS == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert spin_magnitude > 0.0
    assert minkowski_dot(spin_four, spin_four) == pytest.approx(-1.0, rel=2.0e-15)


def test_cold_start_moving_charge_drives_neutral_rfs_response() -> None:
    rider = _particle_state("neutron", position_mm=(0.0, 0.1, 0.0))
    driver = _particle_state("proton", beta=(0.1, 0.0, 0.0))

    _, _, soa, _, *_ = _run(
        rider=rider,
        driver=driver,
        magnetic_dipole=_rfs_config(),
        steps=24,
        h_step=1.0e-4,
    )

    assert soa is not None
    np.testing.assert_array_equal(soa.q_observer, 0.0)
    assert float(soa.local_magnetic_field_z_t[0, 0]) == 0.0
    assert np.max(soa.local_magnetic_field_z_t[:, 0]) > 0.0
    # For +Q moving along +x, a neutron at +y with spin +z and signed mu_n<0
    # is pushed toward +y by the full-G response once retarded history exists.
    assert float(soa.Py[-1, 0]) > 0.0
    assert np.count_nonzero(soa.Py[:, 0]) > 0


def test_rfs_off_is_an_exact_neutral_baseline() -> None:
    gradient_t_m = 1.0e12
    gradient_field = ExternalFieldConfig(
        magnetic_field_gradient_t_per_m=(
            (-0.5 * gradient_t_m, 0.0, 0.0),
            (0.0, -0.5 * gradient_t_m, 0.0),
            (0.0, 0.0, gradient_t_m),
        )
    )
    disabled = MagneticDipoleConfig(enabled=False)
    gradient_trajectory, _, _, _, *_ = _run(
        rider=_particle_state("neutron"),
        driver=_empty_state(),
        magnetic_dipole=disabled,
        external_field=gradient_field,
    )
    zero_trajectory, _, _, _, *_ = _run(
        rider=_particle_state("neutron"),
        driver=_empty_state(),
        magnetic_dipole=disabled,
        external_field=ExternalFieldConfig(),
    )

    for gradient_state, zero_state in zip(
        gradient_trajectory, zero_trajectory, strict=True
    ):
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
        ):
            np.testing.assert_array_equal(gradient_state[key], zero_state[key])


def test_rfs_rejects_non_medina_dynamic_radiation_reaction() -> None:
    with pytest.raises(
        NotImplementedError, match="only the explicitly named.*medina_lad"
    ):
        _run(
            rider=_particle_state("neutron", position_mm=(0.0, 0.1, 0.0)),
            driver=_particle_state("proton", beta=(0.1, 0.0, 0.0)),
            magnetic_dipole=_rfs_config(),
            radiation_reaction_mode="power_matched_damping",
        )


@pytest.mark.parametrize("mode", ("medina", "medina_rr", "lad_medina", "lad-medina"))
def test_rfs_accepts_canonical_medina_aliases(mode: str) -> None:
    _run(
        rider=_particle_state("neutron"),
        driver=_empty_state(),
        magnetic_dipole=_rfs_config(),
        radiation_reaction_mode=mode,
    )


def test_rfs_spin_midpoint_uses_rr_force_at_both_covariant_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_terms = rfs.rfs_charge_radiation_reaction_terms_native
    recorded_calls: list[dict[str, np.ndarray]] = []

    def recording_terms(**kwargs: Any) -> rfs.RFSRadiationReactionTerms:
        recorded_calls.append(
            {
                "four_velocity_mm_ns": np.asarray(
                    kwargs["four_velocity_mm_ns"], dtype=float
                ).copy(),
                "spin_four_vector": np.asarray(
                    kwargs["spin_four_vector"], dtype=float
                ).copy(),
                "applied_radiation_reaction_force_native": np.asarray(
                    kwargs["applied_radiation_reaction_force_native"], dtype=float
                ).copy(),
            }
        )
        return real_terms(**kwargs)

    monkeypatch.setattr(
        rfs,
        "rfs_charge_radiation_reaction_terms_native",
        recording_terms,
    )
    rest_spin = np.asarray((0.3, -0.4, 0.5), dtype=float)
    rest_spin /= np.linalg.norm(rest_spin)
    beta_start = np.asarray((0.12, -0.04, 0.03), dtype=float)
    beta_end = np.asarray((0.14, -0.01, 0.02), dtype=float)
    applied_force = np.asarray((0.8, -0.3, 0.5), dtype=float)

    rest_end = equations._advance_rfs_rest_spin(
        rest_spin,
        beta_start=beta_start,
        beta_end=beta_end,
        field_tensor=np.zeros((4, 4), dtype=float),
        partial_f=np.zeros((4, 4, 4), dtype=float),
        charge_native=0.0,
        mass_amu=1.7,
        magnetic_moment_native=0.0,
        spin_quantum_number=0.5,
        proper_time_step_ns=2.0e-3,
        applied_radiation_reaction_force_native=applied_force,
    )

    assert len(recorded_calls) == 2
    midpoint_beta = 0.5 * (beta_start + beta_end)
    np.testing.assert_allclose(
        recorded_calls[0]["four_velocity_mm_ns"],
        equations._four_velocity_native(beta_start),
    )
    np.testing.assert_allclose(
        recorded_calls[1]["four_velocity_mm_ns"],
        equations._four_velocity_native(midpoint_beta),
    )
    np.testing.assert_array_equal(
        recorded_calls[0]["applied_radiation_reaction_force_native"], applied_force
    )
    np.testing.assert_array_equal(
        recorded_calls[1]["applied_radiation_reaction_force_native"], applied_force
    )
    for call in recorded_calls:
        assert minkowski_dot(
            call["four_velocity_mm_ns"], call["spin_four_vector"]
        ) == pytest.approx(0.0, abs=2.0e-14)
        assert minkowski_dot(
            call["spin_four_vector"], call["spin_four_vector"]
        ) == pytest.approx(-1.0, rel=2.0e-15)

    spin_end = boost_rest_polarization(rest_end, beta_end)
    u_end = equations._four_velocity_native(beta_end)
    assert minkowski_dot(u_end, spin_end) == pytest.approx(0.0, abs=2.0e-14)
    assert minkowski_dot(spin_end, spin_end) == pytest.approx(-1.0, rel=2.0e-15)


def test_zero_rr_force_is_exactly_the_legacy_rfs_spin_step() -> None:
    rest_spin = np.asarray((0.2, 0.7, -0.3), dtype=float)
    rest_spin /= np.linalg.norm(rest_spin)
    common: dict[str, Any] = {
        "beta_start": np.asarray((0.07, -0.02, 0.03), dtype=float),
        "beta_end": np.asarray((0.08, -0.015, 0.025), dtype=float),
        "field_tensor": electromagnetic_field_tensor_native(
            (0.2, -0.1, 0.3), (0.4, 0.1, -0.2)
        ),
        "partial_f": np.zeros((4, 4, 4), dtype=float),
        "charge_native": 1.2e-5,
        "mass_amu": 0.9,
        "magnetic_moment_native": -2.0e-15,
        "spin_quantum_number": 0.5,
        "proper_time_step_ns": 1.0e-4,
    }

    legacy = equations._advance_rfs_rest_spin(rest_spin, **common)
    explicit_zero = equations._advance_rfs_rest_spin(
        rest_spin,
        **common,
        applied_radiation_reaction_force_native=np.zeros(3, dtype=float),
    )

    np.testing.assert_array_equal(explicit_zero, legacy)


def test_unprimed_rfs_medina_step_applies_no_reaction_impulse() -> None:
    electron = _particle_state("electron", beta=(0.01, 0.02, 0.0))
    config = _rfs_config(rider_species="electron")
    external_field = ExternalFieldConfig(
        electric_field_native=(electric_field_v_per_m_to_native(1.0e4), 0.0, 0.0)
    )

    off_trajectory, _, off_soa, _, *_ = _run(
        rider=electron,
        driver=_empty_state(),
        magnetic_dipole=config,
        steps=2,
        h_step=1.0e-6,
        radiation_reaction_mode="off",
        external_field=external_field,
        startup_mode=StartupMode.INERTIAL_PREHISTORY,
    )
    medina_trajectory, _, medina_soa, _, *_ = _run(
        rider=electron,
        driver=_empty_state(),
        magnetic_dipole=config,
        steps=2,
        h_step=1.0e-6,
        radiation_reaction_mode="medina_lad",
        external_field=external_field,
        startup_mode=StartupMode.INERTIAL_PREHISTORY,
    )

    assert off_soa is not None
    assert medina_soa is not None
    assert not np.any(medina_soa.medina_force_derivative_ready)
    assert not np.any(medina_soa.medina_impulse_capped)
    np.testing.assert_array_equal(medina_soa.radiation_reaction_work, 0.0)
    np.testing.assert_array_equal(medina_soa.radiation_energy_applied, 0.0)
    for key in (
        "x",
        "y",
        "z",
        "Px",
        "Py",
        "Pz",
        "spin_x",
        "spin_y",
        "spin_z",
    ):
        np.testing.assert_array_equal(
            off_trajectory[-1][key], medina_trajectory[-1][key]
        )

    # Exact RFS off/on controls share one spatial-momentum-authoritative input
    # boundary.  A derivative-unprimed Medina step therefore has no separate
    # reaction impulse or kinematic effect.
    final_state = medina_trajectory[-1]
    mass = float(final_state["m"][0])
    mechanical_momentum = np.asarray(
        [final_state[key][0] for key in ("Px", "Py", "Pz")], dtype=float
    )
    expected_gamma = float(
        np.sqrt(
            1.0
            + np.dot(mechanical_momentum, mechanical_momentum) / (mass * C_MMNS) ** 2
        )
    )
    expected_beta = mechanical_momentum / (expected_gamma * mass * C_MMNS)
    assert float(final_state["gamma"][0]) == expected_gamma
    np.testing.assert_allclose(
        [final_state[key][0] for key in ("bx", "by", "bz")],
        expected_beta,
        rtol=0.0,
        atol=2.0e-18,
    )
    assert float(final_state["Pt"][0]) == pytest.approx(
        expected_gamma * mass * C_MMNS,
        rel=0.0,
        abs=np.spacing(expected_gamma * mass * C_MMNS),
    )


def test_capped_medina_force_drives_constraint_compatible_rfs_spin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_compute = equations.compute_medina_radiation_reaction
    real_advance = equations._advance_rfs_rest_spin
    uncapped_forces: list[np.ndarray] = []
    spin_forces: list[np.ndarray] = []

    def amplified_medina(**kwargs: Any) -> MedinaRadiationReactionResult:
        result = real_compute(**kwargs)
        external_force = np.asarray(kwargs["external_force"], dtype=float)
        coordinate_dt = float(kwargs["coordinate_dt"])
        force = 2.0 * external_force
        impulse = force * coordinate_dt
        uncapped_forces.append(force.copy())
        return replace(
            result,
            radiation_reaction_force=(
                float(force[0]),
                float(force[1]),
                float(force[2]),
            ),
            radiation_reaction_impulse=(
                float(impulse[0]),
                float(impulse[1]),
                float(impulse[2]),
            ),
        )

    def recording_advance(*args: Any, **kwargs: Any) -> np.ndarray:
        spin_forces.append(
            np.asarray(
                kwargs["applied_radiation_reaction_force_native"], dtype=float
            ).copy()
        )
        return cast(np.ndarray, real_advance(*args, **kwargs))

    monkeypatch.setattr(
        equations, "compute_medina_radiation_reaction", amplified_medina
    )
    monkeypatch.setattr(equations, "_advance_rfs_rest_spin", recording_advance)
    trajectory, _, soa, _, *_ = _run(
        rider=_particle_state("electron", beta=(0.01, 0.02, 0.0)),
        driver=_empty_state(),
        magnetic_dipole=_rfs_config(rider_species="electron"),
        steps=3,
        h_step=1.0e-6,
        radiation_reaction_mode="medina_lad",
        external_field=ExternalFieldConfig(
            electric_field_native=(
                electric_field_v_per_m_to_native(1.0e4),
                0.0,
                0.0,
            )
        ),
    )

    assert soa is not None
    assert len(uncapped_forces) == 2
    assert len(spin_forces) == 2
    np.testing.assert_array_equal(spin_forces[0], 0.0)
    # The production guard caps the two-times-external impulse to 25% of the
    # external impulse. Spin must follow that applied force, not the amplified
    # uncapped kernel result.
    np.testing.assert_allclose(spin_forces[1], 0.125 * uncapped_forces[1])
    assert bool(soa.medina_force_derivative_ready[-1, 0])
    assert bool(soa.medina_impulse_capped[-1, 0])

    final_state = trajectory[-1]
    beta_end = np.asarray(
        [final_state[key][0] for key in ("bx", "by", "bz")], dtype=float
    )
    gamma_end = float(final_state["gamma"][0])
    spin_end = boost_rest_polarization(
        [final_state[key][0] for key in ("spin_x", "spin_y", "spin_z")],
        beta_end,
    )
    u_end = C_MMNS * np.concatenate(([gamma_end], gamma_end * beta_end))
    assert minkowski_dot(u_end, spin_end) / C_MMNS == pytest.approx(0.0, abs=2.0e-14)
    assert minkowski_dot(spin_end, spin_end) == pytest.approx(-1.0, rel=2.0e-14)


def test_medina_predictor_and_post_kick_endpoint_share_one_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    electron = _particle_state("electron", beta=(0.01, 0.02, 0.0))
    electron.update(
        {
            "spin_x": np.array([0.0]),
            "spin_y": np.array([0.0]),
            "spin_z": np.array([1.0]),
            "local_magnetic_field_x_t": np.array([0.0]),
            "local_magnetic_field_y_t": np.array([0.0]),
            "local_magnetic_field_z_t": np.array([0.0]),
            "magnetic_moment_j_per_t": np.array([0.0]),
            "magnetic_moment_native": np.array([0.0]),
            "spin_quantum_number": np.array([0.5]),
            "gyromagnetic_ratio_rad_s_t": np.array([0.0]),
            "magnetic_dipole_active": np.array([True]),
            "spin_precession_active": np.array([True]),
            "stern_gerlach_active": np.array([False]),
            "dipole_source_canonical_ready": np.array([False]),
            "medina_external_force_x": np.array([0.0]),
            "medina_external_force_y": np.array([0.0]),
            "medina_external_force_z": np.array([0.0]),
            "medina_external_force_sample_time": np.array([-1.0e-6]),
            "medina_force_derivative_ready": np.array([False]),
            "medina_impulse_capped": np.array([False]),
        }
    )
    raw_pt = float(electron["Pt"][0] - 1.0e-9)
    electron["Pt"][0] = raw_pt

    real_compute = equations.compute_medina_radiation_reaction
    real_advance = equations._advance_rfs_rest_spin
    kernel_calls: list[tuple[dict[str, Any], MedinaRadiationReactionResult]] = []
    spin_forces: list[np.ndarray] = []

    def forced_medina(**kwargs: Any) -> MedinaRadiationReactionResult:
        base = real_compute(**kwargs)
        predictor_dt = float(kwargs["coordinate_dt"])
        external_force = np.asarray(kwargs["external_force"], dtype=float)
        impulse = 0.1 * external_force * predictor_dt
        forced = replace(
            base,
            radiation_reaction_force=tuple(impulse / predictor_dt),
            radiation_reaction_impulse=tuple(impulse),
            reaction_work=-7.0 * predictor_dt,
            far_radiated_power=11.0,
            far_radiated_energy=11.0 * predictor_dt,
            cross_field_energy=13.0,
            cross_field_energy_change=-3.0 * predictor_dt,
        )
        kernel_calls.append((dict(kwargs), forced))
        return forced

    def recording_advance(*args: Any, **kwargs: Any) -> np.ndarray:
        spin_forces.append(
            np.asarray(
                kwargs["applied_radiation_reaction_force_native"], dtype=float
            ).copy()
        )
        return cast(np.ndarray, real_advance(*args, **kwargs))

    monkeypatch.setattr(equations, "compute_medina_radiation_reaction", forced_medina)
    monkeypatch.setattr(equations, "_advance_rfs_rest_spin", recording_advance)

    h_step = 1.0e-6
    result = equations.retarded_equations_of_motion(
        h_step,
        [electron],
        [],
        0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.INERTIAL_PREHISTORY,
        self_consistency=SelfConsistencyConfig(
            enabled=True,
            max_iterations=2,
            mass_shell_relaxation=0.7,
        ),
        radiation_reaction_mode="medina_lad",
        external_field=ExternalFieldConfig(magnetic_field_native=(0.0, 1.0e7, 0.0)),
        magnetic_dipole=_rfs_config(rider_species="electron"),
    )

    assert len(kernel_calls) == 2
    assert len(spin_forces) == 1
    final_kwargs, final_kernel = kernel_calls[-1]
    predictor_dt = float(final_kwargs["coordinate_dt"])
    external_force = np.asarray(final_kwargs["external_force"], dtype=float)
    impulse = np.asarray(final_kernel.radiation_reaction_impulse, dtype=float)
    mass = float(electron["m"][0])
    previous_beta = np.asarray(
        [electron[key][0] for key in ("bx", "by", "bz")], dtype=float
    )
    previous_momentum = float(electron["gamma"][0]) * mass * C_MMNS * previous_beta
    predictor_momentum = previous_momentum + external_force * predictor_dt
    final_momentum = predictor_momentum + impulse
    final_gamma = float(
        np.sqrt(1.0 + np.dot(final_momentum, final_momentum) / (mass * C_MMNS) ** 2)
    )
    final_beta = final_momentum / (final_gamma * mass * C_MMNS)
    final_dt = h_step * final_gamma

    assert result["medina_external_force_sample_time"][0] == pytest.approx(
        float(electron["t"][0]) + 0.5 * predictor_dt
    )
    assert result["radiation_reaction_work"][0] == pytest.approx(
        final_kernel.reaction_work
    )
    assert result["radiation_energy"][0] == pytest.approx(
        final_kernel.far_radiated_energy
    )
    assert result["medina_cross_field_energy"][0] == pytest.approx(13.0)
    assert result["medina_cross_field_energy_change"][0] == pytest.approx(
        final_kernel.cross_field_energy_change
    )
    np.testing.assert_allclose(spin_forces[0], impulse / predictor_dt)
    np.testing.assert_allclose(
        [result[key][0] for key in ("Px", "Py", "Pz")], final_momentum
    )
    assert result["Pt"][0] == pytest.approx(final_gamma * mass * C_MMNS)
    assert result["gamma"][0] == pytest.approx(final_gamma)
    np.testing.assert_allclose(
        [result[key][0] for key in ("bx", "by", "bz")], final_beta
    )
    np.testing.assert_allclose(
        [result[key][0] - electron[key][0] for key in ("x", "y", "z")],
        h_step * final_momentum / mass,
    )
    assert result["t"][0] - electron["t"][0] == pytest.approx(final_dt)
    np.testing.assert_allclose(
        [result[key][0] for key in ("bdotx", "bdoty", "bdotz")],
        (final_beta - previous_beta) / (C_MMNS * final_dt),
    )
    predictor_gamma = float(
        np.sqrt(
            1.0 + np.dot(predictor_momentum, predictor_momentum) / (mass * C_MMNS) ** 2
        )
    )
    assert result["mass_shell_projection_energy"][0] == pytest.approx(
        C_MMNS * (predictor_gamma * mass * C_MMNS - raw_pt)
    )


def test_rfs_rejects_approximate_charge_source_history() -> None:
    with pytest.raises(ValueError, match="requires COLD_START"):
        _run(
            rider=_particle_state("neutron", position_mm=(0.0, 0.1, 0.0)),
            driver=_particle_state("proton", beta=(0.1, 0.0, 0.0)),
            magnetic_dipole=_rfs_config(),
            startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        )


def test_rfs_history_guard_includes_charged_rider_sources() -> None:
    reversed_config = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=True,
        stern_gerlach_force_enabled=True,
        rider=MagneticDipoleParticleConfig(species="proton", polarization=0.0),
        driver=MagneticDipoleParticleConfig(species="neutron", polarization=1.0),
    )

    with pytest.raises(ValueError, match="requires COLD_START"):
        _run(
            rider=_particle_state("proton", beta=(0.1, 0.0, 0.0)),
            driver=_particle_state("neutron", position_mm=(0.0, 0.1, 0.0)),
            magnetic_dipole=reversed_config,
            startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        )


def test_rfs_rejects_shrunken_partial_polarization() -> None:
    with pytest.raises(
        ValueError, match="Partial polarization requires a weighted ensemble"
    ):
        _run(
            rider=_particle_state("neutron", position_mm=(0.0, 0.1, 0.0)),
            driver=_particle_state("proton", beta=(0.1, 0.0, 0.0)),
            magnetic_dipole=_rfs_config(rider_polarization=0.5),
        )
