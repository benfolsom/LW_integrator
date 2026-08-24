"""Focused regression tests for the finite inertial-prehistory boundary model."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.external_fields import electric_field_v_per_m_to_native
from core.integration_runner import (
    _estimate_inertial_prehistory_duration_ns,
    _build_inertial_coasting_history,
    _initialize_magnetic_dipole_state,
    _maximum_beta_magnitude,
    retarded_integrator,
)
from core.charge_source_interactions import (
    evaluate_retarded_charge_source_interaction_native,
)
from core.dipole_source_interactions import (
    evaluate_retarded_dipole_source_interaction_native,
)
from core.retarded_dipole_fields import (
    evaluate_retarded_dipole_field_gradient_native,
)
from core.retarded_fields import (
    ObserverEvent,
    RetardedHistoryError,
    evaluate_retarded_charge_field_gradient_native,
)
from core.self_consistency import SelfConsistencyConfig
from core.species import get_species
from core.types import (
    DipoleSourceConfig,
    ExternalFieldConfig,
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    SimulationType,
    StartupMode,
)


def _state(
    *,
    position_mm: tuple[float, float, float],
    beta: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass_amu: float = 1.0,
    observer_charge: float = 0.0,
    source_charge: float | None = None,
    magnetic_moment_native: float = 0.0,
    spin: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> dict[str, np.ndarray]:
    beta_vector = np.asarray(beta, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - float(beta_vector @ beta_vector))
    momentum = gamma * mass_amu * C_MMNS * beta_vector
    source_q = observer_charge if source_charge is None else float(source_charge)
    return {
        "x": np.array([position_mm[0]], dtype=float),
        "y": np.array([position_mm[1]], dtype=float),
        "z": np.array([position_mm[2]], dtype=float),
        "t": np.array([0.0]),
        "Px": np.array([momentum[0]]),
        "Py": np.array([momentum[1]]),
        "Pz": np.array([momentum[2]]),
        "Pt": np.array([gamma * mass_amu * C_MMNS]),
        "gamma": np.array([gamma]),
        "bx": np.array([beta_vector[0]]),
        "by": np.array([beta_vector[1]]),
        "bz": np.array([beta_vector[2]]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "q": np.array([observer_charge]),
        "q_species": np.array([observer_charge]),
        "q_observer": np.array([observer_charge]),
        "q_source": np.array([source_q]),
        "macro_population": np.array([1.0]),
        "m": np.array([mass_amu]),
        "m_species": np.array([mass_amu]),
        "char_time": np.array([0.0]),
        "spin_x": np.array([spin[0]]),
        "spin_y": np.array([spin[1]]),
        "spin_z": np.array([spin[2]]),
        "magnetic_moment_native": np.array([magnetic_moment_native]),
        "magnetic_dipole_active": np.array([magnetic_moment_native != 0.0]),
        "_dead_particles": np.array([False]),
    }


def _species_state(
    species_name: str,
    *,
    position_mm: tuple[float, float, float],
    beta: tuple[float, float, float] = (0.0, 0.0, 0.0),
    source_charge: float | None = None,
) -> dict[str, np.ndarray]:
    species = get_species(species_name)
    charge = float(species.charge_e) * ELEMENTARY_CHARGE
    return _state(
        position_mm=position_mm,
        beta=beta,
        mass_amu=species.mass_amu,
        observer_charge=charge,
        source_charge=source_charge,
    )


@pytest.mark.parametrize("knot_count", [2, 4, 8])
def test_sparse_inertial_history_has_exact_positions_and_times(knot_count: int) -> None:
    active = _state(
        position_mm=(1.25, -0.75, 0.5),
        beta=(0.12, -0.04, 0.03),
        observer_charge=0.5,
        magnetic_moment_native=0.2,
    )
    duration_ns = 0.025

    history = _build_inertial_coasting_history(
        active,
        duration_ns,
        knot_count=knot_count,
    )

    assert len(history) == knot_count
    offsets = np.linspace(-duration_ns, 0.0, knot_count)
    for state, offset in zip(history, offsets):
        assert float(state["t"][0]) == pytest.approx(float(offset), abs=0.0)
        for axis, beta_component, initial_position in zip(
            "xyz",
            (0.12, -0.04, 0.03),
            (1.25, -0.75, 0.5),
        ):
            assert float(state[axis][0]) == pytest.approx(
                initial_position + beta_component * C_MMNS * float(offset),
                rel=0.0,
                abs=2.0e-15,
            )
            np.testing.assert_array_equal(state[f"bdot{axis}"], 0.0)
        assert np.isnan(float(state["medina_external_force_sample_time"][0]))
        assert not bool(state["medina_force_derivative_ready"][0])

    for key in ("x", "y", "z", "t", "Px", "Py", "Pz", "Pt", "gamma"):
        np.testing.assert_array_equal(history[-1][key], active[key])


def test_exact_charge_and_dipole_fields_are_invariant_to_doubled_duration() -> None:
    source = _state(
        position_mm=(0.0, 0.0, 0.0),
        beta=(0.05, -0.02, 0.01),
        source_charge=0.7,
        magnetic_moment_native=0.08,
        spin=(0.2, -0.3, 0.9327379053088815),
    )
    observer = ObserverEvent(0.0, (2.0, 0.5, -0.25))
    histories = [
        _build_inertial_coasting_history(source, duration, knot_count=4)
        for duration in (0.05, 0.10)
    ]

    charge_results = [
        evaluate_retarded_charge_field_gradient_native(
            history,
            observer,
            relative_step=2.0e-4,
            minimum_step_mm=1.0e-12,
        )
        for history in histories
    ]
    dipole_results = [
        evaluate_retarded_dipole_field_gradient_native(
            history,
            observer,
            relative_step=8.0e-4,
            minimum_step_mm=1.0e-12,
            minimum_separation_mm=1.0e-6,
        )
        for history in histories
    ]

    np.testing.assert_allclose(
        charge_results[1].field.four_potential,
        charge_results[0].field.four_potential,
        rtol=2.0e-12,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        charge_results[1].field.field_tensor,
        charge_results[0].field.field_tensor,
        rtol=2.0e-11,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        charge_results[1].partial_a,
        charge_results[0].partial_a,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        charge_results[1].partial_f,
        charge_results[0].partial_f,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    for field_name in ("four_potential", "field_tensor", "partial_a"):
        np.testing.assert_allclose(
            getattr(dipole_results[1], field_name),
            getattr(dipole_results[0], field_name),
            rtol=2.0e-7,
            atol=2.0e-11,
        )
    # The third nested centered difference is most sensitive to the root
    # solver's last few ulps.  Its norm is nevertheless stable well below the
    # stencil-convergence accuracy expected of this finite-difference oracle.
    assert (
        np.linalg.norm(dipole_results[1].partial_f - dipole_results[0].partial_f)
        / np.linalg.norm(dipole_results[0].partial_f)
        < 2.0e-7
    )


def _run_simple_inertial(
    rider: dict[str, np.ndarray],
    driver: dict[str, np.ndarray],
    *,
    steps: int,
    h_step: float = 1.0e-6,
    radiation_reaction_mode: str = "off",
    external_field: ExternalFieldConfig | None = None,
    magnetic_dipole: MagneticDipoleConfig | None = None,
):
    return retarded_integrator(
        steps=steps,
        h_step=h_step,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=copy.deepcopy(rider),
        init_driver=copy.deepcopy(driver),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        startup_mode=StartupMode.INERTIAL_PREHISTORY,
        radiation_reaction_mode=radiation_reaction_mode,
        external_field=external_field,
        magnetic_dipole=magnetic_dipole,
        self_consistency=SelfConsistencyConfig(enabled=False),
        use_numba=False,
    )


def _independent_total_canonical_offset(
    source_history: list[dict[str, np.ndarray]],
    observer_state: dict[str, np.ndarray],
    magnetic: MagneticDipoleConfig,
) -> np.ndarray:
    """Evaluate q(A_charge + A_dipole)/c without the startup preflight."""

    beta = np.array(
        [float(observer_state[f"b{axis}"][0]) for axis in "xyz"], dtype=float
    )
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    four_velocity = gamma * C_MMNS * np.concatenate(((1.0,), beta))
    event = ObserverEvent(
        float(observer_state["t"][0]),
        tuple(float(observer_state[axis][0]) for axis in "xyz"),
    )
    observer_charge = float(observer_state["q_observer"][0])
    source = magnetic.source
    charge = evaluate_retarded_charge_source_interaction_native(
        source_history,
        event,
        four_velocity_mm_ns=four_velocity,
        observer_charge_native=observer_charge,
        proper_time_step_ns=0.0,
        relative_step=max(1.0e-4, source.relative_stencil_step),
        minimum_step_mm=max(1.0e-15, source.minimum_stencil_step_mm),
        root_tolerance_mm=source.root_tolerance_mm,
        max_root_iterations=source.max_root_iterations,
    )
    dipole = evaluate_retarded_dipole_source_interaction_native(
        source_history,
        event,
        four_velocity_mm_ns=four_velocity,
        observer_charge_native=observer_charge,
        proper_time_step_ns=0.0,
        relative_step=source.relative_stencil_step,
        minimum_step_mm=source.minimum_stencil_step_mm,
        minimum_separation_mm=source.minimum_separation_mm,
        root_tolerance_mm=source.root_tolerance_mm,
        max_root_iterations=source.max_root_iterations,
    )
    return charge.canonical_potential_momentum + dipole.canonical_potential_momentum


def test_inertial_prefix_is_hidden_from_both_public_trajectory_forms() -> None:
    rider = _species_state(
        "neutron", position_mm=(-1.0, 0.0, 0.0), beta=(0.01, 0.0, 0.0)
    )
    driver = _species_state(
        "neutron", position_mm=(1.0, 0.0, 0.0), beta=(-0.01, 0.0, 0.0)
    )
    magnetic = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=True,
        stern_gerlach_force_enabled=False,
        rider=MagneticDipoleParticleConfig(species="neutron"),
        driver=MagneticDipoleParticleConfig(species="neutron"),
    )

    rider_traj, driver_traj, rider_soa, driver_soa, *_ = _run_simple_inertial(
        rider,
        driver,
        steps=4,
        magnetic_dipole=magnetic,
    )

    assert len(rider_traj) == 4
    assert len(driver_traj) == 4
    assert rider_soa is not None and rider_soa.n_steps == 4
    assert driver_soa is not None and driver_soa.n_steps == 4
    np.testing.assert_array_equal(rider_traj[0]["t"], rider["t"])
    np.testing.assert_array_equal(driver_traj[0]["t"], driver["t"])
    np.testing.assert_array_equal(rider_traj[0]["x"], rider["x"])
    np.testing.assert_array_equal(driver_traj[0]["x"], driver["x"])
    assert float(np.min(rider_soa.t)) >= 0.0
    assert float(np.min(driver_soa.t)) >= 0.0


def test_public_startup_momentum_is_rebased_by_charge_plus_dipole_once() -> None:
    rider = _species_state(
        "electron",
        position_mm=(-5.0e-9, 0.0, 0.0),
        beta=(0.0, 1.0e-3, 0.0),
    )
    driver = _species_state(
        "proton",
        position_mm=(5.0e-9, 0.0, 0.0),
        beta=(0.0, -2.0e-3, 0.0),
    )
    magnetic = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=False,
        stern_gerlach_force_enabled=False,
        source=DipoleSourceConfig(
            model="covariant_retarded_point",
            minimum_separation_mm=2.0e-9,
        ),
        rider=MagneticDipoleParticleConfig(species="electron"),
        driver=MagneticDipoleParticleConfig(species="proton"),
    )

    initialized_rider = copy.deepcopy(rider)
    initialized_driver = copy.deepcopy(driver)
    _initialize_magnetic_dipole_state(
        initialized_rider,
        magnetic.rider,
        magnetic,
        role="rider",
    )
    _initialize_magnetic_dipole_state(
        initialized_driver,
        magnetic.driver,
        magnetic,
        role="driver",
    )
    duration = _estimate_inertial_prehistory_duration_ns(
        initialized_rider,
        initialized_driver,
        magnetic,
    )
    driver_prefix = _build_inertial_coasting_history(
        initialized_driver,
        duration,
    )
    initial_offset = _independent_total_canonical_offset(
        driver_prefix,
        initialized_rider,
        magnetic,
    )

    rider_traj, driver_traj, *_ = _run_simple_inertial(
        rider,
        driver,
        steps=2,
        h_step=1.0e-24,
        magnetic_dipole=magnetic,
    )
    component_keys = ("Pt", "Px", "Py", "Pz")
    input_mechanical = np.array([float(rider[key][0]) for key in component_keys])
    public_initial = np.array([float(rider_traj[0][key][0]) for key in component_keys])

    np.testing.assert_allclose(
        public_initial,
        input_mechanical + initial_offset,
        rtol=3.0e-12,
        atol=1.0e-16,
    )
    for key in ("x", "y", "z", "bx", "by", "bz", "gamma"):
        np.testing.assert_array_equal(rider_traj[0][key], rider[key])
    assert bool(rider_traj[0]["charge_source_canonical_ready"][0])
    assert bool(rider_traj[0]["dipole_source_canonical_ready"][0])

    recovered_mechanical = public_initial - initial_offset
    np.testing.assert_allclose(
        recovered_mechanical,
        input_mechanical,
        rtol=3.0e-12,
        atol=1.0e-16,
    )
    mass_shell = recovered_mechanical[0] ** 2 - float(
        recovered_mechanical[1:] @ recovered_mechanical[1:]
    )
    expected_mass_shell = (float(rider["m"][0]) * C_MMNS) ** 2
    assert mass_shell == pytest.approx(expected_mass_shell, rel=3.0e-12)

    # Reconstruct the source history available at the first evolved sample and
    # check the canonical relation there as well.  If startup qA/c were added a
    # second time, this residual would be approximately one whole offset.
    driver_history_at_step_one = driver_prefix[:-1] + driver_traj
    evolved_offset = _independent_total_canonical_offset(
        driver_history_at_step_one,
        rider_traj[1],
        magnetic,
    )
    evolved_beta = np.array(
        [float(rider_traj[1][f"b{axis}"][0]) for axis in "xyz"], dtype=float
    )
    evolved_gamma = float(rider_traj[1]["gamma"][0])
    evolved_mechanical = (
        evolved_gamma
        * float(rider_traj[1]["m"][0])
        * C_MMNS
        * np.concatenate(((1.0,), evolved_beta))
    )
    evolved_canonical = np.array(
        [float(rider_traj[1][key][0]) for key in component_keys]
    )
    np.testing.assert_allclose(
        evolved_canonical,
        evolved_mechanical + evolved_offset,
        rtol=3.0e-10,
        atol=1.0e-15,
    )


def test_inertial_prefix_does_not_prime_medina_force_derivative() -> None:
    electron = get_species("electron")
    rider = _species_state(
        "electron",
        position_mm=(-1.0, 0.0, 0.0),
        beta=(0.01, 0.0, 0.0),
        source_charge=0.0,
    )
    driver = _species_state(
        "neutron",
        position_mm=(1.0, 0.0, 0.0),
        source_charge=0.0,
    )
    magnetic = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=True,
        stern_gerlach_force_enabled=False,
        rider=MagneticDipoleParticleConfig(species="electron"),
        driver=MagneticDipoleParticleConfig(species="neutron"),
    )
    field = ExternalFieldConfig(
        electric_field_native=(electric_field_v_per_m_to_native(1.0e4), 0.0, 0.0)
    )

    off_traj, _, _, _, *_ = _run_simple_inertial(
        rider,
        driver,
        steps=2,
        radiation_reaction_mode="off",
        external_field=field,
        magnetic_dipole=magnetic,
    )
    medina_traj, _, medina_soa, _, *_ = _run_simple_inertial(
        rider,
        driver,
        steps=2,
        radiation_reaction_mode="medina_lad",
        external_field=field,
        magnetic_dipole=magnetic,
    )

    assert electron.charge_e == -1
    assert medina_soa is not None
    assert np.isnan(float(medina_soa.medina_external_force_sample_time[0, 0]))
    assert np.isfinite(float(medina_soa.medina_external_force_sample_time[1, 0]))
    assert not np.any(medina_soa.medina_force_derivative_ready)
    assert not np.any(medina_soa.medina_impulse_capped)
    for key in ("x", "y", "z", "t", "Px", "Py", "Pz", "Pt", "gamma"):
        np.testing.assert_array_equal(medina_traj[-1][key], off_traj[-1][key])


def test_inertial_history_rejects_nonzero_acceleration_and_nonfinite_beta() -> None:
    accelerating = _state(position_mm=(0.0, 0.0, 0.0))
    accelerating["bdotx"][:] = 1.0e-3
    with pytest.raises(ValueError, match="requires zero initial bdot"):
        _build_inertial_coasting_history(accelerating, 0.1)

    nonfinite = _state(position_mm=(0.0, 0.0, 0.0))
    nonfinite["bx"][:] = np.nan
    with pytest.raises(ValueError, match="finite beta components"):
        _maximum_beta_magnitude(nonfinite)


def test_runtime_missing_exact_history_raises_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.dipole_source_interactions as source_interactions
    import core.integration_runner as integration_runner

    neutron = get_species("neutron")
    rider = _species_state("neutron", position_mm=(-5.0e-5, 0.0, 0.0))
    driver = _species_state("neutron", position_mm=(5.0e-5, 0.0, 0.0))
    magnetic = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=False,
        stern_gerlach_force_enabled=False,
        source=DipoleSourceConfig(
            model="covariant_retarded_point",
            minimum_separation_mm=2.0e-9,
        ),
        rider=MagneticDipoleParticleConfig(species="neutron"),
        driver=MagneticDipoleParticleConfig(species="neutron"),
    )

    monkeypatch.setattr(
        integration_runner,
        "_preflight_inertial_exact_histories",
        lambda rider_history, driver_history, **kwargs: (
            np.zeros((len(rider_history[-1]["x"]), 4), dtype=float),
            np.zeros((len(driver_history[-1]["x"]), 4), dtype=float),
        ),
    )

    def missing_history(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RetardedHistoryError("synthetic runtime history loss")

    monkeypatch.setattr(
        source_interactions,
        "evaluate_retarded_dipole_source_interaction_native",
        missing_history,
    )

    assert neutron.magnetic_moment_j_t is not None
    with pytest.raises(RetardedHistoryError, match="synthetic runtime history loss"):
        _run_simple_inertial(
            rider,
            driver,
            steps=2,
            magnetic_dipole=magnetic,
        )
