from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from core.adaptive_pair_return import (
    AdaptivePairControllerState,
    run_exact_pair_adaptive_window,
)
from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.exact_pair_trial import (
    ExactPairEOMOptions,
    commit_accepted_exact_pair_step_doubling_trial,
    make_exact_role_eom_advance,
    solve_exact_pair_slab_trial,
    solve_exact_pair_step_doubling_trial,
)
from core.integration_runner import (
    _apply_inertial_canonical_rebase,
    _build_inertial_coasting_history,
    _estimate_inertial_prehistory_duration_ns,
    _initialize_magnetic_dipole_state,
    _preflight_inertial_exact_histories,
)
from core.magnetic_dipole import minkowski_dot
from core.self_consistency import SelfConsistencyConfig
from core.shared_lab_time import SharedLabTimeError
from core.spin_self_force_reduction_history import (
    AcceptedPairIntrinsicSpinReductionHistory,
    build_accepted_pair_intrinsic_spin_reduction_diagnostic_candidate,
)
from core.species import get_species
from core.step_doubling import (
    ErrorScale,
    StepControllerConfig,
    StepDoublingTolerances,
)
from core.types import (
    ChronoMatchingMode,
    DipoleSourceConfig,
    GrowableTrajectoryBuilder,
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    StartupMode,
    TrialTrajectoryHistory,
)


def _state(time_ns: float, position_mm: float) -> dict[str, np.ndarray]:
    return {
        "x": np.array([position_mm]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "t": np.array([time_ns]),
        "Px": np.array([0.0]),
        "Py": np.array([0.0]),
        "Pz": np.array([0.0]),
        "Pt": np.array([1.0]),
        "gamma": np.array([1.0]),
        "bx": np.array([0.0]),
        "by": np.array([0.0]),
        "bz": np.array([0.0]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "q": np.array([0.0]),
        "q_source": np.array([0.0]),
        "m": np.array([1.0]),
        "_exact_source_start_four_potential": np.zeros((1, 4)),
        "_exact_source_endpoint_rebase_required": np.array([False]),
    }


def _accepted(position_mm: float) -> GrowableTrajectoryBuilder:
    builder = GrowableTrajectoryBuilder(2, 1)
    builder.append_step(_state(0.0, position_mm))
    return builder


def _coasting_state(
    time_ns: float,
    position_mm: float,
    gamma: float,
) -> dict[str, np.ndarray]:
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    zero = np.array([0.0])
    state = _state(time_ns, position_mm)
    state.update(
        {
            "Pz": np.array([gamma * C_MMNS * beta]),
            "Pt": np.array([gamma * C_MMNS]),
            "gamma": np.array([gamma]),
            "bz": np.array([beta]),
            "q_observer": zero.copy(),
            "spin_x": zero.copy(),
            "spin_y": zero.copy(),
            "spin_z": np.array([1.0]),
            "magnetic_moment_native": zero.copy(),
            "magnetic_dipole_active": zero.copy(),
            "spin_precession_active": zero.copy(),
            "stern_gerlach_active": zero.copy(),
            "origin_x": np.array([position_mm]),
            "origin_y": zero.copy(),
            "origin_z": zero.copy(),
            "beta_avg_x": zero.copy(),
            "beta_avg_y": zero.copy(),
            "beta_avg_z": np.array([beta]),
            "beta_samples": np.array([1.0]),
        }
    )
    return state


def _coasting_history(position_mm: float, gamma: float) -> GrowableTrajectoryBuilder:
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    builder = GrowableTrajectoryBuilder(2, 1, magnetic_dipole=True)
    builder.append_step(
        _coasting_state(
            -0.1,
            position_mm - beta * C_MMNS * 0.1,
            gamma,
        )
    )
    builder.append_step(_coasting_state(0.0, position_mm, gamma))
    return builder


def _charged_species_state(
    species_name: str,
    *,
    position_mm: float,
) -> dict[str, np.ndarray]:
    species = get_species(species_name)
    state = _state(0.0, position_mm)
    charge = float(species.charge_e) * ELEMENTARY_CHARGE
    state.update(
        {
            "Pt": np.array([species.mass_amu * C_MMNS]),
            "q": np.array([charge]),
            "q_species": np.array([charge]),
            "q_observer": np.array([charge]),
            "q_source": np.array([charge]),
            "macro_population": np.array([1.0]),
            "m": np.array([species.mass_amu]),
            "m_species": np.array([species.mass_amu]),
            "char_time": np.array([0.0]),
            "_dead_particles": np.array([False]),
        }
    )
    return state


def _charged_accepted_pair(
    *,
    include_dipole_source: bool = False,
    exact_retarded_update: str = "first_order_endpoint",
    intrinsic_spin_self_reaction_mode: str = "off",
) -> tuple[
    GrowableTrajectoryBuilder,
    GrowableTrajectoryBuilder,
    MagneticDipoleConfig,
]:
    rider = _charged_species_state("electron", position_mm=-5.0e-3)
    driver = _charged_species_state("proton", position_mm=5.0e-3)
    magnetic = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=True,
        stern_gerlach_force_enabled=True,
        source=DipoleSourceConfig(
            model=("covariant_retarded_point" if include_dipole_source else "off")
        ),
        exact_retarded_update=exact_retarded_update,
        intrinsic_spin_self_reaction_mode=intrinsic_spin_self_reaction_mode,
        rider=MagneticDipoleParticleConfig(species="electron"),
        driver=MagneticDipoleParticleConfig(species="proton"),
    )
    _initialize_magnetic_dipole_state(
        rider,
        magnetic.rider,
        magnetic,
        role="rider",
    )
    _initialize_magnetic_dipole_state(
        driver,
        magnetic.driver,
        magnetic,
        role="driver",
    )
    duration = _estimate_inertial_prehistory_duration_ns(rider, driver, magnetic)
    rider_seed = _build_inertial_coasting_history(rider, duration)
    driver_seed = _build_inertial_coasting_history(driver, duration)
    rider_offset, driver_offset = _preflight_inertial_exact_histories(
        rider_seed,
        driver_seed,
        magnetic_dipole=magnetic,
        charge_field_required=True,
        dipole_field_required=include_dipole_source,
    )
    _apply_inertial_canonical_rebase(
        rider_seed[-1],
        rider_offset,
        charge_field_ready=True,
        dipole_field_ready=include_dipole_source,
    )
    _apply_inertial_canonical_rebase(
        driver_seed[-1],
        driver_offset,
        charge_field_ready=True,
        dipole_field_ready=include_dipole_source,
    )
    rider_builder = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
    driver_builder = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
    for state in rider_seed:
        rider_builder.append_step(state)
    for state in driver_seed:
        driver_builder.append_step(state)
    return rider_builder, driver_builder, magnetic


def _advance(scale: float, seen: list[object]):
    def advance(
        proper_step_ns: float,
        observer_start: dict[str, np.ndarray],
        source_start: dict[str, np.ndarray],
        exact_source_history: object,
    ) -> dict[str, np.ndarray]:
        seen.append(exact_source_history)
        result = copy.deepcopy(observer_start)
        result["t"] = np.array([float(observer_start["t"][0]) + scale * proper_step_ns])
        result["x"] = np.array([float(observer_start["x"][0]) + proper_step_ns])
        result["_exact_source_start_four_potential"] = np.zeros((1, 4))
        result["_exact_source_endpoint_rebase_required"] = np.array([False])
        result["spin_x"] = np.array([0.0])
        result["spin_y"] = np.array([0.0])
        result["spin_z"] = np.array([1.0])
        result["radiation_energy"] = np.array([proper_step_ns**2])
        result["radiation_reaction_work"] = np.array([0.0])
        result["medina_cross_field_energy_change"] = np.array([0.0])
        result["mass_shell_projection_energy"] = np.array([0.0])
        assert float(source_start["t"][0]) == pytest.approx(
            float(observer_start["t"][0])
        )
        return result

    return advance


def test_one_slab_trial_is_unpublished_and_endpoint_finalized() -> None:
    rider_builder = _accepted(-1.0)
    driver_builder = _accepted(1.0)
    accepted_rider = rider_builder.build_current()
    accepted_driver = driver_builder.build_current()
    rider_x_before = accepted_rider.x.copy()
    driver_x_before = accepted_driver.x.copy()
    rider_seen: list[object] = []
    driver_seen: list[object] = []

    trial = solve_exact_pair_slab_trial(
        accepted_rider_history=accepted_rider,
        accepted_driver_history=accepted_driver,
        advance_rider=_advance(2.0, rider_seen),
        advance_driver=_advance(4.0, driver_seen),
        delta_time_ns=0.2,
        rider_initial_proper_step_ns=0.1,
        driver_initial_proper_step_ns=0.1,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
    )

    assert trial.pair.rider.proper_step_ns == pytest.approx(0.1)
    assert trial.pair.driver.proper_step_ns == pytest.approx(0.05)
    assert trial.pair.rider.coordinate_time_ns == pytest.approx(0.2)
    assert trial.pair.driver.coordinate_time_ns == pytest.approx(0.2)
    assert trial.rider_history.n_steps == 2
    assert trial.driver_history.n_steps == 2
    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1
    np.testing.assert_array_equal(accepted_rider.x, rider_x_before)
    np.testing.assert_array_equal(accepted_driver.x, driver_x_before)
    assert all(history is accepted_driver for history in rider_seen)
    assert all(history is accepted_rider for history in driver_seen)
    for state in (trial.pair.rider.state, trial.pair.driver.state):
        assert "_exact_source_start_four_potential" not in state
        assert "_exact_source_endpoint_rebase_required" not in state


def test_next_slab_accepts_the_pair_commit_time_envelope() -> None:
    rider_builder = GrowableTrajectoryBuilder(2, 1)
    driver_builder = GrowableTrajectoryBuilder(2, 1)
    rider_builder.append_step(_state(0.0, -1.0))
    driver_builder.append_step(_state(1.5e-18, 1.0))

    trial = solve_exact_pair_slab_trial(
        accepted_rider_history=rider_builder.build_current(),
        accepted_driver_history=driver_builder.build_current(),
        advance_rider=_advance(2.0, []),
        advance_driver=_advance(4.0, []),
        delta_time_ns=0.2,
        rider_initial_proper_step_ns=0.1,
        driver_initial_proper_step_ns=0.05,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
    )

    assert trial.pair.synchronization_residual_ns <= 2.0e-18


def test_next_slab_rejects_times_outside_the_pair_commit_envelope() -> None:
    rider_builder = GrowableTrajectoryBuilder(2, 1)
    driver_builder = GrowableTrajectoryBuilder(2, 1)
    rider_builder.append_step(_state(0.0, -1.0))
    driver_builder.append_step(_state(2.1e-18, 1.0))

    with pytest.raises(SharedLabTimeError, match="starts are not synchronized"):
        solve_exact_pair_slab_trial(
            accepted_rider_history=rider_builder.build_current(),
            accepted_driver_history=driver_builder.build_current(),
            advance_rider=_advance(2.0, []),
            advance_driver=_advance(4.0, []),
            delta_time_ns=0.2,
            rider_initial_proper_step_ns=0.1,
            driver_initial_proper_step_ns=0.05,
            magnetic_dipole=MagneticDipoleConfig(),
            include_dipole_source=False,
        )


def test_second_half_trial_sees_midpoint_overlay_without_publishing_it() -> None:
    rider_builder = _accepted(-1.0)
    driver_builder = _accepted(1.0)
    accepted_rider = rider_builder.build_current()
    accepted_driver = driver_builder.build_current()
    midpoint = solve_exact_pair_slab_trial(
        accepted_rider_history=accepted_rider,
        accepted_driver_history=accepted_driver,
        advance_rider=_advance(2.0, []),
        advance_driver=_advance(4.0, []),
        delta_time_ns=0.1,
        rider_initial_proper_step_ns=0.05,
        driver_initial_proper_step_ns=0.025,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
    )
    rider_seen: list[object] = []
    driver_seen: list[object] = []

    endpoint = solve_exact_pair_slab_trial(
        accepted_rider_history=accepted_rider,
        accepted_driver_history=accepted_driver,
        rider_prior_tail=(midpoint.pair.rider.state,),
        driver_prior_tail=(midpoint.pair.driver.state,),
        advance_rider=_advance(2.0, rider_seen),
        advance_driver=_advance(4.0, driver_seen),
        delta_time_ns=0.1,
        rider_initial_proper_step_ns=0.05,
        driver_initial_proper_step_ns=0.025,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
    )

    assert endpoint.pair.target_time_ns == pytest.approx(0.2)
    assert endpoint.rider_history.n_steps == 3
    assert endpoint.driver_history.n_steps == 3
    assert all(isinstance(history, TrialTrajectoryHistory) for history in rider_seen)
    assert all(isinstance(history, TrialTrajectoryHistory) for history in driver_seen)
    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1


def test_failed_driver_trial_leaves_both_accepted_histories_unchanged() -> None:
    rider_builder = _accepted(-1.0)
    driver_builder = _accepted(1.0)
    accepted_rider = rider_builder.build_current()
    accepted_driver = driver_builder.build_current()

    def fail(*_args: object) -> dict[str, np.ndarray]:
        raise RuntimeError("driver trial failure")

    with pytest.raises(SharedLabTimeError, match="driver trial failed"):
        solve_exact_pair_slab_trial(
            accepted_rider_history=accepted_rider,
            accepted_driver_history=accepted_driver,
            advance_rider=_advance(2.0, []),
            advance_driver=fail,
            delta_time_ns=0.2,
            rider_initial_proper_step_ns=0.1,
            driver_initial_proper_step_ns=0.1,
            magnetic_dipole=MagneticDipoleConfig(),
            include_dipole_source=False,
        )

    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1
    assert accepted_rider.n_steps == 1
    assert accepted_driver.n_steps == 1


def test_eom_adapter_forwards_trial_history_and_causal_spin_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.self_consistency as self_consistency_module

    received: dict[str, object] = {}

    def fake_self_consistent_step(*args: object, **kwargs: object):
        received["args"] = args
        received.update(kwargs)
        return copy.deepcopy(args[2][0])  # type: ignore[index]

    monkeypatch.setattr(
        self_consistency_module,
        "self_consistent_step",
        fake_self_consistent_step,
    )
    magnetic = MagneticDipoleConfig(enabled=True)
    options = ExactPairEOMOptions(
        aperture_radius_mm=1.0,
        magnetic_dipole=magnetic,
        self_consistency=SelfConsistencyConfig.standard(),
        radiation_reaction_mode="medina_lad",
        step_idx=7,
    )
    callback = make_exact_role_eom_advance(options)
    accepted = _accepted(1.0).build_current()
    observer = _state(0.0, -1.0)
    source = _state(0.0, 1.0)

    callback(0.01, observer, source, accepted)

    args = received["args"]
    assert args[7] is options.self_consistency  # type: ignore[index]
    assert args[8] is ChronoMatchingMode.FAST  # type: ignore[index]
    assert args[9] is StartupMode.INERTIAL_PREHISTORY  # type: ignore[index]
    assert received["exact_source_history"] is accepted
    assert received["exact_source_spin_interpolation_model"] == "causal_frozen_c1"
    assert received["radiation_reaction_mode"] == "medina_lad"
    assert received["magnetic_dipole"] is magnetic


def test_eom_adapter_rejects_variable_geometry() -> None:
    with pytest.raises(ValueError, match="fixed_geometry"):
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=MagneticDipoleConfig(enabled=True),
            self_consistency=SelfConsistencyConfig.variable_geometry(),
        )


def test_real_neutral_eom_trial_lands_shared_time_without_publication() -> None:
    rider_builder = _coasting_history(-1.0, 2.0)
    driver_builder = _coasting_history(1.0, 1.25)
    magnetic = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=False,
        stern_gerlach_force_enabled=False,
    )
    advance = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=10.0,
            magnetic_dipole=magnetic,
            self_consistency=SelfConsistencyConfig.standard(),
        )
    )

    trial = solve_exact_pair_slab_trial(
        accepted_rider_history=rider_builder.build_current(),
        accepted_driver_history=driver_builder.build_current(),
        advance_rider=advance,
        advance_driver=advance,
        delta_time_ns=0.01,
        rider_initial_proper_step_ns=0.005,
        driver_initial_proper_step_ns=0.008,
        magnetic_dipole=magnetic,
        include_dipole_source=False,
    )

    assert trial.pair.rider.proper_step_ns == pytest.approx(0.005)
    assert trial.pair.driver.proper_step_ns == pytest.approx(0.008)
    assert trial.pair.rider.coordinate_time_ns == pytest.approx(0.01)
    assert trial.pair.driver.coordinate_time_ns == pytest.approx(0.01)
    assert rider_builder.accepted_steps == 2
    assert driver_builder.accepted_steps == 2


def test_step_doubling_keeps_both_paths_unpublished_and_sums_half_diagnostics() -> None:
    rider_builder = _accepted(-1.0)
    driver_builder = _accepted(1.0)
    trial = solve_exact_pair_step_doubling_trial(
        accepted_rider_history=rider_builder.build_current(),
        accepted_driver_history=driver_builder.build_current(),
        advance_rider=_advance(2.0, []),
        advance_driver=_advance(4.0, []),
        delta_time_ns=0.2,
        rider_initial_proper_step_ns=0.1,
        driver_initial_proper_step_ns=0.05,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        tolerances=StepDoublingTolerances(
            position_mm=ErrorScale(1.0, 0.0),
            mechanical_momentum_native=ErrorScale(1.0, 0.0),
            rest_spin=ErrorScale(1.0, 0.0),
            diagnostics_native=ErrorScale(1.0e-4, 0.0),
        ),
    )

    assert not trial.assessment.accepted
    assert trial.assessment.diagnostics_error > 1.0
    assert trial.midpoint.pair.target_time_ns == pytest.approx(0.1)
    assert trial.refined.pair.target_time_ns == pytest.approx(0.2)
    assert trial.refined.rider_history.n_steps == 3
    assert trial.refined.driver_history.n_steps == 3
    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1
    with pytest.raises(SharedLabTimeError, match="rejected"):
        commit_accepted_exact_pair_step_doubling_trial(
            trial,
            rider_builder=rider_builder,
            driver_builder=driver_builder,
        )
    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1


def test_accepted_step_doubling_commits_only_the_two_half_path() -> None:
    rider_builder = _accepted(-1.0)
    driver_builder = _accepted(1.0)
    trial = solve_exact_pair_step_doubling_trial(
        accepted_rider_history=rider_builder.build_current(),
        accepted_driver_history=driver_builder.build_current(),
        advance_rider=_advance(2.0, []),
        advance_driver=_advance(4.0, []),
        delta_time_ns=0.2,
        rider_initial_proper_step_ns=0.1,
        driver_initial_proper_step_ns=0.05,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        tolerances=StepDoublingTolerances(
            position_mm=ErrorScale(1.0, 0.0),
            mechanical_momentum_native=ErrorScale(1.0, 0.0),
            rest_spin=ErrorScale(1.0, 0.0),
            diagnostics_native=ErrorScale(1.0, 0.0),
        ),
    )
    assert trial.assessment.accepted

    rows = commit_accepted_exact_pair_step_doubling_trial(
        trial,
        rider_builder=rider_builder,
        driver_builder=driver_builder,
    )

    assert rows == (1, 2)
    assert rider_builder.accepted_steps == 3
    assert driver_builder.accepted_steps == 3
    rider = rider_builder.build_current()
    driver = driver_builder.build_current()
    np.testing.assert_allclose(rider.t[:, 0], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(driver.t[:, 0], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(rider.radiation_energy[1:, 0], [0.0025, 0.0025])
    np.testing.assert_allclose(driver.radiation_energy[1:, 0], [0.000625, 0.000625])


def test_two_row_preflight_failure_publishes_neither_midpoint_nor_endpoint() -> None:
    rider_builder = _accepted(-1.0)
    driver_builder = _accepted(1.0)
    trial = solve_exact_pair_step_doubling_trial(
        accepted_rider_history=rider_builder.build_current(),
        accepted_driver_history=driver_builder.build_current(),
        advance_rider=_advance(2.0, []),
        advance_driver=_advance(4.0, []),
        delta_time_ns=0.2,
        rider_initial_proper_step_ns=0.1,
        driver_initial_proper_step_ns=0.05,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        tolerances=StepDoublingTolerances(
            position_mm=ErrorScale(1.0, 0.0),
            mechanical_momentum_native=ErrorScale(1.0, 0.0),
            rest_spin=ErrorScale(1.0, 0.0),
            diagnostics_native=ErrorScale(1.0, 0.0),
        ),
    )
    trial.refined.pair.rider.state["x"][0] = np.nan

    with pytest.raises(ValueError, match="x must contain only finite"):
        commit_accepted_exact_pair_step_doubling_trial(
            trial,
            rider_builder=rider_builder,
            driver_builder=driver_builder,
        )

    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1


def test_health_gate_blocks_commit_even_when_error_norm_accepts() -> None:
    rider_builder = _accepted(-1.0)
    driver_builder = _accepted(1.0)
    nominal = _advance(2.0, [])

    def capped(*args: object) -> dict[str, np.ndarray]:
        state = nominal(*args)  # type: ignore[arg-type]
        state["medina_impulse_capped"] = np.array([True])
        return state

    trial = solve_exact_pair_step_doubling_trial(
        accepted_rider_history=rider_builder.build_current(),
        accepted_driver_history=driver_builder.build_current(),
        advance_rider=capped,
        advance_driver=_advance(4.0, []),
        delta_time_ns=0.2,
        rider_initial_proper_step_ns=0.1,
        driver_initial_proper_step_ns=0.05,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        tolerances=StepDoublingTolerances(
            position_mm=ErrorScale(1.0, 0.0),
            mechanical_momentum_native=ErrorScale(1.0, 0.0),
            rest_spin=ErrorScale(1.0, 0.0),
            diagnostics_native=ErrorScale(1.0, 0.0),
        ),
    )

    assert trial.assessment.accepted
    assert not trial.accepted
    assert any("Medina impulse cap" in failure for failure in trial.health_failures)
    with pytest.raises(SharedLabTimeError, match="Medina impulse cap"):
        commit_accepted_exact_pair_step_doubling_trial(
            trial,
            rider_builder=rider_builder,
            driver_builder=driver_builder,
        )
    assert rider_builder.accepted_steps == 1
    assert driver_builder.accepted_steps == 1


@pytest.mark.parametrize(
    ("radiation_reaction_mode", "include_dipole_source"),
    [("off", False), ("medina_lad", False), ("medina_lad", True)],
)
def test_charged_exact_rfs_step_doubling_uses_trial_history_without_commit(
    radiation_reaction_mode: str,
    include_dipole_source: bool,
) -> None:
    rider_builder, driver_builder, magnetic = _charged_accepted_pair(
        include_dipole_source=include_dipole_source
    )
    accepted_count = rider_builder.accepted_steps
    advance = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=magnetic,
            self_consistency=SelfConsistencyConfig.standard(),
            radiation_reaction_mode=radiation_reaction_mode,
        )
    )
    loose = StepDoublingTolerances(
        position_mm=ErrorScale(1.0, 1.0),
        mechanical_momentum_native=ErrorScale(1.0, 1.0),
        rest_spin=ErrorScale(1.0, 1.0),
        diagnostics_native=ErrorScale(1.0, 1.0),
    )

    trial = solve_exact_pair_step_doubling_trial(
        accepted_rider_history=rider_builder.build_current(),
        accepted_driver_history=driver_builder.build_current(),
        advance_rider=advance,
        advance_driver=advance,
        delta_time_ns=1.0e-8,
        rider_initial_proper_step_ns=1.0e-8,
        driver_initial_proper_step_ns=1.0e-8,
        magnetic_dipole=magnetic,
        include_dipole_source=include_dipole_source,
        tolerances=loose,
    )

    assert trial.accepted
    assert not trial.health_failures
    assert rider_builder.accepted_steps == accepted_count
    assert driver_builder.accepted_steps == accepted_count
    assert np.all(np.isfinite(trial.refined.pair.rider.state["Px"]))
    assert np.all(np.isfinite(trial.refined.pair.driver.state["Px"]))
    assert trial.refined.rider_history.n_steps == accepted_count + 2
    assert trial.refined.driver_history.n_steps == accepted_count + 2
    if radiation_reaction_mode == "medina_lad":
        for endpoint in (trial.full.pair, trial.midpoint.pair):
            assert not bool(endpoint.rider.state["medina_force_derivative_ready"][0])
            assert not bool(endpoint.driver.state["medina_force_derivative_ready"][0])
        assert bool(trial.refined.pair.rider.state["medina_force_derivative_ready"][0])
        assert bool(trial.refined.pair.driver.state["medina_force_derivative_ready"][0])


def test_intrinsic_spin_diagnostic_off_never_calls_reduction_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.spin_self_force_reduction_oracle as reduction_oracle

    rider_builder, driver_builder, magnetic = _charged_accepted_pair(
        exact_retarded_update="second_order_start_taylor_endpoint",
        intrinsic_spin_self_reaction_mode="off",
    )

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("feature-off path evaluated intrinsic-spin reduction")

    monkeypatch.setattr(
        reduction_oracle,
        "evaluate_retarded_potential_intrinsic_spin_reduction_native",
        fail_if_called,
    )
    advance = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=magnetic,
            self_consistency=SelfConsistencyConfig.standard(),
            radiation_reaction_mode="medina_lad",
        )
    )

    trial = solve_exact_pair_slab_trial(
        accepted_rider_history=rider_builder.build_current(),
        accepted_driver_history=driver_builder.build_current(),
        advance_rider=advance,
        advance_driver=advance,
        delta_time_ns=1.0e-8,
        rider_initial_proper_step_ns=1.0e-8,
        driver_initial_proper_step_ns=1.0e-8,
        magnetic_dipole=magnetic,
        include_dipole_source=False,
    )

    assert np.all(np.isfinite(trial.pair.rider.state["Px"]))
    assert np.all(np.isfinite(trial.pair.driver.state["Px"]))
    assert "_intrinsic_spin_start_analytical_reduction" not in trial.pair.rider.state
    assert "_intrinsic_spin_start_analytical_reduction" not in trial.pair.driver.state


def test_short_adaptive_window_runs_charged_rfs_medina_and_dipole_source() -> None:
    rider_builder, driver_builder, magnetic = _charged_accepted_pair(
        include_dipole_source=True,
        exact_retarded_update="second_order_start_taylor_endpoint",
        intrinsic_spin_self_reaction_mode="diagnostic",
    )
    start_time = float(rider_builder.build_current().t[-1, 0])
    advance = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=magnetic,
            self_consistency=SelfConsistencyConfig.standard(),
            radiation_reaction_mode="medina_lad",
        )
    )
    loose = StepDoublingTolerances(
        position_mm=ErrorScale(1.0, 1.0),
        mechanical_momentum_native=ErrorScale(1.0, 1.0),
        rest_spin=ErrorScale(1.0, 1.0),
        diagnostics_native=ErrorScale(1.0, 1.0),
    )

    result = run_exact_pair_adaptive_window(
        rider_builder=rider_builder,
        driver_builder=driver_builder,
        advance_rider=advance,
        advance_driver=advance,
        controller_state=AdaptivePairControllerState(
            current_step_ns=1.0e-8,
            rider_proper_step_guess_ns=1.0e-8,
            driver_proper_step_guess_ns=1.0e-8,
        ),
        controller_config=StepControllerConfig(method_order=1),
        tolerances=loose,
        target_time_ns=start_time + 2.0e-8,
        minimum_step_ns=1.0e-12,
        maximum_step_ns=1.0e-8,
        maximum_attempts=4,
        maximum_accepted_slabs=2,
        public_sample_interval_ns=1.5e-8,
        magnetic_dipole=magnetic,
        include_dipole_source=True,
        intrinsic_spin_reduction_history=(
            AcceptedPairIntrinsicSpinReductionHistory.empty()
        ),
        build_intrinsic_spin_reduction_candidate=(
            build_accepted_pair_intrinsic_spin_reduction_diagnostic_candidate
        ),
    )

    assert result.completed
    assert result.accepted_slabs == 2
    assert result.rejected_trials == 0
    rider = rider_builder.build_current()
    driver = driver_builder.build_current()
    np.testing.assert_array_equal(rider.t, driver.t)
    assert bool(rider.medina_force_derivative_ready[-1, 0])
    assert bool(driver.medina_force_derivative_ready[-1, 0])
    assert not np.any(rider.medina_impulse_capped)
    assert not np.any(driver.medina_impulse_capped)
    assert result.public_output_state.selected_rows[-1] == rider.n_steps - 1
    assert result.intrinsic_spin_reduction_history is not None
    reduction = result.intrinsic_spin_reduction_history
    assert reduction.rider.sample_count == 4
    assert reduction.driver.sample_count == 4
    for trace in (reduction.rider_diagnostics, reduction.driver_diagnostics):
        assert trace.total_records == 4
        assert trace.total_records == (
            trace.analytical_records + trace.causal_records + trace.unavailable_records
        )
        assert len(trace.records) == 4
        assert all(
            record.route
            in {
                "analytical_smooth_segment",
                "unavailable_insufficient_accepted_history",
            }
            for record in trace.records
        )
        assert all(
            record.linear_spin_four_force_native is not None
            for record in trace.records
            if record.route == "analytical_smooth_segment"
        )
    for history in (reduction.rider, reduction.driver):
        assert np.all(np.isfinite(history.four_velocity_mm_ns))
        assert np.all(np.isfinite(history.non_self_four_acceleration_mm_ns2))
        assert np.all(np.isfinite(history.physical_spin_four_native))
        for velocity, acceleration, spin in zip(
            history.four_velocity_mm_ns,
            history.non_self_four_acceleration_mm_ns2,
            history.physical_spin_four_native,
        ):
            acceleration_scale = max(
                float(np.linalg.norm(velocity) * np.linalg.norm(acceleration)),
                1.0,
            )
            spin_scale = max(
                float(np.linalg.norm(velocity) * np.linalg.norm(spin)),
                1.0,
            )
            assert abs(minkowski_dot(velocity, acceleration)) <= (
                2.0e-12 * acceleration_scale
            )
            assert abs(minkowski_dot(velocity, spin)) <= 2.0e-12 * spin_scale


def test_real_neutral_eom_step_doubling_accepts_identical_coasting_paths() -> None:
    rider_builder = _coasting_history(-1.0, 2.0)
    driver_builder = _coasting_history(1.0, 1.25)
    magnetic = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=False,
        stern_gerlach_force_enabled=False,
    )
    advance = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=10.0,
            magnetic_dipole=magnetic,
            self_consistency=SelfConsistencyConfig.standard(),
        )
    )
    tolerances = StepDoublingTolerances(
        position_mm=ErrorScale(1.0e-12, 1.0e-12),
        mechanical_momentum_native=ErrorScale(1.0e-12, 1.0e-12),
        rest_spin=ErrorScale(1.0e-12, 1.0e-12),
        diagnostics_native=ErrorScale(1.0e-12, 1.0e-12),
    )

    trial = solve_exact_pair_step_doubling_trial(
        accepted_rider_history=rider_builder.build_current(),
        accepted_driver_history=driver_builder.build_current(),
        advance_rider=advance,
        advance_driver=advance,
        delta_time_ns=0.01,
        rider_initial_proper_step_ns=0.005,
        driver_initial_proper_step_ns=0.008,
        magnetic_dipole=magnetic,
        include_dipole_source=False,
        tolerances=tolerances,
    )

    assert trial.assessment.accepted
    assert trial.assessment.normalized_error <= 1.0e-2
    assert trial.refined.pair.target_time_ns == pytest.approx(0.01)
    assert rider_builder.accepted_steps == 2
    assert driver_builder.accepted_steps == 2
