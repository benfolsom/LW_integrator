from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from core.adaptive_pair_return import (
    AdaptivePairControllerState,
    AdaptivePairPublicOutputState,
    attempt_exact_pair_adaptive_step,
    run_exact_pair_adaptive_window,
)
from core.causal_c5_dipole_provider import (
    AcceptedPairCausalC5SourceHistory,
    GrowableAcceptedPairCausalC5SourceHistory,
    evaluate_causal_c5_dipole_source_collection_native,
)
from core.causal_c5_source_history import CausalC5HistoryUnavailableError
from core.constants import C_MMNS
from core.integration_checkpoint import AcceptedPairCheckpointStore
from core.integration_runner import IntegrationCancelled
from core.retarded_fields import ObserverEvent
from core.shared_lab_time import SharedLabTimeError
from core.spin_self_force_reduction_history import (
    AcceptedPairIntrinsicSpinReductionHistory,
    build_accepted_pair_intrinsic_spin_reduction_candidate,
    build_accepted_pair_intrinsic_spin_reduction_diagnostic_candidate,
)
from core.step_doubling import (
    ErrorScale,
    StepControllerConfig,
    StepDoublingTolerances,
)
from core.types import GrowableTrajectoryBuilder, MagneticDipoleConfig


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
    }


def _pair() -> tuple[GrowableTrajectoryBuilder, GrowableTrajectoryBuilder]:
    rider = GrowableTrajectoryBuilder(1, 1)
    driver = GrowableTrajectoryBuilder(1, 1)
    rider.append_step(_state(0.0, -1.0))
    driver.append_step(_state(0.0, 1.0))
    return rider, driver


def _magnetic_pair() -> tuple[GrowableTrajectoryBuilder, GrowableTrajectoryBuilder]:
    rider = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
    driver = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
    for builder, position, moment in (
        (rider, -1.0, 2.0e-6),
        (driver, 1.0, -3.0e-6),
    ):
        state = _state(0.0, position)
        state.update(
            {
                "spin_x": np.array([0.0]),
                "spin_y": np.array([0.0]),
                "spin_z": np.array([1.0]),
                "magnetic_moment_native": np.array([moment]),
                "magnetic_dipole_active": np.array([1.0]),
            }
        )
        builder.append_step(state)
    return rider, driver


def _magnetic_history_pair(
    count: int,
) -> tuple[GrowableTrajectoryBuilder, GrowableTrajectoryBuilder]:
    rider = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
    driver = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
    for step in range(count):
        time = 0.01 * step
        angle = 0.02 * time
        for builder, position, moment in (
            (rider, -1.0, 2.0e-6),
            (driver, 1.0, -3.0e-6),
        ):
            state = _state(time, position)
            state.update(
                {
                    "spin_x": np.array([np.sin(angle)]),
                    "spin_y": np.array([0.0]),
                    "spin_z": np.array([np.cos(angle)]),
                    "magnetic_moment_native": np.array([moment]),
                    "magnetic_dipole_active": np.array([1.0]),
                }
            )
            builder.append_step(state)
    return rider, driver


def _advance(scale: float, *, cap: bool = False, dead: bool = False):
    def advance(
        proper_step_ns: float,
        observer_start: dict[str, np.ndarray],
        _source_start: dict[str, np.ndarray],
        _exact_source_history: object,
    ) -> dict[str, np.ndarray]:
        result = copy.deepcopy(observer_start)
        result["t"] = np.array([float(observer_start["t"][0]) + scale * proper_step_ns])
        result["x"] = np.array([float(observer_start["x"][0]) + proper_step_ns])
        result["spin_x"] = np.array([0.0])
        result["spin_y"] = np.array([0.0])
        result["spin_z"] = np.array([1.0])
        result["radiation_energy"] = np.array([proper_step_ns**2])
        result["radiation_reaction_work"] = np.array([0.0])
        result["medina_cross_field_energy_change"] = np.array([0.0])
        result["mass_shell_projection_energy"] = np.array([0.0])
        result["medina_impulse_capped"] = np.array([cap])
        result["_dead_particles"] = np.array([dead])
        result["_exact_source_start_four_potential"] = np.zeros((1, 4))
        result["_exact_source_endpoint_rebase_required"] = np.array([False])
        result["_intrinsic_spin_start_four_velocity"] = np.array(
            [[1.0, float(observer_start["x"][0]), 0.0, 0.0]]
        )
        result["_intrinsic_spin_start_non_self_four_acceleration"] = np.array(
            [[0.0, proper_step_ns, 0.0, 0.0]]
        )
        result["_intrinsic_spin_start_physical_four_spin"] = np.array(
            [[0.0, 0.0, 0.0, 1.0]]
        )
        return result

    return advance


def _diagnostic_unavailable_advance(scale: float):
    ordinary = _advance(scale)

    def advance(*args, **kwargs):
        result = ordinary(*args, **kwargs)
        result["_intrinsic_spin_start_four_velocity"] = np.array(
            [[C_MMNS, 0.0, 0.0, 0.0]]
        )
        result["_intrinsic_spin_start_non_self_four_acceleration"] = np.zeros((1, 4))
        result["_intrinsic_spin_start_physical_four_spin"] = np.array(
            [[0.0, 0.0, 0.0, 1.0]]
        )
        result["_intrinsic_spin_start_analytical_reduction"] = [None]
        result["_intrinsic_spin_start_analytical_unavailable_reason"] = [
            "spin segment boundary"
        ]
        result["_intrinsic_spin_charge_native"] = np.array([1.0])
        result["_intrinsic_spin_mass_amu"] = np.array([1.0])
        result["_intrinsic_spin_g_factor"] = np.array([2.0])
        return result

    return advance


def _tolerances(diagnostic_absolute: float) -> StepDoublingTolerances:
    return StepDoublingTolerances(
        position_mm=ErrorScale(1.0, 0.0),
        mechanical_momentum_native=ErrorScale(1.0, 0.0),
        rest_spin=ErrorScale(1.0, 0.0),
        diagnostics_native=ErrorScale(diagnostic_absolute, 0.0),
    )


def _controller() -> AdaptivePairControllerState:
    return AdaptivePairControllerState(
        current_step_ns=0.2,
        rider_proper_step_guess_ns=0.1,
        driver_proper_step_guess_ns=0.05,
    )


def _initial_spin_reduction_history() -> AcceptedPairIntrinsicSpinReductionHistory:
    return AcceptedPairIntrinsicSpinReductionHistory.empty()


_build_spin_reduction_candidate = build_accepted_pair_intrinsic_spin_reduction_candidate


def _attempt(
    rider: GrowableTrajectoryBuilder,
    driver: GrowableTrajectoryBuilder,
    *,
    rider_advance,
    tolerances: StepDoublingTolerances,
    controller_state: AdaptivePairControllerState | None = None,
    intrinsic_spin_reduction_history=None,
    build_intrinsic_spin_reduction_candidate=None,
    causal_c5_source_history=None,
    growable_causal_c5_source_history=None,
):
    return attempt_exact_pair_adaptive_step(
        rider_builder=rider,
        driver_builder=driver,
        advance_rider=rider_advance,
        advance_driver=_advance(4.0),
        controller_state=controller_state or _controller(),
        controller_config=StepControllerConfig(method_order=1),
        tolerances=tolerances,
        minimum_step_ns=0.001,
        maximum_step_ns=1.0,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        intrinsic_spin_reduction_history=intrinsic_spin_reduction_history,
        build_intrinsic_spin_reduction_candidate=(
            build_intrinsic_spin_reduction_candidate
        ),
        causal_c5_source_history=causal_c5_source_history,
        growable_causal_c5_source_history=growable_causal_c5_source_history,
    )


def test_accepted_attempt_commits_refined_rows_and_advances_controller() -> None:
    rider, driver = _pair()

    attempt = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0),
    )

    assert attempt.accepted
    assert attempt.committed_rows == (1, 2)
    assert rider.accepted_steps == 3
    assert driver.accepted_steps == 3
    assert attempt.controller_state.accepted_slabs == 1
    assert attempt.controller_state.rejected_trials == 0


def test_accepted_attempt_adopts_spin_history_after_joint_preflight() -> None:
    rider, driver = _pair()
    accepted = _initial_spin_reduction_history()
    before = accepted.to_checkpoint_payload()

    attempt = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0),
        intrinsic_spin_reduction_history=accepted,
        build_intrinsic_spin_reduction_candidate=_build_spin_reduction_candidate,
    )

    assert attempt.accepted
    assert accepted.to_checkpoint_payload() == before
    assert attempt.intrinsic_spin_reduction_history is not None
    assert attempt.intrinsic_spin_reduction_history.rider.sample_count == 2
    assert attempt.intrinsic_spin_reduction_history.driver.sample_count == 2
    assert rider.accepted_steps == driver.accepted_steps == 3


def test_rejected_attempt_never_builds_or_adopts_spin_history() -> None:
    rider, driver = _pair()
    accepted = _initial_spin_reduction_history()
    calls = []

    def unexpected_candidate(*args):
        calls.append(args)
        return _build_spin_reduction_candidate(*args)

    attempt = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0e-4),
        intrinsic_spin_reduction_history=accepted,
        build_intrinsic_spin_reduction_candidate=unexpected_candidate,
    )

    assert not attempt.accepted
    assert calls == []
    assert attempt.intrinsic_spin_reduction_history is accepted
    assert rider.accepted_steps == driver.accepted_steps == 1


def test_causal_c5_history_advances_only_after_joint_acceptance() -> None:
    rider, driver = _magnetic_pair()
    accepted = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        rider.build_current(),
        driver.build_current(),
    )

    accepted_attempt = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0),
        causal_c5_source_history=accepted,
    )

    assert accepted_attempt.accepted
    assert accepted.rider.sources[0].history.sample_count == 1
    assert accepted_attempt.causal_c5_source_history is not None
    assert (
        accepted_attempt.causal_c5_source_history.rider.sources[0].history.sample_count
        == 3
    )
    assert (
        accepted_attempt.causal_c5_source_history.driver.sources[0].history.sample_count
        == 3
    )


def test_rejected_attempt_does_not_touch_causal_c5_history() -> None:
    rider, driver = _magnetic_pair()
    accepted = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        rider.build_current(),
        driver.build_current(),
    )

    attempt = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0e-4),
        causal_c5_source_history=accepted,
    )

    assert not attempt.accepted
    assert attempt.causal_c5_source_history is accepted
    assert rider.accepted_steps == driver.accepted_steps == 1


def test_growable_causal_c5_attempt_commits_only_after_acceptance() -> None:
    rider, driver = _magnetic_pair()
    growable = GrowableAcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        rider.build_current(),
        driver.build_current(),
    )

    rejected = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0e-4),
        growable_causal_c5_source_history=growable,
    )
    assert not rejected.accepted
    assert growable.build_current().rider.sources[0].history.sample_count == 1

    accepted = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0),
        growable_causal_c5_source_history=growable,
    )
    assert accepted.accepted
    assert accepted.causal_c5_source_history is not None
    assert accepted.causal_c5_source_history.rider.sources[0].history.sample_count == 3
    assert growable.build_current().driver.sources[0].history.sample_count == 3


def test_immutable_and_growable_causal_c5_inputs_are_mutually_exclusive() -> None:
    rider, driver = _magnetic_pair()
    immutable = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        rider.build_current(),
        driver.build_current(),
    )
    growable = GrowableAcceptedPairCausalC5SourceHistory.from_accepted(immutable)

    with pytest.raises(ValueError, match="cannot both be supplied"):
        _attempt(
            rider,
            driver,
            rider_advance=_advance(2.0),
            tolerances=_tolerances(1.0),
            causal_c5_source_history=immutable,
            growable_causal_c5_source_history=growable,
        )


def test_causal_c5_candidate_failure_precedes_pair_publication() -> None:
    rider, driver = _magnetic_history_pair(18)
    accepted = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        rider.build_current(),
        driver.build_current(),
    )

    with pytest.raises(
        CausalC5HistoryUnavailableError,
        match="condition-number limit",
    ):
        _attempt(
            rider,
            driver,
            rider_advance=_advance(2.0),
            tolerances=_tolerances(1.0),
            causal_c5_source_history=accepted,
        )

    assert rider.accepted_steps == driver.accepted_steps == 18
    assert accepted.rider.sources[0].history.sample_count == 18


def test_spin_history_preflight_failure_leaves_pair_unpublished() -> None:
    rider, driver = _pair()

    def fail_candidate(*_args):
        raise RuntimeError("diagnostic sample construction failed")

    with pytest.raises(RuntimeError, match="sample construction"):
        _attempt(
            rider,
            driver,
            rider_advance=_advance(2.0),
            tolerances=_tolerances(1.0),
            intrinsic_spin_reduction_history=_initial_spin_reduction_history(),
            build_intrinsic_spin_reduction_candidate=fail_candidate,
        )

    assert rider.accepted_steps == driver.accepted_steps == 1


def test_error_rejection_shrinks_without_publishing() -> None:
    rider, driver = _pair()

    attempt = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0e-4),
    )

    assert not attempt.accepted
    assert attempt.controller_state.current_step_ns < 0.2
    assert attempt.controller_state.rejected_trials == 1
    assert rider.accepted_steps == 1
    assert driver.accepted_steps == 1


def test_medina_cap_is_retried_at_smaller_step_without_publication() -> None:
    rider, driver = _pair()

    attempt = _attempt(
        rider,
        driver,
        rider_advance=_advance(2.0, cap=True),
        tolerances=_tolerances(1.0),
    )

    assert not attempt.accepted
    assert attempt.trial.assessment.accepted
    assert attempt.controller_state.current_step_ns < 0.2
    assert rider.accepted_steps == 1
    assert driver.accepted_steps == 1


def test_particle_death_is_fatal_and_never_published() -> None:
    rider, driver = _pair()

    with pytest.raises(SharedLabTimeError, match="particle death"):
        _attempt(
            rider,
            driver,
            rider_advance=_advance(2.0, dead=True),
            tolerances=_tolerances(1.0),
        )

    assert rider.accepted_steps == 1
    assert driver.accepted_steps == 1


def test_checkpoint_restore_reproduces_next_adaptive_attempt_bitwise(
    tmp_path: Path,
) -> None:
    continuous_rider, continuous_driver = _pair()
    first = _attempt(
        continuous_rider,
        continuous_driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0),
    )
    checkpoint = AcceptedPairCheckpointStore(
        tmp_path / "pair.checkpoint",
        compatibility_payload={"physics": "adaptive-pair-test"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    checkpoint.write(
        rider=continuous_rider.build_current(),
        driver=continuous_driver.build_current(),
        controller_state=first.controller_state.to_checkpoint_state(),
        public_output_state={"cursor": 0},
    )

    continuous_second = _attempt(
        continuous_rider,
        continuous_driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0),
        controller_state=first.controller_state,
    )

    reopened = AcceptedPairCheckpointStore(
        tmp_path / "pair.checkpoint",
        compatibility_payload={"physics": "adaptive-pair-test"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=True,
    )
    restored_rider = GrowableTrajectoryBuilder(1, 1)
    restored_driver = GrowableTrajectoryBuilder(1, 1)
    reopened.restore_pair(restored_rider, restored_driver)
    restored_controller = AdaptivePairControllerState.from_checkpoint_state(
        reopened.controller_state
    )
    restored_second = _attempt(
        restored_rider,
        restored_driver,
        rider_advance=_advance(2.0),
        tolerances=_tolerances(1.0),
        controller_state=restored_controller,
    )

    assert restored_second.controller_state == continuous_second.controller_state
    for restored, continuous in (
        (restored_rider.build_current(), continuous_rider.build_current()),
        (restored_driver.build_current(), continuous_driver.build_current()),
    ):
        for name in (
            "x",
            "t",
            "Px",
            "Pt",
            "gamma",
            "radiation_energy",
            "mass_shell_projection_energy",
        ):
            np.testing.assert_array_equal(
                np.asarray(getattr(restored, name)),
                np.asarray(getattr(continuous, name)),
            )


def _run_window(
    *,
    public_interval_ns: float,
    diagnostic_absolute: float = 1.0,
    target_time_ns: float = 0.65,
    minimum_step_ns: float = 0.001,
    maximum_attempts: int = 20,
    intrinsic_spin_reduction_history=None,
    build_intrinsic_spin_reduction_candidate=None,
    rider_advance=None,
    driver_advance=None,
):
    rider, driver = _pair()
    result = run_exact_pair_adaptive_window(
        rider_builder=rider,
        driver_builder=driver,
        advance_rider=rider_advance or _advance(2.0),
        advance_driver=driver_advance or _advance(4.0),
        controller_state=_controller(),
        controller_config=StepControllerConfig(method_order=1),
        tolerances=_tolerances(diagnostic_absolute),
        target_time_ns=target_time_ns,
        minimum_step_ns=minimum_step_ns,
        maximum_step_ns=1.0,
        maximum_attempts=maximum_attempts,
        maximum_accepted_slabs=20,
        public_sample_interval_ns=public_interval_ns,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        intrinsic_spin_reduction_history=intrinsic_spin_reduction_history,
        build_intrinsic_spin_reduction_candidate=(
            build_intrinsic_spin_reduction_candidate
        ),
    )
    return rider, driver, result


def test_live_diagnostic_trace_switches_from_unavailable_to_causal() -> None:
    _rider, _driver, result = _run_window(
        public_interval_ns=0.15,
        intrinsic_spin_reduction_history=_initial_spin_reduction_history(),
        build_intrinsic_spin_reduction_candidate=(
            build_accepted_pair_intrinsic_spin_reduction_diagnostic_candidate
        ),
        rider_advance=_diagnostic_unavailable_advance(2.0),
        driver_advance=_diagnostic_unavailable_advance(4.0),
    )

    assert result.intrinsic_spin_reduction_history is not None
    for trace in (
        result.intrinsic_spin_reduction_history.rider_diagnostics,
        result.intrinsic_spin_reduction_history.driver_diagnostics,
    ):
        assert trace.total_records >= 6
        assert trace.unavailable_records == 5
        assert trace.causal_records == trace.total_records - 5
        assert trace.analytical_records == 0
        assert all(
            record.causal_condition_number is not None
            and np.isfinite(record.causal_condition_number)
            for record in trace.records[5:]
        )


def test_public_output_cadence_does_not_change_accepted_dynamics() -> None:
    dense_rider, dense_driver, dense = _run_window(public_interval_ns=0.15)
    sparse_rider, sparse_driver, sparse = _run_window(public_interval_ns=0.5)

    assert dense.completed and sparse.completed
    assert dense.final_time_ns == pytest.approx(0.65, abs=1.0e-15)
    assert dense.controller_state == sparse.controller_state
    assert dense.attempts == sparse.attempts
    assert dense.public_output_state.selected_rows != (
        sparse.public_output_state.selected_rows
    )
    assert dense.public_output_state.selected_rows[-1] == dense_rider.accepted_steps - 1
    assert sparse.public_output_state.selected_rows[-1] == (
        sparse_rider.accepted_steps - 1
    )
    for dense_history, sparse_history in (
        (dense_rider.build_current(), sparse_rider.build_current()),
        (dense_driver.build_current(), sparse_driver.build_current()),
    ):
        for name in ("x", "t", "radiation_energy", "spin_z"):
            np.testing.assert_array_equal(
                np.asarray(getattr(dense_history, name)),
                np.asarray(getattr(sparse_history, name)),
            )


def test_rejected_trials_publish_neither_history_nor_output_rows() -> None:
    rider, driver, result = _run_window(
        public_interval_ns=0.1,
        diagnostic_absolute=1.0e-3,
        target_time_ns=0.2,
    )

    assert result.completed
    assert result.rejected_trials > 0
    assert rider.accepted_steps == 1 + 2 * result.accepted_slabs
    assert driver.accepted_steps == rider.accepted_steps
    assert max(result.public_output_state.selected_rows) < rider.accepted_steps


def test_optional_attempt_diagnostics_record_each_controller_component() -> None:
    rider, driver = _pair()
    result = run_exact_pair_adaptive_window(
        rider_builder=rider,
        driver_builder=driver,
        advance_rider=_advance(2.0),
        advance_driver=_advance(4.0),
        controller_state=_controller(),
        controller_config=StepControllerConfig(method_order=1),
        tolerances=_tolerances(1.0),
        target_time_ns=0.2,
        minimum_step_ns=0.001,
        maximum_step_ns=1.0,
        maximum_attempts=10,
        maximum_accepted_slabs=10,
        public_sample_interval_ns=0.1,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        record_attempt_diagnostics=True,
    )

    assert len(result.attempt_diagnostics) == result.attempts
    record = result.attempt_diagnostics[0]
    assert record.attempted_step_ns == 0.2
    assert record.accepted
    assert record.normalized_error == max(
        record.position_error,
        record.mechanical_momentum_error,
        record.rest_spin_error,
        record.diagnostics_error,
    )
    assert record.mechanical_momentum_error_index == (0, 0)


def test_attempt_diagnostics_are_disabled_by_default() -> None:
    _, _, result = _run_window(
        public_interval_ns=0.1,
        target_time_ns=0.2,
    )

    assert result.attempt_diagnostics == ()


def test_window_fails_when_attempt_budget_is_exhausted() -> None:
    with pytest.raises(SharedLabTimeError, match="maximum trial attempts"):
        _run_window(
            public_interval_ns=0.1,
            diagnostic_absolute=1.0e-30,
            target_time_ns=0.2,
            maximum_attempts=1,
        )


def test_window_fails_on_irreducible_minimum_step_rejection() -> None:
    with pytest.raises(SharedLabTimeError, match="diagnostics=") as caught:
        _run_window(
            public_interval_ns=0.1,
            diagnostic_absolute=1.0e-30,
            target_time_ns=0.2,
            minimum_step_ns=0.2,
        )

    assert "position=" in str(caught.value)
    assert "momentum=" in str(caught.value)
    assert "spin=" in str(caught.value)


def test_window_resume_accepts_the_pair_commit_time_envelope() -> None:
    rider = GrowableTrajectoryBuilder(1, 1)
    driver = GrowableTrajectoryBuilder(1, 1)
    rider.append_step(_state(0.0, -1.0))
    driver.append_step(_state(1.5e-18, 1.0))

    result = run_exact_pair_adaptive_window(
        rider_builder=rider,
        driver_builder=driver,
        advance_rider=_advance(2.0),
        advance_driver=_advance(4.0),
        controller_state=_controller(),
        controller_config=StepControllerConfig(method_order=1),
        tolerances=_tolerances(1.0),
        target_time_ns=0.2,
        minimum_step_ns=0.001,
        maximum_step_ns=1.0,
        maximum_attempts=10,
        maximum_accepted_slabs=10,
        public_sample_interval_ns=0.1,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        absolute_time_tolerance_ns=1.0e-18,
        relative_time_tolerance=0.0,
    )

    assert result.completed


def test_window_writes_complete_checkpoint_with_public_cursor(tmp_path: Path) -> None:
    rider, driver = _pair()
    store = AcceptedPairCheckpointStore(
        tmp_path / "window.checkpoint",
        compatibility_payload={"physics": "adaptive-window-test"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    result = run_exact_pair_adaptive_window(
        rider_builder=rider,
        driver_builder=driver,
        advance_rider=_advance(2.0),
        advance_driver=_advance(4.0),
        controller_state=_controller(),
        controller_config=StepControllerConfig(method_order=1),
        tolerances=_tolerances(1.0),
        target_time_ns=0.2,
        minimum_step_ns=0.001,
        maximum_step_ns=1.0,
        maximum_attempts=10,
        maximum_accepted_slabs=10,
        public_sample_interval_ns=0.15,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        checkpoint_store=store,
        intrinsic_spin_reduction_history=_initial_spin_reduction_history(),
        build_intrinsic_spin_reduction_candidate=_build_spin_reduction_candidate,
    )

    reopened = AcceptedPairCheckpointStore(
        tmp_path / "window.checkpoint",
        compatibility_payload={"physics": "adaptive-window-test"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=True,
    )
    assert reopened.manifest["status"] == "complete"
    assert (
        AdaptivePairControllerState.from_checkpoint_state(reopened.controller_state)
        == result.controller_state
    )
    assert (
        AdaptivePairPublicOutputState.from_checkpoint_state(
            reopened.public_output_state
        )
        == result.public_output_state
    )
    assert reopened.intrinsic_spin_reduction_state is not None
    restored_reduction = (
        AcceptedPairIntrinsicSpinReductionHistory.from_checkpoint_payload(
            reopened.intrinsic_spin_reduction_state
        )
    )
    assert result.intrinsic_spin_reduction_history is not None
    assert (
        restored_reduction.to_checkpoint_payload()
        == result.intrinsic_spin_reduction_history.to_checkpoint_payload()
    )


def test_window_cancel_flushes_latest_joint_history(tmp_path: Path) -> None:
    rider, driver = _pair()
    directory = tmp_path / "cancel.checkpoint"
    store = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "adaptive-window-cancel"},
        interval_knots=100,
        interval_seconds=0.0,
        resume=False,
    )

    with pytest.raises(IntegrationCancelled):
        run_exact_pair_adaptive_window(
            rider_builder=rider,
            driver_builder=driver,
            advance_rider=_advance(2.0),
            advance_driver=_advance(4.0),
            controller_state=_controller(),
            controller_config=StepControllerConfig(method_order=1),
            tolerances=_tolerances(1.0),
            target_time_ns=0.2,
            minimum_step_ns=0.001,
            maximum_step_ns=1.0,
            maximum_attempts=10,
            maximum_accepted_slabs=10,
            public_sample_interval_ns=0.1,
            magnetic_dipole=MagneticDipoleConfig(),
            include_dipole_source=False,
            checkpoint_store=store,
            cancel_callback=lambda: True,
        )

    reopened = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "adaptive-window-cancel"},
        interval_knots=100,
        interval_seconds=0.0,
        resume=True,
    )
    assert reopened.manifest["status"] == "running"
    assert reopened.committed_knots == 1
    restored_rider = GrowableTrajectoryBuilder(1, 1)
    restored_driver = GrowableTrajectoryBuilder(1, 1)
    reopened.restore_pair(restored_rider, restored_driver)
    assert restored_rider.accepted_steps == restored_driver.accepted_steps == 1


def test_window_resume_reproduces_history_and_output_selection_bitwise(
    tmp_path: Path,
) -> None:
    continuous_rider, continuous_driver, continuous = _run_window(
        public_interval_ns=0.15,
        intrinsic_spin_reduction_history=_initial_spin_reduction_history(),
        build_intrinsic_spin_reduction_candidate=_build_spin_reduction_candidate,
    )
    interrupted_rider, interrupted_driver = _pair()
    directory = tmp_path / "resume-window.checkpoint"
    store = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "adaptive-window-resume-test"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    with pytest.raises(SharedLabTimeError, match="maximum accepted slabs"):
        run_exact_pair_adaptive_window(
            rider_builder=interrupted_rider,
            driver_builder=interrupted_driver,
            advance_rider=_advance(2.0),
            advance_driver=_advance(4.0),
            controller_state=_controller(),
            controller_config=StepControllerConfig(method_order=1),
            tolerances=_tolerances(1.0),
            target_time_ns=0.65,
            minimum_step_ns=0.001,
            maximum_step_ns=1.0,
            maximum_attempts=20,
            maximum_accepted_slabs=1,
            public_sample_interval_ns=0.15,
            magnetic_dipole=MagneticDipoleConfig(),
            include_dipole_source=False,
            checkpoint_store=store,
            intrinsic_spin_reduction_history=_initial_spin_reduction_history(),
            build_intrinsic_spin_reduction_candidate=(_build_spin_reduction_candidate),
        )

    reopened = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "adaptive-window-resume-test"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=True,
    )
    resumed_rider = GrowableTrajectoryBuilder(1, 1)
    resumed_driver = GrowableTrajectoryBuilder(1, 1)
    reopened.restore_pair(resumed_rider, resumed_driver)
    assert reopened.intrinsic_spin_reduction_state is not None
    restored_reduction = (
        AcceptedPairIntrinsicSpinReductionHistory.from_checkpoint_payload(
            reopened.intrinsic_spin_reduction_state
        )
    )
    resumed = run_exact_pair_adaptive_window(
        rider_builder=resumed_rider,
        driver_builder=resumed_driver,
        advance_rider=_advance(2.0),
        advance_driver=_advance(4.0),
        controller_state=AdaptivePairControllerState.from_checkpoint_state(
            reopened.controller_state
        ),
        controller_config=StepControllerConfig(method_order=1),
        tolerances=_tolerances(1.0),
        target_time_ns=0.65,
        minimum_step_ns=0.001,
        maximum_step_ns=1.0,
        maximum_attempts=20,
        maximum_accepted_slabs=20,
        public_sample_interval_ns=0.15,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        public_output_state=(
            AdaptivePairPublicOutputState.from_checkpoint_state(
                reopened.public_output_state
            )
        ),
        checkpoint_store=reopened,
        intrinsic_spin_reduction_history=restored_reduction,
        build_intrinsic_spin_reduction_candidate=_build_spin_reduction_candidate,
    )

    assert resumed.controller_state == continuous.controller_state
    assert resumed.public_output_state == continuous.public_output_state
    assert resumed.intrinsic_spin_reduction_history is not None
    assert continuous.intrinsic_spin_reduction_history is not None
    assert (
        resumed.intrinsic_spin_reduction_history.to_checkpoint_payload()
        == continuous.intrinsic_spin_reduction_history.to_checkpoint_payload()
    )
    for restored, expected in (
        (resumed_rider.build_current(), continuous_rider.build_current()),
        (resumed_driver.build_current(), continuous_driver.build_current()),
    ):
        for name in ("x", "t", "radiation_energy", "spin_z"):
            np.testing.assert_array_equal(
                np.asarray(getattr(restored, name)),
                np.asarray(getattr(expected, name)),
            )


@pytest.mark.parametrize("growable", (False, True), ids=("immutable", "growable"))
def test_window_resume_reproduces_causal_c5_coefficients_bitwise(
    tmp_path: Path,
    growable: bool,
) -> None:
    controller = AdaptivePairControllerState(
        current_step_ns=0.02,
        rider_proper_step_guess_ns=0.01,
        driver_proper_step_guess_ns=0.005,
    )
    controller_config = StepControllerConfig(
        method_order=1,
        maximum_growth_factor=1.0,
    )
    continuous_rider, continuous_driver = _magnetic_history_pair(18)
    continuous_c5 = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        continuous_rider.build_current(),
        continuous_driver.build_current(),
    )
    continuous_c5_arguments = (
        {
            "growable_causal_c5_source_history": (
                GrowableAcceptedPairCausalC5SourceHistory.from_accepted(
                    continuous_c5
                )
            )
        }
        if growable
        else {"causal_c5_source_history": continuous_c5}
    )
    continuous = run_exact_pair_adaptive_window(
        rider_builder=continuous_rider,
        driver_builder=continuous_driver,
        advance_rider=_advance(2.0),
        advance_driver=_advance(4.0),
        controller_state=controller,
        controller_config=controller_config,
        tolerances=_tolerances(1.0),
        target_time_ns=0.21,
        minimum_step_ns=0.001,
        maximum_step_ns=1.0,
        maximum_attempts=20,
        maximum_accepted_slabs=20,
        public_sample_interval_ns=0.15,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        **continuous_c5_arguments,
    )

    interrupted_rider, interrupted_driver = _magnetic_history_pair(18)
    interrupted_c5 = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        interrupted_rider.build_current(),
        interrupted_driver.build_current(),
    )
    interrupted_c5_arguments = (
        {
            "growable_causal_c5_source_history": (
                GrowableAcceptedPairCausalC5SourceHistory.from_accepted(
                    interrupted_c5
                )
            )
        }
        if growable
        else {"causal_c5_source_history": interrupted_c5}
    )
    directory = tmp_path / f"c5-resume-{growable}.checkpoint"
    store = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "c5-resume"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    with pytest.raises(SharedLabTimeError, match="maximum accepted slabs"):
        run_exact_pair_adaptive_window(
            rider_builder=interrupted_rider,
            driver_builder=interrupted_driver,
            advance_rider=_advance(2.0),
            advance_driver=_advance(4.0),
            controller_state=controller,
            controller_config=controller_config,
            tolerances=_tolerances(1.0),
            target_time_ns=0.21,
            minimum_step_ns=0.001,
            maximum_step_ns=1.0,
            maximum_attempts=20,
            maximum_accepted_slabs=1,
            public_sample_interval_ns=0.15,
            magnetic_dipole=MagneticDipoleConfig(),
            include_dipole_source=False,
            checkpoint_store=store,
            **interrupted_c5_arguments,
        )

    reopened = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "c5-resume"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=True,
    )
    resumed_rider = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
    resumed_driver = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
    reopened.restore_pair(resumed_rider, resumed_driver)
    restored_c5 = reopened.restore_causal_c5_source_history(
        resumed_rider.build_current(),
        resumed_driver.build_current(),
    )
    assert restored_c5 is not None
    resumed_c5_arguments = (
        {
            "growable_causal_c5_source_history": (
                GrowableAcceptedPairCausalC5SourceHistory.from_accepted(restored_c5)
            )
        }
        if growable
        else {"causal_c5_source_history": restored_c5}
    )
    resumed = run_exact_pair_adaptive_window(
        rider_builder=resumed_rider,
        driver_builder=resumed_driver,
        advance_rider=_advance(2.0),
        advance_driver=_advance(4.0),
        controller_state=AdaptivePairControllerState.from_checkpoint_state(
            reopened.controller_state
        ),
        controller_config=controller_config,
        tolerances=_tolerances(1.0),
        target_time_ns=0.21,
        minimum_step_ns=0.001,
        maximum_step_ns=1.0,
        maximum_attempts=20,
        maximum_accepted_slabs=20,
        public_sample_interval_ns=0.15,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        public_output_state=AdaptivePairPublicOutputState.from_checkpoint_state(
            reopened.public_output_state
        ),
        checkpoint_store=reopened,
        **resumed_c5_arguments,
    )

    assert continuous.causal_c5_source_history is not None
    assert resumed.causal_c5_source_history is not None
    assert resumed.controller_state == continuous.controller_state
    rebuilt = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        continuous_rider.build_current(),
        continuous_driver.build_current(),
    )
    for resumed_collection, continuous_collection, rebuilt_collection in (
        (
            resumed.causal_c5_source_history.rider,
            continuous.causal_c5_source_history.rider,
            rebuilt.rider,
        ),
        (
            resumed.causal_c5_source_history.driver,
            continuous.causal_c5_source_history.driver,
            rebuilt.driver,
        ),
    ):
        for resumed_source, continuous_source, rebuilt_source in zip(
            resumed_collection.sources,
            continuous_collection.sources,
            rebuilt_collection.sources,
        ):
            for resumed_segment, continuous_segment, rebuilt_segment in zip(
                resumed_source.history.frozen_segments,
                continuous_source.history.frozen_segments,
                rebuilt_source.history.frozen_segments,
            ):
                np.testing.assert_array_equal(
                    resumed_segment.position_coefficients_mm,
                    continuous_segment.position_coefficients_mm,
                )
                np.testing.assert_array_equal(
                    rebuilt_segment.position_coefficients_mm,
                    continuous_segment.position_coefficients_mm,
                )
                np.testing.assert_array_equal(
                    resumed_segment.rest_spin_stereographic_coefficients,
                    continuous_segment.rest_spin_stereographic_coefficients,
                )
                np.testing.assert_array_equal(
                    rebuilt_segment.rest_spin_stereographic_coefficients,
                    continuous_segment.rest_spin_stereographic_coefficients,
                )

        event = ObserverEvent(
            time_ns=0.175,
            position_mm=np.asarray((15.0, 0.0, 0.0)),
        )
        resumed_response = evaluate_causal_c5_dipole_source_collection_native(
            resumed_collection,
            event,
            root_tolerance_mm=1.0e-12,
        )
        continuous_response = evaluate_causal_c5_dipole_source_collection_native(
            continuous_collection,
            event,
            root_tolerance_mm=1.0e-12,
        )
        for name in ("four_potential", "field_tensor", "partial_f"):
            np.testing.assert_array_equal(
                getattr(resumed_response, name),
                getattr(continuous_response, name),
            )


def test_window_resume_reproduces_diagnostic_route_trace_exactly(
    tmp_path: Path,
) -> None:
    diagnostic_builder = (
        build_accepted_pair_intrinsic_spin_reduction_diagnostic_candidate
    )
    continuous_rider, continuous_driver, continuous = _run_window(
        public_interval_ns=0.15,
        intrinsic_spin_reduction_history=_initial_spin_reduction_history(),
        build_intrinsic_spin_reduction_candidate=diagnostic_builder,
        rider_advance=_diagnostic_unavailable_advance(2.0),
        driver_advance=_diagnostic_unavailable_advance(4.0),
    )
    interrupted_rider, interrupted_driver = _pair()
    directory = tmp_path / "diagnostic-resume.checkpoint"
    store = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "diagnostic-resume-test"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    with pytest.raises(SharedLabTimeError, match="maximum accepted slabs"):
        run_exact_pair_adaptive_window(
            rider_builder=interrupted_rider,
            driver_builder=interrupted_driver,
            advance_rider=_diagnostic_unavailable_advance(2.0),
            advance_driver=_diagnostic_unavailable_advance(4.0),
            controller_state=_controller(),
            controller_config=StepControllerConfig(method_order=1),
            tolerances=_tolerances(1.0),
            target_time_ns=0.65,
            minimum_step_ns=0.001,
            maximum_step_ns=1.0,
            maximum_attempts=20,
            maximum_accepted_slabs=1,
            public_sample_interval_ns=0.15,
            magnetic_dipole=MagneticDipoleConfig(),
            include_dipole_source=False,
            checkpoint_store=store,
            intrinsic_spin_reduction_history=_initial_spin_reduction_history(),
            build_intrinsic_spin_reduction_candidate=diagnostic_builder,
        )

    reopened = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "diagnostic-resume-test"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=True,
    )
    resumed_rider = GrowableTrajectoryBuilder(1, 1)
    resumed_driver = GrowableTrajectoryBuilder(1, 1)
    reopened.restore_pair(resumed_rider, resumed_driver)
    assert reopened.intrinsic_spin_reduction_state is not None
    restored_reduction = (
        AcceptedPairIntrinsicSpinReductionHistory.from_checkpoint_payload(
            reopened.intrinsic_spin_reduction_state
        )
    )
    resumed = run_exact_pair_adaptive_window(
        rider_builder=resumed_rider,
        driver_builder=resumed_driver,
        advance_rider=_diagnostic_unavailable_advance(2.0),
        advance_driver=_diagnostic_unavailable_advance(4.0),
        controller_state=AdaptivePairControllerState.from_checkpoint_state(
            reopened.controller_state
        ),
        controller_config=StepControllerConfig(method_order=1),
        tolerances=_tolerances(1.0),
        target_time_ns=0.65,
        minimum_step_ns=0.001,
        maximum_step_ns=1.0,
        maximum_attempts=20,
        maximum_accepted_slabs=20,
        public_sample_interval_ns=0.15,
        magnetic_dipole=MagneticDipoleConfig(),
        include_dipole_source=False,
        public_output_state=AdaptivePairPublicOutputState.from_checkpoint_state(
            reopened.public_output_state
        ),
        checkpoint_store=reopened,
        intrinsic_spin_reduction_history=restored_reduction,
        build_intrinsic_spin_reduction_candidate=diagnostic_builder,
    )

    assert resumed.intrinsic_spin_reduction_history is not None
    assert continuous.intrinsic_spin_reduction_history is not None
    assert (
        resumed.intrinsic_spin_reduction_history.to_checkpoint_payload()
        == continuous.intrinsic_spin_reduction_history.to_checkpoint_payload()
    )
    for restored, expected in (
        (resumed_rider.build_current(), continuous_rider.build_current()),
        (resumed_driver.build_current(), continuous_driver.build_current()),
    ):
        np.testing.assert_array_equal(restored.x, expected.x)
        np.testing.assert_array_equal(restored.t, expected.t)
