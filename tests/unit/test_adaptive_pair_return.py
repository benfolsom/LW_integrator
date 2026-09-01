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
from core.integration_checkpoint import AcceptedPairCheckpointStore
from core.integration_runner import IntegrationCancelled
from core.shared_lab_time import SharedLabTimeError
from core.spin_self_force_reduction_history import (
    AcceptedPairIntrinsicSpinReductionHistory,
    build_accepted_pair_intrinsic_spin_reduction_candidate,
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
):
    rider, driver = _pair()
    result = run_exact_pair_adaptive_window(
        rider_builder=rider,
        driver_builder=driver,
        advance_rider=_advance(2.0),
        advance_driver=_advance(4.0),
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
