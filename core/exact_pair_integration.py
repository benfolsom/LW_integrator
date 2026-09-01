"""Production orchestration for checkpointed exact-pair adaptive runs.

The numerical stepper lives in :mod:`core.adaptive_pair_return`.  This module
connects it to initialized inertial histories, the maintained equations of
motion, append-only checkpoints, cancellation, and the legacy integrator
return shape used by the CLI and GUI.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, cast

import numpy as np

from .adaptive_pair_return import (
    AdaptivePairControllerState,
    AdaptivePairPublicOutputState,
    IntrinsicSpinReductionCandidate,
    run_exact_pair_adaptive_window,
)
from .exact_pair_trial import ExactPairEOMOptions, make_exact_role_eom_advance
from .integration_checkpoint import AcceptedPairCheckpointStore
from .self_consistency import SelfConsistencyConfig
from .spin_self_force_reduction_history import (
    AcceptedPairIntrinsicSpinReductionHistory,
    build_accepted_pair_intrinsic_spin_reduction_candidate,
)
from .step_doubling import ErrorScale, StepControllerConfig, StepDoublingTolerances
from .types import (
    AdaptivePairReturnConfig,
    CheckpointConfig,
    ChronoMatchingMode,
    ExternalFieldConfig,
    GrowableTrajectoryBuilder,
    MagneticDipoleConfig,
    ParticleState,
    Trajectory,
    TrajectoryArrays,
)


def _scaled_tolerances(scale: float) -> StepDoublingTolerances:
    """Return the validated scale-1 first-pass error model."""

    return StepDoublingTolerances(
        position_mm=ErrorScale(scale * 1.0e-15, scale * 1.0e-10),
        mechanical_momentum_native=ErrorScale(scale * 1.0e-14, scale * 1.0e-10),
        rest_spin=ErrorScale(scale * 1.0e-13, scale * 1.0e-10),
        diagnostics_native=ErrorScale(scale * 1.0e-13, scale * 1.0e-8),
    )


def _new_builder_from_seed(
    seed: Sequence[ParticleState],
    *,
    magnetic_dipole: bool,
) -> GrowableTrajectoryBuilder:
    if not seed:
        raise ValueError("exact-pair adaptive seed history must not be empty")
    particle_count = int(np.asarray(seed[-1].get("x", np.zeros(0))).size)
    if particle_count != 1:
        raise ValueError("exact-pair adaptive mode requires one particle per role")
    builder = GrowableTrajectoryBuilder(
        max(8, len(seed) + 1),
        particle_count,
        magnetic_dipole=magnetic_dipole,
    )
    for state in seed:
        builder.append_step(state)
    return builder


def run_exact_pair_adaptive_integrator(
    *,
    rider_seed: Sequence[ParticleState],
    driver_seed: Sequence[ParticleState],
    initial_step_ns: float,
    requested_public_samples: int,
    aperture_radius_mm: float,
    magnetic_dipole: MagneticDipoleConfig,
    self_consistency: SelfConsistencyConfig | None,
    chrono_mode: ChronoMatchingMode,
    radiation_reaction_mode: str,
    external_field: ExternalFieldConfig | None,
    adaptive: AdaptivePairReturnConfig,
    checkpoint: CheckpointConfig,
    compatibility_payload: dict[str, Any],
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> tuple[
    Trajectory,
    Trajectory,
    TrajectoryArrays,
    TrajectoryArrays,
    list[dict[str, float]],
]:
    """Run or resume the strict one-rider/one-driver adaptive production path."""

    if not adaptive.enabled or adaptive.target_lab_time_ns is None:
        raise ValueError("exact-pair adaptive production mode is not enabled")
    if not checkpoint.enabled:
        raise ValueError("exact-pair adaptive production mode requires checkpointing")
    checkpoint_directory = checkpoint.resume_from or checkpoint.directory
    if checkpoint_directory is None:  # pragma: no cover - CheckpointConfig invariant
        raise ValueError("exact-pair adaptive checkpoint directory is required")

    resume = checkpoint.resume_from is not None
    store = AcceptedPairCheckpointStore(
        checkpoint_directory,
        compatibility_payload=compatibility_payload,
        interval_knots=checkpoint.interval_steps,
        interval_seconds=checkpoint.interval_seconds,
        resume=resume,
    )
    public_output: AdaptivePairPublicOutputState | None = None
    controller: AdaptivePairControllerState | None = None
    reduction_history: AcceptedPairIntrinsicSpinReductionHistory | None = None
    reduction_candidate_builder: IntrinsicSpinReductionCandidate | None = None
    reduction_diagnostic_enabled = bool(
        magnetic_dipole.exact_retarded_update == "second_order_start_taylor_endpoint"
    )
    if resume:
        rider_builder = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
        driver_builder = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
        store.restore_pair(rider_builder, driver_builder)
        controller = AdaptivePairControllerState.from_checkpoint_state(
            store.controller_state
        )
        public_output = AdaptivePairPublicOutputState.from_checkpoint_state(
            store.public_output_state
        )
        if reduction_diagnostic_enabled:
            payload = store.intrinsic_spin_reduction_state
            if payload is None:
                raise ValueError(
                    "second-order exact-pair checkpoint has no intrinsic-spin "
                    "diagnostic history"
                )
            reduction_history = (
                AcceptedPairIntrinsicSpinReductionHistory.from_checkpoint_payload(
                    payload
                )
            )
        active_row = public_output.selected_rows[0]
    else:
        rider_builder = _new_builder_from_seed(
            rider_seed,
            magnetic_dipole=magnetic_dipole.enabled,
        )
        driver_builder = _new_builder_from_seed(
            driver_seed,
            magnetic_dipole=magnetic_dipole.enabled,
        )
        if rider_builder.accepted_steps != driver_builder.accepted_steps:
            raise ValueError("exact-pair adaptive seed histories must be aligned")
        active_row = len(rider_seed) - 1
        if reduction_diagnostic_enabled:
            reduction_history = AcceptedPairIntrinsicSpinReductionHistory.empty()

    if reduction_diagnostic_enabled:
        reduction_candidate_builder = (
            build_accepted_pair_intrinsic_spin_reduction_candidate
        )

    active_start_time_ns = float(rider_builder.build_current().t[active_row, 0])
    active_duration_ns = adaptive.target_lab_time_ns - active_start_time_ns
    if active_duration_ns <= 0.0:
        raise ValueError(
            "exact-pair adaptive target time must follow the active start event"
        )

    initial_controller = controller or AdaptivePairControllerState(
        current_step_ns=initial_step_ns,
        rider_proper_step_guess_ns=initial_step_ns,
        driver_proper_step_guess_ns=initial_step_ns,
    )
    public_interval = adaptive.public_sample_interval_ns
    if public_interval is None:
        public_interval = active_duration_ns / float(
            max(1, requested_public_samples - 1)
        )

    advance = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=aperture_radius_mm,
            magnetic_dipole=magnetic_dipole,
            self_consistency=self_consistency,
            chrono_mode=chrono_mode,
            radiation_reaction_mode=radiation_reaction_mode,
            external_field=external_field,
            cancel_callback=cancel_callback,
        )
    )

    def progress(current_time_ns: float, target_time_ns: float) -> None:
        if progress_callback is None:
            return
        total = max(1, requested_public_samples)
        fraction = min(
            max(
                (current_time_ns - active_start_time_ns)
                / (target_time_ns - active_start_time_ns),
                0.0,
            ),
            1.0,
        )
        progress_callback(min(total, int(fraction * total)), total)

    result = run_exact_pair_adaptive_window(
        rider_builder=rider_builder,
        driver_builder=driver_builder,
        advance_rider=advance,
        advance_driver=advance,
        controller_state=initial_controller,
        controller_config=StepControllerConfig(method_order=1),
        tolerances=_scaled_tolerances(adaptive.tolerance_scale),
        target_time_ns=adaptive.target_lab_time_ns,
        minimum_step_ns=initial_step_ns * adaptive.minimum_step_factor,
        maximum_step_ns=initial_step_ns * adaptive.maximum_step_factor,
        maximum_attempts=adaptive.maximum_attempts,
        maximum_accepted_slabs=adaptive.maximum_accepted_slabs,
        public_sample_interval_ns=public_interval,
        magnetic_dipole=magnetic_dipole,
        include_dipole_source=magnetic_dipole.source.active,
        public_output_state=public_output,
        checkpoint_store=store,
        spin_interpolation_model="causal_frozen_c1",
        absolute_time_tolerance_ns=adaptive.shared_time_absolute_tolerance_ns,
        relative_time_tolerance=adaptive.shared_time_relative_tolerance,
        cancel_callback=cancel_callback,
        accepted_progress_callback=progress,
        intrinsic_spin_reduction_history=reduction_history,
        build_intrinsic_spin_reduction_candidate=reduction_candidate_builder,
    )
    if progress_callback is not None and result.completed:
        progress_callback(
            max(1, requested_public_samples),
            max(1, requested_public_samples),
        )

    rider_full = rider_builder.build_current()
    driver_full = driver_builder.build_current()
    # Keep every accepted midpoint and endpoint in the returned trajectory.
    # Radiation, Medina work, and projection arrays contain per-knot increments;
    # decimating here would corrupt their sums. Plot/export decimation remains a
    # separate presentation concern.
    from .integration_runner import _slice_trajectory_arrays

    rider = _slice_trajectory_arrays(rider_full, active_row, rider_full.n_steps)
    driver = _slice_trajectory_arrays(driver_full, active_row, driver_full.n_steps)
    if rider is None or driver is None:  # pragma: no cover - concrete inputs
        raise RuntimeError("exact-pair adaptive histories unexpectedly disappeared")
    rider_legacy = rider.to_legacy()
    driver_legacy = driver.to_legacy()
    summary = {
        "completed": result.completed,
        "accepted_slabs": result.accepted_slabs,
        "rejected_trials": result.rejected_trials,
        "attempts": result.attempts,
        "final_time_ns": result.final_time_ns,
        "checkpoint_directory": str(store.directory),
        "checkpoint_resumed": resume,
        "accepted_history_knots": rider_full.n_steps,
        "public_selected_rows": len(result.public_output_state.selected_rows),
        "intrinsic_spin_reduction_samples": (
            None
            if result.intrinsic_spin_reduction_history is None
            else {
                "rider": result.intrinsic_spin_reduction_history.rider.sample_count,
                "driver": result.intrinsic_spin_reduction_history.driver.sample_count,
            }
        ),
    }
    cast(dict[str, Any], rider_legacy[-1])["_adaptive_pair_return"] = dict(summary)
    cast(dict[str, Any], driver_legacy[-1])["_adaptive_pair_return"] = dict(summary)
    return rider_legacy, driver_legacy, rider, driver, []


__all__ = ["run_exact_pair_adaptive_integrator"]
