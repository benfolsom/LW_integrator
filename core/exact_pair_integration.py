"""Production orchestration for checkpointed exact-pair adaptive runs.

The numerical stepper lives in :mod:`core.adaptive_pair_return`.  This module
connects it to initialized inertial histories, the maintained equations of
motion, append-only checkpoints, cancellation, and the legacy integrator
return shape used by the CLI and GUI.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Sequence, cast

import numpy as np

from .adaptive_pair_return import (
    AdaptivePairControllerState,
    AdaptivePairPublicOutputState,
    IntrinsicSpinReductionCandidate,
    run_exact_pair_adaptive_window,
)
from .causal_c5_dipole_provider import (
    AcceptedPairCausalC5SourceHistory,
    GrowableAcceptedPairCausalC5SourceHistory,
)
from .causal_local_source_history import AcceptedPairCausalLocalSourceHistory
from .exact_pair_trial import ExactPairEOMOptions, make_exact_role_eom_advance
from .growable_causal_local_source_history import (
    GrowableAcceptedPairCausalLocalSourceHistory,
)
from .integration_checkpoint import AcceptedPairCheckpointStore
from .self_consistency import SelfConsistencyConfig
from .spin_self_force_reduction_history import (
    AcceptedPairIntrinsicSpinReductionHistory,
    build_accepted_pair_intrinsic_spin_reduction_candidate,
    build_accepted_pair_intrinsic_spin_reduction_diagnostic_candidate,
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

# The fifteen-knot spin derivative fit remains comfortably below its existing
# condition-number guard when neighboring accepted intervals grow by at most
# five percent.  Larger jumps can make smooth data numerically singular even
# though every individual adaptive step is otherwise healthy.
_CAUSAL_C5_MAXIMUM_STEP_GROWTH = 1.05


def _step_controller_config(*, causal_c5_enabled: bool) -> StepControllerConfig:
    """Return the ordinary controller with the C5 cadence safeguard applied."""

    return StepControllerConfig(
        method_order=1,
        maximum_growth_factor=(
            _CAUSAL_C5_MAXIMUM_STEP_GROWTH if causal_c5_enabled else 2.0
        ),
    )


def _scaled_tolerances(
    scale: float,
    diagnostics_absolute: float | None = None,
    *,
    position_momentum_scale: float = 1.0,
    diagnostic_ulp_floor: bool = False,
    diagnostic_ulp_multiplier: float = 4.0,
) -> StepDoublingTolerances:
    """Return the validated scale-1 first-pass error model."""

    return StepDoublingTolerances(
        position_mm=ErrorScale(
            scale * 1.0e-15 * position_momentum_scale,
            scale * 1.0e-10 * position_momentum_scale,
        ),
        mechanical_momentum_native=ErrorScale(
            scale * 1.0e-14 * position_momentum_scale,
            scale * 1.0e-10 * position_momentum_scale,
        ),
        rest_spin=ErrorScale(scale * 1.0e-13, scale * 1.0e-10),
        diagnostics_native=ErrorScale(
            scale * 1.0e-13 if diagnostics_absolute is None else diagnostics_absolute,
            scale * 1.0e-8,
        ),
        diagnostic_ulp_floor=diagnostic_ulp_floor,
        diagnostic_ulp_multiplier=diagnostic_ulp_multiplier,
    )


def _spin_reaction_summary(
    history: AcceptedPairIntrinsicSpinReductionHistory | None, mode: str
) -> dict[str, Any] | None:
    """Keep diagnostic-only output stable; add work only for applied recoil."""
    if history is None or mode not in {"diagnostic", "experimental_linear_spin"}:
        return None
    experimental = mode == "experimental_linear_spin"
    summary: dict[str, Any] = {
        "mode": mode if experimental else "diagnostic_only",
        "applied_as_force": experimental,
    }
    if experimental:
        summary["omitted_reaction_terms"] = [
            "magnetic_dipole_squared",
            "finite_size_matching",
        ]
    for role in ("rider", "driver"):
        trace = getattr(history, f"{role}_diagnostics")
        counts: dict[str, Any] = {
            "total": trace.total_records,
            "analytical": trace.analytical_records,
            "causal": trace.causal_records,
            "unavailable": trace.unavailable_records,
        }
        if experimental:
            counts.update(
                feedback_applied_records=trace.feedback_applied_records,
                feedback_evaluated_records=trace.feedback_evaluated_records,
                feedback_work_native=trace.feedback_work_native,
                feedback_absolute_work_native=trace.feedback_absolute_work_native,
                feedback_four_impulse_native=trace.feedback_four_impulse_native,
                feedback_energy_adjustment_native=trace.feedback_energy_adjustment_native,
                feedback_absolute_energy_adjustment_native=trace.feedback_absolute_energy_adjustment_native,
            )
        summary[role] = counts
    return summary


def _new_builder_from_seed(
    seed: Sequence[ParticleState],
    *,
    magnetic_dipole: bool,
) -> GrowableTrajectoryBuilder:
    if not seed:
        raise ValueError("exact-pair adaptive seed history must not be empty")
    particle_count = int(np.asarray(seed[-1].get("x", np.zeros(0))).size)
    if particle_count < 1:
        raise ValueError("exact-pair adaptive mode requires nonempty bunches")
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
    space_charge: Any = None,
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
    initial_causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None,
    initial_causal_local_source_history: (
        AcceptedPairCausalLocalSourceHistory | None
    ) = None,
) -> tuple[
    Trajectory,
    Trajectory,
    TrajectoryArrays,
    TrajectoryArrays,
    list[dict[str, float]],
]:
    """Run or resume adaptive exact pairs or charge-source bunches."""

    if not adaptive.enabled or adaptive.target_lab_time_ns is None:
        raise ValueError("exact-pair adaptive production mode is not enabled")
    if not checkpoint.enabled:
        raise ValueError("exact-pair adaptive production mode requires checkpointing")
    checkpoint_directory = checkpoint.resume_from or checkpoint.directory
    if checkpoint_directory is None:  # pragma: no cover - CheckpointConfig invariant
        raise ValueError("exact-pair adaptive checkpoint directory is required")

    if not rider_seed or not driver_seed:
        raise ValueError("exact-pair adaptive seed histories must not be empty")
    bunch_mode = len(rider_seed[-1]["x"]) > 1 or len(driver_seed[-1]["x"]) > 1
    if bunch_mode and (
        magnetic_dipole.source.active
        or magnetic_dipole.intrinsic_spin_self_reaction_mode != "off"
    ):
        raise NotImplementedError(
            "adaptive bunches support charge sources with spin self-reaction off"
        )
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
        not bunch_mode
        and magnetic_dipole.exact_retarded_update
        == "second_order_start_taylor_endpoint"
    )
    causal_c5_enabled = bool(
        magnetic_dipole.source.active
        and magnetic_dipole.source.history_model == "causal_c5"
    )
    causal_local_enabled = bool(
        magnetic_dipole.source.active
        and magnetic_dipole.source.history_model == "causal_local_jet"
    )
    growable_c5_history: GrowableAcceptedPairCausalC5SourceHistory | None = None
    growable_local_history: GrowableAcceptedPairCausalLocalSourceHistory | None = None
    if resume:
        rider_builder = GrowableTrajectoryBuilder(
            8, len(rider_seed[-1]["x"]), magnetic_dipole=True
        )
        driver_builder = GrowableTrajectoryBuilder(
            8, len(driver_seed[-1]["x"]), magnetic_dipole=True
        )
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
        if causal_c5_enabled:
            restored_c5 = store.restore_causal_c5_source_history(
                rider_builder.build_current(),
                driver_builder.build_current(),
            )
            if restored_c5 is None:
                raise ValueError(
                    "causal C5 exact-pair checkpoint has no frozen source history"
                )
            growable_c5_history = (
                GrowableAcceptedPairCausalC5SourceHistory.from_accepted(restored_c5)
            )
        if causal_local_enabled:
            restored_local = store.restore_causal_local_source_history(
                rider_builder.build_current(),
                driver_builder.build_current(),
            )
            if restored_local is None:
                raise ValueError(
                    "causal local exact-pair checkpoint has no source history"
                )
            growable_local_history = (
                GrowableAcceptedPairCausalLocalSourceHistory.from_accepted(
                    restored_local
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
        if causal_c5_enabled:
            accepted_c5 = initial_causal_c5_source_history
            if accepted_c5 is None:
                accepted_c5 = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
                    rider_builder.build_current(),
                    driver_builder.build_current(),
                )
            growable_c5_history = (
                GrowableAcceptedPairCausalC5SourceHistory.from_accepted(accepted_c5)
            )
        if causal_local_enabled:
            accepted_local = initial_causal_local_source_history
            if accepted_local is None:
                accepted_local = (
                    AcceptedPairCausalLocalSourceHistory.from_trajectory_arrays(
                        rider_builder.build_current(),
                        driver_builder.build_current(),
                    )
                )
            growable_local_history = (
                GrowableAcceptedPairCausalLocalSourceHistory.from_accepted(
                    accepted_local
                )
            )

    if reduction_diagnostic_enabled:
        reduction_candidate_builder = (
            build_accepted_pair_intrinsic_spin_reduction_diagnostic_candidate
            if magnetic_dipole.intrinsic_spin_self_reaction_mode
            in {"diagnostic", "experimental_linear_spin"}
            else build_accepted_pair_intrinsic_spin_reduction_candidate
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
            space_charge=space_charge,
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
        controller_config=_step_controller_config(
            causal_c5_enabled=(causal_c5_enabled or causal_local_enabled)
        ),
        tolerances=_scaled_tolerances(
            adaptive.tolerance_scale,
            adaptive.diagnostics_absolute_tolerance_native,
            position_momentum_scale=adaptive.position_momentum_tolerance_scale,
            diagnostic_ulp_floor=adaptive.diagnostic_ulp_floor,
            diagnostic_ulp_multiplier=adaptive.diagnostic_ulp_multiplier,
        ),
        target_time_ns=adaptive.target_lab_time_ns,
        minimum_step_ns=initial_step_ns * adaptive.minimum_step_factor,
        maximum_step_ns=initial_step_ns * adaptive.maximum_step_factor,
        maximum_attempts=adaptive.maximum_attempts,
        maximum_accepted_slabs=adaptive.maximum_accepted_slabs,
        public_sample_interval_ns=public_interval,
        magnetic_dipole=magnetic_dipole,
        include_dipole_source=magnetic_dipole.source.active,
        same_bunch_fields=bool(space_charge is not None and space_charge.enabled),
        public_output_state=public_output,
        checkpoint_store=store,
        spin_interpolation_model="causal_frozen_c1",
        absolute_time_tolerance_ns=adaptive.shared_time_absolute_tolerance_ns,
        relative_time_tolerance=adaptive.shared_time_relative_tolerance,
        cancel_callback=cancel_callback,
        accepted_progress_callback=progress,
        record_attempt_diagnostics=adaptive.diagnostic_ulp_floor,
        intrinsic_spin_reduction_history=reduction_history,
        build_intrinsic_spin_reduction_candidate=reduction_candidate_builder,
        growable_causal_c5_source_history=growable_c5_history,
        growable_causal_local_source_history=growable_local_history,
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
        "dipole_source_history": (
            "causal_c5"
            if causal_c5_enabled
            else "causal_local_jet" if causal_local_enabled else "causal_frozen_c1"
        ),
        "intrinsic_spin_self_reaction_diagnostics": _spin_reaction_summary(
            result.intrinsic_spin_reduction_history,
            magnetic_dipole.intrinsic_spin_self_reaction_mode,
        ),
    }
    if adaptive.diagnostic_ulp_floor:
        summary["diagnostic_ulp_floor"] = {
            "reference": "per-diagnostic increment magnitude; projection uses endpoint kinetic energy; maximum of full and refined",
            "controller_policy": "exclude floor-dominated diagnostic entries",
            "multiplier": adaptive.diagnostic_ulp_multiplier,
            "floor_applied_trials": sum(
                bool(d.floored_groups) for d in result.attempt_diagnostics
            ),
            "floor_changed_acceptances": sum(
                d.accepted and d.diagnostic_floor_changed_acceptance
                for d in result.attempt_diagnostics
            ),
            "attempt_diagnostics": [asdict(d) for d in result.attempt_diagnostics],
        }
    cast(dict[str, Any], rider_legacy[-1])["_adaptive_pair_return"] = dict(summary)
    cast(dict[str, Any], driver_legacy[-1])["_adaptive_pair_return"] = dict(summary)
    return rider_legacy, driver_legacy, rider, driver, []


__all__ = ["run_exact_pair_adaptive_integrator"]
