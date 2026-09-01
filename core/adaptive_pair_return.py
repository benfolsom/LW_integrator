"""Transactional adaptive attempts and bounded exact-pair run windows.

The accepted source history and the sparse public-output selection are
deliberately independent.  Every accepted step-doubling midpoint and endpoint
remains available to future retarded providers, while output cadence selects
only existing accepted row indices and cannot influence the equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol

import numpy as np

from .causal_c5_dipole_provider import (
    AcceptedPairCausalC5SourceHistory,
    build_accepted_pair_causal_c5_candidate,
)
from .exact_pair_trial import (
    AdvanceRoleTrial,
    ExactPairStepDoublingTrial,
    commit_accepted_exact_pair_step_doubling_trial,
    solve_exact_pair_step_doubling_trial,
)
from .shared_lab_time import SharedLabTimeError
from .spin_self_force_reduction_history import (
    AcceptedPairIntrinsicSpinReductionHistory,
)
from .step_doubling import (
    StepControllerConfig,
    StepDoublingTolerances,
    propose_next_step_ns,
)
from .types import GrowableTrajectoryBuilder, MagneticDipoleConfig, TrajectoryArrays


class _AcceptedPairCheckpoint(Protocol):
    def due(self, accepted_knots: int, *, force: bool = False) -> bool: ...

    def write(
        self,
        *,
        rider: TrajectoryArrays,
        driver: TrajectoryArrays,
        controller_state: dict[str, Any],
        public_output_state: dict[str, Any],
        intrinsic_spin_reduction_state: dict[str, object] | None = None,
        causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None,
        complete: bool = False,
    ) -> None: ...


IntrinsicSpinReductionCandidate = Callable[
    [ExactPairStepDoublingTrial, AcceptedPairIntrinsicSpinReductionHistory],
    AcceptedPairIntrinsicSpinReductionHistory,
]


@dataclass(frozen=True)
class AdaptivePairControllerState:
    """Checkpointable scalar state for the $1+1$ adaptive controller."""

    current_step_ns: float
    rider_proper_step_guess_ns: float
    driver_proper_step_guess_ns: float
    accepted_slabs: int = 0
    rejected_trials: int = 0

    def __post_init__(self) -> None:
        steps = (
            self.current_step_ns,
            self.rider_proper_step_guess_ns,
            self.driver_proper_step_guess_ns,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in steps):
            raise ValueError("adaptive pair step sizes must be finite and positive")
        if self.accepted_slabs < 0 or self.rejected_trials < 0:
            raise ValueError("adaptive pair counters must be non-negative")

    def to_checkpoint_state(self) -> dict[str, Any]:
        """Return a strict JSON-compatible controller payload."""

        return {
            "schema_version": 1,
            "current_step_ns": float(self.current_step_ns),
            "rider_proper_step_guess_ns": float(self.rider_proper_step_guess_ns),
            "driver_proper_step_guess_ns": float(self.driver_proper_step_guess_ns),
            "accepted_slabs": int(self.accepted_slabs),
            "rejected_trials": int(self.rejected_trials),
        }

    @classmethod
    def from_checkpoint_state(
        cls,
        payload: dict[str, Any],
    ) -> "AdaptivePairControllerState":
        """Restore the exact scalar state, rejecting unknown schema revisions."""

        if set(payload) != {
            "schema_version",
            "current_step_ns",
            "rider_proper_step_guess_ns",
            "driver_proper_step_guess_ns",
            "accepted_slabs",
            "rejected_trials",
        }:
            raise ValueError("adaptive pair checkpoint controller fields are invalid")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported adaptive pair controller schema")
        return cls(
            current_step_ns=float(payload["current_step_ns"]),
            rider_proper_step_guess_ns=float(payload["rider_proper_step_guess_ns"]),
            driver_proper_step_guess_ns=float(payload["driver_proper_step_guess_ns"]),
            accepted_slabs=int(payload["accepted_slabs"]),
            rejected_trials=int(payload["rejected_trials"]),
        )


@dataclass(frozen=True)
class AdaptivePairAttempt:
    """Result of one accepted or rejected transactional adaptive attempt."""

    trial: ExactPairStepDoublingTrial
    controller_state: AdaptivePairControllerState
    committed_rows: tuple[int, int] | None
    intrinsic_spin_reduction_history: (
        AcceptedPairIntrinsicSpinReductionHistory | None
    ) = None
    causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None

    @property
    def accepted(self) -> bool:
        return self.committed_rows is not None


@dataclass(frozen=True)
class AdaptivePairPublicOutputState:
    """Checkpointable sparse view over the complete accepted source history."""

    sample_interval_ns: float
    next_sample_time_ns: float
    selected_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(self.sample_interval_ns) or self.sample_interval_ns <= 0.0:
            raise ValueError("public output interval must be finite and positive")
        if not np.isfinite(self.next_sample_time_ns):
            raise ValueError("next public output time must be finite")
        if any(
            isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer))
            for row in self.selected_rows
        ):
            raise ValueError("public output rows must be integer row indices")
        rows = tuple(int(row) for row in self.selected_rows)
        if not rows or rows[0] < 0:
            raise ValueError("public output must contain an initial accepted row")
        if any(current <= previous for previous, current in zip(rows, rows[1:])):
            raise ValueError("public output rows must be strictly increasing")
        if self.next_sample_time_ns + self.sample_interval_ns <= (
            self.next_sample_time_ns
        ):
            raise ValueError("public output interval is below time resolution")
        object.__setattr__(self, "selected_rows", rows)

    def to_checkpoint_state(self) -> dict[str, Any]:
        """Return the complete sparse-output cursor as strict JSON data."""

        return {
            "schema_version": 1,
            "sample_interval_ns": float(self.sample_interval_ns),
            "next_sample_time_ns": float(self.next_sample_time_ns),
            "selected_rows": [int(row) for row in self.selected_rows],
        }

    @classmethod
    def from_checkpoint_state(
        cls,
        payload: dict[str, Any],
    ) -> "AdaptivePairPublicOutputState":
        """Restore one sparse-output cursor, rejecting partial schemas."""

        if set(payload) != {
            "schema_version",
            "sample_interval_ns",
            "next_sample_time_ns",
            "selected_rows",
        }:
            raise ValueError(
                "adaptive pair public-output checkpoint fields are invalid"
            )
        if payload["schema_version"] != 1:
            raise ValueError("unsupported adaptive pair public-output schema")
        rows = payload["selected_rows"]
        if not isinstance(rows, list):
            raise ValueError("adaptive pair public-output rows must be a list")
        if any(isinstance(row, bool) or not isinstance(row, int) for row in rows):
            raise ValueError("adaptive pair public-output rows must contain integers")
        return cls(
            sample_interval_ns=float(payload["sample_interval_ns"]),
            next_sample_time_ns=float(payload["next_sample_time_ns"]),
            selected_rows=tuple(int(row) for row in rows),
        )


@dataclass(frozen=True)
class AdaptivePairRunResult:
    """Outcome of one bounded internal adaptive integration window."""

    controller_state: AdaptivePairControllerState
    public_output_state: AdaptivePairPublicOutputState
    attempts: int
    accepted_slabs: int
    rejected_trials: int
    final_time_ns: float
    completed: bool
    attempt_diagnostics: tuple["AdaptivePairAttemptDiagnostics", ...] = ()
    intrinsic_spin_reduction_history: (
        AcceptedPairIntrinsicSpinReductionHistory | None
    ) = None
    causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None


@dataclass(frozen=True)
class AdaptivePairAttemptDiagnostics:
    """Optional read-only error trace for controller calibration."""

    attempted_step_ns: float
    accepted: bool
    normalized_error: float
    position_error: float
    mechanical_momentum_error: float
    rest_spin_error: float
    diagnostics_error: float
    position_error_index: tuple[int, ...]
    mechanical_momentum_error_index: tuple[int, ...]
    rest_spin_error_index: tuple[int, ...]
    diagnostics_error_index: tuple[int, ...]


def _accepted_pair_time_ns(
    rider_builder: GrowableTrajectoryBuilder,
    driver_builder: GrowableTrajectoryBuilder,
    *,
    tolerance_ns: float,
) -> float:
    if rider_builder.accepted_steps != driver_builder.accepted_steps:
        raise SharedLabTimeError("accepted rider and driver histories are misaligned")
    rider_time = np.asarray(rider_builder.build_current().t[-1], dtype=np.float64)
    driver_time = np.asarray(driver_builder.build_current().t[-1], dtype=np.float64)
    if rider_time.shape != (1,) or driver_time.shape != (1,):
        raise SharedLabTimeError(
            "adaptive pair return mode currently requires one particle per role"
        )
    values = (float(rider_time[0]), float(driver_time[0]))
    if not all(np.isfinite(value) for value in values):
        raise SharedLabTimeError("accepted pair coordinate time is not finite")
    if abs(values[0] - values[1]) > tolerance_ns:
        raise SharedLabTimeError("accepted rider and driver times are not synchronized")
    return values[0]


def _latest_pair_slab_scale_ns(
    rider_builder: GrowableTrajectoryBuilder,
    driver_builder: GrowableTrajectoryBuilder,
) -> float:
    """Estimate the last full slab from its retained endpoint interval.

    Adaptive acceptance publishes a midpoint and endpoint for every slab, so
    twice the final stored interval is the previous shared slab size.  The
    inertial seed also supplies a finite positive interval for the fresh-run
    boundary.
    """

    scales: list[float] = []
    for role, builder in (("rider", rider_builder), ("driver", driver_builder)):
        times = np.asarray(builder.build_current().t[:, 0], dtype=np.float64)
        if times.size < 2:
            scales.append(0.0)
            continue
        interval = float(times[-1] - times[-2])
        if not np.isfinite(interval) or interval <= 0.0:
            raise SharedLabTimeError(
                f"accepted {role} history has no positive final time interval"
            )
        scales.append(2.0 * interval)
    return max(scales)


def _initial_public_output_state(
    *,
    accepted_rows: int,
    current_time_ns: float,
    sample_interval_ns: float,
) -> AdaptivePairPublicOutputState:
    return AdaptivePairPublicOutputState(
        sample_interval_ns=sample_interval_ns,
        next_sample_time_ns=current_time_ns + sample_interval_ns,
        selected_rows=(accepted_rows - 1,),
    )


def _select_public_rows(
    state: AdaptivePairPublicOutputState,
    *,
    committed_rows: tuple[int, int],
    rider_builder: GrowableTrajectoryBuilder,
    time_tolerance_ns: float,
    include_final_row: bool,
) -> AdaptivePairPublicOutputState:
    selected = list(state.selected_rows)
    next_time = state.next_sample_time_ns
    times = np.asarray(rider_builder.build_current().t[:, 0], dtype=np.float64)
    for row in committed_rows:
        row_time = float(times[row])
        if row_time + time_tolerance_ns >= next_time:
            if row != selected[-1]:
                selected.append(row)
            crossed = 1 + math.floor(
                (row_time + time_tolerance_ns - next_time) / state.sample_interval_ns
            )
            next_time += crossed * state.sample_interval_ns
    if include_final_row and committed_rows[-1] != selected[-1]:
        selected.append(committed_rows[-1])
    return AdaptivePairPublicOutputState(
        sample_interval_ns=state.sample_interval_ns,
        next_sample_time_ns=next_time,
        selected_rows=tuple(selected),
    )


def attempt_exact_pair_adaptive_step(
    *,
    rider_builder: GrowableTrajectoryBuilder,
    driver_builder: GrowableTrajectoryBuilder,
    advance_rider: AdvanceRoleTrial,
    advance_driver: AdvanceRoleTrial,
    controller_state: AdaptivePairControllerState,
    controller_config: StepControllerConfig,
    tolerances: StepDoublingTolerances,
    minimum_step_ns: float,
    maximum_step_ns: float,
    magnetic_dipole: MagneticDipoleConfig,
    include_dipole_source: bool,
    spin_interpolation_model: str = "causal_frozen_c1",
    absolute_time_tolerance_ns: float = 1.0e-18,
    relative_time_tolerance: float = 1.0e-12,
    intrinsic_spin_reduction_history: (
        AcceptedPairIntrinsicSpinReductionHistory | None
    ) = None,
    build_intrinsic_spin_reduction_candidate: (
        IntrinsicSpinReductionCandidate | None
    ) = None,
    causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None,
) -> AdaptivePairAttempt:
    """Try one slab, committing only a healthy accepted two-half path.

    Medina caps are recoverable local-step failures and force a shrink. Other
    hard health failures indicate invalid state or solver semantics and abort
    rather than being hidden by repeated retries.
    """

    if (intrinsic_spin_reduction_history is None) != (
        build_intrinsic_spin_reduction_candidate is None
    ):
        raise ValueError(
            "intrinsic-spin history and candidate builder must be supplied together"
        )

    accepted_rider = rider_builder.build_current()
    accepted_driver = driver_builder.build_current()
    trial = solve_exact_pair_step_doubling_trial(
        accepted_rider_history=accepted_rider,
        accepted_driver_history=accepted_driver,
        advance_rider=advance_rider,
        advance_driver=advance_driver,
        delta_time_ns=controller_state.current_step_ns,
        rider_initial_proper_step_ns=(controller_state.rider_proper_step_guess_ns),
        driver_initial_proper_step_ns=(controller_state.driver_proper_step_guess_ns),
        magnetic_dipole=magnetic_dipole,
        include_dipole_source=include_dipole_source,
        tolerances=tolerances,
        method_order=controller_config.method_order,
        spin_interpolation_model=spin_interpolation_model,
        absolute_time_tolerance_ns=absolute_time_tolerance_ns,
        relative_time_tolerance=relative_time_tolerance,
    )
    cap_failures = tuple(
        failure for failure in trial.health_failures if "Medina impulse cap" in failure
    )
    fatal_failures = tuple(
        failure for failure in trial.health_failures if failure not in cap_failures
    )
    if fatal_failures:
        raise SharedLabTimeError(
            "fatal exact-pair trial health failure: " + "; ".join(fatal_failures)
        )

    accepted = bool(trial.assessment.accepted and not cap_failures)
    controller_error = float(trial.assessment.normalized_error)
    if cap_failures:
        controller_error = max(controller_error, 4.0)
    next_step_ns = propose_next_step_ns(
        controller_state.current_step_ns,
        controller_error,
        accepted=accepted,
        config=controller_config,
        minimum_step_ns=minimum_step_ns,
        maximum_step_ns=maximum_step_ns,
    )
    scale = next_step_ns / controller_state.current_step_ns
    rider_guess = 2.0 * trial.refined.pair.rider.proper_step_ns * scale
    driver_guess = 2.0 * trial.refined.pair.driver.proper_step_ns * scale

    committed_rows = None
    next_intrinsic_spin_history = intrinsic_spin_reduction_history
    next_causal_c5_history = causal_c5_source_history
    if accepted:
        if build_intrinsic_spin_reduction_candidate is not None:
            if intrinsic_spin_reduction_history is None:  # pragma: no cover
                raise RuntimeError("validated intrinsic-spin history is missing")
            candidate = build_intrinsic_spin_reduction_candidate(
                trial,
                intrinsic_spin_reduction_history,
            )
            if not isinstance(candidate, AcceptedPairIntrinsicSpinReductionHistory):
                raise TypeError(
                    "intrinsic-spin candidate builder must return accepted pair "
                    "history"
                )
            next_intrinsic_spin_history = candidate
        if causal_c5_source_history is not None:
            next_causal_c5_history = build_accepted_pair_causal_c5_candidate(
                trial,
                causal_c5_source_history,
            )
        committed_rows = commit_accepted_exact_pair_step_doubling_trial(
            trial,
            rider_builder=rider_builder,
            driver_builder=driver_builder,
        )
    next_state = AdaptivePairControllerState(
        current_step_ns=next_step_ns,
        rider_proper_step_guess_ns=rider_guess,
        driver_proper_step_guess_ns=driver_guess,
        accepted_slabs=controller_state.accepted_slabs + int(accepted),
        rejected_trials=controller_state.rejected_trials + int(not accepted),
    )
    return AdaptivePairAttempt(
        trial=trial,
        controller_state=next_state,
        committed_rows=committed_rows,
        intrinsic_spin_reduction_history=next_intrinsic_spin_history,
        causal_c5_source_history=next_causal_c5_history,
    )


def run_exact_pair_adaptive_window(
    *,
    rider_builder: GrowableTrajectoryBuilder,
    driver_builder: GrowableTrajectoryBuilder,
    advance_rider: AdvanceRoleTrial,
    advance_driver: AdvanceRoleTrial,
    controller_state: AdaptivePairControllerState,
    controller_config: StepControllerConfig,
    tolerances: StepDoublingTolerances,
    target_time_ns: float,
    minimum_step_ns: float,
    maximum_step_ns: float,
    maximum_attempts: int,
    maximum_accepted_slabs: int,
    public_sample_interval_ns: float,
    magnetic_dipole: MagneticDipoleConfig,
    include_dipole_source: bool,
    public_output_state: AdaptivePairPublicOutputState | None = None,
    checkpoint_store: _AcceptedPairCheckpoint | None = None,
    spin_interpolation_model: str = "causal_frozen_c1",
    absolute_time_tolerance_ns: float = 1.0e-18,
    relative_time_tolerance: float = 1.0e-12,
    record_attempt_diagnostics: bool = False,
    cancel_callback: Callable[[], bool] | None = None,
    accepted_progress_callback: Callable[[float, float], None] | None = None,
    intrinsic_spin_reduction_history: (
        AcceptedPairIntrinsicSpinReductionHistory | None
    ) = None,
    build_intrinsic_spin_reduction_candidate: (
        IntrinsicSpinReductionCandidate | None
    ) = None,
    causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None,
) -> AdaptivePairRunResult:
    """Advance accepted pair history to a bounded shared lab-time target.

    The final proposed slab is clipped to ``target_time_ns``.  A rejected trial
    publishes neither source-history rows nor public-output rows. Public output
    is a list of accepted row indices; no interpolation or state mutation is
    performed for output sampling. An optional variable-length pair checkpoint
    is written only after accepted pair commits.

    The public exact-pair production surface selects this substrate only after
    enforcing its one-particle, exact-history, checkpoint, and scheduler guards.
    """

    target_time_ns = float(target_time_ns)
    minimum_step_ns = float(minimum_step_ns)
    maximum_step_ns = float(maximum_step_ns)
    public_sample_interval_ns = float(public_sample_interval_ns)
    maximum_attempts = int(maximum_attempts)
    maximum_accepted_slabs = int(maximum_accepted_slabs)
    scalar_values = (
        target_time_ns,
        minimum_step_ns,
        maximum_step_ns,
        public_sample_interval_ns,
        absolute_time_tolerance_ns,
        relative_time_tolerance,
    )
    if not all(np.isfinite(value) for value in scalar_values):
        raise ValueError("adaptive pair run controls must be finite")
    if minimum_step_ns <= 0.0 or maximum_step_ns < minimum_step_ns:
        raise ValueError("adaptive pair run step bounds are invalid")
    if not minimum_step_ns <= controller_state.current_step_ns <= maximum_step_ns:
        raise ValueError("initial adaptive pair step is outside the declared bounds")
    if public_sample_interval_ns <= 0.0:
        raise ValueError("public sample interval must be positive")
    if maximum_attempts < 1 or maximum_accepted_slabs < 1:
        raise ValueError("adaptive pair run limits must be positive")
    if absolute_time_tolerance_ns < 0.0 or relative_time_tolerance < 0.0:
        raise ValueError("adaptive pair time tolerances must be non-negative")
    if absolute_time_tolerance_ns == 0.0 and relative_time_tolerance == 0.0:
        raise ValueError("at least one adaptive pair time tolerance must be positive")
    if (intrinsic_spin_reduction_history is None) != (
        build_intrinsic_spin_reduction_candidate is None
    ):
        raise ValueError(
            "intrinsic-spin history and candidate builder must be supplied together"
        )

    latest_slab_scale_ns = _latest_pair_slab_scale_ns(
        rider_builder,
        driver_builder,
    )
    current_time = _accepted_pair_time_ns(
        rider_builder,
        driver_builder,
        tolerance_ns=(
            2.0
            * (
                absolute_time_tolerance_ns
                + relative_time_tolerance * latest_slab_scale_ns
            )
        ),
    )
    requested_window_ns = max(0.0, target_time_ns - current_time)
    completion_tolerance = (
        absolute_time_tolerance_ns + relative_time_tolerance * requested_window_ns
    )
    if target_time_ns < current_time - completion_tolerance:
        raise ValueError("adaptive pair target time precedes accepted history")

    if public_output_state is None:
        output_state = _initial_public_output_state(
            accepted_rows=rider_builder.accepted_steps,
            current_time_ns=current_time,
            sample_interval_ns=public_sample_interval_ns,
        )
    else:
        output_state = public_output_state
        if output_state.sample_interval_ns != public_sample_interval_ns:
            raise ValueError("public sample interval conflicts with restored cursor")
        if output_state.selected_rows[-1] >= rider_builder.accepted_steps:
            raise ValueError(
                "public output cursor references an unavailable history row"
            )
        if output_state.next_sample_time_ns <= current_time - completion_tolerance:
            raise ValueError("public output cursor precedes accepted history")

    attempts = 0
    accepted_slabs = 0
    rejected_trials = 0
    attempt_diagnostics: list[AdaptivePairAttemptDiagnostics] = []
    state = controller_state
    reduction_history = intrinsic_spin_reduction_history
    c5_history = causal_c5_source_history
    completed = target_time_ns - current_time <= completion_tolerance

    def flush_interrupted_checkpoint() -> None:
        if checkpoint_store is None:
            return
        checkpoint_store.write(
            rider=rider_builder.build_current(),
            driver=driver_builder.build_current(),
            controller_state=state.to_checkpoint_state(),
            public_output_state=output_state.to_checkpoint_state(),
            intrinsic_spin_reduction_state=(
                None
                if reduction_history is None
                else reduction_history.to_checkpoint_payload()
            ),
            causal_c5_source_history=c5_history,
            complete=False,
        )

    while not completed:
        from .integration_runner import IntegrationCancelled

        if cancel_callback is not None and cancel_callback():
            flush_interrupted_checkpoint()
            raise IntegrationCancelled("Integration cancelled by caller.")
        if attempts >= maximum_attempts:
            raise SharedLabTimeError(
                "adaptive pair window exhausted its maximum trial attempts"
            )
        if accepted_slabs >= maximum_accepted_slabs:
            raise SharedLabTimeError(
                "adaptive pair window exhausted its maximum accepted slabs"
            )

        remaining = target_time_ns - current_time
        attempted_step = min(state.current_step_ns, remaining)
        clipped = attempted_step < state.current_step_ns
        attempt_state = state
        if clipped:
            scale = attempted_step / state.current_step_ns
            attempt_state = replace(
                state,
                current_step_ns=attempted_step,
                rider_proper_step_guess_ns=(state.rider_proper_step_guess_ns * scale),
                driver_proper_step_guess_ns=(state.driver_proper_step_guess_ns * scale),
            )
        attempt_minimum = min(minimum_step_ns, attempted_step)
        try:
            result = attempt_exact_pair_adaptive_step(
                rider_builder=rider_builder,
                driver_builder=driver_builder,
                advance_rider=advance_rider,
                advance_driver=advance_driver,
                controller_state=attempt_state,
                controller_config=controller_config,
                tolerances=tolerances,
                minimum_step_ns=attempt_minimum,
                maximum_step_ns=maximum_step_ns,
                magnetic_dipole=magnetic_dipole,
                include_dipole_source=include_dipole_source,
                spin_interpolation_model=spin_interpolation_model,
                absolute_time_tolerance_ns=absolute_time_tolerance_ns,
                relative_time_tolerance=relative_time_tolerance,
                intrinsic_spin_reduction_history=reduction_history,
                build_intrinsic_spin_reduction_candidate=(
                    build_intrinsic_spin_reduction_candidate
                ),
                causal_c5_source_history=c5_history,
            )
        except IntegrationCancelled:
            flush_interrupted_checkpoint()
            raise
        attempts += 1
        state = result.controller_state
        reduction_history = result.intrinsic_spin_reduction_history
        c5_history = result.causal_c5_source_history
        if record_attempt_diagnostics:
            assessment = result.trial.assessment
            attempt_diagnostics.append(
                AdaptivePairAttemptDiagnostics(
                    attempted_step_ns=attempted_step,
                    accepted=result.accepted,
                    normalized_error=assessment.normalized_error,
                    position_error=assessment.position_error,
                    mechanical_momentum_error=assessment.mechanical_momentum_error,
                    rest_spin_error=assessment.rest_spin_error,
                    diagnostics_error=assessment.diagnostics_error,
                    position_error_index=assessment.position_error_index,
                    mechanical_momentum_error_index=(
                        assessment.mechanical_momentum_error_index
                    ),
                    rest_spin_error_index=assessment.rest_spin_error_index,
                    diagnostics_error_index=assessment.diagnostics_error_index,
                )
            )
        if not result.accepted:
            rejected_trials += 1
            shrink_tolerance = np.finfo(np.float64).eps * max(
                attempted_step, attempt_minimum
            )
            if state.current_step_ns >= attempted_step - shrink_tolerance:
                assessment = result.trial.assessment
                raise SharedLabTimeError(
                    "adaptive pair trial was rejected at the minimum usable step; "
                    f"normalized error={assessment.normalized_error:.6e} "
                    f"(position={assessment.position_error:.6e}, "
                    f"momentum={assessment.mechanical_momentum_error:.6e}, "
                    f"spin={assessment.rest_spin_error:.6e}, "
                    f"diagnostics={assessment.diagnostics_error:.6e})"
                )
            continue

        if result.committed_rows is None:  # pragma: no cover - property invariant
            raise RuntimeError("accepted adaptive pair result has no committed rows")
        accepted_slabs += 1
        current_time = _accepted_pair_time_ns(
            rider_builder,
            driver_builder,
            tolerance_ns=(
                2.0
                * (
                    absolute_time_tolerance_ns
                    + relative_time_tolerance * attempted_step
                )
            ),
        )
        completed = target_time_ns - current_time <= completion_tolerance
        if current_time > target_time_ns + completion_tolerance:
            raise SharedLabTimeError("adaptive pair window overshot its target time")
        output_state = _select_public_rows(
            output_state,
            committed_rows=result.committed_rows,
            rider_builder=rider_builder,
            time_tolerance_ns=completion_tolerance,
            include_final_row=completed,
        )
        if checkpoint_store is not None and checkpoint_store.due(
            rider_builder.accepted_steps,
            force=completed,
        ):
            checkpoint_store.write(
                rider=rider_builder.build_current(),
                driver=driver_builder.build_current(),
                controller_state=state.to_checkpoint_state(),
                public_output_state=output_state.to_checkpoint_state(),
                intrinsic_spin_reduction_state=(
                    None
                    if reduction_history is None
                    else reduction_history.to_checkpoint_payload()
                ),
                causal_c5_source_history=c5_history,
                complete=completed,
            )
        if accepted_progress_callback is not None:
            accepted_progress_callback(current_time, target_time_ns)

    return AdaptivePairRunResult(
        controller_state=state,
        public_output_state=output_state,
        attempts=attempts,
        accepted_slabs=accepted_slabs,
        rejected_trials=rejected_trials,
        final_time_ns=current_time,
        completed=completed,
        attempt_diagnostics=tuple(attempt_diagnostics),
        intrinsic_spin_reduction_history=reduction_history,
        causal_c5_source_history=c5_history,
    )


__all__ = [
    "AdaptivePairAttempt",
    "AdaptivePairAttemptDiagnostics",
    "AdaptivePairControllerState",
    "AdaptivePairPublicOutputState",
    "AdaptivePairRunResult",
    "IntrinsicSpinReductionCandidate",
    "attempt_exact_pair_adaptive_step",
    "run_exact_pair_adaptive_window",
]
