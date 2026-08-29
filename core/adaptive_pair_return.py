"""One transactional adaptive attempt for the exact-retarded return mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .exact_pair_trial import (
    AdvanceRoleTrial,
    ExactPairStepDoublingTrial,
    commit_accepted_exact_pair_step_doubling_trial,
    solve_exact_pair_step_doubling_trial,
)
from .shared_lab_time import SharedLabTimeError
from .step_doubling import (
    StepControllerConfig,
    StepDoublingTolerances,
    propose_next_step_ns,
)
from .types import GrowableTrajectoryBuilder, MagneticDipoleConfig


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

    @property
    def accepted(self) -> bool:
        return self.committed_rows is not None


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
) -> AdaptivePairAttempt:
    """Try one slab, committing only a healthy accepted two-half path.

    Medina caps are recoverable local-step failures and force a shrink. Other
    hard health failures indicate invalid state or solver semantics and abort
    rather than being hidden by repeated retries.
    """

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
    if accepted:
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
    )


__all__ = [
    "AdaptivePairAttempt",
    "AdaptivePairControllerState",
    "attempt_exact_pair_adaptive_step",
]
