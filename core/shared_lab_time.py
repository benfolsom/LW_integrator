"""Shared-lab-time substrate for the exact-retarded two-body return mode.

The production integrator advances each bunch by a proper-time increment.  A
future electron--proton return calculation instead needs both accepted source
events to land on one coordinate-time barrier.  This module solves the two
proper increments independently against immutable accepted histories and then
publishes the pair jointly.

It deliberately contains no integration-runner dispatch.  Trial callbacks
must be side-effect free: they may use temporary histories, but may not mutate
accepted source history, Medina state, caches, checkpoints, or public output.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .types import GrowableTrajectoryBuilder, ParticleState


class SharedLabTimeError(RuntimeError):
    """Raised when a proper-time endpoint cannot reach one lab-time barrier."""


@dataclass(frozen=True)
class ProperTimeEndpoint:
    """One accepted candidate at a requested lab-time barrier."""

    state: ParticleState
    proper_step_ns: float
    coordinate_time_ns: float
    residual_ns: float
    evaluations: int


@dataclass(frozen=True)
class SharedLabTimePair:
    """Two provisional endpoints synchronized to one coordinate time."""

    start_time_ns: float
    target_time_ns: float
    rider: ProperTimeEndpoint
    driver: ProperTimeEndpoint

    @property
    def synchronization_residual_ns(self) -> float:
        return abs(self.rider.coordinate_time_ns - self.driver.coordinate_time_ns)


AdvanceTrial = Callable[[float], ParticleState]


def _single_particle_time(state: ParticleState, role: str) -> float:
    if "t" not in state:
        raise SharedLabTimeError(f"{role} trial state has no coordinate time")
    values = np.asarray(state["t"], dtype=np.float64)
    if values.shape != (1,):
        raise SharedLabTimeError(
            f"{role} shared-lab-time mode currently requires exactly one particle"
        )
    value = float(values[0])
    if not np.isfinite(value):
        raise SharedLabTimeError(f"{role} trial coordinate time is not finite")
    return value


def solve_proper_step_to_lab_time(
    advance_trial: AdvanceTrial,
    *,
    role: str,
    start_time_ns: float,
    target_time_ns: float,
    initial_proper_step_ns: float,
    absolute_tolerance_ns: float = 1.0e-18,
    relative_tolerance: float = 1.0e-12,
    max_iterations: int = 64,
    max_bracket_expansions: int = 20,
    maximum_proper_step_ns: float = np.inf,
) -> ProperTimeEndpoint:
    """Solve ``t(tau + h) = target_time_ns`` with a bracketed secant method.

    The lower endpoint is the already accepted state at ``h=0`` and is never
    re-evaluated.  The upper endpoint expands geometrically until it brackets
    the target.  Secant proposals are kept away from the outer five percent of
    the bracket; bisection is used otherwise.  This preserves a monotone,
    auditable failure mode without assuming a constant Lorentz factor.
    """

    start_time_ns = float(start_time_ns)
    target_time_ns = float(target_time_ns)
    initial_proper_step_ns = float(initial_proper_step_ns)
    absolute_tolerance_ns = float(absolute_tolerance_ns)
    relative_tolerance = float(relative_tolerance)
    maximum_proper_step_ns = float(maximum_proper_step_ns)
    max_iterations = int(max_iterations)
    max_bracket_expansions = int(max_bracket_expansions)
    scalar_values = (
        start_time_ns,
        target_time_ns,
        initial_proper_step_ns,
        absolute_tolerance_ns,
        relative_tolerance,
    )
    if not all(np.isfinite(value) for value in scalar_values):
        raise ValueError("shared-lab-time solver inputs must be finite")
    if target_time_ns <= start_time_ns:
        raise ValueError("target_time_ns must be later than start_time_ns")
    if initial_proper_step_ns <= 0.0:
        raise ValueError("initial_proper_step_ns must be positive")
    if absolute_tolerance_ns < 0.0 or relative_tolerance < 0.0:
        raise ValueError("time tolerances must be non-negative")
    if absolute_tolerance_ns == 0.0 and relative_tolerance == 0.0:
        raise ValueError("at least one time tolerance must be positive")
    if max_iterations < 1 or max_bracket_expansions < 0:
        raise ValueError("iteration limits are invalid")
    if np.isnan(maximum_proper_step_ns) or maximum_proper_step_ns <= 0.0:
        raise ValueError("maximum_proper_step_ns must be positive")

    interval_ns = target_time_ns - start_time_ns
    tolerance_ns = absolute_tolerance_ns + relative_tolerance * interval_ns
    lower_h = 0.0
    lower_residual = -interval_ns
    upper_h = min(initial_proper_step_ns, maximum_proper_step_ns)
    evaluations = 0

    def evaluate(proper_step_ns: float) -> tuple[ParticleState, float, float]:
        nonlocal evaluations
        try:
            state = advance_trial(float(proper_step_ns))
        except Exception as exc:
            raise SharedLabTimeError(
                f"{role} trial failed at proper step {proper_step_ns:.17g} ns"
            ) from exc
        evaluations += 1
        endpoint_time = _single_particle_time(state, role)
        if endpoint_time <= start_time_ns:
            raise SharedLabTimeError(f"{role} trial did not advance coordinate time")
        return state, endpoint_time, endpoint_time - target_time_ns

    upper_state, upper_time, upper_residual = evaluate(upper_h)
    previous_upper_time = upper_time
    expansions = 0
    while upper_residual < 0.0:
        if expansions >= max_bracket_expansions or upper_h >= maximum_proper_step_ns:
            raise SharedLabTimeError(
                f"{role} could not bracket target lab time {target_time_ns:.17g} ns"
            )
        next_upper_h = min(2.0 * upper_h, maximum_proper_step_ns)
        if next_upper_h <= upper_h:
            raise SharedLabTimeError(
                f"{role} proper-time bracket cannot expand further"
            )
        upper_h = next_upper_h
        upper_state, upper_time, upper_residual = evaluate(upper_h)
        if upper_time <= previous_upper_time:
            raise SharedLabTimeError(
                f"{role} endpoint coordinate time is not monotone in proper time"
            )
        previous_upper_time = upper_time
        expansions += 1

    best_state = upper_state
    best_h = upper_h
    best_time = upper_time
    best_residual = upper_residual
    if abs(best_residual) <= tolerance_ns:
        return ProperTimeEndpoint(
            state=copy.deepcopy(best_state),
            proper_step_ns=best_h,
            coordinate_time_ns=best_time,
            residual_ns=best_residual,
            evaluations=evaluations,
        )

    for _ in range(max_iterations):
        denominator = upper_residual - lower_residual
        if denominator == 0.0 or not np.isfinite(denominator):
            candidate_h = 0.5 * (lower_h + upper_h)
        else:
            candidate_h = upper_h - upper_residual * ((upper_h - lower_h) / denominator)
            width = upper_h - lower_h
            inner_lower = lower_h + 0.05 * width
            inner_upper = upper_h - 0.05 * width
            if candidate_h <= inner_lower or candidate_h >= inner_upper:
                candidate_h = 0.5 * (lower_h + upper_h)

        candidate_state, candidate_time, candidate_residual = evaluate(candidate_h)
        if abs(candidate_residual) <= tolerance_ns:
            return ProperTimeEndpoint(
                state=copy.deepcopy(candidate_state),
                proper_step_ns=candidate_h,
                coordinate_time_ns=candidate_time,
                residual_ns=candidate_residual,
                evaluations=evaluations,
            )
        if not lower_residual < candidate_residual < upper_residual:
            raise SharedLabTimeError(
                f"{role} endpoint coordinate time is not monotone in proper time"
            )
        if abs(candidate_residual) < abs(best_residual):
            best_state = candidate_state
            best_h = candidate_h
            best_time = candidate_time
            best_residual = candidate_residual
        if candidate_residual < 0.0:
            lower_h = candidate_h
            lower_residual = candidate_residual
        else:
            upper_h = candidate_h
            upper_residual = candidate_residual

        if np.nextafter(lower_h, upper_h) >= upper_h:
            break

    raise SharedLabTimeError(
        f"{role} proper-time solve did not converge after {evaluations} trials; "
        f"best lab-time residual was {best_residual:.6e} ns at h={best_h:.6e} ns"
    )


def solve_shared_lab_time_pair(
    *,
    advance_rider: AdvanceTrial,
    advance_driver: AdvanceTrial,
    start_time_ns: float,
    delta_time_ns: float,
    rider_initial_proper_step_ns: float,
    driver_initial_proper_step_ns: float,
    absolute_tolerance_ns: float = 1.0e-18,
    relative_tolerance: float = 1.0e-12,
    max_iterations: int = 64,
    max_bracket_expansions: int = 20,
    maximum_proper_step_ns: float = np.inf,
) -> SharedLabTimePair:
    """Return two provisional $1+1$ endpoints at one target lab time."""

    delta_time_ns = float(delta_time_ns)
    if not np.isfinite(delta_time_ns) or delta_time_ns <= 0.0:
        raise ValueError("delta_time_ns must be finite and positive")
    target_time_ns = float(start_time_ns) + delta_time_ns
    rider = solve_proper_step_to_lab_time(
        advance_rider,
        role="rider",
        start_time_ns=start_time_ns,
        target_time_ns=target_time_ns,
        initial_proper_step_ns=rider_initial_proper_step_ns,
        absolute_tolerance_ns=absolute_tolerance_ns,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        max_bracket_expansions=max_bracket_expansions,
        maximum_proper_step_ns=maximum_proper_step_ns,
    )
    driver = solve_proper_step_to_lab_time(
        advance_driver,
        role="driver",
        start_time_ns=start_time_ns,
        target_time_ns=target_time_ns,
        initial_proper_step_ns=driver_initial_proper_step_ns,
        absolute_tolerance_ns=absolute_tolerance_ns,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        max_bracket_expansions=max_bracket_expansions,
        maximum_proper_step_ns=maximum_proper_step_ns,
    )
    tolerance_ns = float(absolute_tolerance_ns) + float(relative_tolerance) * abs(
        delta_time_ns
    )
    pair = SharedLabTimePair(
        start_time_ns=float(start_time_ns),
        target_time_ns=target_time_ns,
        rider=rider,
        driver=driver,
    )
    if pair.synchronization_residual_ns > 2.0 * tolerance_ns:
        raise SharedLabTimeError(
            "rider and driver endpoint times exceed the shared-barrier tolerance"
        )
    return pair


def commit_shared_lab_time_pair(
    pair: SharedLabTimePair,
    *,
    rider_builder: GrowableTrajectoryBuilder,
    driver_builder: GrowableTrajectoryBuilder,
    absolute_tolerance_ns: float = 1.0e-18,
    relative_tolerance: float = 1.0e-12,
) -> int:
    """Validate both endpoints, then append them at one accepted boundary."""

    if rider_builder.accepted_steps != driver_builder.accepted_steps:
        raise SharedLabTimeError("accepted rider and driver histories are misaligned")
    tolerance_ns = float(absolute_tolerance_ns) + float(relative_tolerance) * abs(
        pair.target_time_ns - pair.start_time_ns
    )
    rider_time = _single_particle_time(pair.rider.state, "rider")
    driver_time = _single_particle_time(pair.driver.state, "driver")
    if (
        abs(rider_time - pair.target_time_ns) > tolerance_ns
        or abs(driver_time - pair.target_time_ns) > tolerance_ns
        or abs(rider_time - driver_time) > 2.0 * tolerance_ns
    ):
        raise SharedLabTimeError(
            "provisional endpoints do not share the target lab time"
        )

    # Preflight both states and both geometric allocations before either row is
    # visible.  Once this point passes, normal validation and capacity-growth
    # failures cannot leave a half-published accepted pair.  Process-level
    # failures are recovered from the last atomic pair checkpoint.
    rider_builder.validate_append_step(pair.rider.state)
    driver_builder.validate_append_step(pair.driver.state)
    rider_builder.reserve_append_capacity()
    driver_builder.reserve_append_capacity()
    rider_row = rider_builder.append_step(pair.rider.state)
    driver_row = driver_builder.append_step(pair.driver.state)
    if rider_row != driver_row:
        raise RuntimeError("joint accepted-pair row indices diverged")
    return rider_row


__all__ = [
    "ProperTimeEndpoint",
    "SharedLabTimeError",
    "SharedLabTimePair",
    "commit_shared_lab_time_pair",
    "solve_proper_step_to_lab_time",
    "solve_shared_lab_time_pair",
]
