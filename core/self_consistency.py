"""Optional self-consistency checks for Liénard–Wiechert integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .types import ChronoMatchingMode, ParticleState, Trajectory

StepFunction = Callable[
    [float, Trajectory, Trajectory, int, float, Any, ChronoMatchingMode],
    ParticleState,
]


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency iterations."""

    enabled: bool = False
    tolerance: float = 1e-6
    max_iterations: int = 3
    debug: bool = False


def self_consistent_step(
    step_function: StepFunction,
    h_step: float,
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    index_traj: int,
    aperture_radius: float,
    sim_type: Any,
    config: Optional[SelfConsistencyConfig],
    chrono_mode: ChronoMatchingMode,
) -> ParticleState:
    """Optionally refine an integration step until the Lorentz factor converges.

    The provided ``step_function`` is executed repeatedly using the latest
    candidate state until the relative change in ``γ`` falls below the
    tolerance defined in ``config`` or the maximum number of iterations is
    reached. ``chrono_mode`` is forwarded to the supplied ``step_function`` so
    that chrono-matching can either follow the legacy fast path or the new
    averaged sampling strategy.
    """

    result = step_function(
        h_step,
        trajectory,
        trajectory_ext,
        index_traj,
        aperture_radius,
        sim_type,
        chrono_mode,
    )

    if config is None or not config.enabled:
        return result

    candidate = result
    max_rel_change = 0.0
    for iteration in range(config.max_iterations):
        mutable_traj = list(trajectory)
        next_index = index_traj + 1
        if next_index < len(mutable_traj):
            mutable_traj[next_index] = candidate
        else:
            mutable_traj.append(candidate)

        refined = step_function(
            h_step,
            mutable_traj,
            trajectory_ext,
            index_traj,
            aperture_radius,
            sim_type,
            chrono_mode,
        )

        gamma_prev = np.asarray(candidate.get("gamma", np.array([])))
        gamma_new = np.asarray(refined.get("gamma", np.array([])))
        if gamma_prev.size == 0 or gamma_new.size == 0:
            candidate = refined
            break

        denom = np.where(np.abs(gamma_prev) < 1e-12, 1e-12, np.abs(gamma_prev))
        max_rel_change = float(np.max(np.abs((gamma_new - gamma_prev) / denom)))
        if max_rel_change < config.tolerance:
            if config.debug:
                print(
                    f"Self-consistency converged in {iteration + 1} iterations (Δγ={max_rel_change:.3e})"
                )
            candidate = refined
            break

        candidate = refined
    else:
        if config.debug:
            print(
                f"Warning: Self-consistency did not converge in {config.max_iterations} iterations (Δγ={max_rel_change:.3e})"
            )

    return candidate


__all__ = ["SelfConsistencyConfig", "self_consistent_step"]
