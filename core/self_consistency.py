"""Optional self-consistency checks for Liénard–Wiechert integration.

Self-consistency iterations refine each integration step by repeatedly evaluating
the equations of motion until the Lorentz factor (gamma) converges. This helps
prevent numerical instabilities and energy jumps in relativistic simulations,
especially near conducting boundaries or during close particle approaches.

Enable self-consistency checks when:
- Simulating high-energy particles (gamma > 10)
- Using small time steps or narrow apertures
- Observing unexpected energy jumps or divergences
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .types import ChronoMatchingMode, ParticleState, StartupMode, Trajectory

StepFunction = Callable[
    [
        float,
        Trajectory,
        Trajectory,
        int,
        float,
        Any,
        ChronoMatchingMode,
        StartupMode,
    ],
    ParticleState,
]


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency iterations.

    Self-consistency is now ENABLED BY DEFAULT to prevent energy jumps and
    numerical instabilities in relativistic simulations.

    Attributes
    ----------
    enabled : bool
        Whether to perform self-consistency iterations. Default is True.
    tolerance : float
        Relative convergence tolerance for gamma. Iterations stop when
        max|Δγ/γ| < tolerance. Default is 1e-6.
    max_iterations : int
        Maximum number of refinement iterations per step. Default is 5.
    debug : bool
        If True, print convergence information for each step. Default is False.

    Examples
    --------
    Standard configuration (default)::

        config = SelfConsistencyConfig()
        # enabled=True, tolerance=1e-6, max_iterations=5

    Disable for testing/comparison::

        config = SelfConsistencyConfig(enabled=False)

    Aggressive convergence for stability::

        config = SelfConsistencyConfig(
            tolerance=1e-8,
            max_iterations=10,
            debug=True
        )
    """

    enabled: bool = True
    tolerance: float = 1e-6
    max_iterations: int = 5
    debug: bool = False

    @classmethod
    def standard(cls) -> "SelfConsistencyConfig":
        """Return standard configuration for typical relativistic simulations.

        This is the default configuration: enabled with moderate convergence
        criteria suitable for most high-energy particle tracking applications.
        """
        return cls(enabled=True, tolerance=1e-6, max_iterations=5)

    @classmethod
    def disabled(cls) -> "SelfConsistencyConfig":
        """Return configuration with self-consistency disabled.

        Use only for testing, benchmarking, or comparison with legacy code.
        Not recommended for production simulations.
        """
        return cls(enabled=False)

    @classmethod
    def aggressive(cls) -> "SelfConsistencyConfig":
        """Return aggressive configuration for maximum numerical stability.

        Uses tight convergence tolerance and more iterations to prevent
        energy jumps in challenging scenarios (ultra-relativistic particles,
        narrow apertures, or close approaches to conducting boundaries).
        """
        return cls(enabled=True, tolerance=1e-8, max_iterations=10, debug=False)


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
    startup_mode: StartupMode,
) -> ParticleState:
    """Optionally refine an integration step until the Lorentz factor converges.

    The provided ``step_function`` is executed repeatedly using the latest
    candidate state until the relative change in ``γ`` falls below the
    tolerance defined in ``config`` or the maximum number of iterations is
    reached. ``chrono_mode`` and ``startup_mode`` are forwarded to the supplied
    ``step_function`` so that chrono-matching and early-step behaviour follow
    the requested strategies.
    """

    result = step_function(
        h_step,
        trajectory,
        trajectory_ext,
        index_traj,
        aperture_radius,
        sim_type,
        chrono_mode,
        startup_mode,
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
            startup_mode,
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
