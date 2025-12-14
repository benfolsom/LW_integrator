"""Optional self-consistency checks for Liénard–Wiechert integration.

Self-consistency iterations refine each integration step by iterating within
the force calculation for each particle until the Lorentz factor (gamma)
converges. This solves the circular dependency where gamma depends on forces,
which in turn depend on gamma.

This implementation matches the original Gaussian self-consistent integrator
approach, where iteration occurs within the particle update loop rather than
at the trajectory level.

Enable self-consistency checks when:
- Simulating high-energy particles (gamma > 10)
- Using small time steps or narrow apertures
- Observing unexpected energy jumps or divergences
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

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
        Optional["SelfConsistencyConfig"],
    ],
    ParticleState,
]


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency iterations.

    Self-consistency is now ENABLED BY DEFAULT to prevent energy jumps and
    numerical instabilities in relativistic simulations.

    The iterations occur WITHIN the force calculation loop for each particle,
    solving the circular dependency between gamma and the forces that depend
    on gamma. This is the correct implementation matching the original
    Gaussian self-consistent integrator.

    Attributes
    ----------
    enabled : bool
        Whether to perform self-consistency iterations. Default is True.
    tolerance : float
        Relative convergence tolerance for gamma. Iterations stop when
        |Δγ/γ| < tolerance for each particle. Default is 1e-6.
    max_iterations : int
        Maximum number of refinement iterations per particle per step. Default is 5.
    mass_shell_tolerance : float
        Relative error threshold for mass-shell projection. If the relative
        mass-shell error |Pt² - P² - (mc)²|/(mc)² exceeds this tolerance,
        Pt will be clamped to √(P² + (mc)²) to enforce the constraint.
        Default is 1e-2 (1%). Set to a very large value (e.g., 1e10) to
        effectively disable mass-shell projection.
    verbosity : int
        Verbosity level for convergence information. Default is 0.
        0 = silent (no output)
        1 = basic (one line per particle showing convergence status)
        2 = detailed (full convergence details including all gamma values)

    Examples
    --------
    Standard configuration (default)::

        config = SelfConsistencyConfig()
        # enabled=True, tolerance=1e-6, max_iterations=5, mass_shell_tolerance=1e-2

    Disable for testing/comparison::

        config = SelfConsistencyConfig(enabled=False)

    Aggressive convergence for stability::

        config = SelfConsistencyConfig(
            tolerance=1e-8,
            max_iterations=10,
            mass_shell_tolerance=1e-3,
            verbosity=2
        )
    """

    enabled: bool = True
    tolerance: float = 1e-6
    max_iterations: int = 5
    mass_shell_tolerance: float = 1e-2
    verbosity: int = 0

    @classmethod
    def standard(cls) -> "SelfConsistencyConfig":
        """Return standard configuration for typical relativistic simulations.

        This is the default configuration: enabled with moderate convergence
        criteria suitable for most high-energy particle tracking applications.
        """
        return cls(
            enabled=True, tolerance=1e-6, max_iterations=5, mass_shell_tolerance=1e-2
        )

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
        return cls(
            enabled=True,
            tolerance=1e-8,
            max_iterations=10,
            mass_shell_tolerance=1e-3,
            verbosity=0,
        )


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
    """Execute a single integration step, optionally with self-consistency.

    This function now serves as a thin wrapper that passes the self-consistency
    configuration down to the equations of motion. The actual iterative
    refinement occurs WITHIN the force calculation loop for each particle,
    not at the trajectory level.

    This matches the original Gaussian self-consistent integrator design,
    where each particle's update iterates until gamma converges, solving
    the circular dependency between gamma and the forces.

    Parameters
    ----------
    step_function : StepFunction
        The equations of motion function to call. Must accept a
        self_consistency parameter as its final argument.
    h_step : float
        Time step for integration.
    trajectory : Trajectory
        Current trajectory history.
    trajectory_ext : Trajectory
        External/driver trajectory history.
    index_traj : int
        Current index in trajectory.
    aperture_radius : float
        Aperture radius for boundary conditions.
    sim_type : SimulationType
        Type of simulation (conducting wall, etc.).
    config : Optional[SelfConsistencyConfig]
        Self-consistency configuration. If None or disabled, no iteration occurs.
    chrono_mode : ChronoMatchingMode
        Retarded time matching mode.
    startup_mode : StartupMode
        Early-step handling mode.

    Returns
    -------
    ParticleState
        Updated particle state for the next time step. If self-consistency is
        enabled, each particle in this state has been iteratively refined until
        gamma converged.
    """

    # Simply call the step function and pass the config through
    # The iteration logic is now INSIDE retarded_equations_of_motion
    result = step_function(
        h_step,
        trajectory,
        trajectory_ext,
        index_traj,
        aperture_radius,
        sim_type,
        chrono_mode,
        startup_mode,
        config,  # Pass config to equations - iteration happens there
    )

    return result


__all__ = ["SelfConsistencyConfig", "self_consistent_step"]
