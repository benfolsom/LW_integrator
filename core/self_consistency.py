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
        Optional[int],
    ],
    ParticleState,
]


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency iterations.

    Self-consistency is now ENABLED BY DEFAULT to prevent energy jumps and
    numerical instabilities in relativistic simulations.

    The iterations occur WITHIN the force calculation loop for each particle,
    ensuring both the relativistic mass-shell constraint and gamma consistency
    are satisfied.

    CONVERGENCE STRATEGY:
    Three convergence modes with distinct behaviors:

    1. "fixed_geometry" (formerly "mass_shell_only"):
       - Fixed geometry (positions, retarded distances computed once)
       - Pt projected onto mass shell each iteration
       - One-way mass-shell convergence check
       - Fastest, use for most cases

    2. "variable_geometry" (formerly "full_iteration"):
       - Variable geometry (positions/distances recomputed each iteration)
       - Pt projected onto mass shell each iteration
       - One-way mass-shell convergence check
       - More accurate when particle moves significantly

    3. "bidirectional_search" (NEW):
       - Variable geometry (positions/distances recomputed each iteration)
       - Symmetric relaxation of BOTH Pt and P (no full projection)
       - Bidirectional convergence check (forward AND backward)
       - Exploratory mode, finds mutually consistent (P, Pt, geometry) state
       - Slowest, use for diagnostics or when standard modes fail

    Attributes
    ----------
    enabled : bool
        Whether to perform self-consistency iterations. Default is True.
    convergence_mode : str
        Convergence mode determining iteration strategy. Options:
        - "fixed_geometry": Pt projection, fixed geometry (default, fastest)
        - "variable_geometry": Pt projection, variable geometry (accurate, slower)
        - "bidirectional_search": Symmetric relaxation, variable geometry (exploratory, slowest)

        Differences:
        - fixed_geometry: Geometry computed once, Pt projected, one-way check
        - variable_geometry: Geometry recomputed each iteration, Pt projected, one-way check
        - bidirectional_search: Geometry recomputed, symmetric P+Pt relaxation, bidirectional check

        Legacy aliases supported: "mass_shell_only" → "fixed_geometry",
                                  "full_iteration" → "variable_geometry"

        Default is "fixed_geometry".
    target_ms_tolerance : float
        TARGET mass-shell convergence criterion used inside the iteration loop.
        Iterations continue until |Pt² - P² - (mc)²|/(mc)² < target_ms_tolerance.
        Default is 1e-6 (0.0001%).
    mass_shell_tolerance : float
        SAFETY NET threshold enforced after the loop. If the final mass-shell error
        exceeds this value, Pt is clamped to √(P² + (mc)²) as a fallback.
        Should be larger (looser) than target_ms_tolerance.
        Default is 1e-2 (1%).
    mass_shell_relaxation : float
        Relaxation weight applied after Pt correction (used in both modes).
        Pt_final = α*Pt_corrected + (1-α)*Pt_old where α = mass_shell_relaxation.
        - 1.0 = full correction (aggressive, fastest convergence)
        - 0.7 = recommended (good balance, default)
        - 0.5 = conservative (more stable, slower)
        Default is 0.7.

    max_iterations : int
        Maximum number of refinement iterations per particle per step. Default is 10.
        Increased from 5 to accommodate dual-criterion convergence.
    verbosity : int
        Verbosity level for convergence information. Default is 0.
        0 = silent (no output)
        1 = summary (one line per step: converged/failed with final errors)
        2 = failures only (detailed output only for non-converged steps)
        3 = full detail (iteration-by-iteration for all steps, very large logs)

    Notes on bidirectional_search mode
    -----------------------------------
    This mode uses symmetric relaxation on both Pt and P, which is non-standard:
    - Allows P to be modified (not just from forces)
    - Searches for mutually consistent (P, Pt, geometry) state
    - May find solutions when geometry/force errors accumulate
    - Trade-off: slightly corrupts force integration for global consistency
    - Use conservatively with relaxation weight ≈ 0.5-0.7
    - Experimental - validate results carefully

    Examples
    --------
    Standard configuration (default, fixed geometry)::

        config = SelfConsistencyConfig()
        # enabled=True, convergence_mode="fixed_geometry"
        # Pt projection with fixed geometry (fast)
        # target_ms_tolerance=1e-6, mass_shell_relaxation=0.7, max_iterations=10

    Variable geometry mode (high accuracy, updates geometry)::

        config = SelfConsistencyConfig(
            convergence_mode="variable_geometry",
            target_ms_tolerance=1e-6,
            mass_shell_relaxation=0.7,
            max_iterations=20,
        )

    Bidirectional search mode (exploratory, symmetric relaxation)::

        config = SelfConsistencyConfig(
            convergence_mode="bidirectional_search",
            target_ms_tolerance=1e-6,
            mass_shell_relaxation=0.5,  # Conservative for symmetric relax
            max_iterations=30,
            verbosity=2,  # Monitor convergence
        )

    Aggressive convergence for ultra-relativistic particles::

        config = SelfConsistencyConfig(
            convergence_mode="full_iteration",  # Variable geometry for accuracy
            target_ms_tolerance=1e-8,
            mass_shell_tolerance=1e-3,
            mass_shell_relaxation=1.0,  # Full projection
            max_iterations=20,
            verbosity=2
        )

    Disable for testing/comparison::

        config = SelfConsistencyConfig(enabled=False)
    """

    enabled: bool = True
    convergence_mode: str = "fixed_geometry"  # "fixed_geometry", "variable_geometry", or "bidirectional_search"
    target_ms_tolerance: float = 1e-6  # Mass-shell loop convergence criterion
    mass_shell_tolerance: float = 1e-2  # Safety net after loop
    mass_shell_relaxation: float = 0.7  # Relaxation weight applied after correction
    max_iterations: int = 10  # Maximum SC iterations per particle per step
    verbosity: int = 0

    # Mode name aliases for backwards compatibility
    _MODE_ALIASES = {
        "mass_shell_only": "fixed_geometry",
        "full_iteration": "variable_geometry",
    }

    def __post_init__(self):
        """Normalize mode name using aliases."""
        if self.convergence_mode in self._MODE_ALIASES:
            object.__setattr__(
                self, "convergence_mode", self._MODE_ALIASES[self.convergence_mode]
            )

    @classmethod
    def standard(cls) -> "SelfConsistencyConfig":
        """Return standard configuration for typical relativistic simulations.

        This is the default configuration: enabled with mass-shell projection
        and mass-shell-only convergence check (no velocity check).
        Suitable for most high-energy particle tracking applications.
        """
        return cls(
            enabled=True,
            convergence_mode="fixed_geometry",
            target_ms_tolerance=1e-6,
            mass_shell_tolerance=1e-2,
            mass_shell_relaxation=0.7,
            max_iterations=10,
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

        Uses tight convergence tolerances and more iterations to prevent
        energy jumps in challenging scenarios (ultra-relativistic particles,
        narrow apertures, or close approaches to conducting boundaries).

        Uses full iteration mode for maximum accuracy.
        """
        return cls(
            enabled=True,
            convergence_mode="variable_geometry",
            target_ms_tolerance=1e-8,
            mass_shell_tolerance=1e-3,
            mass_shell_relaxation=1.0,  # Full projection for aggressive mode
            max_iterations=20,
            verbosity=0,
        )

    @classmethod
    def full_iteration(cls, tolerance: float = 1e-6) -> "SelfConsistencyConfig":
        """Return full position/momentum iteration configuration.

        Recomputes positions, retarded distances, and forces at each SC iteration.
        Most accurate but computationally expensive. Use when light iteration modes
        fail to converge or when geometric changes during the timestep are significant.

        Parameters
        ----------
        tolerance : float
            Target tolerance for both mass-shell and gamma convergence.

        Returns
        -------
        SelfConsistencyConfig
            Configuration using full iteration mode.
        """
        return cls(
            enabled=True,
            convergence_mode="variable_geometry",
            target_ms_tolerance=tolerance,
            mass_shell_tolerance=1e-2,
            mass_shell_relaxation=0.7,
            max_iterations=20,
            verbosity=0,
        )

    @classmethod
    def bidirectional(cls, tolerance: float = 1e-6) -> "SelfConsistencyConfig":
        """Create a bidirectional search configuration.

        Uses symmetric relaxation to explore mutually consistent states.
        Experimental mode - use for diagnostics or when standard modes fail.

        Parameters
        ----------
        tolerance : float
            Target mass-shell tolerance for convergence criterion.

        Returns
        -------
        SelfConsistencyConfig
            Configuration with bidirectional search enabled.
        """
        return cls(
            enabled=True,
            convergence_mode="bidirectional_search",
            target_ms_tolerance=tolerance,
            mass_shell_tolerance=1e-2,
            mass_shell_relaxation=0.5,  # Conservative for symmetric relaxation
            max_iterations=30,  # More iterations needed
            verbosity=2,  # Monitor convergence
        )


__all__ = ["SelfConsistencyConfig", "self_consistent_step"]


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
    step_idx: Optional[int] = None,
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
    step_idx : Optional[int]
        Integration step number for context in error messages.

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
        config,
        step_idx,
    )

    return result


__all__ = ["SelfConsistencyConfig", "self_consistent_step"]
