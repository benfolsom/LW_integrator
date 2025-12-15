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
    ensuring both the relativistic mass-shell constraint and gamma consistency
    are satisfied.

    CONVERGENCE STRATEGY:
    Both convergence criteria must be satisfied in all modes:
    1. Mass-shell error: |Pt² - P² - (mc)²|/(mc)² < target_ms_tolerance
    2. Gamma consistency: |γ_velocity - γ_energy| / γ < target_gamma_tolerance

    The mode determines HOW Pt is corrected during iterations, not what is checked.

    Attributes
    ----------
    enabled : bool
        Whether to perform self-consistency iterations. Default is True.
    convergence_mode : str
        Pt correction mode during iterations. Options:
        - "mass_shell_only": Project Pt onto mass shell with relaxation (default)
        - "dual_weighted": Blend mass-shell and velocity-based Pt, then apply relaxation
        Both modes check both convergence criteria.
        Default is "mass_shell_only".
    target_ms_tolerance : float
        TARGET mass-shell convergence criterion used inside the iteration loop.
        Iterations continue until |Pt² - P² - (mc)²|/(mc)² < target_ms_tolerance.
        Default is 1e-6 (0.0001%).
    target_gamma_tolerance : float
        TARGET gamma consistency criterion used inside the iteration loop.
        Iterations continue until |γ_velocity - γ_energy| / γ < target_gamma_tolerance.
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
    dual_weight : float
        Blending weight between mass-shell and velocity-based Pt (dual_weighted mode only).
        Pt_blend = w*Pt_mass_shell + (1-w)*Pt_velocity where w = dual_weight.
        - 1.0 = pure mass-shell (equivalent to mass_shell_only)
        - 0.5 = equal weighting (default, balanced)
        - 0.0 = pure velocity-based (kinematic only)
        Default is 0.5. Ignored in mass_shell_only mode.
    max_iterations : int
        Maximum number of refinement iterations per particle per step. Default is 10.
        Increased from 5 to accommodate dual-criterion convergence.
    verbosity : int
        Verbosity level for convergence information. Default is 0.
        0 = silent (no output)
        1 = summary (one line per step: converged/failed with final errors)
        2 = failures only (detailed output only for non-converged steps)
        3 = full detail (iteration-by-iteration for all steps, very large logs)

    Examples
    --------
    Standard configuration (default, mass-shell only)::

        config = SelfConsistencyConfig()
        # enabled=True, convergence_mode="mass_shell_only"
        # target_ms_tolerance=1e-6, target_gamma_tolerance=1e-6
        # mass_shell_relaxation=0.7, max_iterations=10

    Dual-weighted mode (blend velocity and mass-shell)::

        config = SelfConsistencyConfig(
            convergence_mode="dual_weighted",
            dual_weight=0.5,  # Equal weighting
            mass_shell_relaxation=0.7,
        )

    Aggressive convergence for ultra-relativistic particles::

        config = SelfConsistencyConfig(
            convergence_mode="mass_shell_only",
            target_ms_tolerance=1e-8,
            target_gamma_tolerance=1e-8,
            mass_shell_tolerance=1e-3,
            mass_shell_relaxation=1.0,  # Full projection
            max_iterations=15,
            verbosity=2
        )

    Disable for testing/comparison::

        config = SelfConsistencyConfig(enabled=False)
    """

    enabled: bool = True
    convergence_mode: str = "mass_shell_only"  # "mass_shell_only" or "dual_weighted"
    target_ms_tolerance: float = 1e-6  # Mass-shell loop convergence criterion
    target_gamma_tolerance: float = 1e-6  # Gamma loop convergence criterion
    mass_shell_tolerance: float = 1e-2  # Safety net after loop
    mass_shell_relaxation: float = 0.7  # Relaxation weight applied after correction
    dual_weight: float = 0.5  # Blending weight (dual_weighted mode only)
    max_iterations: int = 10  # Increased for dual criteria
    verbosity: int = 0

    @classmethod
    def standard(cls) -> "SelfConsistencyConfig":
        """Return standard configuration for typical relativistic simulations.

        This is the default configuration: enabled with mass-shell projection
        suitable for most high-energy particle tracking applications.
        """
        return cls(
            enabled=True,
            convergence_mode="mass_shell_only",
            target_ms_tolerance=1e-6,
            target_gamma_tolerance=1e-6,
            mass_shell_tolerance=1e-2,
            mass_shell_relaxation=0.7,
            dual_weight=0.5,
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
        """
        return cls(
            enabled=True,
            convergence_mode="mass_shell_only",
            target_ms_tolerance=1e-8,
            target_gamma_tolerance=1e-8,
            mass_shell_tolerance=1e-3,
            mass_shell_relaxation=1.0,  # Full projection for aggressive mode
            dual_weight=0.5,
            max_iterations=15,
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
