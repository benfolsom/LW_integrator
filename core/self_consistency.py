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

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional

from .types import (
    ChronoMatchingMode,
    GammaReconciliationMethod,
    ParticleState,
    StartupMode,
    Trajectory,
)

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
        Optional[Any],
    ],
    ParticleState,
]


@lru_cache(maxsize=None)
def _signature_parameters(step_function: StepFunction):
    return inspect.signature(step_function).parameters


def canonicalize_self_consistency_mode(mode: object) -> str:
    """Return the maintained self-consistency mode name."""

    mode_str = str(mode)
    aliases = {
        "mass_shell_only": "fixed_geometry",
        "full_iteration": "variable_geometry",
    }
    return aliases.get(mode_str, mode_str)


@dataclass
class SelfConsistencyConfig:
    """Configuration for self-consistency iterations.

    Self-consistency is now ENABLED BY DEFAULT to prevent energy jumps and
    numerical instabilities in relativistic simulations.

    The iterations occur WITHIN the force calculation loop for each particle,
    ensuring both the relativistic mass-shell constraint and gamma consistency
    are satisfied.

    CONVERGENCE STRATEGY:
    Two convergence modes with distinct behaviors:

    1. "fixed_geometry":
       - Fixed geometry (positions, retarded distances computed once)
       - Pt projected onto mass shell each iteration
       - One-way mass-shell convergence check
       - Fastest, use for most cases

    2. "variable_geometry":
       - Variable geometry (positions/distances recomputed each iteration)
       - Pt projected onto mass shell each iteration
       - One-way mass-shell convergence check
       - More accurate when particle moves significantly

    CHRONO-MATCH INTERPOLATION:
    When computing retarded times for Liénard-Wiechert fields, the code searches
    backward through the source particle trajectory to find t_ret = t_obs - R/c.
    With coarse timesteps, the "nearest" match may have significant time residual.
    Interpolation blends adjacent trajectory points when residual exceeds tolerance.

    Attributes
    ----------
    enabled : bool
        Whether to perform self-consistency iterations. Default is True.
    convergence_mode : str
        Convergence mode determining iteration strategy. Options:
        - "fixed_geometry": Pt projection, fixed geometry (default, fastest)
        - "variable_geometry": Pt projection, variable geometry (accurate, slower)

        Differences:
        - fixed_geometry: Geometry computed once, Pt projected, one-way check
        - variable_geometry: Geometry recomputed each iteration, Pt projected, one-way check

        Historical aliases are still normalized when loading older configs.
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
    chrono_interpolate : bool
        Enable interpolation in chrono-matching when time residual exceeds tolerance.
        When True, source-particle quantities (velocity, acceleration, gamma) are
        linearly interpolated between bracketing trajectory indices.
        Default is False (use nearest discrete sample).
    chrono_tolerance : float
        Time residual tolerance for chrono-matching, in nanoseconds.
        If |t_matched - t_target| > chrono_tolerance, interpolation is applied
        (if chrono_interpolate=True) or a warning is issued (if verbosity >= 2).
        Default is 1e-3 ns (1 picosecond).
    chrono_matching_mode : str
        Chrono-matching algorithm mode. Options:
        - "FAST": Single-sample delay Δt = R(1 + β·n̂)/c (default)
        - "AVERAGED": Reserved for APPROXIMATE_BACK_HISTORY startup mode only.
          Not recommended for general use until fully validated (~2-5× slower).
        Default is "FAST".
    chrono_high_precision : bool
        Enable high-precision chrono-matching features. When True:
        - Uses cubic (Catmull-Rom) interpolation instead of linear
        - Interpolates particle positions (x/y/z) in addition to velocities
        - Provides smoother derivatives for acceleration terms
        Adds ~3-5% overhead. Useful for γ > 1000. Default is False.
    chrono_adaptive_tolerance : bool
        Automatically set chrono_tolerance = 0.1 × timestep_h. When True,
        overrides the manual chrono_tolerance setting and scales with the
        integration timestep. Useful for variable-timestep simulations.
        Default is False (use fixed tolerance).

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

    Aggressive convergence for ultra-relativistic particles::

        config = SelfConsistencyConfig(
            convergence_mode="variable_geometry",
            target_ms_tolerance=1e-8,
            mass_shell_tolerance=1e-3,
            mass_shell_relaxation=1.0,
            max_iterations=20,
            verbosity=2,
        )

    Disable for testing/comparison::

        config = SelfConsistencyConfig(enabled=False)
    """

    enabled: bool = True
    convergence_mode: str = "fixed_geometry"  # "fixed_geometry" or "variable_geometry"
    target_ms_tolerance: float = 1e-6  # Mass-shell loop convergence criterion
    mass_shell_tolerance: float = 1e-2  # Safety net after loop
    mass_shell_relaxation: float = 0.7  # Relaxation weight applied after correction

    # Gamma reconciliation parameters
    gamma_reconciliation_method: GammaReconciliationMethod = (
        GammaReconciliationMethod.DISABLED
    )
    gamma_reconciliation_low_beta_threshold: float = 0.9  # Below this: trust energy
    gamma_reconciliation_high_beta_threshold: float = 0.99  # Above this: trust velocity
    gamma_reconciliation_low_beta_weight: float = 0.8  # α for β < low threshold
    gamma_reconciliation_high_beta_weight: float = 0.2  # α for β > high threshold
    gamma_reconciliation_mid_beta_weight: float = 0.5  # α for mid range
    gamma_reconciliation_fixed_weight: float = 0.5  # α for FIXED_WEIGHTED method

    chrono_interpolate: bool = False  # Enable chrono-match interpolation
    chrono_tolerance: float = 1e-3  # Time residual tolerance (ns)
    chrono_matching_mode: str = (
        "FAST"  # "FAST" or "AVERAGED" (AVERAGED for APPROXIMATE_BACK_HISTORY only)
    )
    chrono_high_precision: bool = False  # Enable cubic + position interpolation
    chrono_adaptive_tolerance: bool = False  # Auto-set tolerance = 0.1 × timestep
    max_iterations: int = 10  # Maximum SC iterations per particle per step
    verbosity: int = 0

    def __post_init__(self):
        """Normalize mode name using historical aliases."""
        object.__setattr__(
            self,
            "convergence_mode",
            canonicalize_self_consistency_mode(self.convergence_mode),
        )

    @classmethod
    def standard(cls) -> "SelfConsistencyConfig":
        """Return standard configuration for typical relativistic simulations.

        This is the default configuration: enabled with fixed-geometry
        self-consistency. Suitable for most high-energy particle tracking
        applications.
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

        Use only for testing, benchmarking, or controlled reference comparisons.
        Not recommended for production simulations.
        """
        return cls(enabled=False)

    @classmethod
    def aggressive(cls) -> "SelfConsistencyConfig":
        """Return aggressive configuration for maximum numerical stability.

        Uses tight convergence tolerances and more iterations to prevent
        energy jumps in challenging scenarios (ultra-relativistic particles,
        narrow apertures, or close approaches to conducting boundaries).

        Uses variable-geometry mode for maximum accuracy.
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
    def variable_geometry(cls, tolerance: float = 1e-6) -> "SelfConsistencyConfig":
        """Return variable-geometry iteration configuration.

        Recomputes positions, retarded distances, and forces at each SC iteration.
        Most accurate but computationally expensive. Use when lighter iteration modes
        fail to converge or when geometric changes during the timestep are significant.

        Parameters
        ----------
        tolerance : float
            Target tolerance for both mass-shell and gamma convergence.

        Returns
        -------
        SelfConsistencyConfig
            Configuration using variable-geometry mode.
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
    space_charge: Optional[Any] = None,
    cancel_callback: Optional[Any] = None,
    traj_soa: Optional[Any] = None,
    traj_ext_soa: Optional[Any] = None,
    radiation_reaction_mode: Optional[str] = "off",
    external_field: Optional[Any] = None,
    pseudo_grid_space_charge_source_charges: Optional[Any] = None,
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
    cancel_callback : Optional[callable]
        Optional predicate to check for cancellation. If provided and returns True,
        the equations of motion should raise IntegrationCancelled.

    Returns
    -------
    ParticleState
        Updated particle state for the next time step. If self-consistency is
        enabled, each particle in this state has been iteratively refined until
        gamma converged.
    """

    # Check whether step_function accepts SOA keyword arguments
    _sig_params = _signature_parameters(step_function)
    _accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in _sig_params.values()
    )
    _accepts_soa = "traj_soa" in _sig_params or _accepts_var_kwargs

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
        cancel_callback,
        **({"space_charge": space_charge} if space_charge is not None else {}),
        **(
            {
                "pseudo_grid_space_charge_source_charges": (
                    pseudo_grid_space_charge_source_charges
                )
            }
            if pseudo_grid_space_charge_source_charges is not None
            and (
                "pseudo_grid_space_charge_source_charges" in _sig_params
                or _accepts_var_kwargs
            )
            else {}
        ),
        **(
            {"external_field": external_field}
            if external_field is not None
            and ("external_field" in _sig_params or _accepts_var_kwargs)
            else {}
        ),
        **({"traj_soa": traj_soa} if _accepts_soa and traj_soa is not None else {}),
        **(
            {"traj_ext_soa": traj_ext_soa}
            if _accepts_soa and traj_ext_soa is not None
            else {}
        ),
        **(
            {"radiation_reaction_mode": radiation_reaction_mode}
            if "radiation_reaction_mode" in _sig_params or _accepts_var_kwargs
            else {}
        ),
    )

    return result


__all__ = [
    "SelfConsistencyConfig",
    "canonicalize_self_consistency_mode",
    "self_consistent_step",
]
