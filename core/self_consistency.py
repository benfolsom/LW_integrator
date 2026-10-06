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
import warnings
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


SUPPORTED_CONVERGENCE_MODES = ("fixed_geometry", "variable_geometry")
MIN_ENABLED_MAX_ITERATIONS = 2


class SelfConsistencyOptionWarning(UserWarning):
    """Warn that a self-consistency/chrono option has no effect in this run."""


class ChronoHighPrecisionDeprecationWarning(FutureWarning):
    """``chrono_high_precision`` is deprecated and ignored by the SOA runner."""


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

    Self-consistency is ENABLED BY DEFAULT (fixed geometry, two iterations)
    in every entry point: this dataclass, the CLI, the GUI/testbed and sweeps.

    The iterations occur WITHIN the force calculation loop for each particle.
    Convergence is a one-way mass-shell check, evaluated from the second
    iteration onward; exhausting ``max_iterations`` raises
    ``SelfConsistencyNonConvergenceError``.

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
        If ``|t_matched - t_target|`` > chrono_tolerance, interpolation is applied
        (if chrono_interpolate=True) or a warning is issued (if verbosity >= 2).
        Default is 1e-3 ns (1 picosecond).
    chrono_matching_mode : str
        Recorded for configuration round-trips only; the solver does not read
        it. The retardation mode actually used is the separate ``chrono_mode``
        argument of :func:`core.integration_runner.retarded_integrator`
        (CLI ``chrono_mode``; testbed/GUI ``chrono_matching_mode`` option).
        Options: "FAST" (default) or "AVERAGED" (diagnostic).
    chrono_high_precision : bool
        Deprecated; accepted for saved configs but has no effect in the
        maintained runner, which always uses the structure-of-arrays chrono
        path (linear interpolation only). Only the legacy non-SOA fallback
        used by direct API calls implements cubic/position interpolation.
        Setting it emits :class:`ChronoHighPrecisionDeprecationWarning` at run
        start. Default is False.
    chrono_adaptive_tolerance : bool
        Automatically set chrono_tolerance = 0.1 × timestep_h. When True,
        overrides the manual chrono_tolerance setting and scales with the
        integration timestep. Useful for variable-timestep simulations.
        Default is False (use fixed tolerance).

    max_iterations : int
        Maximum number of refinement iterations per particle per step. Default is 2.
        The mass-shell check runs from the second iteration onward; exhausting
        the limit raises ``SelfConsistencyNonConvergenceError``. Values below 2
        are rejected when ``enabled`` is True.
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
        # target_ms_tolerance=1e-6, mass_shell_relaxation=0.7, max_iterations=2

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
    chrono_high_precision: bool = False  # Deprecated; ignored by the SOA runner
    chrono_adaptive_tolerance: bool = False  # Auto-set tolerance = 0.1 × timestep
    max_iterations: int = 2  # Maximum SC iterations per particle per step
    verbosity: int = 0

    def __post_init__(self):
        """Normalize the mode name and reject configurations that cannot run."""
        mode = canonicalize_self_consistency_mode(self.convergence_mode)
        if mode not in SUPPORTED_CONVERGENCE_MODES:
            raise ValueError(
                f"Unknown self-consistency convergence_mode {self.convergence_mode!r}; "
                f"expected one of {', '.join(SUPPORTED_CONVERGENCE_MODES)} "
                "(historical aliases: mass_shell_only, full_iteration). The former "
                "'dual_independent' mode no longer exists."
            )
        object.__setattr__(self, "convergence_mode", mode)
        if self.enabled and int(self.max_iterations) < MIN_ENABLED_MAX_ITERATIONS:
            raise ValueError(
                "self-consistency max_iterations must be at least "
                f"{MIN_ENABLED_MAX_ITERATIONS} when self-consistency is enabled "
                f"(got {self.max_iterations}). The mass-shell convergence check "
                "starts on the second iteration, so a single pass always ends in "
                "SelfConsistencyNonConvergenceError; the former single-pass plus "
                "safety-net behaviour was removed in June 2026. Use 2 (the "
                "default) or disable self-consistency."
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
            max_iterations=2,
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
    pseudo_grid_space_charge_source_trajectory: Optional[Any] = None,
    pseudo_grid_space_charge_source_soa: Optional[Any] = None,
    pseudo_grid_space_charge_source_radii_mm: Optional[Any] = None,
    macroparticle_smearing: Optional[Any] = None,
    beamline_geometry: Optional[Any] = None,
    magnetic_dipole: Optional[Any] = None,
    exact_source_history: Optional[Any] = None,
    exact_dipole_source_collection: Optional[Any] = None,
    exact_source_spin_interpolation_model: str = "centered_c1",
    exact_same_bunch_ranges: tuple[slice, ...] = (),
    _sampled_inclusion_enabled: bool = True,
    _lab_time_target_ns: float | None = None,
    _instantaneous_source_endpoint: Optional[Any] = None,
    _pseudo_grid_potential_context: Optional[Any] = None,
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
    exact_source_history : Optional[Any]
        Optional immutable source-history view used only by exact retarded field
        providers. Legacy chronology and startup gating continue to use
        ``trajectory_ext`` and ``traj_ext_soa``.
    exact_dipole_source_collection : Optional[Any]
        Optional independent causal-$C^5$ history for the dipole Maxwell
        source. The ordinary charge provider still receives
        ``exact_source_history``.
    exact_source_spin_interpolation_model : str
        Spin interpolation contract for the exact source-history view. Trial
        overlays use ``"causal_frozen_c1"`` so appending a trial endpoint cannot
        revise already accepted source segments.

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
        **(
            {"_lab_time_target_ns": _lab_time_target_ns}
            if _lab_time_target_ns is not None
            else {}
        ),
        **(
            {"_instantaneous_source_endpoint": _instantaneous_source_endpoint}
            if _instantaneous_source_endpoint is not None
            else {}
        ),
        **({"space_charge": space_charge} if space_charge is not None else {}),
        **(
            {"_sampled_inclusion_enabled": False}
            if not _sampled_inclusion_enabled
            and ("_sampled_inclusion_enabled" in _sig_params or _accepts_var_kwargs)
            else {}
        ),
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
            {
                "pseudo_grid_space_charge_source_trajectory": (
                    pseudo_grid_space_charge_source_trajectory
                )
            }
            if pseudo_grid_space_charge_source_trajectory is not None
            and (
                "pseudo_grid_space_charge_source_trajectory" in _sig_params
                or _accepts_var_kwargs
            )
            else {}
        ),
        **(
            {
                "pseudo_grid_space_charge_source_soa": (
                    pseudo_grid_space_charge_source_soa
                )
            }
            if pseudo_grid_space_charge_source_soa is not None
            and (
                "pseudo_grid_space_charge_source_soa" in _sig_params
                or _accepts_var_kwargs
            )
            else {}
        ),
        **(
            {
                "pseudo_grid_space_charge_source_radii_mm": (
                    pseudo_grid_space_charge_source_radii_mm
                )
            }
            if pseudo_grid_space_charge_source_radii_mm is not None
            and (
                "pseudo_grid_space_charge_source_radii_mm" in _sig_params
                or _accepts_var_kwargs
            )
            else {}
        ),
        **(
            {"_pseudo_grid_potential_context": _pseudo_grid_potential_context}
            if _pseudo_grid_potential_context is not None
            and ("_pseudo_grid_potential_context" in _sig_params or _accepts_var_kwargs)
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
        **(
            {"macroparticle_smearing": macroparticle_smearing}
            if macroparticle_smearing is not None
            and ("macroparticle_smearing" in _sig_params or _accepts_var_kwargs)
            else {}
        ),
        **(
            {"beamline_geometry": beamline_geometry}
            if beamline_geometry is not None
            and ("beamline_geometry" in _sig_params or _accepts_var_kwargs)
            else {}
        ),
        **(
            {"magnetic_dipole": magnetic_dipole}
            if magnetic_dipole is not None
            and ("magnetic_dipole" in _sig_params or _accepts_var_kwargs)
            else {}
        ),
        **(
            {"exact_same_bunch_ranges": exact_same_bunch_ranges}
            if exact_same_bunch_ranges
            and ("exact_same_bunch_ranges" in _sig_params or _accepts_var_kwargs)
            else {}
        ),
        **(
            {"exact_source_history": exact_source_history}
            if exact_source_history is not None
            and ("exact_source_history" in _sig_params or _accepts_var_kwargs)
            else {}
        ),
        **(
            {"exact_dipole_source_collection": exact_dipole_source_collection}
            if exact_dipole_source_collection is not None
            and ("exact_dipole_source_collection" in _sig_params or _accepts_var_kwargs)
            else {}
        ),
        **(
            {
                "exact_source_spin_interpolation_model": (
                    exact_source_spin_interpolation_model
                )
            }
            if "exact_source_spin_interpolation_model" in _sig_params
            or _accepts_var_kwargs
            else {}
        ),
    )

    return result


def self_consistency_runtime_warnings(
    config: Optional[SelfConsistencyConfig],
    *,
    radiation_reaction_mode: Optional[str],
    exact_path: bool,
    chrono_mode: Optional[ChronoMatchingMode] = None,
) -> list[tuple[type[Warning], str]]:
    """Return warnings for options that have no effect in the selected run.

    Warnings only: none of these combinations is rejected.
    """

    messages: list[tuple[type[Warning], str]] = []
    if exact_path and chrono_mode is ChronoMatchingMode.AVERAGED:
        messages.append(
            (
                SelfConsistencyOptionWarning,
                "chrono_mode AVERAGED has no effect on the exact RFS/dipole path "
                "(INERTIAL_PREHISTORY); the exact provider solves the light cone "
                "directly.",
            )
        )
    if config is None:
        return messages

    method = config.gamma_reconciliation_method
    if method is not GammaReconciliationMethod.DISABLED:
        inert_reason = None
        if not config.enabled:
            inert_reason = "self-consistency is disabled"
        elif exact_path:
            inert_reason = (
                "the exact RFS/dipole path derives beta from on-shell momentum"
            )
        elif str(radiation_reaction_mode or "").lower() in {
            "medina_lad",
            "medina_lad_validity",
        }:
            inert_reason = (
                "radiation_reaction_mode='medina_lad' derives beta from on-shell "
                "momentum, so gamma_velocity equals gamma_energy"
            )
        if inert_reason is not None:
            messages.append(
                (
                    SelfConsistencyOptionWarning,
                    f"gamma_reconciliation_method={method.name} has no effect: "
                    f"{inert_reason}. Gamma reconciliation is a diagnostic-only "
                    "option; use DISABLED.",
                )
            )

    chrono_flags = [
        name
        for name in (
            "chrono_interpolate",
            "chrono_high_precision",
            "chrono_adaptive_tolerance",
        )
        if bool(getattr(config, name, False))
    ]
    if exact_path and chrono_flags:
        messages.append(
            (
                SelfConsistencyOptionWarning,
                "No effect on the exact RFS/dipole path (INERTIAL_PREHISTORY), "
                f"where sampled chrono matching is bypassed: {', '.join(chrono_flags)}.",
            )
        )
    elif not config.chrono_interpolate:
        orphan = [
            name
            for name in ("chrono_high_precision", "chrono_adaptive_tolerance")
            if bool(getattr(config, name, False))
        ]
        if orphan:
            messages.append(
                (
                    SelfConsistencyOptionWarning,
                    "No effect without chrono_interpolate=True: "
                    f"{', '.join(orphan)}.",
                )
            )
    if config.chrono_high_precision:
        messages.append(
            (
                ChronoHighPrecisionDeprecationWarning,
                "chrono_high_precision is deprecated and has no effect in the "
                "maintained (structure-of-arrays) runner; results equal "
                "chrono_high_precision=False.",
            )
        )
    return messages


def emit_self_consistency_runtime_warnings(
    config: Optional[SelfConsistencyConfig],
    *,
    radiation_reaction_mode: Optional[str],
    exact_path: bool,
    chrono_mode: Optional[ChronoMatchingMode] = None,
    logger: Optional[Callable[[str], Any]] = None,
) -> list[str]:
    """Emit :func:`self_consistency_runtime_warnings` once each at run start."""

    emitted: list[str] = []
    for category, message in self_consistency_runtime_warnings(
        config,
        radiation_reaction_mode=radiation_reaction_mode,
        exact_path=exact_path,
        chrono_mode=chrono_mode,
    ):
        warnings.warn(message, category, stacklevel=3)
        if logger is not None:
            logger(f"[WARNING] {message}")
        emitted.append(message)
    return emitted


__all__ = [
    "ChronoHighPrecisionDeprecationWarning",
    "SelfConsistencyOptionWarning",
    "emit_self_consistency_runtime_warnings",
    "self_consistency_runtime_warnings",
    "SelfConsistencyConfig",
    "canonicalize_self_consistency_mode",
    "self_consistent_step",
]
