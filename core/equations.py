"""Retarded equations of motion for the Liénard–Wiechert solver.

The implementation intentionally mirrors the behaviour of the validated legacy
code so that historical regression data remains applicable.  The heavy lifting
is performed inside :func:`retarded_equations_of_motion`, which calculates the
covariant updates for momentum, position, and acceleration for each particle.

Physical Foundation
-------------------

The integrator evolves particles in coordinate time with step h = Δt, updating
conjugate momentum from retarded electromagnetic forces, then deriving positions
and velocities.

Conjugate vs. Kinetic Momentum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The conjugate (canonical) momentum includes electromagnetic potentials::

    P^μ = γ·m·V^μ + (e/c)·A^μ

For spatial components: P_i = γ·m·v_i + (e/c)·A_i

The kinetic (mechanical) momentum is::

    P_kinetic = P - (e/c)·A = γ·m·v

Position Updates
~~~~~~~~~~~~~~~~

Spatial positions are updated in coordinate time using the kinetic momentum::

    Δx = v·h = (P_kinetic / (γ·m))·h

The 1/γ factor is **essential**: it ensures that velocity v = P_kinetic/(γ·m)
remains subluminal even as momentum grows with γ.

Velocity Calculation
~~~~~~~~~~~~~~~~~~~~

Velocity (beta) is computed from the coordinate-time displacement::

    β = v/c = Δx/(c·h)

Note: This does **not** include a γ factor in the denominator. The time dilation
is already accounted for in the position update formula.

Self-Consistency Iterations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ultra-relativistic particles (γ ≫ 1), forces depend strongly on γ through
the retarded field geometry (k-factor, field Lorentz contraction). The integrator
resolves the circular dependency γ → forces → P → γ through iterations:

1. Use γ_n-1 to compute retarded forces
2. Update conjugate momentum P_n from those forces
3. Update positions using the **same** γ_n-1: Δx = (h/(γ_n-1·m))·P_kinetic
4. Compute velocity: β = Δx/(c·h)
5. Derive two independent γ estimates:

   - From energy: γ_E = (Pt - e·Φ)/(mc)
   - From velocity: γ_V = 1/√(1-β²)

6. Check convergence: |γ_E - γ_V|/γ_E < ε (typically ε = 10⁻⁶)

If not converged, the next iteration uses γ_n = γ_E and repeats. Using a
**consistent** γ throughout each iteration for both forces and positions ensures
the velocity extracted from Δx corresponds physically to the computed momentum.

See :class:`core.self_consistency.SelfConsistencyConfig` for configuration.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .constants import C_MMNS
from .distances import (
    ChronoMatchResult,
    chrono_match_indices,
    compute_instantaneous_distance,
    compute_retarded_distance,
)
from .self_consistency import SelfConsistencyConfig
from .types import (
    ChronoMatchingMode,
    GammaReconciliationMethod,
    ParticleState,
    SimulationType,
    StartupMode,
    Trajectory,
)
from .vectorized_interactions import (
    compute_vectorized_contributions,
    gather_external_samples,
)


class GammaBlowupError(Exception):
    """Exception raised when gamma exceeds threshold during integration.

    This exception signals to the integration runner that the timestep should
    be reduced and the step retried. This is raised for ALL gamma blowups,
    including extreme values (> 1e20, NaN, or Inf). The integration runner
    will attempt timestep reduction, and only after exhausting retry attempts
    will the particle be marked as dead.

    Attributes
    ----------
    step_idx : int
        Integration step number where the blowup occurred.
    particle_idx : int
        Index of the particle that experienced the blowup.
    gamma_value : float
        The gamma value that triggered the blowup.
    iteration : int
        Self-consistency iteration number where the blowup was detected.
    is_hard_blowup : bool
        True if this was a hard blowup (NaN/Inf or > 1e20), used for logging.
    """

    def __init__(
        self,
        step_idx: int,
        particle_idx: int,
        gamma_value: float,
        iteration: int,
        is_hard_blowup: bool = False,
    ):
        self.step_idx = step_idx
        self.particle_idx = particle_idx
        self.gamma_value = gamma_value
        self.iteration = iteration
        self.is_hard_blowup = is_hard_blowup
        severity = "Hard" if is_hard_blowup else "Soft"
        super().__init__(
            f"{severity} gamma blowup at step {step_idx}, particle {particle_idx}: "
            f"γ={gamma_value:.2e} (iteration {iteration})"
        )


def _ensure_startup_metadata(state: ParticleState) -> None:
    """Initialize origin positions and beta averaging metadata if not present."""
    if "origin_x" not in state:
        state["origin_x"] = np.copy(state.get("x", np.array([])))
    if "origin_y" not in state:
        state["origin_y"] = np.copy(state.get("y", np.array([])))
    if "origin_z" not in state:
        state["origin_z"] = np.copy(state.get("z", np.array([])))
    if "beta_avg_x" not in state:
        state["beta_avg_x"] = np.copy(state.get("bx", np.array([])))
    if "beta_avg_y" not in state:
        state["beta_avg_y"] = np.copy(state.get("by", np.array([])))
    if "beta_avg_z" not in state:
        state["beta_avg_z"] = np.copy(state.get("bz", np.array([])))
    if "beta_samples" not in state:
        state["beta_samples"] = np.ones_like(state.get("x", np.array([])), dtype=float)


def _extract_self_consistency_params(
    self_consistency: Optional[SelfConsistencyConfig],
) -> tuple[bool, str, float, float, float, int, int]:
    """Extract self-consistency configuration parameters.

    Returns
    -------
    tuple[bool, str, float, float, float, int, int]
        A tuple containing (enabled, convergence_mode, target_ms_tolerance,
        mass_shell_tolerance, mass_shell_relaxation, max_iterations, verbosity).
    """
    is_enabled = self_consistency is not None and self_consistency.enabled
    convergence_mode = (
        self_consistency.convergence_mode
        if self_consistency is not None
        else "fixed_geometry"
    )
    target_ms_tolerance = (
        self_consistency.target_ms_tolerance if self_consistency is not None else 1e-6
    )
    mass_shell_tolerance = (
        self_consistency.mass_shell_tolerance if self_consistency is not None else 1e-2
    )
    mass_shell_relaxation = (
        self_consistency.mass_shell_relaxation if self_consistency is not None else 0.7
    )
    max_iterations = (
        self_consistency.max_iterations if self_consistency is not None else 10
    )
    verbosity = self_consistency.verbosity if self_consistency is not None else 0

    return (
        is_enabled,
        convergence_mode,
        target_ms_tolerance,
        mass_shell_tolerance,
        mass_shell_relaxation,
        max_iterations,
        verbosity,
    )


def _initialize_result_state(current_state: ParticleState) -> ParticleState:
    """Create a copy of the current particle state for the next time step.

    Parameters
    ----------
    current_state : ParticleState
        The current state at this time step.

    Returns
    -------
    ParticleState
        A deep copy with all arrays duplicated, including dead particle metadata.
    """
    result = {
        "x": np.copy(current_state["x"]),
        "y": np.copy(current_state["y"]),
        "z": np.copy(current_state["z"]),
        "t": np.copy(current_state["t"]),
        "Px": np.copy(current_state["Px"]),
        "Py": np.copy(current_state["Py"]),
        "Pz": np.copy(current_state["Pz"]),
        "Pt": np.copy(current_state["Pt"]),
        "gamma": np.copy(current_state["gamma"]),
        "bx": np.copy(current_state["bx"]),
        "by": np.copy(current_state["by"]),
        "bz": np.copy(current_state["bz"]),
        "bdotx": np.copy(current_state["bdotx"]),
        "bdoty": np.copy(current_state["bdoty"]),
        "bdotz": np.copy(current_state["bdotz"]),
        "q": current_state["q"],
        "char_time": current_state.get("char_time", np.zeros_like(current_state["x"])),
        "m": current_state.get("m", np.ones_like(current_state["x"])),
        "dummy": np.zeros_like(current_state["bdotz"]),
        "origin_x": np.copy(current_state["origin_x"]),
        "origin_y": np.copy(current_state["origin_y"]),
        "origin_z": np.copy(current_state["origin_z"]),
        "beta_avg_x": np.copy(current_state["beta_avg_x"]),
        "beta_avg_y": np.copy(current_state["beta_avg_y"]),
        "beta_avg_z": np.copy(current_state["beta_avg_z"]),
        "beta_samples": np.copy(current_state["beta_samples"]),
    }

    # Preserve dead particle metadata to prevent redundant logging
    if "_dead_particles" in current_state:
        result["_dead_particles"] = np.copy(current_state["_dead_particles"])
    if "_particle_failure_info" in current_state:
        # Deep copy the failure info dict
        result["_particle_failure_info"] = {
            k: v.copy() if isinstance(v, dict) else v
            for k, v in current_state["_particle_failure_info"].items()
        }

    return result


def _get_particle_charge(state: ParticleState, particle_idx: int):
    """Extract charge for a single particle, handling scalar or array charge."""
    charge = state["q"]
    if hasattr(charge, "__getitem__"):
        return charge[particle_idx]
    return charge


def _get_particle_mass(state: ParticleState, particle_idx: int):
    """Extract mass for a single particle, handling scalar or array mass."""
    mass = state["m"]
    if hasattr(mass, "__getitem__"):
        return mass[particle_idx]
    return mass


def _get_particle_char_time(state: ParticleState, particle_idx: int):
    """Extract characteristic time for a single particle, handling scalar or array."""
    char_time = state["char_time"]
    if hasattr(char_time, "__getitem__"):
        return char_time[particle_idx]
    return char_time


def _compute_approximate_retarded_distance(
    current_state: ParticleState,
    external_state: ParticleState,
    particle_idx: int,
    time_step_idx: int,
) -> tuple[dict, np.ndarray]:
    """Compute approximate retarded distance using constant velocity assumption.

    This is used in APPROXIMATE_BACK_HISTORY startup mode to estimate retardation
    effects when full historical data is not yet available.

    The retarded distance accounts for source motion during light travel time using
    the Liénard-Wiechert formula: R_ret = R / (1 - β_source·n̂)

    For numerical stability at ultra-relativistic energies (γ > 10⁵), we use the
    algebraically equivalent factored form:
        R_ret = R × (1 + β·n̂) / (1 - (β·n̂)²)

    This formulation divides by a denominator that goes to zero ~2× more slowly,
    reducing catastrophic cancellation and providing better precision.

    Validated for:
        - 500 GeV electrons (γ ≈ 978,474, β ≈ 0.9999999999995)
        - 20 TeV protons (γ ≈ 21,321, β ≈ 0.999999999)

    Parameters
    ----------
    current_state : ParticleState
        Observer particle state (the bunch being updated).
    external_state : ParticleState
        Source particle state (the external bunch).
    particle_idx : int
        Index of the observer particle within current_state.
    time_step_idx : int
        Current trajectory index.

    Returns
    -------
    tuple[dict, np.ndarray]
        A tuple of (nhat dictionary with corrected R, bounded_indices array).
    """
    sample_count = len(external_state["x"])
    indices_bounded = np.full(sample_count, time_step_idx, dtype=int)

    # Compute instantaneous distance and direction vector n̂
    nhat = compute_instantaneous_distance(current_state, external_state, particle_idx)

    # Compute β_source · n̂ where n̂ points FROM source TO observer
    beta_ext_dot_nhat = (
        external_state["bx"] * nhat["nx"]
        + external_state["by"] * nhat["ny"]
        + external_state["bz"] * nhat["nz"]
    )

    # Use factored form for numerical stability: R_ret = R × (1+β·n̂) / (1-(β·n̂)²)
    # This is algebraically equivalent to R / (1 - β·n̂) but more stable because:
    #   (1 - (β·n̂)²) = (1 - β·n̂)(1 + β·n̂)
    # The factored denominator goes to zero ~2× more slowly as β·n̂ → 1
    numerator = 1.0 + beta_ext_dot_nhat
    denominator = 1.0 - beta_ext_dot_nhat**2

    # Clamp denominator to prevent division by zero
    # Physical interpretation: β·n̂ → 1 means source moving at light speed away
    # from observer. Light never catches up, so R_ret → ∞ (correct physics).
    # Clamping gives huge but finite R_ret; forces become negligible (as expected).
    #
    # k_threshold = 1e-12 supports particles up to γ ≈ 7×10⁵:
    #   - Covers 500 GeV electrons (γ ≈ 978,474)
    #   - Covers 20 TeV protons (γ ≈ 21,321)
    #   - R_ret saturates at ~10¹² mm (1000 km) in extreme cases
    #   - Such large R leads to negligible forces (correct behavior)
    #
    # Force calculation has additional safety: K_CUTOFF_HARD = 1e-20 filters
    # interactions where k = (1 - β·n̂) is extremely small.
    k_threshold = 1e-12
    denominator = np.where(
        np.abs(denominator) < k_threshold,
        np.copysign(k_threshold, denominator),  # Preserve sign for negative β·n̂
        denominator,
    )

    # Apply correction to retarded distance
    nhat["R"] = nhat["R"] * numerator / denominator

    return nhat, indices_bounded


def _compute_full_retarded_distance(
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    time_step_idx: int,
    particle_idx: int,
    chrono_mode: ChronoMatchingMode,
    self_consistency: Optional[SelfConsistencyConfig] = None,
    timestep_h: float = 1e-3,
) -> tuple[dict, np.ndarray, Optional[ChronoMatchResult]]:
    """Compute retarded distance using full chronological matching.

    This uses the complete trajectory history to find the proper retarded time
    for each external source particle.

    Returns
    -------
    tuple[dict, np.ndarray, Optional[ChronoMatchResult]]
        A tuple of (nhat dictionary, bounded_indices array, chrono_match_result).
        chrono_match_result is None if interpolation is disabled.
    """
    # Check if chrono-match interpolation is enabled
    chrono_interpolate = False
    chrono_tolerance = 1e-3
    chrono_high_precision = False
    chrono_adaptive_tolerance = False
    verbosity = 0

    if self_consistency is not None:
        chrono_interpolate = self_consistency.chrono_interpolate
        chrono_tolerance = self_consistency.chrono_tolerance
        chrono_high_precision = getattr(
            self_consistency, "chrono_high_precision", False
        )
        chrono_adaptive_tolerance = getattr(
            self_consistency, "chrono_adaptive_tolerance", False
        )
        verbosity = self_consistency.verbosity

    retarded_result = chrono_match_indices(
        trajectory,
        trajectory_ext,
        time_step_idx,
        particle_idx,
        mode=chrono_mode,
        interpolate=chrono_interpolate,
        tolerance=chrono_tolerance,
        verbosity=verbosity,
        high_precision=chrono_high_precision,
        adaptive_tolerance=chrono_adaptive_tolerance,
        timestep_h=timestep_h,
    )

    # Handle both legacy (array) and new (ChronoMatchResult) returns
    if isinstance(retarded_result, ChronoMatchResult):
        retarded_indices = retarded_result.indices
        chrono_match_result = retarded_result
    else:
        retarded_indices = retarded_result
        chrono_match_result = None

    max_external_idx = len(trajectory_ext) - 1
    indices_bounded = np.minimum(np.maximum(retarded_indices, 0), max_external_idx)

    nhat = compute_retarded_distance(
        trajectory,
        trajectory_ext,
        time_step_idx,
        particle_idx,
        indices_bounded,
    )

    return nhat, indices_bounded, chrono_match_result


def _calculate_travel_distance(
    origin_position: tuple[float, float, float],
    current_position: tuple[float, float, float],
) -> float:
    """Calculate Euclidean distance between origin and current position."""
    dx = current_position[0] - origin_position[0]
    dy = current_position[1] - origin_position[1]
    dz = current_position[2] - origin_position[2]
    return float(np.sqrt(dx**2 + dy**2 + dz**2))


def _compute_gating_threshold(
    nhat: dict,
    beta_avg_x: float,
    beta_avg_y: float,
    beta_avg_z: float,
) -> float:
    """Compute the minimum travel distance before external forces are applied.

    The threshold ensures the observer particle has traveled far enough that
    light from the external source's initial position could have reached it.

    Light propagates at speed c. For a particle moving at velocity β relative
    to the source, the relative closing speed is c(1 - β·n̂), where n̂ points
    from source to observer.

    Physical examples (for approaching particles, β·n̂ < 0):
    - Stationary (β=0): threshold = 0 (forces apply immediately)
    - Low velocity (β=0.1): threshold ≈ 0.091·R
    - Moderate (β=0.5): threshold ≈ 0.33·R
    - Relativistic (β=0.9): threshold ≈ 0.47·R
    - Ultra-relativistic (β→1): threshold → R/2 (approaches limit, never exceeds)

    Formula: threshold = β·R / (1 - β·n̂)

    Special handling for β·n̂ ≥ 1 (receding at or above light speed):
    Return very large threshold (effectively infinite) to suppress forces.
    """
    beta_avg_dot_nhat = (
        beta_avg_x * nhat["nx"] + beta_avg_y * nhat["ny"] + beta_avg_z * nhat["nz"]
    )

    # Compute denominator: (1 - β·n̂)
    # β·n̂ < 0: approaching → denominator > 1 → small threshold (meet quickly)
    # β·n̂ > 0: receding → denominator < 1 → large threshold (takes longer)
    # β·n̂ → 1: receding at c → denominator → 0 → threshold → ∞ (never meet)
    denominators = 1.0 - beta_avg_dot_nhat

    # Handle edge case: particles receding at or faster than light speed
    # For β·n̂ ≥ 1, light never catches the observer, so threshold = ∞
    # Use a very large but finite value to avoid numerical issues
    LARGE_THRESHOLD = 1e12  # effectively infinite for simulation purposes

    # For denominators ≤ 0 or very small, use large threshold
    # Also handle case where β·n̂ is very close to 1 (denominator near 0)
    MIN_DENOMINATOR = 1e-6  # corresponds to β·n̂ = 0.999999

    # Calculate particle speed magnitude
    beta_magnitude = np.sqrt(beta_avg_x**2 + beta_avg_y**2 + beta_avg_z**2)

    thresholds = np.where(
        denominators > MIN_DENOMINATOR,
        beta_magnitude * nhat["R"] / denominators,
        LARGE_THRESHOLD,
    )

    if thresholds.size > 0:
        return float(np.max(np.maximum(thresholds, 0.0)))
    return 0.0


def _should_apply_external_forces(
    startup_mode: StartupMode,
    nhat: dict,
    current_state: ParticleState,
    particle_idx: int,
) -> bool:
    """Determine whether external forces should be applied to this particle.

    In COLD_START mode, forces are suppressed until the particle has traveled
    far enough from its origin for retardation effects to be meaningful.
    """
    if startup_mode is not StartupMode.COLD_START or nhat["R"].size == 0:
        return True

    origin_position = (
        current_state["origin_x"][particle_idx],
        current_state["origin_y"][particle_idx],
        current_state["origin_z"][particle_idx],
    )

    current_position = (
        current_state["x"][particle_idx],
        current_state["y"][particle_idx],
        current_state["z"][particle_idx],
    )

    travel_distance = _calculate_travel_distance(origin_position, current_position)

    beta_avg_x = current_state["beta_avg_x"][particle_idx]
    beta_avg_y = current_state["beta_avg_y"][particle_idx]
    beta_avg_z = current_state["beta_avg_z"][particle_idx]

    threshold = _compute_gating_threshold(nhat, beta_avg_x, beta_avg_y, beta_avg_z)

    return travel_distance >= threshold


def _get_current_particle_gamma_and_beta(
    current_state: ParticleState,
    result_state: ParticleState,
    particle_idx: int,
    sc_iteration: int,
    sc_enabled: bool,
) -> tuple[float, tuple[float, float, float]]:
    """Get gamma and beta values for the current self-consistency iteration.

    On the first iteration, use values from the input state.
    On subsequent iterations, use the updated values from the result state.
    """
    if sc_enabled and sc_iteration > 0:
        gamma = result_state["gamma"][particle_idx]
        beta_vector = (
            result_state["bx"][particle_idx],
            result_state["by"][particle_idx],
            result_state["bz"][particle_idx],
        )
    else:
        gamma = current_state["gamma"][particle_idx]
        beta_vector = (
            current_state["bx"][particle_idx],
            current_state["by"][particle_idx],
            current_state["bz"][particle_idx],
        )

    return gamma, beta_vector


def _limit_beta_magnitude(
    beta_x: float, beta_y: float, beta_z: float
) -> tuple[float, float, float]:
    """Ensure beta magnitude stays below the speed of light.

    Uses float64 precision to allow beta extremely close to c (1 - 1e-16).
    This allows gamma up to ~1e8 while maintaining numerical stability.

    Returns
    -------
    tuple[float, float, float]
        The (possibly scaled) beta components (βx, βy, βz).
    """
    # Use float64 for high precision in beta calculations
    bx64 = np.float64(beta_x)
    by64 = np.float64(beta_y)
    bz64 = np.float64(beta_z)

    beta_magnitude = np.sqrt(bx64**2 + by64**2 + bz64**2)

    # Allow beta extremely close to 1.0, limited only by float64 precision
    # 1 - 1e-16 corresponds to gamma ~ 1e8
    beta_max_allowed = np.float64(1.0) - np.float64(1e-16)

    if beta_magnitude >= beta_max_allowed:
        scale_factor = beta_max_allowed / beta_magnitude
        return (
            float(bx64 * scale_factor),
            float(by64 * scale_factor),
            float(bz64 * scale_factor),
        )

    return beta_x, beta_y, beta_z


def _calculate_one_minus_beta_squared(
    beta_x: float, beta_y: float, beta_z: float
) -> float:
    """Calculate 1 - β² using Kahan compensated summation for numerical stability.

    At ultra-relativistic speeds (β → 1), direct calculation of 1 - β² suffers from
    catastrophic cancellation. This function uses Kahan summation to accurately
    compute β² first, then returns 1 - β².

    Parameters
    ----------
    beta_x, beta_y, beta_z : float
        Velocity components normalized by c.

    Returns
    -------
    float
        1 - β² with improved numerical accuracy for β ≈ 1.
    """
    # Use float64 for high precision
    bx64 = np.float64(beta_x)
    by64 = np.float64(beta_y)
    bz64 = np.float64(beta_z)

    # Kahan compensated summation for β² = βx² + βy² + βz²
    # This reduces floating-point errors when summing squares
    sum_beta_sq = np.float64(0.0)
    compensation = np.float64(0.0)

    for beta_component in [bx64, by64, bz64]:
        term = beta_component**2 - compensation
        temp_sum = sum_beta_sq + term
        compensation = (temp_sum - sum_beta_sq) - term
        sum_beta_sq = temp_sum

    beta_squared = sum_beta_sq

    # Clamp beta_squared just below 1.0 to prevent infinity
    max_beta_squared = (np.float64(1.0) - np.float64(1e-16)) ** 2
    if beta_squared >= max_beta_squared:
        beta_squared = max_beta_squared

    one_minus_beta_sq = np.float64(1.0) - beta_squared

    # Safety check: ensure non-negative result
    if one_minus_beta_sq <= np.float64(0.0):
        one_minus_beta_sq = np.float64(1.0) - max_beta_squared

    return float(one_minus_beta_sq)


def _calculate_gamma_from_beta(beta_x: float, beta_y: float, beta_z: float) -> float:
    """Calculate Lorentz factor from velocity components.

    γ = 1 / √(1 - β²)

    Uses Kahan summation and float64 precision to handle extremely relativistic
    particles accurately, avoiding catastrophic cancellation at β → 1.
    """
    one_minus_beta_sq = _calculate_one_minus_beta_squared(beta_x, beta_y, beta_z)
    return float(1.0 / np.sqrt(np.float64(one_minus_beta_sq)))


def _compute_radiation_reaction_term(
    axis: str,
    beta_component: float,
    beta_dot_component: float,
    gamma_current: float,
    gamma_previous: float,
    time_step: float,
    mass: float,
) -> tuple[float, float]:
    """Compute radiation reaction force component for a given axis.

    The radiation reaction has two terms:
    - LHS: Change in gamma times acceleration times velocity
    - RHS: Cubic gamma times acceleration squared times velocity

    Parameters
    ----------
    axis : str
        Axis name ('x', 'y', or 'z') for debug purposes.
    beta_component : float
        Velocity component (βx, βy, or βz).
    beta_dot_component : float
        Acceleration component (β̇x, β̇y, or β̇z).
    gamma_current : float
        Current Lorentz factor.
    gamma_previous : float
        Lorentz factor from previous time step.
    time_step : float
        Time step size.
    mass : float
        Particle rest mass.

    Returns
    -------
    tuple[float, float]
        The (lhs_term, rhs_term) of the radiation reaction force.
    """
    # Left-hand side: change in gamma contribution
    lhs_term = (
        (gamma_current - gamma_previous)
        / (time_step * gamma_current)
        * mass
        * beta_dot_component
        * beta_component
        * C_MMNS**2
    )

    # Right-hand side: cubic gamma contribution
    rhs_term = (
        -(gamma_current**3)
        * (mass * beta_dot_component**2 * C_MMNS**2)
        * beta_component
        * C_MMNS
    )

    return lhs_term, rhs_term


def _update_beta_running_average(
    previous_avg: tuple[float, float, float],
    previous_sample_count: float,
    new_beta: tuple[float, float, float],
) -> tuple[tuple[float, float, float], float]:
    """Update running average of beta components with a new sample.

    Returns
    -------
    tuple[tuple[float, float, float], float]
        Updated (beta_avg_x, beta_avg_y, beta_avg_z) and new sample count.
    """
    new_sample_count = previous_sample_count + 1.0

    avg_x = (previous_avg[0] * previous_sample_count + new_beta[0]) / new_sample_count
    avg_y = (previous_avg[1] * previous_sample_count + new_beta[1]) / new_sample_count
    avg_z = (previous_avg[2] * previous_sample_count + new_beta[2]) / new_sample_count

    return (avg_x, avg_y, avg_z), new_sample_count


def _check_mass_shell_convergence(
    Pt: float,
    Px: float,
    Py: float,
    Pz: float,
    particle_mass: float,
    C_MMNS: float,
    tolerance: float,
) -> tuple[bool, float]:
    """Check if mass-shell constraint is satisfied (PRIMARY convergence criterion).

    Mass-shell constraint: Pt² - P² = (mc)²

    Returns
    -------
    tuple[bool, float]
        (has_converged, relative_mass_shell_error)
    """
    P_spatial_sq = Px**2 + Py**2 + Pz**2
    mass_shell_rhs = (particle_mass * C_MMNS) ** 2
    mass_shell_lhs = Pt**2 - P_spatial_sq

    mass_shell_error_abs = abs(mass_shell_lhs - mass_shell_rhs)
    mass_shell_error_rel = mass_shell_error_abs / max(mass_shell_rhs, 1e-40)

    has_converged = mass_shell_error_rel < tolerance

    return has_converged, mass_shell_error_rel


def _check_gamma_consistency(
    gamma_velocity: float,
    gamma_energy: float,
    tolerance: float,
) -> tuple[bool, float]:
    """Check gamma consistency (DIAGNOSTIC check after convergence).

    This verifies that gamma from velocity matches gamma from energy.
    If mass-shell is satisfied, these should match to machine precision.

    Returns
    -------
    tuple[bool, float]
        (is_consistent, relative_gamma_error)
    """
    gamma_abs_change = abs(gamma_velocity - gamma_energy)
    gamma_rel_change = gamma_abs_change / max(abs(gamma_velocity), 1e-12)
    is_consistent = gamma_rel_change < tolerance

    return is_consistent, gamma_rel_change


def _print_convergence_info(
    particle_idx: int,
    iteration: int,
    gamma_from_velocity: float,
    gamma_from_energy: float,
    gamma_mass_shell: float,
    mass_shell_error: float,
    gamma_consistency_error: float,
    converged: bool,
    max_iterations: int,
    verbosity: int = 1,
    step_idx: Optional[int] = None,
    convergence_mode: str = "fixed_geometry",
    particle_position: Optional[tuple[float, float, float]] = None,
    particle_time: Optional[float] = None,
) -> None:
    """Print debug information about self-consistency convergence.

    Shows mass-shell convergence status.

    Parameters
    ----------
    gamma_from_velocity : float
        Gamma computed from velocity: γ = 1/√(1-β²)
    gamma_from_energy : float
        Gamma computed from kinetic energy: γ = (Pt - q·Φ)/(mc)
    gamma_mass_shell : float
        Gamma computed from mass-shell constraint: γ = √(P²+(mc)²)/(mc)
    mass_shell_error : float
        Relative mass-shell error: |Pt² - P² - (mc)²|/(mc)²
    gamma_consistency_error : float
        Relative gamma consistency error: |γ_velocity - γ_energy| / γ
    verbosity : int
        0 = silent (no output)
        1 = summary (one line per step)
        2 = failures only (detailed only for non-converged)
        3 = full detail (all iterations)
    step_idx : Optional[int]
        Integration step number for context in error messages
    convergence_mode : str
        Convergence mode: "fixed_geometry" or "variable_geometry"
    """
    if verbosity == 0:
        return

    # Basic output (verbosity >= 1)
    if converged:
        status = f"converged in {iteration + 1} iter"
    else:
        status = f"max iter ({max_iterations}) reached"

    # Prepare step prefix if step_idx is provided
    step_prefix = f"Step {step_idx}, " if step_idx is not None else ""

    # Adjust output based on convergence mode
    if verbosity == 1:
        # Summary: one line per particle
        if convergence_mode == "fixed_geometry":
            print(
                f"    {step_prefix}P{particle_idx}: {status}, E_ms={mass_shell_error:.3e}"
            )
        else:
            print(
                f"    {step_prefix}P{particle_idx}: {status}, E_ms={mass_shell_error:.3e}, "
                f"E_gamma={gamma_consistency_error:.3e}"
            )
    elif verbosity == 2:
        # Failures only: detailed output only for non-converged steps
        if not converged:
            print(f"    {step_prefix}Particle {particle_idx}: {status}")
            print(f"      Mass-shell error = {mass_shell_error:.15e}")
            # Print position and time for failures
            if particle_position is not None and particle_time is not None:
                x, y, z = particle_position
                print(f"      Position: x={x:.6e} mm, y={y:.6e} mm, z={z:.6e} mm")
                print(f"      Time: t={particle_time:.6e} ns")
            # Only print gamma values, no "dual convergence" or gamma consistency messages
            print(f"      γ_velocity (from β)        = {gamma_from_velocity:.15e}")
            print(f"      γ_energy   (from Pt - q·Φ) = {gamma_from_energy:.15e}")
            print(f"      γ_mass_shell (√(P²+(mc)²)/(mc)) = {gamma_mass_shell:.15e}")
        else:
            # For converged steps at verbosity 2, just show summary
            if convergence_mode == "fixed_geometry":
                print(
                    f"    {step_prefix}P{particle_idx}: {status}, E_ms={mass_shell_error:.3e}"
                )
            else:
                print(
                    f"    {step_prefix}P{particle_idx}: {status}, E_ms={mass_shell_error:.3e}, "
                    f"E_gamma={gamma_consistency_error:.3e}"
                )
    else:  # verbosity >= 3
        # Full detail: multi-line output with full precision for all steps
        print(f"    {step_prefix}Particle {particle_idx}: {status}")
        print(f"      Mass-shell error = {mass_shell_error:.15e}")
        # Print position and time for verbosity 3 when showing failures
        if (
            not converged
            and particle_position is not None
            and particle_time is not None
        ):
            x, y, z = particle_position
            print(f"      Position: x={x:.6e} mm, y={y:.6e} mm, z={z:.6e} mm")
            print(f"      Time: t={particle_time:.6e} ns")
        # Only print gamma values, no "dual convergence" or gamma consistency messages
        print(f"      γ_velocity (from β)        = {gamma_from_velocity:.15e}")
        print(f"      γ_energy   (from Pt - q·Φ) = {gamma_from_energy:.15e}")
        print(f"      γ_mass_shell (√(P²+(mc)²)/(mc)) = {gamma_mass_shell:.15e}")


def retarded_equations_of_motion(
    h: float,
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    index_traj: int,
    aperture_radius: float,
    sim_type: SimulationType,
    chrono_mode: ChronoMatchingMode = ChronoMatchingMode.AVERAGED,
    startup_mode: StartupMode = StartupMode.COLD_START,
    self_consistency: Optional[SelfConsistencyConfig] = None,
    step_idx: Optional[int] = None,
    cancel_callback: Optional[Any] = None,
) -> ParticleState:
    """Core equations of motion mirroring the validated legacy implementation.

    Parameters
    ----------
    h:
        Time step between trajectory samples.
    trajectory:
        Mutable view over the rider bunch history.
    trajectory_ext:
        History of the external bunch (driver, image or opposing bunch).
    index_traj:
        Index of the current time step within ``trajectory``.
    aperture_radius:
        Aperture radius supplied to the image generators.
    sim_type:
        Simulation boundary type encoded as :class:`SimulationType`.
    chrono_mode:
        Retardation sampling strategy; ``FAST`` retains the legacy single
        sample, whereas ``AVERAGED`` blends ``R / c`` and ``2R / c`` emission
        times for the external bunch.
    startup_mode:
        Early-step handling strategy; ``COLD_START`` suppresses external forces
        until sufficient observer travel has occurred, while
        ``APPROXIMATE_BACK_HISTORY`` assumes constant source velocity to
        reconstruct an analytic history.
    self_consistency:
        Optional configuration for self-consistency iterations. If provided and
        enabled, each particle's update will iterate until gamma converges,
        solving the circular dependency between forces and gamma.
    step_idx:
        Optional integration step number for context in error messages.
    cancel_callback:
        Optional predicate to check for cancellation. If provided and returns True,
        raises IntegrationCancelled to abort the integration.

    Returns
    -------
    ParticleState
        Updated particle state for the next time step.
    """
    # Ensure metadata for startup mode is initialized
    _ensure_startup_metadata(trajectory[index_traj])

    # Initialize result state as a copy of current state
    current_state = trajectory[index_traj]
    result = _initialize_result_state(current_state)

    num_particles = len(current_state["x"])

    # Track particles marked dead in this step
    particles_marked_dead_this_step = 0

    # Extract self-consistency configuration
    (
        sc_enabled,
        sc_convergence_mode,
        sc_target_ms_tolerance,
        sc_mass_shell_tolerance,
        sc_mass_shell_relaxation,
        sc_max_iterations,
        sc_verbosity,
    ) = _extract_self_consistency_params(self_consistency)

    # Extract chrono-match parameters (needed for interpolation later)
    chrono_high_precision = False
    if self_consistency is not None:
        chrono_high_precision = getattr(
            self_consistency, "chrono_high_precision", False
        )

    # Import IntegrationCancelled at top of function for use in cancel checks
    from .integration_runner import IntegrationCancelled

    # Process each particle independently
    for particle_idx in range(num_particles):
        # Check for cancellation before processing each particle
        if cancel_callback is not None and cancel_callback():
            raise IntegrationCancelled("Integration cancelled by caller.")

        # Skip particles that are already marked dead
        if "_dead_particles" in result and result["_dead_particles"][particle_idx]:
            # Copy previous state for dead particle (don't recompute)
            for key in [
                "x",
                "y",
                "z",
                "t",
                "bx",
                "by",
                "bz",
                "gamma",
                "Px",
                "Py",
                "Pz",
                "Pt",
                "bdotx",
                "bdoty",
                "bdotz",
            ]:
                if key in current_state:
                    result[key][particle_idx] = current_state[key][particle_idx]
            continue

        # Working state for SC iterations - tracks evolving state
        # On iteration 0, use current_state values
        # On iteration k > 0, use values from previous iteration
        working_beta_x = current_state["bx"][particle_idx]
        working_beta_y = current_state["by"][particle_idx]
        working_beta_z = current_state["bz"][particle_idx]
        working_gamma = current_state["gamma"][particle_idx]
        working_x = current_state["x"][particle_idx]
        working_y = current_state["y"][particle_idx]
        working_z = current_state["z"][particle_idx]

        # Self-consistency loop: iterate until gamma converges
        for sc_iteration in range(sc_max_iterations):
            # Check for cancellation during self-consistency iterations
            if cancel_callback is not None and cancel_callback():
                raise IntegrationCancelled("Integration cancelled by caller.")

            if sc_verbosity >= 3 and sc_iteration > 0:
                print(
                    f"    Particle {particle_idx} iteration {sc_iteration}: "
                    f"Starting refinement"
                )
                print(
                    f"      Working state: βx={working_beta_x:.15e}, "
                    f"βy={working_beta_y:.15e}, βz={working_beta_z:.15e}, "
                    f"γ={working_gamma:.15e}"
                )
                # Also print what result[bx/bz] contains to verify it's from previous iteration
                print(
                    f"      result[bx]={result['bx'][particle_idx]:.15e}, "
                    f"result[bz]={result['bz'][particle_idx]:.15e}"
                )

            # ================================================================
            # STEP 1: Determine observer state for retarded distance calculation
            # ================================================================
            # In variable_geometry mode, use position from previous iteration
            # In fixed_geometry mode, use initial position for all iterations
            if (
                sc_convergence_mode in ("variable_geometry", "full_iteration")
                and sc_iteration > 0
            ):
                # Create temporary state with updated position for retarded distance calc
                observer_state = {
                    "x": np.array([working_x]),
                    "y": np.array([working_y]),
                    "z": np.array([working_z]),
                    "t": np.array([current_state["t"][particle_idx]]),
                    "bx": np.array([working_beta_x]),
                    "by": np.array([working_beta_y]),
                    "bz": np.array([working_beta_z]),
                    "gamma": np.array([working_gamma]),
                    "origin_x": current_state["origin_x"],
                    "origin_y": current_state["origin_y"],
                    "origin_z": current_state["origin_z"],
                    "beta_avg_x": current_state["beta_avg_x"],
                    "beta_avg_y": current_state["beta_avg_y"],
                    "beta_avg_z": current_state["beta_avg_z"],
                }
                observer_particle_idx = 0  # Using single-element arrays

                if sc_verbosity >= 3:
                    print(
                        f"      Full iteration: Using updated position "
                        f"x={working_x:.6e}, y={working_y:.6e}, z={working_z:.6e}"
                    )
            else:
                # Use current_state position (start of timestep)
                observer_state = current_state
                observer_particle_idx = particle_idx

            # ================================================================
            # STEP 2: Early check for COLD_START gating
            # ================================================================
            # For COLD_START, check if we should skip force computation entirely
            # This avoids expensive retarded distance calculations during startup phase
            skip_external_forces = False
            if startup_mode is StartupMode.COLD_START:
                # Check if particle has traveled far enough from origin
                # This is the same check done in _should_apply_external_forces
                # but done here to avoid computing retarded distances needlessly
                origin_position = (
                    current_state["origin_x"][particle_idx],
                    current_state["origin_y"][particle_idx],
                    current_state["origin_z"][particle_idx],
                )
                current_position = (
                    current_state["x"][particle_idx],
                    current_state["y"][particle_idx],
                    current_state["z"][particle_idx],
                )
                travel_distance = _calculate_travel_distance(
                    origin_position, current_position
                )

                # Estimate threshold without computing full retarded distances
                # Use maximum possible R from trajectory_ext bounds
                beta_avg_x = current_state["beta_avg_x"][particle_idx]
                beta_avg_y = current_state["beta_avg_y"][particle_idx]
                beta_avg_z = current_state["beta_avg_z"][particle_idx]
                beta_avg_mag = np.sqrt(beta_avg_x**2 + beta_avg_y**2 + beta_avg_z**2)

                # Estimate max R from external trajectory bounds
                # This handles arbitrary separations (mm to hundreds of meters)
                if trajectory_ext[index_traj]["x"].size > 0:
                    ext_x = trajectory_ext[index_traj]["x"]
                    ext_y = trajectory_ext[index_traj]["y"]
                    ext_z = trajectory_ext[index_traj]["z"]
                    dx = current_position[0] - ext_x
                    dy = current_position[1] - ext_y
                    dz = current_position[2] - ext_z
                    distances = np.sqrt(dx**2 + dy**2 + dz**2)
                    estimated_max_R = (
                        float(np.max(distances)) if distances.size > 0 else 1000.0
                    )
                else:
                    # Fallback if no external particles
                    estimated_max_R = 1000.0

                # Correct formula: threshold = β·R / (1 - β·n̂)
                # For early check, use conservative estimate assuming worst case
                # (minimum threshold = particle approaching head-on)
                # For β·n̂ = -1 (approaching): threshold = β·R/2
                # For β·n̂ = 0 (perpendicular): threshold = β·R
                # For β·n̂ = +1 (receding): threshold → ∞
                # Use worst case (approaching) for conservative early gating
                # Worst case: β·n̂ = -beta_avg_mag → denominator = 1 + beta_avg_mag
                estimated_threshold = (
                    beta_avg_mag * estimated_max_R / (1.0 + beta_avg_mag)
                )

                # Skip if travel distance is definitely below threshold
                if travel_distance < estimated_threshold:
                    skip_external_forces = True

            # ================================================================
            # STEP 3: Compute retarded distances to external sources
            # ================================================================
            # Only compute if forces will actually be applied
            chrono_result: Optional[ChronoMatchResult] = None
            nhat = None
            indices_bounded = None

            if not skip_external_forces:
                if startup_mode is StartupMode.APPROXIMATE_BACK_HISTORY:
                    nhat, indices_bounded = _compute_approximate_retarded_distance(
                        observer_state,
                        trajectory_ext[index_traj],
                        observer_particle_idx,
                        index_traj,
                    )
                else:
                    # For variable geometry modes, need to create trajectory with observer_state
                    if (
                        sc_convergence_mode in ("variable_geometry", "full_iteration")
                        and sc_iteration > 0
                    ):
                        # Create temporary trajectory for retarded distance calculation
                        temp_trajectory = trajectory.copy()
                        temp_trajectory[index_traj] = observer_state
                        nhat, indices_bounded, chrono_result = (
                            _compute_full_retarded_distance(
                                temp_trajectory,
                                trajectory_ext,
                                index_traj,
                                observer_particle_idx,
                                chrono_mode,
                                self_consistency,
                                timestep_h=h,
                            )
                        )
                    else:
                        nhat, indices_bounded, chrono_result = (
                            _compute_full_retarded_distance(
                                trajectory,
                                trajectory_ext,
                                index_traj,
                                particle_idx,
                                chrono_mode,
                                self_consistency,
                                timestep_h=h,
                            )
                        )

            # Initialize position and time from current_state
            # These will be updated after force calculation
            result["x"][particle_idx] = current_state["x"][particle_idx]
            result["y"][particle_idx] = current_state["y"][particle_idx]
            result["z"][particle_idx] = current_state["z"][particle_idx]
            result["t"][particle_idx] = current_state["t"][particle_idx]

            # Start accumulation from initial momentum
            # Always start from current_state - forces will be recomputed using updated beta
            accumulated_momentum_x = current_state["Px"][particle_idx]
            accumulated_momentum_y = current_state["Py"][particle_idx]
            accumulated_momentum_z = current_state["Pz"][particle_idx]
            accumulated_momentum_t = current_state["Pt"][particle_idx]

            # Accumulated field contributions (used in position update)
            accumulated_field_x = 0.0
            accumulated_field_y = 0.0
            accumulated_field_z = 0.0

            # Accumulated scalar potential (used in gamma calculation)
            accumulated_scalar_potential = 0.0

            # Extract particle properties
            particle_charge = _get_particle_charge(current_state, particle_idx)
            particle_mass = _get_particle_mass(current_state, particle_idx)

            # ================================================================
            # STEP 4: Determine if external forces should be applied
            # ================================================================
            # If we already determined to skip (COLD_START early exit), use that
            # Otherwise, do the full check with computed nhat values
            if skip_external_forces:
                apply_forces = False
            elif nhat is not None:
                # Do full gating check with actual retarded distances
                apply_forces = _should_apply_external_forces(
                    startup_mode, nhat, current_state, particle_idx
                )
            else:
                # No nhat computed, no forces to apply
                apply_forces = False

            # Use working state values for force calculations
            # These evolve across SC iterations
            particle_gamma = working_gamma
            particle_beta = (working_beta_x, working_beta_y, working_beta_z)

            # ================================================================
            # STEP 4: Compute and accumulate external force contributions
            # ================================================================
            if apply_forces and nhat["R"].size > 0:
                # Gather external particle data at retarded times (with interpolation if enabled)
                if chrono_result is not None:
                    # Use interpolation (with cubic and position interpolation if high-precision)
                    external_samples = gather_external_samples(
                        trajectory_ext,
                        indices_bounded,
                        indices_next=chrono_result.indices_next,
                        weights=chrono_result.weights,
                        indices_prev=chrono_result.indices_prev,
                        indices_next2=chrono_result.indices_next2,
                        use_cubic=chrono_result.use_cubic,
                        interpolate_positions=chrono_high_precision,
                    )
                else:
                    # Legacy path: no interpolation
                    external_samples = gather_external_samples(
                        trajectory_ext,
                        indices_bounded,
                    )

                # Compute electromagnetic force contributions
                (
                    delta_momentum_x,
                    delta_momentum_y,
                    delta_momentum_z,
                    delta_momentum_t,
                    delta_field_x,
                    delta_field_y,
                    delta_field_z,
                    delta_scalar_potential,
                ) = compute_vectorized_contributions(
                    h=h,
                    charge_i=float(particle_charge),
                    mass_i=float(particle_mass),
                    gamma_i=particle_gamma,
                    beta_vec=particle_beta,
                    nhat_nx=np.asarray(nhat["nx"], dtype=float),
                    nhat_ny=np.asarray(nhat["ny"], dtype=float),
                    nhat_nz=np.asarray(nhat["nz"], dtype=float),
                    R_separation=np.asarray(nhat["R"], dtype=float),
                    samples=external_samples,
                    apply_external=apply_forces,
                    verbosity=sc_verbosity,
                )

                # Debug: Log what forces were computed
                if sc_verbosity >= 3 and sc_enabled:
                    print(
                        f"      Force contributions: ΔPx={delta_momentum_x:.15e}, "
                        f"ΔPy={delta_momentum_y:.15e}, ΔPz={delta_momentum_z:.15e}, "
                        f"ΔPt={delta_momentum_t:.15e}"
                    )
                    print(
                        f"      Using particle_beta=({particle_beta[0]:.15e}, "
                        f"{particle_beta[1]:.15e}, {particle_beta[2]:.15e}), "
                        f"gamma={particle_gamma:.15e}"
                    )

                # Accumulate momentum changes
                accumulated_momentum_x += delta_momentum_x
                accumulated_momentum_y += delta_momentum_y
                accumulated_momentum_z += delta_momentum_z
                accumulated_momentum_t += delta_momentum_t

                # Accumulate field contributions
                accumulated_field_x += delta_field_x
                accumulated_field_y += delta_field_y
                accumulated_field_z += delta_field_z

                # Accumulate scalar potential
                accumulated_scalar_potential += delta_scalar_potential

                if sc_verbosity >= 3 and sc_enabled and sc_iteration > 0:
                    print(
                        f"      After forces: ΔPt={delta_momentum_t:.15e}, "
                        f"accumulated_pt={accumulated_momentum_t:.15e}"
                    )

            # ================================================================
            # STEP 4: Update momentum and derive gamma from Pt
            # ================================================================
            result["Px"][particle_idx] = accumulated_momentum_x
            result["Py"][particle_idx] = accumulated_momentum_y
            result["Pz"][particle_idx] = accumulated_momentum_z
            result["Pt"][particle_idx] = accumulated_momentum_t

            # ================================================================
            # STEP 4a: Correct Pt during SC iterations based on mode
            # ================================================================
            # CRITICAL: Enforce constraints at each iteration
            # Mode determines HOW we correct Pt, but both modes check both errors
            if sc_enabled and sc_iteration > 0:
                # Compute mass-shell-constrained Pt
                Px_64 = np.float64(result["Px"][particle_idx])
                Py_64 = np.float64(result["Py"][particle_idx])
                Pz_64 = np.float64(result["Pz"][particle_idx])
                P_spatial_sq = Px_64**2 + Py_64**2 + Pz_64**2
                mass_shell_rhs = np.float64(particle_mass * C_MMNS) ** 2
                Pt_from_mass_shell = np.sqrt(P_spatial_sq + mass_shell_rhs)

                Pt_before_correction = np.float64(result["Pt"][particle_idx])

                # Determine Pt and P correction based on mode
                if sc_convergence_mode in (
                    "fixed_geometry",
                    "variable_geometry",
                    "mass_shell_only",
                    "full_iteration",
                ):
                    # Modes 1 & 2: Project Pt onto mass-shell (asymmetric relaxation)
                    Pt_corrected = Pt_from_mass_shell

                    if sc_verbosity >= 3:
                        mode_name = (
                            sc_convergence_mode
                            if sc_convergence_mode
                            in ("fixed_geometry", "variable_geometry")
                            else (
                                "fixed_geometry"
                                if sc_convergence_mode == "mass_shell_only"
                                else "variable_geometry"
                            )
                        )
                        print(
                            f"      Mode: {mode_name}, Pt_ms={Pt_from_mass_shell:.6e}"
                        )

                    # Apply relaxation to Pt only (asymmetric)
                    relaxation_weight = sc_mass_shell_relaxation
                    Pt_final = (
                        relaxation_weight * Pt_corrected
                        + (1.0 - relaxation_weight) * Pt_before_correction
                    )

                    result["Pt"][particle_idx] = float(Pt_final)
                    # P_xyz unchanged (from forces)

                else:
                    raise ValueError(f"Unknown convergence_mode: {sc_convergence_mode}")

                # Log relaxation details
                if sc_verbosity >= 3:
                    correction_magnitude = abs(Pt_final - Pt_before_correction)
                    print(
                        f"      After relaxation (α={relaxation_weight}): "
                        f"Pt {Pt_before_correction:.6e} → {Pt_final:.6e} "
                        f"(Δ={correction_magnitude:.6e})"
                    )

                if sc_verbosity >= 3:
                    correction_magnitude = abs(Pt_final - Pt_before_correction)
                    print(
                        f"      After relaxation (α={relaxation_weight}): "
                        f"Pt {Pt_before_correction:.6e} → {Pt_final:.6e} "
                        f"(Δ={correction_magnitude:.6e})"
                    )

            # ================================================================
            # STEP 4b: Compute gamma from energy
            # ================================================================
            # Gamma from relativistic energy with scalar potential correction:
            # γ = (Pt - q·Φ) / (mc) where Φ = Σ(q_j / (R_sep_j * k_factor_j))
            # This gives the correct kinetic energy, accounting for electromagnetic potential
            # Use float64 precision for gamma calculation
            scalar_potential_contribution = np.float64(
                particle_charge * accumulated_scalar_potential
            )
            kinetic_energy = (
                np.float64(result["Pt"][particle_idx]) - scalar_potential_contribution
            )
            gamma_from_energy = kinetic_energy / np.float64(particle_mass * C_MMNS)
            result["gamma"][particle_idx] = gamma_from_energy

            # Calculate gamma from mass-shell constraint for logging (if needed)
            # Compute Pt from mass-shell: Pt = √(P² + (mc)²)
            Px_64 = np.float64(result["Px"][particle_idx])
            Py_64 = np.float64(result["Py"][particle_idx])
            Pz_64 = np.float64(result["Pz"][particle_idx])
            P_spatial_sq = Px_64**2 + Py_64**2 + Pz_64**2
            mass_shell_rhs = np.float64(particle_mass * C_MMNS) ** 2
            Pt_from_mass_shell = np.sqrt(P_spatial_sq + mass_shell_rhs)
            Pt_before_projection = np.float64(result["Pt"][particle_idx])

            gamma_mass_shell = Pt_from_mass_shell / (particle_mass * C_MMNS)

            if sc_verbosity >= 3 and sc_enabled and sc_iteration > 0:
                # Use Pt BEFORE projection to show the actual difference
                gamma_from_conjugate_before = Pt_before_projection / (
                    particle_mass * C_MMNS
                )
                print(f"      γ_energy (Pt - q·Φ)/(mc) = {gamma_from_energy:.15e}")
                print(
                    f"      γ_conjugate (Pt/(mc), before projection) = {gamma_from_conjugate_before:.15e}"
                )
                print(
                    f"      γ_mass_shell (√(P²+(mc)²)/(mc)) = {gamma_mass_shell:.15e}"
                )
                print(
                    f"      Scalar potential term q·Φ = {scalar_potential_contribution:.15e}"
                )
                # Show the mass-shell violation
                mass_shell_violation = abs(
                    gamma_from_conjugate_before - gamma_mass_shell
                )
                print(
                    f"      Mass-shell violation |γ_conjugate - γ_mass_shell| = {mass_shell_violation:.15e}"
                )

            # Update x^0 = dt = dtau * gamma
            result["t"][particle_idx] = (
                current_state["t"][particle_idx] + h * result["gamma"][particle_idx]
            )

            # ================================================================
            # STEP 5: Update spatial positions
            # ================================================================
            # Position update in proper time formulation: dx/dτ = v·γ
            # Since h = dτ = dt/γ, we have: dx = v·γ·dτ = (P/m)·h
            # where v = P/(γ·m) and γ cancels in the product v·γ = P/m
            result["x"][particle_idx] = current_state["x"][particle_idx] + h / (
                particle_mass
            ) * (result["Px"][particle_idx] - accumulated_field_x * particle_mass)
            result["y"][particle_idx] = current_state["y"][particle_idx] + h / (
                particle_mass
            ) * (result["Py"][particle_idx] - accumulated_field_y * particle_mass)
            result["z"][particle_idx] = current_state["z"][particle_idx] + h / (
                particle_mass
            ) * (result["Pz"][particle_idx] - accumulated_field_z * particle_mass)

            # ================================================================
            # STEP 6: Compute velocity (beta) from position changes
            # ================================================================
            position_change_x = (
                result["x"][particle_idx] - current_state["x"][particle_idx]
            )
            position_change_y = (
                result["y"][particle_idx] - current_state["y"][particle_idx]
            )
            position_change_z = (
                result["z"][particle_idx] - current_state["z"][particle_idx]
            )

            # β = Δx/(c·Δt) using the actual coordinate-time step
            # Since Δx = (P/m)·h and Δt = γ·h, we get β = P/(γ·m·c)
            coordinate_dt = result["t"][particle_idx] - current_state["t"][particle_idx]
            if coordinate_dt == 0.0:
                coordinate_dt = h * result["gamma"][particle_idx]
            beta_x = position_change_x / (C_MMNS * coordinate_dt)
            beta_y = position_change_y / (C_MMNS * coordinate_dt)
            beta_z = position_change_z / (C_MMNS * coordinate_dt)

            # Enforce speed of light limit IMMEDIATELY after calculation
            beta_x_limited, beta_y_limited, beta_z_limited = _limit_beta_magnitude(
                beta_x,
                beta_y,
                beta_z,
            )

            result["bx"][particle_idx] = beta_x_limited
            result["by"][particle_idx] = beta_y_limited
            result["bz"][particle_idx] = beta_z_limited

            # Compute gamma from the (possibly limited) beta
            # This is compared against gamma_from_energy for self-consistency
            gamma_from_velocity = _calculate_gamma_from_beta(
                beta_x_limited, beta_y_limited, beta_z_limited
            )

            # Debug: Print newly computed beta on all iterations when verbosity >= 3
            if sc_verbosity >= 3:
                print(
                    f"      Newly computed β: βx={beta_x_limited:.15e}, "
                    f"βy={beta_y_limited:.15e}, βz={beta_z_limited:.15e}"
                )

            if sc_verbosity >= 3 and sc_enabled and sc_iteration > 0:
                beta_total = np.sqrt(
                    beta_x_limited**2 + beta_y_limited**2 + beta_z_limited**2
                )
                print(
                    f"      γ_velocity (from β over Δt={coordinate_dt:.15e}) = {gamma_from_velocity:.15e}, "
                    f"βtot={beta_total:.15e}"
                )

            # ================================================================
            # GAMMA RECONCILIATION (configurable)
            # ================================================================
            # Reconcile gamma_from_energy (from Pt) and gamma_from_velocity (from beta)
            # to prevent dual-gamma inconsistency and blowups
            if (
                sc_enabled
                and self_consistency is not None
                and self_consistency.gamma_reconciliation_enabled
            ):
                gamma_from_energy = result["gamma"][particle_idx]
                beta_total = np.sqrt(
                    beta_x_limited**2 + beta_y_limited**2 + beta_z_limited**2
                )

                # Determine reconciliation method
                method = self_consistency.gamma_reconciliation_method

                if method == GammaReconciliationMethod.DISABLED:
                    # No reconciliation - use energy-based gamma (already set)
                    gamma_reconciled = gamma_from_energy
                    alpha = 1.0  # For logging

                elif method == GammaReconciliationMethod.USE_VELOCITY:
                    # Always use velocity-based gamma
                    gamma_reconciled = gamma_from_velocity
                    alpha = 0.0  # For logging

                elif method == GammaReconciliationMethod.USE_ENERGY:
                    # Always use energy-based gamma (same as DISABLED)
                    gamma_reconciled = gamma_from_energy
                    alpha = 1.0  # For logging

                elif method == GammaReconciliationMethod.FIXED_WEIGHTED:
                    # Fixed 50/50 weighted average (or custom weight)
                    alpha = self_consistency.gamma_reconciliation_fixed_weight
                    gamma_reconciled = (
                        alpha * gamma_from_energy + (1.0 - alpha) * gamma_from_velocity
                    )

                elif method == GammaReconciliationMethod.ADAPTIVE_WEIGHTED:
                    # Adaptive weighting based on velocity regime (default)
                    low_threshold = (
                        self_consistency.gamma_reconciliation_low_beta_threshold
                    )
                    high_threshold = (
                        self_consistency.gamma_reconciliation_high_beta_threshold
                    )
                    low_weight = self_consistency.gamma_reconciliation_low_beta_weight
                    high_weight = self_consistency.gamma_reconciliation_high_beta_weight
                    mid_weight = self_consistency.gamma_reconciliation_mid_beta_weight

                    if beta_total < low_threshold:
                        alpha = low_weight  # Trust energy at lower velocities
                    elif beta_total > high_threshold:
                        alpha = high_weight  # Trust velocity near speed of light
                    else:
                        alpha = mid_weight  # Balanced weighting

                    gamma_reconciled = (
                        alpha * gamma_from_energy + (1.0 - alpha) * gamma_from_velocity
                    )
                else:
                    # Fallback to energy-based gamma for unknown methods
                    gamma_reconciled = gamma_from_energy
                    alpha = 1.0

                # Update gamma and Pt to be consistent
                result["gamma"][particle_idx] = gamma_reconciled
                result["Pt"][particle_idx] = gamma_reconciled * particle_mass * C_MMNS

                # Rescale spatial momentum to preserve mass shell: Pt² = P² + (mc)²
                P_magnitude_sq = (
                    result["Pt"][particle_idx] ** 2 - (particle_mass * C_MMNS) ** 2
                )
                if P_magnitude_sq > 0:
                    P_magnitude = np.sqrt(P_magnitude_sq)
                    current_P_mag = np.sqrt(
                        result["Px"][particle_idx] ** 2
                        + result["Py"][particle_idx] ** 2
                        + result["Pz"][particle_idx] ** 2
                    )
                    if current_P_mag > 1e-20:
                        scale_factor = P_magnitude / current_P_mag
                        result["Px"][particle_idx] *= scale_factor
                        result["Py"][particle_idx] *= scale_factor
                        result["Pz"][particle_idx] *= scale_factor

                if sc_verbosity >= 3:
                    print(
                        f"      Gamma reconciliation ({method.name}): α={alpha:.2f}, β={beta_total:.6f}, "
                        f"γ_energy={gamma_from_energy:.6e}, γ_velocity={gamma_from_velocity:.6e}, "
                        f"γ_reconciled={gamma_reconciled:.6e}"
                    )

            # ================================================================
            # STEP 7: Compute acceleration (beta-dot)
            # ================================================================
            beta_change_x = (
                result["bx"][particle_idx] - current_state["bx"][particle_idx]
            )
            beta_change_y = (
                result["by"][particle_idx] - current_state["by"][particle_idx]
            )
            beta_change_z = (
                result["bz"][particle_idx] - current_state["bz"][particle_idx]
            )

            # β-dot = dβ/dt where dt is coordinate time
            # Use the same coordinate-time interval employed for β
            time_factor = C_MMNS * coordinate_dt
            result["bdotx"][particle_idx] = beta_change_x / time_factor
            result["bdoty"][particle_idx] = beta_change_y / time_factor
            result["bdotz"][particle_idx] = beta_change_z / time_factor

            # ================================================================
            # STEP 8: Apply radiation reaction corrections
            # ================================================================
            particle_char_time = _get_particle_char_time(current_state, particle_idx)

            # Compute current and previous beta_dot magnitudes
            beta_dot_magnitude = np.sqrt(
                result["bdotx"][particle_idx] ** 2
                + result["bdoty"][particle_idx] ** 2
                + result["bdotz"][particle_idx] ** 2
            )
            beta_dot_prev_magnitude = np.sqrt(
                current_state["bdotx"][particle_idx] ** 2
                + current_state["bdoty"][particle_idx] ** 2
                + current_state["bdotz"][particle_idx] ** 2
            )

            # Apply radiation reaction if beta_dot has changed significantly (0.1% default)
            beta_dot_change_fraction = (
                abs(beta_dot_magnitude - beta_dot_prev_magnitude)
                / beta_dot_prev_magnitude
                if beta_dot_prev_magnitude > 0.0
                else 0.0
            )
            if beta_dot_change_fraction >= 0.001:  # 0.1% threshold
                # Compute radiation reaction for all three axes
                rad_lhs_x, rad_rhs_x = _compute_radiation_reaction_term(
                    axis="x",
                    beta_component=result["bx"][particle_idx],
                    beta_dot_component=result["bdotx"][particle_idx],
                    gamma_current=result["gamma"][particle_idx],
                    gamma_previous=current_state["gamma"][particle_idx],
                    time_step=h,
                    mass=float(particle_mass),
                )

                rad_lhs_y, rad_rhs_y = _compute_radiation_reaction_term(
                    axis="y",
                    beta_component=result["by"][particle_idx],
                    beta_dot_component=result["bdoty"][particle_idx],
                    gamma_current=result["gamma"][particle_idx],
                    gamma_previous=current_state["gamma"][particle_idx],
                    time_step=h,
                    mass=float(particle_mass),
                )

                rad_lhs_z, rad_rhs_z = _compute_radiation_reaction_term(
                    axis="z",
                    beta_component=result["bz"][particle_idx],
                    beta_dot_component=result["bdotz"][particle_idx],
                    gamma_current=result["gamma"][particle_idx],
                    gamma_previous=current_state["gamma"][particle_idx],
                    time_step=h,
                    mass=float(particle_mass),
                )

                # Apply corrections to all three axes
                radiation_correction_x = (
                    particle_char_time
                    * (rad_lhs_x + rad_rhs_x)
                    / (particle_mass * C_MMNS)
                )
                radiation_correction_y = (
                    particle_char_time
                    * (rad_lhs_y + rad_rhs_y)
                    / (particle_mass * C_MMNS)
                )
                radiation_correction_z = (
                    particle_char_time
                    * (rad_lhs_z + rad_rhs_z)
                    / (particle_mass * C_MMNS)
                )

                result["bdotx"][particle_idx] += radiation_correction_x
                result["bdoty"][particle_idx] += radiation_correction_y
                result["bdotz"][particle_idx] += radiation_correction_z

            # ================================================================
            # STEP 9: Update running average of beta
            # ================================================================
            previous_beta_avg = (
                current_state["beta_avg_x"][particle_idx],
                current_state["beta_avg_y"][particle_idx],
                current_state["beta_avg_z"][particle_idx],
            )
            previous_sample_count = float(current_state["beta_samples"][particle_idx])

            new_beta = (
                result["bx"][particle_idx],
                result["by"][particle_idx],
                result["bz"][particle_idx],
            )

            updated_beta_avg, updated_sample_count = _update_beta_running_average(
                previous_beta_avg,
                previous_sample_count,
                new_beta,
            )

            result["beta_samples"][particle_idx] = updated_sample_count
            result["beta_avg_x"][particle_idx] = updated_beta_avg[0]
            result["beta_avg_y"][particle_idx] = updated_beta_avg[1]
            result["beta_avg_z"][particle_idx] = updated_beta_avg[2]

            # ================================================================
            # STEP 10: Update working state and check convergence
            # ================================================================
            # Update working state with newly computed values for next iteration
            new_working_beta_x = result["bx"][particle_idx]
            new_working_beta_y = result["by"][particle_idx]
            new_working_beta_z = result["bz"][particle_idx]
            new_working_gamma = result["gamma"][particle_idx]

            if sc_enabled and sc_iteration > 0:
                # Check mass-shell convergence
                (
                    converged,
                    mass_shell_error_rel,
                ) = _check_mass_shell_convergence(
                    result["Pt"][particle_idx],
                    result["Px"][particle_idx],
                    result["Py"][particle_idx],
                    result["Pz"][particle_idx],
                    particle_mass,
                    C_MMNS,
                    sc_target_ms_tolerance,
                )

                # Set dummy gamma consistency error (not checked)
                gamma_consistency_error = 0.0

                if converged:
                    if sc_verbosity > 0:
                        _print_convergence_info(
                            particle_idx,
                            sc_iteration,
                            gamma_from_velocity,
                            gamma_from_energy,
                            gamma_mass_shell,
                            mass_shell_error_rel,
                            gamma_consistency_error,
                            converged=True,
                            max_iterations=sc_max_iterations,
                            verbosity=sc_verbosity,
                            step_idx=step_idx,
                            convergence_mode=sc_convergence_mode,
                            particle_position=(
                                result["x"][particle_idx],
                                result["y"][particle_idx],
                                result["z"][particle_idx],
                            ),
                            particle_time=result["t"][particle_idx],
                        )
                    break
                elif sc_iteration == sc_max_iterations - 1:
                    if sc_verbosity > 0:
                        _print_convergence_info(
                            particle_idx,
                            sc_iteration,
                            gamma_from_velocity,
                            gamma_from_energy,
                            gamma_mass_shell,
                            mass_shell_error_rel,
                            gamma_consistency_error,
                            converged=False,
                            max_iterations=sc_max_iterations,
                            verbosity=sc_verbosity,
                            step_idx=step_idx,
                            convergence_mode=sc_convergence_mode,
                            particle_position=(
                                result["x"][particle_idx],
                                result["y"][particle_idx],
                                result["z"][particle_idx],
                            ),
                            particle_time=result["t"][particle_idx],
                        )

            # Update working state for next iteration
            working_beta_x = new_working_beta_x
            working_beta_y = new_working_beta_y
            working_beta_z = new_working_beta_z
            working_gamma = new_working_gamma
            working_x = result["x"][particle_idx]
            working_y = result["y"][particle_idx]
            working_z = result["z"][particle_idx]

            # Gamma blowup detection: ALL blowups now trigger retry attempts
            # The integration runner will reduce timestep and retry, only marking
            # particle as dead after exhausting all retry attempts.
            # - Soft threshold (1e8): likely recoverable with smaller timestep
            # - Hard threshold (1e20 or NaN/Inf): less likely but still attempt recovery
            if sc_enabled:
                is_nan_or_inf = np.isnan(working_gamma) or np.isinf(working_gamma)
                gamma_soft_threshold = 1e8
                gamma_hard_threshold = 1e20

                # Check for any gamma blowup
                if is_nan_or_inf or working_gamma > gamma_soft_threshold:
                    # Skip if particle already dead (suppress redundant errors)
                    already_dead = (
                        "_dead_particles" in result
                        and result["_dead_particles"][particle_idx]
                    )
                    if already_dead:
                        break  # Exit self-consistency loop for this particle

                    # Determine if this is a hard or soft blowup (for logging/metrics)
                    is_hard = is_nan_or_inf or working_gamma > gamma_hard_threshold

                    if sc_verbosity >= 1:
                        severity = "Hard" if is_hard else "Soft"
                        print(
                            f"    [WARNING] Step {step_idx if step_idx is not None else '?'}, "
                            f"Particle {particle_idx}/{num_particles}, Iteration {sc_iteration}: "
                            f"{severity} gamma blowup (γ={working_gamma:.2e}), requesting timestep reduction"
                        )

                    # Raise exception to signal integration_runner to reduce timestep
                    # The runner will attempt recovery for ALL blowups (soft and hard)
                    raise GammaBlowupError(
                        step_idx if step_idx is not None else -1,
                        particle_idx,
                        working_gamma,
                        sc_iteration,
                        is_hard_blowup=is_hard,
                    )

        # ================================================================
        # AFTER self-consistency loop: Apply mass-shell projection if needed
        # ================================================================
        if sc_enabled:
            # Check final mass-shell error
            Px_64 = np.float64(result["Px"][particle_idx])
            Py_64 = np.float64(result["Py"][particle_idx])
            Pz_64 = np.float64(result["Pz"][particle_idx])
            Pt_64 = np.float64(result["Pt"][particle_idx])
            P_spatial_sq = Px_64**2 + Py_64**2 + Pz_64**2
            mass_shell_rhs = np.float64(particle_mass * C_MMNS) ** 2
            mass_shell_error_final = (
                np.abs(Pt_64**2 - P_spatial_sq - mass_shell_rhs) / mass_shell_rhs
            )

            if mass_shell_error_final > sc_mass_shell_tolerance:
                # Projection needed as final safety net
                Pt_from_mass_shell = np.sqrt(P_spatial_sq + mass_shell_rhs)

                if sc_verbosity >= 2:
                    print(
                        f"    ⚠️  Final mass-shell projection: Pt {Pt_64:.6e} → "
                        f"{Pt_from_mass_shell:.6e} (error was {mass_shell_error_final:.2e})"
                    )

                result["Pt"][particle_idx] = float(Pt_from_mass_shell)

                # Recalculate gamma with projected Pt
                scalar_potential_contribution = (
                    particle_charge * accumulated_scalar_potential
                )
                kinetic_energy = (
                    result["Pt"][particle_idx] - scalar_potential_contribution
                )
                result["gamma"][particle_idx] = kinetic_energy / (
                    particle_mass * C_MMNS
                )

    # Log summary if any particles died in this step
    if particles_marked_dead_this_step > 0:
        print(
            f"  [SUMMARY] Step {step_idx if step_idx is not None else '?'}: "
            f"{particles_marked_dead_this_step}/{num_particles} particles marked dead in this step"
        )

    return result


__all__ = ["retarded_equations_of_motion"]
