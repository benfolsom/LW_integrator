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

from typing import Optional

import numpy as np

from .constants import C_MMNS
from .distances import (
    chrono_match_indices,
    compute_instantaneous_distance,
    compute_retarded_distance,
)
from .self_consistency import SelfConsistencyConfig
from .types import (
    ChronoMatchingMode,
    ParticleState,
    SimulationType,
    StartupMode,
    Trajectory,
)
from .vectorized_interactions import (
    compute_vectorized_contributions,
    gather_external_samples,
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
) -> tuple[bool, float, int, int]:
    """Extract self-consistency configuration parameters.

    Returns
    -------
    tuple[bool, float, int, int]
        A tuple containing (enabled, tolerance, max_iterations, verbosity).
    """
    is_enabled = self_consistency is not None and self_consistency.enabled
    tolerance = self_consistency.tolerance if self_consistency is not None else 1e-6
    max_iterations = (
        self_consistency.max_iterations if self_consistency is not None else 1
    )
    verbosity = self_consistency.verbosity if self_consistency is not None else 0

    return is_enabled, tolerance, max_iterations, verbosity


def _initialize_result_state(current_state: ParticleState) -> ParticleState:
    """Create a copy of the current particle state for the next time step.

    Parameters
    ----------
    current_state : ParticleState
        The current state at this time step.

    Returns
    -------
    ParticleState
        A deep copy with all arrays duplicated.
    """
    return {
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

    Returns
    -------
    tuple[dict, np.ndarray]
        A tuple of (nhat dictionary, bounded_indices array).
    """
    sample_count = len(external_state["x"])
    indices_bounded = np.full(sample_count, time_step_idx, dtype=int)

    nhat = compute_instantaneous_distance(current_state, external_state, particle_idx)

    # Correct distance for source motion during light travel time
    beta_ext_dot_nhat = (
        external_state["bx"] * nhat["nx"]
        + external_state["by"] * nhat["ny"]
        + external_state["bz"] * nhat["nz"]
    )
    nhat["R"] = nhat["R"] * (1.0 + beta_ext_dot_nhat)

    return nhat, indices_bounded


def _compute_full_retarded_distance(
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    time_step_idx: int,
    particle_idx: int,
    chrono_mode: ChronoMatchingMode,
) -> tuple[dict, np.ndarray]:
    """Compute retarded distance using full chronological matching.

    This uses the complete trajectory history to find the proper retarded time
    for each external source particle.

    Returns
    -------
    tuple[dict, np.ndarray]
        A tuple of (nhat dictionary, bounded_indices array).
    """
    retarded_indices = chrono_match_indices(
        trajectory,
        trajectory_ext,
        time_step_idx,
        particle_idx,
        mode=chrono_mode,
    )

    max_external_idx = len(trajectory_ext) - 1
    indices_bounded = np.minimum(np.maximum(retarded_indices, 0), max_external_idx)

    nhat = compute_retarded_distance(
        trajectory,
        trajectory_ext,
        time_step_idx,
        particle_idx,
        indices_bounded,
    )

    return nhat, indices_bounded


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

    The threshold is based on the retarded distance and average velocity of the
    observer particle, ensuring the particle has traveled far enough for
    retardation effects to be meaningful.
    """
    beta_avg_dot_nhat = (
        beta_avg_x * nhat["nx"] + beta_avg_y * nhat["ny"] + beta_avg_z * nhat["nz"]
    )
    thresholds = nhat["R"] * (1.0 - beta_avg_dot_nhat)

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


def _calculate_gamma_from_beta(beta_x: float, beta_y: float, beta_z: float) -> float:
    """Calculate Lorentz factor from velocity components.

    γ = 1 / √(1 - β²)

    Uses float64 precision to handle extremely relativistic particles accurately.
    """
    # Use float64 for high precision
    bx64 = np.float64(beta_x)
    by64 = np.float64(beta_y)
    bz64 = np.float64(beta_z)

    beta_squared = bx64**2 + by64**2 + bz64**2

    # Clamp beta_squared just below 1.0 to prevent infinity
    # This corresponds to the same limit used in _limit_beta_magnitude
    max_beta_squared = (np.float64(1.0) - np.float64(1e-16)) ** 2
    if beta_squared >= max_beta_squared:
        beta_squared = max_beta_squared

    denominator = np.float64(1.0) - beta_squared

    # With float64 precision, denominator should never be exactly zero
    # if beta was properly limited, but check anyway
    if denominator <= np.float64(0.0):
        # Use the maximum gamma corresponding to our beta limit
        return float(1.0 / np.sqrt(np.float64(1.0) - max_beta_squared))

    return float(1.0 / np.sqrt(denominator))


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


def _should_apply_radiation_reaction(
    lhs_term: float,
    rhs_term: float,
    char_time: float,
) -> bool:
    """Determine if radiation reaction correction should be applied.

    Radiation effects are only significant when the force terms exceed
    a threshold based on the characteristic time scale.
    """
    threshold = char_time / 1e1  # Changed from 1e2 to 1e1 for 10x more sensitivity
    return lhs_term > threshold or rhs_term > threshold


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


def _check_self_consistency_convergence(
    gamma_new: float,
    gamma_previous: float,
    tolerance: float,
) -> tuple[bool, float, float]:
    """Check if gamma has converged between self-consistency iterations.

    Returns
    -------
    tuple[bool, float, float]
        (has_converged, absolute_change, relative_change)
    """
    gamma_absolute_change = abs(gamma_new - gamma_previous)
    gamma_relative_change = gamma_absolute_change / max(abs(gamma_previous), 1e-12)
    has_converged = gamma_relative_change < tolerance

    return has_converged, gamma_absolute_change, gamma_relative_change


def _print_convergence_info(
    particle_idx: int,
    iteration: int,
    gamma_from_velocity: float,
    gamma_from_energy: float,
    gamma_abs_change: float,
    gamma_rel_change: float,
    converged: bool,
    max_iterations: int,
    verbosity: int = 1,
) -> None:
    """Print debug information about self-consistency convergence.

    Compares gamma computed from velocity (kinematics) vs gamma computed
    from energy (conjugate momentum). Self-consistency requires these to match.

    Parameters
    ----------
    gamma_from_velocity : float
        Gamma computed from velocity: γ = 1/√(1-β²)
    gamma_from_energy : float
        Gamma computed from kinetic energy: γ = (Pt - q·Φ)/(mc)
    verbosity : int
        0 = silent (no output)
        1 = basic (one line per particle)
        2 = detailed (full convergence details)
    """
    if verbosity == 0:
        return

    # Basic output (verbosity >= 1)
    if converged:
        status = f"converged in {iteration + 1} iter"
    else:
        status = f"max iter ({max_iterations}) reached"

    if verbosity == 1:
        # Truncated: one line per particle
        print(
            f"    P{particle_idx}: {status}, Δγ/γ={gamma_rel_change:.6e}, "
            f"γ_energy={gamma_from_energy:.6e}"
        )
    else:  # verbosity >= 2
        # Detailed: multi-line output with full precision
        print(f"    Particle {particle_idx}: {status}")
        print(f"      Δγ/γ = {gamma_rel_change:.15e}")
        print(f"      γ_velocity = {gamma_from_velocity:.15e}")
        print(f"      γ_energy   = {gamma_from_energy:.15e}")
        print(f"      Δγ_abs = {gamma_abs_change:.15e}")


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

    # Extract self-consistency configuration
    (
        sc_enabled,
        sc_tolerance,
        sc_max_iterations,
        sc_verbosity,
    ) = _extract_self_consistency_params(self_consistency)

    # Process each particle independently
    for particle_idx in range(num_particles):
        # Self-consistency loop: iterate until gamma converges
        for sc_iteration in range(sc_max_iterations):
            if sc_verbosity >= 2 and sc_iteration > 0:
                print(
                    f"    Particle {particle_idx} iteration {sc_iteration}: "
                    f"Starting refinement"
                )

            # ================================================================
            # STEP 1: Compute retarded distances to external sources
            # ================================================================
            if startup_mode is StartupMode.APPROXIMATE_BACK_HISTORY:
                nhat, indices_bounded = _compute_approximate_retarded_distance(
                    current_state,
                    trajectory_ext[index_traj],
                    particle_idx,
                    index_traj,
                )
            else:
                nhat, indices_bounded = _compute_full_retarded_distance(
                    trajectory,
                    trajectory_ext,
                    index_traj,
                    particle_idx,
                    chrono_mode,
                )

            # Copy position and time (unchanged by forces)
            result["x"][particle_idx] = current_state["x"][particle_idx]
            result["y"][particle_idx] = current_state["y"][particle_idx]
            result["z"][particle_idx] = current_state["z"][particle_idx]
            result["t"][particle_idx] = current_state["t"][particle_idx]

            # Start accumulation from initial momentum
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
            # STEP 2: Determine if external forces should be applied
            # ================================================================
            apply_forces = _should_apply_external_forces(
                startup_mode, nhat, current_state, particle_idx
            )

            # Get gamma for this iteration (may use updated values)
            # This gamma will be used consistently for forces AND positions
            particle_gamma, _ = _get_current_particle_gamma_and_beta(
                current_state, result, particle_idx, sc_iteration, sc_enabled
            )

            # Get beta separately for force calculations
            if sc_enabled and sc_iteration > 0:
                particle_beta = (
                    result["bx"][particle_idx],
                    result["by"][particle_idx],
                    result["bz"][particle_idx],
                )
            else:
                particle_beta = (
                    current_state["bx"][particle_idx],
                    current_state["by"][particle_idx],
                    current_state["bz"][particle_idx],
                )

            # ================================================================
            # STEP 3: Compute and accumulate external force contributions
            # ================================================================
            if apply_forces and nhat["R"].size > 0:
                # Gather external particle data at retarded times
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

                if sc_verbosity >= 2 and sc_enabled and sc_iteration > 0:
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

            # Gamma from relativistic energy with scalar potential correction:
            # γ = (Pt - q²·Φ) / (mc) where Φ = Σ(q_j / (R_sep_j * k_factor_j))
            # This gives the correct kinetic energy, accounting for electromagnetic potential
            scalar_potential_contribution = (
                particle_charge * accumulated_scalar_potential
            )
            kinetic_energy = result["Pt"][particle_idx] - scalar_potential_contribution
            gamma_from_energy = kinetic_energy / (particle_mass * C_MMNS)
            result["gamma"][particle_idx] = gamma_from_energy

            if sc_verbosity >= 2 and sc_enabled and sc_iteration > 0:
                gamma_from_conjugate = result["Pt"][particle_idx] / (
                    particle_mass * C_MMNS
                )
                print(f"      Gamma from kinetic energy: γ={gamma_from_energy:.15e}")
                print(f"      Gamma from conjugate Pt: γ={gamma_from_conjugate:.15e}")
                print(
                    f"      Scalar potential correction: {scalar_potential_contribution:.15e}"
                )

            # Update proper time
            result["t"][particle_idx] = (
                current_state["t"][particle_idx] + h * result["gamma"][particle_idx]
            )

            # ================================================================
            # STEP 5: Update spatial positions
            # ================================================================
            # Correct relativistic position update: Δx = v·h = (P_kinetic/(γ·m))·h
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

            # β = Δx/(c·h) for coordinate time stepping
            # No gamma factor needed here since positions were updated with 1/γ
            beta_x = position_change_x / (C_MMNS * h * particle_gamma)
            beta_y = position_change_y / (C_MMNS * h * particle_gamma)
            beta_z = position_change_z / (C_MMNS * h * particle_gamma)

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

            if sc_verbosity >= 2 and sc_enabled and sc_iteration > 0:
                beta_total = np.sqrt(
                    beta_x_limited**2 + beta_y_limited**2 + beta_z_limited**2
                )
                print(
                    f"      Gamma from β: γ={gamma_from_velocity:.15e}, "
                    f"βtot={beta_total:.15e}"
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
            # Use the gamma from this iteration for time dilation
            time_factor = C_MMNS * h * result["gamma"][particle_idx] * particle_gamma
            result["bdotx"][particle_idx] = beta_change_x / time_factor
            result["bdoty"][particle_idx] = beta_change_y / time_factor
            result["bdotz"][particle_idx] = beta_change_z / time_factor

            # ================================================================
            # STEP 8: Apply radiation reaction corrections
            # ================================================================
            particle_char_time = _get_particle_char_time(current_state, particle_idx)

            # Compute z-component radiation reaction first
            rad_lhs_z, rad_rhs_z = _compute_radiation_reaction_term(
                axis="z",
                beta_component=result["bz"][particle_idx],
                beta_dot_component=result["bdotz"][particle_idx],
                gamma_current=result["gamma"][particle_idx],
                gamma_previous=current_state["gamma"][particle_idx],
                time_step=h,
                mass=float(particle_mass),
            )

            # Only apply radiation reaction if forces are significant
            if _should_apply_radiation_reaction(
                rad_lhs_z, rad_rhs_z, float(particle_char_time)
            ):
                radiation_correction_z = (
                    particle_char_time
                    * (rad_lhs_z + rad_rhs_z)
                    / (particle_mass * C_MMNS)
                )
                result["bdotz"][particle_idx] += radiation_correction_z

                # If z-axis needs correction, apply to x and y as well
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

                result["bdotx"][particle_idx] += radiation_correction_x
                result["bdoty"][particle_idx] += radiation_correction_y

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
            # STEP 10: Check self-consistency convergence
            # ================================================================
            if sc_enabled and sc_iteration > 0:
                # Self-consistency requires gamma from energy to match gamma from velocity
                gamma_from_energy = float(result["gamma"][particle_idx])

                (
                    has_converged,
                    gamma_abs_change,
                    gamma_rel_change,
                ) = _check_self_consistency_convergence(
                    gamma_from_energy,
                    gamma_from_velocity,
                    sc_tolerance,
                )

                if has_converged:
                    if sc_verbosity > 0:
                        _print_convergence_info(
                            particle_idx,
                            sc_iteration,
                            gamma_from_velocity,
                            gamma_from_energy,
                            gamma_abs_change,
                            gamma_rel_change,
                            converged=True,
                            max_iterations=sc_max_iterations,
                            verbosity=sc_verbosity,
                        )
                    break
                elif sc_iteration == sc_max_iterations - 1:
                    if sc_verbosity > 0:
                        _print_convergence_info(
                            particle_idx,
                            sc_iteration,
                            gamma_from_velocity,
                            gamma_from_energy,
                            gamma_abs_change,
                            gamma_rel_change,
                            converged=False,
                            max_iterations=sc_max_iterations,
                            verbosity=sc_verbosity,
                        )

    return result


__all__ = ["retarded_equations_of_motion"]
