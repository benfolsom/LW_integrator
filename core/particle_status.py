"""Utilities for tracking and managing particle status in multiparticle simulations.

This module provides helper functions for handling particle failures during integration,
including marking particles as "dead" when they experience numerical instabilities and
excluding them from final metric calculations.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from core.types import ParticleState, Trajectory


def mark_particle_dead(
    state: ParticleState,
    particle_idx: int,
    step: int,
    reason: str,
    gamma_value: Optional[float] = None,
    iteration: Optional[int] = None,
    details: Optional[Dict[str, object]] = None,
) -> None:
    """Mark a specific particle as dead due to numerical failure.

    This function modifies the state in-place to:
    1. Set the dead flag for the particle
    2. Record failure information
    3. Zero out the particle's charge so it no longer contributes to fields

    Parameters
    ----------
    state : ParticleState
        The particle state dictionary to modify
    particle_idx : int
        Index of the particle that failed
    step : int
        Integration step at which failure occurred
    reason : str
        Reason for failure (e.g., "gamma_blowup", "energy_jump")
    gamma_value : float, optional
        The gamma value at failure (if applicable)
    iteration : int, optional
        Self-consistency iteration at failure (if applicable)
    details : dict, optional
        Additional serializable metadata describing the failure or loss.
    """
    try:
        # Initialize dead particle tracking if not present
        if "_dead_particles" not in state:
            num_particles = len(state.get("gamma", []))
            state["_dead_particles"] = np.zeros(num_particles, dtype=bool)
        if "_particle_failure_info" not in state:
            state["_particle_failure_info"] = {}

        # Mark this particle as dead
        state["_dead_particles"][particle_idx] = True

        # Record failure information
        failure_info = {
            "step": step,
            "reason": reason,
        }
        if gamma_value is not None:
            failure_info["gamma_value"] = float(gamma_value)
        if iteration is not None:
            failure_info["iteration"] = iteration
        if details:
            failure_info.update(details)
        if "t" in state:
            failure_info["time_ns"] = float(state["t"][particle_idx])

        state["_particle_failure_info"][particle_idx] = failure_info

        # Zero out the particle's charge to neutralize it
        if "q" in state:
            state["q"][particle_idx] = 0.0
        elif "stripped_ions" in state:
            state["stripped_ions"][particle_idx] = 0.0
        else:
            print(
                f"      [WARNING] Particle {particle_idx}: Neither 'q' nor 'stripped_ions' found - cannot neutralize"
            )

    except Exception as e:
        print(
            f"      [ERROR] mark_particle_dead failed for particle {particle_idx}: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
        raise


def propagate_dead_particle_status(
    current_state: ParticleState, previous_state: ParticleState
) -> None:
    """Copy dead particle status from previous step to current step.

    Once a particle is marked dead, it remains dead for all subsequent steps.
    This function ensures consistency by copying the dead particle status forward.

    Parameters
    ----------
    current_state : ParticleState
        The current state to update
    previous_state : ParticleState
        The previous state containing dead particle information
    """
    if "_dead_particles" in previous_state:
        # Copy dead particle mask
        current_state["_dead_particles"] = previous_state["_dead_particles"].copy()
        # Copy failure info (dict needs deep copy of nested dicts)
        previous_failure_info = previous_state.get("_particle_failure_info", {})
        current_state["_particle_failure_info"] = {
            k: v.copy() for k, v in previous_failure_info.items()
        }

        # Ensure dead particles have zero charge
        dead_mask = current_state["_dead_particles"]
        if "q" in current_state:
            current_state["q"][dead_mask] = 0.0
        elif "stripped_ions" in current_state:
            current_state["stripped_ions"][dead_mask] = 0.0


def get_alive_particle_indices(state: ParticleState) -> np.ndarray:
    """Return indices of alive (non-dead) particles.

    Parameters
    ----------
    state : ParticleState
        Particle state to query

    Returns
    -------
    np.ndarray
        Integer array of indices corresponding to alive particles
    """
    dead_mask = state.get("_dead_particles")
    if dead_mask is None or not np.any(dead_mask):
        # No failures tracked, all particles are alive
        num_particles = len(state.get("gamma", []))
        return np.arange(num_particles)

    return np.where(~dead_mask)[0]


def count_alive_particles(state: ParticleState) -> int:
    """Count the number of alive (non-dead) particles.

    Parameters
    ----------
    state : ParticleState
        Particle state to query

    Returns
    -------
    int
        Number of alive particles
    """
    return len(get_alive_particle_indices(state))


def all_particles_dead(state: ParticleState) -> bool:
    """Check if all particles in a state are dead.

    Parameters
    ----------
    state : ParticleState
        Particle state to query

    Returns
    -------
    bool
        True if all particles are dead, False otherwise
    """
    dead_mask = state.get("_dead_particles")
    if dead_mask is None:
        return False  # No dead particles tracked

    return np.all(dead_mask)


def get_particle_failure_summary(trajectory: Trajectory) -> Dict[int, Dict]:
    """Extract failure information for all particles from trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        Complete trajectory to analyze

    Returns
    -------
    Dict[int, Dict]
        Dictionary mapping particle index to failure information:
        {
            particle_idx: {
                "step": int,
                "reason": str,
                "gamma_value": float,  # optional
                "iteration": int,      # optional
                "time_ns": float       # optional
            }
        }
    """
    if not trajectory:
        return {}

    final_state = trajectory[-1]
    return final_state.get("_particle_failure_info", {})


def compute_alive_particle_average(state: ParticleState, field: str) -> Optional[float]:
    """Compute average of a field excluding dead particles.

    Parameters
    ----------
    state : ParticleState
        Particle state containing the field
    field : str
        Name of the field to average (e.g., "gamma", "Pz")

    Returns
    -------
    float or None
        Mean value across alive particles, or None if all particles are dead
    """
    if field not in state:
        return None

    field_array = state[field]
    dead_mask = state.get("_dead_particles")

    if dead_mask is None or not np.any(dead_mask):
        # No dead particles, use all
        return float(np.mean(field_array))

    alive_mask = ~dead_mask
    if not np.any(alive_mask):
        # All particles are dead
        return None

    return float(np.mean(field_array[alive_mask]))


def get_alive_particle_values(state: ParticleState, field: str) -> Optional[np.ndarray]:
    """Extract values for a field from alive particles only.

    Parameters
    ----------
    state : ParticleState
        Particle state containing the field
    field : str
        Name of the field to extract

    Returns
    -------
    np.ndarray or None
        Array of values from alive particles, or None if all dead
    """
    if field not in state:
        return None

    field_array = state[field]
    alive_indices = get_alive_particle_indices(state)

    if len(alive_indices) == 0:
        return None

    return field_array[alive_indices]


def format_failure_summary(failure_info: Dict[int, Dict]) -> str:
    """Format particle failure information as a human-readable string.

    Parameters
    ----------
    failure_info : Dict[int, Dict]
        Failure information from get_particle_failure_summary()

    Returns
    -------
    str
        Formatted summary string
    """
    if not failure_info:
        return "No particle failures"

    lines = [f"Particle failures: {len(failure_info)} total"]
    for particle_idx, info in sorted(failure_info.items()):
        step = info.get("step", "?")
        reason = info.get("reason", "unknown")
        gamma = info.get("gamma_value")
        iteration = info.get("iteration")
        time_ns = info.get("time_ns")

        detail = f"  Particle {particle_idx}: {reason} at step {step}"
        if gamma is not None:
            detail += f", γ={gamma:.2e}"
        if iteration is not None:
            detail += f", iteration {iteration}"
        if time_ns is not None:
            detail += f", t={time_ns:.3e} ns"

        lines.append(detail)

    return "\n".join(lines)


def validate_particle_status_consistency(state: ParticleState) -> List[str]:
    """Validate that particle status metadata is consistent.

    Checks that:
    - Dead particles have zero charge
    - Dead particle mask and failure info are consistent
    - Array sizes match

    Parameters
    ----------
    state : ParticleState
        Particle state to validate

    Returns
    -------
    List[str]
        List of inconsistency messages (empty if all is well)
    """
    issues = []

    if "_dead_particles" not in state:
        # No dead particle tracking, nothing to validate
        return issues

    dead_mask = state["_dead_particles"]
    failure_info = state.get("_particle_failure_info", {})

    # Check array sizes
    num_particles = len(state.get("gamma", []))
    if len(dead_mask) != num_particles:
        issues.append(
            f"Dead mask size ({len(dead_mask)}) != particle count ({num_particles})"
        )

    # Check that dead particles have zero charge
    charge_field = (
        "q" if "q" in state else "stripped_ions" if "stripped_ions" in state else None
    )
    if charge_field:
        dead_indices = np.where(dead_mask)[0]
        for idx in dead_indices:
            if state[charge_field][idx] != 0.0:
                issues.append(
                    f"Dead particle {idx} has non-zero charge "
                    f"({state[charge_field][idx]})"
                )

    # Check that failure info matches dead mask
    for particle_idx in failure_info.keys():
        if not dead_mask[particle_idx]:
            issues.append(
                f"Particle {particle_idx} in failure_info but not marked dead"
            )

    for particle_idx in np.where(dead_mask)[0]:
        if particle_idx not in failure_info:
            issues.append(
                f"Particle {particle_idx} marked dead but no failure_info recorded"
            )

    return issues
