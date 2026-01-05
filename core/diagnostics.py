"""Diagnostic utilities for analyzing integration results and detecting anomalies.

This module provides tools for post-simulation analysis and runtime monitoring
of physical quantities like energy, momentum, and charge conservation.

IMPORTANT: Energy is calculated from gamma and rest mass (E = γmc²), NOT from
conjugate momenta Pt which is used internally in the covariant formulation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import C_MMNS
from .types import ParticleState, Trajectory


def compute_total_energy(state: ParticleState) -> float:
    """Compute total energy of a particle state in MeV.

    Uses the relativistic energy formula E = γmc² where gamma is the
    Lorentz factor and m is the rest mass.

    Parameters
    ----------
    state : ParticleState
        Particle state dictionary containing 'gamma' and 'm' (rest mass in amu).

    Returns
    -------
    float
        Total energy in MeV.

    Notes
    -----
    This correctly computes energy from the Lorentz factor, not from the
    conjugate momentum Pt used internally in the equations of motion.
    """
    if "gamma" not in state or "m" not in state:
        raise ValueError(
            "State must contain 'gamma' and 'm' fields for energy calculation"
        )

    gamma = np.asarray(state["gamma"])
    mass = np.asarray(state["m"])

    # E = γmc² where c is in mm/ns and mass in amu
    # This gives energy in amu·(mm/ns)² = amu·c² units
    # Conversion factor from amu·c² to MeV is built into C_MMNS
    total_energy = float(np.sum(gamma * mass * C_MMNS * C_MMNS))

    return total_energy


def compute_kinetic_energy(state: ParticleState) -> float:
    """Compute total kinetic energy of a particle state in MeV.

    Uses KE = (γ - 1)mc²

    Parameters
    ----------
    state : ParticleState
        Particle state dictionary containing 'gamma' and 'm'.

    Returns
    -------
    float
        Total kinetic energy in MeV.
    """
    if "gamma" not in state or "m" not in state:
        raise ValueError("State must contain 'gamma' and 'm' fields")

    gamma = np.asarray(state["gamma"])
    mass = np.asarray(state["m"])

    # KE = (γ - 1)mc²
    kinetic_energy = float(np.sum((gamma - 1.0) * mass * C_MMNS * C_MMNS))

    return kinetic_energy


def compute_total_momentum(state: ParticleState) -> Tuple[float, float, float]:
    """Compute total momentum components from beta and gamma.

    Uses p = γmβc

    Parameters
    ----------
    state : ParticleState
        Particle state dictionary containing 'bx', 'by', 'bz', 'gamma', 'm'.

    Returns
    -------
    tuple[float, float, float]
        Total momentum components (px, py, pz) in amu·mm/ns.

    Notes
    -----
    This computes physical momentum, not the conjugate momenta Px, Py, Pz
    used in the covariant formulation.
    """
    if not all(k in state for k in ["bx", "by", "bz", "gamma", "m"]):
        raise ValueError("State must contain 'bx', 'by', 'bz', 'gamma', 'm' fields")

    bx = np.asarray(state["bx"])
    by = np.asarray(state["by"])
    bz = np.asarray(state["bz"])
    gamma = np.asarray(state["gamma"])
    mass = np.asarray(state["m"])

    # p = γmβc
    px = float(np.sum(gamma * mass * bx * C_MMNS))
    py = float(np.sum(gamma * mass * by * C_MMNS))
    pz = float(np.sum(gamma * mass * bz * C_MMNS))

    return px, py, pz


def analyze_trajectory_energy(
    trajectory: Trajectory,
    check_interval: int = 1,
    relative_threshold: float = 1.0,
) -> Dict[str, any]:
    """Analyze energy evolution throughout a trajectory.

    Computes energy from gamma and mass at each step and identifies sudden
    jumps that exceed the specified relative threshold.

    Parameters
    ----------
    trajectory : Trajectory
        List of particle states from an integration run.
    check_interval : int, optional
        Check energy every N steps. Default is 1 (check every step).
    relative_threshold : float, optional
        Relative energy change threshold for detecting jumps (e.g., 1.0 = 100%).
        Default is 1.0.

    Returns
    -------
    dict
        Analysis results containing:
        - 'energies': List of total energies at each checked step
        - 'kinetic_energies': List of kinetic energies at each checked step
        - 'step_indices': List of step indices where energy was computed
        - 'relative_changes': List of relative energy changes between steps
        - 'jumps_detected': List of (step, relative_change) tuples for jumps
        - 'max_relative_change': Maximum relative energy change observed
        - 'initial_energy': Energy at first step
        - 'final_energy': Energy at last step
        - 'conservation_error': Relative difference between final and initial
    """
    if len(trajectory) == 0:
        return {
            "energies": [],
            "kinetic_energies": [],
            "step_indices": [],
            "relative_changes": [],
            "jumps_detected": [],
            "max_relative_change": 0.0,
            "initial_energy": 0.0,
            "final_energy": 0.0,
            "conservation_error": 0.0,
        }

    energies = []
    kinetic_energies = []
    step_indices = []
    relative_changes = []
    jumps_detected = []

    for i in range(0, len(trajectory), check_interval):
        try:
            energy = compute_total_energy(trajectory[i])
            ke = compute_kinetic_energy(trajectory[i])
            energies.append(energy)
            kinetic_energies.append(ke)
            step_indices.append(i)

            if len(energies) > 1:
                prev_energy = energies[-2]
                if prev_energy > 0:
                    rel_change = abs(energy - prev_energy) / prev_energy
                    relative_changes.append(rel_change)

                    if rel_change > relative_threshold:
                        jumps_detected.append((i, rel_change))
                else:
                    relative_changes.append(0.0)
        except (ValueError, KeyError):
            # Skip steps that don't have required fields
            continue

    initial_energy = energies[0] if energies else 0.0
    final_energy = energies[-1] if energies else 0.0
    conservation_error = (
        abs(final_energy - initial_energy) / initial_energy
        if initial_energy > 0
        else 0.0
    )
    max_relative_change = max(relative_changes) if relative_changes else 0.0

    return {
        "energies": energies,
        "kinetic_energies": kinetic_energies,
        "step_indices": step_indices,
        "relative_changes": relative_changes,
        "jumps_detected": jumps_detected,
        "max_relative_change": max_relative_change,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "conservation_error": conservation_error,
    }


def print_energy_analysis(
    analysis: Dict[str, any],
    threshold: Optional[float] = None,
) -> None:
    """Print formatted energy analysis results.

    Parameters
    ----------
    analysis : dict
        Results from :func:`analyze_trajectory_energy`.
    threshold : float, optional
        Highlight jumps exceeding this threshold. If None, uses the threshold
        from the analysis.
    """
    print("\n" + "=" * 70)
    print("ENERGY ANALYSIS (using E = γmc²)")
    print("=" * 70)
    print(f"Initial energy:       {analysis['initial_energy']:.6e} MeV")
    print(f"Final energy:         {analysis['final_energy']:.6e} MeV")
    print(
        f"Conservation error:   {analysis['conservation_error']:.6e} ({analysis['conservation_error'] * 100:.4f}%)"
    )
    print(
        f"Max relative change:  {analysis['max_relative_change']:.6e} ({analysis['max_relative_change'] * 100:.4f}%)"
    )

    if analysis["jumps_detected"]:
        print(f"\n⚠️  {len(analysis['jumps_detected'])} energy jump(s) detected:")
        for step, rel_change in analysis["jumps_detected"]:
            print(f"   Step {step}: ΔE/E = {rel_change:.6e} ({rel_change * 100:.2f}%)")
    else:
        print("\n✓ No significant energy jumps detected")
    print("=" * 70 + "\n")


def check_superluminal_velocities(trajectory: Trajectory) -> Dict[str, any]:
    """Check for beta >= 1.0 (superluminal velocities) in trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        List of particle states to check.

    Returns
    -------
    dict
        Results containing:
        - 'violations_found': Boolean indicating if any violations were found
        - 'violation_steps': List of step indices where beta >= 1.0
        - 'max_beta': Maximum beta value observed
        - 'max_beta_step': Step index where max beta occurred
    """
    violations = []
    max_beta = 0.0
    max_beta_step = 0

    for i, state in enumerate(trajectory):
        if "bx" in state and "by" in state and "bz" in state:
            beta_squared = state["bx"] ** 2 + state["by"] ** 2 + state["bz"] ** 2
            beta = np.sqrt(beta_squared)
            beta_max_this_step = float(np.max(beta))

            if beta_max_this_step > max_beta:
                max_beta = beta_max_this_step
                max_beta_step = i

            if beta_max_this_step >= 1.0:
                violations.append(i)

    return {
        "violations_found": len(violations) > 0,
        "violation_steps": violations,
        "max_beta": max_beta,
        "max_beta_step": max_beta_step,
    }


def check_gamma_consistency(
    trajectory: Trajectory, tolerance: float = 1e-6
) -> Dict[str, any]:
    """Check that gamma is consistent with beta throughout the trajectory.

    Verifies that γ = 1/√(1 - β²) holds to within tolerance.

    Parameters
    ----------
    trajectory : Trajectory
        List of particle states to check.
    tolerance : float, optional
        Relative tolerance for gamma consistency. Default is 1e-6.

    Returns
    -------
    dict
        Results containing:
        - 'consistent': Boolean indicating if all checks passed
        - 'inconsistent_steps': List of step indices with inconsistencies
        - 'max_relative_error': Maximum relative error observed
        - 'max_error_step': Step where maximum error occurred
    """
    inconsistent_steps = []
    max_rel_error = 0.0
    max_error_step = 0

    for i, state in enumerate(trajectory):
        if not all(k in state for k in ["bx", "by", "bz", "gamma"]):
            continue

        bx = np.asarray(state["bx"])
        by = np.asarray(state["by"])
        bz = np.asarray(state["bz"])
        gamma_stored = np.asarray(state["gamma"])

        beta_squared = bx**2 + by**2 + bz**2
        # Clamp to prevent numerical issues
        beta_squared = np.minimum(beta_squared, 0.9999999999)

        gamma_calculated = 1.0 / np.sqrt(1.0 - beta_squared)

        # Check relative error
        rel_error = np.abs(gamma_calculated - gamma_stored) / gamma_stored
        max_rel_this_step = float(np.max(rel_error))

        if max_rel_this_step > max_rel_error:
            max_rel_error = max_rel_this_step
            max_error_step = i

        if max_rel_this_step > tolerance:
            inconsistent_steps.append(i)

    return {
        "consistent": len(inconsistent_steps) == 0,
        "inconsistent_steps": inconsistent_steps,
        "max_relative_error": max_rel_error,
        "max_error_step": max_error_step,
    }


def validate_trajectory(
    trajectory: Trajectory,
    energy_threshold: float = 1.0,
    conservation_tolerance: float = 0.01,
    verbose: bool = True,
) -> Dict[str, any]:
    """Perform comprehensive validation of a trajectory.

    Checks energy conservation (using E = γmc²), detects jumps, verifies
    gamma-beta consistency, and looks for superluminal velocities.

    Parameters
    ----------
    trajectory : Trajectory
        List of particle states to validate.
    energy_threshold : float, optional
        Relative energy change threshold for jump detection. Default is 1.0 (100%).
    conservation_tolerance : float, optional
        Relative tolerance for energy conservation. Default is 0.01 (1%).
    verbose : bool, optional
        If True, print detailed analysis. Default is True.

    Returns
    -------
    dict
        Validation results containing:
        - 'passed': Boolean indicating overall pass/fail
        - 'energy_analysis': Results from energy analysis
        - 'superluminal_check': Results from velocity check
        - 'gamma_consistency_check': Results from gamma-beta consistency check
        - 'warnings': List of warning messages
        - 'errors': List of error messages
    """
    warnings = []
    errors = []

    # Energy analysis
    energy_analysis = analyze_trajectory_energy(
        trajectory,
        check_interval=1,
        relative_threshold=energy_threshold,
    )

    if energy_analysis["conservation_error"] > conservation_tolerance:
        errors.append(
            f"Energy conservation violated: {energy_analysis['conservation_error'] * 100:.4f}% "
            f"(tolerance: {conservation_tolerance * 100:.4f}%)"
        )

    if energy_analysis["jumps_detected"]:
        warnings.append(
            f"{len(energy_analysis['jumps_detected'])} energy jumps detected "
            f"(threshold: {energy_threshold * 100:.1f}%)"
        )

    # Superluminal velocity check
    superluminal_check = check_superluminal_velocities(trajectory)

    if superluminal_check["violations_found"]:
        errors.append(
            f"Superluminal velocities detected at {len(superluminal_check['violation_steps'])} steps"
        )

    if superluminal_check["max_beta"] > 0.9999:
        warnings.append(
            f"Very high beta detected: {superluminal_check['max_beta']:.10f} "
            f"at step {superluminal_check['max_beta_step']}"
        )

    # Gamma consistency check
    gamma_consistency_check = check_gamma_consistency(trajectory, tolerance=1e-6)

    if not gamma_consistency_check["consistent"]:
        warnings.append(
            f"Gamma-beta inconsistency detected at {len(gamma_consistency_check['inconsistent_steps'])} steps "
            f"(max error: {gamma_consistency_check['max_relative_error']:.6e})"
        )

    passed = len(errors) == 0

    if verbose:
        print("\n" + "=" * 70)
        print("TRAJECTORY VALIDATION")
        print("=" * 70)
        print(f"Steps analyzed: {len(trajectory)}")
        print(f"Status: {'✓ PASSED' if passed else '✗ FAILED'}")

        if warnings:
            print(f"\n⚠️  {len(warnings)} warning(s):")
            for w in warnings:
                print(f"   {w}")

        if errors:
            print(f"\n✗ {len(errors)} error(s):")
            for e in errors:
                print(f"   {e}")

        if not warnings and not errors:
            print("\n✓ No issues detected")

        print("=" * 70 + "\n")

        if not passed:
            print_energy_analysis(energy_analysis)

    return {
        "passed": passed,
        "energy_analysis": energy_analysis,
        "superluminal_check": superluminal_check,
        "gamma_consistency_check": gamma_consistency_check,
        "warnings": warnings,
        "errors": errors,
    }


def find_radiation_reaction_activations(
    trajectory: Trajectory,
) -> List[Tuple[int, int]]:
    """Identify steps where radiation reaction force was likely active.

    Looks for sudden changes in acceleration (bdot) that indicate the
    radiation reaction force was triggered.

    Parameters
    ----------
    trajectory : Trajectory
        List of particle states to analyze.

    Returns
    -------
    list[tuple[int, int]]
        List of (step_index, particle_index) tuples where radiation
        reaction likely activated.
    """
    activations = []

    for i in range(1, len(trajectory)):
        if "bdotz" not in trajectory[i] or "bdotz" not in trajectory[i - 1]:
            continue

        bdotz_curr = np.asarray(trajectory[i]["bdotz"])
        bdotz_prev = np.asarray(trajectory[i - 1]["bdotz"])

        # Look for sudden changes in acceleration
        delta_bdotz = np.abs(bdotz_curr - bdotz_prev)

        # Threshold based on magnitude (radiation reaction typically adds
        # significant damping when active)
        threshold = 1e-3  # amu·mm/ns³ characteristic scale

        activated = np.where(delta_bdotz > threshold)[0]
        for particle_idx in activated:
            activations.append((i, int(particle_idx)))

    return activations


__all__ = [
    "compute_total_energy",
    "compute_kinetic_energy",
    "compute_total_momentum",
    "analyze_trajectory_energy",
    "print_energy_analysis",
    "check_superluminal_velocities",
    "check_gamma_consistency",
    "validate_trajectory",
    "find_radiation_reaction_activations",
]
