"""Metrics for optimization of LW integrator simulations.

This module provides functions to compute various metrics from trajectory data
that are useful for optimization, including:
- Maximum energy gain
- Energy gain at specific positions
- Transverse deflection detection
- Energy efficiency metrics
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from core.types import ParticleState


def compute_max_energy_gain(
    trajectory: List[ParticleState],
    initial_gamma: float,
    rest_energy_mev: float,
) -> float:
    """Compute maximum energy gain in GeV along trajectory.

    Parameters
    ----------
    trajectory : List[ParticleState]
        Particle trajectory states
    initial_gamma : float
        Initial Lorentz factor
    rest_energy_mev : float
        Rest energy in MeV

    Returns
    -------
    float
        Maximum energy gain in GeV
    """
    rest_energy_gev = rest_energy_mev * 1e-3
    max_gain = 0.0

    for state in trajectory:
        gamma = np.mean(state["gamma"])
        delta_e = (gamma - initial_gamma) * rest_energy_gev
        if delta_e > max_gain:
            max_gain = delta_e

    return max_gain


def compute_delta_energy_components(trajectory: ParticleState) -> Tuple[float, float]:
    """Compute total and longitudinal energy changes from trajectory.

    Parameters
    ----------
    trajectory : ParticleState
        Particle trajectory state (structured array or dict with arrays)

    Returns
    -------
    Tuple[float, float]
        (delta_E_total_GeV, delta_E_z_GeV)
        - delta_E_total: Total energy change from Δγ
        - delta_E_z: Longitudinal energy change from γ·βz
    """
    # Electron rest mass in GeV
    m_e = 0.000510999  # GeV/c²

    gamma = np.asarray(trajectory["gamma"])
    pz = np.asarray(trajectory["Pz"])

    # Total energy change from gamma
    delta_E_total = (gamma[-1] - gamma[0]) * m_e

    # Longitudinal energy (E_z = γ·m·c²·βz)
    # β_z = P_z / (γ·m·c)
    # For simplicity, we compute change in γ·βz
    # Note: In natural units where c=1, P_z ≈ γ·m·βz
    # So E_z ≈ P_z (in momentum units)

    # Convert momentum to energy-like quantity
    # Assume Pz is in MeV/c, convert to GeV
    pz_gev = pz * 1e-3
    delta_E_z = pz_gev[-1] - pz_gev[0]

    return delta_E_total, delta_E_z


def compute_energy_gain_near_aperture(
    trajectory: List[ParticleState],
    initial_gamma: float,
    rest_energy_mev: float,
    aperture_z: float,
    search_range_mm: float = 50.0,
) -> Tuple[float, float, int]:
    """Compute maximum energy gain near aperture position.

    Parameters
    ----------
    trajectory : List[ParticleState]
        Particle trajectory states
    initial_gamma : float
        Initial Lorentz factor
    rest_energy_mev : float
        Rest energy in MeV
    aperture_z : float
        Z position of aperture in mm
    search_range_mm : float, optional
        Distance range to search on either side of aperture (default: 50mm)

    Returns
    -------
    Tuple[float, float, int]
        (max_energy_gain_GeV, z_position_at_max, step_index)
    """
    rest_energy_gev = rest_energy_mev * 1e-3
    max_gain = -np.inf
    max_z = None
    max_step = -1

    z_min = aperture_z - search_range_mm
    z_max = aperture_z + search_range_mm

    for i, state in enumerate(trajectory):
        z_pos = np.mean(state["z"])
        if z_min <= z_pos <= z_max:
            gamma = np.mean(state["gamma"])
            delta_e = (gamma - initial_gamma) * rest_energy_gev
            if delta_e > max_gain:
                max_gain = delta_e
                max_z = z_pos
                max_step = i

    if max_z is None:
        # No data in range, return zero
        return 0.0, aperture_z, -1

    return max_gain, max_z, max_step


def compute_relative_energy_gain(
    trajectory: List[ParticleState],
    initial_gamma: float,
) -> float:
    """Compute maximum relative energy gain (ΔE/E₀).

    Parameters
    ----------
    trajectory : List[ParticleState]
        Particle trajectory states
    initial_gamma : float
        Initial Lorentz factor

    Returns
    -------
    float
        Maximum relative energy gain (dimensionless)
    """
    max_relative_gain = 0.0

    for state in trajectory:
        gamma = np.mean(state["gamma"])
        relative_gain = (gamma - initial_gamma) / initial_gamma
        if relative_gain > max_relative_gain:
            max_relative_gain = relative_gain

    return max_relative_gain


def detect_transverse_deflection(
    trajectory: List[ParticleState],
    energy_jump_threshold: float = 0.1,
    energy_dip_threshold: float = 0.05,
    initial_gamma: Optional[float] = None,
) -> List[Tuple[int, str, float]]:
    """Detect transverse deflections by finding energy jumps followed by dips.

    Energy jumps followed by large energy dips often indicate strong transverse
    deflections rather than true acceleration.

    Parameters
    ----------
    trajectory : List[ParticleState]
        Particle trajectory states
    energy_jump_threshold : float, optional
        Relative energy jump to detect (default: 0.1 = 10%)
    energy_dip_threshold : float, optional
        Relative energy dip after jump (default: 0.05 = 5%)
    initial_gamma : float, optional
        Initial gamma for reference. If None, uses first trajectory point.

    Returns
    -------
    List[Tuple[int, str, float]]
        List of (step_index, event_type, magnitude) tuples
        event_type can be "jump", "dip", or "deflection"
    """
    if len(trajectory) < 3:
        return []

    if initial_gamma is None:
        initial_gamma = np.mean(trajectory[0]["gamma"])

    events = []
    gamma_prev = np.mean(trajectory[0]["gamma"])

    for i in range(1, len(trajectory)):
        gamma_curr = np.mean(trajectory[i]["gamma"])

        # Check for jump
        relative_change = (gamma_curr - gamma_prev) / initial_gamma
        if relative_change > energy_jump_threshold:
            events.append((i, "jump", relative_change))

            # Look ahead for dip
            if i + 1 < len(trajectory):
                gamma_next = np.mean(trajectory[i + 1]["gamma"])
                dip = (gamma_curr - gamma_next) / initial_gamma
                if dip > energy_dip_threshold:
                    events.append((i + 1, "dip", dip))
                    events.append((i, "deflection", relative_change))

        gamma_prev = gamma_curr

    return events


def compute_trajectory_metrics(
    trajectory: List[ParticleState],
    initial_state: ParticleState,
    rest_energy_mev: float,
    aperture_z: Optional[float] = None,
) -> Dict[str, float]:
    """Compute comprehensive metrics for a trajectory.

    Parameters
    ----------
    trajectory : List[ParticleState]
        Particle trajectory states
    initial_state : ParticleState
        Initial particle state
    rest_energy_mev : float
        Rest energy in MeV
    aperture_z : float, optional
        Z position of aperture. If provided, computes near-aperture metrics.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'max_energy_gain_gev': Maximum energy gain in GeV
        - 'max_relative_gain': Maximum ΔE/E₀
        - 'final_energy_gain_gev': Energy gain at final step
        - 'near_aperture_max_gev': Max energy near aperture (if aperture_z given)
        - 'near_aperture_z_mm': Z position of near-aperture max
        - 'max_transverse_displacement_mm': Maximum transverse displacement
        - 'num_deflection_events': Number of detected deflection events
    """
    initial_gamma = float(initial_state["gamma"][0])
    rest_energy_gev = rest_energy_mev * 1e-3

    metrics = {}

    # Max energy gain
    metrics["max_energy_gain_gev"] = compute_max_energy_gain(
        trajectory, initial_gamma, rest_energy_mev
    )

    # Relative gain
    metrics["max_relative_gain"] = compute_relative_energy_gain(
        trajectory, initial_gamma
    )

    # Final energy gain
    if len(trajectory) > 0:
        final_gamma = np.mean(trajectory[-1]["gamma"])
        metrics["final_energy_gain_gev"] = (
            final_gamma - initial_gamma
        ) * rest_energy_gev
    else:
        metrics["final_energy_gain_gev"] = 0.0

    # Near-aperture metrics
    if aperture_z is not None:
        max_gain, max_z, _ = compute_energy_gain_near_aperture(
            trajectory, initial_gamma, rest_energy_mev, aperture_z
        )
        metrics["near_aperture_max_gev"] = max_gain
        metrics["near_aperture_z_mm"] = max_z
    else:
        metrics["near_aperture_max_gev"] = 0.0
        metrics["near_aperture_z_mm"] = 0.0

    # Transverse displacement
    initial_x = float(initial_state["x"][0])
    initial_y = float(initial_state["y"][0])
    max_displacement = 0.0

    for state in trajectory:
        x = np.mean(state["x"])
        y = np.mean(state["y"])
        displacement = np.sqrt((x - initial_x) ** 2 + (y - initial_y) ** 2)
        if displacement > max_displacement:
            max_displacement = displacement

    metrics["max_transverse_displacement_mm"] = max_displacement

    # Deflection detection
    deflections = detect_transverse_deflection(trajectory, initial_gamma=initial_gamma)
    deflection_count = sum(
        1 for _, event_type, _ in deflections if event_type == "deflection"
    )
    metrics["num_deflection_events"] = deflection_count

    return metrics


def compute_energy_at_position(
    trajectory: List[ParticleState],
    target_z: float,
    initial_gamma: float,
    rest_energy_mev: float,
    tolerance_mm: float = 1.0,
) -> Optional[float]:
    """Compute energy gain at a specific z position.

    Parameters
    ----------
    trajectory : List[ParticleState]
        Particle trajectory states
    target_z : float
        Target z position in mm
    initial_gamma : float
        Initial Lorentz factor
    rest_energy_mev : float
        Rest energy in MeV
    tolerance_mm : float, optional
        Position tolerance in mm (default: 1.0)

    Returns
    -------
    Optional[float]
        Energy gain in GeV at target position, or None if position not found
    """
    rest_energy_gev = rest_energy_mev * 1e-3

    for state in trajectory:
        z_pos = np.mean(state["z"])
        if abs(z_pos - target_z) <= tolerance_mm:
            gamma = np.mean(state["gamma"])
            return (gamma - initial_gamma) * rest_energy_gev

    return None


def compute_percent_energy_gain(
    trajectory: List[ParticleState],
    initial_gamma: float,
) -> float:
    """Compute maximum percent energy gain.

    Parameters
    ----------
    trajectory : List[ParticleState]
        Particle trajectory states
    initial_gamma : float
        Initial Lorentz factor

    Returns
    -------
    float
        Maximum percent energy gain (e.g., 15.5 for 15.5%)
    """
    return compute_relative_energy_gain(trajectory, initial_gamma) * 100.0
