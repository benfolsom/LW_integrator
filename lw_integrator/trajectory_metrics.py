"""Maintained helpers for normalized trajectory payloads and energy summaries."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from core.types import ParticleState

__all__ = [
    "compute_delta_energy_components",
    "compute_delta_energy_series",
    "extract_series",
    "normalize_state",
]


def normalize_state(state: ParticleState) -> ParticleState:
    """Normalize particle-state values to NumPy arrays.

    Metadata keys prefixed with ``_`` are preserved as-is because they can carry
    non-array control flags emitted by the integrator.
    """
    normalized: ParticleState = {}
    for key, value in state.items():
        if key.startswith("_"):
            normalized[key] = value
            continue

        if isinstance(value, np.ndarray):
            normalized[key] = value
        elif np.isscalar(value):
            normalized[key] = np.asarray([value], dtype=float)
        else:
            normalized[key] = np.asarray(value, dtype=float)
    return normalized


def extract_series(states: Iterable[ParticleState], field: str) -> np.ndarray:
    """Extract a scalar history for a named particle-state field."""
    return np.asarray([state[field][0] for state in states], dtype=float)


def compute_delta_energy_series(
    states: List[ParticleState],
    initial_state: ParticleState,
    rest_energy_mev: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute total energy change and z-position series."""
    gamma_series = extract_series(states, "gamma")
    initial_gamma = float(initial_state["gamma"][0])
    rest_energy_gev = rest_energy_mev * 1e-3
    delta_energy_gev = (gamma_series - initial_gamma) * rest_energy_gev
    z_series = extract_series(states, "z")
    return delta_energy_gev, z_series


def compute_delta_energy_components(
    states: List[ParticleState],
    initial_state: ParticleState,
    rest_energy_mev: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute total and longitudinal energy change series."""
    gamma_series = extract_series(states, "gamma")
    initial_gamma = float(initial_state["gamma"][0])
    rest_energy_gev = rest_energy_mev * 1e-3
    delta_energy_total = (gamma_series - initial_gamma) * rest_energy_gev

    bz_series = extract_series(states, "bz")
    initial_bz = float(initial_state["bz"][0])
    delta_energy_z = (
        gamma_series * bz_series - initial_gamma * initial_bz
    ) * rest_energy_gev

    z_series = extract_series(states, "z")
    return delta_energy_total, delta_energy_z, z_series
