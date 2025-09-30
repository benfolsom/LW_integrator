"""Bunch initialization helpers for the core Liénard–Wiechert integrator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

C_MMNS = 299.792458  # Speed of light in mm/ns (matches legacy constant)
AMU_TO_MEV = 931.49410242  # Atomic mass unit → MeV/c^2 conversion
ELEMENTARY_CHARGE_GU = 4.803204712570263e-10  # Elementary charge (Gaussian units)

ParticleState = Dict[str, np.ndarray]


@dataclass
class BunchRequest:
    """Input parameters for :func:`create_bunch_from_energy`."""

    kinetic_energy_mev: float
    mass_amu: float
    charge_sign: float
    position_z: float = 0.0
    particle_count: int = 1
    transverse_radius: float = 0.0
    transverse_momentum: float = 0.0


def _compute_gamma(kinetic_energy_mev: float, mass_amu: float) -> float:
    rest_energy = mass_amu * AMU_TO_MEV
    return kinetic_energy_mev / rest_energy + 1.0


def create_bunch_from_energy(
    *,
    kinetic_energy_mev: float,
    mass_amu: float,
    charge_sign: float,
    position_z: float = 0.0,
    particle_count: int = 1,
    transverse_radius: float = 0.0,
    transverse_momentum: float = 0.0,
) -> Tuple[ParticleState, float]:
    """Generate a particle state dictionary from kinetic energy inputs.

    Returns a dictionary matching the core integrator expectations plus the
    total rest energy in MeV (for compatibility with legacy helpers).
    """

    gamma = _compute_gamma(kinetic_energy_mev, mass_amu)
    beta = math.sqrt(1.0 - 1.0 / (gamma**2)) if gamma > 1.0 else 0.0
    particle_mass = mass_amu
    macro_charge = charge_sign * ELEMENTARY_CHARGE_GU
    char_time = 2.0 / 3.0 * macro_charge**2 / (particle_mass * C_MMNS**3)

    count = particle_count
    zeros = np.zeros(count, dtype=float)

    Px = np.full(count, transverse_momentum * particle_mass, dtype=float)
    Py = zeros.copy()
    Pz = np.full(count, gamma * particle_mass * C_MMNS * beta, dtype=float)
    Pt = np.full(count, gamma * particle_mass * C_MMNS, dtype=float)

    state: ParticleState = {
        "x": np.full(count, transverse_radius, dtype=float),
        "y": np.full(count, -transverse_radius, dtype=float),
        "z": np.full(count, position_z, dtype=float),
        "t": zeros.copy(),
        "Px": Px,
        "Py": Py,
        "Pz": Pz,
        "Pt": Pt,
        "gamma": np.full(count, gamma, dtype=float),
        "bx": Px / (particle_mass * C_MMNS * np.maximum(gamma, 1e-12)),
        "by": Py / (particle_mass * C_MMNS * np.maximum(gamma, 1e-12)),
        "bz": np.full(count, beta, dtype=float),
        "bdotx": zeros.copy(),
        "bdoty": zeros.copy(),
        "bdotz": zeros.copy(),
        "q": np.full(count, macro_charge, dtype=float),
        "m": np.full(count, particle_mass, dtype=float),
        "char_time": np.full(count, char_time, dtype=float),
    }

    rest_energy_mev = mass_amu * AMU_TO_MEV
    return state, rest_energy_mev