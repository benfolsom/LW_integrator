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
    transverse_offset_x: float = 0.0
    transverse_offset_y: float = 0.0
    transverse_spread: float = 0.0


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
    transverse_offset_x: float = 0.0,
    transverse_offset_y: float = 0.0,
    transverse_spread: float = 0.0,
) -> Tuple[ParticleState, float]:
    """Generate a particle state dictionary from kinetic energy inputs.

    Parameters
    ----------
    kinetic_energy_mev : float
        Kinetic energy in MeV
    mass_amu : float
        Particle mass in atomic mass units
    charge_sign : float
        Charge sign (+1 or -1)
    position_z : float, optional
        Starting z position in mm (default: 0.0)
    particle_count : int, optional
        Number of particles in bunch (default: 1)
    transverse_radius : float, optional
        DEPRECATED: Use transverse_offset_x/y instead. Single transverse offset in mm (default: 0.0)
    transverse_momentum : float, optional
        Transverse momentum spread (uniform ±transverse_momentum) in amu*mm/ns (default: 0.0)
    transverse_offset_x : float, optional
        Center x-position of bunch in mm (default: 0.0, on-axis)
    transverse_offset_y : float, optional
        Center y-position of bunch in mm (default: 0.0, on-axis)
    transverse_spread : float, optional
        Half-width of uniform transverse distribution in mm (default: 0.0)
        Particles distributed in [offset ± spread] for both x and y

    Returns
    -------
    state : ParticleState
        Dictionary with particle state arrays
    rest_energy_mev : float
        Rest energy in MeV

    Notes
    -----
    - If transverse_spread > 0, particles are uniformly distributed in a square:
      x ∈ [transverse_offset_x - transverse_spread, transverse_offset_x + transverse_spread]
      y ∈ [transverse_offset_y - transverse_spread, transverse_offset_y + transverse_spread]
    - If transverse_spread = 0, all particles are placed at (transverse_offset_x, transverse_offset_y)
    - transverse_radius is deprecated but maintained for backward compatibility
    """

    gamma = _compute_gamma(kinetic_energy_mev, mass_amu)
    beta = math.sqrt(1.0 - 1.0 / (gamma**2)) if gamma > 1.0 else 0.0
    particle_mass = mass_amu
    macro_charge = charge_sign * ELEMENTARY_CHARGE_GU
    char_time = 2.0 / 3.0 * macro_charge**2 / (particle_mass * C_MMNS**3)

    count = particle_count
    zeros = np.zeros(count, dtype=float)

    # Handle transverse positions with offset and spread
    if transverse_spread > 0.0:
        # Distribute particles uniformly in a square around the offset
        x = np.random.uniform(
            transverse_offset_x - transverse_spread,
            transverse_offset_x + transverse_spread,
            count,
        )
        y = np.random.uniform(
            transverse_offset_y - transverse_spread,
            transverse_offset_y + transverse_spread,
            count,
        )
    else:
        # All particles at the offset position (or legacy transverse_radius if specified)
        if transverse_radius != 0.0:
            # Backward compatibility: use old transverse_radius parameter
            x = np.full(count, transverse_radius, dtype=float)
            y = np.full(count, -transverse_radius, dtype=float)
        else:
            x = np.full(count, transverse_offset_x, dtype=float)
            y = np.full(count, transverse_offset_y, dtype=float)

    # Handle transverse momentum with spread
    if transverse_momentum > 0.0:
        Px = (
            np.random.uniform(-transverse_momentum, transverse_momentum, count)
            * particle_mass
        )
        Py = (
            np.random.uniform(-transverse_momentum, transverse_momentum, count)
            * particle_mass
        )
    else:
        Px = zeros.copy()
        Py = zeros.copy()

    # Longitudinal momentum
    Pz = np.full(count, gamma * particle_mass * C_MMNS * beta, dtype=float)

    # Total momentum and gamma (recompute for accuracy when Px, Py non-zero)
    P_total = np.sqrt(Px**2 + Py**2 + Pz**2)
    Pt = np.sqrt(P_total**2 + (particle_mass * C_MMNS) ** 2)
    gamma_arr = Pt / (particle_mass * C_MMNS)

    # Velocity components
    bx = Px / (gamma_arr * particle_mass * C_MMNS)
    by = Py / (gamma_arr * particle_mass * C_MMNS)
    bz = Pz / (gamma_arr * particle_mass * C_MMNS)

    state: ParticleState = {
        "x": x,
        "y": y,
        "z": np.full(count, position_z, dtype=float),
        "t": zeros.copy(),
        "Px": Px,
        "Py": Py,
        "Pz": Pz,
        "Pt": Pt,
        "gamma": gamma_arr,
        "bx": bx,
        "by": by,
        "bz": bz,
        "bdotx": zeros.copy(),
        "bdoty": zeros.copy(),
        "bdotz": zeros.copy(),
        "q": np.full(count, macro_charge, dtype=float),
        "m": np.full(count, particle_mass, dtype=float),
        "char_time": np.full(count, char_time, dtype=float),
    }

    rest_energy_mev = mass_amu * AMU_TO_MEV
    return state, rest_energy_mev


def create_bunch_from_params(
    *,
    starting_distance: float,
    transv_mom: float,
    starting_Pz: float,
    stripped_ions: float,
    m_particle: float,
    transv_dist: float = 0.0,
    transv_offset_x: float = 0.0,
    transv_offset_y: float = 0.0,
    pcount: int = 1,
    charge_sign: float = 1.0,
    seed: int | None = None,
) -> Tuple[ParticleState, float]:
    """Generate particle state from legacy-style parameters.

    This function provides a non-legacy alternative to legacy.bunch_inits.init_bunch
    with support for transverse offset.

    Parameters
    ----------
    starting_distance : float
        Starting z-position in mm
    transv_mom : float
        Transverse momentum spread (uniform ±transv_mom) in amu*mm/ns
    starting_Pz : float
        Initial longitudinal momentum per unit mass (specific momentum) in mm/ns
    stripped_ions : float
        Number of elementary charges
    m_particle : float
        Particle mass in amu
    transv_dist : float, optional
        Half-width of transverse distribution in mm (default: 0.0)
    transv_offset_x : float, optional
        Center x-position of bunch in mm (default: 0.0)
    transv_offset_y : float, optional
        Center y-position of bunch in mm (default: 0.0)
    pcount : int, optional
        Number of particles (default: 1)
    charge_sign : float, optional
        Charge sign (+1 or -1, default: 1.0)
    seed : int, optional
        Random seed for reproducibility (default: None)

    Returns
    -------
    state : ParticleState
        Particle state dictionary
    rest_energy_mev : float
        Rest energy in MeV
    """
    if seed is not None:
        np.random.seed(seed)

    macro_charge = charge_sign * stripped_ions * ELEMENTARY_CHARGE_GU
    char_time = 2.0 / 3.0 * macro_charge**2 / (m_particle * C_MMNS**3)

    # Generate transverse momentum with spread
    if transv_mom > 0.0:
        Px = np.random.uniform(-transv_mom, transv_mom, pcount) * m_particle
        Py = np.random.uniform(-transv_mom, transv_mom, pcount) * m_particle
    else:
        Px = np.zeros(pcount, dtype=float)
        Py = np.zeros(pcount, dtype=float)

    # Longitudinal momentum with small spread
    Pz = np.random.uniform(starting_Pz, starting_Pz + 0.1, pcount) * m_particle

    # Total momentum and energy
    Pt = np.sqrt(Px**2 + Py**2 + Pz**2 + (m_particle * C_MMNS) ** 2)
    gamma = Pt / (m_particle * C_MMNS)

    # Velocity components
    bx = Px / (gamma * m_particle * C_MMNS)
    by = Py / (gamma * m_particle * C_MMNS)
    bz = Pz / (gamma * m_particle * C_MMNS)

    # Generate transverse positions with offset and spread
    if transv_dist > 0.0:
        x = np.random.uniform(
            transv_offset_x - transv_dist, transv_offset_x + transv_dist, pcount
        )
        y = np.random.uniform(
            transv_offset_y - transv_dist, transv_offset_y + transv_dist, pcount
        )
    else:
        x = np.full(pcount, transv_offset_x, dtype=float)
        y = np.full(pcount, transv_offset_y, dtype=float)

    # Longitudinal position with small spread
    z = np.random.uniform(starting_distance - 1e-6, starting_distance + 1e-6, pcount)
    t = np.zeros(pcount, dtype=float)

    state: ParticleState = {
        "x": x,
        "y": y,
        "z": z,
        "t": t,
        "Px": Px,
        "Py": Py,
        "Pz": Pz,
        "Pt": Pt,
        "gamma": gamma,
        "bx": bx,
        "by": by,
        "bz": bz,
        "bdotx": np.zeros(pcount, dtype=float),
        "bdoty": np.zeros(pcount, dtype=float),
        "bdotz": np.zeros(pcount, dtype=float),
        "q": np.full(pcount, macro_charge, dtype=float),
        "m": np.full(pcount, m_particle, dtype=float),
        "char_time": np.full(pcount, char_time, dtype=float),
    }

    rest_energy_mev = m_particle * AMU_TO_MEV
    return state, rest_energy_mev
