"""Bunch initialization helpers for the core Liénard–Wiechert integrator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from core.constants import (
    C_MMNS,
    ELEMENTARY_CHARGE,
    ELEMENTARY_CHARGE_STATC,
)

AMU_TO_MEV = 931.49410242  # Atomic mass unit → MeV/c^2 conversion
ELEMENTARY_CHARGE_GU = ELEMENTARY_CHARGE_STATC
"""Elementary charge in statcoulombs, retained for cgs analysis compatibility.

Do not use this value directly in particle states. The integrator evolves
charges in native solver units, so state dictionaries must use
``ELEMENTARY_CHARGE``.
"""

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
    transverse_geometry: str = "square"


_TRANSVERSE_GEOMETRY_ALIASES = {
    "square": "square",
    "uniform_square": "square",
    "uniform": "square",
    "random_square": "square",
    "point": "point",
    "center": "point",
    "centered": "point",
    "gaussian": "gaussian",
    "normal": "gaussian",
    "ring": "ring",
    "circle": "ring",
    "circular": "ring",
}


def _normalize_transverse_geometry(geometry: str | None) -> str:
    key = "square" if geometry is None else str(geometry).strip().lower()
    key = key.replace("-", "_").replace(" ", "_")
    try:
        return _TRANSVERSE_GEOMETRY_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_TRANSVERSE_GEOMETRY_ALIASES))
        raise ValueError(
            f"Unsupported transverse_geometry {geometry!r}. Allowed values: {allowed}"
        ) from exc


def _transverse_positions(
    *,
    count: int,
    transverse_offset_x: float,
    transverse_offset_y: float,
    transverse_spread: float,
    transverse_geometry: str | None,
    legacy_transverse_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    geometry = _normalize_transverse_geometry(transverse_geometry)
    radius = abs(float(transverse_spread))

    if geometry == "ring":
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        return (
            transverse_offset_x + radius * np.cos(angles),
            transverse_offset_y + radius * np.sin(angles),
        )

    if geometry == "gaussian" and radius > 0.0:
        return (
            np.random.normal(transverse_offset_x, radius, count),
            np.random.normal(transverse_offset_y, radius, count),
        )

    if geometry == "square" and radius > 0.0:
        return (
            np.random.uniform(
                transverse_offset_x - radius,
                transverse_offset_x + radius,
                count,
            ),
            np.random.uniform(
                transverse_offset_y - radius,
                transverse_offset_y + radius,
                count,
            ),
        )

    if geometry == "square" and legacy_transverse_radius != 0.0:
        return (
            np.full(count, legacy_transverse_radius, dtype=float),
            np.full(count, -legacy_transverse_radius, dtype=float),
        )

    return (
        np.full(count, transverse_offset_x, dtype=float),
        np.full(count, transverse_offset_y, dtype=float),
    )


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
    transverse_geometry: str = "square",
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
        Size parameter for the selected transverse geometry in mm (default: 0.0).
        For square geometry this is the uniform half-width, for gaussian geometry
        this is sigma, and for ring geometry this is radius.
    transverse_geometry : str, optional
        Transverse layout: "square"/"uniform_square", "point", "gaussian", or
        "ring"/"circle" (default: "square").

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
    macro_charge = charge_sign * ELEMENTARY_CHARGE
    char_time = 2.0 / 3.0 * macro_charge**2 / (particle_mass * C_MMNS**3)

    count = particle_count
    zeros = np.zeros(count, dtype=float)

    x, y = _transverse_positions(
        count=count,
        transverse_offset_x=transverse_offset_x,
        transverse_offset_y=transverse_offset_y,
        transverse_spread=transverse_spread,
        transverse_geometry=transverse_geometry,
        legacy_transverse_radius=transverse_radius,
    )

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
    transverse_geometry: str = "square",
) -> Tuple[ParticleState, float]:
    """Generate particle state from historical parameter names.

    This function is the maintained particle-bunch initializer for configs that
    still use the original parameter naming scheme, with support for transverse
    offset.

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
    transverse_geometry : str, optional
        Transverse layout: "square"/"uniform_square", "point", "gaussian", or
        "ring"/"circle" (default: "square"). For ring layout, transv_dist is
        interpreted as ring radius.

    Returns
    -------
    state : ParticleState
        Particle state dictionary
    rest_energy_mev : float
        Rest energy in MeV
    """
    if seed is not None:
        np.random.seed(seed)

    macro_charge = charge_sign * stripped_ions * ELEMENTARY_CHARGE
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

    x, y = _transverse_positions(
        count=pcount,
        transverse_offset_x=transv_offset_x,
        transverse_offset_y=transv_offset_y,
        transverse_spread=transv_dist,
        transverse_geometry=transverse_geometry,
    )

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
