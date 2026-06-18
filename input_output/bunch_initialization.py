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

AMU_TO_MEV = 931.49410242
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


def _build_charge_state(
    *,
    charge_sign: float,
    stripped_ions: float,
    particle_mass_amu: float,
    particle_count: int,
    macro_population: float = 1.0,
) -> dict[str, np.ndarray]:
    q_species = float(charge_sign) * ELEMENTARY_CHARGE * float(stripped_ions)
    q_observer = q_species
    q_source = q_species * float(macro_population)
    m_species = float(particle_mass_amu)
    char_time = (2.0 / 3.0) * q_observer**2 / (m_species * C_MMNS**3)
    return {
        "q_species": np.full(particle_count, q_species, dtype=float),
        "q_observer": np.full(particle_count, q_observer, dtype=float),
        "q_source": np.full(particle_count, q_source, dtype=float),
        "macro_population": np.full(particle_count, macro_population, dtype=float),
        "m_species": np.full(particle_count, m_species, dtype=float),
        "q": np.full(particle_count, q_source, dtype=float),
        "m": np.full(particle_count, m_species, dtype=float),
        "char_time": np.full(particle_count, char_time, dtype=float),
    }


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
    longitudinal_spread: float = 0.0,
) -> Tuple[ParticleState, float]:
    """Generate a particle state dictionary from kinetic energy inputs."""

    gamma = _compute_gamma(kinetic_energy_mev, mass_amu)
    beta = math.sqrt(1.0 - 1.0 / (gamma**2)) if gamma > 1.0 else 0.0
    particle_mass = mass_amu
    charge_state = _build_charge_state(
        charge_sign=charge_sign,
        stripped_ions=1.0,
        particle_mass_amu=particle_mass,
        particle_count=particle_count,
        macro_population=1.0,
    )

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

    Pz = np.full(count, gamma * particle_mass * C_MMNS * beta, dtype=float)
    P_total = np.sqrt(Px**2 + Py**2 + Pz**2)
    Pt = np.sqrt(P_total**2 + (particle_mass * C_MMNS) ** 2)
    gamma_arr = Pt / (particle_mass * C_MMNS)

    bx = Px / (gamma_arr * particle_mass * C_MMNS)
    by = Py / (gamma_arr * particle_mass * C_MMNS)
    bz = Pz / (gamma_arr * particle_mass * C_MMNS)

    state: ParticleState = {
        "x": x,
        "y": y,
        "z": (
            np.random.normal(position_z, longitudinal_spread, count)
            if longitudinal_spread > 0.0
            else np.full(count, position_z, dtype=float)
        ),
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
    }
    state.update(charge_state)

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
    long_dist: float = 0.0,
    transv_offset_x: float = 0.0,
    transv_offset_y: float = 0.0,
    pcount: int = 1,
    charge_sign: float = 1.0,
    seed: int | None = None,
    transverse_geometry: str = "square",
    charge_multiplier: float = 1.0,
) -> Tuple[ParticleState, float]:
    """Generate particle state from historical parameter names."""
    if seed is not None:
        np.random.seed(seed)

    charge_state = _build_charge_state(
        charge_sign=charge_sign,
        stripped_ions=stripped_ions,
        particle_mass_amu=m_particle,
        particle_count=pcount,
        macro_population=charge_multiplier,
    )

    if transv_mom > 0.0:
        Px = np.random.uniform(-transv_mom, transv_mom, pcount) * m_particle
        Py = np.random.uniform(-transv_mom, transv_mom, pcount) * m_particle
    else:
        Px = np.zeros(pcount, dtype=float)
        Py = np.zeros(pcount, dtype=float)

    Pz = np.random.uniform(starting_Pz, starting_Pz + 0.1, pcount) * m_particle
    Pt = np.sqrt(Px**2 + Py**2 + Pz**2 + (m_particle * C_MMNS) ** 2)
    gamma = Pt / (m_particle * C_MMNS)

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

    if long_dist > 0.0:
        z = np.random.normal(starting_distance, long_dist, pcount)
    else:
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
    }
    state.update(charge_state)

    rest_energy_mev = m_particle * AMU_TO_MEV
    return state, rest_energy_mev
