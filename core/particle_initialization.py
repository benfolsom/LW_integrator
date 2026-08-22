"""
Professional particle initialization module for electromagnetic field simulations.

This module provides harmonized particle state initialization between the
maintained solver and archived reference workflows, ensuring consistent
physics across both paths.
"""

from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np

from .constants import C_MMNS, ELEMENTARY_CHARGE

Scalar = Union[float, int]
ParticleParams = Mapping[str, Scalar]


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
    char_time_value = (2.0 / 3.0) * q_observer**2 / (m_species * C_MMNS**3)
    return {
        "q_species": np.full(particle_count, q_species, dtype=float),
        "q_observer": np.full(particle_count, q_observer, dtype=float),
        "q_source": np.full(particle_count, q_source, dtype=float),
        "macro_population": np.full(particle_count, macro_population, dtype=float),
        "m_species": np.full(particle_count, m_species, dtype=float),
        "q": np.full(particle_count, q_source, dtype=float),
        "m": np.full(particle_count, m_species, dtype=float),
        "char_time": np.full(particle_count, char_time_value, dtype=float),
    }


def create_particle_state(
    starting_distance: float,
    transv_momentum: float,
    starting_pz: float,
    stripped_ions: float,
    particle_mass_amu: float,
    transv_distance: float,
    particle_count: int,
    charge_sign: float,
    charge_multiplier: float = 1.0,
) -> Tuple[Dict[str, Any], float]:
    """
    Create particle state initialization compatible with both archived-reference
    and modern integrator flows.

    Parameters:
    -----------
    starting_distance : float
        Initial longitudinal position (mm)
    transv_momentum : float
        Initial transverse momentum
    starting_pz : float
        Initial longitudinal momentum
    stripped_ions : float
        Number of stripped electrons (ionization state)
    particle_mass_amu : float
        Particle mass in atomic mass units
    transv_distance : float
        Transverse separation distance
    particle_count : int
        Number of particles in bunch
    charge_sign : float
        Charge sign (+1 or -1)
    charge_multiplier : float
        Source macro-population multiplier for macroparticle simulations.

    Returns:
    --------
    Tuple[Dict[str, Any], float]
        Particle state dictionary and rest energy in MeV
    """

    amu_to_mev = 931.494
    rest_energy_mev = particle_mass_amu * amu_to_mev

    positions_x = np.zeros(particle_count)
    positions_y = np.full(particle_count, transv_distance)
    positions_z = np.full(particle_count, starting_distance)

    momenta_x = np.full(particle_count, transv_momentum)
    momenta_y = np.zeros(particle_count)
    momenta_z = np.full(particle_count, starting_pz)

    charge_state = _build_charge_state(
        charge_sign=charge_sign,
        stripped_ions=stripped_ions,
        particle_mass_amu=particle_mass_amu,
        particle_count=particle_count,
        macro_population=charge_multiplier,
    )

    times = np.zeros(particle_count)

    Px = momenta_x.copy()
    Py = momenta_y.copy()
    Pz = momenta_z.copy()
    Pt = np.sqrt(Px**2 + Py**2 + Pz**2 + (particle_mass_amu * C_MMNS) ** 2)

    gammas = Pt / (particle_mass_amu * C_MMNS)

    bx = Px / (gammas * particle_mass_amu * C_MMNS)
    by = Py / (gammas * particle_mass_amu * C_MMNS)
    bz = Pz / (gammas * particle_mass_amu * C_MMNS)

    bdotx = np.zeros(particle_count)
    bdoty = np.zeros(particle_count)
    bdotz = np.zeros(particle_count)

    particle_state = {
        "x": positions_x,
        "y": positions_y,
        "z": positions_z,
        "t": times,
        "px": momenta_x,
        "py": momenta_y,
        "pz": momenta_z,
        "Px": Px,
        "Py": Py,
        "Pz": Pz,
        "Pt": Pt,
        "bx": bx,
        "by": by,
        "bz": bz,
        "bdotx": bdotx,
        "bdoty": bdoty,
        "bdotz": bdotz,
        "gamma": gammas,
        "count": particle_count,
        "rest_energy_mev": rest_energy_mev,
    }
    particle_state.update(charge_state)

    return particle_state, rest_energy_mev


def _orthonormal_transverse_axes(
    axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors orthonormal to ``axis`` and to each other.

    Uses a stable Gram-Schmidt: pick the coordinate axis least aligned with
    ``axis`` to avoid degeneracy, then cross-product twice.
    """
    n = axis / np.linalg.norm(axis)
    candidates = np.eye(3)
    dots = np.abs(candidates @ n)
    ref = candidates[int(np.argmin(dots))]
    u = ref - n * np.dot(ref, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return u, v


def _bunch_3d_offsets(
    particle_count: int,
    transverse_radius: float,
    longitudinal_span: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian offsets filling a 3D ellipsoid.

    Returns (long_offsets, u_offsets, v_offsets) in the local bunch basis
    where ``n`` is the longitudinal (momentum) axis and ``u``, ``v`` are
    the transverse axes. The sigmas are transverse_radius and
    longitudinal_span/2. For particle_count == 1 the single particle sits
    at the centroid (zero offset).
    """
    if particle_count <= 1:
        return np.zeros(1), np.zeros(1), np.zeros(1)
    if rng is None:
        long_offsets = np.random.normal(0.0, longitudinal_span * 0.5, particle_count)
        u_offsets = np.random.normal(0.0, transverse_radius, particle_count)
        v_offsets = np.random.normal(0.0, transverse_radius, particle_count)
    else:
        long_offsets = rng.normal(0.0, longitudinal_span * 0.5, particle_count)
        u_offsets = rng.normal(0.0, transverse_radius, particle_count)
        v_offsets = rng.normal(0.0, transverse_radius, particle_count)
    return long_offsets, u_offsets, v_offsets


def create_particle_state_3d(
    starting_position_mm: tuple[float, float, float],
    momentum_axis: tuple[float, float, float],
    kinetic_energy_mev: float,
    stripped_ions: float,
    particle_mass_amu: float,
    particle_count: int,
    charge_sign: float,
    transverse_distance_mm: float = 0.0,
    transverse_momentum: float = 0.0,
    longitudinal_span_mm: float = 0.0,
    transverse_axes: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ) = None,
    charge_multiplier: float = 1.0,
) -> Tuple[Dict[str, Any], float]:
    """
    Create a particle state with arbitrary 3D bunch orientation and position.

    The bunch's longitudinal (momentum) direction is along ``momentum_axis``.
    All particles share the same longitudinal momentum magnitude, derived from
    ``kinetic_energy_mev`` via the relativistic energy-momentum relation.

    The returned state dict has the same keys and shapes as
    :func:`create_particle_state`, making it drop-in compatible with the
    integrator.
    """
    amu_to_mev = 931.494
    rest_energy_mev = particle_mass_amu * amu_to_mev

    n = np.asarray(momentum_axis, dtype=float)
    n = n / np.linalg.norm(n)

    if transverse_axes is None:
        u, v = _orthonormal_transverse_axes(n)
    else:
        u = np.asarray(transverse_axes[0], dtype=float)
        u = u / np.linalg.norm(u)
        v = np.asarray(transverse_axes[1], dtype=float)
        v = v / np.linalg.norm(v)

    centroid = np.asarray(starting_position_mm, dtype=float)

    gamma = 1.0 + kinetic_energy_mev / rest_energy_mev
    beta = np.sqrt(max(0.0, 1.0 - 1.0 / gamma**2))
    p_long = gamma * particle_mass_amu * beta * C_MMNS

    if particle_count > 1:
        long_offsets, u_offsets, v_offsets = _bunch_3d_offsets(
            particle_count,
            transverse_distance_mm,
            longitudinal_span_mm,
        )
    else:
        long_offsets = np.zeros(1)
        u_offsets = np.array([transverse_distance_mm])
        v_offsets = np.zeros(1)

    positions = (
        centroid[None, :]
        + long_offsets[:, None] * n[None, :]
        + u_offsets[:, None] * u[None, :]
        + v_offsets[:, None] * v[None, :]
    )
    positions_x = positions[:, 0].copy()
    positions_y = positions[:, 1].copy()
    positions_z = positions[:, 2].copy()

    momenta_vec = p_long * n + transverse_momentum * u
    momenta_x = np.full(particle_count, momenta_vec[0])
    momenta_y = np.full(particle_count, momenta_vec[1])
    momenta_z = np.full(particle_count, momenta_vec[2])

    charge_state = _build_charge_state(
        charge_sign=charge_sign,
        stripped_ions=stripped_ions,
        particle_mass_amu=particle_mass_amu,
        particle_count=particle_count,
        macro_population=charge_multiplier,
    )

    times = np.zeros(particle_count)

    Px = momenta_x.copy()
    Py = momenta_y.copy()
    Pz = momenta_z.copy()
    Pt = np.sqrt(Px**2 + Py**2 + Pz**2 + (particle_mass_amu * C_MMNS) ** 2)

    gammas = Pt / (particle_mass_amu * C_MMNS)

    bx = Px / (gammas * particle_mass_amu * C_MMNS)
    by = Py / (gammas * particle_mass_amu * C_MMNS)
    bz = Pz / (gammas * particle_mass_amu * C_MMNS)

    bdotx = np.zeros(particle_count)
    bdoty = np.zeros(particle_count)
    bdotz = np.zeros(particle_count)

    particle_state = {
        "x": positions_x,
        "y": positions_y,
        "z": positions_z,
        "t": times,
        "px": momenta_x,
        "py": momenta_y,
        "pz": momenta_z,
        "Px": Px,
        "Py": Py,
        "Pz": Pz,
        "Pt": Pt,
        "bx": bx,
        "by": by,
        "bz": bz,
        "bdotx": bdotx,
        "bdoty": bdoty,
        "bdotz": bdotz,
        "gamma": gammas,
        "count": particle_count,
        "rest_energy_mev": rest_energy_mev,
    }
    particle_state.update(charge_state)

    return particle_state, rest_energy_mev


def _as_float(value: Scalar) -> float:
    return float(value)


def _as_int(value: Scalar) -> int:
    return int(value)


def initialize_particle_bunches(
    rider_params: ParticleParams,
    driver_params: ParticleParams,
    charge_multiplier: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float]:
    """
    Initialize both rider and driver particle bunches.

    Parameters:
    -----------
    rider_params : Dict[str, float]
        Rider particle parameters
    driver_params : Dict[str, float]
        Driver particle parameters
    charge_multiplier : float
        Source macro-population multiplier for macroparticle simulations.

    Returns:
    --------
    Tuple[Dict[str, Any], Dict[str, Any], float, float]
        Rider state, driver state, rider rest energy, driver rest energy
    """

    rider_state, rider_energy = create_particle_state(
        _as_float(rider_params["starting_distance"]),
        _as_float(rider_params["transv_momentum"]),
        _as_float(rider_params["starting_pz"]),
        _as_float(rider_params["stripped_ions"]),
        _as_float(rider_params["particle_mass_amu"]),
        _as_float(rider_params["transv_distance"]),
        _as_int(rider_params["particle_count"]),
        _as_float(rider_params["charge_sign"]),
        charge_multiplier=charge_multiplier,
    )

    driver_state, driver_energy = create_particle_state(
        _as_float(driver_params["starting_distance"]),
        _as_float(driver_params["transv_momentum"]),
        _as_float(driver_params["starting_pz"]),
        _as_float(driver_params["stripped_ions"]),
        _as_float(driver_params["particle_mass_amu"]),
        -_as_float(rider_params["transv_distance"]),
        _as_int(driver_params["particle_count"]),
        _as_float(driver_params["charge_sign"]),
        charge_multiplier=charge_multiplier,
    )

    return rider_state, driver_state, rider_energy, driver_energy
