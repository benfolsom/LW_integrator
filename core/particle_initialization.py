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
        Multiplier for particle charge (for macroparticle simulations). Default 1.0.

    Returns:
    --------
    Tuple[Dict[str, Any], float]
        Particle state dictionary and rest energy in MeV
    """

    # Physical constants matching the archived reference values exactly.
    amu_to_mev = 931.494  # Conversion factor

    # Calculate rest energy
    rest_energy_mev = particle_mass_amu * amu_to_mev

    # Initialize particle arrays
    positions_x = np.zeros(particle_count)
    positions_y = np.full(particle_count, transv_distance)
    positions_z = np.full(particle_count, starting_distance)

    momenta_x = np.full(particle_count, transv_momentum)
    momenta_y = np.zeros(particle_count)
    momenta_z = np.full(particle_count, starting_pz)

    # Convert charge to amu-mm-ns units (must match the reference values exactly).
    charges = np.full(
        particle_count,
        charge_sign * ELEMENTARY_CHARGE * stripped_ions * charge_multiplier,
    )
    masses = np.full(particle_count, particle_mass_amu)

    # Initialize all required integrator fields
    times = np.zeros(particle_count)

    # Calculate characteristic time for radiation reaction
    # char_time = (2/3) * q^2 / (m * c^3)
    q_value = charge_sign * ELEMENTARY_CHARGE * stripped_ions * charge_multiplier
    char_time_value = (2.0 / 3.0) * q_value**2 / (particle_mass_amu * C_MMNS**3)
    char_times = np.full(particle_count, char_time_value)

    # Calculate initial gamma and momenta from input momentum
    # Match the archived reference initialization: Pt = sqrt(Px^2 + Py^2 + Pz^2 + (mc)^2)
    Px = momenta_x.copy()
    Py = momenta_y.copy()
    Pz = momenta_z.copy()
    Pt = np.sqrt(Px**2 + Py**2 + Pz**2 + (particle_mass_amu * C_MMNS) ** 2)

    # Calculate gamma from relativistic energy-momentum relation
    gammas = Pt / (particle_mass_amu * C_MMNS)

    # Calculate beta (velocity) from momentum and gamma
    bx = Px / (gammas * particle_mass_amu * C_MMNS)
    by = Py / (gammas * particle_mass_amu * C_MMNS)
    bz = Pz / (gammas * particle_mass_amu * C_MMNS)

    # Initialize accelerations
    bdotx = np.zeros(particle_count)
    bdoty = np.zeros(particle_count)
    bdotz = np.zeros(particle_count)

    # Create particle state dictionary (compatible with both integrators)
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
        "q": charges,
        "m": masses,
        "char_time": char_times,
        "count": particle_count,
        "rest_energy_mev": rest_energy_mev,
    }

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
    # Coordinate axis least aligned with n (smallest |dot|).
    dots = np.abs(candidates @ n)
    ref = candidates[int(np.argmin(dots))]
    u = ref - n * np.dot(ref, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return u, v


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

    Parameters:
    -----------
    starting_position_mm : tuple[float, float, float]
        Bunch centroid position in mm.
    momentum_axis : tuple[float, float, float]
        Direction of the bunch's longitudinal momentum (need not be unit).
    kinetic_energy_mev : float
        Kinetic energy per particle in MeV.
    stripped_ions : float
        Number of stripped electrons (ionization state).
    particle_mass_amu : float
        Particle mass in atomic mass units.
    particle_count : int
        Number of particles in the bunch.
    charge_sign : float
        Charge sign (+1 or -1).
    transverse_distance_mm : float
        Transverse offset of each particle from the axis, applied along the
        first transverse axis. Default 0.0.
    transverse_momentum : float
        Transverse momentum component applied along the first transverse axis.
        Default 0.0.
    longitudinal_span_mm : float
        Spread of particles along the momentum axis, centered on
        ``starting_position_mm``. Particles are distributed evenly across this
        span. Default 0.0.
    transverse_axes : tuple[tuple[float,float,float], tuple[float,float,float]] | None
        Pair of orthonormal axes perpendicular to ``momentum_axis`` along which
        transverse spread is applied. If ``None``, auto-computed via stable
        Gram-Schmidt.
    charge_multiplier : float
        Multiplier for particle charge (macroparticle simulations). Default 1.0.

    Returns:
    --------
    Tuple[Dict[str, Any], float]
        Particle state dictionary and rest energy in MeV.
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

    # Relativistic momentum magnitude from kinetic energy.
    gamma = 1.0 + kinetic_energy_mev / rest_energy_mev
    beta = np.sqrt(max(0.0, 1.0 - 1.0 / gamma**2))
    p_long = gamma * particle_mass_amu * beta * C_MMNS

    # Per-particle longitudinal offsets (even spread across span, centered).
    if particle_count > 1 and longitudinal_span_mm != 0.0:
        fracs = np.linspace(-0.5, 0.5, particle_count)
        long_offsets = fracs * longitudinal_span_mm
    else:
        long_offsets = np.zeros(particle_count)

    # Per-particle positions: centroid + longitudinal offset along n +
    # transverse offset along u.
    positions = (
        centroid[None, :]
        + long_offsets[:, None] * n[None, :]
        + transverse_distance_mm * u[None, :]
    )
    positions_x = positions[:, 0].copy()
    positions_y = positions[:, 1].copy()
    positions_z = positions[:, 2].copy()

    # Per-particle momenta: longitudinal along n + transverse along u.
    # All particles share the same momentum (no per-particle momentum spread
    # in this initializer).
    momenta_vec = p_long * n + transverse_momentum * u
    momenta_x = np.full(particle_count, momenta_vec[0])
    momenta_y = np.full(particle_count, momenta_vec[1])
    momenta_z = np.full(particle_count, momenta_vec[2])

    # Charge and mass (same convention as create_particle_state).
    q_value = charge_sign * ELEMENTARY_CHARGE * stripped_ions * charge_multiplier
    charges = np.full(particle_count, q_value)
    masses = np.full(particle_count, particle_mass_amu)

    times = np.zeros(particle_count)

    char_time_value = (2.0 / 3.0) * q_value**2 / (particle_mass_amu * C_MMNS**3)
    char_times = np.full(particle_count, char_time_value)

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
        "q": charges,
        "m": masses,
        "char_time": char_times,
        "count": particle_count,
        "rest_energy_mev": rest_energy_mev,
    }

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
        Multiplier for particle charges (for macroparticle simulations). Default 1.0.

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
        -_as_float(rider_params["transv_distance"]),  # Opposite transverse position
        _as_int(driver_params["particle_count"]),
        _as_float(driver_params["charge_sign"]),
        charge_multiplier=charge_multiplier,
    )

    return rider_state, driver_state, rider_energy, driver_energy
