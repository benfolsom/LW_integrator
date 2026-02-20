"""
Professional particle initialization module for electromagnetic field simulations.

This module provides harmonized particle state initialization between legacy and modern
integrator systems, ensuring consistent physics across both implementations.
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
    Create particle state initialization compatible with both legacy and modern integrators.

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

    # Physical constants (matching legacy values exactly)
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

    # Convert charge to amu-mm-ns units (must match legacy exactly!)
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
    # Following legacy initialization: Pt = sqrt(Px^2 + Py^2 + Pz^2 + (mc)^2)
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
