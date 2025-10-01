"""
Professional particle initialization module for electromagnetic field simulations.

This module provides harmonized particle state initialization between legacy and modern
integrator systems, ensuring consistent physics across both implementations.
"""

from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np

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

    charges = np.full(particle_count, charge_sign * stripped_ions)
    masses = np.full(particle_count, particle_mass_amu)

    # Initialize all required integrator fields
    times = np.zeros(particle_count)
    char_times = np.full(particle_count, 1e-15)  # Characteristic time scale

    # Initialize velocities and accelerations
    bx = np.zeros(particle_count)
    by = np.zeros(particle_count)
    bz = np.zeros(particle_count)
    bdotx = np.zeros(particle_count)
    bdoty = np.zeros(particle_count)
    bdotz = np.zeros(particle_count)

    # Calculate initial gamma and momenta
    gammas = np.ones(particle_count)  # Initialize to rest
    Px = momenta_x.copy()
    Py = momenta_y.copy()
    Pz = momenta_z.copy()
    Pt = np.sqrt(Px**2 + Py**2 + Pz**2)

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
    rider_params: ParticleParams, driver_params: ParticleParams
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float]:
    """
    Initialize both rider and driver particle bunches.

    Parameters:
    -----------
    rider_params : Dict[str, float]
        Rider particle parameters
    driver_params : Dict[str, float]
        Driver particle parameters

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
    )

    return rider_state, driver_state, rider_energy, driver_energy
