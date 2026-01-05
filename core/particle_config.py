"""Particle configuration defaults and field definitions.

This module contains the default parameter sets for rider and driver bunches,
as well as field definitions used throughout the GUI and validation scripts.
These were previously scattered in examples/validation/core_vs_legacy_benchmark.py
but are now centralized here for easier access from core, GUI, and validation code.
"""

from typing import Dict, Tuple

__all__ = [
    "PARTICLE_PARAM_FIELDS",
    "DEFAULT_RIDER_PARAMS",
    "DEFAULT_DRIVER_PARAMS",
]

# Field names for particle parameter dictionaries
PARTICLE_PARAM_FIELDS: Tuple[str, ...] = (
    "starting_distance",
    "transv_mom",
    "starting_Pz",
    "stripped_ions",
    "m_particle",
    "transv_dist",
    "transv_offset_x",
    "transv_offset_y",
    "pcount",
    "charge_sign",
)

# Default parameters for the rider (trailing) bunch
# These represent a typical electron bunch configuration
DEFAULT_RIDER_PARAMS: Dict[str, float | int] = {
    "starting_distance": 1.0e-6,  # Initial separation in mm
    "transv_mom": 0.0,  # Transverse momentum in amu·mm/ns
    "starting_Pz": 1.01e6,  # Longitudinal momentum in amu·mm/ns
    "stripped_ions": 1.0,  # Charge state (1 for electrons)
    "m_particle": 1.007319468,  # Particle mass in amu (~proton mass for legacy reasons)
    "transv_dist": 2.0e-4,  # Transverse distribution size in mm
    "transv_offset_x": 0.0,  # Transverse offset x in mm
    "transv_offset_y": 0.0,  # Transverse offset y in mm
    "pcount": 5,  # Number of particles in bunch
    "charge_sign": -1.0,  # Charge sign (-1 for electrons/negative)
}

# Default parameters for the driver (leading) bunch
# These represent a typical ion bunch configuration
DEFAULT_DRIVER_PARAMS: Dict[str, float | int] = {
    "starting_distance": 1000.0,  # Initial separation in mm
    "transv_mom": 0.0,  # Transverse momentum in amu·mm/ns
    "starting_Pz": -1.01e6 / 207.2 * 1.007319468,  # Scaled for ion mass
    "stripped_ions": 54.0,  # Charge state (e.g., Xe54+)
    "m_particle": 207.2,  # Particle mass in amu (typical ion)
    "transv_dist": 2.0e-4 - 8.0e-2,  # Transverse distribution size in mm
    "transv_offset_x": 0.0,  # Transverse offset x in mm
    "transv_offset_y": 0.0,  # Transverse offset y in mm
    "pcount": 5,  # Number of particles in bunch
    "charge_sign": 1.0,  # Charge sign (+1 for ions/positive)
}
