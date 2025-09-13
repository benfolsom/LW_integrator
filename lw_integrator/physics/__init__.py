"""
Physics module for electromagnetic field calculations.

Contains Lienard-Wiechert potentials, radiation, relativistic mechanics,
and type-safe simulation configuration.
"""

from .constants import (
    C_CGS, C_MMNS, ELECTRON_MASS, PROTON_MASS, 
    ELEMENTARY_CHARGE_ESU, COULOMB_CONSTANT,
    TYPICAL_TIMESTEP, TYPICAL_DISTANCE
)
from .simulation_types import SimulationType, SimulationConfig, create_simulation_config

__all__ = [
    # Physical constants
    'C_CGS', 'C_MMNS', 'ELECTRON_MASS', 'PROTON_MASS', 
    'ELEMENTARY_CHARGE_ESU', 'COULOMB_CONSTANT',
    'TYPICAL_TIMESTEP', 'TYPICAL_DISTANCE',
    
    # Simulation configuration  
    'SimulationType', 'SimulationConfig', 'create_simulation_config'
]
