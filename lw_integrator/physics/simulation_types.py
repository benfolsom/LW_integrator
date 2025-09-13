"""
Simulation type definitions for the LW Integrator package.

This module provides type-safe enumerations and configuration for different
simulation scenarios, replacing magic numbers with clear, documented constants.

Key Features:
- SimulationType enum for different physics scenarios
- Type-safe configuration with NamedTuple
- Default parameter sets for each simulation type
- Validation functions for parameter consistency

Author: Ben Folsom
Date: 2025-09-12
"""

from enum import IntEnum
from typing import NamedTuple, Optional, Union
import numpy as np
from .constants import C_CGS, ELECTRON_MASS, PROTON_MASS

class SimulationType(IntEnum):
    """
    Enumeration of supported simulation types.
    
    Each simulation type corresponds to different physical scenarios
    with specific boundary conditions and interaction mechanisms.
    """
    
    # Type 0: Conducting plane with aperture
    CONDUCTING_PLANE_WITH_APERTURE = 0
    
    # Type 1: Switching semiconductor (time-dependent conductivity)
    SWITCHING_SEMICONDUCTOR = 1
    
    # Type 2: Free particle bunches (no boundaries)
    FREE_PARTICLE_BUNCHES = 2
    
    def __str__(self) -> str:
        """Human-readable description of simulation type."""
        descriptions = {
            SimulationType.CONDUCTING_PLANE_WITH_APERTURE: "Conducting Plane with Aperture",
            SimulationType.SWITCHING_SEMICONDUCTOR: "Switching Semiconductor",
            SimulationType.FREE_PARTICLE_BUNCHES: "Free Particle Bunches"
        }
        return descriptions[self]
    
    @property
    def has_wall_interactions(self) -> bool:
        """Whether this simulation type includes wall/boundary interactions."""
        return self in [SimulationType.CONDUCTING_PLANE_WITH_APERTURE, 
                       SimulationType.SWITCHING_SEMICONDUCTOR]
    
    @property
    def requires_aperture_size(self) -> bool:
        """Whether this simulation type requires aperture size specification."""
        return self == SimulationType.CONDUCTING_PLANE_WITH_APERTURE
    
    @property
    def supports_time_dependent_conductivity(self) -> bool:
        """Whether this simulation type supports time-dependent conductivity."""
        return self == SimulationType.SWITCHING_SEMICONDUCTOR


class SimulationConfig(NamedTuple):
    """
    Type-safe configuration for LW integrator simulations.
    
    This replaces the previous practice of passing simulation type as
    an integer and provides clear documentation of required parameters.
    """
    
    simulation_type: SimulationType
    
    # Aperture parameters (for types 0 and 1)
    aperture_size: Optional[float] = None  # cm, radius for circular aperture
    wall_position: Optional[float] = None  # cm, z-position of conducting wall
    
    # Conductivity parameters (for type 1)
    conductivity: Optional[float] = None  # S/m, electrical conductivity
    switching_time: Optional[float] = None  # s, time when conductivity changes
    
    # Particle parameters
    particle_mass: float = ELECTRON_MASS  # g
    particle_charge: float = 4.803e-10  # esu (elementary charge in Gaussian CGS)
    
    # Integration parameters
    dt: float = 1e-15  # s, timestep
    total_time: float = 1e-12  # s, total simulation time
    
    # Gaussian integrator parameters
    convergence_tolerance: float = 1e-6  # Relative tolerance for convergence
    max_iterations: int = 5  # Maximum self-consistent iterations
    debug_mode: bool = False  # Enable debug output
    
    # Cavity and simulation geometry parameters
    cavity_spacing: float = 1e-2  # cm, spacing between cavity elements
    z_cutoff: float = 1e-2  # cm, z-position cutoff for switching simulation
    
    # Output parameters
    output_interval: int = 10  # Save every N timesteps
    
    def validate(self) -> bool:
        """
        Validate configuration parameters for consistency.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        
        # Check aperture requirements
        if self.simulation_type.requires_aperture_size:
            if self.aperture_size is None or self.aperture_size <= 0:
                raise ValueError(f"Simulation type {self.simulation_type} requires positive aperture_size")
            if self.wall_position is None:
                raise ValueError(f"Simulation type {self.simulation_type} requires wall_position")
        
        # Check conductivity requirements
        if self.simulation_type.supports_time_dependent_conductivity:
            if self.conductivity is None or self.conductivity < 0:
                raise ValueError(f"Simulation type {self.simulation_type} requires non-negative conductivity")
        
        # Check basic physical parameters
        if self.particle_mass <= 0:
            raise ValueError("particle_mass must be positive")
        if self.particle_charge == 0:
            raise ValueError("particle_charge cannot be zero")
        if self.dt <= 0:
            raise ValueError("timestep dt must be positive")
        if self.total_time <= 0:
            raise ValueError("total_time must be positive")
        if self.output_interval <= 0:
            raise ValueError("output_interval must be positive")
            
        return True


def create_simulation_config(
    simulation_type: Union[SimulationType, int],
    **kwargs
) -> SimulationConfig:
    """
    Create a validated simulation configuration.
    
    Args:
        simulation_type: Type of simulation (enum or integer)
        **kwargs: Additional configuration parameters
        
    Returns:
        SimulationConfig: Validated configuration object
        
    Examples:
        >>> config = create_simulation_config(
        ...     SimulationType.CONDUCTING_PLANE_WITH_APERTURE,
        ...     aperture_size=1e-3,  # 1 mm radius
        ...     wall_position=0.0
        ... )
        >>> config.validate()
        True
    """
    
    # Convert integer to enum if needed
    if isinstance(simulation_type, int):
        simulation_type = SimulationType(simulation_type)
    
    # Get default parameters for this simulation type
    defaults = get_default_config(simulation_type)
    
    # Override with user-provided parameters
    config_dict = defaults._asdict()
    config_dict.update(kwargs)
    config_dict['simulation_type'] = simulation_type
    
    # Create and validate configuration
    config = SimulationConfig(**config_dict)
    config.validate()
    
    return config


def get_default_config(sim_type: SimulationType) -> SimulationConfig:
    """
    Get default configuration for a simulation type.
    
    Args:
        sim_type: Type of simulation
        
    Returns:
        SimulationConfig: Default configuration for the simulation type
    """
    
    base_config = {
        'particle_mass': ELECTRON_MASS,
        'particle_charge': 4.803e-10,  # elementary charge in esu
        'dt': 1e-15,
        'total_time': 1e-12,
        'output_interval': 10,
        'convergence_tolerance': 1e-6,
        'max_iterations': 5,
        'debug_mode': False,
        'cavity_spacing': 1e-2,
        'z_cutoff': 1e-2
    }
    
    if sim_type == SimulationType.CONDUCTING_PLANE_WITH_APERTURE:
        return SimulationConfig(
            simulation_type=sim_type,
            aperture_size=1e-3,  # 1 mm
            wall_position=0.0,   # at z=0
            **base_config
        )
    
    elif sim_type == SimulationType.SWITCHING_SEMICONDUCTOR:
        return SimulationConfig(
            simulation_type=sim_type,
            aperture_size=1e-3,  # 1 mm
            wall_position=0.0,   # at z=0
            conductivity=1e6,    # 1 MS/m (typical semiconductor)
            switching_time=1e-13,  # 100 fs
            **base_config
        )
    
    elif sim_type == SimulationType.FREE_PARTICLE_BUNCHES:
        return SimulationConfig(
            simulation_type=sim_type,
            **base_config
        )
    
    else:
        raise ValueError(f"Unknown simulation type: {sim_type}")


# Legacy support functions for backward compatibility
def get_simulation_type_name(sim_type: Union[SimulationType, int]) -> str:
    """Get human-readable name for simulation type (legacy support)."""
    if isinstance(sim_type, int):
        sim_type = SimulationType(sim_type)
    return str(sim_type)


def is_wall_simulation(sim_type: Union[SimulationType, int]) -> bool:
    """Check if simulation type involves wall interactions (legacy support)."""
    if isinstance(sim_type, int):
        sim_type = SimulationType(sim_type)
    return sim_type.has_wall_interactions


# Export commonly used types and functions
__all__ = [
    'SimulationType',
    'SimulationConfig', 
    'create_simulation_config',
    'get_default_config',
    'get_simulation_type_name',
    'is_wall_simulation'
]