"""
Core module for particle dynamics and simulation infrastructure.

Contains fundamental data structures and simulation algorithms
including the production-ready Gaussian self-consistent integrator.

Author: Ben Folsom (human oversight)
Date: 2025-09-13
"""

from ..physics.simulation_types import (
    SimulationType, SimulationConfig, 
    create_simulation_config, get_default_config
)

from .gaussian_integrator import (
    GaussianLiénardWiechertIntegrator,
    gaussian_retarded_integrator3
)

from .integration import (
    LiénardWiechertIntegrator,
    conducting_flat, switching_flat,
    static_integrator, retarded_integrator3
)

from .particles import ParticleEnsemble

__all__ = [
    # Simulation configuration
    'SimulationType', 'SimulationConfig', 
    'create_simulation_config', 'get_default_config',
    
    # Main integrators
    'GaussianLiénardWiechertIntegrator', 'gaussian_retarded_integrator3',
    'LiénardWiechertIntegrator', 'retarded_integrator3',
    
    # Wall and surface functions
    'conducting_flat', 'switching_flat', 'static_integrator',
    
    # Particle data structures
    'ParticleEnsemble'
]
