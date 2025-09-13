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

from .self_consistent_integrator import (
    SelfConsistentLienardWiechertIntegrator,
    self_consistent_retarded_integrator
)

from .integration import (
    LienardWiechertIntegrator,
    conducting_flat, switching_flat,
    static_integrator, retarded_integrator
)

from .particles import ParticleEnsemble

__all__ = [
    # Simulation configuration
    'SimulationType', 'SimulationConfig', 
    'create_simulation_config', 'get_default_config',
    
    # Main integrators
    'SelfConsistentLienardWiechertIntegrator', 'self_consistent_retarded_integrator',
    'LienardWiechertIntegrator', 'retarded_integrator',
    
    # Wall and surface functions
    'conducting_flat', 'switching_flat', 'static_integrator',
    
    # Particle data structures
    'ParticleEnsemble'
]
