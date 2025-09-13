"""
Core module for particle dynamics and simulation infrastructure.

Contains fundamental data structures and simulation algorithms
including the production-ready Gaussian self-consistent integrator.

Now organized with clear, distinct module names:
- core_algorithms.py: Fundamental electromagnetic physics
- performance.py: JIT-optimized high-performance implementations
- physics_enhanced.py: Self-consistent enhanced accuracy
- unified_interface.py: Main entry point with auto-optimization

Author: Ben Folsom (human oversight)  
Date: 2025-09-13 (Reorganized with clear module names)
"""

# New clear module structure
from .unified_interface import (
    LienardWiechertIntegrator, create_integrator,
    get_available_implementations, print_implementation_info
)

from .trajectory_integrator import (
    LienardWiechertIntegrator as TrajectoryLienardWiechertIntegrator,
    conducting_flat, switching_flat,
    static_integrator, retarded_integrator
)

from .self_consistent_fields import (
    SelfConsistentLienardWiechertIntegrator,
    self_consistent_retarded_integrator
)

# Performance module (requires Numba)
try:
    from .performance import OptimizedLienardWiechertIntegrator
except ImportError:
    OptimizedLienardWiechertIntegrator = None

from ..physics.simulation_types import (
    SimulationType, SimulationConfig, 
    create_simulation_config, get_default_config
)

from .particles import ParticleEnsemble

# Legacy compatibility removed - use direct module imports

__all__ = [
    # Main interface (recommended)
    "LienardWiechertIntegrator", "create_integrator", 
    "get_available_implementations", "print_implementation_info",
    
    # Direct implementation access
    "TrajectoryLienardWiechertIntegrator", "OptimizedLienardWiechertIntegrator",
    "SelfConsistentLienardWiechertIntegrator",
    
    # Simulation configuration
    "SimulationType", "SimulationConfig", "create_simulation_config", "get_default_config",
    
    # Data structures
    "ParticleEnsemble"
]
