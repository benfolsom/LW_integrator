"""
LW Integrator: Production-Ready Lienard-Wiechert Electromagnetic Field Simulator

A high-performance package for simulating electromagnetic interactions using 
exact Lienard-Wiechert retarded potentials with GeV-scale capability.

Key Features:
- Ultra-relativistic particle dynamics (up to 100+ GeV)
- Exact electromagnetic retardation effects
- JIT-optimized performance (670k+ force calculations/sec)
- Comprehensive physics validation
- Production-ready package structure
- Unified interface with automatic optimization

Quick Start:
    from lw_integrator import LienardWiechertIntegrator
    
    # Auto-optimizing interface (recommended)
    integrator = LienardWiechertIntegrator()  # Uses Numba optimization if available
    
    # Check implementation
    print(f"Using: {integrator.implementation_type}")  # 'optimized' or 'standard'

Author: Ben Folsom (human oversight)
Version: 1.0.0
Date: 2025-09-12
"""

# Version information
__version__ = "1.0.0"
__author__ = "Ben Folsom"
__email__ = "ben.folsom@maxlab.lu.se"

# Core imports for easy access
from .core.integrator import (
    LienardWiechertIntegrator, 
    create_integrator,
    get_available_implementations,
    print_implementation_info,
    NUMBA_AVAILABLE
)
from .core.adaptive_timestep import AdaptiveTimestepController
from .core.self_consistent_integrator import SelfConsistentLienardWiechertIntegrator, self_consistent_retarded_integrator

# Legacy imports (for direct access to specific implementations)
from .core.integration import LienardWiechertIntegrator as StandardLienardWiechertIntegrator
try:
    from .core.optimized_integration import OptimizedLienardWiechertIntegrator
except ImportError:
    OptimizedLienardWiechertIntegrator = None  

# Physics constants and simulation types
from .physics.constants import (
    C_CGS, C_MMNS, PROTON_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE_ESU,
    COULOMB_CONSTANT, TYPICAL_TIMESTEP, TYPICAL_DISTANCE
)
from .physics.simulation_types import SimulationType, SimulationConfig, create_simulation_config

# Convenience aliases
LWIntegrator = LienardWiechertIntegrator  # Unified interface
StandardLWIntegrator = StandardLienardWiechertIntegrator  # Direct standard access
OptimizedLWIntegrator = OptimizedLienardWiechertIntegrator  # Direct optimized access (if available)
SelfConsistentLWIntegrator = SelfConsistentLienardWiechertIntegrator

__all__ = [
    # Version info
    "__version__", "__author__", "__email__",
    
    # Core unified interface
    "LienardWiechertIntegrator", "LWIntegrator",  # Unified auto-optimizing interface
    "create_integrator", "get_available_implementations", "print_implementation_info",
    
    # Direct implementation access
    "StandardLienardWiechertIntegrator", "StandardLWIntegrator",  # Always available
    "OptimizedLienardWiechertIntegrator", "OptimizedLWIntegrator",  # If Numba available
    "SelfConsistentLienardWiechertIntegrator", "SelfConsistentLWIntegrator",
    "AdaptiveTimestepController",
    
    # Legacy functions
    "self_consistent_retarded_integrator",
    
    # Simulation configuration
    "SimulationType", "SimulationConfig", "create_simulation_config",
    
    # Constants (imported from physics.constants)
    "C_CGS", "C_MMNS", "PROTON_MASS", "ELECTRON_MASS", "ELEMENTARY_CHARGE_ESU",
    "COULOMB_CONSTANT", "TYPICAL_TIMESTEP", "TYPICAL_DISTANCE"
]

# Package metadata
__doc_url__ = "https://lw-integrator.readthedocs.io/"
__source_url__ = "https://github.com/username/LW_integrator"
__description__ = "Production-ready Lienard-Wiechert electromagnetic field simulator"
