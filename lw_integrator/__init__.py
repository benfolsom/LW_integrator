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

Author: Ben Folsom (human oversight)
Version: 1.0.0
Date: 2025-09-12
"""

# Version information
__version__ = "1.0.0"
__author__ = "Ben Folsom"
__email__ = "ben.folsom@maxlab.lu.se"

# Core imports for easy access
from .core.integration import LiénardWiechertIntegrator
from .core.optimized_integration import OptimizedLiénardWiechertIntegrator  
from .core.adaptive_timestep import AdaptiveTimestepController

# Physics constants
from .physics.constants import *

# Convenience aliases
LWIntegrator = LiénardWiechertIntegrator
OptimizedLWIntegrator = OptimizedLiénardWiechertIntegrator

__all__ = [
    # Version info
    "__version__", "__author__", "__email__",
    
    # Core classes
    "LiénardWiechertIntegrator", "LWIntegrator", 
    "OptimizedLiénardWiechertIntegrator", "OptimizedLWIntegrator",
    "AdaptiveTimestepController",
    
    # Constants (imported from physics.constants)
    "C_MMNS", "PROTON_MASS", "ELECTRON_MASS", "ELEMENTARY_CHARGE",
    "COULOMB_CONSTANT", "TYPICAL_TIMESTEP", "TYPICAL_DISTANCE"
]

# Package metadata
__doc_url__ = "https://lw-integrator.readthedocs.io/"
__source_url__ = "https://github.com/username/LW_integrator"
__description__ = "Production-ready Lienard-Wiechert electromagnetic field simulator"
