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
from .core.adaptive_timestep import AdaptiveTimestepController
from .core.gaussian_integrator import GaussianLiénardWiechertIntegrator, gaussian_retarded_integrator3

# Note: OptimizedLiénardWiechertIntegrator requires Numba and is imported separately
# from .core.optimized_integration import OptimizedLiénardWiechertIntegrator  

# Physics constants and simulation types
from .physics.constants import *
from .physics.simulation_types import SimulationType, SimulationConfig, create_simulation_config

# Convenience aliases
LWIntegrator = LiénardWiechertIntegrator
# OptimizedLWIntegrator = OptimizedLiénardWiechertIntegrator
GaussianLWIntegrator = GaussianLiénardWiechertIntegrator

__all__ = [
    # Version info
    "__version__", "__author__", "__email__",
    
    # Core classes
    "LiénardWiechertIntegrator", "LWIntegrator", 
    # "OptimizedLiénardWiechertIntegrator", "OptimizedLWIntegrator",  # Requires Numba
    "GaussianLiénardWiechertIntegrator", "GaussianLWIntegrator",
    "AdaptiveTimestepController",
    
    # Legacy functions
    "gaussian_retarded_integrator3",
    
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
