"""
Unified Electromagnetic Field Integration Interface

CAI: Main entry point for electromagnetic field simulations with automatic
optimization selection and graceful fallback between implementations.
Provides a clean, consistent API regardless of which backend is used.

Implementation Hierarchy:
1. Core algorithms (always available)
2. Performance optimization (if Numba available) 
3. Enhanced physics accuracy (when precision is critical)
4. Unified interface (auto-selection and compatibility)

Key Features:
- Automatic performance optimization selection
- Graceful fallback when dependencies unavailable
- 100% API compatibility through inheritance
- Real-time implementation reporting
- Clean constructor interface

Usage Patterns:
- Auto-optimizing: `LienardWiechertIntegrator()`
- Force standard: `LienardWiechertIntegrator(use_optimized=False)`
- Check implementation: `integrator.implementation_type`

This module provides the recommended interface for all electromagnetic
field simulations, combining ease of use with optimal performance.

Author: Ben Folsom (human oversight)  
Date: 2025-09-13 (Renamed from integrator.py)
"""

import warnings
from typing import Optional, Dict, Any

# Standard implementation (always available)
from .trajectory_integrator import LienardWiechertIntegrator as TrajectoryLienardWiechertIntegrator

# Try to import optimized version
try:
    from .performance import OptimizedLienardWiechertIntegrator
    NUMBA_AVAILABLE = True
except ImportError:
    OptimizedLienardWiechertIntegrator = None
    NUMBA_AVAILABLE = False


class LienardWiechertIntegrator(TrajectoryLienardWiechertIntegrator):
    """
    Unified Lienard-Wiechert electromagnetic field integrator.
    
    Inherits from StandardLienardWiechertIntegrator to ensure API compatibility.
    Optionally uses optimized implementations for performance-critical operations
    when Numba is available.
    
    This provides a stable API while allowing performance optimizations
    when dependencies are available.
    """
    
    def __init__(self, use_optimized: bool = True, use_adaptive_timestep: bool = True, epsilon: float = 1e-15):
        """
        Initialize the integrator with optional optimization.
        
        Args:
            use_optimized: If True, use optimized implementations when available
            use_adaptive_timestep: Enable adaptive timestep control
            epsilon: Numerical epsilon for calculations
        """
        # Always initialize the standard implementation for API compatibility
        super().__init__(use_adaptive_timestep=use_adaptive_timestep, epsilon=epsilon)
        
        self.use_optimized = use_optimized and NUMBA_AVAILABLE
        self._optimized_integrator = None
        
        if self.use_optimized:
            try:
                self._optimized_integrator = OptimizedLienardWiechertIntegrator(
                    use_adaptive_timestep=use_adaptive_timestep, 
                    epsilon=epsilon
                )
                self._implementation_type = "optimized"
            except Exception as e:
                warnings.warn(
                    f"Failed to initialize optimized integrator: {e}. "
                    "Using standard implementation.",
                    UserWarning
                )
                self._optimized_integrator = None
                self.use_optimized = False
                self._implementation_type = "standard"
        else:
            self._implementation_type = "standard"
            
        if not NUMBA_AVAILABLE and use_optimized:
            warnings.warn(
                "Numba not available, using standard implementation. "
                "Install numba for performance optimization: pip install numba",
                UserWarning
            )
    
    @property
    def implementation_type(self) -> str:
        """Get the current implementation type ('standard' or 'optimized')."""
        return self._implementation_type
    
    @property
    def is_optimized(self) -> bool:
        """Check if using optimized implementation."""
        return self.use_optimized and self._optimized_integrator is not None
    
    def calculate_electromagnetic_force_optimized(self, *args, **kwargs):
        """
        Use optimized force calculation if available, otherwise fall back to standard.
        
        This method provides a performance boost for force calculations when Numba
        is available, while maintaining full compatibility with the standard API.
        """
        if self.is_optimized:
            # Use optimized vectorized calculation
            return self._optimized_integrator.vectorized_static_integration(*args, **kwargs)
        else:
            # Fall back to standard implementation
            return self.calculate_electromagnetic_force(*args, **kwargs)
    
    def __repr__(self) -> str:
        """String representation showing implementation type."""
        return f"LienardWiechertIntegrator(implementation='{self._implementation_type}')"


# Convenience functions for direct access
def create_integrator(use_optimized: bool = True, **kwargs) -> LienardWiechertIntegrator:
    """
    Create a Lienard-Wiechert integrator with optional optimization.
    
    Args:
        use_optimized: Use optimized implementation if available
        **kwargs: Additional arguments passed to the integrator
        
    Returns:
        Configured integrator instance
    """
    return LienardWiechertIntegrator(use_optimized=use_optimized, **kwargs)


def get_available_implementations() -> Dict[str, bool]:
    """
    Get information about available implementations.
    
    Returns:
        Dictionary with implementation availability
    """
    return {
        "standard": True,  # Always available
        "optimized": NUMBA_AVAILABLE,
        "numba_available": NUMBA_AVAILABLE
    }


def print_implementation_info():
    """Print information about available implementations."""
    info = get_available_implementations()
    print("Lienard-Wiechert Integrator Implementation Info:")
    print(f"  Standard Implementation: ✅ Available")
    if info["optimized"]:
        print(f"  Optimized Implementation: ✅ Available (Numba)")
    else:
        print(f"  Optimized Implementation: ❌ Not Available (install numba)")
    
    print(f"\nRecommended usage:")
    if info["optimized"]:
        print(f"  integrator = LienardWiechertIntegrator()  # Auto-optimized")
    else:
        print(f"  integrator = LienardWiechertIntegrator()  # Standard")
        print(f"  # Install numba for performance: pip install numba")


# Legacy compatibility - export both trajectory and optimized if available
__all__ = [
    "LienardWiechertIntegrator",
    "TrajectoryLienardWiechertIntegrator", 
    "create_integrator",
    "get_available_implementations",
    "print_implementation_info",
    "NUMBA_AVAILABLE"
]

if NUMBA_AVAILABLE:
    __all__.append("OptimizedLienardWiechertIntegrator")