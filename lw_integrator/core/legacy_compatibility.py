"""
Backward Compatibility Layer

CAI: Maintains compatibility with old module names while encouraging
migration to the new, clearer module organization.

This module provides import aliases so existing code continues to work
while providing deprecation warnings to encourage migration.
"""

import warnings
from typing import Any

# New module imports
from .core_algorithms import LienardWiechertIntegrator
from .performance import OptimizedLienardWiechertIntegrator
from .unified_interface import (
    create_integrator, 
    get_available_implementations,
    print_implementation_info,
    NUMBA_AVAILABLE
)

def _deprecation_warning(old_name: str, new_name: str) -> None:
    """Issue a deprecation warning for old module names."""
    warnings.warn(
        f"Module '{old_name}' is deprecated. Use '{new_name}' instead. "
        f"The old name will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

# Create compatibility layer
class _DeprecatedModuleWrapper:
    """Wrapper that issues deprecation warnings on access."""
    
    def __init__(self, new_module: Any, old_name: str, new_name: str):
        self._module = new_module
        self._old_name = old_name
        self._new_name = new_name
        self._warned = False
    
    def __getattr__(self, name: str) -> Any:
        if not self._warned:
            _deprecation_warning(self._old_name, self._new_name)
            self._warned = True
        return getattr(self._module, name)

# Make old module names available with deprecation warnings
import sys
import importlib

# Only create these aliases when the old names are actually imported
def __getattr__(name: str) -> Any:
    if name == "integration":
        from . import core_algorithms
        _deprecation_warning("integration", "core_algorithms")
        return core_algorithms
    elif name == "optimized_integration":
        from . import performance  
        _deprecation_warning("optimized_integration", "performance")
        return performance
    elif name == "integrator":
        from . import unified_interface
        _deprecation_warning("integrator", "unified_interface") 
        return unified_interface
    elif name == "self_consistent_integrator":
        from . import physics_enhanced
        _deprecation_warning("self_consistent_integrator", "physics_enhanced")
        return physics_enhanced
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")