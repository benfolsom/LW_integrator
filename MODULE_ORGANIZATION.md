# LW Integrator Module Organization

## Overview

The LW Integrator package provides a clear hierarchy of integration modules for electromagnetic field simulation:

```
lw_integrator/core/
├── integrator.py          # 🎯 MAIN INTERFACE - Use this!
├── integration.py         # Standard implementation (base)
├── optimized_integration.py  # Performance-optimized with Numba
└── self_consistent_integrator.py  # Enhanced accuracy wrapper
```

## Module Roles

### 1. `integrator.py` - **Unified Interface** ⭐
- **Primary entry point** - use this for all new code
- Automatically selects best available implementation
- Full API compatibility with inheritance from standard implementation
- Graceful fallback when Numba unavailable
- Clean API: `LienardWiechertIntegrator(use_optimized=True)`

### 2. `integration.py` - **Standard Implementation**
- Core algorithms for Lienard-Wiechert electromagnetic fields
- Complete API with retarded field calculations
- Always available (no external dependencies)
- Fallback implementation used by unified interface

### 3. `optimized_integration.py` - **Performance Layer**
- JIT-compiled vectorized operations via Numba
- 10-100x performance improvements for large particle systems
- Requires `numba` package
- Used automatically by unified interface when available

### 4. `self_consistent_integrator.py` - **Enhanced Accuracy**
- Higher-level wrapper for improved physics accuracy
- Self-consistent field iterations
- Eliminates unphysical energy discontinuities
- Built on top of standard implementation

## Recommended Usage

### For New Code:
```python
from lw_integrator import LienardWiechertIntegrator

# Auto-optimizing interface
integrator = LienardWiechertIntegrator()  # Uses optimized if available
print(f"Using: {integrator.implementation_type}")  # 'optimized' or 'standard'

# Force standard implementation
integrator = LienardWiechertIntegrator(use_optimized=False)

# Check what's available
from lw_integrator.core.integrator import print_implementation_info
print_implementation_info()
```

### For Production/Enhanced Accuracy:
```python
from lw_integrator import SelfConsistentLienardWiechertIntegrator

# Enhanced accuracy with self-consistent iterations
integrator = SelfConsistentLienardWiechertIntegrator()
```

### For Direct Access (Legacy):
```python
# Direct access to specific implementations
from lw_integrator import StandardLienardWiechertIntegrator
from lw_integrator import OptimizedLienardWiechertIntegrator  # if available

standard = StandardLienardWiechertIntegrator()
optimized = OptimizedLienardWiechertIntegrator()  # Requires numba
```

## Performance Characteristics

| Implementation | Dependencies | Performance | Use Case |
|---------------|-------------|-------------|----------|
| **Unified** | None/Numba | Auto-optimized | **Recommended for all new code** |
| Standard | None | Baseline | Compatibility, debugging |
| Optimized | Numba | 10-100x faster | Large particle systems |
| Self-Consistent | None | Baseline + accuracy | Production simulations |

## API Compatibility

All implementations provide the same core API:
- `dist_euclid()` - Distance calculations
- `calculate_electromagnetic_force()` - Force computations
- `eqsofmotion_retarded()` - Retarded field integration
- Radiation reaction support with `_apply_radiation_reaction()`

The unified interface (`integrator.py`) inherits from the standard implementation, ensuring 100% API compatibility while providing performance optimizations when available.

## Installation for Optimization

```bash
# For performance optimization
pip install numba

# Verify optimization available
python -c "from lw_integrator.core.integrator import print_implementation_info; print_implementation_info()"
```

## Migration Guide

### From Old Code:
```python
# Old direct import
from lw_integrator.core.integration import LienardWiechertIntegrator

# New unified import (recommended)
from lw_integrator import LienardWiechertIntegrator
```

### API is Identical:
All existing code using `LienardWiechertIntegrator` will work without changes. The unified interface provides the same methods with automatic optimization.

---

**Summary**: Use `integrator.py` for all new development. It provides automatic optimization, graceful fallback, and full API compatibility with the standard implementation.