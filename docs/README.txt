Covariant integrator libraries with demo jupyter notebooks. These demos are only configured for two-particle tests with no conducting surface.

## NEW: Corrected Gaussian Self-Consistent Integrator (September 2025)
**RECOMMENDED**: Use `gaussian_retarded_integrator_corrected.py` for improved electromagnetic field calculations.

### Key Features:
- ✅ Eliminates unphysical energy discontinuities found in original Gaussian integrator
- ✅ Preserves exact physics of retarded_integrator3 
- ✅ Adds self-consistent Gaussian enhancement for better accuracy
- ✅ Same interface as existing notebooks expect

### Usage:
```python
from gaussian_retarded_integrator_corrected import gaussian_retarded_integrator3
# Use exactly like original integrator - same parameters and interface
```

### Demonstration:
- **`gaussian_corrected_demo.ipynb`** - Comprehensive comparison of original vs corrected integrator
- **`local/docs/session_progress.md`** - Complete development documentation
- **`local/tests/verify_fix.py`** - Verification script proving the fix works

## Original Documentation:

The unconventional units used throughout are mm.ns.amu; these are used to avoid overflow or rounding errors across a large energy range.

Note that the the conjugate momentum can be read as real, physical momentum only in the initialization step, 
after which they are dependent on the external potential.

To set values to a desired starting energy, one can adjust the Pz, Px, and Py values and do a test run.
The initial energy and gamma for both the test particle "rider" and driving particle "driver" are based on these and are printed by default.

For the demos provided here, the main variables to consider are transv_dist, which is the starting transverse offset between the particles;
 Pz, which is starting conjugate momentum on the beam axis; and step_size, which determines the precision of the simulation. 
 However, taking step sizes less than about 1e-7 (ns) does not always lead to more reliable results (and dramatically increases simulation time):
 I suspect that function for syncing the present integration step with the retarded step of the incoming field (chrono_jn) becomes unstable
 in this regime, as it relies on a value-matching algorithm that not work well with such small floats. 