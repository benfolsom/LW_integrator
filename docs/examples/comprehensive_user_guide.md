# LW Integrator - Human-Readable Documentation Example

## Overview

The LW (Liénard-Wiechert) Integrator is a covariant electromagnetic particle tracking code designed for high-precision accelerator physics simulations. This documentation provides a complete human-readable reference for users and developers.

## Key Features

### Physics Implementation
- **Covariant electromagnetic field calculations** using Benjamin Folsom's validated formalism
- **Conjugate momentum formulation**: P_α = m·V_α + (e/c)·A_α (Jackson formalism)
- **Gaussian electromagnetic units** with amu-mm-ns unit system
- **Radiation reaction effects** including Abraham-Lorentz-Dirac dynamics
- **Retarded field calculations** for self-consistent particle interactions

### Computational Capabilities
- **Adaptive time-stepping** for stability and accuracy
- **Self-consistent field evolution** for multi-particle systems
- **Conducting aperture modeling** with proper boundary conditions
- **Energy gain/loss tracking** for beam dynamics analysis
- **High-precision integration** with configurable tolerances

## Installation and Setup

### Prerequisites
```bash
# Required packages
pip install numpy scipy matplotlib pandas

# Optional for development
pip install pytest sphinx jupyter
```

### Quick Start
```python
import physics.constants as const
import physics.particle_initialization as particles
import core.adaptive_integration as integrator

# Create a particle
particle = particles.create_particle(
    energy_mev=10000,           # 10 GeV particle
    position=[0, 0, 0],         # mm
    velocity_direction=[0, 0, 1], # z-direction
    charge_gaussian=const.ELEMENTARY_CHARGE_GAUSSIAN,
    mass_amu=const.ELECTRON_MASS_AMU
)

# Run simulation
config = integrator.SimulationConfig(
    dt_initial=1e-6,           # ns
    aperture_radius=1.0,       # mm
    radiation_reaction=True
)

results = integrator.AdaptiveLienardWiechertIntegrator(config).integrate(
    particles=[particle],
    t_final=1e-3              # ns
)
```

## Core Physics Modules

### physics.constants
**Purpose**: Fundamental physics constants in Gaussian units
```python
C_MMNS = 299.792458                    # mm/ns (speed of light)
ELEMENTARY_CHARGE_GAUSSIAN = 1.178734e-5  # Gaussian charge units
ELECTRON_MASS_AMU = 5.485799e-4        # amu
```

### physics.particle_initialization
**Purpose**: Particle creation and initialization
- `create_particle()` - Standard particle creation
- `ParticleSpecies` - Predefined particle types
- `validate_initial_conditions()` - Physics consistency checks

### core.adaptive_integration
**Purpose**: Main integration algorithms
- `AdaptiveLienardWiechertIntegrator` - Core integrator class
- `SimulationConfig` - Configuration parameters
- Automatic time-step adaptation for stability

## Analysis Tools

### analysis.aperture_verification
**Purpose**: Beam-aperture interaction analysis
```python
from analysis import aperture_verification as av

# Initialize particle beam
particles = av.enhanced_beam_initialization(
    n_particles=100,
    beam_sigma=0.1,      # mm
    aperture_radius=1.0, # mm
    energy_mev=10000
)

# Run simulation
results = av.run_enhanced_simulation(
    particles=particles,
    aperture_radius=1.0,
    total_length=2.0,    # mm
    n_steps=100
)
```

### analysis.interactive_analysis
**Purpose**: Quick testing and analysis
```python
from analysis import interactive_analysis as ia

# Quick test simulation
results = ia.quick_test(
    aperture_mm=1.0,
    n_particles=30,
    beam_sigma_mm=0.1,
    energy_mev=10000
)
```

## Physics Validation

### Unit System Consistency
The integrator uses Benjamin Folsom's carefully designed unit system:
- **Length**: millimeters (mm)
- **Time**: nanoseconds (ns)
- **Mass**: atomic mass units (amu)
- **Electromagnetic units**: Gaussian CGS
- **Charge conversion**: SI → Gaussian → amu-mm-ns units

### Validated Physics
✅ **Conjugate momentum formalism** - Correct implementation of P_α = m·V_α + (e/c)·A_α
✅ **Gaussian electromagnetic units** - No 4π factors, consistent field calculations
✅ **Energy conservation** - Verified for isolated particle systems
✅ **Radiation reaction** - Abraham-Lorentz-Dirac force implementation
✅ **Aperture interactions** - Proper boundary condition handling

## Configuration Examples

### Basic Simulation Config
```python
from physics.simulation_types import SimulationConfig

config = SimulationConfig(
    dt_initial=1e-6,        # Initial time step (ns)
    dt_min=1e-12,          # Minimum time step (ns)
    dt_max=1e-3,           # Maximum time step (ns)
    tolerance=1e-10,       # Convergence tolerance
    radiation_reaction=True,
    aperture_radius=1.0,   # mm
    save_trajectories=True
)
```

### High-Precision Config
```python
config = SimulationConfig(
    dt_initial=1e-8,       # Smaller initial step
    tolerance=1e-12,       # Tighter tolerance
    max_iterations=10000,  # More iterations allowed
    radiation_reaction=True,
    retardation_effects=True,
    save_fields=True       # Save field data
)
```

## Development Guidelines

### Import Structure
All modules use absolute imports:
```python
# ✅ Correct
from physics.constants import C_MMNS
from core.adaptive_integration import AdaptiveLienardWiechertIntegrator

# ❌ Avoid relative imports
from ..physics.constants import C_MMNS  # DON'T DO THIS
```

### Filing Protocol
- All documentation goes in `docs/` directory
- Temporary work in `filing_system/session_work_YYYYMMDD/`
- No loose files in workspace root
- Follow professional scientific coding standards

### Testing
```bash
# Run physics validation tests
python -m pytest tests/physics_tests/

# Run unit tests
python -m pytest tests/unit_tests/

# Run integration tests
python -m pytest tests/ -k "integration"
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all modules use absolute imports
   - Check Python path includes workspace root
   - Verify virtual environment activation

2. **Physics Inconsistencies**
   - Verify Gaussian charge units (not SI)
   - Check mm-ns unit consistency
   - Validate initial conditions

3. **Numerical Issues**
   - Reduce time step for better stability
   - Increase convergence tolerance for speed
   - Enable adaptive stepping

### Performance Optimization
- Use optimized integrators for production runs
- Enable parallel processing for multi-particle systems
- Consider reduced precision for preliminary studies

## References

- **Jackson, J.D.** "Classical Electrodynamics" - Theoretical foundation
- **Benjamin Folsom** - Original LW integrator design and validation
- **Abraham-Lorentz-Dirac** - Radiation reaction formalism
- **Gaussian CGS Units** - Electromagnetic unit system reference

---

**Documentation Version**: 1.0.0
**Last Updated**: September 17, 2025
**Physics Validation**: Complete ✅
**Import System**: Clean ✅
**Filing Protocol**: Compliant ✅
