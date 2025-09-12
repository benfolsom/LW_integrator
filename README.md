# LW Integrator - Electromagnetic Field Simulator

## Overview

Covariant integrator libraries with demo jupyter notebooks for simulating Lienard-Wiechert electromagnetic fields with relativistic charged particles. These demos are currently configured for two-particle tests with optional conducting surface boundary conditions.

**Current Status**: Development version (v2.0 overhaul in progress) - see original working code in master branch

## Original Description (from README.txt)

The unconventional units used throughout are mm.ns.amu; these are used to avoid overflow or rounding errors across a large energy range.

Note that the conjugate momentum can be read as real, physical momentum only in the initialization step, after which they are dependent on the external potential.

To set values to a desired starting energy, one can adjust the Pz, Px, and Py values and do a test run. The initial energy and gamma for both the test particle "rider" and driving particle "driver" are based on these and are printed by default.

For the demos provided here, the main variables to consider are:
- `transv_dist`: starting transverse offset between particles
- `Pz`: starting conjugate momentum on the beam axis  
- `step_size`: determines the precision of the simulation

**Important limitation**: Taking step sizes less than about 1e-7 (ns) does not always lead to more reliable results and dramatically increases simulation time. The function for syncing the present integration step with the retarded step of the incoming field (`chrono_jn`) becomes unstable in this regime.

## Known Critical Issues

⚠️ **PRIORITY #1**: Code instability for electron energies in the GeV range - requires immediate attention

## Current Development Goals

The ongoing v2.0 overhaul aims to address:
- Performance optimization (currently O(N²) scaling)
- Code modularity and testing
- Energy stability issues
- Preparation for larger particle simulations

## Installation (Development Version)

```bash
# Clone and switch to development branch
git clone https://github.com/benfolsom/LW_integrator.git
cd LW_integrator
git checkout overhaul-dev

# Set up development environment  
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
pip install -e .
```

## Original Working Code

For the current stable (but unoptimized) version, use the master branch and refer to the Jupyter notebook demonstrations.

## Physics Implementation

This package implements:
- Lienard-Wiechert retarded electromagnetic potentials
- Relativistic particle dynamics with proper time integration
- Synchrotron radiation reaction forces
- Conducting surface boundary conditions with image charges
- Covariant formulation for numerical stability

## Units Convention

- **Length**: millimeters (mm)
- **Time**: nanoseconds (ns)  
- **Mass**: atomic mass units (amu)
- **Derived units**: mm·ns⁻¹ for velocity, etc.

## Development Status

**In Progress**: Complete codebase overhaul for performance and stability
**Next Priority**: Resolve GeV energy range instabilities
**Testing**: Establishing comprehensive validation against original results

## License

MIT License - see LICENSE file for details.

## Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api.md)
- [Physics Background](docs/physics.md)
- [Performance Guide](docs/performance.md)

## Testing

Run the test suite:

```bash
pytest lw_integrator/tests/
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{folsom2024lw,
  title={High-performance Lienard-Wiechert field integrator},
  author={Folsom, Ben},
  journal={Nuclear Instruments and Methods in Physics Research},
  year={2024}
}
```

## Original Research

This software was developed to produce the results in the attached research paper on electromagnetic field calculations for particle accelerator physics.
