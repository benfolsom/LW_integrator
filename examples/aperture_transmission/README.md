# Aperture Transmission Analysis

This example demonstrates electromagnetic effects on particle transmission through a 10 micron diameter aperture across a wide energy range from 1 MeV to 100 GeV.

## Overview

The simulation studies:
- **Particle Types**: Electrons and Protons
- **Energy Range**: 1 MeV to 100 GeV (6 orders of magnitude)
- **Aperture**: 10 micron diameter (5 micron radius)
- **Physics**: Full electromagnetic interactions with retarded fields

## Key Physics

### Relativistic Regime Transition
- **Electrons**: Become relativistic at ~0.5 MeV (γ ≈ 2)
- **Protons**: Become relativistic at ~938 MeV (γ ≈ 2)

### Electromagnetic Effects
- **Low Energy**: Classical Coulomb scattering dominates
- **High Energy**: Relativistic field effects, radiation reaction
- **Aperture Interactions**: Geometric focusing, electromagnetic deflection

### Expected Results
- **Transmission Rate**: Energy-dependent due to EM deflection
- **Deflection Magnitude**: Decreases with energy (1/γ² scaling)
- **Species Comparison**: Mass-dependent relativistic effects

## Running the Analysis

```bash
cd examples/aperture_transmission
python aperture_analysis.py
```

## Output Files

- `aperture_transmission_analysis.png` - Main transmission and deflection plots
- `physics_analysis_summary.png` - Relativistic physics analysis
- `transmission_data.json` - Raw simulation data

## Expected Physics Insights

1. **Electron Transmission**: High transmission at all energies due to low mass
2. **Proton Transmission**: More complex behavior due to larger mass
3. **Electromagnetic Deflection**: Inversely proportional to momentum
4. **Relativistic Effects**: Clear transition from classical to relativistic regime

## Technical Notes

- Uses 10 particles per energy point for statistical sampling
- 8mm initial beam spread (larger than 10μm aperture)
- 500 integration steps over 200mm propagation distance
- Retarded electromagnetic field calculations
- Full relativistic equations of motion
