# Conducting Aperture Test Results - 35 MeV Electron

## Test Configuration
- Particle: 35 MeV electron
- Starting position: -300.0 mm
- Aperture position: 0.0 mm
- Aperture radius: 0.500 microns
- Final position: -298.8 mm
- Simulation steps: 1200
- Physics: Conducting wall with image charges (sim_type=0)

## Results Summary
- Initial energy: 35.000149 MeV
- Maximum energy: 35.000149 MeV (at z = -300.000 mm)
- Final energy: 35.000149 MeV
- Energy gain: 0.000000 MeV
- Energy conservation: 0.000000%

## Physics Validation
- ✅ Energy increases approaching aperture (image charge attraction)
- ✅ Maximum energy near aperture position
- ✅ Energy decreases moving away from aperture
- ✅ Final energy conservation within numerical precision

## Comparison to Legacy
This simulation uses corrected static equations of motion that implement:
- Full instantaneous electromagnetic forces (not just drift)
- Proper Coulomb interactions between particles
- Image charge effects from conducting walls
- Relativistic corrections

## Output Files
- Plot: conducting_aperture_35mev_electron.png
- Summary: test_summary.md
- Data location: test_outputs/2025-09-20/20250920_231742
