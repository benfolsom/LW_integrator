# Integrator Testbed Tests

This directory contains tests for the maintained testbed helper functionality.

## Running the Tests

To run the tests, you need to have `pytest` installed:

```bash
pip install pytest numpy
```

Then run the tests:

```bash
# From the project root
python -m pytest tests/test_integrator_testbed.py -v

# Or run directly
pytest tests/test_integrator_testbed.py -v
```

## Test Coverage

The test suite covers:

1. **Energy Series Computation** (`TestComputeDeltaEnergySeries`)
   - Zero energy change scenarios
   - Linear energy gain
   - Energy loss scenarios
   - Validates ΔE vs Δz calculations

2. **Filename Generation** (`TestFilenameGeneration`)
   - Timestamp-based filenames
   - Config name incorporation
   - Ensures unique filenames for each run

3. **Plot Validation** (`TestPlotValidation`)
   - Validates ΔE vs Δz plotting (not Δγ/γ)
   - Checks relative position calculations
   - Ensures proper units (GeV for energy, mm for distance)

4. **Configuration Management** (`TestConfigManagement`)
   - Config snapshot structure
   - JSON serialization/deserialization
   - All required fields present

5. **State Normalization** (`TestNormalizeState`)
   - Scalar-to-array normalization for trajectory payloads
   - Metadata preservation for integrator halt markers

## Key Assertions

### Energy Plotting
- Plots show **ΔE (GeV) vs Δz (mm)**, not Δγ/γ vs time
- Energy changes are computed using `lw_integrator.trajectory_metrics.compute_delta_energy_series`
- Initial position is subtracted to show Δz

### Filename Convention
- Format: `{config_name}_{plot_type}_{timestamp}.png`
- Example: `electronwall10_3gev_energy_20251022_143025.png`
- Timestamp format: `YYYYMMDD_HHMMSS`

## Integration with Notebook

These tests validate the core logic used by the maintained testbed helpers,
ensuring that:
- Energy calculations are correct
- Filenames are unique and descriptive
- Plot data uses the correct axes and units
- Configuration management works properly

## Future Enhancements

Consider adding tests for:
- Transverse plot data validation
- Trajectory export format validation
- Widget state management
- Observer deduplication logic
