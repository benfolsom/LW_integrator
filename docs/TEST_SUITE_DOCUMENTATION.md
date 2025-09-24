# LW Integrator Comprehensive Test Suite

This document describes the complete test suite for the LW (Liénard-Wiechert) Integrator, including test structure, usage instructions, and validation methodologies.

## Test Suite Overview

The LW Integrator test suite provides comprehensive validation of electromagnetic particle tracking across multiple simulation types, particle species, and performance scenarios. The test framework is built using pytest and follows industry best practices for physics simulation testing.

### Test Categories

- **Unit Tests**: Individual function and method validation
- **Integration Tests**: Complete simulation workflow validation
- **Performance Tests**: Scaling and optimization verification
- **Physics Validation**: Conservation law and radiation reaction testing
- **Multi-Species Tests**: Cross-species interaction validation
- **Benchmark Tests**: Large-scale performance characterization

## Test Structure

```
tests/
├── __init__.py                          # Test package initialization
├── pytest.ini                          # Pytest configuration
├── test_config.py                      # Test utilities and configurations
├── run_tests.py                        # Comprehensive test runner
├── unit/                               # Unit tests
│   └── test_integrator_units.py        # Individual function tests
├── integration/                        # Integration tests
│   ├── test_simulation_types.py        # Simulation type validation
│   └── test_aperture_interactions.py   # Aperture and boundary tests
└── benchmarks/                         # Performance and validation
    ├── test_performance_scaling.py     # Scaling and optimization tests
    └── test_multi_species_validation.py # Multi-species and radiation tests
```

## Quick Start

### Running Tests

**Run all tests:**
```bash
cd tests/
python run_tests.py all
```

**Run specific test categories:**
```bash
# Unit tests only
python run_tests.py 1

# Performance tests
python run_tests.py 3

# Physics validation
python run_tests.py 5
```

**Run with pytest directly:**
```bash
# Fast tests (excludes slow performance tests)
pytest -m "not slow" -v

# Unit tests only
pytest unit/ -v

# Physics validation
pytest -m physics -v

# Specific test file
pytest integration/test_simulation_types.py -v
```

### Test Markers

The test suite uses pytest markers for organization:

- `@pytest.mark.unit`: Unit tests for individual functions
- `@pytest.mark.integration`: Full workflow integration tests
- `@pytest.mark.performance`: Performance and scaling tests
- `@pytest.mark.physics`: Physics conservation and validation tests
- `@pytest.mark.slow`: Long-running tests (>30 seconds)
- `@pytest.mark.gpu`: GPU-accelerated tests (if available)

## Test Configuration

### Particle Species

The test suite supports multiple particle species with realistic physics parameters:

```python
ELECTRON    = ParticleSpecies("ELECTRON", -1, 0.511)      # MeV
PROTON      = ParticleSpecies("PROTON", 1, 938.3)         # MeV
GOLD_ION    = ParticleSpecies("GOLD_ION", 79, 183627)     # MeV (Au-197)
LEAD_ION    = ParticleSpecies("LEAD_ION", 82, 193687)     # MeV (Pb-208)
```

### Test Scenarios

**Simulation Types:**
- Type 1: Simplified electromagnetic tracking
- Type 2: Full retarded field calculation
- Type 3: Complete electromagnetic simulation with radiation reaction

**Particle Counts:**
- Small scale: 2-10 particles
- Medium scale: 25-100 particles
- Large scale: 200-500+ particles

**Physical Scenarios:**
- Head-on collisions
- Near-miss aperture interactions
- Multi-species beam interactions
- High-field radiation reaction scenarios
- Heavy ion collision dynamics

## Physics Validation

### Conservation Laws

The test suite validates fundamental physics conservation:

```python
def validate_physics_conservation(trajectory_rider, trajectory_driver,
                                rider_species, driver_species, tolerance=1e-2):
    """
    Validates:
    - Energy conservation (within tolerance)
    - Momentum conservation (3D vector)
    - Charge conservation (exact)
    """
```

**Typical tolerances:**
- Energy conservation: 1-2% (accounts for numerical precision and radiation losses)
- Momentum conservation: 1-2% (accounts for field momentum transfer)
- Charge conservation: Exact (fundamental requirement)

### Radiation Reaction

High-field scenarios validate radiation reaction physics:

```python
def check_radiation_reaction_activation(trajectories, field_threshold=1e12):
    """
    Checks for:
    - Maximum electromagnetic field strengths
    - Energy loss detection
    - Field gradient calculations
    - Radiation power estimates
    """
```

## Performance Benchmarks

### Scaling Tests

Performance tests validate computational scaling:

| Particle Count | Expected Time | Memory Usage | Validation |
|----------------|---------------|--------------|-------------|
| 2-10          | <1 second     | <10 MB       | Full physics |
| 25-50         | <5 seconds    | <50 MB       | Conservation |
| 100-200       | <30 seconds   | <200 MB      | Stability |
| 500+          | <5 minutes    | <1 GB        | Numerical |

### Optimization Validation

Tests verify optimized integrator performance:

```python
def test_optimization_effectiveness():
    """
    Compares:
    - Optimized vs standard integrator speed
    - Result accuracy between implementations
    - Memory usage optimization
    """
```

Expected speedup: 1.5-3x for particle counts >20

## Error Handling and Debugging

### Common Test Failures

**Physics Conservation Failures:**
- Check particle initialization parameters
- Verify step size is appropriate for energy scale
- Ensure aperture positions don't interfere with trajectories

**Performance Failures:**
- Reduce particle count for development testing
- Check system memory availability
- Verify optimized integrator is being used

**Numerical Stability Issues:**
- Decrease step size for high-energy scenarios
- Check for NaN/infinite values in trajectories
- Validate initial particle distributions

### Debugging Tools

**Verbose Output:**
```bash
pytest -v -s  # Show print statements
```

**Physics Debugging:**
```python
# Enable detailed physics output
config.debug_physics = True
```

**Performance Profiling:**
```bash
pytest --profile  # Enable performance profiling
```

## Continuous Integration

### Pre-commit Hooks

Tests run automatically before commits:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

### Test Requirements

**Fast Test Suite** (for CI):
- Must complete in <2 minutes
- Covers all simulation types
- Validates basic physics conservation
- Tests up to 25 particles per bunch

**Complete Test Suite** (for releases):
- Comprehensive validation including slow tests
- Large-scale performance verification
- Multi-species radiation reaction validation
- Memory usage and optimization benchmarks

## Adding New Tests

### Test Development Guidelines

1. **Follow pytest conventions:**
   ```python
   def test_descriptive_name():
       """Clear docstring describing test purpose."""
       # Arrange
       # Act
       # Assert
   ```

2. **Use test configuration utilities:**
   ```python
   from tests.test_config import TestConfiguration, create_bunch_uniform_distribution

   config = TestConfiguration(particle_count=10, ...)
   bunch = create_bunch_uniform_distribution(config, ELECTRON, "gaussian")
   ```

3. **Validate physics conservation:**
   ```python
   conservation_result = validate_physics_conservation(
       trajectory_rider, trajectory_driver,
       rider_species, driver_species
   )
   assert conservation_result["energy_conserved"]
   ```

4. **Use appropriate markers:**
   ```python
   @pytest.mark.physics
   @pytest.mark.slow  # If test takes >30 seconds
   def test_heavy_computation():
       pass
   ```

### Test File Organization

- **Unit tests**: `tests/unit/test_[module]_units.py`
- **Integration tests**: `tests/integration/test_[feature].py`
- **Performance tests**: `tests/benchmarks/test_[performance_aspect].py`

## Test Data and Fixtures

### Standard Test Configurations

```python
# Quick validation (development)
QUICK_CONFIG = TestConfiguration(
    particle_count=2, total_steps=10, step_size=1e-4
)

# Standard testing (CI)
STANDARD_CONFIG = TestConfiguration(
    particle_count=10, total_steps=25, step_size=1e-5
)

# Comprehensive validation (release)
COMPREHENSIVE_CONFIG = TestConfiguration(
    particle_count=50, total_steps=100, step_size=5e-6
)
```

## Documentation and Reporting

### Test Reports

The test runner generates comprehensive reports:

```bash
python run_tests.py all
# Generates:
# - Pass/fail summary
# - Performance benchmarks
# - Physics validation results
# - Memory usage analysis
```

### Integration with Documentation

Test examples are integrated into user documentation:

- Example notebooks demonstrate test scenarios
- API documentation includes test-validated examples
- User manual references test configurations for different physics scenarios

## References and Standards

- **IEEE Standards**: Computational physics simulation validation
- **Accelerator Physics**: Standard particle tracking benchmarks
- **Numerical Methods**: Conservation law testing in electromagnetic simulations
- **Software Engineering**: pytest best practices for scientific computing

---

*Last updated: 2025-09-18*
*For questions or issues, see the developer guide or file an issue in the project repository.*
