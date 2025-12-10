"""Basic standalone tests for optimization module (no pytest required).

Run with: python optimization/test_basic.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from optimization.metrics import (
    compute_energy_at_position,
    compute_max_energy_gain,
    compute_relative_energy_gain,
    detect_transverse_deflection,
)
from optimization.parameter_sweep import ParameterGrid, create_energy_aperture_grid


def create_mock_trajectory(n_steps=10, gamma_values=None):
    """Create a mock trajectory for testing."""
    if gamma_values is None:
        gamma_values = np.linspace(1000, 1010, n_steps)

    trajectory = []
    for i, gamma in enumerate(gamma_values):
        state = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([i * 10.0]),
            "t": np.array([i * 0.01]),
            "Px": np.array([0.0]),
            "Py": np.array([0.0]),
            "Pz": np.array([gamma * 0.99]),
            "Pt": np.array([gamma]),
            "gamma": np.array([gamma]),
            "bx": np.array([0.0]),
            "by": np.array([0.0]),
            "bz": np.array([0.99]),
            "m": np.array([1.0]),
            "charge": np.array([-1.0]),
        }
        trajectory.append(state)

    return trajectory


def test_compute_max_energy_gain():
    """Test maximum energy gain computation."""
    print("Testing compute_max_energy_gain...")

    trajectory = create_mock_trajectory(
        n_steps=5, gamma_values=[1000, 1005, 1010, 1008, 1012]
    )
    initial_gamma = 1000.0
    rest_energy_mev = 0.511

    max_gain = compute_max_energy_gain(trajectory, initial_gamma, rest_energy_mev)

    # Max gamma is 1012, so delta = 12
    expected = 12 * 0.511 * 1e-3
    assert abs(max_gain - expected) < 1e-6, f"Expected {expected}, got {max_gain}"
    print("  ✓ Passed")


def test_compute_relative_energy_gain():
    """Test relative energy gain computation."""
    print("Testing compute_relative_energy_gain...")

    trajectory = create_mock_trajectory(n_steps=3, gamma_values=[1000, 1010, 1020])
    initial_gamma = 1000.0

    relative_gain = compute_relative_energy_gain(trajectory, initial_gamma)

    expected = 0.02  # (1020 - 1000) / 1000
    assert abs(relative_gain - expected) < 1e-9, (
        f"Expected {expected}, got {relative_gain}"
    )
    print("  ✓ Passed")


def test_detect_transverse_deflection():
    """Test transverse deflection detection."""
    print("Testing detect_transverse_deflection...")

    gamma_values = [1000, 1000, 1150, 1050, 1050]
    trajectory = create_mock_trajectory(n_steps=5, gamma_values=gamma_values)

    events = detect_transverse_deflection(
        trajectory, energy_jump_threshold=0.1, energy_dip_threshold=0.05
    )

    event_types = [event[1] for event in events]
    assert "jump" in event_types, "Should detect jump"
    assert "dip" in event_types, "Should detect dip"
    assert "deflection" in event_types, "Should detect deflection"
    print("  ✓ Passed")


def test_compute_energy_at_position():
    """Test energy computation at specific position."""
    print("Testing compute_energy_at_position...")

    trajectory = create_mock_trajectory(
        n_steps=5, gamma_values=[1000, 1005, 1010, 1015, 1020]
    )
    initial_gamma = 1000.0
    rest_energy_mev = 0.511

    # Step 2 is at z=20mm with gamma=1010
    energy = compute_energy_at_position(
        trajectory,
        target_z=20.0,
        initial_gamma=initial_gamma,
        rest_energy_mev=rest_energy_mev,
        tolerance_mm=1.0,
    )

    expected = 10 * 0.511 * 1e-3
    assert energy is not None, "Should find energy at position"
    assert abs(energy - expected) < 1e-6, f"Expected {expected}, got {energy}"
    print("  ✓ Passed")


def test_parameter_grid_creation():
    """Test creating a parameter grid."""
    print("Testing ParameterGrid creation...")

    params = {"aperture": [0.1, 0.2, 0.3], "energy": [1.0, 10.0]}
    grid = ParameterGrid(params)

    assert len(grid) == 6, f"Expected 6 combinations, got {len(grid)}"
    assert grid.get_grid_shape() == (3, 2), (
        f"Expected shape (3, 2), got {grid.get_grid_shape()}"
    )
    print("  ✓ Passed")


def test_parameter_grid_iteration():
    """Test iterating over parameter grid."""
    print("Testing ParameterGrid iteration...")

    params = {"a": [1, 2], "b": [10, 20]}
    grid = ParameterGrid(params)
    configs = list(grid)

    assert len(configs) == 4, f"Expected 4 configs, got {len(configs)}"
    assert {"a": 1, "b": 10} in configs, "Missing config {a:1, b:10}"
    assert {"a": 2, "b": 20} in configs, "Missing config {a:2, b:20}"
    print("  ✓ Passed")


def test_create_energy_aperture_grid():
    """Test creating standard energy-aperture grid."""
    print("Testing create_energy_aperture_grid...")

    apertures = [0.01, 0.1, 1.0]
    energies = [1.0, 10.0, 100.0]

    grid = create_energy_aperture_grid(
        aperture_sizes_mm=apertures, energies_gev=energies
    )

    assert len(grid) == 9, f"Expected 9 combinations, got {len(grid)}"
    assert grid.param_names == ["aperture_radius", "initial_energy_gev"], (
        f"Unexpected param names: {grid.param_names}"
    )
    print("  ✓ Passed")


def test_create_energy_aperture_grid_defaults():
    """Test creating grid with default parameters."""
    print("Testing create_energy_aperture_grid with defaults...")

    grid = create_energy_aperture_grid()

    # Default should have 20 points per dimension
    assert len(grid) == 400, f"Expected 400 combinations, got {len(grid)}"
    assert grid.param_names == ["aperture_radius", "initial_energy_gev"], (
        f"Unexpected param names: {grid.param_names}"
    )
    print("  ✓ Passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Optimization Module Tests")
    print("=" * 60 + "\n")

    tests = [
        test_compute_max_energy_gain,
        test_compute_relative_energy_gain,
        test_detect_transverse_deflection,
        test_compute_energy_at_position,
        test_parameter_grid_creation,
        test_parameter_grid_iteration,
        test_create_energy_aperture_grid,
        test_create_energy_aperture_grid_defaults,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ Failed: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
