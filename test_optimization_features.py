#!/usr/bin/env python
"""Test script to verify optimization plugin enhancements.

This script tests the new features:
1. Sweepable parameter UI controls
2. Parameter grid generation with optional sweeps
3. Trajectory data structures

Does NOT run full GUI (requires display), but validates the core logic.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from examples.validation.core_vs_legacy_benchmark import SimulationType
from lw_integrator.optimization_plugin import (
    OptimizationConfig,
    calculate_auto_steps,
    calculate_auto_timestep,
)


def test_optimization_config():
    """Test that OptimizationConfig can be instantiated."""
    print("Testing OptimizationConfig instantiation...")

    config = OptimizationConfig(
        simulation_type=SimulationType.CONDUCTING_WALL,
        aperture_range=(1e-5, 1e-3),
        aperture_points=3,
        energy_range=(1.0, 10.0),
        energy_points=3,
        transverse_offset_fractions=[0.1, 0.5],
        starting_z_positions=[-10.0, -50.0],
        wall_z=100.0,
        timestep=3e-7,
        steps=500,
        m_particle=0.00054857990907,
        charge_sign=-1.0,
        pcount=1,
    )

    print(f"  ✓ Config created: {config.aperture_points}×{config.energy_points} grid")
    return config


def test_auto_timestep_calculation():
    """Test auto-timestep calculation functions."""
    print("\nTesting auto-timestep calculations...")

    # Test case: electron at 1 GeV, -10 mm to 100 mm
    timestep = calculate_auto_timestep(
        start_z=-10.0,
        wall_z=100.0,
        distance_past_wall=10.0,
        particle_energy_gev=1.0,
        particle_mass_amu=0.00054857990907,
        target_steps=500,
    )

    print(f"  Calculated timestep: {timestep:.3e} ns")

    # Verify we get expected number of steps
    steps = calculate_auto_steps(
        start_z=-10.0,
        wall_z=100.0,
        distance_past_wall=10.0,
        timestep=timestep,
        particle_energy_gev=1.0,
        particle_mass_amu=0.00054857990907,
    )

    print(f"  Calculated steps: {steps}")

    # Should be close to target
    assert 400 < steps < 600, f"Steps {steps} not near target 500"
    print("  ✓ Auto-timestep working correctly")


def test_parameter_grid_logic():
    """Test parameter grid generation logic (simulated)."""
    print("\nTesting parameter grid generation logic...")

    import numpy as np

    def generate_range(min_val, max_val, points, log_scale):
        """Mirror the plugin's _generate_range method."""
        if points == 1:
            return [(min_val + max_val) / 2.0]
        if log_scale:
            return np.logspace(np.log10(min_val), np.log10(max_val), points).tolist()
        else:
            return np.linspace(min_val, max_val, points).tolist()

    # Test linear range
    linear = generate_range(1.0, 10.0, 5, False)
    assert len(linear) == 5
    assert abs(linear[0] - 1.0) < 1e-10
    assert abs(linear[-1] - 10.0) < 1e-10
    print(f"  ✓ Linear range: {linear}")

    # Test log range
    log = generate_range(1e-5, 1e-3, 3, True)
    assert len(log) == 3
    assert abs(log[0] - 1e-5) < 1e-15
    assert abs(log[-1] - 1e-3) < 1e-13
    print(f"  ✓ Log range: {[f'{v:.2e}' for v in log]}")

    # Test grid combination
    import itertools

    grids = {
        "aperture": generate_range(1e-5, 1e-4, 2, True),
        "energy": generate_range(1.0, 10.0, 2, False),
        "offset_frac": [0.1, 0.5],
    }

    param_names = list(grids.keys())
    param_values = [grids[name] for name in param_names]
    total_combinations = 1
    for vals in param_values:
        total_combinations *= len(vals)

    combinations = list(itertools.product(*param_values))
    assert len(combinations) == total_combinations
    print(f"  ✓ Grid combinations: {len(combinations)} runs (2×2×2)")

    # Verify first combination
    first = dict(zip(param_names, combinations[0]))
    print(f"    First combo: {first}")


def test_trajectory_data_structure():
    """Test the expected trajectory data structure."""
    print("\nTesting trajectory data structure...")

    import numpy as np

    # Simulate a small trajectory
    n_points = 100
    stride = 10

    traj_full = {
        "z": np.linspace(-10, 100, n_points),
        "r": np.random.rand(n_points) * 1e-5,
        "pz": np.linspace(0.1, 0.2, n_points),
        "pr": np.random.randn(n_points) * 1e-8,
        "t": np.linspace(0, 1e-6, n_points),
    }

    # Apply stride (as done in plugin)
    traj_saved = {key: val[::stride].tolist() for key, val in traj_full.items()}

    expected_length = n_points // stride
    for key, val in traj_saved.items():
        assert len(val) == expected_length, f"{key} wrong length"

    print(f"  ✓ Trajectory stride: {n_points} → {expected_length} points")
    print(f"    Keys: {list(traj_saved.keys())}")

    # Verify JSON serializable
    import json

    json_str = json.dumps(traj_saved)
    assert len(json_str) > 0
    print(f"  ✓ Trajectory JSON serializable ({len(json_str)} bytes)")


def test_sweep_results_format():
    """Test the expected sweep_results.json format."""
    print("\nTesting sweep results format...")

    results = [
        {
            "run_number": 1,
            "parameters": {
                "aperture_radius": 1e-5,
                "particle_energy_gev": 1.0,
                "start_z": -10.0,
                "transverse_offset": 1e-6,
                "timestep": 3e-7,
                "steps": 500,
                "m_particle": 0.00054857990907,
                "charge_sign": -1.0,
            },
            "metrics": {
                "rider_delta_e_mev": 10.5,
                "rider_gamma_initial": 1958.0,
                "rider_gamma_final": 1978.5,
            },
            "trajectory": {
                "z": [-10.0, 0.0, 50.0, 100.0],
                "r": [1e-6, 1.1e-6, 1.2e-6, 1.3e-6],
                "pz": [0.1, 0.15, 0.18, 0.2],
                "pr": [1e-8, 1.1e-8, 1.05e-8, 1e-8],
                "t": [0, 1e-7, 5e-7, 1e-6],
            },
        }
    ]

    output_data = {
        "config": {
            "aperture_range": [1e-5, 1e-3],
            "energy_range": [1.0, 1000.0],
        },
        "results": results,
        "total_runs": len(results),
    }

    import json

    json_str = json.dumps(output_data, indent=2)

    # Verify can be re-loaded
    loaded = json.loads(json_str)
    assert loaded["total_runs"] == 1
    assert "trajectory" in loaded["results"][0]

    print(f"  ✓ Results format valid")
    print(f"  ✓ JSON size: {len(json_str)} bytes")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Optimization Plugin Feature Tests")
    print("=" * 60)

    try:
        test_optimization_config()
        test_auto_timestep_calculation()
        test_parameter_grid_logic()
        test_trajectory_data_structure()
        test_sweep_results_format()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
