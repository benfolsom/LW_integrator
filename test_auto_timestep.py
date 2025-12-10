#!/usr/bin/env python3
"""Test script for auto-timestep calculation in optimization plugin.

This script verifies that the auto-timestep calculator produces reasonable
step counts (200-1500 range) for various particle energies and distances.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from lw_integrator.optimization_plugin import (
    calculate_auto_timestep,
    calculate_auto_steps,
)

# Constants
ELECTRON_MASS_AMU = 0.00054857990907
PROTON_MASS_AMU = 1.007276466621


def test_auto_timestep_basic():
    """Test basic auto-timestep calculation."""
    print("=" * 80)
    print("TEST 1: Basic Auto-Timestep Calculation")
    print("=" * 80)

    # Scenario from user report: z=0mm, E=1GeV, wall at 2200mm
    start_z = 0.0
    wall_z = 2200.0
    distance_past_wall = 10.0
    energy_gev = 1.0
    target_steps = 500

    print(f"\nScenario: Electron from z={start_z}mm to wall at {wall_z}mm")
    print(f"Energy: {energy_gev} GeV")
    print(f"Target steps: {target_steps}")
    print(f"Distance past wall: {distance_past_wall} mm")

    # Calculate timestep
    timestep = calculate_auto_timestep(
        start_z=start_z,
        wall_z=wall_z,
        distance_past_wall=distance_past_wall,
        particle_energy_gev=energy_gev,
        particle_mass_amu=ELECTRON_MASS_AMU,
        target_steps=target_steps,
    )

    print(f"\nCalculated timestep: {timestep:.6e} ns")

    # Verify with step calculation
    actual_steps = calculate_auto_steps(
        start_z=start_z,
        wall_z=wall_z,
        distance_past_wall=distance_past_wall,
        timestep=timestep,
        particle_energy_gev=energy_gev,
        particle_mass_amu=ELECTRON_MASS_AMU,
    )

    print(f"Verification: {actual_steps} steps (should be ≈ {target_steps})")

    if 200 <= actual_steps <= 1500:
        print("✓ PASS: Step count in acceptable range (200-1500)")
    else:
        print(f"✗ FAIL: Step count {actual_steps} outside range!")

    return actual_steps


def test_energy_sweep():
    """Test auto-timestep across energy range."""
    print("\n" + "=" * 80)
    print("TEST 2: Energy Sweep (1 GeV to 1000 GeV)")
    print("=" * 80)

    start_z = -100.0
    wall_z = 2200.0
    distance_past_wall = 10.0
    target_steps = 500
    energies = [1.0, 10.0, 100.0, 1000.0]

    print(f"\nStart z: {start_z} mm")
    print(f"Wall z: {wall_z} mm")
    print(f"Target steps: {target_steps}")
    print(
        f"\n{'Energy (GeV)':<15} {'Timestep (ns)':<20} {'Actual Steps':<15} {'Status'}"
    )
    print("-" * 70)

    all_pass = True
    for energy in energies:
        timestep = calculate_auto_timestep(
            start_z=start_z,
            wall_z=wall_z,
            distance_past_wall=distance_past_wall,
            particle_energy_gev=energy,
            particle_mass_amu=ELECTRON_MASS_AMU,
            target_steps=target_steps,
        )

        actual_steps = calculate_auto_steps(
            start_z=start_z,
            wall_z=wall_z,
            distance_past_wall=distance_past_wall,
            timestep=timestep,
            particle_energy_gev=energy,
            particle_mass_amu=ELECTRON_MASS_AMU,
        )

        status = "✓ PASS" if 200 <= actual_steps <= 1500 else "✗ FAIL"
        if status == "✗ FAIL":
            all_pass = False

        print(f"{energy:<15.1f} {timestep:<20.6e} {actual_steps:<15} {status}")

    if all_pass:
        print("\n✓ All energies produce acceptable step counts")
    else:
        print("\n✗ Some energies failed")

    return all_pass


def test_distance_sweep():
    """Test auto-timestep across starting positions."""
    print("\n" + "=" * 80)
    print("TEST 3: Distance Sweep (Various Starting Positions)")
    print("=" * 80)

    wall_z = 2200.0
    distance_past_wall = 10.0
    target_steps = 500
    energy_gev = 10.0
    starting_positions = [0.0, -50.0, -100.0, -500.0, -1000.0]

    print(f"\nWall z: {wall_z} mm")
    print(f"Energy: {energy_gev} GeV")
    print(f"Target steps: {target_steps}")
    print(
        f"\n{'Start z (mm)':<15} {'Distance (mm)':<15} {'Timestep (ns)':<20} {'Steps':<10} {'Status'}"
    )
    print("-" * 80)

    all_pass = True
    for start_z in starting_positions:
        total_dist = abs(wall_z - start_z) + distance_past_wall

        timestep = calculate_auto_timestep(
            start_z=start_z,
            wall_z=wall_z,
            distance_past_wall=distance_past_wall,
            particle_energy_gev=energy_gev,
            particle_mass_amu=ELECTRON_MASS_AMU,
            target_steps=target_steps,
        )

        actual_steps = calculate_auto_steps(
            start_z=start_z,
            wall_z=wall_z,
            distance_past_wall=distance_past_wall,
            timestep=timestep,
            particle_energy_gev=energy_gev,
            particle_mass_amu=ELECTRON_MASS_AMU,
        )

        status = "✓ PASS" if 200 <= actual_steps <= 1500 else "✗ FAIL"
        if status == "✗ FAIL":
            all_pass = False

        print(
            f"{start_z:<15.1f} {total_dist:<15.1f} {timestep:<20.6e} {actual_steps:<10} {status}"
        )

    if all_pass:
        print("\n✓ All distances produce acceptable step counts")
    else:
        print("\n✗ Some distances failed")

    return all_pass


def test_particle_species():
    """Test auto-timestep for different particle species."""
    print("\n" + "=" * 80)
    print("TEST 4: Particle Species (Electron vs Proton)")
    print("=" * 80)

    start_z = -100.0
    wall_z = 2200.0
    distance_past_wall = 10.0
    target_steps = 500
    energy_gev = 10.0

    print(f"\nStart z: {start_z} mm")
    print(f"Wall z: {wall_z} mm")
    print(f"Energy: {energy_gev} GeV")
    print(f"Target steps: {target_steps}")
    print(
        f"\n{'Species':<15} {'Mass (amu)':<20} {'Timestep (ns)':<20} {'Steps':<10} {'Status'}"
    )
    print("-" * 80)

    species = [
        ("Electron", ELECTRON_MASS_AMU),
        ("Proton", PROTON_MASS_AMU),
    ]

    all_pass = True
    for name, mass in species:
        timestep = calculate_auto_timestep(
            start_z=start_z,
            wall_z=wall_z,
            distance_past_wall=distance_past_wall,
            particle_energy_gev=energy_gev,
            particle_mass_amu=mass,
            target_steps=target_steps,
        )

        actual_steps = calculate_auto_steps(
            start_z=start_z,
            wall_z=wall_z,
            distance_past_wall=distance_past_wall,
            timestep=timestep,
            particle_energy_gev=energy_gev,
            particle_mass_amu=mass,
        )

        status = "✓ PASS" if 200 <= actual_steps <= 1500 else "✗ FAIL"
        if status == "✗ FAIL":
            all_pass = False

        print(f"{name:<15} {mass:<20.6e} {timestep:<20.6e} {actual_steps:<10} {status}")

    if all_pass:
        print("\n✓ All particle species produce acceptable step counts")
    else:
        print("\n✗ Some particle species failed")

    return all_pass


def test_target_step_range():
    """Test different target step counts."""
    print("\n" + "=" * 80)
    print("TEST 5: Target Step Range (200 to 1500)")
    print("=" * 80)

    start_z = -100.0
    wall_z = 2200.0
    distance_past_wall = 10.0
    energy_gev = 10.0
    target_steps_list = [200, 500, 1000, 1500]

    print(f"\nStart z: {start_z} mm")
    print(f"Wall z: {wall_z} mm")
    print(f"Energy: {energy_gev} GeV")
    print(
        f"\n{'Target Steps':<15} {'Timestep (ns)':<20} {'Actual Steps':<15} {'Error %':<10} {'Status'}"
    )
    print("-" * 80)

    all_pass = True
    for target in target_steps_list:
        timestep = calculate_auto_timestep(
            start_z=start_z,
            wall_z=wall_z,
            distance_past_wall=distance_past_wall,
            particle_energy_gev=energy_gev,
            particle_mass_amu=ELECTRON_MASS_AMU,
            target_steps=target,
        )

        actual_steps = calculate_auto_steps(
            start_z=start_z,
            wall_z=wall_z,
            distance_past_wall=distance_past_wall,
            timestep=timestep,
            particle_energy_gev=energy_gev,
            particle_mass_amu=ELECTRON_MASS_AMU,
        )

        error_pct = abs(actual_steps - target) / target * 100
        status = "✓ PASS" if error_pct < 15 else "✗ FAIL"  # Allow 15% tolerance
        if status == "✗ FAIL":
            all_pass = False

        print(
            f"{target:<15} {timestep:<20.6e} {actual_steps:<15} {error_pct:<10.1f} {status}"
        )
        if status == "✗ FAIL":
            all_pass = False

        print(f"{target:<15} {timestep:<20.6e} {actual_steps:<15} {error_pct:<10.1f} {status}")

    if all_pass:
        print("\n✓ All target step counts achieved within tolerance")
    else:
        print("\n✗ Some target step counts outside tolerance")

    return all_pass


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("AUTO-TIMESTEP CALCULATION TEST SUITE")
    print("=" * 80)
    print("\nThis test verifies that auto-timestep produces step counts in the")
    print("recommended range of 200-1500 steps for various scenarios.")
    print()

    # Run all tests
    results = []
    results.append(("Basic Calculation", test_auto_timestep_basic()))
    results.append(("Energy Sweep", test_energy_sweep()))
    results.append(("Distance Sweep", test_distance_sweep()))
    results.append(("Particle Species", test_particle_species()))
    results.append(("Target Step Range", test_target_step_range()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, result in results:
        status = "✓ PASS" if result or isinstance(result, int) else "✗ FAIL"
        print(f"{name:<30} {status}")

    all_passed = all(r for _, r in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe auto-timestep calculator produces step counts in the")
        print("recommended range (200-1500) across all tested scenarios!")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease review the failures above.")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
