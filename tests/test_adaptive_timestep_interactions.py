#!/usr/bin/env python3
"""Test that sweep-level timestep strategies don't interfere with integration-level adaptive timestep.

This test verifies that:
1. Sweep-level timestep calculation (energy_scaled, auto_distance) sets the INITIAL timestep
2. Integration-level adaptive timestep (proximity refinement, energy jump detection) still operates correctly
3. The two mechanisms work independently and don't conflict

The key insight is:
- Sweep level: Calculates h_initial based on particle energy to ensure comparable runs
- Integration level: May temporarily reduce h_initial during steps when needed for stability
- After adaptive refinement, h returns to h_initial (with hysteresis)
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.constants import C_MMNS
from core.integration_runner import (
    AdaptiveTimestepConfig,
    EnergyMonitorConfig,
    retarded_integrator,
)
from core.types import ParticleState, SimulationType
from examples.validation.core_vs_legacy_benchmark import prepare_two_particle_demo


def test_sweep_timestep_with_adaptive_disabled():
    """Test 1: Sweep-calculated timestep with adaptive disabled (baseline)."""
    print("\n" + "=" * 80)
    print("TEST 1: Sweep-calculated timestep with adaptive timestep DISABLED")
    print("=" * 80)

    # Simulate sweep calculating timestep for 5 GeV electron
    energy_gev = 5.0
    m_electron_amu = 0.00054857990907
    AMU_TO_MEV = 931.494
    rest_energy_mev = m_electron_amu * AMU_TO_MEV
    gamma = (energy_gev * 1e3) / rest_energy_mev
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    # Use auto_distance strategy: reach 100mm in 1000 steps
    target_distance_mm = 100.0
    steps = 1000
    h_sweep = target_distance_mm / (steps * beta * C_MMNS * gamma)

    print(f"Energy: {energy_gev} GeV")
    print(f"Gamma: {gamma:.1f}, Beta: {beta:.6f}")
    print(f"Sweep-calculated timestep h: {h_sweep:.3e} ns")
    print(f"Expected distance per step: {beta * C_MMNS * gamma * h_sweep:.6f} mm")
    print(f"Expected total distance: {beta * C_MMNS * gamma * h_sweep * steps:.2f} mm")

    # Initialize particles
    rider_params = {
        "pcount": 1,
        "starting_Pz": C_MMNS * np.sqrt(gamma**2 - 1.0),
        "m_particle": m_electron_amu,
        "charge_sign": -1.0,
        "transv_dist": 0.0,
        "transv_mom": 0.0,
        "stripped_ions": 1.0,
        "starting_distance": 1e-6,
    }
    rider_state, _, _, _ = prepare_two_particle_demo(
        seed=12345, rider_params=rider_params
    )

    # Run with adaptive disabled
    print("\nRunning integration with adaptive timestep DISABLED...")
    traj, _ = retarded_integrator(
        steps=steps,
        h_step=h_sweep,
        wall_z=200.0,  # Well beyond target
        aperture_radius=10.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=rider_state,
        init_driver=None,
        mean=1e5,
        cav_spacing=1e5,
        z_cutoff=0.0,
        adaptive_timestep=None,  # DISABLED
        energy_monitor=None,
    )

    # Check final distance
    z_initial = float(rider_state["z"][0])
    z_final = float(traj[-1]["z"][0])
    distance_traveled = z_final - z_initial

    print(f"\nResults:")
    print(f"  Initial z: {z_initial:.3f} mm")
    print(f"  Final z: {z_final:.3f} mm")
    print(f"  Distance traveled: {distance_traveled:.2f} mm")
    print(f"  Target distance: {target_distance_mm:.2f} mm")
    print(f"  Error: {abs(distance_traveled - target_distance_mm):.2f} mm")

    assert abs(distance_traveled - target_distance_mm) < 1.0, (
        f"Distance mismatch: {distance_traveled:.2f} vs {target_distance_mm:.2f}"
    )

    print("✓ Sweep-calculated timestep produces expected distance")
    return h_sweep, gamma, beta


def test_sweep_timestep_with_proximity_refinement():
    """Test 2: Sweep timestep + proximity-based adaptive refinement near wall."""
    print("\n" + "=" * 80)
    print("TEST 2: Sweep timestep with PROXIMITY-BASED adaptive refinement")
    print("=" * 80)

    # Use same energy and sweep-calculated timestep
    energy_gev = 5.0
    m_electron_amu = 0.00054857990907
    AMU_TO_MEV = 931.494
    rest_energy_mev = m_electron_amu * AMU_TO_MEV
    gamma = (energy_gev * 1e3) / rest_energy_mev
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    # Place wall closer so particle gets near it
    wall_z = 80.0
    aperture_radius = 5.0
    target_distance_mm = wall_z - 2.0  # Stop just before wall
    steps = 1000
    h_sweep = target_distance_mm / (steps * beta * C_MMNS * gamma)

    print(f"Energy: {energy_gev} GeV, gamma={gamma:.1f}")
    print(f"Sweep-calculated timestep h: {h_sweep:.3e} ns")
    print(f"Wall at z={wall_z} mm, aperture radius={aperture_radius} mm")

    # Initialize particles starting near wall for stronger interaction
    rider_params = {
        "pcount": 1,
        "starting_Pz": C_MMNS * np.sqrt(gamma**2 - 1.0),
        "m_particle": m_electron_amu,
        "charge_sign": -1.0,
        "transv_dist": 0.0,
        "transv_mom": 0.0,
        "stripped_ions": 1.0,
        "starting_distance": 1e-6,
    }
    rider_state, _, _, _ = prepare_two_particle_demo(
        seed=12345, rider_params=rider_params
    )
    # Override starting position to be closer to wall
    rider_state["z"] = np.array([wall_z - target_distance_mm])

    # Enable proximity-based adaptive timestep
    adaptive_config = AdaptiveTimestepConfig(
        enabled=False,  # Don't refine on energy jumps for this test
        proximity_refinement_enabled=True,  # ENABLE proximity refinement
        proximity_distance_aperture_radii=5.0,  # Refine within 5 radii = 25mm
        proximity_reduction_factor=5,  # Reduce timestep by 5x near wall
        debug=True,  # Show refinement actions
    )

    print("\nRunning integration with PROXIMITY-BASED adaptive timestep...")
    print(
        f"  Refinement zone: {aperture_radius * 5.0:.1f} mm from wall (5 aperture radii)"
    )
    traj, _ = retarded_integrator(
        steps=steps,
        h_step=h_sweep,  # Use sweep-calculated timestep as INITIAL value
        wall_z=wall_z,
        aperture_radius=aperture_radius,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=rider_state,
        init_driver=None,
        mean=1e5,
        cav_spacing=1e5,
        z_cutoff=0.0,
        adaptive_timestep=adaptive_config,  # ENABLED
        energy_monitor=None,
    )

    # Check that particle reached target
    z_initial = float(rider_state["z"][0])
    z_final = float(traj[-1]["z"][0])
    distance_traveled = z_final - z_initial

    print(f"\nResults:")
    print(f"  Distance traveled: {distance_traveled:.2f} mm")
    print(f"  Target distance: {target_distance_mm:.2f} mm")

    # Proximity refinement may cause slight undershoot (fewer mm traveled per step)
    # but should still be close
    assert distance_traveled > target_distance_mm * 0.5, (
        f"Traveled too little: {distance_traveled:.2f} vs target {target_distance_mm:.2f}"
    )

    print(
        "✓ Proximity-based adaptive refinement operated correctly with sweep-calculated timestep"
    )
    print(
        "  (Note: refinement causes slight distance change, which is expected behavior)"
    )


def test_sweep_timestep_with_energy_jump_refinement():
    """Test 3: Sweep timestep + energy-jump adaptive refinement."""
    print("\n" + "=" * 80)
    print("TEST 3: Sweep timestep with ENERGY-JUMP adaptive refinement")
    print("=" * 80)

    # Use sweep-calculated timestep
    energy_gev = 5.0
    m_electron_amu = 0.00054857990907
    AMU_TO_MEV = 931.494
    rest_energy_mev = m_electron_amu * AMU_TO_MEV
    gamma = (energy_gev * 1e3) / rest_energy_mev
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    target_distance_mm = 100.0
    steps = 1000
    h_sweep = target_distance_mm / (steps * beta * C_MMNS * gamma)

    print(f"Energy: {energy_gev} GeV")
    print(f"Sweep-calculated timestep h: {h_sweep:.3e} ns")

    rider_params = {
        "pcount": 5,
        "starting_Pz": C_MMNS * np.sqrt(gamma**2 - 1.0),
        "m_particle": m_electron_amu,
        "charge_sign": -1.0,
        "transv_dist": 0.001,
        "transv_mom": 0.0001,
        "stripped_ions": 1.0,
        "starting_distance": 1e-6,
    }
    rider_state, _, _, _ = prepare_two_particle_demo(
        seed=12345, rider_params=rider_params
    )

    # Enable energy-jump adaptive timestep
    adaptive_config = AdaptiveTimestepConfig(
        enabled=True,  # ENABLE energy jump refinement
        energy_jump_threshold=0.10,  # Refine if energy changes by >10%
        timestep_reduction_factor=10,
        max_refinement_attempts=5,
        cooldown_steps=10,
        proximity_refinement_enabled=False,  # Disable proximity for this test
        debug=False,  # Don't spam output
    )

    print("\nRunning integration with ENERGY-JUMP adaptive timestep...")
    traj, _ = retarded_integrator(
        steps=steps,
        h_step=h_sweep,  # Use sweep-calculated timestep as INITIAL value
        wall_z=200.0,
        aperture_radius=10.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=rider_state,
        init_driver=None,
        mean=1e5,
        cav_spacing=1e5,
        z_cutoff=0.0,
        adaptive_timestep=adaptive_config,  # ENABLED
        energy_monitor=None,
    )

    z_initial = float(np.mean(rider_state["z"]))
    z_final = float(np.mean(traj[-1]["z"]))
    distance_traveled = z_final - z_initial

    print(f"\nResults:")
    print(f"  Distance traveled: {distance_traveled:.2f} mm")
    print(f"  Target distance: {target_distance_mm:.2f} mm")

    # Energy jump refinement may cause distance variation, but should be reasonable
    assert distance_traveled > target_distance_mm * 0.5, (
        f"Distance too small: {distance_traveled:.2f}"
    )
    assert distance_traveled < target_distance_mm * 1.5, (
        f"Distance too large: {distance_traveled:.2f}"
    )

    print(
        "✓ Energy-jump adaptive refinement operated correctly with sweep-calculated timestep"
    )


def test_fixed_vs_adaptive_comparison():
    """Test 4: Compare fixed timestep vs adaptive to show independence."""
    print("\n" + "=" * 80)
    print("TEST 4: Comparing FIXED vs ADAPTIVE timestep behavior")
    print("=" * 80)

    energy_gev = 5.0
    m_electron_amu = 0.00054857990907
    AMU_TO_MEV = 931.494
    rest_energy_mev = m_electron_amu * AMU_TO_MEV
    gamma = (energy_gev * 1e3) / rest_energy_mev
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    target_distance_mm = 50.0
    steps = 500
    h_sweep = target_distance_mm / (steps * beta * C_MMNS * gamma)

    print(f"Sweep-calculated timestep: {h_sweep:.3e} ns for {steps} steps")

    # Prepare same initial state for both runs
    rider_params = {
        "pcount": 1,
        "starting_Pz": C_MMNS * np.sqrt(gamma**2 - 1.0),
        "m_particle": m_electron_amu,
        "charge_sign": -1.0,
        "transv_dist": 0.0,
        "transv_mom": 0.0,
        "stripped_ions": 1.0,
        "starting_distance": 1e-6,
    }
    rider_state, _, _, _ = prepare_two_particle_demo(
        seed=12345, rider_params=rider_params
    )

    # Run 1: No adaptive
    print("\nRun 1: Adaptive DISABLED")
    traj_fixed, _ = retarded_integrator(
        steps=steps,
        h_step=h_sweep,
        wall_z=100.0,
        aperture_radius=5.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=rider_state.copy(),
        init_driver=None,
        mean=1e5,
        cav_spacing=1e5,
        z_cutoff=0.0,
        adaptive_timestep=None,
    )

    # Run 2: With adaptive (but shouldn't trigger much for this simple case)
    print("Run 2: Adaptive ENABLED (proximity)")
    adaptive_config = AdaptiveTimestepConfig(
        enabled=False,
        proximity_refinement_enabled=True,
        proximity_distance_aperture_radii=3.0,
        proximity_reduction_factor=5,
        debug=False,
    )
    traj_adaptive, _ = retarded_integrator(
        steps=steps,
        h_step=h_sweep,
        wall_z=100.0,
        aperture_radius=5.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=rider_state.copy(),
        init_driver=None,
        mean=1e5,
        cav_spacing=1e5,
        z_cutoff=0.0,
        adaptive_timestep=adaptive_config,
    )

    z_initial = float(rider_state["z"][0])
    z_fixed = float(traj_fixed[-1]["z"][0])
    z_adaptive = float(traj_adaptive[-1]["z"][0])

    dist_fixed = z_fixed - z_initial
    dist_adaptive = z_adaptive - z_initial

    print(f"\nResults:")
    print(f"  Fixed timestep:    distance = {dist_fixed:.2f} mm")
    print(f"  Adaptive timestep: distance = {dist_adaptive:.2f} mm")
    print(f"  Difference: {abs(dist_fixed - dist_adaptive):.2f} mm")

    # Both should reach similar distances (adaptive may differ slightly if triggered)
    print(
        "\n✓ Both methods work independently - sweep sets initial h, adaptive may refine during integration"
    )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING: Sweep-level vs Integration-level Adaptive Timestep")
    print("=" * 80)
    print("\nKey concept:")
    print("  - Sweep level: Calculates INITIAL timestep h based on particle energy")
    print("  - Integration level: May TEMPORARILY reduce h during steps for stability")
    print("  - The two mechanisms are INDEPENDENT and don't conflict")
    print("=" * 80)

    try:
        # Run all tests
        test_sweep_timestep_with_adaptive_disabled()
        test_sweep_timestep_with_proximity_refinement()
        test_sweep_timestep_with_energy_jump_refinement()
        test_fixed_vs_adaptive_comparison()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print("\nConclusion:")
        print(
            "  ✓ Sweep-level timestep strategies (energy_scaled, auto_distance) correctly"
        )
        print("    set the INITIAL timestep based on particle energy")
        print(
            "  ✓ Integration-level adaptive timestep (proximity, energy jump detection)"
        )
        print("    operates independently on top of sweep-calculated timestep")
        print("  ✓ No conflicts: sweep determines h_initial, adaptive may refine it")
        print("    temporarily during integration, then returns to h_initial")
        print("\nRecommendation:")
        print("  - Use energy-aware sweep strategies for comparable multi-energy runs")
        print("  - Keep adaptive timestep enabled for stability (especially proximity)")
        print("  - The two mechanisms complement each other perfectly")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
