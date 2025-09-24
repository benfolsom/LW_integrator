#!/usr/bin/env python3
"""
Quick test to verify position ranges with legacy-matched approach.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.trajectory_integrator import LienardWiechertIntegrator
from physics.constants import C_MMNS
from tests.test_config import (
    ELECTRON,
    create_bunch_uniform_distribution,
    TestConfiguration,
)


def test_legacy_matched_position_ranges():
    """Test position ranges with legacy-matched setup."""

    integrator = LienardWiechertIntegrator()

    config = TestConfiguration(
        particle_count=3,
        transverse_separation=2.0,
        starting_distance=-50.0,
        step_size=2e-6,  # Legacy step size
        total_steps=50,  # Short test - 0.1 ns total
        sim_type=0,
        wall_z=0.0,
        aperture_r=5.0,
        z_cutoff=50.0,
    )

    # Create bunch with legacy momentum formulation
    bunch = create_bunch_uniform_distribution(config, ELECTRON, "line")

    # Set up 5 GeV electron using legacy approach
    energy_gev = 5.0
    rest_energy_mev = 0.511
    total_energy_mev = energy_gev * 1000.0
    gamma = total_energy_mev / rest_energy_mev
    mass_amu = rest_energy_mev / 931.494

    # Calculate momentum
    momentum_magnitude = (total_energy_mev / 931.494) * C_MMNS
    Pz_value = momentum_magnitude * 0.99999
    Pt_value = np.sqrt(Pz_value**2 + mass_amu**2 * C_MMNS**2)
    gamma_calc = Pt_value / (mass_amu * C_MMNS)
    bz = Pz_value / (gamma_calc * mass_amu * C_MMNS)

    bunch["Pt"] = np.full(3, Pt_value)
    bunch["Pz"] = np.full(3, Pz_value)
    bunch["Px"] = np.zeros(3)
    bunch["Py"] = np.zeros(3)
    bunch["gamma"] = np.full(3, gamma_calc)
    bunch["bz"] = np.full(3, bz)
    bunch["bx"] = np.zeros(3)
    bunch["by"] = np.zeros(3)
    bunch["x"] = np.array([-1.0, 0.0, 1.0])
    bunch["y"] = np.zeros(3)
    bunch["z"] = np.full(3, -50.0)

    # Create driver
    driver_bunch = bunch.copy()
    for key in driver_bunch:
        if hasattr(driver_bunch[key], "__len__") and len(driver_bunch[key]) > 1:
            driver_bunch[key] = driver_bunch[key][:1]
    driver_bunch["x"] = np.array([1000.0])
    driver_bunch["y"] = np.array([1000.0])
    driver_bunch["z"] = np.array([-1000.0])

    print("Legacy-matched test:")
    print(f"  Initial energy: {energy_gev} GeV")
    print(f"  Initial gamma: {gamma_calc:.1f}")
    print(f"  Beta z: {bz:.6f}")
    print(f"  Step size: {config.step_size} ns")
    print(f"  Total steps: {config.total_steps}")
    print(f"  Total time: {config.total_steps * config.step_size} ns")
    print(
        f"  Expected travel distance: {bz * C_MMNS * config.total_steps * config.step_size:.1f} mm"
    )

    # Run simulation
    trajectory_rider, trajectory_driver = integrator.integrate_retarded_fields(
        static_steps=1,
        ret_steps=config.total_steps - 1,
        h_step=config.step_size,
        wall_Z=0.0,
        apt_R=5.0,
        sim_type=0,
        init_rider=bunch,
        init_driver=driver_bunch,
        bunch_dist=1e5,
        z_cutoff=50.0,
    )

    print("\nTrajectory analysis:")
    print(f"  Steps completed: {len(trajectory_rider)}")

    if len(trajectory_rider) > 0:
        all_z = []
        all_energies = []

        for step in trajectory_rider:
            if "z" in step and "gamma" in step and len(step["z"]) > 0:
                z_mm = step["z"][0]
                gamma = step["gamma"][0]
                energy_mev = gamma * rest_energy_mev
                energy_gev = energy_mev / 1000.0

                all_z.append(z_mm)
                all_energies.append(energy_gev)

        if all_z:
            print(f"  Z range: {min(all_z):.3f} to {max(all_z):.3f} mm")
            print(f"  Travel distance: {max(all_z) - min(all_z):.3f} mm")
            print(
                f"  Energy range: {min(all_energies):.6f} to {max(all_energies):.6f} GeV"
            )
            print(f"  Energy change: {max(all_energies) - min(all_energies):.6f} GeV")

            # Check reasonableness
            travel_distance = max(all_z) - min(all_z)
            if travel_distance < 200:  # Less than 200mm
                print(f"  ✅ Position range looks reasonable: {travel_distance:.1f} mm")
            else:
                print(f"  ❌ Position range still too large: {travel_distance:.1f} mm")


if __name__ == "__main__":
    test_legacy_matched_position_ranges()
