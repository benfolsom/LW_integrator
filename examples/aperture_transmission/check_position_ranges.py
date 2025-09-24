#!/usr/bin/env python3
"""
Quick check of position ranges in the corrected simulation.
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


def check_position_ranges():
    """Check that position data is reasonable."""

    integrator = LienardWiechertIntegrator()

    config = TestConfiguration(
        particle_count=3,
        transverse_separation=2.0,
        starting_distance=-50.0,
        step_size=1e-5,
        total_steps=100,  # Short simulation
        sim_type=0,
        wall_z=0.0,
        aperture_r=5.0,
        z_cutoff=50.0,
    )

    # Create bunch with correct energy setup
    bunch = create_bunch_uniform_distribution(config, ELECTRON, "line")
    energy_gev = 5.0
    target_pt = energy_gev / C_MMNS
    bunch["Pt"] = np.full(3, target_pt)
    bunch["Pz"] = bunch["Pt"] * 0.99999
    bunch["Px"] = np.zeros(3)
    bunch["Py"] = np.zeros(3)
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

    # Run simulation
    trajectory_rider, trajectory_driver = integrator.integrate_retarded_fields(
        static_steps=20,
        ret_steps=80,
        h_step=1e-5,
        wall_Z=0.0,
        apt_R=5.0,
        sim_type=0,
        init_rider=bunch,
        init_driver=driver_bunch,
        bunch_dist=1e5,
        z_cutoff=50.0,
    )

    print("Position analysis:")
    print(f"  Total steps: {len(trajectory_rider)}")

    if len(trajectory_rider) > 0:
        # Extract all positions
        all_z = []
        all_energies = []

        for step in trajectory_rider:
            if "z" in step and "Pt" in step and len(step["z"]) > 0:
                z_mm = step["z"][0]
                pt = step["Pt"][0]
                energy = pt * C_MMNS

                all_z.append(z_mm)
                all_energies.append(energy)

        if all_z:
            print(f"  Z range: {min(all_z):.3f} to {max(all_z):.3f} mm")
            print("  Expected: -50 to +50 mm (100 mm total)")
            print(
                f"  Energy range: {min(all_energies):.6f} to {max(all_energies):.6f} GeV"
            )
            print(f"  Energy change: {max(all_energies) - min(all_energies):.6f} GeV")

            # Check if position is reasonable
            z_range = max(all_z) - min(all_z)
            if z_range < 200:  # Less than 200mm seems reasonable
                print(f"  ✅ Position range looks reasonable: {z_range:.1f} mm")
            else:
                print(f"  ❌ Position range seems too large: {z_range:.1f} mm")


if __name__ == "__main__":
    check_position_ranges()
