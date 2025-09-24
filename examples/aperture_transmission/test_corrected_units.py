#!/usr/bin/env python3
"""
Quick test to verify the corrected position and energy data.
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


def test_corrected_units():
    """Test that the corrected units produce reasonable results."""

    integrator = LienardWiechertIntegrator()

    # Create simple test
    config = TestConfiguration(
        particle_count=3,
        transverse_separation=2.0,
        starting_distance=-50.0,
        step_size=1e-5,
        total_steps=100,
        sim_type=0,
        wall_z=0.0,
        aperture_r=5.0,
        z_cutoff=50.0,
    )

    # Create bunch using corrected setup
    bunch = create_bunch_uniform_distribution(config, ELECTRON, "line")

    # Set 5 GeV energy using corrected formula
    energy_gev = 5.0
    rest_energy_mev = 0.511  # electron rest mass
    energy_mev = energy_gev * 1000.0
    gamma = energy_mev / rest_energy_mev
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    mass_amu = 0.511 / 931.494  # electron mass in amu

    bunch["Pt"] = np.full(3, gamma * mass_amu * C_MMNS)
    bunch["Pz"] = bunch["Pt"] * beta
    bunch["Px"] = np.zeros(3)
    bunch["Py"] = np.zeros(3)
    bunch["gamma"] = np.full(3, gamma)
    bunch["bz"] = np.full(3, beta)
    bunch["bx"] = np.zeros(3)
    bunch["by"] = np.zeros(3)

    # Position within aperture
    bunch["x"] = np.array([-1.0, 0.0, 1.0])
    bunch["y"] = np.zeros(3)
    bunch["z"] = np.full(3, -50.0)

    print("Initial setup (corrected):")
    print(f"  Target energy: {energy_gev} GeV")
    print(f"  Gamma: {gamma:.1f}")
    print(f"  Beta: {beta:.6f}")
    print(f"  Pt (integrator): {bunch['Pt'][0]:.6f} amu*mm/ns")

    # Check energy conversion
    energy_check = bunch["Pt"][0] * C_MMNS * 931.494 / 1000.0
    print(f"  Energy check: {energy_check:.3f} GeV (should be ~{energy_gev})")

    # Create driver
    driver_bunch = bunch.copy()
    for key in driver_bunch:
        if hasattr(driver_bunch[key], "__len__") and len(driver_bunch[key]) > 1:
            driver_bunch[key] = driver_bunch[key][:1]
    driver_bunch["x"] = np.array([1000.0])
    driver_bunch["y"] = np.array([1000.0])
    driver_bunch["z"] = np.array([-1000.0])

    # Run short simulation
    trajectory_rider, trajectory_driver = integrator.integrate_retarded_fields(
        static_steps=10,
        ret_steps=40,
        h_step=1e-5,
        wall_Z=0.0,
        apt_R=5.0,
        sim_type=0,
        init_rider=bunch,
        init_driver=driver_bunch,
        bunch_dist=1e5,
        z_cutoff=50.0,
    )

    print("\nTrajectory analysis:")
    print(f"  Steps: {len(trajectory_rider)}")

    # Check first and last steps
    if len(trajectory_rider) > 0:
        first_step = trajectory_rider[0]
        last_step = trajectory_rider[-1]

        first_z = first_step["z"][0] if "z" in first_step else "N/A"
        last_z = last_step["z"][0] if "z" in last_step else "N/A"

        first_pt = first_step["Pt"][0] if "Pt" in first_step else None
        last_pt = last_step["Pt"][0] if "Pt" in last_step else None

        first_energy = first_pt * C_MMNS * 931.494 / 1000.0 if first_pt else None
        last_energy = last_pt * C_MMNS * 931.494 / 1000.0 if last_pt else None

        print(f"  First step: z = {first_z} mm, E = {first_energy:.3f} GeV")
        print(f"  Last step:  z = {last_z} mm, E = {last_energy:.3f} GeV")
        print(f"  Energy change: {last_energy - first_energy:.6f} GeV")
        print(f"  Position range: {last_z - first_z:.1f} mm")


if __name__ == "__main__":
    test_corrected_units()
