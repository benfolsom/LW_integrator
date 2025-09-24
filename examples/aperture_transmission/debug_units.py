#!/usr/bin/env python3
"""
Debug script to check the actual position units and values in the trajectory data.
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


def debug_trajectory_units():
    """Debug what units the trajectory actually uses."""

    integrator = LienardWiechertIntegrator()
    aperture_radius = 5.0  # 5 microns radius

    # Create a simple test configuration
    config = TestConfiguration(
        particle_count=3,
        transverse_separation=2.0,
        starting_distance=-50.0,  # Start 50mm before aperture
        step_size=1e-5,
        total_steps=100,  # Just a few steps for debugging
        sim_type=0,  # Conducting wall
        wall_z=0.0,  # Aperture at z=0
        aperture_r=aperture_radius,
        z_cutoff=50.0,  # End at 50mm after aperture
    )

    # Create electron bunch
    bunch = create_bunch_uniform_distribution(config, ELECTRON, "line")

    # Set specific energy - 5 GeV
    energy_gev = 5.0
    rest_energy_gev = ELECTRON.mass_mev / 1000.0
    gamma = energy_gev / rest_energy_gev
    momentum_gev = np.sqrt(energy_gev**2 - rest_energy_gev**2)
    beta = momentum_gev / energy_gev
    momentum_scale = 1e-3

    bunch["Pt"] = np.full(3, momentum_gev * 1000 * momentum_scale)
    bunch["Pz"] = bunch["Pt"] * beta
    bunch["Px"] = np.zeros(3)
    bunch["Py"] = np.zeros(3)
    bunch["gamma"] = np.full(3, gamma)
    bunch["bz"] = np.full(3, beta)
    bunch["bx"] = np.zeros(3)
    bunch["by"] = np.zeros(3)

    # Position particles carefully
    bunch["x"] = np.array([-1.0, 0.0, 1.0])  # Within aperture
    bunch["y"] = np.zeros(3)
    bunch["z"] = np.full(3, -50.0)  # Start at -50

    print("Initial bunch positions:")
    print(f"  x: {bunch['x']}")
    print(f"  y: {bunch['y']}")
    print(f"  z: {bunch['z']}")
    print(f"  Pt: {bunch['Pt']}")
    print(f"  Expected initial energy: {energy_gev} GeV")
    print(f"  Calculated initial energy: {bunch['Pt'][0] * C_MMNS:.3f} GeV")

    # Create driver bunch
    driver_bunch = bunch.copy()
    for key in driver_bunch:
        if hasattr(driver_bunch[key], "__len__") and len(driver_bunch[key]) > 1:
            driver_bunch[key] = driver_bunch[key][:1]  # Take only first element
    driver_bunch["x"] = np.array([1000.0])
    driver_bunch["y"] = np.array([1000.0])
    driver_bunch["z"] = np.array([-1000.0])

    # Run short simulation
    print("\nRunning simulation...")
    print(f"  wall_Z: {config.wall_z}")
    print(f"  aperture_r: {config.aperture_r}")
    print(f"  z_cutoff: {config.z_cutoff}")

    trajectory_rider, trajectory_driver = integrator.integrate_retarded_fields(
        static_steps=10,
        ret_steps=90,
        h_step=1e-5,
        wall_Z=config.wall_z,
        apt_R=config.aperture_r,
        sim_type=config.sim_type,
        init_rider=bunch,
        init_driver=driver_bunch,
        bunch_dist=1e5,
        z_cutoff=config.z_cutoff,
    )

    print("\nTrajectory analysis:")
    print(f"  Number of steps: {len(trajectory_rider)}")

    # Check first few steps
    for i, step in enumerate(trajectory_rider[:5]):
        if "z" in step and len(step["z"]) > 0:
            z_vals = step["z"][:3]  # First 3 particles
            pt_vals = step["Pt"][:3] if "Pt" in step else None
            energy_vals = pt_vals * C_MMNS if pt_vals is not None else None

            print(f"  Step {i}:")
            print(f"    z positions: {z_vals}")
            print(f"    Pt values: {pt_vals}")
            print(f"    Energies (GeV): {energy_vals}")

    # Check last few steps
    print("\n  Last few steps:")
    for i, step in enumerate(trajectory_rider[-3:]):
        step_idx = len(trajectory_rider) - 3 + i
        if "z" in step and len(step["z"]) > 0:
            z_vals = step["z"][:3]
            pt_vals = step["Pt"][:3] if "Pt" in step else None
            energy_vals = pt_vals * C_MMNS if pt_vals is not None else None

            print(f"  Step {step_idx}:")
            print(f"    z positions: {z_vals}")
            print(f"    Pt values: {pt_vals}")
            print(f"    Energies (GeV): {energy_vals}")

    # Check overall range
    all_z = []
    all_energies = []
    for step in trajectory_rider:
        if "z" in step and len(step["z"]) > 0 and "Pt" in step:
            all_z.extend(step["z"])
            all_energies.extend(step["Pt"] * C_MMNS)

    if all_z:
        print("\nOverall statistics:")
        print(f"  Z range: {min(all_z):.3f} to {max(all_z):.3f}")
        print(f"  Energy range: {min(all_energies):.6f} to {max(all_energies):.6f} GeV")
        print(
            f"  Z units appear to be: {'mm' if max(all_z) > 10 else 'meters or other'}"
        )


if __name__ == "__main__":
    debug_trajectory_units()
