#!/usr/bin/env python3
"""
Test energy conversion using the same method as the test code.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from physics.constants import C_MMNS
from tests.test_config import (
    ELECTRON,
    create_bunch_uniform_distribution,
    TestConfiguration,
)


def test_with_test_config():
    """Test using the same method as the test configuration."""

    config = TestConfiguration(
        particle_count=3,
        transverse_separation=2.0,
        starting_distance=-50.0,
        step_size=1e-5,
        total_steps=50,
        sim_type=0,
        wall_z=0.0,
        aperture_r=5.0,
        z_cutoff=50.0,
    )

    # Create bunch using test method
    bunch = create_bunch_uniform_distribution(config, ELECTRON, "line")

    print("Test configuration bunch:")
    print(f"  Available keys: {list(bunch.keys())}")
    print(f"  Pt: {bunch['Pt'][0]:.6f}")
    print(f"  mass: {bunch['mass'][0]:.6f}")
    if "q" in bunch:
        print(f"  charge: {bunch['q'][0]:.6f}")
    if "gamma" in bunch:
        print(f"  gamma: {bunch['gamma'][0]:.6f}")

    # Calculate energy using test method
    initial_energy = bunch["Pt"][0] * C_MMNS
    print(f"  Energy (test method): {initial_energy:.6f} (units?)")

    # The electron typical energy is:
    print(f"  Electron typical energy: {ELECTRON.typical_energy_gev} GeV")

    # Check what the ratio is
    if initial_energy != 0:
        ratio = initial_energy / ELECTRON.typical_energy_gev
        print(f"  Ratio: {ratio:.6f}")

        # So the conversion factor is this ratio
        print(f"  To get GeV: energy_GeV = (Pt * C_MMNS) / {ratio:.6f}")
    else:
        print("  Pt is zero - bunch not properly initialized")


if __name__ == "__main__":
    test_with_test_config()
