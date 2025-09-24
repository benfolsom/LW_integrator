"""
Integration tests for aperture interactions and beam dynamics.

This module tests particle interactions with apertures, walls, and boundaries,
including near-miss scenarios and z_cutoff functionality.

Author: Ben Folsom
Date: 2025-09-18
"""

import pytest
import numpy as np
import time

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.trajectory_integrator import LienardWiechertIntegrator
from tests.test_config import (
    TestConfiguration,
    ELECTRON,
    PROTON,
    create_bunch_uniform_distribution,
)


class TestApertureInteractions:
    """Test particle interactions with apertures and boundaries."""

    def setup_method(self):
        """Setup for each test method."""
        self.integrator = LienardWiechertIntegrator()
        self.tolerance = 1e-2

    @pytest.mark.integration
    @pytest.mark.parametrize("aperture_radius", [5.0, 10.0, 20.0, 50.0])
    def test_aperture_transmission(self, aperture_radius: float):
        """Test particle transmission through different aperture sizes."""

        config = TestConfiguration(
            particle_count=10,
            transverse_separation=aperture_radius
            * 1.5,  # Some particles beyond aperture
            starting_distance=150.0,
            step_size=1e-5,
            total_steps=200,
            sim_type=2,
            wall_z=100.0,  # Wall close enough to interact
            aperture_r=aperture_radius,
            z_cutoff=50.0,  # Wall switching point
        )

        # Create beam with circular distribution
        rider_bunch = create_bunch_uniform_distribution(config, PROTON, "circle")
        driver_bunch = create_bunch_uniform_distribution(config, PROTON, "circle")
        driver_bunch["z"] += 25.0

        print("\\n🕳️  Testing aperture transmission")
        print(f"   Aperture radius: {aperture_radius}mm")
        print(f"   Beam distribution radius: {config.transverse_separation/2:.1f}mm")
        print(f"   Wall position: {config.wall_z}mm")
        print(f"   z_cutoff: {config.z_cutoff}mm")

        # Count particles initially within aperture
        initial_radius_rider = np.sqrt(rider_bunch["x"] ** 2 + rider_bunch["y"] ** 2)
        initial_radius_driver = np.sqrt(driver_bunch["x"] ** 2 + driver_bunch["y"] ** 2)

        initial_transmitted_rider = np.sum(initial_radius_rider <= aperture_radius)
        initial_transmitted_driver = np.sum(initial_radius_driver <= aperture_radius)

        print(
            f"   Initial particles within aperture: R={initial_transmitted_rider}, D={initial_transmitted_driver}"
        )

        start_time = time.time()

        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=20,
            ret_steps=config.total_steps - 20,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=rider_bunch,
            init_driver=driver_bunch,
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        simulation_time = time.time() - start_time
        print(f"   Simulation time: {simulation_time:.3f}s")

        # Check final transmission
        final_rider = trajectory_rider[-1]
        final_driver = trajectory_driver[-1]

        final_radius_rider = np.sqrt(final_rider["x"] ** 2 + final_rider["y"] ** 2)
        final_radius_driver = np.sqrt(final_driver["x"] ** 2 + final_driver["y"] ** 2)

        # Count particles that made it through
        final_transmitted_rider = np.sum(
            final_radius_rider <= aperture_radius * 1.1
        )  # Small tolerance
        final_transmitted_driver = np.sum(final_radius_driver <= aperture_radius * 1.1)

        print(
            f"   Final particles within aperture: R={final_transmitted_rider}, D={final_transmitted_driver}"
        )

        # Validate basic physics
        assert len(trajectory_rider) == config.total_steps + 1
        assert len(trajectory_driver) == config.total_steps + 1

        # Check that particles progressed in z
        assert np.all(
            final_rider["z"] >= rider_bunch["z"]
        ), "Rider particles moved backward"
        assert np.all(
            final_driver["z"] >= driver_bunch["z"]
        ), "Driver particles moved backward"

        # Transmission rate should be reasonable
        if initial_transmitted_rider > 0:
            transmission_rate_rider = (
                final_transmitted_rider / initial_transmitted_rider
            )
            print(f"   Rider transmission rate: {transmission_rate_rider:.2f}")
            assert transmission_rate_rider >= 0.1, "Transmission rate too low for rider"

        if initial_transmitted_driver > 0:
            transmission_rate_driver = (
                final_transmitted_driver / initial_transmitted_driver
            )
            print(f"   Driver transmission rate: {transmission_rate_driver:.2f}")
            assert (
                transmission_rate_driver >= 0.1
            ), "Transmission rate too low for driver"

        print(f"   ✅ Aperture {aperture_radius}mm test passed")

    @pytest.mark.integration
    @pytest.mark.parametrize("wall_position", [50.0, 100.0, 200.0])
    def test_wall_interactions(self, wall_position: float):
        """Test particle interactions with walls at different positions."""

        config = TestConfiguration(
            particle_count=5,
            transverse_separation=8.0,
            starting_distance=120.0,
            step_size=1e-5,
            total_steps=150,
            sim_type=2,
            wall_z=wall_position,
            aperture_r=10.0,
            z_cutoff=wall_position - 25.0,
        )

        rider_bunch = create_bunch_uniform_distribution(config, ELECTRON, "line")
        driver_bunch = create_bunch_uniform_distribution(config, ELECTRON, "line")
        driver_bunch["z"] += 15.0

        print("\\n🧱 Testing wall interactions")
        print(f"   Wall position: {wall_position}mm")
        print(f"   z_cutoff: {config.z_cutoff}mm")
        print(f"   Starting distance: {config.starting_distance}mm")

        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=15,
            ret_steps=config.total_steps - 15,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=rider_bunch,
            init_driver=driver_bunch,
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        # Check interaction with wall
        final_rider = trajectory_rider[-1]
        final_driver = trajectory_driver[-1]

        max_z_rider = np.max(final_rider["z"])
        max_z_driver = np.max(final_driver["z"])

        print(
            f"   Max final z positions: R={max_z_rider:.1f}mm, D={max_z_driver:.1f}mm"
        )

        # Particles should not significantly exceed wall position
        # (some tolerance for numerical effects and field interactions)
        wall_tolerance = 10.0  # mm

        if wall_position < 150:  # Only check for nearby walls
            assert (
                max_z_rider <= wall_position + wall_tolerance
            ), f"Rider exceeded wall: {max_z_rider} > {wall_position + wall_tolerance}"
            assert (
                max_z_driver <= wall_position + wall_tolerance
            ), f"Driver exceeded wall: {max_z_driver} > {wall_position + wall_tolerance}"

        print(f"   ✅ Wall {wall_position}mm test passed")

    @pytest.mark.integration
    def test_z_cutoff_functionality(self):
        """Test z_cutoff boundary switching functionality."""

        config = TestConfiguration(
            particle_count=3,
            transverse_separation=5.0,
            starting_distance=100.0,
            step_size=1e-5,
            total_steps=300,  # Long simulation to cross boundaries
            sim_type=2,
            wall_z=80.0,
            aperture_r=15.0,
            z_cutoff=40.0,  # Wall switches at 40mm
        )

        rider_bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        driver_bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        driver_bunch["z"] += 10.0

        print("\\n✂️  Testing z_cutoff functionality")
        print(f"   Initial wall_Z: {config.wall_z}mm")
        print(f"   z_cutoff: {config.z_cutoff}mm")
        print("   Expected wall advancement when particles cross z_cutoff")

        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=30,
            ret_steps=config.total_steps - 30,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=rider_bunch,
            init_driver=driver_bunch,
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        # Check that simulation completed
        assert len(trajectory_rider) == config.total_steps + 1
        assert len(trajectory_driver) == config.total_steps + 1

        # Analyze z progression
        z_positions = []
        for step in range(0, len(trajectory_rider), 50):  # Sample every 50 steps
            avg_z = (
                np.mean(trajectory_rider[step]["z"])
                + np.mean(trajectory_driver[step]["z"])
            ) / 2
            z_positions.append(avg_z)

        print(f"   Z progression (sampled): {[f'{z:.1f}' for z in z_positions]}")

        # Check final positions
        final_z_rider = np.mean(trajectory_rider[-1]["z"])
        final_z_driver = np.mean(trajectory_driver[-1]["z"])

        print(f"   Final average z: R={final_z_rider:.1f}mm, D={final_z_driver:.1f}mm")

        # Particles should have progressed beyond initial z_cutoff
        assert final_z_rider > config.z_cutoff, "Rider didn't progress beyond z_cutoff"
        assert (
            final_z_driver > config.z_cutoff
        ), "Driver didn't progress beyond z_cutoff"

        print("   ✅ z_cutoff test passed")

    @pytest.mark.integration
    def test_near_miss_scenarios(self):
        """Test various near-miss scenarios with different geometries."""

        scenarios = [
            {"name": "Very close approach", "separation": 1.0, "aperture": 5.0},
            {"name": "Aperture grazing", "separation": 9.5, "aperture": 10.0},
            {"name": "Wall proximity", "separation": 15.0, "aperture": 20.0},
        ]

        for scenario in scenarios:
            print(f"\\n🎯 Testing {scenario['name']}")

            config = TestConfiguration(
                particle_count=2,
                transverse_separation=scenario["separation"],
                starting_distance=150.0,
                step_size=5e-6,  # Fine steps for accuracy
                total_steps=400,
                sim_type=2,
                wall_z=100.0,
                aperture_r=scenario["aperture"],
                z_cutoff=50.0,
            )

            rider_bunch = create_bunch_uniform_distribution(config, PROTON, "line")
            driver_bunch = create_bunch_uniform_distribution(config, PROTON, "line")
            driver_bunch["z"] += 20.0

            print(f"   Separation: {scenario['separation']}mm")
            print(f"   Aperture: {scenario['aperture']}mm")

            start_time = time.time()

            trajectory_rider, trajectory_driver = (
                self.integrator.integrate_retarded_fields(
                    static_steps=40,
                    ret_steps=config.total_steps - 40,
                    h_step=config.step_size,
                    wall_Z=config.wall_z,
                    apt_R=config.aperture_r,
                    sim_type=config.sim_type,
                    init_rider=rider_bunch,
                    init_driver=driver_bunch,
                    bunch_dist=1e5,
                    z_cutoff=config.z_cutoff,
                )
            )

            simulation_time = time.time() - start_time

            # Validate numerical stability
            final_rider = trajectory_rider[-1]
            final_driver = trajectory_driver[-1]

            # Check for NaN or infinite values
            assert np.all(
                np.isfinite(final_rider["Pt"])
            ), f"NaN/inf in {scenario['name']} rider momentum"
            assert np.all(
                np.isfinite(final_driver["Pt"])
            ), f"NaN/inf in {scenario['name']} driver momentum"
            assert np.all(
                np.isfinite(final_rider["x"])
            ), f"NaN/inf in {scenario['name']} rider positions"
            assert np.all(
                np.isfinite(final_driver["x"])
            ), f"NaN/inf in {scenario['name']} driver positions"

            # Energy conservation check
            initial_energy = np.sum(trajectory_rider[0]["Pt"]) + np.sum(
                trajectory_driver[0]["Pt"]
            )
            final_energy = np.sum(final_rider["Pt"]) + np.sum(final_driver["Pt"])
            energy_change = abs(final_energy - initial_energy) / initial_energy

            print(f"   Energy conservation: {energy_change:.2e}")
            print(f"   Simulation time: {simulation_time:.3f}s")

            # More relaxed tolerance for near-miss scenarios
            near_miss_tolerance = 5e-2  # 5%
            assert (
                energy_change < near_miss_tolerance
            ), f"Energy not conserved in {scenario['name']}: {energy_change:.2e}"

            print(f"   ✅ {scenario['name']} passed")


if __name__ == "__main__":
    # Allow running tests directly
    test_instance = TestApertureInteractions()
    test_instance.setup_method()

    print("Running aperture interaction tests...")

    # Run basic tests
    test_instance.test_aperture_transmission(10.0)
    test_instance.test_wall_interactions(100.0)
    test_instance.test_z_cutoff_functionality()

    print("\\n✅ All direct aperture tests passed!")
