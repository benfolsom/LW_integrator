"""
Integration tests for different simulation types.

This module tests the electromagnetic integrator with different simulation
types (1, 2, 3) and particle configurations, ensuring proper physics
behavior across all scenarios.

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
    ParticleSpecies,
    TestConfiguration,
    ELECTRON,
    PROTON,
    GOLD_ION,
    create_bunch_uniform_distribution,
    validate_physics_conservation,
    check_radiation_reaction_activation,
)


class TestSimulationTypes:
    """Test different simulation types with various particle configurations."""

    def setup_method(self):
        """Setup for each test method."""
        self.integrator = LienardWiechertIntegrator()
        self.tolerance = 1e-2  # 1% tolerance for physics conservation

    @pytest.mark.integration
    @pytest.mark.parametrize("sim_type", [1, 2, 3])
    @pytest.mark.parametrize("particle_species", [ELECTRON, PROTON, GOLD_ION])
    def test_simulation_types_basic(
        self, sim_type: int, particle_species: ParticleSpecies
    ):
        """Test basic two-particle interactions for different simulation types."""

        # Configure test for this simulation type
        config = TestConfiguration(
            particle_count=2,
            transverse_separation=20.0,  # Wide separation for stable test
            starting_distance=150.0,  # Well above 100mm requirement
            step_size=1e-5,
            total_steps=100,
            sim_type=sim_type,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        # Create test bunches
        rider_bunch = create_bunch_uniform_distribution(
            config, particle_species, "line"
        )
        driver_bunch = create_bunch_uniform_distribution(
            config, particle_species, "line"
        )

        # Offset driver bunch
        driver_bunch["z"] += 50.0  # 50mm behind rider

        print(f"\\n🧪 Testing sim_type={sim_type} with {particle_species.name}")
        print(f"   Initial separation: {config.transverse_separation}mm")
        print(f"   Starting distance: {config.starting_distance}mm")
        print(f"   Particle energy: {particle_species.typical_energy_gev}GeV")

        # Run simulation
        start_time = time.time()

        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=10,
            ret_steps=config.total_steps - 10,
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

        # Validate results
        assert len(trajectory_rider) == config.total_steps + 1
        assert len(trajectory_driver) == config.total_steps + 1

        # Physics conservation tests
        initial_rider = trajectory_rider[0]
        final_rider = trajectory_rider[-1]
        initial_driver = trajectory_driver[0]
        final_driver = trajectory_driver[-1]

        # Check individual bunch conservation
        rider_conservation = validate_physics_conservation(
            initial_rider, final_rider, self.tolerance
        )
        driver_conservation = validate_physics_conservation(
            initial_driver, final_driver, self.tolerance
        )

        print(
            f"   Rider energy conservation: {rider_conservation['energy_conservation']['relative_change']:.2e}"
        )
        print(
            f"   Driver energy conservation: {driver_conservation['energy_conservation']['relative_change']:.2e}"
        )

        # System-wide conservation (combining both bunches)
        initial_total_energy = np.sum(initial_rider["Pt"]) + np.sum(
            initial_driver["Pt"]
        )
        final_total_energy = np.sum(final_rider["Pt"]) + np.sum(final_driver["Pt"])
        total_energy_change = (
            abs(final_total_energy - initial_total_energy) / initial_total_energy
        )

        print(f"   Total system energy conservation: {total_energy_change:.2e}")

        # Assert conservation within tolerance
        assert rider_conservation["energy_conservation"][
            "passed"
        ], "Rider energy not conserved"
        assert driver_conservation["energy_conservation"][
            "passed"
        ], "Driver energy not conserved"
        assert (
            total_energy_change < self.tolerance
        ), f"Total energy change {total_energy_change:.2e} exceeds tolerance"

        # Check that particles moved forward
        assert np.all(
            final_rider["z"] > initial_rider["z"]
        ), "Rider particles moved backward"
        assert np.all(
            final_driver["z"] > initial_driver["z"]
        ), "Driver particles moved backward"

        print(f"   ✅ {particle_species.name} sim_type={sim_type} test passed")

    @pytest.mark.integration
    @pytest.mark.parametrize("particle_count", [2, 5, 10, 25])
    def test_particle_count_scaling(self, particle_count: int):
        """Test performance and accuracy with different particle counts."""

        config = TestConfiguration(
            particle_count=particle_count,
            transverse_separation=10.0,
            starting_distance=200.0,
            step_size=2e-5,
            total_steps=50,
            sim_type=2,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        # Use protons for this test
        rider_bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        driver_bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        driver_bunch["z"] += 30.0

        print(f"\\n🔢 Testing {particle_count} particles per bunch")

        start_time = time.time()

        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=5,
            ret_steps=config.total_steps - 5,
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

        # Calculate performance metrics
        particles_per_second = (
            particle_count * 2 * config.total_steps
        ) / simulation_time

        print(f"   Simulation time: {simulation_time:.3f}s")
        print(f"   Performance: {particles_per_second:.0f} particle-steps/second")

        # Basic validation
        assert len(trajectory_rider) == config.total_steps + 1
        assert len(trajectory_driver) == config.total_steps + 1

        # Physics conservation
        initial_energy = np.sum(trajectory_rider[0]["Pt"]) + np.sum(
            trajectory_driver[0]["Pt"]
        )
        final_energy = np.sum(trajectory_rider[-1]["Pt"]) + np.sum(
            trajectory_driver[-1]["Pt"]
        )
        energy_conservation = abs(final_energy - initial_energy) / initial_energy

        print(f"   Energy conservation: {energy_conservation:.2e}")
        assert (
            energy_conservation < self.tolerance
        ), f"Energy not conserved for {particle_count} particles"

        # Performance regression test (adjust based on expected performance)
        expected_min_performance = 1000  # particle-steps/second
        assert (
            particles_per_second > expected_min_performance
        ), f"Performance too slow: {particles_per_second:.0f} < {expected_min_performance}"

        print(f"   ✅ {particle_count} particles test passed")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_particle_count(self):
        """Test with large particle count (100+ particles)."""

        particle_count = 100

        config = TestConfiguration(
            particle_count=particle_count,
            transverse_separation=5.0,
            starting_distance=250.0,
            step_size=5e-5,  # Larger steps for performance
            total_steps=20,  # Fewer steps for performance
            sim_type=1,  # Simple simulation type
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        # Use electrons for faster simulation
        rider_bunch = create_bunch_uniform_distribution(config, ELECTRON, "gaussian")
        driver_bunch = create_bunch_uniform_distribution(config, ELECTRON, "gaussian")
        driver_bunch["z"] += 20.0

        print(f"\\n🚀 Large scale test: {particle_count} particles per bunch")
        print("   Distribution: Gaussian")
        print(f"   Steps: {config.total_steps}")

        start_time = time.time()

        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=2,
            ret_steps=config.total_steps - 2,
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
        print(f"   Total particle-steps: {particle_count * 2 * config.total_steps}")

        # Validation
        assert len(trajectory_rider) == config.total_steps + 1
        assert len(trajectory_driver) == config.total_steps + 1

        # Check for any NaN or infinite values
        final_rider = trajectory_rider[-1]
        final_driver = trajectory_driver[-1]

        assert np.all(
            np.isfinite(final_rider["x"])
        ), "NaN/inf in final rider x positions"
        assert np.all(np.isfinite(final_rider["Pt"])), "NaN/inf in final rider momenta"
        assert np.all(
            np.isfinite(final_driver["x"])
        ), "NaN/inf in final driver x positions"
        assert np.all(
            np.isfinite(final_driver["Pt"])
        ), "NaN/inf in final driver momenta"

        print("   ✅ Large scale test passed")

    @pytest.mark.integration
    def test_close_approach_radiation_reaction(self):
        """Test radiation reaction activation during close particle approach."""

        config = TestConfiguration(
            particle_count=2,
            transverse_separation=0.5,  # Very close approach - 0.5mm
            starting_distance=100.0,
            step_size=1e-6,  # Very fine time steps
            total_steps=500,
            sim_type=2,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        # Use heavy ions for maximum electromagnetic interaction
        rider_bunch = create_bunch_uniform_distribution(config, GOLD_ION, "line")
        driver_bunch = create_bunch_uniform_distribution(config, GOLD_ION, "line")
        driver_bunch["z"] += 10.0  # Close longitudinal separation too

        print("\\n⚡ Radiation reaction test")
        print(f"   Particle type: {GOLD_ION.name}")
        print(f"   Transverse separation: {config.transverse_separation}mm")
        print(f"   Step size: {config.step_size}ns")

        start_time = time.time()

        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=10,
            ret_steps=config.total_steps - 10,
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

        # Check for radiation reaction activation
        radiation_analysis = check_radiation_reaction_activation(trajectory_rider)

        print(f"   Max acceleration: {radiation_analysis['max_acceleration']:.2e} m/s²")
        print(f"   Radiation detected: {radiation_analysis['radiation_detected']}")
        print(f"   Active steps: {len(radiation_analysis['radiation_active_steps'])}")

        # Validate energy transfer
        initial_rider_energy = np.sum(trajectory_rider[0]["Pt"])
        final_rider_energy = np.sum(trajectory_rider[-1]["Pt"])
        energy_change = (
            abs(final_rider_energy - initial_rider_energy) / initial_rider_energy
        )

        print(f"   Rider energy change: {energy_change:.2e}")

        # For close approach with heavy ions, we expect significant energy exchange
        assert (
            energy_change > 1e-6
        ), "No significant energy exchange detected in close approach"

        print("   ✅ Radiation reaction test passed")


if __name__ == "__main__":
    # Allow running tests directly
    test_instance = TestSimulationTypes()
    test_instance.setup_method()

    print("Running simulation type tests...")

    # Run a basic test
    test_instance.test_simulation_types_basic(2, PROTON)
    test_instance.test_particle_count_scaling(5)

    print("\\n✅ All direct tests passed!")
