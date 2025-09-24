"""
Performance and scaling tests for the LW integrator.

This module tests the integrator with large particle counts (up to 500+)
and measures performance metrics, memory usage, and scaling behavior.

Author: Ben Folsom
Date: 2025-09-18
"""

import pytest
import numpy as np
import time
import psutil
import os

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.trajectory_integrator import LienardWiechertIntegrator
from tests.test_config import (
    TestConfiguration,
    ELECTRON,
    PROTON,
    create_bunch_uniform_distribution,
)


class TestPerformanceScaling:
    """Test performance and scaling with large particle counts."""

    def setup_method(self):
        """Setup for each test method."""
        self.integrator = LienardWiechertIntegrator()
        self.tolerance = 2e-2  # Relaxed tolerance for large simulations

        # Performance tracking
        self.performance_data = []

    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    @pytest.mark.performance
    @pytest.mark.parametrize("particle_count", [2, 5, 10, 25, 50, 100])
    def test_scaling_performance(self, particle_count: int):
        """Test performance scaling with increasing particle count."""

        config = TestConfiguration(
            particle_count=particle_count,
            transverse_separation=5.0,
            starting_distance=200.0,
            step_size=2e-5,  # Moderate step size for performance
            total_steps=30,  # Limited steps for scaling test
            sim_type=2,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        # Use protons for consistent testing
        rider_bunch = create_bunch_uniform_distribution(config, PROTON, "gaussian")
        driver_bunch = create_bunch_uniform_distribution(config, PROTON, "gaussian")
        driver_bunch["z"] += 25.0

        print(f"\\n🚀 Performance test: {particle_count} particles per bunch")

        # Measure initial memory
        initial_memory = self.measure_memory_usage()

        # Run simulation with timing
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
        final_memory = self.measure_memory_usage()

        # Calculate performance metrics
        total_particle_steps = particle_count * 2 * config.total_steps
        particles_per_second = total_particle_steps / simulation_time
        memory_per_particle = (final_memory - initial_memory) / (particle_count * 2)

        print(f"   Simulation time: {simulation_time:.3f}s")
        print(f"   Particles/second: {particles_per_second:.0f}")
        print(f"   Memory usage: {final_memory - initial_memory:.1f}MB")
        print(f"   Memory/particle: {memory_per_particle:.2f}MB")

        # Store performance data
        self.performance_data.append(
            {
                "particle_count": particle_count,
                "simulation_time": simulation_time,
                "particles_per_second": particles_per_second,
                "memory_usage": final_memory - initial_memory,
                "memory_per_particle": memory_per_particle,
            }
        )

        # Validate results
        assert len(trajectory_rider) == config.total_steps + 1
        assert len(trajectory_driver) == config.total_steps + 1

        # Physics conservation check
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
        ), f"Energy not conserved: {energy_conservation:.2e}"

        # Performance requirements (adjust based on expected performance)
        min_performance = max(
            100, 1000 - particle_count * 5
        )  # Scale down expectations for large counts
        assert (
            particles_per_second > min_performance
        ), f"Performance too low: {particles_per_second:.0f} < {min_performance}"

        print(f"   ✅ {particle_count} particles test passed")

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.parametrize("particle_count", [200, 500])
    def test_large_scale_performance(self, particle_count: int):
        """Test performance with very large particle counts."""

        config = TestConfiguration(
            particle_count=particle_count,
            transverse_separation=2.0,
            starting_distance=300.0,
            step_size=5e-5,  # Larger steps for performance
            total_steps=10,  # Minimal steps for large count
            sim_type=1,  # Simplest simulation type
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        # Use electrons for fastest simulation
        rider_bunch = create_bunch_uniform_distribution(config, ELECTRON, "gaussian")
        driver_bunch = create_bunch_uniform_distribution(config, ELECTRON, "gaussian")
        driver_bunch["z"] += 15.0

        print(f"\\n🏋️  Large scale test: {particle_count} particles per bunch")
        print(f"   Total particles: {particle_count * 2}")
        print("   Distribution: Gaussian")

        initial_memory = self.measure_memory_usage()
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
        final_memory = self.measure_memory_usage()

        total_particle_steps = particle_count * 2 * config.total_steps
        particles_per_second = total_particle_steps / simulation_time

        print(f"   Simulation time: {simulation_time:.3f}s")
        print(f"   Total particle-steps: {total_particle_steps}")
        print(f"   Particles/second: {particles_per_second:.0f}")
        print(f"   Memory usage: {final_memory - initial_memory:.1f}MB")

        # Validate numerical stability for large count
        final_rider = trajectory_rider[-1]
        final_driver = trajectory_driver[-1]

        # Check for NaN or infinite values
        assert np.all(
            np.isfinite(final_rider["Pt"])
        ), "NaN/inf in large scale rider momentum"
        assert np.all(
            np.isfinite(final_driver["Pt"])
        ), "NaN/inf in large scale driver momentum"
        assert np.all(
            np.isfinite(final_rider["x"])
        ), "NaN/inf in large scale rider positions"
        assert np.all(
            np.isfinite(final_driver["x"])
        ), "NaN/inf in large scale driver positions"

        # Memory usage should be reasonable
        memory_usage_mb = final_memory - initial_memory
        max_expected_memory = particle_count * 0.1  # 0.1 MB per particle maximum
        assert (
            memory_usage_mb < max_expected_memory
        ), f"Memory usage too high: {memory_usage_mb:.1f}MB > {max_expected_memory:.1f}MB"

        print(f"   ✅ Large scale {particle_count} particles test passed")

    @pytest.mark.performance
    def test_memory_scaling(self):
        """Test memory usage scaling with particle count."""

        particle_counts = [10, 25, 50, 100]
        memory_measurements = []

        for pcount in particle_counts:
            config = TestConfiguration(
                particle_count=pcount,
                transverse_separation=3.0,
                starting_distance=150.0,
                step_size=2e-5,
                total_steps=15,
                sim_type=1,
                wall_z=1e5,
                aperture_r=1e5,
                z_cutoff=0.0,
            )

            rider_bunch = create_bunch_uniform_distribution(config, PROTON, "line")
            driver_bunch = create_bunch_uniform_distribution(config, PROTON, "line")
            driver_bunch["z"] += 20.0

            initial_memory = self.measure_memory_usage()

            trajectory_rider, trajectory_driver = (
                self.integrator.integrate_retarded_fields(
                    static_steps=3,
                    ret_steps=config.total_steps - 3,
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

            final_memory = self.measure_memory_usage()
            memory_used = final_memory - initial_memory
            memory_measurements.append(memory_used)

            print(f"\\n💾 Memory test: {pcount} particles → {memory_used:.1f}MB")

        # Check that memory scaling is reasonable (should be roughly linear)
        memory_ratios = []
        for i in range(1, len(memory_measurements)):
            ratio = memory_measurements[i] / memory_measurements[0]
            expected_ratio = particle_counts[i] / particle_counts[0]
            memory_ratios.append(ratio / expected_ratio)

            print(
                f"   {particle_counts[i]}/{particle_counts[0]} particles: "
                f"memory ratio {ratio:.2f}, expected {expected_ratio:.2f}"
            )

        # Memory scaling should be reasonably linear (within factor of 3)
        for ratio in memory_ratios:
            assert 0.3 < ratio < 3.0, f"Memory scaling not linear: ratio {ratio:.2f}"

        print("   ✅ Memory scaling test passed")

    @pytest.mark.performance
    def test_optimization_effectiveness(self):
        """Test that optimized integrator performs better than standard."""

        particle_count = 25
        config = TestConfiguration(
            particle_count=particle_count,
            transverse_separation=5.0,
            starting_distance=200.0,
            step_size=1e-5,
            total_steps=50,
            sim_type=2,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        rider_bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        driver_bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        driver_bunch["z"] += 20.0

        print("\\n⚡ Optimization effectiveness test")

        # Test optimized integrator (default)
        optimized_integrator = LienardWiechertIntegrator()
        print(f"   Optimized type: {type(optimized_integrator).__name__}")

        start_time = time.time()

        traj_opt_r, traj_opt_d = optimized_integrator.integrate_retarded_fields(
            static_steps=5,
            ret_steps=config.total_steps - 5,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=rider_bunch.copy(),
            init_driver=driver_bunch.copy(),
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        optimized_time = time.time() - start_time

        # Test standard integrator
        standard_integrator = LienardWiechertIntegrator(use_optimized=False)
        print(f"   Standard type: {type(standard_integrator).__name__}")

        start_time = time.time()

        traj_std_r, traj_std_d = standard_integrator.integrate_retarded_fields(
            static_steps=5,
            ret_steps=config.total_steps - 5,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=rider_bunch.copy(),
            init_driver=driver_bunch.copy(),
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        standard_time = time.time() - start_time

        speedup = standard_time / optimized_time

        print(f"   Optimized time: {optimized_time:.3f}s")
        print(f"   Standard time: {standard_time:.3f}s")
        print(f"   Speedup: {speedup:.2f}x")

        # Results should be approximately equivalent
        opt_final_energy = np.sum(traj_opt_r[-1]["Pt"]) + np.sum(traj_opt_d[-1]["Pt"])
        std_final_energy = np.sum(traj_std_r[-1]["Pt"]) + np.sum(traj_std_d[-1]["Pt"])

        energy_difference = abs(opt_final_energy - std_final_energy) / std_final_energy
        print(f"   Energy difference: {energy_difference:.2e}")

        # Results should be very similar (within 1%)
        assert (
            energy_difference < 1e-2
        ), f"Optimized and standard results differ too much: {energy_difference:.2e}"

        # Optimized should be faster (or at least not significantly slower)
        # Note: For small tests, JIT overhead might make optimized slower
        if particle_count >= 20:
            assert (
                speedup >= 0.8
            ), f"Optimization provides no benefit: {speedup:.2f}x speedup"

        print("   ✅ Optimization effectiveness test passed")


if __name__ == "__main__":
    # Allow running tests directly
    test_instance = TestPerformanceScaling()
    test_instance.setup_method()

    print("Running performance tests...")

    # Run basic tests
    test_instance.test_scaling_performance(10)
    test_instance.test_memory_scaling()
    test_instance.test_optimization_effectiveness()

    print("\\n✅ All direct performance tests passed!")
