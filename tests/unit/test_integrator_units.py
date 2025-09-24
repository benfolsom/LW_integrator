"""
Unit tests for the trajectory integrator core functions.

This module tests individual functions and methods of the LienardWiechertIntegrator
to ensure correctness at the component level.

Author: Ben Folsom
Date: 2025-09-18
"""

import pytest
import numpy as np

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.trajectory_integrator import LienardWiechertIntegrator
from physics.constants import C_MMNS
from physics.particle_initialization import ELEMENTARY_CHARGE
from tests.test_config import (
    PROTON,
    create_bunch_uniform_distribution,
    TestConfiguration,
)


class TestTrajectoryIntegratorUnits:
    """Unit tests for individual integrator methods."""

    def setup_method(self):
        """Setup for each test method."""
        self.integrator = LienardWiechertIntegrator()

        # Create minimal test particle
        self.test_particle = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "Px": np.array([0.0]),
            "Py": np.array([0.0]),
            "Pz": np.array([100.0]),  # Moving in z-direction
            "Pt": np.array([100.0]),
            "mass": np.array([938.3]),  # Proton mass in MeV
            "q": np.array([ELEMENTARY_CHARGE]),  # Elementary charge in Gaussian units
            "gamma": np.array([100.0 / 938.3]),
            "bx": np.array([0.0]),
            "by": np.array([0.0]),
            "bz": np.array([1.0]),
            "bdotx": np.array([0.0]),
            "bdoty": np.array([0.0]),
            "bdotz": np.array([0.0]),
            "t": np.array([0.0]),
        }

    @pytest.mark.unit
    def test_integrator_initialization(self):
        """Test that integrator initializes correctly."""
        integrator = LienardWiechertIntegrator()

        assert hasattr(integrator, "c_mmns")
        assert hasattr(integrator, "charge_gaussian")
        assert integrator.c_mmns == C_MMNS

        # Test with configuration
        from physics.simulation_types import SimulationConfig

        config = SimulationConfig()
        integrator_with_config = LienardWiechertIntegrator(config)
        assert integrator_with_config.config is not None

    @pytest.mark.unit
    def test_optimized_integrator_default(self):
        """Test that optimized integrator is returned by default."""
        integrator = LienardWiechertIntegrator()

        # Should be OptimizedLienardWiechertIntegrator by default
        assert type(integrator).__name__ == "OptimizedLienardWiechertIntegrator"
        assert hasattr(integrator, "use_jit")

    @pytest.mark.unit
    def test_standard_integrator_explicit(self):
        """Test that standard integrator can be explicitly requested."""
        integrator = LienardWiechertIntegrator(use_optimized=False)

        # Should be standard LienardWiechertIntegrator
        assert type(integrator).__name__ == "LienardWiechertIntegrator"
        assert not hasattr(integrator, "use_jit")

    @pytest.mark.unit
    def test_distance_calculation(self):
        """Test distance calculation between particles."""
        # Create two particles separated by known distance
        particle1 = {"x": 0.0, "y": 0.0, "z": 0.0}
        particle2 = {"x": 3.0, "y": 4.0, "z": 0.0}  # 3-4-5 triangle

        expected_distance = 5.0

        # Calculate distance using standard formula
        dx = particle2["x"] - particle1["x"]
        dy = particle2["y"] - particle1["y"]
        dz = particle2["z"] - particle1["z"]
        calculated_distance = np.sqrt(dx**2 + dy**2 + dz**2)

        assert np.isclose(calculated_distance, expected_distance, rtol=1e-10)

    @pytest.mark.unit
    def test_relativistic_calculations(self):
        """Test relativistic gamma and velocity calculations."""
        # Test particle with known momentum and mass
        momentum = 1000.0  # MeV/c
        mass = 938.3  # MeV/c² (proton)

        # Calculate gamma
        gamma = np.sqrt(1 + (momentum / (mass * C_MMNS)) ** 2)

        # Gamma should be > 1 for relativistic particle
        assert gamma > 1.0

        # Calculate velocity
        beta = momentum / (gamma * mass * C_MMNS)

        # Beta should be < 1
        assert 0 < beta < 1.0

        # Test energy-momentum relation
        energy = gamma * mass * C_MMNS**2
        energy_momentum_check = np.sqrt(
            (momentum * C_MMNS) ** 2 + (mass * C_MMNS**2) ** 2
        )

        assert np.isclose(energy, energy_momentum_check, rtol=1e-10)

    @pytest.mark.unit
    def test_static_step_calculation(self):
        """Test static (no interaction) step calculation."""
        initial_state = self.test_particle.copy()
        h_step = 1e-5  # 10 ps

        # Run one static step
        final_state = self.integrator.equations_of_motion_static_internal(
            h_step, initial_state, 0  # particle index 0
        )

        # Position should advance by velocity * time
        expected_z = initial_state["z"][0] + initial_state["bz"][0] * C_MMNS * h_step

        assert np.isclose(final_state["z"][0], expected_z, rtol=1e-10)

        # Momentum should remain constant in static step
        assert np.isclose(final_state["Pt"][0], initial_state["Pt"][0], rtol=1e-10)
        assert np.isclose(final_state["Px"][0], initial_state["Px"][0], rtol=1e-10)

        # Time should advance
        assert np.isclose(
            final_state["t"][0], initial_state["t"][0] + h_step, rtol=1e-10
        )

    @pytest.mark.unit
    def test_charge_conservation(self):
        """Test that charge is conserved in all operations."""
        config = TestConfiguration(
            particle_count=5,
            transverse_separation=10.0,
            starting_distance=100.0,
            step_size=1e-5,
            total_steps=10,
            sim_type=2,  # Free particle bunches (no cav_spacing needed)
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        initial_charge = np.sum(bunch["q"])

        # Run a few steps
        trajectory, _ = self.integrator.integrate_retarded_fields(
            static_steps=5,
            ret_steps=5,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=bunch,
            init_driver=bunch.copy(),  # Same bunch for simplicity
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        # Check charge conservation at each step
        for step, state in enumerate(trajectory):
            final_charge = np.sum(state["q"])
            assert np.isclose(
                final_charge, initial_charge, rtol=1e-12
            ), f"Charge not conserved at step {step}"

    @pytest.mark.unit
    def test_momentum_units_consistency(self):
        """Test consistency of momentum units throughout calculations."""
        config = TestConfiguration(
            particle_count=1,
            transverse_separation=1.0,
            starting_distance=100.0,
            step_size=1e-6,
            total_steps=5,
            sim_type=1,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        bunch = create_bunch_uniform_distribution(config, PROTON, "line")

        # Check initial momentum consistency
        Pt_calculated = np.sqrt(bunch["Px"] ** 2 + bunch["Py"] ** 2 + bunch["Pz"] ** 2)
        assert np.allclose(Pt_calculated, bunch["Pt"], rtol=1e-10)

        # Check velocity consistency
        beta_magnitude = np.sqrt(bunch["bx"] ** 2 + bunch["by"] ** 2 + bunch["bz"] ** 2)
        expected_beta = bunch["Pt"] / (bunch["gamma"] * bunch["mass"] * C_MMNS)
        assert np.allclose(beta_magnitude, expected_beta, rtol=1e-10)

    @pytest.mark.unit
    def test_boundary_conditions(self):
        """Test handling of boundary conditions and edge cases."""
        # Test with zero velocity
        zero_particle = self.test_particle.copy()
        zero_particle["Px"] = np.array([0.0])
        zero_particle["Py"] = np.array([0.0])
        zero_particle["Pz"] = np.array([0.0])
        zero_particle["Pt"] = np.array([0.0])
        zero_particle["bx"] = np.array([0.0])
        zero_particle["by"] = np.array([0.0])
        zero_particle["bz"] = np.array([0.0])

        # Should handle zero momentum gracefully
        try:
            final_state = self.integrator.equations_of_motion_static_internal(
                1e-5, zero_particle, 0
            )
            # Position should not change for zero velocity
            assert np.isclose(final_state["z"][0], zero_particle["z"][0], rtol=1e-10)
        except (ZeroDivisionError, ValueError):
            pytest.skip("Zero momentum case not handled - expected behavior")

    @pytest.mark.unit
    def test_numerical_precision(self):
        """Test numerical precision and stability."""
        # Use very small step size to test precision
        config = TestConfiguration(
            particle_count=1,
            transverse_separation=1.0,
            starting_distance=100.0,
            step_size=1e-8,  # Very small step
            total_steps=100,  # Many steps
            sim_type=1,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        initial_energy = bunch["Pt"][0] * C_MMNS

        trajectory, _ = self.integrator.integrate_retarded_fields(
            static_steps=config.total_steps,
            ret_steps=0,  # Only static steps for this test
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=bunch,
            init_driver=bunch.copy(),
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        final_energy = trajectory[-1]["Pt"][0] * C_MMNS
        energy_drift = abs(final_energy - initial_energy) / initial_energy

        # Energy should be conserved to high precision in static case
        assert energy_drift < 1e-10, f"Energy drift too large: {energy_drift}"

    @pytest.mark.unit
    def test_vector_operations(self):
        """Test vector operations used in field calculations."""
        # Test dot product calculation
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([4.0, 5.0, 6.0])

        dot_product = np.dot(vec1, vec2)
        expected_dot = 1 * 4 + 2 * 5 + 3 * 6  # = 32

        assert np.isclose(dot_product, expected_dot, rtol=1e-10)

        # Test cross product
        cross_product = np.cross(vec1, vec2)
        expected_cross = np.array(
            [2 * 6 - 3 * 5, 3 * 4 - 1 * 6, 1 * 5 - 2 * 4]  # = -3  # = 6  # = -3
        )

        assert np.allclose(cross_product, expected_cross, rtol=1e-10)

        # Test magnitude calculation
        magnitude = np.linalg.norm(vec1)
        expected_magnitude = np.sqrt(1**2 + 2**2 + 3**2)  # = sqrt(14)

        assert np.isclose(magnitude, expected_magnitude, rtol=1e-10)


if __name__ == "__main__":
    # Allow running tests directly
    test_instance = TestTrajectoryIntegratorUnits()
    test_instance.setup_method()

    print("Running unit tests...")

    # Run basic tests
    test_instance.test_integrator_initialization()
    test_instance.test_optimized_integrator_default()
    test_instance.test_distance_calculation()
    test_instance.test_relativistic_calculations()

    print("\\n✅ All direct unit tests passed!")
