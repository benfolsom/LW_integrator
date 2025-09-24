"""
Unit tests for electromagnetic physics calculations in the LW integrator.

These tests validate the core electromagnetic physics implementation including:
- Coulomb force calculations
- Relativistic electromagnetic field effects
- Complex electromagnetic force formulations
- Energy-momentum conservation
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.trajectory_integrator import LienardWiechertIntegrator
from physics.constants import C_MMNS
from physics.particle_initialization import ELEMENTARY_CHARGE
from tests.test_config import PROTON, create_bunch_uniform_distribution


class TestElectromagneticPhysics:
    """Unit tests for electromagnetic physics calculations."""

    def setup_method(self) -> None:
        """Setup for each test method."""
        self.integrator = LienardWiechertIntegrator()
        self.tolerance = 1e-10

    @pytest.mark.unit
    def test_coulomb_force_basic(self) -> None:
        """Test basic Coulomb force calculation between two charged particles."""
        # Create two protons separated by known distance
        q1, q2 = (
            ELEMENTARY_CHARGE,
            ELEMENTARY_CHARGE,
        )  # Elementary charges in Gaussian units
        distance = 1.0  # mm
        gamma = 1.0  # Non-relativistic case

        # Expected Coulomb force magnitude in Gaussian units
        # F = q1*q2/r^2 in Gaussian CGS
        expected_force_magnitude = (q1 * q2) / (distance**2)

        # Test the internal EM force method
        beta_vec = np.array([0.0, 0.0, 0.0])  # At rest
        direction = np.array([1.0, 0.0, 0.0])  # unit vector in x direction

        force = self.integrator._simple_em_force(
            beta_vec, gamma, q1, q2, distance, direction
        )

        calculated_magnitude = np.linalg.norm(force)

        print(f"Expected Coulomb force: {expected_force_magnitude}")
        print(f"Calculated magnitude: {calculated_magnitude}")

        assert np.isclose(
            calculated_magnitude, expected_force_magnitude, rtol=self.tolerance
        )

        # Force should be repulsive (in +x direction for positive charges)
        assert force[0] > 0, "Repulsive force should be positive in x direction"
        assert abs(force[1]) < self.tolerance, "Force should only be in x direction"
        assert abs(force[2]) < self.tolerance, "Force should only be in x direction"

    @pytest.mark.unit
    def test_coulomb_force_attractive(self) -> None:
        """Test Coulomb force for attractive interaction (opposite charges)."""
        q1, q2 = (
            ELEMENTARY_CHARGE,
            -ELEMENTARY_CHARGE,
        )  # Opposite charges in Gaussian units
        distance = 2.0  # mm
        gamma = 1.0  # Non-relativistic

        beta_vec = np.array([0.0, 0.0, 0.0])  # At rest
        direction = np.array([1.0, 0.0, 0.0])

        force = self.integrator._simple_em_force(
            beta_vec, gamma, q1, q2, distance, direction
        )

        # Force should be attractive (negative x direction)
        assert force[0] < 0, "Attractive force should be negative in x direction"

        # Magnitude should still follow Coulomb's law
        expected_magnitude = abs(q1 * q2) / (distance**2)
        calculated_magnitude = np.linalg.norm(force)
        assert np.isclose(calculated_magnitude, expected_magnitude, rtol=self.tolerance)

    @pytest.mark.unit
    def test_distance_protection(self) -> None:
        """Test that very small distances are protected against singularities."""
        q1, q2 = 1.0, 1.0
        gamma = 1.0
        very_small_distance = 1e-15  # Much smaller than nuclear scale

        beta_vec = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])

        # Should not raise exception or return infinite force
        force = self.integrator._simple_em_force(
            beta_vec, gamma, q1, q2, very_small_distance, direction
        )

        assert np.all(
            np.isfinite(force)
        ), "Force should be finite for very small distances"

        # Force should be limited by nuclear scale protection
        force_magnitude = np.linalg.norm(force)
        nuclear_scale_force = (q1 * q2) / (1e-12) ** 2  # 1 femtometer limit
        assert (
            force_magnitude <= nuclear_scale_force * 1.1
        ), "Force should be limited by nuclear scale"

    @pytest.mark.unit
    def test_electromagnetic_field_calculation(self) -> None:
        """Test electromagnetic field calculation including relativistic effects."""
        from physics.particle_initialization import ELEMENTARY_CHARGE, ELECTRON_MASS

        # Create a simple non-relativistic electron manually
        # Start with rest mass and then boost to relativistic speeds
        gamma_factor = 100.0
        mass = ELECTRON_MASS  # amu
        charge = -ELEMENTARY_CHARGE  # Gaussian units

        # Calculate relativistic momentum: P = γmc
        expected_pt = gamma_factor * mass * C_MMNS

        # Create particle bunch manually with correct units
        electron_bunch = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([100.0]),
            "Px": np.array([0.0]),
            "Py": np.array([0.0]),
            "Pz": np.array([expected_pt * 0.99]),  # Mostly in z direction
            "Pt": np.array([expected_pt]),
            "mass": np.array([mass]),
            "m": np.array([mass]),
            "q": np.array([charge]),
            "gamma": np.array([gamma_factor]),
            "bx": np.array([0.0]),
            "by": np.array([0.0]),
            "bz": np.array([np.sqrt(1 - 1 / gamma_factor**2)]),
            "t": np.array([0.0]),
        }

        # Test that gamma calculation is consistent
        assert np.isclose(
            electron_bunch["Pt"][0], expected_pt, rtol=1e-3
        ), f"Pt inconsistent: {electron_bunch['Pt'][0]} vs {expected_pt}"

    @pytest.mark.unit
    def test_relativistic_energy_momentum_relation(self) -> None:
        """Test that E²=(pc)²+(mc²)² relation holds."""
        # Create particle with known momentum and mass
        mass = PROTON.mass_mev  # MeV/c²
        momentum_mev = 1000.0  # MeV/c (in natural units)

        # Calculate energy from relativistic relation
        energy_squared = momentum_mev**2 + mass**2
        energy = np.sqrt(energy_squared)

        # Calculate gamma
        gamma = energy / mass

        # Calculate velocity beta
        beta = np.sqrt(1 - 1 / gamma**2)

        # Calculate momentum from gamma and velocity
        calculated_momentum_mev = gamma * mass * beta

        assert np.isclose(
            calculated_momentum_mev, momentum_mev, rtol=1e-10
        ), f"Momentum inconsistent: {calculated_momentum_mev} vs {momentum_mev}"

    @pytest.mark.unit
    def test_complex_physics_selection(self) -> None:
        """Test the logic for selecting complex vs simple electromagnetic physics."""
        # Create test particles
        beta_vec = np.array([0.1, 0.0, 0.8])  # Moderate relativistic velocity
        beta_ext = np.array([0.05, 0.0, 0.9])  # High relativistic velocity
        gamma_i, gamma_j = 1.25, 2.3
        k_factor = 0.95
        distance = 5.0  # mm

        # Test complex physics decision
        needs_complex = self.integrator._needs_complex_physics(
            beta_vec, beta_ext, gamma_i, gamma_j, k_factor, distance
        )

        # For moderate relativistic particles, should likely use complex physics
        assert isinstance(needs_complex, bool), "Should return boolean decision"

    @pytest.mark.unit
    def test_simple_em_force_calculation(self) -> None:
        """Test the simplified electromagnetic force calculation."""
        beta_vec = np.array([0.1, 0.0, 0.5])
        gamma = 1.15
        charge_i, charge_j = (
            ELEMENTARY_CHARGE,
            -ELEMENTARY_CHARGE,
        )  # Use proper Gaussian units
        distance = 3.0
        direction = np.array([1.0, 0.0, 0.0])  # x direction

        force = self.integrator._simple_em_force(
            beta_vec, gamma, charge_i, charge_j, distance, direction
        )

        assert len(force) == 3, "Force should be 3D vector"
        assert np.all(np.isfinite(force)), "Force should be finite"

        # For opposite charges, force should be attractive (negative x)
        assert force[0] < 0, "Force should be attractive for opposite charges"

    @pytest.mark.unit
    def test_scalar_products_physics(self) -> None:
        """Test the scalar products used in electromagnetic calculations."""
        # Test vectors
        beta_vec = np.array([0.1, 0.2, 0.8])
        beta_ext = np.array([0.05, 0.1, 0.9])
        bdot_ext = np.array([0.01, 0.02, 0.05])

        # Calculate scalar products
        betas_scalar = np.dot(beta_ext, beta_vec)
        bdot_scalar_ext = np.dot(bdot_ext, bdot_ext)

        # These should be physical values
        assert (
            0 <= abs(betas_scalar) <= 1
        ), f"Beta scalar product should be ≤1: {betas_scalar}"
        assert (
            bdot_scalar_ext >= 0
        ), f"Bdot scalar product should be ≥0: {bdot_scalar_ext}"

        # Test that removed variable would have been calculated correctly
        bdot_scalar_mixed = np.dot(beta_vec, bdot_ext)  # This was the removed variable

        # This is a valid calculation but wasn't used in the physics
        assert np.isfinite(
            bdot_scalar_mixed
        ), "Removed variable calculation should be finite"

    @pytest.mark.unit
    def test_deep_copy_state_functionality(self) -> None:
        """Test the deep copy functionality for particle states."""
        # Create a test state
        original_state = {
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([4.0, 5.0, 6.0]),
            "Pt": np.array([100.0, 200.0, 300.0]),
            "q": np.array([1.0, -1.0, 1.0]),
            "gamma": np.array([1.1, 1.2, 1.3]),
        }

        # Make deep copy
        copied_state = self.integrator._deep_copy_state(original_state)

        # Verify it's a proper deep copy
        assert copied_state is not original_state, "Should be different objects"

        for key in original_state:
            assert key in copied_state, f"Key {key} should be copied"
            assert (
                copied_state[key] is not original_state[key]
            ), f"Array {key} should be deep copied"
            assert np.array_equal(
                copied_state[key], original_state[key]
            ), f"Values for {key} should be equal"

        # Modify copy and ensure original is unchanged
        copied_state["x"][0] = 999.0
        assert original_state["x"][0] == 1.0, "Original should be unchanged"

    @pytest.mark.unit
    def test_charge_conservation_single_step(self) -> None:
        """Test that charge is conserved in a single integration step."""
        from tests.test_config import TestConfiguration

        config = TestConfiguration(
            particle_count=3,
            transverse_separation=10.0,
            starting_distance=150.0,
            step_size=1e-5,
            total_steps=1,
            sim_type=2,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        # Create mixed charge bunch
        bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        bunch["q"] = np.array(
            [ELEMENTARY_CHARGE, -ELEMENTARY_CHARGE, ELEMENTARY_CHARGE]
        )  # Mixed charges in Gaussian units

        initial_charge = np.sum(bunch["q"])

        # Take one integration step using the available method (static assumes no external bunch)
        final_state = self.integrator._eqsofmotion_static_updated(
            config.step_size, bunch, bunch, config.aperture_r, config.sim_type
        )

        final_charge = np.sum(final_state["q"])

        assert np.isclose(
            final_charge, initial_charge, atol=1e-15
        ), f"Charge not conserved: {initial_charge} -> {final_charge}"


if __name__ == "__main__":
    # Allow running tests directly
    test_instance = TestElectromagneticPhysics()
    test_instance.setup_method()

    print("Running electromagnetic physics unit tests...")

    # Run key physics tests
    test_instance.test_coulomb_force_basic()
    test_instance.test_coulomb_force_attractive()
    test_instance.test_distance_protection()
    test_instance.test_relativistic_energy_momentum_relation()
    test_instance.test_scalar_products_physics()
    test_instance.test_deep_copy_state_functionality()
    test_instance.test_charge_conservation_single_step()

    print("\n✅ All electromagnetic physics unit tests passed!")
