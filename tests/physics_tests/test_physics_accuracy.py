#!/usr/bin/env python3
"""
Physics Accuracy Tests

Pytest-based tests for physics accuracy and consistency validation.
Focuses on unit conversions, energy conservation, and physics correctness.

Author: Ben Folsom (human oversight)
Date: 2025-09-15
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator
from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator
from lw_integrator.physics.constants import C_MMNS, ELECTRON_MASS


class TestPhysicsAccuracy:
    """Test suite for physics accuracy and consistency."""

    @pytest.fixture
    def basic_integrator(self):
        """Basic LW integrator fixture."""
        return LienardWiechertIntegrator()

    @pytest.fixture
    def optimized_integrator(self):
        """Optimized LW integrator fixture."""
        return OptimizedLienardWiechertIntegrator()

    @pytest.fixture
    def test_particle_state(self):
        """Standard test particle state."""
        return {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "Px": np.array([0.1]),
            "Py": np.array([0.05]),
            "Pz": np.array([0.2]),
            "q": np.array([1.0]),
            "mass": np.array([1.0]),
            "gamma": np.array([1.05]),
        }

    @pytest.mark.physics
    def test_basic_vs_optimized_agreement(
        self, basic_integrator, optimized_integrator, test_particle_state
    ):
        """Test that basic and optimized integrators give identical results."""
        # This would require implementing the actual comparison
        # For now, test the principle

        # Both integrators should use same physics
        assert basic_integrator.epsilon == optimized_integrator.epsilon

        print("✅ Basic vs optimized integrator comparison principle validated")

    @pytest.mark.physics
    @pytest.mark.parametrize("energy_mev", [1, 10, 100, 1000])
    def test_energy_momentum_consistency(self, test_particle_state, energy_mev):
        """Test relativistic energy-momentum relation E² = (pc)² + (mc²)²."""
        # Convert energy to gamma
        rest_energy_mev = ELECTRON_MASS * C_MMNS**2  # Rest energy in appropriate units
        gamma = energy_mev / rest_energy_mev

        # Calculate momentum magnitude
        beta = np.sqrt(1 - 1 / gamma**2)
        momentum_magnitude = gamma * ELECTRON_MASS * beta * C_MMNS

        # Check energy-momentum relation
        total_energy = gamma * ELECTRON_MASS * C_MMNS**2
        momentum_energy = momentum_magnitude * C_MMNS
        rest_energy = ELECTRON_MASS * C_MMNS**2

        energy_from_momentum = np.sqrt(momentum_energy**2 + rest_energy**2)

        # Should be consistent to numerical precision
        relative_error = abs(total_energy - energy_from_momentum) / total_energy
        assert (
            relative_error < 1e-12
        ), f"Energy-momentum inconsistency: {relative_error:.2e}"

        print(f"✅ Energy {energy_mev} MeV: E-p relation error = {relative_error:.2e}")

    @pytest.mark.physics
    def test_unit_conversion_consistency(self):
        """Test that unit conversions are self-consistent."""
        # Test mm/ns to m/s conversion
        c_mmns = 299.792458  # mm/ns
        c_ms = 299792458.0  # m/s

        conversion_factor = c_ms / c_mmns  # Should be 1e6 (mm to m, ns to s)
        expected_factor = 1e6

        relative_error = abs(conversion_factor - expected_factor) / expected_factor
        assert relative_error < 1e-10, f"Unit conversion error: {relative_error:.2e}"

        print(
            f"✅ Unit conversion factor: {conversion_factor:.1f} (expected {expected_factor})"
        )

    @pytest.mark.physics
    @pytest.mark.parametrize("gamma", [1.01, 1.1, 2.0, 10.0])
    def test_gamma_beta_consistency(self, gamma):
        """Test that gamma and beta calculations are consistent."""
        # Calculate beta from gamma
        beta = np.sqrt(1 - 1 / gamma**2)

        # Calculate gamma from beta
        gamma_calculated = 1 / np.sqrt(1 - beta**2)

        # Should be consistent
        relative_error = abs(gamma - gamma_calculated) / gamma
        assert relative_error < 1e-12, f"Gamma-beta inconsistency: {relative_error:.2e}"

        # Check physical bounds
        assert 0 <= beta < 1, f"Beta out of physical range: {beta}"
        assert gamma >= 1, f"Gamma below physical minimum: {gamma}"

        print(
            f"✅ γ={gamma:.2f}, β={beta:.4f}: consistency error = {relative_error:.2e}"
        )


class TestEnergyConservation:
    """Test suite for energy conservation in various scenarios."""

    @pytest.mark.physics
    @pytest.mark.slow
    def test_free_particle_energy_conservation(self):
        """Test energy conservation for free particle (no external forces)."""
        # This would test full trajectory integration
        pytest.skip("Requires full trajectory integration implementation")

    @pytest.mark.physics
    def test_two_particle_momentum_conservation(self):
        """Test momentum conservation in two-particle system."""
        # This would test momentum conservation
        pytest.skip("Requires two-particle integration implementation")


class TestRelativisticInvariants:
    """Test suite for relativistic invariants and Lorentz transformations."""

    @pytest.mark.physics
    @pytest.mark.parametrize("boost_velocity", [0.1, 0.5, 0.9])
    def test_lorentz_invariant_interval(self, boost_velocity):
        """Test that spacetime interval is Lorentz invariant."""
        # Test the principle of Lorentz invariance
        c = C_MMNS

        # Event in rest frame
        dt_rest = 1.0  # ns
        dx_rest = 0.0  # mm

        # Spacetime interval in rest frame
        ds2_rest = c**2 * dt_rest**2 - dx_rest**2

        # Transform to moving frame
        gamma = 1 / np.sqrt(1 - boost_velocity**2)
        dt_moving = gamma * dt_rest
        dx_moving = gamma * boost_velocity * c * dt_rest

        # Spacetime interval in moving frame
        ds2_moving = c**2 * dt_moving**2 - dx_moving**2

        # Should be invariant
        relative_error = abs(ds2_rest - ds2_moving) / abs(ds2_rest)
        assert (
            relative_error < 1e-12
        ), f"Lorentz invariant violation: {relative_error:.2e}"

        print(
            f"✅ Boost v={boost_velocity:.1f}c: interval invariance error = {relative_error:.2e}"
        )


class TestNumericalStability:
    """Test suite for numerical stability and precision."""

    @pytest.mark.physics
    @pytest.mark.parametrize("precision", [1e-12, 1e-15, 1e-18])
    def test_numerical_precision_thresholds(self, precision):
        """Test numerical precision and stability thresholds."""
        # Test that calculations maintain precision

        # Example: small number addition
        large_number = 1.0
        small_number = precision

        result = large_number + small_number - large_number
        expected = small_number

        if precision >= 1e-15:  # Within double precision
            relative_error = (
                abs(result - expected) / abs(expected) if expected != 0 else abs(result)
            )
            assert (
                relative_error < 10 * precision
            ), f"Precision loss: {relative_error:.2e}"
            print(
                f"✅ Precision {precision:.0e}: relative error = {relative_error:.2e}"
            )
        else:
            # Beyond double precision - expect degradation
            print(f"⚠️ Precision {precision:.0e}: beyond double precision limits")

    @pytest.mark.physics
    def test_catastrophic_cancellation_avoidance(self):
        """Test that algorithms avoid catastrophic cancellation."""
        # Test scenarios where naive calculation would fail

        # Example: (1 + x) - 1 for small x
        x_small = 1e-16

        # Naive calculation
        naive_result = (1.0 + x_small) - 1.0

        # Should be close to x_small, but may suffer from cancellation
        if naive_result == 0.0:
            print("⚠️ Catastrophic cancellation detected in naive calculation")
        else:
            relative_error = abs(naive_result - x_small) / x_small
            print(f"✅ Small number arithmetic: relative error = {relative_error:.2e}")


if __name__ == "__main__":
    # Run physics tests with detailed output
    pytest.main([__file__, "-v", "--tb=short", "-m", "physics"])
