"""Unit tests for relativistic position updates and self-consistency physics.

This module tests the fundamental physics of position updates in coordinate time
and the self-consistency iteration scheme for ultra-relativistic particles.
"""

import numpy as np
import pytest

from core.equations import (
    _calculate_gamma_from_beta,
    _limit_beta_magnitude,
)

# Physical constants (matching core/constants.py)
C_MMNS = 299.792458  # mm/ns
ELECTRON_MASS_MEV = 0.5109989461  # MeV/c^2


class TestPositionUpdatePhysics:
    """Test that position updates follow correct relativistic formulas."""

    def test_position_from_momentum_includes_gamma_factor(self):
        """Verify position update includes 1/γ factor: Δx = (P/(γm))h."""
        gamma = 10000.0
        mass = ELECTRON_MASS_MEV
        h = 1e-3  # mm/c

        # For ultra-relativistic particle, β ≈ 1
        beta_true = np.sqrt(1.0 - 1.0 / (gamma**2))
        velocity = beta_true * C_MMNS

        # Kinetic momentum P = γ·m·v
        P_kinetic = gamma * mass * velocity

        # Correct position update: Δx = (h/(γ·m))·P
        delta_x_correct = (h / (gamma * mass)) * P_kinetic

        # This should equal v·h
        delta_x_expected = velocity * h

        np.testing.assert_allclose(
            delta_x_correct,
            delta_x_expected,
            rtol=1e-10,
            err_msg="Position update Δx = (h/(γm))P should equal v·h",
        )

    def test_velocity_from_position_no_gamma_factor(self):
        """Verify velocity calculation: β = Δx/(c·h), no γ factor."""
        gamma = 20000.0
        mass = ELECTRON_MASS_MEV
        h = 1e-3  # mm/c

        beta_true = np.sqrt(1.0 - 1.0 / (gamma**2))
        velocity = beta_true * C_MMNS
        P_kinetic = gamma * mass * velocity

        # Position update with 1/γ
        delta_x = (h / (gamma * mass)) * P_kinetic

        # Velocity calculation WITHOUT γ
        beta_calculated = delta_x / (C_MMNS * h)

        np.testing.assert_allclose(
            beta_calculated,
            beta_true,
            rtol=1e-10,
            err_msg="β = Δx/(c·h) should recover true velocity",
        )

    def test_wrong_formula_gives_wrong_velocity(self):
        """Show that missing 1/γ in position update causes β to be wrong."""
        gamma = 10000.0
        mass = ELECTRON_MASS_MEV
        h = 1e-3  # mm/c

        beta_true = np.sqrt(1.0 - 1.0 / (gamma**2))
        velocity = beta_true * C_MMNS
        P_kinetic = gamma * mass * velocity

        # WRONG position update (missing 1/γ)
        delta_x_wrong = (h / mass) * P_kinetic

        # This gives displacement that's γ times too large
        assert delta_x_wrong / (velocity * h) == pytest.approx(gamma, rel=1e-6), (
            "Wrong formula gives Δx = γ·v·h instead of v·h"
        )

        # If we compute β from this wrong Δx without γ correction
        beta_wrong = delta_x_wrong / (C_MMNS * h)

        # We get β ≈ γ (exceeds 1!)
        assert beta_wrong > 1.0, "Wrong formula gives β > 1"

    def test_compensating_errors_cancel(self):
        """Show that two wrongs (missing 1/γ in position and velocity) make a right."""
        gamma = 5000.0
        mass = ELECTRON_MASS_MEV
        h = 1e-3  # mm/c

        beta_true = np.sqrt(1.0 - 1.0 / (gamma**2))
        velocity = beta_true * C_MMNS
        P_kinetic = gamma * mass * velocity

        # WRONG position update (missing 1/γ)
        delta_x_wrong = (h / mass) * P_kinetic

        # WRONG velocity calc (dividing by γ compensates)
        beta_compensated = delta_x_wrong / (C_MMNS * h * gamma)

        # These errors cancel!
        np.testing.assert_allclose(
            beta_compensated,
            beta_true,
            rtol=1e-10,
            err_msg="Compensating errors accidentally give right answer",
        )

        # But this only works if the SAME gamma is used in both places


class TestSelfConsistencyPhysics:
    """Test self-consistency iteration convergence properties."""

    def test_converges_when_gamma_is_correct(self):
        """Verify that iteration converges immediately when using correct γ."""
        gamma_true = 15000.0
        mass = ELECTRON_MASS_MEV
        h = 1e-3  # mm/c

        beta_true = np.sqrt(1.0 - 1.0 / (gamma_true**2))
        velocity = beta_true * C_MMNS
        P_kinetic = gamma_true * mass * velocity

        # Use correct gamma throughout
        delta_x = (h / (gamma_true * mass)) * P_kinetic
        beta_calc = delta_x / (C_MMNS * h)

        gamma_from_velocity = _calculate_gamma_from_beta(beta_calc, 0.0, 0.0)

        # Should match within numerical precision
        relative_error = abs(gamma_from_velocity - gamma_true) / gamma_true
        assert relative_error < 1e-6, "Should converge immediately with correct gamma"

    def test_iterates_to_convergence_from_wrong_gamma(self):
        """Verify iteration converges even starting from wrong γ."""
        gamma_true = 20000.0
        gamma_wrong = 10000.0  # Start with 2× error
        mass = ELECTRON_MASS_MEV
        h = 1e-3  # mm/c

        beta_true = np.sqrt(1.0 - 1.0 / (gamma_true**2))
        velocity = beta_true * C_MMNS
        P_kinetic = gamma_true * mass * velocity

        # Iteration 0: Use wrong gamma
        delta_x_0 = (h / (gamma_wrong * mass)) * P_kinetic
        beta_0 = delta_x_0 / (C_MMNS * h)

        # Beta is too large, will be clamped
        if beta_0 >= 1.0:
            beta_0_lim, _, _ = _limit_beta_magnitude(beta_0, 0.0, 0.0)
        else:
            beta_0_lim = beta_0

        gamma_from_velocity_0 = _calculate_gamma_from_beta(beta_0_lim, 0.0, 0.0)

        # Won't match yet
        error_0 = abs(gamma_from_velocity_0 - gamma_true) / gamma_true
        assert error_0 > 1e-3, "Should not converge on first iteration with wrong gamma"

        # Iteration 1: Use gamma_true (from energy)
        delta_x_1 = (h / (gamma_true * mass)) * P_kinetic
        beta_1 = delta_x_1 / (C_MMNS * h)
        beta_1_lim, _, _ = _limit_beta_magnitude(beta_1, 0.0, 0.0)

        gamma_from_velocity_1 = _calculate_gamma_from_beta(beta_1_lim, 0.0, 0.0)

        # Should converge now
        error_1 = abs(gamma_from_velocity_1 - gamma_true) / gamma_true
        assert error_1 < 1e-6, "Should converge on second iteration with correct gamma"

    def test_extreme_energy_jump_recovers(self):
        """Test recovery from large gamma mismatch (simulating energy jump)."""
        gamma_old = 1000.0
        gamma_new = 20000.0  # 20× jump!
        mass = ELECTRON_MASS_MEV
        h = 1e-3  # mm/c

        beta_new = np.sqrt(1.0 - 1.0 / (gamma_new**2))
        velocity_new = beta_new * C_MMNS
        P_kinetic_new = gamma_new * mass * velocity_new

        # Iteration with old gamma fails badly
        delta_x_old = (h / (gamma_old * mass)) * P_kinetic_new
        beta_old = delta_x_old / (C_MMNS * h)

        assert beta_old > 1.0, "Using old gamma gives superluminal velocity"

        beta_old_lim, _, _ = _limit_beta_magnitude(beta_old, 0.0, 0.0)
        gamma_from_velocity_old = _calculate_gamma_from_beta(beta_old_lim, 0.0, 0.0)

        # Huge error
        error_old = abs(gamma_from_velocity_old - gamma_new) / gamma_new
        assert error_old > 0.5, "Should have large error with old gamma"

        # But next iteration with correct gamma recovers
        delta_x_new = (h / (gamma_new * mass)) * P_kinetic_new
        beta_new_calc = delta_x_new / (C_MMNS * h)
        beta_new_lim, _, _ = _limit_beta_magnitude(beta_new_calc, 0.0, 0.0)

        gamma_from_velocity_new = _calculate_gamma_from_beta(beta_new_lim, 0.0, 0.0)

        # Converges!
        error_new = abs(gamma_from_velocity_new - gamma_new) / gamma_new
        assert error_new < 1e-6, "Should recover in one iteration after energy jump"

    @pytest.mark.parametrize(
        "gamma_true",
        [100, 1000, 10000, 50000],
    )
    def test_self_consistency_at_various_gamma(self, gamma_true):
        """Test self-consistency holds across range of gamma values."""
        mass = ELECTRON_MASS_MEV
        h = 1e-3  # mm/c

        beta_true = np.sqrt(1.0 - 1.0 / (gamma_true**2))
        velocity = beta_true * C_MMNS
        P_kinetic = gamma_true * mass * velocity

        # Correct formulas
        delta_x = (h / (gamma_true * mass)) * P_kinetic
        beta_calc = delta_x / (C_MMNS * h)
        beta_lim, _, _ = _limit_beta_magnitude(beta_calc, 0.0, 0.0)

        gamma_from_velocity = _calculate_gamma_from_beta(beta_lim, 0.0, 0.0)

        # Energy-based gamma
        Pt = gamma_true * mass * C_MMNS
        gamma_from_energy = Pt / (mass * C_MMNS)

        # Should be self-consistent
        relative_error = (
            abs(gamma_from_velocity - gamma_from_energy) / gamma_from_energy
        )
        assert relative_error < 1e-6, f"Self-consistency failed at γ={gamma_true}"


class TestBetaClamping:
    """Test that beta limiting works correctly and doesn't interfere with physics."""

    def test_beta_below_limit_not_clamped(self):
        """Verify that reasonable beta values are not modified."""
        beta_values = [0.5, 0.9, 0.99, 0.999, 0.9999]

        for beta in beta_values:
            bx, by, bz = _limit_beta_magnitude(0.0, 0.0, beta)
            beta_magnitude = np.sqrt(bx**2 + by**2 + bz**2)

            assert beta_magnitude == pytest.approx(beta, abs=1e-15), (
                f"β={beta} should not be modified"
            )

    def test_beta_above_limit_is_clamped(self):
        """Verify that superluminal velocities are clamped."""
        beta_superluminal = [1.0, 1.1, 2.0, 10.0]

        for beta in beta_superluminal:
            bx, by, bz = _limit_beta_magnitude(0.0, 0.0, beta)
            beta_magnitude = np.sqrt(bx**2 + by**2 + bz**2)

            assert beta_magnitude < 1.0, f"β={beta} should be clamped below c"
            # Should be very close to 1 but not equal
            assert beta_magnitude > 0.99, "Clamped beta should be close to 1"

    def test_clamped_beta_gives_finite_gamma(self):
        """Verify clamped beta produces finite gamma."""
        beta_superluminal = 5.0

        bx, by, bz = _limit_beta_magnitude(0.0, 0.0, beta_superluminal)
        gamma = _calculate_gamma_from_beta(bx, by, bz)

        assert np.isfinite(gamma), "Gamma from clamped beta should be finite"
        assert gamma > 1.0, "Gamma should be greater than 1"

    def test_extreme_gamma_not_artificially_limited(self):
        """Verify that high but physical gamma values work correctly."""
        # Test gamma up to 1e8 (near beta clamping limit)
        gamma_extreme = 6.7e7

        beta = np.sqrt(1.0 - 1.0 / (gamma_extreme**2))

        bx, by, bz = _limit_beta_magnitude(0.0, 0.0, beta)
        beta_magnitude = np.sqrt(bx**2 + by**2 + bz**2)

        # Should not be clamped
        np.testing.assert_allclose(
            beta_magnitude,
            beta,
            rtol=1e-10,
            err_msg=f"Physical β at γ={gamma_extreme:.2e} should not be clamped",
        )

        gamma_recovered = _calculate_gamma_from_beta(bx, by, bz)

        # At β this close to 1, float64 has limited representable spacing near
        # 1.0; the recovered γ should stay in the intended ultra-relativistic
        # regime without being clamped to a much lower ceiling.
        np.testing.assert_allclose(
            gamma_recovered,
            gamma_extreme,
            rtol=2e-3,
            err_msg=f"Should accurately recover γ={gamma_extreme:.2e}",
        )
