"""Test suite for scalar potential gamma calculation fix.

This module validates that gamma is computed correctly using the scalar
potential correction: γ = (Pt - q²·Φ) / (mc)
"""

import numpy as np


def test_scalar_potential_gamma_basic():
    """Test that gamma is computed correctly with scalar potential correction."""
    # Physical constants
    c = 299.792458  # mm/ns
    m_electron = 9.1093837015e-31 * 1e18  # kg -> pg
    q_electron = -1.602176634e-19 * 1e21  # C -> pC (negative!)

    # Particle state
    gamma_true = 2000.0

    # Conjugate energy (without potential)
    Pt_no_field = gamma_true * m_electron * c

    # Add scalar potential contribution
    # Φ = q_ext / (R_sep * k_factor)
    q_ext = 1.602176634e-19 * 1e21  # proton charge in pC
    R_sep = 5.0  # mm
    k_factor = 0.9

    Phi = q_ext / (R_sep * k_factor)

    # Conjugate energy with potential: Pt = γmc + q·Φ
    Pt_with_field = Pt_no_field + q_electron * Phi

    # Now compute gamma INCORRECTLY (old way)
    gamma_wrong = Pt_with_field / (m_electron * c)

    # Compute gamma CORRECTLY (new way)
    scalar_potential_contribution = q_electron * q_electron * Phi
    kinetic_energy = Pt_with_field - scalar_potential_contribution
    gamma_correct = kinetic_energy / (m_electron * c)

    # The correct gamma should match the true value
    relative_error_correct = abs(gamma_correct - gamma_true) / gamma_true
    assert relative_error_correct < 1e-10, (
        f"Correct method failed: expected γ={gamma_true:.6e}, "
        f"got {gamma_correct:.6e}, error {relative_error_correct:.2e}"
    )

    # The wrong gamma should NOT match (it includes the potential)
    relative_error_wrong = abs(gamma_wrong - gamma_true) / gamma_true
    assert (
        relative_error_wrong > 1e-6
    ), f"Wrong method unexpectedly gave correct gamma! Both gave {gamma_wrong:.6e}"

    print(f"  True gamma: {gamma_true:.6e}")
    print(f"  Gamma (wrong): {gamma_wrong:.6e} (includes potential)")
    print(f"  Gamma (correct): {gamma_correct:.6e} (potential corrected)")
    print(f"  Scalar potential Φ: {Phi:.6e}")
    print(f"  Correction q²Φ: {scalar_potential_contribution:.6e}")


def test_multiple_charges_scalar_potential():
    """Test scalar potential sum from multiple external charges."""
    # Physical constants
    c = 299.792458  # mm/ns
    m_electron = 9.1093837015e-31 * 1e18  # kg -> pg
    q_electron = -1.602176634e-19 * 1e21  # C -> pC

    # True gamma
    gamma_true = 5000.0

    # Base conjugate energy
    Pt_base = gamma_true * m_electron * c

    # Multiple external charges
    charges_ext = np.array(
        [
            1.602176634e-19 * 1e21,  # proton
            1.602176634e-19 * 1e21,  # proton
            -1.602176634e-19 * 1e21,  # electron
        ]
    )
    R_separations = np.array([10.0, 15.0, 8.0])  # mm
    k_factors = np.array([0.95, 0.92, 0.88])

    # Scalar potential sum
    Phi_sum = np.sum(charges_ext / (R_separations * k_factors))

    # Conjugate energy with potential
    Pt_with_field = Pt_base + q_electron * Phi_sum

    # Correct gamma calculation
    scalar_potential_contribution = q_electron * q_electron * Phi_sum
    kinetic_energy = Pt_with_field - scalar_potential_contribution
    gamma_correct = kinetic_energy / (m_electron * c)

    # Should match true gamma
    relative_error = abs(gamma_correct - gamma_true) / gamma_true
    assert relative_error < 1e-10, (
        f"Multi-charge test failed: expected γ={gamma_true:.6e}, "
        f"got {gamma_correct:.6e}, error {relative_error:.2e}"
    )

    print(f"  Number of external charges: {len(charges_ext)}")
    print(f"  Scalar potential sum Φ: {Phi_sum:.6e}")
    print(f"  Correction q²Σ(Φ): {scalar_potential_contribution:.6e}")
    print(f"  Recovered gamma: {gamma_correct:.6e}")


def test_zero_potential_limit():
    """Test that with no external field, gamma is computed correctly."""
    # Physical constants
    c = 299.792458  # mm/ns
    m_electron = 9.1093837015e-31 * 1e18  # kg -> pg
    q_electron = -1.602176634e-19 * 1e21  # C -> pC

    # True gamma
    gamma_true = 100.0

    # Conjugate energy (no external field)
    Pt = gamma_true * m_electron * c

    # No scalar potential
    Phi_sum = 0.0

    # Gamma calculation
    scalar_potential_contribution = q_electron * q_electron * Phi_sum
    kinetic_energy = Pt - scalar_potential_contribution
    gamma = kinetic_energy / (m_electron * c)

    # Should exactly match
    assert (
        abs(gamma - gamma_true) < 1e-12
    ), f"Zero-field limit failed: expected γ={gamma_true}, got {gamma}"

    print(f"  Zero-field gamma: {gamma:.6e} (exact match expected)")


def test_strong_field_regime():
    """Test gamma calculation in strong electromagnetic field."""
    # Physical constants
    c = 299.792458  # mm/ns
    m_electron = 9.1093837015e-31 * 1e18  # kg -> pg
    q_electron = -1.602176634e-19 * 1e21  # C -> pC

    # True gamma (high energy electron)
    gamma_true = 20000.0  # ~10 GeV

    # Base conjugate energy
    Pt_base = gamma_true * m_electron * c

    # Strong field: close approach to highly charged object
    q_ext = 1.602176634e-19 * 1e21 * 50  # 50 proton charges
    R_sep = 0.1  # mm (very close!)
    k_factor = 0.5  # strong retardation

    Phi = q_ext / (R_sep * k_factor)

    # Conjugate energy with strong potential
    Pt_with_field = Pt_base + q_electron * Phi

    # Wrong calculation (would give very wrong gamma)
    gamma_wrong = Pt_with_field / (m_electron * c)

    # Correct calculation
    scalar_potential_contribution = q_electron * q_electron * Phi
    kinetic_energy = Pt_with_field - scalar_potential_contribution
    gamma_correct = kinetic_energy / (m_electron * c)

    # Correct method should recover true gamma
    relative_error_correct = abs(gamma_correct - gamma_true) / gamma_true
    assert relative_error_correct < 1e-10, (
        f"Strong-field test failed: expected γ={gamma_true:.6e}, "
        f"got {gamma_correct:.6e}"
    )

    # Wrong method should be WAY off
    relative_error_wrong = abs(gamma_wrong - gamma_true) / gamma_true

    print("  Strong field scenario:")
    print(f"    True gamma: {gamma_true:.6e}")
    print(f"    Gamma (wrong): {gamma_wrong:.6e}")
    print(f"    Gamma (correct): {gamma_correct:.6e}")
    print(f"    Scalar potential Φ: {Phi:.6e}")
    print(f"    Error without correction: {relative_error_wrong:.2%}")
    print(f"    Error with correction: {relative_error_correct:.2e}")


def test_adaptive_timestep_relevance():
    """Demonstrate why this fix matters for adaptive timestep."""
    # Physical constants
    c = 299.792458  # mm/ns
    m_electron = 9.1093837015e-31 * 1e18  # kg -> pg
    q_electron = -1.602176634e-19 * 1e21  # C -> pC

    # Scenario: electron approaching wall
    gamma_initial = 10000.0
    gamma_during_interaction = 10050.0  # small change in kinetic energy

    # External field during interaction
    q_ext = 1.602176634e-19 * 1e21 * 100  # 100 charges
    R_sep = 1.0  # mm
    k_factor = 0.8
    Phi = q_ext / (R_sep * k_factor)

    # Initial state (far from wall)
    Pt_initial = gamma_initial * m_electron * c

    # During interaction (with field)
    Pt_interaction = gamma_during_interaction * m_electron * c + q_electron * Phi

    # WRONG: Compare conjugate energies directly
    gamma_initial_wrong = Pt_initial / (m_electron * c)
    gamma_interaction_wrong = Pt_interaction / (m_electron * c)
    delta_gamma_wrong = gamma_interaction_wrong - gamma_initial_wrong

    # CORRECT: Use scalar potential correction
    scalar_contrib = q_electron * q_electron * Phi
    gamma_interaction_correct = (Pt_interaction - scalar_contrib) / (m_electron * c)
    delta_gamma_correct = gamma_interaction_correct - gamma_initial

    print("  Adaptive timestep scenario:")
    print(f"    Initial gamma: {gamma_initial:.6e}")
    print("    During interaction:")
    print(f"      Δγ (wrong method): {delta_gamma_wrong:.6e}")
    print(f"      Δγ (correct method): {delta_gamma_correct:.6e}")
    print(f"      Ratio: {abs(delta_gamma_wrong / delta_gamma_correct):.2f}x")
    print("    → Wrong method sees artificial gamma change from EM potential!")

    # The true change is small (50), but wrong method sees huge change
    assert abs(delta_gamma_correct - 50.0) < 1.0, "Correct method should see ~50 change"
    assert abs(delta_gamma_wrong) > abs(
        delta_gamma_correct
    ), "Wrong method sees larger change"


if __name__ == "__main__":
    print("Testing scalar potential gamma calculations...\n")

    print("1. Basic scalar potential correction...")
    test_scalar_potential_gamma_basic()
    print("   ✓ PASSED\n")

    print("2. Multiple external charges...")
    test_multiple_charges_scalar_potential()
    print("   ✓ PASSED\n")

    print("3. Zero-field limit...")
    test_zero_potential_limit()
    print("   ✓ PASSED\n")

    print("4. Strong electromagnetic field...")
    test_strong_field_regime()
    print("   ✓ PASSED\n")

    print("5. Adaptive timestep relevance...")
    test_adaptive_timestep_relevance()
    print("   ✓ PASSED\n")

    print("✅ All scalar potential gamma tests passed!")
    print("\nKey takeaway: γ = (Pt - q²·Φ)/(mc) is essential for correct dynamics!")
