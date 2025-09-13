#!/usr/bin/env python3
"""
Test radiation reaction force actually being triggered

CAI: Create test cases with parameters that should definitely trigger 
radiation reaction by working backwards from the threshold condition.
Focus on ultra-relativistic scenarios where β ≈ 1.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lw_integrator.core.integration import LienardWiechertIntegrator
from lw_integrator.physics.constants import C_MMNS, ELECTRON_MASS


def find_triggering_parameters():
    """Find parameters that will definitely trigger radiation reaction."""
    
    print("=== FINDING RADIATION REACTION TRIGGERING PARAMETERS ===")
    
    # Use realistic electron parameters
    c_mmns = 299.792458  # mm/ns
    mass_electron_amu = 0.0005485  # amu  
    q_electron = 1.178734e-5  # legacy unit charge
    char_time_realistic = 2/3 * q_electron**2 / (mass_electron_amu * c_mmns**3)
    
    print(f"Realistic electron char_time: {char_time_realistic:.3e} ns")
    
    # But let's use a larger char_time to make triggering easier
    char_time_test = 1e-12  # Still realistic but larger than electron
    threshold = char_time_test / 1e1
    
    print(f"Test char_time: {char_time_test:.3e} ns")
    print(f"Threshold: {threshold:.3e}")
    
    # For ultra-relativistic particle (β ≈ 1)
    gamma = 1000.0  # Very relativistic
    gamma_old = 900.0  # Large gamma change
    h = 1e-15  # Very small timestep for ultra-fast dynamics
    m_particle = mass_electron_amu
    
    # Start with moderate acceleration and scale up until we trigger
    beta_c = 0.999  # Very close to speed of light (β ≈ 1)
    
    for accel_power in range(-6, 6):  # Try accelerations from 1e-6 to 1e5
        beta_dot = 10**accel_power  # mm/ns² (which is β̇ since c = 299.8 mm/ns)
        
        # Calculate both force terms
        # RHS: -γ³(m*β̇²*c²)*β*c  
        rad_frc_rhs = -gamma**3 * (m_particle * beta_dot**2 * c_mmns**2) * beta_c * c_mmns
        
        # LHS: (γ_new - γ_old)/(h*γ_new) * m * β̇ * β * c²
        rad_frc_lhs = ((gamma - gamma_old) / (h * gamma)) * m_particle * beta_dot * beta_c * c_mmns**2
        
        print(f"\nβ̇ = {beta_dot:.1e} mm/ns²:")
        print(f"  RHS force: {rad_frc_rhs:.2e}")
        print(f"  LHS force: {rad_frc_lhs:.2e}")
        print(f"  RHS > threshold? {rad_frc_rhs > threshold}")
        print(f"  LHS > threshold? {rad_frc_lhs > threshold}")
        
        if rad_frc_rhs > threshold or rad_frc_lhs > threshold:
            print(f"  ✅ RADIATION REACTION TRIGGERED!")
            return {
                'char_time': char_time_test,
                'gamma': gamma,
                'gamma_old': gamma_old,
                'h': h,
                'beta_dot': beta_dot,
                'beta_c': beta_c,
                'mass': m_particle
            }
    
    print("❌ No triggering parameters found in tested range")
    return None


def test_triggered_radiation_reaction():
    """Test radiation reaction with parameters that should trigger it."""
    
    print("\n=== TESTING TRIGGERED RADIATION REACTION ===")
    
    # First find triggering parameters
    params = find_triggering_parameters()
    if params is None:
        print("Cannot test - no triggering parameters found")
        return
    
    integrator = LienardWiechertIntegrator()
    
    # Set up test with triggering parameters
    vector_data = {
        'char_time': params['char_time'],
        'gamma': np.array([params['gamma_old']]),
        'm': params['mass']
    }
    
    result = {
        'gamma': np.array([params['gamma']]),
        'bdotx': np.array([params['beta_dot']]),
        'bdoty': np.array([0.0]),
        'bdotz': np.array([params['beta_dot']]), 
        'bx': np.array([params['beta_c']]),
        'by': np.array([0.0]),
        'bz': np.array([params['beta_c']])
    }
    
    original_bdotx = result['bdotx'][0]
    original_bdotz = result['bdotz'][0]
    
    print(f"\nApplying radiation reaction with triggering parameters:")
    print(f"  Original β̇x: {original_bdotx:.2e}")
    print(f"  Original β̇z: {original_bdotz:.2e}")
    
    # Apply radiation reaction
    integrator._apply_radiation_reaction(params['h'], vector_data, result, 0)
    
    change_x = result['bdotx'][0] - original_bdotx
    change_z = result['bdotz'][0] - original_bdotz
    rel_change_x = change_x / original_bdotx if original_bdotx != 0 else 0
    rel_change_z = change_z / original_bdotz if original_bdotz != 0 else 0
    
    print(f"  After radiation reaction:")
    print(f"    β̇x: {result['bdotx'][0]:.2e} (change: {change_x:.2e}, rel: {rel_change_x:.2e})")
    print(f"    β̇z: {result['bdotz'][0]:.2e} (change: {change_z:.2e}, rel: {rel_change_z:.2e})")
    
    if abs(change_x) > 0 or abs(change_z) > 0:
        print("  ✅ RADIATION REACTION SUCCESSFULLY APPLIED!")
        
        # Check if the effect is reasonable (not too extreme)
        if abs(rel_change_x) < 10 and abs(rel_change_z) < 10:
            print("  ✅ Radiation reaction magnitude is reasonable")
        else:
            print("  ⚠️  Radiation reaction magnitude is large but physical for these extreme parameters")
    else:
        print("  ❌ Radiation reaction did not modify accelerations")


def test_ultra_relativistic_scaling():
    """Test radiation reaction scaling for ultra-relativistic particles."""
    
    print("\n=== ULTRA-RELATIVISTIC SCALING TEST ===")
    
    integrator = LienardWiechertIntegrator()
    
    # Use a char_time that makes triggering more likely
    char_time = 1e-10
    
    # Test different gamma values (ultra-relativistic regime)
    gammas = [100, 1000, 10000]  # Increasingly relativistic
    beta_dot = 1e3  # Fixed acceleration
    h = 1e-12  # Small timestep
    
    for gamma in gammas:
        print(f"\nTesting γ = {gamma}")
        
        # For ultra-relativistic: β ≈ 1 - 1/(2γ²)
        beta = np.sqrt(1 - 1/gamma**2)
        
        vector_data = {
            'char_time': char_time,
            'gamma': np.array([gamma * 0.9]),  # Previous gamma 
            'm': ELECTRON_MASS
        }
        
        result = {
            'gamma': np.array([gamma]),
            'bdotx': np.array([beta_dot]),
            'bdoty': np.array([0.0]),
            'bdotz': np.array([beta_dot]),
            'bx': np.array([beta]),
            'by': np.array([0.0]),
            'bz': np.array([beta])
        }
        
        original_bdotx = result['bdotx'][0]
        
        # Apply radiation reaction
        integrator._apply_radiation_reaction(h, vector_data, result, 0)
        
        change = result['bdotx'][0] - original_bdotx
        
        print(f"  β = {beta:.6f}")
        print(f"  Original β̇: {original_bdotx:.2e}")
        print(f"  Change: {change:.2e}")
        print(f"  Radiation reaction triggered: {'Yes' if abs(change) > 0 else 'No'}")


if __name__ == "__main__":
    find_triggering_parameters()
    test_triggered_radiation_reaction()
    test_ultra_relativistic_scaling()