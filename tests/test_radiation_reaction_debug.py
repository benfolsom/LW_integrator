#!/usr/bin/env python3
"""
Debug Test Suite for Radiation Reaction Force Implementation

CAI: Direct comparison with legacy code to debug the 30+ order magnitude jumps.
This test uses the exact same parameters and logic as the legacy implementation.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lw_integrator.core.integration import LienardWiechertIntegrator
from lw_integrator.physics.constants import C_MMNS, ELECTRON_MASS


def debug_radiation_reaction_calculation():
    """Debug the radiation reaction calculation step by step."""
    
    print("=== DEBUGGING RADIATION REACTION CALCULATION ===")
    
    # Use the exact values from failing test
    char_time = 1e-3  # From original failing test
    m_particle = ELECTRON_MASS
    gamma = 15.0
    gamma_old = 10.0
    h = 1e-9
    bdotx = 1e12
    bx = 0.3
    c_mmns = C_MMNS
    
    print(f"Input parameters:")
    print(f"  char_time = {char_time}")
    print(f"  m_particle = {m_particle}")
    print(f"  gamma = {gamma}")
    print(f"  gamma_old = {gamma_old}")
    print(f"  h = {h}")
    print(f"  bdotx = {bdotx}")
    print(f"  bx = {bx}")
    print(f"  c_mmns = {c_mmns}")
    
    # Calculate radiation force components step by step
    print(f"\nCalculating radiation force components:")
    
    # RHS term: -γ³(m*β̇²*c²)*β*c
    rhs_part1 = gamma**3
    rhs_part2 = m_particle * bdotx**2 * c_mmns**2
    rhs_part3 = bx * c_mmns
    rad_frc_x_rhs = -rhs_part1 * rhs_part2 * rhs_part3
    
    print(f"  RHS calculation:")
    print(f"    γ³ = {rhs_part1}")
    print(f"    m*β̇²*c² = {rhs_part2:.2e}")
    print(f"    β*c = {rhs_part3}")
    print(f"    rad_frc_x_rhs = -γ³*(m*β̇²*c²)*(β*c) = {rad_frc_x_rhs:.2e}")
    
    # LHS term: (γ_new - γ_old)/(h*γ_new) * m * β̇ * β * c²
    lhs_part1 = (gamma - gamma_old) / (h * gamma)
    lhs_part2 = m_particle * bdotx * bx * c_mmns**2
    rad_frc_x_lhs = lhs_part1 * lhs_part2
    
    print(f"  LHS calculation:")
    print(f"    (γ_new - γ_old)/(h*γ_new) = {lhs_part1:.2e}")
    print(f"    m*β̇*β*c² = {lhs_part2:.2e}")
    print(f"    rad_frc_x_lhs = {rad_frc_x_lhs:.2e}")
    
    # Threshold check
    threshold = char_time / 1e1
    print(f"\nThreshold check:")
    print(f"  threshold = char_time/10 = {threshold}")
    print(f"  abs(rad_frc_x_rhs) = {abs(rad_frc_x_rhs):.2e}")
    print(f"  abs(rad_frc_x_lhs) = {abs(rad_frc_x_lhs):.2e}")
    print(f"  RHS > threshold? {abs(rad_frc_x_rhs) > threshold}")
    print(f"  LHS > threshold? {abs(rad_frc_x_lhs) > threshold}")
    
    # Final correction
    if abs(rad_frc_x_rhs) > threshold or abs(rad_frc_x_lhs) > threshold:
        correction = char_time * (rad_frc_x_lhs + rad_frc_x_rhs) / (m_particle * c_mmns)
        total_force = rad_frc_x_lhs + rad_frc_x_rhs
        
        print(f"\nCorrection calculation:")
        print(f"  total_force = RHS + LHS = {total_force:.2e}")
        print(f"  correction = char_time * total_force / (m * c)")
        print(f"  correction = {char_time} * {total_force:.2e} / ({m_particle} * {c_mmns})")
        print(f"  correction = {correction:.2e}")
        
        new_bdotx = bdotx + correction
        relative_change = correction / bdotx
        
        print(f"\nFinal result:")
        print(f"  original bdotx = {bdotx:.2e}")
        print(f"  correction = {correction:.2e}")  
        print(f"  new bdotx = {new_bdotx:.2e}")
        print(f"  relative change = {relative_change:.2e}")
        
        if abs(relative_change) > 100:
            print(f"  ❌ PROBLEM: Relative change > 100!")
        else:
            print(f"  ✅ Relative change seems reasonable")
    else:
        print(f"\nRadiation reaction not activated (below threshold)")


def test_with_realistic_char_time():
    """Test with realistic characteristic time."""
    
    print("\n=== TESTING WITH REALISTIC CHAR_TIME ===")
    
    # Calculate realistic electron char_time
    c_mmns = 299.792458  # mm/ns
    mass_electron_amu = 0.0005485  # amu  
    q_electron = 1.178734e-5  # legacy unit charge
    char_time_realistic = 2/3 * q_electron**2 / (mass_electron_amu * c_mmns**3)
    
    print(f"Realistic electron char_time: {char_time_realistic:.3e} ns")
    print(f"Original test char_time: 1e-3 ns")
    print(f"Ratio (test/realistic): {1e-3 / char_time_realistic:.2e}")
    
    # Test with same parameters but realistic char_time
    integrator = LienardWiechertIntegrator()
    
    vector_data = {
        'char_time': char_time_realistic,
        'gamma': np.array([10.0]),
        'm': ELECTRON_MASS
    }
    
    result = {
        'gamma': np.array([15.0]),
        'bdotx': np.array([1e12]),
        'bdoty': np.array([1e12]),
        'bdotz': np.array([1e12]),
        'bx': np.array([0.3]),
        'by': np.array([0.3]),
        'bz': np.array([0.9])
    }
    
    original_bdotx = result['bdotx'][0]
    
    # Apply radiation reaction
    integrator._apply_radiation_reaction(1e-9, vector_data, result, 0)
    
    change = result['bdotx'][0] - original_bdotx
    relative_change = change / original_bdotx if original_bdotx != 0 else 0
    
    print(f"\nResult with realistic char_time:")
    print(f"  Original bdotx: {original_bdotx:.2e}")
    print(f"  Change: {change:.2e}")
    print(f"  Relative change: {relative_change:.2e}")
    
    if abs(relative_change) < 1.0:
        print(f"  ✅ Reasonable scaling with realistic char_time")
    else:
        print(f"  ❌ Still problematic scaling")


if __name__ == "__main__":
    debug_radiation_reaction_calculation()
    test_with_realistic_char_time()