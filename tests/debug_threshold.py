#!/usr/bin/env python3
"""
Debug threshold calculation for realistic accelerations
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lw_integrator.physics.constants import C_MMNS, ELECTRON_MASS


def debug_threshold_calculation():
    """Debug why radiation reaction doesn't trigger for realistic accelerations."""
    
    print("=== DEBUGGING THRESHOLD CALCULATION ===")
    
    # Test parameters matching the realistic case where no radiation reaction occurred
    char_time = 1e-3
    m_particle = ELECTRON_MASS 
    gamma = 15.0
    gamma_old = 10.0
    h = 1e-9
    bdotx = 1e-2  # Realistic acceleration that showed no radiation reaction
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
    
    # Calculate force components
    rad_frc_x_rhs = -gamma**3 * (m_particle * bdotx**2 * c_mmns**2) * bx * c_mmns
    rad_frc_x_lhs = ((gamma - gamma_old) / (h * gamma)) * m_particle * bdotx * bx * c_mmns**2
    
    threshold = char_time / 1e1
    
    print(f"\nForce calculation:")
    print(f"  rad_frc_x_rhs = {rad_frc_x_rhs:.2e}")
    print(f"  rad_frc_x_lhs = {rad_frc_x_lhs:.2e}")
    print(f"  threshold = {threshold:.2e}")
    
    print(f"\nThreshold checks:")
    print(f"  rad_frc_x_rhs > threshold? {rad_frc_x_rhs > threshold} ({rad_frc_x_rhs:.2e} > {threshold:.2e})")
    print(f"  rad_frc_x_lhs > threshold? {rad_frc_x_lhs > threshold} ({rad_frc_x_lhs:.2e} > {threshold:.2e})")
    
    should_trigger = (rad_frc_x_rhs > threshold) or (rad_frc_x_lhs > threshold)
    print(f"  Should trigger? {should_trigger}")
    
    # Compare with unrealistic case
    print(f"\n=== COMPARISON WITH UNREALISTIC CASE ===")
    bdotx_unrealistic = 1e12
    rad_frc_x_rhs_unreal = -gamma**3 * (m_particle * bdotx_unrealistic**2 * c_mmns**2) * bx * c_mmns
    rad_frc_x_lhs_unreal = ((gamma - gamma_old) / (h * gamma)) * m_particle * bdotx_unrealistic * bx * c_mmns**2
    
    print(f"Unrealistic acceleration ({bdotx_unrealistic:.2e}):")
    print(f"  rad_frc_x_rhs = {rad_frc_x_rhs_unreal:.2e}")
    print(f"  rad_frc_x_lhs = {rad_frc_x_lhs_unreal:.2e}")
    print(f"  RHS > threshold? {rad_frc_x_rhs_unreal > threshold}")
    print(f"  LHS > threshold? {rad_frc_x_lhs_unreal > threshold}")
    
    # Show scaling factors
    accel_ratio = bdotx_unrealistic / bdotx
    rhs_ratio = rad_frc_x_rhs_unreal / rad_frc_x_rhs if rad_frc_x_rhs != 0 else float('inf')
    lhs_ratio = rad_frc_x_lhs_unreal / rad_frc_x_lhs if rad_frc_x_lhs != 0 else float('inf')
    
    print(f"\nScaling analysis:")
    print(f"  Acceleration ratio: {accel_ratio:.2e}")
    print(f"  RHS force ratio: {rhs_ratio:.2e}")
    print(f"  LHS force ratio: {lhs_ratio:.2e}")
    print(f"  Expected RHS scaling (∝ a²): {accel_ratio**2:.2e}")
    print(f"  Expected LHS scaling (∝ a): {accel_ratio:.2e}")


if __name__ == "__main__":
    debug_threshold_calculation()