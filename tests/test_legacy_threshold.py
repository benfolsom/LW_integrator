#!/usr/bin/env python3
"""
Test radiation reaction with exact legacy threshold logic
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lw_integrator.core.integration import LienardWiechertIntegrator
from lw_integrator.physics.constants import C_MMNS, ELECTRON_MASS


def test_legacy_threshold_logic():
    """Test with exact legacy threshold logic."""
    
    print("=== TESTING LEGACY THRESHOLD LOGIC ===")
    
    integrator = LienardWiechertIntegrator()
    
    # Use the same failing test parameters
    vector_data = {
        'char_time': 1e-3,
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
    print(f"Original bdotx: {original_bdotx:.2e}")
    
    # Apply radiation reaction with corrected threshold logic
    integrator._apply_radiation_reaction(1e-9, vector_data, result, 0)
    
    change = result['bdotx'][0] - original_bdotx
    relative_change = change / original_bdotx if original_bdotx != 0 else 0
    
    print(f"After radiation reaction: {result['bdotx'][0]:.2e}")
    print(f"Change: {change:.2e}")
    print(f"Relative change: {relative_change:.2e}")
    
    if abs(relative_change) < 100:
        print(f"✅ Improved scaling with legacy threshold logic")
    else:
        print(f"❌ Still problematic scaling")


def test_manual_threshold_calculation():
    """Manually check what the legacy threshold logic does."""
    
    print("\n=== MANUAL THRESHOLD CALCULATION ===")
    
    # Parameters from failing test
    char_time = 1e-3
    m_particle = ELECTRON_MASS 
    gamma = 15.0
    gamma_old = 10.0
    h = 1e-9
    bdotx = 1e12
    bx = 0.3
    c_mmns = C_MMNS
    
    # Calculate force components
    rad_frc_x_rhs = -gamma**3 * (m_particle * bdotx**2 * c_mmns**2) * bx * c_mmns
    rad_frc_x_lhs = ((gamma - gamma_old) / (h * gamma)) * m_particle * bdotx * bx * c_mmns**2
    
    threshold = char_time / 1e1
    
    print(f"Force components:")
    print(f"  rad_frc_x_rhs = {rad_frc_x_rhs:.2e}")
    print(f"  rad_frc_x_lhs = {rad_frc_x_lhs:.2e}")
    print(f"  threshold = {threshold}")
    
    print(f"\nLegacy threshold checks:")
    print(f"  rad_frc_x_rhs > threshold? {rad_frc_x_rhs > threshold}")
    print(f"  rad_frc_x_lhs > threshold? {rad_frc_x_lhs > threshold}")
    
    # The radiation reaction should trigger if either condition is true
    should_trigger = (rad_frc_x_rhs > threshold) or (rad_frc_x_lhs > threshold)
    print(f"  Should trigger radiation reaction? {should_trigger}")
    
    if should_trigger:
        correction = char_time * (rad_frc_x_lhs + rad_frc_x_rhs) / (m_particle * c_mmns)
        print(f"\nCorrection calculation:")
        print(f"  total_force = {rad_frc_x_lhs + rad_frc_x_rhs:.2e}")
        print(f"  correction = {correction:.2e}")
        print(f"  relative change = {correction / bdotx:.2e}")


if __name__ == "__main__":
    test_legacy_threshold_logic()
    test_manual_threshold_calculation()