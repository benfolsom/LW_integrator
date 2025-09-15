#!/usr/bin/env python3
"""
Test with progressively smaller char_time to find the sweet spot
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lw_integrator.core.integration import LienardWiechertIntegrator
from lw_integrator.physics.constants import C_MMNS, ELECTRON_MASS


def test_char_time_scaling():
    """Test radiation reaction with different char_time values."""
    
    print("=== TESTING CHAR_TIME SCALING ===")
    
    integrator = LienardWiechertIntegrator()
    
    # Test different char_time values
    char_times = [1e-3, 1e-6, 1e-9, 1e-12, 6.268e-15]  # From large to realistic
    
    base_acceleration = 1e12  # Keep the same unrealistic acceleration
    
    for char_time in char_times:
        print(f"\nTesting char_time = {char_time:.3e}")
        
        vector_data = {
            'char_time': char_time,
            'gamma': np.array([10.0]),
            'm': ELECTRON_MASS
        }
        
        result = {
            'gamma': np.array([15.0]),
            'bdotx': np.array([base_acceleration]),
            'bdoty': np.array([base_acceleration]),
            'bdotz': np.array([base_acceleration]),
            'bx': np.array([0.3]),
            'by': np.array([0.3]),
            'bz': np.array([0.9])
        }
        
        original_bdotx = result['bdotx'][0]
        
        # Apply radiation reaction
        integrator._apply_radiation_reaction(1e-9, vector_data, result, 0)
        
        change = result['bdotx'][0] - original_bdotx
        relative_change = change / original_bdotx if original_bdotx != 0 else 0
        
        print(f"  Original bdotx: {original_bdotx:.2e}")
        print(f"  Change: {change:.2e}")
        print(f"  Relative change: {relative_change:.2e}")
        
        if abs(relative_change) < 1.0:
            print(f"  ✅ Reasonable scaling")
        elif abs(relative_change) < 100:
            print(f"  ⚠️  Large but manageable")  
        else:
            print(f"  ❌ Unphysical scaling")


def test_realistic_acceleration_with_legacy_char_time():
    """Test with realistic acceleration and legacy char_time values."""
    
    print("\n=== TESTING REALISTIC ACCELERATION ===")
    
    integrator = LienardWiechertIntegrator()
    
    # From legacy code: typical bdot values are -8e-9 to 8e-2
    accelerations = [1e-8, 1e-5, 1e-2]  # Realistic range from legacy
    char_time = 1e-3  # Original test value
    
    for accel in accelerations:
        print(f"\nTesting acceleration = {accel:.2e} mm/ns²")
        
        vector_data = {
            'char_time': char_time,
            'gamma': np.array([10.0]),
            'm': ELECTRON_MASS
        }
        
        result = {
            'gamma': np.array([15.0]),
            'bdotx': np.array([accel]),
            'bdoty': np.array([accel]),
            'bdotz': np.array([accel]),
            'bx': np.array([0.3]),
            'by': np.array([0.3]),
            'bz': np.array([0.9])
        }
        
        original_bdotx = result['bdotx'][0]
        
        # Apply radiation reaction
        integrator._apply_radiation_reaction(1e-9, vector_data, result, 0)
        
        change = result['bdotx'][0] - original_bdotx
        relative_change = change / original_bdotx if original_bdotx != 0 else 0
        
        print(f"  Original bdotx: {original_bdotx:.2e}")
        print(f"  Change: {change:.2e}")
        print(f"  Relative change: {relative_change:.2e}")
        
        if abs(relative_change) < 0.1:
            print(f"  ✅ Small correction (as expected)")
        elif abs(relative_change) < 10.0:
            print(f"  ⚠️  Moderate correction")
        else:
            print(f"  ❌ Large correction")


if __name__ == "__main__":
    test_char_time_scaling()
    test_realistic_acceleration_with_legacy_char_time()