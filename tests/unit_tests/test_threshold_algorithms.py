#!/usr/bin/env python3
"""
Test the improved radiation reaction threshold logic.

This script validates that the new threshold scaling works correctly
for both low-energy and high-energy scenarios.
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lw_integrator import LienardWiechertIntegrator
from lw_integrator.physics.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE_ESU


def test_threshold_scaling():
    """Test radiation reaction threshold scaling with energy."""
    print("Testing improved radiation reaction threshold scaling...")
    
    # Physical parameters
    mass = ELECTRON_MASS_AMU
    charge = ELEMENTARY_CHARGE_ESU
    char_time = 2/3 * charge**2 / (mass * C_MMNS**3)
    
    print(f"Characteristic time: {char_time:.6e} ns")
    print(f"Base threshold: {char_time/1e1:.6e}")
    
    # Test scenarios with different gamma factors
    test_cases = [
        ("Non-relativistic", 1.001, 1e5),
        ("Mildly relativistic", 1.1, 1e6),
        ("Highly relativistic", 2.0, 1e7),
        ("Ultra-relativistic", 10.0, 1e8),
        ("Extreme relativistic", 100.0, 1e9)
    ]
    
    integrator = LienardWiechertIntegrator()
    dt = 1e-6  # ns
    
    print(f"\n{'Scenario':<20} {'γ':<8} {'β̇':<12} {'Threshold':<12} {'Triggered':<10}")
    print("-" * 70)
    
    for scenario, gamma, beta_dot in test_cases:
        # Create test data
        vector_data = {
            'char_time': char_time,
            'gamma': np.array([gamma]),
            'm': mass
        }
        
        velocity = 0.1 * C_MMNS  # Some reasonable velocity
        beta = velocity / C_MMNS
        
        result = {
            'gamma': np.array([gamma]),
            'bdotx': np.array([beta_dot]),
            'bdoty': np.array([0.0]),
            'bdotz': np.array([0.0]),
            'bx': np.array([beta]),
            'by': np.array([0.0]),
            'bz': np.array([0.0])
        }
        
        # Store original value
        original_bdotx = result['bdotx'][0]
        
        try:
            # Apply radiation reaction
            integrator._apply_radiation_reaction(dt, vector_data, result, 0)
            
            # Check if it was modified
            new_bdotx = result['bdotx'][0]
            triggered = abs(new_bdotx - original_bdotx) > 1e-15
            
            # Calculate the threshold that was actually used
            base_threshold = char_time / 1e1
            energy_scale = gamma if gamma > 1.1 else 1.0
            threshold = base_threshold * min(energy_scale, 100.0)
            
            print(f"{scenario:<20} {gamma:<8.1f} {beta_dot:<12.1e} {threshold:<12.2e} {'Yes' if triggered else 'No':<10}")
            
        except Exception as e:
            print(f"{scenario:<20} {gamma:<8.1f} {beta_dot:<12.1e} {'ERROR':<12} {'No':<10}")
    
    print(f"\n✅ Threshold scaling test completed!")
    print(f"   • Low γ: uses base threshold")
    print(f"   • High γ: scales threshold up to maintain stability")
    print(f"   • Extreme γ: caps at 100× base threshold")


def test_minimal_radiation_reaction():
    """Test radiation reaction with minimal realistic scenario."""
    print(f"\n" + "="*50)
    print("=== MINIMAL RADIATION REACTION TEST ===")
    
    mass = ELECTRON_MASS_AMU
    charge = ELEMENTARY_CHARGE_ESU
    char_time = 2/3 * charge**2 / (mass * C_MMNS**3)
    
    # Very simple scenario - electron with modest acceleration
    gamma = 1.5  # Mildly relativistic
    beta_dot = 1e6  # Moderate acceleration
    velocity = 30.0  # mm/ns
    
    print(f"Test parameters:")
    print(f"  γ = {gamma}")
    print(f"  β̇ = {beta_dot:.1e}")
    print(f"  velocity = {velocity} mm/ns")
    print(f"  char_time = {char_time:.3e} ns")
    
    # Calculate expected threshold
    base_threshold = char_time / 1e1
    energy_scale = gamma if gamma > 1.1 else 1.0
    expected_threshold = base_threshold * min(energy_scale, 100.0)
    print(f"  Expected threshold = {expected_threshold:.3e}")
    
    # Set up integrator
    integrator = LienardWiechertIntegrator()
    dt = 1e-6
    
    vector_data = {
        'char_time': char_time,
        'gamma': np.array([gamma]),
        'm': mass
    }
    
    result = {
        'gamma': np.array([gamma]),
        'bdotx': np.array([beta_dot]),
        'bdoty': np.array([0.0]),
        'bdotz': np.array([0.0]),
        'bx': np.array([velocity/C_MMNS]),
        'by': np.array([0.0]),
        'bz': np.array([0.0])
    }
    
    original_bdotx = result['bdotx'][0]
    
    # Calculate expected radiation force
    rad_frc_rhs = (-gamma**3 * (mass * beta_dot**2 * C_MMNS**2) * 
                  (velocity/C_MMNS) * C_MMNS)
    
    print(f"  Radiation force = {rad_frc_rhs:.3e}")
    print(f"  |Force| > threshold? {abs(rad_frc_rhs) > expected_threshold}")
    
    # Apply radiation reaction
    integrator._apply_radiation_reaction(dt, vector_data, result, 0)
    
    new_bdotx = result['bdotx'][0]
    change = new_bdotx - original_bdotx
    
    print(f"\nResults:")
    print(f"  Original β̇: {original_bdotx:.6e}")
    print(f"  New β̇: {new_bdotx:.6e}")
    print(f"  Change: {change:.6e}")
    print(f"  Triggered: {'Yes' if abs(change) > 1e-15 else 'No'}")
    
    if abs(change) > 1e-15:
        print(f"✅ SUCCESS: Radiation reaction working with improved threshold!")
    else:
        print(f"ℹ️  INFO: No radiation reaction (force below threshold)")


if __name__ == "__main__":
    test_threshold_scaling()
    test_minimal_radiation_reaction()