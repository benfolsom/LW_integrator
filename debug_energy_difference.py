"""
Debug Energy Difference Between Original and New Code

Investigation of the energy component (Pt) difference in relativistic collision systems.
The momentum components match perfectly but energy differs by ~300x.

This could be due to:
1. Different energy-momentum consistency enforcement
2. Different relativistic calculation methods
3. Different handling of proper time updates

Author: Ben Folsom 
Date: 2025-09-12
"""

import numpy as np
import sys
import os

# Add paths
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./lw_integrator'))

from lw_integrator.core.integration import LiénardWiechertIntegrator
from lw_integrator.physics.constants import *
import covariant_integrator_library as original_lib


def create_collision_system():
    """Create identical collision system for debugging."""
    v_approach = 0.1  # 0.1c
    gamma = 1.0 / np.sqrt(1 - v_approach**2)
    
    particles = {
        'x': np.array([-5e-6, 5e-6]),    # Start 10 μm apart
        'y': np.array([0.0, 0.0]),
        'z': np.array([0.0, 0.0]),
        't': np.array([0.0, 0.0]),
        'Px': np.array([gamma * PROTON_MASS * v_approach * C_MMNS,
                       -gamma * PROTON_MASS * v_approach * C_MMNS]),
        'Py': np.array([0.0, 0.0]),
        'Pz': np.array([0.0, 0.0]),
        'Pt': np.array([gamma * PROTON_MASS * C_MMNS**2,
                       gamma * PROTON_MASS * C_MMNS**2]),
        'gamma': np.array([gamma, gamma]),
        'bx': np.array([v_approach, -v_approach]),
        'by': np.array([0.0, 0.0]),
        'bz': np.array([0.0, 0.0]),
        'bdotx': np.array([0.0, 0.0]),
        'bdoty': np.array([0.0, 0.0]),
        'bdotz': np.array([0.0, 0.0]),
        'q': 1.0,
        'char_time': np.array([1e-4, 1e-4]),
        'm': 938.3
    }
    
    return particles


def debug_energy_calculation():
    """Debug the energy calculation differences."""
    print("🔍 DEBUGGING ENERGY CALCULATION DIFFERENCES")
    print("="*60)
    
    # Create test system
    particles = create_collision_system()
    h = 1e-6  # 1 ns timestep
    
    print(f"Initial conditions:")
    print(f"  Separation: {abs(particles['x'][1] - particles['x'][0])*1e6:.1f} nm")
    print(f"  Approach velocity: {particles['bx'][0]:.3f}c")
    print(f"  Initial gamma: {particles['gamma'][0]:.6f}")
    print(f"  Initial energy: {particles['Pt'][0]:.2f} MeV")
    print(f"  Initial momentum: {particles['Px'][0]:.2f} MeV/c")
    
    # Test original code
    particles_orig = {key: np.copy(val) if isinstance(val, np.ndarray) else val 
                     for key, val in particles.items()}
    
    apt_R = np.inf
    sim_type = 2
    result_orig = original_lib.eqsofmotion_static(h, particles_orig, particles_orig, apt_R, sim_type)
    
    print(f"\n📊 ORIGINAL CODE RESULTS:")
    print(f"  ΔPx = {result_orig['Px'][0] - particles['Px'][0]:.2e} MeV/c")
    print(f"  ΔPt = {result_orig['Pt'][0] - particles['Pt'][0]:.2e} MeV")
    print(f"  Final gamma = {result_orig['gamma'][0]:.6f}")
    print(f"  Final beta_x = {result_orig['bx'][0]:.6f}")
    
    # Test new code
    integrator = LiénardWiechertIntegrator()
    particles_new = {key: np.copy(val) if isinstance(val, np.ndarray) else val 
                    for key, val in particles.items()}
    
    result_new = integrator.eqsofmotion_static(h, particles_new, particles_new)
    
    print(f"\n📊 NEW CODE RESULTS:")
    print(f"  ΔPx = {result_new['Px'][0] - particles['Px'][0]:.2e} MeV/c")
    print(f"  ΔPt = {result_new['Pt'][0] - particles['Pt'][0]:.2e} MeV")
    print(f"  Final gamma = {result_new['gamma'][0]:.6f}")
    print(f"  Final beta_x = {result_new['bx'][0]:.6f}")
    
    # Energy-momentum consistency check
    print(f"\n🔬 ENERGY-MOMENTUM CONSISTENCY:")
    
    # Original code
    E_orig = result_orig['Pt'][0]
    px_orig = result_orig['Px'][0] 
    py_orig = result_orig['Py'][0]
    pz_orig = result_orig['Pz'][0]
    m = particles['m']
    
    # Check E² = (pc)² + (mc²)² for original
    E2_orig = E_orig**2
    p2c2_orig = px_orig**2 + py_orig**2 + pz_orig**2
    mc2_orig = m * C_MMNS**2
    expected_E2_orig = p2c2_orig + mc2_orig**2
    
    print(f"  Original code:")
    print(f"    E² = {E2_orig:.2e}")
    print(f"    (pc)² + (mc²)² = {expected_E2_orig:.2e}")
    print(f"    Difference = {abs(E2_orig - expected_E2_orig):.2e}")
    print(f"    Relative error = {abs(E2_orig - expected_E2_orig)/expected_E2_orig:.2e}")
    
    # New code
    E_new = result_new['Pt'][0]
    px_new = result_new['Px'][0]
    py_new = result_new['Py'][0] 
    pz_new = result_new['Pz'][0]
    
    E2_new = E_new**2
    p2c2_new = px_new**2 + py_new**2 + pz_new**2
    expected_E2_new = p2c2_new + mc2_orig**2
    
    print(f"  New code:")
    print(f"    E² = {E2_new:.2e}")
    print(f"    (pc)² + (mc²)² = {expected_E2_new:.2e}")
    print(f"    Difference = {abs(E2_new - expected_E2_new):.2e}")
    print(f"    Relative error = {abs(E2_new - expected_E2_new)/expected_E2_new:.2e}")
    
    # Direct comparison
    print(f"\n⚖️  DIRECT COMPARISON:")
    print(f"  Energy difference: {abs(E_orig - E_new):.2e} MeV")
    print(f"  Momentum difference: {abs(px_orig - px_new):.2e} MeV/c")
    print(f"  Energy ratio: {E_orig/E_new:.6f}")
    
    print(f"\n🤔 ANALYSIS:")
    if abs(px_orig - px_new) < 1e-10:
        print("  ✅ Momentum changes are identical")
    else:
        print("  ❌ Momentum changes differ")
        
    if abs(E_orig - E_new) < 1e-6:
        print("  ✅ Energy changes are similar")
    else:
        print("  ❌ Energy changes differ significantly")
        
    # Check which one maintains energy-momentum relation better
    rel_error_orig = abs(E2_orig - expected_E2_orig)/expected_E2_orig
    rel_error_new = abs(E2_new - expected_E2_new)/expected_E2_new
    
    if rel_error_new < rel_error_orig:
        print("  ✅ New code maintains energy-momentum relation better")
    elif rel_error_orig < rel_error_new:
        print("  ⚠️  Original code maintains energy-momentum relation better")
    else:
        print("  ⚖️  Both codes maintain energy-momentum relation equally")


if __name__ == "__main__":
    debug_energy_calculation()
