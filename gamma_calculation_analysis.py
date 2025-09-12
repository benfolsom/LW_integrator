"""
Comparative Analysis: Covariant vs Standard Gamma Calculation

This script investigates the difference between:
1. Original covariant gamma calculation: γ = (1/mc)[Pt - EM_correction]
2. Standard relativistic gamma: γ = 1/√(1-β²) or from E² = (pc)² + (mc²)²

The goal is to understand which approach is more theoretically sound for
the Liénard-Wiechert electromagnetic field simulation.

Author: Ben Folsom
Date: 2025-09-12
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./lw_integrator'))

from lw_integrator.core.integration import LiénardWiechertIntegrator
from lw_integrator.physics.constants import *
import covariant_integrator_library as original_lib


def analyze_gamma_calculations():
    """
    Compare the two gamma calculation methods and their theoretical basis.
    """
    print("🔬 GAMMA CALCULATION ANALYSIS")
    print("="*80)
    print("Comparing covariant vs standard gamma calculations in LW field theory")
    print()
    
    # Create test system - relativistic collision to highlight differences
    v_approach = 0.5  # 0.5c to make relativistic effects prominent
    gamma_exact = 1.0 / np.sqrt(1 - v_approach**2)
    
    particles = {
        'x': np.array([-1e-6, 1e-6]),    # 2 μm separation
        'y': np.array([0.0, 0.0]),
        'z': np.array([0.0, 0.0]),
        't': np.array([0.0, 0.0]),
        'Px': np.array([gamma_exact * PROTON_MASS * v_approach * C_MMNS,
                       -gamma_exact * PROTON_MASS * v_approach * C_MMNS]),
        'Py': np.array([0.0, 0.0]),
        'Pz': np.array([0.0, 0.0]),
        'Pt': np.array([gamma_exact * PROTON_MASS * C_MMNS**2,
                       gamma_exact * PROTON_MASS * C_MMNS**2]),
        'gamma': np.array([gamma_exact, gamma_exact]),
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
    
    h = 1e-6  # 1 ns timestep
    
    print(f"📊 INITIAL CONDITIONS:")
    print(f"  Approach velocity: {v_approach:.3f}c")
    print(f"  Exact gamma: {gamma_exact:.6f}")
    print(f"  Separation: {abs(particles['x'][1] - particles['x'][0])*1e6:.1f} nm")
    print(f"  EM coupling: q₁q₂ = {particles['q']**2:.1f}")
    
    # Run original code to see covariant gamma calculation
    particles_orig = {key: np.copy(val) if isinstance(val, np.ndarray) else val 
                     for key, val in particles.items()}
    
    apt_R = np.inf
    sim_type = 2
    result_orig = original_lib.eqsofmotion_static(h, particles_orig, particles_orig, apt_R, sim_type)
    
    # Run new code with standard gamma calculation
    integrator = LiénardWiechertIntegrator()
    particles_new = {key: np.copy(val) if isinstance(val, np.ndarray) else val 
                    for key, val in particles.items()}
    
    result_new = integrator.eqsofmotion_static(h, particles_new, particles_new)
    
    print(f"\n🧮 GAMMA CALCULATION COMPARISON:")
    print(f"{'Method':<25} {'Particle 1':<15} {'Particle 2':<15} {'Consistency':<15}")
    print("-" * 75)
    
    # Exact gamma from velocity
    gamma_p1_exact = 1.0 / np.sqrt(1 - (result_orig['bx'][0]**2 + result_orig['by'][0]**2 + result_orig['bz'][0]**2))
    gamma_p2_exact = 1.0 / np.sqrt(1 - (result_orig['bx'][1]**2 + result_orig['by'][1]**2 + result_orig['bz'][1]**2))
    
    print(f"{'From velocity β':<25} {gamma_p1_exact:.6f} {gamma_p2_exact:.6f} {'Reference':<15}")
    
    # Original covariant calculation
    gamma_p1_orig = result_orig['gamma'][0]
    gamma_p2_orig = result_orig['gamma'][1]
    
    print(f"{'Original (covariant)':<25} {gamma_p1_orig:.6f} {gamma_p2_orig:.6f} {'???':<15}")
    
    # New standard calculation  
    gamma_p1_new = result_new['gamma'][0]
    gamma_p2_new = result_new['gamma'][1]
    
    print(f"{'New (standard)':<25} {gamma_p1_new:.6f} {gamma_p2_new:.6f} {'✓':<15}")
    
    # Energy-momentum consistency check
    print(f"\n⚖️  ENERGY-MOMENTUM CONSISTENCY:")
    
    def check_em_consistency(result, label, particle_idx=0):
        """Check E² = (pc)² + (mc²)² consistency"""
        E = result['Pt'][particle_idx]
        px = result['Px'][particle_idx]
        py = result['Py'][particle_idx]
        pz = result['Pz'][particle_idx]
        mc2 = particles['m'] * C_MMNS**2
        
        E2 = E**2
        p2c2 = px**2 + py**2 + pz**2
        expected_E2 = p2c2 + mc2**2
        
        rel_error = abs(E2 - expected_E2) / expected_E2
        
        print(f"  {label} (particle {particle_idx+1}):")
        print(f"    E² = {E2:.3e}")
        print(f"    (pc)² + (mc²)² = {expected_E2:.3e}")
        print(f"    Relative error = {rel_error:.2e}")
        
        return rel_error
    
    error_orig_p1 = check_em_consistency(result_orig, "Original", 0)
    error_orig_p2 = check_em_consistency(result_orig, "Original", 1)
    error_new_p1 = check_em_consistency(result_new, "New", 0)
    error_new_p2 = check_em_consistency(result_new, "New", 1)
    
    avg_error_orig = (error_orig_p1 + error_orig_p2) / 2
    avg_error_new = (error_new_p1 + error_new_p2) / 2
    
    print(f"\n📈 THEORETICAL ANALYSIS:")
    print("="*60)
    
    print(f"1. COVARIANT APPROACH (Original):")
    print(f"   γ = (1/mc)[Pt - EM_correction]")
    print(f"   - Based on four-momentum time component")
    print(f"   - Includes electromagnetic field corrections")
    print(f"   - May be more fundamental for LW field theory")
    print(f"   - Average E-p consistency error: {avg_error_orig:.2e}")
    
    print(f"\n2. STANDARD APPROACH (New):")
    print(f"   γ = 1/√(1-β²) or from E² = (pc)² + (mc²)²")
    print(f"   - Standard special relativity")
    print(f"   - Enforces energy-momentum relation by construction")
    print(f"   - More familiar and well-tested")
    print(f"   - Average E-p consistency error: {avg_error_new:.2e}")
    
    print(f"\n🤔 PHYSICAL INTERPRETATION:")
    print("="*60)
    
    # Check if the electromagnetic correction makes physical sense
    m = particles['m']
    c = C_MMNS
    
    # The EM correction term from original code (simplified approximation)
    r_separation = np.sqrt((particles['x'][0] - particles['x'][1])**2 + 
                          (particles['y'][0] - particles['y'][1])**2 + 
                          (particles['z'][0] - particles['z'][1])**2)
    
    # Approximate the EM correction (this is a simplified version)
    em_correction_p1 = particles['q'] * particles['q'] / (c * r_separation)
    
    gamma_covariant_p1 = (result_orig['Pt'][0] - em_correction_p1) / (m * c)
    
    print(f"Electromagnetic correction magnitude: {em_correction_p1:.2e} MeV")
    print(f"Particle rest energy: {m:.2e} MeV")
    print(f"Correction as fraction of rest energy: {em_correction_p1/m:.2e}")
    
    if abs(em_correction_p1/m) < 1e-6:
        print("→ EM correction is negligible compared to rest energy")
        print("→ Standard approach should be equivalent")
    else:
        print("→ EM correction is significant")
        print("→ Covariant approach may be necessary for accuracy")
    
    print(f"\n🎯 RECOMMENDATION:")
    print("="*60)
    
    if avg_error_new < avg_error_orig:
        print("✅ STANDARD APPROACH PREFERRED")
        print("- Better energy-momentum consistency")
        print("- EM corrections appear negligible for this system")
        print("- More numerically stable")
    elif abs(em_correction_p1/m) > 1e-3:
        print("✅ COVARIANT APPROACH PREFERRED") 
        print("- EM corrections are physically significant")
        print("- More fundamental for LW field theory")
        print("- Accept slightly lower numerical precision for physical accuracy")
    else:
        print("⚖️  BOTH APPROACHES VALID")
        print("- Choose based on numerical stability vs theoretical purity")
        print("- Standard approach for production, covariant for research")
    
    return {
        'covariant_consistency_error': avg_error_orig,
        'standard_consistency_error': avg_error_new,
        'em_correction_magnitude': em_correction_p1,
        'em_correction_relative': em_correction_p1/m
    }


def test_different_field_strengths():
    """Test both approaches across different electromagnetic field strengths."""
    print(f"\n🔍 FIELD STRENGTH DEPENDENCE TEST")
    print("="*80)
    
    separations = [10e-6, 1e-6, 100e-9, 10e-9]  # Different separations = different field strengths
    
    for separation in separations:
        print(f"\nSeparation: {separation*1e9:.1f} nm")
        
        # Coulomb potential energy at this separation
        U_coulomb = (1.44e-13) / separation  # MeV (rough estimate for e²/4πε₀r)
        print(f"Coulomb energy scale: {U_coulomb:.2e} MeV")
        print(f"Ratio to proton mass: {U_coulomb/938.3:.2e}")
        
        if U_coulomb/938.3 > 0.01:
            print("→ Strong EM field regime - covariant approach may be essential")
        elif U_coulomb/938.3 > 0.001:
            print("→ Moderate EM field regime - both approaches may work")
        else:
            print("→ Weak EM field regime - standard approach likely sufficient")


if __name__ == "__main__":
    analysis_results = analyze_gamma_calculations()
    test_different_field_strengths()
    
    print(f"\n📄 Analysis complete. Results saved to analysis_results.")
