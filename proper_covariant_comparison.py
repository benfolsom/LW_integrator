"""
Proper Comparison: Original Covariant vs Standard Gamma

This script provides a fair comparison between:
1. Original covariant implementation (lines 305-306 from covariant_integrator_library.py)
2. Standard relativistic gamma calculation

The goal is to understand the validity and appropriate use cases for each approach
without dismissing the theoretical foundation of the covariant derivation.

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


def create_test_scenarios():
    """Create different test scenarios to evaluate both approaches."""
    scenarios = []
    
    # Scenario 1: Weak field, non-relativistic
    scenarios.append({
        'name': 'Weak Field Non-Relativistic',
        'separation': 10e-6,  # 10 μm
        'velocity': 0.01,     # 0.01c
        'description': 'Large separation, low velocity - EM effects minimal'
    })
    
    # Scenario 2: Moderate field, mildly relativistic  
    scenarios.append({
        'name': 'Moderate Field Relativistic',
        'separation': 1e-6,   # 1 μm  
        'velocity': 0.1,      # 0.1c
        'description': 'Medium separation, moderate velocity'
    })
    
    # Scenario 3: Strong field, highly relativistic
    scenarios.append({
        'name': 'Strong Field Highly Relativistic', 
        'separation': 100e-9, # 100 nm
        'velocity': 0.5,      # 0.5c
        'description': 'Close separation, high velocity - EM effects significant'
    })
    
    # Scenario 4: Ultra-strong field
    scenarios.append({
        'name': 'Ultra-Strong Field',
        'separation': 10e-9,  # 10 nm
        'velocity': 0.1,      # 0.1c  
        'description': 'Very close separation - extreme EM fields'
    })
    
    return scenarios


def create_particle_system(separation: float, velocity: float):
    """Create particle system for testing."""
    gamma_exact = 1.0 / np.sqrt(1 - velocity**2)
    
    return {
        'x': np.array([-separation/2, separation/2]),
        'y': np.array([0.0, 0.0]),
        'z': np.array([0.0, 0.0]),
        't': np.array([0.0, 0.0]),
        'Px': np.array([gamma_exact * PROTON_MASS * velocity * C_MMNS,
                       -gamma_exact * PROTON_MASS * velocity * C_MMNS]),
        'Py': np.array([0.0, 0.0]),
        'Pz': np.array([0.0, 0.0]),
        'Pt': np.array([gamma_exact * PROTON_MASS * C_MMNS**2,
                       gamma_exact * PROTON_MASS * C_MMNS**2]),
        'gamma': np.array([gamma_exact, gamma_exact]),
        'bx': np.array([velocity, -velocity]),
        'by': np.array([0.0, 0.0]),
        'bz': np.array([0.0, 0.0]),
        'bdotx': np.array([0.0, 0.0]),
        'bdoty': np.array([0.0, 0.0]),
        'bdotz': np.array([0.0, 0.0]),
        'q': 1.0,
        'char_time': np.array([1e-4, 1e-4]),
        'm': 938.3
    }


def analyze_scenario(scenario: dict):
    """Analyze a specific test scenario."""
    print(f"\\n🔍 SCENARIO: {scenario['name']}")
    print("="*70)
    print(f"Description: {scenario['description']}")
    print(f"Separation: {scenario['separation']*1e9:.1f} nm")
    print(f"Approach velocity: {scenario['velocity']:.3f}c")
    
    # Create particle system
    particles = create_particle_system(scenario['separation'], scenario['velocity'])
    h = 1e-6  # 1 ns timestep
    
    # Calculate theoretical scales
    gamma_theory = 1.0 / np.sqrt(1 - scenario['velocity']**2)
    coulomb_energy = 1.44e-13 / scenario['separation']  # Rough e²/4πε₀r in MeV
    
    print(f"Theoretical gamma: {gamma_theory:.6f}")
    print(f"Coulomb energy scale: {coulomb_energy:.2e} MeV")
    print(f"EM to rest mass ratio: {coulomb_energy/938.3:.2e}")
    
    try:
        # Test original covariant approach
        particles_orig = {key: np.copy(val) if isinstance(val, np.ndarray) else val 
                         for key, val in particles.items()}
        
        apt_R = np.inf
        sim_type = 2
        result_orig = original_lib.eqsofmotion_static(h, particles_orig, particles_orig, apt_R, sim_type)
        
        # Test standard approach  
        integrator = LiénardWiechertIntegrator()
        particles_new = {key: np.copy(val) if isinstance(val, np.ndarray) else val 
                        for key, val in particles.items()}
        
        result_new = integrator.eqsofmotion_static(h, particles_new, particles_new)
        
        print(f"\\n📊 RESULTS:")
        print(f"{'Method':<20} {'γ₁':<12} {'γ₂':<12} {'Status':<15}")
        print("-" * 65)
        print(f"{'Theoretical':<20} {gamma_theory:.6f} {gamma_theory:.6f} {'Reference':<15}")
        print(f"{'Original (cov)':<20} {result_orig['gamma'][0]:.6f} {result_orig['gamma'][1]:.6f} {'???':<15}")
        print(f"{'New (standard)':<20} {result_new['gamma'][0]:.6f} {result_new['gamma'][1]:.6f} {'✓':<15}")
        
        # Energy-momentum consistency check
        def check_consistency(result, label):
            errors = []
            for i in range(2):
                E = result['Pt'][i]
                px, py, pz = result['Px'][i], result['Py'][i], result['Pz'][i]
                mc2 = particles['m'] * C_MMNS**2
                
                E2 = E**2
                p2c2 = px**2 + py**2 + pz**2
                expected_E2 = p2c2 + mc2**2
                
                rel_error = abs(E2 - expected_E2) / expected_E2
                errors.append(rel_error)
            
            avg_error = np.mean(errors)
            print(f"{label} E-p consistency: {avg_error:.2e}")
            return avg_error
        
        error_orig = check_consistency(result_orig, "Original")
        error_new = check_consistency(result_new, "New")
        
        # Physical validity assessment
        gamma_orig_valid = (result_orig['gamma'][0] >= 1.0 and result_orig['gamma'][1] >= 1.0 and
                           not np.isnan(result_orig['gamma'][0]) and not np.isnan(result_orig['gamma'][1]))
        gamma_new_valid = (result_new['gamma'][0] >= 1.0 and result_new['gamma'][1] >= 1.0)
        
        print(f"\\n🎯 ASSESSMENT:")
        
        if not gamma_orig_valid:
            print("❌ Original: Unphysical gamma values")
            verdict = "Standard approach preferred (original has numerical issues)"
        elif gamma_orig_valid and error_orig < error_new:
            print("✅ Original: Better energy-momentum consistency")
            verdict = "Covariant approach preferred (better physics)"
        elif abs(error_orig - error_new) < 1e-6:
            print("⚖️  Both approaches give similar results")
            verdict = "Either approach acceptable"
        else:
            print("✅ Standard: Better energy-momentum consistency")
            verdict = "Standard approach preferred (better numerics)"
        
        return {
            'scenario': scenario['name'],
            'gamma_theory': gamma_theory,
            'gamma_orig': result_orig['gamma'][0] if gamma_orig_valid else np.nan,
            'gamma_new': result_new['gamma'][0],
            'error_orig': error_orig,
            'error_new': error_new,
            'coulomb_ratio': coulomb_energy/938.3,
            'verdict': verdict,
            'orig_valid': gamma_orig_valid
        }
        
    except Exception as e:
        print(f"❌ Error in scenario: {e}")
        return {
            'scenario': scenario['name'],
            'error': str(e),
            'verdict': 'Failed to execute'
        }


def comprehensive_comparison():
    """Run comprehensive comparison across multiple scenarios."""
    print("🔬 COMPREHENSIVE COVARIANT vs STANDARD COMPARISON")
    print("="*80)
    print("Evaluating the original covariant gamma calculation across different regimes")
    print()
    
    scenarios = create_test_scenarios()
    results = []
    
    for scenario in scenarios:
        result = analyze_scenario(scenario)
        results.append(result)
    
    # Summary analysis
    print(f"\\n📋 SUMMARY ANALYSIS")
    print("="*80)
    
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("❌ No valid results obtained")
        return results
    
    # Count verdicts
    covariant_preferred = sum(1 for r in valid_results if 'covariant' in r['verdict'].lower())
    standard_preferred = sum(1 for r in valid_results if 'standard' in r['verdict'].lower())
    either_ok = sum(1 for r in valid_results if 'either' in r['verdict'].lower())
    
    print(f"Covariant approach preferred: {covariant_preferred}/{len(valid_results)} scenarios")
    print(f"Standard approach preferred: {standard_preferred}/{len(valid_results)} scenarios")
    print(f"Either approach acceptable: {either_ok}/{len(valid_results)} scenarios")
    
    # Field strength correlation
    strong_field_scenarios = [r for r in valid_results if r['coulomb_ratio'] > 1e-6]
    weak_field_scenarios = [r for r in valid_results if r['coulomb_ratio'] <= 1e-6]
    
    print(f"\\nStrong field scenarios: {len(strong_field_scenarios)}")
    print(f"Weak field scenarios: {len(weak_field_scenarios)}")
    
    print(f"\\n🎯 OVERALL RECOMMENDATION:")
    print("="*60)
    
    if covariant_preferred > standard_preferred:
        print("✅ COVARIANT APPROACH GENERALLY PREFERRED")
        print("- Original theoretical derivation appears valid")
        print("- Better physics in electromagnetic field regimes")
        print("- Consider fixing implementation bugs rather than discarding theory")
    elif standard_preferred > covariant_preferred:
        print("✅ STANDARD APPROACH GENERALLY PREFERRED") 
        print("- More numerically stable")
        print("- Sufficient accuracy for most scenarios")
        print("- Covariant effects may be negligible in practice")
    else:
        print("⚖️  BOTH APPROACHES HAVE MERIT")
        print("- Choose based on specific use case")
        print("- Standard for stability, covariant for theoretical completeness")
    
    return results


if __name__ == "__main__":
    results = comprehensive_comparison()
    
    print(f"\\n📄 Analysis complete.")
    print("\\nKey insight: The original covariant derivation should be evaluated")
    print("on its theoretical merits, not dismissed due to implementation issues.")
