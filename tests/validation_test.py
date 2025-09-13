#!/usr/bin/env python3
"""
Self-Consistent Integrator Validation Test

This test verifies that the new self-consistent Lienard-Wiechert integrator
produces equivalent results to the original retarded integration algorithm.
Provides documented validation for the physics correctness of the refactored package.

Author: Ben Folsom  
Date: September 13, 2025
Purpose: Production validation of self-consistent integrator
"""

import sys
import os
import numpy as np

# Add the package to the path
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator')

def test_integrator_equivalence():
    """
    Test that self-consistent integrator produces equivalent results to original.
    
    This test verifies the physics correctness of the refactored implementation
    by comparing final particle states between the two methods.
    """
    print("🔬 SELF-CONSISTENT vs ORIGINAL INTEGRATOR VALIDATION")
    print("="*60)
    
    try:
        from lw_integrator import (
            SelfConsistentLiénardWiechertIntegrator,
            SimulationType, 
            create_simulation_config
        )
        from lw_integrator.core.integration import retarded_integrator
        
        # Test configuration for comparison
        config = create_simulation_config(
            SimulationType.FREE_PARTICLE_BUNCHES,
            dt=1e-16,
            convergence_tolerance=1e-8,
            max_iterations=5,
            debug_mode=False  # Disable debug for clean comparison
        )
        
        # Test particle states - slightly relativistic scenario
        init_rider = {
            'x': np.array([0.0]), 'y': np.array([0.0]), 'z': np.array([1e-3]),
            'vx': np.array([0.0]), 'vy': np.array([0.0]), 'vz': np.array([0.1*299.8]),
            'gamma': np.array([1.005]),
            'Px': np.array([0.0]), 'Py': np.array([0.0]), 'Pz': np.array([100.0]),
            'Pt': np.array([938.3]),
            'bx': np.array([0.0]), 'by': np.array([0.0]), 'bz': np.array([0.1]),
            'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0]),
            't': np.array([0.0]), 'q': 1.0, 'm': 938.3, 'char_time': np.array([1e-4])
        }
        
        init_driver = {
            'x': np.array([0.0]), 'y': np.array([0.0]), 'z': np.array([0.0]),
            'vx': np.array([0.0]), 'vy': np.array([0.0]), 'vz': np.array([0.0]),
            'gamma': np.array([1.0]),
            'Px': np.array([0.0]), 'Py': np.array([0.0]), 'Pz': np.array([938.3]),
            'Pt': np.array([938.3]),
            'bx': np.array([0.0]), 'by': np.array([0.0]), 'bz': np.array([0.0]),
            'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0]),
            't': np.array([0.0]), 'q': 1.0, 'm': 938.3, 'char_time': np.array([1e-4])
        }
        
        # Integration parameters
        steps_tot = 50  # Small number for testing
        h_step = 1e-16
        wall_Z = 0.0
        apt_R = 1e-3
        
        print("Running original retarded integrator...")
        original_result = retarded_integrator(
            1, steps_tot-1, h_step, wall_Z, apt_R, 
            int(SimulationType.FREE_PARTICLE_BUNCHES),
            init_rider, init_driver, 1e-2, 1e-2, 1e-2
        )
        
        print("Running self-consistent integrator...")
        integrator = SelfConsistentLiénardWiechertIntegrator(config)
        new_result = integrator.integrate(
            init_rider, init_driver, steps_tot, h_step, wall_Z, apt_R
        )
        
        # Extract final states for comparison
        original_final = original_result[0][-1]  # Final rider state
        new_final = new_result[0][-1]           # Final rider state
        
        # Compare key physics quantities
        def extract_scalar(value):
            """Extract scalar from potential array."""
            return value[0] if hasattr(value, '__iter__') and len(value) > 0 else value
        
        original_gamma = extract_scalar(original_final['gamma'])
        new_gamma = extract_scalar(new_final['gamma'])
        
        original_z = extract_scalar(original_final['z'])
        new_z = extract_scalar(new_final['z'])
        
        original_vz = extract_scalar(original_final['vz'])
        new_vz = extract_scalar(new_final['vz'])
        
        # Calculate relative differences
        gamma_error = abs(new_gamma - original_gamma) / original_gamma
        position_error = abs(new_z - original_z) / abs(original_z) if original_z != 0 else abs(new_z)
        velocity_error = abs(new_vz - original_vz) / abs(original_vz) if original_vz != 0 else abs(new_vz)
        
        print(f"\nComparison Results:")
        print(f"  Original final γ: {original_gamma:.8f}")
        print(f"  Self-consistent γ: {new_gamma:.8f}")
        print(f"  Relative γ error: {gamma_error:.2e}")
        print()
        print(f"  Original final z: {original_z:.6e} mm")
        print(f"  Self-consistent z: {new_z:.6e} mm")
        print(f"  Relative z error: {position_error:.2e}")
        print()
        print(f"  Original final vz: {original_vz:.6e} mm/ns")
        print(f"  Self-consistent vz: {new_vz:.6e} mm/ns")
        print(f"  Relative vz error: {velocity_error:.2e}")
        
        # Validation criteria
        max_allowed_error = 1e-6  # 0.0001% tolerance
        
        if gamma_error < max_allowed_error and position_error < max_allowed_error and velocity_error < max_allowed_error:
            print(f"\n✅ VALIDATION PASSED!")
            print(f"   Self-consistent integrator matches original results within {max_allowed_error:.0e} tolerance")
            print(f"   Maximum error: {max(gamma_error, position_error, velocity_error):.2e}")
            return True
        else:
            print(f"\n⚠️  VALIDATION NEEDS REVIEW")
            print(f"   Some errors exceed {max_allowed_error:.0e} tolerance")
            print(f"   Consider adjusting convergence parameters or timestep")
            print(f"   Maximum error: {max(gamma_error, position_error, velocity_error):.2e}")
            return False
            
    except ImportError as e:
        print(f"⚠️  Cannot run comparison - missing modules: {e}")
        print("This test requires the refactored package to be complete.")
        return False
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_physics_conservation():
    """
    Test that basic physics conservation laws are maintained.
    
    Verifies energy-momentum conservation in the self-consistent formulation.
    """
    print("\n🔬 PHYSICS CONSERVATION TEST")
    print("="*40)
    
    try:
        from lw_integrator import (
            SelfConsistentLiénardWiechertIntegrator,
            SimulationType, 
            create_simulation_config
        )
        
        # Free particle test (no external forces)
        config = create_simulation_config(
            SimulationType.FREE_PARTICLE_BUNCHES,
            dt=1e-17,  # Very small timestep
            convergence_tolerance=1e-10,
            max_iterations=3
        )
        
        # Single particle, should conserve energy
        init_rider = {
            'x': np.array([0.0]), 'y': np.array([0.0]), 'z': np.array([0.0]),
            'vx': np.array([100.0]), 'vy': np.array([0.0]), 'vz': np.array([50.0]),
            'gamma': np.array([1.1]),  # Mildly relativistic
            'Px': np.array([100.0]), 'Py': np.array([0.0]), 'Pz': np.array([50.0]),
            'Pt': np.array([1000.0]),
            'bx': np.array([0.334]), 'by': np.array([0.0]), 'bz': np.array([0.167]),
            'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0]),
            't': np.array([0.0]), 'q': 1.0, 'm': 938.3, 'char_time': np.array([1e-4])
        }
        
        # No driver particle for this test
        init_driver = {
            'x': np.array([1e6]), 'y': np.array([1e6]), 'z': np.array([1e6]),  # Very far away
            'vx': np.array([0.0]), 'vy': np.array([0.0]), 'vz': np.array([0.0]),
            'gamma': np.array([1.0]),
            'Px': np.array([0.0]), 'Py': np.array([0.0]), 'Pz': np.array([938.3]),
            'Pt': np.array([938.3]),
            'bx': np.array([0.0]), 'by': np.array([0.0]), 'bz': np.array([0.0]),
            'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0]),
            't': np.array([0.0]), 'q': 1.0, 'm': 938.3, 'char_time': np.array([1e-4])
        }
        
        steps_tot = 10  # Short integration
        h_step = 1e-17
        
        integrator = SelfConsistentLiénardWiechertIntegrator(config)
        result = integrator.integrate(
            init_rider, init_driver, steps_tot, h_step, 0.0, 1e-3
        )
        
        # Check energy conservation
        initial_gamma = init_rider['gamma'][0]
        final_gamma = result[0][-1]['gamma']
        if hasattr(final_gamma, '__iter__'):
            final_gamma = final_gamma[0]
            
        energy_change = abs(final_gamma - initial_gamma) / initial_gamma
        
        print(f"  Initial γ: {initial_gamma:.8f}")
        print(f"  Final γ: {final_gamma:.8f}")
        print(f"  Relative energy change: {energy_change:.2e}")
        
        if energy_change < 1e-8:  # Very tight tolerance for free particle
            print("✅ Energy conservation maintained")
            return True
        else:
            print("⚠️  Energy not well conserved - check implementation")
            return False
            
    except Exception as e:
        print(f"❌ Conservation test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 SELF-CONSISTENT INTEGRATOR VALIDATION SUITE")
    print("="*80)
    print("Comprehensive validation of the refactored LW integrator package")
    print()
    
    tests = [
        test_integrator_equivalence,
        test_physics_conservation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL {total} VALIDATION TESTS PASSED!")
        print("✅ Self-consistent integrator is validated for production use")
        print("✅ Physics equivalence confirmed with original implementation")
        print("✅ Conservation laws maintained")
    else:
        print(f"⚠️  {passed}/{total} validation tests passed")
        print("Some issues may need attention before production deployment")
    
    print("\n📝 This validation test should be run after any major changes")
    print("   to the self-consistent integrator implementation.")
    
    sys.exit(0 if passed == total else 1)