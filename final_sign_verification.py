#!/usr/bin/env python3
"""
FINAL VERIFICATION: Sign error analysis conclusion
"""
import numpy as np
import sys
sys.path.append('/home/benfol/work/LW_windows')

from core.performance import OptimizedLienardWiechertIntegrator

def final_sign_verification():
    """Final test to confirm the electromagnetic force signs are correct"""
    print("🎯 FINAL SIGN ERROR ANALYSIS")
    print("="*60)
    
    # Two oppositely charged particles - should attract
    init_rider = {
        'x': np.array([0.0]), 'y': np.array([0.0]), 'z': np.array([0.0]), 't': np.array([0.0]),
        'Px': np.array([1e-6]), 'Py': np.array([0.0]), 'Pz': np.array([0.0]),
        'Pt': np.array([3.168e3]), 'gamma': np.array([3368.997510]),
        'q': np.array([+0.00001179]),  # POSITIVE charge
        'm': np.array([1.0]), 'char_time': np.array([1e-15]),
        'bx': np.array([0.0]), 'by': np.array([0.0]), 'bz': np.array([0.0]),
        'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0])
    }
    
    init_driver = {
        'x': np.array([0.5]), 'y': np.array([0.0]), 'z': np.array([0.0]), 't': np.array([0.0]),
        'Px': np.array([0.0]), 'Py': np.array([0.0]), 'Pz': np.array([0.0]),
        'Pt': np.array([299.792458]), 'gamma': np.array([1.003165]),
        'q': np.array([-0.00063652]),  # NEGATIVE charge
        'm': np.array([1.0]), 'char_time': np.array([1e-15]),
        'bx': np.array([0.0]), 'by': np.array([0.0]), 'bz': np.array([0.0]),
        'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0])
    }
    
    print("TEST 1: Opposite charges (should attract)")
    print(f"  P1 at (0,0,0): q = +{init_rider['q'][0]:.2e}")
    print(f"  P2 at (0.5,0,0): q = {init_driver['q'][0]:.2e}")
    print(f"  Expected: P1 pulled toward P2 → Force on P1 = (+Fx, 0, 0)")
    print()
    
    # Run single step
    integrator = OptimizedLienardWiechertIntegrator()
    rider_traj, driver_traj = integrator.integrate_retarded_fields(
        static_steps=1, ret_steps=1, h_step=1e-9,
        wall_Z=100000.0, apt_R=100000.0, sim_type=2,
        init_rider=init_rider, init_driver=init_driver,
        bunch_dist=0.5, z_cutoff=10.0
    )
    
    # Check results
    p1_initial = np.array([init_rider['Px'][0], init_rider['Py'][0], init_rider['Pz'][0]])
    p1_final = np.array([rider_traj[1]['Px'][0], rider_traj[1]['Py'][0], rider_traj[1]['Pz'][0]])
    dp1 = p1_final - p1_initial
    force1 = dp1 / 1e-9
    
    print(f"  RESULT: Force on P1 = ({force1[0]:.2e}, {force1[1]:.2e}, {force1[2]:.2e})")
    print(f"  ✅ CORRECT: Fx > 0 (attractive)" if force1[0] > 0 else f"  ❌ WRONG: Fx < 0 (repulsive)")
    print()
    
    # TEST 2: Same sign charges  
    print("TEST 2: Same charges (should repel)")
    init_driver_same = init_driver.copy()
    init_driver_same['q'] = np.array([+0.00063652])  # POSITIVE charge (same as rider)
    
    print(f"  P1 at (0,0,0): q = +{init_rider['q'][0]:.2e}")
    print(f"  P2 at (0.5,0,0): q = +{init_driver_same['q'][0]:.2e}")
    print(f"  Expected: P1 pushed away from P2 → Force on P1 = (-Fx, 0, 0)")
    print()
    
    rider_traj2, driver_traj2 = integrator.integrate_retarded_fields(
        static_steps=1, ret_steps=1, h_step=1e-9,
        wall_Z=100000.0, apt_R=100000.0, sim_type=2,
        init_rider=init_rider, init_driver=init_driver_same,
        bunch_dist=0.5, z_cutoff=10.0
    )
    
    p1_final2 = np.array([rider_traj2[1]['Px'][0], rider_traj2[1]['Py'][0], rider_traj2[1]['Pz'][0]])
    dp1_2 = p1_final2 - p1_initial
    force1_2 = dp1_2 / 1e-9
    
    print(f"  RESULT: Force on P1 = ({force1_2[0]:.2e}, {force1_2[1]:.2e}, {force1_2[2]:.2e})")
    print(f"  ✅ CORRECT: Fx < 0 (repulsive)" if force1_2[0] < 0 else f"  ❌ WRONG: Fx > 0 (attractive)")
    print()
    
    # CONCLUSION
    print("="*60)
    if force1[0] > 0 and force1_2[0] < 0:
        print("🎉 CONCLUSION: ELECTROMAGNETIC SIGNS ARE CORRECT!")
        print("   The updated integrator properly handles:")
        print("   • Opposite charges → Attraction (+Fx)")  
        print("   • Same charges → Repulsion (-Fx)")
        print("   • The magnitude includes relativistic effects (~3000x Coulomb)")
        print("\n   The 'sign error' was in the manual calculation expectation,")
        print("   not in the integrator physics!")
    else:
        print("❌ CONCLUSION: There is still a sign error in the integrator")
        
if __name__ == "__main__":
    final_sign_verification()