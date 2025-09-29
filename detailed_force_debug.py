#!/usr/bin/env python3
"""
Detailed debug of electromagnetic force calculations
"""
import numpy as np
import sys
sys.path.append('/home/benfol/work/LW_windows')

from core.performance import OptimizedLienardWiechertIntegrator

def debug_electromagnetic_force():
    """Debug electromagnetic force calculations step by step"""
    print("🔍 DETAILED ELECTROMAGNETIC FORCE DEBUG")
    print("="*60)
    
    # Simple two-particle setup with proper legacy format
    # Rider: energetic proton-like particle  
    init_rider = {
        'x': np.array([0.0]), 'y': np.array([0.0]), 'z': np.array([0.0]), 't': np.array([0.0]),
        'Px': np.array([1e-6]), 'Py': np.array([0.0]), 'Pz': np.array([0.0]),  # Small momentum
        'Pt': np.array([3.168e3]),  # Estimated total momentum for gamma~3369
        'gamma': np.array([3368.997510]),
        'q': np.array([0.00001179]),
        'm': np.array([1.0]),  # 1 amu
        'char_time': np.array([1e-15]),
        'bx': np.array([0.0]), 'by': np.array([0.0]), 'bz': np.array([0.0]),
        'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0])
    }
    
    # Driver: slow anti-proton-like particle
    init_driver = {
        'x': np.array([0.5]), 'y': np.array([0.0]), 'z': np.array([0.0]), 't': np.array([0.0]),
        'Px': np.array([0.0]), 'Py': np.array([0.0]), 'Pz': np.array([0.0]),
        'Pt': np.array([299.792458]),  # Rest energy in these units
        'gamma': np.array([1.003165]),
        'q': np.array([-0.00063652]),
        'm': np.array([1.0]),  # 1 amu
        'char_time': np.array([1e-15]),
        'bx': np.array([0.0]), 'by': np.array([0.0]), 'bz': np.array([0.0]),
        'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0])
    }
    
    print(f"Setup:")
    print(f"  P1: q={init_rider['q'][0]:.8e}, pos=({init_rider['x'][0]},{init_rider['y'][0]},{init_rider['z'][0]})")
    print(f"  P2: q={init_driver['q'][0]:.8e}, pos=({init_driver['x'][0]},{init_driver['y'][0]},{init_driver['z'][0]})")
    print(f"  q1*q2 = {init_rider['q'][0]*init_driver['q'][0]:.8e}")
    r_vec = np.array([init_driver['x'][0]-init_rider['x'][0], init_driver['y'][0]-init_rider['y'][0], init_driver['z'][0]-init_rider['z'][0]])
    print(f"  Distance = {np.linalg.norm(r_vec):.3f}")
    print()
    
    # Test updated integrator
    print("Testing Updated Integrator:")
    updated = OptimizedLienardWiechertIntegrator()
    
    # Add detailed debug prints to check internal calculations
    print("  Calling single integration step...")
    
    # Get initial state
    p1_initial = np.array([init_rider['Px'][0], init_rider['Py'][0], init_rider['Pz'][0]])
    p2_initial = np.array([init_driver['Px'][0], init_driver['Py'][0], init_driver['Pz'][0]])
    
    # Run integration using the proper method
    rider_traj, driver_traj = updated.integrate_retarded_fields(
        static_steps=1,
        ret_steps=1,
        h_step=1e-9,
        wall_Z=100000.0,
        apt_R=100000.0,
        sim_type=2,
        init_rider=init_rider,
        init_driver=init_driver,
        bunch_dist=0.5,
        z_cutoff=10.0
    )
    
    # Check momentum changes
    if len(rider_traj) > 1:
        p1_final = np.array([rider_traj[1]['Px'][0], rider_traj[1]['Py'][0], rider_traj[1]['Pz'][0]])
        dp1 = p1_final - p1_initial
        print(f"  P1 momentum change: Δpx={dp1[0]:.12e}, Δpy={dp1[1]:.12e}, Δpz={dp1[2]:.12e}")
        print(f"  P1 force estimate: Fx={dp1[0]/1e-9:.12e}, Fy={dp1[1]/1e-9:.12e}, Fz={dp1[2]/1e-9:.12e}")
    else:
        print("  ❌ No rider trajectory data available")
    
    if len(driver_traj) > 1:
        p2_final = np.array([driver_traj[1]['Px'][0], driver_traj[1]['Py'][0], driver_traj[1]['Pz'][0]])
        dp2 = p2_final - p2_initial
        print(f"  P2 momentum change: Δpx={dp2[0]:.12e}, Δpy={dp2[1]:.12e}, Δpz={dp2[2]:.12e}")
        print(f"  P2 force estimate: Fx={dp2[0]/1e-9:.12e}, Fy={dp2[1]/1e-9:.12e}, Fz={dp2[2]/1e-9:.12e}")
    else:
        print("  ❌ No driver trajectory data available")
        
    print()
    
    # Manual force calculation
    print("Manual Coulomb Force Calculation:")
    r_vec = np.array([init_driver['x'][0]-init_rider['x'][0], init_driver['y'][0]-init_rider['y'][0], init_driver['z'][0]-init_rider['z'][0]])
    r_mag = np.linalg.norm(r_vec)
    r_hat = r_vec / r_mag
    
    # Gaussian units: F = q1*q2/r^2 * r_hat
    q1q2 = init_rider['q'][0] * init_driver['q'][0]
    coulomb_force_mag = abs(q1q2) / (r_mag**2)
    coulomb_force_vec = np.sign(q1q2) * coulomb_force_mag * r_hat
    
    print(f"  r_vec = ({r_vec[0]:.3f}, {r_vec[1]:.3f}, {r_vec[2]:.3f})")
    print(f"  r_mag = {r_mag:.3f}")
    print(f"  r_hat = ({r_hat[0]:.3f}, {r_hat[1]:.3f}, {r_hat[2]:.3f})")
    print(f"  q1*q2 = {q1q2:.12e}")
    print(f"  |F| = {coulomb_force_mag:.12e}")
    print(f"  F_vec = ({coulomb_force_vec[0]:.12e}, {coulomb_force_vec[1]:.12e}, {coulomb_force_vec[2]:.12e})")
    print(f"  Expected direction: {'attractive' if q1q2 < 0 else 'repulsive'}")

if __name__ == "__main__":
    debug_electromagnetic_force()