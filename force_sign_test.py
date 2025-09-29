# Direct Force Analysis - Isolated Sign Error Test
import numpy as np
import sys
sys.path.append('/home/benfol/work/LW_windows')

from core.trajectory_integrator import LienardWiechertIntegrator
sys.path.append('legacy')
from covariant_integrator_library import retarded_integrator3

print("🔍 DIRECT ELECTROMAGNETIC FORCE ANALYSIS")
print("="*60)

# Create identical two-particle setup
c_mmns = 299.792458
mass1, mass2 = 1.007319468, 207.2
q1, q2 = 1.0 * 1.178734E-5, -54.0 * 1.178734E-5
pz1 = 1.01e6 * mass1
gamma1 = np.sqrt(1 + (pz1/(mass1*c_mmns))**2)
pz2 = -pz1 / 207.2 * 1.007319468
gamma2 = np.sqrt(1 + (pz2/(mass2*c_mmns))**2)

# IDENTICAL initial positions for both integrators
separation = 0.5  # mm

# Updated integrator format
updated_p1 = {
    'x': np.array([0.0]), 'y': np.array([0.0]), 'z': np.array([0.0]), 't': np.array([0.0]),
    'Px': np.array([0.0]), 'Py': np.array([0.0]), 'Pz': np.array([pz1]),
    'Pt': np.array([gamma1 * mass1 * c_mmns]), 'gamma': np.array([gamma1]),
    'bx': np.array([0.0]), 'by': np.array([0.0]), 'bz': np.array([pz1/(gamma1*mass1*c_mmns)]),
    'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0]),
    'q': np.array([q1]), 'm': np.array([mass1]), 'char_time': np.array([2/3 * q1**2 / (mass1 * c_mmns**3)])
}

updated_p2 = {
    'x': np.array([separation]), 'y': np.array([0.0]), 'z': np.array([0.0]), 't': np.array([0.0]),
    'Px': np.array([0.0]), 'Py': np.array([0.0]), 'Pz': np.array([pz2]),
    'Pt': np.array([gamma2 * mass2 * c_mmns]), 'gamma': np.array([gamma2]),
    'bx': np.array([0.0]), 'by': np.array([0.0]), 'bz': np.array([pz2/(gamma2*mass2*c_mmns)]),
    'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0]),
    'q': np.array([q2]), 'm': np.array([mass2]), 'char_time': np.array([2/3 * q2**2 / (mass2 * c_mmns**3)])
}

# Legacy integrator format (identical positions)
legacy_p1 = {
    'x': np.array([0.0]), 'y': np.array([0.0]), 'z': np.array([0.0]), 't': np.array([0.0]),
    'Px': np.array([0.0]), 'Py': np.array([0.0]), 'Pz': np.array([pz1]),
    'Pt': np.array([gamma1 * mass1 * c_mmns]), 'gamma': np.array([gamma1]),
    'bx': np.array([0.0]), 'by': np.array([0.0]), 'bz': np.array([pz1/(gamma1*mass1*c_mmns)]),
    'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0]),
    'q': q1, 'm': mass1, 'char_time': 2/3 * q1**2 / (mass1 * c_mmns**3)
}

legacy_p2 = {
    'x': np.array([separation]), 'y': np.array([0.0]), 'z': np.array([0.0]), 't': np.array([0.0]),
    'Px': np.array([0.0]), 'Py': np.array([0.0]), 'Pz': np.array([pz2]),
    'Pt': np.array([gamma2 * mass2 * c_mmns]), 'gamma': np.array([gamma2]),
    'bx': np.array([0.0]), 'by': np.array([0.0]), 'bz': np.array([pz2/(gamma2*mass2*c_mmns)]),
    'bdotx': np.array([0.0]), 'bdoty': np.array([0.0]), 'bdotz': np.array([0.0]),
    'q': q2, 'm': mass2, 'char_time': 2/3 * q2**2 / (mass2 * c_mmns**3)
}

print(f"Initial setup:")
print(f"  P1: pos=(0,0,0), q={q1:.8f}, γ={gamma1:.6f}")
print(f"  P2: pos=({separation},0,0), q={q2:.8f}, γ={gamma2:.6f}")
print(f"  Expected: ATTRACTIVE force (q1*q2 = {q1*q2:.8e} < 0)")

# Run single integration step
test_params = {'static_steps': 1, 'ret_steps': 1, 'step_size': 1e-9, 'sim_type': 2,
               'wall_pos': 1e5, 'aperture': 1e5, 'bunch_dist': 1e5, 'z_cutoff': 0, 'cav_spacing': 1e5}

print(f"\nRunning single step integration (h={test_params['step_size']:.0e})...")

# Updated integrator
integrator = LienardWiechertIntegrator()
updated_traj1, updated_traj2 = integrator.integrate_retarded_fields(
    test_params['static_steps'], test_params['ret_steps'], test_params['step_size'],
    test_params['wall_pos'], test_params['aperture'], test_params['sim_type'],
    updated_p1, updated_p2, test_params['bunch_dist'], test_params['z_cutoff'], test_params['cav_spacing']
)

# Legacy integrator  
legacy_traj1, legacy_traj2 = retarded_integrator3(
    test_params['static_steps'], test_params['ret_steps'], test_params['step_size'],
    test_params['wall_pos'], test_params['aperture'], test_params['sim_type'],
    legacy_p1, legacy_p2, test_params['bunch_dist'], test_params['cav_spacing'], test_params['z_cutoff']
)

print(f"✅ Integration completed")

# Analyze momentum changes (which reflect applied forces)
print(f"\n=== MOMENTUM CHANGE ANALYSIS ===")
if len(updated_traj1) >= 2 and len(legacy_traj1) >= 2:
    # Get momentum at retarded step (skip static step)
    updated_dp1_x = updated_traj1[-1]['Px'][0] - updated_traj1[1]['Px'][0]  # Retarded change only
    updated_dp1_y = updated_traj1[-1]['Py'][0] - updated_traj1[1]['Py'][0]
    updated_dp1_z = updated_traj1[-1]['Pz'][0] - updated_traj1[1]['Pz'][0]
    
    legacy_dp1_x = legacy_traj1[-1]['Px'][0] - legacy_traj1[1]['Px'][0]
    legacy_dp1_y = legacy_traj1[-1]['Py'][0] - legacy_traj1[1]['Py'][0] 
    legacy_dp1_z = legacy_traj1[-1]['Pz'][0] - legacy_traj1[1]['Pz'][0]
    
    dt = test_params['step_size']
    
    print(f"Particle 1 momentum changes (retarded step only):")
    print(f"  Updated: Δpx={updated_dp1_x:.12f}, Δpy={updated_dp1_y:.12f}, Δpz={updated_dp1_z:.12f}")
    print(f"  Legacy:  Δpx={legacy_dp1_x:.12f}, Δpy={legacy_dp1_y:.12f}, Δpz={legacy_dp1_z:.12f}")
    
    # Convert to forces (F = dp/dt)
    updated_fx, updated_fy, updated_fz = updated_dp1_x/dt, updated_dp1_y/dt, updated_dp1_z/dt
    legacy_fx, legacy_fy, legacy_fz = legacy_dp1_x/dt, legacy_dp1_y/dt, legacy_dp1_z/dt
    
    print(f"\nForces on Particle 1:")
    print(f"  Updated: Fx={updated_fx:.8e}, Fy={updated_fy:.8e}, Fz={updated_fz:.8e}")
    print(f"  Legacy:  Fx={legacy_fx:.8e}, Fy={legacy_fy:.8e}, Fz={legacy_fz:.8e}")
    
    # Sign analysis
    print(f"\n=== SIGN ANALYSIS ===")
    print(f"X-force signs: Updated={np.sign(updated_fx):+.0f}, Legacy={np.sign(legacy_fx):+.0f}")
    if abs(updated_fx) > 1e-15 and abs(legacy_fx) > 1e-15:
        if np.sign(updated_fx) != np.sign(legacy_fx):
            print(f"🚨 X-FORCE SIGN ERROR DETECTED!")
        else:
            print(f"✅ X-force signs match")
    
    # Expected X-force direction: P1 should be pulled toward P2 (positive X direction)
    # Since P2 is at (+0.5, 0, 0) and P1 is at (0, 0, 0), attractive force should be +X
    expected_fx_sign = +1  # Attractive force toward +X
    print(f"\nExpected vs Actual X-force direction:")
    print(f"  Expected: {expected_fx_sign:+.0f} (P1 pulled toward P2 at +X)")
    print(f"  Updated:  {np.sign(updated_fx):+.0f} {'✅ CORRECT' if np.sign(updated_fx) == expected_fx_sign else '❌ WRONG'}")
    print(f"  Legacy:   {np.sign(legacy_fx):+.0f} {'✅ CORRECT' if np.sign(legacy_fx) == expected_fx_sign else '❌ WRONG'}")

else:
    print("❌ Insufficient trajectory points for analysis")

print("="*60)