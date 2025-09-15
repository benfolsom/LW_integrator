#!/usr/bin/env python3
"""
Cross-Verification: Legacy vs Basic vs Optimized

This script directly compares all three integrator implementations to verify
that the refactoring maintains perfect physics agreement.
"""

import numpy as np
import time
import sys
import pickle
import os

# Add paths for imports
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator')
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator/legacy')

from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator
from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator

print("🔬 CROSS-VERIFICATION: Legacy vs Basic vs Optimized")
print("="*70)


def check_legacy_imports():
    """Check if legacy components are available."""
    try:
        from bunch_inits import init_bunch
        from covariant_integrator_library import retarded_integrator3
        print("✅ Legacy integrator imported successfully")
        return True, init_bunch, retarded_integrator3
    except ImportError as e:
        print(f"❌ Legacy integrator not available: {e}")
        return False, None, None


def setup_simple_test_case():
    """Create a simple test case using legacy parameter format."""
    # Use the same parameters as in legacy_verification.py
    
    # Physical constants
    c_ms = 299792458
    
    # Particle parameters
    transv_dist = 1e-4
    m_particle_rider = 1.007319468  # proton - amu
    m_particle_driver = 1.007319468  # proton - amu (simplified)
    stripped_ions_rider = 1.
    stripped_ions_driver = 1.
    charge_sign_rider = -1.
    charge_sign_driver = 1.
    
    # Initial momentum and position (reduced for simple test)
    starting_Pz_rider = 1e5   # Reduced energy for stability
    starting_Pz_driver = -starting_Pz_rider  # Head-on collision
    transv_mom_rider = 0.
    transv_mom_driver = 0.
    starting_distance_rider = 1e-6
    starting_distance_driver = 100.
    
    # Particle counts (simplified)
    pcount_rider = 1
    pcount_driver = 1
    
    return {
        'c_ms': c_ms, 
        'transv_dist': transv_dist,
        'm_particle_rider': m_particle_rider, 
        'm_particle_driver': m_particle_driver,
        'stripped_ions_rider': stripped_ions_rider, 
        'stripped_ions_driver': stripped_ions_driver,
        'charge_sign_rider': charge_sign_rider, 
        'charge_sign_driver': charge_sign_driver,
        'starting_Pz_rider': starting_Pz_rider, 
        'starting_Pz_driver': starting_Pz_driver,
        'transv_mom_rider': transv_mom_rider, 
        'transv_mom_driver': transv_mom_driver,
        'starting_distance_rider': starting_distance_rider, 
        'starting_distance_driver': starting_distance_driver,
        'pcount_rider': pcount_rider, 
        'pcount_driver': pcount_driver
    }


def run_legacy_integration(params, steps=10, dt=1e-9):
    """Run legacy integrator with given parameters."""
    legacy_available, init_bunch, retarded_integrator3 = check_legacy_imports()
    
    if not legacy_available:
        return None
    
    print(f"🔄 Running LEGACY integration...")
    print(f"  Particles: {params['pcount_rider'] + params['pcount_driver']}")
    print(f"  Steps: {steps}")
    print(f"  Step size: {dt:.2e}")
    
    start_time = time.time()
    
    # Initialize rider bunch
    init_rider, E_MeV_rest_rider = init_bunch(
        params['starting_distance_rider'], params['transv_mom_rider'], 
        params['starting_Pz_rider'], params['stripped_ions_rider'],
        params['m_particle_rider'], params['transv_dist'], 
        params['pcount_rider'], params['charge_sign_rider']
    )
    
    # Initialize driver bunch  
    init_driver, E_MeV_rest_driver = init_bunch(
        params['starting_distance_driver'], params['transv_mom_driver'], 
        params['starting_Pz_driver'], params['stripped_ions_driver'],
        params['m_particle_driver'], params['transv_dist'], 
        params['pcount_driver'], params['charge_sign_driver']
    )
    
    # Combine into single arrays for integration (create W matrix format)
    n_rider = params['pcount_rider']
    n_driver = params['pcount_driver']
    n_total = n_rider + n_driver
    
    # Create W matrix (legacy format: x, y, z, px, py, pz, t for each particle)
    W_combined = np.zeros((n_total, 7))
    qdq_combined = np.zeros(n_total)
    
    # Fill in rider particles
    for i in range(n_rider):
        W_combined[i, 0] = init_rider['x'][i]
        W_combined[i, 1] = init_rider['y'][i] 
        W_combined[i, 2] = init_rider['z'][i]
        W_combined[i, 3] = init_rider['Px'][i]
        W_combined[i, 4] = init_rider['Py'][i]
        W_combined[i, 5] = init_rider['Pz'][i]
        W_combined[i, 6] = init_rider['t'][i]
        qdq_combined[i] = init_rider['q']  # q is a single value for all rider particles
    
    # Fill in driver particles
    for i in range(n_driver):
        idx = n_rider + i
        W_combined[idx, 0] = init_driver['x'][i]
        W_combined[idx, 1] = init_driver['y'][i]
        W_combined[idx, 2] = init_driver['z'][i]
        W_combined[idx, 3] = init_driver['Px'][i]
        W_combined[idx, 4] = init_driver['Py'][i]
        W_combined[idx, 5] = init_driver['Pz'][i]
        W_combined[idx, 6] = init_driver['t'][i]
        qdq_combined[idx] = init_driver['q']  # q is a single value for all driver particles
    
    # Run integration
    W_final = retarded_integrator3(W_combined, qdq_combined, dt, steps)
    
    computation_time = time.time() - start_time
    
    print(f"  ✅ Completed in {computation_time:.4f} seconds")
    
    return {
        'W_initial': W_combined,
        'W_final': W_final,
        'qdq': qdq_combined,
        'computation_time': computation_time,
        'success': True
    }


def convert_legacy_to_dict(W_final, qdq):
    """Convert legacy format to dictionary format for comparison."""
    n_particles = len(qdq)
    
    # Convert momentum to velocity (legacy uses momentum, modern uses velocity)
    particles = {
        'x': W_final[:n_particles, 0],
        'y': W_final[:n_particles, 1],
        'z': W_final[:n_particles, 2],
        'vx': np.zeros(n_particles),  # Will calculate from momentum
        'vy': np.zeros(n_particles),  
        'vz': np.zeros(n_particles),
        'm': np.array([1.007319468 * 931.494102e6 for _ in range(n_particles)]),  # Convert amu to eV/c²
        'q': qdq  # Already in correct units
    }
    
    # Convert momentum to velocity: v = p / (gamma * m)
    c_mmns = 299.792458  # mm/ns
    for i in range(n_particles):
        px, py, pz = W_final[i, 3], W_final[i, 4], W_final[i, 5]
        m_amu = 1.007319468  # proton mass in amu
        
        # Calculate gamma from total momentum
        p_total = np.sqrt(px**2 + py**2 + pz**2)
        gamma = np.sqrt(1 + (p_total / (m_amu * c_mmns))**2)
        
        # Convert to velocity
        particles['vx'][i] = px / (gamma * m_amu) * 1e6  # Convert mm/ns to m/s
        particles['vy'][i] = py / (gamma * m_amu) * 1e6
        particles['vz'][i] = pz / (gamma * m_amu) * 1e6
    
    return particles


def legacy_to_modern_particles(W_initial, qdq):
    """Convert legacy initial conditions to modern format."""
    n_particles = len(qdq)
    
    particles = {
        'x': W_initial[:n_particles, 0] * 1e-3,  # Convert mm to m
        'y': W_initial[:n_particles, 1] * 1e-3,
        'z': W_initial[:n_particles, 2] * 1e-3,
        'vx': np.zeros(n_particles),
        'vy': np.zeros(n_particles),
        'vz': np.zeros(n_particles),
        'm': np.array([1.007319468 * 931.494102e6 for _ in range(n_particles)]),  # amu to eV/c²
        'q': qdq  # Already in correct format from legacy
    }
    
    # Convert momentum to velocity
    c_mmns = 299.792458  # mm/ns
    for i in range(n_particles):
        px, py, pz = W_initial[i, 3], W_initial[i, 4], W_initial[i, 5]
        m_amu = 1.007319468  # proton mass in amu
        
        # Calculate gamma from total momentum
        p_total = np.sqrt(px**2 + py**2 + pz**2)
        gamma = np.sqrt(1 + (p_total / (m_amu * c_mmns))**2)
        
        # Convert to velocity in m/s
        particles['vx'][i] = px / (gamma * m_amu) * 1e6  # mm/ns to m/s
        particles['vy'][i] = py / (gamma * m_amu) * 1e6
        particles['vz'][i] = pz / (gamma * m_amu) * 1e6
    
    return particles


def run_modern_integration(integrator, particles, steps, dt, name):
    """Run basic or optimized integrator."""
    print(f"🔄 Running {name} integration...")
    print(f"  Particles: {len(particles['x'])}")
    print(f"  Steps: {steps}")
    print(f"  Step size: {dt:.2e}")
    
    current_particles = {k: v.copy() for k, v in particles.items()}
    
    start_time = time.time()
    
    # Integration loop
    for step in range(steps):
        updated_particles = integrator.eqsofmotion_static(
            dt, current_particles, current_particles
        )
        current_particles = updated_particles
    
    computation_time = time.time() - start_time
    
    print(f"  ✅ Completed in {computation_time:.4f} seconds")
    
    return {
        'final_particles': current_particles,
        'computation_time': computation_time,
        'success': True
    }


def compare_final_states(legacy_particles, basic_particles, opt_particles):
    """Compare final states from all three integrators."""
    print(f"\n🔍 CROSS-INTEGRATOR COMPARISON")
    print("-"*50)
    
    n_particles = len(legacy_particles['x'])
    
    # Compare Legacy vs Basic
    max_pos_diff_lb = 0.0
    max_vel_diff_lb = 0.0
    
    # Compare Legacy vs Optimized
    max_pos_diff_lo = 0.0
    max_vel_diff_lo = 0.0
    
    # Compare Basic vs Optimized
    max_pos_diff_bo = 0.0
    max_vel_diff_bo = 0.0
    
    for i in range(n_particles):
        # Legacy vs Basic
        pos_diff_lb = max(abs(legacy_particles['x'][i] - basic_particles['x'][i]),
                         abs(legacy_particles['y'][i] - basic_particles['y'][i]),
                         abs(legacy_particles['z'][i] - basic_particles['z'][i]))
        vel_diff_lb = max(abs(legacy_particles['vx'][i] - basic_particles['vx'][i]),
                         abs(legacy_particles['vy'][i] - basic_particles['vy'][i]),
                         abs(legacy_particles['vz'][i] - basic_particles['vz'][i]))
        
        # Legacy vs Optimized
        pos_diff_lo = max(abs(legacy_particles['x'][i] - opt_particles['x'][i]),
                         abs(legacy_particles['y'][i] - opt_particles['y'][i]),
                         abs(legacy_particles['z'][i] - opt_particles['z'][i]))
        vel_diff_lo = max(abs(legacy_particles['vx'][i] - opt_particles['vx'][i]),
                         abs(legacy_particles['vy'][i] - opt_particles['vy'][i]),
                         abs(legacy_particles['vz'][i] - opt_particles['vz'][i]))
        
        # Basic vs Optimized
        pos_diff_bo = max(abs(basic_particles['x'][i] - opt_particles['x'][i]),
                         abs(basic_particles['y'][i] - opt_particles['y'][i]),
                         abs(basic_particles['z'][i] - opt_particles['z'][i]))
        vel_diff_bo = max(abs(basic_particles['vx'][i] - opt_particles['vx'][i]),
                         abs(basic_particles['vy'][i] - opt_particles['vy'][i]),
                         abs(basic_particles['vz'][i] - opt_particles['vz'][i]))
        
        max_pos_diff_lb = max(max_pos_diff_lb, pos_diff_lb)
        max_vel_diff_lb = max(max_vel_diff_lb, vel_diff_lb)
        max_pos_diff_lo = max(max_pos_diff_lo, pos_diff_lo)
        max_vel_diff_lo = max(max_vel_diff_lo, vel_diff_lo)
        max_pos_diff_bo = max(max_pos_diff_bo, pos_diff_bo)
        max_vel_diff_bo = max(max_vel_diff_bo, vel_diff_bo)
    
    print(f"Legacy vs Basic:")
    print(f"  Position difference: {max_pos_diff_lb:.2e}")
    print(f"  Velocity difference: {max_vel_diff_lb:.2e}")
    
    print(f"Legacy vs Optimized:")
    print(f"  Position difference: {max_pos_diff_lo:.2e}")
    print(f"  Velocity difference: {max_vel_diff_lo:.2e}")
    
    print(f"Basic vs Optimized:")
    print(f"  Position difference: {max_pos_diff_bo:.2e}")
    print(f"  Velocity difference: {max_vel_diff_bo:.2e}")
    
    # Assessment
    tolerances = [
        (1e-14, "🎯 PERFECT MATCH - Machine precision agreement!"),
        (1e-10, "✅ EXCELLENT AGREEMENT - Within numerical precision"),
        (1e-6, "⚠️  GOOD AGREEMENT - Small differences"),
        (float('inf'), "❌ SIGNIFICANT DIFFERENCES - Investigation needed")
    ]
    
    def assess_differences(pos_diff, vel_diff):
        for tol, message in tolerances:
            if pos_diff < tol and vel_diff < tol:
                return message
        return tolerances[-1][1]
    
    print(f"\nAssessment:")
    print(f"  Legacy vs Basic: {assess_differences(max_pos_diff_lb, max_vel_diff_lb)}")
    print(f"  Legacy vs Optimized: {assess_differences(max_pos_diff_lo, max_vel_diff_lo)}")
    print(f"  Basic vs Optimized: {assess_differences(max_pos_diff_bo, max_vel_diff_bo)}")
    
    # Return comparison results
    return {
        'legacy_vs_basic': {
            'max_pos_diff': max_pos_diff_lb,
            'max_vel_diff': max_vel_diff_lb,
            'perfect_match': max_pos_diff_lb < 1e-14 and max_vel_diff_lb < 1e-14
        },
        'legacy_vs_optimized': {
            'max_pos_diff': max_pos_diff_lo,
            'max_vel_diff': max_vel_diff_lo,
            'perfect_match': max_pos_diff_lo < 1e-14 and max_vel_diff_lo < 1e-14
        },
        'basic_vs_optimized': {
            'max_pos_diff': max_pos_diff_bo,
            'max_vel_diff': max_vel_diff_bo,
            'perfect_match': max_pos_diff_bo < 1e-14 and max_vel_diff_bo < 1e-14
        }
    }


def main():
    """Main cross-verification function."""
    
    # Setup test parameters
    params = setup_simple_test_case()
    steps = 10
    dt = 1e-9
    
    print(f"Test Parameters:")
    print(f"  Total particles: {params['pcount_rider'] + params['pcount_driver']}")
    print(f"  Integration steps: {steps}")
    print(f"  Time step: {dt:.2e} seconds")
    print()
    
    # Run legacy integration
    legacy_result = run_legacy_integration(params, steps, dt)
    
    if legacy_result is None:
        print("❌ Cannot perform cross-verification without legacy integrator")
        return
    
    # Convert legacy initial conditions to modern format
    initial_particles = legacy_to_modern_particles(legacy_result['W_initial'], legacy_result['qdq'])
    
    # Initialize modern integrators
    basic_integrator = LienardWiechertIntegrator()
    opt_integrator = OptimizedLienardWiechertIntegrator()
    
    # Run basic integration
    basic_result = run_modern_integration(
        basic_integrator, initial_particles, steps, dt, "BASIC"
    )
    
    # Run optimized integration
    opt_result = run_modern_integration(
        opt_integrator, initial_particles, steps, dt, "OPTIMIZED"
    )
    
    # Convert legacy final state to dictionary format
    legacy_final = convert_legacy_to_dict(legacy_result['W_final'], legacy_result['qdq'])
    
    # Compare all three results
    comparison = compare_final_states(
        legacy_final, 
        basic_result['final_particles'], 
        opt_result['final_particles']
    )
    
    # Performance summary
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("-"*30)
    print(f"Legacy time:    {legacy_result['computation_time']:.4f}s")
    print(f"Basic time:     {basic_result['computation_time']:.4f}s")
    print(f"Optimized time: {opt_result['computation_time']:.4f}s")
    
    if basic_result['computation_time'] > 0 and opt_result['computation_time'] > 0:
        basic_vs_opt_speedup = basic_result['computation_time'] / opt_result['computation_time']
        print(f"Basic→Optimized speedup: {basic_vs_opt_speedup:.2f}x")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("="*30)
    
    if (comparison['legacy_vs_basic']['perfect_match'] and 
        comparison['legacy_vs_optimized']['perfect_match'] and 
        comparison['basic_vs_optimized']['perfect_match']):
        print("✅ ALL THREE INTEGRATORS PRODUCE IDENTICAL RESULTS!")
        print("   The refactoring maintains perfect physics accuracy.")
    else:
        print("⚠️  Some differences detected between integrators.")
        print("   Further investigation may be needed.")


if __name__ == "__main__":
    main()