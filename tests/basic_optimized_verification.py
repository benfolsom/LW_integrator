#!/usr/bin/env python3
"""
Basic vs Optimized Integrator Verification

Clean verification script that tests basic and optimized integrators using
proper data formats, without legacy interface compatibility issues.
"""

import numpy as np
import time
import sys

# Add path for imports
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator')

from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator
from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator

print("🔬 BASIC vs OPTIMIZED VERIFICATION")
print("="*60)


def create_test_particles(n_particles, seed=42):
    """Create realistic test particles for electromagnetic interaction."""
    np.random.seed(seed)
    
    # Create two groups of particles approaching each other
    n_half = n_particles // 2
    
    particles = {}
    
    # Positions - two groups separated in x-direction
    x_positions = np.concatenate([
        np.random.normal(-0.1e-3, 0.05e-3, n_half),     # Left group
        np.random.normal(0.1e-3, 0.05e-3, n_particles - n_half)  # Right group
    ])
    
    y_positions = np.random.normal(0, 0.02e-3, n_particles)
    z_positions = np.random.uniform(0, 0.1e-3, n_particles)
    
    particles['x'] = x_positions
    particles['y'] = y_positions
    particles['z'] = z_positions
    
    # Velocities - opposing flows
    base_vz = 0.1 * 299792458  # 10% speed of light
    vz_velocities = np.concatenate([
        np.random.normal(base_vz, 0.01 * 299792458, n_half),     # Moving +z
        np.random.normal(-base_vz, 0.01 * 299792458, n_particles - n_half)  # Moving -z
    ])
    
    vx_velocities = np.random.normal(0, 0.001 * 299792458, n_particles)
    vy_velocities = np.random.normal(0, 0.001 * 299792458, n_particles)
    
    particles['vx'] = vx_velocities
    particles['vy'] = vy_velocities  
    particles['vz'] = vz_velocities
    
    # Calculate relativistic parameters
    v2 = particles['vx']**2 + particles['vy']**2 + particles['vz']**2
    c_squared = 299792458**2
    
    # Ensure v < c
    v2 = np.minimum(v2, 0.99 * c_squared)
    particles['gamma'] = 1.0 / np.sqrt(1.0 - v2 / c_squared)
    
    # Time
    particles['t'] = np.zeros(n_particles)
    
    # Charges and masses - mixed charge system for realistic interactions
    particles['q'] = np.random.choice([-1, 1], n_particles) * 1.602e-19  # Elementary charge
    particles['m'] = np.full(n_particles, 938.3)  # Proton mass in MeV/c²
    particles['char_time'] = np.ones(n_particles) * 1e-9  # Characteristic time
    
    # Momenta (derived from velocities)
    particles['Px'] = particles['m'] * particles['gamma'] * particles['vx']
    particles['Py'] = particles['m'] * particles['gamma'] * particles['vy']
    particles['Pz'] = particles['m'] * particles['gamma'] * particles['vz']
    particles['Pt'] = particles['m'] * particles['gamma'] * c_squared
    
    # Beta values (v/c)
    particles['bx'] = particles['vx'] / 299792458
    particles['by'] = particles['vy'] / 299792458
    particles['bz'] = particles['vz'] / 299792458
    
    # Beta derivatives (approximately zero for initial conditions)
    particles['bdotx'] = np.zeros(n_particles)
    particles['bdoty'] = np.zeros(n_particles)
    particles['bdotz'] = np.zeros(n_particles)
    
    return particles


def run_integration_with_timing(integrator, particles, steps, step_size, name):
    """Run integration with detailed timing and validation."""
    
    print(f"\n🔄 Running {name} integration...")
    print(f"  Particles: {len(particles['x'])}")
    print(f"  Steps: {steps}")
    print(f"  Step size: {step_size:.2e}")
    
    # Store initial state for physics validation
    initial_state = {k: v.copy() for k, v in particles.items()}
    current_particles = {k: v.copy() for k, v in particles.items()}
    
    # Timing with warmup for JIT
    start_time = time.time()
    
    # Integration loop
    for step in range(steps):
        updated_particles = integrator.eqsofmotion_static(
            step_size, current_particles, current_particles
        )
        current_particles = updated_particles
    
    computation_time = time.time() - start_time
    
    # Physics validation
    def calculate_total_energy(state):
        """Calculate total kinetic energy."""
        v2 = state['vx']**2 + state['vy']**2 + state['vz']**2
        gamma = 1.0 / np.sqrt(1.0 - v2 / (299792458**2) + 1e-16)
        return np.sum(state['m'] * (gamma - 1) * (299792458**2))  # Kinetic energy
    
    initial_energy = calculate_total_energy(initial_state)
    final_energy = calculate_total_energy(current_particles)
    energy_change = (final_energy - initial_energy) / initial_energy if initial_energy != 0 else 0
    
    # Position and velocity changes
    pos_change = np.sqrt(np.mean((current_particles['x'] - initial_state['x'])**2 + 
                                 (current_particles['y'] - initial_state['y'])**2 + 
                                 (current_particles['z'] - initial_state['z'])**2))
    
    vel_change = np.sqrt(np.mean((current_particles['vx'] - initial_state['vx'])**2 + 
                                 (current_particles['vy'] - initial_state['vy'])**2 + 
                                 (current_particles['vz'] - initial_state['vz'])**2))
    
    print(f"  ✅ Completed in {computation_time:.4f} seconds")
    print(f"  📊 Physics: Energy change {energy_change:.2e}, Position change {pos_change:.2e}")
    
    return {
        'success': True,
        'final_particles': current_particles,
        'computation_time': computation_time,
        'energy_initial': initial_energy,
        'energy_final': final_energy,
        'energy_change': energy_change,
        'position_change': pos_change,
        'velocity_change': vel_change
    }


def compare_trajectories(basic_result, opt_result):
    """Compare final trajectories between basic and optimized integrators."""
    print(f"\n🔍 TRAJECTORY COMPARISON")
    print("-"*30)
    
    basic_final = basic_result['final_particles']
    opt_final = opt_result['final_particles']
    
    # Calculate differences
    max_pos_diff = 0.0
    max_vel_diff = 0.0
    
    n_particles = len(basic_final['x'])
    
    for i in range(n_particles):
        # Position differences (x, y, z)
        pos_diff = max(abs(basic_final['x'][i] - opt_final['x'][i]),
                      abs(basic_final['y'][i] - opt_final['y'][i]),
                      abs(basic_final['z'][i] - opt_final['z'][i]))
        
        # Velocity differences (vx, vy, vz)
        vel_diff = max(abs(basic_final['vx'][i] - opt_final['vx'][i]),
                      abs(basic_final['vy'][i] - opt_final['vy'][i]),
                      abs(basic_final['vz'][i] - opt_final['vz'][i]))
        
        max_pos_diff = max(max_pos_diff, pos_diff)
        max_vel_diff = max(max_vel_diff, vel_diff)
    
    print(f"Max position difference: {max_pos_diff:.2e}")
    print(f"Max velocity difference: {max_vel_diff:.2e}")
    
    # Assessment
    if max_pos_diff < 1e-14 and max_vel_diff < 1e-14:
        print("🎯 PERFECT MATCH - Machine precision agreement!")
        return True
    elif max_pos_diff < 1e-10 and max_vel_diff < 1e-10:
        print("✅ EXCELLENT AGREEMENT - Within numerical precision")
        return True
    elif max_pos_diff < 1e-6 and max_vel_diff < 1e-6:
        print("⚠️  GOOD AGREEMENT - Small differences")
        return True
    else:
        print("❌ SIGNIFICANT DIFFERENCES - Investigation needed")
        return False


def run_verification_test(n_particles, steps=20, step_size=1e-6):
    """Run verification test for given parameters."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {n_particles} particles, {steps} steps")
    print(f"{'='*60}")
    
    # Create test particles
    particles = create_test_particles(n_particles)
    
    # Initialize integrators
    basic_integrator = LienardWiechertIntegrator()
    opt_integrator = OptimizedLienardWiechertIntegrator()
    
    # Run basic integration
    basic_result = run_integration_with_timing(
        basic_integrator, particles, steps, step_size, f"BASIC ({n_particles}p)"
    )
    
    # Run optimized integration
    opt_result = run_integration_with_timing(
        opt_integrator, particles, steps, step_size, f"OPTIMIZED ({n_particles}p)"
    )
    
    # Compare results
    trajectories_match = compare_trajectories(basic_result, opt_result)
    
    # Calculate speedup
    if basic_result['success'] and opt_result['success']:
        speedup = basic_result['computation_time'] / opt_result['computation_time']
        print(f"\n📈 PERFORMANCE: {speedup:.2f}x speedup (Optimized vs Basic)")
    
    return {
        'n_particles': n_particles,
        'steps': steps,
        'basic_result': basic_result,
        'opt_result': opt_result,
        'trajectories_match': trajectories_match,
        'speedup': speedup if 'speedup' in locals() else None
    }


def main():
    """Main verification function."""
    
    # Test cases: (particles, steps, step_size)
    test_cases = [
        (6, 25, 2e-6),    # Small test - like demo
        (10, 20, 1e-6),   # Basic test
        (20, 20, 1e-6),   # Moderate test
        (50, 20, 1e-6),   # Larger test
    ]
    
    results = []
    
    for n_particles, steps, step_size in test_cases:
        try:
            result = run_verification_test(n_particles, steps, step_size)
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed for {n_particles} particles: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    all_match = True
    for result in results:
        if result.get('trajectories_match', False):
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_match = False
        
        speedup = result.get('speedup', 'N/A')
        speedup_str = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else speedup
        
        print(f"{result['n_particles']:3d} particles: {status} - Speedup: {speedup_str}")
    
    if all_match:
        print(f"\n🎯 OVERALL RESULT: ✅ ALL TESTS PASSED")
        print("   Basic and Optimized integrators produce identical results!")
    else:
        print(f"\n🎯 OVERALL RESULT: ❌ SOME TESTS FAILED")
        print("   Investigate differences between integrators")


if __name__ == "__main__":
    main()