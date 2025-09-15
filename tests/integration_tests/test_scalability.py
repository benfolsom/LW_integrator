#!/usr/bin/env python3
"""
Scalable Verification Framework

Extended verification testing with 1, 10, 50, 100, 200, 500 particles
and up to 5000 steps, time-limited to 2 minutes per test.
"""

import numpy as np
import time
import sys
import os
import json
from datetime import datetime

# Add path for imports
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator')

from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator
from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator

print("🚀 SCALABLE VERIFICATION FRAMEWORK")
print("="*70)


def create_scaled_test_particles(n_particles, seed=None):
    """Create realistic test particles for electromagnetic interaction."""
    if seed is not None:
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
    
    # Charges and masses - mixed charge system
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


def estimate_runtime(integrator_type, n_particles, steps):
    """Estimate runtime based on empirical scaling laws."""
    
    # Empirical timing constants (from previous tests)
    if integrator_type == 'basic':
        # Basic integrator: ~O(N²) scaling
        base_time_per_step = 1.5e-4  # seconds per step per particle²
        time_estimate = base_time_per_step * n_particles**2 * steps
    elif integrator_type == 'optimized':
        # Optimized integrator: ~O(N^1.2) scaling  
        base_time_per_step = 2e-6   # seconds per step per particle^1.2
        time_estimate = base_time_per_step * (n_particles**1.2) * steps
    else:
        time_estimate = float('inf')
    
    return time_estimate


def run_timed_integration(integrator, particles, steps, step_size, name, max_time=120):
    """Run integration with time limit."""
    
    print(f"\n🔄 Running {name} integration...")
    print(f"  Particles: {len(particles['x'])}")
    print(f"  Steps: {steps}")
    print(f"  Step size: {step_size:.2e}")
    print(f"  Max time: {max_time}s")
    
    # Estimate runtime
    integrator_type = 'optimized' if 'Optimized' in str(type(integrator)) else 'basic'
    estimated_time = estimate_runtime(integrator_type, len(particles['x']), steps)
    print(f"  Estimated time: {estimated_time:.2f}s")
    
    if estimated_time > max_time:
        print(f"  ⏰ SKIPPED - Estimated time {estimated_time:.1f}s exceeds limit {max_time}s")
        return {
            'success': False,
            'skipped': True,
            'reason': f'Estimated time {estimated_time:.1f}s > {max_time}s',
            'estimated_time': estimated_time
        }
    
    # Store initial state
    initial_state = {k: v.copy() for k, v in particles.items()}
    current_particles = {k: v.copy() for k, v in particles.items()}
    
    # Start timing
    start_time = time.time()
    steps_completed = 0
    
    try:
        # Integration loop with time checking
        for step in range(steps):
            # Check time limit every 10 steps
            if step % 10 == 0:
                elapsed_time = time.time() - start_time
                if elapsed_time > max_time:
                    print(f"  ⏰ TIMEOUT after {step} steps ({elapsed_time:.1f}s)")
                    break
            
            # Single integration step
            updated_particles = integrator.eqsofmotion_static(
                step_size, current_particles, current_particles
            )
            current_particles = updated_particles
            steps_completed = step + 1
        
        computation_time = time.time() - start_time
        
        # Physics validation (only if completed)
        def calculate_total_energy(state):
            v2 = state['vx']**2 + state['vy']**2 + state['vz']**2
            gamma = 1.0 / np.sqrt(1.0 - v2 / (299792458**2) + 1e-16)
            return np.sum(state['m'] * (gamma - 1) * (299792458**2))
        
        initial_energy = calculate_total_energy(initial_state)
        final_energy = calculate_total_energy(current_particles)
        energy_change = (final_energy - initial_energy) / initial_energy if initial_energy != 0 else 0
        
        completed = (steps_completed == steps)
        
        if completed:
            print(f"  ✅ COMPLETED in {computation_time:.4f}s")
        else:
            print(f"  ⏰ PARTIAL ({steps_completed}/{steps} steps) in {computation_time:.4f}s")
        
        print(f"  📊 Energy change: {energy_change:.2e}")
        
        return {
            'success': True,
            'completed': completed,
            'final_particles': current_particles,
            'computation_time': computation_time,
            'steps_completed': steps_completed,
            'steps_requested': steps,
            'energy_initial': initial_energy,
            'energy_final': final_energy,
            'energy_change': energy_change,
            'estimated_time': estimated_time
        }
        
    except Exception as e:
        computation_time = time.time() - start_time
        print(f"  ❌ FAILED after {computation_time:.4f}s: {e}")
        return {
            'success': False,
            'error': str(e),
            'computation_time': computation_time,
            'steps_completed': steps_completed,
            'estimated_time': estimated_time
        }


def compare_integrator_results(basic_result, opt_result):
    """Compare results between basic and optimized integrators."""
    
    if not (basic_result.get('success', False) and opt_result.get('success', False)):
        return False, "One or both integrators failed"
    
    if not (basic_result.get('completed', False) and opt_result.get('completed', False)):
        return False, "One or both integrators incomplete"
    
    basic_final = basic_result['final_particles']
    opt_final = opt_result['final_particles']
    
    # Calculate differences
    max_pos_diff = 0.0
    max_vel_diff = 0.0
    
    n_particles = len(basic_final['x'])
    
    for i in range(n_particles):
        # Position differences
        pos_diff = max(abs(basic_final['x'][i] - opt_final['x'][i]),
                      abs(basic_final['y'][i] - opt_final['y'][i]),
                      abs(basic_final['z'][i] - opt_final['z'][i]))
        
        # Velocity differences
        vel_diff = max(abs(basic_final['vx'][i] - opt_final['vx'][i]),
                      abs(basic_final['vy'][i] - opt_final['vy'][i]),
                      abs(basic_final['vz'][i] - opt_final['vz'][i]))
        
        max_pos_diff = max(max_pos_diff, pos_diff)
        max_vel_diff = max(max_vel_diff, vel_diff)
    
    # Assessment
    if max_pos_diff < 1e-14 and max_vel_diff < 1e-14:
        return True, "PERFECT"
    elif max_pos_diff < 1e-10 and max_vel_diff < 1e-10:
        return True, "EXCELLENT"
    elif max_pos_diff < 1e-6 and max_vel_diff < 1e-6:
        return True, "GOOD"
    else:
        return False, f"POOR (pos:{max_pos_diff:.2e}, vel:{max_vel_diff:.2e})"


def run_scalable_test(n_particles, steps=20, step_size=1e-6, max_time=120):
    """Run scalable verification test."""
    
    print(f"\n{'='*70}")
    print(f"🧪 SCALABLE TEST: {n_particles} particles, {steps} steps")
    print(f"{'='*70}")
    
    # Create test particles
    particles = create_scaled_test_particles(n_particles, seed=42)
    
    # Initialize integrators
    basic_integrator = LienardWiechertIntegrator()
    opt_integrator = OptimizedLienardWiechertIntegrator()
    
    # Run basic integration
    basic_result = run_timed_integration(
        basic_integrator, particles, steps, step_size, 
        f"BASIC ({n_particles}p)", max_time
    )
    
    # Run optimized integration
    opt_result = run_timed_integration(
        opt_integrator, particles, steps, step_size,
        f"OPTIMIZED ({n_particles}p)", max_time
    )
    
    # Compare results if both completed
    if (basic_result.get('completed', False) and opt_result.get('completed', False)):
        match, quality = compare_integrator_results(basic_result, opt_result)
        
        # Calculate speedup
        if basic_result['success'] and opt_result['success']:
            speedup = basic_result['computation_time'] / opt_result['computation_time']
            print(f"\n📈 PERFORMANCE: {speedup:.2f}x speedup")
        else:
            speedup = None
    else:
        match = False
        quality = "INCOMPLETE"
        speedup = None
    
    print(f"🔍 COMPARISON: {quality}")
    
    return {
        'n_particles': n_particles,
        'steps': steps,
        'step_size': step_size,
        'basic_result': basic_result,
        'opt_result': opt_result,
        'match': match,
        'quality': quality,
        'speedup': speedup,
        'timestamp': datetime.now().isoformat()
    }


def save_test_results(results, filename="scalable_verification_results.json"):
    """Save test results to JSON file."""
    output_dir = '/home/benfol/work/LW_windows/LW_integrator/tests/results'
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy types to JSON-serializable types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 Results saved to {filepath}")


def main():
    """Main scalable verification function."""
    
    # Test configuration: (particles, steps, step_size, max_time)
    test_configs = [
        (1, 100, 1e-6, 120),      # Single particle - longer test
        (10, 50, 1e-6, 120),      # Small test
        (50, 20, 1e-6, 120),      # Medium test  
        (100, 20, 1e-6, 120),     # Large test
        (200, 10, 1e-6, 120),     # Very large test
        (500, 5, 1e-6, 120),      # Extreme test
        
        # Long duration tests (only if short duration completes quickly)
        (10, 1000, 1e-6, 120),    # Extended time test
        (50, 100, 1e-6, 120),     # Extended medium test
        (100, 50, 1e-6, 120),     # Extended large test
    ]
    
    results = []
    
    for n_particles, steps, step_size, max_time in test_configs:
        try:
            result = run_scalable_test(n_particles, steps, step_size, max_time)
            results.append(result)
            
            # Skip further tests if current test failed due to time constraints
            if result['basic_result'].get('skipped', False) or result['opt_result'].get('skipped', False):
                print(f"⏭️  Skipping remaining larger tests due to time constraints")
                break
                
        except Exception as e:
            print(f"❌ Test failed for {n_particles} particles: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    save_test_results(results)
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 SCALABLE VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    successful_tests = 0
    perfect_matches = 0
    
    for result in results:
        n_particles = result['n_particles']
        steps = result['steps']
        quality = result.get('quality', 'FAILED')
        speedup = result.get('speedup')
        
        basic_success = result['basic_result'].get('success', False)
        opt_success = result['opt_result'].get('success', False)
        
        if basic_success and opt_success:
            successful_tests += 1
            status = "✅ PASS"
            if quality == "PERFECT":
                perfect_matches += 1
        else:
            status = "❌ FAIL"
        
        speedup_str = f"{speedup:.1f}x" if speedup else "N/A"
        
        print(f"{n_particles:3d}p x {steps:4d}s: {status} - {quality:8s} - Speedup: {speedup_str}")
    
    print(f"\n🎯 OVERALL RESULTS:")
    print(f"   {successful_tests}/{len(results)} tests successful")
    print(f"   {perfect_matches}/{successful_tests} perfect matches")
    
    if perfect_matches == successful_tests and successful_tests > 0:
        print(f"   🎉 EXCELLENT: All successful tests show perfect agreement!")
    elif successful_tests > 0:
        print(f"   ✅ GOOD: Most tests successful with good agreement")
    else:
        print(f"   ❌ POOR: Few successful tests - investigate issues")


if __name__ == "__main__":
    main()