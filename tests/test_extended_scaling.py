#!/usr/bin/env python3
"""
Extended Large-Scale Performance Testing for Optimized Integrator

CAI: Push the optimized integrator to its true limits since it shows exceptional performance.
"""

import numpy as np
import time
import sys

# Add parent directory to path for imports
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator')

# Import integrator implementations
try:
    from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator
    print("✅ Successfully imported optimized integrator")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def create_test_particles(n_particles: int, spatial_scale: float = 1e-3) -> dict:
    """Create test particle data for performance testing."""
    np.random.seed(42)  # Reproducible results
    
    # Random positions in sphere
    phi = np.random.uniform(0, 2*np.pi, n_particles)
    costheta = np.random.uniform(-1, 1, n_particles)
    theta = np.arccos(costheta)
    r = spatial_scale * np.random.uniform(0.1, 1.0, n_particles)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Small random velocities (non-relativistic)
    velocity_scale = 0.1  # v/c
    vx = velocity_scale * np.random.normal(0, 0.1, n_particles)
    vy = velocity_scale * np.random.normal(0, 0.1, n_particles)
    vz = velocity_scale * np.random.normal(0, 0.1, n_particles)
    
    bx, by, bz = vx, vy, vz
    
    # Calculate gamma factors
    v2 = bx**2 + by**2 + bz**2
    gamma = 1.0 / np.sqrt(1.0 - v2)
    
    # Small accelerations
    bdotx = np.random.normal(0, 0.01, n_particles)
    bdoty = np.random.normal(0, 0.01, n_particles)
    bdotz = np.random.normal(0, 0.01, n_particles)
    
    # Initialize momenta and energies
    Px = np.zeros(n_particles)
    Py = np.zeros(n_particles)
    Pz = np.zeros(n_particles)
    Pt = np.full(n_particles, 938.3)  # Proton rest energy in MeV
    
    # Charges and masses
    charges = np.random.choice([-1, 1], n_particles)  # Mixed charge
    masses = np.full(n_particles, 938.3)  # Proton mass in MeV/c²
    char_times = np.full(n_particles, 1.0)
    
    return {
        'x': x, 'y': y, 'z': z,
        'vx': vx, 'vy': vy, 'vz': vz,
        'bx': bx, 'by': by, 'bz': bz,
        'bdotx': bdotx, 'bdoty': bdoty, 'bdotz': bdotz,
        'Px': Px, 'Py': Py, 'Pz': Pz, 'Pt': Pt,
        'gamma': gamma, 't': np.zeros(n_particles),
        'q': charges, 'm': masses, 'char_time': char_times
    }


def test_optimized_scaling():
    """Test optimized integrator scaling to high particle counts."""
    print("🚀 EXTENDED OPTIMIZED INTEGRATOR PERFORMANCE TEST")
    print("="*70)
    
    # Extended particle counts - go higher since optimized shows great performance
    particle_counts = [1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000]
    max_time_limit = 60.0  # Allow 1 minute per test
    
    integrator = OptimizedLienardWiechertIntegrator()
    
    results = {
        'particle_counts': [],
        'computation_times': [],
        'memory_estimate_mb': [],
        'interactions_per_second': []
    }
    
    for n_particles in particle_counts:
        print(f"\n🧪 Testing {n_particles:,} particles...")
        
        # Memory estimate
        particle_memory = n_particles * 160  # bytes per particle
        force_memory = n_particles * n_particles * 8 * 4  # force interactions
        total_memory_mb = (particle_memory + force_memory) / (1024 * 1024)
        
        print(f"  📊 Estimated memory: {total_memory_mb:.1f} MB")
        
        if total_memory_mb > 2000:  # 2GB limit
            print(f"  ⚠️  Memory estimate exceeds 2GB limit, stopping tests")
            break
        
        try:
            # Create test particles
            particles = create_test_particles(n_particles)
            print(f"  ✅ Created {n_particles:,} test particles")
            
            # Measure performance
            start_time = time.time()
            result = integrator.eqsofmotion_static(1e-6, particles, particles)
            computation_time = time.time() - start_time
            
            if computation_time > max_time_limit:
                print(f"  ⏱️  Stopping - computation time {computation_time:.1f}s exceeds limit")
                break
            
            # Calculate performance metrics
            interactions = n_particles * (n_particles - 1)
            interactions_per_second = interactions / computation_time
            
            # Store results
            results['particle_counts'].append(n_particles)
            results['computation_times'].append(computation_time)
            results['memory_estimate_mb'].append(total_memory_mb)
            results['interactions_per_second'].append(interactions_per_second)
            
            print(f"  ⏱️  Time: {computation_time*1000:.1f} ms")
            print(f"  ⚡ Interactions/sec: {interactions_per_second:.2e}")
            print(f"  🔥 Time per interaction: {computation_time*1e9/interactions:.2f} ns")
            
        except Exception as e:
            print(f"  ❌ Failed at {n_particles:,} particles: {e}")
            break
    
    return results


def analyze_extended_results(results):
    """Analyze the extended performance results."""
    print(f"\n{'='*70}")
    print("EXTENDED PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    
    if not results['particle_counts']:
        print("❌ No successful tests to analyze")
        return
    
    # Performance table
    print(f"\n📊 Extended Performance Results:")
    print("="*70)
    print(f"{'Particles':>9} | {'Time (s)':>10} | {'Mem (MB)':>10} | {'Int/sec':>12} | {'ns/int':>8}")
    print("-" * 70)
    
    for i in range(len(results['particle_counts'])):
        n = results['particle_counts'][i]
        time_s = results['computation_times'][i]
        mem_mb = results['memory_estimate_mb'][i]
        int_per_sec = results['interactions_per_second'][i]
        ns_per_int = time_s * 1e9 / (n * (n - 1))
        
        print(f"{n:9,d} | {time_s:10.3f} | {mem_mb:10.1f} | {int_per_sec:.2e} | {ns_per_int:8.2f}")
    
    # Scaling analysis
    if len(results['particle_counts']) >= 3:
        counts = np.array(results['particle_counts'])
        times = np.array(results['computation_times'])
        
        # Fit to N² scaling
        log_n = np.log(counts)
        log_t = np.log(times)
        
        coeffs = np.polyfit(log_n, log_t, 1)
        scaling_exponent = coeffs[0]
        r_squared = np.corrcoef(log_n, log_t)[0, 1]**2
        
        print(f"\n🔍 Scaling Analysis:")
        print(f"  Scaling: O(N^{scaling_exponent:.3f}), R² = {r_squared:.4f}")
        
        # Predict larger scales
        for target_n in [50000, 100000]:
            predicted_time = np.exp(coeffs[1]) * (target_n ** scaling_exponent)
            if predicted_time < 3600:  # Less than 1 hour
                print(f"  Predicted {target_n:,} particles: {predicted_time:.1f} seconds")
            else:
                print(f"  Predicted {target_n:,} particles: {predicted_time/3600:.1f} hours")
    
    # Memory limits
    max_particles = results['particle_counts'][-1]
    max_memory = results['memory_estimate_mb'][-1]
    
    print(f"\n💾 Memory Analysis:")
    print(f"  Maximum tested: {max_particles:,} particles ({max_memory:.1f} MB)")
    
    # Estimate limits
    for memory_limit_gb in [4, 8, 16, 32]:
        memory_limit_mb = memory_limit_gb * 1024
        # Solve for N where N²*32 bytes ≈ memory_limit
        estimated_max_n = int(np.sqrt(memory_limit_mb * 1024 * 1024 / 32))
        print(f"  {memory_limit_gb:2d}GB RAM limit: ~{estimated_max_n:,} particles")


def main():
    """Run extended performance testing."""
    print("⚡ OPTIMIZED LW INTEGRATOR EXTENDED SCALING TEST")
    print("="*70)
    print("Testing optimized integrator up to practical memory/time limits")
    print("="*70)
    
    results = test_optimized_scaling()
    analyze_extended_results(results)
    
    print(f"\n{'='*70}")
    print("🎯 EXTENDED TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()