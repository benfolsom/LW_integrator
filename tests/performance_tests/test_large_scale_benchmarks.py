#!/usr/bin/env python3
"""
Large-Scale Performance Testing for LW Integrator

CAI: Test performance scaling up to practical limits to identify bottlenecks
and determine requirements for parallelization strategies.
"""

import numpy as np
import time
import sys
import traceback
from typing import Dict, List, Any, Tuple

# Add parent directory to path for imports
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator')

# Import integrator implementations
try:
    from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator as BasicLWIntegrator
    from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator
    from lw_integrator.core.unified_interface import LienardWiechertIntegrator as UnifiedLWIntegrator
    print("✅ Successfully imported all integrator implementations")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def create_test_particles(n_particles: int, spatial_scale: float = 1e-3) -> Dict[str, np.ndarray]:
    """
    Create test particle data for performance testing.
    
    Args:
        n_particles: Number of particles to create
        spatial_scale: Spatial scale in mm
        
    Returns:
        Dictionary with particle data arrays
    """
    # Create particles in a roughly spherical distribution
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
    
    # Convert to beta (v/c)
    bx = vx
    by = vy 
    bz = vz
    
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


def measure_integration_performance(integrator, particles: Dict, timestep: float = 1e-6) -> Tuple[float, bool]:
    """
    Measure single integration step performance.
    
    Args:
        integrator: Integrator instance
        particles: Particle data
        timestep: Integration timestep
        
    Returns:
        (computation_time, success)
    """
    try:
        start_time = time.time()
        result = integrator.eqsofmotion_static(timestep, particles, particles)
        computation_time = time.time() - start_time
        
        # Basic validation
        if not isinstance(result, dict) or 'Px' not in result:
            return float('inf'), False
            
        return computation_time, True
        
    except Exception as e:
        print(f"    ❌ Integration failed: {e}")
        return float('inf'), False


def test_memory_usage():
    """
    Estimate memory usage for different particle counts.
    """
    print("\n📊 MEMORY USAGE ESTIMATION")
    print("="*60)
    
    particle_counts = [50, 100, 200, 500, 1000, 2000, 5000]
    
    for n in particle_counts:
        # Calculate memory for particle data
        # Each particle has ~20 float64 values = 20 * 8 = 160 bytes
        particle_memory = n * 160  # bytes
        
        # Force calculations: O(N²) distance and force storage
        force_memory = n * n * 8 * 4  # 4 force components per interaction
        
        total_memory_mb = (particle_memory + force_memory) / (1024 * 1024)
        
        print(f"  {n:4d} particles: {total_memory_mb:6.1f} MB "
              f"(particles: {particle_memory/1024:.1f} KB, forces: {force_memory/(1024*1024):.1f} MB)")


def run_large_scale_performance_test():
    """
    Run comprehensive large-scale performance testing.
    """
    print("🚀 LARGE-SCALE PERFORMANCE TESTING")
    print("="*80)
    print("Testing electromagnetic force calculations up to practical limits")
    print("="*80)
    
    # Test particle counts - start conservative and work up
    particle_counts = [50, 100, 150, 200, 300, 500, 750, 1000]
    max_time_limit = 30.0  # seconds - stop if single step takes longer
    
    # Test each integrator type
    integrator_configs = [
        ("Basic Trajectory", BasicLWIntegrator()),
        ("Unified Auto-Optimized", UnifiedLWIntegrator()),
        ("Direct Optimized", OptimizedLienardWiechertIntegrator())
    ]
    
    results = {}
    
    for integrator_name, integrator in integrator_configs:
        print(f"\n{'='*60}")
        print(f"TESTING {integrator_name.upper()}")
        print(f"{'='*60}")
        
        results[integrator_name] = {
            'particle_counts': [],
            'computation_times': [],
            'particles_per_second': [],
            'interactions_per_second': [],
            'max_feasible_particles': 0
        }
        
        for n_particles in particle_counts:
            print(f"\n🧪 Testing {n_particles} particles...")
            
            try:
                # Create test particles
                particles = create_test_particles(n_particles)
                print(f"  ✅ Created {n_particles} test particles")
                
                # Measure performance
                computation_time, success = measure_integration_performance(integrator, particles)
                
                if not success or computation_time > max_time_limit:
                    print(f"  ⏱️  Stopping at {n_particles} particles - "
                          f"computation time {computation_time:.2f}s exceeds limit")
                    results[integrator_name]['max_feasible_particles'] = n_particles - 1
                    break
                
                # Calculate performance metrics
                particles_per_second = n_particles / computation_time
                interactions = n_particles * (n_particles - 1)  # N*(N-1) interactions
                interactions_per_second = interactions / computation_time
                
                # Store results
                results[integrator_name]['particle_counts'].append(n_particles)
                results[integrator_name]['computation_times'].append(computation_time)
                results[integrator_name]['particles_per_second'].append(particles_per_second)
                results[integrator_name]['interactions_per_second'].append(interactions_per_second)
                
                print(f"  ⏱️  Time: {computation_time*1000:.1f} ms")
                print(f"  🔥 Particles/sec: {particles_per_second:.0f}")
                print(f"  ⚡ Interactions/sec: {interactions_per_second:.2e}")
                
                # Update max feasible
                results[integrator_name]['max_feasible_particles'] = n_particles
                
            except Exception as e:
                print(f"  ❌ Failed at {n_particles} particles: {e}")
                traceback.print_exc()
                break
    
    return results


def analyze_scaling_performance(results: Dict):
    """
    Analyze and report scaling performance results.
    """
    print(f"\n{'='*80}")
    print("LARGE-SCALE PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Performance summary table
    print(f"\n📊 Maximum Feasible Particle Counts:")
    print("="*50)
    for integrator_name, data in results.items():
        max_particles = data['max_feasible_particles']
        if data['computation_times']:
            max_time = max(data['computation_times']) * 1000
            print(f"  {integrator_name:25s}: {max_particles:4d} particles ({max_time:.1f} ms)")
        else:
            print(f"  {integrator_name:25s}: Failed immediately")
    
    # Detailed performance table
    print(f"\n📊 Performance Comparison Table:")
    print("="*80)
    print(f"{'Particles':>9} | {'Basic (ms)':>10} | {'Unified (ms)':>12} | {'Optimized (ms)':>14} | {'Best Speedup':>12}")
    print("-" * 80)
    
    # Find common particle counts
    all_counts = set()
    for data in results.values():
        all_counts.update(data['particle_counts'])
    common_counts = sorted(all_counts)
    
    for n in common_counts:
        row = f"{n:9d} |"
        times = {}
        
        for integrator_name, data in results.items():
            if n in data['particle_counts']:
                idx = data['particle_counts'].index(n)
                time_ms = data['computation_times'][idx] * 1000
                times[integrator_name] = time_ms
                row += f" {time_ms:9.1f} |"
            else:
                row += f" {'---':>9s} |"
        
        # Calculate best speedup
        if len(times) > 1:
            min_time = min(times.values())
            max_time = max(times.values())
            speedup = max_time / min_time
            row += f" {speedup:10.1f}x"
        
        print(row)
    
    # Scaling analysis
    print(f"\n🔍 Scaling Analysis:")
    print("="*50)
    
    for integrator_name, data in results.items():
        if len(data['particle_counts']) >= 3:
            counts = np.array(data['particle_counts'])
            times = np.array(data['computation_times'])
            
            # Fit to N² scaling
            log_n = np.log(counts)
            log_t = np.log(times)
            
            # Linear fit in log space: log(t) = a*log(N) + b
            coeffs = np.polyfit(log_n, log_t, 1)
            scaling_exponent = coeffs[0]
            r_squared = np.corrcoef(log_n, log_t)[0, 1]**2
            
            # Predict time for 10,000 particles
            predicted_10k = np.exp(coeffs[1]) * (10000 ** scaling_exponent)
            
            print(f"  {integrator_name}:")
            print(f"    Scaling: O(N^{scaling_exponent:.2f}), R² = {r_squared:.4f}")
            print(f"    Predicted 10k particles: {predicted_10k:.1f} seconds")
    
    # Memory and parallelization insights
    print(f"\n💡 Parallelization Insights:")
    print("="*50)
    
    # Find where each integrator becomes impractical
    for integrator_name, data in results.items():
        max_particles = data['max_feasible_particles']
        if max_particles > 0:
            total_interactions = max_particles * (max_particles - 1)
            print(f"  {integrator_name}:")
            print(f"    Max practical: {max_particles} particles")
            print(f"    Total interactions: {total_interactions:,}")
            
            if max_particles < 500:
                print(f"    🔴 Limited scalability - needs algorithmic improvements")
            elif max_particles < 1000:
                print(f"    🟡 Moderate scalability - parallelization beneficial")
            else:
                print(f"    🟢 Good scalability - parallelization for larger systems")


def main():
    """
    Run the comprehensive large-scale performance testing.
    """
    print("⚡ LW INTEGRATOR LARGE-SCALE PERFORMANCE TESTING")
    print("="*80)
    print("Testing performance scaling up to practical computational limits")
    print("="*80)
    
    # Memory usage estimation
    test_memory_usage()
    
    # Large-scale performance testing
    results = run_large_scale_performance_test()
    
    # Analysis and recommendations
    analyze_scaling_performance(results)
    
    print(f"\n{'='*80}")
    print("🎯 LARGE-SCALE TESTING COMPLETE")
    print("="*80)
    print("Ready for parallelization strategy design based on observed limits!")


if __name__ == "__main__":
    main()