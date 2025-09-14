#!/usr/bin/env python3
"""
Simplified performance scaling test for LW integrator.

This test measures how the electromagnetic force calculation scales
with particle count, which is the main computational bottleneck.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lw_integrator import LienardWiechertIntegrator
from lw_integrator.core import ParticleEnsemble
from lw_integrator.physics.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE_ESU


def create_test_particles(n_particles: int) -> dict:
    """Create test particle data in the format expected by integrator."""
    
    # Create particles in grid formation to avoid overlaps
    grid_size = int(np.ceil(np.sqrt(n_particles)))
    spacing = 0.1  # mm spacing
    
    x = np.zeros(n_particles)
    y = np.zeros(n_particles)
    z = np.zeros(n_particles)
    
    for i in range(n_particles):
        row = i // grid_size
        col = i % grid_size
        x[i] = col * spacing - (grid_size-1) * spacing / 2
        y[i] = row * spacing - (grid_size-1) * spacing / 2
        z[i] = 0.0
    
    # Create velocity arrays - slightly relativistic electrons
    vx = np.zeros(n_particles)
    vy = np.zeros(n_particles) 
    vz = np.full(n_particles, 10.0)  # mm/ns = ~0.033c
    
    # Convert to beta (v/c)
    bx = vx / C_MMNS
    by = vy / C_MMNS
    bz = vz / C_MMNS
    
    # Calculate gamma factors
    beta_sq = bx**2 + by**2 + bz**2
    gamma = 1.0 / np.sqrt(1 - beta_sq)
    
    # Particle properties
    mass = np.full(n_particles, ELECTRON_MASS_AMU)
    charge = np.full(n_particles, ELEMENTARY_CHARGE_ESU)
    
    return {
        'x': x, 'y': y, 'z': z,
        'bx': bx, 'by': by, 'bz': bz,
        'gamma': gamma,
        'm': mass,
        'q': charge,
        'char_time': np.full(n_particles, 2/3 * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS_AMU * C_MMNS**3))
    }


def time_force_calculation(integrator, particles, n_trials=5):
    """Time the electromagnetic force calculation."""
    
    times = []
    
    for trial in range(n_trials):
        start_time = time.perf_counter()
        
        # Call the main computational method
        try:
            # Test retarded force calculation (main bottleneck)
            forces = integrator.eqsofmotion_retarded(
                h=1e-6,  # time step
                vector=particles,
                vector_ext=particles,
                apt_R=np.inf,  # no aperture
                sim_type=1
            )
            
        except Exception as e:
            print(f"Error in force calculation: {e}")
            return float('nan')
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Return median time (more robust than mean)
    return np.median(times)


def simple_performance_test():
    """Test performance scaling with particle count."""
    print("="*60)
    print("LW INTEGRATOR SIMPLE PERFORMANCE TEST")
    print("="*60)
    
    # Test particle counts
    particle_counts = [1, 2, 5, 10, 20, 50, 100]
    
    integrator = LienardWiechertIntegrator()
    print(f"Using integrator: {integrator.implementation_type}")
    
    results = {
        'counts': [],
        'times': [],
        'time_per_particle': [],
        'particles_per_second': []
    }
    
    print(f"\nTesting electromagnetic force calculation scaling:")
    print(f"{'Particles':<10} {'Time (ms)':<12} {'ms/particle':<12} {'particles/s':<12}")
    print("-" * 50)
    
    for n_particles in particle_counts:
        try:
            # Create test particles
            particles = create_test_particles(n_particles)
            
            # Time the calculation
            calc_time = time_force_calculation(integrator, particles)
            
            if not np.isnan(calc_time):
                time_per_particle = calc_time / n_particles
                particles_per_sec = n_particles / calc_time
                
                results['counts'].append(n_particles)
                results['times'].append(calc_time)
                results['time_per_particle'].append(time_per_particle)
                results['particles_per_second'].append(particles_per_sec)
                
                print(f"{n_particles:<10} {calc_time*1000:<12.3f} {time_per_particle*1000:<12.4f} {particles_per_sec:<12.1f}")
            else:
                print(f"{n_particles:<10} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
                
        except Exception as e:
            print(f"{n_particles:<10} ERROR: {e}")
    
    return results


def analyze_scaling(results):
    """Analyze scaling behavior."""
    if len(results['counts']) < 2:
        print("\nNot enough data for scaling analysis")
        return
    
    print(f"\n" + "="*50)
    print("SCALING ANALYSIS")
    print("="*50)
    
    counts = np.array(results['counts'])
    times = np.array(results['times'])
    
    # Fit to power law: time = a * N^b
    if len(counts) >= 3:
        log_counts = np.log(counts)
        log_times = np.log(times)
        
        # Linear fit in log space
        coeffs = np.polyfit(log_counts, log_times, 1)
        scaling_exponent = coeffs[0]
        
        print(f"Scaling analysis:")
        print(f"  Scaling exponent: {scaling_exponent:.2f}")
        
        if scaling_exponent < 1.2:
            print(f"  ✅ Nearly linear scaling (excellent)")
        elif scaling_exponent < 1.8:
            print(f"  ⚠️  Between linear and quadratic")
        else:
            print(f"  ❌ Quadratic or worse scaling")
        
        # Theoretical complexity analysis
        if scaling_exponent < 1.2:
            complexity = "O(N)"
        elif scaling_exponent < 1.8:
            complexity = "O(N^1.5)"
        else:
            complexity = "O(N²)"
            
        print(f"  Estimated complexity: {complexity}")
        
        # Predict performance for larger systems
        print(f"\nExtrapolated performance:")
        for pred_n in [200, 500, 1000]:
            if pred_n > max(counts):
                pred_time = np.exp(coeffs[1]) * pred_n**scaling_exponent
                print(f"  {pred_n:4d} particles: ~{pred_time*1000:.1f} ms per calculation")
    
    # Efficiency analysis
    efficiency = np.array(results['particles_per_second'])
    if len(efficiency) > 1:
        efficiency_change = (efficiency[-1] - efficiency[0]) / efficiency[0] * 100
        print(f"\nEfficiency analysis:")
        print(f"  Initial efficiency: {efficiency[0]:.1f} particles/second")
        print(f"  Final efficiency: {efficiency[-1]:.1f} particles/second")
        print(f"  Efficiency change: {efficiency_change:+.1f}%")
        
        if efficiency_change > -20:
            print(f"  ✅ Good efficiency retention")
        elif efficiency_change > -50:
            print(f"  ⚠️  Moderate efficiency loss")
        else:
            print(f"  ❌ Poor scaling efficiency")


def plot_results(results):
    """Generate performance plots."""
    if len(results['counts']) < 2:
        print("Not enough data for plotting")
        return
    
    print(f"\n📊 Generating performance plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    counts = np.array(results['counts'])
    times = np.array(results['times'])
    
    # Plot 1: Total time vs particles
    ax1.loglog(counts, times * 1000, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Particles')
    ax1.set_ylabel('Calculation Time (ms)')
    ax1.set_title('Total Calculation Time')
    ax1.grid(True, alpha=0.3)
    
    # Add theoretical scaling lines
    if len(counts) > 2:
        x_theory = np.array([counts[0], counts[-1]])
        
        # O(N) scaling
        y_linear = times[0] * 1000 * (x_theory / counts[0])
        ax1.loglog(x_theory, y_linear, '--', alpha=0.7, color='green', label='O(N)')
        
        # O(N²) scaling
        y_quadratic = times[0] * 1000 * (x_theory / counts[0])**2
        ax1.loglog(x_theory, y_quadratic, ':', alpha=0.7, color='red', label='O(N²)')
        
        ax1.legend()
    
    # Plot 2: Time per particle
    ax2.semilogx(counts, np.array(results['time_per_particle']) * 1000, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Particles')
    ax2.set_ylabel('Time per Particle (ms)')
    ax2.set_title('Per-Particle Calculation Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Particles per second
    ax3.semilogx(counts, results['particles_per_second'], 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Particles')
    ax3.set_ylabel('Particles per Second')
    ax3.set_title('Processing Efficiency')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Memory estimation
    # Estimate memory usage (rough approximation)
    memory_per_particle = 8 * 10  # ~10 double values per particle
    memory_mb = counts * memory_per_particle / (1024**2)
    
    ax4.loglog(counts, memory_mb, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Particles')
    ax4.set_ylabel('Estimated Memory (MB)')
    ax4.set_title('Memory Usage Estimate')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lw_performance_simple.png', dpi=150, bbox_inches='tight')
    print("📊 Plot saved: lw_performance_simple.png")
    
    try:
        plt.show()
    except:
        pass


def test_specific_operations():
    """Test timing of specific operations."""
    print(f"\n" + "="*50)
    print("OPERATION-SPECIFIC TIMING")
    print("="*50)
    
    n_particles = 50  # Moderate size for detailed analysis
    particles = create_test_particles(n_particles)
    integrator = LienardWiechertIntegrator()
    
    operations = {
        "Particle creation": lambda: create_test_particles(n_particles),
        "Distance calculation": lambda: integrator.dist_euclid(particles, particles),
        "Static EM forces": lambda: integrator.eqsofmotion_static(1e-6, particles, particles, sim_type=1),
    }
    
    print(f"Testing with {n_particles} particles:")
    
    for op_name, operation in operations.items():
        times = []
        for _ in range(5):
            try:
                start = time.perf_counter()
                result = operation()
                end = time.perf_counter()
                times.append(end - start)
            except Exception as e:
                print(f"{op_name:<25}: ERROR - {e}")
                break
        
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            print(f"{op_name:<25}: {avg_time*1000:.3f} ± {std_time*1000:.3f} ms")


if __name__ == "__main__":
    try:
        # Run simple performance test
        results = simple_performance_test()
        
        # Analyze scaling
        analyze_scaling(results)
        
        # Test specific operations
        test_specific_operations()
        
        # Generate plots if we have data
        if results['counts']:
            plot_results(results)
        
        print(f"\n✅ Performance testing complete!")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()