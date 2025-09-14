#!/usr/bin/env python3
"""
Performance scaling test for LW integrator with varying particle counts.

This test measures execution time and memory usage as the number of particles
increases to identify performance bottlenecks and scaling behavior.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile
import gc

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lw_integrator import LienardWiechertIntegrator
from lw_integrator.core import ParticleEnsemble
from lw_integrator.physics.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE_ESU


def create_test_particles(n_particles: int) -> ParticleEnsemble:
    """Create test particle ensemble with realistic parameters."""
    particles = ParticleEnsemble(n_particles)
    
    # Set up electron beam parameters
    particles.mass[:] = ELECTRON_MASS_AMU
    particles.charge[:] = ELEMENTARY_CHARGE_ESU
    
    # Create spread-out initial positions (avoid overlaps)
    if n_particles == 1:
        particles.positions[0] = [0.0, 0.0, 0.0]
    else:
        # Arrange in grid pattern for multiple particles
        grid_size = int(np.ceil(np.sqrt(n_particles)))
        spacing = 0.1  # mm spacing between particles
        
        for i in range(n_particles):
            row = i // grid_size
            col = i % grid_size
            particles.positions[i] = [
                col * spacing - (grid_size-1) * spacing / 2,
                row * spacing - (grid_size-1) * spacing / 2,
                0.0
            ]
    
    # Initial velocities - slightly relativistic
    particles.velocities[:, 2] = 30.0  # mm/ns = 0.1c
    particles.update_gamma()
    
    return particles


def time_integration_step(integrator, particles, dt=1e-6, n_steps=10):
    """Time a series of integration steps."""
    start_time = time.perf_counter()
    
    for step in range(n_steps):
        # Simple electric field for testing
        def test_field(x, y, z, t):
            return np.array([1e5, 0.0, 0.0])  # Constant E-field
        
        # Single integration step
        result = integrator.integrate_step(
            particles, dt, test_field, 
            simulation_type="free_particle"
        )
        
        # Update particle states
        particles.positions += particles.velocities * dt
        particles.update_gamma()
    
    end_time = time.perf_counter()
    return (end_time - start_time) / n_steps  # Average time per step


def memory_usage_test():
    """Test memory usage for particle creation."""
    import psutil
    import os
    
    def get_memory_mb():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    particle_counts = [10, 100, 500, 1000, 2000, 5000]
    memory_usage = []
    
    print("Testing memory usage...")
    
    for n in particle_counts:
        gc.collect()  # Clean up before measurement
        
        mem_before = get_memory_mb()
        particles = create_test_particles(n)
        mem_after = get_memory_mb()
        
        memory_per_particle = (mem_after - mem_before) / n * 1024  # KB per particle
        memory_usage.append(memory_per_particle)
        
        print(f"  {n:4d} particles: {memory_per_particle:.2f} KB/particle "
              f"({mem_after - mem_before:.1f} MB total)")
        
        del particles  # Clean up
    
    return particle_counts, memory_usage


def performance_scaling_test():
    """Test performance scaling with particle count."""
    print("="*60)
    print("LW INTEGRATOR PERFORMANCE SCALING TEST")
    print("="*60)
    
    # Test different particle counts
    particle_counts = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    
    # Test different integrator implementations
    integrator_types = [
        ("Trajectory", "trajectory"),
        ("Optimized", "optimized"),
        ("Unified", "unified")
    ]
    
    results = {}
    
    for impl_name, impl_type in integrator_types:
        print(f"\nTesting {impl_name} implementation...")
        
        try:
            if impl_type == "trajectory":
                from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator as TestIntegrator
            elif impl_type == "optimized":
                from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator as TestIntegrator
            else:  # unified
                TestIntegrator = LienardWiechertIntegrator
            
            integrator = TestIntegrator()
            
        except ImportError as e:
            print(f"  ⚠️  {impl_name} implementation not available: {e}")
            continue
        
        times = []
        
        for n_particles in particle_counts:
            print(f"  Testing {n_particles:3d} particles...", end=" ")
            
            try:
                # Create particles
                particles = create_test_particles(n_particles)
                
                # Time the integration
                avg_time = time_integration_step(integrator, particles, n_steps=5)
                times.append(avg_time)
                
                print(f"{avg_time*1000:.3f} ms/step")
                
            except Exception as e:
                print(f"ERROR: {e}")
                times.append(float('nan'))
        
        results[impl_name] = (particle_counts, times)
    
    return results


def detailed_timing_breakdown():
    """Detailed timing breakdown for specific operations."""
    print(f"\n" + "="*50)
    print("DETAILED TIMING BREAKDOWN")
    print("="*50)
    
    n_particles = 100
    particles = create_test_particles(n_particles)
    integrator = LienardWiechertIntegrator()
    dt = 1e-6
    
    # Test individual operations
    operations = {
        "Particle creation": lambda: create_test_particles(n_particles),
        "Gamma update": lambda: particles.update_gamma(),
        "Position update": lambda: setattr(particles, 'positions', 
                                         particles.positions + particles.velocities * dt),
        "Memory copy": lambda: particles.positions.copy(),
    }
    
    for op_name, operation in operations.items():
        # Time the operation multiple times
        times = []
        for _ in range(10):
            start = time.perf_counter()
            operation()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"{op_name:<20}: {avg_time*1000:.3f} ± {std_time*1000:.3f} ms")


def plot_scaling_results(results):
    """Generate plots of scaling results."""
    print(f"\n📊 Generating performance plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Time vs particles
    for impl_name, (counts, times) in results.items():
        # Remove NaN values
        valid_idx = ~np.isnan(times)
        valid_counts = np.array(counts)[valid_idx]
        valid_times = np.array(times)[valid_idx]
        
        if len(valid_times) > 0:
            ax1.loglog(valid_counts, np.array(valid_times) * 1000, 
                      'o-', label=impl_name, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Number of Particles')
    ax1.set_ylabel('Time per Step (ms)')
    ax1.set_title('Integration Time Scaling')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add theoretical scaling lines
    if len(results) > 0:
        # Get reference data for scaling lines
        ref_counts, ref_times = next(iter(results.values()))
        valid_idx = ~np.isnan(ref_times)
        if np.any(valid_idx):
            ref_count = np.array(ref_counts)[valid_idx][0]
            ref_time = np.array(ref_times)[valid_idx][0]
            
            x_theory = np.array([1, max(ref_counts)])
            
            # O(N) scaling
            y_linear = ref_time * 1000 * (x_theory / ref_count)
            ax1.loglog(x_theory, y_linear, '--', alpha=0.5, color='gray', label='O(N)')
            
            # O(N²) scaling  
            y_quadratic = ref_time * 1000 * (x_theory / ref_count)**2
            ax1.loglog(x_theory, y_quadratic, ':', alpha=0.5, color='red', label='O(N²)')
    
    ax1.legend()
    
    # Plot 2: Performance efficiency
    for impl_name, (counts, times) in results.items():
        valid_idx = ~np.isnan(times)
        valid_counts = np.array(counts)[valid_idx]
        valid_times = np.array(times)[valid_idx]
        
        if len(valid_times) > 1:
            # Calculate particles per second
            particles_per_sec = valid_counts / valid_times
            ax2.semilogx(valid_counts, particles_per_sec, 
                        'o-', label=impl_name, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Particles')
    ax2.set_ylabel('Particles Processed per Second')
    ax2.set_title('Processing Efficiency')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('lw_integrator_performance_scaling.png', dpi=150, bbox_inches='tight')
    print("📊 Plot saved: lw_integrator_performance_scaling.png")
    
    try:
        plt.show()
    except:
        pass  # Handle non-interactive environments


def scaling_analysis():
    """Analyze scaling behavior."""
    print(f"\n" + "="*50)
    print("SCALING ANALYSIS")
    print("="*50)
    
    # Test larger particle counts for scaling analysis
    large_counts = [100, 200, 500, 1000]
    
    print("Testing scaling with larger particle counts...")
    
    integrator = LienardWiechertIntegrator()
    times = []
    
    for n in large_counts:
        particles = create_test_particles(n)
        avg_time = time_integration_step(integrator, particles, n_steps=3)
        times.append(avg_time)
        
        print(f"  {n:4d} particles: {avg_time*1000:.2f} ms/step "
              f"({avg_time*1000/n:.4f} ms/particle)")
    
    # Analyze scaling
    if len(times) >= 2:
        # Fit to power law: time = a * N^b
        log_counts = np.log(large_counts)
        log_times = np.log(times)
        
        # Linear fit in log space
        coeffs = np.polyfit(log_counts, log_times, 1)
        scaling_exponent = coeffs[0]
        
        print(f"\nScaling analysis:")
        print(f"  Scaling exponent: {scaling_exponent:.2f}")
        
        if scaling_exponent < 1.2:
            print(f"  ✅ Nearly linear scaling (good!)")
        elif scaling_exponent < 1.8:
            print(f"  ⚠️  Between linear and quadratic")
        else:
            print(f"  ❌ Quadratic or worse scaling")
        
        # Predict time for larger systems
        for pred_n in [2000, 5000, 10000]:
            pred_time = np.exp(coeffs[1]) * pred_n**scaling_exponent
            print(f"  Predicted {pred_n:5d} particles: {pred_time*1000:.1f} ms/step")


if __name__ == "__main__":
    try:
        # Run performance tests
        results = performance_scaling_test()
        
        # Detailed timing
        detailed_timing_breakdown()
        
        # Memory usage test
        try:
            memory_counts, memory_usage = memory_usage_test()
        except ImportError:
            print("\n⚠️  psutil not available for memory testing")
        
        # Scaling analysis
        scaling_analysis()
        
        # Generate plots
        if results:
            plot_scaling_results(results)
        
        print(f"\n✅ Performance testing complete!")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()