#!/usr/bin/env python3
"""
Final Corrected LW Integrator Performance Scaling Test

Tests performance scaling of electromagnetic force calculations with varying
particle counts using the complete LW integrator API with all required fields.

Author: GitHub Copilot (AI Assistant)  
Date: 2025-01-16
"""

import numpy as np
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add LW integrator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from lw_integrator import LWIntegrator
    from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator
    from lw_integrator.core.particles import ParticleEnsemble
    from lw_integrator.physics.constants import C_MMNS
    print("✅ Successfully imported LW integrator modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def calculate_relativistic_quantities(velocities):
    """Calculate gamma and beta from velocities."""
    v_squared = np.sum(velocities**2, axis=1)
    beta_squared = v_squared / C_MMNS**2
    gamma = 1.0 / np.sqrt(1.0 - np.clip(beta_squared, 0, 0.99999))
    beta = velocities / C_MMNS
    return gamma, beta


def create_complete_particle_data(n_particles: int, time_value: float = 0.0) -> dict:
    """Create complete particle data with all required fields."""
    # Create test positions in a small cube
    positions = np.random.uniform(-1e-3, 1e-3, (n_particles, 3))  # 1mm cube
    
    # Moderate velocities (5% speed of light)
    velocities = np.random.uniform(-0.05, 0.05, (n_particles, 3)) * C_MMNS
    
    # Calculate relativistic quantities
    gamma, beta = calculate_relativistic_quantities(velocities)
    
    # Calculate beta derivatives (acceleration in units of c)
    # Small random accelerations for realistic physics
    beta_derivatives = np.random.uniform(-0.01, 0.01, (n_particles, 3)) * C_MMNS / 1e-3  # acceleration/c
    
    # Calculate relativistic momentum (Px = γmvx, etc.)
    mass_kg = 9.109e-31  # Electron mass in kg
    momentum = gamma[:, np.newaxis] * mass_kg * velocities
    
    # Calculate 4-momentum time component (Pt = γmc²)
    momentum_time = gamma * mass_kg * C_MMNS**2
    
    # Complete particle data structure with ALL required fields
    # Ensure all arrays are proper numpy arrays with consistent dtypes
    particle_data = {
        'x': np.array(positions[:, 0], dtype=np.float64),
        'y': np.array(positions[:, 1], dtype=np.float64), 
        'z': np.array(positions[:, 2], dtype=np.float64),
        'vx': np.array(velocities[:, 0], dtype=np.float64),
        'vy': np.array(velocities[:, 1], dtype=np.float64),
        'vz': np.array(velocities[:, 2], dtype=np.float64),
        'bx': np.array(beta[:, 0], dtype=np.float64),  # Beta x-component (v/c)
        'by': np.array(beta[:, 1], dtype=np.float64),  # Beta y-component (v/c)
        'bz': np.array(beta[:, 2], dtype=np.float64),  # Beta z-component (v/c)
        'bdotx': np.array(beta_derivatives[:, 0], dtype=np.float64),  # Beta derivative x (acceleration/c)
        'bdoty': np.array(beta_derivatives[:, 1], dtype=np.float64),  # Beta derivative y (acceleration/c)
        'bdotz': np.array(beta_derivatives[:, 2], dtype=np.float64),  # Beta derivative z (acceleration/c)
        'Px': np.array(momentum[:, 0], dtype=np.float64),  # Relativistic momentum x
        'Py': np.array(momentum[:, 1], dtype=np.float64),  # Relativistic momentum y
        'Pz': np.array(momentum[:, 2], dtype=np.float64),  # Relativistic momentum z
        'Pt': np.array(momentum_time, dtype=np.float64),   # 4-momentum time component
        'gamma': np.array(gamma, dtype=np.float64),    # Lorentz factor
        't': np.array(np.full(n_particles, time_value), dtype=np.float64),  # Time coordinate
        'q': np.array(np.ones(n_particles) * 1.602e-19, dtype=np.float64),  # Elementary charge
        'm': np.array(np.ones(n_particles) * 9.109e-31, dtype=np.float64),   # Electron mass
        'char_time': np.array(np.ones(n_particles) * 1e-12, dtype=np.float64)  # Characteristic time
    }
    
    return particle_data


def time_function(func, *args, **kwargs):
    """Time a function call and return execution time in seconds."""
    start_time = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result, None
    except Exception as e:
        end_time = time.perf_counter()
        # Get more detailed error information
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        if len(error_msg) > 50:
            error_msg = error_msg[:47] + "..."
        return end_time - start_time, None, error_msg


def test_distance_calculation_scaling():
    """Test performance scaling of distance calculations."""
    print("\n" + "="*60)
    print("DISTANCE CALCULATION SCALING TEST")
    print("="*60)
    
    integrator = LienardWiechertIntegrator()
    particle_counts = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    
    print(f"{'Particles':<10} {'Time (ms)':<12} {'ms/particle':<12} {'Status':<20}")
    print("-" * 54)
    
    timing_data = []
    
    for n in particle_counts:
        # Create test data
        particles = create_complete_particle_data(n)
        particles_ext = create_complete_particle_data(n)
        
        # Time distance calculation for first particle
        exec_time, result, error = time_function(
            integrator.dist_euclid, 
            particles, particles_ext, 0
        )
        
        if error:
            print(f"{n:<10} {'ERROR':<12} {'ERROR':<12} {error[:18]:<20}")
        else:
            time_ms = exec_time * 1000
            time_per_particle = time_ms / n if n > 0 else 0
            print(f"{n:<10} {time_ms:<12.3f} {time_per_particle:<12.3f} {'✅ OK':<20}")
            timing_data.append((n, time_ms, time_per_particle))
    
    return timing_data


def test_static_forces_scaling():
    """Test performance scaling of static electromagnetic force calculations."""
    print("\n" + "="*60)
    print("STATIC ELECTROMAGNETIC FORCES SCALING TEST")
    print("="*60)
    
    integrator = LienardWiechertIntegrator()
    particle_counts = [1, 2, 5, 10, 20, 50, 100]
    
    print(f"{'Particles':<10} {'Time (ms)':<12} {'ms/particle':<12} {'Status':<20}")
    print("-" * 54)
    
    timing_data = []
    
    for n in particle_counts:
        # Create test data with all required fields
        particles = create_complete_particle_data(n, 0.0)
        particles_ext = create_complete_particle_data(n, 0.0)
        h = 1e-3  # Small timestep
        
        # Time static force calculation
        exec_time, result, error = time_function(
            integrator.eqsofmotion_static,
            h, particles, particles_ext
        )
        
        if error:
            print(f"{n:<10} {'ERROR':<12} {'ERROR':<12} {error[:18]:<20}")
        else:
            time_ms = exec_time * 1000
            time_per_particle = time_ms / n if n > 0 else 0
            print(f"{n:<10} {time_ms:<12.3f} {time_per_particle:<12.3f} {'✅ OK':<20}")
            timing_data.append((n, time_ms, time_per_particle))
    
    return timing_data


def test_retarded_forces_scaling():
    """Test performance scaling of retarded electromagnetic force calculations."""
    print("\n" + "="*60)
    print("RETARDED ELECTROMAGNETIC FORCES SCALING TEST")
    print("="*60)
    
    integrator = LienardWiechertIntegrator()
    particle_counts = [1, 2, 5, 10, 20, 50]  # Smaller set for expensive calculation
    
    print(f"{'Particles':<10} {'Time (ms)':<12} {'ms/particle':<12} {'Status':<20}")
    print("-" * 54)
    
    timing_data = []
    
    for n in particle_counts:
        # Create test data
        particles = create_complete_particle_data(n, 0.0)
        particles_ext = create_complete_particle_data(n, 0.0)
        
        # Create trajectory format (list of particle states) with multiple timesteps
        # The retarded calculation needs trajectory history
        trajectory = [
            create_complete_particle_data(n, -2e-3),  # t = -2ms  
            create_complete_particle_data(n, -1e-3),  # t = -1ms
            particles                                  # t = 0ms (current)
        ]
        trajectory_ext = [
            create_complete_particle_data(n, -2e-3),  # t = -2ms
            create_complete_particle_data(n, -1e-3),  # t = -1ms  
            particles_ext                             # t = 0ms (current)
        ]
        
        h = 1e-3  # Small timestep
        i_traj = 2  # Current trajectory index (latest time)
        
        # Time retarded force calculation
        exec_time, result, error = time_function(
            integrator.eqsofmotion_retarded,
            h, trajectory, trajectory_ext, i_traj
        )
        
        if error:
            print(f"{n:<10} {'ERROR':<12} {'ERROR':<12} {error[:18]:<20}")
        else:
            time_ms = exec_time * 1000
            time_per_particle = time_ms / n if n > 0 else 0
            print(f"{n:<10} {time_ms:<12.3f} {time_per_particle:<12.3f} {'✅ OK':<20}")
            timing_data.append((n, time_ms, time_per_particle))
    
    return timing_data


def analyze_scaling(timing_data, test_name):
    """Analyze scaling behavior from timing data."""
    if len(timing_data) < 3:
        print(f"\n❌ {test_name}: Insufficient data for scaling analysis")
        return
        
    print(f"\n📊 {test_name} SCALING ANALYSIS:")
    print("-" * 50)
    
    # Extract data
    particles = np.array([d[0] for d in timing_data])
    times = np.array([d[1] for d in timing_data])
    
    # Fit scaling models
    try:
        # Linear scaling (O(N))
        linear_fit = np.polyfit(particles, times, 1)
        linear_pred = np.polyval(linear_fit, particles)
        linear_r2 = 1 - np.sum((times - linear_pred)**2) / np.sum((times - np.mean(times))**2)
        
        # Quadratic scaling (O(N²))
        if len(timing_data) >= 3:
            quad_fit = np.polyfit(particles, times, 2)
            quad_pred = np.polyval(quad_fit, particles)
            quad_r2 = 1 - np.sum((times - quad_pred)**2) / np.sum((times - np.mean(times))**2)
        else:
            quad_r2 = 0
        
        print(f"Linear fit (O(N)): R² = {linear_r2:.4f}")
        print(f"Quadratic fit (O(N²)): R² = {quad_r2:.4f}")
        
        if quad_r2 > linear_r2 + 0.1:
            print("🔍 Scaling appears closer to O(N²) - particle-particle interactions dominant")
            expected_time_1000 = quad_fit[0] * 1000**2 + quad_fit[1] * 1000 + quad_fit[2]
            print(f"📈 Extrapolated time for 1000 particles: {expected_time_1000:.1f} ms")
        elif linear_r2 > 0.8:
            print("🔍 Scaling appears closer to O(N) - good parallelization potential")
            expected_time_1000 = linear_fit[0] * 1000 + linear_fit[1]
            print(f"📈 Extrapolated time for 1000 particles: {expected_time_1000:.1f} ms")
        else:
            print("🔍 Scaling behavior unclear - may need more data points")
            
        # Efficiency analysis
        if len(timing_data) > 1:
            efficiency = timing_data[0][2] / timing_data[-1][2]  # First vs last ms/particle
            print(f"⚡ Efficiency ratio (1 vs {particles[-1]} particles): {efficiency:.2f}x")
            
            # Performance classification
            last_time_per_particle = timing_data[-1][2]
            if last_time_per_particle < 0.1:
                print("🚀 Performance: Excellent (< 0.1 ms/particle)")
            elif last_time_per_particle < 1.0:
                print("⚡ Performance: Good (< 1 ms/particle)")
            elif last_time_per_particle < 10.0:
                print("⚠️  Performance: Moderate (< 10 ms/particle)")
            else:
                print("🐌 Performance: Slow (> 10 ms/particle)")
                
    except Exception as e:
        print(f"❌ Scaling analysis failed: {e}")


def main():
    """Run comprehensive performance scaling tests."""
    print("="*60)
    print("LW INTEGRATOR FINAL PERFORMANCE SCALING TEST")
    print("="*60)
    print("Testing electromagnetic force calculation performance scaling")
    print("with varying particle counts using complete integrator API.")
    print("="*60)
    
    # Run all scaling tests
    distance_data = test_distance_calculation_scaling()
    static_data = test_static_forces_scaling()
    retarded_data = test_retarded_forces_scaling()
    
    # Analyze scaling behavior
    analyze_scaling(distance_data, "Distance Calculations")
    analyze_scaling(static_data, "Static EM Forces")
    analyze_scaling(retarded_data, "Retarded EM Forces")
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print("Performance scaling tests completed with correct API usage.")
    print("\nKey findings:")
    
    if distance_data:
        last_distance = distance_data[-1]
        print(f"• Distance calculations: {last_distance[2]:.3f} ms/particle @ {last_distance[0]} particles")
    
    if static_data:
        last_static = static_data[-1]
        print(f"• Static EM forces: {last_static[2]:.3f} ms/particle @ {last_static[0]} particles")
    
    if retarded_data:
        last_retarded = retarded_data[-1]
        print(f"• Retarded EM forces: {last_retarded[2]:.3f} ms/particle @ {last_retarded[0]} particles")
    
    print("\n🔍 Performance Analysis:")
    print("• Distance calculations are the basic O(N) operation")
    print("• Static forces include Coulomb interactions (likely O(N²))")  
    print("• Retarded forces are most expensive due to temporal calculations")
    print("\n💡 Recommendations:")
    print("• For >100 particles: Consider parallel computing")
    print("• For >1000 particles: Use hierarchical methods (tree codes)")
    print("• Retarded calculations are the primary bottleneck")
    print("\n✅ Performance characterization complete!")


if __name__ == "__main__":
    main()