#!/usr/bin/env python3
"""
Corrected LW Integrator Performance Scaling Test

Tests performance scaling of electromagnetic force calculations with varying
particle counts using the actual LW integrator API.

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
    print("✅ Successfully imported LW integrator modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def create_test_particles(n_particles: int) -> dict:
    """Create test particle ensemble with realistic parameters."""
    # Create test positions in a small cube
    positions = np.random.uniform(-1e-3, 1e-3, (n_particles, 3))  # 1mm cube
    
    # Moderate velocities (10% speed of light)
    velocities = np.random.uniform(-0.1, 0.1, (n_particles, 3)) * 299.792458  # mm/ns
    
    # Particle data structure for trajectory integrator
    particle_data = {
        'x': positions[:, 0],
        'y': positions[:, 1], 
        'z': positions[:, 2],
        'vx': velocities[:, 0],
        'vy': velocities[:, 1],
        'vz': velocities[:, 2],
        'q': np.ones(n_particles) * 1.602e-19,  # Elementary charge
        'm': np.ones(n_particles) * 9.109e-31,   # Electron mass
        'char_time': np.ones(n_particles) * 1e-12  # Characteristic time
    }
    
    return particle_data


def create_trajectory_data(particle_data: dict) -> list:
    """Create trajectory format needed by integrator methods."""
    # The integrator expects a list of trajectory states
    return [particle_data]


def time_function(func, *args, **kwargs):
    """Time a function call and return execution time in seconds."""
    start_time = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time, result, None
    except Exception as e:
        end_time = time.perf_counter()
        return end_time - start_time, None, str(e)


def test_distance_calculation_scaling():
    """Test performance scaling of distance calculations."""
    print("\n" + "="*60)
    print("DISTANCE CALCULATION SCALING TEST")
    print("="*60)
    
    integrator = LienardWiechertIntegrator()
    particle_counts = [1, 2, 5, 10, 20, 50, 100, 200]
    
    print(f"{'Particles':<10} {'Time (ms)':<12} {'ms/particle':<12} {'Error':<30}")
    print("-" * 64)
    
    timing_data = []
    
    for n in particle_counts:
        # Create test data
        particles = create_test_particles(n)
        particles_ext = create_test_particles(n)  # External particles
        
        # Time distance calculation for first particle
        exec_time, result, error = time_function(
            integrator.dist_euclid, 
            particles, particles_ext, 0
        )
        
        if error:
            print(f"{n:<10} {'ERROR':<12} {'ERROR':<12} {error[:28]:<30}")
        else:
            time_ms = exec_time * 1000
            time_per_particle = time_ms / n if n > 0 else 0
            print(f"{n:<10} {time_ms:<12.3f} {time_per_particle:<12.3f} {'OK':<30}")
            timing_data.append((n, time_ms, time_per_particle))
    
    return timing_data


def test_static_forces_scaling():
    """Test performance scaling of static electromagnetic force calculations."""
    print("\n" + "="*60)
    print("STATIC ELECTROMAGNETIC FORCES SCALING TEST")
    print("="*60)
    
    integrator = LienardWiechertIntegrator()
    particle_counts = [1, 2, 5, 10, 20, 50, 100]
    
    print(f"{'Particles':<10} {'Time (ms)':<12} {'ms/particle':<12} {'Error':<30}")
    print("-" * 64)
    
    timing_data = []
    
    for n in particle_counts:
        # Create test data
        particles = create_test_particles(n)
        particles_ext = create_test_particles(n)
        h = 1e-3  # Small timestep
        
        # Time static force calculation
        exec_time, result, error = time_function(
            integrator.eqsofmotion_static,
            h, particles, particles_ext
        )
        
        if error:
            print(f"{n:<10} {'ERROR':<12} {'ERROR':<12} {error[:28]:<30}")
        else:
            time_ms = exec_time * 1000
            time_per_particle = time_ms / n if n > 0 else 0
            print(f"{n:<10} {time_ms:<12.3f} {time_per_particle:<12.3f} {'OK':<30}")
            timing_data.append((n, time_ms, time_per_particle))
    
    return timing_data


def test_retarded_forces_scaling():
    """Test performance scaling of retarded electromagnetic force calculations."""
    print("\n" + "="*60)
    print("RETARDED ELECTROMAGNETIC FORCES SCALING TEST")
    print("="*60)
    
    integrator = LienardWiechertIntegrator()
    particle_counts = [1, 2, 5, 10, 20]  # Smaller set for more expensive calculation
    
    print(f"{'Particles':<10} {'Time (ms)':<12} {'ms/particle':<12} {'Error':<30}")
    print("-" * 64)
    
    timing_data = []
    
    for n in particle_counts:
        # Create test data
        particles = create_test_particles(n)
        particles_ext = create_test_particles(n)
        
        # Create trajectory format
        trajectory = create_trajectory_data(particles)
        trajectory_ext = create_trajectory_data(particles_ext)
        
        h = 1e-3  # Small timestep
        i_traj = 0  # Current trajectory index
        
        # Time retarded force calculation
        exec_time, result, error = time_function(
            integrator.eqsofmotion_retarded,
            h, trajectory, trajectory_ext, i_traj
        )
        
        if error:
            print(f"{n:<10} {'ERROR':<12} {'ERROR':<12} {error[:28]:<30}")
        else:
            time_ms = exec_time * 1000
            time_per_particle = time_ms / n if n > 0 else 0
            print(f"{n:<10} {time_ms:<12.3f} {time_per_particle:<12.3f} {'OK':<30}")
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
        quad_fit = np.polyfit(particles, times, 2)
        quad_pred = np.polyval(quad_fit, particles)
        quad_r2 = 1 - np.sum((times - quad_pred)**2) / np.sum((times - np.mean(times))**2)
        
        print(f"Linear fit (O(N)): R² = {linear_r2:.4f}")
        print(f"Quadratic fit (O(N²)): R² = {quad_r2:.4f}")
        
        if quad_r2 > linear_r2 + 0.1:
            print("🔍 Scaling appears closer to O(N²) - particle-particle interactions")
        elif linear_r2 > 0.8:
            print("🔍 Scaling appears closer to O(N) - good parallelization potential")
        else:
            print("🔍 Scaling behavior unclear - may need more data points")
            
        # Efficiency analysis
        if len(timing_data) > 1:
            efficiency = timing_data[0][2] / timing_data[-1][2]  # First vs last ms/particle
            print(f"Efficiency ratio (1 particle vs {particles[-1]}): {efficiency:.2f}")
            
    except Exception as e:
        print(f"❌ Scaling analysis failed: {e}")


def main():
    """Run comprehensive performance scaling tests."""
    print("="*60)
    print("LW INTEGRATOR CORRECTED PERFORMANCE SCALING TEST")
    print("="*60)
    print("Testing electromagnetic force calculation performance scaling")
    print("with varying particle counts using actual integrator API")
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
    print("SUMMARY")
    print("="*60)
    print("Performance scaling tests completed using actual LW integrator API.")
    print("Key findings:")
    print("• Distance calculations: Basic O(N) particle-particle distance computation")
    print("• Static EM forces: Coulomb interactions without retardation effects")
    print("• Retarded EM forces: Full Lienard-Wiechert field calculations")
    print("\nFor large particle ensembles, retarded calculations will be most expensive.")
    print("Consider parallelization or hierarchical methods for >100 particles.")
    print("✅ Performance testing complete!")


if __name__ == "__main__":
    main()