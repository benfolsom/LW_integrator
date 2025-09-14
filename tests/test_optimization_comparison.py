#!/usr/bin/env python3
"""
Comprehensive Performance Comparison: Optimized vs Unoptimized Implementations

Tests the performance difference between:
1. Basic trajectory_integrator (unoptimized)
2. Unified interface with Numba optimizations
3. Direct optimized implementation

This addresses your question about current performance optimization usage.

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
    # Basic unoptimized implementation
    from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator as BasicIntegrator
    
    # Unified interface (auto-optimized)
    from lw_integrator import LWIntegrator as UnifiedIntegrator
    
    # Direct optimized implementation
    from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator as DirectOptimized
    
    from lw_integrator.physics.constants import C_MMNS
    print("✅ Successfully imported all implementations")
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
    np.random.seed(42)  # Reproducible results
    
    # Create test positions in a small cube
    positions = np.random.uniform(-1e-3, 1e-3, (n_particles, 3))
    
    # Moderate velocities (5% speed of light)
    velocities = np.random.uniform(-0.05, 0.05, (n_particles, 3)) * C_MMNS
    
    # Calculate relativistic quantities
    gamma, beta = calculate_relativistic_quantities(velocities)
    
    # Calculate beta derivatives
    beta_derivatives = np.random.uniform(-0.01, 0.01, (n_particles, 3)) * C_MMNS / 1e-3
    
    # Calculate relativistic momentum
    mass_kg = 9.109e-31
    momentum = gamma[:, np.newaxis] * mass_kg * velocities
    momentum_time = gamma * mass_kg * C_MMNS**2
    
    # Complete particle data structure
    particle_data = {
        'x': np.array(positions[:, 0], dtype=np.float64),
        'y': np.array(positions[:, 1], dtype=np.float64), 
        'z': np.array(positions[:, 2], dtype=np.float64),
        'vx': np.array(velocities[:, 0], dtype=np.float64),
        'vy': np.array(velocities[:, 1], dtype=np.float64),
        'vz': np.array(velocities[:, 2], dtype=np.float64),
        'bx': np.array(beta[:, 0], dtype=np.float64),
        'by': np.array(beta[:, 1], dtype=np.float64),
        'bz': np.array(beta[:, 2], dtype=np.float64),
        'bdotx': np.array(beta_derivatives[:, 0], dtype=np.float64),
        'bdoty': np.array(beta_derivatives[:, 1], dtype=np.float64),
        'bdotz': np.array(beta_derivatives[:, 2], dtype=np.float64),
        'Px': np.array(momentum[:, 0], dtype=np.float64),
        'Py': np.array(momentum[:, 1], dtype=np.float64),
        'Pz': np.array(momentum[:, 2], dtype=np.float64),
        'Pt': np.array(momentum_time, dtype=np.float64),
        'gamma': np.array(gamma, dtype=np.float64),
        't': np.array(np.full(n_particles, time_value), dtype=np.float64),
        'q': np.array(np.ones(n_particles) * 1.602e-19, dtype=np.float64),
        'm': np.array(np.ones(n_particles) * 9.109e-31, dtype=np.float64),
        'char_time': np.array(np.ones(n_particles) * 1e-12, dtype=np.float64)
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
        error_msg = f"{type(e).__name__}: {str(e)}"
        if len(error_msg) > 40:
            error_msg = error_msg[:37] + "..."
        return end_time - start_time, None, error_msg


def test_implementation_performance(implementation_class, name, particle_counts):
    """Test performance of a specific implementation."""
    print(f"\n{'='*60}")
    print(f"TESTING {name.upper()} IMPLEMENTATION")
    print(f"{'='*60}")
    
    try:
        integrator = implementation_class()
        print(f"✅ {name} integrator created successfully")
        
        # Check if it has optimization info
        if hasattr(integrator, '_implementation_type'):
            print(f"   Implementation type: {integrator._implementation_type}")
        if hasattr(integrator, 'use_optimized'):
            print(f"   Using optimized: {integrator.use_optimized}")
            
    except Exception as e:
        print(f"❌ Failed to create {name} integrator: {e}")
        return None
    
    timing_data = []
    
    print(f"\n📊 Static Forces Performance:")
    print(f"{'Particles':<10} {'Time (ms)':<12} {'ms/particle':<12} {'Status':<20}")
    print("-" * 54)
    
    for n in particle_counts:
        # Create test data
        particles = create_complete_particle_data(n, 0.0)
        particles_ext = create_complete_particle_data(n, 0.0)
        h = 1e-3
        
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


def compare_implementations():
    """Compare all available implementations."""
    print("="*80)
    print("LW INTEGRATOR IMPLEMENTATION PERFORMANCE COMPARISON")
    print("="*80)
    print("Testing optimized vs unoptimized electromagnetic force calculations")
    print("="*80)
    
    # Test parameters
    particle_counts = [1, 2, 5, 10, 20, 50]
    
    # Test all implementations
    implementations = [
        (BasicIntegrator, "Basic Trajectory"),
        (UnifiedIntegrator, "Unified Auto-Optimized"),
    ]
    
    # Try to add direct optimized if available
    try:
        implementations.append((DirectOptimized, "Direct Optimized"))
    except:
        print("⚠️  Direct optimized implementation not available")
    
    results = {}
    
    for impl_class, name in implementations:
        timing_data = test_implementation_performance(impl_class, name, particle_counts)
        if timing_data:
            results[name] = timing_data
    
    return results


def analyze_performance_differences(results):
    """Analyze performance differences between implementations."""
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    if len(results) < 2:
        print("❌ Not enough implementations to compare")
        return
    
    # Get data for comparison
    impl_names = list(results.keys())
    
    print(f"\n📊 Performance Comparison Table:")
    print(f"{'Particles':<10} ", end="")
    for name in impl_names:
        print(f"{name[:15]:<18}", end="")
    print(" Speedup")
    print("-" * (10 + 18 * len(impl_names) + 10))
    
    # Compare at different particle counts
    max_particles = min(len(results[name]) for name in impl_names)
    
    for i in range(max_particles):
        particle_count = results[impl_names[0]][i][0]
        print(f"{particle_count:<10} ", end="")
        
        times = []
        for name in impl_names:
            time_ms = results[name][i][1]
            times.append(time_ms)
            print(f"{time_ms:<18.3f}", end="")
        
        # Calculate speedup (basic vs best)
        if len(times) >= 2 and times[0] > 0:
            speedup = times[0] / min(times[1:]) if len(times) > 1 else 1.0
            print(f"{speedup:<10.2f}x")
        else:
            print("N/A")
    
    # Performance scaling analysis
    print(f"\n🔍 Scaling Analysis:")
    for name, timing_data in results.items():
        if len(timing_data) >= 3:
            particles = np.array([d[0] for d in timing_data])
            times = np.array([d[1] for d in timing_data])
            
            # Fit to quadratic
            try:
                quad_fit = np.polyfit(particles, times, 2)
                quad_pred = np.polyval(quad_fit, particles)
                quad_r2 = 1 - np.sum((times - quad_pred)**2) / np.sum((times - np.mean(times))**2)
                
                print(f"  {name}: O(N²) fit R² = {quad_r2:.4f}")
                
                # Predict 1000 particles
                pred_1000 = np.polyval(quad_fit, 1000)
                print(f"    Predicted 1000 particles: {pred_1000:.1f} ms")
                
            except:
                print(f"  {name}: Scaling analysis failed")


def main():
    """Run comprehensive implementation comparison."""
    results = compare_implementations()
    analyze_performance_differences(results)
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS ON OPTIMIZATION STATUS")
    print(f"{'='*80}")
    print("🔍 Current optimization status assessment:")
    print("• Basic trajectory integrator: Pure Python, no JIT optimization")
    print("• Unified interface: Should use Numba when available")
    print("• Performance bottleneck: O(N²) particle-particle electromagnetic forces")
    print("\n💡 For parallelization with temporal dependencies:")
    print("• Particle-level parallelization: Limited by force coupling")
    print("• Spatial domain decomposition: Better for large N")
    print("• Pipeline parallelization: Overlap communication/computation")
    print("• Consider: Tree algorithms (Fast Multipole Method) for N > 1000")
    print("\n✅ Performance comparison complete!")


if __name__ == "__main__":
    main()