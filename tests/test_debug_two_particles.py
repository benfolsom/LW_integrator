#!/usr/bin/env python3
"""
Focused debugging test for LW integrator with 2 particles
"""

import numpy as np
import sys
import os

# Add LW integrator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator
from lw_integrator.physics.constants import C_MMNS

def calculate_relativistic_quantities(velocities):
    """Calculate gamma and beta from velocities."""
    v_squared = np.sum(velocities**2, axis=1)
    beta_squared = v_squared / C_MMNS**2
    gamma = 1.0 / np.sqrt(1.0 - np.clip(beta_squared, 0, 0.99999))
    beta = velocities / C_MMNS
    return gamma, beta

def create_simple_particle_data(n_particles: int, time_value: float = 0.0) -> dict:
    """Create minimal particle data."""
    np.random.seed(42)  # Reproducible results
    
    # Simple test case
    positions = np.random.uniform(-1e-3, 1e-3, (n_particles, 3))
    velocities = np.random.uniform(-0.01, 0.01, (n_particles, 3)) * C_MMNS
    
    gamma, beta = calculate_relativistic_quantities(velocities)
    beta_derivatives = np.random.uniform(-0.001, 0.001, (n_particles, 3)) * C_MMNS / 1e-3
    
    mass_kg = 9.109e-31
    momentum = gamma[:, np.newaxis] * mass_kg * velocities
    momentum_time = gamma * mass_kg * C_MMNS**2
    
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

def test_static_forces():
    """Test static forces with 2 particles."""
    print("Testing static forces with 2 particles...")
    
    integrator = LienardWiechertIntegrator()
    
    particles = create_simple_particle_data(2, 0.0)
    particles_ext = create_simple_particle_data(2, 0.0)
    
    print("Particle data created successfully")
    print(f"Particle arrays shapes:")
    for key, val in particles.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.shape} {val.dtype}")
    
    try:
        result = integrator.eqsofmotion_static(1e-3, particles, particles_ext)
        print("✅ Static forces calculation succeeded!")
        print(f"Result keys: {list(result.keys())}")
        return True
    except Exception as e:
        print(f"❌ Static forces failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Focused LW integrator debugging test")
    print("=" * 50)
    
    success = test_static_forces()
    
    if success:
        print("\n✅ Test passed! Ready for full scaling test")
    else:
        print("\n❌ Test failed - need to debug further")

if __name__ == "__main__":
    main()