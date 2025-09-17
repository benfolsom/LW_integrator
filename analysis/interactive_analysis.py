"""
Interactive Analysis Tools for Aperture Studies

Provides fast, interactive analysis capabilities for quick parameter exploration
and visualization of particle-aperture interactions.

Key Features:
- Optimized simulation parameters for fast execution (minutes vs hours)
- Real-time parameter adjustment
- Focus on close wall interactions (<10 microns)
- Comprehensive visualization

Author: LW Integrator Development Team  
Date: 2025-09-15
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# Try to import physics constants, fallback if not available
try:
    from physics.constants import C_MMNS, ELEMENTARY_CHARGE_GAUSSIAN, ELECTRON_MASS_AMU
    CONSTANTS_AVAILABLE = True
    # Use the validated Gaussian units
    SPEED_OF_LIGHT = C_MMNS  # mm/ns
    ELEMENTARY_CHARGE = ELEMENTARY_CHARGE_GAUSSIAN  # Gaussian units
    ELECTRON_MASS = ELECTRON_MASS_AMU  # amu
except ImportError:
    # Fallback constants
    SPEED_OF_LIGHT = 2.99792458e8  # m/s
    ELEMENTARY_CHARGE = 1.602176634e-19  # C
    ELECTRON_MASS = 9.1093837015e-31  # kg
    CONSTANTS_AVAILABLE = False


@dataclass
class FastSimConfig:
    """Optimized configuration for fast interactive analysis"""
    aperture_radius: float  # meters
    n_particles: int
    beam_sigma: float  # meters - IMPORTANT: should be much smaller than aperture
    n_steps: int = 75  # Reduced from 500 for speed
    total_length: float = 2.0  # meters
    energy_mev: float = 10000.0  # 10 GeV electrons
    close_approach_threshold: float = 10e-6  # 10 microns
    save_frequency: int = 3  # Save every 3rd step
    
    def __post_init__(self):
        # Auto-adjust steps for different aperture sizes
        if self.aperture_radius < 0.5e-3:  # < 0.5mm
            self.n_steps = min(150, int(self.n_steps * 2))  # More detail for small apertures
        elif self.aperture_radius > 2e-3:  # > 2mm
            self.n_steps = max(50, int(self.n_steps * 0.7))  # Fewer steps for large apertures
        
        # Critical: Ensure beam sigma is reasonable compared to aperture
        if self.beam_sigma > self.aperture_radius * 0.3:
            print(f"Warning: Beam sigma ({self.beam_sigma*1000:.2f}mm) is large compared to aperture ({self.aperture_radius*1000:.1f}mm)")
            print(f"Recommended beam sigma: <{self.aperture_radius*0.3*1000:.2f}mm for reasonable survival rates")


class SimpleParticle:
    """Simplified particle for fast interactive analysis"""
    def __init__(self, x, y, z, vx, vy, vz, energy, particle_id):
        self.position = np.array([x, y, z])
        self.velocity = np.array([vx, vy, vz])
        self.energy = energy  # Total energy in Joules
        self.particle_id = particle_id
        self.time = 0.0
        self.is_alive = True
        self.trajectory = {
            'positions': [self.position.copy()],
            'velocities': [self.velocity.copy()],
            'energies': [self.energy],
            'times': [self.time],
            'wall_distances': []
        }
    
    def update_trajectory(self):
        """Add current state to trajectory"""
        self.trajectory['positions'].append(self.position.copy())
        self.trajectory['velocities'].append(self.velocity.copy())
        self.trajectory['energies'].append(self.energy)
        self.trajectory['times'].append(self.time)
    
    def get_kinetic_energy_mev(self):
        """Get kinetic energy in MeV"""
        gamma = self.energy / (ELECTRON_MASS * SPEED_OF_LIGHT**2)
        kinetic_joules = (gamma - 1) * ELECTRON_MASS * SPEED_OF_LIGHT**2
        return kinetic_joules / (ELEMENTARY_CHARGE * 1e6)  # Convert to MeV
    
    def get_wall_distance(self, aperture_radius):
        """Calculate distance to cylindrical wall"""
        r_transverse = np.sqrt(self.position[0]**2 + self.position[1]**2)
        return aperture_radius - r_transverse
    
    def check_collision(self, aperture_radius):
        """Check for wall collision"""
        wall_distance = self.get_wall_distance(aperture_radius)
        if wall_distance <= 0:
            self.is_alive = False
            return True
        return False


def initialize_beam(config: FastSimConfig):
    """Initialize beam of particles with proper size matching"""
    particles = []
    
    # 10 GeV electron parameters
    total_energy_joules = config.energy_mev * ELEMENTARY_CHARGE * 1e6
    gamma = total_energy_joules / (ELECTRON_MASS * SPEED_OF_LIGHT**2)
    beta = np.sqrt(1 - 1/gamma**2)
    velocity_magnitude = beta * SPEED_OF_LIGHT
    
    print(f"Initializing {config.n_particles} particles:")
    print(f"  Energy: {config.energy_mev} MeV")
    print(f"  γ: {gamma:.1f}")
    print(f"  β: {beta:.6f}")
    print(f"  Aperture: {config.aperture_radius*1000:.1f}mm")
    print(f"  Beam σ: {config.beam_sigma*1000:.3f}mm")
    print(f"  Beam σ/Aperture ratio: {config.beam_sigma/config.aperture_radius:.3f}")
    
    for i in range(config.n_particles):
        # Gaussian transverse distribution - ensure reasonable size
        x = np.random.normal(0, config.beam_sigma)
        y = np.random.normal(0, config.beam_sigma)
        z = 0.0  # Start at entrance
        
        # Calculate required divergence for this aperture/beam combination
        # Maximum transverse displacement over drift length
        max_displacement = config.aperture_radius - 3*config.beam_sigma  # 3-sigma safety margin
        drift_length = config.total_length
        max_angle = max_displacement / drift_length  # radians
        
        # Use 10% of maximum allowable divergence for safety
        angle_sigma = max_angle * 0.1
        
        vx = np.random.normal(0, velocity_magnitude * angle_sigma)
        vy = np.random.normal(0, velocity_magnitude * angle_sigma) 
        vz = velocity_magnitude
        
        particle = SimpleParticle(x, y, z, vx, vy, vz, total_energy_joules, i)
        particles.append(particle)
    
    # Check initial distribution
    initial_r = [np.sqrt(p.position[0]**2 + p.position[1]**2) for p in particles]
    inside_aperture = sum(1 for r in initial_r if r < config.aperture_radius)
    
    print(f"  Initial particles inside aperture: {inside_aperture}/{config.n_particles} ({inside_aperture/config.n_particles:.1%})")
    
    return particles


def run_fast_simulation(particles, config: FastSimConfig):
    """Run optimized simulation with proper physics"""
    print(f"Running simulation: {config.aperture_radius*1000:.1f}mm aperture, {config.n_steps} steps")
    
    # Calculate time step properly - dt should be time, not distance
    # For relativistic particles, use v ≈ c for time estimation
    total_time = config.total_length / SPEED_OF_LIGHT  # Time to traverse length at speed of light
    dt = total_time / config.n_steps
    alive_particles = [p for p in particles if p.is_alive]
    
    collision_events = []
    close_approaches = []
    
    start_time = time.time()
    
    for step in range(config.n_steps):
        if step % max(1, config.n_steps // 5) == 0:
            print(f"  Step {step:3d}/{config.n_steps}, alive: {len(alive_particles):3d}")
        
        for particle in alive_particles[:]:  # Copy list to modify during iteration
            # Simple drift dynamics (enhanced with EM fields in full version)
            particle.position += particle.velocity * dt
            particle.time += dt
            
            # Simple radiation energy loss (placeholder for realistic physics)
            if particle.get_kinetic_energy_mev() > 9000:  # Above threshold
                energy_loss_rate = 1e-8  # Very small for drift case
                particle.energy -= energy_loss_rate * ELEMENTARY_CHARGE * 1e6
            
            # Check wall distance
            wall_distance = particle.get_wall_distance(config.aperture_radius)
            particle.trajectory['wall_distances'].append(wall_distance)
            
            # Record close approaches
            if wall_distance < config.close_approach_threshold:
                close_approaches.append({
                    'particle_id': particle.particle_id,
                    'time': particle.time,
                    'position': particle.position.copy(),
                    'wall_distance': wall_distance,
                    'step': step
                })
            
            # Check for collision
            if particle.check_collision(config.aperture_radius):
                collision_events.append({
                    'particle_id': particle.particle_id,
                    'time': particle.time,
                    'position': particle.position.copy(),
                    'step': step
                })
                alive_particles.remove(particle)
            
            # Update trajectory (save every few steps for speed)
            if step % config.save_frequency == 0:
                particle.update_trajectory()
    
    # Final trajectory update
    for particle in particles:
        if particle.is_alive:
            particle.update_trajectory()
    
    simulation_time = time.time() - start_time
    
    results = {
        'config': config,
        'initial_particles': len(particles),
        'final_particles': len(alive_particles),
        'survival_rate': len(alive_particles) / len(particles),
        'collision_events': collision_events,
        'close_approaches': close_approaches,
        'simulation_time': simulation_time,
        'particles': particles
    }
    
    print(f"Simulation complete in {simulation_time:.1f}s")
    print(f"Survival rate: {results['survival_rate']:.1%} ({len(alive_particles)}/{len(particles)})")
    print(f"Close approaches (<10μm): {len(close_approaches)}")
    print(f"Wall collisions: {len(collision_events)}")
    
    return results


def quick_test(aperture_mm=1.0, n_particles=30, beam_sigma_mm=0.1, n_steps=75, energy_mev=10000):
    """Quick test function for interactive parameter exploration"""
    
    print(f"Quick Test Configuration:")
    print(f"  Aperture: {aperture_mm:.1f}mm")
    print(f"  Particles: {n_particles}")
    print(f"  Beam σ: {beam_sigma_mm:.2f}mm")
    print(f"  Steps: {n_steps}")
    print(f"  Energy: {energy_mev} MeV")
    print()
    
    # Create configuration
    config = FastSimConfig(
        aperture_radius=aperture_mm * 1e-3,
        n_particles=n_particles,
        beam_sigma=beam_sigma_mm * 1e-3,
        n_steps=n_steps,
        energy_mev=energy_mev
    )
    
    # Run simulation
    start_time = time.time()
    particles = initialize_beam(config)
    results = run_fast_simulation(particles, config)
    test_time = time.time() - start_time
    
    print(f"\\nTotal test time: {test_time:.1f}s")
    
    return results


# Predefined realistic configurations
def get_realistic_configs():
    """Get realistic test configurations with proper beam-aperture ratios and ultra-low divergence"""
    return [
        FastSimConfig(aperture_radius=5e-3, n_particles=30, beam_sigma=0.1e-3, n_steps=50),   # 5mm aperture, 0.1mm beam
        FastSimConfig(aperture_radius=2e-3, n_particles=30, beam_sigma=0.05e-3, n_steps=75),  # 2mm aperture, 0.05mm beam  
        FastSimConfig(aperture_radius=1e-3, n_particles=35, beam_sigma=0.02e-3, n_steps=100), # 1mm aperture, 0.02mm beam
        FastSimConfig(aperture_radius=0.5e-3, n_particles=30, beam_sigma=0.01e-3, n_steps=125), # 0.5mm aperture, 0.01mm beam
        FastSimConfig(aperture_radius=0.2e-3, n_particles=25, beam_sigma=0.005e-3, n_steps=150), # 0.2mm aperture, 0.005mm beam
    ]


def run_realistic_test_suite():
    """Run a realistic test suite with proper beam-aperture ratios"""
    configs = get_realistic_configs()
    
    print("=== Realistic Aperture Test Suite ===")
    print("Configurations with proper beam-aperture ratios")
    print("Expected: Some particles survive, some close approaches")
    print()
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\\nTest {i+1}/{len(configs)}:")
        print(f"Aperture: {config.aperture_radius*1000:.1f}mm, Beam σ: {config.beam_sigma*1000:.2f}mm")
        print(f"Ratio: {config.beam_sigma/config.aperture_radius:.3f}")
        
        particles = initialize_beam(config)
        result = run_fast_simulation(particles, config)
        results.append(result)
        
        print(f"Result: {result['survival_rate']:.1%} survival, {len(result['close_approaches'])} close approaches")
    
    return results


if __name__ == "__main__":
    print("Interactive Analysis Tools - Realistic Test")
    results = run_realistic_test_suite()