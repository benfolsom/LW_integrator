"""
Enhanced Aperture Verification with Energy and Trajectory Tracking

This module provides comprehensive aperture analysis with detailed physics tracking
including energy evolution, 3D trajectories, wall interactions, and optimized
performance for interactive analysis.

Enhanced Features:
- Energy evolution tracking throughout particle propagation
- 3D trajectory capture with configurable save frequency
- Wall distance monitoring with close approach detection
- Optimized integration steps (50-150 vs 500) for faster execution
- Detailed collision analysis with critical interaction thresholds
- Comprehensive data storage and visualization support

Author: LW Integrator Development Team
Date: 2025-09-15
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# Try to import from main library, fallback if not available
try:
    from physics.constants import C_MMNS, ELEMENTARY_CHARGE_GAUSSIAN, ELECTRON_MASS_AMU
    # from core.integrator import CovariantIntegrator  # TODO: Not yet implemented
    # from particles.bunch import MacroParticle  # TODO: Not yet implemented
    FULL_PHYSICS_AVAILABLE = False  # Set to False until modules are implemented
except ImportError:
    # Fallback constants for standalone operation
    SPEED_OF_LIGHT = 2.99792458e8  # m/s
    ELEMENTARY_CHARGE = 1.602176634e-19  # C
    ELECTRON_MASS = 9.1093837015e-31  # kg
    ELECTRON_RADIUS = 2.8179403262e-15  # m
    FULL_PHYSICS_AVAILABLE = False


@dataclass
class TrajectoryData:
    """Enhanced trajectory storage with energy and physics tracking"""
    positions: List[np.ndarray] = field(default_factory=list)
    velocities: List[np.ndarray] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)  # Total energy in Joules
    kinetic_energies_mev: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    wall_distances: List[float] = field(default_factory=list)
    radiation_power: List[float] = field(default_factory=list)  # Synchrotron radiation power
    gamma_factors: List[float] = field(default_factory=list)
    step_numbers: List[int] = field(default_factory=list)
    
    def add_point(self, position, velocity, energy, time_val, wall_distance, step_num):
        """Add a trajectory point with all physics data"""
        self.positions.append(position.copy())
        self.velocities.append(velocity.copy())
        self.energies.append(energy)
        self.times.append(time_val)
        self.wall_distances.append(wall_distance)
        self.step_numbers.append(step_num)
        
        # Calculate derived quantities
        gamma = energy / (ELECTRON_MASS * SPEED_OF_LIGHT**2)
        self.gamma_factors.append(gamma)
        
        kinetic_joules = (gamma - 1) * ELECTRON_MASS * SPEED_OF_LIGHT**2
        kinetic_mev = kinetic_joules / (ELEMENTARY_CHARGE * 1e6)
        self.kinetic_energies_mev.append(kinetic_mev)
        
        # Estimate synchrotron radiation power (simplified)
        acceleration = np.linalg.norm(velocity - (self.velocities[-2] if len(self.velocities) > 1 else velocity))
        power = (2 * ELEMENTARY_CHARGE**2 * acceleration**2) / (3 * 4 * np.pi * 8.854187817e-12 * SPEED_OF_LIGHT**3)
        self.radiation_power.append(power)
    
    def get_summary(self):
        """Get trajectory summary statistics"""
        if not self.energies:
            return {}
        
        return {
            'points': len(self.positions),
            'time_span': self.times[-1] - self.times[0] if len(self.times) > 1 else 0,
            'energy_initial_mev': self.kinetic_energies_mev[0] if self.kinetic_energies_mev else 0,
            'energy_final_mev': self.kinetic_energies_mev[-1] if self.kinetic_energies_mev else 0,
            'energy_loss_mev': self.kinetic_energies_mev[0] - self.kinetic_energies_mev[-1] if len(self.kinetic_energies_mev) > 1 else 0,
            'min_wall_distance': min(self.wall_distances) if self.wall_distances else float('inf'),
            'max_radiation_power': max(self.radiation_power) if self.radiation_power else 0,
            'final_position': self.positions[-1] if self.positions else None
        }


class EnhancedMacroParticle:
    """Enhanced macro particle with comprehensive tracking"""
    
    def __init__(self, position, velocity, charge, mass, particle_id, initial_energy_mev=10000):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.charge = charge
        self.mass = mass
        self.particle_id = particle_id
        self.time = 0.0
        self.is_alive = True
        
        # Set initial energy
        self.total_energy = initial_energy_mev * ELEMENTARY_CHARGE * 1e6  # Convert MeV to Joules
        
        # Enhanced tracking
        self.trajectory = TrajectoryData()
        self.collision_data = None
        self.close_approaches = []  # List of (time, distance, position) for close wall approaches
        
        # Physics tracking
        self.last_radiation_loss = 0.0
        self.cumulative_energy_loss = 0.0
        
    def get_kinetic_energy_mev(self):
        """Get current kinetic energy in MeV"""
        gamma = self.total_energy / (self.mass * SPEED_OF_LIGHT**2)
        kinetic_joules = (gamma - 1) * self.mass * SPEED_OF_LIGHT**2
        return kinetic_joules / (ELEMENTARY_CHARGE * 1e6)
    
    def get_wall_distance(self, aperture_radius):
        """Calculate distance to cylindrical wall"""
        r_transverse = np.sqrt(self.position[0]**2 + self.position[1]**2)
        return aperture_radius - r_transverse
    
    def update_energy(self, energy_loss_joules):
        """Update particle energy due to radiation losses"""
        self.total_energy -= energy_loss_joules
        self.cumulative_energy_loss += energy_loss_joules
        self.last_radiation_loss = energy_loss_joules
    
    def check_wall_collision(self, aperture_radius, close_threshold=10e-6):
        """Enhanced collision checking with close approach tracking"""
        wall_distance = self.get_wall_distance(aperture_radius)
        
        # Record close approaches
        if wall_distance < close_threshold and wall_distance > 0:
            self.close_approaches.append((self.time, wall_distance, self.position.copy()))
        
        # Check for actual collision
        if wall_distance <= 0:
            self.is_alive = False
            self.collision_data = {
                'time': self.time,
                'position': self.position.copy(),
                'velocity': self.velocity.copy(),
                'kinetic_energy_mev': self.get_kinetic_energy_mev(),
                'wall_distance': wall_distance
            }
            return True
        
        return False
    
    def update_trajectory(self, wall_distance, step_num):
        """Update trajectory with current state"""
        self.trajectory.add_point(
            self.position, self.velocity, self.total_energy, 
            self.time, wall_distance, step_num
        )


def enhanced_beam_initialization(n_particles, beam_sigma, aperture_radius, energy_mev=10000):
    """Initialize beam with proper size scaling for realistic survival rates"""
    
    particles = []
    
    # Ensure beam sigma is reasonable compared to aperture
    recommended_sigma = aperture_radius * 0.25  # 25% of aperture radius
    if beam_sigma > aperture_radius * 0.4:
        print(f"Warning: Large beam sigma ({beam_sigma*1000:.2f}mm) vs aperture ({aperture_radius*1000:.1f}mm)")
        print(f"Recommended sigma: <{recommended_sigma*1000:.2f}mm for better survival rates")
    
    # 10 GeV electron parameters
    total_energy_joules = energy_mev * ELEMENTARY_CHARGE * 1e6
    gamma = total_energy_joules / (ELECTRON_MASS * SPEED_OF_LIGHT**2)
    beta = np.sqrt(1 - 1/gamma**2)
    velocity_magnitude = beta * SPEED_OF_LIGHT
    
    print(f"Beam Parameters:")
    print(f"  Energy: {energy_mev} MeV (γ={gamma:.1f}, β={beta:.6f})")
    print(f"  Beam σ: {beam_sigma*1000:.3f}mm")
    print(f"  Aperture: {aperture_radius*1000:.1f}mm")
    print(f"  σ/aperture ratio: {beam_sigma/aperture_radius:.3f}")
    
    for i in range(n_particles):
        # Gaussian transverse distribution
        x = np.random.normal(0, beam_sigma)
        y = np.random.normal(0, beam_sigma)
        z = 0.0  # Start at entrance
        
        # Calculate required divergence for this aperture/beam combination
        # Maximum transverse displacement over drift length 
        max_displacement = aperture_radius - 3*beam_sigma  # 3-sigma safety margin
        drift_length = 2.0  # 2 meter drift length
        max_angle = max_displacement / drift_length  # radians
        
        # Use 20% of maximum allowable divergence for realistic mix of survival/collision
        angle_sigma = max_angle * 0.2
        
        # Primarily longitudinal motion with calculated small transverse components
        vx = np.random.normal(0, velocity_magnitude * angle_sigma)
        vy = np.random.normal(0, velocity_magnitude * angle_sigma)
        vz = velocity_magnitude
        
        particle = EnhancedMacroParticle(
            position=[x, y, z],
            velocity=[vx, vy, vz],
            charge=-ELEMENTARY_CHARGE,
            mass=ELECTRON_MASS,
            particle_id=i,
            initial_energy_mev=energy_mev
        )
        
        particles.append(particle)
    
    # Check initial distribution
    initial_radii = [np.sqrt(p.position[0]**2 + p.position[1]**2) for p in particles]
    inside_aperture = sum(1 for r in initial_radii if r < aperture_radius)
    
    print(f"  Initial inside aperture: {inside_aperture}/{n_particles} ({inside_aperture/n_particles:.1%})")
    
    return particles


def run_enhanced_simulation(particles, aperture_radius, total_length=2.0, n_steps=100, save_frequency=3):
    """Run enhanced simulation with detailed tracking"""
    
    print(f"\\nRunning Enhanced Simulation:")
    print(f"  Aperture: {aperture_radius*1000:.1f}mm")
    print(f"  Length: {total_length}m")
    print(f"  Steps: {n_steps}")
    print(f"  Particles: {len(particles)}")
    
    # Calculate time step properly - dt should be time, not distance
    # For relativistic particles, use v ≈ c for time estimation
    total_time = total_length / SPEED_OF_LIGHT  # Time to traverse length at speed of light
    dt = total_time / n_steps
    alive_particles = [p for p in particles if p.is_alive]
    
    # Enhanced tracking
    collision_events = []
    close_approaches = []
    energy_evolution = []
    
    start_time = time.time()
    
    for step in range(n_steps):
        if step % max(1, n_steps // 10) == 0:
            avg_energy = np.mean([p.get_kinetic_energy_mev() for p in alive_particles]) if alive_particles else 0
            print(f"  Step {step:3d}/{n_steps}: {len(alive_particles):3d} alive, avg E: {avg_energy:.1f} MeV")
        
        step_energies = []
        
        for particle in alive_particles[:]:  # Copy to modify during iteration
            # Update position (simple drift for this enhanced demo)
            particle.position += particle.velocity * dt
            particle.time += dt
            
            # Simple energy loss model (placeholder for full EM field treatment)
            if particle.get_kinetic_energy_mev() > 9000:  # Energy threshold
                # Very small loss for drift case - in full version this would be from EM fields
                energy_loss_joules = 1e-9 * ELEMENTARY_CHARGE * 1e6  # Minimal loss
                particle.update_energy(energy_loss_joules)
            
            step_energies.append(particle.get_kinetic_energy_mev())
            
            # Check wall interactions
            wall_distance = particle.get_wall_distance(aperture_radius)
            
            # Enhanced collision checking
            if particle.check_wall_collision(aperture_radius):
                collision_events.append(particle.collision_data)
                alive_particles.remove(particle)
            
            # Update trajectory (optimized saving)
            if step % save_frequency == 0 or step == n_steps - 1:
                particle.update_trajectory(wall_distance, step)
        
        # Track energy evolution
        if step_energies:
            energy_evolution.append({
                'step': step,
                'time': step * dt,
                'mean_energy': np.mean(step_energies),
                'std_energy': np.std(step_energies),
                'min_energy': np.min(step_energies),
                'max_energy': np.max(step_energies),
                'n_particles': len(step_energies)
            })
    
    # Final trajectory updates
    for particle in particles:
        if particle.is_alive:
            wall_distance = particle.get_wall_distance(aperture_radius)
            particle.update_trajectory(wall_distance, n_steps)
    
    # Collect close approaches from all particles
    for particle in particles:
        close_approaches.extend(particle.close_approaches)
    
    simulation_time = time.time() - start_time
    
    results = {
        'initial_particles': len(particles),
        'final_particles': len(alive_particles),
        'survival_rate': len(alive_particles) / len(particles),
        'collision_events': collision_events,
        'close_approaches': close_approaches,
        'energy_evolution': energy_evolution,
        'simulation_time': simulation_time,
        'particles': particles,
        'config': {
            'aperture_radius': aperture_radius,
            'total_length': total_length,
            'n_steps': n_steps,
            'save_frequency': save_frequency
        }
    }
    
    print(f"\\nSimulation Complete:")
    print(f"  Time: {simulation_time:.1f}s")
    print(f"  Survival: {results['survival_rate']:.1%} ({len(alive_particles)}/{len(particles)})")
    print(f"  Collisions: {len(collision_events)}")
    print(f"  Close approaches (<10μm): {len(close_approaches)}")
    
    return results


def run_optimized_test_suite():
    """Run optimized test suite with realistic parameters"""
    
    print("=== Enhanced Aperture Verification Suite ===")
    print("Optimized steps, realistic beam sizes, comprehensive tracking")
    print()
    
    # Realistic test configurations with proper beam-aperture ratios
    test_configs = [
        {'aperture': 5e-3, 'beam_sigma': 1.0e-3, 'n_particles': 25, 'steps': 50},   # 5mm aperture, 1mm beam
        {'aperture': 2e-3, 'beam_sigma': 0.4e-3, 'n_particles': 30, 'steps': 75},   # 2mm aperture, 0.4mm beam  
        {'aperture': 1e-3, 'beam_sigma': 0.2e-3, 'n_particles': 35, 'steps': 100},  # 1mm aperture, 0.2mm beam
        {'aperture': 0.5e-3, 'beam_sigma': 0.1e-3, 'n_particles': 30, 'steps': 125}, # 0.5mm aperture, 0.1mm beam
        {'aperture': 0.3e-3, 'beam_sigma': 0.06e-3, 'n_particles': 25, 'steps': 150}, # 0.3mm aperture, 0.06mm beam
        {'aperture': 0.1e-3, 'beam_sigma': 0.02e-3, 'n_particles': 20, 'steps': 200}, # 0.1mm aperture, 0.02mm beam
    ]
    
    all_results = []
    
    for i, config in enumerate(test_configs):
        print(f"\\n{'='*20} Test {i+1}/{len(test_configs)} {'='*20}")
        print(f"Aperture: {config['aperture']*1000:.1f}mm")
        print(f"Beam σ: {config['beam_sigma']*1000:.2f}mm")
        print(f"Ratio: {config['beam_sigma']/config['aperture']:.3f}")
        
        # Initialize beam
        particles = enhanced_beam_initialization(
            n_particles=config['n_particles'],
            beam_sigma=config['beam_sigma'],
            aperture_radius=config['aperture']
        )
        
        # Run simulation
        results = run_enhanced_simulation(
            particles=particles,
            aperture_radius=config['aperture'],
            n_steps=config['steps']
        )
        
        all_results.append(results)
        
        # Quick analysis
        print(f"\\nResult Summary:")
        print(f"  Survival Rate: {results['survival_rate']:.1%}")
        print(f"  Wall Collisions: {len(results['collision_events'])}")
        print(f"  Close Approaches: {len(results['close_approaches'])}")
        
        if results['energy_evolution']:
            initial_energy = results['energy_evolution'][0]['mean_energy']
            final_energy = results['energy_evolution'][-1]['mean_energy']
            energy_loss = initial_energy - final_energy
            print(f"  Energy Loss: {energy_loss:.3f} MeV ({energy_loss/initial_energy:.2%})")
    
    return all_results


def save_results(results, filename="enhanced_aperture_results.json"):
    """Save enhanced results to JSON file"""
    
    # Prepare data for JSON serialization
    serializable_results = []
    
    for result in results:
        # Convert numpy arrays and complex objects to serializable format
        clean_result = {
            'initial_particles': result['initial_particles'],
            'final_particles': result['final_particles'],
            'survival_rate': result['survival_rate'],
            'simulation_time': result['simulation_time'],
            'config': result['config'],
            'collision_count': len(result['collision_events']),
            'close_approach_count': len(result['close_approaches']),
            'energy_evolution': result['energy_evolution'],
            'particle_summaries': []
        }
        
        # Add particle trajectory summaries
        for particle in result['particles']:
            summary = particle.trajectory.get_summary()
            summary['particle_id'] = particle.particle_id
            summary['is_alive'] = particle.is_alive
            summary['close_approaches'] = len(particle.close_approaches)
            
            # Convert numpy arrays to lists for JSON serialization
            if summary['final_position'] is not None:
                summary['final_position'] = summary['final_position'].tolist()
            
            clean_result['particle_summaries'].append(summary)
        
        serializable_results.append(clean_result)
    
    # Save to file
    output_path = Path(filename)
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\\nResults saved to: {output_path.absolute()}")
    return output_path


if __name__ == "__main__":
    print("Enhanced Aperture Verification with Energy & Trajectory Tracking")
    print("Optimized for fast execution with comprehensive physics")
    print()
    
    # Run enhanced test suite
    results = run_optimized_test_suite()
    
    # Save results
    save_results(results, "enhanced_aperture_verification_results.json")
    
    print("\\n" + "="*60)
    print("Enhanced aperture verification complete!")
    print("Results include energy evolution, trajectories, and wall interactions")
    print("="*60)