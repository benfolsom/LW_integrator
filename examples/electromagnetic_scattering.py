"""
Example: Two-Particle Electromagnetic Scattering

This example demonstrates a basic electromagnetic scattering simulation
using the LW integrator package with two charged particles.
"""

import numpy as np
import matplotlib.pyplot as plt
from lw_integrator.core.integration import LiénardWiechertIntegrator
from lw_integrator.core.optimized_integration import OptimizedLiénardWiechertIntegrator
from lw_integrator.physics.constants import *


def create_scattering_system(impact_parameter: float = 1e-6, 
                           initial_velocity: float = 0.1) -> dict:
    """
    Create a two-particle scattering system.
    
    Args:
        impact_parameter: Perpendicular separation (mm)
        initial_velocity: Initial velocity as fraction of c
        
    Returns:
        Particle data dictionary
    """
    # Calculate relativistic quantities
    gamma = 1.0 / np.sqrt(1 - initial_velocity**2)
    
    particles = {
        'x': np.array([-5e-6, 5e-6]),        # Start 10 μm apart
        'y': np.array([0.0, impact_parameter]), # Impact parameter offset
        'z': np.array([0.0, 0.0]),
        't': np.array([0.0, 0.0]),
        'Px': np.array([gamma * PROTON_MASS * initial_velocity * C_MMNS, 
                       -gamma * PROTON_MASS * initial_velocity * C_MMNS]),
        'Py': np.array([0.0, 0.0]),
        'Pz': np.array([0.0, 0.0]),
        'Pt': np.array([gamma * PROTON_MASS * C_MMNS**2,
                       gamma * PROTON_MASS * C_MMNS**2]),
        'gamma': np.array([gamma, gamma]),
        'bx': np.array([initial_velocity, -initial_velocity]),
        'by': np.array([0.0, 0.0]),
        'bz': np.array([0.0, 0.0]),
        'bdotx': np.array([0.0, 0.0]),
        'bdoty': np.array([0.0, 0.0]),
        'bdotz': np.array([0.0, 0.0]),
        'q': ELEMENTARY_CHARGE,
        'char_time': np.array([1e-4, 1e-4]),
        'm': PROTON_MASS
    }
    
    return particles


def simulate_scattering(integrator, particles, total_time=1e-3, timestep=1e-6):
    """
    Simulate electromagnetic scattering between two particles.
    
    Args:
        integrator: LW integrator instance
        particles: Initial particle data
        total_time: Simulation duration (ns)
        timestep: Integration timestep (ns)
        
    Returns:
        List of particle states over time
    """
    trajectory = []
    current_particles = {key: np.copy(val) if isinstance(val, np.ndarray) else val 
                        for key, val in particles.items()}
    
    n_steps = int(total_time / timestep)
    
    print(f"Running {n_steps} integration steps...")
    print(f"Initial separation: {np.sqrt((particles['x'][1] - particles['x'][0])**2 + (particles['y'][1] - particles['y'][0])**2)*1e6:.1f} nm")
    print(f"Initial kinetic energy: {(particles['gamma'][0] - 1) * PROTON_MASS:.1f} MeV")
    
    for step in range(n_steps):
        # Store current state
        trajectory.append({key: np.copy(val) if isinstance(val, np.ndarray) else val 
                          for key, val in current_particles.items()})
        
        # Electromagnetic integration step
        result = integrator.eqsofmotion_static(timestep, current_particles, current_particles)
        
        # Update particle data
        current_particles.update(result)
        
        # Update positions (simple Euler integration)
        current_particles['x'] += current_particles['bx'] * C_MMNS * timestep
        current_particles['y'] += current_particles['by'] * C_MMNS * timestep
        current_particles['z'] += current_particles['bz'] * C_MMNS * timestep
        current_particles['t'] += timestep
        
        # Update velocities from momenta
        p_magnitude = np.sqrt(current_particles['Px']**2 + 
                             current_particles['Py']**2 + 
                             current_particles['Pz']**2)
        current_particles['gamma'] = np.sqrt(1 + (p_magnitude / (PROTON_MASS * C_MMNS))**2)
        
        current_particles['bx'] = current_particles['Px'] / (current_particles['gamma'] * PROTON_MASS * C_MMNS)
        current_particles['by'] = current_particles['Py'] / (current_particles['gamma'] * PROTON_MASS * C_MMNS)
        current_particles['bz'] = current_particles['Pz'] / (current_particles['gamma'] * PROTON_MASS * C_MMNS)
        
        # Progress indicator
        if step % (n_steps // 10) == 0:
            progress = 100 * step / n_steps
            separation = np.sqrt((current_particles['x'][1] - current_particles['x'][0])**2 + 
                               (current_particles['y'][1] - current_particles['y'][0])**2)
            print(f"Progress: {progress:.0f}% - Separation: {separation*1e6:.1f} nm")
    
    # Final state
    trajectory.append(current_particles)
    
    return trajectory


def analyze_scattering(trajectory):
    """
    Analyze scattering results and conservation laws.
    
    Args:
        trajectory: List of particle states
        
    Returns:
        Analysis results dictionary
    """
    initial_state = trajectory[0]
    final_state = trajectory[-1]
    
    # Energy conservation
    initial_energy = np.sum(initial_state['Pt'])
    final_energy = np.sum(final_state['Pt'])
    energy_drift = (final_energy - initial_energy) / initial_energy
    
    # Momentum conservation
    initial_momentum = np.array([np.sum(initial_state['Px']), 
                                np.sum(initial_state['Py']), 
                                np.sum(initial_state['Pz'])])
    final_momentum = np.array([np.sum(final_state['Px']), 
                              np.sum(final_state['Py']), 
                              np.sum(final_state['Pz'])])
    momentum_drift = np.linalg.norm(final_momentum - initial_momentum) / np.linalg.norm(initial_momentum)
    
    # Scattering angles
    initial_velocity_1 = np.array([initial_state['bx'][0], initial_state['by'][0], initial_state['bz'][0]])
    final_velocity_1 = np.array([final_state['bx'][0], final_state['by'][0], final_state['bz'][0]])
    
    scattering_angle = np.arccos(np.dot(initial_velocity_1, final_velocity_1) / 
                                (np.linalg.norm(initial_velocity_1) * np.linalg.norm(final_velocity_1)))
    
    # Closest approach
    min_separation = float('inf')
    for state in trajectory:
        separation = np.sqrt((state['x'][1] - state['x'][0])**2 + 
                           (state['y'][1] - state['y'][0])**2)
        min_separation = min(min_separation, separation)
    
    results = {
        'energy_conservation': energy_drift,
        'momentum_conservation': momentum_drift,
        'scattering_angle_deg': np.degrees(scattering_angle),
        'closest_approach_nm': min_separation * 1e6,
        'initial_energy_MeV': initial_energy,
        'final_energy_MeV': final_energy
    }
    
    return results


def plot_trajectory(trajectory):
    """
    Plot particle trajectories and analysis.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract trajectory data
    times = [state['t'][0] for state in trajectory]
    x1 = [state['x'][0] * 1e6 for state in trajectory]  # Convert to nm
    y1 = [state['y'][0] * 1e6 for state in trajectory]
    x2 = [state['x'][1] * 1e6 for state in trajectory]
    y2 = [state['y'][1] * 1e6 for state in trajectory]
    
    # Position trajectories
    ax1.plot(x1, y1, 'b-', label='Particle 1', linewidth=2)
    ax1.plot(x2, y2, 'r-', label='Particle 2', linewidth=2)
    ax1.set_xlabel('X Position (nm)')
    ax1.set_ylabel('Y Position (nm)')
    ax1.set_title('Particle Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Separation vs time
    separations = [np.sqrt((state['x'][1] - state['x'][0])**2 + (state['y'][1] - state['y'][0])**2) * 1e6 
                  for state in trajectory]
    ax2.plot(times, separations, 'g-', linewidth=2)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Separation (nm)')
    ax2.set_title('Particle Separation vs Time')
    ax2.grid(True, alpha=0.3)
    
    # Energy conservation
    energies = [np.sum(state['Pt']) for state in trajectory]
    ax3.plot(times, energies, 'purple', linewidth=2)
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Total Energy (MeV)')
    ax3.set_title('Energy Conservation')
    ax3.grid(True, alpha=0.3)
    
    # Momentum components
    px_total = [np.sum(state['Px']) for state in trajectory]
    py_total = [np.sum(state['Py']) for state in trajectory]
    ax4.plot(times, px_total, 'b-', label='Px total', linewidth=2)
    ax4.plot(times, py_total, 'r-', label='Py total', linewidth=2)
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Total Momentum (MeV/c)')
    ax4.set_title('Momentum Conservation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('electromagnetic_scattering.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Main example: electromagnetic scattering simulation.
    """
    print("🔬 ELECTROMAGNETIC SCATTERING EXAMPLE")
    print("="*50)
    
    # Create particle system
    impact_parameter = 500e-9  # 500 nm impact parameter
    initial_velocity = 0.1     # 0.1c initial velocity
    
    particles = create_scattering_system(impact_parameter, initial_velocity)
    
    print(f"Initial conditions:")
    print(f"  Impact parameter: {impact_parameter*1e9:.0f} nm")
    print(f"  Initial velocity: {initial_velocity:.1f}c")
    print(f"  Particle separation: {np.sqrt((particles['x'][1] - particles['x'][0])**2 + (particles['y'][1] - particles['y'][0])**2)*1e6:.1f} nm")
    print()
    
    # Initialize integrator
    integrator = LiénardWiechertIntegrator()
    
    # Run simulation
    print("Starting electromagnetic scattering simulation...")
    trajectory = simulate_scattering(
        integrator, 
        particles, 
        total_time=5e-4,  # 0.5 μs
        timestep=1e-6     # 1 ns timestep
    )
    
    # Analyze results
    print("\nAnalyzing scattering results...")
    results = analyze_scattering(trajectory)
    
    print(f"\n📊 SCATTERING ANALYSIS RESULTS")
    print("="*50)
    print(f"Energy conservation: {results['energy_conservation']:.2e} (relative drift)")
    print(f"Momentum conservation: {results['momentum_conservation']:.2e} (relative drift)")
    print(f"Scattering angle: {results['scattering_angle_deg']:.1f}°")
    print(f"Closest approach: {results['closest_approach_nm']:.1f} nm")
    print(f"Initial energy: {results['initial_energy_MeV']:.1f} MeV")
    print(f"Final energy: {results['final_energy_MeV']:.1f} MeV")
    
    # Validation
    if abs(results['energy_conservation']) < 1e-3:
        print("✅ Energy conservation: EXCELLENT")
    else:
        print("⚠️  Energy conservation: Review needed")
        
    if abs(results['momentum_conservation']) < 1e-2:
        print("✅ Momentum conservation: GOOD")
    else:
        print("⚠️  Momentum conservation: Review needed")
    
    # Generate plots
    print(f"\n📈 Generating trajectory plots...")
    plot_trajectory(trajectory)
    print("✅ Plots saved as 'electromagnetic_scattering.png'")
    
    print(f"\n🎉 Electromagnetic scattering example complete!")
    return results


if __name__ == "__main__":
    main()
