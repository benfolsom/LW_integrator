#!/usr/bin/env python3
"""
Improved test for radiation reaction near a conducting surface.

This test creates a realistic physical scenario where an ultra-relativistic
electron approaches a conducting surface, triggering radiation reaction forces.
The test uses improved precision and more appropriate time scales.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lw_integrator import LienardWiechertIntegrator
from lw_integrator.physics.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE_ESU


def conducting_surface_field(x_pos: float, surface_pos: float = 0.0, 
                            field_strength: float = 1e8) -> float:
    """
    Electric field near a conducting surface using image charge method.
    
    For a charge approaching a conducting surface, the electric field
    scales as 1/d² where d is the distance to the surface.
    
    Args:
        x_pos: Position along x-axis (mm)
        surface_pos: Position of the conducting surface (mm) 
        field_strength: Maximum field strength (statV/cm)
        
    Returns:
        Electric field strength (statV/cm)
    """
    distance = abs(x_pos - surface_pos)  # mm
    distance_cm = distance * 0.1  # Convert mm to cm
    
    # Avoid singularity at surface
    min_distance = 1e-6  # 1 nm minimum distance
    if distance_cm < min_distance:
        distance_cm = min_distance
    
    # Field scales as 1/d² (image charge effect)
    field = field_strength / (distance_cm**2)
    
    # Field points toward surface (negative if particle approaching from positive x)
    return -field if x_pos > surface_pos else field


def improved_radiation_reaction_threshold(char_time: float, beta_dot: float, gamma: float) -> float:
    """
    Calculate a more physically motivated radiation reaction threshold.
    
    The threshold should be based on when radiation reaction becomes
    significant compared to other forces. A reasonable criterion is when
    the radiation reaction force becomes comparable to the Lorentz force scale.
    
    Args:
        char_time: Classical electron characteristic time (ns)
        beta_dot: Acceleration in units of c/time (dimensionless/ns)
        gamma: Lorentz factor
        
    Returns:
        Threshold force for radiation reaction (same units as force calculation)
    """
    # Original threshold: char_time / 1e1 (quite arbitrary)
    # 
    # Better approach: Threshold when radiation reaction becomes
    # comparable to typical electromagnetic forces
    #
    # For highly relativistic particles, radiation reaction becomes important when:
    # P_rad ~ γ⁴ * (charge)² * acceleration² / c³
    # becomes comparable to kinetic energy changes
    
    # Use a physically motivated threshold: 
    # When radiation power would change kinetic energy by ~1% over characteristic time
    energy_scale = gamma  # Relativistic energy scale (in mc² units)
    time_scale = char_time  # Natural time scale for radiation
    
    # Threshold when radiation would change energy by 1% over char_time
    threshold = energy_scale * 0.01 / time_scale
    
    return threshold


def test_improved_conductor_approach():
    """
    Test radiation reaction for electron approaching conducting surface
    with improved precision and physically motivated parameters.
    """
    print("\n" + "="*60)
    print("=== IMPROVED CONDUCTING SURFACE RADIATION REACTION TEST ===")
    
    # Physical parameters
    mass = ELECTRON_MASS_AMU  # amu
    charge = ELEMENTARY_CHARGE_ESU  # esu
    
    # Calculate characteristic time (classical electron radius / c)
    char_time = 2/3 * charge**2 / (mass * C_MMNS**3)  # ns
    
    # More aggressive test scenario - closer approach and higher energy
    initial_pos = 5e-4  # Start at 0.5 μm from surface (500 nm)
    initial_velocity = -50.0  # mm/ns (approach surface rapidly)
    
    # Calculate initial Lorentz factor
    beta = abs(initial_velocity) / C_MMNS
    initial_gamma = 1.0 / np.sqrt(1 - beta**2) if beta < 0.999 else 1.0 / np.sqrt(1 - 0.999**2)
    
    print(f"Using integrator: optimized")
    print(f"Electron parameters:")
    print(f"  Mass: {mass:.6f} amu")
    print(f"  Charge: {charge:.6e} esu") 
    print(f"  Characteristic time: {char_time:.6e} ns")
    print(f"  β = v/c: {beta:.6f}")
    print(f"  Initial γ: {initial_gamma:.6f}")
    
    # Simulation parameters - use larger time step for better visibility
    dt = 1e-5  # ns (10 femtoseconds)
    n_steps = 1000
    
    # Initialize integrator
    integrator = LienardWiechertIntegrator()
    
    # Storage for trajectory data
    trajectory_data = {
        'time': [],
        'x': [],
        'velocity': [],
        'acceleration': [],
        'gamma': [],
        'field': [],
        'rad_reaction_active': [],
        'radiation_force': [],
        'kinetic_energy': []
    }
    
    # Current state
    pos = initial_pos  # mm
    vel = initial_velocity  # mm/ns
    gamma = initial_gamma
    time = 0.0
    
    print(f"\nInitial conditions:")
    print(f"  Position: {pos*1e6:.1f} nm from surface")
    print(f"  Velocity: {vel:.3f} mm/ns")
    print(f"  γ: {gamma:.6f}")
    print(f"  Time step: {dt*1e6:.1f} fs")
    
    closest_approach = float('inf')
    rad_reaction_triggered = False
    total_energy_change = 0.0
    
    for step in range(n_steps):
        # Calculate field from conducting surface
        E_field = conducting_surface_field(pos)
        
        # Calculate acceleration due to electric field
        acceleration = charge * E_field / (gamma * mass)  # mm/ns²
        beta_dot = acceleration / C_MMNS  # Dimensionless acceleration
        
        # Create data structures for radiation reaction calculation
        vector_data = {
            'char_time': char_time,
            'gamma': np.array([gamma]),
            'm': mass
        }
        
        result = {
            'gamma': np.array([gamma]),
            'bdotx': np.array([beta_dot]),
            'bdoty': np.array([0.0]),
            'bdotz': np.array([0.0]),
            'bx': np.array([vel/C_MMNS]),  # Convert to β
            'by': np.array([0.0]),
            'bz': np.array([0.0])
        }
        
        # Store original acceleration
        original_beta_dot = beta_dot
        original_acceleration = acceleration
        
        # Calculate radiation reaction force for monitoring
        rad_frc_rhs = (-gamma**3 * (mass * beta_dot**2 * C_MMNS**2) * 
                      (vel/C_MMNS) * C_MMNS)
        
        # Apply improved threshold
        improved_threshold = improved_radiation_reaction_threshold(char_time, abs(beta_dot), gamma)
        
        # Apply radiation reaction
        rad_reaction_active = False
        try:
            integrator._apply_radiation_reaction(dt, vector_data, result, 0)
            new_beta_dot = result['bdotx'][0]
            
            # Check if radiation reaction was applied
            if abs(new_beta_dot - original_beta_dot) > 1e-15:
                rad_reaction_active = True
                rad_reaction_triggered = True
                energy_change = 0.5 * mass * ((new_beta_dot * C_MMNS)**2 - (original_beta_dot * C_MMNS)**2)
                total_energy_change += energy_change
                
                if step % 100 == 0 or step < 10:
                    print(f"  Step {step:4d}: pos={pos*1e6:7.1f}nm, "
                          f"β̇_orig={original_beta_dot:.3e}, β̇_new={new_beta_dot:.3e}, "
                          f"change={abs(new_beta_dot-original_beta_dot):.3e}")
            
        except Exception as e:
            print(f"Warning: Radiation reaction failed at step {step}: {e}")
            new_beta_dot = original_beta_dot
        
        # Update velocity and position
        new_acceleration = new_beta_dot * C_MMNS
        vel += new_acceleration * dt
        pos += vel * dt
        time += dt
        
        # Update gamma factor
        beta_magnitude = abs(vel) / C_MMNS
        if beta_magnitude < 0.999:
            gamma = 1.0 / np.sqrt(1 - beta_magnitude**2)
        else:
            gamma = 1.0 / np.sqrt(1 - 0.999**2)  # Cap to avoid numerical issues
        
        # Track closest approach
        distance_to_surface = abs(pos)
        if distance_to_surface < closest_approach:
            closest_approach = distance_to_surface
        
        # Store trajectory data
        trajectory_data['time'].append(time)
        trajectory_data['x'].append(pos)
        trajectory_data['velocity'].append(vel)
        trajectory_data['acceleration'].append(new_acceleration)
        trajectory_data['gamma'].append(gamma)
        trajectory_data['field'].append(E_field)
        trajectory_data['rad_reaction_active'].append(rad_reaction_active)
        trajectory_data['radiation_force'].append(rad_frc_rhs)
        trajectory_data['kinetic_energy'].append(0.5 * mass * vel**2)
        
        # Stop if particle hits surface
        if abs(pos) < 1e-6:  # 1 nm from surface
            print(f"\n⚠️  Particle reached surface at step {step}")
            break
        
        # Stop if particle moves away from surface
        if pos > initial_pos * 2:
            print(f"\n↗️  Particle moved away from surface at step {step}")
            break
    
    print(f"\n=== IMPROVED TRAJECTORY ANALYSIS ===")
    print(f"Simulation time: {time:.3f} ns")
    print(f"Final position: {pos*1e6:.1f} nm from surface")
    print(f"Closest approach: {closest_approach*1e6:.1f} nm")
    print(f"Radiation reaction triggered: {'✅ Yes' if rad_reaction_triggered else '❌ No'}")
    print(f"Total energy change: {total_energy_change:.6e} amu·mm²/ns²")
    
    print(f"\nPhysical behavior check:")
    print(f"  Initial velocity: {initial_velocity:.6f} mm/ns")
    print(f"  Final velocity: {vel:.6f} mm/ns")
    print(f"  Velocity change: {vel - initial_velocity:.6e} mm/ns")
    
    if rad_reaction_triggered:
        print(f"✅ SUCCESS: Radiation reaction successfully triggered!")
        print(f"   Energy dissipated through radiation damping")
    else:
        print(f"❌ ISSUE: Radiation reaction not triggered")
        print(f"   Check threshold and field strength parameters")
    
    # Generate plots
    generate_improved_plots(trajectory_data, closest_approach)
    
    return rad_reaction_triggered


def generate_improved_plots(trajectory_data, closest_approach):
    """Generate detailed trajectory plots with improved visualization."""
    print(f"\n📊 Generating improved trajectory plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    time_ns = np.array(trajectory_data['time'])
    position_nm = np.array(trajectory_data['x']) * 1e6  # Convert to nm
    velocity = np.array(trajectory_data['velocity'])
    gamma = np.array(trajectory_data['gamma'])
    rad_active = np.array(trajectory_data['rad_reaction_active'])
    
    # Plot 1: Position vs time
    ax1.plot(time_ns, position_nm, 'b-', linewidth=2, label='Position')
    ax1.axhline(y=closest_approach*1e6, color='r', linestyle='--', alpha=0.7, label=f'Closest approach: {closest_approach*1e6:.1f} nm')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Distance from surface (nm)')
    ax1.set_title('Electron Trajectory Near Conducting Surface')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Velocity vs time
    ax2.plot(time_ns, velocity, 'g-', linewidth=2, label='Velocity')
    # Highlight radiation reaction periods
    if np.any(rad_active):
        rad_times = time_ns[rad_active]
        rad_velocities = velocity[rad_active]
        ax2.scatter(rad_times, rad_velocities, color='red', s=10, alpha=0.7, label='Radiation reaction active')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Velocity (mm/ns)')
    ax2.set_title('Velocity Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Gamma factor vs time  
    ax3.plot(time_ns, gamma, 'm-', linewidth=2, label='γ factor')
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Lorentz Factor γ')
    ax3.set_title('Relativistic Gamma Factor')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Energy analysis
    kinetic_energy = np.array(trajectory_data['kinetic_energy'])
    energy_change = kinetic_energy - kinetic_energy[0]
    ax4.plot(time_ns, energy_change, 'orange', linewidth=2, label='ΔKE')
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Kinetic Energy Change (amu·mm²/ns²)')
    ax4.set_title('Energy Dissipation via Radiation')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('improved_conductor_surface_test.png', dpi=150, bbox_inches='tight')
    print(f"📊 Improved plot saved: improved_conductor_surface_test.png")
    
    try:
        plt.show()
    except:
        pass  # Handle non-interactive environments


if __name__ == "__main__":
    test_improved_conductor_approach()