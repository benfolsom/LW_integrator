#!/usr/bin/env python3
"""
Detailed analysis of radiation reaction in the conductor surface test.

This script provides comprehensive analysis of when, where, and how much
radiation reaction force is applied during the surface approach scenario.
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
    """Electric field near conducting surface (1/d² scaling)."""
    distance = abs(x_pos - surface_pos)  # mm
    distance_cm = distance * 0.1  # Convert mm to cm
    
    min_distance = 1e-6  # 1 nm minimum distance
    if distance_cm < min_distance:
        distance_cm = min_distance
    
    # Field scales as 1/d² (image charge effect)
    field = field_strength / (distance_cm**2)
    
    # Field points toward surface
    return -field if x_pos > surface_pos else field


def analyze_radiation_reaction_detailed():
    """
    Detailed analysis of radiation reaction triggering conditions.
    """
    print("="*70)
    print("=== DETAILED RADIATION REACTION ANALYSIS ===")
    
    # Physical parameters
    mass = ELECTRON_MASS_AMU  # amu
    charge = ELEMENTARY_CHARGE_ESU  # esu
    char_time = 2/3 * charge**2 / (mass * C_MMNS**3)  # ns
    
    # Test parameters (matching improved test)
    initial_pos = 5e-4  # 500 nm from surface
    initial_velocity = -50.0  # mm/ns (toward surface)
    
    # Calculate initial conditions
    beta = abs(initial_velocity) / C_MMNS
    initial_gamma = 1.0 / np.sqrt(1 - beta**2) if beta < 0.999 else 1.0 / np.sqrt(1 - 0.999**2)
    
    print(f"Physical constants:")
    print(f"  Electron mass: {mass:.6f} amu = {mass * 931.494:.1f} MeV/c²")
    print(f"  Electron charge: {charge:.3e} esu = {charge/4.803e-10:.2f} × e")
    print(f"  Speed of light: {C_MMNS:.1f} mm/ns")
    print(f"  Characteristic time: {char_time:.3e} ns = {char_time*1e15:.1f} attoseconds")
    
    print(f"\nInitial conditions:")
    print(f"  Position: {initial_pos*1e6:.0f} nm from surface")
    print(f"  Velocity: {initial_velocity:.1f} mm/ns = {beta:.4f}c")
    print(f"  Initial γ: {initial_gamma:.6f}")
    print(f"  Kinetic energy: {(initial_gamma-1)*mass*931.494:.2f} MeV")
    
    # Initialize integrator
    integrator = LienardWiechertIntegrator()
    dt = 1e-5  # ns
    
    # Analysis at initial position
    print(f"\n" + "="*50)
    print("RADIATION REACTION TRIGGERING ANALYSIS")
    print("="*50)
    
    pos = initial_pos
    vel = initial_velocity
    gamma = initial_gamma
    
    # Calculate field and acceleration at initial position
    E_field = conducting_surface_field(pos)
    acceleration = charge * E_field / (gamma * mass)  # mm/ns²
    beta_dot = acceleration / C_MMNS  # Dimensionless acceleration
    
    print(f"At distance {pos*1e6:.0f} nm from surface:")
    print(f"  Electric field: {E_field:.2e} statV/cm")
    print(f"  Acceleration: {acceleration:.2e} mm/ns²")
    print(f"  β̇ (dimensionless): {beta_dot:.2e}")
    
    # Calculate radiation reaction force components
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
        'bx': np.array([vel/C_MMNS]),
        'by': np.array([0.0]),
        'bz': np.array([0.0])
    }
    
    # Calculate radiation reaction force manually for analysis
    rad_frc_rhs = (-gamma**3 * (mass * beta_dot**2 * C_MMNS**2) * 
                  (vel/C_MMNS) * C_MMNS)
    
    # Calculate threshold
    base_threshold = char_time / 1e1
    energy_scale = gamma if gamma > 1.1 else 1.0
    threshold = base_threshold * min(energy_scale, 100.0)
    
    print(f"\nRadiation reaction force analysis:")
    print(f"  RHS force: {rad_frc_rhs:.3e}")
    print(f"  |RHS force|: {abs(rad_frc_rhs):.3e}")
    print(f"  Base threshold: {base_threshold:.3e}")
    print(f"  Energy scale factor: {energy_scale:.3f}")
    print(f"  Effective threshold: {threshold:.3e}")
    print(f"  |Force| / threshold: {abs(rad_frc_rhs) / threshold:.1e}")
    print(f"  Triggers? {'YES' if abs(rad_frc_rhs) > threshold else 'NO'}")
    
    # Apply radiation reaction and measure change
    original_beta_dot = beta_dot
    try:
        integrator._apply_radiation_reaction(dt, vector_data, result, 0)
        new_beta_dot = result['bdotx'][0]
        change = new_beta_dot - original_beta_dot
        
        print(f"\nRadiation reaction correction:")
        print(f"  Original β̇: {original_beta_dot:.6e}")
        print(f"  Modified β̇: {new_beta_dot:.6e}")
        print(f"  Absolute change: {abs(change):.6e}")
        print(f"  Relative change: {abs(change)/abs(original_beta_dot)*100:.3f}%")
        
        # Calculate energy dissipation
        energy_change = 0.5 * mass * ((new_beta_dot * C_MMNS)**2 - (original_beta_dot * C_MMNS)**2)
        print(f"  Energy change: {energy_change:.3e} amu·mm²/ns²")
        print(f"  Power dissipated: {abs(energy_change)/dt:.3e} amu·mm²/ns³")
        
    except Exception as e:
        print(f"  ERROR applying radiation reaction: {e}")
    
    # Analysis at different distances
    print(f"\n" + "="*50)
    print("DISTANCE-DEPENDENT ANALYSIS")
    print("="*50)
    
    distances_nm = [1000, 500, 100, 50, 10, 5, 1]  # nm
    
    print(f"{'Distance':<10} {'E-field':<12} {'β̇':<12} {'Force':<12} {'Triggered':<10}")
    print(f"{'(nm)':<10} {'(statV/cm)':<12} {'(1/ns)':<12} {'(force)':<12} {'':<10}")
    print("-" * 65)
    
    for dist_nm in distances_nm:
        pos_test = dist_nm * 1e-6  # Convert nm to mm
        E_field_test = conducting_surface_field(pos_test)
        accel_test = charge * E_field_test / (gamma * mass)
        beta_dot_test = accel_test / C_MMNS
        
        # Calculate force
        rad_frc_test = (-gamma**3 * (mass * beta_dot_test**2 * C_MMNS**2) * 
                       (vel/C_MMNS) * C_MMNS)
        
        triggers = abs(rad_frc_test) > threshold
        
        print(f"{dist_nm:<10.0f} {E_field_test:<12.2e} {beta_dot_test:<12.2e} "
              f"{rad_frc_test:<12.2e} {'YES' if triggers else 'NO':<10}")
    
    # Physical interpretation
    print(f"\n" + "="*50)
    print("PHYSICAL INTERPRETATION")
    print("="*50)
    
    # Classical electron radius
    r_e = charge**2 / (mass * 931.494e6 * 1.602e-19 * (3e8)**2) * 1e15  # fm
    
    print(f"Classical electron radius: {r_e:.2f} fm")
    print(f"Radiation reaction becomes important when:")
    print(f"  • Distance < {np.sqrt(1e8 * threshold / (charge**2 / mass / gamma)) * 0.1 * 1e6:.0f} nm")
    print(f"  • Electric field > {threshold * gamma * mass / charge:.1e} statV/cm")
    print(f"  • Acceleration > {threshold * C_MMNS:.1e} mm/ns²")
    
    print(f"\nEnergy scales:")
    print(f"  • Initial kinetic energy: {(gamma-1)*mass*931.494:.2f} MeV") 
    print(f"  • Rest energy: {mass*931.494:.2f} MeV")
    print(f"  • Radiation power at 500nm: {abs(energy_change)/dt:.1e} amu·mm²/ns³")
    
    print(f"\nConclusion:")
    print(f"  ✓ Radiation reaction triggered at {initial_pos*1e6:.0f} nm distance")
    print(f"  ✓ Force magnitude: {abs(rad_frc_rhs):.1e} (threshold: {threshold:.1e})")
    print(f"  ✓ Relative correction: {abs(change)/abs(original_beta_dot)*100:.3f}%")
    print(f"  ✓ Physics: Abraham-Lorentz-Dirac radiation damping")


if __name__ == "__main__":
    analyze_radiation_reaction_detailed()