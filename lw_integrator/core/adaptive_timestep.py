"""
Adaptive Timestep Algorithm for Electromagnetic Retardation

CAI: Implement intelligent timestep scaling to handle cases where the 
electromagnetic retardation delay δt >> simulation timestep Δt.

This handles the ultra-relativistic regime where particles nearly chase
their own electromagnetic signals, requiring adaptive computation strategies.

Author: Ben Folsom (human oversight)  
Date: 2025-09-12
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import warnings

# Constants
C_MMNS = 299.792458  # mm/ns


class AdaptiveTimestepController:
    """
    Adaptive timestep controller for electromagnetic retardation simulations.
    
    CAI: Manages timestep scaling based on retardation delay ratios δt/Δt
    to maintain accuracy while enabling efficient computation of ultra-relativistic
    electromagnetic interactions.
    """
    
    def __init__(self, 
                 base_timestep: float = 1e-4,  # ns
                 max_retardation_ratio: float = 1.0,  # δt/Δt
                 min_timestep: float = 1e-8,  # ns
                 max_timestep: float = 1e-2,  # ns
                 adaptation_factor: float = 0.5):
        """
        Initialize adaptive timestep controller.
        
        Args:
            base_timestep: Default simulation timestep (ns)
            max_retardation_ratio: Maximum allowed δt/Δt before adaptation
            min_timestep: Minimum allowed timestep (ns)
            max_timestep: Maximum allowed timestep (ns)  
            adaptation_factor: Timestep scaling factor (0.1-0.9)
        """
        self.base_timestep = base_timestep
        self.max_retardation_ratio = max_retardation_ratio
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.adaptation_factor = adaptation_factor
        
        # Tracking variables
        self.current_timestep = base_timestep
        self.adaptation_count = 0
        self.max_delta_t_encountered = 0.0
        
    def calculate_retardation_delay(self, R: float, beta_dot_nhat: float) -> float:
        """
        Calculate electromagnetic retardation delay using stable formula.
        
        CAI: δt = R/(c*(1-β·n̂)) - the numerically stable formulation
        
        Args:
            R: Distance between particles (mm)
            beta_dot_nhat: β·n̂ (velocity dot unit vector)
            
        Returns:
            Retardation delay δt (ns)
        """
        denominator = 1.0 - beta_dot_nhat
        epsilon = 1e-15
        
        if abs(denominator) < epsilon:
            # CAI: Near-collinear case - retardation becomes very large
            return np.inf
        
        delta_t = R / (C_MMNS * denominator)
        return delta_t
    
    def assess_timestep_adequacy(self, 
                                particle_positions: np.ndarray,
                                particle_velocities: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Assess whether current timestep is adequate for retardation effects.
        
        CAI: Check all particle pairs for retardation delay ratios δt/Δt
        
        Args:
            particle_positions: Array of positions (N, 3) in mm
            particle_velocities: Array of velocities β = v/c (N, 3)
            
        Returns:
            (adequate, max_ratio, diagnostics)
        """
        n_particles = len(particle_positions)
        max_ratio = 0.0
        problematic_pairs = []
        
        diagnostics = {
            'n_particles': n_particles,
            'current_timestep': self.current_timestep,
            'max_ratio': 0.0,
            'problematic_pairs': [],
            'total_pairs_checked': 0
        }
        
        # CAI: Check all particle pairs
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                # Calculate separation vector (from i to j)
                dr = particle_positions[j] - particle_positions[i]
                R = np.linalg.norm(dr)
                
                if R < 1e-12:  # Avoid division by zero
                    continue
                
                # Unit vector from i to j  
                n_hat = dr / R
                
                # CAI: β·n̂ for particle i (the field emitter)
                # This represents how much particle i's motion is aligned with
                # the direction to particle j (where the field propagates)
                beta_i = particle_velocities[i]
                beta_dot_nhat = np.dot(beta_i, n_hat)
                
                # CAI: Also check particle j as emitter (symmetric case)
                beta_j = particle_velocities[j]
                beta_dot_nhat_j = np.dot(beta_j, -n_hat)  # Reverse direction
                
                # Calculate retardation delays for both directions
                delta_t_i_to_j = self.calculate_retardation_delay(R, beta_dot_nhat)
                delta_t_j_to_i = self.calculate_retardation_delay(R, beta_dot_nhat_j)
                
                # Use the larger retardation delay (worst case)
                delta_t = max(delta_t_i_to_j, delta_t_j_to_i) if np.isfinite(delta_t_i_to_j) and np.isfinite(delta_t_j_to_i) else max(delta_t_i_to_j, delta_t_j_to_i)
                
                if np.isfinite(delta_t):
                    ratio = delta_t / self.current_timestep
                    
                    if ratio > max_ratio:
                        max_ratio = ratio
                        self.max_delta_t_encountered = max(self.max_delta_t_encountered, delta_t)
                    
                    if ratio > self.max_retardation_ratio:
                        problematic_pairs.append({
                            'particles': (i, j),
                            'separation_nm': R * 1e6,
                            'beta_dot_nhat_i': beta_dot_nhat,
                            'beta_dot_nhat_j': beta_dot_nhat_j,
                            'delta_t_ns': delta_t,
                            'ratio': ratio
                        })
                
                diagnostics['total_pairs_checked'] += 1
        
        diagnostics['max_ratio'] = max_ratio
        diagnostics['problematic_pairs'] = problematic_pairs
        
        adequate = max_ratio <= self.max_retardation_ratio
        return adequate, max_ratio, diagnostics
    
    def adapt_timestep(self, max_ratio: float) -> float:
        """
        Adapt timestep based on maximum retardation ratio.
        
        CAI: Scale timestep to bring δt/Δt into acceptable range
        Uses logarithmic scaling for extreme ratios to avoid excessive reduction.
        
        Args:
            max_ratio: Maximum δt/Δt ratio encountered
            
        Returns:
            New timestep (ns)
        """
        if max_ratio <= self.max_retardation_ratio:
            # CAI: Current timestep is adequate
            return self.current_timestep
        
        # CAI: For extreme ratios, use logarithmic scaling to avoid excessive reduction
        if max_ratio > 100:
            # Use logarithmic approach for very large ratios
            log_reduction = np.log10(max_ratio / self.max_retardation_ratio) 
            scaling_factor = 10**(-log_reduction * self.adaptation_factor)
        else:
            # Use linear approach for moderate ratios
            target_ratio = self.max_retardation_ratio * self.adaptation_factor
            scaling_factor = target_ratio / max_ratio
        
        new_timestep = self.current_timestep * scaling_factor
        
        # CAI: Apply limits
        new_timestep = max(self.min_timestep, min(self.max_timestep, new_timestep))
        
        self.adaptation_count += 1
        old_timestep = self.current_timestep
        self.current_timestep = new_timestep
        
        print(f"    Adaptation details:")
        print(f"      Max ratio: {max_ratio:.1f}")
        print(f"      Scaling factor: {scaling_factor:.4f}")
        print(f"      Old timestep: {old_timestep*1e6:.2f} μs")
        print(f"      New timestep: {new_timestep*1e6:.2f} μs")
        
        return new_timestep
    
    def calculate_adaptive_timestep(self, particle_data: Dict[str, np.ndarray]) -> float:
        """
        Calculate adaptive timestep for given particle data.
        
        CAI: Convenience method that combines assessment and adaptation.
        
        Args:
            particle_data: Dictionary containing particle positions and velocities
            
        Returns:
            Adaptive timestep (ns)
        """
        # Extract positions and velocities from particle data
        positions = np.column_stack([particle_data['x'], particle_data['y'], particle_data['z']])
        velocities = np.column_stack([particle_data['bx'], particle_data['by'], particle_data['bz']])
        
        # Assess current timestep adequacy
        adequate, max_ratio, diagnostics = self.assess_timestep_adequacy(positions, velocities)
        
        if adequate:
            return self.current_timestep
        else:
            return self.adapt_timestep(max_ratio)
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get comprehensive status report of adaptive timestep system.
        
        Returns:
            Status dictionary with all relevant metrics
        """
        return {
            'current_timestep_ns': self.current_timestep,
            'base_timestep_ns': self.base_timestep,
            'adaptation_count': self.adaptation_count,
            'timestep_reduction_factor': self.current_timestep / self.base_timestep,
            'max_delta_t_encountered_ns': self.max_delta_t_encountered,
            'max_retardation_ratio_limit': self.max_retardation_ratio,
            'timestep_limits': {
                'min_ns': self.min_timestep,
                'max_ns': self.max_timestep
            }
        }


def test_adaptive_timestep_algorithm():
    """
    Test the adaptive timestep algorithm with GeV scenarios.
    
    CAI: Verify that the algorithm correctly identifies and handles
    problematic retardation delay ratios.
    """
    print("🔬 TESTING ADAPTIVE TIMESTEP ALGORITHM")
    print("="*60)
    
    # CAI: Initialize controller
    controller = AdaptiveTimestepController(
        base_timestep=1e-4,  # 100 μs
        max_retardation_ratio=1.0,
        adaptation_factor=0.5
    )
    
    print(f"Initial Configuration:")
    print(f"  Base timestep: {controller.base_timestep*1e6:.1f} μs")
    print(f"  Max δt/Δt ratio: {controller.max_retardation_ratio}")
    print(f"  Adaptation factor: {controller.adaptation_factor}")
    print()
    
    # CAI: Test scenarios - using correct gamma calculation
    test_scenarios = [
        {
            'name': 'Non-relativistic (100 MeV)',
            'gamma': 1 + 100/938.3,  # γ = 1 + KE/rest_mass
            'separation_nm': 100.0
        },
        {
            'name': 'Moderately relativistic (1 GeV)',
            'gamma': 1 + 1000/938.3,  # γ ≈ 2.07
            'separation_nm': 10.0
        },
        {
            'name': 'Ultra-relativistic (3 GeV)',
            'gamma': 1 + 3000/938.3,  # γ ≈ 4.20
            'separation_nm': 2.7
        },
        {
            'name': 'Extreme case (3 GeV at critical separation)',
            'gamma': 3197,  # Use our known problematic case
            'separation_nm': 2.7
        }
    ]
    
    for scenario in test_scenarios:
        print(f"Testing: {scenario['name']}")
        print("-" * 40)
        
        # CAI: Set up particle configuration for worst-case retardation
        gamma = scenario['gamma']
        beta = np.sqrt(1 - 1/gamma**2) if gamma > 1 else gamma * 0.1
        separation_mm = scenario['separation_nm'] * 1e-6
        
        # CAI: Create collinear motion scenario (particle chasing its own signal)
        # Particle 1 at origin, Particle 2 ahead in direction of motion
        positions = np.array([
            [0.0, 0.0, 0.0],                    # Particle 1 (emitter)
            [0.0, 0.0, separation_mm]           # Particle 2 (receiver) ahead
        ])
        
        # CAI: Both particles moving in +z direction (collinear with separation)
        # This creates β·n̂ ≈ β, maximizing retardation delay
        velocities = np.array([
            [0.0, 0.0, beta],                   # Particle 1 velocity
            [0.0, 0.0, beta]                    # Particle 2 velocity  
        ])
        
        print(f"  γ = {gamma:.1f}, β = {beta:.9f}")
        print(f"  Separation = {scenario['separation_nm']:.1f} nm")
        
        # CAI: Assess timestep adequacy
        adequate, max_ratio, diagnostics = controller.assess_timestep_adequacy(
            positions, velocities
        )
        
        print(f"  Max δt/Δt ratio = {max_ratio:.1f}")
        print(f"  Timestep adequate? {'Yes' if adequate else 'No'}")
        
        if not adequate:
            # CAI: Adapt timestep
            old_timestep = controller.current_timestep
            new_timestep = controller.adapt_timestep(max_ratio)
            
            print(f"  Timestep adapted: {old_timestep*1e6:.1f} → {new_timestep*1e6:.1f} μs")
            print(f"  Reduction factor: {new_timestep/old_timestep:.3f}")
            
            # CAI: Verify adaptation worked
            adequate_new, max_ratio_new, _ = controller.assess_timestep_adequacy(
                positions, velocities
            )
            print(f"  New δt/Δt ratio = {max_ratio_new:.2f}")
            print(f"  Now adequate? {'Yes' if adequate_new else 'No'}")
        
        print()
    
    # CAI: Final status
    status = controller.get_status_report()
    print("📊 FINAL STATUS")
    print("-" * 20)
    print(f"Final timestep: {status['current_timestep_ns']*1e6:.2f} μs")
    print(f"Total adaptations: {status['adaptation_count']}")
    print(f"Overall reduction: {status['timestep_reduction_factor']:.3f}x")


if __name__ == "__main__":
    print("⚡ ADAPTIVE TIMESTEP ALGORITHM FOR ELECTROMAGNETIC RETARDATION")
    print("="*80)
    print("Handling ultra-relativistic electromagnetic retardation delays")
    print()
    
    test_adaptive_timestep_algorithm()
    
    print("\n" + "="*80)
    print("🎯 ADAPTIVE TIMESTEP ALGORITHM READY")
    print("="*80)
    print("Next: Integrate with main simulation loop for production use")
