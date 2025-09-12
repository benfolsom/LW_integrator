"""
Corrected Covariant Gamma Implementation

This implements the theoretically correct covariant gamma calculation:
γ = (1/mc)[Pt - EM_correction]

Based on the first-principles derivation from LW_covariant_eoms_v12.ipynb
and the Bordovitsyn (2003) paper referenced in the notebooks.

The key insight is that in Liénard-Wiechert field theory, the gamma factor
should include electromagnetic field corrections to be fully consistent
with the covariant formulation.

Author: Ben Folsom  
Date: 2025-09-12
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./lw_integrator'))

from lw_integrator.core.integration import LiénardWiechertIntegrator
from lw_integrator.physics.constants import *


class CovariantLiénardWiechertIntegrator(LiénardWiechertIntegrator):
    """
    Liénard-Wiechert integrator with proper covariant gamma calculation.
    
    This implementation uses the first-principles covariant derivation:
    γ = (1/mc)[Pt - q·q_ext/c·R·(1-β̃·ñ)]
    
    where the electromagnetic correction term comes from the retarded 
    Liénard-Wiechert potential contribution to the four-momentum.
    """
    
    def calculate_covariant_gamma(self, 
                                 particles: dict, 
                                 external_particles: dict,
                                 particle_idx: int,
                                 external_idx: int) -> float:
        """
        Calculate gamma using covariant derivation with EM field corrections.
        
        Args:
            particles: Source particle data
            external_particles: External particle data  
            particle_idx: Index of particle to calculate gamma for
            external_idx: Index of external particle providing EM field
            
        Returns:
            Corrected gamma factor including EM field effects
        """
        # Get particle properties
        m = particles['m']  # Rest mass
        c = C_MMNS         # Speed of light
        q = particles['q'] # Charge
        q_ext = external_particles['q'] # External charge
        
        # Current energy-momentum
        Pt = particles['Pt'][particle_idx]
        
        # Calculate separation vector and distance
        dx = particles['x'][particle_idx] - external_particles['x'][external_idx]
        dy = particles['y'][particle_idx] - external_particles['y'][external_idx] 
        dz = particles['z'][particle_idx] - external_particles['z'][external_idx]
        
        R = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Avoid division by zero
        if R < 1e-15:
            # If particles are at same location, use standard calculation
            beta_squared = (particles['bx'][particle_idx]**2 + 
                           particles['by'][particle_idx]**2 + 
                           particles['bz'][particle_idx]**2)
            return 1.0 / np.sqrt(1.0 - beta_squared)
        
        # Unit vector pointing from external to source particle
        nx = dx / R
        ny = dy / R  
        nz = dz / R
        
        # External particle velocity (beta vector)
        beta_ext_x = external_particles['bx'][external_idx]
        beta_ext_y = external_particles['by'][external_idx]
        beta_ext_z = external_particles['bz'][external_idx]
        
        # Retardation factor K = 1 - β⃗_ext · ñ
        # This accounts for light-travel time effects
        beta_dot_n = beta_ext_x * nx + beta_ext_y * ny + beta_ext_z * nz
        K = 1.0 - beta_dot_n
        
        # Avoid division by zero in retardation factor
        if abs(K) < 1e-10:
            # Extreme retardation case - use standard calculation
            beta_squared = (particles['bx'][particle_idx]**2 + 
                           particles['by'][particle_idx]**2 + 
                           particles['bz'][particle_idx]**2)
            return 1.0 / np.sqrt(1.0 - beta_squared)
        
        # Electromagnetic correction term from covariant derivation
        # This comes from the Liénard-Wiechert potential contribution
        em_correction = (q * q_ext) / (c * R * K)
        
        # Covariant gamma calculation: γ = (1/mc)[Pt - EM_correction]
        gamma_covariant = (Pt - em_correction) / (m * c)
        
        # Ensure gamma is physical (≥ 1)
        if gamma_covariant < 1.0:
            # If covariant calculation gives unphysical result,
            # fall back to standard calculation
            beta_squared = (particles['bx'][particle_idx]**2 + 
                           particles['by'][particle_idx]**2 + 
                           particles['bz'][particle_idx]**2)
            gamma_standard = 1.0 / np.sqrt(1.0 - beta_squared)
            
            print(f"Warning: Covariant gamma = {gamma_covariant:.6f} < 1, using standard = {gamma_standard:.6f}")
            return gamma_standard
        
        return gamma_covariant
    
    def eqsofmotion_static(self, h: float, 
                          vector: dict, 
                          vector_ext: dict,
                          apt_R: float = np.inf,
                          sim_type: int = 2) -> dict:
        """
        Static electromagnetic field integration with covariant gamma calculation.
        
        This extends the base implementation to use the theoretically correct
        covariant gamma calculation for consistency with LW field theory.
        """
        # First, run the standard calculation to get momentum updates
        result = super().eqsofmotion_static(h, vector, vector_ext, apt_R, sim_type)
        
        # Now recalculate gamma using covariant approach
        for i in range(len(vector['x'])):
            gamma_sum = 0.0
            count = 0
            
            # Sum contributions from all external particles
            for j in range(len(vector_ext['x'])):
                if i != j:  # Don't include self-interaction
                    gamma_contrib = self.calculate_covariant_gamma(result, vector_ext, i, j)
                    gamma_sum += gamma_contrib
                    count += 1
            
            if count > 0:
                # Average gamma from all external particle contributions
                result['gamma'][i] = gamma_sum / count
            else:
                # No external particles - use standard calculation
                beta_squared = (result['bx'][i]**2 + result['by'][i]**2 + result['bz'][i]**2)
                result['gamma'][i] = 1.0 / np.sqrt(1.0 - beta_squared)
        
        # Update lab time using corrected gamma
        for i in range(len(vector['x'])):
            result['t'][i] = vector['t'][i] + h * result['gamma'][i]
        
        return result


def test_covariant_implementation():
    """Test the corrected covariant implementation."""
    print("🧪 TESTING CORRECTED COVARIANT IMPLEMENTATION")
    print("="*80)
    
    # Create test system
    v_approach = 0.1  # 0.1c
    gamma_exact = 1.0 / np.sqrt(1 - v_approach**2)
    
    particles = {
        'x': np.array([-1e-6, 1e-6]),    # 2 μm separation
        'y': np.array([0.0, 0.0]),
        'z': np.array([0.0, 0.0]),
        't': np.array([0.0, 0.0]),
        'Px': np.array([gamma_exact * PROTON_MASS * v_approach * C_MMNS,
                       -gamma_exact * PROTON_MASS * v_approach * C_MMNS]),
        'Py': np.array([0.0, 0.0]),
        'Pz': np.array([0.0, 0.0]),
        'Pt': np.array([gamma_exact * PROTON_MASS * C_MMNS**2,
                       gamma_exact * PROTON_MASS * C_MMNS**2]),
        'gamma': np.array([gamma_exact, gamma_exact]),
        'bx': np.array([v_approach, -v_approach]),
        'by': np.array([0.0, 0.0]),
        'bz': np.array([0.0, 0.0]),
        'bdotx': np.array([0.0, 0.0]),
        'bdoty': np.array([0.0, 0.0]),
        'bdotz': np.array([0.0, 0.0]),
        'q': 1.0,
        'char_time': np.array([1e-4, 1e-4]),
        'm': 938.3
    }
    
    h = 1e-6
    
    print(f"Initial conditions:")
    print(f"  Velocity: {v_approach:.3f}c")
    print(f"  Separation: {abs(particles['x'][1] - particles['x'][0])*1e6:.1f} nm")
    print(f"  Initial gamma: {gamma_exact:.6f}")
    
    # Test corrected covariant integrator
    covariant_integrator = CovariantLiénardWiechertIntegrator()
    particles_test = {key: np.copy(val) if isinstance(val, np.ndarray) else val 
                     for key, val in particles.items()}
    
    result_covariant = covariant_integrator.eqsofmotion_static(h, particles_test, particles_test)
    
    print(f"\n📊 Results:")
    print(f"  Final gamma (particle 1): {result_covariant['gamma'][0]:.6f}")
    print(f"  Final gamma (particle 2): {result_covariant['gamma'][1]:.6f}")
    
    # Check energy-momentum consistency
    for i in range(2):
        E = result_covariant['Pt'][i]
        px = result_covariant['Px'][i]
        py = result_covariant['Py'][i]
        pz = result_covariant['Pz'][i]
        mc2 = particles['m'] * C_MMNS**2
        
        E2 = E**2
        p2c2 = px**2 + py**2 + pz**2
        expected_E2 = p2c2 + mc2**2
        
        rel_error = abs(E2 - expected_E2) / expected_E2
        print(f"  Particle {i+1} E-p consistency error: {rel_error:.2e}")
    
    print(f"\n✅ Corrected covariant implementation test complete")


if __name__ == "__main__":
    test_covariant_implementation()
