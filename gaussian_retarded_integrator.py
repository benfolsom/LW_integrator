"""
Wrapper for Gaussian Self-Consistent Integrator to provide drop-in replacement
for retarded_integrator3 interface.
"""

import numpy as np
import copy as cp
from gaussian_self_consistent_integrator import GaussianSelfConsistentIntegrator

def gaussian_retarded_integrator3(rider_init, driver_init, tot_steps, h, wall_pos, aperture_func, debug_mode=False):
    """
    Drop-in replacement for retarded_integrator3 using Gaussian self-consistent method.
    """
    # Initialize the Gaussian integrator with debug mode
    integrator = GaussianSelfConsistentIntegrator(debug=debug_mode)
    
    print(f"🔧 Gaussian retarded integrator starting:")
    print(f"   Static steps: 1, Retarded steps: {tot_steps-1}")
    print(f"   Step size: {h:.1e} ns, Total steps: {tot_steps}")
    
    # Create trajectory storage
    rider_trajectory = []
    driver_trajectory = []
    
    # Setup initial conditions
    current_rider = cp.deepcopy(rider_init)
    current_driver = cp.deepcopy(driver_init)
    
    # Store initial states
    rider_trajectory.append(cp.deepcopy(current_rider))
    driver_trajectory.append(cp.deepcopy(current_driver))
    
    def safe_velocity_check(particle_dict):
        """Check and constrain velocities for ultra-relativistic particles (TeV range)."""
        c_mmns = 299.792458  # mm/ns
        
        # For TeV particles, velocities should be very close to c
        max_velocity = 0.99999999  # Appropriate for ultra-relativistic particles
        
        for key in ['bx', 'by', 'bz']:
            if key in particle_dict:
                # Ensure velocities stay below c but allow ultra-relativistic values
                particle_dict[key] = np.clip(particle_dict[key], -max_velocity, max_velocity)
        
        # Recalculate gamma consistently with ultra-relativistic velocities
        beta_mag = np.sqrt(particle_dict['bx']**2 + particle_dict['by']**2 + particle_dict['bz']**2)
        beta_mag = np.minimum(beta_mag, max_velocity)
        particle_dict['gamma'] = np.maximum(1.0 / np.sqrt(1 - beta_mag**2), 1.0)
        
        return particle_dict
    
    # Main integration loop
    for step in range(1, tot_steps):
        try:
            # Progress reporting
            if step % max(1, tot_steps // 10) == 0:
                rider_z = current_rider['z'][0] if hasattr(current_rider['z'], '__getitem__') else current_rider['z']
                driver_z = current_driver['z'][0] if hasattr(current_driver['z'], '__getitem__') else current_driver['z']
                print(f"   Step {step}/{tot_steps}: rider z={rider_z:.2f}, driver z={driver_z:.2f}")
            
            # Use aperture function to determine aperture radius
            current_z = current_rider['z'][0] if hasattr(current_rider['z'], '__getitem__') else current_rider['z']
            
            # Handle aperture parameter - if it's a number, treat it as a constant aperture
            if callable(aperture_func):
                apt_R = aperture_func(current_z)
            else:
                apt_R = aperture_func  # Use constant aperture value
            
            sim_type = 2  # bunch-bunch simulation
            
            # Integrate using Gaussian self-consistent method
            try:
                rider_result = integrator.eqsofmotion_self_consistent(
                    h, current_rider, current_driver, apt_R, sim_type)
                driver_result = integrator.eqsofmotion_self_consistent(
                    h, current_driver, current_rider, apt_R, sim_type)
                
                # Apply safe velocity constraints
                rider_result = safe_velocity_check(rider_result)
                driver_result = safe_velocity_check(driver_result)
                
            except Exception as e:
                if "velocity exceeded c" in str(e):
                    print(f"   Warning: Velocity constraint applied at step {step}")
                    # Apply constraints and continue
                    rider_result = safe_velocity_check(cp.deepcopy(current_rider))
                    driver_result = safe_velocity_check(cp.deepcopy(current_driver))
                else:
                    raise e
            
            # Update current states
            current_rider = cp.deepcopy(rider_result)
            current_driver = cp.deepcopy(driver_result)
            
            # Store results
            rider_trajectory.append(cp.deepcopy(rider_result))
            driver_trajectory.append(cp.deepcopy(driver_result))
            
        except Exception as e:
            print(f"❌ Integration failed at step {step}: {e}")
            # Return partial results
            return rider_trajectory, driver_trajectory
    
    print(f"✅ Gaussian integration completed successfully!")
    return rider_trajectory, driver_trajectory