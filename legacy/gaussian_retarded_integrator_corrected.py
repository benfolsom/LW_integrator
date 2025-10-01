"""
Corrected Gaussian Self-Consistent Integrator - Exact copy of retarded_integrator3 logic
with Gaussian enhancement only in the core integration step.
"""

import numpy as np
import copy as cp
from covariant_integrator_library import static_integrator, eqsofmotion_retarded, switching_flat, conducting_flat


def gaussian_enhanced_step(h_step, trajectory, trajectory_drv, i_traj, apt_R, sim_type, max_iter=5, tolerance=1e-10):
    """
    Gaussian self-consistent integration step that iterates to self-consistency
    instead of using the standard single-step retarded integration.

    Args:
        h_step: Integration time step
        trajectory: Rider trajectory array
        trajectory_drv: Driver trajectory array
        i_traj: Current trajectory index
        apt_R: Aperture radius
        sim_type: Simulation type
        max_iter: Maximum iterations for self-consistency
        tolerance: Convergence tolerance for gamma values

    Returns:
        Updated trajectory point dictionary
    """
    # Start with the standard retarded step as initial guess
    traj_guess = eqsofmotion_retarded(h_step, trajectory, trajectory_drv, i_traj, apt_R, sim_type)

    # Iterate to self-consistency
    for iteration in range(max_iter):
        # Store previous iteration
        traj_prev = cp.deepcopy(traj_guess)

        # Use the current guess to compute fields and update trajectory
        # This creates a temporary trajectory array with our guess at position i_traj+1
        temp_trajectory = trajectory.copy()
        if len(temp_trajectory) > i_traj + 1:
            temp_trajectory[i_traj + 1] = traj_guess
        else:
            temp_trajectory.append(traj_guess)

        # Compute new step using the updated trajectory (self-consistent fields)
        traj_new = eqsofmotion_retarded(h_step, temp_trajectory, trajectory_drv, i_traj, apt_R, sim_type)

        # Check convergence using gamma values (most sensitive physics quantity)
        if 'gamma' in traj_prev and 'gamma' in traj_new:
            gamma_prev = np.array(traj_prev['gamma'])
            gamma_new = np.array(traj_new['gamma'])

            # Calculate relative change in gamma
            max_rel_change = np.max(np.abs((gamma_new - gamma_prev) / gamma_prev))

            if max_rel_change < tolerance:
                # Converged!
                return traj_new

        # Update guess for next iteration
        traj_guess = traj_new

    # If we reach here, iteration didn't converge within max_iter
    # Return the last computed value with a warning
    print(
        f"Warning: Gaussian iteration didn't converge in {max_iter} steps (max change: {max_rel_change:.2e})"
    )
    return traj_guess

def gaussian_retarded_integrator3(init_rider, init_driver, steps_tot, h_step, wall_Z, apt_R, debug_mode=False):
    """
    Corrected Gaussian integrator that copies the EXACT logic from retarded_integrator3
    but enhances only the core integration step with Gaussian self-consistent method.
    """
    if debug_mode:
        print("Debug: Corrected Gaussian retarded integrator starting:")
        print(f"   Total steps: {steps_tot}")
        print(f"   Step size: {h_step:.1e} ns")

    # Set default parameters for simulation type 2 (bunch-bunch interactions)
    sim_type = 2  # bunch-bunch simulation
    steps_init = 1  # Use 1 static step like original
    steps_retarded = steps_tot - steps_init
    mean = 1E5  # bunch distance
    cav_spacing = 1E5  # cavity spacing
    z_cutoff = 0  # cutoff position

    # Phase 1: Static integrator (identical to original)
    trajectory, trajectory_drv = static_integrator(
        steps_init, h_step, wall_Z, apt_R, sim_type, init_rider, init_driver, mean, cav_spacing, z_cutoff
    )

    # Phase 2: Create new trajectory arrays (identical to original)
    trajectory_new = [{}] * steps_tot
    trajectory_drv_new = [{}] * steps_tot
    counter = 0  # actually should be passed in from static integrator, but not implemented yet

    print(f"Static phase complete, starting retarded integration...")

    # Phase 3: Main integration loop (EXACT COPY of original logic)
    for i in range(steps_tot):
        if i <= steps_init:
            # Copy static results (identical to original)
            trajectory_new[i] = trajectory[i-1]
            trajectory_drv_new[i] = trajectory_drv[i-1]  # note that init_wall is a dummy vector
        else:
            # RETARDED INTEGRATION - Enhanced with Gaussian self-consistent method
            trajectory_new[i] = gaussian_enhanced_step(h_step, trajectory_new, trajectory_drv_new, i-1, apt_R, sim_type)

            # Handle different simulation types (identical to original)
            if sim_type == 1:
                trajectory_drv_new[i] = switching_flat(trajectory_new[i], wall_Z, apt_R, z_cutoff)
                if np.mean(trajectory_new[i]['z']) > z_cutoff:
                    z_cutoff += cav_spacing
                    wall_Z += cav_spacing
                    ###focusing
                    #trajectory_new[i]['x'] = [1e-6]
                    #trajectory_new[i]['y'] = [1e-6]
            elif sim_type == 0:
                trajectory_drv_new[i] = conducting_flat(trajectory_new[i], wall_Z, apt_R)
            elif sim_type == 2:
                trajectory_drv_new[i] = gaussian_enhanced_step(h_step, trajectory_drv_new, trajectory_new, i-1, apt_R, sim_type)

    print("Corrected Gaussian integration complete!")
    print(f"   Total trajectory points: rider={len(trajectory_new)}, driver={len(trajectory_drv_new)}")

    return trajectory_new, trajectory_drv_new
