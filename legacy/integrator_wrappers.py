from covariant_integrator_library import retarded_integrator as retarded_integrator_original
from numba_optimized_integrator import retarded_integrator_numba
import time

def retarded_integrator_with_optimization(steps, h_step, wall_Z, apt_R, sim_type, 
                                        init_rider, init_driver, mean, cav_spacing, z_cutoff,
                                        use_numba=True, benchmark=False):
    """
    Wrapper function that allows switching between original and Numba-optimized integrators.
    
    Parameters:
    -----------
    use_numba : bool
        If True, use Numba-optimized version. If False, use original version.
    benchmark : bool
        If True, run both versions and compare performance.
    """
    
    if benchmark:
        print("Running performance benchmark...")
        
        # Run original version
        print("Testing original integrator...")
        start_time = time.time()
        traj_orig, traj_drv_orig = retarded_integrator_original(
            steps, h_step, wall_Z, apt_R, sim_type, 
            init_rider, init_driver, mean, cav_spacing, z_cutoff
        )
        original_time = time.time() - start_time
        
        # Run Numba version
        print("Testing Numba-optimized integrator...")
        start_time = time.time()
        traj_numba, traj_drv_numba = retarded_integrator_numba(
            steps, h_step, wall_Z, apt_R, sim_type, 
            init_rider, init_driver, mean, cav_spacing, z_cutoff
        )
        numba_time = time.time() - start_time
        
        # Report results
        speedup = original_time / numba_time
        print(f"\ PERFORMANCE RESULTS:")
        print(f"   Original integrator: {original_time:.2f} seconds")
        print(f"   Numba integrator: {numba_time:.2f} seconds")
        print(f"   Speedup: {speedup:.1f}x")
        
        # Basic consistency check
        final_rider_gamma_orig = traj_orig[-1]['gamma'][0]
        final_rider_gamma_numba = traj_numba[-1]['gamma'][0]
        gamma_diff = abs(final_rider_gamma_orig - final_rider_gamma_numba) / final_rider_gamma_orig
        print(f"   Final gamma difference: {gamma_diff:.2e} (relative)")
        
        if gamma_diff < 1e-10:
            print("Results are numerically consistent")
        else:
            print("Results may have numerical differences")
            
        return traj_numba, traj_drv_numba
        
    elif use_numba:
        return retarded_integrator_numba(
            steps, h_step, wall_Z, apt_R, sim_type, 
            init_rider, init_driver, mean, cav_spacing, z_cutoff
        )
    else:
        return retarded_integrator_original(
            steps, h_step, wall_Z, apt_R, sim_type, 
            init_rider, init_driver, mean, cav_spacing, z_cutoff
        )

# Alias for backward compatibility
retarded_integrator = retarded_integrator_with_optimization