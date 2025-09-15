#!/usr/bin/env python3
"""
Debug legacy trajectory format
"""

import numpy as np
import sys

# Add paths for imports
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator/legacy')

try:
    from bunch_inits import init_bunch
    from covariant_integrator_library import retarded_integrator3
    
    # Test parameters
    starting_distance_rider = 1e-6
    transv_mom_rider = 0.
    starting_Pz_rider = 1e5
    stripped_ions_rider = 1.
    m_particle_rider = 1.007319468
    transv_dist = 1e-4
    pcount_rider = 1
    charge_sign_rider = -1.
    
    starting_distance_driver = 100.
    transv_mom_driver = 0.
    starting_Pz_driver = -1e5
    stripped_ions_driver = 1.
    m_particle_driver = 1.007319468
    pcount_driver = 1
    charge_sign_driver = 1.
    
    # Initialize particles
    init_rider, _ = init_bunch(
        starting_distance_rider, transv_mom_rider, starting_Pz_rider, stripped_ions_rider,
        m_particle_rider, transv_dist, pcount_rider, charge_sign_rider
    )
    
    init_driver, _ = init_bunch(
        starting_distance_driver, transv_mom_driver, starting_Pz_driver, stripped_ions_driver,
        m_particle_driver, transv_dist, pcount_driver, charge_sign_driver
    )
    
    print("Init rider keys:", init_rider.keys())
    print("Init driver keys:", init_driver.keys())
    
    # Run short integration
    static_steps = 1
    ret_steps = 3
    step_size = 1e-9
    wall_pos = 1E5
    aperture = 1E5
    sim_type = 2
    bunch_dist = 1E5
    cav_spacing = 1E5
    z_cutoff = 0
    
    retarded_traj, retarded_drv_traj = retarded_integrator3(
        static_steps, ret_steps, step_size, wall_pos, aperture, sim_type,
        init_rider, init_driver, bunch_dist, cav_spacing, z_cutoff
    )
    
    print(f"\nRetarded trajectory shape: {np.array(retarded_traj).shape}")
    print(f"Retarded driver trajectory shape: {np.array(retarded_drv_traj).shape}")
    
    print(f"\nFirst few retarded_traj elements:")
    for i, step in enumerate(retarded_traj[:3]):
        print(f"  Step {i}: type={type(step)}, shape={np.array(step).shape if hasattr(step, 'shape') or isinstance(step, (list, tuple)) else 'scalar'}")
        print(f"    Content: {step}")
    
    print(f"\nFirst few retarded_drv_traj elements:")
    for i, step in enumerate(retarded_drv_traj[:3]):
        print(f"  Step {i}: type={type(step)}, shape={np.array(step).shape if hasattr(step, 'shape') or isinstance(step, (list, tuple)) else 'scalar'}")
        print(f"    Content: {step}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()