#!/usr/bin/env python3
"""
Legacy Integrator Verification

Standalone script to run legacy integrator and save results for comparison.
Uses the original demo notebook setup to establish physics baseline.
"""

import numpy as np
import time
import sys
import pickle
import os

# Add paths for imports
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator')
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator/legacy')

print("🔬 LEGACY INTEGRATOR VERIFICATION")
print("="*50)

def check_legacy_imports():
    """Check if legacy components are available."""
    try:
        from bunch_inits import init_bunch
        from covariant_integrator_library import retarded_integrator3
        print("✅ Legacy integrator imported successfully")
        return True, init_bunch, retarded_integrator3
    except ImportError as e:
        print(f"❌ Legacy integrator not available: {e}")
        return False, None, None


def setup_legacy_parameters(n_particles_rider=3, n_particles_driver=3):
    """Set up simulation parameters for legacy integrator."""
    
    # Physical constants (from demo)
    c_ms = 299792458
    
    # Particle parameters (from demo)
    transv_dist = 1e-4
    m_particle_rider = 1.007319468  # proton - amu
    m_particle_driver = 207.2  # lead, amu
    stripped_ions_rider = 1.
    stripped_ions_driver = 54.
    charge_sign_rider = -1.
    charge_sign_driver = 1.
    
    # Initial momentum and position (from demo)
    starting_Pz_rider = 1.01e6   # High energy
    starting_Pz_driver = -starting_Pz_rider/m_particle_driver*m_particle_rider
    transv_mom_rider = 0.
    transv_mom_driver = transv_mom_rider
    starting_distance_rider = 1e-6
    starting_distance_driver = 100.
    
    # Simulation parameters
    sim_type = 2    # bunch-bunch simulations
    pcount_rider = n_particles_rider
    pcount_driver = n_particles_driver
    
    # Integration parameters (from demo)
    static_steps = 1
    ret_steps = 25
    step_size = 2e-6
    
    # Additional parameters (from demo)
    bunch_dist = 1E5
    cav_spacing = 1E5
    aperture = 1E5
    z_cutoff = 0
    wall_pos = 1E5
    
    params = {
        'c_ms': c_ms, 'transv_dist': transv_dist,
        'm_particle_rider': m_particle_rider, 'm_particle_driver': m_particle_driver,
        'stripped_ions_rider': stripped_ions_rider, 'stripped_ions_driver': stripped_ions_driver,
        'charge_sign_rider': charge_sign_rider, 'charge_sign_driver': charge_sign_driver,
        'starting_Pz_rider': starting_Pz_rider, 'starting_Pz_driver': starting_Pz_driver,
        'transv_mom_rider': transv_mom_rider, 'transv_mom_driver': transv_mom_driver,
        'starting_distance_rider': starting_distance_rider, 'starting_distance_driver': starting_distance_driver,
        'sim_type': sim_type, 'pcount_rider': pcount_rider, 'pcount_driver': pcount_driver,
        'static_steps': static_steps, 'ret_steps': ret_steps, 'step_size': step_size,
        'bunch_dist': bunch_dist, 'cav_spacing': cav_spacing, 'aperture': aperture,
        'z_cutoff': z_cutoff, 'wall_pos': wall_pos
    }
    
    print(f"🎯 Legacy Parameters Set:")
    print(f"  Rider particles: {pcount_rider}, Driver particles: {pcount_driver}")
    print(f"  Integration steps: {ret_steps}, Step size: {step_size}")
    print(f"  Transverse separation: {transv_dist}")
    
    return params


def initialize_legacy_particles(params, init_bunch):
    """Initialize particles using legacy bunch initialization."""
    try:
        # Rider bunch initialization
        init_rider, E_MeV_rest_rider = init_bunch(
            params['starting_distance_rider'], params['transv_mom_rider'], 
            params['starting_Pz_rider'], params['stripped_ions_rider'],
            params['m_particle_rider'], params['transv_dist'], 
            params['pcount_rider'], params['charge_sign_rider']
        )
        print(f"Rider: E_rest = {E_MeV_rest_rider}")
        
        # Driver bunch initialization  
        init_driver, E_MeV_rest_driver = init_bunch(
            params['starting_distance_driver'], params['transv_mom_driver'], 
            params['starting_Pz_driver'], params['stripped_ions_driver'],
            params['m_particle_driver'], params['transv_dist'], 
            params['pcount_driver'], params['charge_sign_driver']
        )
        print(f"Driver: E_rest = {E_MeV_rest_driver}")
        
        print(f"✅ Legacy particles initialized:")
        print(f"  Rider particles: {len(init_rider['x'])}")
        print(f"  Driver particles: {len(init_driver['x'])}")
        
        return init_rider, init_driver, E_MeV_rest_rider, E_MeV_rest_driver
        
    except Exception as e:
        print(f"❌ Legacy particle initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def run_legacy_integration(params, init_rider, init_driver, retarded_integrator3):
    """Run integration using legacy code."""
    try:
        print("🔄 Running LEGACY integration...")
        start_time = time.time()
        
        # Run exactly as in demo notebook
        retarded_traj, retarded_drv_traj = retarded_integrator3(
            params['static_steps'], params['ret_steps'], params['step_size'],
            params['wall_pos'], params['aperture'], params['sim_type'],
            init_rider, init_driver, params['bunch_dist'], 
            params['cav_spacing'], params['z_cutoff']
        )
        
        computation_time = time.time() - start_time
        
        print(f"✅ Legacy integration completed in {computation_time:.4f}s")
        print(f"  Rider trajectory steps: {len(retarded_traj)}")
        print(f"  Driver trajectory steps: {len(retarded_drv_traj)}")
        
        return {
            'success': True,
            'rider_trajectory': retarded_traj,
            'driver_trajectory': retarded_drv_traj,
            'computation_time': computation_time,
            'params': params,
            'init_rider': init_rider,
            'init_driver': init_driver
        }
        
    except Exception as e:
        print(f"❌ Legacy integration failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def save_results(result, filename):
    """Save legacy integration results to file."""
    try:
        output_dir = '/home/benfol/work/LW_windows/LW_integrator/tests/results'
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"💾 Results saved to {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return None


def run_legacy_verification_test(n_particles_rider=3, n_particles_driver=3):
    """Run complete legacy verification test."""
    
    print(f"\n{'='*60}")
    print(f"🧪 LEGACY TEST: {n_particles_rider}+{n_particles_driver} particles")
    print(f"{'='*60}")
    
    # Check imports
    legacy_available, init_bunch, retarded_integrator3 = check_legacy_imports()
    if not legacy_available:
        return None
    
    # Setup parameters
    params = setup_legacy_parameters(n_particles_rider, n_particles_driver)
    
    # Initialize particles
    init_rider, init_driver, E_MeV_rest_rider, E_MeV_rest_driver = initialize_legacy_particles(
        params, init_bunch
    )
    
    if init_rider is None or init_driver is None:
        return None
    
    # Run integration
    result = run_legacy_integration(params, init_rider, init_driver, retarded_integrator3)
    
    if result['success']:
        # Save results
        filename = f"legacy_result_{n_particles_rider}+{n_particles_driver}p.pkl"
        save_results(result, filename)
    
    return result


def main():
    """Main legacy verification function."""
    
    # Test cases: (rider_particles, driver_particles)
    test_cases = [
        (3, 3),      # Demo case
        (5, 5),      # Small test
        (10, 10),    # Medium test
    ]
    
    results = []
    
    for n_rider, n_driver in test_cases:
        try:
            result = run_legacy_verification_test(n_rider, n_driver)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Test failed for {n_rider}+{n_driver} particles: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 LEGACY VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = 0
    for i, result in enumerate(results):
        test_case = test_cases[i]
        if result.get('success', False):
            status = "✅ PASS"
            successful_tests += 1
            time_taken = result.get('computation_time', 'N/A')
            print(f"{test_case[0]}+{test_case[1]} particles: {status} - Time: {time_taken:.4f}s")
        else:
            status = "❌ FAIL"
            print(f"{test_case[0]}+{test_case[1]} particles: {status}")
    
    print(f"\n🎯 OVERALL RESULT: {successful_tests}/{len(test_cases)} tests passed")
    if successful_tests == len(test_cases):
        print("   Legacy integrator baseline established successfully!")
    else:
        print("   Some legacy tests failed - check setup")


if __name__ == "__main__":
    main()