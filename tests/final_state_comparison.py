#!/usr/bin/env python3
"""
Final State Comparison: Legacy vs Basic vs Optimized

Extract and compare final positions and energies from all three integrators.
Focus on single rider + single driver particle (legacy limitation).
"""

import numpy as np
import time
import sys

# Add paths for imports
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator')
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator/legacy')

from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator
from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator

print("🔬 FINAL STATE COMPARISON: Legacy vs Basic vs Optimized")
print("="*70)


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


def setup_legacy_parameters():
    """Set up parameters for single rider + single driver test."""
    
    # Physical constants
    c_ms = 299792458
    
    # Particle parameters (single rider + single driver)
    transv_dist = 1e-4
    m_particle_rider = 1.007319468  # proton - amu
    m_particle_driver = 1.007319468  # proton - amu (simplified)
    stripped_ions_rider = 1.
    stripped_ions_driver = 1.
    charge_sign_rider = -1.
    charge_sign_driver = 1.
    
    # Initial momentum and position
    starting_Pz_rider = 1e5   # Moderate energy
    starting_Pz_driver = -starting_Pz_rider  # Head-on collision
    transv_mom_rider = 0.
    transv_mom_driver = 0.
    starting_distance_rider = 1e-6
    starting_distance_driver = 100.
    
    # Particle counts (legacy limitation)
    pcount_rider = 1
    pcount_driver = 1
    
    # Integration parameters
    static_steps = 1
    ret_steps = 10  # Short integration
    step_size = 1e-9
    
    # Additional parameters for legacy integrator
    bunch_dist = 1E5
    cav_spacing = 1E5
    aperture = 1E5
    z_cutoff = 0
    wall_pos = 1E5
    sim_type = 2
    
    return {
        'c_ms': c_ms, 'transv_dist': transv_dist,
        'm_particle_rider': m_particle_rider, 'm_particle_driver': m_particle_driver,
        'stripped_ions_rider': stripped_ions_rider, 'stripped_ions_driver': stripped_ions_driver,
        'charge_sign_rider': charge_sign_rider, 'charge_sign_driver': charge_sign_driver,
        'starting_Pz_rider': starting_Pz_rider, 'starting_Pz_driver': starting_Pz_driver,
        'transv_mom_rider': transv_mom_rider, 'transv_mom_driver': transv_mom_driver,
        'starting_distance_rider': starting_distance_rider, 'starting_distance_driver': starting_distance_driver,
        'pcount_rider': pcount_rider, 'pcount_driver': pcount_driver,
        'static_steps': static_steps, 'ret_steps': ret_steps, 'step_size': step_size,
        'bunch_dist': bunch_dist, 'cav_spacing': cav_spacing, 'aperture': aperture,
        'z_cutoff': z_cutoff, 'wall_pos': wall_pos, 'sim_type': sim_type
    }


def run_legacy_integration(params, init_bunch, retarded_integrator3):
    """Run legacy integrator and extract final state."""
    print(f"🔄 Running LEGACY integration...")
    print(f"  Particles: {params['pcount_rider'] + params['pcount_driver']}")
    print(f"  Steps: {params['ret_steps']}")
    print(f"  Step size: {params['step_size']:.2e}")
    
    start_time = time.time()
    
    # Initialize rider bunch
    init_rider, E_MeV_rest_rider = init_bunch(
        params['starting_distance_rider'], params['transv_mom_rider'], 
        params['starting_Pz_rider'], params['stripped_ions_rider'],
        params['m_particle_rider'], params['transv_dist'], 
        params['pcount_rider'], params['charge_sign_rider']
    )
    
    # Initialize driver bunch  
    init_driver, E_MeV_rest_driver = init_bunch(
        params['starting_distance_driver'], params['transv_mom_driver'], 
        params['starting_Pz_driver'], params['stripped_ions_driver'],
        params['m_particle_driver'], params['transv_dist'], 
        params['pcount_driver'], params['charge_sign_driver']
    )
    
    # Run legacy integration (returns trajectories)
    retarded_traj, retarded_drv_traj = retarded_integrator3(
        params['static_steps'], params['ret_steps'], params['step_size'],
        params['wall_pos'], params['aperture'], params['sim_type'],
        init_rider, init_driver, params['bunch_dist'], 
        params['cav_spacing'], params['z_cutoff']
    )
    
    computation_time = time.time() - start_time
    
    # Extract final states (last step of each trajectory)
    rider_final = retarded_traj[-1]  # Last step dictionary
    driver_final = retarded_drv_traj[-1]  # Last step dictionary
    
    print(f"  ✅ Completed in {computation_time:.4f} seconds")
    
    # Calculate final energies
    def calculate_energy(state_dict, mass_amu):
        """Calculate total energy from legacy state dictionary."""
        # Extract momentum from dictionary
        px = state_dict['Px'][0] if isinstance(state_dict['Px'], np.ndarray) else state_dict['Px']
        py = state_dict['Py'][0] if isinstance(state_dict['Py'], np.ndarray) else state_dict['Py']
        pz = state_dict['Pz'][0] if isinstance(state_dict['Pz'], np.ndarray) else state_dict['Pz']
        
        c_mmns = 299.792458  # mm/ns
        
        # Total momentum
        p_total = np.sqrt(px**2 + py**2 + pz**2)
        
        # Relativistic energy
        rest_energy = mass_amu * c_mmns**2  # amu * (mm/ns)^2
        total_energy = np.sqrt(p_total**2 * c_mmns**2 + rest_energy**2)
        kinetic_energy = total_energy - rest_energy
        
        return {
            'total_energy': total_energy,
            'kinetic_energy': kinetic_energy,
            'momentum': p_total
        }
    
    rider_energy = calculate_energy(rider_final, params['m_particle_rider'])
    driver_energy = calculate_energy(driver_final, params['m_particle_driver'])
    
    # Extract final positions
    def extract_position(state_dict):
        """Extract position from state dictionary."""
        x = state_dict['x'][0] if isinstance(state_dict['x'], np.ndarray) else state_dict['x']
        y = state_dict['y'][0] if isinstance(state_dict['y'], np.ndarray) else state_dict['y']
        z = state_dict['z'][0] if isinstance(state_dict['z'], np.ndarray) else state_dict['z']
        return [x, y, z]
    
    rider_pos = extract_position(rider_final)
    driver_pos = extract_position(driver_final)
    
    return {
        'computation_time': computation_time,
        'rider_final_pos': rider_pos,  # [x, y, z] in mm
        'driver_final_pos': driver_pos,  # [x, y, z] in mm
        'rider_energy': rider_energy,
        'driver_energy': driver_energy,
        'success': True
    }


def create_modern_particles(params):
    """Create modern format particles from legacy parameters."""
    
    # Convert legacy initialization to modern format
    c_ms = 299792458  # m/s
    
    # Simple initialization for comparison
    particles = {
        'x': np.array([params['starting_distance_rider'] * 1e-3, params['starting_distance_driver'] * 1e-3]),  # mm to m
        'y': np.array([params['transv_dist'] * 1e-3, params['transv_dist'] * 1e-3]),  # mm to m  
        'z': np.array([params['starting_distance_rider'] * 1e-3, params['starting_distance_driver'] * 1e-3]),  # mm to m
        'vx': np.array([0.0, 0.0]),
        'vy': np.array([0.0, 0.0]),
        'vz': np.array([0.0, 0.0]),  # Will calculate from momentum
        'm': np.array([params['m_particle_rider'] * 931.494102e6, params['m_particle_driver'] * 931.494102e6]),  # amu to eV/c²
        'q': np.array([params['charge_sign_rider'] * params['stripped_ions_rider'] * 1.602176634e-19,
                      params['charge_sign_driver'] * params['stripped_ions_driver'] * 1.602176634e-19]),  # Coulombs
        't': np.array([0.0, 0.0]),  # Time
        'char_time': np.array([1e-18, 1e-18])  # Characteristic time (from legacy)
    }
    
    # Convert momentum to velocity
    c_mmns = 299.792458  # mm/ns
    
    # Rider particle
    pz_rider = params['starting_Pz_rider']  # amu*mm/ns
    m_rider = params['m_particle_rider']    # amu
    gamma_rider = np.sqrt(1 + (pz_rider / (m_rider * c_mmns))**2)
    v_rider = pz_rider / (gamma_rider * m_rider) * 1e6  # mm/ns to m/s
    particles['vz'][0] = v_rider
    
    # Driver particle  
    pz_driver = params['starting_Pz_driver']  # amu*mm/ns
    m_driver = params['m_particle_driver']    # amu
    gamma_driver = np.sqrt(1 + (pz_driver / (m_driver * c_mmns))**2)
    v_driver = pz_driver / (gamma_driver * m_driver) * 1e6  # mm/ns to m/s
    particles['vz'][1] = v_driver
    
    # Calculate gamma factors for both particles
    v2_rider = particles['vx'][0]**2 + particles['vy'][0]**2 + particles['vz'][0]**2
    v2_driver = particles['vx'][1]**2 + particles['vy'][1]**2 + particles['vz'][1]**2
    
    gamma_rider_modern = 1.0 / np.sqrt(1.0 - v2_rider / c_ms**2 + 1e-16)
    gamma_driver_modern = 1.0 / np.sqrt(1.0 - v2_driver / c_ms**2 + 1e-16)
    
    particles['gamma'] = np.array([gamma_rider_modern, gamma_driver_modern])
    
    # Momenta (derived from velocities)
    particles['Px'] = particles['m'] * particles['gamma'] * particles['vx']
    particles['Py'] = particles['m'] * particles['gamma'] * particles['vy']
    particles['Pz'] = particles['m'] * particles['gamma'] * particles['vz']
    particles['Pt'] = particles['m'] * particles['gamma'] * c_ms
    
    # Beta values (v/c)
    particles['bx'] = particles['vx'] / c_ms
    particles['by'] = particles['vy'] / c_ms
    particles['bz'] = particles['vz'] / c_ms
    
    # Beta derivatives (approximately zero for initial conditions)
    particles['bdotx'] = np.array([0.0, 0.0])
    particles['bdoty'] = np.array([0.0, 0.0])
    particles['bdotz'] = np.array([0.0, 0.0])
    
    return particles


def run_modern_integration(integrator, particles, steps, dt, name):
    """Run basic or optimized integrator and extract final state."""
    print(f"🔄 Running {name} integration...")
    print(f"  Particles: {len(particles['x'])}")
    print(f"  Steps: {steps}")
    print(f"  Step size: {dt:.2e}")
    
    current_particles = {k: v.copy() for k, v in particles.items()}
    
    start_time = time.time()
    
    # Integration loop
    for step in range(steps):
        updated_particles = integrator.eqsofmotion_static(
            dt, current_particles, current_particles
        )
        current_particles = updated_particles
    
    computation_time = time.time() - start_time
    
    print(f"  ✅ Completed in {computation_time:.4f} seconds")
    
    # Calculate final energies
    def calculate_energy(particles, i):
        """Calculate energy for particle i."""
        v2 = particles['vx'][i]**2 + particles['vy'][i]**2 + particles['vz'][i]**2
        c = 299792458  # m/s
        gamma = 1.0 / np.sqrt(1.0 - v2 / c**2 + 1e-16)
        
        rest_energy = particles['m'][i]  # eV/c²
        total_energy = gamma * rest_energy
        kinetic_energy = total_energy - rest_energy
        momentum = np.sqrt(gamma**2 - 1) * rest_energy / c  # eV·s/m
        
        return {
            'total_energy': total_energy,
            'kinetic_energy': kinetic_energy,
            'momentum': momentum
        }
    
    rider_energy = calculate_energy(current_particles, 0)
    driver_energy = calculate_energy(current_particles, 1)
    
    return {
        'computation_time': computation_time,
        'rider_final_pos': [current_particles['x'][0], current_particles['y'][0], current_particles['z'][0]],
        'driver_final_pos': [current_particles['x'][1], current_particles['y'][1], current_particles['z'][1]],
        'rider_energy': rider_energy,
        'driver_energy': driver_energy,
        'success': True
    }


def compare_final_states(legacy_result, basic_result, opt_result):
    """Compare final positions and energies."""
    print(f"\n🔍 FINAL STATE COMPARISON")
    print("="*50)
    
    print(f"FINAL POSITIONS (m):")
    print(f"  Legacy  - Rider: [{legacy_result['rider_final_pos'][0]*1e-3:.6e}, {legacy_result['rider_final_pos'][1]*1e-3:.6e}, {legacy_result['rider_final_pos'][2]*1e-3:.6e}]")
    print(f"  Basic   - Rider: [{basic_result['rider_final_pos'][0]:.6e}, {basic_result['rider_final_pos'][1]:.6e}, {basic_result['rider_final_pos'][2]:.6e}]")
    print(f"  Optimized-Rider: [{opt_result['rider_final_pos'][0]:.6e}, {opt_result['rider_final_pos'][1]:.6e}, {opt_result['rider_final_pos'][2]:.6e}]")
    
    print(f"\n  Legacy  - Driver: [{legacy_result['driver_final_pos'][0]*1e-3:.6e}, {legacy_result['driver_final_pos'][1]*1e-3:.6e}, {legacy_result['driver_final_pos'][2]*1e-3:.6e}]")
    print(f"  Basic   - Driver: [{basic_result['driver_final_pos'][0]:.6e}, {basic_result['driver_final_pos'][1]:.6e}, {basic_result['driver_final_pos'][2]:.6e}]")
    print(f"  Optimized-Driver: [{opt_result['driver_final_pos'][0]:.6e}, {opt_result['driver_final_pos'][1]:.6e}, {opt_result['driver_final_pos'][2]:.6e}]")
    
    print(f"\nKINETIC ENERGIES:")
    print(f"  Legacy  - Rider: {legacy_result['rider_energy']['kinetic_energy']:.6e}")
    print(f"  Basic   - Rider: {basic_result['rider_energy']['kinetic_energy']:.6e}")
    print(f"  Optimized-Rider: {opt_result['rider_energy']['kinetic_energy']:.6e}")
    
    print(f"\n  Legacy  - Driver: {legacy_result['driver_energy']['kinetic_energy']:.6e}")
    print(f"  Basic   - Driver: {basic_result['driver_energy']['kinetic_energy']:.6e}")
    print(f"  Optimized-Driver: {opt_result['driver_energy']['kinetic_energy']:.6e}")
    
    # Calculate differences
    print(f"\nPOSITION DIFFERENCES (Legacy vs Basic vs Optimized):")
    
    # Convert legacy positions to meters for comparison
    legacy_rider_pos_m = np.array(legacy_result['rider_final_pos']) * 1e-3
    legacy_driver_pos_m = np.array(legacy_result['driver_final_pos']) * 1e-3
    
    basic_rider_pos = np.array(basic_result['rider_final_pos'])
    basic_driver_pos = np.array(basic_result['driver_final_pos'])
    
    opt_rider_pos = np.array(opt_result['rider_final_pos'])
    opt_driver_pos = np.array(opt_result['driver_final_pos'])
    
    # Position differences
    rider_pos_diff_lb = np.linalg.norm(legacy_rider_pos_m - basic_rider_pos)
    rider_pos_diff_lo = np.linalg.norm(legacy_rider_pos_m - opt_rider_pos)
    rider_pos_diff_bo = np.linalg.norm(basic_rider_pos - opt_rider_pos)
    
    driver_pos_diff_lb = np.linalg.norm(legacy_driver_pos_m - basic_driver_pos)
    driver_pos_diff_lo = np.linalg.norm(legacy_driver_pos_m - opt_driver_pos)
    driver_pos_diff_bo = np.linalg.norm(basic_driver_pos - opt_driver_pos)
    
    print(f"  Rider  - Legacy vs Basic: {rider_pos_diff_lb:.2e} m")
    print(f"  Rider  - Legacy vs Opt:   {rider_pos_diff_lo:.2e} m")
    print(f"  Rider  - Basic vs Opt:    {rider_pos_diff_bo:.2e} m")
    print(f"  Driver - Legacy vs Basic: {driver_pos_diff_lb:.2e} m")
    print(f"  Driver - Legacy vs Opt:   {driver_pos_diff_lo:.2e} m")
    print(f"  Driver - Basic vs Opt:    {driver_pos_diff_bo:.2e} m")
    
    # Energy differences
    print(f"\nENERGY DIFFERENCES:")
    rider_energy_diff_lb = abs(legacy_result['rider_energy']['kinetic_energy'] - basic_result['rider_energy']['kinetic_energy'])
    rider_energy_diff_lo = abs(legacy_result['rider_energy']['kinetic_energy'] - opt_result['rider_energy']['kinetic_energy'])
    rider_energy_diff_bo = abs(basic_result['rider_energy']['kinetic_energy'] - opt_result['rider_energy']['kinetic_energy'])
    
    driver_energy_diff_lb = abs(legacy_result['driver_energy']['kinetic_energy'] - basic_result['driver_energy']['kinetic_energy'])
    driver_energy_diff_lo = abs(legacy_result['driver_energy']['kinetic_energy'] - opt_result['driver_energy']['kinetic_energy'])
    driver_energy_diff_bo = abs(basic_result['driver_energy']['kinetic_energy'] - opt_result['driver_energy']['kinetic_energy'])
    
    print(f"  Rider  - Legacy vs Basic: {rider_energy_diff_lb:.2e}")
    print(f"  Rider  - Legacy vs Opt:   {rider_energy_diff_lo:.2e}")
    print(f"  Rider  - Basic vs Opt:    {rider_energy_diff_bo:.2e}")
    print(f"  Driver - Legacy vs Basic: {driver_energy_diff_lb:.2e}")
    print(f"  Driver - Legacy vs Opt:   {driver_energy_diff_lo:.2e}")
    print(f"  Driver - Basic vs Opt:    {driver_energy_diff_bo:.2e}")
    
    return {
        'position_diffs': {
            'rider_legacy_vs_basic': rider_pos_diff_lb,
            'rider_legacy_vs_opt': rider_pos_diff_lo,
            'rider_basic_vs_opt': rider_pos_diff_bo,
            'driver_legacy_vs_basic': driver_pos_diff_lb,
            'driver_legacy_vs_opt': driver_pos_diff_lo,
            'driver_basic_vs_opt': driver_pos_diff_bo
        },
        'energy_diffs': {
            'rider_legacy_vs_basic': rider_energy_diff_lb,
            'rider_legacy_vs_opt': rider_energy_diff_lo,
            'rider_basic_vs_opt': rider_energy_diff_bo,
            'driver_legacy_vs_basic': driver_energy_diff_lb,
            'driver_legacy_vs_opt': driver_energy_diff_lo,
            'driver_basic_vs_opt': driver_energy_diff_bo
        }
    }


def main():
    """Main comparison function."""
    
    # Setup test parameters
    params = setup_legacy_parameters()
    steps = params['ret_steps']
    dt = params['step_size']
    
    print(f"Test Parameters:")
    print(f"  Total particles: {params['pcount_rider'] + params['pcount_driver']}")
    print(f"  Integration steps: {steps}")
    print(f"  Time step: {dt:.2e} seconds")
    print()
    
    # Check legacy imports
    legacy_available, init_bunch, retarded_integrator3 = check_legacy_imports()
    
    if not legacy_available:
        print("❌ Cannot perform comparison without legacy integrator")
        return
    
    # Run legacy integration
    legacy_result = run_legacy_integration(params, init_bunch, retarded_integrator3)
    
    # Create modern particles from legacy parameters
    initial_particles = create_modern_particles(params)
    
    # Initialize modern integrators
    basic_integrator = LienardWiechertIntegrator()
    opt_integrator = OptimizedLienardWiechertIntegrator()
    
    # Run basic integration
    basic_result = run_modern_integration(
        basic_integrator, initial_particles, steps, dt, "BASIC"
    )
    
    # Run optimized integration
    opt_result = run_modern_integration(
        opt_integrator, initial_particles, steps, dt, "OPTIMIZED"
    )
    
    # Compare final states
    comparison = compare_final_states(legacy_result, basic_result, opt_result)
    
    # Performance summary
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("-"*30)
    print(f"Legacy time:    {legacy_result['computation_time']:.4f}s")
    print(f"Basic time:     {basic_result['computation_time']:.4f}s")
    print(f"Optimized time: {opt_result['computation_time']:.4f}s")
    
    if basic_result['computation_time'] > 0 and opt_result['computation_time'] > 0:
        basic_vs_opt_speedup = basic_result['computation_time'] / opt_result['computation_time']
        print(f"Basic→Optimized speedup: {basic_vs_opt_speedup:.2f}x")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("="*30)
    
    # Check basic vs optimized agreement (should be perfect)
    if (comparison['position_diffs']['rider_basic_vs_opt'] < 1e-14 and 
        comparison['position_diffs']['driver_basic_vs_opt'] < 1e-14 and
        comparison['energy_diffs']['rider_basic_vs_opt'] < 1e-14 and
        comparison['energy_diffs']['driver_basic_vs_opt'] < 1e-14):
        print("✅ BASIC vs OPTIMIZED: Perfect agreement!")
    else:
        print("⚠️  BASIC vs OPTIMIZED: Some differences detected")
    
    # Check legacy vs modern agreement (approximate due to different formulations)
    max_legacy_pos_diff = max(
        comparison['position_diffs']['rider_legacy_vs_basic'],
        comparison['position_diffs']['driver_legacy_vs_basic']
    )
    max_legacy_energy_diff = max(
        comparison['energy_diffs']['rider_legacy_vs_basic'],
        comparison['energy_diffs']['driver_legacy_vs_basic']
    )
    
    if max_legacy_pos_diff < 1e-6 and max_legacy_energy_diff < 1e6:  # Reasonable tolerances
        print("✅ LEGACY vs MODERN: Good agreement within expected tolerances")
    else:
        print(f"⚠️  LEGACY vs MODERN: Differences detected (pos: {max_legacy_pos_diff:.2e}, energy: {max_legacy_energy_diff:.2e})")
        print("   Note: Some differences expected due to different formulations")


if __name__ == "__main__":
    main()