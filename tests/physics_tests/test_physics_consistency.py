#!/usr/bin/env python3
"""
Physics-Consistent Comparison: Legacy vs Basic vs Optimized

Corrected comparison using consistent momentum-velocity relationships
and proper unit conversions. Focuses on the core physics rather than
implementation differences.
"""

import numpy as np
import time
import sys

# Add paths for imports
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator')
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator/legacy')

from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator
from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator

print("🔬 PHYSICS-CONSISTENT COMPARISON: Legacy vs Basic vs Optimized")
print("="*70)
print("Focus: Consistent momentum-velocity relationships and unit conversions")
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


def setup_test_parameters():
    """Set up identical test parameters for all integrators."""
    
    # Physical constants (must be consistent across all codes)
    c_mmns = 299.792458  # mm/ns (legacy units)
    c_ms = 299792458     # m/s (modern units)
    
    # Single particle test (legacy limitation)
    n_particles = 2
    
    # Particle parameters 
    m_particle_amu = 1.007319468  # proton mass in amu
    stripped_ions = 1.
    charge_sign_rider = -1.
    charge_sign_driver = 1.
    
    # Initial conditions (moderate energy for stability)
    starting_Pz_amu_mmns = 1e5   # amu*mm/ns
    transv_mom = 0.              # No transverse momentum
    transv_dist = 1e-4           # mm
    starting_distance_rider = 1e-6  # mm  
    starting_distance_driver = 100. # mm
    
    # Integration parameters
    dt_ns = 1e-9  # 1 ns timestep
    steps = 10    # Short integration
    
    return {
        'c_mmns': c_mmns,
        'c_ms': c_ms,
        'n_particles': n_particles,
        'm_particle_amu': m_particle_amu,
        'stripped_ions': stripped_ions,
        'charge_sign_rider': charge_sign_rider,
        'charge_sign_driver': charge_sign_driver,
        'starting_Pz_amu_mmns': starting_Pz_amu_mmns,
        'transv_mom': transv_mom,
        'transv_dist': transv_dist,
        'starting_distance_rider': starting_distance_rider,
        'starting_distance_driver': starting_distance_driver,
        'dt_ns': dt_ns,
        'steps': steps
    }


def run_legacy_integration(params, init_bunch, retarded_integrator3):
    """Run legacy integrator using original physics."""
    print(f"🔄 Running LEGACY integration...")
    
    start_time = time.time()
    
    # Initialize particles using legacy function exactly as in original code
    init_rider, E_rest_rider = init_bunch(
        params['starting_distance_rider'],  # starting_distance
        params['transv_mom'],               # transv_mom
        params['starting_Pz_amu_mmns'],     # starting_Pz
        params['stripped_ions'],            # stripped_ions
        params['m_particle_amu'],           # m_particle
        params['transv_dist'],              # transv_dist
        1,                                  # pcount (rider)
        params['charge_sign_rider']         # charge_sign
    )
    
    init_driver, E_rest_driver = init_bunch(
        params['starting_distance_driver'], # starting_distance
        params['transv_mom'],               # transv_mom
        -params['starting_Pz_amu_mmns'],    # starting_Pz (opposite direction)
        params['stripped_ions'],            # stripped_ions
        params['m_particle_amu'],           # m_particle
        params['transv_dist'],              # transv_dist
        1,                                  # pcount (driver)
        params['charge_sign_driver']        # charge_sign
    )
    
    # Run integration using legacy parameters
    retarded_traj, retarded_drv_traj = retarded_integrator3(
        1,                    # static_steps
        params['steps'],      # ret_steps
        params['dt_ns'],      # step_size
        1E5,                  # wall_pos
        1E5,                  # aperture
        2,                    # sim_type
        init_rider,           # init_rider
        init_driver,          # init_driver
        1E5,                  # bunch_dist
        1E5,                  # cav_spacing
        0                     # z_cutoff
    )
    
    computation_time = time.time() - start_time
    
    # Extract final states
    rider_final = retarded_traj[-1]
    driver_final = retarded_drv_traj[-1]
    
    print(f"  ✅ Completed in {computation_time:.4f} seconds")
    
    return {
        'computation_time': computation_time,
        'rider_final': rider_final,
        'driver_final': driver_final,
        'rider_initial': init_rider,
        'driver_initial': init_driver,
        'success': True
    }


def create_modern_particles_from_legacy(legacy_result, params):
    """Create modern particle format using exactly the same physics as legacy."""
    
    # Extract initial conditions from legacy initialization
    rider_init = legacy_result['rider_initial']
    driver_init = legacy_result['driver_initial']
    
    # Modern particles dictionary with exact legacy physics
    particles = {
        # Positions: Convert mm to m
        'x': np.array([rider_init['x'][0] * 1e-3, driver_init['x'][0] * 1e-3]),
        'y': np.array([rider_init['y'][0] * 1e-3, driver_init['y'][0] * 1e-3]),
        'z': np.array([rider_init['z'][0] * 1e-3, driver_init['z'][0] * 1e-3]),
        
        # Time
        't': np.array([rider_init['t'][0], driver_init['t'][0]]),
        
        # Velocities: Use legacy beta values converted to m/s
        'vx': np.array([rider_init['bx'][0] * params['c_ms'], driver_init['bx'][0] * params['c_ms']]),
        'vy': np.array([rider_init['by'][0] * params['c_ms'], driver_init['by'][0] * params['c_ms']]),
        'vz': np.array([rider_init['bz'][0] * params['c_ms'], driver_init['bz'][0] * params['c_ms']]),
        
        # Gamma factors
        'gamma': np.array([rider_init['gamma'][0], driver_init['gamma'][0]]),
        
        # Masses and charges: Use legacy values exactly
        'm': np.array([rider_init['m'], driver_init['m']]) * 931.494102e6 / 1.007319468,  # Convert to eV/c² 
        'q': np.array([rider_init['q'], driver_init['q']]),
        
        # Characteristic time
        'char_time': np.array([rider_init['char_time'], driver_init['char_time']]),
        
        # Momenta: Convert from legacy amu*mm/ns to modern units 
        'Px': np.array([rider_init['Px'][0], driver_init['Px'][0]]) * 931.494102e6 / params['c_ms'] * 1e-3,  # eV/c
        'Py': np.array([rider_init['Py'][0], driver_init['Py'][0]]) * 931.494102e6 / params['c_ms'] * 1e-3,
        'Pz': np.array([rider_init['Pz'][0], driver_init['Pz'][0]]) * 931.494102e6 / params['c_ms'] * 1e-3,
        'Pt': np.array([rider_init['Pt'][0], driver_init['Pt'][0]]) * 931.494102e6 / params['c_ms'],  # eV
        
        # Beta values (dimensionless)
        'bx': np.array([rider_init['bx'][0], driver_init['bx'][0]]),
        'by': np.array([rider_init['by'][0], driver_init['by'][0]]),
        'bz': np.array([rider_init['bz'][0], driver_init['bz'][0]]),
        
        # Beta derivatives
        'bdotx': np.array([rider_init['bdotx'][0], driver_init['bdotx'][0]]),
        'bdoty': np.array([rider_init['bdoty'][0], driver_init['bdoty'][0]]),
        'bdotz': np.array([rider_init['bdotz'][0], driver_init['bdotz'][0]])
    }
    
    print(f"  Converted legacy initial conditions to modern format")
    print(f"  Rider velocity: ({particles['vx'][0]:.2e}, {particles['vy'][0]:.2e}, {particles['vz'][0]:.2e}) m/s")
    print(f"  Driver velocity: ({particles['vx'][1]:.2e}, {particles['vy'][1]:.2e}, {particles['vz'][1]:.2e}) m/s")
    
    return particles


def run_modern_integration(integrator, particles, params, name):
    """Run basic or optimized integrator using consistent physics."""
    print(f"🔄 Running {name} integration...")
    
    current_particles = {k: v.copy() for k, v in particles.items()}
    
    start_time = time.time()
    
    # Integration loop
    for step in range(params['steps']):
        updated_particles = integrator.eqsofmotion_static(
            params['dt_ns'] * 1e-9,  # Convert ns to s
            current_particles, 
            current_particles
        )
        current_particles = updated_particles
    
    computation_time = time.time() - start_time
    
    print(f"  ✅ Completed in {computation_time:.4f} seconds")
    
    return {
        'computation_time': computation_time,
        'final_particles': current_particles,
        'success': True
    }


def extract_comparable_values(legacy_result, basic_result, opt_result, params):
    """Extract physics quantities for comparison using consistent formulas."""
    
    def calculate_physics_quantities(state_dict, is_legacy=False):
        """Calculate physics quantities consistently."""
        if is_legacy:
            # Legacy format
            x = state_dict['x'][0] if isinstance(state_dict['x'], np.ndarray) else state_dict['x']
            y = state_dict['y'][0] if isinstance(state_dict['y'], np.ndarray) else state_dict['y']
            z = state_dict['z'][0] if isinstance(state_dict['z'], np.ndarray) else state_dict['z']
            
            # Legacy uses amu*mm/ns units
            px = state_dict['Px'][0] if isinstance(state_dict['Px'], np.ndarray) else state_dict['Px']
            py = state_dict['Py'][0] if isinstance(state_dict['Py'], np.ndarray) else state_dict['Py']
            pz = state_dict['Pz'][0] if isinstance(state_dict['Pz'], np.ndarray) else state_dict['Pz']
            
            # Calculate energy using legacy formula
            m_amu = params['m_particle_amu']
            c_mmns = params['c_mmns']
            p_total_amu_mmns = np.sqrt(px**2 + py**2 + pz**2)
            
            # Relativistic energy in legacy units
            rest_energy_amu_mmns2 = m_amu * c_mmns**2
            total_energy_amu_mmns2 = np.sqrt(p_total_amu_mmns**2 * c_mmns**2 + rest_energy_amu_mmns2**2)
            kinetic_energy_amu_mmns2 = total_energy_amu_mmns2 - rest_energy_amu_mmns2
            
            # Convert to Joules for comparison
            amu_kg = 1.66053907e-27
            energy_J = kinetic_energy_amu_mmns2 * amu_kg * (1e6)**2  # mm/ns to m/s conversion
            
            return {
                'position': np.array([x*1e-3, y*1e-3, z*1e-3]),  # Convert mm to m
                'momentum_magnitude': p_total_amu_mmns * amu_kg * 1e6,  # Convert to kg⋅m/s
                'kinetic_energy_J': energy_J,
                'gamma': state_dict['gamma'][0] if isinstance(state_dict['gamma'], np.ndarray) else state_dict['gamma']
            }
        
        else:
            # Modern format
            x, y, z = state_dict['x'], state_dict['y'], state_dict['z']
            vx, vy, vz = state_dict['vx'], state_dict['vy'], state_dict['vz']
            m = state_dict['m']
            
            # Calculate gamma
            v2 = vx**2 + vy**2 + vz**2
            c = params['c_ms']
            gamma = 1.0 / np.sqrt(1.0 - v2 / c**2 + 1e-16)
            
            # Relativistic momentum
            p_magnitude = gamma * np.sqrt(m**2 * v2) / c  # Should be in SI units
            
            # Kinetic energy
            rest_energy = m  # Already in eV
            total_energy = gamma * rest_energy
            kinetic_energy = total_energy - rest_energy
            kinetic_energy_J = kinetic_energy * 1.602176634e-19  # Convert eV to J
            
            return {
                'position': np.array([x, y, z]),
                'momentum_magnitude': p_magnitude * 1.602176634e-19 / c,  # Convert to kg⋅m/s
                'kinetic_energy_J': kinetic_energy_J,
                'gamma': gamma
            }
    
    # Extract physics for both particles
    rider_legacy = calculate_physics_quantities(legacy_result['rider_final'], is_legacy=True)
    driver_legacy = calculate_physics_quantities(legacy_result['driver_final'], is_legacy=True)
    
    rider_basic = calculate_physics_quantities({'x': basic_result['final_particles']['x'][0],
                                               'y': basic_result['final_particles']['y'][0],
                                               'z': basic_result['final_particles']['z'][0],
                                               'vx': basic_result['final_particles']['vx'][0],
                                               'vy': basic_result['final_particles']['vy'][0],
                                               'vz': basic_result['final_particles']['vz'][0],
                                               'm': basic_result['final_particles']['m'][0]})
    
    rider_opt = calculate_physics_quantities({'x': opt_result['final_particles']['x'][0],
                                             'y': opt_result['final_particles']['y'][0],
                                             'z': opt_result['final_particles']['z'][0],
                                             'vx': opt_result['final_particles']['vx'][0],
                                             'vy': opt_result['final_particles']['vy'][0],
                                             'vz': opt_result['final_particles']['vz'][0],
                                             'm': opt_result['final_particles']['m'][0]})
    
    driver_basic = calculate_physics_quantities({'x': basic_result['final_particles']['x'][1],
                                                'y': basic_result['final_particles']['y'][1],
                                                'z': basic_result['final_particles']['z'][1],
                                                'vx': basic_result['final_particles']['vx'][1],
                                                'vy': basic_result['final_particles']['vy'][1],
                                                'vz': basic_result['final_particles']['vz'][1],
                                                'm': basic_result['final_particles']['m'][1]})
    
    driver_opt = calculate_physics_quantities({'x': opt_result['final_particles']['x'][1],
                                              'y': opt_result['final_particles']['y'][1],
                                              'z': opt_result['final_particles']['z'][1],
                                              'vx': opt_result['final_particles']['vx'][1],
                                              'vy': opt_result['final_particles']['vy'][1],
                                              'vz': opt_result['final_particles']['vz'][1],
                                              'm': opt_result['final_particles']['m'][1]})
    
    return {
        'rider': {'legacy': rider_legacy, 'basic': rider_basic, 'optimized': rider_opt},
        'driver': {'legacy': driver_legacy, 'basic': driver_basic, 'optimized': driver_opt}
    }


def print_comparison_results(physics_data, legacy_result, basic_result, opt_result):
    """Print detailed comparison results."""
    
    print(f"\n🔍 PHYSICS COMPARISON RESULTS")
    print("="*60)
    
    for particle_type in ['rider', 'driver']:
        print(f"\n{particle_type.upper()} PARTICLE:")
        print("-" * 30)
        
        leg = physics_data[particle_type]['legacy']
        bas = physics_data[particle_type]['basic']
        opt = physics_data[particle_type]['optimized']
        
        print(f"POSITIONS (m):")
        print(f"  Legacy:    [{leg['position'][0]:.6e}, {leg['position'][1]:.6e}, {leg['position'][2]:.6e}]")
        print(f"  Basic:     [{bas['position'][0]:.6e}, {bas['position'][1]:.6e}, {bas['position'][2]:.6e}]")
        print(f"  Optimized: [{opt['position'][0]:.6e}, {opt['position'][1]:.6e}, {opt['position'][2]:.6e}]")
        
        print(f"\nKINETIC ENERGY (J):")
        print(f"  Legacy:    {leg['kinetic_energy_J']:.6e}")
        print(f"  Basic:     {bas['kinetic_energy_J']:.6e}")
        print(f"  Optimized: {opt['kinetic_energy_J']:.6e}")
        
        print(f"\nGAMMA FACTORS:")
        print(f"  Legacy:    {leg['gamma']:.6f}")
        print(f"  Basic:     {bas['gamma']:.6f}")
        print(f"  Optimized: {opt['gamma']:.6f}")
        
        # Calculate differences
        pos_diff_lb = np.linalg.norm(leg['position'] - bas['position'])
        pos_diff_lo = np.linalg.norm(leg['position'] - opt['position'])
        pos_diff_bo = np.linalg.norm(bas['position'] - opt['position'])
        
        energy_diff_lb = abs(leg['kinetic_energy_J'] - bas['kinetic_energy_J'])
        energy_diff_lo = abs(leg['kinetic_energy_J'] - opt['kinetic_energy_J'])
        energy_diff_bo = abs(bas['kinetic_energy_J'] - opt['kinetic_energy_J'])
        
        print(f"\nDIFFERENCES:")
        print(f"  Position - Legacy vs Basic:     {pos_diff_lb:.2e} m")
        print(f"  Position - Legacy vs Optimized: {pos_diff_lo:.2e} m")
        print(f"  Position - Basic vs Optimized:  {pos_diff_bo:.2e} m")
        print(f"  Energy   - Legacy vs Basic:     {energy_diff_lb:.2e} J")
        print(f"  Energy   - Legacy vs Optimized: {energy_diff_lo:.2e} J")
        print(f"  Energy   - Basic vs Optimized:  {energy_diff_bo:.2e} J")
    
    # Performance summary
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("-" * 30)
    print(f"Legacy time:    {legacy_result['computation_time']:.4f}s")
    print(f"Basic time:     {basic_result['computation_time']:.4f}s")
    print(f"Optimized time: {opt_result['computation_time']:.4f}s")
    
    if basic_result['computation_time'] > 0 and opt_result['computation_time'] > 0:
        speedup = basic_result['computation_time'] / opt_result['computation_time']
        print(f"Basic→Optimized speedup: {speedup:.2f}x")
    
    # Final assessment
    print(f"\n🎯 PHYSICS ASSESSMENT")
    print("=" * 30)
    
    # Check basic vs optimized agreement
    max_pos_diff_bo = max(
        np.linalg.norm(physics_data['rider']['basic']['position'] - physics_data['rider']['optimized']['position']),
        np.linalg.norm(physics_data['driver']['basic']['position'] - physics_data['driver']['optimized']['position'])
    )
    max_energy_diff_bo = max(
        abs(physics_data['rider']['basic']['kinetic_energy_J'] - physics_data['rider']['optimized']['kinetic_energy_J']),
        abs(physics_data['driver']['basic']['kinetic_energy_J'] - physics_data['driver']['optimized']['kinetic_energy_J'])
    )
    
    if max_pos_diff_bo < 1e-14 and max_energy_diff_bo < 1e-14:
        print("✅ BASIC vs OPTIMIZED: Perfect agreement!")
    else:
        print(f"⚠️  BASIC vs OPTIMIZED: Some differences (pos: {max_pos_diff_bo:.2e}, energy: {max_energy_diff_bo:.2e})")
    
    # Check legacy vs modern agreement
    max_pos_diff_legacy = max(
        np.linalg.norm(physics_data['rider']['legacy']['position'] - physics_data['rider']['basic']['position']),
        np.linalg.norm(physics_data['driver']['legacy']['position'] - physics_data['driver']['basic']['position'])
    )
    max_energy_diff_legacy = max(
        abs(physics_data['rider']['legacy']['kinetic_energy_J'] - physics_data['rider']['basic']['kinetic_energy_J']),
        abs(physics_data['driver']['legacy']['kinetic_energy_J'] - physics_data['driver']['basic']['kinetic_energy_J'])
    )
    
    print(f"\nLEGACY vs MODERN:")
    print(f"  Position differences: {max_pos_diff_legacy:.2e} m")
    print(f"  Energy differences:   {max_energy_diff_legacy:.2e} J")
    
    if max_pos_diff_legacy < 1e-10 and max_energy_diff_legacy < 1e-15:
        print("✅ EXCELLENT agreement - Physics consistent across implementations!")
    elif max_pos_diff_legacy < 1e-6 and max_energy_diff_legacy < 1e-12:
        print("✅ GOOD agreement - Minor differences within expected tolerances")
    else:
        print("⚠️  SIGNIFICANT differences - May indicate unit conversion or physics implementation issues")


def main():
    """Main physics-consistent comparison function."""
    
    # Setup consistent test parameters
    params = setup_test_parameters()
    
    print(f"Test Parameters:")
    print(f"  Particles: {params['n_particles']}")
    print(f"  Integration steps: {params['steps']}")
    print(f"  Time step: {params['dt_ns']:.2e} ns")
    print(f"  Initial momentum: ±{params['starting_Pz_amu_mmns']:.2e} amu⋅mm/ns")
    print()
    
    # Check legacy imports
    legacy_available, init_bunch, retarded_integrator3 = check_legacy_imports()
    
    if not legacy_available:
        print("❌ Cannot perform comparison without legacy integrator")
        return
    
    # Run legacy integration
    legacy_result = run_legacy_integration(params, init_bunch, retarded_integrator3)
    
    # Create modern particles using exact legacy physics
    initial_particles = create_modern_particles_from_legacy(legacy_result, params)
    
    # Initialize modern integrators
    basic_integrator = LienardWiechertIntegrator()
    opt_integrator = OptimizedLienardWiechertIntegrator()
    
    # Run basic integration
    basic_result = run_modern_integration(
        basic_integrator, initial_particles, params, "BASIC"
    )
    
    # Run optimized integration
    opt_result = run_modern_integration(
        opt_integrator, initial_particles, params, "OPTIMIZED"
    )
    
    # Extract comparable physics quantities
    physics_data = extract_comparable_values(legacy_result, basic_result, opt_result, params)
    
    # Print detailed comparison
    print_comparison_results(physics_data, legacy_result, basic_result, opt_result)


if __name__ == "__main__":
    main()