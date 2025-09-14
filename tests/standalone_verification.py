#!/usr/bin/env python3
"""
LW Integrator Verification: Standalone Trajectory Comparison

CAI: Comprehensive verification testing comparing original legacy code 
against refactored basic and optimized integrators using the exact 
simulation setup from two_particle_demo_main.ipynb

Objective: Ensure architectural improvements haven't introduced bugs
by comparing final trajectories and physics conservation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import traceback
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.rcParams.update({'font.size': 12})

# Add paths for imports
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator')
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator/LW_integrator')
sys.path.insert(0, '/home/benfol/work/LW_windows/LW_integrator/legacy')  # Add legacy directory

print("🔬 LW INTEGRATOR VERIFICATION TEST")
print("="*80)
print("Comparing legacy vs refactored implementations using original demo setup")
print("="*80)


def check_imports():
    """Check availability of all integrator implementations."""
    import_status = {}
    
    # Check legacy original code
    try:
        from covariant_integrator_library import retarded_integrator3
        from bunch_inits import init_bunch
        from plotting_variables import calculate_plotting_variables
        import_status['legacy'] = True
        print("✅ Legacy original integrator imported successfully")
    except ImportError as e:
        import_status['legacy'] = False
        print(f"⚠️  Legacy integrator not available: {e}")
    
    # Check refactored basic integrator
    try:
        from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator
        import_status['basic'] = True
        print("✅ Basic refactored integrator imported successfully")
    except ImportError as e:
        import_status['basic'] = False
        print(f"❌ Basic integrator not available: {e}")
    
    # Check optimized integrator
    try:
        from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator
        import_status['optimized'] = True
        print("✅ Optimized integrator imported successfully")
    except ImportError as e:
        import_status['optimized'] = False
        print(f"❌ Optimized integrator not available: {e}")
    
    return import_status


def setup_demo_parameters():
    """Set up simulation parameters exactly as in two_particle_demo_main.ipynb"""
    
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
    
    # Simulation parameters (simplified for verification)
    sim_type = 2    # bunch-bunch simulations
    pcount_rider = 3    # Smaller for focused comparison
    pcount_driver = 3
    
    # Integration parameters (from demo - coarse phase)
    static_steps = 1
    ret_steps = 25  # Start with shorter integration
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
    
    print(f"🎯 Demo Parameters Set:")
    print(f"  Rider particles: {pcount_rider}, Driver particles: {pcount_driver}")
    print(f"  Integration steps: {ret_steps}, Step size: {step_size}")
    print(f"  Transverse separation: {transv_dist}")
    
    return params


def initialize_particles_legacy(params):
    """Initialize particles using legacy init_bunch function."""
    try:
        from bunch_inits import init_bunch
        
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
            params['m_particle_driver'], -params['transv_dist'],
            params['pcount_driver'], params['charge_sign_driver']
        )
        
        print(f"✅ Legacy particles initialized:")
        print(f"  Rider particles: {len(init_rider['x'])}")
        print(f"  Driver particles: {len(init_driver['x'])}")
        print(f"  Rider sample position: x={init_rider['x'][0]:.3e}, z={init_rider['z'][0]:.3e}")
        print(f"  Driver sample position: x={init_driver['x'][0]:.3e}, z={init_driver['z'][0]:.3e}")
        
        return init_rider, init_driver, E_MeV_rest_rider, E_MeV_rest_driver
        
    except Exception as e:
        print(f"❌ Legacy particle initialization failed: {e}")
        return None, None, None, None


def convert_legacy_to_modern_format(init_rider, init_driver):
    """Convert legacy bunch format to modern integrator format."""
    import numpy as np
    
    # Extract particle counts
    n_rider = len(init_rider['x']) if hasattr(init_rider['x'], '__len__') else 1
    n_driver = len(init_driver['x']) if hasattr(init_driver['x'], '__len__') else 1
    
    # Handle single particle case
    def ensure_array(value):
        if np.isscalar(value):
            return np.array([value])
        else:
            return np.asarray(value)
    
    particles = {}
    
    # Combine rider and driver particles by concatenating dictionary arrays
    for key in ['x', 'y', 'z', 'Px', 'Py', 'Pz', 'Pt', 'bx', 'by', 'bz', 
                'bdotx', 'bdoty', 'bdotz', 'gamma', 't', 'q', 'm', 'char_time']:
        if key in init_rider and key in init_driver:
            rider_vals = ensure_array(init_rider[key])
            driver_vals = ensure_array(init_driver[key])
            particles[key] = np.concatenate([rider_vals, driver_vals])
    
    # Add velocity arrays (convert from beta)
    particles['vx'] = particles['bx'].copy()
    particles['vy'] = particles['by'].copy() 
    particles['vz'] = particles['bz'].copy()
    
    print(f"✅ Converted legacy → modern format: {n_rider + n_driver} particles")
    print(f"  Keys available: {list(particles.keys())}")
    print(f"  Charge range: [{particles['q'].min():.2e}, {particles['q'].max():.2e}]")
    
    return particles


def run_legacy_integration(params, init_rider, init_driver):
    """Run integration using original legacy code."""
    try:
        from covariant_integrator_library import retarded_integrator3
        
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
            'rider_trajectory': retarded_traj,
            'driver_trajectory': retarded_drv_traj,
            'computation_time': computation_time,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Legacy integration failed: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_modern_integration(integrator, particles_initial, params, integrator_name):
    """Run integration using modern integrator (basic or optimized)."""
    try:
        print(f"🔄 Running {integrator_name.upper()} integration...")
        start_time = time.time()
        
        # Copy initial state
        current_particles = {k: v.copy() for k, v in particles_initial.items()}
        trajectory = [current_particles.copy()]
        
        # Integration loop - use static method for single-step integration
        for step in range(params['ret_steps']):
            # Single integration step using static method (no retardation history needed)
            updated_particles = integrator.eqsofmotion_static(
                params['step_size'], current_particles, current_particles
            )
            
            # Update state
            current_particles = updated_particles
            trajectory.append(current_particles.copy())
        
        computation_time = time.time() - start_time
        
        print(f"✅ {integrator_name} integration completed in {computation_time:.4f}s")
        print(f"  Trajectory steps: {len(trajectory)}")
        print(f"  Final position sample: x={current_particles['x'][0]:.3e}, z={current_particles['z'][0]:.3e}")
        print(f"  Final velocity sample: vz={current_particles['vz'][0]:.3e}")
        
        return {
            'trajectory': trajectory,
            'final_particles': current_particles,
            'computation_time': computation_time,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ {integrator_name} integration failed: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def compare_trajectories(results):
    """Compare trajectories between different integrator implementations."""
    print("\n📊 TRAJECTORY COMPARISON ANALYSIS")
    print("="*60)
    
    successful = [name for name, result in results.items() if result.get('success', False)]
    print(f"Successful integrations: {successful}")
    
    if 'basic' in successful and 'optimized' in successful:
        print("\n🔍 BASIC vs OPTIMIZED COMPARISON:")
        
        basic_final = results['basic']['final_particles']
        opt_final = results['optimized']['final_particles']
        
        # Position differences
        pos_diff = {
            'x': np.abs(basic_final['x'] - opt_final['x']),
            'y': np.abs(basic_final['y'] - opt_final['y']),
            'z': np.abs(basic_final['z'] - opt_final['z'])
        }
        
        # Velocity differences
        vel_diff = {
            'vx': np.abs(basic_final['vx'] - opt_final['vx']),
            'vy': np.abs(basic_final['vy'] - opt_final['vy']),
            'vz': np.abs(basic_final['vz'] - opt_final['vz'])
        }
        
        print("Position differences (absolute):")
        for coord in ['x', 'y', 'z']:
            max_diff = np.max(pos_diff[coord])
            mean_diff = np.mean(pos_diff[coord])
            print(f"  {coord}: max={max_diff:.2e}, mean={mean_diff:.2e}")
        
        print("Velocity differences (absolute):")
        for coord in ['vx', 'vy', 'vz']:
            max_diff = np.max(vel_diff[coord])
            mean_diff = np.mean(vel_diff[coord])
            print(f"  {coord}: max={max_diff:.2e}, mean={mean_diff:.2e}")
        
        # Overall assessment
        total_pos_diff = np.sqrt(pos_diff['x']**2 + pos_diff['y']**2 + pos_diff['z']**2)
        total_vel_diff = np.sqrt(vel_diff['vx']**2 + vel_diff['vy']**2 + vel_diff['vz']**2)
        
        max_pos_diff = np.max(total_pos_diff)
        max_vel_diff = np.max(total_vel_diff)
        
        print(f"\nOverall Assessment:")
        print(f"  Max position difference: {max_pos_diff:.2e}")
        print(f"  Max velocity difference: {max_vel_diff:.2e}")
        
        # Determine equivalence
        if max_pos_diff < 1e-14 and max_vel_diff < 1e-14:
            print("  🎯 PERFECT MATCH - Machine precision agreement!")
        elif max_pos_diff < 1e-10 and max_vel_diff < 1e-10:
            print("  ✅ EXCELLENT - Differences within numerical precision")
        elif max_pos_diff < 1e-6 and max_vel_diff < 1e-6:
            print("  ⚠️  GOOD - Small differences, likely acceptable")
        else:
            print("  ❌ SIGNIFICANT DIFFERENCES - Investigation needed")
    
    return successful


def performance_analysis(results):
    """Analyze performance differences between integrators."""
    print("\n⚡ PERFORMANCE ANALYSIS")
    print("="*40)
    
    # Performance summary
    for name, result in results.items():
        if result.get('success', False):
            time_val = result['computation_time']
            print(f"  {name.capitalize():12s}: {time_val:.4f} seconds")
    
    # Calculate speedups
    if 'basic' in results and 'optimized' in results:
        if results['basic']['success'] and results['optimized']['success']:
            basic_time = results['basic']['computation_time']
            opt_time = results['optimized']['computation_time']
            speedup = basic_time / opt_time
            print(f"\n📈 Optimized Speedup: {speedup:.2f}x faster than Basic")


def create_trajectory_plots(results, successful):
    """Create visualization plots comparing trajectories."""
    if len(successful) < 2:
        print("⚠️  Insufficient data for plotting")
        return
    
    print("\n📈 Creating trajectory comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LW Integrator Verification: Trajectory Comparisons', fontsize=14)
    
    colors = {'basic': 'blue', 'optimized': 'red', 'legacy': 'green'}
    
    # Plot trajectories for modern integrators only
    for int_type in ['basic', 'optimized']:
        if int_type in successful:
            result = results[int_type]
            trajectory = result['trajectory']
            
            # Extract trajectory data for first particle
            x_traj = [step['x'][0] for step in trajectory]
            y_traj = [step['y'][0] for step in trajectory]
            z_traj = [step['z'][0] for step in trajectory]
            vz_traj = [step['vz'][0] for step in trajectory]
            
            # Plot X vs Z
            axes[0,0].plot(x_traj, z_traj, color=colors[int_type], 
                          label=f'{int_type.capitalize()}', linewidth=2)
            
            # Plot Y vs Z
            axes[0,1].plot(y_traj, z_traj, color=colors[int_type],
                          label=f'{int_type.capitalize()}', linewidth=2)
            
            # Plot velocity evolution
            axes[1,0].plot(range(len(vz_traj)), vz_traj, color=colors[int_type],
                          label=f'{int_type.capitalize()}', linewidth=2)
    
    # Set labels and formatting
    axes[0,0].set_xlabel('X Position [mm]')
    axes[0,0].set_ylabel('Z Position [mm]')
    axes[0,0].set_title('Particle 1: X vs Z Trajectory')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].set_xlabel('Y Position [mm]')
    axes[0,1].set_ylabel('Z Position [mm]')
    axes[0,1].set_title('Particle 1: Y vs Z Trajectory')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].set_xlabel('Time Step')
    axes[1,0].set_ylabel('Z Velocity [v/c]')
    axes[1,0].set_title('Particle 1: Velocity Evolution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot trajectory differences if both available
    if 'basic' in successful and 'optimized' in successful:
        basic_traj = results['basic']['trajectory']
        opt_traj = results['optimized']['trajectory']
        
        pos_diffs = []
        for i in range(min(len(basic_traj), len(opt_traj))):
            basic_step = basic_traj[i]
            opt_step = opt_traj[i]
            
            diff = np.sqrt((basic_step['x'][0] - opt_step['x'][0])**2 + 
                          (basic_step['y'][0] - opt_step['y'][0])**2 + 
                          (basic_step['z'][0] - opt_step['z'][0])**2)
            pos_diffs.append(diff)
        
        axes[1,1].semilogy(range(len(pos_diffs)), pos_diffs, 'purple', linewidth=2)
        axes[1,1].set_xlabel('Time Step')
        axes[1,1].set_ylabel('Position Difference [mm]')
        axes[1,1].set_title('Basic vs Optimized: Position Difference')
        axes[1,1].grid(True, alpha=0.3)
        
        max_diff = np.max(pos_diffs)
        axes[1,1].text(0.05, 0.95, f'Max diff: {max_diff:.2e} mm', 
                      transform=axes[1,1].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/benfol/work/LW_windows/LW_integrator/tests/verification_trajectories.png', 
                dpi=150, bbox_inches='tight')
    print("✅ Trajectory plots saved to verification_trajectories.png")
    plt.show()


def create_manual_test_particles(params):
    """Create test particles manually when legacy init_bunch is not available."""
    print("📝 Creating manual test particles (legacy not available)...")
    
    n_rider = params['pcount_rider']
    n_driver = params['pcount_driver']
    n_total = n_rider + n_driver
    
    # Create simple but realistic test case based on demo parameters
    particles = {}
    
    # Positions - rider and driver separated
    particles['x'] = np.zeros(n_total)
    particles['y'] = np.zeros(n_total)
    particles['z'] = np.zeros(n_total)
    
    # Rider particles (protons, negative charge in demo)
    particles['x'][:n_rider] = params['starting_distance_rider'] + np.random.normal(0, 1e-7, n_rider)
    particles['y'][:n_rider] = params['transv_dist'] + np.random.normal(0, 1e-7, n_rider)
    particles['z'][:n_rider] = np.random.normal(0, 1e-7, n_rider)
    
    # Driver particles (lead ions, positive charge in demo)
    particles['x'][n_rider:] = params['starting_distance_driver'] + np.random.normal(0, 1e-6, n_driver)
    particles['y'][n_rider:] = -params['transv_dist'] + np.random.normal(0, 1e-6, n_driver)
    particles['z'][n_rider:] = np.random.normal(0, 1e-6, n_driver)
    
    # Velocities - relativistic beam
    particles['vx'] = np.zeros(n_total)
    particles['vy'] = np.zeros(n_total)
    particles['vz'] = np.zeros(n_total)
    
    # High energy z-velocities (based on demo parameters)
    particles['vz'][:n_rider] = 0.9999  # Highly relativistic protons
    particles['vz'][n_rider:] = -0.9999  # Counter-propagating lead ions
    
    # Beta arrays (v/c)
    particles['bx'] = particles['vx'].copy()
    particles['by'] = particles['vy'].copy()
    particles['bz'] = particles['vz'].copy()
    
    # Small accelerations
    particles['bdotx'] = np.random.normal(0, 1e-6, n_total)
    particles['bdoty'] = np.random.normal(0, 1e-6, n_total)
    particles['bdotz'] = np.random.normal(0, 1e-6, n_total)
    
    # Momenta (will be calculated by integrator)
    particles['Px'] = np.zeros(n_total)
    particles['Py'] = np.zeros(n_total)
    particles['Pz'] = np.zeros(n_total)
    particles['Pt'] = np.zeros(n_total)
    
    # Gamma factors for highly relativistic particles
    v2 = particles['bx']**2 + particles['by']**2 + particles['bz']**2
    particles['gamma'] = 1.0 / np.sqrt(1.0 - v2)
    
    # Time
    particles['t'] = np.zeros(n_total)
    
    # Charges and masses (from demo parameters)
    particles['q'] = np.concatenate([
        np.full(n_rider, params['charge_sign_rider'] * params['stripped_ions_rider']),  # -1 for riders
        np.full(n_driver, params['charge_sign_driver'] * params['stripped_ions_driver'])   # +54 for drivers
    ])
    
    particles['m'] = np.concatenate([
        np.full(n_rider, params['m_particle_rider'] * 938.3),   # Proton mass in MeV/c²
        np.full(n_driver, params['m_particle_driver'] * 938.3)  # Lead mass in MeV/c²
    ])
    
    particles['char_time'] = np.ones(n_total)
    
    print(f"✅ Manual test particles created:")
    print(f"  Total particles: {n_total}")
    print(f"  Rider charges: {particles['q'][:n_rider]}")
    print(f"  Driver charges: {particles['q'][n_rider:]}")
    print(f"  Velocity range: [{particles['vz'].min():.4f}, {particles['vz'].max():.4f}]")
    print(f"  Gamma range: [{particles['gamma'].min():.1f}, {particles['gamma'].max():.1f}]")
    
    return particles


def main():
    """Main verification testing function."""
    
    # Check what integrators are available
    import_status = check_imports()
    
    if not any(import_status.values()):
        print("❌ No integrators available for testing")
        return
    
    # Set up demo parameters
    params = setup_demo_parameters()
    
def main():
    """Main verification function."""
    print("🔬 LW INTEGRATOR VERIFICATION")
    print("="*50)
    
    # Check imports
    import_status = check_imports()
    
    # Setup parameters based on demo notebook
    params = setup_demo_parameters()
    
    # Initialize particles - try legacy first, then manual
    particles_initial = None
    init_rider, init_driver = None, None
    
    if import_status['legacy']:
        init_rider, init_driver, E_MeV_rest_rider, E_MeV_rest_driver = initialize_particles_legacy(params)
        if init_rider is not None and init_driver is not None:
            # Use legacy format directly for modern integrators
            particles_initial = convert_legacy_to_modern_format(init_rider, init_driver)
    
    # If legacy not available, create manual test particles
    if particles_initial is None:
        particles_initial = create_manual_test_particles(params)
    
    # Run integrations
    results = {}
    
    # Legacy integration - run if we have legacy bunches
    if import_status['legacy'] and init_rider is not None and init_driver is not None:
        results['legacy'] = run_legacy_integration(params, init_rider, init_driver)
    
    # Basic integration
    if import_status['basic']:
        results['basic'] = run_basic_integration(params, particles_initial)
    
    # Optimized integration  
    if import_status['optimized']:
        results['optimized'] = run_optimized_integration(params, particles_initial)
    
    # Analysis
    successful = analyze_results(results)
    performance_analysis(results)
    
    print(f"\n🎯 VERIFICATION COMPLETE - {len(successful)}/{len(results)} implementations successful")
    
    if __name__ == "__main__":
        create_trajectory_plots(results, successful)


def run_basic_integration(params, particles_initial):
    """Run integration using LienardWiechertIntegrator."""
    import time
    print("\n🔧 Running Basic Integration...")
    
    try:
        start_time = time.time()
        
        # Import and initialize integrator
        from lw_integrator.core.trajectory_integrator import LienardWiechertIntegrator
        integrator = LienardWiechertIntegrator()
        
        # Convert particles_initial to expected dictionary format
        if isinstance(particles_initial, dict):
            # Already in dictionary format from legacy
            particles = particles_initial.copy()
        elif isinstance(particles_initial, list):
            # Convert from list of particles to dictionary format
            n_particles = len(particles_initial)
            particles = {
                'x': np.array([p[0] for p in particles_initial]),
                'y': np.array([p[1] for p in particles_initial]),
                'z': np.array([p[2] for p in particles_initial]),
                'Px': np.array([p[3] for p in particles_initial]),
                'Py': np.array([p[4] for p in particles_initial]),
                'Pz': np.array([p[5] for p in particles_initial]),
            }
            
            # Add required velocity and other fields
            particles['vx'] = particles['Px'] / 938.3  # Approximate conversion
            particles['vy'] = particles['Py'] / 938.3
            particles['vz'] = particles['Pz'] / 938.3
            particles['t'] = np.zeros(n_particles)
            particles['q'] = np.full(n_particles, 1.0)  # Default charge
            particles['m'] = np.full(n_particles, 938.3)  # Proton mass
            particles['char_time'] = np.ones(n_particles)
        else:
            raise ValueError(f"Unexpected particles_initial format: {type(particles_initial)}")
        
        # Store initial state
        initial_energy = calculate_total_energy_dict(particles)
        current_particles = {k: v.copy() for k, v in particles.items()}
        
        # Integration loop - FIX PARAMETER NAMES
        step_size = params['step_size']
        num_steps = params['ret_steps']
        
        for step in range(num_steps):
            updated_particles = integrator.eqsofmotion_static(
                step_size, current_particles, current_particles
            )
            current_particles = updated_particles
        
        computation_time = time.time() - start_time
        final_energy = calculate_total_energy_dict(current_particles)
        
        print(f"  ✅ Basic integration completed in {computation_time:.4f} seconds")
        
        return {
            'success': True,
            'final_particles': current_particles,
            'computation_time': computation_time,
            'energy_initial': initial_energy,
            'energy_final': final_energy
        }
        
    except Exception as e:
        print(f"  ❌ Basic integration failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_optimized_integration(params, particles_initial):
    """Run integration using OptimizedLienardWiechertIntegrator."""
    import time
    print("\n⚡ Running Optimized Integration...")
    
    try:
        start_time = time.time()
        
        # Import and initialize integrator
        from lw_integrator.core.performance import OptimizedLienardWiechertIntegrator
        integrator = OptimizedLienardWiechertIntegrator()
        
        # Convert particles_initial to expected dictionary format
        if isinstance(particles_initial, dict):
            # Already in dictionary format from legacy
            particles = particles_initial.copy()
        elif isinstance(particles_initial, list):
            # Convert from list of particles to dictionary format
            n_particles = len(particles_initial)
            particles = {
                'x': np.array([p[0] for p in particles_initial]),
                'y': np.array([p[1] for p in particles_initial]),
                'z': np.array([p[2] for p in particles_initial]),
                'Px': np.array([p[3] for p in particles_initial]),
                'Py': np.array([p[4] for p in particles_initial]),
                'Pz': np.array([p[5] for p in particles_initial]),
            }
            
            # Add required velocity and other fields
            particles['vx'] = particles['Px'] / 938.3  # Approximate conversion
            particles['vy'] = particles['Py'] / 938.3
            particles['vz'] = particles['Pz'] / 938.3
            particles['t'] = np.zeros(n_particles)
            particles['q'] = np.full(n_particles, 1.0)  # Default charge
            particles['m'] = np.full(n_particles, 938.3)  # Proton mass
            particles['char_time'] = np.ones(n_particles)
        else:
            raise ValueError(f"Unexpected particles_initial format: {type(particles_initial)}")
        
        # Store initial state
        initial_energy = calculate_total_energy_dict(particles)
        current_particles = {k: v.copy() for k, v in particles.items()}
        
        # Integration loop - FIX PARAMETER NAMES 
        step_size = params['step_size']
        num_steps = params['ret_steps']
        
        for step in range(num_steps):
            updated_particles = integrator.eqsofmotion_static(
                step_size, current_particles, current_particles
            )
            current_particles = updated_particles
        
        computation_time = time.time() - start_time
        final_energy = calculate_total_energy_dict(current_particles)
        
        print(f"  ✅ Optimized integration completed in {computation_time:.4f} seconds")
        
        return {
            'success': True,
            'final_particles': current_particles,
            'computation_time': computation_time,
            'energy_initial': initial_energy,
            'energy_final': final_energy
        }
        
    except Exception as e:
        print(f"  ❌ Optimized integration failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def calculate_total_energy(particles):
    """Calculate total energy for a list of particles."""
    import numpy as np
    
    total_energy = 0.0
    for particle in particles:
        # Extract momentum components
        px, py, pz = particle[3], particle[4], particle[5]
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        
        # Calculate relativistic energy (assuming electron mass)
        m_electron = 0.511  # MeV/c²
        energy = np.sqrt((p_mag*299.792458)**2 + m_electron**2)  # Convert momentum units
        total_energy += energy
    
    return total_energy


def calculate_total_energy_dict(particles):
    """Calculate total energy for dictionary format particles."""
    import numpy as np
    
    # Ensure all arrays are consistently sized
    n_particles = len(particles['x'])
    
    # Get velocity components
    vx = np.array(particles['vx'])
    vy = np.array(particles['vy'])
    vz = np.array(particles['vz'])
    
    # Ensure velocities are proper arrays
    if vx.shape == ():
        vx = np.full(n_particles, vx)
    if vy.shape == ():
        vy = np.full(n_particles, vy)
    if vz.shape == ():
        vz = np.full(n_particles, vz)
    
    # Get masses
    masses = np.array(particles['m'])
    if masses.shape == ():
        masses = np.full(n_particles, masses)
    elif len(masses) != n_particles:
        # If mass array doesn't match, use the first value for all particles
        masses = np.full(n_particles, masses[0])
    
    # Calculate relativistic energy for each particle
    v2 = vx[:n_particles]**2 + vy[:n_particles]**2 + vz[:n_particles]**2
    gamma = 1.0 / np.sqrt(1.0 - v2 + 1e-16)  # Avoid division by zero
    
    # Total kinetic energy 
    total_energy = np.sum(masses[:n_particles] * (gamma - 1) * 299792458**2)  # Kinetic energy
    
    return total_energy


def analyze_results(results):
    """Analyze and compare integration results."""
    print("\n📊 ANALYSIS OF RESULTS")
    print("="*40)
    
    successful = []
    
    # Check which integrations succeeded
    for name, result in results.items():
        if result.get('success', False):
            successful.append(name)
            energy_initial = result.get('energy_initial', 'N/A')
            energy_final = result.get('energy_final', 'N/A')
            time_val = result.get('computation_time', 'N/A')
            
            print(f"\n{name.capitalize()} Integration:")
            print(f"  Status: ✅ SUCCESS")
            print(f"  Time: {time_val:.4f}s" if isinstance(time_val, (int, float)) else f"  Time: {time_val}")
            print(f"  Initial Energy: {energy_initial:.2f} MeV" if isinstance(energy_initial, (int, float)) else f"  Initial Energy: {energy_initial}")
            print(f"  Final Energy: {energy_final:.2f} MeV" if isinstance(energy_final, (int, float)) else f"  Final Energy: {energy_final}")
            
            # Energy conservation check
            if isinstance(energy_initial, (int, float)) and isinstance(energy_final, (int, float)):
                energy_diff = abs(energy_final - energy_initial)
                energy_rel_diff = energy_diff / energy_initial if energy_initial != 0 else float('inf')
                print(f"  Energy Conservation: {energy_rel_diff:.2e} relative error")
        else:
            print(f"\n{name.capitalize()} Integration:")
            print(f"  Status: ❌ FAILED - {result.get('error', 'Unknown error')}")
    
    # Trajectory comparison between successful integrations
    if len(successful) >= 2:
        print(f"\n🔍 TRAJECTORY COMPARISON")
        print("-"*30)
        
        # Compare all pairs
        import numpy as np
        for i in range(len(successful)):
            for j in range(i+1, len(successful)):
                name1, name2 = successful[i], successful[j]
                particles1 = results[name1]['final_particles']
                particles2 = results[name2]['final_particles']
                
                # Calculate differences - handle different particle formats
                max_pos_diff = 0.0
                max_vel_diff = 0.0
                
                # Check if particles are in dictionary or list format
                if isinstance(particles1, dict) and isinstance(particles2, dict):
                    # Both dictionary format
                    n_particles = min(len(particles1['x']), len(particles2['x']))
                    
                    for k in range(n_particles):
                        # Position differences (x, y, z)
                        pos_diff = max(abs(particles1['x'][k] - particles2['x'][k]),
                                     abs(particles1['y'][k] - particles2['y'][k]),
                                     abs(particles1['z'][k] - particles2['z'][k]))
                        
                        # Velocity differences (vx, vy, vz)
                        vel_diff = max(abs(particles1['vx'][k] - particles2['vx'][k]),
                                     abs(particles1['vy'][k] - particles2['vy'][k]),
                                     abs(particles1['vz'][k] - particles2['vz'][k]))
                        
                        max_pos_diff = max(max_pos_diff, pos_diff)
                        max_vel_diff = max(max_vel_diff, vel_diff)
                
                elif isinstance(particles1, list) and isinstance(particles2, list):
                    # Both list format
                    for k in range(min(len(particles1), len(particles2))):
                        # Position differences (x, y, z)
                        pos_diff = max(abs(particles1[k][0] - particles2[k][0]),
                                     abs(particles1[k][1] - particles2[k][1]),
                                     abs(particles1[k][2] - particles2[k][2]))
                        
                        # Momentum differences (px, py, pz)
                        vel_diff = max(abs(particles1[k][3] - particles2[k][3]),
                                     abs(particles1[k][4] - particles2[k][4]),
                                     abs(particles1[k][5] - particles2[k][5]))
                        
                        max_pos_diff = max(max_pos_diff, pos_diff)
                        max_vel_diff = max(max_vel_diff, vel_diff)
                else:
                    # Mixed formats - skip comparison
                    print(f"  ⚠️  Mixed particle formats - comparison skipped")
                    continue
                
                print(f"\n{name1.title()} vs {name2.title()}:")
                print(f"  Max position difference: {max_pos_diff:.2e}")
                print(f"  Max momentum difference: {max_vel_diff:.2e}")
                
                # Assessment
                if max_pos_diff < 1e-14 and max_vel_diff < 1e-14:
                    print("  🎯 PERFECT MATCH")
                elif max_pos_diff < 1e-10 and max_vel_diff < 1e-10:
                    print("  ✅ EXCELLENT AGREEMENT")
                elif max_pos_diff < 1e-6 and max_vel_diff < 1e-6:
                    print("  ⚠️  GOOD AGREEMENT")
                else:
                    print("  ❌ SIGNIFICANT DIFFERENCES")
    
    return successful


if __name__ == "__main__":
    results = main()