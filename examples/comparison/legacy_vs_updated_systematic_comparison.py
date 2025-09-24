#!/usr/bin/env python3
"""
Systematic Legacy vs Updated Code Comparison

This module creates systematic comparisons between legacy and updated
implementations, starting with simple cases and building up to the
full two-particle demo.

Author: Ben Folsom
Date: 2025-09-19
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.trajectory_integrator import LienardWiechertIntegrator
from physics.particle_initialization import ParticleSpecies
from physics.standardized_input_creation import StandardizedInputCreator
from physics.constants import C_MMNS


class LegacyUpdatedComparison:
    """Systematic comparison between legacy and updated implementations."""

    def __init__(self):
        self.integrator = LienardWiechertIntegrator()
        self.creator = StandardizedInputCreator()

    def create_legacy_compatible_bunch(
        self,
        starting_distance: float,
        transv_mom: float,
        starting_Pz: float,
        stripped_ions: float,
        m_particle: float,
        transv_dist: float,
        pcount: int,
        charge_sign: float,
        macro_pop: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Create bunch using exact legacy method for comparison.

        This replicates the legacy bunch_inits.py logic exactly.
        """
        c_mmns = 299.792458  # mm/ns

        # Legacy macroparticle setup
        mass = m_particle * macro_pop  # macroparticle mass
        q = charge_sign * 1.178734E-5 * stripped_ions * macro_pop
        char_time = 2/3 * q**2 / (mass * c_mmns**3)

        # Legacy momentum initialization
        Px = np.random.uniform(-transv_mom, transv_mom, pcount) * mass
        Py = np.random.uniform(-transv_mom, transv_mom, pcount) * mass
        Pz = np.random.uniform(starting_Pz, starting_Pz + 0.1, pcount) * mass
        Pt = np.sqrt(Px**2 + Py**2 + Pz**2 + mass**2 * c_mmns**2)
        gamma = Pt / (mass * c_mmns)

        # Legacy velocity initialization
        bx = Px / (gamma * mass * c_mmns)
        by = Py / (gamma * mass * c_mmns)
        bz = Pz / (gamma * mass * c_mmns)

        # Legacy position initialization
        x = np.random.uniform(transv_dist, transv_dist, pcount)
        y = np.random.uniform(transv_dist, transv_dist, pcount)
        z = np.random.uniform(starting_distance, starting_distance, pcount)
        t = np.zeros(pcount)

        # Legacy acceleration initialization
        bdotx = np.zeros(pcount)
        bdoty = np.zeros(pcount)
        bdotz = np.zeros(pcount)

        return {
            'x': x, 'y': y, 'z': z, 't': t,
            'Px': Px, 'Py': Py, 'Pz': Pz, 'Pt': Pt,
            'bx': bx, 'by': by, 'bz': bz,
            'bdotx': bdotx, 'bdoty': bdoty, 'bdotz': bdotz,
            'gamma': gamma,
            'q': np.full(pcount, q),  # Ensure array
            'm': np.full(pcount, mass),  # Ensure array
            'char_time': np.full(pcount, char_time)  # Ensure array
        }

    def create_updated_equivalent_bunch(
        self,
        starting_distance: float,
        transv_mom: float,
        starting_Pz: float,
        stripped_ions: int,
        m_particle: float,
        transv_dist: float,
        pcount: int,
        charge_sign: int
    ) -> Dict[str, np.ndarray]:
        """
        """
        Create equivalent bunch using updated standardized methods.
        """
        # Convert legacy parameters to updated ParticleSpecies
        species = ParticleSpecies.ion(
            mass_amu=m_particle * macro_pop,
            charge_state=charge_sign * stripped_ions * macro_pop
        )

        # Calculate energy from legacy Pz
        mass = m_particle * macro_pop
        momentum_magnitude = starting_Pz * mass  # amu*mm/ns
        Pt = np.sqrt(momentum_magnitude**2 + (mass * C_MMNS)**2)
        gamma = Pt / (mass * C_MMNS)
        energy_mev = gamma * mass * 931.494  # Convert to MeV

        # Use standardized creation for single bunch
        from physics.particle_initialization import create_particle_bunch
        from physics.standardized_input_creation import _add_legacy_fields

        bunch = create_particle_bunch(
            n_particles=pcount,
            species=species,
            energy_mev=energy_mev,
            position=(transv_dist, 0.0, starting_distance),
            momentum_direction=(transv_mom, 0.0, starting_Pz),
            bunch_size=(0.0, 0.0),  # Point particles for comparison
            distribution="uniform"
        )

        # Add legacy fields
        bunch = _add_legacy_fields(bunch, species)

        return bunch

    def compare_bunch_initialization(self) -> Dict[str, Any]:
        """
        Compare legacy vs updated bunch initialization with exact legacy parameters.
        """
        print("=== Bunch Initialization Comparison ===")

        # Legacy two-particle demo parameters
        legacy_params = {
            'starting_distance': 1e-6,
            'transv_mom': 0.0,
            'starting_Pz': 1.01e6,
            'stripped_ions': 1.0,
            'm_particle': 1.007319468,  # proton amu
            'transv_dist': 1e-4,
            'pcount': 10,
            'charge_sign': -1.0,
            'macro_pop': 1.0
        }

        # Fix random seed for reproducibility
        np.random.seed(42)
        legacy_bunch = self.create_legacy_compatible_bunch(**legacy_params)

        np.random.seed(42)
        updated_bunch = self.create_updated_equivalent_bunch(**legacy_params)

        comparison = {}

        # Compare key quantities
        for key in ['Pt', 'gamma', 'q', 'm', 'char_time']:
            if key in legacy_bunch and key in updated_bunch:
                # Handle array vs scalar values
                legacy_val = legacy_bunch[key][0] if hasattr(legacy_bunch[key], '__getitem__') else legacy_bunch[key]
                updated_val = updated_bunch[key][0] if hasattr(updated_bunch[key], '__getitem__') else updated_bunch[key]

                diff = abs(legacy_val - updated_val) / abs(legacy_val) * 100
                comparison[key] = {
                    'legacy': legacy_val,
                    'modern': updated_val,
                    'diff_percent': diff
                }
                print(f"{key:10s}: Legacy={legacy_val:.6e}, Modern={updated_val:.6e}, Diff={diff:.3f}%")

        return comparison

    def test_simple_two_particle_case(self) -> Tuple[List, List, List, List]:
        """
        Test simple two-particle case with heavy, highly charged driver.

        This follows your suggestion: heavy driver with high charge and negligible Pz.
        """
        print("\\n=== Simple Two-Particle Test Case ===")

        # Create simple test case with your specifications
        # Heavy, highly charged driver with negligible Pz
        rider_params = {
            'starting_distance': 1e-6,
            'transv_mom': 0.0,
            'starting_Pz': 1.01e6,  # High momentum rider
            'stripped_ions': 1.0,
            'm_particle': 1.007319468,  # proton
            'transv_dist': 1e-4,
            'pcount': 1,  # Single particle for simplicity
            'charge_sign': -1.0
        }

        driver_params = {
            'starting_distance': 100.0,
            'transv_mom': 0.0,
            'starting_Pz': 1e-10,  # Negligible Pz as requested
            'stripped_ions': 54.0,  # Highly charged (Lead)
            'm_particle': 207.2,  # Heavy (Lead)
            'transv_dist': -1e-4,  # Opposite side
            'pcount': 1,  # Single particle
            'charge_sign': 1.0
        }

        # Create bunches
        np.random.seed(42)
        legacy_rider = self.create_legacy_compatible_bunch(**rider_params)
        legacy_driver = self.create_legacy_compatible_bunch(**driver_params)

        # Run legacy-style simulation
        def get_val(data, key, idx=0):
            """Helper to get value whether it's array or scalar"""
            val = data[key]
            return val[idx] if hasattr(val, '__getitem__') and len(val) > idx else val

        print(f"Rider: q={get_val(legacy_rider, 'q'):.2e}, m={get_val(legacy_rider, 'm'):.3f}, Pz={get_val(legacy_rider, 'Pz'):.2e}")
        print(f"Driver: q={get_val(legacy_driver, 'q'):.2e}, m={get_val(legacy_driver, 'm'):.3f}, Pz={get_val(legacy_driver, 'Pz'):.2e}")

        # Short simulation with fine steps
        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=1,
            ret_steps=25,  # Short for debugging
            h_step=2e-6,
            wall_Z=1e5,
            apt_R=1e5,
            sim_type=2,  # Bunch-bunch
            init_rider=legacy_rider,
            init_driver=legacy_driver,
            bunch_dist=1e5,
            z_cutoff=0
        )

        print(f"Simulation completed: {len(trajectory_rider)} steps")
        print(f"Rider initial: z={get_val(trajectory_rider[0], 'z'):.3f}, Pt={get_val(trajectory_rider[0], 'Pt'):.6f}")
        print(f"Rider final: z={get_val(trajectory_rider[-1], 'z'):.3f}, Pt={get_val(trajectory_rider[-1], 'Pt'):.6f}")
        print(f"Driver initial: z={get_val(trajectory_driver[0], 'z'):.3f}, Pt={get_val(trajectory_driver[0], 'Pt'):.6f}")
        print(f"Driver final: z={get_val(trajectory_driver[-1], 'z'):.3f}, Pt={get_val(trajectory_driver[-1], 'Pt'):.6f}")

        return trajectory_rider, trajectory_driver, legacy_rider, legacy_driver

    def reproduce_two_particle_demo(self) -> Dict[str, Any]:
        """
        Reproduce the legacy two-particle demo with exact parameters.
        """
        print("\\n=== Two-Particle Demo Reproduction ===")

        # Exact legacy parameters from two_particle_demo_main.ipynb
        rider_params = {
            'starting_distance': 1e-6,
            'transv_mom': 0.0,
            'starting_Pz': 1.01e6,
            'stripped_ions': 1.0,
            'm_particle': 1.007319468,  # proton
            'transv_dist': 1e-4,
            'pcount': 10,
            'charge_sign': -1.0
        }

        driver_params = {
            'starting_distance': 100.0,
            'transv_mom': 0.0,
            'starting_Pz': -1.01e6 / 207.2 * 1.007319468,  # Exact legacy formula
            'stripped_ions': 54.0,
            'm_particle': 207.2,  # Lead
            'transv_dist': -1e-4,
            'pcount': 10,
            'charge_sign': 1.0
        }

        # Create legacy bunches
        np.random.seed(42)  # For reproducibility
        legacy_rider = self.create_legacy_compatible_bunch(**rider_params)
        legacy_driver = self.create_legacy_compatible_bunch(**driver_params)

        print(f"Legacy rider energy: {legacy_rider['Pt'][0] / (legacy_rider['m'][0] * C_MMNS) * legacy_rider['m'][0] * 931.494:.1f} MeV")
        print(f"Legacy driver energy: {legacy_driver['Pt'][0] / (legacy_driver['m'][0] * C_MMNS) * legacy_driver['m'][0] * 931.494:.1f} MeV")

        # Run two-stage simulation like in legacy demo
        print("\\nRunning coarse simulation...")
        traj_pre, drv_traj_pre = self.integrator.integrate_retarded_fields(
            static_steps=1,
            ret_steps=25,
            h_step=2e-6,
            wall_Z=1e5,
            apt_R=1e5,
            sim_type=2,
            init_rider=legacy_rider,
            init_driver=legacy_driver,
            bunch_dist=1e5,
            z_cutoff=0
        )

        print("\\nRunning fine simulation...")
        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=1,
            ret_steps=100,  # Reduced for testing
            h_step=3e-6,
            wall_Z=1e5,
            apt_R=1e5,
            sim_type=2,
            init_rider=traj_pre[-1],
            init_driver=drv_traj_pre[-1],
            bunch_dist=1e5,
            z_cutoff=0
        )

        return {
            'trajectory_rider': trajectory_rider,
            'trajectory_driver': trajectory_driver,
            'legacy_rider': legacy_rider,
            'legacy_driver': legacy_driver,
            'coarse_rider': traj_pre,
            'coarse_driver': drv_traj_pre
        }

    def analyze_macroparticle_effects(self, macro_populations: List[float]) -> Dict[str, Any]:
        """
        Analyze how macroparticle population affects results.
        """
        print("\\n=== Macroparticle Population Analysis ===")

        base_params = {
            'starting_distance': 1e-6,
            'transv_mom': 0.0,
            'starting_Pz': 1.01e6,
            'stripped_ions': 1.0,
            'm_particle': 1.007319468,
            'transv_dist': 1e-4,
            'pcount': 1,
            'charge_sign': -1.0
        }

        results = {}

        for macro_pop in macro_populations:
            print(f"\\nTesting macro_pop = {macro_pop}")
            params = base_params.copy()
            params['macro_pop'] = macro_pop

            bunch = self.create_legacy_compatible_bunch(**params)

            # Quick simulation (use sim_type=2 with dummy driver)
            dummy_driver = bunch.copy()
            dummy_driver['z'] = np.array([1e6])  # Far away
            dummy_driver['Pz'] = np.array([1e-10])  # Nearly stationary

            traj, _ = self.integrator.integrate_retarded_fields(
                static_steps=1,
                ret_steps=10,
                h_step=2e-6,
                wall_Z=1e5,
                apt_R=1e5,
                sim_type=2,  # Bunch-bunch with dummy driver
                init_rider=bunch,
                init_driver=dummy_driver,
                bunch_dist=1e5,
                z_cutoff=0
            )

            results[macro_pop] = {
                'initial_energy': traj[0]['Pt'][0] / (traj[0]['m'][0] * C_MMNS) * traj[0]['m'][0] * 931.494,
                'mass': bunch['m'][0],
                'charge': bunch['q'][0],
                'char_time': bunch['char_time'][0]
            }

            print(f"  Mass: {bunch['m'][0]:.6f} amu")
            print(f"  Charge: {bunch['q'][0]:.6e}")
            print(f"  Energy: {results[macro_pop]['initial_energy']:.1f} MeV")

        return results


def main():
    """Run systematic comparison between legacy and updated code."""

    print("=== Systematic Legacy vs Updated Code Comparison ===\\n")

    comparison = LegacyUpdatedComparison()

    # Step 1: Compare bunch initialization
    print("Step 1: Bunch initialization comparison")
    init_comparison = comparison.compare_bunch_initialization()

    # Step 2: Simple two-particle case with heavy, charged driver
    print("\\nStep 2: Simple two-particle case")
    rider_traj, driver_traj, rider_init, driver_init = comparison.test_simple_two_particle_case()

    # Step 3: Reproduce full two-particle demo
    print("\\nStep 3: Two-particle demo reproduction")
    demo_results = comparison.reproduce_two_particle_demo()

    # Step 4: Macroparticle analysis
    print("\\nStep 4: Macroparticle population analysis")
    macro_results = comparison.analyze_macroparticle_effects([1.0, 10.0, 100.0, 1000.0])

    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Legacy vs Updated Code Comparison', fontsize=16)

    # Plot 1: Simple case rider trajectory
    if rider_traj:
        positions = [step['z'][0] for step in rider_traj]
        energies = [step['Pt'][0] / (step['m'][0] * C_MMNS) * step['m'][0] * 931.494 for step in rider_traj]
        ax1.plot(positions, energies, 'b-o', markersize=3)
        ax1.set_title('Simple Case: Rider Energy vs Position')
        ax1.set_xlabel('Position (mm)')
        ax1.set_ylabel('Energy (MeV)')
        ax1.grid(True, alpha=0.3)

    # Plot 2: Demo case energy evolution
    if 'trajectory_rider' in demo_results:
        traj = demo_results['trajectory_rider']
        positions = [step['z'][0] for step in traj]
        energies = [step['Pt'][0] / (step['m'][0] * C_MMNS) * step['m'][0] * 931.494 for step in traj]
        energy_changes = [(e - energies[0])/energies[0] * 100 for e in energies]
        ax2.plot(positions, energy_changes, 'r-o', markersize=2)
        ax2.set_title('Demo Case: Relative Energy Change')
        ax2.set_xlabel('Position (mm)')
        ax2.set_ylabel('ΔE/E₀ (%)')
        ax2.grid(True, alpha=0.3)

    # Plot 3: Macroparticle scaling
    if macro_results:
        macro_pops = list(macro_results.keys())
        masses = [macro_results[mp]['mass'] for mp in macro_pops]
        charges = [abs(macro_results[mp]['charge']) for mp in macro_pops]
        ax3.loglog(macro_pops, masses, 'go-', label='Mass')
        ax3_twin = ax3.twinx()
        ax3_twin.loglog(macro_pops, charges, 'ro-', label='Charge')
        ax3.set_title('Macroparticle Scaling')
        ax3.set_xlabel('Macroparticle Population')
        ax3.set_ylabel('Mass (amu)', color='g')
        ax3_twin.set_ylabel('Charge (Gaussian)', color='r')
        ax3.grid(True, alpha=0.3)

    # Plot 4: Initialization comparison
    if init_comparison:
        keys = list(init_comparison.keys())
        diffs = [init_comparison[k]['diff_percent'] for k in keys]
        ax4.bar(keys, diffs)
        ax4.set_title('Legacy vs Updated Initialization Differences')
        ax4.set_ylabel('Difference (%)')
        ax4.set_yscale('log')
        ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('legacy_modern_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\\n=== Summary ===")
    print(f"Initialization comparison: {'PASS' if all(init_comparison[k]['diff_percent'] < 1e-10 for k in init_comparison) else 'DIFFERENCES FOUND'}")
    print(f"Simple case completed with {len(rider_traj) if rider_traj else 0} steps")
    print(f"Demo case completed with {len(demo_results.get('trajectory_rider', [])) if demo_results else 0} steps")
    print(f"Macroparticle analysis: {len(macro_results)} populations tested")
    print("\\nComparison plots saved as: legacy_modern_comparison.png")


if __name__ == "__main__":
    main()
