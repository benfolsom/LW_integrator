#!/usr/bin/env python3
"""
Direct Legacy vs Updated Trajectory Comparison

This module runs both legacy and updated integrators with identical parameters
and compares the final trajectories directly to identify any differences.

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

# Add legacy path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "legacy"))

from core.trajectory_integrator import LienardWiechertIntegrator

# Import legacy functions
try:
    from covariant_integrator_library import retarded_integrator3
    from bunch_inits import init_bunch
    from plotting_variables import calculate_plotting_variables

    LEGACY_AVAILABLE = True
except ImportError as e:
    print(f"Legacy imports failed: {e}")
    LEGACY_AVAILABLE = False


class DirectTrajectoryComparison:
    """Direct comparison of legacy vs updated trajectory calculations."""

    def __init__(self):
        self.updated_integrator = LienardWiechertIntegrator()

    def run_legacy_simulation(
        self, legacy_params: Dict[str, Any]
    ) -> Tuple[List, List, Dict, Dict]:
        """Run simulation using legacy integrator."""
        if not LEGACY_AVAILABLE:
            raise RuntimeError("Legacy code not available")

        # Create legacy bunches
        rider_params = legacy_params["rider"]
        driver_params = legacy_params["driver"]
        sim_params = legacy_params["simulation"]

        print("Creating legacy bunches...")
        init_rider, E_MeV_rest_rider = init_bunch(**rider_params)
        init_driver, E_MeV_rest_driver = init_bunch(**driver_params)

        print(f"Legacy rider energy: {E_MeV_rest_rider:.1f} MeV")
        print(f"Legacy driver energy: {E_MeV_rest_driver:.1f} MeV")

        # Debug legacy energy calculation
        print("\\n=== Legacy Energy Calculation Analysis ===")
        c_mmns = 299.792458
        rider_gamma = init_rider["Pt"][0] / (init_rider["m"] * c_mmns)
        rider_total_energy = rider_gamma * init_rider["m"] * 931.494
        print(
            f"Rider - Legacy E_MeV: {E_MeV_rest_rider:.1f}, Physics Total Energy: {rider_total_energy:.1f}"
        )

        driver_gamma = init_driver["Pt"][0] / (init_driver["m"] * c_mmns)
        driver_total_energy = driver_gamma * init_driver["m"] * 931.494
        print(
            f"Driver - Legacy E_MeV: {E_MeV_rest_driver:.1f}, Physics Total Energy: {driver_total_energy:.1f}"
        )

        # Run legacy integrator
        print("Running legacy integrator...")
        legacy_traj_rider, legacy_traj_driver = retarded_integrator3(
            sim_params["static_steps"],
            sim_params["ret_steps"],
            sim_params["step_size"],
            sim_params["wall_pos"],
            sim_params["aperture"],
            sim_params["sim_type"],
            init_rider,
            init_driver,
            sim_params["bunch_dist"],
            sim_params["cav_spacing"],
            sim_params["z_cutoff"],
        )

        return legacy_traj_rider, legacy_traj_driver, init_rider, init_driver

    def run_updated_simulation(
        self, legacy_params: Dict[str, Any]
    ) -> Tuple[List, List, Dict, Dict]:
        """Run simulation using updated integrator with legacy parameters."""

        # Convert legacy parameters to updated bunches
        rider_params = legacy_params["rider"]
        driver_params = legacy_params["driver"]
        sim_params = legacy_params["simulation"]

        print("Creating updated bunches from legacy parameters...")
        updated_rider = self._create_updated_bunch_from_legacy(**rider_params)
        updated_driver = self._create_updated_bunch_from_legacy(**driver_params)

        # Calculate energies for comparison
        rider_energy = self._calculate_energy_mev(updated_rider)
        driver_energy = self._calculate_energy_mev(updated_driver)
        print(f"Updated rider energy: {rider_energy:.1f} MeV")
        print(f"Updated driver energy: {driver_energy:.1f} MeV")

        # Debug: Check momentum and gamma values
        print(
            f"Updated rider: Pt={updated_rider['Pt'][0]:.2e}, gamma={updated_rider['gamma'][0]:.2f}, mass={updated_rider['m'][0]:.6f}"
        )
        print(
            f"Updated driver: Pt={updated_driver['Pt'][0]:.2e}, gamma={updated_driver['gamma'][0]:.2f}, mass={updated_driver['m'][0]:.6f}"
        )

        # Run updated integrator
        print("Running updated integrator...")
        updated_traj_rider, updated_traj_driver = (
            self.updated_integrator.integrate_retarded_fields(
                static_steps=sim_params["static_steps"],
                ret_steps=sim_params["ret_steps"],
                h_step=sim_params["step_size"],
                wall_Z=sim_params["wall_pos"],
                apt_R=sim_params["aperture"],
                sim_type=sim_params["sim_type"],
                init_rider=updated_rider,
                init_driver=updated_driver,
                bunch_dist=sim_params["bunch_dist"],
                z_cutoff=sim_params["z_cutoff"],
            )
        )

        return updated_traj_rider, updated_traj_driver, updated_rider, updated_driver

    def _create_updated_bunch_from_legacy(
        self,
        starting_distance: float,
        transv_mom: float,
        starting_Pz: float,
        stripped_ions: float,
        m_particle: float,
        transv_dist: float,
        pcount: int,
        charge_sign: float,
    ) -> Dict[str, np.ndarray]:
        """Create updated bunch using exact legacy formulation."""

        c_mmns = 299.792458  # mm/ns
        macro_pop = 1

        # Legacy calculations
        mass = m_particle * macro_pop
        q = charge_sign * 1.178734e-5 * stripped_ions * macro_pop
        char_time = 2 / 3 * q**2 / (mass * c_mmns**3)

        # Use same random seed for reproducible comparison
        np.random.seed(42)

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
            "x": x,
            "y": y,
            "z": z,
            "t": t,
            "Px": Px,
            "Py": Py,
            "Pz": Pz,
            "Pt": Pt,
            "bx": bx,
            "by": by,
            "bz": bz,
            "bdotx": bdotx,
            "bdoty": bdoty,
            "bdotz": bdotz,
            "gamma": gamma,
            "q": np.full(pcount, q),  # Updated integrator needs arrays
            "m": np.full(pcount, mass),  # Updated integrator needs arrays
            "char_time": np.full(pcount, char_time),  # Updated integrator needs arrays
        }

    def _calculate_energy_mev(self, bunch: Dict[str, np.ndarray]) -> float:
        """Calculate energy in MeV from bunch data."""
        mass_amu = bunch["m"] if isinstance(bunch["m"], (int, float)) else bunch["m"][0]
        gamma = bunch["gamma"][0]
        rest_energy_mev = mass_amu * 931.494  # MeV
        return gamma * rest_energy_mev

    def compare_trajectories(
        self,
        legacy_traj: List[Dict],
        modern_traj: List[Dict],
        particle_name: str = "particle",
    ) -> Dict[str, Any]:
        """Compare legacy and updated trajectories step by step."""

        print(f"\\n=== {particle_name.title()} Trajectory Comparison ===")
        print(f"Legacy steps: {len(legacy_traj)}")
        print(f"Updated steps: {len(modern_traj)}")

        # Ensure same number of steps for comparison
        min_steps = min(len(legacy_traj), len(modern_traj))

        differences = {
            "positions": [],
            "energies": [],
            "momenta": [],
            "relative_energy_diff": [],
            "step_numbers": [],
        }

        for step in range(min_steps):
            legacy_step = legacy_traj[step]
            updated_step = modern_traj[step]

            # Compare first particle in each bunch
            # Positions
            legacy_z = legacy_step["z"][0]
            updated_z = updated_step["z"][0]
            pos_diff = abs(legacy_z - updated_z)

            # Energies (convert both to MeV)
            # Handle legacy vs updated data structure differences
            legacy_mass = (
                legacy_step["m"]
                if isinstance(legacy_step["m"], (int, float))
                else legacy_step["m"][0]
            )
            updated_mass = (
                updated_step["m"][0]
                if "m" in updated_step
                else updated_step.get("mass", [1.0])[0]
            )

            c_mmns = 299.792458
            legacy_gamma = legacy_step["Pt"][0] / (legacy_mass * c_mmns)
            updated_gamma = updated_step["Pt"][0] / (updated_mass * c_mmns)

            legacy_energy = legacy_gamma * legacy_mass * 931.494
            updated_energy = updated_gamma * updated_mass * 931.494
            energy_diff = abs(legacy_energy - updated_energy)
            rel_energy_diff = (
                energy_diff / legacy_energy * 100 if legacy_energy > 0 else 0
            )

            # Momenta
            legacy_pt = legacy_step["Pt"][0]
            updated_pt = updated_step["Pt"][0]
            momentum_diff = abs(legacy_pt - updated_pt)

            differences["positions"].append(pos_diff)
            differences["energies"].append(energy_diff)
            differences["momenta"].append(momentum_diff)
            differences["relative_energy_diff"].append(rel_energy_diff)
            differences["step_numbers"].append(step)

            if step % 10 == 0 or step < 5:  # Print every 10th step or first 5
                print(
                    f"Step {step:3d}: pos_diff={pos_diff:.6f} mm, "
                    f"energy_diff={energy_diff:.6f} MeV ({rel_energy_diff:.2e}%), "
                    f"momentum_diff={momentum_diff:.6f}"
                )

        # Summary statistics
        max_pos_diff = max(differences["positions"])
        max_energy_diff = max(differences["energies"])
        max_rel_energy_diff = max(differences["relative_energy_diff"])

        print(f"\\nSummary for {particle_name}:")
        print(f"  Max position difference: {max_pos_diff:.6f} mm")
        print(f"  Max energy difference: {max_energy_diff:.6f} MeV")
        print(f"  Max relative energy difference: {max_rel_energy_diff:.2e}%")

        return differences

    def create_comparison_plots(
        self,
        legacy_rider: List[Dict],
        updated_rider: List[Dict],
        legacy_driver: List[Dict],
        updated_driver: List[Dict],
        rider_diffs: Dict[str, Any],
        driver_diffs: Dict[str, Any],
    ):
        """Create comprehensive comparison plots."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Legacy vs Updated Trajectory Comparison", fontsize=16)

        # Plot 1: Energy trajectories
        ax1.set_title("Energy Trajectories")

        # Extract energies
        c_mmns = 299.792458
        steps = min(len(legacy_rider), len(updated_rider))

        legacy_rider_energies = []
        updated_rider_energies = []
        positions = []

        for step in range(steps):
            # Legacy energy
            legacy_mass = (
                legacy_rider[step]["m"]
                if isinstance(legacy_rider[step]["m"], (int, float))
                else legacy_rider[step]["m"][0]
            )
            legacy_gamma = legacy_rider[step]["Pt"][0] / (legacy_mass * c_mmns)
            legacy_energy = legacy_gamma * legacy_mass * 931.494
            legacy_rider_energies.append(legacy_energy)

            # Updated energy
            updated_mass = (
                updated_rider[step]["m"][0] if "m" in updated_rider[step] else 1.007
            )
            updated_gamma = updated_rider[step]["Pt"][0] / (updated_mass * c_mmns)
            updated_energy = updated_gamma * updated_mass * 931.494
            updated_rider_energies.append(updated_energy)

            positions.append(legacy_rider[step]["z"][0])

        ax1.plot(
            positions, legacy_rider_energies, "b-", label="Legacy Rider", linewidth=2
        )
        ax1.plot(
            positions, updated_rider_energies, "r--", label="Updated Rider", linewidth=2
        )
        ax1.set_xlabel("Position (mm)")
        ax1.set_ylabel("Energy (MeV)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Energy differences
        ax2.set_title("Energy Differences")
        ax2.plot(
            rider_diffs["step_numbers"],
            rider_diffs["relative_energy_diff"],
            "g-",
            label="Rider",
        )
        ax2.plot(
            driver_diffs["step_numbers"],
            driver_diffs["relative_energy_diff"],
            "m-",
            label="Driver",
        )
        ax2.set_xlabel("Step Number")
        ax2.set_ylabel("Relative Energy Difference (%)")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Position differences
        ax3.set_title("Position Differences")
        ax3.plot(
            rider_diffs["step_numbers"], rider_diffs["positions"], "g-", label="Rider"
        )
        ax3.plot(
            driver_diffs["step_numbers"],
            driver_diffs["positions"],
            "m-",
            label="Driver",
        )
        ax3.set_xlabel("Step Number")
        ax3.set_ylabel("Position Difference (mm)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Momentum differences
        ax4.set_title("Momentum Differences")
        ax4.plot(
            rider_diffs["step_numbers"], rider_diffs["momenta"], "g-", label="Rider"
        )
        ax4.plot(
            driver_diffs["step_numbers"], driver_diffs["momenta"], "m-", label="Driver"
        )
        ax4.set_xlabel("Step Number")
        ax4.set_ylabel("Momentum Difference (amu*mm/ns)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "legacy_modern_trajectory_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.show()


def create_two_particle_demo_parameters() -> Dict[str, Any]:
    """Create parameters matching the legacy two-particle demo exactly."""

    return {
        "rider": {
            "starting_distance": 1e-6,
            "transv_mom": 0.0,
            "starting_Pz": 1.01e6,
            "stripped_ions": 1.0,
            "m_particle": 1.007319468,  # proton
            "transv_dist": 1e-4,
            "pcount": 10,
            "charge_sign": -1.0,
        },
        "driver": {
            "starting_distance": 100.0,
            "transv_mom": 0.0,
            "starting_Pz": -1.01e6 / 207.2 * 1.007319468,  # Legacy formula
            "stripped_ions": 54.0,
            "m_particle": 207.2,  # Lead
            "transv_dist": -1e-4,
            "pcount": 10,
            "charge_sign": 1.0,
        },
        "simulation": {
            "static_steps": 1,
            "ret_steps": 25,
            "step_size": 2e-6,
            "wall_pos": 1e5,
            "aperture": 1e5,
            "sim_type": 2,
            "bunch_dist": 1e5,
            "cav_spacing": 1e5,
            "z_cutoff": 0,
        },
    }


def main():
    """Run direct trajectory comparison between legacy and updated implementations."""

    print("=== Direct Legacy vs Updated Trajectory Comparison ===\\n")

    if not LEGACY_AVAILABLE:
        print("❌ Legacy code not available. Cannot run comparison.")
        print("Make sure legacy modules are in the Python path.")
        return

    comparison = DirectTrajectoryComparison()
    params = create_two_particle_demo_parameters()

    try:
        # Run legacy simulation
        print("🔧 Running Legacy Simulation...")
        legacy_rider, legacy_driver, legacy_init_rider, legacy_init_driver = (
            comparison.run_legacy_simulation(params)
        )

        # Run updated simulation
        print("\\n🚀 Running Updated Simulation...")
        updated_rider, updated_driver, updated_init_rider, updated_init_driver = (
            comparison.run_updated_simulation(params)
        )

        # Compare trajectories
        print("\\n📊 Comparing Trajectories...")
        rider_differences = comparison.compare_trajectories(
            legacy_rider, updated_rider, "rider"
        )
        driver_differences = comparison.compare_trajectories(
            legacy_driver, updated_driver, "driver"
        )

        # Create plots
        print("\\n📈 Creating Comparison Plots...")
        comparison.create_comparison_plots(
            legacy_rider,
            updated_rider,
            legacy_driver,
            updated_driver,
            rider_differences,
            driver_differences,
        )

        # Final summary
        print("\\n=== Final Summary ===")
        print(f"✅ Legacy simulation: {len(legacy_rider)} steps completed")
        print(f"✅ Updated simulation: {len(updated_rider)} steps completed")

        # Check if differences are significant
        max_rider_energy_diff = max(rider_differences["relative_energy_diff"])
        max_driver_energy_diff = max(driver_differences["relative_energy_diff"])

        if max_rider_energy_diff < 1e-10 and max_driver_energy_diff < 1e-10:
            print("🎯 PERFECT AGREEMENT: Differences below numerical precision")
        elif max_rider_energy_diff < 1e-6 and max_driver_energy_diff < 1e-6:
            print("✅ EXCELLENT AGREEMENT: Differences below 0.0001%")
        else:
            print("⚠️  DIFFERENCES DETECTED:")
            print(f"   Max rider energy difference: {max_rider_energy_diff:.2e}%")
            print(f"   Max driver energy difference: {max_driver_energy_diff:.2e}%")

        print(
            "\\n📁 Comparison plots saved as: legacy_modern_trajectory_comparison.png"
        )

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
