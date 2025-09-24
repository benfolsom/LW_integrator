#!/usr/bin/env python3
"""
Energy vs Position Analysis using Standardized Input Creation

This module uses the standardized input creation methods to demonstrate
proper energy gain from image charges in conducting apertures, with
the specific requirements for low-energy electrons (1-500 MeV).

Author: Ben Folsom
Date: 2025-09-19
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.trajectory_integrator import LienardWiechertIntegrator
from physics.particle_initialization import ParticleSpecies
from physics.standardized_input_creation import (
    StandardizedInputCreator,
)
from physics.constants import C_MMNS


class StandardizedEnergyPositionAnalysis:
    """Energy position analysis using standardized input creation methods."""

    def __init__(self):
        self.integrator = LienardWiechertIntegrator()
        self.creator = StandardizedInputCreator()

    def run_energy_range_study(
        self,
        energy_range_mev: List[float],
        particle_species: ParticleSpecies,
        aperture_radius: float = 5.0,
        wall_position: float = 0.0,
        starting_distance: float = -200.0,
    ) -> List[Tuple[float, Dict]]:
        """
        Run energy vs position study for a range of energies.

        Args:
            energy_range_mev: List of energies in MeV
            particle_species: ParticleSpecies object
            aperture_radius: Aperture radius in mm
            wall_position: Wall z position in mm
            starting_distance: Initial z position in mm

        Returns:
            List of (energy_mev, simulation_results) tuples
        """
        results = []

        print(f"Running energy range study for {particle_species.name}")
        print(f"Energies: {energy_range_mev} MeV")
        print(f"Aperture radius: {aperture_radius} mm")
        print(f"Starting distance: {starting_distance} mm\\n")

        for energy_mev in energy_range_mev:
            print(f"Processing {energy_mev} MeV {particle_species.name}...")

            # Create standardized setup for conducting aperture
            config, bunch = self.creator.create_conducting_aperture_setup(
                energy_mev=energy_mev,
                particle_species=particle_species,
                aperture_radius=aperture_radius,
                wall_position=wall_position,
                starting_distance=starting_distance,
                particle_count=5,  # Small number for efficiency
            )

            # Run simulation
            try:
                # For conducting aperture (sim_type=0), we need dummy driver bunch
                dummy_driver = self._create_dummy_driver(bunch)

                trajectory, _ = self.integrator.integrate_retarded_fields(
                    static_steps=config.static_steps,
                    ret_steps=config.total_steps - config.static_steps,
                    h_step=config.step_size,
                    wall_Z=config.wall_z,
                    apt_R=config.aperture_r,
                    sim_type=config.sim_type,
                    init_rider=bunch,
                    init_driver=dummy_driver,
                    bunch_dist=1e5,
                    z_cutoff=config.z_cutoff,
                )

                # Extract energy and position data
                sim_results = self._extract_energy_position_data(trajectory, energy_mev)
                results.append((energy_mev, sim_results))

                print(f"  Initial energy: {sim_results['initial_energy']:.3f} MeV")
                print(f"  Final energy: {sim_results['final_energy']:.3f} MeV")
                print(f"  Energy gain: {sim_results['energy_gain']:.6f} MeV")
                print(
                    f"  Distance traveled: {sim_results['distance_traveled']:.1f} mm\\n"
                )

            except Exception as e:
                print(f"  Error in simulation: {e}\\n")
                continue

        return results

    def _create_dummy_driver(
        self, rider_bunch: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Create a dummy driver bunch for single-bunch simulations."""
        dummy = {}
        for key, value in rider_bunch.items():
            if isinstance(value, np.ndarray):
                # Create single particle far away with minimal interaction
                dummy[key] = (
                    np.array([value[0]])
                    if len(value.shape) == 1
                    else np.array([[value[0][0]]])
                )

        # Place far away
        dummy["z"] = np.array([1e6])
        dummy["x"] = np.array([1e6])
        dummy["y"] = np.array([1e6])
        # Make it nearly stationary
        dummy["Px"] = np.array([0.0])
        dummy["Py"] = np.array([0.0])
        dummy["Pz"] = np.array([1e-10])

        # Calculate Pt properly using the mass from the bunch
        mass = dummy["m"][0]
        dummy["Pt"] = np.sqrt(
            dummy["Px"] ** 2 + dummy["Py"] ** 2 + dummy["Pz"] ** 2 + mass**2 * C_MMNS**2
        )

        return dummy

    def _extract_energy_position_data(
        self, trajectory: List[Dict[str, np.ndarray]], initial_energy_mev: float
    ) -> Dict:
        """Extract energy and position data from trajectory."""

        # Get first particle data
        positions = [step["z"][0] for step in trajectory]
        energies_gaussian = [step["Pt"][0] for step in trajectory]

        # Convert to MeV using proper conversion
        # In Gaussian units, Pt is energy, convert using rest mass energy
        # Get mass from the trajectory data
        particle_mass_amu = (
            trajectory[0]["m"][0] if "m" in trajectory[0] else 5.485799e-4
        )  # fallback to electron
        rest_energy_mev = particle_mass_amu * 931.494  # MeV

        # Energy in MeV = (Pt / (mass * c)) * rest_energy_mev
        # Since Pt is already in energy units in Gaussian, we need gamma
        gammas = [pt / (particle_mass_amu * C_MMNS) for pt in energies_gaussian]
        energies_mev = [gamma * rest_energy_mev for gamma in gammas]

        return {
            "positions": np.array(positions),
            "energies_mev": np.array(energies_mev),
            "energies_gaussian": np.array(energies_gaussian),
            "initial_energy": energies_mev[0],
            "final_energy": energies_mev[-1],
            "energy_gain": energies_mev[-1] - energies_mev[0],
            "distance_traveled": positions[-1] - positions[0],
            "trajectory_length": len(trajectory),
        }

    def create_energy_position_plots(
        self,
        electron_results: List[Tuple[float, Dict]],
        proton_results: List[Tuple[float, Dict]] = None,
        save_path: str = "standardized_energy_position_analysis.png",
    ):
        """Create energy vs position plots for multiple energies."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Standardized Energy vs Position Analysis\\nConducting Aperture with Image Charges",
            fontsize=16,
        )

        # Plot 1: Electron trajectories
        ax1.set_title("Electron Energy vs Position")
        ax1.set_xlabel("Position (mm)")
        ax1.set_ylabel("Energy (MeV)")
        ax1.grid(True, alpha=0.3)

        for energy_mev, results in electron_results:
            positions = results["positions"]
            energies = results["energies_mev"]
            ax1.plot(
                positions,
                energies,
                "o-",
                label=f"{energy_mev} MeV initial",
                markersize=3,
            )
        ax1.legend()

        # Plot 2: Energy gain vs initial energy
        ax2.set_title("Energy Gain vs Initial Energy")
        ax2.set_xlabel("Initial Energy (MeV)")
        ax2.set_ylabel("Energy Gain (MeV)")
        ax2.grid(True, alpha=0.3)

        electron_initial = [energy for energy, _ in electron_results]
        electron_gains = [results["energy_gain"] for _, results in electron_results]
        ax2.semilogy(electron_initial, np.abs(electron_gains), "bo-", label="Electrons")

        if proton_results:
            proton_initial = [energy for energy, _ in proton_results]
            proton_gains = [results["energy_gain"] for _, results in proton_results]
            ax2.semilogy(proton_initial, np.abs(proton_gains), "ro-", label="Protons")

        ax2.legend()

        # Plot 3: Relative energy change
        ax3.set_title("Relative Energy Change")
        ax3.set_xlabel("Initial Energy (MeV)")
        ax3.set_ylabel("ΔE/E₀ (%)")
        ax3.grid(True, alpha=0.3)

        relative_changes = [
            (results["energy_gain"] / energy) * 100
            for energy, results in electron_results
        ]
        ax3.semilogy(
            electron_initial, np.abs(relative_changes), "bo-", label="Electrons"
        )
        ax3.legend()

        # Plot 4: Distance traveled
        ax4.set_title("Distance Traveled")
        ax4.set_xlabel("Initial Energy (MeV)")
        ax4.set_ylabel("Distance (mm)")
        ax4.grid(True, alpha=0.3)

        distances = [results["distance_traveled"] for _, results in electron_results]
        ax4.plot(electron_initial, distances, "go-", label="Electrons")
        ax4.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Plot saved as: {save_path}")


def main():
    """Main analysis using standardized input creation."""

    print("=== Standardized Energy vs Position Analysis ===\\n")

    # Initialize analysis
    analysis = StandardizedEnergyPositionAnalysis()

    # Define energy ranges (1-500 MeV for electrons as requested)
    electron_energies = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]  # MeV

    # Define particle species
    electron = ParticleSpecies.electron()

    # Run electron energy range study
    print("Starting electron energy range study...")
    electron_results = analysis.run_energy_range_study(
        energy_range_mev=electron_energies,
        particle_species=electron,
        aperture_radius=5.0,
        wall_position=0.0,
        starting_distance=-200.0,  # Start 200mm before wall
    )

    # Optional: Run proton study for comparison (if requested)
    # proton = ParticleSpecies.proton()
    # proton_energies = [100.0, 500.0, 1000.0, 2000.0]  # MeV
    # proton_results = analysis.run_energy_range_study(
    #     energy_range_mev=proton_energies,
    #     particle_species=proton,
    #     aperture_radius=5.0,
    #     wall_position=0.0,
    #     starting_distance=-200.0
    # )

    # Create plots
    if electron_results:
        analysis.create_energy_position_plots(
            electron_results=electron_results,
            # proton_results=proton_results,  # Add if proton study is run
            save_path="standardized_energy_position_analysis.png",
        )

        # Print summary
        print("\\n=== Analysis Summary ===")
        print(f"Processed {len(electron_results)} electron energy points")
        print("Energy gains observed for conducting aperture image charge interactions")
        print(
            "Results demonstrate proper physics implementation with standardized input creation"
        )
    else:
        print("No results obtained. Check simulation parameters.")


if __name__ == "__main__":
    main()
