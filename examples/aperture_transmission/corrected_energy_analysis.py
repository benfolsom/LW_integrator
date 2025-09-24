#!/usr/bin/env python3
"""
Corrected energy vs position plots for conducting aperture.
Uses consistent units throughout.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.trajectory_integrator import LienardWiechertIntegrator
from physics.constants import C_MMNS
from tests.test_config import (
    ELECTRON,
    PROTON,
    create_bunch_uniform_distribution,
    TestConfiguration,
)


class ConductingApertureAnalysis:
    """Analysis of energy changes through conducting aperture."""

    def __init__(self):
        self.integrator = LienardWiechertIntegrator()
        self.aperture_radius = 5.0  # 5 microns radius

    def create_particle_bunch(self, particle_species, energy_gev: float):
        """Create particles with specific energy using test configuration method."""

        config = TestConfiguration(
            particle_count=5,
            transverse_separation=2.0,
            starting_distance=-50.0,
            step_size=1e-5,
            total_steps=1000,
            sim_type=0,
            wall_z=0.0,
            aperture_r=self.aperture_radius,
            z_cutoff=50.0,
        )

        # Create base bunch
        bunch = create_bunch_uniform_distribution(config, particle_species, "line")

        # Manually set momentum to get desired energy
        # Based on test code: energy = Pt * C_MMNS
        # So: Pt = energy_gev / C_MMNS (in test units)
        target_pt = energy_gev / C_MMNS

        # Update bunch momentum
        bunch["Pt"] = np.full(config.particle_count, target_pt)

        # Set mostly z-directed motion
        beta = 0.99999  # Very relativistic
        bunch["Pz"] = bunch["Pt"] * beta
        bunch["Px"] = np.zeros(config.particle_count)
        bunch["Py"] = np.zeros(config.particle_count)

        # Position within aperture
        bunch["x"] = np.linspace(-1.0, 1.0, config.particle_count)
        bunch["y"] = np.zeros(config.particle_count)
        bunch["z"] = np.full(config.particle_count, -50.0)

        # Set gamma (approximate for high energy)
        mass_gev = particle_species.mass_mev / 1000.0
        gamma_approx = energy_gev / mass_gev
        bunch["gamma"] = np.full(config.particle_count, gamma_approx)
        bunch["bz"] = np.full(config.particle_count, beta)
        bunch["bx"] = np.zeros(config.particle_count)
        bunch["by"] = np.zeros(config.particle_count)

        return bunch, config

    def run_simulation(self, particle_species, energy_gev: float):
        """Run conducting aperture simulation."""

        print(f"  Running {particle_species.name} at {energy_gev:.1f} GeV...")

        bunch, config = self.create_particle_bunch(particle_species, energy_gev)

        # Check initial energy
        initial_energy_check = bunch["Pt"][0] * C_MMNS
        print(f"    Initial energy check: {initial_energy_check:.3f} GeV")

        # Create distant driver
        driver_bunch = self.create_particle_bunch(particle_species, energy_gev)[0]
        for key in driver_bunch:
            if hasattr(driver_bunch[key], "__len__") and len(driver_bunch[key]) > 1:
                driver_bunch[key] = driver_bunch[key][:1]
        driver_bunch["x"] = np.array([1000.0])
        driver_bunch["y"] = np.array([1000.0])
        driver_bunch["z"] = np.array([-1000.0])

        # Run simulation
        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=100,
            ret_steps=900,
            h_step=1e-5,
            wall_Z=0.0,
            apt_R=self.aperture_radius,
            sim_type=0,  # Conducting wall
            init_rider=bunch,
            init_driver=driver_bunch,
            bunch_dist=1e5,
            z_cutoff=50.0,
        )

        return {
            "energy_gev": energy_gev,
            "particle_species": particle_species.name,
            "trajectory": trajectory_rider,
        }

    def create_plots(self, electron_results, proton_results):
        """Create energy vs position plots."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Electron Energy vs Position
        ax1.set_title(
            "Electron Energy Through\nConducting 10 μm Aperture",
            fontsize=14,
            fontweight="bold",
        )

        for result in electron_results:
            if result is None:
                continue

            trajectory = result["trajectory"]
            initial_energy_gev = result["energy_gev"]

            z_positions = []
            energies = []

            for step in trajectory:
                if len(step.get("z", [])) > 0 and len(step.get("Pt", [])) > 0:
                    z_mm = step["z"][0]
                    pt = step["Pt"][0]
                    # Use same conversion as test code
                    energy_gev = pt * C_MMNS

                    z_positions.append(z_mm)
                    energies.append(energy_gev)

            if z_positions and energies:
                # Convert mm to meters for plotting
                z_meters = np.array(z_positions) / 1000.0

                ax1.plot(
                    z_meters,
                    energies,
                    label=f"{initial_energy_gev:.1f} GeV initial",
                    linewidth=2,
                    alpha=0.8,
                )

        ax1.set_xlabel("Position (m)", fontsize=12)
        ax1.set_ylabel("Total Energy (GeV)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axvline(
            x=0,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Conducting Aperture",
        )

        # Plot 2: Proton Energy vs Position
        ax2.set_title(
            "Proton Energy Through\nConducting 10 μm Aperture",
            fontsize=14,
            fontweight="bold",
        )

        for result in proton_results:
            if result is None:
                continue

            trajectory = result["trajectory"]
            initial_energy_gev = result["energy_gev"]

            z_positions = []
            energies = []

            for step in trajectory:
                if len(step.get("z", [])) > 0 and len(step.get("Pt", [])) > 0:
                    z_mm = step["z"][0]
                    pt = step["Pt"][0]
                    # Use same conversion as test code
                    energy_gev = pt * C_MMNS

                    z_positions.append(z_mm)
                    energies.append(energy_gev)

            if z_positions and energies:
                # Convert mm to meters for plotting
                z_meters = np.array(z_positions) / 1000.0

                ax2.plot(
                    z_meters,
                    energies,
                    label=f"{initial_energy_gev:.1f} GeV initial",
                    linewidth=2,
                    alpha=0.8,
                )

        ax2.set_xlabel("Position (m)", fontsize=12)
        ax2.set_ylabel("Total Energy (GeV)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axvline(
            x=0,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Conducting Aperture",
        )

        plt.tight_layout()

        # Save plot
        output_path = os.path.join(
            os.path.dirname(__file__), "corrected_conducting_aperture_energy.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n📊 Corrected energy plot saved: {output_path}")
        plt.close()

    def run_analysis(self):
        """Run complete analysis."""

        print("🚀 Corrected Conducting Aperture Analysis")
        print("==========================================")

        # Electron analysis
        print("\n🔬 Electron Analysis:")
        electron_energies = [1.0, 5.0, 20.0]
        electron_results = []
        for energy in electron_energies:
            result = self.run_simulation(ELECTRON, energy)
            electron_results.append(result)

        # Proton analysis
        print("\n🔬 Proton Analysis:")
        proton_energies = [2.0, 10.0, 30.0]
        proton_results = []
        for energy in proton_energies:
            result = self.run_simulation(PROTON, energy)
            proton_results.append(result)

        # Create plots
        print("\n📊 Creating corrected plots...")
        self.create_plots(electron_results, proton_results)

        print("\n✅ Corrected analysis complete!")


if __name__ == "__main__":
    analysis = ConductingApertureAnalysis()
    analysis.run_analysis()
