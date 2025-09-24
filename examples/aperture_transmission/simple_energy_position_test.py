#!/usr/bin/env python3
"""
Simple test script to generate energy vs position plots for particles passing through aperture.
This focuses on just a few energy points to demonstrate energy conservation.
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


class SimpleEnergyPositionAnalysis:
    """Simple analysis to demonstrate energy changes through conducting aperture."""

    def __init__(self):
        self.integrator = LienardWiechertIntegrator()
        self.aperture_radius = 5.0  # 5 microns radius (10 micron diameter)
        self.propagation_distance = (
            100.0  # mm - total distance for energy gain analysis
        )

    def create_particle_bunch(
        self, particle_species, energy_gev: float, particle_count: int = 10
    ):
        """Create a particle bunch with specific energy and minimal transverse momentum."""

        # Create test configuration
        config = TestConfiguration(
            particle_count=particle_count,
            transverse_separation=2.0,  # mm, much smaller to ensure particles pass through aperture
            starting_distance=-50.0,  # Start 50mm before the aperture
            step_size=1e-5,
            total_steps=1000,  # More steps for 100mm propagation
            sim_type=0,  # Conducting wall with image charges
            wall_z=0.0,  # Aperture at z=0
            aperture_r=self.aperture_radius,
            z_cutoff=self.propagation_distance,
        )

        # Create base bunch
        bunch = create_bunch_uniform_distribution(config, particle_species, "line")

        # Calculate relativistic parameters for the specified energy
        rest_energy_gev = particle_species.mass_mev / 1000.0  # Convert MeV to GeV
        total_energy_gev = energy_gev

        # Skip if energy is below rest mass
        if total_energy_gev <= rest_energy_gev:
            print(
                f"  Skipping {energy_gev:.1f} GeV - below rest mass ({rest_energy_gev:.3f} GeV)"
            )
            return None

        # Calculate gamma and momentum
        gamma = total_energy_gev / rest_energy_gev

        # Calculate beta
        beta = np.sqrt(1.0 - 1.0 / gamma**2)

        # Update particle properties using correct integrator units
        # Pt = gamma * mass * c (in amu * mm/ns units)
        mass_amu = (
            particle_species.mass_mev / 931.494
        )  # Convert MeV to amu (1 amu = 931.494 MeV)

        bunch["Pt"] = np.full(particle_count, gamma * mass_amu * C_MMNS)
        bunch["Pz"] = bunch["Pt"] * beta  # Mostly moving in z direction

        # Minimal transverse momentum to ensure particles go through aperture
        bunch["Px"] = np.zeros(particle_count)
        bunch["Py"] = np.zeros(particle_count)

        # Position particles in a small line within the aperture radius
        bunch["x"] = np.linspace(
            -1.0, 1.0, particle_count
        )  # Within ±1 micron (aperture is 5 micron radius)
        bunch["y"] = np.zeros(particle_count)  # All on axis
        bunch["z"] = np.full(particle_count, config.starting_distance)  # Start at -50mm

        bunch["gamma"] = np.full(particle_count, gamma)
        bunch["bz"] = np.full(particle_count, beta)
        bunch["bx"] = np.zeros(particle_count)
        bunch["by"] = np.zeros(particle_count)

        return bunch

    def run_single_energy_simulation(self, particle_species, energy_gev: float):
        """Run simulation for a single energy point."""

        print(f"  Running {particle_species.name} at {energy_gev:.1f} GeV...")

        # Create particle bunch
        bunch = self.create_particle_bunch(particle_species, energy_gev)
        if bunch is None:
            return None

        # Create a dummy driver bunch (no interaction for this study)
        driver_bunch = self.create_particle_bunch(
            particle_species, energy_gev, particle_count=1
        )
        if driver_bunch is None:
            return None
        driver_bunch["x"] = np.array([1000.0])  # Far away to minimize interaction
        driver_bunch["y"] = np.array([1000.0])
        driver_bunch["z"] = np.array([-1000.0])

        # Run simulation with conducting aperture
        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=100,
            ret_steps=900,  # More retarded steps for conducting aperture physics
            h_step=1e-5,
            wall_Z=0.0,  # Aperture at z=0
            apt_R=self.aperture_radius,
            sim_type=0,  # Conducting wall with image charges
            init_rider=bunch,
            init_driver=driver_bunch,
            bunch_dist=1e5,
            z_cutoff=self.propagation_distance,
        )

        return {
            "energy_gev": energy_gev,
            "particle_species": particle_species.name,
            "trajectory": trajectory_rider,
        }

    def create_energy_position_plots(self, electron_results, proton_results):
        """Create energy vs position plots showing energy increase from image charges."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Electron Energy vs Position
        ax1.set_title(
            "Electron Energy Increase\nThrough Conducting 10 μm Aperture",
            fontsize=14,
            fontweight="bold",
        )

        for result in electron_results:
            if result is None:
                continue

            trajectory = result["trajectory"]
            energy_gev = result["energy_gev"]

            # Extract position and energy data for first particle
            z_positions = []
            energies = []

            for step in trajectory:
                if len(step.get("z", [])) > 0 and len(step.get("Pt", [])) > 0:
                    z_mm = step["z"][0]  # First particle position in mm
                    pt = step["Pt"][0]  # First particle total momentum in amu*mm/ns
                    # Convert Pt to energy: E = Pt * c (already in correct integrator units)
                    energy_integrator = pt * C_MMNS  # amu*(mm/ns)^2
                    # Convert to GeV: 1 amu*(mm/ns)^2 = 1 amu*c^2 = 931.494 MeV
                    energy_gev = energy_integrator * 931.494 / 1000.0  # Convert to GeV

                    z_positions.append(z_mm)
                    energies.append(energy_gev)

            if z_positions and energies:
                # Convert mm to meters for plotting
                z_meters = np.array(z_positions) / 1000.0

                ax1.plot(
                    z_meters,
                    energies,
                    label=f"{energy_gev:.1f} GeV initial",
                    linewidth=2,
                    alpha=0.8,
                )

        ax1.set_xlabel("Position (m)", fontsize=12)
        ax1.set_ylabel("Total Energy (GeV)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add vertical line at aperture position (conducting wall at z=0)
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
            "Proton Energy Increase\nThrough Conducting 10 μm Aperture",
            fontsize=14,
            fontweight="bold",
        )

        for result in proton_results:
            if result is None:
                continue

            trajectory = result["trajectory"]
            energy_gev = result["energy_gev"]

            # Extract position and energy data for first particle
            z_positions = []
            energies = []

            for step in trajectory:
                if len(step.get("z", [])) > 0 and len(step.get("Pt", [])) > 0:
                    z_mm = step["z"][0]  # First particle position in mm
                    pt = step["Pt"][0]  # First particle total momentum in amu*mm/ns
                    # Convert Pt to energy: E = Pt * c (already in correct integrator units)
                    energy_integrator = pt * C_MMNS  # amu*(mm/ns)^2
                    # Convert to GeV: 1 amu*(mm/ns)^2 = 1 amu*c^2 = 931.494 MeV
                    energy_gev = energy_integrator * 931.494 / 1000.0  # Convert to GeV

                    z_positions.append(z_mm)
                    energies.append(energy_gev)

            if z_positions and energies:
                # Convert mm to meters for plotting
                z_meters = np.array(z_positions) / 1000.0

                ax2.plot(
                    z_meters,
                    energies,
                    label=f"{energy_gev:.1f} GeV initial",
                    linewidth=2,
                    alpha=0.8,
                )

        ax2.set_xlabel("Position (m)", fontsize=12)
        ax2.set_ylabel("Total Energy (GeV)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add vertical line at aperture position
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
            os.path.dirname(__file__), "conducting_aperture_energy_gain.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n📊 Conducting aperture energy gain plot saved: {output_path}")

        # Also save to show
        plt.show()


def main():
    """Main analysis function."""

    print("🚀 Conducting Aperture Energy Gain Analysis")
    print("=" * 55)
    print("Demonstrating energy increase from image charges in conducting aperture.")
    print(
        "Particles travel ~100mm and should show energy gain from electromagnetic effects.\n"
    )

    # Initialize analysis
    analysis = SimpleEnergyPositionAnalysis()

    # Use fewer energy points for cleaner plots
    electron_energies = [1.0, 5.0, 20.0]  # GeV
    proton_energies = [2.0, 10.0, 30.0]  # GeV (all above rest mass)

    # Run analysis for electrons
    print("🔬 Electron Analysis (conducting aperture):")
    electron_results = []
    for energy in electron_energies:
        result = analysis.run_single_energy_simulation(ELECTRON, energy)
        electron_results.append(result)

    # Run analysis for protons
    print("\n🔬 Proton Analysis (conducting aperture):")
    proton_results = []
    for energy in proton_energies:
        result = analysis.run_single_energy_simulation(PROTON, energy)
        proton_results.append(result)

    # Create visualization
    print("\n📊 Creating energy gain plots for conducting aperture...")
    analysis.create_energy_position_plots(electron_results, proton_results)

    print("\n✅ Analysis complete!")
    print(
        "Expected: Energy should increase due to acceleration from image charge fields."
    )


if __name__ == "__main__":
    main()
