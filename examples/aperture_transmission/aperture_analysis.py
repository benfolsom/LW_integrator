"""
Aperture Transmission Analysis for Electrons and Protons

This example demonstrates the electromagnetic effects on particle transmission
through a 10 micron aperture at various energies from 1 MeV to 100 GeV.

The simulations show:
1. Energy-dependent transmission rates
2. Electromagnetic deflection effects
3. Comparison between electrons and protons
4. Relativistic physics validation

Author: Ben Folsom
Date: 2025-09-18
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time

# Add paths for the LW integrator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.trajectory_integrator import LienardWiechertIntegrator
from physics.constants import C_MMNS
from tests.test_config import (
    ELECTRON,
    PROTON,
    create_bunch_uniform_distribution,
    TestConfiguration,
)


class ApertureTransmissionAnalysis:
    """Analysis of particle transmission through apertures at various energies."""

    def __init__(self):
        self.integrator = LienardWiechertIntegrator()
        self.aperture_radius = (
            5.0  # 10 micron diameter = 5 micron radius in mm (1 micron = 0.001 mm)
        )
        self.results = {}

    def create_particle_bunch(
        self, particle_species, energy_gev: float, particle_count: int = 10
    ) -> Dict[str, np.ndarray]:
        """Create a particle bunch with specified energy."""

        config = TestConfiguration(
            particle_count=particle_count,
            transverse_separation=8.0,  # 8mm beam spread (larger than aperture)
            starting_distance=200.0,  # Start 200mm upstream
            step_size=1e-5,
            total_steps=500,
            sim_type=2,  # Free particle bunches
            wall_z=1e5,
            aperture_r=self.aperture_radius,
            z_cutoff=0.0,
        )

        # Create base bunch
        bunch = create_bunch_uniform_distribution(config, particle_species, "circle")

        # Calculate relativistic parameters for the specified energy
        rest_energy_gev = particle_species.mass_mev / 1000.0  # Convert MeV to GeV
        total_energy_gev = energy_gev

        # Calculate gamma and momentum
        gamma = total_energy_gev / rest_energy_gev
        momentum_gev = np.sqrt(total_energy_gev**2 - rest_energy_gev**2)

        # Calculate beta
        beta = momentum_gev / total_energy_gev

        # Update particle properties (in integrator units)
        # Note: Need to be careful about unit conversions
        momentum_scale = 1e-3  # Adjust for integrator units

        bunch["Pt"] = np.full(
            particle_count, momentum_gev * 1000 * momentum_scale
        )  # Convert to MeV, then scale
        bunch["Pz"] = bunch["Pt"] * beta  # Mostly moving in z direction
        bunch["Px"] = np.zeros(particle_count)
        bunch["Py"] = np.zeros(particle_count)

        bunch["gamma"] = np.full(particle_count, gamma)
        bunch["bz"] = np.full(particle_count, beta)
        bunch["bx"] = np.zeros(particle_count)
        bunch["by"] = np.zeros(particle_count)

        # Add char_time for electromagnetic calculations
        # char_time = mass / (charge * C_MMNS) for each particle
        particle_count = len(bunch["q"])
        char_time = np.array(
            [
                (
                    bunch["mass"][i] / (bunch["q"][i] * C_MMNS)
                    if bunch["q"][i] != 0
                    else 1.0
                )
                for i in range(particle_count)
            ]
        )
        bunch["char_time"] = char_time

        return bunch

    def run_aperture_simulation(
        self, particle_species, energy_gev: float
    ) -> Dict[str, Any]:
        """Run aperture transmission simulation for given particle type and energy."""

        print(f"  Running {particle_species.name} at {energy_gev:.1f} GeV...")

        # Create particle bunch
        bunch = self.create_particle_bunch(particle_species, energy_gev)

        # Create a dummy driver bunch (no interaction for this study)
        driver_bunch = self.create_particle_bunch(
            particle_species, energy_gev, particle_count=1
        )
        driver_bunch["x"] = np.array([1000.0])  # Far away to minimize interaction
        driver_bunch["y"] = np.array([1000.0])
        driver_bunch["z"] = np.array([-1000.0])

        # Count initial particles within aperture
        initial_radius = np.sqrt(bunch["x"] ** 2 + bunch["y"] ** 2)
        initial_transmitted = np.sum(initial_radius <= self.aperture_radius)

        start_time = time.time()

        # Run simulation
        trajectory_rider, trajectory_driver = self.integrator.integrate_retarded_fields(
            static_steps=50,
            ret_steps=450,
            h_step=1e-5,
            wall_Z=1e5,
            apt_R=self.aperture_radius,
            sim_type=2,
            init_rider=bunch,
            init_driver=driver_bunch,
            bunch_dist=1e5,
            z_cutoff=0.0,
        )

        simulation_time = time.time() - start_time

        # Analyze final state
        final_state = trajectory_rider[-1]
        final_radius = np.sqrt(final_state["x"] ** 2 + final_state["y"] ** 2)
        final_transmitted = np.sum(
            final_radius <= self.aperture_radius * 1.1
        )  # Small tolerance

        # Calculate transmission rate
        transmission_rate = (
            final_transmitted / initial_transmitted if initial_transmitted > 0 else 0.0
        )

        # Calculate average deflection
        initial_px = bunch["Px"]
        final_px = final_state["Px"]
        avg_deflection = np.mean(np.abs(final_px - initial_px))

        # Calculate final positions for analysis
        final_z = np.mean(final_state["z"])

        return {
            "energy_gev": energy_gev,
            "initial_transmitted": initial_transmitted,
            "final_transmitted": final_transmitted,
            "transmission_rate": transmission_rate,
            "avg_deflection": avg_deflection,
            "final_z": final_z,
            "simulation_time": simulation_time,
            "gamma": bunch["gamma"][0],
            "beta": bunch["bz"][0],
            "trajectory": trajectory_rider,  # Store full trajectory for energy vs position plots
        }

    def run_energy_scan(
        self, particle_species, energies_gev: List[float]
    ) -> List[Dict[str, Any]]:
        """Run aperture transmission analysis across energy range."""

        print(f"\n🔬 Aperture Transmission Analysis: {particle_species.name}")
        print(f"   Aperture: {self.aperture_radius*2:.1f} microns diameter")
        print(f"   Energies: {energies_gev[0]:.3f} - {energies_gev[-1]:.1f} GeV")

        results = []

        for energy in energies_gev:
            try:
                result = self.run_aperture_simulation(particle_species, energy)
                results.append(result)

                print(
                    f"    {energy:6.1f} GeV: γ={result['gamma']:8.1f}, β={result['beta']:.4f}, "
                    f"T={result['transmission_rate']:.3f}, Δp={result['avg_deflection']:.2e}"
                )

            except Exception as e:
                print(f"    {energy:6.1f} GeV: FAILED - {str(e)}")
                continue

        return results

    def create_transmission_plot(
        self, electron_results: List[Dict], proton_results: List[Dict]
    ) -> None:
        """Create energy vs position plots for electrons and protons."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Electron Energy vs Position
        ax1.set_title(
            "Electron Energy vs Position\n(10 μm diameter aperture)",
            fontsize=14,
            fontweight="bold",
        )

        for result in electron_results:
            if "trajectory" not in result:
                continue

            trajectory = result["trajectory"]
            energy_gev = result["energy_gev"]

            # Extract position and energy data for first particle
            z_positions = []
            energies = []

            for step in trajectory:
                if len(step.get("z", [])) > 0 and len(step.get("Pt", [])) > 0:
                    z_mm = step["z"][0]  # First particle position in mm
                    pt = step["Pt"][0]  # First particle total momentum
                    energy = pt * C_MMNS  # Convert to GeV

                    z_positions.append(z_mm)
                    energies.append(energy)

            if z_positions and energies:
                # Convert mm to meters for plotting
                z_meters = np.array(z_positions) / 1000.0

                # Plot with different colors for different energies
                alpha = 0.7 if len(electron_results) > 5 else 0.9
                linewidth = 1.5 if len(electron_results) > 5 else 2.0

                ax1.plot(
                    z_meters,
                    energies,
                    label=f"{energy_gev:.1f} GeV",
                    alpha=alpha,
                    linewidth=linewidth,
                )

        ax1.set_xlabel("Position (m)", fontsize=12)
        ax1.set_ylabel("Total Energy (GeV)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

        # Add vertical line at aperture position (assume aperture at z=0)
        ax1.axvline(
            x=0, color="red", linestyle="--", alpha=0.7, linewidth=2, label="Aperture"
        )

        # Plot 2: Proton Energy vs Position
        ax2.set_title(
            "Proton Energy vs Position\n(10 μm diameter aperture)",
            fontsize=14,
            fontweight="bold",
        )

        for result in proton_results:
            if "trajectory" not in result:
                continue

            trajectory = result["trajectory"]
            energy_gev = result["energy_gev"]

            # Extract position and energy data for first particle
            z_positions = []
            energies = []

            for step in trajectory:
                if len(step.get("z", [])) > 0 and len(step.get("Pt", [])) > 0:
                    z_mm = step["z"][0]  # First particle position in mm
                    pt = step["Pt"][0]  # First particle total momentum
                    energy = pt * C_MMNS  # Convert to GeV

                    z_positions.append(z_mm)
                    energies.append(energy)

            if z_positions and energies:
                # Convert mm to meters for plotting
                z_meters = np.array(z_positions) / 1000.0

                # Plot with different colors for different energies
                alpha = 0.7 if len(proton_results) > 5 else 0.9
                linewidth = 1.5 if len(proton_results) > 5 else 2.0

                ax2.plot(
                    z_meters,
                    energies,
                    label=f"{energy_gev:.1f} GeV",
                    alpha=alpha,
                    linewidth=linewidth,
                )

        ax2.set_xlabel("Position (m)", fontsize=12)
        ax2.set_ylabel("Total Energy (GeV)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

        # Add vertical line at aperture position
        ax2.axvline(
            x=0, color="red", linestyle="--", alpha=0.7, linewidth=2, label="Aperture"
        )

        plt.tight_layout()

        # Save plot
        output_path = os.path.join(os.path.dirname(__file__), "energy_vs_position.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n📊 Energy vs Position plot saved: {output_path}")

        plt.show()

    def create_physics_summary_plot(
        self, electron_results: List[Dict], proton_results: List[Dict]
    ) -> None:
        """Create additional physics analysis plots."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Extract data
        e_energies = [r["energy_gev"] for r in electron_results]
        e_gammas = [r["gamma"] for r in electron_results]
        e_betas = [r["beta"] for r in electron_results]

        p_energies = [r["energy_gev"] for r in proton_results]
        p_gammas = [r["gamma"] for r in proton_results]
        p_betas = [r["beta"] for r in proton_results]

        # Plot 1: Lorentz factor
        ax1.loglog(e_energies, e_gammas, "b-o", label="Electrons", linewidth=2)
        ax1.loglog(p_energies, p_gammas, "r-s", label="Protons", linewidth=2)
        ax1.set_xlabel("Energy (GeV)")
        ax1.set_ylabel("Lorentz Factor (γ)")
        ax1.set_title("Relativistic γ vs Energy")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Velocity
        ax2.semilogx(e_energies, e_betas, "b-o", label="Electrons", linewidth=2)
        ax2.semilogx(p_energies, p_betas, "r-s", label="Protons", linewidth=2)
        ax2.set_xlabel("Energy (GeV)")
        ax2.set_ylabel("Velocity (β = v/c)")
        ax2.set_title("Relativistic β vs Energy")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1.02)

        # Plot 3: Energy breakdown
        e_rest = [ELECTRON.mass_mev / 1000.0] * len(e_energies)
        e_kinetic = [total - rest for total, rest in zip(e_energies, e_rest)]

        p_rest = [PROTON.mass_mev / 1000.0] * len(p_energies)
        p_kinetic = [total - rest for total, rest in zip(p_energies, p_rest)]

        ax3.loglog(e_energies, e_kinetic, "b-o", label="Electron Kinetic", linewidth=2)
        ax3.loglog(e_energies, e_rest, "b--", label="Electron Rest", linewidth=1)
        ax3.loglog(p_energies, p_kinetic, "r-s", label="Proton Kinetic", linewidth=2)
        ax3.loglog(p_energies, p_rest, "r--", label="Proton Rest", linewidth=1)
        ax3.set_xlabel("Total Energy (GeV)")
        ax3.set_ylabel("Energy (GeV)")
        ax3.set_title("Kinetic vs Rest Energy")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot 4: Simulation performance
        e_sim_times = [r["simulation_time"] for r in electron_results]
        p_sim_times = [r["simulation_time"] for r in proton_results]

        ax4.semilogx(e_energies, e_sim_times, "b-o", label="Electrons", linewidth=2)
        ax4.semilogx(p_energies, p_sim_times, "r-s", label="Protons", linewidth=2)
        ax4.set_xlabel("Energy (GeV)")
        ax4.set_ylabel("Simulation Time (s)")
        ax4.set_title("Computational Performance")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()

        # Save plot
        output_path = os.path.join(
            os.path.dirname(__file__), "physics_analysis_summary.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"📊 Physics analysis saved: {output_path}")

        plt.show()


def main():
    """Main analysis function."""

    print("🚀 LW Integrator: Aperture Transmission Analysis")
    print("=" * 60)
    print("Analyzing electromagnetic effects on particle transmission")
    print("through a 10 micron diameter aperture at relativistic energies.\n")

    # Initialize analysis
    analysis = ApertureTransmissionAnalysis()

    # Define energy range: 1 MeV to 100 GeV
    energies_gev = np.logspace(-3, 2, 15)  # 0.001 GeV (1 MeV) to 100 GeV

    # Run analysis for electrons
    electron_results = analysis.run_energy_scan(ELECTRON, energies_gev)

    # Run analysis for protons
    proton_results = analysis.run_energy_scan(PROTON, energies_gev)

    # Create visualization
    if electron_results and proton_results:
        print("\n📊 Creating visualization...")
        analysis.create_transmission_plot(electron_results, proton_results)

        # Save data (exclude large trajectory data)
        import json

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(v) for v in obj]
            return obj

        # Filter out trajectory data for JSON saving
        def filter_trajectory_data(results):
            filtered = []
            for result in results:
                filtered_result = {k: v for k, v in result.items() if k != "trajectory"}
                filtered.append(filtered_result)
            return filtered

        data_path = os.path.join(os.path.dirname(__file__), "transmission_data.json")
        with open(data_path, "w") as f:
            json.dump(
                {
                    "electrons": convert_numpy_types(
                        filter_trajectory_data(electron_results)
                    ),
                    "protons": convert_numpy_types(
                        filter_trajectory_data(proton_results)
                    ),
                    "aperture_radius_mm": float(analysis.aperture_radius),
                },
                f,
                indent=2,
            )
        print(f"💾 Data saved: {data_path}")

    else:
        print("❌ No results obtained - check simulation parameters")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
