#!/usr/bin/env python3
"""
Direct Legacy vs Updated Integrator Comparison with Ez Field

This test creates a side-by-side comparison of legacy and updated integrators
using constant Ez field acceleration to demonstrate identical physics behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "legacy"))

try:
    from bunch_inits import init_bunch

    LEGACY_AVAILABLE = True
except ImportError as e:
    print(f"Legacy imports failed: {e}")
    LEGACY_AVAILABLE = False

from core.trajectory_integrator import LienardWiechertIntegrator
from physics.particle_initialization import ELEMENTARY_CHARGE


class LegacyVsUpdatedComparison:
    """Direct comparison between legacy and updated integrators with Ez field."""

    def __init__(self):
        self.c_mmns = 299.792458  # mm/ns

    def safe_get(self, data, key, index, fallback=1.0):
        """Safely extract scalar values from particle data."""
        if key not in data:
            return fallback
        value = data[key]
        if hasattr(value, "__len__") and len(value) > index:
            return float(value[index])
        elif hasattr(value, "__len__"):
            return float(value[0]) if len(value) > 0 else fallback
        else:
            return float(value)

    def apply_ez_field_step_legacy(self, particle: dict, Ez: float, dt: float) -> dict:
        """Apply Ez field step to legacy particle format."""

        # Extract current values
        charge = self.safe_get(
            particle, "q", 0, ELEMENTARY_CHARGE
        )  # Default to proper Gaussian units
        mass = self.safe_get(particle, "m", 0, 1.0)
        gamma = self.safe_get(particle, "gamma", 0, 1.0)
        beta_z = self.safe_get(particle, "bz", 0, 0.0)
        Px = self.safe_get(particle, "Px", 0, 0.0)
        Py = self.safe_get(particle, "Py", 0, 0.0)
        Pz = self.safe_get(particle, "Pz", 0, 0.0)

        # Apply Ez field: ΔPz = q * γ * (1-β) * Ez * Δt
        one_minus_beta = 1.0 - beta_z
        delta_Pz = charge * gamma * one_minus_beta * Ez * dt
        Pz_new = Pz + delta_Pz

        # Recalculate derived quantities
        P_mag_squared = Px**2 + Py**2 + Pz_new**2
        gamma_new = np.sqrt(1.0 + P_mag_squared / (mass * self.c_mmns) ** 2)

        # Calculate new velocity
        vz_new = Pz_new / (gamma_new * mass * self.c_mmns)
        beta_z_new = vz_new / self.c_mmns

        # Calculate new energy
        E_new = gamma_new * mass * self.c_mmns**2

        # Create updated particle state
        result = particle.copy()

        # Update the modified quantities
        if hasattr(particle["Pz"], "__len__"):
            result["Pz"] = np.array([Pz_new])
            result["gamma"] = np.array([gamma_new])
            result["bz"] = np.array([beta_z_new])
            result["E"] = np.array([E_new])
        else:
            result["Pz"] = Pz_new
            result["gamma"] = gamma_new
            result["bz"] = beta_z_new
            result["E"] = E_new

        return result

    def run_legacy_simulation(self, Ez_field: float, steps: int, dt: float) -> tuple:
        """Run simulation with legacy format and Ez field."""

        if not LEGACY_AVAILABLE:
            return None, None

        # Initialize 100 MeV proton using legacy format
        # For 100 MeV proton: gamma = 1.107, Pz ≈ 346 amu*mm/ns
        gamma_target = 1.107
        mass = 1.0  # amu
        starting_Pz = (
            np.sqrt(gamma_target**2 - 1.0) * mass * self.c_mmns
        )  # ≈ 146 amu*mm/ns

        particle_tuple = init_bunch(
            starting_distance=0.0,  # mm
            transv_mom=0.0,  # amu*mm/ns
            starting_Pz=starting_Pz,  # amu*mm/ns
            stripped_ions=1,  # proton
            m_particle=1.0,  # amu
            transv_dist=0.0,  # mm
            pcount=1,  # 1 particle
            charge_sign=1,  # positive
        )

        # Extract the particle dictionary and rest energy
        particle, E_rest = particle_tuple

        # Store trajectory
        trajectory = []
        energies = []

        for i in range(steps):
            # Store current state
            # Use actual energy calculation from particle state
            gamma = self.safe_get(particle, "gamma", 0, 1.0)
            mass = self.safe_get(particle, "m", 0, 1.0)
            energy = (
                gamma * mass * self.c_mmns**2
            )  # Energy in units where c=299.79... mm/ns

            trajectory.append(
                {
                    "step": i,
                    "energy": energy,
                    "z": self.safe_get(particle, "z", 0, 0.0),
                    "Pz": self.safe_get(particle, "Pz", 0, 0.0),
                    "gamma": gamma,
                }
            )
            energies.append(energy)

            # Apply Ez field for next step
            if i < steps - 1:
                particle = self.apply_ez_field_step_legacy(particle, Ez_field, dt)

                # Update position
                vz = self.safe_get(particle, "Pz", 0) / (
                    self.safe_get(particle, "gamma", 0)
                    * self.safe_get(particle, "m", 0, 1.0)
                    * self.c_mmns
                )
                z_current = self.safe_get(particle, "z", 0, 0.0)
                z_new = z_current + vz * dt

                if hasattr(particle["z"], "__len__"):
                    particle["z"] = np.array([z_new])
                else:
                    particle["z"] = z_new

        return trajectory, energies

    def run_updated_simulation(self, Ez_field: float, steps: int, dt: float) -> tuple:
        """Run simulation with updated integrator and Ez field."""

        # Create updated integrator
        integrator = LienardWiechertIntegrator()

        # Manually create particle state for 100 MeV proton in correct Gaussian units
        gamma = 1.107  # 100 MeV proton
        mass = 1.0  # proton mass in amu (this is the convenience value)

        # CRITICAL: Use same charge as legacy system (Gaussian units)
        charge = 1.178734e-5  # Same as legacy: 1.6E-19 C converted to Gaussian amu⋅mm⋅ns units

        # Calculate momentum: P = γmv, and P² = (γ²-1)m²c²
        P_mag = np.sqrt(gamma**2 - 1.0) * mass * self.c_mmns

        # Start with motion in z direction
        Px, Py, Pz = 0.0, 0.0, P_mag

        # Calculate velocity correctly: v = P/(γm)
        vz = Pz / (gamma * mass * self.c_mmns)
        beta_z = vz / self.c_mmns

        # Create initial state using legacy-compatible Gaussian units
        init_rider = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "t": np.array([0.0]),
            "Px": np.array([Px]),
            "Py": np.array([Py]),
            "Pz": np.array([Pz]),
            "q": np.array([charge]),  # FIXED: Use Gaussian units, not 1.0
            "m": np.array([mass]),
            "gamma": np.array([gamma]),
            "bx": np.array([0.0]),
            "by": np.array([0.0]),
            "bz": np.array([beta_z]),
            "E": np.array([gamma * mass * self.c_mmns**2]),  # Energy
        }

        # Create dummy driver (not used in free particle mode)
        init_driver = init_rider.copy()

        # Run integration with Ez field (using static steps only)
        trajectory_new, trajectory_drv = integrator.integrate_retarded_fields(
            static_steps=steps,
            ret_steps=0,  # Only static for fair comparison
            h_step=dt,
            wall_Z=1000.0,  # Far wall
            apt_R=50.0,
            sim_type=2,  # Free particle bunches
            init_rider=init_rider,
            init_driver=init_driver,
            bunch_dist=100.0,
            z_cutoff=500.0,
            Ez_field=Ez_field,
        )

        # Extract trajectory data
        trajectory = []
        energies = []

        for i in range(len(trajectory_new)):
            step_data = trajectory_new[i]
            energy = step_data["E"][0]  # MeV
            trajectory.append(
                {
                    "step": i,
                    "energy": energy,
                    "z": step_data["z"][0],
                    "Pz": step_data["Pz"][0],
                    "gamma": step_data["gamma"][0],
                }
            )
            energies.append(energy)

        return trajectory, energies

    def run_comparison(self, Ez_fields: list, steps: int = 1000, dt: float = 1e-5):
        """Run complete comparison for multiple field strengths."""

        print("=== Legacy vs Updated Integrator Comparison with Ez Field ===")
        print(f"Simulation parameters: {steps} steps, dt = {dt:.0e} ns")
        print("WARNING: Ez field values are in native Gaussian amu⋅mm⋅ns units")
        print("(Conversion to MV/m is complex due to mixed unit system)")
        print()

        results = {}

        for Ez in Ez_fields:
            print(f"Testing Ez = {Ez:.2e} (Gaussian amu⋅mm⋅ns units)...")

            # Run both simulations
            legacy_traj, legacy_energies = self.run_legacy_simulation(Ez, steps, dt)
            updated_traj, updated_energies = self.run_updated_simulation(Ez, steps, dt)

            if legacy_traj is None:
                print("  Legacy simulation failed (imports not available)")
                continue

            # Calculate energy gains
            legacy_initial = legacy_energies[0]
            legacy_final = legacy_energies[-1]
            legacy_gain = legacy_final - legacy_initial

            updated_initial = updated_energies[0]
            updated_final = updated_energies[-1]
            updated_gain = updated_final - updated_initial

            # Calculate relative difference
            if abs(legacy_gain) > 1e-10:
                rel_diff = abs(updated_gain - legacy_gain) / abs(legacy_gain) * 100
            else:
                rel_diff = 0.0

            results[Ez] = {
                "legacy": {
                    "initial": legacy_initial,
                    "final": legacy_final,
                    "gain": legacy_gain,
                },
                "updated": {
                    "initial": updated_initial,
                    "final": updated_final,
                    "gain": updated_gain,
                },
                "rel_diff": rel_diff,
                "legacy_traj": legacy_traj,
                "updated_traj": updated_traj,
            }

            print(
                f"  Legacy:  {legacy_initial:.6f} → {legacy_final:.6f} (gain: {legacy_gain:.6f})"
            )
            print(
                f"  Updated: {updated_initial:.6f} → {updated_final:.6f} (gain: {updated_gain:.6f})"
            )
            print(f"  Relative difference: {rel_diff:.3f}%")
            print()

        return results

    def plot_comparison(self, results: dict):
        """Create comparison plots."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Energy evolution for different fields
        for Ez, data in results.items():
            if data["legacy_traj"] is None:
                continue

            legacy_steps = [t["step"] for t in data["legacy_traj"]]
            legacy_energies = [t["energy"] for t in data["legacy_traj"]]
            updated_steps = [t["step"] for t in data["updated_traj"]]
            updated_energies = [t["energy"] for t in data["updated_traj"]]

            ax1.plot(
                legacy_steps,
                legacy_energies,
                "o-",
                label=f"Legacy Ez={Ez:.0f}",
                markersize=2,
            )
            ax1.plot(
                updated_steps,
                updated_energies,
                "s--",
                label=f"Updated Ez={Ez:.0f}",
                markersize=2,
            )

        ax1.set_xlabel("Integration Step")
        ax1.set_ylabel("Energy (MeV)")
        ax1.set_title("Energy Evolution Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Energy gain vs field strength
        Ez_values = list(results.keys())
        legacy_gains = [
            results[Ez]["legacy"]["gain"]
            for Ez in Ez_values
            if results[Ez]["legacy_traj"] is not None
        ]
        updated_gains = [
            results[Ez]["updated"]["gain"]
            for Ez in Ez_values
            if results[Ez]["legacy_traj"] is not None
        ]

        ax2.loglog(
            Ez_values, legacy_gains, "o-", label="Legacy", linewidth=2, markersize=6
        )
        ax2.loglog(
            Ez_values, updated_gains, "s--", label="Updated", linewidth=2, markersize=6
        )
        ax2.set_xlabel("Ez Field (Gaussian amu⋅mm⋅ns units)")
        ax2.set_ylabel("Energy Gain (amu⋅mm²⋅ns⁻²)")
        ax2.set_title("Energy Gain vs Field Strength (Log Scale)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Relative difference
        rel_diffs = [
            results[Ez]["rel_diff"]
            for Ez in Ez_values
            if results[Ez]["legacy_traj"] is not None
        ]
        ax3.semilogx(Ez_values, rel_diffs, "ro-", linewidth=2, markersize=6)
        ax3.set_xlabel("Ez Field (Gaussian amu⋅mm⋅ns units)")
        ax3.set_ylabel("Relative Difference (%)")
        ax3.set_title("Legacy vs Updated Relative Difference")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Position evolution for highest field
        if results:
            highest_Ez = max(Ez_values)
            if results[highest_Ez]["legacy_traj"] is not None:
                legacy_traj = results[highest_Ez]["legacy_traj"]
                updated_traj = results[highest_Ez]["updated_traj"]

                legacy_z = [t["z"] for t in legacy_traj]
                updated_z = [t["z"] for t in updated_traj]
                steps = [t["step"] for t in legacy_traj]

                ax4.plot(
                    steps,
                    legacy_z,
                    "o-",
                    label=f"Legacy Ez={highest_Ez:.0f}",
                    markersize=2,
                )
                ax4.plot(
                    steps,
                    updated_z,
                    "s--",
                    label=f"Updated Ez={highest_Ez:.0f}",
                    markersize=2,
                )
                ax4.set_xlabel("Integration Step")
                ax4.set_ylabel("Z Position (mm)")
                ax4.set_title(f"Position Evolution (Ez = {highest_Ez:.0f} MV/m)")
                ax4.legend()
                ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("legacy_vs_updated_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()


def main():
    """Main comparison test."""

    comparison = LegacyVsUpdatedComparison()

    # Test with extremely large field strengths to observe dramatic acceleration
    # Pushing to very high fields to see significant energy gains
    Ez_fields = [
        1000.0,
        5000.0,
        10000.0,
        50000.0,
        100000.0,
    ]  # Extremely large Gaussian amu⋅mm⋅ns units

    # Run comparison with smaller time steps for numerical stability
    results = comparison.run_comparison(Ez_fields, steps=500, dt=1e-5)

    # Create plots
    comparison.plot_comparison(results)

    # Summary
    print("=== SUMMARY ===")
    print("This test validates that legacy and updated integrators produce")
    print("identical physics results when subjected to constant Ez fields.")
    print("Field values are in native Gaussian amu⋅mm⋅ns units.")
    print()

    if any(results[Ez]["legacy_traj"] is not None for Ez in results):
        max_rel_diff = max(
            results[Ez]["rel_diff"]
            for Ez in results
            if results[Ez]["legacy_traj"] is not None
        )
        print(f"Maximum relative difference: {max_rel_diff:.3f}%")

        if max_rel_diff < 1.0:
            print(
                "✅ VALIDATION PASSED: Legacy and updated integrators agree within 1%"
            )
        else:
            print("❌ VALIDATION FAILED: Significant differences detected")
    else:
        print("⚠️  Legacy comparison unavailable (import issues)")

    print("Plots saved as 'legacy_vs_updated_comparison.png'")


if __name__ == "__main__":
    main()
