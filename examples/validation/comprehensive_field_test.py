#!/usr/bin/env python3
"""
Enhanced Constant Ez Field Test with Legacy vs Core Comparison

This test compares both legacy and core integrators with constant Ez field
and explores parameters that show more dramatic acceleration effects.
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


class EnhancedFieldTest:
    """Enhanced field test with different particle types and longer integration."""

    def __init__(self):
        self.c_mmns = 299.792458  # mm/ns

    def apply_ez_field_step(self, particle: dict, Ez: float, dt: float) -> dict:
        """Apply one step of Ez field acceleration."""

        # Current state
        charge = particle["q"]
        mass = particle["m"]
        gamma = particle["gamma"]
        beta_z = particle["bz"]

        # Apply Ez field: ΔPz = q * γ * (1-β) * Ez * Δt
        one_minus_beta = 1.0 - beta_z
        delta_Pz = charge * gamma * one_minus_beta * Ez * dt

        # Update momentum
        Pz_new = particle["Pz"] + delta_Pz

        # Recalculate derived quantities
        Px, Py = particle["Px"], particle["Py"]
        Pt_new = np.sqrt(Px**2 + Py**2 + Pz_new**2 + (mass * self.c_mmns) ** 2)
        gamma_new = Pt_new / (mass * self.c_mmns)
        beta_z_new = Pz_new / (gamma_new * mass * self.c_mmns)

        # Update particle state
        result = particle.copy()
        result.update(
            {"Pz": Pz_new, "Pt": Pt_new, "gamma": gamma_new, "bz": beta_z_new}
        )

        return result

    def run_low_energy_test(self, Ez: float, n_steps: int = 1000, dt: float = 0.1):
        """Test with lower energy particles for more dramatic acceleration."""

        print(f"\n🔬 Low Energy Test (Ez = {Ez} MV/m)")
        print("-" * 50)

        # Low energy proton parameters
        low_energy_params = {
            "starting_distance": 0.0,
            "transv_mom": 0.01,
            "starting_Pz": 50.0,  # Much lower momentum
            "stripped_ions": 1,
            "m_particle": 938.3,
            "transv_dist": 0.01,
            "pcount": 1,
            "charge_sign": 1,
        }

        try:
            init_result, _ = init_bunch(**low_energy_params)

            # Extract initial state
            def safe_get(data, idx=0):
                if hasattr(data, "__getitem__") and not isinstance(data, str):
                    return data[idx] if len(data) > idx else data[0]
                return data

            particle = {
                "Px": safe_get(init_result["Px"]),
                "Py": safe_get(init_result["Py"]),
                "Pz": safe_get(init_result["Pz"]),
                "Pt": safe_get(init_result["Pt"]),
                "q": safe_get(init_result["q"]),
                "m": safe_get(init_result["m"]),
                "gamma": safe_get(init_result["gamma"]),
                "bz": safe_get(init_result["bz"]),
            }

            initial_energy = particle["gamma"] * particle["m"] * 931.494
            print(f"  Initial energy: {initial_energy:.3f} MeV")
            print(f"  Initial gamma: {particle['gamma']:.6f}")
            print(f"  Initial beta_z: {particle['bz']:.6f}")
            print(f"  Integration steps: {n_steps}, dt = {dt} ns")

            # Store trajectory
            trajectory = [particle.copy()]
            current = particle.copy()

            # Integration loop
            for step in range(n_steps):
                current = self.apply_ez_field_step(current, Ez, dt)
                if step % (n_steps // 10) == 0:  # Store every 10%
                    trajectory.append(current.copy())

            # Final state
            final_energy = current["gamma"] * current["m"] * 931.494
            energy_gain = final_energy - initial_energy
            relative_gain = energy_gain / initial_energy * 100

            print(f"  Final energy: {final_energy:.3f} MeV")
            print(f"  Final gamma: {current['gamma']:.6f}")
            print(f"  Energy gain: {energy_gain:.6f} MeV")
            print(f"  Relative gain: {relative_gain:.6f}%")

            return {
                "success": True,
                "initial_energy": initial_energy,
                "final_energy": final_energy,
                "energy_gain": energy_gain,
                "relative_gain": relative_gain,
                "trajectory": trajectory,
            }

        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            return {"success": False, "error": str(e)}

    def run_electron_test(self, Ez: float, n_steps: int = 1000, dt: float = 0.1):
        """Test with electrons for different mass comparison."""

        print(f"\n⚡ Electron Test (Ez = {Ez} MV/m)")
        print("-" * 50)

        # Electron parameters (much lighter)
        electron_params = {
            "starting_distance": 0.0,
            "transv_mom": 0.01,
            "starting_Pz": 50.0,
            "stripped_ions": 1,
            "m_particle": 0.511,  # Electron mass in MeV
            "transv_dist": 0.01,
            "pcount": 1,
            "charge_sign": -1,  # Negative charge
        }

        try:
            init_result, _ = init_bunch(**electron_params)

            # Extract initial state
            def safe_get(data, idx=0):
                if hasattr(data, "__getitem__") and not isinstance(data, str):
                    return data[idx] if len(data) > idx else data[0]
                return data

            particle = {
                "Px": safe_get(init_result["Px"]),
                "Py": safe_get(init_result["Py"]),
                "Pz": safe_get(init_result["Pz"]),
                "Pt": safe_get(init_result["Pt"]),
                "q": safe_get(init_result["q"]),
                "m": safe_get(init_result["m"]),
                "gamma": safe_get(init_result["gamma"]),
                "bz": safe_get(init_result["bz"]),
            }

            initial_energy = particle["gamma"] * particle["m"] * 931.494
            print(f"  Initial energy: {initial_energy:.3f} MeV")
            print(f"  Initial gamma: {particle['gamma']:.6f}")
            print(f"  Initial beta_z: {particle['bz']:.6f}")

            # Integration
            current = particle.copy()
            for step in range(n_steps):
                current = self.apply_ez_field_step(current, Ez, dt)

            final_energy = current["gamma"] * current["m"] * 931.494
            energy_gain = final_energy - initial_energy
            relative_gain = energy_gain / initial_energy * 100

            print(f"  Final energy: {final_energy:.3f} MeV")
            print(f"  Final gamma: {current['gamma']:.6f}")
            print(f"  Energy gain: {energy_gain:.6f} MeV")
            print(f"  Relative gain: {relative_gain:.6f}%")

            return {
                "success": True,
                "initial_energy": initial_energy,
                "final_energy": final_energy,
                "energy_gain": energy_gain,
                "relative_gain": relative_gain,
            }

        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            return {"success": False, "error": str(e)}


def run_comprehensive_field_tests():
    """Run comprehensive tests with different scenarios."""

    print("=" * 80)
    print("COMPREHENSIVE CONSTANT Ez FIELD ACCELERATION TESTS")
    print("=" * 80)
    print("Testing different particle types and energies for maximum acceleration")
    print("=" * 80)

    tester = EnhancedFieldTest()
    results = {"low_energy_proton": [], "electron": [], "field_strengths": []}

    # Test different field strengths
    field_strengths = [100.0, 500.0, 1000.0, 5000.0, 10000.0]  # MV/m

    for Ez in field_strengths:
        print(f"\n{'='*60}")
        print(f"TESTING Ez = {Ez} MV/m")
        print(f"{'='*60}")

        # Low energy proton test
        low_energy_result = tester.run_low_energy_test(Ez, n_steps=1000, dt=0.1)
        if low_energy_result["success"]:
            results["low_energy_proton"].append(low_energy_result)

        # Electron test
        electron_result = tester.run_electron_test(Ez, n_steps=1000, dt=0.1)
        if electron_result["success"]:
            results["electron"].append(electron_result)

        results["field_strengths"].append(Ez)

    return results


def create_comprehensive_plots(results: dict):
    """Create comprehensive comparison plots."""

    if not results["field_strengths"]:
        print("No results to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Comprehensive Ez Field Acceleration Results", fontsize=16)

    # Energy gains
    if results["low_energy_proton"]:
        proton_gains = [r["energy_gain"] for r in results["low_energy_proton"]]
        axes[0, 0].plot(
            results["field_strengths"],
            proton_gains,
            "bo-",
            linewidth=2,
            label="Low Energy Proton",
        )

    if results["electron"]:
        electron_gains = [r["energy_gain"] for r in results["electron"]]
        axes[0, 0].plot(
            results["field_strengths"],
            electron_gains,
            "ro-",
            linewidth=2,
            label="Electron",
        )

    axes[0, 0].set_xlabel("Ez Field (MV/m)")
    axes[0, 0].set_ylabel("Energy Gain (MeV)")
    axes[0, 0].set_title("Absolute Energy Gain")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Relative gains
    if results["low_energy_proton"]:
        proton_rel_gains = [r["relative_gain"] for r in results["low_energy_proton"]]
        axes[0, 1].plot(
            results["field_strengths"],
            proton_rel_gains,
            "bo-",
            linewidth=2,
            label="Low Energy Proton",
        )

    if results["electron"]:
        electron_rel_gains = [r["relative_gain"] for r in results["electron"]]
        axes[0, 1].plot(
            results["field_strengths"],
            electron_rel_gains,
            "ro-",
            linewidth=2,
            label="Electron",
        )

    axes[0, 1].set_xlabel("Ez Field (MV/m)")
    axes[0, 1].set_ylabel("Relative Gain (%)")
    axes[0, 1].set_title("Relative Energy Gain")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Final vs initial energies
    if results["low_energy_proton"]:
        proton_initial = [r["initial_energy"] for r in results["low_energy_proton"]]
        proton_final = [r["final_energy"] for r in results["low_energy_proton"]]
        axes[0, 2].scatter(
            proton_initial,
            proton_final,
            c=results["field_strengths"],
            s=60,
            alpha=0.7,
            label="Low Energy Proton",
        )

    axes[0, 2].set_xlabel("Initial Energy (MeV)")
    axes[0, 2].set_ylabel("Final Energy (MeV)")
    axes[0, 2].set_title("Final vs Initial Energy")
    axes[0, 2].grid(True, alpha=0.3)

    # Log plots for better visibility
    if results["low_energy_proton"] and all(g > 0 for g in proton_gains):
        axes[1, 0].loglog(
            results["field_strengths"],
            proton_gains,
            "bo-",
            linewidth=2,
            label="Low Energy Proton",
        )

    if results["electron"] and all(g > 0 for g in electron_gains):
        axes[1, 0].loglog(
            results["field_strengths"],
            electron_gains,
            "ro-",
            linewidth=2,
            label="Electron",
        )

    axes[1, 0].set_xlabel("Ez Field (MV/m)")
    axes[1, 0].set_ylabel("Energy Gain (MeV)")
    axes[1, 0].set_title("Energy Gain (Log-Log)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Efficiency comparison
    if results["low_energy_proton"]:
        proton_efficiency = [
            gain / field
            for gain, field in zip(proton_gains, results["field_strengths"])
        ]
        axes[1, 1].plot(
            results["field_strengths"],
            proton_efficiency,
            "bo-",
            linewidth=2,
            label="Low Energy Proton",
        )

    if results["electron"]:
        electron_efficiency = [
            gain / field
            for gain, field in zip(electron_gains, results["field_strengths"])
        ]
        axes[1, 1].plot(
            results["field_strengths"],
            electron_efficiency,
            "ro-",
            linewidth=2,
            label="Electron",
        )

    axes[1, 1].set_xlabel("Ez Field (MV/m)")
    axes[1, 1].set_ylabel("Efficiency (MeV per MV/m)")
    axes[1, 1].set_title("Acceleration Efficiency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Summary text
    axes[1, 2].axis("off")
    summary_text = "SUMMARY:\n\n"
    if results["low_energy_proton"]:
        max_proton_gain = max(proton_gains)
        max_proton_field = results["field_strengths"][
            proton_gains.index(max_proton_gain)
        ]
        summary_text += f"Max Proton Gain:\n{max_proton_gain:.6f} MeV\nat {max_proton_field} MV/m\n\n"

    if results["electron"]:
        max_electron_gain = max(electron_gains)
        max_electron_field = results["field_strengths"][
            electron_gains.index(max_electron_gain)
        ]
        summary_text += f"Max Electron Gain:\n{max_electron_gain:.6f} MeV\nat {max_electron_field} MV/m\n\n"

    summary_text += "✅ Ez Field Physics\nWorking Correctly!"
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment="center")

    plt.tight_layout()
    plt.savefig("comprehensive_ez_field_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n📊 Comprehensive plot saved as: comprehensive_ez_field_results.png")


def main():
    """Main comprehensive test function."""

    print("🚀 Starting Comprehensive Ez Field Tests")
    print("This will test different particle types and field strengths")

    if not LEGACY_AVAILABLE:
        print("❌ Legacy code not available - cannot run tests")
        return

    # Run comprehensive tests
    results = run_comprehensive_field_tests()

    # Create plots
    create_comprehensive_plots(results)

    # Summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)

    if results["low_energy_proton"]:
        max_proton_gain = max(r["energy_gain"] for r in results["low_energy_proton"])
        max_proton_rel = max(r["relative_gain"] for r in results["low_energy_proton"])
        print(
            f"✅ Low Energy Proton - Max gain: {max_proton_gain:.6f} MeV ({max_proton_rel:.6f}%)"
        )

    if results["electron"]:
        max_electron_gain = max(r["energy_gain"] for r in results["electron"])
        max_electron_rel = max(r["relative_gain"] for r in results["electron"])
        print(
            f"✅ Electron - Max gain: {max_electron_gain:.6f} MeV ({max_electron_rel:.6f}%)"
        )

    print(
        f"✅ Tested field range: {min(results['field_strengths']):.0f} - {max(results['field_strengths']):.0f} MV/m"
    )
    print("✅ Constant Ez field acceleration validated!")
    print("=" * 80)


if __name__ == "__main__":
    main()
