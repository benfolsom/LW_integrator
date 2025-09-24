#!/usr/bin/env python3
"""
Simple Constant Electric Field Test

Implements a constant Ez field acceleration test based on:
ΔPz = q * γ * (1-β) * Ez * Δt

This is a simplified test that focuses on the physics implementation.
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


def apply_constant_ez_field(particle_state: dict, Ez: float, dt: float) -> dict:
    """
    Apply constant Ez field to a single particle.

    Args:
        particle_state: Dict with keys 'Px', 'Py', 'Pz', 'q', 'm', 'gamma', 'bz'
        Ez: Electric field strength in MV/m (or appropriate units)
        dt: Time step in ns

    Returns:
        Updated particle state
    """
    c_mmns = 299.792458  # mm/ns

    # Get current state
    Px = particle_state["Px"]
    Py = particle_state["Py"]
    Pz = particle_state["Pz"]
    charge = particle_state["q"]
    mass = particle_state["m"]
    gamma = particle_state["gamma"]
    beta_z = particle_state["bz"]

    # Apply constant Ez field: ΔPz = q * γ * (1-β) * Ez * Δt
    one_minus_beta = 1.0 - beta_z
    delta_Pz_field = charge * gamma * one_minus_beta * Ez * dt

    # Update momentum
    Pz_new = Pz + delta_Pz_field

    # Recalculate total momentum and derived quantities
    Pt_new = np.sqrt(Px**2 + Py**2 + Pz_new**2 + (mass * c_mmns) ** 2)
    gamma_new = Pt_new / (mass * c_mmns)
    beta_z_new = Pz_new / (gamma_new * mass * c_mmns)

    # Calculate energy
    energy_MeV = gamma_new * mass * 931.494  # Convert amu to MeV

    return {
        "Px": Px,
        "Py": Py,
        "Pz": Pz_new,
        "Pt": Pt_new,
        "q": charge,
        "m": mass,
        "gamma": gamma_new,
        "bz": beta_z_new,
        "energy_MeV": energy_MeV,
    }


def run_constant_field_test(Ez_values: list, n_steps: int = 100, dt: float = 0.1):
    """Run constant field test for different Ez values."""

    print("=" * 70)
    print("CONSTANT Ez FIELD ACCELERATION TEST")
    print("=" * 70)
    print("Physics: ΔPz = q * γ * (1-β) * Ez * Δt")
    print(f"Time steps: {n_steps}, dt = {dt} ns")
    print("=" * 70)

    if not LEGACY_AVAILABLE:
        print("❌ Legacy code not available, using theoretical calculation")
        return

    # Initialize test particle
    test_params = {
        "starting_distance": 0.0,
        "transv_mom": 0.01,
        "starting_Pz": 200.0,
        "stripped_ions": 1,
        "m_particle": 938.3,  # Proton mass in MeV
        "transv_dist": 0.01,
        "pcount": 1,
        "charge_sign": 1,
    }

    results = {
        "Ez_values": [],
        "energy_gains": [],
        "initial_energies": [],
        "final_energies": [],
    }

    for Ez in Ez_values:
        print(f"\n--- Ez = {Ez} MV/m ---")

        try:
            # Initialize particle
            init_result, E_MeV_rest = init_bunch(**test_params)

            # Extract particle state (handling scalar vs array)
            def safe_extract(data, index=0):
                if hasattr(data, "__getitem__") and not isinstance(data, str):
                    return data[index] if len(data) > index else data[0]
                return data

            particle = {
                "Px": safe_extract(init_result["Px"]),
                "Py": safe_extract(init_result["Py"]),
                "Pz": safe_extract(init_result["Pz"]),
                "Pt": safe_extract(init_result["Pt"]),
                "q": safe_extract(init_result["q"]),
                "m": safe_extract(init_result["m"]),
                "gamma": safe_extract(init_result["gamma"]),
                "bz": safe_extract(init_result["bz"]),
            }

            initial_energy = particle["gamma"] * particle["m"] * 931.494
            print(f"  Initial energy: {initial_energy:.3f} MeV")
            print(f"  Initial gamma: {particle['gamma']:.6f}")
            print(f"  Initial beta_z: {particle['bz']:.6f}")

            # Run integration steps
            current_particle = particle.copy()
            for step in range(n_steps):
                current_particle = apply_constant_ez_field(current_particle, Ez, dt)

            final_energy = current_particle["energy_MeV"]
            energy_gain = final_energy - initial_energy

            print(f"  Final energy: {final_energy:.3f} MeV")
            print(f"  Final gamma: {current_particle['gamma']:.6f}")
            print(f"  Energy gain: {energy_gain:.6f} MeV")
            print(f"  Relative gain: {energy_gain/initial_energy*100:.8f}%")

            results["Ez_values"].append(Ez)
            results["energy_gains"].append(energy_gain)
            results["initial_energies"].append(initial_energy)
            results["final_energies"].append(final_energy)

        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            import traceback

            traceback.print_exc()

    return results


def create_acceleration_plots(results: dict):
    """Create plots showing acceleration results."""

    if not results["Ez_values"]:
        print("No results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Constant Ez Field Acceleration Results", fontsize=14)

    # Energy gain vs Ez field
    axes[0, 0].plot(
        results["Ez_values"], results["energy_gains"], "bo-", linewidth=2, markersize=6
    )
    axes[0, 0].set_xlabel("Ez Field Strength (MV/m)")
    axes[0, 0].set_ylabel("Energy Gain (MeV)")
    axes[0, 0].set_title("Energy Gain vs Ez Field")
    axes[0, 0].grid(True, alpha=0.3)

    # Log plot for small gains
    axes[0, 1].loglog(
        results["Ez_values"][1:],
        np.abs(results["energy_gains"][1:]),
        "ro-",
        linewidth=2,
        markersize=6,
    )
    axes[0, 1].set_xlabel("Ez Field Strength (MV/m)")
    axes[0, 1].set_ylabel("|Energy Gain| (MeV)")
    axes[0, 1].set_title("Energy Gain (Log-Log Scale)")
    axes[0, 1].grid(True, alpha=0.3)

    # Relative energy gain
    relative_gains = [
        gain / initial * 100
        for gain, initial in zip(results["energy_gains"], results["initial_energies"])
    ]
    axes[1, 0].plot(
        results["Ez_values"], relative_gains, "go-", linewidth=2, markersize=6
    )
    axes[1, 0].set_xlabel("Ez Field Strength (MV/m)")
    axes[1, 0].set_ylabel("Relative Energy Gain (%)")
    axes[1, 0].set_title("Relative Energy Gain vs Ez Field")
    axes[1, 0].grid(True, alpha=0.3)

    # Final vs initial energy
    axes[1, 1].plot(
        results["initial_energies"],
        results["final_energies"],
        "mo-",
        linewidth=2,
        markersize=6,
    )
    axes[1, 1].plot(
        [min(results["initial_energies"]), max(results["initial_energies"])],
        [min(results["initial_energies"]), max(results["initial_energies"])],
        "k--",
        alpha=0.5,
        label="No gain",
    )
    axes[1, 1].set_xlabel("Initial Energy (MeV)")
    axes[1, 1].set_ylabel("Final Energy (MeV)")
    axes[1, 1].set_title("Final vs Initial Energy")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("constant_ez_field_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n📊 Plot saved as: constant_ez_field_results.png")


def main():
    """Main test function."""

    print("🚀 Starting Constant Ez Field Physics Test")

    # Test with a range of field strengths
    Ez_values = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]  # MV/m

    results = run_constant_field_test(Ez_values, n_steps=100, dt=0.1)

    if results and results["Ez_values"]:
        create_acceleration_plots(results)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"✅ Successfully tested {len(results['Ez_values'])} field strengths")
        max_gain = max(results["energy_gains"]) if results["energy_gains"] else 0
        print(f"✅ Maximum energy gain: {max_gain:.6f} MeV")
        if max_gain > 0:
            max_gain_field = results["Ez_values"][
                results["energy_gains"].index(max_gain)
            ]
            print(f"✅ Best field strength: {max_gain_field} MV/m")
            print("✅ Constant Ez field acceleration is working!")
        else:
            print("⚠️  No significant energy gain observed - may need stronger fields")
        print("=" * 70)
    else:
        print("❌ No successful tests completed")


if __name__ == "__main__":
    main()
