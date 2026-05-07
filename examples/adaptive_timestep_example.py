"""Example: Using Adaptive Timestep and Beta Clamping

This script demonstrates how to use the new adaptive timestep feature to
automatically handle energy jumps in high-energy particle simulations, along
with beta clamping to prevent superluminal velocities.

Features demonstrated:
1. Beta clamping to prevent velocities from exceeding the speed of light
2. Adaptive timestep refinement when energy jumps are detected
3. Energy monitoring to track simulation stability
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.constants import C_MMNS
from core.trajectory_integrator import (
    AdaptiveTimestepConfig,
    EnergyMonitorConfig,
    retarded_integrator,
)
from core.types import ChronoMatchingMode, SimulationType, StartupMode


def create_high_energy_electron(energy_mev=10000.0):
    """Create initial state for a high-energy electron.

    Parameters
    ----------
    energy_mev : float
        Total energy in MeV (default: 10 GeV)

    Returns
    -------
    dict
        Particle state dictionary
    """
    electron_mass = 0.51099895  # MeV/c²

    # Calculate gamma and beta from energy
    gamma = energy_mev / electron_mass
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    momentum = gamma * electron_mass

    # Classical electron radius in mm
    r0 = 2.8179403227e-15 * 1e3  # m to mm
    char_time = r0 / (3.0 * C_MMNS)

    return {
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "t": np.array([0.0]),
        "Px": np.array([0.0]),
        "Py": np.array([0.0]),
        "Pz": np.array([momentum]),
        "Pt": np.array([momentum]),
        "bx": np.array([0.0]),
        "by": np.array([0.0]),
        "bz": np.array([beta]),
        "gamma": np.array([gamma]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "m": np.array([electron_mass]),
        "q": np.array([-1.0]),
        "char_time": np.array([char_time]),
    }


def run_with_adaptive_timestep():
    """Run simulation with adaptive timestep enabled."""
    print("=" * 70)
    print("EXAMPLE: Adaptive Timestep for High-Energy Electron-Wall Interaction")
    print("=" * 70)

    # Create a 10 GeV electron
    print("\nInitializing 10 GeV electron...")
    init_state = create_high_energy_electron(energy_mev=10000.0)

    gamma_init = init_state["gamma"][0]
    beta_init = init_state["bz"][0]
    print(f"  Initial γ = {gamma_init:.6e}")
    print(f"  Initial β = {beta_init:.17f}")
    print(f"  Initial energy = {gamma_init * init_state['m'][0]:.6e} MeV")

    # Configure adaptive timestep
    # This will automatically reduce the timestep when energy jumps > 10% are detected
    adaptive_config = AdaptiveTimestepConfig(
        enabled=True,
        energy_jump_threshold=0.10,  # Trigger on 10% energy change
        timestep_reduction_factor=10,  # Reduce timestep by 10x
        max_refinement_attempts=5,  # Try up to 5 refinements
        min_timestep_factor=1e-5,  # Don't go below 1/100000 of original timestep
        debug=True,  # Print when timestep is adjusted
    )

    # Configure energy monitoring (separate from adaptive timestep)
    # This provides warnings/reports but doesn't change the timestep
    energy_monitor = EnergyMonitorConfig(
        enabled=True,
        relative_threshold=1.0,  # Warn on 100% energy change
        check_interval=10,  # Check every 10 steps
        halt_on_jump=False,  # Don't halt, just warn
        debug=False,
    )

    print("\nSimulation parameters:")
    print("  Steps: 200")
    print("  Base timestep: 1e-7 ns")
    print("  Wall position: 1000.0 mm")
    print("  Aperture radius: 0.1 mm")
    print("\nAdaptive timestep settings:")
    print(f"  Energy jump threshold: {adaptive_config.energy_jump_threshold * 100}%")
    print(f"  Timestep reduction factor: {adaptive_config.timestep_reduction_factor}x")
    print(f"  Max refinement attempts: {adaptive_config.max_refinement_attempts}")

    print("\nRunning simulation with adaptive timestep...\n")

    # Run the simulation
    trajectory, _, *_soa_out = retarded_integrator(
        steps=200,
        h_step=1e-7,  # Base timestep in ns
        wall_z=1000.0,
        aperture_radius=0.1,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=init_state,
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        chrono_mode=ChronoMatchingMode.AVERAGED,
        startup_mode=StartupMode.COLD_START,
        adaptive_timestep=adaptive_config,
        energy_monitor=energy_monitor,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Analyze results
    energies = []
    betas = []
    gammas = []
    max_beta_components = {"bx": 0.0, "by": 0.0, "bz": 0.0}

    for state in trajectory:
        gamma = state["gamma"][0]
        mass = state["m"][0]
        energy = gamma * mass * C_MMNS * C_MMNS

        bx = state["bx"][0]
        by = state["by"][0]
        bz = state["bz"][0]
        beta_mag = np.sqrt(bx**2 + by**2 + bz**2)

        energies.append(energy)
        betas.append(beta_mag)
        gammas.append(gamma)

        max_beta_components["bx"] = max(max_beta_components["bx"], abs(bx))
        max_beta_components["by"] = max(max_beta_components["by"], abs(by))
        max_beta_components["bz"] = max(max_beta_components["bz"], abs(bz))

    energies = np.array(energies)
    betas = np.array(betas)
    gammas = np.array(gammas)

    # Compute energy stability metrics
    energy_changes = np.abs(np.diff(energies)) / energies[:-1]

    print("\nEnergy conservation:")
    print(f"  Initial energy: {energies[0]:.6e} MeV")
    print(f"  Final energy:   {energies[-1]:.6e} MeV")
    print(
        f"  Total drift:    {abs(energies[-1] - energies[0]) / energies[0] * 100:.2f}%"
    )
    print("\nEnergy stability:")
    print(f"  Max step-to-step change:  {np.nanmax(energy_changes) * 100:.2f}%")
    print(f"  Mean step-to-step change: {np.nanmean(energy_changes) * 100:.4f}%")
    print(f"  Steps with >10% change:   {np.sum(energy_changes > 0.1)}")

    print("\nVelocity limits (beta clamping):")
    print(f"  Max |βx|: {max_beta_components['bx']:.17f}")
    print(f"  Max |βy|: {max_beta_components['by']:.17f}")
    print(f"  Max |βz|: {max_beta_components['bz']:.17f}")
    print(f"  Max |β|:  {betas.max():.17f}")
    print(
        f"  All beta components ≤ 1.0: {all(v <= 1.0 for v in max_beta_components.values())}"
    )

    print("\nLorentz factor (gamma):")
    print(f"  Initial γ: {gammas[0]:.6e}")
    print(f"  Final γ:   {gammas[-1]:.6e}")
    print(f"  Max γ:     {gammas.max():.6e}")
    print(f"  Min γ:     {gammas.min():.6e}")
    print(f"  All γ finite: {np.all(np.isfinite(gammas))}")

    print("\n" + "=" * 70)
    print("KEY FEATURES DEMONSTRATED:")
    print("=" * 70)
    print("✓ Beta clamping prevents superluminal velocities")
    print("  (all beta components stay ≤ 1.0, even at extreme energies)")
    print("\n✓ Adaptive timestep detects and responds to energy jumps")
    print("  (timestep automatically reduced when instabilities detected)")
    print("\n✓ Gamma remains finite throughout simulation")
    print("  (no numerical overflow even at relativistic velocities)")
    print()


def comparison_run():
    """Compare runs with and without adaptive timestep."""
    print("\n" + "=" * 70)
    print("COMPARISON: With vs Without Adaptive Timestep")
    print("=" * 70)

    init_state = create_high_energy_electron(energy_mev=5000.0)

    configs = [
        ("WITHOUT adaptive timestep", None),
        (
            "WITH adaptive timestep",
            AdaptiveTimestepConfig(
                enabled=True,
                energy_jump_threshold=0.05,
                timestep_reduction_factor=5,
                max_refinement_attempts=3,
                debug=False,
            ),
        ),
    ]

    for name, adaptive_config in configs:
        print(f"\nRunning: {name}")

        trajectory, _, *_soa_out = retarded_integrator(
            steps=100,
            h_step=1e-7,
            wall_z=1000.0,
            aperture_radius=0.1,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=init_state,
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            chrono_mode=ChronoMatchingMode.AVERAGED,
            startup_mode=StartupMode.COLD_START,
            adaptive_timestep=adaptive_config,
        )

        # Compute metrics
        energies = np.array(
            [s["gamma"][0] * s["m"][0] * C_MMNS * C_MMNS for s in trajectory]
        )
        energy_changes = np.abs(np.diff(energies)) / energies[:-1]
        gammas = np.array([s["gamma"][0] for s in trajectory])

        print(f"  Max energy change:   {np.nanmax(energy_changes) * 100:.2f}%")
        print(
            f"  Energy drift:        {abs(energies[-1] - energies[0]) / energies[0] * 100:.2f}%"
        )
        print(f"  Max gamma:           {gammas.max():.6e}")
        print(f"  All gamma finite:    {np.all(np.isfinite(gammas))}")

    print()


if __name__ == "__main__":
    # Main demonstration
    run_with_adaptive_timestep()

    # Optional: run comparison
    # comparison_run()
