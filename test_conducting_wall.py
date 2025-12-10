#!/usr/bin/env python3
"""
Test script for conducting wall (ELECTRON_WALL) mode.

Runs a simulation using the provided configuration and analyzes:
1. Energy gain/loss behavior
2. Field interactions with the conducting wall
3. Trajectory and momentum evolution
4. Comparison with expected physics

Usage:
    python test_conducting_wall.py [config_path]
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add parent to path if needed
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from core.types import SimulationType
from examples.validation.core_vs_legacy_benchmark import run_benchmark
from optimization.metrics import compute_delta_energy_components


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def analyze_energy_evolution(trajectory: list) -> dict:
    """Analyze energy evolution throughout trajectory."""
    # Extract arrays from list of states
    gamma = np.array([np.mean(state["gamma"]) for state in trajectory])
    pz = np.array([np.mean(state["Pz"]) for state in trajectory])
    px = np.array([np.mean(state["Px"]) for state in trajectory])
    py = np.array([np.mean(state["Py"]) for state in trajectory])
    z = np.array([np.mean(state["z"]) for state in trajectory])
    x = np.array([np.mean(state["x"]) for state in trajectory])
    y = np.array([np.mean(state["y"]) for state in trajectory])

    # Electron rest mass in GeV
    m_e = 0.000510999  # GeV/c²

    # Total energy from gamma
    E_total = gamma * m_e  # GeV

    # Longitudinal momentum magnitude
    p_long = np.abs(pz)

    # Transverse momentum magnitude
    p_trans = np.sqrt(px**2 + py**2)

    # Total momentum magnitude
    p_total = np.sqrt(px**2 + py**2 + pz**2)

    # Transverse position
    r_trans = np.sqrt(x**2 + y**2)

    analysis = {
        "initial_gamma": gamma[0],
        "final_gamma": gamma[-1],
        "initial_E_total_GeV": E_total[0],
        "final_E_total_GeV": E_total[-1],
        "delta_E_total_GeV": E_total[-1] - E_total[0],
        "delta_E_total_eV": (E_total[-1] - E_total[0]) * 1e9,
        "initial_z_mm": z[0],
        "final_z_mm": z[-1],
        "distance_traveled_mm": z[-1] - z[0],
        "initial_Pz": pz[0],
        "final_Pz": pz[-1],
        "delta_Pz": pz[-1] - pz[0],
        "initial_p_trans": p_trans[0],
        "final_p_trans": p_trans[-1],
        "max_p_trans": np.max(p_trans),
        "mean_p_trans": np.mean(p_trans),
        "initial_r_trans_mm": r_trans[0],
        "final_r_trans_mm": r_trans[-1],
        "max_r_trans_mm": np.max(r_trans),
    }

    return analysis

    # Electron rest mass in GeV
    m_e = 0.000510999  # GeV/c²

    # Total energy from gamma
    E_total = gamma * m_e  # GeV

    # Longitudinal momentum magnitude
    p_long = np.abs(pz)

    # Transverse momentum magnitude
    p_trans = np.sqrt(px**2 + py**2)

    # Total momentum magnitude
    p_total = np.sqrt(px**2 + py**2 + pz**2)

    analysis = {
        "initial_gamma": gamma[0],
        "final_gamma": gamma[-1],
        "initial_E_total_GeV": E_total[0],
        "final_E_total_GeV": E_total[-1],
        "delta_E_total_GeV": E_total[-1] - E_total[0],
        "delta_E_total_eV": (E_total[-1] - E_total[0]) * 1e9,
        "initial_z_mm": z[0],
        "final_z_mm": z[-1],
        "distance_traveled_mm": z[-1] - z[0],
        "initial_Pz": pz[0],
        "final_Pz": pz[-1],
        "delta_Pz": pz[-1] - pz[0],
        "initial_p_trans": p_trans[0],
        "final_p_trans": p_trans[-1],
        "max_p_trans": np.max(p_trans),
        "mean_p_trans": np.mean(p_trans),
    }

    return analysis


def print_analysis(analysis: dict, config: dict):
    """Pretty-print the analysis results."""
    print("\n" + "=" * 70)
    print("CONDUCTING WALL SIMULATION ANALYSIS")
    print("=" * 70)

    print("\n### Configuration ###")
    print(f"Aperture radius: {config['core_params']['aperture_radius'] * 1000:.1f} μm")
    print(f"Wall position: {config['core_params']['wall_z']:.2f} mm")
    print(f"Time step: {config['core_params']['time_step']:.2e} mm/c")
    print(f"Steps: {config['steps']}")
    print(f"Image subcharges: {config['image_subcharge_count']}")
    print(f"Self-consistency: {config['self_consistency_enabled']}")
    print(f"Adaptive timestep: {config['adaptive_timestep_enabled']}")

    print("\n### Initial Particle State ###")
    rider = config["rider_params"]
    print(
        f"Starting z: {rider['starting_distance']:.6e} mm (distance to wall: {config['core_params']['wall_z'] - rider['starting_distance']:.2f} mm)"
    )
    print(f"Starting Pz: {rider['starting_Pz']:.2e} MeV/c")
    print(f"Transverse momentum: {rider['transv_mom']:.2e} MeV/c")
    print(
        f"Transverse offset: {rider['transv_dist']:.6e} mm ({rider['transv_dist'] * 1000:.3f} μm)"
    )
    print(f"Particle count: {rider['pcount']}")

    print("\n### Energy Evolution ###")
    print(f"Initial γ: {analysis['initial_gamma']:.6f}")
    print(f"Final γ: {analysis['final_gamma']:.6f}")
    print(f"Δγ: {analysis['final_gamma'] - analysis['initial_gamma']:.6e}")
    print(f"\nInitial E_total: {analysis['initial_E_total_GeV']:.9f} GeV")
    print(f"Final E_total: {analysis['final_E_total_GeV']:.9f} GeV")
    print(f"ΔE_total: {analysis['delta_E_total_GeV']:.6e} GeV")
    print(f"ΔE_total: {analysis['delta_E_total_eV']:.3f} eV")

    print("\n### Trajectory ###")
    print(f"Initial z: {analysis['initial_z_mm']:.6e} mm")
    print(f"Final z: {analysis['final_z_mm']:.6f} mm")
    print(f"Distance traveled: {analysis['distance_traveled_mm']:.6f} mm")

    print("\n### Momentum ###")
    print(f"Initial Pz: {analysis['initial_Pz']:.6e} MeV/c")
    print(f"Final Pz: {analysis['final_Pz']:.6e} MeV/c")
    print(f"ΔPz: {analysis['delta_Pz']:.6e} MeV/c")
    print(f"\nInitial p_trans: {analysis['initial_p_trans']:.6e} MeV/c")
    print(f"Final p_trans: {analysis['final_p_trans']:.6e} MeV/c")
    print(f"Max p_trans: {analysis['max_p_trans']:.6e} MeV/c")
    print(f"Mean p_trans: {analysis['mean_p_trans']:.6e} MeV/c")

    print("\n### Transverse Position ###")
    print(
        f"Initial r_trans: {analysis['initial_r_trans_mm']:.6e} mm ({analysis['initial_r_trans_mm'] * 1000:.3f} μm)"
    )
    print(
        f"Final r_trans: {analysis['final_r_trans_mm']:.6e} mm ({analysis['final_r_trans_mm'] * 1000:.3f} μm)"
    )
    print(
        f"Max r_trans: {analysis['max_r_trans_mm']:.6e} mm ({analysis['max_r_trans_mm'] * 1000:.3f} μm)"
    )

    aperture_um = config["core_params"]["aperture_radius"] * 1000
    r_over_a = (
        (analysis["initial_r_trans_mm"] * 1000) / aperture_um if aperture_um > 0 else 0
    )
    print(
        f"\nr/a ratio (initial): {r_over_a:.6f} (r={analysis['initial_r_trans_mm'] * 1000:.3f} μm, a={aperture_um:.1f} μm)"
    )

    print("\n" + "=" * 70)


def check_physics_consistency(analysis: dict, config: dict):
    """Check if results are physically consistent."""
    print("\n### Physics Consistency Checks ###")

    warnings = []

    # Check if particle reached the wall
    wall_z = config["core_params"]["wall_z"]
    final_z = analysis["final_z_mm"]
    initial_z = analysis["initial_z_mm"]

    if final_z < wall_z * 0.9:
        warnings.append(
            f"Particle only reached {final_z:.2f} mm, did not pass through wall at {wall_z:.2f} mm"
        )

    # Check energy scale
    delta_E_eV = analysis["delta_E_total_eV"]
    initial_E_GeV = analysis["initial_E_total_GeV"]

    if abs(delta_E_eV) < 10.0:
        warnings.append(
            f"Energy change ({delta_E_eV:.3f} eV) is very small for a {initial_E_GeV:.1f} GeV particle"
        )

    # Check transverse offset vs aperture
    aperture_mm = config["core_params"]["aperture_radius"]
    transv_dist_mm = config["rider_params"]["transv_dist"]
    r_over_a = transv_dist_mm / aperture_mm if aperture_mm > 0 else 0

    if r_over_a < 0.01:
        warnings.append(
            f"Transverse offset ({transv_dist_mm * 1e6:.1f} nm) is extremely small compared to aperture ({aperture_mm * 1000:.1f} μm), r/a = {r_over_a:.6f}"
        )

    # Check distance to wall
    distance_to_wall = wall_z - initial_z
    aperture_radii_to_wall = distance_to_wall / aperture_mm if aperture_mm > 0 else 0

    if aperture_radii_to_wall > 1000:
        warnings.append(
            f"Starting position is very far from wall ({distance_to_wall:.1f} mm = {aperture_radii_to_wall:.0f} aperture radii). Most trajectory is in free space with weak image charge interaction."
        )

    # Check step resolution near aperture
    steps = config["steps"]
    avg_step_size = distance_to_wall / steps if steps > 0 else 0
    near_wall_region = aperture_mm * 5  # Define "near wall" as ~5 aperture radii
    steps_in_near_wall = near_wall_region / avg_step_size if avg_step_size > 0 else 0

    if steps_in_near_wall < 50:
        warnings.append(
            f"Only ~{steps_in_near_wall:.0f} steps in critical near-wall region ({near_wall_region * 1000:.1f} μm). May need finer resolution or more steps."
        )

    if warnings:
        print("\n⚠️  WARNINGS:")
        for i, warning in enumerate(warnings, 1):
            print(f"{i}. {warning}")
    else:
        print("✓ No obvious physics inconsistencies detected")

    print()


def main():
    """Run conducting wall test."""
    # Default config path
    default_config = "configs/testbed_runs/electronwallv11_60micron.json"
    config_path = sys.argv[1] if len(sys.argv) > 1 else default_config

    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Verify it's a conducting wall simulation
    if config["simulation_type"] != "CONDUCTING_WALL":
        print(f"ERROR: Expected CONDUCTING_WALL, got {config['simulation_type']}")
        return 1

    print("Running simulation...")

    # Run the simulation - returns (metrics, payload)
    metrics, payload = run_benchmark(
        steps=config["steps"],
        simulation_type=SimulationType.CONDUCTING_WALL,
        rider_params=config["rider_params"],
        driver_params=None,
        seed=config["seed"],
        legacy_enabled=False,
        return_trajectories=True,
        image_subcharge_count=config["image_subcharge_count"],
        use_image_weighting=config["use_image_weighting"],
        self_consistency_enabled=config["self_consistency_enabled"],
        self_consistency_tolerance=config["self_consistency_tolerance"],
        self_consistency_max_iterations=config["self_consistency_max_iterations"],
        self_consistency_verbosity=0,  # Force quiet mode for analysis
        energy_monitor_enabled=config["energy_monitor_enabled"],
        energy_monitor_threshold=config["energy_monitor_threshold"],
        energy_monitor_check_interval=config["energy_monitor_check_interval"],
        energy_monitor_halt_on_jump=config.get("energy_monitor_halt_on_jump", False),
        energy_monitor_debug=config.get("energy_monitor_debug", False),
        adaptive_timestep_enabled=config["adaptive_timestep_enabled"],
        adaptive_timestep_threshold=config["adaptive_timestep_threshold"],
        adaptive_timestep_reduction_factor=config.get(
            "adaptive_timestep_reduction_factor", 10
        ),
        adaptive_timestep_max_attempts=config.get("adaptive_timestep_max_attempts", 5),
        adaptive_timestep_min_factor=config.get("adaptive_timestep_min_factor", 0.01),
        adaptive_timestep_cooldown_steps=config.get(
            "adaptive_timestep_cooldown_steps", 10
        ),
        adaptive_timestep_probe_threshold=config.get(
            "adaptive_timestep_probe_threshold", 0.01
        ),
        adaptive_timestep_max_probe_steps=config.get(
            "adaptive_timestep_max_probe_steps", 3
        ),
        adaptive_timestep_debug=config.get("adaptive_timestep_debug", False),
        **{k: v for k, v in config["core_params"].items() if k != "time_step"},
    )

    print("✓ Simulation complete")

    # Extract trajectory from payload
    if "core" in payload and "rider" in payload["core"]:
        trajectory = payload["core"]["rider"]
    else:
        print(f"ERROR: Unexpected payload structure. Keys: {payload.keys()}")
        return 1

    # Analyze results
    analysis = analyze_energy_evolution(trajectory)

    # Print detailed analysis
    print_analysis(analysis, config)

    # Check physics consistency
    check_physics_consistency(analysis, config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
