"""Run optimization sweeps for LW integrator.

Usage:
    python run_optimization.py --mode quick
    python run_optimization.py --mode sweep
    python run_optimization.py --mode finetune --aperture 0.06 --energy 10.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "examples" / "validation"))

from examples.validation.core_vs_legacy_benchmark import (
    SimulationType,
    run_benchmark,
)
from optimization.metrics import compute_trajectory_metrics
from optimization.visualization import plot_energy_heatmap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_base_config():
    """Load base config from electronwallv11_60micron.json."""
    config_path = Path("configs/testbed_runs/electronwallv11_60micron.json")
    with open(config_path) as f:
        return json.load(f)


def create_rider_params(energy_gev, start_z_mm, aperture_z_mm=2200.0):
    """Create rider params for given energy and position.

    start_z_mm is distance BEFORE aperture (e.g., 10mm means 10mm before aperture).
    starting_distance is absolute z position.
    """
    # Electron mass in amu
    m_e = 0.00054857990907
    # Convert energy to gamma: E = gamma * m_e * c^2, m_e*c^2 = 0.511 MeV
    gamma = energy_gev * 1e3 / 0.511
    # For ultra-relativistic: Pz ≈ gamma * m_e * c (in amu*mm/ns units, c ≈ 299.792458)
    c_mmns = 299.792458
    Pz = gamma * m_e * c_mmns

    # starting_distance is absolute z position (aperture is at aperture_z_mm)
    # If we want particle to start start_z_mm BEFORE aperture, subtract
    starting_distance = aperture_z_mm - start_z_mm

    return {
        "starting_distance": starting_distance,
        "transv_mom": 1.2e-05,
        "starting_Pz": Pz,
        "stripped_ions": 1.0,
        "m_particle": m_e,
        "transv_dist": 2e-06,
        "pcount": 1,
        "charge_sign": -1.0,
    }


def run_single_sim(
    aperture_mm,
    energy_gev,
    start_z_mm,
    base_config,
    enable_sc=False,
    enable_adaptive=False,
):
    """Run single simulation."""
    # Create params
    rider_params = create_rider_params(energy_gev, start_z_mm)

    # Core params
    core_params = {
        "time_step": base_config["core_params"]["time_step"],
        "wall_z": base_config["core_params"]["wall_z"],
        "aperture_radius": aperture_mm,
    }

    # Run benchmark - SC doesn't consume extra steps
    # Keep steps modest - proximity-based adaptive timestep will refine near aperture
    steps = 2000
    result = run_benchmark(
        steps=steps,
        simulation_type=SimulationType.CONDUCTING_WALL,
        rider_params=rider_params,
        driver_params=None,
        seed=base_config["seed"],
        legacy_enabled=False,
        return_trajectories=True,
        image_subcharge_count=base_config["image_subcharge_count"] if enable_sc else 4,
        use_image_weighting=base_config["use_image_weighting"],
        self_consistency_enabled=enable_sc,
        self_consistency_tolerance=base_config["self_consistency_tolerance"],
        self_consistency_max_iterations=3 if enable_sc else 1,
        self_consistency_verbosity=0,
        energy_monitor_enabled=enable_adaptive,
        energy_monitor_threshold=base_config["energy_monitor_threshold"],
        energy_monitor_check_interval=base_config["energy_monitor_check_interval"],
        energy_monitor_halt_on_jump=False,
        energy_monitor_debug=False,
        adaptive_timestep_enabled=enable_adaptive,
        adaptive_timestep_threshold=base_config["adaptive_timestep_threshold"],
        adaptive_timestep_reduction_factor=base_config[
            "adaptive_timestep_reduction_factor"
        ],
        adaptive_timestep_max_attempts=base_config["adaptive_timestep_max_attempts"],
        adaptive_timestep_min_factor=base_config["adaptive_timestep_min_factor"],
        adaptive_timestep_cooldown_steps=base_config[
            "adaptive_timestep_cooldown_steps"
        ],
        adaptive_timestep_probe_threshold=base_config[
            "adaptive_timestep_probe_threshold"
        ],
        adaptive_timestep_max_probe_steps=base_config[
            "adaptive_timestep_max_probe_steps"
        ],
        adaptive_timestep_debug=False,
        **core_params,
    )

    metrics_dict, payload = result
    trajectory = payload["core"]["rider"]
    initial_state = payload["initial_states"]["rider"]

    # Compute metrics
    metrics = compute_trajectory_metrics(
        trajectory=trajectory,
        initial_state=initial_state,
        rest_energy_mev=0.511,
        aperture_z=core_params["wall_z"],
    )

    return metrics


def run_quick_sweep():
    """Quick 3x3x3 sweep."""
    logger.info("Running quick sweep")

    base_config = load_base_config()
    output_dir = Path("test_outputs/optimization/quick_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Small grid - focus on realistic interaction regions
    apertures = [0.001, 0.06, 1.0]  # 1 μm, 60 μm, 1 mm
    energies = [10.0, 50.0, 100.0]  # 10, 50, 100 GeV (skip low energy)
    start_positions = [
        15.0,
        20.0,
        25.0,
    ]  # mm before aperture (typical interaction range)

    results = []
    total = len(apertures) * len(energies) * len(start_positions)
    count = 0

    for aperture in apertures:
        for energy in energies:
            for start_z in start_positions:
                count += 1
                logger.info(
                    f"[{count}/{total}] Aperture={aperture}mm, E={energy}GeV, z={start_z}mm"
                )

                try:
                    metrics = run_single_sim(
                        aperture,
                        energy,
                        start_z,
                        base_config,
                        enable_sc=False,
                        enable_adaptive=False,
                    )
                    results.append(
                        {
                            "aperture_mm": aperture,
                            "energy_gev": energy,
                            "start_z_mm": start_z,
                            **metrics,
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    results.append(
                        {
                            "aperture_mm": aperture,
                            "energy_gev": energy,
                            "start_z_mm": start_z,
                            "error": str(e),
                        }
                    )

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_dir}/results.json")
    return results


def run_full_sweep():
    """Full sweep: 1nm-1mm apertures, 1MeV-500GeV energies."""
    logger.info("Running full sweep - this will take hours!")

    base_config = load_base_config()
    output_dir = Path("test_outputs/optimization/full_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full ranges as specified
    apertures = np.logspace(-6, 0, 15)  # 1 nm to 1 mm
    energies = np.logspace(-3, 2.7, 15)  # 1 MeV to ~500 GeV
    start_positions = np.logspace(1, 5.2, 10)  # 10mm to ~150m before aperture

    results = []
    total = len(apertures) * len(energies) * len(start_positions)
    count = 0

    for aperture in apertures:
        for energy in energies:
            for start_z in start_positions:
                count += 1
                logger.info(
                    f"[{count}/{total}] Aperture={aperture:.2e}mm, E={energy:.2e}GeV"
                )

                try:
                    metrics = run_single_sim(
                        aperture,
                        energy,
                        start_z,
                        base_config,
                        enable_sc=False,
                        enable_adaptive=False,
                    )
                    results.append(
                        {
                            "aperture_mm": aperture,
                            "energy_gev": energy,
                            "start_z_mm": start_z,
                            **metrics,
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    results.append(
                        {
                            "aperture_mm": aperture,
                            "energy_gev": energy,
                            "start_z_mm": start_z,
                            "error": str(e),
                        }
                    )

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create heatmap
    try:
        metric_array = np.zeros((len(energies), len(apertures)))
        for r in results:
            if "error" not in r:
                i = list(energies).index(r["energy_gev"])
                j = list(apertures).index(r["aperture_mm"])
                metric_array[i, j] = r["max_energy_gain_gev"]

        plot_energy_heatmap(
            aperture_sizes=apertures,
            energies=energies,
            metric_values=metric_array,
            save_path=output_dir / "heatmap.png",
        )
        logger.info(f"Heatmap saved to {output_dir}/heatmap.png")
    except Exception as e:
        logger.error(f"Failed to create heatmap: {e}")

    logger.info(f"Results saved to {output_dir}/results.json")
    return results


def run_finetune(center_aperture=0.06, center_energy=10.0):
    """Fine-tune around a specific configuration."""
    logger.info(
        f"Fine-tuning around aperture={center_aperture}mm, energy={center_energy}GeV"
    )

    base_config = load_base_config()
    output_dir = Path("test_outputs/optimization/finetune")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fine grid around center
    apertures = center_aperture * np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])
    energies = center_energy * np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    start_positions = [10.0, 12.5, 15.0, 17.5, 20.0]

    results = []
    total = len(apertures) * len(energies) * len(start_positions)
    count = 0

    for aperture in apertures:
        for energy in energies:
            for start_z in start_positions:
                count += 1
                logger.info(
                    f"[{count}/{total}] Aperture={aperture:.4f}mm, E={energy:.2f}GeV, z={start_z}mm"
                )

                try:
                    metrics = run_single_sim(
                        aperture,
                        energy,
                        start_z,
                        base_config,
                        enable_sc=True,  # Enable SC for fine-tuning
                        enable_adaptive=True,  # Enable adaptive timestep
                    )
                    results.append(
                        {
                            "aperture_mm": aperture,
                            "energy_gev": energy,
                            "start_z_mm": start_z,
                            **metrics,
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    results.append(
                        {
                            "aperture_mm": aperture,
                            "energy_gev": energy,
                            "start_z_mm": start_z,
                            "error": str(e),
                        }
                    )

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Find best
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda r: r["max_energy_gain_gev"])
        logger.info(f"\nBest configuration:")
        logger.info(f"  Aperture: {best['aperture_mm']:.4f} mm")
        logger.info(f"  Energy: {best['energy_gev']:.2f} GeV")
        logger.info(f"  Start Z: {best['start_z_mm']:.2f} mm")
        logger.info(f"  Max gain: {best['max_energy_gain_gev']:.6f} GeV")

    logger.info(f"Results saved to {output_dir}/results.json")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run LW integrator optimization")
    parser.add_argument(
        "--mode",
        choices=["quick", "sweep", "finetune"],
        default="quick",
        help="Optimization mode",
    )
    parser.add_argument(
        "--aperture", type=float, default=0.06, help="Center aperture for finetune (mm)"
    )
    parser.add_argument(
        "--energy", type=float, default=10.0, help="Center energy for finetune (GeV)"
    )

    args = parser.parse_args()

    if args.mode == "quick":
        results = run_quick_sweep()
    elif args.mode == "sweep":
        results = run_full_sweep()
    elif args.mode == "finetune":
        results = run_finetune(args.aperture, args.energy)

    logger.info("Done!")


if __name__ == "__main__":
    main()
