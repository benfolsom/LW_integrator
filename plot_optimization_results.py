"""Compatibility wrapper for optimization result reporting.

This script preserves support for the older list-based ``run_optimization.py``
result format while delegating modern optimization JSON payloads to the packaged
``lw_integrator.optimization_results`` entry point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_results(json_path: Path) -> Any:
    """Load results from a JSON file."""
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_heatmap(results, output_path=None):
    """Create a heatmap for the legacy list-based result format."""
    apertures = sorted(list(set(r["aperture_mm"] for r in results if "error" not in r)))
    energies = sorted(list(set(r["energy_gev"] for r in results if "error" not in r)))

    metric_dict = {}
    for result in results:
        if "error" in result:
            continue
        key = (result["aperture_mm"], result["energy_gev"])
        metric_dict.setdefault(key, []).append(result["max_energy_gain_gev"])

    metric_array = np.zeros((len(energies), len(apertures)))
    for energy_index, energy in enumerate(energies):
        for aperture_index, aperture in enumerate(apertures):
            key = (aperture, energy)
            if key in metric_dict:
                metric_array[energy_index, aperture_index] = np.mean(metric_dict[key])

    fig, ax = plt.subplots(figsize=(10, 8))
    x_mesh, y_mesh = np.meshgrid(apertures, energies)
    image = ax.pcolormesh(x_mesh, y_mesh, metric_array, cmap="viridis", shading="auto")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Aperture Size (mm)", fontsize=12)
    ax.set_ylabel("Initial Energy (GeV)", fontsize=12)
    ax.set_title("Max Energy Gain (GeV)", fontsize=14)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Energy Gain (GeV)", fontsize=12)

    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")

    return fig


def print_summary(results):
    """Print summary statistics for the legacy list-based result format."""
    valid = [result for result in results if "error" not in result]
    errors = [result for result in results if "error" in result]

    print("\nResults Summary:")
    print(f"  Total runs: {len(results)}")
    print(f"  Successful: {len(valid)}")
    print(f"  Errors: {len(errors)}")

    if not valid:
        return

    gains = [result["max_energy_gain_gev"] for result in valid]
    print("\nEnergy Gain Statistics:")
    print(f"  Min: {min(gains):.6f} GeV")
    print(f"  Max: {max(gains):.6f} GeV")
    print(f"  Mean: {np.mean(gains):.6f} GeV")
    print(f"  Median: {np.median(gains):.6f} GeV")

    sorted_results = sorted(
        valid, key=lambda result: result["max_energy_gain_gev"], reverse=True
    )
    print("\nTop 5 Configurations:")
    for index, result in enumerate(sorted_results[:5], start=1):
        print(
            f"  {index}. Aperture={result['aperture_mm']:.4f}mm, "
            f"Energy={result['energy_gev']:.1f}GeV, "
            f"StartZ={result['start_z_mm']:.1f}mm: "
            f"ΔE={result['max_energy_gain_gev']:.6f}GeV "
            f"({result['max_relative_gain'] * 100:.3f}%)"
        )

    deflections = [result["num_deflection_events"] for result in valid]
    print("\nDeflection Events:")
    print(f"  Total: {sum(deflections)}")
    print(f"  Configs with deflections: {sum(1 for count in deflections if count > 0)}")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Visualize optimization results")
    parser.add_argument("json_path", type=Path, help="Path to results JSON file")
    parser.add_argument("--output", "-o", type=Path, help="Output path for heatmap")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point for the compatibility wrapper."""
    args = parse_args(argv)
    if not args.json_path.exists():
        print(f"Error: {args.json_path} not found")
        return 1

    results = load_results(args.json_path)

    if isinstance(results, dict):
        from lw_integrator.optimization_results import main as packaged_main

        forwarded = [str(args.json_path)]
        if args.output is not None:
            forwarded.extend(["--output", str(args.output)])
        return packaged_main(forwarded)

    if not isinstance(results, list):
        print("Error: Unsupported optimization results format")
        return 1

    print_summary(results)

    apertures = set(result.get("aperture_mm") for result in results if "error" not in result)
    energies = set(result.get("energy_gev") for result in results if "error" not in result)

    if len(apertures) > 1 and len(energies) > 1:
        output_path = args.output or args.json_path.parent / "heatmap.png"
        create_heatmap(results, output_path)
    else:
        print("\nNot enough parameter variation for heatmap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
