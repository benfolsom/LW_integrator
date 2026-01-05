"""Visualize optimization results from JSON files.

Usage:
    python plot_optimization_results.py test_outputs/optimization/quick_sweep/results.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def create_heatmap(results, output_path=None):
    """Create heatmap from results."""
    # Extract unique parameter values
    apertures = sorted(list(set(r["aperture_mm"] for r in results if "error" not in r)))
    energies = sorted(list(set(r["energy_gev"] for r in results if "error" not in r)))

    # Average over start_z for each aperture-energy combo
    metric_dict = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["aperture_mm"], r["energy_gev"])
        if key not in metric_dict:
            metric_dict[key] = []
        metric_dict[key].append(r["max_energy_gain_gev"])

    # Create 2D array
    metric_array = np.zeros((len(energies), len(apertures)))
    for i, energy in enumerate(energies):
        for j, aperture in enumerate(apertures):
            key = (aperture, energy)
            if key in metric_dict:
                metric_array[i, j] = np.mean(metric_dict[key])

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    X, Y = np.meshgrid(apertures, energies)
    im = ax.pcolormesh(X, Y, metric_array, cmap="viridis", shading="auto")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Aperture Size (mm)", fontsize=12)
    ax.set_ylabel("Initial Energy (GeV)", fontsize=12)
    ax.set_title("Max Energy Gain (GeV)", fontsize=14)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Energy Gain (GeV)", fontsize=12)

    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")

    return fig


def print_summary(results):
    """Print summary statistics."""
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print(f"\nResults Summary:")
    print(f"  Total runs: {len(results)}")
    print(f"  Successful: {len(valid)}")
    print(f"  Errors: {len(errors)}")

    if valid:
        gains = [r["max_energy_gain_gev"] for r in valid]
        print(f"\nEnergy Gain Statistics:")
        print(f"  Min: {min(gains):.6f} GeV")
        print(f"  Max: {max(gains):.6f} GeV")
        print(f"  Mean: {np.mean(gains):.6f} GeV")
        print(f"  Median: {np.median(gains):.6f} GeV")

        # Top 5 results
        sorted_results = sorted(
            valid, key=lambda x: x["max_energy_gain_gev"], reverse=True
        )
        print(f"\nTop 5 Configurations:")
        for i, r in enumerate(sorted_results[:5], 1):
            print(
                f"  {i}. Aperture={r['aperture_mm']:.4f}mm, Energy={r['energy_gev']:.1f}GeV, "
                f"StartZ={r['start_z_mm']:.1f}mm: ΔE={r['max_energy_gain_gev']:.6f}GeV "
                f"({r['max_relative_gain'] * 100:.3f}%)"
            )

        # Deflection summary
        deflections = [r["num_deflection_events"] for r in valid]
        print(f"\nDeflection Events:")
        print(f"  Total: {sum(deflections)}")
        print(f"  Configs with deflections: {sum(1 for d in deflections if d > 0)}")


def main():
    parser = argparse.ArgumentParser(description="Visualize optimization results")
    parser.add_argument("json_path", help="Path to results JSON file")
    parser.add_argument("--output", "-o", help="Output path for heatmap")

    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)

    # Load results
    results = load_results(json_path)

    # Print summary
    print_summary(results)

    # Create heatmap if multiple apertures and energies
    apertures = set(r.get("aperture_mm") for r in results if "error" not in r)
    energies = set(r.get("energy_gev") for r in results if "error" not in r)

    if len(apertures) > 1 and len(energies) > 1:
        output_path = args.output or json_path.parent / "heatmap.png"
        create_heatmap(results, output_path)
    else:
        print("\nNot enough parameter variation for heatmap")

        print("\nNot enough parameter variation for heatmap")

if __name__ == "__main__":
    main()
