#!/usr/bin/env python3
"""Compatibility wrapper for CSV-based sweep heatmap plotting."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from lw_integrator.sweep_heatmap import create_smooth_heatmap as create_shared_heatmap


def load_sweep_data(
    csv_path: Path,
    energy_min: float | None = None,
    energy_max: float | None = None,
    gain_filter: str = "positive",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load filtered sweep data from a CSV export."""
    energies = []
    apertures = []
    gains = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            energy = float(row["energy_GeV"])
            aperture = float(row["aperture_mm"])
            gain = float(row["percent_gain"])

            if energy_min is not None and energy < energy_min:
                continue
            if energy_max is not None and energy > energy_max:
                continue
            if gain_filter == "positive" and gain <= 0:
                continue
            if gain_filter == "negative" and gain >= 0:
                continue

            energies.append(energy)
            apertures.append(aperture)
            gains.append(gain)

    return np.asarray(energies), np.asarray(apertures), np.asarray(gains)


def build_parser() -> argparse.ArgumentParser:
    """Build the CSV heatmap CLI parser."""
    parser = argparse.ArgumentParser(
        description="Create smooth heatmaps from exported sweep CSV data"
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV file with sweep results")
    parser.add_argument(
        "-o", "--output", default="smooth_heatmap.png", help="Output PNG file path"
    )
    parser.add_argument(
        "--energy-min", type=float, default=None, help="Minimum energy (GeV)"
    )
    parser.add_argument(
        "--energy-max", type=float, default=None, help="Maximum energy (GeV)"
    )
    parser.add_argument(
        "--gain-filter",
        choices=["positive", "negative", "all"],
        default="positive",
        help="Filter by gain sign",
    )
    parser.add_argument(
        "--log-energy",
        action="store_true",
        default=True,
        help="Use log scale for energy axis",
    )
    parser.add_argument(
        "--linear-energy",
        action="store_false",
        dest="log_energy",
        help="Use linear scale for energy axis",
    )
    parser.add_argument(
        "--resolution", type=int, default=800, help="Grid resolution (default: 800)"
    )
    parser.add_argument("--title", default=None, help="Plot title")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render a CSV-backed heatmap through the shared heatmap implementation."""
    args = build_parser().parse_args(argv)

    energies, apertures, gains = load_sweep_data(
        args.input_csv,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        gain_filter=args.gain_filter,
    )

    if gains.size == 0:
        print("Error: No data points match the filter criteria!")
        return 1

    print(f"Loaded {gains.size} data points")

    create_shared_heatmap(
        energies,
        apertures,
        gains,
        output_path=args.output,
        param1_label="Initial Energy (GeV)",
        param2_label="Aperture Radius (mm)",
        log_param1=args.log_energy,
        grid_resolution=args.resolution,
        log_colorbar=bool(np.all(gains > 0)),
        show_all_gains=args.gain_filter == "all",
        title=args.title,
        dpi=args.dpi,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
