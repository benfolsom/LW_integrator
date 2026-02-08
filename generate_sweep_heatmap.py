#!/usr/bin/env python3
"""Generate publication-quality smooth heatmap from sweep results.

This script creates ultra-smooth interpolated heatmaps with:
- 5-pass edge blur filtering
- Logarithmic color scale for gain values
- Optional logarithmic energy axis
- Adaptive contour levels with enhanced visibility
- Density-based region filtering

Usage:
    python generate_sweep_heatmap.py results/sweeps/20260206_040038_11topapertureE_sweep30
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree


def load_sweep_results(sweep_dir):
    """Load sweep results from JSON file."""
    results_file = sweep_dir / "sweep_results.json"
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        sys.exit(1)

    with open(results_file, "r") as f:
        data = json.load(f)

    return data


def extract_data(
    data,
    energy_min=None,
    energy_max=None,
    aperture_min=None,
    aperture_max=None,
    gain_filter="positive",
    gain_min=None,
    gain_max=None,
):
    """Extract energies, apertures, and gains from sweep results.

    Parameters:
    -----------
    data : dict
        Loaded sweep results JSON
    energy_min : float, optional
        Minimum energy threshold (GeV)
    energy_max : float, optional
        Maximum energy threshold (GeV)
    aperture_min : float, optional
        Minimum aperture threshold (mm)
    aperture_max : float, optional
        Maximum aperture threshold (mm)
    gain_filter : str
        'positive' (only positive gains), 'negative' (only negative), 'all' (all values)
    gain_min : float, optional
        Minimum gain threshold (%)
    gain_max : float, optional
        Maximum gain threshold (%)

    Returns:
    --------
    energies, apertures, gains : arrays
        Filtered data arrays
    """
    energies = []
    apertures = []
    gains = []

    for result in data["results"]:
        params = result.get("parameters", {})
        metrics = result.get("metrics", {})

        if not metrics:  # Skip results without metrics
            continue

        energy = params["particle_energy_gev"]
        aperture = params["aperture_radius"]
        gain = metrics.get("percent_delta_e", 0)

        # Apply filters
        if energy_min is not None and energy < energy_min:
            continue
        if energy_max is not None and energy > energy_max:
            continue
        if aperture_min is not None and aperture < aperture_min:
            continue
        if aperture_max is not None and aperture > aperture_max:
            continue

        if gain_filter == "positive" and gain <= 0:
            continue
        elif gain_filter == "negative" and gain >= 0:
            continue

        if gain_min is not None and gain < gain_min:
            continue
        if gain_max is not None and gain > gain_max:
            continue

        energies.append(energy)
        apertures.append(aperture)
        gains.append(gain)

    return np.array(energies), np.array(apertures), np.array(gains)


def create_smooth_heatmap(
    energies,
    apertures,
    gains,
    output_path="heatmap.png",
    log_energy=True,
    log_aperture=False,
    grid_resolution=800,
    smoothing_sigma=3.0,
    edge_blur_iterations=5,
    edge_blur_sigma=4.0,
    neighbor_radius=0.12,
    min_neighbors=2,
    max_distance=0.10,
    alpha_threshold=0.02,
    num_contours_low=4,
    num_contours_high=7,
    contour_threshold=1.0,
    title=None,
    show_title=True,
    figsize=(12, 8),
    dpi=300,
    show_all_gains=False,
):
    """
    Create ultra-smooth interpolated heatmap with 5-pass filtering.

    Parameters:
    -----------
    energies, apertures, gains : array-like
        Data points to plot
    output_path : str
        Output file path for PNG
    log_energy : bool
        Use logarithmic scale for energy axis
    log_aperture : bool
        Use logarithmic scale for aperture axis
    grid_resolution : int
        Grid points per dimension (higher = smoother but slower)
    smoothing_sigma : float
        Gaussian smoothing sigma for gain data
    edge_blur_iterations : int
        Number of blur passes on alpha mask (5 for publication quality)
    edge_blur_sigma : float
        Sigma for edge blur
    neighbor_radius : float
        Radius for neighbor counting (normalized units)
    min_neighbors : int
        Minimum neighbors for inclusion
    max_distance : float
        Maximum distance to nearest point (normalized units)
    alpha_threshold : float
        Minimum alpha value to display
    num_contours_low : int
        Number of contour levels below contour_threshold
    num_contours_high : int
        Number of contour levels above contour_threshold
    contour_threshold : float
        Gain value separating low/high contour density
    title : str, optional
        Plot title (auto-generated if None)
    show_title : bool
        If True, display the title; if False, hide it but preserve headspace
    figsize : tuple
        Figure size in inches
    dpi : int
        Resolution for output image
    show_all_gains : bool
        If True, show both positive and negative gains (don't mask negatives)
    """

    print(f"Creating smooth heatmap with {len(gains)} data points...")

    # Work in log space for energy if requested
    if log_energy:
        x_data = np.log10(energies)
        x_label = "Initial Energy (GeV)"
    else:
        x_data = np.array(energies)
        x_label = "Initial Energy (GeV)"

    # Work in log space for aperture if requested
    if log_aperture:
        y_data = np.log10(apertures)
        y_label = "Aperture Radius (mm)"
    else:
        y_data = np.array(apertures)
        y_label = "Aperture Radius (mm)"

    # Create grid
    x_grid_1d = np.linspace(min(x_data), max(x_data), grid_resolution)
    y_grid_1d = np.linspace(min(y_data), max(y_data), grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid_1d, y_grid_1d)

    # Interpolate gains onto grid
    points = np.array([x_data, y_data]).T
    values = np.array(gains)

    # Cubic interpolation
    gain_grid = griddata(points, values, (X_grid, Y_grid), method="cubic")

    # Fill NaN values with nearest neighbor
    nan_mask = np.isnan(gain_grid)
    if nan_mask.any():
        gain_grid_nearest = griddata(points, values, (X_grid, Y_grid), method="nearest")
        gain_grid[nan_mask] = gain_grid_nearest[nan_mask]

    # Build KDTree for density checking
    x_range = max(x_data) - min(x_data)
    y_range = max(y_data) - min(y_data)

    grid_points = np.array([X_grid.flatten(), Y_grid.flatten()]).T
    grid_points_normalized = grid_points.copy()
    grid_points_normalized[:, 0] = (grid_points[:, 0] - min(x_data)) / x_range
    grid_points_normalized[:, 1] = (grid_points[:, 1] - min(y_data)) / y_range

    points_normalized = points.copy()
    points_normalized[:, 0] = (points[:, 0] - min(x_data)) / x_range
    points_normalized[:, 1] = (points[:, 1] - min(y_data)) / y_range

    tree_normalized = KDTree(points_normalized)

    # Calculate distances and neighbor counts
    distances, indices = tree_normalized.query(grid_points_normalized)
    distances_2d = distances.reshape(X_grid.shape)

    neighbor_counts = tree_normalized.query_ball_point(
        grid_points_normalized, neighbor_radius, return_length=True
    )
    neighbor_counts_2d = neighbor_counts.reshape(X_grid.shape)

    # Create smooth alpha channel
    alpha_dist = 1.0 - (distances_2d / max_distance)
    alpha_dist = np.clip(alpha_dist, 0, 1)

    alpha_neighbors = neighbor_counts_2d / (min_neighbors * 2.0)
    alpha_neighbors = np.clip(alpha_neighbors, 0, 1)

    alpha = np.maximum(alpha_dist, alpha_neighbors)

    # Apply 5-pass blur for ultra-smooth edges
    print(f"Applying {edge_blur_iterations}-pass edge blur filtering...")
    for _ in range(edge_blur_iterations):
        alpha = gaussian_filter(alpha, sigma=edge_blur_sigma)

    # Smooth gain data
    gain_grid_smooth = gaussian_filter(gain_grid, sigma=smoothing_sigma)

    # Apply alpha mask
    gain_grid_final = np.ma.masked_where(alpha < alpha_threshold, gain_grid_smooth)

    # Handle negative/zero values for log scale
    # Only mask negatives if we're NOT showing all gains and have positive values
    if np.any(values > 0) and not show_all_gains:
        gain_grid_final = np.ma.masked_where(gain_grid_final <= 0, gain_grid_final)

    print(f"Grid points: {X_grid.size}")
    print(
        f"Valid data points: {(~gain_grid_final.mask).sum()} "
        f"({100 * (~gain_grid_final.mask).sum() / X_grid.size:.1f}%)"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine color scale
    if np.all(values > 0) and not show_all_gains:
        vmin = max(np.min(gains), 0.001)
        vmax = np.max(gains)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        color_label = "Energy Gain (%)"
    elif show_all_gains and np.any(values < 0):
        # For mixed positive/negative, use linear scale
        vmin = np.min(gains)
        vmax = np.max(gains)
        norm = None
        color_label = "Energy Gain (%)"
    else:
        vmin = np.min(gains)
        vmax = np.max(gains)
        norm = None
        color_label = "Energy Gain (%)"

    # Convert back to linear energy for plotting if needed
    if log_energy:
        X_plot = 10**X_grid
    else:
        X_plot = X_grid

    # Convert back to linear aperture for plotting if needed
    if log_aperture:
        Y_plot = 10**Y_grid
    else:
        Y_plot = Y_grid

    # Plot with pcolormesh for smooth continuous colorbar
    im = ax.pcolormesh(
        X_plot,
        Y_plot,
        gain_grid_final,
        cmap="viridis",
        norm=norm,
        shading="gouraud",
        edgecolors="none",
        linewidth=0,
    )
    cbar = plt.colorbar(im, ax=ax, label=color_label)

    # Create contour levels
    if np.all(values > 0) and not show_all_gains:
        low_levels = np.logspace(
            np.log10(vmin), np.log10(contour_threshold), num_contours_low
        )
        high_levels = np.logspace(
            np.log10(contour_threshold), np.log10(vmax), num_contours_high
        )
        high_levels = high_levels[high_levels <= vmax]
        contour_levels = np.sort(np.unique(np.concatenate([low_levels, high_levels])))
    else:
        contour_levels = np.linspace(vmin, vmax, num_contours_low + num_contours_high)

    # Draw contours
    contours = ax.contour(
        X_plot,
        Y_plot,
        gain_grid_final,
        levels=contour_levels,
        colors="white",
        alpha=0.35,
        linewidths=0.5,
    )

    # Add contour labels with subtle outline
    labels = ax.clabel(
        contours, inline=True, fontsize=8, fmt="%.2f%%", inline_spacing=10
    )

    for label in labels:
        label.set_path_effects(
            [
                PathEffects.withStroke(linewidth=1.2, foreground="black", alpha=0.3),
                PathEffects.Normal(),
            ]
        )
        label.set_color("#CCCCCC")

    # Set scales and labels
    if log_energy:
        ax.set_xscale("log")
    if log_aperture:
        ax.set_yscale("log")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    if title is None:
        if np.all(values > 0) and not show_all_gains:
            title = "Energy Gain Map: Positive Gains"
        else:
            title = "Energy Gain Map"

    if show_title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        # Add invisible title to preserve headspace
        ax.set_title(" ", fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.2, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"✓ Heatmap saved to: {output_path}")
    plt.close()


def generate_heatmap(
    sweep_dir,
    energy_min=None,
    energy_max=None,
    aperture_min=None,
    aperture_max=None,
    gain_filter="positive",
    gain_min=None,
    gain_max=None,
    output_name="sweep_heatmap_smooth.png",
    log_energy=True,
    log_aperture=False,
    show_title=True,
    resolution=800,
    dpi=300,
):
    """Generate publication-quality smooth heatmap from sweep results."""
    sweep_dir = Path(sweep_dir)

    # Load data
    data = load_sweep_results(sweep_dir)
    energies, apertures, gains = extract_data(
        data,
        energy_min=energy_min,
        energy_max=energy_max,
        aperture_min=aperture_min,
        aperture_max=aperture_max,
        gain_filter=gain_filter,
        gain_min=gain_min,
        gain_max=gain_max,
    )

    if len(gains) == 0:
        print("Error: No data points match the filter criteria!")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"Sweep: {sweep_dir.name}")
    print(f"{'=' * 70}")
    print(f"Total runs in file: {data.get('total_runs', len(data['results']))}")
    print(f"Results with metrics: {len(energies)}")
    print(f"\nParameter ranges:")
    print(f"  Energy: {energies.min():.2f} - {energies.max():.2f} GeV")
    print(f"  Aperture: {apertures.min():.4f} - {apertures.max():.4f} mm")
    print(f"\nGain statistics:")
    print(f"  Min: {gains.min():.4f}%")
    print(f"  Max: {gains.max():.4f}%")
    print(f"  Mean: {gains.mean():.4f}%")
    print(f"  Median: {np.median(gains):.4f}%")
    print()

    # Determine title
    title_parts = []
    if gain_filter == "positive":
        title_parts.append("Positive Gains")
    elif gain_filter == "negative":
        title_parts.append("Negative Gains")

    if energy_min is not None:
        title_parts.append(f"E ≥ {energy_min} GeV")
    if aperture_min is not None:
        title_parts.append(f"a ≥ {aperture_min} mm")

    if title_parts:
        title = f"Energy Gain Map: {', '.join(title_parts)}"
    else:
        title = "Energy Gain Map"

    # Create heatmap
    output_path = sweep_dir / output_name
    create_smooth_heatmap(
        energies,
        apertures,
        gains,
        output_path=output_path,
        log_energy=log_energy,
        log_aperture=log_aperture,
        grid_resolution=resolution,
        title=title,
        show_title=show_title,
        dpi=dpi,
        show_all_gains=(gain_filter == "all"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality smooth heatmap from sweep results"
    )
    parser.add_argument(
        "sweep_dir",
        help="Path to sweep directory (containing sweep_results.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="sweep_heatmap_smooth.png",
        help="Output filename (default: sweep_heatmap_smooth.png)",
    )
    parser.add_argument(
        "--energy-min",
        type=float,
        default=None,
        help="Minimum energy threshold (GeV)",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=None,
        help="Maximum energy threshold (GeV)",
    )
    parser.add_argument(
        "--aperture-min",
        type=float,
        default=None,
        help="Minimum aperture threshold (mm)",
    )
    parser.add_argument(
        "--aperture-max",
        type=float,
        default=None,
        help="Maximum aperture threshold (mm)",
    )
    parser.add_argument(
        "--gain-filter",
        choices=["positive", "negative", "all"],
        default="positive",
        help="Filter by gain sign (default: positive)",
    )
    parser.add_argument(
        "--gain-min",
        type=float,
        default=None,
        help="Minimum gain threshold (%%)",
    )
    parser.add_argument(
        "--gain-max",
        type=float,
        default=None,
        help="Maximum gain threshold (%%)",
    )
    parser.add_argument(
        "--log-energy",
        action="store_true",
        default=True,
        help="Use log scale for energy axis (default)",
    )
    parser.add_argument(
        "--linear-energy",
        action="store_false",
        dest="log_energy",
        help="Use linear scale for energy axis",
    )
    parser.add_argument(
        "--log-aperture",
        action="store_true",
        default=False,
        help="Use log scale for aperture axis",
    )
    parser.add_argument(
        "--linear-aperture",
        action="store_false",
        dest="log_aperture",
        help="Use linear scale for aperture axis (default)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=800,
        help="Grid resolution (default: 800)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300)",
    )
    parser.add_argument(
        "--no-title",
        action="store_false",
        dest="show_title",
        default=True,
        help="Hide the plot title (preserves headspace)",
    )

    args = parser.parse_args()

    generate_heatmap(
        args.sweep_dir,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        aperture_min=args.aperture_min,
        aperture_max=args.aperture_max,
        gain_filter=args.gain_filter,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        output_name=args.output,
        log_energy=args.log_energy,
        log_aperture=args.log_aperture,
        show_title=args.show_title,
        resolution=args.resolution,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
