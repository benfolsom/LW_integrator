#!/usr/bin/env python3
"""
Smooth Heatmap Plotter for LW Integrator 2D Sweep Results

This script creates publication-quality heatmaps with:
- Ultra-smooth interpolation and edge masking
- Logarithmic color scale for gain values
- Optional logarithmic energy axis
- Adaptive contour levels with enhanced visibility
- Density-based region filtering
"""

import argparse
import csv
import os

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree


def load_sweep_data(csv_path, energy_min=None, energy_max=None, gain_filter="positive"):
    """
    Load sweep data from CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to CSV file with columns: energy_GeV, aperture_mm, percent_gain
    energy_min : float, optional
        Minimum energy threshold (GeV)
    energy_max : float, optional
        Maximum energy threshold (GeV)
    gain_filter : str
        'positive' (only positive gains), 'negative' (only negative), 'all' (all values)

    Returns:
    --------
    energies, apertures, gains : lists
        Filtered data arrays
    """
    energies = []
    apertures = []
    gains = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            energy = float(row["energy_GeV"])
            aperture = float(row["aperture_mm"])
            gain = float(row["percent_gain"])

            # Apply filters
            if energy_min is not None and energy < energy_min:
                continue
            if energy_max is not None and energy > energy_max:
                continue

            if gain_filter == "positive" and gain <= 0:
                continue
            elif gain_filter == "negative" and gain >= 0:
                continue

            energies.append(energy)
            apertures.append(aperture)
            gains.append(gain)

    return energies, apertures, gains


def create_smooth_heatmap(
    energies,
    apertures,
    gains,
    output_path="heatmap.png",
    log_energy=True,
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
    figsize=(12, 8),
    dpi=300,
):
    """
    Create ultra-smooth interpolated heatmap.

    Parameters:
    -----------
    energies, apertures, gains : array-like
        Data points to plot
    output_path : str
        Output file path for PNG
    log_energy : bool
        Use logarithmic scale for energy axis
    grid_resolution : int
        Grid points per dimension (higher = smoother but slower)
    smoothing_sigma : float
        Gaussian smoothing sigma for gain data
    edge_blur_iterations : int
        Number of blur passes on alpha mask
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
    figsize : tuple
        Figure size in inches
    dpi : int
        Resolution for output image
    """

    print(f"Creating smooth heatmap with {len(gains)} data points...")

    # Work in log space for energy if requested
    if log_energy:
        x_data = np.log10(energies)
        x_label = "Initial Energy (GeV)"
    else:
        x_data = np.array(energies)
        x_label = "Initial Energy (GeV)"

    y_data = np.array(apertures)

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

    # Apply multiple blur passes for ultra-smooth edges
    for _ in range(edge_blur_iterations):
        alpha = gaussian_filter(alpha, sigma=edge_blur_sigma)

    # Smooth gain data
    gain_grid_smooth = gaussian_filter(gain_grid, sigma=smoothing_sigma)

    # Apply alpha mask
    gain_grid_final = np.ma.masked_where(alpha < alpha_threshold, gain_grid_smooth)

    # Handle negative/zero values for log scale
    if np.any(values > 0):
        gain_grid_final = np.ma.masked_where(gain_grid_final <= 0, gain_grid_final)

    print(f"Grid points: {X_grid.size}")
    print(
        f"Valid data points: {(~gain_grid_final.mask).sum()} "
        f"({100 * (~gain_grid_final.mask).sum() / X_grid.size:.1f}%)"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine color scale
    if np.all(values > 0):
        vmin = max(np.min(gains), 0.001)
        vmax = np.max(gains)
        norm = LogNorm(vmin=vmin, vmax=vmax)
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

    # Plot with pcolormesh for smooth continuous colorbar
    im = ax.pcolormesh(
        X_plot,
        Y_grid,
        gain_grid_final,
        cmap="viridis",
        norm=norm,
        shading="gouraud",
        edgecolors="none",
        linewidth=0,
    )
    cbar = plt.colorbar(im, ax=ax, label=color_label)

    # Create contour levels
    if np.all(values > 0):
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
        Y_grid,
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

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Aperture Radius (mm)", fontsize=12)

    if title is None:
        if np.all(values > 0):
            title = "Energy Gain Map: Positive Gains"
        else:
            title = "Energy Gain Map"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Heatmap saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create smooth heatmaps from LW integrator sweep data"
    )
    parser.add_argument("input_csv", help="Input CSV file with sweep results")
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

    args = parser.parse_args()

    # Load data
    energies, apertures, gains = load_sweep_data(
        args.input_csv,
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        gain_filter=args.gain_filter,
    )

    if len(gains) == 0:
        print("Error: No data points match the filter criteria!")
        return 1

    print(f"Loaded {len(gains)} data points")

    # Create heatmap
    create_smooth_heatmap(
        energies,
        apertures,
        gains,
        output_path=args.output,
        log_energy=args.log_energy,
        grid_resolution=args.resolution,
        title=args.title,
        dpi=args.dpi,
    )

    return 0


if __name__ == "__main__":
    exit(main())
