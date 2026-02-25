#!/usr/bin/env python3
"""Generate publication-quality smooth heatmap from sweep results.

This script creates ultra-smooth interpolated heatmaps with:
- 5-pass edge blur filtering
- Optional logarithmic color scale for gain values (--log-colorbar)
- Optional logarithmic energy/parameter axes (--log-param1, --log-param2)
- Adaptive contour levels with enhanced visibility
- Density-based region filtering
- Combined positive/negative gains mode (--absolute-gains)
- Colored marker overlays (red=positive, blue=negative) in absolute-gains mode
- Gridlines disabled by default (enable with --gridlines)

Usage:
    # Standard positive-gains plot
    python generate_sweep_heatmap.py results/sweeps/20260206_040038_11topapertureE_sweep30

    # Combined gains with absolute values and log colorbar
    python generate_sweep_heatmap.py results/sweeps/sweep_dir --absolute-gains --log-colorbar

    # Linear energy axis
    python generate_sweep_heatmap.py results/sweeps/sweep_dir --linear-param1
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


def detect_swept_parameters(data):
    """Detect which parameters were swept in the results.

    Returns:
    --------
    swept_params : list of str
        List of parameter names that have more than one unique value
    param_labels : dict
        Mapping of parameter names to display labels
    """
    if not data.get("results"):
        return [], {}

    # Collect all parameter values
    all_param_values = {}
    for result in data["results"]:
        params = result.get("parameters", {})
        for key, value in params.items():
            if key not in all_param_values:
                all_param_values[key] = []
            all_param_values[key].append(value)

    # Find parameters with more than one unique value
    swept_params = []
    for param_name, values in all_param_values.items():
        unique_values = set(values)
        if len(unique_values) > 1:
            swept_params.append(param_name)

    # Define display labels for common parameters
    param_labels = {
        "particle_energy_gev": "Initial Energy (GeV)",
        "initial_energy_gev": "Initial Energy (GeV)",
        "aperture_radius": "Aperture Radius (mm)",
        "driver_starting_distance": "Driver Starting Distance (mm)",
        "wall_z": "Wall Position (mm)",
        "rider_m_particle": "Rider Mass (amu)",
        "driver_m_particle": "Driver Mass (amu)",
        "rider_pcount": "Rider Particle Count",
        "driver_pcount": "Driver Particle Count",
        "rider_stripped_ions": "Rider Stripped Ions",
        "driver_stripped_ions": "Driver Stripped Ions",
        "rider_transverse_momentum": "Rider Transverse Momentum",
        "driver_transverse_momentum": "Driver Transverse Momentum",
        "rider_transv_dist": "Rider Transverse Distribution",
        "driver_transv_dist": "Driver Transverse Distribution",
    }

    return swept_params, param_labels


def extract_data(
    data,
    param1_name="particle_energy_gev",
    param2_name="aperture_radius",
    param1_min=None,
    param1_max=None,
    param2_min=None,
    param2_max=None,
    gain_filter="positive",
    gain_min=None,
    gain_max=None,
    separate_sign=False,
):
    """Extract parameter values and gains from sweep results.

    Parameters:
    -----------
    data : dict
        Loaded sweep results JSON
    param1_name : str
        Name of first parameter (typically energy)
    param2_name : str
        Name of second parameter (aperture, driver distance, etc.)
    param1_min : float, optional
        Minimum threshold for param1
    param1_max : float, optional
        Maximum threshold for param1
    param2_min : float, optional
        Minimum threshold for param2
    param2_max : float, optional
        Maximum threshold for param2
    gain_filter : str
        'positive' (only positive gains), 'negative' (only negative), 'all' (all values)
    gain_min : float, optional
        Minimum gain threshold (%)
    gain_max : float, optional
        Maximum gain threshold (%)
    separate_sign : bool
        If True, return separate arrays for positive and negative gains

    Returns:
    --------
    If separate_sign=False:
        param1_values, param2_values, gains : arrays
            Filtered data arrays
    If separate_sign=True:
        (param1_pos, param2_pos, gains_pos, param1_neg, param2_neg, gains_neg) : tuple of arrays
            Separate arrays for positive and negative gains
    """
    param1_values = []
    param2_values = []
    gains = []

    # For separate_sign mode
    param1_pos = []
    param2_pos = []
    gains_pos = []
    param1_neg = []
    param2_neg = []
    gains_neg = []

    for result in data["results"]:
        params = result.get("parameters", {})
        metrics = result.get("metrics", {})

        if not metrics:  # Skip results without metrics
            continue

        # Get parameter values - handle both old and new naming
        if param1_name in params:
            param1 = params[param1_name]
        elif param1_name == "particle_energy_gev" and "initial_energy_gev" in params:
            param1 = params["initial_energy_gev"]
        elif param1_name == "initial_energy_gev" and "particle_energy_gev" in params:
            param1 = params["particle_energy_gev"]
        else:
            continue  # Skip if parameter not found

        if param2_name in params:
            param2 = params[param2_name]
        else:
            continue  # Skip if parameter not found

        gain = metrics.get("percent_delta_e", 0)

        # Apply filters
        if param1_min is not None and param1 < param1_min:
            continue
        if param1_max is not None and param1 > param1_max:
            continue
        if param2_min is not None and param2 < param2_min:
            continue
        if param2_max is not None and param2 > param2_max:
            continue

        # Apply gain sign filter
        if gain_filter == "positive" and gain <= 0:
            continue
        elif gain_filter == "negative" and gain >= 0:
            continue

        if gain_min is not None and gain < gain_min:
            continue
        if gain_max is not None and gain > gain_max:
            continue

        if separate_sign:
            # Separate positive and negative gains
            if gain > 0:
                param1_pos.append(param1)
                param2_pos.append(param2)
                gains_pos.append(gain)
            else:
                param1_neg.append(param1)
                param2_neg.append(param2)
                gains_neg.append(gain)
        else:
            param1_values.append(param1)
            param2_values.append(param2)
            gains.append(gain)

    if separate_sign:
        return (
            np.array(param1_pos),
            np.array(param2_pos),
            np.array(gains_pos),
            np.array(param1_neg),
            np.array(param2_neg),
            np.array(gains_neg),
        )
    else:
        return np.array(param1_values), np.array(param2_values), np.array(gains)


def create_smooth_heatmap(
    param1_values,
    param2_values,
    gains,
    output_path="heatmap.png",
    param1_label="Parameter 1",
    param2_label="Parameter 2",
    log_param1=True,
    log_param2=False,
    log_colorbar=False,
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
    show_gridlines=False,
    marker_alpha=0.4,
    param1_pos=None,
    param2_pos=None,
    gains_pos=None,
    param1_neg=None,
    param2_neg=None,
    gains_neg=None,
):
    """
    Create ultra-smooth interpolated heatmap with 5-pass filtering.

    Parameters:
    -----------
    param1_values, param2_values, gains : array-like
        Data points to plot
    output_path : str
        Output file path for PNG
    param1_label : str
        Label for first parameter (x-axis)
    param2_label : str
        Label for second parameter (y-axis)
    log_param1 : bool
        Use logarithmic scale for param1 axis
    log_param2 : bool
        Use logarithmic scale for param2 axis
    log_colorbar : bool
        Use logarithmic scale for colorbar (gain values)
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
    show_gridlines : bool
        If True, show grid lines on plot
    marker_alpha : float
        Alpha transparency for marker overlays (default: 0.3)
    param1_pos, param2_pos, gains_pos : array-like, optional
        Positive gain data points for marker overlay
    param1_neg, param2_neg, gains_neg : array-like, optional
        Negative gain data points for marker overlay
    """

    print(f"Creating smooth heatmap with {len(gains)} data points...")

    # Check if param1 should be converted to MeV (if in GeV and values are small)
    convert_param1_to_mev = False
    if "gev" in param1_label.lower() and np.max(param1_values) < 1.0:
        # Convert from GeV to MeV
        convert_param1_to_mev = True
        param1_values_plot = param1_values * 1000  # GeV to MeV
        param1_label_plot = param1_label.replace("GeV", "MeV").replace("(GeV)", "(MeV)")
        print(f"Converting param1 to MeV (max value: {np.max(param1_values):.4f} GeV)")
        if param1_pos is not None and len(param1_pos) > 0:
            param1_pos = param1_pos * 1000
        if param1_neg is not None and len(param1_neg) > 0:
            param1_neg = param1_neg * 1000
    else:
        param1_values_plot = param1_values
        param1_label_plot = param1_label

    # Check if param2 should be converted to microns (if in mm and values are small)
    convert_param2_to_microns = False
    if "mm" in param2_label.lower() and np.max(param2_values) < 0.1:
        # Convert from mm to microns
        convert_param2_to_microns = True
        param2_values_plot = param2_values * 1000  # mm to microns
        param2_label_plot = param2_label.replace("mm", "μm").replace("(mm)", "(μm)")
        print(
            f"Converting param2 to microns (max value: {np.max(param2_values):.4f} mm)"
        )
        if param2_pos is not None and len(param2_pos) > 0:
            param2_pos = param2_pos * 1000
        if param2_neg is not None and len(param2_neg) > 0:
            param2_neg = param2_neg * 1000
    else:
        param2_values_plot = param2_values
        param2_label_plot = param2_label

    # Work in log space for param1 if requested
    if log_param1:
        x_data = np.log10(param1_values_plot)
    else:
        x_data = np.array(param1_values_plot)

    # Work in log space for param2 if requested
    if log_param2:
        y_data = np.log10(param2_values_plot)
    else:
        y_data = np.array(param2_values_plot)

    x_label = param1_label_plot
    y_label = param2_label_plot

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
    # Check if we have negative values that would be problematic for log scale
    has_negative = np.any(values < 0)
    has_positive = np.any(values > 0)

    # Only mask negatives if we're NOT showing all gains and have positive values
    if has_positive and not show_all_gains:
        gain_grid_final = np.ma.masked_where(gain_grid_final <= 0, gain_grid_final)

    # For absolute-gains mode with log colorbar, clamp interpolated values to positive
    # (cubic interpolation can create negative values at edges even with all-positive input)
    if show_all_gains and log_colorbar and not has_negative:
        # Input data is all positive (absolute values), but interpolation might undershoot
        # Clamp negative/zero values to small positive value to enable log scale
        if np.ma.is_masked(gain_grid_final):
            # Handle masked array
            unmasked_data = np.array(gain_grid_final.data)  # type: ignore
            if np.any(unmasked_data > 0):
                min_positive = np.min(unmasked_data[unmasked_data > 0])
                # Use max of (min_positive * 0.01) or a small epsilon to avoid zeros
                clamp_value = max(min_positive * 0.01, 1e-6)
                unmasked_data = np.maximum(unmasked_data, clamp_value)
                gain_grid_final = np.ma.masked_array(
                    unmasked_data, mask=gain_grid_final.mask
                )  # type: ignore
        else:
            # Handle regular array
            if np.any(gain_grid_final > 0):
                min_positive = np.min(gain_grid_final[gain_grid_final > 0])  # type: ignore
                # Use max of (min_positive * 0.01) or a small epsilon to avoid zeros
                clamp_value = max(min_positive * 0.01, 1e-6)
                gain_grid_final = np.maximum(gain_grid_final, clamp_value)

    # Warn if using log scale with negative values being ignored
    if log_colorbar and has_negative and not show_all_gains:
        print("WARNING: Negative gain values are being ignored for log-scale colorbar")
        print("         Use --absolute-gains mode to include negative gains")

    print(f"Grid points: {X_grid.size}")
    print(
        f"Valid data points: {(~gain_grid_final.mask).sum()} "
        f"({100 * (~gain_grid_final.mask).sum() / X_grid.size:.1f}%)"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine color scale
    vmin = np.nanmin(gain_grid_final)
    vmax = np.nanmax(gain_grid_final)

    if log_colorbar and vmin > 0:
        # User explicitly requested log scale and all values are positive
        vmin = max(vmin, 0.001)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        color_label = (
            "Energy Gain (% absolute)" if show_all_gains else "Energy Gain (%)"
        )
    elif log_colorbar and vmin <= 0:
        # User requested log scale but we have negative values - cannot use log scale
        if show_all_gains:
            print(
                "WARNING: Log colorbar requested but interpolated grid contains negative values"
            )
            print(
                "         (This can occur at grid edges even with absolute-value input data)"
            )
            print("         Using linear scale instead.")
        else:
            print("WARNING: Log colorbar requested but data contains negative values")
            print(
                "         Using linear scale instead. Use --absolute-gains for log scale with mixed signs."
            )
        norm = None
        color_label = (
            "Energy Gain (% absolute)" if show_all_gains else "Energy Gain (%)"
        )
    elif show_all_gains and has_negative:
        # Mixed positive/negative with absolute values - use linear scale
        norm = None
        color_label = "Energy Gain (% absolute)"
    else:
        # Linear scale
        norm = None
        color_label = "Energy Gain (%)"

    # Convert back to linear scale for plotting if needed
    if log_param1:
        X_plot = 10**X_grid
    else:
        X_plot = X_grid

    if log_param2:
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
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(color_label, fontsize=14)

    # Create contour levels
    if log_colorbar and vmin > 0:
        # Logarithmic contour levels
        low_levels = np.logspace(
            np.log10(vmin), np.log10(contour_threshold), num_contours_low
        )
        high_levels = np.logspace(
            np.log10(contour_threshold), np.log10(vmax), num_contours_high
        )
        high_levels = high_levels[high_levels <= vmax]
        contour_levels = np.sort(np.unique(np.concatenate([low_levels, high_levels])))
    else:
        # Linear contour levels
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
    # Use 10^x notation for log scale (matches colorbar formatting)
    if log_colorbar and vmin > 0:
        # Custom formatter for 10^x notation (without % since colorbar title has it)
        def fmt_log(x):
            if x == 0:
                return "0"
            exponent = np.log10(x)
            return f"$10^{{{exponent:.1f}}}$"

        labels = ax.clabel(
            contours, inline=True, fontsize=14, fmt=fmt_log, inline_spacing=10
        )
    elif vmax < 0.01:
        # Exponential notation for very small values
        labels = ax.clabel(
            contours, inline=True, fontsize=14, fmt="%.2e%%", inline_spacing=10
        )
    else:
        # Standard notation for linear scale with reasonable values
        labels = ax.clabel(
            contours, inline=True, fontsize=14, fmt="%.2f%%", inline_spacing=10
        )

    for label in labels:
        label.set_path_effects(
            [
                PathEffects.withStroke(linewidth=1.2, foreground="black", alpha=0.3),
                PathEffects.Normal(),
            ]
        )
        label.set_color("#CCCCCC")
        label.set_alpha(0.9)  # High opacity for text fill
        label.set_zorder(200)  # Place labels above markers

    # Set scales and labels
    if log_param1:
        ax.set_xscale("log")
    if log_param2:
        ax.set_yscale("log")

    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14, length=6, width=1)
    ax.tick_params(axis="both", which="minor", labelsize=11, length=4, width=0.8)

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

    # Add gridlines if requested (default is False)
    if show_gridlines:
        ax.grid(True, alpha=0.2, which="both")

    # Overlay markers for positive and negative gains (if provided)
    has_pos_markers = param1_pos is not None and len(param1_pos) > 0
    has_neg_markers = param1_neg is not None and len(param1_neg) > 0

    if has_pos_markers:
        # These are guaranteed to be non-None here due to has_pos_markers check
        assert param1_pos is not None
        assert param2_pos is not None

        # Markers are already in correct units (converted earlier if needed)
        ax.scatter(
            param1_pos,
            param2_pos,
            c="red",
            s=15,
            alpha=marker_alpha,
            edgecolors="white",
            linewidths=0.3,
            label="Gains",
            zorder=100,
        )

    if has_neg_markers:
        # These are guaranteed to be non-None here due to has_neg_markers check
        assert param1_neg is not None
        assert param2_neg is not None

        # Markers are already in correct units (converted earlier if needed)
        ax.scatter(
            param1_neg,
            param2_neg,
            c="blue",
            s=15,
            alpha=marker_alpha,
            edgecolors="white",
            linewidths=0.3,
            label="Losses (abs. val)",
            zorder=100,
        )

    # Add legend if we have markers
    if has_pos_markers or has_neg_markers:
        legend = ax.legend(
            loc="upper right",
            fontsize=12,
            framealpha=1.0,
            facecolor="white",
            edgecolor="gray",
            frameon=True,
            markerscale=2.0,  # Make legend markers 2x larger
        )
        legend.get_frame().set_linewidth(0.5)  # Thinner border
        legend.set_zorder(100)  # Ensure legend is on top

        # Make legend markers more opaque
        for handle in legend.legend_handles:
            if handle is not None and hasattr(handle, "set_alpha"):
                handle.set_alpha(0.95)  # High opacity for legend markers

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"✓ Heatmap saved to: {output_path}")
    plt.close()


def generate_heatmap(
    sweep_dir,
    param1_name=None,
    param2_name=None,
    param1_min=None,
    param1_max=None,
    param2_min=None,
    param2_max=None,
    gain_filter="positive",
    gain_min=None,
    gain_max=None,
    output_name="sweep_heatmap_smooth.png",
    log_param1=True,
    log_param2=False,
    log_colorbar=False,
    show_title=True,
    resolution=800,
    dpi=300,
    absolute_gains=False,
    show_gridlines=False,
    marker_alpha=0.3,
    num_contours_low=4,
    num_contours_high=7,
    contour_threshold=1.0,
):
    """Generate publication-quality smooth heatmap from sweep results.

    Auto-detects swept parameters if not specified.
    """
    sweep_dir = Path(sweep_dir)

    # Load data
    data = load_sweep_results(sweep_dir)

    # Auto-detect swept parameters if not specified
    if param1_name is None or param2_name is None:
        swept_params, param_labels = detect_swept_parameters(data)

        if len(swept_params) < 2:
            print(
                f"Error: Need exactly 2 swept parameters, found {len(swept_params)}: {swept_params}"
            )
            sys.exit(1)

        # Prioritize energy as param1
        energy_params = ["initial_energy_gev", "particle_energy_gev"]
        param1_name = None
        for ep in energy_params:
            if ep in swept_params:
                param1_name = ep
                break

        if param1_name is None:
            param1_name = swept_params[0]

        # Second parameter is the other swept param
        remaining = [p for p in swept_params if p != param1_name]
        param2_name = remaining[0] if remaining else swept_params[1]

        print(f"Auto-detected swept parameters: {param1_name}, {param2_name}")
    else:
        swept_params, param_labels = detect_swept_parameters(data)

    # Safety checks for parameter names
    if param1_name is None or param2_name is None:
        print("Error: Could not determine parameters to plot")
        sys.exit(1)

    # Get display labels (with fallback to parameter name)
    param1_label = str(param_labels.get(param1_name, param1_name))
    param2_label = str(param_labels.get(param2_name, param2_name))

    # Extract data - use separate_sign for absolute_gains mode
    if absolute_gains:
        # Extract positive and negative gains separately
        result_tuple = extract_data(
            data,
            param1_name=param1_name,
            param2_name=param2_name,
            param1_min=param1_min,
            param1_max=param1_max,
            param2_min=param2_min,
            param2_max=param2_max,
            gain_filter="all",  # Get all gains for combined plot
            gain_min=gain_min,
            gain_max=gain_max,
            separate_sign=True,
        )
        # Unpack 6-element tuple
        assert len(result_tuple) == 6
        param1_pos = result_tuple[0]
        param2_pos = result_tuple[1]
        gains_pos = result_tuple[2]
        param1_neg = result_tuple[3]
        param2_neg = result_tuple[4]
        gains_neg = result_tuple[5]

        # Combine data using absolute values for negative gains
        param1_vals = []
        param2_vals = []
        gains = []

        if len(gains_pos) > 0:
            param1_vals.extend(param1_pos)
            param2_vals.extend(param2_pos)
            gains.extend(gains_pos)

        if len(gains_neg) > 0:
            param1_vals.extend(param1_neg)
            param2_vals.extend(param2_neg)
            gains.extend(np.abs(gains_neg))  # Use absolute values

        param1_vals = np.array(param1_vals)
        param2_vals = np.array(param2_vals)
        gains = np.array(gains)

        if len(gains) == 0:
            print("Error: No data points match the filter criteria!")
            sys.exit(1)
    else:
        # Standard extraction
        result_tuple = extract_data(
            data,
            param1_name=param1_name,
            param2_name=param2_name,
            param1_min=param1_min,
            param1_max=param1_max,
            param2_min=param2_min,
            param2_max=param2_max,
            gain_filter=gain_filter,
            gain_min=gain_min,
            gain_max=gain_max,
            separate_sign=False,
        )
        # Unpack 3-element tuple
        assert len(result_tuple) == 3
        param1_vals = result_tuple[0]
        param2_vals = result_tuple[1]
        gains = result_tuple[2]

        if len(gains) == 0:
            print("Error: No data points match the filter criteria!")
            sys.exit(1)

        # Set to None for marker overlay
        param1_pos = None
        param2_pos = None
        gains_pos = None
        param1_neg = None
        param2_neg = None
        gains_neg = None

    print(f"\n{'=' * 70}")
    print(f"Sweep: {sweep_dir.name}")
    print(f"{'=' * 70}")
    print(f"Total runs in file: {data.get('total_runs', len(data['results']))}")

    if absolute_gains:
        n_pos = len(gains_pos) if gains_pos is not None and len(gains_pos) > 0 else 0
        n_neg = len(gains_neg) if gains_neg is not None and len(gains_neg) > 0 else 0
        print(f"Results with metrics: {n_pos} positive, {n_neg} negative")
    else:
        print(f"Results with metrics: {len(param1_vals)}")

    print("\nParameter ranges:")
    print(f"  {param1_label}: {param1_vals.min():.4g} - {param1_vals.max():.4g}")
    print(f"  {param2_label}: {param2_vals.min():.4g} - {param2_vals.max():.4g}")
    print("\nGain statistics (absolute values):")
    print(f"  Min: {gains.min():.4f}%")
    print(f"  Max: {gains.max():.4f}%")
    print(f"  Mean: {gains.mean():.4f}%")
    print(f"  Median: {np.median(gains):.4f}%")
    print()

    # Determine title
    title_parts = []
    if absolute_gains:
        n_pos = len(gains_pos) if gains_pos is not None and len(gains_pos) > 0 else 0
        n_neg = len(gains_neg) if gains_neg is not None and len(gains_neg) > 0 else 0
        title_parts.append(f"Absolute Values: {n_pos} positive, {n_neg} negative")
    elif gain_filter == "positive":
        title_parts.append("Positive Gains")
    elif gain_filter == "negative":
        title_parts.append("Negative Gains")

    if param1_min is not None:
        title_parts.append(f"{param1_label} ≥ {param1_min}")
    if param2_min is not None:
        title_parts.append(f"{param2_label} ≥ {param2_min}")

    if title_parts:
        title = f"Energy Gain Map: {', '.join(title_parts)}"
    else:
        title = "Energy Gain Map"

    # Create heatmap
    output_path = sweep_dir / output_name
    create_smooth_heatmap(
        param1_vals,
        param2_vals,
        gains,
        output_path=str(output_path),
        param1_label=param1_label,
        param2_label=param2_label,
        log_param1=log_param1,
        log_param2=log_param2,
        log_colorbar=log_colorbar,
        grid_resolution=resolution,
        title=title,
        show_title=show_title,
        dpi=dpi,
        show_all_gains=(gain_filter == "all" or absolute_gains),
        show_gridlines=show_gridlines,
        marker_alpha=marker_alpha,
        num_contours_low=num_contours_low,
        num_contours_high=num_contours_high,
        contour_threshold=contour_threshold,
        param1_pos=param1_pos if absolute_gains else None,
        param2_pos=param2_pos if absolute_gains else None,
        gains_pos=gains_pos if absolute_gains else None,
        param1_neg=param1_neg if absolute_gains else None,
        param2_neg=param2_neg if absolute_gains else None,
        gains_neg=gains_neg if absolute_gains else None,
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
        "--param1",
        type=str,
        default=None,
        help="First parameter name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--param2",
        type=str,
        default=None,
        help="Second parameter name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--param1-min",
        type=float,
        default=None,
        help="Minimum threshold for first parameter",
    )
    parser.add_argument(
        "--param1-max",
        type=float,
        default=None,
        help="Maximum threshold for first parameter",
    )
    parser.add_argument(
        "--param2-min",
        type=float,
        default=None,
        help="Minimum threshold for second parameter",
    )
    parser.add_argument(
        "--param2-max",
        type=float,
        default=None,
        help="Maximum threshold for second parameter",
    )
    # Keep legacy arguments for backward compatibility
    parser.add_argument(
        "--energy-min",
        type=float,
        default=None,
        help="(Legacy) Minimum energy threshold - use --param1-min instead",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=None,
        help="(Legacy) Maximum energy threshold - use --param1-max instead",
    )
    parser.add_argument(
        "--aperture-min",
        type=float,
        default=None,
        help="(Legacy) Minimum aperture threshold - use --param2-min instead",
    )
    parser.add_argument(
        "--aperture-max",
        type=float,
        default=None,
        help="(Legacy) Maximum aperture threshold - use --param2-max instead",
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
        "--log-param1",
        action="store_true",
        default=True,
        help="Use logarithmic scale for first parameter (default: True)",
    )
    parser.add_argument(
        "--linear-param1",
        action="store_false",
        dest="log_param1",
        help="Use linear scale for first parameter",
    )
    parser.add_argument(
        "--log-param2",
        action="store_true",
        default=False,
        help="Use logarithmic scale for second parameter",
    )
    # Keep legacy arguments
    parser.add_argument(
        "--log-energy",
        action="store_true",
        dest="log_param1",
        help="(Legacy) Use logarithmic energy axis - use --log-param1",
    )
    parser.add_argument(
        "--linear-energy",
        action="store_false",
        dest="log_param1",
        help="(Legacy) Use linear energy axis - use --linear-param1",
    )
    parser.add_argument(
        "--log-aperture",
        action="store_true",
        dest="log_param2",
        help="(Legacy) Use logarithmic aperture axis - use --log-param2",
    )
    parser.add_argument(
        "--linear-aperture",
        action="store_false",
        dest="log_aperture",
        help="Use linear scale for aperture axis (default)",
    )
    parser.add_argument(
        "--log-colorbar",
        action="store_true",
        default=False,
        help="Use logarithmic scale for colorbar (gain values)",
    )
    parser.add_argument(
        "--absolute-gains",
        action="store_true",
        default=False,
        help="Show both positive and negative gains using absolute values with colored markers",
    )
    parser.add_argument(
        "--gridlines",
        action="store_true",
        default=False,
        help="Show gridlines on plot (default: False)",
    )
    parser.add_argument(
        "--num-contours",
        type=int,
        default=None,
        help="Total number of contour levels (default: 11 total, split between low/high)",
    )
    parser.add_argument(
        "--num-contours-low",
        type=int,
        default=4,
        help="Number of contour levels below threshold (default: 4)",
    )
    parser.add_argument(
        "--num-contours-high",
        type=int,
        default=7,
        help="Number of contour levels above threshold (default: 7)",
    )
    parser.add_argument(
        "--contour-threshold",
        type=float,
        default=1.0,
        help="Gain threshold separating low/high contour density (default: 1.0)",
    )
    parser.add_argument(
        "--marker-alpha",
        type=float,
        default=0.4,
        help="Alpha transparency for marker overlays in absolute-gains mode (default: 0.4)",
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

    # Handle legacy arguments
    param1_min = args.param1_min if args.param1_min is not None else args.energy_min
    param1_max = args.param1_max if args.param1_max is not None else args.energy_max
    param2_min = args.param2_min if args.param2_min is not None else args.aperture_min
    param2_max = args.param2_max if args.param2_max is not None else args.aperture_max

    # Handle contour arguments
    if args.num_contours is not None:
        # If --num-contours is specified, split it roughly 40/60 between low/high
        num_contours_low = max(1, int(args.num_contours * 0.4))
        num_contours_high = max(1, args.num_contours - num_contours_low)
    else:
        num_contours_low = args.num_contours_low
        num_contours_high = args.num_contours_high

    generate_heatmap(
        sweep_dir=args.sweep_dir,
        param1_name=args.param1,
        param2_name=args.param2,
        param1_min=param1_min,
        param1_max=param1_max,
        param2_min=param2_min,
        param2_max=param2_max,
        gain_filter=args.gain_filter,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        output_name=args.output,
        log_param1=args.log_param1,
        log_param2=args.log_param2,
        log_colorbar=args.log_colorbar,
        show_title=args.show_title,
        resolution=args.resolution,
        dpi=args.dpi,
        absolute_gains=args.absolute_gains,
        show_gridlines=args.gridlines,
        marker_alpha=args.marker_alpha,
        num_contours_low=num_contours_low,
        num_contours_high=num_contours_high,
        contour_threshold=args.contour_threshold,
    )


if __name__ == "__main__":
    main()
