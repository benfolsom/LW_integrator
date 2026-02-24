#!/usr/bin/env python3
"""Plot energy gains from logcache sweep data with live updates.

This script monitors a sweep log file and automatically regenerates the plot
as new data arrives. It can also create a single static plot like the original.

Usage:
    # Live mode (auto-updates):
    ./plot_from_logcache_live.py --live [logfile]

    # Single plot (like original):
    ./plot_from_logcache_live.py [logfile]

    # Live mode with custom refresh interval:
    ./plot_from_logcache_live.py --live --interval 5 [logfile]
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Check for required dependencies
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata
except ImportError as e:
    print("=" * 80)
    print("ERROR: Missing required dependencies")
    print("=" * 80)
    print(f"\n{e}\n")
    print("This script requires matplotlib and scipy for plotting.")
    print("\nTo install the required packages, run:")
    print("\n  pip install matplotlib scipy\n")
    print("Or if using a virtual environment:")
    print("\n  source .venv/bin/activate")
    print("  pip install matplotlib scipy\n")
    print("=" * 80)
    sys.exit(1)


def parse_sweep_log(log_file, verbose=True):
    """Parse sweep log file and extract run parameters and metrics.

    Only returns data from the MOST RECENT sweep in the log file.
    If multiple sweeps are detected (e.g., user cancelled and restarted),
    all data from previous sweeps is discarded.

    Returns separate arrays for positive and negative/zero gains, plus metadata.

    Returns
    -------
    tuple
        (energies_pos, x_values_pos, percent_gains_pos,
         energies_neg, x_values_neg, percent_gains_neg,
         stats, param_metadata)
    """
    energies_pos = []
    x_values_pos = []  # Generic x-axis values (aperture or driver_distance)
    percent_gains_pos = []

    energies_neg = []
    x_values_neg = []
    percent_gains_neg = []

    total_runs = 0
    runs_with_metrics = 0
    runs_with_positive_gains = 0
    runs_with_negative_gains = 0
    last_run_num = 0
    sweep_count = 0
    current_sweep_start_line = 0

    current_run = {}

    # Metadata about the sweep parameters
    param_metadata = {
        "sweep_type": None,  # "CONDUCTING_WALL" or "BUNCH_TO_BUNCH"
        "x_param_name": None,  # e.g., "aperture" or "driver_starting_distance"
        "x_label": None,  # Human-readable label for plots
        "x_units": None,  # e.g., "mm"
        "y_param_name": "energy",  # Typically energy for both types
        "y_label": "Energy",
        "y_units": "GeV",
    }

    try:
        with open(log_file, "r") as f:
            for line_num, line in enumerate(f, start=1):
                # Detect new sweep start - RESET all accumulated data
                match = re.search(
                    r"Starting BLIND SWEEP.*?:\s*(\d+)\s+total runs", line
                )
                if match:
                    # New sweep detected - discard previous sweep data
                    if sweep_count > 0 and verbose:
                        print(
                            f"[INFO] New sweep detected at line {line_num}. Discarding previous sweep data."
                        )

                    sweep_count += 1
                    current_sweep_start_line = line_num

                    # Reset all data structures
                    energies_pos = []
                    x_values_pos = []
                    percent_gains_pos = []
                    energies_neg = []
                    x_values_neg = []
                    percent_gains_neg = []
                    runs_with_metrics = 0
                    runs_with_positive_gains = 0
                    runs_with_negative_gains = 0
                    last_run_num = 0
                    current_run = {}

                    # Update total runs for this new sweep
                    total_runs = int(match.group(1))

                # Match run start with parameters - CONDUCTING_WALL format
                match = re.search(
                    r"\[START\] Run (\d+)/\d+: a=([0-9.e+-]+)mm, E=([0-9.e+-]+)GeV",
                    line,
                )
                if match:
                    # Set metadata on first detection
                    if param_metadata["sweep_type"] is None:
                        param_metadata["sweep_type"] = "CONDUCTING_WALL"
                        param_metadata["x_param_name"] = "aperture"
                        param_metadata["x_label"] = "Aperture Radius"
                        param_metadata["x_units"] = "mm"

                    current_run = {
                        "run_num": int(match.group(1)),
                        "x_value": float(match.group(2)),
                        "energy": float(match.group(3)),
                    }
                    last_run_num = max(last_run_num, current_run["run_num"])

                # Match BUNCH_TO_BUNCH truncated format - capture parameters
                # Format: Run #   1 | initial_energy_gev=1 ... driver_starting_distance=300 | ...
                match = re.search(
                    r"Run #\s+(\d+)\s+\|.*?initial_energy_gev=([0-9.e+-]+).*?driver_starting_distance=([0-9.e+-]+)",
                    line,
                )
                if match:
                    # Set metadata on first detection
                    if param_metadata["sweep_type"] is None:
                        param_metadata["sweep_type"] = "BUNCH_TO_BUNCH"
                        param_metadata["x_param_name"] = "driver_starting_distance"
                        param_metadata["x_label"] = "Driver Starting Distance"
                        param_metadata["x_units"] = "mm"
                        param_metadata["y_label"] = "Initial Energy"

                    run_num = int(match.group(1))
                    energy = float(match.group(2))
                    driver_dist = float(match.group(3))

                    # Store as current_run for subsequent metric matching
                    current_run = {
                        "run_num": run_num,
                        "x_value": driver_dist,
                        "energy": energy,
                    }
                    last_run_num = max(last_run_num, run_num)

                # Match metrics - try both max_percent_energy_gain and percent_delta_e
                match = re.search(r"max_percent_energy_gain:\s*([0-9.e+-]+)%", line)
                if not match:
                    match = re.search(r"percent_delta_e:\s*([0-9.e+-]+)%", line)

                if match and current_run:
                    gain = float(match.group(1))
                    runs_with_metrics += 1

                    # Filter out gains with absolute value > 200% (unrealistic data)
                    if abs(gain) <= 200.0:
                        # Separate positive and negative/zero gains
                        if gain > 0:
                            energies_pos.append(current_run["energy"])
                            x_values_pos.append(current_run["x_value"])
                            percent_gains_pos.append(gain)
                            runs_with_positive_gains += 1
                        else:
                            energies_neg.append(current_run["energy"])
                            x_values_neg.append(current_run["x_value"])
                            percent_gains_neg.append(gain)
                            runs_with_negative_gains += 1

                    current_run = {}
    except FileNotFoundError:
        if verbose:
            print(f"Warning: Log file {log_file} not found yet")
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            {"total": 0, "completed": 0, "last_run": 0, "sweep_count": 0},
            param_metadata,
        )

    stats = {
        "total": total_runs,
        "completed": runs_with_metrics,
        "positive_gains": runs_with_positive_gains,
        "negative_gains": runs_with_negative_gains,
        "last_run": last_run_num,
        "sweep_count": sweep_count,
    }

    if verbose:
        if sweep_count > 1:
            print(
                f"Multiple sweeps detected: {sweep_count} total (using only the most recent)"
            )
            print(f"Current sweep started at line {current_sweep_start_line}")
        print(f"Expected total runs: {total_runs}")
        print(f"Runs with metrics found: {runs_with_metrics}")
        print(f"Runs with positive gains: {runs_with_positive_gains}")
        print(f"Runs with negative/zero gains: {runs_with_negative_gains}")
        if total_runs > 0:
            print(
                f"Progress: {runs_with_metrics}/{total_runs} ({100 * runs_with_metrics / total_runs:.1f}%)"
            )

    # Convert to numpy arrays
    energies_pos = np.array(energies_pos) if energies_pos else None
    x_values_pos = np.array(x_values_pos) if x_values_pos else None
    percent_gains_pos = np.array(percent_gains_pos) if percent_gains_pos else None

    energies_neg = np.array(energies_neg) if energies_neg else None
    x_values_neg = np.array(x_values_neg) if x_values_neg else None
    percent_gains_neg = np.array(percent_gains_neg) if percent_gains_neg else None

    return (
        energies_pos,
        x_values_pos,
        percent_gains_pos,
        energies_neg,
        x_values_neg,
        percent_gains_neg,
        stats,
        param_metadata,
    )


def create_1d_curves_plot(
    energies, apertures, percent_gains, output_file, stats=None, live_mode=False
):
    """Create 1D plot with multiple curves (one per aperture value).

    Shows gain vs energy for each unique aperture value.
    If more than 100 apertures, subsamples to show roughly 100 curves.
    Legend shows max 25 labels (always includes min and max apertures).
    """
    # Group data by aperture value
    unique_apertures = np.unique(apertures)

    if len(unique_apertures) < 2:
        # Not enough apertures for multi-curve plot
        return

    # Subsample if we have more than 100 apertures
    MAX_CURVES = 100
    if len(unique_apertures) > MAX_CURVES:
        # Sample every Nth aperture to get roughly MAX_CURVES
        step = len(unique_apertures) // MAX_CURVES
        if step < 1:
            step = 1
        aperture_indices = np.arange(0, len(unique_apertures), step)
        selected_apertures = unique_apertures[aperture_indices]
    else:
        selected_apertures = unique_apertures

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color map for curves
    import matplotlib.cm as cm

    colors = cm.viridis(np.linspace(0, 1, len(selected_apertures)))

    # Determine which curves should have labels in legend
    MAX_LEGEND_LABELS = 25
    if len(selected_apertures) <= MAX_LEGEND_LABELS:
        # Show all labels
        labeled_indices = set(range(len(selected_apertures)))
    else:
        # Subsample labels, always including first and last (min and max apertures)
        labeled_indices = {0, len(selected_apertures) - 1}  # Always include min and max

        # Add evenly spaced indices for remaining labels
        remaining_labels = MAX_LEGEND_LABELS - 2  # Already have min and max
        if remaining_labels > 0:
            step = (len(selected_apertures) - 1) / (remaining_labels + 1)
            for i in range(1, remaining_labels + 1):
                idx = int(round(i * step))
                if idx not in labeled_indices and 0 < idx < len(selected_apertures) - 1:
                    labeled_indices.add(idx)

    # Plot each aperture as a separate curve
    for i, aperture in enumerate(selected_apertures):
        # Get data points for this aperture
        mask = np.abs(apertures - aperture) < 1e-10  # Account for float precision
        energies_at_aperture = energies[mask]
        gains_at_aperture = percent_gains[mask]

        if len(energies_at_aperture) == 0:
            continue

        # Sort by energy for smooth curves
        sort_idx = np.argsort(energies_at_aperture)
        energies_sorted = energies_at_aperture[sort_idx]
        gains_sorted = gains_at_aperture[sort_idx]

        # Plot curve with label only if in labeled_indices
        label = f"a = {aperture:.4f} mm" if i in labeled_indices else None
        ax.plot(
            energies_sorted,
            gains_sorted,
            "o-",
            color=colors[i],
            linewidth=2,
            markersize=4,
            label=label,
            alpha=0.8,
        )

    ax.set_xlabel("Energy (GeV)", fontsize=12)
    ax.set_ylabel("Energy Gain (%)", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Build title
    title = "Energy Gain vs Energy (per aperture)"
    if stats:
        title += f"\n({stats['completed']}/{stats['total']} runs, {len(selected_apertures)} apertures shown"
        if len(unique_apertures) > len(selected_apertures):
            title += f" of {len(unique_apertures)} total"
        title += ")"
    if live_mode:
        title += " [LIVE]"

    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, which="both", linestyle="--", linewidth=0.5)

    # Legend - place outside if many curves
    # Only show labels for selected curves (max 25)
    n_labels = len(labeled_indices)
    if n_labels <= 5:
        ax.legend(loc="best", fontsize=10)
    else:
        if n_labels < len(selected_apertures):
            legend_title = (
                f"Apertures ({n_labels} of {len(selected_apertures)} labeled)"
            )
        else:
            legend_title = "Apertures"
        ax.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, title=legend_title
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def create_combined_gains_plot(
    energies_pos,
    x_values_pos,
    percent_gains_pos,
    energies_neg,
    x_values_neg,
    percent_gains_neg,
    output_file,
    stats=None,
    live_mode=False,
    param_metadata=None,
):
    """Create heatmap showing both positive and negative energy gains.

    Uses viridis colormap with absolute values for all gains.
    Distinction via markers: red for positive, blue for negative.
    """
    # Extract axis labels from metadata
    if param_metadata:
        x_label = f"{param_metadata['x_label']} ({param_metadata['x_units']})"
        y_label = f"{param_metadata['y_label']} ({param_metadata['y_units']})"
    else:
        # Fallback to generic labels
        x_label = "X Parameter"
        y_label = "Energy (GeV)"

    # Combine positive and negative data, using absolute values
    energies_all = []
    x_values_all = []
    gains_all = []

    if energies_pos is not None and len(energies_pos) > 0:
        energies_all.extend(energies_pos)
        x_values_all.extend(x_values_pos)
        gains_all.extend(percent_gains_pos)

    if energies_neg is not None and len(energies_neg) > 0:
        energies_all.extend(energies_neg)
        x_values_all.extend(x_values_neg)
        # Use absolute values for negative gains
        gains_all.extend(np.abs(percent_gains_neg))

    if len(energies_all) == 0:
        return  # No data to plot

    energies_all = np.array(energies_all)
    x_values_all = np.array(x_values_all)
    gains_all = np.array(gains_all)

    # Create regular grid for interpolation
    n_points = 200
    energy_grid = np.linspace(energies_all.min(), energies_all.max(), n_points)
    x_grid = np.linspace(x_values_all.min(), x_values_all.max(), n_points)
    energy_mesh, x_mesh = np.meshgrid(energy_grid, x_grid)

    # Try interpolation with fallback chain
    gain_interpolated = None
    method_used = None

    for method in ["nearest", "linear", "cubic"]:
        try:
            gain_interpolated = griddata(
                (energies_all, x_values_all),
                gains_all,
                (energy_mesh, x_mesh),
                method=method,
            )
            method_used = method
            break
        except Exception:
            if method == "cubic":
                pass
            continue

    if gain_interpolated is None:
        return  # Can't create plot

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    vmin, vmax = np.nanmin(gain_interpolated), np.nanmax(gain_interpolated)
    dynamic_range = vmax / vmin if vmin > 0 else float("inf")

    # Use log scale if dynamic range is large and values are positive
    use_log_scale = dynamic_range > 10 and vmin > 0

    if not use_log_scale or dynamic_range < 1.1:
        # Linear scale or very small range
        norm = None
        # For very small ranges, ensure contour levels are properly spaced
        if vmax - vmin < 1e-6:
            levels = np.linspace(vmin, vmax, 10)
        else:
            levels = 20
    else:
        # Log scale
        from matplotlib.colors import LogNorm

        norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
        levels = np.logspace(np.log10(max(vmin, 1e-10)), np.log10(vmax), 20)

    # Use viridis colormap (same as positive-only plot)
    contour = ax.contourf(
        energy_mesh,
        x_mesh,
        gain_interpolated,
        levels=levels,
        cmap="viridis",
        norm=norm,
    )
    cbar = plt.colorbar(contour, ax=ax, label="Energy Gain (% absolute)")

    # Set explicit colorbar ticks
    if use_log_scale:
        # For log scale, use logarithmically spaced ticks
        tick_values = np.logspace(np.log10(max(vmin, 1e-10)), np.log10(vmax), 8)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{v:.2g}" for v in tick_values])
    else:
        # For linear scale, use linearly spaced ticks
        tick_values = np.linspace(vmin, vmax, 8)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{v:.3g}" for v in tick_values])

    # Overlay positive data points in red
    if energies_pos is not None and len(energies_pos) > 0:
        ax.scatter(
            energies_pos,
            x_values_pos,
            c="red",
            s=20,
            alpha=0.4,
            edgecolors="white",
            linewidths=0.5,
            label="Positive gains",
        )

    # Overlay negative data points in blue (showing absolute values)
    if energies_neg is not None and len(energies_neg) > 0:
        ax.scatter(
            energies_neg,
            x_values_neg,
            c="blue",
            s=20,
            alpha=0.4,
            edgecolors="white",
            linewidths=0.5,
            label="Negative gains (absolute)",
        )

    if energies_pos is not None and energies_neg is not None:
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xlabel(y_label, fontsize=12)
    ax.set_ylabel(x_label, fontsize=12)

    # Build title
    n_pos = len(energies_pos) if energies_pos is not None else 0
    n_neg = len(energies_neg) if energies_neg is not None else 0
    title = f"Energy Gains (Absolute Values): {n_pos} positive, {n_neg} negative"
    if stats:
        title += f"\n({stats['completed']}/{stats['total']} total runs"
        if stats.get("sweep_count", 0) > 1:
            title += f", sweep #{stats['sweep_count']}"
        title += ")"
    if live_mode:
        title += " [LIVE]"

    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def create_contour_plot(
    energies,
    x_values,
    percent_gains,
    output_file,
    stats=None,
    live_mode=False,
    param_metadata=None,
):
    """Create a contour plot of energy gains vs energy and x parameter.

    Handles degenerate cases (quasi-1D data) gracefully by:
    1. Detecting if one dimension has very small variation
    2. Falling back to 1D line plots for degenerate dimensions
    3. Using robust interpolation (nearest-neighbor first) to avoid Qhull errors
    4. Handling small dynamic ranges with linear scaling

    Also creates a 1D multi-curve plot (if multiple apertures) showing gain vs energy
    for each aperture value.
    """
    # Extract axis labels from metadata
    if param_metadata:
        x_label = f"{param_metadata['x_label']} ({param_metadata['x_units']})"
        y_label = f"{param_metadata['y_label']} ({param_metadata['y_units']})"
    else:
        # Fallback to generic labels
        x_label = "X Parameter"
        y_label = "Energy (GeV)"

    # Detect degenerate dimensions (very small relative variation)
    RELATIVE_VARIATION_THRESHOLD = 0.01  # 1% variation threshold

    x_range = x_values.max() - x_values.min()
    x_mean = x_values.mean()
    x_rel_var = x_range / x_mean if x_mean != 0 else x_range

    energy_range = energies.max() - energies.min()
    energy_mean = energies.mean()
    energy_rel_var = energy_range / energy_mean if energy_mean != 0 else energy_range

    x_degenerate = x_rel_var < RELATIVE_VARIATION_THRESHOLD
    energy_degenerate = energy_rel_var < RELATIVE_VARIATION_THRESHOLD

    if x_degenerate or energy_degenerate:
        # Handle degenerate case with 1D plot
        # Validate array sizes match
        if len(energies) != len(x_values) or len(energies) != len(percent_gains):
            print(
                f"Warning: Array size mismatch - energies: {len(energies)}, x_values: {len(x_values)}, gains: {len(percent_gains)}"
            )
            # Skip this update if data is inconsistent
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if x_degenerate and not energy_degenerate:
            # X parameter is constant, plot gain vs energy
            # Sort by energy for cleaner line plot
            sort_idx = np.argsort(energies)
            ax.plot(
                energies[sort_idx],
                percent_gains[sort_idx],
                "o-",
                linewidth=2,
                markersize=6,
            )
            ax.set_xlabel(y_label, fontsize=12)
            ax.set_ylabel("Energy Gain (%)", fontsize=12)
            if param_metadata:
                title = f"Energy Gain vs Energy ({param_metadata['x_param_name']} ≈ {x_mean:.3f} {param_metadata['x_units']})"
            else:
                title = f"Energy Gain vs Energy (X ≈ {x_mean:.3f})"
        elif energy_degenerate and not x_degenerate:
            # Energy is constant, plot gain vs x parameter
            # Sort by x_values for cleaner line plot
            sort_idx = np.argsort(x_values)
            ax.plot(
                x_values[sort_idx],
                percent_gains[sort_idx],
                "o-",
                linewidth=2,
                markersize=6,
            )
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel("Energy Gain (%)", fontsize=12)
            if param_metadata:
                title = f"Energy Gain vs {param_metadata['x_label']} (energy ≈ {energy_mean:.3f} {param_metadata['y_units']})"
            else:
                title = f"Energy Gain vs X Parameter (energy ≈ {energy_mean:.3f})"
        else:
            # Both degenerate - just show the data points
            ax.scatter([0] * len(percent_gains), percent_gains, s=100)
            ax.set_ylabel("Energy Gain (%)", fontsize=12)
            ax.set_xticks([])
            if param_metadata:
                title = f"Energy Gain (E ≈ {energy_mean:.3f} {param_metadata['y_units']}, {param_metadata['x_param_name']} ≈ {x_mean:.3f} {param_metadata['x_units']})"
            else:
                title = f"Energy Gain (E ≈ {energy_mean:.3f}, X ≈ {x_mean:.3f})"

        if stats:
            title += f"\n({stats['completed']}/{stats['total']} runs)"
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        return

    # Non-degenerate case: create 2D contour plot
    # Create regular grid for interpolation
    n_points = 200
    energy_grid = np.linspace(energies.min(), energies.max(), n_points)
    x_grid = np.linspace(x_values.min(), x_values.max(), n_points)
    energy_mesh, x_mesh = np.meshgrid(energy_grid, x_grid)

    # Try interpolation with fallback chain: nearest -> linear -> cubic
    # Nearest is most robust (no Qhull), linear is smoother, cubic is smoothest
    gain_interpolated = None
    method_used = None

    for method in ["nearest", "linear", "cubic"]:
        try:
            gain_interpolated = griddata(
                (energies, x_values),
                percent_gains,
                (energy_mesh, x_mesh),
                method=method,
            )
            method_used = method
            break
        except Exception as e:
            if method == "cubic":
                # Last resort failed, stick with whatever we got from earlier methods
                pass
            continue

    if gain_interpolated is None:
        print(
            f"ERROR: All interpolation methods failed. Cannot create 2D plot. Falling back to scatter."
        )
        # Last resort: just scatter plot the raw data
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            energies, apertures, c=percent_gains, s=100, cmap="viridis", edgecolors="k"
        )
        ax.set_xlabel("Energy (GeV)", fontsize=12)
        ax.set_ylabel("Aperture (mm)", fontsize=12)
        title = "Energy Gain (scatter plot - interpolation failed)"
        if stats:
            title += f"\n({stats['completed']}/{stats['total']} runs)"
        ax.set_title(title, fontsize=14)
        plt.colorbar(scatter, ax=ax, label="Energy Gain (%)")
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        # Also create 1D multi-curve plot if we have multiple apertures
        if not aperture_degenerate:
            curves_output = output_file.replace(".png", "_1d_curves.png")
            try:
                create_1d_curves_plot(
                    energies, apertures, percent_gains, curves_output, stats, live_mode
                )
            except Exception as e:
                print(f"Note: Could not create 1D curves plot: {e}")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine if we should use log scale based on dynamic range
    vmin, vmax = np.nanmin(gain_interpolated), np.nanmax(gain_interpolated)
    dynamic_range = vmax / vmin if vmin > 0 else float("inf")

    # Use log scale if dynamic range is large and values are positive
    use_log_scale = dynamic_range > 10 and vmin > 0

    if not use_log_scale or dynamic_range < 1.1:
        # Linear scale or very small range
        norm = None
        # For very small ranges, ensure contour levels are properly spaced
        if vmax - vmin < 1e-6:
            levels = np.linspace(vmin, vmax, 10)
        else:
            levels = 20
    else:
        # Log scale
        from matplotlib.colors import LogNorm

        norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
        levels = np.logspace(np.log10(max(vmin, 1e-10)), np.log10(vmax), 20)

    # Create contour plot
    try:
        # Use viridis colormap (good for continuous data)
        contour = ax.contourf(
            energy_mesh,
            x_mesh,
            gain_interpolated,
            levels=levels,
            cmap="viridis",
            norm=norm,
        )
        cbar = plt.colorbar(contour, ax=ax, label="Energy Gain (%)")

        # Set explicit colorbar ticks
        if use_log_scale:
            # For log scale, use logarithmically spaced ticks
            tick_values = np.logspace(np.log10(max(vmin, 1e-10)), np.log10(vmax), 8)
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels([f"{v:.2g}" for v in tick_values])
        else:
            # For linear scale, use linearly spaced ticks
            tick_values = np.linspace(vmin, vmax, 8)
            cbar.set_ticks(tick_values)
            cbar.set_ticklabels([f"{v:.3g}" for v in tick_values])

    except ValueError as e:
        # Fallback if contouring fails (e.g., non-increasing levels)
        contour = ax.contourf(
            energy_mesh, aperture_mesh, gain_interpolated, cmap="viridis"
        )
        cbar = plt.colorbar(contour, ax=ax, label="Energy Gain (%)")

        # Set ticks for fallback case (linear scale)
        tick_values = np.linspace(vmin, vmax, 8)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{v:.3g}" for v in tick_values])

    # Overlay original data points
    # Overlay the actual data points
    ax.scatter(
        energies,
        x_values,
        c="red",
        s=20,
        alpha=0.5,
        edgecolors="white",
        linewidths=0.5,
        label="Data points",
    )
    ax.legend(loc="upper right", fontsize=9)

    ax.set_xlabel(y_label, fontsize=12)
    ax.set_ylabel(x_label, fontsize=12)

    # Build title
    if param_metadata:
        title = f"Energy Gain: {param_metadata['x_label']} vs {param_metadata['y_label']} ({method_used} interpolation)"
    else:
        title = f"Energy Gain: X vs Y ({method_used} interpolation)"
    if stats:
        title += f"\n({stats['completed']}/{stats['total']} runs"
        if stats.get("sweep_count", 0) > 1:
            title += f", sweep #{stats['sweep_count']}"
        title += ")"
    if live_mode:
        title += " [LIVE]"

    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    # Also create 1D multi-curve plot showing gain vs energy for each aperture
    curves_output = output_file.replace(".png", "_1d_curves.png")
    try:
        create_1d_curves_plot(
            energies, apertures, percent_gains, curves_output, stats, live_mode
        )
        print(f"  Also created 1D curves plot: {curves_output}")
    except Exception as e:
        # Don't fail if 1D plot creation fails
        pass


def find_latest_log(logcache_dir="logcache"):
    """Find the most recently modified sweep log file."""
    logcache_path = Path(logcache_dir)
    if not logcache_path.exists():
        return None

    sweep_logs = list(logcache_path.glob("*sweep*.log"))
    if not sweep_logs:
        return None

    return max(sweep_logs, key=lambda p: p.stat().st_mtime)


def live_monitor(log_file, output_file, interval=3):
    """Monitor log file and regenerate plot when data changes."""
    print(f"\n{'=' * 80}")
    print(f"LIVE MODE: Monitoring {log_file}")
    print(f"Plot will update every {interval} seconds when new data arrives")
    print(f"Output: {output_file}")
    print(f"Press Ctrl+C to stop")
    print(f"{'=' * 80}\n")

    last_completed = 0
    last_sweep_count = 0
    last_update_time = None
    update_count = 0

    try:
        while True:
            # Parse log file
            (
                energies_pos,
                x_values_pos,
                percent_gains_pos,
                energies_neg,
                x_values_neg,
                percent_gains_neg,
                stats,
                param_metadata,
            ) = parse_sweep_log(log_file, verbose=False)

            current_sweep = stats.get("sweep_count", 0)

            # Check if we have new data OR a new sweep has started
            if energies_pos is not None and (
                stats["completed"] > last_completed or current_sweep > last_sweep_count
            ):
                timestamp = datetime.now().strftime("%H:%M:%S")

                if current_sweep > last_sweep_count:
                    print(
                        f"\n[{timestamp}] 🔄 NEW SWEEP DETECTED (sweep #{current_sweep})!"
                    )
                    print(f"  Previous sweep data has been discarded.")
                    last_sweep_count = current_sweep
                    update_count = 0  # Reset update counter for new sweep

                print(f"\n[{timestamp}] New data detected!")
                if stats["total"] > 0:
                    print(
                        f"  Completed runs: {stats['completed']}/{stats['total']} ({stats['completed'] / stats['total'] * 100:.1f}%)"
                    )
                else:
                    print(
                        f"  Completed runs: {stats['completed']} (total not yet determined)"
                    )
                print(f"  Positive gains: {stats['positive_gains']}")
                print(f"  Generating plot...")

                # Generate updated plots (positive gains)
                create_contour_plot(
                    energies_pos,
                    x_values_pos,
                    percent_gains_pos,
                    output_file,
                    stats=stats,
                    live_mode=True,
                    param_metadata=param_metadata,
                )

                # Generate combined gains plot (positive + negative)
                if energies_neg is not None and len(energies_neg) > 0:
                    combined_output = output_file.replace(".png", "_combined.png")
                    # Generate combined plot
                    combined_output = (
                        str(Path(output_file).with_suffix("")) + "_combined.png"
                    )
                    create_combined_gains_plot(
                        energies_pos,
                        x_values_pos,
                        percent_gains_pos,
                        energies_neg,
                        x_values_neg,
                        percent_gains_neg,
                        combined_output,
                        stats=stats,
                        live_mode=True,
                        param_metadata=param_metadata,
                    )

                last_completed = stats["completed"]
                last_update_time = datetime.now()
                update_count += 1

                print(f"  ✓ Plot updated (update #{update_count})")

                # Check if sweep is complete
                if stats["total"] > 0 and stats["completed"] >= stats["total"]:
                    print(f"\n{'=' * 80}")
                    print(f"SWEEP COMPLETE!")
                    print(f"Total updates: {update_count}")
                    print(f"Final plot: {output_file}")
                    print(f"{'=' * 80}\n")
                    break
            elif energies_pos is None:
                # No data yet, wait
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Waiting for data... (checking every {interval}s)")

            # Wait before next check
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 80}")
        print(f"MONITORING STOPPED BY USER")
        print(f"Total updates: {update_count}")
        if last_update_time:
            print(f"Last update: {last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Final plot: {output_file}")
        print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot energy gains from logcache sweep data (static or live mode)"
    )
    parser.add_argument(
        "logfile",
        nargs="?",
        help="Path to sweep log file (auto-detects latest if not provided)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live monitoring mode (auto-refresh when data changes)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3,
        help="Refresh interval in seconds for live mode (default: 3)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: logcache/latest_sweep_plot.png)",
    )

    args = parser.parse_args()

    # Determine log file
    if args.logfile:
        log_file = Path(args.logfile)
    else:
        log_file = find_latest_log()
        if log_file is None:
            print("ERROR: No sweep log files found in logcache/")
            print("Please specify a log file explicitly.")
            sys.exit(1)
        print(f"Auto-detected latest sweep log: {log_file}")

    if not log_file.exists():
        print(f"ERROR: Log file not found: {log_file}")
        sys.exit(1)

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path("logcache/latest_sweep_plot.png")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Run in appropriate mode
    if args.live:
        live_monitor(str(log_file), str(output_file), interval=args.interval)
    else:
        # Single static plot
        print(f"Parsing log file: {log_file}")
        (
            energies_pos,
            x_values_pos,
            percent_gains_pos,
            energies_neg,
            x_values_neg,
            percent_gains_neg,
            stats,
            param_metadata,
        ) = parse_sweep_log(str(log_file), verbose=True)

        if energies_pos is None or len(energies_pos) == 0:
            print("ERROR: No positive gain data found in log file")
            sys.exit(1)

        print(f"\nGenerating plots...")
        create_contour_plot(
            energies_pos,
            x_values_pos,
            percent_gains_pos,
            str(output_file),
            stats=stats,
            param_metadata=param_metadata,
        )
        print(f"Positive gains plot saved to: {output_file}")

        # Generate combined gains plot if we have negative data
        if energies_neg is not None and len(energies_neg) > 0:
            combined_output = output_file.replace(".png", "_combined.png")
            create_combined_gains_plot(
                energies_pos,
                x_values_pos,
                percent_gains_pos,
                energies_neg,
                x_values_neg,
                percent_gains_neg,
                combined_output,
                stats=stats,
                param_metadata=param_metadata,
            )
            print(f"Combined gains plot saved to: {combined_output}")


if __name__ == "__main__":
    main()
