"""Visualization tools for LW integrator optimization results.

This module provides functions to create heatmaps, 2D parameter plots,
and dual-curve energy plots (total ΔE vs ΔE_z).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize


def plot_energy_heatmap(
    aperture_sizes: np.ndarray,
    energies: np.ndarray,
    metric_values: np.ndarray,
    metric_name: str = "Max Energy Gain (GeV)",
    log_aperture: bool = True,
    log_energy: bool = True,
    log_metric: bool = False,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create a heatmap of energy gain vs aperture size and initial energy.

    Parameters
    ----------
    aperture_sizes : np.ndarray
        1D array of aperture sizes in mm
    energies : np.ndarray
        1D array of initial energies in GeV
    metric_values : np.ndarray
        2D array of metric values, shape (len(energies), len(aperture_sizes))
    metric_name : str, optional
        Name of metric for colorbar label
    log_aperture : bool, optional
        Use log scale for aperture axis (default: True)
    log_energy : bool, optional
        Use log scale for energy axis (default: True)
    log_metric : bool, optional
        Use log scale for metric colorbar (default: False)
    cmap : str, optional
        Matplotlib colormap name (default: 'viridis')
    figsize : Tuple[float, float], optional
        Figure size in inches
    save_path : Path, optional
        If provided, save figure to this path

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(aperture_sizes, energies)

    # Handle log scales and normalization
    norm = LogNorm() if log_metric else Normalize()

    # Plot heatmap
    im = ax.pcolormesh(X, Y, metric_values, cmap=cmap, norm=norm, shading="auto")

    # Set scales
    if log_aperture:
        ax.set_xscale("log")
    if log_energy:
        ax.set_yscale("log")

    # Labels and title
    ax.set_xlabel("Aperture Size (mm)", fontsize=12)
    ax.set_ylabel("Initial Energy (GeV)", fontsize=12)
    ax.set_title(f"{metric_name} vs Aperture Size and Energy", fontsize=14)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_name, fontsize=12)

    # Grid
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")

    return fig


def plot_dual_energy_curves(
    z_positions: np.ndarray,
    delta_e_total: np.ndarray,
    delta_e_z: np.ndarray,
    z_rel: Optional[np.ndarray] = None,
    aperture_z: Optional[float] = None,
    title: str = "Energy Change vs Position",
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot both total ΔE and ΔE_z on the same axes.

    Parameters
    ----------
    z_positions : np.ndarray
        Absolute z positions in mm
    delta_e_total : np.ndarray
        Total energy change in GeV (from Δγ)
    delta_e_z : np.ndarray
        Longitudinal energy change in GeV (from Δβ_z)
    z_rel : np.ndarray, optional
        Relative z positions (z - z_0). If None, computed from z_positions.
    aperture_z : float, optional
        Z position of aperture, will be marked with vertical line
    title : str, optional
        Plot title
    figsize : Tuple[float, float], optional
        Figure size in inches
    save_path : Path, optional
        If provided, save figure to this path

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use relative positions if provided, else compute
    if z_rel is None:
        z_rel = z_positions - z_positions[0]

    # Plot both curves
    ax.plot(
        z_rel,
        delta_e_total,
        label="Total ΔE (from Δγ)",
        color="blue",
        linewidth=2,
        alpha=0.8,
    )
    ax.plot(
        z_rel,
        delta_e_z,
        label="Longitudinal ΔE_z (from Δβ_z)",
        color="red",
        linewidth=2,
        alpha=0.8,
        linestyle="--",
    )

    # Mark aperture position if provided
    if aperture_z is not None:
        aperture_rel = aperture_z - z_positions[0]
        ax.axvline(
            aperture_rel,
            color="green",
            linestyle=":",
            linewidth=2,
            alpha=0.6,
            label=f"Aperture (z={aperture_z:.2f} mm)",
        )

    # Labels and formatting
    ax.set_xlabel("Δz (mm)", fontsize=12)
    ax.set_ylabel("ΔE (GeV)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved dual energy plot to {save_path}")

    return fig


def plot_parameter_slice(
    param_values: np.ndarray,
    metric_values: np.ndarray,
    param_name: str,
    metric_name: str,
    log_x: bool = False,
    log_y: bool = False,
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a 1D slice through parameter space.

    Parameters
    ----------
    param_values : np.ndarray
        Parameter values (x-axis)
    metric_values : np.ndarray
        Metric values (y-axis)
    param_name : str
        Name of parameter for x-axis label
    metric_name : str
        Name of metric for y-axis label
    log_x : bool, optional
        Use log scale for x-axis
    log_y : bool, optional
        Use log scale for y-axis
    figsize : Tuple[float, float], optional
        Figure size in inches
    save_path : Path, optional
        If provided, save figure to this path

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(param_values, metric_values, marker="o", linewidth=2, markersize=6)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f"{metric_name} vs {param_name}", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved parameter slice to {save_path}")

    return fig


def plot_optimization_summary(
    results: Dict[str, Any],
    primary_metric: str = "max_energy_gain_gev",
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create a comprehensive summary plot of optimization results.

    Creates a multi-panel figure showing:
    - Heatmap of primary metric vs aperture and energy
    - Histogram of primary metric values
    - Top configurations table

    Parameters
    ----------
    results : Dict[str, Any]
        Results from parameter sweep
    primary_metric : str, optional
        Name of primary metric to visualize
    figsize : Tuple[float, float], optional
        Figure size in inches
    save_path : Path, optional
        If provided, save figure to this path

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Extract data
    metrics = results["metrics"]
    params = results["parameters"]

    # Get metric values
    metric_values = [m.get(primary_metric, np.nan) for m in metrics]
    valid_mask = ~np.isnan(metric_values)
    metric_values = np.array(metric_values)

    # Panel 1: Heatmap (if 2D grid)
    if len(results["param_names"]) == 2:
        ax1 = fig.add_subplot(gs[0, :])

        param1_name = results["param_names"][0]
        param2_name = results["param_names"][1]

        # Extract unique values for each parameter
        param1_vals = sorted(list(set(p[param1_name] for p in params)))
        param2_vals = sorted(list(set(p[param2_name] for p in params)))

        # Reshape metric values
        grid_values = metric_values.reshape(results["grid_shape"])

        X, Y = np.meshgrid(param1_vals, param2_vals)
        im = ax1.pcolormesh(X, Y, grid_values, cmap="viridis", shading="auto")

        ax1.set_xlabel(param1_name, fontsize=11)
        ax1.set_ylabel(param2_name, fontsize=11)
        ax1.set_title(f"{primary_metric} Heatmap", fontsize=12)
        fig.colorbar(im, ax=ax1, label=primary_metric)
        ax1.grid(True, alpha=0.3)

    # Panel 2: Histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(metric_values[valid_mask], bins=30, edgecolor="black", alpha=0.7)
    ax2.set_xlabel(primary_metric, fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Distribution of Metric Values", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Top configurations
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    # Find top 5 configurations
    top_indices = np.argsort(metric_values[valid_mask])[-5:][::-1]
    valid_indices = np.where(valid_mask)[0]
    top_indices = valid_indices[top_indices]

    table_data = []
    for rank, idx in enumerate(top_indices, 1):
        row = [f"#{rank}"]
        for pname in results["param_names"]:
            val = params[idx][pname]
            if isinstance(val, float):
                row.append(f"{val:.3e}")
            else:
                row.append(str(val))
        row.append(f"{metric_values[idx]:.4f}")
        table_data.append(row)

    col_labels = ["Rank"] + results["param_names"] + [primary_metric]

    table = ax3.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    ax3.set_title("Top 5 Configurations", fontsize=12, pad=20)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved optimization summary to {save_path}")

    return fig


def create_interactive_plot(
    results: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> None:
    """Create an interactive HTML plot using plotly (if available).

    Parameters
    ----------
    results : Dict[str, Any]
        Results from parameter sweep
    output_path : Path, optional
        Path to save HTML file. If None, displays in browser.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed. Install with: pip install plotly")
        return

    # Extract data
    params = results["parameters"]
    metrics = results["metrics"]

    if len(results["param_names"]) == 2:
        param1_name = results["param_names"][0]
        param2_name = results["param_names"][1]

        # Get all unique metric names
        metric_names = list(metrics[0].keys()) if metrics else []

        # Create subplots for each metric
        n_metrics = len(metric_names)
        fig = make_subplots(
            rows=1,
            cols=n_metrics,
            subplot_titles=metric_names,
        )

        for col_idx, metric_name in enumerate(metric_names, 1):
            param1_vals = sorted(list(set(p[param1_name] for p in params)))
            param2_vals = sorted(list(set(p[param2_name] for p in params)))

            metric_values = [m.get(metric_name, np.nan) for m in metrics]
            grid_values = np.array(metric_values).reshape(results["grid_shape"])

            heatmap = go.Heatmap(
                x=param1_vals,
                y=param2_vals,
                z=grid_values,
                colorscale="Viridis",
                name=metric_name,
            )

            fig.add_trace(heatmap, row=1, col=col_idx)

        fig.update_layout(
            title_text="Optimization Results - Interactive Heatmaps",
            height=500,
        )

        if output_path is not None:
            fig.write_html(str(output_path))
            print(f"Saved interactive plot to {output_path}")
        else:
            fig.show()
    else:
        print("Interactive plots currently only support 2D parameter grids")
