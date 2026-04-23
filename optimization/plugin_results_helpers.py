"""Pure helpers for optimization plugin result loading and visualization."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def convert_legacy_trajectory_data(
    data: Dict[str, Any], m_particle_amu: float, amu_to_mev: float
) -> Dict[str, Any]:
    """Convert legacy trajectory JSON data into the plugin's result shape."""
    rider_data = data.get("core", {}).get("rider", {})

    positions = rider_data.get("positions_mm", {})
    x = positions.get("x", [])
    y = positions.get("y", [])
    z_pos = positions.get("z", [])
    r = [np.sqrt(xi**2 + yi**2) for xi, yi in zip(x, y)] if x and y else []

    momenta = rider_data.get("conjugate_momenta", {})
    pz = momenta.get("Pz", [])
    px = momenta.get("Px", [])
    py = momenta.get("Py", [])
    pr = [np.sqrt(pxi**2 + pyi**2) for pxi, pyi in zip(px, py)] if px and py else []

    t = rider_data.get("time_ns", [])
    gamma_hist = rider_data.get("gamma_hist", [])
    rest_energy_mev = m_particle_amu * amu_to_mev

    if gamma_hist:
        gamma_initial = gamma_hist[0]
        gamma_final = gamma_hist[-1]
        delta_e_mev = (gamma_final - gamma_initial) * rest_energy_mev
    else:
        gamma_initial = 1.0
        gamma_final = 1.0
        delta_e_mev = 0.0

    return {
        "run_number": 1,
        "parameters": {
            "aperture_radius": data.get("aperture_radius", 0),
            "particle_energy_gev": (gamma_initial - 1) * rest_energy_mev / 1000.0,
            "start_z": z_pos[0] if z_pos else 0,
            "wall_z": data.get("wall_z", 0),
            "simulation_type": data.get("simulation_type", "UNKNOWN"),
        },
        "metrics": {
            "rider_delta_e_mev": delta_e_mev,
            "rider_gamma_initial": gamma_initial,
            "rider_gamma_final": gamma_final,
        },
        "trajectory": {
            "z": z_pos,
            "r": r,
            "pz": pz,
            "pr": pr,
            "t": t,
        },
    }


def summarize_result_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a sweep/trajectory result into one metrics-summary row."""
    params = result.get("parameters", {})
    metrics = result.get("metrics", {})
    dist_info = result.get("_distance_info", {})

    z_start = dist_info.get("z_start", 0)
    z_end = dist_info.get("z_end", 0)

    return {
        "run_num": result.get("run_number", ""),
        "aperture": params.get("aperture_radius", 0),
        "energy": params.get("particle_energy_gev", 0),
        "start_z": params.get("starting_z", params.get("start_z", 0)),
        "delta_e": metrics.get("rider_delta_e_mev", 0),
        "traveled": abs(z_end - z_start),
        "gamma_initial": metrics.get("rider_gamma_initial", 0),
        "gamma_final": metrics.get("rider_gamma_final", 0),
        "emit_x": metrics.get("rider_emittance_x_mm_mrad", ""),
        "emit_y": metrics.get("rider_emittance_y_mm_mrad", ""),
        "norm_emit_x": metrics.get("rider_norm_emittance_x_mm_mrad", ""),
        "norm_emit_y": metrics.get("rider_norm_emittance_y_mm_mrad", ""),
        "beta_x": metrics.get("rider_beta_x_m", ""),
        "beta_y": metrics.get("rider_beta_y_m", ""),
    }


def collect_summary_plot_data(results: list[Dict[str, Any]]) -> Dict[str, list[float]]:
    """Collect aperture/energy/delta-E arrays from summary results."""
    apertures = []
    energies = []
    delta_es = []

    for result in results:
        row = summarize_result_row(result)
        apertures.append(row["aperture"])
        energies.append(row["energy"])
        delta_es.append(row["delta_e"])

    return {
        "apertures": apertures,
        "energies": energies,
        "delta_es": delta_es,
    }


def build_summary_heatmap_grid(
    results: list[Dict[str, Any]],
) -> tuple[list[float], list[float], np.ndarray] | None:
    """Build a summary heatmap grid when both aperture and energy were swept."""
    plot_data = collect_summary_plot_data(results)
    apertures = plot_data["apertures"]
    energies = plot_data["energies"]
    delta_es = plot_data["delta_es"]

    unique_a = sorted(set(apertures))
    unique_e = sorted(set(energies))
    if len(unique_a) <= 1 or len(unique_e) <= 1:
        return None

    grid = np.zeros((len(unique_e), len(unique_a)))
    for index, result in enumerate(results):
        row = summarize_result_row(result)
        a_idx = unique_a.index(row["aperture"])
        e_idx = unique_e.index(row["energy"])
        grid[e_idx, a_idx] = delta_es[index]

    return unique_a, unique_e, grid


def build_trajectory_plot_data(
    selected_results: list[Dict[str, Any]], m_particle_amu: float, amu_to_mev: float
) -> Dict[str, Any]:
    """Prepare trajectory and heatmap data for the viewer plots."""
    rest_mev = m_particle_amu * amu_to_mev
    series = []
    heatmap = {"apertures": [], "energies": [], "delta_es": []}

    for result in selected_results:
        traj = result.get("trajectory", {})
        row = summarize_result_row(result)
        z = np.array(traj.get("z", []))
        r = np.array(traj.get("r", []))
        if len(z) == 0:
            continue

        energy_mev_initial = (row["gamma_initial"] - 1) * rest_mev
        if len(z) > 1:
            z_range = z[-1] - z[0]
            if abs(z_range) > 1e-6:
                energy_delta = row["delta_e"] * (z - z[0]) / z_range
            else:
                energy_delta = np.zeros_like(z)
        else:
            energy_delta = np.array([0.0])

        series.append(
            {
                "run_num": row["run_num"],
                "aperture": row["aperture"],
                "energy": row["energy"],
                "delta_e": row["delta_e"],
                "z": z,
                "r": r,
                "energy_delta": energy_delta,
                "energy_mev_initial": energy_mev_initial,
            }
        )

        heatmap["apertures"].append(row["aperture"])
        heatmap["energies"].append(row["energy"])
        heatmap["delta_es"].append(row["delta_e"])

    return {"series": series, "heatmap": heatmap}


__all__ = [
    "build_summary_heatmap_grid",
    "build_trajectory_plot_data",
    "collect_summary_plot_data",
    "convert_legacy_trajectory_data",
    "summarize_result_row",
]
