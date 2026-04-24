"""CLI helpers for plotting saved single-run trajectory files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from core.constants import ELECTRON_MASS_AMU

AMU_TO_MEV = 931.494


def _array(values: Any) -> np.ndarray:
    """Convert a saved list-like payload into a float numpy array."""
    return np.asarray(values if values is not None else [], dtype=float)


def infer_rest_energy_mev(
    particle_payload: Dict[str, Any], default_mass_amu: float = ELECTRON_MASS_AMU
) -> float:
    """Infer rest energy from saved pt/gamma history or fall back to default mass."""
    gamma = _array(particle_payload.get("gamma_hist", []))
    pt_hist = _array(particle_payload.get("pt_hist", []))
    valid = np.isfinite(gamma) & np.isfinite(pt_hist) & (gamma > 0)
    if np.any(valid):
        inferred_mass_amu = float(np.median(pt_hist[valid] / gamma[valid]))
        if inferred_mass_amu > 0 and np.isfinite(inferred_mass_amu):
            return inferred_mass_amu * AMU_TO_MEV
    return default_mass_amu * AMU_TO_MEV


def _extract_particle_series(
    particle_payload: Dict[str, Any],
    *,
    default_mass_amu: float = ELECTRON_MASS_AMU,
) -> Dict[str, np.ndarray | float]:
    """Extract standard plotting arrays from one saved particle payload."""
    positions = particle_payload.get("positions_mm", {})
    momenta = particle_payload.get("conjugate_momenta", {})

    x = _array(positions.get("x", []))
    y = _array(positions.get("y", []))
    z = _array(positions.get("z", []))
    px = _array(momenta.get("Px", []))
    py = _array(momenta.get("Py", []))
    pz = _array(momenta.get("Pz", []))
    gamma = _array(particle_payload.get("gamma_hist", []))
    time_ns = _array(particle_payload.get("time_ns", []))

    r = np.sqrt(x**2 + y**2)
    pr = np.sqrt(px**2 + py**2)
    rest_energy_mev = infer_rest_energy_mev(
        particle_payload, default_mass_amu=default_mass_amu
    )
    delta_e_mev = (
        (gamma - gamma[0]) * rest_energy_mev if gamma.size else np.asarray([], float)
    )

    return {
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "px": px,
        "py": py,
        "pz": pz,
        "pr": pr,
        "gamma": gamma,
        "time_ns": time_ns,
        "delta_e_mev": delta_e_mev,
        "rest_energy_mev": rest_energy_mev,
    }


def _plot_saved_json_trajectory(
    input_path: Path,
    output_path: Path,
    *,
    default_mass_amu: float = ELECTRON_MASS_AMU,
) -> Path:
    """Plot a saved single-run JSON trajectory file."""
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    particles = ["rider"]
    if "driver" in data.get("core", {}):
        particles.append("driver")

    fig, axes = plt.subplots(3, len(particles), figsize=(6 * len(particles), 10))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.reshape(3, len(particles))

    core_source = data.get("core", {})

    for column, particle_name in enumerate(particles):
        energy_ax = axes[0, column]
        radial_ax = axes[1, column]
        gamma_ax = axes[2, column]
        particle_payload = core_source.get(particle_name)
        if not particle_payload:
            continue

        series = _extract_particle_series(
            particle_payload, default_mass_amu=default_mass_amu
        )
        z = series["z"]
        if z.size == 0:
            continue

        energy_ax.plot(z, series["delta_e_mev"], label=particle_name.title(), linewidth=1.8)
        radial_ax.plot(z, series["r"], label=particle_name.title(), linewidth=1.8)
        gamma_ax.plot(z, series["gamma"], label=particle_name.title(), linewidth=1.8)

        title = particle_name.capitalize()
        energy_ax.set_title(f"{title} ΔE vs z", fontsize=11, fontweight="bold")
        radial_ax.set_title(f"{title} r vs z", fontsize=11, fontweight="bold")
        gamma_ax.set_title(f"{title} γ vs z", fontsize=11, fontweight="bold")

        energy_ax.set_xlabel("z position (mm)")
        energy_ax.set_ylabel("ΔE (MeV)")
        radial_ax.set_xlabel("z position (mm)")
        radial_ax.set_ylabel("r (mm)")
        gamma_ax.set_xlabel("z position (mm)")
        gamma_ax.set_ylabel("γ")

        for axis in (energy_ax, radial_ax, gamma_ax):
            axis.grid(True, alpha=0.3)
            axis.legend()

    title_parts = []
    if data.get("config_label"):
        title_parts.append(str(data["config_label"]))
    elif data.get("config_name"):
        title_parts.append(str(data["config_name"]))
    if data.get("simulation_type"):
        title_parts.append(str(data["simulation_type"]))
    fig.suptitle(
        " | ".join(title_parts) if title_parts else input_path.name,
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_saved_npz_trajectory(
    input_path: Path,
    output_path: Path,
    *,
    mass_amu: float = ELECTRON_MASS_AMU,
) -> Path:
    """Plot a saved single-run or optimization NPZ trajectory file."""
    with np.load(input_path) as data:
        z = _array(data.get("z", []))
        r = _array(data.get("r", []))
        pz = _array(data.get("pz", []))
        pr = _array(data.get("pr", []))
        gamma = _array(data.get("gamma", []))

    rest_energy_mev = mass_amu * AMU_TO_MEV
    delta_e_mev = (gamma - gamma[0]) * rest_energy_mev if gamma.size else np.asarray([])

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax_r = fig.add_subplot(gs[0, 0])
    ax_pz = fig.add_subplot(gs[0, 1])
    ax_pr = fig.add_subplot(gs[1, 0])
    ax_gamma = fig.add_subplot(gs[1, 1])
    ax_energy = fig.add_subplot(gs[2, :])

    ax_r.plot(z, r, linewidth=1.8)
    ax_r.set_title("Radial Position", fontweight="bold")
    ax_r.set_xlabel("z position (mm)")
    ax_r.set_ylabel("r (mm)")

    ax_pz.plot(z, pz, linewidth=1.8)
    ax_pz.set_title("Longitudinal Momentum", fontweight="bold")
    ax_pz.set_xlabel("z position (mm)")
    ax_pz.set_ylabel("Pz")

    ax_pr.plot(z, pr, linewidth=1.8)
    ax_pr.set_title("Transverse Momentum", fontweight="bold")
    ax_pr.set_xlabel("z position (mm)")
    ax_pr.set_ylabel("Pr")

    ax_gamma.plot(z, gamma, linewidth=1.8)
    ax_gamma.set_title("Lorentz Factor", fontweight="bold")
    ax_gamma.set_xlabel("z position (mm)")
    ax_gamma.set_ylabel("γ")

    ax_energy.plot(z, delta_e_mev, linewidth=1.8)
    ax_energy.set_title("Energy Change", fontweight="bold")
    ax_energy.set_xlabel("z position (mm)")
    ax_energy.set_ylabel("ΔE (MeV)")

    for axis in (ax_r, ax_pz, ax_pr, ax_gamma, ax_energy):
        axis.grid(True, alpha=0.3)

    fig.suptitle(input_path.name, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_saved_trajectory(
    input_path: Path,
    *,
    output_path: Optional[Path] = None,
    mass_amu: float = ELECTRON_MASS_AMU,
) -> Path:
    """Plot a saved trajectory file and return the written PNG path."""
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_plot.png")

    suffix = input_path.suffix.lower()
    if suffix == ".json":
        return _plot_saved_json_trajectory(
            input_path, output_path, default_mass_amu=mass_amu
        )
    if suffix == ".npz":
        return _plot_saved_npz_trajectory(input_path, output_path, mass_amu=mass_amu)
    raise ValueError(f"Unsupported trajectory file type: {input_path.suffix}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse trajectory plotting CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot saved single-run trajectory JSON or NPZ files"
    )
    parser.add_argument("input_path", type=Path, help="Saved trajectory JSON or NPZ file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional output PNG path",
    )
    parser.add_argument(
        "--mass-amu",
        type=float,
        default=ELECTRON_MASS_AMU,
        help="Fallback particle mass for NPZ plots or JSON files without pt/gamma inference",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the saved trajectory plotter CLI."""
    args = parse_args(argv)
    if not args.input_path.exists():
        print(f"ERROR: Trajectory file not found: {args.input_path}")
        return 1

    try:
        output_path = plot_saved_trajectory(
            args.input_path, output_path=args.output, mass_amu=args.mass_amu
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Saved trajectory plot to: {output_path}")
    return 0


__all__ = [
    "AMU_TO_MEV",
    "infer_rest_energy_mev",
    "main",
    "plot_saved_trajectory",
]


if __name__ == "__main__":
    raise SystemExit(main())
