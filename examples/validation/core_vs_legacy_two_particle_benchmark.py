#!/usr/bin/env python3
"""Benchmark the core integrator against the legacy two-particle demo."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LEGACY_ROOT = PROJECT_ROOT / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from core.trajectory_integrator import SimulationType, retarded_integrator
from legacy.bunch_inits import init_bunch  # type: ignore
from legacy.covariant_integrator_library import (  # type: ignore
    retarded_integrator as legacy_retarded_integrator,
)

FIELDS_TO_TRACK: Tuple[str, ...] = ("x", "y", "z", "Px", "Py", "Pz", "Pt", "gamma")


def _normalize_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    normalized: Dict[str, np.ndarray] = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            normalized[key] = value
        elif np.isscalar(value):
            normalized[key] = np.asarray([value], dtype=float)
        else:
            normalized[key] = np.asarray(value, dtype=float)
    return normalized


def _convert_legacy_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    converted: Dict[str, np.ndarray] = {}
    length = len(state["x"])
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            converted[key] = value
        elif key in {"q", "m", "char_time"}:
            converted[key] = np.full(length, value, dtype=float)
        else:
            converted[key] = np.asarray(value, dtype=float)
    return converted


def _extract_series(states: Iterable[Dict[str, np.ndarray]], field: str) -> np.ndarray:
    return np.asarray([state[field][0] for state in states], dtype=float)


def prepare_two_particle_demo(seed: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    np.random.seed(seed)
    rider_state, _ = init_bunch(
        starting_distance=1e-6,
        transv_mom=0.0,
        starting_Pz=1.01e6,
        stripped_ions=1.0,
        m_particle=1.007319468,
        transv_dist=2e-4,
        pcount=5,
        charge_sign=-1.0,
    )

    driver_state, _ = init_bunch(
        starting_distance=1000.0,
        transv_mom=0.0,
        starting_Pz=-1.01e6 / 207.2 * 1.007319468,
        stripped_ions=54.0,
        m_particle=207.2,
        transv_dist=2e-4 - 8e-2,
        pcount=5,
        charge_sign=1.0,
    )
    return rider_state, driver_state


def run_legacy_integrator(
    rider_state: Dict[str, np.ndarray],
    driver_state: Dict[str, np.ndarray],
    steps: int,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    legacy_traj, legacy_drv = legacy_retarded_integrator(
        steps,
        2.2e-7,
        1e5,
        1e5,
        2,
        rider_state,
        driver_state,
        1e5,
        1e5,
        0.0,
    )
    return [ _normalize_state(state) for state in legacy_traj ], [ _normalize_state(state) for state in legacy_drv ]


def run_core_integrator(
    rider_state: Dict[str, np.ndarray],
    driver_state: Dict[str, np.ndarray],
    steps: int,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    core_traj, core_drv = retarded_integrator(
        steps=steps,
        h_step=2.2e-7,
        wall_z=1e5,
        aperture_radius=1e5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_convert_legacy_state(copy.deepcopy(rider_state)),
        init_driver=_convert_legacy_state(copy.deepcopy(driver_state)),
        mean=1e5,
        cav_spacing=1e5,
        z_cutoff=0.0,
    )
    return [ _normalize_state(state) for state in core_traj ], [ _normalize_state(state) for state in core_drv ]


def compute_metrics(
    legacy: Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]],
    core: Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for label, legacy_states, core_states in (
        ("rider", legacy[0], core[0]),
        ("driver", legacy[1], core[1]),
    ):
        summary: Dict[str, float] = {}
        for field in FIELDS_TO_TRACK:
            legacy_series = _extract_series(legacy_states, field)
            core_series = _extract_series(core_states, field)
            diff = core_series - legacy_series
            max_abs = float(np.max(np.abs(diff)))
            rel = np.abs(diff) / np.maximum(np.abs(legacy_series), 1e-12)
            summary[f"{field}_max_abs"] = max_abs
            summary[f"{field}_max_rel_pct"] = float(np.max(rel) * 100.0)
            summary[f"{field}_final_abs"] = float(diff[-1])
        metrics[label] = summary
    return metrics


def summarise_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    def line(prefix: str, field: str, label: str) -> str:
        abs_key = f"{field}_max_abs"
        rel_key = f"{field}_max_rel_pct"
        return (
            f"  {prefix:<6s} {field:<3s} : max |Δ| = {metrics[label][abs_key]:.3e}, "
            f"max rel = {metrics[label][rel_key]:.3e}%"
        )

    lines = ["Benchmark summary (relative to legacy trajectories):"]
    for label in ("rider", "driver"):
        lines.append(f"- {label.capitalize()}")
        for field in ("z", "Pt", "gamma"):
            lines.append(line(label, field, label))
    return "\n".join(lines)


def export_metrics(metrics: Dict[str, Dict[str, float]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


def plot_results(
    legacy: Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]],
    core: Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]],
    *,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    steps = np.arange(len(legacy[0]))
    legacy_rider_z = _extract_series(legacy[0], "z")
    core_rider_z = _extract_series(core[0], "z")
    legacy_driver_z = _extract_series(legacy[1], "z")
    core_driver_z = _extract_series(core[1], "z")

    rider_gamma_diff = _extract_series(core[0], "gamma") - _extract_series(legacy[0], "gamma")
    driver_gamma_diff = _extract_series(core[1], "gamma") - _extract_series(legacy[1], "gamma")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(steps, legacy_rider_z, "--", label="Legacy rider")
    axes[0].plot(steps, core_rider_z, label="Core rider")
    axes[0].plot(steps, legacy_driver_z, "--", label="Legacy driver")
    axes[0].plot(steps, core_driver_z, label="Core driver")
    axes[0].set_title("Trajectory overlap (z position)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("z (mm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, rider_gamma_diff, label="Δγ rider")
    axes[1].plot(steps, driver_gamma_diff, label="Δγ driver")
    axes[1].set_title("Gamma difference (core − legacy)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Δγ")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run_benchmark(
    *,
    steps: int = 40,
    seed: int = 12345,
    save_json: Path | None = None,
    save_fig: Path | None = None,
    show: bool = False,
    plot: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run the legacy vs core comparison and return the computed metrics."""

    rider_state, driver_state = prepare_two_particle_demo(seed)
    legacy_results = run_legacy_integrator(rider_state, driver_state, steps)
    core_results = run_core_integrator(rider_state, driver_state, steps)

    metrics = compute_metrics(legacy_results, core_results)

    if save_json is not None:
        export_metrics(metrics, save_json)
        print(f"Metrics written to {save_json}")

    if plot:
        plot_results(
            legacy_results,
            core_results,
            save_path=save_fig,
            show=show,
        )
        if save_fig is not None:
            print(f"Plot written to {save_fig}")

    return metrics


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40, help="Number of integration steps to run")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for bunch initialisation")
    parser.add_argument("--save-json", type=Path, help="Write metrics to this JSON file")
    parser.add_argument("--save-fig", type=Path, help="Write comparison plot to this path")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation entirely")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Warning: ignoring unrecognised arguments: {unknown}")
    return args


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = run_benchmark(
        steps=args.steps,
        seed=args.seed,
        save_json=args.save_json,
        save_fig=args.save_fig,
        show=args.show,
        plot=not args.no_plot,
    )
    print(summarise_metrics(metrics))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
