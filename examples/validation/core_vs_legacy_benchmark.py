#!/usr/bin/env python3
"""Benchmark helpers for comparing the core and legacy two-particle integrators."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple

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

ParticleState = Dict[str, np.ndarray]
TrajectoryPair = Tuple[List[ParticleState], List[ParticleState]]

FIELDS_TO_TRACK: Tuple[str, ...] = ("x", "y", "z", "Px", "Py", "Pz", "Pt", "gamma")

CB_PALETTE = {
    "rider_primary": "#0072B2",
    "driver_primary": "#D55E00",
    "legacy_rider": "#56B4E9",
    "legacy_driver": "#E69F00",
}

DEFAULT_SAVE_DPI = 300
MAX_RECOMMENDED_DPI = 600
MIN_RECOMMENDED_DPI = 150

PARTICLE_PARAM_FIELDS: Tuple[str, ...] = (
    "starting_distance",
    "transv_mom",
    "starting_Pz",
    "stripped_ions",
    "m_particle",
    "transv_dist",
    "pcount",
    "charge_sign",
)

DEFAULT_RIDER_PARAMS: Dict[str, float | int] = {
    "starting_distance": 1.0e-6,
    "transv_mom": 0.0,
    "starting_Pz": 1.01e6,
    "stripped_ions": 1.0,
    "m_particle": 1.007319468,
    "transv_dist": 2.0e-4,
    "pcount": 5,
    "charge_sign": -1.0,
}

DEFAULT_DRIVER_PARAMS: Dict[str, float | int] = {
    "starting_distance": 1000.0,
    "transv_mom": 0.0,
    "starting_Pz": -1.01e6 / 207.2 * 1.007319468,
    "stripped_ions": 54.0,
    "m_particle": 207.2,
    "transv_dist": 2.0e-4 - 8.0e-2,
    "pcount": 5,
    "charge_sign": 1.0,
}


def _normalize_state(state: ParticleState) -> ParticleState:
    normalized: ParticleState = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            normalized[key] = value
        elif np.isscalar(value):
            normalized[key] = np.asarray([value], dtype=float)
        else:
            normalized[key] = np.asarray(value, dtype=float)
    return normalized


def _convert_legacy_state(state: ParticleState) -> ParticleState:
    converted: ParticleState = {}
    length = len(state["x"])
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            converted[key] = value
        elif key in {"q", "m", "char_time"}:
            converted[key] = np.full(length, value, dtype=float)
        else:
            converted[key] = np.asarray(value, dtype=float)
    return converted


def _extract_series(states: Iterable[ParticleState], field: str) -> np.ndarray:
    return np.asarray([state[field][0] for state in states], dtype=float)


def _merge_particle_params(
    base: Dict[str, float | int], overrides: Optional[MutableMapping[str, float | int]]
) -> Dict[str, float | int]:
    merged = dict(base)
    if overrides:
        for key, value in overrides.items():
            if key not in PARTICLE_PARAM_FIELDS:
                raise KeyError(f"Unknown particle parameter: {key}")
            merged[key] = value
    return merged


def prepare_two_particle_demo(
    seed: int,
    *,
    rider_params: Optional[MutableMapping[str, float | int]] = None,
    driver_params: Optional[MutableMapping[str, float | int]] = None,
) -> Tuple[ParticleState, ParticleState, float, float]:
    np.random.seed(seed)
    resolved_rider = _merge_particle_params(DEFAULT_RIDER_PARAMS, rider_params)
    resolved_driver = _merge_particle_params(DEFAULT_DRIVER_PARAMS, driver_params)

    rider_state, rider_rest_mev = init_bunch(**resolved_rider)
    driver_state, driver_rest_mev = init_bunch(**resolved_driver)

    return rider_state, driver_state, float(rider_rest_mev), float(driver_rest_mev)


def run_legacy_integrator(
    rider_state: ParticleState,
    driver_state: ParticleState,
    steps: int,
    *,
    time_step: float,
    wall_z: float | None,
    aperture_radius: float | None,
    simulation_type: SimulationType,
    mean: float | None = None,
    cav_spacing: float | None = None,
    z_cutoff: float | None = None,
) -> TrajectoryPair:
    if aperture_radius is None:
        raise ValueError("aperture_radius is required for the legacy integrator")

    resolved_wall_z = 0.0 if wall_z is None else wall_z
    resolved_cav_spacing = 0.0 if cav_spacing is None else cav_spacing
    resolved_z_cutoff = 0.0 if z_cutoff is None else z_cutoff

    legacy_traj, legacy_drv = legacy_retarded_integrator(
        steps,
        time_step,
        resolved_wall_z,
        aperture_radius,
        int(simulation_type),
        rider_state,
        driver_state,
        0.0 if mean is None else mean,
        resolved_cav_spacing,
        resolved_z_cutoff,
    )
    return (
        [_normalize_state(state) for state in legacy_traj],
        [_normalize_state(state) for state in legacy_drv],
    )


def run_core_integrator(
    rider_state: ParticleState,
    driver_state: ParticleState,
    steps: int,
    *,
    time_step: float,
    wall_z: float | None,
    aperture_radius: float | None,
    simulation_type: SimulationType,
    mean: float | None,
    cav_spacing: float | None,
    z_cutoff: float | None,
) -> TrajectoryPair:
    if aperture_radius is None:
        raise ValueError("aperture_radius is required for the core integrator")

    resolved_wall_z = 0.0 if wall_z is None else wall_z
    resolved_mean = 0.0 if mean is None else mean
    resolved_cav_spacing = 0.0 if cav_spacing is None else cav_spacing
    resolved_z_cutoff = 0.0 if z_cutoff is None else z_cutoff

    core_traj, core_drv = retarded_integrator(
        steps=steps,
        h_step=time_step,
        wall_z=resolved_wall_z,
        aperture_radius=aperture_radius,
        sim_type=simulation_type,
        init_rider=_convert_legacy_state(copy.deepcopy(rider_state)),
        init_driver=_convert_legacy_state(copy.deepcopy(driver_state)),
        mean=resolved_mean,
        cav_spacing=resolved_cav_spacing,
        z_cutoff=resolved_z_cutoff,
    )
    return (
        [_normalize_state(state) for state in core_traj],
        [_normalize_state(state) for state in core_drv],
    )


def compute_metrics(
    legacy: TrajectoryPair, core: TrajectoryPair
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


def summarise_metrics(metrics: Optional[Dict[str, Dict[str, float]]]) -> str:
    if not metrics:
        return "Legacy comparison skipped; no metrics computed."

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
    legacy: TrajectoryPair,
    core: TrajectoryPair,
    *,
    save_path: Path | None = None,
    show: bool = False,
    dpi: int = DEFAULT_SAVE_DPI,
) -> None:
    if dpi < MIN_RECOMMENDED_DPI or dpi > MAX_RECOMMENDED_DPI:
        raise ValueError(
            f"dpi must be between {MIN_RECOMMENDED_DPI} and {MAX_RECOMMENDED_DPI}, received {dpi}"
        )

    steps_axis = np.arange(len(legacy[0]))
    legacy_rider_z = _extract_series(legacy[0], "z")
    core_rider_z = _extract_series(core[0], "z")
    legacy_driver_z = _extract_series(legacy[1], "z")
    core_driver_z = _extract_series(core[1], "z")

    rider_gamma_diff = _extract_series(core[0], "gamma") - _extract_series(
        legacy[0], "gamma"
    )
    driver_gamma_diff = _extract_series(core[1], "gamma") - _extract_series(
        legacy[1], "gamma"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        constrained_layout=True,
        dpi=dpi,
    )

    scatter_kwargs = {"s": 90, "alpha": 0.8, "linewidth": 0}

    axes[0].scatter(
        steps_axis,
        legacy_rider_z,
        label="Legacy rider",
        color=CB_PALETTE["legacy_rider"],
        **scatter_kwargs,
    )
    axes[0].scatter(
        steps_axis,
        core_rider_z,
        label="Core rider",
        color=CB_PALETTE["rider_primary"],
        **scatter_kwargs,
    )
    axes[0].scatter(
        steps_axis,
        legacy_driver_z,
        label="Legacy driver",
        color=CB_PALETTE["legacy_driver"],
        **scatter_kwargs,
    )
    axes[0].scatter(
        steps_axis,
        core_driver_z,
        label="Core driver",
        color=CB_PALETTE["driver_primary"],
        **scatter_kwargs,
    )
    axes[0].set_title("Trajectory overlap (z position)", fontsize=18)
    axes[0].set_xlabel("Step", fontsize=16)
    axes[0].set_ylabel("z (mm)", fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=12)

    axes[1].scatter(
        steps_axis,
        rider_gamma_diff,
        label="Δγ rider",
        color=CB_PALETTE["rider_primary"],
        **scatter_kwargs,
    )
    axes[1].scatter(
        steps_axis,
        driver_gamma_diff,
        label="Δγ driver",
        color=CB_PALETTE["driver_primary"],
        **scatter_kwargs,
    )
    axes[1].set_title("Gamma difference (core − legacy)", fontsize=18)
    axes[1].set_xlabel("Step", fontsize=16)
    axes[1].set_ylabel("Δγ", fontsize=16)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=12)

    for axis in axes:
        axis.tick_params(axis="both", which="major", labelsize=13)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def compute_delta_energy_series(
    states: List[ParticleState],
    initial_state: ParticleState,
    rest_energy_mev: float,
) -> Tuple[np.ndarray, np.ndarray]:
    gamma_series = _extract_series(states, "gamma")
    initial_gamma = float(initial_state["gamma"][0])
    rest_energy_gev = rest_energy_mev * 1e-3
    delta_energy_gev = (gamma_series - initial_gamma) * rest_energy_gev
    z_series = _extract_series(states, "z")
    return delta_energy_gev, z_series


def run_benchmark(
    *,
    steps: int = 40,
    seed: int = 12345,
    rider_params: Optional[MutableMapping[str, float | int]] = None,
    driver_params: Optional[MutableMapping[str, float | int]] = None,
    legacy_enabled: bool = True,
    simulation_type: SimulationType = SimulationType.BUNCH_TO_BUNCH,
    time_step: float = 2.2e-7,
    wall_z: float | None = 1e5,
    aperture_radius: float | None = 1e5,
    mean: float | None = 1e5,
    cav_spacing: float | None = 1e5,
    z_cutoff: float | None = 0.0,
    save_json: Path | None = None,
    save_fig: Path | None = None,
    show: bool = False,
    plot: bool = True,
    return_trajectories: bool = False,
    plot_dpi: int = DEFAULT_SAVE_DPI,
    log_messages: Optional[List[str]] = None,
):
    def _log(message: str) -> None:
        if log_messages is not None:
            log_messages.append(message)
        else:
            print(message)

    rider_state, driver_state, rider_rest_mev, driver_rest_mev = (
        prepare_two_particle_demo(
            seed,
            rider_params=rider_params,
            driver_params=driver_params,
        )
    )
    rider_initial = _normalize_state(copy.deepcopy(rider_state))
    driver_initial = _normalize_state(copy.deepcopy(driver_state))

    legacy_results: Optional[TrajectoryPair] = None
    if legacy_enabled:
        legacy_results = run_legacy_integrator(
            copy.deepcopy(rider_state),
            copy.deepcopy(driver_state),
            steps,
            time_step=time_step,
            wall_z=wall_z,
            aperture_radius=aperture_radius,
            simulation_type=simulation_type,
            mean=mean,
            cav_spacing=cav_spacing,
            z_cutoff=z_cutoff,
        )

    core_results = run_core_integrator(
        copy.deepcopy(rider_state),
        copy.deepcopy(driver_state),
        steps,
        time_step=time_step,
        wall_z=wall_z,
        aperture_radius=aperture_radius,
        simulation_type=simulation_type,
        mean=mean,
        cav_spacing=cav_spacing,
        z_cutoff=z_cutoff,
    )

    metrics: Optional[Dict[str, Dict[str, float]]] = None
    if legacy_results is not None:
        metrics = compute_metrics(legacy_results, core_results)

        if save_json is not None:
            export_metrics(metrics, save_json)
            _log(f"Metrics written to {save_json}")

        if plot:
            plot_results(
                legacy_results,
                core_results,
                save_path=save_fig,
                show=show,
                dpi=plot_dpi,
            )
            if save_fig is not None:
                _log(f"Plot written to {save_fig}")
    elif plot:
        _log("Legacy comparison disabled; skipping overlay plot.")

    if return_trajectories:
        payload = {
            "core": {
                "rider": core_results[0],
                "driver": core_results[1],
            },
            "initial_states": {
                "rider": rider_initial,
                "driver": driver_initial,
            },
            "rest_energy_mev": {
                "rider": rider_rest_mev,
                "driver": driver_rest_mev,
            },
        }
        if legacy_results is not None:
            payload["legacy"] = {
                "rider": legacy_results[0],
                "driver": legacy_results[1],
            }
        return metrics, payload

    return metrics


def _add_particle_arguments(
    parser: argparse.ArgumentParser,
    prefix: str,
    defaults: Dict[str, float | int],
) -> None:
    group = parser.add_argument_group(f"{prefix.capitalize()} particle initialisation")
    for field, default in defaults.items():
        arg = f"--{prefix}-{field.replace('_', '-')}"
        arg_kwargs = {
            "default": default,
            "help": f"{field.replace('_', ' ')} (default: {default})",
        }
        if isinstance(default, int):
            group.add_argument(arg, type=int, **arg_kwargs)
        else:
            group.add_argument(arg, type=float, **arg_kwargs)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    sim_choices = {
        "conducting_wall": SimulationType.CONDUCTING_WALL,
        "switching_wall": SimulationType.SWITCHING_WALL,
        "bunch_to_bunch": SimulationType.BUNCH_TO_BUNCH,
    }

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=40, help="Number of integration steps to run"
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="Random seed for bunch initialisation"
    )
    parser.add_argument(
        "--time-step", type=float, default=2.2e-7, help="Integrator time step (ns)"
    )
    parser.add_argument("--wall-z", type=float, default=1e5, help="Wall position (mm)")
    parser.add_argument(
        "--aperture-radius", type=float, default=1e5, help="Aperture radius (mm)"
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=1e5,
        help="Mean bunch separation for core integrator",
    )
    parser.add_argument(
        "--cav-spacing",
        type=float,
        default=1e5,
        help="Cavity spacing for core integrator",
    )
    parser.add_argument(
        "--z-cutoff", type=float, default=0.0, help="z-cutoff for both integrators"
    )
    parser.add_argument(
        "--simulation-type",
        choices=sorted(sim_choices.keys()),
        default="bunch_to_bunch",
        help="Core integrator simulation type",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Skip the legacy integrator (faster but no comparison metrics)",
    )
    parser.add_argument(
        "--save-json", type=Path, help="Write metrics to this JSON file"
    )
    parser.add_argument(
        "--save-fig", type=Path, help="Write comparison plot to this path"
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plots interactively"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plot generation entirely"
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=DEFAULT_SAVE_DPI,
        help=(
            "DPI for saved plots ("
            f"{MIN_RECOMMENDED_DPI}"
            "–"
            f"{MAX_RECOMMENDED_DPI}"
            ")"
        ),
    )

    _add_particle_arguments(parser, "rider", DEFAULT_RIDER_PARAMS)
    _add_particle_arguments(parser, "driver", DEFAULT_DRIVER_PARAMS)

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Warning: ignoring unrecognised arguments: {unknown}")

    args.simulation_type = sim_choices[args.simulation_type]
    return args


def _collect_particle_args(
    args: argparse.Namespace, prefix: str
) -> Dict[str, float | int]:
    params: Dict[str, float | int] = {}
    for field in PARTICLE_PARAM_FIELDS:
        attr = f"{prefix}_{field}"
        if hasattr(args, attr):
            params[field] = getattr(args, attr)
    return params


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    rider_params = _collect_particle_args(args, "rider")
    driver_params = _collect_particle_args(args, "driver")

    metrics = run_benchmark(
        steps=args.steps,
        seed=args.seed,
        rider_params=rider_params,
        driver_params=driver_params,
        legacy_enabled=not args.core_only,
        simulation_type=args.simulation_type,
        time_step=args.time_step,
        wall_z=args.wall_z,
        aperture_radius=args.aperture_radius,
        mean=args.mean,
        cav_spacing=args.cav_spacing,
        z_cutoff=args.z_cutoff,
        save_json=args.save_json,
        save_fig=args.save_fig,
        show=args.show,
        plot=not args.no_plot,
        plot_dpi=args.plot_dpi,
    )

    print(summarise_metrics(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
