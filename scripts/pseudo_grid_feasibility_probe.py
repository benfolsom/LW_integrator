"""Run lightweight pseudo-grid sanity and scale probes.

This script is intentionally deterministic and small enough for local iteration.
It is not a substitute for full physics validation sweeps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import C_MMNS
from core.integration_runner import retarded_integrator
from core.types import PseudoGridConfig, SimulationType


@dataclass(frozen=True)
class ProbeResult:
    label: str
    particle_count: int
    active_count: int | None
    steps: int
    elapsed_s: float
    finite: bool
    max_abs_x_mm: float
    max_abs_z_mm: float
    max_gamma: float
    min_gamma: float
    final_mean_z_mm: float
    retained_rider_start_index: int | None
    retained_driver_start_index: int | None
    dropped_rider_samples: int
    dropped_driver_samples: int


@dataclass(frozen=True)
class ComparisonResult:
    label: str
    reference_label: str
    candidate_label: str
    max_abs_x_delta_mm: float
    max_abs_z_delta_mm: float
    max_abs_gamma_delta: float
    candidate_elapsed_s: float
    reference_elapsed_s: float
    speed_ratio_reference_over_candidate: float


def _make_bunch(
    *,
    n_particles: int,
    z_mm: float,
    x_offset_mm: float,
    charge_scale: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-0.75, 0.75, n_particles) + x_offset_mm
    if n_particles > 1:
        x += rng.normal(0.0, 0.02, n_particles)
    y = rng.normal(0.0, 0.01, n_particles)
    z = np.full(n_particles, z_mm, dtype=float)
    gamma = np.ones(n_particles, dtype=float)
    mass = np.ones(n_particles, dtype=float)
    zeros = np.zeros(n_particles, dtype=float)
    q_pattern = rng.normal(0.0, 1.0, n_particles)
    q_pattern -= float(np.mean(q_pattern))
    if np.max(np.abs(q_pattern)) > 0.0:
        q_pattern /= float(np.max(np.abs(q_pattern)))
    return {
        "x": x.astype(float),
        "y": y.astype(float),
        "z": z,
        "t": zeros.copy(),
        "Px": zeros.copy(),
        "Py": zeros.copy(),
        "Pz": zeros.copy(),
        "Pt": gamma * mass * C_MMNS,
        "gamma": gamma,
        "bx": zeros.copy(),
        "by": zeros.copy(),
        "bz": zeros.copy(),
        "bdotx": zeros.copy(),
        "bdoty": zeros.copy(),
        "bdotz": zeros.copy(),
        "q": charge_scale * q_pattern,
        "m": mass,
        "char_time": np.full(n_particles, 1.0e-3, dtype=float),
    }


def _clone_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in state.items()}


def _run_probe_case(
    *,
    label: str,
    n_particles: int,
    steps: int,
    h_step: float,
    charge_scale: float,
    active_count: int | None,
    causal_history_pruning: bool,
) -> tuple[ProbeResult, Any, Any]:
    rider = _make_bunch(
        n_particles=n_particles,
        z_mm=-0.5,
        x_offset_mm=0.0,
        charge_scale=charge_scale,
        seed=100 + n_particles,
    )
    driver = _make_bunch(
        n_particles=n_particles,
        z_mm=0.5,
        x_offset_mm=0.08,
        charge_scale=-charge_scale,
        seed=200 + n_particles,
    )
    pseudo_grid = None
    if active_count is not None:
        pseudo_grid = PseudoGridConfig(
            enabled=True,
            active_rider_count=active_count,
            active_driver_count=active_count,
            passive_neighbor_count=min(4, active_count),
            causal_history_pruning_enabled=causal_history_pruning,
            causal_history_safety_margin_steps=0,
        )

    start = time.perf_counter()
    rider_trajectory, driver_trajectory, rider_soa, driver_soa = retarded_integrator(
        steps=steps,
        h_step=h_step,
        wall_z=0.0,
        aperture_radius=10.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_clone_state(rider),
        init_driver=_clone_state(driver),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        pseudo_grid=pseudo_grid,
        use_numba=False,
        radiation_reaction_mode="power_matched_damping",
    )
    elapsed_s = time.perf_counter() - start

    if rider_soa is None or driver_soa is None:
        raise RuntimeError("retarded_integrator did not return trajectory arrays")

    schedule = rider_trajectory[-1].get("_pseudo_grid_schedule")
    result = ProbeResult(
        label=label,
        particle_count=n_particles,
        active_count=active_count,
        steps=steps,
        elapsed_s=elapsed_s,
        finite=bool(
            np.all(np.isfinite(rider_soa.x))
            and np.all(np.isfinite(rider_soa.z))
            and np.all(np.isfinite(rider_soa.gamma))
            and np.all(np.isfinite(driver_soa.x))
            and np.all(np.isfinite(driver_soa.z))
            and np.all(np.isfinite(driver_soa.gamma))
        ),
        max_abs_x_mm=float(
            max(np.max(np.abs(rider_soa.x)), np.max(np.abs(driver_soa.x)))
        ),
        max_abs_z_mm=float(
            max(np.max(np.abs(rider_soa.z)), np.max(np.abs(driver_soa.z)))
        ),
        max_gamma=float(max(np.max(rider_soa.gamma), np.max(driver_soa.gamma))),
        min_gamma=float(min(np.min(rider_soa.gamma), np.min(driver_soa.gamma))),
        final_mean_z_mm=float(np.mean(rider_soa.z[-1])),
        retained_rider_start_index=(
            None if schedule is None else schedule.rider_retained_history_start_index
        ),
        retained_driver_start_index=(
            None if schedule is None else schedule.driver_retained_history_start_index
        ),
        dropped_rider_samples=(
            0 if schedule is None else schedule.rider_dropped_history_samples
        ),
        dropped_driver_samples=(
            0 if schedule is None else schedule.driver_dropped_history_samples
        ),
    )
    return result, rider_soa, driver_soa


def _compare(
    *,
    label: str,
    reference: ProbeResult,
    candidate: ProbeResult,
    reference_rider_soa: Any,
    candidate_rider_soa: Any,
) -> ComparisonResult:
    return ComparisonResult(
        label=label,
        reference_label=reference.label,
        candidate_label=candidate.label,
        max_abs_x_delta_mm=float(
            np.max(np.abs(candidate_rider_soa.x - reference_rider_soa.x))
        ),
        max_abs_z_delta_mm=float(
            np.max(np.abs(candidate_rider_soa.z - reference_rider_soa.z))
        ),
        max_abs_gamma_delta=float(
            np.max(np.abs(candidate_rider_soa.gamma - reference_rider_soa.gamma))
        ),
        candidate_elapsed_s=candidate.elapsed_s,
        reference_elapsed_s=reference.elapsed_s,
        speed_ratio_reference_over_candidate=(
            reference.elapsed_s / candidate.elapsed_s
            if candidate.elapsed_s > 0.0
            else float("inf")
        ),
    )


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    steps = args.steps
    h_step = args.h_step
    small_n = args.small_n
    scale_n = args.scale_n
    active_count = min(args.active_count, scale_n)

    zero_full, zero_full_rider, _zero_full_driver = _run_probe_case(
        label="zero_charge_full",
        n_particles=small_n,
        steps=steps,
        h_step=h_step,
        charge_scale=0.0,
        active_count=None,
        causal_history_pruning=False,
    )
    zero_pseudo, zero_pseudo_rider, _zero_pseudo_driver = _run_probe_case(
        label="zero_charge_pseudo_reduced",
        n_particles=small_n,
        steps=steps,
        h_step=h_step,
        charge_scale=0.0,
        active_count=min(args.active_count, small_n),
        causal_history_pruning=True,
    )
    weak_full, weak_full_rider, _weak_full_driver = _run_probe_case(
        label="weak_charge_full_small_n",
        n_particles=small_n,
        steps=steps,
        h_step=h_step,
        charge_scale=args.charge_scale,
        active_count=None,
        causal_history_pruning=False,
    )
    weak_pseudo, weak_pseudo_rider, _weak_pseudo_driver = _run_probe_case(
        label="weak_charge_pseudo_small_n",
        n_particles=small_n,
        steps=steps,
        h_step=h_step,
        charge_scale=args.charge_scale,
        active_count=min(args.active_count, small_n),
        causal_history_pruning=True,
    )
    scale_pseudo, _scale_pseudo_rider, _scale_pseudo_driver = _run_probe_case(
        label="weak_charge_pseudo_scale_n",
        n_particles=scale_n,
        steps=steps,
        h_step=h_step,
        charge_scale=args.charge_scale,
        active_count=active_count,
        causal_history_pruning=True,
    )

    results = [zero_full, zero_pseudo, weak_full, weak_pseudo, scale_pseudo]
    comparisons = [
        _compare(
            label="zero_charge_full_vs_pseudo",
            reference=zero_full,
            candidate=zero_pseudo,
            reference_rider_soa=zero_full_rider,
            candidate_rider_soa=zero_pseudo_rider,
        ),
        _compare(
            label="weak_charge_full_vs_pseudo_small_n",
            reference=weak_full,
            candidate=weak_pseudo,
            reference_rider_soa=weak_full_rider,
            candidate_rider_soa=weak_pseudo_rider,
        ),
    ]

    if args.include_full_scale:
        scale_full, scale_full_rider, _scale_full_driver = _run_probe_case(
            label="weak_charge_full_scale_n",
            n_particles=scale_n,
            steps=steps,
            h_step=h_step,
            charge_scale=args.charge_scale,
            active_count=None,
            causal_history_pruning=False,
        )
        results.append(scale_full)
        comparisons.append(
            _compare(
                label="weak_charge_full_vs_pseudo_scale_n",
                reference=scale_full,
                candidate=scale_pseudo,
                reference_rider_soa=scale_full_rider,
                candidate_rider_soa=_scale_pseudo_rider,
            )
        )

    return {
        "results": [asdict(result) for result in results],
        "comparisons": [asdict(comparison) for comparison in comparisons],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-n", type=int, default=24)
    parser.add_argument("--scale-n", type=int, default=128)
    parser.add_argument("--active-count", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--h-step", type=float, default=1.0e-4)
    parser.add_argument("--charge-scale", type=float, default=2.0e-2)
    parser.add_argument("--include-full-scale", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output = run_probe(args)
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print("Pseudo-grid feasibility probe")
        print("=============================")
        for result in output["results"]:
            print(
                f"{result['label']}: N={result['particle_count']} active={result['active_count']} "
                f"elapsed={result['elapsed_s']:.4f}s finite={result['finite']} "
                f"gamma=[{result['min_gamma']:.6g}, {result['max_gamma']:.6g}] "
                f"retained=({result['retained_rider_start_index']}, "
                f"{result['retained_driver_start_index']})"
            )
        print()
        for comparison in output["comparisons"]:
            print(
                f"{comparison['label']}: dx={comparison['max_abs_x_delta_mm']:.3e} mm, "
                f"dz={comparison['max_abs_z_delta_mm']:.3e} mm, "
                f"dgamma={comparison['max_abs_gamma_delta']:.3e}, "
                f"speed_ratio={comparison['speed_ratio_reference_over_candidate']:.2f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
