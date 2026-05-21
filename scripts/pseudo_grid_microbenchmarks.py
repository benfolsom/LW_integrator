"""Microbenchmark pseudo-grid reduced-mode overheads.

This script times the main reduced-mode phases independently so small-N probe
results can be interpreted against active solve and bookkeeping overheads.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import C_MMNS  # noqa: E402
from core.integration_runner import (  # noqa: E402
    SelfConsistencyConfig,
    _build_partial_soa,
)
from core.pseudo_grid import (  # noqa: E402
    build_pseudo_grid_step_schedule,
    build_self_excluded_space_charge_source_charges,
    initialize_pseudo_grid_planner_state,
    record_pseudo_grid_history_times,
    reconstruct_full_state_from_active_result,
    slice_particle_state,
    slice_trajectory_particle_history,
)
from core.self_consistency import self_consistent_step  # noqa: E402
from core.equations import retarded_equations_of_motion  # noqa: E402
from core.types import (  # noqa: E402
    ChronoMatchingMode,
    ParticleState,
    PseudoGridConfig,
    SimulationType,
    SpaceChargeConfig,
    StartupMode,
    Trajectory,
    TrajectoryArrays,
)


@dataclass(frozen=True)
class MicrobenchmarkResult:
    label: str
    particle_count: int
    active_count: int
    neighbor_count: int
    history_steps: int
    repeats: int
    active_solve_repeats: int
    space_charge_enabled: bool
    schedule_us: float
    observer_slice_us: float
    source_slice_us: float
    space_charge_matrix_us: float | None
    active_solve_us: float | None
    reconstruct_us: float
    overhead_without_active_solve_us: float
    retained_source_start_index: int | None
    passive_count: int


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _median_us(fn: Callable[[], Any], repeats: int, *, warmups: int = 1) -> float:
    for _ in range(max(0, warmups)):
        fn()
    elapsed: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        elapsed.append((time.perf_counter() - start) * 1.0e6)
    return float(np.median(np.asarray(elapsed, dtype=float)))


def _make_bunch(
    *,
    n_particles: int,
    z_mm: float,
    beta_z: float,
    charge_scale: float,
    seed: int,
) -> ParticleState:
    rng = np.random.default_rng(seed)
    if abs(beta_z) >= 1.0:
        raise ValueError("beta_z must satisfy |beta_z| < 1")
    gamma_value = 1.0 / np.sqrt(1.0 - beta_z**2)
    mass = np.ones(n_particles, dtype=float)
    zeros = np.zeros(n_particles, dtype=float)
    x = np.linspace(-0.75, 0.75, n_particles) + rng.normal(
        0.0,
        0.02,
        n_particles,
    )
    y = rng.normal(0.0, 0.01, n_particles)
    q_pattern = rng.normal(0.0, 1.0, n_particles)
    q_pattern -= float(np.mean(q_pattern))
    max_abs_charge = float(np.max(np.abs(q_pattern))) if n_particles else 0.0
    if max_abs_charge > 0.0:
        q_pattern /= max_abs_charge
    return {
        "x": x.astype(float),
        "y": y.astype(float),
        "z": np.full(n_particles, z_mm, dtype=float),
        "t": zeros.copy(),
        "Px": zeros.copy(),
        "Py": zeros.copy(),
        "Pz": np.full(n_particles, gamma_value * C_MMNS * beta_z, dtype=float),
        "Pt": gamma_value * mass * C_MMNS,
        "gamma": np.full(n_particles, gamma_value, dtype=float),
        "bx": zeros.copy(),
        "by": zeros.copy(),
        "bz": np.full(n_particles, beta_z, dtype=float),
        "bdotx": zeros.copy(),
        "bdoty": zeros.copy(),
        "bdotz": zeros.copy(),
        "q": charge_scale * q_pattern,
        "m": mass,
        "char_time": np.full(n_particles, 1.0e-3, dtype=float),
        "origin_x": x.astype(float).copy(),
        "origin_y": y.astype(float).copy(),
        "origin_z": np.full(n_particles, z_mm, dtype=float),
        "beta_avg_x": zeros.copy(),
        "beta_avg_y": zeros.copy(),
        "beta_avg_z": np.full(n_particles, beta_z, dtype=float),
        "beta_samples": np.ones(n_particles, dtype=float),
    }


def _copy_state(state: ParticleState) -> ParticleState:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in state.items()
    }


def _advance_state(
    state: ParticleState, *, h_step: float, step_idx: int
) -> ParticleState:
    advanced = _copy_state(state)
    gamma = np.asarray(advanced["gamma"], dtype=float)
    beta_z = np.asarray(advanced["bz"], dtype=float)
    coordinate_dt = h_step * gamma
    advanced["z"] = np.asarray(advanced["origin_z"], dtype=float) + (
        float(step_idx) * coordinate_dt * C_MMNS * beta_z
    )
    advanced["t"] = float(step_idx) * coordinate_dt
    return advanced


def _make_histories(
    *,
    n_particles: int,
    history_steps: int,
    h_step: float,
    charge_scale: float,
) -> tuple[Trajectory, Trajectory]:
    rider0 = _make_bunch(
        n_particles=n_particles,
        z_mm=-0.03,
        beta_z=0.12,
        charge_scale=charge_scale,
        seed=100 + n_particles,
    )
    driver0 = _make_bunch(
        n_particles=n_particles,
        z_mm=0.03,
        beta_z=-0.12,
        charge_scale=-charge_scale,
        seed=200 + n_particles,
    )
    rider_history = [
        _advance_state(rider0, h_step=h_step, step_idx=step_idx)
        for step_idx in range(history_steps)
    ]
    driver_history = [
        _advance_state(driver0, h_step=h_step, step_idx=step_idx)
        for step_idx in range(history_steps)
    ]
    return rider_history, driver_history


def _make_planner_state(
    rider_history: Trajectory,
    driver_history: Trajectory,
    *,
    pair_reuse_window: int,
):
    planner_state = initialize_pseudo_grid_planner_state(
        rider_particle_count=len(np.asarray(rider_history[-1]["x"])),
        driver_particle_count=len(np.asarray(driver_history[-1]["x"])),
        pair_reuse_window=pair_reuse_window,
    )
    for rider_state, driver_state in zip(rider_history, driver_history):
        record_pseudo_grid_history_times(planner_state, rider_state, driver_state)
    return planner_state


def _active_result_state(
    previous_state: ParticleState, active_indices: np.ndarray
) -> ParticleState:
    active_state = slice_particle_state(previous_state, active_indices)
    if active_indices.size == 0:
        return active_state
    active_state["x"] = np.asarray(active_state["x"], dtype=float) + 1.0e-6
    active_state["z"] = np.asarray(active_state["z"], dtype=float) + 1.0e-6
    active_state["gamma"] = np.asarray(active_state["gamma"], dtype=float) + 1.0e-9
    return active_state


def _time_active_solve(
    *,
    h_step: float,
    observer_active_history: Trajectory,
    source_active_history: Trajectory,
    observer_active_soa: TrajectoryArrays | None,
    source_active_soa: TrajectoryArrays | None,
    space_charge: SpaceChargeConfig | None,
    pseudo_grid_space_charge_source_charges: np.ndarray | None,
) -> None:
    self_consistent_step(
        retarded_equations_of_motion,
        h_step,
        observer_active_history,
        source_active_history,
        len(observer_active_history) - 1,
        10.0,
        SimulationType.BUNCH_TO_BUNCH,
        SelfConsistencyConfig(enabled=False),
        ChronoMatchingMode.FAST,
        StartupMode.COLD_START,
        step_idx=len(observer_active_history) - 1,
        space_charge=space_charge,
        radiation_reaction_mode="power_matched_damping",
        pseudo_grid_space_charge_source_charges=pseudo_grid_space_charge_source_charges,
        traj_soa=observer_active_soa,
        traj_ext_soa=source_active_soa,
    )


def run_case(
    *,
    n_particles: int,
    active_count: int,
    neighbor_count: int,
    history_steps: int,
    h_step: float,
    charge_scale: float,
    repeats: int,
    include_space_charge: bool,
    include_active_solve: bool,
    active_solve_repeats: int,
) -> MicrobenchmarkResult:
    rider_history, driver_history = _make_histories(
        n_particles=n_particles,
        history_steps=history_steps,
        h_step=h_step,
        charge_scale=charge_scale,
    )
    config = PseudoGridConfig(
        enabled=True,
        active_rider_count=min(active_count, n_particles),
        active_driver_count=min(active_count, n_particles),
        passive_neighbor_count=min(neighbor_count, active_count),
        causal_history_pruning_enabled=True,
        causal_history_safety_margin_steps=0,
    )

    def build_schedule():
        planner_state = _make_planner_state(
            rider_history,
            driver_history,
            pair_reuse_window=config.pair_reuse_window,
        )
        return build_pseudo_grid_step_schedule(
            rider_history[-1],
            driver_history[-1],
            step_index=history_steps,
            config=config,
            planner_state=planner_state,
        )

    schedule = build_schedule()
    source_start_index = schedule.driver_history_start_index or 0
    observer_active_history = slice_trajectory_particle_history(
        rider_history,
        schedule.rider_active_indices,
    )
    source_active_history = slice_trajectory_particle_history(
        driver_history,
        schedule.driver_active_indices,
        start_index=source_start_index,
        q_override=schedule.driver_effective_source_charges,
    )
    active_result = _active_result_state(
        rider_history[-1],
        schedule.rider_active_indices,
    )
    observer_active_soa = _build_partial_soa(
        observer_active_history,
        len(observer_active_history),
    )
    source_active_soa = _build_partial_soa(
        source_active_history,
        len(source_active_history),
    )
    space_charge = (
        SpaceChargeConfig(
            enabled=True,
            retarded=True,
            softening_mm=0.3,
            min_retarded_steps=0,
        )
        if include_space_charge
        else None
    )

    sc_matrix = None
    if include_space_charge:
        sc_matrix = build_self_excluded_space_charge_source_charges(
            rider_history[-1],
            schedule.rider_active_indices,
            schedule.rider_passive_map,
            weighting_mode=config.source_weighting_mode,
        )

    schedule_us = _median_us(build_schedule, repeats)
    observer_slice_us = _median_us(
        lambda: slice_trajectory_particle_history(
            rider_history,
            schedule.rider_active_indices,
        ),
        repeats,
    )
    source_slice_us = _median_us(
        lambda: slice_trajectory_particle_history(
            driver_history,
            schedule.driver_active_indices,
            start_index=source_start_index,
            q_override=schedule.driver_effective_source_charges,
        ),
        repeats,
    )
    space_charge_matrix_us = None
    if include_space_charge:
        space_charge_matrix_us = _median_us(
            lambda: build_self_excluded_space_charge_source_charges(
                rider_history[-1],
                schedule.rider_active_indices,
                schedule.rider_passive_map,
                weighting_mode=config.source_weighting_mode,
            ),
            repeats,
        )
    reconstruct_us = _median_us(
        lambda: reconstruct_full_state_from_active_result(
            rider_history[-1],
            schedule.rider_active_indices,
            active_result,
            schedule.rider_passive_map,
        ),
        repeats,
    )
    active_solve_us = None
    if include_active_solve:
        active_solve_us = _median_us(
            lambda: _time_active_solve(
                h_step=h_step,
                observer_active_history=observer_active_history,
                source_active_history=source_active_history,
                observer_active_soa=observer_active_soa,
                source_active_soa=source_active_soa,
                space_charge=space_charge,
                pseudo_grid_space_charge_source_charges=sc_matrix,
            ),
            active_solve_repeats,
            warmups=1,
        )

    overhead_without_active_solve_us = sum(
        value
        for value in (
            schedule_us,
            observer_slice_us,
            source_slice_us,
            space_charge_matrix_us,
            reconstruct_us,
        )
        if value is not None
    )
    label = (
        f"N{n_particles}_K{config.active_rider_count}_M{config.passive_neighbor_count}"
    )
    if include_space_charge:
        label += "_retarded_sc"
    return MicrobenchmarkResult(
        label=label,
        particle_count=n_particles,
        active_count=config.active_rider_count,
        neighbor_count=config.passive_neighbor_count,
        history_steps=history_steps,
        repeats=repeats,
        active_solve_repeats=active_solve_repeats if include_active_solve else 0,
        space_charge_enabled=include_space_charge,
        schedule_us=schedule_us,
        observer_slice_us=observer_slice_us,
        source_slice_us=source_slice_us,
        space_charge_matrix_us=space_charge_matrix_us,
        active_solve_us=active_solve_us,
        reconstruct_us=reconstruct_us,
        overhead_without_active_solve_us=overhead_without_active_solve_us,
        retained_source_start_index=schedule.driver_history_start_index,
        passive_count=int(schedule.rider_passive_map.passive_indices.size),
    )


def run_microbenchmarks(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    results: list[MicrobenchmarkResult] = []
    for n_particles in _parse_int_list(args.particle_counts):
        for active_count in _parse_int_list(args.active_counts):
            if active_count > n_particles:
                continue
            for neighbor_count in _parse_int_list(args.neighbor_counts):
                if neighbor_count > active_count:
                    continue
                results.append(
                    run_case(
                        n_particles=n_particles,
                        active_count=active_count,
                        neighbor_count=neighbor_count,
                        history_steps=args.history_steps,
                        h_step=args.h_step,
                        charge_scale=args.charge_scale,
                        repeats=args.repeats,
                        include_space_charge=args.include_space_charge,
                        include_active_solve=args.include_active_solve,
                        active_solve_repeats=args.active_solve_repeats,
                    )
                )
    return {"results": [asdict(result) for result in results]}


def _write_outputs(output_dir: Path, output: dict[str, list[dict[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pseudo_grid_microbenchmarks.json").write_text(
        json.dumps(output, indent=2, sort_keys=True)
    )
    rows = output["results"]
    csv_path = output_dir / "pseudo_grid_microbenchmarks.csv"
    if not rows:
        csv_path.write_text("")
        return
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_us(value: float | None) -> str:
    return "-" if value is None else f"{value:10.1f}"


def _summarize(output: dict[str, list[dict[str, Any]]]) -> None:
    rows = output["results"]
    print("Pseudo-grid microbenchmarks")
    print("===========================")
    print(
        "label                         schedule   slice_obs   slice_src "
        "  sc_matrix active_solve reconstruct overhead_no_solve"
    )
    for row in rows:
        print(
            f"{row['label']:<28s} "
            f"{_format_us(row['schedule_us'])} "
            f"{_format_us(row['observer_slice_us'])} "
            f"{_format_us(row['source_slice_us'])} "
            f"{_format_us(row['space_charge_matrix_us'])} "
            f"{_format_us(row['active_solve_us'])} "
            f"{_format_us(row['reconstruct_us'])} "
            f"{_format_us(row['overhead_without_active_solve_us'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particle-counts", default="64,128,256")
    parser.add_argument("--active-counts", default="8,16,32")
    parser.add_argument("--neighbor-counts", default="2,4")
    parser.add_argument("--history-steps", type=int, default=24)
    parser.add_argument("--h-step", type=float, default=1.0e-4)
    parser.add_argument("--charge-scale", type=float, default=5.0e-3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--include-space-charge", action="store_true")
    parser.add_argument("--include-active-solve", action="store_true")
    parser.add_argument("--active-solve-repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.history_steps < 2:
        raise ValueError("history_steps must be at least 2")
    output = run_microbenchmarks(args)
    if args.output_dir is not None:
        _write_outputs(args.output_dir, output)
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        _summarize(output)
        if args.output_dir is not None:
            print(f"wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
