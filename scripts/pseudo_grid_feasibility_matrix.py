"""Run a pseudo-grid feasibility matrix.

The matrix is designed for local screening, not formal validation. It can cover
stationary cases, crossing cases, instantaneous or retarded same-bunch
space-charge cases, adaptive-timestep crossings, stronger charge regimes, and
longer stability windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.pseudo_grid_feasibility_probe import (  # noqa: E402
    _compare,
    _run_probe_case,
)


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_space_charge_modes(value: str) -> list[bool]:
    modes: list[bool] = []
    for item in value.split(","):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode in {"instant", "instantaneous", "coulomb"}:
            modes.append(False)
        elif mode in {"retarded", "lw"}:
            modes.append(True)
        else:
            raise ValueError("space-charge modes must be instantaneous and/or retarded")
    return modes


def _tag_float(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-", "m").replace(".", "p")


def _run_case_pair(
    *,
    scenario: str,
    n_particles: int,
    active_count: int,
    passive_neighbor_count: int,
    steps: int,
    h_step: float,
    charge_scale: float,
    z_separation_mm: float,
    rider_beta_z: float,
    driver_beta_z: float,
    full_reference: bool,
    space_charge_enabled: bool = False,
    space_charge_retarded: bool = False,
    space_charge_softening_mm: float = 0.3,
    space_charge_min_retarded_steps: int | None = None,
    adaptive_timestep_enabled: bool = False,
    adaptive_energy_jump_threshold: float = 0.1,
    adaptive_timestep_reduction_factor: int = 3,
    adaptive_min_timestep_factor: float = 1.0e-3,
    adaptive_proximity_refinement_enabled: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pseudo_label = (
        f"{scenario}_pseudo_N{n_particles}_K{active_count}_M{passive_neighbor_count}"
    )
    pseudo_result, pseudo_rider_soa, _pseudo_driver_soa = _run_probe_case(
        label=pseudo_label,
        scenario=scenario,
        n_particles=n_particles,
        steps=steps,
        h_step=h_step,
        charge_scale=charge_scale,
        active_count=min(active_count, n_particles),
        passive_neighbor_count=passive_neighbor_count,
        causal_history_pruning=True,
        z_separation_mm=z_separation_mm,
        rider_beta_z=rider_beta_z,
        driver_beta_z=driver_beta_z,
        space_charge_enabled=space_charge_enabled,
        space_charge_retarded=space_charge_retarded,
        space_charge_softening_mm=space_charge_softening_mm,
        space_charge_min_retarded_steps=space_charge_min_retarded_steps,
        adaptive_timestep_enabled=adaptive_timestep_enabled,
        adaptive_energy_jump_threshold=adaptive_energy_jump_threshold,
        adaptive_timestep_reduction_factor=adaptive_timestep_reduction_factor,
        adaptive_min_timestep_factor=adaptive_min_timestep_factor,
        adaptive_proximity_refinement_enabled=adaptive_proximity_refinement_enabled,
    )
    result_rows = [asdict(pseudo_result)]
    comparison_rows: list[dict[str, Any]] = []

    if full_reference:
        full_label = f"{scenario}_full_N{n_particles}"
        full_result, full_rider_soa, _full_driver_soa = _run_probe_case(
            label=full_label,
            scenario=scenario,
            n_particles=n_particles,
            steps=steps,
            h_step=h_step,
            charge_scale=charge_scale,
            active_count=None,
            causal_history_pruning=False,
            z_separation_mm=z_separation_mm,
            rider_beta_z=rider_beta_z,
            driver_beta_z=driver_beta_z,
            space_charge_enabled=space_charge_enabled,
            space_charge_retarded=space_charge_retarded,
            space_charge_softening_mm=space_charge_softening_mm,
            space_charge_min_retarded_steps=space_charge_min_retarded_steps,
            adaptive_timestep_enabled=adaptive_timestep_enabled,
            adaptive_energy_jump_threshold=adaptive_energy_jump_threshold,
            adaptive_timestep_reduction_factor=adaptive_timestep_reduction_factor,
            adaptive_min_timestep_factor=adaptive_min_timestep_factor,
            adaptive_proximity_refinement_enabled=adaptive_proximity_refinement_enabled,
        )
        result_rows.append(asdict(full_result))
        comparison_rows.append(
            asdict(
                _compare(
                    label=f"{scenario}_full_vs_pseudo_N{n_particles}_K{active_count}_M{passive_neighbor_count}",
                    reference=full_result,
                    candidate=pseudo_result,
                    reference_rider_soa=full_rider_soa,
                    candidate_rider_soa=pseudo_rider_soa,
                )
            )
        )

    return result_rows, comparison_rows


def _base_scenarios(args: argparse.Namespace) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = [
        {
            "name": "crossing_zero_charge",
            "steps": args.crossing_steps,
            "charge_scale": 0.0,
            "z_separation_mm": args.crossing_z_separation_mm,
            "rider_beta_z": args.crossing_beta,
            "driver_beta_z": -args.crossing_beta,
            "space_charge_enabled": False,
            "space_charge_retarded": False,
            "adaptive_timestep_enabled": False,
        }
    ]

    for charge_scale in _parse_float_list(args.charge_scales):
        tag = _tag_float(charge_scale)
        scenarios.extend(
            [
                {
                    "name": f"stationary_charge_{tag}",
                    "steps": args.stationary_steps,
                    "charge_scale": charge_scale,
                    "z_separation_mm": 1.0,
                    "rider_beta_z": 0.0,
                    "driver_beta_z": 0.0,
                    "space_charge_enabled": False,
                    "space_charge_retarded": False,
                    "adaptive_timestep_enabled": False,
                },
                {
                    "name": f"crossing_charge_{tag}",
                    "steps": args.crossing_steps,
                    "charge_scale": charge_scale,
                    "z_separation_mm": args.crossing_z_separation_mm,
                    "rider_beta_z": args.crossing_beta,
                    "driver_beta_z": -args.crossing_beta,
                    "space_charge_enabled": False,
                    "space_charge_retarded": False,
                    "adaptive_timestep_enabled": False,
                },
            ]
        )

    return scenarios


def _space_charge_scenarios(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.include_space_charge:
        return []

    scenarios: list[dict[str, Any]] = []
    for retarded in _parse_space_charge_modes(args.space_charge_modes):
        mode = "retarded" if retarded else "instantaneous"
        for charge_scale in _parse_float_list(args.space_charge_scales):
            tag = _tag_float(charge_scale)
            scenarios.extend(
                [
                    {
                        "name": f"stationary_space_charge_{mode}_{tag}",
                        "steps": args.stationary_steps,
                        "charge_scale": charge_scale,
                        "z_separation_mm": 1.0,
                        "rider_beta_z": 0.0,
                        "driver_beta_z": 0.0,
                        "space_charge_enabled": True,
                        "space_charge_retarded": retarded,
                        "adaptive_timestep_enabled": False,
                    },
                    {
                        "name": f"crossing_space_charge_{mode}_{tag}",
                        "steps": args.crossing_steps,
                        "charge_scale": charge_scale,
                        "z_separation_mm": args.crossing_z_separation_mm,
                        "rider_beta_z": args.crossing_beta,
                        "driver_beta_z": -args.crossing_beta,
                        "space_charge_enabled": True,
                        "space_charge_retarded": retarded,
                        "adaptive_timestep_enabled": False,
                    },
                ]
            )
    return scenarios


def _adaptive_crossing_scenarios(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.include_adaptive_crossing:
        return []

    scenarios: list[dict[str, Any]] = []
    for charge_scale in _parse_float_list(args.charge_scales):
        tag = _tag_float(charge_scale)
        scenarios.append(
            {
                "name": f"crossing_adaptive_charge_{tag}",
                "steps": args.adaptive_crossing_steps,
                "charge_scale": charge_scale,
                "z_separation_mm": args.crossing_z_separation_mm,
                "rider_beta_z": args.crossing_beta,
                "driver_beta_z": -args.crossing_beta,
                "space_charge_enabled": False,
                "space_charge_retarded": False,
                "adaptive_timestep_enabled": True,
            }
        )

    if args.include_space_charge:
        for retarded in _parse_space_charge_modes(args.space_charge_modes):
            mode = "retarded" if retarded else "instantaneous"
            for charge_scale in _parse_float_list(args.space_charge_scales):
                tag = _tag_float(charge_scale)
                scenarios.append(
                    {
                        "name": f"crossing_adaptive_space_charge_{mode}_{tag}",
                        "steps": args.adaptive_crossing_steps,
                        "charge_scale": charge_scale,
                        "z_separation_mm": args.crossing_z_separation_mm,
                        "rider_beta_z": args.crossing_beta,
                        "driver_beta_z": -args.crossing_beta,
                        "space_charge_enabled": True,
                        "space_charge_retarded": retarded,
                        "adaptive_timestep_enabled": True,
                    }
                )
    return scenarios


def _strong_regime_scenarios(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.include_strong_regimes:
        return []

    scenarios: list[dict[str, Any]] = []
    for charge_scale in _parse_float_list(args.strong_charge_scales):
        tag = _tag_float(charge_scale)
        scenarios.append(
            {
                "name": f"crossing_strong_charge_{tag}",
                "steps": args.crossing_steps,
                "charge_scale": charge_scale,
                "z_separation_mm": args.crossing_z_separation_mm,
                "rider_beta_z": args.crossing_beta,
                "driver_beta_z": -args.crossing_beta,
                "space_charge_enabled": False,
                "space_charge_retarded": False,
                "adaptive_timestep_enabled": False,
            }
        )
        if args.include_space_charge:
            for retarded in _parse_space_charge_modes(args.space_charge_modes):
                mode = "retarded" if retarded else "instantaneous"
                scenarios.append(
                    {
                        "name": f"crossing_strong_space_charge_{mode}_{tag}",
                        "steps": args.crossing_steps,
                        "charge_scale": charge_scale,
                        "z_separation_mm": args.crossing_z_separation_mm,
                        "rider_beta_z": args.crossing_beta,
                        "driver_beta_z": -args.crossing_beta,
                        "space_charge_enabled": True,
                        "space_charge_retarded": retarded,
                        "adaptive_timestep_enabled": False,
                    }
                )
    return scenarios


def _long_stability_scenarios(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.include_long_stability:
        return []

    scenarios: list[dict[str, Any]] = []
    for charge_scale in _parse_float_list(args.long_stability_charge_scales):
        tag = _tag_float(charge_scale)
        scenarios.append(
            {
                "name": f"long_crossing_charge_{tag}",
                "steps": args.long_stability_steps,
                "charge_scale": charge_scale,
                "z_separation_mm": args.crossing_z_separation_mm,
                "rider_beta_z": args.crossing_beta,
                "driver_beta_z": -args.crossing_beta,
                "space_charge_enabled": False,
                "space_charge_retarded": False,
                "adaptive_timestep_enabled": False,
            }
        )
        if args.include_space_charge:
            for retarded in _parse_space_charge_modes(args.space_charge_modes):
                mode = "retarded" if retarded else "instantaneous"
                scenarios.append(
                    {
                        "name": f"long_crossing_space_charge_{mode}_{tag}",
                        "steps": args.long_stability_steps,
                        "charge_scale": charge_scale,
                        "z_separation_mm": args.crossing_z_separation_mm,
                        "rider_beta_z": args.crossing_beta,
                        "driver_beta_z": -args.crossing_beta,
                        "space_charge_enabled": True,
                        "space_charge_retarded": retarded,
                        "adaptive_timestep_enabled": False,
                    }
                )
    return scenarios


def run_matrix(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    particle_counts = _parse_int_list(args.particle_counts)
    active_counts = _parse_int_list(args.active_counts)
    neighbor_counts = _parse_int_list(args.neighbor_counts)
    result_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    scenarios = [
        *_base_scenarios(args),
        *_space_charge_scenarios(args),
        *_adaptive_crossing_scenarios(args),
        *_strong_regime_scenarios(args),
        *_long_stability_scenarios(args),
    ]

    for scenario in scenarios:
        for n_particles in particle_counts:
            full_reference = n_particles <= args.full_reference_max_n
            for active_count in active_counts:
                if active_count > n_particles:
                    continue
                for passive_neighbor_count in neighbor_counts:
                    if passive_neighbor_count > active_count:
                        continue
                    results, comparisons = _run_case_pair(
                        scenario=scenario["name"],
                        n_particles=n_particles,
                        active_count=active_count,
                        passive_neighbor_count=passive_neighbor_count,
                        steps=scenario["steps"],
                        h_step=args.h_step,
                        charge_scale=scenario["charge_scale"],
                        z_separation_mm=scenario["z_separation_mm"],
                        rider_beta_z=scenario["rider_beta_z"],
                        driver_beta_z=scenario["driver_beta_z"],
                        full_reference=full_reference,
                        space_charge_enabled=scenario["space_charge_enabled"],
                        space_charge_retarded=scenario["space_charge_retarded"],
                        space_charge_softening_mm=args.space_charge_softening_mm,
                        space_charge_min_retarded_steps=args.space_charge_min_retarded_steps,
                        adaptive_timestep_enabled=scenario["adaptive_timestep_enabled"],
                        adaptive_energy_jump_threshold=(
                            args.adaptive_energy_jump_threshold
                        ),
                        adaptive_timestep_reduction_factor=(
                            args.adaptive_timestep_reduction_factor
                        ),
                        adaptive_min_timestep_factor=args.adaptive_min_timestep_factor,
                        adaptive_proximity_refinement_enabled=(
                            args.adaptive_proximity_refinement
                        ),
                    )
                    result_rows.extend(results)
                    comparison_rows.extend(comparisons)

    return {"results": result_rows, "comparisons": comparison_rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(output_dir: Path, output: dict[str, list[dict[str, Any]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pseudo_grid_feasibility_matrix.json").write_text(
        json.dumps(output, indent=2, sort_keys=True)
    )
    _write_csv(output_dir / "pseudo_grid_feasibility_results.csv", output["results"])
    _write_csv(
        output_dir / "pseudo_grid_feasibility_comparisons.csv",
        output["comparisons"],
    )


def _summarize(output: dict[str, list[dict[str, Any]]]) -> None:
    results = output["results"]
    comparisons = output["comparisons"]
    finite_count = sum(1 for row in results if row["finite"])
    crossed_count = sum(1 for row in results if row["interaction_point_crossed"])
    retarded_sc_count = sum(
        1
        for row in results
        if row["space_charge_enabled"] and row["space_charge_retarded"]
    )
    adaptive_count = sum(1 for row in results if row["adaptive_timestep_enabled"])
    print("Pseudo-grid feasibility matrix")
    print("==============================")
    print(
        f"runs={len(results)} finite={finite_count}/{len(results)} "
        f"crossed={crossed_count} retarded_sc={retarded_sc_count} "
        f"adaptive={adaptive_count}"
    )
    if comparisons:
        max_dx = max(row["max_abs_x_delta_mm"] for row in comparisons)
        max_dz = max(row["max_abs_z_delta_mm"] for row in comparisons)
        max_dgamma = max(row["max_abs_gamma_delta"] for row in comparisons)
        speed_ratios = np.array(
            [row["speed_ratio_reference_over_candidate"] for row in comparisons],
            dtype=float,
        )
        print(
            f"comparisons={len(comparisons)} max_dx={max_dx:.3e} mm "
            f"max_dz={max_dz:.3e} mm max_dgamma={max_dgamma:.3e} "
            f"median_speed_ratio={float(np.median(speed_ratios)):.2f}x"
        )
        print("slowest/least speedup comparisons:")
        for row in sorted(
            comparisons, key=lambda item: item["speed_ratio_reference_over_candidate"]
        )[:5]:
            print(
                f"  {row['label']}: speed={row['speed_ratio_reference_over_candidate']:.2f}x "
                f"dx={row['max_abs_x_delta_mm']:.3e} dz={row['max_abs_z_delta_mm']:.3e}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particle-counts", default="24,128")
    parser.add_argument("--active-counts", default="8,16")
    parser.add_argument("--neighbor-counts", default="2,4")
    parser.add_argument("--full-reference-max-n", type=int, default=24)
    parser.add_argument("--stationary-steps", type=int, default=4)
    parser.add_argument("--crossing-steps", type=int, default=24)
    parser.add_argument("--h-step", type=float, default=1.0e-4)
    parser.add_argument("--charge-scales", default="2.0e-2")
    parser.add_argument("--include-space-charge", action="store_true")
    parser.add_argument("--space-charge-scales", default="5.0e-3")
    parser.add_argument("--space-charge-modes", default="instantaneous")
    parser.add_argument("--space-charge-softening-mm", type=float, default=0.3)
    parser.add_argument("--space-charge-min-retarded-steps", type=int)
    parser.add_argument("--include-adaptive-crossing", action="store_true")
    parser.add_argument("--adaptive-crossing-steps", type=int, default=24)
    parser.add_argument("--adaptive-energy-jump-threshold", type=float, default=0.1)
    parser.add_argument("--adaptive-timestep-reduction-factor", type=int, default=3)
    parser.add_argument("--adaptive-min-timestep-factor", type=float, default=1.0e-3)
    parser.add_argument("--adaptive-proximity-refinement", action="store_true")
    parser.add_argument("--include-strong-regimes", action="store_true")
    parser.add_argument("--strong-charge-scales", default="5.0e-2,1.0e-1")
    parser.add_argument("--include-long-stability", action="store_true")
    parser.add_argument("--long-stability-steps", type=int, default=96)
    parser.add_argument("--long-stability-charge-scales", default="2.0e-2")
    parser.add_argument("--crossing-beta", type=float, default=0.12)
    parser.add_argument("--crossing-z-separation-mm", type=float, default=0.06)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    output = run_matrix(args)
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
