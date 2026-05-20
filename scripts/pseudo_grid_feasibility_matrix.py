"""Run a small pseudo-grid feasibility matrix.

The matrix is designed for quick local screening, not formal validation. It
includes stationary weak-charge cases and crossing cases where both bunches have
time to reach and pass the nominal interaction point at z=0.
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


def run_matrix(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    particle_counts = _parse_int_list(args.particle_counts)
    active_counts = _parse_int_list(args.active_counts)
    neighbor_counts = _parse_int_list(args.neighbor_counts)
    result_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    scenarios = [
        {
            "name": "stationary_weak",
            "steps": args.stationary_steps,
            "charge_scale": args.charge_scale,
            "z_separation_mm": 1.0,
            "rider_beta_z": 0.0,
            "driver_beta_z": 0.0,
        },
        {
            "name": "crossing_zero_charge",
            "steps": args.crossing_steps,
            "charge_scale": 0.0,
            "z_separation_mm": args.crossing_z_separation_mm,
            "rider_beta_z": args.crossing_beta,
            "driver_beta_z": -args.crossing_beta,
        },
        {
            "name": "crossing_weak",
            "steps": args.crossing_steps,
            "charge_scale": args.charge_scale,
            "z_separation_mm": args.crossing_z_separation_mm,
            "rider_beta_z": args.crossing_beta,
            "driver_beta_z": -args.crossing_beta,
        },
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
    print("Pseudo-grid feasibility matrix")
    print("==============================")
    print(
        f"runs={len(results)} finite={finite_count}/{len(results)} crossed={crossed_count}"
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
    parser.add_argument("--charge-scale", type=float, default=2.0e-2)
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
