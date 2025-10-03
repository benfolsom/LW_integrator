#!/usr/bin/env python3
"""Systematic comparison utilities for legacy vs core integrators."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LEGACY_ROOT = PROJECT_ROOT / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))
VALIDATION_DIR = PROJECT_ROOT / "examples" / "validation"
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from core_vs_legacy_benchmark import (  # type: ignore[import]
    run_benchmark,
)

FIELDS_OF_INTEREST: Tuple[str, ...] = ("z", "Pt", "gamma")


@dataclass
class ScenarioResult:
    seed: int
    steps: int
    rider_metrics: Dict[str, float]
    driver_metrics: Dict[str, float]

    def max_relative_difference(self) -> float:
        rel_keys = [f"{field}_max_rel_pct" for field in FIELDS_OF_INTEREST]
        values = [
            max(self.rider_metrics[key], self.driver_metrics[key]) for key in rel_keys
        ]
        return max(values)


def _format_metric_block(metrics: Dict[str, float], prefix: str) -> Iterable[str]:
    for field in FIELDS_OF_INTEREST:
        abs_key = f"{field}_max_abs"
        rel_key = f"{field}_max_rel_pct"
        yield (
            f"    {prefix:<6s} {field:<5s} | max |Δ| = {metrics[abs_key]:.3e}"
            f", max rel = {metrics[rel_key]:.3e}%"
        )


def evaluate_scenario(seed: int, steps: int) -> ScenarioResult:
    metrics = run_benchmark(steps=steps, seed=seed, plot=False)
    return ScenarioResult(seed, steps, metrics["rider"], metrics["driver"])


def run_suite(seeds: List[int], steps_list: List[int]) -> List[ScenarioResult]:
    results: List[ScenarioResult] = []
    for seed in seeds:
        for steps in steps_list:
            print(f"Running comparison: seed={seed}, steps={steps}")
            result = evaluate_scenario(seed, steps)
            results.append(result)
            print_summary(result)
            print()
    return results


def print_summary(result: ScenarioResult) -> None:
    print(f"Scenario seed={result.seed}, steps={result.steps}")
    for line in _format_metric_block(result.rider_metrics, "rider"):
        print(line)
    for line in _format_metric_block(result.driver_metrics, "driver"):
        print(line)
    print(f"    max relative difference: {result.max_relative_difference():.3e}%")


def summarise_results(results: List[ScenarioResult]) -> Dict[str, float]:
    if not results:
        return {"max_relative_difference": float("nan"), "scenarios": 0}
    max_rel = max(result.max_relative_difference() for result in results)
    return {"max_relative_difference": max_rel, "scenarios": len(results)}


def export_results(results: List[ScenarioResult], destination: Path) -> None:
    payload = {
        "scenarios": [
            {
                "seed": result.seed,
                "steps": result.steps,
                "rider": result.rider_metrics,
                "driver": result.driver_metrics,
            }
            for result in results
        ],
        "summary": summarise_results(results),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[40],
        help="List of integration step counts to evaluate",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[12345, 20240101],
        help="Random seeds for bunch initialisation",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        help="Optional path to write the aggregated results as JSON",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_suite(args.seeds, args.steps)
    summary = summarise_results(results)

    print("=== Overall Summary ===")
    print(f"Scenarios evaluated: {summary['scenarios']}")
    print(f"Maximum relative difference: {summary['max_relative_difference']:.3e}%")

    if args.save_json is not None:
        export_results(results, args.save_json)
        print(f"Results exported to {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
