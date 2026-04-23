#!/usr/bin/env python3
"""Tabulate best runs from optimization results.

This compatibility wrapper reads optimization_results.json and displays the
best runs using the shared optimization-result summary helpers.

The preferred interface for saved-result inspection is now:

    python -m lw_integrator --results-file <optimization_results.json>

Usage:
    python scripts/tabulate_best_runs.py <path_to_optimization_results.json> [--top N]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from optimization.plugin_results_helpers import (
    parse_results_payload,
    summarize_optimization_evaluation_counts,
    summarize_optimization_top_results,
)


def format_value(value: Any, precision: int = 4) -> str:
    """Format a numeric value for display."""
    if isinstance(value, (int, float)):
        if abs(value) < 1e-3 or abs(value) > 1e4:
            return f"{value:.{precision}e}"
        else:
            return f"{value:.{precision}f}"
    return str(value)


def tabulate_optimization_results(data: Dict[str, Any], top_n: int = 20) -> None:
    """Tabulate normalized top optimization results from saved JSON."""
    top_results = summarize_optimization_top_results(data, limit=top_n)
    if not top_results:
        print("ERROR: No successful optimization results found")
        return

    counts = summarize_optimization_evaluation_counts(data)
    print("=" * 120)
    print(f"TOP {len(top_results)} OPTIMIZATION RUNS")
    print(f"Objective: {data.get('objective', 'unknown')}")
    if "all_evaluations" in data:
        print(f"Total Evaluations: {len(data.get('all_evaluations', []))}")
        print(
            f"Successful Evaluations: {counts['successful_evaluation_count']}"
        )
    print("=" * 120)
    print()

    print(
        f"{'Rank':<6} {'Eval#':<8} {'Objective':<15} {'ΔE/E (%)':<15} {'ΔE (MeV)':<15} {'Fitness':<12}"
    )
    print("-" * 120)

    for result in top_results:
        print(
            f"{result.get('rank', '?'):<6} "
            f"{result.get('evaluation', '?'):<8} "
            f"{format_value(result.get('metric_value', float('nan')), 6):<15} "
            f"{format_value(result.get('percent_energy_gain', float('nan')), 6):<15} "
            f"{format_value(result.get('delta_e_mev', float('nan')), 6):<15} "
            f"{format_value(result.get('fitness', float('nan')), 4):<12}"
        )

    print("-" * 120)
    print()

    print("=" * 120)
    print("DETAILED METRICS FOR TOP RESULTS")
    print("=" * 120)
    print()

    for result in top_results[:5]:
        label = f"RANK #{result.get('rank', '?')}"
        if result.get("evaluation") is not None:
            label += f" (Evaluation {result['evaluation']})"
        print(label)
        print("-" * 80)
        print("Parameters:")
        for key, val in result.get("parameters", {}).items():
            print(f"  {key:<40} {format_value(val, 6)}")

        print("\nObjective:")
        print(
            f"  metric_value:                           {format_value(result.get('metric_value', float('nan')), 8)}"
        )
        if result.get("fitness") is not None:
            print(
                f"  fitness:                                {format_value(result['fitness'], 8)}"
            )

        metrics = result.get("metrics", {})
        if metrics:
            print("\nMetrics:")
            for key, val in sorted(metrics.items()):
                print(f"  {key:<40} {format_value(val, 8)}")
        print()


def print_summary_statistics(data: Dict[str, Any]) -> None:
    """Print aggregate optimization evaluation statistics when available."""
    if "all_evaluations" not in data:
        return

    counts = summarize_optimization_evaluation_counts(data)
    total = len(data.get("all_evaluations", []))
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)

    print(f"Total evaluations:      {total}")
    if total > 0:
        print(
            "Successful:             "
            f"{counts['successful_evaluation_count']} "
            f"({100 * counts['successful_evaluation_count'] / total:.1f}%)"
        )
        print(
            "Halted early:           "
            f"{counts['halted_evaluation_count']} "
            f"({100 * counts['halted_evaluation_count'] / total:.1f}%)"
        )
        print(
            "Failed:                 "
            f"{counts['failed_evaluation_count']} "
            f"({100 * counts['failed_evaluation_count'] / total:.1f}%)"
        )

    if "best_value" in data:
        print(f"\nBest value found:       {format_value(data['best_value'], 8)}")
    if "function_evaluations" in data:
        print(f"Function evaluations:   {data['function_evaluations']}")
    if "timestamp" in data:
        print(f"Timestamp:              {data['timestamp']}")


def main():
    parser = argparse.ArgumentParser(
        description="Tabulate best runs from optimization results JSON"
    )
    parser.add_argument(
        "results_file", type=str, help="Path to optimization_results.json file"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top results to display (default: 20)",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "evaluations", "top_n"],
        default="auto",
        help="Source of data to tabulate (default: auto - use all_evaluations if available)",
    )

    args = parser.parse_args()

    # Load results file
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"ERROR: File not found: {results_path}")
        sys.exit(1)

    try:
        with open(results_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        sys.exit(1)

    try:
        parsed = parse_results_payload(data, m_particle_amu=1.0, amu_to_mev=931.494)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if parsed["kind"] != "optimization":
        print("ERROR: This script only supports optimization_results.json files")
        sys.exit(1)

    if args.source == "evaluations" and "all_evaluations" not in data:
        print("ERROR: No 'all_evaluations' found in results file")
        sys.exit(1)
    if args.source == "top_n" and "top_n_results" not in data:
        print("ERROR: No 'top_n_results' found in results file")
        sys.exit(1)

    if args.source == "auto":
        if "all_evaluations" in data:
            print("Using all_evaluations data (more detailed metrics)\n")
        elif "top_n_results" in data:
            print("Using top_n_results data\n")

    tabulate_optimization_results(data, top_n=args.top)
    print_summary_statistics(data)


if __name__ == "__main__":
    main()
