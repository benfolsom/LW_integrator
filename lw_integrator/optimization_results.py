"""Packaged CLI for tabulating saved optimization results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from optimization.plugin_results_helpers import (
    parse_results_payload,
    summarize_optimization_evaluation_counts,
    summarize_optimization_top_results,
)

__all__ = ["main"]


def _format_value(value: Any, precision: int = 4) -> str:
    """Format a numeric value for display."""
    if isinstance(value, (int, float)):
        if abs(value) < 1e-3 or abs(value) > 1e4:
            return f"{value:.{precision}e}"
        return f"{value:.{precision}f}"
    return str(value)


def _print_top_results(data: Dict[str, Any], *, top_n: int = 20) -> None:
    """Print normalized top optimization results from saved JSON."""
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
        print(f"Successful Evaluations: {counts['successful_evaluation_count']}")
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
            f"{_format_value(result.get('metric_value', float('nan')), 6):<15} "
            f"{_format_value(result.get('percent_energy_gain', float('nan')), 6):<15} "
            f"{_format_value(result.get('delta_e_mev', float('nan')), 6):<15} "
            f"{_format_value(result.get('fitness', float('nan')), 4):<12}"
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
        for key, value in result.get("parameters", {}).items():
            print(f"  {key:<40} {_format_value(value, 6)}")

        print("\nObjective:")
        print(
            "  metric_value:                           "
            f"{_format_value(result.get('metric_value', float('nan')), 8)}"
        )
        if result.get("fitness") is not None:
            print(
                "  fitness:                                "
                f"{_format_value(result['fitness'], 8)}"
            )

        metrics = result.get("metrics", {})
        if metrics:
            print("\nMetrics:")
            for key, value in sorted(metrics.items()):
                print(f"  {key:<40} {_format_value(value, 8)}")
        print()


def _print_summary_statistics(data: Dict[str, Any]) -> None:
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
        print(f"\nBest value found:       {_format_value(data['best_value'], 8)}")
    if "function_evaluations" in data:
        print(f"Function evaluations:   {data['function_evaluations']}")
    if "timestamp" in data:
        print(f"Timestamp:              {data['timestamp']}")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse optimization results CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Tabulate best runs from optimization results JSON"
    )
    parser.add_argument(
        "results_file", type=Path, help="Path to optimization_results.json file"
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
    return parser.parse_args(argv)


def _load_optimization_results(results_path: Path) -> Dict[str, Any]:
    """Load and validate an optimization results file."""
    if not results_path.exists():
        raise FileNotFoundError(f"File not found: {results_path}")

    with open(results_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    parsed = parse_results_payload(
        data,
        m_particle_amu=1.0,
        amu_to_mev=931.494,
    )
    if parsed["kind"] != "optimization":
        raise ValueError("This command only supports optimization_results.json files")
    return data


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the packaged optimization results CLI."""
    args = _parse_args(argv)

    try:
        data = _load_optimization_results(args.results_file)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse JSON: {exc}")
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.source == "evaluations" and "all_evaluations" not in data:
        print("ERROR: No 'all_evaluations' found in results file")
        return 1
    if args.source == "top_n" and "top_n_results" not in data:
        print("ERROR: No 'top_n_results' found in results file")
        return 1

    if args.source == "auto":
        if "all_evaluations" in data:
            print("Using all_evaluations data (more detailed metrics)\n")
        elif "top_n_results" in data:
            print("Using top_n_results data\n")

    _print_top_results(data, top_n=args.top)
    _print_summary_statistics(data)
    return 0
