#!/usr/bin/env python3
"""Tabulate best runs from optimization results.

This script reads optimization_results.json and displays the best runs
in a formatted table, including full metrics for diagnostics.

Usage:
    python scripts/tabulate_best_runs.py <path_to_optimization_results.json> [--top N]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def format_value(value: Any, precision: int = 4) -> str:
    """Format a numeric value for display."""
    if isinstance(value, (int, float)):
        if abs(value) < 1e-3 or abs(value) > 1e4:
            return f"{value:.{precision}e}"
        else:
            return f"{value:.{precision}f}"
    return str(value)


def format_parameters(params: Dict[str, Any], max_width: int = 80) -> str:
    """Format parameters dictionary as compact string."""
    param_strs = [f"{k}={format_value(v, 3)}" for k, v in params.items()]
    result = ", ".join(param_strs)
    if len(result) > max_width:
        # Truncate and add ellipsis
        result = result[: max_width - 3] + "..."
    return result


def tabulate_from_all_evaluations(data: Dict[str, Any], top_n: int = 20) -> None:
    """Tabulate best runs from all_evaluations list."""
    if "all_evaluations" not in data:
        print("ERROR: No 'all_evaluations' found in results file")
        return

    all_evals = data["all_evaluations"]
    objective = data.get("objective", "unknown")
    maximize = "max" in objective.lower()

    # Filter successful evaluations
    successful = [
        e
        for e in all_evals
        if not e.get("failed", False) and not e.get("halted_early", False)
    ]

    if not successful:
        print("ERROR: No successful evaluations found")
        return

    # Sort by objective value (or raw_objective_value if available)
    def get_sort_key(eval_rec):
        if "raw_objective_value" in eval_rec:
            return eval_rec["raw_objective_value"]
        elif "objective_value" in eval_rec:
            return eval_rec["objective_value"]
        return float("-inf") if maximize else float("inf")

    sorted_evals = sorted(successful, key=get_sort_key, reverse=maximize)
    top_evals = sorted_evals[:top_n]

    print("=" * 120)
    print(f"TOP {len(top_evals)} OPTIMIZATION RUNS")
    print(f"Objective: {objective}")
    print(f"Total Evaluations: {len(all_evals)}")
    print(f"Successful Evaluations: {len(successful)}")
    print("=" * 120)
    print()

    # Print header
    print(
        f"{'Rank':<6} {'Eval#':<8} {'Objective':<15} {'ΔE/E (%)':<15} {'ΔE (MeV)':<15} {'Penalty':<12}"
    )
    print("-" * 120)

    # Print top runs
    for rank, eval_rec in enumerate(top_evals, 1):
        eval_num = eval_rec.get("evaluation", "?")
        obj_val = eval_rec.get(
            "raw_objective_value", eval_rec.get("objective_value", float("nan"))
        )
        penalty = eval_rec.get("soft_penalty", 0.0)

        # Extract key metrics
        metrics = eval_rec.get("metrics", {})
        pct_energy = metrics.get(
            "max_percent_energy_gain", metrics.get("percent_delta_e", float("nan"))
        )
        delta_e_mev = metrics.get(
            "delta_e_mev", metrics.get("rider_delta_e_mev", float("nan"))
        )

        print(
            f"{rank:<6} {eval_num:<8} {format_value(obj_val, 6):<15} "
            f"{format_value(pct_energy, 6):<15} {format_value(delta_e_mev, 6):<15} "
            f"{format_value(penalty, 4):<12}"
        )

    print("-" * 120)
    print()

    # Print detailed info for top 5
    print("=" * 120)
    print("DETAILED METRICS FOR TOP 5 RUNS")
    print("=" * 120)
    print()

    for rank, eval_rec in enumerate(top_evals[:5], 1):
        eval_num = eval_rec.get("evaluation", "?")
        print(f"RANK #{rank} (Evaluation {eval_num})")
        print("-" * 80)

        # Parameters
        print("Parameters:")
        params = eval_rec.get("parameters", {})
        for key, val in params.items():
            print(f"  {key:<40} {format_value(val, 6)}")

        # Objective values
        print("\nObjective:")
        obj_val = eval_rec.get(
            "raw_objective_value", eval_rec.get("objective_value", float("nan"))
        )
        print(f"  objective_value (raw):                   {format_value(obj_val, 8)}")
        if "soft_penalty" in eval_rec:
            print(
                f"  soft_penalty:                            {format_value(eval_rec['soft_penalty'], 6)}"
            )
        if "fitness" in eval_rec:
            print(
                f"  fitness (for optimizer):                 {format_value(eval_rec['fitness'], 8)}"
            )

        # Metrics
        metrics = eval_rec.get("metrics", {})
        if metrics:
            print("\nMetrics:")
            for key, val in sorted(metrics.items()):
                print(f"  {key:<40} {format_value(val, 8)}")

        print()


def tabulate_from_top_n_results(data: Dict[str, Any]) -> None:
    """Tabulate from top_n_results in the JSON."""
    if "top_n_results" not in data:
        print("ERROR: No 'top_n_results' found in results file")
        return

    top_n_results = data["top_n_results"]
    objective = data.get("objective", "unknown")

    print("=" * 120)
    print(f"TOP {len(top_n_results)} RESULTS FROM FINAL POPULATION")
    print(f"Objective: {objective}")
    print("=" * 120)
    print()

    # Print header
    print(
        f"{'Rank':<6} {'Metric Value':<15} {'Fitness':<15} {'Has Metrics?':<13} {'Parameters'}"
    )
    print("-" * 120)

    # Print results
    for result in top_n_results:
        rank = result.get("rank", "?")
        metric_val = result.get("metric_value", float("nan"))
        fitness = result.get("fitness", float("nan"))
        has_metrics = "Yes" if "metrics" in result else "No"
        params = result.get("parameters", {})
        param_str = format_parameters(params, max_width=60)

        print(
            f"{rank:<6} {format_value(metric_val, 6):<15} "
            f"{format_value(fitness, 6):<15} {has_metrics:<13} {param_str}"
        )

    print("-" * 120)
    print()

    # If metrics are available, show detailed view for top 3
    top_with_metrics = [r for r in top_n_results if "metrics" in r]
    if top_with_metrics:
        print("=" * 120)
        print("DETAILED METRICS FOR TOP 3 RESULTS")
        print("=" * 120)
        print()

        for result in top_with_metrics[:3]:
            rank = result.get("rank", "?")
            print(f"RANK #{rank}")
            print("-" * 80)

            # Parameters
            print("Parameters:")
            params = result.get("parameters", {})
            for key, val in params.items():
                print(f"  {key:<40} {format_value(val, 6)}")

            # Metrics
            metrics = result.get("metrics", {})
            if metrics:
                print("\nMetrics:")
                for key, val in sorted(metrics.items()):
                    print(f"  {key:<40} {format_value(val, 8)}")

            print()


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

    # Determine which tabulation method to use
    has_evaluations = "all_evaluations" in data
    has_top_n = "top_n_results" in data

    if args.source == "auto":
        if has_evaluations:
            print("Using all_evaluations data (more detailed metrics)\n")
            tabulate_from_all_evaluations(data, top_n=args.top)
        elif has_top_n:
            print("Using top_n_results data\n")
            tabulate_from_top_n_results(data)
        else:
            print("ERROR: No recognized data structure found in results file")
            sys.exit(1)
    elif args.source == "evaluations":
        if has_evaluations:
            tabulate_from_all_evaluations(data, top_n=args.top)
        else:
            print("ERROR: No 'all_evaluations' found in results file")
            sys.exit(1)
    elif args.source == "top_n":
        if has_top_n:
            tabulate_from_top_n_results(data)
        else:
            print("ERROR: No 'top_n_results' found in results file")
            sys.exit(1)

    # Print summary statistics
    if has_evaluations:
        print("\n" + "=" * 120)
        print("SUMMARY STATISTICS")
        print("=" * 120)

        all_evals = data["all_evaluations"]
        successful = [e for e in all_evals if not e.get("failed", False)]
        halted = [e for e in all_evals if e.get("halted_early", False)]
        failed = [e for e in all_evals if e.get("failed", False)]

        print(f"Total evaluations:      {len(all_evals)}")
        print(
            f"Successful:             {len(successful)} ({100 * len(successful) / len(all_evals):.1f}%)"
        )
        print(
            f"Halted early:           {len(halted)} ({100 * len(halted) / len(all_evals):.1f}%)"
        )
        print(
            f"Failed:                 {len(failed)} ({100 * len(failed) / len(all_evals):.1f}%)"
        )

        if "best_value" in data:
            print(f"\nBest value found:       {format_value(data['best_value'], 8)}")
        if "function_evaluations" in data:
            print(f"Function evaluations:   {data['function_evaluations']}")
        if "timestamp" in data:
            print(f"Timestamp:              {data['timestamp']}")


if __name__ == "__main__":
    main()
