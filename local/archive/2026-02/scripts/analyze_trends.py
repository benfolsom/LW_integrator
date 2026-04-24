#!/usr/bin/env python3
"""
Trend Analysis: Good vs Bad Optimization Runs

Analyzes parameter trends to identify what makes a good run vs a bad run.
Compares top performers against bottom performers to find key differentiators.

Usage:
    python3 local/analyze_trends.py [--percentile P] [--detail]
"""

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def parse_optimization_log(log_path):
    """Parse a single optimization log file and extract evaluations."""
    results = []
    current_eval = None

    try:
        with open(log_path, "r") as f:
            for line in f:
                # Match evaluation lines with parameters
                eval_match = re.search(r"Evaluation (\d+): (.+)", line)
                if eval_match:
                    eval_num = int(eval_match.group(1))
                    params_str = eval_match.group(2)

                    # Parse parameter key=value pairs
                    params = {}
                    for param_match in re.finditer(r"(\w+)=([\d.e+-]+)", params_str):
                        key = param_match.group(1)
                        value_str = param_match.group(2)
                        try:
                            value = float(value_str)
                            params[key] = value
                        except ValueError:
                            params[key] = value_str

                    current_eval = {
                        "eval_num": eval_num,
                        "params": params,
                        "log_file": log_path.name,
                    }

                # Match energy gain metric
                gain_match = re.search(
                    r"max_percent_energy_gain:\s*([-\d.e+-]+)%", line
                )
                if gain_match and current_eval:
                    gain = float(gain_match.group(1))
                    current_eval["energy_gain"] = gain
                    results.append(current_eval)
                    current_eval = None

    except Exception as e:
        pass

    return results


def analyze_parameter_trends(good_runs, bad_runs, param_name):
    """Analyze a specific parameter comparing good vs bad runs."""
    good_values = [
        r["params"].get(param_name) for r in good_runs if param_name in r["params"]
    ]
    bad_values = [
        r["params"].get(param_name) for r in bad_runs if param_name in r["params"]
    ]

    good_values = [v for v in good_values if isinstance(v, (int, float))]
    bad_values = [v for v in bad_values if isinstance(v, (int, float))]

    if not good_values or not bad_values:
        return None

    good_avg = statistics.mean(good_values)
    bad_avg = statistics.mean(bad_values)
    good_median = statistics.median(good_values)
    bad_median = statistics.median(bad_values)
    good_std = statistics.stdev(good_values) if len(good_values) > 1 else 0
    bad_std = statistics.stdev(bad_values) if len(bad_values) > 1 else 0

    # Calculate relative difference
    if bad_avg != 0:
        rel_diff_avg = ((good_avg - bad_avg) / abs(bad_avg)) * 100
    else:
        rel_diff_avg = float("inf") if good_avg > 0 else 0

    if bad_median != 0:
        rel_diff_median = ((good_median - bad_median) / abs(bad_median)) * 100
    else:
        rel_diff_median = float("inf") if good_median > 0 else 0

    return {
        "param": param_name,
        "good_avg": good_avg,
        "bad_avg": bad_avg,
        "good_median": good_median,
        "bad_median": bad_median,
        "good_std": good_std,
        "bad_std": bad_std,
        "rel_diff_avg": rel_diff_avg,
        "rel_diff_median": rel_diff_median,
        "good_min": min(good_values),
        "good_max": max(good_values),
        "bad_min": min(bad_values),
        "bad_max": max(bad_values),
    }


def format_value(value):
    """Format a value for display."""
    if abs(value) < 1e-3 or abs(value) > 1e6:
        return f"{value:.4e}"
    else:
        return f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze parameter trends: good vs bad runs"
    )
    parser.add_argument(
        "--percentile",
        "-p",
        type=int,
        default=10,
        help="Percentile for top/bottom split (default: 10 = top/bottom 10%%)",
    )
    parser.add_argument(
        "--detail",
        "-d",
        action="store_true",
        help="Show detailed statistics including std dev and ranges",
    )

    args = parser.parse_args()

    # Find logcache
    script_dir = Path(__file__).parent
    logcache_dir = script_dir.parent / "logcache"

    if not logcache_dir.exists():
        print(f"Error: logcache directory not found at {logcache_dir}")
        sys.exit(1)

    # Parse all logs
    print("Parsing optimization logs...")
    all_results = []
    log_files = sorted(logcache_dir.glob("*optimization*.log"))

    for log_file in log_files:
        results = parse_optimization_log(log_file)
        if results:
            all_results.extend(results)

    if not all_results:
        print("No results found!")
        sys.exit(1)

    print(f"Found {len(all_results)} total evaluations")
    print()

    # Sort by energy gain
    all_results.sort(key=lambda x: x.get("energy_gain", float("-inf")), reverse=True)

    # Split into good and bad runs based on percentile
    percentile = args.percentile
    n_top = max(1, int(len(all_results) * percentile / 100))
    n_bottom = n_top

    good_runs = all_results[:n_top]
    bad_runs = all_results[-n_bottom:]

    print("=" * 100)
    print("📊 TREND ANALYSIS: GOOD vs BAD RUNS")
    print("=" * 100)
    print()
    print(f"Total evaluations: {len(all_results)}")
    print(f"Top {percentile}% (Good runs): {len(good_runs)} evaluations")
    print(f"Bottom {percentile}% (Bad runs): {len(bad_runs)} evaluations")
    print()
    print(f"Best energy gain:  {good_runs[0]['energy_gain']:.6e}%")
    print(f"Worst energy gain: {bad_runs[-1]['energy_gain']:.6e}%")
    print()

    # Get all parameter names
    all_params = set()
    for result in all_results:
        all_params.update(result["params"].keys())

    # Analyze each parameter
    trends = []
    for param in sorted(all_params):
        trend = analyze_parameter_trends(good_runs, bad_runs, param)
        if trend:
            trends.append(trend)

    # Sort by absolute relative difference
    trends.sort(key=lambda x: abs(x["rel_diff_avg"]), reverse=True)

    print("=" * 100)
    print("🔍 KEY DIFFERENTIATORS (sorted by impact)")
    print("=" * 100)
    print()

    for i, trend in enumerate(trends, 1):
        param = trend["param"]
        good_avg = trend["good_avg"]
        bad_avg = trend["bad_avg"]
        rel_diff = trend["rel_diff_avg"]

        # Determine trend direction
        if abs(rel_diff) < 1:
            indicator = "≈"
            impact = "MINIMAL"
        elif abs(rel_diff) < 10:
            indicator = "↗" if rel_diff > 0 else "↘"
            impact = "LOW"
        elif abs(rel_diff) < 50:
            indicator = "⬆" if rel_diff > 0 else "⬇"
            impact = "MODERATE"
        else:
            indicator = "🔺" if rel_diff > 0 else "🔻"
            impact = "HIGH"

        print(f"{i}. {param}")
        print(f"   Impact: {impact} {indicator} ({rel_diff:+.1f}% difference)")
        print(f"   Good runs (avg): {format_value(good_avg)}")
        print(f"   Bad runs (avg):  {format_value(bad_avg)}")

        if args.detail:
            print(f"   Good runs (median): {format_value(trend['good_median'])}")
            print(f"   Bad runs (median):  {format_value(trend['bad_median'])}")
            print(f"   Good runs (std):    {format_value(trend['good_std'])}")
            print(f"   Bad runs (std):     {format_value(trend['bad_std'])}")
            print(
                f"   Good runs (range):  [{format_value(trend['good_min'])}, {format_value(trend['good_max'])}]"
            )
            print(
                f"   Bad runs (range):   [{format_value(trend['bad_min'])}, {format_value(trend['bad_max'])}]"
            )

        print()

    # Summary of key findings
    print("=" * 100)
    print("💡 KEY FINDINGS")
    print("=" * 100)
    print()

    high_impact = [t for t in trends if abs(t["rel_diff_avg"]) >= 50]
    moderate_impact = [t for t in trends if 10 <= abs(t["rel_diff_avg"]) < 50]
    low_impact = [t for t in trends if 1 <= abs(t["rel_diff_avg"]) < 10]

    print(f"High-impact parameters ({len(high_impact)}):")
    for t in high_impact:
        direction = "HIGHER" if t["rel_diff_avg"] > 0 else "LOWER"
        print(
            f"  • {t['param']:30s}: Good runs use {direction} values ({t['rel_diff_avg']:+.1f}%)"
        )
    print()

    if moderate_impact:
        print(f"Moderate-impact parameters ({len(moderate_impact)}):")
        for t in moderate_impact:
            direction = "HIGHER" if t["rel_diff_avg"] > 0 else "LOWER"
            print(
                f"  • {t['param']:30s}: Good runs use {direction} values ({t['rel_diff_avg']:+.1f}%)"
            )
        print()

    # Recommendations
    print("=" * 100)
    print("✅ RECOMMENDATIONS FOR GOOD RUNS")
    print("=" * 100)
    print()

    for t in high_impact[:5]:  # Top 5 high-impact parameters
        if t["rel_diff_avg"] > 0:
            print(f"✓ INCREASE {t['param']}")
            print(
                f"  Target range: {format_value(t['good_min'])} - {format_value(t['good_max'])}"
            )
            print(f"  Optimal (avg): {format_value(t['good_avg'])}")
        else:
            print(f"✓ DECREASE {t['param']}")
            print(
                f"  Target range: {format_value(t['good_min'])} - {format_value(t['good_max'])}"
            )
            print(f"  Optimal (avg): {format_value(t['good_avg'])}")
        print()

    # Distribution overlap analysis
    print("=" * 100)
    print("📈 PARAMETER DISTRIBUTION OVERLAP")
    print("=" * 100)
    print()

    for t in trends:
        # Check if good/bad ranges overlap
        overlap = not (t["good_max"] < t["bad_min"] or t["bad_max"] < t["good_min"])
        if not overlap:
            print(f"🎯 {t['param']:30s}: NO OVERLAP - Clear separator!")
            print(
                f"   Good range: [{format_value(t['good_min'])}, {format_value(t['good_max'])}]"
            )
            print(
                f"   Bad range:  [{format_value(t['bad_min'])}, {format_value(t['bad_max'])}]"
            )
            if t["good_max"] < t["bad_min"]:
                print(
                    f"   → Good runs use LOWER values (< {format_value(t['bad_min'])})"
                )
            else:
                print(
                    f"   → Good runs use HIGHER values (> {format_value(t['bad_max'])})"
                )
            print()


if __name__ == "__main__":
    main()
