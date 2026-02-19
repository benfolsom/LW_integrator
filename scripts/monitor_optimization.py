#!/usr/bin/env python3
"""
Live Optimization Monitor

Continuously monitors optimization logs in logcache/ and displays
real-time updates on the best parameter combinations found.

Usage:
    python3 local/monitor_optimization.py [--interval SECONDS] [--top N]

Options:
    --interval, -i    Update interval in seconds (default: 60)
    --top, -n         Number of top results to display (default: 5)
    --once            Run once and exit (no continuous monitoring)
    --compact         Compact output format
    --latest          Monitor only the latest/current run (default: all runs)
    --run FILE        Monitor specific run by log filename
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class OptimizationMonitor:
    """Monitor and analyze optimization logs in real-time."""

    def __init__(self, logcache_dir, top_n=5, latest_only=False, specific_run=None):
        self.logcache_dir = Path(logcache_dir)
        self.top_n = top_n
        self.latest_only = latest_only
        self.specific_run = specific_run
        self.previous_best = None
        self.previous_eval_count = 0
        self.start_time = datetime.now()
        self.current_run_file = None

    def parse_optimization_log(self, log_path):
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
                        for param_match in re.finditer(
                            r"(\w+)=([\d.e+-]+)", params_str
                        ):
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
                            "timestamp": log_path.stat().st_mtime,
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
            pass  # Silently skip problematic files

        return results

    def analyze_logs(self):
        """Analyze optimization logs in directory."""
        all_results = []
        log_files = sorted(self.logcache_dir.glob("*optimization*.log"))

        # Filter log files based on mode
        if self.specific_run:
            # Monitor specific run by filename
            log_files = [f for f in log_files if self.specific_run in f.name]
            if not log_files:
                print(f"Warning: No log file found matching '{self.specific_run}'")
                return [], []
        elif self.latest_only:
            # Monitor only the most recent run
            if log_files:
                latest_file = max(log_files, key=lambda f: f.stat().st_mtime)
                log_files = [latest_file]
                # Track current run file
                if self.current_run_file != latest_file.name:
                    self.current_run_file = latest_file.name
                    self.previous_best = None  # Reset best when switching runs
                    self.previous_eval_count = 0

        for log_file in log_files:
            results = self.parse_optimization_log(log_file)
            if results:
                all_results.extend(results)

        # Sort by energy gain (descending)
        all_results.sort(
            key=lambda x: x.get("energy_gain", float("-inf")), reverse=True
        )

        return all_results, log_files

    def format_value(self, value, width=12):
        """Format parameter value for display."""
        if isinstance(value, float):
            if abs(value) < 1e-3 or abs(value) > 1e6:
                return f"{value:.4e}".rjust(width)
            else:
                return f"{value:.6f}".rjust(width)
        return str(value).rjust(width)

    def display_summary(self, all_results, log_files, compact=False):
        """Display optimization summary."""
        if not all_results:
            print("No optimization results found yet.")
            return

        # Clear screen and show header
        if not compact:
            print("\033[2J\033[H")  # Clear screen
            print("=" * 100)
            if self.latest_only:
                print("🔬 LIVE OPTIMIZATION MONITOR - LATEST RUN ONLY")
            elif self.specific_run:
                print(f"🔬 LIVE OPTIMIZATION MONITOR - {self.specific_run}")
            else:
                print("🔬 LIVE OPTIMIZATION MONITOR - ALL RUNS")
            print("=" * 100)
            print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            if self.latest_only and log_files:
                print(f"Current Run: {log_files[0].name}")
                print(f"Evaluations in this run: {len(all_results)}")
            elif self.specific_run:
                print(f"Monitoring: {self.specific_run}")
                print(f"Evaluations: {len(all_results)}")
            else:
                print(
                    f"Monitoring: {len(log_files)} log files | {len(all_results)} total evaluations"
                )
            print(f"Runtime: {(datetime.now() - self.start_time).total_seconds():.0f}s")
            print("=" * 100)

            # Show which parameters are being varied
            if all_results:
                all_param_keys = set()
                for result in all_results:
                    all_param_keys.update(result["params"].keys())

                # Determine which parameters have variation
                varied_params = []
                for param in sorted(all_param_keys):
                    values = [
                        r["params"].get(param)
                        for r in all_results
                        if param in r["params"]
                    ]
                    values = [v for v in values if isinstance(v, (int, float))]
                    if len(values) > 1 and min(values) != max(values):
                        varied_params.append(param)

                if varied_params:
                    print()
                    print(f"📊 Parameters Being Varied ({len(varied_params)}):")
                    print("  " + ", ".join(varied_params))
                    print()

            print()

        # Statistics
        positive_gains = [
            r["energy_gain"] for r in all_results if r.get("energy_gain", 0) > 0
        ]
        negative_gains = [
            r["energy_gain"] for r in all_results if r.get("energy_gain", 0) <= 0
        ]

        best_gain = max(positive_gains) if positive_gains else 0.0
        avg_positive = (
            sum(positive_gains) / len(positive_gains) if positive_gains else 0.0
        )

        # Check for new best
        new_best = False
        if self.previous_best is None or best_gain > self.previous_best:
            new_best = True
            self.previous_best = best_gain

        # Check for new evaluations
        new_evals = len(all_results) - self.previous_eval_count
        self.previous_eval_count = len(all_results)

        if compact:
            # Compact one-line summary
            status = "🆕 NEW BEST!" if new_best else "✓"
            run_info = ""
            if self.latest_only and log_files:
                run_info = f"[{log_files[0].name[:20]}] "
            print(
                f"{status} [{datetime.now().strftime('%H:%M:%S')}] {run_info}"
                f"Evals: {len(all_results):4d} (+{new_evals:3d}) | "
                f"Best: {best_gain:.6e}% | "
                f"Positive: {len(positive_gains):4d} ({100 * len(positive_gains) / len(all_results):.1f}%)"
            )
        else:
            # Full summary
            if new_best:
                print("🆕 " + "=" * 30 + " NEW BEST RESULT! " + "=" * 30 + " 🆕")
                print()

            print("📊 OVERALL STATISTICS")
            print("-" * 100)
            print(
                f"  Total Evaluations:     {len(all_results):6d}  (New: +{new_evals})"
            )
            print(
                f"  Positive Gains:        {len(positive_gains):6d}  ({100 * len(positive_gains) / len(all_results):5.1f}%)"
            )
            print(
                f"  Negative/Zero Gains:   {len(negative_gains):6d}  ({100 * len(negative_gains) / len(all_results):5.1f}%)"
            )
            print()
            print(f"  🏆 Best Gain:          {best_gain:12.6e}%")
            print(f"  📈 Avg Positive Gain:  {avg_positive:12.6e}%")
            print()

        # Top results
        if not compact:
            print(f"🏆 TOP {self.top_n} PARAMETER COMBINATIONS")
            print("=" * 100)

        for i, result in enumerate(all_results[: self.top_n], 1):
            gain = result.get("energy_gain", 0)

            if compact:
                params = result["params"]
                energy = params.get("initial_energy_gev", "N/A")
                rider_p = params.get("transverse_momentum", "N/A")
                rider_d = params.get("rider_transv_dist", "N/A")
                print(
                    f"  #{i}: {gain:.6e}% | "
                    f"E={energy:6.2f} GeV, "
                    f"r_p={rider_p:.4e}, r_d={rider_d:.5f}"
                )
            else:
                medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"#{i}"
                print(f"\n{medal} Rank {i}: Energy Gain = {gain:.6e}%")
                print(f"  Source: {result['log_file']}")
                print(f"  Evaluation: {result['eval_num']}")

                params = result["params"]
                if params:
                    # Categorize parameters more intelligently
                    # Key physics parameters first
                    key_params = {}
                    rider_params = {}
                    driver_params = {}
                    geometry_params = {}

                    for k, v in params.items():
                        if k in ["aperture_radius", "wall_z", "cavity_spacing"]:
                            geometry_params[k] = v
                        elif k.startswith("driver_"):
                            driver_params[k] = v
                        elif k in [
                            "initial_energy_gev",
                            "driver_energy_gev",
                            "transverse_momentum",
                            "rider_transv_dist",
                        ]:
                            key_params[k] = v
                        else:
                            rider_params[k] = v

                    # Display in logical order
                    if key_params:
                        print("  Key Parameters:")
                        # Show energies first
                        for key in ["initial_energy_gev", "driver_energy_gev"]:
                            if key in key_params:
                                value = key_params[key]
                                formatted_val = self.format_value(value)
                                label = (
                                    "Rider Energy (GeV)"
                                    if key == "initial_energy_gev"
                                    else "Driver Energy (GeV)"
                                )
                                print(f"    {label:25s} = {formatted_val}")
                        # Then other key params
                        for key in sorted(key_params.keys()):
                            if key not in ["initial_energy_gev", "driver_energy_gev"]:
                                value = key_params[key]
                                formatted_val = self.format_value(value)
                                print(f"    {key:25s} = {formatted_val}")

                    if geometry_params:
                        print("  Geometry:")
                        for key in sorted(geometry_params.keys()):
                            value = geometry_params[key]
                            formatted_val = self.format_value(value)
                            print(f"    {key:25s} = {formatted_val}")

                    if rider_params:
                        print("  Rider (Other):")
                        for key in sorted(rider_params.keys()):
                            value = rider_params[key]
                            formatted_val = self.format_value(value)
                            print(f"    {key:25s} = {formatted_val}")

                    if driver_params:
                        print("  Driver:")
                        for key in sorted(driver_params.keys()):
                            value = driver_params[key]
                            formatted_val = self.format_value(value)
                            print(f"    {key:25s} = {formatted_val}")

        if not compact:
            print()
            print("=" * 100)

            # Key parameter insights
            top_10_percent = max(1, len(all_results) // 10)
            top_performers = all_results[:top_10_percent]

            param_values = defaultdict(list)
            for result in top_performers:
                for key, value in result["params"].items():
                    if isinstance(value, (int, float)):
                        param_values[key].append(value)

            print(f"💡 KEY INSIGHTS (Top {top_10_percent} performers)")
            print("-" * 100)

            key_params = [
                "initial_energy_gev",
                "transverse_momentum",
                "rider_transv_dist",
                "driver_stripped_ions",
                "driver_transv_mom",
                "driver_transv_dist",
            ]

            for param in key_params:
                if param in param_values and param_values[param]:
                    values = param_values[param]
                    avg = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)

                    print(
                        f"  {param:25s}: Avg={self.format_value(avg)} "
                        f"Range=[{self.format_value(min_val, 10)}, {self.format_value(max_val, 10)}]"
                    )

            print("=" * 100)

    def run_once(self, compact=False):
        """Run analysis once."""
        all_results, log_files = self.analyze_logs()
        self.display_summary(all_results, log_files, compact=compact)
        return all_results

    def run_continuous(self, interval=60, compact=False):
        """Run continuous monitoring with periodic updates."""
        print(f"Starting optimization monitor (update every {interval}s)")
        print("Press Ctrl+C to stop")
        print()

        try:
            while True:
                all_results, log_files = self.analyze_logs()
                self.display_summary(all_results, log_files, compact=compact)

                if not compact:
                    print(f"\n⏳ Next update in {interval} seconds... (Ctrl+C to stop)")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped by user")
            print(f"Final statistics: {len(all_results)} evaluations analyzed")
            if self.previous_best:
                print(f"Best energy gain: {self.previous_best:.6e}%")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor optimization progress in real-time"
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=60,
        help="Update interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=5,
        help="Number of top results to display (default: 5)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (no continuous monitoring)",
    )
    parser.add_argument(
        "--compact",
        "-c",
        action="store_true",
        help="Compact output format (one line per update)",
    )
    parser.add_argument(
        "--latest",
        "-l",
        action="store_true",
        help="Monitor only the latest/current run (not all historical runs)",
    )
    parser.add_argument(
        "--run",
        "-r",
        type=str,
        help="Monitor specific run by log filename (partial match)",
    )
    parser.add_argument(
        "--logcache",
        type=str,
        help="Path to logcache directory (default: ../logcache)",
    )

    args = parser.parse_args()

    # Determine logcache directory
    if args.logcache:
        logcache_dir = Path(args.logcache)
    else:
        # Default: look for logcache relative to script location
        script_dir = Path(__file__).parent
        logcache_dir = script_dir.parent / "logcache"

    if not logcache_dir.exists():
        print(f"Error: logcache directory not found at {logcache_dir}")
        print("Please specify the correct path with --logcache")
        sys.exit(1)

    monitor = OptimizationMonitor(
        logcache_dir, top_n=args.top, latest_only=args.latest, specific_run=args.run
    )

    if args.once:
        monitor.run_once(compact=args.compact)
    else:
        monitor.run_continuous(interval=args.interval, compact=args.compact)


if __name__ == "__main__":
    main()
