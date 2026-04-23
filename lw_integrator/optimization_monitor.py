"""Packaged CLI for monitoring optimization progress from logcache files."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from optimization.log_monitor import (
    MONITORED_INSIGHT_PARAMETERS,
    analyze_optimization_logs,
    collect_varied_parameters,
    summarize_parameter_ranges,
)


class OptimizationMonitor:
    """Monitor and display optimization progress in real time."""

    def __init__(
        self,
        logcache_dir: Path,
        *,
        top_n: int = 5,
        latest_only: bool = False,
        specific_run: str | None = None,
    ) -> None:
        self.logcache_dir = Path(logcache_dir)
        self.top_n = top_n
        self.latest_only = latest_only
        self.specific_run = specific_run
        self.previous_best: float | None = None
        self.previous_eval_count = 0
        self.start_time = datetime.now()
        self.current_run_file: str | None = None

    def analyze_logs(self) -> tuple[list[Dict[str, Any]], list[Path]]:
        """Analyze optimization logs and reset counters when latest run flips."""
        all_results, log_files = analyze_optimization_logs(
            self.logcache_dir,
            latest_only=self.latest_only,
            specific_run=self.specific_run,
        )
        if self.specific_run and not log_files:
            print(f"Warning: No log file found matching '{self.specific_run}'")
            return [], []

        if self.latest_only and log_files:
            latest_file = log_files[0]
            if self.current_run_file != latest_file.name:
                self.current_run_file = latest_file.name
                self.previous_best = None
                self.previous_eval_count = 0

        return all_results, log_files

    @staticmethod
    def format_value(value: Any, width: int = 12) -> str:
        """Format parameter values for aligned monitor output."""
        if isinstance(value, float):
            if abs(value) < 1e-3 or abs(value) > 1e6:
                return f"{value:.4e}".rjust(width)
            return f"{value:.6f}".rjust(width)
        return str(value).rjust(width)

    def display_summary(
        self,
        all_results: list[Dict[str, Any]],
        log_files: list[Path],
        *,
        compact: bool = False,
    ) -> None:
        """Display optimization summary output."""
        if not all_results:
            print("No optimization results found yet.")
            return

        if not compact:
            print("\033[2J\033[H")
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
                    f"Monitoring: {len(log_files)} log files | "
                    f"{len(all_results)} total evaluations"
                )
            runtime = (datetime.now() - self.start_time).total_seconds()
            print(f"Runtime: {runtime:.0f}s")
            print("=" * 100)

            varied_params = collect_varied_parameters(all_results)
            if varied_params:
                print()
                print(f"📊 Parameters Being Varied ({len(varied_params)}):")
                print("  " + ", ".join(varied_params))
                print()

            print()

        positive_gains = [
            result["energy_gain"]
            for result in all_results
            if result.get("energy_gain", 0) > 0
        ]
        negative_gains = [
            result["energy_gain"]
            for result in all_results
            if result.get("energy_gain", 0) <= 0
        ]
        best_gain = max(positive_gains) if positive_gains else 0.0
        avg_positive = (
            sum(positive_gains) / len(positive_gains) if positive_gains else 0.0
        )

        new_best = self.previous_best is None or best_gain > self.previous_best
        self.previous_best = best_gain

        new_evals = len(all_results) - self.previous_eval_count
        self.previous_eval_count = len(all_results)

        if compact:
            status = "🆕 NEW BEST!" if new_best else "✓"
            run_info = ""
            if self.latest_only and log_files:
                run_info = f"[{log_files[0].name[:20]}] "
            positive_pct = 100 * len(positive_gains) / len(all_results)
            print(
                f"{status} [{datetime.now().strftime('%H:%M:%S')}] {run_info}"
                f"Evals: {len(all_results):4d} (+{new_evals:3d}) | "
                f"Best: {best_gain:.6e}% | "
                f"Positive: {len(positive_gains):4d} ({positive_pct:.1f}%)"
            )
        else:
            if new_best:
                print("🆕 " + "=" * 30 + " NEW BEST RESULT! " + "=" * 30 + " 🆕")
                print()

            print("📊 OVERALL STATISTICS")
            print("-" * 100)
            positive_pct = 100 * len(positive_gains) / len(all_results)
            negative_pct = 100 * len(negative_gains) / len(all_results)
            print(
                f"  Total Evaluations:     {len(all_results):6d}  (New: +{new_evals})"
            )
            print(
                f"  Positive Gains:        {len(positive_gains):6d}  ({positive_pct:5.1f}%)"
            )
            print(
                f"  Negative/Zero Gains:   {len(negative_gains):6d}  ({negative_pct:5.1f}%)"
            )
            print()
            print(f"  🏆 Best Gain:          {best_gain:12.6e}%")
            print(f"  📈 Avg Positive Gain:  {avg_positive:12.6e}%")
            print()

        if not compact:
            print(f"🏆 TOP {self.top_n} PARAMETER COMBINATIONS")
            print("=" * 100)

        for index, result in enumerate(all_results[: self.top_n], start=1):
            gain = result.get("energy_gain", 0)
            params = result.get("params", {})

            if compact:
                energy = params.get("initial_energy_gev", "N/A")
                rider_p = params.get("transverse_momentum", "N/A")
                rider_d = params.get("rider_transv_dist", "N/A")
                print(
                    f"  #{index}: {gain:.6e}% | "
                    f"E={energy:6.2f} GeV, "
                    f"r_p={rider_p:.4e}, r_d={rider_d:.5f}"
                )
                continue

            medal = ["🥇", "🥈", "🥉"][index - 1] if index <= 3 else f"#{index}"
            print(f"\n{medal} Rank {index}: Energy Gain = {gain:.6e}%")
            print(f"  Source: {result['log_file']}")
            print(f"  Evaluation: {result['eval_num']}")
            if not params:
                continue

            key_params = {}
            rider_params = {}
            driver_params = {}
            geometry_params = {}
            for key, value in params.items():
                if key in {"aperture_radius", "wall_z", "cavity_spacing"}:
                    geometry_params[key] = value
                elif key.startswith("driver_"):
                    driver_params[key] = value
                elif key in {
                    "initial_energy_gev",
                    "driver_energy_gev",
                    "transverse_momentum",
                    "rider_transv_dist",
                }:
                    key_params[key] = value
                else:
                    rider_params[key] = value

            if key_params:
                print("  Key Parameters:")
                for key in ("initial_energy_gev", "driver_energy_gev"):
                    if key in key_params:
                        label = (
                            "Rider Energy (GeV)"
                            if key == "initial_energy_gev"
                            else "Driver Energy (GeV)"
                        )
                        print(
                            f"    {label:25s} = {self.format_value(key_params[key])}"
                        )
                for key in sorted(key_params):
                    if key not in {"initial_energy_gev", "driver_energy_gev"}:
                        print(
                            f"    {key:25s} = {self.format_value(key_params[key])}"
                        )

            if geometry_params:
                print("  Geometry:")
                for key in sorted(geometry_params):
                    print(
                        f"    {key:25s} = {self.format_value(geometry_params[key])}"
                    )

            if rider_params:
                print("  Rider (Other):")
                for key in sorted(rider_params):
                    print(f"    {key:25s} = {self.format_value(rider_params[key])}")

            if driver_params:
                print("  Driver:")
                for key in sorted(driver_params):
                    print(f"    {key:25s} = {self.format_value(driver_params[key])}")

        if not compact:
            print()
            print("=" * 100)
            insight_count = max(1, len(all_results) // 10)
            print(f"💡 KEY INSIGHTS (Top {insight_count} performers)")
            print("-" * 100)
            summaries = summarize_parameter_ranges(
                all_results,
                parameter_names=MONITORED_INSIGHT_PARAMETERS,
            )
            for parameter in MONITORED_INSIGHT_PARAMETERS:
                if parameter not in summaries:
                    continue
                summary = summaries[parameter]
                print(
                    f"  {parameter:25s}: Avg={self.format_value(summary['average'])} "
                    f"Range=[{self.format_value(summary['min'], 10)}, "
                    f"{self.format_value(summary['max'], 10)}]"
                )
            print("=" * 100)

    def run_once(self, *, compact: bool = False) -> list[Dict[str, Any]]:
        """Run one monitor refresh and return parsed results."""
        all_results, log_files = self.analyze_logs()
        self.display_summary(all_results, log_files, compact=compact)
        return all_results

    def run_continuous(self, *, interval: int = 60, compact: bool = False) -> int:
        """Run the live monitor until interrupted."""
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
            if self.previous_best is not None:
                print(f"Best energy gain: {self.previous_best:.6e}%")
        return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse optimization monitor command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor optimization progress in real time"
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
        type=Path,
        help="Path to logcache directory (default: ./logcache)",
    )
    return parser.parse_args(argv)


def resolve_logcache_dir(logcache: Optional[Path]) -> Path:
    """Resolve the logcache directory used by the packaged monitor."""
    if logcache is not None:
        return logcache
    return Path(__file__).resolve().parent.parent / "logcache"


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the packaged optimization monitor CLI."""
    args = parse_args(argv)
    logcache_dir = resolve_logcache_dir(args.logcache)
    if not logcache_dir.exists():
        print(f"Error: logcache directory not found at {logcache_dir}")
        print("Please specify the correct path with --logcache")
        return 1

    monitor = OptimizationMonitor(
        logcache_dir,
        top_n=args.top,
        latest_only=args.latest,
        specific_run=args.run,
    )
    if args.once:
        monitor.run_once(compact=args.compact)
        return 0
    return monitor.run_continuous(interval=args.interval, compact=args.compact)
