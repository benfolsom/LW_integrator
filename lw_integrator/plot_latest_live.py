"""Packaged entry point for live plotting of the latest sweep log."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from lw_integrator.logcache_plotter import (
    find_latest_log,
    main as plot_from_logcache_main,
)


def parse_args(argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse wrapper arguments and preserve unknown flags for the plotter."""
    parser = argparse.ArgumentParser(
        description="Launch live plotting for the latest sweep log"
    )
    parser.add_argument(
        "--logcache",
        type=Path,
        default=Path("logcache"),
        help="Directory to search for the latest sweep log (default: ./logcache)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path to forward to the live plotter",
    )
    return parser.parse_known_args(argv)


def resolve_latest_sweep_log(logcache_dir: Path) -> Path | None:
    """Resolve the most recently modified sweep log from a logcache directory."""
    latest = find_latest_log(str(logcache_dir))
    return Path(latest) if latest is not None else None


def main(argv: Optional[list[str]] = None) -> int:
    """Launch the latest-log live plotter."""
    args, forwarded = parse_args(argv)
    latest_log = resolve_latest_sweep_log(args.logcache)
    if latest_log is None:
        print(f"ERROR: No sweep log files found in {args.logcache}")
        print("Run a sweep first or specify a log file with lw-plot-from-logcache-live.")
        return 1

    plot_args = ["--live", str(latest_log), *forwarded]
    if args.output is not None:
        plot_args.extend(["--output", str(args.output)])

    print(f"Latest sweep log: {latest_log}")
    return plot_from_logcache_main(plot_args)
