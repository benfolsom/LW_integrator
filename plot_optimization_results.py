"""Compatibility wrapper for packaged optimization result reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

def load_results(json_path: Path) -> Any:
    """Load results from a JSON file."""
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Visualize optimization results")
    parser.add_argument("json_path", type=Path, help="Path to results JSON file")
    parser.add_argument("--output", "-o", type=Path, help="Output path for heatmap")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point for the compatibility wrapper."""
    args = parse_args(argv)
    if not args.json_path.exists():
        print(f"Error: {args.json_path} not found")
        return 1

    results = load_results(args.json_path)

    if not isinstance(results, dict):
        print("Error: Legacy list-based optimization plots are no longer supported")
        return 1

    from lw_integrator.optimization_results import main as packaged_main

    forwarded = [str(args.json_path)]
    if args.output is not None:
        forwarded.extend(["--output", str(args.output)])
    return packaged_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
