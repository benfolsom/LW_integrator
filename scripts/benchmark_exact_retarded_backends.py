"""Canonical entry point for the exact-retarded backend benchmark.

The implementation remains importable from its former dipole-source filename
so existing research commands do not break during the alpha-stage config
migration. New commands and documentation should use this module.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.benchmark_dipole_source_backends import (
    _compare_trajectories,
    _trajectory_fingerprint,
    main,
    parse_args,
    run_benchmark,
)

__all__ = [
    "_compare_trajectories",
    "_trajectory_fingerprint",
    "main",
    "parse_args",
    "run_benchmark",
]


if __name__ == "__main__":
    raise SystemExit(main())
