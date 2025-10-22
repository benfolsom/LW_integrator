"""Example: driving the ``lw-simulate`` entry point programmatically.

The ``lw_integrator.cli.main`` function underpins both the ``lw-simulate``
console script and the ``python -m lw_integrator`` module entry point.  This
example shows how to call it directly from Python code, passing the desired
command-line arguments as a list.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lw_integrator.cli as lw_cli  # noqa: E402


def run_default_scenario() -> int:
    """Execute the default configuration (quiet mode to suppress stdout)."""

    return lw_cli.main(["--quiet"])


def run_custom_steps(output_path: Path) -> int:
    """Execute a short simulation and request a JSON summary file."""

    argv = ["--steps", "250", "--time-step", "5e-4", "--output", str(output_path)]
    return lw_cli.main(argv)


def main() -> int:
    print("Running lw-simulate with the default configuration...")
    default_status = run_default_scenario()
    print(f"Default run exit status: {default_status}\n")

    print("Running lw-simulate with custom steps and JSON output...")
    json_path = Path("test_outputs/entrypoint_demo/summary.json")
    custom_status = run_custom_steps(json_path)
    if json_path.exists():
        print(f"Summary written to {json_path.resolve()}")
    else:
        print("Summary file was not created.")
    return 0 if (default_status == 0 and custom_status == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
