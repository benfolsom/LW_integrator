"""Benchmark an end-to-end RFS integration without writing simulation outputs.

The input is an ordinary testbed/GUI JSON configuration.  Plotting and file
exports are disabled in memory, while all physics settings remain unchanged.
The report includes wall timings and complete numerical trajectory payloads
for the fields most relevant to RFS translation and spin transport.

For example::

    .venv/bin/python scripts/benchmark_rfs_integration.py \
        studies/.../capture_smoke_atomic_relaxed_ms1e4_rr_off_dipole_on.json \
        --warmups 1 --repeats 5 --output /tmp/rfs-integration.json

Use ``--compare-to`` to quantify elementwise trajectory differences from an
earlier report made with the same physical model and configuration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lw_integrator import testbed_runner
from core import trajectory_integrator
from scripts.benchmark_rfs_retarded_fields import (
    _hardware_metadata,
    _maximum_resident_mebibytes,
)

_TRAJECTORY_KEYS = (
    "t",
    "x",
    "y",
    "z",
    "Px",
    "Py",
    "Pz",
    "Pt",
    "gamma",
    "bx",
    "by",
    "bz",
    "bdotx",
    "bdoty",
    "bdotz",
    "radiation_power",
    "radiation_energy",
    "radiation_energy_applied",
    "spin_x",
    "spin_y",
    "spin_z",
    "local_magnetic_field_x_t",
    "local_magnetic_field_y_t",
    "local_magnetic_field_z_t",
)

_OUTPUT_FLAGS = (
    "energy_display",
    "energy_save",
    "transverse_display",
    "transverse_save",
    "beta_display",
    "beta_save",
    "momentum_display",
    "momentum_save",
    "gamma_display",
    "gamma_save",
    "zposition_display",
    "zposition_save",
    "trajectory_save",
    "save_log_file",
)


def _benchmark_options(config_path: Path) -> Any:
    options = testbed_runner.load_config(config_path)
    for name in _OUTPUT_FLAGS:
        if hasattr(options, name):
            setattr(options, name, False)
    if hasattr(options, "self_consistency_verbosity"):
        options.self_consistency_verbosity = 0
    if hasattr(options, "adaptive_timestep_debug"):
        options.adaptive_timestep_debug = False
    return options


def _run_once(options: Any) -> tuple[Any, tuple[Any, ...], float]:
    original_integrator = trajectory_integrator.retarded_integrator
    captured: dict[str, tuple[Any, ...]] = {}

    def capture_integrator(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        result = original_integrator(*args, **kwargs)
        captured["result"] = result
        return result

    trajectory_integrator.retarded_integrator = capture_integrator
    started = time.perf_counter()
    try:
        run_result = testbed_runner.run_testbed(
            copy.deepcopy(options),
            log=lambda _message: None,
        )
    finally:
        trajectory_integrator.retarded_integrator = original_integrator
    wall_seconds = time.perf_counter() - started
    if "result" not in captured:
        raise RuntimeError("testbed runner did not invoke the core integrator")
    return run_result, captured["result"], wall_seconds


def _trajectory_payload(trajectory: Sequence[dict[str, Any]]) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    digest = hashlib.sha256()
    for key in _TRAJECTORY_KEYS:
        if not trajectory or any(key not in state for state in trajectory):
            continue
        stacked = np.stack(
            [np.asarray(state[key], dtype=float) for state in trajectory],
            axis=0,
        )
        contiguous = np.ascontiguousarray(stacked, dtype=np.float64)
        arrays[key] = contiguous.tolist()
        digest.update(key.encode("utf-8"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return {
        "steps": len(trajectory),
        "selected_keys": list(arrays),
        "arrays": arrays,
        "selected_arrays_sha256": digest.hexdigest(),
    }


def _timing_summary(samples: Sequence[float]) -> dict[str, float]:
    return {
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "maximum_seconds": max(samples),
    }


def _difference_metrics(current: Any, reference: Any) -> dict[str, float]:
    left = np.asarray(current, dtype=float)
    right = np.asarray(reference, dtype=float)
    if left.shape != right.shape:
        raise ValueError(f"trajectory shape changed from {right.shape} to {left.shape}")
    difference = np.abs(left - right)
    scale = np.maximum(np.abs(right), np.finfo(float).tiny)
    maximum_absolute = float(np.max(difference))
    reference_scale = max(float(np.max(np.abs(right))), np.finfo(float).tiny)
    return {
        "maximum_absolute": maximum_absolute,
        "maximum_elementwise_relative": float(np.max(difference / scale)),
        "maximum_scale_relative": maximum_absolute / reference_scale,
    }


def compare_reports(
    current: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for role in ("rider", "driver"):
        current_role = current["trajectory"][role]
        reference_role = reference["trajectory"][role]
        current_arrays = current_role["arrays"]
        reference_arrays = reference_role["arrays"]
        if current_arrays.keys() != reference_arrays.keys():
            raise ValueError(f"{role} trajectory fields changed")
        field_metrics = {
            key: _difference_metrics(current_arrays[key], reference_arrays[key])
            for key in current_arrays
        }
        comparison[role] = {
            "selected_arrays_sha256_equal": _selected_arrays_sha256(current_role)
            == _selected_arrays_sha256(reference_role),
            "fields": field_metrics,
            "maximum_absolute": max(
                (metric["maximum_absolute"] for metric in field_metrics.values()),
                default=0.0,
            ),
            "maximum_scale_relative": max(
                (metric["maximum_scale_relative"] for metric in field_metrics.values()),
                default=0.0,
            ),
        }
    return comparison


def _selected_arrays_sha256(role_payload: dict[str, Any]) -> str:
    """Read the explicit hash name, accepting schema-v1 reports."""

    value = role_payload.get("selected_arrays_sha256", role_payload.get("sha256"))
    if not isinstance(value, str):
        raise ValueError("trajectory report does not contain a selected-array hash")
    return value


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    options = _benchmark_options(args.config)
    for _ in range(args.warmups):
        _run_once(options)

    wall_samples: list[float] = []
    reported_samples: list[float] = []
    final_integrator_result: tuple[Any, ...] | None = None
    final_run_result: Any = None
    for _ in range(args.repeats):
        run_result, integrator_result, wall_seconds = _run_once(options)
        final_run_result = run_result
        final_integrator_result = integrator_result
        wall_samples.append(wall_seconds)
        reported_samples.append(float(run_result.duration_s))
    if final_integrator_result is None:
        raise AssertionError("positive repeat count did not produce a result")

    rider_trajectory = final_integrator_result[0]
    driver_trajectory = final_integrator_result[1]
    report: dict[str, Any] = {
        "schema_version": 2,
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "parameters": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "steps": int(options.steps),
        },
        "hardware": _hardware_metadata(),
        "timing": {
            "wall": _timing_summary(wall_samples),
            "runner_reported": _timing_summary(reported_samples),
        },
        "maximum_resident_memory_mib": _maximum_resident_mebibytes(),
        "run_status": {
            "halted_early": bool(final_run_result.halted_early),
            "halt_reason": final_run_result.halt_reason,
            "dead_particles": int(final_run_result.num_particles_dead),
        },
        "trajectory": {
            "rider": _trajectory_payload(rider_trajectory),
            "driver": _trajectory_payload(driver_trajectory),
        },
    }
    if args.compare_to is not None:
        reference = json.loads(args.compare_to.read_text(encoding="utf-8"))
        report["comparison"] = compare_reports(report, reference)
    return report


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=_positive_integer, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args(argv)
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if not args.config.is_file():
        parser.error(f"configuration does not exist: {args.config}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
