"""Benchmark the exact retarded charge-field workload used by RFS.

The benchmark deliberately exercises public field evaluators instead of an
isolated arithmetic kernel.  The ``field`` workload includes history
extraction, light-cone solution, quintic interpolation, and Lienard--Wiechert
field construction.  The ``gradient`` workload also evaluates the eight
displaced events used by the centered spacetime derivative.

Run from the repository root, for example::

    .venv/bin/python scripts/benchmark_rfs_retarded_fields.py \
        --events 16 --repeats 5 --output /tmp/rfs-retarded-fields.json

Use ``--compare-to`` with an earlier JSON report to check root, field, and
gradient parity after an implementation change.  Timing comparisons should be
made on an otherwise idle machine with the same Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core import retarded_fields
from core.constants import C_MMNS, ELEMENTARY_CHARGE

FieldEvaluator = Callable[..., Any]


def _public_evaluators() -> tuple[FieldEvaluator, FieldEvaluator, str]:
    """Return native-Gaussian evaluators when present, otherwise SI ones."""

    native_field = getattr(
        retarded_fields, "evaluate_retarded_charge_field_native", None
    )
    native_gradient = getattr(
        retarded_fields, "evaluate_retarded_charge_field_gradient_native", None
    )
    if native_field is not None and native_gradient is not None:
        return native_field, native_gradient, "native_gaussian"
    return (
        retarded_fields.evaluate_retarded_charge_field_si,
        retarded_fields.evaluate_retarded_charge_field_gradient_si,
        "si",
    )


def _source_history(
    *, history_steps: int, source_count: int
) -> list[dict[str, np.ndarray]]:
    """Construct smooth, accelerated, subluminal source worldlines."""

    if history_steps < 4:
        raise ValueError("history_steps must be at least four")
    if source_count < 1:
        raise ValueError("source_count must be positive")

    times_ns = np.linspace(-0.04, 0.004, history_steps)
    angular_frequency_ns = 2.0 * np.pi / 0.08
    states: list[dict[str, np.ndarray]] = []
    positions = np.empty((history_steps, source_count, 3), dtype=float)
    betas = np.empty_like(positions)
    beta_dots_s = np.empty_like(positions)

    for source_index in range(source_count):
        phase = 0.37 * source_index
        base_beta = np.asarray(
            (
                0.055 + 0.004 * source_index,
                -0.025 + 0.002 * source_index,
                0.012 - 0.001 * source_index,
            )
        )
        amplitude = np.asarray((0.006, 0.004, 0.003))
        component_phase = np.asarray((phase, phase + 0.7, phase - 0.4))
        angles = (
            angular_frequency_ns * times_ns[:, np.newaxis]
            + component_phase[np.newaxis, :]
        )
        betas[:, source_index] = base_beta + amplitude * np.sin(angles)
        beta_dots_s[:, source_index] = (
            amplitude * angular_frequency_ns * 1.0e9 * np.cos(angles)
        )
        oscillatory_integral_ns = (
            -amplitude / angular_frequency_ns * np.cos(angles)
            + amplitude / angular_frequency_ns * np.cos(component_phase)[np.newaxis, :]
        )
        source_offset_mm = np.asarray(
            (0.12 * source_index, 0.18 * source_index, -0.07 * source_index)
        )
        positions[:, source_index] = source_offset_mm + C_MMNS * (
            times_ns[:, np.newaxis] * base_beta + oscillatory_integral_ns
        )

    if float(np.max(np.linalg.norm(betas, axis=2))) >= 1.0:
        raise AssertionError("benchmark source worldline is not timelike")

    charges = ELEMENTARY_CHARGE * np.asarray(
        [
            (-1.0 if index % 2 else 1.0) * (1.0 + 0.1 * index)
            for index in range(source_count)
        ]
    )
    bdot_native = beta_dots_s / (C_MMNS * 1.0e9)
    for step in range(history_steps):
        states.append(
            {
                "t": np.full(source_count, times_ns[step], dtype=float),
                "x": positions[step, :, 0].copy(),
                "y": positions[step, :, 1].copy(),
                "z": positions[step, :, 2].copy(),
                "bx": betas[step, :, 0].copy(),
                "by": betas[step, :, 1].copy(),
                "bz": betas[step, :, 2].copy(),
                "bdotx": bdot_native[step, :, 0].copy(),
                "bdoty": bdot_native[step, :, 1].copy(),
                "bdotz": bdot_native[step, :, 2].copy(),
                "q": charges.copy(),
                "q_source": charges.copy(),
                "_dead_particles": np.zeros(source_count, dtype=bool),
            }
        )
    return states


def _observer_events(event_count: int) -> list[retarded_fields.ObserverEvent]:
    if event_count < 1:
        raise ValueError("events must be positive")
    progress = np.linspace(0.0, 1.0, event_count)
    return [
        retarded_fields.ObserverEvent(
            time_ns=float(0.0008 * value),
            position_mm=(
                float(2.0 + 0.08 * value),
                float(0.95 - 0.05 * value),
                float(-0.45 + 0.03 * value),
            ),
        )
        for value in progress
    ]


def _maximum_resident_mebibytes() -> float:
    maximum_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        maximum_rss *= 1024.0
    return maximum_rss / 1024.0**2


def _sysctl_value(name: str) -> str | None:
    if sys.platform != "darwin":
        return None
    completed = subprocess.run(
        ("sysctl", "-n", name),
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _hardware_metadata() -> dict[str, Any]:
    physical_memory = _sysctl_value("hw.memsize")
    logical_cpu_count = _sysctl_value("hw.logicalcpu")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_brand": _sysctl_value("machdep.cpu.brand_string"),
        "physical_memory_bytes": (
            int(physical_memory) if physical_memory is not None else None
        ),
        "logical_cpu_count": (
            int(logical_cpu_count) if logical_cpu_count is not None else os.cpu_count()
        ),
    }


def _sha256_arrays(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _length_metres(result: Any, stem: str) -> np.ndarray:
    metres_name = f"{stem}_m"
    millimetres_name = f"{stem}_mm"
    if hasattr(result, metres_name):
        return np.asarray(getattr(result, metres_name), dtype=float)
    if hasattr(result, millimetres_name):
        return np.asarray(getattr(result, millimetres_name), dtype=float) * 1.0e-3
    raise AttributeError(
        f"retarded-field result provides neither {metres_name} nor "
        f"{millimetres_name}"
    )


def _capture_field_payload(results: Sequence[Any]) -> dict[str, Any]:
    tensors = np.stack([np.asarray(result.field_tensor) for result in results])
    times = np.stack([np.asarray(result.retarded_time_ns) for result in results])
    residuals = np.stack(
        [_length_metres(result, "light_cone_residual") for result in results]
    )
    separations = np.stack([_length_metres(result, "separation") for result in results])
    valid = np.stack([np.asarray(result.valid_sources) for result in results])
    checksum_arrays = (tensors, times, residuals, separations, valid.astype(float))
    return {
        "field_tensor": tensors.tolist(),
        "retarded_time_ns": times.tolist(),
        "light_cone_residual_m": residuals.tolist(),
        "separation_m": separations.tolist(),
        "valid_sources": valid.tolist(),
        "sha256": _sha256_arrays(checksum_arrays),
        "maximum_absolute_light_cone_residual_m": float(np.nanmax(np.abs(residuals))),
    }


def _capture_gradient_payload(results: Sequence[Any]) -> dict[str, Any]:
    center_payload = _capture_field_payload([result.field for result in results])
    partial_f = np.stack([np.asarray(result.partial_f) for result in results])
    stencil_times = np.stack(
        [np.asarray(result.stencil_retarded_time_ns) for result in results]
    )
    steps = np.asarray(
        [float(_length_metres(result, "stencil_step")) for result in results]
    )
    return {
        "center": center_payload,
        "partial_f": partial_f.tolist(),
        "stencil_retarded_time_ns": stencil_times.tolist(),
        "stencil_step_m": steps.tolist(),
        "sha256": _sha256_arrays((partial_f, stencil_times, steps)),
    }


def _time_workload(
    operation: Callable[[], Sequence[Any]], *, warmups: int, repeats: int
) -> tuple[list[Any], dict[str, float]]:
    for _ in range(warmups):
        operation()
    samples = []
    final_results: Sequence[Any] = ()
    for _ in range(repeats):
        started = time.perf_counter()
        final_results = operation()
        samples.append(time.perf_counter() - started)
    return list(final_results), {
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "maximum_seconds": max(samples),
    }


def _difference_metrics(current: Any, reference: Any) -> dict[str, float]:
    left = np.asarray(current, dtype=float)
    right = np.asarray(reference, dtype=float)
    if left.shape != right.shape:
        raise ValueError(
            f"parity payload shape changed from {right.shape} to {left.shape}"
        )
    difference = np.abs(left - right)
    scale = np.maximum(np.abs(right), np.finfo(float).tiny)
    return {
        "maximum_absolute": float(np.nanmax(difference)),
        "maximum_relative": float(np.nanmax(difference / scale)),
    }


def compare_reports(
    current: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Return numerical differences between two benchmark reports."""

    current_payload = current["parity"]
    reference_payload = reference["parity"]
    comparison: dict[str, Any] = {
        "field_retarded_time_ns": _difference_metrics(
            current_payload["field"]["retarded_time_ns"],
            reference_payload["field"]["retarded_time_ns"],
        ),
        "field_light_cone_residual_m": _difference_metrics(
            current_payload["field"]["light_cone_residual_m"],
            reference_payload["field"]["light_cone_residual_m"],
        ),
        "field_separation_m": _difference_metrics(
            current_payload["field"]["separation_m"],
            reference_payload["field"]["separation_m"],
        ),
        "gradient_stencil_retarded_time_ns": _difference_metrics(
            current_payload["gradient"]["stencil_retarded_time_ns"],
            reference_payload["gradient"]["stencil_retarded_time_ns"],
        ),
        "gradient_stencil_step_m": _difference_metrics(
            current_payload["gradient"]["stencil_step_m"],
            reference_payload["gradient"]["stencil_step_m"],
        ),
    }
    if current["unit_system"] == reference["unit_system"]:
        comparison.update(
            {
                "field_tensor": _difference_metrics(
                    current_payload["field"]["field_tensor"],
                    reference_payload["field"]["field_tensor"],
                ),
                "gradient_partial_f": _difference_metrics(
                    current_payload["gradient"]["partial_f"],
                    reference_payload["gradient"]["partial_f"],
                ),
            }
        )
    else:
        comparison["raw_tensor_comparison"] = {
            "skipped": True,
            "reason": (
                "raw field tensors and derivatives use different unit systems; "
                "use normalized light-cone and integration-trajectory parity"
            ),
        }
    return comparison


def _valid_field_root_count(results: Sequence[Any]) -> int:
    return sum(int(np.count_nonzero(result.valid_sources)) for result in results)


def _valid_gradient_root_count(results: Sequence[Any]) -> int:
    center_roots = sum(
        int(np.count_nonzero(result.field.valid_sources)) for result in results
    )
    stencil_roots = sum(
        int(np.count_nonzero(np.isfinite(result.stencil_retarded_time_ns)))
        for result in results
    )
    return center_roots + stencil_roots


def _count_interpolated_samples(
    operation: Callable[[], Sequence[Any]],
    root_counter: Callable[[Sequence[Any]], int],
) -> dict[str, float | int] | None:
    sample_function = getattr(retarded_fields, "_quintic_worldline_sample", None)
    if sample_function is None:
        return None
    sample_count = 0

    def counted_sample(*args: Any, **kwargs: Any) -> Any:
        nonlocal sample_count
        sample_count += 1
        return sample_function(*args, **kwargs)

    setattr(retarded_fields, "_quintic_worldline_sample", counted_sample)
    try:
        results = operation()
    finally:
        setattr(retarded_fields, "_quintic_worldline_sample", sample_function)
    root_count = root_counter(results)
    if root_count == 0:
        mean_samples = 0.0
        mean_iterations = 0.0
    else:
        mean_samples = sample_count / root_count
        mean_iterations = (sample_count - root_count) / root_count
    return {
        "retarded_roots": root_count,
        "interpolated_samples": sample_count,
        "mean_samples_per_root": mean_samples,
        "mean_iterations_per_root": mean_iterations,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    field_evaluator, gradient_evaluator, unit_system = _public_evaluators()
    history = _source_history(
        history_steps=int(args.history_steps), source_count=int(args.sources)
    )
    events = _observer_events(int(args.events))

    def field_workload() -> list[Any]:
        return [field_evaluator(history, event) for event in events]

    def gradient_workload() -> list[Any]:
        return [
            gradient_evaluator(history, event, relative_step=args.relative_step)
            for event in events
        ]

    field_results, field_timing = _time_workload(
        field_workload, warmups=args.warmups, repeats=args.repeats
    )
    gradient_results, gradient_timing = _time_workload(
        gradient_workload, warmups=args.warmups, repeats=args.repeats
    )
    sample_counts = {
        "field": _count_interpolated_samples(field_workload, _valid_field_root_count),
        "gradient": _count_interpolated_samples(
            gradient_workload, _valid_gradient_root_count
        ),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "unit_system": unit_system,
        "parameters": {
            "history_steps": int(args.history_steps),
            "sources": int(args.sources),
            "events": int(args.events),
            "relative_step": float(args.relative_step),
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
        },
        "hardware": _hardware_metadata(),
        "timing": {"field": field_timing, "gradient": gradient_timing},
        "root_solver": sample_counts,
        "maximum_resident_memory_mib": _maximum_resident_mebibytes(),
        "parity": {
            "field": _capture_field_payload(field_results),
            "gradient": _capture_gradient_payload(gradient_results),
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
    parser.add_argument("--history-steps", type=_positive_integer, default=257)
    parser.add_argument("--sources", type=_positive_integer, default=2)
    parser.add_argument("--events", type=_positive_integer, default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=_positive_integer, default=5)
    parser.add_argument("--relative-step", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args(argv)
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if not math.isfinite(args.relative_step) or not 0.0 < args.relative_step < 0.1:
        parser.error("--relative-step must be finite and in (0, 0.1)")
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
