"""Benchmark explicit Metal root proposals against the strict Numba CPU path.

This is a study-only orchestrator.  It reconstructs one real capture source
history and observer event, measures the maintained
``numba_roots_exact_serial`` kernel, and passes the same prepared float64
worldline to a standalone Swift/Metal benchmark.  Metal is never selected by
the integrator and this script refuses to run outside Darwin arm64.

The Metal kernel may propose float32 knot brackets and retarded times.  The
Swift harness certifies each bracket from its two original float64 endpoints,
uses a complete float64 binary-search fallback when certification fails, and
then executes the strict float64 root solve on the CPU.  Approximate GPU
values therefore never become accepted roots directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence
from unittest import mock

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.constants import C_MMNS  # noqa: E402
from core import retarded_dipole_fields as rdf  # noqa: E402
from core import retarded_fields as rf  # noqa: E402
from core.retarded_dipole_numba_roots import (  # noqa: E402
    NUMBA_AVAILABLE,
    evaluate_source_roots_exact_serial,
)

_BUNDLE_MAGIC = b"LWMTLR02"
_DEFAULT_COUNTS = (129, 258, 298, 512, 1024, 2048, 4096, 8192)
_ROOT_TOLERANCE_MM = 1.0e-21
_MAX_ROOT_ITERATIONS = 96
_RELATIVE_STENCIL_STEP = 1.0e-3


def _array(mapping: dict[str, Any], name: str) -> np.ndarray:
    return np.ascontiguousarray(mapping[name], dtype=np.float64)


def _load_source_and_event(
    report_path: Path,
    *,
    source_role: str,
    observer_role: str,
) -> tuple[
    rf._PreparedSourceHistory,
    rdf._PreparedDipoleHistory,
    rf.ObserverEvent,
]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source = report["core"][source_role]
    observer = report["core"][observer_role]
    positions = source["positions_mm"]
    betas = source["beta"]
    extra = source["additional_soa_step_arrays"]
    constants = source["particle_constants"]
    count = len(source["time_ns"])
    arrays = rf._HistoryArrays(
        time_ns=_array(source, "time_ns")[:, np.newaxis],
        position_mm=np.column_stack(
            (_array(positions, "x"), _array(positions, "y"), _array(positions, "z"))
        )[:, np.newaxis, :],
        beta=np.column_stack(
            (_array(betas, "bx"), _array(betas, "by"), _array(betas, "bz"))
        )[:, np.newaxis, :],
        beta_prime_per_mm=np.column_stack(
            (_array(extra, "bdotx"), _array(extra, "bdoty"), _array(extra, "bdotz"))
        )[:, np.newaxis, :],
        charge_native=np.asarray([constants["q_source"]], dtype=np.float64),
        dead=np.zeros((count, 1), dtype=np.bool_),
    )
    worldline = rf._prepare_source_history(arrays, 0)
    spin_mapping = source["rest_spin"]
    rest_spin = np.column_stack(
        (
            _array(spin_mapping, "x"),
            _array(spin_mapping, "y"),
            _array(spin_mapping, "z"),
        )
    )
    spin_norm = np.linalg.norm(rest_spin, axis=1)
    preserved_magnitude = (
        float(spin_norm[0])
        if spin_norm.size
        and np.allclose(spin_norm, spin_norm[0], rtol=1.0e-10, atol=1.0e-12)
        else None
    )
    dipole_source = rdf._PreparedDipoleSource(
        identity=source_role,
        worldline=worldline,
        rest_spin=rest_spin,
        rest_spin_derivative_per_ns=rdf._source_spin_slopes_per_ns(
            rest_spin, worldline.time_ns
        ),
        preserved_rest_spin_magnitude=preserved_magnitude,
        magnetic_moment_native=float(constants["magnetic_moment_native"]),
    )
    prepared_dipole = rdf._PreparedDipoleHistory(
        arrays=arrays,
        source_identities=(source_role,),
        sources={0: dipole_source},
    )
    observer_positions = observer["positions_mm"]
    event = rf.ObserverEvent(
        time_ns=float(observer["time_ns"][-1]),
        position_mm=(
            float(observer_positions["x"][-1]),
            float(observer_positions["y"][-1]),
            float(observer_positions["z"][-1]),
        ),
    )
    return worldline, prepared_dipole, event


def _observer_batch(
    source: rf._PreparedSourceHistory,
    center: rf.ObserverEvent,
    event_count: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    center_sample = rf._solve_retarded_sample(
        source,
        observer_time_ns=float(center.time_ns),
        observer_position_mm=np.asarray(center.position_mm, dtype=np.float64),
        root_tolerance_mm=_ROOT_TOLERANCE_MM,
        max_root_iterations=_MAX_ROOT_ITERATIONS,
    )
    if center_sample is None:
        raise ValueError("capture report does not bracket its center light cone")
    step_mm = max(1.0e-15, _RELATIVE_STENCIL_STEP * center_sample.separation_mm)
    offsets = rdf._full_gradient_stencil_offsets()
    base_time = np.asarray(
        [float(center.time_ns) + offset[0] * step_mm / C_MMNS for offset in offsets],
        dtype=np.float64,
    )
    center_position = np.asarray(center.position_mm, dtype=np.float64)
    base_position = np.asarray(
        [
            center_position + step_mm * np.asarray(offset[1:], dtype=np.float64)
            for offset in offsets
        ],
        dtype=np.float64,
    )
    repetitions = int(math.ceil(event_count / len(offsets)))
    event_time = np.tile(base_time, repetitions)[:event_count]
    event_position = np.tile(base_position, (repetitions, 1))[:event_count]
    return (
        np.ascontiguousarray(event_time),
        np.ascontiguousarray(event_position),
        float(step_mm),
    )


def _digest_numba_result(result: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    for value in result:
        contiguous = np.ascontiguousarray(value)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _displaced_event(
    center: rf.ObserverEvent,
    offset: tuple[int, int, int, int],
    step_mm: float,
) -> rf.ObserverEvent:
    position = np.asarray(center.position_mm, dtype=np.float64)
    position = position + step_mm * np.asarray(offset[1:], dtype=np.float64)
    return rf.ObserverEvent(
        time_ns=float(center.time_ns) + offset[0] * step_mm / C_MMNS,
        position_mm=tuple(float(value) for value in position),
    )


def _event_key(event: rf.ObserverEvent) -> tuple[float, float, float, float]:
    return (
        float(event.time_ns),
        *(float(value) for value in event.position_mm),
    )


def _assemble_cached_hertz(
    prepared: rdf._PreparedDipoleHistory,
    center: rf.ObserverEvent,
    step_mm: float,
    cached: dict[tuple[float, float, float, float], rdf.RetardedDipoleHertzResult],
) -> rdf.RetardedDipoleFieldGradientResult:
    def lookup(
        _prepared: rdf._PreparedDipoleHistory,
        event: rf.ObserverEvent,
        **_options: Any,
    ) -> rdf.RetardedDipoleHertzResult:
        return cached[_event_key(event)]

    with (
        mock.patch.object(rdf, "_prepare_dipole_history", return_value=prepared),
        mock.patch.object(
            rdf,
            "_evaluate_prepared_hertz_tensor_native",
            side_effect=lookup,
        ),
    ):
        return rdf.evaluate_retarded_dipole_field_gradient_native(
            None,  # type: ignore[arg-type]
            center,
            stencil_step_mm=step_mm,
            minimum_separation_mm=2.0e-9,
            root_tolerance_mm=_ROOT_TOLERANCE_MM,
            max_root_iterations=_MAX_ROOT_ITERATIONS,
            backend="python",
        )


def _float32_hertz_quantization_audit(
    prepared: rdf._PreparedDipoleHistory,
    center: rf.ObserverEvent,
    step_mm: float,
) -> dict[str, Any]:
    exact_cache: dict[
        tuple[float, float, float, float], rdf.RetardedDipoleHertzResult
    ] = {}
    quantized_cache: dict[
        tuple[float, float, float, float], rdf.RetardedDipoleHertzResult
    ] = {}
    hertz_exact: list[np.ndarray] = []
    hertz_quantized: list[np.ndarray] = []
    for offset in rdf._full_gradient_stencil_offsets():
        event = _displaced_event(center, offset, step_mm)
        result = rdf._evaluate_prepared_hertz_tensor_native(
            prepared,
            event,
            require_complete_history=True,
            minimum_separation_mm=2.0e-9,
            root_tolerance_mm=_ROOT_TOLERANCE_MM,
            max_root_iterations=_MAX_ROOT_ITERATIONS,
        )
        quantized_hertz = np.asarray(result.hertz_tensor, dtype=np.float32).astype(
            np.float64
        )
        exact_cache[_event_key(event)] = result
        quantized_cache[_event_key(event)] = rdf.RetardedDipoleHertzResult(
            hertz_tensor=quantized_hertz,
            source_identities=result.source_identities,
            retarded_time_ns=result.retarded_time_ns,
            light_cone_residual_mm=result.light_cone_residual_mm,
            separation_mm=result.separation_mm,
            valid_sources=result.valid_sources,
        )
        hertz_exact.append(result.hertz_tensor)
        hertz_quantized.append(quantized_hertz)

    exact = _assemble_cached_hertz(prepared, center, step_mm, exact_cache)
    quantized = _assemble_cached_hertz(prepared, center, step_mm, quantized_cache)

    def comparison(name: str) -> dict[str, Any]:
        reference = np.asarray(getattr(exact, name), dtype=np.float64)
        candidate = np.asarray(getattr(quantized, name), dtype=np.float64)
        absolute = np.abs(candidate - reference)
        reference_scale = float(np.max(np.abs(reference)))
        nonzero = np.abs(reference) > 0.0
        relative = absolute[nonzero] / np.abs(reference[nonzero])
        return {
            "maximum_absolute_difference": float(np.max(absolute)),
            "maximum_difference_over_reference_maximum": (
                float(np.max(absolute)) / reference_scale
                if reference_scale > 0.0
                else 0.0
            ),
            "median_nonzero_element_relative_difference": (
                float(np.median(relative)) if relative.size else 0.0
            ),
            "maximum_nonzero_element_relative_difference": (
                float(np.max(relative)) if relative.size else 0.0
            ),
        }

    hertz_reference = np.stack(hertz_exact)
    hertz_candidate = np.stack(hertz_quantized)
    return {
        "method": (
            "exact float64 roots and Hertz tensors, then one idealized float32 "
            "round-trip before unchanged float64 derivative assembly"
        ),
        "interpretation": (
            "lower bound on float32 field error; a real float32 Metal Hertz "
            "kernel adds arithmetic error"
        ),
        "hertz_maximum_difference_over_reference_maximum": float(
            np.max(np.abs(hertz_candidate - hertz_reference))
            / np.max(np.abs(hertz_reference))
        ),
        "arrays": {
            name: comparison(name)
            for name in (
                "four_potential",
                "partial_a",
                "field_tensor",
                "partial_f",
                "electric_field_native",
                "magnetic_field_native",
            )
        },
    }


def _digest_complete_provider(
    result: rdf.RetardedDipoleFieldGradientResult,
) -> str:
    digest = hashlib.sha256()
    for value in (
        result.four_potential,
        result.partial_a,
        result.electric_field_native,
        result.magnetic_field_native,
        result.field_tensor,
        result.partial_f,
        result.hertz.hertz_tensor,
        result.hertz.retarded_time_ns,
        result.hertz.light_cone_residual_mm,
        result.hertz.separation_mm,
        result.hertz.valid_sources,
        np.asarray([result.stencil_step_mm], dtype=np.float64),
        result.stencil_offsets,
        result.stencil_retarded_time_ns,
        result.stencil_light_cone_residual_mm,
        np.asarray([result.lorenz_gauge_residual_per_mm], dtype=np.float64),
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _time_complete_provider(
    prepared: rdf._PreparedDipoleHistory,
    center: rf.ObserverEvent,
    step_mm: float,
    *,
    backend: str,
    repeats: int = 21,
) -> dict[str, Any]:
    def evaluate() -> rdf.RetardedDipoleFieldGradientResult:
        with mock.patch.object(rdf, "_prepare_dipole_history", return_value=prepared):
            return rdf.evaluate_retarded_dipole_field_gradient_native(
                None,  # type: ignore[arg-type]
                center,
                stencil_step_mm=step_mm,
                minimum_separation_mm=2.0e-9,
                root_tolerance_mm=_ROOT_TOLERANCE_MM,
                max_root_iterations=_MAX_ROOT_ITERATIONS,
                backend=backend,
            )

    result = evaluate()
    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = evaluate()
        end = time.perf_counter_ns()
        times_ms.append((end - start) / 1.0e6)
    return {
        "median_ms": statistics.median(times_ms),
        "minimum_ms": min(times_ms),
        "maximum_ms": max(times_ms),
        "repeats": repeats,
        "complete_result_sha256": _digest_complete_provider(result),
    }


def _time_numba(
    source: rf._PreparedSourceHistory,
    event_time: np.ndarray,
    event_position: np.ndarray,
    *,
    repeats: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, ...]]:
    arguments = (
        source.time_ns,
        source.position_mm,
        source.segment_duration_ns,
        source.position_coefficients_mm,
        bool(source.ended_by_loss),
        event_time,
        event_position,
        _ROOT_TOLERANCE_MM,
        _MAX_ROOT_ITERATIONS,
    )
    result = evaluate_source_roots_exact_serial(*arguments)
    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = evaluate_source_roots_exact_serial(*arguments)
        end = time.perf_counter_ns()
        times_ms.append((end - start) / 1.0e6)
    return (
        {
            "median_ms": statistics.median(times_ms),
            "minimum_ms": min(times_ms),
            "maximum_ms": max(times_ms),
            "repeats": repeats,
            "complete_result_sha256": _digest_numba_result(result),
        },
        result,
    )


def _write_bundle(
    path: Path,
    source: rf._PreparedSourceHistory,
    event_time: np.ndarray,
    event_position: np.ndarray,
    expected: tuple[np.ndarray, ...],
) -> None:
    knot_count = int(source.time_ns.size)
    event_count = int(event_time.size)
    status = np.asarray(expected[0], dtype="<i8")
    retarded_time = np.asarray(expected[2], dtype="<f8")
    if status.shape != (event_count,) or retarded_time.shape != (event_count,):
        raise ValueError("unexpected strict CPU result shape")
    header = struct.pack(
        "<QQdQ",
        knot_count,
        event_count,
        _ROOT_TOLERANCE_MM,
        _MAX_ROOT_ITERATIONS,
    )
    values = (
        np.asarray(source.time_ns, dtype="<f8"),
        np.asarray(source.position_mm, dtype="<f8"),
        np.asarray(source.segment_duration_ns, dtype="<f8"),
        np.asarray(source.position_coefficients_mm, dtype="<f8"),
        np.asarray(event_time, dtype="<f8"),
        np.asarray(event_position, dtype="<f8"),
        status,
        retarded_time,
    )
    path.write_bytes(
        _BUNDLE_MAGIC + header + b"".join(value.tobytes() for value in values)
    )


def _parse_counts(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError(
            "counts must be comma-separated positive integers"
        )
    return result


def run_benchmark(
    input_path: Path,
    *,
    counts: tuple[int, ...],
    source_role: str,
    observer_role: str,
    timing_event_target: int,
) -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "the explicit Metal study requires Darwin arm64; portable CPU remains the default"
        )
    if not NUMBA_AVAILABLE:
        raise RuntimeError("strict CPU comparison requires the optional Numba runtime")
    original_input = input_path.read_bytes()
    source, prepared_dipole, center = _load_source_and_event(
        input_path,
        source_role=source_role,
        observer_role=observer_role,
    )
    maximum_events = max(counts)
    event_time, event_position, stencil_step_mm = _observer_batch(
        source, center, maximum_events
    )
    quantization_audit = _float32_hertz_quantization_audit(
        prepared_dipole,
        center,
        stencil_step_mm,
    )
    provider_backends = [
        backend
        for backend in (
            "python",
            "numba_roots_exact_serial",
            "numba_full_strict_serial",
        )
        if backend in rdf.RETARDED_DIPOLE_BACKENDS
    ]
    provider_timings = {
        backend: _time_complete_provider(
            prepared_dipole,
            center,
            stencil_step_mm,
            backend=backend,
        )
        for backend in provider_backends
    }

    # Compile outside every reported timing and retain one maximum-batch oracle.
    evaluate_source_roots_exact_serial(
        source.time_ns,
        source.position_mm,
        source.segment_duration_ns,
        source.position_coefficients_mm,
        bool(source.ended_by_loss),
        event_time[:1],
        event_position[:1],
        _ROOT_TOLERANCE_MM,
        _MAX_ROOT_ITERATIONS,
    )
    cpu_reports: dict[str, Any] = {}
    maximum_result: tuple[np.ndarray, ...] | None = None
    for count in counts:
        repeats = max(5, min(41, int(math.ceil(timing_event_target / count))))
        timing, result = _time_numba(
            source,
            event_time[:count],
            event_position[:count],
            repeats=repeats,
        )
        cpu_reports[str(count)] = timing
        if count == maximum_events:
            maximum_result = result
    assert maximum_result is not None

    swift_source = REPOSITORY_ROOT / "scripts" / "benchmark_metal_retarded_roots.swift"
    with tempfile.TemporaryDirectory(prefix="lw-metal-roots-") as temporary:
        temporary_path = Path(temporary)
        bundle_path = temporary_path / "capture_roots.bin"
        executable_path = temporary_path / "benchmark_metal_retarded_roots"
        _write_bundle(
            bundle_path,
            source,
            event_time,
            event_position,
            maximum_result,
        )
        compile_start = time.perf_counter_ns()
        compile_result = subprocess.run(
            ["xcrun", "swiftc", "-O", str(swift_source), "-o", str(executable_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        compile_ms = (time.perf_counter_ns() - compile_start) / 1.0e6
        swift_result = subprocess.run(
            [
                str(executable_path),
                "--input",
                str(bundle_path),
                "--counts",
                ",".join(str(value) for value in counts),
                "--timing-event-target",
                str(timing_event_target),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metal_report = json.loads(swift_result.stdout)
        bundle_sha256 = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    for scenario in metal_report["scenarios"]:
        count_key = str(int(scenario["events"]))
        numba_ms = float(cpu_reports[count_key]["median_ms"])
        hybrid_ms = float(scenario["certified_hybrid_median_ms"])
        scenario["strict_numba_cpu_median_ms"] = numba_ms
        scenario["strict_numba_cpu_over_hybrid"] = numba_ms / hybrid_ms

    if input_path.read_bytes() != original_input:
        raise RuntimeError("benchmark modified its capture input")
    return {
        "schema_version": 2,
        "study_only": True,
        "production_integrator_dispatch_changed": False,
        "selection_policy": {
            "portable_cpu_default": True,
            "metal_requires_explicit_benchmark_invocation": True,
            "required_platform": "Darwin arm64",
            "automatic_dispatch": False,
        },
        "input": {
            "path": str(input_path.resolve()),
            "sha256": hashlib.sha256(original_input).hexdigest(),
            "source_role": source_role,
            "observer_role": observer_role,
            "history_knots": int(source.time_ns.size),
            "maximum_events": maximum_events,
            "stencil_step_mm": stencil_step_mm,
            "temporary_bundle_sha256": bundle_sha256,
        },
        "strict_cpu": {
            "backend": "numba_roots_exact_serial",
            "numba_version": __import__("numba").__version__,
            "numpy_version": np.__version__,
            "timings": cpu_reports,
        },
        "complete_dipole_gradient_provider": provider_timings,
        "float32_hertz_quantization_audit": quantization_audit,
        "swift_metal": metal_report,
        "swift_compile_ms": compile_ms,
        "swift_compile_stdout": compile_result.stdout,
        "swift_compile_stderr": compile_result.stderr,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="archived capture report JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=_parse_counts, default=_DEFAULT_COUNTS)
    parser.add_argument("--source-role", default="driver")
    parser.add_argument("--observer-role", default="rider")
    parser.add_argument("--timing-event-target", type=int, default=5000)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.timing_event_target < 1:
        parser.error("--timing-event-target must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(
        args.input,
        counts=args.counts,
        source_role=args.source_role,
        observer_role=args.observer_role,
        timing_event_target=args.timing_event_target,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
