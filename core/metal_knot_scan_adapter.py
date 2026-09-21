"""Optional Apple-Metal knot-bracket proposal backend.

The GPU returns float32 segment proposals only.  Callers must certify those
proposals with :func:`core.compute_backends.certify_candidate_segments_float64`
before using them to seed the authoritative float64 root solver.
"""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import os
import platform
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from .compute_backends import (
    BackendCapabilities,
    BackendSelfTestResult,
    ComputeBackendUnavailableError,
    KnotScanBatch,
    certify_candidate_segments_float64,
    latest_light_cone_segments_float64,
    strictly_timelike_source_chords_float64,
)

_ERROR_CAPACITY = 2048
_BUILD_LOCK = threading.RLock()


def _native_source_path() -> Path:
    return Path(__file__).resolve().parent / "native" / "metal_knot_scan.mm"


def _compiled_library_path() -> Path:
    source = _native_source_path()
    if not source.is_file():
        raise ComputeBackendUnavailableError(
            f"Metal native source asset is missing: {source}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache_dir = Path.home() / "Library" / "Caches" / "lw-integrator" / "metal"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"liblw_metal_knot_scan_{digest}.dylib"


def _compile_native_library() -> Path:
    output = _compiled_library_path()
    if output.is_file():
        return output
    source = _native_source_path()
    temporary = output.with_name(f"{output.stem}.{os.getpid()}.tmp.dylib")
    command = (
        "xcrun",
        "clang++",
        "-std=c++17",
        "-O3",
        "-fobjc-arc",
        "-Wno-deprecated-declarations",
        "-dynamiclib",
        str(source),
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-o",
        str(temporary),
    )
    with _BUILD_LOCK:
        if output.is_file():
            return output
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=120.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ComputeBackendUnavailableError(
                f"Metal native adapter compilation could not start: {exc}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise ComputeBackendUnavailableError(
                "Metal native adapter compilation failed: " + detail
            )
        temporary.replace(output)
    return output


def _load_library() -> ctypes.CDLL:
    library = ctypes.CDLL(str(_compile_native_library()))
    library.lw_metal_knot_scan_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    library.lw_metal_knot_scan_create.restype = ctypes.c_void_p
    library.lw_metal_knot_scan_destroy.argtypes = [ctypes.c_void_p]
    library.lw_metal_knot_scan_destroy.restype = None
    library.lw_metal_knot_scan_device_name.argtypes = [ctypes.c_void_p]
    library.lw_metal_knot_scan_device_name.restype = ctypes.c_char_p
    float_pointer = ctypes.POINTER(ctypes.c_float)
    int_pointer = ctypes.POINTER(ctypes.c_int32)
    library.lw_metal_knot_scan_candidates.argtypes = [
        ctypes.c_void_p,
        float_pointer,
        float_pointer,
        ctypes.c_uint32,
        float_pointer,
        float_pointer,
        ctypes.c_uint32,
        ctypes.c_uint32,
        int_pointer,
        int_pointer,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.lw_metal_knot_scan_candidates.restype = ctypes.c_int
    return library


class MetalKnotScanBackend:
    """Real safe-math Metal proposal backend with synchronous dispatch."""

    def __init__(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ComputeBackendUnavailableError(
                "the Metal adapter requires Apple-silicon macOS"
            )
        self._library = _load_library()
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        context = self._library.lw_metal_knot_scan_create(error, len(error))
        if not context:
            detail = error.value.decode("utf-8", errors="replace")
            raise ComputeBackendUnavailableError(
                f"Metal backend initialization failed: {detail}"
            )
        self._context = ctypes.c_void_p(context)
        name = self._library.lw_metal_knot_scan_device_name(self._context)
        self._device_name = (
            name.decode("utf-8", errors="replace") if name else "Apple Metal GPU"
        )
        self._closed = False
        atexit.register(self.close)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="metal",
            device_name=self._device_name,
            accelerator=True,
            exact_float64=False,
            approximate_candidates_only=True,
        )

    def close(self) -> None:
        if not self._closed:
            self._library.lw_metal_knot_scan_destroy(self._context)
            self._closed = True

    def candidate_segments(self, batch: KnotScanBatch) -> np.ndarray:
        if self._closed:
            raise ComputeBackendUnavailableError("Metal backend is closed")
        observer_times = np.ascontiguousarray(batch.observer_time_ns, dtype=np.float32)
        observer_positions = np.ascontiguousarray(
            batch.observer_position_mm, dtype=np.float32
        )
        source_times = np.ascontiguousarray(batch.source_time_ns, dtype=np.float32)
        source_positions = np.ascontiguousarray(
            batch.source_position_mm, dtype=np.float32
        )
        alive_counts = np.ascontiguousarray(batch.alive_counts, dtype=np.int32)
        output = np.empty((batch.event_count, batch.source_count), dtype=np.int32)
        error = ctypes.create_string_buffer(_ERROR_CAPACITY)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        int_pointer = ctypes.POINTER(ctypes.c_int32)
        status = self._library.lw_metal_knot_scan_candidates(
            self._context,
            observer_times.ctypes.data_as(float_pointer),
            observer_positions.ctypes.data_as(float_pointer),
            ctypes.c_uint32(batch.event_count),
            source_times.ctypes.data_as(float_pointer),
            source_positions.ctypes.data_as(float_pointer),
            ctypes.c_uint32(batch.source_time_ns.shape[0]),
            ctypes.c_uint32(batch.source_count),
            alive_counts.ctypes.data_as(int_pointer),
            output.ctypes.data_as(int_pointer),
            error,
            len(error),
        )
        if status != 0:
            detail = error.value.decode("utf-8", errors="replace")
            raise ComputeBackendUnavailableError(
                f"Metal proposal dispatch failed ({status}): {detail}"
            )
        return output.astype(np.int64, copy=False)

    def self_test(self) -> BackendSelfTestResult:
        source_times = np.linspace(-0.02, 0.002, 45)[:, np.newaxis]
        source_positions = np.zeros((source_times.shape[0], 1, 3), dtype=float)
        batch = KnotScanBatch(
            observer_time_ns=np.array((0.0, 0.0005), dtype=float),
            observer_position_mm=np.array(
                ((1.0, 0.0, 0.0), (1.5, 0.0, 0.0)), dtype=float
            ),
            source_time_ns=source_times,
            source_position_mm=source_positions,
            alive_counts=np.array((source_times.shape[0],), dtype=np.int64),
        )
        reference = latest_light_cone_segments_float64(batch)
        proposals = self.candidate_segments(batch)
        proof = strictly_timelike_source_chords_float64(batch)
        certified = certify_candidate_segments_float64(
            batch, proposals, strictly_timelike_sources=proof
        )
        if not np.array_equal(certified.segment_indices, reference):
            return BackendSelfTestResult(False, "certified Metal proposal mismatch")
        bad = proposals.copy()
        bad[0, 0] += 1
        recovered = certify_candidate_segments_float64(
            batch, bad, strictly_timelike_sources=proof
        )
        if not (
            bool(recovered.cpu_fallbacks[0, 0])
            and np.array_equal(recovered.segment_indices, reference)
        ):
            return BackendSelfTestResult(False, "float64 fallback self-test failed")
        return BackendSelfTestResult(
            True, "safe-math proposals passed float64 certification and fallback"
        )


@lru_cache(maxsize=1)
def create_knot_scan_backend() -> MetalKnotScanBackend:
    """Return the process-wide persistent Metal proposal backend."""

    return MetalKnotScanBackend()


__all__ = ["MetalKnotScanBackend", "create_knot_scan_backend"]
