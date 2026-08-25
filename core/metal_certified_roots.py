"""Certified Metal bracket proposals for exact retarded CPU kernels.

Metal contributes only float32 candidate segment indices.  Every accepted
proposal is checked against the original float64 history endpoints, and the
authoritative strict float64 CPU solver performs the root and field physics.
Any unavailable, ambiguous, or failed GPU work falls back to the unchanged
CPU search.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from .compute_backends import (
    KnotScanBatch,
    resolve_knot_scan_backend,
)
from .constants import C_MMNS

# The raw proposal/root crossover starts near 1,024 events, but original-float64
# certification and the unchanged full Hertz work move the measured production
# crossover to roughly 8,192 events on the M5 Pro.  Stay conservative.
DEFAULT_METAL_EVENT_THRESHOLD = 8192


class _Worldline(Protocol):
    time_ns: np.ndarray
    position_mm: np.ndarray
    _metal_timelike_count: int
    _metal_timelike_proof: bool


@dataclass(frozen=True)
class MetalCertifiedRootDiagnostics:
    """Process-local counters for accelerator use and exact fallback."""

    calls: int
    below_threshold_calls: int
    dispatches: int
    accepted_proposals: int
    cpu_fallbacks: int
    dispatch_failures: int


_COUNTER_LOCK = threading.RLock()
_COUNTERS = {
    "calls": 0,
    "below_threshold_calls": 0,
    "dispatches": 0,
    "accepted_proposals": 0,
    "cpu_fallbacks": 0,
    "dispatch_failures": 0,
}


def _increment(**values: int) -> None:
    with _COUNTER_LOCK:
        for name, value in values.items():
            _COUNTERS[name] += int(value)


def metal_certified_root_diagnostics() -> MetalCertifiedRootDiagnostics:
    """Return a snapshot of process-local Metal dispatch diagnostics."""

    with _COUNTER_LOCK:
        return MetalCertifiedRootDiagnostics(**_COUNTERS)


def reset_metal_certified_root_diagnostics() -> None:
    """Reset process-local counters; intended for tests and run provenance."""

    with _COUNTER_LOCK:
        for name in _COUNTERS:
            _COUNTERS[name] = 0


def _strict_timelike_chord_proof(worldline: _Worldline) -> bool:
    """Incrementally prove all stored source chords are strictly timelike."""

    count = int(worldline.time_ns.size)
    checked = int(worldline._metal_timelike_count)
    if checked > count:
        checked = 0
        worldline._metal_timelike_proof = True
    if not worldline._metal_timelike_proof:
        worldline._metal_timelike_count = count
        return False
    if count < 2:
        worldline._metal_timelike_count = count
        return False
    if checked == count:
        return True

    first_chord = max(0, checked - 1)
    delta_time = np.diff(worldline.time_ns[first_chord:count])
    displacement = np.diff(worldline.position_mm[first_chord:count], axis=0)
    light_distance = C_MMNS * delta_time
    chord_distance = np.linalg.norm(displacement, axis=1)
    scale = np.maximum(np.abs(light_distance), np.abs(chord_distance))
    margin = (
        64.0 * np.finfo(np.float64).eps * np.maximum(scale, np.finfo(np.float64).tiny)
    )
    proof = bool(
        delta_time.size > 0
        and np.all(np.isfinite(chord_distance))
        and np.all(chord_distance + margin < light_distance)
    )
    worldline._metal_timelike_proof = proof
    worldline._metal_timelike_count = count
    return proof


def certified_metal_segments(
    sources: Mapping[int, object],
    observer_time_ns: np.ndarray,
    observer_position_mm: np.ndarray,
    *,
    event_threshold: int = DEFAULT_METAL_EVENT_THRESHOLD,
) -> dict[int, np.ndarray] | None:
    """Return exact-certified segment hints keyed by prepared source index.

    ``None`` means the caller must use its ordinary strict CPU route.  Small
    batches intentionally remain on the CPU because measured dispatch and
    transfer overhead exceeds the useful work below the crossover.
    """

    source_items = tuple(sources.items())
    event_times = np.asarray(observer_time_ns, dtype=np.float64)
    event_positions = np.asarray(observer_position_mm, dtype=np.float64)
    _increment(calls=1)
    if not source_items or int(event_times.size) < int(event_threshold):
        _increment(below_threshold_calls=1)
        return None

    worldlines = [getattr(source, "worldline") for _, source in source_items]
    alive_counts = np.asarray(
        [int(worldline.time_ns.size) for worldline in worldlines], dtype=np.int64
    )
    maximum_knots = int(np.max(alive_counts, initial=0))
    if maximum_knots < 2:
        _increment(below_threshold_calls=1)
        return None

    source_times = np.zeros((maximum_knots, len(source_items)), dtype=np.float64)
    source_positions = np.zeros((maximum_knots, len(source_items), 3), dtype=np.float64)
    timelike = np.zeros(len(source_items), dtype=bool)
    for column, worldline in enumerate(worldlines):
        count = int(alive_counts[column])
        source_times[:count, column] = worldline.time_ns
        source_positions[:count, column] = worldline.position_mm
        timelike[column] = _strict_timelike_chord_proof(worldline)

    batch = KnotScanBatch(
        observer_time_ns=event_times,
        observer_position_mm=event_positions,
        source_time_ns=source_times,
        source_position_mm=source_positions,
        alive_counts=alive_counts,
    )
    try:
        resolution = resolve_knot_scan_backend("metal")
        proposals = resolution.backend.candidate_segments(batch)
        certified_segments, accepted, fallbacks = _certified_hints_float64(
            batch, proposals, timelike
        )
    except Exception:
        # The explicit backend is capability-checked before integration starts.
        # A later device/dispatch failure is numerical infrastructure failure,
        # not a reason to change the physical result or abort an otherwise
        # recoverable CPU run.
        _increment(dispatch_failures=1)
        return None

    _increment(
        dispatches=1,
        accepted_proposals=int(np.count_nonzero(accepted)),
        cpu_fallbacks=int(np.count_nonzero(fallbacks)),
    )
    return {
        source_index: certified_segments[:, column]
        for column, (source_index, _source) in enumerate(source_items)
    }


def _certified_hints_float64(
    batch: KnotScanBatch,
    proposals: np.ndarray,
    timelike_sources: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Certify proposals in vectorized float64 and mark failures for CPU search.

    Rejected proposals are encoded as ``-2``.  That sentinel tells the strict
    kernel to perform its original logarithmic float64 bracket search, avoiding
    the obsolete full-history scan used by the generic study certifier.
    """

    proposed = np.asarray(proposals, dtype=np.int64)
    expected = (batch.event_count, batch.source_count)
    if proposed.shape != expected:
        raise ValueError("Metal proposals must have shape [events, sources]")
    proof = np.asarray(timelike_sources, dtype=bool)
    if proof.shape != (batch.source_count,):
        raise ValueError("timelike proof must contain one value per source")

    source_columns = np.broadcast_to(
        np.arange(batch.source_count, dtype=np.int64), expected
    )
    alive = np.broadcast_to(batch.alive_counts[np.newaxis, :], expected)
    in_bounds = (proposed >= 0) & (proposed < alive - 1)
    safe_lower = np.where(in_bounds, proposed, 0)
    safe_upper = np.minimum(safe_lower + 1, batch.source_time_ns.shape[0] - 1)

    lower_position = batch.source_position_mm[safe_lower, source_columns]
    upper_position = batch.source_position_mm[safe_upper, source_columns]
    observer_position = batch.observer_position_mm[:, np.newaxis, :]
    lower_residual = C_MMNS * (
        batch.observer_time_ns[:, np.newaxis]
        - batch.source_time_ns[safe_lower, source_columns]
    ) - np.linalg.norm(observer_position - lower_position, axis=2)
    upper_residual = C_MMNS * (
        batch.observer_time_ns[:, np.newaxis]
        - batch.source_time_ns[safe_upper, source_columns]
    ) - np.linalg.norm(observer_position - upper_position, axis=2)
    latest_internal_knot = (upper_residual < 0.0) | (safe_upper == alive - 1)
    accepted = (
        proof[np.newaxis, :]
        & in_bounds
        & (lower_residual >= 0.0)
        & (upper_residual <= 0.0)
        & latest_internal_knot
    )
    hints = np.where(accepted, proposed, -2).astype(np.int64, copy=False)
    return hints, accepted, ~accepted


__all__ = [
    "DEFAULT_METAL_EVENT_THRESHOLD",
    "MetalCertifiedRootDiagnostics",
    "certified_metal_segments",
    "metal_certified_root_diagnostics",
    "reset_metal_certified_root_diagnostics",
]
