"""Process-local diagnostics for the analytical dipole Hertz response."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalyticDipoleHertzDiagnostics:
    calls: int
    analytical_calls: int
    fallback_calls: int
    fallback_segment_boundary: int
    fallback_mutable_tail: int
    fallback_loss_wavefront: int
    fallback_short_history: int
    valid_sources: int
    minimum_boundary_fraction: float


_LOCK = threading.RLock()
_COUNTERS: dict[str, int | float] = {
    "calls": 0,
    "analytical_calls": 0,
    "fallback_calls": 0,
    "fallback_segment_boundary": 0,
    "fallback_mutable_tail": 0,
    "fallback_loss_wavefront": 0,
    "fallback_short_history": 0,
    "valid_sources": 0,
    "minimum_boundary_fraction": float("inf"),
}


def record_analytic_dipole_hertz_response(
    *,
    valid_sources: int,
    minimum_boundary_fraction: float,
    fallback_reason: str | None,
) -> None:
    """Record one provider call without changing numerical arithmetic."""

    with _LOCK:
        _COUNTERS["calls"] = int(_COUNTERS["calls"]) + 1
        _COUNTERS["valid_sources"] = int(_COUNTERS["valid_sources"]) + int(
            valid_sources
        )
        _COUNTERS["minimum_boundary_fraction"] = min(
            float(_COUNTERS["minimum_boundary_fraction"]),
            float(minimum_boundary_fraction),
        )
        if fallback_reason is None:
            _COUNTERS["analytical_calls"] = int(_COUNTERS["analytical_calls"]) + 1
            return
        _COUNTERS["fallback_calls"] = int(_COUNTERS["fallback_calls"]) + 1
        if "segment-boundary guard" in fallback_reason:
            key = "fallback_segment_boundary"
        elif "mutable final" in fallback_reason:
            key = "fallback_mutable_tail"
        elif "termination wavefront" in fallback_reason:
            key = "fallback_loss_wavefront"
        else:
            key = "fallback_short_history"
        _COUNTERS[key] = int(_COUNTERS[key]) + 1


def analytic_dipole_hertz_diagnostics() -> AnalyticDipoleHertzDiagnostics:
    """Return a stable snapshot for reports and acceptance tests."""

    with _LOCK:
        return AnalyticDipoleHertzDiagnostics(
            calls=int(_COUNTERS["calls"]),
            analytical_calls=int(_COUNTERS["analytical_calls"]),
            fallback_calls=int(_COUNTERS["fallback_calls"]),
            fallback_segment_boundary=int(_COUNTERS["fallback_segment_boundary"]),
            fallback_mutable_tail=int(_COUNTERS["fallback_mutable_tail"]),
            fallback_loss_wavefront=int(_COUNTERS["fallback_loss_wavefront"]),
            fallback_short_history=int(_COUNTERS["fallback_short_history"]),
            valid_sources=int(_COUNTERS["valid_sources"]),
            minimum_boundary_fraction=float(_COUNTERS["minimum_boundary_fraction"]),
        )


def reset_analytic_dipole_hertz_diagnostics() -> None:
    """Reset counters at the start of an analytical charge-plus-dipole run."""

    with _LOCK:
        for key in _COUNTERS:
            _COUNTERS[key] = float("inf") if key == "minimum_boundary_fraction" else 0


__all__ = [
    "AnalyticDipoleHertzDiagnostics",
    "analytic_dipole_hertz_diagnostics",
    "record_analytic_dipole_hertz_response",
    "reset_analytic_dipole_hertz_diagnostics",
]
