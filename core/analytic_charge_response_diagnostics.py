"""Process-local diagnostics for the experimental analytical charge response."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalyticChargeResponseDiagnostics:
    calls: int
    analytical_calls: int
    fallback_calls: int
    fallback_segment_boundary: int
    fallback_nontimelike_bound: int
    fallback_nonfinite: int
    valid_sources: int
    minimum_segment_margin_ratio: float


_LOCK = threading.RLock()
_COUNTERS: dict[str, int | float] = {
    "calls": 0,
    "analytical_calls": 0,
    "fallback_calls": 0,
    "fallback_segment_boundary": 0,
    "fallback_nontimelike_bound": 0,
    "fallback_nonfinite": 0,
    "valid_sources": 0,
    "minimum_segment_margin_ratio": float("inf"),
}


def record_analytic_charge_response(
    *,
    valid_sources: int,
    minimum_segment_margin_ratio: float,
    fallback_reason: str | None,
) -> None:
    """Record one provider call without affecting its numerical result."""

    with _LOCK:
        _COUNTERS["calls"] = int(_COUNTERS["calls"]) + 1
        _COUNTERS["valid_sources"] = int(_COUNTERS["valid_sources"]) + int(
            valid_sources
        )
        _COUNTERS["minimum_segment_margin_ratio"] = min(
            float(_COUNTERS["minimum_segment_margin_ratio"]),
            float(minimum_segment_margin_ratio),
        )
        if fallback_reason is None:
            _COUNTERS["analytical_calls"] = int(_COUNTERS["analytical_calls"]) + 1
            return
        _COUNTERS["fallback_calls"] = int(_COUNTERS["fallback_calls"]) + 1
        if "velocity_bound_is_not_timelike" in fallback_reason:
            key = "fallback_nontimelike_bound"
        elif "segment_boundary" in fallback_reason:
            key = "fallback_segment_boundary"
        else:
            key = "fallback_nonfinite"
        _COUNTERS[key] = int(_COUNTERS[key]) + 1


def analytic_charge_response_diagnostics() -> AnalyticChargeResponseDiagnostics:
    """Return a stable snapshot for run provenance and tests."""

    with _LOCK:
        return AnalyticChargeResponseDiagnostics(
            calls=int(_COUNTERS["calls"]),
            analytical_calls=int(_COUNTERS["analytical_calls"]),
            fallback_calls=int(_COUNTERS["fallback_calls"]),
            fallback_segment_boundary=int(_COUNTERS["fallback_segment_boundary"]),
            fallback_nontimelike_bound=int(_COUNTERS["fallback_nontimelike_bound"]),
            fallback_nonfinite=int(_COUNTERS["fallback_nonfinite"]),
            valid_sources=int(_COUNTERS["valid_sources"]),
            minimum_segment_margin_ratio=float(
                _COUNTERS["minimum_segment_margin_ratio"]
            ),
        )


def reset_analytic_charge_response_diagnostics() -> None:
    """Reset counters at the start of an explicitly analytical run."""

    with _LOCK:
        for key in _COUNTERS:
            _COUNTERS[key] = (
                float("inf") if key == "minimum_segment_margin_ratio" else 0
            )


__all__ = [
    "AnalyticChargeResponseDiagnostics",
    "analytic_charge_response_diagnostics",
    "record_analytic_charge_response",
    "reset_analytic_charge_response_diagnostics",
]
