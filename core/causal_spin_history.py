"""Causal, append-stable rest-spin slopes for retarded source histories.

The existing centered cubic-Hermite spin interpolant uses the next accepted
knot when assigning an interior-knot slope.  That is accurate on a completed
fixed history, but appending a future knot can revise a segment that an earlier
light-cone query already consumed.  This module provides the separately
testable slope rule needed by the future multirate history mode.

Each knot receives one slope based only on data available when that knot is
accepted.  Once two knots exist, their slopes are frozen.  Later knots use the
derivative at the newest point of the quadratic through the newest three
samples, which is second-order on a smooth nonuniform grid.  Reusing the same
frozen slope on both adjacent Hermite segments gives a causal C1 interpolant.
"""

from __future__ import annotations

import numpy as np


def _validated_spin_history(
    rest_spin: np.ndarray,
    time_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spin = np.asarray(rest_spin, dtype=np.float64)
    time = np.asarray(time_ns, dtype=np.float64)
    if spin.ndim != 2:
        raise ValueError("rest_spin must have shape [knots, components]")
    if spin.shape[1] < 1:
        raise ValueError("rest_spin needs at least one component")
    if time.shape != (spin.shape[0],):
        raise ValueError("time_ns must have one value per spin knot")
    if not np.all(np.isfinite(spin)) or not np.all(np.isfinite(time)):
        raise ValueError("spin history and time_ns must contain only finite values")
    if time.size > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("spin-history times must be strictly increasing")
    return spin, time


def _causal_slope_at_knot(
    spin: np.ndarray,
    time_ns: np.ndarray,
    knot: int,
) -> np.ndarray:
    if knot < 2:
        return np.asarray(
            (spin[1] - spin[0]) / (time_ns[1] - time_ns[0]),
            dtype=np.float64,
        )

    t0, t1, t2 = (float(value) for value in time_ns[knot - 2 : knot + 1])
    weight0 = (t2 - t1) / ((t0 - t1) * (t0 - t2))
    weight1 = (t2 - t0) / ((t1 - t0) * (t1 - t2))
    weight2 = (2.0 * t2 - t0 - t1) / ((t2 - t0) * (t2 - t1))
    return np.asarray(
        weight0 * spin[knot - 2] + weight1 * spin[knot - 1] + weight2 * spin[knot],
        dtype=np.float64,
    )


def causal_frozen_spin_slopes_per_ns(
    rest_spin: np.ndarray,
    time_ns: np.ndarray,
) -> np.ndarray:
    """Return causal C1 Hermite slopes for a complete accepted prefix.

    A one-knot history has no queryable interpolation segment and receives a
    zero placeholder.  When the second knot arrives, both endpoint slopes are
    fixed to their secant.  Every subsequent slope depends only on that knot
    and its two predecessors.
    """

    spin, time = _validated_spin_history(rest_spin, time_ns)
    count = int(time.size)
    slopes = np.zeros_like(spin)
    if count < 2:
        return slopes
    slopes[0] = _causal_slope_at_knot(spin, time, 0)
    slopes[1] = slopes[0]
    for knot in range(2, count):
        slopes[knot] = _causal_slope_at_knot(spin, time, knot)
    return slopes


def append_causal_frozen_spin_slopes_per_ns(
    previous_slopes: np.ndarray,
    rest_spin: np.ndarray,
    time_ns: np.ndarray,
) -> np.ndarray:
    """Extend causal slopes while proving the already-queryable prefix frozen."""

    spin, time = _validated_spin_history(rest_spin, time_ns)
    previous = np.asarray(previous_slopes, dtype=np.float64)
    if previous.ndim != 2 or previous.shape[1:] != spin.shape[1:]:
        raise ValueError("previous_slopes must match the spin component shape")
    old_count = int(previous.shape[0])
    if old_count > spin.shape[0]:
        raise ValueError("previous_slopes cannot exceed the spin-history length")
    if not np.all(np.isfinite(previous)):
        raise ValueError("previous_slopes must contain only finite values")

    expected_old = causal_frozen_spin_slopes_per_ns(
        spin[:old_count],
        time[:old_count],
    )
    if not np.array_equal(previous, expected_old):
        raise ValueError("previous_slopes do not match the causal accepted prefix")

    new_count = int(spin.shape[0])
    if new_count == old_count:
        return previous.copy()
    slopes = np.empty_like(spin)
    if old_count:
        slopes[:old_count] = previous

    if old_count <= 1 and new_count >= 2:
        first = _causal_slope_at_knot(spin, time, 0)
        slopes[0] = first
        slopes[1] = first
        start = 2
    else:
        start = old_count
    for knot in range(start, new_count):
        slopes[knot] = _causal_slope_at_knot(spin, time, knot)
    return slopes


__all__ = [
    "append_causal_frozen_spin_slopes_per_ns",
    "causal_frozen_spin_slopes_per_ns",
]
