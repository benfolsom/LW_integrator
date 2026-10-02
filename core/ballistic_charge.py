"""Closed-form light cone for a massive, uniformly moving point charge.

Decimal arithmetic preserves ct minus longitudinal displacement when both
coordinates are large. Input binary64 events are treated as exact; the bound
covers root arithmetic and rounding, not uncertainty in those input events.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Sequence

import numpy as np

from .constants import C_MMNS
from .precise_charge import validated_precise_velocity


@dataclass(frozen=True)
class BallisticRetardedPoint:
    """Rounded diagnostics plus independently resolved separation components."""

    time_ns: float
    separation_vector_mm: np.ndarray
    residual_mm: float
    time_error_bound_ns: float
    longitudinal_separation_mm: float
    transverse_separation_vector_mm: np.ndarray


def ballistic_retarded_point(
    *,
    observer_time_ns: float,
    observer_position_mm: Sequence[float] | np.ndarray,
    anchor_time_ns: float,
    anchor_position_mm: Sequence[float] | np.ndarray,
    source_proper_velocity: Sequence[float] | np.ndarray,
) -> BallisticRetardedPoint:
    """Solve the positive quadratic root, with a conservative rounding bound.

    With simultaneous longitudinal separation L, transverse squared radius P²,
    and A=delta(2-delta), the retarded distance is
    (beta L + sqrt(L²+A P²))/A. For L<0 use the rationalized root. The
    longitudinal light-cone coordinate is formed as (d_parallel-ct)+delta*ct.
    The 90-digit arithmetic allowance is inflated by 10**20 and 1/A; the final
    time conversion error is measured before rounding the bound upwards.
    Supported magnitudes are |u| <= 1e12 and event coordinates <= 1e100 mm.
    """
    u = validated_precise_velocity(source_proper_velocity)
    observer = np.asarray(observer_position_mm, dtype=float)
    anchor = np.asarray(anchor_position_mm, dtype=float)
    for vector in (observer, anchor):
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("ballistic positions must contain three finite values")
    if not np.isfinite(observer_time_ns) or not np.isfinite(anchor_time_ns):
        raise ValueError("ballistic times must be finite")
    if (
        np.linalg.norm(u) > 1e12
        or np.max(np.abs(np.r_[observer, anchor])) > 1e100
        or max(abs(observer_time_ns), abs(anchor_time_ns)) > 1e100 / C_MMNS
    ):
        raise ValueError("ballistic root exceeds its documented bound domain")
    with localcontext() as context:
        context.prec = 90
        d = Decimal.from_float
        c = d(C_MMNS)
        proper = [d(float(x)) for x in u]
        magnitude = sum((x * x for x in proper), Decimal(0)).sqrt()
        gamma = (1 + magnitude * magnitude).sqrt()
        deficit = 1 / (gamma * (gamma + magnitude))
        speed = 1 - deficit
        direction = (
            [x / magnitude for x in proper]
            if magnitude
            else [Decimal(1), Decimal(0), Decimal(0)]
        )
        displacement = [d(float(x)) - d(float(y)) for x, y in zip(observer, anchor)]
        ct = c * (d(float(observer_time_ns)) - d(float(anchor_time_ns)))
        longitudinal = sum((x * y for x, y in zip(displacement, direction)), Decimal(0))
        perpendicular = [x - longitudinal * y for x, y in zip(displacement, direction)]
        transverse2 = sum((x * x for x in perpendicular), Decimal(0))
        present_longitudinal = (longitudinal - ct) + deficit * ct
        invariant = deficit * (2 - deficit)
        discriminant = (present_longitudinal**2 + invariant * transverse2).sqrt()
        if present_longitudinal >= 0:
            radius = (speed * present_longitudinal + discriminant) / invariant
        else:
            radius = (present_longitudinal**2 + transverse2) / (
                discriminant - speed * present_longitudinal
            )
        if radius <= 0:
            raise ValueError("the observer cannot coincide with the ballistic source")
        source_ct = ct - radius
        exact_time = d(float(anchor_time_ns)) + source_ct / c
        rounded_time = float(exact_time)
        scale = max(Decimal(1), abs(source_ct), abs(ct), abs(longitudinal), radius)
        arithmetic_allowance = Decimal("1e-65") * scale / (invariant * c)
        bound = abs(d(rounded_time) - exact_time) + arithmetic_allowance
        separation_longitudinal = present_longitudinal + speed * radius
        separation = [
            x + separation_longitudinal * y for x, y in zip(perpendicular, direction)
        ]
        # Evaluate the residual at the returned binary64 time, in Decimal.
        rounded_ct = c * (d(rounded_time) - d(float(anchor_time_ns)))
        vector_at_rounded_root = [
            x - speed * y * rounded_ct for x, y in zip(displacement, direction)
        ]
        residual = (
            ct
            - rounded_ct
            - sum((x * x for x in vector_at_rounded_root), Decimal(0)).sqrt()
        )
        return BallisticRetardedPoint(
            time_ns=rounded_time,
            separation_vector_mm=np.asarray([float(x) for x in separation]),
            residual_mm=float(residual),
            time_error_bound_ns=float(np.nextafter(float(bound), np.inf)),
            longitudinal_separation_mm=float(separation_longitudinal),
            transverse_separation_vector_mm=np.asarray(
                [float(x) for x in perpendicular]
            ),
        )
