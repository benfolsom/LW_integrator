"""Closed-form light cone for a massive, uniformly moving point charge.

Decimal arithmetic preserves ct minus longitudinal displacement when both
coordinates are large. Input binary64 events are treated as exact; the bound
covers root arithmetic and rounding, not uncertainty in those input events.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache
from math import isfinite
from struct import pack, unpack
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


def _compiled_velocity_constants(proper: tuple[float, ...]) -> np.ndarray | None:
    # Tuple equality merges +0.0 and -0.0. A byte key retains every input bit.
    return _cached_velocity_constants(pack("!3d", *proper))


@lru_cache(maxsize=256)
def _cached_velocity_constants(key: bytes) -> np.ndarray | None:
    """Resolve immutable velocity constants once, retaining their Decimal values."""
    proper = unpack("!3d", key)
    if not all(isfinite(x) for x in proper):
        return None
    if max(abs(x) for x in proper) > 5e11 and np.linalg.norm(proper) > 1e12:
        return None
    with localcontext() as context:
        context.prec = 90
        values = [Decimal.from_float(x) for x in proper]
        magnitude = sum((x * x for x in values), Decimal(0)).sqrt()
        gamma = (1 + magnitude * magnitude).sqrt()
        deficit = 1 / (gamma * (gamma + magnitude))
        direction = (
            [x / magnitude for x in values]
            if magnitude
            else [Decimal(1), Decimal(0), Decimal(0)]
        )
        values_to_split = [deficit, 1 - deficit, *direction]
        context.prec = 400
        constants = np.zeros((5, 5))
        for row, value in enumerate(values_to_split):
            remainder = value
            for column in range(4):
                part = float(remainder)
                constants[row, column] = part
                remainder -= Decimal.from_float(part)
            constants[row, 4] = (
                float(np.nextafter(float(abs(remainder)), np.inf)) if remainder else 0.0
            )
        constants.flags.writeable = False
        return constants


def ballistic_retarded_point(
    *,
    observer_time_ns: float,
    observer_position_mm: Sequence[float] | np.ndarray,
    anchor_time_ns: float,
    anchor_position_mm: Sequence[float] | np.ndarray,
    source_proper_velocity: Sequence[float] | np.ndarray,
    observer_time_low_ns: float = 0.0,
    observer_position_low_mm: Sequence[float] | np.ndarray | None = None,
    observer_time_tail_ns: float = 0.0,
    observer_position_tail_mm: Sequence[float] | np.ndarray | None = None,
) -> BallisticRetardedPoint:
    """Certify every rounded diagnostic, falling back to the Decimal oracle."""
    from .certified_ballistic import ballistic_geometry_certified

    u = np.asarray(source_proper_velocity, dtype=float)
    observer = np.asarray(observer_position_mm, dtype=float)
    anchor = np.asarray(anchor_position_mm, dtype=float)
    low = (
        None
        if observer_position_low_mm is None
        else np.asarray(observer_position_low_mm, dtype=float)
    )
    tail = (
        None
        if observer_position_tail_mm is None
        else np.asarray(observer_position_tail_mm, dtype=float)
    )
    # The existing public validation and exception text remain in the oracle.
    if (
        u.shape == observer.shape == anchor.shape == (3,)
        and (low is None or low.shape == (3,))
        and (tail is None or tail.shape == (3,))
    ):
        constants = _compiled_velocity_constants(tuple(float(x) for x in u))
        certified = False
        if constants is not None:
            values, certified = ballistic_geometry_certified(
                observer_time_ns,
                observer,
                anchor_time_ns,
                anchor,
                u,
                observer_time_low_ns,
                low,
                observer_time_tail_ns,
                tail,
                constants,
            )
        if certified:
            return BallisticRetardedPoint(
                time_ns=float(values[0]),
                separation_vector_mm=values[1:4].copy(),
                residual_mm=float(values[4]),
                time_error_bound_ns=float(values[5]),
                longitudinal_separation_mm=float(values[6]),
                transverse_separation_vector_mm=values[7:10].copy(),
            )
    return ballistic_retarded_point_decimal(
        observer_time_ns=observer_time_ns,
        observer_position_mm=observer_position_mm,
        anchor_time_ns=anchor_time_ns,
        anchor_position_mm=anchor_position_mm,
        source_proper_velocity=source_proper_velocity,
        observer_time_low_ns=observer_time_low_ns,
        observer_position_low_mm=observer_position_low_mm,
        observer_time_tail_ns=observer_time_tail_ns,
        observer_position_tail_mm=observer_position_tail_mm,
    )


def ballistic_retarded_point_decimal(
    *,
    observer_time_ns: float,
    observer_position_mm: Sequence[float] | np.ndarray,
    anchor_time_ns: float,
    anchor_position_mm: Sequence[float] | np.ndarray,
    source_proper_velocity: Sequence[float] | np.ndarray,
    observer_time_low_ns: float = 0.0,
    observer_position_low_mm: Sequence[float] | np.ndarray | None = None,
    observer_time_tail_ns: float = 0.0,
    observer_position_tail_mm: Sequence[float] | np.ndarray | None = None,
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
        if observer_position_low_mm is not None:
            low = np.asarray(observer_position_low_mm, dtype=float)
            if low.shape != (3,) or not np.all(np.isfinite(low)):
                raise ValueError("ballistic observer remainders must be finite")
            displacement = [
                v + d(float(remainder)) for v, remainder in zip(displacement, low)
            ]
        if observer_position_tail_mm is not None:
            tail_array = np.asarray(observer_position_tail_mm, dtype=float)
            if tail_array.shape != (3,) or not np.all(np.isfinite(tail_array)):
                raise ValueError("ballistic observer remainders must be finite")
            displacement = [
                value + d(float(tail))
                for value, tail in zip(displacement, observer_position_tail_mm)
            ]
        if not np.isfinite(observer_time_low_ns) or not np.isfinite(
            observer_time_tail_ns
        ):
            raise ValueError("ballistic observer time remainder must be finite")
        ct = c * (
            d(float(observer_time_ns))
            + d(float(observer_time_low_ns))
            + d(float(observer_time_tail_ns))
            - d(float(anchor_time_ns))
        )
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
