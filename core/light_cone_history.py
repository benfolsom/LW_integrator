"""Resolved quintic charge worldlines in a segment-local null frame.

A segment stores w = c(t-t0) - e.(x-x0) and two transverse displacements.
The frame is the normalized sum of endpoint velocities, with the x axis as
fallback when that sum is zero. Endpoint transformations use 80 Decimal digits
so normalizing a frame does not introduce an artificial light-speed deficit.
Decimal is used while preparing knots, never as an interpolation model.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Sequence, cast

import numpy as np

from .constants import C_MMNS


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _dot(a: Sequence[Decimal], b: Sequence[Decimal]) -> Decimal:
    return sum((x * y for x, y in zip(a, b)), Decimal(0))


def _normalize(vector: Sequence[Decimal]) -> list[Decimal]:
    length = _dot(vector, vector).sqrt()
    return [v / length for v in vector]


@dataclass(frozen=True)
class LightConeSegment:
    """One immutable local null quintic; accepted endpoints remain authoritative."""

    time_ns: float
    duration_ns: float
    frame: np.ndarray
    frame_low: np.ndarray
    coefficients_mm: np.ndarray
    origin_position_mm: np.ndarray
    origin_position_low_mm: np.ndarray
    origin_time_low_ns: float
    speed_deficit: np.ndarray
    proper_velocity_knots: np.ndarray | None = None
    origin_position_tail_mm: np.ndarray | None = None
    origin_time_tail_ns: float = 0.0
    endpoint_null_mm: np.ndarray | None = None
    # Rows 0–2 are upper position high/low/tail (mm); row 3 is time (ns).
    endpoint_coordinates: np.ndarray | None = None

    def __post_init__(self) -> None:
        for value in (
            self.frame,
            self.frame_low,
            self.coefficients_mm,
            self.origin_position_mm,
            self.origin_position_low_mm,
            self.speed_deficit,
            self.proper_velocity_knots,
            self.origin_position_tail_mm,
            self.endpoint_null_mm,
            self.endpoint_coordinates,
        ):
            if value is not None:
                value.flags.writeable = False

    def sample(
        self, fraction: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        tau = float(fraction)
        values = self.coefficients_mm
        position = cast(np.ndarray, np.polynomial.polynomial.polyval(tau, values))
        first = cast(
            np.ndarray,
            np.polynomial.polynomial.polyval(
                tau, values[1:] * np.arange(1, 6)[:, None]
            ),
        ) / (C_MMNS * self.duration_ns)
        second = (
            np.polynomial.polynomial.polyval(
                tau, values[2:] * (np.arange(2, 6) * np.arange(1, 5))[:, None]
            )
            / (C_MMNS * self.duration_ns) ** 2
        )
        longitudinal = C_MMNS * self.duration_ns * tau - position[0]
        lab_position = self.origin_position_mm + (
            np.array([longitudinal, position[1], position[2]]) @ self.frame
        )
        beta = np.array([1.0 - first[0], first[1], first[2]]) @ self.frame
        beta_prime = np.array([-second[0], second[1], second[2]]) @ self.frame
        invariant = first[0] * (2.0 - first[0]) - np.dot(first[1:], first[1:])
        return lab_position, beta, beta_prime, float(invariant)


def build_light_cone_segment(
    times: np.ndarray,
    positions: np.ndarray,
    betas: np.ndarray,
    beta_primes: np.ndarray,
    *,
    proper_velocity: np.ndarray | None = None,
    position_low: np.ndarray | None = None,
    time_low: np.ndarray | None = None,
    position_tail: np.ndarray | None = None,
    time_tail: np.ndarray | None = None,
) -> LightConeSegment:
    """Transform endpoint Hermite data without subtracting rounded unit speeds."""
    with localcontext() as context:
        context.prec = 80
        t = [_decimal(v) for v in times]
        if time_low is not None:
            t = [v + _decimal(low) for v, low in zip(t, time_low)]
        if time_tail is not None:
            t = [value + _decimal(tail) for value, tail in zip(t, time_tail)]
        duration = t[1] - t[0]
        if duration <= 0:
            raise ValueError("source coordinate times must be strictly increasing")
        c = _decimal(C_MMNS)
        length = c * duration
        raw = (
            [_decimal(v) for v in betas[0] + betas[1]]
            if proper_velocity is None
            else [_decimal(v) for v in proper_velocity[0] + proper_velocity[1]]
        )
        if (
            np.linalg.norm(betas[0] + betas[1]) < 1e-12
            or float(betas[0] @ betas[1]) < 0.0
        ):
            raw = [Decimal(1), Decimal(0), Decimal(0)]
        axis = _normalize(raw)
        least = min(range(3), key=lambda i: abs(axis[i]))
        fallback = [Decimal(int(i == least)) for i in range(3)]
        transverse = _normalize(
            [
                axis[1] * fallback[2] - axis[2] * fallback[1],
                axis[2] * fallback[0] - axis[0] * fallback[2],
                axis[0] * fallback[1] - axis[1] * fallback[0],
            ]
        )
        other = [
            axis[1] * transverse[2] - axis[2] * transverse[1],
            axis[2] * transverse[0] - axis[0] * transverse[2],
            axis[0] * transverse[1] - axis[1] * transverse[0],
        ]
        axes = [axis, transverse, other]
        frame = np.array([[float(v) for v in row] for row in axes])
        frame_low = np.array(
            [
                [float(v - _decimal(high)) for v, high in zip(row, h)]
                for row, h in zip(axes, frame)
            ]
        )
        x = [[_decimal(v) for v in row] for row in positions]
        if position_low is not None:
            x = [
                [v + _decimal(low) for v, low in zip(row, lows)]
                for row, lows in zip(x, position_low)
            ]
        if position_tail is not None:
            x = [
                [value + _decimal(tail) for value, tail in zip(row, tails)]
                for row, tails in zip(x, position_tail)
            ]
        displacement = [b - a for a, b in zip(x[0], x[1])]
        end = [
            length - _dot(axis, displacement),
            _dot(transverse, displacement),
            _dot(other, displacement),
        ]
        velocities = []
        deficits = []
        for knot in range(2):
            if proper_velocity is None:
                beta = [_decimal(v) for v in betas[knot]]
                magnitude = _dot(beta, beta).sqrt()
                deficit = 1 - magnitude
                slope = 1 - _dot(axis, beta)
            else:
                u = [_decimal(v) for v in proper_velocity[knot]]
                magnitude = _dot(u, u).sqrt()
                gamma = (1 + magnitude * magnitude).sqrt()
                deficit = 1 / (gamma * (gamma + magnitude))
                beta = [v / gamma for v in u]
                if magnitude == 0:
                    slope = Decimal(1)
                else:
                    direction = [v / magnitude for v in u]
                    difference = [a - b for a, b in zip(axis, direction)]
                    slope = (
                        deficit + (magnitude / gamma) * _dot(difference, difference) / 2
                    )
            deficits.append(float(deficit))
            velocities.append([slope, _dot(transverse, beta), _dot(other, beta)])
        acceleration = []
        for row in beta_primes:
            a = [_decimal(v) for v in row]
            acceleration.append([-_dot(axis, a), _dot(transverse, a), _dot(other, a)])
        data = np.zeros((6, 3))
        ballistic = (
            np.array_equal(betas[0], betas[1])
            and np.all(beta_primes == 0.0)
            and (
                proper_velocity is None
                or np.array_equal(proper_velocity[0], proper_velocity[1])
            )
        )
        for component in range(3):
            c1 = length * velocities[0][component]
            c2 = length * length * acceleration[0][component] / 2
            remainder = end[component] - c1 - c2
            vr = length * velocities[1][component] - c1 - 2 * c2
            ar = length * length * acceleration[1][component] - 2 * c2
            coeff = [
                Decimal(0),
                c1,
                c2,
                10 * remainder - 4 * vr + ar / 2,
                -15 * remainder + 7 * vr - ar,
                6 * remainder - 3 * vr + ar / 2,
            ]
            # Match the established coasting endpoint-roundoff rule only when
            # the accepted lab displacement agrees with the endpoint velocity.
            roundoff = (
                8
                * np.finfo(float).eps
                * max(np.max(np.abs(positions)), abs(float(length)), 1e-300)
            )
            if ballistic and abs(float(remainder)) <= roundoff:
                coeff[2:] = [Decimal(0)] * 4
            data[:, component] = [float(v) for v in coeff]
        return LightConeSegment(
            float(times[0]),
            float(duration),
            frame,
            frame_low,
            data,
            np.array(positions[0], copy=True),
            (
                np.zeros(3)
                if position_low is None
                else np.array(position_low[0], copy=True)
            ),
            0.0 if time_low is None else float(time_low[0]),
            np.array(deficits),
            None if proper_velocity is None else np.array(proper_velocity, copy=True),
            (
                np.zeros(3)
                if position_tail is None
                else np.array(position_tail[0], copy=True)
            ),
            0.0 if time_tail is None else float(time_tail[0]),
            np.array([float(value) for value in end]),
            np.array(
                [
                    positions[1],
                    np.zeros(3) if position_low is None else position_low[1],
                    np.zeros(3) if position_tail is None else position_tail[1],
                    [
                        times[1],
                        0.0 if time_low is None else time_low[1],
                        0.0 if time_tail is None else time_tail[1],
                    ],
                ]
            ),
        )


def endpoint_geometry(
    segment: LightConeSegment,
    observer_time_ns: float,
    observer_position_mm: np.ndarray,
    observer_time_low_ns: float = 0.0,
    observer_position_low_mm: np.ndarray | None = None,
    observer_time_tail_ns: float = 0.0,
    observer_position_tail_mm: np.ndarray | None = None,
    *,
    include_endpoint: bool = False,
) -> np.ndarray:
    """Resolve anchor geometry, optionally at both accepted endpoints.

    The second anchor keeps evaluation near the upper knot consistent with
    the next segment's lower knot, including at a zero coordinate time.
    """
    with localcontext() as context:
        context.prec = 80
        dt = (
            _decimal(observer_time_ns)
            + _decimal(observer_time_low_ns)
            - _decimal(segment.time_ns)
            - _decimal(segment.origin_time_low_ns)
        )
        dt += _decimal(observer_time_tail_ns) - _decimal(segment.origin_time_tail_ns)
        displacement = [
            _decimal(v) - _decimal(x) - _decimal(low)
            for v, x, low in zip(
                observer_position_mm,
                segment.origin_position_mm,
                segment.origin_position_low_mm,
            )
        ]
        if segment.origin_position_tail_mm is not None:
            displacement = [
                value - _decimal(tail)
                for value, tail in zip(displacement, segment.origin_position_tail_mm)
            ]
        if observer_position_tail_mm is not None:
            displacement = [
                value + _decimal(tail)
                for value, tail in zip(displacement, observer_position_tail_mm)
            ]
        if observer_position_low_mm is not None:
            displacement = [
                v + _decimal(low)
                for v, low in zip(displacement, observer_position_low_mm)
            ]
        axes = [
            [_decimal(high) + _decimal(low) for high, low in zip(frame_high, frame_low)]
            for frame_high, frame_low in zip(segment.frame, segment.frame_low)
        ]
        parallel = _dot(axes[0], displacement)
        coordinate = _decimal(C_MMNS) * dt
        geometry = [
            float(coordinate - parallel),
            float(parallel),
            float(_dot(axes[1], displacement)),
            float(_dot(axes[2], displacement)),
            float(coordinate + parallel),
        ]
        endpoint = segment.endpoint_coordinates
        if include_endpoint and endpoint is not None:
            dt += (
                _decimal(segment.time_ns)
                + _decimal(segment.origin_time_low_ns)
                + _decimal(segment.origin_time_tail_ns)
                - sum((_decimal(v) for v in endpoint[3]), Decimal(0))
            )
            displacement = [
                value
                + _decimal(high)
                + _decimal(low)
                + _decimal(tail)
                - sum((_decimal(v) for v in endpoint[:3, axis]), Decimal(0))
                for axis, (value, high, low, tail) in enumerate(
                    zip(
                        displacement,
                        segment.origin_position_mm,
                        segment.origin_position_low_mm,
                        (
                            np.zeros(3)
                            if segment.origin_position_tail_mm is None
                            else segment.origin_position_tail_mm
                        ),
                    )
                )
            ]
            parallel = _dot(axes[0], displacement)
            coordinate = _decimal(C_MMNS) * dt
            geometry.extend(
                [
                    float(coordinate - parallel),
                    float(parallel),
                    float(_dot(axes[1], displacement)),
                    float(_dot(axes[2], displacement)),
                    float(coordinate + parallel),
                ]
            )
        return np.array(geometry)


def separation_in_velocity_frame(
    segment: LightConeSegment, separation: np.ndarray, proper_velocity: np.ndarray
) -> np.ndarray:
    """Rotate resolved separation components without forming a large lab vector."""
    with localcontext() as context:
        context.prec = 80
        axes = [
            [
                _decimal(part_high) + _decimal(part_low)
                for part_high, part_low in zip(high, low)
            ]
            for high, low in zip(segment.frame, segment.frame_low)
        ]
        u = [_decimal(v) for v in proper_velocity]
        if _dot(u, u) == 0:
            u = [Decimal(1), Decimal(0), Decimal(0)]
        direction = _normalize(u)
        least = min(range(3), key=lambda i: abs(direction[i]))
        fixed = [Decimal(int(i == least)) for i in range(3)]
        transverse = _normalize(
            [
                direction[1] * fixed[2] - direction[2] * fixed[1],
                direction[2] * fixed[0] - direction[0] * fixed[2],
                direction[0] * fixed[1] - direction[1] * fixed[0],
            ]
        )
        other = [
            direction[1] * transverse[2] - direction[2] * transverse[1],
            direction[2] * transverse[0] - direction[0] * transverse[2],
            direction[0] * transverse[1] - direction[1] * transverse[0],
        ]
        return np.array(
            [
                float(
                    sum(
                        (
                            _decimal(value) * _dot(axis, new_axis)
                            for value, axis in zip(separation, axes)
                        ),
                        Decimal(0),
                    )
                )
                for new_axis in [direction, transverse, other]
            ]
        )


def segment_speed_deficit_bound(segment: LightConeSegment) -> float:
    """Outward-rounded Bernstein lower bound on 1-|beta| for a segment.

    Bound the degree-eight invariant 2w'-w'^2-|x_perp'|^2 directly. Bounding
    |beta| first would round its norm to one for high-gamma coasting sources.
    """
    from math import comb

    proper_bound = getattr(segment, "speed_deficit_bound", None)
    if proper_bound is not None:
        return float(proper_bound())
    with localcontext() as context:
        context.prec = 80
        length = _decimal(C_MMNS * segment.duration_ns)
        derivative = [
            [
                _decimal(segment.coefficients_mm[order + 1, component])
                * (order + 1)
                / length
                for order in range(5)
            ]
            for component in range(3)
        ]
        power = [Decimal(0)] * 9
        for order in range(5):
            power[order] += 2 * derivative[0][order]
            for other in range(5):
                for component in range(3):
                    power[order + other] -= (
                        derivative[component][order] * derivative[component][other]
                    )
        controls = [
            sum(
                (
                    power[order] * Decimal(comb(index, order)) / Decimal(comb(8, order))
                    for order in range(index + 1)
                ),
                Decimal(0),
            )
            for index in range(9)
        ]
        invariant = min(controls)
        if invariant <= 0 or invariant > 1:
            return 0.0
        deficit = invariant / (1 + (1 - invariant).sqrt())
        return float(np.nextafter(float(deficit), 0.0))
