"""Experimental append-stable full-dipole interpolation from accepted samples.

Geometry-only primitive: position and velocity use caller-consistent length/time
units; the speed limit is explicit. Dipole tensors are NOT reduced to rest-spin
vectors. Five accepted samples on each side determine a frozen knot derivative.
The latest five intervals remain unavailable, never extrapolated. C4 position
and C3 dipole joins do not by themselves prove derivative accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import cast

import numpy as np


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Finite accepted samples required")
    result.flags.writeable = False
    return result


def _endpoint_polynomial(
    left: np.ndarray, right: np.ndarray, width: float
) -> np.ndarray:
    """Power coefficients on [0,1] matching endpoint derivatives."""
    order = len(left) - 1
    coefficients = np.zeros((2 * order + 2,) + left.shape[1:])
    for k in range(order + 1):
        coefficients[k] = left[k] * width**k / factorial(k)
    matrix = np.array(
        [
            [factorial(j) / factorial(j - k) for j in range(order + 1, 2 * order + 2)]
            for k in range(order + 1)
        ]
    )
    remainder = np.array(
        [
            right[k] * width**k
            - sum(
                factorial(j) / factorial(j - k) * coefficients[j]
                for j in range(k, order + 1)
            )
            for k in range(order + 1)
        ]
    )
    coefficients[order + 1 :] = np.linalg.solve(
        matrix, remainder.reshape(order + 1, -1)
    ).reshape(remainder.shape)
    return coefficients


@dataclass(frozen=True)
class FullDipoleSegment:
    start: float
    duration: float
    position: np.ndarray
    dipole: np.ndarray
    position_error: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.position_error) or self.position_error < 0:
            raise ValueError("Finite nonnegative position error required")
        if not np.isfinite([self.start, self.duration]).all() or self.duration <= 0:
            raise ValueError("Finite start and positive duration required")
        for name, shape in (("position", (10, 3)), ("dipole", (8, 4, 4))):
            value = _readonly(getattr(self, name))
            if value.shape != shape:
                raise ValueError("Invalid full-dipole polynomial shape")
            object.__setattr__(self, name, value)
        if not np.allclose(
            self.dipole, -self.dipole.swapaxes(1, 2), atol=1e-15, rtol=0
        ):
            raise ValueError("Antisymmetric segment dipole required")

    @property
    def end(self) -> float:
        return self.start + self.duration

    def sample(self, time: float, derivative: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if not self.start <= time <= self.end:
            raise ValueError("No source extrapolation")
        s = (time - self.start) / self.duration
        values = tuple(
            np.polynomial.polynomial.polyval(
                s, np.polynomial.polynomial.polyder(c, derivative, axis=0)
            )
            / self.duration**derivative
            for c in (self.position, self.dipole)
        )
        return values[0], values[1]


@dataclass(frozen=True)
class FullDipoleHistory:
    """Immutable reference builder; not the optimized growable backend."""

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    dipole: np.ndarray
    speed_limit: float
    segments: tuple[FullDipoleSegment, ...] = ()
    derivative_degree: int = 10
    integrate_velocity: bool = True
    position_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if np.asarray(self.time).ndim != 1:
            raise ValueError("One-dimensional accepted times required")
        if (
            not isinstance(self.integrate_velocity, bool)
            or not np.isfinite(self.position_tolerance)
            or self.position_tolerance < 0
        ):
            raise ValueError(
                "Explicit position mode and nonnegative tolerance required"
            )
        if not isinstance(
            self.derivative_degree, int
        ) or self.derivative_degree not in (8, 10):
            raise ValueError("Derivative degree must be 8 or 10")
        count = len(self.time)
        for name, shape in (
            ("time", (count,)),
            ("position", (count, 3)),
            ("velocity", (count, 3)),
            ("dipole", (count, 4, 4)),
        ):
            value = _readonly(getattr(self, name))
            if value.shape != shape:
                raise ValueError("Inconsistent accepted full-dipole sample shapes")
            object.__setattr__(self, name, value)
        if (
            not np.isfinite(self.speed_limit)
            or self.speed_limit <= 0
            or np.any(np.diff(self.time) <= 0)
            or np.any(np.linalg.norm(self.velocity, axis=1) >= self.speed_limit)
        ):
            raise ValueError(
                "Increasing times and subluminal accepted velocities required"
            )
        if not np.allclose(
            self.dipole, -self.dipole.swapaxes(1, 2), atol=1e-15, rtol=0
        ):
            raise ValueError("Antisymmetric full dipole required")
        segments = tuple(self.segments)
        if len(segments) > max(0, count - 11):
            raise ValueError("Published segments exceed accepted derivative windows")
        for i, segment in enumerate(segments):
            if (
                not isinstance(segment, FullDipoleSegment)
                or segment.start != self.time[i + 5]
                or not np.isclose(segment.end, self.time[i + 6], rtol=0, atol=1e-13)
            ):
                raise ValueError(
                    "Published segment does not match accepted time prefix"
                )
        object.__setattr__(self, "segments", segments)

    def _derivatives(self, knot: int) -> tuple[np.ndarray, np.ndarray]:
        selection = slice(knot - 5, knot + 6)
        times = self.time[selection] - self.time[knot]
        scale = max(abs(times[0]), abs(times[-1]))
        matrix = np.polynomial.polynomial.polyvander(
            times / scale, self.derivative_degree
        )
        if np.linalg.cond(matrix) > 1e8:
            raise ValueError("Accepted derivative window is ill-conditioned")
        # Fit velocity, not large absolute position values, for higher x derivatives.
        v = self.velocity[selection] - self.velocity[knot]
        d = self.dipole[selection] - self.dipole[knot]
        vc = np.linalg.lstsq(matrix, v, rcond=None)[0]
        dc = np.linalg.lstsq(matrix, d.reshape(11, 16), rcond=None)[0].reshape(
            self.derivative_degree + 1, 4, 4
        )
        position = np.array(
            [self.position[knot], self.velocity[knot]]
            + [vc[k] * factorial(k) / scale**k for k in range(1, 4)]
        )
        dipole = np.array(
            [self.dipole[knot]] + [dc[k] * factorial(k) / scale**k for k in range(1, 4)]
        )
        return position, dipole

    def completed(self) -> FullDipoleHistory:
        """Publish only intervals with complete, accepted derivative windows."""
        from math import comb

        segments = list(self.segments)
        for left in range(5 + len(segments), len(self.time) - 6):
            xp, dp = self._derivatives(left)
            xq, dq = self._derivatives(left + 1)
            width = self.time[left + 1] - self.time[left]
            if self.integrate_velocity:
                v = _endpoint_polynomial(xp[1:], xq[1:], width)
                x = np.zeros((10, 3))
                x[0] = segments[-1].sample(segments[-1].end)[0] if segments else xp[0]
                for k in range(len(v)):
                    x[k + 1] = width * v[k] / (k + 1)
            else:
                origin = xp[0].copy()
                xp[0] -= origin
                xq[0] -= origin
                x = _endpoint_polynomial(xp, xq, width)
                x[0] += origin
            error = float(
                np.linalg.norm(
                    np.polynomial.polynomial.polyval(1.0, x) - self.position[left + 1]
                )
            )
            roundoff = (
                64
                * np.finfo(float).eps
                * max(
                    np.linalg.norm(self.position[left]),
                    np.linalg.norm(self.position[left + 1]),
                )
            )
            if error > self.position_tolerance + roundoff:
                raise ValueError(
                    f"Integrated source position error {error:.6e} exceeds caller tolerance {self.position_tolerance + roundoff:.6e}"
                )
            d = _endpoint_polynomial(dp, dq, width)
            power = np.array([k * x[k] / width for k in range(1, len(x))])
            degree = len(power) - 1
            controls = [
                sum(comb(i, k) / comb(degree, k) * power[k] for k in range(i + 1))
                for i in range(degree + 1)
            ]
            if max(np.linalg.norm(c) for c in controls) >= self.speed_limit:
                raise ValueError("Full source interval lacks a subluminal speed bound")
            segments.append(
                FullDipoleSegment(float(self.time[left]), float(width), x, d, error)
            )
        return FullDipoleHistory(
            self.time,
            self.position,
            self.velocity,
            self.dipole,
            self.speed_limit,
            tuple(segments),
            self.derivative_degree,
            self.integrate_velocity,
            self.position_tolerance,
        )

    def append(
        self,
        time: float,
        position: np.ndarray,
        velocity: np.ndarray,
        dipole: np.ndarray,
    ) -> FullDipoleHistory:
        """Rejected candidates leave the published prefix untouched."""
        return FullDipoleHistory(
            np.r_[self.time, time],
            np.vstack((self.position, position)),
            np.vstack((self.velocity, velocity)),
            np.concatenate((self.dipole, np.asarray(dipole)[None])),
            self.speed_limit,
            self.segments,
            self.derivative_degree,
            self.integrate_velocity,
            self.position_tolerance,
        ).completed()

    @property
    def published_until(self) -> float | None:
        return self.segments[-1].end if self.segments else None

    def segment_at(self, time: float) -> FullDipoleSegment:
        """Reject unready source times; use the shared derivative at exact joins."""
        if (
            not np.isfinite(time)
            or not self.segments
            or time < self.segments[0].start
            or time > self.segments[-1].end
        ):
            raise ValueError("Source time outside published full-dipole history")
        index = int(np.searchsorted(self.time, time, side="right")) - 6
        return self.segments[min(index, len(self.segments) - 1)]

    def to_checkpoint_payload(self) -> dict[str, object]:
        """Versioned history-only restart; caller owns the particle checkpoint."""
        return {
            "format": "full-dipole-history-v1",
            "time": self.time.tolist(),
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "dipole": self.dipole.tolist(),
            "speed_limit": self.speed_limit,
            "published_segments": len(self.segments),
            "derivative_degree": self.derivative_degree,
            "integrate_velocity": self.integrate_velocity,
            "position_tolerance": self.position_tolerance,
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, object]) -> FullDipoleHistory:
        if payload.get("format") != "full-dipole-history-v1":
            raise ValueError("Unsupported full-dipole history checkpoint")
        raw = cls(
            np.asarray(payload["time"]),
            np.asarray(payload["position"]),
            np.asarray(payload["velocity"]),
            np.asarray(payload["dipole"]),
            float(cast(float, payload["speed_limit"])),
            derivative_degree=cast(int, payload["derivative_degree"]),
            integrate_velocity=cast(bool, payload["integrate_velocity"]),
            position_tolerance=float(cast(float, payload["position_tolerance"])),
        ).completed()
        count = payload["published_segments"]
        if not isinstance(count, int) or not 0 <= count <= len(raw.segments):
            raise ValueError("Invalid published history prefix")
        return cls(
            raw.time,
            raw.position,
            raw.velocity,
            raw.dipole,
            raw.speed_limit,
            raw.segments[:count],
            raw.derivative_degree,
            raw.integrate_velocity,
            raw.position_tolerance,
        )
