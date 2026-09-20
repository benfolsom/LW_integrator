"""Experimental append-stable full-dipole interpolation from accepted samples.

Geometry-only primitive: position and velocity use caller-consistent length/time
units; the speed limit is explicit. Dipole tensors are NOT reduced to rest-spin
vectors. Five accepted samples on each side determine a frozen knot derivative.
The latest five intervals remain unavailable, never extrapolated. C4 position
and C3 dipole joins do not by themselves prove derivative accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import factorial
from typing import cast

import numpy as np


class SourcePositionError(ValueError):
    """The reconstructed source position exceeds its caller's accuracy budget."""


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
    dipole_change: np.ndarray | None = None
    dipole_error: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.position_error) or self.position_error < 0:
            raise ValueError("Finite nonnegative position error required")
        if not np.isfinite([self.start, self.duration]).all() or self.duration <= 0:
            raise ValueError("Finite start and positive duration required")
        if not np.isfinite(self.dipole_error) or self.dipole_error < 0:
            raise ValueError("Finite nonnegative dipole error required")
        dipole_shape = (11, 4, 4) if self.dipole_change is not None else (8, 4, 4)
        position_shape = np.shape(self.position)
        if position_shape not in ((10, 3), (12, 3)):
            raise ValueError("Invalid source position polynomial shape")
        for name, shape in (("position", position_shape), ("dipole", dipole_shape)):
            value = _readonly(getattr(self, name))
            if value.shape != shape:
                raise ValueError("Invalid full-dipole polynomial shape")
            object.__setattr__(self, name, value)
        if self.dipole_change is not None:
            change = _readonly(self.dipole_change)
            if change.shape != dipole_shape or not np.array_equal(
                change, -change.swapaxes(1, 2)
            ):
                raise ValueError("Invalid connected dipole change coefficients")
            if not np.array_equal(change[1:], self.dipole[1:]):
                raise ValueError("Connected absolute and change derivatives disagree")
            object.__setattr__(self, "dipole_change", change)
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
    inertial_until: float | None = None
    startup_fit: str = "centered"
    dipole_reference: np.ndarray | None = None
    dipole_high: np.ndarray | None = None
    dipole_low: np.ndarray | None = None
    derivative_sampling: str = "consecutive"
    sampling_start_knot: int = 0
    dipole_reconstruction: str = "endpoint"
    dipole_tolerance: float | None = None
    geometry_reconstruction: str = "endpoint"
    fit_sample_spacing: float | None = None

    def __post_init__(self) -> None:
        if self.geometry_reconstruction not in ("endpoint", "connected_single_fit"):
            raise ValueError("Unknown geometry reconstruction")
        if self.geometry_reconstruction == "connected_single_fit":
            if (
                self.dipole_reconstruction != "connected_direct"
                or not self.integrate_velocity
                or self.derivative_degree != 10
                or self.derivative_sampling != "consecutive"
                or isinstance(self.fit_sample_spacing, (bool, str))
                or self.fit_sample_spacing is None
                or not np.isfinite(self.fit_sample_spacing)
                or self.fit_sample_spacing <= 0
            ):
                raise ValueError(
                    "Single-fit geometry requires connected dipoles, degree ten and positive fixed fit spacing"
                )
        elif self.fit_sample_spacing is not None:
            raise ValueError("Fixed fit spacing requires single-fit geometry")
        if self.dipole_reconstruction not in ("endpoint", "connected_direct"):
            raise ValueError("Unknown dipole reconstruction")
        if self.dipole_reconstruction == "connected_direct":
            if (
                self.dipole_reference is None
                or self.inertial_until is None
                or self.startup_fit != "one_sided"
                or self.derivative_sampling != "consecutive"
                or self.dipole_tolerance is None
                or isinstance(self.dipole_tolerance, (bool, str))
                or not np.isfinite(self.dipole_tolerance)
                or self.dipole_tolerance <= 0
            ):
                raise ValueError(
                    "Connected dipoles require preserved one-sided inertial history, consecutive sampling and a positive drift budget"
                )
        elif self.dipole_tolerance is not None:
            raise ValueError(
                "Dipole drift budget requires connected_direct reconstruction"
            )
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
        if (
            self.derivative_sampling not in ("consecutive", "time_balanced")
            or type(self.sampling_start_knot) is not int
            or not 0 <= self.sampling_start_knot <= count
            or (
                self.derivative_sampling == "consecutive"
                and self.sampling_start_knot != 0
            )
        ):
            raise ValueError("Invalid derivative sampling policy or transition")
        if self.startup_fit not in ("centered", "one_sided"):
            raise ValueError("Startup fit must be centered or one_sided")
        if self.startup_fit == "one_sided" and self.inertial_until is None:
            raise ValueError(
                "One-sided startup fit requires declared inertial prehistory"
            )
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
        preserved = (self.dipole_reference, self.dipole_high, self.dipole_low)
        if any(value is not None for value in preserved):
            if (
                any(value is None for value in preserved)
                or self.derivative_degree != 10
            ):
                raise ValueError(
                    "Preserved dipoles require all components and degree ten"
                )
            for name, shape in (
                ("dipole_reference", (4, 4)),
                ("dipole_high", (count, 4, 4)),
                ("dipole_low", (count, 4, 4)),
            ):
                value = _readonly(getattr(self, name))
                if value.shape != shape or not np.array_equal(
                    value, -value.swapaxes(-1, -2)
                ):
                    raise ValueError("Invalid preserved dipole components")
                object.__setattr__(self, name, value)
            from .preserved_source import rounded_state

            actual = rounded_state(
                self.dipole_reference, self.dipole_high, self.dipole_low
            )
            if not np.array_equal(actual, self.dipole):
                raise ValueError("Preserved dipole components disagree with samples")
        if self.inertial_until is not None:
            if not np.isfinite(self.inertial_until) or not np.any(
                self.time == self.inertial_until
            ):
                raise ValueError("Inertial prehistory must end at an accepted knot")
            past = self.time <= self.inertial_until
            expected = (
                self.position[0]
                + (self.time[past] - self.time[0])[:, None] * self.velocity[0]
            )
            scale = max(1.0, float(np.max(np.abs(expected))))
            if (
                not np.all(self.velocity[past] == self.velocity[0])
                or not np.all(self.dipole[past] == self.dipole[0])
                or not np.allclose(
                    self.position[past],
                    expected,
                    rtol=0,
                    atol=64 * np.finfo(float).eps * scale,
                )
            ):
                raise ValueError(
                    "Declared inertial prehistory is not coasting with constant dipole"
                )
            if self.dipole_reference is not None and (
                not np.all(self.dipole_high[past] == self.dipole_high[0])
                or not np.all(self.dipole_low[past] == self.dipole_low[0])
            ):
                raise ValueError("Preserved inertial dipole must be constant")
        segments = tuple(self.segments)
        if len(segments) > max(0, count - 11):
            raise ValueError("Published segments exceed accepted derivative windows")
        for i, segment in enumerate(segments):
            if not isinstance(segment, FullDipoleSegment):
                raise ValueError("Published segment must be a FullDipoleSegment")
            expected_position_degree = (
                12
                if (
                    self.geometry_reconstruction == "connected_single_fit"
                    and segment.start >= self.inertial_until
                )
                else 10
            )
            if len(segment.position) != expected_position_degree:
                raise ValueError("Published geometry disagrees with history mode")
            connected = (
                self.dipole_reconstruction == "connected_direct"
                and segment.start >= self.inertial_until
            )
            if connected != (segment.dipole_change is not None):
                raise ValueError(
                    "Published dipole representation disagrees with history mode"
                )
            if connected and segment.dipole_error > self.dipole_tolerance:
                raise ValueError("Published dipole exceeds its endpoint drift budget")
            if (
                not isinstance(segment, FullDipoleSegment)
                or segment.start != self.time[i + 5]
                or not np.isclose(segment.end, self.time[i + 6], rtol=0, atol=1e-13)
            ):
                raise ValueError(
                    "Published segment does not match accepted time prefix"
                )
            if (
                self.inertial_until is not None
                and self.time[i + 6] <= self.inertial_until
            ):
                if np.any(segment.position[2:] != 0) or np.any(segment.dipole[1:] != 0):
                    raise ValueError(
                        "Published segment contradicts declared inertial prehistory"
                    )
        object.__setattr__(self, "segments", segments)

    def _window_start(self, knot: int) -> int:
        first = knot - 5
        if (
            self.startup_fit == "one_sided"
            and self.inertial_until is not None
            and self.time[knot] > self.inertial_until
        ):
            first = max(first, int(np.searchsorted(self.time, self.inertial_until)))
        return first

    def _derivatives(self, knot: int) -> tuple[np.ndarray, np.ndarray]:
        if self.inertial_until is not None and self.time[knot] <= self.inertial_until:
            # The prescribed past is known, not inferred from post-start motion.
            # The boundary knot shares these derivatives with the first evolved
            # interval; future knots still use the normal accepted-data fit.
            position = np.zeros((5, 3))
            dipole = np.zeros((4, 4, 4))
            position[0], position[1] = self.position[knot], self.velocity[0]
            dipole[0] = self.dipole[0]
            return position, dipole
        selection = self._derivative_indices(knot)
        if selection is None:
            raise ValueError(
                "Source derivatives require sufficient accepted samples on their smooth branch"
            )
        first = int(selection[0])
        times = self.time[selection] - self.time[knot]
        offset = 0.0
        if first != knot - 5:
            # Keep the fit basis centered on its window, including one-sided fits.
            offset = (times[0] + times[-1]) / 2
            times = times - offset
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

        def derivative(coefficients, order):
            if offset == 0.0:
                return coefficients[order] * factorial(order) / scale**order
            return (
                np.polynomial.polynomial.polyval(
                    -offset / scale,
                    np.polynomial.polynomial.polyder(coefficients, order, axis=0),
                )
                / scale**order
            )

        position = np.array(
            [self.position[knot], self.velocity[knot]]
            + [derivative(vc, k) for k in range(1, 4)]
        )
        dipole = np.array(
            [self.dipole[knot]] + [derivative(dc, k) for k in range(1, 4)]
        )
        return position, dipole

    def _derivative_indices(self, knot):
        if (
            self.fit_sample_spacing is not None
            and self.time[knot] > self.inertial_until
        ):
            from .single_fit_geometry import fixed_spacing_indices

            return fixed_spacing_indices(self, knot)
        if (
            self.derivative_sampling == "time_balanced"
            and knot >= self.sampling_start_knot
        ):
            from .history_sampling import time_balanced_indices

            minimum = (
                int(np.searchsorted(self.time, self.inertial_until))
                if self.startup_fit == "one_sided"
                and self.time[knot] > self.inertial_until
                else 0
            )
            return time_balanced_indices(self.time, knot, minimum)
        first = self._window_start(knot)
        return np.arange(first, first + 11) if first + 11 <= len(self.time) else None

    def with_time_balanced_sampling(self):
        """Change only unpublished knot derivatives; keep the published past."""
        if self.derivative_sampling == "time_balanced":
            return self
        return replace(
            self,
            derivative_sampling="time_balanced",
            sampling_start_knot=6 + len(self.segments) if self.segments else 0,
        )

    def with_connected_dipoles(self, tolerance: float):
        """Opt in before any evolved interval is published; never rewrite the past."""
        if self.inertial_until is None or any(
            s.end > self.inertial_until for s in self.segments
        ):
            raise ValueError(
                "Select connected dipoles before publishing evolved history"
            )
        return replace(
            self, dipole_reconstruction="connected_direct", dipole_tolerance=tolerance
        ).completed()

    def with_single_fit_geometry(
        self, *, fit_sample_spacing: float, dipole_tolerance: float
    ):
        """Select the coordinated candidate before publishing evolved intervals."""
        if self.inertial_until is None or any(
            s.end > self.inertial_until for s in self.segments
        ):
            raise ValueError(
                "Select single-fit geometry before publishing evolved history"
            )
        return replace(
            self,
            geometry_reconstruction="connected_single_fit",
            fit_sample_spacing=fit_sample_spacing,
            dipole_reconstruction="connected_direct",
            dipole_tolerance=dipole_tolerance,
        ).completed()

    def completed(self) -> FullDipoleHistory:
        """Publish only intervals with complete, accepted derivative windows."""
        from math import comb

        segments = list(self.segments)
        for left in range(5 + len(segments), len(self.time) - 6):
            indices_l, indices_r = self._derivative_indices(
                left
            ), self._derivative_indices(left + 1)
            if indices_l is None or indices_r is None:
                break  # Wait for accepted samples; never extrapolate startup data.
            xp, dp = self._derivatives(left)
            xq, dq = self._derivatives(left + 1)
            width = self.time[left + 1] - self.time[left]
            if (
                self.geometry_reconstruction == "connected_single_fit"
                and self.time[left] >= self.inertial_until
            ):
                from .single_fit_geometry import connected_position

                x = connected_position(self, left, segments[-1] if segments else None)
            elif self.integrate_velocity:
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
                raise SourcePositionError(
                    f"Integrated source position error {error:.6e} exceeds caller tolerance {self.position_tolerance + roundoff:.6e}"
                )
            dipole_change, dipole_error = None, 0.0
            if (
                self.dipole_reconstruction == "connected_direct"
                and self.time[left] >= self.inertial_until
            ):
                from .connected_dipole_history import connected_coefficients

                d, dipole_change, dipole_error = connected_coefficients(
                    self, left, segments[-1] if segments else None
                )
            elif self.dipole_reference is None:
                d = _endpoint_polynomial(dp, dq, width)
            elif (
                self.inertial_until is not None
                and self.time[left + 1] <= self.inertial_until
            ):
                d = np.zeros((8, 4, 4))
                d[0] = self.dipole[left]
            else:
                from .preserved_history_map import indexed_segment_map
                from .preserved_source import rounded_state

                selection = np.unique(np.r_[indices_l, indices_r, left, left + 1])
                mapped_left = int(np.searchsorted(selection, left))
                reconstruction = indexed_segment_map(
                    tuple(self.time[selection]),
                    mapped_left,
                    tuple(np.searchsorted(selection, indices_l)),
                    tuple(np.searchsorted(selection, indices_r)),
                    self.inertial_until is not None
                    and self.time[left] <= self.inertial_until,
                    self.inertial_until is not None
                    and self.time[left + 1] <= self.inertial_until,
                )
                dh, dl = reconstruction.apply(
                    self.dipole_high[selection],
                    self.dipole_low[selection],
                    reference_index=mapped_left,
                )
                d = dh + dl
                d[0] = rounded_state(
                    self.dipole_reference, self.dipole_high[left], self.dipole_low[left]
                )
            power = np.array([k * x[k] / width for k in range(1, len(x))])
            degree = len(power) - 1
            controls = [
                sum(comb(i, k) / comb(degree, k) * power[k] for k in range(i + 1))
                for i in range(degree + 1)
            ]
            if max(np.linalg.norm(c) for c in controls) >= self.speed_limit:
                raise ValueError("Full source interval lacks a subluminal speed bound")
            segments.append(
                FullDipoleSegment(
                    float(self.time[left]),
                    float(width),
                    x,
                    d,
                    error,
                    dipole_change,
                    dipole_error,
                )
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
            self.inertial_until,
            self.startup_fit,
            self.dipole_reference,
            self.dipole_high,
            self.dipole_low,
            self.derivative_sampling,
            self.sampling_start_knot,
            self.dipole_reconstruction,
            self.dipole_tolerance,
            self.geometry_reconstruction,
            self.fit_sample_spacing,
        )

    def append(
        self,
        time: float,
        position: np.ndarray,
        velocity: np.ndarray,
        dipole: np.ndarray,
        *,
        dipole_parts: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> FullDipoleHistory:
        """Rejected candidates leave the published prefix untouched."""
        if (dipole_parts is None) != (self.dipole_reference is None):
            raise ValueError("Preserved history requires preserved append data")
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
            self.inertial_until,
            self.startup_fit,
            self.dipole_reference,
            (
                None
                if dipole_parts is None
                else np.concatenate((self.dipole_high, dipole_parts[0][None]))
            ),
            (
                None
                if dipole_parts is None
                else np.concatenate((self.dipole_low, dipole_parts[1][None]))
            ),
            self.derivative_sampling,
            self.sampling_start_knot,
            self.dipole_reconstruction,
            self.dipole_tolerance,
            self.geometry_reconstruction,
            self.fit_sample_spacing,
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
        payload = {
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
        if self.inertial_until is not None:
            payload["format"] = "full-dipole-history-v2"
            payload["inertial_until"] = self.inertial_until
        if self.startup_fit == "one_sided":
            payload["format"] = "full-dipole-history-v3"
            payload["startup_fit"] = self.startup_fit
        if self.dipole_reference is not None:
            payload.update(
                format="full-dipole-history-v4",
                startup_fit=self.startup_fit,
                dipole_reference=self.dipole_reference.tolist(),
                dipole_high=self.dipole_high.tolist(),
                dipole_low=self.dipole_low.tolist(),
            )
        if self.derivative_sampling == "time_balanced":
            payload.update(
                format="full-dipole-history-v5",
                startup_fit=self.startup_fit,
                derivative_sampling=self.derivative_sampling,
                sampling_start_knot=self.sampling_start_knot,
            )
        if self.dipole_reconstruction == "connected_direct":
            payload.update(
                format="full-dipole-history-v6",
                dipole_reconstruction=self.dipole_reconstruction,
                dipole_tolerance=self.dipole_tolerance,
            )
        if self.geometry_reconstruction == "connected_single_fit":
            payload.update(
                format="full-dipole-history-v7",
                geometry_reconstruction=self.geometry_reconstruction,
                fit_sample_spacing=self.fit_sample_spacing,
            )
        return payload

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, object]) -> FullDipoleHistory:
        version = payload.get("format")
        single_fit = version == "full-dipole-history-v7"
        if single_fit:
            if (
                payload.get("geometry_reconstruction") != "connected_single_fit"
                or "fit_sample_spacing" not in payload
            ):
                raise ValueError(
                    "History v7 requires single-fit geometry and fit spacing"
                )
        elif "geometry_reconstruction" in payload or "fit_sample_spacing" in payload:
            raise ValueError("Single-fit settings require history v7")
        connected = version in ("full-dipole-history-v6", "full-dipole-history-v7")
        if connected:
            if (
                payload.get("dipole_reconstruction") != "connected_direct"
                or "dipole_tolerance" not in payload
            ):
                raise ValueError("History v6 requires connected dipole mode and budget")
            # Reuse the v4 preserved-data schema; v6 additionally selects reconstruction.
            version = "full-dipole-history-v4"
        elif "dipole_reconstruction" in payload or "dipole_tolerance" in payload:
            raise ValueError("Connected dipole settings require history v6")
        if version not in (
            "full-dipole-history-v1",
            "full-dipole-history-v2",
            "full-dipole-history-v3",
            "full-dipole-history-v4",
            "full-dipole-history-v5",
        ):
            raise ValueError("Unsupported full-dipole history checkpoint")
        if version not in ("full-dipole-history-v4", "full-dipole-history-v5") and (
            version in ("full-dipole-history-v2", "full-dipole-history-v3")
        ) != (payload.get("inertial_until") is not None):
            raise ValueError(
                "History version and inertial prehistory declaration disagree"
            )
        if version == "full-dipole-history-v3":
            if payload.get("startup_fit") != "one_sided":
                raise ValueError("History v3 requires one_sided startup fit")
        elif (
            version not in ("full-dipole-history-v4", "full-dipole-history-v5")
            and "startup_fit" in payload
        ):
            raise ValueError("Startup fit selector requires history v3")
        component_keys = ("dipole_reference", "dipole_high", "dipole_low")
        preserved = version == "full-dipole-history-v4" or (
            version == "full-dipole-history-v5"
            and any(key in payload for key in component_keys)
        )
        if preserved:
            if (
                any(payload.get(key) is None for key in component_keys)
                or "startup_fit" not in payload
            ):
                raise ValueError(
                    "History v4 requires preserved components and startup policy"
                )
        elif any(key in payload for key in component_keys):
            raise ValueError("Preserved dipole data require history v4")
        if version == "full-dipole-history-v5":
            if (
                payload.get("derivative_sampling") != "time_balanced"
                or "sampling_start_knot" not in payload
            ):
                raise ValueError(
                    "History v5 requires time-balanced sampling and transition"
                )
        elif "derivative_sampling" in payload or "sampling_start_knot" in payload:
            raise ValueError("Derivative sampling selector requires history v5")
        raw = cls(
            np.asarray(payload["time"]),
            np.asarray(payload["position"]),
            np.asarray(payload["velocity"]),
            np.asarray(payload["dipole"]),
            float(cast(float, payload["speed_limit"])),
            dipole_reconstruction="connected_direct" if connected else "endpoint",
            dipole_tolerance=payload["dipole_tolerance"] if connected else None,
            geometry_reconstruction=(
                "connected_single_fit" if single_fit else "endpoint"
            ),
            fit_sample_spacing=payload["fit_sample_spacing"] if single_fit else None,
            derivative_degree=cast(int, payload["derivative_degree"]),
            integrate_velocity=cast(bool, payload["integrate_velocity"]),
            position_tolerance=float(cast(float, payload["position_tolerance"])),
            inertial_until=(
                float(cast(float, payload["inertial_until"]))
                if payload.get("inertial_until") is not None
                else None
            ),
            startup_fit=(
                cast(str, payload["startup_fit"])
                if version
                in (
                    "full-dipole-history-v3",
                    "full-dipole-history-v4",
                    "full-dipole-history-v5",
                )
                else "centered"
            ),
            **(
                {key: np.asarray(payload[key]) for key in component_keys}
                if preserved
                else {}
            ),
            derivative_sampling=cast(
                str, payload.get("derivative_sampling", "consecutive")
            ),
            sampling_start_knot=cast(int, payload.get("sampling_start_knot", 0)),
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
            raw.inertial_until,
            raw.startup_fit,
            raw.dipole_reference,
            raw.dipole_high,
            raw.dipole_low,
            raw.derivative_sampling,
            raw.sampling_start_knot,
            raw.dipole_reconstruction,
            raw.dipole_tolerance,
            raw.geometry_reconstruction,
            raw.fit_sample_spacing,
        )
