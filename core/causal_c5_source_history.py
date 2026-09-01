"""Causal, checkpointable $C^5$ source segments from accepted samples.

This module is deliberately independent of the retarded-field provider.  It
turns an accepted source prefix into immutable polynomial segments and makes
the readiness boundary explicit.  A caller may construct a candidate state
for an adaptive trial, but the accepted state is unchanged unless that
candidate is committed.

Position derivatives through fifth order are reconstructed from a seven-knot
window.  Unit rest spin is represented by two stereographic coordinates; a
degree-ten fit over fifteen accepted knots estimates derivatives through fifth
order.  Neighboring degree-eleven Hermite segments reuse the same knot
derivatives and are therefore $C^5$ in the represented coordinates.

The fixed stereographic frame is part of the checkpointed model.  A history
that approaches its excluded chart pole fails closed instead of silently
changing frames or renormalizing component polynomials.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Mapping, Sequence, cast

import numpy as np

from .constants import C_MMNS

_CHECKPOINT_SCHEMA_VERSION = 1
_POSITION_HALF_WINDOW = 3
_SPIN_HALF_WINDOW = 7
_SPIN_FIT_DEGREE = 10
_CONTINUITY_ORDER = 5
_CHART_POLE_TOLERANCE = 1.0e-8
_MAXIMUM_CONDITION_NUMBER = 1.0e5


class CausalC5HistoryUnavailableError(RuntimeError):
    """Raised when no frozen $C^5$ segment covers a requested source time."""


def _readonly_array(
    value: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    shape: tuple[int, ...] | None,
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(result, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validated_frame(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    frame = _readonly_array(value, shape=(3, 3), name="stereographic_frame")
    if not np.allclose(frame.T @ frame, np.eye(3), rtol=0.0, atol=2.0e-13):
        raise ValueError("stereographic_frame must be orthonormal")
    if not np.isclose(np.linalg.det(frame), 1.0, rtol=0.0, atol=2.0e-13):
        raise ValueError("stereographic_frame must be right handed")
    return frame


def _readonly_index_array(
    value: Sequence[Sequence[int]] | np.ndarray,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    result = np.array(result, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def _hermite_endpoint_matrix() -> np.ndarray:
    size = 2 * (_CONTINUITY_ORDER + 1)
    matrix = np.zeros((size, size), dtype=np.float64)
    row = 0
    for endpoint in (0.0, 1.0):
        for derivative_order in range(_CONTINUITY_ORDER + 1):
            for power in range(derivative_order, size):
                coefficient = math.factorial(power) / math.factorial(
                    power - derivative_order
                )
                matrix[row, power] = coefficient * endpoint ** (
                    power - derivative_order
                )
            row += 1
    matrix.setflags(write=False)
    return matrix


def _hermite_endpoint_coefficients(
    *,
    duration_ns: float,
    start_derivatives: np.ndarray,
    end_derivatives: np.ndarray,
) -> np.ndarray:
    duration = float(duration_ns)
    start = np.asarray(start_derivatives, dtype=np.float64)
    end = np.asarray(end_derivatives, dtype=np.float64)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("segment duration must be finite and positive")
    if start.ndim != 2 or start.shape[0] != _CONTINUITY_ORDER + 1:
        raise ValueError("endpoint derivatives must contain orders zero through five")
    if (
        end.shape != start.shape
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(end))
    ):
        raise ValueError("endpoint derivatives must be matching finite matrices")
    right_hand_side = np.empty((12, start.shape[1]), dtype=np.float64)
    row = 0
    for endpoint in (start, end):
        for derivative_order in range(_CONTINUITY_ORDER + 1):
            right_hand_side[row] = (
                duration**derivative_order * endpoint[derivative_order]
            )
            row += 1
    return cast(
        np.ndarray,
        np.linalg.solve(_hermite_endpoint_matrix(), right_hand_side),
    )


def _scaled_derivative_weights(
    times_ns: np.ndarray,
    *,
    center_index: int,
    derivative_order: int,
) -> tuple[np.ndarray, float]:
    offsets = np.asarray(times_ns, dtype=np.float64) - float(times_ns[center_index])
    scale = float(np.max(np.abs(offsets)))
    if scale <= 0.0:
        raise ValueError("accepted sample times must span a nonzero interval")
    normalized = offsets / scale
    powers = np.arange(times_ns.size, dtype=np.float64)[:, np.newaxis]
    system = normalized[np.newaxis, :] ** powers
    right_hand_side = np.zeros(times_ns.size, dtype=np.float64)
    right_hand_side[derivative_order] = float(math.factorial(derivative_order))
    return (
        cast(
            np.ndarray,
            np.linalg.solve(system, right_hand_side) / scale**derivative_order,
        ),
        float(np.linalg.cond(system)),
    )


def _position_derivatives_at_knot(
    history: "CausalC5SourceHistory",
    knot_index: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    start = knot_index - _POSITION_HALF_WINDOW
    stop = knot_index + _POSITION_HALF_WINDOW + 1
    if start < 0 or stop > history.sample_count:
        raise CausalC5HistoryUnavailableError(
            "position derivative window is not fully accepted"
        )
    times = history.time_ns[start:stop]
    acceleration = C_MMNS**2 * history.beta_prime_per_mm[start:stop]
    acceleration_delta = acceleration - acceleration[_POSITION_HALF_WINDOW]
    higher_derivatives: list[np.ndarray] = []
    conditions: list[float] = []
    for derivative_order in (1, 2, 3):
        weights, condition = _scaled_derivative_weights(
            times,
            center_index=_POSITION_HALF_WINDOW,
            derivative_order=derivative_order,
        )
        higher_derivatives.append(weights @ acceleration_delta)
        conditions.append(condition)
    derivatives = np.asarray(
        (
            history.position_mm[knot_index],
            C_MMNS * history.beta[knot_index],
            C_MMNS**2 * history.beta_prime_per_mm[knot_index],
            *higher_derivatives,
        ),
        dtype=np.float64,
    )
    return derivatives, max(conditions), np.arange(start, stop, dtype=np.int64)


def _spin_to_stereographic(
    rest_spin: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    local = np.asarray(rest_spin, dtype=np.float64) @ frame
    denominator = 1.0 + local[:, 2]
    if np.any(denominator <= _CHART_POLE_TOLERANCE):
        raise CausalC5HistoryUnavailableError(
            "accepted spin history reaches the fixed stereographic chart pole"
        )
    return cast(np.ndarray, local[:, :2] / denominator[:, np.newaxis])


def _spin_chart_derivatives_at_knot(
    history: "CausalC5SourceHistory",
    chart: np.ndarray,
    knot_index: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    start = knot_index - _SPIN_HALF_WINDOW
    stop = knot_index + _SPIN_HALF_WINDOW + 1
    if start < 0 or stop > history.sample_count:
        raise CausalC5HistoryUnavailableError(
            "spin derivative window is not fully accepted"
        )
    selected_times = history.time_ns[start:stop]
    offsets = selected_times - float(history.time_ns[knot_index])
    scale = float(np.max(np.abs(offsets)))
    if scale <= 0.0:
        raise ValueError("accepted spin times must span a nonzero interval")
    design = np.vander(
        offsets / scale,
        N=_SPIN_FIT_DEGREE + 1,
        increasing=True,
    )
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        design,
        chart[start:stop],
        rcond=None,
    )
    if rank != _SPIN_FIT_DEGREE + 1:
        raise CausalC5HistoryUnavailableError(
            "accepted spin derivative fit is rank deficient"
        )
    derivatives = np.empty((6, 2), dtype=np.float64)
    derivatives[0] = chart[knot_index]
    for derivative_order in range(1, 6):
        derivatives[derivative_order] = (
            math.factorial(derivative_order)
            * coefficients[derivative_order]
            / scale**derivative_order
        )
    return derivatives, float(np.linalg.cond(design)), np.arange(start, stop)


@dataclass(frozen=True)
class FrozenC5SourceSegment:
    """One immutable accepted source segment ready for retarded queries."""

    left_knot_index: int
    start_time_ns: float
    duration_ns: float
    position_coefficients_mm: np.ndarray
    rest_spin_stereographic_coefficients: np.ndarray
    stereographic_frame: np.ndarray
    position_condition_number: float
    spin_condition_number: float
    position_window_indices: np.ndarray
    spin_window_indices: np.ndarray

    def __post_init__(self) -> None:
        left = int(self.left_knot_index)
        start = float(self.start_time_ns)
        duration = float(self.duration_ns)
        conditions = (
            float(self.position_condition_number),
            float(self.spin_condition_number),
        )
        if left < 0 or not np.isfinite(start):
            raise ValueError("frozen segment index and start time are invalid")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("frozen segment duration must be finite and positive")
        if any(not np.isfinite(value) or value <= 0.0 for value in conditions):
            raise ValueError("frozen segment condition numbers must be positive")
        object.__setattr__(self, "left_knot_index", left)
        object.__setattr__(self, "start_time_ns", start)
        object.__setattr__(self, "duration_ns", duration)
        object.__setattr__(
            self,
            "position_coefficients_mm",
            _readonly_array(
                self.position_coefficients_mm,
                shape=(12, 3),
                name="position_coefficients_mm",
            ),
        )
        object.__setattr__(
            self,
            "rest_spin_stereographic_coefficients",
            _readonly_array(
                self.rest_spin_stereographic_coefficients,
                shape=(12, 2),
                name="rest_spin_stereographic_coefficients",
            ),
        )
        object.__setattr__(
            self,
            "stereographic_frame",
            _validated_frame(self.stereographic_frame),
        )
        object.__setattr__(self, "position_condition_number", conditions[0])
        object.__setattr__(self, "spin_condition_number", conditions[1])
        object.__setattr__(
            self,
            "position_window_indices",
            _readonly_index_array(
                self.position_window_indices,
                shape=(2, 7),
                name="position_window_indices",
            ),
        )
        object.__setattr__(
            self,
            "spin_window_indices",
            _readonly_index_array(
                self.spin_window_indices,
                shape=(2, 15),
                name="spin_window_indices",
            ),
        )

    @property
    def end_time_ns(self) -> float:
        return self.start_time_ns + self.duration_ns

    def to_checkpoint_payload(self) -> dict[str, object]:
        return {
            "left_knot_index": self.left_knot_index,
            "start_time_ns": self.start_time_ns,
            "duration_ns": self.duration_ns,
            "position_coefficients_mm": self.position_coefficients_mm.tolist(),
            "rest_spin_stereographic_coefficients": (
                self.rest_spin_stereographic_coefficients.tolist()
            ),
            "stereographic_frame": self.stereographic_frame.tolist(),
            "position_condition_number": self.position_condition_number,
            "spin_condition_number": self.spin_condition_number,
            "position_window_indices": self.position_window_indices.tolist(),
            "spin_window_indices": self.spin_window_indices.tolist(),
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "FrozenC5SourceSegment":
        required = {
            "left_knot_index",
            "start_time_ns",
            "duration_ns",
            "position_coefficients_mm",
            "rest_spin_stereographic_coefficients",
            "stereographic_frame",
            "position_condition_number",
            "spin_condition_number",
            "position_window_indices",
            "spin_window_indices",
        }
        if set(payload) != required:
            raise ValueError("frozen C5 source segment checkpoint keys do not match")
        return cls(
            left_knot_index=int(cast(int, payload["left_knot_index"])),
            start_time_ns=float(cast(float, payload["start_time_ns"])),
            duration_ns=float(cast(float, payload["duration_ns"])),
            position_coefficients_mm=np.asarray(
                payload["position_coefficients_mm"], dtype=np.float64
            ),
            rest_spin_stereographic_coefficients=np.asarray(
                payload["rest_spin_stereographic_coefficients"], dtype=np.float64
            ),
            stereographic_frame=np.asarray(
                payload["stereographic_frame"], dtype=np.float64
            ),
            position_condition_number=float(
                cast(float, payload["position_condition_number"])
            ),
            spin_condition_number=float(cast(float, payload["spin_condition_number"])),
            position_window_indices=np.asarray(
                payload["position_window_indices"], dtype=np.int64
            ),
            spin_window_indices=np.asarray(
                payload["spin_window_indices"], dtype=np.int64
            ),
        )


@dataclass(frozen=True)
class CausalC5SourceHistory:
    """Accepted samples and every causally ready frozen $C^5$ segment."""

    time_ns: np.ndarray
    position_mm: np.ndarray
    beta: np.ndarray
    beta_prime_per_mm: np.ndarray
    rest_spin: np.ndarray
    stereographic_frame: np.ndarray
    frozen_segments: tuple[FrozenC5SourceSegment, ...] = ()

    def __post_init__(self) -> None:
        times = _readonly_array(self.time_ns, shape=None, name="time_ns")
        if times.ndim != 1 or np.any(np.diff(times) <= 0.0):
            raise ValueError("accepted source times must increase strictly")
        count = int(times.size)
        object.__setattr__(self, "time_ns", times)
        for name in ("position_mm", "beta", "beta_prime_per_mm", "rest_spin"):
            object.__setattr__(
                self,
                name,
                _readonly_array(
                    getattr(self, name),
                    shape=(count, 3),
                    name=name,
                ),
            )
        if count and np.any(np.sum(self.beta * self.beta, axis=1) >= 1.0):
            raise ValueError("accepted source beta magnitude must be below one")
        if count and not np.allclose(
            np.linalg.norm(self.rest_spin, axis=1),
            1.0,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError("accepted source rest spin must have unit magnitude")
        object.__setattr__(
            self,
            "stereographic_frame",
            _validated_frame(self.stereographic_frame),
        )
        segments = tuple(self.frozen_segments)
        if segments and segments[0].left_knot_index != _SPIN_HALF_WINDOW:
            raise ValueError("frozen C5 source history starts at the wrong knot")
        for sequence_index, segment in enumerate(segments):
            if sequence_index and (
                segment.left_knot_index
                != segments[sequence_index - 1].left_knot_index + 1
            ):
                raise ValueError("frozen C5 source segments must be contiguous")
            left = segment.left_knot_index
            if left + 1 >= count:
                raise ValueError("frozen C5 segment exceeds accepted samples")
            if segment.start_time_ns != times[left] or segment.duration_ns != (
                times[left + 1] - times[left]
            ):
                raise ValueError("frozen C5 segment times do not match accepted knots")
            if not np.array_equal(
                segment.stereographic_frame,
                self.stereographic_frame,
            ):
                raise ValueError("frozen C5 segment changed stereographic frame")
            expected_position_windows = np.stack(
                (
                    np.arange(left - _POSITION_HALF_WINDOW, left + 4),
                    np.arange(left + 1 - _POSITION_HALF_WINDOW, left + 5),
                )
            )
            expected_spin_windows = np.stack(
                (
                    np.arange(left - _SPIN_HALF_WINDOW, left + 8),
                    np.arange(left + 1 - _SPIN_HALF_WINDOW, left + 9),
                )
            )
            if not np.array_equal(
                segment.position_window_indices,
                expected_position_windows,
            ) or not np.array_equal(
                segment.spin_window_indices,
                expected_spin_windows,
            ):
                raise ValueError("frozen C5 segment window indices are inconsistent")
        object.__setattr__(self, "frozen_segments", segments)

    @classmethod
    def empty(
        cls,
        *,
        stereographic_frame: Sequence[Sequence[float]] | np.ndarray = np.eye(3),
    ) -> "CausalC5SourceHistory":
        return cls(
            time_ns=np.zeros(0),
            position_mm=np.zeros((0, 3)),
            beta=np.zeros((0, 3)),
            beta_prime_per_mm=np.zeros((0, 3)),
            rest_spin=np.zeros((0, 3)),
            stereographic_frame=np.asarray(stereographic_frame, dtype=np.float64),
        )

    @property
    def sample_count(self) -> int:
        return int(self.time_ns.size)

    @property
    def readiness_left_knot(self) -> int | None:
        if not self.frozen_segments:
            return None
        return self.frozen_segments[-1].left_knot_index

    def _build_segment(self, left_knot_index: int) -> FrozenC5SourceSegment:
        left = int(left_knot_index)
        right = left + 1
        chart = _spin_to_stereographic(self.rest_spin, self.stereographic_frame)
        start_position, start_position_condition, start_position_window = (
            _position_derivatives_at_knot(self, left)
        )
        end_position, end_position_condition, end_position_window = (
            _position_derivatives_at_knot(self, right)
        )
        start_spin, start_spin_condition, start_spin_window = (
            _spin_chart_derivatives_at_knot(self, chart, left)
        )
        end_spin, end_spin_condition, end_spin_window = _spin_chart_derivatives_at_knot(
            self, chart, right
        )
        maximum_position_condition = max(
            start_position_condition,
            end_position_condition,
        )
        maximum_spin_condition = max(start_spin_condition, end_spin_condition)
        if maximum_position_condition > _MAXIMUM_CONDITION_NUMBER:
            raise CausalC5HistoryUnavailableError(
                "accepted position derivative fit exceeds the condition-number limit"
            )
        if maximum_spin_condition > _MAXIMUM_CONDITION_NUMBER:
            raise CausalC5HistoryUnavailableError(
                "accepted spin derivative fit exceeds the condition-number limit"
            )
        duration = float(self.time_ns[right] - self.time_ns[left])
        return FrozenC5SourceSegment(
            left_knot_index=left,
            start_time_ns=float(self.time_ns[left]),
            duration_ns=duration,
            position_coefficients_mm=_hermite_endpoint_coefficients(
                duration_ns=duration,
                start_derivatives=start_position,
                end_derivatives=end_position,
            ),
            rest_spin_stereographic_coefficients=_hermite_endpoint_coefficients(
                duration_ns=duration,
                start_derivatives=start_spin,
                end_derivatives=end_spin,
            ),
            stereographic_frame=self.stereographic_frame,
            position_condition_number=maximum_position_condition,
            spin_condition_number=maximum_spin_condition,
            position_window_indices=np.stack(
                (start_position_window, end_position_window)
            ),
            spin_window_indices=np.stack((start_spin_window, end_spin_window)),
        )

    def append_accepted(
        self,
        *,
        time_ns: float,
        position_mm: Sequence[float],
        beta: Sequence[float],
        beta_prime_per_mm: Sequence[float],
        rest_spin: Sequence[float],
    ) -> "CausalC5SourceHistory":
        """Return a candidate accepted state without mutating this history."""

        time = float(time_ns)
        if not np.isfinite(time) or (
            self.sample_count and time <= float(self.time_ns[-1])
        ):
            raise ValueError("accepted source time must be finite and increasing")
        vectors = tuple(
            _readonly_array(value, shape=(3,), name=name)
            for value, name in (
                (position_mm, "position_mm"),
                (beta, "beta"),
                (beta_prime_per_mm, "beta_prime_per_mm"),
                (rest_spin, "rest_spin"),
            )
        )
        candidate = CausalC5SourceHistory(
            time_ns=np.concatenate((self.time_ns, np.asarray((time,)))),
            position_mm=np.vstack((self.position_mm, vectors[0])),
            beta=np.vstack((self.beta, vectors[1])),
            beta_prime_per_mm=np.vstack((self.beta_prime_per_mm, vectors[2])),
            rest_spin=np.vstack((self.rest_spin, vectors[3])),
            stereographic_frame=self.stereographic_frame,
            frozen_segments=self.frozen_segments,
        )
        segments = list(candidate.frozen_segments)
        next_left = (
            _SPIN_HALF_WINDOW if not segments else segments[-1].left_knot_index + 1
        )
        while next_left + 1 + _SPIN_HALF_WINDOW < candidate.sample_count:
            segments.append(candidate._build_segment(next_left))
            next_left += 1
        if len(segments) == len(candidate.frozen_segments):
            return candidate
        return CausalC5SourceHistory(
            time_ns=candidate.time_ns,
            position_mm=candidate.position_mm,
            beta=candidate.beta,
            beta_prime_per_mm=candidate.beta_prime_per_mm,
            rest_spin=candidate.rest_spin,
            stereographic_frame=candidate.stereographic_frame,
            frozen_segments=tuple(segments),
        )

    def segment_at(self, source_time_ns: float) -> FrozenC5SourceSegment:
        time = float(source_time_ns)
        if not np.isfinite(time):
            raise ValueError("source_time_ns must be finite")
        if self.frozen_segments:
            starts = np.fromiter(
                (segment.start_time_ns for segment in self.frozen_segments),
                dtype=np.float64,
            )
            index = int(np.searchsorted(starts, time, side="right") - 1)
            if index >= 0:
                segment = self.frozen_segments[index]
                if time <= segment.end_time_ns:
                    return segment
        raise CausalC5HistoryUnavailableError(
            "no causally frozen C5 source segment covers the requested time"
        )

    def to_checkpoint_payload(self) -> dict[str, object]:
        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "time_ns": self.time_ns.tolist(),
            "position_mm": self.position_mm.tolist(),
            "beta": self.beta.tolist(),
            "beta_prime_per_mm": self.beta_prime_per_mm.tolist(),
            "rest_spin": self.rest_spin.tolist(),
            "stereographic_frame": self.stereographic_frame.tolist(),
            "frozen_segments": [
                segment.to_checkpoint_payload() for segment in self.frozen_segments
            ],
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "CausalC5SourceHistory":
        required = {
            "schema_version",
            "time_ns",
            "position_mm",
            "beta",
            "beta_prime_per_mm",
            "rest_spin",
            "stereographic_frame",
            "frozen_segments",
        }
        if set(payload) != required:
            raise ValueError("causal C5 source-history checkpoint keys do not match")
        if int(cast(int, payload["schema_version"])) != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported causal C5 source-history checkpoint schema")
        raw_segments = payload["frozen_segments"]
        if not isinstance(raw_segments, list) or any(
            not isinstance(segment, Mapping) for segment in raw_segments
        ):
            raise ValueError("frozen C5 source segments must be JSON objects")
        return cls(
            time_ns=np.asarray(payload["time_ns"], dtype=np.float64),
            position_mm=np.asarray(payload["position_mm"], dtype=np.float64),
            beta=np.asarray(payload["beta"], dtype=np.float64),
            beta_prime_per_mm=np.asarray(
                payload["beta_prime_per_mm"], dtype=np.float64
            ),
            rest_spin=np.asarray(payload["rest_spin"], dtype=np.float64),
            stereographic_frame=np.asarray(
                payload["stereographic_frame"], dtype=np.float64
            ),
            frozen_segments=tuple(
                FrozenC5SourceSegment.from_checkpoint_payload(segment)
                for segment in raw_segments
            ),
        )


__all__ = [
    "CausalC5HistoryUnavailableError",
    "CausalC5SourceHistory",
    "FrozenC5SourceSegment",
]
