"""Growable transactional storage for accepted causal $C^5$ source history.

The immutable :mod:`core.causal_c5_source_history` object is the reference
model.  It deliberately copies its arrays, which makes ownership obvious but
is quadratic when used for every accepted step.  This module preserves the
same frozen-segment algebra while separating tentative storage from the
published boundary:

* preflight writes only beyond the visible sample count;
* every newly ready segment is constructed before publication;
* rejection leaves prior published views unchanged; and
* commit advances the visible sample and segment counts without copying the
  accepted prefix.

Concurrent mutation and evaluation of the same builder is unsupported.  A
published view is read-only and remains a stable prefix after later commits.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import itertools
from typing import overload

import numpy as np

from .causal_c5_source_history import (
    CausalC5RetardedRoot,
    CausalC5SourceHistory,
    FrozenC5SourceSegment,
)

_SPIN_HALF_WINDOW = 7
_TOKEN_COUNTER = itertools.count(1)


def _readonly_prefix(values: np.ndarray, stop: int) -> np.ndarray:
    result = values[:stop].view()
    result.setflags(write=False)
    return result


class _SegmentSequence(Sequence[FrozenC5SourceSegment]):
    """Read-only prefix plus an optional detached candidate tail."""

    def __init__(
        self,
        accepted: list[FrozenC5SourceSegment],
        accepted_stop: int,
        tail: Sequence[FrozenC5SourceSegment] = (),
    ) -> None:
        self._accepted = accepted
        self._accepted_stop = int(accepted_stop)
        self._tail = tuple(tail)

    def __len__(self) -> int:
        return self._accepted_stop + len(self._tail)

    def _item(self, index: int) -> FrozenC5SourceSegment:
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("frozen segment index is out of range")
        if normalized < self._accepted_stop:
            return self._accepted[normalized]
        return self._tail[normalized - self._accepted_stop]

    @overload
    def __getitem__(self, index: int) -> FrozenC5SourceSegment: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FrozenC5SourceSegment, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> FrozenC5SourceSegment | tuple[FrozenC5SourceSegment, ...]:
        if isinstance(index, slice):
            return tuple(self._item(item) for item in range(*index.indices(len(self))))
        return self._item(index)

    def __iter__(self) -> Iterator[FrozenC5SourceSegment]:
        for index in range(len(self)):
            yield self._item(index)


class CausalC5PublishedHistory:
    """One immutable visible prefix of a growable history."""

    def __init__(
        self,
        owner: "GrowableCausalC5SourceHistory",
        *,
        sample_stop: int,
        accepted_segment_stop: int,
        candidate_segments: Sequence[FrozenC5SourceSegment] = (),
    ) -> None:
        self._owner = owner
        self._sample_stop = int(sample_stop)
        self._segments = _SegmentSequence(
            owner._segments,
            accepted_segment_stop,
            candidate_segments,
        )

    @property
    def time_ns(self) -> np.ndarray:
        return _readonly_prefix(self._owner._time_ns, self._sample_stop)

    @property
    def position_mm(self) -> np.ndarray:
        return _readonly_prefix(self._owner._position_mm, self._sample_stop)

    @property
    def beta(self) -> np.ndarray:
        return _readonly_prefix(self._owner._beta, self._sample_stop)

    @property
    def beta_prime_per_mm(self) -> np.ndarray:
        return _readonly_prefix(self._owner._beta_prime_per_mm, self._sample_stop)

    @property
    def step_start_beta_prime_per_mm(self) -> np.ndarray:
        return _readonly_prefix(
            self._owner._step_start_beta_prime_per_mm,
            self._sample_stop,
        )

    @property
    def step_start_beta_prime_ready(self) -> np.ndarray:
        return _readonly_prefix(
            self._owner._step_start_beta_prime_ready,
            self._sample_stop,
        )

    @property
    def rest_spin(self) -> np.ndarray:
        return _readonly_prefix(self._owner._rest_spin, self._sample_stop)

    @property
    def stereographic_frame(self) -> np.ndarray:
        return self._owner.stereographic_frame

    @property
    def frozen_segments(self) -> Sequence[FrozenC5SourceSegment]:
        return self._segments

    @property
    def sample_count(self) -> int:
        return self._sample_stop

    @property
    def readiness_left_knot(self) -> int | None:
        if not self.frozen_segments:
            return None
        return self.frozen_segments[-1].left_knot_index

    def segment_at(self, source_time_ns: float) -> FrozenC5SourceSegment:
        return CausalC5SourceHistory.segment_at(self, source_time_ns)  # type: ignore[arg-type]

    def solve_retarded_root(
        self,
        *,
        observer_time_ns: float,
        observer_position_mm: Sequence[float],
        root_tolerance_mm: float = 1.0e-21,
        max_root_iterations: int = 96,
        minimum_separation_mm: float = 1.0e-15,
    ) -> CausalC5RetardedRoot:
        return CausalC5SourceHistory.solve_retarded_root(  # type: ignore[arg-type]
            self,
            observer_time_ns=observer_time_ns,
            observer_position_mm=observer_position_mm,
            root_tolerance_mm=root_tolerance_mm,
            max_root_iterations=max_root_iterations,
            minimum_separation_mm=minimum_separation_mm,
        )


@dataclass(frozen=True)
class CausalC5AppendTransaction:
    """A fully preflighted append that has not crossed the visible boundary."""

    candidate: CausalC5PublishedHistory
    _owner_token: int
    _generation: int
    _serial: int
    _start: int
    _stop: int
    _new_segments: tuple[FrozenC5SourceSegment, ...]


class GrowableCausalC5SourceHistory:
    """Geometrically growing accepted samples with explicit append commit."""

    def __init__(
        self,
        *,
        initial_capacity: int = 16,
        stereographic_frame: Sequence[Sequence[float]] | np.ndarray = np.eye(3),
    ) -> None:
        capacity = int(initial_capacity)
        if capacity < 1:
            raise ValueError("initial_capacity must be positive")
        reference = CausalC5SourceHistory.empty(stereographic_frame=stereographic_frame)
        self.stereographic_frame = reference.stereographic_frame
        self._time_ns = np.empty(capacity, dtype=np.float64)
        self._position_mm = np.empty((capacity, 3), dtype=np.float64)
        self._beta = np.empty((capacity, 3), dtype=np.float64)
        self._beta_prime_per_mm = np.empty((capacity, 3), dtype=np.float64)
        self._step_start_beta_prime_per_mm = np.empty((capacity, 3), dtype=np.float64)
        self._step_start_beta_prime_ready = np.zeros(capacity, dtype=bool)
        self._rest_spin = np.empty((capacity, 3), dtype=np.float64)
        self._sample_count = 0
        self._segments: list[FrozenC5SourceSegment] = []
        self._token = next(_TOKEN_COUNTER)
        self._generation = 0
        self._candidate_serial = 0

    @classmethod
    def from_history(
        cls,
        history: CausalC5SourceHistory | CausalC5PublishedHistory,
        *,
        minimum_capacity: int = 16,
    ) -> "GrowableCausalC5SourceHistory":
        count = int(history.sample_count)
        capacity = max(int(minimum_capacity), 1)
        while capacity < count:
            capacity *= 2
        result = cls(
            initial_capacity=capacity,
            stereographic_frame=history.stereographic_frame,
        )
        result._time_ns[:count] = history.time_ns
        result._position_mm[:count] = history.position_mm
        result._beta[:count] = history.beta
        result._beta_prime_per_mm[:count] = history.beta_prime_per_mm
        result._step_start_beta_prime_per_mm[:count] = (
            history.step_start_beta_prime_per_mm
        )
        result._step_start_beta_prime_ready[:count] = (
            history.step_start_beta_prime_ready
        )
        result._rest_spin[:count] = history.rest_spin
        result._sample_count = count
        result._segments.extend(history.frozen_segments)
        return result

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def allocated_capacity(self) -> int:
        return int(self._time_ns.size)

    @property
    def frozen_segment_count(self) -> int:
        return len(self._segments)

    def build_current(self) -> CausalC5PublishedHistory:
        return CausalC5PublishedHistory(
            self,
            sample_stop=self._sample_count,
            accepted_segment_stop=len(self._segments),
        )

    def _ensure_capacity(self, required: int) -> None:
        if required <= self.allocated_capacity:
            return
        capacity = self.allocated_capacity
        while capacity < required:
            capacity *= 2
        replacements = (
            ("_time_ns", (capacity,)),
            ("_position_mm", (capacity, 3)),
            ("_beta", (capacity, 3)),
            ("_beta_prime_per_mm", (capacity, 3)),
            ("_step_start_beta_prime_per_mm", (capacity, 3)),
            ("_step_start_beta_prime_ready", (capacity,)),
            ("_rest_spin", (capacity, 3)),
        )
        for name, shape in replacements:
            old = getattr(self, name)
            new = np.empty(shape, dtype=old.dtype)
            new[: self._sample_count] = old[: self._sample_count]
            setattr(self, name, new)

    @staticmethod
    def _finite_matrix(
        values: Sequence[Sequence[float]] | np.ndarray,
        *,
        rows: int,
        name: str,
    ) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (rows, 3) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite array with shape {(rows, 3)}")
        return np.array(result, dtype=np.float64, copy=True)

    def preflight_append_many(
        self,
        *,
        time_ns: Sequence[float] | np.ndarray,
        position_mm: Sequence[Sequence[float]] | np.ndarray,
        beta: Sequence[Sequence[float]] | np.ndarray,
        beta_prime_per_mm: Sequence[Sequence[float]] | np.ndarray,
        rest_spin: Sequence[Sequence[float]] | np.ndarray,
        step_start_beta_prime_per_mm: (
            Sequence[Sequence[float]] | np.ndarray | None
        ) = None,
        step_start_beta_prime_ready: Sequence[bool] | np.ndarray | None = None,
    ) -> CausalC5AppendTransaction:
        """Build a detached candidate without advancing the visible prefix."""

        times = np.asarray(time_ns, dtype=np.float64)
        if times.ndim != 1 or not times.size or not np.all(np.isfinite(times)):
            raise ValueError("time_ns must be a nonempty finite vector")
        rows = int(times.size)
        if np.any(np.diff(times) <= 0.0) or (
            self._sample_count and times[0] <= self._time_ns[self._sample_count - 1]
        ):
            raise ValueError("accepted source times must increase strictly")
        positions = self._finite_matrix(position_mm, rows=rows, name="position_mm")
        velocities = self._finite_matrix(beta, rows=rows, name="beta")
        accelerations = self._finite_matrix(
            beta_prime_per_mm,
            rows=rows,
            name="beta_prime_per_mm",
        )
        if step_start_beta_prime_per_mm is None:
            step_start_accelerations = np.zeros((rows, 3), dtype=np.float64)
        else:
            step_start_accelerations = self._finite_matrix(
                step_start_beta_prime_per_mm,
                rows=rows,
                name="step_start_beta_prime_per_mm",
            )
        if step_start_beta_prime_ready is None:
            step_start_ready = np.zeros(rows, dtype=bool)
        else:
            step_start_ready = np.asarray(step_start_beta_prime_ready, dtype=bool)
            if step_start_ready.shape != (rows,):
                raise ValueError(
                    f"step_start_beta_prime_ready must have shape {(rows,)}"
                )
            step_start_ready = np.array(step_start_ready, dtype=bool, copy=True)
        spins = self._finite_matrix(rest_spin, rows=rows, name="rest_spin")
        if np.any(np.sum(velocities * velocities, axis=1) >= 1.0):
            raise ValueError("accepted source beta magnitude must be below one")
        if not np.allclose(
            np.linalg.norm(spins, axis=1),
            1.0,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError("accepted source rest spin must have unit magnitude")

        start = self._sample_count
        stop = start + rows
        # From this point onward the unused tail may be overwritten.  Invalidate
        # any older candidate even if the numerical segment preflight below
        # fails before it can return a replacement transaction.
        self._candidate_serial += 1
        self._ensure_capacity(stop)
        self._time_ns[start:stop] = times
        self._position_mm[start:stop] = positions
        self._beta[start:stop] = velocities
        self._beta_prime_per_mm[start:stop] = accelerations
        self._step_start_beta_prime_per_mm[start:stop] = step_start_accelerations
        self._step_start_beta_prime_ready[start:stop] = step_start_ready
        self._rest_spin[start:stop] = spins

        sample_view = CausalC5PublishedHistory(
            self,
            sample_stop=stop,
            accepted_segment_stop=len(self._segments),
        )
        new_segments: list[FrozenC5SourceSegment] = []
        next_left = (
            _SPIN_HALF_WINDOW
            if not self._segments
            else self._segments[-1].left_knot_index + 1
        )
        while next_left + 1 + _SPIN_HALF_WINDOW < stop:
            new_segments.append(
                CausalC5SourceHistory._build_segment(sample_view, next_left)  # type: ignore[arg-type]
            )
            next_left += 1

        candidate = CausalC5PublishedHistory(
            self,
            sample_stop=stop,
            accepted_segment_stop=len(self._segments),
            candidate_segments=new_segments,
        )
        return CausalC5AppendTransaction(
            candidate=candidate,
            _owner_token=self._token,
            _generation=self._generation,
            _serial=self._candidate_serial,
            _start=start,
            _stop=stop,
            _new_segments=tuple(new_segments),
        )

    def can_commit(self, transaction: CausalC5AppendTransaction) -> bool:
        return bool(
            transaction._owner_token == self._token
            and transaction._generation == self._generation
            and transaction._serial == self._candidate_serial
            and transaction._start == self._sample_count
        )

    def commit(
        self, transaction: CausalC5AppendTransaction
    ) -> CausalC5PublishedHistory:
        """Publish a preflighted candidate; no numerical work remains here."""

        if not self.can_commit(transaction):
            raise RuntimeError("causal C5 append transaction is stale or foreign")
        self._segments.extend(transaction._new_segments)
        self._sample_count = transaction._stop
        self._generation += 1
        return self.build_current()


__all__ = [
    "CausalC5AppendTransaction",
    "CausalC5PublishedHistory",
    "GrowableCausalC5SourceHistory",
]
