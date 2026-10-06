"""CPU node batches, with strict resolved roots and original source sums.

The lab-coordinate binary search supplies hints only. The maintained provider
checks them in its resolved endpoint geometry and falls back when ambiguous.
No resolved history is flattened into a legacy position polynomial.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np

from .. import retarded_fields as rf
from ..constants import C_MMNS
from ..resolved_charge_batch import resolved_charge_jets_strict_serial


@dataclass
class NodeWarmStart:
    """Proposals from completed evaluations on an accepted source history.

    Roots predict the next segment only. The authoritative solver keeps its
    original initial iterate, stopping rule, and arithmetic for bit parity.
    """

    source_keys: tuple[int, ...] = ()
    times: np.ndarray | None = None
    segments: np.ndarray | None = None
    roots: np.ndarray | None = None
    positions: np.ndarray | None = None
    rays: np.ndarray | None = None
    inverse_kappa: np.ndarray | None = None
    reused: np.ndarray | None = None
    proposals: int = 0
    hits: int = 0
    inertial_events: int = 0
    batch_fallbacks: int = 0

    def propose(
        self,
        sources: Mapping[int, rf._PreparedSourceHistory],
        times: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray | None:
        shape = (len(times), len(sources))
        if (
            self.source_keys != tuple(sources)
            or self.segments is None
            or self.segments.shape != shape
            or self.times is None
            or np.any(times < self.times)
        ):
            return None
        hints = self.segments.copy()
        reused = np.zeros(shape, dtype=bool)
        assert self.roots is not None
        assert self.positions is not None
        assert self.rays is not None
        assert self.inverse_kappa is not None
        delta_t = times - self.times
        delta_x = (positions - self.positions) / C_MMNS
        for column, source in enumerate(sources.values()):
            # -1 denotes an analytic inertial root. It still predicts the
            # first resolved segment when the ray enters accepted history.
            valid = hints[:, column] >= -1
            # Linearize the previous light cone, including lattice motion.
            # This predicts a segment only; it never seeds the Newton solver.
            shift = (
                delta_t[valid]
                - np.sum(self.rays[valid, column] * delta_x[valid], axis=1)
            ) * self.inverse_kappa[valid, column]
            predicted = self.roots[valid, column] + shift
            hints[valid, column] = np.clip(
                np.searchsorted(source.time_ns, predicted, side="right") - 1,
                0,
                len(source.time_ns) - 2,
            )
            indices = hints[valid, column]
            local_times = times[valid]
            local_positions = positions[valid]
            lower = C_MMNS * (local_times - source.time_ns[indices]) - np.linalg.norm(
                local_positions - source.position_mm[indices], axis=1
            )
            upper = C_MMNS * (
                local_times - source.time_ns[indices + 1]
            ) - np.linalg.norm(
                local_positions - source.position_mm[indices + 1], axis=1
            )
            usable = (lower >= 0) & (upper <= 0)
            reused[valid, column] = usable
            rejected = np.flatnonzero(valid)[~usable]
            if len(rejected):
                # Rounded lab endpoints reject proposals only. They never
                # certify a segment: every final hint still reaches the same
                # resolved endpoint oracle, including the batched fallback.
                hints[rejected, column] = node_segment_hints(
                    {column: source}, times[rejected], positions[rejected]
                )[:, 0]
        self.reused = reused
        return hints


def node_segment_hints(
    sources: Mapping[int, rf._PreparedSourceHistory],
    times: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Vectorized float64 proposals; not authoritative brackets or roots."""
    hints = np.full((len(times), len(sources)), -2, dtype=np.int64)
    for column, source in enumerate(sources.values()):
        count = len(source.time_ns)
        if count < 2:
            continue
        lower = np.zeros(len(times), dtype=np.int64)
        upper = np.full(len(times), count - 1, dtype=np.int64)

        def residual(indices: np.ndarray) -> np.ndarray:
            return cast(
                np.ndarray,
                C_MMNS * (times - source.time_ns[indices])
                - np.linalg.norm(positions - source.position_mm[indices], axis=1),
            )

        bracketed = (residual(lower) >= 0) & (residual(upper) <= 0)
        while np.any(upper - lower > 1):
            middle = lower + (upper - lower) // 2
            positive = residual(middle) >= 0
            active = upper - lower > 1
            lower = np.where(active & positive, middle, lower)
            upper = np.where(active & ~positive, middle, upper)
        hints[bracketed, column] = lower[bracketed]
    return hints


def resolved_node_fields(
    prepared: rf._PreparedHistory,
    times: np.ndarray,
    positions: np.ndarray,
    *,
    hints: np.ndarray | None = None,
    warm_start: NodeWarmStart | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return E/B only, preserving the provider's strict jet zeroth values.

    Blocks bound the temporary jet arrays. All roots, inertial certificates,
    fields, and faithfully rounded source reductions remain on the CPU.
    """
    warm = False
    if hints is None and warm_start is not None:
        hints = warm_start.propose(prepared.sources, times, positions)
        warm = hints is not None
    if hints is None:
        hints = node_segment_hints(prepared.sources, times, positions)
    if np.shape(hints) != (len(times), len(prepared.sources)):
        raise ValueError("node segment hints must have shape [events, sources]")
    electric = np.zeros((len(times), 3))
    magnetic = np.zeros_like(electric)
    sources = tuple(prepared.sources.items())
    segments = (
        np.full(hints.shape, -2, dtype=np.int64) if warm_start is not None else None
    )
    roots = np.full(hints.shape, np.nan) if warm_start is not None else None
    rays = np.zeros((*hints.shape, 3)) if warm_start is not None else None
    inverse_kappa = np.zeros(hints.shape) if warm_start is not None else None
    proposals = hits = inertial = batch_fallbacks = 0
    for start in range(0, len(times), 16):
        stop = min(len(times), start + 16)
        count = (stop - start) * len(sources)
        charges = np.zeros(count)
        vectors = np.zeros((6, count, 3))
        frames = np.zeros(count, dtype=bool)
        snaps = np.zeros(count, dtype=bool)
        ballistic_frames = np.zeros((count, 4))
        ballistic_mask = np.zeros(count, dtype=bool)
        for event in range(start, stop):
            for column, (index, source) in enumerate(sources):
                sample = rf._solve_retarded_sample(
                    source,
                    observer_time_ns=float(times[event]),
                    observer_position_mm=positions[event],
                    root_tolerance_mm=rf._DEFAULT_ROOT_TOLERANCE_MM,
                    max_root_iterations=rf._DEFAULT_MAX_ROOT_ITERATIONS,
                    compiled=True,
                    _defer_ballistic_frame=True,
                    segment_hint=int(hints[event, column]),
                )
                if sample is None:
                    raise rf.RetardedHistoryError(
                        "source history does not bracket the observer light cone"
                    )
                if sample.source_proper_velocity is None:
                    raise ValueError("node batches require resolved source histories")
                if warm_start is not None:
                    assert segments is not None and roots is not None
                    roots[event, column] = sample.time_ns
                    if sample.source_segment_index is not None:
                        segments[event, column] = sample.source_segment_index
                        if warm:
                            proposals += 1
                            assert warm_start.reused is not None
                            reused = warm_start.reused[event, column]
                            batch_fallbacks += int(not reused)
                            hits += int(
                                reused
                                and hints[event, column] >= 0
                                and sample.source_segment_index == hints[event, column]
                                and source._metal_timelike_proof
                                and source._metal_timelike_count == len(source.time_ns)
                            )
                    else:
                        segments[event, column] = -1
                        inertial += 1
                assert sample.precise_separation_mm is not None
                row = (event - start) * len(sources) + column
                charges[row] = prepared.arrays.charge_native[index]
                vectors[0, row] = sample.precise_separation_mm
                vectors[1, row] = sample.source_proper_velocity
                vectors[2, row] = sample.beta_prime_per_mm
                if sample.beta_jerk_per_mm2 is not None:
                    vectors[3, row] = sample.beta_jerk_per_mm2
                frame = sample.precise_separation_frame_mm
                if sample.ballistic_frame_components is not None:
                    ballistic_frames[row] = sample.ballistic_frame_components
                    ballistic_mask[row] = True
                if frame is not None:
                    vectors[4, row] = frame
                    frames[row] = True
                if sample.beta_snap_per_mm3 is not None:
                    vectors[5, row] = sample.beta_snap_per_mm3
                    snaps[row] = True
        if warm_start is not None:
            ray = vectors[0] / np.linalg.norm(vectors[0], axis=1)[:, None]
            beta = (
                vectors[1] / np.hypot(1.0, np.linalg.norm(vectors[1], axis=1))[:, None]
            )
            assert rays is not None and inverse_kappa is not None
            rays[start:stop] = ray.reshape(stop - start, len(sources), 3)
            kappa = 1 - np.sum(ray * beta, axis=1)
            inverse_kappa[start:stop] = np.divide(
                1.0, kappa, out=np.zeros_like(kappa), where=kappa > 0
            ).reshape(stop - start, len(sources))
        _, e, b, _ = resolved_charge_jets_strict_serial(
            charges, *vectors, frames, snaps, ballistic_frames, ballistic_mask
        )
        e = e[:, :, 0].reshape(stop - start, len(sources), 3)
        b = b[:, :, 0].reshape(stop - start, len(sources), 3)
        for row, event in enumerate(range(start, stop)):
            electric[event] = rf._sum_source_terms(e[row])
            magnetic[event] = rf._sum_source_terms(b[row])
    if warm_start is not None:
        # Publish only after every root and field succeeded; partial failures
        # must not replace the previous accepted proposals.
        warm_start.source_keys = tuple(prepared.sources)
        warm_start.times = times.copy()
        warm_start.segments = segments
        warm_start.roots = roots
        warm_start.positions = positions.copy()
        warm_start.rays = rays
        warm_start.inverse_kappa = inverse_kappa
        warm_start.proposals += proposals
        warm_start.hits += hits
        warm_start.inertial_events += inertial
        warm_start.batch_fallbacks += batch_fallbacks
    return electric, magnetic


def correction_node_fields(
    history: rf.TrajectoryHistory | rf.ExactCloudHistory,
    time_ns: float,
    positions_mm: np.ndarray,
    *,
    warm_start: NodeWarmStart | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = rf._prepare_history(history, ())
    times = np.full(len(positions_mm), time_ns)
    return resolved_node_fields(prepared, times, positions_mm, warm_start=warm_start)
