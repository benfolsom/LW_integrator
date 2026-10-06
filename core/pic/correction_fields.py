"""CPU node batches, with strict resolved roots and original source sums.

The lab-coordinate binary search supplies hints only. The maintained provider
checks them in its resolved endpoint geometry and falls back when ambiguous.
No resolved history is flattened into a legacy position polynomial.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np

from .. import retarded_fields as rf
from ..constants import C_MMNS
from ..resolved_charge_batch import resolved_charge_jets_strict_serial


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
) -> tuple[np.ndarray, np.ndarray]:
    """Return E/B only, preserving the provider's strict jet zeroth values.

    Blocks bound the temporary jet arrays. All roots, inertial certificates,
    fields, and faithfully rounded source reductions remain on the CPU.
    """
    if hints is None:
        hints = node_segment_hints(prepared.sources, times, positions)
    if np.shape(hints) != (len(times), len(prepared.sources)):
        raise ValueError("node segment hints must have shape [events, sources]")
    electric = np.zeros((len(times), 3))
    magnetic = np.zeros_like(electric)
    sources = tuple(prepared.sources.items())
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
        _, e, b, _ = resolved_charge_jets_strict_serial(
            charges, *vectors, frames, snaps, ballistic_frames, ballistic_mask
        )
        e = e[:, :, 0].reshape(stop - start, len(sources), 3)
        b = b[:, :, 0].reshape(stop - start, len(sources), 3)
        for row, event in enumerate(range(start, stop)):
            electric[event] = rf._sum_source_terms(e[row])
            magnetic[event] = rf._sum_source_terms(b[row])
    return electric, magnetic


def correction_node_fields(
    history: rf.TrajectoryHistory | rf.ExactCloudHistory,
    time_ns: float,
    positions_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = rf._prepare_history(history, ())
    times = np.full(len(positions_mm), time_ns)
    return resolved_node_fields(prepared, times, positions_mm)
