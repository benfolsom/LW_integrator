"""Distance and retarded-time utilities for the LW integrator.

The helpers in this module translate particle positions into geometric
quantities (distance, direction cosines, retarded indices) that feed the
covariant equations of motion. Sampled matching uses stored positions to
bracket the observer's light cone; it remains a diagnostic field route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, cast

import numpy as np

from .constants import C_MMNS, NUMERICAL_EPSILON
from .prepared_history_cache import (
    AppendAwarePreparedHistoryCache,
    history_storage_snapshot,
)
from .types import (
    ChronoMatchingMode,
    IndexedTrajectoryArrays,
    ParticleState,
    Trajectory,
    TrajectoryArrays,
)

DistanceResult = Dict[str, np.ndarray]
TrajectoryArraysLike = TrajectoryArrays | IndexedTrajectoryArrays
_MAX_VECTORIZED_CHRONO_MATCH_CELLS = 1_000_000
_MONOTONE_TIME_CACHE: AppendAwarePreparedHistoryCache[
    TrajectoryArraysLike, np.ndarray
] = AppendAwarePreparedHistoryCache(max_entries=16)


def _soa_row(traj: TrajectoryArraysLike, field_name: str, step: int) -> np.ndarray:
    if isinstance(traj, IndexedTrajectoryArrays):
        return traj.row(field_name, step)
    return np.asarray(getattr(traj, field_name))[step]


def _soa_scalar(
    traj: TrajectoryArraysLike,
    field_name: str,
    step: int,
    particle_idx: int,
) -> float:
    if isinstance(traj, IndexedTrajectoryArrays):
        return traj.scalar(field_name, step, particle_idx)
    return float(np.asarray(getattr(traj, field_name))[step, particle_idx])


def _soa_values_at_steps(
    traj: TrajectoryArraysLike,
    field_name: str,
    steps: np.ndarray,
    particle_indices: np.ndarray,
) -> np.ndarray:
    if isinstance(traj, IndexedTrajectoryArrays):
        return traj.values_at_steps(field_name, steps, particle_indices)
    return np.asarray(getattr(traj, field_name))[steps, particle_indices]


def _soa_time_columns(
    traj: TrajectoryArraysLike,
    up_to_step: int,
) -> np.ndarray:
    if isinstance(traj, IndexedTrajectoryArrays):
        return traj.time_columns(up_to_step)
    return np.asarray(traj.t)[: up_to_step + 1, :]


def _monotone_time_columns(traj: TrajectoryArraysLike) -> np.ndarray:
    """Certify finite, nondecreasing times once per accepted append or rewrite.

    Unmanaged histories retain the scan because their mutations are untracked.
    Only the certificate is cached; no growing time array is copied or retained.
    """

    if history_storage_snapshot(traj) is None:
        return np.zeros(traj.n_particles, dtype=bool)

    def check(history: TrajectoryArraysLike, start: int = 0) -> np.ndarray:
        times = _soa_time_columns(history, history.n_steps - 1)[start:]
        return cast(
            np.ndarray,
            np.all(np.isfinite(times), axis=0)
            & np.all(times[1:] >= times[:-1], axis=0),
        )

    return _MONOTONE_TIME_CACHE.prepare(
        traj,
        variant="finite_nondecreasing_times",
        prepare_full=check,
        append=lambda previous, history, old_stop: (
            previous & check(history, max(0, old_stop - 1))
        ),
    ).value


def _batched_time_upper_bound(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """First strictly later row per certified column, or the row count."""
    low = np.zeros(targets.size, dtype=int)
    high = np.full(targets.size, times.shape[0], dtype=int)
    columns = np.arange(targets.size)
    while np.any(low < high):
        active = low < high
        middle = (low + high) // 2
        later = times[np.minimum(middle, times.shape[0] - 1), columns] > targets
        high = np.where(active & later, middle, high)
        low = np.where(active & ~later, middle + 1, low)
    return low


def _soa_constant(traj: TrajectoryArraysLike, field_name: str) -> np.ndarray:
    if isinstance(traj, IndexedTrajectoryArrays):
        return traj.constant(field_name)
    return np.asarray(getattr(traj, field_name))


@dataclass
class ChronoMatchResult:
    """Results from chrono-matching with interpolation support.

    Attributes
    ----------
    indices : np.ndarray
        Primary trajectory indices for each source particle.
    indices_next : np.ndarray
        Next trajectory indices (for interpolation). Equal to indices when no interpolation.
    weights : np.ndarray
        Interpolation weights in [0, 1]. weight=1.0 means use indices only,
        weight=0.0 means use indices_next only. Linear interpolation:
        value = weight * value[indices] + (1-weight) * value[indices_next]
    residuals : np.ndarray
        Time residuals ``|t_matched - t_target|`` for each source particle (in ns).
    max_residual : float
        Maximum residual across all source particles (in ns).
    needs_interpolation : np.ndarray
        Boolean mask indicating which particles needed interpolation.
    use_cubic : bool
        Whether cubic interpolation is used (requires 4 points).
    indices_prev : Optional[np.ndarray]
        Previous trajectory index (for cubic interpolation). None for linear.
    indices_next2 : Optional[np.ndarray]
        Second-next trajectory index (for cubic interpolation). None for linear.
    """

    indices: np.ndarray
    indices_next: np.ndarray
    weights: np.ndarray
    residuals: np.ndarray
    max_residual: float
    needs_interpolation: np.ndarray
    use_cubic: bool = False
    indices_prev: Optional[np.ndarray] = None
    indices_next2: Optional[np.ndarray] = None


def _resolve_external_step_index(
    trajectory_ext: Trajectory,
    observer_index: int,
) -> int:
    if not trajectory_ext:
        raise ValueError("trajectory_ext must contain at least one state")
    return min(max(int(observer_index), 0), len(trajectory_ext) - 1)


def _sampled_light_cone_match(
    source_event: Callable[[int], tuple[float, np.ndarray]],
    last: int,
    observer_time: float,
    observer_position: np.ndarray,
    interpolate: bool,
) -> tuple[float, int] | None:
    """Find the light-cone crossing on a piecewise-linear timelike history.

    Return None if the available history does not bracket a crossing
    (including cold starts), so callers retain their delay fallback. This
    matcher does not construct prehistory or replace the exact field provider.
    """

    def residual(time: float, position: np.ndarray) -> float:
        return C_MMNS * (observer_time - time) - np.linalg.norm(
            observer_position - position
        )

    lo, hi = 0, last
    t_lo, x_lo = source_event(lo)
    t_hi, x_hi = source_event(hi)
    first_residual, last_residual = residual(t_lo, x_lo), residual(t_hi, x_hi)
    if first_residual < 0 or last_residual > 0:
        return None
    if first_residual == 0:
        return t_lo, lo
    if last_residual == 0:
        return t_hi, hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        time, position = source_event(mid)
        value = residual(time, position)
        if value == 0:
            return time, mid
        if value > 0:
            lo, t_lo, x_lo = mid, time, position
        else:
            hi, t_hi, x_hi = mid, time, position
    if not interpolate or t_hi <= t_lo:
        return t_hi, hi
    # Solve inside this one segment; interpolating the residual itself would
    # still give the wrong emission time for transverse source motion.
    lower, upper = t_lo, t_hi
    for _ in range(64):
        mid_time = lower + (upper - lower) / 2
        if mid_time == lower or mid_time == upper:
            break
        weight = (mid_time - t_lo) / (t_hi - t_lo)
        position = x_lo + weight * (x_hi - x_lo)
        value = residual(mid_time, position)
        if value == 0:
            return mid_time, hi
        if value > 0:
            lower = mid_time
        else:
            upper = mid_time
    return lower + (upper - lower) / 2, hi


def _compute_delta_t(
    *,
    mode: ChronoMatchingMode,
    distance: float,
    b_nhat: float,
    sample_index: int,
    index_traj: int,
    index_part: int,
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
) -> float:
    """Resolve the retardation interval for a single particle sample.

    ``FAST`` computes ``delta_t = R * (1 + β·n̂) / (c * (1 - (β·n̂)²))``, which
    equals ``R / (c * (1 - β·n̂))``, with ``n̂ = (x_obs - x_src) / R`` pointing
    from the source sample to the observer and ``β`` the source velocity at that
    sample. The factored form is used for numerical stability; the denominator
    is clamped to ``±1e-12``. This is the exact uniform-motion delay only when
    ``β`` is parallel to ``n̂``. It supplies only a fallback when the stored
    history does not bracket the light cone; otherwise the matcher solves
    against the stored source positions.

    ``AVERAGED`` uses the mean line-of-sight velocity from history samples
    selected at delays ``R / c`` and ``2R / c``. Neither fallback solves the
    light-cone equation for general source motion.
    """

    if mode is ChronoMatchingMode.FAST:
        # Use factored form: delta_t = R * (1+β·n̂) / (c * (1-(β·n̂)²))
        numerator = 1.0 + b_nhat
        denominator = 1.0 - b_nhat**2

        # Clamp denominator to prevent division by zero
        # k_threshold = 1e-12 supports particles up to γ ≈ 7×10⁵
        k_threshold = 1e-12
        if abs(denominator) < k_threshold:
            denominator = np.copysign(k_threshold, denominator)

        return distance * numerator / (C_MMNS * denominator)

    current_source_index = _resolve_external_step_index(trajectory_ext, index_traj)
    time_offsets = np.array([distance / C_MMNS, 2.0 * distance / C_MMNS], dtype=float)
    sampled_b = 0.0

    for offset in time_offsets:
        target_time = trajectory_ext[current_source_index]["t"][sample_index] - offset
        matched_index = _locate_retarded_index(
            trajectory_ext,
            current_source_index,
            sample_index,
            target_time,
        )
        nhat_offset = compute_instantaneous_distance(
            trajectory[index_traj], trajectory_ext[matched_index], index_part
        )
        sampled_b += _dot_beta_nhat(
            trajectory_ext[matched_index], nhat_offset, sample_index
        )

    averaged_b = sampled_b / time_offsets.size

    # Use factored form: delta_t = R * (1+β·n̂) / (c * (1-(β·n̂)²))
    numerator = 1.0 + averaged_b
    denominator = 1.0 - averaged_b**2

    # Clamp denominator to prevent division by zero
    k_threshold = 1e-12
    if abs(denominator) < k_threshold:
        denominator = np.copysign(k_threshold, denominator)

    return float(distance * numerator / (C_MMNS * denominator))


def _dot_beta_nhat(
    state: ParticleState, nhat: DistanceResult, sample_index: int
) -> float:
    return float(
        state["bx"][sample_index] * nhat["nx"][sample_index]
        + state["by"][sample_index] * nhat["ny"][sample_index]
        + state["bz"][sample_index] * nhat["nz"][sample_index]
    )


def _locate_retarded_index_soa(
    t_col: np.ndarray,
    index_traj: int,
    target_time: float,
    *,
    side: str = "right",
    monotone: bool = False,
) -> int:
    """SOA fast path on a pre-sliced time column.

    Certified finite, nondecreasing columns use binary search. Other columns
    keep the original first-match scan, including nonmonotone histories.
    """
    bounded_index_traj = min(index_traj, len(t_col) - 1)
    if len(t_col) == 0:
        return bounded_index_traj
    earliest_time = float(t_col[0])
    if target_time <= 0.0 and earliest_time >= 0.0:
        return bounded_index_traj
    if target_time <= earliest_time:
        return 0
    if monotone:
        return min(
            int(
                np.searchsorted(
                    t_col[: bounded_index_traj + 1],
                    target_time,
                    side="left" if side == "left" else "right",
                )
            ),
            bounded_index_traj,
        )

    matched_idx = bounded_index_traj
    if side == "left":
        for candidate_idx in range(bounded_index_traj + 1):
            if t_col[candidate_idx] >= target_time:
                matched_idx = candidate_idx
                break
    else:
        for candidate_idx in range(bounded_index_traj + 1):
            if t_col[candidate_idx] > target_time:
                matched_idx = candidate_idx
                break
    return matched_idx


def _locate_retarded_index(
    trajectory_ext: Trajectory,
    index_traj: int,
    sample_index: int,
    target_time: float,
) -> int:
    bounded_index_traj = min(index_traj, len(trajectory_ext) - 1)
    earliest_time = float(trajectory_ext[0]["t"][sample_index])
    if target_time <= 0.0 and earliest_time >= 0.0:
        return bounded_index_traj
    if target_time <= earliest_time:
        return 0

    for candidate_index in range(bounded_index_traj + 1):
        if trajectory_ext[candidate_index]["t"][sample_index] >= target_time:
            return candidate_index
    return bounded_index_traj


def _compute_delta_t_soa(
    *,
    mode: ChronoMatchingMode,
    distance: float,
    b_nhat: float,
    sample_index: int,
    index_traj: int,
    index_part: int,
    traj: TrajectoryArraysLike,
    traj_ext: TrajectoryArraysLike,
) -> float:
    """SOA equivalent of :func:`_compute_delta_t`."""
    if mode is ChronoMatchingMode.FAST:
        numerator = 1.0 + b_nhat
        denominator = 1.0 - b_nhat**2

        k_threshold = 1e-12
        if abs(denominator) < k_threshold:
            denominator = np.copysign(k_threshold, denominator)

        return distance * numerator / (C_MMNS * denominator)

    current_source_index = min(index_traj, traj_ext.n_steps - 1)
    time_offsets = np.array([distance / C_MMNS, 2.0 * distance / C_MMNS], dtype=float)
    sampled_b = 0.0
    t_col = _soa_time_columns(traj_ext, current_source_index)[:, sample_index]

    for offset in time_offsets:
        target_time = (
            _soa_scalar(traj_ext, "t", current_source_index, sample_index) - offset
        )
        matched_index = _locate_retarded_index_soa(
            t_col,
            current_source_index,
            target_time,
            side="left",
        )

        dx = _soa_scalar(traj, "x", index_traj, index_part) - _soa_scalar(
            traj_ext, "x", matched_index, sample_index
        )
        dy = _soa_scalar(traj, "y", index_traj, index_part) - _soa_scalar(
            traj_ext, "y", matched_index, sample_index
        )
        dz = _soa_scalar(traj, "z", index_traj, index_part) - _soa_scalar(
            traj_ext, "z", matched_index, sample_index
        )
        sample_distance = (dx * dx + dy * dy + dz * dz) ** 0.5
        if sample_distance < NUMERICAL_EPSILON:
            nx = ny = nz = 0.0
        else:
            nx = dx / sample_distance
            ny = dy / sample_distance
            nz = dz / sample_distance

        sampled_b += (
            _soa_scalar(traj_ext, "bx", matched_index, sample_index) * nx
            + _soa_scalar(traj_ext, "by", matched_index, sample_index) * ny
            + _soa_scalar(traj_ext, "bz", matched_index, sample_index) * nz
        )

    averaged_b = sampled_b / time_offsets.size
    numerator = 1.0 + averaged_b
    denominator = 1.0 - averaged_b**2

    k_threshold = 1e-12
    if abs(denominator) < k_threshold:
        denominator = np.copysign(k_threshold, denominator)

    return float(distance * numerator / (C_MMNS * denominator))


def compute_instantaneous_distance(
    vector: ParticleState, vector_ext: ParticleState, index: int
) -> DistanceResult:
    """Compute Euclidean distance and direction cosines for a particle pair.

    Parameters
    ----------
    vector:
        Reference particle state (typically the bunch being updated).
    vector_ext:
        External particle state sampled at the same trajectory index.
    index:
        Index of the particle within ``vector`` to evaluate against the entire
        ``vector_ext`` bunch.
    """

    result: DistanceResult = {}

    dx = vector["x"][index] - vector_ext["x"]
    dy = vector["y"][index] - vector_ext["y"]
    dz = vector["z"][index] - vector_ext["z"]
    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    too_close = distance < NUMERICAL_EPSILON
    safe_dist = np.where(too_close, NUMERICAL_EPSILON, distance)

    result["R"] = safe_dist
    result["nx"] = np.where(too_close, 0.0, dx / safe_dist)
    result["ny"] = np.where(too_close, 0.0, dy / safe_dist)
    result["nz"] = np.where(too_close, 0.0, dz / safe_dist)

    return result


def compute_retarded_distance_soa(
    traj: TrajectoryArraysLike,
    traj_ext: TrajectoryArraysLike,
    index_traj: int,
    index_part: int,
    indices_ret: np.ndarray,
) -> DistanceResult:
    """SOA fast path for compute_retarded_distance.

    Avoids per-step dict lookups by accessing 2-D array slices directly.
    """
    indices = np.asarray(indices_ret, dtype=int)
    particle_indices = np.arange(indices.size)

    dx = _soa_scalar(traj, "x", index_traj, index_part) - _soa_values_at_steps(
        traj_ext, "x", indices, particle_indices
    )
    dy = _soa_scalar(traj, "y", index_traj, index_part) - _soa_values_at_steps(
        traj_ext, "y", indices, particle_indices
    )
    dz = _soa_scalar(traj, "z", index_traj, index_part) - _soa_values_at_steps(
        traj_ext, "z", indices, particle_indices
    )
    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    too_close = distance < NUMERICAL_EPSILON
    safe_dist = np.where(too_close, NUMERICAL_EPSILON, distance)

    return {
        "R": safe_dist,
        "nx": np.where(too_close, 0.0, dx / safe_dist),
        "ny": np.where(too_close, 0.0, dy / safe_dist),
        "nz": np.where(too_close, 0.0, dz / safe_dist),
    }


def compute_retarded_distance(
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    index_traj: int,
    index_part: int,
    indices_ret: Iterable[int],
) -> DistanceResult:
    """Compute retarded distance quantities between two trajectories.

    The input ``indices_ret`` should already be chrono-matched; this function
    simply evaluates the geometric terms for each matched particle.
    """

    prototype = trajectory_ext[
        _resolve_external_step_index(trajectory_ext, index_traj)
    ]["x"]
    result: DistanceResult = {
        "R": np.zeros_like(prototype),
        "nx": np.zeros_like(prototype),
        "ny": np.zeros_like(prototype),
        "nz": np.zeros_like(prototype),
    }

    for j, idx in enumerate(indices_ret):
        dx = trajectory[index_traj]["x"][index_part] - trajectory_ext[idx]["x"][j]
        dy = trajectory[index_traj]["y"][index_part] - trajectory_ext[idx]["y"][j]
        dz = trajectory[index_traj]["z"][index_part] - trajectory_ext[idx]["z"][j]
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        if distance < NUMERICAL_EPSILON:
            result["R"][j] = NUMERICAL_EPSILON
            result["nx"][j] = 0.0
            result["ny"][j] = 0.0
            result["nz"][j] = 0.0
            continue

        result["R"][j] = distance
        result["nx"][j] = dx / distance
        result["ny"][j] = dy / distance
        result["nz"][j] = dz / distance

    return result


def chrono_match_indices_soa(
    traj: TrajectoryArraysLike,
    traj_ext: TrajectoryArraysLike,
    index_traj: int,
    index_part: int,
    *,
    mode: ChronoMatchingMode = ChronoMatchingMode.FAST,
    interpolate: bool = False,
    tolerance: float = 1e-3,
    verbosity: int = 0,
    high_precision: bool = False,
    adaptive_tolerance: bool = False,
    timestep_h: Optional[float] = None,
) -> "np.ndarray | ChronoMatchResult":
    """SOA fast path for chrono_match_indices.

    Uses direct 2-D array indexing instead of per-step dict access.
    """
    effective_tolerance = tolerance
    if adaptive_tolerance and timestep_h is not None and timestep_h > 0:
        effective_tolerance = 0.1 * timestep_h

    current_source_index = min(index_traj, traj_ext.n_steps - 1)

    dx = _soa_scalar(traj, "x", index_traj, index_part) - _soa_row(
        traj_ext, "x", current_source_index
    )
    dy = _soa_scalar(traj, "y", index_traj, index_part) - _soa_row(
        traj_ext, "y", current_source_index
    )
    dz = _soa_scalar(traj, "z", index_traj, index_part) - _soa_row(
        traj_ext, "z", current_source_index
    )
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    too_close = distance < NUMERICAL_EPSILON
    safe_dist = np.where(too_close, NUMERICAL_EPSILON, distance)
    nhat = {
        "R": safe_dist,
        "nx": np.where(too_close, 0.0, dx / safe_dist),
        "ny": np.where(too_close, 0.0, dy / safe_dist),
        "nz": np.where(too_close, 0.0, dz / safe_dist),
    }

    n_particles = traj_ext.n_particles
    index_traj_new = np.empty(n_particles, dtype=int)
    index_traj_new.fill(current_source_index)

    if interpolate:
        index_traj_next = np.empty(n_particles, dtype=int)
        weights = np.ones(n_particles, dtype=float)
        residuals = np.zeros(n_particles, dtype=float)
        needs_interp = np.zeros(n_particles, dtype=bool)
        if high_precision:
            index_traj_prev = np.empty(n_particles, dtype=int)  # noqa: F841
            index_traj_next2 = np.empty(n_particles, dtype=int)  # noqa: F841

    # Pre-extract time columns — key SOA win: avoids per-step dict walk
    # shape [current_source_index+1, n_particles]
    t_cols = _soa_time_columns(traj_ext, current_source_index)
    monotone_columns = _monotone_time_columns(traj_ext)

    if (
        mode is ChronoMatchingMode.FAST
        and not interpolate
        and n_particles > 0
        and t_cols.size <= _MAX_VECTORIZED_CHRONO_MATCH_CELLS
    ):
        b_nhat_all = (
            _soa_row(traj_ext, "bx", current_source_index) * nhat["nx"]
            + _soa_row(traj_ext, "by", current_source_index) * nhat["ny"]
            + _soa_row(traj_ext, "bz", current_source_index) * nhat["nz"]
        )
        denominator = 1.0 - b_nhat_all
        singular = np.abs(denominator) < 1e-15

        factored_denominator = 1.0 - b_nhat_all**2
        factored_denominator = np.where(
            np.abs(factored_denominator) < 1e-12,
            np.copysign(1e-12, factored_denominator),
            factored_denominator,
        )
        delta_t = nhat["R"] * (1.0 + b_nhat_all) / (C_MMNS * factored_denominator)
        if np.any(singular):
            char_time = _soa_constant(traj_ext, "char_time")[:n_particles]
            delta_t[singular] = np.where(
                char_time[singular] > 0.0,
                10.0 * char_time[singular],
                1e-3,
            )

        t_ext_new = _soa_row(traj_ext, "t", current_source_index) - delta_t
        earliest_times = t_cols[0, :]
        searchable_target = (t_ext_new > 0.0) | (earliest_times < 0.0)
        before_earliest = searchable_target & (t_ext_new <= earliest_times)
        if np.any(before_earliest):
            index_traj_new[before_earliest] = 0
        if np.any(searchable_target):
            if np.all(monotone_columns):
                matched_indices = _batched_time_upper_bound(t_cols, t_ext_new)
                has_match = searchable_target & (matched_indices < t_cols.shape[0])
                index_traj_new[has_match] = matched_indices[has_match]
            else:
                # Keep first-match semantics for nonmonotone or untracked data.
                matches = t_cols > t_ext_new[np.newaxis, :]
                has_match = searchable_target & np.any(matches, axis=0)
                if np.any(has_match):
                    matched_indices = np.argmax(matches, axis=0)
                    index_traj_new[has_match] = matched_indices[has_match]
        # Match the actual observer light cone, not the instantaneous delay
        # estimate. For a timelike history its residual decreases with time.
        residual = C_MMNS * (_soa_scalar(traj, "t", index_traj, index_part) - t_cols)
        distance_squared = np.zeros_like(residual)
        steps = np.arange(current_source_index + 1)[:, None]
        particles = np.arange(n_particles)[None, :]
        for axis in ("x", "y", "z"):
            delta = _soa_scalar(
                traj, axis, index_traj, index_part
            ) - _soa_values_at_steps(traj_ext, axis, steps, particles)
            distance_squared += delta**2
        residual -= np.sqrt(distance_squared)
        bracketed = (residual[0] >= 0) & (residual[-1] <= 0)
        matches = residual <= 0
        crossing = np.any(matches, axis=0)
        indices = np.argmax(matches, axis=0)
        index_traj_new[bracketed & crossing] = indices[bracketed & crossing]
        return index_traj_new

    for sample_index in range(n_particles):
        bx = _soa_scalar(traj_ext, "bx", current_source_index, sample_index)
        by = _soa_scalar(traj_ext, "by", current_source_index, sample_index)
        bz = _soa_scalar(traj_ext, "bz", current_source_index, sample_index)
        b_nhat = (
            bx * nhat["nx"][sample_index]
            + by * nhat["ny"][sample_index]
            + bz * nhat["nz"][sample_index]
        )

        denominator = 1.0 - b_nhat
        if abs(denominator) < 1e-15:
            char_t = _soa_constant(traj_ext, "char_time")[sample_index]
            delta_t = 10.0 * char_t if char_t > 0 else 1e-3
        elif mode is ChronoMatchingMode.AVERAGED:
            delta_t = _compute_delta_t_soa(
                mode=mode,
                distance=nhat["R"][sample_index],
                b_nhat=b_nhat,
                sample_index=sample_index,
                index_traj=index_traj,
                index_part=index_part,
                traj=traj,
                traj_ext=traj_ext,
            )
        else:
            delta_t = _compute_delta_t(
                mode=mode,
                distance=nhat["R"][sample_index],
                b_nhat=b_nhat,
                sample_index=sample_index,
                index_traj=index_traj,
                index_part=index_part,
                trajectory=None,
                trajectory_ext=None,
            )

        t_ext_new = (
            _soa_scalar(traj_ext, "t", current_source_index, sample_index) - delta_t
        )
        light_cone_match = _sampled_light_cone_match(
            lambda step, sample=sample_index: (
                _soa_scalar(traj_ext, "t", step, sample),
                np.array(
                    [_soa_scalar(traj_ext, a, step, sample) for a in ("x", "y", "z")]
                ),
            ),
            current_source_index,
            _soa_scalar(traj, "t", index_traj, index_part),
            np.array(
                [_soa_scalar(traj, a, index_traj, index_part) for a in ("x", "y", "z")]
            ),
            interpolate,
        )

        index_traj_new[sample_index] = current_source_index
        if interpolate:
            index_traj_next[sample_index] = current_source_index

        t_col = t_cols[:, sample_index]
        if light_cone_match is not None:
            t_ext_new, matched_idx = light_cone_match
        elif t_ext_new < 0 and float(t_col[0]) >= 0.0:
            continue
        else:
            matched_idx = _locate_retarded_index_soa(
                t_col,
                current_source_index,
                t_ext_new,
                monotone=bool(monotone_columns[sample_index]),
            )
        index_traj_new[sample_index] = matched_idx

        if interpolate:
            t_matched = _soa_scalar(traj_ext, "t", matched_idx, sample_index)
            residual = abs(t_matched - t_ext_new)
            residuals[sample_index] = residual

            if residual > effective_tolerance and matched_idx > 0:
                needs_interp[sample_index] = True
                idx_before = matched_idx - 1
                idx_after = matched_idx
                t_before = _soa_scalar(traj_ext, "t", idx_before, sample_index)
                t_after = _soa_scalar(traj_ext, "t", idx_after, sample_index)
                dt_span = t_after - t_before
                if dt_span > NUMERICAL_EPSILON:
                    w = np.clip((t_ext_new - t_before) / dt_span, 0.0, 1.0)
                    weights[sample_index] = w
                    index_traj_next[sample_index] = idx_before
                else:
                    weights[sample_index] = 1.0
                    index_traj_next[sample_index] = matched_idx

    if not interpolate:
        return index_traj_new

    return ChronoMatchResult(
        indices=index_traj_new,
        indices_next=index_traj_next,
        weights=weights,
        residuals=residuals,
        max_residual=float(np.max(residuals)) if len(residuals) > 0 else 0.0,
        needs_interpolation=needs_interp,
        use_cubic=False,
    )


def chrono_match_indices(
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    index_traj: int,
    index_part: int,
    *,
    mode: ChronoMatchingMode = ChronoMatchingMode.FAST,
    interpolate: bool = False,
    tolerance: float = 1e-3,
    verbosity: int = 0,
    high_precision: bool = False,
    adaptive_tolerance: bool = False,
    timestep_h: Optional[float] = None,
) -> np.ndarray | ChronoMatchResult:
    """Find retarded indices by matching the observer's light cone to history.

    A bracketed crossing uses the stored piecewise-linear source positions.
    With interpolation off, select the first knot at or after the crossing. The
    FAST/AVERAGED delay estimates remain fallbacks for unbracketed histories.

    Parameters
    ----------
    trajectory, trajectory_ext:
        Historical rider and external bunch states.
    index_traj:
        Step within ``trajectory`` currently being updated.
    index_part:
        Particle within ``trajectory[index_traj]`` to match against the entire
        external bunch.
    mode:
        ``ChronoMatchingMode.FAST`` (default) uses the single-sample delay
        ``Δt = R (1 + β·n̂) / (c (1 - (β·n̂)²)) = R / (c (1 - β·n̂))`` with
        ``n̂`` pointing from the source sample to the observer (see
        ``core.types.ChronoMatchingMode``). ``ChronoMatchingMode.AVERAGED`` blends
        two samples corresponding to emission after ``R / c`` (stationary
        source) and ``2R / c`` (ultrarelativistic source), which can provide a
        smoother retardation sequence for high-``γ`` bunches.
    interpolate:
        If True, return ChronoMatchResult with interpolation weights when the
        time residual exceeds tolerance. If False, return simple index array.
    tolerance:
        tolerance in nanoseconds. If ``|t_matched - t_target|`` > tolerance,
        interpolation is flagged as needed.
    verbosity:
        If >= 2, print warnings when residuals exceed tolerance.
    high_precision:
        Deprecated. In this non-SOA fallback only, use cubic interpolation and
        position interpolation (at least 4 trajectory points). The SOA matcher
        used by the maintained runner ignores it.
    adaptive_tolerance:
        If True, automatically set tolerance = 0.1 × timestep_h.
    timestep_h:
        Average timestep (ns) for adaptive tolerance calculation.

    Returns
    -------
    numpy.ndarray or ChronoMatchResult
        If interpolate=False: indices array without interpolation metadata.
        If interpolate=True: ChronoMatchResult with interpolation data.
    """

    # Adaptive tolerance: auto-set based on timestep
    effective_tolerance = tolerance
    if adaptive_tolerance and timestep_h is not None and timestep_h > 0:
        effective_tolerance = 0.1 * timestep_h
        if verbosity >= 3:
            print(
                f"  [Chrono-match] Adaptive tolerance: {effective_tolerance:.3e} ns (0.1 × {timestep_h:.3e} ns)"
            )

    current_source_index = _resolve_external_step_index(trajectory_ext, index_traj)
    current_source_state = trajectory_ext[current_source_index]

    nhat = compute_instantaneous_distance(
        trajectory[index_traj], current_source_state, index_part
    )
    n_particles = len(current_source_state["x"])
    index_traj_new = np.empty(n_particles, dtype=int)

    # For interpolation mode, track additional data
    if interpolate:
        index_traj_next = np.empty(n_particles, dtype=int)
        weights = np.ones(n_particles, dtype=float)
        residuals = np.zeros(n_particles, dtype=float)
        needs_interp = np.zeros(n_particles, dtype=bool)

        # For cubic interpolation (high precision mode)
        if high_precision:
            index_traj_prev = np.empty(n_particles, dtype=int)
            index_traj_next2 = np.empty(n_particles, dtype=int)

    for sample_index in range(n_particles):
        b_nhat = (
            current_source_state["bx"][sample_index] * nhat["nx"][sample_index]
            + current_source_state["by"][sample_index] * nhat["ny"][sample_index]
            + current_source_state["bz"][sample_index] * nhat["nz"][sample_index]
        )

        denominator = 1.0 - b_nhat
        epsilon = 1e-15

        if abs(denominator) < epsilon:
            if (
                "char_time" in current_source_state
                and len(current_source_state["char_time"]) > sample_index
            ):
                max_retardation = 10.0 * current_source_state["char_time"][sample_index]
            else:
                if len(current_source_state["t"]) > 1:
                    max_retardation = 10.0 * current_source_state["t"][1]
                else:
                    max_retardation = 1e-3
            delta_t = max_retardation
        else:
            delta_t = _compute_delta_t(
                mode=mode,
                distance=nhat["R"][sample_index],
                b_nhat=b_nhat,
                sample_index=sample_index,
                index_traj=index_traj,
                index_part=index_part,
                trajectory=trajectory,
                trajectory_ext=trajectory_ext,
            )

        t_ext_new = current_source_state["t"][sample_index] - delta_t
        light_cone_match = _sampled_light_cone_match(
            lambda step, sample=sample_index: (
                float(trajectory_ext[step]["t"][sample]),
                np.array([trajectory_ext[step][a][sample] for a in ("x", "y", "z")]),
            ),
            current_source_index,
            float(trajectory[index_traj]["t"][index_part]),
            np.array([trajectory[index_traj][a][index_part] for a in ("x", "y", "z")]),
            interpolate,
        )

        index_traj_new[sample_index] = current_source_index
        if interpolate:
            index_traj_next[sample_index] = current_source_index

        if light_cone_match is not None:
            t_ext_new, matched_idx = light_cone_match
        elif t_ext_new < 0 and float(trajectory_ext[0]["t"][sample_index]) >= 0.0:
            continue
        else:
            matched_idx = _locate_retarded_index(
                trajectory_ext,
                current_source_index,
                sample_index,
                t_ext_new,
            )

        index_traj_new[sample_index] = matched_idx

        # If interpolation is enabled, compute residual and interpolation weights
        if interpolate:
            t_matched = trajectory_ext[matched_idx]["t"][sample_index]
            residual = abs(t_matched - t_ext_new)
            residuals[sample_index] = residual

            # Check if we need interpolation (residual exceeds tolerance)
            if residual > effective_tolerance and matched_idx > 0:
                needs_interp[sample_index] = True

                if (
                    high_precision
                    and matched_idx >= 2
                    and matched_idx < current_source_index - 1
                ):
                    # Cubic interpolation using 4 points
                    # Use indices: matched_idx-2, matched_idx-1, matched_idx, matched_idx+1
                    idx_m2 = matched_idx - 2
                    idx_m1 = matched_idx - 1
                    idx_0 = matched_idx
                    idx_p1 = min(matched_idx + 1, current_source_index)

                    t_m1 = trajectory_ext[idx_m1]["t"][sample_index]
                    t_0 = trajectory_ext[idx_0]["t"][sample_index]

                    # Store all 4 indices for cubic interpolation
                    index_traj_prev[sample_index] = idx_m2
                    index_traj_next[sample_index] = idx_m1
                    index_traj_new[sample_index] = idx_0
                    index_traj_next2[sample_index] = idx_p1

                    # Compute cubic weight (normalized parameter in [0,1] between t_m1 and t_0)
                    dt_span = t_0 - t_m1
                    if dt_span > NUMERICAL_EPSILON:
                        u = (t_ext_new - t_m1) / dt_span
                        weights[sample_index] = np.clip(u, 0.0, 1.0)
                    else:
                        weights[sample_index] = 1.0
                else:
                    # Linear interpolation (fallback or standard mode)
                    # Find the bracketing indices
                    # matched_idx has t > t_ext_new, so we want matched_idx-1 and matched_idx
                    idx_before = matched_idx - 1
                    idx_after = matched_idx

                    t_before = trajectory_ext[idx_before]["t"][sample_index]
                    t_after = trajectory_ext[idx_after]["t"][sample_index]

                    # Compute linear interpolation weight
                    # weight for idx_after
                    dt_span = t_after - t_before
                    if dt_span > NUMERICAL_EPSILON:
                        # weight for idx_after
                        weight_after = (t_ext_new - t_before) / dt_span
                        weight_after = np.clip(weight_after, 0.0, 1.0)
                        weights[sample_index] = weight_after
                        index_traj_new[sample_index] = idx_after
                        index_traj_next[sample_index] = idx_before
                    else:
                        # Degenerate case: same time at both indices
                        weights[sample_index] = 1.0
                        index_traj_next[sample_index] = matched_idx

                    if high_precision:
                        # Initialize cubic indices even if not used
                        index_traj_prev[sample_index] = matched_idx
                        index_traj_next2[sample_index] = matched_idx
            else:
                # No interpolation needed
                index_traj_next[sample_index] = matched_idx
                weights[sample_index] = 1.0
                if high_precision:
                    index_traj_prev[sample_index] = matched_idx
                    index_traj_next2[sample_index] = matched_idx

    if not interpolate:
        return index_traj_new

    # Return full interpolation result
    max_res = float(np.max(residuals)) if len(residuals) > 0 else 0.0

    # Print diagnostics if requested
    # Only print if there are particles needing interpolation (verbosity >= 2)
    # OR if in maximum verbosity mode (verbosity >= 3, show even if none need interpolation)
    n_bad = int(np.sum(needs_interp))

    if verbosity >= 3 or (
        verbosity >= 2 and n_bad > 0 and max_res > effective_tolerance
    ):
        mode_str = "cubic" if high_precision else "linear"
        print(
            f"  [Chrono-match] Max residual: {max_res:.3e} ns (tolerance: {effective_tolerance:.3e} ns)"
        )
        print(
            f"  [Chrono-match] {n_bad}/{n_particles} particles need {mode_str} interpolation"
        )

    return ChronoMatchResult(
        indices=index_traj_new,
        indices_next=index_traj_next,
        weights=weights,
        residuals=residuals,
        max_residual=max_res,
        needs_interpolation=needs_interp,
        use_cubic=high_precision,
        indices_prev=index_traj_prev if high_precision else None,
        indices_next2=index_traj_next2 if high_precision else None,
    )


__all__ = [
    "DistanceResult",
    "ChronoMatchResult",
    "compute_instantaneous_distance",
    "compute_retarded_distance",
    "chrono_match_indices",
    "ChronoMatchingMode",
]
