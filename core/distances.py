"""Distance and retarded-time utilities for the LW integrator.

The helpers in this module translate particle positions into geometric
quantities (distance, direction cosines, retarded indices) that feed the
covariant equations of motion. They preserve the validated reference behavior
used by the current solver so historical regression data remains comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from .constants import C_MMNS, NUMERICAL_EPSILON
from .types import ChronoMatchingMode, ParticleState, Trajectory, TrajectoryArrays

DistanceResult = Dict[str, np.ndarray]


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
        Time residuals |t_matched - t_target| for each source particle (in ns).
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

    Uses the correct Liénard-Wiechert formula: delta_t = R / (c * (1 - β·n̂))

    For numerical stability at ultra-relativistic energies, this is computed using
    the factored form: delta_t = R * (1 + β·n̂) / (c * (1 - (β·n̂)²))

    ``FAST`` mode uses the validated single-sample delay by evaluating the causal delay
    once using the instantaneous line-of-sight projection ``β·n̂``.  ``AVERAGED``
    samples two physical extremes for the emission time: ``R / c`` (a
    stationary source particle) and ``2R / c`` (a source moving at the speed of
    light along ``n̂``).  The averaged projection from those two samples is used
    to compute ``Δt`` which damps aggressive kicks for ultra-relativistic
    particles.

    Validated for:
        - 500 GeV electrons (γ ≈ 978,474, β ≈ 0.9999999999995)
        - 20 TeV protons (γ ≈ 21,321, β ≈ 0.999999999)
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

    time_offsets = np.array([distance / C_MMNS, 2.0 * distance / C_MMNS], dtype=float)
    sampled_b = 0.0

    for offset in time_offsets:
        target_time = trajectory_ext[index_traj]["t"][sample_index] - offset
        matched_index = _locate_retarded_index(
            trajectory_ext, index_traj, sample_index, target_time
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
) -> int:
    """SOA fast path: binary search on pre-sliced time column."""
    if target_time <= 0.0:
        return index_traj
    idx = int(np.searchsorted(t_col, target_time, side="left"))
    return min(idx, index_traj)


def _locate_retarded_index(
    trajectory_ext: Trajectory,
    index_traj: int,
    sample_index: int,
    target_time: float,
) -> int:
    if target_time <= 0.0:
        return index_traj

    for k in range(index_traj, -1, -1):
        candidate_index = index_traj - k
        if trajectory_ext[candidate_index]["t"][sample_index] >= target_time:
            return candidate_index
    return 0


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
    traj: TrajectoryArrays,
    traj_ext: TrajectoryArrays,
    index_traj: int,
    index_part: int,
    indices_ret: np.ndarray,
) -> DistanceResult:
    """SOA fast path for compute_retarded_distance.

    Avoids per-step dict lookups by accessing 2-D array slices directly.
    """
    x_obs = traj.x[index_traj, index_part]
    y_obs = traj.y[index_traj, index_part]
    z_obs = traj.z[index_traj, index_part]

    n = len(indices_ret)
    R = np.empty(n)
    nx = np.empty(n)
    ny = np.empty(n)
    nz = np.empty(n)

    for j, idx in enumerate(indices_ret):
        dx = x_obs - traj_ext.x[idx, j]
        dy = y_obs - traj_ext.y[idx, j]
        dz = z_obs - traj_ext.z[idx, j]
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        if d < NUMERICAL_EPSILON:
            R[j] = NUMERICAL_EPSILON
            nx[j] = ny[j] = nz[j] = 0.0
        else:
            R[j] = d
            nx[j] = dx / d
            ny[j] = dy / d
            nz[j] = dz / d

    return {"R": R, "nx": nx, "ny": ny, "nz": nz}


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

    prototype = trajectory_ext[index_traj]["x"]
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
    traj: TrajectoryArrays,
    traj_ext: TrajectoryArrays,
    index_traj: int,
    index_part: int,
    *,
    mode: ChronoMatchingMode = ChronoMatchingMode.AVERAGED,
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

    dx = traj.x[index_traj, index_part] - traj_ext.x[index_traj, :]
    dy = traj.y[index_traj, index_part] - traj_ext.y[index_traj, :]
    dz = traj.z[index_traj, index_part] - traj_ext.z[index_traj, :]
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

    if interpolate:
        index_traj_next = np.empty(n_particles, dtype=int)
        weights = np.ones(n_particles, dtype=float)
        residuals = np.zeros(n_particles, dtype=float)
        needs_interp = np.zeros(n_particles, dtype=bool)
        if high_precision:
            index_traj_prev = np.empty(n_particles, dtype=int)  # noqa: F841
            index_traj_next2 = np.empty(n_particles, dtype=int)  # noqa: F841

    # Pre-extract time columns — key SOA win: avoids per-step dict walk
    # shape [index_traj+1, n_particles]
    t_cols = traj_ext.t[: index_traj + 1, :]

    for sample_index in range(n_particles):
        bx = traj_ext.bx[index_traj, sample_index]
        by = traj_ext.by[index_traj, sample_index]
        bz = traj_ext.bz[index_traj, sample_index]
        b_nhat = (
            bx * nhat["nx"][sample_index]
            + by * nhat["ny"][sample_index]
            + bz * nhat["nz"][sample_index]
        )

        denominator = 1.0 - b_nhat
        if abs(denominator) < 1e-15:
            char_t = traj_ext.char_time[sample_index]
            delta_t = 10.0 * char_t if char_t > 0 else 1e-3
        elif mode is ChronoMatchingMode.AVERAGED:
            # TODO: SOA AVERAGED mode — fall back to current step
            delta_t = 0.0
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

        t_ext_new = traj_ext.t[index_traj, sample_index] - delta_t

        index_traj_new[sample_index] = index_traj
        if interpolate:
            index_traj_next[sample_index] = index_traj

        if t_ext_new < 0:
            continue

        # SOA binary search on the pre-extracted column
        t_col = t_cols[:, sample_index]
        matched_idx = _locate_retarded_index_soa(t_col, index_traj, t_ext_new)
        index_traj_new[sample_index] = matched_idx

        if interpolate:
            t_matched = traj_ext.t[matched_idx, sample_index]
            residual = abs(t_matched - t_ext_new)
            residuals[sample_index] = residual

            if residual > effective_tolerance and matched_idx > 0:
                needs_interp[sample_index] = True
                idx_before = matched_idx - 1
                idx_after = matched_idx
                t_before = traj_ext.t[idx_before, sample_index]
                t_after = traj_ext.t[idx_after, sample_index]
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
    mode: ChronoMatchingMode = ChronoMatchingMode.AVERAGED,
    interpolate: bool = False,
    tolerance: float = 1e-3,
    verbosity: int = 0,
    high_precision: bool = False,
    adaptive_tolerance: bool = False,
    timestep_h: Optional[float] = None,
) -> np.ndarray | ChronoMatchResult:
    """Find retarded indices for a particle using chrono-matching.

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
        ``ChronoMatchingMode.FAST`` uses the validated single-sample
        delay ``Δt = R (1 + β·n̂) / c``. ``ChronoMatchingMode.AVERAGED`` blends
        two samples corresponding to emission after ``R / c`` (stationary
        source) and ``2R / c`` (ultrarelativistic source), which can provide a
        smoother retardation sequence for high-``γ`` bunches.
    interpolate:
        If True, return ChronoMatchResult with interpolation weights when the
        time residual exceeds tolerance. If False, return simple index array.
    tolerance:
        tolerance in nanoseconds. If |t_matched - t_target| > tolerance,
        interpolation is flagged as needed.
    verbosity:
        If >= 2, print warnings when residuals exceed tolerance.
    high_precision:
        If True, use cubic interpolation and position interpolation. Requires at
        least 4 trajectory points for accurate cubic fit.
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

    nhat = compute_instantaneous_distance(
        trajectory[index_traj], trajectory_ext[index_traj], index_part
    )
    n_particles = len(trajectory_ext[index_traj]["x"])
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
            trajectory_ext[index_traj]["bx"][sample_index] * nhat["nx"][sample_index]
            + trajectory_ext[index_traj]["by"][sample_index] * nhat["ny"][sample_index]
            + trajectory_ext[index_traj]["bz"][sample_index] * nhat["nz"][sample_index]
        )

        denominator = 1.0 - b_nhat
        epsilon = 1e-15

        if abs(denominator) < epsilon:
            if (
                "char_time" in trajectory_ext[index_traj]
                and len(trajectory_ext[index_traj]["char_time"]) > sample_index
            ):
                max_retardation = (
                    10.0 * trajectory_ext[index_traj]["char_time"][sample_index]
                )
            else:
                if len(trajectory_ext[index_traj]["t"]) > 1:
                    max_retardation = 10.0 * trajectory_ext[index_traj]["t"][1]
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

        t_ext_new = trajectory_ext[index_traj]["t"][sample_index] - delta_t

        index_traj_new[sample_index] = index_traj
        if interpolate:
            index_traj_next[sample_index] = index_traj

        if t_ext_new < 0:
            continue

        # Find the trajectory index that brackets or is nearest to t_ext_new
        matched_idx = index_traj
        for k in range(index_traj, -1, -1):
            if trajectory_ext[index_traj - k]["t"][sample_index] > t_ext_new:
                matched_idx = index_traj - k
                break

        index_traj_new[sample_index] = matched_idx

        # If interpolation is enabled, compute residual and interpolation weights
        if interpolate:
            t_matched = trajectory_ext[matched_idx]["t"][sample_index]
            residual = abs(t_matched - t_ext_new)
            residuals[sample_index] = residual

            # Check if we need interpolation (residual exceeds tolerance)
            if residual > effective_tolerance and matched_idx > 0:
                needs_interp[sample_index] = True

                if high_precision and matched_idx >= 2 and matched_idx < index_traj - 1:
                    # Cubic interpolation using 4 points
                    # Use indices: matched_idx-2, matched_idx-1, matched_idx, matched_idx+1
                    idx_m2 = matched_idx - 2
                    idx_m1 = matched_idx - 1
                    idx_0 = matched_idx
                    idx_p1 = min(matched_idx + 1, index_traj)

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
