"""Distance and retarded-time utilities for the LW integrator.

The helpers in this module translate particle positions into geometric
quantities (distance, direction cosines, retarded indices) that feed the
covariant equations of motion.  They intentionally mirror the behaviour of
the legacy implementation so that validation data remains comparable.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .constants import C_MMNS, NUMERICAL_EPSILON
from .types import ChronoMatchingMode, ParticleState, Trajectory


DistanceResult = Dict[str, np.ndarray]


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

    ``FAST`` mode mirrors the legacy code path by evaluating the causal delay
    once using the instantaneous line-of-sight projection ``β·n̂``.  ``AVERAGED``
    samples two physical extremes for the emission time: ``R / c`` (a
    stationary source particle) and ``2R / c`` (a source moving at the speed of
    light along ``n̂``).  The averaged projection from those two samples is used
    to compute ``Δt`` which damps aggressive kicks for ultra-relativistic
    particles.
    """

    if mode is ChronoMatchingMode.FAST:
        return distance * (1.0 + b_nhat) / C_MMNS

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
    return distance * (1.0 + averaged_b) / C_MMNS


def _dot_beta_nhat(
    state: ParticleState, nhat: DistanceResult, sample_index: int
) -> float:
    return (
        state["bx"][sample_index] * nhat["nx"][sample_index]
        + state["by"][sample_index] * nhat["ny"][sample_index]
        + state["bz"][sample_index] * nhat["nz"][sample_index]
    )


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

    result: DistanceResult = {
        "R": np.zeros_like(vector_ext["x"]),
        "nx": np.zeros_like(vector_ext["x"]),
        "ny": np.zeros_like(vector_ext["x"]),
        "nz": np.zeros_like(vector_ext["x"]),
    }

    for j in range(len(vector_ext["x"])):
        dx = vector["x"][index] - vector_ext["x"][j]
        dy = vector["y"][index] - vector_ext["y"][j]
        dz = vector["z"][index] - vector_ext["z"][j]
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

    result: DistanceResult = {
        "R": np.zeros_like(trajectory[index_traj]["x"]),
        "nx": np.zeros_like(trajectory[index_traj]["x"]),
        "ny": np.zeros_like(trajectory[index_traj]["x"]),
        "nz": np.zeros_like(trajectory[index_traj]["x"]),
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


def chrono_match_indices(
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    index_traj: int,
    index_part: int,
    *,
    mode: ChronoMatchingMode = ChronoMatchingMode.AVERAGED,
) -> np.ndarray:
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
        ``ChronoMatchingMode.FAST`` reproduces the historical single-sample
        delay ``Δt = R (1 + β·n̂) / c``. ``ChronoMatchingMode.AVERAGED`` blends
        two samples corresponding to emission after ``R / c`` (stationary
        source) and ``2R / c`` (ultrarelativistic source), which can provide a
        smoother retardation sequence for high-``γ`` bunches.

    Returns
    -------
    numpy.ndarray
        Indices into ``trajectory_ext`` describing which historical slice
        influences each particle in the external bunch. Legacy equivalent is
        called ``chrono_jn``.
    """

    nhat = compute_instantaneous_distance(
        trajectory[index_traj], trajectory_ext[index_traj], index_part
    )
    index_traj_new = np.empty(len(trajectory_ext[index_traj]["x"]), dtype=int)

    for sample_index in range(len(trajectory_ext[index_traj]["x"])):
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
        if t_ext_new < 0:
            continue

        for k in range(index_traj, -1, -1):
            if trajectory_ext[index_traj - k]["t"][sample_index] > t_ext_new:
                index_traj_new[sample_index] = index_traj - k
                break

    return index_traj_new


__all__ = [
    "DistanceResult",
    "compute_instantaneous_distance",
    "compute_retarded_distance",
    "chrono_match_indices",
    "ChronoMatchingMode",
]
