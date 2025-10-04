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
from .types import ParticleState, Trajectory


DistanceResult = Dict[str, np.ndarray]


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
) -> np.ndarray:
    """Find retarded indices for a particle using chrono-matching.

    The solver needs to know which historical states of ``trajectory_ext``
    influence each particle of ``trajectory``.  Retardation is approximated by
    walking backwards in time until the causal signal arrives, matching the
    behaviour of the benchmarked legacy routine. Legacy equivalent is called "chrono_jn".
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
            delta_t = (
                nhat["R"][sample_index]
                * (1 + b_nhat)
                * trajectory_ext[index_traj]["gamma"][sample_index] ** 2
                #/ trajectory[index_traj]["gamma"][index_part] #there's no proper time in this module!!!
                / C_MMNS
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
]
