"""Distance and retarded-time utilities for the LW integrator."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .constants import C_MMNS
from .types import ParticleState, Trajectory


DistanceResult = Dict[str, np.ndarray]


def compute_instantaneous_distance(
    vector: ParticleState, vector_ext: ParticleState, index: int
) -> DistanceResult:
    """Compute Euclidean distance and direction cosines for a particle pair."""

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
        result["R"][j] = np.sqrt(dx**2 + dy**2 + dz**2)
        result["nx"][j] = dx / result["R"][j]
        result["ny"][j] = dy / result["R"][j]
        result["nz"][j] = dz / result["R"][j]

    return result


def compute_retarded_distance(
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    index_traj: int,
    index_part: int,
    indices_ret: Iterable[int],
) -> DistanceResult:
    """Compute retarded distance quantities between two trajectories."""

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
        result["R"][j] = np.sqrt(dx**2 + dy**2 + dz**2)
        result["nx"][j] = dx / result["R"][j]
        result["ny"][j] = dy / result["R"][j]
        result["nz"][j] = dz / result["R"][j]

    return result


def chrono_match_indices(
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    index_traj: int,
    index_part: int,
) -> np.ndarray:
    """Find retarded indices for a given particle using chrono-matching."""

    nhat = compute_instantaneous_distance(
        trajectory[index_traj], trajectory_ext[index_traj], index_part
    )
    index_traj_new = np.empty(len(trajectory_ext[index_traj]["x"]), dtype=int)

    for l in range(len(trajectory_ext[index_traj]["x"])):
        b_nhat = (
            trajectory_ext[index_traj]["bx"][l] * nhat["nx"][l]
            + trajectory_ext[index_traj]["by"][l] * nhat["ny"][l]
            + trajectory_ext[index_traj]["bz"][l] * nhat["nz"][l]
        )

        denominator = 1.0 - b_nhat
        epsilon = 1e-15

        if abs(denominator) < epsilon:
            if (
                "char_time" in trajectory_ext[index_traj]
                and len(trajectory_ext[index_traj]["char_time"]) > l
            ):
                max_retardation = 10.0 * trajectory_ext[index_traj]["char_time"][l]
            else:
                if len(trajectory_ext[index_traj]["t"]) > 1:
                    max_retardation = 10.0 * trajectory_ext[index_traj]["t"][1]
                else:
                    max_retardation = 1e-3
            delta_t = max_retardation
        else:
            delta_t = (
                nhat["R"][l]
                * (1 + b_nhat)
                * trajectory_ext[index_traj]["gamma"][l] ** 2
                / trajectory[index_traj]["gamma"][index_part]
                / C_MMNS
            )

        t_ext_new = trajectory_ext[index_traj]["t"][l] - delta_t

        if t_ext_new < 0:
            index_traj_new[l] = index_traj
        else:
            for k in range(index_traj, -1, -1):
                if trajectory_ext[index_traj - k]["t"][l] > t_ext_new:
                    index_traj_new[l] = index_traj - k
                    break

    return index_traj_new


__all__ = [
    "DistanceResult",
    "compute_instantaneous_distance",
    "compute_retarded_distance",
    "chrono_match_indices",
]
