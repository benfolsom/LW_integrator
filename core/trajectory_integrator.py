"""Retarded Liénard–Wiechert field integrator (validated physics).

This module is a direct, fully documented transcription of the working
implementation found in ``legacy/covariant_integrator_library.py``.  The goal is
not to change the underlying physics, but to present the code in a maintainable
form with explicit helper utilities, type hints, and clearer entry points.

Only organisational changes have been made – the numerical operations are the
same as the proven legacy version so that existing validation data continues to
hold.  The legacy module remains in place for cross-checks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Tuple

from .self_consistency import SelfConsistencyConfig, self_consistent_step

import numpy as np

# ---------------------------------------------------------------------------
# Shared constants and type aliases
# ---------------------------------------------------------------------------

C_MMNS: float = 299.792458  # Speed of light in mm / ns (exactly as in legacy)

ParticleState = Dict[str, np.ndarray]
Trajectory = List[ParticleState]


class SimulationType(IntEnum):
    """Supported simulation modes (matches legacy integer flags)."""

    CONDUCTING_WALL = 0
    SWITCHING_WALL = 1
    BUNCH_TO_BUNCH = 2


@dataclass
class IntegratorConfig:
    """Container for simulation parameters used by :func:`retarded_integrator`."""

    steps: int
    time_step: float
    wall_position: float
    aperture_radius: float
    simulation_type: SimulationType
    bunch_mean: float = 0.0
    cavity_spacing: float = 0.0
    z_cutoff: float = 0.0


# ---------------------------------------------------------------------------
# Helper utilities mirroring the legacy implementation
# ---------------------------------------------------------------------------


def _random_sign() -> int:
    """Return +1 or -1 with equal probability (legacy helper)."""

    return 1 if random.random() < 0.5 else -1


def _zeros_like_state(vector: ParticleState) -> ParticleState:
    """Create an empty particle state dictionary with the same array layout."""

    result: ParticleState = {
        "x": np.zeros_like(vector["x"]),
        "y": np.zeros_like(vector["y"]),
        "z": np.zeros_like(vector["z"]),
        "t": np.zeros_like(vector["t"]),
        "Px": np.zeros_like(vector["Px"]),
        "Py": np.zeros_like(vector["Py"]),
        "Pz": np.zeros_like(vector["Pz"]),
        "Pt": np.zeros_like(vector["Pt"]),
        "gamma": np.zeros_like(vector["gamma"]),
        "bx": np.zeros_like(vector["bx"]),
        "by": np.zeros_like(vector["by"]),
        "bz": np.zeros_like(vector["bz"]),
        "bdotx": np.zeros_like(vector["bdotx"]),
        "bdoty": np.zeros_like(vector["bdoty"]),
        "bdotz": np.zeros_like(vector["bdotz"]),
        "q": np.copy(vector["q"]),
    }

    if "m" in vector:
        result["m"] = np.copy(vector["m"])
    if "char_time" in vector:
        result["char_time"] = np.copy(vector["char_time"])

    return result


def generate_conducting_image(
    vector: ParticleState, wall_z: float, aperture_radius: float
) -> ParticleState:
    """Legacy `_conducting_flat` rewritten with documentation."""

    result = _zeros_like_state(vector)

    for i in range(len(vector["x"])):
        r = np.sqrt(vector["x"][i] ** 2 + vector["y"][i] ** 2)

        if vector["z"][i] >= wall_z:
            result["q"].fill(0.0)
        else:
            result["q"] = -vector["q"]
            result["z"][i] = wall_z + abs(wall_z - vector["z"][i])

        R_dist = abs(result["z"][i] - vector["z"][i])

        if R_dist / 2 > aperture_radius:
            theta = np.arccos(-2 * (aperture_radius**2) / (R_dist**2) + 1)
            sign_x = _random_sign()
            sign_y = _random_sign()

            if theta < np.pi / 4:
                shift = 2 * R_dist * np.tan(theta)
                result["x"][i] = (
                    vector["x"][i]
                    + (aperture_radius + shift / np.sqrt(2)) * sign_x
                )
                result["y"][i] = (
                    vector["y"][i]
                    + (aperture_radius + shift / np.sqrt(2)) * sign_y
                )
                result["q"] = result["q"] * (
                    1
                    - 2 * (aperture_radius**2) / (R_dist**2) * 1 / (1 - np.cos(np.pi / 2))
                )
            else:
                shift = 0
                result["q"].fill(0.0)
                result["x"][i] = vector["x"][i]
                result["y"][i] = vector["y"][i]
        else:
            result["q"].fill(0.0)
            result["x"][i] = vector["x"][i]
            result["y"][i] = vector["y"][i]

        result["Px"][i] = vector["Px"][i]
        result["Py"][i] = vector["Py"][i]
        result["Pz"][i] = -vector["Pz"][i]
        result["Pt"][i] = vector["Pt"][i]
        result["gamma"][i] = vector["gamma"][i]
        result["bx"][i] = vector["bx"][i]
        result["by"][i] = vector["by"][i]
        result["bz"][i] = -vector["bz"][i]
        result["bdotx"][i] = vector["bdotx"][i]
        result["bdoty"][i] = vector["bdoty"][i]
        result["bdotz"][i] = -vector["bdotz"][i]
        result["t"][i] = vector["t"][i]

    return result


def generate_switching_image(
    vector: ParticleState, wall_z: float, aperture_radius: float, cut_z: float
) -> ParticleState:
    """Legacy `_switching_flat` implementation with docstrings."""

    result = _zeros_like_state(vector)
    result["q"] = -np.copy(vector["q"])

    for i in range(len(vector["x"])):
        if vector["z"][i] >= cut_z:
            result["q"].fill(0.0)
        else:
            result["x"][i] = vector["x"][i]
            result["y"][i] = vector["y"][i]
            result["z"][i] = wall_z + abs(wall_z - vector["z"][i])

        result["Px"][i] = vector["Px"][i]
        result["Py"][i] = vector["Py"][i]
        result["Pz"][i] = -vector["Pz"][i]
        result["Pt"][i] = vector["Pt"][i]
        result["gamma"][i] = vector["gamma"][i]
        result["bx"][i] = vector["bx"][i]
        result["by"][i] = vector["by"][i]
        result["bz"][i] = -vector["bz"][i]
        result["bdotx"][i] = vector["bdotx"][i]
        result["bdoty"][i] = vector["bdoty"][i]
        result["bdotz"][i] = -vector["bdotz"][i]
        result["t"][i] = vector["t"][i]

    return result


def compute_instantaneous_distance(
    vector: ParticleState, vector_ext: ParticleState, index: int
) -> Dict[str, np.ndarray]:
    """Clone of legacy `_dist_euclid` (instantaneous distances)."""

    result = {
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
) -> Dict[str, np.ndarray]:
    """Clone of legacy `_dist_euclid_ret` (retarded distances)."""

    result = {
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
    """Legacy ``chrono_jn`` – find retarded-time indices."""

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
                * trajectory_ext[index_traj]["gamma"][l]**2
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


# ---------------------------------------------------------------------------
# Core equations of motion (copy of legacy implementation)
# ---------------------------------------------------------------------------


def retarded_equations_of_motion(
    h: float,
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    index_traj: int,
    aperture_radius: float,
    sim_type: SimulationType,
) -> ParticleState:
    """Clone of legacy `_eqsofmotion_retarded` with light refactoring."""

    result: ParticleState = {
        "x": np.copy(trajectory[index_traj]["x"]),
        "y": np.copy(trajectory[index_traj]["y"]),
        "z": np.copy(trajectory[index_traj]["z"]),
        "t": np.copy(trajectory[index_traj]["t"]),
        "Px": np.copy(trajectory[index_traj]["Px"]),
        "Py": np.copy(trajectory[index_traj]["Py"]),
        "Pz": np.copy(trajectory[index_traj]["Pz"]),
        "Pt": np.copy(trajectory[index_traj]["Pt"]),
        "gamma": np.copy(trajectory[index_traj]["gamma"]),
        "bx": np.copy(trajectory[index_traj]["bx"]),
        "by": np.copy(trajectory[index_traj]["by"]),
        "bz": np.copy(trajectory[index_traj]["bz"]),
        "bdotx": np.copy(trajectory[index_traj]["bdotx"]),
        "bdoty": np.copy(trajectory[index_traj]["bdoty"]),
        "bdotz": np.copy(trajectory[index_traj]["bdotz"]),
        "q": trajectory[index_traj]["q"],
        "char_time": trajectory[index_traj]["char_time"],
        "m": trajectory[index_traj]["m"],
        "dummy": np.zeros_like(trajectory[index_traj]["bdotz"]),
    }

    for l in range(len(trajectory[index_traj]["x"])):
        indices_new = chrono_match_indices(trajectory, trajectory_ext, index_traj, l)
        max_ext_idx = len(trajectory_ext) - 1
        indices_new_bounded = np.minimum(np.maximum(indices_new, 0), max_ext_idx)

        nhat = compute_retarded_distance(
            trajectory, trajectory_ext, index_traj, l, indices_new_bounded
        )

        result["x"][l] = trajectory[index_traj]["x"][l]
        result["y"][l] = trajectory[index_traj]["y"][l]
        result["z"][l] = trajectory[index_traj]["z"][l]
        result["t"][l] = trajectory[index_traj]["t"][l]

        accumulated_px = trajectory[index_traj]["Px"][l]
        accumulated_py = trajectory[index_traj]["Py"][l]
        accumulated_pz = trajectory[index_traj]["Pz"][l]
        accumulated_pt = trajectory[index_traj]["Pt"][l]

        accumulated_x_field = 0.0
        accumulated_y_field = 0.0
        accumulated_z_field = 0.0

        charge_i = (
            trajectory[index_traj]["q"][l]
            if hasattr(trajectory[index_traj]["q"], "__getitem__")
            else trajectory[index_traj]["q"]
        )

        mass_i = (
            trajectory[index_traj]["m"][l]
            if hasattr(trajectory[index_traj]["m"], "__getitem__")
            else trajectory[index_traj]["m"]
        )

        for j in range(len(trajectory_ext[0]["x"])):
            ext_idx = indices_new_bounded[j]
            if ext_idx >= len(trajectory_ext) or j >= len(trajectory_ext[ext_idx]["x"]):
                continue

            if hasattr(trajectory_ext[ext_idx]["q"], "__getitem__"):
                charge_j = trajectory_ext[ext_idx]["q"][j]
            else:
                charge_j = trajectory_ext[ext_idx]["q"]

            beta_vec = (
                trajectory[index_traj]["bx"][l],
                trajectory[index_traj]["by"][l],
                trajectory[index_traj]["bz"][l],
            )
            beta_ext = (
                trajectory_ext[ext_idx]["bx"][j],
                trajectory_ext[ext_idx]["by"][j],
                trajectory_ext[ext_idx]["bz"][j],
            )
            k_factor = 1 - np.dot(
                beta_ext, (nhat["nx"][j], nhat["ny"][j], nhat["nz"][j])
            )

            if abs(k_factor) < 1e-15:
                continue

            bdot_ext = (
                trajectory_ext[ext_idx]["bdotx"][j],
                trajectory_ext[ext_idx]["bdoty"][j],
                trajectory_ext[ext_idx]["bdotz"][j],
            )
            bdot_scalar_ext = np.dot(beta_ext, bdot_ext)
            betas_scalar = np.dot(beta_ext, beta_vec)

            gamma_i = trajectory[index_traj]["gamma"][l]
            gamma_j = trajectory_ext[ext_idx]["gamma"][j]

            if gamma_j > 1e6 or gamma_i > 1e6:
                continue

            v_betas_scalar = gamma_j * gamma_i * C_MMNS**2 * (1.0 - betas_scalar)
            v_beta_dot_mixed_scalar = (
                gamma_j**4
                * gamma_i
                * C_MMNS**2
                * bdot_scalar_ext
                - gamma_i
                * C_MMNS
                * np.dot(
                    beta_vec,
                    np.multiply(bdot_ext, C_MMNS * gamma_j**2)
                    + np.multiply(beta_ext, bdot_scalar_ext) * C_MMNS * gamma_j**4,
                )
            )

            if abs(charge_i) < 1e-20 or abs(charge_j) < 1e-20:
                print(
                    f"DEBUG RETARDED: Zero charge skip - qi={charge_i:.8f}, qj={charge_j:.8f}"
                )
                continue

            charge_factor = (
                h
                * charge_i
                * charge_j
                / (k_factor**3 * C_MMNS**3 * nhat["R"][j] ** 2 * gamma_j**3)
            )

            accumulated_px += charge_factor * (
                -v_betas_scalar
                * trajectory_ext[ext_idx]["bx"][j]
                * k_factor
                * C_MMNS
                * gamma_j**2
                + v_beta_dot_mixed_scalar
                * k_factor
                * gamma_j
                * nhat["nx"][j]
                * nhat["R"][j]
                + gamma_j**2
                * nhat["nx"][j] ** 2
                * nhat["R"][j]
                * v_betas_scalar
                * (
                    trajectory_ext[ext_idx]["bdotx"][j]
                    + trajectory_ext[ext_idx]["bdotx"][j]
                    * bdot_scalar_ext
                    * gamma_j**2
                )
                + v_betas_scalar * C_MMNS * nhat["nx"][j]
            )

            accumulated_py += charge_factor * (
                -v_betas_scalar
                * trajectory_ext[ext_idx]["by"][j]
                * k_factor
                * C_MMNS
                * gamma_j**2
                + v_beta_dot_mixed_scalar
                * k_factor
                * gamma_j
                * nhat["ny"][j]
                * nhat["R"][j]
                + gamma_j**2
                * nhat["ny"][j] ** 2
                * nhat["R"][j]
                * v_betas_scalar
                * (
                    trajectory_ext[ext_idx]["bdoty"][j]
                    + trajectory_ext[ext_idx]["bdoty"][j]
                    * bdot_scalar_ext
                    * gamma_j**2
                )
                + v_betas_scalar * C_MMNS * nhat["ny"][j]
            )

            accumulated_pz += charge_factor * (
                -v_betas_scalar
                * trajectory_ext[ext_idx]["bz"][j]
                * k_factor
                * C_MMNS
                * gamma_j**2
                + v_beta_dot_mixed_scalar
                * k_factor
                * gamma_j
                * nhat["nz"][j]
                * nhat["R"][j]
                + gamma_j**2
                * nhat["nz"][j] ** 2
                * nhat["R"][j]
                * v_betas_scalar
                * (
                    trajectory_ext[ext_idx]["bdotz"][j]
                    + trajectory_ext[ext_idx]["bdotz"][j]
                    * bdot_scalar_ext
                    * gamma_j**2
                )
                + v_betas_scalar * C_MMNS * nhat["nz"][j]
            )

            accumulated_pt += (
                h
                * charge_i
                * charge_j
                / (k_factor**3 * C_MMNS**3 * nhat["R"][j] ** 2 * gamma_j**3)
            ) * (
                v_beta_dot_mixed_scalar * k_factor * gamma_j * nhat["R"][j]
                - v_betas_scalar * k_factor * C_MMNS * gamma_j**2
                - bdot_scalar_ext * v_betas_scalar * gamma_j**4 * nhat["R"][j]
                + v_betas_scalar * C_MMNS
            )

            field_contribution = (
                h
                / mass_i
                * charge_i
                / C_MMNS
                * charge_j
                / (nhat["R"][j] * k_factor)
            )
            accumulated_x_field += (
                field_contribution * trajectory_ext[ext_idx]["bx"][j]
            )
            accumulated_y_field += (
                field_contribution * trajectory_ext[ext_idx]["by"][j]
            )
            accumulated_z_field += (
                field_contribution * trajectory_ext[ext_idx]["bz"][j]
            )

        result["Px"][l] = accumulated_px
        result["Py"][l] = accumulated_py
        result["Pz"][l] = accumulated_pz
        result["Pt"][l] = accumulated_pt

        result["gamma"][l] = result["Pt"][l] / (mass_i * C_MMNS)
        result["t"][l] = trajectory[index_traj]["t"][l] + h * result["gamma"][l]

        result["x"][l] = (
            trajectory[index_traj]["x"][l]
            + h / mass_i * (result["Px"][l] - accumulated_x_field * mass_i)
        )
        result["y"][l] = (
            trajectory[index_traj]["y"][l]
            + h / mass_i * (result["Py"][l] - accumulated_y_field * mass_i)
        )
        result["z"][l] = (
            trajectory[index_traj]["z"][l]
            + h / mass_i * (result["Pz"][l] - accumulated_z_field * mass_i)
        )

        result["bx"][l] = (
            result["x"][l] - trajectory[index_traj]["x"][l]
        ) / (C_MMNS * h * result["gamma"][l])
        result["by"][l] = (
            result["y"][l] - trajectory[index_traj]["y"][l]
        ) / (C_MMNS * h * result["gamma"][l])
        result["bz"][l] = (
            result["z"][l] - trajectory[index_traj]["z"][l]
        ) / (C_MMNS * h * result["gamma"][l])

        btots = np.sqrt(
            result["bx"][l] ** 2 + result["by"][l] ** 2 + result["bz"][l] ** 2
        )
        if btots >= 1.0:
            btots_limited = 0.9999999999999
            scale = btots_limited / btots
            result["bx"][l] *= scale
            result["by"][l] *= scale
            result["bz"][l] *= scale
            btots = btots_limited

        result["gamma"][l] = 1.0 / np.sqrt(1 - btots**2)

        result["bdotx"][l] = (
            result["bx"][l] - trajectory[index_traj]["bx"][l]
        ) / (C_MMNS * h * result["gamma"][l])
        result["bdoty"][l] = (
            result["by"][l] - trajectory[index_traj]["by"][l]
        ) / (C_MMNS * h * result["gamma"][l])
        result["bdotz"][l] = (
            result["bz"][l] - trajectory[index_traj]["bz"][l]
        ) / (C_MMNS * h * result["gamma"][l])

        rad_frc_z_rhs = (
            -result["gamma"][l] ** 3
            * (mass_i * result["bdotz"][l] ** 2 * C_MMNS**2)
            * result["bz"][l]
            * C_MMNS
        )
        rad_frc_z_lhs = (
            (result["gamma"][l] - trajectory[index_traj]["gamma"][l])
            / (h * result["gamma"][l])
            * mass_i
            * result["bdotz"][l]
            * result["bz"][l]
            * C_MMNS**2
        )

        char_time_i = (
            trajectory[index_traj]["char_time"][l]
            if hasattr(trajectory[index_traj]["char_time"], "__getitem__")
            else trajectory[index_traj]["char_time"]
        )

        if rad_frc_z_rhs > (char_time_i / 1e1) or rad_frc_z_lhs > (char_time_i / 1e1):
            result["bdotz"][l] += char_time_i * (
                rad_frc_z_lhs + rad_frc_z_rhs
            ) / (mass_i * C_MMNS)

            rad_frc_x_rhs = (
                -result["gamma"][l] ** 3
                * (mass_i * result["bdotx"][l] ** 2 * C_MMNS**2)
                * result["bx"][l]
                * C_MMNS
            )
            rad_frc_x_lhs = (
                (result["gamma"][l] - trajectory[index_traj]["gamma"][l])
                / (h * result["gamma"][l])
                * mass_i
                * result["bdotx"][l]
                * result["bx"][l]
                * C_MMNS**2
            )
            rad_frc_y_rhs = (
                -result["gamma"][l] ** 3
                * (mass_i * result["bdoty"][l] ** 2 * C_MMNS**2)
                * result["by"][l]
                * C_MMNS
            )
            rad_frc_y_lhs = (
                (result["gamma"][l] - trajectory[index_traj]["gamma"][l])
                / (h * result["gamma"][l])
                * mass_i
                * result["bdoty"][l]
                * result["by"][l]
                * C_MMNS**2
            )

            result["bdotx"][l] += char_time_i * (
                rad_frc_x_lhs + rad_frc_x_rhs
            ) / (mass_i * C_MMNS)
            result["bdoty"][l] += char_time_i * (
                rad_frc_y_lhs + rad_frc_y_rhs
            ) / (mass_i * C_MMNS)

    return result


# ---------------------------------------------------------------------------
# Public integrator API
# ---------------------------------------------------------------------------


def retarded_integrator(
    steps: int,
    h_step: float,
    wall_z: float,
    aperture_radius: float,
    sim_type: SimulationType,
    init_rider: ParticleState,
    init_driver: Optional[ParticleState],
    mean: float,
    cav_spacing: float,
    z_cutoff: float,
    self_consistency: Optional[SelfConsistencyConfig] = None,
) -> Tuple[Trajectory, Trajectory]:
    """Fully compatible replacement for the legacy ``retarded_integrator``."""

    trajectory: Trajectory = [{} for _ in range(steps)]
    trajectory_drv: Trajectory = [{} for _ in range(steps)]

    for i in range(steps):
        if i == 0:
            trajectory[i] = init_rider
            if sim_type == SimulationType.CONDUCTING_WALL:
                trajectory_drv[i] = generate_conducting_image(
                    init_rider, wall_z, aperture_radius
                )
            elif sim_type == SimulationType.SWITCHING_WALL:
                trajectory_drv[i] = generate_switching_image(
                    init_rider, wall_z, aperture_radius, z_cutoff
                )
            elif sim_type == SimulationType.BUNCH_TO_BUNCH:
                if init_driver is None:
                    raise ValueError(
                        "SimulationType.BUNCH_TO_BUNCH requires init_driver state"
                    )
                trajectory_drv[i] = init_driver
        else:
            trajectory[i] = self_consistent_step(
                retarded_equations_of_motion,
                h_step,
                trajectory,
                trajectory_drv,
                i - 1,
                aperture_radius,
                sim_type,
                self_consistency,
            )

            if sim_type == SimulationType.SWITCHING_WALL:
                trajectory_drv[i] = generate_switching_image(
                    trajectory[i], wall_z, aperture_radius, z_cutoff
                )
                if np.mean(trajectory[i]["z"]) > z_cutoff:
                    z_cutoff += cav_spacing
                    wall_z += cav_spacing
            elif sim_type == SimulationType.CONDUCTING_WALL:
                trajectory_drv[i] = generate_conducting_image(
                    trajectory[i], wall_z, aperture_radius
                )
            elif sim_type == SimulationType.BUNCH_TO_BUNCH:
                if init_driver is None:
                    raise ValueError(
                        "SimulationType.BUNCH_TO_BUNCH requires init_driver state"
                    )
                trajectory_drv[i] = self_consistent_step(
                    retarded_equations_of_motion,
                    h_step,
                    trajectory_drv,
                    trajectory,
                    i - 1,
                    aperture_radius,
                    sim_type,
                    self_consistency,
                )

    return trajectory, trajectory_drv


def run_integrator(
    config: IntegratorConfig,
    init_rider: ParticleState,
    init_driver: Optional[ParticleState],
) -> Tuple[Trajectory, Trajectory]:
    """Convenience wrapper using :class:`IntegratorConfig`."""

    return retarded_integrator(
        steps=config.steps,
        h_step=config.time_step,
        wall_z=config.wall_position,
        aperture_radius=config.aperture_radius,
        sim_type=config.simulation_type,
        init_rider=init_rider,
        init_driver=init_driver,
        mean=config.bunch_mean,
        cav_spacing=config.cavity_spacing,
        z_cutoff=config.z_cutoff,
    )


__all__ = [
    "SimulationType",
    "IntegratorConfig",
    "retarded_integrator",
    "retarded_equations_of_motion",
    "generate_conducting_image",
    "generate_switching_image",
    "run_integrator",
]
