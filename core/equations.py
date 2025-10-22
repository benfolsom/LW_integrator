"""Retarded equations of motion for the Liénard–Wiechert solver.

The implementation intentionally mirrors the behaviour of the validated legacy
code so that historical regression data remains applicable.  The heavy lifting
is performed inside :func:`retarded_equations_of_motion`, which calculates the
covariant updates for momentum, position, and acceleration for each particle.
"""

from __future__ import annotations


import numpy as np

from .constants import C_MMNS
from .distances import (
    chrono_match_indices,
    compute_instantaneous_distance,
    compute_retarded_distance,
)
from .types import (
    ChronoMatchingMode,
    ParticleState,
    SimulationType,
    StartupMode,
    Trajectory,
)


def _ensure_startup_metadata(state: ParticleState) -> None:
    if "origin_x" not in state:
        state["origin_x"] = np.copy(state.get("x", np.array([])))
    if "origin_y" not in state:
        state["origin_y"] = np.copy(state.get("y", np.array([])))
    if "origin_z" not in state:
        state["origin_z"] = np.copy(state.get("z", np.array([])))
    if "beta_avg_x" not in state:
        state["beta_avg_x"] = np.copy(state.get("bx", np.array([])))
    if "beta_avg_y" not in state:
        state["beta_avg_y"] = np.copy(state.get("by", np.array([])))
    if "beta_avg_z" not in state:
        state["beta_avg_z"] = np.copy(state.get("bz", np.array([])))
    if "beta_samples" not in state:
        state["beta_samples"] = np.ones_like(state.get("x", np.array([])), dtype=float)


def retarded_equations_of_motion(
    h: float,
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    index_traj: int,
    aperture_radius: float,
    sim_type: SimulationType,
    chrono_mode: ChronoMatchingMode = ChronoMatchingMode.AVERAGED,
    startup_mode: StartupMode = StartupMode.COLD_START,
) -> ParticleState:
    """Core equations of motion mirroring the validated legacy implementation.

    Parameters
    ----------
    h:
        Time step between trajectory samples.
    trajectory:
        Mutable view over the rider bunch history.
    trajectory_ext:
        History of the external bunch (driver, image or opposing bunch).
    index_traj:
        Index of the current time step within ``trajectory``.
    aperture_radius:
        Aperture radius supplied to the image generators.
    sim_type:
        Simulation boundary type encoded as :class:`SimulationType`.
    chrono_mode:
        Retardation sampling strategy; ``FAST`` retains the legacy single
        sample, whereas ``AVERAGED`` blends ``R / c`` and ``2R / c`` emission
        times for the external bunch.
    startup_mode:
        Early-step handling strategy; ``COLD_START`` suppresses external forces
        until sufficient observer travel has occurred, while
        ``APPROXIMATE_BACK_HISTORY`` assumes constant source velocity to
        reconstruct an analytic history.

    Returns
    -------
    ParticleState
        Updated particle state for the next time step.
    """

    _ensure_startup_metadata(trajectory[index_traj])

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
        "char_time": trajectory[index_traj].get(
            "char_time", np.zeros_like(trajectory[index_traj]["x"])
        ),
        "m": trajectory[index_traj].get("m", np.ones_like(trajectory[index_traj]["x"])),
        "dummy": np.zeros_like(trajectory[index_traj]["bdotz"]),
        "origin_x": np.copy(trajectory[index_traj]["origin_x"]),
        "origin_y": np.copy(trajectory[index_traj]["origin_y"]),
        "origin_z": np.copy(trajectory[index_traj]["origin_z"]),
        "beta_avg_x": np.copy(trajectory[index_traj]["beta_avg_x"]),
        "beta_avg_y": np.copy(trajectory[index_traj]["beta_avg_y"]),
        "beta_avg_z": np.copy(trajectory[index_traj]["beta_avg_z"]),
        "beta_samples": np.copy(trajectory[index_traj]["beta_samples"]),
    }

    for particle_index in range(len(trajectory[index_traj]["x"])):
        if startup_mode is StartupMode.APPROXIMATE_BACK_HISTORY:
            sample_count = len(trajectory_ext[index_traj]["x"])
            indices_new_bounded = np.full(sample_count, index_traj, dtype=int)
            nhat = compute_instantaneous_distance(
                trajectory[index_traj], trajectory_ext[index_traj], particle_index
            )
            beta_ext_dot_nhat = (
                trajectory_ext[index_traj]["bx"] * nhat["nx"]
                + trajectory_ext[index_traj]["by"] * nhat["ny"]
                + trajectory_ext[index_traj]["bz"] * nhat["nz"]
            )
            nhat["R"] = nhat["R"] * (1.0 + beta_ext_dot_nhat)
        else:
            indices_new = chrono_match_indices(
                trajectory,
                trajectory_ext,
                index_traj,
                particle_index,
                mode=chrono_mode,
            )
            max_ext_idx = len(trajectory_ext) - 1
            indices_new_bounded = np.minimum(np.maximum(indices_new, 0), max_ext_idx)

            nhat = compute_retarded_distance(
                trajectory,
                trajectory_ext,
                index_traj,
                particle_index,
                indices_new_bounded,
            )

        result["x"][particle_index] = trajectory[index_traj]["x"][particle_index]
        result["y"][particle_index] = trajectory[index_traj]["y"][particle_index]
        result["z"][particle_index] = trajectory[index_traj]["z"][particle_index]
        result["t"][particle_index] = trajectory[index_traj]["t"][particle_index]

        accumulated_px = trajectory[index_traj]["Px"][particle_index]
        accumulated_py = trajectory[index_traj]["Py"][particle_index]
        accumulated_pz = trajectory[index_traj]["Pz"][particle_index]
        accumulated_pt = trajectory[index_traj]["Pt"][particle_index]

        accumulated_x_field = 0.0
        accumulated_y_field = 0.0
        accumulated_z_field = 0.0

        charge_i = (
            trajectory[index_traj]["q"][particle_index]
            if hasattr(trajectory[index_traj]["q"], "__getitem__")
            else trajectory[index_traj]["q"]
        )

        mass_i = (
            trajectory[index_traj]["m"][particle_index]
            if hasattr(trajectory[index_traj]["m"], "__getitem__")
            else trajectory[index_traj]["m"]
        )

        apply_external = True
        if startup_mode is StartupMode.COLD_START and nhat["R"].size > 0:
            origin = (
                trajectory[index_traj]["origin_x"][particle_index],
                trajectory[index_traj]["origin_y"][particle_index],
                trajectory[index_traj]["origin_z"][particle_index],
            )
            current = (
                trajectory[index_traj]["x"][particle_index],
                trajectory[index_traj]["y"][particle_index],
                trajectory[index_traj]["z"][particle_index],
            )
            travel_distance = float(
                np.sqrt(
                    (current[0] - origin[0]) ** 2
                    + (current[1] - origin[1]) ** 2
                    + (current[2] - origin[2]) ** 2
                )
            )

            beta_avg_x = trajectory[index_traj]["beta_avg_x"][particle_index]
            beta_avg_y = trajectory[index_traj]["beta_avg_y"][particle_index]
            beta_avg_z = trajectory[index_traj]["beta_avg_z"][particle_index]
            beta_avg_dot_nhat = (
                beta_avg_x * nhat["nx"]
                + beta_avg_y * nhat["ny"]
                + beta_avg_z * nhat["nz"]
            )
            thresholds = nhat["R"] * (1.0 - beta_avg_dot_nhat)
            if thresholds.size > 0:
                gating_threshold = float(np.max(np.maximum(thresholds, 0.0)))
                apply_external = travel_distance >= gating_threshold

        for j in range(len(trajectory_ext[0]["x"])):
            ext_idx = indices_new_bounded[j]
            if ext_idx >= len(trajectory_ext) or j >= len(trajectory_ext[ext_idx]["x"]):
                continue

            if not apply_external:
                continue

            if hasattr(trajectory_ext[ext_idx]["q"], "__getitem__"):
                charge_j = trajectory_ext[ext_idx]["q"][j]
            else:
                charge_j = trajectory_ext[ext_idx]["q"]

            beta_vec = (
                trajectory[index_traj]["bx"][particle_index],
                trajectory[index_traj]["by"][particle_index],
                trajectory[index_traj]["bz"][particle_index],
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

            gamma_i = trajectory[index_traj]["gamma"][particle_index]
            gamma_j = trajectory_ext[ext_idx]["gamma"][j]

            if gamma_j > 1e6 or gamma_i > 1e6:
                continue

            v_betas_scalar = gamma_j * gamma_i * C_MMNS**2 * (1.0 - betas_scalar)
            v_beta_dot_mixed_scalar = (
                gamma_j**4 * gamma_i * C_MMNS**2 * bdot_scalar_ext
                - gamma_i
                * C_MMNS
                * np.dot(
                    beta_vec,
                    np.multiply(bdot_ext, C_MMNS * gamma_j**2)
                    + np.multiply(beta_ext, bdot_scalar_ext) * C_MMNS * gamma_j**4,
                )
            )

            if abs(charge_i) < 1e-20 or abs(charge_j) < 1e-20:
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
                    + trajectory_ext[ext_idx]["bdotx"][j] * bdot_scalar_ext * gamma_j**2
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
                    + trajectory_ext[ext_idx]["bdoty"][j] * bdot_scalar_ext * gamma_j**2
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
                    + trajectory_ext[ext_idx]["bdotz"][j] * bdot_scalar_ext * gamma_j**2
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
                h / mass_i * charge_i / C_MMNS * charge_j / (nhat["R"][j] * k_factor)
            )
            accumulated_x_field += field_contribution * trajectory_ext[ext_idx]["bx"][j]
            accumulated_y_field += field_contribution * trajectory_ext[ext_idx]["by"][j]
            accumulated_z_field += field_contribution * trajectory_ext[ext_idx]["bz"][j]

        result["Px"][particle_index] = accumulated_px
        result["Py"][particle_index] = accumulated_py
        result["Pz"][particle_index] = accumulated_pz
        result["Pt"][particle_index] = accumulated_pt

        result["gamma"][particle_index] = result["Pt"][particle_index] / (
            mass_i * C_MMNS
        )
        result["t"][particle_index] = (
            trajectory[index_traj]["t"][particle_index]
            + h * result["gamma"][particle_index]
        )

        result["x"][particle_index] = trajectory[index_traj]["x"][
            particle_index
        ] + h / mass_i * (result["Px"][particle_index] - accumulated_x_field * mass_i)
        result["y"][particle_index] = trajectory[index_traj]["y"][
            particle_index
        ] + h / mass_i * (result["Py"][particle_index] - accumulated_y_field * mass_i)
        result["z"][particle_index] = trajectory[index_traj]["z"][
            particle_index
        ] + h / mass_i * (result["Pz"][particle_index] - accumulated_z_field * mass_i)

        result["bx"][particle_index] = (
            result["x"][particle_index] - trajectory[index_traj]["x"][particle_index]
        ) / (C_MMNS * h * result["gamma"][particle_index])
        result["by"][particle_index] = (
            result["y"][particle_index] - trajectory[index_traj]["y"][particle_index]
        ) / (C_MMNS * h * result["gamma"][particle_index])
        result["bz"][particle_index] = (
            result["z"][particle_index] - trajectory[index_traj]["z"][particle_index]
        ) / (C_MMNS * h * result["gamma"][particle_index])

        btots = np.sqrt(
            result["bx"][particle_index] ** 2
            + result["by"][particle_index] ** 2
            + result["bz"][particle_index] ** 2
        )
        if btots >= 1.0:
            btots_limited = 0.9999999999999
            scale = btots_limited / btots
            result["bx"][particle_index] *= scale
            result["by"][particle_index] *= scale
            result["bz"][particle_index] *= scale
            btots = btots_limited

        result["gamma"][particle_index] = 1.0 / np.sqrt(1 - btots**2)

        result["bdotx"][particle_index] = (
            result["bx"][particle_index] - trajectory[index_traj]["bx"][particle_index]
        ) / (C_MMNS * h * result["gamma"][particle_index])
        result["bdoty"][particle_index] = (
            result["by"][particle_index] - trajectory[index_traj]["by"][particle_index]
        ) / (C_MMNS * h * result["gamma"][particle_index])
        result["bdotz"][particle_index] = (
            result["bz"][particle_index] - trajectory[index_traj]["bz"][particle_index]
        ) / (C_MMNS * h * result["gamma"][particle_index])

        rad_frc_z_rhs = (
            -result["gamma"][particle_index] ** 3
            * (mass_i * result["bdotz"][particle_index] ** 2 * C_MMNS**2)
            * result["bz"][particle_index]
            * C_MMNS
        )
        rad_frc_z_lhs = (
            (
                result["gamma"][particle_index]
                - trajectory[index_traj]["gamma"][particle_index]
            )
            / (h * result["gamma"][particle_index])
            * mass_i
            * result["bdotz"][particle_index]
            * result["bz"][particle_index]
            * C_MMNS**2
        )

        char_time_i = (
            trajectory[index_traj]["char_time"][particle_index]
            if hasattr(trajectory[index_traj]["char_time"], "__getitem__")
            else trajectory[index_traj]["char_time"]
        )

        if rad_frc_z_rhs > (char_time_i / 1e1) or rad_frc_z_lhs > (char_time_i / 1e1):
            result["bdotz"][particle_index] += (
                char_time_i * (rad_frc_z_lhs + rad_frc_z_rhs) / (mass_i * C_MMNS)
            )

            rad_frc_x_rhs = (
                -result["gamma"][particle_index] ** 3
                * (mass_i * result["bdotx"][particle_index] ** 2 * C_MMNS**2)
                * result["bx"][particle_index]
                * C_MMNS
            )
            rad_frc_x_lhs = (
                (
                    result["gamma"][particle_index]
                    - trajectory[index_traj]["gamma"][particle_index]
                )
                / (h * result["gamma"][particle_index])
                * mass_i
                * result["bdotx"][particle_index]
                * result["bx"][particle_index]
                * C_MMNS**2
            )

            rad_frc_y_rhs = (
                -result["gamma"][particle_index] ** 3
                * (mass_i * result["bdoty"][particle_index] ** 2 * C_MMNS**2)
                * result["by"][particle_index]
                * C_MMNS
            )
            rad_frc_y_lhs = (
                (
                    result["gamma"][particle_index]
                    - trajectory[index_traj]["gamma"][particle_index]
                )
                / (h * result["gamma"][particle_index])
                * mass_i
                * result["bdoty"][particle_index]
                * result["by"][particle_index]
                * C_MMNS**2
            )

            result["bdotx"][particle_index] += (
                char_time_i * (rad_frc_x_lhs + rad_frc_x_rhs) / (mass_i * C_MMNS)
            )
            result["bdoty"][particle_index] += (
                char_time_i * (rad_frc_y_lhs + rad_frc_y_rhs) / (mass_i * C_MMNS)
            )

        prev_samples = float(trajectory[index_traj]["beta_samples"][particle_index])
        current_samples = prev_samples + 1.0
        prev_avg_x = trajectory[index_traj]["beta_avg_x"][particle_index]
        prev_avg_y = trajectory[index_traj]["beta_avg_y"][particle_index]
        prev_avg_z = trajectory[index_traj]["beta_avg_z"][particle_index]

        result["beta_samples"][particle_index] = current_samples
        result["beta_avg_x"][particle_index] = (
            prev_avg_x * prev_samples + result["bx"][particle_index]
        ) / current_samples
        result["beta_avg_y"][particle_index] = (
            prev_avg_y * prev_samples + result["by"][particle_index]
        ) / current_samples
        result["beta_avg_z"][particle_index] = (
            prev_avg_z * prev_samples + result["bz"][particle_index]
        ) / current_samples

    return result


__all__ = ["retarded_equations_of_motion"]
