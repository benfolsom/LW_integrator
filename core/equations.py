"""Retarded equations of motion for the Liénard–Wiechert solver.

The implementation intentionally mirrors the behaviour of the validated legacy
code so that historical regression data remains applicable.  The heavy lifting
is performed inside :func:`retarded_equations_of_motion`, which calculates the
covariant updates for momentum, position, and acceleration for each particle.
"""

from __future__ import annotations


import numpy as np

from .constants import C_MMNS
from .distances import compute_retarded_distance, chrono_match_indices
from .types import ParticleState, SimulationType, Trajectory


def retarded_equations_of_motion(
    h: float,
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    index_traj: int,
    aperture_radius: float,
    sim_type: SimulationType,
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

    Returns
    -------
    ParticleState
        Updated particle state for the next time step.
    """

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

        result["Px"][l] = accumulated_px
        result["Py"][l] = accumulated_py
        result["Pz"][l] = accumulated_pz
        result["Pt"][l] = accumulated_pt

        result["gamma"][l] = result["Pt"][l] / (mass_i * C_MMNS)
        result["t"][l] = trajectory[index_traj]["t"][l] + h * result["gamma"][l]

        result["x"][l] = trajectory[index_traj]["x"][l] + h / mass_i * (
            result["Px"][l] - accumulated_x_field * mass_i
        )
        result["y"][l] = trajectory[index_traj]["y"][l] + h / mass_i * (
            result["Py"][l] - accumulated_y_field * mass_i
        )
        result["z"][l] = trajectory[index_traj]["z"][l] + h / mass_i * (
            result["Pz"][l] - accumulated_z_field * mass_i
        )

        result["bx"][l] = (result["x"][l] - trajectory[index_traj]["x"][l]) / (
            C_MMNS * h * result["gamma"][l]
        )
        result["by"][l] = (result["y"][l] - trajectory[index_traj]["y"][l]) / (
            C_MMNS * h * result["gamma"][l]
        )
        result["bz"][l] = (result["z"][l] - trajectory[index_traj]["z"][l]) / (
            C_MMNS * h * result["gamma"][l]
        )

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

        result["bdotx"][l] = (result["bx"][l] - trajectory[index_traj]["bx"][l]) / (
            C_MMNS * h * result["gamma"][l]
        )
        result["bdoty"][l] = (result["by"][l] - trajectory[index_traj]["by"][l]) / (
            C_MMNS * h * result["gamma"][l]
        )
        result["bdotz"][l] = (result["bz"][l] - trajectory[index_traj]["bz"][l]) / (
            C_MMNS * h * result["gamma"][l]
        )

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
            result["bdotz"][l] += (
                char_time_i * (rad_frc_z_lhs + rad_frc_z_rhs) / (mass_i * C_MMNS)
            )

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

            result["bdotx"][l] += (
                char_time_i * (rad_frc_x_lhs + rad_frc_x_rhs) / (mass_i * C_MMNS)
            )
            result["bdoty"][l] += (
                char_time_i * (rad_frc_y_lhs + rad_frc_y_rhs) / (mass_i * C_MMNS)
            )

    return result


__all__ = ["retarded_equations_of_motion"]
