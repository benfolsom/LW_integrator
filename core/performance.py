"""
Numba-accelerated retarded integrator utilities.

This module is a faithful, structured transcription of
``legacy/numba_optimized_integrator.py``.  The goal is to expose the validated
optimised routines in a predictable API while leaving the original legacy file
untouched for regression comparison.
"""

# mypy: ignore-errors

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .images import generate_conducting_image, generate_switching_image
from .integrator import retarded_integrator
from .self_consistency import SelfConsistencyConfig, self_consistent_step
from .types import IntegratorConfig, ParticleState, SimulationType

C_MMNS = 299.792458  # mm/ns (identical to legacy constant)

try:  # pragma: no cover - optional dependency path
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def jit(*_args, **_kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator

    def prange(*_args, **_kwargs):  # type: ignore[misc]
        return range(*_args)


@dataclass
class OptimisationOptions:
    """Control flags for :func:`run_optimised_integrator`."""

    use_numba: bool = True
    run_benchmark: bool = False
    self_consistency: Optional[SelfConsistencyConfig] = None


# ---------------------------------------------------------------------------
# Helper utilities (converted directly from legacy helpers)
# ---------------------------------------------------------------------------


def _ensure_array(value: np.ndarray | float | int, size: int) -> np.ndarray:
    if np.isscalar(value):
        return np.full(size, value, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def dict_to_arrays(particle_dict: ParticleState) -> Tuple[Dict[str, np.ndarray], int]:
    n_particles = len(particle_dict["x"])
    arrays: Dict[str, np.ndarray] = {}
    for key, value in particle_dict.items():
        if key == "dummy":
            continue
        arrays[key] = _ensure_array(value, n_particles)
    return arrays, n_particles


def arrays_to_dict(arrays: Dict[str, np.ndarray]) -> ParticleState:
    return {key: value for key, value in arrays.items()}


# ---------------------------------------------------------------------------
# Numba kernels (verbatim physics from the legacy implementation)
# ---------------------------------------------------------------------------


@jit(nopython=True, fastmath=True)
def _compute_euclidean_distance(x_i, y_i, z_i, x_j, y_j, z_j):
    dx = x_i - x_j
    dy = y_i - y_j
    dz = z_i - z_j
    R = np.sqrt(dx * dx + dy * dy + dz * dz)
    if R < 1e-15:
        return R, 0.0, 0.0, 0.0
    return R, dx / R, dy / R, dz / R


@jit(nopython=True, fastmath=True, parallel=True)
def _compute_electromagnetic_forces(
    x,
    y,
    z,
    px,
    py,
    pz,
    pt,
    gamma,
    bx,
    by,
    bz,
    q,
    m,
    x_ext,
    y_ext,
    z_ext,
    px_ext,
    py_ext,
    pz_ext,
    pt_ext,
    gamma_ext,
    bx_ext,
    by_ext,
    bz_ext,
    bdotx_ext,
    bdoty_ext,
    bdotz_ext,
    q_ext,
    h,
    n_particles,
    n_ext_particles,
):
    px_new = np.copy(px)
    py_new = np.copy(py)
    pz_new = np.copy(pz)
    pt_new = np.copy(pt)

    x_field = np.zeros(n_particles)
    y_field = np.zeros(n_particles)
    z_field = np.zeros(n_particles)

    for i in prange(n_particles):
        if abs(q[i]) < 1e-20:
            continue

        for j in range(n_ext_particles):
            if abs(q_ext[j]) < 1e-20:
                continue

            R, nx, ny, nz = _compute_euclidean_distance(
                x[i], y[i], z[i], x_ext[j], y_ext[j], z_ext[j]
            )
            if R < 1e-15:
                continue

            beta_vec = np.array([bx[i], by[i], bz[i]])
            beta_ext = np.array([bx_ext[j], by_ext[j], bz_ext[j]])
            nhat_vec = np.array([nx, ny, nz])

            k_factor = 1.0 - np.dot(beta_ext, nhat_vec)
            if abs(k_factor) < 1e-15:
                continue

            bdot_ext = np.array([bdotx_ext[j], bdoty_ext[j], bdotz_ext[j]])
            bdot_scalar_ext = np.dot(beta_ext, bdot_ext)
            betas_scalar = np.dot(beta_ext, beta_vec)

            v_betas_scalar = (
                gamma_ext[j] * gamma[i] * C_MMNS * C_MMNS * (1.0 - betas_scalar)
            )
            v_beta_dot_mixed_scalar = gamma_ext[j] ** 4 * gamma[
                i
            ] * C_MMNS * C_MMNS * bdot_scalar_ext - gamma[i] * C_MMNS * np.dot(
                beta_vec,
                bdot_ext * C_MMNS * gamma_ext[j] ** 2
                + beta_ext * bdot_scalar_ext * C_MMNS * gamma_ext[j] ** 4,
            )

            charge_factor = (
                h
                * q[i]
                * q_ext[j]
                / (k_factor**3 * C_MMNS**3 * R * R * gamma_ext[j] ** 3)
            )

            force_x = charge_factor * (
                -bx_ext[j] * v_betas_scalar * k_factor * C_MMNS * gamma_ext[j] ** 2
                + v_beta_dot_mixed_scalar * k_factor * gamma_ext[j] * nx * R
                + gamma_ext[j] ** 2
                * nx
                * nx
                * R
                * v_betas_scalar
                * (bdotx_ext[j] + bdotx_ext[j] * bdot_scalar_ext * gamma_ext[j] ** 2)
                + v_betas_scalar * C_MMNS * nx
            )
            force_y = charge_factor * (
                -by_ext[j] * v_betas_scalar * k_factor * C_MMNS * gamma_ext[j] ** 2
                + v_beta_dot_mixed_scalar * k_factor * gamma_ext[j] * ny * R
                + gamma_ext[j] ** 2
                * ny
                * ny
                * R
                * v_betas_scalar
                * (bdoty_ext[j] + bdoty_ext[j] * bdot_scalar_ext * gamma_ext[j] ** 2)
                + v_betas_scalar * C_MMNS * ny
            )
            force_z = charge_factor * (
                -bz_ext[j] * v_betas_scalar * k_factor * C_MMNS * gamma_ext[j] ** 2
                + v_beta_dot_mixed_scalar * k_factor * gamma_ext[j] * nz * R
                + gamma_ext[j] ** 2
                * nz
                * nz
                * R
                * v_betas_scalar
                * (bdotz_ext[j] + bdotz_ext[j] * bdot_scalar_ext * gamma_ext[j] ** 2)
                + v_betas_scalar * C_MMNS * nz
            )
            force_t = (
                h
                * q[i]
                * q_ext[j]
                / (k_factor**3 * C_MMNS**3 * R * R * gamma_ext[j] ** 3)
            ) * (
                v_beta_dot_mixed_scalar * k_factor * gamma_ext[j] * R
                - v_betas_scalar * k_factor * C_MMNS * gamma_ext[j] ** 2
                - bdot_scalar_ext * v_betas_scalar * gamma_ext[j] ** 4 * R
                + v_betas_scalar * C_MMNS
            )

            px_new[i] += force_x
            py_new[i] += force_y
            pz_new[i] += force_z
            pt_new[i] += force_t

            field_factor = h / m[i] * q[i] / C_MMNS * q_ext[j]
            x_field[i] += field_factor * bx_ext[j] / (R * k_factor)
            y_field[i] += field_factor * by_ext[j] / (R * k_factor)
            z_field[i] += field_factor * bz_ext[j] / (R * k_factor)

    return px_new, py_new, pz_new, pt_new, x_field, y_field, z_field


@jit(nopython=True, fastmath=True, parallel=True)
def _update_particle_kinematics(
    x,
    y,
    z,
    t,
    px,
    py,
    pz,
    pt,
    gamma,
    bx,
    by,
    bz,
    bdotx,
    bdoty,
    bdotz,
    m,
    char_time,
    h,
    x_field,
    y_field,
    z_field,
    n_particles,
):
    x_new = np.copy(x)
    y_new = np.copy(y)
    z_new = np.copy(z)
    t_new = np.copy(t)
    gamma_new = np.copy(gamma)
    bx_new = np.copy(bx)
    by_new = np.copy(by)
    bz_new = np.copy(bz)
    bdotx_new = np.copy(bdotx)
    bdoty_new = np.copy(bdoty)
    bdotz_new = np.copy(bdotz)

    for i in prange(n_particles):
        gamma_new[i] = pt[i] / (m[i] * C_MMNS)
        t_new[i] = t[i] + h * gamma_new[i]

        x_new[i] = x[i] + h / m[i] * (px[i] - x_field[i] * m[i])
        y_new[i] = y[i] + h / m[i] * (py[i] - y_field[i] * m[i])
        z_new[i] = z[i] + h / m[i] * (pz[i] - z_field[i] * m[i])

        bx_new[i] = (x_new[i] - x[i]) / (C_MMNS * h * gamma_new[i])
        by_new[i] = (y_new[i] - y[i]) / (C_MMNS * h * gamma_new[i])
        bz_new[i] = (z_new[i] - z[i]) / (C_MMNS * h * gamma_new[i])

        btots = np.sqrt(bx_new[i] ** 2 + by_new[i] ** 2 + bz_new[i] ** 2)
        if btots >= 1.0:
            limit = 0.9999999999999
            scale = limit / btots
            bx_new[i] *= scale
            by_new[i] *= scale
            bz_new[i] *= scale
            btots = limit

        gamma_new[i] = 1.0 / np.sqrt(1.0 - btots * btots)

        bdotx_new[i] = (bx_new[i] - bx[i]) / (C_MMNS * h * gamma_new[i])
        bdoty_new[i] = (by_new[i] - by[i]) / (C_MMNS * h * gamma_new[i])
        bdotz_new[i] = (bz_new[i] - bz[i]) / (C_MMNS * h * gamma_new[i])

        rad_frc_z_rhs = (
            -gamma_new[i] ** 3
            * (m[i] * bdotz_new[i] ** 2 * C_MMNS * C_MMNS)
            * bz_new[i]
            * C_MMNS
        )
        rad_frc_z_lhs = (
            (gamma_new[i] - gamma[i])
            / (h * gamma_new[i])
            * m[i]
            * bdotz_new[i]
            * bz_new[i]
            * C_MMNS
            * C_MMNS
        )

        if rad_frc_z_rhs > (char_time[i] / 10.0) or rad_frc_z_lhs > (
            char_time[i] / 10.0
        ):
            bdotz_new[i] += (
                char_time[i] * (rad_frc_z_lhs + rad_frc_z_rhs) / (m[i] * C_MMNS)
            )

            rad_frc_x_rhs = (
                -gamma_new[i] ** 3
                * (m[i] * bdotx_new[i] ** 2 * C_MMNS * C_MMNS)
                * bx_new[i]
                * C_MMNS
            )
            rad_frc_x_lhs = (
                (gamma_new[i] - gamma[i])
                / (h * gamma_new[i])
                * m[i]
                * bdotx_new[i]
                * bx_new[i]
                * C_MMNS
                * C_MMNS
            )
            rad_frc_y_rhs = (
                -gamma_new[i] ** 3
                * (m[i] * bdoty_new[i] ** 2 * C_MMNS * C_MMNS)
                * by_new[i]
                * C_MMNS
            )
            rad_frc_y_lhs = (
                (gamma_new[i] - gamma[i])
                / (h * gamma_new[i])
                * m[i]
                * bdoty_new[i]
                * by_new[i]
                * C_MMNS
                * C_MMNS
            )

            bdotx_new[i] += (
                char_time[i] * (rad_frc_x_lhs + rad_frc_x_rhs) / (m[i] * C_MMNS)
            )
            bdoty_new[i] += (
                char_time[i] * (rad_frc_y_lhs + rad_frc_y_rhs) / (m[i] * C_MMNS)
            )

    return (
        x_new,
        y_new,
        z_new,
        t_new,
        gamma_new,
        bx_new,
        by_new,
        bz_new,
        bdotx_new,
        bdoty_new,
        bdotz_new,
    )


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------


def eqsofmotion_retarded_numba(
    h: float,
    trajectory,
    trajectory_ext,
    index_traj: int,
    aperture_radius: float,
    sim_type: SimulationType,
) -> ParticleState:
    current_arrays, n_particles = dict_to_arrays(trajectory[index_traj])
    ext_arrays, n_ext_particles = dict_to_arrays(trajectory_ext[index_traj])

    px_new, py_new, pz_new, pt_new, x_field, y_field, z_field = (
        _compute_electromagnetic_forces(
            current_arrays["x"],
            current_arrays["y"],
            current_arrays["z"],
            current_arrays["Px"],
            current_arrays["Py"],
            current_arrays["Pz"],
            current_arrays["Pt"],
            current_arrays["gamma"],
            current_arrays["bx"],
            current_arrays["by"],
            current_arrays["bz"],
            current_arrays["q"],
            current_arrays["m"],
            ext_arrays["x"],
            ext_arrays["y"],
            ext_arrays["z"],
            ext_arrays["Px"],
            ext_arrays["Py"],
            ext_arrays["Pz"],
            ext_arrays["Pt"],
            ext_arrays["gamma"],
            ext_arrays["bx"],
            ext_arrays["by"],
            ext_arrays["bz"],
            ext_arrays["bdotx"],
            ext_arrays["bdoty"],
            ext_arrays["bdotz"],
            ext_arrays["q"],
            h,
            n_particles,
            n_ext_particles,
        )
    )

    (
        x_new,
        y_new,
        z_new,
        t_new,
        gamma_new,
        bx_new,
        by_new,
        bz_new,
        bdotx_new,
        bdoty_new,
        bdotz_new,
    ) = _update_particle_kinematics(
        current_arrays["x"],
        current_arrays["y"],
        current_arrays["z"],
        current_arrays["t"],
        px_new,
        py_new,
        pz_new,
        pt_new,
        current_arrays["gamma"],
        current_arrays["bx"],
        current_arrays["by"],
        current_arrays["bz"],
        current_arrays["bdotx"],
        current_arrays["bdoty"],
        current_arrays["bdotz"],
        current_arrays["m"],
        current_arrays["char_time"],
        h,
        x_field,
        y_field,
        z_field,
        n_particles,
    )

    result_arrays = {
        "x": x_new,
        "y": y_new,
        "z": z_new,
        "t": t_new,
        "Px": px_new,
        "Py": py_new,
        "Pz": pz_new,
        "Pt": pt_new,
        "gamma": gamma_new,
        "bx": bx_new,
        "by": by_new,
        "bz": bz_new,
        "bdotx": bdotx_new,
        "bdoty": bdoty_new,
        "bdotz": bdotz_new,
        "q": current_arrays["q"],
        "m": current_arrays["m"],
        "char_time": current_arrays["char_time"],
        "dummy": np.zeros_like(bdotz_new),
    }

    return arrays_to_dict(result_arrays)


def retarded_integrator_numba(
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
) -> Tuple[Tuple[ParticleState, ...], Tuple[ParticleState, ...]]:
    warnings.filterwarnings("ignore")

    trajectory = [dict() for _ in range(steps)]
    trajectory_drv = [dict() for _ in range(steps)]

    wall_position = wall_z
    switching_cutoff = z_cutoff

    for i in range(steps):
        if i == 0:
            trajectory[i] = init_rider
            if sim_type == SimulationType.CONDUCTING_WALL:
                trajectory_drv[i] = generate_conducting_image(
                    init_rider, wall_position, aperture_radius
                )
            elif sim_type == SimulationType.SWITCHING_WALL:
                trajectory_drv[i] = generate_switching_image(
                    init_rider, wall_position, aperture_radius, switching_cutoff
                )
            elif sim_type == SimulationType.BUNCH_TO_BUNCH:
                if init_driver is None:
                    raise ValueError("Driver bunch required for bunch-to-bunch mode")
                trajectory_drv[i] = init_driver
            else:
                trajectory_drv[i] = init_driver or init_rider
            continue

        trajectory[i] = self_consistent_step(
            eqsofmotion_retarded_numba,
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
                trajectory[i], wall_position, aperture_radius, switching_cutoff
            )
            if np.mean(trajectory[i]["z"]) > switching_cutoff:
                switching_cutoff += cav_spacing
                wall_position += cav_spacing
        elif sim_type == SimulationType.CONDUCTING_WALL:
            trajectory_drv[i] = generate_conducting_image(
                trajectory[i], wall_position, aperture_radius
            )
        elif sim_type == SimulationType.BUNCH_TO_BUNCH:
            if init_driver is None:
                raise ValueError("Driver bunch required for bunch-to-bunch mode")
            trajectory_drv[i] = self_consistent_step(
                eqsofmotion_retarded_numba,
                h_step,
                trajectory_drv,
                trajectory,
                i - 1,
                aperture_radius,
                sim_type,
                self_consistency,
            )

    return tuple(trajectory), tuple(trajectory_drv)


def run_optimised_integrator(
    config: IntegratorConfig,
    init_rider: ParticleState,
    init_driver: Optional[ParticleState],
    options: Optional[OptimisationOptions] = None,
) -> Tuple[Tuple[ParticleState, ...], Tuple[ParticleState, ...]]:
    opts = options or OptimisationOptions()

    use_numba = opts.use_numba and NUMBA_AVAILABLE
    if opts.use_numba and not NUMBA_AVAILABLE:
        print("Warning: Numba unavailable – falling back to pure Python integrator.")

    if opts.run_benchmark and NUMBA_AVAILABLE:
        import time

        start = time.time()
        base_traj, base_drv = retarded_integrator(
            config.steps,
            config.time_step,
            config.wall_position,
            config.aperture_radius,
            config.simulation_type,
            init_rider,
            init_driver,
            config.bunch_mean,
            config.cavity_spacing,
            config.z_cutoff,
            opts.self_consistency,
        )
        base_elapsed = time.time() - start

        start = time.time()
        numba_traj, numba_drv = retarded_integrator_numba(
            config.steps,
            config.time_step,
            config.wall_position,
            config.aperture_radius,
            config.simulation_type,
            init_rider,
            init_driver,
            config.bunch_mean,
            config.cavity_spacing,
            config.z_cutoff,
            opts.self_consistency,
        )
        numba_elapsed = time.time() - start

        if numba_elapsed > 0:
            print(f"Benchmark speedup: {base_elapsed / numba_elapsed:.2f}x")
        else:
            print("Benchmark speedup: instantaneous (numba_elapsed≈0)")

        return numba_traj, numba_drv

    if use_numba:
        return retarded_integrator_numba(
            config.steps,
            config.time_step,
            config.wall_position,
            config.aperture_radius,
            config.simulation_type,
            init_rider,
            init_driver,
            config.bunch_mean,
            config.cavity_spacing,
            config.z_cutoff,
            opts.self_consistency,
        )

    return retarded_integrator(
        config.steps,
        config.time_step,
        config.wall_position,
        config.aperture_radius,
        config.simulation_type,
        init_rider,
        init_driver,
        config.bunch_mean,
        config.cavity_spacing,
        config.z_cutoff,
        opts.self_consistency,
    )


__all__ = [
    "OptimisationOptions",
    "dict_to_arrays",
    "arrays_to_dict",
    "eqsofmotion_retarded_numba",
    "retarded_integrator_numba",
    "run_optimised_integrator",
]
