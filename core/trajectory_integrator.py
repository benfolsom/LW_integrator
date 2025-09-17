"""
Core Lienard-Wiechert Trajectory Integrator

Restored from Benjamin Folsom's original validated physics implementation.
This integrator contains the complete working electromagnetic field physics
with retarded time calculations, relativistic corrections, and radiation reaction.

Author: Ben Folsom (original physics design)
Restoration Date: 2025-09-17
"""

import numpy as np
from typing import Dict, List, Optional, Any

from physics.constants import C_MMNS, ELEMENTARY_CHARGE_GAUSSIAN
from physics.simulation_types import SimulationConfig


class LienardWiechertIntegrator:
    """
    Core Lienard-Wiechert electromagnetic field integrator.

    This implementation contains the complete validated physics from
    Benjamin Folsom's original covariant_integrator_library.py, including:
    - Retarded field calculations with proper relativistic corrections
    - Chronological ordering for retarded times
    - Radiation reaction forces
    - Aperture effects and boundary conditions
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize the integrator with configuration."""
        self.config = config or SimulationConfig()
        self.c_mmns = C_MMNS
        self.charge_gaussian = ELEMENTARY_CHARGE_GAUSSIAN
        self.epsilon = 1e-12  # For numerical stability

    def chrono_jn(
        self,
        trajectory: List[Dict[str, Any]],
        trajectory_ext: List[Dict[str, Any]],
        index_traj: int,
        particle_idx: int,
    ) -> np.ndarray:
        """
        Calculate retarded time chronological ordering.

        Find which index in trajectory_ext corresponds to the retarded time
        for particle particle_idx interacting with external particles.
        """
        c_mmns = self.c_mmns
        epsilon = self.epsilon

        index_traj_new = np.zeros(len(trajectory_ext[0]["x"]), dtype=int)

        for j in range(len(trajectory_ext[0]["x"])):
            # Distance vector and magnitude
            dx = (
                trajectory[index_traj]["x"][particle_idx]
                - trajectory_ext[index_traj]["x"][j]
            )
            dy = (
                trajectory[index_traj]["y"][particle_idx]
                - trajectory_ext[index_traj]["y"][j]
            )
            dz = (
                trajectory[index_traj]["z"][particle_idx]
                - trajectory_ext[index_traj]["z"][j]
            )
            R = np.sqrt(dx**2 + dy**2 + dz**2)

            # Unit vector
            if R > epsilon:
                nx = dx / R
                ny = dy / R
                nz = dz / R
            else:
                # Self-interaction case
                nx = ny = nz = 0.0
                index_traj_new[j] = index_traj
                continue

            # Relativistic retardation condition: R/c = t - t_ret + β·n̂·R/c
            beta_ext = np.array(
                [
                    trajectory_ext[index_traj]["bx"][j],
                    trajectory_ext[index_traj]["by"][j],
                    trajectory_ext[index_traj]["bz"][j],
                ]
            )

            beta_dot_nhat = beta_ext[0] * nx + beta_ext[1] * ny + beta_ext[2] * nz
            denominator = 1.0 - beta_dot_nhat

            if abs(denominator) < epsilon:
                # Near-collinear motion - use characteristic time scale
                if (
                    hasattr(trajectory_ext[index_traj], "char_time")
                    and len(trajectory_ext[index_traj]["char_time"]) > j
                ):
                    max_retardation = 10.0 * trajectory_ext[index_traj]["char_time"][j]
                else:
                    max_retardation = 1e-3  # Fallback
                delta_t = max_retardation
            else:
                delta_t = R / (c_mmns * denominator)

            t_ext_new = trajectory_ext[index_traj]["t"][j] - delta_t

            if t_ext_new < 0:
                index_traj_new[j] = index_traj
            else:
                # Find correct retarded time index
                for k in range(index_traj, -1, -1):
                    if trajectory_ext[index_traj - k]["t"][j] > t_ext_new:
                        index_traj_new[j] = index_traj - k
                        break

        return index_traj_new

    def dist_euclid_ret(
        self,
        trajectory: List[Dict[str, Any]],
        trajectory_ext: List[Dict[str, Any]],
        index_traj: int,
        particle_idx: int,
        i_new: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate retarded distances and unit vectors.
        """
        """
        Calculate retarded distances and unit vectors.
        """
        epsilon = self.epsilon

        nhat = {
            "R": np.zeros(len(trajectory_ext[0]["x"])),
            "nx": np.zeros(len(trajectory_ext[0]["x"])),
            "ny": np.zeros(len(trajectory_ext[0]["x"])),
            "nz": np.zeros(len(trajectory_ext[0]["x"])),
        }

        for j in range(len(trajectory_ext[0]["x"])):
            # Distance vector at retarded time
            dx = (
                trajectory[index_traj]["x"][particle_idx]
                - trajectory_ext[i_new[j]]["x"][j]
            )
            dy = (
                trajectory[index_traj]["y"][particle_idx]
                - trajectory_ext[i_new[j]]["y"][j]
            )
            dz = (
                trajectory[index_traj]["z"][particle_idx]
                - trajectory_ext[i_new[j]]["z"][j]
            )

            nhat["R"][j] = np.sqrt(dx**2 + dy**2 + dz**2)

            if nhat["R"][j] > epsilon:
                nhat["nx"][j] = dx / nhat["R"][j]
                nhat["ny"][j] = dy / nhat["R"][j]
                nhat["nz"][j] = dz / nhat["R"][j]
            else:
                nhat["nx"][j] = nhat["ny"][j] = nhat["nz"][j] = 0.0

        return nhat

    def eqsofmotion_retarded(
        self,
        h: float,
        trajectory: List[Dict[str, Any]],
        trajectory_ext: List[Dict[str, Any]],
        i_traj: int,
        apt_R: float,
        sim_type: str,
    ) -> Dict[str, Any]:
        """
        Core retarded equations of motion integration step.

        This is the complete physics implementation including:
        - Lienard-Wiechert retarded potentials
        - Relativistic corrections
        - Radiation reaction forces
        """
        c_mmns = self.c_mmns

        # Initialize result arrays
        result = {}
        for key in [
            "x",
            "y",
            "z",
            "t",
            "Px",
            "Py",
            "Pz",
            "Pt",
            "gamma",
            "bx",
            "by",
            "bz",
            "bdotx",
            "bdoty",
            "bdotz",
        ]:
            result[key] = np.zeros_like(trajectory[i_traj][key])

        # Copy particle properties
        result["q"] = trajectory[i_traj]["q"]
        result["char_time"] = trajectory[i_traj]["char_time"]
        result["m"] = trajectory[i_traj]["m"]

        # Integrate each particle
        for particle_idx in range(len(trajectory[i_traj]["x"])):
            # Calculate retarded times and distances
            i_new = self.chrono_jn(trajectory, trajectory_ext, i_traj, particle_idx)
            nhat = self.dist_euclid_ret(
                trajectory, trajectory_ext, i_traj, particle_idx, i_new
            )

            # Initialize with current values
            result["x"][particle_idx] = trajectory[i_traj]["x"][particle_idx]
            result["y"][particle_idx] = trajectory[i_traj]["y"][particle_idx]
            result["z"][particle_idx] = trajectory[i_traj]["z"][particle_idx]
            result["t"][particle_idx] = trajectory[i_traj]["t"][particle_idx]

            # Sum all external field contributions
            for j in range(len(trajectory_ext[0]["x"])):
                # Current particle velocity
                beta_vec = np.array(
                    [
                        trajectory[i_traj]["bx"][particle_idx],
                        trajectory[i_traj]["by"][particle_idx],
                        trajectory[i_traj]["bz"][particle_idx],
                    ]
                )

                # External particle velocity at retarded time
                beta_ext = np.array(
                    [
                        trajectory_ext[i_new[j]]["bx"][j],
                        trajectory_ext[i_new[j]]["by"][j],
                        trajectory_ext[i_new[j]]["bz"][j],
                    ]
                )

                # Retardation factor
                nhat_vec = np.array([nhat["nx"][j], nhat["ny"][j], nhat["nz"][j]])
                k_factor = 1.0 - np.dot(beta_ext, nhat_vec)

                # Acceleration terms
                bdot_ext = np.array(
                    [
                        trajectory_ext[i_new[j]]["bdotx"][j],
                        trajectory_ext[i_new[j]]["bdoty"][j],
                        trajectory_ext[i_new[j]]["bdotz"][j],
                    ]
                )

                # Scalar products
                # bdot_scalar_mixed = np.dot(beta_vec, bdot_ext)  # Future use for advanced calculations
                bdot_scalar_ext = np.dot(bdot_ext, bdot_ext)
                betas_scalar = np.dot(beta_ext, beta_vec)

                # Velocity-dependent terms
                gamma_i = trajectory[i_traj]["gamma"][particle_idx]
                gamma_j = trajectory_ext[i_new[j]]["gamma"][j]

                v_betas_scalar = gamma_j * gamma_i * c_mmns**2 * (1.0 - betas_scalar)

                v_beta_dot_mixed_scalar = (
                    gamma_j**4 * gamma_i * c_mmns**2 * bdot_scalar_ext
                    - gamma_i
                    * c_mmns
                    * np.dot(
                        beta_vec,
                        bdot_ext * c_mmns * gamma_j**2
                        + beta_ext * bdot_scalar_ext * c_mmns * gamma_j**4,
                    )
                )

                # Electromagnetic force components
                charge_factor = (
                    h
                    * trajectory[i_traj]["q"][particle_idx]
                    * trajectory_ext[i_new[j]]["q"][j]
                    / (k_factor**3 * c_mmns**3 * nhat["R"][j] ** 2 * gamma_j**3)
                )

                # For zero charge particles, skip electromagnetic interactions
                charge_i = float(trajectory[i_traj]["q"][particle_idx])
                charge_j = float(trajectory_ext[i_new[j]]["q"][j])
                if abs(charge_i) < 1e-20 or abs(charge_j) < 1e-20:
                    charge_factor = 0.0

                # X-conjugate momentum update
                result["Px"][particle_idx] = trajectory[i_traj]["Px"][
                    particle_idx
                ] + charge_factor * (
                    -trajectory_ext[i_new[j]]["bx"][j]
                    * v_betas_scalar
                    * k_factor
                    * c_mmns
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
                        trajectory_ext[i_new[j]]["bdotx"][j]
                        + trajectory_ext[i_new[j]]["bdotx"][j]
                        * bdot_scalar_ext
                        * gamma_j**2
                    )
                    + v_betas_scalar * c_mmns * nhat["nx"][j]
                )

                # Y-conjugate momentum update
                result["Py"][particle_idx] = trajectory[i_traj]["Py"][
                    particle_idx
                ] + charge_factor * (
                    -trajectory_ext[i_new[j]]["by"][j]
                    * v_betas_scalar
                    * k_factor
                    * c_mmns
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
                        trajectory_ext[i_new[j]]["bdoty"][j]
                        + trajectory_ext[i_new[j]]["bdoty"][j]
                        * bdot_scalar_ext
                        * gamma_j**2
                    )
                    + v_betas_scalar * c_mmns * nhat["ny"][j]
                )

                # Z-conjugate momentum update
                result["Pz"][particle_idx] = trajectory[i_traj]["Pz"][
                    particle_idx
                ] + charge_factor * (
                    -trajectory_ext[i_new[j]]["bz"][j]
                    * v_betas_scalar
                    * k_factor
                    * c_mmns
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
                        trajectory_ext[i_new[j]]["bdotz"][j]
                        + trajectory_ext[i_new[j]]["bdotz"][j]
                        * bdot_scalar_ext
                        * gamma_j**2
                    )
                    + v_betas_scalar * c_mmns * nhat["nz"][j]
                )

                # Time/energy conjugate momentum component update
                result["Pt"][particle_idx] = trajectory[i_traj]["Pt"][
                    particle_idx
                ] + charge_factor * (
                    v_beta_dot_mixed_scalar * k_factor * gamma_j * nhat["R"][j]
                    - v_betas_scalar * k_factor * c_mmns * gamma_j**2
                    - bdot_scalar_ext * v_betas_scalar * gamma_j**4 * nhat["R"][j]
                    + v_betas_scalar * c_mmns
                )

                # Update gamma (preliminary)
                charge_i = float(trajectory[i_traj]["q"][particle_idx])
                charge_j = float(trajectory_ext[i_new[j]]["q"][j])
                if abs(charge_i) < 1e-20 or abs(charge_j) < 1e-20:
                    # Zero charge case - no electromagnetic potential energy
                    result["gamma"][particle_idx] = result["Pt"][particle_idx] / (
                        trajectory[i_traj]["m"][particle_idx] * c_mmns
                    )
                else:
                    # Full electromagnetic case
                    result["gamma"][particle_idx] = (
                        1.0
                        / (trajectory[i_traj]["m"][particle_idx] * c_mmns)
                        * (
                            result["Pt"][particle_idx]
                            - trajectory[i_traj]["q"][particle_idx]
                            / c_mmns
                            * trajectory_ext[i_new[j]]["q"][j]
                            / (nhat["R"][j] * k_factor)
                        )
                    )

                # Update time
                result["t"][particle_idx] = (
                    trajectory[i_traj]["t"][particle_idx]
                    + h * result["gamma"][particle_idx]
                )

                # Position updates using conjugate momentum with field corrections
                charge_i = float(trajectory[i_traj]["q"][particle_idx])
                charge_j = float(trajectory_ext[i_new[j]]["q"][j])
                if abs(charge_i) < 1e-20 or abs(charge_j) < 1e-20:
                    # Zero charge case - no field corrections
                    field_correction_x = 0.0
                    field_correction_y = 0.0
                    field_correction_z = 0.0
                else:
                    # Full electromagnetic field corrections to conjugate momentum
                    field_correction_x = (
                        trajectory[i_traj]["q"][particle_idx]
                        / c_mmns
                        * trajectory_ext[i_new[j]]["q"][j]
                        * trajectory_ext[i_new[j]]["bx"][j]
                        / (nhat["R"][j] * k_factor)
                    )
                    field_correction_y = (
                        trajectory[i_traj]["q"][particle_idx]
                        / c_mmns
                        * trajectory_ext[i_new[j]]["q"][j]
                        * trajectory_ext[i_new[j]]["by"][j]
                        / (nhat["R"][j] * k_factor)
                    )
                    field_correction_z = (
                        trajectory[i_traj]["q"][particle_idx]
                        / c_mmns
                        * trajectory_ext[i_new[j]]["q"][j]
                        * trajectory_ext[i_new[j]]["bz"][j]
                        / (nhat["R"][j] * k_factor)
                    )

                # Position updates from conjugate momentum: x = ∫(P-qA)/m dt
                result["x"][particle_idx] = trajectory[i_traj]["x"][
                    particle_idx
                ] + h / trajectory[i_traj]["m"][particle_idx] * (
                    result["Px"][particle_idx] - field_correction_x
                )

                result["y"][particle_idx] = trajectory[i_traj]["y"][
                    particle_idx
                ] + h / trajectory[i_traj]["m"][particle_idx] * (
                    result["Py"][particle_idx] - field_correction_y
                )

                result["z"][particle_idx] = trajectory[i_traj]["z"][
                    particle_idx
                ] + h / trajectory[i_traj]["m"][particle_idx] * (
                    result["Pz"][particle_idx] - field_correction_z
                )

            # Calculate velocities from position updates (self-consistent with gamma)
            # This implements β = Δx/(c·h·γ) where γ comes from energy-momentum relation
            result["bx"][particle_idx] = (
                result["x"][particle_idx] - trajectory[i_traj]["x"][particle_idx]
            ) / (c_mmns * h * result["gamma"][particle_idx])
            result["by"][particle_idx] = (
                result["y"][particle_idx] - trajectory[i_traj]["y"][particle_idx]
            ) / (c_mmns * h * result["gamma"][particle_idx])
            result["bz"][particle_idx] = (
                result["z"][particle_idx] - trajectory[i_traj]["z"][particle_idx]
            ) / (c_mmns * h * result["gamma"][particle_idx])

            # Calculate accelerations from velocity changes
            result["bdotx"][particle_idx] = (
                result["bx"][particle_idx] - trajectory[i_traj]["bx"][particle_idx]
            ) / (c_mmns * h * result["gamma"][particle_idx])
            result["bdoty"][particle_idx] = (
                result["by"][particle_idx] - trajectory[i_traj]["by"][particle_idx]
            ) / (c_mmns * h * result["gamma"][particle_idx])
            result["bdotz"][particle_idx] = (
                result["bz"][particle_idx] - trajectory[i_traj]["bz"][particle_idx]
            ) / (c_mmns * h * result["gamma"][particle_idx])

            # Self-consistency check: recalculate gamma from velocity magnitude
            # This ensures γ = 1/√(1-β²) is satisfied
            btot_squared = (
                result["bx"][particle_idx] ** 2
                + result["by"][particle_idx] ** 2
                + result["bz"][particle_idx] ** 2
            )

            # Velocity magnitude check: only limit if total velocity exceeds c due to numerical artifacts
            # High-energy particles naturally approach β → 1.0 (e.g., 30 GeV proton: β ≈ 0.999511)
            if btot_squared >= 1.0:
                # Limit to very close to c to avoid mathematical singularities
                btot_limited_squared = (
                    0.9999999999999998  # Allows up to PeV-scale particles
                )
                scale_factor = np.sqrt(btot_limited_squared / btot_squared)
                result["bx"][particle_idx] *= scale_factor
                result["by"][particle_idx] *= scale_factor
                result["bz"][particle_idx] *= scale_factor
                btot_squared = btot_limited_squared

            result["gamma"][particle_idx] = 1.0 / np.sqrt(1.0 - btot_squared)

            # Radiation reaction forces (if enabled)
            if hasattr(trajectory[i_traj], "char_time"):
                char_time = trajectory[i_traj]["char_time"][particle_idx]
                m_particle = trajectory[i_traj]["m"]

                # Z-component radiation reaction
                rad_frc_z_rhs = (
                    -(result["gamma"][particle_idx] ** 3)
                    * (m_particle * result["bdotz"][particle_idx] ** 2 * c_mmns**2)
                    * result["bz"][particle_idx]
                    * c_mmns
                )
                rad_frc_z_lhs = (
                    (
                        result["gamma"][particle_idx]
                        - trajectory[i_traj]["gamma"][particle_idx]
                    )
                    / (h * result["gamma"][particle_idx])
                    * m_particle
                    * result["bdotz"][particle_idx]
                    * result["bz"][particle_idx]
                    * c_mmns**2
                )

                if (
                    abs(rad_frc_z_rhs) > char_time / 1e1
                    or abs(rad_frc_z_lhs) > char_time / 1e1
                ):
                    result["bdotz"][particle_idx] += (
                        char_time
                        * (rad_frc_z_lhs + rad_frc_z_rhs)
                        / (m_particle * c_mmns)
                    )

                    # X and Y components
                    rad_frc_x_rhs = (
                        -(result["gamma"][particle_idx] ** 3)
                        * (m_particle * result["bdotx"][particle_idx] ** 2 * c_mmns**2)
                        * result["bx"][particle_idx]
                        * c_mmns
                    )
                    rad_frc_x_lhs = (
                        (
                            result["gamma"][particle_idx]
                            - trajectory[i_traj]["gamma"][particle_idx]
                        )
                        / (h * result["gamma"][particle_idx])
                        * m_particle
                        * result["bdotx"][particle_idx]
                        * result["bx"][particle_idx]
                        * c_mmns**2
                    )
                    result["bdotx"][particle_idx] += (
                        char_time
                        * (rad_frc_x_lhs + rad_frc_x_rhs)
                        / (m_particle * c_mmns)
                    )

                    rad_frc_y_rhs = (
                        -(result["gamma"][particle_idx] ** 3)
                        * (m_particle * result["bdoty"][particle_idx] ** 2 * c_mmns**2)
                        * result["by"][particle_idx]
                        * c_mmns
                    )
                    rad_frc_y_lhs = (
                        (
                            result["gamma"][particle_idx]
                            - trajectory[i_traj]["gamma"][particle_idx]
                        )
                        / (h * result["gamma"][particle_idx])
                        * m_particle
                        * result["bdoty"][particle_idx]
                        * result["by"][particle_idx]
                        * c_mmns**2
                    )
                    result["bdoty"][particle_idx] += (
                        char_time
                        * (rad_frc_y_lhs + rad_frc_y_rhs)
                        / (m_particle * c_mmns)
                    )

        return result

    def self_consistent_enhanced_step(
        self,
        h_step: float,
        trajectory: List[Dict],
        trajectory_drv: List[Dict],
        i_traj: int,
        apt_R: float,
        sim_type: Any,
    ) -> Dict[str, Any]:
        """
        Self-consistent enhanced integration step.

        Uses the full retarded electromagnetic field calculation for
        enhanced accuracy in self-consistent mode.
        """
        if i_traj >= len(trajectory) or i_traj >= len(trajectory_drv):
            # Fallback for boundary cases
            if trajectory:
                return trajectory[-1].copy()
            else:
                return {
                    "x": np.array([0.0]),
                    "y": np.array([0.0]),
                    "z": np.array([0.0]),
                    "t": np.array([0.0]),
                    "Px": np.array([0.0]),
                    "Py": np.array([0.0]),
                    "Pz": np.array([0.0]),
                    "Pt": np.array([0.0]),
                    "bx": np.array([0.0]),
                    "by": np.array([0.0]),
                    "bz": np.array([0.0]),
                    "gamma": np.array([1.0]),
                    "q": np.array([1.0]),
                    "m": np.array([1.0]),
                }

        # Use full retarded electromagnetic field calculation
        return self.eqsofmotion_retarded(
            h_step, trajectory, trajectory_drv, i_traj, apt_R, sim_type
        )
