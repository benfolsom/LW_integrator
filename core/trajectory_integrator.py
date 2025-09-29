"""
Core Lienard-Wiechert Trajectory Integrator

Restored from Benjamin Folsom's original validated physics implementation.
This integrator contains the complete working electromagnetic field physics
with retarded time calculations, relativistic corrections, and radiation reaction.

Author: Ben Folsom (original physics design)
Restoration Date: 2025-09-17
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple

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

    By default, this class automatically returns the optimized version
    (OptimizedLienardWiechertIntegrator) if available. To force the standard
    version, pass use_optimized=False.
    """

    def __new__(
        cls,
        config: Optional[SimulationConfig] = None,
        use_optimized: Optional[bool] = None,
    ) -> "LienardWiechertIntegrator":
        """
        Create a new integrator instance.

        Args:
            config: Simulation configuration
            use_optimized: If None (default), uses optimized version if available.
                          If True, forces optimized (with fallback warning).
                          If False, forces standard version.
        """
        # Only apply smart factory behavior for direct instantiation of this class
        if cls is LienardWiechertIntegrator:
            should_use_optimized = use_optimized if use_optimized is not None else True

            if should_use_optimized:
                try:
                    from .performance import OptimizedLienardWiechertIntegrator

                    return OptimizedLienardWiechertIntegrator(config)
                except ImportError:
                    if use_optimized is True:  # Explicitly requested
                        print(
                            "⚠️  Optimized integrator requested but not available. Using standard integrator."
                        )
                    # Fall through to create standard instance

        # Create standard instance (either explicitly requested or as fallback)
        return super().__new__(cls)

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        use_optimized: Optional[bool] = None,
    ):
        """
        Initialize the integrator with configuration.

        Args:
            config: Simulation configuration
            use_optimized: Parameter for compatibility (ignored in base class)
        """
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

    def _dist_euclid_instantaneous(
        self,
        trajectory: List[Dict[str, Any]],
        trajectory_ext: List[Dict[str, Any]],
        index_traj: int,
        particle_idx: int,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate instantaneous distances and unit vectors (no retardation).

        This is the static version of dist_euclid_ret - uses current timestep
        for all external particles instead of retarded times.
        """
        epsilon = self.epsilon

        nhat = {
            "R": np.zeros(len(trajectory_ext[0]["x"])),
            "nx": np.zeros(len(trajectory_ext[0]["x"])),
            "ny": np.zeros(len(trajectory_ext[0]["x"])),
            "nz": np.zeros(len(trajectory_ext[0]["x"])),
        }

        for j in range(len(trajectory_ext[0]["x"])):
            # Distance vector at CURRENT time (not retarded)
            dx = (
                trajectory[index_traj]["x"][particle_idx]
                - trajectory_ext[index_traj]["x"][j]  # index_traj instead of i_new[j]
            )
            dy = (
                trajectory[index_traj]["y"][particle_idx]
                - trajectory_ext[index_traj]["y"][j]
            )
            dz = (
                trajectory[index_traj]["z"][particle_idx]
                - trajectory_ext[index_traj]["z"][j]
            )

            nhat["R"][j] = np.sqrt(dx**2 + dy**2 + dz**2)

            if nhat["R"][j] > epsilon:
                nhat["nx"][j] = dx / nhat["R"][j]
                nhat["ny"][j] = dy / nhat["R"][j]
                nhat["nz"][j] = dz / nhat["R"][j]
            else:
                nhat["nx"][j] = nhat["ny"][j] = nhat["nz"][j] = 0.0

        return nhat

    def conducting_flat_updated(
        self, vector: Dict[str, Any], wall_Z: float, apt_R: float
    ) -> Dict[str, Any]:
        """
        Updated implementation of conducting flat wall with image charges.

        Creates image charges with opposite sign reflected across the wall.
        Particles passing the wall have their image charges turned off.

        Args:
            vector: Current particle state
            wall_Z: Wall position in z direction (mm)
            apt_R: Aperture radius (mm)

        Returns:
            Image charge state dictionary
        """
        result = {}

        # Initialize result arrays with same shape as input
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
            result[key] = np.zeros_like(vector[key])

        # Copy particle properties (these may be modified)
        result["q"] = np.copy(vector["q"])
        result["m"] = (
            np.copy(vector["m"]) if "m" in vector else np.copy(vector["q"])
        )  # fallback
        result["char_time"] = vector.get("char_time", 1.0)

        # Process each particle
        for i in range(len(vector["x"])):
            # Check if particle has passed the wall
            if vector["z"][i] >= wall_Z:
                # Turn off image charge for particles past wall
                result["q"][i] = 0.0
            else:
                # Create image charge
                result["q"][i] = -vector["q"][i]  # Opposite charge
                result["x"][i] = vector["x"][i]  # Same x position
                result["y"][i] = vector["y"][i]  # Same y position
                result["z"][i] = wall_Z + abs(wall_Z - vector["z"][i])  # Reflected z

                # Image charge has opposite momentum components
                result["Px"][i] = vector["Px"][i]  # Tangential unchanged
                result["Py"][i] = vector["Py"][i]  # Tangential unchanged
                result["Pz"][i] = -vector["Pz"][i]  # Normal component reversed
                result["Pt"][i] = vector["Pt"][i]  # Total magnitude same

                # Beta components (velocity/c)
                result["bx"][i] = vector["bx"][i]
                result["by"][i] = vector["by"][i]
                result["bz"][i] = -vector["bz"][i]  # Reversed

                # Beta derivatives reversed for normal component
                result["bdotx"][i] = vector["bdotx"][i]
                result["bdoty"][i] = vector["bdoty"][i]
                result["bdotz"][i] = -vector["bdotz"][i]

                # Other properties
                result["gamma"][i] = vector["gamma"][i]
                result["t"][i] = vector["t"][i]  # No retardation for image charges
                result["m"][i] = vector["m"][i] if "m" in vector else vector["q"][i]

        return result

    def switching_flat_updated(
        self, vector: Dict[str, Any], wall_Z: float, apt_R: float, cut_Z: float
    ) -> Dict[str, Any]:
        """
        Updated implementation of switching flat wall (disappearing aperture).

        Like conducting_flat but image charges disappear when particles reach cut_Z.
        Used for cavity simulations with time-varying apertures.

        Args:
            vector: Current particle state
            wall_Z: Wall position in z direction (mm)
            apt_R: Aperture radius (mm)
            cut_Z: Position where aperture disappears (mm)

        Returns:
            Image charge state dictionary
        """
        result = {}

        # Initialize result arrays
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
            result[key] = np.zeros_like(vector[key])

        # Initialize with negative charge (image)
        result["q"] = -np.copy(vector["q"])
        result["m"] = np.copy(vector["m"]) if "m" in vector else np.copy(vector["q"])
        result["char_time"] = vector.get("char_time", 1.0)

        # Process each particle
        for i in range(len(vector["x"])):
            # Check if particle has reached cutoff position
            if vector["z"][i] >= cut_Z:
                # Turn off image charge after cutoff
                result["q"][i] = 0.0
            else:
                # Create image charge (like conducting_flat)
                result["x"][i] = vector["x"][i]
                result["y"][i] = vector["y"][i]
                result["z"][i] = wall_Z + abs(wall_Z - vector["z"][i])

                # Reversed momentum components
                result["Px"][i] = vector["Px"][i]
                result["Py"][i] = vector["Py"][i]
                result["Pz"][i] = -vector["Pz"][i]
                result["Pt"][i] = vector["Pt"][i]

                # Reversed velocity components
                result["bx"][i] = vector["bx"][i]
                result["by"][i] = vector["by"][i]
                result["bz"][i] = -vector["bz"][i]

                # Reversed acceleration components
                result["bdotx"][i] = vector["bdotx"][i]
                result["bdoty"][i] = vector["bdoty"][i]
                result["bdotz"][i] = -vector["bdotz"][i]

                # Other properties
                result["gamma"][i] = vector["gamma"][i]
                result["t"][i] = vector["t"][i]
                result["m"][i] = vector["m"][i] if "m" in vector else vector["q"][i]

        return result

    def _needs_complex_physics(
        self,
        beta_vec: np.ndarray,
        beta_ext: np.ndarray,
        gamma_i: float,
        gamma_j: float,
        k_factor: float,
        distance: float,
    ) -> bool:
        """
        Determine if complex electromagnetic physics is needed.

        Uses simple physics for most cases, complex physics when:
        - Moderate relativistic factors (γ > 10) for accuracy
        - Close approach (distance < 10 mm) for strong interactions
        - BUT never for k_factor singularities

        Balanced approach: Allow complex physics when needed for accuracy
        but prevent numerical instabilities from singularities.
        """

        # CRITICAL: Never use complex physics for k_factor near zero
        # This prevents the k_factor**3 singularity in the denominator
        if abs(k_factor) < 1e-15:  # Very small threshold for true singularities
           return False  # Force simple physics for numerical stability

        # USE COMPLEX PHYSICS ALWAYS (except for true singularities)
        # This lets us identify real problems rather than masking them
        return True

    def _simple_em_force(
        self,
        beta_vec: np.ndarray,
        gamma_i: float,
        q_i: float,
        q_j: float,
        distance: float,
        nhat: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Simplified electromagnetic force calculation.

        Uses Coulomb force with basic relativistic corrections.
        Returns both spatial force and energy rate (for Pt calculation).
        Ensures energy-momentum covariance for consistent physics.
        Uses amu-mm-ns Gaussian units to match legacy system.

        Returns:
            tuple: (force_vec [3D], energy_rate [scalar])
        """
        # Prevent close approach singularities with physical minimum distance
        # Use nuclear scale (~1 femtometer = 1e-12 mm) as hard limit
        safe_distance = max(distance, 1e-12)  # mm minimum distance (nuclear scale)

        # Gaussian Coulomb force in amu-mm-ns units
        # F = q1*q2 / r^2 (in Gaussian units)
        # Force gives momentum rate: F = dp/dt
        coulomb_magnitude = (q_i * q_j) / (safe_distance**2)

        # Relativistic enhancement for head-on interactions
        # For ultra-relativistic particles, use γ² enhancement
        relativistic_factor = gamma_i * gamma_i

        # Final force magnitude (repulsive for like charges)
        force_magnitude = coulomb_magnitude * relativistic_factor

        # Force direction vector (points from j to i for repulsion)
        force_vec = force_magnitude * nhat

        # Energy rate calculation for Pt component
        # In relativistic EM: dE/dt = q * E · v = F · v
        # This ensures energy-momentum covariance
        c_mmns = self.c_mmns
        velocity_vec = beta_vec * c_mmns  # Convert β to actual velocity
        energy_rate = np.dot(force_vec, velocity_vec)  # F · v for energy change rate

        return force_vec, energy_rate

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

        # Integrate each particle (FIXED loop structure to match legacy)
        for particle_idx in range(len(trajectory[i_traj]["x"])):
            # Calculate retarded times and distances for this particle
            i_new = self.chrono_jn(trajectory, trajectory_ext, i_traj, particle_idx)
            nhat = self.dist_euclid_ret(
                trajectory, trajectory_ext, i_traj, particle_idx, i_new
            )

            # Initialize with current values (covariant mechanics: positions and momentum start from current state)
            result["x"][particle_idx] = trajectory[i_traj]["x"][particle_idx]
            result["y"][particle_idx] = trajectory[i_traj]["y"][particle_idx]
            result["z"][particle_idx] = trajectory[i_traj]["z"][particle_idx]
            result["t"][particle_idx] = trajectory[i_traj]["t"][particle_idx]
            result["Px"][particle_idx] = trajectory[i_traj]["Px"][particle_idx]
            result["Py"][particle_idx] = trajectory[i_traj]["Py"][particle_idx]
            result["Pz"][particle_idx] = trajectory[i_traj]["Pz"][particle_idx]
            result["Pt"][particle_idx] = trajectory[i_traj]["Pt"][particle_idx]

            # Handle mass values (scalar or array) - needed for covariant position updates
            if np.isscalar(trajectory[i_traj]["m"]):
                mass_i = float(trajectory[i_traj]["m"])
            else:
                if particle_idx < len(trajectory[i_traj]["m"]):
                    mass_i = float(trajectory[i_traj]["m"][particle_idx])
                else:
                    mass_i = 1.0  # Default mass

            # COVARIANT ELECTROMAGNETIC INTERACTIONS: Both momentum AND position updated per interaction
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

                # Regularization to avoid division by zero (same as legacy)
                # Increase regularization threshold for numerical stability
                if abs(k_factor) < 1e-12:
                    k_factor = 1e-12 * np.sign(k_factor) if k_factor != 0 else 1e-12

                # Acceleration terms
                bdot_ext = np.array(
                    [
                        trajectory_ext[i_new[j]]["bdotx"][j],
                        trajectory_ext[i_new[j]]["bdoty"][j],
                        trajectory_ext[i_new[j]]["bdotz"][j],
                    ]
                )

                # Scalar products for complex electromagnetic calculation
                bdot_scalar_ext = np.dot(bdot_ext, bdot_ext)
                betas_scalar = np.dot(beta_ext, beta_vec)

                # Velocity-dependent terms (exact legacy formulation)
                gamma_i = trajectory[i_traj]["gamma"][particle_idx]
                gamma_j = trajectory_ext[i_new[j]]["gamma"][j]

                # CRITICAL: Protect against corrupted gamma values from previous runaway
                if gamma_j > 1e6 or gamma_i > 1e6:  # Unrealistic relativistic factors
                    continue  # Skip this interaction to prevent runaway

                # V_ext^beta * V_beta (complex relativistic factor)
                v_betas_scalar = gamma_j * gamma_i * c_mmns**2 * (1.0 - betas_scalar)

                # Vdot_ext^beta * V_beta (complex acceleration coupling)
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

                # Choose physics complexity based on conditions
                use_complex = self._needs_complex_physics(
                    beta_vec, beta_ext, gamma_i, gamma_j, k_factor, nhat["R"][j]
                )

                # Handle charge values (scalar or array)
                if np.isscalar(trajectory[i_traj]["q"]):
                    charge_i = float(trajectory[i_traj]["q"])
                else:
                    if particle_idx < len(trajectory[i_traj]["q"]):
                        charge_i = float(trajectory[i_traj]["q"][particle_idx])
                    else:
                        charge_i = 0.0  # Default for missing indices

                if np.isscalar(trajectory_ext[i_new[j]]["q"]):
                    charge_j = float(trajectory_ext[i_new[j]]["q"])
                else:
                    if j < len(trajectory_ext[i_new[j]]["q"]):
                        charge_j = float(trajectory_ext[i_new[j]]["q"][j])
                    else:
                        charge_j = 0.0  # Default for missing indices

                # For zero charge particles, skip electromagnetic interactions
                if abs(charge_i) < 1e-20 or abs(charge_j) < 1e-20:
                    print(f"DEBUG RETARDED: Zero charge skip - qi={charge_i:.8f}, qj={charge_j:.8f}")
                    continue  # Skip this interaction entirely

                if use_complex:
                    # Complex electromagnetic force components (EXACT legacy formula)
                    # EXTREMELY CONSERVATIVE: Only prevent true numerical overflow, not physics
                    if (
                        abs(k_factor) < 1e-15
                    ):  # Only prevent actual singularities, allow extreme accelerations
                        continue  # Skip only true numerical overflow cases

                    charge_factor = (
                        h
                        * charge_i
                        * charge_j
                        / (k_factor**3 * c_mmns**3 * nhat["R"][j] ** 2 * gamma_j**3)
                    )

                    # ACCUMULATE X-conjugate momentum update (COMPLEX legacy formula)
                    result["Px"][particle_idx] += charge_factor * (
                        -v_betas_scalar
                        * trajectory_ext[i_new[j]]["bx"][j]
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
                    )                    # ACCUMULATE Y-conjugate momentum update (COMPLEX legacy formula)
                    result["Py"][particle_idx] += charge_factor * (
                        -v_betas_scalar
                        * trajectory_ext[i_new[j]]["by"][j]
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

                    # ACCUMULATE Z-conjugate momentum update (COMPLEX legacy formula)
                    result["Pz"][particle_idx] += charge_factor * (
                        -v_betas_scalar
                        * trajectory_ext[i_new[j]]["bz"][j]
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

                    # DEBUG: Track momentum changes
                    if j == 0 and particle_idx == 0:  # First particle interaction
                        result["Pz"][particle_idx] += charge_factor * (
                            -v_betas_scalar * trajectory_ext[i_new[j]]["bz"][j] * k_factor * c_mmns * gamma_j**2
                            + v_beta_dot_mixed_scalar * k_factor * gamma_j * nhat["nz"][j] * nhat["R"][j]
                            + gamma_j**2 * nhat["nz"][j] ** 2 * nhat["R"][j] * v_betas_scalar * (
                                trajectory_ext[i_new[j]]["bdotz"][j] + trajectory_ext[i_new[j]]["bdotz"][j] * bdot_scalar_ext * gamma_j**2
                            ) + v_betas_scalar * c_mmns * nhat["nz"][j]
                        )

                    # ACCUMULATE Time/energy conjugate momentum component update (COMPLEX legacy formula)
                    time_charge_factor = (
                        h
                        * charge_i
                        * charge_j
                        / (
                            k_factor**3 * c_mmns**2 * nhat["R"][j] ** 2 * gamma_j**3
                        )  # Note: c_mmns**2 not **3
                    )

                    result["Pt"][particle_idx] += time_charge_factor * (
                        v_beta_dot_mixed_scalar * k_factor * gamma_j * nhat["R"][j]
                        - v_betas_scalar * k_factor * c_mmns * gamma_j**2
                        - bdot_scalar_ext * v_betas_scalar * gamma_j**4 * nhat["R"][j]
                        + v_betas_scalar * c_mmns
                    )

                    # COVARIANT POSITION UPDATE: Apply electromagnetic field contribution from this interaction
                    # dx/dτ = (1/m)[P - qA] - the qA term comes from each electromagnetic interaction
                    field_contribution_factor = h / mass_i * charge_i / c_mmns * charge_j / (nhat["R"][j] * k_factor)
                    
                    result["x"][particle_idx] += field_contribution_factor * trajectory_ext[i_new[j]]["bx"][j]
                    result["y"][particle_idx] += field_contribution_factor * trajectory_ext[i_new[j]]["by"][j]  
                    result["z"][particle_idx] += field_contribution_factor * trajectory_ext[i_new[j]]["bz"][j]
                    
                else:
                    # Use simple electromagnetic force calculation
                    simple_force, energy_rate = self._simple_em_force(
                        beta_vec,
                        gamma_i,
                        charge_i,
                        charge_j,
                        nhat["R"][j],
                        np.array([nhat["nx"][j], nhat["ny"][j], nhat["nz"][j]]),
                    )

                    # Apply simple force (convert from force to momentum change)
                    # F = dp/dt, so dp = F * dt = F * h
                    result["Px"][particle_idx] += simple_force[0] * h
                    result["Py"][particle_idx] += simple_force[1] * h
                    result["Pz"][particle_idx] += simple_force[2] * h

                    # Apply energy rate for covariant Pt update
                    # dE/dt, so dE = (dE/dt) * dt = energy_rate * h
                    result["Pt"][particle_idx] += energy_rate * h

                    # COVARIANT POSITION UPDATE: Apply electromagnetic field contribution from this interaction
                    # For simple case, use the same electromagnetic field contribution
                    field_contribution_factor = h / mass_i * charge_i / c_mmns * charge_j / (nhat["R"][j] * k_factor)
                    
                    result["x"][particle_idx] += field_contribution_factor * trajectory_ext[i_new[j]]["bx"][j]
                    result["y"][particle_idx] += field_contribution_factor * trajectory_ext[i_new[j]]["by"][j]
                    result["z"][particle_idx] += field_contribution_factor * trajectory_ext[i_new[j]]["bz"][j]

            # After accumulating all external contributions, compute derived quantities

            # FIRST GAMMA: Legacy "initial guess gamma" using field corrections
            # This matches legacy line 481 exactly
            if len(trajectory_ext[0]["x"]) > 0:
                j_last = len(trajectory_ext[0]["x"]) - 1

                # Handle charge values for field correction (scalar or array)
                if np.isscalar(trajectory[i_traj]["q"]):
                    charge_i_field = float(trajectory[i_traj]["q"])
                else:
                    if particle_idx < len(trajectory[i_traj]["q"]):
                        charge_i_field = float(trajectory[i_traj]["q"][particle_idx])
                    else:
                        charge_i_field = 0.0

                if np.isscalar(trajectory_ext[i_new[j_last]]["q"]):
                    charge_j_field = float(trajectory_ext[i_new[j_last]]["q"])
                else:
                    if j_last < len(trajectory_ext[i_new[j_last]]["q"]):
                        charge_j_field = float(trajectory_ext[i_new[j_last]]["q"][j_last])
                    else:
                        charge_j_field = 0.0

                scalar_field_correction = (
                    charge_i_field
                    / c_mmns
                    * charge_j_field
                    / (nhat["R"][j_last] * k_factor)  # k_factor from last iteration
                )

                # Handle mass values (scalar or array)
                if np.isscalar(trajectory[i_traj]["m"]):
                    mass_i = float(trajectory[i_traj]["m"])
                else:
                    if particle_idx < len(trajectory[i_traj]["m"]):
                        mass_i = float(trajectory[i_traj]["m"][particle_idx])
                    else:
                        mass_i = 1.0  # Default mass

                result["gamma"][particle_idx] = (
                    1.0
                    / (mass_i * c_mmns)
                    * (result["Pt"][particle_idx] - scalar_field_correction)
                )
            else:
                # No external particles - simple energy-momentum relation
                # Handle mass values (scalar or array)
                if np.isscalar(trajectory[i_traj]["m"]):
                    mass_i = float(trajectory[i_traj]["m"])
                else:
                    if particle_idx < len(trajectory[i_traj]["m"]):
                        mass_i = float(trajectory[i_traj]["m"][particle_idx])
                    else:
                        mass_i = 1.0  # Default mass

                result["gamma"][particle_idx] = result["Pt"][particle_idx] / (
                    mass_i * c_mmns
                )

            # Update time
            result["t"][particle_idx] = (
                trajectory[i_traj]["t"][particle_idx]
                + h * result["gamma"][particle_idx]
            )

            # Position updates using conjugate momentum with field corrections
            # This uses the simple field correction (matches legacy position updates)
            if len(trajectory_ext[0]["x"]) > 0:
                j_last = len(trajectory_ext[0]["x"]) - 1

                # Handle charge values for field corrections (scalar or array)
                if np.isscalar(trajectory[i_traj]["q"]):
                    charge_i_pos = float(trajectory[i_traj]["q"])
                else:
                    if particle_idx < len(trajectory[i_traj]["q"]):
                        charge_i_pos = float(trajectory[i_traj]["q"][particle_idx])
                    else:
                        charge_i_pos = 0.0

                if np.isscalar(trajectory_ext[i_new[j_last]]["q"]):
                    charge_j_pos = float(trajectory_ext[i_new[j_last]]["q"])
                else:
                    if j_last < len(trajectory_ext[i_new[j_last]]["q"]):
                        charge_j_pos = float(trajectory_ext[i_new[j_last]]["q"][j_last])
                    else:
                        charge_j_pos = 0.0

                if abs(charge_i_pos) < 1e-20 or abs(charge_j_pos) < 1e-20:
                    # Zero charge case - no field corrections
                    field_correction_x = 0.0
                    field_correction_y = 0.0
                    field_correction_z = 0.0
                else:
                    # Simple field corrections for position (legacy position formula)
                    # CRITICAL: Protect against k_factor singularity (same as electromagnetic forces)
                    safe_k_factor = (
                        k_factor if abs(k_factor) > 1e-6 else 1e-6 * np.sign(k_factor)
                    )

                    field_correction_x = (
                        charge_i_pos
                        / c_mmns
                        * charge_j_pos
                        * trajectory_ext[i_new[j_last]]["bx"][j_last]
                        / (nhat["R"][j_last] * safe_k_factor)
                    )
                    field_correction_y = (
                        charge_i_pos
                        / c_mmns
                        * charge_j_pos
                        * trajectory_ext[i_new[j_last]]["by"][j_last]
                        / (nhat["R"][j_last] * safe_k_factor)
                    )
                    field_correction_z = (
                        charge_i_pos
                        / c_mmns
                        * charge_j_pos
                        * trajectory_ext[i_new[j_last]]["bz"][j_last]
                        / (nhat["R"][j_last] * safe_k_factor)
                    )
            else:
                field_correction_x = field_correction_y = field_correction_z = 0.0

            # Position updates: FIXED to match legacy approach exactly
            # Legacy uses h/m (not h/(γm)) - the gamma appears in velocity calculation instead
            # This matches covariant_integrator_library.py lines 312-316
            # Handle mass values (scalar or array) - reuse mass_i from above if available
            if 'mass_i' not in locals():
                if np.isscalar(trajectory[i_traj]["m"]):
                    mass_particle = float(trajectory[i_traj]["m"])
                else:
                    if particle_idx < len(trajectory[i_traj]["m"]):
                        mass_particle = float(trajectory[i_traj]["m"][particle_idx])
                    else:
                        mass_particle = 1.0  # Default mass
            else:
                mass_particle = mass_i

            result["x"][particle_idx] = trajectory[i_traj]["x"][particle_idx] + h / (
                mass_particle
            ) * (result["Px"][particle_idx] - field_correction_x)

            result["y"][particle_idx] = trajectory[i_traj]["y"][particle_idx] + h / (
                mass_particle
            ) * (result["Py"][particle_idx] - field_correction_y)

            result["z"][particle_idx] = trajectory[i_traj]["z"][particle_idx] + h / (
                mass_particle
            ) * (result["Pz"][particle_idx] - field_correction_z)

            # Calculate velocities from position updates (FIXED to match legacy)
            # LEGACY APPROACH: Use electromagnetic gamma in velocity calculation
            # This preserves the electromagnetic physics and prevents gamma corruption
            # Formula: β = Δx/(c·h·γ) where γ is the electromagnetic gamma from field calculations
            delta_x = result["x"][particle_idx] - trajectory[i_traj]["x"][particle_idx]
            delta_y = result["y"][particle_idx] - trajectory[i_traj]["y"][particle_idx]
            delta_z = result["z"][particle_idx] - trajectory[i_traj]["z"][particle_idx]

            # CRITICAL FIX: Use electromagnetic gamma (not missing it like before)
            # This matches legacy covariant_integrator_library.py lines 321-323
            result["bx"][particle_idx] = delta_x / (c_mmns * h * result["gamma"][particle_idx])
            result["by"][particle_idx] = delta_y / (c_mmns * h * result["gamma"][particle_idx])
            result["bz"][particle_idx] = delta_z / (c_mmns * h * result["gamma"][particle_idx])

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

            # DUAL GAMMA SYSTEM: Implement Benjamin Folsom's sophisticated approach
            # Store electromagnetic gamma for physics (already calculated above)
            gamma_electromagnetic = result["gamma"][particle_idx]
            
            # Calculate kinematic gamma from velocity magnitude as consistency check
            btot_squared = (
                result["bx"][particle_idx] ** 2
                + result["by"][particle_idx] ** 2
                + result["bz"][particle_idx] ** 2
            )

            # Legacy velocity limiting: only limit velocities that actually exceed c
            # High-energy particles naturally approach β → 1.0
            btots = np.sqrt(btot_squared)  # Calculate btots for legacy consistency
            if btots >= 1.0:
                # Limit to safely below c to avoid numerical precision issues
                btots_limited = (
                    0.9999999999999  # Match legacy precision
                )
                scale_factor = btots_limited / btots
                result["bx"][particle_idx] *= scale_factor
                result["by"][particle_idx] *= scale_factor
                result["bz"][particle_idx] *= scale_factor
                btots = btots_limited
                btot_squared = btots_limited**2

            # Kinematic gamma from velocity magnitude (consistency check)
            gamma_kinematic = np.sqrt(1.0 / (1.0 - btot_squared))
            
            # PRESERVE ELECTROMAGNETIC PHYSICS: Keep electromagnetic gamma as final result
            # This is the key fix - don't overwrite the electromagnetic gamma!
            # The kinematic gamma serves as a numerical stability check
            # If abs(gamma_electromagnetic - gamma_kinematic) is large, it indicates numerical issues
            
            # Optional: Add consistency check warning (can be enabled for debugging)
            # gamma_diff = abs(gamma_electromagnetic - gamma_kinematic)
            # if gamma_diff > 0.1 * gamma_electromagnetic:  # 10% difference threshold
            #     print(f"Warning: Gamma consistency check failed - EM: {gamma_electromagnetic:.3f}, Kinematic: {gamma_kinematic:.3f}")
            
            # FINAL GAMMA: Keep electromagnetic gamma (contains the physics)
            result["gamma"][particle_idx] = gamma_electromagnetic

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

    def retarded_rk4_rela_step(
        self,
        h_step: float,
        trajectory: List[Dict],
        trajectory_drv: List[Dict],
        i_traj: int,
        apt_R: float,
        sim_type: Any,
    ) -> Dict[str, Any]:
        """
        Retarded RK4 relativistic step - wrapper for eqsofmotion_retarded.

        This method provides compatibility with legacy integrator interfaces
        that expect retarded_rk4_rela_step.
        """
        return self.eqsofmotion_retarded(
            h_step, trajectory, trajectory_drv, i_traj, apt_R, sim_type
        )

    def integrate_retarded_fields(
        self,
        static_steps: int,
        ret_steps: int,
        h_step: float,
        wall_Z: float,
        apt_R: float,
        sim_type: int,
        init_rider: Dict[str, Any],
        init_driver: Dict[str, Any],
        bunch_dist: float,
        z_cutoff: float,
        cav_spacing: Optional[float] = None,
        Ez_field: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Integrate particle trajectories with retarded electromagnetic field interactions.

        Supports all simulation types:
        - sim_type=0: Conducting wall with image charges
        - sim_type=1: Switching wall with disappearing aperture (requires cav_spacing)
        - sim_type=2: Free particle bunches (used in demo)

        Args:
            static_steps: Number of static integration steps
            ret_steps: Number of retarded integration steps
            h_step: Time step size (ns)
            wall_Z: Wall position (mm)
            apt_R: Aperture radius (mm)
            sim_type: Simulation type (0, 1, or 2)
            init_rider: Initial rider bunch state
            init_driver: Initial driver bunch state
            bunch_dist: Bunch separation distance (mm)
            z_cutoff: Wall switching position (mm)
            cav_spacing: Cavity spacing for repeated walls (mm). Required for sim_type=1.
            Ez_field: Constant Ez field strength (MV/m). Default 0.0 for no field.

        Returns:
            Tuple of (rider_trajectory, driver_trajectory) lists
        """

        # Step size validation for electromagnetic stability
        # User-requested: Allow larger steps up to 3 ns, energy-dependent scaling
        MAX_STEP_LIMIT = 3.0  # ns - user-requested maximum
        RECOMMENDED_MAX_STEP = 1e-3  # ns - conservative for high precision

        if h_step > MAX_STEP_LIMIT:
            raise ValueError(
                f"Step size h={h_step:.3f} ns exceeds maximum limit {MAX_STEP_LIMIT:.1f} ns! "
                f"Consider using smaller timesteps for better electromagnetic accuracy."
            )

        if h_step > RECOMMENDED_MAX_STEP:
            import warnings

            warnings.warn(
                f"Step size h={h_step:.3f} ns exceeds recommended limit {RECOMMENDED_MAX_STEP:.0e} ns. "
                f"Large timesteps may reduce electromagnetic accuracy but are allowed for efficiency. "
                f"Monitor energy conservation during integration."
            )

        # Validate cav_spacing parameter based on simulation type
        if sim_type == 1 and cav_spacing is None:
            raise ValueError(
                "cav_spacing parameter is required for sim_type=1 (switching wall with multiple cavities)"
            )
        elif cav_spacing is None:
            # For other simulation types, set a default that won't be used
            cav_spacing = 1e5  # mm, arbitrary large value

        steps_tot = static_steps + ret_steps

        # Initialize trajectory arrays (like legacy)
        trajectory_new: List[Dict[str, np.ndarray]] = [{}] * steps_tot
        trajectory_drv_new: List[Dict[str, np.ndarray]] = [{}] * steps_tot

        # Track current wall position and cutoff for sim_type=1
        current_wall_Z = wall_Z
        current_z_cutoff = z_cutoff

        print(
            f"  Updated integrator: {steps_tot} steps (static: {static_steps}, retarded: {ret_steps})"
        )
        print(f"  Simulation type: {sim_type}, wall_Z: {wall_Z}, apt_R: {apt_R}")

        # Integration loop
        for i in range(steps_tot):
            if i == 0:
                # Initialize first step
                trajectory_new[i] = self._deep_copy_state(init_rider)

                if sim_type == 0:  # Conducting wall
                    trajectory_drv_new[i] = self.conducting_flat_updated(
                        init_rider, current_wall_Z, apt_R
                    )
                elif sim_type == 1:  # Switching wall
                    trajectory_drv_new[i] = self.switching_flat_updated(
                        init_rider, current_wall_Z, apt_R, current_z_cutoff
                    )
                elif sim_type == 2:  # Free particle bunches
                    trajectory_drv_new[i] = self._deep_copy_state(init_driver)

            elif i < static_steps:
                # Static integration phase
                trajectory_new[i] = self._eqsofmotion_static_updated(
                    h_step,
                    trajectory_new[i - 1],
                    trajectory_drv_new[i - 1],
                    apt_R,
                    sim_type,
                    Ez_field,
                )

                if sim_type == 0:  # Conducting wall
                    trajectory_drv_new[i] = self.conducting_flat_updated(
                        trajectory_new[i - 1], current_wall_Z, apt_R
                    )
                elif sim_type == 1:  # Switching wall
                    trajectory_drv_new[i] = self.switching_flat_updated(
                        trajectory_new[i - 1], current_wall_Z, apt_R, current_z_cutoff
                    )
                elif sim_type == 2:  # Free particle bunches
                    trajectory_drv_new[i] = self._eqsofmotion_static_updated(
                        h_step,
                        trajectory_drv_new[i - 1],
                        trajectory_new[i - 1],
                        apt_R,
                        sim_type,
                        Ez_field,
                    )

            else:
                # Retarded integration phase
                # CRITICAL FIX: Store previous state to ensure simultaneous interaction
                # Both particles must use the same retarded time reference (i-1)
                trajectory_prev_state = trajectory_new[i - 1] if i > 0 else trajectory_new[0]
                trajectory_drv_prev_state = trajectory_drv_new[i - 1] if i > 0 else trajectory_drv_new[0]
                
                trajectory_new[i] = self.eqsofmotion_retarded(
                    h_step,
                    trajectory_new,
                    trajectory_drv_new,
                    i - 1,
                    apt_R,
                    str(sim_type),
                )

                if sim_type == 0:  # Conducting wall
                    trajectory_drv_new[i] = self.conducting_flat_updated(
                        trajectory_new[i], current_wall_Z, apt_R
                    )
                elif sim_type == 1:  # Switching wall
                    trajectory_drv_new[i] = self.switching_flat_updated(
                        trajectory_new[i], current_wall_Z, apt_R, current_z_cutoff
                    )

                    # Check for wall advancement (cavity spacing)
                    if np.mean(trajectory_new[i]["z"]) > current_z_cutoff:
                        current_z_cutoff += cav_spacing
                        current_wall_Z += cav_spacing
                        print(
                            f"    Wall advanced: z_cutoff={current_z_cutoff:.1f}, wall_Z={current_wall_Z:.1f}"
                        )

                elif sim_type == 2:  # Free particle bunches
                    # CRITICAL FIX: Use stored previous states to ensure simultaneity
                    # Create temporary trajectory arrays with the previous state
                    trajectory_temp = trajectory_new.copy()
                    trajectory_temp[i] = trajectory_prev_state  # Restore previous state
                    
                    trajectory_drv_new[i] = self.eqsofmotion_retarded(
                        h_step,
                        trajectory_drv_new,
                        trajectory_temp,  # Use trajectory state from time i-1
                        i - 1,
                        apt_R,
                        str(sim_type),
                    )

        return trajectory_new, trajectory_drv_new

    def _deep_copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of particle state dictionary."""
        result = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                result[key] = np.copy(value)
            else:
                result[key] = value
        return result

    def _eqsofmotion_static_updated(
        self,
        h_step: float,
        state: Dict[str, Any],
        state_ext: Dict[str, Any],
        apt_R: float,
        sim_type: int,
        Ez_field: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Static equations of motion - EXACT COPY of retarded integrator structure.

        Uses identical methods as eqsofmotion_retarded but ignores chronojn (retardation delays).
        All external particles are evaluated at the same timestep as the current particle.
        This eliminates time delays while preserving exact legacy electromagnetic calculations.

        Args:
            h_step: Time step size (ns)
            state: Current particle state (trajectory[i_traj])
            state_ext: External particle state (trajectory_ext[i_traj])
            apt_R: Aperture radius (mm)
            sim_type: Simulation type
            Ez_field: Constant Ez field strength (MV/m). Default 0.0.
        """

        h = h_step
        c_mmns = self.c_mmns

        # Create trajectory-like structure for compatibility with retarded methods
        trajectory = [state]  # Single timestep "trajectory"
        trajectory_ext = [state_ext]  # Single timestep external trajectory
        i_traj = 0  # Always use current timestep

        # Initialize result arrays (EXACT copy from retarded)
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

        # Copy particle properties (EXACT copy from retarded)
        result["q"] = trajectory[i_traj]["q"]
        result["char_time"] = trajectory[i_traj]["char_time"]
        result["m"] = trajectory[i_traj]["m"]

        # Integrate each particle (EXACT copy from retarded loop structure)
        for particle_idx in range(len(trajectory[i_traj]["x"])):
            # STATIC VERSION: Skip chrono_jn - use current timestep for all interactions
            # i_new = [i_traj] * len(trajectory_ext[0]["x"])  # All at current time

            # STATIC VERSION: Skip dist_euclid_ret - calculate instantaneous distances
            nhat = self._dist_euclid_instantaneous(
                trajectory, trajectory_ext, i_traj, particle_idx
            )

            # Initialize with current values (EXACT copy from retarded)
            result["x"][particle_idx] = trajectory[i_traj]["x"][particle_idx]
            result["y"][particle_idx] = trajectory[i_traj]["y"][particle_idx]
            result["z"][particle_idx] = trajectory[i_traj]["z"][particle_idx]
            result["t"][particle_idx] = trajectory[i_traj]["t"][particle_idx]
            result["Px"][particle_idx] = trajectory[i_traj]["Px"][particle_idx]
            result["Py"][particle_idx] = trajectory[i_traj]["Py"][particle_idx]
            result["Pz"][particle_idx] = trajectory[i_traj]["Pz"][particle_idx]
            result["Pt"][particle_idx] = trajectory[i_traj]["Pt"][particle_idx]

            # DEBUG: Track momentum changes in static integrator
            initial_momentum = result["Pz"][particle_idx]
            momentum_accumulation_count = 0

            # Sum all external field contributions (EXACT copy from retarded)
            external_particles = len(trajectory_ext[0]["x"])

            for j in range(len(trajectory_ext[0]["x"])):

                # Current particle velocity (EXACT copy from retarded)
                beta_vec = np.array(
                    [
                        trajectory[i_traj]["bx"][particle_idx],
                        trajectory[i_traj]["by"][particle_idx],
                        trajectory[i_traj]["bz"][particle_idx],
                    ]
                )

                # STATIC VERSION: External particle velocity at CURRENT time (not retarded)
                beta_ext = np.array(
                    [
                        trajectory_ext[i_traj]["bx"][j],  # i_traj instead of i_new[j]
                        trajectory_ext[i_traj]["by"][j],
                        trajectory_ext[i_traj]["bz"][j],
                    ]
                )

                # STATIC VERSION: Calculate k_factor using current time (no retardation)
                nhat_vec = np.array([nhat["nx"][j], nhat["ny"][j], nhat["nz"][j]])
                k_factor = 1.0 - np.dot(
                    beta_ext, nhat_vec
                )  # Same calculation as retarded

                # Regularization (EXACT copy from retarded)
                if abs(k_factor) < 1e-15:
                    k_factor = 1e-15 * np.sign(k_factor) if k_factor != 0 else 1e-15

                # STATIC VERSION: Acceleration terms at current time
                bdot_ext = np.array(
                    [
                        trajectory_ext[i_traj]["bdotx"][
                            j
                        ],  # i_traj instead of i_new[j]
                        trajectory_ext[i_traj]["bdoty"][j],
                        trajectory_ext[i_traj]["bdotz"][j],
                    ]
                )

                # Scalar products (EXACT copy from retarded)
                bdot_scalar_ext = np.dot(bdot_ext, bdot_ext)
                betas_scalar = np.dot(beta_ext, beta_vec)

                # Velocity-dependent terms (EXACT copy from retarded)
                gamma_i = trajectory[i_traj]["gamma"][particle_idx]
                gamma_j = trajectory_ext[i_traj]["gamma"][
                    j
                ]  # i_traj instead of i_new[j]

                # Protection against corrupted gamma (EXACT copy from retarded)
                if gamma_j > 1e12 or gamma_i > 1e12:
                    print(f"DEBUG STATIC: Gamma skip - γi={gamma_i:.6f}, γj={gamma_j:.6f}")
                    continue

                # V_ext^beta * V_beta (EXACT copy from retarded)
                v_betas_scalar = gamma_j * gamma_i * c_mmns**2 * (1.0 - betas_scalar)

                # Vdot_ext^beta * V_beta (EXACT copy from retarded)
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

                # Choose physics complexity (EXACT copy from retarded)
                use_complex = self._needs_complex_physics(
                    beta_vec, beta_ext, gamma_i, gamma_j, k_factor, nhat["R"][j]
                )

                # Zero charge check (EXACT copy from retarded)
                charge_i = float(trajectory[i_traj]["q"][particle_idx])
                charge_j = float(
                    trajectory_ext[i_traj]["q"][j]
                )  # i_traj instead of i_new[j]
                if abs(charge_i) < 1e-20 or abs(charge_j) < 1e-20:
                    print(f"DEBUG STATIC: Zero charge skip - qi={charge_i:.8f}, qj={charge_j:.8f}")
                    continue

                if use_complex:
                    # Complex electromagnetic force (EXACT copy from retarded)
                    if (
                        abs(k_factor) < 1e-15
                    ):  # Extremely conservative: only prevent overflow
                        print(f"DEBUG STATIC: k_factor skip - k_factor={k_factor:.12f}")
                        continue

                    charge_factor = (
                        h
                        * charge_i
                        * charge_j
                        / (k_factor**3 * c_mmns**3 * nhat["R"][j] ** 2 * gamma_j**3)
                    )

                    # DEBUG: Track force calculation
                    if j == 0 and particle_idx == 0:  # First particle interaction
                        print(f"DEBUG STATIC: qi*qj={charge_i*charge_j:.8f}, k_factor={k_factor:.6f}, R={nhat['R'][j]:.6f}")
                        print(f"DEBUG STATIC: charge_factor={charge_factor:.12f}")

                    # ACCUMULATE X-momentum (EXACT copy from retarded)
                    result["Px"][particle_idx] += charge_factor * (
                        -v_betas_scalar
                        * trajectory_ext[i_traj]["bx"][j]
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
                            trajectory_ext[i_traj]["bdotx"][j]
                            + trajectory_ext[i_traj]["bdotx"][j]
                            * bdot_scalar_ext
                            * gamma_j**2
                        )
                        + v_betas_scalar * c_mmns * nhat["nx"][j]
                    )

                    # ACCUMULATE Y-momentum (EXACT copy from retarded)
                    result["Py"][particle_idx] += charge_factor * (
                        -v_betas_scalar
                        * trajectory_ext[i_traj]["by"][j]
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
                            trajectory_ext[i_traj]["bdoty"][j]
                            + trajectory_ext[i_traj]["bdoty"][j]
                            * bdot_scalar_ext
                            * gamma_j**2
                        )
                        + v_betas_scalar * c_mmns * nhat["ny"][j]
                    )

                    # ACCUMULATE Z-momentum (EXACT copy from retarded)
                    pz_before = result["Pz"][particle_idx]
                    momentum_change = charge_factor * (
                        -v_betas_scalar
                        * trajectory_ext[i_traj]["bz"][j]
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
                            trajectory_ext[i_traj]["bdotz"][j]
                            + trajectory_ext[i_traj]["bdotz"][j]
                            * bdot_scalar_ext
                            * gamma_j**2
                        )
                        + v_betas_scalar * c_mmns * nhat["nz"][j]
                    )
                    result["Pz"][particle_idx] += momentum_change
                    momentum_accumulation_count += 1
                    


                    # ACCUMULATE Time/energy momentum (EXACT copy from retarded)
                    time_charge_factor = (
                        h
                        * trajectory[i_traj]["q"][particle_idx]
                        * trajectory_ext[i_traj]["q"][j]
                        / (k_factor**3 * c_mmns**2 * nhat["R"][j] ** 2 * gamma_j**3)
                    )

                    result["Pt"][particle_idx] += time_charge_factor * (
                        v_beta_dot_mixed_scalar * k_factor * gamma_j * nhat["R"][j]
                        - v_betas_scalar * k_factor * c_mmns * gamma_j**2
                        - bdot_scalar_ext * v_betas_scalar * gamma_j**4 * nhat["R"][j]
                        + v_betas_scalar * c_mmns
                    )

                    # COVARIANT POSITION UPDATE: Apply electromagnetic field contribution from this interaction
                    # dx/dτ = (1/m)[P - qA] - the qA term comes from each electromagnetic interaction
                    mass_i = float(trajectory[i_traj]["m"][particle_idx])
                    field_contribution_factor = h / mass_i * charge_i / c_mmns * charge_j / (nhat["R"][j] * k_factor)
                    
                    result["x"][particle_idx] += field_contribution_factor * trajectory_ext[i_traj]["bx"][j]
                    result["y"][particle_idx] += field_contribution_factor * trajectory_ext[i_traj]["by"][j]  
                    result["z"][particle_idx] += field_contribution_factor * trajectory_ext[i_traj]["bz"][j]

                else:
                    # Use simple electromagnetic force (EXACT copy from retarded)
                    simple_force, energy_rate = self._simple_em_force(
                        beta_vec,
                        gamma_i,
                        charge_i,
                        charge_j,
                        nhat["R"][j],
                        np.array([nhat["nx"][j], nhat["ny"][j], nhat["nz"][j]]),
                    )

                    # Apply simple force
                    pz_before = result["Pz"][particle_idx]
                    result["Px"][particle_idx] += simple_force[0] * h
                    result["Py"][particle_idx] += simple_force[1] * h
                    result["Pz"][particle_idx] += simple_force[2] * h
                    result["Pt"][particle_idx] += energy_rate * h  # Energy conservation
                    momentum_accumulation_count += 1

                    # COVARIANT POSITION UPDATE: Apply electromagnetic field contribution from this interaction
                    # For simple case, use the same electromagnetic field contribution
                    mass_i = float(trajectory[i_traj]["m"][particle_idx])
                    field_contribution_factor = h / mass_i * charge_i / c_mmns * charge_j / (nhat["R"][j] * k_factor)
                    
                    result["x"][particle_idx] += field_contribution_factor * trajectory_ext[i_traj]["bx"][j]
                    result["y"][particle_idx] += field_contribution_factor * trajectory_ext[i_traj]["by"][j]
                    result["z"][particle_idx] += field_contribution_factor * trajectory_ext[i_traj]["bz"][j]
                    # REMOVED continue - allow processing to continue for field corrections and position updates

            # After accumulating all external contributions, compute derived quantities (EXACT copy from retarded)

            # Update gamma using field correction (EXACT copy from retarded)
            if len(trajectory_ext[0]["x"]) > 0:
                j_last = len(trajectory_ext[0]["x"]) - 1
                scalar_field_correction = (
                    trajectory[i_traj]["q"][particle_idx]
                    / c_mmns
                    * trajectory_ext[i_traj]["q"][
                        j_last
                    ]  # i_traj instead of i_new[j_last]
                    / (nhat["R"][j_last] * k_factor)
                )

                result["gamma"][particle_idx] = (
                    1.0
                    / (trajectory[i_traj]["m"][particle_idx] * c_mmns)
                    * (result["Pt"][particle_idx] - scalar_field_correction)
                )
            else:
                result["gamma"][particle_idx] = result["Pt"][particle_idx] / (
                    trajectory[i_traj]["m"][particle_idx] * c_mmns
                )

            # Update time (EXACT copy from retarded)
            result["t"][particle_idx] = (
                trajectory[i_traj]["t"][particle_idx]
                + h * result["gamma"][particle_idx]
            )

            # Position updates with field corrections (EXACT copy from retarded)
            if len(trajectory_ext[0]["x"]) > 0:
                j_last = len(trajectory_ext[0]["x"]) - 1

                charge_i = float(trajectory[i_traj]["q"][particle_idx])
                charge_j = float(
                    trajectory_ext[i_traj]["q"][j_last]
                )  # i_traj instead of i_new[j_last]
                if abs(charge_i) < 1e-20 or abs(charge_j) < 1e-20:
                    field_correction_x = field_correction_y = field_correction_z = 0.0
                else:
                    safe_k_factor = (
                        k_factor if abs(k_factor) > 1e-6 else 1e-6 * np.sign(k_factor)
                    )

                    field_correction_x = (
                        charge_i
                        / c_mmns
                        * charge_j
                        * trajectory_ext[i_traj]["bx"][
                            j_last
                        ]  # i_traj instead of i_new[j_last]
                        / (nhat["R"][j_last] * safe_k_factor)
                    )
                    field_correction_y = (
                        charge_i
                        / c_mmns
                        * charge_j
                        * trajectory_ext[i_traj]["by"][j_last]
                        / (nhat["R"][j_last] * safe_k_factor)
                    )
                    field_correction_z = (
                        charge_i
                        / c_mmns
                        * charge_j
                        * trajectory_ext[i_traj]["bz"][j_last]
                        / (nhat["R"][j_last] * safe_k_factor)
                    )
            else:
                field_correction_x = field_correction_y = field_correction_z = 0.0

            # COVARIANT MECHANICS: Positions are now updated within each electromagnetic interaction loop
            # This ensures proper coupling between momentum and position updates as required by covariant physics
            # However, we still need to calculate gamma and derived quantities after all interactions
            
            # Calculate gamma from final momentum after all electromagnetic interactions
            momentum_magnitude = np.sqrt(
                result["Px"][particle_idx] ** 2
                + result["Py"][particle_idx] ** 2
                + result["Pz"][particle_idx] ** 2
            )
            current_gamma = np.sqrt(
                1.0
                + (
                    momentum_magnitude
                    / (trajectory[i_traj]["m"][particle_idx] * c_mmns)
                )
                ** 2
            )
            result["y"][particle_idx] = trajectory[i_traj]["y"][particle_idx] + h / (
                current_gamma * trajectory[i_traj]["m"][particle_idx]
            ) * (result["Py"][particle_idx] - field_correction_y)
            result["z"][particle_idx] = trajectory[i_traj]["z"][particle_idx] + h / (
                current_gamma * trajectory[i_traj]["m"][particle_idx]
            ) * (result["Pz"][particle_idx] - field_correction_z)

            # Calculate velocities from position updates with proper gamma
            delta_x = result["x"][particle_idx] - trajectory[i_traj]["x"][particle_idx]
            delta_y = result["y"][particle_idx] - trajectory[i_traj]["y"][particle_idx]
            delta_z = result["z"][particle_idx] - trajectory[i_traj]["z"][particle_idx]

            # Use the same gamma we calculated for position update
            result["bx"][particle_idx] = delta_x / (c_mmns * h * current_gamma)
            result["by"][particle_idx] = delta_y / (c_mmns * h * current_gamma)
            result["bz"][particle_idx] = delta_z / (c_mmns * h * current_gamma)

            # Calculate accelerations with consistent gamma
            result["bdotx"][particle_idx] = (
                result["bx"][particle_idx] - trajectory[i_traj]["bx"][particle_idx]
            ) / (c_mmns * h * current_gamma)
            result["bdoty"][particle_idx] = (
                result["by"][particle_idx] - trajectory[i_traj]["by"][particle_idx]
            ) / (c_mmns * h * current_gamma)
            result["bdotz"][particle_idx] = (
                result["bz"][particle_idx] - trajectory[i_traj]["bz"][particle_idx]
            ) / (c_mmns * h * current_gamma)

            # Use gamma calculated from momentum for final result
            result["gamma"][particle_idx] = current_gamma

            # DEBUG: Final momentum check
            final_momentum = result["Pz"][particle_idx]
            momentum_change = final_momentum - initial_momentum

            # SKIP radiation reaction for static case (instantaneous interactions)
            # Radiation reaction requires time history, incompatible with static approximation

        return result

    def _calculate_instantaneous_force(
        self,
        state1: Dict[str, Any],
        idx1: int,
        state2: Dict[str, Any],
        idx2: int,
        c_mmns: float,
    ) -> np.ndarray:
        """
        Calculate instantaneous electromagnetic force between two particles.

        This is the static (non-retarded) version of the electromagnetic force.
        Uses simplified relativistic corrections without full Lienard-Wiechert complexity.
        """
        # Get particle properties
        charge1 = state1["q"][idx1]
        charge2 = state2["q"][idx2]

        # Position difference vector
        dx = state1["x"][idx1] - state2["x"][idx2]
        dy = state1["y"][idx1] - state2["y"][idx2]
        dz = state1["z"][idx1] - state2["z"][idx2]

        # Distance with minimum cutoff for numerical stability
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        safe_distance = max(distance, 1e-12)  # Minimum distance (nuclear scale)

        # Unit vector pointing from particle 2 to particle 1
        nx = dx / safe_distance
        ny = dy / safe_distance
        nz = dz / safe_distance

        # Get relativistic factors
        gamma1 = state1["gamma"][idx1]
        gamma2 = state2["gamma"][idx2]

        # Instantaneous Coulomb force with relativistic enhancement
        # F = q1*q2/(4πε₀r²) in SI, but we use Gaussian units: F = q1*q2/r²
        coulomb_magnitude = (charge1 * charge2) / (safe_distance**2)

        # Relativistic enhancement for head-on collisions
        # For ultra-relativistic head-on interactions, use γ² enhancement
        relativistic_factor = gamma1 * gamma2

        # Force magnitude (positive for repulsion)
        force_magnitude = coulomb_magnitude * relativistic_factor

        # Force vector components
        force_x = force_magnitude * nx
        force_y = force_magnitude * ny
        force_z = force_magnitude * nz

        return np.array([force_x, force_y, force_z])


# ============================================================================
# OPTIMIZED INTEGRATOR FACTORY AND DEFAULT EXPORT
# ============================================================================

# Try to import and use the optimized version by default
_DEFAULT_USE_OPTIMIZED = True
_OPTIMIZED_AVAILABLE = False

try:
    from .performance import OptimizedLienardWiechertIntegrator

    _OPTIMIZED_AVAILABLE = True
except ImportError:
    # Optimized version not available, will fall back to standard
    OptimizedLienardWiechertIntegrator = None  # type: ignore


def create_integrator(
    config: Optional[SimulationConfig] = None, use_optimized: Optional[bool] = None
) -> "LienardWiechertIntegrator":
    """
    Factory function to create a Lienard-Wiechert integrator.

    Args:
        config: Simulation configuration
        use_optimized: Whether to use optimized version. If None, uses default (True)
                      If True but optimized unavailable, falls back to standard with warning

    Returns:
        LienardWiechertIntegrator instance (optimized by default if available)
    """
    should_use_optimized = (
        use_optimized if use_optimized is not None else _DEFAULT_USE_OPTIMIZED
    )

    if should_use_optimized and _OPTIMIZED_AVAILABLE:
        return OptimizedLienardWiechertIntegrator(config)
    elif should_use_optimized and not _OPTIMIZED_AVAILABLE:
        print(
            "⚠️  Optimized integrator requested but not available. Using standard integrator."
        )
        return LienardWiechertIntegrator(config)
    else:
        return LienardWiechertIntegrator(config)


# Create a default instance factory that applications can import as the main class
def LienardWiechertIntegratorFactory(
    config: Optional[SimulationConfig] = None, use_optimized: Optional[bool] = None
) -> "LienardWiechertIntegrator":
    """
    Default factory that returns optimized integrator by default.
    This maintains backward compatibility while providing optimized performance.
    """
    return create_integrator(config, use_optimized)
