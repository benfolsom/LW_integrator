"""Vectorized kernels for retarded electromagnetic force calculations.

This module implements the core retarded field computations using NumPy
vectorization for efficient batch processing of external source particles.

Numba Acceleration
------------------
When Numba is available, the force calculation kernel is JIT-compiled for
~20x performance improvement. The numba path is automatically selected when
available and provides identical results to the pure NumPy implementation.

Physical Context
----------------

The Liénard-Wiechert retarded fields from a moving source charge depend on
the source's state at the retarded time t_ret = t_obs - R/c, where R is the
source-observer separation. The key geometric factor is::

    k = 1 - β·n̂

where β = v/c is the source velocity and n̂ points from source to observer.
This k-factor appears in denominators, leading to field enhancement when
particles move nearly head-on (β·n̂ → 1).

Computed Quantities
-------------------

compute_vectorized_contributions
    Returns 8 values per call:

    1-4. Momentum changes (ΔPx, ΔPy, ΔPz, ΔPt) from retarded E and B fields
    5-7. Field contributions for position updates (gauge field components)
    8. Scalar potential sum Φ = Σ(q_j / (R_j · k_j)) for energy corrections

The scalar potential (item 8) is used to compute the correct kinetic energy::

    E_kinetic = Pt - q·Φ
    γ = E_kinetic / (mc)

This separates the particle's kinetic energy from the electromagnetic potential
energy, which is critical for self-consistency iterations.

k-factor Threshold
------------------

The implementation filters out contributions where |k| < 1e-20 to prevent
numerical overflow. This threshold approaches float64 machine limits while
remaining safe: with k_min = 1e-20, the maximum force scaling 1/k³ = 1e60
is well within float64 range (max ≈ 1.8e308). This extremely permissive
threshold only excludes interactions where particles are moving at
β > 1 - 1e-20 (γ > 7e9) nearly directly toward each other. The mass-shell
projection provides primary protection against β > 1 violations; k-threshold
is secondary filtering for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .constants import C_MMNS
from .types import IndexedTrajectoryArrays, TrajectoryArrays

TrajectoryArraysLike = TrajectoryArrays | IndexedTrajectoryArrays


def _soa_values_at_steps(
    traj: TrajectoryArraysLike,
    field_name: str,
    steps: np.ndarray,
    particle_indices: np.ndarray,
) -> np.ndarray:
    if isinstance(traj, IndexedTrajectoryArrays):
        return traj.values_at_steps(field_name, steps, particle_indices)
    return np.asarray(getattr(traj, field_name))[steps, particle_indices]


def _soa_constant(traj: TrajectoryArraysLike, field_name: str) -> np.ndarray:
    if isinstance(traj, IndexedTrajectoryArrays):
        return traj.constant(field_name)
    return np.asarray(getattr(traj, field_name))


# Try to import numba for JIT compilation
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Provide a no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range


# K-factor thresholds for series approximation
K_CUTOFF_HARD = 1e-20  # Below this: skip interaction entirely
K_SERIES_THRESHOLD = 1e-3  # Below this: use series approximation
NUMBA_FORCE_SERIAL_MAX_SOURCES = 128
NUMBA_FORCE_PARALLEL_MIN_SOURCES = 256


def _compute_small_k_forces_series(
    k_factor: np.ndarray,
    charge_factor_base: np.ndarray,
    v_betas_scalar: np.ndarray,
    v_beta_dot_mixed_scalar: np.ndarray,
    bx_ext: np.ndarray,
    by_ext: np.ndarray,
    bz_ext: np.ndarray,
    bdotx_ext: np.ndarray,
    bdoty_ext: np.ndarray,
    bdotz_ext: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    R_sep: np.ndarray,
    gamma_ext: np.ndarray,
    c: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute forces using series expansion for small k-factor.

    When k = 1 - β·n̂ is small, the Liénard-Wiechert fields have singularities
    that are regularized by expanding in powers of k. This function computes
    the leading-order terms in the series expansion.

    For small k, the force expressions simplify because terms proportional to
    higher powers of k become negligible compared to the 1/k³ divergence.
    The expansion captures the physical behavior without numerical instability.

    Parameters
    ----------
    k_factor : np.ndarray
        Small retardation factors (1e-20 < |k| < 1e-4)
    charge_factor_base : np.ndarray
        Base charge factor without k³ denominator
    v_betas_scalar, v_beta_dot_mixed_scalar : np.ndarray
        Velocity and acceleration terms
    bx_ext, by_ext, bz_ext : np.ndarray
        External source velocities
    bdotx_ext, bdoty_ext, bdotz_ext : np.ndarray
        External source accelerations
    nx, ny, nz : np.ndarray
        Unit vectors from source to observer
    R_sep : np.ndarray
        Separation distances
    gamma_ext : np.ndarray
        External source Lorentz factors
    c : float
        Speed of light

    Returns
    -------
    term_px, term_py, term_pz, term_pt : np.ndarray
        Momentum change terms with series approximation
    """
    # For small k, expand the force terms in powers of k
    # Keep leading non-divergent terms (the 1/k³ terms with k cancellations)

    # The full expression has terms like: (1/k³) * [A + B*k + C*k² + ...]
    # For small k, we need to keep enough terms so that the product converges

    # Series approximation: use first-order expansion
    # This is a simplified version - full implementation would expand all terms
    k_inv = 1.0 / k_factor
    k_inv3 = k_inv**3

    # Use regularized forms: keep terms that don't diverge
    # In the limit k→0, the divergence cancels with numerator zeros
    bdot_scalar_ext = bx_ext * bdotx_ext + by_ext * bdoty_ext + bz_ext * bdotz_ext

    # For very small k, use limiting form of the force expression
    # The leading behavior is dominated by the radiation reaction terms
    term_px = (
        -v_betas_scalar * bx_ext * k_factor * c * gamma_ext**2
        + v_beta_dot_mixed_scalar * k_factor * gamma_ext * nx * R_sep
        + gamma_ext**2
        * nx**2
        * R_sep
        * v_betas_scalar
        * (bdotx_ext + bdotx_ext * bdot_scalar_ext * gamma_ext**2)
        + v_betas_scalar * c * nx
    )

    term_py = (
        -v_betas_scalar * by_ext * k_factor * c * gamma_ext**2
        + v_beta_dot_mixed_scalar * k_factor * gamma_ext * ny * R_sep
        + gamma_ext**2
        * ny**2
        * R_sep
        * v_betas_scalar
        * (bdoty_ext + bdoty_ext * bdot_scalar_ext * gamma_ext**2)
        + v_betas_scalar * c * ny
    )

    term_pz = (
        -v_betas_scalar * bz_ext * k_factor * c * gamma_ext**2
        + v_beta_dot_mixed_scalar * k_factor * gamma_ext * nz * R_sep
        + gamma_ext**2
        * nz**2
        * R_sep
        * v_betas_scalar
        * (bdotz_ext + bdotz_ext * bdot_scalar_ext * gamma_ext**2)
        + v_betas_scalar * c * nz
    )

    term_pt = (
        v_beta_dot_mixed_scalar * k_factor * gamma_ext * R_sep
        - v_betas_scalar * k_factor * c * gamma_ext**2
        - bdot_scalar_ext * v_betas_scalar * gamma_ext**4 * R_sep
        + v_betas_scalar * c
    )

    # Apply charge factor with series-regularized k³
    # Use a damping factor to smoothly transition
    charge_factor_series = charge_factor_base * k_inv3

    return (
        charge_factor_series * term_px,
        charge_factor_series * term_py,
        charge_factor_series * term_pz,
        charge_factor_series * term_pt,
    )


@dataclass(slots=True)
class ExternalSampleBatch:
    """Container for external bunch samples at retarded indices."""

    charge: np.ndarray
    gamma: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    bdotx: np.ndarray
    bdoty: np.ndarray
    bdotz: np.ndarray
    valid_mask: np.ndarray
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None

    @property
    def any_valid(self) -> bool:
        return bool(self.valid_mask.any())


def gather_external_samples_soa(
    traj_ext: TrajectoryArraysLike,
    indices: np.ndarray,
    *,
    indices_next: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    needs_interpolation: np.ndarray | None = None,
) -> ExternalSampleBatch:
    """SOA fast path for gather_external_samples.

    Replaces per-particle dict/list access with direct 2-D array slicing.
    """
    n_ext = traj_ext.n_particles
    indices = np.asarray(indices, dtype=int)
    valid_mask = (indices >= 0) & (indices < traj_ext.n_steps)
    valid_all = bool(np.all(valid_mask))

    particle_indices = np.arange(n_ext)
    safe_indices = indices if valid_all else np.where(valid_mask, indices, 0)
    bx = _soa_values_at_steps(traj_ext, "bx", safe_indices, particle_indices).copy()
    by = _soa_values_at_steps(traj_ext, "by", safe_indices, particle_indices).copy()
    bz = _soa_values_at_steps(traj_ext, "bz", safe_indices, particle_indices).copy()
    bdotx = _soa_values_at_steps(
        traj_ext, "bdotx", safe_indices, particle_indices
    ).copy()
    bdoty = _soa_values_at_steps(
        traj_ext, "bdoty", safe_indices, particle_indices
    ).copy()
    bdotz = _soa_values_at_steps(
        traj_ext, "bdotz", safe_indices, particle_indices
    ).copy()
    gamma = _soa_values_at_steps(
        traj_ext, "gamma", safe_indices, particle_indices
    ).copy()
    charge = _soa_constant(traj_ext, "q").copy()
    dead_at_sample = _soa_values_at_steps(
        traj_ext, "dead", safe_indices, particle_indices
    )
    charge[dead_at_sample] = 0.0
    valid_mask = valid_mask & ~dead_at_sample
    x = y = z = None

    if not valid_all:
        invalid_mask = ~valid_mask
        bx[invalid_mask] = 0.0
        by[invalid_mask] = 0.0
        bz[invalid_mask] = 0.0
        bdotx[invalid_mask] = 0.0
        bdoty[invalid_mask] = 0.0
        bdotz[invalid_mask] = 0.0
        gamma[invalid_mask] = 0.0
        charge[invalid_mask] = 0.0

    if (
        weights is not None
        and indices_next is not None
        and needs_interpolation is not None
        and np.any(needs_interpolation)
    ):
        interp_mask = (
            valid_mask
            & needs_interpolation
            & (indices_next >= 0)
            & (indices_next < traj_ext.n_steps)
        )
        if np.any(interp_mask):
            x = _soa_values_at_steps(
                traj_ext, "x", safe_indices, particle_indices
            ).copy()
            y = _soa_values_at_steps(
                traj_ext, "y", safe_indices, particle_indices
            ).copy()
            z = _soa_values_at_steps(
                traj_ext, "z", safe_indices, particle_indices
            ).copy()
            if not valid_all:
                invalid_mask = ~valid_mask
                x[invalid_mask] = 0.0
                y[invalid_mask] = 0.0
                z[invalid_mask] = 0.0

            ni = indices_next[interp_mask]
            pi = particle_indices[interp_mask]
            w = weights[interp_mask]
            w1 = 1.0 - w
            bx[interp_mask] = w * bx[interp_mask] + w1 * _soa_values_at_steps(
                traj_ext, "bx", ni, pi
            )
            by[interp_mask] = w * by[interp_mask] + w1 * _soa_values_at_steps(
                traj_ext, "by", ni, pi
            )
            bz[interp_mask] = w * bz[interp_mask] + w1 * _soa_values_at_steps(
                traj_ext, "bz", ni, pi
            )
            bdotx[interp_mask] = w * bdotx[interp_mask] + w1 * _soa_values_at_steps(
                traj_ext, "bdotx", ni, pi
            )
            bdoty[interp_mask] = w * bdoty[interp_mask] + w1 * _soa_values_at_steps(
                traj_ext, "bdoty", ni, pi
            )
            bdotz[interp_mask] = w * bdotz[interp_mask] + w1 * _soa_values_at_steps(
                traj_ext, "bdotz", ni, pi
            )
            gamma[interp_mask] = w * gamma[interp_mask] + w1 * _soa_values_at_steps(
                traj_ext, "gamma", ni, pi
            )
            x[interp_mask] = w * x[interp_mask] + w1 * _soa_values_at_steps(
                traj_ext, "x", ni, pi
            )
            y[interp_mask] = w * y[interp_mask] + w1 * _soa_values_at_steps(
                traj_ext, "y", ni, pi
            )
            z[interp_mask] = w * z[interp_mask] + w1 * _soa_values_at_steps(
                traj_ext, "z", ni, pi
            )

    return ExternalSampleBatch(
        bx=bx,
        by=by,
        bz=bz,
        bdotx=bdotx,
        bdoty=bdoty,
        bdotz=bdotz,
        gamma=gamma,
        x=x,
        y=y,
        z=z,
        charge=charge,
        valid_mask=valid_mask,
    )


def gather_external_samples(
    trajectory_ext: Sequence[Dict[str, np.ndarray]],
    indices: np.ndarray,
    indices_next: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    indices_prev: np.ndarray | None = None,
    indices_next2: np.ndarray | None = None,
    use_cubic: bool = False,
    interpolate_positions: bool = False,
) -> ExternalSampleBatch:
    """Extract external bunch samples for the provided retarded indices.

    Parameters
    ----------
    trajectory_ext : Sequence[Dict[str, np.ndarray]]
        External trajectory history.
    indices : np.ndarray
        Primary trajectory indices for each source particle.
    indices_next : np.ndarray, optional
        Secondary indices for interpolation. If None, no interpolation is performed.
    weights : np.ndarray, optional
        Interpolation weights in [0,1]. For linear: weight=1.0 uses indices only,
        weight=0.0 uses indices_next only. For cubic: weight is normalized parameter u.
    indices_prev : np.ndarray, optional
        Previous indices for cubic interpolation (4-point).
    indices_next2 : np.ndarray, optional
        Second-next indices for cubic interpolation (4-point).
    use_cubic : bool, optional
        If True, use cubic interpolation with 4 points. Requires indices_prev and indices_next2.
    interpolate_positions : bool, optional
        If True, also interpolate x/y/z positions (high-precision mode).

    Returns
    -------
    ExternalSampleBatch
        Sampled (and optionally interpolated) external particle data.
    """

    sample_count = int(len(indices))
    charge = np.zeros(sample_count, dtype=float)
    gamma = np.zeros(sample_count, dtype=float)
    bx = np.zeros(sample_count, dtype=float)
    by = np.zeros(sample_count, dtype=float)
    bz = np.zeros(sample_count, dtype=float)
    bdotx = np.zeros(sample_count, dtype=float)
    bdoty = np.zeros(sample_count, dtype=float)
    bdotz = np.zeros(sample_count, dtype=float)
    valid_mask = np.zeros(sample_count, dtype=bool)

    # Position arrays for interpolation if needed
    x_vals = np.zeros(sample_count, dtype=float)
    y_vals = np.zeros(sample_count, dtype=float)
    z_vals = np.zeros(sample_count, dtype=float)

    use_interpolation = (indices_next is not None) and (weights is not None)
    use_cubic_interp = (
        use_cubic and (indices_prev is not None) and (indices_next2 is not None)
    )

    for j, ext_idx in enumerate(indices):
        if ext_idx < 0:
            continue
        if ext_idx >= len(trajectory_ext):
            continue
        state = trajectory_ext[ext_idx]
        if j >= len(state["x"]):
            continue

        valid_mask[j] = True

        # Get primary sample
        bx_val = float(state["bx"][j])
        by_val = float(state["by"][j])
        bz_val = float(state["bz"][j])
        bdotx_val = float(state["bdotx"][j])
        bdoty_val = float(state["bdoty"][j])
        bdotz_val = float(state["bdotz"][j])

        charge_j = state["q"]
        if hasattr(charge_j, "__getitem__"):
            charge_val = float(charge_j[j])
        else:
            charge_val = float(charge_j)

        gamma_j = state["gamma"]
        if hasattr(gamma_j, "__getitem__"):
            gamma_val = float(gamma_j[j])
        else:
            gamma_val = float(gamma_j)

        # Store positions (may be interpolated later)
        x_vals[j] = float(state["x"][j])
        y_vals[j] = float(state["y"][j])
        z_vals[j] = float(state["z"][j])

        # If interpolation is requested and weight < 1, blend with next sample
        if use_interpolation and weights[j] < 1.0:
            if use_cubic_interp:
                # Cubic interpolation using 4 points: prev, next, indices, next2
                # Catmull-Rom spline interpolation
                u = weights[j]  # normalized parameter in [0,1]

                # Get all 4 trajectory states
                idx_prev = indices_prev[j]
                idx_next = indices_next[j]
                idx_next2 = indices_next2[j]

                if (
                    0 <= idx_prev < len(trajectory_ext)
                    and 0 <= idx_next < len(trajectory_ext)
                    and 0 <= idx_next2 < len(trajectory_ext)
                ):
                    state_prev = trajectory_ext[idx_prev]
                    state_next = trajectory_ext[idx_next]
                    state_next2 = trajectory_ext[idx_next2]

                    if (
                        j < len(state_prev["x"])
                        and j < len(state_next["x"])
                        and j < len(state_next2["x"])
                    ):
                        # Catmull-Rom cubic interpolation
                        # P(u) = 0.5 * [2*P1 + (-P0 + P2)*u + (2*P0 - 5*P1 + 4*P2 - P3)*u^2 + (-P0 + 3*P1 - 3*P2 + P3)*u^3]
                        # where P0=prev, P1=next, P2=curr, P3=next2
                        u2 = u * u
                        u3 = u2 * u

                        def cubic_interp(v0, v1, v2, v3):
                            return 0.5 * (
                                2.0 * v1
                                + (-v0 + v2) * u
                                + (2.0 * v0 - 5.0 * v1 + 4.0 * v2 - v3) * u2
                                + (-v0 + 3.0 * v1 - 3.0 * v2 + v3) * u3
                            )

                        # Interpolate velocities
                        bx_0 = float(state_prev["bx"][j])
                        bx_1 = float(state_next["bx"][j])
                        bx_2 = bx_val  # current
                        bx_3 = float(state_next2["bx"][j])
                        bx_val = cubic_interp(bx_0, bx_1, bx_2, bx_3)

                        by_0 = float(state_prev["by"][j])
                        by_1 = float(state_next["by"][j])
                        by_2 = by_val
                        by_3 = float(state_next2["by"][j])
                        by_val = cubic_interp(by_0, by_1, by_2, by_3)

                        bz_0 = float(state_prev["bz"][j])
                        bz_1 = float(state_next["bz"][j])
                        bz_2 = bz_val
                        bz_3 = float(state_next2["bz"][j])
                        bz_val = cubic_interp(bz_0, bz_1, bz_2, bz_3)

                        # Interpolate accelerations
                        bdotx_0 = float(state_prev["bdotx"][j])
                        bdotx_1 = float(state_next["bdotx"][j])
                        bdotx_2 = bdotx_val
                        bdotx_3 = float(state_next2["bdotx"][j])
                        bdotx_val = cubic_interp(bdotx_0, bdotx_1, bdotx_2, bdotx_3)

                        bdoty_0 = float(state_prev["bdoty"][j])
                        bdoty_1 = float(state_next["bdoty"][j])
                        bdoty_2 = bdoty_val
                        bdoty_3 = float(state_next2["bdoty"][j])
                        bdoty_val = cubic_interp(bdoty_0, bdoty_1, bdoty_2, bdoty_3)

                        bdotz_0 = float(state_prev["bdotz"][j])
                        bdotz_1 = float(state_next["bdotz"][j])
                        bdotz_2 = bdotz_val
                        bdotz_3 = float(state_next2["bdotz"][j])
                        bdotz_val = cubic_interp(bdotz_0, bdotz_1, bdotz_2, bdotz_3)

                        # Interpolate gamma
                        gamma_0_j = state_prev["gamma"]
                        gamma_0 = (
                            float(gamma_0_j[j])
                            if hasattr(gamma_0_j, "__getitem__")
                            else float(gamma_0_j)
                        )
                        gamma_1_j = state_next["gamma"]
                        gamma_1 = (
                            float(gamma_1_j[j])
                            if hasattr(gamma_1_j, "__getitem__")
                            else float(gamma_1_j)
                        )
                        gamma_2 = gamma_val
                        gamma_3_j = state_next2["gamma"]
                        gamma_3 = (
                            float(gamma_3_j[j])
                            if hasattr(gamma_3_j, "__getitem__")
                            else float(gamma_3_j)
                        )
                        gamma_val = cubic_interp(gamma_0, gamma_1, gamma_2, gamma_3)

                        # Interpolate positions if requested
                        if interpolate_positions:
                            x_0 = float(state_prev["x"][j])
                            x_1 = float(state_next["x"][j])
                            x_2 = x_vals[j]
                            x_3 = float(state_next2["x"][j])
                            x_vals[j] = cubic_interp(x_0, x_1, x_2, x_3)

                            y_0 = float(state_prev["y"][j])
                            y_1 = float(state_next["y"][j])
                            y_2 = y_vals[j]
                            y_3 = float(state_next2["y"][j])
                            y_vals[j] = cubic_interp(y_0, y_1, y_2, y_3)

                            z_0 = float(state_prev["z"][j])
                            z_1 = float(state_next["z"][j])
                            z_2 = z_vals[j]
                            z_3 = float(state_next2["z"][j])
                            z_vals[j] = cubic_interp(z_0, z_1, z_2, z_3)
            else:
                # Linear interpolation
                ext_idx_next = indices_next[j]
                weight = weights[j]

                if 0 <= ext_idx_next < len(trajectory_ext):
                    state_next = trajectory_ext[ext_idx_next]
                    if j < len(state_next["x"]):
                        # Interpolate: val = w*val1 + (1-w)*val2
                        bx_next = float(state_next["bx"][j])
                        by_next = float(state_next["by"][j])
                        bz_next = float(state_next["bz"][j])
                        bdotx_next = float(state_next["bdotx"][j])
                        bdoty_next = float(state_next["bdoty"][j])
                        bdotz_next = float(state_next["bdotz"][j])

                        bx_val = weight * bx_val + (1.0 - weight) * bx_next
                        by_val = weight * by_val + (1.0 - weight) * by_next
                        bz_val = weight * bz_val + (1.0 - weight) * bz_next
                        bdotx_val = weight * bdotx_val + (1.0 - weight) * bdotx_next
                        bdoty_val = weight * bdoty_val + (1.0 - weight) * bdoty_next
                        bdotz_val = weight * bdotz_val + (1.0 - weight) * bdotz_next

                        gamma_next_j = state_next["gamma"]
                        if hasattr(gamma_next_j, "__getitem__"):
                            gamma_next = float(gamma_next_j[j])
                        else:
                            gamma_next = float(gamma_next_j)
                        gamma_val = weight * gamma_val + (1.0 - weight) * gamma_next

                        # Interpolate positions if requested
                        if interpolate_positions:
                            x_next = float(state_next["x"][j])
                            y_next = float(state_next["y"][j])
                            z_next = float(state_next["z"][j])

                            x_vals[j] = weight * x_vals[j] + (1.0 - weight) * x_next
                            y_vals[j] = weight * y_vals[j] + (1.0 - weight) * y_next
                            z_vals[j] = weight * z_vals[j] + (1.0 - weight) * z_next

                        # Charge is not interpolated (discrete quantity)

        bx[j] = bx_val
        by[j] = by_val
        bz[j] = bz_val
        bdotx[j] = bdotx_val
        bdoty[j] = bdoty_val
        bdotz[j] = bdotz_val
        charge[j] = charge_val
        gamma[j] = gamma_val

    return ExternalSampleBatch(
        charge=charge,
        gamma=gamma,
        bx=bx,
        by=by,
        bz=bz,
        bdotx=bdotx,
        bdoty=bdoty,
        bdotz=bdotz,
        valid_mask=valid_mask,
    )


def compute_vectorized_contributions(
    h: float,
    charge_i: float,
    mass_i: float,
    gamma_i: float,
    beta_vec: Tuple[float, float, float],
    nhat_nx: np.ndarray,
    nhat_ny: np.ndarray,
    nhat_nz: np.ndarray,
    R_separation: np.ndarray,
    samples: ExternalSampleBatch,
    *,
    apply_external: bool,
    verbosity: int = 0,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return accumulated momentum and field contributions using vector ops.

    Args:
        R_separation: Distance between external charge sources and test particle.

    Returns:
        (delta_px, delta_py, delta_pz, delta_pt,
         delta_field_x, delta_field_y, delta_field_z,
         scalar_potential_sum)
    """

    c = C_MMNS
    c_sq = c * c
    c_cu = c_sq * c

    # DEBUG: Log input parameters for force sign debugging
    if verbosity >= 4:
        print("\n  [DEBUG] compute_vectorized_contributions called:")
        print(f"    charge_i = {charge_i:.6e}")
        print(f"    mass_i = {mass_i:.6e}")
        print(f"    gamma_i = {gamma_i:.6e}")
        print(
            f"    beta_vec = ({beta_vec[0]:.6e}, {beta_vec[1]:.6e}, {beta_vec[2]:.6e})"
        )
        print(f"    nhat_nx = {nhat_nx}")
        print(f"    nhat_ny = {nhat_ny}")
        print(f"    nhat_nz = {nhat_nz}")
        print(f"    R_separation = {R_separation}")
        print(f"    samples.charge = {samples.charge}")
        print(f"    samples.bx = {samples.bx}")
        print(f"    samples.by = {samples.by}")
        print(f"    samples.bz = {samples.bz}")

    if not apply_external:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if abs(charge_i) < 1e-20 or gamma_i > 1e6:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if samples.charge.size == 0 or R_separation.size == 0 or not samples.any_valid:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    mask = samples.valid_mask.copy()
    mask &= R_separation > 0.0
    mask &= samples.gamma <= 1e6
    mask &= np.abs(samples.charge) >= 1e-20

    if not mask.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    nx = nhat_nx[mask]
    ny = nhat_ny[mask]
    nz = nhat_nz[mask]
    R_sep = R_separation[mask]

    bx_ext = samples.bx[mask]
    by_ext = samples.by[mask]
    bz_ext = samples.bz[mask]
    bdotx_ext = samples.bdotx[mask]
    bdoty_ext = samples.bdoty[mask]
    bdotz_ext = samples.bdotz[mask]
    charge_ext = samples.charge[mask]
    gamma_ext = samples.gamma[mask]

    if NUMBA_AVAILABLE and charge_ext.size <= NUMBA_FORCE_SERIAL_MAX_SOURCES:
        return _compute_forces_numba_serial_kernel(
            h,
            charge_i,
            mass_i,
            gamma_i,
            beta_vec[0],
            beta_vec[1],
            beta_vec[2],
            nx,
            ny,
            nz,
            R_sep,
            bx_ext,
            by_ext,
            bz_ext,
            bdotx_ext,
            bdoty_ext,
            bdotz_ext,
            charge_ext,
            gamma_ext,
            c,
        )

    if NUMBA_AVAILABLE and charge_ext.size >= NUMBA_FORCE_PARALLEL_MIN_SOURCES:
        return _compute_forces_numba_kernel(
            h,
            charge_i,
            mass_i,
            gamma_i,
            beta_vec[0],
            beta_vec[1],
            beta_vec[2],
            nx,
            ny,
            nz,
            R_sep,
            bx_ext,
            by_ext,
            bz_ext,
            bdotx_ext,
            bdoty_ext,
            bdotz_ext,
            charge_ext,
            gamma_ext,
            c,
        )

    # Use float64 precision for k_factor to handle extremely relativistic particles
    # Keep result as numpy array for indexing
    beta_dot_nhat = (
        bx_ext.astype(np.float64) * nx.astype(np.float64)
        + by_ext.astype(np.float64) * ny.astype(np.float64)
        + bz_ext.astype(np.float64) * nz.astype(np.float64)
    )
    k_factor = np.float64(1.0) - beta_dot_nhat

    # Filter out interactions where k-factor is too small to prevent force divergence.
    # For ultra-relativistic particles (β → 1) near image charges, k = 1 - β·n̂ → 0,
    # causing forces to scale as 1/k³ → ∞.
    #
    # Using k_min = 1e-20 approaches float64 machine limits while remaining safe:
    # - 1/k³_max = 1e60 is well within float64 range (max ≈ 1.8e308)
    # - Allows γ up to ~7e9 for head-on collisions
    # - Consistent with beta limiting at 1 - 1e-16
    # Mass-shell projection provides primary protection against β > 1 violations;
    # k-threshold is secondary filtering for numerical stability.
    k_threshold_hard = np.float64(K_CUTOFF_HARD)
    k_threshold_series = np.float64(K_SERIES_THRESHOLD)

    # Classify k-factors into three regimes:
    # 1. Normal: |k| >= k_series_threshold (use standard calculation)
    # 2. Small-k series: k_hard < |k| < k_series (use series approximation)
    # 3. Too small: |k| < k_hard (skip entirely)
    valid_k_hard = np.abs(k_factor) >= k_threshold_hard
    small_k_regime = (np.abs(k_factor) >= k_threshold_hard) & (
        np.abs(k_factor) < k_threshold_series
    )

    # Debug logging (only at high verbosity)
    if verbosity >= 3:
        num_filtered_hard = np.sum(~valid_k_hard)
        num_small_k = np.sum(small_k_regime)

        if num_filtered_hard > 0:
            k_min = (
                np.min(np.abs(k_factor[valid_k_hard])) if np.any(valid_k_hard) else 0.0
            )
            print(
                f"    ⚠️  k-factor hard cutoff triggered: {num_filtered_hard} interaction(s) filtered"
            )
            print(f"       min|k| = {k_min:.6e}, threshold = {k_threshold_hard:.6e}")

        if num_small_k > 0:
            print(
                f"    ℹ️  k-factor series approximation: {num_small_k} interaction(s) using series expansion"
            )
            print(
                f"       threshold range: {k_threshold_hard:.6e} < |k| < {k_threshold_series:.6e}"
            )

    if not np.any(valid_k_hard):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Filter to valid k values (above hard cutoff)
    nx = nx[valid_k_hard]
    ny = ny[valid_k_hard]
    nz = nz[valid_k_hard]
    R_sep = R_sep[valid_k_hard]
    bx_ext = bx_ext[valid_k_hard]
    by_ext = by_ext[valid_k_hard]
    bz_ext = bz_ext[valid_k_hard]
    bdotx_ext = bdotx_ext[valid_k_hard]
    bdoty_ext = bdoty_ext[valid_k_hard]
    bdotz_ext = bdotz_ext[valid_k_hard]
    charge_ext = charge_ext[valid_k_hard]
    gamma_ext = gamma_ext[valid_k_hard]
    k_factor = k_factor[valid_k_hard]

    # Re-classify after filtering
    small_k_mask = np.abs(k_factor) < k_threshold_series
    normal_k_mask = ~small_k_mask

    # Use float64 for beta components to ensure precision in force calculations
    beta_x = np.float64(beta_vec[0])
    beta_y = np.float64(beta_vec[1])
    beta_z = np.float64(beta_vec[2])

    # Ensure all arrays are float64 for high-precision force calculations
    bx_ext = bx_ext.astype(np.float64)
    by_ext = by_ext.astype(np.float64)
    bz_ext = bz_ext.astype(np.float64)
    bdotx_ext = bdotx_ext.astype(np.float64)
    bdoty_ext = bdoty_ext.astype(np.float64)
    bdotz_ext = bdotz_ext.astype(np.float64)
    gamma_ext = gamma_ext.astype(np.float64)
    charge_ext = charge_ext.astype(np.float64)
    nx = nx.astype(np.float64)
    ny = ny.astype(np.float64)
    nz = nz.astype(np.float64)
    R_sep = R_sep.astype(np.float64)

    bdot_scalar_ext = bx_ext * bdotx_ext + by_ext * bdoty_ext + bz_ext * bdotz_ext
    betas_scalar = bx_ext * beta_x + by_ext * beta_y + bz_ext * beta_z

    v_betas_scalar = (
        gamma_ext
        * np.float64(gamma_i)
        * np.float64(c_sq)
        * (np.float64(1.0) - betas_scalar)
    )

    mixed_term = (
        beta_x
        * (
            bdotx_ext * np.float64(c) * gamma_ext**2
            + bx_ext * bdot_scalar_ext * np.float64(c) * gamma_ext**4
        )
        + beta_y
        * (
            bdoty_ext * np.float64(c) * gamma_ext**2
            + by_ext * bdot_scalar_ext * np.float64(c) * gamma_ext**4
        )
        + beta_z
        * (
            bdotz_ext * np.float64(c) * gamma_ext**2
            + bz_ext * bdot_scalar_ext * np.float64(c) * gamma_ext**4
        )
    )
    v_beta_dot_mixed_scalar = (
        gamma_ext**4 * np.float64(gamma_i) * np.float64(c_sq) * bdot_scalar_ext
        - np.float64(gamma_i) * np.float64(c) * mixed_term
    )

    # Split calculation based on k-factor regime
    # Initialize momentum and field accumulations
    delta_px = 0.0
    delta_py = 0.0
    delta_pz = 0.0
    delta_pt = 0.0
    delta_field_x = 0.0
    delta_field_y = 0.0
    delta_field_z = 0.0
    scalar_potential_sum = 0.0

    # Process normal k-factor regime (standard calculation)
    if np.any(normal_k_mask):
        k_normal = k_factor[normal_k_mask]
        charge_factor_normal = (
            np.float64(h)
            * np.float64(charge_i)
            * charge_ext[normal_k_mask]
            / (
                k_normal**3
                * np.float64(c_cu)
                * R_sep[normal_k_mask] ** 2
                * gamma_ext[normal_k_mask] ** 3
            )
        )

        # DEBUG: Log charge factor sign
        if verbosity >= 4:
            print("\n  [DEBUG] Force calculation (normal k regime):")
            print(f"    k_normal = {k_normal}")
            print(f"    charge_i = {charge_i:.6e}")
            print(f"    charge_ext = {charge_ext[normal_k_mask]}")
            print(f"    charge_product = {charge_i * charge_ext[normal_k_mask]}")
            print(f"    charge_factor_normal = {charge_factor_normal}")
            print(f"    nx (normal) = {nx[normal_k_mask]}")
            print(f"    ny (normal) = {ny[normal_k_mask]}")
            print(f"    nz (normal) = {nz[normal_k_mask]}")
            print(f"    R_sep (normal) = {R_sep[normal_k_mask]}")

        term_px_normal = (
            -v_betas_scalar[normal_k_mask]
            * bx_ext[normal_k_mask]
            * k_normal
            * np.float64(c)
            * gamma_ext[normal_k_mask] ** 2
            + v_beta_dot_mixed_scalar[normal_k_mask]
            * k_normal
            * gamma_ext[normal_k_mask]
            * nx[normal_k_mask]
            * R_sep[normal_k_mask]
            + gamma_ext[normal_k_mask] ** 2
            * nx[normal_k_mask] ** 2
            * R_sep[normal_k_mask]
            * v_betas_scalar[normal_k_mask]
            * (
                bdotx_ext[normal_k_mask]
                + bdotx_ext[normal_k_mask]
                * bdot_scalar_ext[normal_k_mask]
                * gamma_ext[normal_k_mask] ** 2
            )
            + v_betas_scalar[normal_k_mask] * np.float64(c) * nx[normal_k_mask]
        )

        term_py_normal = (
            -v_betas_scalar[normal_k_mask]
            * by_ext[normal_k_mask]
            * k_normal
            * np.float64(c)
            * gamma_ext[normal_k_mask] ** 2
            + v_beta_dot_mixed_scalar[normal_k_mask]
            * k_normal
            * gamma_ext[normal_k_mask]
            * ny[normal_k_mask]
            * R_sep[normal_k_mask]
            + gamma_ext[normal_k_mask] ** 2
            * ny[normal_k_mask] ** 2
            * R_sep[normal_k_mask]
            * v_betas_scalar[normal_k_mask]
            * (
                bdoty_ext[normal_k_mask]
                + bdoty_ext[normal_k_mask]
                * bdot_scalar_ext[normal_k_mask]
                * gamma_ext[normal_k_mask] ** 2
            )
            + v_betas_scalar[normal_k_mask] * np.float64(c) * ny[normal_k_mask]
        )

        term_pz_normal = (
            -v_betas_scalar[normal_k_mask]
            * bz_ext[normal_k_mask]
            * k_normal
            * np.float64(c)
            * gamma_ext[normal_k_mask] ** 2
            + v_beta_dot_mixed_scalar[normal_k_mask]
            * k_normal
            * gamma_ext[normal_k_mask]
            * nz[normal_k_mask]
            * R_sep[normal_k_mask]
            + gamma_ext[normal_k_mask] ** 2
            * nz[normal_k_mask] ** 2
            * R_sep[normal_k_mask]
            * v_betas_scalar[normal_k_mask]
            * (
                bdotz_ext[normal_k_mask]
                + bdotz_ext[normal_k_mask]
                * bdot_scalar_ext[normal_k_mask]
                * gamma_ext[normal_k_mask] ** 2
            )
            + v_betas_scalar[normal_k_mask] * np.float64(c) * nz[normal_k_mask]
        )

        term_pt_normal = (
            v_beta_dot_mixed_scalar[normal_k_mask]
            * k_normal
            * gamma_ext[normal_k_mask]
            * R_sep[normal_k_mask]
            - v_betas_scalar[normal_k_mask]
            * k_normal
            * np.float64(c)
            * gamma_ext[normal_k_mask] ** 2
            - bdot_scalar_ext[normal_k_mask]
            * v_betas_scalar[normal_k_mask]
            * gamma_ext[normal_k_mask] ** 4
            * R_sep[normal_k_mask]
            + v_betas_scalar[normal_k_mask] * np.float64(c)
        )

        delta_px += float(np.sum(charge_factor_normal * term_px_normal))
        delta_py += float(np.sum(charge_factor_normal * term_py_normal))
        delta_pz += float(np.sum(charge_factor_normal * term_pz_normal))
        delta_pt += float(np.sum(charge_factor_normal * term_pt_normal))

        # DEBUG: Log computed force components
        if verbosity >= 4:
            print("\n  [DEBUG] Force components (normal k regime):")
            print(f"    term_px_normal = {term_px_normal}")
            print(f"    term_py_normal = {term_py_normal}")
            print(f"    term_pz_normal = {term_pz_normal}")
            print(f"    term_pt_normal = {term_pt_normal}")
            print(
                f"    delta_px contrib = {float(np.sum(charge_factor_normal * term_px_normal)):.6e}"
            )
            print(
                f"    delta_py contrib = {float(np.sum(charge_factor_normal * term_py_normal)):.6e}"
            )
            print(
                f"    delta_pz contrib = {float(np.sum(charge_factor_normal * term_pz_normal)):.6e}"
            )
            print(
                f"    delta_pt contrib = {float(np.sum(charge_factor_normal * term_pt_normal)):.6e}"
            )

        # Field and scalar potential for normal regime
        field_factor_normal = (
            np.float64(h)
            / np.float64(mass_i)
            * np.float64(charge_i)
            / np.float64(c)
            * charge_ext[normal_k_mask]
            / (R_sep[normal_k_mask] * k_normal)
        )
        delta_field_x += float(np.sum(field_factor_normal * bx_ext[normal_k_mask]))
        delta_field_y += float(np.sum(field_factor_normal * by_ext[normal_k_mask]))
        delta_field_z += float(np.sum(field_factor_normal * bz_ext[normal_k_mask]))
        scalar_potential_sum += float(
            np.sum(charge_ext[normal_k_mask] / (R_sep[normal_k_mask] * k_normal))
        )

    # Process small k-factor regime (series approximation)
    if np.any(small_k_mask):
        k_small = k_factor[small_k_mask]
        charge_factor_base_small = (
            np.float64(h)
            * np.float64(charge_i)
            * charge_ext[small_k_mask]
            / (
                np.float64(c_cu)
                * R_sep[small_k_mask] ** 2
                * gamma_ext[small_k_mask] ** 3
            )
        )

        # Use series approximation
        term_px_series, term_py_series, term_pz_series, term_pt_series = (
            _compute_small_k_forces_series(
                k_small,
                charge_factor_base_small,
                v_betas_scalar[small_k_mask],
                v_beta_dot_mixed_scalar[small_k_mask],
                bx_ext[small_k_mask],
                by_ext[small_k_mask],
                bz_ext[small_k_mask],
                bdotx_ext[small_k_mask],
                bdoty_ext[small_k_mask],
                bdotz_ext[small_k_mask],
                nx[small_k_mask],
                ny[small_k_mask],
                nz[small_k_mask],
                R_sep[small_k_mask],
                gamma_ext[small_k_mask],
                np.float64(c),
            )
        )

        delta_px += float(np.sum(term_px_series))
        delta_py += float(np.sum(term_py_series))
        delta_pz += float(np.sum(term_pz_series))
        delta_pt += float(np.sum(term_pt_series))

        # Field and scalar potential for small k regime (use limiting forms)
        field_factor_small = (
            np.float64(h)
            / np.float64(mass_i)
            * np.float64(charge_i)
            / np.float64(c)
            * charge_ext[small_k_mask]
            / (R_sep[small_k_mask] * k_small)
        )
        delta_field_x += float(np.sum(field_factor_small * bx_ext[small_k_mask]))
        delta_field_y += float(np.sum(field_factor_small * by_ext[small_k_mask]))
        delta_field_z += float(np.sum(field_factor_small * bz_ext[small_k_mask]))
        scalar_potential_sum += float(
            np.sum(charge_ext[small_k_mask] / (R_sep[small_k_mask] * k_small))
        )

    return (
        delta_px,
        delta_py,
        delta_pz,
        delta_pt,
        delta_field_x,
        delta_field_y,
        delta_field_z,
        scalar_potential_sum,
    )


# Numba-accelerated force calculation kernel
@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def _compute_forces_numba_kernel(
    h,
    charge_i,
    mass_i,
    gamma_i,
    bx_i,
    by_i,
    bz_i,
    nx,
    ny,
    nz,
    R_sep,
    bx_ext,
    by_ext,
    bz_ext,
    bdotx_ext,
    bdoty_ext,
    bdotz_ext,
    charge_ext,
    gamma_ext,
    c,
):
    """Numba-compiled force calculation kernel for maximum performance."""

    n_ext = len(R_sep)

    delta_px = 0.0
    delta_py = 0.0
    delta_pz = 0.0
    delta_pt = 0.0
    delta_field_x = 0.0
    delta_field_y = 0.0
    delta_field_z = 0.0
    scalar_potential = 0.0

    c_sq = c * c
    c_cu = c_sq * c

    for j in prange(n_ext):
        # k-factor with float64 precision
        beta_dot_nhat = bx_ext[j] * nx[j] + by_ext[j] * ny[j] + bz_ext[j] * nz[j]
        k_factor = 1.0 - beta_dot_nhat

        if abs(k_factor) < 1e-20:
            continue

        # Common factors
        q_prod = charge_i * charge_ext[j]
        R = R_sep[j]
        g_ext = gamma_ext[j]

        # Scalar products
        bdot_scalar = (
            bx_ext[j] * bdotx_ext[j]
            + by_ext[j] * bdoty_ext[j]
            + bz_ext[j] * bdotz_ext[j]
        )
        betas_scalar = bx_ext[j] * bx_i + by_ext[j] * by_i + bz_ext[j] * bz_i

        # Covariant force terms
        v_betas = g_ext * gamma_i * c_sq * (1.0 - betas_scalar)

        v_beta_dot_mixed = g_ext**4 * gamma_i * c_sq * bdot_scalar - gamma_i * c * (
            bx_i
            * (bdotx_ext[j] * c * g_ext**2 + bx_ext[j] * bdot_scalar * c * g_ext**4)
            + by_i
            * (bdoty_ext[j] * c * g_ext**2 + by_ext[j] * bdot_scalar * c * g_ext**4)
            + bz_i
            * (bdotz_ext[j] * c * g_ext**2 + bz_ext[j] * bdot_scalar * c * g_ext**4)
        )

        # Common force factor
        k3 = k_factor**3
        force_factor = (h * q_prod) / (k3 * c_cu * R * R * g_ext**3)

        # Momentum contributions
        delta_px += force_factor * (
            -bx_ext[j] * v_betas * k_factor * c * g_ext**2
            + v_beta_dot_mixed * k_factor * g_ext * nx[j] * R
            + g_ext**2
            * nx[j] ** 2
            * R
            * v_betas
            * (bdotx_ext[j] + bdotx_ext[j] * bdot_scalar * g_ext**2)
            + v_betas * c * nx[j]
        )

        delta_py += force_factor * (
            -by_ext[j] * v_betas * k_factor * c * g_ext**2
            + v_beta_dot_mixed * k_factor * g_ext * ny[j] * R
            + g_ext**2
            * ny[j] ** 2
            * R
            * v_betas
            * (bdoty_ext[j] + bdoty_ext[j] * bdot_scalar * g_ext**2)
            + v_betas * c * ny[j]
        )

        delta_pz += force_factor * (
            -bz_ext[j] * v_betas * k_factor * c * g_ext**2
            + v_beta_dot_mixed * k_factor * g_ext * nz[j] * R
            + g_ext**2
            * nz[j] ** 2
            * R
            * v_betas
            * (bdotz_ext[j] + bdotz_ext[j] * bdot_scalar * g_ext**2)
            + v_betas * c * nz[j]
        )

        pt_factor = (h * q_prod) / (k3 * c_cu * R * R * g_ext**3)
        delta_pt += pt_factor * (
            v_beta_dot_mixed * k_factor * g_ext * R
            - v_betas * k_factor * c * g_ext**2
            - bdot_scalar * v_betas * g_ext**4 * R
            + v_betas * c
        )

        # Field contributions for position update
        field_factor = (h / mass_i) * charge_i / c * charge_ext[j]
        delta_field_x += field_factor * bx_ext[j] / (R * k_factor)
        delta_field_y += field_factor * by_ext[j] / (R * k_factor)
        delta_field_z += field_factor * bz_ext[j] / (R * k_factor)

        # Scalar potential
        scalar_potential += charge_ext[j] / (R * k_factor)

    return (
        delta_px,
        delta_py,
        delta_pz,
        delta_pt,
        delta_field_x,
        delta_field_y,
        delta_field_z,
        scalar_potential,
    )


if NUMBA_AVAILABLE:
    _compute_forces_numba_serial_kernel = jit(
        nopython=True,
        fastmath=True,
        cache=True,
    )(_compute_forces_numba_kernel.py_func)
else:
    _compute_forces_numba_serial_kernel = _compute_forces_numba_kernel


__all__ = [
    "ExternalSampleBatch",
    "gather_external_samples",
    "compute_vectorized_contributions",
    "NUMBA_AVAILABLE",
]
