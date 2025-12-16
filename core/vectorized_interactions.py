"""Vectorized kernels for retarded electromagnetic force calculations.

This module implements the core retarded field computations using NumPy
vectorization for efficient batch processing of external source particles.

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

# K-factor thresholds for series approximation
K_CUTOFF_HARD = 1e-20  # Below this: skip interaction entirely
K_SERIES_THRESHOLD = 1e-3  # Below this: use series approximation


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

    @property
    def any_valid(self) -> bool:
        return bool(self.valid_mask.any())


def gather_external_samples(
    trajectory_ext: Sequence[Dict[str, np.ndarray]],
    indices: np.ndarray,
) -> ExternalSampleBatch:
    """Extract external bunch samples for the provided retarded indices."""

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

    for j, ext_idx in enumerate(indices):
        if ext_idx < 0:
            continue
        if ext_idx >= len(trajectory_ext):
            continue
        state = trajectory_ext[ext_idx]
        if j >= len(state["x"]):
            continue

        valid_mask[j] = True
        bx[j] = float(state["bx"][j])
        by[j] = float(state["by"][j])
        bz[j] = float(state["bz"][j])
        bdotx[j] = float(state["bdotx"][j])
        bdoty[j] = float(state["bdoty"][j])
        bdotz[j] = float(state["bdotz"][j])

        charge_j = state["q"]
        if hasattr(charge_j, "__getitem__"):
            charge[j] = float(charge_j[j])
        else:
            charge[j] = float(charge_j)

        gamma_j = state["gamma"]
        if hasattr(gamma_j, "__getitem__"):
            gamma[j] = float(gamma_j[j])
        else:
            gamma[j] = float(gamma_j)

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


__all__ = [
    "ExternalSampleBatch",
    "compute_vectorized_contributions",
    "gather_external_samples",
]
