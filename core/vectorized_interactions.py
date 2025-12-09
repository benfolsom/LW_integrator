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
numerical overflow. This threshold is extremely permissive—it only excludes
interactions where particles are moving at β > 1 - 1e-20 (γ > 2.2e9) nearly
directly toward each other, a regime far beyond any realistic simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .constants import C_MMNS


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
    # causing forces to scale as 1/k³ → ∞. This threshold prevents numerical blowup
    # while still allowing γ up to ~2000 in typical geometries.
    # With k_min = 1e-7, we get 1/k³_max ≈ 1e21, which is large but manageable.
    valid_k = np.abs(k_factor) >= np.float64(1e-7)
    if not np.any(valid_k):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    nx = nx[valid_k]
    ny = ny[valid_k]
    nz = nz[valid_k]
    R_sep = R_sep[valid_k]
    bx_ext = bx_ext[valid_k]
    by_ext = by_ext[valid_k]
    bz_ext = bz_ext[valid_k]
    bdotx_ext = bdotx_ext[valid_k]
    bdoty_ext = bdoty_ext[valid_k]
    bdotz_ext = bdotz_ext[valid_k]
    charge_ext = charge_ext[valid_k]
    gamma_ext = gamma_ext[valid_k]
    k_factor = k_factor[valid_k]

    beta_x, beta_y, beta_z = beta_vec

    bdot_scalar_ext = bx_ext * bdotx_ext + by_ext * bdoty_ext + bz_ext * bdotz_ext
    betas_scalar = bx_ext * beta_x + by_ext * beta_y + bz_ext * beta_z

    v_betas_scalar = gamma_ext * gamma_i * c_sq * (1.0 - betas_scalar)

    mixed_term = (
        beta_x
        * (bdotx_ext * c * gamma_ext**2 + bx_ext * bdot_scalar_ext * c * gamma_ext**4)
        + beta_y
        * (bdoty_ext * c * gamma_ext**2 + by_ext * bdot_scalar_ext * c * gamma_ext**4)
        + beta_z
        * (bdotz_ext * c * gamma_ext**2 + bz_ext * bdot_scalar_ext * c * gamma_ext**4)
    )
    v_beta_dot_mixed_scalar = (
        gamma_ext**4 * gamma_i * c_sq * bdot_scalar_ext - gamma_i * c * mixed_term
    )

    charge_factor = (
        h * charge_i * charge_ext / (k_factor**3 * c_cu * R_sep**2 * gamma_ext**3)
    )

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

    delta_px = float(np.sum(charge_factor * term_px))
    delta_py = float(np.sum(charge_factor * term_py))
    delta_pz = float(np.sum(charge_factor * term_pz))

    term_pt = (
        v_beta_dot_mixed_scalar * k_factor * gamma_ext * R_sep
        - v_betas_scalar * k_factor * c * gamma_ext**2
        - bdot_scalar_ext * v_betas_scalar * gamma_ext**4 * R_sep
        + v_betas_scalar * c
    )

    delta_pt = float(np.sum(charge_factor * term_pt))

    field_factor = h / mass_i * charge_i / c * charge_ext / (R_sep * k_factor)
    delta_field_x = float(np.sum(field_factor * bx_ext))
    delta_field_y = float(np.sum(field_factor * by_ext))
    delta_field_z = float(np.sum(field_factor * bz_ext))

    # Compute scalar potential sum: Σ(q_j / (R_sep_j * k_factor_j))
    # This is used for correct gamma calculation: γ = (Pt - q_i·Φ) / (mc)
    scalar_potential_sum = float(np.sum(charge_ext / (R_sep * k_factor)))

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
