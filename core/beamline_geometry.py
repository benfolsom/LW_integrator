"""Geometry-based line-of-sight screening for retarded field calculations.

Beam-pipe-like occluders block direct retarded field contributions when the
source particle (at its retarded position) is outside the pipe's transverse
aperture. Residual fields arrive naturally because the test is applied at
the retarded source position.
"""

from __future__ import annotations

import numpy as np

from .types import BeamlineGeometryConfig, Occluder


def _occluder_transverse_distance_sq(
    positions: np.ndarray,
    occluder: Occluder,
) -> np.ndarray:
    """Squared transverse distance of each position from the occluder axis.

    ``positions`` has shape (N, 3). Returns shape (N,).
    """
    axis = np.asarray(occluder.axis, dtype=float)
    center = np.asarray(occluder.center_mm, dtype=float)
    rel = positions - center  # (N, 3)
    # Project onto axis
    axial = rel @ axis  # (N,)
    transverse = rel - np.outer(axial, axis)  # (N, 3)
    return np.einsum("ij,ij->i", transverse, transverse)


def _occluder_axial_position(
    positions: np.ndarray,
    occluder: Occluder,
) -> np.ndarray:
    """Signed axial position of each particle along the occluder axis.

    Zero at the center; the cylinder extends from -length/2 to +length/2.
    """
    axis = np.asarray(occluder.axis, dtype=float)
    center = np.asarray(occluder.center_mm, dtype=float)
    rel = positions - center
    return rel @ axis


def compute_visibility_mask(
    source_positions: np.ndarray,
    geometry: BeamlineGeometryConfig,
) -> np.ndarray:
    """Return a boolean mask: True where the source is visible (not occluded).

    A source particle is visible if it is inside at least one occluder's
    transverse aperture (within radius) AND within that occluder's axial
    extent (within length/2 of center along axis).

    ``source_positions`` has shape (N, 3). Returns shape (N,) bool array.
    If geometry is disabled or has no occluders, all positions are visible.
    """
    if not geometry.enabled or not geometry.occluders:
        return np.ones(source_positions.shape[0], dtype=bool)

    positions = np.asarray(source_positions, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)

    visible = np.zeros(positions.shape[0], dtype=bool)
    half_length = 0.0
    radius_sq = 0.0
    for occluder in geometry.occluders:
        dist_sq = _occluder_transverse_distance_sq(positions, occluder)
        axial = _occluder_axial_position(positions, occluder)
        half_length = occluder.length_mm * 0.5
        radius_sq = occluder.radius_mm * occluder.radius_mm
        inside = (dist_sq < radius_sq) & (np.abs(axial) <= half_length)
        visible |= inside
    return visible


def compute_directional_visibility_mask(
    source_positions: np.ndarray,
    geometry: BeamlineGeometryConfig,
    observer_direction: tuple[float, float, float],
) -> np.ndarray:
    """Direction-specific visibility: source must be inside the observer's pipe.

    For each source particle, the relevant occluder is the one whose axis is
    most aligned with the observer's propagation direction. The source is
    visible only if it is inside that occluder's transverse aperture and axial
    extent. This models the physical geometry: a driver particle inside the
    electron pipe (z-axis) has line of sight down z to the rider; once it
    exits the electron pipe (``|y| > R``), its fields can no longer propagate
    along z to reach the rider.

    Parameters
    ----------
    source_positions: shape (N, 3), retarded source positions.
    geometry: beamline geometry config.
    observer_direction: the observer bunch's propagation direction (e.g.
        (0,0,1) for a +z rider). The occluder whose axis is most aligned
        with this direction is selected as the line-of-sight pipe.

    Returns
    -------
    Boolean mask of shape (N,). True = visible (source inside the
    observer's pipe). If geometry is disabled or has no occluders, all
    positions are visible.
    """
    if not geometry.enabled or not geometry.occluders:
        return np.ones(source_positions.shape[0], dtype=bool)

    positions = np.asarray(source_positions, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)

    obs_dir = np.asarray(observer_direction, dtype=float)
    obs_norm = float(np.linalg.norm(obs_dir))
    if obs_norm < 1e-15:
        return np.ones(positions.shape[0], dtype=bool)
    obs_dir = obs_dir / obs_norm

    # Select the occluder whose axis is most aligned with the observer direction.
    best_occluder = None
    best_alignment = -1.0
    for occluder in geometry.occluders:
        axis = np.asarray(occluder.axis, dtype=float)
        alignment = abs(float(np.dot(axis, obs_dir)))
        if alignment > best_alignment:
            best_alignment = alignment
            best_occluder = occluder

    if best_occluder is None:
        return np.ones(positions.shape[0], dtype=bool)

    dist_sq = _occluder_transverse_distance_sq(positions, best_occluder)
    axial = _occluder_axial_position(positions, best_occluder)
    half_length = best_occluder.length_mm * 0.5
    radius_sq = best_occluder.radius_mm * best_occluder.radius_mm
    visible = (dist_sq < radius_sq) & (np.abs(axial) <= half_length)
    return visible


__all__ = [
    "compute_visibility_mask",
    "compute_directional_visibility_mask",
]
