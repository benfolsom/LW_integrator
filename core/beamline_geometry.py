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


__all__ = [
    "compute_visibility_mask",
]
