"""Kinematic reconstruction for retarded source histories.

The trajectory field named ``bdot`` is historically calculated from the
velocity change over the preceding step.  It is therefore an interval average,
not the instantaneous endpoint value required by a Lienard--Wiechert source
worldline.  The helpers here derive the endpoint value from accepted velocity
samples and their actual, possibly unequal, coordinate times.

Only accepted samples supplied by the caller are inspected.  An interior knot
uses one accepted sample on either side; the two history boundaries use the
corresponding one-sided quadratic.  This is appropriate for a retarded event
whose reconstruction window is already in the observer's accepted past.
"""

from __future__ import annotations

from typing import Sequence, cast

import numpy as np

from .constants import C_MMNS


def coordinate_beta_prime_from_four_kinematics(
    four_velocity_mm_ns: Sequence[float] | np.ndarray,
    four_acceleration_mm_ns2: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convert ``du/dtau`` into instantaneous ``d beta / d(c t)``.

    For ``u = gamma c (1, beta)``, the spatial coordinate acceleration obeys

    ``d beta/d(ct) = (a_spatial - beta*a_time) / (gamma**2*c**2)``.

    This conversion changes coordinates only. It does not add a missing force
    sector to the supplied four-acceleration.
    """

    velocity = np.asarray(four_velocity_mm_ns, dtype=np.float64)
    acceleration = np.asarray(four_acceleration_mm_ns2, dtype=np.float64)
    if velocity.shape != (4,) or acceleration.shape != (4,):
        raise ValueError("four-velocity and four-acceleration must have shape (4,)")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(acceleration)):
        raise ValueError("four-kinematics must contain only finite values")
    gamma = float(velocity[0] / C_MMNS)
    if not np.isfinite(gamma) or gamma < 1.0:
        raise ValueError("four-velocity must have a finite gamma of at least one")
    beta = velocity[1:] / velocity[0]
    beta_squared = float(beta @ beta)
    if beta_squared >= 1.0:
        raise ValueError("four-velocity must represent a subluminal beta")
    return cast(
        np.ndarray,
        (acceleration[1:] - beta * acceleration[0]) / (gamma * gamma * C_MMNS * C_MMNS),
    )


def _three_point_first_derivative_weights(
    coordinate_mm: np.ndarray,
    *,
    center_index: int,
) -> np.ndarray:
    """Return scaled Lagrange weights for one first derivative."""

    center = int(center_index)
    offsets = np.asarray(coordinate_mm, dtype=np.float64) - float(coordinate_mm[center])
    scale = float(np.max(np.abs(offsets)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("source coordinate samples must span a finite interval")
    normalized = offsets / scale
    system = normalized[np.newaxis, :] ** np.arange(3, dtype=np.float64)[:, None]
    right_hand_side = np.asarray((0.0, 1.0 / scale, 0.0), dtype=np.float64)
    return cast(np.ndarray, np.linalg.solve(system, right_hand_side))


def reconstruct_instantaneous_beta_prime_per_mm(
    time_ns: Sequence[float] | np.ndarray,
    beta: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return instantaneous ``d beta / d(c t)`` at accepted source knots.

    Three or more samples use the derivative of the local quadratic through
    the knot and its nearest accepted neighbours.  Two samples reduce to the
    common secant at both endpoints.  A single inertial seed has zero resolved
    acceleration because no derivative can yet be inferred.
    """

    times = np.asarray(time_ns, dtype=np.float64)
    velocities = np.asarray(beta, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("source times must be one-dimensional")
    if velocities.shape != (times.size, 3):
        raise ValueError("source beta must have shape (samples, 3)")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(velocities)):
        raise ValueError("source times and beta must contain only finite values")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("source times must increase strictly")
    if times.size and np.any(np.sum(velocities * velocities, axis=1) >= 1.0):
        raise ValueError("source beta magnitude must remain below one")

    sample_count = int(times.size)
    result = np.zeros((sample_count, 3), dtype=np.float64)
    if sample_count < 2:
        return result

    coordinate_mm = C_MMNS * times
    if sample_count == 2:
        secant = (velocities[1] - velocities[0]) / (coordinate_mm[1] - coordinate_mm[0])
        result[:] = secant
        return result

    for knot in range(sample_count):
        if knot == 0:
            indices = slice(0, 3)
            center = 0
        elif knot == sample_count - 1:
            indices = slice(sample_count - 3, sample_count)
            center = 2
        else:
            indices = slice(knot - 1, knot + 2)
            center = 1
        selected_coordinate = coordinate_mm[indices]
        weights = _three_point_first_derivative_weights(
            selected_coordinate,
            center_index=center,
        )
        result[knot] = weights @ velocities[indices]
    return result


__all__ = [
    "coordinate_beta_prime_from_four_kinematics",
    "reconstruct_instantaneous_beta_prime_per_mm",
]
