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
    mechanical_gamma = np.hypot(1.0, np.linalg.norm(velocity[1:] / C_MMNS))
    if beta_squared >= 1.0 and not np.isclose(
        gamma, mechanical_gamma, rtol=8 * np.finfo(float).eps, atol=0.0
    ):
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
    *,
    inertial_prefix_stop: int = 0,
    inertial_proper_velocity: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Return instantaneous ``d beta / d(c t)`` at accepted source knots.

    Three or more samples use the derivative of the local quadratic through
    the knot and its nearest accepted neighbours.  Two samples reduce to the
    common secant at both endpoints.  A single inertial seed has zero resolved
    acceleration because no derivative can yet be inferred. An explicitly
    analytic inertial prefix may supply its proper velocity when beta rounds
    to one; accepted samples after that prefix retain the original checks.
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
    prefix = int(inertial_prefix_stop)
    if prefix < 0 or prefix > times.size:
        raise ValueError("inertial prefix must lie within the supplied source samples")
    if prefix:
        if inertial_proper_velocity is None:
            raise ValueError(
                "an inertial prefix requires its mechanical proper velocity"
            )
        proper = np.asarray(inertial_proper_velocity, dtype=float)
        if proper.shape != (3,) or not np.all(np.isfinite(proper)):
            raise ValueError(
                "inertial proper velocity must contain three finite values"
            )
        gamma = np.sqrt(1.0 + float(proper @ proper))
        if not np.allclose(
            velocities[:prefix], proper / gamma, rtol=0.0, atol=4 * np.finfo(float).eps
        ):
            raise ValueError("inertial prefix beta must agree with its proper velocity")
    if times.size and np.any(
        np.sum(velocities[prefix:] * velocities[prefix:], axis=1) >= 1.0
    ):
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
        # Centering keeps constant beta at exactly zero despite weight roundoff.
        result[knot] = weights @ (velocities[indices] - velocities[knot])
    return result


__all__ = [
    "coordinate_beta_prime_from_four_kinematics",
    "reconstruct_instantaneous_beta_prime_per_mm",
]


def reconstruct_resolved_beta_prime_per_mm(
    time_ns: np.ndarray, beta: np.ndarray, proper_velocity: np.ndarray
) -> np.ndarray:
    """Differentiate precise mechanical samples without rounding beta to one."""
    from decimal import Decimal, localcontext

    times = np.asarray(time_ns, dtype=float)
    proper = np.asarray(proper_velocity, dtype=float)
    velocities = np.asarray(beta, dtype=float)
    if proper.shape != velocities.shape or velocities.shape != (times.size, 3):
        raise ValueError("resolved source kinematics must match the sample count")
    if not np.all(np.isfinite(proper)) or not np.all(np.isfinite(times)):
        raise ValueError("resolved source kinematics must be finite")
    if times.size > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("source times must increase strictly")
    result = np.zeros_like(velocities)
    if len(times) < 2 or np.all(proper == proper[0]):
        return result
    with localcontext() as context:
        context.prec = 80
        samples = []
        for vector in proper:
            values = [Decimal.from_float(float(v)) for v in vector]
            gamma = (1 + sum((v * v for v in values), Decimal(0))).sqrt()
            samples.append([v / gamma for v in values])
        coordinate = C_MMNS * times
        for knot in range(len(times)):
            if len(times) == 2:
                indices = [0, 1]
                weights = np.array([-1.0, 1.0]) / (coordinate[1] - coordinate[0])
            else:
                left = max(0, min(knot - 1, len(times) - 3))
                indices = list(range(left, left + 3))
                weights = _three_point_first_derivative_weights(
                    coordinate[indices], center_index=knot - left
                )
            for component in range(3):
                result[knot, component] = float(
                    sum(
                        (
                            Decimal.from_float(float(weight))
                            * (samples[index][component] - samples[knot][component])
                            for index, weight in zip(indices, weights)
                        ),
                        Decimal(0),
                    )
                )
    return result
