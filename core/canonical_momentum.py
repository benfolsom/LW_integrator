"""Canonical-momentum helpers for ordinary electromagnetic potentials.

The maintained equations use the native scaled-Gaussian convention

``P^mu = p^mu + (q / c) A^mu``

and

``dP^alpha/dtau = (q / c) u_beta partial^alpha A^beta``.

This module keeps that contraction explicit.  ``partial_a[lambda, nu]`` is
the covariant coordinate derivative ``partial_lambda A^nu`` for
``x=(ct,x,y,z)``.  The first index must therefore be raised before it is
contracted into the canonical equation.  No mass, Lorentz-factor, or
coordinate-time factor belongs in these helpers; the four-velocity already
contains ``gamma`` and the caller supplies a proper-time step.

Only the ordinary Maxwell potential belongs here.  The RFS response quantity
``B_mu = F*_(mu nu) a^nu`` is not an electromagnetic vector potential and
must not be added to canonical momentum.
"""

from __future__ import annotations

from typing import Sequence, cast

import numpy as np

from .constants import C_MMNS

VectorLike = Sequence[float] | np.ndarray
MatrixLike = Sequence[Sequence[float]] | np.ndarray

_MINKOWSKI_SIGNS = np.array((1.0, -1.0, -1.0, -1.0), dtype=float)


def _four_vector(value: VectorLike, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(np.ndarray, vector)


def _potential_gradient(value: MatrixLike) -> np.ndarray:
    gradient = np.asarray(value, dtype=float)
    if gradient.shape != (4, 4):
        raise ValueError("partial_a must have shape (4, 4)")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("partial_a must contain only finite values")
    return cast(np.ndarray, gradient)


def canonical_potential_momentum_native(
    four_potential: VectorLike,
    *,
    charge_native: float,
) -> np.ndarray:
    """Return the four-momentum offset ``(q/c) A^mu``.

    ``four_potential`` uses the solver's contravariant ``A^mu=(phi,A)``
    convention.  The result has native momentum units and is independent of
    the integration step size.
    """

    potential = _four_vector(four_potential, name="four_potential")
    charge = float(charge_native)
    if not np.isfinite(charge):
        raise ValueError("charge_native must be finite")
    return cast(np.ndarray, charge * potential / C_MMNS)


def canonical_four_force_from_potential_gradient_native(
    *,
    four_velocity_mm_ns: VectorLike,
    partial_a: MatrixLike,
    charge_native: float,
) -> np.ndarray:
    """Return ``(q/c) u_beta partial^alpha A^beta``.

    The result is ``dP^alpha/dtau`` in native force units.  The derivative
    array is ordered ``partial_a[lambda, nu] = partial_lambda A^nu`` per mm.
    """

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    gradient = _potential_gradient(partial_a)
    charge = float(charge_native)
    if not np.isfinite(charge):
        raise ValueError("charge_native must be finite")

    velocity_covariant = _MINKOWSKI_SIGNS * velocity
    derivative_contravariant = _MINKOWSKI_SIGNS[:, np.newaxis] * gradient
    return cast(
        np.ndarray,
        (charge / C_MMNS) * (derivative_contravariant @ velocity_covariant),
    )


def canonical_four_impulse_from_potential_gradient_native(
    *,
    four_velocity_mm_ns: VectorLike,
    partial_a: MatrixLike,
    charge_native: float,
    proper_time_step_ns: float,
) -> np.ndarray:
    """Return the first-order canonical impulse over ``proper_time_step_ns``."""

    step = float(proper_time_step_ns)
    if not np.isfinite(step):
        raise ValueError("proper_time_step_ns must be finite")
    return cast(
        np.ndarray,
        step
        * canonical_four_force_from_potential_gradient_native(
            four_velocity_mm_ns=four_velocity_mm_ns,
            partial_a=partial_a,
            charge_native=charge_native,
        ),
    )


def mechanical_four_momentum_native(
    canonical_four_momentum: VectorLike,
    four_potential: VectorLike,
    *,
    charge_native: float,
) -> np.ndarray:
    """Recover ``p^mu=P^mu-(q/c)A^mu`` at one observer event."""

    canonical = _four_vector(
        canonical_four_momentum,
        name="canonical_four_momentum",
    )
    return cast(
        np.ndarray,
        canonical
        - canonical_potential_momentum_native(
            four_potential,
            charge_native=charge_native,
        ),
    )


__all__ = [
    "canonical_four_force_from_potential_gradient_native",
    "canonical_four_impulse_from_potential_gradient_native",
    "canonical_potential_momentum_native",
    "mechanical_four_momentum_native",
]
