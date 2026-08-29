"""Canonical-momentum helpers for ordinary electromagnetic potentials.

The maintained equations use the native scaled-Gaussian convention

``P^mu = p^mu + (q / c) A^mu``

and

``dP^alpha/dtau = (q / c) u_beta partial^alpha A^beta``.

This module keeps that contraction explicit as a convention oracle and also
provides the equivalent mechanical Lorentz response.  The exact source path
advances mechanical momentum and reconstructs canonical momentum from the
accepted endpoint potential. ``partial_a[lambda, nu]`` is
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

from typing import Sequence, Union, cast

import numpy as np

from .constants import C_MMNS

VectorLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]
GradientLike = Union[Sequence[Sequence[Sequence[float]]], np.ndarray]

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


def _field_tensor(value: MatrixLike) -> np.ndarray:
    field = np.asarray(value, dtype=float)
    if field.shape != (4, 4):
        raise ValueError("field_tensor must have shape (4, 4)")
    if not np.all(np.isfinite(field)):
        raise ValueError("field_tensor must contain only finite values")
    if not np.allclose(field, -field.T, rtol=0.0, atol=1.0e-15):
        raise ValueError("field_tensor must be antisymmetric")
    return cast(np.ndarray, field)


def _field_gradient(value: GradientLike) -> np.ndarray:
    gradient = np.asarray(value, dtype=float)
    if gradient.shape != (4, 4, 4):
        raise ValueError("partial_f must have shape (4, 4, 4)")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("partial_f must contain only finite values")
    if not np.allclose(
        gradient,
        -np.swapaxes(gradient, 1, 2),
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("partial_f must be antisymmetric in its field indices")
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


def canonical_four_momentum_native(
    mechanical_four_momentum: VectorLike,
    four_potential: VectorLike,
    *,
    charge_native: float,
) -> np.ndarray:
    """Return ``P^mu = p^mu + (q/c) A^mu`` at one observer event."""

    mechanical = _four_vector(
        mechanical_four_momentum,
        name="mechanical_four_momentum",
    )
    return cast(
        np.ndarray,
        mechanical
        + canonical_potential_momentum_native(
            four_potential,
            charge_native=charge_native,
        ),
    )


def replace_canonical_potential_native(
    canonical_four_momentum: VectorLike,
    start_four_potential: VectorLike,
    end_four_potential: VectorLike,
    *,
    charge_native: float,
) -> np.ndarray:
    """Replace the ordinary potential offset without changing mechanical ``p``.

    This is the explicit accepted-endpoint bookkeeping operation

    ``P_end = P_temporary + (q/c) (A_end - A_start)``.

    It must be applied only after the mechanical endpoint is accepted.  The
    operation is not an electromagnetic impulse or an energy transfer.
    """

    canonical = _four_vector(
        canonical_four_momentum,
        name="canonical_four_momentum",
    )
    start = _four_vector(start_four_potential, name="start_four_potential")
    end = _four_vector(end_four_potential, name="end_four_potential")
    return cast(np.ndarray, canonical + float(charge_native) * (end - start) / C_MMNS)


def mechanical_lorentz_four_force_native(
    *,
    four_velocity_mm_ns: VectorLike,
    field_tensor: MatrixLike,
    charge_native: float,
) -> np.ndarray:
    """Return the gauge-invariant Lorentz force ``(q/c) F^(mu nu) u_nu``.

    ``field_tensor`` follows the repository's contravariant native-Gaussian
    convention.  This helper deliberately has no vector-potential argument:
    accepted canonical momentum is reconstructed from the endpoint potential
    separately.
    """

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    field = _field_tensor(field_tensor)
    charge = float(charge_native)
    if not np.isfinite(charge):
        raise ValueError("charge_native must be finite")
    return cast(
        np.ndarray,
        (charge / C_MMNS) * (field @ (_MINKOWSKI_SIGNS * velocity)),
    )


def mechanical_lorentz_four_impulse_native(
    *,
    four_velocity_mm_ns: VectorLike,
    field_tensor: MatrixLike,
    charge_native: float,
    proper_time_step_ns: float,
) -> np.ndarray:
    """Return the first-order mechanical Lorentz impulse over ``delta tau``."""

    step = float(proper_time_step_ns)
    if not np.isfinite(step):
        raise ValueError("proper_time_step_ns must be finite")
    return cast(
        np.ndarray,
        step
        * mechanical_lorentz_four_force_native(
            four_velocity_mm_ns=four_velocity_mm_ns,
            field_tensor=field_tensor,
            charge_native=charge_native,
        ),
    )


def mechanical_lorentz_four_force_derivative_native(
    *,
    four_velocity_mm_ns: VectorLike,
    four_acceleration_mm_ns2: VectorLike,
    field_tensor: MatrixLike,
    partial_f: GradientLike,
    charge_native: float,
) -> np.ndarray:
    """Return ``d/dtau [(q/c) F^(mu nu) u_nu]``.

    ``partial_f[lambda, mu, nu]`` is ``partial_lambda F^(mu nu)`` for
    ``x^lambda=(ct,x,y,z)`` in millimetres.  Consequently the convective
    derivative is ``u^lambda partial_lambda F`` with no additional factor of
    ``c``.  ``four_acceleration_mm_ns2`` is ``du^mu/dtau`` and may include all
    forces acting at the start event, not only the Lorentz term represented by
    ``field_tensor``.

    This helper supplies the analytical derivative needed by a second-order
    proper-time Taylor update.  It does not differentiate RFS moment response,
    radiation reaction, or another force sector.
    """

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    acceleration = _four_vector(
        four_acceleration_mm_ns2,
        name="four_acceleration_mm_ns2",
    )
    field = _field_tensor(field_tensor)
    gradient = _field_gradient(partial_f)
    charge = float(charge_native)
    if not np.isfinite(charge):
        raise ValueError("charge_native must be finite")

    velocity_covariant = _MINKOWSKI_SIGNS * velocity
    acceleration_covariant = _MINKOWSKI_SIGNS * acceleration
    convective_field_derivative = np.einsum(
        "l,lmn->mn",
        velocity,
        gradient,
    )
    return cast(
        np.ndarray,
        (charge / C_MMNS)
        * (
            convective_field_derivative @ velocity_covariant
            + field @ acceleration_covariant
        ),
    )


def mechanical_lorentz_second_order_four_impulse_native(
    *,
    four_velocity_mm_ns: VectorLike,
    four_acceleration_mm_ns2: VectorLike,
    field_tensor: MatrixLike,
    partial_f: GradientLike,
    charge_native: float,
    proper_time_step_ns: float,
) -> np.ndarray:
    """Return the second-order Taylor impulse ``h K + h^2 dK/dtau / 2``.

    The force is the ordinary Lorentz four-force ``K``.  Accuracy is second
    order only when the supplied acceleration is the complete start-event
    acceleration required by the force derivative.  The caller remains
    responsible for a matching second-order worldline update and for treating
    force sectors whose derivatives are unavailable.
    """

    step = float(proper_time_step_ns)
    if not np.isfinite(step):
        raise ValueError("proper_time_step_ns must be finite")
    force = mechanical_lorentz_four_force_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        field_tensor=field_tensor,
        charge_native=charge_native,
    )
    force_derivative = mechanical_lorentz_four_force_derivative_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        four_acceleration_mm_ns2=four_acceleration_mm_ns2,
        field_tensor=field_tensor,
        partial_f=partial_f,
        charge_native=charge_native,
    )
    return cast(
        np.ndarray,
        step * force + 0.5 * step * step * force_derivative,
    )


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
    "canonical_four_momentum_native",
    "canonical_four_force_from_potential_gradient_native",
    "canonical_four_impulse_from_potential_gradient_native",
    "canonical_potential_momentum_native",
    "mechanical_lorentz_four_force_native",
    "mechanical_lorentz_four_force_derivative_native",
    "mechanical_lorentz_four_impulse_native",
    "mechanical_lorentz_second_order_four_impulse_native",
    "mechanical_four_momentum_native",
    "replace_canonical_potential_native",
]
