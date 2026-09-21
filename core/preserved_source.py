"""Preserve accepted increments and evaluate D=kS without absolute-value loss.

The RHS and external/self-field solvers remain float64. Decimal arithmetic is
limited to accepted-state accumulation and the shared constitutive algebra.
It cannot recover errors already present in those solvers or RK stage rates.
"""

from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np

from . import momentum_center as model
from .constants import C_MMNS


def decimal_array(value):
    return np.array(
        [Decimal.from_float(float(x)) for x in np.ravel(value)], dtype=object
    ).reshape(np.shape(value))


def split_array(value):
    high = np.asarray(value, dtype=float)
    low = np.asarray(value - decimal_array(high), dtype=float)
    if not np.isfinite(high).all() or not np.isfinite(low).all():
        raise ValueError("Finite preserved source components required")
    return high, low


def rounded_state(reference, high, low):
    with localcontext() as context:
        context.prec = 80
        return np.asarray(
            decimal_array(reference) + decimal_array(high) + decimal_array(low),
            dtype=float,
        )


def rk_increment(high, low, rates, width):
    """Accumulate RK4 increments before adding them to the large reference."""
    with localcontext() as context:
        context.prec = 80
        increment = sum(
            weight * decimal_array(rate) for weight, rate in zip((1, 2, 2, 1), rates)
        )
        increment *= Decimal.from_float(float(width)) / 6
        return split_array(decimal_array(high) + decimal_array(low) + increment)


def _solve(matrix, right):
    a, b = matrix.copy(), right.copy()
    for k in range(len(b)):
        pivot = max(range(k, len(b)), key=lambda j: abs(a[j, k]))
        if not a[pivot, k]:
            raise np.linalg.LinAlgError("Singular preserved velocity solve")
        a[[k, pivot]], b[[k, pivot]] = a[[pivot, k]], b[[pivot, k]]
        for j in range(k + 1, len(b)):
            factor = a[j, k] / a[k, k]
            a[j, k:] -= factor * a[k, k:]
            b[j] -= factor * b[k]
    result = np.full(len(b), Decimal(0), dtype=object)
    for k in reversed(range(len(b))):
        result[k] = (b[k] - a[k, k + 1 :] @ result[k + 1 :]) / a[k, k]
    return result


def dipole_parts(
    reference, high, low, particle, potential, field, gradient, dipole_reference
):
    """Same full-spin normalization, including the supplied solved self-field.

    Arguments A,F,partial F use length-time units. State and returned D use native
    units. Both spin and normalization variations are retained. This is NOT a
    high-precision solve of the implicit self-field root.
    """
    with localcontext() as context:
        context.prec = 80
        state = decimal_array(reference) + decimal_array(high) + decimal_array(low)
        speed = Decimal.from_float(C_MMNS)
        state[4:] /= speed
        q = Decimal.from_float(particle.charge)
        p = state[4:8] - q * decimal_array(potential)
        spin = model.unpack(state[8:])
        metric = decimal_array(model.METRIC)

        def dot(a, b):
            return (metric * a) @ b

        def contraction(a, b):
            return sum((a * metric[:, None] * metric * b).flat)

        parameters = SimpleNamespace(
            charge=q,
            bare_mass=Decimal.from_float(particle.bare_mass),
            coupling=Decimal.from_float(particle.coupling),
        )
        direction, _, _, coefficient, _, _ = model._velocity_system(
            p,
            spin,
            decimal_array(field),
            decimal_array(gradient),
            parameters,
            dot(p, p),
            contraction=contraction,
            solve=_solve,
            metric=metric,
            identity=decimal_array(np.eye(4)),
            half=Decimal("0.5"),
        )
        norm = dot(direction, direction)
        if norm <= 0 or direction[0] <= 0 or dot(p, p) <= 0 or p[0] <= 0:
            raise model.VelocityDomainError("No timelike preserved source state")
        coupling = coefficient / norm.sqrt()
        dipole = speed * coupling * spin
        return split_array(dipole - decimal_array(dipole_reference))
