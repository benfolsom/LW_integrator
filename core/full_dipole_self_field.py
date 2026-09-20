"""Radiative (half retarded minus advanced) full-current self field, c=1.

The angular Green-function identity removes the source singularity before
differentiation. Inputs are local proper-time Taylor coefficients, not future
source samples. This specifies the time-antisymmetric finite part only; it is
not a finite-size model or a derivation of the conservative counterterms.
See docs/full_dipole_self_reaction.md for the prescription and its limits.
"""

from functools import lru_cache
from typing import Any

import numpy as np

from .momentum_center import METRIC, contract, dot


@lru_cache(maxsize=8)
def _sphere(order: int) -> tuple[np.ndarray, np.ndarray]:
    if type(order) is not int or order < 6 or order > 64:
        raise ValueError("Angular order must be an integer between 6 and 64")
    z, weights = np.polynomial.legendre.leggauss(order)
    phi = np.arange(2 * order) * np.pi / order
    directions = np.array(
        [
            [np.sqrt(1 - t * t) * np.cos(p), np.sqrt(1 - t * t) * np.sin(p), t]
            for t in z
            for p in phi
        ]
    )
    return np.c_[np.ones(len(directions)), -directions], np.repeat(
        weights / (4 * order), 2 * order
    )


def _divide(series: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(series)
    for k in range(len(series)):
        result[k] = (
            series[k]
            - sum(
                denominator[j].reshape((-1,) + (1,) * (series.ndim - 2)) * result[k - j]
                for j in range(1, k + 1)
            )
        ) / denominator[0].reshape((-1,) + (1,) * (series.ndim - 2))
    return result


def _differentiate(series: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    derivative = (
        np.arange(1, len(series)).reshape((-1,) + (1,) * (series.ndim - 1)) * series[1:]
    )
    return _divide(derivative, denominator[: len(derivative)])


def regular_self_field(
    velocity_coefficients: np.ndarray,
    dipole_coefficients: np.ndarray,
    *,
    charge: float,
    angular_order: int = 8,
) -> dict[str, Any]:
    """Return F and partial F separately for the charge and full dipole.

    Coefficient n is the n-th proper-time derivative divided by n!, through
    order four. Velocity is dz/dtau and dipole is D (not D/u0). All four
    derivatives of velocity are needed for the dipole field gradient.
    Angular integration is performed in the instantaneous velocity rest frame
    and transformed back, avoiding narrow lab-frame relativistic beams.
    """
    u, d = map(
        lambda x: np.asarray(x, dtype=float),
        (velocity_coefficients, dipole_coefficients),
    )
    if (
        u.shape != (5, 4)
        or d.shape != (5, 4, 4)
        or not np.isfinite(u).all()
        or not np.isfinite(d).all()
        or not np.isfinite(charge)
    ):
        raise ValueError(
            "Finite order-four proper-time velocity and dipole coefficients required"
        )
    if u[0, 0] <= 0 or abs(dot(u[0], u[0]) - 1) > 1e-11 * max(1.0, np.dot(u[0], u[0])):
        raise ValueError("Future normalized velocity required")
    if not np.allclose(d, -d.swapaxes(1, 2), rtol=0, atol=1e-14):
        raise ValueError("Full antisymmetric dipole coefficients required")
    boost = np.eye(4)
    boost[0, 0] = u[0, 0]
    boost[0, 1:] = boost[1:, 0] = -u[0, 1:]
    boost[1:, 1:] += np.outer(u[0, 1:], u[0, 1:]) / (u[0, 0] + 1)
    inverse = METRIC[:, None] * boost.T * METRIC[None, :]
    ur = u @ boost.T
    dr = np.einsum("ai,kij,bj->kab", boost, d, boost)
    ell, weights = _sphere(angular_order)  # covariant null direction
    kappa = ur @ ell.T
    if np.min(kappa[0]) <= 0:
        raise ValueError("Positive null-plane source Jacobian required")
    qseries = _divide(np.broadcast_to(charge * ur[:, None, :], (5, len(ell), 4)), kappa)
    dseries = _divide(np.einsum("kij,nj->kni", dr, ell), kappa)
    qc, dc = [qseries], [dseries]
    for _ in range(3):
        qc.append(_differentiate(qc[-1], kappa))
    for _ in range(4):
        dc.append(_differentiate(dc[-1], kappa))
    q0, q1, q2, q3 = [entry[0] for entry in qc]
    d0, d1, d2, d3, d4 = [entry[0] for entry in dc]
    upper = ell * METRIC[None, :]

    def product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.einsum("ni,i,ni->n", a, METRIC, b)

    def average(value: np.ndarray) -> np.ndarray:
        return inverse @ np.einsum("n,ni->i", weights, value)

    # Integrate the local force by parts at fixed null direction. These
    # source-history identities separate reversible bound momentum from
    # outward radiation, and require no fitted fourth derivative for B or J.
    bound_integrands = (
        -charge * q1 + upper * product(q0, q1)[:, None],
        -charge * d2
        - upper * (product(d0, q2) - product(q0, d2) - product(d1, q1))[:, None],
        -upper * (product(d0, d3) - product(d1, d2))[:, None],
    )
    bound_rate_integrands = (
        -charge * q2 + upper * (product(q1, q1) + product(q0, q2))[:, None],
        -charge * d3
        - upper * (product(d0, q3) - product(q0, d3) - 2 * product(q1, d2))[:, None],
        -upper * (product(d0, d4) - product(d2, d2))[:, None],
    )
    radiation_scalars = (product(q1, q1), 2 * product(q1, d2), product(d2, d2))
    balance = {}
    for key, b, bd, power in zip(
        ("q_squared", "qD", "D_squared"),
        bound_integrands,
        bound_rate_integrands,
        radiation_scalars,
    ):
        balance[key] = dict(
            bound_momentum=average(b),
            bound_momentum_rate=average(kappa[0, :, None] * bd),
            outward_radiation_rate=average(-upper * (kappa[0] * power)[:, None]),
        )
    # A_rad = -< L(q u/kappa) + ell_nu L^2(D^{mu nu}/kappa) >.
    qseries = _differentiate(qseries, kappa)
    dseries = _differentiate(_differentiate(dseries, kappa), kappa)
    fields, gradients = [], []
    for series in (qseries, dseries):
        first = _differentiate(series, kappa)
        second = _differentiate(first, kappa)
        da = -np.einsum("n,na,ni->ai", weights, ell, first[0])
        dda = -np.einsum("n,nk,na,ni->kai", weights, ell, ell, second[0])
        f = METRIC[:, None] * da - METRIC[None, :] * da.T
        df = METRIC[None, :, None] * dda - METRIC[None, None, :] * dda.swapaxes(1, 2)
        transformed = inverse @ f @ inverse.T
        derivative = np.einsum("ak,ib,jc,abc->kij", boost, inverse, inverse, df)
        # Preserve the tensor's exact antisymmetric representation after a
        # floating-point boost (not a projection of particle state).
        fields.append(0.5 * (transformed - transformed.T))
        gradients.append(0.5 * (derivative - derivative.swapaxes(1, 2)))
    return dict(
        charge_field=fields[0],
        dipole_field=fields[1],
        charge_gradient=gradients[0],
        dipole_gradient=gradients[1],
        field=fields[0] + fields[1],
        gradient=gradients[0] + gradients[1],
        prescription="time_antisymmetric_point_current",
        angular_order=angular_order,
        source_balance=balance,
    )


def reaction_sectors(
    response: dict[str, Any], velocity: np.ndarray, dipole: np.ndarray, *, charge: float
) -> dict[str, dict[str, np.ndarray]]:
    """Same-current proper force and torque: q^2, qD, and D^2, no spin truncation."""
    u, d = np.asarray(velocity), np.asarray(dipole)

    def magnetic(f: np.ndarray, df: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mixed = f * METRIC[None, :]
        return (
            0.5 * METRIC * np.array([contract(entry, d) for entry in df]),
            mixed @ d + d @ mixed.T,
        )

    fq, fd = response["charge_field"], response["dipole_field"]
    cross_force, cross_torque = magnetic(fq, response["charge_gradient"])
    squared_force, squared_torque = magnetic(fd, response["dipole_gradient"])
    return dict(
        q_squared=dict(force=charge * fq @ (METRIC * u), torque=np.zeros((4, 4))),
        qD=dict(force=charge * fd @ (METRIC * u) + cross_force, torque=cross_torque),
        D_squared=dict(force=squared_force, torque=squared_torque),
    )
