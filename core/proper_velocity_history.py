"""Endpoint-constrained, timelike proper-velocity charge interpolation.

Positions accepted by the stepper remain authoritative. A quartic bubble
preserves endpoint u and du/d(ct), while Newton iteration fixes its integral.
Failure to reach the constraint or resolve quadrature raises a named error.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from decimal import Decimal, localcontext
from typing import cast

import numpy as np
from numba import njit  # type: ignore[import-untyped]

from .constants import C_MMNS
from .light_cone_history import LightConeSegment
from .light_cone_roots import geometry_anchor_residual, register_jitable

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(128)
_CHECK_NODES, _CHECK_WEIGHTS = np.polynomial.legendre.leggauss(256)
_CONSTRAINT_EPS = 64 * np.finfo(float).eps
_MAX_NEWTON_ITERATIONS = 24


class ProperVelocityConstraintError(ValueError):
    """The accepted segment cannot be represented at the requested accuracy."""


@register_jitable
def _u_at(coefficients: np.ndarray, tau: float) -> np.ndarray:
    value: np.ndarray = coefficients[4].copy()
    for order in range(3, -1, -1):
        value = value * tau + coefficients[order]
    return value


@register_jitable
def _null_beta(u: np.ndarray) -> np.ndarray:
    gamma = np.hypot(1.0, np.linalg.norm(u))
    deficit = (
        (1 + u[1] ** 2 + u[2] ** 2) / (gamma * (gamma + u[0]))
        if u[0] >= 0
        else 1 - u[0] / gamma
    )
    return np.array([deficit, u[1] / gamma, u[2] / gamma])


@register_jitable
def integrate_null_u(
    coefficients: np.ndarray,
    length: float,
    tau: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    start: float = 0.0,
) -> np.ndarray:
    span = tau - start
    if span == 0.0:
        return np.zeros(3)
    if np.all(coefficients[1:] == 0.0):
        return length * span * _null_beta(coefficients[0])
    value = np.zeros(3)
    for index in range(nodes.size):
        fraction = start + 0.5 * span * (nodes[index] + 1)
        value += weights[index] * _null_beta(_u_at(coefficients, fraction))
    return value * (0.5 * length * span)


@register_jitable
def sample_constrained_u(
    coefficients: np.ndarray,
    length: float,
    tau: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    """Null position and analytic velocity derivatives through third order."""
    u = np.zeros((4, 3))
    u[0] = _u_at(coefficients, tau)
    u[1] = (
        coefficients[1]
        + 2 * tau * coefficients[2]
        + 3 * tau**2 * coefficients[3]
        + 4 * tau**3 * coefficients[4]
    ) / length
    u[2] = (
        coefficients[2] + 3 * tau * coefficients[3] + 6 * tau**2 * coefficients[4]
    ) / length**2
    u[3] = (coefficients[3] + 4 * tau * coefficients[4]) / length**3
    gamma = np.zeros(4)
    gamma[0] = np.hypot(1.0, np.linalg.norm(u[0]))
    for order in range(1, 4):
        squared = 0.0
        product = 0.0
        for left in range(order + 1):
            squared += np.dot(u[left], u[order - left])
        for left in range(1, order):
            product += gamma[left] * gamma[order - left]
        gamma[order] = (squared - product) / (2 * gamma[0])
    reciprocal = np.zeros(4)
    reciprocal[0] = 1 / gamma[0]
    for order in range(1, 4):
        for left in range(1, order + 1):
            reciprocal[order] -= gamma[left] * reciprocal[order - left] / gamma[0]
    beta = np.zeros((4, 3))
    for order in range(4):
        for left in range(order + 1):
            beta[order] += u[left] * reciprocal[order - left]
    if u[0, 0] >= 0.0:
        numerator = np.zeros(4)
        denominator = np.zeros(4)
        for order in range(4):
            for left in range(order + 1):
                numerator[order] += np.dot(u[left, 1:], u[order - left, 1:])
                denominator[order] += gamma[left] * (
                    gamma[order - left] + u[order - left, 0]
                )
        numerator[0] += 1.0
        inverse = np.zeros(4)
        inverse[0] = 1 / denominator[0]
        for order in range(1, 4):
            for left in range(1, order + 1):
                inverse[order] -= (
                    denominator[left] * inverse[order - left] / denominator[0]
                )
        deficit = np.zeros(4)
        for order in range(4):
            for left in range(order + 1):
                deficit[order] += numerator[left] * inverse[order - left]
        beta[:, 0] = deficit
    else:
        beta[:, 0] *= -1
        beta[0, 0] += 1
    return (
        integrate_null_u(coefficients, length, tau, nodes, weights),
        beta[0],
        beta[1],
        2 * beta[2],
        6 * beta[3],
        u[0],
        1 / gamma[0] ** 2,
    )


@register_jitable
def proper_residual(
    coefficients: np.ndarray,
    length: float,
    tau: float,
    geometry: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    offset = tau
    start = 0.0
    anchored = geometry.size >= 12
    anchor_residual = geometry[10] if anchored else 0.0
    if tau > 0.5 and geometry.size >= 10:
        start = 1.0
        offset = tau - start
        if anchored:
            anchor_residual = geometry[11]
        geometry = geometry[5:10]
    value = integrate_null_u(coefficients, length, tau, nodes, weights, start)
    velocity = _null_beta(_u_at(coefficients, tau))
    parallel = geometry[1] - length * offset + value[0]
    transverse = geometry[2:4] - value[1:]
    transverse_squared = np.dot(transverse, transverse)
    radius = np.sqrt(parallel**2 + transverse_squared)
    difference = (
        transverse_squared / (radius + parallel) if parallel > 0 else radius - parallel
    )
    residual = geometry[0] - value[0] - difference
    if parallel <= 0:
        a = geometry[0] - value[0]
        b = geometry[4] - 2 * length * offset + value[0]
        denominator = a + parallel + radius
        if denominator > 0:
            residual = (a * b - transverse_squared) / denominator
    if anchored:
        residual = (
            anchor_residual
            if offset == 0.0
            else anchor_residual + (residual - geometry_anchor_residual(geometry))
        )
    kappa = (
        (difference + parallel * velocity[0] - np.dot(transverse, velocity[1:]))
        / radius
        if radius > 0
        else np.nan
    )
    return residual, kappa, radius, np.array([parallel, transverse[0], transverse[1]])


@register_jitable
def solve_constrained_u(
    coefficients: np.ndarray,
    length: float,
    geometry: np.ndarray,
    lower: float,
    upper: float,
    tolerance: float,
    iterations: int,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, tuple[float, float, float, np.ndarray]]:
    for endpoint in (lower, upper):
        result = proper_residual(
            coefficients, length, endpoint, geometry, nodes, weights
        )
        if result[0] == 0:
            return endpoint, result
    tau = 0.5 * (lower + upper)
    for _ in range(iterations):
        result = proper_residual(coefficients, length, tau, geometry, nodes, weights)
        residual, kappa = result[:2]
        if residual == 0 or (kappa > 0 and abs(residual) / kappa <= tolerance):
            return tau, result
        if residual > 0:
            lower = tau
        else:
            upper = tau
        if np.nextafter(lower, upper) >= upper:
            left = proper_residual(
                coefficients, length, lower, geometry, nodes, weights
            )
            right = proper_residual(
                coefficients, length, upper, geometry, nodes, weights
            )
            return (lower, left) if abs(left[0]) < abs(right[0]) else (upper, right)
        proposal = tau + residual / (length * kappa) if kappa > 0 else np.nan
        if not np.isfinite(proposal) or proposal <= lower or proposal >= upper:
            proposal = 0.5 * (lower + upper)
        tau = proposal
    raise ProperVelocityConstraintError(
        "proper-velocity light-cone root did not converge"
    )


solve_constrained_u_numba = njit(fastmath=False, cache=True)(solve_constrained_u)


@dataclass(frozen=True)
class ProperVelocitySegment(LightConeSegment):
    u_coefficients: np.ndarray = field(default_factory=lambda: np.zeros((5, 3)))
    constraint_residual_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    constraint_tolerance_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    newton_iterations: int = 0
    quadrature_nodes: np.ndarray = field(default_factory=lambda: _NODES)
    quadrature_weights: np.ndarray = field(default_factory=lambda: _WEIGHTS)

    def __post_init__(self) -> None:
        super().__post_init__()
        for value in (
            self.u_coefficients,
            self.constraint_residual_mm,
            self.constraint_tolerance_mm,
            self.quadrature_nodes,
            self.quadrature_weights,
        ):
            value.flags.writeable = False

    def null_sample(
        self, fraction: float
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
    ]:
        return sample_constrained_u(
            self.u_coefficients,
            C_MMNS * self.duration_ns,
            float(fraction),
            self.quadrature_nodes,
            self.quadrature_weights,
        )

    def sample(
        self, fraction: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        position, first, second, _, _, _, invariant = self.null_sample(fraction)
        lab_position = (
            self.origin_position_mm
            + np.array(
                [
                    C_MMNS * self.duration_ns * fraction - position[0],
                    position[1],
                    position[2],
                ]
            )
            @ self.frame
        )
        beta = np.array([1 - first[0], first[1], first[2]]) @ self.frame
        prime = np.array([-second[0], second[1], second[2]]) @ self.frame
        return lab_position, beta, prime, invariant

    def speed_deficit_bound(self) -> float:
        from math import comb

        # Bound the exact stored polynomial before binary64 rounding. Include
        # a conservative allowance for the four multiply/add Horner stages.
        with localcontext() as context:
            context.prec = 80
            d = Decimal.from_float
            controls = [
                [
                    sum(
                        (
                            d(float(self.u_coefficients[order, axis]))
                            * Decimal(comb(control, order))
                            / Decimal(comb(4, order))
                            for order in range(control + 1)
                        ),
                        Decimal(0),
                    )
                    for axis in range(3)
                ]
                for control in range(5)
            ]
            norm = max(sum((v * v for v in row), Decimal(0)).sqrt() for row in controls)
            coefficient_sums = [
                sum(
                    (abs(d(float(v))) for v in self.u_coefficients[:, axis]), Decimal(0)
                )
                for axis in range(3)
            ]
            evaluation_error = (
                Decimal(64)
                * d(float(np.finfo(float).eps))
                * sum((value * value for value in coefficient_sums), Decimal(0)).sqrt()
            )
            norm += evaluation_error
            gamma = (1 + norm * norm).sqrt()
            bound = float(1 / (gamma * (gamma + norm)))
        return float(np.nextafter(bound, 0.0))


def _constraint_data(
    coefficients: np.ndarray, length: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractions = 0.5 * (_NODES + 1)
    u = np.array([_u_at(coefficients, float(tau)) for tau in fractions])
    gamma = np.hypot(1.0, np.linalg.norm(u, axis=1))
    beta = np.array([_null_beta(value) for value in u])
    value = cast(np.ndarray, 0.5 * length * (_WEIGHTS @ beta))
    scale = cast(np.ndarray, 0.5 * length * (_WEIGHTS @ np.abs(beta)))
    jacobian = np.empty((3, 3))
    factor = 0.5 * length * _WEIGHTS * fractions**2 * (1 - fractions) ** 2 / gamma**3
    for row in range(3):
        for column in range(3):
            if row == column:
                numerator = 1 + np.sum(
                    u[:, [i for i in range(3) if i != row]] ** 2, axis=1
                )
            else:
                numerator = -u[:, row] * u[:, column]
            jacobian[row, column] = factor @ numerator
    jacobian[0] *= -1
    return value, jacobian, scale


def constrain_proper_velocity_segment(
    base: LightConeSegment, beta_primes: np.ndarray
) -> ProperVelocitySegment:
    if base.proper_velocity_knots is None or base.endpoint_null_mm is None:
        raise ProperVelocityConstraintError(
            "proper_velocity requires resolved mechanical u and accepted endpoint coordinates"
        )
    length = C_MMNS * base.duration_ns
    target = base.endpoint_null_mm
    with localcontext() as context:
        context.prec = 80
        d = Decimal.from_float
        w, trans1, trans2 = (d(float(value)) for value in target)
        if w * (2 * d(length) - w) - trans1**2 - trans2**2 <= 0:
            raise ProperVelocityConstraintError(
                "proper_velocity accepted endpoint chord is not strictly timelike"
            )
        axes = [
            [d(float(high)) + d(float(low)) for high, low in zip(row, lows)]
            for row, lows in zip(base.frame, base.frame_low)
        ]
        local_u = []
        local_du = []
        for proper, prime in zip(base.proper_velocity_knots, beta_primes):
            u = [d(float(value)) for value in proper]
            a = [d(float(value)) for value in prime]
            gamma = (1 + sum((value * value for value in u), Decimal(0))).sqrt()
            dot = sum((ui / gamma * ai for ui, ai in zip(u, a)), Decimal(0))
            du = [gamma * ai + ui * gamma**2 * dot for ui, ai in zip(u, a)]
            local_u.append(
                [
                    float(sum((axis[i] * u[i] for i in range(3)), Decimal(0)))
                    for axis in axes
                ]
            )
            local_du.append(
                [
                    float(sum((axis[i] * du[i] for i in range(3)), Decimal(0)))
                    for axis in axes
                ]
            )
    u0, u1 = np.array(local_u)
    du0, du1 = length * np.array(local_du)
    initial = np.zeros((5, 3))
    initial[0] = u0
    initial[1] = du0
    initial[2] = 3 * (u1 - u0) - 2 * du0 - du1
    initial[3] = 2 * (u0 - u1) + du0 + du1
    coefficients = initial.copy()
    bubble = np.zeros(3)
    iteration = 0
    for iteration in range(_MAX_NEWTON_ITERATIONS + 1):
        value, jacobian, absolute = _constraint_data(coefficients, length)
        scale = np.maximum.reduce(
            (np.abs(target), absolute, np.full(3, abs(length) * 1e-300))
        )
        tolerance = np.maximum(_CONSTRAINT_EPS * scale, np.nextafter(0.0, 1.0))
        residual = value - target
        error = float(np.max(np.abs(residual) / tolerance))
        if error <= 1:
            break
        if iteration == _MAX_NEWTON_ITERATIONS:
            raise ProperVelocityConstraintError(
                f"proper_velocity integral constraint did not converge after {iteration} Newton iterations; normalized residual={error:.6e}"
            )
        row_scale = np.maximum(scale, abs(length) * 1e-150)
        try:
            correction = np.linalg.solve(
                jacobian / row_scale[:, None], -residual / row_scale
            )
        except np.linalg.LinAlgError as exc:
            raise ProperVelocityConstraintError(
                "proper_velocity integral constraint has a singular Newton Jacobian"
            ) from exc
        if not np.all(np.isfinite(correction)):
            raise ProperVelocityConstraintError(
                "proper_velocity Newton correction is not finite"
            )
        for backtrack in range(16):
            trial_bubble = bubble + correction * 2.0 ** (-backtrack)
            trial = initial.copy()
            trial[2] += trial_bubble
            trial[3] -= 2 * trial_bubble
            trial[4] += trial_bubble
            candidate = integrate_null_u(trial, length, 1.0, _NODES, _WEIGHTS)
            if np.max(np.abs(candidate - target) / tolerance) < error:
                coefficients = trial
                bubble = trial_bubble
                break
        else:
            raise ProperVelocityConstraintError(
                f"proper_velocity Newton line search failed; normalized residual={error:.6e}"
            )
    accepted = integrate_null_u(coefficients, length, 1.0, _NODES, _WEIGHTS)
    check = integrate_null_u(coefficients, length, 1.0, _CHECK_NODES, _CHECK_WEIGHTS)
    if np.any(abs(check - accepted) > tolerance) or np.any(
        abs(accepted - target) > tolerance
    ):
        raise ProperVelocityConstraintError(
            "proper_velocity quadrature does not resolve the accepted endpoint constraint to roundoff"
        )
    values = {
        entry.name: getattr(base, entry.name) for entry in fields(LightConeSegment)
    }
    return ProperVelocitySegment(
        **values,
        u_coefficients=coefficients,
        constraint_residual_mm=accepted - target,
        constraint_tolerance_mm=tolerance,
        newton_iterations=iteration,
    )


def validate_source_history_representation(value: str) -> str:
    if value not in {"light_cone_quintic", "proper_velocity"}:
        raise ValueError(
            "source_history_representation must be light_cone_quintic or proper_velocity"
        )
    return value
