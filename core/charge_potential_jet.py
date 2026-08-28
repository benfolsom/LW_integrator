"""Second-order analytical observer jet of a retarded charge potential.

This is a validation oracle for the potential-first provider design.  One
already-bracketed retarded root selects a quintic worldline segment.  A
four-variable Taylor jet then differentiates the implicit light-cone equation
and the Lienard--Wiechert potential without displaced observer events or
additional root solves.

Coordinates are ``x=(ct,x,y,z)`` in millimetres.  The returned first and second
derivatives are ordinary derivatives, not factorial-scaled Taylor coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np

from .constants import C_MMNS


@dataclass(frozen=True)
class ChargePotentialJetResult:
    """Potential and its first two observer-coordinate derivatives."""

    four_potential: np.ndarray
    partial_a: np.ndarray
    partial2_a: np.ndarray
    retarded_time_ns: float
    retarded_coordinate_gradient: np.ndarray
    retarded_coordinate_hessian: np.ndarray
    light_cone_jet_residual: float


@dataclass(frozen=True)
class _Jet2:
    value: float
    gradient: np.ndarray
    hessian: np.ndarray

    @classmethod
    def constant(cls, value: float) -> "_Jet2":
        return cls(float(value), np.zeros(4), np.zeros((4, 4)))

    @classmethod
    def variable(cls, value: float, index: int) -> "_Jet2":
        gradient = np.zeros(4)
        gradient[index] = 1.0
        return cls(float(value), gradient, np.zeros((4, 4)))

    def __add__(self, other: object) -> "_Jet2":
        rhs = _as_jet(other)
        return _Jet2(
            self.value + rhs.value,
            self.gradient + rhs.gradient,
            self.hessian + rhs.hessian,
        )

    def __radd__(self, other: object) -> "_Jet2":
        return self + other

    def __neg__(self) -> "_Jet2":
        return _Jet2(-self.value, -self.gradient, -self.hessian)

    def __sub__(self, other: object) -> "_Jet2":
        return self + (-_as_jet(other))

    def __rsub__(self, other: object) -> "_Jet2":
        return _as_jet(other) - self

    def __mul__(self, other: object) -> "_Jet2":
        rhs = _as_jet(other)
        return _Jet2(
            self.value * rhs.value,
            rhs.value * self.gradient + self.value * rhs.gradient,
            rhs.value * self.hessian
            + self.value * rhs.hessian
            + np.outer(self.gradient, rhs.gradient)
            + np.outer(rhs.gradient, self.gradient),
        )

    def __rmul__(self, other: object) -> "_Jet2":
        return self * other

    def reciprocal(self) -> "_Jet2":
        if self.value == 0.0:
            raise ZeroDivisionError("cannot invert a zero Taylor jet")
        first = -1.0 / self.value**2
        second = 2.0 / self.value**3
        return _Jet2(
            1.0 / self.value,
            first * self.gradient,
            first * self.hessian + second * np.outer(self.gradient, self.gradient),
        )

    def __truediv__(self, other: object) -> "_Jet2":
        return self * _as_jet(other).reciprocal()

    def __rtruediv__(self, other: object) -> "_Jet2":
        return _as_jet(other) / self

    def sqrt(self) -> "_Jet2":
        if self.value <= 0.0:
            raise ValueError("Taylor-jet square root requires a positive value")
        root = np.sqrt(self.value)
        first = 0.5 / root
        second = -0.25 / (self.value * root)
        return _Jet2(
            root,
            first * self.gradient,
            first * self.hessian + second * np.outer(self.gradient, self.gradient),
        )


def _as_jet(value: object) -> _Jet2:
    if isinstance(value, _Jet2):
        return value
    return _Jet2.constant(float(cast(float, value)))


def _polynomial(coefficients: Sequence[float], argument: _Jet2) -> _Jet2:
    result = _Jet2.constant(0.0)
    for coefficient in reversed(tuple(float(value) for value in coefficients)):
        result = result * argument + coefficient
    return result


def _dot(left: Sequence[_Jet2], right: Sequence[_Jet2]) -> _Jet2:
    return sum((a * b for a, b in zip(left, right)), _Jet2.constant(0.0))


def _norm(vector: Sequence[_Jet2]) -> _Jet2:
    return _dot(vector, vector).sqrt()


def quintic_charge_potential_jet_native(
    *,
    observer_time_ns: float,
    observer_position_mm: Sequence[float],
    charge_native: float,
    segment_start_time_ns: float,
    segment_duration_ns: float,
    position_coefficients_mm: np.ndarray,
    retarded_time_ns: float,
    jet_newton_iterations: int = 4,
) -> ChargePotentialJetResult:
    """Differentiate one retarded quintic charge source at an observer event."""

    observer_time = float(observer_time_ns)
    observer_position = np.asarray(observer_position_mm, dtype=float)
    charge = float(charge_native)
    start_time = float(segment_start_time_ns)
    duration = float(segment_duration_ns)
    coefficients = np.asarray(position_coefficients_mm, dtype=float)
    root_time = float(retarded_time_ns)
    iterations = int(jet_newton_iterations)
    if not np.isfinite(observer_time):
        raise ValueError("observer_time_ns must be finite")
    if observer_position.shape != (3,) or not np.all(np.isfinite(observer_position)):
        raise ValueError("observer_position_mm must contain three finite values")
    if not np.isfinite(charge):
        raise ValueError("charge_native must be finite")
    if not np.isfinite(start_time) or not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("the segment start and positive duration must be finite")
    if coefficients.shape != (6, 3) or not np.all(np.isfinite(coefficients)):
        raise ValueError("position_coefficients_mm must have finite shape (6, 3)")
    if not np.isfinite(root_time):
        raise ValueError("retarded_time_ns must be finite")
    if root_time < start_time or root_time > start_time + duration:
        raise ValueError("retarded_time_ns must lie inside the selected segment")
    if iterations < 2:
        raise ValueError("jet_newton_iterations must be at least two")

    observer_coordinates = (
        _Jet2.variable(C_MMNS * observer_time, 0),
        *(
            _Jet2.variable(float(observer_position[index]), index + 1)
            for index in range(3)
        ),
    )
    source_coordinate = _Jet2.constant(C_MMNS * root_time)
    segment_start_coordinate = C_MMNS * start_time
    segment_duration_coordinate = C_MMNS * duration

    def source_state(coordinate: _Jet2) -> tuple[list[_Jet2], list[_Jet2]]:
        normalized_time = (
            coordinate - segment_start_coordinate
        ) / segment_duration_coordinate
        position = [
            _polynomial(coefficients[:, component], normalized_time)
            for component in range(3)
        ]
        beta = [
            _polynomial(
                tuple(
                    order * coefficients[order, component] / segment_duration_coordinate
                    for order in range(1, 6)
                ),
                normalized_time,
            )
            for component in range(3)
        ]
        return position, beta

    light_cone = _Jet2.constant(0.0)
    for _ in range(iterations):
        source_position, source_beta = source_state(source_coordinate)
        separation_vector = [
            observer_coordinates[index + 1] - source_position[index]
            for index in range(3)
        ]
        separation = _norm(separation_vector)
        direction = [component / separation for component in separation_vector]
        light_cone = observer_coordinates[0] - source_coordinate - separation
        derivative = -1.0 + _dot(direction, source_beta)
        # The safeguarded scalar solver owns the base retarded root.  Jet
        # Newton solves only the implicit derivative coefficients.  Projecting
        # the already-toleranced scalar residual to zero prevents an
        # ultra-relativistic 1/kappa amplification from toggling the base value
        # between adjacent floating-point roots on successive jet iterations.
        light_cone = _Jet2(0.0, light_cone.gradient, light_cone.hessian)
        source_coordinate = source_coordinate - light_cone / derivative

    source_position, source_beta = source_state(source_coordinate)
    separation_vector = [
        observer_coordinates[index + 1] - source_position[index] for index in range(3)
    ]
    separation = _norm(separation_vector)
    direction = [component / separation for component in separation_vector]
    kappa = 1.0 - _dot(direction, source_beta)
    if kappa.value <= 1.0e-14:
        raise ValueError(
            "retarded potential is singular because 1 - n.beta is too small"
        )
    scalar_potential = charge / (kappa * separation)
    potential = [scalar_potential] + [
        scalar_potential * component for component in source_beta
    ]
    light_cone = observer_coordinates[0] - source_coordinate - separation

    partial_a = np.stack([component.gradient for component in potential], axis=1)
    partial2_a = np.stack([component.hessian for component in potential], axis=2)
    return ChargePotentialJetResult(
        four_potential=np.asarray([component.value for component in potential]),
        partial_a=partial_a,
        partial2_a=partial2_a,
        retarded_time_ns=source_coordinate.value / C_MMNS,
        retarded_coordinate_gradient=source_coordinate.gradient,
        retarded_coordinate_hessian=source_coordinate.hessian,
        light_cone_jet_residual=max(
            abs(light_cone.value),
            float(np.max(np.abs(light_cone.gradient))),
            float(np.max(np.abs(light_cone.hessian))),
        ),
    )


__all__ = ["ChargePotentialJetResult", "quintic_charge_potential_jet_native"]
