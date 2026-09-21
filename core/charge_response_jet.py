"""One-root analytical jet of the stable Lienard--Wiechert charge response.

Unlike :mod:`core.charge_potential_jet`, this validation oracle does not form
all first and second derivatives of ``A``.  It evaluates the stable six field
combinations and differentiates them once.  The returned tensors exist only to
compare with the maintained API; a production contraction kernel can consume
the six values and their directional derivatives directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import C_MMNS
from .rfs import electromagnetic_field_tensor_native


@dataclass(frozen=True)
class ChargeResponseJetResult:
    """Potential value and validation-form charge response derivatives."""

    four_potential: np.ndarray
    partial_a: np.ndarray
    antisymmetric_response: np.ndarray
    partial_antisymmetric_response: np.ndarray
    field_tensor: np.ndarray
    partial_f: np.ndarray
    kappa: float
    light_cone_residual_mm: float


@dataclass(frozen=True)
class _Jet1:
    value: float
    gradient: np.ndarray

    @classmethod
    def constant(cls, value: float) -> "_Jet1":
        return cls(float(value), np.zeros(4))

    @classmethod
    def variable(cls, value: float, index: int) -> "_Jet1":
        gradient = np.zeros(4)
        gradient[index] = 1.0
        return cls(float(value), gradient)

    def __add__(self, other: object) -> "_Jet1":
        rhs = _as_jet(other)
        return _Jet1(self.value + rhs.value, self.gradient + rhs.gradient)

    def __radd__(self, other: object) -> "_Jet1":
        return self + other

    def __neg__(self) -> "_Jet1":
        return _Jet1(-self.value, -self.gradient)

    def __sub__(self, other: object) -> "_Jet1":
        return self + (-_as_jet(other))

    def __rsub__(self, other: object) -> "_Jet1":
        return _as_jet(other) - self

    def __mul__(self, other: object) -> "_Jet1":
        rhs = _as_jet(other)
        return _Jet1(
            self.value * rhs.value,
            rhs.value * self.gradient + self.value * rhs.gradient,
        )

    def __rmul__(self, other: object) -> "_Jet1":
        return self * other

    def reciprocal(self) -> "_Jet1":
        inverse = 1.0 / self.value
        return _Jet1(inverse, -(inverse * inverse) * self.gradient)

    def __truediv__(self, other: object) -> "_Jet1":
        return self * _as_jet(other).reciprocal()

    def __rtruediv__(self, other: object) -> "_Jet1":
        return _as_jet(other) / self

    def sqrt(self) -> "_Jet1":
        root = np.sqrt(self.value)
        return _Jet1(root, 0.5 * self.gradient / root)


def _as_jet(value: object) -> _Jet1:
    if isinstance(value, _Jet1):
        return value
    return _Jet1.constant(float(value))


def _polynomial(coefficients: Sequence[float], argument: _Jet1) -> _Jet1:
    result = _Jet1.constant(0.0)
    for coefficient in reversed(tuple(float(value) for value in coefficients)):
        result = result * argument + coefficient
    return result


def _dot(left: Sequence[_Jet1], right: Sequence[_Jet1]) -> _Jet1:
    return sum((a * b for a, b in zip(left, right)), _Jet1.constant(0.0))


def _cross(left: Sequence[_Jet1], right: Sequence[_Jet1]) -> list[_Jet1]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _norm(vector: Sequence[_Jet1]) -> _Jet1:
    return _dot(vector, vector).sqrt()


def quintic_charge_response_jet_native(
    *,
    observer_time_ns: float,
    observer_position_mm: Sequence[float],
    charge_native: float,
    segment_start_time_ns: float,
    segment_duration_ns: float,
    position_coefficients_mm: np.ndarray,
    retarded_time_ns: float,
) -> ChargeResponseJetResult:
    """Return the stable charge response and its analytical spacetime derivative."""

    observer_position = np.asarray(observer_position_mm, dtype=float)
    coefficients = np.asarray(position_coefficients_mm, dtype=float)
    duration_coordinate = C_MMNS * float(segment_duration_ns)
    root_coordinate = C_MMNS * float(retarded_time_ns)
    normalized_root = (
        root_coordinate - C_MMNS * float(segment_start_time_ns)
    ) / duration_coordinate
    source_position_value = np.asarray(
        [
            sum(
                coefficients[order, component] * normalized_root**order
                for order in range(6)
            )
            for component in range(3)
        ]
    )
    source_beta_value = np.asarray(
        [
            sum(
                order
                * coefficients[order, component]
                * normalized_root ** (order - 1)
                / duration_coordinate
                for order in range(1, 6)
            )
            for component in range(3)
        ]
    )
    source_beta_prime_value = np.asarray(
        [
            sum(
                order
                * (order - 1)
                * coefficients[order, component]
                * normalized_root ** (order - 2)
                / duration_coordinate**2
                for order in range(2, 6)
            )
            for component in range(3)
        ]
    )
    separation_value = observer_position - source_position_value
    radius_value = float(np.linalg.norm(separation_value))
    if radius_value <= 0.0:
        raise ValueError("the observer cannot coincide with a point-charge source")
    direction_value = separation_value / radius_value
    kappa_value = 1.0 - float(direction_value @ source_beta_value)
    if kappa_value <= 1.0e-14:
        raise ValueError(
            "retarded response is singular because 1 - n.beta is too small"
        )

    root_gradient = np.concatenate(([1.0], -direction_value)) / kappa_value
    source_coordinate = _Jet1(root_coordinate, root_gradient)
    observer_coordinates = (
        _Jet1.variable(C_MMNS * float(observer_time_ns), 0),
        *(
            _Jet1.variable(float(observer_position[index]), index + 1)
            for index in range(3)
        ),
    )
    normalized_time = (
        source_coordinate - C_MMNS * float(segment_start_time_ns)
    ) / duration_coordinate
    source_position = [
        _polynomial(coefficients[:, component], normalized_time)
        for component in range(3)
    ]
    source_beta = [
        _polynomial(
            tuple(
                order * coefficients[order, component] / duration_coordinate
                for order in range(1, 6)
            ),
            normalized_time,
        )
        for component in range(3)
    ]
    source_beta_prime = [
        _polynomial(
            tuple(
                order
                * (order - 1)
                * coefficients[order, component]
                / duration_coordinate**2
                for order in range(2, 6)
            ),
            normalized_time,
        )
        for component in range(3)
    ]
    separation_vector = [
        observer_coordinates[index + 1] - source_position[index] for index in range(3)
    ]
    radius = _norm(separation_vector)
    direction = [component / radius for component in separation_vector]
    kappa = 1.0 - _dot(direction, source_beta)
    beta_squared = _dot(source_beta, source_beta)
    direction_minus_beta = [direction[index] - source_beta[index] for index in range(3)]
    velocity_prefactor = (1.0 - beta_squared) / (
        kappa * kappa * kappa * radius * radius
    )
    radiation_cross = _cross(
        direction,
        _cross(direction_minus_beta, source_beta_prime),
    )
    radiation_prefactor = 1.0 / (kappa * kappa * kappa * radius)
    electric = [
        float(charge_native)
        * (
            velocity_prefactor * direction_minus_beta[index]
            + radiation_prefactor * radiation_cross[index]
        )
        for index in range(3)
    ]
    magnetic = _cross(direction, electric)
    # Use the maintained, well-conditioned scalar ordering for the base six
    # response values.  The Taylor algebra supplies only their derivatives.
    # This matters near kappa -> 0, where an algebraically equivalent generic
    # product ordering can lose several additional digits.
    beta_squared_value = float(source_beta_value @ source_beta_value)
    velocity_value = (
        (1.0 - beta_squared_value)
        * (direction_value - source_beta_value)
        / (kappa_value**3 * radius_value**2)
    )
    radiation_value = np.cross(
        direction_value,
        np.cross(direction_value - source_beta_value, source_beta_prime_value),
    ) / (kappa_value**3 * radius_value)
    electric_value = float(charge_native) * (velocity_value + radiation_value)
    magnetic_value = np.cross(direction_value, electric_value)
    antisymmetric_response = np.asarray(
        (
            -electric_value[0],
            -electric_value[1],
            -electric_value[2],
            -magnetic_value[2],
            magnetic_value[1],
            -magnetic_value[0],
        )
    )
    partial_antisymmetric_response = np.empty((4, 6))
    for derivative in range(4):
        ex, ey, ez = (component.gradient[derivative] for component in electric)
        bx, by, bz = (component.gradient[derivative] for component in magnetic)
        partial_antisymmetric_response[derivative] = (
            -ex,
            -ey,
            -ez,
            -bz,
            by,
            -bx,
        )
    field_tensor = electromagnetic_field_tensor_native(electric_value, magnetic_value)

    partial_f = np.zeros((4, 4, 4))
    for derivative in range(4):
        ex, ey, ez = (component.gradient[derivative] for component in electric)
        bx, by, bz = (component.gradient[derivative] for component in magnetic)
        partial_f[derivative] = np.asarray(
            (
                (0.0, -ex, -ey, -ez),
                (ex, 0.0, -bz, by),
                (ey, bz, 0.0, -bx),
                (ez, -by, bx, 0.0),
            )
        )
    scalar_potential_jet = float(charge_native) / (kappa * radius)
    potential_jets = [
        scalar_potential_jet,
        *(scalar_potential_jet * component for component in source_beta),
    ]
    # Retain the maintained scalar ordering for A itself.  The jet values are
    # algebraically equivalent, while only their analytical derivatives are
    # consumed below.
    scalar_potential = float(charge_native) / (kappa_value * radius_value)
    four_potential = scalar_potential * np.concatenate(([1.0], source_beta_value))
    partial_a = np.asarray(
        [
            [potential_jets[component].gradient[derivative] for component in range(4)]
            for derivative in range(4)
        ]
    )
    light_cone_residual = (
        C_MMNS * (float(observer_time_ns) - float(retarded_time_ns)) - radius_value
    )
    return ChargeResponseJetResult(
        four_potential=four_potential,
        partial_a=partial_a,
        antisymmetric_response=antisymmetric_response,
        partial_antisymmetric_response=partial_antisymmetric_response,
        field_tensor=field_tensor,
        partial_f=partial_f,
        kappa=kappa.value,
        light_cone_residual_mm=light_cone_residual,
    )


__all__ = ["ChargeResponseJetResult", "quintic_charge_response_jet_native"]
