"""Smooth-segment directional jets of retarded charge and dipole potentials.

The linear-spin reduction-of-order bridge does not need complete third- and
fourth-derivative tensors of the Maxwell four-potential.  It needs only

``u^k partial_k partial_l partial_m A^n``,
``a^k partial_k partial_l partial_m A^n``, and
``u^k u^r partial_k partial_r partial_l partial_m A^n``.

This module evaluates those three contractions analytically inside one smooth
source-history segment.  The scalar retarded root is supplied by the existing
safeguarded light-cone solver; factorial-scaled Taylor arithmetic then
differentiates the implicit light cone and potential without displaced
observer events or additional root solves.

Coordinates are ``x=(ct,x,y,z)`` in millimetres.  The source worldline is the
maintained quintic segment.  A dipole source additionally uses the maintained
cubic rest-spin segment.  The functions require the retarded root to lie
strictly inside the segment: higher derivatives are not unique at the current
worldline or spin interpolation knots.

The implementation retains a dense internal Taylor coefficient table as a
transparent validation oracle, but returns only the derivative combinations
used by :mod:`core.potential_jet_rfs`.  A later compiled sparse kernel may omit
the internal coefficients proved unnecessary by this oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
from math import factorial
from typing import TYPE_CHECKING, Hashable, Sequence, cast

import numpy as np

from .constants import C_MMNS

if TYPE_CHECKING:
    from .retarded_fields import ObserverEvent, TrajectoryHistory

_DIMENSION = 4
_METRIC_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0), dtype=float)


def _levi_civita_upper() -> np.ndarray:
    tensor = np.zeros((4, 4, 4, 4), dtype=float)
    for permutation in permutations(range(4)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(4)
            for right in range(left + 1, 4)
        )
        # epsilon_0123=+1; raising all indices with (+---) changes the sign.
        tensor[permutation] = -1.0 if inversions % 2 == 0 else 1.0
    return tensor


_LEVI_CIVITA_UPPER = _levi_civita_upper()


@dataclass(frozen=True)
class PotentialDirectionalDerivatives:
    """Summable potential derivatives required by the local RFS bridge."""

    four_potential: np.ndarray
    partial_a: np.ndarray
    partial2_a: np.ndarray
    partial3_a_along_velocity: np.ndarray
    partial3_a_along_acceleration: np.ndarray
    partial4_a_along_velocity_twice: np.ndarray


@dataclass(frozen=True)
class PotentialDirectionalDerivativeJet:
    """One source-segment derivative result plus retarded-root diagnostics."""

    derivatives: PotentialDirectionalDerivatives
    retarded_time_ns: float
    light_cone_jet_residual: float
    segment_fraction: float

    @property
    def four_potential(self) -> np.ndarray:
        return self.derivatives.four_potential

    @property
    def partial_a(self) -> np.ndarray:
        return self.derivatives.partial_a

    @property
    def partial2_a(self) -> np.ndarray:
        return self.derivatives.partial2_a

    @property
    def partial3_a_along_velocity(self) -> np.ndarray:
        return self.derivatives.partial3_a_along_velocity

    @property
    def partial3_a_along_acceleration(self) -> np.ndarray:
        return self.derivatives.partial3_a_along_acceleration

    @property
    def partial4_a_along_velocity_twice(self) -> np.ndarray:
        return self.derivatives.partial4_a_along_velocity_twice


@dataclass(frozen=True)
class RetardedPotentialDirectionalJetProviderResult:
    """Summed smooth-segment derivatives or an explicit unavailability signal."""

    derivatives: PotentialDirectionalDerivatives | None
    available: bool
    unavailable_reason: str | None
    source_identities: tuple[Hashable, ...]
    valid_sources: np.ndarray
    retarded_time_ns: np.ndarray
    source_segment_index: np.ndarray
    source_segment_fraction: np.ndarray
    source_jet_residual: np.ndarray


@dataclass(frozen=True)
class _JetSpace:
    order: int
    multiindices: tuple[tuple[int, int, int, int], ...]
    index: dict[tuple[int, int, int, int], int]
    product_splits: tuple[tuple[tuple[int, int], ...], ...]


@lru_cache(maxsize=None)
def _jet_space(order: int) -> _JetSpace:
    selected_order = int(order)
    if selected_order < 0:
        raise ValueError("Taylor-jet order must be non-negative")
    multiindices: list[tuple[int, int, int, int]] = []
    for total_degree in range(selected_order + 1):
        for alpha in product(range(total_degree + 1), repeat=_DIMENSION):
            if sum(alpha) == total_degree:
                multiindices.append(cast(tuple[int, int, int, int], alpha))
    indices = tuple(multiindices)
    index = {alpha: position for position, alpha in enumerate(indices)}
    product_splits: list[tuple[tuple[int, int], ...]] = []
    for alpha in indices:
        splits: list[tuple[int, int]] = []
        for beta_value in product(*(range(value + 1) for value in alpha)):
            beta = cast(tuple[int, int, int, int], tuple(beta_value))
            gamma = cast(
                tuple[int, int, int, int],
                tuple(alpha[item] - beta[item] for item in range(_DIMENSION)),
            )
            splits.append((index[beta], index[gamma]))
        product_splits.append(tuple(splits))
    return _JetSpace(
        order=selected_order,
        multiindices=indices,
        index=index,
        product_splits=tuple(product_splits),
    )


@dataclass(frozen=True)
class _TaylorJet:
    space: _JetSpace
    coefficients: np.ndarray

    @classmethod
    def constant(cls, space: _JetSpace, value: float) -> "_TaylorJet":
        coefficients = np.zeros(len(space.multiindices), dtype=float)
        coefficients[0] = float(value)
        return cls(space, coefficients)

    @classmethod
    def variable(
        cls,
        space: _JetSpace,
        value: float,
        index: int,
    ) -> "_TaylorJet":
        result = cls.constant(space, value)
        coefficients = result.coefficients.copy()
        alpha = [0, 0, 0, 0]
        alpha[int(index)] = 1
        coefficients[space.index[cast(tuple[int, int, int, int], tuple(alpha))]] = 1.0
        return cls(space, coefficients)

    @property
    def value(self) -> float:
        return float(self.coefficients[0])

    def with_value(self, value: float) -> "_TaylorJet":
        coefficients = self.coefficients.copy()
        coefficients[0] = float(value)
        return _TaylorJet(self.space, coefficients)

    def derivative(self, *indices: int) -> float:
        if len(indices) > self.space.order:
            raise ValueError("requested derivative exceeds the Taylor-jet order")
        alpha = [0, 0, 0, 0]
        for index in indices:
            alpha[int(index)] += 1
        multiindex = cast(tuple[int, int, int, int], tuple(alpha))
        coefficient = self.coefficients[self.space.index[multiindex]]
        scale = 1
        for multiplicity in multiindex:
            scale *= factorial(multiplicity)
        return float(scale * coefficient)

    def _coerce(self, other: object) -> "_TaylorJet":
        if isinstance(other, _TaylorJet):
            if other.space is not self.space:
                raise ValueError("Taylor jets must share one coefficient space")
            return other
        return _TaylorJet.constant(self.space, float(cast(float, other)))

    def __add__(self, other: object) -> "_TaylorJet":
        right = self._coerce(other)
        return _TaylorJet(self.space, self.coefficients + right.coefficients)

    def __radd__(self, other: object) -> "_TaylorJet":
        return self + other

    def __neg__(self) -> "_TaylorJet":
        return _TaylorJet(self.space, -self.coefficients)

    def __sub__(self, other: object) -> "_TaylorJet":
        return self + (-self._coerce(other))

    def __rsub__(self, other: object) -> "_TaylorJet":
        return self._coerce(other) - self

    def __mul__(self, other: object) -> "_TaylorJet":
        right = self._coerce(other)
        coefficients = np.empty(len(self.space.multiindices), dtype=float)
        for result_index, splits in enumerate(self.space.product_splits):
            total = 0.0
            for left_index, right_index in splits:
                total += self.coefficients[left_index] * right.coefficients[right_index]
            coefficients[result_index] = total
        return _TaylorJet(self.space, coefficients)

    def __rmul__(self, other: object) -> "_TaylorJet":
        return self * other

    def reciprocal(self) -> "_TaylorJet":
        if self.value == 0.0:
            raise ZeroDivisionError("cannot invert a zero Taylor jet")
        coefficients = np.zeros(len(self.space.multiindices), dtype=float)
        coefficients[0] = 1.0 / self.value
        for result_index in range(1, len(self.space.multiindices)):
            total = 0.0
            for left_index, right_index in self.space.product_splits[result_index]:
                if left_index == 0:
                    continue
                total += self.coefficients[left_index] * coefficients[right_index]
            coefficients[result_index] = -total / self.value
        return _TaylorJet(self.space, coefficients)

    def __truediv__(self, other: object) -> "_TaylorJet":
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other: object) -> "_TaylorJet":
        return self._coerce(other) / self

    def sqrt(self) -> "_TaylorJet":
        if self.value <= 0.0:
            raise ValueError("Taylor-jet square root requires a positive value")
        root = _TaylorJet.constant(self.space, float(np.sqrt(self.value)))
        # Newton doubles the correct polynomial degree per iteration.
        for _ in range(4):
            root = 0.5 * (root + self / root)
        return root


def _polynomial(
    coefficients: Sequence[float],
    argument: _TaylorJet,
) -> _TaylorJet:
    result = _TaylorJet.constant(argument.space, 0.0)
    for coefficient in reversed(tuple(float(value) for value in coefficients)):
        result = result * argument + coefficient
    return result


def _dot(
    left: Sequence[_TaylorJet],
    right: Sequence[_TaylorJet],
) -> _TaylorJet:
    if not left:
        raise ValueError("Taylor-jet dot product requires a non-empty vector")
    result = _TaylorJet.constant(left[0].space, 0.0)
    for left_value, right_value in zip(left, right):
        result += left_value * right_value
    return result


def _norm(vector: Sequence[_TaylorJet]) -> _TaylorJet:
    return _dot(vector, vector).sqrt()


def _four_vector(value: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (4,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain four finite values")
    return cast(np.ndarray, vector)


def _validated_segment(
    *,
    observer_time_ns: float,
    observer_position_mm: Sequence[float],
    segment_start_time_ns: float,
    segment_duration_ns: float,
    position_coefficients_mm: np.ndarray,
    retarded_time_ns: float,
) -> tuple[float, np.ndarray, float, float, np.ndarray, float]:
    observer_time = float(observer_time_ns)
    observer_position = np.asarray(observer_position_mm, dtype=float)
    start_time = float(segment_start_time_ns)
    duration = float(segment_duration_ns)
    coefficients = np.asarray(position_coefficients_mm, dtype=float)
    root_time = float(retarded_time_ns)
    if not np.isfinite(observer_time):
        raise ValueError("observer_time_ns must be finite")
    if observer_position.shape != (3,) or not np.all(np.isfinite(observer_position)):
        raise ValueError("observer_position_mm must contain three finite values")
    if not np.isfinite(start_time) or not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("the segment start and positive duration must be finite")
    if coefficients.shape != (6, 3) or not np.all(np.isfinite(coefficients)):
        raise ValueError("position_coefficients_mm must have finite shape (6, 3)")
    if not np.isfinite(root_time):
        raise ValueError("retarded_time_ns must be finite")
    fraction = (root_time - start_time) / duration
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "retarded_time_ns must lie strictly inside the selected smooth segment"
        )
    return (
        observer_time,
        observer_position,
        start_time,
        duration,
        coefficients,
        root_time,
    )


def _observer_coordinates(
    space: _JetSpace,
    observer_time_ns: float,
    observer_position_mm: np.ndarray,
) -> tuple[_TaylorJet, _TaylorJet, _TaylorJet, _TaylorJet]:
    return cast(
        tuple[_TaylorJet, _TaylorJet, _TaylorJet, _TaylorJet],
        (
            _TaylorJet.variable(space, C_MMNS * observer_time_ns, 0),
            *(
                _TaylorJet.variable(
                    space,
                    float(observer_position_mm[index]),
                    index + 1,
                )
                for index in range(3)
            ),
        ),
    )


def _directional_potential_result(
    *,
    potential: Sequence[_TaylorJet],
    velocity: np.ndarray,
    acceleration: np.ndarray,
    retarded_coordinate: _TaylorJet,
    light_cone: _TaylorJet,
    segment_fraction: float,
) -> PotentialDirectionalDerivativeJet:
    four_potential = np.asarray([component.value for component in potential])
    partial_a = np.zeros((4, 4), dtype=float)
    partial2_a = np.zeros((4, 4, 4), dtype=float)
    partial3_velocity = np.zeros((4, 4, 4), dtype=float)
    partial3_acceleration = np.zeros((4, 4, 4), dtype=float)
    partial4_velocity_twice = np.zeros((4, 4, 4), dtype=float)
    for left in range(4):
        for right in range(4):
            for component in range(4):
                potential_component = potential[component]
                partial2_a[left, right, component] = potential_component.derivative(
                    left, right
                )
                for direction in range(4):
                    third = potential_component.derivative(direction, left, right)
                    partial3_velocity[left, right, component] += (
                        velocity[direction] * third
                    )
                    partial3_acceleration[left, right, component] += (
                        acceleration[direction] * third
                    )
                    for second_direction in range(4):
                        partial4_velocity_twice[left, right, component] += (
                            velocity[direction]
                            * velocity[second_direction]
                            * potential_component.derivative(
                                direction,
                                second_direction,
                                left,
                                right,
                            )
                        )
    for derivative_index in range(4):
        for component in range(4):
            partial_a[derivative_index, component] = potential[component].derivative(
                derivative_index
            )
    residual = float(np.max(np.abs(light_cone.coefficients)))
    return PotentialDirectionalDerivativeJet(
        derivatives=PotentialDirectionalDerivatives(
            four_potential=four_potential,
            partial_a=partial_a,
            partial2_a=partial2_a,
            partial3_a_along_velocity=partial3_velocity,
            partial3_a_along_acceleration=partial3_acceleration,
            partial4_a_along_velocity_twice=partial4_velocity_twice,
        ),
        retarded_time_ns=retarded_coordinate.value / C_MMNS,
        light_cone_jet_residual=residual,
        segment_fraction=float(segment_fraction),
    )


def sum_potential_directional_derivatives_native(
    *values: PotentialDirectionalDerivatives,
) -> PotentialDirectionalDerivatives:
    """Add charge/dipole/source contributions without changing their ordering."""

    four_potential = np.zeros(4, dtype=float)
    partial_a = np.zeros((4, 4), dtype=float)
    partial2_a = np.zeros((4, 4, 4), dtype=float)
    partial3_velocity = np.zeros((4, 4, 4), dtype=float)
    partial3_acceleration = np.zeros((4, 4, 4), dtype=float)
    partial4_velocity_twice = np.zeros((4, 4, 4), dtype=float)
    for value in values:
        four_potential += value.four_potential
        partial_a += value.partial_a
        partial2_a += value.partial2_a
        partial3_velocity += value.partial3_a_along_velocity
        partial3_acceleration += value.partial3_a_along_acceleration
        partial4_velocity_twice += value.partial4_a_along_velocity_twice
    return PotentialDirectionalDerivatives(
        four_potential=four_potential,
        partial_a=partial_a,
        partial2_a=partial2_a,
        partial3_a_along_velocity=partial3_velocity,
        partial3_a_along_acceleration=partial3_acceleration,
        partial4_a_along_velocity_twice=partial4_velocity_twice,
    )


def quintic_charge_potential_directional_jet_native(
    *,
    observer_time_ns: float,
    observer_position_mm: Sequence[float],
    charge_native: float,
    segment_start_time_ns: float,
    segment_duration_ns: float,
    position_coefficients_mm: np.ndarray,
    retarded_time_ns: float,
    four_velocity_mm_ns: Sequence[float],
    four_acceleration_mm_ns2: Sequence[float],
    jet_newton_iterations: int = 5,
) -> PotentialDirectionalDerivativeJet:
    """Return the required charge-potential contractions from one root."""

    (
        observer_time,
        observer_position,
        start_time,
        duration,
        coefficients,
        root_time,
    ) = _validated_segment(
        observer_time_ns=observer_time_ns,
        observer_position_mm=observer_position_mm,
        segment_start_time_ns=segment_start_time_ns,
        segment_duration_ns=segment_duration_ns,
        position_coefficients_mm=position_coefficients_mm,
        retarded_time_ns=retarded_time_ns,
    )
    charge = float(charge_native)
    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    acceleration = _four_vector(
        four_acceleration_mm_ns2,
        name="four_acceleration_mm_ns2",
    )
    iterations = int(jet_newton_iterations)
    if not np.isfinite(charge):
        raise ValueError("charge_native must be finite")
    if iterations < 4:
        raise ValueError("jet_newton_iterations must be at least four")

    space = _jet_space(4)
    observer = _observer_coordinates(space, observer_time, observer_position)
    root_coordinate = _TaylorJet.constant(space, C_MMNS * root_time)
    start_coordinate = C_MMNS * start_time
    duration_coordinate = C_MMNS * duration

    def source_state(
        coordinate: _TaylorJet,
    ) -> tuple[list[_TaylorJet], list[_TaylorJet]]:
        normalized_time = (coordinate - start_coordinate) / duration_coordinate
        position = [
            _polynomial(coefficients[:, component], normalized_time)
            for component in range(3)
        ]
        beta = [
            _polynomial(
                tuple(
                    order * coefficients[order, component] / duration_coordinate
                    for order in range(1, 6)
                ),
                normalized_time,
            )
            for component in range(3)
        ]
        return position, beta

    light_cone = _TaylorJet.constant(space, 0.0)
    for _ in range(iterations):
        source_position, source_beta = source_state(root_coordinate)
        separation_vector = [
            observer[index + 1] - source_position[index] for index in range(3)
        ]
        separation = _norm(separation_vector)
        direction = [component / separation for component in separation_vector]
        light_cone = observer[0] - root_coordinate - separation
        derivative = -1.0 + _dot(direction, source_beta)
        root_coordinate = root_coordinate - light_cone.with_value(0.0) / derivative

    source_position, source_beta = source_state(root_coordinate)
    separation_vector = [
        observer[index + 1] - source_position[index] for index in range(3)
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
    light_cone = observer[0] - root_coordinate - separation
    return _directional_potential_result(
        potential=potential,
        velocity=velocity,
        acceleration=acceleration,
        retarded_coordinate=root_coordinate,
        light_cone=light_cone,
        segment_fraction=(root_time - start_time) / duration,
    )


def _spin_coefficients(
    start_spin: np.ndarray,
    end_spin: np.ndarray,
    start_slope_per_ns: np.ndarray,
    end_slope_per_ns: np.ndarray,
    duration_ns: float,
) -> np.ndarray:
    coefficients = np.empty((4, 3), dtype=float)
    start_tangent = duration_ns * start_slope_per_ns
    end_tangent = duration_ns * end_slope_per_ns
    coefficients[0] = start_spin
    coefficients[1] = start_tangent
    coefficients[2] = (
        -3.0 * start_spin - 2.0 * start_tangent + 3.0 * end_spin - end_tangent
    )
    coefficients[3] = 2.0 * start_spin + start_tangent - 2.0 * end_spin + end_tangent
    return coefficients


def _hodge_dual(jet_tensor: np.ndarray) -> np.ndarray:
    sample = cast(_TaylorJet, jet_tensor[0, 0])
    result = np.empty((4, 4), dtype=object)
    for mu in range(4):
        for nu in range(4):
            total = _TaylorJet.constant(sample.space, 0.0)
            for alpha in range(4):
                for beta in range(4):
                    coefficient = (
                        0.5
                        * _LEVI_CIVITA_UPPER[mu, nu, alpha, beta]
                        * _METRIC_SIGNS[alpha]
                        * _METRIC_SIGNS[beta]
                    )
                    if coefficient != 0.0:
                        total += coefficient * jet_tensor[alpha, beta]
            result[mu, nu] = total
    return result


def quintic_dipole_potential_directional_jet_native(
    *,
    observer_time_ns: float,
    observer_position_mm: Sequence[float],
    magnetic_moment_native: float,
    segment_start_time_ns: float,
    segment_duration_ns: float,
    position_coefficients_mm: np.ndarray,
    rest_spin_start: Sequence[float],
    rest_spin_end: Sequence[float],
    rest_spin_start_derivative_per_ns: Sequence[float],
    rest_spin_end_derivative_per_ns: Sequence[float],
    preserved_rest_spin_magnitude: float | None,
    retarded_time_ns: float,
    four_velocity_mm_ns: Sequence[float],
    four_acceleration_mm_ns2: Sequence[float],
    jet_newton_iterations: int = 6,
) -> PotentialDirectionalDerivativeJet:
    """Return the required dipole-potential contractions from one root."""

    (
        observer_time,
        observer_position,
        start_time,
        duration,
        position_coefficients,
        root_time,
    ) = _validated_segment(
        observer_time_ns=observer_time_ns,
        observer_position_mm=observer_position_mm,
        segment_start_time_ns=segment_start_time_ns,
        segment_duration_ns=segment_duration_ns,
        position_coefficients_mm=position_coefficients_mm,
        retarded_time_ns=retarded_time_ns,
    )
    moment = float(magnetic_moment_native)
    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    acceleration = _four_vector(
        four_acceleration_mm_ns2,
        name="four_acceleration_mm_ns2",
    )
    spin_start = np.asarray(rest_spin_start, dtype=float)
    spin_end = np.asarray(rest_spin_end, dtype=float)
    spin_start_slope = np.asarray(rest_spin_start_derivative_per_ns, dtype=float)
    spin_end_slope = np.asarray(rest_spin_end_derivative_per_ns, dtype=float)
    spin_vectors = (spin_start, spin_end, spin_start_slope, spin_end_slope)
    iterations = int(jet_newton_iterations)
    if not np.isfinite(moment):
        raise ValueError("magnetic_moment_native must be finite")
    if any(
        vector.shape != (3,) or not np.all(np.isfinite(vector))
        for vector in spin_vectors
    ):
        raise ValueError(
            "source spin values and slopes must contain three finite values"
        )
    if preserved_rest_spin_magnitude is not None and (
        not np.isfinite(preserved_rest_spin_magnitude)
        or preserved_rest_spin_magnitude < 0.0
    ):
        raise ValueError(
            "preserved_rest_spin_magnitude must be finite and non-negative"
        )
    if iterations < 4:
        raise ValueError("jet_newton_iterations must be at least four")

    space = _jet_space(5)
    observer = _observer_coordinates(space, observer_time, observer_position)
    root_coordinate = _TaylorJet.constant(space, C_MMNS * root_time)
    start_coordinate = C_MMNS * start_time
    duration_coordinate = C_MMNS * duration
    spin_coefficients = _spin_coefficients(
        spin_start,
        spin_end,
        spin_start_slope,
        spin_end_slope,
        duration,
    )

    def source_state(
        coordinate: _TaylorJet,
    ) -> tuple[list[_TaylorJet], list[_TaylorJet], list[_TaylorJet]]:
        normalized_time = (coordinate - start_coordinate) / duration_coordinate
        source_position = [
            _polynomial(position_coefficients[:, component], normalized_time)
            for component in range(3)
        ]
        source_beta = [
            _polynomial(
                tuple(
                    order
                    * position_coefficients[order, component]
                    / duration_coordinate
                    for order in range(1, 6)
                ),
                normalized_time,
            )
            for component in range(3)
        ]
        source_spin = [
            _polynomial(spin_coefficients[:, component], normalized_time)
            for component in range(3)
        ]
        target = preserved_rest_spin_magnitude
        if target is not None:
            if target == 0.0:
                source_spin = [_TaylorJet.constant(space, 0.0) for _ in range(3)]
            else:
                magnitude = _norm(source_spin)
                if magnitude.value <= 1.0e-15:
                    raise ValueError(
                        "constant-magnitude source-spin interpolation crossed zero"
                    )
                source_spin = [
                    component * (target / magnitude) for component in source_spin
                ]
        return source_position, source_beta, source_spin

    light_cone = _TaylorJet.constant(space, 0.0)
    for _ in range(iterations):
        source_position, source_beta, _ = source_state(root_coordinate)
        separation_vector = [
            observer[index + 1] - source_position[index] for index in range(3)
        ]
        separation = _norm(separation_vector)
        direction = [component / separation for component in separation_vector]
        light_cone = observer[0] - root_coordinate - separation
        derivative = -1.0 + _dot(direction, source_beta)
        root_coordinate = root_coordinate - light_cone.with_value(0.0) / derivative

    source_position, source_beta, source_spin = source_state(root_coordinate)
    separation_vector = [
        observer[index + 1] - source_position[index] for index in range(3)
    ]
    separation = _norm(separation_vector)
    direction = [component / separation for component in separation_vector]
    kappa = 1.0 - _dot(direction, source_beta)
    if kappa.value <= 1.0e-14:
        raise ValueError(
            "retarded dipole response is singular because 1 - n.beta is too small"
        )
    beta_squared = _dot(source_beta, source_beta)
    gamma = 1.0 / (1.0 - beta_squared).sqrt()
    projection = _dot(source_beta, source_spin)
    boost_coefficient = gamma * gamma / (gamma + 1.0) * projection
    moment_four = np.empty(4, dtype=object)
    moment_four[0] = moment * gamma * projection
    for component in range(3):
        moment_four[component + 1] = moment * (
            source_spin[component] + boost_coefficient * source_beta[component]
        )
    four_velocity = np.empty(4, dtype=object)
    four_velocity[0] = gamma * C_MMNS
    for component in range(3):
        four_velocity[component + 1] = gamma * C_MMNS * source_beta[component]
    wedge = np.empty((4, 4), dtype=object)
    for mu in range(4):
        for nu in range(4):
            wedge[mu, nu] = (
                four_velocity[mu] * moment_four[nu]
                - moment_four[mu] * four_velocity[nu]
            ) / C_MMNS
    hertz = _hodge_dual(wedge)
    invariant_distance = gamma * separation * kappa
    for mu in range(4):
        for nu in range(4):
            hertz[mu, nu] = hertz[mu, nu] / invariant_distance

    potential: list[_TaylorJet] = []
    for mu in range(4):
        # A derivative of a jet is a scalar.  Reconstruct the shifted Taylor
        # series so subsequent derivatives represent partial_rho H^(mu rho).
        coefficients = np.zeros(len(space.multiindices), dtype=float)
        for alpha_index, alpha in enumerate(space.multiindices):
            total = 0.0
            for rho in range(4):
                shifted = list(alpha)
                shifted[rho] += 1
                if sum(shifted) > space.order:
                    continue
                shifted_index = space.index[
                    cast(tuple[int, int, int, int], tuple(shifted))
                ]
                total += (
                    cast(_TaylorJet, hertz[mu, rho]).coefficients[shifted_index]
                    * shifted[rho]
                )
            coefficients[alpha_index] = total
        potential.append(_TaylorJet(space, coefficients))

    light_cone = observer[0] - root_coordinate - separation
    return _directional_potential_result(
        potential=potential,
        velocity=velocity,
        acceleration=acceleration,
        retarded_coordinate=root_coordinate,
        light_cone=light_cone,
        segment_fraction=(root_time - start_time) / duration,
    )


def _validated_boundary_guard(value: float) -> float:
    guard = float(value)
    if not np.isfinite(guard) or guard < 0.0 or guard >= 0.5:
        raise ValueError("boundary_guard_fraction must be finite in [0, 0.5)")
    return guard


def _segment_at_retarded_time(
    times_ns: np.ndarray,
    retarded_time_ns: float,
) -> tuple[int, float]:
    if times_ns.size < 2:
        raise ValueError("source has no smooth two-knot worldline segment")
    segment = int(np.searchsorted(times_ns, retarded_time_ns, side="right") - 1)
    segment = min(max(segment, 0), int(times_ns.size) - 2)
    duration = float(times_ns[segment + 1] - times_ns[segment])
    fraction = (float(retarded_time_ns) - float(times_ns[segment])) / duration
    return segment, float(fraction)


def _provider_unavailable(
    *,
    reason: str,
    source_identities: tuple[Hashable, ...],
    valid_sources: np.ndarray,
    retarded_time_ns: np.ndarray,
    segment_index: np.ndarray,
    segment_fraction: np.ndarray,
    jet_residual: np.ndarray,
) -> RetardedPotentialDirectionalJetProviderResult:
    return RetardedPotentialDirectionalJetProviderResult(
        derivatives=None,
        available=False,
        unavailable_reason=reason,
        source_identities=source_identities,
        valid_sources=valid_sources,
        retarded_time_ns=retarded_time_ns,
        source_segment_index=segment_index,
        source_segment_fraction=segment_fraction,
        source_jet_residual=jet_residual,
    )


def evaluate_retarded_charge_potential_directional_jet_native(
    history: "TrajectoryHistory",
    observer_event: "ObserverEvent",
    *,
    four_velocity_mm_ns: Sequence[float],
    four_acceleration_mm_ns2: Sequence[float],
    excluded_source_indices: Sequence[int] = (),
    require_complete_history: bool = True,
    boundary_guard_fraction: float = 1.0e-6,
    minimum_separation_mm: float = 1.0e-15,
    root_tolerance_mm: float = 1.0e-21,
    max_root_iterations: int = 96,
) -> RetardedPotentialDirectionalJetProviderResult:
    """Evaluate charge-source directional jets on smooth history segments.

    No numerical higher-derivative fallback is hidden here.  If any valid
    source root lies at or inside the declared segment-boundary guard, the
    result is explicitly unavailable so the caller can use the separately
    validated accepted-history reduction instead.
    """

    from .retarded_fields import (
        RetardedHistoryError,
        _prepare_history,
        _solve_retarded_sample,
        _source_terminated_before_light_cone,
        _validated_root_options,
    )

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    acceleration = _four_vector(
        four_acceleration_mm_ns2,
        name="four_acceleration_mm_ns2",
    )
    guard = _validated_boundary_guard(boundary_guard_fraction)
    minimum_separation = float(minimum_separation_mm)
    if not np.isfinite(minimum_separation) or minimum_separation <= 0.0:
        raise ValueError("minimum_separation_mm must be finite and positive")
    tolerance, iterations = _validated_root_options(
        root_tolerance_mm,
        max_root_iterations,
    )
    prepared = _prepare_history(history, excluded_source_indices)
    arrays = prepared.arrays
    identities: tuple[Hashable, ...] = tuple(range(arrays.n_sources))
    valid_sources = np.zeros(arrays.n_sources, dtype=bool)
    retarded_times = np.full(arrays.n_sources, np.nan, dtype=float)
    segment_indices = np.full(arrays.n_sources, -1, dtype=int)
    segment_fractions = np.full(arrays.n_sources, np.nan, dtype=float)
    jet_residuals = np.full(arrays.n_sources, np.nan, dtype=float)
    missing_sources: list[int] = []
    observer_time = float(observer_event.time_ns)
    observer_position = np.asarray(observer_event.position_mm, dtype=float)
    for source_index, source in prepared.sources.items():
        sample = _solve_retarded_sample(
            source,
            observer_time_ns=observer_time,
            observer_position_mm=observer_position,
            root_tolerance_mm=tolerance,
            max_root_iterations=iterations,
        )
        if sample is None:
            if _source_terminated_before_light_cone(
                source,
                observer_time_ns=observer_time,
                observer_position_mm=observer_position,
            ):
                continue
            missing_sources.append(source_index)
            continue
        if sample.separation_mm <= minimum_separation:
            raise ValueError(
                "observer event is within minimum_separation_mm of charge "
                f"source index {source_index}: {sample.separation_mm:.17g} <= "
                f"{minimum_separation:.17g} mm"
            )
        valid_sources[source_index] = True
        retarded_times[source_index] = sample.time_ns
    if require_complete_history and missing_sources:
        raise RetardedHistoryError(
            "source history does not bracket the observer light cone for source "
            f"indices {missing_sources}"
        )

    source_results: list[PotentialDirectionalDerivatives] = []
    for source_index, source in prepared.sources.items():
        if not valid_sources[source_index]:
            continue
        root_time = float(retarded_times[source_index])
        segment, fraction = _segment_at_retarded_time(source.time_ns, root_time)
        segment_indices[source_index] = segment
        segment_fractions[source_index] = fraction
        if fraction <= guard or fraction >= 1.0 - guard:
            return _provider_unavailable(
                reason=(
                    "charge retarded root is inside the nonsmooth worldline "
                    f"segment-boundary guard for source {source_index}: "
                    f"fraction={fraction:.17g}"
                ),
                source_identities=identities,
                valid_sources=valid_sources,
                retarded_time_ns=retarded_times,
                segment_index=segment_indices,
                segment_fraction=segment_fractions,
                jet_residual=jet_residuals,
            )
        result = quintic_charge_potential_directional_jet_native(
            observer_time_ns=observer_time,
            observer_position_mm=observer_position,
            charge_native=float(arrays.charge_native[source_index]),
            segment_start_time_ns=float(source.time_ns[segment]),
            segment_duration_ns=float(source.segment_duration_ns[segment]),
            position_coefficients_mm=source.position_coefficients_mm[segment],
            retarded_time_ns=root_time,
            four_velocity_mm_ns=velocity,
            four_acceleration_mm_ns2=acceleration,
        )
        jet_residuals[source_index] = result.light_cone_jet_residual
        source_results.append(result.derivatives)
    derivatives = sum_potential_directional_derivatives_native(*source_results)
    return RetardedPotentialDirectionalJetProviderResult(
        derivatives=derivatives,
        available=True,
        unavailable_reason=None,
        source_identities=identities,
        valid_sources=valid_sources,
        retarded_time_ns=retarded_times,
        source_segment_index=segment_indices,
        source_segment_fraction=segment_fractions,
        source_jet_residual=jet_residuals,
    )


def evaluate_retarded_dipole_potential_directional_jet_native(
    history: "TrajectoryHistory",
    observer_event: "ObserverEvent",
    *,
    four_velocity_mm_ns: Sequence[float],
    four_acceleration_mm_ns2: Sequence[float],
    source_identities: Sequence[Hashable] | None = None,
    observer_source_identity: Hashable | None = None,
    excluded_source_identities: Sequence[Hashable] = (),
    require_complete_history: bool = True,
    boundary_guard_fraction: float = 1.0e-6,
    require_frozen_spin_segment: bool = True,
    minimum_separation_mm: float = 1.0e-15,
    root_tolerance_mm: float = 1.0e-21,
    max_root_iterations: int = 96,
    spin_interpolation_model: str = "centered_c1",
) -> RetardedPotentialDirectionalJetProviderResult:
    """Evaluate dipole-source directional jets on smooth frozen segments."""

    from .retarded_dipole_fields import (
        _evaluate_prepared_hertz_tensor_native,
        _prepare_dipole_history,
    )

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    acceleration = _four_vector(
        four_acceleration_mm_ns2,
        name="four_acceleration_mm_ns2",
    )
    guard = _validated_boundary_guard(boundary_guard_fraction)
    minimum_separation = float(minimum_separation_mm)
    if not np.isfinite(minimum_separation) or minimum_separation <= 0.0:
        raise ValueError("minimum_separation_mm must be finite and positive")
    from .retarded_fields import _validated_root_options

    tolerance, iterations = _validated_root_options(
        root_tolerance_mm,
        max_root_iterations,
    )
    prepared = _prepare_dipole_history(
        history,
        source_identities=source_identities,
        observer_source_identity=observer_source_identity,
        excluded_source_identities=excluded_source_identities,
        spin_interpolation_model=spin_interpolation_model,
    )
    center = _evaluate_prepared_hertz_tensor_native(
        prepared,
        observer_event,
        require_complete_history=require_complete_history,
        minimum_separation_mm=minimum_separation,
        root_tolerance_mm=tolerance,
        max_root_iterations=iterations,
    )
    segment_indices = np.full(len(center.source_identities), -1, dtype=int)
    segment_fractions = np.full(len(center.source_identities), np.nan, dtype=float)
    jet_residuals = np.full(len(center.source_identities), np.nan, dtype=float)
    source_results: list[PotentialDirectionalDerivatives] = []
    observer_time = float(observer_event.time_ns)
    observer_position = np.asarray(observer_event.position_mm, dtype=float)
    for source_index, source in prepared.sources.items():
        if not center.valid_sources[source_index]:
            continue
        root_time = float(center.retarded_time_ns[source_index])
        times = source.worldline.time_ns
        try:
            segment, fraction = _segment_at_retarded_time(times, root_time)
        except ValueError as exc:
            return _provider_unavailable(
                reason=f"dipole source {source.identity!r}: {exc}",
                source_identities=center.source_identities,
                valid_sources=center.valid_sources,
                retarded_time_ns=center.retarded_time_ns,
                segment_index=segment_indices,
                segment_fraction=segment_fractions,
                jet_residual=jet_residuals,
            )
        segment_indices[source_index] = segment
        segment_fractions[source_index] = fraction
        if require_frozen_spin_segment and segment == int(times.size) - 2:
            return _provider_unavailable(
                reason=(
                    "dipole retarded root lies in the mutable final source-spin "
                    f"segment for source {source.identity!r}"
                ),
                source_identities=center.source_identities,
                valid_sources=center.valid_sources,
                retarded_time_ns=center.retarded_time_ns,
                segment_index=segment_indices,
                segment_fraction=segment_fractions,
                jet_residual=jet_residuals,
            )
        if fraction <= guard or fraction >= 1.0 - guard:
            return _provider_unavailable(
                reason=(
                    "dipole retarded root is inside the nonsmooth worldline/spin "
                    f"segment-boundary guard for source {source.identity!r}: "
                    f"fraction={fraction:.17g}"
                ),
                source_identities=center.source_identities,
                valid_sources=center.valid_sources,
                retarded_time_ns=center.retarded_time_ns,
                segment_index=segment_indices,
                segment_fraction=segment_fractions,
                jet_residual=jet_residuals,
            )
        result = quintic_dipole_potential_directional_jet_native(
            observer_time_ns=observer_time,
            observer_position_mm=observer_position,
            magnetic_moment_native=source.magnetic_moment_native,
            segment_start_time_ns=float(times[segment]),
            segment_duration_ns=float(source.worldline.segment_duration_ns[segment]),
            position_coefficients_mm=(
                source.worldline.position_coefficients_mm[segment]
            ),
            rest_spin_start=source.rest_spin[segment],
            rest_spin_end=source.rest_spin[segment + 1],
            rest_spin_start_derivative_per_ns=(
                source.rest_spin_derivative_per_ns[segment]
            ),
            rest_spin_end_derivative_per_ns=(
                source.rest_spin_derivative_per_ns[segment + 1]
            ),
            preserved_rest_spin_magnitude=source.preserved_rest_spin_magnitude,
            retarded_time_ns=root_time,
            four_velocity_mm_ns=velocity,
            four_acceleration_mm_ns2=acceleration,
        )
        jet_residuals[source_index] = result.light_cone_jet_residual
        source_results.append(result.derivatives)
    derivatives = sum_potential_directional_derivatives_native(*source_results)
    return RetardedPotentialDirectionalJetProviderResult(
        derivatives=derivatives,
        available=True,
        unavailable_reason=None,
        source_identities=center.source_identities,
        valid_sources=center.valid_sources,
        retarded_time_ns=center.retarded_time_ns,
        source_segment_index=segment_indices,
        source_segment_fraction=segment_fractions,
        source_jet_residual=jet_residuals,
    )


__all__ = [
    "PotentialDirectionalDerivativeJet",
    "PotentialDirectionalDerivatives",
    "RetardedPotentialDirectionalJetProviderResult",
    "evaluate_retarded_charge_potential_directional_jet_native",
    "evaluate_retarded_dipole_potential_directional_jet_native",
    "quintic_charge_potential_directional_jet_native",
    "quintic_dipole_potential_directional_jet_native",
    "sum_potential_directional_derivatives_native",
]
