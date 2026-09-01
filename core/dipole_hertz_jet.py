"""Third-order analytical observer jet of one retarded dipole Hertz tensor.

This module is a validation oracle for the potential-first dipole provider.  A
single already-bracketed retarded root selects one smooth quintic-worldline and
cubic-spin segment.  A four-variable Taylor jet then differentiates the
implicit light-cone equation and the covariant Hertz tensor without displaced
observer events or additional root solves.

Coordinates are ``x=(ct,x,y,z)`` in millimetres.  Taylor coefficients are
factorial-scaled internally; all returned derivatives are ordinary physical
derivatives.  The caller must use the finite-difference provider at a spin or
worldline segment boundary, where the current C1 spin interpolant does not
possess unique second and third derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial
from typing import TYPE_CHECKING, Hashable, Iterable, Sequence, cast

import numpy as np

from .constants import C_MMNS
from .rfs import fields_from_tensor_native

if TYPE_CHECKING:
    from .retarded_dipole_fields import (
        RetardedDipoleFieldGradientResult,
        RetardedDipoleResponseGradientResult,
    )
    from .retarded_fields import ObserverEvent, TrajectoryHistory

_DIMENSION = 4
_ORDER = 3
_METRIC_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0))


def _multiindices() -> tuple[tuple[int, int, int, int], ...]:
    values: list[tuple[int, int, int, int]] = []
    for total_degree in range(_ORDER + 1):
        for alpha in product(range(total_degree + 1), repeat=_DIMENSION):
            if sum(alpha) == total_degree:
                values.append(cast(tuple[int, int, int, int], alpha))
    return tuple(values)


_MULTIINDICES = _multiindices()
_INDEX = {alpha: index for index, alpha in enumerate(_MULTIINDICES)}
_ZERO = (0, 0, 0, 0)


def _splits(
    alpha: tuple[int, int, int, int],
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for beta in product(*(range(value + 1) for value in alpha)):
        gamma = cast(
            tuple[int, int, int, int],
            tuple(alpha[index] - beta[index] for index in range(_DIMENSION)),
        )
        result.append((_INDEX[cast(tuple[int, int, int, int], beta)], _INDEX[gamma]))
    return tuple(result)


_PRODUCT_SPLITS = tuple(_splits(alpha) for alpha in _MULTIINDICES)


def _levi_civita_upper() -> np.ndarray:
    tensor = np.zeros((4, 4, 4, 4), dtype=float)
    for permutation in permutations(range(4)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(4)
            for right in range(left + 1, 4)
        )
        # epsilon_0123=+1 and raising all four indices with (+---) changes sign.
        tensor[permutation] = -1.0 if inversions % 2 == 0 else 1.0
    return tensor


_LEVI_CIVITA_UPPER = _levi_civita_upper()


@dataclass(frozen=True)
class DipoleHertzResponseJetResult:
    """Hertz tensor and complete ordinary dipole response at one event."""

    hertz_tensor: np.ndarray
    four_potential: np.ndarray
    partial_a: np.ndarray
    electric_field_native: np.ndarray
    magnetic_field_native: np.ndarray
    field_tensor: np.ndarray
    partial_f: np.ndarray
    retarded_time_ns: float
    retarded_coordinate_gradient: np.ndarray
    retarded_coordinate_hessian: np.ndarray
    retarded_coordinate_third_derivative: np.ndarray
    light_cone_jet_residual: float
    segment_fraction: float


@dataclass(frozen=True)
class DipoleHertzSparseResponseJetResult:
    """The 34 production response values from one smooth source segment."""

    four_potential: np.ndarray
    antisymmetric_response: np.ndarray
    partial_antisymmetric_response: np.ndarray
    retarded_time_ns: float
    light_cone_jet_residual: float
    segment_fraction: float


@dataclass(frozen=True)
class DipoleHertzJetProviderResult:
    """History-facing analytical result and explicit fallback diagnostics."""

    response: "RetardedDipoleFieldGradientResult | RetardedDipoleResponseGradientResult"
    used_analytic_response: bool
    fallback_reason: str | None
    source_segment_index: np.ndarray
    source_segment_fraction: np.ndarray
    source_jet_residual: np.ndarray


@dataclass(frozen=True)
class _Jet3:
    coefficients: np.ndarray

    @classmethod
    def constant(cls, value: float) -> "_Jet3":
        coefficients = np.zeros(len(_MULTIINDICES), dtype=float)
        coefficients[0] = float(value)
        return cls(coefficients)

    @classmethod
    def variable(cls, value: float, index: int) -> "_Jet3":
        coefficients = np.zeros(len(_MULTIINDICES), dtype=float)
        coefficients[0] = float(value)
        alpha = [0, 0, 0, 0]
        alpha[index] = 1
        coefficients[_INDEX[cast(tuple[int, int, int, int], tuple(alpha))]] = 1.0
        return cls(coefficients)

    @property
    def value(self) -> float:
        return float(self.coefficients[0])

    def with_value(self, value: float) -> "_Jet3":
        coefficients = self.coefficients.copy()
        coefficients[0] = float(value)
        return _Jet3(coefficients)

    def derivative(self, *indices: int) -> float:
        if len(indices) > _ORDER:
            raise ValueError("a third-order jet cannot provide a higher derivative")
        alpha = [0, 0, 0, 0]
        for index in indices:
            alpha[index] += 1
        multiindex = cast(tuple[int, int, int, int], tuple(alpha))
        coefficient = self.coefficients[_INDEX[multiindex]]
        scale = 1
        for multiplicity in multiindex:
            scale *= factorial(multiplicity)
        return float(scale * coefficient)

    def __add__(self, other: object) -> "_Jet3":
        return _Jet3(self.coefficients + _as_jet(other).coefficients)

    def __radd__(self, other: object) -> "_Jet3":
        return self + other

    def __neg__(self) -> "_Jet3":
        return _Jet3(-self.coefficients)

    def __sub__(self, other: object) -> "_Jet3":
        return self + (-_as_jet(other))

    def __rsub__(self, other: object) -> "_Jet3":
        return _as_jet(other) - self

    def __mul__(self, other: object) -> "_Jet3":
        right = _as_jet(other)
        coefficients = np.empty(len(_MULTIINDICES), dtype=float)
        for result_index, splits in enumerate(_PRODUCT_SPLITS):
            total = 0.0
            for left_index, right_index in splits:
                total += self.coefficients[left_index] * right.coefficients[right_index]
            coefficients[result_index] = total
        return _Jet3(coefficients)

    def __rmul__(self, other: object) -> "_Jet3":
        return self * other

    def reciprocal(self) -> "_Jet3":
        if self.value == 0.0:
            raise ZeroDivisionError("cannot invert a zero Taylor jet")
        coefficients = np.zeros(len(_MULTIINDICES), dtype=float)
        coefficients[0] = 1.0 / self.value
        for result_index in range(1, len(_MULTIINDICES)):
            total = 0.0
            # Remove beta=0.  The remaining complement always has lower degree,
            # so its reciprocal coefficient is already available.
            for left_index, right_index in _PRODUCT_SPLITS[result_index]:
                if left_index == 0:
                    continue
                total += self.coefficients[left_index] * coefficients[right_index]
            coefficients[result_index] = -total / self.value
        return _Jet3(coefficients)

    def __truediv__(self, other: object) -> "_Jet3":
        return self * _as_jet(other).reciprocal()

    def __rtruediv__(self, other: object) -> "_Jet3":
        return _as_jet(other) / self

    def sqrt(self) -> "_Jet3":
        if self.value <= 0.0:
            raise ValueError("Taylor-jet square root requires a positive value")
        root = _Jet3.constant(float(np.sqrt(self.value)))
        # Newton doubles the correct Taylor order on every iteration.  Three
        # iterations are sufficient through degree three from a constant seed.
        for _ in range(3):
            root = 0.5 * (root + self / root)
        return root


def _as_jet(value: object) -> _Jet3:
    if isinstance(value, _Jet3):
        return value
    return _Jet3.constant(float(cast(float, value)))


def _polynomial(coefficients: Iterable[float], argument: _Jet3) -> _Jet3:
    result = _Jet3.constant(0.0)
    for coefficient in reversed(tuple(float(value) for value in coefficients)):
        result = result * argument + coefficient
    return result


def _dot(left: Sequence[_Jet3], right: Sequence[_Jet3]) -> _Jet3:
    return sum((a * b for a, b in zip(left, right)), _Jet3.constant(0.0))


def _norm(vector: Sequence[_Jet3]) -> _Jet3:
    return _dot(vector, vector).sqrt()


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
    result = np.empty((4, 4), dtype=object)
    for mu in range(4):
        for nu in range(4):
            total = _Jet3.constant(0.0)
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


def polynomial_dipole_hertz_response_jet_native(
    *,
    observer_time_ns: float,
    observer_position_mm: Sequence[float],
    magnetic_moment_native: float,
    segment_start_time_ns: float,
    segment_duration_ns: float,
    position_coefficients_mm: np.ndarray,
    rest_spin_coefficients: np.ndarray,
    preserved_rest_spin_magnitude: float | None,
    retarded_time_ns: float,
    jet_newton_iterations: int = 4,
) -> DipoleHertzResponseJetResult:
    """Differentiate one smooth polynomial worldline/spin source segment.

    This pure-Python routine is a validation oracle.  It accepts arbitrary
    polynomial degrees so studies can compare the production interpolation
    against smoother causal-history candidates without changing production
    dispatch.
    """

    observer_time = float(observer_time_ns)
    observer_position = np.asarray(observer_position_mm, dtype=float)
    moment = float(magnetic_moment_native)
    start_time = float(segment_start_time_ns)
    duration = float(segment_duration_ns)
    position_coefficients = np.asarray(position_coefficients_mm, dtype=float)
    spin_coefficients = np.asarray(rest_spin_coefficients, dtype=float)
    root_time = float(retarded_time_ns)
    iterations = int(jet_newton_iterations)
    if not np.isfinite(observer_time):
        raise ValueError("observer_time_ns must be finite")
    if observer_position.shape != (3,) or not np.all(np.isfinite(observer_position)):
        raise ValueError("observer_position_mm must contain three finite values")
    if not np.isfinite(moment):
        raise ValueError("magnetic_moment_native must be finite")
    if not np.isfinite(start_time) or not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("the segment start and positive duration must be finite")
    if (
        position_coefficients.ndim != 2
        or position_coefficients.shape[0] < 2
        or position_coefficients.shape[1] != 3
        or not np.all(np.isfinite(position_coefficients))
    ):
        raise ValueError(
            "position_coefficients_mm must have finite shape (degree+1, 3)"
        )
    if (
        spin_coefficients.ndim != 2
        or spin_coefficients.shape[0] < 1
        or spin_coefficients.shape[1] != 3
        or not np.all(np.isfinite(spin_coefficients))
    ):
        raise ValueError("rest_spin_coefficients must have finite shape (degree+1, 3)")
    if preserved_rest_spin_magnitude is not None and (
        not np.isfinite(preserved_rest_spin_magnitude)
        or preserved_rest_spin_magnitude < 0.0
    ):
        raise ValueError(
            "preserved_rest_spin_magnitude must be finite and non-negative"
        )
    if not np.isfinite(root_time):
        raise ValueError("retarded_time_ns must be finite")
    fraction = (root_time - start_time) / duration
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "retarded_time_ns must lie strictly inside the selected smooth segment"
        )
    if iterations < 3:
        raise ValueError("jet_newton_iterations must be at least three")

    observer_coordinates = (
        _Jet3.variable(C_MMNS * observer_time, 0),
        *(
            _Jet3.variable(float(observer_position[index]), index + 1)
            for index in range(3)
        ),
    )
    root_coordinate = _Jet3.constant(C_MMNS * root_time)
    start_coordinate = C_MMNS * start_time
    duration_coordinate = C_MMNS * duration

    def source_state(
        source_coordinate: _Jet3,
    ) -> tuple[list[_Jet3], list[_Jet3], list[_Jet3]]:
        normalized_time = (source_coordinate - start_coordinate) / duration_coordinate
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
                    for order in range(1, position_coefficients.shape[0])
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
                source_spin = [_Jet3.constant(0.0) for _ in range(3)]
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

    light_cone = _Jet3.constant(0.0)
    for _ in range(iterations):
        source_position, source_beta, _ = source_state(root_coordinate)
        separation_vector = [
            observer_coordinates[index + 1] - source_position[index]
            for index in range(3)
        ]
        separation = _norm(separation_vector)
        direction = [component / separation for component in separation_vector]
        light_cone = observer_coordinates[0] - root_coordinate - separation
        derivative = -1.0 + _dot(direction, source_beta)
        # The scalar root solver owns the base root; solve only derivative slots.
        root_coordinate = root_coordinate - light_cone.with_value(0.0) / derivative

    source_position, source_beta, source_spin = source_state(root_coordinate)
    separation_vector = [
        observer_coordinates[index + 1] - source_position[index] for index in range(3)
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

    four_potential = np.zeros(4, dtype=float)
    partial_a = np.zeros((4, 4), dtype=float)
    partial_f = np.zeros((4, 4, 4), dtype=float)
    hertz_value = np.empty((4, 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            hertz_value[mu, nu] = hertz[mu, nu].value
    for mu in range(4):
        for rho in range(4):
            four_potential[mu] += hertz[mu, rho].derivative(rho)
    for derivative_index in range(4):
        for mu in range(4):
            for rho in range(4):
                partial_a[derivative_index, mu] += hertz[mu, rho].derivative(
                    derivative_index, rho
                )
    field_tensor = np.zeros((4, 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            field_tensor[mu, nu] = (
                _METRIC_SIGNS[mu] * partial_a[mu, nu]
                - _METRIC_SIGNS[nu] * partial_a[nu, mu]
            )
            for derivative_index in range(4):
                first = 0.0
                second = 0.0
                for rho in range(4):
                    first += hertz[nu, rho].derivative(derivative_index, mu, rho)
                    second += hertz[mu, rho].derivative(derivative_index, nu, rho)
                partial_f[derivative_index, mu, nu] = (
                    _METRIC_SIGNS[mu] * first - _METRIC_SIGNS[nu] * second
                )

    electric, magnetic = fields_from_tensor_native(field_tensor)
    light_cone = observer_coordinates[0] - root_coordinate - separation
    residual = max(abs(value) for value in light_cone.coefficients)
    return DipoleHertzResponseJetResult(
        hertz_tensor=hertz_value,
        four_potential=four_potential,
        partial_a=partial_a,
        electric_field_native=electric,
        magnetic_field_native=magnetic,
        field_tensor=field_tensor,
        partial_f=partial_f,
        retarded_time_ns=root_coordinate.value / C_MMNS,
        retarded_coordinate_gradient=np.asarray(
            [root_coordinate.derivative(index) for index in range(4)]
        ),
        retarded_coordinate_hessian=np.asarray(
            [
                [root_coordinate.derivative(left, right) for right in range(4)]
                for left in range(4)
            ]
        ),
        retarded_coordinate_third_derivative=np.asarray(
            [
                [
                    [
                        root_coordinate.derivative(first, second, third)
                        for third in range(4)
                    ]
                    for second in range(4)
                ]
                for first in range(4)
            ]
        ),
        light_cone_jet_residual=float(residual),
        segment_fraction=float(fraction),
    )


def quintic_dipole_hertz_response_jet_native(
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
    jet_newton_iterations: int = 4,
) -> DipoleHertzResponseJetResult:
    """Preserve the production quintic-worldline/cubic-spin oracle surface."""

    observer_time = float(observer_time_ns)
    observer_position = np.asarray(observer_position_mm, dtype=float)
    moment = float(magnetic_moment_native)
    start_time = float(segment_start_time_ns)
    duration = float(segment_duration_ns)
    position_coefficients = np.asarray(position_coefficients_mm, dtype=float)
    spin_start = np.asarray(rest_spin_start, dtype=float)
    spin_end = np.asarray(rest_spin_end, dtype=float)
    spin_start_slope = np.asarray(rest_spin_start_derivative_per_ns, dtype=float)
    spin_end_slope = np.asarray(rest_spin_end_derivative_per_ns, dtype=float)
    root_time = float(retarded_time_ns)
    iterations = int(jet_newton_iterations)
    vectors = (spin_start, spin_end, spin_start_slope, spin_end_slope)
    if not np.isfinite(observer_time):
        raise ValueError("observer_time_ns must be finite")
    if observer_position.shape != (3,) or not np.all(np.isfinite(observer_position)):
        raise ValueError("observer_position_mm must contain three finite values")
    if not np.isfinite(moment):
        raise ValueError("magnetic_moment_native must be finite")
    if not np.isfinite(start_time) or not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("the segment start and positive duration must be finite")
    if position_coefficients.shape != (6, 3) or not np.all(
        np.isfinite(position_coefficients)
    ):
        raise ValueError("position_coefficients_mm must have finite shape (6, 3)")
    if any(
        vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in vectors
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
    if not np.isfinite(root_time):
        raise ValueError("retarded_time_ns must be finite")
    fraction = (root_time - start_time) / duration
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "retarded_time_ns must lie strictly inside the selected smooth segment"
        )
    if iterations < 3:
        raise ValueError("jet_newton_iterations must be at least three")
    spin_coefficients = _spin_coefficients(
        spin_start,
        spin_end,
        spin_start_slope,
        spin_end_slope,
        duration,
    )
    return polynomial_dipole_hertz_response_jet_native(
        observer_time_ns=observer_time,
        observer_position_mm=observer_position,
        magnetic_moment_native=moment,
        segment_start_time_ns=start_time,
        segment_duration_ns=duration,
        position_coefficients_mm=position_coefficients,
        rest_spin_coefficients=spin_coefficients,
        preserved_rest_spin_magnitude=preserved_rest_spin_magnitude,
        retarded_time_ns=root_time,
        jet_newton_iterations=iterations,
    )


def quintic_dipole_hertz_response_jet_numba_native(
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
) -> DipoleHertzResponseJetResult:
    """Return the same smooth-segment response through strict Numba algebra."""

    from .dipole_hertz_jet_numba import (
        quintic_dipole_hertz_response_coefficients_strict_serial,
    )

    observer_position = np.asarray(observer_position_mm, dtype=float)
    position_coefficients = np.asarray(position_coefficients_mm, dtype=float)
    spin_start = np.asarray(rest_spin_start, dtype=float)
    spin_end = np.asarray(rest_spin_end, dtype=float)
    spin_start_slope = np.asarray(rest_spin_start_derivative_per_ns, dtype=float)
    spin_end_slope = np.asarray(rest_spin_end_derivative_per_ns, dtype=float)
    start_time = float(segment_start_time_ns)
    duration = float(segment_duration_ns)
    root_time = float(retarded_time_ns)
    fraction = (root_time - start_time) / duration
    if observer_position.shape != (3,) or not np.all(np.isfinite(observer_position)):
        raise ValueError("observer_position_mm must contain three finite values")
    if position_coefficients.shape != (6, 3) or not np.all(
        np.isfinite(position_coefficients)
    ):
        raise ValueError("position_coefficients_mm must have finite shape (6, 3)")
    vectors = (spin_start, spin_end, spin_start_slope, spin_end_slope)
    if any(
        vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in vectors
    ):
        raise ValueError(
            "source spin values and slopes must contain three finite values"
        )
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "retarded_time_ns must lie strictly inside the selected smooth segment"
        )
    preserve_magnitude = preserved_rest_spin_magnitude is not None
    preserved_magnitude = (
        0.0
        if preserved_rest_spin_magnitude is None
        else float(preserved_rest_spin_magnitude)
    )
    spin_coefficients = _spin_coefficients(
        spin_start,
        spin_end,
        spin_start_slope,
        spin_end_slope,
        duration,
    )
    (
        status,
        hertz,
        potential,
        partial_a,
        field,
        partial_f,
        root_gradient,
        root_hessian,
        root_third,
        residual,
    ) = quintic_dipole_hertz_response_coefficients_strict_serial(
        float(observer_time_ns),
        observer_position,
        float(magnetic_moment_native),
        start_time,
        duration,
        position_coefficients,
        spin_coefficients,
        preserve_magnitude,
        preserved_magnitude,
        root_time,
    )
    if status == 1:
        raise ValueError("constant-magnitude source-spin interpolation crossed zero")
    if status == 2:
        raise ValueError(
            "retarded dipole response is singular because 1 - n.beta is too small"
        )
    if status == 3:
        raise ValueError("source beta magnitude must be less than one")
    if status != 0:
        raise RuntimeError(f"unexpected analytical Hertz kernel status {status}")
    electric, magnetic = fields_from_tensor_native(field)
    return DipoleHertzResponseJetResult(
        hertz_tensor=hertz,
        four_potential=potential,
        partial_a=partial_a,
        electric_field_native=electric,
        magnetic_field_native=magnetic,
        field_tensor=field,
        partial_f=partial_f,
        retarded_time_ns=root_time,
        retarded_coordinate_gradient=root_gradient,
        retarded_coordinate_hessian=root_hessian,
        retarded_coordinate_third_derivative=root_third,
        light_cone_jet_residual=float(residual),
        segment_fraction=float(fraction),
    )


def quintic_dipole_hertz_sparse_response_numba_native(
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
) -> DipoleHertzSparseResponseJetResult:
    """Return only the compact potential/response surface through strict Numba."""

    from .dipole_hertz_jet_numba import (
        quintic_dipole_hertz_sparse_response_strict_serial,
    )

    observer_position = np.asarray(observer_position_mm, dtype=float)
    position_coefficients = np.asarray(position_coefficients_mm, dtype=float)
    spin_start = np.asarray(rest_spin_start, dtype=float)
    spin_end = np.asarray(rest_spin_end, dtype=float)
    spin_start_slope = np.asarray(rest_spin_start_derivative_per_ns, dtype=float)
    spin_end_slope = np.asarray(rest_spin_end_derivative_per_ns, dtype=float)
    start_time = float(segment_start_time_ns)
    duration = float(segment_duration_ns)
    root_time = float(retarded_time_ns)
    fraction = (root_time - start_time) / duration
    if observer_position.shape != (3,) or not np.all(np.isfinite(observer_position)):
        raise ValueError("observer_position_mm must contain three finite values")
    if position_coefficients.shape != (6, 3) or not np.all(
        np.isfinite(position_coefficients)
    ):
        raise ValueError("position_coefficients_mm must have finite shape (6, 3)")
    vectors = (spin_start, spin_end, spin_start_slope, spin_end_slope)
    if any(
        vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in vectors
    ):
        raise ValueError(
            "source spin values and slopes must contain three finite values"
        )
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "retarded_time_ns must lie strictly inside the selected smooth segment"
        )
    preserve_magnitude = preserved_rest_spin_magnitude is not None
    preserved_magnitude = (
        0.0
        if preserved_rest_spin_magnitude is None
        else float(preserved_rest_spin_magnitude)
    )
    spin_coefficients = _spin_coefficients(
        spin_start,
        spin_end,
        spin_start_slope,
        spin_end_slope,
        duration,
    )
    status, potential, packed_field, packed_partial_f, residual = (
        quintic_dipole_hertz_sparse_response_strict_serial(
            float(observer_time_ns),
            observer_position,
            float(magnetic_moment_native),
            start_time,
            duration,
            position_coefficients,
            spin_coefficients,
            preserve_magnitude,
            preserved_magnitude,
            root_time,
        )
    )
    if status == 1:
        raise ValueError("constant-magnitude source-spin interpolation crossed zero")
    if status == 2:
        raise ValueError(
            "retarded dipole response is singular because 1 - n.beta is too small"
        )
    if status == 3:
        raise ValueError("source beta magnitude must be less than one")
    if status != 0:
        raise RuntimeError(f"unexpected sparse analytical Hertz status {status}")
    return DipoleHertzSparseResponseJetResult(
        four_potential=potential,
        antisymmetric_response=packed_field,
        partial_antisymmetric_response=packed_partial_f,
        retarded_time_ns=root_time,
        light_cone_jet_residual=float(residual),
        segment_fraction=float(fraction),
    )


def evaluate_retarded_dipole_field_gradient_hertz_jet_native(
    history: "TrajectoryHistory",
    observer_event: "ObserverEvent",
    *,
    source_identities: Sequence[Hashable] | None = None,
    observer_source_identity: Hashable | None = None,
    excluded_source_identities: Sequence[Hashable] = (),
    require_complete_history: bool = True,
    boundary_guard_fraction: float = 1.0e-6,
    require_frozen_spin_segment: bool = True,
    fallback_relative_step: float = 1.0e-3,
    fallback_minimum_step_mm: float = 1.0e-15,
    fallback_stencil_step_mm: float | None = None,
    minimum_separation_mm: float = 1.0e-15,
    root_tolerance_mm: float = 1.0e-21,
    max_root_iterations: int = 96,
    response_kernel: str = "python",
    fallback_backend: str = "numba_full_strict_serial",
    spin_interpolation_model: str = "centered_c1",
) -> DipoleHertzJetProviderResult:
    """Evaluate the analytical response, falling back at nonsmooth knots.

    The current source-spin representation is only C1 across history knots.
    A retarded root within ``boundary_guard_fraction`` of either end of its
    selected segment therefore delegates the entire source sum to the existing
    129-event provider.  By default the mutable final spin segment also falls
    back: appending the next source knot changes its former endpoint slope.
    Frozen interior events use one scalar retarded root per source and the exact
    local polynomial jet above.
    """

    # Local imports keep this validation oracle independent from the production
    # provider's import graph until the backend is accepted for integration.
    from .retarded_dipole_fields import (
        RetardedDipoleFieldGradientResult,
        _evaluate_prepared_dipole_roots_numba_exact_serial,
        _evaluate_prepared_hertz_tensor_native,
        _prepare_dipole_history,
        evaluate_retarded_dipole_field_gradient_native,
    )

    boundary_guard = float(boundary_guard_fraction)
    if not np.isfinite(boundary_guard) or boundary_guard < 0.0 or boundary_guard >= 0.5:
        raise ValueError("boundary_guard_fraction must be finite in [0, 0.5)")
    if response_kernel not in (
        "python",
        "numba_strict_serial",
        "numba_sparse_strict_serial",
    ):
        raise ValueError(
            "response_kernel must be 'python', 'numba_strict_serial', or "
            "'numba_sparse_strict_serial'"
        )
    prepared = _prepare_dipole_history(
        history,
        source_identities=source_identities,
        observer_source_identity=observer_source_identity,
        excluded_source_identities=excluded_source_identities,
        spin_interpolation_model=spin_interpolation_model,
    )
    if response_kernel == "numba_sparse_strict_serial":
        center = _evaluate_prepared_dipole_roots_numba_exact_serial(
            prepared,
            (observer_event,),
            require_complete_history=require_complete_history,
            minimum_separation_mm=float(minimum_separation_mm),
            root_tolerance_mm=float(root_tolerance_mm),
            max_root_iterations=int(max_root_iterations),
        )[0]
    else:
        center = _evaluate_prepared_hertz_tensor_native(
            prepared,
            observer_event,
            require_complete_history=require_complete_history,
            minimum_separation_mm=float(minimum_separation_mm),
            root_tolerance_mm=float(root_tolerance_mm),
            max_root_iterations=int(max_root_iterations),
        )
    segment_index = np.full(len(center.source_identities), -1, dtype=int)
    segment_fraction = np.full(len(center.source_identities), np.nan, dtype=float)
    jet_residual = np.full(len(center.source_identities), np.nan, dtype=float)

    def fallback(reason: str) -> DipoleHertzJetProviderResult:
        from .analytic_dipole_hertz_diagnostics import (
            record_analytic_dipole_hertz_response,
        )

        full_response = evaluate_retarded_dipole_field_gradient_native(
            history,
            observer_event,
            source_identities=source_identities,
            observer_source_identity=observer_source_identity,
            excluded_source_identities=excluded_source_identities,
            require_complete_history=require_complete_history,
            relative_step=fallback_relative_step,
            minimum_step_mm=fallback_minimum_step_mm,
            stencil_step_mm=fallback_stencil_step_mm,
            minimum_separation_mm=minimum_separation_mm,
            root_tolerance_mm=root_tolerance_mm,
            max_root_iterations=max_root_iterations,
            backend=fallback_backend,
            spin_interpolation_model=spin_interpolation_model,
        )
        response: (
            RetardedDipoleFieldGradientResult | RetardedDipoleResponseGradientResult
        )
        if response_kernel == "numba_sparse_strict_serial":
            from .antisymmetric_response_rfs import (
                pack_antisymmetric_response_native,
                pack_partial_antisymmetric_response_native,
            )
            from .retarded_dipole_fields import (
                RetardedDipoleResponseGradientResult,
                RetardedDipoleRootResult,
            )

            response = RetardedDipoleResponseGradientResult(
                four_potential=full_response.four_potential,
                antisymmetric_response=pack_antisymmetric_response_native(
                    full_response.field_tensor
                ),
                partial_antisymmetric_response=(
                    pack_partial_antisymmetric_response_native(full_response.partial_f)
                ),
                root=RetardedDipoleRootResult(
                    source_identities=full_response.hertz.source_identities,
                    retarded_time_ns=full_response.hertz.retarded_time_ns,
                    light_cone_residual_mm=(full_response.hertz.light_cone_residual_mm),
                    separation_mm=full_response.hertz.separation_mm,
                    valid_sources=full_response.hertz.valid_sources,
                ),
                used_analytic_response=False,
                fallback_reason=reason,
                source_segment_index=segment_index.copy(),
                source_segment_fraction=segment_fraction.copy(),
                source_jet_residual=jet_residual.copy(),
            )
        else:
            response = full_response
        finite_fraction = segment_fraction[np.isfinite(segment_fraction)]
        minimum_boundary_fraction = (
            float(np.min(np.minimum(finite_fraction, 1.0 - finite_fraction)))
            if finite_fraction.size
            else float("inf")
        )
        record_analytic_dipole_hertz_response(
            valid_sources=int(np.count_nonzero(center.valid_sources)),
            minimum_boundary_fraction=minimum_boundary_fraction,
            fallback_reason=reason,
        )
        return DipoleHertzJetProviderResult(
            response=response,
            used_analytic_response=False,
            fallback_reason=reason,
            source_segment_index=segment_index,
            source_segment_fraction=segment_fraction,
            source_jet_residual=jet_residual,
        )

    source_results: list[
        DipoleHertzResponseJetResult | DipoleHertzSparseResponseJetResult
    ] = []
    for source_array_index, source in prepared.sources.items():
        if not center.valid_sources[source_array_index]:
            if source.worldline.ended_by_loss and source.worldline.time_ns.size:
                final_source_time = float(source.worldline.time_ns[-1])
                final_source_position = np.asarray(
                    source.worldline.position_mm[-1], dtype=float
                )
                final_separation = float(
                    np.linalg.norm(
                        np.asarray(observer_event.position_mm, dtype=float)
                        - final_source_position
                    )
                )
                termination_residual = (
                    C_MMNS * (float(observer_event.time_ns) - final_source_time)
                    - final_separation
                )
                reference_step = (
                    float(fallback_stencil_step_mm)
                    if fallback_stencil_step_mm is not None
                    else max(
                        float(fallback_minimum_step_mm),
                        float(fallback_relative_step) * final_separation,
                    )
                )
                if abs(termination_residual) <= 6.0 * reference_step:
                    return fallback(
                        "observer is inside the source-termination wavefront "
                        f"guard for source {source.identity!r}"
                    )
            continue
        root_time = float(center.retarded_time_ns[source_array_index])
        times = source.worldline.time_ns
        if times.size < 2:
            return fallback(
                f"source {source.identity!r} has no smooth two-knot segment"
            )
        selected_segment = int(np.searchsorted(times, root_time, side="right") - 1)
        selected_segment = min(max(selected_segment, 0), int(times.size) - 2)
        duration = float(source.worldline.segment_duration_ns[selected_segment])
        fraction = (root_time - float(times[selected_segment])) / duration
        segment_index[source_array_index] = selected_segment
        segment_fraction[source_array_index] = fraction
        if require_frozen_spin_segment and selected_segment == int(times.size) - 2:
            return fallback(
                "retarded root lies in the mutable final source-spin segment "
                f"for source {source.identity!r}"
            )
        if fraction <= boundary_guard or fraction >= 1.0 - boundary_guard:
            return fallback(
                "retarded root is inside the nonsmooth segment-boundary guard "
                f"for source {source.identity!r}: fraction={fraction:.17g}"
            )
        if response_kernel == "python":
            evaluator = quintic_dipole_hertz_response_jet_native
        elif response_kernel == "numba_strict_serial":
            evaluator = quintic_dipole_hertz_response_jet_numba_native
        else:
            evaluator = quintic_dipole_hertz_sparse_response_numba_native
        result = evaluator(
            observer_time_ns=float(observer_event.time_ns),
            observer_position_mm=observer_event.position_mm,
            magnetic_moment_native=source.magnetic_moment_native,
            segment_start_time_ns=float(times[selected_segment]),
            segment_duration_ns=duration,
            position_coefficients_mm=source.worldline.position_coefficients_mm[
                selected_segment
            ],
            rest_spin_start=source.rest_spin[selected_segment],
            rest_spin_end=source.rest_spin[selected_segment + 1],
            rest_spin_start_derivative_per_ns=source.rest_spin_derivative_per_ns[
                selected_segment
            ],
            rest_spin_end_derivative_per_ns=source.rest_spin_derivative_per_ns[
                selected_segment + 1
            ],
            preserved_rest_spin_magnitude=source.preserved_rest_spin_magnitude,
            retarded_time_ns=root_time,
        )
        jet_residual[source_array_index] = result.light_cone_jet_residual
        source_results.append(result)

    four_potential = np.zeros(4, dtype=float)
    if response_kernel == "numba_sparse_strict_serial":
        from .retarded_dipole_fields import RetardedDipoleResponseGradientResult

        packed_field = np.zeros(6, dtype=float)
        packed_partial_f = np.zeros((4, 6), dtype=float)
        for result in source_results:
            assert isinstance(result, DipoleHertzSparseResponseJetResult)
            four_potential += result.four_potential
            packed_field += result.antisymmetric_response
            packed_partial_f += result.partial_antisymmetric_response
        response = RetardedDipoleResponseGradientResult(
            four_potential=four_potential,
            antisymmetric_response=packed_field,
            partial_antisymmetric_response=packed_partial_f,
            root=center,
            used_analytic_response=True,
            fallback_reason=None,
            source_segment_index=segment_index.copy(),
            source_segment_fraction=segment_fraction.copy(),
            source_jet_residual=jet_residual.copy(),
        )
    else:
        partial_a = np.zeros((4, 4), dtype=float)
        field_tensor = np.zeros((4, 4), dtype=float)
        partial_f = np.zeros((4, 4, 4), dtype=float)
        for result in source_results:
            assert isinstance(result, DipoleHertzResponseJetResult)
            four_potential += result.four_potential
            partial_a += result.partial_a
            field_tensor += result.field_tensor
            partial_f += result.partial_f
        electric, magnetic = fields_from_tensor_native(field_tensor)
        response = RetardedDipoleFieldGradientResult(
            four_potential=four_potential,
            partial_a=partial_a,
            electric_field_native=electric,
            magnetic_field_native=magnetic,
            field_tensor=field_tensor,
            partial_f=partial_f,
            hertz=center,
            stencil_step_mm=0.0,
            stencil_offsets=np.zeros((1, 4), dtype=int),
            stencil_retarded_time_ns=center.retarded_time_ns[np.newaxis, :],
            stencil_light_cone_residual_mm=center.light_cone_residual_mm[np.newaxis, :],
            lorenz_gauge_residual_per_mm=float(np.trace(partial_a)),
        )
    from .analytic_dipole_hertz_diagnostics import (
        record_analytic_dipole_hertz_response,
    )

    finite_fraction = segment_fraction[np.isfinite(segment_fraction)]
    minimum_boundary_fraction = (
        float(np.min(np.minimum(finite_fraction, 1.0 - finite_fraction)))
        if finite_fraction.size
        else float("inf")
    )
    record_analytic_dipole_hertz_response(
        valid_sources=int(np.count_nonzero(center.valid_sources)),
        minimum_boundary_fraction=minimum_boundary_fraction,
        fallback_reason=None,
    )
    return DipoleHertzJetProviderResult(
        response=response,
        used_analytic_response=True,
        fallback_reason=None,
        source_segment_index=segment_index,
        source_segment_fraction=segment_fraction,
        source_jet_residual=jet_residual,
    )


__all__ = [
    "DipoleHertzJetProviderResult",
    "DipoleHertzResponseJetResult",
    "DipoleHertzSparseResponseJetResult",
    "evaluate_retarded_dipole_field_gradient_hertz_jet_native",
    "polynomial_dipole_hertz_response_jet_native",
    "quintic_dipole_hertz_response_jet_native",
    "quintic_dipole_hertz_response_jet_numba_native",
    "quintic_dipole_hertz_sparse_response_numba_native",
]
