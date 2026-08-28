"""Strict-float64 third-order Hertz-response jet kernel."""

from __future__ import annotations

from itertools import permutations, product
from math import factorial
from typing import cast

import numpy as np
from numba import njit  # type: ignore[import-untyped]

from .constants import C_MMNS

_DIMENSION = 4
_ORDER = 3


def _build_tables() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    multiindices: list[tuple[int, int, int, int]] = []
    for total_degree in range(_ORDER + 1):
        for alpha in product(range(total_degree + 1), repeat=_DIMENSION):
            if sum(alpha) == total_degree:
                multiindices.append(cast(tuple[int, int, int, int], alpha))
    index = {alpha: position for position, alpha in enumerate(multiindices)}
    maximum_splits = 8
    split_count = np.zeros(len(multiindices), dtype=np.int64)
    split_left = np.full((len(multiindices), maximum_splits), -1, dtype=np.int64)
    split_right = np.full((len(multiindices), maximum_splits), -1, dtype=np.int64)
    for result_index, alpha in enumerate(multiindices):
        splits = list(product(*(range(value + 1) for value in alpha)))
        split_count[result_index] = len(splits)
        for split_index, beta in enumerate(splits):
            beta_key = cast(tuple[int, int, int, int], beta)
            gamma = cast(
                tuple[int, int, int, int],
                tuple(alpha[axis] - beta[axis] for axis in range(_DIMENSION)),
            )
            split_left[result_index, split_index] = index[beta_key]
            split_right[result_index, split_index] = index[gamma]

    def derivative_slot(indices: tuple[int, ...]) -> tuple[int, float]:
        alpha = [0, 0, 0, 0]
        for derivative_index in indices:
            alpha[derivative_index] += 1
        multiindex = cast(tuple[int, int, int, int], tuple(alpha))
        scale = 1
        for multiplicity in multiindex:
            scale *= factorial(multiplicity)
        return index[multiindex], float(scale)

    derivative_index = np.empty((4, 4, 4), dtype=np.int64)
    derivative_scale = np.empty((4, 4, 4), dtype=np.float64)
    for first in range(4):
        for second in range(4):
            for third in range(4):
                slot, scale = derivative_slot((first, second, third))
                derivative_index[first, second, third] = slot
                derivative_scale[first, second, third] = scale
    first_index = np.empty(4, dtype=np.int64)
    second_index = np.empty((4, 4), dtype=np.int64)
    second_scale = np.empty((4, 4), dtype=np.float64)
    for first in range(4):
        first_index[first] = derivative_slot((first,))[0]
        for second in range(4):
            slot, scale = derivative_slot((first, second))
            second_index[first, second] = slot
            second_scale[first, second] = scale
    return (
        split_count,
        split_left,
        split_right,
        first_index,
        second_index,
        second_scale,
        derivative_index,
        derivative_scale,
    )


(
    _SPLIT_COUNT,
    _SPLIT_LEFT,
    _SPLIT_RIGHT,
    _FIRST_INDEX,
    _SECOND_INDEX,
    _SECOND_SCALE,
    _THIRD_INDEX,
    _THIRD_SCALE,
) = _build_tables()
_JET_SIZE = int(_SPLIT_COUNT.size)
_METRIC_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0))


def _levi_civita_upper() -> np.ndarray:
    tensor = np.zeros((4, 4, 4, 4), dtype=np.float64)
    for permutation in permutations(range(4)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(4)
            for right in range(left + 1, 4)
        )
        tensor[permutation] = -1.0 if inversions % 2 == 0 else 1.0
    return tensor


_LEVI_CIVITA_UPPER = _levi_civita_upper()


@njit(cache=True, fastmath=False, inline="always")
def _constant(value: float) -> np.ndarray:
    result = np.zeros(_JET_SIZE, dtype=np.float64)
    result[0] = value
    return result


@njit(cache=True, fastmath=False, inline="always")
def _variable(value: float, index: int) -> np.ndarray:
    result = _constant(value)
    result[_FIRST_INDEX[index]] = 1.0
    return result  # type: ignore[no-any-return]


@njit(cache=True, fastmath=False, inline="always")
def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(_JET_SIZE, dtype=np.float64)
    for result_index in range(_JET_SIZE):
        total = 0.0
        for split_index in range(_SPLIT_COUNT[result_index]):
            total += (
                left[_SPLIT_LEFT[result_index, split_index]]
                * right[_SPLIT_RIGHT[result_index, split_index]]
            )
        result[result_index] = total
    return result


@njit(cache=True, fastmath=False, inline="always")
def _reciprocal(value: np.ndarray) -> np.ndarray:
    result = np.zeros(_JET_SIZE, dtype=np.float64)
    result[0] = 1.0 / value[0]
    for result_index in range(1, _JET_SIZE):
        total = 0.0
        for split_index in range(_SPLIT_COUNT[result_index]):
            left_index = _SPLIT_LEFT[result_index, split_index]
            if left_index == 0:
                continue
            right_index = _SPLIT_RIGHT[result_index, split_index]
            total += value[left_index] * result[right_index]
        result[result_index] = -total / value[0]
    return result


@njit(cache=True, fastmath=False, inline="always")
def _divide(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _multiply(left, _reciprocal(right))  # type: ignore[no-any-return]


@njit(cache=True, fastmath=False, inline="always")
def _sqrt(value: np.ndarray) -> np.ndarray:
    result = _constant(np.sqrt(value[0]))
    for _ in range(3):
        result = 0.5 * (result + _divide(value, result))
    return result  # type: ignore[no-any-return]


@njit(cache=True, fastmath=False, inline="always")
def _polynomial(coefficients: np.ndarray, argument: np.ndarray) -> np.ndarray:
    result = _constant(0.0)
    for index in range(coefficients.size - 1, -1, -1):
        result = _multiply(result, argument)
        result[0] += coefficients[index]
    return result  # type: ignore[no-any-return]


@njit(cache=True, fastmath=False, inline="always")
def _dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = _constant(0.0)
    for index in range(left.shape[0]):
        result += _multiply(left[index], right[index])
    return result  # type: ignore[no-any-return]


@njit(cache=True, fastmath=False, inline="always")
def _source_state(
    source_coordinate: np.ndarray,
    start_coordinate: float,
    duration_coordinate: float,
    position_coefficients_mm: np.ndarray,
    spin_coefficients: np.ndarray,
    preserve_magnitude: bool,
    preserved_magnitude: float,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    normalized_time = (
        source_coordinate - _constant(start_coordinate)
    ) / duration_coordinate
    source_position = np.empty((3, _JET_SIZE), dtype=np.float64)
    source_beta = np.empty((3, _JET_SIZE), dtype=np.float64)
    source_spin = np.empty((3, _JET_SIZE), dtype=np.float64)
    beta_coefficients = np.empty(5, dtype=np.float64)
    for component in range(3):
        source_position[component] = _polynomial(
            position_coefficients_mm[:, component], normalized_time
        )
        for order in range(1, 6):
            beta_coefficients[order - 1] = (
                order * position_coefficients_mm[order, component] / duration_coordinate
            )
        source_beta[component] = _polynomial(beta_coefficients, normalized_time)
        source_spin[component] = _polynomial(
            spin_coefficients[:, component], normalized_time
        )
    if preserve_magnitude:
        if preserved_magnitude == 0.0:
            source_spin[:, :] = 0.0
        else:
            magnitude = _sqrt(_dot(source_spin, source_spin))
            if magnitude[0] <= 1.0e-15:
                return 1, source_position, source_beta, source_spin
            scale = _divide(_constant(preserved_magnitude), magnitude)
            for component in range(3):
                source_spin[component] = _multiply(source_spin[component], scale)
    return 0, source_position, source_beta, source_spin


@njit(cache=True, fastmath=False)
def quintic_dipole_hertz_response_coefficients_strict_serial(
    observer_time_ns: float,
    observer_position_mm: np.ndarray,
    magnetic_moment_native: float,
    segment_start_time_ns: float,
    segment_duration_ns: float,
    position_coefficients_mm: np.ndarray,
    spin_coefficients: np.ndarray,
    preserve_magnitude: bool,
    preserved_magnitude: float,
    retarded_time_ns: float,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Return status, H, A, partial-A, F, partial-F, root derivatives, residual."""

    observer = np.empty((4, _JET_SIZE), dtype=np.float64)
    observer[0] = _variable(C_MMNS * observer_time_ns, 0)
    for component in range(3):
        observer[component + 1] = _variable(
            observer_position_mm[component], component + 1
        )
    source_coordinate = _constant(C_MMNS * retarded_time_ns)
    start_coordinate = C_MMNS * segment_start_time_ns
    duration_coordinate = C_MMNS * segment_duration_ns

    light_cone = _constant(0.0)
    for _ in range(4):
        status, source_position, source_beta, _ = _source_state(
            source_coordinate,
            start_coordinate,
            duration_coordinate,
            position_coefficients_mm,
            spin_coefficients,
            preserve_magnitude,
            preserved_magnitude,
        )
        if status != 0:
            empty4 = np.zeros(4, dtype=np.float64)
            empty44 = np.zeros((4, 4), dtype=np.float64)
            empty444 = np.zeros((4, 4, 4), dtype=np.float64)
            return (
                status,
                empty44,
                empty4,
                empty44,
                empty44,
                empty444,
                empty4,
                empty44,
                empty444,
                np.nan,
            )
        separation_vector = np.empty((3, _JET_SIZE), dtype=np.float64)
        for component in range(3):
            separation_vector[component] = (
                observer[component + 1] - source_position[component]
            )
        separation = _sqrt(_dot(separation_vector, separation_vector))
        direction = np.empty((3, _JET_SIZE), dtype=np.float64)
        for component in range(3):
            direction[component] = _divide(separation_vector[component], separation)
        light_cone = observer[0] - source_coordinate - separation
        light_cone[0] = 0.0
        derivative = _constant(-1.0) + _dot(direction, source_beta)
        source_coordinate = source_coordinate - _divide(light_cone, derivative)

    status, source_position, source_beta, source_spin = _source_state(
        source_coordinate,
        start_coordinate,
        duration_coordinate,
        position_coefficients_mm,
        spin_coefficients,
        preserve_magnitude,
        preserved_magnitude,
    )
    separation_vector = np.empty((3, _JET_SIZE), dtype=np.float64)
    for component in range(3):
        separation_vector[component] = (
            observer[component + 1] - source_position[component]
        )
    separation = _sqrt(_dot(separation_vector, separation_vector))
    direction = np.empty((3, _JET_SIZE), dtype=np.float64)
    for component in range(3):
        direction[component] = _divide(separation_vector[component], separation)
    kappa = _constant(1.0) - _dot(direction, source_beta)
    if kappa[0] <= 1.0e-14:
        status = 2
    beta_squared = _dot(source_beta, source_beta)
    if beta_squared[0] >= 1.0:
        status = 3
    if status != 0:
        empty4 = np.zeros(4, dtype=np.float64)
        empty44 = np.zeros((4, 4), dtype=np.float64)
        empty444 = np.zeros((4, 4, 4), dtype=np.float64)
        return (
            status,
            empty44,
            empty4,
            empty44,
            empty44,
            empty444,
            empty4,
            empty44,
            empty444,
            np.nan,
        )
    gamma = _reciprocal(_sqrt(_constant(1.0) - beta_squared))
    projection = _dot(source_beta, source_spin)
    boost_coefficient = _multiply(
        _divide(_multiply(gamma, gamma), gamma + _constant(1.0)), projection
    )
    moment_four = np.empty((4, _JET_SIZE), dtype=np.float64)
    moment_four[0] = magnetic_moment_native * _multiply(gamma, projection)
    for component in range(3):
        moment_four[component + 1] = magnetic_moment_native * (
            source_spin[component]
            + _multiply(boost_coefficient, source_beta[component])
        )
    four_velocity = np.empty((4, _JET_SIZE), dtype=np.float64)
    four_velocity[0] = C_MMNS * gamma
    for component in range(3):
        four_velocity[component + 1] = C_MMNS * _multiply(gamma, source_beta[component])
    wedge = np.empty((4, 4, _JET_SIZE), dtype=np.float64)
    for mu in range(4):
        for nu in range(4):
            wedge[mu, nu] = (
                _multiply(four_velocity[mu], moment_four[nu])
                - _multiply(moment_four[mu], four_velocity[nu])
            ) / C_MMNS
    hertz_jet = np.empty((4, 4, _JET_SIZE), dtype=np.float64)
    invariant_distance = _multiply(_multiply(gamma, separation), kappa)
    for mu in range(4):
        for nu in range(4):
            dual = _constant(0.0)
            for alpha in range(4):
                for beta in range(4):
                    coefficient = (
                        0.5
                        * _LEVI_CIVITA_UPPER[mu, nu, alpha, beta]
                        * _METRIC_SIGNS[alpha]
                        * _METRIC_SIGNS[beta]
                    )
                    if coefficient != 0.0:
                        dual += coefficient * wedge[alpha, beta]
            hertz_jet[mu, nu] = _divide(dual, invariant_distance)

    hertz = hertz_jet[:, :, 0].copy()
    potential = np.zeros(4, dtype=np.float64)
    partial_a = np.zeros((4, 4), dtype=np.float64)
    field = np.zeros((4, 4), dtype=np.float64)
    partial_f = np.zeros((4, 4, 4), dtype=np.float64)
    for mu in range(4):
        for rho in range(4):
            potential[mu] += hertz_jet[mu, rho, _FIRST_INDEX[rho]]
    for derivative_index in range(4):
        for mu in range(4):
            for rho in range(4):
                partial_a[derivative_index, mu] += (
                    hertz_jet[mu, rho, _SECOND_INDEX[derivative_index, rho]]
                    * _SECOND_SCALE[derivative_index, rho]
                )
    for mu in range(4):
        for nu in range(4):
            field[mu, nu] = (
                _METRIC_SIGNS[mu] * partial_a[mu, nu]
                - _METRIC_SIGNS[nu] * partial_a[nu, mu]
            )
            for derivative_index in range(4):
                first = 0.0
                second = 0.0
                for rho in range(4):
                    first += (
                        hertz_jet[
                            nu,
                            rho,
                            _THIRD_INDEX[derivative_index, mu, rho],
                        ]
                        * _THIRD_SCALE[derivative_index, mu, rho]
                    )
                    second += (
                        hertz_jet[
                            mu,
                            rho,
                            _THIRD_INDEX[derivative_index, nu, rho],
                        ]
                        * _THIRD_SCALE[derivative_index, nu, rho]
                    )
                partial_f[derivative_index, mu, nu] = (
                    _METRIC_SIGNS[mu] * first - _METRIC_SIGNS[nu] * second
                )

    root_gradient = np.empty(4, dtype=np.float64)
    root_hessian = np.empty((4, 4), dtype=np.float64)
    root_third = np.empty((4, 4, 4), dtype=np.float64)
    for first in range(4):
        root_gradient[first] = source_coordinate[_FIRST_INDEX[first]]
        for second in range(4):
            root_hessian[first, second] = (
                source_coordinate[_SECOND_INDEX[first, second]]
                * _SECOND_SCALE[first, second]
            )
            for third in range(4):
                root_third[first, second, third] = (
                    source_coordinate[_THIRD_INDEX[first, second, third]]
                    * _THIRD_SCALE[first, second, third]
                )
    final_light_cone = observer[0] - source_coordinate - separation
    residual = 0.0
    for coefficient in final_light_cone:
        residual = max(residual, abs(coefficient))
    return (
        0,
        hertz,
        potential,
        partial_a,
        field,
        partial_f,
        root_gradient,
        root_hessian,
        root_third,
        residual,
    )


__all__ = ["quintic_dipole_hertz_response_coefficients_strict_serial"]
