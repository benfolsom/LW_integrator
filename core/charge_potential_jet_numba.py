"""Strict-float64 compiled kernel for the analytical charge potential jet."""

from __future__ import annotations

import numpy as np
from numba import njit

from .constants import C_MMNS

_JET_SIZE = 21
_GRADIENT_START = 1
_HESSIAN_START = 5


@njit(cache=True, fastmath=False)
def _constant(value: float) -> np.ndarray:
    result = np.zeros(_JET_SIZE, dtype=np.float64)
    result[0] = value
    return result


@njit(cache=True, fastmath=False)
def _variable(value: float, index: int) -> np.ndarray:
    result = _constant(value)
    result[_GRADIENT_START + index] = 1.0
    return result


@njit(cache=True, fastmath=False)
def _add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left + right


@njit(cache=True, fastmath=False)
def _subtract(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left - right


@njit(cache=True, fastmath=False)
def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(_JET_SIZE, dtype=np.float64)
    result[0] = left[0] * right[0]
    for row in range(4):
        left_gradient = left[_GRADIENT_START + row]
        right_gradient = right[_GRADIENT_START + row]
        result[_GRADIENT_START + row] = (
            right[0] * left_gradient + left[0] * right_gradient
        )
        for column in range(4):
            index = _HESSIAN_START + 4 * row + column
            result[index] = (
                right[0] * left[index]
                + left[0] * right[index]
                + left_gradient * right[_GRADIENT_START + column]
                + right_gradient * left[_GRADIENT_START + column]
            )
    return result


@njit(cache=True, fastmath=False)
def _reciprocal(value: np.ndarray) -> np.ndarray:
    result = np.empty(_JET_SIZE, dtype=np.float64)
    inverse = 1.0 / value[0]
    first = -(inverse * inverse)
    second = 2.0 * inverse * inverse * inverse
    result[0] = inverse
    for row in range(4):
        row_gradient = value[_GRADIENT_START + row]
        result[_GRADIENT_START + row] = first * row_gradient
        for column in range(4):
            index = _HESSIAN_START + 4 * row + column
            result[index] = (
                first * value[index]
                + second * row_gradient * value[_GRADIENT_START + column]
            )
    return result


@njit(cache=True, fastmath=False)
def _divide(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _multiply(left, _reciprocal(right))


@njit(cache=True, fastmath=False)
def _sqrt(value: np.ndarray) -> np.ndarray:
    result = np.empty(_JET_SIZE, dtype=np.float64)
    root = np.sqrt(value[0])
    first = 0.5 / root
    second = -0.25 / (value[0] * root)
    result[0] = root
    for row in range(4):
        row_gradient = value[_GRADIENT_START + row]
        result[_GRADIENT_START + row] = first * row_gradient
        for column in range(4):
            index = _HESSIAN_START + 4 * row + column
            result[index] = (
                first * value[index]
                + second * row_gradient * value[_GRADIENT_START + column]
            )
    return result


@njit(cache=True, fastmath=False)
def _polynomial(coefficients: np.ndarray, argument: np.ndarray) -> np.ndarray:
    result = _constant(0.0)
    for index in range(coefficients.size - 1, -1, -1):
        result = _add(_multiply(result, argument), _constant(coefficients[index]))
    return result


@njit(cache=True, fastmath=False)
def _dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = _constant(0.0)
    for index in range(left.shape[0]):
        result = _add(result, _multiply(left[index], right[index]))
    return result


@njit(cache=True, fastmath=False)
def _norm(vector: np.ndarray) -> np.ndarray:
    return _sqrt(_dot(vector, vector))


@njit(cache=True, fastmath=False)
def quintic_charge_potential_jet_strict_serial(
    observer_time_ns: float,
    observer_position_mm: np.ndarray,
    charge_native: float,
    segment_start_time_ns: float,
    segment_duration_ns: float,
    position_coefficients_mm: np.ndarray,
    retarded_time_ns: float,
    jet_newton_iterations: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    """Return ``A, dA, d2A`` using the strict compiled Taylor algebra."""

    observer_coordinates = np.empty((4, _JET_SIZE), dtype=np.float64)
    observer_coordinates[0] = _variable(C_MMNS * observer_time_ns, 0)
    for index in range(3):
        observer_coordinates[index + 1] = _variable(
            observer_position_mm[index], index + 1
        )
    source_coordinate = _constant(C_MMNS * retarded_time_ns)
    segment_start_coordinate = C_MMNS * segment_start_time_ns
    segment_duration_coordinate = C_MMNS * segment_duration_ns
    light_cone = _constant(0.0)

    for _ in range(jet_newton_iterations):
        normalized_time = _divide(
            _subtract(source_coordinate, _constant(segment_start_coordinate)),
            _constant(segment_duration_coordinate),
        )
        source_position = np.empty((3, _JET_SIZE), dtype=np.float64)
        source_beta = np.empty((3, _JET_SIZE), dtype=np.float64)
        for component in range(3):
            source_position[component] = _polynomial(
                position_coefficients_mm[:, component], normalized_time
            )
            beta_coefficients = np.empty(5, dtype=np.float64)
            for order in range(1, 6):
                beta_coefficients[order - 1] = (
                    order
                    * position_coefficients_mm[order, component]
                    / segment_duration_coordinate
                )
            source_beta[component] = _polynomial(beta_coefficients, normalized_time)
        separation_vector = np.empty((3, _JET_SIZE), dtype=np.float64)
        for component in range(3):
            separation_vector[component] = _subtract(
                observer_coordinates[component + 1], source_position[component]
            )
        separation = _norm(separation_vector)
        direction = np.empty((3, _JET_SIZE), dtype=np.float64)
        for component in range(3):
            direction[component] = _divide(separation_vector[component], separation)
        light_cone = _subtract(
            _subtract(observer_coordinates[0], source_coordinate), separation
        )
        derivative = _add(_constant(-1.0), _dot(direction, source_beta))
        light_cone[0] = 0.0
        source_coordinate = _subtract(
            source_coordinate, _divide(light_cone, derivative)
        )

    normalized_time = _divide(
        _subtract(source_coordinate, _constant(segment_start_coordinate)),
        _constant(segment_duration_coordinate),
    )
    source_position = np.empty((3, _JET_SIZE), dtype=np.float64)
    source_beta = np.empty((3, _JET_SIZE), dtype=np.float64)
    for component in range(3):
        source_position[component] = _polynomial(
            position_coefficients_mm[:, component], normalized_time
        )
        beta_coefficients = np.empty(5, dtype=np.float64)
        for order in range(1, 6):
            beta_coefficients[order - 1] = (
                order
                * position_coefficients_mm[order, component]
                / segment_duration_coordinate
            )
        source_beta[component] = _polynomial(beta_coefficients, normalized_time)
    separation_vector = np.empty((3, _JET_SIZE), dtype=np.float64)
    for component in range(3):
        separation_vector[component] = _subtract(
            observer_coordinates[component + 1], source_position[component]
        )
    separation = _norm(separation_vector)
    direction = np.empty((3, _JET_SIZE), dtype=np.float64)
    for component in range(3):
        direction[component] = _divide(separation_vector[component], separation)
    kappa = _subtract(_constant(1.0), _dot(direction, source_beta))
    scalar_potential = _divide(_constant(charge_native), _multiply(kappa, separation))
    potential = np.empty((4, _JET_SIZE), dtype=np.float64)
    potential[0] = scalar_potential
    for component in range(3):
        potential[component + 1] = _multiply(scalar_potential, source_beta[component])
    light_cone = _subtract(
        _subtract(observer_coordinates[0], source_coordinate), separation
    )

    four_potential = potential[:, 0].copy()
    partial_a = np.empty((4, 4), dtype=np.float64)
    partial2_a = np.empty((4, 4, 4), dtype=np.float64)
    for component in range(4):
        for derivative in range(4):
            partial_a[derivative, component] = potential[
                component, _GRADIENT_START + derivative
            ]
            for second_derivative in range(4):
                partial2_a[derivative, second_derivative, component] = potential[
                    component,
                    _HESSIAN_START + 4 * derivative + second_derivative,
                ]
    root_gradient = source_coordinate[_GRADIENT_START:_HESSIAN_START].copy()
    root_hessian = source_coordinate[_HESSIAN_START:].reshape((4, 4)).copy()
    residual = np.max(np.abs(light_cone))
    return (
        four_potential,
        partial_a,
        partial2_a,
        source_coordinate[0] / C_MMNS,
        root_gradient,
        root_hessian,
        residual,
    )


__all__ = ["quintic_charge_potential_jet_strict_serial"]
