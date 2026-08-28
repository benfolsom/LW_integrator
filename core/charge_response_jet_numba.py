"""Strict-float64 compiled charge-response jet prototype."""

from __future__ import annotations

import numpy as np
from numba import njit

from .constants import C_MMNS
from .retarded_dipole_numba_roots import _STATUS_VALID, _solve_retarded_sample

_SIZE = 5
_STATUS_CHARGE_ZERO_SEPARATION = 20
_STATUS_CHARGE_SUPERLUMINAL_SOURCE = 21
_STATUS_CHARGE_SINGULAR_KAPPA = 22


@njit(cache=True, fastmath=False)
def _constant(value: float) -> np.ndarray:
    result = np.zeros(_SIZE, dtype=np.float64)
    result[0] = value
    return result


@njit(cache=True, fastmath=False)
def _variable(value: float, index: int) -> np.ndarray:
    result = _constant(value)
    result[index + 1] = 1.0
    return result


@njit(cache=True, fastmath=False)
def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(_SIZE, dtype=np.float64)
    result[0] = left[0] * right[0]
    for index in range(4):
        result[index + 1] = right[0] * left[index + 1] + left[0] * right[index + 1]
    return result


@njit(cache=True, fastmath=False)
def _reciprocal(value: np.ndarray) -> np.ndarray:
    result = np.empty(_SIZE, dtype=np.float64)
    inverse = 1.0 / value[0]
    result[0] = inverse
    factor = -(inverse * inverse)
    for index in range(4):
        result[index + 1] = factor * value[index + 1]
    return result


@njit(cache=True, fastmath=False)
def _divide(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _multiply(left, _reciprocal(right))


@njit(cache=True, fastmath=False)
def _sqrt(value: np.ndarray) -> np.ndarray:
    result = np.empty(_SIZE, dtype=np.float64)
    root = np.sqrt(value[0])
    result[0] = root
    for index in range(4):
        result[index + 1] = 0.5 * value[index + 1] / root
    return result


@njit(cache=True, fastmath=False)
def _polynomial(coefficients: np.ndarray, argument: np.ndarray) -> np.ndarray:
    result = _constant(0.0)
    for index in range(coefficients.size - 1, -1, -1):
        result = _multiply(result, argument)
        result[0] += coefficients[index]
    return result


@njit(cache=True, fastmath=False)
def _dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = _constant(0.0)
    for index in range(left.shape[0]):
        product = _multiply(left[index], right[index])
        result += product
    return result


@njit(cache=True, fastmath=False)
def _cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty((3, _SIZE), dtype=np.float64)
    result[0] = _multiply(left[1], right[2]) - _multiply(left[2], right[1])
    result[1] = _multiply(left[2], right[0]) - _multiply(left[0], right[2])
    result[2] = _multiply(left[0], right[1]) - _multiply(left[1], right[0])
    return result


@njit(cache=True, fastmath=False)
def quintic_charge_response_coefficients_strict_serial(
    observer_time_ns: float,
    observer_position_mm: np.ndarray,
    charge_native: float,
    segment_start_time_ns: float,
    segment_duration_ns: float,
    position_coefficients_mm: np.ndarray,
    retarded_time_ns: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """Return ``A, partial_A, packed_F, partial_packed_F, kappa, residual``."""

    duration_coordinate = C_MMNS * segment_duration_ns
    root_coordinate = C_MMNS * retarded_time_ns
    start_coordinate = C_MMNS * segment_start_time_ns
    normalized_root = (root_coordinate - start_coordinate) / duration_coordinate
    source_position_value = np.zeros(3, dtype=np.float64)
    source_beta_value = np.zeros(3, dtype=np.float64)
    source_beta_prime_value = np.zeros(3, dtype=np.float64)
    for component in range(3):
        for order in range(6):
            source_position_value[component] += (
                position_coefficients_mm[order, component] * normalized_root**order
            )
        for order in range(1, 6):
            source_beta_value[component] += (
                order
                * position_coefficients_mm[order, component]
                * normalized_root ** (order - 1)
                / duration_coordinate
            )
        for order in range(2, 6):
            source_beta_prime_value[component] += (
                order
                * (order - 1)
                * position_coefficients_mm[order, component]
                * normalized_root ** (order - 2)
                / (duration_coordinate * duration_coordinate)
            )
    separation_value = observer_position_mm - source_position_value
    radius_value = np.sqrt(np.dot(separation_value, separation_value))
    direction_value = separation_value / radius_value
    kappa_value = 1.0 - np.dot(direction_value, source_beta_value)
    root_gradient = np.empty(4, dtype=np.float64)
    root_gradient[0] = 1.0 / kappa_value
    for index in range(3):
        root_gradient[index + 1] = -direction_value[index] / kappa_value

    observer = np.empty((4, _SIZE), dtype=np.float64)
    observer[0] = _variable(C_MMNS * observer_time_ns, 0)
    for index in range(3):
        observer[index + 1] = _variable(observer_position_mm[index], index + 1)
    source_coordinate = _constant(root_coordinate)
    source_coordinate[1:] = root_gradient
    normalized_time = _divide(
        source_coordinate - _constant(start_coordinate),
        _constant(duration_coordinate),
    )
    source_position = np.empty((3, _SIZE), dtype=np.float64)
    source_beta = np.empty((3, _SIZE), dtype=np.float64)
    source_beta_prime = np.empty((3, _SIZE), dtype=np.float64)
    for component in range(3):
        source_position[component] = _polynomial(
            position_coefficients_mm[:, component], normalized_time
        )
        beta_coefficients = np.empty(5, dtype=np.float64)
        beta_prime_coefficients = np.empty(4, dtype=np.float64)
        for order in range(1, 6):
            beta_coefficients[order - 1] = (
                order * position_coefficients_mm[order, component] / duration_coordinate
            )
        for order in range(2, 6):
            beta_prime_coefficients[order - 2] = (
                order
                * (order - 1)
                * position_coefficients_mm[order, component]
                / (duration_coordinate * duration_coordinate)
            )
        source_beta[component] = _polynomial(beta_coefficients, normalized_time)
        source_beta_prime[component] = _polynomial(
            beta_prime_coefficients, normalized_time
        )
    separation = np.empty((3, _SIZE), dtype=np.float64)
    for index in range(3):
        separation[index] = observer[index + 1] - source_position[index]
    radius = _sqrt(_dot(separation, separation))
    direction = np.empty((3, _SIZE), dtype=np.float64)
    for index in range(3):
        direction[index] = _divide(separation[index], radius)
    kappa = _constant(1.0) - _dot(direction, source_beta)
    beta_squared = _dot(source_beta, source_beta)
    direction_minus_beta = direction - source_beta
    velocity_prefactor = _divide(
        _constant(1.0) - beta_squared,
        _multiply(_multiply(_multiply(kappa, kappa), kappa), _multiply(radius, radius)),
    )
    radiation_prefactor = _reciprocal(
        _multiply(_multiply(_multiply(kappa, kappa), kappa), radius)
    )
    radiation_cross = _cross(direction, _cross(direction_minus_beta, source_beta_prime))
    electric = np.empty((3, _SIZE), dtype=np.float64)
    for index in range(3):
        electric[index] = charge_native * (
            _multiply(velocity_prefactor, direction_minus_beta[index])
            + _multiply(radiation_prefactor, radiation_cross[index])
        )
    magnetic = _cross(direction, electric)

    beta_squared_value = np.dot(source_beta_value, source_beta_value)
    velocity_value = (
        (1.0 - beta_squared_value)
        * (direction_value - source_beta_value)
        / (kappa_value**3 * radius_value**2)
    )
    radiation_value = np.cross(
        direction_value,
        np.cross(direction_value - source_beta_value, source_beta_prime_value),
    ) / (kappa_value**3 * radius_value)
    electric_value = charge_native * (velocity_value + radiation_value)
    magnetic_value = np.cross(direction_value, electric_value)
    ex, ey, ez = electric_value
    bx, by, bz = magnetic_value
    packed = np.empty(6, dtype=np.float64)
    packed[0] = -ex
    packed[1] = -ey
    packed[2] = -ez
    packed[3] = -bz
    packed[4] = by
    packed[5] = -bx
    partial_packed = np.empty((4, 6), dtype=np.float64)
    for derivative in range(4):
        exd = electric[0, derivative + 1]
        eyd = electric[1, derivative + 1]
        ezd = electric[2, derivative + 1]
        bxd = magnetic[0, derivative + 1]
        byd = magnetic[1, derivative + 1]
        bzd = magnetic[2, derivative + 1]
        partial_packed[derivative, 0] = -exd
        partial_packed[derivative, 1] = -eyd
        partial_packed[derivative, 2] = -ezd
        partial_packed[derivative, 3] = -bzd
        partial_packed[derivative, 4] = byd
        partial_packed[derivative, 5] = -bxd

    scalar_potential_jet = _divide(
        _constant(charge_native), _multiply(kappa, radius)
    )
    potential_jet = np.empty((4, _SIZE), dtype=np.float64)
    potential_jet[0] = scalar_potential_jet
    for component in range(3):
        potential_jet[component + 1] = _multiply(
            scalar_potential_jet, source_beta[component]
        )
    partial_a = np.empty((4, 4), dtype=np.float64)
    for derivative in range(4):
        for component in range(4):
            partial_a[derivative, component] = potential_jet[component, derivative + 1]

    scalar_potential = charge_native / (kappa_value * radius_value)
    potential = np.empty(4, dtype=np.float64)
    potential[0] = scalar_potential
    potential[1:] = scalar_potential * source_beta_value
    residual = C_MMNS * (observer_time_ns - retarded_time_ns) - radius_value
    return potential, partial_a, packed, partial_packed, kappa_value, residual


@njit(cache=True, fastmath=False)
def _materialize_response_tensors(
    packed: np.ndarray,
    partial_packed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    field = np.zeros((4, 4), dtype=np.float64)
    partial_f = np.zeros((4, 4, 4), dtype=np.float64)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for pair_index in range(6):
        first, second = pairs[pair_index]
        field[first, second] = packed[pair_index]
        field[second, first] = -packed[pair_index]
        for derivative in range(4):
            partial_f[derivative, first, second] = partial_packed[
                derivative, pair_index
            ]
            partial_f[derivative, second, first] = -partial_packed[
                derivative, pair_index
            ]
    return field, partial_f


@njit(cache=True, fastmath=False)
def quintic_charge_response_jet_strict_serial(
    observer_time_ns: float,
    observer_position_mm: np.ndarray,
    charge_native: float,
    segment_start_time_ns: float,
    segment_duration_ns: float,
    position_coefficients_mm: np.ndarray,
    retarded_time_ns: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compatibility oracle returning materialized ``A, F, partial_F``."""

    potential, _partial_a, packed, partial_packed, kappa, residual = (
        quintic_charge_response_coefficients_strict_serial(
            observer_time_ns,
            observer_position_mm,
            charge_native,
            segment_start_time_ns,
            segment_duration_ns,
            position_coefficients_mm,
            retarded_time_ns,
        )
    )
    field, partial_f = _materialize_response_tensors(packed, partial_packed)
    return potential, field, partial_f, kappa, residual


@njit(cache=True, fastmath=False)
def evaluate_charge_response_coefficients_one_event_strict_serial(
    time_ns: np.ndarray,
    position_mm: np.ndarray,
    segment_duration_ns: np.ndarray,
    position_coefficients_mm: np.ndarray,
    charge_native: float,
    ended_by_loss: bool,
    observer_time_ns: float,
    observer_position_mm: np.ndarray,
    root_tolerance_mm: float,
    max_root_iterations: int,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    int,
]:
    """Fuse one root and one analytical response-coefficient evaluation."""

    (
        status,
        source_time_ns,
        source_x_mm,
        source_y_mm,
        source_z_mm,
        source_beta_x,
        source_beta_y,
        source_beta_z,
        _source_beta_prime_x,
        _source_beta_prime_y,
        _source_beta_prime_z,
        source_residual_mm,
        source_separation_mm,
    ) = _solve_retarded_sample(
        time_ns,
        position_mm,
        segment_duration_ns,
        position_coefficients_mm,
        observer_time_ns,
        observer_position_mm[0],
        observer_position_mm[1],
        observer_position_mm[2],
        root_tolerance_mm,
        max_root_iterations,
        ended_by_loss,
    )
    if status != _STATUS_VALID:
        return (
            status,
            np.zeros(4, dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            np.zeros((4, 6), dtype=np.float64),
            np.nan,
            source_time_ns,
            source_residual_mm,
            source_separation_mm,
            -1,
        )
    if source_separation_mm <= 0.0:
        status = _STATUS_CHARGE_ZERO_SEPARATION
    beta_squared = (
        source_beta_x * source_beta_x
        + source_beta_y * source_beta_y
        + source_beta_z * source_beta_z
    )
    if status == _STATUS_VALID and beta_squared >= 1.0:
        status = _STATUS_CHARGE_SUPERLUMINAL_SOURCE
    if status == _STATUS_VALID:
        direction_x = (observer_position_mm[0] - source_x_mm) / source_separation_mm
        direction_y = (observer_position_mm[1] - source_y_mm) / source_separation_mm
        direction_z = (observer_position_mm[2] - source_z_mm) / source_separation_mm
        kappa_at_root = 1.0 - (
            direction_x * source_beta_x
            + direction_y * source_beta_y
            + direction_z * source_beta_z
        )
        if kappa_at_root <= 1.0e-14:
            status = _STATUS_CHARGE_SINGULAR_KAPPA
    if status != _STATUS_VALID:
        return (
            status,
            np.zeros(4, dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            np.zeros((4, 6), dtype=np.float64),
            np.nan,
            source_time_ns,
            source_residual_mm,
            source_separation_mm,
            -1,
        )
    lower = 0
    upper = time_ns.size
    while lower < upper:
        middle = lower + (upper - lower) // 2
        if time_ns[middle] <= source_time_ns:
            lower = middle + 1
        else:
            upper = middle
    segment = lower - 1
    if segment < 0:
        segment = 0
    if segment >= segment_duration_ns.size:
        segment = segment_duration_ns.size - 1
    potential, partial_a, packed, partial_packed, kappa, _ = (
        quintic_charge_response_coefficients_strict_serial(
            observer_time_ns,
            observer_position_mm,
            charge_native,
            time_ns[segment],
            segment_duration_ns[segment],
            position_coefficients_mm[segment],
            source_time_ns,
        )
    )
    return (
        status,
        potential,
        partial_a,
        packed,
        partial_packed,
        kappa,
        source_time_ns,
        source_residual_mm,
        source_separation_mm,
        segment,
    )


@njit(cache=True, fastmath=False)
def evaluate_charge_response_jet_one_event_strict_serial(
    time_ns: np.ndarray,
    position_mm: np.ndarray,
    segment_duration_ns: np.ndarray,
    position_coefficients_mm: np.ndarray,
    charge_native: float,
    ended_by_loss: bool,
    observer_time_ns: float,
    observer_position_mm: np.ndarray,
    root_tolerance_mm: float,
    max_root_iterations: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Compatibility wrapper materializing the validation tensors."""

    result = evaluate_charge_response_coefficients_one_event_strict_serial(
        time_ns,
        position_mm,
        segment_duration_ns,
        position_coefficients_mm,
        charge_native,
        ended_by_loss,
        observer_time_ns,
        observer_position_mm,
        root_tolerance_mm,
        max_root_iterations,
    )
    field, partial_f = _materialize_response_tensors(result[3], result[4])
    return result[0], result[1], field, partial_f, result[5], result[7], result[8]


__all__ = [
    "evaluate_charge_response_coefficients_one_event_strict_serial",
    "evaluate_charge_response_jet_one_event_strict_serial",
    "quintic_charge_response_coefficients_strict_serial",
    "quintic_charge_response_jet_strict_serial",
]
