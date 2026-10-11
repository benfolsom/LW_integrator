"""Local precise charge kinematics, independent of stored worldline segments.

All derivatives use x=(ct,x,y,z). Acceleration and jerk are d beta/d(ct)
and d2 beta/d(ct)2 at the selected root. A precise input must describe that
same event; it does not change the quintic history representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence, TypeVar
import numpy as np
from numba import njit  # type: ignore[import-untyped]

if TYPE_CHECKING:
    _Function = TypeVar("_Function", bound=Callable)

    def register_jitable(function: _Function) -> _Function:
        return function

else:
    from numba.extending import register_jitable

from .constants import C_MMNS

_JET_SIZE = 21
_GRADIENT_START = 1
_HESSIAN_START = 5


@register_jitable
def _constant(value: float) -> np.ndarray:
    result: np.ndarray = np.zeros(_JET_SIZE, dtype=np.float64)
    result[0] = value
    return result


@register_jitable
def _variable(value: float, index: int) -> np.ndarray:
    result: np.ndarray = _constant(value)
    result[_GRADIENT_START + index] = 1.0
    return result


@register_jitable
def _add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result: np.ndarray = left + right
    return result


@register_jitable
def _subtract(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result: np.ndarray = left - right
    return result


@register_jitable
def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.empty(_JET_SIZE, dtype=np.float64)
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


@register_jitable
def _reciprocal(value: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.empty(_JET_SIZE, dtype=np.float64)
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


@register_jitable
def _divide(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _multiply(left, _reciprocal(right))


@register_jitable
def _sqrt(value: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.empty(_JET_SIZE, dtype=np.float64)
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


@register_jitable
def _polynomial(coefficients: np.ndarray, argument: np.ndarray) -> np.ndarray:
    result: np.ndarray = _constant(0.0)
    for index in range(coefficients.size - 1, -1, -1):
        result = _add(_multiply(result, argument), _constant(coefficients[index]))
    return result


@register_jitable
def _dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result: np.ndarray = _constant(0.0)
    for index in range(left.shape[0]):
        result = _add(result, _multiply(left[index], right[index]))
    return result


@register_jitable
def _norm(vector: np.ndarray) -> np.ndarray:
    return _sqrt(_dot(vector, vector))


@register_jitable
def precise_speed(proper_velocity: np.ndarray) -> tuple[float, np.ndarray]:
    """Return deficit and direction without subtracting from unit speed."""
    magnitude = np.sqrt(np.dot(proper_velocity, proper_velocity))
    gamma = np.sqrt(1.0 + magnitude * magnitude)
    direction = np.zeros(3)
    if magnitude == 0.0:
        direction[0] = 1.0
    else:
        direction = proper_velocity / magnitude
    return (1.0 / gamma) / (gamma + magnitude), direction


def validated_precise_velocity(value: Sequence[float] | np.ndarray) -> np.ndarray:
    result: np.ndarray = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("source_proper_velocity must contain three finite values")
    return result


@register_jitable
def resolved_separation_frame(
    longitudinal_mm: float,
    transverse_vector_mm: np.ndarray,
    proper_velocity: np.ndarray,
) -> np.ndarray:
    """Project an independently resolved transverse displacement.

    Analytic roots must keep these parts separate: recombining them into a
    large lab vector before projection discards the small transverse part.
    """
    _deficit, direction = precise_speed(proper_velocity)
    frame = _frame(direction)
    return np.asarray(
        (
            longitudinal_mm,
            np.dot(frame[1], transverse_vector_mm),
            np.dot(frame[2], transverse_vector_mm),
        )
    )


@register_jitable
def _frame(direction: np.ndarray) -> np.ndarray:
    axis = np.zeros(3)
    axis[np.argmin(np.abs(direction))] = 1.0
    transverse = np.cross(direction, axis)
    transverse /= np.sqrt(np.dot(transverse, transverse))
    frame = np.empty((3, 3))
    frame[0] = direction
    frame[1] = transverse
    frame[2] = np.cross(direction, transverse)
    return frame


@register_jitable
def _product_roundoff(left: float, right: float, product: float) -> float:
    # Dekker splitting: retain the low part of each binary64 product.
    splitter = 134217729.0
    split_left = splitter * left
    high_left = split_left - (split_left - left)
    low_left = left - high_left
    split_right = splitter * right
    high_right = split_right - (split_right - right)
    low_right = right - high_right
    return (
        (high_left * high_right - product)
        + high_left * low_right
        + low_left * high_right
    ) + low_left * low_right


@register_jitable
def _determinant(a: float, b: float, c: float, d: float) -> float:
    first = a * b
    second = c * d
    difference = first - second
    virtual = difference - first
    subtraction_roundoff = (first - (difference - virtual)) + (-second - virtual)
    return difference + (
        subtraction_roundoff
        + _product_roundoff(a, b, first)
        - _product_roundoff(c, d, second)
    )


@register_jitable
def _project_vector(
    vector: np.ndarray,
    proper_velocity: np.ndarray,
    frame: np.ndarray,
    magnitude: float,
) -> np.ndarray:
    result: np.ndarray = np.empty(3)
    result[0] = np.dot(frame[0], vector)
    if magnitude == 0.0:
        result[1] = np.dot(frame[1], vector)
        result[2] = np.dot(frame[2], vector)
        return result
    # Project through u cross vector, rather than a dot product with a
    # rounded transverse basis. Parallel binary64 vectors then give exactly
    # zero, and small physical angles retain the low product differences.
    crossed = np.empty(3)
    crossed[0] = (
        _determinant(proper_velocity[1], vector[2], proper_velocity[2], vector[1])
        / magnitude
    )
    crossed[1] = (
        _determinant(proper_velocity[2], vector[0], proper_velocity[0], vector[2])
        / magnitude
    )
    crossed[2] = (
        _determinant(proper_velocity[0], vector[1], proper_velocity[1], vector[0])
        / magnitude
    )
    result[1] = np.dot(frame[2], crossed)
    result[2] = -np.dot(frame[1], crossed)
    return result


@register_jitable
def _geometry(
    observer: np.ndarray,
    shift: np.ndarray,
    separation: np.ndarray,
    frame: np.ndarray,
    deficit: float,
    acceleration: np.ndarray,
    jerk: np.ndarray,
    snap: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Evaluate local jets in a fixed frame aligned with the root velocity."""
    shift2 = _multiply(shift, shift)
    shift3 = _multiply(shift2, shift)
    r = np.empty((3, _JET_SIZE))
    beta = np.empty((3, _JET_SIZE))
    beta_prime = np.empty((3, _JET_SIZE))
    for axis in range(3):
        coordinate = _constant(separation[axis])
        for component in range(3):
            coordinate += frame[axis, component] * observer[component + 1]
        motion = 0.5 * acceleration[axis] * shift2 + jerk[axis] * shift3 / 6.0
        beta[axis] = acceleration[axis] * shift + 0.5 * jerk[axis] * shift2
        if axis == 0:
            coordinate -= shift
            coordinate += deficit * shift
            beta[axis][0] += 1.0 - deficit
        r[axis] = coordinate - motion
        beta_prime[axis] = (
            _constant(acceleration[axis])
            + jerk[axis] * shift
            + 0.5 * snap[axis] * shift2
        )
    radius = _norm(r)
    if r[0, 0] >= 0.0:
        chord = _divide(_multiply(r[1], r[1]) + _multiply(r[2], r[2]), radius + r[0])
        radius = r[0] + chord
    else:
        radius = -r[0] + _divide(
            _multiply(r[1], r[1]) + _multiply(r[2], r[2]), radius - r[0]
        )
        chord = radius - r[0]
    longitudinal_deficit = (
        _constant(deficit) - acceleration[0] * shift - 0.5 * jerk[0] * shift2
    )
    # R*kappa = R - r.beta, keeping the unresolved longitudinal deficit.
    denominator = (
        _multiply(longitudinal_deficit, radius)
        + _multiply(_constant(1.0) - longitudinal_deficit, chord)
        - _multiply(beta[1], r[1])
        - _multiply(beta[2], r[2])
    )
    difference = np.empty((3, _JET_SIZE))
    difference[0] = longitudinal_deficit - _divide(chord, radius)
    for axis in (1, 2):
        difference[axis] = _divide(r[axis], radius) - beta[axis]
    invariant = (
        _multiply(longitudinal_deficit, _constant(2.0) - longitudinal_deficit)
        - _multiply(beta[1], beta[1])
        - _multiply(beta[2], beta[2])
    )
    # F = (observer_ct - observer_parallel) - delta*s + a_parallel*s²/2
    #     + j_parallel*s³/6 - (R-L). Its constant is owned by the root solver.
    residual = observer[0].copy()
    for component in range(3):
        residual -= frame[0, component] * observer[component + 1]
    residual -= deficit * shift
    residual += 0.5 * acceleration[0] * shift2 + jerk[0] * shift3 / 6.0
    residual -= chord
    residual[0] = 0.0
    direction = np.empty((3, _JET_SIZE))
    direction[0] = _constant(1.0) - _divide(chord, radius)
    for axis in (1, 2):
        direction[axis] = _divide(r[axis], radius)
    return radius, denominator, difference, invariant, beta_prime, residual, direction


@register_jitable
def precise_charge_jets(
    charge: float,
    separation_vector: np.ndarray,
    proper_velocity: np.ndarray,
    acceleration_per_mm: np.ndarray,
    jerk_per_mm2: np.ndarray,
    separation_frame_mm: np.ndarray | None = None,
    snap_per_mm3: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return second-order A jets, first-order E/B jets, and root jets.

    This local Taylor germ uses exact kinematics supplied at the root. It is
    suitable for ballistic sources and for a future light-cone segment caller.
    Scalar root accuracy and input geometry remain the caller's responsibility.
    separation_frame_mm optionally supplies independently resolved (L, P, Q)
    in the velocity-aligned frame, as returned by resolved_separation_frame.
    It avoids losing transverse geometry when a large lab vector is rounded.
    """
    if (
        separation_vector.shape != (3,)
        or proper_velocity.shape != (3,)
        or acceleration_per_mm.shape != (3,)
        or jerk_per_mm2.shape != (3,)
        or not np.isfinite(charge)
        or not np.all(np.isfinite(separation_vector))
        or not np.all(np.isfinite(proper_velocity))
        or not np.all(np.isfinite(acceleration_per_mm))
        or not np.all(np.isfinite(jerk_per_mm2))
    ):
        raise ValueError("precise charge kinematics must contain finite three-vectors")
    if np.dot(separation_vector, separation_vector) <= 0.0:
        raise ValueError("the observer cannot coincide with a point-charge source")
    deficit, direction = precise_speed(proper_velocity)
    frame = _frame(direction)
    magnitude = np.sqrt(np.dot(proper_velocity, proper_velocity))
    separation_frame = _project_vector(
        separation_vector, proper_velocity, frame, magnitude
    )
    if separation_frame_mm is not None:
        if separation_frame_mm.shape != (3,) or not np.all(
            np.isfinite(separation_frame_mm)
        ):
            raise ValueError("resolved separation must contain three finite values")
        separation_frame = separation_frame_mm
    acceleration = _project_vector(
        acceleration_per_mm, proper_velocity, frame, magnitude
    )
    jerk = _project_vector(jerk_per_mm2, proper_velocity, frame, magnitude)
    snap = np.zeros(3)
    if snap_per_mm3 is not None:
        if snap_per_mm3.shape != (3,) or not np.all(np.isfinite(snap_per_mm3)):
            raise ValueError("precise charge snap must contain three finite values")
        snap = _project_vector(snap_per_mm3, proper_velocity, frame, magnitude)
    observer = np.empty((4, _JET_SIZE))
    for index in range(4):
        observer[index] = _variable(0.0, index)
    shift = _constant(0.0)
    for iteration in range(3):
        radius, denominator, difference, invariant, beta_prime, residual, n = _geometry(
            observer, shift, separation_frame, frame, deficit, acceleration, jerk, snap
        )
        if radius[0] <= 0.0 or denominator[0] <= 0.0:
            raise ValueError("precise charge kernel requires nondegenerate geometry")
        kappa = _divide(denominator, radius)
        shift += _divide(residual, kappa)
    radius, denominator, difference, invariant, beta_prime, residual, n = _geometry(
        observer, shift, separation_frame, frame, deficit, acceleration, jerk, snap
    )
    kappa = _divide(denominator, radius)
    # Second-order radiation jets are authoritative only when the caller
    # supplies snap (or the source is uniform). Stored quintics supply it.
    beta = np.empty((3, _JET_SIZE))
    shift2 = _multiply(shift, shift)
    for axis in range(3):
        beta[axis] = (
            _constant(0.0) + acceleration[axis] * shift + 0.5 * jerk[axis] * shift2
        )
        if axis == 0:
            beta[axis][0] = 1.0 - deficit
    prefactor = _divide(
        _constant(charge), _multiply(_multiply(denominator, denominator), denominator)
    )
    electric = np.empty((3, _JET_SIZE))
    radiation = _cross_jet(n, _cross_jet(difference, beta_prime))
    for axis in range(3):
        electric[axis] = _multiply(
            prefactor,
            _multiply(_multiply(invariant, radius), difference[axis])
            + _multiply(_multiply(radius, radius), radiation[axis]),
        )
    magnetic = _cross_jet(n, electric)
    potential = np.empty((4, _JET_SIZE))
    potential[0] = _divide(_constant(charge), denominator)
    for component in range(3):
        velocity = _constant(0.0)
        for axis in range(3):
            velocity += frame[axis, component] * beta[axis]
        potential[component + 1] = _multiply(potential[0], velocity)
    electric_lab = np.zeros((3, _JET_SIZE))
    magnetic_lab = np.zeros((3, _JET_SIZE))
    for component in range(3):
        for axis in range(3):
            electric_lab[component] += frame[axis, component] * electric[axis]
            magnetic_lab[component] += frame[axis, component] * magnetic[axis]
    if (
        np.all(acceleration_per_mm == 0.0)
        and np.all(jerk_per_mm2 == 0.0)
        and np.all(snap == 0.0)
    ):
        # For uniform motion use the simultaneous-separation closed form.
        # This avoids cancellations between implicit-root derivatives at the
        # beaming-cone shoulder, where the longitudinal numerator is near zero.
        longitudinal = separation_frame[0]
        transverse1 = separation_frame[1]
        transverse2 = separation_frame[2]
        radius_value = np.sqrt(np.dot(separation_frame, separation_frame))
        chord_value = (
            (transverse1**2 + transverse2**2) / (radius_value + longitudinal)
            if longitudinal >= 0.0
            else radius_value - longitudinal
        )
        present = np.empty((3, _JET_SIZE))
        for axis in range(3):
            present[axis] = _constant(0.0)
            for component in range(3):
                present[axis] += frame[axis, component] * observer[component + 1]
        present[0][0] = deficit * radius_value - chord_value
        present[0] -= observer[0]
        present[0] += deficit * observer[0]
        present[1][0] = transverse1
        present[2][0] = transverse2
        invariant_value = deficit * (2.0 - deficit)
        d = _sqrt(
            _multiply(present[0], present[0])
            + invariant_value
            * (_multiply(present[1], present[1]) + _multiply(present[2], present[2]))
        )
        phi = _divide(_constant(charge), d)
        potential[0] = phi
        for component in range(3):
            potential[component + 1] = direction[component] * (1.0 - deficit) * phi
        prefactor = _divide(
            _constant(charge * invariant_value), _multiply(_multiply(d, d), d)
        )
        for axis in range(3):
            electric[axis] = _multiply(prefactor, present[axis])
        # B = beta cross E in the velocity frame.
        magnetic[0] = _constant(0.0)
        magnetic[1] = -(1.0 - deficit) * electric[2]
        magnetic[2] = (1.0 - deficit) * electric[1]
        electric_lab[:] = 0.0
        magnetic_lab[:] = 0.0
        for component in range(3):
            for axis in range(3):
                electric_lab[component] += frame[axis, component] * electric[axis]
                magnetic_lab[component] += frame[axis, component] * magnetic[axis]
    return potential, electric_lab, magnetic_lab, shift, residual, kappa[0]


@register_jitable
def _cross_jet(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.empty((3, _JET_SIZE))
    result[0] = _multiply(left[1], right[2]) - _multiply(left[2], right[1])
    result[1] = _multiply(left[2], right[0]) - _multiply(left[0], right[2])
    result[2] = _multiply(left[0], right[1]) - _multiply(left[1], right[0])
    return result


@register_jitable
def precise_segment_jets(
    observer_position: np.ndarray,
    charge: float,
    start_time: float,
    duration: float,
    coefficients: np.ndarray,
    root_time: float,
    proper_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Adapt the selected segment's local acceleration and jerk, without editing it."""
    normalized = (root_time - start_time) / duration
    position = np.zeros(3)
    acceleration = np.zeros(3)
    jerk = np.zeros(3)
    for axis in range(3):
        for order in range(6):
            position[axis] += coefficients[order, axis] * normalized**order
        for order in range(2, 6):
            acceleration[axis] += (
                order
                * (order - 1)
                * coefficients[order, axis]
                * normalized ** (order - 2)
                / (C_MMNS * duration) ** 2
            )
        for order in range(3, 6):
            jerk[axis] += (
                order
                * (order - 1)
                * (order - 2)
                * coefficients[order, axis]
                * normalized ** (order - 3)
                / (C_MMNS * duration) ** 3
            )
    return precise_charge_jets(
        charge, observer_position - position, proper_velocity, acceleration, jerk
    )


precise_charge_jets_strict_serial = njit(cache=True, fastmath=False)(
    precise_charge_jets
)
