"""Strict serial CPU kernels for the retarded-dipole provider.

The roots-exact entry point compiles only independent light-cone searches; the
provider then recomputes the authoritative final sample and all Hertz
arithmetic in Python.  The full-strict entry point also compiles the final
worldline sample, spin interpolation, moment boost, Hodge dual, and per-source
Hertz tensor with ``fastmath=False``.  Source addition and finite-difference
reductions remain in Python.  Numba is never imported by the default backend.
"""

from __future__ import annotations

from itertools import permutations
from typing import Callable, Tuple

import numpy as np

from .constants import C_MMNS

try:  # pragma: no cover - availability is tested through the provider seam.
    from numba import jit
    from numba.core.errors import NumbaError

    NUMBA_AVAILABLE = True
    NUMBA_COMPILATION_ERRORS: tuple[type[BaseException], ...] = (NumbaError,)
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False
    NUMBA_COMPILATION_ERRORS = ()

    def jit(
        *args: object, **kwargs: object
    ) -> Callable[[Callable[..., object]], object]:
        del args, kwargs

        def decorator(function: Callable[..., object]) -> object:
            return function

        return decorator


_STATUS_VALID = 0
_STATUS_MISSING_HISTORY = 1
_STATUS_TERMINATED_SOURCE = 2
_STATUS_MINIMUM_SEPARATION = 3
_STATUS_SUPERLUMINAL_SOURCE = 4
_STATUS_SINGULAR_KAPPA = 5
_STATUS_SPIN_INTERPOLATION_ZERO = 6


def _permutation_sign(indices: Tuple[int, int, int, int]) -> float:
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1.0 if inversions % 2 else 1.0


_LEVI_CIVITA_UPPER = np.zeros((4, 4, 4, 4), dtype=np.float64)
for _indices in permutations(range(4)):
    _LEVI_CIVITA_UPPER[_indices] = -_permutation_sign(_indices)


@jit(nopython=True, fastmath=False, nogil=True, cache=True, inline="always")
def _norm3(x: float, y: float, z: float) -> float:
    return np.sqrt(x * x + y * y + z * z)


@jit(nopython=True, fastmath=False, nogil=True, cache=True, inline="always")
def _knot_light_cone_residual_mm(
    time_ns: np.ndarray,
    position_mm: np.ndarray,
    knot_index: int,
    observer_time_ns: float,
    observer_x_mm: float,
    observer_y_mm: float,
    observer_z_mm: float,
) -> float:
    dx = observer_x_mm - position_mm[knot_index, 0]
    dy = observer_y_mm - position_mm[knot_index, 1]
    dz = observer_z_mm - position_mm[knot_index, 2]
    separation_mm = _norm3(dx, dy, dz)
    return C_MMNS * (observer_time_ns - time_ns[knot_index]) - separation_mm


@jit(nopython=True, fastmath=False, nogil=True, cache=True, inline="always")
def _quintic_worldline_sample(
    time_ns: np.ndarray,
    segment_duration_ns: np.ndarray,
    position_coefficients_mm: np.ndarray,
    segment_index: int,
    sample_time_ns: float,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    duration = segment_duration_ns[segment_index]
    tau = (sample_time_ns - time_ns[segment_index]) / duration
    if tau < 0.0:
        tau = 0.0
    elif tau > 1.0:
        tau = 1.0
    tau2 = tau**2
    tau3 = tau**3
    tau4 = tau**4
    tau5 = tau**5

    output = np.empty(9, dtype=np.float64)
    for axis in range(3):
        c0 = position_coefficients_mm[segment_index, 0, axis]
        c1 = position_coefficients_mm[segment_index, 1, axis]
        c2 = position_coefficients_mm[segment_index, 2, axis]
        c3 = position_coefficients_mm[segment_index, 3, axis]
        c4 = position_coefficients_mm[segment_index, 4, axis]
        c5 = position_coefficients_mm[segment_index, 5, axis]
        output[axis] = c0 + c1 * tau + c2 * tau2 + c3 * tau3 + c4 * tau4 + c5 * tau5
        output[3 + axis] = (
            (c1 + 2.0 * c2 * tau + 3.0 * c3 * tau2 + 4.0 * c4 * tau3 + 5.0 * c5 * tau4)
            / duration
            / C_MMNS
        )
        output[6 + axis] = (
            (2.0 * c2 + 6.0 * c3 * tau + 12.0 * c4 * tau2 + 20.0 * c5 * tau3)
            / duration**2
            / C_MMNS**2
        )
    return (
        output[0],
        output[1],
        output[2],
        output[3],
        output[4],
        output[5],
        output[6],
        output[7],
        output[8],
    )


@jit(nopython=True, fastmath=False, nogil=True, cache=True, inline="always")
def _solve_retarded_sample(
    time_ns: np.ndarray,
    position_mm: np.ndarray,
    segment_duration_ns: np.ndarray,
    position_coefficients_mm: np.ndarray,
    observer_time_ns: float,
    observer_x_mm: float,
    observer_y_mm: float,
    observer_z_mm: float,
    root_tolerance_mm: float,
    max_root_iterations: int,
    ended_by_loss: bool,
    certified_segment: int = -2,
) -> Tuple[
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    knot_count = time_ns.size
    nan = np.nan
    if knot_count < 2:
        status = _STATUS_MISSING_HISTORY
        if ended_by_loss:
            if knot_count == 0:
                status = _STATUS_TERMINATED_SOURCE
            else:
                last_residual = _knot_light_cone_residual_mm(
                    time_ns,
                    position_mm,
                    0,
                    observer_time_ns,
                    observer_x_mm,
                    observer_y_mm,
                    observer_z_mm,
                )
                if last_residual > 0.0:
                    status = _STATUS_TERMINATED_SOURCE
        return (
            status,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
        )

    lower_index = 0
    upper_index = knot_count - 1
    lower_residual = _knot_light_cone_residual_mm(
        time_ns,
        position_mm,
        lower_index,
        observer_time_ns,
        observer_x_mm,
        observer_y_mm,
        observer_z_mm,
    )
    upper_residual = _knot_light_cone_residual_mm(
        time_ns,
        position_mm,
        upper_index,
        observer_time_ns,
        observer_x_mm,
        observer_y_mm,
        observer_z_mm,
    )
    if lower_residual < 0.0 or upper_residual > 0.0:
        status = (
            _STATUS_TERMINATED_SOURCE
            if ended_by_loss and upper_residual > 0.0
            else _STATUS_MISSING_HISTORY
        )
        return (
            status,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
            nan,
        )

    hint_is_valid = False
    if 0 <= certified_segment < knot_count - 1:
        hint_lower = _knot_light_cone_residual_mm(
            time_ns,
            position_mm,
            certified_segment,
            observer_time_ns,
            observer_x_mm,
            observer_y_mm,
            observer_z_mm,
        )
        hint_upper = _knot_light_cone_residual_mm(
            time_ns,
            position_mm,
            certified_segment + 1,
            observer_time_ns,
            observer_x_mm,
            observer_y_mm,
            observer_z_mm,
        )
        hint_is_valid = (
            hint_lower >= 0.0
            and hint_upper <= 0.0
            and (hint_upper < 0.0 or certified_segment + 1 == knot_count - 1)
        )
        if hint_is_valid:
            lower_index = certified_segment
            upper_index = certified_segment + 1
    if not hint_is_valid:
        while upper_index - lower_index > 1:
            middle_index = lower_index + (upper_index - lower_index) // 2
            middle_residual = _knot_light_cone_residual_mm(
                time_ns,
                position_mm,
                middle_index,
                observer_time_ns,
                observer_x_mm,
                observer_y_mm,
                observer_z_mm,
            )
            if middle_residual >= 0.0:
                lower_index = middle_index
            else:
                upper_index = middle_index

    segment_index = lower_index
    lower_time_ns = time_ns[segment_index]
    upper_time_ns = time_ns[segment_index + 1]
    trial_time_ns = 0.5 * (lower_time_ns + upper_time_ns)

    for _ in range(max_root_iterations):
        (
            source_x_mm,
            source_y_mm,
            source_z_mm,
            source_beta_x,
            source_beta_y,
            source_beta_z,
            _source_beta_prime_x,
            _source_beta_prime_y,
            _source_beta_prime_z,
        ) = _quintic_worldline_sample(
            time_ns,
            segment_duration_ns,
            position_coefficients_mm,
            segment_index,
            trial_time_ns,
        )
        dx = observer_x_mm - source_x_mm
        dy = observer_y_mm - source_y_mm
        dz = observer_z_mm - source_z_mm
        separation_mm = _norm3(dx, dy, dz)
        residual_mm = C_MMNS * (observer_time_ns - trial_time_ns) - separation_mm
        if abs(residual_mm) <= root_tolerance_mm:
            lower_time_ns = trial_time_ns
            upper_time_ns = trial_time_ns
            break
        if residual_mm > 0.0:
            lower_time_ns = trial_time_ns
        else:
            upper_time_ns = trial_time_ns
        if np.nextafter(lower_time_ns, upper_time_ns) >= upper_time_ns:
            break

        derivative_mm_per_ns = np.nan
        if separation_mm > 0.0:
            projection = (
                dx / separation_mm * source_beta_x
                + dy / separation_mm * source_beta_y
                + dz / separation_mm * source_beta_z
            )
            derivative_mm_per_ns = -C_MMNS * (1.0 - projection)
        available_time_ns = (
            upper_time_ns - trial_time_ns
            if residual_mm > 0.0
            else trial_time_ns - lower_time_ns
        )
        maximum_newton_residual_mm = -derivative_mm_per_ns * available_time_ns
        if (
            np.isfinite(derivative_mm_per_ns)
            and derivative_mm_per_ns < 0.0
            and np.isfinite(maximum_newton_residual_mm)
            and abs(residual_mm) < maximum_newton_residual_mm
        ):
            next_trial_time_ns = trial_time_ns - residual_mm / derivative_mm_per_ns
        else:
            next_trial_time_ns = np.nan

        if next_trial_time_ns == trial_time_ns:
            target_time_ns = upper_time_ns if residual_mm > 0.0 else lower_time_ns
            trial_time_ns = np.nextafter(trial_time_ns, target_time_ns)
        elif (
            not np.isfinite(next_trial_time_ns)
            or next_trial_time_ns <= lower_time_ns
            or next_trial_time_ns >= upper_time_ns
        ):
            trial_time_ns = 0.5 * (lower_time_ns + upper_time_ns)
        else:
            trial_time_ns = next_trial_time_ns

    retarded_time_ns = 0.5 * (lower_time_ns + upper_time_ns)
    (
        source_x_mm,
        source_y_mm,
        source_z_mm,
        source_beta_x,
        source_beta_y,
        source_beta_z,
        source_beta_prime_x,
        source_beta_prime_y,
        source_beta_prime_z,
    ) = _quintic_worldline_sample(
        time_ns,
        segment_duration_ns,
        position_coefficients_mm,
        segment_index,
        retarded_time_ns,
    )
    dx = observer_x_mm - source_x_mm
    dy = observer_y_mm - source_y_mm
    dz = observer_z_mm - source_z_mm
    separation_mm = _norm3(dx, dy, dz)
    residual_mm = C_MMNS * (observer_time_ns - retarded_time_ns) - separation_mm
    return (
        _STATUS_VALID,
        retarded_time_ns,
        source_x_mm,
        source_y_mm,
        source_z_mm,
        source_beta_x,
        source_beta_y,
        source_beta_z,
        source_beta_prime_x,
        source_beta_prime_y,
        source_beta_prime_z,
        residual_mm,
        separation_mm,
    )


@jit(nopython=True, fastmath=False, nogil=True, cache=True, inline="always")
def _right_search(time_ns: np.ndarray, target_time_ns: float) -> int:
    lower = 0
    upper = time_ns.size
    while lower < upper:
        middle = lower + (upper - lower) // 2
        if target_time_ns < time_ns[middle]:
            upper = middle
        else:
            lower = middle + 1
    return lower


@jit(nopython=True, fastmath=False, nogil=True, cache=True, inline="always")
def _interpolate_rest_spin_c1_strict(
    time_ns: np.ndarray,
    rest_spin: np.ndarray,
    rest_spin_derivative_per_ns: np.ndarray,
    retarded_time_ns: float,
    preserve_magnitude: bool,
    preserved_magnitude: float,
) -> Tuple[int, float, float, float]:
    count = time_ns.size
    if count == 1:
        return _STATUS_VALID, rest_spin[0, 0], rest_spin[0, 1], rest_spin[0, 2]

    segment = _right_search(time_ns, retarded_time_ns) - 1
    if segment < 0:
        segment = 0
    elif segment > count - 2:
        segment = count - 2
    duration_ns = time_ns[segment + 1] - time_ns[segment]
    fraction = (retarded_time_ns - time_ns[segment]) / duration_ns
    if fraction < 0.0:
        fraction = 0.0
    elif fraction > 1.0:
        fraction = 1.0
    fraction_squared = fraction * fraction
    fraction_cubed = fraction_squared * fraction
    h00 = 2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0
    h10 = fraction_cubed - 2.0 * fraction_squared + fraction
    h01 = -2.0 * fraction_cubed + 3.0 * fraction_squared
    h11 = fraction_cubed - fraction_squared

    interpolated = np.empty(3, dtype=np.float64)
    for axis in range(3):
        interpolated[axis] = (
            h00 * rest_spin[segment, axis]
            + h10 * duration_ns * rest_spin_derivative_per_ns[segment, axis]
            + h01 * rest_spin[segment + 1, axis]
            + h11 * duration_ns * rest_spin_derivative_per_ns[segment + 1, axis]
        )
    if preserve_magnitude:
        if preserved_magnitude == 0.0:
            interpolated[0] = 0.0
            interpolated[1] = 0.0
            interpolated[2] = 0.0
        else:
            magnitude = _norm3(interpolated[0], interpolated[1], interpolated[2])
            if magnitude <= 1.0e-15:
                return _STATUS_SPIN_INTERPOLATION_ZERO, np.nan, np.nan, np.nan
            scale = preserved_magnitude / magnitude
            interpolated[0] *= scale
            interpolated[1] *= scale
            interpolated[2] *= scale
    return _STATUS_VALID, interpolated[0], interpolated[1], interpolated[2]


@jit(nopython=True, fastmath=False, nogil=True, cache=True, inline="always")
def _moment_hertz_tensor_strict(
    magnetic_moment_native: float,
    spin_x: float,
    spin_y: float,
    spin_z: float,
    source_beta_x: float,
    source_beta_y: float,
    source_beta_z: float,
    dx: float,
    dy: float,
    dz: float,
    separation_mm: float,
) -> Tuple[int, np.ndarray]:
    beta_squared = (
        source_beta_x * source_beta_x
        + source_beta_y * source_beta_y
        + source_beta_z * source_beta_z
    )
    empty = np.zeros((4, 4), dtype=np.float64)
    if beta_squared >= 1.0:
        return _STATUS_SUPERLUMINAL_SOURCE, empty
    gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    direction_x = dx / separation_mm
    direction_y = dy / separation_mm
    direction_z = dz / separation_mm
    kappa = 1.0 - (
        direction_x * source_beta_x
        + direction_y * source_beta_y
        + direction_z * source_beta_z
    )
    if kappa <= 1.0e-14:
        return _STATUS_SINGULAR_KAPPA, empty
    invariant_retarded_distance_mm = gamma * separation_mm * kappa

    projection = (
        source_beta_x * spin_x + source_beta_y * spin_y + source_beta_z * spin_z
    )
    boost_coefficient = gamma * gamma / (gamma + 1.0) * projection
    moment_four = np.empty(4, dtype=np.float64)
    moment_four[0] = magnetic_moment_native * gamma * projection
    moment_four[1] = magnetic_moment_native * (
        spin_x + boost_coefficient * source_beta_x
    )
    moment_four[2] = magnetic_moment_native * (
        spin_y + boost_coefficient * source_beta_y
    )
    moment_four[3] = magnetic_moment_native * (
        spin_z + boost_coefficient * source_beta_z
    )
    four_velocity = np.empty(4, dtype=np.float64)
    four_velocity[0] = gamma * C_MMNS
    four_velocity[1] = gamma * C_MMNS * source_beta_x
    four_velocity[2] = gamma * C_MMNS * source_beta_y
    four_velocity[3] = gamma * C_MMNS * source_beta_z

    wedge = np.empty((4, 4), dtype=np.float64)
    for mu in range(4):
        for nu in range(4):
            wedge[mu, nu] = (
                four_velocity[mu] * moment_four[nu]
                - moment_four[mu] * four_velocity[nu]
            ) / C_MMNS

    signs = (1.0, -1.0, -1.0, -1.0)
    hertz = np.empty((4, 4), dtype=np.float64)
    for mu in range(4):
        for nu in range(4):
            total = 0.0
            for alpha in range(4):
                for beta in range(4):
                    total += (
                        _LEVI_CIVITA_UPPER[mu, nu, alpha, beta]
                        * signs[alpha]
                        * signs[beta]
                        * wedge[alpha, beta]
                    )
            hertz[mu, nu] = 0.5 * total / invariant_retarded_distance_mm
    return _STATUS_VALID, hertz


@jit(nopython=True, fastmath=False, nogil=True, cache=True, inline="always")
def _evaluate_one_full_strict_event(
    time_ns: np.ndarray,
    position_mm: np.ndarray,
    segment_duration_ns: np.ndarray,
    position_coefficients_mm: np.ndarray,
    rest_spin: np.ndarray,
    rest_spin_derivative_per_ns: np.ndarray,
    preserve_magnitude: bool,
    preserved_magnitude: float,
    magnetic_moment_native: float,
    ended_by_loss: bool,
    observer_time_ns: float,
    observer_x_mm: float,
    observer_y_mm: float,
    observer_z_mm: float,
    minimum_separation_mm: float,
    root_tolerance_mm: float,
    max_root_iterations: int,
    certified_segment: int = -2,
) -> Tuple[int, np.ndarray, float, float, float, bool]:
    (
        status,
        retarded_time_ns,
        source_x_mm,
        source_y_mm,
        source_z_mm,
        source_beta_x,
        source_beta_y,
        source_beta_z,
        _source_beta_prime_x,
        _source_beta_prime_y,
        _source_beta_prime_z,
        residual_mm,
        separation_mm,
    ) = _solve_retarded_sample(
        time_ns,
        position_mm,
        segment_duration_ns,
        position_coefficients_mm,
        observer_time_ns,
        observer_x_mm,
        observer_y_mm,
        observer_z_mm,
        root_tolerance_mm,
        max_root_iterations,
        ended_by_loss,
        certified_segment,
    )
    empty = np.zeros((4, 4), dtype=np.float64)
    if status == _STATUS_TERMINATED_SOURCE:
        return status, empty, np.nan, np.nan, np.nan, False
    if status != _STATUS_VALID:
        return _STATUS_MISSING_HISTORY, empty, np.nan, np.nan, np.nan, False
    if separation_mm <= minimum_separation_mm:
        return (
            _STATUS_MINIMUM_SEPARATION,
            empty,
            retarded_time_ns,
            residual_mm,
            separation_mm,
            False,
        )

    spin_status, spin_x, spin_y, spin_z = _interpolate_rest_spin_c1_strict(
        time_ns,
        rest_spin,
        rest_spin_derivative_per_ns,
        retarded_time_ns,
        preserve_magnitude,
        preserved_magnitude,
    )
    if spin_status != _STATUS_VALID:
        return spin_status, empty, retarded_time_ns, residual_mm, separation_mm, False
    hertz_status, hertz = _moment_hertz_tensor_strict(
        magnetic_moment_native,
        spin_x,
        spin_y,
        spin_z,
        source_beta_x,
        source_beta_y,
        source_beta_z,
        observer_x_mm - source_x_mm,
        observer_y_mm - source_y_mm,
        observer_z_mm - source_z_mm,
        separation_mm,
    )
    return (
        hertz_status,
        hertz,
        retarded_time_ns,
        residual_mm,
        separation_mm,
        hertz_status == _STATUS_VALID,
    )


@jit(nopython=True, fastmath=False, nogil=True, cache=True)
def evaluate_source_events_full_strict_serial(
    time_ns: np.ndarray,
    position_mm: np.ndarray,
    segment_duration_ns: np.ndarray,
    position_coefficients_mm: np.ndarray,
    rest_spin: np.ndarray,
    rest_spin_derivative_per_ns: np.ndarray,
    preserve_magnitude: bool,
    preserved_magnitude: float,
    magnetic_moment_native: float,
    ended_by_loss: bool,
    observer_time_ns: np.ndarray,
    observer_position_mm: np.ndarray,
    minimum_separation_mm: float,
    root_tolerance_mm: float,
    max_root_iterations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one source at every event with strict serial arithmetic."""

    event_count = observer_time_ns.size
    status = np.empty(event_count, dtype=np.int64)
    hertz = np.zeros((event_count, 4, 4), dtype=np.float64)
    retarded_time_ns = np.full(event_count, np.nan, dtype=np.float64)
    residual_mm = np.full(event_count, np.nan, dtype=np.float64)
    separation_mm = np.full(event_count, np.nan, dtype=np.float64)
    valid = np.zeros(event_count, dtype=np.bool_)
    for event_index in range(event_count):
        (
            status[event_index],
            hertz[event_index],
            retarded_time_ns[event_index],
            residual_mm[event_index],
            separation_mm[event_index],
            valid[event_index],
        ) = _evaluate_one_full_strict_event(
            time_ns,
            position_mm,
            segment_duration_ns,
            position_coefficients_mm,
            rest_spin,
            rest_spin_derivative_per_ns,
            preserve_magnitude,
            preserved_magnitude,
            magnetic_moment_native,
            ended_by_loss,
            observer_time_ns[event_index],
            observer_position_mm[event_index, 0],
            observer_position_mm[event_index, 1],
            observer_position_mm[event_index, 2],
            minimum_separation_mm,
            root_tolerance_mm,
            max_root_iterations,
        )
    return status, hertz, retarded_time_ns, residual_mm, separation_mm, valid


@jit(nopython=True, fastmath=False, nogil=True, cache=True)
def evaluate_source_events_full_strict_from_segments_serial(
    time_ns: np.ndarray,
    position_mm: np.ndarray,
    segment_duration_ns: np.ndarray,
    position_coefficients_mm: np.ndarray,
    rest_spin: np.ndarray,
    rest_spin_derivative_per_ns: np.ndarray,
    preserve_magnitude: bool,
    preserved_magnitude: float,
    magnetic_moment_native: float,
    ended_by_loss: bool,
    observer_time_ns: np.ndarray,
    observer_position_mm: np.ndarray,
    minimum_separation_mm: float,
    root_tolerance_mm: float,
    max_root_iterations: int,
    certified_segments: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate strict events from certified brackets, with CPU fallback."""

    event_count = observer_time_ns.size
    status = np.empty(event_count, dtype=np.int64)
    hertz = np.zeros((event_count, 4, 4), dtype=np.float64)
    retarded_time_ns = np.full(event_count, np.nan, dtype=np.float64)
    residual_mm = np.full(event_count, np.nan, dtype=np.float64)
    separation_mm = np.full(event_count, np.nan, dtype=np.float64)
    valid = np.zeros(event_count, dtype=np.bool_)
    for event_index in range(event_count):
        (
            status[event_index],
            hertz[event_index],
            retarded_time_ns[event_index],
            residual_mm[event_index],
            separation_mm[event_index],
            valid[event_index],
        ) = _evaluate_one_full_strict_event(
            time_ns,
            position_mm,
            segment_duration_ns,
            position_coefficients_mm,
            rest_spin,
            rest_spin_derivative_per_ns,
            preserve_magnitude,
            preserved_magnitude,
            magnetic_moment_native,
            ended_by_loss,
            observer_time_ns[event_index],
            observer_position_mm[event_index, 0],
            observer_position_mm[event_index, 1],
            observer_position_mm[event_index, 2],
            minimum_separation_mm,
            root_tolerance_mm,
            max_root_iterations,
            int(certified_segments[event_index]),
        )
    return status, hertz, retarded_time_ns, residual_mm, separation_mm, valid


@jit(nopython=True, fastmath=False, nogil=True, cache=True)
def evaluate_source_roots_exact_serial(
    time_ns: np.ndarray,
    position_mm: np.ndarray,
    segment_duration_ns: np.ndarray,
    position_coefficients_mm: np.ndarray,
    ended_by_loss: bool,
    observer_time_ns: np.ndarray,
    observer_position_mm: np.ndarray,
    root_tolerance_mm: float,
    max_root_iterations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve one source at every event without changing event order."""

    event_count = observer_time_ns.size
    status = np.empty(event_count, dtype=np.int64)
    source_sample = np.full((event_count, 9), np.nan, dtype=np.float64)
    retarded_time_ns = np.full(event_count, np.nan, dtype=np.float64)
    residual_mm = np.full(event_count, np.nan, dtype=np.float64)
    separation_mm = np.full(event_count, np.nan, dtype=np.float64)
    for event_index in range(event_count):
        (
            status[event_index],
            retarded_time_ns[event_index],
            source_sample[event_index, 0],
            source_sample[event_index, 1],
            source_sample[event_index, 2],
            source_sample[event_index, 3],
            source_sample[event_index, 4],
            source_sample[event_index, 5],
            source_sample[event_index, 6],
            source_sample[event_index, 7],
            source_sample[event_index, 8],
            residual_mm[event_index],
            separation_mm[event_index],
        ) = _solve_retarded_sample(
            time_ns,
            position_mm,
            segment_duration_ns,
            position_coefficients_mm,
            observer_time_ns[event_index],
            observer_position_mm[event_index, 0],
            observer_position_mm[event_index, 1],
            observer_position_mm[event_index, 2],
            root_tolerance_mm,
            max_root_iterations,
            ended_by_loss,
        )
    return status, source_sample, retarded_time_ns, residual_mm, separation_mm


__all__ = [
    "NUMBA_AVAILABLE",
    "evaluate_source_events_full_strict_serial",
    "evaluate_source_events_full_strict_from_segments_serial",
    "evaluate_source_roots_exact_serial",
]
