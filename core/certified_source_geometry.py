"""Compiled geometry with a binary64 rounding certificate and Decimal fallback.

Each scalar is (high, low, absolute error). Error-free sums and products retain
two components; discarded components, underflow, and the reference's 80-digit
rounding are enclosed by the third. No fastmath or reassociation is permitted.
See docs/certified_source_geometry.md for the enclosure and rounding argument.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .constants import C_MMNS


@njit(cache=True, error_model="numpy")
def _up(value):
    return math.nextafter(value, math.inf)


@njit(cache=True, error_model="numpy")
def _sum(a, b):
    s = a + b
    v = s - a
    return s, (a - (s - v)) + (b - v)


@njit(cache=True, error_model="numpy")
def _product(a, b):
    p = a * b
    ca, cb = 134217729.0 * a, 134217729.0 * b
    ah, bh = ca - (ca - a), cb - (cb - b)
    al, bl = a - ah, b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


@njit(cache=True, error_model="numpy")
def _float(a):
    return float(a), 0.0, 0.0


@njit(cache=True, error_model="numpy")
def _neg(a):
    return -a[0], -a[1], a[2]


@njit(cache=True, error_model="numpy")
def _magnitude(a):
    return _up(_up(abs(a[0]) + abs(a[1])) + a[2])


@njit(cache=True, error_model="numpy")
def _add_float(a, b):
    s, t = _sum(a[0], b)
    u, v = _sum(a[1], t)
    h, w = _sum(s, u)
    l, discarded = _sum(w, v)
    h, l = _sum(h, l)
    return h, l, _up(a[2] + abs(discarded))


@njit(cache=True, error_model="numpy")
def _decimal_round(a):
    # 80 significant decimal digits: relative error <= 1e-79. The fixed
    # floor also covers all underflow in the binary transforms and bounds.
    error = _up(_up(1e-79 * _magnitude(a)) + 1e-280)
    return a[0], a[1], _up(a[2] + error)


@njit(cache=True, error_model="numpy")
def _add(a, b):
    value = _add_float(_add_float(a, b[0]), b[1])
    return _decimal_round((value[0], value[1], _up(value[2] + b[2])))


@njit(cache=True, error_model="numpy")
def _mul(a, b):
    value = _float(0.0)
    for av in (a[0], a[1]):
        for bv in (b[0], b[1]):
            h, l = _product(av, bv)
            value = _add_float(_add_float(value, h), l)
    propagated = _up(_up(a[2] * _magnitude(b)) + _up(b[2] * _up(abs(a[0]) + abs(a[1]))))
    return _decimal_round(
        (value[0], value[1], _up(_up(value[2] + propagated) + 1e-280))
    )


@njit(cache=True, error_model="numpy")
def _div(a, b):
    lower = math.nextafter(abs(b[0]) - _up(abs(b[1]) + b[2]), -math.inf)
    if lower <= 0.0:
        return 0.0, 0.0, math.inf
    q = _float(a[0] / b[0])
    for _ in range(2):
        r = _add(a, _neg(_mul(q, b)))
        q = _add_float(q, (r[0] + r[1]) / b[0])
    residual = _add(a, _neg(_mul(q, b)))
    return _decimal_round((q[0], q[1], _up(_magnitude(residual) / lower)))


@njit(cache=True, error_model="numpy")
def _sqrt(a):
    if a[0] <= 0.0 or a[0] < _up(abs(a[1]) + a[2]):
        return 0.0, 0.0, math.inf
    q = _float(math.sqrt(a[0]))
    for _ in range(2):
        residual = _add(a, _neg(_mul(q, q)))
        q = _add_float(q, (residual[0] + residual[1]) / (2.0 * q[0]))
    residual = _add(a, _neg(_mul(q, q)))
    lower = math.nextafter(q[0] - abs(q[1]), -math.inf)
    return _decimal_round((q[0], q[1], _up(_magnitude(residual) / lower)))


@njit(cache=True, error_model="numpy")
def _round(a):
    candidate = a[0] + a[1]
    if not math.isfinite(candidate) or candidate == 0.0:
        return candidate, False
    residual = _add_float(a, -candidate)
    gap = min(
        candidate - math.nextafter(candidate, -math.inf),
        math.nextafter(candidate, math.inf) - candidate,
    )
    return candidate, _magnitude(residual) < 0.5 * gap


@njit(cache=True, error_model="numpy")
def _dot(a, b):
    total = _float(0.0)
    for i in range(3):
        total = _add(total, _mul(a[i], b[i]))
    return total


@njit(cache=True, error_model="numpy")
def _domain(values):
    for v in values.flat:
        if not math.isfinite(v) or abs(v) > 1e80:
            return False
    return True


@njit(cache=True, fastmath=False, error_model="numpy")
def endpoint_geometry_certified(
    frame,
    frame_low,
    origin_position,
    origin_low,
    origin_tail,
    origin_time,
    origin_time_low,
    origin_time_tail,
    observer_position,
    observer_low,
    observer_tail,
    observer_time,
    observer_time_low,
    observer_time_tail,
    endpoint,
):
    """Packed coordinates have position/time rows and high/low/tail columns."""
    origins = np.zeros((4, 3))
    observer = np.zeros((4, 3))
    for j in range(3):
        origins[j, 0], origins[j, 1] = origin_position[j], origin_low[j]
        observer[j, 0] = observer_position[j]
        if origin_tail is not None:
            origins[j, 2] = origin_tail[j]
        if observer_low is not None:
            observer[j, 1] = observer_low[j]
        if observer_tail is not None:
            observer[j, 2] = observer_tail[j]
    origins[3] = (origin_time, origin_time_low, origin_time_tail)
    observer[3] = (observer_time, observer_time_low, observer_time_tail)
    size = 10 if endpoint is not None else 5
    output = np.empty(size)
    if not (
        _domain(frame) and _domain(frame_low) and _domain(origins) and _domain(observer)
    ):
        return output, False
    if endpoint is not None and not _domain(endpoint):
        return output, False
    axes = np.empty((3, 3, 3))
    for i in range(3):
        for j in range(3):
            axes[i, j] = _add(_float(frame[i, j]), _float(frame_low[i, j]))
    for anchor in range(size // 5):
        displacement = np.empty((3, 3))
        # Enclose the original evaluation order, including upper reanchoring.
        dt = _add(_float(observer[3, 0]), _float(observer[3, 1]))
        dt = _add(dt, _neg(_float(origins[3, 0])))
        dt = _add(dt, _neg(_float(origins[3, 1])))
        dt = _add(dt, _add(_float(observer[3, 2]), _neg(_float(origins[3, 2]))))
        for j in range(3):
            d = _add(_float(observer[j, 0]), _neg(_float(origins[j, 0])))
            d = _add(d, _neg(_float(origins[j, 1])))
            d = _add(d, _neg(_float(origins[j, 2])))
            d = _add(d, _float(observer[j, 2]))
            displacement[j] = _add(d, _float(observer[j, 1]))
        if anchor and endpoint is not None:
            origin_time = _add(_float(origins[3, 0]), _float(origins[3, 1]))
            origin_time = _add(origin_time, _float(origins[3, 2]))
            end_time = _float(0.0)
            for k in range(3):
                end_time = _add(end_time, _float(endpoint[3, k]))
            dt = _add(dt, _add(origin_time, _neg(end_time)))
            for j in range(3):
                end_position = _float(0.0)
                for k in range(3):
                    displacement[j] = _add(displacement[j], _float(origins[j, k]))
                    end_position = _add(end_position, _float(endpoint[k, j]))
                displacement[j] = _add(displacement[j], _neg(end_position))
        parallel = _dot(axes[0], displacement)
        coordinate = _mul(_float(C_MMNS), dt)
        values = (
            _add(coordinate, _neg(parallel)),
            parallel,
            _dot(axes[1], displacement),
            _dot(axes[2], displacement),
            _add(coordinate, parallel),
        )
        for j in range(5):
            output[anchor * 5 + j], ok = _round(values[j])
            if not ok:
                return output, False
    return output, True


@njit(cache=True, fastmath=False, error_model="numpy")
def separation_frame_certified(frame, frame_low, separation, proper):
    output = np.empty(3)
    if not (
        _domain(frame)
        and _domain(frame_low)
        and _domain(separation)
        and _domain(proper)
    ):
        return output, False
    axes = np.empty((3, 3, 3))
    direction = np.empty((3, 3))
    for i in range(3):
        direction[i] = _float(proper[i])
        for j in range(3):
            axes[i, j] = _add(_float(frame[i, j]), _float(frame_low[i, j]))
    if np.all(proper == 0.0):
        direction[0] = _float(1.0)
    length = _sqrt(_dot(direction, direction))
    for i in range(3):
        direction[i] = _div(direction[i], length)
    # Dividing by a common positive norm preserves the binary64 input ordering;
    # distinct input components differ by far more than 80-digit roundoff.
    least = 0
    for i in range(1, 3):
        if abs(proper[i]) < abs(proper[least]):
            least = i
    if np.all(proper == 0.0):
        least = 1
    transverse = np.empty((3, 3))
    fixed = np.zeros((3, 3))
    fixed[least, 0] = 1.0
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        transverse[i] = _add(
            _mul(direction[j], fixed[k]), _neg(_mul(direction[k], fixed[j]))
        )
    length = _sqrt(_dot(transverse, transverse))
    for i in range(3):
        transverse[i] = _div(transverse[i], length)
    other = np.empty((3, 3))
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        other[i] = _add(
            _mul(direction[j], transverse[k]), _neg(_mul(direction[k], transverse[j]))
        )
    for i, new_axis in enumerate((direction, transverse, other)):
        total = _float(0.0)
        for j in range(3):
            total = _add(total, _mul(_float(separation[j]), _dot(axes[j], new_axis)))
        output[i], ok = _round(total)
        if not ok:
            return output, False
    return output, True
