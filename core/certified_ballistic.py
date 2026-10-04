"""Compiled geometry with a binary64 rounding certificate and Decimal fallback.

Each scalar is (high, low, third, fourth, absolute error). Error-free sums and products retain
four components; discarded components, underflow, and the reference's 90-digit
rounding are enclosed by the fifth component. No fastmath or reassociation is permitted.
See docs/certified_source_geometry.md for the enclosure and rounding argument.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .constants import C_MMNS


@njit(cache=True, inline="always", error_model="numpy")
def _up(value):
    return math.nextafter(value, math.inf)


@njit(cache=True, inline="always", error_model="numpy")
def _sum(a, b):
    s = a + b
    v = s - a
    return s, (a - (s - v)) + (b - v)


@njit(cache=True, inline="always", error_model="numpy")
def _product(a, b):
    p = a * b
    ca, cb = 134217729.0 * a, 134217729.0 * b
    ah, bh = ca - (ca - a), cb - (cb - b)
    al, bl = a - ah, b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


@njit(cache=True, inline="always", error_model="numpy")
def _float(a):
    return float(a), 0.0, 0.0, 0.0, 0.0


@njit(cache=True, inline="always", error_model="numpy")
def _neg(a):
    return -a[0], -a[1], -a[2], -a[3], a[4]


@njit(cache=True, inline="always", error_model="numpy")
def _magnitude(a):
    return _up(_up(_up(_up(abs(a[0]) + abs(a[1])) + abs(a[2])) + abs(a[3])) + a[4])


@njit(cache=True, inline="always", error_model="numpy")
def _add_float(a, b):
    h, remainder = _sum(a[0], b)
    l, remainder = _sum(a[1], remainder)
    t, remainder = _sum(a[2], remainder)
    z, remainder = _sum(a[3], remainder)
    t, z = _sum(t, z)
    l, t = _sum(l, t)
    h, l = _sum(h, l)
    return h, l, t, z, _up(a[4] + abs(remainder))


@njit(cache=True, error_model="numpy")
def _decimal_round(a):
    # 90 significant decimal digits: relative error <= 1e-89. The fixed
    # floor also covers all underflow in the binary transforms and bounds.
    error = _up(_up(1e-89 * _magnitude(a)) + 1e-280)
    return a[0], a[1], a[2], a[3], _up(a[4] + error)


@njit(cache=True, error_model="numpy")
def _add(a, b):
    value = (a[0], a[1], a[2], a[3], a[4])
    for i in range(4):
        value = _add_float(value, b[i])
    return _decimal_round(
        (value[0], value[1], value[2], value[3], _up(value[4] + b[4]))
    )


@njit(cache=True, error_model="numpy")
def _mul(a, b):
    value = _float(0.0)
    discarded = 0.0
    for i in range(4):
        av = a[i]
        for j in range(4):
            bv = b[j]
            if av == 0.0 or bv == 0.0:
                continue
            if i + j > 3:
                discarded = _up(discarded + _up(abs(av) * abs(bv)))
                continue
            h, l = _product(av, bv)
            value = _add_float(_add_float(value, h), l)
    propagated = _up(
        _up(a[4] * _magnitude(b))
        + _up(b[4] * _magnitude((a[0], a[1], a[2], a[3], 0.0)))
    )
    return _decimal_round(
        (
            value[0],
            value[1],
            value[2],
            value[3],
            _up(_up(_up(value[4] + propagated) + discarded) + 1e-280),
        )
    )


@njit(cache=True, error_model="numpy")
def _div(a, b):
    lower = math.nextafter(
        abs(b[0]) - _magnitude((b[1], b[2], b[3], 0.0, b[4])), -math.inf
    )
    if lower <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, math.inf
    q = _float(a[0] / b[0])
    for _ in range(3):
        r = _add(a, _neg(_mul(q, b)))
        q = _add_float(q, (r[0] + r[1]) / b[0])
    residual = _add(a, _neg(_mul(q, b)))
    return _decimal_round((q[0], q[1], q[2], q[3], _up(_magnitude(residual) / lower)))


@njit(cache=True, error_model="numpy")
def _sqrt(a):
    if a[0] <= 0.0 or a[0] < _magnitude((a[1], a[2], a[3], 0.0, a[4])):
        return 0.0, 0.0, 0.0, 0.0, math.inf
    q = _float(math.sqrt(a[0]))
    for _ in range(2):
        residual = _add(a, _neg(_mul(q, q)))
        q = _add_float(q, (residual[0] + residual[1]) / (2.0 * q[0]))
    residual = _add(a, _neg(_mul(q, q)))
    lower = math.nextafter(q[0] - _magnitude((q[1], q[2], q[3], 0.0, 0.0)), -math.inf)
    return _decimal_round((q[0], q[1], q[2], q[3], _up(_magnitude(residual) / lower)))


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


@njit(cache=True, error_model="numpy")
def _abs(a):
    for i in range(4):
        if a[i] < 0.0:
            return _neg(a)
        if a[i] > 0.0:
            break
    return a[0], a[1], a[2], a[3], a[4]


@njit(cache=True, error_model="numpy")
def _maximum(a, b):
    selected = a if a[0] >= b[0] else b
    ar = _magnitude((a[1], a[2], a[3], 0.0, a[4]))
    br = _magnitude((b[1], b[2], b[3], 0.0, b[4]))
    if abs(a[0] - b[0]) <= _up(ar + br):
        difference = _magnitude(_add(a, _neg(b)))
        return (
            selected[0],
            selected[1],
            selected[2],
            selected[3],
            _up(selected[4] + difference),
        )
    return selected


@njit(cache=True, fastmath=False, error_model="numpy")
def ballistic_geometry_certified(
    observer_time,
    observer,
    anchor_time,
    anchor,
    proper,
    observer_time_low,
    observer_low,
    observer_time_tail,
    observer_tail,
    constants,
):
    """Return all ten rounded Decimal diagnostics, or request the oracle."""
    output = np.empty(10)
    if not (_domain(observer) and _domain(anchor) and _domain(proper)):
        return output, False
    if observer_low is not None and not _domain(observer_low):
        return output, False
    if observer_tail is not None and not _domain(observer_tail):
        return output, False
    times = np.array(
        (observer_time, anchor_time, observer_time_low, observer_time_tail)
    )
    if not _domain(times):
        return output, False
    u = np.empty((3, 5))
    for i in range(3):
        u[i] = _float(proper[i])
    deficit = (
        constants[0, 0],
        constants[0, 1],
        constants[0, 2],
        constants[0, 3],
        constants[0, 4],
    )
    speed = (
        constants[1, 0],
        constants[1, 1],
        constants[1, 2],
        constants[1, 3],
        constants[1, 4],
    )
    direction = np.empty((3, 5))
    for i in range(3):
        direction[i] = constants[i + 2]
    displacement = np.empty((3, 5))
    for i in range(3):
        d = _add(_float(observer[i]), _neg(_float(anchor[i])))
        if observer_low is not None:
            d = _add(d, _float(observer_low[i]))
        if observer_tail is not None:
            d = _add(d, _float(observer_tail[i]))
        displacement[i] = d
    ct = _add(_float(observer_time), _float(observer_time_low))
    ct = _add(ct, _float(observer_time_tail))
    ct = _mul(_float(C_MMNS), _add(ct, _neg(_float(anchor_time))))
    longitudinal = _dot(displacement, direction)
    perpendicular = np.empty((3, 5))
    for i in range(3):
        perpendicular[i] = _add(displacement[i], _neg(_mul(longitudinal, direction[i])))
    transverse2 = _dot(perpendicular, perpendicular)
    present = _add(_add(longitudinal, _neg(ct)), _mul(deficit, ct))
    invariant = _mul(deficit, _add(_float(2.0), _neg(deficit)))
    discriminant = _sqrt(_add(_mul(present, present), _mul(invariant, transverse2)))
    # An uncertain branch uses Decimal; branch selection itself is never relaxed.
    if abs(present[0]) <= _magnitude(
        (present[1], present[2], present[3], 0.0, present[4])
    ):
        return output, False
    if present[0] >= 0.0:
        radius = _div(_add(_mul(speed, present), discriminant), invariant)
    else:
        radius = _div(
            _add(_mul(present, present), transverse2),
            _add(discriminant, _neg(_mul(speed, present))),
        )
    if radius[0] <= 0.0:
        return output, False
    source_ct = _add(ct, _neg(radius))
    exact_time = _add(_float(anchor_time), _div(source_ct, _float(C_MMNS)))
    rounded_time, ok = _round(exact_time)
    if not ok:
        return output, False
    scale = _float(1.0)
    for value in (source_ct, ct, longitudinal, radius):
        scale = _maximum(scale, _abs(value))
    # 1e-65 is a decimal literal, not the exact value of its binary64 spelling.
    # Enclose that spelling error (less than one binary64 ulp) explicitly.
    allowance_literal = (1e-65, 0.0, 0.0, 0.0, 2e-81)
    allowance = _div(_mul(allowance_literal, scale), _mul(invariant, _float(C_MMNS)))
    bound = _add(_abs(_add(_float(rounded_time), _neg(exact_time))), allowance)
    separation_longitudinal = _add(present, _mul(speed, radius))
    separation = np.empty((3, 5))
    for i in range(3):
        separation[i] = _add(
            perpendicular[i], _mul(separation_longitudinal, direction[i])
        )
    rounded_ct = _mul(
        _float(C_MMNS), _add(_float(rounded_time), _neg(_float(anchor_time)))
    )
    rounded_vector = np.empty((3, 5))
    for i in range(3):
        rounded_vector[i] = _add(
            displacement[i], _neg(_mul(_mul(speed, direction[i]), rounded_ct))
        )
    residual = _add(
        _add(ct, _neg(rounded_ct)), _neg(_sqrt(_dot(rounded_vector, rounded_vector)))
    )
    output[0] = rounded_time
    values = np.empty((9, 5))
    values[:3] = separation
    values[3], values[4], values[5] = residual, bound, separation_longitudinal
    values[6:] = perpendicular
    for i in range(9):
        # Exact axis directions are known Decimal constants. On that axis the
        # perpendicular subtraction is value-value; other zero displacements
        # also stay exactly zero through the reference's sum starting at +0.
        if i >= 6:
            axis = i - 6
            if (
                abs(direction[axis, 0]) == 1.0
                and np.all(direction[axis, 1:4] == 0.0)
                and direction[axis, 4] <= 5e-324
            ):
                others_zero = True
                for j in range(3):
                    if j != axis and np.any(direction[j, :4] != 0.0):
                        others_zero = False
                if others_zero:
                    output[i + 1] = 0.0
                    continue
        if i in (0, 1, 2, 6, 7, 8):
            axis = i if i < 3 else i - 6
            if (
                np.all(direction[axis, :4] == 0.0)
                and direction[axis, 4] <= 5e-324
                and (observer_low is None or observer_low[axis] == 0.0)
                and (observer_tail is None or observer_tail[axis] == 0.0)
                and observer[axis] != anchor[axis]
                and (
                    observer[axis] == 0.0 or 2.0**-20 <= abs(observer[axis]) <= 2.0**20
                )
                and (anchor[axis] == 0.0 or 2.0**-20 <= abs(anchor[axis]) <= 2.0**20)
            ):
                # These dyadic coordinates and their difference need fewer
                # than 90 decimal digits. The reference is exact even at a
                # binary64 midpoint, so ordinary subtraction gives its tie.
                output[i + 1] = observer[axis] - anchor[axis]
                continue
            if (
                np.all(direction[axis, :4] == 0.0)
                and observer[axis] == anchor[axis]
                and not (observer[axis] == 0.0 and np.signbit(observer[axis]))
                and (observer_low is None or observer_low[axis] == 0.0)
                and (observer_tail is None or observer_tail[axis] == 0.0)
            ):
                output[i + 1] = 0.0
                continue
        output[i + 1], ok = _round(values[i])
        if not ok:
            return output, False
    output[5] = math.nextafter(output[5], math.inf)
    return output, True
