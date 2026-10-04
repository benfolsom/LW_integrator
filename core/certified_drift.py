"""Certified rounded remainders for the observer's accepted drift arithmetic."""

import numpy as np
from numba import njit

from .certified_ballistic import (
    _add,
    _div,
    _domain,
    _dot,
    _float,
    _mul,
    _neg,
    _round,
    _sqrt,
)
from .constants import C_MMNS


@njit(cache=True, fastmath=False, error_model="numpy")
def drift_remainders_certified(
    current, rounded, start_u, end_u, h, start_gamma, end_gamma, centered, on_shell
):
    """Return speed deficit, low parts, and tails, or request Decimal."""
    output = np.empty(9)
    if not (
        _domain(current) and _domain(rounded) and _domain(start_u) and _domain(end_u)
    ):
        return output, False
    if not np.isfinite(h) or abs(h) > 1e80:
        return output, False
    start = np.empty((3, 5))
    end = np.empty((3, 5))
    for i in range(3):
        start[i], end[i] = _float(start_u[i]), _float(end_u[i])
    end_norm = _sqrt(_dot(end, end))
    if np.all(end_u == 0.0):
        end_norm = _float(0.0)
    true_end_gamma = _sqrt(_add(_float(1.0), _mul(end_norm, end_norm)))
    selected_end_gamma = true_end_gamma if on_shell else _float(end_gamma)
    if centered and on_shell:
        selected_start_gamma = _sqrt(_add(_float(1.0), _dot(start, start)))
    elif centered:
        selected_start_gamma = _float(start_gamma)
    else:
        selected_start_gamma = selected_end_gamma
    scale = _div(_float(h), _float(2.0))
    deficit = _div(_float(1.0), _mul(true_end_gamma, _add(true_end_gamma, end_norm)))
    output[0], ok = _round(deficit)
    if not ok:
        return output, False
    for i in range(4):
        value = _add(_float(current[i, 0]), _float(current[i, 1]))
        value = _add(value, _float(current[i, 2]))
        if i == 0:
            increment = _mul(scale, _add(selected_start_gamma, selected_end_gamma))
        else:
            increment = _mul(
                _mul(scale, _float(C_MMNS)), _add(start[i - 1], end[i - 1])
            )
        value = _add(value, increment)
        remainder = _add(value, _neg(_float(rounded[i])))
        low, ok = _round(remainder)
        if not ok:
            return output, False
        tail, ok = _round(_add(remainder, _neg(_float(low))))
        if not ok:
            return output, False
        output[1 + i], output[5 + i] = low, tail
    return output, True
