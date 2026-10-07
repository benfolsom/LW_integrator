"""Compiled polynomial work with the reference derivative normalization.

The Python sampler is still the authority. In particular, LLVM's expansion of
``length**3`` can round differently from Python's power operation. Keep powers
and NumPy division outside the compiled Horner loops. The residual dispatcher
is the same strict arithmetic already used by the compiled root solver.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .light_cone_roots import null_residual

null_residual_strict = njit(cache=True, fastmath=False)(null_residual)


@njit(cache=True, fastmath=False)
def _sample_polynomials(
    coefficients: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value = np.zeros(3)
    first = np.zeros(3)
    second = np.zeros(3)
    third = np.zeros(3)
    for component in range(3):
        for order in range(5, -1, -1):
            value[component] = value[component] * tau + coefficients[order, component]
        for order in range(5, 0, -1):
            first[component] = (
                first[component] * tau + order * coefficients[order, component]
            )
        for order in range(5, 1, -1):
            second[component] = (
                second[component] * tau
                + order * (order - 1) * coefficients[order, component]
            )
        for order in range(5, 2, -1):
            third[component] = (
                third[component] * tau
                + order * (order - 1) * (order - 2) * coefficients[order, component]
            )
    return value, first, second, third


def sample_null_quintic_strict(
    coefficients: np.ndarray, length: float, tau: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retain the original float64 powers and array division rounding."""
    value, first, second, third = _sample_polynomials(coefficients, tau)
    return value, first / length, second / length**2, third / length**3
