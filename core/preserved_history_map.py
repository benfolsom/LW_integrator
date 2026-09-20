"""General accepted-window dipole Hermite maps, formed with rational weights.

Same degree-ten endpoint interpolation and degree-seven joined dipole as the
ordinary history, with no floating-point cancellation during map construction.
"""

from fractions import Fraction as Q
from functools import lru_cache
from math import factorial

import numpy as np

from .compensated_history import CompensatedHistoryMap


def _lagrange(nodes, point, derivative):
    result = []
    for i, node in enumerate(nodes):
        coefficients = [Q(1)]
        denominator = Q(1)
        for j, other in enumerate(nodes):
            if i == j:
                continue
            coefficients = [Q(0), *coefficients]
            for k in range(len(coefficients) - 1):
                coefficients[k] -= other * coefficients[k + 1]
            # The in-place loop uses the untouched next coefficient.
            denominator *= node - other
        result.append(
            sum(
                coefficients[k]
                * factorial(k)
                / factorial(k - derivative)
                * point ** (k - derivative)
                for k in range(derivative, len(coefficients))
            )
            / denominator
        )
    return result


@lru_cache(maxsize=4096)
def segment_map(times, left, first_left, first_right, inertial_left, inertial_right):
    """Return a map from the specified accepted samples to normalized D(s)."""
    return indexed_segment_map(
        times,
        left,
        tuple(range(first_left, first_left + 11)),
        tuple(range(first_right, first_right + 11)),
        inertial_left,
        inertial_right,
    )


@lru_cache(maxsize=4096)
def indexed_segment_map(
    times, left, indices_left, indices_right, inertial_left, inertial_right
):
    """Same endpoint polynomial, allowing nonconsecutive accepted samples."""
    width = Q(times[left + 1]) - Q(times[left])
    nodes = [(Q(t) - Q(times[left])) / width for t in times]
    endpoints = []
    for knot, indices, inertial in (
        (left, indices_left, inertial_left),
        (left + 1, indices_right, inertial_right),
    ):
        rows = np.full((4, len(times)), Q(0), dtype=object)
        rows[0, knot] = Q(1)
        if not inertial:
            for order in range(1, 4):
                rows[order, list(indices)] = _lagrange(
                    [nodes[i] for i in indices], nodes[knot], order
                )
        endpoints.append(rows)
    a, b = endpoints
    coefficients = np.full((8, len(times)), Q(0), dtype=object)
    for k in range(4):
        coefficients[k] = a[k] / factorial(k)
    matrix = np.array(
        [[Q(factorial(j), factorial(j - k)) for j in range(4, 8)] for k in range(4)],
        dtype=object,
    )
    rhs = np.array(
        [
            b[k]
            - sum(
                factorial(j) // factorial(j - k) * coefficients[j] for j in range(k, 4)
            )
            for k in range(4)
        ]
    )
    for k in range(4):
        divisor = matrix[k, k]
        matrix[k] /= divisor
        rhs[k] /= divisor
        for j in range(4):
            if j != k:
                factor = matrix[j, k]
                matrix[j] -= factor * matrix[k]
                rhs[j] -= factor * rhs[k]
    coefficients[4:] = rhs
    return CompensatedHistoryMap(coefficients)
