"""Direct dipole derivatives with integrated lower orders and endpoint drift checks."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import comb, factorial
from typing import TYPE_CHECKING

import numpy as np

from .compensated_history import CompensatedHistoryMap
from .preserved_history_map import _lagrange

if TYPE_CHECKING:
    from .full_dipole_history import FullDipoleHistory, FullDipoleSegment


class SourceDipoleError(ValueError):
    """Reconstructed dipole differs from an accepted endpoint beyond its budget."""


@lru_cache(maxsize=2048)
def _direct_map(
    times: tuple[float, ...], start: float, width: float
) -> CompensatedHistoryMap:
    """Reuse immutable weights when a checkpoint rebuilds the same saved interval."""
    nodes = [(Fraction(t) - Fraction(start)) / Fraction(width) for t in times]
    return CompensatedHistoryMap(
        np.array(
            [
                [v / factorial(k) for v in _lagrange(nodes, Fraction(0), k)]
                for k in range(11)
            ],
            dtype=object,
        )
    )


def connected_coefficients(
    history: FullDipoleHistory, left: int, previous: FullDipoleSegment | None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return absolute coefficients, small-change coefficients, and endpoint error.

    Only coefficients zero through three are adjusted. The represented change
    is relative to the declared startup dipole, not the large absolute tensor.
    """
    assert history.inertial_until is not None and history.dipole_tolerance is not None
    assert history.dipole_high is not None and history.dipole_low is not None
    start = int(np.searchsorted(history.time, history.inertial_until))
    knot = left + 1 if left == start else left
    selected = history._derivative_indices(knot)
    if selected is None:
        raise ValueError("Connected dipoles require accepted fitting samples")
    width = history.time[left + 1] - history.time[left]
    constrained = (
        history.startup_dipole_fit == "constrained"
        and history.time[selected[0]] == history.inertial_until
    )
    mapping = (
        _startup_map(
            tuple(history.time[selected]),
            history.inertial_until,
            history.time[left],
            width,
        )
        if constrained
        else _direct_map(tuple(history.time[selected]), history.time[left], width)
    )
    high, low = mapping.apply(
        history.dipole_high[selected],
        history.dipole_low[selected],
        reference_index=0 if constrained else int(np.searchsorted(selected, left)),
    )
    change = high + low
    for order in range(4):
        value = np.zeros((4, 4))
        if previous is not None and previous.dipole_change is not None:
            value = (
                np.polynomial.polynomial.polyval(
                    1.0,
                    np.polynomial.polynomial.polyder(
                        previous.dipole_change, order, axis=0
                    ),
                )
                / previous.duration**order
            )
        change[order] = value * width**order / factorial(order)
    delta_map = CompensatedHistoryMap([[-1, 1]])
    high, low = delta_map.apply(
        history.dipole_high[[start, left + 1]],
        history.dipole_low[[start, left + 1]],
        reference_index=0,
    )
    error = float(
        np.linalg.norm(np.polynomial.polynomial.polyval(1.0, change) - (high + low)[0])
    )
    if error > history.dipole_tolerance:
        raise SourceDipoleError(
            f"Connected dipole endpoint drift {error:.6e} exceeds budget {history.dipole_tolerance:.6e}"
        )
    absolute = change.copy()
    absolute[0] += history.dipole[start]
    return absolute, change, error


@lru_cache(maxsize=32)
def _startup_weights(times: tuple[float, ...], origin: float):
    """Exact least-squares weights for powers four through ten at startup.

    The first four coefficients are known from the inertial past. Rational
    elimination avoids loss of precision in the small polynomial fit; the
    resulting map still evaluates preserved samples with compensated arithmetic.
    """
    span = Fraction(times[-1]) - Fraction(origin)
    nodes = [(Fraction(t) - Fraction(origin)) / span for t in times]
    design = [[node**k for k in range(4, 11)] for node in nodes]
    augmented = [
        [sum(row[i] * row[j] for row in design) for j in range(7)]
        + [row[i] for row in design]
        for i in range(7)
    ]
    for pivot in range(7):
        divisor = augmented[pivot][pivot]
        if not divisor:
            raise ValueError("Singular constrained startup fit")
        augmented[pivot] = [value / divisor for value in augmented[pivot]]
        for row in range(7):
            if row != pivot:
                factor = augmented[row][pivot]
                augmented[row] = [
                    a - factor * b for a, b in zip(augmented[row], augmented[pivot])
                ]
    return tuple(tuple(row[7:]) for row in augmented)


@lru_cache(maxsize=128)
def _startup_map(times: tuple[float, ...], origin: float, left: float, width: float):
    fitted = _startup_weights(times, origin)
    span = Fraction(times[-1]) - Fraction(origin)
    offset = (Fraction(left) - Fraction(origin)) / span
    step = Fraction(width) / span
    transform = [
        [
            (
                Fraction(0)
                if degree < order
                else comb(degree, order) * offset ** (degree - order) * step**order
            )
            for degree in range(4, 11)
        ]
        for order in range(11)
    ]
    return CompensatedHistoryMap(
        np.array(
            [
                [
                    sum(row[j] * fitted[j][sample] for j in range(7))
                    for sample in range(len(times))
                ]
                for row in transform
            ],
            dtype=object,
        )
    )
