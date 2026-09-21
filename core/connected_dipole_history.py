"""Direct dipole derivatives with integrated lower orders and endpoint drift checks."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial
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
    mapping = _direct_map(tuple(history.time[selected]), history.time[left], width)
    high, low = mapping.apply(
        history.dipole_high[selected],
        history.dipole_low[selected],
        reference_index=int(np.searchsorted(selected, left)),
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
