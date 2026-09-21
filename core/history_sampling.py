"""Deterministic accepted-sample selection for nonuniform source histories.

The earliest sufficient time prefix defines every selection. Appending later
samples cannot change it. No state interpolation or extrapolation occurs here.
"""

import numpy as np
from numpy.typing import ArrayLike


def _spread(times: np.ndarray, first: int, last: int, targets: np.ndarray) -> list[int]:
    result: list[int] = []
    for i, target in enumerate(targets):
        low = first if not result else result[-1] + 1
        high = last - (len(targets) - i - 1)
        index = int(np.clip(np.searchsorted(times, target), low, high))
        if index > low and abs(times[index - 1] - target) <= abs(times[index] - target):
            index -= 1
        result.append(index)
    return result


def time_balanced_indices(
    times: ArrayLike, knot: int, minimum_index: int = 0
) -> np.ndarray | None:
    """Eleven samples around a knot, or None until enough future data exists.

    The first five neighbours on each side define the target radius. On the
    more densely sampled side, choose points spread over that radius instead
    of five adjacent points. Retain the established one-sided startup window
    for its first five knots; it must still pass the native fit checks.
    """
    times = np.asarray(times)
    if knot - minimum_index < 5:
        if minimum_index + 11 > len(times):
            return None
        return np.arange(minimum_index, minimum_index + 11)
    if knot + 5 >= len(times):
        return None
    radius = max(times[knot] - times[knot - 5], times[knot + 5] - times[knot])
    # Avoid turning an exact uniform-grid endpoint into an extra wait through
    # subtraction roundoff. This never creates an unrecorded sample.
    tolerance = 8 * np.finfo(float).eps * max(abs(times[knot]), radius)
    first = max(
        minimum_index,
        int(np.searchsorted(times, times[knot] - radius + tolerance, side="right")) - 1,
    )
    last = int(np.searchsorted(times, times[knot] + radius - tolerance))
    if last >= len(times):
        return None
    first = min(first, knot - 5)
    last = max(last, knot + 5)
    left = _spread(
        times, first, knot - 1, np.linspace(times[first], times[knot], 6)[:-1]
    )
    right = _spread(times, knot + 1, last, np.linspace(times[knot], times[last], 6)[1:])
    return np.array([*left, knot, *right])
