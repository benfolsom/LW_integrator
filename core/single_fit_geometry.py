"""Causal fixed-spacing samples and connected single-fit source motion."""

from math import factorial
import numpy as np


def fixed_spacing_indices(history, knot):
    spacing = history.fit_sample_spacing
    # Repeated timestep addition accumulates rounding in accepted timestamps.
    # Bound clock matching by that arithmetic scale, not a physical fit budget.
    tolerance = (
        8
        * np.finfo(float).eps
        * len(history.time)
        * max(float(np.max(np.abs(history.time))), spacing)
    )
    if tolerance >= 1e-6 * spacing:
        raise ValueError("Accepted time precision cannot resolve fixed fit spacing")
    elapsed = (history.time[knot] - history.inertial_until) / spacing
    # Anchor both grids to the same physical clock, rather than shifting the
    # fitting window at every finer recording point.
    nearest = np.rint(elapsed)
    tick = (
        nearest if abs(elapsed - nearest) * spacing <= tolerance else np.floor(elapsed)
    )
    origin = history.inertial_until + (tick - 5) * spacing
    origin = max(origin, history.inertial_until)
    targets = origin + np.arange(11) * spacing
    if targets[-1] > history.time[-1] + tolerance:
        return None
    right = np.minimum(np.searchsorted(history.time, targets), len(history.time) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(
        abs(history.time[left] - targets) < abs(history.time[right] - targets),
        left,
        right,
    )
    if len(np.unique(indices)) != 11 or np.any(
        abs(history.time[indices] - targets) > tolerance
    ):
        raise ValueError("Fixed fit spacing requires matching accepted recording times")
    return indices


def connected_position(history, left, previous):
    knot = left + 1 if history.time[left] == history.inertial_until else left
    selection = history._derivative_indices(knot)
    if selection is None:
        raise ValueError("Single-fit geometry requires accepted samples")
    width = history.time[left + 1] - history.time[left]
    times = history.time[selection] - history.time[left]
    center = (times[0] + times[-1]) / 2
    scale = (times[-1] - times[0]) / 2
    matrix = np.polynomial.polynomial.polyvander((times - center) / scale, 10)
    changes = history.velocity[selection] - history.velocity[left]
    coefficients = np.linalg.lstsq(matrix, changes, rcond=None)[0]
    position = np.zeros((12, 3))
    for order in range(11):
        value = (
            np.polynomial.polynomial.polyval(
                -center / scale,
                np.polynomial.polynomial.polyder(coefficients, order, axis=0),
            )
            / scale**order
        )
        if order == 0:
            value += history.velocity[left]
        position[order + 1] = value * width ** (order + 1) / factorial(order + 1)
    # Preserve x and its first four derivatives across each published boundary.
    for order in range(5):
        if previous is not None:
            value = previous.sample(previous.end, order)[0]
        else:
            value = (
                history.position[left]
                if order == 0
                else history.velocity[left] if order == 1 else np.zeros(3)
            )
        position[order] = value * width**order / factorial(order)
    return position
