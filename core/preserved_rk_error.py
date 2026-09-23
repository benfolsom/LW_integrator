"""RK3/RK4 step-doubling estimates without publishing source histories.

This is the numerical building block for mixed-method internal error control,
not a replacement for the existing DOP853 controller. The caller supplies all
state and reaction rates in one vector so they receive identical weights.
"""

from decimal import localcontext
from typing import Callable

import numpy as np

from .preserved_source import decimal_array, rk_increment, rounded_state


def step_doubling(
    reference: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    width: float,
    rhs: Callable[[np.ndarray], np.ndarray],
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return two-half-step increments, their estimated error, and call count.

    The estimated error is (full step - two half steps)/(2**order - 1),
    computed from preserved increments rather than rounded absolute states.
    No extrapolation is applied. This estimate assumes a smooth right-hand side
    on the interval; source-history boundaries still require caller treatment.
    Coordinate zero is laboratory time, whose derivative must be exactly one.
    """
    if method not in ("rk3", "rk4"):
        raise ValueError("Step doubling requires rk3 or rk4")
    reference, high, low = [
        np.asarray(value, dtype=float).copy() for value in (reference, high, low)
    ]
    if (
        reference.ndim != 1
        or reference.size < 1
        or high.shape != reference.shape
        or low.shape != reference.shape
        or not np.isfinite([reference, high, low]).all()
    ):
        raise ValueError("Matching finite preserved vectors required")
    if not np.isscalar(width) or not np.isfinite(width) or width <= 0:
        raise ValueError("Positive finite step required")
    start = rounded_state(reference, high, low)[0]
    middle, end = start + width / 2, start + width
    if not start < middle < end or not np.isfinite(end):
        raise ValueError("Step halves require distinct finite laboratory times")
    evaluations = 0

    def rate(value: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        result = np.asarray(rhs(value), dtype=float)
        evaluations += 1
        if (
            result.shape != reference.shape
            or not np.isfinite(result).all()
            or result[0] != 1.0
        ):
            raise ValueError("Finite matching rate with clock derivative one required")
        return result.copy()

    def step(
        head: np.ndarray, tail: np.ndarray, begin: float, finish: float
    ) -> tuple[np.ndarray, np.ndarray]:
        span = finish - begin

        def stage(
            rates: tuple[np.ndarray, ...],
            weights: tuple[int, ...],
            divisor: int,
            time: float,
        ) -> np.ndarray:
            parts = rk_increment(
                head, tail, rates, span, weights=weights, divisor=divisor
            )
            value = rounded_state(reference, *parts)
            value[0] = time
            return rate(value)

        value = rounded_state(reference, head, tail)
        value[0] = begin
        k1 = rate(value)
        k2 = stage((k1,), (1,), 2, begin + span / 2)
        rates: tuple[np.ndarray, ...]
        weights: tuple[int, ...]
        if method == "rk3":
            k3 = stage((k1, k2), (-1, 2), 1, finish)
            rates, weights = (k1, k2, k3), (1, 4, 1)
        else:
            k3 = stage((k2,), (1,), 2, begin + span / 2)
            k4 = stage((k3,), (1,), 1, finish)
            rates, weights = (k1, k2, k3, k4), (1, 2, 2, 1)
        result_high, result_low = rk_increment(head, tail, rates, span, weights=weights)
        result_high[0], result_low[0] = finish - reference[0], 0.0
        return result_high, result_low

    full_high, full_low = step(high, low, start, end)
    half_high, half_low = step(high, low, start, middle)
    fine_high, fine_low = step(half_high, half_low, middle, end)
    order = 3 if method == "rk3" else 4
    with localcontext() as context:
        context.prec = 80
        error = np.asarray(
            (
                decimal_array(full_high)
                + decimal_array(full_low)
                - decimal_array(fine_high)
                - decimal_array(fine_low)
            )
            / (2**order - 1),
            dtype=float,
        )
    error[0] = 0.0
    return fine_high, fine_low, error, evaluations
