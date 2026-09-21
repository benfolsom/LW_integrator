"""Fixed DOP853 stages with preserved state increments and exposed weights.

The caller supplies the full equations and integrates diagnostics with these
same weights. This is not SciPy's adaptive controller or dense-output history.
"""

from decimal import Decimal, localcontext

import numpy as np
from scipy.integrate import DOP853

from .preserved_source import decimal_array, split_array, rounded_state


def weighted_increment(high, low, rates, weights, width):
    if len(rates) != len(weights):
        raise ValueError("Every stage rate requires a matching weight")
    with localcontext() as context:
        context.prec = 80
        change = np.full(np.shape(high), Decimal(0), dtype=object)
        for rate, weight in zip(rates, weights):
            change += decimal_array(rate) * Decimal.from_float(float(weight))
        change *= Decimal.from_float(float(width))
        return split_array(decimal_array(high) + decimal_array(low) + change)


def dop853_step(reference, high, low, width, rhs):
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Positive finite DOP853 step required")
    state = rounded_state(reference, high, low)
    rates = []
    for index, fraction in enumerate(DOP853.C):
        stage_high, stage_low = weighted_increment(
            high, low, rates, DOP853.A[index, :index], width
        )
        trial = rounded_state(reference, stage_high, stage_low)
        trial[0] = state[0] + float(fraction) * width
        rates.append(rhs(trial))
    result = weighted_increment(high, low, rates, DOP853.B, width)
    return *result, tuple(float(w) for w in DOP853.B)
