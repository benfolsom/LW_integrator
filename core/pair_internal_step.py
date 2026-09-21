"""Adaptive internal motion and reaction integration with no history publication."""

import numpy as np

from .pair_step_error import reaction_rate_vector, reaction_ledger, physical_error_norm
from .preserved_adaptive import integrate_interval
from .preserved_source import rounded_state


def integrate_particle(reference, high, low, width, particle, provider, settings):
    from .momentum_center_pair import dynamics_native

    reference, high, low = [
        np.r_[value, np.zeros(41)] for value in (reference, high, low)
    ]

    def rhs(value):
        rate, diagnostic = dynamics_native(value[:14], particle, provider)
        rate[0] = 1.0  # The lab-time parametrization has this exact clock rate.
        return np.r_[rate, reaction_rate_vector(diagnostic["length_time"])]

    high, low, statistics = integrate_interval(
        reference,
        high,
        low,
        width,
        rhs,
        lambda old, proposed, error: physical_error_norm(
            proposed, error, particle, provider, settings
        ),
        maximum_trials=settings.maximum_trials,
    )
    integrals = rounded_state(reference, high, low)[14:]
    return high[:14], low[:14], reaction_ledger(integrals), statistics
