"""Preserved internal steps inside one source-publication interval.

The caller supplies an error norm with physical units and may append reaction
integrals to the state. All components use the same stage weights. This module
never publishes source history and never accepts a failed domain/root solve.
"""

import numpy as np
from scipy.integrate import DOP853

from .preserved_runge_kutta import dop853_step
from .preserved_source import rounded_state


class InternalAccuracyError(ValueError):
    """A refused interval with the completed local-error trials attached."""

    def __init__(self, message, reports):
        self.reports = [dict(row) for row in reports]
        super().__init__(message)


def embedded_error(rates, width):
    """DOP853 component error correction, using SciPy's published coefficients."""
    rates = np.asarray(rates)
    error5 = rates.T @ DOP853.E5
    error3 = rates.T @ DOP853.E3
    denominator = np.hypot(np.abs(error5), 0.1 * np.abs(error3))
    correction = np.divide(
        np.abs(error5), denominator, out=np.ones_like(error5), where=denominator > 0
    )
    return width * error5 * correction


def integrate_interval(
    reference,
    high,
    low,
    width,
    rhs,
    assess_error,
    *,
    maximum_trials=512,
    method="dop853",
):
    """Advance one interval with private adaptive internal steps.

    `assess_error(old, proposed, embedded_error)` returns a finite nonnegative
    normalized error; acceptance requires at most one. A component maximum,
    rather than an RMS over unrelated units, is the intended physical adapter.
    Coordinate zero is laboratory time in the caller's time units.
    """
    if method not in ("dop853", "rk3", "rk4"):
        raise ValueError("Internal method must be dop853, rk3, or rk4")
    error_order = {"dop853": 8, "rk3": 4, "rk4": 5}[method]
    reference, high, low = [
        np.asarray(v, dtype=float).copy() for v in (reference, high, low)
    ]
    if (
        reference.ndim != 1
        or high.shape != reference.shape
        or low.shape != reference.shape
        or not np.isfinite([reference, high, low]).all()
    ):
        raise ValueError("Matching finite preserved vectors required")
    if (
        not np.isfinite(width)
        or width <= 0
        or type(maximum_trials) is not int
        or maximum_trials < 1
    ):
        raise ValueError("Positive interval and integer trial budget required")
    initial = rounded_state(reference, high, low)
    end = initial[0] + width
    if end <= initial[0]:
        raise ValueError("Requested interval is below representable laboratory time")
    step = width
    reports = []
    accepted = rejected = evaluations = 0
    cached_rate = None
    for _ in range(maximum_trials):
        old = rounded_state(reference, high, low)
        if old[0] >= end:
            return (
                high,
                low,
                dict(
                    accepted_steps=accepted,
                    rejected_trials=rejected,
                    evaluations=evaluations,
                    reports=reports,
                ),
            )
        step = min(step, end - old[0])
        if step < 10 * abs(np.spacing(old[0])):
            raise InternalAccuracyError(
                "Internal accuracy requires an unrepresentable time step", reports
            )
        rates = []

        def stage(value):
            nonlocal evaluations, cached_rate
            if not rates and cached_rate is not None:
                rate = cached_rate.copy()
            else:
                rate = np.asarray(rhs(value))
                evaluations += 1
            if (
                rate.shape != reference.shape
                or not np.isfinite(rate).all()
                or rate[0] != 1.0
            ):
                raise ValueError(
                    "Finite matching rate with laboratory clock derivative one required"
                )
            rates.append(rate)
            return rate

        if method == "dop853":
            next_high, next_low, _ = dop853_step(reference, high, low, step, stage)
        else:
            from .preserved_rk_error import step_doubling

            next_high, next_low, error, calls = step_doubling(
                reference, high, low, step, rhs, method=method
            )
            evaluations += calls
        next_time = min(end, old[0] + step)
        next_high[0], next_low[0] = next_time - reference[0], 0.0
        proposed = rounded_state(reference, next_high, next_low)
        final_rate = np.asarray(rhs(proposed))
        evaluations += 1
        if (
            final_rate.shape != reference.shape
            or not np.isfinite(final_rate).all()
            or final_rate[0] != 1.0
        ):
            raise ValueError(
                "Finite matching rate with laboratory clock derivative one required"
            )
        if method == "dop853":
            error = embedded_error([*rates, final_rate], step)
        error[0] = 0.0  # Clock is set directly, not estimated by quadrature.
        norm = float(assess_error(old, proposed, error))
        if not np.isfinite(norm) or norm < 0:
            raise ValueError("Finite nonnegative internal error norm required")
        passed = norm <= 1
        reports.append(
            dict(
                start_time=float(old[0]),
                width=float(step),
                normalized_error=norm,
                accepted=passed,
            )
        )
        factor = (
            4.0 if norm == 0 else min(4.0, max(0.2, 0.9 * norm ** (-1 / error_order)))
        )
        if passed:
            high, low = next_high, next_low
            cached_rate = final_rate
            accepted += 1
            if next_time >= end:
                return (
                    high,
                    low,
                    dict(
                        accepted_steps=accepted,
                        rejected_trials=rejected,
                        evaluations=evaluations,
                        reports=reports,
                    ),
                )
        else:
            cached_rate = rates[0] if method == "dop853" else None
            rejected += 1
            factor = min(1.0, factor)
        step *= factor
    raise InternalAccuracyError(
        f"Internal adaptive trial budget exhausted after {maximum_trials} trials; "
        f"last normalized error {reports[-1]['normalized_error']:.6e}",
        reports,
    )
