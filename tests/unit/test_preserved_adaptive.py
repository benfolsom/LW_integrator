"""Internal adaptive steps must preserve small changes and every integral."""

import numpy as np
import pytest
from scipy.integrate import DOP853

from core.preserved_adaptive import (
    InternalAccuracyError,
    embedded_error,
    integrate_interval,
)


def test_embedded_coefficients_match_independent_scipy_implementation():
    solver = DOP853(lambda t, y: y, 0, np.ones(3), 1)
    rates = np.random.default_rng(14).normal(size=(13, 3))
    np.testing.assert_array_equal(
        embedded_error(rates, 0.125), solver._estimate_error(rates, 0.125)
    )


def test_tiny_motion_and_reaction_integral_have_matching_weights():
    reference = np.array([0.0, 1.0, 0.0])
    high, low = np.zeros(3), np.zeros(3)
    head, tail, stats = integrate_interval(
        reference,
        high,
        low,
        0.1,
        lambda y: np.array([1.0, 5e-23 * y[0] ** 4, 10e-23 * y[0] ** 4]),
        lambda old, new, err: np.max(abs(err)) / 1e-30,
        maximum_trials=1,
    )
    assert head[1] > 0 and reference[1] + head[1] == 1
    assert abs((head[1] + tail[1]) / 1e-28 - 1) < 1e-13
    assert head[2] == 2 * head[1] and tail[2] == 2 * tail[1]
    assert stats["accepted_steps"] > 0
    np.testing.assert_array_equal(high, 0)
    np.testing.assert_array_equal(low, 0)


def test_piecewise_rate_can_be_resolved_without_publishing_extra_samples():
    ref = np.zeros(2)

    def rhs(y):
        return np.array([1.0, 1.0 if y[0] < 0.37 else 3.0])

    head, tail, stats = integrate_interval(
        ref,
        ref,
        ref,
        1.0,
        rhs,
        lambda old, new, err: np.max(abs(err))
        / (1e-10 + 1e-8 * max(abs(new[1]), abs(old[1]))),
    )
    assert abs(head[1] + tail[1] - (0.37 + 3 * 0.63)) < 1e-6
    assert stats["rejected_trials"] > 0
    assert stats["accepted_steps"] > 1


def test_failure_cannot_modify_input():
    reference, high, low = np.zeros(2), np.zeros(2), np.zeros(2)

    def fail(y):
        raise ValueError("physical domain")

    with pytest.raises(ValueError, match="physical domain"):
        integrate_interval(reference, high, low, 1, fail, lambda *args: 0)
    np.testing.assert_array_equal(reference, 0)
    np.testing.assert_array_equal(high, 0)
    np.testing.assert_array_equal(low, 0)


def test_trial_budget_retains_accuracy_evidence_without_accepting_state():
    state = np.zeros(2)
    with pytest.raises(InternalAccuracyError, match="after 2 trials") as caught:
        integrate_interval(
            state,
            state,
            state,
            1.0,
            lambda value: np.array([1.0, value[0]]),
            lambda old, new, error: 2.0,
            maximum_trials=2,
        )
    assert len(caught.value.reports) == 2
    assert all(not row["accepted"] for row in caught.value.reports)
    assert all(row["normalized_error"] == 2 for row in caught.value.reports)
    np.testing.assert_array_equal(state, 0)
