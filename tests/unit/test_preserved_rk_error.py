"""Independent accuracy checks for the mixed-method adaptive building block."""

import numpy as np
import pytest

from core.preserved_rk_error import step_doubling


@pytest.mark.parametrize("method,order,calls", [("rk3", 3, 9), ("rk4", 4, 12)])
def test_nonlinear_error_estimate_and_expected_order(method, order, calls):
    reference = np.array([0.0, 1.0])
    zero = np.zeros(2)
    errors = []
    for width in (0.02, 0.01):
        high, low, estimate, count = step_doubling(
            reference,
            zero,
            zero,
            width,
            lambda state: np.array([1.0, state[1] ** 2]),
            method=method,
        )
        actual = high[1] + low[1] - width / (1 - width)
        assert count == calls
        assert estimate[0] == 0
        assert estimate[1] / actual == pytest.approx(1, rel=0.06)
        errors.append(abs(actual))
    assert errors[0] / errors[1] == pytest.approx(2 ** (order + 1), rel=0.1)
    np.testing.assert_array_equal(zero, 0)


@pytest.mark.parametrize("method", ["rk3", "rk4"])
def test_estimates_tiny_changes_and_integrates_reaction_with_same_weights(method):
    reference = np.array([0.0, 1.0, 0.0])
    zero = np.zeros(3)
    high, low, error, _ = step_doubling(
        reference,
        zero,
        zero,
        0.1,
        lambda state: np.array([1.0, 1e-23 * state[0] ** 4, 2e-23 * state[0] ** 4]),
        method=method,
    )
    assert high[1] > 0 and reference[1] + high[1] == 1
    assert error[1] != 0
    assert high[2] == 2 * high[1]
    assert low[2] == 2 * low[1]
    assert error[2] == 2 * error[1]
    actual = high[1] + low[1] - 1e-23 * 0.1**5 / 5
    # For a time-only rate both formulas reduce to Simpson quadrature (order
    # four). RK3's general third-order denominator therefore overestimates this
    # special case by 15/7; it must not be silently promoted to fourth order.
    ratio = 15 / 7 if method == "rk3" else 1
    assert error[1] / actual == pytest.approx(ratio, rel=1e-12)


@pytest.mark.parametrize("method", ["rk3", "rk4"])
def test_rejected_stage_leaves_inputs_unchanged(method):
    reference, high, low = np.array([0.0, 1.0]), np.zeros(2), np.zeros(2)
    original = [value.copy() for value in (reference, high, low)]

    def rhs(state):
        if state[0] > 0:
            raise ValueError("physical domain")
        return np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="physical domain"):
        step_doubling(reference, high, low, 0.1, rhs, method=method)
    for value, saved in zip((reference, high, low), original):
        np.testing.assert_array_equal(value, saved)


@pytest.mark.parametrize("rate", [[0, 1], [1, np.nan], [1]])
def test_invalid_rates_are_rejected(rate):
    zero = np.zeros(2)
    with pytest.raises(ValueError, match="Finite matching rate"):
        step_doubling(zero, zero, zero, 0.1, lambda state: rate, method="rk3")


def test_unrepresentable_half_step_is_rejected():
    reference = np.array([1.0, 0.0])
    zero = np.zeros(2)
    with pytest.raises(ValueError, match="distinct finite laboratory times"):
        step_doubling(reference, zero, zero, np.spacing(1.0), None, method="rk4")
