"""Cached derivative bookkeeping preserves every order-three permutation."""

from itertools import product
from math import factorial

import numpy as np
import pytest

from core.dipole_hertz_jet import _INDEX
from core.response_taylor import ResponseTaylor, _derivative_location


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_all_derivative_orders_match_original_arithmetic(scale):
    data = np.random.default_rng(818).normal(size=(5, 35)) * scale
    value = ResponseTaylor(data)
    for order in range(4):
        for indices in product(range(4), repeat=order):
            alpha = [indices.count(i) for i in range(4)]
            expected = data[:, _INDEX[tuple(alpha)]] * np.prod(
                [factorial(n) for n in alpha]
            )
            actual = value.derivative(*indices)
            np.testing.assert_array_equal(actual, expected)
            actual[0] = 0
            np.testing.assert_array_equal(value.coefficients, data)
    assert _derivative_location.cache_info().maxsize == 128


def test_unsupported_derivative_order_is_not_silently_truncated():
    with pytest.raises(KeyError):
        ResponseTaylor.constant(1).derivative(0, 0, 0, 0)
