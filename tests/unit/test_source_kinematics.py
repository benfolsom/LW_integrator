from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.source_kinematics import reconstruct_instantaneous_beta_prime_per_mm


def test_nonuniform_quadratic_beta_derivative_is_exact_at_every_knot() -> None:
    times_ns = np.asarray((-0.7, -0.31, -0.08, 0.19, 0.63), dtype=float)
    coordinate_mm = C_MMNS * times_ns
    linear = np.asarray((1.1e-4, -0.8e-4, 0.3e-4))
    quadratic = np.asarray((0.7e-7, 0.2e-7, -0.4e-7))
    beta = (
        np.asarray((0.12, -0.03, 0.01))[None, :]
        + coordinate_mm[:, None] * linear[None, :]
        + coordinate_mm[:, None] ** 2 * quadratic[None, :]
    )
    expected = linear[None, :] + 2.0 * coordinate_mm[:, None] * quadratic[None, :]

    actual = reconstruct_instantaneous_beta_prime_per_mm(times_ns, beta)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-17)


def test_two_samples_reduce_to_the_interval_secant() -> None:
    times_ns = np.asarray((0.2, 0.7))
    beta = np.asarray(((0.1, 0.0, -0.02), (0.13, 0.01, -0.01)))
    expected = (beta[1] - beta[0]) / (C_MMNS * (times_ns[1] - times_ns[0]))

    actual = reconstruct_instantaneous_beta_prime_per_mm(times_ns, beta)

    np.testing.assert_allclose(actual[0], expected, rtol=4.0e-16, atol=0.0)
    np.testing.assert_allclose(actual[1], expected, rtol=4.0e-16, atol=0.0)


@pytest.mark.parametrize(
    ("times", "beta", "message"),
    (
        (np.asarray((0.0, 0.0)), np.zeros((2, 3)), "increase strictly"),
        (np.asarray((0.0,)), np.zeros((2, 3)), "shape"),
        (np.asarray((0.0,)), np.asarray(((1.0, 0.0, 0.0),)), "below one"),
    ),
)
def test_invalid_source_samples_fail_closed(
    times: np.ndarray,
    beta: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reconstruct_instantaneous_beta_prime_per_mm(times, beta)
