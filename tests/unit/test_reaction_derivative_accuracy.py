"""Independent ordinary-motion derivatives and stable-startup regressions."""

from math import factorial

import numpy as np
import pytest

from core import momentum_center as m
from core.constants import C_MMNS as c
from core.full_dipole_reaction import ordinary_source_coefficients
from core.full_dipole_self_field import regular_self_field
from core.momentum_center_pair import smooth_start_native
from tests.unit.test_full_dipole_reaction import magnetic


def test_centered_full_spin_coefficients_against_uniform_field_solution():
    particle = m.Particle(charge=0.2, g=2)
    state = m.initial_state(
        np.zeros(4), [1.25, 0.75, 0, 0], [0.2, 0.1, 0.3], particle, magnetic
    )
    _, data = m.evaluate(state, particle, magnetic)
    generator = (
        particle.charge
        / data["kinetic_mass"]
        * magnetic(state[:4])[2]
        * m.METRIC[None, :]
    )
    exact_u = np.zeros((5, 4))
    exact_d = np.zeros((5, 4, 4))
    exact_u[0], exact_d[0] = data["proper_velocity"], data["proper_dipole"]
    for n in range(1, 5):
        exact_u[n] = generator @ exact_u[n - 1] / n
        exact_d[n] = (generator @ exact_d[n - 1] + exact_d[n - 1] @ generator.T) / n
    u, d, _ = ordinary_source_coefficients(state, particle, magnetic, window=0.5)
    for actual, expected in ((u, exact_u), (d, exact_d)):
        for n in range(1, 5):
            assert (
                np.linalg.norm(actual[n] - expected[n]) / np.linalg.norm(expected[n])
                < 1e-5
            )
    for n in range(1, 5):
        residual = sum(m.dot(u[k], u[n - k]) for k in range(n + 1))
        assert abs(residual) < 1e-15


def test_centered_derivatives_do_not_invent_missing_causal_sources():
    particle = m.Particle()
    state = m.initial_state(
        np.zeros(4), [1, 0, 0, 0], [0.2, 0.1, 0.3], particle, magnetic
    )

    def past_only(event):
        if event[0] > 0:
            raise ValueError("Source outside published history")
        return magnetic(event)

    with pytest.raises(ValueError, match="published history"):
        ordinary_source_coefficients(state, particle, past_only, window=0.1)


def test_c7_startup_has_stable_factored_derivatives_near_end():
    def constant(time, position):
        return (
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.zeros((4, 4)),
            np.zeros((4, 4)),
            np.zeros((4, 4, 4)),
        )

    provider = smooth_start_native(constant, 1 / c, 7)
    normalization = factorial(15) / factorial(7) ** 2
    for s in (0.01, 0.5, 0.99, 0.9999):
        _, da, _, df = provider(s / c, np.zeros(3))
        s = (s / c) / (1 / c)  # compare at the actual floating-point input coordinate
        first = normalization * (s * (1 - s)) ** 7
        second = normalization * 7 * (1 - 2 * s) * (s * (1 - s)) ** 6
        assert da[0, 1] == pytest.approx(first, rel=3e-13, abs=1e-30)
        assert df[0, 0, 1] == pytest.approx(second, rel=3e-13, abs=1e-30)
    for time in (-1.0, 0.0, 1 / c, 2 / c):
        _, da, _, df = provider(time, np.zeros(3))
        np.testing.assert_array_equal(da, 0)
        np.testing.assert_array_equal(df, 0)


def test_bound_momentum_derivative_on_exact_uniform_field_motion():
    from scipy.linalg import expm

    particle = m.Particle(charge=0.2, g=2)
    state = m.initial_state(
        np.zeros(4), [1.25, 0.75, 0, 0], [0.2, 0.1, 0.3], particle, magnetic
    )
    _, data = m.evaluate(state, particle, magnetic)
    generator = (
        particle.charge
        / data["kinetic_mass"]
        * magnetic(state[:4])[2]
        * m.METRIC[None, :]
    )

    def sample(tau):
        transform = expm(tau * generator)
        u = np.zeros((5, 4))
        d = np.zeros((5, 4, 4))
        u[0] = transform @ data["proper_velocity"]
        d[0] = transform @ data["proper_dipole"] @ transform.T
        for n in range(1, 5):
            u[n] = generator @ u[n - 1] / n
            d[n] = (generator @ d[n - 1] + d[n - 1] @ generator.T) / n
        return regular_self_field(u, d, charge=0.2)["source_balance"]

    h = 0.001
    left, center, right = sample(-h), sample(0), sample(h)
    for key in center:
        numeric = (right[key]["bound_momentum"] - left[key]["bound_momentum"]) / (2 * h)
        expected = center[key]["bound_momentum_rate"]
        if key == "qD":
            # This uniformly rotating charge/spin fixture has constant cross
            # bound momentum. Use an absolute zero control in fixture units.
            np.testing.assert_allclose(numeric, 0, atol=1e-16)
            np.testing.assert_allclose(expected, 0, atol=1e-16)
        else:
            assert np.linalg.norm(numeric - expected) / np.linalg.norm(expected) < 1e-7
