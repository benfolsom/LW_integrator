"""Regression for roundoff in nearly lightlike ballistic source histories."""

import math

import numpy as np
import pytest

from core import retarded_fields
from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.exact_retarded_numba import NUMBA_AVAILABLE
from core.retarded_fields import ObserverEvent, evaluate_retarded_charge_field_native

pytestmark = pytest.mark.unit


def _history(times: np.ndarray, positions: np.ndarray, betas: np.ndarray):
    return [
        {
            "t": np.array([time]),
            "x": np.array([position[0]]),
            "y": np.array([position[1]]),
            "z": np.array([position[2]]),
            "bx": np.array([beta[0]]),
            "by": np.array([beta[1]]),
            "bz": np.array([beta[2]]),
            "bdotx": np.array([0.0]),
            "bdoty": np.array([0.0]),
            "bdotz": np.array([0.0]),
            "q": np.array([-ELEMENTARY_CHARGE]),
            "q_source": np.array([-ELEMENTARY_CHARGE]),
            "_dead_particles": np.array([False]),
        }
        for time, position, beta in zip(times, positions, betas)
    ]


@pytest.mark.parametrize("backend", ("python", "numba_full_strict_serial"))
@pytest.mark.parametrize("semantics", ("preceding_interval", "instantaneous"))
@pytest.mark.parametrize("u", (0.0, 0.3, 3.0))
@pytest.mark.parametrize("x", (-25.0, -28.0, -32.0))
def test_ballistic_field_and_acceleration(backend, semantics, u, x):
    if backend == "numba_full_strict_serial" and not NUMBA_AVAILABLE:
        pytest.skip("Numba is unavailable")

    radius = 1.0
    tau = radius / C_MMNS
    gamma = 1.0 + 3000.0 / 0.938272
    speed = math.sqrt(1.0 - gamma**-2)
    impact = u * radius / gamma
    quadratic = C_MMNS**2 * (1.0 - speed**2)
    linear = 2.0 * speed * C_MMNS * radius
    constant = -(impact**2 + radius**2)
    t_news = (
        -2.0 * constant / (linear + math.sqrt(linear**2 - 4.0 * quadratic * constant))
    )
    t_start = t_news - math.exp(15.5) * tau
    t_lo = 1.2 * ((1.0 + speed) * t_start - tau) / (1.0 - speed)
    times = np.concatenate(
        (-np.geomspace(abs(t_lo), 1e-8 * tau, 160), np.linspace(0.0, tau, 8))
    )
    positions = np.zeros((times.size, 3))
    positions[:, 2] = radius - speed * C_MMNS * times
    betas = np.zeros_like(positions)
    betas[:, 2] = -speed
    history = _history(times, positions, betas)

    t_event = t_news - math.exp(x) * tau
    event = ObserverEvent(t_event, (impact, 0.0, speed * C_MMNS * t_event))
    zeta = 2.0 * speed * C_MMNS * t_event - radius
    expected_ez = (
        -ELEMENTARY_CHARGE
        * (1.0 - speed**2)
        * zeta
        / (zeta**2 + (impact / gamma) ** 2) ** 1.5
    )
    field = evaluate_retarded_charge_field_native(
        history,
        event,
        backend=backend,
        source_acceleration_semantics=semantics,
    )
    assert abs(field.electric_field_native[2] / expected_ez - 1.0) < 1e-6

    arrays = retarded_fields._extract_history(history)
    source = retarded_fields._prepare_source_history(
        arrays, 0, source_acceleration_semantics=semantics
    )
    retarded_time = float(field.retarded_time_ns[0])
    segment = int(np.searchsorted(source.time_ns, retarded_time, side="right") - 1)
    _, _, beta_prime = retarded_fields._quintic_worldline_sample(
        source, segment, retarded_time
    )
    np.testing.assert_array_equal(beta_prime, 0.0)


def test_smooth_acceleration_keeps_analytic_worldline():
    times = np.array((-0.01, 0.0, 0.01))
    coordinate = C_MMNS * times
    cubic = 1e-4
    positions = np.zeros((times.size, 3))
    positions[:, 2] = 0.2 * coordinate + cubic * coordinate**3
    betas = np.zeros_like(positions)
    betas[:, 2] = 0.2 + 3.0 * cubic * coordinate**2
    beta_primes = np.zeros_like(positions)
    beta_primes[:, 2] = 6.0 * cubic * coordinate

    durations, coefficients = retarded_fields._quintic_position_coefficients_mm(
        times, positions, betas, beta_primes
    )
    for segment, duration in enumerate(durations):
        fraction = 0.4
        s = coordinate[segment] + fraction * C_MMNS * duration
        powers = fraction ** np.arange(6)
        actual_z = float(coefficients[segment, :, 2] @ powers)
        expected_z = 0.2 * s + cubic * s**3
        assert actual_z == pytest.approx(expected_z, rel=1e-13, abs=1e-15)
        assert np.any(coefficients[segment, 2:, 2] != 0.0)
