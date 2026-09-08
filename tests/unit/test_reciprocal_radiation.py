"""Calibrate the reciprocal measurement against prescribed classical sources."""

import numpy as np
import pytest
from core.constants import C_MMNS as c
from core.jakobsen_pair import FrozenSource
from core.jakobsen_step import JakobsenParticle
from core.reciprocal_radiation import (
    ReciprocalRadiationSampler,
    amplitude_flux,
    charge_radiation_amplitudes,
)
from core.radiation_flux_oracle import gauss_legendre_sphere_quadrature


def reference_source(*, beta=0.0, acceleration=0.0, mu_second=0.0, sign=1.0):
    # q=m=1 and g=2c makes the rest-spin coefficients equal moment coefficients.
    p = JakobsenParticle(1.0, 1.0, 2 * c)

    def row(t):
        return np.r_[
            t,
            [beta * c * t + 0.5 * acceleration * t * t, 0, 0],
            [beta + acceleration * t / c, 0, 0],
            [acceleration / c**2, 0, 0],
            [0, 0, sign * (1.0 + 0.5 * mu_second * t * t)],
            [0, 0, sign * mu_second * t],
        ]

    source = FrozenSource(p, [row(-0.1)], [], [])
    source.append(row(0.1))
    if not mu_second:
        source.spin_segments = [np.zeros((4, 3))]
    return source


def measure(sources, order=8, radius=256.0):
    quad = gauss_legendre_sphere_quadrature(
        polar_order=order, azimuthal_order=2 * order
    )
    amps = ReciprocalRadiationSampler(sources).amplitudes(
        cut_time_ns=0.0, radius_mm=radius, quadrature=quad
    )
    return {degree: amplitude_flux(a, quad) for degree, a in amps.items()}


@pytest.mark.parametrize("beta", [0.0, 0.8])
def test_uniform_charge_has_no_radiation(beta):
    result = measure([reference_source(beta=beta)])[2]
    # Absolute native power tolerance; no relative error against zero.
    assert abs(result.total.energy_rate_native) < 1e-20


@pytest.mark.parametrize("order", [4, 8])
@pytest.mark.parametrize("radius", [128.0, 256.0])
def test_accelerated_charge_matches_instantaneous_rest_larmor(order, radius):
    result = measure([reference_source(acceleration=0.8)], order, radius)[2]
    expected = 2 * 0.8**2 / (3 * c**3)
    assert abs(result.q_squared.energy_rate_native / expected - 1) < 1e-3
    assert np.linalg.norm(result.q_squared.momentum_rate_native) * c / expected < 1e-6


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_stationary_changing_dipole_matches_radiation_and_interference(sign):
    sources = [
        reference_source(acceleration=0.8),
        reference_source(mu_second=2.4, sign=sign),
    ]
    result = measure(sources)[2]
    expected = 2 * 2.4**2 / (3 * c**3)
    assert abs(result.mu_squared.energy_rate_native / expected - 1) < 1e-3
    cross = 2 * np.cross([0.8, 0, 0], [0, 0, sign * 2.4]) / (3 * c**4)
    np.testing.assert_allclose(
        result.q_mu_interference.momentum_rate_native, cross, rtol=1e-3, atol=1e-18
    )
    assert abs(result.q_mu_interference.energy_rate_native) < expected * 1e-6


def test_integrated_accelerating_charge_matches_direct_far_formula():
    from scipy.integrate import simpson

    quad = gauss_legendre_sphere_quadrature(polar_order=8, azimuthal_order=16)
    source = reference_source(beta=0.3, acceleration=8.0)
    sampler = ReciprocalRadiationSampler([source])
    estimates = []
    references = []
    times = np.linspace(-0.015, 0.015, 17)
    for t in times:
        estimates.append(
            amplitude_flux(
                sampler.amplitudes(cut_time_ns=t, radius_mm=256.0, quadrature=quad)[2],
                quad,
            ).total.energy_rate_native
        )
        references.append(
            amplitude_flux(
                charge_radiation_amplitudes([source], cut_time_ns=t, quadrature=quad),
                quad,
            ).total.energy_rate_native
        )
    assert abs(simpson(estimates, x=times) / simpson(references, x=times) - 1) < 1e-3


def test_emission_time_weight_matches_relativistic_lienard_power():
    source = reference_source(beta=0.8, acceleration=0.8)
    quad = gauss_legendre_sphere_quadrature(polar_order=32, azimuthal_order=64)
    actual = (
        ReciprocalRadiationSampler([source])
        .emission_self_flux(emission_time_ns=0.0, radius_mm=256.0, quadrature=quad)[2][
            0
        ]
        .q_squared
    )
    gamma = 1 / np.sqrt(1 - 0.8**2)
    expected = 2 * gamma**6 * 0.8**2 / (3 * c**3)
    assert abs(actual.energy_rate_native / expected - 1) < 1e-3
    np.testing.assert_allclose(
        actual.momentum_rate_native, [expected * 0.8 / c, 0, 0], rtol=1e-3, atol=1e-18
    )
