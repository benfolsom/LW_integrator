"""Rest-spin rotation equivalent to covariant rotation-free velocity transport."""

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.experimental_spin_reaction import (
    transport_rest_spin_for_impulse_native,
    transport_spin_between_velocities,
)
from core.magnetic_dipole import (
    boost_rest_polarization,
    rest_polarization_from_four_vector,
)


def test_rest_rotation_matches_independent_four_vector_transport():
    rng = np.random.default_rng(715)
    mass = 1.3
    for _ in range(100):
        w, delta_w, spin = rng.normal(size=(3, 3))
        w *= 3
        gamma = np.hypot(1, np.linalg.norm(w))
        new_gamma = np.hypot(1, np.linalg.norm(w + delta_w))
        before, after = c * np.r_[gamma, w], c * np.r_[new_gamma, w + delta_w]
        lab_spin = boost_rest_polarization(spin, w / gamma)
        expected = rest_polarization_from_four_vector(
            transport_spin_between_velocities(lab_spin, before, after),
            (w + delta_w) / new_gamma,
        )
        actual = transport_rest_spin_for_impulse_native(
            spin, mass * c * w, mass * c * delta_w, mass
        )
        np.testing.assert_allclose(
            actual, expected, rtol=1e-12, atol=1e-13 * np.linalg.norm(spin)
        )
        assert np.linalg.norm(actual) == pytest.approx(np.linalg.norm(spin), rel=8e-16)


@pytest.mark.parametrize("proper_speed", [0, 1, 70, 1e6])
def test_repeated_inverse_kicks_do_not_accumulate_boost_subtraction_error(proper_speed):
    spin = np.array([0.3, 0.4, 0.5])
    original = spin.copy()
    mass = 0.001
    momentum = mass * c * np.array([proper_speed, 0, 0.0])
    impulse = mass * c * np.array([0, 1e-5, -3e-6])
    for _ in range(1000):
        spin = transport_rest_spin_for_impulse_native(spin, momentum, impulse, mass)
        spin = transport_rest_spin_for_impulse_native(
            spin, momentum + impulse, -impulse, mass
        )
    np.testing.assert_allclose(spin, original, rtol=0, atol=2e-13)


def test_small_transverse_kick_has_the_thomas_rotation_sign():
    gamma, delta_w = 7.0, np.array([0, 1e-7, 0])
    w = np.array([np.sqrt(gamma**2 - 1), 0, 0])
    spin = np.array([1, 0, 0.0])
    actual = transport_rest_spin_for_impulse_native(spin, c * w, c * delta_w, 1.0)
    rotation = -np.cross(w, delta_w) / (gamma + 1)
    expected = spin + np.cross(rotation, spin)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-15)


def test_zero_and_collinear_impulses_preserve_rest_components_exactly():
    spin = np.array([0.3, -0.2, 0.7])
    p = np.array([1e6, 0, 0])
    for impulse in (np.zeros(3), np.array([100, 0, 0])):
        np.testing.assert_array_equal(
            transport_rest_spin_for_impulse_native(spin, p, impulse, 1), spin
        )


@pytest.mark.parametrize("mass", [0, -1, np.inf, np.nan])
def test_invalid_mass_rejected(mass):
    with pytest.raises(ValueError, match="mass"):
        transport_rest_spin_for_impulse_native([1, 0, 0], [1, 0, 0], [0, 0, 0], mass)
