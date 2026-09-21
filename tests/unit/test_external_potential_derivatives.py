"""Uniform prescribed potential derivatives and their exact leading motion."""

from dataclasses import fields

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.external_potential_derivatives import (
    supports_uniform_external_potential,
    uniform_external_potential_derivatives_native,
)
from core.magnetic_dipole import boost_rest_polarization
from core.potential_jet_rfs import potential_directional_rfs_reduction_jet_native
from core.rfs import electromagnetic_field_tensor_native
from core.types import ExternalFieldConfig


@pytest.mark.parametrize("beta", [0.02, 0.8, 0.9999])
def test_uniform_potential_reproduces_lorentz_motion_and_exact_derivatives(beta):
    electric = np.array([0.3, -0.2, 0.1])
    magnetic = np.array([-1.0, 0.7, 1.3])
    config = ExternalFieldConfig(
        electric_field_native=tuple(electric), magnetic_field_native=tuple(magnetic)
    )
    derivative = uniform_external_potential_derivatives_native(
        config, position_mm=(1, 2, 3)
    )
    velocity3 = beta * np.array([2 / 3, -2 / 3, 1 / 3])
    gamma = 1 / np.sqrt(1 - beta**2)
    velocity = c * gamma * np.r_[1, velocity3]
    spin = boost_rest_polarization((0, 0, 1), velocity3)
    charge, mass, invariant_spin = -0.7, 1.3, 0.8
    result = potential_directional_rfs_reduction_jet_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        partial_a=derivative.partial_a,
        partial2_a=derivative.partial2_a,
        partial3_a_along_velocity=derivative.partial3_a_along_velocity,
        partial3_a_along_acceleration=derivative.partial3_a_along_acceleration,
        partial4_a_along_velocity_twice=derivative.partial4_a_along_velocity_twice,
        charge_native=charge,
        mass_amu=mass,
        invariant_spin_native=invariant_spin,
        magnetic_moment_native=invariant_spin * charge / (mass * c),
    )
    force = (
        charge
        * gamma
        * np.r_[electric @ velocity3, electric + np.cross(velocity3, magnetic)]
    )
    np.testing.assert_allclose(result.four_acceleration, force / mass, rtol=2e-14)
    matrix = (
        charge
        / (mass * c)
        * electromagnetic_field_tensor_native(electric, magnetic)
        @ np.diag([1, -1, -1, -1])
    )
    np.testing.assert_allclose(
        result.four_jerk, matrix @ result.four_acceleration, rtol=2e-14
    )
    np.testing.assert_allclose(
        result.four_snap, matrix @ matrix @ result.four_acceleration, rtol=4e-14
    )
    np.testing.assert_allclose(
        result.normalized_spin_first_derivative,
        matrix @ spin,
        rtol=4e-14,
        atol=4e-15 * np.linalg.norm(matrix @ spin),
    )
    np.testing.assert_allclose(
        result.normalized_spin_second_derivative,
        matrix @ matrix @ spin,
        rtol=4e-14,
        atol=4e-15 * np.linalg.norm(matrix @ matrix @ spin),
    )


def test_potential_values_and_derivative_orientation_match_by_displacement():
    config = ExternalFieldConfig(
        electric_field_native=(0.3, -0.2, 0.1), magnetic_field_native=(-1, 0.7, 1.3)
    )
    center = np.array([1, 2, 3.0])
    result = uniform_external_potential_derivatives_native(config, position_mm=center)
    for i in range(3):
        shift = np.eye(3)[i] * 0.25
        plus, minus = [
            uniform_external_potential_derivatives_native(
                config, position_mm=center + s * shift
            )
            for s in (1, -1)
        ]
        np.testing.assert_allclose(
            (plus.four_potential - minus.four_potential) / 0.5,
            result.partial_a[i + 1],
            rtol=1e-14,
            atol=1e-15,
        )
    for name in (
        "partial2_a",
        "partial3_a_along_velocity",
        "partial3_a_along_acceleration",
        "partial4_a_along_velocity_twice",
    ):
        assert not np.any(getattr(result, name))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(x_min=0),
        dict(t_max=1),
        dict(magnetic_field_gradient_t_per_m=((1, 0, 0), (0, -1, 0), (0, 0, 0))),
    ],
)
def test_nonuniform_or_bounded_fields_are_explicitly_unavailable(kwargs):
    config = ExternalFieldConfig(**kwargs)
    assert not supports_uniform_external_potential(config)
    with pytest.raises(ValueError, match="unbounded uniform"):
        uniform_external_potential_derivatives_native(config, position_mm=(0, 0, 0))
    config.enabled = False
    assert supports_uniform_external_potential(config)
    result = uniform_external_potential_derivatives_native(
        config, position_mm=(0, 0, 0)
    )
    assert all(not np.any(getattr(result, item.name)) for item in fields(result))
