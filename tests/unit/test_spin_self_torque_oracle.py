from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.magnetic_dipole import (
    NATIVE_ENERGY_UNIT_J,
    boost_rest_polarization,
    magnetic_moment_native_to_j_per_t,
)
from core.spin_self_torque_oracle import (
    evaluate_inertial_point_magnetic_dipole_self_torque_native,
)


def test_inertial_self_torque_matches_rest_frame_si_point_law() -> None:
    moment_native = np.array((1.2e-8, -0.7e-8, 0.3e-8))
    moment_third_native = np.array((-0.4e-8, 0.9e-8, 1.1e-8))
    result = evaluate_inertial_point_magnetic_dipole_self_torque_native(
        four_velocity_mm_ns=(C_MMNS, 0.0, 0.0, 0.0),
        magnetic_moment_four_vector_native=np.r_[0.0, moment_native],
        magnetic_moment_third_proper_derivative_native=np.r_[0.0, moment_third_native],
    )

    mu0 = 4.0 * np.pi * 1.0e-7
    c_m_s = C_MMNS * 1.0e6
    moment_scale = magnetic_moment_native_to_j_per_t(1.0)
    expected_si = (
        mu0
        / (6.0 * np.pi * c_m_s**3)
        * np.cross(
            moment_scale * moment_native,
            moment_scale * 1.0e27 * moment_third_native,
        )
    )
    np.testing.assert_allclose(
        result.spin_torque_native,
        np.r_[0.0, expected_si / NATIVE_ENERGY_UNIT_J],
        rtol=2.0e-15,
        atol=0.0,
    )
    assert result.four_velocity_dot_torque_native == pytest.approx(0.0)
    assert result.magnetic_moment_dot_torque_native == pytest.approx(0.0)
    assert result.inertial_worldline_only
    assert not result.spin_torque_native.flags.writeable


def test_inertial_self_torque_is_covariant_under_finite_boost() -> None:
    rest_moment = np.array((0.8e-8, -1.1e-8, 0.5e-8))
    rest_moment_third = np.array((1.3e-8, 0.4e-8, -0.9e-8))
    rest = evaluate_inertial_point_magnetic_dipole_self_torque_native(
        four_velocity_mm_ns=(C_MMNS, 0.0, 0.0, 0.0),
        magnetic_moment_four_vector_native=np.r_[0.0, rest_moment],
        magnetic_moment_third_proper_derivative_native=np.r_[0.0, rest_moment_third],
    )

    beta = np.array((0.63, -0.21, 0.38))
    gamma = 1.0 / np.sqrt(1.0 - beta @ beta)
    boosted = evaluate_inertial_point_magnetic_dipole_self_torque_native(
        four_velocity_mm_ns=C_MMNS * np.r_[gamma, gamma * beta],
        magnetic_moment_four_vector_native=boost_rest_polarization(rest_moment, beta),
        magnetic_moment_third_proper_derivative_native=boost_rest_polarization(
            rest_moment_third, beta
        ),
    )
    expected_torque = boost_rest_polarization(rest.spin_torque_native[1:], beta)

    np.testing.assert_allclose(
        boosted.spin_torque_native,
        expected_torque,
        rtol=3.0e-15,
        atol=2.0e-30,
    )
    assert abs(boosted.four_velocity_dot_torque_native) < 1.0e-28
    assert abs(boosted.magnetic_moment_dot_torque_native) < 1.0e-35


def test_inertial_self_torque_rejects_invalid_four_vectors() -> None:
    common = {
        "four_velocity_mm_ns": (C_MMNS, 0.0, 0.0, 0.0),
        "magnetic_moment_four_vector_native": (0.0, 1.0e-8, 0.0, 0.0),
        "magnetic_moment_third_proper_derivative_native": (
            0.0,
            0.0,
            1.0e-8,
            0.0,
        ),
    }
    with pytest.raises(ValueError, match="u.u"):
        evaluate_inertial_point_magnetic_dipole_self_torque_native(
            **{**common, "four_velocity_mm_ns": (0.9 * C_MMNS, 0.0, 0.0, 0.0)}
        )
    with pytest.raises(ValueError, match="u.mu"):
        evaluate_inertial_point_magnetic_dipole_self_torque_native(
            **{
                **common,
                "magnetic_moment_four_vector_native": (1.0e-8, 0.0, 0.0, 0.0),
            }
        )
