from __future__ import annotations

import numpy as np
import pytest

from core.constants import (
    ELEMENTARY_CHARGE,
    ELEMENTARY_CHARGE_STATC,
    STATCOULOMB_TO_NATIVE_CHARGE,
)
from core.external_fields import NATIVE_FORCE_UNIT_NEWTON
from core.magnetic_dipole import (
    C_M_S,
    HBAR_J_S,
    HBAR_NATIVE,
    NATIVE_ACTION_UNIT_J_S,
    NATIVE_ENERGY_UNIT_J,
    advance_spin_uniform_fields,
    boost_rest_polarization,
    dual_electromagnetic_tensor,
    electric_field_native_to_v_per_m,
    electric_field_v_per_m_to_native,
    electromagnetic_field_tensor,
    fields_from_electromagnetic_tensor,
    force_native_to_newton,
    force_newton_to_native,
    instantaneous_bmt_angular_velocity,
    magnetic_field_native_to_tesla,
    magnetic_field_tesla_to_native,
    magnetic_gradient_native_per_mm_to_t_per_m,
    magnetic_gradient_t_per_m_to_native_per_mm,
    magnetic_moment_j_per_t_to_native,
    magnetic_moment_native_to_j_per_t,
    minkowski_dot,
    momentum_kg_m_s_to_native,
    momentum_native_to_kg_m_s,
    rest_polarization_from_four_vector,
    rotate_spin_rodrigues,
    signed_gyromagnetic_ratio,
    stern_gerlach_rest_force_newton,
    stern_gerlach_rest_impulse_native,
)
from core.species import get_species


def test_si_native_bridges_round_trip_and_preserve_force_definition() -> None:
    native_electric = 3.25e4
    native_magnetic = -7.5e3
    native_force = 2.75
    native_momentum = -4.125

    assert electric_field_v_per_m_to_native(
        electric_field_native_to_v_per_m(native_electric)
    ) == pytest.approx(native_electric)
    assert magnetic_field_tesla_to_native(
        magnetic_field_native_to_tesla(native_magnetic)
    ) == pytest.approx(native_magnetic)
    assert force_newton_to_native(
        force_native_to_newton(native_force)
    ) == pytest.approx(native_force)
    assert momentum_kg_m_s_to_native(
        momentum_native_to_kg_m_s(native_momentum)
    ) == pytest.approx(native_momentum)
    assert force_native_to_newton(1.0) == pytest.approx(NATIVE_FORCE_UNIT_NEWTON)


def test_native_magnetic_conversion_matches_beta_cross_b_force() -> None:
    beta = 0.2
    magnetic_native = 19.0
    magnetic_tesla = magnetic_field_native_to_tesla(magnetic_native)

    equivalent_electric_si = beta * C_M_S * magnetic_tesla
    equivalent_electric_native = electric_field_v_per_m_to_native(
        equivalent_electric_si
    )

    assert equivalent_electric_native == pytest.approx(beta * magnetic_native)


def test_native_moment_gradient_and_action_bridges_are_mutually_consistent() -> None:
    moment_j_t = -9.662_365_3e-27
    gradient_t_m = 7.5
    moment_native = magnetic_moment_j_per_t_to_native(moment_j_t)
    gradient_native = magnetic_gradient_t_per_m_to_native_per_mm(gradient_t_m)

    assert magnetic_moment_native_to_j_per_t(moment_native) == pytest.approx(moment_j_t)
    assert magnetic_gradient_native_per_mm_to_t_per_m(gradient_native) == pytest.approx(
        gradient_t_m
    )
    assert (
        moment_native * gradient_native * NATIVE_FORCE_UNIT_NEWTON
    ) == pytest.approx(moment_j_t * gradient_t_m, rel=2.0e-15)
    assert HBAR_NATIVE * NATIVE_ACTION_UNIT_J_S == pytest.approx(HBAR_J_S)
    assert NATIVE_ENERGY_UNIT_J == pytest.approx(NATIVE_FORCE_UNIT_NEWTON * 1.0e-3)


def test_magnetic_boundaries_share_the_exact_gaussian_charge_scale() -> None:
    exact_gaussian_elementary_charge = (
        ELEMENTARY_CHARGE_STATC * STATCOULOMB_TO_NATIVE_CHARGE
    )

    assert ELEMENTARY_CHARGE == exact_gaussian_elementary_charge


def test_field_tensor_round_trip_and_lorentz_force_signs() -> None:
    electric = np.array([2.0e6, -3.0e6, 5.0e6])
    magnetic = np.array([0.7, -1.1, 0.4])
    beta = np.array([0.2, -0.1, 0.3])
    gamma = 1.0 / np.sqrt(1.0 - beta @ beta)
    tensor = electromagnetic_field_tensor(electric, magnetic)
    four_velocity_covariant = np.concatenate(([gamma * C_M_S], -gamma * C_M_S * beta))

    recovered_electric, recovered_magnetic = fields_from_electromagnetic_tensor(tensor)
    tensor_force = tensor @ four_velocity_covariant
    expected_spatial = gamma * (electric + np.cross(beta * C_M_S, magnetic))

    np.testing.assert_allclose(recovered_electric, electric)
    np.testing.assert_allclose(recovered_magnetic, magnetic)
    np.testing.assert_allclose(tensor_force[1:], expected_spatial)
    np.testing.assert_allclose(tensor + tensor.T, 0.0)


def test_dual_tensor_has_lorentzian_involution() -> None:
    tensor = electromagnetic_field_tensor((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    dual = dual_electromagnetic_tensor(tensor)

    assert dual[0, 1] == pytest.approx(4.0)
    assert dual[1, 2] == pytest.approx(-3.0 / C_M_S)
    np.testing.assert_allclose(dual_electromagnetic_tensor(dual), -tensor)


def test_rest_polarization_boost_preserves_invariants_and_round_trips() -> None:
    rest_spin = np.array([0.3, -0.4, np.sqrt(0.75)])
    beta = np.array([0.3, 0.2, -0.1])
    gamma = 1.0 / np.sqrt(1.0 - beta @ beta)
    four_velocity_over_c = np.concatenate(([gamma], gamma * beta))

    boosted = boost_rest_polarization(rest_spin, beta)

    assert minkowski_dot(boosted, four_velocity_over_c) == pytest.approx(
        0.0, abs=2.0e-15
    )
    assert minkowski_dot(boosted, boosted) == pytest.approx(-(rest_spin @ rest_spin))
    np.testing.assert_allclose(
        rest_polarization_from_four_vector(boosted, beta), rest_spin
    )


def test_rodrigues_rotation_has_analytic_sign_and_preserves_norm() -> None:
    initial = np.array([1.0, 0.0, 0.0])
    angular_velocity = np.array([0.0, 0.0, 2.0])

    rotated = rotate_spin_rodrigues(initial, angular_velocity, delta_time_s=np.pi / 4.0)

    np.testing.assert_allclose(rotated, (0.0, 1.0, 0.0), atol=2.0e-15)
    assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(initial))


def test_electron_uniform_b_precession_uses_signed_moment() -> None:
    electron = get_species("electron")
    gyro = signed_gyromagnetic_ratio(
        electron.magnetic_moment_j_t, electron.spin_quantum_number
    )
    magnetic_field_t = 0.25
    quarter_period = np.pi / (2.0 * abs(gyro) * magnetic_field_t)

    rotated = advance_spin_uniform_fields(
        (1.0, 0.0, 0.0),
        beta=(0.0, 0.0, 0.0),
        electric_field_v_m=(0.0, 0.0, 0.0),
        magnetic_field_t=(0.0, 0.0, magnetic_field_t),
        charge_coulomb=electron.charge_coulomb,
        mass_kg=electron.mass_kg,
        gyromagnetic_ratio_rad_s_t=gyro,
        delta_time_s=quarter_period,
    )

    np.testing.assert_allclose(rotated, (0.0, 1.0, 0.0), atol=2.0e-15)
    assert np.linalg.norm(rotated) == pytest.approx(1.0)


def test_neutral_neutron_has_finite_bmt_limit_and_precesses() -> None:
    neutron = get_species("neutron")
    gyro = signed_gyromagnetic_ratio(
        neutron.magnetic_moment_j_t, neutron.spin_quantum_number
    )
    beta = np.array([0.0, 0.0, 0.6])
    gamma = 1.25

    omega = instantaneous_bmt_angular_velocity(
        beta=beta,
        electric_field_v_m=(0.0, 0.0, 0.0),
        magnetic_field_t=(0.0, 0.0, 2.0),
        charge_coulomb=0.0,
        mass_kg=neutron.mass_kg,
        gyromagnetic_ratio_rad_s_t=gyro,
    )

    assert np.all(np.isfinite(omega))
    assert omega[2] == pytest.approx(-gyro * 2.0 / gamma)
    assert omega[2] > 0.0


def test_bmt_g_equals_two_reduces_to_dirac_coefficients() -> None:
    charge_to_mass = -3.0e8
    mass_kg = 2.0e-27
    charge_coulomb = charge_to_mass * mass_kg
    beta = np.array([0.3, -0.2, 0.1])
    gamma = 1.0 / np.sqrt(1.0 - beta @ beta)
    electric = np.array([2.0e5, -4.0e5, 3.0e5])
    magnetic = np.array([0.2, 0.3, -0.7])

    omega = instantaneous_bmt_angular_velocity(
        beta=beta,
        electric_field_v_m=electric,
        magnetic_field_t=magnetic,
        charge_coulomb=charge_coulomb,
        mass_kg=mass_kg,
        gyromagnetic_ratio_rad_s_t=charge_to_mass,
    )
    expected = -charge_to_mass * (
        magnetic / gamma - np.cross(beta, electric) / ((gamma + 1.0) * C_M_S)
    )

    np.testing.assert_allclose(omega, expected)


def test_signed_gyromagnetic_ratio_rejects_inconsistent_zero_spin() -> None:
    assert signed_gyromagnetic_ratio(0.0, 0.0) == 0.0
    with pytest.raises(ValueError, match="nonzero moment"):
        signed_gyromagnetic_ratio(1.0, 0.0)


def test_stern_gerlach_rest_force_preserves_moment_and_gradient_sign() -> None:
    gradient = np.zeros((3, 3))
    gradient[2, 0] = 8.0

    positive = stern_gerlach_rest_force_newton((0.0, 0.0, 2.0e-26), gradient)
    negative = stern_gerlach_rest_force_newton((0.0, 0.0, -2.0e-26), gradient)

    np.testing.assert_allclose(positive, (1.6e-25, 0.0, 0.0))
    np.testing.assert_allclose(negative, (-1.6e-25, 0.0, 0.0))


def test_stern_gerlach_uniform_field_has_zero_force_and_impulse() -> None:
    zero_gradient = np.zeros((3, 3))

    np.testing.assert_array_equal(
        stern_gerlach_rest_force_newton((1.0e-26, 0.0, 0.0), zero_gradient),
        np.zeros(3),
    )
    assert stern_gerlach_rest_impulse_native(
        (1.0e-26, 0.0, 0.0), zero_gradient, proper_time_step_ns=4.0
    ) == (0.0, 0.0, 0.0)


def test_stern_gerlach_native_impulse_matches_force_times_native_time() -> None:
    gradient = np.zeros((3, 3))
    gradient[0, 1] = 2.0
    moment = (3.0e-26, 0.0, 0.0)

    impulse = stern_gerlach_rest_impulse_native(moment, gradient, 5.0)
    force = stern_gerlach_rest_force_newton(moment, gradient)

    assert impulse[1] == pytest.approx(force[1] / NATIVE_FORCE_UNIT_NEWTON * 5.0)
    assert impulse[0] == 0.0
    assert impulse[2] == 0.0
