"""Equivalence tests for the potential-jet RFS algebra."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from core.constants import C_MMNS
from core.magnetic_dipole import boost_rest_polarization
from core.potential_jet_rfs import (
    potential_derivative_rfs_response_native,
    potential_directional_rfs_reduction_jet_native,
)
from core.rfs import (
    MINKOWSKI_METRIC,
    electromagnetic_field_tensor_native,
    rfs_four_force_native,
    rfs_spin_rhs_native,
)

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0), dtype=float)


def _field_tensor(partial_a: np.ndarray) -> np.ndarray:
    partial_up_a = _SIGNS[:, np.newaxis] * partial_a
    return partial_up_a - partial_up_a.T


def _field_gradient(partial2_a: np.ndarray) -> np.ndarray:
    return _SIGNS[np.newaxis, :, np.newaxis] * partial2_a - _SIGNS[
        np.newaxis, np.newaxis, :
    ] * np.swapaxes(partial2_a, 1, 2)


@pytest.mark.parametrize("seed", range(32))
def test_potential_derivative_response_matches_field_tensor_path(seed: int) -> None:
    rng = np.random.default_rng(seed)
    beta = rng.uniform(-0.75, 0.75, size=3)
    beta *= 0.92 / max(0.92, float(np.linalg.norm(beta)))
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = C_MMNS * gamma * np.concatenate(([1.0], beta))
    rest_spin = rng.normal(size=3)
    rest_spin /= np.linalg.norm(rest_spin)
    spin = boost_rest_polarization(rest_spin, beta)

    partial_a = rng.normal(scale=3.0e-4, size=(4, 4))
    raw_hessian = rng.normal(scale=8.0e-5, size=(4, 4, 4))
    partial2_a = 0.5 * (raw_hessian + np.swapaxes(raw_hessian, 0, 1))
    charge = float(rng.uniform(-2.0, 2.0))
    moment = float(rng.uniform(-3.0e-3, 3.0e-3))
    mass = float(rng.uniform(0.25, 4.0))
    invariant_spin = float(rng.uniform(0.1, 2.0))

    field_tensor = _field_tensor(partial_a)
    partial_f = _field_gradient(partial2_a)
    expected_force = rfs_four_force_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        field_tensor=field_tensor,
        partial_f=partial_f,
        charge_native=charge,
        magnetic_moment_native=moment,
    )
    expected_spin = rfs_spin_rhs_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        field_tensor=field_tensor,
        partial_f=partial_f,
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )

    result = potential_derivative_rfs_response_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        partial_a=partial_a,
        partial2_a=partial2_a,
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )

    force_scale = max(float(np.linalg.norm(expected_force)), 1.0e-30)
    spin_scale = max(float(np.linalg.norm(expected_spin)), 1.0e-30)
    assert np.linalg.norm(result.total_four_force - expected_force) <= (
        2.0e-15 * force_scale
    )
    assert np.linalg.norm(result.spin_rhs - expected_spin) <= 3.0e-15 * spin_scale
    np.testing.assert_allclose(
        result.charge_four_force + result.dipole_four_force,
        result.total_four_force,
        rtol=0.0,
        atol=0.0,
    )


def test_diagonal_first_derivatives_do_not_change_mechanical_rfs_response() -> None:
    beta = np.asarray((0.2, -0.3, 0.1))
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = C_MMNS * gamma * np.concatenate(([1.0], beta))
    spin = boost_rest_polarization((0.3, 0.4, np.sqrt(0.75)), beta)
    partial_a = np.arange(16, dtype=float).reshape(4, 4) * 1.0e-6
    partial2_a = np.zeros((4, 4, 4), dtype=float)
    kwargs = dict(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        partial2_a=partial2_a,
        charge_native=-1.0,
        mass_amu=1.0,
        magnetic_moment_native=2.0e-3,
        invariant_spin_native=0.5,
    )
    baseline = potential_derivative_rfs_response_native(
        partial_a=partial_a,
        **kwargs,
    )
    changed = partial_a.copy()
    changed[np.diag_indices(4)] += (1.0, 2.0, 3.0, 4.0)
    candidate = potential_derivative_rfs_response_native(
        partial_a=changed,
        **kwargs,
    )
    np.testing.assert_array_equal(candidate.total_four_force, baseline.total_four_force)
    np.testing.assert_array_equal(candidate.spin_rhs, baseline.spin_rhs)


def test_validation_rejects_noncommuting_potential_hessian() -> None:
    partial2_a = np.zeros((4, 4, 4), dtype=float)
    partial2_a[0, 1, 2] = 1.0
    with pytest.raises(ValueError, match="derivative indices must commute"):
        potential_derivative_rfs_response_native(
            four_velocity_mm_ns=(C_MMNS, 0.0, 0.0, 0.0),
            spin_four_vector=(0.0, 0.0, 0.0, 1.0),
            partial_a=np.zeros((4, 4)),
            partial2_a=partial2_a,
            charge_native=1.0,
            mass_amu=1.0,
            magnetic_moment_native=1.0,
            invariant_spin_native=1.0,
        )


def test_directional_reduction_jet_is_exact_in_a_uniform_field() -> None:
    charge = -0.8
    mass = 1.7
    invariant_spin = 0.6
    moment = invariant_spin * charge / (mass * C_MMNS)
    beta = np.asarray((0.21, -0.08, 0.04))
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = C_MMNS * gamma * np.concatenate(([1.0], beta))
    spin = boost_rest_polarization((0.3, 0.4, np.sqrt(0.75)), beta)
    field = electromagnetic_field_tensor_native(
        (2.0e-4, -1.0e-4, 0.5e-4),
        (0.3e-4, 0.7e-4, -0.2e-4),
    )
    partial_a = 0.5 * _SIGNS[:, np.newaxis] * field
    zero_hessian = np.zeros((4, 4, 4))

    result = potential_directional_rfs_reduction_jet_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        partial_a=partial_a,
        partial2_a=zero_hessian,
        partial3_a_along_velocity=zero_hessian,
        partial3_a_along_acceleration=zero_hessian,
        partial4_a_along_velocity_twice=zero_hessian,
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )

    linear_map = charge / (mass * C_MMNS) * field @ MINKOWSKI_METRIC
    expected_acceleration = linear_map @ velocity
    expected_jerk = linear_map @ expected_acceleration
    expected_snap = linear_map @ expected_jerk
    expected_spin_first = linear_map @ spin
    expected_spin_second = linear_map @ expected_spin_first
    expected_spin_third = linear_map @ expected_spin_second
    np.testing.assert_allclose(
        result.four_acceleration,
        expected_acceleration,
        rtol=3.0e-15,
        atol=1.0e-18,
    )
    np.testing.assert_allclose(
        result.four_jerk,
        expected_jerk,
        rtol=5.0e-15,
        atol=1.0e-21,
    )
    np.testing.assert_allclose(
        result.four_snap,
        expected_snap,
        rtol=8.0e-15,
        atol=1.0e-24,
    )
    np.testing.assert_allclose(
        result.normalized_spin_first_derivative,
        expected_spin_first,
        rtol=3.0e-15,
        atol=1.0e-18,
    )
    np.testing.assert_allclose(
        result.normalized_spin_second_derivative,
        expected_spin_second,
        rtol=5.0e-15,
        atol=1.0e-21,
    )
    np.testing.assert_allclose(
        result.normalized_spin_third_derivative,
        expected_spin_third,
        rtol=8.0e-15,
        atol=1.0e-24,
    )
    np.testing.assert_array_equal(result.dipole_four_force_first_derivative, 0.0)
    np.testing.assert_array_equal(result.dipole_four_force_second_derivative, 0.0)


def _time_polynomial_potential_derivatives(
    coordinate_time_mm: float,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    derivatives = []
    for derivative_order in range(1, 5):
        value = np.zeros(4)
        for polynomial_order in range(derivative_order, 5):
            value += (
                coefficients[polynomial_order]
                * coordinate_time_mm ** (polynomial_order - derivative_order)
                / math.factorial(polynomial_order - derivative_order)
            )
        derivatives.append(value)
    partial_a = np.zeros((4, 4))
    partial2_a = np.zeros((4, 4, 4))
    partial3_a = np.zeros((4, 4, 4, 4))
    partial4_a = np.zeros((4, 4, 4, 4, 4))
    partial_a[0] = derivatives[0]
    partial2_a[0, 0] = derivatives[1]
    partial3_a[0, 0, 0] = derivatives[2]
    partial4_a[0, 0, 0, 0] = derivatives[3]
    return partial_a, partial2_a, partial3_a, partial4_a


def _rk4_state(
    initial_state: np.ndarray,
    target_time_ns: float,
    rhs: Callable[[np.ndarray], np.ndarray],
    *,
    substeps: int = 80,
) -> np.ndarray:
    if target_time_ns == 0.0:
        return initial_state.copy()
    step = target_time_ns / substeps
    state = initial_state.copy()
    for _ in range(substeps):
        first = rhs(state)
        second = rhs(state + 0.5 * step * first)
        third = rhs(state + 0.5 * step * second)
        fourth = rhs(state + step * third)
        state += step * (first + 2.0 * second + 2.0 * third + fourth) / 6.0
    return state


def test_directional_reduction_jet_matches_a_smooth_local_trajectory() -> None:
    charge = 0.7
    mass = 1.3
    moment = 2.0e-4
    invariant_spin = 0.8
    coefficients = np.asarray(
        (
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0e-5, -2.0e-5, 0.5e-5),
            (0.0, 2.0e-6, 1.0e-6, -1.5e-6),
            (0.0, -0.8e-6, 0.6e-6, 1.1e-6),
            (0.0, 0.3e-6, -0.2e-6, 0.4e-6),
        )
    )
    beta = np.asarray((0.12, -0.04, 0.03))
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = C_MMNS * gamma * np.concatenate(([1.0], beta))
    spin = boost_rest_polarization((0.2, -0.3, np.sqrt(0.87)), beta)
    partial_a, partial2_a, partial3_a, partial4_a = (
        _time_polynomial_potential_derivatives(0.0, coefficients)
    )
    leading = potential_derivative_rfs_response_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        partial_a=partial_a,
        partial2_a=partial2_a,
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )
    acceleration = leading.total_four_force / mass
    analytical = potential_directional_rfs_reduction_jet_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        partial_a=partial_a,
        partial2_a=partial2_a,
        partial3_a_along_velocity=np.einsum("k,klmn->lmn", velocity, partial3_a),
        partial3_a_along_acceleration=np.einsum(
            "k,klmn->lmn", acceleration, partial3_a
        ),
        partial4_a_along_velocity_twice=np.einsum(
            "k,l,klmnr->mnr",
            velocity,
            velocity,
            partial4_a,
        ),
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )

    def rhs(state: np.ndarray) -> np.ndarray:
        event_partial_a, event_partial2_a, _, _ = (
            _time_polynomial_potential_derivatives(state[0], coefficients)
        )
        response = potential_derivative_rfs_response_native(
            four_velocity_mm_ns=state[1:5],
            spin_four_vector=state[5:9],
            partial_a=event_partial_a,
            partial2_a=event_partial2_a,
            charge_native=charge,
            mass_amu=mass,
            magnetic_moment_native=moment,
            invariant_spin_native=invariant_spin,
        )
        return np.concatenate(
            ((state[1],), response.total_four_force / mass, response.spin_rhs)
        )

    initial_state = np.concatenate(((0.0,), velocity, spin))
    errors = []
    for step in (2.0e-3, 1.0e-3):
        states = {
            offset: _rk4_state(initial_state, offset * step, rhs)
            for offset in (-2, -1, 0, 1, 2)
        }
        accelerations = {}
        spin_rates = {}
        for offset, state in states.items():
            event_partial_a, event_partial2_a, _, _ = (
                _time_polynomial_potential_derivatives(state[0], coefficients)
            )
            response = potential_derivative_rfs_response_native(
                four_velocity_mm_ns=state[1:5],
                spin_four_vector=state[5:9],
                partial_a=event_partial_a,
                partial2_a=event_partial2_a,
                charge_native=charge,
                mass_amu=mass,
                magnetic_moment_native=moment,
                invariant_spin_native=invariant_spin,
            )
            accelerations[offset] = response.total_four_force / mass
            spin_rates[offset] = response.spin_rhs
        numerical_jerk = (
            accelerations[-2]
            - 8.0 * accelerations[-1]
            + 8.0 * accelerations[1]
            - accelerations[2]
        ) / (12.0 * step)
        numerical_snap = (
            -accelerations[-2]
            + 16.0 * accelerations[-1]
            - 30.0 * accelerations[0]
            + 16.0 * accelerations[1]
            - accelerations[2]
        ) / (12.0 * step**2)
        numerical_spin_second = (
            spin_rates[-2] - 8.0 * spin_rates[-1] + 8.0 * spin_rates[1] - spin_rates[2]
        ) / (12.0 * step)
        numerical_spin_third = (
            -spin_rates[-2]
            + 16.0 * spin_rates[-1]
            - 30.0 * spin_rates[0]
            + 16.0 * spin_rates[1]
            - spin_rates[2]
        ) / (12.0 * step**2)
        error = np.concatenate(
            (
                numerical_jerk - analytical.four_jerk,
                numerical_snap - analytical.four_snap,
                numerical_spin_second - analytical.normalized_spin_second_derivative,
                numerical_spin_third - analytical.normalized_spin_third_derivative,
            )
        )
        errors.append(float(np.linalg.norm(error)))

    # Both spacings have already reached floating-point cancellation in the
    # five-point numerical reference; the analytical jet agrees well below
    # the scale required by the self-force comparisons.
    assert max(errors) < 1.0e-12
