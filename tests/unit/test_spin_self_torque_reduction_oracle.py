from __future__ import annotations

import numpy as np

from core.constants import C_MMNS
from core.magnetic_dipole import boost_rest_polarization
from core.potential_jet_rfs import potential_derivative_rfs_response_native
from core.rfs import electromagnetic_field_tensor_native
from core.spin_self_torque_reduction_oracle import (
    evaluate_causal_sampled_fermi_walker_magnetic_torque_reduction_native,
    evaluate_potential_directional_magnetic_torque_reduction_native,
    evaluate_sampled_fermi_walker_magnetic_torque_reduction_native,
)

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0))


def _hyperbolic_samples(
    proper_times: np.ndarray,
    *,
    rapidity_rate_per_ns: float,
    moment_frequency_per_ns: float,
) -> tuple[np.ndarray, np.ndarray]:
    rapidity = rapidity_rate_per_ns * proper_times
    velocities = C_MMNS * np.column_stack(
        (
            np.cosh(rapidity),
            np.sinh(rapidity),
            np.zeros_like(proper_times),
            np.zeros_like(proper_times),
        )
    )
    moment_x = 0.8e-8 * np.cos(moment_frequency_per_ns * proper_times)
    moment_y = 1.1e-8 * np.sin(moment_frequency_per_ns * proper_times)
    moments = np.column_stack(
        (
            moment_x * np.sinh(rapidity),
            moment_x * np.cosh(rapidity),
            moment_y,
            np.full_like(proper_times, -0.4e-8),
        )
    )
    return velocities, moments


def _boost_four_vectors(vectors: np.ndarray, beta: np.ndarray) -> np.ndarray:
    gamma = 1.0 / np.sqrt(1.0 - beta @ beta)
    temporal = vectors[:, 0]
    spatial = vectors[:, 1:]
    projections = spatial @ beta
    boosted_temporal = gamma * (temporal + projections)
    boosted_spatial = (
        spatial
        + ((gamma - 1.0) / (beta @ beta)) * projections[:, None] * beta
        + gamma * temporal[:, None] * beta
    )
    return np.column_stack((boosted_temporal, boosted_spatial))


def _rk4_state(
    initial_state: np.ndarray,
    proper_time_ns: float,
    rhs,
    *,
    substeps: int = 80,
) -> np.ndarray:
    if proper_time_ns == 0.0:
        return initial_state.copy()
    step = proper_time_ns / substeps
    state = initial_state.copy()
    for _ in range(substeps):
        first = rhs(state)
        second = rhs(state + 0.5 * step * first)
        third = rhs(state + 0.5 * step * second)
        fourth = rhs(state + step * third)
        state += step * (first + 2.0 * second + 2.0 * third + fourth) / 6.0
    return state


def test_centered_reconstruction_converges_for_accelerated_physical_curve() -> None:
    rapidity_rate = 0.31
    frequency = 0.73
    exact_first = np.array((0.0, 0.0, 1.1e-8 * frequency, 0.0))
    exact_third = np.array((0.0, 0.0, -1.1e-8 * frequency**3, 0.0))
    errors = []
    for step in (0.12, 0.06, 0.03):
        times = step * np.arange(-3.0, 4.0)
        velocities, moments = _hyperbolic_samples(
            times,
            rapidity_rate_per_ns=rapidity_rate,
            moment_frequency_per_ns=frequency,
        )
        result = evaluate_sampled_fermi_walker_magnetic_torque_reduction_native(
            proper_times_ns=times,
            four_velocity_samples_mm_ns=velocities,
            magnetic_moment_four_samples_native=moments,
        )
        errors.append(
            np.linalg.norm(
                result.magnetic_moment_third_fermi_walker_derivative_native
                - exact_third
            )
        )
        np.testing.assert_allclose(
            result.magnetic_moment_first_fermi_walker_derivative_native,
            exact_first,
            rtol=2.0e-5,
            atol=2.0e-18,
        )
        np.testing.assert_allclose(
            result.reconstructed_four_acceleration_mm_ns2,
            (0.0, C_MMNS * rapidity_rate, 0.0, 0.0),
            rtol=2.0e-8,
            atol=2.0e-13,
        )
        assert result.uses_future_samples
        assert result.reduction_of_order_reference
        assert result.leading_non_self_samples_required

    # A seven-point centered third derivative is fourth-order accurate.
    assert errors[0] / errors[1] > 15.0
    assert errors[1] / errors[2] > 15.0


def test_causal_reconstruction_uses_only_newest_accepted_history() -> None:
    step = 0.01
    times = step * np.arange(-7.0, 1.0)
    rapidity_rate = 0.27
    frequency = 0.61
    velocities, moments = _hyperbolic_samples(
        times,
        rapidity_rate_per_ns=rapidity_rate,
        moment_frequency_per_ns=frequency,
    )
    result = evaluate_causal_sampled_fermi_walker_magnetic_torque_reduction_native(
        proper_times_ns=times,
        four_velocity_samples_mm_ns=velocities,
        magnetic_moment_four_samples_native=moments,
    )

    assert result.center_index == times.size - 1
    assert result.stencil_kind == "backward_accepted_history"
    assert not result.uses_future_samples
    np.testing.assert_allclose(
        result.magnetic_moment_first_fermi_walker_derivative_native,
        (0.0, 0.0, 1.1e-8 * frequency, 0.0),
        rtol=2.0e-8,
        atol=2.0e-17,
    )
    np.testing.assert_allclose(
        result.magnetic_moment_third_fermi_walker_derivative_native,
        (0.0, 0.0, -1.1e-8 * frequency**3, 0.0),
        rtol=3.0e-5,
        atol=2.0e-15,
    )
    assert result.maximum_sample_velocity_norm_residual_mm2_ns2 < 1.0e-10
    assert result.maximum_sample_velocity_moment_residual_native_mm_ns < 1.0e-20
    assert not result.torque_comparator.reduction_of_order_performed


def test_sampled_fermi_walker_reconstruction_is_covariant_under_boost() -> None:
    times = 0.025 * np.arange(-3.0, 4.0)
    velocities, moments = _hyperbolic_samples(
        times,
        rapidity_rate_per_ns=0.31,
        moment_frequency_per_ns=0.73,
    )
    rest = evaluate_sampled_fermi_walker_magnetic_torque_reduction_native(
        proper_times_ns=times,
        four_velocity_samples_mm_ns=velocities,
        magnetic_moment_four_samples_native=moments,
    )
    beta = np.array((0.47, -0.26, 0.19))
    boosted = evaluate_sampled_fermi_walker_magnetic_torque_reduction_native(
        proper_times_ns=times,
        four_velocity_samples_mm_ns=_boost_four_vectors(velocities, beta),
        magnetic_moment_four_samples_native=_boost_four_vectors(moments, beta),
    )

    for boosted_value, rest_value in (
        (
            boosted.magnetic_moment_first_fermi_walker_derivative_native,
            rest.magnetic_moment_first_fermi_walker_derivative_native,
        ),
        (
            boosted.magnetic_moment_third_fermi_walker_derivative_native,
            rest.magnetic_moment_third_fermi_walker_derivative_native,
        ),
        (
            boosted.torque_comparator.total_spin_torque_native,
            rest.torque_comparator.total_spin_torque_native,
        ),
    ):
        np.testing.assert_allclose(
            boosted_value,
            _boost_four_vectors(rest_value[None, :], beta)[0],
            rtol=2.0e-7,
            atol=2.0e-18,
        )


def test_analytical_potential_reduction_matches_sampled_leading_trajectory() -> None:
    charge = -0.8
    mass = 1.7
    moment = 2.0e-4
    invariant_spin = 0.8
    beta = np.asarray((0.12, -0.04, 0.03))
    gamma = 1.0 / np.sqrt(1.0 - beta @ beta)
    velocity = C_MMNS * gamma * np.r_[1.0, beta]
    spin = boost_rest_polarization((0.2, -0.3, np.sqrt(0.87)), beta)
    field = electromagnetic_field_tensor_native(
        (2.0e-2, -1.0e-2, 0.5e-2),
        (0.3e-2, 0.7e-2, -0.2e-2),
    )
    partial_a = 0.5 * _SIGNS[:, None] * field
    zeros = np.zeros((4, 4, 4))
    analytical = evaluate_potential_directional_magnetic_torque_reduction_native(
        four_velocity_mm_ns=velocity,
        normalized_spin_four_vector=spin,
        partial_a=partial_a,
        partial2_a=zeros,
        partial3_a_along_velocity=zeros,
        partial3_a_along_acceleration=zeros,
        partial4_a_along_velocity_twice=zeros,
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )

    def rhs(state: np.ndarray) -> np.ndarray:
        response = potential_derivative_rfs_response_native(
            four_velocity_mm_ns=state[:4],
            spin_four_vector=state[4:],
            partial_a=partial_a,
            partial2_a=zeros,
            charge_native=charge,
            mass_amu=mass,
            magnetic_moment_native=moment,
            invariant_spin_native=invariant_spin,
        )
        return np.r_[response.total_four_force / mass, response.spin_rhs]

    initial = np.r_[velocity, spin]
    errors = []
    for step in (0.4, 0.2, 0.1):
        times = step * np.arange(-3.0, 4.0)
        states = np.asarray([_rk4_state(initial, time, rhs) for time in times])
        sampled = evaluate_sampled_fermi_walker_magnetic_torque_reduction_native(
            proper_times_ns=times,
            four_velocity_samples_mm_ns=states[:, :4],
            magnetic_moment_four_samples_native=moment * states[:, 4:],
        )
        errors.append(
            np.linalg.norm(
                sampled.magnetic_moment_third_fermi_walker_derivative_native
                - analytical.fermi_walker_derivatives.third_fermi_walker_derivative_native
            )
        )
        np.testing.assert_allclose(
            sampled.magnetic_moment_first_fermi_walker_derivative_native,
            analytical.fermi_walker_derivatives.first_fermi_walker_derivative_native,
            rtol=2.0e-8,
            atol=2.0e-16,
        )

    # This cross-check reaches finite-difference cancellation before a clean
    # refinement ratio; the separate hyperbolic test above establishes the
    # sampled oracle's fourth-order convergence.
    assert max(errors) < 2.0e-16
    assert analytical.analytical_potential_derivatives_only
    assert analytical.reduction_of_order_performed
