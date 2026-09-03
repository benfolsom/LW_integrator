from __future__ import annotations

import numpy as np

from core.constants import C_MMNS
from core.spin_self_torque_reduction_oracle import (
    evaluate_causal_sampled_fermi_walker_magnetic_torque_reduction_native,
    evaluate_sampled_fermi_walker_magnetic_torque_reduction_native,
)


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
