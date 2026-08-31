from __future__ import annotations

import math

import numpy as np
import pytest

from core.constants import C_MMNS
from core.spin_self_force_oracle import (
    evaluate_jakobsen_intrinsic_spin_radiation_balance_native,
)
from core.spin_self_force_reduction_oracle import (
    evaluate_causal_sampled_intrinsic_spin_reduction_native,
    evaluate_sampled_intrinsic_spin_reduction_native,
)


def test_irregular_stencil_reconstructs_polynomial_leading_dynamics() -> None:
    times = np.array((-0.7, -0.2, 0.0, 0.3, 0.9))
    velocities = np.zeros((times.size, 4))
    velocities[:, 0] = C_MMNS
    velocities[:, 1] = 0.2 * times + 0.3 * times**2 - 0.1 * times**3 + 0.05 * times**4
    accelerations = np.zeros_like(velocities)
    accelerations[:, 1] = 0.2 + 0.6 * times - 0.3 * times**2 + 0.2 * times**3
    spins = np.zeros_like(velocities)
    spins[:, 3] = 1.0 + 0.4 * times - 0.2 * times**2 + 0.1 * times**3

    result = evaluate_sampled_intrinsic_spin_reduction_native(
        proper_times_ns=times,
        four_velocity_samples_mm_ns=velocities,
        non_self_four_acceleration_samples_mm_ns2=accelerations,
        physical_spin_four_samples_native=spins,
        charge_native=0.7,
        mass_amu=1.3,
        g_factor=2.2,
    )

    assert result.stencil_kind == "centered_reference"
    assert result.uses_future_samples
    assert result.evaluation_proper_time_ns == 0.0
    assert np.isfinite(result.scaled_vandermonde_condition_number)
    np.testing.assert_allclose(
        result.reconstructed_four_jerk_mm_ns3,
        (0.0, 0.6, 0.0, 0.0),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.reconstructed_four_snap_mm_ns4,
        (0.0, -0.6, 0.0, 0.0),
        rtol=0.0,
        atol=8.0e-15,
    )
    np.testing.assert_allclose(
        result.reconstructed_spin_four_derivative_native,
        (0.0, 0.0, 0.0, 0.4),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.reconstructed_spin_four_second_derivative_native,
        (0.0, 0.0, 0.0, -0.4),
        rtol=0.0,
        atol=8.0e-15,
    )
    np.testing.assert_allclose(
        result.velocity_derivative_residual_mm_ns2,
        0.0,
        atol=2.0e-15,
    )
    scale = max(
        np.linalg.norm(
            result.radiation_balance.self_force.linear_spin_radiative_balance_rate_native
        ),
        np.finfo(float).tiny,
    )
    assert np.linalg.norm(result.radiation_balance.balance_residual_native) < (
        3.0e-14 * scale
    )


def _circular_state(
    coordinate_time_ns: float,
    *,
    orbit_radius_mm: float,
    angular_frequency_per_ns: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase = angular_frequency_per_ns * coordinate_time_ns
    radial = np.array((math.cos(phase), math.sin(phase), 0.0))
    tangent = np.array((-math.sin(phase), math.cos(phase), 0.0))
    velocity = orbit_radius_mm * angular_frequency_per_ns * tangent
    four_velocity = np.r_[gamma * C_MMNS, gamma * velocity]
    four_acceleration = np.r_[
        0.0,
        -(gamma**2) * orbit_radius_mm * angular_frequency_per_ns**2 * radial,
    ]
    four_jerk = np.r_[
        0.0,
        -(gamma**3) * orbit_radius_mm * angular_frequency_per_ns**3 * tangent,
    ]
    four_snap = np.r_[
        0.0,
        gamma**4 * orbit_radius_mm * angular_frequency_per_ns**4 * radial,
    ]
    return four_velocity, four_acceleration, four_jerk, four_snap


def test_circular_reduction_converges_fourth_order_to_unreduced_oracle() -> None:
    charge = 0.8
    mass = 1.0
    g_factor = 2.3
    orbit_radius = 0.03
    angular_frequency = 1.7
    beta = orbit_radius * angular_frequency / C_MMNS
    gamma = 1.0 / math.sqrt(1.0 - beta**2)
    proper_frequency = gamma * angular_frequency
    spin = np.array((0.0, 0.0, 0.0, 0.6))
    velocity, acceleration, jerk, snap = _circular_state(
        0.0,
        orbit_radius_mm=orbit_radius,
        angular_frequency_per_ns=angular_frequency,
        gamma=gamma,
    )
    exact = evaluate_jakobsen_intrinsic_spin_radiation_balance_native(
        charge_native=charge,
        mass_amu=mass,
        g_factor=g_factor,
        four_velocity_mm_ns=velocity,
        four_acceleration_mm_ns2=acceleration,
        four_jerk_mm_ns3=jerk,
        four_snap_mm_ns4=snap,
        spin_four_vector_native=spin,
        spin_four_derivative_native=np.zeros(4),
        spin_four_second_derivative_native=np.zeros(4),
    )

    errors = []
    for dimensionless_step in (0.2, 0.1, 0.05):
        step = dimensionless_step / proper_frequency
        proper_times = step * np.arange(-2.0, 3.0)
        velocity_samples = []
        acceleration_samples = []
        for proper_time in proper_times:
            sample_velocity, sample_acceleration, _, _ = _circular_state(
                gamma * proper_time,
                orbit_radius_mm=orbit_radius,
                angular_frequency_per_ns=angular_frequency,
                gamma=gamma,
            )
            velocity_samples.append(sample_velocity)
            acceleration_samples.append(sample_acceleration)
        reduced = evaluate_sampled_intrinsic_spin_reduction_native(
            proper_times_ns=proper_times,
            four_velocity_samples_mm_ns=np.asarray(velocity_samples),
            non_self_four_acceleration_samples_mm_ns2=np.asarray(acceleration_samples),
            physical_spin_four_samples_native=np.repeat(spin[None, :], 5, axis=0),
            charge_native=charge,
            mass_amu=mass,
            g_factor=g_factor,
        )
        errors.append(
            np.linalg.norm(
                reduced.radiation_balance.self_force.linear_spin_self_force_native
                - exact.self_force.linear_spin_self_force_native
            )
        )
        assert np.linalg.norm(reduced.velocity_derivative_residual_mm_ns2) < (
            2.0e-4 * np.linalg.norm(acceleration)
        )

    assert errors[0] / errors[1] > 15.0
    assert errors[1] / errors[2] > 15.0


def test_causal_irregular_stencil_recovers_endpoint_polynomial() -> None:
    times = np.array((-0.9, -0.55, -0.3, -0.1, -0.03, 0.0))
    velocities = np.zeros((times.size, 4))
    velocities[:, 0] = C_MMNS
    velocities[:, 1] = (
        0.2 * times
        + 0.3 * times**2
        - 0.1 * times**3
        + 0.05 * times**4
        - 0.02 * times**5
    )
    accelerations = np.zeros_like(velocities)
    accelerations[:, 1] = (
        0.2 + 0.6 * times - 0.3 * times**2 + 0.2 * times**3 - 0.1 * times**4
    )
    spins = np.zeros_like(velocities)
    spins[:, 3] = (
        1.0
        + 0.4 * times
        - 0.2 * times**2
        + 0.1 * times**3
        - 0.03 * times**4
        + 0.01 * times**5
    )

    result = evaluate_causal_sampled_intrinsic_spin_reduction_native(
        proper_times_ns=times,
        four_velocity_samples_mm_ns=velocities,
        non_self_four_acceleration_samples_mm_ns2=accelerations,
        physical_spin_four_samples_native=spins,
        charge_native=0.7,
        mass_amu=1.3,
        g_factor=2.2,
    )

    assert result.center_index == times.size - 1
    assert result.stencil_kind == "backward_accepted_history"
    assert not result.uses_future_samples
    assert result.evaluation_proper_time_ns == times[-1]
    np.testing.assert_allclose(
        result.reconstructed_four_jerk_mm_ns3,
        (0.0, 0.6, 0.0, 0.0),
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        result.reconstructed_four_snap_mm_ns4,
        (0.0, -0.6, 0.0, 0.0),
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        result.reconstructed_spin_four_derivative_native,
        (0.0, 0.0, 0.0, 0.4),
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        result.reconstructed_spin_four_second_derivative_native,
        (0.0, 0.0, 0.0, -0.4),
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        result.velocity_derivative_residual_mm_ns2,
        0.0,
        atol=2.0e-12,
    )


def test_causal_circular_reduction_converges_at_fourth_order() -> None:
    charge = 0.8
    mass = 1.0
    g_factor = 2.3
    orbit_radius = 0.03
    angular_frequency = 1.7
    beta = orbit_radius * angular_frequency / C_MMNS
    gamma = 1.0 / math.sqrt(1.0 - beta**2)
    proper_frequency = gamma * angular_frequency
    spin = np.array((0.0, 0.0, 0.0, 0.6))
    velocity, acceleration, jerk, snap = _circular_state(
        0.0,
        orbit_radius_mm=orbit_radius,
        angular_frequency_per_ns=angular_frequency,
        gamma=gamma,
    )
    exact = evaluate_jakobsen_intrinsic_spin_radiation_balance_native(
        charge_native=charge,
        mass_amu=mass,
        g_factor=g_factor,
        four_velocity_mm_ns=velocity,
        four_acceleration_mm_ns2=acceleration,
        four_jerk_mm_ns3=jerk,
        four_snap_mm_ns4=snap,
        spin_four_vector_native=spin,
        spin_four_derivative_native=np.zeros(4),
        spin_four_second_derivative_native=np.zeros(4),
    )

    errors = []
    for dimensionless_step in (0.08, 0.04, 0.02):
        step = dimensionless_step / proper_frequency
        proper_times = step * np.arange(-5.0, 1.0)
        velocity_samples = []
        acceleration_samples = []
        for proper_time in proper_times:
            sample_velocity, sample_acceleration, _, _ = _circular_state(
                gamma * proper_time,
                orbit_radius_mm=orbit_radius,
                angular_frequency_per_ns=angular_frequency,
                gamma=gamma,
            )
            velocity_samples.append(sample_velocity)
            acceleration_samples.append(sample_acceleration)
        reduced = evaluate_causal_sampled_intrinsic_spin_reduction_native(
            proper_times_ns=proper_times,
            four_velocity_samples_mm_ns=np.asarray(velocity_samples),
            non_self_four_acceleration_samples_mm_ns2=np.asarray(acceleration_samples),
            physical_spin_four_samples_native=np.repeat(spin[None, :], 6, axis=0),
            charge_native=charge,
            mass_amu=mass,
            g_factor=g_factor,
        )
        errors.append(
            np.linalg.norm(
                reduced.radiation_balance.self_force.linear_spin_self_force_native
                - exact.self_force.linear_spin_self_force_native
            )
        )
        assert not reduced.uses_future_samples

    assert errors[0] / errors[1] > 14.0
    assert errors[1] / errors[2] > 14.0


def test_reduction_rejects_bad_time_and_shape_inputs() -> None:
    times = np.linspace(-0.2, 0.2, 5)
    samples = np.zeros((5, 4))
    samples[:, 0] = C_MMNS
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_sampled_intrinsic_spin_reduction_native(
            proper_times_ns=times[::-1],
            four_velocity_samples_mm_ns=samples,
            non_self_four_acceleration_samples_mm_ns2=np.zeros_like(samples),
            physical_spin_four_samples_native=np.zeros_like(samples),
            charge_native=0.0,
            mass_amu=1.0,
            g_factor=2.0,
        )
    with pytest.raises(ValueError, match="shape"):
        evaluate_sampled_intrinsic_spin_reduction_native(
            proper_times_ns=times,
            four_velocity_samples_mm_ns=samples[:, :3],
            non_self_four_acceleration_samples_mm_ns2=np.zeros_like(samples),
            physical_spin_four_samples_native=np.zeros_like(samples),
            charge_native=0.0,
            mass_amu=1.0,
            g_factor=2.0,
        )
    with pytest.raises(ValueError, match="at least 6"):
        evaluate_causal_sampled_intrinsic_spin_reduction_native(
            proper_times_ns=times,
            four_velocity_samples_mm_ns=samples,
            non_self_four_acceleration_samples_mm_ns2=np.zeros_like(samples),
            physical_spin_four_samples_native=np.zeros_like(samples),
            charge_native=0.0,
            mass_amu=1.0,
            g_factor=2.0,
        )
