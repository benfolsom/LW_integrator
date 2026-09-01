"""Directional higher-potential derivatives on smooth retarded segments."""

from __future__ import annotations

import numpy as np
import pytest

from core.charge_potential_jet import quintic_charge_potential_jet_native
from core.constants import C_MMNS
from core.dipole_hertz_jet import quintic_dipole_hertz_response_jet_native
from core.potential_jet_rfs import potential_derivative_rfs_response_native
from core.retarded_potential_directional_jet import (
    evaluate_retarded_charge_potential_directional_jet_native,
    evaluate_retarded_dipole_potential_directional_jet_native,
    quintic_charge_potential_directional_jet_native,
    quintic_dipole_potential_directional_jet_native,
    sum_potential_directional_derivatives_native,
)
from core.retarded_fields import ObserverEvent
from core.spin_self_force_reduction_oracle import (
    evaluate_causal_sampled_intrinsic_spin_reduction_native,
    evaluate_sampled_intrinsic_spin_reduction_native,
    evaluate_retarded_potential_intrinsic_spin_reduction_native,
)


def _static_segment(distance_mm: float) -> tuple[float, np.ndarray]:
    retarded_time_ns = -float(distance_mm) / C_MMNS
    return retarded_time_ns, np.zeros((6, 3), dtype=float)


def _static_history(
    times_ns: np.ndarray,
    *,
    charge_native: float = 1.7,
    magnetic_moment_native: float = 2.3,
) -> list[dict[str, np.ndarray]]:
    result: list[dict[str, np.ndarray]] = []
    for time_ns in times_ns:
        result.append(
            {
                "t": np.asarray((time_ns,)),
                "x": np.asarray((0.0,)),
                "y": np.asarray((0.0,)),
                "z": np.asarray((0.0,)),
                "bx": np.asarray((0.0,)),
                "by": np.asarray((0.0,)),
                "bz": np.asarray((0.0,)),
                "bdotx": np.asarray((0.0,)),
                "bdoty": np.asarray((0.0,)),
                "bdotz": np.asarray((0.0,)),
                "q": np.asarray((charge_native,)),
                "q_source": np.asarray((charge_native,)),
                "spin_x": np.asarray((0.0,)),
                "spin_y": np.asarray((0.0,)),
                "spin_z": np.asarray((1.0,)),
                "magnetic_moment_native": np.asarray((magnetic_moment_native,)),
                "magnetic_dipole_active": np.asarray(
                    (float(magnetic_moment_native != 0.0),)
                ),
                "_dead_particles": np.asarray((False,)),
            }
        )
    return result


def test_static_charge_returns_only_required_directional_derivatives() -> None:
    distance = 10.0
    charge = 1.7
    velocity_x = 0.2
    acceleration_x = 0.03
    retarded_time, coefficients = _static_segment(distance)

    result = quintic_charge_potential_directional_jet_native(
        observer_time_ns=0.0,
        observer_position_mm=(distance, 0.0, 0.0),
        charge_native=charge,
        segment_start_time_ns=retarded_time - 1.0,
        segment_duration_ns=2.0,
        position_coefficients_mm=coefficients,
        retarded_time_ns=retarded_time,
        four_velocity_mm_ns=(C_MMNS, velocity_x, 0.0, 0.0),
        four_acceleration_mm_ns2=(0.0, acceleration_x, 0.0, 0.0),
    )

    np.testing.assert_allclose(
        result.four_potential,
        (charge / distance, 0.0, 0.0, 0.0),
        rtol=0.0,
        atol=3.0e-17,
    )
    assert result.partial2_a[1, 1, 0] == pytest.approx(
        2.0 * charge / distance**3,
        rel=2.0e-15,
    )
    assert result.partial3_a_along_velocity[1, 1, 0] == pytest.approx(
        -6.0 * velocity_x * charge / distance**4,
        rel=2.0e-15,
    )
    assert result.partial3_a_along_acceleration[1, 1, 0] == pytest.approx(
        -6.0 * acceleration_x * charge / distance**4,
        rel=2.0e-15,
    )
    assert result.partial4_a_along_velocity_twice[1, 1, 0] == pytest.approx(
        24.0 * velocity_x**2 * charge / distance**5,
        rel=2.0e-15,
    )
    assert result.light_cone_jet_residual == 0.0


def test_static_dipole_returns_only_required_directional_derivatives() -> None:
    distance = 10.0
    moment = 2.3
    velocity_x = 0.2
    acceleration_x = 0.03
    retarded_time, coefficients = _static_segment(distance)
    spin_z = np.asarray((0.0, 0.0, 1.0))
    zero_slope = np.zeros(3)

    result = quintic_dipole_potential_directional_jet_native(
        observer_time_ns=0.0,
        observer_position_mm=(distance, 0.0, 0.0),
        magnetic_moment_native=moment,
        segment_start_time_ns=retarded_time - 1.0,
        segment_duration_ns=2.0,
        position_coefficients_mm=coefficients,
        rest_spin_start=spin_z,
        rest_spin_end=spin_z,
        rest_spin_start_derivative_per_ns=zero_slope,
        rest_spin_end_derivative_per_ns=zero_slope,
        preserved_rest_spin_magnitude=1.0,
        retarded_time_ns=retarded_time,
        four_velocity_mm_ns=(C_MMNS, velocity_x, 0.0, 0.0),
        four_acceleration_mm_ns2=(0.0, acceleration_x, 0.0, 0.0),
    )

    # For mu=mu zhat and r=R xhat, A_y=mu/R^2 in the maintained convention.
    np.testing.assert_allclose(
        result.four_potential,
        (0.0, 0.0, moment / distance**2, 0.0),
        rtol=0.0,
        atol=4.0e-18,
    )
    assert result.partial2_a[1, 1, 2] == pytest.approx(
        6.0 * moment / distance**4,
        rel=2.0e-15,
    )
    assert result.partial3_a_along_velocity[1, 1, 2] == pytest.approx(
        -24.0 * velocity_x * moment / distance**5,
        rel=2.0e-15,
    )
    assert result.partial3_a_along_acceleration[1, 1, 2] == pytest.approx(
        -24.0 * acceleration_x * moment / distance**5,
        rel=2.0e-15,
    )
    assert result.partial4_a_along_velocity_twice[1, 1, 2] == pytest.approx(
        120.0 * velocity_x**2 * moment / distance**6,
        rel=2.0e-15,
    )
    assert result.light_cone_jet_residual == 0.0


def test_moving_charge_matches_independent_second_order_potential_jet() -> None:
    duration = 0.004
    start_time = -0.006
    root_time = -0.003
    fraction = (root_time - start_time) / duration
    coefficients = np.asarray(
        (
            (0.2, -0.1, 0.3),
            (0.015, 0.006, -0.003),
            (3.0e-4, -2.0e-4, 1.0e-4),
            (-2.0e-5, 1.0e-5, 3.0e-5),
            (1.0e-6, 2.0e-6, -1.0e-6),
            (-1.0e-7, 1.0e-7, 2.0e-7),
        )
    )
    source_position = np.asarray(
        [
            np.polynomial.polynomial.polyval(fraction, coefficients[:, component])
            for component in range(3)
        ]
    )
    separation = np.asarray((2.4, 3.1, -1.7))
    observer_position = source_position + separation
    observer_time = root_time + float(np.linalg.norm(separation)) / C_MMNS
    arguments = {
        "observer_time_ns": observer_time,
        "observer_position_mm": observer_position,
        "charge_native": -1.37,
        "segment_start_time_ns": start_time,
        "segment_duration_ns": duration,
        "position_coefficients_mm": coefficients,
        "retarded_time_ns": root_time,
    }

    reference = quintic_charge_potential_jet_native(**arguments)
    directional = quintic_charge_potential_directional_jet_native(
        **arguments,
        four_velocity_mm_ns=(1.1 * C_MMNS, 20.0, -5.0, 3.0),
        four_acceleration_mm_ns2=(0.2, -0.1, 0.04, 0.03),
    )

    np.testing.assert_allclose(
        directional.four_potential,
        reference.four_potential,
        rtol=3.0e-15,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        directional.partial_a,
        reference.partial_a,
        rtol=8.0e-15,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        directional.partial2_a,
        reference.partial2_a,
        rtol=3.0e-14,
        atol=2.0e-12,
    )


def test_moving_rotating_dipole_matches_independent_third_order_hertz_jet() -> None:
    duration = 0.003
    start_time = -0.005
    root_time = -0.0032
    fraction = (root_time - start_time) / duration
    coefficients = np.asarray(
        (
            (-0.3, 0.1, 0.2),
            (0.022, -0.009, 0.004),
            (-4.0e-4, 2.0e-4, 1.0e-4),
            (3.0e-5, -2.0e-5, 1.0e-5),
            (-2.0e-6, 1.0e-6, 2.0e-6),
            (1.0e-7, -2.0e-7, 1.0e-7),
        )
    )
    source_position = np.asarray(
        [
            np.polynomial.polynomial.polyval(fraction, coefficients[:, component])
            for component in range(3)
        ]
    )
    separation = np.asarray((1.4, -0.8, 1.1))
    observer_position = source_position + separation
    observer_time = root_time + float(np.linalg.norm(separation)) / C_MMNS
    arguments = {
        "observer_time_ns": observer_time,
        "observer_position_mm": observer_position,
        "magnetic_moment_native": -1.2,
        "segment_start_time_ns": start_time,
        "segment_duration_ns": duration,
        "position_coefficients_mm": coefficients,
        "rest_spin_start": (0.8, 0.0, 0.6),
        "rest_spin_end": (0.3, 0.7, 0.64),
        "rest_spin_start_derivative_per_ns": (4.0, -3.0, 1.0),
        "rest_spin_end_derivative_per_ns": (-2.0, 5.0, -1.0),
        "preserved_rest_spin_magnitude": None,
        "retarded_time_ns": root_time,
    }

    reference = quintic_dipole_hertz_response_jet_native(**arguments)
    directional = quintic_dipole_potential_directional_jet_native(
        **arguments,
        four_velocity_mm_ns=(1.05 * C_MMNS, 18.0, 7.0, -4.0),
        four_acceleration_mm_ns2=(0.1, -0.08, 0.03, 0.02),
    )

    np.testing.assert_allclose(
        directional.four_potential,
        reference.four_potential,
        rtol=8.0e-15,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        directional.partial_a,
        reference.partial_a,
        rtol=2.0e-14,
        atol=2.0e-12,
    )
    signs = np.asarray((1.0, -1.0, -1.0, -1.0))
    reconstructed_partial_f = np.zeros((4, 4, 4))
    for derivative_index in range(4):
        for mu in range(4):
            for nu in range(4):
                reconstructed_partial_f[derivative_index, mu, nu] = (
                    signs[mu] * directional.partial2_a[derivative_index, mu, nu]
                    - signs[nu]
                    * directional.partial2_a[derivative_index, nu, mu]
                )
    np.testing.assert_allclose(
        reconstructed_partial_f,
        reference.partial_f,
        rtol=4.0e-13,
        atol=3.0e-10,
    )


@pytest.mark.parametrize(
    "evaluator,extra",
    (
        (quintic_charge_potential_directional_jet_native, {"charge_native": 1.0}),
        (
            quintic_dipole_potential_directional_jet_native,
            {
                "magnetic_moment_native": 1.0,
                "rest_spin_start": (0.0, 0.0, 1.0),
                "rest_spin_end": (0.0, 0.0, 1.0),
                "rest_spin_start_derivative_per_ns": (0.0, 0.0, 0.0),
                "rest_spin_end_derivative_per_ns": (0.0, 0.0, 0.0),
                "preserved_rest_spin_magnitude": 1.0,
            },
        ),
    ),
)
def test_higher_derivative_jet_rejects_segment_boundaries(
    evaluator: object,
    extra: dict[str, object],
) -> None:
    function = evaluator
    with pytest.raises(ValueError, match="strictly inside"):
        function(  # type: ignore[operator]
            observer_time_ns=0.0,
            observer_position_mm=(10.0, 0.0, 0.0),
            segment_start_time_ns=-1.0,
            segment_duration_ns=2.0,
            position_coefficients_mm=np.zeros((6, 3)),
            retarded_time_ns=-1.0,
            four_velocity_mm_ns=(C_MMNS, 0.0, 0.0, 0.0),
            four_acceleration_mm_ns2=(0.0, 0.0, 0.0, 0.0),
            **extra,
        )


def test_history_providers_sum_charge_and_dipole_smooth_segments() -> None:
    distance = 10.0
    times = np.linspace(-0.1, 0.0, 9)
    history = _static_history(times)
    event = ObserverEvent(time_ns=0.0, position_mm=(distance, 0.0, 0.0))
    velocity = (C_MMNS, 0.2, 0.0, 0.0)
    acceleration = (0.0, 0.03, 0.0, 0.0)

    charge = evaluate_retarded_charge_potential_directional_jet_native(
        history,
        event,
        four_velocity_mm_ns=velocity,
        four_acceleration_mm_ns2=acceleration,
    )
    dipole = evaluate_retarded_dipole_potential_directional_jet_native(
        history,
        event,
        source_identities=("source",),
        four_velocity_mm_ns=velocity,
        four_acceleration_mm_ns2=acceleration,
    )

    assert charge.available and charge.derivatives is not None
    assert dipole.available and dipole.derivatives is not None
    total = sum_potential_directional_derivatives_native(
        charge.derivatives,
        dipole.derivatives,
    )
    np.testing.assert_allclose(
        total.four_potential,
        (1.7 / distance, 0.0, 2.3 / distance**2, 0.0),
        rtol=2.0e-15,
        atol=4.0e-18,
    )
    assert total.partial3_a_along_velocity[1, 1, 0] == pytest.approx(
        -6.0 * 0.2 * 1.7 / distance**4,
        rel=3.0e-15,
    )
    assert total.partial3_a_along_velocity[1, 1, 2] == pytest.approx(
        -24.0 * 0.2 * 2.3 / distance**5,
        rel=3.0e-15,
    )
    np.testing.assert_array_equal(charge.valid_sources, (True,))
    np.testing.assert_array_equal(dipole.valid_sources, (True,))


def test_history_providers_report_boundary_derivative_unavailable() -> None:
    distance = 10.0
    times = np.linspace(-0.1, 0.0, 5)
    boundary_time = float(times[2])
    event = ObserverEvent(
        time_ns=boundary_time + distance / C_MMNS,
        position_mm=(distance, 0.0, 0.0),
    )
    history = _static_history(times)
    kwargs = {
        "four_velocity_mm_ns": (C_MMNS, 0.0, 0.0, 0.0),
        "four_acceleration_mm_ns2": (0.0, 0.0, 0.0, 0.0),
    }

    charge = evaluate_retarded_charge_potential_directional_jet_native(
        history,
        event,
        **kwargs,
    )
    dipole = evaluate_retarded_dipole_potential_directional_jet_native(
        history,
        event,
        source_identities=("source",),
        require_frozen_spin_segment=False,
        **kwargs,
    )

    assert charge.available is False and charge.derivatives is None
    assert dipole.available is False and dipole.derivatives is None
    assert charge.unavailable_reason is not None
    assert dipole.unavailable_reason is not None
    assert "segment-boundary guard" in charge.unavailable_reason
    assert "segment-boundary guard" in dipole.unavailable_reason


def test_retarded_history_evaluates_complete_potential_only_spin_reduction() -> None:
    distance = 10.0
    history = _static_history(np.linspace(-0.1, 0.0, 9))
    beta_y = 0.05
    gamma = 1.0 / np.sqrt(1.0 - beta_y**2)
    velocity = np.asarray((gamma * C_MMNS, 0.0, gamma * C_MMNS * beta_y, 0.0))
    normalized_spin = np.asarray((0.0, 0.0, 0.0, 1.0))

    result = evaluate_retarded_potential_intrinsic_spin_reduction_native(
        source_history=history,
        observer_event=ObserverEvent(0.0, (distance, 0.0, 0.0)),
        four_velocity_mm_ns=velocity,
        normalized_spin_four_vector=normalized_spin,
        charge_native=-0.5,
        mass_amu=1.0,
        invariant_spin_native=0.75,
        g_factor=2.1,
        dipole_source_identities=("source",),
    )

    assert result.available
    assert result.unavailable_reason is None
    assert result.reduction is not None
    assert result.leading_four_acceleration_mm_ns2 is not None
    np.testing.assert_array_equal(
        result.reduction.leading_dynamics.four_acceleration,
        result.leading_four_acceleration_mm_ns2,
    )
    assert np.all(
        np.isfinite(
            result.reduction.radiation_balance.self_force.linear_spin_self_force_native
        )
    )
    # The point source is off the observer and both source sectors are active.
    np.testing.assert_array_equal(result.charge_provider.valid_sources, (True,))
    np.testing.assert_array_equal(result.dipole_provider.valid_sources, (True,))


def test_retarded_reduction_matches_centered_and_causal_sampled_oracles() -> None:
    """One weak smooth leading trajectory checks both independent references."""

    source_history = _static_history(
        np.linspace(-1.0, 0.2, 49),
        charge_native=1000.0,
        magnetic_moment_native=0.0,
    )
    observer_charge = -1.0
    observer_mass = 1.0
    invariant_spin = 0.75
    g_factor = 2.1
    intrinsic_moment = (
        g_factor * observer_charge * invariant_spin / (2.0 * observer_mass * C_MMNS)
    )
    beta_y = 0.03
    gamma = 1.0 / np.sqrt(1.0 - beta_y**2)
    state = np.concatenate(
        (
            np.asarray((0.0, 10.0, 0.0, 0.0)),
            np.asarray((gamma * C_MMNS, 0.0, gamma * C_MMNS * beta_y, 0.0)),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        )
    )

    def leading_rhs(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coordinates = value[:4]
        velocity = value[4:8]
        spin = value[8:12]
        provider = evaluate_retarded_charge_potential_directional_jet_native(
            source_history,
            ObserverEvent(
                coordinates[0] / C_MMNS,
                tuple(coordinates[1:4]),
            ),
            four_velocity_mm_ns=velocity,
            four_acceleration_mm_ns2=np.zeros(4),
        )
        assert provider.available and provider.derivatives is not None
        derivatives = provider.derivatives
        leading = potential_derivative_rfs_response_native(
            four_velocity_mm_ns=velocity,
            spin_four_vector=spin,
            partial_a=derivatives.partial_a,
            partial2_a=derivatives.partial2_a,
            charge_native=observer_charge,
            mass_amu=observer_mass,
            magnetic_moment_native=intrinsic_moment,
            invariant_spin_native=invariant_spin,
        )
        acceleration = leading.total_four_force / observer_mass
        return np.concatenate((velocity, acceleration, leading.spin_rhs)), acceleration

    proper_step = 2.0e-4
    states = [state.copy()]
    accelerations = [leading_rhs(state)[1]]
    for _ in range(10):
        k1, _ = leading_rhs(state)
        k2, _ = leading_rhs(state + 0.5 * proper_step * k1)
        k3, _ = leading_rhs(state + 0.5 * proper_step * k2)
        k4, _ = leading_rhs(state + proper_step * k3)
        state += proper_step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        states.append(state.copy())
        accelerations.append(leading_rhs(state)[1])
    state_samples = np.asarray(states)
    acceleration_samples = np.asarray(accelerations)
    proper_times = proper_step * np.arange(len(states), dtype=float)

    def analytical_at(index: int) -> np.ndarray:
        sample = state_samples[index]
        result = evaluate_retarded_potential_intrinsic_spin_reduction_native(
            source_history=source_history,
            observer_event=ObserverEvent(
                sample[0] / C_MMNS,
                tuple(sample[1:4]),
            ),
            four_velocity_mm_ns=sample[4:8],
            normalized_spin_four_vector=sample[8:12],
            charge_native=observer_charge,
            mass_amu=observer_mass,
            invariant_spin_native=invariant_spin,
            g_factor=g_factor,
            dipole_source_identities=("source",),
        )
        assert result.available and result.reduction is not None
        return (
            result.reduction.radiation_balance.self_force.linear_spin_self_force_native
        )

    centered = evaluate_sampled_intrinsic_spin_reduction_native(
        proper_times_ns=proper_times[3:8],
        four_velocity_samples_mm_ns=state_samples[3:8, 4:8],
        non_self_four_acceleration_samples_mm_ns2=acceleration_samples[3:8],
        physical_spin_four_samples_native=(invariant_spin * state_samples[3:8, 8:12]),
        charge_native=observer_charge,
        mass_amu=observer_mass,
        g_factor=g_factor,
        center_index=2,
    )
    causal = evaluate_causal_sampled_intrinsic_spin_reduction_native(
        proper_times_ns=proper_times[-6:],
        four_velocity_samples_mm_ns=state_samples[-6:, 4:8],
        non_self_four_acceleration_samples_mm_ns2=acceleration_samples[-6:],
        physical_spin_four_samples_native=(invariant_spin * state_samples[-6:, 8:12]),
        charge_native=observer_charge,
        mass_amu=observer_mass,
        g_factor=g_factor,
    )

    centered_force = centered.radiation_balance.self_force.linear_spin_self_force_native
    causal_force = causal.radiation_balance.self_force.linear_spin_self_force_native
    centered_analytical = analytical_at(5)
    causal_analytical = analytical_at(10)
    assert np.linalg.norm(centered_force - centered_analytical) <= (
        2.0e-8 * np.linalg.norm(centered_analytical)
    )
    assert np.linalg.norm(causal_force - causal_analytical) <= (
        7.0e-7 * np.linalg.norm(causal_analytical)
    )
    assert np.max(np.abs(centered.velocity_derivative_residual_mm_ns2)) < 3.0e-11
    assert np.max(np.abs(causal.velocity_derivative_residual_mm_ns2)) < 6.0e-11
