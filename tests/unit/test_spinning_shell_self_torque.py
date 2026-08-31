from __future__ import annotations

import math

import numpy as np
import pytest

from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.external_fields import AMU_KG
from core.magnetic_dipole import NATIVE_ACTION_UNIT_J_S
from core.radiation_flux_oracle import (
    gauss_legendre_sphere_quadrature,
    integrate_radiation_sphere_flux_native,
)
from core.spinning_shell_self_torque import (
    count_harmonic_spinning_shell_transfer_poles_native,
    evaluate_harmonic_spinning_shell_response_native,
    evaluate_harmonic_spinning_shell_transfer_native,
    evaluate_spinning_shell_angular_balance_native,
    evaluate_spinning_shell_local_self_torque_native,
    reconstruct_harmonic_spinning_shell_impulse_response_native,
)


def _harmonic_derivatives(
    *, moment_native: float, angular_frequency_per_ns: float, time_ns: float
) -> np.ndarray:
    return np.asarray(
        [
            moment_native
            * angular_frequency_per_ns**order
            * math.cos(angular_frequency_per_ns * time_ns + order * math.pi / 2.0)
            for order in range(9)
        ]
    )


def test_static_shell_stores_constant_field_angular_momentum() -> None:
    derivatives = np.zeros(9)
    derivatives[0] = 2.3e-8

    result = evaluate_spinning_shell_angular_balance_native(
        charge_native=ELEMENTARY_CHARGE,
        shell_radius_mm=0.04,
        observation_radius_mm=12.0,
        shell_retarded_moment_derivatives_native=derivatives,
        observation_retarded_moment_derivatives_native=derivatives,
    )

    assert result.near_field_angular_momentum_native > 0.0
    assert result.wave_zone_angular_momentum_native == 0.0
    assert result.field_angular_momentum_native == (
        result.near_field_angular_momentum_native
    )
    assert result.self_torque_native == 0.0
    assert result.outward_angular_momentum_rate_native == 0.0
    assert result.field_angular_momentum_rate_native == 0.0
    assert result.balance_residual_native == 0.0


def test_retarded_shell_angular_momentum_balance_closes_for_harmonic_moment() -> None:
    moment_native = 1.7e-8
    angular_frequency_per_ns = 0.8
    observation_time_ns = 0.37

    for shell_radius_mm in (0.08, 0.04, 0.02):
        observation_radius_mm = 25.0
        shell_time_ns = observation_time_ns - shell_radius_mm / C_MMNS
        observer_time_ns = observation_time_ns - observation_radius_mm / C_MMNS
        result = evaluate_spinning_shell_angular_balance_native(
            charge_native=-ELEMENTARY_CHARGE,
            shell_radius_mm=shell_radius_mm,
            observation_radius_mm=observation_radius_mm,
            shell_retarded_moment_derivatives_native=_harmonic_derivatives(
                moment_native=moment_native,
                angular_frequency_per_ns=angular_frequency_per_ns,
                time_ns=shell_time_ns,
            ),
            observation_retarded_moment_derivatives_native=_harmonic_derivatives(
                moment_native=moment_native,
                angular_frequency_per_ns=angular_frequency_per_ns,
                time_ns=observer_time_ns,
            ),
        )

        scale = max(
            abs(result.self_torque_native),
            abs(result.outward_angular_momentum_rate_native),
            abs(result.field_angular_momentum_rate_native),
        )
        assert abs(result.balance_residual_native) < 4.0e-15 * scale


def test_local_shell_torque_separates_reversible_and_radiative_terms() -> None:
    derivatives = _harmonic_derivatives(
        moment_native=2.1e-8,
        angular_frequency_per_ns=1.3,
        time_ns=0.23,
    )
    first = evaluate_spinning_shell_local_self_torque_native(
        charge_native=ELEMENTARY_CHARGE,
        shell_radius_mm=0.08,
        current_moment_derivatives_native=derivatives,
    )
    half = evaluate_spinning_shell_local_self_torque_native(
        charge_native=ELEMENTARY_CHARGE,
        shell_radius_mm=0.04,
        current_moment_derivatives_native=derivatives,
    )

    assert first.total_self_torque_native == pytest.approx(
        first.time_symmetric_torque_native + first.radiation_reaction_torque_native,
        rel=0.0,
        abs=0.0,
    )
    assert first.time_symmetric_torque_native != 0.0
    assert first.radiation_reaction_torque_native != 0.0

    # The leading time-symmetric term is proportional to 1/R, while the
    # leading radiation-reaction term is proportional to R^2.  The small
    # higher-order corrections make these ratios only asymptotically exact.
    assert half.time_symmetric_torque_native / first.time_symmetric_torque_native == (
        pytest.approx(2.0, rel=2.0e-7)
    )
    reaction_ratio = (
        half.radiation_reaction_torque_native
        / first.radiation_reaction_torque_native
    )
    assert reaction_ratio == pytest.approx(0.25, rel=2.0e-7)


def test_shell_oracle_rejects_invalid_geometry_and_derivatives() -> None:
    derivatives = np.zeros(9)
    with pytest.raises(ValueError, match="exceed"):
        evaluate_spinning_shell_angular_balance_native(
            charge_native=ELEMENTARY_CHARGE,
            shell_radius_mm=1.0,
            observation_radius_mm=1.0,
            shell_retarded_moment_derivatives_native=derivatives,
            observation_retarded_moment_derivatives_native=derivatives,
        )
    with pytest.raises(ValueError, match="0 through 8"):
        evaluate_spinning_shell_local_self_torque_native(
            charge_native=ELEMENTARY_CHARGE,
            shell_radius_mm=1.0,
            current_moment_derivatives_native=np.zeros(8),
        )


@pytest.mark.parametrize("dimensionless_frequency", (1.0e-5, 0.1, 1.0, 3.0))
def test_exact_harmonic_shell_self_work_balances_radiated_power(
    dimensionless_frequency: float,
) -> None:
    radius_mm = 0.7
    frequency_per_ns = dimensionless_frequency * C_MMNS / radius_mm
    angular_velocity_per_ns = (0.02 - 0.01j) * C_MMNS / radius_mm

    result = evaluate_harmonic_spinning_shell_response_native(
        charge_native=-ELEMENTARY_CHARGE,
        shell_radius_mm=radius_mm,
        drive_angular_frequency_per_ns=frequency_per_ns,
        angular_velocity_amplitude_per_ns=angular_velocity_per_ns,
    )

    assert result.dimensionless_frequency == pytest.approx(
        dimensionless_frequency, rel=2.0e-16
    )
    assert result.maximum_surface_beta == pytest.approx(
        abs(0.02 - 0.01j), rel=2.0e-16
    )
    assert result.radiated_power_native > 0.0
    assert result.average_self_torque_work_rate_native < 0.0
    assert result.average_self_torque_work_rate_native == pytest.approx(
        -result.radiated_power_native,
        rel=3.0e-14,
        abs=0.0,
    )
    assert abs(result.average_power_balance_residual_native) < (
        3.0e-14 * result.radiated_power_native
    )


def test_exact_shell_point_limit_matches_independent_sphere_flux() -> None:
    radius_mm = 0.7
    dimensionless_frequency = 0.02
    frequency_per_ns = dimensionless_frequency * C_MMNS / radius_mm
    response = evaluate_harmonic_spinning_shell_response_native(
        charge_native=ELEMENTARY_CHARGE,
        shell_radius_mm=radius_mm,
        drive_angular_frequency_per_ns=frequency_per_ns,
        angular_velocity_amplitude_per_ns=0.01 * C_MMNS / radius_mm,
    )

    # At the phase of maximum moment acceleration, the point-dipole sphere
    # flux is twice its cycle average.  This is an independent Maxwell-stress
    # calculation, not the shell power formula evaluated a second way.
    quadrature = gauss_legendre_sphere_quadrature(polar_order=24, azimuthal_order=48)
    directions = quadrature.directions
    moment_second_derivative = np.array(
        (
            0.0,
            0.0,
            response.magnetic_moment_amplitude_native.real * frequency_per_ns**2,
        )
    )
    magnetic = np.cross(
        directions,
        np.cross(
            directions,
            np.broadcast_to(moment_second_derivative, directions.shape),
        ),
    ) / (C_MMNS**2 * 5.0)
    electric = -np.cross(directions, magnetic)
    zeros = np.zeros_like(electric)
    sphere = integrate_radiation_sphere_flux_native(
        quadrature=quadrature,
        radius_mm=5.0,
        charge_electric_field_native=zeros,
        charge_magnetic_field_native=zeros,
        dipole_electric_field_native=electric,
        dipole_magnetic_field_native=magnetic,
    )
    sphere_average_power = 0.5 * sphere.mu_squared.energy_rate_native

    assert response.point_dipole_radiated_power_native == pytest.approx(
        sphere_average_power,
        rel=3.0e-14,
    )
    assert response.radiated_power_native / sphere_average_power == pytest.approx(
        response.finite_size_power_ratio,
        rel=3.0e-14,
    )
    assert response.finite_size_power_ratio == pytest.approx(
        1.0,
        rel=1.0e-4,
    )


def test_exact_shell_reduces_to_bonga_radiation_torque_at_low_frequency() -> None:
    charge_native = -ELEMENTARY_CHARGE
    radius_mm = 0.7
    surface_beta = 0.01
    angular_velocity_per_ns = surface_beta * C_MMNS / radius_mm
    relative_errors = []

    for dimensionless_frequency in (0.2, 0.1):
        frequency_per_ns = dimensionless_frequency * C_MMNS / radius_mm
        exact = evaluate_harmonic_spinning_shell_response_native(
            charge_native=charge_native,
            shell_radius_mm=radius_mm,
            drive_angular_frequency_per_ns=frequency_per_ns,
            angular_velocity_amplitude_per_ns=angular_velocity_per_ns,
        )
        local = evaluate_spinning_shell_local_self_torque_native(
            charge_native=charge_native,
            shell_radius_mm=radius_mm,
            current_moment_derivatives_native=_harmonic_derivatives(
                moment_native=exact.magnetic_moment_amplitude_native.real,
                angular_frequency_per_ns=frequency_per_ns,
                time_ns=0.0,
            ),
        )

        exact_in_phase_torque = exact.self_torque_amplitude_native.real
        assert exact_in_phase_torque < 0.0
        assert local.radiation_reaction_torque_native < 0.0
        relative_errors.append(
            abs(
                local.radiation_reaction_torque_native - exact_in_phase_torque
            )
            / abs(exact_in_phase_torque)
        )

    # Bonga's retained x^4, x^6, and x^8 dissipative terms are the beginning
    # of the exact harmonic response.  The first omitted relative term is
    # O(x^6), so halving x should reduce this error by about 64.
    assert relative_errors[1] < relative_errors[0] / 50.0
    assert relative_errors[1] < 1.0e-9


def test_exact_harmonic_shell_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        evaluate_harmonic_spinning_shell_response_native(
            charge_native=ELEMENTARY_CHARGE,
            shell_radius_mm=1.0,
            drive_angular_frequency_per_ns=0.0,
            angular_velocity_amplitude_per_ns=1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        evaluate_harmonic_spinning_shell_response_native(
            charge_native=ELEMENTARY_CHARGE,
            shell_radius_mm=1.0,
            drive_angular_frequency_per_ns=1.0,
            angular_velocity_amplitude_per_ns=complex(np.nan, 0.0),
        )


def test_complex_transfer_matches_exact_real_frequency_response() -> None:
    radius_mm = 0.7
    dimensionless_frequency = 0.02
    frequency_per_ns = dimensionless_frequency * C_MMNS / radius_mm
    response = evaluate_harmonic_spinning_shell_response_native(
        charge_native=-ELEMENTARY_CHARGE,
        shell_radius_mm=radius_mm,
        drive_angular_frequency_per_ns=frequency_per_ns,
        angular_velocity_amplitude_per_ns=1.0,
    )
    transfer = evaluate_harmonic_spinning_shell_transfer_native(
        charge_native=-ELEMENTARY_CHARGE,
        shell_radius_mm=radius_mm,
        shell_mass_amu=ELECTRON_MASS_AMU,
        friction_coefficient_native=0.0,
        complex_angular_frequency_per_ns=frequency_per_ns,
    )

    assert transfer.dimensionless_complex_frequency == pytest.approx(
        dimensionless_frequency,
        rel=2.0e-16,
    )
    assert transfer.radiation_reaction_coefficient_native == pytest.approx(
        response.radiation_reaction_coefficient_native,
        rel=2.0e-14,
    )
    assert transfer.mechanical_moment_of_inertia_kg_m2 > 0.0
    assert transfer.angular_velocity_per_torque_native == pytest.approx(
        1.0j / transfer.denominator_native,
        rel=0.0,
        abs=0.0,
    )


def test_argument_principle_distinguishes_exact_and_truncated_causality() -> None:
    common = {
        "charge_native": -ELEMENTARY_CHARGE,
        "shell_radius_mm": 1.0e-6,
        "shell_mass_amu": ELECTRON_MASS_AMU,
        "friction_coefficient_native": 0.0,
    }
    exact_counts = []
    for bound in (100.0, 200.0, 400.0):
        exact_counts.append(
            count_harmonic_spinning_shell_transfer_poles_native(
                real_dimensionless_bounds=(-bound, bound),
                imaginary_dimensionless_bounds=(1.0e-4, bound),
                response_model="exact",
                samples_per_edge=2048,
                **common,
            )
        )

    approximate_counts = []
    for samples_per_edge in (1024, 2048):
        approximate_counts.append(
            count_harmonic_spinning_shell_transfer_poles_native(
                real_dimensionless_bounds=(-200.0, 200.0),
                imaginary_dimensionless_bounds=(1.0e-4, 200.0),
                response_model="small_radius_truncation",
                samples_per_edge=samples_per_edge,
                **common,
            )
        )

    for exact in exact_counts:
        assert exact.zero_count == 0
        assert exact.winding_rounding_residual < 1.0e-12
        assert exact.minimum_denominator_magnitude_native > 0.0
    for approximate in approximate_counts:
        assert approximate.zero_count == 2
        assert approximate.winding_rounding_residual < 1.0e-12
        assert approximate.minimum_denominator_magnitude_native > 0.0


def test_argument_principle_finds_exact_response_poles_below_real_axis() -> None:
    result = count_harmonic_spinning_shell_transfer_poles_native(
        charge_native=-ELEMENTARY_CHARGE,
        shell_radius_mm=1.0e-6,
        shell_mass_amu=ELECTRON_MASS_AMU,
        friction_coefficient_native=0.0,
        real_dimensionless_bounds=(-20.0, 20.0),
        imaginary_dimensionless_bounds=(-20.0, -1.0e-4),
        samples_per_edge=2048,
        response_model="exact",
    )

    assert result.zero_count == 13
    assert result.winding_rounding_residual < 1.0e-12


def test_transfer_and_pole_count_reject_invalid_controls() -> None:
    with pytest.raises(ValueError, match="response_model"):
        evaluate_harmonic_spinning_shell_transfer_native(
            charge_native=ELEMENTARY_CHARGE,
            shell_radius_mm=1.0,
            shell_mass_amu=ELECTRON_MASS_AMU,
            friction_coefficient_native=0.0,
            complex_angular_frequency_per_ns=1.0,
            response_model="unknown",
        )
    with pytest.raises(ValueError, match="at least 16"):
        count_harmonic_spinning_shell_transfer_poles_native(
            charge_native=ELEMENTARY_CHARGE,
            shell_radius_mm=1.0,
            shell_mass_amu=ELECTRON_MASS_AMU,
            friction_coefficient_native=0.0,
            real_dimensionless_bounds=(-1.0, 1.0),
            imaginary_dimensionless_bounds=(0.1, 1.0),
            samples_per_edge=8,
        )


def _friction_for_dimensionless_damping(
    *, shell_radius_mm: float, dimensionless_friction: float
) -> float:
    radius_m = shell_radius_mm * 1.0e-3
    inertia = (
        2.0
        * ELECTRON_MASS_AMU
        * AMU_KG
        * radius_m**2
        / 3.0
    )
    inertial_action_si = inertia * (C_MMNS * 1.0e6) / radius_m
    return (
        dimensionless_friction
        * inertial_action_si
        / NATIVE_ACTION_UNIT_J_S
    )


def test_exact_impulse_response_converges_to_zero_before_impulse() -> None:
    radius_mm = 1.0e-6
    friction = _friction_for_dimensionless_damping(
        shell_radius_mm=radius_mm,
        dimensionless_friction=0.1,
    )
    times = np.array((-0.2, -0.1, -0.05, -0.02, -0.01, 0.01, 0.05, 0.2))
    common = {
        "charge_native": -ELEMENTARY_CHARGE,
        "shell_radius_mm": radius_mm,
        "shell_mass_amu": ELECTRON_MASS_AMU,
        "friction_coefficient_native": friction,
        "dimensionless_times": times,
        "response_model": "exact",
    }
    coarse = reconstruct_harmonic_spinning_shell_impulse_response_native(
        max_abs_dimensionless_frequency=200.0,
        frequency_sample_count=20001,
        **common,
    )
    fine = reconstruct_harmonic_spinning_shell_impulse_response_native(
        max_abs_dimensionless_frequency=400.0,
        frequency_sample_count=40001,
        **common,
    )

    assert coarse.inertial_reference_subtracted
    assert fine.inertial_reference_subtracted
    assert fine.dimensionless_friction == pytest.approx(0.1, rel=2.0e-16)
    assert coarse.maximum_preimpulse_absolute_response < 2.0e-9
    assert fine.maximum_preimpulse_absolute_response < 1.0e-9
    np.testing.assert_allclose(
        fine.normalized_angular_velocity_response[times > 0.0],
        coarse.normalized_angular_velocity_response[times > 0.0],
        rtol=0.0,
        atol=2.0e-9,
    )
    assert fine.maximum_imaginary_residual < 2.0e-14
    assert not fine.dimensionless_times.flags.writeable
    assert not fine.normalized_angular_velocity_response.flags.writeable


def test_truncated_impulse_response_retains_converged_preimpulse_signal() -> None:
    radius_mm = 1.0e-6
    friction = _friction_for_dimensionless_damping(
        shell_radius_mm=radius_mm,
        dimensionless_friction=0.1,
    )
    times = np.array((-0.2, -0.1, -0.05, -0.02, -0.01, -0.005, 0.02, 0.1))
    common = {
        "charge_native": -ELEMENTARY_CHARGE,
        "shell_radius_mm": radius_mm,
        "shell_mass_amu": ELECTRON_MASS_AMU,
        "friction_coefficient_native": friction,
        "dimensionless_times": times,
        "response_model": "small_radius_truncation",
    }
    coarse = reconstruct_harmonic_spinning_shell_impulse_response_native(
        max_abs_dimensionless_frequency=400.0,
        frequency_sample_count=80001,
        **common,
    )
    fine = reconstruct_harmonic_spinning_shell_impulse_response_native(
        max_abs_dimensionless_frequency=800.0,
        frequency_sample_count=160001,
        **common,
    )

    assert not fine.inertial_reference_subtracted
    assert fine.maximum_preimpulse_absolute_response > 0.3
    np.testing.assert_allclose(
        fine.normalized_angular_velocity_response,
        coarse.normalized_angular_velocity_response,
        rtol=0.0,
        atol=5.0e-3,
    )
    assert fine.maximum_imaginary_residual < 2.0e-14


def test_impulse_response_rejects_zero_friction_and_even_grid() -> None:
    common = {
        "charge_native": ELEMENTARY_CHARGE,
        "shell_radius_mm": 1.0,
        "shell_mass_amu": ELECTRON_MASS_AMU,
        "dimensionless_times": (-0.1, 0.1),
        "max_abs_dimensionless_frequency": 10.0,
    }
    with pytest.raises(ValueError, match="positive"):
        reconstruct_harmonic_spinning_shell_impulse_response_native(
            friction_coefficient_native=0.0,
            frequency_sample_count=257,
            **common,
        )
    with pytest.raises(ValueError, match="odd"):
        reconstruct_harmonic_spinning_shell_impulse_response_native(
            friction_coefficient_native=1.0,
            frequency_sample_count=258,
            **common,
        )
