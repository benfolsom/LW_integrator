from __future__ import annotations

import math

import numpy as np
import pytest

from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.medina_radiation_reaction import compute_medina_radiation_reaction
from core.radiation_flux_oracle import (
    gauss_legendre_sphere_quadrature,
    integrate_radiation_sphere_flux_native,
)
from core.rfs import MINKOWSKI_METRIC
from core.spin_self_force_oracle import (
    evaluate_jakobsen_linear_spin_self_force_native,
)


def _boost(beta: np.ndarray) -> np.ndarray:
    beta_squared = float(beta @ beta)
    gamma = 1.0 / math.sqrt(1.0 - beta_squared)
    result = np.eye(4)
    result[0, 0] = gamma
    result[0, 1:] = gamma * beta
    result[1:, 0] = gamma * beta
    if beta_squared > 0.0:
        result[1:, 1:] += (gamma - 1.0) * np.outer(beta, beta) / beta_squared
    return result


def _evaluate(**overrides: object):
    values: dict[str, object] = {
        "charge_native": -ELEMENTARY_CHARGE,
        "mass_amu": ELECTRON_MASS_AMU,
        "four_velocity_mm_ns": np.array((C_MMNS, 0.0, 0.0, 0.0)),
        "four_acceleration_mm_ns2": np.zeros(4),
        "four_jerk_mm_ns3": np.array((0.0, 0.7, -0.2, 0.4)),
        "four_snap_mm_ns4": np.array((0.0, -0.5, 0.3, 0.8)),
        "spin_four_vector_native": np.array((0.0, 0.2, -0.4, 0.7)),
        "spin_four_derivative_native": np.array((0.0, -0.3, 0.6, 0.1)),
        "magnetic_moment_four_vector_native": np.array((0.0, 1.1e-8, -0.4e-8, 0.8e-8)),
        "magnetic_moment_four_derivative_native": np.array(
            (0.0, -0.7e-8, 0.2e-8, 0.5e-8)
        ),
    }
    values.update(overrides)
    return evaluate_jakobsen_linear_spin_self_force_native(**values)


def test_rest_frame_formula_reduces_to_ordinary_three_vector_cross_products() -> None:
    charge = 0.8
    mass = 1.7
    jerk = np.array((0.3, -0.4, 0.2))
    snap = np.array((-0.1, 0.8, 0.5))
    spin = np.array((0.2, 0.3, -0.7))
    spin_derivative = np.array((-0.6, 0.1, 0.4))
    moment = np.array((1.1e-8, -0.9e-8, 0.2e-8))
    moment_derivative = np.array((0.3e-8, 0.7e-8, -0.5e-8))

    result = _evaluate(
        charge_native=charge,
        mass_amu=mass,
        four_jerk_mm_ns3=np.r_[0.0, jerk],
        four_snap_mm_ns4=np.r_[0.0, snap],
        spin_four_vector_native=np.r_[0.0, spin],
        spin_four_derivative_native=np.r_[0.0, spin_derivative],
        magnetic_moment_four_vector_native=np.r_[0.0, moment],
        magnetic_moment_four_derivative_native=np.r_[0.0, moment_derivative],
    )

    subtracted = moment - charge * spin / (mass * C_MMNS)
    subtracted_derivative = moment_derivative - charge * spin_derivative / (
        mass * C_MMNS
    )
    expected_bracket = (
        np.cross(jerk, moment_derivative)
        + np.cross(snap, subtracted)
        + np.cross(jerk, subtracted_derivative)
    )
    expected_force = 2.0 * charge / (3.0 * C_MMNS**4) * expected_bracket

    np.testing.assert_allclose(
        result.magnetization_bracket_native,
        np.r_[0.0, expected_bracket],
        rtol=3.0e-15,
        atol=1.0e-30,
    )
    np.testing.assert_allclose(
        result.linear_spin_self_force_native,
        np.r_[0.0, expected_force],
        rtol=3.0e-15,
        atol=1.0e-30,
    )


def test_derivative_includes_change_of_the_body_frame() -> None:
    acceleration = np.array((0.0, 0.0, 0.0, 0.4 * C_MMNS))
    jerk = np.array((0.0, 1.0, 0.0, 0.0))
    moment = np.array((0.0, 0.0, 2.0, 0.0))

    result = _evaluate(
        charge_native=0.5,
        mass_amu=2.0,
        four_acceleration_mm_ns2=acceleration,
        four_jerk_mm_ns3=jerk,
        four_snap_mm_ns4=np.zeros(4),
        spin_four_vector_native=np.zeros(4),
        spin_four_derivative_native=np.zeros(4),
        magnetic_moment_four_vector_native=moment,
        magnetic_moment_four_derivative_native=np.zeros(4),
    )

    # epsilon^0_(123)=+1, so the changing-frame contribution is
    # J_x M_y (A_z/c) in the temporal component.  The rest-frame projector
    # subsequently removes that temporal component from the force.
    assert result.cross_product_derivative_native[0] == pytest.approx(0.8)
    np.testing.assert_array_equal(result.cross_product_derivative_native[1:], 0.0)
    np.testing.assert_array_equal(result.projected_magnetization_bracket_native, 0.0)


def test_charge_ald_coefficient_matches_medina_in_instantaneous_rest_frame() -> None:
    mass = 1.9
    charge = -0.7
    jerk = np.array((0.4, -0.2, 0.9))
    result = _evaluate(
        charge_native=charge,
        mass_amu=mass,
        four_jerk_mm_ns3=np.r_[0.0, jerk],
        spin_four_vector_native=np.zeros(4),
        spin_four_derivative_native=np.zeros(4),
        magnetic_moment_four_vector_native=np.zeros(4),
        magnetic_moment_four_derivative_native=np.zeros(4),
    )
    medina = compute_medina_radiation_reaction(
        external_force=(0.0, 0.0, 0.0),
        external_force_time_derivative=mass * jerk,
        beta=(0.0, 0.0, 0.0),
        acceleration=(0.0, 0.0, 0.0),
        gamma=1.0,
        mass=mass,
        charge=charge,
        coordinate_dt=0.0,
    )

    np.testing.assert_allclose(
        result.charge_ald_self_force_native[1:],
        medina.radiation_reaction_force,
        rtol=3.0e-15,
        atol=0.0,
    )
    assert result.charge_ald_self_force_native[0] == 0.0


def test_result_is_covariant_under_a_constant_lorentz_boost() -> None:
    rest = _evaluate()
    boost = _boost(np.array((0.21, -0.13, 0.07)))
    transformed = _evaluate(
        four_velocity_mm_ns=boost @ np.array((C_MMNS, 0.0, 0.0, 0.0)),
        four_acceleration_mm_ns2=boost @ np.zeros(4),
        four_jerk_mm_ns3=boost @ np.array((0.0, 0.7, -0.2, 0.4)),
        four_snap_mm_ns4=boost @ np.array((0.0, -0.5, 0.3, 0.8)),
        spin_four_vector_native=boost @ np.array((0.0, 0.2, -0.4, 0.7)),
        spin_four_derivative_native=boost @ np.array((0.0, -0.3, 0.6, 0.1)),
        magnetic_moment_four_vector_native=boost
        @ np.array((0.0, 1.1e-8, -0.4e-8, 0.8e-8)),
        magnetic_moment_four_derivative_native=boost
        @ np.array((0.0, -0.7e-8, 0.2e-8, 0.5e-8)),
    )

    for field_name in (
        "intrinsic_subtracted_moment_native",
        "moment_derivative_cross_native",
        "cross_product_derivative_native",
        "magnetization_bracket_native",
        "projected_magnetization_bracket_native",
        "linear_spin_self_force_native",
        "charge_ald_self_force_native",
        "total_self_force_through_linear_spin_native",
    ):
        np.testing.assert_allclose(
            getattr(transformed, field_name),
            boost @ getattr(rest, field_name),
            rtol=7.0e-14,
            atol=2.0e-28,
        )


def test_projection_makes_the_four_force_orthogonal_to_velocity() -> None:
    boost = _boost(np.array((0.31, 0.08, -0.17)))
    result = _evaluate(
        four_velocity_mm_ns=boost @ np.array((C_MMNS, 0.0, 0.0, 0.0)),
        four_jerk_mm_ns3=boost @ np.array((0.0, 0.7, -0.2, 0.4)),
        four_snap_mm_ns4=boost @ np.array((0.0, -0.5, 0.3, 0.8)),
        spin_four_vector_native=boost @ np.array((0.0, 0.2, -0.4, 0.7)),
        spin_four_derivative_native=boost @ np.array((0.0, -0.3, 0.6, 0.1)),
        magnetic_moment_four_vector_native=boost
        @ np.array((0.0, 1.1e-8, -0.4e-8, 0.8e-8)),
        magnetic_moment_four_derivative_native=boost
        @ np.array((0.0, -0.7e-8, 0.2e-8, 0.5e-8)),
    )

    scale = max(
        np.linalg.norm(result.total_self_force_through_linear_spin_native) * C_MMNS,
        np.finfo(float).tiny,
    )
    assert abs(result.four_velocity_dot_linear_spin_force_native) < 2.0e-14 * scale
    assert abs(result.four_velocity_dot_total_force_native) < 2.0e-14 * scale


def test_neutral_moment_has_no_linear_or_charge_self_force() -> None:
    result = _evaluate(charge_native=0.0)

    np.testing.assert_array_equal(result.linear_spin_self_force_native, 0.0)
    np.testing.assert_array_equal(result.charge_ald_self_force_native, 0.0)
    np.testing.assert_array_equal(
        result.total_self_force_through_linear_spin_native, 0.0
    )


@pytest.mark.parametrize(
    ("polar_order", "azimuthal_order"),
    ((4, 8), (8, 16)),
)
def test_periodic_low_speed_q_mu_impulse_balances_independent_sphere_flux(
    polar_order: int, azimuthal_order: int
) -> None:
    """Close one periodic q-mu impulse without assuming a far-power formula.

    This is the leading slow-motion limit of a charge oscillating on x while
    a prescribed magnetization oscillates on y.  Setting physical spin to
    zero selects Jakobsen's generic magnetization/susceptibility term and
    removes the supplemental spin--radiative-field correction.  The local
    total derivative returns to its initial value after one period.

    The outward momentum is obtained independently: construct the standard
    electric- and magnetic-dipole radiation fields on a sphere, then let the
    maintained Maxwell-stress integrator determine their q-mu cross term.
    """

    charge = 0.8
    position_amplitude_mm = 0.03
    moment_amplitude_native = 1.4e-8
    angular_frequency_per_ns = 1.7
    period_ns = 2.0 * math.pi / angular_frequency_per_ns
    times_ns = np.linspace(0.0, period_ns, 65)
    radius_mm = 40.0
    quadrature = gauss_legendre_sphere_quadrature(
        polar_order=polar_order,
        azimuthal_order=azimuthal_order,
    )
    directions = quadrature.directions

    reaction_force_z = np.empty(times_ns.size)
    outward_momentum_rate_z = np.empty(times_ns.size)
    interference_energy_rate = np.empty(times_ns.size)
    for index, time_ns in enumerate(times_ns):
        phase = angular_frequency_per_ns * time_ns
        cosine = math.cos(phase)
        sine = math.sin(phase)
        acceleration_x = -position_amplitude_mm * angular_frequency_per_ns**2 * cosine
        jerk_x = position_amplitude_mm * angular_frequency_per_ns**3 * sine
        snap_x = position_amplitude_mm * angular_frequency_per_ns**4 * cosine
        moment_y = moment_amplitude_native * cosine
        moment_derivative_y = -moment_amplitude_native * angular_frequency_per_ns * sine
        moment_second_derivative_y = (
            -moment_amplitude_native * angular_frequency_per_ns**2 * cosine
        )

        local = _evaluate(
            charge_native=charge,
            mass_amu=1.0,
            four_acceleration_mm_ns2=(0.0, acceleration_x, 0.0, 0.0),
            four_jerk_mm_ns3=(0.0, jerk_x, 0.0, 0.0),
            four_snap_mm_ns4=(0.0, snap_x, 0.0, 0.0),
            spin_four_vector_native=np.zeros(4),
            spin_four_derivative_native=np.zeros(4),
            magnetic_moment_four_vector_native=(0.0, 0.0, moment_y, 0.0),
            magnetic_moment_four_derivative_native=(
                0.0,
                0.0,
                moment_derivative_y,
                0.0,
            ),
        )
        reaction_force_z[index] = local.linear_spin_self_force_native[3]

        electric_dipole_second_derivative = np.array(
            (charge * acceleration_x, 0.0, 0.0)
        )
        magnetic_dipole_second_derivative = np.array(
            (0.0, moment_second_derivative_y, 0.0)
        )
        charge_electric = np.cross(
            directions,
            np.cross(
                directions,
                electric_dipole_second_derivative[np.newaxis, :],
            ),
        ) / (C_MMNS**2 * radius_mm)
        charge_magnetic = np.cross(directions, charge_electric)
        dipole_magnetic = np.cross(
            directions,
            np.cross(
                directions,
                magnetic_dipole_second_derivative[np.newaxis, :],
            ),
        ) / (C_MMNS**2 * radius_mm)
        dipole_electric = -np.cross(directions, dipole_magnetic)
        flux = integrate_radiation_sphere_flux_native(
            quadrature=quadrature,
            radius_mm=radius_mm,
            charge_electric_field_native=charge_electric,
            charge_magnetic_field_native=charge_magnetic,
            dipole_electric_field_native=dipole_electric,
            dipole_magnetic_field_native=dipole_magnetic,
        )
        outward_momentum_rate_z[index] = flux.q_mu_interference.momentum_rate_native[2]
        interference_energy_rate[index] = flux.q_mu_interference.energy_rate_native

    intervals = np.diff(times_ns)
    reaction_impulse_z = float(
        np.sum(0.5 * (reaction_force_z[:-1] + reaction_force_z[1:]) * intervals)
    )
    outward_momentum_z = float(
        np.sum(
            0.5
            * (outward_momentum_rate_z[:-1] + outward_momentum_rate_z[1:])
            * intervals
        )
    )
    expected_outward_momentum_z = (
        charge
        * position_amplitude_mm
        * moment_amplitude_native
        * angular_frequency_per_ns**4
        * period_ns
        / (3.0 * C_MMNS**4)
    )

    assert outward_momentum_z == pytest.approx(expected_outward_momentum_z, rel=2.0e-14)
    assert reaction_impulse_z == pytest.approx(
        -expected_outward_momentum_z, rel=2.0e-14
    )
    assert reaction_impulse_z + outward_momentum_z == pytest.approx(
        0.0, abs=3.0e-14 * abs(expected_outward_momentum_z)
    )
    assert np.max(np.abs(interference_energy_rate)) < (
        2.0e-14 * C_MMNS * np.max(np.abs(outward_momentum_rate_z))
    )


def test_static_g_two_intrinsic_moment_cancels_linear_spin_correction() -> None:
    charge = -0.9
    mass = 1.3
    spin = np.array((0.0, 0.2, -0.4, 0.7))
    minimal_moment = charge * spin / (mass * C_MMNS)
    result = _evaluate(
        charge_native=charge,
        mass_amu=mass,
        spin_four_vector_native=spin,
        spin_four_derivative_native=np.zeros(4),
        magnetic_moment_four_vector_native=minimal_moment,
        magnetic_moment_four_derivative_native=np.zeros(4),
    )

    np.testing.assert_allclose(
        result.intrinsic_subtracted_moment_native, 0.0, rtol=0.0, atol=2.0e-30
    )
    np.testing.assert_allclose(
        result.linear_spin_self_force_native, 0.0, rtol=0.0, atol=2.0e-30
    )
    np.testing.assert_array_equal(result.linear_spin_self_torque_native, 0.0)


def test_input_validation_rejects_bad_mass_and_nonphysical_velocity() -> None:
    with pytest.raises(ValueError, match="mass_amu"):
        _evaluate(mass_amu=0.0)
    with pytest.raises(ValueError, match="future-directed"):
        _evaluate(four_velocity_mm_ns=(-C_MMNS, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="u.u"):
        _evaluate(four_velocity_mm_ns=(C_MMNS, 1.0, 0.0, 0.0))


def test_metric_and_boost_helper_are_consistent() -> None:
    boost = _boost(np.array((0.2, -0.1, 0.3)))
    np.testing.assert_allclose(
        boost.T @ MINKOWSKI_METRIC @ boost,
        MINKOWSKI_METRIC,
        rtol=3.0e-15,
        atol=3.0e-16,
    )
