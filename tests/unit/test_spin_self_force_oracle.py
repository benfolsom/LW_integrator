from __future__ import annotations

import math

import numpy as np
import pytest

from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.medina_radiation_reaction import compute_medina_radiation_reaction
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
