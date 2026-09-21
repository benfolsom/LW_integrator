"""Derivative, charge-limit and retained-order tests for experimental coupling."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from core.constants import C_MMNS as c
from core.jakobsen import ordinary_response_native, ordinary_force_rate_native
from core.jakobsen_reaction import reaction_response_native
from core.rfs import electromagnetic_field_tensor_native


def case(g=2.0, spin_scale=1e-3):
    w = np.array([0.4, -0.2, 0.8])
    gamma = np.sqrt(1 + w @ w)
    u = c * np.r_[gamma, w]
    rest = c * spin_scale * np.array([0.1, 0.2, -0.3])
    s0 = w @ rest
    spin = np.r_[s0, rest + w * s0 / (1 + gamma)]
    f = (
        c
        * c
        * electromagnetic_field_tensor_native([0.02, 0.01, -0.01], [0.1, 0.2, 0.3])
    )
    df = np.zeros((4, 4, 4))
    df[0] = 0.2 * f
    return dict(
        four_velocity_mm_ns=u,
        spin_angular_momentum=spin,
        field_tensor=f,
        partial_f=df,
        charge_native=1.0,
        mass_amu=1.0,
        g=g,
    )


@pytest.mark.parametrize("g", [0.0, 2.0, 5.5856946893])
def test_directional_force_derivative_matches_independent_leading_trajectory(g):
    args = case(g)
    signs = np.array([1, -1, -1, -1])
    f = args["field_tensor"]

    def rhs(t, y):
        u = y[1:5]
        s = y[5:9]
        field = f * (1 + 0.2 * y[0])
        electric = field @ (signs * u) / c
        sd = (
            g / (2 * c) * field @ (signs * s)
            + (g / 2 - 1) * u * ((s * signs) @ electric) / c**2
        )
        return np.r_[u[0], electric, sd]

    y0 = np.r_[0.0, args["four_velocity_mm_ns"], args["spin_angular_momentum"]]
    h = 1e-5 / c
    forces = []
    for end in (-h, h):
        y = solve_ivp(rhs, (0, end), y0, method="DOP853", rtol=1e-12, atol=1e-14).y[
            :, -1
        ]
        shifted = dict(
            args,
            four_velocity_mm_ns=y[1:5],
            spin_angular_momentum=y[5:9],
            field_tensor=f * (1 + 0.2 * y[0]),
        )
        forces.append(ordinary_response_native(**shifted).four_force)
    expected = (forces[1] - forces[0]) / (2 * h)
    response = ordinary_response_native(**args)
    force0 = f @ (signs * args["four_velocity_mm_ns"]) / c
    expected += f @ (signs * (response.four_force - force0)) / c
    actual = ordinary_force_rate_native(
        **args, partial_f_proper_rate=np.zeros((4, 4, 4))
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-5)


@pytest.mark.parametrize("spin_scale", [0.0, 1e-3])
@pytest.mark.parametrize("g", [0.0, 2.0, 5.5856946893])
def test_charge_medina_equals_projected_ald_and_total_force_is_transverse(
    g, spin_scale
):
    args = case(g, spin_scale)
    result = reaction_response_native(**args, partial_f_proper_rate=np.zeros((4, 4, 4)))
    np.testing.assert_allclose(result.charge_ald_agreement_residual, 0, atol=1e-14)
    u = args["four_velocity_mm_ns"]
    assert abs((u * np.array([1, -1, -1, -1])) @ result.four_force) < 1e-8
    if spin_scale == 0:
        np.testing.assert_array_equal(result.intrinsic_spin_reaction, np.zeros(4))
        np.testing.assert_array_equal(
            result.charge_reaction_spin_correction, np.zeros(4)
        )
        np.testing.assert_array_equal(result.radiative_balance_correction, np.zeros(4))
        ordinary = ordinary_response_native(**args)
        np.testing.assert_allclose(
            result.four_force, ordinary.four_force + result.leading_charge_reaction
        )


def test_reaction_is_spin_linear_and_balance_term_is_not_a_force():
    responses = [
        reaction_response_native(
            **case(5.5856946893, s), partial_f_proper_rate=np.zeros((4, 4, 4))
        )
        for s in (0.0, 0.02, 0.04)
    ]
    for name in (
        "four_force",
        "spin_four_rate",
        "spin_momentum_offset_rate",
        "intrinsic_spin_reaction",
        "charge_reaction_spin_correction",
    ):
        a, b, d = [getattr(r, name) for r in responses]
        np.testing.assert_allclose(d - a, 2 * (b - a), rtol=2e-8, atol=1e-10)
    args = case(5.5856946893, 0.02)
    result = responses[1]
    ordinary = ordinary_response_native(**args)
    canonical_change = (
        result.four_force
        + result.spin_momentum_offset_rate
        - ordinary.four_force
        - ordinary.spin_momentum_offset_rate
    )
    expected = (
        result.leading_charge_reaction
        + result.charge_reaction_spin_correction
        + result.intrinsic_spin_reaction
        + result.radiative_balance_correction
    )
    np.testing.assert_allclose(canonical_change, expected, rtol=1e-8, atol=1e-11)


def test_missing_gradient_derivative_rejected():
    with pytest.raises(ValueError, match="derivative required"):
        reaction_response_native(**case(), partial_f_proper_rate=None)


def test_spin_charge_reaction_matches_medina_odd_part_without_small_signal_subtraction():
    from core.jakobsen_reaction import _medina_four_force
    from core.jakobsen import _ordinary_force_rate_parts_native

    args = case(5.5856946893, 0.02)
    ordinary = ordinary_response_native(**args)
    rate0, rate_s = _ordinary_force_rate_parts_native(
        **args, partial_f_proper_rate=np.zeros((4, 4, 4))
    )
    u = args["four_velocity_mm_ns"]
    f0 = args["field_tensor"] @ (u * np.array([1, -1, -1, -1])) / c
    fs = ordinary.spin_four_force
    expected = 0.5 * (
        _medina_four_force(u, f0 + fs, rate0 + rate_s, 1.0, 1.0)
        - _medina_four_force(u, f0 - fs, rate0 - rate_s, 1.0, 1.0)
    )
    result = reaction_response_native(**args, partial_f_proper_rate=np.zeros((4, 4, 4)))
    np.testing.assert_allclose(
        result.charge_reaction_spin_correction, expected, rtol=1e-10, atol=1e-16
    )
    tiny = reaction_response_native(
        **case(5.5856946893, 2e-18), partial_f_proper_rate=np.zeros((4, 4, 4))
    )
    np.testing.assert_allclose(
        tiny.charge_reaction_spin_correction,
        result.charge_reaction_spin_correction * 1e-16,
        rtol=2e-14,
        atol=1e-35,
    )
