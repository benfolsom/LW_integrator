"""Native units, constraints, and spin-order momentum inversion contracts."""

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.jakobsen import (
    ordinary_response_native,
    canonical_momentum_native,
    velocity_from_canonical_spatial_native,
)
from core.rfs import electromagnetic_field_tensor_native


@pytest.mark.parametrize("beta", [0.05, 0.8, 0.999])
@pytest.mark.parametrize("g", [2.0, 2.00231930436, 5.5856946893])
def test_axial_native_force_and_momentum(beta, g):
    gamma = 1 / np.sqrt(1 - beta**2)
    mass, charge, spin, b, slope = 1.2, -0.7, 0.03, 0.4, 0.2
    u = c * gamma * np.array([1, 0, 0, beta])
    s = spin * gamma * np.array([beta, 0, 0, 1])
    field = electromagnetic_field_tensor_native([0, 0, 0], [0, 0, b])
    gradient = np.zeros((4, 4, 4))
    gradient[3] = electromagnetic_field_tensor_native([0, 0, 0], [0, 0, slope])
    gradient[1] = electromagnetic_field_tensor_native([0, 0, 0], [-slope / 2, 0, 0])
    gradient[2] = electromagnetic_field_tensor_native([0, 0, 0], [0, -slope / 2, 0])
    result = ordinary_response_native(
        four_velocity_mm_ns=u,
        spin_angular_momentum=s,
        field_tensor=field,
        partial_f=gradient,
        charge_native=charge,
        mass_amu=mass,
        g=g,
    )
    mu = g * charge * spin / (2 * mass * c)
    np.testing.assert_allclose(
        result.four_force,
        mu * gamma**2 * slope * np.array([beta, 0, 0, 1]),
        rtol=2e-10,
        atol=1e-16,
    )
    np.testing.assert_allclose(
        result.spin_momentum_offset, -mu * b * u / c**2, rtol=2e-10, atol=1e-18
    )
    np.testing.assert_allclose(result.spin_four_rate, 0, atol=1e-15)


def test_zero_spin_canonical_is_ordinary_offset():
    u = c * np.array([1.25, 0.75, 0, 0])
    a = np.array([0.4, 0.1, -0.2, 0.3])
    field = electromagnetic_field_tensor_native([0.2, 0.1, 0], [0, 0, 0.3])
    p = canonical_momentum_native(
        four_velocity_mm_ns=u,
        rest_spin_angular_momentum=np.zeros(3),
        four_potential=a,
        field_tensor=field,
        charge_native=0.7,
        mass_amu=1.2,
        g=2,
    )
    np.testing.assert_allclose(p, 1.2 * u + 0.7 * a / c, rtol=1e-15)
    recovered = velocity_from_canonical_spatial_native(
        canonical_spatial_momentum=p[1:],
        rest_spin_angular_momentum=np.zeros(3),
        four_potential=a,
        field_tensor=field,
        charge_native=0.7,
        mass_amu=1.2,
        g=2,
    )
    np.testing.assert_allclose(recovered, u, rtol=1e-15, atol=1e-15)


def test_inverse_error_is_spin_squared_not_a_hidden_exact_solve():
    w = np.array([0.4, -0.2, 0.7])
    u = c * np.r_[np.sqrt(1 + w @ w), w]
    a = np.array([0.4, 0.1, -0.2, 0.3]) * c**2
    field = c**2 * electromagnetic_field_tensor_native([0.2, 0.1, 0.05], [0.1, 0, 0.3])
    errors = []
    for scale in (1e-3, 5e-4, 2.5e-4):
        rest = c * scale * np.array([0.3, 0.2, 0.4])
        p = canonical_momentum_native(
            four_velocity_mm_ns=u,
            rest_spin_angular_momentum=rest,
            four_potential=a,
            field_tensor=field,
            charge_native=0.7,
            mass_amu=1.2,
            g=5.5856946893,
        )
        recovered = velocity_from_canonical_spatial_native(
            canonical_spatial_momentum=p[1:],
            rest_spin_angular_momentum=rest,
            four_potential=a,
            field_tensor=field,
            charge_native=0.7,
            mass_amu=1.2,
            g=5.5856946893,
        )
        errors.append(np.linalg.norm(recovered - u))
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 4, rtol=0.003)


@pytest.mark.parametrize("bad", ["velocity", "spin", "field", "mass"])
def test_invalid_input_rejected(bad):
    args = dict(
        four_velocity_mm_ns=np.array([c, 0, 0, 0]),
        spin_angular_momentum=np.array([0, 0, 0, 0.1]),
        field_tensor=np.zeros((4, 4)),
        partial_f=np.zeros((4, 4, 4)),
        charge_native=1,
        mass_amu=1,
        g=2,
    )
    if bad == "velocity":
        args["four_velocity_mm_ns"][0] = -c
    elif bad == "spin":
        args["spin_angular_momentum"][0] = 1
    elif bad == "field":
        args["field_tensor"][0, 0] = 1
    else:
        args["mass_amu"] = -1
    with pytest.raises(ValueError):
        ordinary_response_native(**args)


@pytest.mark.parametrize("g", [0, 2, 5.5856946893])
def test_reaction_changes_canonical_balance_by_known_radiative_term(g):
    from core.spin_self_force_oracle import body_frame_cross

    w = np.array([0.4, -0.2, 0.8])
    gamma = np.sqrt(1 + w @ w)
    u = c * np.r_[gamma, w]
    rest = c * np.array([0.1, 0.2, -0.3]) * 1e-4
    s0 = w @ rest
    spin = np.r_[s0, rest + w * s0 / (1 + gamma)]
    field = c * c * electromagnetic_field_tensor_native([0, 0, 0], [0.1, 0.2, 0.3])
    force3 = c * c * np.array([1e-3, -2e-3, 3e-3])
    extra = np.r_[w @ force3 / gamma, force3]
    args = dict(
        four_velocity_mm_ns=u,
        spin_angular_momentum=spin,
        field_tensor=field,
        partial_f=np.zeros((4, 4, 4)),
        charge_native=1,
        mass_amu=1,
        g=g,
    )
    ordinary = ordinary_response_native(**args)
    reacted = ordinary_response_native(
        **args, leading_charge_reaction_four_force_native=extra
    )
    a0 = field @ (u * np.array([1, -1, -1, -1])) / c
    delta = (
        (u / c)
        * ((spin * np.array([1, -1, -1, -1])) @ body_frame_cross(a0, extra, u / c))
        / c**3
    )
    np.testing.assert_allclose(
        reacted.charge_radiative_balance_correction, delta, rtol=1e-12, atol=1e-15
    )
    observed = (
        reacted.four_force
        - ordinary.four_force
        + reacted.spin_momentum_offset_rate
        - ordinary.spin_momentum_offset_rate
        - extra
    )
    np.testing.assert_allclose(observed, delta, rtol=1e-7, atol=2e-12)
    # Reject the naive work identity; do not 'repair' it by adding delta as force.
    assert abs(delta[0]) > 1e-5
