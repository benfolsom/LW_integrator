"""Independent circular-motion and Thomas--BMT spin derivatives.

The spin/orbit frequency ratio is 1+(g/2-1)*gamma for transverse uniform B.
See Rafelski et al., arXiv:1712.01825, homogeneous-field limit, and the standard
Thomas--BMT equation. This reference differentiates elementary rotations and
the Lorentz boost, not the production RFS derivative expressions.
"""

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.external_potential_derivatives import (
    uniform_external_potential_derivatives_native,
)
from core.spin_self_force_oracle import (
    evaluate_jakobsen_intrinsic_spin_radiation_balance_native,
)
from core.spin_self_force_reduction_oracle import (
    evaluate_potential_directional_intrinsic_spin_reduction_native,
)
from core.types import ExternalFieldConfig


def circular_bmt_reference(beta, phase, omega, g_factor):
    gamma = 1 / np.sqrt(1 - beta**2)
    spin_omega = (1 + (g_factor / 2 - 1) * gamma) * omega
    spin_phase = phase * spin_omega / omega
    initial_spin = np.array([1, 2, 3.0]) / np.sqrt(14)
    rotation = np.array(
        [
            [np.cos(spin_phase), -np.sin(spin_phase), 0],
            [np.sin(spin_phase), np.cos(spin_phase), 0],
            [0, 0, 1],
        ]
    )
    spin = rotation @ initial_spin
    cross_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    spin1 = spin_omega * (cross_z @ spin)
    spin2 = spin_omega**2 * (cross_z @ cross_z @ spin)
    b = beta * np.array([np.cos(phase), np.sin(phase), 0])
    b1 = omega * (cross_z @ b)
    b2 = omega**2 * (cross_z @ cross_z @ b)
    b3 = omega**3 * (cross_z @ cross_z @ cross_z @ b)
    d = b @ spin
    d1 = b1 @ spin + b @ spin1
    d2 = b2 @ spin + 2 * b1 @ spin1 + b @ spin2
    k = gamma**2 / (gamma + 1)
    return dict(
        velocity=gamma * c * np.r_[1, b],
        acceleration=gamma * c * np.r_[0, b1],
        jerk=gamma * c * np.r_[0, b2],
        snap=gamma * c * np.r_[0, b3],
        spin=np.r_[gamma * d, spin + k * b * d],
        spin1=np.r_[gamma * d1, spin1 + k * (b1 * d + b * d1)],
        spin2=np.r_[gamma * d2, spin2 + k * (b2 * d + 2 * b1 * d1 + b * d2)],
    )


@pytest.mark.parametrize("beta", [0.02, 0.8, 0.99, 0.9999])
@pytest.mark.parametrize("phase", [0.0, 0.07, 0.2])
def test_analytical_spin_reduction_matches_rotations_with_anomalous_g(beta, phase):
    charge, mass, invariant_spin, g = -0.7, 1.3, 0.8, 2.00231930436
    field = 400.0
    omega = -charge * field / (mass * c)
    reference = circular_bmt_reference(beta, phase, omega, g)
    potential = uniform_external_potential_derivatives_native(
        ExternalFieldConfig(magnetic_field_native=(0, 0, field)), position_mm=(0, 0, 0)
    )
    actual = evaluate_potential_directional_intrinsic_spin_reduction_native(
        four_velocity_mm_ns=reference["velocity"],
        normalized_spin_four_vector=reference["spin"],
        partial_a=potential.partial_a,
        partial2_a=potential.partial2_a,
        partial3_a_along_velocity=potential.partial3_a_along_velocity,
        partial3_a_along_acceleration=potential.partial3_a_along_acceleration,
        partial4_a_along_velocity_twice=potential.partial4_a_along_velocity_twice,
        charge_native=charge,
        mass_amu=mass,
        invariant_spin_native=invariant_spin,
        g_factor=g,
    )
    for name, key in (
        ("four_acceleration", "acceleration"),
        ("four_jerk", "jerk"),
        ("four_snap", "snap"),
        ("normalized_spin_first_derivative", "spin1"),
        ("normalized_spin_second_derivative", "spin2"),
    ):
        np.testing.assert_allclose(
            getattr(actual.leading_dynamics, name),
            reference[key],
            rtol=2e-12,
            atol=2e-12 * np.max(abs(reference[key])),
        )
    expected = evaluate_jakobsen_intrinsic_spin_radiation_balance_native(
        charge_native=charge,
        mass_amu=mass,
        g_factor=g,
        four_velocity_mm_ns=reference["velocity"],
        four_acceleration_mm_ns2=reference["acceleration"],
        four_jerk_mm_ns3=reference["jerk"],
        four_snap_mm_ns4=reference["snap"],
        spin_four_vector_native=invariant_spin * reference["spin"],
        spin_four_derivative_native=invariant_spin * reference["spin1"],
        spin_four_second_derivative_native=invariant_spin * reference["spin2"],
    )
    expected_force = expected.self_force.linear_spin_self_force_native
    np.testing.assert_allclose(
        actual.radiation_balance.self_force.linear_spin_self_force_native,
        expected_force,
        rtol=3e-11,
        atol=3e-11 * np.max(abs(expected_force)),
    )
