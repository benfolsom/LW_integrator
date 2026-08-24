"""Exact convention tests for the ordinary electromagnetic potential seam."""

from __future__ import annotations

import numpy as np
import pytest

from core.canonical_momentum import (
    canonical_four_force_from_potential_gradient_native,
    canonical_four_impulse_from_potential_gradient_native,
    canonical_potential_momentum_native,
    mechanical_four_momentum_native,
)
from core.constants import C_MMNS
from core.dipole_fields import static_point_dipole_field_native
from core.rfs import (
    electromagnetic_field_tensor_native,
    rfs_four_force_native,
)

_SIGNS = np.array((1.0, -1.0, -1.0, -1.0))


def _field_tensor_from_partial_a(partial_a: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            result[mu, nu] = (
                _SIGNS[mu] * partial_a[mu, nu] - _SIGNS[nu] * partial_a[nu, mu]
            )
    return result


def test_potential_offset_is_step_independent_and_round_trips() -> None:
    potential = np.array((0.7, -0.2, 0.4, 0.9))
    charge = -1.3
    mechanical = np.array((5.0, -2.0, 3.0, 7.0))

    offset = canonical_potential_momentum_native(
        potential,
        charge_native=charge,
    )
    canonical = mechanical + offset

    np.testing.assert_array_equal(offset, charge * potential / C_MMNS)
    np.testing.assert_array_equal(
        mechanical_four_momentum_native(
            canonical,
            potential,
            charge_native=charge,
        ),
        mechanical,
    )


def test_canonical_impulse_is_linear_in_proper_time() -> None:
    velocity = np.array((1.3 * C_MMNS, 0.2 * C_MMNS, -0.1 * C_MMNS, 0.0))
    partial_a = np.arange(16, dtype=float).reshape(4, 4) / 17.0

    first = canonical_four_impulse_from_potential_gradient_native(
        four_velocity_mm_ns=velocity,
        partial_a=partial_a,
        charge_native=0.8,
        proper_time_step_ns=0.125,
    )
    second = canonical_four_impulse_from_potential_gradient_native(
        four_velocity_mm_ns=velocity,
        partial_a=partial_a,
        charge_native=0.8,
        proper_time_step_ns=0.25,
    )

    np.testing.assert_array_equal(second, 2.0 * first)


def test_canonical_minus_convective_potential_derivative_is_lorentz_force() -> None:
    rng = np.random.default_rng(271828)
    beta = np.array((0.21, -0.13, 0.08))
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = gamma * C_MMNS * np.concatenate(((1.0,), beta))
    partial_a = rng.normal(size=(4, 4))
    charge = -0.73

    canonical_force = canonical_four_force_from_potential_gradient_native(
        four_velocity_mm_ns=velocity,
        partial_a=partial_a,
        charge_native=charge,
    )
    convective_potential_force = (
        charge / C_MMNS * np.einsum("l,la->a", velocity, partial_a)
    )
    mechanical_force = canonical_force - convective_potential_force

    field_tensor = _field_tensor_from_partial_a(partial_a)
    rfs_lorentz_force = rfs_four_force_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=np.array((0.0, 0.0, 0.0, 1.0)),
        field_tensor=field_tensor,
        partial_f=np.zeros((4, 4, 4)),
        charge_native=charge,
        magnetic_moment_native=0.0,
    )
    np.testing.assert_allclose(
        mechanical_force,
        rfs_lorentz_force,
        rtol=2.0e-15,
        atol=2.0e-14,
    )


def test_static_dipole_canonical_and_convective_terms_recover_magnetic_force() -> None:
    radius = 2.5
    moment = 1.7
    charge = 0.6
    beta_x = 0.12
    gamma = 1.0 / np.sqrt(1.0 - beta_x**2)
    velocity = gamma * C_MMNS * np.array((1.0, beta_x, 0.0, 0.0))
    dipole = static_point_dipole_field_native(
        separation_vector_mm=(radius, 0.0, 0.0),
        magnetic_moment_native=moment,
        rest_spin_direction=(0.0, 0.0, 1.0),
    )
    partial_a = np.zeros((4, 4), dtype=float)
    # StaticDipoleFieldResult stores [potential component, coordinate], while
    # partial_a stores [coordinate, potential component].
    partial_a[1:, 1:] = dipole.vector_potential_gradient_native.T

    canonical_force = canonical_four_force_from_potential_gradient_native(
        four_velocity_mm_ns=velocity,
        partial_a=partial_a,
        charge_native=charge,
    )
    convective_potential_force = (
        charge / C_MMNS * np.einsum("l,la->a", velocity, partial_a)
    )
    mechanical_force = canonical_force - convective_potential_force
    expected_y = charge * gamma * beta_x * moment / radius**3

    assert canonical_force[2] == pytest.approx(-expected_y)
    assert convective_potential_force[2] == pytest.approx(-2.0 * expected_y)
    assert mechanical_force[2] == pytest.approx(expected_y)

    field_tensor = electromagnetic_field_tensor_native(
        dipole.electric_field_native,
        dipole.magnetic_field_native,
    )
    expected_lorentz = rfs_four_force_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=np.array((0.0, 0.0, 0.0, 1.0)),
        field_tensor=field_tensor,
        partial_f=np.zeros((4, 4, 4)),
        charge_native=charge,
        magnetic_moment_native=0.0,
    )
    np.testing.assert_allclose(mechanical_force, expected_lorentz, atol=1.0e-16)


@pytest.mark.parametrize(
    "callable_name, kwargs, message",
    [
        (
            "offset",
            {"four_potential": (0.0, 1.0, 2.0), "charge_native": 1.0},
            "shape",
        ),
        (
            "force",
            {
                "four_velocity_mm_ns": (C_MMNS, 0.0, 0.0, 0.0),
                "partial_a": np.zeros((3, 3)),
                "charge_native": 1.0,
            },
            "shape",
        ),
        (
            "impulse",
            {
                "four_velocity_mm_ns": (C_MMNS, 0.0, 0.0, 0.0),
                "partial_a": np.zeros((4, 4)),
                "charge_native": 1.0,
                "proper_time_step_ns": np.nan,
            },
            "proper_time_step_ns",
        ),
    ],
)
def test_invalid_inputs_fail_explicitly(
    callable_name: str,
    kwargs: dict[str, object],
    message: str,
) -> None:
    if callable_name == "offset":
        function = canonical_potential_momentum_native
    elif callable_name == "force":
        function = canonical_four_force_from_potential_gradient_native
    else:
        function = canonical_four_impulse_from_potential_gradient_native
    with pytest.raises(ValueError, match=message):
        function(**kwargs)
