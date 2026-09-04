from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.translating_magnetic_shell_kinematics import (
    build_counterrotating_shell_surface_state_native,
)


def _state(
    *, beta_x: float = 0.0, acceleration: float = 0.0, rotation_sign: float = 1.0
):
    return build_counterrotating_shell_surface_state_native(
        center_time_ns=0.7,
        center_position_mm=(1.0, -2.0, 0.5),
        center_beta_x=beta_x,
        center_proper_acceleration_mm_ns2=acceleration,
        shell_radii_mm=(0.009, 0.011),
        shell_charges_native=(2.0 * ELEMENTARY_CHARGE, -2.0 * ELEMENTARY_CHARGE),
        shell_angular_velocities_per_ns=(
            rotation_sign * 0.4,
            -rotation_sign * 0.4,
        ),
        rotation_axis_rest=(1.0, 2.0, -1.0),
        polar_order=8,
        azimuthal_order=16,
        shell_rotation_phases_rad=(0.3, -0.2),
    )


def test_counterrotating_shells_close_charge_dipole_and_moment() -> None:
    state = _state()

    assert state.net_charge_native == pytest.approx(0.0, abs=1.0e-30)
    electric_dipole_scale = float(
        np.sum(
            np.abs(state.charge_native) * np.linalg.norm(state.rest_position_mm, axis=1)
        )
    )
    np.testing.assert_allclose(
        state.rest_electric_dipole_native_mm,
        0.0,
        rtol=0.0,
        atol=5.0e-16 * electric_dipole_scale,
    )
    np.testing.assert_allclose(
        state.rest_magnetic_moment_native,
        state.expected_rest_magnetic_moment_native,
        rtol=3.0e-15,
        atol=0.0,
    )
    assert state.maximum_internal_beta < 2.0e-5
    assert not state.position_mm.flags.writeable


def test_boosted_surface_events_share_one_fermi_rest_slice() -> None:
    beta_x = 0.63
    state = _state(beta_x=beta_x)
    gamma = 1.0 / np.sqrt(1.0 - beta_x**2)
    center = np.array((1.0, -2.0, 0.5))
    spacetime_displacement = np.column_stack(
        (
            C_MMNS * (state.event_time_ns - state.center_time_ns),
            state.position_mm - center,
        )
    )
    center_four_velocity = np.array((gamma * C_MMNS, gamma * beta_x * C_MMNS, 0.0, 0.0))
    orthogonality = (
        spacetime_displacement[:, 0] * center_four_velocity[0]
        - spacetime_displacement[:, 1:] @ center_four_velocity[1:]
    )

    np.testing.assert_allclose(
        orthogonality,
        0.0,
        rtol=0.0,
        atol=3.0e-12 * C_MMNS * np.max(np.linalg.norm(state.rest_position_mm, axis=1)),
    )
    velocity_norm = state.four_velocity_mm_ns[:, 0] ** 2 - np.sum(
        state.four_velocity_mm_ns[:, 1:] ** 2, axis=1
    )
    np.testing.assert_allclose(
        velocity_norm,
        C_MMNS**2,
        rtol=3.0e-15,
        atol=0.0,
    )
    internal_beta = state.rest_velocity_mm_ns / C_MMNS
    internal_gamma = 1.0 / np.sqrt(1.0 - np.sum(internal_beta**2, axis=1))
    expected_four_velocity = np.empty_like(state.four_velocity_mm_ns)
    expected_four_velocity[:, 0] = (
        gamma * internal_gamma * C_MMNS * (1.0 + beta_x * internal_beta[:, 0])
    )
    expected_four_velocity[:, 1] = (
        gamma * internal_gamma * C_MMNS * (internal_beta[:, 0] + beta_x)
    )
    expected_four_velocity[:, 2:] = (
        internal_gamma[:, None] * state.rest_velocity_mm_ns[:, 1:]
    )
    np.testing.assert_allclose(
        state.four_velocity_mm_ns,
        expected_four_velocity,
        rtol=3.0e-15,
        atol=2.0e-14,
    )
    assert np.max(np.linalg.norm(state.beta, axis=1)) < 1.0


def test_acceleration_reports_born_rigidity_control_parameter() -> None:
    acceleration = 4.0e4
    state = _state(acceleration=acceleration)

    assert state.maximum_born_rigidity_parameter == pytest.approx(
        acceleration * 0.011 / C_MMNS**2
    )
    np.testing.assert_allclose(
        state.born_lapse,
        1.0 + acceleration * state.rest_position_mm[:, 0] / C_MMNS**2,
    )


def test_reversing_both_rotations_reverses_moment_only() -> None:
    forward = _state(rotation_sign=1.0)
    reverse = _state(rotation_sign=-1.0)

    np.testing.assert_array_equal(reverse.rest_position_mm, forward.rest_position_mm)
    np.testing.assert_allclose(
        reverse.rest_velocity_mm_ns,
        -forward.rest_velocity_mm_ns,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reverse.rest_magnetic_moment_native,
        -forward.rest_magnetic_moment_native,
        rtol=0.0,
        atol=0.0,
    )


def test_rejects_coincident_shell_radii_and_superluminal_rotation() -> None:
    common = {
        "center_time_ns": 0.0,
        "center_position_mm": (0.0, 0.0, 0.0),
        "center_beta_x": 0.0,
        "center_proper_acceleration_mm_ns2": 0.0,
        "shell_charges_native": (ELEMENTARY_CHARGE, -ELEMENTARY_CHARGE),
        "rotation_axis_rest": (0.0, 0.0, 1.0),
        "polar_order": 4,
        "azimuthal_order": 8,
    }
    with pytest.raises(ValueError, match="positive and distinct"):
        build_counterrotating_shell_surface_state_native(
            shell_radii_mm=(0.01, 0.01),
            shell_angular_velocities_per_ns=(1.0, -1.0),
            **common,
        )
    with pytest.raises(ValueError, match="light speed"):
        build_counterrotating_shell_surface_state_native(
            shell_radii_mm=(1.0, 1.1),
            shell_angular_velocities_per_ns=(C_MMNS, -C_MMNS),
            **common,
        )
    with pytest.raises(ValueError, match="neutral"):
        build_counterrotating_shell_surface_state_native(
            shell_radii_mm=(1.0, 1.1),
            shell_charges_native=(ELEMENTARY_CHARGE, -0.5 * ELEMENTARY_CHARGE),
            shell_angular_velocities_per_ns=(1.0, -1.0),
            **{
                key: value
                for key, value in common.items()
                if key != "shell_charges_native"
            },
        )
    with pytest.raises(ValueError, match="angular velocities"):
        build_counterrotating_shell_surface_state_native(
            shell_radii_mm=(1.0, 1.1),
            shell_angular_velocities_per_ns=(1.0, 1.0),
            **common,
        )
