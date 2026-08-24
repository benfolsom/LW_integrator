"""Cross-sector tests for ordinary charge response to a dipole source."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.dipole_source_interactions import (
    dipole_source_interaction_from_field_native,
    evaluate_retarded_dipole_source_interaction_native,
)
from core.retarded_dipole_fields import (
    evaluate_retarded_dipole_field_gradient_native,
)
from core.retarded_fields import ObserverEvent
from core.rfs import rfs_four_force_native


def _static_dipole_history(moment_native: float) -> list[dict[str, np.ndarray]]:
    result = []
    for time_ns in np.linspace(-0.02, 0.002, 45):
        result.append(
            {
                "t": np.array([time_ns]),
                "x": np.array([0.0]),
                "y": np.array([0.0]),
                "z": np.array([0.0]),
                "bx": np.array([0.0]),
                "by": np.array([0.0]),
                "bz": np.array([0.0]),
                "bdotx": np.array([0.0]),
                "bdoty": np.array([0.0]),
                "bdotz": np.array([0.0]),
                "q": np.array([0.0]),
                "q_source": np.array([0.0]),
                "spin_x": np.array([0.0]),
                "spin_y": np.array([0.0]),
                "spin_z": np.array([1.0]),
                "magnetic_moment_native": np.array([moment_native]),
                "magnetic_dipole_active": np.array([1.0]),
                "_dead_particles": np.array([False]),
            }
        )
    return result


def test_static_dipole_canonical_response_recovers_lorentz_force() -> None:
    radius = 2.0
    moment = 1.4
    charge = -0.8
    beta_x = 0.15
    gamma = 1.0 / np.sqrt(1.0 - beta_x**2)
    velocity = gamma * C_MMNS * np.array((1.0, beta_x, 0.0, 0.0))
    proper_step = 0.03
    interaction = evaluate_retarded_dipole_source_interaction_native(
        _static_dipole_history(moment),
        ObserverEvent(0.0, (radius, 0.0, 0.0)),
        four_velocity_mm_ns=velocity,
        observer_charge_native=charge,
        proper_time_step_ns=proper_step,
        source_identities=("source",),
        observer_source_identity="observer",
        stencil_step_mm=1.0e-3 * radius,
    )

    expected_potential_y = charge * moment / (C_MMNS * radius**2)
    assert interaction.canonical_potential_momentum[2] == pytest.approx(
        expected_potential_y,
        rel=3.0e-6,
    )
    np.testing.assert_allclose(
        interaction.canonical_four_impulse,
        proper_step * interaction.canonical_four_force,
        rtol=1.0e-15,
        atol=1.0e-18,
    )

    convective_potential_force = (
        charge
        / C_MMNS
        * np.einsum(
            "l,la->a",
            velocity,
            interaction.field.partial_a,
        )
    )
    mechanical_force = interaction.canonical_four_force - convective_potential_force
    expected_lorentz = rfs_four_force_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=np.array((0.0, 0.0, 0.0, 1.0)),
        field_tensor=interaction.field.field_tensor,
        partial_f=np.zeros((4, 4, 4)),
        charge_native=charge,
        magnetic_moment_native=0.0,
    )
    np.testing.assert_allclose(
        mechanical_force,
        expected_lorentz,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        interaction.mechanical_four_force,
        expected_lorentz,
        rtol=3.0e-16,
        atol=1.0e-16,
    )
    np.testing.assert_allclose(
        interaction.mechanical_four_impulse,
        proper_step * expected_lorentz,
        rtol=3.0e-16,
        atol=1.0e-16,
    )


def test_neutral_observer_receives_field_but_no_canonical_response() -> None:
    interaction = evaluate_retarded_dipole_source_interaction_native(
        _static_dipole_history(2.0),
        ObserverEvent(0.0, (1.0, 0.0, 0.0)),
        four_velocity_mm_ns=(C_MMNS, 0.0, 0.0, 0.0),
        observer_charge_native=0.0,
        proper_time_step_ns=0.1,
        stencil_step_mm=5.0e-4,
    )

    assert interaction.field.magnetic_field_native[2] == pytest.approx(
        -2.0,
        rel=4.0e-6,
    )
    np.testing.assert_array_equal(interaction.canonical_potential_momentum, 0.0)
    np.testing.assert_array_equal(interaction.canonical_four_force, 0.0)
    np.testing.assert_array_equal(interaction.canonical_four_impulse, 0.0)
    np.testing.assert_array_equal(interaction.mechanical_four_force, 0.0)
    np.testing.assert_array_equal(interaction.mechanical_four_impulse, 0.0)


def test_cached_dipole_field_is_recontracted_with_each_trial_velocity() -> None:
    history = _static_dipole_history(1.1)
    event = ObserverEvent(0.0, (1.5, 0.2, -0.1))
    field = evaluate_retarded_dipole_field_gradient_native(
        history,
        event,
        stencil_step_mm=1.0e-3,
    )
    trial_betas = (
        np.array((0.03, 0.08, -0.02)),
        np.array((-0.04, 0.12, 0.01)),
    )
    interactions = []
    for beta in trial_betas:
        gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
        interactions.append(
            dipole_source_interaction_from_field_native(
                field,
                four_velocity_mm_ns=(gamma * C_MMNS * np.concatenate(((1.0,), beta))),
                observer_charge_native=-0.7,
                proper_time_step_ns=0.015,
            )
        )

    assert interactions[0].field is field
    assert interactions[1].field is field
    np.testing.assert_array_equal(
        interactions[0].canonical_potential_momentum,
        interactions[1].canonical_potential_momentum,
    )
    assert not np.array_equal(
        interactions[0].canonical_four_force,
        interactions[1].canonical_four_force,
    )
    for interaction in interactions:
        np.testing.assert_allclose(
            interaction.canonical_four_impulse,
            0.015 * interaction.canonical_four_force,
            rtol=2.0e-15,
            atol=2.0e-14,
        )
