"""Exact retarded charge-potential and canonical-response tests."""

from __future__ import annotations

import numpy as np
import pytest

from core.charge_source_interactions import (
    charge_source_interaction_from_field_native,
    evaluate_retarded_charge_source_interaction_native,
)
from core.constants import C_MMNS
from core.retarded_fields import (
    ObserverEvent,
    evaluate_retarded_charge_field_gradient_native,
    evaluate_retarded_charge_field_native,
)
from core.rfs import rfs_four_force_native

_SIGNS = np.array((1.0, -1.0, -1.0, -1.0))


def _uniform_charge_history(
    *,
    knot_count: int,
    charge_native: float,
    beta: np.ndarray,
    start_time_ns: float = -0.03,
    end_time_ns: float = 0.003,
) -> list[dict[str, np.ndarray]]:
    times = np.linspace(start_time_ns, end_time_ns, knot_count)
    positions = C_MMNS * times[:, np.newaxis] * beta[np.newaxis, :]
    return [
        {
            "t": np.array([time_ns]),
            "x": np.array([position[0]]),
            "y": np.array([position[1]]),
            "z": np.array([position[2]]),
            "bx": np.array([beta[0]]),
            "by": np.array([beta[1]]),
            "bz": np.array([beta[2]]),
            "bdotx": np.array([0.0]),
            "bdoty": np.array([0.0]),
            "bdotz": np.array([0.0]),
            "q": np.array([charge_native]),
            "q_source": np.array([charge_native]),
            "_dead_particles": np.array([False]),
        }
        for time_ns, position in zip(times, positions)
    ]


def _field_tensor_from_partial_a(partial_a: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            result[mu, nu] = (
                _SIGNS[mu] * partial_a[mu, nu] - _SIGNS[nu] * partial_a[nu, mu]
            )
    return result


def test_static_retarded_charge_potential_is_gaussian_coulomb_potential() -> None:
    source_charge = 1.7
    radius = 2.3
    history = _uniform_charge_history(
        knot_count=4,
        charge_native=source_charge,
        beta=np.zeros(3),
    )

    result = evaluate_retarded_charge_field_native(
        history,
        ObserverEvent(0.0, (radius, 0.0, 0.0)),
    )

    np.testing.assert_allclose(
        result.four_potential,
        (source_charge / radius, 0.0, 0.0, 0.0),
        rtol=2.0e-15,
        atol=1.0e-15,
    )


def test_uniform_motion_potential_uses_retarded_kappa_and_source_beta() -> None:
    source_charge = -0.9
    beta = np.array((0.17, -0.08, 0.03))
    observer_position = np.array((1.1, 0.7, -0.2))
    history = _uniform_charge_history(
        knot_count=8,
        charge_native=source_charge,
        beta=beta,
    )

    result = evaluate_retarded_charge_field_native(
        history,
        ObserverEvent(0.0, tuple(observer_position)),
    )
    retarded_time = result.retarded_time_ns[0]
    separation_vector = observer_position - C_MMNS * retarded_time * beta
    separation = float(np.linalg.norm(separation_vector))
    direction = separation_vector / separation
    scalar_potential = source_charge / ((1.0 - float(direction @ beta)) * separation)

    np.testing.assert_allclose(
        result.four_potential,
        scalar_potential * np.concatenate(((1.0,), beta)),
        rtol=3.0e-14,
        atol=1.0e-14,
    )


def test_centered_potential_gradient_recovers_retarded_field_tensor() -> None:
    source_charge = 1.3
    beta = np.array((0.19, -0.06, 0.04))
    history = _uniform_charge_history(
        knot_count=8,
        charge_native=source_charge,
        beta=beta,
    )
    result = evaluate_retarded_charge_field_gradient_native(
        history,
        ObserverEvent(0.0, (1.0, 0.8, -0.3)),
        relative_step=1.0e-5,
    )

    reconstructed = _field_tensor_from_partial_a(result.partial_a)
    np.testing.assert_allclose(
        reconstructed,
        result.field.field_tensor,
        rtol=3.0e-9,
        atol=2.0e-9,
    )


def test_charge_source_canonical_response_recovers_lorentz_force() -> None:
    source_charge = 1.4
    observer_charge = -0.7
    source_beta = np.array((0.11, -0.04, 0.02))
    observer_beta = np.array((-0.08, 0.13, 0.03))
    gamma = 1.0 / np.sqrt(1.0 - float(observer_beta @ observer_beta))
    velocity = gamma * C_MMNS * np.concatenate(((1.0,), observer_beta))
    step = 0.025
    interaction = evaluate_retarded_charge_source_interaction_native(
        _uniform_charge_history(
            knot_count=8,
            charge_native=source_charge,
            beta=source_beta,
        ),
        ObserverEvent(0.0, (1.2, 0.5, -0.4)),
        four_velocity_mm_ns=velocity,
        observer_charge_native=observer_charge,
        proper_time_step_ns=step,
        relative_step=1.0e-5,
    )

    np.testing.assert_allclose(
        interaction.canonical_four_impulse,
        step * interaction.canonical_four_force,
        rtol=2.0e-15,
        atol=2.0e-14,
    )
    convective_potential_force = (
        observer_charge
        / C_MMNS
        * np.einsum("l,la->a", velocity, interaction.field.partial_a)
    )
    mechanical_force = interaction.canonical_four_force - convective_potential_force
    expected_lorentz = rfs_four_force_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=np.array((0.0, 0.0, 0.0, 1.0)),
        field_tensor=interaction.field.field.field_tensor,
        partial_f=np.zeros((4, 4, 4)),
        charge_native=observer_charge,
        magnetic_moment_native=0.0,
    )
    np.testing.assert_allclose(
        mechanical_force,
        expected_lorentz,
        rtol=3.0e-9,
        atol=3.0e-9,
    )
    np.testing.assert_allclose(
        interaction.mechanical_four_force,
        expected_lorentz,
        rtol=3.0e-16,
        atol=1.0e-16,
    )
    np.testing.assert_allclose(
        interaction.mechanical_four_impulse,
        step * expected_lorentz,
        rtol=3.0e-16,
        atol=1.0e-16,
    )


def test_neutral_observer_receives_charge_field_but_no_canonical_response() -> None:
    interaction = evaluate_retarded_charge_source_interaction_native(
        _uniform_charge_history(
            knot_count=4,
            charge_native=2.0,
            beta=np.zeros(3),
        ),
        ObserverEvent(0.0, (1.0, 0.0, 0.0)),
        four_velocity_mm_ns=(C_MMNS, 0.0, 0.0, 0.0),
        observer_charge_native=0.0,
        proper_time_step_ns=0.1,
    )

    assert interaction.field.field.electric_field_native[0] == pytest.approx(2.0)
    np.testing.assert_array_equal(interaction.canonical_potential_momentum, 0.0)
    np.testing.assert_array_equal(interaction.canonical_four_force, 0.0)
    np.testing.assert_array_equal(interaction.canonical_four_impulse, 0.0)
    np.testing.assert_array_equal(interaction.mechanical_four_force, 0.0)
    np.testing.assert_array_equal(interaction.mechanical_four_impulse, 0.0)


def test_cached_charge_field_is_recontracted_with_each_trial_velocity() -> None:
    history = _uniform_charge_history(
        knot_count=8,
        charge_native=1.2,
        beta=np.array((0.13, -0.02, 0.04)),
    )
    event = ObserverEvent(0.0, (1.0, 0.4, -0.3))
    field = evaluate_retarded_charge_field_gradient_native(
        history,
        event,
        relative_step=1.0e-5,
    )
    trial_betas = (
        np.array((0.02, 0.07, -0.01)),
        np.array((-0.05, 0.11, 0.03)),
    )
    interactions = []
    for beta in trial_betas:
        gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
        interactions.append(
            charge_source_interaction_from_field_native(
                field,
                four_velocity_mm_ns=(gamma * C_MMNS * np.concatenate(((1.0,), beta))),
                observer_charge_native=-0.6,
                proper_time_step_ns=0.02,
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
            0.02 * interaction.canonical_four_force,
            rtol=2.0e-15,
            atol=2.0e-14,
        )


def test_sparse_uniform_inertial_history_is_invariant_for_two_four_eight_knots() -> (
    None
):
    source_charge = -1.1
    observer_charge = 0.6
    source_beta = np.array((0.16, -0.07, 0.05))
    observer_beta = np.array((-0.04, 0.09, -0.02))
    gamma = 1.0 / np.sqrt(1.0 - float(observer_beta @ observer_beta))
    velocity = gamma * C_MMNS * np.concatenate(((1.0,), observer_beta))
    interactions = [
        evaluate_retarded_charge_source_interaction_native(
            _uniform_charge_history(
                knot_count=knot_count,
                charge_native=source_charge,
                beta=source_beta,
            ),
            ObserverEvent(0.0, (0.9, 0.6, -0.25)),
            four_velocity_mm_ns=velocity,
            observer_charge_native=observer_charge,
            proper_time_step_ns=0.02,
            relative_step=2.0e-5,
        )
        for knot_count in (2, 4, 8)
    ]

    reference = interactions[-1]
    for interaction in interactions[:-1]:
        np.testing.assert_allclose(
            interaction.field.field.four_potential,
            reference.field.field.four_potential,
            rtol=3.0e-13,
            atol=3.0e-13,
        )
        np.testing.assert_allclose(
            interaction.field.field.field_tensor,
            reference.field.field.field_tensor,
            rtol=3.0e-13,
            atol=3.0e-13,
        )
        np.testing.assert_allclose(
            interaction.field.partial_a,
            reference.field.partial_a,
            rtol=2.0e-10,
            atol=2.0e-10,
        )
        np.testing.assert_allclose(
            interaction.field.partial_f,
            reference.field.partial_f,
            rtol=2.0e-10,
            atol=2.0e-10,
        )
        np.testing.assert_allclose(
            interaction.canonical_four_force,
            reference.canonical_four_force,
            rtol=2.0e-10,
            atol=2.0e-10,
        )
