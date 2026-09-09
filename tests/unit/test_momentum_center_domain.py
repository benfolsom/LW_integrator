"""Reject unusable velocities without confusing proper speed with lab speed."""

import numpy as np
import pytest

from core import momentum_center as model


@pytest.mark.parametrize("entry", [np.nan, np.inf])
def test_nonfinite_gradient_is_rejected(entry):
    with pytest.raises(model.VelocityDomainError, match="Finite"):
        model.velocity(
            np.array([1.0, 0, 0, 0]),
            np.zeros((4, 4)),
            np.zeros((4, 4)),
            np.full((4, 4, 4), entry),
            model.Particle(),
        )


@pytest.mark.parametrize("spatial, expected", [(1.0, 1.0), (2.0, 4.0)])
def test_lightlike_and_spacelike_directions_are_rejected(
    monkeypatch, spatial, expected
):
    def direction(*args):
        return (
            np.array([1.0, spatial, 0, 0]),
            np.zeros(4),
            np.zeros((4, 4)),
            1.0,
            1.0,
            np.zeros(4),
        )

    monkeypatch.setattr(model, "velocity_direction", direction)
    with pytest.raises(model.VelocityDomainError) as caught:
        model.velocity(
            np.ones(4),
            np.zeros((4, 4)),
            np.zeros((4, 4)),
            np.zeros((4, 4, 4)),
            model.Particle(),
        )
    assert caught.value.beta_squared == expected
    assert caught.value.direction_norm_squared <= 0


def test_large_spatial_proper_velocity_is_allowed():
    momentum = np.array([2.0, np.sqrt(3), 0, 0])
    u = model.velocity(
        momentum,
        np.zeros((4, 4)),
        np.zeros((4, 4)),
        np.zeros((4, 4, 4)),
        model.Particle(),
    )[0]
    assert np.linalg.norm(u[1:]) > 1
    assert np.linalg.norm(u[1:]) / u[0] < 1


def test_nonfinite_supplied_potential_derivative_is_rejected():
    state = np.r_[np.zeros(4), [1.0, 0, 0, 0], np.zeros(6)]
    with pytest.raises(ValueError, match="Finite potential response"):
        model.evaluate(
            state,
            model.Particle(),
            lambda event: (
                np.zeros(4),
                np.full((4, 4), np.nan),
                np.zeros((4, 4)),
                np.zeros((4, 4, 4)),
            ),
        )


@pytest.mark.parametrize("beta", [0.9, 0.99])
def test_momentum_frame_monitor_does_not_confuse_fast_coasting_with_blowup(beta):
    gamma = 1 / np.sqrt(1 - beta**2)

    def provider(event):
        return (np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4, 4)))

    particle = model.Particle()
    state = model.initial_state(
        np.zeros(4),
        gamma * np.array([1.0, 0, beta, 0]),
        [0.1, 0.2, 0.3],
        particle,
        provider,
    )
    _, diagnostic = model.evaluate(state, particle, provider)
    assert abs(diagnostic["momentum_rest_frame_beta_squared"]) < 2e-13
    assert diagnostic["action_interaction_relative"] == 0
