"""Compensated derivatives stay on one side of declared startup boundaries."""

import numpy as np
import pytest

from core import momentum_center as m
from core.full_dipole_reaction import (
    ordinary_source_coefficients,
    inherit_reaction_domain,
)
from core.coupled_dipole_reaction import BoostedPotential
from tests.unit.test_full_dipole_reaction import magnetic


@pytest.mark.parametrize("time, side", [(0.1, "forward"), (0.9, "backward")])
def test_one_sided_compensated_derivatives_match_uniform_field(time, side):
    particle = m.Particle(charge=0.2, g=2)
    state = m.initial_state(
        [time, 0, 0, 0], [1.25, 0.75, 0, 0], [0.2, 0.1, 0.3], particle, magnetic
    )
    events = []

    def provider(event):
        events.append(event[0])
        return magnetic(event)

    provider.reaction_fit_domain = lambda event, u: (
        0 if 0 <= event[0] < 1 else 1,
        event[0] / u[0],
        (1 - event[0]) / u[0],
    )
    u, d, data = ordinary_source_coefficients(
        state, particle, provider, window=0.5, method="boundary_aware"
    )
    _, center = m.evaluate(state, particle, magnetic)
    generator = (
        particle.charge
        / center["kinetic_mass"]
        * magnetic(state[:4])[2]
        * m.METRIC[None, :]
    )
    ref_u, ref_d = np.zeros((5, 4)), np.zeros((5, 4, 4))
    ref_u[0], ref_d[0] = center["proper_velocity"], center["proper_dipole"]
    for n in range(1, 5):
        ref_u[n] = generator @ ref_u[n - 1] / n
        ref_d[n] = (generator @ ref_d[n - 1] + ref_d[n - 1] @ generator.T) / n
    assert data["sampling_side"] == side
    assert min(events) >= 0 and max(events) < 1
    for actual, expected in ((u, ref_u), (d, ref_d)):
        for order in range(1, 5):
            assert (
                np.linalg.norm(actual[order] - expected[order])
                / np.linalg.norm(expected[order])
                < 1e-5
            )


def test_trial_self_field_wrapper_retains_domain_information():
    def base(event):
        return magnetic(event)

    base.reaction_fit_domain = lambda event, u: ("region", 1.0, 2.0)
    wrapped = inherit_reaction_domain(base, lambda event: magnetic(event))
    assert wrapped.reaction_fit_domain is base.reaction_fit_domain
    boost = np.eye(4)
    boosted = BoostedPotential(wrapped, boost)
    assert boosted.reaction_fit_domain(np.zeros(4), np.array([1, 0, 0, 0])) == (
        "region",
        1.0,
        2.0,
    )


def test_insufficient_smooth_interval_is_reported_not_silently_shrunk():
    particle = m.Particle(charge=0.2)
    state = m.initial_state(
        np.zeros(4), [1, 0, 0, 0], [0.2, 0.1, 0.3], particle, magnetic
    )

    def provider(event):
        return magnetic(event)

    provider.reaction_fit_domain = lambda event, u: (0, 0.01, 0.01)
    with pytest.raises(ValueError, match="does not fit"):
        ordinary_source_coefficients(
            state, particle, provider, window=0.1, method="boundary_aware"
        )


@pytest.mark.parametrize("beta", [0.0, 0.8, 0.99])
def test_kinetic_predictor_is_same_equations_in_different_coordinates(beta):
    particle = m.Particle(charge=0.2, g=2)
    gamma = 1 / np.sqrt(1 - beta * beta)

    def provider(event):
        a, da, f, df = magnetic(event)
        return a + np.array([0.3, 0.02, 0.01, 0]), da, f, df

    state = m.initial_state(
        [0.1, 0.2, 0, 0],
        [gamma, gamma * beta, 0, 0],
        [0.2, 0.1, 0.3],
        particle,
        provider,
    )
    stored_rate, stored = m.evaluate(state, particle, provider)
    physical_state = state.copy()
    physical_state[4:8] = stored["kinetic_momentum"]
    kinetic_rate, kinetic = m.evaluate_kinetic(physical_state, particle, provider)
    np.testing.assert_array_equal(kinetic_rate[4:8], stored["momentum_rate"])
    np.testing.assert_array_equal(kinetic_rate[:4], stored_rate[:4])
    np.testing.assert_array_equal(kinetic_rate[8:], stored_rate[8:])
    for key in stored:
        np.testing.assert_array_equal(kinetic[key], stored[key])
