"""Native conversion, reciprocal advancement, and complete restart checks."""

import copy
import json

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.full_dipole_history import FullDipoleHistory
from core.full_dipole_response import response
from core import momentum_center as model
from core.momentum_center_pair import (
    FullDipoleProvider,
    MomentumCenterParticle,
    advance_pair,
    dynamics_native,
    initial_state_native,
    initialize_pair,
)


def zero_provider(time, position):
    return np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4, 4))


def history(position, dipole=None, beta=None):
    time = np.linspace(-2, 0, 201) / c
    beta = np.zeros(3) if beta is None else np.asarray(beta)
    tensor = np.zeros((4, 4)) if dipole is None else dipole
    return FullDipoleHistory(
        time,
        np.asarray(position) + c * time[:, None] * beta,
        np.broadcast_to(c * beta, (len(time), 3)),
        np.broadcast_to(tensor, (len(time), 4, 4)),
        c,
        position_tolerance=1e-9,
    ).completed()


@pytest.mark.parametrize("beta", [[0, 0, 0], [0.65, -0.2, 0.1]])
def test_native_free_motion(beta):
    particle = MomentumCenterParticle(0.03 * c, 1, 2)
    gamma = 1 / np.sqrt(1 - np.dot(beta, beta))
    direction = gamma * np.r_[1, beta]
    state = initial_state_native(
        [0, 0, 0, 0], direction, c * np.array([0.2, 0.1, 0.3]), particle, zero_provider
    )
    rate, diagnostic = dynamics_native(state, particle, zero_provider)
    np.testing.assert_allclose(rate[:4], np.r_[1, c * np.array(beta)], atol=1e-13)
    np.testing.assert_array_equal(rate[4:], 0)
    np.testing.assert_allclose(diagnostic["kinetic_momentum_native"], c * direction)
    assert abs(diagnostic["length_time"]["mass_constraint"]) < 1e-14


def test_native_source_matches_length_time_full_response():
    tensor = np.zeros((4, 4))
    tensor[0, 1], tensor[1, 0] = 0.02 * c, -0.02 * c
    tensor[1, 2], tensor[2, 1] = 0.03 * c, -0.03 * c
    source = history([0, 0, 0], tensor, [0.2, 0.1, 0])
    provider = FullDipoleProvider(source, 0.03 * c)
    observer = np.array([1, 0.2, 0.3])
    actual = provider(0, observer)
    # Locate the reference root independently, then use the same coefficient
    # data in length-time units. Tests all powers of c in the native adapter.
    segment = next(
        s
        for s in source.segments
        if -c * s.start >= np.linalg.norm(observer - s.sample(s.start)[0])
        and -c * s.end <= np.linalg.norm(observer - s.sample(s.end)[0])
    )
    expected = response(
        np.r_[0, observer],
        c * segment.start,
        c * segment.duration,
        segment.position,
        segment.dipole / c,
        charge=0.03,
        allow_boundary=True,
    )
    for value, key in zip(
        actual, ("four_potential", "partial_a", "field_tensor", "partial_f")
    ):
        np.testing.assert_allclose(value, c * expected[key], rtol=2e-13, atol=1e-14)


def test_static_native_charge_normalization():
    q = 0.02
    actual = FullDipoleProvider(history([0, 0, 0]), q)(0, [1, 0, 0])
    np.testing.assert_allclose(actual[0], [q, 0, 0, 0], atol=1e-16)
    assert actual[1][1, 0] == pytest.approx(-q)
    assert actual[2][0, 1] == pytest.approx(-q)


@pytest.mark.parametrize("beta", [[0, 0, 0], [0.65, -0.2, 0.1]])
def test_full_tensor_matches_existing_magnetic_provider(beta):
    from scipy.optimize import brentq
    from core.dipole_hertz_jet import polynomial_dipole_hertz_response_jet_native

    beta = np.asarray(beta)
    u = np.r_[1, beta] / np.sqrt(1 - beta @ beta)
    rest = np.array([0.2, -0.1, 0.3])
    s0 = u[1:] @ rest
    spin = np.r_[s0, rest + u[1:] * s0 / (1 + u[0])]
    moment = 0.04
    source = history([0, 0, 0], moment * model.spin_tensor(u, spin), beta)
    observer = np.array([0.6, 0.1, 0.2])
    actual = FullDipoleProvider(source, 0)(0, observer)
    root = brentq(
        lambda t: -c * t - np.linalg.norm(observer - c * t * beta),
        -2 / c,
        0,
        xtol=1e-16,
    )
    segment = source.segment_at(root)
    expected = polynomial_dipole_hertz_response_jet_native(
        observer_time_ns=0,
        observer_position_mm=observer,
        magnetic_moment_native=moment,
        segment_start_time_ns=segment.start,
        segment_duration_ns=segment.duration,
        position_coefficients_mm=segment.position,
        rest_spin_coefficients=rest[None, :],
        preserved_rest_spin_magnitude=None,
        retarded_time_ns=root,
    )
    for value, key in zip(
        actual, ("four_potential", "partial_a", "field_tensor", "partial_f")
    ):
        np.testing.assert_allclose(
            value, getattr(expected, key), rtol=2e-10, atol=1e-12
        )


def make_pair(charge=0.03):
    particles = [MomentumCenterParticle(charge * c, 1) for _ in range(2)]
    histories = []
    for x, spin in ((-0.5, [0.02, 0.03, 0.1]), (0.5, [-0.01, 0.04, 0.08])):
        tensor = model.spin_tensor(np.array([1, 0, 0, 0]), np.r_[0, spin])
        histories.append(history([x, 0, 0], c * charge * tensor))
    states = []
    for i, spin in enumerate(([0.02, 0.03, 0.1], [-0.01, 0.04, 0.08])):
        provider = FullDipoleProvider(histories[1 - i], particles[1 - i].charge_native)
        state = initial_state_native(
            [0, -0.5 + i, 0, 0],
            [1, 0, 0, 0],
            c * np.array(spin),
            particles[i],
            provider,
        )
        states.append(state)
    # Only the final accepted source sample is corrected. The older coasting
    # past remains explicitly prescribed, not a self-consistent past solution.
    updated = []
    for i, h in enumerate(histories):
        rate, diagnostic = dynamics_native(
            states[i],
            particles[i],
            FullDipoleProvider(histories[1 - i], particles[1 - i].charge_native),
        )
        velocities, dipoles = h.velocity.copy(), h.dipole.copy()
        velocities[-1], dipoles[-1] = rate[1:4], diagnostic["proper_dipole_native"]
        updated.append(
            FullDipoleHistory(
                h.time, h.position, velocities, dipoles, c, position_tolerance=1e-9
            ).completed()
        )
    return initialize_pair(particles, states, updated)


@pytest.mark.parametrize("charge", [0, 0.03])
def test_reciprocal_checkpoint_resume_and_no_input_mutation(charge):
    payload = make_pair(charge)
    original = copy.deepcopy(payload)
    whole, records = advance_pair(payload, 0.01 / c, steps=2)
    first, _ = advance_pair(payload, 0.01 / c)
    restarted, _ = advance_pair(json.loads(json.dumps(first)), 0.01 / c)
    assert whole == restarted
    assert payload == original
    assert whole["accepted_steps"] == 2
    for record in records:
        for particle in record["particles"]:
            assert abs(particle["length_time"]["mass_constraint"]) < 1e-10
            assert np.linalg.norm(particle["length_time"]["spin_constraint"]) < 1e-10
    if charge:
        assert np.linalg.norm(np.asarray(whole["states"])[:, 5:8]) > 0


def test_failure_is_atomic_and_models_cannot_be_mixed():
    payload = make_pair()
    original = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="outside published history"):
        advance_pair(payload, 3 / c)
    assert payload == original
    bad = copy.deepcopy(payload)
    bad["model"] = "experimental_jakobsen_reciprocal_frozen_v1"
    with pytest.raises(ValueError, match="Wrong nonlinear pair"):
        advance_pair(bad, 0.01 / c)
    with pytest.raises(ValueError, match="not implemented"):
        MomentumCenterParticle(1, 1, reaction_mode="medina_lad")


def test_nonfuture_initial_momentum_rejected():
    with pytest.raises(ValueError, match="timelike initial"):
        initial_state_native(
            [0, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0],
            MomentumCenterParticle(1, 1),
            zero_provider,
        )
