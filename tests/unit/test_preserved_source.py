"""Preserved producer, general intervals, malformed data and restart checks."""

import copy
import json
from dataclasses import replace
from decimal import Decimal, localcontext
from math import comb

import numpy as np
import pytest

from core import momentum_center as model
from core.constants import C_MMNS as c
from core.full_dipole_history import FullDipoleHistory
from core.momentum_center_pair import (
    MomentumCenterParticle,
    initial_state_native,
    initialize_pair,
    advance_pair,
)
from core.preserved_source import (
    decimal_array,
    split_array,
    rounded_state,
    rk_increment,
    dipole_parts,
)


@pytest.mark.parametrize(
    "weights,divisor", [((1, 2, 2, 1), 6), ((1, 4, 1), 6), ((1,), 1)]
)
def test_increment_not_recovered_from_rounded_absolute(weights, divisor):
    ref = np.ones(14)
    high, low = np.zeros(14), np.zeros(14)
    rates = [np.full(14, 1e-23) for _ in weights]
    for _ in range(10):
        high, low = rk_increment(
            high, low, rates, 0.1, weights=weights, divisor=divisor
        )
    np.testing.assert_array_equal(rounded_state(ref, high, low), ref)
    np.testing.assert_allclose(high, 1e-23, rtol=1e-15, atol=0)
    assert np.any(low)


@pytest.mark.parametrize("beta", [0.0, 0.8, 0.99])
def test_normalization_reuses_full_model(beta):
    particle = model.Particle(0.03, 1.0, 2.0)
    field = np.zeros((4, 4))
    field[1, 2], field[2, 1] = 0.2, -0.2
    gradient = np.zeros((4, 4, 4))
    gradient[1] = field * 0.1
    potential = np.array([0.1, 0.2, 0, 0])

    def provider(x):
        return potential, np.zeros((4, 4)), field, gradient

    state = model.initial_state(
        np.zeros(4),
        np.array([1, beta, 0, 0]),
        np.array([0.01, 0.02, 0.03]),
        particle,
        provider,
    )
    _, data = model.evaluate(state, particle, provider)
    native = state.copy()
    native[4:] *= c
    hi, lo = dipole_parts(
        native,
        np.zeros(14),
        np.zeros(14),
        particle,
        potential,
        field,
        gradient,
        np.zeros((4, 4)),
    )
    np.testing.assert_allclose(
        hi + lo, c * data["proper_dipole"], rtol=2e-13, atol=1e-16
    )


def test_spin_and_normalization_changes_both_survive():
    particle = model.Particle(0.03, 1)
    ref = np.zeros(14)
    ref[4] = c
    ref[11] = 0.1 * c
    zero = np.zeros(14)
    a, f, g = np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4, 4))
    h0, l0 = dipole_parts(ref, zero, zero, particle, a, f, g, f)
    # Constant spin, tiny kinetic-mass change: k=q/m changes as well.
    delta = zero.copy()
    delta[4] = 1e-23 * c
    h1, l1 = dipole_parts(ref, delta, zero, particle, a, f, g, h0)
    with localcontext() as context:
        context.prec = 80
        diff = decimal_array(h1) + decimal_array(l1) - decimal_array(l0)
        expected = (
            -decimal_array(h0)[1, 2]
            * Decimal.from_float(delta[4])
            / Decimal.from_float(c)
        )
        assert abs((diff[1, 2] - expected) / expected) < Decimal("1e-14")
    # Constant p and a spin-only increment must independently appear in D.
    delta = zero.copy()
    delta[11] = 1e-24
    h2, l2 = dipole_parts(ref, delta, zero, particle, a, f, g, h0)
    with localcontext() as context:
        context.prec = 80
        diff = decimal_array(h2) + decimal_array(l2) - decimal_array(l0)
        assert abs(float(diff[1, 2]) / (particle.charge * delta[11]) - 1) < 1e-14


def polynomial_history(power=5, irregular=False):
    width = 1.25 / 256
    times = np.arange(-12, 30) * width
    if irregular:
        times[13:] += 0.04 * width * np.sin(np.arange(1, len(times) - 12))
    reference = np.zeros((4, 4))
    reference[1, 2], reference[2, 1] = 0.003, -0.003
    with localcontext() as context:
        context.prec = 80
        values = np.full((len(times), 4, 4), Decimal(0), dtype=object)
        for i, t in enumerate(times):
            values[i, 1, 2] = (
                Decimal("0.000123456789") * Decimal.from_float(max(t, 0)) ** power
            )
            values[i, 2, 1] = -values[i, 1, 2]
        hi, lo = split_array(values)
    return FullDipoleHistory(
        times,
        np.zeros((len(times), 3)),
        np.zeros((len(times), 3)),
        rounded_state(reference, hi, lo),
        1.0,
        inertial_until=0.0,
        startup_fit="one_sided",
        dipole_reference=reference,
        dipole_high=hi,
        dipole_low=lo,
    ).completed()


@pytest.mark.parametrize("power", [4, 5, 6, 7])
@pytest.mark.parametrize("irregular", [False, True])
def test_general_segments_preserve_polynomial_and_restart(power, irregular):
    history = polynomial_history(power, irregular)
    restored = FullDipoleHistory.from_checkpoint_payload(
        json.loads(json.dumps(history.to_checkpoint_payload()))
    )
    for segment, other in zip(history.segments, restored.segments):
        np.testing.assert_array_equal(segment.dipole, other.dipole)
        if segment.start < 0:
            assert not np.any(segment.dipole[1:])
            continue
        expected = np.zeros(8)
        for k in range(1, power + 1):
            expected[k] = (
                0.000123456789
                * comb(power, k)
                * segment.start ** (power - k)
                * segment.duration**k
            )
        actual = segment.dipole[:, 1, 2].copy()
        actual[0] = 0
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-13
    # Adjacent intervals share value and the three defined derivatives.
    for a, b in zip(history.segments, history.segments[1:]):
        if a.start < 0:
            continue
        for order in range(1, 4):
            x, y = a.sample(a.end, order)[1], b.sample(b.start, order)[1]
            assert np.linalg.norm(x - y) <= 1e-12 * max(np.linalg.norm(x), 1e-20)


def fresh_pair(mode="off"):
    times = np.arange(-80, 1) * 0.02 / c
    particles = [
        MomentumCenterParticle(
            0.03 * c, 1, reaction_mode=mode, reaction_derivative_method="analytic"
        )
    ] * 2
    histories, states = [], []
    for i in range(2):

        def zero(t, x):
            return np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4, 4))

        state = initial_state_native(
            [0, i - 0.5, 0, 0],
            [1, 0, 0, 0],
            c * np.array([0.02, 0.03, 0.1]),
            particles[i],
            zero,
        )
        histories.append(
            FullDipoleHistory(
                times,
                np.tile(state[1:4], (len(times), 1)),
                np.zeros((len(times), 3)),
                np.tile(0.03 * model.unpack(state[8:]), (len(times), 1, 1)),
                c,
                position_tolerance=1e-8,
                inertial_until=0,
                startup_fit="one_sided",
            ).completed()
        )
        states.append(state)
    return initialize_pair(
        particles,
        states,
        histories,
        startup_duration_ns=0.5 / c,
        preserve_source_increments=True,
    )


@pytest.mark.parametrize("mode", ["off", "full_dipole_coupled"])
def test_live_preserved_checkpoint_restart_and_atomic_failure(mode):
    payload = fresh_pair(mode)
    original = copy.deepcopy(payload)
    whole, _ = advance_pair(payload, 0.02 / c, 2)
    middle, _ = advance_pair(payload, 0.02 / c, 1)
    restarted, _ = advance_pair(json.loads(json.dumps(middle)), 0.02 / c, 1)
    assert whole == restarted
    assert payload == original
    assert all(h["format"] == "full-dipole-history-v4" for h in whole["histories"])
    assert np.any(whole["source_precision"]["low"])
    with pytest.raises(ValueError):
        advance_pair(payload, 4 / c, 1)
    assert payload == original


def test_reject_incomplete_or_contradictory_precision_metadata():
    h = polynomial_history()
    payload = h.to_checkpoint_payload()
    payload["format"] = "full-dipole-history-v3"
    with pytest.raises(ValueError, match="require history v4"):
        FullDipoleHistory.from_checkpoint_payload(payload)
    with pytest.raises(ValueError, match="all components"):
        replace(h, dipole_low=None)
    pair = fresh_pair()
    del pair["source_precision"]
    with pytest.raises(ValueError, match="metadata"):
        advance_pair(pair, 0.01 / c)


@pytest.mark.parametrize("beta", [0.0, 0.8, 0.99])
@pytest.mark.parametrize("power", [4, 5, 7])
def test_production_history_high_response_coefficient(beta, power):
    from core.full_dipole_response import response_taylor

    h = polynomial_history(power)
    segment = next(s for s in h.segments if s.start == 0)
    x = segment.position.copy()
    x[1, 0] = beta * segment.duration
    d = segment.dipole.copy()
    d[0] = 0  # Isolate the changing source, not a large constant background.
    exact = np.zeros_like(d)
    exact[power, 1, 2] = 0.000123456789 * segment.duration**power
    exact[power, 2, 1] = -exact[power, 1, 2]
    source_time = 0.3 * segment.duration
    events = np.zeros((5, 4))
    events[0] = [source_time + np.sqrt(1 + (beta * source_time) ** 2), 0, 1, 0]
    events[1, 0] = 1
    actual = response_taylor(events, 0.0, segment.duration, x, d, charge=0)[3][4]
    expected = response_taylor(events, 0.0, segment.duration, x, exact, charge=0)[3][4]
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-9


def test_preserved_publication_prefix_and_append_contract():
    h = polynomial_history()
    source = replace(
        h,
        time=h.time[:23],
        position=h.position[:23],
        velocity=h.velocity[:23],
        dipole=h.dipole[:23],
        segments=(),
        dipole_high=h.dipole_high[:23],
        dipole_low=h.dipole_low[:23],
    ).completed()
    prefix = source.segments
    for i in range(23, len(h.time)):
        source = source.append(
            h.time[i],
            h.position[i],
            h.velocity[i],
            h.dipole[i],
            dipole_parts=(h.dipole_high[i], h.dipole_low[i]),
        )
    assert all(a is b for a, b in zip(prefix, source.segments))
    for a, b in zip(source.segments, h.segments):
        np.testing.assert_array_equal(a.dipole, b.dipole)
    with pytest.raises(ValueError, match="preserved append data"):
        source.append(0.5, np.zeros(3), np.zeros(3), h.dipole[-1])
