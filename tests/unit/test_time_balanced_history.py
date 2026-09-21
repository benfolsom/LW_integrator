"""Nonuniform sampling must remain causal, append-stable and restartable."""

from fractions import Fraction as Q
import json
from math import factorial

import numpy as np
import pytest

from core.full_dipole_history import FullDipoleHistory
from core.history_sampling import time_balanced_indices


@pytest.mark.parametrize("ratio", [2, 8, 16])
def test_selection_spreads_time_and_freezes_at_earliest_prefix(ratio):
    times = np.r_[np.arange(21) * 0.025, 0.5 + np.arange(1, 180) * (0.025 / ratio)]
    knot = 19
    assert time_balanced_indices(times[: knot + 6], knot) is None
    found = None
    for size in range(knot + 6, len(times) + 1):
        chosen = time_balanced_indices(times[:size], knot)
        if chosen is not None:
            if found is None:
                found = chosen
            np.testing.assert_array_equal(chosen, found)
    assert found is not None and found[5] == knot
    assert len(set(found)) == 11
    nodes = times[found] - times[knot]
    assert (
        np.linalg.cond(np.polynomial.polynomial.polyvander(nodes / max(abs(nodes)), 10))
        < 1e8
    )
    assert np.max(np.diff(found[5:])) > 1


def source(times, beta=0.0):
    position = np.column_stack((0.5 + beta * times, times * 0, times * 0))
    velocity = np.tile([beta, 0, 0], (len(times), 1))
    reference = np.zeros((4, 4))
    reference[1, 2], reference[2, 1] = 0.003, -0.003
    high, low, dipole = [np.zeros((len(times), 4, 4)) for _ in range(3)]
    for i, t in enumerate(times):
        value = Q(1, 10**12) * Q(float(t)) ** 7
        head = float(value)
        tail = float(value - Q(head))
        high[i, 1, 2], high[i, 2, 1] = head, -head
        low[i, 1, 2], low[i, 2, 1] = tail, -tail
        dipole[i, 1, 2] = float(Q(0.003) + value)
        dipole[i, 2, 1] = -dipole[i, 1, 2]
    return FullDipoleHistory(
        times,
        position,
        velocity,
        dipole,
        1.0,
        position_tolerance=1e-12,
        dipole_reference=reference,
        dipole_high=high,
        dipole_low=low,
        derivative_sampling="time_balanced",
    )


@pytest.mark.parametrize("ratio", [2, 8, 16])
@pytest.mark.parametrize("beta", [0.0, 0.8, 0.99])
def test_tiny_polynomial_high_derivatives_across_spacing_transition(ratio, beta):
    times = np.r_[np.arange(21) * 0.025, 0.5 + np.arange(1, 100) * (0.025 / ratio)]
    history = source(times, beta).completed()
    assert len(history.segments) > 20
    for segment in history.segments:
        seventh = segment.sample(segment.start + segment.duration / 2, 7)[1][1, 2]
        assert abs(seventh / (factorial(7) * 1e-12) - 1) < 1e-9
    restored = FullDipoleHistory.from_checkpoint_payload(
        json.loads(json.dumps(history.to_checkpoint_payload()))
    )
    for a, b in zip(history.segments, restored.segments):
        np.testing.assert_array_equal(a.position, b.position)
        np.testing.assert_array_equal(a.dipole, b.dipole)


def test_explicit_transition_preserves_published_past_and_shared_derivatives():
    from dataclasses import replace

    times = np.r_[np.arange(21) * 0.025, 0.5 + np.arange(1, 100) * (0.025 / 8)]
    complete = source(times)
    old = replace(source(times[:21]), derivative_sampling="consecutive").completed()
    new = old.with_time_balanced_sampling()
    assert all(a is b for a, b in zip(old.segments, new.segments))
    first_new = len(old.segments)
    for i in range(21, len(times)):
        prefix = new.segments
        new = new.append(
            times[i],
            complete.position[i],
            complete.velocity[i],
            complete.dipole[i],
            dipole_parts=(complete.dipole_high[i], complete.dipole_low[i]),
        )
        assert all(a is b for a, b in zip(prefix, new.segments))
    restored = FullDipoleHistory.from_checkpoint_payload(new.to_checkpoint_payload())
    for a, b in zip(new.segments, restored.segments):
        np.testing.assert_array_equal(a.position, b.position)
        np.testing.assert_array_equal(a.dipole, b.dipole)
    left, right = new.segments[first_new - 1 : first_new + 1]
    for order in range(1, 4):
        np.testing.assert_allclose(
            left.sample(left.end, order)[1],
            right.sample(right.start, order)[1],
            atol=1e-23,
            rtol=1e-10,
        )


def test_sampling_checkpoint_cannot_masquerade_as_legacy():
    payload = source(np.arange(20) * 0.025).completed().to_checkpoint_payload()
    assert payload["format"] == "full-dipole-history-v5"
    payload["format"] = "full-dipole-history-v4"
    with pytest.raises(ValueError, match="requires history v5"):
        FullDipoleHistory.from_checkpoint_payload(payload)


@pytest.mark.parametrize("ratio", [2, 8, 16])
def test_fine_to_coarse_spacing_retains_tiny_seventh_derivative(ratio):
    times = np.r_[
        np.arange(41) * (0.025 / ratio), 1.0 / ratio + np.arange(1, 26) * 0.025
    ]
    history = source(times, beta=0.99).completed()
    for segment in history.segments:
        value = segment.sample(segment.start + segment.duration / 2, 7)[1][1, 2]
        assert abs(value / (factorial(7) * 1e-12) - 1) < 1e-9
