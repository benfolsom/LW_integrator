"""Declared coasting past must not acquire motion from later samples."""

import json
from dataclasses import replace

import numpy as np
import pytest

from core.full_dipole_history import FullDipoleHistory


def history(beta=0.0, declared=True):
    time = np.arange(-20, 1) * 0.025
    velocity = np.tile([beta, 0, 0], (len(time), 1))
    position = np.column_stack((0.5 + beta * time, 0 * time, 0 * time))
    dipole = np.zeros((len(time), 4, 4))
    dipole[:, 1, 2], dipole[:, 2, 1] = 0.1, -0.1
    return FullDipoleHistory(
        time,
        position,
        velocity,
        dipole,
        1.0,
        position_tolerance=1e-7,
        inertial_until=0.0 if declared else None,
    ).completed()


@pytest.mark.parametrize("beta", [0.0, 0.8, 0.99])
def test_later_motion_does_not_change_prescribed_past_or_restart(beta):
    h = history(beta)
    prefix = h.segments
    for index in range(1, 9):
        t = index * 0.025
        # Smooth departure: x changes at order eight and D at order seven.
        x = np.array([0.5 + beta * t, 1e-5 * t**8, 0])
        v = np.array([beta, 8e-5 * t**7, 0])
        d = h.dipole[0].copy()
        d[0, 3], d[3, 0] = 1e-5 * t**7, -1e-5 * t**7
        h = h.append(t, x, v, d)
    assert all(a is b for a, b in zip(prefix, h.segments))
    for segment in h.segments:
        if segment.start >= 0:
            continue
        assert not np.any(segment.position[2:])
        assert not np.any(segment.dipole[1:])
        np.testing.assert_allclose(
            segment.position[1] / segment.duration, [beta, 0, 0], rtol=0, atol=2e-15
        )
    restored = FullDipoleHistory.from_checkpoint_payload(
        json.loads(json.dumps(h.to_checkpoint_payload()))
    )
    assert restored.inertial_until == 0
    assert restored.to_checkpoint_payload() == h.to_checkpoint_payload()
    for a, b in zip(h.segments, restored.segments):
        np.testing.assert_array_equal(a.position, b.position)
        np.testing.assert_array_equal(a.dipole, b.dipole)


def test_old_checkpoint_stays_undeclared():
    h = history(declared=False)
    payload = h.to_checkpoint_payload()
    assert payload["format"] == "full-dipole-history-v1"
    assert "inertial_until" not in payload
    assert FullDipoleHistory.from_checkpoint_payload(payload).inertial_until is None


def test_invalid_prehistory_declaration_is_rejected():
    h = history()
    with pytest.raises(ValueError, match="accepted knot"):
        replace(h, inertial_until=0.001)
    changed = h.velocity.copy()
    changed[-1, 1] = 0.01
    with pytest.raises(ValueError, match="not coasting"):
        replace(h, velocity=changed)
    payload = h.to_checkpoint_payload()
    del payload["inertial_until"]
    with pytest.raises(ValueError, match="version"):
        FullDipoleHistory.from_checkpoint_payload(payload)


def test_startup_knot_shares_exact_pre_start_derivatives():
    h = history()
    for index in range(1, 7):
        t = index * 0.025
        d = h.dipole[0].copy()
        d[0, 3], d[3, 0] = t**7, -(t**7)
        h = h.append(t, [0.5, 0, 0], [0, 0, 0], d)
    right = next(s for s in h.segments if s.start == 0)
    for order in range(1, 4):
        np.testing.assert_array_equal(right.sample(0, order)[1], 0)
    assert np.any(right.dipole[4:])  # Evolved motion was not removed.


def test_native_pair_carries_explicit_history_declaration_through_restart():
    from core import momentum_center_pair as pair
    from core.constants import C_MMNS as c
    from tests.unit.test_momentum_center_pair import (
        history as native_history,
        zero_provider,
    )

    particles = [pair.MomentumCenterParticle(0, 1) for _ in range(2)]
    histories = [native_history([x, 0, 0]) for x in (-0.5, 0.5)]
    states = [
        pair.initial_state_native(
            [0, x, 0, 0], [1, 0, 0, 0], np.zeros(3), p, zero_provider
        )
        for x, p in zip((-0.5, 0.5), particles)
    ]
    payload = pair.initialize_pair(
        particles,
        states,
        histories,
        startup_duration_ns=0.5 / c,
        inertial_prehistory=True,
    )
    assert all(h["format"] == "full-dipole-history-v2" for h in payload["histories"])
    first, _ = pair.advance_pair(payload, 0.01 / c)
    resumed, _ = pair.advance_pair(json.loads(json.dumps(first)), 0.01 / c)
    together, _ = pair.advance_pair(payload, 0.01 / c, 2)
    assert together == resumed
