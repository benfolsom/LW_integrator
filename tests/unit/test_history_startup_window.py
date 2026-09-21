"""Right-side startup fits must not mix two distinct smooth branches."""

import json
from dataclasses import replace
from math import comb, factorial

import numpy as np
import pytest

from core.full_dipole_history import FullDipoleHistory


def initial(mode="one_sided"):
    times = np.arange(-20, 1) * 0.05
    return FullDipoleHistory(
        times,
        np.zeros((21, 3)),
        np.zeros((21, 3)),
        np.zeros((21, 4, 4)),
        1.0,
        position_tolerance=1e-8,
        inertial_until=0.0,
        startup_fit=mode,
    ).completed()


def append(history, index):
    t = index * 0.05
    d = np.zeros((4, 4))
    d[1, 2], d[2, 1] = 1e-4 * t**5, -1e-4 * t**5
    return history.append(t, [1e-4 * t**5, 0, 0], [5e-4 * t**4, 0, 0], d)


def test_known_right_polynomial_and_high_derivatives():
    h = initial()
    prefix = h.segments
    for index in range(1, 12):
        h = append(h, index)
    assert all(a is b for a, b in zip(prefix, h.segments))
    for s in h.segments:
        if s.start < 0:
            assert not np.any(s.position)
            assert not np.any(s.dipole)
            continue
        expected = np.zeros(10)
        for k in range(6):
            expected[k] = 1e-4 * comb(5, k) * s.start ** (5 - k) * s.duration**k
        np.testing.assert_allclose(s.position[:, 0], expected, atol=1e-16, rtol=0)
        np.testing.assert_allclose(s.dipole[:, 1, 2], expected[:8], atol=1e-16, rtol=0)
        # Compare the normalized coefficient error. Dividing tiny coefficient
        # noise by duration**8 is not an independent high-derivative guarantee.
        assert (
            np.linalg.norm(s.sample(s.start + s.duration / 2, 8)[0])
            * s.duration**8
            / factorial(8)
            < 1e-16
        )


def test_wait_for_actual_post_start_samples_and_restart():
    h = initial()
    for index in range(1, 10):
        h = append(h, index)
    assert h.published_until == 0
    with pytest.raises(ValueError, match="outside published"):
        h.segment_at(0.01)
    saved = h.to_checkpoint_payload()
    assert saved["format"] == "full-dipole-history-v3"
    restored = FullDipoleHistory.from_checkpoint_payload(json.loads(json.dumps(saved)))
    a, b = append(h, 10), append(restored, 10)
    assert a.published_until > 0
    assert a.to_checkpoint_payload() == b.to_checkpoint_payload()
    for sa, sb in zip(a.segments, b.segments):
        np.testing.assert_array_equal(sa.position, sb.position)
        np.testing.assert_array_equal(sa.dipole, sb.dipole)


def test_legacy_startup_is_not_silently_reinterpreted():
    h = initial("centered")
    for index in range(1, 12):
        h = append(h, index)
    saved = h.to_checkpoint_payload()
    assert saved["format"] == "full-dipole-history-v2"
    restored = FullDipoleHistory.from_checkpoint_payload(saved)
    assert restored.startup_fit == "centered"
    first = next(s for s in restored.segments if s.start == 0)
    assert np.linalg.norm(first.sample(first.duration / 2, 8)[0]) > 1
    for a, b in zip(h.segments, restored.segments):
        np.testing.assert_array_equal(a.position, b.position)


def test_invalid_startup_declarations():
    h = initial()
    with pytest.raises(ValueError, match="requires declared"):
        replace(h, inertial_until=None)
    with pytest.raises(ValueError, match="Startup fit"):
        replace(h, startup_fit="guess")
    payload = h.to_checkpoint_payload()
    payload["format"] = "full-dipole-history-v2"
    with pytest.raises(ValueError, match="selector requires"):
        FullDipoleHistory.from_checkpoint_payload(payload)
