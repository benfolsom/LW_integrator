"""Accuracy and causality checks for the experimental full-tensor history."""

import numpy as np
import pytest

from core.full_dipole_history import FullDipoleHistory


def fixture(width):
    t = np.arange(0.0, 2.0 + width / 2, width)
    x = np.column_stack((0.1 * np.sin(t), 0.03 * np.cos(t), 0 * t))
    v = np.column_stack((0.1 * np.cos(t), -0.03 * np.sin(t), 0 * t))
    d = np.zeros((len(t), 4, 4))
    d[:, 1, 2] = 0.2 * np.sin(t)
    d[:, 2, 1] = -0.2 * np.sin(t)
    d[:, 0, 3] = 0.1 * np.cos(t)
    d[:, 3, 0] = -0.1 * np.cos(t)
    return FullDipoleHistory(t, x, v, d, 1.0)


def test_shared_derivatives_match_analytic_functions():
    h = fixture(0.05).completed()
    errors = []
    for segment in h.segments:
        t = segment.start + 0.37 * segment.duration
        x4, d3 = segment.sample(t, 4)[0], segment.sample(t, 3)[1]
        errors.append(np.linalg.norm(x4 - [0.1 * np.sin(t), 0.03 * np.cos(t), 0]))
        assert d3[1, 2] == pytest.approx(-0.2 * np.cos(t), abs=2e-8)
        assert d3[0, 3] == pytest.approx(0.1 * np.sin(t), abs=2e-8)
    assert max(errors) < 2e-7


def test_append_freezes_prefix_and_delays_unready_intervals():
    raw = fixture(0.05)
    h = FullDipoleHistory(
        raw.time[:20], raw.position[:20], raw.velocity[:20], raw.dipole[:20], 1.0
    ).completed()
    assert h.segments[-1].end == pytest.approx(h.time[-6])
    old = h.segments
    h = h.append(raw.time[20], raw.position[20], raw.velocity[20], raw.dipole[20])
    assert all(a is b for a, b in zip(old, h.segments))
    assert len(h.segments) == len(old) + 1
    with pytest.raises(ValueError, match="extrapolation"):
        h.segments[-1].sample(h.time[-1])
    with pytest.raises(ValueError):
        h.append(h.time[-1], raw.position[20], raw.velocity[20], raw.dipole[20])
    assert old[0].position.flags.writeable is False


def test_join_derivatives_agree_without_revising_past():
    h = fixture(0.05).completed()
    for left, right in zip(h.segments[:-1], h.segments[1:]):
        for k in range(5):
            np.testing.assert_allclose(
                left.sample(left.end, k)[0], right.sample(right.start, k)[0], atol=2e-10
            )
        for k in range(4):
            np.testing.assert_allclose(
                left.sample(left.end, k)[1], right.sample(right.start, k)[1], atol=2e-10
            )


def test_invalid_speed_or_tensor_is_rejected():
    h = fixture(0.05)
    with pytest.raises(ValueError, match="subluminal"):
        FullDipoleHistory(h.time, h.position, h.velocity, h.dipole, 0.01)
    bad = h.dipole.copy()
    bad[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="Antisymmetric"):
        FullDipoleHistory(h.time, h.position, h.velocity, bad, 1.0)


@pytest.mark.parametrize(
    "degree, integrated", [(8, False), (8, True), (10, False), (10, True)]
)
def test_checkpoint_roundtrip_preserves_published_history(degree, integrated):
    import json
    from dataclasses import replace

    h = replace(
        fixture(0.05), derivative_degree=degree, integrate_velocity=integrated
    ).completed()
    restored = FullDipoleHistory.from_checkpoint_payload(
        json.loads(json.dumps(h.to_checkpoint_payload()))
    )
    assert len(restored.segments) == len(h.segments)
    for old, new in zip(h.segments, restored.segments):
        np.testing.assert_array_equal(old.position, new.position)
        np.testing.assert_array_equal(old.dipole, new.dipole)
    assert restored.segment_at(restored.published_until) is restored.segments[-1]
    with pytest.raises(ValueError, match="outside published"):
        restored.segment_at(restored.time[-1])


def test_higher_derivative_accuracy_improves_with_sampling():
    errors = []
    for width in (0.2, 0.1, 0.05):
        h = fixture(width).completed()
        # Earliest coarse windows need a longer fixture.
        if not h.segments:
            continue
        error = max(
            np.linalg.norm(
                s.sample(s.start + 0.37 * s.duration, 3)[1][1, 2]
                + 0.2 * np.cos(s.start + 0.37 * s.duration)
            )
            for s in h.segments
        )
        errors.append(error)
    assert errors[-1] < errors[0]


def test_time_unit_conversion_preserves_source_polynomials():
    raw = fixture(0.05)
    original = raw.completed()
    c = 299.792458
    converted = FullDipoleHistory(
        raw.time / c, raw.position, raw.velocity * c, raw.dipole, c
    ).completed()
    for a, b in zip(original.segments, converted.segments):
        for fraction in (0.2, 0.7):
            np.testing.assert_allclose(
                a.sample(a.start + fraction * a.duration)[0],
                b.sample(b.start + fraction * b.duration)[0],
                atol=1e-14,
            )
            np.testing.assert_allclose(
                a.sample(a.start + fraction * a.duration, 3)[1],
                b.sample(b.start + fraction * b.duration, 3)[1] / c**3,
                atol=1e-11,
            )


def test_large_coordinate_offset_does_not_pollute_high_derivatives():
    raw = fixture(0.05)
    original = raw.completed()
    shifted = FullDipoleHistory(
        raw.time, raw.position + 1e9, raw.velocity, raw.dipole, 1.0
    ).completed()
    for a, b in zip(original.segments, shifted.segments):
        np.testing.assert_array_equal(a.position[1:], b.position[1:])


@pytest.mark.parametrize(
    "integrated, message", [(True, "position error"), (False, "subluminal speed bound")]
)
def test_impossible_between_knot_speed_is_rejected(integrated, message):
    raw = fixture(0.05)
    positions = raw.position.copy()
    positions[7, 0] += 10.0
    with pytest.raises(ValueError, match=message):
        FullDipoleHistory(
            raw.time,
            positions,
            raw.velocity,
            raw.dipole,
            1.0,
            integrate_velocity=integrated,
            position_tolerance=0.0 if integrated else 1e-8,
        ).completed()
