from __future__ import annotations

import json
import math

import numpy as np
import pytest

from core.causal_c5_source_history import (
    CausalC5HistoryUnavailableError,
    CausalC5SourceHistory,
)
from core.constants import C_MMNS


def _accepted_sample(index: int, step_ns: float = 0.002) -> dict[str, object]:
    nominal = float(index) * step_ns
    time = nominal + 0.04 * step_ns * math.sin(0.71 * index)
    angular_frequency = 7.0
    position = np.asarray(
        (
            0.03 * time + 0.002 * math.sin(angular_frequency * time),
            -0.01 * time + 0.001 * math.cos(0.6 * angular_frequency * time),
            0.0005 * math.sin(0.4 * angular_frequency * time),
        )
    )
    velocity = np.asarray(
        (
            0.03 + 0.002 * angular_frequency * math.cos(angular_frequency * time),
            -0.01
            - 0.0006 * angular_frequency * math.sin(0.6 * angular_frequency * time),
            0.0002 * angular_frequency * math.cos(0.4 * angular_frequency * time),
        )
    )
    acceleration = np.asarray(
        (
            -0.002 * angular_frequency**2 * math.sin(angular_frequency * time),
            -0.00036 * angular_frequency**2 * math.cos(0.6 * angular_frequency * time),
            -0.00008 * angular_frequency**2 * math.sin(0.4 * angular_frequency * time),
        )
    )
    cone = 0.73
    spin_angle = 31.0 * time + 0.2
    spin = np.asarray(
        (
            math.sin(cone) * math.cos(spin_angle),
            math.sin(cone) * math.sin(spin_angle),
            math.cos(cone),
        )
    )
    return {
        "time_ns": time,
        "position_mm": position,
        "beta": velocity / C_MMNS,
        "beta_prime_per_mm": acceleration / C_MMNS**2,
        "rest_spin": spin,
    }


def _history(count: int) -> CausalC5SourceHistory:
    history = CausalC5SourceHistory.empty()
    for index in range(count):
        history = history.append_accepted(**_accepted_sample(index))
    return history


def _polynomial_derivative(
    coefficients: np.ndarray,
    fraction: float,
    order: int,
    duration_ns: float,
) -> np.ndarray:
    result = np.zeros(coefficients.shape[1], dtype=float)
    for power in range(order, coefficients.shape[0]):
        scale = math.factorial(power) / math.factorial(power - order)
        result += scale * coefficients[power] * fraction ** (power - order)
    return result / duration_ns**order


def _spin_from_chart(chart: np.ndarray, frame: np.ndarray) -> np.ndarray:
    radius_squared = float(chart @ chart)
    local = np.asarray(
        (
            2.0 * chart[0],
            2.0 * chart[1],
            1.0 - radius_squared,
        )
    ) / (1.0 + radius_squared)
    return local @ frame.T


def test_segment_is_published_only_after_complete_future_window() -> None:
    assert not _history(15).frozen_segments
    ready = _history(16)
    assert len(ready.frozen_segments) == 1
    segment = ready.frozen_segments[0]
    assert segment.left_knot_index == 7
    np.testing.assert_array_equal(segment.spin_window_indices[0], np.arange(15))
    np.testing.assert_array_equal(segment.spin_window_indices[1], np.arange(1, 16))
    with pytest.raises(ValueError):
        segment.spin_window_indices[0, 0] = 99


def test_append_does_not_mutate_accepted_prefix_or_frozen_segment() -> None:
    accepted = _history(18)
    old_times = accepted.time_ns.copy()
    old_position = accepted.position_mm.copy()
    old_coefficients = tuple(
        (
            segment.position_coefficients_mm.copy(),
            segment.rest_spin_stereographic_coefficients.copy(),
        )
        for segment in accepted.frozen_segments
    )
    candidate = accepted.append_accepted(**_accepted_sample(18))
    np.testing.assert_array_equal(accepted.time_ns, old_times)
    np.testing.assert_array_equal(accepted.position_mm, old_position)
    for segment, (position, spin) in zip(candidate.frozen_segments, old_coefficients):
        np.testing.assert_array_equal(segment.position_coefficients_mm, position)
        np.testing.assert_array_equal(
            segment.rest_spin_stereographic_coefficients,
            spin,
        )
    with pytest.raises(ValueError):
        accepted.time_ns[0] = -1.0


def test_adjacent_segments_share_derivatives_through_fifth_order() -> None:
    history = _history(20)
    left, right = history.frozen_segments[:2]
    for order in range(6):
        np.testing.assert_allclose(
            _polynomial_derivative(
                left.position_coefficients_mm,
                1.0,
                order,
                left.duration_ns,
            ),
            _polynomial_derivative(
                right.position_coefficients_mm,
                0.0,
                order,
                right.duration_ns,
            ),
            rtol=2.0e-9,
            atol=2.0e-8,
        )
        np.testing.assert_allclose(
            _polynomial_derivative(
                left.rest_spin_stereographic_coefficients,
                1.0,
                order,
                left.duration_ns,
            ),
            _polynomial_derivative(
                right.rest_spin_stereographic_coefficients,
                0.0,
                order,
                right.duration_ns,
            ),
            rtol=2.0e-9,
            atol=2.0e-7,
        )


def test_stereographic_segment_preserves_unit_spin() -> None:
    segment = _history(16).frozen_segments[0]
    for fraction in np.linspace(0.0, 1.0, 101):
        chart = _polynomial_derivative(
            segment.rest_spin_stereographic_coefficients,
            float(fraction),
            0,
            segment.duration_ns,
        )
        spin = _spin_from_chart(chart, segment.stereographic_frame)
        np.testing.assert_allclose(spin @ spin, 1.0, rtol=0.0, atol=4.0e-16)


def test_unready_time_fails_closed() -> None:
    history = _history(17)
    segment = history.frozen_segments[0]
    assert (
        history.segment_at(segment.start_time_ns + 0.5 * segment.duration_ns) is segment
    )
    assert history.segment_at(segment.end_time_ns) is history.frozen_segments[1]
    with pytest.raises(CausalC5HistoryUnavailableError, match="no causally frozen"):
        history.segment_at(history.time_ns[2])
    with pytest.raises(CausalC5HistoryUnavailableError, match="no causally frozen"):
        history.segment_at(history.time_ns[-1])


def test_checkpoint_roundtrip_preserves_frozen_coefficients_bitwise() -> None:
    history = _history(22)
    payload = json.loads(json.dumps(history.to_checkpoint_payload(), allow_nan=False))
    restored = CausalC5SourceHistory.from_checkpoint_payload(payload)
    np.testing.assert_array_equal(restored.time_ns, history.time_ns)
    np.testing.assert_array_equal(restored.position_mm, history.position_mm)
    np.testing.assert_array_equal(restored.rest_spin, history.rest_spin)
    assert restored.readiness_left_knot == history.readiness_left_knot
    for expected, actual in zip(history.frozen_segments, restored.frozen_segments):
        np.testing.assert_array_equal(
            actual.position_coefficients_mm,
            expected.position_coefficients_mm,
        )
        np.testing.assert_array_equal(
            actual.rest_spin_stereographic_coefficients,
            expected.rest_spin_stereographic_coefficients,
        )
        np.testing.assert_array_equal(
            actual.spin_window_indices,
            expected.spin_window_indices,
        )


def test_fixed_stereographic_chart_pole_fails_closed() -> None:
    history = CausalC5SourceHistory.empty()
    for index in range(15):
        sample = _accepted_sample(index)
        sample["rest_spin"] = (0.0, 0.0, -1.0)
        history = history.append_accepted(**sample)
    sample = _accepted_sample(15)
    sample["rest_spin"] = (0.0, 0.0, -1.0)
    with pytest.raises(CausalC5HistoryUnavailableError, match="chart pole"):
        history.append_accepted(**sample)
