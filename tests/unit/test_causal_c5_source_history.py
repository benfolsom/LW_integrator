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
from core.dipole_hertz_jet import evaluate_causal_c5_dipole_hertz_response_native
from core.retarded_fields import ObserverEvent


def _accepted_sample_at_time(time: float) -> dict[str, object]:
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


def _accepted_sample(index: int, step_ns: float = 0.002) -> dict[str, object]:
    nominal = float(index) * step_ns
    time = nominal + 0.04 * step_ns * math.sin(0.71 * index)
    return _accepted_sample_at_time(time)


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


def test_bounded_nonuniform_cadence_matches_incremental_history_bitwise() -> None:
    increments = 0.002 * (1.0 + 0.2 * np.sin(0.73 * np.arange(40, dtype=np.float64)))
    times = np.concatenate((np.zeros(1), np.cumsum(increments)))
    samples = tuple(_accepted_sample_at_time(float(time)) for time in times)
    incremental = CausalC5SourceHistory.empty()
    for sample in samples:
        incremental = incremental.append_accepted(**sample)

    batch = CausalC5SourceHistory.from_accepted_samples(
        time_ns=times,
        position_mm=np.asarray([sample["position_mm"] for sample in samples]),
        beta=np.asarray([sample["beta"] for sample in samples]),
        beta_prime_per_mm=np.asarray(
            [sample["beta_prime_per_mm"] for sample in samples]
        ),
        rest_spin=np.asarray([sample["rest_spin"] for sample in samples]),
    )

    assert len(batch.frozen_segments) == len(incremental.frozen_segments)
    assert (
        max(segment.spin_condition_number for segment in batch.frozen_segments) < 1.0e4
    )
    for expected, actual in zip(incremental.frozen_segments, batch.frozen_segments):
        np.testing.assert_array_equal(
            actual.position_coefficients_mm,
            expected.position_coefficients_mm,
        )
        np.testing.assert_array_equal(
            actual.rest_spin_stereographic_coefficients,
            expected.rest_spin_stereographic_coefficients,
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


def test_frozen_worldline_solves_its_own_light_cone() -> None:
    history = _history(20)
    segment = history.frozen_segments[1]
    expected_time = segment.start_time_ns + 0.37 * segment.duration_ns
    source_position, _velocity = segment.position_velocity_at(expected_time)
    displacement = np.asarray((1.2, -0.4, 0.3))
    observer_position = source_position + displacement
    event = ObserverEvent(
        expected_time + float(np.linalg.norm(displacement)) / C_MMNS,
        tuple(observer_position),
    )
    root = history.solve_retarded_root(
        observer_time_ns=event.time_ns,
        observer_position_mm=event.position_mm,
    )
    np.testing.assert_allclose(
        root.retarded_time_ns,
        expected_time,
        rtol=0.0,
        atol=2.0e-18,
    )
    assert root.segment is segment
    assert abs(root.residual_mm) < 2.0e-14
    response = evaluate_causal_c5_dipole_hertz_response_native(
        history,
        event,
        magnetic_moment_native=-0.7,
    )
    np.testing.assert_allclose(
        response.retarded_time_ns,
        expected_time,
        rtol=0.0,
        atol=2.0e-18,
    )
    assert np.all(np.isfinite(response.four_potential))
    assert np.all(np.isfinite(response.partial_f))


def test_checkpoint_restore_reproduces_c5_hertz_response() -> None:
    history = _history(20)
    segment = history.frozen_segments[1]
    root_time = segment.start_time_ns + 0.41 * segment.duration_ns
    source_position, _velocity = segment.position_velocity_at(root_time)
    observer_position = source_position + np.asarray((0.8, 0.2, -0.1))
    event = ObserverEvent(
        root_time + float(np.linalg.norm(observer_position - source_position)) / C_MMNS,
        tuple(observer_position),
    )
    restored = CausalC5SourceHistory.from_checkpoint_payload(
        json.loads(json.dumps(history.to_checkpoint_payload(), allow_nan=False))
    )
    expected = evaluate_causal_c5_dipole_hertz_response_native(
        history,
        event,
        magnetic_moment_native=0.3,
    )
    actual = evaluate_causal_c5_dipole_hertz_response_native(
        restored,
        event,
        magnetic_moment_native=0.3,
    )
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_light_cone_in_unready_tail_fails_closed() -> None:
    history = _history(20)
    source_time = float(history.time_ns[-1])
    source_position = history.position_mm[-1]
    event = ObserverEvent(
        source_time + 1.0 / C_MMNS, tuple(source_position + (1, 0, 0))
    )
    with pytest.raises(CausalC5HistoryUnavailableError, match="unready future"):
        history.solve_retarded_root(
            observer_time_ns=event.time_ns,
            observer_position_mm=event.position_mm,
        )


def test_relativistic_frozen_worldline_root_is_not_a_slow_speed_approximation() -> None:
    beta = np.asarray((0.81, -0.13, 0.07))
    history = CausalC5SourceHistory.empty()
    for index in range(20):
        time = -0.03 + index * 0.001
        angle = 14.0 * time
        history = history.append_accepted(
            time_ns=time,
            position_mm=C_MMNS * beta * time,
            beta=beta,
            beta_prime_per_mm=(0.0, 0.0, 0.0),
            rest_spin=(
                math.sin(0.6) * math.cos(angle),
                math.sin(0.6) * math.sin(angle),
                math.cos(0.6),
            ),
        )
    segment = history.frozen_segments[1]
    expected_time = segment.start_time_ns + 0.63 * segment.duration_ns
    source_position, _velocity = segment.position_velocity_at(expected_time)
    displacement = np.asarray((-0.5, 1.1, 0.4))
    observer_position = source_position + displacement
    root = history.solve_retarded_root(
        observer_time_ns=(expected_time + float(np.linalg.norm(displacement)) / C_MMNS),
        observer_position_mm=observer_position,
    )
    np.testing.assert_allclose(
        root.retarded_time_ns,
        expected_time,
        rtol=0.0,
        atol=5.0e-18,
    )
    np.testing.assert_allclose(root.source_beta, beta, rtol=0.0, atol=4.0e-14)
