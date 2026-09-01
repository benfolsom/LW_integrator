from __future__ import annotations

import math

import numpy as np
import pytest

from core.causal_c5_source_history import (
    CausalC5HistoryUnavailableError,
    CausalC5SourceHistory,
)
from core.constants import C_MMNS
from core.dipole_hertz_jet import evaluate_causal_c5_dipole_hertz_response_native
from core.growable_causal_c5_source_history import GrowableCausalC5SourceHistory
from core.retarded_fields import ObserverEvent


def _sample(time_ns: float) -> dict[str, object]:
    time = float(time_ns)
    beta = np.asarray((0.2, -0.04, 0.01))
    angle = 0.3 + 4.0 * time
    spin = np.asarray(
        (
            math.sin(0.7) * math.cos(angle),
            math.sin(0.7) * math.sin(angle),
            math.cos(0.7),
        )
    )
    return {
        "time_ns": time,
        "position_mm": C_MMNS * beta * time,
        "beta": beta,
        "beta_prime_per_mm": np.zeros(3),
        "rest_spin": spin,
    }


def _preflight(
    builder: GrowableCausalC5SourceHistory,
    times: np.ndarray,
):
    samples = tuple(_sample(float(time)) for time in times)
    return builder.preflight_append_many(
        time_ns=times,
        position_mm=np.asarray([sample["position_mm"] for sample in samples]),
        beta=np.asarray([sample["beta"] for sample in samples]),
        beta_prime_per_mm=np.asarray(
            [sample["beta_prime_per_mm"] for sample in samples]
        ),
        rest_spin=np.asarray([sample["rest_spin"] for sample in samples]),
    )


def _immutable(times: np.ndarray) -> CausalC5SourceHistory:
    history = CausalC5SourceHistory.empty()
    for time in times:
        history = history.append_accepted(**_sample(float(time)))
    return history


def test_rejected_preflight_leaves_published_prefix_unchanged() -> None:
    builder = GrowableCausalC5SourceHistory(initial_capacity=4)
    first = _preflight(builder, np.arange(18) * 0.01)
    accepted = builder.commit(first)
    old_times = accepted.time_ns.copy()
    old_segments = tuple(accepted.frozen_segments)

    rejected = _preflight(builder, np.asarray((0.18, 0.19)))

    assert builder.sample_count == 18
    np.testing.assert_array_equal(accepted.time_ns, old_times)
    assert tuple(accepted.frozen_segments) == old_segments
    assert rejected.candidate.sample_count == 20


def test_stale_or_foreign_transaction_cannot_commit() -> None:
    first = GrowableCausalC5SourceHistory()
    second = GrowableCausalC5SourceHistory()
    transaction = _preflight(first, np.arange(4) * 0.01)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        second.commit(transaction)

    replacement = _preflight(first, np.arange(5) * 0.01)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        first.commit(transaction)
    assert first.commit(replacement).sample_count == 5


def test_failed_numerical_preflight_invalidates_overwritten_candidate() -> None:
    builder = GrowableCausalC5SourceHistory()
    builder.commit(_preflight(builder, np.arange(15) * 0.01))
    pending = _preflight(builder, np.asarray((0.15,)))
    invalid = _sample(0.15)
    invalid["rest_spin"] = np.asarray((0.0, 0.0, -1.0))

    with pytest.raises(CausalC5HistoryUnavailableError, match="chart pole"):
        builder.preflight_append_many(
            time_ns=(0.15,),
            position_mm=(invalid["position_mm"],),
            beta=(invalid["beta"],),
            beta_prime_per_mm=(invalid["beta_prime_per_mm"],),
            rest_spin=(invalid["rest_spin"],),
        )
    with pytest.raises(RuntimeError, match="stale or foreign"):
        builder.commit(pending)


def test_growable_segments_match_immutable_oracle_bitwise() -> None:
    increments = 0.01 * (
        1.0 + 0.2 * np.sin(0.61 * np.arange(40, dtype=np.float64))
    )
    times = np.concatenate((np.zeros(1), np.cumsum(increments)))
    expected = _immutable(times)
    builder = GrowableCausalC5SourceHistory(initial_capacity=2)
    for start in range(0, times.size, 2):
        builder.commit(_preflight(builder, times[start : start + 2]))
    actual = builder.build_current()

    np.testing.assert_array_equal(actual.time_ns, expected.time_ns)
    assert len(actual.frozen_segments) == len(expected.frozen_segments)
    for expected_segment, actual_segment in zip(
        expected.frozen_segments, actual.frozen_segments
    ):
        np.testing.assert_array_equal(
            actual_segment.position_coefficients_mm,
            expected_segment.position_coefficients_mm,
        )
        np.testing.assert_array_equal(
            actual_segment.rest_spin_stereographic_coefficients,
            expected_segment.rest_spin_stereographic_coefficients,
        )
    assert builder.allocated_capacity < 2 * times.size


def test_growable_view_reproduces_provider_response() -> None:
    times = np.arange(24, dtype=np.float64) * 0.01
    expected = _immutable(times)
    builder = GrowableCausalC5SourceHistory.from_history(expected)
    actual = builder.build_current()
    segment = expected.frozen_segments[2]
    root_time = segment.start_time_ns + 0.4 * segment.duration_ns
    source_position, _velocity = segment.position_velocity_at(root_time)
    displacement = np.asarray((1.0, -0.3, 0.2))
    event = ObserverEvent(
        root_time + float(np.linalg.norm(displacement)) / C_MMNS,
        tuple(source_position + displacement),
    )

    reference = evaluate_causal_c5_dipole_hertz_response_native(
        expected,
        event,
        magnetic_moment_native=-0.7,
    )
    result = evaluate_causal_c5_dipole_hertz_response_native(
        actual,  # type: ignore[arg-type]
        event,
        magnetic_moment_native=-0.7,
    )
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        np.testing.assert_array_equal(getattr(result, name), getattr(reference, name))


def test_commit_cost_does_not_rebuild_old_segments() -> None:
    builder = GrowableCausalC5SourceHistory(initial_capacity=2)
    builder.commit(_preflight(builder, np.arange(20) * 0.01))
    accepted = builder.build_current()
    old_segments = tuple(accepted.frozen_segments)

    for index in range(20, 200):
        builder.commit(_preflight(builder, np.asarray((0.01 * index,))))

    current = builder.build_current()
    assert all(
        old is new
        for old, new in zip(old_segments, current.frozen_segments[: len(old_segments)])
    )
    assert builder.allocated_capacity < 2 * builder.sample_count
