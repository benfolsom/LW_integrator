"""Tests for explicitly timed local source history and publication."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.causal_local_source_history import (
    AcceptedPairCausalLocalSourceHistory,
    CausalLocalDipoleSourceCollection,
    CausalLocalSourceHistory,
)
from core.growable_causal_local_source_history import (
    GrowableAcceptedPairCausalLocalSourceHistory,
    GrowableCausalLocalSourceHistory,
)
from core.types import GrowableTrajectoryBuilder


def _state(step: int, *, acceleration: float | None = None) -> dict[str, np.ndarray]:
    time = 0.01 * step
    result = {
        "x": np.asarray((0.002 * step,)),
        "y": np.asarray((0.0,)),
        "z": np.asarray((0.0,)),
        "t": np.asarray((time,)),
        "Px": np.asarray((0.0,)),
        "Py": np.asarray((0.0,)),
        "Pz": np.asarray((0.0,)),
        "Pt": np.asarray((1.0,)),
        "gamma": np.asarray((1.0,)),
        "bx": np.asarray((0.01,)),
        "by": np.asarray((0.0,)),
        "bz": np.asarray((0.0,)),
        # Deliberately unrelated: the local history must never consume this.
        "bdotx": np.asarray((9000.0 + step,)),
        "bdoty": np.asarray((-8000.0 - step,)),
        "bdotz": np.asarray((7000.0 + step,)),
        "spin_x": np.asarray((0.0,)),
        "spin_y": np.asarray((0.0,)),
        "spin_z": np.asarray((1.0,)),
        "q": np.asarray((1.0,)),
        "m": np.asarray((1.0,)),
        "magnetic_moment_native": np.asarray((-2.0e-6,)),
        "magnetic_dipole_active": np.asarray((True,)),
    }
    if acceleration is not None:
        result.update(
            {
                "source_start_beta_prime_x_per_mm": np.asarray((acceleration,)),
                "source_start_beta_prime_y_per_mm": np.asarray((2.0 * acceleration,)),
                "source_start_beta_prime_z_per_mm": np.asarray((-acceleration,)),
                "source_start_beta_prime_ready": np.asarray((True,)),
            }
        )
    return result


def _trajectory(count: int = 5):
    builder = GrowableTrajectoryBuilder(4, 1, magnetic_dipole=True)
    builder.append_step(_state(0))
    for step in range(1, count):
        builder.append_step(_state(step, acceleration=100.0 + step))
    return builder.build_current()


def _history() -> CausalLocalSourceHistory:
    times = np.arange(5, dtype=np.float64) * 0.01
    return CausalLocalSourceHistory.from_accepted_samples(
        time_ns=times,
        position_mm=np.column_stack((0.2 * times, np.zeros((5, 2)))),
        beta=np.tile(np.asarray((0.01, 0.0, 0.0)), (5, 1)),
        rest_spin=np.tile(np.asarray((0.0, 0.0, 1.0)), (5, 1)),
        stereographic_frame=np.eye(3),
        interval_start_beta_prime_per_mm=np.column_stack(
            (np.arange(4, dtype=np.float64), np.zeros((4, 2)))
        ),
        interval_start_acceleration_ready=np.asarray((True, True, False, True)),
    )


def test_trajectory_conversion_aligns_interval_acceleration_and_ignores_bdot() -> None:
    collection = CausalLocalDipoleSourceCollection.from_trajectory_arrays(
        _trajectory(),
        identity_prefix="rider",
    )
    history = collection.sources[0].history

    np.testing.assert_array_equal(
        history.interval_start_beta_prime_per_mm[:, 0],
        np.asarray((101.0, 102.0, 103.0, 104.0)),
    )
    np.testing.assert_array_equal(
        history.interval_start_acceleration_ready,
        np.ones(4, dtype=bool),
    )
    assert not np.any(history.interval_start_beta_prime_per_mm[:, 0] > 1000.0)


def test_checkpoint_round_trip_preserves_exact_timing_bitwise() -> None:
    expected = _history()
    payload = json.loads(json.dumps(expected.to_checkpoint_payload()))
    actual = CausalLocalSourceHistory.from_checkpoint_payload(payload)

    for name in (
        "time_ns",
        "position_mm",
        "beta",
        "rest_spin",
        "stereographic_frame",
        "interval_start_beta_prime_per_mm",
        "interval_start_acceleration_ready",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_rejected_preflight_leaves_published_prefix_unchanged() -> None:
    builder = GrowableCausalLocalSourceHistory.from_history(_history())
    accepted = builder.build_current()
    old_times = accepted.time_ns.copy()
    transaction = builder.preflight_append_samples(
        (
            # The final interval and endpoint from the longer reference history.
            # Its acceleration is stored against the previous accepted time.
            _accepted_sample(5, acceleration=105.0),
        )
    )

    with pytest.raises(ValueError, match="increase strictly"):
        builder.preflight_append_samples((_accepted_sample(4, acceleration=104.0),))
    with pytest.raises(RuntimeError, match="stale or foreign"):
        builder.commit(transaction)
    np.testing.assert_array_equal(accepted.time_ns, old_times)
    assert builder.sample_count == 5


def _accepted_sample(step: int, *, acceleration: float):
    from core.causal_local_source_history import accepted_local_source_sample_from_state

    return accepted_local_source_sample_from_state(
        _state(step, acceleration=acceleration),
        0,
    )


def test_pair_transaction_publishes_midpoint_and_endpoint_together() -> None:
    trajectory = _trajectory(4)
    accepted = AcceptedPairCausalLocalSourceHistory.from_trajectory_arrays(
        trajectory,
        trajectory,
    )
    growable = GrowableAcceptedPairCausalLocalSourceHistory.from_accepted(accepted)
    rows = (
        _state(4, acceleration=104.0),
        _state(5, acceleration=105.0),
    )
    transaction = growable.preflight_states(rider_states=rows, driver_states=rows)

    assert growable.build_current().rider.sources[0].history.sample_count == 4
    candidate = transaction.candidate.rider.sources[0].history
    assert candidate.sample_count == 6
    np.testing.assert_array_equal(
        candidate.interval_start_beta_prime_per_mm[-2:, 0],
        np.asarray((104.0, 105.0)),
    )
    committed = growable.commit(transaction)
    assert committed.rider.sources[0].history.sample_count == 6
    assert committed.driver.sources[0].history.sample_count == 6
    assert accepted.rider.sources[0].history.sample_count == 4


def test_pair_preflight_failure_publishes_neither_role() -> None:
    trajectory = _trajectory(4)
    accepted = AcceptedPairCausalLocalSourceHistory.from_trajectory_arrays(
        trajectory,
        trajectory,
    )
    growable = GrowableAcceptedPairCausalLocalSourceHistory.from_accepted(accepted)
    invalid_driver = _state(4, acceleration=104.0)
    invalid_driver["magnetic_moment_native"] = np.asarray((-3.0e-6,))

    with pytest.raises(ValueError, match="changed magnetic moment"):
        growable.preflight_states(
            rider_states=(_state(4, acceleration=104.0),),
            driver_states=(invalid_driver,),
        )
    current = growable.build_current()
    assert current.rider.sources[0].history.sample_count == 4
    assert current.driver.sources[0].history.sample_count == 4


def test_single_interval_commits_grow_geometrically_without_rewriting_prefix() -> None:
    builder = GrowableCausalLocalSourceHistory.from_history(
        _history(),
        minimum_capacity=2,
    )
    accepted = builder.build_current()
    original_times = accepted.time_ns.copy()
    for step in range(5, 205):
        transaction = builder.preflight_append_samples(
            (_accepted_sample(step, acceleration=100.0 + step),)
        )
        builder.commit(transaction)

    np.testing.assert_array_equal(accepted.time_ns, original_times)
    assert builder.sample_count == 205
    assert builder.allocated_capacity < 2 * builder.sample_count
