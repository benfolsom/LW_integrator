from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.causal_c5_dipole_provider import (
    AcceptedPairCausalC5SourceHistory,
    CausalC5DipoleSourceCollection,
    GrowableAcceptedPairCausalC5SourceHistory,
    evaluate_causal_c5_dipole_source_collection_native,
)
from core.causal_c5_source_history import CausalC5HistoryUnavailableError
from core.retarded_fields import ObserverEvent
from core.types import GrowableTrajectoryBuilder


def _state(step: int, *, particles: int = 2) -> dict[str, np.ndarray]:
    time = 0.01 * step
    angles = np.asarray((0.12 * time, -0.08 * time))[:particles]
    return {
        "x": np.asarray((-0.5, 0.75))[:particles],
        "y": np.asarray((0.1, -0.2))[:particles],
        "z": np.zeros(particles),
        "t": np.full(particles, time),
        "Px": np.zeros(particles),
        "Py": np.zeros(particles),
        "Pz": np.zeros(particles),
        "Pt": np.ones(particles),
        "gamma": np.ones(particles),
        "bx": np.zeros(particles),
        "by": np.zeros(particles),
        "bz": np.zeros(particles),
        "bdotx": np.zeros(particles),
        "bdoty": np.zeros(particles),
        "bdotz": np.zeros(particles),
        "spin_x": np.sin(angles),
        "spin_y": np.zeros(particles),
        "spin_z": np.cos(angles),
        "q": np.ones(particles),
        "m": np.ones(particles),
        "magnetic_moment_native": np.asarray((2.0e-6, -3.0e-6))[:particles],
        "magnetic_dipole_active": np.ones(particles),
    }


def _trajectory(count: int = 21, *, particles: int = 2):
    builder = GrowableTrajectoryBuilder(8, particles, magnetic_dipole=True)
    for step in range(count):
        builder.append_step(_state(step, particles=particles))
    return builder.build_current()


def _event() -> ObserverEvent:
    return ObserverEvent(time_ns=0.125, position_mm=np.asarray((10.0, 0.0, 0.0)))


def test_ordered_source_sum_matches_manual_addition_bitwise() -> None:
    collection = CausalC5DipoleSourceCollection.from_trajectory_arrays(
        _trajectory(),
        identity_prefix="driver",
    )
    combined = evaluate_causal_c5_dipole_source_collection_native(
        collection,
        _event(),
        root_tolerance_mm=1.0e-12,
    )
    first = evaluate_causal_c5_dipole_source_collection_native(
        CausalC5DipoleSourceCollection((collection.sources[0],)),
        _event(),
        root_tolerance_mm=1.0e-12,
    )
    second = evaluate_causal_c5_dipole_source_collection_native(
        CausalC5DipoleSourceCollection((collection.sources[1],)),
        _event(),
        root_tolerance_mm=1.0e-12,
    )

    assert tuple(item.identity for item in combined.source_results) == (
        "driver:0",
        "driver:1",
    )
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        expected = np.array(getattr(first, name), copy=True)
        expected += getattr(second, name)
        np.testing.assert_array_equal(getattr(combined, name), expected)


def test_exclusion_preserves_declared_source_identity() -> None:
    collection = CausalC5DipoleSourceCollection.from_trajectory_arrays(
        _trajectory(),
        identity_prefix="driver",
    )
    result = evaluate_causal_c5_dipole_source_collection_native(
        collection,
        _event(),
        excluded_source_identities=("driver:0",),
        root_tolerance_mm=1.0e-12,
    )
    direct = evaluate_causal_c5_dipole_source_collection_native(
        CausalC5DipoleSourceCollection((collection.sources[1],)),
        _event(),
        root_tolerance_mm=1.0e-12,
    )
    assert tuple(item.identity for item in result.source_results) == ("driver:1",)
    np.testing.assert_array_equal(result.partial_f, direct.partial_f)


def test_first_unready_source_controls_failure_order() -> None:
    full = CausalC5DipoleSourceCollection.from_trajectory_arrays(
        _trajectory(),
        identity_prefix="driver",
    )
    late_event = ObserverEvent(
        time_ns=0.18,
        position_mm=np.asarray((10.0, 0.0, 0.0)),
    )
    with pytest.raises(
        CausalC5HistoryUnavailableError,
        match="source identity 'driver:0'",
    ):
        evaluate_causal_c5_dipole_source_collection_native(
            full,
            late_event,
            root_tolerance_mm=1.0e-12,
        )


def test_pair_candidate_append_is_detached() -> None:
    trajectory = _trajectory(particles=1)
    accepted = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        trajectory,
        trajectory,
    )
    state = _state(21, particles=1)
    rider_candidate = accepted.rider.append_accepted_state(state)

    assert accepted.rider.sources[0].history.sample_count == 21
    assert rider_candidate.sources[0].history.sample_count == 22
    old_segments = accepted.rider.sources[0].history.frozen_segments
    new_segments = rider_candidate.sources[0].history.frozen_segments
    for old, new in zip(old_segments, new_segments[: len(old_segments)]):
        assert old is new
        np.testing.assert_array_equal(
            old.position_coefficients_mm,
            new.position_coefficients_mm,
        )


def test_provider_result_arrays_are_read_only() -> None:
    collection = CausalC5DipoleSourceCollection.from_trajectory_arrays(
        _trajectory(),
        identity_prefix="driver",
    )
    result = evaluate_causal_c5_dipole_source_collection_native(
        collection,
        _event(),
        root_tolerance_mm=1.0e-12,
    )

    for name in (
        "four_potential",
        "partial_a",
        "electric_field_native",
        "magnetic_field_native",
        "field_tensor",
        "partial_f",
    ):
        assert not getattr(result, name).flags.writeable


def test_growable_pair_commit_matches_immutable_candidate_bitwise() -> None:
    trajectory = _trajectory()
    accepted = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        trajectory,
        trajectory,
    )
    growable = GrowableAcceptedPairCausalC5SourceHistory.from_accepted(accepted)
    states = (_state(21), _state(22))
    transaction = growable.preflight_states(
        rider_states=states,
        driver_states=states,
    )

    assert growable.build_current().rider.sources[0].history.sample_count == 21
    assert transaction.candidate.rider.sources[0].history.sample_count == 23
    expected_rider = accepted.rider
    for state in states:
        expected_rider = expected_rider.append_accepted_state(state)
    for candidate_source, expected_source in zip(
        transaction.candidate.rider.sources,
        expected_rider.sources,
    ):
        for candidate_segment, expected_segment in zip(
            candidate_source.history.frozen_segments,
            expected_source.history.frozen_segments,
        ):
            np.testing.assert_array_equal(
                candidate_segment.position_coefficients_mm,
                expected_segment.position_coefficients_mm,
            )
            np.testing.assert_array_equal(
                candidate_segment.rest_spin_stereographic_coefficients,
                expected_segment.rest_spin_stereographic_coefficients,
            )

    committed = growable.commit(transaction)
    assert committed.rider.sources[0].history.sample_count == 23
    assert committed.driver.sources[0].history.sample_count == 23
    assert accepted.rider.sources[0].history.sample_count == 21


def test_growable_pair_failure_publishes_neither_role() -> None:
    trajectory = _trajectory()
    accepted = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        trajectory,
        trajectory,
    )
    growable = GrowableAcceptedPairCausalC5SourceHistory.from_accepted(accepted)
    invalid_driver = _state(21)
    invalid_driver["magnetic_moment_native"] = np.asarray((2.0e-6, -4.0e-6))

    with pytest.raises(ValueError, match="changed magnetic moment"):
        growable.preflight_states(
            rider_states=(_state(21),),
            driver_states=(invalid_driver,),
        )

    current = growable.build_current()
    assert all(source.history.sample_count == 21 for source in current.rider.sources)
    assert all(source.history.sample_count == 21 for source in current.driver.sources)


def test_duplicate_identities_fail_closed() -> None:
    collection = CausalC5DipoleSourceCollection.from_trajectory_arrays(
        _trajectory(),
        identity_prefix="driver",
    )
    with pytest.raises(ValueError, match="identities must be unique"):
        CausalC5DipoleSourceCollection(
            (collection.sources[0], replace(collection.sources[1], identity="driver:0"))
        )
