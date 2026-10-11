"""The fused source loop retains every scalar jet bit and source position."""

import numpy as np
import pytest

from core.precise_charge import (
    precise_charge_jets_strict_serial,
    resolved_separation_frame,
)
from core.resolved_charge_batch import resolved_charge_jets_strict_serial

pytestmark = pytest.mark.unit


def test_batch_matches_scalar_jets_including_optional_geometry_and_snap():
    rng = np.random.default_rng(481)
    count = 8
    charges = rng.normal(size=count)
    separations = rng.normal(size=(count, 3)) * 10
    velocities = rng.normal(size=(count, 3))
    velocities[0] = [0, 0, 1e8]
    accelerations = rng.normal(size=(count, 3)) * 1e-5
    jerks = rng.normal(size=(count, 3)) * 1e-7
    frames = separations.copy()
    snaps = rng.normal(size=(count, 3)) * 1e-9
    accelerations[0] = jerks[0] = snaps[0] = 0
    has_frame = np.arange(count) % 2 == 0
    has_snap = np.arange(count) % 3 == 0
    actual = resolved_charge_jets_strict_serial(
        charges,
        separations,
        velocities,
        accelerations,
        jerks,
        frames,
        snaps,
        has_frame,
        has_snap,
    )
    for index in range(count):
        expected = precise_charge_jets_strict_serial(
            charges[index],
            separations[index],
            velocities[index],
            accelerations[index],
            jerks[index],
            frames[index] if has_frame[index] else None,
            snaps[index] if has_snap[index] else None,
        )
        for batch, scalar in zip(actual[:3], expected[:3]):
            assert batch[index].tobytes() == scalar.tobytes()
        assert actual[3][index].tobytes() == np.float64(expected[5]).tobytes()


def test_empty_batch_keeps_output_shapes():
    vectors = np.empty((0, 3))
    mask = np.zeros(0, dtype=bool)
    results = resolved_charge_jets_strict_serial(
        np.empty(0), vectors, vectors, vectors, vectors, vectors, vectors, mask, mask
    )
    assert [item.shape for item in results] == [
        (0, 4, 21),
        (0, 3, 21),
        (0, 3, 21),
        (0,),
    ]


def test_fused_ballistic_projection_matches_python_frame_by_bytes():
    rng = np.random.default_rng(487)
    count = 6
    velocities = rng.normal(size=(count, 3))
    velocities[0] = 0
    velocities[1] *= 1e10
    components = rng.normal(size=(count, 4))
    separations = rng.normal(size=(count, 3))
    zeros = np.zeros((count, 3))
    mask = np.ones(count, dtype=bool)
    charges = np.linspace(-2, 2, count)
    actual = resolved_charge_jets_strict_serial(
        charges,
        separations,
        velocities,
        zeros,
        zeros,
        zeros,
        zeros,
        mask,
        mask,
        components,
        mask,
    )
    for index in range(count):
        frame = resolved_separation_frame(
            components[index, 0], components[index, 1:], velocities[index]
        )
        expected = precise_charge_jets_strict_serial(
            charges[index],
            separations[index],
            velocities[index],
            zeros[index],
            zeros[index],
            frame,
            zeros[index],
        )
        for batch, scalar in zip(actual[:3], expected[:3]):
            assert batch[index].tobytes() == scalar.tobytes()
        assert actual[3][index].tobytes() == np.float64(expected[5]).tobytes()


def test_pair_workers_retain_scalar_bits_and_worker_mask():
    from numba import get_num_threads, set_num_threads

    from core.resolved_charge_batch import resolved_charge_jets_strict

    rng = np.random.default_rng(491)
    count = 128
    charges = rng.normal(size=count)
    vectors = rng.normal(size=(6, count, 3))
    vectors[0] *= 10
    vectors[2] *= 1e-5
    vectors[3] *= 1e-7
    vectors[5] *= 1e-9
    vectors[1, 0] = [0, 0, 1e8]
    vectors[2, 0] = vectors[3, 0] = vectors[5, 0] = 0
    flags = np.arange(count) % 2 == 0
    ballistic = rng.normal(size=(count, 4))
    arguments = (charges, *vectors, flags, ~flags, ballistic, ~flags)
    expected = resolved_charge_jets_strict_serial(*arguments)
    previous = get_num_threads()
    try:
        for workers in (1, 2):
            set_num_threads(workers)
            for actual, reference in zip(
                resolved_charge_jets_strict(*arguments), expected
            ):
                assert actual.tobytes() == reference.tobytes()
            assert get_num_threads() == workers
        # Different invalid rows must report the same first source's error,
        # irrespective of which worker encounters an error first.
        bad_arguments = tuple(value.copy() for value in arguments)
        bad_arguments[1][0] = 0
        bad_arguments[0][3] = np.nan
        for workers in (1, 2):
            set_num_threads(workers)
            with pytest.raises(ValueError, match="observer cannot coincide"):
                resolved_charge_jets_strict(*bad_arguments)
            assert get_num_threads() == workers
    finally:
        set_num_threads(previous)


def test_observer_batch_and_response_reductions_match_across_workers():
    from dataclasses import fields

    from numba import get_num_threads, set_num_threads

    from core.retarded_fields import (
        ObserverEvent,
        RetardedHistoryError,
        evaluate_retarded_charge_fields_native,
        evaluate_retarded_charge_response_gradient_native,
    )
    from tests.unit.test_exact_retarded_charge_numba import _uniform_charge_history

    # Replicate signed sources so both the multi-observer batch and a single
    # response cross the worker threshold, with cancellation in source sums.
    history = [
        {key: np.tile(values, 32) for key, values in state.items()}
        for state in _uniform_charge_history()
    ]
    events = tuple(ObserverEvent(t, (1.2, 0.5, -0.4)) for t in (0, 0.001, 0.002))
    previous = get_num_threads()
    results = []
    try:
        for workers in (1, 2):
            set_num_threads(workers)
            results.append(
                (
                    *evaluate_retarded_charge_fields_native(
                        history, events, backend="numba_full_strict_serial"
                    ),
                    evaluate_retarded_charge_response_gradient_native(
                        history,
                        events[0],
                        observer_four_velocity_mm_ns=(300, 10, 0, 0),
                    ),
                )
            )
            # The earlier missing event wins over a later coincident source,
            # as in the original ordered per-event provider.
            with pytest.raises(RetardedHistoryError, match="does not bracket"):
                evaluate_retarded_charge_fields_native(
                    history,
                    (ObserverEvent(-1, (1, 0, 0)), ObserverEvent(0, (0, 0, 0))),
                    backend="numba_full_strict_serial",
                )
        for reference, actual in zip(*results):
            for field in fields(reference):
                expected = getattr(reference, field.name)
                value = getattr(actual, field.name)
                if isinstance(expected, np.ndarray):
                    assert value.tobytes() == expected.tobytes(), field.name
                else:
                    assert value == expected, field.name
    finally:
        set_num_threads(previous)
