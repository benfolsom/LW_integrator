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
