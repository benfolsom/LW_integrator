"""Native checkpoint versions preserve resolved knots and load older rows."""

import hashlib
import json

import numpy as np
import pytest

from core.integration_checkpoint import (
    AcceptedPairCheckpointStore,
    IntegrationCheckpointStore,
)
from core.resolved_knot import RESOLVED_KNOT_FIELDS
from core.types import EXACT_SOURCE_REDUCTION_FIELDS, TrajectoryBuilder
from tests.unit.test_accepted_pair_checkpoint import _state

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_resolved_knots_checkpoint_round_trip(tmp_path, pair, legacy):
    builder = TrajectoryBuilder(3, 1)
    for step in range(3):
        state = _state(step, offset=1.0)
        state["exact_source_cell_sigma_mm"] = np.array([0.3])
        state["exact_source_original_count"] = np.array([12.0])
        for index, name in enumerate(RESOLVED_KNOT_FIELDS):
            state[name] = np.array([float(index + step + 1) * 1e-30])
        builder.set_step(step, state)
    trajectory = builder.build()
    directory = tmp_path / "checkpoint"
    compatibility = {"physics": "resolved-knots"}
    if pair:
        store_class = AcceptedPairCheckpointStore
        arguments = dict(
            compatibility_payload=compatibility, interval_knots=1, interval_seconds=0.0
        )
        store = store_class(directory, **arguments, resume=False)
        store.write(
            rider=trajectory,
            driver=trajectory,
            controller_state={},
            public_output_state={},
        )
    else:
        store_class = IntegrationCheckpointStore
        arguments = dict(
            compatibility_payload=compatibility,
            total_steps=3,
            requested_steps=3,
            active_start=0,
            interval_steps=1,
            interval_seconds=0.0,
        )
        store = store_class(directory, **arguments, resume=False)
        store.write(step_index=2, rider=trajectory, driver=trajectory, loop_state={})
    if legacy:
        manifest = json.loads(store.manifest_path.read_text())
        manifest["schema_version"] = 4 if pair else 1
        for chunk in manifest["chunks"]:
            path = directory / chunk["file"]
            with np.load(path) as archive:
                arrays = {
                    name: archive[name]
                    for name in archive.files
                    if name.split("__")[-1] not in RESOLVED_KNOT_FIELDS
                }
            np.savez(path, **arrays)
            chunk["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Older checkpoints have neither resolved knots nor reduction metadata.
        constants = manifest["constants"]
        path = directory / constants["file"]
        with np.load(path) as archive:
            arrays = {
                name: archive[name]
                for name in archive.files
                if name.split("__")[-1] not in EXACT_SOURCE_REDUCTION_FIELDS
            }
        np.savez(path, **arrays)
        constants["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        store.manifest_path.write_text(json.dumps(manifest))
    reopened = store_class(directory, **arguments, resume=True)
    restored = TrajectoryBuilder(3, 1)
    if pair:
        reopened.restore_pair(restored, TrajectoryBuilder(3, 1))
    else:
        reopened.restore_builder(restored, "rider")
    output = restored.build()
    for name in RESOLVED_KNOT_FIELDS:
        np.testing.assert_array_equal(
            getattr(output, name), 0.0 if legacy else getattr(trajectory, name)
        )
    np.testing.assert_array_equal(output.x, trajectory.x)
    for name in EXACT_SOURCE_REDUCTION_FIELDS:
        np.testing.assert_array_equal(
            getattr(output, name), 0.0 if legacy else getattr(trajectory, name)
        )


def test_legacy_empty_channels_work_in_indexed_history():
    from dataclasses import replace
    from core.types import IndexedTrajectoryArrays
    from core.retarded_fields import _prepare_history
    from tests.unit.test_high_gamma_stored_history import _coast

    history, _ = _coast(1e3, steps=12)
    legacy = replace(
        history, **{name: np.empty((0, 0)) for name in RESOLVED_KNOT_FIELDS}
    )
    indexed = IndexedTrajectoryArrays(legacy, np.array([0]))
    assert len(_prepare_history(indexed, ()).sources[0].light_cone_segments) == 11
