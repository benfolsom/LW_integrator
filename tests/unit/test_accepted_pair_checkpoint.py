from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.causal_c5_dipole_provider import AcceptedPairCausalC5SourceHistory
from core.integration_checkpoint import (
    AcceptedPairCheckpointStore,
    CheckpointCompatibilityError,
    CheckpointError,
)
from core.types import GrowableTrajectoryBuilder, TrajectoryArrays


def _state(step: int, *, offset: float) -> dict[str, np.ndarray]:
    value = offset + float(step)
    return {
        "x": np.array([value]),
        "y": np.array([0.5 * value]),
        "z": np.array([0.0]),
        "t": np.array([0.01 * step]),
        "Px": np.array([0.1 * value]),
        "Py": np.array([0.0]),
        "Pz": np.array([0.0]),
        "Pt": np.array([1.0]),
        "gamma": np.array([1.0]),
        "bx": np.array([0.0]),
        "by": np.array([0.0]),
        "bz": np.array([0.0]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "spin_x": np.array([0.0]),
        "spin_y": np.array([0.0]),
        "spin_z": np.array([1.0]),
        "medina_external_force_sample_time": np.array(
            [np.nan if step == 0 else 0.01 * (step - 0.5)]
        ),
        "q": np.array([1.0]),
        "m": np.array([1.0]),
        "magnetic_moment_native": np.array([1.0e-6]),
        "magnetic_dipole_active": np.array([1.0]),
    }


def _history(offset: float, count: int) -> GrowableTrajectoryBuilder:
    builder = GrowableTrajectoryBuilder(2, 1, magnetic_dipole=True)
    for step in range(count):
        builder.append_step(_state(step, offset=offset))
    return builder


def _assert_same(left: TrajectoryArrays, right: TrajectoryArrays) -> None:
    for name in (
        "x",
        "y",
        "z",
        "t",
        "Px",
        "Py",
        "Pz",
        "Pt",
        "gamma",
        "spin_x",
        "spin_y",
        "spin_z",
        "medina_external_force_sample_time",
        "q",
        "m",
        "magnetic_moment_native",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(left, name)),
            np.asarray(getattr(right, name)),
        )


def test_variable_length_pair_checkpoint_round_trip(tmp_path: Path) -> None:
    rider = _history(10.0, 2)
    driver = _history(-10.0, 2)
    directory = tmp_path / "pair.checkpoint"
    store = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "exact-retarded-pair"},
        interval_knots=2,
        interval_seconds=0.0,
        resume=False,
    )
    store.write(
        rider=rider.build_current(),
        driver=driver.build_current(),
        controller_state={"current_lab_step_ns": 0.01, "rejections": 1},
        public_output_state={"next_sample_time_ns": 0.02},
    )
    for step in range(2, 7):
        rider.append_step(_state(step, offset=10.0))
        driver.append_step(_state(step, offset=-10.0))
    store.write(
        rider=rider.build_current(),
        driver=driver.build_current(),
        controller_state={"current_lab_step_ns": 0.04, "rejections": 2},
        public_output_state={"next_sample_time_ns": 0.08},
        intrinsic_spin_reduction_state={
            "schema_version": 1,
            "rider": {"proper_times_ns": [0.0, 0.1]},
            "driver": {"proper_times_ns": [0.0, 0.05]},
        },
        complete=True,
    )

    reopened = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "exact-retarded-pair"},
        interval_knots=2,
        interval_seconds=0.0,
        resume=True,
    )
    restored_rider = GrowableTrajectoryBuilder(1, 1, magnetic_dipole=True)
    restored_driver = GrowableTrajectoryBuilder(1, 1, magnetic_dipole=True)
    reopened.restore_pair(restored_rider, restored_driver)

    assert reopened.manifest["schema_version"] == 4
    assert reopened.committed_knots == 7
    assert reopened.controller_state == {
        "current_lab_step_ns": 0.04,
        "rejections": 2,
    }
    assert reopened.public_output_state == {"next_sample_time_ns": 0.08}
    reduction_state = reopened.intrinsic_spin_reduction_state
    assert reduction_state == {
        "schema_version": 1,
        "rider": {"proper_times_ns": [0.0, 0.1]},
        "driver": {"proper_times_ns": [0.0, 0.05]},
    }
    assert reduction_state is not None
    reduction_state["rider"]["proper_times_ns"].append(99.0)
    assert reopened.intrinsic_spin_reduction_state != reduction_state
    _assert_same(restored_rider.build_current(), rider.build_current())
    _assert_same(restored_driver.build_current(), driver.build_current())


def test_pair_checkpoint_appends_and_restores_frozen_c5_coefficients(
    tmp_path: Path,
) -> None:
    rider = _history(10.0, 18)
    driver = _history(-10.0, 18)
    accepted_c5 = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        rider.build_current(),
        driver.build_current(),
    )
    directory = tmp_path / "pair-c5.checkpoint"
    store = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "causal-c5-pair"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    store.write(
        rider=rider.build_current(),
        driver=driver.build_current(),
        controller_state={},
        public_output_state={},
        causal_c5_source_history=accepted_c5,
    )
    for step in range(18, 24):
        rider_state = _state(step, offset=10.0)
        driver_state = _state(step, offset=-10.0)
        rider.append_step(rider_state)
        driver.append_step(driver_state)
        accepted_c5 = AcceptedPairCausalC5SourceHistory(
            rider=accepted_c5.rider.append_accepted_state(rider_state),
            driver=accepted_c5.driver.append_accepted_state(driver_state),
        )
    store.write(
        rider=rider.build_current(),
        driver=driver.build_current(),
        controller_state={},
        public_output_state={},
        causal_c5_source_history=accepted_c5,
        complete=True,
    )

    reopened = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "causal-c5-pair"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=True,
    )
    restored_rider = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
    restored_driver = GrowableTrajectoryBuilder(8, 1, magnetic_dipole=True)
    reopened.restore_pair(restored_rider, restored_driver)
    restored_c5 = reopened.restore_causal_c5_source_history(
        restored_rider.build_current(),
        restored_driver.build_current(),
    )

    assert restored_c5 is not None
    assert reopened.causal_c5_source_history_metadata is not None
    assert reopened.causal_c5_source_history_metadata["schema_version"] == 3
    for restored_collection, expected_collection in (
        (restored_c5.rider, accepted_c5.rider),
        (restored_c5.driver, accepted_c5.driver),
    ):
        assert (
            restored_collection.source_identities
            == expected_collection.source_identities
        )
        for restored_source, expected_source in zip(
            restored_collection.sources,
            expected_collection.sources,
        ):
            assert len(restored_source.history.frozen_segments) == len(
                expected_source.history.frozen_segments
            )
            for restored_segment, expected_segment in zip(
                restored_source.history.frozen_segments,
                expected_source.history.frozen_segments,
            ):
                np.testing.assert_array_equal(
                    restored_segment.position_coefficients_mm,
                    expected_segment.position_coefficients_mm,
                )
                np.testing.assert_array_equal(
                    restored_segment.rest_spin_stereographic_coefficients,
                    expected_segment.rest_spin_stereographic_coefficients,
                )


def test_pair_checkpoint_cannot_drop_active_c5_history(tmp_path: Path) -> None:
    rider = _history(1.0, 18)
    driver = _history(-1.0, 18)
    c5_state = AcceptedPairCausalC5SourceHistory.from_trajectory_arrays(
        rider.build_current(),
        driver.build_current(),
    )
    store = AcceptedPairCheckpointStore(
        tmp_path / "pair-c5.checkpoint",
        compatibility_payload={"physics": "causal-c5-pair"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    store.write(
        rider=rider.build_current(),
        driver=driver.build_current(),
        controller_state={},
        public_output_state={},
        causal_c5_source_history=c5_state,
    )
    rider.append_step(_state(18, offset=1.0))
    driver.append_step(_state(18, offset=-1.0))
    manifest_before = json.loads(store.manifest_path.read_text(encoding="utf-8"))

    with pytest.raises(CheckpointError, match="cannot omit an active causal C5"):
        store.write(
            rider=rider.build_current(),
            driver=driver.build_current(),
            controller_state={},
            public_output_state={},
        )

    assert (
        json.loads(store.manifest_path.read_text(encoding="utf-8")) == manifest_before
    )


def test_pair_checkpoint_rejects_mismatched_history_lengths(tmp_path: Path) -> None:
    store = AcceptedPairCheckpointStore(
        tmp_path / "pair.checkpoint",
        compatibility_payload={"physics": "pair"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    with pytest.raises(CheckpointError, match="equal knot counts"):
        store.write(
            rider=_history(0.0, 2).build_current(),
            driver=_history(0.0, 1).build_current(),
            controller_state={},
            public_output_state={},
        )


def test_pair_checkpoint_rejects_incompatible_resume(tmp_path: Path) -> None:
    directory = tmp_path / "pair.checkpoint"
    store = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "one"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    history = _history(0.0, 1).build_current()
    store.write(
        rider=history,
        driver=history,
        controller_state={},
        public_output_state={},
    )
    with pytest.raises(CheckpointCompatibilityError, match="fingerprint"):
        AcceptedPairCheckpointStore(
            directory,
            compatibility_payload={"physics": "two"},
            interval_knots=1,
            interval_seconds=0.0,
            resume=True,
        )


def test_pair_checkpoint_detects_tampered_chunk(tmp_path: Path) -> None:
    directory = tmp_path / "pair.checkpoint"
    store = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "pair"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    history = _history(0.0, 1).build_current()
    store.write(
        rider=history,
        driver=history,
        controller_state={},
        public_output_state={},
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    chunk = directory / manifest["chunks"][0]["file"]
    chunk.write_bytes(chunk.read_bytes() + b"corrupt")

    reopened = AcceptedPairCheckpointStore(
        directory,
        compatibility_payload={"physics": "pair"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=True,
    )
    with pytest.raises(CheckpointError, match="hash mismatch"):
        reopened.restore_pair(
            GrowableTrajectoryBuilder(1, 1, magnetic_dipole=True),
            GrowableTrajectoryBuilder(1, 1, magnetic_dipole=True),
        )


def test_pair_checkpoint_rejects_changed_particle_constants(tmp_path: Path) -> None:
    store = AcceptedPairCheckpointStore(
        tmp_path / "pair.checkpoint",
        compatibility_payload={"physics": "pair"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    rider = _history(0.0, 1)
    driver = _history(0.0, 1)
    store.write(
        rider=rider.build_current(),
        driver=driver.build_current(),
        controller_state={},
        public_output_state={},
    )
    rider.append_step(_state(1, offset=0.0))
    driver.append_step(_state(1, offset=0.0))
    changed_rider = replace(rider.build_current(), q=np.array([2.0]))

    with pytest.raises(CheckpointCompatibilityError, match="constant q changed"):
        store.write(
            rider=changed_rider,
            driver=driver.build_current(),
            controller_state={},
            public_output_state={},
        )


def test_pair_checkpoint_json_failure_does_not_advance_manifest(
    tmp_path: Path,
) -> None:
    store = AcceptedPairCheckpointStore(
        tmp_path / "pair.checkpoint",
        compatibility_payload={"physics": "pair"},
        interval_knots=1,
        interval_seconds=0.0,
        resume=False,
    )
    rider = _history(0.0, 1).build_current()
    driver = _history(0.0, 1).build_current()
    manifest_before = json.loads(store.manifest_path.read_text(encoding="utf-8"))

    with pytest.raises(CheckpointError, match="finite JSON-compatible"):
        store.write(
            rider=rider,
            driver=driver,
            controller_state={"error": np.nan},
            public_output_state={},
        )

    assert store.committed_knots == 0
    assert (
        json.loads(store.manifest_path.read_text(encoding="utf-8")) == manifest_before
    )
