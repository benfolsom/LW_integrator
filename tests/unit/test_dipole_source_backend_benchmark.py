from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts import benchmark_dipole_source_backends as benchmark


@dataclass
class _FakeTrajectory:
    x: np.ndarray
    halted: np.ndarray
    notes: list[str | None]
    _storage_state: object | None = None
    _storage_array_revision: int | None = None


def test_complete_state_comparison_is_bytewise_and_skips_storage_metadata() -> None:
    reference = _FakeTrajectory(
        x=np.asarray((1.0, 2.0), dtype=np.float64),
        halted=np.asarray((False, False)),
        notes=[None, "done"],
        _storage_state=object(),
        _storage_array_revision=1,
    )
    identical = _FakeTrajectory(
        x=reference.x.copy(),
        halted=reference.halted.copy(),
        notes=list(reference.notes),
        _storage_state=object(),
        _storage_array_revision=99,
    )
    changed = _FakeTrajectory(
        x=np.nextafter(reference.x, np.inf),
        halted=reference.halted.copy(),
        notes=list(reference.notes),
    )

    equal_report = benchmark._compare_trajectories(reference, identical)
    changed_report = benchmark._compare_trajectories(reference, changed)

    assert equal_report["bitwise_equal"] is True
    assert equal_report["compared_array_fields"] == 2
    assert changed_report["bitwise_equal"] is False
    assert changed_report["array_mismatches"] == ["x"]
    assert (
        changed_report["array_mismatch_details"]["x"]["bitwise_mismatch_elements"] == 2
    )


def test_complete_state_fingerprint_changes_with_a_side_channel() -> None:
    reference = _FakeTrajectory(
        x=np.asarray((1.0,), dtype=np.float64),
        halted=np.asarray((False,)),
        notes=[None],
    )
    changed = _FakeTrajectory(
        x=reference.x.copy(),
        halted=reference.halted.copy(),
        notes=["halted"],
    )

    reference_payload = benchmark._trajectory_fingerprint(reference)
    changed_payload = benchmark._trajectory_fingerprint(changed)

    assert reference_payload["array_field_count"] == 2
    assert (
        reference_payload["complete_public_state_sha256"]
        != changed_payload["complete_public_state_sha256"]
    )
