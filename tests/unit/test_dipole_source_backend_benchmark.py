from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts import benchmark_exact_retarded_backends as benchmark


@dataclass
class _FakeTrajectory:
    x: np.ndarray
    halted: np.ndarray
    notes: list[str | None]
    mass_shell_projection_energy: np.ndarray
    local_magnetic_field_z_t: np.ndarray | None = None
    _storage_state: object | None = None
    _storage_array_revision: int | None = None


def test_complete_state_comparison_is_bytewise_and_skips_storage_metadata() -> None:
    reference = _FakeTrajectory(
        x=np.asarray((1.0, 2.0), dtype=np.float64),
        halted=np.asarray((False, False)),
        notes=[None, "done"],
        mass_shell_projection_energy=np.zeros(2),
        _storage_state=object(),
        _storage_array_revision=1,
    )
    identical = _FakeTrajectory(
        x=reference.x.copy(),
        halted=reference.halted.copy(),
        notes=list(reference.notes),
        mass_shell_projection_energy=reference.mass_shell_projection_energy.copy(),
        _storage_state=object(),
        _storage_array_revision=99,
    )
    changed = _FakeTrajectory(
        x=np.nextafter(reference.x, np.inf),
        halted=reference.halted.copy(),
        notes=list(reference.notes),
        mass_shell_projection_energy=reference.mass_shell_projection_energy.copy(),
    )

    equal_report = benchmark._compare_trajectories(reference, identical)
    changed_report = benchmark._compare_trajectories(reference, changed)

    assert equal_report["bitwise_equal"] is True
    assert equal_report["compared_array_fields"] == 3
    assert changed_report["bitwise_equal"] is False
    assert changed_report["array_mismatches"] == ["x"]
    assert (
        changed_report["array_mismatch_details"]["x"]["bitwise_mismatch_elements"] == 2
    )
    assert changed_report["tolerance_passed"] is True


def test_complete_state_fingerprint_changes_with_a_side_channel() -> None:
    reference = _FakeTrajectory(
        x=np.asarray((1.0,), dtype=np.float64),
        halted=np.asarray((False,)),
        notes=[None],
        mass_shell_projection_energy=np.zeros(1),
    )
    changed = _FakeTrajectory(
        x=reference.x.copy(),
        halted=reference.halted.copy(),
        notes=["halted"],
        mass_shell_projection_energy=reference.mass_shell_projection_energy.copy(),
    )

    reference_payload = benchmark._trajectory_fingerprint(reference)
    changed_payload = benchmark._trajectory_fingerprint(changed)

    assert reference_payload["array_field_count"] == 3
    assert (
        reference_payload["complete_public_state_sha256"]
        != changed_payload["complete_public_state_sha256"]
    )


def test_projection_difference_uses_physical_energy_budget() -> None:
    reference = _FakeTrajectory(
        x=np.zeros(1),
        halted=np.asarray((False,)),
        notes=[None],
        mass_shell_projection_energy=np.zeros(1),
    )
    candidate = _FakeTrajectory(
        x=reference.x.copy(),
        halted=reference.halted.copy(),
        notes=list(reference.notes),
        mass_shell_projection_energy=np.asarray((1.0e-24,)),
    )

    report = benchmark._compare_trajectories(reference, candidate)
    detail = report["array_mismatch_details"]["mass_shell_projection_energy"]

    assert report["bitwise_equal"] is False
    assert report["tolerance_passed"] is True
    assert detail["cumulative_absolute_difference_mev"] < 0.025

    candidate.mass_shell_projection_energy[0] = 1.0
    over_budget = benchmark._compare_trajectories(reference, candidate)

    assert over_budget["tolerance_passed"] is False

    candidate.mass_shell_projection_energy[0] = np.nan
    nonfinite_mismatch = benchmark._compare_trajectories(reference, candidate)

    assert nonfinite_mismatch["tolerance_passed"] is False


def test_saved_local_magnetic_field_uses_named_absolute_diagnostic_budget() -> None:
    reference = _FakeTrajectory(
        x=np.zeros(1),
        halted=np.asarray((False,)),
        notes=[None],
        mass_shell_projection_energy=np.zeros(1),
        local_magnetic_field_z_t=np.ones(1),
    )
    candidate = _FakeTrajectory(
        x=reference.x.copy(),
        halted=reference.halted.copy(),
        notes=list(reference.notes),
        mass_shell_projection_energy=reference.mass_shell_projection_energy.copy(),
        local_magnetic_field_z_t=np.asarray((1.0 + 9.0e-13,)),
    )

    accepted = benchmark._compare_trajectories(reference, candidate)
    detail = accepted["array_mismatch_details"]["local_magnetic_field_z_t"]

    assert accepted["tolerance_passed"] is True
    assert detail["diagnostic_absolute_tolerance"] == 1.0e-12
    assert detail["diagnostic_relative_tolerance"] == 0.0
    assert detail["diagnostic_units"] == "T"
    assert detail["force_path_validation"] is False

    candidate.local_magnetic_field_z_t[0] = 1.0 + 1.1e-12
    rejected = benchmark._compare_trajectories(reference, candidate)

    assert rejected["tolerance_passed"] is False
