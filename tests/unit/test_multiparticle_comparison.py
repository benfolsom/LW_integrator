"""Reject misleading comparisons and check the spatial-momentum denominator."""

import json
import copy

import pytest

from scripts.compare_multiparticle_postarrival import compare
from scripts.compare_multiparticle_postarrival import compare_history_overlap


def outputs(tmp_path):
    paths = [tmp_path / "coarse", tmp_path / "fine"]
    for path in paths:
        path.mkdir()
        (path / "initial.json").write_text(json.dumps({"model": "same"}))
        (path / "summary.json").write_text(
            json.dumps(
                dict(
                    status="complete",
                    postarrival_steps=5,
                    reaction="off",
                    fit_spacing_ns=1.0,
                    end_ct_mm=2.5,
                    time_ns=0.1,
                    kinetic_momentum_native=[[1e9, 1.0, 0, 0]],
                )
            )
        )
        (path / "checkpoint.json").write_text(
            json.dumps(
                dict(
                    states=[[0.1, 1, 0, 0, 1e9, 1, 0, 0, 0.1, 0, 0, 0, 0, 0]],
                )
            )
        )
    return paths


def change(path, file, **changes):
    target = path / file
    value = json.loads(target.read_text())
    value.update(changes)
    target.write_text(json.dumps(value))


def test_momentum_denominator_excludes_rest_energy(tmp_path):
    coarse, fine = outputs(tmp_path)
    change(coarse, "summary.json", kinetic_momentum_native=[[1e9, 1.01, 0, 0]])
    result = compare(coarse, fine)
    assert result["kinetic_spatial_momentum_native"]["relative"] == pytest.approx(0.01)
    assert not result["passed"]


@pytest.mark.parametrize(
    "change_args",
    [
        {"status": "failed"},
        {"postarrival_steps": 0},
        {"reaction": "full_dipole_coupled"},
        {"fit_spacing_ns": 2.0},
        {"time_ns": 0.2},
    ],
)
def test_incompatible_or_incomplete_runs_rejected(tmp_path, change_args):
    coarse, fine = outputs(tmp_path)
    change(coarse, "summary.json", **change_args)
    with pytest.raises(ValueError):
        compare(coarse, fine)


def test_different_initial_conditions_rejected(tmp_path):
    coarse, fine = outputs(tmp_path)
    change(coarse, "initial.json", model="different")
    with pytest.raises(ValueError, match="Initial particles"):
        compare(coarse, fine)


def test_identical_results_pass(tmp_path):
    assert compare(*outputs(tmp_path))["passed"]


def mixed_outputs(tmp_path):
    paths = outputs(tmp_path)
    for path in paths:
        change(
            path,
            "summary.json",
            integration_method="rk4",
            rk3_active_count=1,
            selection_spacing_ns=0.05,
            selection_trace=[
                dict(time_ns=0.0, selected_indices=[0]),
                dict(time_ns=0.05, selected_indices=[0]),
            ],
        )
    return paths


def test_mixed_comparison_checks_physical_selection_schedule(tmp_path):
    coarse, fine = mixed_outputs(tmp_path)
    change(coarse, "summary.json", step_ns=0.05)
    change(fine, "summary.json", step_ns=0.025)
    report = compare(coarse, fine)
    assert report["passed"]
    assert report["selection_schedule"]["event_count"] == 2


@pytest.mark.parametrize(
    "changes",
    [
        dict(integration_method="dop853"),
        dict(rk3_active_count=2),
        dict(selection_spacing_ns=0.025),
        dict(selection_trace=[]),
        dict(selection_trace=[dict(time_ns=0.0, selected_indices=[0])]),
        dict(
            selection_trace=[
                dict(time_ns=0.0, selected_indices=[0]),
                dict(time_ns=0.025, selected_indices=[0]),
            ]
        ),
        dict(
            selection_trace=[
                dict(time_ns=0.0, selected_indices=[0]),
                dict(time_ns=0.05, selected_indices=[1]),
            ]
        ),
    ],
)
def test_mixed_comparison_rejects_different_selection_schedules(tmp_path, changes):
    coarse, fine = mixed_outputs(tmp_path)
    change(fine, "summary.json", **changes)
    with pytest.raises(ValueError):
        compare(coarse, fine)


def test_matching_but_truncated_selection_traces_rejected(tmp_path):
    paths = mixed_outputs(tmp_path)
    for path in paths:
        change(
            path,
            "summary.json",
            selection_trace=[dict(time_ns=0.0, selected_indices=[0])],
        )
    with pytest.raises(ValueError, match="cover the completed interval"):
        compare(*paths)


def test_reaction_sector_failure_cannot_hide_in_total(tmp_path):
    coarse, fine = outputs(tmp_path)
    ledger = dict(
        applied_impulse_native=[1, 0, 0, 0],
        applied_torque_native=[0] * 6,
        sectors={
            sector: dict(impulse_native=[1, 0, 0, 0], torque_native=[0] * 6)
            for sector in ("q_squared", "qD", "D_squared")
        },
    )
    for path in (coarse, fine):
        change(path, "summary.json", reaction="full_dipole_coupled")
        change(path, "checkpoint.json", dipole_reaction_ledger=[ledger])
    assert compare(coarse, fine)["passed"]
    bad = copy.deepcopy(ledger)
    bad["sectors"]["D_squared"]["impulse_native"][0] = 1.1
    change(coarse, "checkpoint.json", dipole_reaction_ledger=[bad])
    report = compare(coarse, fine)
    assert report["checks"]["total_impulse_native"]
    assert not report["checks"]["D_squared_impulse_native"]
    assert not report["passed"]


def test_partial_history_matches_times_and_does_not_claim_completion(tmp_path):
    coarse, fine = outputs(tmp_path)
    for path, times in (
        (coarse, [0, 0.1, 0.2, 0.3]),
        (fine, [0, 0.05, 0.1, 0.15, 0.2]),
    ):
        change(path, "summary.json", status="failed", first_arrival_time_ns=0.1)
        history = dict(
            time=times,
            position=[[t, 0, 0] for t in times],
            velocity=[[1, 0, 0] for t in times],
            dipole_high=[[[t]] for t in times],
            dipole_low=[[[0]] for t in times],
        )
        change(path, "checkpoint.json", histories=[history])
    result = compare_history_overlap(coarse, fine)
    assert result["particle_sample_count"] == 2
    assert result["time_range_ns"] == [0.1, 0.2]
    assert "passed" not in result
    assert all(row["absolute"] == 0 for row in result["differences"].values())
    change(fine, "summary.json", first_arrival_time_ns=None)
    with pytest.raises(ValueError, match="evolved source signals"):
        compare_history_overlap(coarse, fine)
