"""Many-source superposition, pair parity, and atomic checkpointed stepping."""

import copy
import json

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.full_dipole_history import FullDipoleHistory
from core.momentum_center_pair import (
    CombinedDipoleProvider,
    FullDipoleProvider,
    MomentumCenterParticle,
    _providers,
    advance_pair,
    dynamics_native,
    initial_state_native,
    initialize_pair,
)
from core.momentum_center_particles import (
    advance_particles,
    configure_pseudogrid_midpoint,
    configure_pseudogrid_reference,
    initialize_particles,
)
from lw_integrator.pair_history_options import configure_run_history


def fixture(
    count=3,
    reaction="off",
    preserved=False,
    method="rk4",
    radius_mm=0.6,
    startup_ct_mm=1.0,
    past_ct_mm=4.0,
    angles=None,
):
    particles, states, histories = [], [], []
    time = np.linspace(-past_ct_mm, 0, 41) / c

    def zero(time, position):
        return np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4, 4))

    zero.taylor_response_length_time = lambda events: tuple(
        np.zeros((5,) + shape) for shape in ((4,), (4, 4), (4, 4), (4, 4, 4))
    )

    for index in range(count):
        angle = index * 2 * np.pi / count if angles is None else angles[index]
        position = np.array([radius_mm * np.cos(angle), radius_mm * np.sin(angle), 0])
        particle = MomentumCenterParticle(
            0.003 * c, 1, reaction_mode=reaction, reaction_derivative_method="analytic"
        )
        state = initial_state_native(
            np.r_[0, position], [1, 0, 0, 0], c * np.array([0, 0, 0.01]), particle, zero
        )
        _, diagnostic = dynamics_native(state, particle, zero)
        histories.append(
            FullDipoleHistory(
                time,
                np.broadcast_to(position, (len(time), 3)),
                np.zeros((len(time), 3)),
                np.broadcast_to(diagnostic["proper_dipole_native"], (len(time), 4, 4)),
                c,
                position_tolerance=1e-8,
                startup_fit="one_sided",
                inertial_until=0.0,
            ).completed()
        )
        particles.append(particle)
        states.append(state)
    options = dict(
        startup_duration_ns=startup_ct_mm / c,
        startup_smoothness=7,
        inertial_prehistory=True,
        preserve_source_increments=preserved,
        integration_method=method,
    )
    return particles, states, histories, options


@pytest.mark.parametrize("preserved,method", [(False, "rk4"), (True, "dop853")])
def test_pair_exact_parity(preserved, method):
    particles, states, histories, options = fixture(
        2, preserved=preserved, method=method
    )
    pair = initialize_pair(particles, states, histories, **options)
    many = initialize_particles(particles, states, histories, **options)
    expected, pair_records = advance_pair(pair, 0.01 / c)
    actual, many_records = advance_particles(many, 0.01 / c)
    actual["model"] = expected["model"]
    if preserved:
        actual["source_precision"]["format"] = expected["source_precision"]["format"]
    assert actual == expected
    for left, right in zip(pair_records[0]["particles"], many_records[0]["particles"]):
        np.testing.assert_array_equal(
            left["proper_dipole_native"], right["proper_dipole_native"]
        )
    with pytest.raises(ValueError, match="Wrong nonlinear pair"):
        advance_pair(many, 0.01 / c)


def test_sum_and_analytical_derivatives_exclude_self():
    particles, _, histories, options = fixture()
    payload = {"startup_duration_ns": 0, "startup_smoothness": 7}
    providers = _providers(payload, particles, histories)
    combined = providers[0]
    assert isinstance(combined, CombinedDipoleProvider)
    assert [p.history for p in combined.providers] == histories[1:]
    position = np.array([0.7, 0.2, 0.3])
    events = np.zeros((5, 4))
    events[0, 1:] = position
    events[1, 0] = 1
    for evaluate in (
        lambda p: p(0, position),
        lambda p: p.taylor_response_length_time(events),
    ):
        first, second = [
            evaluate(FullDipoleProvider(h, particles[i].charge_native))
            for i, h in enumerate(histories)
            if i != 0
        ]
        for actual, a, b in zip(evaluate(combined), first, second):
            np.testing.assert_array_equal(actual, a + b)


def test_three_particle_resume_schedule_and_no_mutation():
    particles, states, histories, options = fixture(preserved=True, method="dop853")
    initial = initialize_particles(particles, states, histories, **options)
    initial = configure_run_history(initial, "auto", None, None, 0.01 / c)
    assert len(initial["dipole_budget_policy"]["absolute_budgets_native"]) == 3
    exact, _ = advance_particles(initial, 0.01 / c, 2)
    scheduled = configure_pseudogrid_reference(initial, 1)
    original = copy.deepcopy(scheduled)
    whole, records = advance_particles(scheduled, 0.01 / c, 2)
    first, _ = advance_particles(scheduled, 0.01 / c)
    restarted, _ = advance_particles(json.loads(json.dumps(first)), 0.01 / c)
    assert whole == restarted
    assert scheduled == original
    without_schedule = copy.deepcopy(whole)
    del without_schedule["pseudogrid_reference"]
    assert without_schedule == exact
    assert (
        records[0]["pseudogrid_reference"]["selected_indices"]
        != records[1]["pseudogrid_reference"]["selected_indices"]
    )
    assert all(
        r["pseudogrid_reference"]["approximate_particle_count"] == 0 for r in records
    )
    assert np.linalg.norm(np.asarray(exact["states"])[:, 5:8]) > 0


def test_three_particle_failure_atomic_and_unequal_times_rejected():
    from core.nonlinear_pair_adaptive import initialize_adaptive

    particles, states, histories, options = fixture()
    initial = initialize_particles(particles, states, histories, **options)
    with pytest.raises(ValueError, match="two-particle checkpoint"):
        initialize_adaptive(initial, None, 0.01 / c)
    original = copy.deepcopy(initial)
    with pytest.raises(ValueError, match="outside published history"):
        advance_particles(initial, 5 / c)
    assert initial == original
    invalid = copy.deepcopy(initial)
    invalid["states"][2][0] = 1
    with pytest.raises(ValueError, match="unequal accepted times"):
        advance_particles(invalid, 0.01 / c)
    invalid = copy.deepcopy(initial)
    invalid["pseudo_grid"] = {"enabled": True}
    with pytest.raises(ValueError, match="Legacy reduced pseudogrid"):
        advance_particles(invalid, 0.01 / c)


def test_three_particle_full_reaction_records():
    particles, states, histories, options = fixture(reaction="full_dipole_coupled")
    initial = initialize_particles(particles, states, histories, **options)
    result, records = advance_particles(initial, 0.001 / c)
    assert len(result["dipole_reaction_ledger"]) == 3
    assert len(records[0]["particles"]) == 3
    for row in result["dipole_reaction_ledger"]:
        assert np.isfinite(row["applied_impulse_native"]).all()
        assert set(row["sectors"]) == {"q_squared", "qD", "D_squared"}


def test_many_particle_cli_checkpoint_and_resume(tmp_path):
    from lw_integrator.nonlinear_particles import main, capabilities

    particles, states, histories, options = fixture(preserved=True, method="dop853")
    initial = initialize_particles(particles, states, histories, **options)
    source, first, second = [
        tmp_path / name for name in ("input.json", "first.json", "second.json")
    ]
    source.write_text(json.dumps(initial))
    assert (
        main(
            [
                "--checkpoint",
                str(source),
                "--output",
                str(first),
                "--step-ns",
                str(0.01 / c),
                "--pseudogrid-reference-active-count",
                "1",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--checkpoint",
                str(first),
                "--output",
                str(second),
                "--step-ns",
                str(0.01 / c),
            ]
        )
        == 0
    )
    result = json.loads(second.read_text())
    assert result["accepted_steps"] == 2
    assert len(result["states"]) == 3
    assert sum(result["pseudogrid_reference"]["activation_count"]) == 2
    assert json.loads(source.read_text()) == initial
    assert capabilities()["pseudo_grid_supported"] is False


@pytest.mark.parametrize("invalid", [0, 4, True, 1.5])
def test_bad_reference_count_rejected(invalid):
    particles, states, histories, options = fixture()
    initial = initialize_particles(particles, states, histories, **options)
    with pytest.raises(ValueError, match="Active count"):
        configure_pseudogrid_reference(initial, invalid)


def test_four_particles_permutation_equivalence():
    particles, states, histories, options = fixture(4)
    order = [2, 0, 3, 1]
    initial = initialize_particles(particles, states, histories, **options)
    permuted = initialize_particles(
        [particles[i] for i in order],
        [states[i] for i in order],
        [histories[i] for i in order],
        **options,
    )
    result, _ = advance_particles(initial, 0.02 / c)
    reordered, _ = advance_particles(permuted, 0.02 / c)
    np.testing.assert_allclose(
        np.asarray(result["states"])[order],
        reordered["states"],
        rtol=2e-14,
        atol=1e-24,
    )


def test_three_particle_internal_adaptation_and_absolute_budgets():
    from core.pair_step_error import InternalStepSettings
    from core.step_doubling import ErrorScale

    particles, states, histories, options = fixture(preserved=True, method="dop853")
    options["internal_step_settings"] = InternalStepSettings(
        *[ErrorScale(1e-8, 1e-6) for _ in range(6)]
    )
    initial = initialize_particles(particles, states, histories, **options)
    initial = configure_run_history(initial, "auto", 1e-10, None, 0.01 / c)
    assert initial["dipole_budget_policy"]["absolute_budgets_native"] == [1e-10] * 3
    result, records = advance_particles(initial, 0.01 / c)
    assert all(
        d["internal_step_control"]["accepted_steps"] >= 1
        for d in records[0]["particles"]
    )
    assert all(s[0] == 0.01 / c for s in result["states"])


def test_bad_reference_checkpoint_is_rejected_without_mutation():
    particles, states, histories, options = fixture()
    initial = configure_pseudogrid_reference(
        initialize_particles(particles, states, histories, **options), 2
    )
    initial["pseudogrid_reference"]["activation_count"][2] = -1
    original = copy.deepcopy(initial)
    with pytest.raises(ValueError, match="schedule counters"):
        advance_particles(initial, 0.01 / c)
    assert initial == original


def test_reduced_midpoint_schedule_restarts_and_keeps_causal_sources():
    particles, states, histories, options = fixture()
    initial = initialize_particles(particles, states, histories, **options)
    reduced = configure_pseudogrid_midpoint(initial, 1, 0.01 / c)
    original = copy.deepcopy(reduced)
    whole, records = advance_particles(reduced, 0.01 / c, 3)
    first, _ = advance_particles(reduced, 0.01 / c)
    second, _ = advance_particles(json.loads(json.dumps(first)), 0.01 / c, 2)
    assert whole == second
    assert reduced == original
    assert len({r["pseudogrid_reduced"]["selected_indices"][0] for r in records}) == 3
    assert all(
        r["pseudogrid_reduced"]["approximate_particle_count"] == 2 for r in records
    )
    assert all(row["evolved_count"] == 0 for row in records[0]["source_sampling"])
    assert all(len(h["time"]) == len(histories[0].time) + 3 for h in whole["histories"])
    assert all(row[0] == whole["states"][0][0] for row in whole["states"])
    exact, _ = advance_particles(initial, 0.01 / c, 3)
    assert not np.array_equal(whole["states"], exact["states"])


def test_reduced_midpoint_rejects_unsupported_physics_and_bad_checkpoint():
    particles, states, histories, options = fixture()
    initial = initialize_particles(particles, states, histories, **options)
    with pytest.raises(ValueError, match="Reduced active count"):
        configure_pseudogrid_midpoint(initial, 3, 0.01 / c)
    preserved_particles, preserved_states, preserved_histories, preserved_options = (
        fixture(preserved=True, method="dop853")
    )
    preserved = initialize_particles(
        preserved_particles, preserved_states, preserved_histories, **preserved_options
    )
    with pytest.raises(ValueError, match="reaction-off, unpreserved RK4"):
        configure_pseudogrid_midpoint(preserved, 1, 0.01 / c)
    reacting_particles, reacting_states, reacting_histories, reacting_options = fixture(
        reaction="full_dipole_coupled"
    )
    reacting = initialize_particles(
        reacting_particles, reacting_states, reacting_histories, **reacting_options
    )
    with pytest.raises(ValueError, match="reaction-off, unpreserved RK4"):
        configure_pseudogrid_midpoint(reacting, 1, 0.01 / c)
    reduced = configure_pseudogrid_midpoint(initial, 1, 0.01 / c)
    reduced["pseudogrid_reduced"]["activation_count"][0] = -1
    original = copy.deepcopy(reduced)
    with pytest.raises(ValueError, match="schedule counters"):
        advance_particles(reduced, 0.01 / c)
    assert reduced == original


def test_reduced_selection_clock_holds_subset_and_restarts():
    particles, states, histories, options = fixture()
    initial = initialize_particles(particles, states, histories, **options)
    scheduled = configure_pseudogrid_midpoint(initial, 1, 0.02 / c)
    whole, records = advance_particles(scheduled, 0.01 / c, 3)
    first, _ = advance_particles(scheduled, 0.01 / c)
    restarted, _ = advance_particles(json.loads(json.dumps(first)), 0.01 / c, 2)
    assert whole == restarted
    assert (
        records[0]["pseudogrid_reduced"]["selected_indices"]
        == records[1]["pseudogrid_reduced"]["selected_indices"]
    )
    assert whole["pseudogrid_reduced"]["selection_count"] == 2
    assert sum(whole["pseudogrid_reduced"]["activation_count"]) == 2
    with pytest.raises(ValueError, match="align with recording clock"):
        advance_particles(scheduled, 0.012 / c)
    with pytest.raises(ValueError, match="selection spacing"):
        configure_pseudogrid_midpoint(initial, 1, None)


def test_reduced_midpoint_matches_short_postarrival_reference():
    particles, states, histories, options = fixture(
        radius_mm=0.06, startup_ct_mm=0.05, past_ct_mm=0.4
    )
    initial = initialize_particles(particles, states, histories, **options)
    exact, _ = advance_particles(
        configure_pseudogrid_reference(initial, 1, 0.005 / c), 0.005 / c, 25
    )
    reduced, records = advance_particles(
        configure_pseudogrid_midpoint(initial, 1, 0.005 / c), 0.005 / c, 25
    )
    assert any(
        sample["evolved_count"] > 0
        for record in records
        for sample in record["source_sampling"]
    )
    exact_states = np.asarray(exact["states"])
    reduced_states = np.asarray(reduced["states"])
    assert np.linalg.norm(exact_states[:, 1:4] - reduced_states[:, 1:4]) < 1e-6
    assert (
        np.linalg.norm(exact_states[:, 5:8] - reduced_states[:, 5:8])
        / np.linalg.norm(exact_states[:, 5:8])
        < 1e-4
    )
    assert (
        max(
            np.linalg.norm(data["length_time"]["spin_constraint"])
            for record in records
            for data in record["particles"]
        )
        < 1e-8
    )
    assert (
        max(
            abs(data["length_time"]["mass_constraint"])
            for record in records
            for data in record["particles"]
        )
        < 1e-5
    )


def test_reduced_midpoint_cli_checkpoint_and_resume(tmp_path):
    from lw_integrator.nonlinear_particles import main

    particles, states, histories, options = fixture()
    initial = initialize_particles(particles, states, histories, **options)
    source, first, second = [
        tmp_path / name for name in ("input.json", "first.json", "second.json")
    ]
    source.write_text(json.dumps(initial))
    assert (
        main(
            [
                "--checkpoint",
                str(source),
                "--output",
                str(first),
                "--step-ns",
                str(0.01 / c),
                "--history-method",
                "preserve",
                "--pseudogrid-midpoint-active-count",
                "1",
                "--pseudogrid-selection-spacing-ns",
                str(0.01 / c),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--checkpoint",
                str(first),
                "--output",
                str(second),
                "--step-ns",
                str(0.01 / c),
                "--history-method",
                "preserve",
            ]
        )
        == 0
    )
    result = json.loads(second.read_text())
    assert result["accepted_steps"] == 2
    assert sum(result["pseudogrid_reduced"]["activation_count"]) == 2
    assert json.loads(source.read_text()) == initial
