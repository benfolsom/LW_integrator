from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.pseudo_grid import (
    PairReuseTracker,
    PassiveNeighborMap,
    accumulate_effective_source_charges,
    build_passive_neighbor_map,
    build_pseudo_grid_step_schedule,
    build_self_excluded_space_charge_source_charges,
    commit_pseudo_grid_step_schedule,
    compute_causal_history_start_index,
    initialize_pseudo_grid_planner_state,
    record_pseudo_grid_history_times,
    reconstruct_full_state_from_active_result,
    select_active_indices,
    slice_particle_state,
    slice_trajectory_particle_history,
    update_activation_history,
)
from core.types import PseudoGridConfig


def _make_state(
    *,
    x: list[float],
    y: list[float] | None = None,
    z: list[float] | None = None,
    q: list[float] | None = None,
    t: list[float] | None = None,
    gamma: list[float] | None = None,
) -> dict[str, np.ndarray]:
    y_vals = y if y is not None else [0.0] * len(x)
    z_vals = z if z is not None else [0.0] * len(x)
    q_vals = q if q is not None else [1.0] * len(x)
    t_vals = t if t is not None else [0.0] * len(x)
    gamma_vals = gamma if gamma is not None else [1.0] * len(x)
    return {
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y_vals, dtype=float),
        "z": np.asarray(z_vals, dtype=float),
        "q": np.asarray(q_vals, dtype=float),
        "t": np.asarray(t_vals, dtype=float),
        "gamma": np.asarray(gamma_vals, dtype=float),
    }


def _make_solver_state(
    *,
    x: list[float],
    y: list[float] | None = None,
    z: list[float] | None = None,
    q: list[float] | None = None,
    t: list[float] | None = None,
    gamma: list[float] | None = None,
    bx: list[float] | None = None,
    by: list[float] | None = None,
    bz: list[float] | None = None,
    beta_avg_x: list[float] | None = None,
    beta_avg_y: list[float] | None = None,
    beta_avg_z: list[float] | None = None,
    beta_samples: list[float] | None = None,
    dead: list[bool] | None = None,
) -> dict[str, np.ndarray]:
    n_particles = len(x)
    y_vals = y if y is not None else [0.0] * n_particles
    z_vals = z if z is not None else [0.0] * n_particles
    q_vals = q if q is not None else [1.0] * n_particles
    t_vals = t if t is not None else [0.0] * n_particles
    gamma_vals = gamma if gamma is not None else [1.0] * n_particles
    bx_vals = bx if bx is not None else [0.0] * n_particles
    by_vals = by if by is not None else [0.0] * n_particles
    bz_vals = bz if bz is not None else [0.0] * n_particles
    beta_avg_x_vals = beta_avg_x if beta_avg_x is not None else list(bx_vals)
    beta_avg_y_vals = beta_avg_y if beta_avg_y is not None else list(by_vals)
    beta_avg_z_vals = beta_avg_z if beta_avg_z is not None else list(bz_vals)
    beta_samples_vals = (
        beta_samples if beta_samples is not None else [1.0] * n_particles
    )
    dead_vals = dead if dead is not None else [False] * n_particles
    zeros = np.zeros(n_particles, dtype=float)
    return {
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y_vals, dtype=float),
        "z": np.asarray(z_vals, dtype=float),
        "t": np.asarray(t_vals, dtype=float),
        "Px": zeros.copy(),
        "Py": zeros.copy(),
        "Pz": zeros.copy(),
        "Pt": np.asarray(gamma_vals, dtype=float),
        "gamma": np.asarray(gamma_vals, dtype=float),
        "bx": np.asarray(bx_vals, dtype=float),
        "by": np.asarray(by_vals, dtype=float),
        "bz": np.asarray(bz_vals, dtype=float),
        "bdotx": zeros.copy(),
        "bdoty": zeros.copy(),
        "bdotz": zeros.copy(),
        "q": np.asarray(q_vals, dtype=float),
        "m": np.ones(n_particles, dtype=float),
        "char_time": np.ones(n_particles, dtype=float) * 1.0e-3,
        "origin_x": np.asarray(x, dtype=float),
        "origin_y": np.asarray(y_vals, dtype=float),
        "origin_z": np.asarray(z_vals, dtype=float),
        "beta_avg_x": np.asarray(beta_avg_x_vals, dtype=float),
        "beta_avg_y": np.asarray(beta_avg_y_vals, dtype=float),
        "beta_avg_z": np.asarray(beta_avg_z_vals, dtype=float),
        "beta_samples": np.asarray(beta_samples_vals, dtype=float),
        "radiation_power": zeros.copy(),
        "radiation_energy": zeros.copy(),
        "radiation_energy_applied": zeros.copy(),
        "_dead_particles": np.asarray(dead_vals, dtype=bool),
    }


def test_select_active_indices_returns_endpoints_for_even_line_distribution():
    state = _make_state(x=[0.0, 1.0, 9.0, 10.0])

    active = select_active_indices(
        state,
        np.array([0, 1, 2, 3], dtype=int),
        active_count=2,
        step_index=10,
    )

    np.testing.assert_array_equal(active, np.array([0, 3], dtype=int))


def test_select_active_indices_prefers_stale_particles_when_count_is_one():
    state = _make_state(x=[0.0, 5.0, 10.0])
    last_active_step = np.array([19, 5, 0], dtype=int)
    activation_count = np.array([4, 1, 0], dtype=int)

    active = select_active_indices(
        state,
        np.array([0, 1, 2], dtype=int),
        active_count=1,
        step_index=20,
        last_active_step=last_active_step,
        activation_count=activation_count,
    )

    np.testing.assert_array_equal(active, np.array([2], dtype=int))


def test_update_activation_history_updates_last_seen_and_counts_in_place():
    last_active_step = np.full(5, -1, dtype=int)
    activation_count = np.zeros(5, dtype=int)

    update_activation_history(
        last_active_step,
        activation_count,
        np.array([1, 3], dtype=int),
        step_index=12,
    )

    np.testing.assert_array_equal(last_active_step, np.array([-1, 12, -1, 12, -1]))
    np.testing.assert_array_equal(activation_count, np.array([0, 1, 0, 1, 0]))


def test_build_passive_neighbor_map_uses_inverse_distance_weights():
    state = _make_state(x=[0.0, 10.0, 20.0])

    neighbor_map = build_passive_neighbor_map(
        state,
        np.array([0, 1, 2], dtype=int),
        np.array([0, 2], dtype=int),
        neighbor_count=2,
        weighting_mode="inverse_distance",
    )

    np.testing.assert_array_equal(
        neighbor_map.passive_indices, np.array([1], dtype=int)
    )
    np.testing.assert_array_equal(
        neighbor_map.neighbor_particle_indices, np.array([[0, 2]], dtype=int)
    )
    np.testing.assert_allclose(neighbor_map.weights, np.array([[0.5, 0.5]]))


def test_build_passive_neighbor_map_nearest_mode_assigns_full_weight_to_first_neighbor():
    state = _make_state(x=[0.0, 1.0, 20.0])

    neighbor_map = build_passive_neighbor_map(
        state,
        np.array([0, 1, 2], dtype=int),
        np.array([0, 2], dtype=int),
        neighbor_count=2,
        weighting_mode="nearest",
    )

    np.testing.assert_array_equal(
        neighbor_map.neighbor_particle_indices, np.array([[0, 2]], dtype=int)
    )
    np.testing.assert_allclose(neighbor_map.weights, np.array([[1.0, 0.0]]))


def test_accumulate_effective_source_charges_adds_passive_charge_to_active_set():
    state = _make_state(x=[0.0, 10.0, 20.0], q=[1.0, 1.0, 1.0])
    neighbor_map = build_passive_neighbor_map(
        state,
        np.array([0, 1, 2], dtype=int),
        np.array([0, 2], dtype=int),
        neighbor_count=2,
        weighting_mode="inverse_distance",
    )

    effective = accumulate_effective_source_charges(
        state,
        np.array([0, 2], dtype=int),
        neighbor_map,
    )

    np.testing.assert_allclose(effective, np.array([1.5, 1.5]))


def test_build_self_excluded_space_charge_source_charges_preserves_non_self_sources():
    state = _make_state(x=[0.0, 10.0, 20.0], q=[1.0, 1.0, 1.0])
    neighbor_map = build_passive_neighbor_map(
        state,
        np.array([0, 1, 2], dtype=int),
        np.array([0, 2], dtype=int),
        neighbor_count=2,
        weighting_mode="inverse_distance",
    )

    charge_matrix = build_self_excluded_space_charge_source_charges(
        state,
        np.array([0, 2], dtype=int),
        neighbor_map,
    )

    np.testing.assert_allclose(charge_matrix, np.array([[0.0, 2.0], [2.0, 0.0]]))


def test_build_self_excluded_space_charge_source_charges_falls_back_to_other_actives_when_only_self_anchor_is_listed():
    state = _make_state(x=[0.0, 1.0, 20.0], q=[1.0, 1.0, 1.0])
    neighbor_map = build_passive_neighbor_map(
        state,
        np.array([0, 1, 2], dtype=int),
        np.array([0, 2], dtype=int),
        neighbor_count=1,
        weighting_mode="nearest",
    )

    charge_matrix = build_self_excluded_space_charge_source_charges(
        state,
        np.array([0, 2], dtype=int),
        neighbor_map,
        weighting_mode="nearest",
    )

    np.testing.assert_allclose(charge_matrix, np.array([[0.0, 2.0], [2.0, 0.0]]))


def test_build_self_excluded_space_charge_source_charges_handles_asymmetric_three_active_distribution():
    state = _make_state(x=[0.0, 4.0, 10.0, 20.0], q=[1.0, 4.0, 2.0, 3.0])
    neighbor_map = build_passive_neighbor_map(
        state,
        np.array([0, 1, 2, 3], dtype=int),
        np.array([0, 2, 3], dtype=int),
        neighbor_count=3,
        weighting_mode="inverse_distance",
    )

    charge_matrix = build_self_excluded_space_charge_source_charges(
        state,
        np.array([0, 2, 3], dtype=int),
        neighbor_map,
        weighting_mode="inverse_distance",
    )

    np.testing.assert_allclose(
        charge_matrix,
        np.array(
            [
                [0.0, 54.0 / 11.0, 45.0 / 11.0],
                [21.0 / 5.0, 0.0, 19.0 / 5.0],
                [17.0 / 5.0, 18.0 / 5.0, 0.0],
            ],
            dtype=float,
        ),
    )


def test_pair_reuse_tracker_penalizes_recent_pairs_and_prunes_old_entries():
    tracker = PairReuseTracker(window_steps=3)
    tracker.note_matches(
        np.array([1, 2], dtype=int),
        np.array([5], dtype=int),
        step_index=10,
    )

    assert tracker.stored_pair_count == 2
    assert tracker.penalty(1, 5, step_index=10) == pytest.approx(1.0)
    assert tracker.penalty(1, 5, step_index=11) == pytest.approx(2.0 / 3.0)

    tracker.prune(step_index=13)

    assert tracker.stored_pair_count == 0
    assert tracker.penalty(1, 5, step_index=13) == pytest.approx(0.0)


def test_compute_causal_history_start_index_uses_light_cone_cutoff_and_margin():
    source_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

    start_index = compute_causal_history_start_index(
        source_times,
        current_observer_time_ns=4.5,
        max_separation_mm=1.5 * C_MMNS,
        safety_margin_steps=1,
    )

    assert start_index == 2


def test_compute_causal_history_start_index_rejects_non_monotonic_time_history():
    with pytest.raises(ValueError, match="monotonically"):
        compute_causal_history_start_index(
            np.array([0.0, 2.0, 1.0], dtype=float),
            current_observer_time_ns=3.0,
            max_separation_mm=C_MMNS,
        )


def test_slice_trajectory_particle_history_applies_q_override_and_remaps_failure_info():
    trajectory = [
        _make_solver_state(x=[0.0, 10.0, 20.0], q=[1.0, 2.0, 3.0], t=[0.0, 0.0, 0.0]),
        _make_solver_state(x=[1.0, 11.0, 21.0], q=[1.0, 2.0, 3.0], t=[1.0, 1.0, 1.0]),
    ]
    trajectory[0]["_particle_failure_info"] = {2: {"reason": "test"}}

    sliced = slice_trajectory_particle_history(
        trajectory,
        np.array([0, 2], dtype=int),
        q_override=np.array([5.0, 7.0], dtype=float),
    )

    assert len(sliced) == 2
    np.testing.assert_array_equal(sliced[0]["x"], np.array([0.0, 20.0]))
    np.testing.assert_array_equal(sliced[1]["t"], np.array([1.0, 1.0]))
    np.testing.assert_array_equal(sliced[0]["q"], np.array([5.0, 7.0]))
    np.testing.assert_array_equal(sliced[1]["q"], np.array([5.0, 7.0]))
    assert sliced[0]["_particle_failure_info"] == {1: {"reason": "test"}}


def test_reconstruct_full_state_from_active_result_applies_weighted_deltas_and_updates_beta_average():
    previous_full_state = _make_solver_state(
        x=[0.0, 10.0, 20.0],
        q=[1.0, 1.0, 1.0],
        bx=[0.1, 0.2, 0.3],
        beta_avg_x=[0.1, 0.2, 0.3],
        beta_samples=[1.0, 2.0, 3.0],
    )
    active_indices = np.array([0, 2], dtype=int)
    active_result_state = slice_particle_state(previous_full_state, active_indices)
    active_result_state["x"] = np.array([1.0, 22.0], dtype=float)
    active_result_state["t"] = np.array([1.0, 1.0], dtype=float)
    active_result_state["bx"] = np.array([0.2, 0.5], dtype=float)
    active_result_state["gamma"] = np.array([1.1, 1.2], dtype=float)
    active_result_state["radiation_power"] = np.array([0.4, 0.8], dtype=float)
    active_result_state["_dead_particles"] = np.array([False, False], dtype=bool)
    passive_map = PassiveNeighborMap(
        passive_indices=np.array([1], dtype=int),
        neighbor_particle_indices=np.array([[0, 2]], dtype=int),
        weights=np.array([[0.25, 0.75]], dtype=float),
    )

    reconstructed = reconstruct_full_state_from_active_result(
        previous_full_state,
        active_indices,
        active_result_state,
        passive_map,
    )

    np.testing.assert_allclose(reconstructed["x"], np.array([1.0, 11.75, 22.0]))
    np.testing.assert_allclose(reconstructed["t"], np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(reconstructed["bx"], np.array([0.2, 0.375, 0.5]))
    np.testing.assert_allclose(
        reconstructed["radiation_power"], np.array([0.4, 0.7, 0.8])
    )
    assert reconstructed["q"][1] == pytest.approx(1.0)
    assert reconstructed["beta_samples"][1] == pytest.approx(3.0)
    assert reconstructed["beta_avg_x"][1] == pytest.approx((0.2 * 2.0 + 0.375) / 3.0)


def test_reconstruct_full_state_from_active_result_ignores_dead_anchors_when_enabled():
    previous_full_state = _make_solver_state(
        x=[0.0, 10.0, 20.0],
        q=[1.0, 1.0, 1.0],
        bx=[0.1, 0.2, 0.3],
        beta_avg_x=[0.1, 0.2, 0.3],
        beta_samples=[1.0, 2.0, 3.0],
    )
    active_indices = np.array([0, 2], dtype=int)
    active_result_state = slice_particle_state(previous_full_state, active_indices)
    active_result_state["x"] = np.array([1.0, 40.0], dtype=float)
    active_result_state["bx"] = np.array([0.2, 0.9], dtype=float)
    active_result_state["q"] = np.array([1.0, 0.0], dtype=float)
    active_result_state["_dead_particles"] = np.array([False, True], dtype=bool)
    passive_map = PassiveNeighborMap(
        passive_indices=np.array([1], dtype=int),
        neighbor_particle_indices=np.array([[0, 2]], dtype=int),
        weights=np.array([[0.25, 0.75]], dtype=float),
    )

    reconstructed = reconstruct_full_state_from_active_result(
        previous_full_state,
        active_indices,
        active_result_state,
        passive_map,
        loss_tracking_enabled=True,
    )

    assert reconstructed["x"][1] == pytest.approx(11.0)
    assert reconstructed["bx"][1] == pytest.approx(0.3)
    assert reconstructed["_dead_particles"][2]
    assert reconstructed["q"][2] == pytest.approx(0.0)


def test_build_pseudo_grid_step_schedule_collects_active_passive_metadata():
    rider_state = _make_state(
        x=[0.0, 10.0, 20.0],
        q=[1.0, 1.0, 1.0],
        t=[0.0, 0.0, 0.0],
    )
    driver_state = _make_state(
        x=[1.0, 11.0, 21.0],
        q=[2.0, 2.0, 2.0],
        t=[0.0, 0.0, 0.0],
    )
    planner_state = initialize_pseudo_grid_planner_state(
        rider_particle_count=3,
        driver_particle_count=3,
        pair_reuse_window=4,
    )
    record_pseudo_grid_history_times(planner_state, rider_state, driver_state)

    schedule = build_pseudo_grid_step_schedule(
        rider_state,
        driver_state,
        step_index=1,
        config=PseudoGridConfig(
            enabled=True,
            active_rider_count=2,
            active_driver_count=2,
            passive_neighbor_count=2,
            causal_history_pruning_enabled=True,
            causal_history_safety_margin_steps=0,
        ),
        planner_state=planner_state,
    )

    np.testing.assert_array_equal(
        schedule.rider_active_indices, np.array([0, 2], dtype=int)
    )
    np.testing.assert_array_equal(
        schedule.driver_active_indices, np.array([0, 2], dtype=int)
    )
    np.testing.assert_array_equal(
        schedule.rider_passive_map.passive_indices, np.array([1], dtype=int)
    )
    np.testing.assert_array_equal(
        schedule.driver_passive_map.passive_indices, np.array([1], dtype=int)
    )
    np.testing.assert_allclose(
        schedule.rider_passive_map.weights, np.array([[0.5, 0.5]])
    )
    np.testing.assert_allclose(
        schedule.driver_passive_map.weights, np.array([[0.5, 0.5]])
    )
    np.testing.assert_allclose(
        schedule.rider_effective_source_charges, np.array([1.5, 1.5])
    )
    np.testing.assert_allclose(
        schedule.driver_effective_source_charges, np.array([3.0, 3.0])
    )
    np.testing.assert_allclose(schedule.pair_reuse_penalties, np.zeros((2, 2)))
    assert schedule.driver_history_start_index == 0
    assert schedule.rider_history_start_index == 0
    assert schedule.max_cross_bunch_separation_mm > 0.0


def test_commit_pseudo_grid_step_schedule_updates_planner_state_and_pair_tracker():
    rider_state = _make_state(x=[0.0, 10.0, 20.0], t=[0.0, 0.0, 0.0])
    driver_state = _make_state(x=[1.0, 11.0, 21.0], t=[0.0, 0.0, 0.0])
    planner_state = initialize_pseudo_grid_planner_state(
        rider_particle_count=3,
        driver_particle_count=3,
        pair_reuse_window=4,
    )
    record_pseudo_grid_history_times(planner_state, rider_state, driver_state)
    schedule = build_pseudo_grid_step_schedule(
        rider_state,
        driver_state,
        step_index=1,
        config=PseudoGridConfig(
            enabled=True,
            active_rider_count=2,
            active_driver_count=2,
            passive_neighbor_count=2,
        ),
        planner_state=planner_state,
    )

    commit_pseudo_grid_step_schedule(planner_state, schedule)

    np.testing.assert_array_equal(
        planner_state.rider_last_active_step, np.array([1, -1, 1], dtype=int)
    )
    np.testing.assert_array_equal(
        planner_state.driver_last_active_step, np.array([1, -1, 1], dtype=int)
    )
    np.testing.assert_array_equal(
        planner_state.rider_activation_count, np.array([1, 0, 1], dtype=int)
    )
    np.testing.assert_array_equal(
        planner_state.driver_activation_count, np.array([1, 0, 1], dtype=int)
    )
    assert planner_state.pair_reuse_tracker.stored_pair_count == 4
    assert planner_state.pair_reuse_tracker.penalty(
        0, 0, step_index=1
    ) == pytest.approx(1.0)
