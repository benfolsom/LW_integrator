from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.pseudo_grid import (
    PairReuseTracker,
    PassiveNeighborMap,
    accumulate_effective_source_charges,
    accumulate_field_representative_charges,
    accumulate_field_representative_charges_and_radii,
    build_field_representative_space_charge_source_charges,
    build_hybrid_space_charge_sources,
    build_passive_neighbor_map,
    build_pseudo_grid_step_schedule,
    build_self_excluded_space_charge_source_charges,
    commit_pseudo_grid_step_schedule,
    compute_causal_history_start_index,
    initialize_pseudo_grid_planner_state,
    record_pseudo_grid_history_times,
    reconstruct_full_state_from_active_result,
    select_active_indices,
    select_field_representative_indices,
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


def _reference_self_excluded_charge_matrix(
    state: dict[str, np.ndarray],
    active_indices: np.ndarray,
    passive_map: PassiveNeighborMap,
    *,
    weighting_mode: str = "inverse_distance",
) -> np.ndarray:
    active = np.asarray(active_indices, dtype=int)
    charges = np.asarray(state["q"], dtype=float)
    active_lookup = {int(particle_idx): idx for idx, particle_idx in enumerate(active)}
    charge_matrix = np.tile(charges[active].astype(float), (active.size, 1))
    np.fill_diagonal(charge_matrix, 0.0)
    if passive_map.is_empty:
        return charge_matrix

    passive_indices = np.asarray(passive_map.passive_indices, dtype=int)
    reference_indices = np.unique(np.concatenate((active, passive_indices)))
    reference_coords = np.column_stack(
        (
            np.asarray(state["x"], dtype=float)[reference_indices],
            np.asarray(state["y"], dtype=float)[reference_indices],
            np.asarray(state["z"], dtype=float)[reference_indices],
        )
    )
    centers = np.mean(reference_coords, axis=0)
    spans = np.ptp(reference_coords, axis=0)
    spans = np.where(spans > 0.0, spans, 1.0)

    def normalized(indices: np.ndarray) -> np.ndarray:
        coords = np.column_stack(
            (
                np.asarray(state["x"], dtype=float)[indices],
                np.asarray(state["y"], dtype=float)[indices],
                np.asarray(state["z"], dtype=float)[indices],
            )
        )
        return (coords - centers) / spans

    active_coords = normalized(active)
    passive_coords = normalized(passive_indices)

    def fallback_weights(passive_row_idx: int, excluded_local_idx: int) -> np.ndarray:
        weights = np.zeros(active.size, dtype=float)
        candidate_indices = np.array(
            [idx for idx in range(active.size) if idx != excluded_local_idx],
            dtype=int,
        )
        if candidate_indices.size == 0:
            return weights
        distances = np.linalg.norm(
            active_coords[candidate_indices]
            - passive_coords[passive_row_idx][np.newaxis, :],
            axis=1,
        )
        if weighting_mode == "nearest":
            weights[candidate_indices[0]] = 1.0
        else:
            inv = 1.0 / np.maximum(distances, 1.0e-12)
            weights[candidate_indices] = inv / np.sum(inv)
        return weights

    for passive_row_idx, passive_particle_idx in enumerate(passive_indices):
        passive_charge = float(charges[passive_particle_idx])
        if passive_charge == 0.0:
            continue
        base_weights = np.zeros(active.size, dtype=float)
        for neighbor_particle_idx, weight in zip(
            passive_map.neighbor_particle_indices[passive_row_idx],
            passive_map.weights[passive_row_idx],
        ):
            base_weights[active_lookup[int(neighbor_particle_idx)]] += float(weight)
        base_weight_total = float(np.sum(base_weights))
        if base_weight_total <= 0.0:
            continue
        base_weights /= base_weight_total
        for observer_local_idx in range(active.size):
            observer_weights = base_weights.copy()
            if observer_weights[observer_local_idx] > 0.0:
                observer_weights[observer_local_idx] = 0.0
                redistributed_total = float(np.sum(observer_weights))
                if redistributed_total > 0.0:
                    observer_weights /= redistributed_total
                else:
                    observer_weights = fallback_weights(
                        passive_row_idx,
                        observer_local_idx,
                    )
            charge_matrix[observer_local_idx] += passive_charge * observer_weights
    return charge_matrix


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


def test_select_field_representatives_include_active_and_add_medoids():
    state = _make_state(x=[0.0, 1.0, 5.0, 9.0, 10.0])
    alive = np.array([0, 1, 2, 3, 4], dtype=int)
    active = np.array([1, 3], dtype=int)

    field = select_field_representative_indices(
        state,
        alive,
        active,
        field_count=4,
    )

    np.testing.assert_array_equal(field[:2], active)
    assert field.size == 4
    assert set(field.tolist()).issubset(set(alive.tolist()))
    assert len(set(field.tolist())) == field.size


def test_accumulate_field_representative_charges_conserves_total_charge():
    state = _make_state(x=[0.0, 1.0, 5.0, 9.0, 10.0], q=[1.0, 2.0, 3.0, 4.0, 5.0])
    alive = np.array([0, 1, 2, 3, 4], dtype=int)
    field = np.array([0, 2, 4], dtype=int)

    effective = accumulate_field_representative_charges(
        state,
        alive,
        field,
        neighbor_count=2,
    )

    assert effective.shape == (3,)
    assert np.sum(effective) == pytest.approx(np.sum(state["q"]))
    assert effective[1] >= state["q"][2]


def test_field_representative_sc_excludes_observer_rep_and_redistributes_deposit():
    state = _make_state(x=[0.0, 1.0, 2.0, 3.0], q=[1.0, 2.0, 3.0, 4.0])
    active = np.array([0, 2], dtype=int)
    field = np.array([0, 1, 2], dtype=int)
    field_source_charges = np.array([5.0, 2.0, 3.5], dtype=float)

    charge_matrix = build_field_representative_space_charge_source_charges(
        state,
        active,
        field,
        field_source_charges,
    )

    np.testing.assert_allclose(
        charge_matrix,
        np.array(
            [
                [0.0, 2.0 + 4.0 * 2.0 / 3.0, 3.5 + 4.0 / 3.0],
                [5.0 + 0.5 / 3.0, 2.0 + 0.5 * 2.0 / 3.0, 0.0],
            ]
        ),
    )


def test_field_representative_sc_allows_active_not_in_field_set():
    state = _make_state(x=[0.0, 1.0, 2.0], q=[1.0, 2.0, 3.0])

    charge_matrix = build_field_representative_space_charge_source_charges(
        state,
        np.array([0], dtype=int),
        np.array([1, 2], dtype=int),
        np.array([2.0, 3.0], dtype=float),
    )

    np.testing.assert_allclose(charge_matrix, np.array([[2.0, 3.0]]))


def test_hybrid_space_charge_sources_add_exact_neighbors_without_double_counting():
    state = _make_state(x=[0.0, 1.0, 10.0], q=[1.0, 1.0, 1.0])
    alive = np.array([0, 1, 2], dtype=int)
    active = np.array([0, 2], dtype=int)
    field = np.array([0, 2], dtype=int)
    field_charges = np.array([1.5, 1.5], dtype=float)

    source_indices, charge_matrix, source_radii = build_hybrid_space_charge_sources(
        state,
        alive,
        active,
        field,
        field_charges,
        field_deposition_neighbor_count=2,
        near_neighbor_count=1,
    )

    np.testing.assert_array_equal(source_indices, np.array([0, 2, 1], dtype=int))
    np.testing.assert_allclose(
        charge_matrix,
        np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        ),
    )
    np.testing.assert_allclose(np.sum(charge_matrix, axis=1), np.array([2.0, 2.0]))
    assert source_radii.shape == (3,)
    assert source_radii[0] > 0.0
    assert source_radii[1] > 0.0
    assert source_radii[2] == 0.0


def test_field_representative_charge_radii_preserve_field_order():
    state = _make_state(x=[0.0, 10.0, 1.0], q=[1.0, 10.0, 3.0])
    alive = np.array([0, 1, 2], dtype=int)
    field = np.array([1, 0], dtype=int)

    charges, radii = accumulate_field_representative_charges_and_radii(
        state,
        alive,
        field,
        neighbor_count=1,
        weighting_mode="nearest",
    )

    np.testing.assert_allclose(charges, np.array([10.0, 4.0]))
    np.testing.assert_allclose(radii, np.array([0.0, np.sqrt(0.75)]))


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


def test_build_passive_neighbor_map_allows_passive_intermediates_before_collapsing():
    state = _make_state(x=[0.0, 1.0, 2.0, 3.0, 10.0], q=[1.0] * 5)
    active = np.array([0, 4], dtype=int)

    neighbor_map = build_passive_neighbor_map(
        state,
        np.array([0, 1, 2, 3, 4], dtype=int),
        active,
        neighbor_count=2,
        weighting_mode="inverse_distance",
    )

    np.testing.assert_array_equal(
        neighbor_map.passive_indices,
        np.array([1, 2, 3], dtype=int),
    )
    np.testing.assert_array_equal(
        neighbor_map.neighbor_particle_indices,
        np.array([[0, 4], [0, 4], [0, 4]], dtype=int),
    )
    np.testing.assert_allclose(np.sum(neighbor_map.weights, axis=1), np.ones(3))
    assert neighbor_map.weights[1, 0] > 0.95
    assert neighbor_map.weights[1, 1] < 0.05

    effective = accumulate_effective_source_charges(state, active, neighbor_map)
    assert np.sum(effective) == pytest.approx(5.0)


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


def test_build_self_excluded_space_charge_source_charges_matches_scalar_reference():
    state = _make_state(
        x=[0.0, 1.5, 4.0, 7.0, 8.0, 14.0, 19.0],
        y=[0.0, 0.2, -0.1, 0.4, -0.3, 0.1, 0.0],
        z=[0.0, 0.5, 0.2, -0.4, 0.1, 0.3, -0.2],
        q=[1.0, -0.5, 2.0, 1.5, -1.0, 3.0, 0.75],
    )
    alive = np.arange(7, dtype=int)
    active = np.array([0, 2, 5], dtype=int)

    for weighting_mode, neighbor_count in (("inverse_distance", 2), ("nearest", 1)):
        neighbor_map = build_passive_neighbor_map(
            state,
            alive,
            active,
            neighbor_count=neighbor_count,
            weighting_mode=weighting_mode,
        )
        charge_matrix = build_self_excluded_space_charge_source_charges(
            state,
            active,
            neighbor_map,
            weighting_mode=weighting_mode,
        )
        expected = _reference_self_excluded_charge_matrix(
            state,
            active,
            neighbor_map,
            weighting_mode=weighting_mode,
        )

        np.testing.assert_allclose(charge_matrix, expected)


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


def test_reconstruct_full_state_from_active_result_can_leave_passives_frozen():
    previous_full_state = _make_solver_state(
        x=[0.0, 10.0, 20.0],
        z=[0.0, 5.0, 10.0],
        t=[0.0, 2.0, 4.0],
        bx=[0.1, 0.2, 0.3],
        bz=[0.0, 0.1, 0.0],
    )
    active_indices = np.array([0, 2], dtype=int)
    active_result_state = slice_particle_state(previous_full_state, active_indices)
    active_result_state["x"] = np.array([1.0, 22.0], dtype=float)
    active_result_state["t"] = np.array([1.0, 5.0], dtype=float)
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
        passive_update_mode="frozen",
    )

    np.testing.assert_allclose(reconstructed["x"], np.array([1.0, 10.0, 22.0]))
    np.testing.assert_allclose(reconstructed["z"], np.array([0.0, 5.0, 10.0]))
    np.testing.assert_allclose(reconstructed["t"], np.array([1.0, 2.0, 5.0]))


def test_reconstruct_full_state_invalidates_passive_medina_force_history():
    previous_full_state = _make_solver_state(
        x=[0.0, 10.0, 20.0],
        q=[1.0, 1.0, 1.0],
        t=[2.0, 2.0, 2.0],
    )
    previous_full_state.update(
        {
            "radiation_reaction_work": np.zeros(3),
            "medina_cross_field_energy": np.zeros(3),
            "medina_cross_field_energy_change": np.zeros(3),
            "medina_force_derivative_ready": np.ones(3, dtype=bool),
            "medina_impulse_capped": np.zeros(3, dtype=bool),
            "medina_external_force_x": np.ones(3),
            "medina_external_force_y": np.zeros(3),
            "medina_external_force_z": np.zeros(3),
            "medina_external_force_sample_time": np.full(3, 1.5),
        }
    )
    active_indices = np.array([0, 2], dtype=int)
    active_result_state = slice_particle_state(previous_full_state, active_indices)
    active_result_state["medina_external_force_x"] = np.array([2.0, 3.0])
    active_result_state["medina_external_force_sample_time"] = np.array([2.5, 2.5])
    passive_map = PassiveNeighborMap(
        passive_indices=np.array([1], dtype=int),
        neighbor_particle_indices=np.array([[0, 2]], dtype=int),
        weights=np.array([[0.5, 0.5]], dtype=float),
    )

    reconstructed = reconstruct_full_state_from_active_result(
        previous_full_state,
        active_indices,
        active_result_state,
        passive_map,
        passive_update_mode="frozen",
    )

    np.testing.assert_array_equal(
        reconstructed["medina_external_force_sample_time"][[0, 2]],
        (2.5, 2.5),
    )
    assert np.isnan(reconstructed["medina_external_force_sample_time"][1])
    assert not reconstructed["medina_force_derivative_ready"][1]


def test_reconstruct_full_state_from_active_result_can_coast_passives_ballistically():
    previous_full_state = _make_solver_state(
        x=[0.0, 10.0, 20.0],
        z=[0.0, 5.0, 10.0],
        t=[0.0, 2.0, 4.0],
        gamma=[1.0, 2.0, 3.0],
        bx=[0.1, 0.2, 0.3],
        bz=[0.0, 0.1, 0.0],
    )
    active_indices = np.array([0, 2], dtype=int)
    active_result_state = slice_particle_state(previous_full_state, active_indices)
    active_result_state["x"] = np.array([1.0, 22.0], dtype=float)
    active_result_state["t"] = np.array([1.0, 5.0], dtype=float)
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
        passive_update_mode="ballistic",
        h_step=0.5,
    )

    passive_dt = 2.0 * 0.5
    np.testing.assert_allclose(
        reconstructed["x"],
        np.array([1.0, 10.0 + 0.2 * C_MMNS * passive_dt, 22.0]),
    )
    np.testing.assert_allclose(
        reconstructed["z"],
        np.array([0.0, 5.0 + 0.1 * C_MMNS * passive_dt, 10.0]),
    )
    np.testing.assert_allclose(reconstructed["t"], np.array([1.0, 3.0, 5.0]))


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


def test_build_pseudo_grid_step_schedule_slow_rotates_active_subset():
    rider_state = _make_state(x=[0.0, 1.0, 2.0, 3.0, 4.0])
    driver_state = _make_state(x=[10.0, 11.0, 12.0, 13.0, 14.0])
    planner_state = initialize_pseudo_grid_planner_state(
        rider_particle_count=5,
        driver_particle_count=5,
        pair_reuse_window=4,
    )
    record_pseudo_grid_history_times(planner_state, rider_state, driver_state)
    config = PseudoGridConfig(
        enabled=True,
        active_rider_count=3,
        active_driver_count=3,
        active_selection_mode="slow_rotating_live",
        active_rotation_interval=3,
        active_rotation_fraction=1.0 / 3.0,
    )

    first = build_pseudo_grid_step_schedule(
        rider_state,
        driver_state,
        step_index=1,
        config=config,
        planner_state=planner_state,
    )
    commit_pseudo_grid_step_schedule(planner_state, first)
    second = build_pseudo_grid_step_schedule(
        rider_state,
        driver_state,
        step_index=2,
        config=config,
        planner_state=planner_state,
    )
    commit_pseudo_grid_step_schedule(planner_state, second)
    third = build_pseudo_grid_step_schedule(
        rider_state,
        driver_state,
        step_index=4,
        config=config,
        planner_state=planner_state,
    )

    np.testing.assert_array_equal(
        second.rider_active_indices, first.rider_active_indices
    )
    assert third.rider_active_indices.size == first.rider_active_indices.size
    assert len(set(third.rider_active_indices) & set(first.rider_active_indices)) >= 2
    assert len(set(third.rider_active_indices) - set(first.rider_active_indices)) == 1


def test_build_pseudo_grid_step_schedule_reports_passive_remap_thresholds():
    rider_state = _make_state(x=[0.0, 0.1, 10.0, 11.0], y=[0.0, 0.0, 0.0, 0.0])
    driver_state = _make_state(x=[0.0, 0.1, 10.0, 11.0], y=[0.0, 0.0, 0.0, 0.0])
    planner_state = initialize_pseudo_grid_planner_state(
        rider_particle_count=4,
        driver_particle_count=4,
        pair_reuse_window=4,
    )
    record_pseudo_grid_history_times(planner_state, rider_state, driver_state)
    config = PseudoGridConfig(
        enabled=True,
        active_rider_count=2,
        active_driver_count=2,
        active_selection_mode="fixed_prefix",
        passive_remap_warning_sigma=0.1,
        passive_remap_trigger_sigma=0.2,
    )

    schedule = build_pseudo_grid_step_schedule(
        rider_state,
        driver_state,
        step_index=5,
        config=config,
        planner_state=planner_state,
    )

    assert schedule.rider_role_diagnostics["passive_remap_warning"] == pytest.approx(
        1.0
    )
    assert schedule.rider_role_diagnostics["passive_remap_trigger"] == pytest.approx(
        1.0
    )
    assert schedule.rider_role_diagnostics["passive_centroid_sigma"] > 0.2


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
    np.testing.assert_array_equal(
        schedule.rider_field_indices, schedule.rider_active_indices
    )
    np.testing.assert_array_equal(
        schedule.driver_field_indices, schedule.driver_active_indices
    )
    np.testing.assert_allclose(
        schedule.rider_field_source_charges, np.array([1.5, 1.5])
    )
    np.testing.assert_allclose(
        schedule.driver_field_source_charges, np.array([3.0, 3.0])
    )
    np.testing.assert_allclose(schedule.pair_reuse_penalties, np.zeros((2, 2)))
    assert schedule.driver_history_start_index == 0
    assert schedule.rider_history_start_index == 0
    assert schedule.max_cross_bunch_separation_mm > 0.0


def test_build_pseudo_grid_step_schedule_supports_extra_field_representatives():
    rider_state = _make_state(x=[0.0, 1.0, 5.0, 9.0, 10.0], q=[1.0] * 5)
    driver_state = _make_state(x=[0.0, 2.0, 4.0, 6.0, 8.0], q=[2.0] * 5)
    planner_state = initialize_pseudo_grid_planner_state(
        rider_particle_count=5,
        driver_particle_count=5,
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
            field_rider_count=4,
            field_driver_count=4,
            field_deposition_neighbor_count=2,
            passive_neighbor_count=2,
        ),
        planner_state=planner_state,
    )

    assert schedule.rider_active_indices.size == 2
    assert schedule.driver_active_indices.size == 2
    assert schedule.rider_field_indices.size == 4
    assert schedule.driver_field_indices.size == 4
    assert set(schedule.rider_active_indices.tolist()).issubset(
        set(schedule.rider_field_indices.tolist())
    )
    assert set(schedule.driver_active_indices.tolist()).issubset(
        set(schedule.driver_field_indices.tolist())
    )
    assert np.sum(schedule.rider_field_source_charges) == pytest.approx(5.0)
    assert np.sum(schedule.driver_field_source_charges) == pytest.approx(10.0)


def test_build_pseudo_grid_step_schedule_advances_causal_history_start_indices():
    rider_state = _make_state(
        x=[0.0, 10.0, 20.0],
        q=[1.0, 1.0, 1.0],
        t=[1.0, 1.0, 1.0],
    )
    driver_state = _make_state(
        x=[1.0, 11.0, 21.0],
        q=[2.0, 2.0, 2.0],
        t=[1.0, 1.0, 1.0],
    )
    planner_state = initialize_pseudo_grid_planner_state(
        rider_particle_count=3,
        driver_particle_count=3,
        pair_reuse_window=4,
    )
    planner_state.rider_history_times_ns = [0.0, 0.5, 1.0]
    planner_state.driver_history_times_ns = [0.0, 0.5, 1.0]

    schedule = build_pseudo_grid_step_schedule(
        rider_state,
        driver_state,
        step_index=3,
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

    assert schedule.driver_history_start_index == 2
    assert schedule.rider_history_start_index == 2


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


def test_charge_localization_stats_detects_localization():
    """The charge-localization helper flags catastrophic charge concentration."""
    from core.integration_runner import _charge_localization_stats

    # Even spread across 4 field reps -> low max fraction, low Gini.
    even = _charge_localization_stats(np.array([1.0, 1.0, 1.0, 1.0]))
    assert even is not None
    assert even["max_anchor_fraction"] == pytest.approx(0.25)
    assert even["gini"] == pytest.approx(0.0, abs=1e-9)

    # Catastrophic localization: 97% of charge on one anchor.
    localized = _charge_localization_stats(np.array([97.0, 1.0, 1.0, 1.0]))
    assert localized is not None
    assert localized["max_anchor_fraction"] == pytest.approx(0.97)
    assert localized["gini"] > 0.7

    signed = _charge_localization_stats(np.array([-2.0, -1.0, -1.0]))
    assert signed is not None
    assert signed["max_anchor_fraction"] == pytest.approx(0.5)

    # Empty / zero-charge returns None.
    assert _charge_localization_stats(np.array([])) is None
    assert _charge_localization_stats(np.array([0.0, 0.0])) is None
    assert _charge_localization_stats(None) is None
