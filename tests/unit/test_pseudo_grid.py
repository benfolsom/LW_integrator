from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.pseudo_grid import (
    PairReuseTracker,
    accumulate_effective_source_charges,
    build_passive_neighbor_map,
    compute_causal_history_start_index,
    select_active_indices,
    update_activation_history,
)


def _make_state(
    *,
    x: list[float],
    y: list[float] | None = None,
    z: list[float] | None = None,
    q: list[float] | None = None,
) -> dict[str, np.ndarray]:
    y_vals = y if y is not None else [0.0] * len(x)
    z_vals = z if z is not None else [0.0] * len(x)
    q_vals = q if q is not None else [1.0] * len(x)
    return {
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y_vals, dtype=float),
        "z": np.asarray(z_vals, dtype=float),
        "q": np.asarray(q_vals, dtype=float),
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
