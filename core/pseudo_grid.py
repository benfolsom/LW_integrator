"""Scheduling and bookkeeping helpers for the experimental pseudo-grid mode.

The reduced active/passive solver path is still under development, but these
helpers provide the bounded-memory pieces that the public configuration surface
already refers to:

- active-subset selection with coverage and recency bias;
- passive-particle nearest-neighbour anchors and weights;
- effective source-charge aggregation from passive to active particles;
- bounded recent-pair tracking without an ``O(N^2)`` history matrix;
- conservative causal-history retention bounds.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree

from .constants import C_MMNS
from .particle_status import get_alive_particle_indices
from .types import ParticleState, PseudoGridConfig, Trajectory


@dataclass(slots=True)
class PassiveNeighborMap:
    """Passive-to-active nearest-neighbour assignments for one bunch step."""

    passive_indices: np.ndarray
    neighbor_particle_indices: np.ndarray
    weights: np.ndarray

    @property
    def is_empty(self) -> bool:
        return self.passive_indices.size == 0


@dataclass(slots=True)
class PairReuseTracker:
    """Track only recently used active rider/driver pairs.

    This keeps bounded recent pair history without constructing a dense
    particle-pair matrix over the full bunches.
    """

    window_steps: int
    _last_seen: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.window_steps < 0:
            raise ValueError("pair-reuse window must be non-negative")

    @property
    def stored_pair_count(self) -> int:
        return len(self._last_seen)

    def prune(self, step_index: int) -> None:
        """Drop pair entries older than the configured window."""
        if self.window_steps == 0:
            self._last_seen.clear()
            return

        cutoff = int(step_index) - self.window_steps
        stale_keys = [
            key for key, last_step in self._last_seen.items() if last_step <= cutoff
        ]
        for key in stale_keys:
            del self._last_seen[key]

    def note_matches(
        self,
        rider_active_indices: np.ndarray,
        driver_active_indices: np.ndarray,
        step_index: int,
    ) -> None:
        """Record the active rider/driver pairs used on one integration step."""
        self.prune(step_index)
        for rider_idx in np.asarray(rider_active_indices, dtype=int).tolist():
            for driver_idx in np.asarray(driver_active_indices, dtype=int).tolist():
                self._last_seen[(int(rider_idx), int(driver_idx))] = int(step_index)

    def penalty(self, rider_idx: int, driver_idx: int, step_index: int) -> float:
        """Return a recency penalty in ``[0, 1]`` for a candidate pair."""
        if self.window_steps == 0:
            return 0.0

        last_step = self._last_seen.get((int(rider_idx), int(driver_idx)))
        if last_step is None:
            return 0.0

        age = int(step_index) - last_step
        if age < 0 or age >= self.window_steps:
            return 0.0
        return float(self.window_steps - age) / float(self.window_steps)


@dataclass(slots=True)
class PseudoGridPlannerState:
    """Mutable cross-step state for pseudo-grid schedule planning."""

    rider_last_active_step: np.ndarray
    rider_activation_count: np.ndarray
    driver_last_active_step: np.ndarray
    driver_activation_count: np.ndarray
    pair_reuse_tracker: PairReuseTracker
    rider_history_times_ns: list[float] = field(default_factory=list)
    driver_history_times_ns: list[float] = field(default_factory=list)


PSEUDO_GRID_PASSIVE_DELTA_FIELDS: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "t",
    "Px",
    "Py",
    "Pz",
    "Pt",
    "gamma",
    "bx",
    "by",
    "bz",
    "bdotx",
    "bdoty",
    "bdotz",
    "radiation_power",
    "radiation_energy",
    "radiation_energy_applied",
)


@dataclass(slots=True, frozen=True)
class PseudoGridStepSchedule:
    """Per-step pseudo-grid schedule snapshot for one outer B2B step.

    ``driver_history_start_index`` is the earliest retained driver-history index
    needed when the rider bunch is the observer. ``rider_history_start_index``
    is the symmetric bound for driver observers.
    """

    step_index: int
    rider_alive_indices: np.ndarray
    driver_alive_indices: np.ndarray
    rider_active_indices: np.ndarray
    driver_active_indices: np.ndarray
    rider_passive_map: PassiveNeighborMap
    driver_passive_map: PassiveNeighborMap
    rider_effective_source_charges: np.ndarray
    driver_effective_source_charges: np.ndarray
    pair_reuse_penalties: np.ndarray
    max_cross_bunch_separation_mm: float
    driver_history_start_index: int | None
    rider_history_start_index: int | None


def initialize_pseudo_grid_planner_state(
    *,
    rider_particle_count: int,
    driver_particle_count: int,
    pair_reuse_window: int,
) -> PseudoGridPlannerState:
    """Create zeroed cross-step pseudo-grid planning state."""
    if rider_particle_count < 0:
        raise ValueError("rider_particle_count must be non-negative")
    if driver_particle_count < 0:
        raise ValueError("driver_particle_count must be non-negative")

    return PseudoGridPlannerState(
        rider_last_active_step=np.full(rider_particle_count, -1, dtype=int),
        rider_activation_count=np.zeros(rider_particle_count, dtype=int),
        driver_last_active_step=np.full(driver_particle_count, -1, dtype=int),
        driver_activation_count=np.zeros(driver_particle_count, dtype=int),
        pair_reuse_tracker=PairReuseTracker(window_steps=pair_reuse_window),
    )


def record_pseudo_grid_history_times(
    planner_state: PseudoGridPlannerState,
    rider_state: ParticleState,
    driver_state: ParticleState,
) -> None:
    """Append representative completed-step times for rider and driver histories."""
    rider_alive = _alive_indices_for_schedule(rider_state)
    driver_alive = _alive_indices_for_schedule(driver_state)
    planner_state.rider_history_times_ns.append(
        _representative_step_time_ns(rider_state, rider_alive)
    )
    planner_state.driver_history_times_ns.append(
        _representative_step_time_ns(driver_state, driver_alive)
    )


def update_activation_history(
    last_active_step: np.ndarray,
    activation_count: np.ndarray,
    active_indices: np.ndarray,
    *,
    step_index: int,
) -> None:
    """Update bounded per-particle activation bookkeeping in place."""
    active = np.asarray(active_indices, dtype=int)
    if active.size == 0:
        return
    last_active_step[active] = int(step_index)
    activation_count[active] += 1


def select_active_indices(
    state: ParticleState,
    alive_indices: np.ndarray,
    *,
    active_count: int,
    step_index: int,
    last_active_step: np.ndarray | None = None,
    activation_count: np.ndarray | None = None,
) -> np.ndarray:
    """Select an active subset using coverage plus recency/staleness bias."""
    if active_count <= 0:
        raise ValueError("active_count must be positive")

    alive = np.asarray(alive_indices, dtype=int)
    if alive.size == 0:
        return np.zeros(0, dtype=int)

    alive = np.unique(alive)
    if active_count >= alive.size:
        return alive.copy()

    coords = _normalized_position_coordinates(
        state, reference_indices=alive, target_indices=alive
    )
    stale_scores = np.full(alive.size, float(step_index + 1), dtype=float)
    if last_active_step is not None:
        stale_scores = np.maximum(
            float(step_index) - np.asarray(last_active_step, dtype=float)[alive],
            0.0,
        )
    stale_scores = _normalize_vector(stale_scores)

    activation_penalties = np.zeros(alive.size, dtype=float)
    if activation_count is not None:
        activation_penalties = _normalize_vector(
            np.asarray(activation_count, dtype=float)[alive]
        )

    selected_local: list[int] = []
    available_mask = np.ones(alive.size, dtype=bool)

    first_scores = stale_scores - 0.25 * activation_penalties
    first_local = _argmax_with_tiebreak(first_scores, alive)
    selected_local.append(first_local)
    available_mask[first_local] = False

    while len(selected_local) < active_count:
        candidate_local = np.flatnonzero(available_mask)
        selected_coords = coords[np.asarray(selected_local, dtype=int)]
        candidate_coords = coords[candidate_local]
        min_distances = np.min(
            np.linalg.norm(
                candidate_coords[:, np.newaxis, :] - selected_coords[np.newaxis, :, :],
                axis=2,
            ),
            axis=1,
        )
        distance_scores = _normalize_vector(min_distances)
        candidate_scores = (
            distance_scores
            + 0.20 * stale_scores[candidate_local]
            - 0.10 * activation_penalties[candidate_local]
        )
        chosen_local_idx = _argmax_with_tiebreak(
            candidate_scores, alive[candidate_local]
        )
        chosen_local = candidate_local[chosen_local_idx]
        selected_local.append(int(chosen_local))
        available_mask[chosen_local] = False

    return alive[np.asarray(selected_local, dtype=int)]


def build_passive_neighbor_map(
    state: ParticleState,
    alive_indices: np.ndarray,
    active_indices: np.ndarray,
    *,
    neighbor_count: int,
    weighting_mode: str = "inverse_distance",
) -> PassiveNeighborMap:
    """Assign passive particles to their nearest active anchors."""
    if neighbor_count <= 0:
        raise ValueError("neighbor_count must be positive")

    alive = np.unique(np.asarray(alive_indices, dtype=int))
    active = np.unique(np.asarray(active_indices, dtype=int))
    if active.size == 0:
        raise ValueError("active_indices must contain at least one particle")

    active_mask = np.isin(alive, active)
    passive = alive[~active_mask]
    if passive.size == 0:
        return _empty_neighbor_map()

    active_coords = _normalized_position_coordinates(
        state, reference_indices=alive, target_indices=active
    )
    passive_coords = _normalized_position_coordinates(
        state, reference_indices=alive, target_indices=passive
    )

    tree = KDTree(active_coords)
    k = min(int(neighbor_count), active.size)
    distances, neighbor_positions = tree.query(passive_coords, k=k)
    distances = np.asarray(distances, dtype=float)
    neighbor_positions = np.asarray(neighbor_positions, dtype=int)
    if k == 1:
        distances = distances[:, np.newaxis]
        neighbor_positions = neighbor_positions[:, np.newaxis]

    neighbor_particle_indices = active[neighbor_positions]
    for row_idx in range(neighbor_particle_indices.shape[0]):
        row_order = np.lexsort((neighbor_particle_indices[row_idx], distances[row_idx]))
        neighbor_particle_indices[row_idx] = neighbor_particle_indices[
            row_idx, row_order
        ]
        distances[row_idx] = distances[row_idx, row_order]

    weights = _compute_neighbor_weights(distances, weighting_mode)
    return PassiveNeighborMap(
        passive_indices=passive,
        neighbor_particle_indices=neighbor_particle_indices,
        weights=weights,
    )


def accumulate_effective_source_charges(
    state: ParticleState,
    active_indices: np.ndarray,
    neighbor_map: PassiveNeighborMap,
) -> np.ndarray:
    """Aggregate passive charge onto the active representatives."""
    active = np.asarray(active_indices, dtype=int)
    charges = np.asarray(state["q"], dtype=float)
    effective = charges[active].astype(float).copy()
    if neighbor_map.is_empty:
        return effective

    active_lookup = {int(particle_idx): idx for idx, particle_idx in enumerate(active)}
    passive_charges = charges[np.asarray(neighbor_map.passive_indices, dtype=int)]
    for row_idx, passive_charge in enumerate(passive_charges):
        neighbors = neighbor_map.neighbor_particle_indices[row_idx]
        weights = neighbor_map.weights[row_idx]
        for neighbor_particle_idx, weight in zip(neighbors, weights):
            effective[active_lookup[int(neighbor_particle_idx)]] += float(
                passive_charge
            ) * float(weight)
    return effective


def build_self_excluded_space_charge_source_charges(
    state: ParticleState,
    active_indices: np.ndarray,
    passive_map: PassiveNeighborMap,
    *,
    weighting_mode: str = "inverse_distance",
) -> np.ndarray:
    """Build observer-specific source charges for reduced intra-bunch space charge.

    The returned matrix has shape ``[n_active_observers, n_active_sources]``.
    Row ``i`` contains the effective source charges seen by active observer ``i``.
    The observer's own active charge is excluded, and any passive charge that
    would otherwise land on the observer's representative is redistributed onto
    the remaining active representatives.
    """
    active = np.asarray(active_indices, dtype=int)
    if active.ndim != 1:
        raise ValueError("active_indices must be a 1-D array")

    active_count = active.size
    charge_matrix = np.zeros((active_count, active_count), dtype=float)
    if active_count == 0:
        return charge_matrix

    charges = np.asarray(state["q"], dtype=float)
    active_lookup = {int(particle_idx): idx for idx, particle_idx in enumerate(active)}
    active_charges = charges[active].astype(float)
    for observer_local_idx in range(active_count):
        charge_matrix[observer_local_idx] = active_charges
        charge_matrix[observer_local_idx, observer_local_idx] = 0.0

    if passive_map.is_empty:
        return charge_matrix

    passive_indices = np.asarray(passive_map.passive_indices, dtype=int)
    neighbor_particle_indices = np.asarray(
        passive_map.neighbor_particle_indices,
        dtype=int,
    )
    neighbor_weights = np.asarray(passive_map.weights, dtype=float)
    if neighbor_particle_indices.shape[0] != passive_indices.size:
        raise ValueError(
            "passive_map.neighbor_particle_indices must align with passive_indices"
        )
    if neighbor_weights.shape != neighbor_particle_indices.shape:
        raise ValueError("passive_map.weights must match neighbor index shape")

    reference_indices = np.unique(np.concatenate((active, passive_indices)))
    active_coords = _normalized_position_coordinates(
        state,
        reference_indices=reference_indices,
        target_indices=active,
    )
    passive_coords = _normalized_position_coordinates(
        state,
        reference_indices=reference_indices,
        target_indices=passive_indices,
    )

    for passive_row_idx, passive_particle_idx in enumerate(passive_indices):
        passive_charge = float(charges[passive_particle_idx])
        if passive_charge == 0.0:
            continue

        base_weights = np.zeros(active_count, dtype=float)
        for neighbor_particle_idx, weight in zip(
            neighbor_particle_indices[passive_row_idx],
            neighbor_weights[passive_row_idx],
        ):
            local_idx = active_lookup.get(int(neighbor_particle_idx))
            if local_idx is None:
                raise ValueError(
                    "passive_map neighbor indices must be members of active_indices"
                )
            base_weights[local_idx] += float(weight)

        base_weight_total = float(np.sum(base_weights))
        if base_weight_total <= 0.0:
            continue
        base_weights /= base_weight_total

        for observer_local_idx in range(active_count):
            observer_weights = base_weights.copy()
            if observer_weights[observer_local_idx] > 0.0:
                observer_weights[observer_local_idx] = 0.0
                redistributed_total = float(np.sum(observer_weights))
                if redistributed_total > 0.0:
                    observer_weights /= redistributed_total
                elif active_count > 1:
                    observer_weights = _fallback_self_excluded_neighbor_weights(
                        passive_coords[passive_row_idx],
                        active_coords,
                        excluded_local_idx=observer_local_idx,
                        weighting_mode=weighting_mode,
                    )
                else:
                    observer_weights = np.zeros(active_count, dtype=float)

            charge_matrix[observer_local_idx] += passive_charge * observer_weights

    return charge_matrix


def compute_causal_history_start_index(
    source_times_ns: np.ndarray,
    *,
    current_observer_time_ns: float,
    max_separation_mm: float,
    safety_margin_steps: int = 0,
) -> int:
    """Return the earliest retained history index under a conservative light cone.

    A source sample older than ``current_observer_time_ns - max_separation_mm / c``
    can no longer influence the current or any future observer step, because all
    later observer times require source times that are newer than that bound.
    """
    if max_separation_mm < 0.0:
        raise ValueError("max_separation_mm must be non-negative")
    if safety_margin_steps < 0:
        raise ValueError("safety_margin_steps must be non-negative")

    source_times = np.asarray(source_times_ns, dtype=float)
    if source_times.ndim != 1:
        raise ValueError("source_times_ns must be a 1-D array")
    if source_times.size == 0:
        return 0
    if np.any(np.diff(source_times) < 0.0):
        raise ValueError("source_times_ns must be monotonically non-decreasing")

    causal_cutoff_time = (
        float(current_observer_time_ns) - float(max_separation_mm) / C_MMNS
    )
    raw_index = int(np.searchsorted(source_times, causal_cutoff_time, side="left"))
    return max(0, raw_index - int(safety_margin_steps))


def slice_particle_state(
    state: ParticleState,
    particle_indices: np.ndarray,
    *,
    q_override: np.ndarray | None = None,
) -> ParticleState:
    """Return a particle-subset copy of a legacy ``ParticleState``."""
    indices = np.asarray(particle_indices, dtype=int)
    if indices.ndim != 1:
        raise ValueError("particle_indices must be a 1-D array")

    particle_count = len(np.asarray(state.get("x", [])))
    active_lookup = {int(particle_idx): idx for idx, particle_idx in enumerate(indices)}
    subset: ParticleState = {}

    for key, value in state.items():
        if key == "_particle_failure_info":
            remapped_failure_info = {}
            for particle_idx, info in value.items():
                local_idx = active_lookup.get(int(particle_idx))
                if local_idx is not None:
                    remapped_failure_info[int(local_idx)] = copy.deepcopy(info)
            if remapped_failure_info:
                subset[key] = remapped_failure_info
            continue

        if key == "_pseudo_grid_schedule":
            continue

        if isinstance(value, np.ndarray):
            if value.ndim >= 1 and value.shape[0] == particle_count:
                subset[key] = np.asarray(value)[indices].copy()
            else:
                subset[key] = np.array(value, copy=True)
            continue

        subset[key] = copy.deepcopy(value)

    if q_override is not None:
        q_array = np.asarray(q_override, dtype=float)
        if "q" not in subset:
            raise KeyError("state must contain 'q' when q_override is supplied")
        if q_array.shape != subset["q"].shape:
            raise ValueError("q_override must match the sliced particle-count shape")
        subset["q"] = q_array.copy()

    return subset


def slice_trajectory_particle_history(
    trajectory: Trajectory,
    particle_indices: np.ndarray,
    *,
    start_index: int = 0,
    q_override: np.ndarray | None = None,
) -> Trajectory:
    """Return a sliced legacy trajectory for one active observer/source subset."""
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    return [
        slice_particle_state(state, particle_indices, q_override=q_override)
        for state in trajectory[start_index:]
    ]


def reconstruct_full_state_from_active_result(
    previous_full_state: ParticleState,
    active_indices: np.ndarray,
    active_result_state: ParticleState,
    passive_map: PassiveNeighborMap,
    *,
    loss_tracking_enabled: bool = True,
) -> ParticleState:
    """Rebuild a full bunch state from an active-only solve result."""
    full_state = _copy_particle_state(previous_full_state)
    active = np.asarray(active_indices, dtype=int)
    if active.size == 0:
        return full_state

    result_particle_count = len(np.asarray(active_result_state.get("x", [])))
    if result_particle_count != active.size:
        raise ValueError(
            "active_result_state particle count must match active_indices length"
        )

    active_lookup = {int(particle_idx): idx for idx, particle_idx in enumerate(active)}
    active_dead_mask = np.asarray(
        active_result_state.get("_dead_particles", np.zeros(active.size, dtype=bool)),
        dtype=bool,
    )

    for key, value in active_result_state.items():
        if key.startswith("_") or key == "dummy" or not isinstance(value, np.ndarray):
            continue
        if key not in full_state:
            continue
        if value.ndim >= 1 and value.shape[0] == active.size:
            full_state[key][active] = value

    if loss_tracking_enabled:
        full_dead_mask = np.asarray(
            full_state.get(
                "_dead_particles",
                np.zeros(len(np.asarray(previous_full_state.get("x", []))), dtype=bool),
            ),
            dtype=bool,
        ).copy()
        full_dead_mask[active] = active_dead_mask
        full_state["_dead_particles"] = full_dead_mask

        previous_failure_info = previous_full_state.get("_particle_failure_info", {})
        full_failure_info = {
            int(particle_idx): copy.deepcopy(info)
            for particle_idx, info in previous_failure_info.items()
        }
        for local_idx, info in active_result_state.get(
            "_particle_failure_info", {}
        ).items():
            if 0 <= int(local_idx) < active.size:
                full_failure_info[int(active[int(local_idx)])] = copy.deepcopy(info)
        if full_failure_info:
            full_state["_particle_failure_info"] = full_failure_info
        else:
            full_state.pop("_particle_failure_info", None)

    active_field_deltas = {
        field_name: np.asarray(active_result_state[field_name], dtype=float)
        - np.asarray(previous_full_state[field_name], dtype=float)[active]
        for field_name in PSEUDO_GRID_PASSIVE_DELTA_FIELDS
        if field_name in active_result_state and field_name in previous_full_state
    }

    passive_indices = np.asarray(passive_map.passive_indices, dtype=int)
    if passive_indices.size == 0:
        return full_state

    full_dead_mask = np.asarray(
        full_state.get(
            "_dead_particles",
            np.zeros(len(np.asarray(previous_full_state.get("x", []))), dtype=bool),
        ),
        dtype=bool,
    )

    for row_idx, passive_idx in enumerate(passive_indices):
        if full_dead_mask[passive_idx]:
            continue

        neighbor_particle_indices = passive_map.neighbor_particle_indices[row_idx]
        local_neighbor_indices = np.asarray(
            [
                active_lookup[int(particle_idx)]
                for particle_idx in neighbor_particle_indices
            ],
            dtype=int,
        )
        weights = np.asarray(passive_map.weights[row_idx], dtype=float).copy()

        if loss_tracking_enabled and active_dead_mask.size > 0:
            alive_anchor_mask = ~active_dead_mask[local_neighbor_indices]
            if np.any(alive_anchor_mask):
                local_neighbor_indices = local_neighbor_indices[alive_anchor_mask]
                weights = weights[alive_anchor_mask]
                weights /= float(np.sum(weights))
            else:
                continue

        for field_name, delta_values in active_field_deltas.items():
            full_state[field_name][passive_idx] = previous_full_state[field_name][
                passive_idx
            ] + float(np.dot(weights, delta_values[local_neighbor_indices]))

        if (
            "beta_avg_x" in previous_full_state
            and "beta_avg_y" in previous_full_state
            and "beta_avg_z" in previous_full_state
            and "beta_samples" in previous_full_state
            and "bx" in full_state
            and "by" in full_state
            and "bz" in full_state
        ):
            previous_avg = (
                float(previous_full_state["beta_avg_x"][passive_idx]),
                float(previous_full_state["beta_avg_y"][passive_idx]),
                float(previous_full_state["beta_avg_z"][passive_idx]),
            )
            previous_sample_count = float(
                previous_full_state["beta_samples"][passive_idx]
            )
            new_beta = (
                float(full_state["bx"][passive_idx]),
                float(full_state["by"][passive_idx]),
                float(full_state["bz"][passive_idx]),
            )
            updated_beta_avg, updated_sample_count = _update_beta_running_average(
                previous_avg,
                previous_sample_count,
                new_beta,
            )
            full_state["beta_samples"][passive_idx] = updated_sample_count
            full_state["beta_avg_x"][passive_idx] = updated_beta_avg[0]
            full_state["beta_avg_y"][passive_idx] = updated_beta_avg[1]
            full_state["beta_avg_z"][passive_idx] = updated_beta_avg[2]

    return full_state


def build_pseudo_grid_step_schedule(
    rider_state: ParticleState,
    driver_state: ParticleState,
    *,
    step_index: int,
    config: PseudoGridConfig,
    planner_state: PseudoGridPlannerState,
) -> PseudoGridStepSchedule:
    """Build one conservative pseudo-grid schedule snapshot for an outer step."""
    if step_index < 1:
        raise ValueError("step_index must be >= 1")
    if config.coverage_strategy != "farthest_point_staleness":
        raise NotImplementedError(
            "Pseudo-grid schedule construction currently supports only "
            "coverage_strategy='farthest_point_staleness'."
        )
    if config.coverage_space != "position":
        raise NotImplementedError(
            "Pseudo-grid schedule construction currently supports only "
            "coverage_space='position'."
        )

    rider_alive = _alive_indices_for_schedule(rider_state)
    driver_alive = _alive_indices_for_schedule(driver_state)

    rider_active = select_active_indices(
        rider_state,
        rider_alive,
        active_count=config.active_rider_count,
        step_index=step_index,
        last_active_step=planner_state.rider_last_active_step,
        activation_count=planner_state.rider_activation_count,
    )
    driver_active = select_active_indices(
        driver_state,
        driver_alive,
        active_count=config.active_driver_count,
        step_index=step_index,
        last_active_step=planner_state.driver_last_active_step,
        activation_count=planner_state.driver_activation_count,
    )

    rider_passive_map = _empty_neighbor_map()
    if rider_active.size > 0:
        rider_passive_map = build_passive_neighbor_map(
            rider_state,
            rider_alive,
            rider_active,
            neighbor_count=config.passive_neighbor_count,
            weighting_mode=config.source_weighting_mode,
        )

    driver_passive_map = _empty_neighbor_map()
    if driver_active.size > 0:
        driver_passive_map = build_passive_neighbor_map(
            driver_state,
            driver_alive,
            driver_active,
            neighbor_count=config.passive_neighbor_count,
            weighting_mode=config.source_weighting_mode,
        )

    rider_effective_source_charges = accumulate_effective_source_charges(
        rider_state,
        rider_active,
        rider_passive_map,
    )
    driver_effective_source_charges = accumulate_effective_source_charges(
        driver_state,
        driver_active,
        driver_passive_map,
    )

    planner_state.pair_reuse_tracker.prune(step_index)
    pair_reuse_penalties = _build_pair_reuse_penalty_matrix(
        planner_state.pair_reuse_tracker,
        rider_active,
        driver_active,
        step_index=step_index,
    )

    max_cross_bunch_separation_mm = _conservative_max_cross_bunch_separation_mm(
        rider_state,
        driver_state,
        rider_alive,
        driver_alive,
    )

    driver_history_start_index: int | None = None
    rider_history_start_index: int | None = None
    if config.causal_history_pruning_enabled:
        driver_history_start_index = compute_causal_history_start_index(
            np.asarray(planner_state.driver_history_times_ns, dtype=float),
            current_observer_time_ns=_representative_step_time_ns(
                rider_state, rider_alive
            ),
            max_separation_mm=max_cross_bunch_separation_mm,
            safety_margin_steps=config.causal_history_safety_margin_steps,
        )
        rider_history_start_index = compute_causal_history_start_index(
            np.asarray(planner_state.rider_history_times_ns, dtype=float),
            current_observer_time_ns=_representative_step_time_ns(
                driver_state, driver_alive
            ),
            max_separation_mm=max_cross_bunch_separation_mm,
            safety_margin_steps=config.causal_history_safety_margin_steps,
        )

    return PseudoGridStepSchedule(
        step_index=step_index,
        rider_alive_indices=rider_alive,
        driver_alive_indices=driver_alive,
        rider_active_indices=rider_active,
        driver_active_indices=driver_active,
        rider_passive_map=rider_passive_map,
        driver_passive_map=driver_passive_map,
        rider_effective_source_charges=rider_effective_source_charges,
        driver_effective_source_charges=driver_effective_source_charges,
        pair_reuse_penalties=pair_reuse_penalties,
        max_cross_bunch_separation_mm=max_cross_bunch_separation_mm,
        driver_history_start_index=driver_history_start_index,
        rider_history_start_index=rider_history_start_index,
    )


def commit_pseudo_grid_step_schedule(
    planner_state: PseudoGridPlannerState,
    schedule: PseudoGridStepSchedule,
) -> None:
    """Commit accepted schedule bookkeeping after one outer step succeeds."""
    update_activation_history(
        planner_state.rider_last_active_step,
        planner_state.rider_activation_count,
        schedule.rider_active_indices,
        step_index=schedule.step_index,
    )
    update_activation_history(
        planner_state.driver_last_active_step,
        planner_state.driver_activation_count,
        schedule.driver_active_indices,
        step_index=schedule.step_index,
    )
    planner_state.pair_reuse_tracker.note_matches(
        schedule.rider_active_indices,
        schedule.driver_active_indices,
        step_index=schedule.step_index,
    )


def _copy_particle_state(state: ParticleState) -> ParticleState:
    copied_state: ParticleState = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            copied_state[key] = np.array(value, copy=True)
        else:
            copied_state[key] = copy.deepcopy(value)
    return copied_state


def _update_beta_running_average(
    previous_avg: tuple[float, float, float],
    previous_sample_count: float,
    new_beta: tuple[float, float, float],
) -> tuple[tuple[float, float, float], float]:
    new_sample_count = previous_sample_count + 1.0
    avg_x = (previous_avg[0] * previous_sample_count + new_beta[0]) / new_sample_count
    avg_y = (previous_avg[1] * previous_sample_count + new_beta[1]) / new_sample_count
    avg_z = (previous_avg[2] * previous_sample_count + new_beta[2]) / new_sample_count
    return (avg_x, avg_y, avg_z), new_sample_count


def _alive_indices_for_schedule(state: ParticleState) -> np.ndarray:
    alive = get_alive_particle_indices(state)
    if alive.size > 0 or "_dead_particles" in state or "gamma" in state:
        return np.asarray(alive, dtype=int)
    return np.arange(len(np.asarray(state.get("x", []))), dtype=int)


def _empty_neighbor_map() -> PassiveNeighborMap:
    return PassiveNeighborMap(
        passive_indices=np.zeros(0, dtype=int),
        neighbor_particle_indices=np.zeros((0, 0), dtype=int),
        weights=np.zeros((0, 0), dtype=float),
    )


def _build_pair_reuse_penalty_matrix(
    tracker: PairReuseTracker,
    rider_active_indices: np.ndarray,
    driver_active_indices: np.ndarray,
    *,
    step_index: int,
) -> np.ndarray:
    rider_active = np.asarray(rider_active_indices, dtype=int)
    driver_active = np.asarray(driver_active_indices, dtype=int)
    penalties = np.zeros((rider_active.size, driver_active.size), dtype=float)
    for rider_idx, rider_particle_idx in enumerate(rider_active):
        for driver_idx, driver_particle_idx in enumerate(driver_active):
            penalties[rider_idx, driver_idx] = tracker.penalty(
                int(rider_particle_idx),
                int(driver_particle_idx),
                step_index=step_index,
            )
    return penalties


def _representative_step_time_ns(
    state: ParticleState,
    alive_indices: np.ndarray,
) -> float:
    times_raw = state.get("t")
    if times_raw is None:
        return 0.0

    times = np.asarray(times_raw, dtype=float)
    if times.size == 0:
        return 0.0

    alive = np.asarray(alive_indices, dtype=int)
    if alive.size == 0:
        return float(np.min(times))
    return float(np.min(times[alive]))


def _conservative_max_cross_bunch_separation_mm(
    rider_state: ParticleState,
    driver_state: ParticleState,
    rider_alive_indices: np.ndarray,
    driver_alive_indices: np.ndarray,
) -> float:
    rider_alive = np.asarray(rider_alive_indices, dtype=int)
    driver_alive = np.asarray(driver_alive_indices, dtype=int)
    if rider_alive.size == 0 or driver_alive.size == 0:
        return 0.0

    rider_coords = np.column_stack(
        (
            np.asarray(rider_state["x"], dtype=float)[rider_alive],
            np.asarray(rider_state["y"], dtype=float)[rider_alive],
            np.asarray(rider_state["z"], dtype=float)[rider_alive],
        )
    )
    driver_coords = np.column_stack(
        (
            np.asarray(driver_state["x"], dtype=float)[driver_alive],
            np.asarray(driver_state["y"], dtype=float)[driver_alive],
            np.asarray(driver_state["z"], dtype=float)[driver_alive],
        )
    )

    rider_min = np.min(rider_coords, axis=0)
    rider_max = np.max(rider_coords, axis=0)
    driver_min = np.min(driver_coords, axis=0)
    driver_max = np.max(driver_coords, axis=0)

    per_axis_extreme = np.maximum(
        np.abs(rider_max - driver_min),
        np.abs(driver_max - rider_min),
    )
    return float(np.linalg.norm(per_axis_extreme))


def _normalized_position_coordinates(
    state: ParticleState,
    *,
    reference_indices: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_indices, dtype=int)
    target = np.asarray(target_indices, dtype=int)
    if target.size == 0:
        return np.zeros((0, 3), dtype=float)

    reference_coords = np.column_stack(
        (
            np.asarray(state["x"], dtype=float)[reference],
            np.asarray(state["y"], dtype=float)[reference],
            np.asarray(state["z"], dtype=float)[reference],
        )
    )
    target_coords = np.column_stack(
        (
            np.asarray(state["x"], dtype=float)[target],
            np.asarray(state["y"], dtype=float)[target],
            np.asarray(state["z"], dtype=float)[target],
        )
    )

    centers = np.mean(reference_coords, axis=0)
    spans = np.ptp(reference_coords, axis=0)
    spans = np.where(spans > 0.0, spans, 1.0)
    return (target_coords - centers) / spans


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    span = float(np.max(arr) - np.min(arr))
    if span <= 0.0:
        return np.zeros_like(arr)
    return (arr - float(np.min(arr))) / span


def _argmax_with_tiebreak(scores: np.ndarray, particle_indices: np.ndarray) -> int:
    best_score = float(np.max(scores))
    candidates = np.flatnonzero(np.isclose(scores, best_score))
    if candidates.size == 1:
        return int(candidates[0])
    candidate_particles = np.asarray(particle_indices, dtype=int)[candidates]
    return int(candidates[int(np.argmin(candidate_particles))])


def _compute_neighbor_weights(
    distances: np.ndarray,
    weighting_mode: str,
) -> np.ndarray:
    if weighting_mode not in {"inverse_distance", "nearest"}:
        raise ValueError("weighting_mode must be 'inverse_distance' or 'nearest'")

    if weighting_mode == "nearest":
        weights = np.zeros_like(distances, dtype=float)
        weights[:, 0] = 1.0
        return weights

    weights = np.zeros_like(distances, dtype=float)
    zero_mask = distances <= 1.0e-12
    if np.any(zero_mask):
        for row_idx in range(distances.shape[0]):
            if np.any(zero_mask[row_idx]):
                zero_count = int(np.sum(zero_mask[row_idx]))
                weights[row_idx, zero_mask[row_idx]] = 1.0 / float(zero_count)

    non_zero_rows = ~np.any(zero_mask, axis=1)
    if np.any(non_zero_rows):
        inv = 1.0 / np.maximum(distances[non_zero_rows], 1.0e-12)
        weights[non_zero_rows] = inv / np.sum(inv, axis=1, keepdims=True)
    return weights


def _fallback_self_excluded_neighbor_weights(
    passive_coord: np.ndarray,
    active_coords: np.ndarray,
    *,
    excluded_local_idx: int,
    weighting_mode: str,
) -> np.ndarray:
    weights = np.zeros(active_coords.shape[0], dtype=float)
    include_mask = np.ones(active_coords.shape[0], dtype=bool)
    include_mask[int(excluded_local_idx)] = False
    candidate_indices = np.flatnonzero(include_mask)
    if candidate_indices.size == 0:
        return weights

    distances = np.linalg.norm(
        active_coords[candidate_indices] - passive_coord[np.newaxis, :],
        axis=1,
    )[np.newaxis, :]
    fallback_weights = _compute_neighbor_weights(distances, weighting_mode)[0]
    weights[candidate_indices] = fallback_weights
    return weights


__all__ = [
    "PassiveNeighborMap",
    "PairReuseTracker",
    "PseudoGridPlannerState",
    "PseudoGridStepSchedule",
    "accumulate_effective_source_charges",
    "build_self_excluded_space_charge_source_charges",
    "build_passive_neighbor_map",
    "build_pseudo_grid_step_schedule",
    "commit_pseudo_grid_step_schedule",
    "compute_causal_history_start_index",
    "initialize_pseudo_grid_planner_state",
    "record_pseudo_grid_history_times",
    "reconstruct_full_state_from_active_result",
    "select_active_indices",
    "slice_particle_state",
    "slice_trajectory_particle_history",
    "update_activation_history",
]
