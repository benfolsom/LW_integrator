"""Scheduling and bookkeeping helpers for the experimental pseudo-grid mode.

The reduced active/passive solver path is still under development, but these
helpers provide the bounded-memory pieces that the public configuration surface
already refers to:

- active-subset selection with coverage and recency bias;
- field-representative selection for weighted retarded LW source sums;
- passive-particle nearest-neighbour anchors and weights;
- effective source-charge aggregation from passive particles;
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
from .types import IndexedTrajectoryArrays, ParticleState, PseudoGridConfig, Trajectory


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


class ActiveTrajectoryView:
    """Lazy legacy-state adapter for an indexed SOA trajectory."""

    def __init__(self, indexed_soa: IndexedTrajectoryArrays) -> None:
        self.indexed_soa = indexed_soa
        self._state_cache: dict[int, ParticleState] = {}

    def __len__(self) -> int:
        return self.indexed_soa.n_steps

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, index: int | slice) -> ParticleState | list[ParticleState]:
        if isinstance(index, slice):
            states: list[ParticleState] = []
            for local_index in range(*index.indices(len(self))):
                states.append(self._state_at(local_index))
            return states
        return self._state_at(index)

    def copy(self) -> list[ParticleState]:
        return [self._state_at(i) for i in range(len(self))]

    def _state_at(self, index: int) -> ParticleState:
        local_index = int(index)
        if local_index < 0:
            local_index += len(self)
        if local_index < 0 or local_index >= len(self):
            raise IndexError("trajectory index out of range")
        cached = self._state_cache.get(local_index)
        if cached is None:
            cached = self.indexed_soa.state_at(local_index)
            self._state_cache[local_index] = cached
        return cached


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
    rider_field_indices: np.ndarray
    driver_field_indices: np.ndarray
    rider_field_source_charges: np.ndarray
    driver_field_source_charges: np.ndarray
    rider_effective_source_charges: np.ndarray
    driver_effective_source_charges: np.ndarray
    pair_reuse_penalties: np.ndarray
    max_cross_bunch_separation_mm: float
    driver_history_start_index: int | None
    rider_history_start_index: int | None
    driver_retained_history_start_index: int | None = None
    rider_retained_history_start_index: int | None = None
    driver_dropped_history_samples: int = 0
    rider_dropped_history_samples: int = 0


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

    x = np.asarray(state["x"], dtype=float)[alive]
    y = np.asarray(state["y"], dtype=float)[alive]
    z = np.asarray(state["z"], dtype=float)[alive]
    centers = np.array((float(np.mean(x)), float(np.mean(y)), float(np.mean(z))))
    spans = np.array((float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(z))))
    spans = np.where(spans > 0.0, spans, 1.0)
    coords = np.empty((alive.size, 3), dtype=float)
    coords[:, 0] = (x - centers[0]) / spans[0]
    coords[:, 1] = (y - centers[1]) / spans[1]
    coords[:, 2] = (z - centers[2]) / spans[2]
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

    dx = coords[:, 0] - coords[first_local, 0]
    dy = coords[:, 1] - coords[first_local, 1]
    dz = coords[:, 2] - coords[first_local, 2]
    min_distances = np.sqrt(dx * dx + dy * dy + dz * dz)
    min_distances[first_local] = 0.0

    while len(selected_local) < active_count:
        candidate_local = np.flatnonzero(available_mask)
        distance_scores = _normalize_vector(min_distances[candidate_local])
        candidate_scores = (
            distance_scores
            + 0.20 * stale_scores[candidate_local]
            - 0.10 * activation_penalties[candidate_local]
        )
        chosen_local_idx = _argmax_with_tiebreak(
            candidate_scores, alive[candidate_local]
        )
        chosen_local = int(candidate_local[chosen_local_idx])
        selected_local.append(chosen_local)
        available_mask[chosen_local] = False
        dx = coords[:, 0] - coords[chosen_local, 0]
        dy = coords[:, 1] - coords[chosen_local, 1]
        dz = coords[:, 2] - coords[chosen_local, 2]
        distances_to_chosen = np.sqrt(dx * dx + dy * dy + dz * dz)
        min_distances = np.minimum(min_distances, distances_to_chosen)
        min_distances[chosen_local] = 0.0

    return alive[np.asarray(selected_local, dtype=int)]


def select_field_representative_indices(
    state: ParticleState,
    alive_indices: np.ndarray,
    active_indices: np.ndarray,
    *,
    field_count: int,
) -> np.ndarray:
    """Select weighted LW source representatives from live particles.

    Active particles are always included so the dynamic observers remain valid
    source samples. Additional representatives are chosen as medoids by
    farthest-point coverage in normalized position space. ``field_count <= 0``
    means "use active particles only" for backward compatibility.
    """
    alive = np.unique(np.asarray(alive_indices, dtype=int))
    active = np.unique(np.asarray(active_indices, dtype=int))
    if alive.size == 0:
        return np.zeros(0, dtype=int)
    active = active[np.isin(active, alive)]
    if active.size == 0:
        return np.zeros(0, dtype=int)

    target_count = active.size if field_count <= 0 else int(field_count)
    target_count = min(max(target_count, active.size), alive.size)
    if target_count == active.size:
        return active.copy()
    if target_count == alive.size:
        return alive.copy()

    coords = _normalized_position_coordinates(
        state,
        reference_indices=alive,
        target_indices=alive,
    )
    alive_lookup = {int(particle_idx): idx for idx, particle_idx in enumerate(alive)}
    selected_local = [alive_lookup[int(particle_idx)] for particle_idx in active]
    selected_mask = np.zeros(alive.size, dtype=bool)
    selected_mask[selected_local] = True

    selected_coords = coords[np.asarray(selected_local, dtype=int)]
    diff = coords[:, np.newaxis, :] - selected_coords[np.newaxis, :, :]
    min_distances = np.min(np.linalg.norm(diff, axis=2), axis=1)
    min_distances[selected_mask] = 0.0

    while int(np.sum(selected_mask)) < target_count:
        candidate_local = np.flatnonzero(~selected_mask)
        chosen_local_idx = _argmax_with_tiebreak(
            min_distances[candidate_local],
            alive[candidate_local],
        )
        chosen_local = int(candidate_local[chosen_local_idx])
        selected_mask[chosen_local] = True
        distances_to_chosen = np.linalg.norm(coords - coords[chosen_local], axis=1)
        min_distances = np.minimum(min_distances, distances_to_chosen)
        min_distances[selected_mask] = 0.0

    selected = alive[np.flatnonzero(selected_mask)]
    active_set = set(int(v) for v in active.tolist())
    active_first = [int(v) for v in active.tolist()]
    extra = [int(v) for v in selected.tolist() if int(v) not in active_set]
    return np.asarray(active_first + extra, dtype=int)


def _field_deposition_weights_for_particles(
    state: ParticleState,
    alive_indices: np.ndarray,
    particle_indices: np.ndarray,
    field_indices: np.ndarray,
    *,
    neighbor_count: int,
    weighting_mode: str = "inverse_distance",
) -> np.ndarray:
    """Return particle-to-field deposition weights.

    Rows correspond to ``particle_indices`` and columns to ``field_indices``.
    A particle that is itself a field representative deposits all of its own
    charge to that representative, matching ``accumulate_field_representative_charges``.
    """
    if neighbor_count <= 0:
        raise ValueError("neighbor_count must be positive")
    particles = np.asarray(particle_indices, dtype=int)
    field = np.asarray(field_indices, dtype=int)
    weights = np.zeros((particles.size, field.size), dtype=float)
    if particles.size == 0 or field.size == 0:
        return weights

    field_lookup = {int(particle_idx): idx for idx, particle_idx in enumerate(field)}
    non_field_rows: list[int] = []
    non_field_particles: list[int] = []
    for row_idx, particle_idx in enumerate(particles.tolist()):
        field_col = field_lookup.get(int(particle_idx))
        if field_col is not None:
            weights[row_idx, field_col] = 1.0
        else:
            non_field_rows.append(row_idx)
            non_field_particles.append(int(particle_idx))

    if not non_field_particles:
        return weights

    alive = np.unique(np.asarray(alive_indices, dtype=int))
    field_coords = _normalized_position_coordinates(
        state,
        reference_indices=alive,
        target_indices=field,
    )
    particle_coords = _normalized_position_coordinates(
        state,
        reference_indices=alive,
        target_indices=np.asarray(non_field_particles, dtype=int),
    )
    k = min(int(neighbor_count), field.size)
    distances, neighbor_positions = KDTree(field_coords).query(particle_coords, k=k)
    distances = np.asarray(distances, dtype=float)
    neighbor_positions = np.asarray(neighbor_positions, dtype=int)
    if k == 1:
        distances = distances[:, np.newaxis]
        neighbor_positions = neighbor_positions[:, np.newaxis]
    neighbor_weights = _compute_neighbor_weights(distances, weighting_mode)
    row_array = np.asarray(non_field_rows, dtype=int)
    np.add.at(
        weights,
        (np.repeat(row_array, k), neighbor_positions.ravel()),
        neighbor_weights.ravel(),
    )
    return weights


def accumulate_field_representative_charges_and_radii(
    state: ParticleState,
    alive_indices: np.ndarray,
    field_indices: np.ndarray,
    *,
    neighbor_count: int,
    weighting_mode: str = "inverse_distance",
) -> tuple[np.ndarray, np.ndarray]:
    """Deposit live source charge and estimate field-rep cloud radii.

    Each non-representative live particle deposits source charge directly to its
    nearest field representatives. The returned radius is the charge-magnitude
    weighted RMS physical distance from each field representative to the finite
    cloud of particles represented by that source. It is intended for same-bunch
    pseudo-grid space-charge softening, not for cross-bunch source weighting.
    """
    if neighbor_count <= 0:
        raise ValueError("neighbor_count must be positive")
    alive = np.unique(np.asarray(alive_indices, dtype=int))
    field_input = np.asarray(field_indices, dtype=int)
    if field_input.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty
    alive_set = set(int(v) for v in alive.tolist())
    seen: set[int] = set()
    field = np.asarray(
        [
            int(v)
            for v in field_input.tolist()
            if int(v) in alive_set and not (int(v) in seen or seen.add(int(v)))
        ],
        dtype=int,
    )
    if field.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty
    charges = np.asarray(state.get("q_source", state["q"]), dtype=float)
    effective = charges[field].astype(float).copy()
    radius_weight = np.abs(effective)
    radius_moment = np.zeros(field.size, dtype=float)
    non_field = alive[~np.isin(alive, field)]
    if non_field.size == 0:
        return effective, np.zeros(field.size, dtype=float)

    weights = _field_deposition_weights_for_particles(
        state,
        alive,
        non_field,
        field,
        neighbor_count=neighbor_count,
        weighting_mode=weighting_mode,
    )
    contributions = charges[non_field][:, np.newaxis] * weights
    np.add.at(
        effective, np.tile(np.arange(field.size), non_field.size), contributions.ravel()
    )

    field_positions = np.column_stack(
        (
            np.asarray(state["x"], dtype=float)[field],
            np.asarray(state["y"], dtype=float)[field],
            np.asarray(state["z"], dtype=float)[field],
        )
    )
    particle_positions = np.column_stack(
        (
            np.asarray(state["x"], dtype=float)[non_field],
            np.asarray(state["y"], dtype=float)[non_field],
            np.asarray(state["z"], dtype=float)[non_field],
        )
    )
    distance_sq = np.sum(
        (particle_positions[:, np.newaxis, :] - field_positions[np.newaxis, :, :])
        ** 2,
        axis=2,
    )
    abs_contributions = np.abs(charges[non_field])[:, np.newaxis] * weights
    radius_weight += np.sum(abs_contributions, axis=0)
    radius_moment += np.sum(abs_contributions * distance_sq, axis=0)
    radii = np.zeros(field.size, dtype=float)
    np.divide(
        radius_moment,
        radius_weight,
        out=radii,
        where=radius_weight > 0.0,
    )
    return effective, np.sqrt(np.maximum(radii, 0.0))


def accumulate_field_representative_charges(
    state: ParticleState,
    alive_indices: np.ndarray,
    field_indices: np.ndarray,
    *,
    neighbor_count: int,
    weighting_mode: str = "inverse_distance",
) -> np.ndarray:
    """Deposit live source charge onto field representatives."""
    charges, _ = accumulate_field_representative_charges_and_radii(
        state,
        alive_indices,
        field_indices,
        neighbor_count=neighbor_count,
        weighting_mode=weighting_mode,
    )
    return charges


def build_passive_neighbor_map(
    state: ParticleState,
    alive_indices: np.ndarray,
    active_indices: np.ndarray,
    *,
    neighbor_count: int,
    weighting_mode: str = "inverse_distance",
) -> PassiveNeighborMap:
    """Assign passive particles using full-bunch neighbors collapsed to actives.

    Passive particles first sample nearest neighbours from the full alive set
    (active and passive alike, excluding themselves). Any passive-to-passive
    links are then algebraically collapsed onto the active representatives so
    downstream reduced-solver code can keep consuming active-only anchors.
    """
    if neighbor_count <= 0:
        raise ValueError("neighbor_count must be positive")

    alive = np.asarray(alive_indices, dtype=int)
    active = np.asarray(active_indices, dtype=int)
    if active.size == 0:
        raise ValueError("active_indices must contain at least one particle")

    max_index = int(max(np.max(alive), np.max(active)))
    active_members = np.zeros(max_index + 1, dtype=bool)
    active_members[active] = True
    passive = alive[~active_members[alive]]
    if passive.size == 0:
        return _empty_neighbor_map()

    if alive.size <= 1:
        raise ValueError("alive_indices must contain at least one non-self neighbor")

    alive_coords = _normalized_position_coordinates(
        state,
        reference_indices=alive,
        target_indices=alive,
    )
    passive_coords = _normalized_position_coordinates(
        state,
        reference_indices=alive,
        target_indices=passive,
    )
    active_coords = _normalized_position_coordinates(
        state,
        reference_indices=alive,
        target_indices=active,
    )

    k = min(int(neighbor_count), alive.size - 1)
    query_k = min(alive.size, k + 1)
    alive_tree = KDTree(alive_coords)
    distances, neighbor_positions = alive_tree.query(passive_coords, k=query_k)
    distances = np.asarray(distances, dtype=float)
    neighbor_positions = np.asarray(neighbor_positions, dtype=int)
    if query_k == 1:
        distances = distances[:, np.newaxis]
        neighbor_positions = neighbor_positions[:, np.newaxis]

    raw_neighbor_particle_indices = np.empty((passive.size, k), dtype=int)
    raw_distances = np.empty((passive.size, k), dtype=float)
    for row_idx, passive_particle_idx in enumerate(passive):
        row_positions = np.asarray(neighbor_positions[row_idx], dtype=int).ravel()
        row_distances = np.asarray(distances[row_idx], dtype=float).ravel()
        row_particle_indices = alive[row_positions]
        non_self_mask = row_particle_indices != int(passive_particle_idx)
        filtered_particles = row_particle_indices[non_self_mask]
        filtered_distances = row_distances[non_self_mask]
        if filtered_particles.size < k:
            raise ValueError(
                "unable to construct passive neighbour list without self matches"
            )
        raw_neighbor_particle_indices[row_idx] = filtered_particles[:k]
        raw_distances[row_idx] = filtered_distances[:k]

    nearest_active_distances, nearest_active_positions = KDTree(active_coords).query(
        passive_coords,
        k=1,
    )
    nearest_active_distances = np.asarray(nearest_active_distances, dtype=float).ravel()
    nearest_active_positions = np.asarray(nearest_active_positions, dtype=int).ravel()

    for row_idx in range(passive.size):
        row_active_mask = active_members[raw_neighbor_particle_indices[row_idx]]
        if np.any(row_active_mask):
            continue
        replacement_col = int(np.argmax(raw_distances[row_idx]))
        raw_neighbor_particle_indices[row_idx, replacement_col] = active[
            nearest_active_positions[row_idx]
        ]
        raw_distances[row_idx, replacement_col] = nearest_active_distances[row_idx]

        row_order = np.lexsort(
            (
                raw_neighbor_particle_indices[row_idx],
                raw_distances[row_idx],
            )
        )
        raw_neighbor_particle_indices[row_idx] = raw_neighbor_particle_indices[
            row_idx,
            row_order,
        ]
        raw_distances[row_idx] = raw_distances[row_idx, row_order]

    raw_weights = _compute_neighbor_weights(raw_distances, weighting_mode)
    active_weights = _collapse_passive_neighbor_weights_to_active(
        passive,
        active,
        raw_neighbor_particle_indices,
        raw_weights,
    )
    return PassiveNeighborMap(
        passive_indices=passive,
        neighbor_particle_indices=np.broadcast_to(
            active[np.newaxis, :],
            (passive.size, active.size),
        ).copy(),
        weights=active_weights,
    )


def accumulate_effective_source_charges(
    state: ParticleState,
    active_indices: np.ndarray,
    neighbor_map: PassiveNeighborMap,
) -> np.ndarray:
    """Aggregate passive charge onto the active representatives."""
    active = np.asarray(active_indices, dtype=int)
    charges = np.asarray(state.get("q_source", state["q"]), dtype=float)
    effective = charges[active].astype(float).copy()
    if neighbor_map.is_empty:
        return effective

    neighbor_particle_indices = np.asarray(
        neighbor_map.neighbor_particle_indices,
        dtype=int,
    )
    max_index = int(max(np.max(active), np.max(neighbor_particle_indices)))
    active_index_lookup = np.full(max_index + 1, -1, dtype=int)
    active_index_lookup[active] = np.arange(active.size, dtype=int)
    neighbor_local_indices = active_index_lookup[neighbor_particle_indices]
    if np.any(neighbor_local_indices < 0):
        raise ValueError(
            "passive_map neighbor indices must be members of active_indices"
        )

    passive_charges = charges[np.asarray(neighbor_map.passive_indices, dtype=int)]
    contributions = passive_charges[:, np.newaxis] * np.asarray(
        neighbor_map.weights,
        dtype=float,
    )
    np.add.at(effective, neighbor_local_indices.ravel(), contributions.ravel())
    return effective


def build_field_representative_space_charge_source_charges(
    state: ParticleState,
    active_indices: np.ndarray,
    field_indices: np.ndarray,
    field_source_charges: np.ndarray,
    *,
    weighting_mode: str = "inverse_distance",
) -> np.ndarray:
    """Build observer-specific same-bunch source charges on field reps.

    The returned matrix has shape ``[n_active_observers, n_field_sources]``.
    Each row starts from the weighted field-representative source charges. If an
    active observer is also a field representative, its own physical source
    charge is excluded and any additional charge deposited onto that same
    representative is redistributed to other field reps for that observer. This
    avoids placing source charge at zero separation from the active observer.
    """
    active = np.asarray(active_indices, dtype=int)
    field = np.asarray(field_indices, dtype=int)
    source_charges = np.asarray(field_source_charges, dtype=float)
    if active.ndim != 1:
        raise ValueError("active_indices must be a 1-D array")
    if field.ndim != 1:
        raise ValueError("field_indices must be a 1-D array")
    if source_charges.shape != (field.size,):
        raise ValueError("field_source_charges must match field_indices length")

    charge_matrix = np.broadcast_to(
        source_charges[np.newaxis, :],
        (active.size, field.size),
    ).astype(float, copy=True)
    if active.size == 0 or field.size == 0:
        return charge_matrix

    charges = np.asarray(state.get("q_source", state["q"]), dtype=float)
    field_lookup = {int(particle_idx): idx for idx, particle_idx in enumerate(field)}
    reference_indices = np.unique(np.concatenate((active, field)))
    field_coords = _normalized_position_coordinates(
        state,
        reference_indices=reference_indices,
        target_indices=field,
    )
    active_coords = _normalized_position_coordinates(
        state,
        reference_indices=reference_indices,
        target_indices=active,
    )
    for observer_local_idx, observer_particle_idx in enumerate(active):
        field_local_idx = field_lookup.get(int(observer_particle_idx))
        if field_local_idx is None:
            continue
        own_charge = float(charges[int(observer_particle_idx)])
        deposited_charge = float(source_charges[field_local_idx]) - own_charge
        charge_matrix[observer_local_idx, field_local_idx] = 0.0
        if abs(deposited_charge) <= 1.0e-30:
            continue
        redistribution_weights = _fallback_self_excluded_neighbor_weights(
            active_coords[observer_local_idx],
            field_coords,
            excluded_local_idx=int(field_local_idx),
            weighting_mode=weighting_mode,
        )
        charge_matrix[observer_local_idx] += deposited_charge * redistribution_weights
    return charge_matrix


def build_hybrid_space_charge_sources(
    state: ParticleState,
    alive_indices: np.ndarray,
    active_indices: np.ndarray,
    field_indices: np.ndarray,
    field_source_charges: np.ndarray,
    *,
    field_deposition_neighbor_count: int,
    near_neighbor_count: int,
    weighting_mode: str = "inverse_distance",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build hybrid same-bunch SC sources for pseudo-grid active observers.

    The source set starts with field representatives. For each active observer,
    up to ``near_neighbor_count`` nearest live non-self particles are evaluated as
    exact sources. Exact-neighbor charge is subtracted from the field-rep
    deposits for that observer so total source charge is conserved without
    double counting. This keeps the singular local part of same-bunch space
    charge from being represented by a few heavily weighted point reps.
    """
    alive = np.unique(np.asarray(alive_indices, dtype=int))
    active = np.asarray(active_indices, dtype=int)
    field = np.asarray(field_indices, dtype=int)
    source_charges = np.asarray(field_source_charges, dtype=float)
    if active.ndim != 1:
        raise ValueError("active_indices must be a 1-D array")
    if field.ndim != 1:
        raise ValueError("field_indices must be a 1-D array")
    if source_charges.shape != (field.size,):
        raise ValueError("field_source_charges must match field_indices length")
    if near_neighbor_count < 0:
        raise ValueError("near_neighbor_count must be non-negative")
    if active.size == 0 or field.size == 0:
        return (
            field.copy(),
            np.zeros((active.size, field.size), dtype=float),
            np.zeros(field.size, dtype=float),
        )

    charges = np.asarray(state.get("q_source", state["q"]), dtype=float)
    _, field_source_radii = accumulate_field_representative_charges_and_radii(
        state,
        alive,
        field,
        neighbor_count=field_deposition_neighbor_count,
        weighting_mode=weighting_mode,
    )
    field_lookup = {int(particle_idx): idx for idx, particle_idx in enumerate(field)}
    near_by_observer: list[np.ndarray] = []
    exact_candidates: list[int] = []
    if near_neighbor_count > 0 and alive.size > 1:
        alive_coords = _normalized_position_coordinates(
            state,
            reference_indices=alive,
            target_indices=alive,
        )
        active_coords_for_query = _normalized_position_coordinates(
            state,
            reference_indices=alive,
            target_indices=active,
        )
        k = min(int(near_neighbor_count), alive.size - 1)
        query_k = min(alive.size, k + 1)
        distances, neighbor_positions = KDTree(alive_coords).query(
            active_coords_for_query,
            k=query_k,
        )
        neighbor_positions = np.asarray(neighbor_positions, dtype=int)
        if query_k == 1:
            neighbor_positions = neighbor_positions[:, np.newaxis]
        for observer_particle_idx, row_positions in zip(active, neighbor_positions):
            row_particles = alive[np.asarray(row_positions, dtype=int).ravel()]
            row_particles = row_particles[row_particles != int(observer_particle_idx)][
                :k
            ]
            near_by_observer.append(row_particles.astype(int, copy=False))
            exact_candidates.extend(int(v) for v in row_particles.tolist())
    else:
        near_by_observer = [np.zeros(0, dtype=int) for _ in active]

    extra_exact = [
        idx for idx in np.unique(exact_candidates) if idx not in field_lookup
    ]
    source_indices = np.asarray(
        field.tolist() + [int(idx) for idx in extra_exact],
        dtype=int,
    )
    source_lookup = {
        int(particle_idx): idx for idx, particle_idx in enumerate(source_indices)
    }
    charge_matrix = np.zeros((active.size, source_indices.size), dtype=float)
    charge_matrix[:, : field.size] = source_charges[np.newaxis, :]
    source_radii = np.zeros(source_indices.size, dtype=float)
    source_radii[: field.size] = field_source_radii

    reference_indices = np.unique(np.concatenate((active, source_indices)))
    field_coords = _normalized_position_coordinates(
        state,
        reference_indices=reference_indices,
        target_indices=field,
    )
    active_coords = _normalized_position_coordinates(
        state,
        reference_indices=reference_indices,
        target_indices=active,
    )

    for observer_local_idx, observer_particle_idx in enumerate(active):
        exact_neighbors = near_by_observer[observer_local_idx]
        if exact_neighbors.size > 0:
            deposition_weights = _field_deposition_weights_for_particles(
                state,
                alive,
                exact_neighbors,
                field,
                neighbor_count=field_deposition_neighbor_count,
                weighting_mode=weighting_mode,
            )
            exact_charges = charges[exact_neighbors].astype(float)
            charge_matrix[observer_local_idx, : field.size] -= np.sum(
                exact_charges[:, np.newaxis] * deposition_weights,
                axis=0,
            )
            for exact_idx, exact_charge in zip(exact_neighbors, exact_charges):
                charge_matrix[
                    observer_local_idx, source_lookup[int(exact_idx)]
                ] += float(exact_charge)

        field_local_idx = field_lookup.get(int(observer_particle_idx))
        if field_local_idx is None:
            continue
        self_location_charge = float(charge_matrix[observer_local_idx, field_local_idx])
        own_charge = float(charges[int(observer_particle_idx)])
        deposited_nonself_charge = self_location_charge - own_charge
        charge_matrix[observer_local_idx, field_local_idx] = 0.0
        if abs(deposited_nonself_charge) <= 1.0e-30:
            continue
        redistribution_weights = _fallback_self_excluded_neighbor_weights(
            active_coords[observer_local_idx],
            field_coords,
            excluded_local_idx=int(field_local_idx),
            weighting_mode=weighting_mode,
        )
        charge_matrix[observer_local_idx, : field.size] += (
            deposited_nonself_charge * redistribution_weights
        )

    return source_indices, charge_matrix, source_radii


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

    charges = np.asarray(state.get("q_source", state["q"]), dtype=float)
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

    max_index = int(max(np.max(active), np.max(neighbor_particle_indices)))
    active_index_lookup = np.full(max_index + 1, -1, dtype=int)
    active_index_lookup[active] = np.arange(active_count, dtype=int)
    neighbor_local_indices = active_index_lookup[neighbor_particle_indices]
    if np.any(neighbor_local_indices < 0):
        raise ValueError(
            "passive_map neighbor indices must be members of active_indices"
        )

    base_weight_matrix = np.zeros((passive_indices.size, active_count), dtype=float)
    passive_row_indices = np.repeat(
        np.arange(passive_indices.size, dtype=int),
        neighbor_local_indices.shape[1],
    )
    np.add.at(
        base_weight_matrix,
        (passive_row_indices, neighbor_local_indices.ravel()),
        neighbor_weights.ravel(),
    )

    base_weight_totals = np.sum(base_weight_matrix, axis=1)
    valid_passive_mask = base_weight_totals > 0.0
    base_weight_matrix[valid_passive_mask] /= base_weight_totals[
        valid_passive_mask,
        np.newaxis,
    ]

    passive_charges = charges[passive_indices].astype(float)
    valid_passive_mask &= passive_charges != 0.0

    denominators = 1.0 - base_weight_matrix
    non_fallback_mask = valid_passive_mask[:, np.newaxis] & (denominators > 1.0e-12)
    contribution_scale = np.zeros_like(base_weight_matrix)
    contribution_scale[non_fallback_mask] = (
        passive_charges[:, np.newaxis] / denominators
    )[non_fallback_mask]

    passive_contribution = contribution_scale.T @ base_weight_matrix
    np.fill_diagonal(passive_contribution, 0.0)
    charge_matrix += passive_contribution

    fallback_mask = valid_passive_mask[:, np.newaxis] & (denominators <= 1.0e-12)
    if active_count > 1 and np.any(fallback_mask):
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
        fallback_passive_rows, fallback_observer_indices = np.nonzero(fallback_mask)
        for passive_row_idx, observer_local_idx in zip(
            fallback_passive_rows,
            fallback_observer_indices,
        ):
            observer_weights = _fallback_self_excluded_neighbor_weights(
                passive_coords[passive_row_idx],
                active_coords,
                excluded_local_idx=int(observer_local_idx),
                weighting_mode=weighting_mode,
            )
            charge_matrix[observer_local_idx] += (
                passive_charges[passive_row_idx] * observer_weights
            )

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
        if "q_source" in subset:
            subset["q_source"] = q_array.copy()

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

    max_index = int(
        max(
            np.max(active),
            np.max(np.asarray(passive_map.neighbor_particle_indices, dtype=int)),
        )
    )
    active_index_lookup = np.full(max_index + 1, -1, dtype=int)
    active_index_lookup[active] = np.arange(active.size, dtype=int)
    local_neighbor_indices = active_index_lookup[
        np.asarray(passive_map.neighbor_particle_indices, dtype=int)
    ]
    if np.any(local_neighbor_indices < 0):
        raise ValueError(
            "passive_map neighbor indices must be members of active_indices"
        )

    valid_passive_mask = ~full_dead_mask[passive_indices]
    weights = np.asarray(passive_map.weights, dtype=float).copy()
    if loss_tracking_enabled and active_dead_mask.size > 0:
        alive_anchor_mask = ~active_dead_mask[local_neighbor_indices]
        weights = np.where(alive_anchor_mask, weights, 0.0)
        weight_sums = np.sum(weights, axis=1)
        valid_passive_mask &= weight_sums > 0.0
        weights[valid_passive_mask] /= weight_sums[valid_passive_mask, np.newaxis]

    if not np.any(valid_passive_mask):
        return full_state

    valid_passive_indices = passive_indices[valid_passive_mask]
    valid_neighbor_indices = local_neighbor_indices[valid_passive_mask]
    valid_weights = weights[valid_passive_mask]

    for field_name, delta_values in active_field_deltas.items():
        weighted_deltas = np.sum(
            valid_weights * delta_values[valid_neighbor_indices],
            axis=1,
        )
        full_state[field_name][valid_passive_indices] = (
            np.asarray(previous_full_state[field_name], dtype=float)[
                valid_passive_indices
            ]
            + weighted_deltas
        )

    if (
        "beta_avg_x" in previous_full_state
        and "beta_avg_y" in previous_full_state
        and "beta_avg_z" in previous_full_state
        and "beta_samples" in previous_full_state
        and "bx" in full_state
        and "by" in full_state
        and "bz" in full_state
    ):
        previous_sample_counts = np.asarray(
            previous_full_state["beta_samples"],
            dtype=float,
        )[valid_passive_indices]
        updated_sample_counts = previous_sample_counts + 1.0
        full_state["beta_samples"][valid_passive_indices] = updated_sample_counts
        for avg_field, beta_field in (
            ("beta_avg_x", "bx"),
            ("beta_avg_y", "by"),
            ("beta_avg_z", "bz"),
        ):
            previous_avg = np.asarray(previous_full_state[avg_field], dtype=float)[
                valid_passive_indices
            ]
            new_beta = np.asarray(full_state[beta_field], dtype=float)[
                valid_passive_indices
            ]
            full_state[avg_field][valid_passive_indices] = (
                previous_avg * previous_sample_counts + new_beta
            ) / updated_sample_counts

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

    rider_field = select_field_representative_indices(
        rider_state,
        rider_alive,
        rider_active,
        field_count=config.field_rider_count,
    )
    driver_field = select_field_representative_indices(
        driver_state,
        driver_alive,
        driver_active,
        field_count=config.field_driver_count,
    )

    rider_field_source_charges = accumulate_field_representative_charges(
        rider_state,
        rider_alive,
        rider_field,
        neighbor_count=config.field_deposition_neighbor_count,
        weighting_mode=config.source_weighting_mode,
    )
    driver_field_source_charges = accumulate_field_representative_charges(
        driver_state,
        driver_alive,
        driver_field,
        neighbor_count=config.field_deposition_neighbor_count,
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
        rider_field_indices=rider_field,
        driver_field_indices=driver_field,
        rider_field_source_charges=rider_field_source_charges,
        driver_field_source_charges=driver_field_source_charges,
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
    if tracker.window_steps == 0 or not tracker._last_seen:
        return penalties

    rider_lookup = {
        int(particle_idx): idx for idx, particle_idx in enumerate(rider_active)
    }
    driver_lookup = {
        int(particle_idx): idx for idx, particle_idx in enumerate(driver_active)
    }
    for (
        rider_particle_idx,
        driver_particle_idx,
    ), last_step in tracker._last_seen.items():
        age = int(step_index) - int(last_step)
        if age < 0 or age >= tracker.window_steps:
            continue
        rider_idx = rider_lookup.get(int(rider_particle_idx))
        if rider_idx is None:
            continue
        driver_idx = driver_lookup.get(int(driver_particle_idx))
        if driver_idx is None:
            continue
        penalties[rider_idx, driver_idx] = float(tracker.window_steps - age) / float(
            tracker.window_steps
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
    tolerance = max(1.0e-12, abs(best_score) * 1.0e-12)
    candidates = np.flatnonzero(scores >= best_score - tolerance)
    if candidates.size == 1:
        return int(candidates[0])
    candidate_particles = np.asarray(particle_indices, dtype=int)[candidates]
    return int(candidates[int(np.argmin(candidate_particles))])


def _collapse_passive_neighbor_weights_to_active(
    passive_indices: np.ndarray,
    active_indices: np.ndarray,
    raw_neighbor_particle_indices: np.ndarray,
    raw_weights: np.ndarray,
) -> np.ndarray:
    passive = np.asarray(passive_indices, dtype=int)
    active = np.asarray(active_indices, dtype=int)
    raw_neighbors = np.asarray(raw_neighbor_particle_indices, dtype=int)
    weights = np.asarray(raw_weights, dtype=float)
    if passive.size == 0:
        return np.zeros((0, active.size), dtype=float)

    max_index = int(max(np.max(passive), np.max(active), np.max(raw_neighbors)))
    active_lookup = np.full(max_index + 1, -1, dtype=int)
    passive_lookup = np.full(max_index + 1, -1, dtype=int)
    active_lookup[active] = np.arange(active.size, dtype=int)
    passive_lookup[passive] = np.arange(passive.size, dtype=int)

    passive_to_active = np.zeros((passive.size, active.size), dtype=float)
    passive_to_passive = np.zeros((passive.size, passive.size), dtype=float)

    row_indices = np.repeat(np.arange(passive.size, dtype=int), raw_neighbors.shape[1])
    flat_neighbors = raw_neighbors.ravel()
    flat_weights = weights.ravel()

    active_cols = active_lookup[flat_neighbors]
    active_mask = active_cols >= 0
    if np.any(active_mask):
        np.add.at(
            passive_to_active,
            (row_indices[active_mask], active_cols[active_mask]),
            flat_weights[active_mask],
        )

    passive_cols = passive_lookup[flat_neighbors]
    passive_mask = passive_cols >= 0
    if np.any(passive_mask):
        np.add.at(
            passive_to_passive,
            (row_indices[passive_mask], passive_cols[passive_mask]),
            flat_weights[passive_mask],
        )

    try:
        active_weights = np.linalg.solve(
            np.eye(passive.size, dtype=float) - passive_to_passive,
            passive_to_active,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "passive neighbour graph could not be collapsed onto active anchors"
        ) from exc

    active_weights = np.where(active_weights > 1.0e-12, active_weights, 0.0)
    row_sums = np.sum(active_weights, axis=1, keepdims=True)
    valid_rows = row_sums[:, 0] > 0.0
    active_weights[valid_rows] /= row_sums[valid_rows]
    return active_weights


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
    "accumulate_field_representative_charges",
    "accumulate_field_representative_charges_and_radii",
    "build_field_representative_space_charge_source_charges",
    "build_hybrid_space_charge_sources",
    "build_self_excluded_space_charge_source_charges",
    "build_passive_neighbor_map",
    "build_pseudo_grid_step_schedule",
    "commit_pseudo_grid_step_schedule",
    "compute_causal_history_start_index",
    "initialize_pseudo_grid_planner_state",
    "record_pseudo_grid_history_times",
    "reconstruct_full_state_from_active_result",
    "select_active_indices",
    "select_field_representative_indices",
    "slice_particle_state",
    "slice_trajectory_particle_history",
    "update_activation_history",
]
