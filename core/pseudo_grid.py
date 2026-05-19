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

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import KDTree

from .constants import C_MMNS
from .types import ParticleState


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
        return PassiveNeighborMap(
            passive_indices=np.zeros(0, dtype=int),
            neighbor_particle_indices=np.zeros((0, 0), dtype=int),
            weights=np.zeros((0, 0), dtype=float),
        )

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


__all__ = [
    "PassiveNeighborMap",
    "PairReuseTracker",
    "accumulate_effective_source_charges",
    "build_passive_neighbor_map",
    "compute_causal_history_start_index",
    "select_active_indices",
    "update_activation_history",
]
