"""Checkpointed split knots at ideal gates and accepted photon recoils.

Public output keeps its requested cadence. Inclusion metadata on an accepted
endpoint publishes its internal source knots to charge and dipole providers.
The same metadata survives trajectory reconstruction and checkpoint reload.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .potential_inclusion import decode_inclusion_state, encode_inclusion_state
from .prepared_history_cache import AppendAwarePreparedHistoryCache
from .resolved_knot import RESOLVED_KNOT_FIELDS
from .types import (
    IndexedTrajectoryArrays,
    ParticleState,
    TrajectoryArrays,
    TrialTrajectoryHistory,
)

KNOT_FIELDS = (
    "t",
    "x",
    "y",
    "z",
    "gamma",
    "bx",
    "by",
    "bz",
    "bdotx",
    "bdoty",
    "bdotz",
    "spin_x",
    "spin_y",
    "spin_z",
    *RESOLVED_KNOT_FIELDS,
)
_EXPANSIONS: AppendAwarePreparedHistoryCache[Any, list[ParticleState] | None] = (
    AppendAwarePreparedHistoryCache(preserve_capacity_growth=True)
)


def save_visibility_knots(
    state: ParticleState, knots: Sequence[ParticleState], *, append: bool = False
) -> None:
    """Publish split knots per source, preserving unrelated inclusion metadata."""
    count = len(state["t"])
    metadata = list(state.get("potential_inclusion_state", [""] * count))
    saved_states = [decode_inclusion_state(str(value)) for value in metadata]
    existing = [saved.pop("exact_gate_history", None) for saved in saved_states]
    keep_existing = append and all(
        value is not None and value["endpoint_time_ns"] == float(state["t"][i])
        for i, value in enumerate(existing)
    )
    # A frozen dead clock can also match stale metadata from an earlier step.
    # Preserve the prior batch only when it belongs to this whole endpoint.
    entries_by_particle = [
        list(value["knots"]) if keep_existing and value is not None else []
        for value in existing
    ]
    # Knot indices identify bunch-wide events. Frozen times after particle
    # loss must not make just one particle merge an interior gate with a kick.
    live = np.flatnonzero(
        ~np.asarray(state.get("_dead_particles", np.zeros(count, dtype=bool)))
    )
    deciding = live if live.size else range(count)
    merge_rows = []
    last_times = [
        entries[-1]["t"] if entries else None for entries in entries_by_particle
    ]
    for row in knots:
        merge_rows.append(
            append
            and all(last is not None for last in last_times)
            and all(last_times[i] == float(row["t"][i]) for i in deciding)
        )
        last_times = [float(value) for value in row["t"]]
    for particle, saved in enumerate(saved_states):
        if knots:
            entries = entries_by_particle[particle]
            for row, merge in zip(knots, merge_rows):
                entry: dict[str, Any] = {
                    k: float(row[k][particle]) for k in KNOT_FIELDS if k in row
                }
                for side in ("before", "after"):
                    prime = row[f"_gate_prime_{side}"][particle]
                    entry[side] = prime.tolist() if np.all(np.isfinite(prime)) else None
                if row.get("_kick_mask", np.zeros(count, dtype=bool))[particle]:
                    entry["velocity_before"] = row["_kick_beta_before"][
                        particle
                    ].tolist()
                    entry["resolved_before"] = row["_kick_resolved_before"][
                        particle
                    ].tolist()
                if merge:
                    # A recoil at a gate endpoint keeps the incoming gate
                    # branch and starts a fresh outgoing velocity branch.
                    # Its old after-prime used the pre-recoil velocity and
                    # must be reconstructed after a kick, not preserved.
                    old = entries[-1]
                    if old.get("before") is not None:
                        entry["before"] = old["before"]
                    if "velocity_before" not in entry and old.get("after") is not None:
                        entry["after"] = old["after"]
                    for field in ("velocity_before", "resolved_before"):
                        if field in old:
                            entry[field] = old[field]
                    entries[-1] = entry
                else:
                    entries.append(entry)
            saved["exact_gate_history"] = {
                "endpoint_time_ns": float(state["t"][particle]),
                "knots": entries,
            }
        metadata[particle] = encode_inclusion_state(saved) if saved else ""
    state["potential_inclusion_state"] = np.asarray(metadata, dtype=str)


def _metadata_rows(history: Any) -> Any:
    if isinstance(history, TrialTrajectoryHistory):
        return list(_metadata_rows(history.base)) + [
            row.get("potential_inclusion_state", []) for row in history.tail
        ]
    if isinstance(history, IndexedTrajectoryArrays):
        history.base.require_current_storage()
        return history.base.potential_inclusion_state[history.start_step :][
            :, history.particle_indices
        ]
    if isinstance(history, TrajectoryArrays):
        history.require_current_storage()
        return history.potential_inclusion_state
    return [row.get("potential_inclusion_state", []) for row in history]


def _rows(history: Any, start: int = 0) -> list[ParticleState]:
    if isinstance(history, TrialTrajectoryHistory):
        return (_rows(history.base) + list(history.tail))[start:]
    if isinstance(history, (TrajectoryArrays, IndexedTrajectoryArrays)):
        return [history.state_at(i) for i in range(start, history.n_steps)]
    return list(history[start:])


def _expand(history: Any) -> list[ParticleState] | None:
    saved_rows = []
    found = False
    for metadata in _metadata_rows(history):
        saved = [
            decode_inclusion_state(str(value)).get("exact_gate_history")
            for value in metadata
        ]
        found |= any(value is not None for value in saved)
        saved_rows.append(saved)
    if not found:
        return None
    rows = _rows(history)
    result: list[ParticleState] = []
    for endpoint, saved in zip(rows, saved_rows):
        valid = [
            (
                value
                if value is not None
                and value["endpoint_time_ns"] == float(endpoint["t"][i])
                else None
            )
            for i, value in enumerate(saved)
        ]
        lengths = {len(value["knots"]) for value in valid if value is not None}
        if lengths:
            if len(lengths) != 1 or any(value is None for value in valid):
                raise ValueError("gate source knots must be aligned across a bunch")
            for knot in range(next(iter(lengths))):
                entries = [value["knots"][knot] for value in valid if value is not None]
                row = dict(endpoint)
                for field in KNOT_FIELDS:
                    if field in entries[0]:
                        row[field] = np.array([entry[field] for entry in entries])
                for side in ("before", "after"):
                    row[f"_gate_prime_{side}"] = np.array(
                        [
                            e[side] if e[side] is not None else [np.nan] * 3
                            for e in entries
                        ]
                    )
                if any("velocity_before" in e for e in entries):
                    row["_kick_beta_before"] = np.array(
                        [e.get("velocity_before", [np.nan] * 3) for e in entries]
                    )
                    row["_kick_resolved_before"] = np.array(
                        [
                            e.get(
                                "resolved_before", [np.nan] * len(RESOLVED_KNOT_FIELDS)
                            )
                            for e in entries
                        ]
                    )
                row["potential_inclusion_state"] = np.full(len(entries), "", dtype=str)
                if result and np.array_equal(row["t"], result[-1]["t"]):
                    result[-1] = row
                else:
                    result.append(row)
        # A public endpoint can itself be a split knot.
        if result and np.array_equal(endpoint["t"], result[-1]["t"]):
            endpoint = dict(endpoint)
            for field in (
                "_gate_prime_before",
                "_gate_prime_after",
                "_kick_beta_before",
                "_kick_resolved_before",
            ):
                if field in result[-1]:
                    endpoint[field] = result[-1][field]
            result[-1] = endpoint
        else:
            result.append(endpoint)
    return result


def expanded_visibility_history(history: Any) -> list[ParticleState] | None:
    """Return a cached expanded view only when accepted split knots exist.

    Persisted metadata is authoritative, including in a fresh interpreter.
    Only newly accepted rows are decoded on append. Prepared providers decide
    which neighbouring one-sided derivatives must be reconstructed.
    """
    if isinstance(history, TrialTrajectoryHistory):
        return _expand(history)
    return _EXPANSIONS.prepare(
        history,
        variant="gate_knots",
        prepare_full=_expand,
        append=_append_expansion,
    ).value


def _append_expansion(
    previous: list[ParticleState] | None, current: Any, old_stop: int
) -> list[ParticleState] | None:
    tail = _rows(current, old_stop)
    expanded_tail = _expand(tail)
    if previous is None:
        if expanded_tail is None:
            return None
        previous = _rows(current, 0)[:old_stop]
    tail = tail if expanded_tail is None else expanded_tail
    if tail and previous and np.array_equal(previous[-1]["t"], tail[0]["t"]):
        # Retain the full coalescing semantics for coincident bunch endpoints.
        return _expand(current)
    previous.extend(tail)
    return previous


def split_free_tail(rows: Sequence[ParticleState], start: int) -> bool:
    """Whether the append's three-knot derivative window avoids every split."""
    return not any(
        "_gate_prime_before" in row or "_kick_beta_before" in row
        for row in rows[max(0, start - 3) :]
    )
