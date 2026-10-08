"""Checkpointed internal knots and one-sided accelerations at ideal gates.

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
_EXPANSIONS = AppendAwarePreparedHistoryCache()


def save_visibility_knots(state: ParticleState, knots: Sequence[ParticleState]) -> None:
    """Publish split knots per source, preserving unrelated inclusion metadata."""
    count = len(state["t"])
    metadata = list(state.get("potential_inclusion_state", [""] * count))
    for particle in range(count):
        saved = decode_inclusion_state(str(metadata[particle]))
        saved.pop("exact_gate_history", None)
        if knots:
            saved["exact_gate_history"] = {
                "endpoint_time_ns": float(state["t"][particle]),
                "knots": [
                    {
                        **{k: float(row[k][particle]) for k in KNOT_FIELDS if k in row},
                        "before": row["_gate_prime_before"][particle].tolist(),
                        "after": row["_gate_prime_after"][particle].tolist(),
                    }
                    for row in knots
                ],
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


def _rows(history: Any) -> list[ParticleState]:
    if isinstance(history, TrialTrajectoryHistory):
        return _rows(history.base) + list(history.tail)
    if isinstance(history, (TrajectoryArrays, IndexedTrajectoryArrays)):
        return [history.state_at(i) for i in range(history.n_steps)]
    return list(history)


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
    result = []
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
                entries = [value["knots"][knot] for value in valid]
                row = dict(endpoint)
                for field in KNOT_FIELDS:
                    if field in entries[0]:
                        row[field] = np.array([entry[field] for entry in entries])
                row["_gate_prime_before"] = np.array([e["before"] for e in entries])
                row["_gate_prime_after"] = np.array([e["after"] for e in entries])
                row["potential_inclusion_state"] = [""] * len(entries)
                if result and np.array_equal(row["t"], result[-1]["t"]):
                    result[-1] = row
                else:
                    result.append(row)
        # A public endpoint can itself be a split knot.
        if result and np.array_equal(endpoint["t"], result[-1]["t"]):
            endpoint = {
                **endpoint,
                "_gate_prime_before": result[-1]["_gate_prime_before"],
                "_gate_prime_after": result[-1]["_gate_prime_after"],
            }
            result[-1] = endpoint
        else:
            result.append(endpoint)
    return result


def expanded_visibility_history(history: Any) -> list[ParticleState] | None:
    """Return a cached expanded view only when accepted gate knots exist.

    Persisted metadata is authoritative, including in a fresh interpreter.
    Rebuild on an append: neighbouring one-sided derivatives may change, but
    the original managed history and rejected trial histories stay untouched.
    """
    if isinstance(history, TrialTrajectoryHistory):
        return _expand(history)
    return _EXPANSIONS.prepare(
        history,
        variant="gate_knots",
        prepare_full=_expand,
        append=lambda previous, current, old_stop: _expand(current),
    ).value
