"""Safe append-aware caching for prepared retarded source histories.

Only :class:`~core.types.TrajectoryArrays` views created by one
:class:`~core.types.TrajectoryBuilder` are cacheable.  Those views share a
builder-owned allocation token and live mutation counters, so this module never
uses the identity of a short-lived ``TrajectoryArrays`` wrapper as a key.

The cache reuses an entry at the same visible stop, extends it when the same
storage grows without a rewrite, and rebuilds it after a rewrite or a request
for a shorter view.  Manually constructed arrays and legacy list histories are
prepared normally on every call.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Generic, Hashable, Literal, TypeVar

import numpy as np

from .types import IndexedTrajectoryArrays, TrajectoryArrays

PreparedT = TypeVar("PreparedT")
HistoryT = TypeVar("HistoryT")
CacheDisposition = Literal["uncached", "miss", "reuse", "append", "rebuild"]


@dataclass(frozen=True)
class HistoryStorageSnapshot:
    """Cache identity and live version for one visible history view."""

    identity: tuple[Hashable, ...]
    visible_stop: int
    generation: int
    rewrite_epoch: int


@dataclass(frozen=True)
class PreparedHistoryCacheResult(Generic[PreparedT]):
    """Prepared value plus the mutation decision that produced it.

    ``revision`` changes after both an append and a full rebuild.  A caller
    that also caches observer-event field results can include this revision in
    its result key; accepted source-history appends can therefore never reuse a
    field result computed from the shorter history.
    """

    value: PreparedT
    disposition: CacheDisposition
    revision: int


@dataclass(frozen=True)
class PreparedHistoryCacheStats:
    uncached: int
    misses: int
    reuses: int
    appends: int
    rebuilds: int


@dataclass
class _CacheEntry(Generic[PreparedT]):
    value: PreparedT
    visible_stop: int
    generation: int
    rewrite_epoch: int
    revision: int


def _array_value_signature(values: np.ndarray) -> tuple[Hashable, ...]:
    """Return a value signature without depending on the array object's id."""

    array = np.ascontiguousarray(values)
    return (array.dtype.str, tuple(int(size) for size in array.shape), array.tobytes())


def history_storage_snapshot(
    history: object,
) -> HistoryStorageSnapshot | None:
    """Return builder storage metadata, or ``None`` for unmanaged histories."""

    if isinstance(history, IndexedTrajectoryArrays):
        base = history.base
        token = base.storage_token
        generation = base.storage_generation
        rewrite_epoch = base.storage_rewrite_epoch
        if token is None or generation is None or rewrite_epoch is None:
            return None
        q_signature: tuple[Hashable, ...] | None = None
        if history.q_override is not None:
            q_signature = _array_value_signature(
                np.asarray(history.q_override, dtype=float)
            )
        identity: tuple[Hashable, ...] = (
            "indexed_trajectory_arrays",
            token,
            int(history.start_step),
            _array_value_signature(np.asarray(history.particle_indices, dtype=int)),
            q_signature,
        )
        return HistoryStorageSnapshot(
            identity=identity,
            visible_stop=history.n_steps,
            generation=generation,
            rewrite_epoch=rewrite_epoch,
        )

    if isinstance(history, TrajectoryArrays):
        token = history.storage_token
        generation = history.storage_generation
        rewrite_epoch = history.storage_rewrite_epoch
        if token is None or generation is None or rewrite_epoch is None:
            return None
        return HistoryStorageSnapshot(
            identity=("trajectory_arrays", token),
            visible_stop=history.n_steps,
            generation=generation,
            rewrite_epoch=rewrite_epoch,
        )

    return None


def history_storage_capacity(history: object) -> int | None:
    """Return usable builder row capacity for a managed history view."""

    if isinstance(history, IndexedTrajectoryArrays):
        capacity = history.base.storage_capacity
        if capacity is None:
            return None
        return max(0, int(capacity) - int(history.start_step))
    if isinstance(history, TrajectoryArrays):
        capacity = history.storage_capacity
        return None if capacity is None else int(capacity)
    return None


class AppendAwarePreparedHistoryCache(Generic[HistoryT, PreparedT]):
    """Bounded cache that distinguishes tail growth from history rewrites."""

    def __init__(self, *, max_entries: int = 16) -> None:
        entries = int(max_entries)
        if entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = entries
        self._entries: OrderedDict[
            tuple[tuple[Hashable, ...], Hashable], _CacheEntry[PreparedT]
        ] = OrderedDict()
        self._lock = RLock()
        self._uncached = 0
        self._misses = 0
        self._reuses = 0
        self._appends = 0
        self._rebuilds = 0

    def clear(self) -> None:
        """Drop prepared values and reset counters."""

        with self._lock:
            self._entries.clear()
            self._uncached = 0
            self._misses = 0
            self._reuses = 0
            self._appends = 0
            self._rebuilds = 0

    def stats(self) -> PreparedHistoryCacheStats:
        """Return an immutable counter snapshot."""

        with self._lock:
            return PreparedHistoryCacheStats(
                uncached=self._uncached,
                misses=self._misses,
                reuses=self._reuses,
                appends=self._appends,
                rebuilds=self._rebuilds,
            )

    def prepare(
        self,
        history: HistoryT,
        *,
        variant: Hashable,
        prepare_full: Callable[[HistoryT], PreparedT],
        append: Callable[[PreparedT, HistoryT, int], PreparedT],
    ) -> PreparedHistoryCacheResult[PreparedT]:
        """Prepare, reuse, extend, or rebuild one history variant safely."""

        try:
            hash(variant)
        except TypeError as exc:
            raise TypeError("prepared-history cache variant must be hashable") from exc

        snapshot = history_storage_snapshot(history)
        if snapshot is None:
            value = prepare_full(history)
            with self._lock:
                self._uncached += 1
            return PreparedHistoryCacheResult(value, "uncached", 0)

        key = (snapshot.identity, variant)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                value = prepare_full(history)
                entry = _CacheEntry(
                    value=value,
                    visible_stop=snapshot.visible_stop,
                    generation=snapshot.generation,
                    rewrite_epoch=snapshot.rewrite_epoch,
                    revision=1,
                )
                self._entries[key] = entry
                self._entries.move_to_end(key)
                self._trim()
                self._misses += 1
                return PreparedHistoryCacheResult(value, "miss", entry.revision)

            disposition: CacheDisposition
            if (
                snapshot.rewrite_epoch != entry.rewrite_epoch
                or snapshot.visible_stop < entry.visible_stop
                or (
                    snapshot.visible_stop > entry.visible_stop
                    and snapshot.generation <= entry.generation
                )
            ):
                value = prepare_full(history)
                entry.value = value
                entry.visible_stop = snapshot.visible_stop
                entry.generation = snapshot.generation
                entry.rewrite_epoch = snapshot.rewrite_epoch
                entry.revision += 1
                self._rebuilds += 1
                disposition = "rebuild"
            elif snapshot.visible_stop > entry.visible_stop:
                try:
                    value = append(entry.value, history, entry.visible_stop)
                except BaseException:
                    # Provider appenders may update preallocated buffers in
                    # place. Never retain a possibly partial mutation after a
                    # validation or numerical failure.
                    del self._entries[key]
                    raise
                entry.value = value
                entry.visible_stop = snapshot.visible_stop
                entry.generation = snapshot.generation
                entry.rewrite_epoch = snapshot.rewrite_epoch
                entry.revision += 1
                self._appends += 1
                disposition = "append"
            else:
                # Builder writes outside this shorter published view do not
                # alter its contents, so a generation-only change remains a
                # valid exact hit.
                entry.generation = snapshot.generation
                self._reuses += 1
                disposition = "reuse"

            self._entries.move_to_end(key)
            return PreparedHistoryCacheResult(entry.value, disposition, entry.revision)

    def _trim(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


__all__ = [
    "AppendAwarePreparedHistoryCache",
    "HistoryStorageSnapshot",
    "PreparedHistoryCacheResult",
    "PreparedHistoryCacheStats",
    "history_storage_capacity",
    "history_storage_snapshot",
]
