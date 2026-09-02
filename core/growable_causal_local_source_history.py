"""Growable transactional storage for causal local source histories.

Tentative midpoint and endpoint rows are written beyond the published prefix.
Only a successful joint commit advances the visible sample count.  A rejected
adaptive trial therefore cannot leak either role's provisional history into a
later retarded query.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Mapping, Sequence

import numpy as np

from .causal_local_source_history import (
    AcceptedLocalSourceSample,
    AcceptedPairCausalLocalSourceHistory,
    CausalLocalDipoleSource,
    CausalLocalDipoleSourceCollection,
    CausalLocalSourceHistory,
    accepted_local_source_sample_from_state,
)
from .types import TrajectoryArrays

_TOKEN_COUNTER = itertools.count(1)


def _readonly_prefix(values: np.ndarray, stop: int) -> np.ndarray:
    result = values[:stop].view()
    result.setflags(write=False)
    return result


class CausalLocalPublishedHistory:
    """A read-only visible prefix owned by one growable history."""

    def __init__(
        self,
        owner: "GrowableCausalLocalSourceHistory",
        *,
        sample_stop: int,
    ) -> None:
        self._owner = owner
        self._sample_stop = int(sample_stop)

    @property
    def time_ns(self) -> np.ndarray:
        return _readonly_prefix(self._owner._time_ns, self._sample_stop)

    @property
    def position_mm(self) -> np.ndarray:
        return _readonly_prefix(self._owner._position_mm, self._sample_stop)

    @property
    def beta(self) -> np.ndarray:
        return _readonly_prefix(self._owner._beta, self._sample_stop)

    @property
    def rest_spin(self) -> np.ndarray:
        return _readonly_prefix(self._owner._rest_spin, self._sample_stop)

    @property
    def stereographic_frame(self) -> np.ndarray:
        return self._owner.stereographic_frame

    @property
    def interval_start_beta_prime_per_mm(self) -> np.ndarray:
        return _readonly_prefix(
            self._owner._interval_start_beta_prime_per_mm,
            max(0, self._sample_stop - 1),
        )

    @property
    def interval_start_acceleration_ready(self) -> np.ndarray:
        return _readonly_prefix(
            self._owner._interval_start_acceleration_ready,
            max(0, self._sample_stop - 1),
        )

    @property
    def sample_count(self) -> int:
        return self._sample_stop

    @property
    def interval_count(self) -> int:
        return max(0, self._sample_stop - 1)

    def to_immutable(self) -> CausalLocalSourceHistory:
        return CausalLocalSourceHistory.from_accepted_samples(
            time_ns=self.time_ns,
            position_mm=self.position_mm,
            beta=self.beta,
            rest_spin=self.rest_spin,
            stereographic_frame=self.stereographic_frame,
            interval_start_beta_prime_per_mm=(self.interval_start_beta_prime_per_mm),
            interval_start_acceleration_ready=(self.interval_start_acceleration_ready),
        )


@dataclass(frozen=True)
class CausalLocalAppendTransaction:
    """A fully validated append that has not crossed the visible boundary."""

    candidate: CausalLocalPublishedHistory
    _owner_token: int
    _generation: int
    _serial: int
    _start: int
    _stop: int


class GrowableCausalLocalSourceHistory:
    """Geometrically growing source samples with explicit append commit."""

    def __init__(
        self,
        *,
        initial_capacity: int = 16,
        stereographic_frame: Sequence[Sequence[float]] | np.ndarray = np.eye(3),
    ) -> None:
        capacity = int(initial_capacity)
        if capacity < 1:
            raise ValueError("initial_capacity must be positive")
        reference = CausalLocalSourceHistory.empty(
            stereographic_frame=stereographic_frame
        )
        self.stereographic_frame = reference.stereographic_frame
        self._time_ns = np.empty(capacity, dtype=np.float64)
        self._position_mm = np.empty((capacity, 3), dtype=np.float64)
        self._beta = np.empty((capacity, 3), dtype=np.float64)
        self._rest_spin = np.empty((capacity, 3), dtype=np.float64)
        self._interval_start_beta_prime_per_mm = np.empty(
            (capacity, 3),
            dtype=np.float64,
        )
        self._interval_start_acceleration_ready = np.zeros(capacity, dtype=bool)
        self._sample_count = 0
        self._token = next(_TOKEN_COUNTER)
        self._generation = 0
        self._candidate_serial = 0

    @classmethod
    def from_history(
        cls,
        history: CausalLocalSourceHistory | CausalLocalPublishedHistory,
        *,
        minimum_capacity: int = 16,
    ) -> "GrowableCausalLocalSourceHistory":
        count = int(history.sample_count)
        capacity = max(1, int(minimum_capacity))
        while capacity < count:
            capacity *= 2
        result = cls(
            initial_capacity=capacity,
            stereographic_frame=history.stereographic_frame,
        )
        result._time_ns[:count] = history.time_ns
        result._position_mm[:count] = history.position_mm
        result._beta[:count] = history.beta
        result._rest_spin[:count] = history.rest_spin
        interval_count = max(0, count - 1)
        result._interval_start_beta_prime_per_mm[:interval_count] = (
            history.interval_start_beta_prime_per_mm
        )
        result._interval_start_acceleration_ready[:interval_count] = (
            history.interval_start_acceleration_ready
        )
        result._sample_count = count
        return result

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def allocated_capacity(self) -> int:
        return int(self._time_ns.size)

    def build_current(self) -> CausalLocalPublishedHistory:
        return CausalLocalPublishedHistory(self, sample_stop=self._sample_count)

    def _ensure_capacity(self, required: int) -> None:
        if required <= self.allocated_capacity:
            return
        capacity = self.allocated_capacity
        while capacity < required:
            capacity *= 2
        replacements = (
            ("_time_ns", (capacity,)),
            ("_position_mm", (capacity, 3)),
            ("_beta", (capacity, 3)),
            ("_rest_spin", (capacity, 3)),
            ("_interval_start_beta_prime_per_mm", (capacity, 3)),
            ("_interval_start_acceleration_ready", (capacity,)),
        )
        for name, shape in replacements:
            old = getattr(self, name)
            new = np.empty(shape, dtype=old.dtype)
            visible = (
                max(0, self._sample_count - 1)
                if name.startswith("_interval_start")
                else self._sample_count
            )
            new[:visible] = old[:visible]
            setattr(self, name, new)

    def preflight_append_samples(
        self,
        samples: Sequence[AcceptedLocalSourceSample],
    ) -> CausalLocalAppendTransaction:
        """Write a validated candidate without publishing its new intervals."""

        rows = tuple(samples)
        self._candidate_serial += 1
        if not rows:
            raise ValueError("accepted source sample sequence must not be empty")
        if self._sample_count < 1:
            raise ValueError("interval endpoints need an existing start sample")
        times = np.asarray([sample.time_ns for sample in rows], dtype=np.float64)
        if (
            np.any(np.diff(times) <= 0.0)
            or times[0] <= self._time_ns[self._sample_count - 1]
        ):
            raise ValueError("accepted source times must increase strictly")
        positions = np.asarray(
            [sample.position_mm for sample in rows], dtype=np.float64
        )
        velocities = np.asarray([sample.beta for sample in rows], dtype=np.float64)
        spins = np.asarray([sample.rest_spin for sample in rows], dtype=np.float64)
        accelerations = np.asarray(
            [sample.interval_start_beta_prime_per_mm for sample in rows],
            dtype=np.float64,
        )
        ready = np.asarray(
            [sample.interval_start_acceleration_ready for sample in rows],
            dtype=bool,
        )
        start = self._sample_count
        stop = start + len(rows)
        self._ensure_capacity(stop)
        self._time_ns[start:stop] = times
        self._position_mm[start:stop] = positions
        self._beta[start:stop] = velocities
        self._rest_spin[start:stop] = spins
        self._interval_start_beta_prime_per_mm[start - 1 : stop - 1] = accelerations
        self._interval_start_acceleration_ready[start - 1 : stop - 1] = ready
        candidate = CausalLocalPublishedHistory(self, sample_stop=stop)
        return CausalLocalAppendTransaction(
            candidate=candidate,
            _owner_token=self._token,
            _generation=self._generation,
            _serial=self._candidate_serial,
            _start=start,
            _stop=stop,
        )

    def can_commit(self, transaction: CausalLocalAppendTransaction) -> bool:
        return bool(
            transaction._owner_token == self._token
            and transaction._generation == self._generation
            and transaction._serial == self._candidate_serial
            and transaction._start == self._sample_count
        )

    def commit(
        self,
        transaction: CausalLocalAppendTransaction,
    ) -> CausalLocalPublishedHistory:
        if not self.can_commit(transaction):
            raise RuntimeError("causal local append transaction is stale or foreign")
        self._sample_count = transaction._stop
        self._generation += 1
        return self.build_current()


@dataclass(frozen=True)
class GrowableCausalLocalCollectionTransaction:
    """All source appends preflighted for one ordered collection."""

    candidate: CausalLocalDipoleSourceCollection
    source_transactions: tuple[CausalLocalAppendTransaction, ...]


class GrowableCausalLocalDipoleSourceCollection:
    """Stable source metadata plus growable accepted local histories."""

    def __init__(
        self,
        *,
        identities: Sequence[str],
        particle_indices: Sequence[int],
        magnetic_moments_native: Sequence[float],
        histories: Sequence[GrowableCausalLocalSourceHistory],
    ) -> None:
        self.identities = tuple(str(value) for value in identities)
        self.particle_indices = tuple(int(value) for value in particle_indices)
        self.magnetic_moments_native = tuple(
            float(value) for value in magnetic_moments_native
        )
        self.histories = tuple(histories)
        size = len(self.identities)
        if not (
            len(self.particle_indices)
            == len(self.magnetic_moments_native)
            == len(self.histories)
            == size
        ):
            raise ValueError("growable causal local source metadata lengths must match")
        if len(set(self.identities)) != size:
            raise ValueError("growable causal local source identities must be unique")
        if len(set(self.particle_indices)) != size:
            raise ValueError("growable causal local particle indices must be unique")
        if any(index < 0 for index in self.particle_indices):
            raise ValueError("growable causal local indices must be non-negative")
        if any(
            not np.isfinite(moment) or moment == 0.0
            for moment in self.magnetic_moments_native
        ):
            raise ValueError("growable causal local moments must be finite and nonzero")

    @classmethod
    def from_collection(
        cls,
        collection: CausalLocalDipoleSourceCollection,
    ) -> "GrowableCausalLocalDipoleSourceCollection":
        histories = []
        for source in collection.sources:
            history = source.history
            if isinstance(history, CausalLocalPublishedHistory):
                immutable = history.to_immutable()
            elif isinstance(history, CausalLocalSourceHistory):
                immutable = history
            else:
                raise TypeError("unsupported causal local source history")
            histories.append(GrowableCausalLocalSourceHistory.from_history(immutable))
        return cls(
            identities=collection.source_identities,
            particle_indices=tuple(
                source.particle_index for source in collection.sources
            ),
            magnetic_moments_native=tuple(
                source.magnetic_moment_native for source in collection.sources
            ),
            histories=tuple(histories),
        )

    @classmethod
    def from_trajectory_arrays(
        cls,
        trajectory: TrajectoryArrays,
        *,
        identity_prefix: str,
    ) -> "GrowableCausalLocalDipoleSourceCollection":
        return cls.from_collection(
            CausalLocalDipoleSourceCollection.from_trajectory_arrays(
                trajectory,
                identity_prefix=identity_prefix,
            )
        )

    def build_current(self) -> CausalLocalDipoleSourceCollection:
        return CausalLocalDipoleSourceCollection(
            tuple(
                CausalLocalDipoleSource(
                    identity=identity,
                    particle_index=particle,
                    magnetic_moment_native=moment,
                    history=history.build_current(),
                )
                for identity, particle, moment, history in zip(
                    self.identities,
                    self.particle_indices,
                    self.magnetic_moments_native,
                    self.histories,
                )
            )
        )

    def preflight_append_states(
        self,
        states: Sequence[Mapping[str, object]],
    ) -> GrowableCausalLocalCollectionTransaction:
        rows = tuple(states)
        if not rows:
            raise ValueError("accepted source state sequence must not be empty")
        transactions: list[CausalLocalAppendTransaction] = []
        candidate_sources: list[CausalLocalDipoleSource] = []
        for identity, particle, moment, history in zip(
            self.identities,
            self.particle_indices,
            self.magnetic_moments_native,
            self.histories,
        ):
            for state in rows:
                if "magnetic_moment_native" in state:
                    values = np.asarray(
                        state["magnetic_moment_native"], dtype=np.float64
                    )
                    if float(values[particle]) != moment:
                        raise ValueError(
                            f"source identity {identity!r} changed magnetic moment"
                        )
            samples = tuple(
                accepted_local_source_sample_from_state(state, particle)
                for state in rows
            )
            transaction = history.preflight_append_samples(samples)
            transactions.append(transaction)
            candidate_sources.append(
                CausalLocalDipoleSource(
                    identity=identity,
                    particle_index=particle,
                    magnetic_moment_native=moment,
                    history=transaction.candidate,
                )
            )
        return GrowableCausalLocalCollectionTransaction(
            candidate=CausalLocalDipoleSourceCollection(tuple(candidate_sources)),
            source_transactions=tuple(transactions),
        )

    def can_commit(
        self,
        transaction: GrowableCausalLocalCollectionTransaction,
    ) -> bool:
        return bool(
            len(transaction.source_transactions) == len(self.histories)
            and all(
                history.can_commit(source_transaction)
                for history, source_transaction in zip(
                    self.histories,
                    transaction.source_transactions,
                )
            )
        )

    def commit(
        self,
        transaction: GrowableCausalLocalCollectionTransaction,
    ) -> CausalLocalDipoleSourceCollection:
        if not self.can_commit(transaction):
            raise RuntimeError("growable causal local collection transaction is stale")
        for history, source_transaction in zip(
            self.histories,
            transaction.source_transactions,
        ):
            history.commit(source_transaction)
        return self.build_current()


@dataclass(frozen=True)
class GrowableAcceptedPairCausalLocalTransaction:
    """A rider/driver candidate with both collection commits preflighted."""

    candidate: AcceptedPairCausalLocalSourceHistory
    rider: GrowableCausalLocalCollectionTransaction
    driver: GrowableCausalLocalCollectionTransaction


class GrowableAcceptedPairCausalLocalSourceHistory:
    """Transaction owner for two jointly accepted local source collections."""

    def __init__(
        self,
        *,
        rider: GrowableCausalLocalDipoleSourceCollection,
        driver: GrowableCausalLocalDipoleSourceCollection,
    ) -> None:
        self.rider = rider
        self.driver = driver

    @classmethod
    def from_accepted(
        cls,
        accepted: AcceptedPairCausalLocalSourceHistory,
    ) -> "GrowableAcceptedPairCausalLocalSourceHistory":
        return cls(
            rider=GrowableCausalLocalDipoleSourceCollection.from_collection(
                accepted.rider
            ),
            driver=GrowableCausalLocalDipoleSourceCollection.from_collection(
                accepted.driver
            ),
        )

    @classmethod
    def from_trajectory_arrays(
        cls,
        rider: TrajectoryArrays,
        driver: TrajectoryArrays,
    ) -> "GrowableAcceptedPairCausalLocalSourceHistory":
        return cls.from_accepted(
            AcceptedPairCausalLocalSourceHistory.from_trajectory_arrays(rider, driver)
        )

    def build_current(self) -> AcceptedPairCausalLocalSourceHistory:
        return AcceptedPairCausalLocalSourceHistory(
            rider=self.rider.build_current(),
            driver=self.driver.build_current(),
        )

    def preflight_states(
        self,
        *,
        rider_states: Sequence[Mapping[str, object]],
        driver_states: Sequence[Mapping[str, object]],
    ) -> GrowableAcceptedPairCausalLocalTransaction:
        rider = self.rider.preflight_append_states(rider_states)
        driver = self.driver.preflight_append_states(driver_states)
        return GrowableAcceptedPairCausalLocalTransaction(
            candidate=AcceptedPairCausalLocalSourceHistory(
                rider=rider.candidate,
                driver=driver.candidate,
            ),
            rider=rider,
            driver=driver,
        )

    def can_commit(
        self,
        transaction: GrowableAcceptedPairCausalLocalTransaction,
    ) -> bool:
        return self.rider.can_commit(transaction.rider) and self.driver.can_commit(
            transaction.driver
        )

    def commit(
        self,
        transaction: GrowableAcceptedPairCausalLocalTransaction,
    ) -> AcceptedPairCausalLocalSourceHistory:
        if not self.can_commit(transaction):
            raise RuntimeError("growable causal local pair transaction is stale")
        self.rider.commit(transaction.rider)
        self.driver.commit(transaction.driver)
        return self.build_current()


__all__ = [
    "CausalLocalAppendTransaction",
    "CausalLocalPublishedHistory",
    "GrowableAcceptedPairCausalLocalSourceHistory",
    "GrowableAcceptedPairCausalLocalTransaction",
    "GrowableCausalLocalCollectionTransaction",
    "GrowableCausalLocalDipoleSourceCollection",
    "GrowableCausalLocalSourceHistory",
]
