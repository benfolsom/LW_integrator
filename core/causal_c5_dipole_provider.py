"""Ordered multi-source dispatch for causally frozen dipole histories.

The objects in this module remain an opt-in validation surface.  The exact-pair
adaptive integrator may now use them for the ordinary field sourced by
intrinsic magnetic moments; this remains separate from the not-yet-enabled
intrinsic-spin self-reaction force.  Their production contracts are:

* source identity and summation order are stable;
* an adaptive trial produces a detached candidate and publishes it only after
  the rider and driver path is jointly accepted; and
* checkpoint restoration reuses frozen coefficients instead of refitting the
  past.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from .causal_c5_source_history import (
    CausalC5HistoryUnavailableError,
    CausalC5SourceHistory,
    FrozenC5SourceSegment,
)
from .dipole_hertz_jet import (
    DipoleHertzResponseJetResult,
    evaluate_causal_c5_dipole_hertz_response_native,
)
from .growable_causal_c5_source_history import (
    CausalC5AppendTransaction,
    CausalC5PublishedHistory,
    GrowableCausalC5SourceHistory,
)
from .rfs import fields_from_tensor_native
from .types import ParticleState, TrajectoryArrays

if TYPE_CHECKING:
    from .exact_pair_trial import ExactPairStepDoublingTrial
    from .retarded_fields import ObserverEvent


def _deterministic_stereographic_frame(rest_spin: np.ndarray) -> np.ndarray:
    """Choose a fixed right-handed chart with the initial spin at its north pole."""

    spin = np.asarray(rest_spin, dtype=np.float64)
    if spin.shape != (3,) or not np.all(np.isfinite(spin)):
        raise ValueError("initial rest spin must be a finite three-vector")
    magnitude = float(np.linalg.norm(spin))
    if not np.isclose(magnitude, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("initial rest spin must have unit magnitude")
    north = spin / magnitude
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ north)))]
    first = np.cross(reference, north)
    first /= np.linalg.norm(first)
    second = np.cross(north, first)
    frame = np.column_stack((first, second, north))
    frame.setflags(write=False)
    return frame


def _state_vector(
    state: Mapping[str, object],
    names: tuple[str, str, str],
    particle_index: int,
) -> np.ndarray:
    values = np.asarray(
        [np.asarray(state[name], dtype=np.float64)[particle_index] for name in names],
        dtype=np.float64,
    )
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"accepted source state has invalid components {names!r}")
    return values


def _state_scalar(
    state: Mapping[str, object],
    name: str,
    particle_index: int,
) -> float:
    values = np.asarray(state[name], dtype=np.float64)
    if values.ndim != 1 or particle_index >= values.size:
        raise ValueError(f"accepted source state has no {name} value")
    result = float(values[particle_index])
    if not np.isfinite(result):
        raise ValueError(f"accepted source state has non-finite {name}")
    return result


@dataclass(frozen=True)
class CausalC5DipoleSource:
    """One stable source identity, constant moment, and accepted history."""

    identity: str
    particle_index: int
    magnetic_moment_native: float
    history: CausalC5SourceHistory | CausalC5PublishedHistory

    def __post_init__(self) -> None:
        identity = str(self.identity)
        particle_index = int(self.particle_index)
        moment = float(self.magnetic_moment_native)
        if not identity:
            raise ValueError("causal C5 source identity must not be empty")
        if particle_index < 0:
            raise ValueError("causal C5 source particle index must be non-negative")
        if not np.isfinite(moment) or moment == 0.0:
            raise ValueError("causal C5 source moment must be finite and nonzero")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "particle_index", particle_index)
        object.__setattr__(self, "magnetic_moment_native", moment)

    def append_accepted_state(
        self,
        state: Mapping[str, object],
    ) -> "CausalC5DipoleSource":
        """Return a detached source with one additional accepted sample."""

        particle = self.particle_index
        if "magnetic_moment_native" in state:
            moment = _state_scalar(state, "magnetic_moment_native", particle)
            if moment != self.magnetic_moment_native:
                raise ValueError(
                    f"source identity {self.identity!r} changed magnetic moment"
                )
        history = self.history.append_accepted(
            time_ns=_state_scalar(state, "t", particle),
            position_mm=_state_vector(state, ("x", "y", "z"), particle),
            beta=_state_vector(state, ("bx", "by", "bz"), particle),
            beta_prime_per_mm=_state_vector(
                state,
                ("bdotx", "bdoty", "bdotz"),
                particle,
            ),
            rest_spin=_state_vector(
                state,
                ("spin_x", "spin_y", "spin_z"),
                particle,
            ),
        )
        return CausalC5DipoleSource(
            identity=self.identity,
            particle_index=particle,
            magnetic_moment_native=self.magnetic_moment_native,
            history=history,
        )


@dataclass(frozen=True)
class CausalC5DipoleSourceCollection:
    """A fixed, ordered set of magnetic sources.

    The tuple order is the floating-point summation order and is therefore part
    of the checkpointed numerical model.  Sources are never sorted implicitly.
    """

    sources: tuple[CausalC5DipoleSource, ...] = ()

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        identities = tuple(source.identity for source in sources)
        particle_indices = tuple(source.particle_index for source in sources)
        if len(set(identities)) != len(identities):
            raise ValueError("causal C5 source identities must be unique")
        if len(set(particle_indices)) != len(particle_indices):
            raise ValueError("causal C5 source particle indices must be unique")
        object.__setattr__(self, "sources", sources)

    @property
    def source_identities(self) -> tuple[str, ...]:
        return tuple(source.identity for source in self.sources)

    @classmethod
    def from_trajectory_arrays(
        cls,
        trajectory: TrajectoryArrays,
        *,
        identity_prefix: str,
        particle_indices: Sequence[int] | None = None,
        source_identities: Sequence[str] | None = None,
        stereographic_frames: Sequence[np.ndarray] | None = None,
        frozen_segments: Sequence[Sequence[FrozenC5SourceSegment]] | None = None,
    ) -> "CausalC5DipoleSourceCollection":
        """Construct all selected sources from one accepted trajectory prefix."""

        trajectory.require_current_storage()
        if particle_indices is None:
            active = np.asarray(trajectory.magnetic_dipole_active, dtype=bool)
            moments = np.asarray(trajectory.magnetic_moment_native, dtype=np.float64)
            indices = tuple(
                int(index)
                for index in np.flatnonzero(
                    active & np.isfinite(moments) & (moments != 0)
                )
            )
        else:
            indices = tuple(int(index) for index in particle_indices)
        if any(index < 0 or index >= trajectory.n_particles for index in indices):
            raise ValueError("causal C5 source particle index is out of bounds")
        identities = (
            tuple(str(value) for value in source_identities)
            if source_identities is not None
            else tuple(f"{identity_prefix}:{index}" for index in indices)
        )
        if len(identities) != len(indices):
            raise ValueError("source identities must match selected particle indices")
        if stereographic_frames is not None and len(stereographic_frames) != len(
            indices
        ):
            raise ValueError("stereographic frames must match selected sources")
        if frozen_segments is not None and len(frozen_segments) != len(indices):
            raise ValueError("frozen segment sets must match selected sources")

        sources: list[CausalC5DipoleSource] = []
        for sequence_index, (identity, particle) in enumerate(zip(identities, indices)):
            spin = np.column_stack(
                (
                    trajectory.spin_x[:, particle],
                    trajectory.spin_y[:, particle],
                    trajectory.spin_z[:, particle],
                )
            )
            frame = (
                _deterministic_stereographic_frame(spin[0])
                if stereographic_frames is None
                else np.asarray(stereographic_frames[sequence_index], dtype=np.float64)
            )
            supplied_segments = (
                None
                if frozen_segments is None
                else tuple(frozen_segments[sequence_index])
            )
            history = CausalC5SourceHistory.from_accepted_samples(
                time_ns=trajectory.t[:, particle],
                position_mm=np.column_stack(
                    (
                        trajectory.x[:, particle],
                        trajectory.y[:, particle],
                        trajectory.z[:, particle],
                    )
                ),
                beta=np.column_stack(
                    (
                        trajectory.bx[:, particle],
                        trajectory.by[:, particle],
                        trajectory.bz[:, particle],
                    )
                ),
                beta_prime_per_mm=np.column_stack(
                    (
                        trajectory.bdotx[:, particle],
                        trajectory.bdoty[:, particle],
                        trajectory.bdotz[:, particle],
                    )
                ),
                rest_spin=spin,
                stereographic_frame=frame,
                frozen_segments=supplied_segments,
            )
            moment = float(trajectory.magnetic_moment_native[particle])
            sources.append(
                CausalC5DipoleSource(
                    identity=identity,
                    particle_index=particle,
                    magnetic_moment_native=moment,
                    history=history,
                )
            )
        return cls(tuple(sources))

    def append_accepted_state(
        self,
        state: Mapping[str, object],
    ) -> "CausalC5DipoleSourceCollection":
        return CausalC5DipoleSourceCollection(
            tuple(source.append_accepted_state(state) for source in self.sources)
        )


@dataclass(frozen=True)
class CausalC5DipoleSourceEvaluation:
    """One identity-labelled source contribution."""

    identity: str
    response: DipoleHertzResponseJetResult


@dataclass(frozen=True)
class CausalC5DipoleProviderResult:
    """Stable-order source sum and the individual auditable contributions."""

    four_potential: np.ndarray
    partial_a: np.ndarray
    electric_field_native: np.ndarray
    magnetic_field_native: np.ndarray
    field_tensor: np.ndarray
    partial_f: np.ndarray
    source_results: tuple[CausalC5DipoleSourceEvaluation, ...]

    def __post_init__(self) -> None:
        shapes = {
            "four_potential": (4,),
            "partial_a": (4, 4),
            "electric_field_native": (3,),
            "magnetic_field_native": (3,),
            "field_tensor": (4, 4),
            "partial_f": (4, 4, 4),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite array with shape {shape}")
            detached = np.array(value, dtype=np.float64, copy=True)
            detached.setflags(write=False)
            object.__setattr__(self, name, detached)
        object.__setattr__(self, "source_results", tuple(self.source_results))


def evaluate_causal_c5_dipole_source_collection_native(
    collection: CausalC5DipoleSourceCollection,
    observer_event: "ObserverEvent",
    *,
    excluded_source_identities: Sequence[str] = (),
    root_tolerance_mm: float = 1.0e-21,
    max_root_iterations: int = 96,
    minimum_separation_mm: float = 1.0e-15,
) -> CausalC5DipoleProviderResult:
    """Evaluate and sum all non-excluded sources in their declared order."""

    excluded = set(str(identity) for identity in excluded_source_identities)
    source_results: list[CausalC5DipoleSourceEvaluation] = []
    potential = np.zeros(4, dtype=np.float64)
    partial_a = np.zeros((4, 4), dtype=np.float64)
    field = np.zeros((4, 4), dtype=np.float64)
    partial_f = np.zeros((4, 4, 4), dtype=np.float64)
    for source in collection.sources:
        if source.identity in excluded:
            continue
        try:
            response = evaluate_causal_c5_dipole_hertz_response_native(
                source.history,  # type: ignore[arg-type]
                observer_event,
                magnetic_moment_native=source.magnetic_moment_native,
                root_tolerance_mm=root_tolerance_mm,
                max_root_iterations=max_root_iterations,
                minimum_separation_mm=minimum_separation_mm,
            )
        except CausalC5HistoryUnavailableError as exc:
            raise CausalC5HistoryUnavailableError(
                f"source identity {source.identity!r}: {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"source identity {source.identity!r}: {exc}") from exc
        potential += response.four_potential
        partial_a += response.partial_a
        field += response.field_tensor
        partial_f += response.partial_f
        source_results.append(CausalC5DipoleSourceEvaluation(source.identity, response))
    electric, magnetic = fields_from_tensor_native(field)
    return CausalC5DipoleProviderResult(
        four_potential=potential,
        partial_a=partial_a,
        electric_field_native=electric,
        magnetic_field_native=magnetic,
        field_tensor=field,
        partial_f=partial_f,
        source_results=tuple(source_results),
    )


@dataclass(frozen=True)
class AcceptedPairCausalC5SourceHistory:
    """Detached causal histories owned by one jointly accepted pair path."""

    rider: CausalC5DipoleSourceCollection
    driver: CausalC5DipoleSourceCollection

    @classmethod
    def from_trajectory_arrays(
        cls,
        rider: TrajectoryArrays,
        driver: TrajectoryArrays,
    ) -> "AcceptedPairCausalC5SourceHistory":
        return cls(
            rider=CausalC5DipoleSourceCollection.from_trajectory_arrays(
                rider,
                identity_prefix="rider",
            ),
            driver=CausalC5DipoleSourceCollection.from_trajectory_arrays(
                driver,
                identity_prefix="driver",
            ),
        )


@dataclass(frozen=True)
class GrowableCausalC5CollectionTransaction:
    """All source appends preflighted for one ordered collection."""

    candidate: CausalC5DipoleSourceCollection
    source_transactions: tuple[CausalC5AppendTransaction, ...]


class GrowableCausalC5DipoleSourceCollection:
    """Stable source metadata plus growable per-source accepted histories."""

    def __init__(
        self,
        *,
        identities: Sequence[str],
        particle_indices: Sequence[int],
        magnetic_moments_native: Sequence[float],
        histories: Sequence[GrowableCausalC5SourceHistory],
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
            raise ValueError("growable causal C5 source metadata lengths must match")
        if len(set(self.identities)) != size:
            raise ValueError("growable causal C5 source identities must be unique")
        if len(set(self.particle_indices)) != size:
            raise ValueError("growable causal C5 particle indices must be unique")
        if any(index < 0 for index in self.particle_indices):
            raise ValueError("growable causal C5 particle indices must be non-negative")
        if any(
            not np.isfinite(moment) or moment == 0.0
            for moment in self.magnetic_moments_native
        ):
            raise ValueError("growable causal C5 moments must be finite and nonzero")

    @classmethod
    def from_collection(
        cls,
        collection: CausalC5DipoleSourceCollection,
    ) -> "GrowableCausalC5DipoleSourceCollection":
        return cls(
            identities=collection.source_identities,
            particle_indices=tuple(
                source.particle_index for source in collection.sources
            ),
            magnetic_moments_native=tuple(
                source.magnetic_moment_native for source in collection.sources
            ),
            histories=tuple(
                GrowableCausalC5SourceHistory.from_history(source.history)
                for source in collection.sources
            ),
        )

    @classmethod
    def from_trajectory_arrays(
        cls,
        trajectory: TrajectoryArrays,
        *,
        identity_prefix: str,
    ) -> "GrowableCausalC5DipoleSourceCollection":
        return cls.from_collection(
            CausalC5DipoleSourceCollection.from_trajectory_arrays(
                trajectory,
                identity_prefix=identity_prefix,
            )
        )

    def build_current(self) -> CausalC5DipoleSourceCollection:
        return CausalC5DipoleSourceCollection(
            tuple(
                CausalC5DipoleSource(
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
    ) -> GrowableCausalC5CollectionTransaction:
        """Preflight all source rows without publishing any source."""

        rows = tuple(states)
        if not rows:
            raise ValueError("accepted source state sequence must not be empty")
        transactions: list[CausalC5AppendTransaction] = []
        candidate_sources: list[CausalC5DipoleSource] = []
        for identity, particle, moment, history in zip(
            self.identities,
            self.particle_indices,
            self.magnetic_moments_native,
            self.histories,
        ):
            for state in rows:
                if "magnetic_moment_native" in state and (
                    _state_scalar(state, "magnetic_moment_native", particle) != moment
                ):
                    raise ValueError(
                        f"source identity {identity!r} changed magnetic moment"
                    )
            transaction = history.preflight_append_many(
                time_ns=np.asarray(
                    [_state_scalar(state, "t", particle) for state in rows]
                ),
                position_mm=np.asarray(
                    [_state_vector(state, ("x", "y", "z"), particle) for state in rows]
                ),
                beta=np.asarray(
                    [
                        _state_vector(state, ("bx", "by", "bz"), particle)
                        for state in rows
                    ]
                ),
                beta_prime_per_mm=np.asarray(
                    [
                        _state_vector(
                            state,
                            ("bdotx", "bdoty", "bdotz"),
                            particle,
                        )
                        for state in rows
                    ]
                ),
                rest_spin=np.asarray(
                    [
                        _state_vector(
                            state,
                            ("spin_x", "spin_y", "spin_z"),
                            particle,
                        )
                        for state in rows
                    ]
                ),
            )
            transactions.append(transaction)
            candidate_sources.append(
                CausalC5DipoleSource(
                    identity=identity,
                    particle_index=particle,
                    magnetic_moment_native=moment,
                    history=transaction.candidate,
                )
            )
        return GrowableCausalC5CollectionTransaction(
            candidate=CausalC5DipoleSourceCollection(tuple(candidate_sources)),
            source_transactions=tuple(transactions),
        )

    def can_commit(self, transaction: GrowableCausalC5CollectionTransaction) -> bool:
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
        transaction: GrowableCausalC5CollectionTransaction,
    ) -> CausalC5DipoleSourceCollection:
        if not self.can_commit(transaction):
            raise RuntimeError("growable causal C5 collection transaction is stale")
        for history, source_transaction in zip(
            self.histories,
            transaction.source_transactions,
        ):
            history.commit(source_transaction)
        return self.build_current()


@dataclass(frozen=True)
class GrowableAcceptedPairCausalC5Transaction:
    """A rider/driver candidate whose two collection commits are preflighted."""

    candidate: AcceptedPairCausalC5SourceHistory
    rider: GrowableCausalC5CollectionTransaction
    driver: GrowableCausalC5CollectionTransaction


class GrowableAcceptedPairCausalC5SourceHistory:
    """Transaction owner for two jointly accepted source collections."""

    def __init__(
        self,
        *,
        rider: GrowableCausalC5DipoleSourceCollection,
        driver: GrowableCausalC5DipoleSourceCollection,
    ) -> None:
        self.rider = rider
        self.driver = driver

    @classmethod
    def from_accepted(
        cls,
        accepted: AcceptedPairCausalC5SourceHistory,
    ) -> "GrowableAcceptedPairCausalC5SourceHistory":
        return cls(
            rider=GrowableCausalC5DipoleSourceCollection.from_collection(
                accepted.rider
            ),
            driver=GrowableCausalC5DipoleSourceCollection.from_collection(
                accepted.driver
            ),
        )

    @classmethod
    def from_trajectory_arrays(
        cls,
        rider: TrajectoryArrays,
        driver: TrajectoryArrays,
    ) -> "GrowableAcceptedPairCausalC5SourceHistory":
        return cls(
            rider=GrowableCausalC5DipoleSourceCollection.from_trajectory_arrays(
                rider,
                identity_prefix="rider",
            ),
            driver=GrowableCausalC5DipoleSourceCollection.from_trajectory_arrays(
                driver,
                identity_prefix="driver",
            ),
        )

    def build_current(self) -> AcceptedPairCausalC5SourceHistory:
        return AcceptedPairCausalC5SourceHistory(
            rider=self.rider.build_current(),
            driver=self.driver.build_current(),
        )

    def preflight_states(
        self,
        *,
        rider_states: Sequence[Mapping[str, object]],
        driver_states: Sequence[Mapping[str, object]],
    ) -> GrowableAcceptedPairCausalC5Transaction:
        rider = self.rider.preflight_append_states(rider_states)
        driver = self.driver.preflight_append_states(driver_states)
        return GrowableAcceptedPairCausalC5Transaction(
            candidate=AcceptedPairCausalC5SourceHistory(
                rider=rider.candidate,
                driver=driver.candidate,
            ),
            rider=rider,
            driver=driver,
        )

    def can_commit(self, transaction: GrowableAcceptedPairCausalC5Transaction) -> bool:
        return self.rider.can_commit(transaction.rider) and self.driver.can_commit(
            transaction.driver
        )

    def commit(
        self,
        transaction: GrowableAcceptedPairCausalC5Transaction,
    ) -> AcceptedPairCausalC5SourceHistory:
        if not self.can_commit(transaction):
            raise RuntimeError("growable causal C5 pair transaction is stale")
        self.rider.commit(transaction.rider)
        self.driver.commit(transaction.driver)
        return self.build_current()


def build_accepted_pair_causal_c5_candidate(
    trial: "ExactPairStepDoublingTrial",
    accepted: AcceptedPairCausalC5SourceHistory,
) -> AcceptedPairCausalC5SourceHistory:
    """Append the authoritative midpoint and endpoint without mutating ``accepted``."""

    rider = accepted.rider
    driver = accepted.driver
    for slab in (trial.midpoint, trial.refined):
        rider = rider.append_accepted_state(slab.pair.rider.state)
        driver = driver.append_accepted_state(slab.pair.driver.state)
    return AcceptedPairCausalC5SourceHistory(rider=rider, driver=driver)


__all__ = [
    "AcceptedPairCausalC5SourceHistory",
    "CausalC5DipoleProviderResult",
    "CausalC5DipoleSource",
    "CausalC5DipoleSourceCollection",
    "CausalC5DipoleSourceEvaluation",
    "GrowableAcceptedPairCausalC5SourceHistory",
    "GrowableAcceptedPairCausalC5Transaction",
    "GrowableCausalC5CollectionTransaction",
    "GrowableCausalC5DipoleSourceCollection",
    "build_accepted_pair_causal_c5_candidate",
    "evaluate_causal_c5_dipole_source_collection_native",
]
