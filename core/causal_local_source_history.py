"""Accepted source samples for causal local-jet reconstruction.

This history has one deliberately narrow purpose: retain the position,
velocity, spin, and exact equations-of-motion acceleration needed by the local
retarded source-jet provider.  It does not store the legacy ``bdot`` output or
any frozen high-degree polynomial.

Acceleration belongs to an accepted interval, not its endpoint row.  For the
interval from sample ``i`` to sample ``i + 1``,
``interval_start_beta_prime_per_mm[i]`` is the instantaneous coordinate
derivative of beta at sample ``i``.  This representation makes the timing
contract explicit and avoids the historical row offset used by the C5
history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence, cast

import numpy as np

from .source_kinematics import coordinate_beta_prime_from_four_kinematics

if TYPE_CHECKING:
    from .exact_pair_trial import ExactPairStepDoublingTrial
    from .types import TrajectoryArrays

_CHECKPOINT_SCHEMA_VERSION = 1
_CHART_POLE_TOLERANCE = 1.0e-12


class CausalLocalSourceHistoryUnavailableError(RuntimeError):
    """Raised when accepted data cannot define the requested local source jet."""


class CausalLocalHistoryView(Protocol):
    """Read-only array surface consumed by local-jet reconstruction."""

    @property
    def time_ns(self) -> np.ndarray: ...

    @property
    def position_mm(self) -> np.ndarray: ...

    @property
    def beta(self) -> np.ndarray: ...

    @property
    def rest_spin(self) -> np.ndarray: ...

    @property
    def stereographic_frame(self) -> np.ndarray: ...

    @property
    def interval_start_beta_prime_per_mm(self) -> np.ndarray: ...

    @property
    def interval_start_acceleration_ready(self) -> np.ndarray: ...

    @property
    def sample_count(self) -> int: ...

    @property
    def interval_count(self) -> int: ...


def _readonly_float_array(
    values: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    shape: tuple[int, ...] | None,
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.all(np.isfinite(result)):
        expected = "the required shape" if shape is None else str(shape)
        raise ValueError(f"{name} must be a finite array with shape {expected}")
    result = np.array(result, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_bool_array(
    values: Sequence[bool] | np.ndarray,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    result = np.array(result, dtype=bool, copy=True)
    result.setflags(write=False)
    return result


def _validated_stereographic_frame(
    value: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    frame = _readonly_float_array(
        value,
        shape=(3, 3),
        name="stereographic_frame",
    )
    if not np.allclose(frame.T @ frame, np.eye(3), rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("stereographic_frame must be orthonormal")
    if not np.isclose(np.linalg.det(frame), 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("stereographic_frame must be right handed")
    return frame


def deterministic_stereographic_frame(rest_spin: np.ndarray) -> np.ndarray:
    """Choose a reproducible right-handed chart centered on an initial spin."""

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
    return _validated_stereographic_frame(np.column_stack((first, second, north)))


def spin_to_stereographic(rest_spin: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Map unit rest-spin rows into one fixed two-coordinate spin chart."""

    local = np.asarray(rest_spin, dtype=np.float64) @ np.asarray(
        frame,
        dtype=np.float64,
    )
    denominator = 1.0 + local[:, 2]
    if np.any(denominator <= _CHART_POLE_TOLERANCE):
        raise CausalLocalSourceHistoryUnavailableError(
            "accepted spin history reaches the fixed stereographic chart pole"
        )
    return cast(np.ndarray, local[:, :2] / denominator[:, np.newaxis])


@dataclass(frozen=True)
class AcceptedLocalSourceSample:
    """One accepted endpoint and the exact acceleration at its interval start."""

    time_ns: float
    position_mm: np.ndarray
    beta: np.ndarray
    rest_spin: np.ndarray
    interval_start_beta_prime_per_mm: np.ndarray
    interval_start_acceleration_ready: bool

    def __post_init__(self) -> None:
        time = float(self.time_ns)
        if not np.isfinite(time):
            raise ValueError("accepted source time must be finite")
        object.__setattr__(self, "time_ns", time)
        for name in (
            "position_mm",
            "beta",
            "rest_spin",
            "interval_start_beta_prime_per_mm",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_float_array(getattr(self, name), shape=(3,), name=name),
            )
        if float(self.beta @ self.beta) >= 1.0:
            raise ValueError("accepted source beta magnitude must be below one")
        if not np.isclose(
            np.linalg.norm(self.rest_spin),
            1.0,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError("accepted source rest spin must have unit magnitude")
        object.__setattr__(
            self,
            "interval_start_acceleration_ready",
            bool(self.interval_start_acceleration_ready),
        )


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


def _accepted_interval_start_beta_prime_per_mm(
    state: Mapping[str, object],
    particle_index: int,
) -> np.ndarray | None:
    ready_values = np.asarray(
        state.get("source_start_beta_prime_ready", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    if (
        ready_values.ndim == 1
        and particle_index < ready_values.size
        and bool(ready_values[particle_index])
    ):
        return _state_vector(
            state,
            (
                "source_start_beta_prime_x_per_mm",
                "source_start_beta_prime_y_per_mm",
                "source_start_beta_prime_z_per_mm",
            ),
            particle_index,
        )

    complete_values = np.asarray(
        state.get("_source_start_acceleration_complete", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    if (
        complete_values.ndim != 1
        or particle_index >= complete_values.size
        or not bool(complete_values[particle_index])
    ):
        return None
    velocity_values = np.asarray(
        state.get("_intrinsic_spin_start_four_velocity"),
        dtype=np.float64,
    )
    acceleration_values = np.asarray(
        state.get("_intrinsic_spin_start_non_self_four_acceleration"),
        dtype=np.float64,
    )
    if (
        velocity_values.ndim != 2
        or acceleration_values.ndim != 2
        or velocity_values.shape[1:] != (4,)
        or acceleration_values.shape != velocity_values.shape
        or particle_index >= velocity_values.shape[0]
    ):
        raise ValueError("accepted source state has invalid start four-kinematics")
    return coordinate_beta_prime_from_four_kinematics(
        velocity_values[particle_index],
        acceleration_values[particle_index],
    )


def accepted_local_source_sample_from_state(
    state: Mapping[str, object],
    particle_index: int,
) -> AcceptedLocalSourceSample:
    """Extract an endpoint plus the exact acceleration used to begin its step."""

    acceleration = _accepted_interval_start_beta_prime_per_mm(state, particle_index)
    return AcceptedLocalSourceSample(
        time_ns=_state_scalar(state, "t", particle_index),
        position_mm=_state_vector(state, ("x", "y", "z"), particle_index),
        beta=_state_vector(state, ("bx", "by", "bz"), particle_index),
        rest_spin=_state_vector(
            state,
            ("spin_x", "spin_y", "spin_z"),
            particle_index,
        ),
        interval_start_beta_prime_per_mm=(
            np.zeros(3, dtype=np.float64) if acceleration is None else acceleration
        ),
        interval_start_acceleration_ready=acceleration is not None,
    )


@dataclass(frozen=True)
class CausalLocalSourceHistory:
    """Immutable accepted source prefix with explicitly timed acceleration."""

    time_ns: np.ndarray
    position_mm: np.ndarray
    beta: np.ndarray
    rest_spin: np.ndarray
    stereographic_frame: np.ndarray
    interval_start_beta_prime_per_mm: np.ndarray
    interval_start_acceleration_ready: np.ndarray

    def __post_init__(self) -> None:
        times = _readonly_float_array(self.time_ns, shape=None, name="time_ns")
        if times.ndim != 1 or np.any(np.diff(times) <= 0.0):
            raise ValueError("accepted source times must increase strictly")
        count = int(times.size)
        interval_count = max(0, count - 1)
        object.__setattr__(self, "time_ns", times)
        for name in ("position_mm", "beta", "rest_spin"):
            object.__setattr__(
                self,
                name,
                _readonly_float_array(
                    getattr(self, name),
                    shape=(count, 3),
                    name=name,
                ),
            )
        object.__setattr__(
            self,
            "interval_start_beta_prime_per_mm",
            _readonly_float_array(
                self.interval_start_beta_prime_per_mm,
                shape=(interval_count, 3),
                name="interval_start_beta_prime_per_mm",
            ),
        )
        object.__setattr__(
            self,
            "interval_start_acceleration_ready",
            _readonly_bool_array(
                self.interval_start_acceleration_ready,
                shape=(interval_count,),
                name="interval_start_acceleration_ready",
            ),
        )
        if count and np.any(np.sum(self.beta * self.beta, axis=1) >= 1.0):
            raise ValueError("accepted source beta magnitude must be below one")
        if count and not np.allclose(
            np.linalg.norm(self.rest_spin, axis=1),
            1.0,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError("accepted source rest spin must have unit magnitude")
        object.__setattr__(
            self,
            "stereographic_frame",
            _validated_stereographic_frame(self.stereographic_frame),
        )

    @classmethod
    def empty(
        cls,
        *,
        stereographic_frame: Sequence[Sequence[float]] | np.ndarray = np.eye(3),
    ) -> "CausalLocalSourceHistory":
        return cls(
            time_ns=np.zeros(0),
            position_mm=np.zeros((0, 3)),
            beta=np.zeros((0, 3)),
            rest_spin=np.zeros((0, 3)),
            stereographic_frame=np.asarray(stereographic_frame, dtype=np.float64),
            interval_start_beta_prime_per_mm=np.zeros((0, 3)),
            interval_start_acceleration_ready=np.zeros(0, dtype=bool),
        )

    @classmethod
    def from_accepted_samples(
        cls,
        *,
        time_ns: Sequence[float] | np.ndarray,
        position_mm: Sequence[Sequence[float]] | np.ndarray,
        beta: Sequence[Sequence[float]] | np.ndarray,
        rest_spin: Sequence[Sequence[float]] | np.ndarray,
        interval_start_beta_prime_per_mm: Sequence[Sequence[float]] | np.ndarray,
        interval_start_acceleration_ready: Sequence[bool] | np.ndarray,
        stereographic_frame: Sequence[Sequence[float]] | np.ndarray = np.eye(3),
    ) -> "CausalLocalSourceHistory":
        return cls(
            time_ns=np.asarray(time_ns, dtype=np.float64),
            position_mm=np.asarray(position_mm, dtype=np.float64),
            beta=np.asarray(beta, dtype=np.float64),
            rest_spin=np.asarray(rest_spin, dtype=np.float64),
            stereographic_frame=np.asarray(stereographic_frame, dtype=np.float64),
            interval_start_beta_prime_per_mm=np.asarray(
                interval_start_beta_prime_per_mm,
                dtype=np.float64,
            ),
            interval_start_acceleration_ready=np.asarray(
                interval_start_acceleration_ready,
                dtype=bool,
            ),
        )

    @property
    def sample_count(self) -> int:
        return int(self.time_ns.size)

    @property
    def interval_count(self) -> int:
        return max(0, self.sample_count - 1)

    def append_accepted_interval(
        self,
        sample: AcceptedLocalSourceSample,
    ) -> "CausalLocalSourceHistory":
        """Return a detached prefix with one accepted interval and endpoint."""

        if self.sample_count == 0:
            raise ValueError("an interval endpoint needs an existing start sample")
        if sample.time_ns <= float(self.time_ns[-1]):
            raise ValueError("accepted source time must increase strictly")
        return CausalLocalSourceHistory(
            time_ns=np.concatenate((self.time_ns, np.asarray((sample.time_ns,)))),
            position_mm=np.vstack((self.position_mm, sample.position_mm)),
            beta=np.vstack((self.beta, sample.beta)),
            rest_spin=np.vstack((self.rest_spin, sample.rest_spin)),
            stereographic_frame=self.stereographic_frame,
            interval_start_beta_prime_per_mm=np.vstack(
                (
                    self.interval_start_beta_prime_per_mm,
                    sample.interval_start_beta_prime_per_mm,
                )
            ),
            interval_start_acceleration_ready=np.concatenate(
                (
                    self.interval_start_acceleration_ready,
                    np.asarray((sample.interval_start_acceleration_ready,)),
                )
            ),
        )

    def to_checkpoint_payload(self) -> dict[str, object]:
        """Return a JSON-compatible exact accepted-history payload."""

        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "time_ns": self.time_ns.tolist(),
            "position_mm": self.position_mm.tolist(),
            "beta": self.beta.tolist(),
            "rest_spin": self.rest_spin.tolist(),
            "stereographic_frame": self.stereographic_frame.tolist(),
            "interval_start_beta_prime_per_mm": (
                self.interval_start_beta_prime_per_mm.tolist()
            ),
            "interval_start_acceleration_ready": (
                self.interval_start_acceleration_ready.tolist()
            ),
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "CausalLocalSourceHistory":
        """Restore a history without recomputing or reindexing acceleration."""

        version = int(cast(int, payload.get("schema_version", -1)))
        if version != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported causal local source-history checkpoint schema: "
                f"{version}"
            )
        required = (
            "time_ns",
            "position_mm",
            "beta",
            "rest_spin",
            "stereographic_frame",
            "interval_start_beta_prime_per_mm",
            "interval_start_acceleration_ready",
        )
        missing = tuple(name for name in required if name not in payload)
        if missing:
            raise ValueError(
                "causal local source-history checkpoint is missing: "
                + ", ".join(missing)
            )
        return cls.from_accepted_samples(
            time_ns=np.asarray(payload["time_ns"], dtype=np.float64),
            position_mm=np.asarray(payload["position_mm"], dtype=np.float64),
            beta=np.asarray(payload["beta"], dtype=np.float64),
            rest_spin=np.asarray(payload["rest_spin"], dtype=np.float64),
            stereographic_frame=np.asarray(
                payload["stereographic_frame"],
                dtype=np.float64,
            ),
            interval_start_beta_prime_per_mm=np.asarray(
                payload["interval_start_beta_prime_per_mm"],
                dtype=np.float64,
            ),
            interval_start_acceleration_ready=np.asarray(
                payload["interval_start_acceleration_ready"],
                dtype=bool,
            ),
        )


@dataclass(frozen=True)
class CausalLocalDipoleSource:
    """One stable source identity, constant moment, and accepted local history."""

    identity: str
    particle_index: int
    magnetic_moment_native: float
    history: CausalLocalHistoryView

    def __post_init__(self) -> None:
        identity = str(self.identity)
        particle = int(self.particle_index)
        moment = float(self.magnetic_moment_native)
        if not identity:
            raise ValueError("causal local source identity must not be empty")
        if particle < 0:
            raise ValueError("causal local source particle index must be non-negative")
        if not np.isfinite(moment) or moment == 0.0:
            raise ValueError("causal local source moment must be finite and nonzero")
        if not hasattr(self.history, "sample_count"):
            raise TypeError("causal local source history has no accepted sample count")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "particle_index", particle)
        object.__setattr__(self, "magnetic_moment_native", moment)

    def append_accepted_state(
        self,
        state: Mapping[str, object],
    ) -> "CausalLocalDipoleSource":
        if not isinstance(self.history, CausalLocalSourceHistory):
            raise TypeError("a published growable history cannot be copied by append")
        if "magnetic_moment_native" in state and (
            _state_scalar(state, "magnetic_moment_native", self.particle_index)
            != self.magnetic_moment_native
        ):
            raise ValueError(
                f"source identity {self.identity!r} changed magnetic moment"
            )
        return CausalLocalDipoleSource(
            identity=self.identity,
            particle_index=self.particle_index,
            magnetic_moment_native=self.magnetic_moment_native,
            history=self.history.append_accepted_interval(
                accepted_local_source_sample_from_state(state, self.particle_index)
            ),
        )


@dataclass(frozen=True)
class CausalLocalDipoleSourceCollection:
    """A fixed source order; the tuple order is the floating-point sum order."""

    sources: tuple[CausalLocalDipoleSource, ...] = ()

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        identities = tuple(source.identity for source in sources)
        particles = tuple(source.particle_index for source in sources)
        if len(set(identities)) != len(identities):
            raise ValueError("causal local source identities must be unique")
        if len(set(particles)) != len(particles):
            raise ValueError("causal local source particle indices must be unique")
        object.__setattr__(self, "sources", sources)

    @property
    def source_identities(self) -> tuple[str, ...]:
        return tuple(source.identity for source in self.sources)

    @classmethod
    def from_trajectory_arrays(
        cls,
        trajectory: "TrajectoryArrays",
        *,
        identity_prefix: str,
        particle_indices: Sequence[int] | None = None,
        source_identities: Sequence[str] | None = None,
        stereographic_frames: Sequence[np.ndarray] | None = None,
    ) -> "CausalLocalDipoleSourceCollection":
        """Construct local histories without consulting legacy ``bdot`` rows."""

        trajectory.require_current_storage()
        if trajectory.n_steps < 1:
            raise ValueError("a causal local source needs an initial accepted sample")
        if particle_indices is None:
            active = np.asarray(trajectory.magnetic_dipole_active, dtype=bool)
            moments = np.asarray(trajectory.magnetic_moment_native, dtype=np.float64)
            indices = tuple(
                int(index)
                for index in np.flatnonzero(
                    active & np.isfinite(moments) & (moments != 0.0)
                )
            )
        else:
            indices = tuple(int(index) for index in particle_indices)
        if any(index < 0 or index >= trajectory.n_particles for index in indices):
            raise ValueError("causal local source particle index is out of bounds")
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

        sources: list[CausalLocalDipoleSource] = []
        for sequence_index, (identity, particle) in enumerate(zip(identities, indices)):
            spin = np.column_stack(
                (
                    trajectory.spin_x[:, particle],
                    trajectory.spin_y[:, particle],
                    trajectory.spin_z[:, particle],
                )
            )
            frame = (
                deterministic_stereographic_frame(spin[0])
                if stereographic_frames is None
                else np.asarray(stereographic_frames[sequence_index], dtype=np.float64)
            )
            interval_beta_prime = np.column_stack(
                (
                    trajectory.source_start_beta_prime_x_per_mm[1:, particle],
                    trajectory.source_start_beta_prime_y_per_mm[1:, particle],
                    trajectory.source_start_beta_prime_z_per_mm[1:, particle],
                )
            )
            history = CausalLocalSourceHistory.from_accepted_samples(
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
                rest_spin=spin,
                stereographic_frame=frame,
                interval_start_beta_prime_per_mm=interval_beta_prime,
                interval_start_acceleration_ready=(
                    trajectory.source_start_beta_prime_ready[1:, particle]
                ),
            )
            sources.append(
                CausalLocalDipoleSource(
                    identity=identity,
                    particle_index=particle,
                    magnetic_moment_native=float(
                        trajectory.magnetic_moment_native[particle]
                    ),
                    history=history,
                )
            )
        return cls(tuple(sources))

    def append_accepted_state(
        self,
        state: Mapping[str, object],
    ) -> "CausalLocalDipoleSourceCollection":
        return CausalLocalDipoleSourceCollection(
            tuple(source.append_accepted_state(state) for source in self.sources)
        )


@dataclass(frozen=True)
class AcceptedPairCausalLocalSourceHistory:
    """Detached local histories owned by one jointly accepted pair path."""

    rider: CausalLocalDipoleSourceCollection
    driver: CausalLocalDipoleSourceCollection

    @classmethod
    def from_trajectory_arrays(
        cls,
        rider: "TrajectoryArrays",
        driver: "TrajectoryArrays",
    ) -> "AcceptedPairCausalLocalSourceHistory":
        return cls(
            rider=CausalLocalDipoleSourceCollection.from_trajectory_arrays(
                rider,
                identity_prefix="rider",
            ),
            driver=CausalLocalDipoleSourceCollection.from_trajectory_arrays(
                driver,
                identity_prefix="driver",
            ),
        )


def build_accepted_pair_causal_local_candidate(
    trial: "ExactPairStepDoublingTrial",
    accepted: AcceptedPairCausalLocalSourceHistory,
) -> AcceptedPairCausalLocalSourceHistory:
    """Append authoritative midpoint and endpoint rows without publishing either."""

    rider = accepted.rider
    driver = accepted.driver
    for slab in (trial.midpoint, trial.refined):
        rider = rider.append_accepted_state(slab.pair.rider.state)
        driver = driver.append_accepted_state(slab.pair.driver.state)
    return AcceptedPairCausalLocalSourceHistory(rider=rider, driver=driver)


__all__ = [
    "AcceptedLocalSourceSample",
    "AcceptedPairCausalLocalSourceHistory",
    "CausalLocalDipoleSource",
    "CausalLocalDipoleSourceCollection",
    "CausalLocalHistoryView",
    "CausalLocalSourceHistory",
    "CausalLocalSourceHistoryUnavailableError",
    "accepted_local_source_sample_from_state",
    "build_accepted_pair_causal_local_candidate",
    "deterministic_stereographic_frame",
    "spin_to_stereographic",
]
