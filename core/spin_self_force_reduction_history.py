"""Immutable accepted-history state for causal intrinsic-spin reduction.

The analytical retarded-potential derivative is undefined at the current
worldline or C1 spin-interpolation knots.  The separately validated backward
six-sample reduction supplies a causal diagnostic at those events, but only if
its samples are accepted leading-order states.

This module provides that state boundary without connecting a force to the
integrator.  Appending a sample returns a new immutable object.  A rejected
adaptive or nonlinear trial can therefore discard its candidate without
mutating accepted history.  The compact state has a strict JSON-compatible
checkpoint payload so restart parity can be tested before production wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence, cast

import numpy as np

from .spin_self_force_reduction_oracle import (
    PotentialDirectionalIntrinsicSpinReductionResult,
    SampledIntrinsicSpinReductionResult,
    evaluate_causal_sampled_intrinsic_spin_reduction_native,
)

if TYPE_CHECKING:
    from .exact_pair_trial import ExactPairStepDoublingTrial

_CHECKPOINT_SCHEMA_VERSION = 1
_MINIMUM_CAUSAL_SAMPLES = 6


def _readonly_matrix(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    rows: int,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (rows, 4):
        raise ValueError(f"{name} must have shape ({rows}, 4)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(matrix, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AcceptedIntrinsicSpinReductionHistory:
    """The newest accepted non-self samples used by the backward reduction."""

    proper_times_ns: np.ndarray
    four_velocity_mm_ns: np.ndarray
    non_self_four_acceleration_mm_ns2: np.ndarray
    physical_spin_four_native: np.ndarray
    maximum_samples: int = _MINIMUM_CAUSAL_SAMPLES

    def __post_init__(self) -> None:
        times = np.asarray(self.proper_times_ns, dtype=float)
        maximum = int(self.maximum_samples)
        if maximum < _MINIMUM_CAUSAL_SAMPLES:
            raise ValueError(
                f"maximum_samples must be at least {_MINIMUM_CAUSAL_SAMPLES}"
            )
        if times.ndim != 1 or times.size > maximum:
            raise ValueError(
                "proper_times_ns must be one-dimensional and no longer than "
                "maximum_samples"
            )
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError("proper_times_ns must be finite and strictly increasing")
        readonly_times = np.array(times, dtype=float, copy=True)
        readonly_times.setflags(write=False)
        rows = int(times.size)
        object.__setattr__(self, "proper_times_ns", readonly_times)
        object.__setattr__(
            self,
            "four_velocity_mm_ns",
            _readonly_matrix(
                self.four_velocity_mm_ns,
                rows=rows,
                name="four_velocity_mm_ns",
            ),
        )
        object.__setattr__(
            self,
            "non_self_four_acceleration_mm_ns2",
            _readonly_matrix(
                self.non_self_four_acceleration_mm_ns2,
                rows=rows,
                name="non_self_four_acceleration_mm_ns2",
            ),
        )
        object.__setattr__(
            self,
            "physical_spin_four_native",
            _readonly_matrix(
                self.physical_spin_four_native,
                rows=rows,
                name="physical_spin_four_native",
            ),
        )
        object.__setattr__(self, "maximum_samples", maximum)

    @classmethod
    def empty(
        cls,
        *,
        maximum_samples: int = _MINIMUM_CAUSAL_SAMPLES,
    ) -> "AcceptedIntrinsicSpinReductionHistory":
        return cls(
            proper_times_ns=np.zeros(0, dtype=float),
            four_velocity_mm_ns=np.zeros((0, 4), dtype=float),
            non_self_four_acceleration_mm_ns2=np.zeros((0, 4), dtype=float),
            physical_spin_four_native=np.zeros((0, 4), dtype=float),
            maximum_samples=maximum_samples,
        )

    @property
    def sample_count(self) -> int:
        return int(self.proper_times_ns.size)

    @property
    def causal_reduction_ready(self) -> bool:
        return self.sample_count >= _MINIMUM_CAUSAL_SAMPLES

    def append_accepted(
        self,
        *,
        proper_time_ns: float,
        four_velocity_mm_ns: Sequence[float],
        non_self_four_acceleration_mm_ns2: Sequence[float],
        physical_spin_four_native: Sequence[float],
    ) -> "AcceptedIntrinsicSpinReductionHistory":
        """Return a candidate accepted state without mutating this history."""

        time = float(proper_time_ns)
        vectors = (
            np.asarray(four_velocity_mm_ns, dtype=float),
            np.asarray(non_self_four_acceleration_mm_ns2, dtype=float),
            np.asarray(physical_spin_four_native, dtype=float),
        )
        if not np.isfinite(time):
            raise ValueError("proper_time_ns must be finite")
        if self.sample_count and time <= float(self.proper_times_ns[-1]):
            raise ValueError("accepted proper times must increase strictly")
        if any(
            vector.shape != (4,) or not np.all(np.isfinite(vector))
            for vector in vectors
        ):
            raise ValueError(
                "accepted velocity, acceleration, and spin must be finite four-vectors"
            )
        times = np.concatenate((self.proper_times_ns, np.asarray((time,))))
        velocity = np.vstack((self.four_velocity_mm_ns, vectors[0]))
        acceleration = np.vstack((self.non_self_four_acceleration_mm_ns2, vectors[1]))
        spin = np.vstack((self.physical_spin_four_native, vectors[2]))
        if times.size > self.maximum_samples:
            times = times[-self.maximum_samples :]
            velocity = velocity[-self.maximum_samples :]
            acceleration = acceleration[-self.maximum_samples :]
            spin = spin[-self.maximum_samples :]
        return AcceptedIntrinsicSpinReductionHistory(
            proper_times_ns=times,
            four_velocity_mm_ns=velocity,
            non_self_four_acceleration_mm_ns2=acceleration,
            physical_spin_four_native=spin,
            maximum_samples=self.maximum_samples,
        )

    def evaluate_causal(
        self,
        *,
        charge_native: float,
        mass_amu: float,
        g_factor: float,
    ) -> SampledIntrinsicSpinReductionResult:
        if not self.causal_reduction_ready:
            raise ValueError(
                "causal intrinsic-spin reduction requires at least six accepted "
                "non-self samples"
            )
        return evaluate_causal_sampled_intrinsic_spin_reduction_native(
            proper_times_ns=self.proper_times_ns,
            four_velocity_samples_mm_ns=self.four_velocity_mm_ns,
            non_self_four_acceleration_samples_mm_ns2=(
                self.non_self_four_acceleration_mm_ns2
            ),
            physical_spin_four_samples_native=self.physical_spin_four_native,
            charge_native=charge_native,
            mass_amu=mass_amu,
            g_factor=g_factor,
        )

    def to_checkpoint_payload(self) -> dict[str, object]:
        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "maximum_samples": self.maximum_samples,
            "proper_times_ns": self.proper_times_ns.tolist(),
            "four_velocity_mm_ns": self.four_velocity_mm_ns.tolist(),
            "non_self_four_acceleration_mm_ns2": (
                self.non_self_four_acceleration_mm_ns2.tolist()
            ),
            "physical_spin_four_native": self.physical_spin_four_native.tolist(),
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "AcceptedIntrinsicSpinReductionHistory":
        required = {
            "schema_version",
            "maximum_samples",
            "proper_times_ns",
            "four_velocity_mm_ns",
            "non_self_four_acceleration_mm_ns2",
            "physical_spin_four_native",
        }
        if set(payload) != required:
            missing = sorted(required - set(payload))
            extra = sorted(set(payload) - required)
            raise ValueError(
                "intrinsic-spin reduction checkpoint keys do not match: "
                f"missing={missing}, extra={extra}"
            )
        if int(cast(int, payload["schema_version"])) != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported intrinsic-spin reduction checkpoint schema")
        return cls(
            proper_times_ns=np.asarray(payload["proper_times_ns"], dtype=float),
            four_velocity_mm_ns=np.asarray(payload["four_velocity_mm_ns"], dtype=float),
            non_self_four_acceleration_mm_ns2=np.asarray(
                payload["non_self_four_acceleration_mm_ns2"], dtype=float
            ),
            physical_spin_four_native=np.asarray(
                payload["physical_spin_four_native"], dtype=float
            ),
            maximum_samples=int(cast(int, payload["maximum_samples"])),
        )


@dataclass(frozen=True)
class AcceptedPairIntrinsicSpinReductionHistory:
    """Checkpointable causal histories for one accepted rider/driver pair.

    This object is deliberately separate from the trajectory builders.  A
    step-doubling trial may construct a replacement object, but the adaptive
    controller only adopts it when the same refined rider/driver path is
    jointly accepted.  The second-order exact equations expose the required
    private pre-self-reaction samples; the pure builder below converts the
    authoritative refined path without reading endpoint ``bdot``.
    """

    rider: AcceptedIntrinsicSpinReductionHistory
    driver: AcceptedIntrinsicSpinReductionHistory
    rider_endpoint_proper_time_ns: float = 0.0
    driver_endpoint_proper_time_ns: float = 0.0

    def __post_init__(self) -> None:
        endpoints = (
            float(self.rider_endpoint_proper_time_ns),
            float(self.driver_endpoint_proper_time_ns),
        )
        if not all(np.isfinite(value) for value in endpoints):
            raise ValueError("accepted endpoint proper times must be finite")
        for role, history, endpoint in (
            ("rider", self.rider, endpoints[0]),
            ("driver", self.driver, endpoints[1]),
        ):
            if history.sample_count and endpoint < float(history.proper_times_ns[-1]):
                raise ValueError(
                    f"{role} endpoint proper time precedes its newest force sample"
                )
        object.__setattr__(self, "rider_endpoint_proper_time_ns", endpoints[0])
        object.__setattr__(self, "driver_endpoint_proper_time_ns", endpoints[1])

    @classmethod
    def empty(
        cls,
        *,
        maximum_samples: int = _MINIMUM_CAUSAL_SAMPLES,
    ) -> "AcceptedPairIntrinsicSpinReductionHistory":
        return cls(
            rider=AcceptedIntrinsicSpinReductionHistory.empty(
                maximum_samples=maximum_samples
            ),
            driver=AcceptedIntrinsicSpinReductionHistory.empty(
                maximum_samples=maximum_samples
            ),
            rider_endpoint_proper_time_ns=0.0,
            driver_endpoint_proper_time_ns=0.0,
        )

    def to_checkpoint_payload(self) -> dict[str, object]:
        """Return the complete pair as strict finite JSON-compatible data."""

        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "rider": self.rider.to_checkpoint_payload(),
            "driver": self.driver.to_checkpoint_payload(),
            "rider_endpoint_proper_time_ns": self.rider_endpoint_proper_time_ns,
            "driver_endpoint_proper_time_ns": self.driver_endpoint_proper_time_ns,
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "AcceptedPairIntrinsicSpinReductionHistory":
        required = {
            "schema_version",
            "rider",
            "driver",
            "rider_endpoint_proper_time_ns",
            "driver_endpoint_proper_time_ns",
        }
        if set(payload) != required:
            missing = sorted(required - set(payload))
            extra = sorted(set(payload) - required)
            raise ValueError(
                "pair intrinsic-spin checkpoint keys do not match: "
                f"missing={missing}, extra={extra}"
            )
        if int(cast(int, payload["schema_version"])) != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported pair intrinsic-spin checkpoint schema")
        rider_payload = payload["rider"]
        driver_payload = payload["driver"]
        if not isinstance(rider_payload, Mapping) or not isinstance(
            driver_payload, Mapping
        ):
            raise ValueError("pair intrinsic-spin histories must be JSON objects")
        return cls(
            rider=AcceptedIntrinsicSpinReductionHistory.from_checkpoint_payload(
                rider_payload
            ),
            driver=AcceptedIntrinsicSpinReductionHistory.from_checkpoint_payload(
                driver_payload
            ),
            rider_endpoint_proper_time_ns=float(
                cast(float, payload["rider_endpoint_proper_time_ns"])
            ),
            driver_endpoint_proper_time_ns=float(
                cast(float, payload["driver_endpoint_proper_time_ns"])
            ),
        )


def _private_start_sample(state: Mapping[str, object]) -> dict[str, np.ndarray]:
    names = {
        "four_velocity_mm_ns": "_intrinsic_spin_start_four_velocity",
        "non_self_four_acceleration_mm_ns2": (
            "_intrinsic_spin_start_non_self_four_acceleration"
        ),
        "physical_spin_four_native": "_intrinsic_spin_start_physical_four_spin",
    }
    sample: dict[str, np.ndarray] = {}
    for public_name, private_name in names.items():
        values = np.asarray(state.get(private_name), dtype=float)
        if values.shape != (1, 4) or not np.all(np.isfinite(values)):
            raise ValueError(
                "accepted exact-pair state has no finite pre-self-reaction "
                f"sample {private_name}"
            )
        sample[public_name] = np.array(values[0], copy=True)
    return sample


def build_accepted_pair_intrinsic_spin_reduction_candidate(
    trial: "ExactPairStepDoublingTrial",
    accepted: AcceptedPairIntrinsicSpinReductionHistory,
) -> AcceptedPairIntrinsicSpinReductionHistory:
    """Build the diagnostic history for one authoritative two-half path.

    The midpoint state contains the leading sample at the accepted slab start;
    the refined endpoint state contains the leading sample at the midpoint.
    The endpoint proper time is advanced by both independently solved proper
    increments and retained for the next slab.  This pure function is intended
    to run before the trajectory commit, so any malformed private sample fails
    without partially publishing rider or driver state.
    """

    def append_role(
        history: AcceptedIntrinsicSpinReductionHistory,
        endpoint_proper_time_ns: float,
        midpoint_state: Mapping[str, object],
        endpoint_state: Mapping[str, object],
        midpoint_step_ns: float,
        endpoint_step_ns: float,
    ) -> tuple[AcceptedIntrinsicSpinReductionHistory, float]:
        start_sample = _private_start_sample(midpoint_state)
        with_start = history.append_accepted(
            proper_time_ns=endpoint_proper_time_ns,
            **start_sample,
        )
        midpoint_time = endpoint_proper_time_ns + float(midpoint_step_ns)
        midpoint_sample = _private_start_sample(endpoint_state)
        with_midpoint = with_start.append_accepted(
            proper_time_ns=midpoint_time,
            **midpoint_sample,
        )
        return with_midpoint, midpoint_time + float(endpoint_step_ns)

    rider, rider_endpoint = append_role(
        accepted.rider,
        accepted.rider_endpoint_proper_time_ns,
        trial.midpoint.pair.rider.state,
        trial.refined.pair.rider.state,
        trial.midpoint.pair.rider.proper_step_ns,
        trial.refined.pair.rider.proper_step_ns,
    )
    driver, driver_endpoint = append_role(
        accepted.driver,
        accepted.driver_endpoint_proper_time_ns,
        trial.midpoint.pair.driver.state,
        trial.refined.pair.driver.state,
        trial.midpoint.pair.driver.proper_step_ns,
        trial.refined.pair.driver.proper_step_ns,
    )
    return AcceptedPairIntrinsicSpinReductionHistory(
        rider=rider,
        driver=driver,
        rider_endpoint_proper_time_ns=rider_endpoint,
        driver_endpoint_proper_time_ns=driver_endpoint,
    )


@dataclass(frozen=True)
class IntrinsicSpinReductionRouteResult:
    """Explicit analytical, causal-boundary, or unavailable route selection."""

    route: str
    analytical_reduction: PotentialDirectionalIntrinsicSpinReductionResult | None
    causal_reduction: SampledIntrinsicSpinReductionResult | None
    unavailable_reason: str | None


def select_intrinsic_spin_reduction_route_native(
    *,
    analytical_reduction: PotentialDirectionalIntrinsicSpinReductionResult | None,
    analytical_unavailable_reason: str | None,
    accepted_history: AcceptedIntrinsicSpinReductionHistory,
    charge_native: float,
    mass_amu: float,
    g_factor: float,
) -> IntrinsicSpinReductionRouteResult:
    """Select one diagnostic route without applying either result as a force."""

    if analytical_reduction is not None:
        if analytical_unavailable_reason is not None:
            raise ValueError(
                "an analytical reduction cannot also carry an unavailable reason"
            )
        return IntrinsicSpinReductionRouteResult(
            route="analytical_smooth_segment",
            analytical_reduction=analytical_reduction,
            causal_reduction=None,
            unavailable_reason=None,
        )
    if analytical_unavailable_reason is None:
        raise ValueError(
            "missing analytical reduction must include its unavailability reason"
        )
    if not accepted_history.causal_reduction_ready:
        return IntrinsicSpinReductionRouteResult(
            route="unavailable_insufficient_accepted_history",
            analytical_reduction=None,
            causal_reduction=None,
            unavailable_reason=analytical_unavailable_reason,
        )
    causal = accepted_history.evaluate_causal(
        charge_native=charge_native,
        mass_amu=mass_amu,
        g_factor=g_factor,
    )
    return IntrinsicSpinReductionRouteResult(
        route="causal_accepted_history_boundary_fallback",
        analytical_reduction=None,
        causal_reduction=causal,
        unavailable_reason=analytical_unavailable_reason,
    )


__all__ = [
    "AcceptedPairIntrinsicSpinReductionHistory",
    "AcceptedIntrinsicSpinReductionHistory",
    "IntrinsicSpinReductionRouteResult",
    "build_accepted_pair_intrinsic_spin_reduction_candidate",
    "select_intrinsic_spin_reduction_route_native",
]
