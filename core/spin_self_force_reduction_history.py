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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence, cast

import numpy as np

from .spin_self_force_reduction_oracle import (
    PotentialDirectionalIntrinsicSpinReductionResult,
    SampledIntrinsicSpinReductionResult,
    evaluate_causal_sampled_intrinsic_spin_reduction_native,
)

if TYPE_CHECKING:
    from .exact_pair_trial import ExactPairStepDoublingTrial

_CHECKPOINT_SCHEMA_VERSION = 2
_MINIMUM_CAUSAL_SAMPLES = 6
_MAXIMUM_DIAGNOSTIC_RECORDS = 4096
_MAXIMUM_CAUSAL_CONDITION_NUMBER = 1.0e5


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


def _finite_four_tuple(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> tuple[float, float, float, float]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (4,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite four-vector")
    return cast(
        tuple[float, float, float, float], tuple(float(item) for item in vector)
    )


@dataclass(frozen=True)
class IntrinsicSpinReductionDiagnosticRecord:
    """One accepted diagnostic evaluation; never an applied force."""

    proper_time_ns: float
    route: str
    analytical_unavailable_reason: str | None
    causal_condition_number: float | None
    linear_spin_four_force_native: tuple[float, float, float, float] | None
    charge_ald_four_force_native: tuple[float, float, float, float] | None
    total_four_force_native: tuple[float, float, float, float] | None
    balance_residual_norm_native: float | None

    def __post_init__(self) -> None:
        time = float(self.proper_time_ns)
        if not np.isfinite(time):
            raise ValueError("diagnostic proper_time_ns must be finite")
        route = str(self.route)
        valid_routes = {
            "analytical_smooth_segment",
            "causal_accepted_history_boundary_fallback",
            "unavailable_insufficient_accepted_history",
            "unavailable_ill_conditioned_causal_fit",
            "unavailable_outside_intrinsic_qmu_model",
        }
        if route not in valid_routes:
            raise ValueError(f"unsupported intrinsic-spin diagnostic route: {route}")
        object.__setattr__(self, "proper_time_ns", time)
        object.__setattr__(self, "route", route)
        condition = self.causal_condition_number
        if condition is not None:
            condition = float(condition)
            if not np.isfinite(condition) or condition <= 0.0:
                raise ValueError("causal condition number must be finite and positive")
            object.__setattr__(self, "causal_condition_number", condition)
        vectors = (
            "linear_spin_four_force_native",
            "charge_ald_four_force_native",
            "total_four_force_native",
        )
        available = not route.startswith("unavailable_")
        for name in vectors:
            value = getattr(self, name)
            if available and value is None:
                raise ValueError(f"available diagnostic route requires {name}")
            if not available and value is not None:
                raise ValueError(f"unavailable diagnostic route cannot contain {name}")
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_four_tuple(value, name=name),
                )
        residual = self.balance_residual_norm_native
        if available:
            if (
                residual is None
                or not np.isfinite(float(residual))
                or float(residual) < 0.0
            ):
                raise ValueError(
                    "available diagnostic route requires a finite non-negative "
                    "balance residual norm"
                )
            object.__setattr__(self, "balance_residual_norm_native", float(residual))
        elif residual is not None:
            raise ValueError("unavailable diagnostic route cannot contain a residual")

    def to_checkpoint_payload(self) -> dict[str, object]:
        return {
            "proper_time_ns": self.proper_time_ns,
            "route": self.route,
            "analytical_unavailable_reason": self.analytical_unavailable_reason,
            "causal_condition_number": self.causal_condition_number,
            "linear_spin_four_force_native": self.linear_spin_four_force_native,
            "charge_ald_four_force_native": self.charge_ald_four_force_native,
            "total_four_force_native": self.total_four_force_native,
            "balance_residual_norm_native": self.balance_residual_norm_native,
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "IntrinsicSpinReductionDiagnosticRecord":
        required = {
            "proper_time_ns",
            "route",
            "analytical_unavailable_reason",
            "causal_condition_number",
            "linear_spin_four_force_native",
            "charge_ald_four_force_native",
            "total_four_force_native",
            "balance_residual_norm_native",
        }
        if set(payload) != required:
            raise ValueError("intrinsic-spin diagnostic record keys do not match")

        def optional_vector(name: str) -> tuple[float, float, float, float] | None:
            value = payload[name]
            if value is None:
                return None
            return _finite_four_tuple(cast(Sequence[float], value), name=name)

        reason = payload["analytical_unavailable_reason"]
        if reason is not None and not isinstance(reason, str):
            raise ValueError("analytical_unavailable_reason must be a string or null")
        return cls(
            proper_time_ns=float(cast(float, payload["proper_time_ns"])),
            route=str(payload["route"]),
            analytical_unavailable_reason=reason,
            causal_condition_number=(
                None
                if payload["causal_condition_number"] is None
                else float(cast(float, payload["causal_condition_number"]))
            ),
            linear_spin_four_force_native=optional_vector(
                "linear_spin_four_force_native"
            ),
            charge_ald_four_force_native=optional_vector(
                "charge_ald_four_force_native"
            ),
            total_four_force_native=optional_vector("total_four_force_native"),
            balance_residual_norm_native=(
                None
                if payload["balance_residual_norm_native"] is None
                else float(cast(float, payload["balance_residual_norm_native"]))
            ),
        )


@dataclass(frozen=True)
class IntrinsicSpinReductionDiagnosticTrace:
    """Bounded recent records plus lifetime route counters."""

    records: tuple[IntrinsicSpinReductionDiagnosticRecord, ...] = ()
    total_records: int = 0
    analytical_records: int = 0
    causal_records: int = 0
    unavailable_records: int = 0
    maximum_records: int = _MAXIMUM_DIAGNOSTIC_RECORDS

    def __post_init__(self) -> None:
        maximum = int(self.maximum_records)
        counters = (
            int(self.total_records),
            int(self.analytical_records),
            int(self.causal_records),
            int(self.unavailable_records),
        )
        if maximum < 1 or any(value < 0 for value in counters):
            raise ValueError("diagnostic trace counts must be non-negative")
        if sum(counters[1:]) != counters[0]:
            raise ValueError("diagnostic route counters must sum to total_records")
        records = tuple(self.records)
        if len(records) > maximum or len(records) > counters[0]:
            raise ValueError("diagnostic trace retention is inconsistent")
        if any(
            records[index].proper_time_ns <= records[index - 1].proper_time_ns
            for index in range(1, len(records))
        ):
            raise ValueError("diagnostic record proper times must increase strictly")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "maximum_records", maximum)
        object.__setattr__(self, "total_records", counters[0])
        object.__setattr__(self, "analytical_records", counters[1])
        object.__setattr__(self, "causal_records", counters[2])
        object.__setattr__(self, "unavailable_records", counters[3])

    def append(
        self,
        record: IntrinsicSpinReductionDiagnosticRecord,
    ) -> "IntrinsicSpinReductionDiagnosticTrace":
        if self.records and record.proper_time_ns <= self.records[-1].proper_time_ns:
            raise ValueError("diagnostic record proper times must increase strictly")
        records = (self.records + (record,))[-self.maximum_records :]
        return IntrinsicSpinReductionDiagnosticTrace(
            records=records,
            total_records=self.total_records + 1,
            analytical_records=self.analytical_records
            + int(record.route == "analytical_smooth_segment"),
            causal_records=self.causal_records
            + int(record.route == "causal_accepted_history_boundary_fallback"),
            unavailable_records=self.unavailable_records
            + int(record.route.startswith("unavailable_")),
            maximum_records=self.maximum_records,
        )

    def to_checkpoint_payload(self) -> dict[str, object]:
        return {
            "maximum_records": self.maximum_records,
            "total_records": self.total_records,
            "analytical_records": self.analytical_records,
            "causal_records": self.causal_records,
            "unavailable_records": self.unavailable_records,
            "records": [record.to_checkpoint_payload() for record in self.records],
        }

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "IntrinsicSpinReductionDiagnosticTrace":
        required = {
            "maximum_records",
            "total_records",
            "analytical_records",
            "causal_records",
            "unavailable_records",
            "records",
        }
        if set(payload) != required or not isinstance(payload["records"], list):
            raise ValueError("intrinsic-spin diagnostic trace keys are invalid")
        raw_records = payload["records"]
        if any(not isinstance(record, Mapping) for record in raw_records):
            raise ValueError("intrinsic-spin diagnostic records must be JSON objects")
        return cls(
            records=tuple(
                IntrinsicSpinReductionDiagnosticRecord.from_checkpoint_payload(record)
                for record in raw_records
            ),
            total_records=int(cast(int, payload["total_records"])),
            analytical_records=int(cast(int, payload["analytical_records"])),
            causal_records=int(cast(int, payload["causal_records"])),
            unavailable_records=int(cast(int, payload["unavailable_records"])),
            maximum_records=int(cast(int, payload["maximum_records"])),
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
    rider_diagnostics: IntrinsicSpinReductionDiagnosticTrace = field(
        default_factory=IntrinsicSpinReductionDiagnosticTrace
    )
    driver_diagnostics: IntrinsicSpinReductionDiagnosticTrace = field(
        default_factory=IntrinsicSpinReductionDiagnosticTrace
    )

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
            rider_diagnostics=IntrinsicSpinReductionDiagnosticTrace(),
            driver_diagnostics=IntrinsicSpinReductionDiagnosticTrace(),
        )

    def to_checkpoint_payload(self) -> dict[str, object]:
        """Return the complete pair as strict finite JSON-compatible data."""

        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "rider": self.rider.to_checkpoint_payload(),
            "driver": self.driver.to_checkpoint_payload(),
            "rider_endpoint_proper_time_ns": self.rider_endpoint_proper_time_ns,
            "driver_endpoint_proper_time_ns": self.driver_endpoint_proper_time_ns,
            "rider_diagnostics": self.rider_diagnostics.to_checkpoint_payload(),
            "driver_diagnostics": self.driver_diagnostics.to_checkpoint_payload(),
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
            "rider_diagnostics",
            "driver_diagnostics",
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
        rider_diagnostics = payload["rider_diagnostics"]
        driver_diagnostics = payload["driver_diagnostics"]
        if not isinstance(rider_diagnostics, Mapping) or not isinstance(
            driver_diagnostics, Mapping
        ):
            raise ValueError("pair intrinsic-spin diagnostics must be JSON objects")
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
            rider_diagnostics=(
                IntrinsicSpinReductionDiagnosticTrace.from_checkpoint_payload(
                    rider_diagnostics
                )
            ),
            driver_diagnostics=(
                IntrinsicSpinReductionDiagnosticTrace.from_checkpoint_payload(
                    driver_diagnostics
                )
            ),
        )


def _private_start_sample(
    state: Mapping[str, object],
) -> dict[str, tuple[float, float, float, float]]:
    names = {
        "four_velocity_mm_ns": "_intrinsic_spin_start_four_velocity",
        "non_self_four_acceleration_mm_ns2": (
            "_intrinsic_spin_start_non_self_four_acceleration"
        ),
        "physical_spin_four_native": "_intrinsic_spin_start_physical_four_spin",
    }
    sample: dict[str, tuple[float, float, float, float]] = {}
    for public_name, private_name in names.items():
        values = np.asarray(state.get(private_name), dtype=float)
        if values.shape != (1, 4) or not np.all(np.isfinite(values)):
            raise ValueError(
                "accepted exact-pair state has no finite pre-self-reaction "
                f"sample {private_name}"
            )
        sample[public_name] = _finite_four_tuple(values[0], name=private_name)
    return sample


def _private_route_inputs(
    state: Mapping[str, object],
) -> tuple[
    PotentialDirectionalIntrinsicSpinReductionResult | None,
    str | None,
    float,
    float,
    float,
]:
    reductions = state.get("_intrinsic_spin_start_analytical_reduction")
    reasons = state.get("_intrinsic_spin_start_analytical_unavailable_reason")
    if not isinstance(reductions, list) or len(reductions) != 1:
        raise ValueError("accepted exact-pair state has no analytical reduction slot")
    analytical = reductions[0]
    if analytical is not None and not isinstance(
        analytical, PotentialDirectionalIntrinsicSpinReductionResult
    ):
        raise ValueError("accepted exact-pair analytical reduction has invalid type")
    if not isinstance(reasons, list) or len(reasons) != 1:
        raise ValueError("accepted exact-pair state has no analytical reason slot")
    reason = reasons[0]
    if reason is not None and not isinstance(reason, str):
        raise ValueError("accepted exact-pair analytical reason must be text or null")

    def scalar(name: str) -> float:
        values = np.asarray(state.get(name), dtype=float)
        if values.shape != (1,) or not np.isfinite(values[0]):
            raise ValueError(f"accepted exact-pair state has no finite {name}")
        return float(values[0])

    return (
        analytical,
        reason,
        scalar("_intrinsic_spin_charge_native"),
        scalar("_intrinsic_spin_mass_amu"),
        scalar("_intrinsic_spin_g_factor"),
    )


def _diagnostic_record(
    *,
    proper_time_ns: float,
    state: Mapping[str, object],
    accepted_history: AcceptedIntrinsicSpinReductionHistory,
) -> IntrinsicSpinReductionDiagnosticRecord:
    analytical, reason, charge, mass, g_factor = _private_route_inputs(state)
    if charge == 0.0 or g_factor == 0.0:
        return IntrinsicSpinReductionDiagnosticRecord(
            proper_time_ns=proper_time_ns,
            route="unavailable_outside_intrinsic_qmu_model",
            analytical_unavailable_reason=reason,
            causal_condition_number=None,
            linear_spin_four_force_native=None,
            charge_ald_four_force_native=None,
            total_four_force_native=None,
            balance_residual_norm_native=None,
        )
    selected = select_intrinsic_spin_reduction_route_native(
        analytical_reduction=analytical,
        analytical_unavailable_reason=reason,
        accepted_history=accepted_history,
        charge_native=charge,
        mass_amu=mass,
        g_factor=g_factor,
    )
    reduction = selected.analytical_reduction or selected.causal_reduction
    if reduction is None:
        return IntrinsicSpinReductionDiagnosticRecord(
            proper_time_ns=proper_time_ns,
            route=selected.route,
            analytical_unavailable_reason=selected.unavailable_reason,
            causal_condition_number=selected.causal_condition_number,
            linear_spin_four_force_native=None,
            charge_ald_four_force_native=None,
            total_four_force_native=None,
            balance_residual_norm_native=None,
        )
    balance = reduction.radiation_balance
    self_force = balance.self_force
    condition = (
        None
        if selected.causal_reduction is None
        else selected.causal_reduction.scaled_vandermonde_condition_number
    )
    return IntrinsicSpinReductionDiagnosticRecord(
        proper_time_ns=proper_time_ns,
        route=selected.route,
        analytical_unavailable_reason=selected.unavailable_reason,
        causal_condition_number=condition,
        linear_spin_four_force_native=(self_force.linear_spin_self_force_native),
        charge_ald_four_force_native=self_force.charge_ald_self_force_native,
        total_four_force_native=(
            self_force.total_self_force_through_linear_spin_native
        ),
        balance_residual_norm_native=float(
            np.linalg.norm(balance.balance_residual_native)
        ),
    )


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
            four_velocity_mm_ns=start_sample["four_velocity_mm_ns"],
            non_self_four_acceleration_mm_ns2=start_sample[
                "non_self_four_acceleration_mm_ns2"
            ],
            physical_spin_four_native=start_sample["physical_spin_four_native"],
        )
        midpoint_time = endpoint_proper_time_ns + float(midpoint_step_ns)
        midpoint_sample = _private_start_sample(endpoint_state)
        with_midpoint = with_start.append_accepted(
            proper_time_ns=midpoint_time,
            four_velocity_mm_ns=midpoint_sample["four_velocity_mm_ns"],
            non_self_four_acceleration_mm_ns2=midpoint_sample[
                "non_self_four_acceleration_mm_ns2"
            ],
            physical_spin_four_native=midpoint_sample["physical_spin_four_native"],
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
        rider_diagnostics=accepted.rider_diagnostics,
        driver_diagnostics=accepted.driver_diagnostics,
    )


def build_accepted_pair_intrinsic_spin_reduction_diagnostic_candidate(
    trial: "ExactPairStepDoublingTrial",
    accepted: AcceptedPairIntrinsicSpinReductionHistory,
) -> AcceptedPairIntrinsicSpinReductionHistory:
    """Advance accepted histories and append live diagnostic route records."""

    def append_role(
        history: AcceptedIntrinsicSpinReductionHistory,
        trace: IntrinsicSpinReductionDiagnosticTrace,
        endpoint_proper_time_ns: float,
        midpoint_state: Mapping[str, object],
        endpoint_state: Mapping[str, object],
        midpoint_step_ns: float,
        endpoint_step_ns: float,
    ) -> tuple[
        AcceptedIntrinsicSpinReductionHistory,
        IntrinsicSpinReductionDiagnosticTrace,
        float,
    ]:
        start_sample = _private_start_sample(midpoint_state)
        with_start = history.append_accepted(
            proper_time_ns=endpoint_proper_time_ns,
            four_velocity_mm_ns=start_sample["four_velocity_mm_ns"],
            non_self_four_acceleration_mm_ns2=start_sample[
                "non_self_four_acceleration_mm_ns2"
            ],
            physical_spin_four_native=start_sample["physical_spin_four_native"],
        )
        next_trace = trace.append(
            _diagnostic_record(
                proper_time_ns=endpoint_proper_time_ns,
                state=midpoint_state,
                accepted_history=with_start,
            )
        )
        midpoint_time = endpoint_proper_time_ns + float(midpoint_step_ns)
        midpoint_sample = _private_start_sample(endpoint_state)
        with_midpoint = with_start.append_accepted(
            proper_time_ns=midpoint_time,
            four_velocity_mm_ns=midpoint_sample["four_velocity_mm_ns"],
            non_self_four_acceleration_mm_ns2=midpoint_sample[
                "non_self_four_acceleration_mm_ns2"
            ],
            physical_spin_four_native=midpoint_sample["physical_spin_four_native"],
        )
        next_trace = next_trace.append(
            _diagnostic_record(
                proper_time_ns=midpoint_time,
                state=endpoint_state,
                accepted_history=with_midpoint,
            )
        )
        return with_midpoint, next_trace, midpoint_time + float(endpoint_step_ns)

    rider, rider_trace, rider_endpoint = append_role(
        accepted.rider,
        accepted.rider_diagnostics,
        accepted.rider_endpoint_proper_time_ns,
        trial.midpoint.pair.rider.state,
        trial.refined.pair.rider.state,
        trial.midpoint.pair.rider.proper_step_ns,
        trial.refined.pair.rider.proper_step_ns,
    )
    driver, driver_trace, driver_endpoint = append_role(
        accepted.driver,
        accepted.driver_diagnostics,
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
        rider_diagnostics=rider_trace,
        driver_diagnostics=driver_trace,
    )


@dataclass(frozen=True)
class IntrinsicSpinReductionRouteResult:
    """Explicit analytical, causal-boundary, or unavailable route selection."""

    route: str
    analytical_reduction: PotentialDirectionalIntrinsicSpinReductionResult | None
    causal_reduction: SampledIntrinsicSpinReductionResult | None
    causal_condition_number: float | None
    unavailable_reason: str | None


def select_intrinsic_spin_reduction_route_native(
    *,
    analytical_reduction: PotentialDirectionalIntrinsicSpinReductionResult | None,
    analytical_unavailable_reason: str | None,
    accepted_history: AcceptedIntrinsicSpinReductionHistory,
    charge_native: float,
    mass_amu: float,
    g_factor: float,
    maximum_causal_condition_number: float = _MAXIMUM_CAUSAL_CONDITION_NUMBER,
) -> IntrinsicSpinReductionRouteResult:
    """Select one diagnostic route without applying either result as a force.

    A causal derivative fit above ``maximum_causal_condition_number`` is
    reported as unavailable.  The condition number measures how strongly
    sample noise and roundoff can be amplified by the unequal-step derivative
    solve; returning no force is safer than passing an unstable estimate to a
    future applied mode.
    """

    condition_limit = float(maximum_causal_condition_number)
    if not np.isfinite(condition_limit) or condition_limit <= 0.0:
        raise ValueError("maximum_causal_condition_number must be finite and positive")

    if analytical_reduction is not None:
        if analytical_unavailable_reason is not None:
            raise ValueError(
                "an analytical reduction cannot also carry an unavailable reason"
            )
        return IntrinsicSpinReductionRouteResult(
            route="analytical_smooth_segment",
            analytical_reduction=analytical_reduction,
            causal_reduction=None,
            causal_condition_number=None,
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
            causal_condition_number=None,
            unavailable_reason=analytical_unavailable_reason,
        )
    causal = accepted_history.evaluate_causal(
        charge_native=charge_native,
        mass_amu=mass_amu,
        g_factor=g_factor,
    )
    condition = float(causal.scaled_vandermonde_condition_number)
    if condition > condition_limit:
        return IntrinsicSpinReductionRouteResult(
            route="unavailable_ill_conditioned_causal_fit",
            analytical_reduction=None,
            causal_reduction=None,
            causal_condition_number=condition,
            unavailable_reason=analytical_unavailable_reason,
        )
    return IntrinsicSpinReductionRouteResult(
        route="causal_accepted_history_boundary_fallback",
        analytical_reduction=None,
        causal_reduction=causal,
        causal_condition_number=condition,
        unavailable_reason=analytical_unavailable_reason,
    )


__all__ = [
    "AcceptedPairIntrinsicSpinReductionHistory",
    "AcceptedIntrinsicSpinReductionHistory",
    "IntrinsicSpinReductionDiagnosticRecord",
    "IntrinsicSpinReductionDiagnosticTrace",
    "IntrinsicSpinReductionRouteResult",
    "build_accepted_pair_intrinsic_spin_reduction_candidate",
    "build_accepted_pair_intrinsic_spin_reduction_diagnostic_candidate",
    "select_intrinsic_spin_reduction_route_native",
]
