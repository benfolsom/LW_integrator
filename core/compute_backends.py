"""Portable backend boundary for exact retarded-field kernel experiments.

The maintained solver remains CPU/NumPy code.  This module is an isolated
study seam for one bounded accelerator candidate: finding light-cone knot
brackets in large batches.  It does not dispatch the RFS force, root solve,
quintic interpolation, field construction, or reduction to a GPU.

``auto`` intentionally resolves to ``cpu`` in this first stage.  Requesting
``metal`` explicitly requires a detected Apple-silicon macOS host and a
separately installed adapter.  The optional adapter is imported only after
those checks, so Linux, Windows, and unsupported Macs never import Metal-only
packages during normal module import or CPU selection.

Metal GPUs do not provide the float64 path used by the authoritative solver.
An accelerator may therefore propose float32 knot segments, but
:func:`certify_candidate_segments_float64` validates them against the original
float64 inputs and falls back to the CPU scan.  Approximate candidates never
become physical results directly.  A conservative, cacheable timelike-chord
check proves that stored residuals decrease strictly for every observer.  On
histories that pass, two float64 endpoint residuals certify a candidate in
constant work.  An absent proof, boundary ambiguity, or failed proposal uses
the complete authoritative CPU scan.
"""

from __future__ import annotations

import importlib
import platform
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, Sequence, cast

import numpy as np

from .constants import C_MMNS


class ComputeBackendName(str, Enum):
    """Backend names accepted by the study interface."""

    AUTO = "auto"
    CPU = "cpu"
    METAL = "metal"


class ComputeBackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested accelerator cannot be used."""


@dataclass(frozen=True)
class KnotScanBatch:
    """Dense batch used to find retarded light-cone history segments.

    Observer arrays have shapes ``[events]`` and ``[events, 3]``.  Source
    arrays have shapes ``[knots, sources]`` and ``[knots, sources, 3]``.
    ``alive_counts[source]`` excludes history samples after particle loss.
    """

    observer_time_ns: np.ndarray
    observer_position_mm: np.ndarray
    source_time_ns: np.ndarray
    source_position_mm: np.ndarray
    alive_counts: np.ndarray

    def __post_init__(self) -> None:
        observer_time = np.asarray(self.observer_time_ns, dtype=np.float64)
        observer_position = np.asarray(self.observer_position_mm, dtype=np.float64)
        source_time = np.asarray(self.source_time_ns, dtype=np.float64)
        source_position = np.asarray(self.source_position_mm, dtype=np.float64)
        alive_counts = np.asarray(self.alive_counts, dtype=np.int64)

        if observer_time.ndim != 1:
            raise ValueError("observer_time_ns must have shape [events]")
        if observer_position.shape != (observer_time.size, 3):
            raise ValueError("observer_position_mm must have shape [events, 3]")
        if source_time.ndim != 2:
            raise ValueError("source_time_ns must have shape [knots, sources]")
        expected_source_position_shape = (*source_time.shape, 3)
        if source_position.shape != expected_source_position_shape:
            raise ValueError("source_position_mm must have shape [knots, sources, 3]")
        if alive_counts.shape != (source_time.shape[1],):
            raise ValueError("alive_counts must contain one value per source")
        if np.any(alive_counts < 0) or np.any(alive_counts > source_time.shape[0]):
            raise ValueError("alive_counts values are outside the history bounds")
        for name, value in (
            ("observer_time_ns", observer_time),
            ("observer_position_mm", observer_position),
            ("source_time_ns", source_time),
            ("source_position_mm", source_position),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
        for source_index, alive_count in enumerate(alive_counts):
            count = int(alive_count)
            if count > 1 and np.any(np.diff(source_time[:count, source_index]) <= 0.0):
                raise ValueError(
                    "alive source coordinate times must be strictly increasing"
                )

        object.__setattr__(self, "observer_time_ns", observer_time)
        object.__setattr__(self, "observer_position_mm", observer_position)
        object.__setattr__(self, "source_time_ns", source_time)
        object.__setattr__(self, "source_position_mm", source_position)
        object.__setattr__(self, "alive_counts", alive_counts)

    @property
    def event_count(self) -> int:
        return int(self.observer_time_ns.size)

    @property
    def source_count(self) -> int:
        return int(self.source_time_ns.shape[1])


@dataclass(frozen=True)
class BackendSelfTestResult:
    """Result returned by an optional backend's mandatory startup self-test."""

    passed: bool
    detail: str


@dataclass(frozen=True)
class BackendCapabilities:
    """Numerical and platform claims made by one knot-scan backend."""

    name: str
    device_name: str
    accelerator: bool
    exact_float64: bool
    approximate_candidates_only: bool


class KnotScanBackend(Protocol):
    """Minimal interface an optional light-cone candidate backend must expose."""

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def self_test(self) -> BackendSelfTestResult: ...

    def candidate_segments(self, batch: KnotScanBatch) -> np.ndarray:
        """Return proposed segment indices with shape ``[events, sources]``."""


@dataclass(frozen=True)
class BackendResolution:
    """Resolved backend plus a user-facing explanation of the selection."""

    requested: ComputeBackendName
    selected: ComputeBackendName
    backend: KnotScanBackend
    reason: str


@dataclass(frozen=True)
class CandidateCertificationResult:
    """Float64-certified segments and which proposals required CPU fallback."""

    segment_indices: np.ndarray
    accepted_proposals: np.ndarray
    cpu_fallbacks: np.ndarray


def _latest_bracket_from_residuals(residuals: np.ndarray) -> int:
    brackets = np.flatnonzero((residuals[:-1] >= 0.0) & (residuals[1:] <= 0.0))
    return int(brackets[-1]) if brackets.size else -1


def _source_residuals_float64(
    batch: KnotScanBatch, event_index: int, source_index: int
) -> np.ndarray:
    alive_count = int(batch.alive_counts[source_index])
    if alive_count == 0:
        return np.zeros(0, dtype=np.float64)
    displacement = (
        batch.observer_position_mm[event_index, np.newaxis, :]
        - batch.source_position_mm[:alive_count, source_index, :]
    )
    separation = np.linalg.norm(displacement, axis=1)
    residuals = (
        C_MMNS
        * (
            batch.observer_time_ns[event_index]
            - batch.source_time_ns[:alive_count, source_index]
        )
        - separation
    )
    return cast(np.ndarray, residuals)


def _source_residual_at_float64(
    batch: KnotScanBatch,
    event_index: int,
    source_index: int,
    knot_index: int,
) -> float:
    separation = float(
        np.linalg.norm(
            batch.observer_position_mm[event_index]
            - batch.source_position_mm[knot_index, source_index]
        )
    )
    return float(
        C_MMNS
        * (
            batch.observer_time_ns[event_index]
            - batch.source_time_ns[knot_index, source_index]
        )
        - separation
    )


def strictly_timelike_source_chords_float64(batch: KnotScanBatch) -> np.ndarray:
    """Return sources whose stored chords prove strictly decreasing residuals.

    For adjacent source knots, ``|dx| < c dt`` and the reverse triangle
    inequality imply ``g[k + 1] < g[k]`` for every observer.  A small float64
    margin keeps near-null or roundoff-ambiguous chords on the fallback path.
    The result depends only on the source history and may be cached until that
    history is appended or replaced.
    """

    result = np.zeros(batch.source_count, dtype=bool)
    relative_margin = 64.0 * np.finfo(np.float64).eps
    for source_index in range(batch.source_count):
        alive_count = int(batch.alive_counts[source_index])
        if alive_count < 2:
            continue
        delta_time = np.diff(batch.source_time_ns[:alive_count, source_index])
        light_distance = C_MMNS * delta_time
        displacement = np.diff(
            batch.source_position_mm[:alive_count, source_index], axis=0
        )
        chord_distance = np.linalg.norm(displacement, axis=1)
        scale = np.maximum(np.abs(light_distance), np.abs(chord_distance))
        margin = relative_margin * np.maximum(scale, np.finfo(np.float64).tiny)
        result[source_index] = bool(
            np.all(np.isfinite(chord_distance))
            and np.all(chord_distance + margin < light_distance)
        )
    return result


def latest_light_cone_segments_float64(batch: KnotScanBatch) -> np.ndarray:
    """Return the authoritative CPU/NumPy latest bracket for every pair."""

    result: np.ndarray = np.full(
        (batch.event_count, batch.source_count), -1, dtype=np.int64
    )
    for event_index in range(batch.event_count):
        for source_index in range(batch.source_count):
            residuals = _source_residuals_float64(batch, event_index, source_index)
            if residuals.size >= 2:
                result[event_index, source_index] = _latest_bracket_from_residuals(
                    residuals
                )
    return result


def certify_candidate_segments_float64(
    batch: KnotScanBatch,
    proposed_segments: Sequence[Sequence[int]] | np.ndarray,
    *,
    strictly_timelike_sources: Sequence[bool] | np.ndarray | None = None,
) -> CandidateCertificationResult:
    """Certify approximate bracket proposals or replace them on the CPU.

    On a conservatively timelike source history, strict residual monotonicity
    makes two original-float64 endpoint checks sufficient.  Exact internal-knot
    roots reject the segment before the knot so the established latest-bracket
    convention is preserved.  Invalid proposals and sources without a cached
    proof use the complete authoritative scan.  The hybrid result is therefore
    identical to :func:`latest_light_cone_segments_float64`, independent of
    accelerator precision.
    """

    proposals = np.asarray(proposed_segments, dtype=np.int64)
    expected_shape = (batch.event_count, batch.source_count)
    if proposals.shape != expected_shape:
        raise ValueError("proposed_segments must have shape [events, sources]")
    certified: np.ndarray = np.full(expected_shape, -1, dtype=np.int64)
    accepted: np.ndarray = np.zeros(expected_shape, dtype=bool)
    fallbacks: np.ndarray = np.zeros(expected_shape, dtype=bool)

    if strictly_timelike_sources is None:
        timelike_sources = strictly_timelike_source_chords_float64(batch)
    else:
        timelike_sources = np.asarray(strictly_timelike_sources, dtype=bool)
        if timelike_sources.shape != (batch.source_count,):
            raise ValueError(
                "strictly_timelike_sources must contain one value per source"
            )

    for event_index in range(batch.event_count):
        for source_index in range(batch.source_count):
            proposal = int(proposals[event_index, source_index])
            alive_count = int(batch.alive_counts[source_index])
            proposal_certified = False
            if timelike_sources[source_index] and alive_count >= 2:
                if 0 <= proposal < alive_count - 1:
                    lower = _source_residual_at_float64(
                        batch, event_index, source_index, proposal
                    )
                    upper = _source_residual_at_float64(
                        batch, event_index, source_index, proposal + 1
                    )
                    proposal_certified = bool(
                        lower >= 0.0
                        and upper <= 0.0
                        and (upper < 0.0 or proposal + 1 == alive_count - 1)
                    )
                elif proposal == -1:
                    first = _source_residual_at_float64(
                        batch, event_index, source_index, 0
                    )
                    last = _source_residual_at_float64(
                        batch, event_index, source_index, alive_count - 1
                    )
                    proposal_certified = bool(first < 0.0 or last > 0.0)
            if proposal_certified:
                certified[event_index, source_index] = proposal
                accepted[event_index, source_index] = True
            else:
                residuals = _source_residuals_float64(batch, event_index, source_index)
                certified[event_index, source_index] = (
                    _latest_bracket_from_residuals(residuals)
                    if residuals.size >= 2
                    else -1
                )
                fallbacks[event_index, source_index] = True

    return CandidateCertificationResult(
        segment_indices=certified,
        accepted_proposals=accepted,
        cpu_fallbacks=fallbacks,
    )


class CpuKnotScanBackend:
    """Portable exact float64 implementation and permanent fallback."""

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name=ComputeBackendName.CPU.value,
            device_name="NumPy CPU",
            accelerator=False,
            exact_float64=True,
            approximate_candidates_only=False,
        )

    def self_test(self) -> BackendSelfTestResult:
        return BackendSelfTestResult(True, "portable float64 CPU reference")

    def candidate_segments(self, batch: KnotScanBatch) -> np.ndarray:
        return latest_light_cone_segments_float64(batch)


MetalBackendLoader = Callable[[], KnotScanBackend]


def _default_metal_backend_loader() -> KnotScanBackend:
    """Load the optional adapter; called only after macOS/arm64 checks."""

    try:
        module = importlib.import_module("lw_integrator_metal_backend")
    except ImportError as exc:
        raise ComputeBackendUnavailableError(
            "Metal was requested on a supported Mac, but the optional "
            "'lw_integrator_metal_backend' adapter is not installed"
        ) from exc
    factory = getattr(module, "create_knot_scan_backend", None)
    if not callable(factory):
        raise ComputeBackendUnavailableError(
            "the optional Metal adapter does not expose " "create_knot_scan_backend()"
        )
    return cast(KnotScanBackend, factory())


def _validated_metal_backend(loader: MetalBackendLoader) -> KnotScanBackend:
    try:
        backend = loader()
    except ComputeBackendUnavailableError:
        raise
    except Exception as exc:
        raise ComputeBackendUnavailableError(
            f"Metal backend initialization failed: {exc}"
        ) from exc
    capabilities = backend.capabilities
    if capabilities.name != ComputeBackendName.METAL.value:
        raise ComputeBackendUnavailableError(
            "Metal adapter reported an unexpected backend name"
        )
    if not capabilities.accelerator:
        raise ComputeBackendUnavailableError(
            "Metal adapter did not report accelerator capability"
        )
    if not capabilities.approximate_candidates_only:
        raise ComputeBackendUnavailableError(
            "this study accepts Metal only as an approximate candidate provider"
        )
    self_test = backend.self_test()
    if not self_test.passed:
        raise ComputeBackendUnavailableError(
            f"Metal backend self-test failed: {self_test.detail}"
        )
    return backend


def resolve_knot_scan_backend(
    requested: str | ComputeBackendName = ComputeBackendName.AUTO,
    *,
    system_name: str | None = None,
    machine: str | None = None,
    metal_loader: MetalBackendLoader | None = None,
) -> BackendResolution:
    """Resolve a backend without surprising cross-platform imports.

    ``auto`` is deliberately conservative and always selects the CPU in this
    study branch.  ``metal`` is an explicit opt-in and fails clearly unless
    all platform, adapter, and self-test checks pass.
    """

    try:
        selection = (
            requested
            if isinstance(requested, ComputeBackendName)
            else ComputeBackendName(str(requested).lower())
        )
    except ValueError as exc:
        choices = ", ".join(backend.value for backend in ComputeBackendName)
        raise ValueError(f"compute backend must be one of: {choices}") from exc

    cpu = CpuKnotScanBackend()
    if selection is ComputeBackendName.AUTO:
        return BackendResolution(
            requested=selection,
            selected=ComputeBackendName.CPU,
            backend=cpu,
            reason=(
                "auto remains the portable CPU path because native provider "
                "batches are below the measured Metal crossover"
            ),
        )
    if selection is ComputeBackendName.CPU:
        return BackendResolution(
            requested=selection,
            selected=ComputeBackendName.CPU,
            backend=cpu,
            reason="portable CPU backend explicitly selected",
        )

    detected_system = platform.system() if system_name is None else str(system_name)
    detected_machine = platform.machine() if machine is None else str(machine)
    if detected_system != "Darwin":
        raise ComputeBackendUnavailableError(
            "Metal requires macOS; no Metal adapter was imported"
        )
    if detected_machine != "arm64":
        raise ComputeBackendUnavailableError(
            "this prototype Metal backend is validated only on Apple silicon "
            "(arm64); no Metal adapter was imported"
        )
    backend = _validated_metal_backend(
        _default_metal_backend_loader if metal_loader is None else metal_loader
    )
    return BackendResolution(
        requested=selection,
        selected=ComputeBackendName.METAL,
        backend=backend,
        reason="Metal explicitly selected and platform/adapter self-test passed",
    )


__all__ = [
    "BackendCapabilities",
    "BackendResolution",
    "BackendSelfTestResult",
    "CandidateCertificationResult",
    "ComputeBackendName",
    "ComputeBackendUnavailableError",
    "CpuKnotScanBackend",
    "KnotScanBackend",
    "KnotScanBatch",
    "certify_candidate_segments_float64",
    "latest_light_cone_segments_float64",
    "resolve_knot_scan_backend",
    "strictly_timelike_source_chords_float64",
]
