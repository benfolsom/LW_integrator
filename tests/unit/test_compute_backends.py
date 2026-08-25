from __future__ import annotations

from dataclasses import dataclass
import platform

import numpy as np
import pytest

from core.compute_backends import (
    BackendCapabilities,
    BackendSelfTestResult,
    ComputeBackendName,
    ComputeBackendUnavailableError,
    KnotScanBatch,
    certify_candidate_segments_float64,
    latest_light_cone_segments_float64,
    resolve_knot_scan_backend,
    strictly_timelike_source_chords_float64,
)
from core.constants import C_MMNS


def _stationary_batch() -> KnotScanBatch:
    source_times = np.linspace(-0.02, 0.002, 45)[:, np.newaxis]
    source_positions = np.zeros((source_times.shape[0], 1, 3))
    return KnotScanBatch(
        observer_time_ns=np.array((0.0, 0.0005)),
        observer_position_mm=np.array(((1.0, 0.0, 0.0), (1.5, 0.0, 0.0))),
        source_time_ns=source_times,
        source_position_mm=source_positions,
        alive_counts=np.array((source_times.shape[0],)),
    )


@dataclass
class _MockMetalBackend:
    self_test_passed: bool = True

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="metal",
            device_name="mock Apple GPU",
            accelerator=True,
            exact_float64=False,
            approximate_candidates_only=True,
        )

    def self_test(self) -> BackendSelfTestResult:
        return BackendSelfTestResult(self.self_test_passed, "mock parity probe")

    def candidate_segments(self, batch: KnotScanBatch) -> np.ndarray:
        return latest_light_cone_segments_float64(batch)


def test_cpu_scan_returns_latest_analytic_stationary_brackets() -> None:
    batch = _stationary_batch()
    segments = latest_light_cone_segments_float64(batch)
    for event_index in range(batch.event_count):
        retarded_time = (
            batch.observer_time_ns[event_index]
            - batch.observer_position_mm[event_index, 0] / C_MMNS
        )
        segment = int(segments[event_index, 0])
        assert batch.source_time_ns[segment, 0] <= retarded_time
        assert batch.source_time_ns[segment + 1, 0] >= retarded_time


def test_float64_certifier_accepts_valid_proposals_and_falls_back_on_bad_ones() -> None:
    batch = _stationary_batch()
    authoritative = latest_light_cone_segments_float64(batch)
    proposals = authoritative.copy()
    proposals[1, 0] = 0

    certified = certify_candidate_segments_float64(batch, proposals)

    np.testing.assert_array_equal(certified.segment_indices, authoritative)
    assert certified.accepted_proposals.tolist() == [[True], [False]]
    assert certified.cpu_fallbacks.tolist() == [[False], [True]]


def test_float64_certifier_rejects_a_valid_but_nonlatest_bracket() -> None:
    source_times = np.array((-0.004, -0.003, -0.002, -0.001))[:, np.newaxis]
    target_residuals = np.array((0.1, -0.1, 0.1, -0.1))
    source_positions = np.zeros((source_times.shape[0], 1, 3))
    source_positions[:, 0, 0] = -C_MMNS * source_times[:, 0] - target_residuals
    batch = KnotScanBatch(
        observer_time_ns=np.array((0.0,)),
        observer_position_mm=np.zeros((1, 3)),
        source_time_ns=source_times,
        source_position_mm=source_positions,
        alive_counts=np.array((source_times.shape[0],)),
    )

    certified = certify_candidate_segments_float64(batch, np.array(((0,),)))

    assert certified.segment_indices.tolist() == [[2]]
    assert certified.accepted_proposals.tolist() == [[False]]
    assert certified.cpu_fallbacks.tolist() == [[True]]


def test_constant_work_certifier_preserves_exact_internal_knot_convention() -> None:
    source_times = np.array((-0.002, -0.001, 0.0))[:, np.newaxis]
    batch = KnotScanBatch(
        observer_time_ns=np.array((-0.001,)),
        observer_position_mm=np.zeros((1, 3)),
        source_time_ns=source_times,
        source_position_mm=np.zeros((3, 1, 3)),
        alive_counts=np.array((3,)),
    )

    assert strictly_timelike_source_chords_float64(batch).tolist() == [True]
    preceding = certify_candidate_segments_float64(batch, np.array(((0,),)))
    following = certify_candidate_segments_float64(batch, np.array(((1,),)))

    assert preceding.segment_indices.tolist() == [[1]]
    assert preceding.accepted_proposals.tolist() == [[False]]
    assert preceding.cpu_fallbacks.tolist() == [[True]]
    assert following.segment_indices.tolist() == [[1]]
    assert following.accepted_proposals.tolist() == [[True]]
    assert following.cpu_fallbacks.tolist() == [[False]]


def test_timelike_proof_can_be_cached_and_near_null_chords_fall_back() -> None:
    batch = _stationary_batch()
    proof = strictly_timelike_source_chords_float64(batch)
    authoritative = latest_light_cone_segments_float64(batch)

    certified = certify_candidate_segments_float64(
        batch,
        authoritative,
        strictly_timelike_sources=proof,
    )

    np.testing.assert_array_equal(certified.segment_indices, authoritative)
    assert np.all(certified.accepted_proposals)
    with pytest.raises(ValueError, match="one value per source"):
        certify_candidate_segments_float64(
            batch,
            authoritative,
            strictly_timelike_sources=np.array((True, False)),
        )


def test_auto_is_safe_cpu_and_never_probes_metal_even_on_a_mac() -> None:
    loader_called = False

    def forbidden_loader() -> _MockMetalBackend:
        nonlocal loader_called
        loader_called = True
        raise AssertionError("auto must not probe the optional Metal adapter")

    resolution = resolve_knot_scan_backend(
        "auto",
        system_name="Darwin",
        machine="arm64",
        metal_loader=forbidden_loader,
    )

    assert resolution.selected is ComputeBackendName.CPU
    assert resolution.backend.capabilities.exact_float64 is True
    assert loader_called is False


def test_explicit_cpu_is_portable_and_never_probes_metal() -> None:
    def forbidden_loader() -> _MockMetalBackend:
        raise AssertionError("CPU selection must not probe Metal")

    resolution = resolve_knot_scan_backend(
        ComputeBackendName.CPU,
        system_name="Linux",
        machine="x86_64",
        metal_loader=forbidden_loader,
    )
    assert resolution.selected is ComputeBackendName.CPU


@pytest.mark.parametrize(
    ("system_name", "machine", "message"),
    [
        ("Linux", "x86_64", "requires macOS"),
        ("Darwin", "x86_64", "only on Apple silicon"),
    ],
)
def test_explicit_metal_fails_before_import_on_unsupported_platforms(
    system_name: str, machine: str, message: str
) -> None:
    loader_called = False

    def forbidden_loader() -> _MockMetalBackend:
        nonlocal loader_called
        loader_called = True
        raise AssertionError("unsupported platforms must not import Metal")

    with pytest.raises(ComputeBackendUnavailableError, match=message):
        resolve_knot_scan_backend(
            "metal",
            system_name=system_name,
            machine=machine,
            metal_loader=forbidden_loader,
        )
    assert loader_called is False


def test_explicit_metal_requires_adapter_and_passing_self_test() -> None:
    with pytest.raises(ComputeBackendUnavailableError, match="initialization failed"):
        resolve_knot_scan_backend(
            "metal",
            system_name="Darwin",
            machine="arm64",
            metal_loader=lambda: (_ for _ in ()).throw(ImportError("not installed")),
        )

    with pytest.raises(ComputeBackendUnavailableError, match="self-test failed"):
        resolve_knot_scan_backend(
            "metal",
            system_name="Darwin",
            machine="arm64",
            metal_loader=lambda: _MockMetalBackend(self_test_passed=False),
        )


def test_explicit_metal_returns_only_validated_candidate_backend() -> None:
    backend = _MockMetalBackend()
    resolution = resolve_knot_scan_backend(
        "metal",
        system_name="Darwin",
        machine="arm64",
        metal_loader=lambda: backend,
    )
    assert resolution.selected is ComputeBackendName.METAL
    assert resolution.backend is backend
    assert resolution.backend.capabilities.approximate_candidates_only is True
    assert resolution.backend.capabilities.exact_float64 is False


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="real Metal adapter requires Apple-silicon macOS",
)
def test_real_metal_adapter_passes_startup_certification() -> None:
    resolution = resolve_knot_scan_backend("metal")

    assert resolution.selected is ComputeBackendName.METAL
    assert resolution.backend.self_test().passed is True


def test_invalid_selection_and_batch_shapes_fail_clearly() -> None:
    with pytest.raises(ValueError, match="compute backend must be one of"):
        resolve_knot_scan_backend("cuda")
    with pytest.raises(ValueError, match="observer_position_mm"):
        KnotScanBatch(
            observer_time_ns=np.zeros(2),
            observer_position_mm=np.zeros((1, 3)),
            source_time_ns=np.zeros((2, 1)),
            source_position_mm=np.zeros((2, 1, 3)),
            alive_counts=np.array((2,)),
        )
