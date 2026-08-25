from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.compute_backends import (
    BackendCapabilities,
    BackendResolution,
    BackendSelfTestResult,
    ComputeBackendName,
    latest_light_cone_segments_float64,
)
from core.metal_certified_roots import (
    certified_metal_segments,
    metal_certified_root_diagnostics,
    reset_metal_certified_root_diagnostics,
)


class _ExactProposalBackend:
    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities("metal", "mock", True, False, True)

    def self_test(self) -> BackendSelfTestResult:
        return BackendSelfTestResult(True, "mock")

    def candidate_segments(self, batch):  # type: ignore[no-untyped-def]
        return latest_light_cone_segments_float64(batch)


def _sources(count: int = 2):  # type: ignore[no-untyped-def]
    times = np.linspace(-0.02, 0.002, 45)
    result = {}
    for index in range(count):
        positions = np.zeros((times.size, 3))
        positions[:, 0] = 0.01 * index
        worldline = SimpleNamespace(
            time_ns=times.copy(),
            position_mm=positions,
            _metal_timelike_count=0,
            _metal_timelike_proof=True,
        )
        result[index] = SimpleNamespace(worldline=worldline)
    return result


def test_certified_metal_segments_obeys_threshold_and_records_dispatch(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    backend = _ExactProposalBackend()
    resolution = BackendResolution(
        requested=ComputeBackendName.METAL,
        selected=ComputeBackendName.METAL,
        backend=backend,
        reason="mock",
    )
    monkeypatch.setattr(
        "core.metal_certified_roots.resolve_knot_scan_backend",
        lambda _name: resolution,
    )
    sources = _sources()
    event_times = np.array((0.0, 0.0005))
    event_positions = np.array(((1.0, 0.0, 0.0), (1.5, 0.0, 0.0)))
    reset_metal_certified_root_diagnostics()

    assert (
        certified_metal_segments(
            sources, event_times, event_positions, event_threshold=3
        )
        is None
    )
    segments = certified_metal_segments(
        sources, event_times, event_positions, event_threshold=2
    )

    assert segments is not None
    assert set(segments) == {0, 1}
    assert all(value.shape == (2,) for value in segments.values())
    diagnostics = metal_certified_root_diagnostics()
    assert diagnostics.calls == 2
    assert diagnostics.below_threshold_calls == 1
    assert diagnostics.dispatches == 1
    assert diagnostics.accepted_proposals == 4
    assert diagnostics.cpu_fallbacks == 0


def test_certified_metal_dispatch_failure_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.metal_certified_roots.resolve_knot_scan_backend",
        lambda _name: (_ for _ in ()).throw(RuntimeError("device lost")),
    )
    reset_metal_certified_root_diagnostics()

    result = certified_metal_segments(
        _sources(1),
        np.array((0.0,)),
        np.array(((1.0, 0.0, 0.0),)),
        event_threshold=1,
    )

    assert result is None
    diagnostics = metal_certified_root_diagnostics()
    assert diagnostics.dispatch_failures == 1
    assert diagnostics.dispatches == 0


def test_timelike_proof_checks_only_appended_chords(monkeypatch) -> None:
    backend = _ExactProposalBackend()
    monkeypatch.setattr(
        "core.metal_certified_roots.resolve_knot_scan_backend",
        lambda _name: BackendResolution(
            ComputeBackendName.METAL,
            ComputeBackendName.METAL,
            backend,
            "mock",
        ),
    )
    sources = _sources(1)
    worldline = sources[0].worldline
    events = np.array((0.0,))
    positions = np.array(((1.0, 0.0, 0.0),))

    assert (
        certified_metal_segments(sources, events, positions, event_threshold=1)
        is not None
    )
    assert worldline._metal_timelike_count == 45
    worldline.time_ns = np.append(worldline.time_ns, 0.003)
    worldline.position_mm = np.vstack((worldline.position_mm, (0.0, 0.0, 0.0)))
    assert (
        certified_metal_segments(sources, events, positions, event_threshold=1)
        is not None
    )
    assert worldline._metal_timelike_count == 46
