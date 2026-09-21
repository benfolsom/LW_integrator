from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import runpy
import json
import numpy as np
import pytest
from types import SimpleNamespace
from core.constants import C_MMNS
from core.single_fit_geometry import fixed_spacing_indices
from core.full_dipole_history import FullDipoleHistory, SourcePositionError


def source(count=61):
    fixture = runpy.run_path(
        str(Path(__file__).with_name("test_connected_dipole_history.py"))
    )["fixture"]
    h = fixture(count)
    t = np.maximum(h.time, 0)
    position = np.zeros_like(h.position)
    velocity = np.zeros_like(h.velocity)
    position[:, 0] = 1e-5 * t**6
    velocity[:, 0] = 6e-5 * t**5
    return replace(h, position=position, velocity=velocity, position_tolerance=1e-12)


def candidate(count=61):
    return source(count).with_single_fit_geometry(
        fit_sample_spacing=2 / 64, dipole_tolerance=1e-20
    )


def test_exact_geometry_and_connections():
    h = candidate()
    for previous, s in zip(h.segments, h.segments[1:]):
        if s.start < 0:
            continue
        assert s.position.shape == (12, 3)
        for order in range(5):
            np.testing.assert_allclose(
                previous.sample(previous.end, order)[0],
                s.sample(s.start, order)[0],
                rtol=1e-10,
                atol=1e-15,
            )
        np.testing.assert_allclose(
            s.sample(s.end)[0][0], 1e-5 * s.end**6, rtol=1e-8, atol=1e-18
        )


def test_restart_and_append_exact_and_preserve_published_prefix():
    original = candidate(35)
    payload = original.to_checkpoint_payload()
    assert payload["format"] == "full-dipole-history-v7"
    restored = FullDipoleHistory.from_checkpoint_payload(
        json.loads(json.dumps(payload))
    )
    data = source()
    for i in range(35, len(data.time)):
        old = original.segments
        args = (data.time[i], data.position[i], data.velocity[i], data.dipole[i])
        kwargs = dict(dipole_parts=(data.dipole_high[i], data.dipole_low[i]))
        original = original.append(*args, **kwargs)
        restored = restored.append(*args, **kwargs)
        assert all(a is b for a, b in zip(old, original.segments))
    assert restored.to_checkpoint_payload() == original.to_checkpoint_payload()
    for a, b in zip(original.segments, restored.segments):
        np.testing.assert_array_equal(a.position, b.position)
        np.testing.assert_array_equal(a.dipole, b.dipole)


def test_wait_for_full_fit_span_and_anchor_finer_knots():
    h = candidate(25)
    assert h.published_until <= 0
    full = candidate()
    a = full._derivative_indices(34)
    b = full._derivative_indices(35)
    np.testing.assert_array_equal(a, b)
    assert full.time[a[-1]] <= full.time[-1]


def test_nonzero_eighth_derivative():
    h = source()
    t = np.maximum(h.time, 0)
    position, velocity = np.zeros_like(h.position), np.zeros_like(h.velocity)
    position[:, 0] = 1e-5 * t**9
    velocity[:, 0] = 9e-5 * t**8
    h = replace(h, position=position, velocity=velocity).with_single_fit_geometry(
        fit_sample_spacing=2 / 64, dipole_tolerance=1e-20
    )
    for segment in h.segments:
        if segment.start < 0:
            continue
        midpoint = (segment.start + segment.end) / 2
        np.testing.assert_allclose(
            segment.sample(midpoint, 8)[0][0],
            1e-5 * 362880 * midpoint,
            rtol=1e-6,
            atol=1e-10,
        )


def test_nonmatching_recording_times_refused():
    with pytest.raises(ValueError, match="matching accepted recording times"):
        source().with_single_fit_geometry(
            fit_sample_spacing=1.5 / 64, dipole_tolerance=1e-20
        )


@pytest.mark.parametrize("grid,count", [(1024, 896), (2048, 1792)])
def test_long_recording_clock_roundoff(grid, count):
    spacing = 1.25 / (1024 * C_MMNS)
    times = np.r_[0.0, np.cumsum(np.full(count, 1.25 / (grid * C_MMNS)))]
    ratio = grid // 1024
    for last in range(10 * ratio, len(times)):
        h = SimpleNamespace(
            time=times[: last + 1], inertial_until=0.0, fit_sample_spacing=spacing
        )
        knot = last - 6 * ratio
        indices = fixed_spacing_indices(h, knot)
        expected = max(knot // ratio - 5, 0) * ratio + np.arange(11) * ratio
        np.testing.assert_array_equal(indices, expected)
        # A restart with more accepted data must not alter the sample choice.
        h.time = times
        np.testing.assert_array_equal(fixed_spacing_indices(h, knot), expected)


def test_budget_refusal_and_invalid_schema():
    h = source()
    broken = h.position.copy()
    broken[25, 0] += 0.01
    with pytest.raises(SourcePositionError):
        replace(h, position=broken).with_single_fit_geometry(
            fit_sample_spacing=2 / 64, dipole_tolerance=1e-20
        )
    payload = candidate().to_checkpoint_payload()
    bad = deepcopy(payload)
    del bad["fit_sample_spacing"]
    with pytest.raises(ValueError, match="v7"):
        FullDipoleHistory.from_checkpoint_payload(bad)
    bad = deepcopy(payload)
    bad["format"] = "full-dipole-history-v6"
    with pytest.raises(ValueError, match="v7"):
        FullDipoleHistory.from_checkpoint_payload(bad)
    with pytest.raises(ValueError, match="publishing"):
        candidate().with_single_fit_geometry(
            fit_sample_spacing=2 / 64, dipole_tolerance=1e-20
        )


@pytest.mark.parametrize("spacing", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError):
        source().with_single_fit_geometry(
            fit_sample_spacing=spacing, dipole_tolerance=1e-20
        )
