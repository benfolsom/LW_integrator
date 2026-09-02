"""Tests for direct local source jets on accepted histories."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.causal_c5_source_history import (
    CausalC5HistoryUnavailableError,
    CausalC5SourceHistory,
)
from core.causal_local_source_jet import (
    CausalLocalSourceJetModelSpreadError,
    LocalSourceJetFitConfig,
    LocalSourceJetModelSpreadConfig,
    _centered_taylor_coefficients,
    _cubic_position_velocity,
    evaluate_causal_local_source_jet_native,
)
from core.constants import C_MMNS
from core.dipole_hertz_jet import (
    DipoleHertzResponseJetResult,
    polynomial_dipole_hertz_response_jet_native,
)
from core.retarded_fields import ObserverEvent


def _position_derivatives(time_ns: float) -> list[np.ndarray]:
    t = float(time_ns)
    return [
        np.asarray(
            (
                0.03 * t + 0.002 * t**2 + 0.0001 * t**3,
                0.2 - 0.01 * t + 0.0003 * t**2,
                -0.1 + 0.005 * t,
            )
        ),
        np.asarray(
            (
                0.03 + 0.004 * t + 0.0003 * t**2,
                -0.01 + 0.0006 * t,
                0.005,
            )
        ),
        np.asarray((0.004 + 0.0006 * t, 0.0006, 0.0)),
        np.asarray((0.0006, 0.0, 0.0)),
        np.zeros(3),
        np.zeros(3),
    ]


def _chart_derivatives(time_ns: float) -> list[np.ndarray]:
    return [
        np.asarray((0.1 + 0.02 * time_ns, -0.08 + 0.01 * time_ns)),
        np.asarray((0.02, 0.01)),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
    ]


def _spin_from_chart(chart: np.ndarray) -> np.ndarray:
    radius_squared = float(chart @ chart)
    return np.asarray((2.0 * chart[0], 2.0 * chart[1], 1.0 - radius_squared)) / (
        1.0 + radius_squared
    )


def _relative_response_component(
    left: DipoleHertzResponseJetResult,
    right: DipoleHertzResponseJetResult,
    name: str,
) -> float:
    first = np.asarray(getattr(left, name))
    second = np.asarray(getattr(right, name))
    scale = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(first - second) / scale)


def _analytic_history(*, acceleration_ready: bool = True) -> CausalC5SourceHistory:
    count = 81
    nominal = np.linspace(-0.04, 0.04, count)
    times = nominal + 1.0e-5 * np.sin(np.arange(count) * 0.7)
    derivatives = [_position_derivatives(time) for time in times]
    positions = np.asarray([value[0] for value in derivatives])
    beta = np.asarray([value[1] / C_MMNS for value in derivatives])
    beta_prime = np.asarray([value[2] / C_MMNS**2 for value in derivatives])
    spin = np.asarray([_spin_from_chart(_chart_derivatives(time)[0]) for time in times])
    step_start = np.zeros_like(beta_prime)
    step_start[1:] = beta_prime[:-1]
    ready = np.full(count, acceleration_ready, dtype=bool)
    ready[0] = False
    return CausalC5SourceHistory.from_accepted_samples(
        time_ns=times,
        position_mm=positions,
        beta=beta,
        beta_prime_per_mm=beta_prime,
        rest_spin=spin,
        stereographic_frame=np.eye(3),
        step_start_beta_prime_per_mm=step_start,
        step_start_beta_prime_ready=ready,
    )


_CIRCULAR_RADIUS_MM = 0.025
_CIRCULAR_ANGULAR_RATE_PER_NS = 24.0
_SPIN_CHART_ANGULAR_RATE_PER_NS = 17.0


def _circular_position_derivatives(
    time_ns: float,
    derivative_order: int,
) -> np.ndarray:
    phase = (
        _CIRCULAR_ANGULAR_RATE_PER_NS * time_ns
        + 0.23
        + derivative_order * math.pi / 2.0
    )
    scale = _CIRCULAR_RADIUS_MM * _CIRCULAR_ANGULAR_RATE_PER_NS**derivative_order
    result = np.asarray((scale * math.cos(phase), scale * math.sin(phase), 0.0))
    if derivative_order == 0:
        result += np.asarray((0.1, -0.03, 0.01 + 0.004 * time_ns))
    elif derivative_order == 1:
        result[2] = 0.004
    return result


def _circular_chart_derivatives(
    time_ns: float,
    derivative_order: int,
) -> np.ndarray:
    rate = _SPIN_CHART_ANGULAR_RATE_PER_NS
    offset = derivative_order * math.pi / 2.0
    return np.asarray(
        (
            0.12 * rate**derivative_order * math.cos(rate * time_ns + 0.1 + offset),
            0.08 * rate**derivative_order * math.sin(rate * time_ns - 0.2 + offset),
        )
    )


def _circular_history(*, sample_count: int = 241) -> CausalC5SourceHistory:
    nominal_times = np.linspace(-0.05, 0.08, sample_count)
    nominal_step = float(nominal_times[1] - nominal_times[0])
    times = nominal_times + 0.13 * nominal_step * np.sin(0.73 * np.arange(sample_count))
    derivatives = [
        [_circular_position_derivatives(time, order) for order in range(3)]
        for time in times
    ]
    positions = np.asarray([value[0] for value in derivatives])
    beta = np.asarray([value[1] / C_MMNS for value in derivatives])
    beta_prime = np.asarray([value[2] / C_MMNS**2 for value in derivatives])
    spin = np.asarray(
        [_spin_from_chart(_circular_chart_derivatives(time, 0)) for time in times]
    )
    step_start = np.zeros_like(beta_prime)
    step_start[1:] = beta_prime[:-1]
    ready = np.ones(sample_count, dtype=bool)
    ready[0] = False
    return CausalC5SourceHistory.from_accepted_samples(
        time_ns=times,
        position_mm=positions,
        beta=beta,
        beta_prime_per_mm=beta_prime,
        rest_spin=spin,
        stereographic_frame=np.eye(3),
        step_start_beta_prime_per_mm=step_start,
        step_start_beta_prime_ready=ready,
    )


def _exact_circular_response(event: ObserverEvent) -> DipoleHertzResponseJetResult:
    observer_position = np.asarray(event.position_mm)

    def light_cone_residual(source_time_ns: float) -> float:
        separation = np.linalg.norm(
            observer_position - _circular_position_derivatives(source_time_ns, 0)
        )
        return C_MMNS * (event.time_ns - source_time_ns) - separation

    lower = -0.05
    upper = 0.08
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if light_cone_residual(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    root = 0.5 * (lower + upper)
    adapter_duration = 0.02
    return polynomial_dipole_hertz_response_jet_native(
        observer_time_ns=event.time_ns,
        observer_position_mm=event.position_mm,
        magnetic_moment_native=-1.7,
        segment_start_time_ns=root - 0.5 * adapter_duration,
        segment_duration_ns=adapter_duration,
        position_coefficients_mm=_centered_taylor_coefficients(
            [_circular_position_derivatives(root, order) for order in range(6)],
            duration_ns=adapter_duration,
        ),
        rest_spin_coefficients=None,
        rest_spin_stereographic_coefficients=_centered_taylor_coefficients(
            [_circular_chart_derivatives(root, order) for order in range(6)],
            duration_ns=adapter_duration,
        ),
        rest_spin_stereographic_frame=np.eye(3),
        preserved_rest_spin_magnitude=None,
        retarded_time_ns=root,
    )


def _event_with_prescribed_cubic_root(
    history: CausalC5SourceHistory,
    root_time_ns: float,
) -> ObserverEvent:
    segment = int(np.searchsorted(history.time_ns, root_time_ns) - 1)
    segment = max(0, min(history.sample_count - 2, segment))
    source_position, _ = _cubic_position_velocity(history, segment, root_time_ns)
    observer_position = np.asarray((0.8, 0.2, 0.3))
    observer_time = (
        root_time_ns + np.linalg.norm(observer_position - source_position) / C_MMNS
    )
    return ObserverEvent(float(observer_time), tuple(observer_position))


def test_local_source_jet_matches_exact_cubic_worldline_and_linear_spin_chart() -> None:
    history = _analytic_history()
    event = ObserverEvent(0.02, (1.0, 0.25, -0.05))
    fit = LocalSourceJetFitConfig(
        half_width_ns=0.015,
        acceleration_degree=3,
        spin_degree=5,
    )

    actual, diagnostics = evaluate_causal_local_source_jet_native(
        history,
        event,
        magnetic_moment_native=-1.7,
        fit=fit,
    )
    root = actual.retarded_time_ns
    duration = 2.0 * fit.half_width_ns
    expected = polynomial_dipole_hertz_response_jet_native(
        observer_time_ns=event.time_ns,
        observer_position_mm=event.position_mm,
        magnetic_moment_native=-1.7,
        segment_start_time_ns=root - 0.5 * duration,
        segment_duration_ns=duration,
        position_coefficients_mm=_centered_taylor_coefficients(
            _position_derivatives(root), duration_ns=duration
        ),
        rest_spin_coefficients=None,
        rest_spin_stereographic_coefficients=_centered_taylor_coefficients(
            _chart_derivatives(root), duration_ns=duration
        ),
        rest_spin_stereographic_frame=np.eye(3),
        preserved_rest_spin_magnitude=None,
        retarded_time_ns=root,
    )

    assert abs(diagnostics.light_cone_residual_mm) <= 1.0e-15
    assert diagnostics.acceleration_sample_indices.size >= 4
    assert diagnostics.spin_sample_indices.size >= 6
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=2.0e-9,
            atol=2.0e-18,
        )


def test_local_source_jet_tracks_exact_circular_source_as_window_shrinks() -> None:
    history = _circular_history()
    event = ObserverEvent(0.03, (0.8, 0.2, 0.3))
    expected = _exact_circular_response(event)
    wide, _ = evaluate_causal_local_source_jet_native(
        history,
        event,
        magnetic_moment_native=-1.7,
        fit=LocalSourceJetFitConfig(half_width_ns=0.02),
    )
    narrow, _ = evaluate_causal_local_source_jet_native(
        history,
        event,
        magnetic_moment_native=-1.7,
        fit=LocalSourceJetFitConfig(half_width_ns=0.005),
    )
    wide_error = np.linalg.norm(wide.partial_f - expected.partial_f) / np.linalg.norm(
        expected.partial_f
    )
    narrow_error = np.linalg.norm(
        narrow.partial_f - expected.partial_f
    ) / np.linalg.norm(expected.partial_f)

    assert wide_error < 3.0e-9
    assert narrow_error < 3.0e-11
    assert narrow_error < 0.02 * wide_error


def test_local_source_jet_is_continuous_across_cubic_root_segment_boundary() -> None:
    history = _circular_history(sample_count=161)
    boundary_time = float(history.time_ns[80])

    def response_at(offset_ns: float) -> DipoleHertzResponseJetResult:
        response, _ = evaluate_causal_local_source_jet_native(
            history,
            _event_with_prescribed_cubic_root(history, boundary_time + offset_ns),
            magnetic_moment_native=-1.7,
            fit=LocalSourceJetFitConfig(half_width_ns=0.01),
        )
        return response

    large_left = response_at(-1.0e-5)
    large_right = response_at(1.0e-5)
    small_left = response_at(-1.0e-8)
    small_right = response_at(1.0e-8)
    large_change = _relative_response_component(
        large_left,
        large_right,
        "partial_f",
    )
    small_change = _relative_response_component(
        small_left,
        small_right,
        "partial_f",
    )

    assert small_change < 0.002 * large_change


def test_tricube_fit_is_continuous_when_a_sample_leaves_the_window() -> None:
    history = _circular_history(sample_count=161)
    departing_index = 75
    perturbed_step_start = np.array(
        history.step_start_beta_prime_per_mm,
        copy=True,
    )
    perturbed_step_start[departing_index + 1] *= 1.08
    perturbed_spin = np.array(history.rest_spin, copy=True)
    chart = _circular_chart_derivatives(
        float(history.time_ns[departing_index]),
        0,
    ) + np.asarray((0.02, -0.015))
    perturbed_spin[departing_index] = _spin_from_chart(chart)
    history = CausalC5SourceHistory.from_accepted_samples(
        time_ns=history.time_ns,
        position_mm=history.position_mm,
        beta=history.beta,
        beta_prime_per_mm=history.beta_prime_per_mm,
        rest_spin=perturbed_spin,
        stereographic_frame=history.stereographic_frame,
        step_start_beta_prime_per_mm=perturbed_step_start,
        step_start_beta_prime_ready=history.step_start_beta_prime_ready,
    )
    half_width = 0.01
    sample_boundary_time = float(history.time_ns[departing_index]) + half_width

    def response_at(
        offset_ns: float,
    ) -> tuple[DipoleHertzResponseJetResult, np.ndarray]:
        response, diagnostics = evaluate_causal_local_source_jet_native(
            history,
            _event_with_prescribed_cubic_root(
                history,
                sample_boundary_time + offset_ns,
            ),
            magnetic_moment_native=-1.7,
            fit=LocalSourceJetFitConfig(half_width_ns=half_width),
        )
        return response, diagnostics.acceleration_sample_indices

    large_left, large_left_indices = response_at(-1.0e-5)
    large_right, large_right_indices = response_at(1.0e-5)
    small_left, small_left_indices = response_at(-1.0e-8)
    small_right, small_right_indices = response_at(1.0e-8)
    assert departing_index in large_left_indices
    assert departing_index not in large_right_indices
    assert departing_index in small_left_indices
    assert departing_index not in small_right_indices
    large_change = _relative_response_component(
        large_left,
        large_right,
        "partial_f",
    )
    small_change = _relative_response_component(
        small_left,
        small_right,
        "partial_f",
    )

    assert small_change < 0.002 * large_change


def test_nested_fits_report_a_stable_circular_response_plateau() -> None:
    response, diagnostics = evaluate_causal_local_source_jet_native(
        _circular_history(),
        ObserverEvent(0.03, (0.8, 0.2, 0.3)),
        magnetic_moment_native=-1.7,
        fit=LocalSourceJetFitConfig(half_width_ns=0.01),
        model_spread=LocalSourceJetModelSpreadConfig(
            narrow_fit=LocalSourceJetFitConfig(half_width_ns=0.005),
            wide_fit=LocalSourceJetFitConfig(half_width_ns=0.02),
            maximum_relative_spread=1.0e-8,
        ),
    )

    assert np.all(np.isfinite(response.partial_f))
    assert diagnostics.model_spread is not None
    assert diagnostics.model_spread.maximum < 4.0e-9


def test_nested_fits_fail_closed_without_a_declared_response_plateau() -> None:
    with pytest.raises(
        CausalLocalSourceJetModelSpreadError,
        match="nested fits exceed their response-spread limit",
    ):
        evaluate_causal_local_source_jet_native(
            _circular_history(),
            ObserverEvent(0.03, (0.8, 0.2, 0.3)),
            magnetic_moment_native=-1.7,
            fit=LocalSourceJetFitConfig(half_width_ns=0.01),
            model_spread=LocalSourceJetModelSpreadConfig(
                narrow_fit=LocalSourceJetFitConfig(half_width_ns=0.005),
                wide_fit=LocalSourceJetFitConfig(half_width_ns=0.02),
                maximum_relative_spread=1.0e-12,
            ),
        )


def test_nested_fits_require_narrow_primary_wide_ordering() -> None:
    with pytest.raises(ValueError, match="narrow < primary < wide"):
        evaluate_causal_local_source_jet_native(
            _analytic_history(),
            ObserverEvent(0.02, (1.0, 0.25, -0.05)),
            magnetic_moment_native=-1.7,
            fit=LocalSourceJetFitConfig(half_width_ns=0.015),
            model_spread=LocalSourceJetModelSpreadConfig(
                narrow_fit=LocalSourceJetFitConfig(half_width_ns=0.02),
                wide_fit=LocalSourceJetFitConfig(half_width_ns=0.03),
            ),
        )


def test_local_source_jet_requires_complete_exact_acceleration_window() -> None:
    history = _analytic_history(acceleration_ready=False)
    event = ObserverEvent(0.02, (1.0, 0.25, -0.05))

    with pytest.raises(
        CausalC5HistoryUnavailableError,
        match="acceleration fit has no accepted exact samples",
    ):
        evaluate_causal_local_source_jet_native(
            history,
            event,
            magnetic_moment_native=-1.7,
            fit=LocalSourceJetFitConfig(half_width_ns=0.015),
        )


def test_local_source_jet_rejects_truncated_physical_window() -> None:
    history = _analytic_history()
    event = ObserverEvent(0.0036, (1.0, 0.25, -0.05))

    with pytest.raises(
        CausalC5HistoryUnavailableError,
        match="full physical window",
    ):
        evaluate_causal_local_source_jet_native(
            history,
            event,
            magnetic_moment_native=-1.7,
            fit=LocalSourceJetFitConfig(half_width_ns=0.04),
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        ({"half_width_ns": 0.0}, "half_width_ns"),
        ({"half_width_ns": 1.0, "acceleration_degree": 2}, "acceleration_degree"),
        ({"half_width_ns": 1.0, "spin_degree": 4}, "spin_degree"),
        (
            {"half_width_ns": 1.0, "maximum_condition_number": math.nan},
            "maximum_condition_number",
        ),
        (
            {"half_width_ns": 1.0, "window_weighting": "invalid"},
            "window_weighting",
        ),
    ),
)
def test_local_source_jet_fit_config_rejects_invalid_values(
    arguments: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LocalSourceJetFitConfig(**arguments)  # type: ignore[arg-type]
