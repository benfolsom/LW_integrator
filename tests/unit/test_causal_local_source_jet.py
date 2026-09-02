"""Tests for direct local source jets on accepted histories."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.causal_local_source_history import (
    CausalLocalDipoleSource,
    CausalLocalDipoleSourceCollection,
    CausalLocalSourceHistory,
    CausalLocalSourceHistoryUnavailableError,
)
from core.causal_local_source_jet import (
    CausalLocalSourceJetModelSpreadError,
    CausalLocalSourceJetScaleSelectionError,
    LocalSourceJetFitConfig,
    LocalSourceJetModelSpreadConfig,
    LocalSourceJetMultiScaleConfig,
    LocalSourceJetScaleConfig,
    _centered_taylor_coefficients,
    _cubic_position_velocity,
    evaluate_configured_causal_local_source_jet_collection_native,
    evaluate_causal_local_source_jet_collection_multiscale_native,
    evaluate_causal_local_source_jet_native,
    evaluate_causal_local_source_jet_multiscale_native,
    local_source_jet_configs_from_source_options,
    local_source_jet_multiscale_config_from_source_options,
)
from core.constants import C_MMNS
from core.dipole_hertz_jet import (
    DipoleHertzResponseJetResult,
    polynomial_dipole_hertz_response_jet_native,
)
from core.retarded_fields import ObserverEvent
from core.types import DipoleSourceConfig


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


def _past_scale(
    name: str,
    widths: tuple[float, float, float],
    *,
    maximum_internal_spread: float = 1.0e-5,
) -> LocalSourceJetScaleConfig:
    fits = tuple(
        LocalSourceJetFitConfig(
            half_width_ns=width,
            window_alignment="past",
        )
        for width in widths
    )
    return LocalSourceJetScaleConfig(
        name=name,
        primary_fit=fits[1],
        model_spread=LocalSourceJetModelSpreadConfig(
            narrow_fit=fits[0],
            wide_fit=fits[2],
            maximum_relative_spread=maximum_internal_spread,
        ),
    )


def _analytic_history(*, acceleration_ready: bool = True) -> CausalLocalSourceHistory:
    count = 81
    nominal = np.linspace(-0.04, 0.04, count)
    times = nominal + 1.0e-5 * np.sin(np.arange(count) * 0.7)
    derivatives = [_position_derivatives(time) for time in times]
    positions = np.asarray([value[0] for value in derivatives])
    beta = np.asarray([value[1] / C_MMNS for value in derivatives])
    beta_prime = np.asarray([value[2] / C_MMNS**2 for value in derivatives])
    spin = np.asarray([_spin_from_chart(_chart_derivatives(time)[0]) for time in times])
    return CausalLocalSourceHistory.from_accepted_samples(
        time_ns=times,
        position_mm=positions,
        beta=beta,
        rest_spin=spin,
        stereographic_frame=np.eye(3),
        interval_start_beta_prime_per_mm=beta_prime[:-1],
        interval_start_acceleration_ready=np.full(
            count - 1,
            acceleration_ready,
            dtype=bool,
        ),
        interval_mean_acceleration_ready=np.ones(count - 1, dtype=bool),
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


def _circular_history(*, sample_count: int = 241) -> CausalLocalSourceHistory:
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
    return CausalLocalSourceHistory.from_accepted_samples(
        time_ns=times,
        position_mm=positions,
        beta=beta,
        rest_spin=spin,
        stereographic_frame=np.eye(3),
        interval_start_beta_prime_per_mm=beta_prime[:-1],
        interval_start_acceleration_ready=np.ones(sample_count - 1, dtype=bool),
        interval_mean_acceleration_ready=np.ones(sample_count - 1, dtype=bool),
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
    history: CausalLocalSourceHistory,
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


def test_past_only_source_jet_tracks_exact_circular_source() -> None:
    history = _circular_history()
    event = ObserverEvent(0.03, (0.8, 0.2, 0.3))
    expected = _exact_circular_response(event)
    actual, diagnostics = evaluate_causal_local_source_jet_native(
        history,
        event,
        magnetic_moment_native=-1.7,
        fit=LocalSourceJetFitConfig(
            half_width_ns=0.005,
            window_alignment="past",
        ),
    )
    error = np.linalg.norm(actual.partial_f - expected.partial_f) / np.linalg.norm(
        expected.partial_f
    )

    assert error < 8.0e-10
    assert diagnostics.acceleration_condition_number < 1.0e4


@pytest.mark.parametrize("acceleration_samples", ("exact_start", "interval_mean"))
def test_past_only_source_jet_is_independent_of_later_accepted_samples(
    acceleration_samples: str,
) -> None:
    history = _circular_history()
    event = ObserverEvent(0.03, (0.8, 0.2, 0.3))
    fit = LocalSourceJetFitConfig(
        half_width_ns=0.005,
        window_alignment="past",
        acceleration_samples=acceleration_samples,
    )
    complete, _ = evaluate_causal_local_source_jet_native(
        history,
        event,
        magnetic_moment_native=-1.7,
        fit=fit,
    )
    stop = int(np.searchsorted(history.time_ns, complete.retarded_time_ns) + 1)
    truncated = CausalLocalSourceHistory.from_accepted_samples(
        time_ns=history.time_ns[:stop],
        position_mm=history.position_mm[:stop],
        beta=history.beta[:stop],
        rest_spin=history.rest_spin[:stop],
        stereographic_frame=history.stereographic_frame,
        interval_start_beta_prime_per_mm=(
            history.interval_start_beta_prime_per_mm[: stop - 1]
        ),
        interval_start_acceleration_ready=(
            history.interval_start_acceleration_ready[: stop - 1]
        ),
        interval_mean_acceleration_ready=(
            history.interval_mean_acceleration_ready[: stop - 1]
        ),
    )
    causal, _ = evaluate_causal_local_source_jet_native(
        truncated,
        event,
        magnetic_moment_native=-1.7,
        fit=fit,
    )

    assert complete.retarded_time_ns == causal.retarded_time_ns
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        np.testing.assert_array_equal(getattr(complete, name), getattr(causal, name))


def test_interval_mean_acceleration_converges_to_exact_circular_response() -> None:
    event = ObserverEvent(0.03, (0.8, 0.2, 0.3))
    expected = _exact_circular_response(event)

    def error(sample_count: int) -> tuple[float, str]:
        actual, diagnostics = evaluate_causal_local_source_jet_native(
            _circular_history(sample_count=sample_count),
            event,
            magnetic_moment_native=-1.7,
            fit=LocalSourceJetFitConfig(
                half_width_ns=0.005,
                acceleration_samples="interval_mean",
                window_alignment="past",
            ),
        )
        relative = np.linalg.norm(
            actual.partial_f - expected.partial_f
        ) / np.linalg.norm(expected.partial_f)
        return float(relative), diagnostics.acceleration_samples

    coarse_error, coarse_semantics = error(121)
    fine_error, fine_semantics = error(481)

    assert coarse_semantics == fine_semantics == "interval_mean"
    assert fine_error < 1.1e-9
    assert fine_error < 0.2 * coarse_error


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
    perturbed_interval_start = np.array(
        history.interval_start_beta_prime_per_mm,
        copy=True,
    )
    perturbed_interval_start[departing_index] *= 1.08
    perturbed_spin = np.array(history.rest_spin, copy=True)
    chart = _circular_chart_derivatives(
        float(history.time_ns[departing_index]),
        0,
    ) + np.asarray((0.02, -0.015))
    perturbed_spin[departing_index] = _spin_from_chart(chart)
    history = CausalLocalSourceHistory.from_accepted_samples(
        time_ns=history.time_ns,
        position_mm=history.position_mm,
        beta=history.beta,
        rest_spin=perturbed_spin,
        stereographic_frame=history.stereographic_frame,
        interval_start_beta_prime_per_mm=perturbed_interval_start,
        interval_start_acceleration_ready=(history.interval_start_acceleration_ready),
        interval_mean_acceleration_ready=(history.interval_mean_acceleration_ready),
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


def test_multiscale_source_jet_uses_longest_scale_for_sparse_history() -> None:
    scales = LocalSourceJetMultiScaleConfig(
        scales=(
            _past_scale("short", (3.0e-4, 5.0e-4, 3.0e-3)),
            _past_scale("long", (3.0e-3, 5.0e-3, 8.0e-3)),
        ),
        maximum_cross_scale_relative_spread=1.0e-5,
    )

    response, diagnostics = evaluate_causal_local_source_jet_multiscale_native(
        _analytic_history(),
        ObserverEvent(0.02, (1.0, 0.25, -0.05)),
        magnetic_moment_native=-1.7,
        scales=scales,
    )

    assert np.all(np.isfinite(response.partial_f))
    assert diagnostics.selected_scale_name == "long"
    assert diagnostics.selected_scale_index == 1
    assert diagnostics.comparison_scale_name is None
    assert diagnostics.cross_scale_spread is None
    assert diagnostics.unavailable_scale_names == ("short",)


def test_multiscale_source_jet_selects_shortest_ready_checked_scale() -> None:
    history = _circular_history(sample_count=481)
    event = ObserverEvent(0.03, (0.8, 0.2, 0.3))
    short = _past_scale("short", (2.0e-3, 3.0e-3, 5.0e-3))
    long = _past_scale("long", (5.0e-3, 7.0e-3, 1.0e-2))
    scales = LocalSourceJetMultiScaleConfig(
        scales=(short, long),
        maximum_cross_scale_relative_spread=1.0e-5,
    )
    expected, _ = evaluate_causal_local_source_jet_native(
        history,
        event,
        magnetic_moment_native=-1.7,
        fit=short.primary_fit,
        model_spread=short.model_spread,
    )

    response, diagnostics = evaluate_causal_local_source_jet_multiscale_native(
        history,
        event,
        magnetic_moment_native=-1.7,
        scales=scales,
    )

    np.testing.assert_array_equal(response.partial_f, expected.partial_f)
    assert diagnostics.selected_scale_name == "short"
    assert diagnostics.selected_scale_index == 0
    assert diagnostics.comparison_scale_name == "long"
    assert diagnostics.cross_scale_spread is not None
    assert diagnostics.cross_scale_spread.maximum < 1.0e-5


def test_multiscale_source_jet_fails_closed_when_ready_scales_disagree() -> None:
    scales = LocalSourceJetMultiScaleConfig(
        scales=(
            _past_scale("short", (2.0e-3, 3.0e-3, 5.0e-3)),
            _past_scale("long", (5.0e-3, 7.0e-3, 1.0e-2)),
        ),
        maximum_cross_scale_relative_spread=1.0e-20,
    )

    with pytest.raises(
        CausalLocalSourceJetScaleSelectionError,
        match="scales do not agree",
    ):
        evaluate_causal_local_source_jet_multiscale_native(
            _circular_history(sample_count=481),
            ObserverEvent(0.03, (0.8, 0.2, 0.3)),
            magnetic_moment_native=-1.7,
            scales=scales,
        )


def test_multiscale_source_jet_requires_a_longer_transition_comparison() -> None:
    scales = LocalSourceJetMultiScaleConfig(
        scales=(
            _past_scale("short", (2.0e-3, 3.0e-3, 5.0e-3)),
            _past_scale(
                "middle",
                (5.0e-3, 7.0e-3, 1.0e-2),
                maximum_internal_spread=1.0e-20,
            ),
            _past_scale(
                "long",
                (1.0e-2, 1.5e-2, 2.0e-2),
                maximum_internal_spread=1.0e-20,
            ),
        )
    )

    with pytest.raises(
        CausalLocalSourceJetScaleSelectionError,
        match="no valid adjacent longer overlap comparison",
    ):
        evaluate_causal_local_source_jet_multiscale_native(
            _circular_history(sample_count=481),
            ObserverEvent(0.03, (0.8, 0.2, 0.3)),
            magnetic_moment_native=-1.7,
            scales=scales,
        )


def test_multiscale_source_jet_does_not_skip_a_failed_adjacent_scale() -> None:
    scales = LocalSourceJetMultiScaleConfig(
        scales=(
            _past_scale("short", (2.0e-3, 3.0e-3, 5.0e-3)),
            _past_scale(
                "middle",
                (5.0e-3, 7.0e-3, 1.0e-2),
                maximum_internal_spread=1.0e-20,
            ),
            _past_scale("long", (1.0e-2, 1.5e-2, 2.0e-2)),
        )
    )

    with pytest.raises(
        CausalLocalSourceJetScaleSelectionError,
        match="no valid adjacent longer overlap comparison",
    ):
        evaluate_causal_local_source_jet_multiscale_native(
            _circular_history(sample_count=481),
            ObserverEvent(0.03, (0.8, 0.2, 0.3)),
            magnetic_moment_native=-1.7,
            scales=scales,
        )


def test_multiscale_collection_preserves_source_order_and_sums_responses() -> None:
    history = _circular_history(sample_count=481)
    collection = CausalLocalDipoleSourceCollection(
        (
            CausalLocalDipoleSource("first", 0, -1.7, history),
            CausalLocalDipoleSource("second", 1, 0.4, history),
        )
    )
    event = ObserverEvent(0.03, (0.8, 0.2, 0.3))
    scales = LocalSourceJetMultiScaleConfig(
        scales=(
            _past_scale("short", (2.0e-3, 3.0e-3, 5.0e-3)),
            _past_scale("long", (5.0e-3, 7.0e-3, 1.0e-2)),
        ),
        maximum_cross_scale_relative_spread=1.0e-5,
    )

    actual = evaluate_causal_local_source_jet_collection_multiscale_native(
        collection,
        event,
        scales=scales,
    )
    expected = [
        evaluate_causal_local_source_jet_multiscale_native(
            history,
            event,
            magnetic_moment_native=moment,
            scales=scales,
        )[0]
        for moment in (-1.7, 0.4)
    ]

    assert tuple(item.identity for item in actual.source_results) == (
        "first",
        "second",
    )
    np.testing.assert_allclose(
        actual.four_potential,
        expected[0].four_potential + expected[1].four_potential,
        rtol=0.0,
        atol=1.0e-30,
    )
    np.testing.assert_allclose(
        actual.partial_f,
        expected[0].partial_f + expected[1].partial_f,
        rtol=0.0,
        atol=1.0e-30,
    )


def test_multiscale_config_requires_ordered_overlapping_physical_scales() -> None:
    short = _past_scale("short", (2.0e-3, 3.0e-3, 4.0e-3))
    nonoverlapping = _past_scale("long", (5.0e-3, 7.0e-3, 1.0e-2))

    with pytest.raises(ValueError, match="must overlap"):
        LocalSourceJetMultiScaleConfig(scales=(short, nonoverlapping))

    with pytest.raises(ValueError, match="shortest to longest"):
        LocalSourceJetMultiScaleConfig(scales=(nonoverlapping, short))


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


def test_nested_fits_require_matching_acceleration_samples() -> None:
    with pytest.raises(ValueError, match="same acceleration samples"):
        evaluate_causal_local_source_jet_native(
            _analytic_history(),
            ObserverEvent(0.02, (1.0, 0.25, -0.05)),
            magnetic_moment_native=-1.7,
            fit=LocalSourceJetFitConfig(half_width_ns=0.015),
            model_spread=LocalSourceJetModelSpreadConfig(
                narrow_fit=LocalSourceJetFitConfig(half_width_ns=0.01),
                wide_fit=LocalSourceJetFitConfig(
                    half_width_ns=0.02,
                    acceleration_samples="interval_mean",
                ),
            ),
        )


def test_local_source_jet_requires_complete_exact_acceleration_window() -> None:
    history = _analytic_history(acceleration_ready=False)
    event = ObserverEvent(0.02, (1.0, 0.25, -0.05))

    with pytest.raises(
        CausalLocalSourceHistoryUnavailableError,
        match="unavailable trusted exact-start sample inside",
    ):
        evaluate_causal_local_source_jet_native(
            history,
            event,
            magnetic_moment_native=-1.7,
            fit=LocalSourceJetFitConfig(half_width_ns=0.015),
        )


def test_local_source_jet_rejects_missing_exact_acceleration_inside_window() -> None:
    history = _analytic_history()
    event = ObserverEvent(0.02, (1.0, 0.25, -0.05))
    fit = LocalSourceJetFitConfig(half_width_ns=0.015)
    reference, _ = evaluate_causal_local_source_jet_native(
        history,
        event,
        magnetic_moment_native=-1.7,
        fit=fit,
    )
    missing = np.array(history.interval_start_acceleration_ready, copy=True)
    missing_index = int(
        np.argmin(
            np.abs(
                history.time_ns[: history.interval_count] - reference.retarded_time_ns
            )
        )
    )
    missing[missing_index] = False
    incomplete = CausalLocalSourceHistory.from_accepted_samples(
        time_ns=history.time_ns,
        position_mm=history.position_mm,
        beta=history.beta,
        rest_spin=history.rest_spin,
        stereographic_frame=history.stereographic_frame,
        interval_start_beta_prime_per_mm=(history.interval_start_beta_prime_per_mm),
        interval_start_acceleration_ready=missing,
        interval_mean_acceleration_ready=(history.interval_mean_acceleration_ready),
    )

    with pytest.raises(
        CausalLocalSourceHistoryUnavailableError,
        match="unavailable trusted exact-start sample inside",
    ):
        evaluate_causal_local_source_jet_native(
            incomplete,
            event,
            magnetic_moment_native=-1.7,
            fit=fit,
        )


def test_local_source_jet_rejects_untrusted_interval_mean_inside_window() -> None:
    history = _analytic_history()
    event = ObserverEvent(0.02, (1.0, 0.25, -0.05))
    fit = LocalSourceJetFitConfig(
        half_width_ns=0.015,
        acceleration_samples="interval_mean",
    )
    reference, _ = evaluate_causal_local_source_jet_native(
        history,
        event,
        magnetic_moment_native=-1.7,
        fit=fit,
    )
    sample_times = 0.5 * (history.time_ns[:-1] + history.time_ns[1:])
    ready = np.array(history.interval_mean_acceleration_ready, copy=True)
    ready[int(np.argmin(np.abs(sample_times - reference.retarded_time_ns)))] = False
    incomplete = CausalLocalSourceHistory.from_accepted_samples(
        time_ns=history.time_ns,
        position_mm=history.position_mm,
        beta=history.beta,
        rest_spin=history.rest_spin,
        stereographic_frame=history.stereographic_frame,
        interval_start_beta_prime_per_mm=(history.interval_start_beta_prime_per_mm),
        interval_start_acceleration_ready=(history.interval_start_acceleration_ready),
        interval_mean_acceleration_ready=ready,
    )

    with pytest.raises(
        CausalLocalSourceHistoryUnavailableError,
        match="unavailable trusted interval-mean sample inside",
    ):
        evaluate_causal_local_source_jet_native(
            incomplete,
            event,
            magnetic_moment_native=-1.7,
            fit=fit,
        )


def test_local_source_jet_rejects_truncated_physical_window() -> None:
    history = _analytic_history()
    event = ObserverEvent(0.0036, (1.0, 0.25, -0.05))

    with pytest.raises(
        CausalLocalSourceHistoryUnavailableError,
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
        (
            {"half_width_ns": 1.0, "window_alignment": "invalid"},
            "window_alignment",
        ),
        (
            {"half_width_ns": 1.0, "acceleration_samples": "invalid"},
            "acceleration_samples",
        ),
    ),
)
def test_local_source_jet_fit_config_rejects_invalid_values(
    arguments: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LocalSourceJetFitConfig(**arguments)  # type: ignore[arg-type]


def test_public_local_source_options_build_primary_and_spread_fits() -> None:
    source = DipoleSourceConfig(
        history_model="causal_local_jet",
        local_jet_narrow_half_width_ns=1.0e-8,
        local_jet_primary_half_width_ns=1.2e-8,
        local_jet_wide_half_width_ns=1.5e-8,
        local_jet_acceleration_degree=6,
        local_jet_spin_degree=7,
    )

    primary, spread = local_source_jet_configs_from_source_options(source)

    assert primary.half_width_ns == 1.2e-8
    assert primary.acceleration_degree == 6
    assert primary.spin_degree == 7
    assert primary.acceleration_samples == "interval_mean"
    assert primary.window_alignment == "past"
    assert spread.narrow_fit.half_width_ns == 1.0e-8
    assert spread.wide_fit.half_width_ns == 1.5e-8
    assert spread.maximum_relative_spread == 1.0e-3


def test_public_multiscale_options_build_and_dispatch_named_scale_ladder() -> None:
    history = _circular_history(sample_count=481)
    collection = CausalLocalDipoleSourceCollection(
        (CausalLocalDipoleSource("source", 0, -1.7, history),)
    )
    event = ObserverEvent(0.03, (0.8, 0.2, 0.3))
    source = DipoleSourceConfig(
        history_model="causal_local_jet",
        local_jet_scales=(
            {
                "name": "short",
                "narrow_half_width_ns": 2.0e-3,
                "primary_half_width_ns": 3.0e-3,
                "wide_half_width_ns": 5.0e-3,
            },
            {
                "name": "long",
                "narrow_half_width_ns": 5.0e-3,
                "primary_half_width_ns": 7.0e-3,
                "wide_half_width_ns": 1.0e-2,
            },
        ),
        local_jet_maximum_relative_spread=1.0e-5,
        local_jet_maximum_cross_scale_relative_spread=1.0e-5,
    )

    scales = local_source_jet_multiscale_config_from_source_options(source)
    actual = evaluate_configured_causal_local_source_jet_collection_native(
        collection,
        event,
        source_options=source,
    )
    expected = evaluate_causal_local_source_jet_collection_multiscale_native(
        collection,
        event,
        scales=scales,
    )

    assert tuple(scale.name for scale in scales.scales) == ("short", "long")
    np.testing.assert_array_equal(actual.partial_f, expected.partial_f)
    assert actual.source_results[0].diagnostics.selected_scale_name == "short"
