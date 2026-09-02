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
    LocalSourceJetFitConfig,
    _centered_taylor_coefficients,
    evaluate_causal_local_source_jet_native,
)
from core.constants import C_MMNS
from core.dipole_hertz_jet import polynomial_dipole_hertz_response_jet_native
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
    ),
)
def test_local_source_jet_fit_config_rejects_invalid_values(
    arguments: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LocalSourceJetFitConfig(**arguments)  # type: ignore[arg-type]
