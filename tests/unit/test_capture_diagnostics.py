"""Focused tests for first-pass two-body capture diagnostics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.capture_diagnostics import (
    ParticleCaptureTrace,
    analyze_first_pass_capture,
    audit_canonical_mechanical_momentum,
    audit_medina_capture_trace,
    reconstruct_mechanical_four_momentum_series_native,
    relativistic_invariant_com_kinetic_energy_native,
    stored_mechanical_four_momentum_series_native,
)
from core.constants import C_MMNS


def _trace(
    *,
    time_ns: np.ndarray,
    position_mm: np.ndarray,
    velocity_mm_per_ns: np.ndarray,
    mass_amu: float,
    observer_charge_native: float,
    source_charge_native: float,
    potential: np.ndarray | None = None,
    radiation_work: np.ndarray | None = None,
    far_energy: np.ndarray | None = None,
    cross_energy: np.ndarray | None = None,
    cross_change: np.ndarray | None = None,
    force_sample_time: np.ndarray | None = None,
    derivative_ready: np.ndarray | None = None,
    capped: np.ndarray | None = None,
    dead: np.ndarray | None = None,
) -> ParticleCaptureTrace:
    count = time_ns.size
    beta = np.asarray(velocity_mm_per_ns, dtype=float) / C_MMNS
    gamma = 1.0 / np.sqrt(1.0 - np.sum(beta * beta, axis=1))
    mechanical = stored_mechanical_four_momentum_series_native(
        gamma, beta, mass_amu=mass_amu
    )
    if potential is None:
        canonical = mechanical.copy()
    else:
        canonical = mechanical + observer_charge_native * potential / C_MMNS

    def floating(value: np.ndarray | None, default: float = 0.0) -> np.ndarray:
        return (
            np.full(count, default, dtype=float)
            if value is None
            else np.asarray(value, dtype=float)
        )

    def boolean(value: np.ndarray | None) -> np.ndarray:
        return (
            np.zeros(count, dtype=bool)
            if value is None
            else np.asarray(value, dtype=bool)
        )

    return ParticleCaptureTrace(
        time_ns=np.asarray(time_ns, dtype=float),
        position_mm=np.asarray(position_mm, dtype=float),
        canonical_four_momentum_native=canonical,
        gamma=gamma,
        beta=beta,
        mass_amu=mass_amu,
        observer_charge_native=observer_charge_native,
        source_charge_native=source_charge_native,
        macro_population=1.0,
        radiation_reaction_work_native=floating(radiation_work),
        far_radiated_energy_native=floating(far_energy),
        medina_cross_field_energy_native=floating(cross_energy),
        medina_cross_field_energy_change_native=floating(cross_change),
        medina_force_derivative_ready=boolean(derivative_ready),
        medina_impulse_capped=boolean(capped),
        medina_external_force_sample_time_ns=floating(
            force_sample_time, default=np.nan
        ),
        dead=boolean(dead),
        ordinary_four_potential_native=potential,
    )


def test_exact_ordinary_potential_reconstructs_mechanical_four_momentum() -> None:
    time = np.array([0.0, 0.5, 1.0])
    velocity = np.array(
        [
            [0.2, -0.1, 0.0],
            [0.1, 0.3, -0.2],
            [-0.2, 0.0, 0.4],
        ]
    )
    potential = np.array(
        [
            [3.0, -2.0, 1.0, 0.5],
            [2.5, -1.0, 0.5, 0.25],
            [2.0, 0.0, -0.5, 0.0],
        ]
    )
    trace = _trace(
        time_ns=time,
        position_mm=np.column_stack((time + 1.0, time * 0.0, time * 0.0)),
        velocity_mm_per_ns=velocity,
        mass_amu=0.75,
        observer_charge_native=-0.4,
        source_charge_native=-0.4,
        potential=potential,
    )

    reconstructed = reconstruct_mechanical_four_momentum_series_native(
        trace.canonical_four_momentum_native,
        potential,
        observer_charge_native=trace.observer_charge_native,
    )
    expected = stored_mechanical_four_momentum_series_native(
        trace.gamma, trace.beta, mass_amu=trace.mass_amu
    )
    np.testing.assert_allclose(reconstructed, expected, rtol=0.0, atol=1.0e-16)

    audit = audit_canonical_mechanical_momentum(trace)
    assert audit.checked
    assert audit.max_relative_residual < 1.0e-16
    assert audit.max_relative_mass_shell_residual < 1.0e-15


def test_relativistic_com_kinetic_energy_is_stable_for_small_motion() -> None:
    masses = (5.485799e-4, 1.007276466812)
    momentum = 2.0e-8
    first = np.array(
        [[np.sqrt((masses[0] * C_MMNS) ** 2 + momentum**2), momentum, 0.0, 0.0]]
    )
    second = np.array(
        [[np.sqrt((masses[1] * C_MMNS) ** 2 + momentum**2), -momentum, 0.0, 0.0]]
    )

    kinetic = relativistic_invariant_com_kinetic_energy_native(
        first,
        second,
        first_mass_amu=masses[0],
        second_mass_amu=masses[1],
    )
    expected = sum(
        C_MMNS
        * momentum**2
        / (np.sqrt((mass * C_MMNS) ** 2 + momentum**2) + mass * C_MMNS)
        for mass in masses
    )
    assert kinetic[0] == pytest.approx(expected, rel=2.0e-15)
    assert kinetic[0] > 0.0


def test_medina_audit_keeps_priming_radiation_out_of_ready_balance() -> None:
    trace = _trace(
        time_ns=np.arange(5.0),
        position_mm=np.column_stack((np.arange(5.0) + 1.0, np.zeros((5, 2)))),
        velocity_mm_per_ns=np.zeros((5, 3)),
        mass_amu=1.0,
        observer_charge_native=1.0,
        source_charge_native=1.0,
        radiation_work=np.array([0.0, 0.0, -0.2, -0.3, -0.4]),
        far_energy=np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        cross_change=np.zeros(5),
        force_sample_time=np.array([np.nan, 1.5, 2.5, 3.5, 4.5]),
        derivative_ready=np.array([False, False, True, True, True]),
    )

    audit = audit_medina_capture_trace(trace)
    assert audit.force_sample_count == 4
    assert audit.derivative_ready_count == 3
    assert audit.unexpected_unready_count == 0
    assert audit.far_radiated_energy_native == pytest.approx(1.0)
    assert audit.signed_reaction_work_native == pytest.approx(-0.9)
    assert audit.balance_residual_native == pytest.approx(0.0, abs=1.0e-16)


def _capturing_pair() -> tuple[ParticleCaptureTrace, ParticleCaptureTrace]:
    first_time = np.arange(5.0)
    first_position = np.array(
        [
            [5.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    first_velocity = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    first = _trace(
        time_ns=first_time,
        position_mm=first_position,
        velocity_mm_per_ns=first_velocity,
        mass_amu=1.0,
        observer_charge_native=-1.0,
        source_charge_native=-1.0,
        radiation_work=np.array([0.0, 0.0, -0.02, -0.02, -0.02]),
        far_energy=np.array([0.0, 0.01, 0.02, 0.02, 0.02]),
        force_sample_time=np.array([np.nan, 1.5, 2.5, 3.5, 4.5]),
        derivative_ready=np.array([False, False, True, True, True]),
    )
    # A different integration-time grid exercises explicit lab-time
    # synchronization.  The second particle is stationary at the origin.
    second_time = np.array([0.0, 0.8, 1.8, 2.8, 4.0])
    second = _trace(
        time_ns=second_time,
        position_mm=np.zeros((5, 3)),
        velocity_mm_per_ns=np.zeros((5, 3)),
        mass_amu=1.0,
        observer_charge_native=1.0,
        source_charge_native=1.0,
    )
    return first, second


def test_first_pass_capture_uses_same_radius_outbound_energy_and_periapsis() -> None:
    first, second = _capturing_pair()

    result = analyze_first_pass_capture(first, second)

    assert result.complete_same_radius_pass
    assert result.diagnostics_valid
    assert result.captured
    assert result.invalid_reasons == ()
    assert result.initial_osculating_energy_native > 0.0
    assert result.outbound_reference_energy_native < 0.0
    assert result.periapsis_time_ns == pytest.approx(2.0)
    assert result.periapsis_separation_mm == pytest.approx(1.0)
    assert result.series.radial_velocity_mm_per_ns[0] < 0.0
    assert result.series.radial_velocity_mm_per_ns[-1] > 0.0
    assert result.total_far_radiated_energy_native == pytest.approx(0.07)
    assert "not conserved for retarded fields" in result.energy_model


def test_capture_rejects_any_medina_cap() -> None:
    first, second = _capturing_pair()
    first = replace(
        first,
        medina_impulse_capped=np.array([False, False, False, True, False]),
    )

    result = analyze_first_pass_capture(first, second)

    assert result.complete_same_radius_pass
    assert not result.diagnostics_valid
    assert not result.captured
    assert "first Medina impulse was capped" in result.invalid_reasons


def test_capture_requires_an_initially_unbound_state() -> None:
    first, second = _capturing_pair()
    beta = first.beta.copy()
    beta[0, 0] = -0.5 / C_MMNS
    gamma = 1.0 / np.sqrt(1.0 - np.sum(beta * beta, axis=1))
    mechanical = stored_mechanical_four_momentum_series_native(
        gamma, beta, mass_amu=first.mass_amu
    )
    first = replace(
        first,
        beta=beta,
        gamma=gamma,
        canonical_four_momentum_native=mechanical,
    )

    result = analyze_first_pass_capture(first, second)

    assert result.initial_osculating_energy_native < 0.0
    assert result.outbound_reference_energy_native < 0.0
    assert not result.diagnostics_valid
    assert not result.captured
    assert "initial osculating state is already bound" in result.invalid_reasons


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("observer_charge_native", np.nan, "charges must be finite"),
        (
            "far_radiated_energy_native",
            np.array([0.0, 0.0, np.nan, 0.0, 0.0]),
            "far_radiated_energy_native must contain only finite",
        ),
        (
            "gamma",
            np.array([2.0, 1.0, 1.0, 1.0, 1.0]),
            "gamma and beta do not describe the same velocity",
        ),
    ],
)
def test_capture_rejects_invalid_trace_data(
    field_name: str,
    replacement: object,
    message: str,
) -> None:
    first, second = _capturing_pair()
    first = replace(first, **{field_name: replacement})

    with pytest.raises(ValueError, match=message):
        analyze_first_pass_capture(first, second)


def test_nonreciprocal_source_observer_weights_have_no_two_body_energy() -> None:
    first, second = _capturing_pair()
    first = replace(first, source_charge_native=-2.0)

    with pytest.raises(ValueError, match="not reciprocal"):
        analyze_first_pass_capture(first, second)
