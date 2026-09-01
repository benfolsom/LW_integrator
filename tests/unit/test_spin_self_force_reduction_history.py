"""Accepted-history lifecycle for the causal spin-reduction fallback."""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.constants import C_MMNS
from core.spin_self_force_reduction_history import (
    AcceptedIntrinsicSpinReductionHistory,
    AcceptedPairIntrinsicSpinReductionHistory,
    IntrinsicSpinReductionDiagnosticRecord,
    IntrinsicSpinReductionDiagnosticTrace,
    select_intrinsic_spin_reduction_route_native,
)
from core.spin_self_force_reduction_oracle import (
    evaluate_potential_directional_intrinsic_spin_reduction_native,
)


def _sample(proper_time_ns: float) -> dict[str, object]:
    time = float(proper_time_ns)
    beta_x = 0.1
    gamma = 1.0 / np.sqrt(1.0 - beta_x**2)
    velocity = np.asarray(
        (
            gamma * C_MMNS,
            gamma * C_MMNS * beta_x,
            0.0,
            0.0,
        )
    )
    acceleration = np.zeros(4)
    # A constant z-directed spin is orthogonal to the x-directed velocity.
    spin = np.asarray((0.0, 0.0, 0.0, 1.0))
    return {
        "proper_time_ns": time,
        "four_velocity_mm_ns": velocity,
        "non_self_four_acceleration_mm_ns2": acceleration,
        "physical_spin_four_native": spin,
    }


def _accepted_history(count: int = 6) -> AcceptedIntrinsicSpinReductionHistory:
    history = AcceptedIntrinsicSpinReductionHistory.empty()
    for index in range(count):
        history = history.append_accepted(**_sample(0.1 * index))
    return history


def _zero_analytical_reduction():
    zero_hessian = np.zeros((4, 4, 4))
    return evaluate_potential_directional_intrinsic_spin_reduction_native(
        four_velocity_mm_ns=(C_MMNS, 0.0, 0.0, 0.0),
        normalized_spin_four_vector=(0.0, 0.0, 0.0, 1.0),
        partial_a=np.zeros((4, 4)),
        partial2_a=zero_hessian,
        partial3_a_along_velocity=zero_hessian,
        partial3_a_along_acceleration=zero_hessian,
        partial4_a_along_velocity_twice=zero_hessian,
        charge_native=1.0,
        mass_amu=1.0,
        invariant_spin_native=1.0,
        g_factor=2.0,
    )


def test_candidate_append_cannot_mutate_accepted_history() -> None:
    accepted = _accepted_history()
    before = accepted.to_checkpoint_payload()

    rejected_candidate = accepted.append_accepted(**_sample(0.6))

    assert accepted.sample_count == 6
    assert rejected_candidate.sample_count == 6
    assert accepted.to_checkpoint_payload() == before
    assert float(accepted.proper_times_ns[-1]) == pytest.approx(0.5)
    assert float(rejected_candidate.proper_times_ns[-1]) == pytest.approx(0.6)
    for array in (
        accepted.proper_times_ns,
        accepted.four_velocity_mm_ns,
        accepted.non_self_four_acceleration_mm_ns2,
        accepted.physical_spin_four_native,
    ):
        assert array.flags.writeable is False


def test_checkpoint_roundtrip_reproduces_next_candidate_and_force() -> None:
    accepted = _accepted_history()
    serialized = json.loads(json.dumps(accepted.to_checkpoint_payload()))
    restored = AcceptedIntrinsicSpinReductionHistory.from_checkpoint_payload(serialized)

    for name in (
        "proper_times_ns",
        "four_velocity_mm_ns",
        "non_self_four_acceleration_mm_ns2",
        "physical_spin_four_native",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(accepted, name))
    original_next = accepted.append_accepted(**_sample(0.6))
    restored_next = restored.append_accepted(**_sample(0.6))
    np.testing.assert_array_equal(
        restored_next.four_velocity_mm_ns,
        original_next.four_velocity_mm_ns,
    )
    original_force = original_next.evaluate_causal(
        charge_native=1.2,
        mass_amu=2.0,
        g_factor=2.1,
    )
    restored_force = restored_next.evaluate_causal(
        charge_native=1.2,
        mass_amu=2.0,
        g_factor=2.1,
    )
    np.testing.assert_array_equal(
        restored_force.radiation_balance.self_force.linear_spin_self_force_native,
        original_force.radiation_balance.self_force.linear_spin_self_force_native,
    )


def test_pair_checkpoint_roundtrip_preserves_both_role_histories() -> None:
    pair = AcceptedPairIntrinsicSpinReductionHistory(
        rider=_accepted_history(),
        driver=_accepted_history(4),
        rider_endpoint_proper_time_ns=0.5,
        driver_endpoint_proper_time_ns=0.4,
    )

    serialized = json.loads(json.dumps(pair.to_checkpoint_payload()))
    restored = AcceptedPairIntrinsicSpinReductionHistory.from_checkpoint_payload(
        serialized
    )

    assert restored.to_checkpoint_payload() == pair.to_checkpoint_payload()
    assert restored.rider.sample_count == 6
    assert restored.driver.sample_count == 4


def test_route_uses_analytical_result_without_causal_history() -> None:
    analytical = _zero_analytical_reduction()
    selected = select_intrinsic_spin_reduction_route_native(
        analytical_reduction=analytical,
        analytical_unavailable_reason=None,
        accepted_history=AcceptedIntrinsicSpinReductionHistory.empty(),
        charge_native=1.0,
        mass_amu=1.0,
        g_factor=2.0,
    )

    assert selected.route == "analytical_smooth_segment"
    assert selected.analytical_reduction is analytical
    assert selected.causal_reduction is None


def test_boundary_route_uses_newest_accepted_sample_without_phase_delay() -> None:
    accepted = _accepted_history()
    selected = select_intrinsic_spin_reduction_route_native(
        analytical_reduction=None,
        analytical_unavailable_reason="retarded root is on a C1 spin knot",
        accepted_history=accepted,
        charge_native=1.2,
        mass_amu=2.0,
        g_factor=2.1,
    )

    assert selected.route == "causal_accepted_history_boundary_fallback"
    assert selected.analytical_reduction is None
    assert selected.causal_reduction is not None
    assert selected.causal_reduction.evaluation_proper_time_ns == pytest.approx(
        accepted.proper_times_ns[-1]
    )
    assert selected.causal_reduction.uses_future_samples is False
    assert np.isfinite(selected.causal_reduction.scaled_vandermonde_condition_number)
    assert selected.causal_condition_number == pytest.approx(
        selected.causal_reduction.scaled_vandermonde_condition_number
    )
    assert selected.unavailable_reason == "retarded root is on a C1 spin knot"


def test_boundary_route_fails_closed_before_six_accepted_samples() -> None:
    selected = select_intrinsic_spin_reduction_route_native(
        analytical_reduction=None,
        analytical_unavailable_reason="segment boundary",
        accepted_history=_accepted_history(5),
        charge_native=1.0,
        mass_amu=1.0,
        g_factor=2.0,
    )

    assert selected.route == "unavailable_insufficient_accepted_history"
    assert selected.analytical_reduction is None
    assert selected.causal_reduction is None
    assert selected.causal_condition_number is None
    assert selected.unavailable_reason == "segment boundary"


def test_boundary_route_rejects_an_ill_conditioned_causal_fit() -> None:
    accepted = _accepted_history()
    candidate = accepted.evaluate_causal(
        charge_native=1.2,
        mass_amu=2.0,
        g_factor=2.1,
    )
    condition = candidate.scaled_vandermonde_condition_number
    selected = select_intrinsic_spin_reduction_route_native(
        analytical_reduction=None,
        analytical_unavailable_reason="segment boundary",
        accepted_history=accepted,
        charge_native=1.2,
        mass_amu=2.0,
        g_factor=2.1,
        maximum_causal_condition_number=0.5 * condition,
    )

    assert selected.route == "unavailable_ill_conditioned_causal_fit"
    assert selected.analytical_reduction is None
    assert selected.causal_reduction is None
    assert selected.causal_condition_number == pytest.approx(condition)
    assert selected.unavailable_reason == "segment boundary"


def test_route_rejects_an_invalid_causal_condition_limit() -> None:
    with pytest.raises(ValueError, match="maximum_causal_condition_number"):
        select_intrinsic_spin_reduction_route_native(
            analytical_reduction=_zero_analytical_reduction(),
            analytical_unavailable_reason=None,
            accepted_history=AcceptedIntrinsicSpinReductionHistory.empty(),
            charge_native=1.0,
            mass_amu=1.0,
            g_factor=2.0,
            maximum_causal_condition_number=np.inf,
        )


def test_ill_conditioned_record_retains_condition_without_force() -> None:
    record = IntrinsicSpinReductionDiagnosticRecord(
        proper_time_ns=0.5,
        route="unavailable_ill_conditioned_causal_fit",
        analytical_unavailable_reason="segment boundary",
        causal_condition_number=1.25e5,
        linear_spin_four_force_native=None,
        charge_ald_four_force_native=None,
        total_four_force_native=None,
        balance_residual_norm_native=None,
    )

    restored = IntrinsicSpinReductionDiagnosticRecord.from_checkpoint_payload(
        json.loads(json.dumps(record.to_checkpoint_payload()))
    )

    assert restored == record
    assert restored.causal_condition_number == pytest.approx(1.25e5)


def test_checkpoint_payload_rejects_unknown_schema_or_keys() -> None:
    payload = _accepted_history().to_checkpoint_payload()
    payload["schema_version"] = 999
    with pytest.raises(ValueError, match="unsupported"):
        AcceptedIntrinsicSpinReductionHistory.from_checkpoint_payload(payload)

    payload = _accepted_history().to_checkpoint_payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="keys do not match"):
        AcceptedIntrinsicSpinReductionHistory.from_checkpoint_payload(payload)


def test_diagnostic_trace_is_bounded_but_preserves_lifetime_route_counts() -> None:
    trace = IntrinsicSpinReductionDiagnosticTrace(maximum_records=2)
    unavailable = IntrinsicSpinReductionDiagnosticRecord(
        proper_time_ns=0.0,
        route="unavailable_insufficient_accepted_history",
        analytical_unavailable_reason="segment boundary",
        causal_condition_number=None,
        linear_spin_four_force_native=None,
        charge_ald_four_force_native=None,
        total_four_force_native=None,
        balance_residual_norm_native=None,
    )
    analytical = IntrinsicSpinReductionDiagnosticRecord(
        proper_time_ns=0.1,
        route="analytical_smooth_segment",
        analytical_unavailable_reason=None,
        causal_condition_number=None,
        linear_spin_four_force_native=(0.0, 1.0, 0.0, 0.0),
        charge_ald_four_force_native=(0.0, 0.0, 0.0, 0.0),
        total_four_force_native=(0.0, 1.0, 0.0, 0.0),
        balance_residual_norm_native=0.0,
    )
    causal = IntrinsicSpinReductionDiagnosticRecord(
        proper_time_ns=0.2,
        route="causal_accepted_history_boundary_fallback",
        analytical_unavailable_reason="segment boundary",
        causal_condition_number=42.0,
        linear_spin_four_force_native=(0.0, 2.0, 0.0, 0.0),
        charge_ald_four_force_native=(0.0, 0.0, 0.0, 0.0),
        total_four_force_native=(0.0, 2.0, 0.0, 0.0),
        balance_residual_norm_native=1.0e-30,
    )
    for record in (unavailable, analytical, causal):
        trace = trace.append(record)

    assert trace.total_records == 3
    assert trace.analytical_records == 1
    assert trace.causal_records == 1
    assert trace.unavailable_records == 1
    assert [record.proper_time_ns for record in trace.records] == [0.1, 0.2]
    restored = IntrinsicSpinReductionDiagnosticTrace.from_checkpoint_payload(
        json.loads(json.dumps(trace.to_checkpoint_payload()))
    )
    assert restored.to_checkpoint_payload() == trace.to_checkpoint_payload()
