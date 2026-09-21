"""Applied magnetic recoil and spin constraints, independent of shell closure."""

import json
import copy
from dataclasses import replace

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.experimental_spin_reaction import (
    LinearSpinFeedbackRecord,
    apply_linear_spin_impulse,
    transport_spin_between_velocities,
)
from core.magnetic_dipole import boost_rest_polarization
from core.rfs import rfs_charge_radiation_reaction_terms_native


def velocity(beta):
    beta = np.asarray(beta)
    gamma = 1 / np.sqrt(1 - beta @ beta)
    return gamma * c * np.r_[1.0, beta]


def boost(beta):
    beta = np.asarray(beta)
    gamma = 1 / np.sqrt(1 - beta @ beta)
    matrix = np.eye(4)
    matrix[0, 0] = gamma
    matrix[0, 1:] = gamma * beta
    matrix[1:, 0] = gamma * beta
    matrix[1:, 1:] += gamma**2 / (gamma + 1) * np.outer(beta, beta)
    return matrix


def dot(a, b):
    return a[0] * b[0] - a[1:] @ b[1:]


@pytest.mark.parametrize("beta", [(0, 0, 0), (0.3, 0.2, -0.1), (0.999999, 0, 0)])
def test_spin_transport_preserves_rest_norm_and_velocity_constraint(beta):
    u = velocity(beta)
    spin = boost_rest_polarization([0.3, 0.4, 0.5], beta)
    change = boost([0.01, -0.02, 0.03])
    v = change @ u
    moved = transport_spin_between_velocities(spin, u, v)
    scale = np.linalg.norm(moved) * np.linalg.norm(v)
    assert abs(dot(moved, v)) <= 2e-12 * scale
    assert dot(moved, moved) == pytest.approx(dot(spin, spin), abs=2e-9, rel=2e-9)


def test_transport_commutes_with_an_independent_frame_boost():
    u, v = velocity([0.1, 0.2, 0]), velocity([-0.15, 0.1, 0.3])
    spin = boost_rest_polarization([0.3, -0.2, 0.7], u[1:] / u[0])
    frame = boost([0.4, -0.1, 0.2])
    np.testing.assert_allclose(
        transport_spin_between_velocities(frame @ spin, frame @ u, frame @ v),
        frame @ transport_spin_between_velocities(spin, u, v),
        rtol=3e-14,
        atol=2e-15,
    )


def test_small_kick_matches_existing_rfs_spin_transport_sign():
    u = velocity([0.1, -0.2, 0.3])
    spin = boost_rest_polarization([0.3, 0.4, 0.5], u[1:] / u[0])
    force = np.array([0.2, -0.1, 0.4])
    terms = rfs_charge_radiation_reaction_terms_native(
        four_velocity_mm_ns=u,
        spin_four_vector=spin,
        applied_radiation_reaction_force_native=force,
        mass_amu=1.0,
    )
    errors = []
    for h in (0.01, 0.005):
        spatial = u[1:] + h * terms.four_acceleration[1:]
        v = np.r_[np.hypot(c, np.linalg.norm(spatial)), spatial]
        moved = transport_spin_between_velocities(spin, u, v)
        errors.append(np.linalg.norm(moved - spin - h * terms.spin_rhs_correction))
    assert errors[1] < 0.3 * errors[0]


def state(beta=(0.1, 0.2, 0), t=0.0):
    u = velocity(beta)
    values = dict(
        t=t,
        x=1.0,
        y=2.0,
        z=3.0,
        m=2.0,
        m_species=2.0,
        gamma=u[0] / c,
        Pt=2 * u[0] + 0.05,
        Px=2 * u[1] + 0.01,
        Py=2 * u[2] + 0.02,
        Pz=2 * u[3] + 0.03,
        spin_x=0.3,
        spin_y=0.4,
        spin_z=0.5,
        radiation_reaction_work=7.0,
        radiation_energy=8.0,
        medina_external_force_x=9.0,
        medina_external_force_y=10.0,
        medina_external_force_z=11.0,
        mass_shell_projection_energy=12.0,
        beta_samples=3.0,
    )
    for i, a in enumerate("xyz"):
        values[f"b{a}"] = beta[i]
        values[f"bdot{a}"] = 0.0
        values[f"beta_avg_{a}"] = beta[i]
    return {k: np.array([v], dtype=float) for k, v in values.items()}


def test_zero_recoil_is_exact_identity_except_its_own_record():
    start, result = state(), state(t=0.1)
    actual = apply_linear_spin_impulse(
        result=result,
        start=start,
        four_force_native=np.zeros(4),
        proper_step_ns=0.1,
        route="test_zero",
        force_ratio=0.0,
    )
    for k, v in result.items():
        np.testing.assert_array_equal(actual[k], v)
    assert not actual["_linear_spin_feedback_record"].applied


def test_applied_impulse_changes_motion_without_changing_medina_accounts():
    start, result = state(), state(t=0.1)
    force = np.array([0.02, 0.2, 0.0, 0.0])
    actual = apply_linear_spin_impulse(
        result=result,
        start=start,
        four_force_native=force,
        proper_step_ns=0.1,
        route="test",
        force_ratio=0.001,
    )
    assert actual["Px"][0] - result["Px"][0] == pytest.approx(0.02)
    assert actual["x"][0] - result["x"][0] == pytest.approx(0.0005)
    for k in (
        "radiation_reaction_work",
        "radiation_energy",
        "medina_external_force_x",
        "medina_external_force_y",
        "medina_external_force_z",
        "mass_shell_projection_energy",
    ):
        np.testing.assert_array_equal(actual[k], result[k])
    assert actual["_linear_spin_feedback_record"].work_native > 0
    assert not actual["_source_start_acceleration_complete"][0]
    np.testing.assert_array_equal(result["x"], [1.0])


def test_stable_work_survives_rest_scale_subtraction():
    start, result = state(beta=(0.0, 0.0, 0.0)), state(beta=(0.0, 0.0, 0.0), t=0.1)
    actual = apply_linear_spin_impulse(
        result=result,
        start=start,
        four_force_native=[0.0, 1e-6, 0.0, 0.0],
        proper_step_ns=0.1,
        route="tiny",
        force_ratio=0.001,
    )
    assert actual["Pt"][0] == result["Pt"][0]
    assert actual["_linear_spin_feedback_record"].work_native == pytest.approx(
        2.5e-15, rel=1e-12
    )


def test_feedback_record_checkpoint_roundtrip_and_key_guards():
    record = LinearSpinFeedbackRecord(
        "test", True, (0.0, 1.0, 2.0, 3.0), 0.1, 0.2, 0.0, 0.01
    )
    restored = LinearSpinFeedbackRecord.from_checkpoint_payload(
        json.loads(json.dumps(record.to_checkpoint_payload()))
    )
    assert restored == record
    with pytest.raises(ValueError, match="keys"):
        LinearSpinFeedbackRecord.from_checkpoint_payload({})
    with pytest.raises(ValueError, match="unapplied"):
        LinearSpinFeedbackRecord(
            "test", False, (0.0, 1.0, 0.0, 0.0), 0.1, 0.0, 0.0, 0.0
        )


@pytest.mark.parametrize("step", [0, -1, float("nan")])
def test_bad_step_rejected(step):
    with pytest.raises(ValueError):
        apply_linear_spin_impulse(
            result=state(t=0.1),
            start=state(),
            four_force_native=np.zeros(4),
            proper_step_ns=step,
            route="test",
            force_ratio=0.0,
        )


@pytest.mark.parametrize("beta", [0.02, 0.8, 0.9999])
def test_known_magnetic_bend_has_first_order_recoil_error(beta):
    """A manufactured transverse force has an independent circular solution.

    This tests the impulse/drift machinery, not the Jakobsen force formula.
    Pure transverse force does no exact work: the spurious kinetic-energy
    change must appear in the new numerical-adjustment account and converge.
    """
    errors, adjustments = [], []
    total_time, omega = 0.4, 0.5
    initial = state(beta=(beta, 0, 0))
    initial_p = 2 * velocity([beta, 0, 0])[1:]
    expected_p = (
        np.array([np.cos(omega * total_time), np.sin(omega * total_time), 0])
        * initial_p[0]
    )
    for steps in (32, 64, 128):
        current = copy.deepcopy(initial)
        h = total_time / steps
        adjustment = 0.0
        for _ in range(steps):
            base = copy.deepcopy(current)
            base["t"] += h * current["gamma"]
            p = (
                2
                * current["gamma"][0]
                * c
                * np.array([current[f"b{a}"][0] for a in "xyz"])
            )
            for i, axis in enumerate("xyz"):
                base[axis] += h * p[i] / 2
            force = np.r_[0, omega * np.array([-p[1], p[0], 0])]
            current = apply_linear_spin_impulse(
                result=base,
                start=current,
                four_force_native=force,
                proper_step_ns=h,
                route="manufactured_bend",
                force_ratio=0.01,
            )
            record = current["_linear_spin_feedback_record"]
            adjustment += record.work_native - record.temporal_impulse_energy_native
        final_p = (
            2 * current["gamma"][0] * c * np.array([current[f"b{a}"][0] for a in "xyz"])
        )
        errors.append(np.linalg.norm(final_p - expected_p) / np.linalg.norm(expected_p))
        adjustments.append(adjustment)
        kinetic_change = c * (
            np.hypot(2 * c, np.linalg.norm(final_p))
            - np.hypot(2 * c, np.linalg.norm(initial_p))
        )
        assert adjustment == pytest.approx(kinetic_change, rel=3e-8)
        assert np.linalg.norm(
            [current[f"spin_{a}"][0] for a in "xyz"]
        ) == pytest.approx(np.linalg.norm([0.3, 0.4, 0.5]), rel=3e-9)
    assert 1.9 < errors[0] / errors[1] < 2.1
    assert 1.9 < errors[1] / errors[2] < 2.1
    assert 1.9 < adjustments[0] / adjustments[1] < 2.1


def test_feedback_lifetime_totals_survive_record_eviction():
    from core.spin_self_force_reduction_history import (
        IntrinsicSpinReductionDiagnosticRecord,
        IntrinsicSpinReductionDiagnosticTrace,
    )

    trace = IntrinsicSpinReductionDiagnosticTrace(maximum_records=2)
    for i, work in enumerate([0.4, -0.6, 0.2, -0.1]):
        feedback = LinearSpinFeedbackRecord(
            "test", True, (0.01, 1.0, 0, 0), 0.1, work, 0.001 * c, 0.01
        )
        trace = trace.append(
            IntrinsicSpinReductionDiagnosticRecord(
                proper_time_ns=0.1 * i,
                route="analytical_smooth_segment",
                analytical_unavailable_reason=None,
                causal_condition_number=None,
                linear_spin_four_force_native=(0.01, 1.0, 0, 0),
                charge_ald_four_force_native=(0.0,) * 4,
                total_four_force_native=(0.01, 1.0, 0, 0),
                balance_residual_norm_native=0.0,
                applied_feedback=feedback,
            )
        )
    assert len(trace.records) == 2
    assert trace.feedback_evaluated_records == trace.feedback_applied_records == 4
    assert trace.feedback_work_native == pytest.approx(-0.1)
    assert trace.feedback_absolute_work_native == pytest.approx(1.3)
    np.testing.assert_allclose(trace.feedback_four_impulse_native, [0.004, 0.4, 0, 0])
    assert trace.feedback_energy_adjustment_native == pytest.approx(-0.1 - 0.004 * c)
    restored = IntrinsicSpinReductionDiagnosticTrace.from_checkpoint_payload(
        json.loads(json.dumps(trace.to_checkpoint_payload()))
    )
    assert restored == trace
    with pytest.raises(ValueError, match="integers"):
        replace(trace, feedback_applied_records=1.5)
    payload = trace.to_checkpoint_payload()
    payload["feedback_totals"] = {}
    with pytest.raises(ValueError, match="keys"):
        IntrinsicSpinReductionDiagnosticTrace.from_checkpoint_payload(payload)


def test_adaptive_reducer_includes_new_work_and_adjustment():
    from core.step_doubling import build_pair_step_doubling_state

    first, second = state(t=0.1), state(t=0.2)
    first["_linear_spin_feedback_record"] = LinearSpinFeedbackRecord(
        "test", True, (0.0, 1.0, 0, 0), 0.1, 0.3, 0.0, 0.01
    )
    second["_linear_spin_feedback_record"] = LinearSpinFeedbackRecord(
        "test", True, (0.0, 1.0, 0, 0), 0.1, -0.1, 0.0, 0.01
    )
    values = build_pair_step_doubling_state(
        rider_states=(first, second), driver_states=(first, second)
    )
    assert values.diagnostics_native.shape == (2, 6)
    np.testing.assert_allclose(values.diagnostics_native[:, -2:], [[0.2, 0.2]] * 2)
    with pytest.raises(ValueError, match="missing a recoil record"):
        build_pair_step_doubling_state(rider_states=(first,), driver_states=(state(),))
