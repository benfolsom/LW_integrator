"""Stored charge fields along production coasting steps, with exact Decimal oracles."""

from decimal import Decimal as D, localcontext
import json
from pathlib import Path

import numpy as np
import pytest

from core.constants import C_MMNS
from core.equations import retarded_equations_of_motion
from core.retarded_fields import (
    ObserverEvent,
    _evaluate_prepared_charge_batch,
    _prepare_history,
    evaluate_retarded_charge_field_native,
)
from core.resolved_knot import initialize_resolved_result
from core.self_consistency import SelfConsistencyConfig
from core.types import SimulationType, TrajectoryBuilder
from tests.unit.test_inertial_prehistory import _state

pytestmark = pytest.mark.unit

ROWS = json.loads(
    (Path(__file__).parent / "data/high_gamma_uniform_charge.json").read_text()
)["rows"]


def _split(value):
    high = float(value)
    low = float(value - D.from_float(high))
    tail = float(value - D.from_float(high) - D.from_float(low))
    return high, low, tail


def _coast(
    gamma, *, position=None, time=None, sign=1, span_mm=2.0, steps=1200, charge=1.0
):
    """Run the real stepper; source charge is read by a later exact field probe.

    A neutral force observer isolates the first-order coasting worldline. The
    prescribed source charge has no self force in this single-particle run.
    """
    state = _state(position_mm=(0.0, 0.0, 0.0), source_charge=charge)
    u = sign * np.sqrt(gamma**2 - 1.0)
    mechanical_gamma = np.hypot(1.0, abs(u))
    state["gamma"][:] = mechanical_gamma
    state["Pt"][:] = mechanical_gamma * C_MMNS
    state["Px"][:] = u * C_MMNS
    state["bx"][:] = u / mechanical_gamma
    initialize_resolved_result(state, state)
    state["source_u_x"][:] = u
    state["source_kinematics_ready"][:] = 1.0
    with localcontext() as context:
        context.prec = 90
        if position is not None:
            for axis, value in zip("xyz", position):
                high, low, tail = _split(value)
                state[axis][:] = high
                state[f"source_position_low_{axis}"][:] = low
                state[f"source_position_tail_{axis}"][:] = tail
        if time is not None:
            high, low, tail = _split(time)
            state["t"][:] = high
            state["source_time_low_ns"][:] = low
            state["source_time_tail_ns"][:] = tail
    # Convert the requested lab distance to proper time, as in AGENTS.md.
    beta = abs(u) / mechanical_gamma
    h = span_mm / (mechanical_gamma * beta * C_MMNS * (steps - 1))
    states = [state]
    neutral_external = [_state(position_mm=(0.0, 1e6, 0.0))]
    consistency = SelfConsistencyConfig(
        enabled=True,
        convergence_mode="fixed_geometry",
        max_iterations=2,
        target_ms_tolerance=1e-6,
        mass_shell_tolerance=0.01,
        mass_shell_relaxation=0.7,
        gamma_reconciliation_method="DISABLED",
    )
    for step in range(1, steps):
        states.append(
            retarded_equations_of_motion(
                h,
                states,
                neutral_external,
                step - 1,
                1e99,
                SimulationType.BUNCH_TO_BUNCH,
                radiation_reaction_mode="medina_lad",
                self_consistency=consistency,
                step_idx=step,
            )
        )
    builder = TrajectoryBuilder(steps, 1)
    for step, row in enumerate(states):
        builder.set_step(step, row)
    return builder.build(), states


def _event(row):
    return ObserverEvent(
        float(row["t"][0]),
        tuple(float(row[a][0]) for a in "xyz"),
        time_low_ns=float(row["source_time_low_ns"][0]),
        position_low_mm=tuple(float(row[f"source_position_low_{a}"][0]) for a in "xyz"),
        time_tail_ns=float(row["source_time_tail_ns"][0]),
        position_tail_mm=tuple(
            float(row[f"source_position_tail_{a}"][0]) for a in "xyz"
        ),
    )


@pytest.fixture(scope="module", params=[10.0**n for n in range(3, 13)])
def coasting_source(request):
    return request.param, _coast(request.param)[0]


@pytest.mark.parametrize("factor", [0.0, 1.0])
def test_coasting_source_and_witness_against_decimal(
    coasting_source, factor, record_property
):
    gamma, source = coasting_source
    row = next(
        row
        for row in ROWS
        if row["gamma"] == gamma
        and row["side"] == "ahead"
        and row["angle_factor"] == factor
    )
    with localcontext() as context:
        context.prec = 90
        u = D.from_float(float(row["proper_velocity"][0]))
        g = (1 + u * u).sqrt()
        theta = D(str(factor)) / D(str(gamma))
        # Independent Taylor sums retain the longitudinal cone displacement
        # that cannot be represented in row['direction'][0] alone.
        sine, cosine = theta, D(1)
        st, ct = theta, D(1)
        for order in range(1, 30):
            st *= -theta * theta / D((2 * order) * (2 * order + 1))
            ct *= -theta * theta / D((2 * order - 1) * (2 * order))
            sine += st
            cosine += ct
        offset = [cosine - u / g, sine, D(0)]
    _, witness = _coast(gamma, position=offset)
    event = _event(witness[-1])
    fields = [
        evaluate_retarded_charge_field_native(source, event, backend=backend)
        for backend in ("python", "numba_full_strict_serial")
    ]
    record_property("gamma", gamma)
    record_property("angle_factor", factor)
    for backend, field in zip(("python", "numba"), fields):
        expected = np.array(row["electric"])
        relative_error = np.linalg.norm(
            field.electric_field_native - expected
        ) / np.linalg.norm(expected)
        record_property(backend + "_field_relative_error", float(relative_error))
        assert relative_error < 2e-13
        assert field.valid_sources[0]
        # This must exercise stored accepted segments, with no analytic prefix.
        assert 0.0 < field.retarded_time_ns[0] < source.t[-1, 0]
    np.testing.assert_allclose(
        fields[1].electric_field_native,
        fields[0].electric_field_native,
        rtol=5e-15,
        atol=0.0,
    )


@pytest.mark.parametrize("gamma", [10.0**n for n in range(3, 13)])
def test_counter_propagating_first_order_impulse(gamma, record_property):
    """Integrate force/work on accepted coasting paths (first-order test particles)."""
    b = 1e-12
    source_charge = witness_charge = 1e-9
    window = 100.0
    with localcontext() as context:
        context.prec = 90
        u = D.from_float(float(np.sqrt(gamma**2 - 1)))
        g = (1 + u * u).sqrt()
        beta = u / g
        source_ct = -D(1000)
        source_time = source_ct / D.from_float(C_MMNS)
        source_position = [beta * source_ct, D(0), D(0)]
        half_ct = D(str(window * b)) / (g * (2 * beta))
        witness_time = -half_ct / D.from_float(C_MMNS)
        witness_position = [beta * half_ct, D(str(b)), D(0)]
    source, _ = _coast(
        gamma,
        time=source_time,
        position=source_position,
        span_mm=1000.0,
        charge=source_charge,
    )
    _, witness = _coast(
        gamma,
        time=witness_time,
        position=witness_position,
        sign=-1,
        span_mm=float(2 * half_ct * beta),
        charge=0.0,
    )
    prepared = _prepare_history(source, ())
    fields = _evaluate_prepared_charge_batch(
        prepared,
        [_event(row) for row in witness],
        backend="numba_full_strict_serial",
        require_complete_history=True,
        root_tolerance_mm=1e-21,
        max_root_iterations=96,
    )
    beta_rider = -float(beta)
    electric = np.array([field.electric_field_native for field in fields])
    magnetic = np.array([field.magnetic_field_native for field in fields])
    force_y = witness_charge * (electric[:, 1] - beta_rider * magnetic[:, 2])
    time = np.array([row["t"][0] for row in witness])
    impulse = np.trapezoid(force_y, time)
    work = np.trapezoid(witness_charge * C_MMNS * beta_rider * electric[:, 0], time)
    expected = (
        2
        * source_charge
        * witness_charge
        * (1 + float(beta) ** 2)
        / (2 * float(beta) * C_MMNS * b)
    )
    # The finite window retains W/sqrt(1+W^2) of the infinite impulse.
    finite_expected = expected * window / np.sqrt(1 + window**2)
    record_property("gamma", gamma)
    record_property(
        "finite_impulse_relative_error", float(abs(impulse / finite_expected - 1))
    )
    record_property(
        "infinite_impulse_relative_error", float(abs(impulse / expected - 1))
    )
    record_property("normalized_net_work", float(abs(work) / (abs(expected) * C_MMNS)))
    assert abs(impulse / finite_expected - 1) < 2e-6
    assert abs(impulse / expected - 1) < 6e-5
    assert abs(work) / (abs(expected) * C_MMNS) < 2e-6


def test_high_gamma_checkpoint_trial_and_append_preserve_fields(tmp_path):
    from core.integration_checkpoint import IntegrationCheckpointStore
    from core.types import TrialTrajectoryHistory
    from core.resolved_knot import RESOLVED_KNOT_FIELDS

    source, rows = _coast(1e12, steps=12)
    builder = TrajectoryBuilder(12, 1)
    for step, row in enumerate(rows[:10]):
        builder.set_step(step, row)
    accepted = builder.build_partial(10)
    original = _prepare_history(accepted, ())
    frozen = [
        segment.coefficients_mm.copy()
        for segment in original.sources[0].light_cone_segments
    ]
    trial = _prepare_history(TrialTrajectoryHistory(accepted, tuple(rows[10:])), ())
    assert len(trial.sources[0].light_cone_segments) == 11
    for segment, before in zip(original.sources[0].light_cone_segments, frozen):
        np.testing.assert_array_equal(segment.coefficients_mm, before)
    store = IntegrationCheckpointStore(
        tmp_path / "checkpoint",
        compatibility_payload={"gamma": 1e12},
        total_steps=12,
        requested_steps=12,
        active_start=0,
        interval_steps=1,
        interval_seconds=0.0,
        resume=False,
    )
    store.write(step_index=9, rider=accepted, driver=accepted, loop_state={})
    resumed = TrajectoryBuilder(12, 1)
    reopened = IntegrationCheckpointStore(
        tmp_path / "checkpoint",
        compatibility_payload={"gamma": 1e12},
        total_steps=12,
        requested_steps=12,
        active_start=0,
        interval_steps=1,
        interval_seconds=0.0,
        resume=True,
    )
    reopened.restore_builder(resumed, "rider")
    for step, row in enumerate(rows[10:], 10):
        resumed.set_step(step, row)
        builder.set_step(step, row)
    restored = resumed.build()
    appended = builder.build()
    for name in RESOLVED_KNOT_FIELDS:
        np.testing.assert_array_equal(getattr(source, name), getattr(restored, name))
    from dataclasses import replace

    event = replace(_event(rows[-1]), position_mm=(float(rows[-1]["x"][0]), 1e-12, 0.0))
    expected = evaluate_retarded_charge_field_native(source, event)
    from core.retarded_fields import _evaluate_prepared_charge_field_native

    trial_field = _evaluate_prepared_charge_field_native(
        trial,
        event,
        require_complete_history=True,
        root_tolerance_mm=1e-21,
        max_root_iterations=96,
    )
    np.testing.assert_array_equal(
        trial_field.electric_field_native, expected.electric_field_native
    )
    for history in (restored, appended):
        actual = evaluate_retarded_charge_field_native(history, event)
        np.testing.assert_array_equal(
            actual.electric_field_native, expected.electric_field_native
        )


def test_stored_high_gamma_potential_jet_keeps_root_fraction():
    from core.retarded_potential_directional_jet import (
        evaluate_retarded_charge_potential_directional_jet_native,
    )
    from core.precise_charge import precise_charge_jets

    gamma = 1e12
    source, _ = _coast(gamma, steps=12, span_mm=3.0)
    with localcontext() as context:
        context.prec = 90
        u = D.from_float(float(np.sqrt(gamma**2 - 1)))
        g = (1 + u * u).sqrt()
        offset = [1 - u / g, D(0), D(0)]
    _, witness = _coast(gamma, steps=12, span_mm=2.13, position=offset)
    event = _event(witness[-1])
    result = evaluate_retarded_charge_potential_directional_jet_native(
        source,
        event,
        four_velocity_mm_ns=(C_MMNS, 0.0, 0.0, 0.0),
        four_acceleration_mm_ns2=(0.0, 0.0, 0.0, 0.0),
    )
    assert result.available
    potential = precise_charge_jets(
        1.0,
        np.array([1.0, 0.0, 0.0]),
        np.array([float(u), 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
    )[0]
    np.testing.assert_allclose(
        result.derivatives.four_potential, potential[:, 0], rtol=2e-13, atol=0.0
    )
    np.testing.assert_allclose(
        result.derivatives.partial_a, potential[:, 1:5].T, rtol=2e-12, atol=0.0
    )
    np.testing.assert_allclose(
        result.derivatives.partial2_a,
        potential[:, 5:].reshape(4, 4, 4).transpose(1, 2, 0),
        rtol=2e-12,
        atol=0.0,
    )
