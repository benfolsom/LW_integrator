"""Decimal uniform-motion references for both charge-kernel contracts."""

import json
from pathlib import Path

import numpy as np
import pytest

from core.ballistic_charge import ballistic_retarded_point
from core.charge_potential_jet import quintic_charge_potential_jet_native
from core.charge_potential_jet_numba import quintic_charge_potential_jet_strict_serial
from core.charge_response_jet import quintic_charge_response_jet_native
from core.charge_response_jet_numba import (
    quintic_charge_response_coefficients_strict_serial,
    quintic_charge_response_jet_strict_serial,
)
from core.constants import C_MMNS
from core.precise_charge import (
    precise_charge_jets,
    precise_charge_jets_strict_serial,
    resolved_separation_frame,
)

ROWS = json.loads(
    (Path(__file__).parent / "data/high_gamma_uniform_charge.json").read_text()
)["rows"]


def arguments(row):
    u = np.asarray(row["proper_velocity"])
    beta = u / np.hypot(1.0, np.linalg.norm(u))
    coefficients = np.zeros((6, 3))
    coefficients[1] = C_MMNS * beta
    return dict(
        observer_time_ns=1 / C_MMNS,
        observer_position_mm=np.asarray(row["direction"]),
        charge_native=1.0,
        segment_start_time_ns=0.0,
        segment_duration_ns=1.0,
        position_coefficients_mm=coefficients,
        retarded_time_ns=0.0,
    )


def assert_norm_close(actual, expected, tolerance=2e-13):
    actual, expected = np.asarray(actual), np.asarray(expected)
    scale = np.linalg.norm(expected)
    assert np.linalg.norm(actual - expected) <= tolerance * scale + 1e-280


@pytest.mark.parametrize(
    "row",
    ROWS,
    ids=lambda row: f"g{row['gamma']:g}-{row['side']}-{row['angle_factor']:g}",
)
def test_precise_kernel_against_decimal(row):
    kwargs = arguments(row)
    u = np.asarray(row["proper_velocity"])
    potential = quintic_charge_potential_jet_native(**kwargs, source_proper_velocity=u)
    response = quintic_charge_response_jet_native(**kwargs, source_proper_velocity=u)
    assert_norm_close(potential.four_potential, row["potential"])
    assert_norm_close(response.four_potential, row["potential"])
    assert_norm_close(-response.antisymmetric_response[:3], row["electric"])
    magnetic = np.asarray(
        (
            -response.antisymmetric_response[5],
            response.antisymmetric_response[4],
            -response.antisymmetric_response[3],
        )
    )
    assert_norm_close(magnetic, row["magnetic"])

    derivative_tolerance = (
        2e-8 if row["side"] == "ahead" and row["angle_factor"] == 1 else 2e-13
    )
    assert_norm_close(
        potential.partial_a[:, 0], row["phi_gradient"], derivative_tolerance
    )
    assert_norm_close(potential.partial2_a[:, :, 0], row["phi_hessian"])
    assert_norm_close(
        -response.partial_antisymmetric_response[:, :3],
        row["electric_gradient"],
        derivative_tolerance,
    )
    assert_norm_close(response.partial_a, potential.partial_a)
    assert response.kappa == pytest.approx(row["kappa"], rel=2e-14)
    # At the cone shoulder, the ideal-angle gradient has a near-zero
    # longitudinal numerator. Rounded input angles and proper velocities,
    # plus its final small difference, limit the achieved gradient accuracy
    # to 1.3e-8 on this grid. Hessians and values use the tighter tolerance.
    # Tiny longitudinal components near theta=1/gamma are assessed with the
    # vector norm, rather than a meaningless relative tolerance near zero.
    for actual, expected in zip(-response.antisymmetric_response[:3], row["electric"]):
        assert (
            abs(actual - expected) <= 2e-13 * np.linalg.norm(row["electric"]) + 1e-280
        )
    args = tuple(kwargs.values())
    compiled_p = quintic_charge_potential_jet_strict_serial(
        *args, source_proper_velocity=u
    )
    compiled_r = quintic_charge_response_coefficients_strict_serial(
        *args, source_proper_velocity=u
    )
    compiled_tensor = quintic_charge_response_jet_strict_serial(
        *args, source_proper_velocity=u
    )
    for actual, expected in zip(
        compiled_p[:3],
        (potential.four_potential, potential.partial_a, potential.partial2_a),
    ):
        assert_norm_close(actual, expected, 5e-15)
    for actual, expected in zip(
        compiled_r[:4],
        (
            response.four_potential,
            response.partial_a,
            response.antisymmetric_response,
            response.partial_antisymmetric_response,
        ),
    ):
        assert_norm_close(actual, expected, 5e-15)
    assert_norm_close(compiled_tensor[1], response.field_tensor, 5e-15)
    assert_norm_close(compiled_tensor[2], response.partial_f, 5e-15)


@pytest.mark.parametrize(
    "row",
    ROWS,
    ids=lambda row: f"g{row['gamma']:g}-{row['side']}-{row['angle_factor']:g}",
)
def test_beta_only_path_records_its_precision_limit(row):
    kwargs = arguments(row)
    # The original compiled potential kernel has no 1e-14 cutoff. Preserve
    # that distinction and measure its beta-only path at every reference.
    for compiled, quantity in (
        (quintic_charge_potential_jet_strict_serial, "potential"),
        (quintic_charge_response_coefficients_strict_serial, "electric"),
    ):
        try:
            result = compiled(*tuple(kwargs.values()))
        except (ValueError, ZeroDivisionError):
            assert row["side"] == "ahead" and row["gamma"] >= 1e7
            continue
        if row["gamma"] <= 1e3:
            value = result[0] if quantity == "potential" else -result[2][:3]
            assert_norm_close(value, row[quantity], 1e-8)
        explicit_none = compiled(*tuple(kwargs.values()), source_proper_velocity=None)
        for value, repeated in zip(result, explicit_none):
            np.testing.assert_array_equal(value, repeated)

    if (
        1.0
        - np.dot(
            kwargs["observer_position_mm"],
            kwargs["position_coefficients_mm"][1] / C_MMNS,
        )
        <= 1e-14
    ):
        for kernel in (
            quintic_charge_potential_jet_native,
            quintic_charge_response_jet_native,
        ):
            with pytest.raises((ValueError, ZeroDivisionError)):
                kernel(**kwargs)
        return
    potential = quintic_charge_potential_jet_native(**kwargs)
    response = quintic_charge_response_jet_native(**kwargs)
    # Existing rounded inputs are deliberately retained. Low-gamma reference
    # accuracy is asserted; high-gamma losses are measured by the report.
    if row["gamma"] <= 1e3:
        assert_norm_close(potential.four_potential, row["potential"], 1e-8)
        assert_norm_close(-response.antisymmetric_response[:3], row["electric"], 1e-8)
    np.testing.assert_array_equal(
        potential.four_potential,
        quintic_charge_potential_jet_native(
            **kwargs, source_proper_velocity=None
        ).four_potential,
    )
    np.testing.assert_array_equal(
        response.partial_f,
        quintic_charge_response_jet_native(
            **kwargs, source_proper_velocity=None
        ).partial_f,
    )


def test_ballistic_root_retains_small_deficit_with_a_large_retardation():
    from decimal import Decimal as D, localcontext

    for gamma in (10.0, 1e6, 1e12):
        u = np.asarray([np.sqrt(gamma * gamma - 1), 0, 0])
        point = ballistic_retarded_point(
            observer_time_ns=0.0,
            observer_position_mm=(1.0, 0, 0),
            anchor_time_ns=0.0,
            anchor_position_mm=(0, 0, 0),
            source_proper_velocity=u,
        )
        with localcontext() as context:
            context.prec = 100
            ud = D.from_float(float(u[0]))
            gd = (1 + ud * ud).sqrt()
            delta = 1 / (gd * (gd + ud))
            expected = -1 / (D.from_float(C_MMNS) * delta)
            error = abs(D.from_float(point.time_ns) - expected)
            assert error <= D.from_float(point.time_error_bound_ns)
        assert_norm_close(point.separation_vector_mm, [float(1 / delta), 0, 0], 2e-15)


def test_accelerated_local_jet_matches_resolved_beta_path():
    # Nonzero acceleration and jerk exercise deficit derivatives; the
    # moderate speed keeps the original polynomial a reliable reference.
    coefficients = np.zeros((6, 3))
    coefficients[1] = [0.8, 0.1, 0.2]
    coefficients[2] = [0.003, -0.002, 0.001]
    coefficients[3] = [0.0004, 0.0002, -0.0001]
    beta = coefficients[1]
    u = beta / np.sqrt(1 - beta @ beta)
    kw = dict(
        observer_time_ns=2 / C_MMNS,
        observer_position_mm=np.array([1.0, 1.0, 1.0]),
        charge_native=1.0,
        segment_start_time_ns=0.0,
        segment_duration_ns=1 / C_MMNS,
        position_coefficients_mm=coefficients,
        retarded_time_ns=0.0,
    )
    for kernel, fields in (
        (quintic_charge_potential_jet_native, ("partial_a", "partial2_a")),
        (quintic_charge_response_jet_native, ("partial_a", "partial_f")),
    ):
        old, new = kernel(**kw), kernel(**kw, source_proper_velocity=u)
        for field in fields:
            assert_norm_close(getattr(new, field), getattr(old, field), 4e-14)


def test_zero_proper_velocity_and_rotated_geometry():
    for u in (np.zeros(3), np.asarray([0.3, -0.8, 0.7])):
        args = (1.0, np.asarray([1.5, -0.2, 0.8]), u, np.zeros(3), np.zeros(3))
        python = precise_charge_jets(*args)
        compiled = precise_charge_jets_strict_serial(*args)
        for actual, expected in zip(compiled[:-1], python[:-1]):
            assert_norm_close(actual, expected, 5e-15)


@pytest.mark.parametrize("scale", [10.0**power for power in range(1, 13)])
@pytest.mark.parametrize("side", [1, -1])
def test_rotated_parallel_ray_against_decimal(scale, side):
    from decimal import Decimal as D, localcontext

    # Exactly parallel binary64 vectors must remain parallel after rotation.
    # An ordinary rounded frame dot product invents transverse offsets that
    # can exceed the physical beaming angle by many orders of magnitude.
    u = scale * np.asarray([0.37, 0.51, 0.73])
    with localcontext() as context:
        context.prec = 90
        ud = [D.from_float(float(component)) for component in u]
        radius = sum(component * component for component in ud).sqrt()
        gamma = (1 + radius * radius).sqrt()
        deficit = 1 / (gamma * (gamma + radius))
        kappa = deficit if side == 1 else 2 - deficit
        phi = 1 / (kappa * radius)
        electric_scale = side * deficit * (2 - deficit) / (kappa**2 * radius**2)
        expected_electric = [float(electric_scale * v / radius) for v in ud]
        expected_potential = [float(phi)] + [float(phi * v / gamma) for v in ud]
    args = (1.0, side * u, u, np.zeros(3), np.zeros(3))
    python = precise_charge_jets(*args)
    compiled = precise_charge_jets_strict_serial(*args)
    assert_norm_close(python[0][:, 0], expected_potential)
    assert_norm_close(python[1][:, 0], expected_electric)
    np.testing.assert_array_equal(python[2][:, 0], np.zeros(3))
    for actual, expected in zip(compiled[:-1], python[:-1]):
        assert_norm_close(actual, expected, 5e-15)


@pytest.mark.parametrize("scale", [10.0**power for power in range(1, 13)])
def test_rotated_ballistic_boundary_keeps_resolved_separation(scale):
    from decimal import Decimal as D, localcontext

    from core.integration_runner import _build_inertial_coasting_history
    from core.retarded_fields import (
        ObserverEvent,
        evaluate_retarded_charge_field_native,
        evaluate_retarded_charge_response_gradient_native,
    )
    from tests.unit.test_inertial_prehistory import _state

    state = _state(position_mm=(0, 0, 0), beta=(0.9, 0, 0), source_charge=1.0)
    u = scale * np.asarray([0.37, 0.51, 0.73])
    for axis, component in zip("xyz", u):
        state[f"P{axis}"][:] = component * state["m"] * C_MMNS
    u = np.asarray(
        [float(state[f"P{axis}"][0] / (state["m"][0] * C_MMNS)) for axis in "xyz"]
    )
    gamma = np.sqrt(1 + u @ u)
    state["gamma"][:] = gamma
    state["Pt"][:] = gamma * state["m"] * C_MMNS
    for axis, component in zip("xyz", u):
        state[f"b{axis}"][:] = component / gamma
    with localcontext() as context:
        context.prec = 90
        ud = [D.from_float(float(component)) for component in u]
        radius = sum(v * v for v in ud).sqrt()
        invariant = 1 / (1 + radius * radius)
        expected = [float(invariant * v / radius**3) for v in ud]
        expected_phi = float(1 / radius)
    point = ballistic_retarded_point(
        observer_time_ns=0,
        observer_position_mm=u,
        anchor_time_ns=0,
        anchor_position_mm=(0, 0, 0),
        source_proper_velocity=u,
    )
    frame = resolved_separation_frame(
        point.longitudinal_separation_mm, point.transverse_separation_vector_mm, u
    )
    args = (1.0, point.separation_vector_mm, u, np.zeros(3), np.zeros(3), frame)
    python = precise_charge_jets(*args)
    compiled = precise_charge_jets_strict_serial(*args)
    assert_norm_close(python[1][:, 0], expected)
    for actual, reference in zip(compiled[:-1], python[:-1]):
        assert_norm_close(actual, reference, 5e-15)

    history = _build_inertial_coasting_history(
        state, 4 * gamma**2 * np.linalg.norm(u) / C_MMNS
    )
    event = ObserverEvent(0.0, tuple(u))
    for backend in ("python", "numba_full_strict_serial"):
        field = evaluate_retarded_charge_field_native(history, event, backend=backend)
        assert_norm_close(field.electric_field_native, expected)
        assert field.four_potential[0] == pytest.approx(expected_phi, rel=2e-13)
    response = evaluate_retarded_charge_response_gradient_native(history, event)
    assert not response.fallback_used
    assert_norm_close(-response.antisymmetric_response[:3], expected)


@pytest.mark.parametrize("gamma", [10.0, 1e6, 1e12])
def test_inertial_boundary_survives_builder_trial_and_backend_selection(
    gamma, tmp_path
):
    from core.integration_runner import _build_inertial_coasting_history
    from core.retarded_fields import (
        ObserverEvent,
        evaluate_retarded_charge_field_native,
        evaluate_retarded_charge_response_gradient_native,
        _prepare_history,
    )
    from core.types import GrowableTrajectoryBuilder, TrialTrajectoryHistory
    from tests.unit.test_inertial_prehistory import _state

    state = _state(position_mm=(0, 0, 0), beta=(0.9, 0, 0), source_charge=1.0)
    u = np.sqrt(gamma * gamma - 1)
    state["gamma"][:] = gamma
    state["bx"][:] = u / gamma
    state["Px"][:] = u * state["m"] * C_MMNS
    state["Pt"][:] = gamma * state["m"] * C_MMNS
    history = _build_inertial_coasting_history(state, 4 * gamma * gamma / C_MMNS)
    builder = GrowableTrajectoryBuilder(len(history) + 2, 1, magnetic_dipole=True)
    for step in history:
        builder.append_step(step)
    arrays = builder.build_current()
    tail = {key: value.copy() for key, value in history[-1].items()}
    tail["bx"][:] = min(float(tail["bx"][0]), np.nextafter(1.0, 0.0))
    tail["t"][:] = 1.0
    tail["x"][:] = C_MMNS * state["bx"][0]
    trial = TrialTrajectoryHistory(arrays, (tail,))
    event = ObserverEvent(0.0, (1.0, 0, 0))
    for current in (history, arrays, trial):
        source = _prepare_history(current, ()).sources[0]
        assert source.inertial_boundary is not None
        for backend in ("python", "numba_full_strict_serial"):
            field = evaluate_retarded_charge_field_native(
                current, event, backend=backend
            )
            assert_norm_close(field.electric_field_native, [1 / gamma**2, 0, 0], 3e-15)
            assert field.four_potential[0] == pytest.approx(1.0, rel=3e-15)
            assert field.retarded_time_error_bound_ns is not None
            assert field.retarded_time_error_bound_ns[0] > 0.0
        response = evaluate_retarded_charge_response_gradient_native(current, event)
        assert not response.fallback_used
        assert_norm_close(response.partial_a[:, 0], [u / gamma, -1, 0, 0], 4e-15)

    from core.integration_checkpoint import IntegrationCheckpointStore
    from core.types import TrajectoryBuilder

    checkpoint_options = dict(
        compatibility_payload={"purpose": "precise-boundary"},
        total_steps=len(history),
        requested_steps=1,
        active_start=len(history) - 1,
        interval_steps=1,
        interval_seconds=0.0,
    )
    store = IntegrationCheckpointStore(
        tmp_path / "checkpoint", resume=False, **checkpoint_options
    )
    store.write(step_index=len(history) - 1, rider=arrays, driver=arrays, loop_state={})
    restored_builder = TrajectoryBuilder(len(history), 1, magnetic_dipole=True)
    resumed = IntegrationCheckpointStore(
        tmp_path / "checkpoint", resume=True, **checkpoint_options
    )
    resumed.restore_builder(restored_builder, "rider")
    restored = restored_builder.build()
    restored_field = evaluate_retarded_charge_field_native(restored, event)
    assert_norm_close(restored_field.electric_field_native, [1 / gamma**2, 0, 0], 3e-15)

    # A completely lost source has no analytic alive interval to evaluate.
    dead = [{key: value.copy() for key, value in state.items()} for state in history]
    for state in dead:
        state["_dead_particles"] = np.ones(1, dtype=bool)
    lost = evaluate_retarded_charge_field_native(dead, event)
    assert not np.any(lost.valid_sources)
    np.testing.assert_array_equal(lost.electric_field_native, np.zeros(3))

    from dataclasses import replace
    from core.integration_checkpoint import _particle_constant
    from core.types import INERTIAL_CHARGE_BOUNDARY_FIELDS

    # Older manual SOA constructors have empty optional boundary constants.
    manual = replace(
        arrays, **{key: np.zeros(0) for key in INERTIAL_CHARGE_BOUNDARY_FIELDS}
    )
    for key in INERTIAL_CHARGE_BOUNDARY_FIELDS:
        np.testing.assert_array_equal(_particle_constant(manual, key), np.zeros(1))


def test_analytic_light_cone_callable_and_root_bound():
    from decimal import Decimal as D, localcontext
    from core.analytic_charge_root import (
        AnalyticLightConeState,
        solve_analytic_charge_light_cone,
    )
    from core.precise_charge import precise_speed

    for gamma in (10.0, 1e6, 1e12):
        u = np.array([np.sqrt(gamma * gamma - 1), 0, 0])
        delta, _ = precise_speed(u)

        def source(ct):
            return AnalyticLightConeState(delta * ct, np.zeros(3), u)

        root = solve_analytic_charge_light_cone(
            source_state=source,
            observer_coordinate_mm=0.0,
            observer_w_mm=-1.0,
            observer_transverse_position_mm=(0, 0, 0),
            axis=(1, 0, 0),
            bracket_mm=(-2 / delta, 0.0),
            minimum_kappa=delta,
        )
        with localcontext() as context:
            context.prec = 80
            expected = -1 / D.from_float(delta)
            assert abs(
                D.from_float(root.source_coordinate_mm) - expected
            ) <= D.from_float(root.coordinate_error_bound_mm)


def test_accelerated_analytic_light_cone_uses_coherent_coordinates():
    from core.analytic_charge_root import (
        AnalyticLightConeState,
        solve_analytic_charge_light_cone,
    )

    def source(ct):
        beta = 0.3 + 0.1 * ct
        return AnalyticLightConeState(
            0.7 * ct - 0.05 * ct * ct,
            np.zeros(3),
            np.array([beta / np.sqrt(1 - beta * beta), 0, 0]),
        )

    root = solve_analytic_charge_light_cone(
        source_state=source,
        observer_coordinate_mm=1.0,
        observer_w_mm=0.0,
        observer_transverse_position_mm=(0, 0.2, 0),
        axis=(1, 0, 0),
        bracket_mm=(-2.0, 0.5),
        minimum_kappa=0.5,
    )
    ct = root.source_coordinate_mm
    direct = 1 - ct - np.hypot(1 - (0.3 * ct + 0.05 * ct * ct), 0.2)
    assert abs(direct) <= root.residual_error_bound_mm + 1e-15
    assert root.coordinate_error_bound_mm < 1e-11
