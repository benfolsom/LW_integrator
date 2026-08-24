from __future__ import annotations

import numpy as np
import pytest

import core.retarded_fields as retarded_fields
from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.retarded_fields import (
    ObserverEvent,
    RetardedHistoryError,
    evaluate_retarded_charge_field_gradient_si,
    evaluate_retarded_charge_field_si,
    lienard_wiechert_charge_field_si,
)
from core.rfs import SPEED_OF_LIGHT_M_S, electromagnetic_field_tensor_si


def _source_history(
    *,
    times_ns: np.ndarray,
    position_mm: np.ndarray,
    beta: np.ndarray,
    beta_dot_s: np.ndarray | None = None,
    charge_native: float = ELEMENTARY_CHARGE,
) -> list[dict[str, np.ndarray]]:
    count = int(times_ns.size)
    if position_mm.shape != (count, 3) or beta.shape != (count, 3):
        raise ValueError("test worldline arrays have inconsistent shapes")
    if beta_dot_s is None:
        beta_dot_s = np.zeros_like(beta)
    bdot_native = beta_dot_s / (C_MMNS * 1.0e9)
    result = []
    for step in range(count):
        result.append(
            {
                "t": np.array([times_ns[step]], dtype=float),
                "x": np.array([position_mm[step, 0]], dtype=float),
                "y": np.array([position_mm[step, 1]], dtype=float),
                "z": np.array([position_mm[step, 2]], dtype=float),
                "bx": np.array([beta[step, 0]], dtype=float),
                "by": np.array([beta[step, 1]], dtype=float),
                "bz": np.array([beta[step, 2]], dtype=float),
                "bdotx": np.array([bdot_native[step, 0]], dtype=float),
                "bdoty": np.array([bdot_native[step, 1]], dtype=float),
                "bdotz": np.array([bdot_native[step, 2]], dtype=float),
                "q": np.array([charge_native], dtype=float),
                "q_source": np.array([charge_native], dtype=float),
                "_dead_particles": np.array([False]),
            }
        )
    return result


def _stationary_history() -> list[dict[str, np.ndarray]]:
    times_ns = np.linspace(-0.02, 0.002, 23)
    return _source_history(
        times_ns=times_ns,
        position_mm=np.zeros((times_ns.size, 3)),
        beta=np.zeros((times_ns.size, 3)),
    )


def test_point_charge_kernel_reduces_to_coulomb_field() -> None:
    radius_m = 2.5e-3
    electric, magnetic = lienard_wiechert_charge_field_si(
        charge_coulomb=1.602_176_634e-19,
        separation_vector_m=(radius_m, 0.0, 0.0),
        source_beta=(0.0, 0.0, 0.0),
        source_beta_dot_s=(0.0, 0.0, 0.0),
    )

    expected = 8.987_551_792_3e9 * 1.602_176_634e-19 / radius_m**2
    np.testing.assert_allclose(electric, (expected, 0.0, 0.0), rtol=1.0e-15)
    np.testing.assert_allclose(magnetic, 0.0, atol=0.0)


def test_stationary_light_cone_root_and_field_are_exact() -> None:
    radius_mm = 1.0
    field = evaluate_retarded_charge_field_si(
        _stationary_history(),
        ObserverEvent(time_ns=0.0, position_mm=(radius_mm, 0.0, 0.0)),
    )

    expected_time_ns = -radius_mm / C_MMNS
    expected_electric = (
        8.987_551_792_3e9 * 1.602_176_634e-19 / (radius_mm * 1.0e-3) ** 2
    )
    assert field.retarded_time_ns[0] == pytest.approx(expected_time_ns, abs=1.0e-16)
    assert abs(field.light_cone_residual_m[0]) <= 1.0e-18
    np.testing.assert_allclose(
        field.electric_field_v_m,
        (expected_electric, 0.0, 0.0),
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(field.magnetic_field_t, 0.0, atol=0.0)


def test_uniform_motion_light_cone_root_matches_analytic_solution() -> None:
    beta_x = 0.2
    times_ns = np.linspace(-0.02, 0.002, 45)
    positions = np.zeros((times_ns.size, 3))
    positions[:, 0] = beta_x * C_MMNS * times_ns
    betas = np.zeros_like(positions)
    betas[:, 0] = beta_x
    observer_x_mm = 1.0
    history = _source_history(
        times_ns=times_ns,
        position_mm=positions,
        beta=betas,
    )

    field = evaluate_retarded_charge_field_si(
        history,
        ObserverEvent(time_ns=0.0, position_mm=(observer_x_mm, 0.0, 0.0)),
    )

    expected_time_ns = -observer_x_mm / (C_MMNS * (1.0 - beta_x))
    assert field.retarded_time_ns[0] == pytest.approx(expected_time_ns, abs=2.0e-16)
    assert abs(field.light_cone_residual_m[0]) <= 1.0e-18


def test_complete_gradient_matches_static_coulomb_jacobian() -> None:
    radius_m = 1.0e-3
    result = evaluate_retarded_charge_field_gradient_si(
        _stationary_history(),
        ObserverEvent(time_ns=0.0, position_mm=(radius_m * 1.0e3, 0.0, 0.0)),
        relative_step=2.0e-5,
    )

    coefficient = 8.987_551_792_3e9 * 1.602_176_634e-19 / radius_m**3
    expected_electric_gradient = coefficient * np.diag((-2.0, 1.0, 1.0))
    recovered_electric_gradient = np.empty((3, 3), dtype=float)
    for coordinate in range(3):
        recovered_electric_gradient[:, coordinate] = (
            -SPEED_OF_LIGHT_M_S * result.partial_f[coordinate + 1, 0, 1:4]
        )

    np.testing.assert_allclose(
        recovered_electric_gradient,
        expected_electric_gradient,
        rtol=2.0e-8,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(result.partial_f[0], 0.0, atol=1.0e-9)
    np.testing.assert_allclose(
        result.partial_f + np.swapaxes(result.partial_f, 1, 2),
        0.0,
        atol=1.0e-12,
    )


def test_gradient_stencil_resolves_a_new_retarded_event_at_every_point() -> None:
    result = evaluate_retarded_charge_field_gradient_si(
        _stationary_history(),
        ObserverEvent(time_ns=0.0, position_mm=(1.0, 0.0, 0.0)),
    )

    # Both coordinate-time and radial-space displacements change the retarded
    # source time. A frozen retarded source sample would fail these assertions.
    assert (
        result.stencil_retarded_time_ns[0, 0, 0]
        != result.stencil_retarded_time_ns[0, 1, 0]
    )
    assert (
        result.stencil_retarded_time_ns[1, 0, 0]
        != result.stencil_retarded_time_ns[1, 1, 0]
    )


def test_gradient_extracts_and_prepares_shared_history_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = retarded_fields._extract_history
    extraction_count = 0

    def counted_extract(history):
        nonlocal extraction_count
        extraction_count += 1
        return original(history)

    monkeypatch.setattr(retarded_fields, "_extract_history", counted_extract)

    evaluate_retarded_charge_field_gradient_si(
        _stationary_history(),
        ObserverEvent(time_ns=0.0, position_mm=(1.0, 0.0, 0.0)),
    )

    assert extraction_count == 1


def test_uniform_motion_complete_gradient_matches_heaviside_field() -> None:
    beta = np.array((0.21, -0.08, 0.04))
    times_ns = np.linspace(-0.03, 0.003, 67)
    positions_mm = times_ns[:, None] * C_MMNS * beta[None, :]
    betas = np.broadcast_to(beta, positions_mm.shape).copy()
    history = _source_history(
        times_ns=times_ns,
        position_mm=positions_mm,
        beta=betas,
    )
    instantaneous_separation_m = np.array((0.8e-3, 1.1e-3, -0.4e-3))
    result = evaluate_retarded_charge_field_gradient_si(
        history,
        ObserverEvent(
            time_ns=0.0,
            position_mm=tuple(instantaneous_separation_m * 1.0e3),
        ),
        relative_step=1.0e-5,
    )

    beta_squared = float(beta @ beta)
    matrix_a = (1.0 - beta_squared) * np.eye(3) + np.outer(beta, beta)
    denominator = float(
        instantaneous_separation_m @ matrix_a @ instantaneous_separation_m
    )
    coefficient = 8.987_551_792_3e9 * 1.602_176_634e-19 * (1.0 - beta_squared)
    expected_electric = coefficient * instantaneous_separation_m * denominator ** (-1.5)
    expected_magnetic = np.cross(beta, expected_electric) / SPEED_OF_LIGHT_M_S
    electric_gradient = np.empty((3, 3), dtype=float)
    magnetic_gradient = np.empty((3, 3), dtype=float)
    a_on_r = matrix_a @ instantaneous_separation_m
    for coordinate in range(3):
        electric_gradient[:, coordinate] = coefficient * (
            np.eye(3)[:, coordinate] * denominator ** (-1.5)
            - 3.0
            * instantaneous_separation_m
            * a_on_r[coordinate]
            * denominator ** (-2.5)
        )
        magnetic_gradient[:, coordinate] = (
            np.cross(beta, electric_gradient[:, coordinate]) / SPEED_OF_LIGHT_M_S
        )
    expected_partial = np.zeros((4, 4, 4), dtype=float)
    expected_partial[0] = electromagnetic_field_tensor_si(
        tuple(-(electric_gradient @ beta)),
        tuple(-(magnetic_gradient @ beta)),
    )
    for coordinate in range(3):
        expected_partial[coordinate + 1] = electromagnetic_field_tensor_si(
            tuple(electric_gradient[:, coordinate]),
            tuple(magnetic_gradient[:, coordinate]),
        )

    np.testing.assert_allclose(
        result.field.electric_field_v_m,
        expected_electric,
        rtol=3.0e-12,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        result.field.magnetic_field_t,
        expected_magnetic,
        rtol=3.0e-12,
        atol=1.0e-24,
    )
    np.testing.assert_allclose(
        result.partial_f,
        expected_partial,
        rtol=2.0e-7,
        atol=1.0e-15,
    )
    medium = evaluate_retarded_charge_field_gradient_si(
        history,
        ObserverEvent(
            time_ns=0.0,
            position_mm=tuple(instantaneous_separation_m * 1.0e3),
        ),
        relative_step=2.0e-5,
    )
    coarse = evaluate_retarded_charge_field_gradient_si(
        history,
        ObserverEvent(
            time_ns=0.0,
            position_mm=tuple(instantaneous_separation_m * 1.0e3),
        ),
        relative_step=4.0e-5,
    )
    fine_error = float(np.linalg.norm(result.partial_f - expected_partial))
    medium_error = float(np.linalg.norm(medium.partial_f - expected_partial))
    coarse_error = float(np.linalg.norm(coarse.partial_f - expected_partial))
    assert coarse_error / medium_error > 3.5
    assert medium_error / fine_error > 3.5


def test_missing_retarded_history_is_explicit() -> None:
    times_ns = np.array((-0.001, 0.0))
    history = _source_history(
        times_ns=times_ns,
        position_mm=np.zeros((2, 3)),
        beta=np.zeros((2, 3)),
    )

    with pytest.raises(RetardedHistoryError, match="does not bracket"):
        evaluate_retarded_charge_field_si(
            history,
            ObserverEvent(time_ns=0.0, position_mm=(10.0, 0.0, 0.0)),
        )

    partial = evaluate_retarded_charge_field_si(
        history,
        ObserverEvent(time_ns=0.0, position_mm=(10.0, 0.0, 0.0)),
        require_complete_history=False,
    )
    assert not partial.valid_sources.any()
    np.testing.assert_allclose(partial.field_tensor, 0.0, atol=0.0)


def test_lost_source_uses_alive_prefix_despite_frozen_dead_timestamps() -> None:
    times_ns = np.array((-0.010, -0.005, -0.002, -0.002))
    history = _source_history(
        times_ns=times_ns,
        position_mm=np.zeros((times_ns.size, 3)),
        beta=np.zeros((times_ns.size, 3)),
    )
    history[-1]["_dead_particles"][0] = True

    field = evaluate_retarded_charge_field_si(
        history,
        ObserverEvent(time_ns=0.0, position_mm=(1.0, 0.0, 0.0)),
    )

    assert field.valid_sources[0]
    assert field.retarded_time_ns[0] == pytest.approx(-1.0 / C_MMNS, abs=1.0e-16)
    assert field.electric_field_v_m[0] > 0.0


def test_lost_source_is_absent_after_its_alive_history_ends() -> None:
    times_ns = np.array((-0.010, -0.008, -0.008))
    history = _source_history(
        times_ns=times_ns,
        position_mm=np.zeros((times_ns.size, 3)),
        beta=np.zeros((times_ns.size, 3)),
    )
    history[-1]["_dead_particles"][0] = True

    field = evaluate_retarded_charge_field_si(
        history,
        ObserverEvent(time_ns=0.0, position_mm=(1.0, 0.0, 0.0)),
    )

    assert not field.valid_sources[0]
    np.testing.assert_array_equal(field.field_tensor, 0.0)
