"""Analytical checks for the charge potential Taylor-jet oracle."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from core.charge_potential_jet import quintic_charge_potential_jet_native
from core.charge_potential_jet_numba import quintic_charge_potential_jet_strict_serial
from core.constants import C_MMNS


def test_static_charge_potential_gradient_and_hessian_are_analytic() -> None:
    charge = -1.7
    observer = np.asarray((3.0, -4.0, 5.0))
    radius = float(np.linalg.norm(observer))
    duration = 20.0
    coefficients = np.zeros((6, 3))
    result = quintic_charge_potential_jet_native(
        observer_time_ns=10.0,
        observer_position_mm=observer,
        charge_native=charge,
        segment_start_time_ns=0.0,
        segment_duration_ns=duration,
        position_coefficients_mm=coefficients,
        retarded_time_ns=10.0 - radius / C_MMNS,
    )

    expected_gradient = -charge * observer / radius**3
    expected_hessian = charge * (
        3.0 * np.outer(observer, observer) / radius**5 - np.eye(3) / radius**3
    )
    np.testing.assert_allclose(
        result.four_potential, (charge / radius, 0, 0, 0), rtol=2e-15
    )
    np.testing.assert_allclose(result.partial_a[0], 0.0, atol=1e-16)
    np.testing.assert_allclose(result.partial_a[1:, 0], expected_gradient, rtol=2e-15)
    np.testing.assert_allclose(result.partial_a[:, 1:], 0.0, atol=1e-16)
    np.testing.assert_allclose(
        result.partial2_a[1:, 1:, 0], expected_hessian, rtol=8e-15
    )
    np.testing.assert_allclose(result.partial2_a[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(result.partial2_a[:, 0], 0.0, atol=1e-15)
    assert result.light_cone_jet_residual < 2.0e-13


def test_retarded_coordinate_gradient_matches_static_light_cone() -> None:
    observer = np.asarray((2.0, 3.0, 6.0))
    radius = float(np.linalg.norm(observer))
    coefficients = np.zeros((6, 3))
    result = quintic_charge_potential_jet_native(
        observer_time_ns=4.0,
        observer_position_mm=observer,
        charge_native=1.0,
        segment_start_time_ns=0.0,
        segment_duration_ns=10.0,
        position_coefficients_mm=coefficients,
        retarded_time_ns=4.0 - radius / C_MMNS,
    )
    expected = np.concatenate(([1.0], -observer / radius))
    np.testing.assert_allclose(
        result.retarded_coordinate_gradient, expected, rtol=2e-15
    )


def test_strict_compiled_jet_matches_python_oracle() -> None:
    coefficients = np.asarray(
        [
            (0.2, -0.1, 0.3),
            (0.1, 0.04, -0.02),
            (0.003, -0.002, 0.001),
            (-0.0002, 0.0001, 0.0003),
            (0.00001, 0.00002, -0.00001),
            (-0.000001, 0.000001, 0.000002),
        ]
    )
    kwargs = dict(
        observer_time_ns=0.012,
        observer_position_mm=(2.4, 3.1, -1.7),
        charge_native=-1.37,
        segment_start_time_ns=-0.004,
        segment_duration_ns=0.004,
        position_coefficients_mm=coefficients,
        retarded_time_ns=-0.002,
    )
    python_result = quintic_charge_potential_jet_native(**kwargs)
    compiled = quintic_charge_potential_jet_strict_serial(
        kwargs["observer_time_ns"],
        np.asarray(kwargs["observer_position_mm"]),
        kwargs["charge_native"],
        kwargs["segment_start_time_ns"],
        kwargs["segment_duration_ns"],
        coefficients,
        kwargs["retarded_time_ns"],
    )
    np.testing.assert_allclose(compiled[0], python_result.four_potential, rtol=2e-15)
    np.testing.assert_allclose(
        compiled[1], python_result.partial_a, rtol=3e-15, atol=1e-14
    )
    np.testing.assert_allclose(
        compiled[2], python_result.partial2_a, rtol=8e-15, atol=1e-12
    )
