"""Analytical tests for the stable charge-response jet."""

from __future__ import annotations

import numpy as np

from core.charge_response_jet import quintic_charge_response_jet_native
from core.charge_response_jet_numba import quintic_charge_response_jet_strict_serial
from core.constants import C_MMNS
from core.rfs import electromagnetic_field_tensor_native


def test_static_charge_response_and_gradient_are_coulomb_analytic() -> None:
    charge = -1.7
    observer = np.asarray((3.0, -4.0, 5.0))
    radius = float(np.linalg.norm(observer))
    coefficients = np.zeros((6, 3))
    result = quintic_charge_response_jet_native(
        observer_time_ns=10.0,
        observer_position_mm=observer,
        charge_native=charge,
        segment_start_time_ns=0.0,
        segment_duration_ns=20.0,
        position_coefficients_mm=coefficients,
        retarded_time_ns=10.0 - radius / C_MMNS,
    )
    electric = charge * observer / radius**3
    expected_field = electromagnetic_field_tensor_native(electric, np.zeros(3))
    np.testing.assert_allclose(
        result.field_tensor, expected_field, rtol=3e-15, atol=2e-17
    )

    electric_gradient = charge * (
        np.eye(3) / radius**3 - 3.0 * np.outer(observer, observer) / radius**5
    )
    expected_partial_f = np.zeros((4, 4, 4))
    for derivative in range(3):
        expected_partial_f[derivative + 1] = electromagnetic_field_tensor_native(
            electric_gradient[derivative], np.zeros(3)
        )
    np.testing.assert_allclose(result.partial_f[0], 0.0, atol=2e-16)
    np.testing.assert_allclose(
        result.partial_f[1:], expected_partial_f[1:], rtol=8e-15, atol=2e-17
    )


def test_compiled_charge_response_matches_python_oracle() -> None:
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
    python_result = quintic_charge_response_jet_native(**kwargs)
    compiled = quintic_charge_response_jet_strict_serial(
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
        compiled[1], python_result.field_tensor, rtol=3e-15, atol=1e-16
    )
    np.testing.assert_allclose(
        compiled[2], python_result.partial_f, rtol=8e-15, atol=1e-12
    )
