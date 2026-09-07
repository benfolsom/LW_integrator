"""Avoid a numerical force-method switch only across proven stationary segments."""

from dataclasses import replace

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.retarded_fields import (
    ObserverEvent,
    _prepare_history,
    _analytical_segment_margin_ratio,
    _stationary_span_margin_ratio,
    evaluate_retarded_charge_response_gradient_native,
)
from tests.unit.test_analytic_charge_response_provider import _uniform_history


def source():
    history = _uniform_history(
        times_ns=np.array([-0.02, -0.01, 0, 0.01]), beta=np.zeros(3)
    )
    return history, _prepare_history(history, ()).sources[0]


def test_stationary_boundary_neighbourhood_keeps_exact_analytical_response():
    history, prepared = source()
    root = -1e-6
    step = 1e-3
    margin, reason = _analytical_segment_margin_ratio(
        source=prepared,
        segment_index=1,
        retarded_time_ns=root,
        observer_stencil_step_mm=step,
    )
    assert reason is None and margin > 1
    response = evaluate_retarded_charge_response_gradient_native(
        history, ObserverEvent(1 / c + root, (1, 0, 0)), relative_step=step
    )
    assert not response.fallback_used
    # q=-1.3 at the origin gives exact Coulomb derivatives at (1,0,0).
    np.testing.assert_allclose(
        -response.antisymmetric_response[:3], [-1.3, 0, 0], rtol=1e-14, atol=0
    )
    assert response.minimum_segment_margin_ratio > 1


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
def test_even_tiny_nonconstant_neighbour_prevents_extension(order):
    _, prepared = source()
    coefficients = prepared.position_coefficients_mm.copy()
    coefficients[2, order, 0] = 1e-100
    changed = replace(prepared, position_coefficients_mm=coefficients)
    margin = _stationary_span_margin_ratio(changed, 1, -1e-6, 1e-3)
    assert margin is not None and margin < 1


def test_certificate_cannot_extend_beyond_available_history():
    _, prepared = source()
    margin = _stationary_span_margin_ratio(prepared, 2, 0.01 - 1e-6, 1e-3)
    assert margin is not None and margin < 1


def test_certificate_extends_only_through_needed_identical_segments():
    _, prepared = source()
    # The displacement reaches two knots. Both sides must remain stationary.
    assert _stationary_span_margin_ratio(prepared, 1, -0.005, 2.0) > 1
    coefficients = prepared.position_coefficients_mm.copy()
    coefficients[0, 1, 1] = 1e-100
    changed = replace(prepared, position_coefficients_mm=coefficients)
    assert _stationary_span_margin_ratio(changed, 1, -0.005, 2.0) < 1


def test_exact_knot_keeps_existing_directional_kernel_guard():
    _, prepared = source()
    margin, reason = _analytical_segment_margin_ratio(
        source=prepared,
        segment_index=2,
        retarded_time_ns=0,
        observer_stencil_step_mm=1e-3,
    )
    assert margin == 0 and reason == "retarded_root_is_on_segment_boundary"
