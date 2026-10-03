"""Regression checks for source-order noise in the exact Taylor correction."""

from dataclasses import replace
import itertools

import numpy as np
import pytest

import core.retarded_fields as fields
from core.exact_source_cloud import exact_cloud_history
from tests.unit.test_exact_retarded_charge_numba import _uniform_charge_history
from tests.unit.test_exact_source_cloud import cloud_config


def test_compensated_source_reduction_retains_cancelled_low_order_terms():
    for values in itertools.permutations((1e16, 1.0, -1e16)):
        terms = np.asarray(values)[:, None]
        np.testing.assert_array_equal(fields._sum_source_terms(terms), [1.0])
    np.testing.assert_array_equal(
        fields._sum_source_terms(np.zeros((0, 4))), np.zeros(4)
    )


@pytest.mark.parametrize("cloud", [False, True])
@pytest.mark.parametrize("backend", ["python", "numba_full_strict_serial"])
def test_exact_charge_values_and_jets_are_source_order_invariant(
    monkeypatch, cloud, backend
):
    history = _uniform_charge_history()
    if cloud:
        history = exact_cloud_history(history, cloud_config())
    event = fields.ObserverEvent(-1e-5, (0.9, 1.1, -0.5))
    reference = fields.evaluate_retarded_charge_field_gradient_native(
        history, event, backend=backend
    )
    center = fields.evaluate_retarded_charge_field_native(
        history, event, backend=backend
    )
    original = fields._prepare_history

    def reversed_sources(*args, **kwargs):
        prepared = original(*args, **kwargs)
        return replace(prepared, sources=dict(reversed(list(prepared.sources.items()))))

    monkeypatch.setattr(fields, "_prepare_history", reversed_sources)
    candidate = fields.evaluate_retarded_charge_field_gradient_native(
        history, event, backend=backend
    )
    candidate_center = fields.evaluate_retarded_charge_field_native(
        history, event, backend=backend
    )
    assert candidate.stencil_step_mm == 0.0
    for name in ("partial_f", "partial_a", "stencil_retarded_time_ns"):
        np.testing.assert_array_equal(
            getattr(reference, name), getattr(candidate, name)
        )
    for name in ("field_tensor", "four_potential", "retarded_time_ns", "valid_sources"):
        np.testing.assert_array_equal(
            getattr(reference.field, name), getattr(candidate.field, name)
        )
        np.testing.assert_array_equal(
            getattr(center, name), getattr(candidate_center, name)
        )


def test_smooth_stencil_converges_quadratically_to_analytic_jets():
    history = _uniform_charge_history()
    event = fields.ObserverEvent(-1e-5, (0.9, 1.1, -0.5))
    analytical = fields.evaluate_retarded_charge_field_gradient_native(history, event)
    errors = []
    for spacing in (2e-3, 1e-3, 5e-4):
        stencil = fields.evaluate_retarded_charge_field_gradient_stencil_native(
            history, event, relative_step=spacing
        )
        errors.append(
            [
                np.max(np.abs(stencil.partial_f - analytical.partial_f)),
                np.max(np.abs(stencil.partial_a - analytical.partial_a)),
            ]
        )
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    np.testing.assert_allclose(ratios, 4.0, rtol=0.02)
