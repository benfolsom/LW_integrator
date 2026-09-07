"""Directional differentiation of the current polynomial, without a new model."""

from dataclasses import replace

import numpy as np
import pytest

from core.antisymmetric_response_rfs import pack_partial_antisymmetric_response_native
from core.constants import C_MMNS
from core.dipole_hertz_jet import (
    polynomial_dipole_hertz_response_jet_native as evaluate,
)
from core.causal_local_source_history import (
    CausalLocalDipoleSource,
    CausalLocalDipoleSourceCollection,
    CausalLocalSourceHistoryUnavailableError,
)
from core.causal_local_source_jet import (
    LocalSourceJetFitConfig,
    LocalSourceJetModelSpreadConfig,
    LocalSourceJetMultiScaleConfig,
    CausalLocalSourceJetModelSpreadError,
    _response_model_spread,
    evaluate_causal_local_source_jet_native as evaluate_history,
    evaluate_causal_local_source_jet_collection_native as evaluate_collection,
    evaluate_causal_local_source_jet_collection_multiscale_native as evaluate_multiscale,
)
from core.retarded_fields import ObserverEvent
from tests.unit.test_causal_local_source_jet import _analytic_history, _past_scale


DIRECTION = C_MMNS * np.array([1.7, 0.4, -0.3, 0.2])
RATE = "partial_antisymmetric_response_along_velocity"


def polynomial_case(beta, chart=True):
    position = np.zeros((6, 3))
    position[0] = [0.1, -0.2, 0.05]
    position[1] = C_MMNS * 0.1 * np.array([beta, 0.0, 0.0])
    position[2] = [0.001, 0.002, -0.001]
    position[5] = [1e-5, -2e-5, 1e-5]
    root = 0.045
    point = np.polynomial.polynomial.polyval(root / 0.1, position)
    displacement = np.array([2.0, 1.0, -0.5])
    kwargs = dict(
        observer_time_ns=root + np.linalg.norm(displacement) / C_MMNS,
        observer_position_mm=point + displacement,
        magnetic_moment_native=-0.7,
        segment_start_time_ns=0.0,
        segment_duration_ns=0.1,
        position_coefficients_mm=position,
        rest_spin_coefficients=(
            None if chart else np.array([[0.2, -0.3, 0.7], [0.01, 0.02, 0.03]])
        ),
        preserved_rest_spin_magnitude=None,
        retarded_time_ns=root,
    )
    if chart:
        kwargs.update(
            rest_spin_stereographic_coefficients=np.array(
                [[0.1, -0.2], [0.03, 0.02], [0.001, -0.002]]
            ),
            rest_spin_stereographic_frame=np.eye(3),
        )
    return kwargs


def displaced_gradient(kwargs, displacement):
    args = dict(kwargs)
    args["observer_time_ns"] += displacement * DIRECTION[0] / C_MMNS
    args["observer_position_mm"] = (
        args["observer_position_mm"] + displacement * DIRECTION[1:]
    )
    # Independent scalar bisection, used ONLY by the validation stencil.
    left, right = 0.0, 0.1
    for _ in range(60):
        root = (left + right) / 2
        source = np.polynomial.polynomial.polyval(
            root / 0.1, args["position_coefficients_mm"]
        )
        residual = C_MMNS * (args["observer_time_ns"] - root) - np.linalg.norm(
            args["observer_position_mm"] - source
        )
        if residual > 0:
            left = root
        else:
            right = root
    args["retarded_time_ns"] = (left + right) / 2
    return pack_partial_antisymmetric_response_native(evaluate(**args).partial_f)


@pytest.mark.parametrize("beta", [0.0, 0.8, 0.999])
@pytest.mark.parametrize("chart", [False, True])
def test_matches_displaced_fixed_polynomial_and_preserves_ordinary_outputs(beta, chart):
    kwargs = polynomial_case(beta, chart)
    baseline = evaluate(**kwargs)
    actual = evaluate(**kwargs, observer_four_velocity_mm_ns=DIRECTION)
    for name in baseline.__dataclass_fields__:
        if name not in (RATE, "directional_light_cone_jet_residual"):
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(baseline, name)
            )
    h = 1e-5 / C_MMNS
    expected = (
        -displaced_gradient(kwargs, 2 * h)
        + 8 * displaced_gradient(kwargs, h)
        - 8 * displaced_gradient(kwargs, -h)
        + displaced_gradient(kwargs, -2 * h)
    ) / (12 * h)
    error = np.linalg.norm(getattr(actual, RATE) - expected) / np.linalg.norm(expected)
    assert error < 2e-7
    assert actual.directional_light_cone_jet_residual < 1e-6


def test_direction_is_linear_and_zero_is_exact():
    kwargs = polynomial_case(0.8)
    a = evaluate(**kwargs, observer_four_velocity_mm_ns=DIRECTION)
    b = evaluate(**kwargs, observer_four_velocity_mm_ns=-3 * DIRECTION)
    # Some packed entries nearly cancel. Compare to the response norm rather
    # than demanding relative machine precision in those small components.
    error = np.linalg.norm(getattr(b, RATE) + 3 * getattr(a, RATE))
    assert error < 5e-14 * np.linalg.norm(getattr(b, RATE))
    zero = evaluate(**kwargs, observer_four_velocity_mm_ns=np.zeros(4))
    np.testing.assert_array_equal(getattr(zero, RATE), np.zeros((4, 6)))
    assert getattr(evaluate(**kwargs), RATE) is None


@pytest.mark.parametrize(
    "direction", [np.zeros(3), [np.nan, 0, 0, 0], [0, np.inf, 0, 0]]
)
def test_invalid_direction_rejected_including_empty_collection(direction):
    with pytest.raises(ValueError, match="four finite"):
        evaluate(**polynomial_case(0), observer_four_velocity_mm_ns=direction)
    with pytest.raises(ValueError, match="four finite"):
        evaluate_collection(
            CausalLocalDipoleSourceCollection(()),
            ObserverEvent(0, (1, 0, 0)),
            fit=LocalSourceJetFitConfig(0.01),
            observer_four_velocity_mm_ns=direction,
        )


@pytest.mark.parametrize("alignment", ["centered", "past"])
def test_history_derivative_matches_refitting_for_exact_polynomial_history(alignment):
    history = _analytic_history()
    event = ObserverEvent(0.02, (1.0, 0.25, -0.05))
    fit = LocalSourceJetFitConfig(half_width_ns=0.012, window_alignment=alignment)
    args = dict(magnetic_moment_native=-0.7, fit=fit)
    ordinary, old_diagnostics = evaluate_history(history, event, **args)
    actual, diagnostics = evaluate_history(
        history, event, **args, observer_four_velocity_mm_ns=DIRECTION
    )
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        np.testing.assert_array_equal(getattr(ordinary, name), getattr(actual, name))
    np.testing.assert_array_equal(
        diagnostics.acceleration_sample_indices,
        old_diagnostics.acceleration_sample_indices,
    )
    np.testing.assert_array_equal(
        diagnostics.spin_sample_indices, old_diagnostics.spin_sample_indices
    )
    h = 1e-5 / C_MMNS
    gradients = []
    for step in (-h, h):
        shifted = ObserverEvent(
            event.time_ns + step * DIRECTION[0] / C_MMNS,
            tuple(np.asarray(event.position_mm) + step * DIRECTION[1:]),
        )
        response, _ = evaluate_history(history, shifted, **args)
        gradients.append(pack_partial_antisymmetric_response_native(response.partial_f))
    expected = (gradients[1] - gradients[0]) / (2 * h)
    assert (
        np.linalg.norm(getattr(actual, RATE) - expected) / np.linalg.norm(expected)
        < 1e-4
    )


def test_history_availability_guard_still_applies():
    with pytest.raises(CausalLocalSourceHistoryUnavailableError):
        evaluate_history(
            _analytic_history(acceleration_ready=False),
            ObserverEvent(0.02, (1, 0.25, -0.05)),
            magnetic_moment_native=1,
            fit=LocalSourceJetFitConfig(0.012),
            observer_four_velocity_mm_ns=DIRECTION,
        )


def test_directional_response_is_included_in_fit_spread():
    response = evaluate(**polynomial_case(0.8), observer_four_velocity_mm_ns=DIRECTION)
    changed = replace(response, **{RATE: 2 * getattr(response, RATE)})
    spread = _response_model_spread([response, changed])
    assert spread.partial_f == 0
    assert spread.directional_partial_f == pytest.approx(0.5)
    assert spread.maximum == pytest.approx(0.5)
    with pytest.raises(ValueError, match="mixed directional"):
        _response_model_spread([response, replace(response, **{RATE: None})])


@pytest.mark.parametrize("multiscale", [False, True])
def test_collection_sums_excludes_and_checks_nested_directional_fits(multiscale):
    history = _analytic_history()
    event = ObserverEvent(0.02, (1, 0.25, -0.05))
    collection = CausalLocalDipoleSourceCollection(
        (
            CausalLocalDipoleSource("first", 0, -0.7, history),
            CausalLocalDipoleSource("second", 1, 0.2, history),
        )
    )
    if multiscale:
        function = evaluate_multiscale
        kwargs = dict(
            scales=LocalSourceJetMultiScaleConfig(
                (
                    _past_scale("short", (0.005, 0.007, 0.01)),
                    _past_scale("long", (0.01, 0.012, 0.015)),
                ),
                maximum_cross_scale_relative_spread=1e-3,
            )
        )
    else:
        function = evaluate_collection
        kwargs = dict(
            fit=LocalSourceJetFitConfig(0.012),
            model_spread=LocalSourceJetModelSpreadConfig(
                LocalSourceJetFitConfig(0.01),
                LocalSourceJetFitConfig(0.015),
                maximum_relative_spread=1e-3,
            ),
        )
    kwargs["observer_four_velocity_mm_ns"] = DIRECTION
    actual = function(collection, event, **kwargs)
    expected = sum(getattr(item.response, RATE) for item in actual.source_results)
    np.testing.assert_array_equal(getattr(actual, RATE), expected)
    assert not getattr(actual, RATE).flags.writeable
    for item in actual.source_results:
        assert item.diagnostics.model_spread.directional_partial_f is not None
    excluded = function(
        collection, event, **kwargs, excluded_source_identities=("first", "second")
    )
    np.testing.assert_array_equal(getattr(excluded, RATE), np.zeros((4, 6)))


def test_directional_fit_disagreement_fails_closed(monkeypatch):
    import core.causal_local_source_jet as provider

    original = provider.polynomial_dipole_hertz_response_jet_native

    def perturbed(**kwargs):
        result = original(**kwargs)
        if kwargs["segment_duration_ns"] < 0.024:
            return replace(result, **{RATE: 10 * getattr(result, RATE)})
        return result

    monkeypatch.setattr(
        provider, "polynomial_dipole_hertz_response_jet_native", perturbed
    )
    with pytest.raises(CausalLocalSourceJetModelSpreadError):
        evaluate_history(
            _analytic_history(),
            ObserverEvent(0.02, (1, 0.25, -0.05)),
            magnetic_moment_native=1,
            fit=LocalSourceJetFitConfig(0.012),
            model_spread=LocalSourceJetModelSpreadConfig(
                LocalSourceJetFitConfig(0.01), LocalSourceJetFitConfig(0.015)
            ),
            observer_four_velocity_mm_ns=DIRECTION,
        )
