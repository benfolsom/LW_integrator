"""Charge contribution to the moment-force time derivative, with shared roots."""

import numpy as np
import pytest

from core.constants import C_MMNS
from core.charge_response_jet import quintic_charge_response_jet_native
from core.retarded_fields import (
    ObserverEvent,
    RetardedHistoryError,
    evaluate_retarded_charge_response_gradient_native as provider,
)
from core.retarded_potential_directional_jet import (
    quintic_charge_response_directional_gradient_native as derivative,
    quintic_charge_potential_directional_jet_native as reference,
)
from tests.unit.test_analytic_charge_response_provider import _uniform_history


RATE = "partial_antisymmetric_response_along_velocity"
DIRECTION = np.array([500.0, 100.0, -70.0, 50.0])


def case(beta=0.0, acceleration=False):
    coefficients = np.zeros((6, 3))
    coefficients[1] = [beta * C_MMNS * 0.1, 0, 0]
    if acceleration:
        coefficients[2] = [1e-4, 2e-4, -1e-4]
        coefficients[3] = [-1e-5, 1e-5, 2e-5]
        if beta > 0.999:
            coefficients[2:] *= 1e-4
    root = 0.04
    source = np.polynomial.polynomial.polyval(root / 0.1, coefficients)
    displacement = np.array([1.0, 0.5, -0.3])
    return dict(
        observer_time_ns=root + np.linalg.norm(displacement) / C_MMNS,
        observer_position_mm=source + displacement,
        charge_native=-1.3,
        segment_start_time_ns=0.0,
        segment_duration_ns=0.1,
        position_coefficients_mm=coefficients,
        retarded_time_ns=root,
    )


def shifted_gradient(args, h):
    args = dict(args)
    args["observer_time_ns"] += h * DIRECTION[0] / C_MMNS
    args["observer_position_mm"] = args["observer_position_mm"] + h * DIRECTION[1:]
    left, right = 0.0, 0.1
    for _ in range(70):
        t = (left + right) / 2
        position = np.polynomial.polynomial.polyval(
            t / 0.1, args["position_coefficients_mm"]
        )
        residual = C_MMNS * (args["observer_time_ns"] - t) - np.linalg.norm(
            args["observer_position_mm"] - position
        )
        if residual > 0:
            left = t
        else:
            right = t
    args["retarded_time_ns"] = (left + right) / 2
    return quintic_charge_response_jet_native(**args).partial_antisymmetric_response


@pytest.mark.parametrize("gamma", [1.0, 2.0, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize("acceleration", [False, True])
def test_matches_independent_stable_response_stencil(gamma, acceleration):
    args = case(np.sqrt(1 - 1 / gamma**2), acceleration)
    actual = getattr(derivative(**args, four_velocity_mm_ns=DIRECTION), RATE)
    h = 1e-5 / C_MMNS
    expected = (
        -shifted_gradient(args, 2 * h)
        + 8 * shifted_gradient(args, h)
        - 8 * shifted_gradient(args, -h)
        + shifted_gradient(args, -2 * h)
    ) / (12 * h)
    error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert error < 3e-6


def test_static_coulomb_second_gradient_is_analytic():
    args = case()
    actual = getattr(derivative(**args, four_velocity_mm_ns=DIRECTION), RATE)
    r = args["observer_position_mm"]
    v = DIRECTION[1:]
    radius = np.linalg.norm(r)
    expected = np.zeros((4, 6))
    for coordinate in range(3):
        for i in range(3):
            expected[coordinate + 1, i] = -args["charge_native"] * (
                -3
                * (
                    (i == coordinate) * np.dot(r, v)
                    + v[i] * r[coordinate]
                    + r[i] * v[coordinate]
                )
                / radius**5
                + 15 * r[i] * r[coordinate] * np.dot(r, v) / radius**7
            )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-12)


def test_subset_matches_broader_potential_reference():
    args = case(0.6, True)
    broad = reference(
        **args, four_velocity_mm_ns=DIRECTION, four_acceleration_mm_ns2=np.zeros(4)
    )
    a3 = broad.partial3_a_along_velocity
    signs = [1, -1, -1, -1]
    expected = np.array(
        [
            [
                signs[m] * a3[coordinate, m, n] - signs[n] * a3[coordinate, n, m]
                for m, n in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            ]
            for coordinate in range(4)
        ]
    )
    np.testing.assert_allclose(
        getattr(derivative(**args, four_velocity_mm_ns=DIRECTION), RATE),
        expected,
        rtol=3e-12,
        atol=1e-10,
    )


def test_provider_preserves_ordinary_outputs_and_reuses_compiled_roots(monkeypatch):
    import core.charge_response_jet_numba as compiled
    import core.retarded_fields as fields

    original = compiled.evaluate_charge_response_coefficients_one_event_strict_serial
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected second scalar root solve")

    history = _uniform_history(
        times_ns=np.linspace(-0.04, 0.004, 121), beta=np.array([0.31, -0.12, 0.07])
    )
    event = ObserverEvent(-1e-5, (0.9, 1.1, -0.5))
    old = provider(history, event, relative_step=2.5e-6)
    monkeypatch.setattr(
        compiled,
        "evaluate_charge_response_coefficients_one_event_strict_serial",
        counted,
    )
    monkeypatch.setattr(fields, "_solve_retarded_sample", forbidden)
    new = provider(
        history, event, relative_step=2.5e-6, observer_four_velocity_mm_ns=DIRECTION
    )
    assert len(calls) == 1 and not new.fallback_used
    for name in old.__dataclass_fields__:
        if name not in (
            RATE,
            "directional_unavailable_reason",
            "directional_jet_residual",
        ):
            np.testing.assert_array_equal(getattr(new, name), getattr(old, name))
    assert getattr(new, RATE) is not None and not getattr(new, RATE).flags.writeable
    assert new.directional_unavailable_reason is None


def test_boundary_fallback_explicitly_withholds_directional_gradient():
    history = _uniform_history(times_ns=np.array([-0.01, 0, 0.01]), beta=np.zeros(3))
    response = provider(
        history,
        ObserverEvent(1 / C_MMNS, (1, 0, 0)),
        observer_four_velocity_mm_ns=DIRECTION,
    )
    assert response.fallback_used
    assert getattr(response, RATE) is None
    assert "fallback" in response.directional_unavailable_reason


def test_excluded_and_missing_sources_remain_distinct():
    history = _uniform_history(times_ns=np.array([-0.01, 0, 0.01]), beta=np.zeros(3))
    event = ObserverEvent(0.02, (1, 0, 0))
    with pytest.raises(RetardedHistoryError):
        provider(history, event, observer_four_velocity_mm_ns=DIRECTION)
    excluded = provider(
        history,
        event,
        excluded_source_indices=(0,),
        observer_four_velocity_mm_ns=DIRECTION,
    )
    np.testing.assert_array_equal(getattr(excluded, RATE), np.zeros((4, 6)))
    assert not excluded.valid_sources.any()


@pytest.mark.parametrize("direction", [[1, 2, 3], [np.nan, 0, 0, 0]])
def test_invalid_direction_rejected(direction):
    with pytest.raises(ValueError):
        derivative(**case(), four_velocity_mm_ns=direction)
    with pytest.raises(ValueError):
        provider(
            [], ObserverEvent(0, (1, 0, 0)), observer_four_velocity_mm_ns=direction
        )


def test_zero_direction_is_exact_zero():
    np.testing.assert_array_equal(
        getattr(derivative(**case(0.6), four_velocity_mm_ns=np.zeros(4)), RATE),
        np.zeros((4, 6)),
    )


@pytest.mark.parametrize("semantics", ["preceding_interval", "instantaneous"])
def test_provider_direction_differentiates_selected_acceleration_convention(semantics):
    history = _uniform_history(
        times_ns=np.linspace(-0.04, 0.004, 121), beta=np.array([0.31, -0.12, 0.07])
    )
    # Deliberately distinguish supplied endpoint acceleration from the
    # derivative reconstructed from these uniform-velocity samples.
    for state in history:
        state["bdotx"][:] = 1e-4
    event = ObserverEvent(-1e-5, (0.9, 1.1, -0.5))
    options = dict(relative_step=2.5e-6, source_acceleration_semantics=semantics)
    ordinary = provider(history, event, **options)
    actual = provider(history, event, observer_four_velocity_mm_ns=DIRECTION, **options)
    assert not actual.fallback_used
    np.testing.assert_array_equal(
        actual.partial_antisymmetric_response, ordinary.partial_antisymmetric_response
    )
    h = 1e-7 / C_MMNS
    gradients = []
    for multiplier in (-2, -1, 1, 2):
        shifted = ObserverEvent(
            event.time_ns + multiplier * h * DIRECTION[0] / C_MMNS,
            tuple(np.asarray(event.position_mm) + multiplier * h * DIRECTION[1:]),
        )
        result = provider(history, shifted, **options)
        assert not result.fallback_used
        np.testing.assert_array_equal(result.segment_index, actual.segment_index)
        gradients.append(result.partial_antisymmetric_response)
    expected = (gradients[0] - 8 * gradients[1] + 8 * gradients[2] - gradients[3]) / (
        12 * h
    )
    assert (
        np.linalg.norm(getattr(actual, RATE) - expected) / np.linalg.norm(expected)
        < 2e-6
    )


def test_multiple_sources_sum_and_exclusion():
    histories = [
        _uniform_history(
            times_ns=np.linspace(-0.04, 0.004, 121),
            beta=np.array(beta),
            charge_native=charge,
        )
        for beta, charge in [([0.31, -0.12, 0.07], -1.3), ([-0.1, 0.2, 0.05], 0.7)]
    ]
    combined = [
        {key: np.concatenate([a[key], b[key]]) for key in a} for a, b in zip(*histories)
    ]
    event = ObserverEvent(-1e-5, (0.9, 1.1, -0.5))
    options = dict(relative_step=2.5e-6, observer_four_velocity_mm_ns=DIRECTION)
    individual = [
        getattr(provider(history, event, **options), RATE) for history in histories
    ]
    together = provider(combined, event, **options)
    np.testing.assert_allclose(getattr(together, RATE), sum(individual), rtol=1e-14)
    excluded = provider(combined, event, excluded_source_indices=(0,), **options)
    np.testing.assert_array_equal(getattr(excluded, RATE), individual[1])
    assert np.isnan(excluded.directional_jet_residual[0])
    assert np.isfinite(excluded.directional_jet_residual[1])
