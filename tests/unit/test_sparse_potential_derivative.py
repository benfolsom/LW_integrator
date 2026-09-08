"""Opt-in sparse potential derivative, fallback, source sum and step contracts."""

import numpy as np
import pytest
from core.constants import C_MMNS as c
from core.dipole_hertz_jet import (
    evaluate_retarded_dipole_field_gradient_hertz_jet_native as evaluate,
)
from core import dipole_hertz_jet_numba as tables
from core.retarded_fields import ObserverEvent
from tests.unit.test_dipole_hertz_jet import _accelerating_rotating_history
from tests.unit.test_retarded_dipole_fields import _dipole_history


@pytest.mark.parametrize("source_beta", [None, 0.0, 0.8, 0.999])
@pytest.mark.parametrize("observer_beta", [0.0, 0.8, 0.999])
def test_sparse_matches_dense_and_preserves_original_outputs(
    source_beta, observer_beta
):
    history = (
        _accelerating_rotating_history()
        if source_beta is None
        else _dipole_history(beta=(source_beta, 0, 0))
    )
    event = ObserverEvent(-1.7e-4, (-1.1, 0.6, 0.8))
    u = c / np.sqrt(1 - observer_beta**2) * np.array([1, 0, observer_beta, 0])
    old = evaluate(history, event, response_kernel="numba_sparse_strict_serial")
    new = evaluate(
        history,
        event,
        response_kernel="numba_sparse_strict_serial",
        observer_four_velocity_mm_ns=u,
        include_partial_a=True,
    )
    dense = evaluate(history, event, response_kernel="numba_strict_serial")
    assert new.used_analytic_response and dense.used_analytic_response
    assert (
        old.response.partial_a is None
        and old.response.four_potential_proper_rate is None
    )
    for name in (
        "four_potential",
        "antisymmetric_response",
        "partial_antisymmetric_response",
    ):
        np.testing.assert_array_equal(
            getattr(old.response, name), getattr(new.response, name)
        )
    np.testing.assert_allclose(
        new.response.partial_a, dense.response.partial_a, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        new.response.four_potential_proper_rate,
        u @ dense.response.partial_a,
        rtol=2e-12,
        atol=2e-10,
    )


def test_only_six_additional_coefficients_are_required():
    assert np.count_nonzero(tables._HERTZ_RESPONSE_USED) == 144
    assert np.count_nonzero(tables._POTENTIAL_RATE_USED) == 150
    assert np.all(tables._POTENTIAL_RATE_USED[tables._HERTZ_RESPONSE_USED])


def test_directional_rate_matches_independent_potential_difference():
    history = _accelerating_rotating_history()
    event = ObserverEvent(-1.7e-4, (-1.1, 0.6, 0.8))
    u = c * np.array([1.4, 0.3, -0.2, 0.6])
    h = 1e-8
    actual = evaluate(
        history,
        event,
        response_kernel="numba_sparse_strict_serial",
        observer_four_velocity_mm_ns=u,
    ).response.four_potential_proper_rate
    values = []
    for sign in (-1, 1):
        shifted = ObserverEvent(
            event.time_ns + sign * h * u[0] / c,
            tuple(np.asarray(event.position_mm) + sign * h * u[1:]),
        )
        value = evaluate(history, shifted, response_kernel="python")
        assert value.used_analytic_response
        values.append(value.response.four_potential)
    np.testing.assert_allclose(
        actual, (values[1] - values[0]) / (2 * h), rtol=3e-7, atol=1e-6
    )


def test_boundary_fallback_supplies_real_derivative_not_zero():
    history = _accelerating_rotating_history()
    knot = history[60]
    position = np.array([knot[k][0] for k in ("x", "y", "z")]) + [1, 0, 0]
    event = ObserverEvent(knot["t"][0] + 1 / c, tuple(position))
    u = c * np.array([1.25, 0.75, 0, 0])
    sparse = evaluate(
        history,
        event,
        response_kernel="numba_sparse_strict_serial",
        observer_four_velocity_mm_ns=u,
        include_partial_a=True,
        fallback_backend="python",
    )
    dense = evaluate(
        history, event, response_kernel="python", fallback_backend="python"
    )
    assert not sparse.used_analytic_response
    np.testing.assert_array_equal(sparse.response.partial_a, dense.response.partial_a)
    np.testing.assert_array_equal(
        sparse.response.four_potential_proper_rate, u @ dense.response.partial_a
    )
    assert np.linalg.norm(sparse.response.four_potential_proper_rate) > 0


def test_multiple_sources_add_in_declared_order():
    history = _accelerating_rotating_history()
    doubled = [
        {key: np.repeat(value, 2) for key, value in state.items()} for state in history
    ]
    event = ObserverEvent(-1.7e-4, (-1.1, 0.6, 0.8))
    u = c * np.array([1.25, 0.75, 0, 0])
    options = dict(
        response_kernel="numba_sparse_strict_serial",
        include_partial_a=True,
        observer_four_velocity_mm_ns=u,
    )
    one = evaluate(history, event, **options).response
    two = evaluate(doubled, event, **options).response
    np.testing.assert_array_equal(two.partial_a, 2 * one.partial_a)
    np.testing.assert_array_equal(
        two.four_potential_proper_rate, 2 * one.four_potential_proper_rate
    )


@pytest.mark.parametrize("direction", [[1, 2, 3], [1, 2, 3, float("nan")]])
def test_invalid_direction_rejected(direction):
    with pytest.raises(ValueError, match="four finite"):
        evaluate(
            _accelerating_rotating_history(),
            ObserverEvent(0, (1, 1, 1)),
            response_kernel="numba_sparse_strict_serial",
            observer_four_velocity_mm_ns=direction,
        )


class CanonicalSource:
    def __init__(self, sparse):
        self.sparse = sparse
        self.history = _accelerating_rotating_history()
        for state in self.history:
            state["magnetic_moment_native"] *= 0.03 * c * c

    def __call__(self, t, x):
        from core.antisymmetric_response_rfs import (
            materialize_antisymmetric_response_native as field,
            materialize_partial_antisymmetric_response_native as gradient,
        )

        result = evaluate(
            self.history,
            ObserverEvent(t, tuple(x)),
            response_kernel=(
                "numba_sparse_strict_serial" if self.sparse else "numba_strict_serial"
            ),
            include_partial_a=self.sparse,
        )
        r = result.response
        if self.sparse:
            return (
                r.four_potential,
                r.partial_a,
                field(r.antisymmetric_response),
                gradient(r.partial_antisymmetric_response),
            )
        return r.four_potential, r.partial_a, r.field_tensor, r.partial_f


@pytest.mark.parametrize("spin", [0.0, 1e-6])
def test_sparse_dense_complete_steps_agree(spin):
    from core.jakobsen_step import (
        JakobsenParticle,
        initial_canonical_state,
        midpoint_step,
    )

    particle = JakobsenParticle(1, 1, 5.5856946893)
    w = np.array([0.4, -0.2, 0.8])
    ends = []
    for sparse in (False, True):
        provider = CanonicalSource(sparse)
        state = initial_canonical_state(
            time_ns=-1.7e-4,
            position_mm=[-1.1, 0.6, 0.8],
            four_velocity_mm_ns=c * np.r_[np.sqrt(1 + w @ w), w],
            rest_spin_angular_momentum=c * spin * np.array([0.2, 0.3, 0.4]),
            particle=particle,
            provider=provider,
        )
        for _ in range(40):
            state = midpoint_step(
                state, 0.02 / (c * 40), particle=particle, provider=provider
            )
        ends.append(state)
    np.testing.assert_allclose(ends[0], ends[1], rtol=1e-13, atol=2e-13)


def test_sparse_adaptive_restart_is_exact():
    import json
    from core.jakobsen_step import JakobsenParticle, initial_canonical_state
    from core.jakobsen_adaptive import checkpoint, integrate
    from core.step_doubling import StepDoublingTolerances, ErrorScale

    provider = CanonicalSource(True)
    particle = JakobsenParticle(1, 1, 5.5856946893)
    state = initial_canonical_state(
        time_ns=-1.7e-4,
        position_mm=[-1.1, 0.6, 0.8],
        four_velocity_mm_ns=c * np.array([1.25, 0.75, 0, 0]),
        rest_spin_angular_momentum=c * np.array([1e-6, 0, 2e-6]),
        particle=particle,
        provider=provider,
    )
    options = dict(
        particle=particle,
        provider=provider,
        provider_id="frozen-test-source",
        maximum_step_ns=1e-5,
        tolerances=StepDoublingTolerances(
            ErrorScale(1e-11, 1e-10),
            ErrorScale(1e-10, 1e-10),
            ErrorScale(1e-13, 1e-9),
            ErrorScale(1e-10, 1e-10),
        ),
    )
    payload = checkpoint(
        state, particle=particle, provider_id=options["provider_id"], next_step_ns=1e-5
    )
    first, _ = integrate(payload, -1.4e-4, **options)
    a, records = integrate(json.loads(json.dumps(first)), -1e-4, **options)
    b, _ = integrate(first, -1e-4, **options)
    assert a == b and all(record["error"] <= 1 for record in records)
