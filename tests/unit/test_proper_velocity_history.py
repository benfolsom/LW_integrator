"""Constrained proper-velocity interpolation against independent integrals."""

from dataclasses import replace

import numpy as np
import pytest

from core.constants import C_MMNS
from core.light_cone_history import (
    build_light_cone_segment,
    segment_speed_deficit_bound,
)
from core.proper_velocity_history import (
    ProperVelocityConstraintError,
    constrain_proper_velocity_segment,
)

pytestmark = pytest.mark.unit
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(256)


def _beta_prime(u, du):
    gamma = np.sqrt(1 + np.dot(u, u))
    return du / gamma - u * np.dot(u, du) / gamma**3


def _example(length):
    u = np.array([[1.0, 0.0, 0.0], [1.0, length, 0.0]])
    gamma = np.sqrt(1 + np.sum(u * u, axis=1))
    beta = u / gamma[:, None]
    primes = np.array([_beta_prime(value, np.array([0.0, 1.0, 0.0])) for value in u])
    accepted = length * (u[0] + u[1]) / (gamma[0] + gamma[1])
    return (
        build_light_cone_segment(
            np.array([0.0, length / C_MMNS]),
            np.array([np.zeros(3), accepted]),
            beta,
            primes,
            proper_velocity=u,
        ),
        primes,
        accepted,
    )


@pytest.mark.parametrize("length", [1.0, 0.1, 0.01])
def test_newton_reaches_integrator_accepted_endpoint(length):
    base, primes, accepted = _example(length)
    original = base.endpoint_null_mm.copy()
    segment = constrain_proper_velocity_segment(base, primes)
    assert segment.newton_iterations > 0
    assert np.all(
        abs(segment.constraint_residual_mm) <= segment.constraint_tolerance_mm
    )
    np.testing.assert_allclose(
        segment.sample(1.0)[0],
        accepted,
        rtol=0.0,
        atol=4 * np.finfo(float).eps * length,
    )
    np.testing.assert_array_equal(base.endpoint_null_mm, original)
    for fraction in np.linspace(0, 1, 41):
        _, _, _, invariant = segment.sample(fraction)
        assert invariant > 0
    assert segment_speed_deficit_bound(segment) > 0
    for fraction, knot in ((0.0, 0), (1.0, 1)):
        u = segment.null_sample(fraction)[5] @ segment.frame
        np.testing.assert_allclose(
            u, base.proper_velocity_knots[knot], rtol=2e-14, atol=1e-15
        )
        np.testing.assert_allclose(
            segment.sample(fraction)[2], primes[knot], rtol=2e-13, atol=2e-14
        )


def test_constraint_failure_is_explicit_without_fallback(monkeypatch):
    import core.proper_velocity_history as interpolation

    base, primes, _ = _example(1.0)
    monkeypatch.setattr(interpolation, "_MAX_NEWTON_ITERATIONS", 0)
    with pytest.raises(ProperVelocityConstraintError, match="did not converge"):
        constrain_proper_velocity_segment(base, primes)
    impossible = replace(base, endpoint_null_mm=np.array([-0.1, 0.0, 0.0]))
    with pytest.raises(ProperVelocityConstraintError, match="not strictly timelike"):
        constrain_proper_velocity_segment(impossible, primes)


def _smooth_u(s):
    return np.stack(
        (1 + 0.2 * np.sin(s), 0.3 * np.cos(1.3 * s), 0.1 * np.exp(0.5 * s)), axis=-1
    )


def _smooth_du(s):
    return np.array([0.2 * np.cos(s), -0.39 * np.sin(1.3 * s), 0.05 * np.exp(0.5 * s)])


def _exact_position(s):
    u = _smooth_u(0.5 * s * (_NODES + 1))
    beta = u / np.sqrt(1 + np.sum(u * u, axis=1))[:, None]
    return 0.5 * s * (_WEIGHTS @ beta)


def test_smooth_worldline_converges_with_step_size():
    errors = []
    default_errors = []
    differences = []
    for length in (1.0, 0.5, 0.25, 0.125):
        u = np.array([_smooth_u(0.0), _smooth_u(length)])
        gamma = np.sqrt(1 + np.sum(u * u, axis=1))
        primes = np.array(
            [_beta_prime(value, _smooth_du(s)) for value, s in zip(u, (0.0, length))]
        )
        base = build_light_cone_segment(
            np.array([0.0, length / C_MMNS]),
            np.array([np.zeros(3), _exact_position(length)]),
            u / gamma[:, None],
            primes,
            proper_velocity=u,
        )
        proper = constrain_proper_velocity_segment(base, primes)
        fraction = 0.37
        actual = proper.sample(fraction)[0]
        default = base.sample(fraction)[0]
        exact = _exact_position(length * fraction)
        errors.append(np.linalg.norm(actual - exact))
        default_errors.append(np.linalg.norm(default - exact))
        differences.append(np.linalg.norm(actual - default))
    for series in (errors, default_errors, differences):
        orders = np.log2(np.array(series[:-1]) / series[1:])
        assert np.all(orders > 5.0), (series, orders)


def test_accelerated_python_and_numba_root_agree():
    from core.proper_velocity_history import (
        solve_constrained_u,
        solve_constrained_u_numba,
    )

    base, primes, _ = _example(1.0)
    segment = constrain_proper_velocity_segment(base, primes)
    fraction = 0.37
    values = segment.null_sample(fraction)[0]
    # Observer at this source position plus a one-millimetre axial null ray.
    geometry = np.array(
        [
            values[0],
            fraction - values[0] + 1,
            values[1],
            values[2],
            2 + 2 * fraction - values[0],
        ]
    )
    arguments = (
        segment.u_coefficients,
        1.0,
        geometry,
        0.0,
        1.0,
        1e-21,
        96,
        segment.quadrature_nodes,
        segment.quadrature_weights,
    )
    python = solve_constrained_u(*arguments)
    compiled = solve_constrained_u_numba(*arguments)
    assert python[0] == pytest.approx(fraction, abs=2e-15)
    assert compiled[0] == pytest.approx(python[0], abs=2e-15)
    np.testing.assert_allclose(compiled[1][3], python[1][3], rtol=2e-15, atol=1e-15)


def test_preparation_does_not_modify_accepted_trajectory():
    from tests.unit.test_high_gamma_stored_history import _coast, _event
    from core.retarded_fields import (
        _prepare_history,
        evaluate_retarded_charge_field_native,
    )

    history, rows = _coast(1e12, steps=12, representation="proper_velocity")
    saved = {
        name: getattr(history, name).copy()
        for name in ("x", "y", "z", "t", "source_u_x")
    }
    prepared = _prepare_history(history, ())
    for name, value in saved.items():
        np.testing.assert_array_equal(getattr(history, name), value)
    for segment in prepared.sources[0].light_cone_segments:
        assert np.all(
            abs(segment.constraint_residual_mm) <= segment.constraint_tolerance_mm
        )
    with pytest.raises(ValueError, match="Metal"):
        evaluate_retarded_charge_field_native(
            history, _event(rows[-1]), backend="metal_certified_full_strict"
        )
    from core.retarded_fields import _prepare_source_history

    with pytest.raises(ValueError, match="spin C5"):
        _prepare_source_history(prepared.arrays, 0, light_cone_enabled=False)


def test_speed_deficit_bound_rounds_outward():
    from decimal import Decimal, localcontext

    base, primes, _ = _example(1.0)
    segment = constrain_proper_velocity_segment(base, primes)
    coefficients = np.zeros((5, 3))
    coefficients[0] = (4249.3423082088175, 3827801.8511173176, 48868809.190180056)
    segment = replace(segment, u_coefficients=coefficients)
    with localcontext() as context:
        context.prec = 80
        norm = sum((Decimal.from_float(float(v)) ** 2 for v in coefficients[0])).sqrt()
        gamma = (1 + norm * norm).sqrt()
        exact = 1 / (gamma * (gamma + norm))
        bound = Decimal.from_float(segment.speed_deficit_bound())
    assert 0 < bound <= exact
