"""Differentiate the maintained full-spin equations, using potential Taylor data."""

import numpy as np

from . import momentum_center as model
from .taylor_series import ORDER, from_series, to_series, solve


def _contract(a, b):
    return np.sum(a * b * model.METRIC[:, None] * model.METRIC[None, :])


def _motion(state, particle, response):
    p, spin = state[4:8], model.unpack(state[8:])
    field, gradient = response[2:]
    mass2 = (model.METRIC * p) @ p
    direction, force, torque, coefficient, _, correction = model._velocity_system(
        p, spin, field, gradient, particle, mass2, contraction=_contract, solve=solve
    )
    norm = (model.METRIC * direction) @ direction
    omega = norm.sqrt().reciprocal()
    u, coupling = omega * direction, coefficient * omega
    rate, spin_rate = model._motion_rates(
        p, u, field, coupling * force, coupling * torque, omega * correction, particle
    )
    return np.r_[u, rate, model.pack(spin_rate)], u, coupling * spin


def source_coefficients(state, particle, provider):
    """Order-four u and D coefficients. Require an explicit analytical provider."""
    evaluator = getattr(provider, "taylor_response", None)
    if evaluator is None:
        raise ValueError(
            "Analytical reaction derivatives require provider.taylor_response; no fitting fallback"
        )
    _, center = model.evaluate(state, particle, provider)  # preserve all domain checks
    kinetic = np.array(state, dtype=float, copy=True)
    kinetic[4:8] = center["kinetic_momentum"]
    values = np.zeros((ORDER + 1, 14))
    values[0] = kinetic
    for n in range(ORDER + 1):
        supplied = evaluator(values[:, :4])
        shapes = ((5, 4), (5, 4, 4), (5, 4, 4), (5, 4, 4, 4))
        if len(supplied) != 4 or any(
            np.shape(v) != shape or not np.isfinite(v).all()
            for v, shape in zip(supplied, shapes)
        ):
            raise ValueError(
                "Finite potential Taylor response with correct shapes required"
            )
        response = tuple(to_series(v) for v in supplied)
        rate, u, d = _motion(to_series(values), particle, response)
        if n < ORDER:
            values[n + 1] = from_series(rate)[n] / (n + 1)
    u, d = from_series(u), from_series(d)
    if not np.isfinite(u).all() or not np.isfinite(d).all():
        raise ValueError("Nonfinite analytical source coefficients")
    return (
        u,
        d,
        dict(
            derivative_method="analytic",
            ordinary_evaluations=ORDER + 1,
            ordinary_velocity=u[0],
            proper_window=None,
            coasting_control=not np.any(u[1:]) and not np.any(d[1:]),
        ),
    )
