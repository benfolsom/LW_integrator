"""Potential-derived reaction inputs and consistent reciprocal history updates."""

from dataclasses import replace
import json
import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.jakobsen_pair import (
    FrozenSource,
    RetardedSourceProvider,
    advance_pair,
    source_row,
)
from core.jakobsen_step import JakobsenParticle, canonical_dynamics
from tests.unit.test_jakobsen_pair import initial


def provider_for(spin, beta, accelerated=False):
    particle = JakobsenParticle(10, 1, 5.5856946893)
    acceleration = 0.01 if accelerated else 0.0
    spin_rate = spin * np.array([0.01, -0.03, 0.02]) if accelerated else np.zeros(3)

    def row(t):
        return np.r_[
            t,
            [beta * c * t, 0.5 * acceleration * c * t * t, 0],
            [beta, acceleration * t, 0],
            [0, acceleration / c, 0],
            spin * np.array([0.2, 0.3, 0.4]) + t * spin_rate,
            spin_rate,
        ]

    source = FrozenSource(particle, [row(-0.1)], [], [])
    source.append(row(0.1))
    return RetardedSourceProvider(source)


@pytest.mark.parametrize("spin", [0.0, 0.3])
@pytest.mark.parametrize("beta", [0.0, 0.8, 0.999])
@pytest.mark.parametrize("accelerated", [False, True])
def test_directional_derivative_matches_displaced_compiled_potential_response(
    spin, beta, accelerated
):
    provider = provider_for(spin, beta, accelerated)
    root = 0.013
    fraction = (root + 0.1) / 0.2
    x = np.polynomial.polynomial.polyval(fraction, provider.source.coefficients[0]) + [
        2.0,
        1.0,
        -0.5,
    ]
    t = root + np.linalg.norm([2.0, 1.0, -0.5]) / c
    u = c * np.array([1.7, 0.4, -0.3, 0.2])
    actual = provider.gradient_proper_rate(t, x, u)

    def value(h):
        return provider(t + h * u[0] / c, x + h * u[1:])[3]

    errors = []
    for h in [2e-5 / c, 1e-5 / c]:
        expected = (-value(2 * h) + 8 * value(h) - 8 * value(-h) + value(-2 * h)) / (
            12 * h
        )
        errors.append(np.linalg.norm(actual - expected) / np.linalg.norm(actual))
    assert max(errors) < 3e-6
    np.testing.assert_array_equal(actual, -actual.swapaxes(1, 2))


def test_derivative_at_join_is_rejected():
    provider = provider_for(0.3, 0.2)
    row = provider.source.rows[-1].copy()
    row[0] += 0.2
    row[1] += 0.2 * c * 0.2
    provider.source.append(row)
    x = provider.source.rows[1][1:4] + [1.0, 0, 0]
    with pytest.raises(ValueError, match="boundary"):
        provider.gradient_proper_rate(0.1 + 1 / c, x, [c, 0, 0, 0])


@pytest.mark.parametrize("beta", [0.0, 0.8, 0.999])
def test_magnetic_derivative_is_resolved_separately_from_charge(beta):
    magnetic = provider_for(0.3, beta, True)
    charge = provider_for(0.0, beta, True)
    t = 0.013 + 3 / c
    x = np.array([beta * c * 0.013, 2.0, 1.0])
    u = c * np.array([20.0, 0.0, 0.0, np.sqrt(399.0)])
    actual = magnetic.gradient_proper_rate(t, x, u) - charge.gradient_proper_rate(
        t, x, u
    )

    def gradient(h):
        time = t + h * u[0] / c
        point = x + h * u[1:]
        return magnetic(time, point)[3] - charge(time, point)[3]

    h = 1e-6 / c
    expected = (
        -gradient(2 * h) + 8 * gradient(h) - 8 * gradient(-h) + gradient(-2 * h)
    ) / (12 * h)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(actual) < 3e-6


@pytest.mark.parametrize("spin", [0.0, 1e-3])
def test_reaction_history_uses_applied_force_and_restart_is_exact(spin):
    # Retain exactly the same initial state and coasting histories for on/off.
    start = initial(spin=spin)
    for p in start["particles"]:
        p["reaction_mode"] = "experimental_linear_spin"
    # Recompute right endpoint derivatives for the newly chosen mode.
    particles = [JakobsenParticle(**p) for p in start["particles"]]
    sources = [FrozenSource(p, **s) for p, s in zip(particles, start["sources"])]
    for i in range(2):
        provider = RetardedSourceProvider(sources[1 - i])
        state = np.array(start["states"][i])
        row = source_row(state, particles[i], provider)
        rhs, response, u = canonical_dynamics(
            state, particle=particles[i], provider=provider
        )
        beta = u[1:] / u[0]
        expected = (response.four_force[1:] - beta * response.four_force[0]) / (
            particles[i].mass_amu * u[0] ** 2
        )
        np.testing.assert_allclose(row[7:10], expected, rtol=5e-16, atol=0)
        np.testing.assert_array_equal(row[13:16], rhs[8:] / (u[0] / c))
        start["sources"][i]["rows"][-1] = row.tolist()
    saved = json.dumps(start)
    first, _ = advance_pair(start, 0.02 / c, 2)
    assert json.dumps(start) == saved
    a, records = advance_pair(first, 0.02 / c, 2)
    b, _ = advance_pair(json.loads(json.dumps(first)), 0.02 / c, 2)
    assert a == b
    swapped = json.loads(saved)
    for key in ("particles", "states", "sources"):
        swapped[key].reverse()
    exchanged, _ = advance_pair(swapped, 0.02 / c, 4)
    np.testing.assert_array_equal(a["states"], exchanged["states"][::-1])
    with pytest.raises(ValueError, match="history unavailable"):
        advance_pair(start, 10 / c)
    assert json.dumps(start) == saved
    assert np.isfinite(records[-1]["mechanical_impulse_residual"]).all()


def test_zero_spin_force_matches_medina_charge_response():
    from core.jakobsen_reaction import _medina_four_force
    from core.jakobsen_step import state_velocity

    start = initial(spin=0.0)
    p = replace(
        JakobsenParticle(**start["particles"][0]),
        reaction_mode="experimental_linear_spin",
    )
    provider = RetardedSourceProvider(
        FrozenSource(JakobsenParticle(**start["particles"][1]), **start["sources"][1])
    )
    state = np.asarray(start["states"][0])
    _, response, u = canonical_dynamics(state, particle=p, provider=provider)
    _, (_, _, f, df) = state_velocity(state, p, provider)
    signs = np.array([1.0, -1.0, -1.0, -1.0])
    force = p.charge_native / c * f @ (signs * u)
    force_rate = (
        p.charge_native
        / c
        * (
            np.einsum("a,aij->ij", u, df) @ (signs * u)
            + f @ (signs * force / p.mass_amu)
        )
    )
    medina = _medina_four_force(u, force, force_rate, p.mass_amu, p.charge_native)
    np.testing.assert_allclose(
        response.four_force, force + medina, rtol=2e-14, atol=1e-14
    )
    np.testing.assert_array_equal(response.intrinsic_spin_reaction, np.zeros(4))
