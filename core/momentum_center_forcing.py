"""Experimental coupling contract for a prescribed force and antisymmetric torque.

This supplies no self-force formula. It solves the full spin constraint with
the supplied force/torque included and rejects incompatible mass-law rates.
It is a building block for matched nonlinear reaction, not a new CLI mode.
"""

import numpy as np
from typing import Any

from . import momentum_center as model


def evaluate_forced(
    state: np.ndarray,
    particle: model.Particle,
    provider: model.Provider,
    force: np.ndarray,
    torque: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Couple fixed proper-time R,T at this state, with no spin expansion.

    A velocity-dependent reaction must satisfy its own self-consistency solve;
    this function treats its inputs as prescribed. No implicit solution of a
    history-dependent reaction is claimed.
    """
    force, torque = np.asarray(force, dtype=float), np.asarray(torque, dtype=float)
    if (
        force.shape != (4,)
        or torque.shape != (4, 4)
        or not np.isfinite(force).all()
        or not np.isfinite(torque).all()
    ):
        raise ValueError("Finite prescribed four-force and torque required")
    if not np.allclose(torque, -torque.T, rtol=0, atol=1e-14):
        raise ValueError("Prescribed torque must be antisymmetric")
    # Validate the original model domain before introducing forcing, reusing
    # the same potential response for both the ordinary and forced solves.
    state = np.asarray(state, dtype=float)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Finite 14-component momentum-center state required")
    values = provider(state[:4])
    base, diagnostic = model.evaluate(state, particle, lambda event: values)
    if not np.any(force) and not np.any(torque):
        return base, diagnostic
    _, da, field, gradient = values
    p, s = diagnostic["kinetic_momentum"], model.unpack(state[8:])
    w, unit_force, unit_torque, coefficient, _, _ = model.velocity_direction(
        p, s, field, gradient, particle
    )
    matrix = model.dot(p, p) * np.eye(4) - particle.charge * (
        s * model.METRIC[None, :]
    ) @ (field * model.METRIC[None, :])
    z = np.linalg.solve(
        matrix, torque @ (model.METRIC * p) + s @ (model.METRIC * force)
    )
    aa, bb, cc = model.dot(w, w), model.dot(w, z), model.dot(z, z) - 1
    discriminant = bb**2 - aa * cc
    if discriminant < 0 or not np.isfinite(discriminant):
        raise model.VelocityDomainError(
            "Prescribed reaction has no real normalized velocity"
        )
    root = np.sqrt(discriminant)
    omega = -cc / (root + bb) if bb >= 0 else (root - bb) / aa
    u = omega * w + z
    if not np.isfinite(u).all() or u[0] <= 0 or model.dot(u, u) <= 0:
        raise model.VelocityDomainError(
            "Prescribed reaction has no future timelike branch"
        )
    coupling = coefficient * omega
    pd = particle.charge * field @ (model.METRIC * u) + coupling * unit_force + force
    sd = np.outer(p, u) - np.outer(u, p) + coupling * unit_torque + torque
    scalar_rate = (
        model.contract(field, sd)
        + np.array([model.contract(entry, s) for entry in gradient]) @ u
    )
    mass_rate = (
        2 * model.dot(p, pd) - particle.bare_mass * particle.coupling * scalar_rate
    )
    scale = 2 * np.linalg.norm(p) * np.linalg.norm(pd) + abs(
        particle.bare_mass * particle.coupling
    ) * (
        np.linalg.norm(field) * np.linalg.norm(sd)
        + np.linalg.norm(gradient) * np.linalg.norm(s) * np.linalg.norm(u)
    )
    if abs(mass_rate) > 1024 * np.finfo(float).eps * max(scale, np.finfo(float).tiny):
        raise ValueError(
            "Prescribed force/torque violates the current mass law; a matched internal-energy law is required"
        )
    constraint_rate = sd @ (model.METRIC * p) + s @ (model.METRIC * pd)
    constraint_scale = np.linalg.norm(sd) * np.linalg.norm(p) + np.linalg.norm(
        s
    ) * np.linalg.norm(pd)
    if np.linalg.norm(constraint_rate) > 1024 * np.finfo(float).eps * max(
        constraint_scale, np.finfo(float).tiny
    ):
        raise ValueError(
            "Prescribed force/torque does not preserve the spin constraint"
        )
    return np.r_[u, pd + particle.charge * u @ da, model.pack(sd)], dict(
        proper_velocity=u,
        kinetic_momentum=p,
        momentum_rate=pd,
        spin_rate=sd,
        constraint_rate=constraint_rate,
        mass_constraint_rate=mass_rate,
        proper_dipole=coupling * s,
        prescribed_force=force.copy(),
        prescribed_torque=torque.copy(),
        scope="Prescribed forcing contract only; not a self-reaction prescription",
    )
