"""Experimental canonical midpoint step for supplied-potential Jakobsen dynamics.

This is an explicit new core API, not a replacement for the legacy RFS EOM.
Native units. State is (t_ns, xyz_mm, P^mu_native, S_rest_native), 11 values.
Providers return (A, partial_A, F, partial_F) at the requested event. Opt-in
linear-spin reaction additionally requires provider.gradient_proper_rate(t,x,u).
"""

from dataclasses import dataclass

import numpy as np

from .constants import C_MMNS as c
from .jakobsen import (
    ordinary_response_native,
    canonical_momentum_native,
    velocity_from_canonical_spatial_native,
)


@dataclass(frozen=True)
class JakobsenParticle:
    charge_native: float
    mass_amu: float
    g: float
    reaction_mode: str = "off"

    def __post_init__(self):
        if (
            not np.isfinite([self.charge_native, self.mass_amu, self.g]).all()
            or self.mass_amu <= 0
        ):
            raise ValueError("Finite particle parameters and positive mass required")
        if self.reaction_mode not in ("off", "experimental_linear_spin"):
            raise ValueError("Unsupported experimental reaction mode")

    def coefficients(self):
        return dict(charge_native=self.charge_native, mass_amu=self.mass_amu, g=self.g)


def initial_canonical_state(
    *,
    time_ns,
    position_mm,
    four_velocity_mm_ns,
    rest_spin_angular_momentum,
    particle,
    provider,
):
    a, _, f, _ = provider(time_ns, np.asarray(position_mm, dtype=float))
    p = canonical_momentum_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        rest_spin_angular_momentum=rest_spin_angular_momentum,
        four_potential=a,
        field_tensor=f,
        **particle.coefficients(),
    )
    state = np.r_[time_ns, position_mm, p, rest_spin_angular_momentum]
    if state.shape != (11,) or not np.isfinite(state).all():
        raise ValueError("Finite canonical state required")
    return state


def state_velocity(state, particle, provider):
    state = np.asarray(state, dtype=float)
    if state.shape != (11,) or not np.isfinite(state).all():
        raise ValueError("Finite 11-value canonical state required")
    a, da, f, df = provider(state[0], state[1:4])
    da = np.asarray(da, dtype=float)
    if da.shape != (4, 4) or not np.isfinite(da).all():
        raise ValueError("Finite (4,4) potential derivative required")
    u = velocity_from_canonical_spatial_native(
        canonical_spatial_momentum=state[5:8],
        rest_spin_angular_momentum=state[8:11],
        four_potential=a,
        field_tensor=f,
        **particle.coefficients(),
    )
    return u, (a, da, f, df)


def canonical_rhs(state, *, particle, provider):
    """Proper-time RHS; canonical P, not cached beta, determines the velocity."""
    return canonical_dynamics(state, particle=particle, provider=provider)[0]


def canonical_dynamics(state, *, particle, provider):
    """Return the RHS, applied response and velocity from one consistent evaluation.

    Source-history endpoints and force accounting must use this same response,
    including reaction when enabled, rather than recomputing an ordinary force.
    """
    u, (a, da, f, df) = state_velocity(state, particle, provider)
    rest = np.asarray(state[8:11])
    w = u / c
    s0 = w[1:] @ rest
    spin = np.r_[s0, rest + w[1:] * s0 / (1 + w[0])]
    args = dict(
        four_velocity_mm_ns=u,
        spin_angular_momentum=spin,
        field_tensor=f,
        partial_f=df,
        **particle.coefficients(),
    )
    if particle.reaction_mode == "experimental_linear_spin":
        from .jakobsen_reaction import reaction_response_native

        derivative = getattr(provider, "gradient_proper_rate", None)
        if not callable(derivative):
            raise ValueError("Reaction requires provider.gradient_proper_rate(t,x,u)")
        result = reaction_response_native(
            **args, partial_f_proper_rate=derivative(state[0], state[1:4], u)
        )
        a0 = result.leading_acceleration
    else:
        result = ordinary_response_native(**args)
        a0 = (
            particle.charge_native
            / (particle.mass_amu * c)
            * f
            @ (u * np.array([1, -1, -1, -1]))
        )
    p_rate = result.four_force + particle.charge_native / c * np.einsum(
        "a,ab->b", u, da
    )
    p_rate += result.spin_momentum_offset_rate
    sd = result.spin_four_rate
    sr = (
        sd[1:]
        - a0[1:] / c * s0 / (1 + w[0])
        - w[1:] * sd[0] / (1 + w[0])
        + w[1:] * s0 * a0[0] / c / (1 + w[0]) ** 2
    )
    return np.r_[w[0], u[1:], p_rate, sr], result, u


def midpoint_step(state, proper_step_ns, *, particle, provider):
    """Pure trial step. Does not mutate accepted state or normalize away errors."""
    if not np.isfinite(proper_step_ns) or proper_step_ns <= 0:
        raise ValueError("Positive finite proper-time step required")
    state = np.asarray(state, dtype=float)
    first = canonical_rhs(state, particle=particle, provider=provider)
    middle = state + 0.5 * proper_step_ns * first
    result = state + proper_step_ns * canonical_rhs(
        middle, particle=particle, provider=provider
    )
    if not np.isfinite(result).all() or result[0] <= state[0]:
        raise ValueError("Nonfinite or non-forward canonical trial")
    return result


def canonical_constraint_residual(state, *, particle, provider):
    """Stored minus reconstructed P0; report it, do not erase it by projection."""
    u, (a, _, f, _) = state_velocity(state, particle, provider)
    expected = canonical_momentum_native(
        four_velocity_mm_ns=u,
        rest_spin_angular_momentum=state[8:11],
        four_potential=a,
        field_tensor=f,
        **particle.coefficients(),
    )
    return float(state[4] - expected[0])
