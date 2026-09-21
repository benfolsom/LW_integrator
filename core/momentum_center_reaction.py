"""Reduced-order charge reaction for the zero-spin momentum-center limit.

Length-time Gaussian units, c=1, (+---). This is Landau--Lifshitz reduction
of the charge LAD force, equivalent to Medina's reduced-order charge result
on a Lorentz-force trajectory. It is not a finite-size convolution, nor a
finite-spin charge/dipole reaction prescription.

Reference: Medina (2006), Eqs. (95), (97)--(100),
https://arxiv.org/abs/physics/0508031.
"""

import numpy as np

from .momentum_center import METRIC, dot


def charge_reaction(
    velocity: np.ndarray,
    field: np.ndarray,
    gradient: np.ndarray,
    *,
    charge: float,
    mass: float,
) -> dict[str, np.ndarray | float | str]:
    """Evaluate reaction, outward flux and bound momentum from one response.

    All derivatives use the *ordinary* Lorentz motion (reduction of order).
    The balance identity is local at this retained reaction order, not a
    measured field integral on a self-consistent radiating trajectory.
    """
    u, f, df = (np.asarray(v, dtype=float) for v in (velocity, field, gradient))
    if (
        u.shape != (4,)
        or f.shape != (4, 4)
        or df.shape != (4, 4, 4)
        or not all(np.isfinite(v).all() for v in (u, f, df))
        or not np.isfinite([charge, mass]).all()
        or mass <= 0
    ):
        raise ValueError(
            "Finite velocity, field, gradient, charge and positive mass required"
        )
    if u[0] <= 0 or abs(dot(u, u) - 1) > 1e-12 * max(1.0, np.dot(u, u)):
        raise ValueError("Future normalized four-velocity required")
    if not np.allclose(f, -f.T, rtol=0, atol=1e-14) or not np.allclose(
        df, -df.swapaxes(1, 2), rtol=0, atol=1e-14
    ):
        raise ValueError("Antisymmetric field and field-gradient tensors required")
    a = charge / mass * (f @ (METRIC * u))
    field_rate = np.einsum("a,aij->ij", u, df)
    jerk = charge / mass * (field_rate @ (METRIC * u) + f @ (METRIC * a))
    coefficient = 2 * charge**2 / 3
    radiation = -coefficient * dot(a, a) * u
    bound_rate = -coefficient * jerk
    reaction = -radiation - bound_rate
    return dict(
        mode="charge_ll",
        force=reaction,
        outward_radiation_rate=radiation,
        bound_momentum=-coefficient * a,
        bound_momentum_rate=bound_rate,
        ordinary_acceleration=a,
        ordinary_jerk=jerk,
        local_balance_residual=reaction + radiation + bound_rate,
        velocity_dot_force=dot(u, reaction),
        response_time_length=coefficient / mass,
        approximation="Reduced order in charge reaction strength; zero spin only",
    )
