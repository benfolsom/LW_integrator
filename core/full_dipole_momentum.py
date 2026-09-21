"""Fixed-time current accounting for a full charge/dipole source, c=1.

For j^nu = q v^nu delta + partial_a(B^{nu a} delta), B=D/u^0,
the fixed-time Lorentz force is the point force plus d(F.B[:,0])/dt.
This endpoint term must accompany particle momentum in a field balance.
No volume integral or self-field regularization is performed here.
"""

import numpy as np
from typing import Any

from .momentum_center import METRIC, contract


def fixed_time_current_balance(
    velocity: np.ndarray,
    dipole_density: np.ndarray,
    density_rate: np.ndarray,
    field: np.ndarray,
    gradient: np.ndarray,
    *,
    charge: float,
) -> dict[str, Any]:
    """Independent weak-current integral and dipole-gradient force comparison.

    velocity=[1,beta]; B=dipole_density=D/u^0 and density_rate=dB/d(ct).
    All field derivatives are partial derivatives in (ct,x,y,z), c=1 here.
    The endpoint is slice-dependent; it is not asserted to transform alone as
    a four-vector independently of the chosen simultaneity hypersurface.
    """
    v, b, bd, f, df = (
        np.asarray(x, dtype=float)
        for x in (velocity, dipole_density, density_rate, field, gradient)
    )
    if any(
        x.shape != shape or not np.isfinite(x).all()
        for x, shape in zip(
            (v, b, bd, f, df), ((4,), (4, 4), (4, 4), (4, 4), (4, 4, 4))
        )
    ) or not np.isfinite(charge):
        raise ValueError("Finite fixed-time current and field response required")
    if v[0] != 1 or np.dot(v[1:], v[1:]) >= 1:
        raise ValueError("Subluminal coordinate velocity [1,beta] required")
    for value in (b, bd, f):
        if not np.allclose(value, -value.T, rtol=0, atol=1e-14):
            raise ValueError("Antisymmetric dipole, dipole rate and field required")
    mixed = f * METRIC[None, :]
    charge_force = charge * mixed @ v
    current_force = charge_force + mixed @ bd[:, 0]
    for i in range(1, 4):
        # d_t delta(x-z(t))=-v^i d_i delta: integration by parts gives
        # a POSITIVE v^i d_i F contribution from the moving time dipole.
        current_force += df[i] @ (METRIC * (v[i] * b[:, 0] - b[:, i]))
    point_force = charge_force + 0.5 * METRIC * np.array(
        [contract(entry, b) for entry in df]
    )
    endpoint = mixed @ b[:, 0]
    endpoint_rate = (
        np.einsum("a,aij->ij", v, df) @ (METRIC * b[:, 0]) + mixed @ bd[:, 0]
    )
    return dict(
        point_force=point_force,
        fixed_time_current_force=current_force,
        endpoint_momentum=endpoint,
        endpoint_momentum_rate=endpoint_rate,
        residual=current_force - point_force - endpoint_rate,
        scope="Local fixed-time source identity only, not particle-plus-field conservation",
    )
