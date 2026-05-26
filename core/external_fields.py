"""Utilities for prescribed external electromagnetic fields.

The maintained integrator's native unit system is not SI or raw cgs. External
field configs therefore store field strengths in native force-per-charge units.
Use :func:`electric_field_v_per_m_to_native` when starting from SI electric
field gradients.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import ELEMENTARY_CHARGE
from .types import ExternalFieldConfig

ELEMENTARY_CHARGE_COULOMB = 1.602176634e-19
AMU_KG = 1.66053906660e-27
MM_PER_M = 1.0e3
NS_PER_S = 1.0e9

# 1 native force unit = 1 amu * mm / ns^2 in SI newtons.
NATIVE_FORCE_UNIT_NEWTON = AMU_KG * (1.0 / MM_PER_M) * (NS_PER_S**2)


def electric_field_v_per_m_to_native(value_v_per_m: float) -> float:
    """Convert an SI electric field in V/m to solver-native field units.

    The conversion is defined by matching the force on one elementary charge:
    ``e_C * E_SI`` in newtons equals
    ``ELEMENTARY_CHARGE * E_native`` in ``amu * mm / ns^2``.
    """
    force_newton = ELEMENTARY_CHARGE_COULOMB * value_v_per_m
    force_native = force_newton / NATIVE_FORCE_UNIT_NEWTON
    return force_native / ELEMENTARY_CHARGE


def compute_uniform_external_field_impulse(
    external_field: ExternalFieldConfig,
    *,
    charge: float,
    gamma: float,
    beta: Tuple[float, float, float],
    h_step: float,
    position: Tuple[float, float, float],
    time: float,
) -> Tuple[float, float, float, float]:
    """Return ``(delta_Px, delta_Py, delta_Pz, delta_Pt)`` from a uniform field.

    The impulse is integrated over the proper-time step used by the integrator:

    ``delta_p = h * q * gamma * (E + beta x B)``

    ``delta_Pt = h * q * gamma * (E dot beta)``

    This is a mechanical Lorentz-force contribution inserted into the canonical
    update. It does not yet track an explicit external vector/scalar potential
    for position updates.
    """
    x, y, z = position
    if not external_field.is_active(x, y, z, time):
        return 0.0, 0.0, 0.0, 0.0

    electric = np.asarray(external_field.electric_field_native, dtype=float)
    magnetic = np.asarray(external_field.magnetic_field_native, dtype=float)
    beta_vec = np.asarray(beta, dtype=float)

    lorentz_force_per_charge = electric + np.cross(beta_vec, magnetic)
    prefactor = h_step * charge * gamma
    delta_p = prefactor * lorentz_force_per_charge
    delta_pt = prefactor * float(np.dot(electric, beta_vec))

    return (
        float(delta_p[0]),
        float(delta_p[1]),
        float(delta_p[2]),
        float(delta_pt),
    )


__all__ = [
    "AMU_KG",
    "ELEMENTARY_CHARGE_COULOMB",
    "NATIVE_FORCE_UNIT_NEWTON",
    "compute_uniform_external_field_impulse",
    "electric_field_v_per_m_to_native",
]
