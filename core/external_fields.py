"""Utilities for prescribed external electromagnetic fields.

The maintained integrator's native unit system is not SI or raw cgs. External
field configs therefore store their base field strengths in native
force-per-charge units. Use :func:`electric_field_v_per_m_to_native` and
:func:`magnetic_field_tesla_to_native` when starting from SI fields. The
optional magnetic-field gradient is stored directly in T/m.
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
C_M_PER_S = 299_792_458.0

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


def electric_field_native_to_v_per_m(value_native: float) -> float:
    """Convert a solver-native electric field component to V/m."""
    force_newton = float(value_native) * ELEMENTARY_CHARGE * NATIVE_FORCE_UNIT_NEWTON
    return force_newton / ELEMENTARY_CHARGE_COULOMB


def magnetic_field_tesla_to_native(value_tesla: float) -> float:
    """Convert tesla to the native field used by ``beta x B``."""
    force_newton = ELEMENTARY_CHARGE_COULOMB * C_M_PER_S * float(value_tesla)
    force_native = force_newton / NATIVE_FORCE_UNIT_NEWTON
    return force_native / ELEMENTARY_CHARGE


def magnetic_field_native_to_tesla(value_native: float) -> float:
    """Convert a native magnetic field component to tesla."""
    force_newton = float(value_native) * ELEMENTARY_CHARGE * NATIVE_FORCE_UNIT_NEWTON
    return force_newton / (ELEMENTARY_CHARGE_COULOMB * C_M_PER_S)


def evaluate_external_field_si(
    external_field: ExternalFieldConfig,
    *,
    position_mm: Tuple[float, float, float],
    time_ns: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local ``(E [V/m], B [T], dB_i/dx_j [T/m])``.

    The configured magnetic field is the value at the coordinate origin.  A
    supplied gradient is applied linearly to the position.  Outside the field
    window all three results are zero.
    """
    x, y, z = (float(value) for value in position_mm)
    if not external_field.is_active(x, y, z, float(time_ns)):
        return np.zeros(3), np.zeros(3), np.zeros((3, 3))
    electric_v_m = np.asarray(
        [
            electric_field_native_to_v_per_m(value)
            for value in external_field.electric_field_native
        ],
        dtype=float,
    )
    magnetic_t = np.asarray(
        [
            magnetic_field_native_to_tesla(value)
            for value in external_field.magnetic_field_native
        ],
        dtype=float,
    )
    gradient = np.asarray(external_field.magnetic_field_gradient_t_per_m, dtype=float)
    position_m = np.asarray((x, y, z), dtype=float) * 1.0e-3
    magnetic_t = magnetic_t + gradient @ position_m
    return electric_v_m, magnetic_t, gradient


def evaluate_external_field_native(
    external_field: ExternalFieldConfig,
    *,
    position_mm: Tuple[float, float, float],
    time_ns: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local native ``(E, B, dB_i/dx_j)`` at one solver event.

    Base fields already stored in native units pass through exactly. The
    user-facing T/m gradient crosses the SI boundary once, becoming native
    magnetic field per millimetre before it is applied to the native position.
    """

    x, y, z = (float(value) for value in position_mm)
    if not external_field.is_active(x, y, z, float(time_ns)):
        return np.zeros(3), np.zeros(3), np.zeros((3, 3))
    electric_native = np.asarray(
        external_field.electric_field_native, dtype=float
    ).copy()
    magnetic_native = np.asarray(
        external_field.magnetic_field_native, dtype=float
    ).copy()
    gradient_t_per_m = np.asarray(
        external_field.magnetic_field_gradient_t_per_m, dtype=float
    )
    gradient_native_per_mm = np.asarray(
        [
            [magnetic_field_tesla_to_native(value) * 1.0e-3 for value in row]
            for row in gradient_t_per_m
        ],
        dtype=float,
    )
    magnetic_native += gradient_native_per_mm @ np.asarray((x, y, z), dtype=float)
    return electric_native, magnetic_native, gradient_native_per_mm


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
    x, y, z = (float(value) for value in position)
    if not external_field.is_active(x, y, z, float(time)):
        return 0.0, 0.0, 0.0, 0.0
    # Keep the legacy uniform-field path bitwise stable. Only the new linear
    # gradient contribution crosses the SI/native unit bridge.
    electric = np.asarray(external_field.electric_field_native, dtype=float)
    magnetic = np.asarray(external_field.magnetic_field_native, dtype=float).copy()
    gradient = np.asarray(external_field.magnetic_field_gradient_t_per_m, dtype=float)
    gradient_field_t = gradient @ (np.asarray((x, y, z), dtype=float) * 1.0e-3)
    magnetic += np.asarray(
        [magnetic_field_tesla_to_native(value) for value in gradient_field_t],
        dtype=float,
    )
    if not np.any(electric) and not np.any(magnetic):
        return 0.0, 0.0, 0.0, 0.0
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
    "C_M_PER_S",
    "compute_uniform_external_field_impulse",
    "electric_field_v_per_m_to_native",
    "electric_field_native_to_v_per_m",
    "evaluate_external_field_native",
    "evaluate_external_field_si",
    "magnetic_field_native_to_tesla",
    "magnetic_field_tesla_to_native",
]
