"""Diagnostic intrinsic-spin radiation identity on matched leading dynamics.

No returned bound-field or radiation term is an additional applied force.
This local identity does not include radiation interference between particles.
"""

import numpy as np
from .constants import C_MMNS as c
from .jakobsen import ordinary_response_native, _response_algebra
from .spin_self_force_oracle import (
    evaluate_jakobsen_intrinsic_spin_radiation_balance_native,
)


def intrinsic_balance_native(
    *,
    four_velocity_mm_ns,
    spin_angular_momentum,
    field_tensor,
    partial_f,
    partial_f_proper_rate,
    charge_native,
    mass_amu,
    g,
):
    args = dict(
        four_velocity_mm_ns=four_velocity_mm_ns,
        spin_angular_momentum=spin_angular_momentum,
        field_tensor=field_tensor,
        partial_f=partial_f,
        charge_native=charge_native,
        mass_amu=mass_amu,
        g=g,
    )
    ordinary = ordinary_response_native(**args)
    u = np.asarray(four_velocity_mm_ns)
    spin = np.asarray(spin_angular_momentum)
    f = np.asarray(field_tensor)
    df = np.asarray(partial_f)
    ddf = np.asarray(partial_f_proper_rate)
    if (
        ddf.shape != (4, 4, 4)
        or not np.isfinite(ddf).all()
        or not np.allclose(ddf, -ddf.swapaxes(1, 2), rtol=0, atol=1e-15)
    ):
        raise ValueError("Finite antisymmetric gradient rate required")
    signs = np.array([1.0, -1.0, -1.0, -1.0])
    k = charge_native / (mass_amu * c)
    a = k * f @ (signs * u)
    fd = np.einsum("a,aij->ij", u, df)
    jerk = k * (fd @ (signs * u) + f @ (signs * a))
    fdd = np.einsum("a,aij->ij", a, df) + np.einsum("a,aij->ij", u, ddf)
    snap = k * (fdd @ (signs * u) + 2 * fd @ (signs * a) + f @ (signs * jerk))
    eps = 1e-24
    spin_second = (
        _response_algebra(
            (u + 1j * eps * a) / c,
            (spin + 1j * eps * ordinary.spin_four_rate) / c,
            (f + 1j * eps * fd) / c**2,
            (df + 1j * eps * ddf) / c**2,
            charge_native,
            mass_amu,
            g,
            np.zeros(4),
        ).spin_four_rate.imag
        / eps
    )
    return evaluate_jakobsen_intrinsic_spin_radiation_balance_native(
        charge_native=charge_native,
        mass_amu=mass_amu,
        g_factor=g,
        four_velocity_mm_ns=u,
        four_acceleration_mm_ns2=a,
        four_jerk_mm_ns3=jerk,
        four_snap_mm_ns4=snap,
        spin_four_vector_native=spin,
        spin_four_derivative_native=ordinary.spin_four_rate,
        spin_four_second_derivative_native=spin_second,
    )
