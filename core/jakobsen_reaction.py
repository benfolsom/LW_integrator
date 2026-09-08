"""Experimental order-reduced charge and intrinsic linear-spin reaction.

No susceptibility or moment-squared radiation is included. Derivatives of
the supplied external potential must describe the same leading worldline.
This module never feeds a computed reaction back into Medina's input force.
"""

from dataclasses import dataclass

import numpy as np

from .constants import C_MMNS as c
from .jakobsen import ordinary_response_native, _ordinary_force_rate_parts_native
from .medina_radiation_reaction import compute_medina_radiation_reaction
from .spin_self_force_oracle import evaluate_jakobsen_linear_spin_self_force_native


@dataclass(frozen=True)
class JakobsenReactionResponse:
    four_force: np.ndarray
    spin_four_rate: np.ndarray
    spin_momentum_offset_rate: np.ndarray
    leading_charge_reaction: np.ndarray
    charge_reaction_spin_correction: np.ndarray
    intrinsic_spin_reaction: np.ndarray
    radiative_balance_correction: np.ndarray
    leading_acceleration: np.ndarray
    charge_ald_agreement_residual: np.ndarray


def _medina_four_force(u, force, force_rate, mass, charge):
    gamma = u[0] / c
    beta = u[1:] / u[0]
    acceleration = force / mass
    result = compute_medina_radiation_reaction(
        external_force=force[1:] / gamma,
        external_force_time_derivative=(
            force_rate[1:] / gamma**2 - force[1:] * acceleration[0] / (c * gamma**3)
        ),
        beta=beta,
        acceleration=(acceleration[1:] - beta * acceleration[0]) / gamma**2,
        gamma=gamma,
        mass=mass,
        charge=charge,
        coordinate_dt=0.0,
    )
    spatial = gamma * np.asarray(result.radiation_reaction_force)
    return np.r_[beta @ spatial, spatial]


def reaction_response_native(
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
    """Matched local response through first spin order and first reaction order.

    ``partial_f_proper_rate`` is d(partial_F)/d(tau), not dF/d(tau).
    Leading charge acceleration supplies derivatives inside explicit spin
    self-force terms. The separate balance correction is NOT applied as force.
    """
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
    _, rate_s = _ordinary_force_rate_parts_native(
        **args, partial_f_proper_rate=partial_f_proper_rate
    )
    u = np.asarray(four_velocity_mm_ns, dtype=float)
    spin = np.asarray(spin_angular_momentum, dtype=float)
    f = np.asarray(field_tensor, dtype=float)
    df = np.asarray(partial_f, dtype=float)
    ddf = np.asarray(partial_f_proper_rate, dtype=float)
    signs = np.array([1, -1, -1, -1])
    a = charge_native / (mass_amu * c) * f @ (signs * u)
    fd = np.einsum("a,aij->ij", u, df)
    jerk = charge_native / (mass_amu * c) * (fd @ (signs * u) + f @ (signs * a))
    fdd = np.einsum("a,aij->ij", a, df) + np.einsum("a,aij->ij", u, ddf)
    snap = (
        charge_native
        / (mass_amu * c)
        * (fdd @ (signs * u) + 2 * fd @ (signs * a) + f @ (signs * jerk))
    )
    force0 = mass_amu * a
    rate0 = mass_amu * jerk
    charge0 = _medina_four_force(u, force0, rate0, mass_amu, charge_native)
    # Medina's four-force is tau_q P dF_ext/dtau. Its linear spin part
    # follows directly, without subtracting two much larger charge forces.
    tau_q = 2 * charge_native**2 / (3 * mass_amu * c**3)
    charge_s = tau_q * (rate_s - u * ((signs * u) @ rate_s) / c**2)
    moment_coefficient = g * charge_native / (2 * mass_amu * c)
    intrinsic = evaluate_jakobsen_linear_spin_self_force_native(
        charge_native=charge_native,
        mass_amu=mass_amu,
        four_velocity_mm_ns=u,
        four_acceleration_mm_ns2=a,
        four_jerk_mm_ns3=jerk,
        four_snap_mm_ns4=snap,
        spin_four_vector_native=spin,
        spin_four_derivative_native=ordinary.spin_four_rate,
        magnetic_moment_four_vector_native=moment_coefficient * spin,
        magnetic_moment_four_derivative_native=moment_coefficient
        * ordinary.spin_four_rate,
    )
    transported = ordinary_response_native(
        **args, leading_charge_reaction_four_force_native=charge0
    )
    return JakobsenReactionResponse(
        four_force=transported.four_force
        + charge_s
        + intrinsic.linear_spin_self_force_native,
        spin_four_rate=transported.spin_four_rate,
        spin_momentum_offset_rate=transported.spin_momentum_offset_rate,
        leading_charge_reaction=charge0,
        charge_reaction_spin_correction=charge_s,
        intrinsic_spin_reaction=intrinsic.linear_spin_self_force_native,
        radiative_balance_correction=transported.charge_radiative_balance_correction,
        leading_acceleration=a + charge0 / mass_amu,
        charge_ald_agreement_residual=charge0 - intrinsic.charge_ald_self_force_native,
    )
