"""Finite-size spinning-shell angular-momentum oracle.

This module implements the slow-variation expansion derived by Bonga,
Poisson, and Yang for a uniformly charged, infinitesimally thin spherical
shell whose magnetic moment changes along a fixed axis:

    https://doi.org/10.1119/1.5054590
    https://arxiv.org/abs/1805.01372

The result is an independent diagnostic model.  It is not a point-particle
self-torque and it is not applied by the equations of motion.  In particular,
the benchmark is linear in charge and magnetic moment, so it belongs to the
``q mu`` interference sector.  It must not be described as a pure ``mu^2``
magnetic-dipole self-reaction law.

Inputs use the solver's native units.  ``moment_derivatives_native[n]`` means
the ``n``th coordinate-time derivative of the signed axial magnetic moment,
with time measured in nanoseconds.  Returned torques use native energy units;
returned field angular momentum uses native action units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import C_MMNS, ELEMENTARY_CHARGE
from .external_fields import ELEMENTARY_CHARGE_COULOMB
from .magnetic_dipole import (
    NATIVE_ACTION_UNIT_J_S,
    NATIVE_ENERGY_UNIT_J,
    magnetic_moment_native_to_j_per_t,
)

_C_M_S = C_MMNS * 1.0e6
_MU_0_SI = 1.256_637_061_27e-6
_NS_PER_S = 1.0e9

# Coefficients in Eqs. (33), (36), and (42), indexed by the even power of
# tau=R/c retained in the finite-size expansion.
_RETARDED_SERIES = (
    (0, 1.0),
    (2, 1.0 / 10.0),
    (4, 1.0 / 280.0),
    (6, 1.0 / 15120.0),
)

# Eq. (34), expanded at the shell's current time.  Even powers of tau are
# time-symmetric.  Odd powers change sign under retarded <-> advanced boundary
# conditions and are the radiation-reaction part of the shell self-torque.
_LOCAL_SELF_TORQUE_SERIES = (
    (0, 1, 1.0),
    (2, 3, -2.0 / 5.0),
    (3, 4, 1.0 / 3.0),
    (4, 5, -6.0 / 35.0),
    (5, 6, 1.0 / 15.0),
    (6, 7, -4.0 / 189.0),
    (7, 8, 1.0 / 175.0),
)


def _derivatives_si_per_s(
    derivatives_native: Sequence[float], *, name: str
) -> np.ndarray:
    derivatives = np.asarray(derivatives_native, dtype=float)
    if derivatives.ndim != 1 or derivatives.size < 9:
        raise ValueError(f"{name} must contain derivatives 0 through 8")
    if not np.all(np.isfinite(derivatives)):
        raise ValueError(f"{name} must contain only finite values")
    moment_scale = magnetic_moment_native_to_j_per_t(1.0)
    orders = np.arange(derivatives.size, dtype=float)
    return derivatives * moment_scale * np.power(_NS_PER_S, orders)


def _charge_coulomb(charge_native: float) -> float:
    charge = float(charge_native)
    if not np.isfinite(charge):
        raise ValueError("charge_native must be finite")
    return charge * ELEMENTARY_CHARGE_COULOMB / ELEMENTARY_CHARGE


def _positive_length_m(value_mm: float, *, name: str) -> float:
    value = float(value_mm)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value * 1.0e-3


@dataclass(frozen=True)
class SpinningShellAngularBalanceResult:
    """Finite-shell angular-momentum ledger at one observation time.

    The conservation convention is

    ``field rate + outward flux + self-torque = residual``.

    The first contribution to ``field_angular_momentum_native`` comes from
    the near field around the shell.  The second is the wave-zone boundary
    contribution inside the selected observation sphere.
    """

    self_torque_native: float
    outward_angular_momentum_rate_native: float
    near_field_angular_momentum_native: float
    wave_zone_angular_momentum_native: float
    field_angular_momentum_native: float
    field_angular_momentum_rate_native: float
    balance_residual_native: float
    shell_light_crossing_time_ns: float


@dataclass(frozen=True)
class SpinningShellLocalTorqueResult:
    """Current-time shell self-torque split by time-reversal behavior."""

    total_self_torque_native: float
    time_symmetric_torque_native: float
    radiation_reaction_torque_native: float
    shell_light_crossing_time_ns: float


def evaluate_spinning_shell_angular_balance_native(
    *,
    charge_native: float,
    shell_radius_mm: float,
    observation_radius_mm: float,
    shell_retarded_moment_derivatives_native: Sequence[float],
    observation_retarded_moment_derivatives_native: Sequence[float],
) -> SpinningShellAngularBalanceResult:
    """Evaluate Bonga--Poisson--Yang Eqs. (33), (36), and (42).

    The two derivative arrays must be evaluated at different source times:
    ``t-R/c`` for ``shell_retarded_*`` and ``t-r0/c`` for
    ``observation_retarded_*``.  Each array contains derivatives zero through
    eight of the signed axial magnetic moment.

    The expansion assumes that the shell light-crossing time is much shorter
    than the magnetic-moment variation time.  Callers must demonstrate
    convergence with shell radius and retained derivative order before using
    it as a physical approximation.
    """

    charge_c = _charge_coulomb(charge_native)
    shell_radius_m = _positive_length_m(shell_radius_mm, name="shell_radius_mm")
    observation_radius_m = _positive_length_m(
        observation_radius_mm, name="observation_radius_mm"
    )
    if observation_radius_m <= shell_radius_m:
        raise ValueError("observation_radius_mm must exceed shell_radius_mm")

    shell = _derivatives_si_per_s(
        shell_retarded_moment_derivatives_native,
        name="shell_retarded_moment_derivatives_native",
    )
    observer = _derivatives_si_per_s(
        observation_retarded_moment_derivatives_native,
        name="observation_retarded_moment_derivatives_native",
    )
    tau_s = shell_radius_m / _C_M_S
    coefficient = _MU_0_SI * charge_c / (6.0 * np.pi)

    shell_torque_series = 0.0
    outward_series = 0.0
    near_field_series = 0.0
    wave_zone_series = 0.0
    near_field_rate_series = 0.0
    wave_zone_rate_series = 0.0
    for even_power, series_coefficient in _RETARDED_SERIES:
        tau_factor = series_coefficient * tau_s**even_power
        shell_torque_series += tau_factor * (
            shell[even_power + 1] + tau_s * shell[even_power + 2]
        )
        outward_series += tau_factor * observer[even_power + 2]
        near_field_series += tau_factor * (
            shell[even_power] + tau_s * shell[even_power + 1]
        )
        wave_zone_series += tau_factor * observer[even_power + 1]
        near_field_rate_series += tau_factor * (
            shell[even_power + 1] + tau_s * shell[even_power + 2]
        )
        wave_zone_rate_series += tau_factor * observer[even_power + 2]

    self_torque_si = -coefficient * shell_torque_series / shell_radius_m
    outward_rate_si = coefficient * outward_series / _C_M_S
    near_field_si = coefficient * near_field_series / shell_radius_m
    wave_zone_si = -coefficient * wave_zone_series / _C_M_S
    field_rate_si = coefficient * (
        near_field_rate_series / shell_radius_m
        - wave_zone_rate_series / _C_M_S
    )

    self_torque = self_torque_si / NATIVE_ENERGY_UNIT_J
    outward_rate = outward_rate_si / NATIVE_ENERGY_UNIT_J
    field_rate = field_rate_si / NATIVE_ENERGY_UNIT_J
    near_field = near_field_si / NATIVE_ACTION_UNIT_J_S
    wave_zone = wave_zone_si / NATIVE_ACTION_UNIT_J_S
    return SpinningShellAngularBalanceResult(
        self_torque_native=self_torque,
        outward_angular_momentum_rate_native=outward_rate,
        near_field_angular_momentum_native=near_field,
        wave_zone_angular_momentum_native=wave_zone,
        field_angular_momentum_native=near_field + wave_zone,
        field_angular_momentum_rate_native=field_rate,
        balance_residual_native=field_rate + outward_rate + self_torque,
        shell_light_crossing_time_ns=(shell_radius_m / _C_M_S * _NS_PER_S),
    )


def evaluate_spinning_shell_local_self_torque_native(
    *,
    charge_native: float,
    shell_radius_mm: float,
    current_moment_derivatives_native: Sequence[float],
) -> SpinningShellLocalTorqueResult:
    """Evaluate and classify the current-time self-torque, Eq. (34).

    Terms with even powers of ``R/c`` are unchanged when retarded fields are
    replaced by advanced fields.  They alter electromagnetic inertia or store
    and return angular momentum, but are not radiation reaction.  Terms with
    odd powers change sign and form the radiation-reaction contribution.
    """

    charge_c = _charge_coulomb(charge_native)
    shell_radius_m = _positive_length_m(shell_radius_mm, name="shell_radius_mm")
    derivatives = _derivatives_si_per_s(
        current_moment_derivatives_native,
        name="current_moment_derivatives_native",
    )
    tau_s = shell_radius_m / _C_M_S
    coefficient = -_MU_0_SI * charge_c / (6.0 * np.pi * shell_radius_m)
    symmetric_series = 0.0
    reaction_series = 0.0
    for tau_power, derivative_order, series_coefficient in _LOCAL_SELF_TORQUE_SERIES:
        term = (
            series_coefficient
            * tau_s**tau_power
            * derivatives[derivative_order]
        )
        if tau_power % 2 == 0:
            symmetric_series += term
        else:
            reaction_series += term

    symmetric = coefficient * symmetric_series / NATIVE_ENERGY_UNIT_J
    reaction = coefficient * reaction_series / NATIVE_ENERGY_UNIT_J
    return SpinningShellLocalTorqueResult(
        total_self_torque_native=symmetric + reaction,
        time_symmetric_torque_native=symmetric,
        radiation_reaction_torque_native=reaction,
        shell_light_crossing_time_ns=(shell_radius_m / _C_M_S * _NS_PER_S),
    )


__all__ = [
    "SpinningShellAngularBalanceResult",
    "SpinningShellLocalTorqueResult",
    "evaluate_spinning_shell_angular_balance_native",
    "evaluate_spinning_shell_local_self_torque_native",
]
