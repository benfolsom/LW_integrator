"""Finite-size spinning-shell angular-momentum and harmonic-response oracles.

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

The exact harmonic response follows Mansuripur and Jakobsen,

    https://doi.org/10.1117/12.2569137
    https://arxiv.org/abs/2008.11264

for the same uniformly charged shell.  Their symbol for magnetic moment
includes a factor of vacuum permeability.  This module instead returns the
ordinary SI/native magnetic moment used throughout the integrator, so that
the point-dipole power can be compared directly with the radiation-sphere
oracle.

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
# The native Gaussian-to-SI field bridge is defined with exact c and the
# conventional pre-2019 SI electromagnetic relation.  Use its corresponding
# permeability here so native sphere flux and SI shell formulas describe the
# same unit system.  The modern measured mu_0 differs by about 1.3e-10.
_MU_0_SI = 4.0 * np.pi * 1.0e-7
_NS_PER_S = 1.0e9
_Z_0_SI = _MU_0_SI * _C_M_S
_NATIVE_POWER_UNIT_W = NATIVE_ENERGY_UNIT_J * _NS_PER_S

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


@dataclass(frozen=True)
class HarmonicSpinningShellResponseResult:
    r"""Exact frequency-domain response of one uniformly charged shell.

    ``radiation_reaction_coefficient_native`` is the complex coefficient
    :math:`\Gamma` in Mansuripur--Jakobsen Eq. (18), expressed in native
    action units.  With the convention
    :math:`\Omega(t)=\operatorname{Re}[\Omega_0e^{-i\omega t}]`, the complex
    self-torque amplitude is :math:`i\Gamma\Omega_0`.

    The real part of :math:`\Gamma` is in phase quadrature with the angular
    velocity and represents reversible electromagnetic inertia.  Its
    imaginary part produces the negative mean self-work that balances the
    outward radiated power.
    """

    dimensionless_frequency: float
    maximum_surface_beta: float
    magnetic_moment_amplitude_native: complex
    radiation_reaction_coefficient_native: complex
    self_torque_amplitude_native: complex
    average_self_torque_work_rate_native: float
    radiated_power_native: float
    point_dipole_radiated_power_native: float
    finite_size_power_ratio: float
    average_power_balance_residual_native: float


def _finite_complex(value: complex, *, name: str) -> complex:
    result = complex(value)
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f"{name} must have finite real and imaginary parts")
    return result


def _sin_minus_x_cos_over_x_cubed(value: float) -> float:
    """Return ``(sin(x) - x*cos(x))/x**3`` without small-x cancellation."""

    x = float(value)
    if abs(x) < 1.0e-3:
        x2 = x * x
        return (
            1.0 / 3.0
            + x2
            * (
                -1.0 / 30.0
                + x2
                * (
                    1.0 / 840.0
                    + x2 * (-1.0 / 45360.0 + x2 / 3991680.0)
                )
            )
        )
    return (np.sin(x) - x * np.cos(x)) / x**3


def evaluate_harmonic_spinning_shell_response_native(
    *,
    charge_native: float,
    shell_radius_mm: float,
    drive_angular_frequency_per_ns: float,
    angular_velocity_amplitude_per_ns: complex,
) -> HarmonicSpinningShellResponseResult:
    """Evaluate the exact harmonic self-torque and power of a charged shell.

    This implements Mansuripur--Jakobsen Eqs. (13), (15), and (18).  The
    shell rotates about a fixed axis with complex angular-velocity amplitude
    ``angular_velocity_amplitude_per_ns`` and real positive drive frequency.
    Complex amplitudes use the time convention ``exp(-i*omega*t)``.

    The shell model is nonrelativistic internally.  ``maximum_surface_beta``
    is returned so callers can reject an angular-velocity amplitude for which
    the equator would approach the speed of light.  The function itself does
    not impose a problem-specific cutoff.

    ``radiated_power_native`` and ``average_self_torque_work_rate_native``
    are independently evaluated from the far-field power and the complex
    self-torque.  Their sum is the reported balance residual.
    """

    charge_c = _charge_coulomb(charge_native)
    radius_m = _positive_length_m(shell_radius_mm, name="shell_radius_mm")
    frequency_per_ns = float(drive_angular_frequency_per_ns)
    if not np.isfinite(frequency_per_ns) or frequency_per_ns <= 0.0:
        raise ValueError("drive_angular_frequency_per_ns must be finite and positive")
    angular_velocity_per_ns = _finite_complex(
        angular_velocity_amplitude_per_ns,
        name="angular_velocity_amplitude_per_ns",
    )

    frequency_per_s = frequency_per_ns * _NS_PER_S
    angular_velocity_per_s = angular_velocity_per_ns * _NS_PER_S
    x = frequency_per_s * radius_m / _C_M_S
    g_over_x_cubed = _sin_minus_x_cos_over_x_cubed(x)

    # Eq. (18), written as g(x)/x^2 = x*g(x)/x^3.  This form retains the
    # correct point limit without subtracting nearly equal sin/cos terms.
    gamma_prefactor_si = _Z_0_SI * charge_c**2 / (6.0 * np.pi)
    cosine_sine_factor = np.cos(x) + x * np.sin(x)
    gamma_si = gamma_prefactor_si * (
        x * g_over_x_cubed * cosine_sine_factor
        + 1.0j * x**4 * g_over_x_cubed**2
    )
    self_torque_amplitude_si = 1.0j * gamma_si * angular_velocity_per_s

    # The paper's moment includes mu_0.  The standard SI moment needed by the
    # integrator is q R^2 Omega / 3, in A m^2 = J/T.
    moment_amplitude_si = charge_c * radius_m**2 * angular_velocity_per_s / 3.0
    moment_native_per_si = 1.0 / magnetic_moment_native_to_j_per_t(1.0)
    moment_amplitude_native = moment_amplitude_si * moment_native_per_si

    radiated_power_si = (
        _Z_0_SI
        * charge_c**2
        * abs(angular_velocity_per_s) ** 2
        / (12.0 * np.pi)
        * x**4
        * g_over_x_cubed**2
    )
    point_power_si = (
        _MU_0_SI
        * abs(moment_amplitude_si) ** 2
        * frequency_per_s**4
        / (12.0 * np.pi * _C_M_S**3)
    )
    # Algebraically this is 0.5*Re(T_0*conj(Omega_0)).  Evaluating that
    # product directly can subtract the much larger reversible quadrature
    # components when Omega_0 carries an arbitrary complex phase.
    average_self_work_si = (
        -0.5 * gamma_si.imag * abs(angular_velocity_per_s) ** 2
    )

    radiated_power_native = radiated_power_si / _NATIVE_POWER_UNIT_W
    self_work_native = average_self_work_si / _NATIVE_POWER_UNIT_W
    point_power_native = point_power_si / _NATIVE_POWER_UNIT_W
    return HarmonicSpinningShellResponseResult(
        dimensionless_frequency=x,
        maximum_surface_beta=(
            abs(angular_velocity_per_s) * radius_m / _C_M_S
        ),
        magnetic_moment_amplitude_native=moment_amplitude_native,
        radiation_reaction_coefficient_native=(
            gamma_si / NATIVE_ACTION_UNIT_J_S
        ),
        self_torque_amplitude_native=(
            self_torque_amplitude_si / NATIVE_ENERGY_UNIT_J
        ),
        average_self_torque_work_rate_native=self_work_native,
        radiated_power_native=radiated_power_native,
        point_dipole_radiated_power_native=point_power_native,
        finite_size_power_ratio=(3.0 * g_over_x_cubed) ** 2,
        average_power_balance_residual_native=(
            self_work_native + radiated_power_native
        ),
    )


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
    "HarmonicSpinningShellResponseResult",
    "SpinningShellAngularBalanceResult",
    "SpinningShellLocalTorqueResult",
    "evaluate_harmonic_spinning_shell_response_native",
    "evaluate_spinning_shell_angular_balance_native",
    "evaluate_spinning_shell_local_self_torque_native",
]
