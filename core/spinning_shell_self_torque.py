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
from .external_fields import AMU_KG, ELEMENTARY_CHARGE_COULOMB
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


@dataclass(frozen=True)
class HarmonicSpinningShellTransferResult:
    """Complex-frequency shell transfer function at one frequency."""

    response_model: str
    complex_angular_frequency_per_ns: complex
    dimensionless_complex_frequency: complex
    mechanical_moment_of_inertia_kg_m2: float
    radiation_reaction_coefficient_native: complex
    denominator_native: complex
    angular_velocity_per_torque_native: complex


@dataclass(frozen=True)
class HarmonicSpinningShellPoleCountResult:
    """Argument-principle zero count inside one dimensionless rectangle.

    Zeros of the transfer-function denominator are poles of the response.
    The result certifies only the finite rectangle supplied by the caller.
    """

    response_model: str
    real_dimensionless_bounds: tuple[float, float]
    imaginary_dimensionless_bounds: tuple[float, float]
    samples_per_edge: int
    zero_count: int
    winding_number: float
    winding_rounding_residual: float
    minimum_denominator_magnitude_native: float


@dataclass(frozen=True)
class HarmonicSpinningShellImpulseResponseResult:
    """Finite-window reconstruction of the normalized impulse response.

    The response is ``I*Omega(t)/L_impulse``, where ``L_impulse`` is the
    applied torque impulse.  Time is reported as the dimensionless variable
    ``c*t/R``.  A bare rigid shell without self-field effects would therefore
    respond as ``exp(-b*c*t/R)`` for positive time, with
    ``b=beta*R/(I*c)``.
    """

    response_model: str
    dimensionless_times: np.ndarray
    normalized_angular_velocity_response: np.ndarray
    maximum_imaginary_residual: float
    maximum_preimpulse_absolute_response: float
    max_abs_dimensionless_frequency: float
    frequency_sample_count: int
    dimensionless_frequency_step: float
    dimensionless_friction: float
    inertial_reference_subtracted: bool


@dataclass(frozen=True)
class NeutralCounterRotatingShellResponseResult:
    """Explicit neutral two-shell realization of the harmonic source.

    The two nearly coincident shells carry charges ``(+q/2, -q/2)`` and
    rotate with amplitudes ``(+Omega, -Omega)``.  Their net charge vanishes,
    while their ordinary magnetic moments have the same sign and add.
    Mansuripur--Jakobsen show that the collective equation of motion is the
    same as for their one-shell parameterization using ``q``, total mass, and
    ``Omega``.
    """

    shell_charges_native: tuple[float, float]
    shell_masses_amu: tuple[float, float]
    shell_angular_velocity_amplitudes_per_ns: tuple[complex, complex]
    shell_magnetic_moment_amplitudes_native: tuple[complex, complex]
    net_charge_native: float
    total_magnetic_moment_amplitude_native: complex
    effective_one_shell_response: HarmonicSpinningShellResponseResult


@dataclass(frozen=True)
class NeutralSpinningShellPulseEnergyBalanceResult:
    """Frequency-resolved energy ledger for a prescribed neutral-shell pulse.

    The sampled angular velocity is periodically extended by the discrete
    Fourier transform.  A physical one-shot pulse should therefore include a
    quiescent buffer at both ends.  The two reported boundary diagnostics let
    callers check that assumption and that radiation near the Nyquist limit is
    negligible.

    This result establishes an integrated ``mu^2`` dissipation balance for
    the explicit counter-rotating two-shell source.  It does not identify a
    time-local bound energy or define a point-particle self-torque.
    """

    sample_count: int
    sample_interval_ns: float
    observation_window_ns: float
    maximum_surface_beta: float
    maximum_boundary_angular_velocity_fraction: float
    nyquist_radiated_energy_fraction: float
    self_torque_work_native: float
    radiated_energy_native: float
    point_dipole_radiated_energy_native: float
    energy_balance_residual_native: float


def _finite_complex(value: complex, *, name: str) -> complex:
    result = complex(value)
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f"{name} must have finite real and imaginary parts")
    return result


def _sin_minus_x_cos_over_x_cubed(value: float) -> float:
    """Return ``(sin(x) - x*cos(x))/x**3`` without small-x cancellation."""

    x = float(value)
    if abs(x) < 0.25:
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


def _exact_radiation_reaction_shape(value: complex) -> complex:
    """Return the analytic dimensionless factor multiplying ``Z0*q^2/6pi``."""

    z = complex(value)
    if abs(z) < 0.25:
        # Taylor series of
        # (sin(z)-z*cos(z))*(1-i*z)*exp(i*z)/z^2.  Retaining both the
        # reversible and dissipative terms is essential for the pole test.
        return (
            z / 3.0
            + 2.0 * z**3 / 15.0
            + 1.0j * z**4 / 9.0
            - 2.0 * z**5 / 35.0
            - 1.0j * z**6 / 45.0
            + 4.0 * z**7 / 567.0
            + 1.0j * z**8 / 525.0
            - 2.0 * z**9 / 4455.0
            - 4.0j * z**10 / 42525.0
            + 4.0 * z**11 / 225225.0
            + 2.0j * z**12 / 654885.0
            - 4.0 * z**13 / 8292375.0
            - 1.0j * z**14 / 14189175.0
        )
    return (
        0.5
        * (1.0 - 1.0j * z)
        / z**2
        * ((1.0j - z) - (1.0j + z) * np.exp(2.0j * z))
    )


def _approximate_radiation_reaction_shape(value: complex) -> complex:
    """Return the small-radius truncation in Mansuripur--Jakobsen Eq. (19)."""

    z = complex(value)
    return (z + 2.0 * z**3 / 5.0 + 1.0j * z**4 / 3.0) / 3.0


def _radiation_reaction_coefficient_si(
    *,
    charge_coulomb: float,
    dimensionless_frequency: complex,
    response_model: str,
) -> complex:
    if response_model == "exact":
        shape = _exact_radiation_reaction_shape(dimensionless_frequency)
    elif response_model == "small_radius_truncation":
        shape = _approximate_radiation_reaction_shape(dimensionless_frequency)
    else:
        raise ValueError(
            "response_model must be 'exact' or 'small_radius_truncation'"
        )
    return _Z_0_SI * charge_coulomb**2 / (6.0 * np.pi) * shape


def _positive_mass_kg(value_amu: float) -> float:
    value = float(value_amu)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("shell_mass_amu must be finite and positive")
    return value * AMU_KG


def evaluate_harmonic_spinning_shell_transfer_native(
    *,
    charge_native: float,
    shell_radius_mm: float,
    shell_mass_amu: float,
    friction_coefficient_native: float,
    complex_angular_frequency_per_ns: complex,
    response_model: str = "exact",
) -> HarmonicSpinningShellTransferResult:
    """Evaluate Mansuripur--Jakobsen Eq. (17) at complex frequency.

    ``friction_coefficient_native`` has native action units because its
    product with angular velocity is a torque.  The exact transfer is

    ``Omega_0/T_0 = i / (I*omega + Gamma(omega) + i*beta)``.

    ``small_radius_truncation`` is retained only as a diagnostic control: the
    paper shows that it introduces upper-half-plane poles and therefore an
    acausal impulse response even where it approximates real-frequency values.
    """

    charge_c = _charge_coulomb(charge_native)
    radius_m = _positive_length_m(shell_radius_mm, name="shell_radius_mm")
    mass_kg = _positive_mass_kg(shell_mass_amu)
    friction_native = float(friction_coefficient_native)
    if not np.isfinite(friction_native) or friction_native < 0.0:
        raise ValueError("friction_coefficient_native must be finite and nonnegative")
    frequency_per_ns = _finite_complex(
        complex_angular_frequency_per_ns,
        name="complex_angular_frequency_per_ns",
    )
    frequency_per_s = frequency_per_ns * _NS_PER_S
    dimensionless_frequency = frequency_per_s * radius_m / _C_M_S
    gamma_si = _radiation_reaction_coefficient_si(
        charge_coulomb=charge_c,
        dimensionless_frequency=dimensionless_frequency,
        response_model=response_model,
    )
    moment_of_inertia = 2.0 * mass_kg * radius_m**2 / 3.0
    denominator_si = (
        moment_of_inertia * frequency_per_s
        + gamma_si
        + 1.0j * friction_native * NATIVE_ACTION_UNIT_J_S
    )
    denominator_native = denominator_si / NATIVE_ACTION_UNIT_J_S
    transfer_native = (
        complex(np.inf, np.inf)
        if denominator_native == 0.0
        else 1.0j / denominator_native
    )
    return HarmonicSpinningShellTransferResult(
        response_model=response_model,
        complex_angular_frequency_per_ns=frequency_per_ns,
        dimensionless_complex_frequency=dimensionless_frequency,
        mechanical_moment_of_inertia_kg_m2=moment_of_inertia,
        radiation_reaction_coefficient_native=(
            gamma_si / NATIVE_ACTION_UNIT_J_S
        ),
        denominator_native=denominator_native,
        angular_velocity_per_torque_native=transfer_native,
    )


def _strict_bounds(
    value: Sequence[float], *, name: str
) -> tuple[float, float]:
    bounds = np.asarray(value, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError(f"{name} must contain two finite values")
    lower, upper = (float(entry) for entry in bounds)
    if lower >= upper:
        raise ValueError(f"{name} must be strictly increasing")
    return lower, upper


def count_harmonic_spinning_shell_transfer_poles_native(
    *,
    charge_native: float,
    shell_radius_mm: float,
    shell_mass_amu: float,
    friction_coefficient_native: float,
    real_dimensionless_bounds: Sequence[float],
    imaginary_dimensionless_bounds: Sequence[float],
    samples_per_edge: int = 2048,
    response_model: str = "exact",
) -> HarmonicSpinningShellPoleCountResult:
    """Count transfer-function poles inside one complex-frequency rectangle.

    The implementation applies Cauchy's argument principle to the analytic
    transfer denominator.  The contour is sampled counterclockwise.  Callers
    must repeat the calculation with denser contour sampling and expanding
    rectangles before interpreting a zero count as causality evidence.
    """

    real_lower, real_upper = _strict_bounds(
        real_dimensionless_bounds, name="real_dimensionless_bounds"
    )
    imaginary_lower, imaginary_upper = _strict_bounds(
        imaginary_dimensionless_bounds, name="imaginary_dimensionless_bounds"
    )
    sample_count = int(samples_per_edge)
    if sample_count < 16:
        raise ValueError("samples_per_edge must be at least 16")

    unit_interval = np.arange(sample_count, dtype=float) / sample_count
    bottom = (real_lower + (real_upper - real_lower) * unit_interval) + (
        1.0j * imaginary_lower
    )
    right = real_upper + 1.0j * (
        imaginary_lower
        + (imaginary_upper - imaginary_lower) * unit_interval
    )
    top = (real_upper - (real_upper - real_lower) * unit_interval) + (
        1.0j * imaginary_upper
    )
    left = real_lower + 1.0j * (
        imaginary_upper
        - (imaginary_upper - imaginary_lower) * unit_interval
    )
    contour = np.concatenate((bottom, right, top, left))

    radius_m = _positive_length_m(shell_radius_mm, name="shell_radius_mm")
    frequencies_per_ns = contour * _C_M_S / radius_m / _NS_PER_S
    denominators = np.asarray(
        [
            evaluate_harmonic_spinning_shell_transfer_native(
                charge_native=charge_native,
                shell_radius_mm=shell_radius_mm,
                shell_mass_amu=shell_mass_amu,
                friction_coefficient_native=friction_coefficient_native,
                complex_angular_frequency_per_ns=frequency,
                response_model=response_model,
            ).denominator_native
            for frequency in frequencies_per_ns
        ],
        dtype=complex,
    )
    if not np.all(np.isfinite(denominators)):
        raise ValueError("transfer denominator became nonfinite on the contour")
    closed = np.concatenate((denominators, denominators[:1]))
    phase = np.unwrap(np.angle(closed))
    winding = float((phase[-1] - phase[0]) / (2.0 * np.pi))
    zero_count = int(np.rint(winding))
    return HarmonicSpinningShellPoleCountResult(
        response_model=response_model,
        real_dimensionless_bounds=(real_lower, real_upper),
        imaginary_dimensionless_bounds=(imaginary_lower, imaginary_upper),
        samples_per_edge=sample_count,
        zero_count=zero_count,
        winding_number=winding,
        winding_rounding_residual=abs(winding - zero_count),
        minimum_denominator_magnitude_native=float(
            np.min(np.abs(denominators))
        ),
    )


def reconstruct_harmonic_spinning_shell_impulse_response_native(
    *,
    charge_native: float,
    shell_radius_mm: float,
    shell_mass_amu: float,
    friction_coefficient_native: float,
    dimensionless_times: Sequence[float],
    max_abs_dimensionless_frequency: float,
    frequency_sample_count: int,
    response_model: str = "exact",
) -> HarmonicSpinningShellImpulseResponseResult:
    """Numerically invert the finite-shell transfer function.

    The transform convention is the paper's ``exp(-i*omega*t)`` inverse.
    Positive mechanical friction is required so the response has no
    adjustable zero-frequency constant.  For the exact model, the known bare
    rigid-shell response is subtracted in frequency space and restored
    analytically in time; this removes the slowly converging Fourier
    representation of the instantaneous inertial jump.  The small-radius
    truncation falls as ``1/omega**4`` and is integrated directly.

    This is a finite-window numerical diagnostic.  Causality claims require
    convergence in both frequency limit and sampling density.
    """

    times = np.asarray(dimensionless_times, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("dimensionless_times must be a nonempty finite vector")
    frequency_limit = float(max_abs_dimensionless_frequency)
    if not np.isfinite(frequency_limit) or frequency_limit <= 0.0:
        raise ValueError(
            "max_abs_dimensionless_frequency must be finite and positive"
        )
    sample_count = int(frequency_sample_count)
    if sample_count < 257 or sample_count % 2 == 0:
        raise ValueError("frequency_sample_count must be odd and at least 257")

    radius_m = _positive_length_m(shell_radius_mm, name="shell_radius_mm")
    mass_kg = _positive_mass_kg(shell_mass_amu)
    friction_native = float(friction_coefficient_native)
    if not np.isfinite(friction_native) or friction_native <= 0.0:
        raise ValueError("friction_coefficient_native must be finite and positive")
    moment_of_inertia = 2.0 * mass_kg * radius_m**2 / 3.0
    inertial_action_si = moment_of_inertia * _C_M_S / radius_m
    inertial_action_native = inertial_action_si / NATIVE_ACTION_UNIT_J_S
    dimensionless_friction = friction_native / inertial_action_native

    dimensionless_frequencies = np.linspace(
        -frequency_limit,
        frequency_limit,
        sample_count,
    )
    frequency_step = float(
        dimensionless_frequencies[1] - dimensionless_frequencies[0]
    )
    frequencies_per_ns = (
        dimensionless_frequencies
        * _C_M_S
        / radius_m
        / _NS_PER_S
    )
    denominators = np.asarray(
        [
            evaluate_harmonic_spinning_shell_transfer_native(
                charge_native=charge_native,
                shell_radius_mm=shell_radius_mm,
                shell_mass_amu=shell_mass_amu,
                friction_coefficient_native=friction_native,
                complex_angular_frequency_per_ns=frequency,
                response_model=response_model,
            ).denominator_native
            for frequency in frequencies_per_ns
        ],
        dtype=complex,
    )
    if not np.all(np.isfinite(denominators)) or np.any(denominators == 0.0):
        raise ValueError("transfer denominator is singular or nonfinite on the grid")
    dimensionless_transfer = 1.0j * inertial_action_native / denominators

    subtract_inertial = response_model == "exact"
    if subtract_inertial:
        inertial_transfer = 1.0j / (
            dimensionless_frequencies + 1.0j * dimensionless_friction
        )
        numerical_transfer = dimensionless_transfer - inertial_transfer
    else:
        numerical_transfer = dimensionless_transfer

    complex_response = np.empty(times.size, dtype=complex)
    for index, dimensionless_time in enumerate(times):
        integrand = numerical_transfer * np.exp(
            -1.0j * dimensionless_frequencies * dimensionless_time
        )
        integral = frequency_step * (
            0.5 * integrand[0]
            + np.sum(integrand[1:-1])
            + 0.5 * integrand[-1]
        ) / (2.0 * np.pi)
        if subtract_inertial:
            if dimensionless_time > 0.0:
                integral += np.exp(
                    -dimensionless_friction * dimensionless_time
                )
            elif dimensionless_time == 0.0:
                integral += 0.5
        complex_response[index] = integral

    response = np.asarray(complex_response.real, dtype=float)
    response.setflags(write=False)
    times = times.copy()
    times.setflags(write=False)
    preimpulse = response[times < 0.0]
    return HarmonicSpinningShellImpulseResponseResult(
        response_model=response_model,
        dimensionless_times=times,
        normalized_angular_velocity_response=response,
        maximum_imaginary_residual=float(
            np.max(np.abs(complex_response.imag))
        ),
        maximum_preimpulse_absolute_response=(
            float(np.max(np.abs(preimpulse))) if preimpulse.size else 0.0
        ),
        max_abs_dimensionless_frequency=frequency_limit,
        frequency_sample_count=sample_count,
        dimensionless_frequency_step=frequency_step,
        dimensionless_friction=dimensionless_friction,
        inertial_reference_subtracted=subtract_inertial,
    )


def evaluate_neutral_counterrotating_shell_response_native(
    *,
    internal_charge_magnitude_native: float,
    total_shell_mass_amu: float,
    shell_radius_mm: float,
    drive_angular_frequency_per_ns: float,
    angular_velocity_amplitude_per_ns: complex,
) -> NeutralCounterRotatingShellResponseResult:
    """Evaluate the paper's neutral, counter-rotating two-shell construction.

    ``internal_charge_magnitude_native`` is the paper's positive parameter
    ``q``; each shell carries half that magnitude with opposite sign.  This
    function does not pretend that the neutral object is a structureless point
    dipole.  It records the internal charges, masses, rotations, and moments,
    then evaluates the exact collective response through the equivalence
    derived immediately after Mansuripur--Jakobsen Eq. (19).
    """

    charge_magnitude = float(internal_charge_magnitude_native)
    if not np.isfinite(charge_magnitude) or charge_magnitude <= 0.0:
        raise ValueError(
            "internal_charge_magnitude_native must be finite and positive"
        )
    _positive_mass_kg(total_shell_mass_amu)
    angular_velocity = _finite_complex(
        angular_velocity_amplitude_per_ns,
        name="angular_velocity_amplitude_per_ns",
    )
    response = evaluate_harmonic_spinning_shell_response_native(
        charge_native=charge_magnitude,
        shell_radius_mm=shell_radius_mm,
        drive_angular_frequency_per_ns=drive_angular_frequency_per_ns,
        angular_velocity_amplitude_per_ns=angular_velocity,
    )
    half_moment = 0.5 * response.magnetic_moment_amplitude_native
    half_mass = 0.5 * float(total_shell_mass_amu)
    return NeutralCounterRotatingShellResponseResult(
        shell_charges_native=(0.5 * charge_magnitude, -0.5 * charge_magnitude),
        shell_masses_amu=(half_mass, half_mass),
        shell_angular_velocity_amplitudes_per_ns=(
            angular_velocity,
            -angular_velocity,
        ),
        shell_magnetic_moment_amplitudes_native=(half_moment, half_moment),
        net_charge_native=0.0,
        total_magnetic_moment_amplitude_native=(2.0 * half_moment),
        effective_one_shell_response=response,
    )


def evaluate_neutral_spinning_shell_pulse_energy_balance_native(
    *,
    internal_charge_magnitude_native: float,
    shell_radius_mm: float,
    sample_times_ns: Sequence[float],
    angular_velocities_per_ns: Sequence[float],
) -> NeutralSpinningShellPulseEnergyBalanceResult:
    """Evaluate an exact finite-shell ``mu^2`` pulse-energy balance.

    This uses Mansuripur--Jakobsen's exact harmonic self-torque independently
    at every discrete Fourier frequency.  The outward energy is evaluated
    from the magnetic-dipole radiation spectrum with the finite-shell form
    factor, rather than copied from the self-torque coefficient.

    Samples must be real, finite, uniformly spaced, and must not repeat the
    periodic endpoint.  For a one-shot pulse, callers should supply zero or
    negligible angular velocity near both ends of the observation window.
    """

    charge_c = _charge_coulomb(internal_charge_magnitude_native)
    if charge_c <= 0.0:
        raise ValueError("internal_charge_magnitude_native must be positive")
    radius_m = _positive_length_m(shell_radius_mm, name="shell_radius_mm")
    times_ns = np.asarray(sample_times_ns, dtype=float)
    angular_velocity_per_ns = np.asarray(angular_velocities_per_ns, dtype=float)
    if times_ns.ndim != 1 or times_ns.size < 16:
        raise ValueError("sample_times_ns must contain at least 16 samples")
    if angular_velocity_per_ns.shape != times_ns.shape:
        raise ValueError("angular_velocities_per_ns must match sample_times_ns")
    if not np.all(np.isfinite(times_ns)) or not np.all(
        np.isfinite(angular_velocity_per_ns)
    ):
        raise ValueError("pulse samples must be finite")
    intervals_ns = np.diff(times_ns)
    if np.any(intervals_ns <= 0.0):
        raise ValueError("sample_times_ns must increase strictly")
    sample_interval_ns = float(intervals_ns[0])
    if not np.allclose(intervals_ns, sample_interval_ns, rtol=2.0e-12, atol=0.0):
        raise ValueError("sample_times_ns must be uniformly spaced")

    sample_count = times_ns.size
    sample_interval_s = sample_interval_ns / _NS_PER_S
    angular_velocity_per_s = angular_velocity_per_ns * _NS_PER_S
    frequencies_per_s = 2.0 * np.pi * np.fft.fftfreq(
        sample_count, d=sample_interval_s
    )
    angular_velocity_transform = (
        sample_interval_s * np.fft.fft(angular_velocity_per_s)
    )
    frequency_spacing_per_s = 2.0 * np.pi / (
        sample_count * sample_interval_s
    )

    gamma_imaginary_si = np.empty(sample_count)
    finite_size_power_ratio = np.empty(sample_count)
    for index, frequency_per_s in enumerate(frequencies_per_s):
        x = frequency_per_s * radius_m / _C_M_S
        gamma_imaginary_si[index] = _radiation_reaction_coefficient_si(
            charge_coulomb=charge_c,
            dimensionless_frequency=x,
            response_model="exact",
        ).imag
        finite_size_power_ratio[index] = (
            3.0 * _sin_minus_x_cos_over_x_cubed(abs(x))
        ) ** 2

    spectral_measure = frequency_spacing_per_s / (2.0 * np.pi)
    transform_norm = np.abs(angular_velocity_transform) ** 2
    self_work_si = -spectral_measure * float(
        np.sum(gamma_imaginary_si * transform_norm)
    )

    moment_transform_si = charge_c * radius_m**2 * angular_velocity_transform / 3.0
    point_spectral_energy_si = (
        _MU_0_SI
        * frequencies_per_s**4
        * np.abs(moment_transform_si) ** 2
        / (6.0 * np.pi * _C_M_S**3)
        * spectral_measure
    )
    radiated_spectrum_si = point_spectral_energy_si * finite_size_power_ratio
    radiated_energy_si = float(np.sum(radiated_spectrum_si))
    point_energy_si = float(np.sum(point_spectral_energy_si))

    peak_angular_velocity = float(np.max(np.abs(angular_velocity_per_ns)))
    boundary_fraction = (
        float(
            max(
                abs(angular_velocity_per_ns[0]),
                abs(angular_velocity_per_ns[-1]),
            )
            / peak_angular_velocity
        )
        if peak_angular_velocity > 0.0
        else 0.0
    )
    absolute_frequency = np.abs(frequencies_per_s)
    nyquist_band = absolute_frequency >= 0.9 * float(np.max(absolute_frequency))
    nyquist_fraction = (
        float(np.sum(radiated_spectrum_si[nyquist_band]) / radiated_energy_si)
        if radiated_energy_si > 0.0
        else 0.0
    )
    energy_unit = NATIVE_ENERGY_UNIT_J
    return NeutralSpinningShellPulseEnergyBalanceResult(
        sample_count=sample_count,
        sample_interval_ns=sample_interval_ns,
        observation_window_ns=sample_count * sample_interval_ns,
        maximum_surface_beta=(
            peak_angular_velocity * _NS_PER_S * radius_m / _C_M_S
        ),
        maximum_boundary_angular_velocity_fraction=boundary_fraction,
        nyquist_radiated_energy_fraction=nyquist_fraction,
        self_torque_work_native=self_work_si / energy_unit,
        radiated_energy_native=radiated_energy_si / energy_unit,
        point_dipole_radiated_energy_native=point_energy_si / energy_unit,
        energy_balance_residual_native=(self_work_si + radiated_energy_si)
        / energy_unit,
    )


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
    "HarmonicSpinningShellImpulseResponseResult",
    "HarmonicSpinningShellResponseResult",
    "HarmonicSpinningShellPoleCountResult",
    "HarmonicSpinningShellTransferResult",
    "NeutralCounterRotatingShellResponseResult",
    "NeutralSpinningShellPulseEnergyBalanceResult",
    "SpinningShellAngularBalanceResult",
    "SpinningShellLocalTorqueResult",
    "count_harmonic_spinning_shell_transfer_poles_native",
    "evaluate_harmonic_spinning_shell_response_native",
    "evaluate_harmonic_spinning_shell_transfer_native",
    "evaluate_neutral_counterrotating_shell_response_native",
    "evaluate_neutral_spinning_shell_pulse_energy_balance_native",
    "evaluate_spinning_shell_angular_balance_native",
    "evaluate_spinning_shell_local_self_torque_native",
    "reconstruct_harmonic_spinning_shell_impulse_response_native",
]
