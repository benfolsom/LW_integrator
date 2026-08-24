"""Pure Medina reduced-order radiation-reaction kernel in native units.

The maintained solver uses scaled Gaussian units with mass in ``amu``, length
in ``mm``, time in ``ns``, and charge scaled so that the Gaussian Coulomb law
has unit coefficient.  In these units the response time

``tau_q = 2 q**2 / (3 m c**3)``

is measured in ns.  No SI conversion factors or ``4 pi epsilon_0`` belong in
this module.

The implementation follows equations (95), (97)--(100) of R. Medina,
"Radiation reaction of a classical quasi-rigid extended particle",
J. Phys. A 39 (2006) 3801--3816,
https://doi.org/10.1088/0305-4470/39/14/021 and
https://arxiv.org/abs/physics/0508031.

In native vector notation Medina's equation (95) is

``F_R = tau_q [d(gamma F_ext)/dt
                - gamma**3 (F_ext . a) v / c**2]``.

The derivative is deliberately expanded as

``d(gamma F_ext)/dt = gamma dF_ext/dt + dgamma/dt F_ext``.

The first term must not be omitted when the applied or inter-particle field
changes along the trajectory.  The caller is responsible for supplying the
*complete lab-time derivative* of the non-radiation-reaction external force,
including explicit time dependence, motion through field gradients, and
retarded-source variation.

This kernel does not clip or otherwise modify the physical result.  A caller
that needs a numerical step guard must implement and report it separately.
It is Medina's slowly varying, physical-point-charge approximation, not the
paper's finite-size causal convolution.  It models charge radiation at order
``q**2`` only; intrinsic-dipole ``q mu`` interference and ``mu**2``
self-reaction are outside its scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Sequence, Tuple

from .constants import C_MMNS

Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class MedinaRadiationReactionResult:
    """Medina force, impulse, and energy diagnostics for one lab-time step.

    Vector units are solver-native:

    * force and radiated momentum rate: ``amu mm / ns^2``;
    * impulse and cross-field momentum: ``amu mm / ns``;
    * ``gamma_force_time_derivative``: ``amu mm / ns^3``.

    Scalar power values use ``amu mm^2 / ns^3`` and scalar work/energy values
    use ``amu mm^2 / ns^2``.  ``reaction_power`` and ``reaction_work`` retain
    their mechanical sign; a negative value removes particle energy.

    ``cross_field_energy`` is Medina's equation (98), and its time derivative
    completes the instantaneous balance

    ``reaction_power + far_radiated_power + cross_field_energy_rate = 0``

    when ``beta`` and ``gamma`` describe the same physical velocity.  The
    reported residual makes any inconsistency visible to the integration
    layer instead of silently folding it into the radiation loss.

    For dynamically consistent inputs ``far_radiated_power`` is non-negative.
    It is intentionally not clamped: a negative value is evidence that the
    supplied force and acceleration do not satisfy the approximation.
    """

    radiation_reaction_force: Vector3
    radiation_reaction_impulse: Vector3
    gamma_force_time_derivative: Vector3
    radiated_momentum_rate: Vector3
    cross_field_momentum: Vector3
    reaction_power: float
    reaction_work: float
    far_radiated_power: float
    far_radiated_energy: float
    cross_field_energy: float
    cross_field_energy_rate: float
    cross_field_energy_change: float
    energy_balance_residual: float
    gamma_time_derivative: float
    response_time: float
    gamma_beta_residual: float


def _vector3(value: Sequence[float], *, name: str) -> Vector3:
    """Return a finite three-vector with an informative validation error."""

    components = tuple(float(component) for component in value)
    if len(components) != 3:
        raise ValueError(f"{name} must contain exactly three components")
    if not all(isfinite(component) for component in components):
        raise ValueError(f"{name} must contain only finite values")
    return components[0], components[1], components[2]


def _dot(left: Vector3, right: Vector3) -> float:
    """Return a three-vector dot product with compensated summation."""

    return float(fsum(a * b for a, b in zip(left, right)))


def _scaled(vector: Vector3, factor: float) -> Vector3:
    return (
        float(factor * vector[0]),
        float(factor * vector[1]),
        float(factor * vector[2]),
    )


def _add(left: Vector3, right: Vector3) -> Vector3:
    return (
        float(left[0] + right[0]),
        float(left[1] + right[1]),
        float(left[2] + right[2]),
    )


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (
        float(left[0] - right[0]),
        float(left[1] - right[1]),
        float(left[2] - right[2]),
    )


def medina_response_time(*, charge: float, mass: float) -> float:
    """Return ``2 q^2 / (3 m c^3)`` in native ns.

    The sign of ``charge`` is immaterial because charge radiation reaction is
    quadratic in charge.  ``mass`` is the dressed particle mass in amu.
    """

    charge_value = float(charge)
    mass_value = float(mass)
    if not isfinite(charge_value):
        raise ValueError("charge must be finite")
    if not isfinite(mass_value) or mass_value <= 0.0:
        raise ValueError("mass must be finite and positive")
    return float(2.0 * charge_value**2 / (3.0 * mass_value * C_MMNS**3))


def compute_medina_radiation_reaction(
    *,
    external_force: Sequence[float],
    external_force_time_derivative: Sequence[float],
    beta: Sequence[float],
    acceleration: Sequence[float],
    gamma: float,
    mass: float,
    charge: float,
    coordinate_dt: float,
) -> MedinaRadiationReactionResult:
    """Evaluate Medina's reduced-order charge radiation reaction.

    Parameters
    ----------
    external_force:
        Current non-radiation-reaction mechanical force ``dp/dt`` in native
        ``amu mm / ns^2``.  It may include prescribed, charge, and dipole
        response forces, but must not include the returned self-force.
    external_force_time_derivative:
        Complete lab-time derivative ``dF_ext/dt`` in
        ``amu mm / ns^3``.  For a retarded interaction this derivative must
        include the changing retarded source event.
    beta:
        Dimensionless lab velocity ``v/c``.
    acceleration:
        Lab acceleration ``dv/dt`` in ``mm / ns^2`` (not ``d beta/dt`` and not
        a proper-time derivative).
    gamma:
        Lorentz factor corresponding to ``beta``.
    mass:
        Dressed particle mass in amu.
    charge:
        Signed native Gaussian charge.  Only ``charge**2`` enters.
    coordinate_dt:
        Non-negative lab-time interval in ns used for the returned first-order
        impulse and work diagnostics.  Passing zero still evaluates the force
        and instantaneous powers.

    Returns
    -------
    MedinaRadiationReactionResult
        Uncapped force and a transparent decomposition into derivative,
        radiated-momentum, mechanical-work, far-radiation, and cross-field
        terms.
    """

    force = _vector3(external_force, name="external_force")
    force_derivative = _vector3(
        external_force_time_derivative,
        name="external_force_time_derivative",
    )
    beta_vector = _vector3(beta, name="beta")
    acceleration_vector = _vector3(acceleration, name="acceleration")

    gamma_value = float(gamma)
    coordinate_dt_value = float(coordinate_dt)
    if not isfinite(gamma_value) or gamma_value < 1.0:
        raise ValueError("gamma must be finite and at least one")
    if not isfinite(coordinate_dt_value) or coordinate_dt_value < 0.0:
        raise ValueError("coordinate_dt must be finite and non-negative")

    beta_squared = _dot(beta_vector, beta_vector)
    if beta_squared > 1.0:
        raise ValueError("beta magnitude must not exceed one")

    response_time = medina_response_time(charge=charge, mass=mass)
    gamma_beta_residual = float(1.0 - beta_squared - 1.0 / gamma_value**2)

    beta_dot_acceleration = _dot(beta_vector, acceleration_vector)
    gamma_time_derivative = float(gamma_value**3 * beta_dot_acceleration / C_MMNS)
    gamma_force_derivative = _add(
        _scaled(force_derivative, gamma_value),
        _scaled(force, gamma_time_derivative),
    )

    force_dot_acceleration = _dot(force, acceleration_vector)
    far_radiated_power = float(response_time * gamma_value**3 * force_dot_acceleration)
    radiated_momentum_rate = _scaled(
        beta_vector,
        far_radiated_power / C_MMNS,
    )

    radiation_reaction_force = _subtract(
        _scaled(gamma_force_derivative, response_time),
        radiated_momentum_rate,
    )
    radiation_reaction_impulse = _scaled(
        radiation_reaction_force,
        coordinate_dt_value,
    )

    velocity = _scaled(beta_vector, C_MMNS)
    reaction_power = _dot(radiation_reaction_force, velocity)
    reaction_work = float(reaction_power * coordinate_dt_value)
    far_radiated_energy = float(far_radiated_power * coordinate_dt_value)

    cross_field_momentum = _scaled(force, -response_time * gamma_value)
    cross_field_energy = _dot(cross_field_momentum, velocity)
    cross_field_energy_rate = float(
        -response_time
        * (
            _dot(gamma_force_derivative, velocity)
            + gamma_value * force_dot_acceleration
        )
    )
    cross_field_energy_change = float(cross_field_energy_rate * coordinate_dt_value)
    energy_balance_residual = float(
        fsum((reaction_power, far_radiated_power, cross_field_energy_rate))
    )

    return MedinaRadiationReactionResult(
        radiation_reaction_force=radiation_reaction_force,
        radiation_reaction_impulse=radiation_reaction_impulse,
        gamma_force_time_derivative=gamma_force_derivative,
        radiated_momentum_rate=radiated_momentum_rate,
        cross_field_momentum=cross_field_momentum,
        reaction_power=reaction_power,
        reaction_work=reaction_work,
        far_radiated_power=far_radiated_power,
        far_radiated_energy=far_radiated_energy,
        cross_field_energy=cross_field_energy,
        cross_field_energy_rate=cross_field_energy_rate,
        cross_field_energy_change=cross_field_energy_change,
        energy_balance_residual=energy_balance_residual,
        gamma_time_derivative=gamma_time_derivative,
        response_time=response_time,
        gamma_beta_residual=gamma_beta_residual,
    )


__all__ = [
    "MedinaRadiationReactionResult",
    "compute_medina_radiation_reaction",
    "medina_response_time",
]
