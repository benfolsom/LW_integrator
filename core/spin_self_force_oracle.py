"""Diagnostic point-particle self-force through first order in spin.

This module translates Eqs. (19) and (20a) of Jakobsen,
Phys. Rev. Lett. 132, 151601 (2024),
https://doi.org/10.1103/PhysRevLett.132.151601, from natural rationalized
Heaviside--Lorentz units to the integrator's native scaled-Gaussian units.

It is deliberately an oracle rather than a production force law.  In
particular, it does not reduce the higher proper-time derivatives, choose a
finite-size matching prescription, or apply an impulse to a trajectory.  The
paper keeps terms only through first order in spin and magnetization; a pure
``mu**2`` recoil is therefore outside this result.

The paper uses ``a`` for four-acceleration.  Elsewhere in this project ``a``
often denotes the normalized spin direction, so this module instead calls the
successive worldline derivatives ``A`` (acceleration), ``J`` (jerk), and ``K``
(snap).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Sequence, Union, cast

import numpy as np

from .constants import C_MMNS
from .rfs import MINKOWSKI_METRIC


VectorLike = Union[Sequence[float], np.ndarray]


def _permutation_sign(indices: Sequence[int]) -> float:
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1.0 if inversions % 2 else 1.0


_LEVI_CIVITA_LOWER = np.zeros((4, 4, 4, 4), dtype=float)
for _indices in permutations(range(4)):
    _LEVI_CIVITA_LOWER[_indices] = _permutation_sign(_indices)

# Jakobsen defines the body-frame cross product with
# epsilon^(mu)_(nu rho sigma) and epsilon^(1230)=+1.  Raising the first
# index of epsilon_(mu nu rho sigma), while leaving the final three down,
# gives precisely that convention for the (+---) metric.
_LEVI_CIVITA_FIRST_UP = np.einsum("ma,anrs->mnrs", MINKOWSKI_METRIC, _LEVI_CIVITA_LOWER)


def _four_vector(value: VectorLike, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _minkowski_dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(left @ MINKOWSKI_METRIC @ right)


def _body_frame_cross(
    left: np.ndarray, right: np.ndarray, frame_vector: np.ndarray
) -> np.ndarray:
    """Return ``epsilon^mu_(nu rho sigma) left^nu right^rho frame^sigma``."""

    return cast(
        np.ndarray,
        np.einsum(
            "mnrs,n,r,s->m",
            _LEVI_CIVITA_FIRST_UP,
            left,
            right,
            frame_vector,
        ),
    )


def _orthogonal_projection(vector: np.ndarray, four_velocity: np.ndarray) -> np.ndarray:
    normalized_velocity = four_velocity / C_MMNS
    normalized_velocity_lower = MINKOWSKI_METRIC @ normalized_velocity
    projector_mixed = np.eye(4) - np.outer(
        normalized_velocity, normalized_velocity_lower
    )
    return cast(np.ndarray, projector_mixed @ vector)


@dataclass(frozen=True)
class JakobsenLinearSpinSelfForceResult:
    """Transparent decomposition of the point-particle self-force oracle.

    All returned four-force or four-momentum-rate arrays have native
    ``amu mm/ns^2`` units.  The
    ``magnetization_bracket_native`` arrays are the corresponding quantities
    before multiplication by ``2 q/(3 c^4)``; their units are
    ``native_charge mm^2/ns^4``.

    ``linear_spin_self_torque_native`` is identically zero because Jakobsen's
    self-torque correction vanishes at the perturbative order retained by the
    paper.  This does not claim that a finite source has no higher-order or
    ``mu**2`` self-torque.

    ``spin_radiative_field_balance_correction_native`` is not an additional
    mechanical force.  It is the term that accompanies the projected
    self-force when supplemental Eq. (33) compares it with instantaneous
    radiated four-momentum.  It is parallel to the four-velocity and is kept
    separate from ``linear_spin_self_force_native`` for that reason.
    """

    intrinsic_subtracted_moment_native: np.ndarray
    moment_derivative_cross_native: np.ndarray
    cross_product_derivative_native: np.ndarray
    magnetization_bracket_native: np.ndarray
    projected_magnetization_bracket_native: np.ndarray
    linear_spin_self_force_native: np.ndarray
    spin_radiative_field_coupling_scalar_native: float
    spin_radiative_field_balance_correction_native: np.ndarray
    linear_spin_radiative_balance_rate_native: np.ndarray
    charge_ald_self_force_native: np.ndarray
    total_self_force_through_linear_spin_native: np.ndarray
    linear_spin_self_torque_native: np.ndarray
    four_velocity_dot_linear_spin_force_native: float
    four_velocity_dot_total_force_native: float
    spin_orthogonality_residual_native: float
    magnetic_moment_orthogonality_residual_native: float


@dataclass(frozen=True)
class JakobsenIntrinsicSpinRadiationBalanceResult:
    """Intrinsic-spin decomposition of supplemental Eq. (33).

    ``self_force`` contains the projected mechanical force and the separate
    radiative-field balance correction.  ``radiated_particle_momentum_rate``
    is the instantaneous loss rate of particle four-momentum; its negative is
    the outward radiated four-momentum rate.  ``bound_field_momentum`` is the
    reversible, Schott-like near-field momentum whose proper-time derivative
    completes the local identity.

    This result applies only to the intrinsic no-susceptibility relation
    ``M = g q S/(2 m c)``.  None of its balance-only terms are additional
    mechanical forces.
    """

    self_force: JakobsenLinearSpinSelfForceResult
    bound_field_momentum_native: np.ndarray
    bound_field_momentum_derivative_native: np.ndarray
    radiated_particle_momentum_rate_native: np.ndarray
    outward_radiated_momentum_rate_native: np.ndarray
    balance_residual_native: np.ndarray


def evaluate_jakobsen_linear_spin_self_force_native(
    *,
    charge_native: float,
    mass_amu: float,
    four_velocity_mm_ns: VectorLike,
    four_acceleration_mm_ns2: VectorLike,
    four_jerk_mm_ns3: VectorLike,
    four_snap_mm_ns4: VectorLike,
    spin_four_vector_native: VectorLike,
    spin_four_derivative_native: VectorLike,
    magnetic_moment_four_vector_native: VectorLike,
    magnetic_moment_four_derivative_native: VectorLike,
) -> JakobsenLinearSpinSelfForceResult:
    """Evaluate the charge ALD term and its leading spin correction.

    Derivatives are with respect to proper time in nanoseconds.  The spin
    four-vector is the *physical* Pauli--Lubanski vector in native action
    units, not the dimensionless RFS polarization vector.  For the current
    particle model callers construct it as ``S * a^mu``, with
    ``S = spin_quantum_number * HBAR_NATIVE``.  The magnetic-moment four-vector
    is likewise ``magnetic_moment_native * a^mu`` when susceptibility is
    absent.

    Restoring ``c`` and converting rationalized Heaviside--Lorentz charge and
    moment to Gaussian units gives

    ``F_qS = (2 q / 3 c^4) P [J x Mdot + d/dtau(J x (M-qS/(mc)))]``.

    The derivative acts on both vectors *and* the body-frame velocity used in
    the cross product.  The function expands that derivative exactly.  It
    also returns ``(2 q^2 / 3 c^3) P J`` as an independent charge-ALD unit and
    sign check.

    Supplemental Eq. (33) compares the projected mechanical self-force with
    radiated four-momentum.  In native Gaussian units its additional local
    radiative-field term is

    ``Delta_rad = (q/(m c^3)) (u/c) S.[A x E_rad]``,

    where ``E_rad=(2q/(3c^3)) P J`` is the leading charge radiative field.
    The returned ``linear_spin_radiative_balance_rate_native`` is
    ``F_qS + Delta_rad``.  ``Delta_rad`` is a balance term, not a mechanical
    force to apply to the particle.  No reduction of order is performed.
    """

    charge = float(charge_native)
    mass = float(mass_amu)
    if not np.isfinite(charge):
        raise ValueError("charge_native must be finite")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_amu must be finite and positive")

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    acceleration = _four_vector(
        four_acceleration_mm_ns2, name="four_acceleration_mm_ns2"
    )
    jerk = _four_vector(four_jerk_mm_ns3, name="four_jerk_mm_ns3")
    snap = _four_vector(four_snap_mm_ns4, name="four_snap_mm_ns4")
    spin = _four_vector(spin_four_vector_native, name="spin_four_vector_native")
    spin_derivative = _four_vector(
        spin_four_derivative_native, name="spin_four_derivative_native"
    )
    moment = _four_vector(
        magnetic_moment_four_vector_native,
        name="magnetic_moment_four_vector_native",
    )
    moment_derivative = _four_vector(
        magnetic_moment_four_derivative_native,
        name="magnetic_moment_four_derivative_native",
    )

    if velocity[0] <= 0.0:
        raise ValueError("four_velocity_mm_ns must be future-directed")
    velocity_norm = _minkowski_dot(velocity, velocity)
    if not np.isclose(velocity_norm, C_MMNS**2, rtol=2.0e-12, atol=2.0e-12 * C_MMNS**2):
        raise ValueError("four_velocity_mm_ns must satisfy u.u = c^2")

    normalized_velocity = velocity / C_MMNS
    normalized_velocity_derivative = acceleration / C_MMNS
    minimal_spin_moment = charge * spin / (mass * C_MMNS)
    minimal_spin_moment_derivative = charge * spin_derivative / (mass * C_MMNS)
    subtracted_moment = moment - minimal_spin_moment
    subtracted_moment_derivative = moment_derivative - minimal_spin_moment_derivative

    moment_derivative_cross = _body_frame_cross(
        jerk, moment_derivative, normalized_velocity
    )
    cross_product_derivative = (
        _body_frame_cross(snap, subtracted_moment, normalized_velocity)
        + _body_frame_cross(jerk, subtracted_moment_derivative, normalized_velocity)
        + _body_frame_cross(jerk, subtracted_moment, normalized_velocity_derivative)
    )
    bracket = moment_derivative_cross + cross_product_derivative
    projected_bracket = _orthogonal_projection(bracket, velocity)

    spin_prefactor = 2.0 * charge / (3.0 * C_MMNS**4)
    linear_spin_force = spin_prefactor * projected_bracket
    projected_jerk = _orthogonal_projection(jerk, velocity)
    charge_radiative_electric_field = 2.0 * charge / (3.0 * C_MMNS**3) * projected_jerk
    radiative_field_cross = _body_frame_cross(
        acceleration,
        charge_radiative_electric_field,
        normalized_velocity,
    )
    radiative_field_coupling_scalar = _minkowski_dot(spin, radiative_field_cross)
    radiative_field_balance_correction = (
        charge
        / (mass * C_MMNS**3)
        * normalized_velocity
        * radiative_field_coupling_scalar
    )
    linear_spin_radiative_balance_rate = (
        linear_spin_force + radiative_field_balance_correction
    )
    charge_ald_force = 2.0 * charge**2 / (3.0 * C_MMNS**3) * projected_jerk
    total_force = charge_ald_force + linear_spin_force

    return JakobsenLinearSpinSelfForceResult(
        intrinsic_subtracted_moment_native=subtracted_moment,
        moment_derivative_cross_native=moment_derivative_cross,
        cross_product_derivative_native=cross_product_derivative,
        magnetization_bracket_native=bracket,
        projected_magnetization_bracket_native=projected_bracket,
        linear_spin_self_force_native=linear_spin_force,
        spin_radiative_field_coupling_scalar_native=(radiative_field_coupling_scalar),
        spin_radiative_field_balance_correction_native=(
            radiative_field_balance_correction
        ),
        linear_spin_radiative_balance_rate_native=(linear_spin_radiative_balance_rate),
        charge_ald_self_force_native=charge_ald_force,
        total_self_force_through_linear_spin_native=total_force,
        linear_spin_self_torque_native=np.zeros(4, dtype=float),
        four_velocity_dot_linear_spin_force_native=_minkowski_dot(
            velocity, linear_spin_force
        ),
        four_velocity_dot_total_force_native=_minkowski_dot(velocity, total_force),
        spin_orthogonality_residual_native=_minkowski_dot(velocity, spin),
        magnetic_moment_orthogonality_residual_native=_minkowski_dot(velocity, moment),
    )


def evaluate_jakobsen_intrinsic_spin_radiation_balance_native(
    *,
    charge_native: float,
    mass_amu: float,
    g_factor: float,
    four_velocity_mm_ns: VectorLike,
    four_acceleration_mm_ns2: VectorLike,
    four_jerk_mm_ns3: VectorLike,
    four_snap_mm_ns4: VectorLike,
    spin_four_vector_native: VectorLike,
    spin_four_derivative_native: VectorLike,
    spin_four_second_derivative_native: VectorLike,
) -> JakobsenIntrinsicSpinRadiationBalanceResult:
    """Evaluate the intrinsic-spin momentum balance in supplemental Eq. (33).

    The published supplement uses ``v^mu`` only in this equation; comparison
    with its definitions and dimensions identifies it as the normalized
    four-velocity ``u^mu/c``.  This implementation uses the paper's
    intrinsic relation ``M=g q S/(2mc)`` and converts the complete identity
    from rationalized natural units to native Gaussian units.

    The identity is

    ``F_qS + Delta_rad = Pdot_rad,particle + dB_bound/dtau``.

    ``Pdot_rad,particle`` is negative when positive four-momentum leaves the
    particle.  The function evaluates its two terms directly rather than
    defining it by the balance residual, so the returned residual is a real
    algebra and unit check.
    """

    charge = float(charge_native)
    mass = float(mass_amu)
    g_value = float(g_factor)
    if not np.isfinite(charge):
        raise ValueError("charge_native must be finite")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_amu must be finite and positive")
    if not np.isfinite(g_value):
        raise ValueError("g_factor must be finite")

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    acceleration = _four_vector(
        four_acceleration_mm_ns2, name="four_acceleration_mm_ns2"
    )
    jerk = _four_vector(four_jerk_mm_ns3, name="four_jerk_mm_ns3")
    snap = _four_vector(four_snap_mm_ns4, name="four_snap_mm_ns4")
    spin = _four_vector(spin_four_vector_native, name="spin_four_vector_native")
    spin_derivative = _four_vector(
        spin_four_derivative_native, name="spin_four_derivative_native"
    )
    spin_second_derivative = _four_vector(
        spin_four_second_derivative_native,
        name="spin_four_second_derivative_native",
    )
    normalized_velocity = velocity / C_MMNS
    normalized_velocity_derivative = acceleration / C_MMNS

    intrinsic_coefficient = g_value * charge / (2.0 * mass * C_MMNS)
    self_force = evaluate_jakobsen_linear_spin_self_force_native(
        charge_native=charge,
        mass_amu=mass,
        four_velocity_mm_ns=velocity,
        four_acceleration_mm_ns2=acceleration,
        four_jerk_mm_ns3=jerk,
        four_snap_mm_ns4=snap,
        spin_four_vector_native=spin,
        spin_four_derivative_native=spin_derivative,
        magnetic_moment_four_vector_native=intrinsic_coefficient * spin,
        magnetic_moment_four_derivative_native=(
            intrinsic_coefficient * spin_derivative
        ),
    )

    acceleration_cross_spin_derivative = _body_frame_cross(
        acceleration, spin_derivative, normalized_velocity
    )
    jerk_cross_spin = _body_frame_cross(jerk, spin, normalized_velocity)
    bound_prefactor = charge**2 / (3.0 * mass * C_MMNS**5)
    bound_momentum = bound_prefactor * (
        g_value * acceleration_cross_spin_derivative + (g_value - 2.0) * jerk_cross_spin
    )

    acceleration_cross_spin_derivative_rate = (
        _body_frame_cross(jerk, spin_derivative, normalized_velocity)
        + _body_frame_cross(acceleration, spin_second_derivative, normalized_velocity)
        + _body_frame_cross(
            acceleration,
            spin_derivative,
            normalized_velocity_derivative,
        )
    )
    jerk_cross_spin_rate = (
        _body_frame_cross(snap, spin, normalized_velocity)
        + _body_frame_cross(jerk, spin_derivative, normalized_velocity)
        + _body_frame_cross(jerk, spin, normalized_velocity_derivative)
    )
    bound_momentum_derivative = bound_prefactor * (
        g_value * acceleration_cross_spin_derivative_rate
        + (g_value - 2.0) * jerk_cross_spin_rate
    )

    acceleration_cross_jerk = _body_frame_cross(acceleration, jerk, normalized_velocity)
    spin_acceleration_jerk_scalar = _minkowski_dot(spin, acceleration_cross_jerk)
    radiated_particle_momentum_rate = (
        charge**2
        * g_value
        / (3.0 * mass)
        * (
            normalized_velocity * spin_acceleration_jerk_scalar / C_MMNS**6
            + _body_frame_cross(
                spin_second_derivative,
                acceleration,
                normalized_velocity,
            )
            / C_MMNS**5
        )
    )
    balance_residual = self_force.linear_spin_radiative_balance_rate_native - (
        radiated_particle_momentum_rate + bound_momentum_derivative
    )

    return JakobsenIntrinsicSpinRadiationBalanceResult(
        self_force=self_force,
        bound_field_momentum_native=bound_momentum,
        bound_field_momentum_derivative_native=bound_momentum_derivative,
        radiated_particle_momentum_rate_native=radiated_particle_momentum_rate,
        outward_radiated_momentum_rate_native=-radiated_particle_momentum_rate,
        balance_residual_native=balance_residual,
    )


__all__ = [
    "JakobsenIntrinsicSpinRadiationBalanceResult",
    "JakobsenLinearSpinSelfForceResult",
    "evaluate_jakobsen_intrinsic_spin_radiation_balance_native",
    "evaluate_jakobsen_linear_spin_self_force_native",
]
