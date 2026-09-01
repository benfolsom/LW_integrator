"""Potential-jet derivative form of the covariant RFS response.

This module is an algebraic validation seam for a future potential-first
retarded provider.  It consumes the first and second coordinate derivatives of
the ordinary Maxwell four-potential directly and never constructs electric or
magnetic three-fields, the electromagnetic field tensor, or its full gradient.

Coordinates are ``x=(ct,x,y,z)`` and
``partial_a[lambda, nu] = partial_lambda A^nu``.  The potential Hessian is
``partial2_a[kappa, lambda, nu] = partial_kappa partial_lambda A^nu``.
The observer spin is held fixed under both coordinate derivatives, matching
the maintained RFS response convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Sequence, Union, cast

import numpy as np

from .constants import C_MMNS

VectorLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]
Tensor3Like = Union[Sequence[Sequence[Sequence[float]]], np.ndarray]

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0), dtype=float)


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
_LEVI_CIVITA_UPPER = -_LEVI_CIVITA_LOWER


@dataclass(frozen=True)
class PotentialDerivativeRFSResponse:
    """Direct charge, magnetic-moment, and spin response at one event."""

    charge_four_force: np.ndarray
    dipole_four_force: np.ndarray
    total_four_force: np.ndarray
    spin_rhs: np.ndarray


@dataclass(frozen=True)
class PotentialDirectionalRFSReductionJet:
    """Leading RFS motion and its first two proper-time derivatives.

    The charge and dipole pieces are kept separate so a caller can identify
    which sector requires the higher potential derivatives.  ``four_jerk``
    and ``four_snap`` are derivatives of four-velocity, not of four-momentum.
    No radiation-reaction term is included.
    """

    leading_response: PotentialDerivativeRFSResponse
    four_acceleration: np.ndarray
    normalized_spin_first_derivative: np.ndarray
    charge_four_force_first_derivative: np.ndarray
    dipole_four_force_first_derivative: np.ndarray
    four_jerk: np.ndarray
    normalized_spin_second_derivative: np.ndarray
    charge_four_force_second_derivative: np.ndarray
    dipole_four_force_second_derivative: np.ndarray
    four_snap: np.ndarray


def _four_vector(value: VectorLike, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(np.ndarray, vector)


def _potential_gradient(value: MatrixLike) -> np.ndarray:
    gradient = np.asarray(value, dtype=float)
    if gradient.shape != (4, 4):
        raise ValueError("partial_a must have shape (4, 4)")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("partial_a must contain only finite values")
    return cast(np.ndarray, gradient)


def _potential_hessian(value: Tensor3Like) -> np.ndarray:
    hessian = np.asarray(value, dtype=float)
    if hessian.shape != (4, 4, 4):
        raise ValueError("partial2_a must have shape (4, 4, 4)")
    if not np.all(np.isfinite(hessian)):
        raise ValueError("partial2_a must contain only finite values")
    if not np.allclose(hessian, np.swapaxes(hessian, 0, 1), rtol=0.0, atol=0.0):
        raise ValueError("partial2_a derivative indices must commute exactly")
    return cast(np.ndarray, hessian)


def _field_on_covector_from_potential_gradient(
    partial_a: np.ndarray,
    covector: np.ndarray,
) -> np.ndarray:
    """Return ``F^(mu nu) covector_nu`` without materializing ``F``."""

    response = np.zeros(4, dtype=float)
    for mu in range(4):
        antisymmetric_row = np.zeros(4, dtype=float)
        for nu in range(4):
            if mu == nu:
                # This coefficient cancels identically in d^mu A^nu-d^nu A^mu.
                # Skipping it prevents pure-gauge diagonal values from creating
                # subtraction roundoff in an otherwise zero contribution.
                continue
            antisymmetric_row[nu] = (
                _SIGNS[mu] * partial_a[mu, nu] - _SIGNS[nu] * partial_a[nu, mu]
            )
        # Retain NumPy's fixed four-term dot ordering without materializing the
        # complete field tensor.  This is both faster than Python scalar
        # accumulation and avoids an extra rounding divergence at very high
        # gamma, where the spin equation contains large cancelling terms.
        response[mu] = antisymmetric_row @ covector
    return response


def _partial_magnetic_potential_covariant(
    partial2_a: np.ndarray,
    spin_four_vector: np.ndarray,
) -> np.ndarray:
    """Return ``partial_lambda (F*_(nu rho) a^rho)`` directly from ``d2A``.

    The Levi-Civita contraction is the algebraic simplification of the
    maintained ``partial_f -> dual gradient -> partial B`` sequence.  It keeps
    only the 4x4 derivative of the RFS magnetic potential as an intermediate.
    """

    derivative = np.einsum(
        "nrab,r,r,b,lab->ln",
        _LEVI_CIVITA_UPPER,
        _SIGNS,
        spin_four_vector,
        _SIGNS,
        partial2_a,
        optimize=False,
    )
    return cast(np.ndarray, derivative * _SIGNS[np.newaxis, :])


def _g_on_covector_from_potential_hessian(
    partial2_a: np.ndarray,
    spin_four_vector: np.ndarray,
    contravariant_vector: np.ndarray,
) -> np.ndarray:
    """Return ``G^(mu nu)[a] vector_nu`` without materializing ``G``."""

    partial_b_covariant = _partial_magnetic_potential_covariant(
        partial2_a,
        spin_four_vector,
    )
    g_covariant_on_vector = (
        partial_b_covariant @ contravariant_vector
        - partial_b_covariant.T @ contravariant_vector
    )
    return cast(np.ndarray, _SIGNS * g_covariant_on_vector)


def potential_derivative_rfs_response_native(
    *,
    four_velocity_mm_ns: VectorLike,
    spin_four_vector: VectorLike,
    partial_a: MatrixLike,
    partial2_a: Tensor3Like,
    charge_native: float,
    mass_amu: float,
    magnetic_moment_native: float,
    invariant_spin_native: float,
) -> PotentialDerivativeRFSResponse:
    """Return the maintained linear RFS response directly from ``dA``/``d2A``.

    The output is algebraically equivalent to calling ``rfs_four_force_native``
    and ``rfs_spin_rhs_native`` after constructing ``F`` and ``partial F``.
    Keeping this as a separate oracle lets the future analytical retarded
    provider be validated before it replaces the field-tensor path.
    """

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    spin = _four_vector(spin_four_vector, name="spin_four_vector")
    gradient = _potential_gradient(partial_a)
    hessian = _potential_hessian(partial2_a)

    charge = float(charge_native)
    mass = float(mass_amu)
    moment = float(magnetic_moment_native)
    invariant_spin = float(invariant_spin_native)
    if not np.isfinite(charge) or not np.isfinite(moment):
        raise ValueError("charge_native and magnetic_moment_native must be finite")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_amu must be finite and positive")
    if not np.isfinite(invariant_spin) or invariant_spin <= 0.0:
        raise ValueError("invariant_spin_native must be finite and positive")

    velocity_covariant = _SIGNS * velocity
    spin_covariant = _SIGNS * spin
    field_on_velocity = _field_on_covector_from_potential_gradient(
        gradient,
        velocity_covariant,
    )
    field_on_spin = _field_on_covector_from_potential_gradient(
        gradient,
        spin_covariant,
    )
    g_on_velocity = _g_on_covector_from_potential_hessian(
        hessian,
        spin,
        velocity,
    )
    g_on_spin = _g_on_covector_from_potential_hessian(
        hessian,
        spin,
        spin,
    )

    charge_force = charge * field_on_velocity / C_MMNS
    dipole_force = moment * g_on_velocity / C_MMNS
    u_dot_f_dot_s = float(velocity_covariant @ field_on_spin)
    charge_to_mass_c = charge / (mass * C_MMNS)
    moment_to_spin = moment / invariant_spin
    orthogonal_field_on_spin = field_on_spin - (velocity * u_dot_f_dot_s / C_MMNS**2)
    spin_rhs = (
        charge_to_mass_c * field_on_spin
        + (moment_to_spin - charge_to_mass_c) * orthogonal_field_on_spin
        + moment * g_on_spin / (mass * C_MMNS)
    )

    return PotentialDerivativeRFSResponse(
        charge_four_force=cast(np.ndarray, charge_force),
        dipole_four_force=cast(np.ndarray, dipole_force),
        total_four_force=cast(np.ndarray, charge_force + dipole_force),
        spin_rhs=cast(np.ndarray, spin_rhs),
    )


def potential_directional_rfs_reduction_jet_native(
    *,
    four_velocity_mm_ns: VectorLike,
    spin_four_vector: VectorLike,
    partial_a: MatrixLike,
    partial2_a: Tensor3Like,
    partial3_a_along_velocity: Tensor3Like,
    partial3_a_along_acceleration: Tensor3Like,
    partial4_a_along_velocity_twice: Tensor3Like,
    charge_native: float,
    mass_amu: float,
    magnetic_moment_native: float,
    invariant_spin_native: float,
) -> PotentialDirectionalRFSReductionJet:
    """Differentiate the leading potential-first RFS motion analytically.

    All derivatives are evaluated on the non-self leading motion.  The full
    third and fourth derivatives of ``A`` are deliberately not accepted.
    Instead the caller supplies only these Hessian-shaped contractions:

    ``partial3_a_along_velocity = u^k partial_k partial2_a``,

    ``partial3_a_along_acceleration = A^k partial_k partial2_a``, and

    ``partial4_a_along_velocity_twice = u^k u^l partial_k partial_l partial2_a``.

    Coordinates are ``(ct,x,y,z)``, so proper-time directional derivatives
    use ``u^k partial_k`` without another factor of ``c``.  This function
    never constructs electric or magnetic three-fields, ``F``, ``partial F``,
    or a complete fourth-derivative potential tensor.

    The returned result contains no Medina or intrinsic-spin self-reaction.
    It is the lower-order derivative input needed by a later reduced
    self-force calculation.
    """

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    spin = _four_vector(spin_four_vector, name="spin_four_vector")
    gradient = _potential_gradient(partial_a)
    hessian = _potential_hessian(partial2_a)
    third_along_velocity = _potential_hessian(
        partial3_a_along_velocity,
    )
    third_along_acceleration = _potential_hessian(
        partial3_a_along_acceleration,
    )
    fourth_along_velocity_twice = _potential_hessian(
        partial4_a_along_velocity_twice,
    )

    charge = float(charge_native)
    mass = float(mass_amu)
    moment = float(magnetic_moment_native)
    invariant_spin = float(invariant_spin_native)
    if not np.isfinite(charge) or not np.isfinite(moment):
        raise ValueError("charge_native and magnetic_moment_native must be finite")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_amu must be finite and positive")
    if not np.isfinite(invariant_spin) or invariant_spin <= 0.0:
        raise ValueError("invariant_spin_native must be finite and positive")

    leading = potential_derivative_rfs_response_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        partial_a=gradient,
        partial2_a=hessian,
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )
    acceleration = leading.total_four_force / mass
    spin_first = leading.spin_rhs

    velocity_covariant = _SIGNS * velocity
    acceleration_covariant = _SIGNS * acceleration
    spin_covariant = _SIGNS * spin
    spin_first_covariant = _SIGNS * spin_first

    partial2_along_velocity = np.einsum(
        "k,kln->ln",
        velocity,
        hessian,
    )
    partial2_along_acceleration = np.einsum(
        "k,kln->ln",
        acceleration,
        hessian,
    )
    partial3_along_velocity_twice = np.einsum(
        "k,kln->ln",
        velocity,
        third_along_velocity,
    )

    field_rate_on_velocity = _field_on_covector_from_potential_gradient(
        partial2_along_velocity,
        velocity_covariant,
    )
    field_on_acceleration = _field_on_covector_from_potential_gradient(
        gradient,
        acceleration_covariant,
    )
    charge_force_first = (
        charge / C_MMNS * (field_rate_on_velocity + field_on_acceleration)
    )

    dipole_force_first = (
        moment
        / C_MMNS
        * (
            _g_on_covector_from_potential_hessian(
                third_along_velocity,
                spin,
                velocity,
            )
            + _g_on_covector_from_potential_hessian(
                hessian,
                spin_first,
                velocity,
            )
            + _g_on_covector_from_potential_hessian(
                hessian,
                spin,
                acceleration,
            )
        )
    )
    jerk = (charge_force_first + dipole_force_first) / mass

    field_on_spin = _field_on_covector_from_potential_gradient(
        gradient,
        spin_covariant,
    )
    field_on_spin_rate = _field_on_covector_from_potential_gradient(
        partial2_along_velocity,
        spin_covariant,
    ) + _field_on_covector_from_potential_gradient(
        gradient,
        spin_first_covariant,
    )
    u_dot_f_dot_s = float(velocity_covariant @ field_on_spin)
    u_dot_f_dot_s_rate = float(
        acceleration_covariant @ field_on_spin + velocity_covariant @ field_on_spin_rate
    )
    charge_to_mass_c = charge / (mass * C_MMNS)
    moment_to_spin = moment / invariant_spin
    moment_to_mass_c = moment / (mass * C_MMNS)
    g_on_spin_rate = (
        _g_on_covector_from_potential_hessian(
            third_along_velocity,
            spin,
            spin,
        )
        + _g_on_covector_from_potential_hessian(
            hessian,
            spin_first,
            spin,
        )
        + _g_on_covector_from_potential_hessian(
            hessian,
            spin,
            spin_first,
        )
    )
    spin_second = (
        moment_to_spin * field_on_spin_rate
        - (moment_to_spin - charge_to_mass_c)
        * (acceleration * u_dot_f_dot_s + velocity * u_dot_f_dot_s_rate)
        / C_MMNS**2
        + moment_to_mass_c * g_on_spin_rate
    )

    jerk_covariant = _SIGNS * jerk
    field_second_on_velocity = (
        _field_on_covector_from_potential_gradient(
            partial3_along_velocity_twice,
            velocity_covariant,
        )
        + _field_on_covector_from_potential_gradient(
            partial2_along_acceleration,
            velocity_covariant,
        )
        + 2.0
        * _field_on_covector_from_potential_gradient(
            partial2_along_velocity,
            acceleration_covariant,
        )
        + _field_on_covector_from_potential_gradient(
            gradient,
            jerk_covariant,
        )
    )
    charge_force_second = charge / C_MMNS * field_second_on_velocity

    dipole_force_second = (
        moment
        / C_MMNS
        * (
            _g_on_covector_from_potential_hessian(
                hessian,
                spin_second,
                velocity,
            )
            + 2.0
            * _g_on_covector_from_potential_hessian(
                hessian,
                spin_first,
                acceleration,
            )
            + _g_on_covector_from_potential_hessian(
                hessian,
                spin,
                jerk,
            )
            + 2.0
            * _g_on_covector_from_potential_hessian(
                third_along_velocity,
                spin_first,
                velocity,
            )
            + 2.0
            * _g_on_covector_from_potential_hessian(
                third_along_velocity,
                spin,
                acceleration,
            )
            + _g_on_covector_from_potential_hessian(
                third_along_acceleration,
                spin,
                velocity,
            )
            + _g_on_covector_from_potential_hessian(
                fourth_along_velocity_twice,
                spin,
                velocity,
            )
        )
    )
    snap = (charge_force_second + dipole_force_second) / mass

    arrays = (
        acceleration,
        spin_first,
        charge_force_first,
        dipole_force_first,
        jerk,
        spin_second,
        charge_force_second,
        dipole_force_second,
        snap,
    )
    for array in arrays:
        array.setflags(write=False)

    return PotentialDirectionalRFSReductionJet(
        leading_response=leading,
        four_acceleration=cast(np.ndarray, acceleration),
        normalized_spin_first_derivative=cast(np.ndarray, spin_first),
        charge_four_force_first_derivative=cast(np.ndarray, charge_force_first),
        dipole_four_force_first_derivative=cast(np.ndarray, dipole_force_first),
        four_jerk=cast(np.ndarray, jerk),
        normalized_spin_second_derivative=cast(np.ndarray, spin_second),
        charge_four_force_second_derivative=cast(np.ndarray, charge_force_second),
        dipole_four_force_second_derivative=cast(np.ndarray, dipole_force_second),
        four_snap=cast(np.ndarray, snap),
    )


__all__ = [
    "PotentialDirectionalRFSReductionJet",
    "PotentialDerivativeRFSResponse",
    "potential_derivative_rfs_response_native",
    "potential_directional_rfs_reduction_jet_native",
]
