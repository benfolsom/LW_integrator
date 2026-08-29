"""Direct RFS contractions from six antisymmetric response coefficients."""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np

from .constants import C_MMNS
from .potential_jet_rfs import PotentialDerivativeRFSResponse

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0))
_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _permutation_sign(indices: Sequence[int]) -> float:
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1.0 if inversions % 2 else 1.0


_EPSILON_UPPER = np.zeros((4, 4, 4, 4))
for _indices in permutations(range(4)):
    _EPSILON_UPPER[_indices] = -_permutation_sign(_indices)

_DUAL_PACKED_COVARIANT = np.zeros((4, 4, 6), dtype=float)
for _dual_first in range(4):
    for _dual_second in range(4):
        for _pair_index, (_first, _second) in enumerate(_PAIRS):
            _DUAL_PACKED_COVARIANT[_dual_first, _dual_second, _pair_index] = (
                _EPSILON_UPPER[_dual_first, _dual_second, _first, _second]
                * _SIGNS[_dual_first]
                * _SIGNS[_dual_second]
                * _SIGNS[_first]
                * _SIGNS[_second]
            )


def _act_on_covector(packed: np.ndarray, covector: np.ndarray) -> np.ndarray:
    f01, f02, f03, f12, f13, f23 = packed
    c0, c1, c2, c3 = covector
    return np.asarray(
        (
            f01 * c1 + f02 * c2 + f03 * c3,
            -f01 * c0 + f12 * c2 + f13 * c3,
            -f02 * c0 - f12 * c1 + f23 * c3,
            -f03 * c0 - f13 * c1 - f23 * c2,
        )
    )


def pack_antisymmetric_response_native(field_tensor: np.ndarray) -> np.ndarray:
    """Pack the six independent upper-triangle coefficients of ``F``."""

    field = np.asarray(field_tensor, dtype=float)
    if field.shape != (4, 4):
        raise ValueError("field_tensor must have shape (4, 4)")
    return np.asarray([field[first, second] for first, second in _PAIRS])


def pack_partial_antisymmetric_response_native(partial_f: np.ndarray) -> np.ndarray:
    """Pack ``partial_lambda F`` without retaining zero/duplicate entries."""

    gradient = np.asarray(partial_f, dtype=float)
    if gradient.shape != (4, 4, 4):
        raise ValueError("partial_f must have shape (4, 4, 4)")
    return np.asarray(
        [
            [gradient[derivative, first, second] for first, second in _PAIRS]
            for derivative in range(4)
        ]
    )


def materialize_antisymmetric_response_native(packed: np.ndarray) -> np.ndarray:
    """Materialize ``F`` only for diagnostics and reference comparisons."""

    coefficients = np.asarray(packed, dtype=float)
    if coefficients.shape != (6,):
        raise ValueError("antisymmetric_response must have shape (6,)")
    field = np.zeros((4, 4), dtype=float)
    for pair_index, (first, second) in enumerate(_PAIRS):
        field[first, second] = coefficients[pair_index]
        field[second, first] = -coefficients[pair_index]
    return field


def materialize_partial_antisymmetric_response_native(
    partial_packed: np.ndarray,
) -> np.ndarray:
    """Materialize ``partial_lambda F`` for diagnostics and fallback audits."""

    coefficients = np.asarray(partial_packed, dtype=float)
    if coefficients.shape != (4, 6):
        raise ValueError("partial_antisymmetric_response must have shape (4, 6)")
    gradient = np.zeros((4, 4, 4), dtype=float)
    for derivative in range(4):
        for pair_index, (first, second) in enumerate(_PAIRS):
            gradient[derivative, first, second] = coefficients[derivative, pair_index]
            gradient[derivative, second, first] = -coefficients[derivative, pair_index]
    return gradient


def antisymmetric_response_charge_force_native(
    *,
    four_velocity_mm_ns: Sequence[float],
    antisymmetric_response: Sequence[float],
    charge_native: float,
) -> np.ndarray:
    """Return ``(q/c) F.u`` directly from six response coefficients."""

    velocity = np.asarray(four_velocity_mm_ns, dtype=float)
    packed = np.asarray(antisymmetric_response, dtype=float)
    if velocity.shape != (4,):
        raise ValueError("four_velocity_mm_ns must have shape (4,)")
    if packed.shape != (6,):
        raise ValueError("antisymmetric_response must have shape (6,)")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(packed)):
        raise ValueError("charge-response inputs must be finite")
    charge = float(charge_native)
    if not np.isfinite(charge):
        raise ValueError("charge_native must be finite")
    return charge * _act_on_covector(packed, _SIGNS * velocity) / C_MMNS


def antisymmetric_response_charge_force_derivative_native(
    *,
    four_velocity_mm_ns: Sequence[float],
    four_acceleration_mm_ns2: Sequence[float],
    antisymmetric_response: Sequence[float],
    partial_antisymmetric_response: Sequence[Sequence[float]],
    charge_native: float,
) -> np.ndarray:
    """Differentiate the packed ordinary charge response along the worldline.

    This is the packed equivalent of
    ``(q/c)[(u^lambda partial_lambda F) u_lower + F a_lower]``.  It preserves
    the potential-first response surface and does not materialize ``F`` or
    ``partial F`` merely to construct a second-order Lorentz impulse.
    """

    velocity = np.asarray(four_velocity_mm_ns, dtype=float)
    acceleration = np.asarray(four_acceleration_mm_ns2, dtype=float)
    packed = np.asarray(antisymmetric_response, dtype=float)
    partial_packed = np.asarray(partial_antisymmetric_response, dtype=float)
    if velocity.shape != (4,) or acceleration.shape != (4,):
        raise ValueError(
            "four_velocity_mm_ns and four_acceleration_mm_ns2 must have shape (4,)"
        )
    if packed.shape != (6,):
        raise ValueError("antisymmetric_response must have shape (6,)")
    if partial_packed.shape != (4, 6):
        raise ValueError("partial_antisymmetric_response must have shape (4, 6)")
    if not all(
        np.all(np.isfinite(value))
        for value in (velocity, acceleration, packed, partial_packed)
    ):
        raise ValueError("charge-response derivative inputs must be finite")
    charge = float(charge_native)
    if not np.isfinite(charge):
        raise ValueError("charge_native must be finite")

    convective_response = np.einsum(
        "l,lp->p",
        velocity,
        partial_packed,
        optimize=False,
    )
    return (
        charge
        * (
            _act_on_covector(convective_response, _SIGNS * velocity)
            + _act_on_covector(packed, _SIGNS * acceleration)
        )
        / C_MMNS
    )


def _partial_magnetic_potential_covariant(
    partial_packed: np.ndarray,
    spin: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "nrp,lp,r->ln",
        _DUAL_PACKED_COVARIANT,
        partial_packed,
        spin,
        optimize=False,
    )


def antisymmetric_response_rfs_native(
    *,
    four_velocity_mm_ns: Sequence[float],
    spin_four_vector: Sequence[float],
    antisymmetric_response: Sequence[float],
    partial_antisymmetric_response: Sequence[Sequence[float]],
    charge_native: float,
    mass_amu: float,
    magnetic_moment_native: float,
    invariant_spin_native: float,
) -> PotentialDerivativeRFSResponse:
    """Return full linear RFS response without constructing tensor middlemen."""

    velocity = np.asarray(four_velocity_mm_ns, dtype=float)
    spin = np.asarray(spin_four_vector, dtype=float)
    packed = np.asarray(antisymmetric_response, dtype=float)
    partial_packed = np.asarray(partial_antisymmetric_response, dtype=float)
    if velocity.shape != (4,) or spin.shape != (4,):
        raise ValueError(
            "four_velocity_mm_ns and spin_four_vector must have shape (4,)"
        )
    if packed.shape != (6,):
        raise ValueError("antisymmetric_response must have shape (6,)")
    if partial_packed.shape != (4, 6):
        raise ValueError("partial_antisymmetric_response must have shape (4, 6)")
    if not all(
        np.all(np.isfinite(value)) for value in (velocity, spin, packed, partial_packed)
    ):
        raise ValueError("RFS response inputs must be finite")
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
    response_on_velocity = _act_on_covector(packed, velocity_covariant)
    response_on_spin = _act_on_covector(packed, spin_covariant)
    partial_b = _partial_magnetic_potential_covariant(partial_packed, spin)
    g_covariant = partial_b - partial_b.T
    g_on_velocity = _SIGNS * (g_covariant @ velocity)
    g_on_spin = _SIGNS * (g_covariant @ spin)

    charge_force = charge * response_on_velocity / C_MMNS
    dipole_force = moment * g_on_velocity / C_MMNS
    u_dot_f_dot_s = float(velocity_covariant @ response_on_spin)
    charge_to_mass_c = charge / (mass * C_MMNS)
    moment_to_spin = moment / invariant_spin
    orthogonal_response_on_spin = response_on_spin - (
        velocity * u_dot_f_dot_s / C_MMNS**2
    )
    spin_rhs = (
        charge_to_mass_c * response_on_spin
        + (moment_to_spin - charge_to_mass_c) * orthogonal_response_on_spin
        + moment * g_on_spin / (mass * C_MMNS)
    )
    return PotentialDerivativeRFSResponse(
        charge_four_force=charge_force,
        dipole_four_force=dipole_force,
        total_four_force=charge_force + dipole_force,
        spin_rhs=spin_rhs,
    )


__all__ = [
    "antisymmetric_response_charge_force_derivative_native",
    "antisymmetric_response_charge_force_native",
    "antisymmetric_response_rfs_native",
    "materialize_antisymmetric_response_native",
    "materialize_partial_antisymmetric_response_native",
    "pack_antisymmetric_response_native",
    "pack_partial_antisymmetric_response_native",
]
