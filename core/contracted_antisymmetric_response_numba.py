"""Strict compiled contractions for compact analytical RFS responses."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from numba import njit

from .constants import C_MMNS

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0), dtype=np.float64)


def _permutation_sign(indices: tuple[int, int, int, int]) -> float:
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1.0 if inversions % 2 else 1.0


def _dual_packed_covariant() -> np.ndarray:
    epsilon_upper = np.zeros((4, 4, 4, 4), dtype=np.float64)
    for indices in permutations(range(4)):
        epsilon_upper[indices] = -_permutation_sign(indices)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    result = np.zeros((4, 4, 6), dtype=np.float64)
    for first in range(4):
        for second in range(4):
            for pair_index, (mu, nu) in enumerate(pairs):
                result[first, second, pair_index] = (
                    epsilon_upper[first, second, mu, nu]
                    * _SIGNS[first]
                    * _SIGNS[second]
                    * _SIGNS[mu]
                    * _SIGNS[nu]
                )
    return result


_DUAL_PACKED_COVARIANT = _dual_packed_covariant()


@njit(cache=True, fastmath=False, inline="always")
def _act_on_covector(packed: np.ndarray, covector: np.ndarray) -> np.ndarray:
    result = np.empty(4, dtype=np.float64)
    result[0] = (
        packed[0] * covector[1] + packed[1] * covector[2] + packed[2] * covector[3]
    )
    result[1] = (
        -packed[0] * covector[0] + packed[3] * covector[2] + packed[4] * covector[3]
    )
    result[2] = (
        -packed[1] * covector[0] - packed[3] * covector[1] + packed[5] * covector[3]
    )
    result[3] = (
        -packed[2] * covector[0] - packed[4] * covector[1] - packed[5] * covector[2]
    )
    return result


@njit(cache=True, fastmath=False)
def antisymmetric_response_charge_force_strict_serial(
    four_velocity_mm_ns: np.ndarray,
    antisymmetric_response: np.ndarray,
    charge_native: float,
) -> np.ndarray:
    """Return ``(q/c) F.u`` without Python/NumPy contraction overhead."""

    return (
        charge_native
        * _act_on_covector(
            antisymmetric_response,
            _SIGNS * four_velocity_mm_ns,
        )
        / C_MMNS
    )


@njit(cache=True, fastmath=False)
def antisymmetric_response_rfs_strict_serial(
    four_velocity_mm_ns: np.ndarray,
    spin_four_vector: np.ndarray,
    antisymmetric_response: np.ndarray,
    partial_antisymmetric_response: np.ndarray,
    charge_native: float,
    mass_amu: float,
    magnetic_moment_native: float,
    invariant_spin_native: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Contract one cached 34-value response into force and spin outputs."""

    velocity_covariant = _SIGNS * four_velocity_mm_ns
    spin_covariant = _SIGNS * spin_four_vector
    response_on_velocity = _act_on_covector(antisymmetric_response, velocity_covariant)
    response_on_spin = _act_on_covector(antisymmetric_response, spin_covariant)

    partial_b = np.zeros((4, 4), dtype=np.float64)
    for derivative in range(4):
        for response_index in range(4):
            total = 0.0
            for spin_index in range(4):
                for pair_index in range(6):
                    total += (
                        _DUAL_PACKED_COVARIANT[response_index, spin_index, pair_index]
                        * partial_antisymmetric_response[derivative, pair_index]
                        * spin_four_vector[spin_index]
                    )
            partial_b[derivative, response_index] = total

    g_on_velocity = np.empty(4, dtype=np.float64)
    g_on_spin = np.empty(4, dtype=np.float64)
    for mu in range(4):
        velocity_total = 0.0
        spin_total = 0.0
        for nu in range(4):
            g_covariant = partial_b[mu, nu] - partial_b[nu, mu]
            velocity_total += g_covariant * four_velocity_mm_ns[nu]
            spin_total += g_covariant * spin_four_vector[nu]
        g_on_velocity[mu] = _SIGNS[mu] * velocity_total
        g_on_spin[mu] = _SIGNS[mu] * spin_total

    charge_force = charge_native * response_on_velocity / C_MMNS
    dipole_force = magnetic_moment_native * g_on_velocity / C_MMNS
    total_force = charge_force + dipole_force
    u_dot_f_dot_s = 0.0
    for component in range(4):
        u_dot_f_dot_s += velocity_covariant[component] * response_on_spin[component]
    charge_to_mass_c = charge_native / (mass_amu * C_MMNS)
    moment_to_spin = magnetic_moment_native / invariant_spin_native
    spin_rhs = np.empty(4, dtype=np.float64)
    for component in range(4):
        orthogonal_response = response_on_spin[component] - (
            four_velocity_mm_ns[component] * u_dot_f_dot_s / C_MMNS**2
        )
        spin_rhs[component] = (
            charge_to_mass_c * response_on_spin[component]
            + (moment_to_spin - charge_to_mass_c) * orthogonal_response
            + magnetic_moment_native * g_on_spin[component] / (mass_amu * C_MMNS)
        )
    return charge_force, dipole_force, total_force, spin_rhs


__all__ = [
    "antisymmetric_response_charge_force_strict_serial",
    "antisymmetric_response_rfs_strict_serial",
]
