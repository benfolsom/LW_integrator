"""Pure magnetic-dipole kinematics, tensor, and unit helpers.

The spin precession convention follows Bargmann, Michel, and Telegdi,
Phys. Rev. Lett. 2, 435 (1959), https://doi.org/10.1103/PhysRevLett.2.435.
The covariant magnetic-potential and Stern--Gerlach context follows Rafelski,
Formanek, and Steinmetz, Eur. Phys. J. C 78, 6 (2018),
https://doi.org/10.1140/epjc/s10052-017-5493-2 and
https://arxiv.org/abs/1712.01825.

This module intentionally does not choose a relativistic field-gradient
extension.  Such extensions are model-dependent and not unique in the cited
literature.  The Stern--Gerlach helper therefore implements only the explicit
static rest-frame limit ``F = grad(mu dot B)`` supplied by its caller.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union, cast

import numpy as np

from .constants import C_MMNS, ELEMENTARY_CHARGE
from .external_fields import (
    AMU_KG,
    ELEMENTARY_CHARGE_COULOMB,
    NATIVE_FORCE_UNIT_NEWTON,
    NS_PER_S,
)

C_M_S = C_MMNS * 1.0e6
"""Speed of light in metres per second, derived from the native constant."""

HBAR_J_S = 1.054571817e-34
"""Reduced Planck constant in J s (exact to the displayed CODATA digits)."""

NATIVE_MOMENTUM_UNIT_KG_M_S = NATIVE_FORCE_UNIT_NEWTON / NS_PER_S
"""One native momentum unit (amu mm/ns) in kg m/s."""

NATIVE_ENERGY_UNIT_J = NATIVE_FORCE_UNIT_NEWTON * 1.0e-3
"""One native energy unit (amu mm^2/ns^2) in joules."""

NATIVE_ACTION_UNIT_J_S = NATIVE_ENERGY_UNIT_J * 1.0e-9
"""One native action unit (amu mm^2/ns) in joule seconds."""

HBAR_NATIVE = HBAR_J_S / NATIVE_ACTION_UNIT_J_S
"""Reduced Planck constant in solver-native action units."""

ELECTRIC_FIELD_NATIVE_TO_V_PER_M = (
    ELEMENTARY_CHARGE * NATIVE_FORCE_UNIT_NEWTON / ELEMENTARY_CHARGE_COULOMB
)
"""SI electric field represented by one native field unit."""

MAGNETIC_FIELD_NATIVE_TO_TESLA = ELECTRIC_FIELD_NATIVE_TO_V_PER_M / C_M_S
"""SI magnetic field represented by one native magnetic-field unit."""

STATIC_REST_GRADIENT_MAX_BETA = 1.0e-2
"""Largest speed accepted by the non-covariant static-rest gradient model."""

VectorLike = Union[Sequence[float], np.ndarray]
MatrixLike = Union[Sequence[Sequence[float]], np.ndarray]


def _vector3(value: VectorLike, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError("{} must contain exactly three components".format(name))
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} must contain only finite values".format(name))
    return vector


def force_newton_to_native(value_newton: float) -> float:
    """Convert force in newtons to ``amu mm/ns^2``."""

    return float(value_newton) / NATIVE_FORCE_UNIT_NEWTON


def force_native_to_newton(value_native: float) -> float:
    """Convert force in ``amu mm/ns^2`` to newtons."""

    return float(value_native) * NATIVE_FORCE_UNIT_NEWTON


def momentum_kg_m_s_to_native(value_kg_m_s: float) -> float:
    """Convert SI momentum to the solver's ``amu mm/ns`` unit."""

    return float(value_kg_m_s) / NATIVE_MOMENTUM_UNIT_KG_M_S


def momentum_native_to_kg_m_s(value_native: float) -> float:
    """Convert solver-native momentum to kg m/s."""

    return float(value_native) * NATIVE_MOMENTUM_UNIT_KG_M_S


def electric_field_v_per_m_to_native(value_v_per_m: float) -> float:
    """Convert an SI electric field to native force-per-charge units."""

    return float(value_v_per_m) / ELECTRIC_FIELD_NATIVE_TO_V_PER_M


def electric_field_native_to_v_per_m(value_native: float) -> float:
    """Convert a native electric field to volts per metre."""

    return float(value_native) * ELECTRIC_FIELD_NATIVE_TO_V_PER_M


def magnetic_field_tesla_to_native(value_tesla: float) -> float:
    """Convert tesla to the native magnetic field used with ``beta x B``."""

    return float(value_tesla) / MAGNETIC_FIELD_NATIVE_TO_TESLA


def magnetic_field_native_to_tesla(value_native: float) -> float:
    """Convert the native magnetic field used with ``beta x B`` to tesla."""

    return float(value_native) * MAGNETIC_FIELD_NATIVE_TO_TESLA


def magnetic_moment_j_per_t_to_native(value_j_per_t: float) -> float:
    """Convert a measured SI moment to native energy per native magnetic field.

    The conversion intentionally uses the same exact native elementary-charge
    field scale as :func:`magnetic_field_tesla_to_native`. This keeps ``mu B``
    and ``mu grad(B)`` consistent with the existing external-field boundary.
    """

    return float(value_j_per_t) * MAGNETIC_FIELD_NATIVE_TO_TESLA / NATIVE_ENERGY_UNIT_J


def magnetic_moment_native_to_j_per_t(value_native: float) -> float:
    """Convert a native magnetic moment to joules per tesla."""

    return float(value_native) * NATIVE_ENERGY_UNIT_J / MAGNETIC_FIELD_NATIVE_TO_TESLA


def magnetic_gradient_t_per_m_to_native_per_mm(value_t_per_m: float) -> float:
    """Convert ``dB/dx`` from T/m to native magnetic field per millimetre."""

    return float(value_t_per_m) / MAGNETIC_FIELD_NATIVE_TO_TESLA * 1.0e-3


def magnetic_gradient_native_per_mm_to_t_per_m(value_native_per_mm: float) -> float:
    """Convert native magnetic field per millimetre to T/m."""

    return float(value_native_per_mm) * MAGNETIC_FIELD_NATIVE_TO_TESLA * 1.0e3


def electromagnetic_field_tensor(
    electric_field_v_m: VectorLike, magnetic_field_t: VectorLike
) -> np.ndarray:
    """Return contravariant ``F^(mu nu)`` for metric ``diag(+1,-1,-1,-1)``.

    Coordinates use ``x^0 = c t``.  The signs are selected so
    ``q F^(mu nu) u_nu`` has spatial part
    ``q gamma (E + v x B)``.
    """

    electric = _vector3(electric_field_v_m, name="electric_field_v_m")
    magnetic = _vector3(magnetic_field_t, name="magnetic_field_t")
    ex, ey, ez = electric / C_M_S
    bx, by, bz = magnetic
    return np.array(
        [
            [0.0, -ex, -ey, -ez],
            [ex, 0.0, -bz, by],
            [ey, bz, 0.0, -bx],
            [ez, -by, bx, 0.0],
        ],
        dtype=float,
    )


def fields_from_electromagnetic_tensor(
    tensor: MatrixLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover SI ``(E, B)`` from :func:`electromagnetic_field_tensor`."""

    field = np.asarray(tensor, dtype=float)
    if field.shape != (4, 4):
        raise ValueError("tensor must have shape (4, 4)")
    if not np.all(np.isfinite(field)):
        raise ValueError("tensor must contain only finite values")
    if not np.allclose(field + field.T, 0.0, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("electromagnetic field tensor must be antisymmetric")

    electric = -C_M_S * field[0, 1:4]
    magnetic = np.array((-field[2, 3], field[1, 3], -field[1, 2]))
    return electric, magnetic


def dual_electromagnetic_tensor(tensor: MatrixLike) -> np.ndarray:
    """Return the Hodge dual of an electromagnetic tensor.

    With the conventions in this module the dual maps
    ``(E, B) -> (-c B, E/c)`` and therefore ``dual(dual(F)) == -F``.
    """

    electric, magnetic = fields_from_electromagnetic_tensor(tensor)
    return electromagnetic_field_tensor(-C_M_S * magnetic, electric / C_M_S)


def minkowski_dot(left: VectorLike, right: VectorLike) -> float:
    """Return a four-vector inner product using signature ``(+---)``."""

    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    if left_vector.shape != (4,) or right_vector.shape != (4,):
        raise ValueError("minkowski_dot operands must have shape (4,)")
    return float(left_vector[0] * right_vector[0] - left_vector[1:] @ right_vector[1:])


def boost_rest_polarization(
    rest_polarization: VectorLike, beta: VectorLike
) -> np.ndarray:
    """Boost a rest-frame polarization vector into a spin four-vector.

    The returned dimensionless vector is
    ``S=(gamma beta.s, s + gamma^2/(gamma+1) (beta.s) beta)``.  It obeys
    ``S.u=0`` and ``S.S=-s.s`` for ``u/c=(gamma, gamma beta)``.
    """

    spin = _vector3(rest_polarization, name="rest_polarization")
    beta_vector = _vector3(beta, name="beta")
    beta_squared = float(beta_vector @ beta_vector)
    if beta_squared >= 1.0:
        raise ValueError("beta magnitude must be less than one")
    gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    projection = float(beta_vector @ spin)
    spatial = spin + (gamma * gamma / (gamma + 1.0)) * projection * beta_vector
    return cast(np.ndarray, np.concatenate(([gamma * projection], spatial)))


def rest_polarization_from_four_vector(
    spin_four_vector: VectorLike, beta: VectorLike
) -> np.ndarray:
    """Invert :func:`boost_rest_polarization` for a physical spin vector."""

    spin = np.asarray(spin_four_vector, dtype=float)
    if spin.shape != (4,) or not np.all(np.isfinite(spin)):
        raise ValueError("spin_four_vector must contain four finite components")
    beta_vector = _vector3(beta, name="beta")
    beta_squared = float(beta_vector @ beta_vector)
    if beta_squared >= 1.0:
        raise ValueError("beta magnitude must be less than one")
    gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    return cast(
        np.ndarray,
        spin[1:] - (gamma / (gamma + 1.0)) * spin[0] * beta_vector,
    )


def signed_gyromagnetic_ratio(
    magnetic_moment_j_t: float, spin_quantum_number: float
) -> float:
    """Return ``mu/(I hbar)`` in rad/s/T, preserving the moment sign."""

    moment = float(magnetic_moment_j_t)
    spin = float(spin_quantum_number)
    if spin < 0.0:
        raise ValueError("spin_quantum_number must be non-negative")
    if spin == 0.0:
        if moment == 0.0:
            return 0.0
        raise ValueError("a nonzero moment cannot be assigned to zero spin")
    return moment / (spin * HBAR_J_S)


def instantaneous_bmt_angular_velocity(
    *,
    beta: VectorLike,
    electric_field_v_m: VectorLike,
    magnetic_field_t: VectorLike,
    charge_coulomb: float,
    mass_kg: float,
    gyromagnetic_ratio_rad_s_t: float,
) -> np.ndarray:
    """Return lab-time spin angular velocity for charged or neutral particles.

    The rest-polarization vector evolves as ``ds/dt = omega x s``.  Written in
    terms of the *signed* gyromagnetic ratio ``gamma_s=mu/(I hbar)``, this is
    the usual BMT expression without a division by charge, so its neutral
    limit is finite.  For zero charge it reduces to rest-frame Larmor torque
    in a comoving Fermi--Walker frame, expressed per lab time::

        omega = -gamma_s [B - gamma/(gamma+1) beta(beta.B) - beta x E/c]

    Translational acceleration caused by a field gradient can add a separate
    Fermi--Walker/Thomas term; this instantaneous uniform-field helper does not
    infer that model-dependent term.
    """

    beta_vector = _vector3(beta, name="beta")
    electric = _vector3(electric_field_v_m, name="electric_field_v_m")
    magnetic = _vector3(magnetic_field_t, name="magnetic_field_t")
    beta_squared = float(beta_vector @ beta_vector)
    if beta_squared >= 1.0:
        raise ValueError("beta magnitude must be less than one")
    if mass_kg <= 0.0 or not np.isfinite(mass_kg):
        raise ValueError("mass_kg must be finite and positive")
    if not np.isfinite(charge_coulomb) or not np.isfinite(gyromagnetic_ratio_rad_s_t):
        raise ValueError("charge and gyromagnetic ratio must be finite")

    gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    charge_to_mass = charge_coulomb / mass_kg
    moment_excess = gyromagnetic_ratio_rad_s_t - charge_to_mass

    magnetic_coefficient = gyromagnetic_ratio_rad_s_t - charge_to_mass * (
        1.0 - 1.0 / gamma
    )
    longitudinal_coefficient = moment_excess * gamma / (gamma + 1.0)
    electric_coefficient = gyromagnetic_ratio_rad_s_t - charge_to_mass * gamma / (
        gamma + 1.0
    )

    bracket = (
        magnetic_coefficient * magnetic
        - longitudinal_coefficient * float(beta_vector @ magnetic) * beta_vector
        - electric_coefficient * np.cross(beta_vector, electric) / C_M_S
    )
    return cast(np.ndarray, -bracket)


def rotate_spin_rodrigues(
    spin: VectorLike, angular_velocity_rad_s: VectorLike, delta_time_s: float
) -> np.ndarray:
    """Advance a spin vector exactly for constant angular velocity."""

    spin_vector = _vector3(spin, name="spin")
    angular_velocity = _vector3(angular_velocity_rad_s, name="angular_velocity_rad_s")
    if not np.isfinite(delta_time_s):
        raise ValueError("delta_time_s must be finite")
    angular_speed = float(np.linalg.norm(angular_velocity))
    if angular_speed == 0.0 or delta_time_s == 0.0:
        return spin_vector.copy()

    axis = angular_velocity / angular_speed
    angle = angular_speed * float(delta_time_s)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return cast(
        np.ndarray,
        (
            spin_vector * cosine
            + np.cross(axis, spin_vector) * sine
            + axis * float(axis @ spin_vector) * (1.0 - cosine)
        ),
    )


def advance_spin_uniform_fields(
    spin: VectorLike,
    *,
    beta: VectorLike,
    electric_field_v_m: VectorLike,
    magnetic_field_t: VectorLike,
    charge_coulomb: float,
    mass_kg: float,
    gyromagnetic_ratio_rad_s_t: float,
    delta_time_s: float,
) -> np.ndarray:
    """Advance rest polarization through a uniform instantaneous BMT step."""

    angular_velocity = instantaneous_bmt_angular_velocity(
        beta=beta,
        electric_field_v_m=electric_field_v_m,
        magnetic_field_t=magnetic_field_t,
        charge_coulomb=charge_coulomb,
        mass_kg=mass_kg,
        gyromagnetic_ratio_rad_s_t=gyromagnetic_ratio_rad_s_t,
    )
    return rotate_spin_rodrigues(spin, angular_velocity, delta_time_s)


def stern_gerlach_rest_force_newton(
    magnetic_moment_vector_j_t: VectorLike,
    magnetic_field_gradient_t_per_m: MatrixLike,
) -> np.ndarray:
    """Return the static rest-limit force ``grad(mu dot B)`` in newtons.

    The gradient must be supplied explicitly with indexing
    ``gradient[field_component, coordinate] = dB_i/dx_j``.  This function does
    not estimate retarded gradients and does not select among inequivalent
    relativistic Stern--Gerlach models.
    """

    moment = _vector3(magnetic_moment_vector_j_t, name="magnetic_moment_vector_j_t")
    gradient = np.asarray(magnetic_field_gradient_t_per_m, dtype=float)
    if gradient.shape != (3, 3):
        raise ValueError("magnetic_field_gradient_t_per_m must have shape (3, 3)")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("magnetic field gradient must contain only finite values")
    return cast(np.ndarray, gradient.T @ moment)


def stern_gerlach_rest_impulse_native(
    magnetic_moment_vector_j_t: VectorLike,
    magnetic_field_gradient_t_per_m: MatrixLike,
    proper_time_step_ns: float,
) -> Tuple[float, float, float]:
    """Return a native momentum impulse for the static particle rest limit.

    At rest, proper and lab time coincide.  Callers must not use this helper as
    an undocumented relativistic gradient-force prescription.
    """

    if not np.isfinite(proper_time_step_ns):
        raise ValueError("proper_time_step_ns must be finite")
    force_newton = stern_gerlach_rest_force_newton(
        magnetic_moment_vector_j_t, magnetic_field_gradient_t_per_m
    )
    impulse_native = (
        force_newton / NATIVE_FORCE_UNIT_NEWTON * float(proper_time_step_ns)
    )
    return cast(
        Tuple[float, float, float],
        tuple(float(component) for component in impulse_native),
    )


__all__ = [
    "C_M_S",
    "ELECTRIC_FIELD_NATIVE_TO_V_PER_M",
    "HBAR_J_S",
    "MAGNETIC_FIELD_NATIVE_TO_TESLA",
    "NATIVE_ACTION_UNIT_J_S",
    "NATIVE_ENERGY_UNIT_J",
    "NATIVE_MOMENTUM_UNIT_KG_M_S",
    "HBAR_NATIVE",
    "STATIC_REST_GRADIENT_MAX_BETA",
    "advance_spin_uniform_fields",
    "boost_rest_polarization",
    "dual_electromagnetic_tensor",
    "electric_field_native_to_v_per_m",
    "electric_field_v_per_m_to_native",
    "electromagnetic_field_tensor",
    "fields_from_electromagnetic_tensor",
    "force_native_to_newton",
    "force_newton_to_native",
    "instantaneous_bmt_angular_velocity",
    "magnetic_field_native_to_tesla",
    "magnetic_field_tesla_to_native",
    "magnetic_gradient_native_per_mm_to_t_per_m",
    "magnetic_gradient_t_per_m_to_native_per_mm",
    "magnetic_moment_j_per_t_to_native",
    "magnetic_moment_native_to_j_per_t",
    "minkowski_dot",
    "momentum_kg_m_s_to_native",
    "momentum_native_to_kg_m_s",
    "rest_polarization_from_four_vector",
    "rotate_spin_rodrigues",
    "signed_gyromagnetic_ratio",
    "stern_gerlach_rest_force_newton",
    "stern_gerlach_rest_impulse_native",
]
