"""Diagnostic pure-magnetic point-dipole self-torque for inertial motion.

The finite neutral-shell oracle approaches the rest-frame reaction torque

``N_RR = mu_0 * mu x d^3(mu)/dt^3 / (6*pi*c^3)``.

This module gives that result its minimal Lorentz-covariant extension along an
inertial worldline.  It is deliberately not an accelerated-particle law.
Unruh, Phys. Rev. A 59, 131 (1999), Eq. (60), shows that acceleration adds
terms involving Fermi--Walker derivatives and the proper acceleration.  Those
terms require a separate derivation, unit audit, and finite-source comparison
before this oracle can be used on a production trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from .constants import C_MMNS
from .magnetic_dipole import (
    NATIVE_ENERGY_UNIT_J,
    magnetic_moment_native_to_j_per_t,
    minkowski_dot,
)
from .spin_self_force_oracle import body_frame_cross


VectorLike = Union[Sequence[float], np.ndarray]

_C_M_S = C_MMNS * 1.0e6
_MU_0_SI = 4.0 * np.pi * 1.0e-7
_NS_PER_S = 1.0e9
_MOMENT_NATIVE_TO_J_PER_T = magnetic_moment_native_to_j_per_t(1.0)
_POINT_TORQUE_COEFFICIENT_NATIVE = (
    _MU_0_SI
    / (6.0 * np.pi * _C_M_S**3)
    * _MOMENT_NATIVE_TO_J_PER_T**2
    * _NS_PER_S**3
    / NATIVE_ENERGY_UNIT_J
)


def _four_vector(value: VectorLike, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


@dataclass(frozen=True)
class InertialMagneticDipoleSelfTorqueResult:
    """Point-limit torque and its covariant consistency residuals.

    The torque is a four-vector in native energy units.  Inputs use the
    integrator's magnetic-moment unit, and the third derivative is with
    respect to proper time measured in nanoseconds.
    """

    spin_torque_native: np.ndarray
    point_torque_coefficient_native: float
    four_velocity_dot_torque_native: float
    magnetic_moment_dot_torque_native: float
    four_velocity_dot_magnetic_moment_native: float
    inertial_worldline_only: bool


def evaluate_inertial_point_magnetic_dipole_self_torque_native(
    *,
    four_velocity_mm_ns: VectorLike,
    magnetic_moment_four_vector_native: VectorLike,
    magnetic_moment_third_proper_derivative_native: VectorLike,
) -> InertialMagneticDipoleSelfTorqueResult:
    """Evaluate the covariant inertial point-dipole reaction torque.

    With ``w = u/c`` and the project's ``(+---)`` Levi-Civita convention,

    ``N^mu = K epsilon^mu_(nu rho sigma) mu^nu mu'''^rho w^sigma``.

    Here ``K`` is the SI point coefficient converted to native units.  In the
    instantaneous rest frame this is exactly
    ``mu_0 * mu x mu''' / (6*pi*c^3)``.  The construction is orthogonal to
    both four-velocity and magnetic moment, so it preserves ``u.S = 0`` and
    fixed spin magnitude when magnetic moment is proportional to spin.

    This function accepts no acceleration because the formula is valid only
    for an inertial worldline.  It must not be used as an approximation to
    the accelerated law without separately establishing that Unruh's
    acceleration-dependent terms are negligible.
    """

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    moment = _four_vector(
        magnetic_moment_four_vector_native,
        name="magnetic_moment_four_vector_native",
    )
    moment_third = _four_vector(
        magnetic_moment_third_proper_derivative_native,
        name="magnetic_moment_third_proper_derivative_native",
    )
    if velocity[0] <= 0.0:
        raise ValueError("four_velocity_mm_ns must be future-directed")
    velocity_norm = minkowski_dot(velocity, velocity)
    if not np.isclose(
        velocity_norm,
        C_MMNS**2,
        rtol=2.0e-12,
        atol=2.0e-12 * C_MMNS**2,
    ):
        raise ValueError("four_velocity_mm_ns must satisfy u.u = c^2")

    velocity_dot_moment = minkowski_dot(velocity, moment)
    orthogonality_scale = C_MMNS * max(float(np.linalg.norm(moment)), 1.0e-300)
    if abs(velocity_dot_moment) > 2.0e-12 * orthogonality_scale:
        raise ValueError("magnetic moment must satisfy u.mu = 0")

    torque = _POINT_TORQUE_COEFFICIENT_NATIVE * body_frame_cross(
        moment,
        moment_third,
        velocity / C_MMNS,
    )
    torque.setflags(write=False)
    return InertialMagneticDipoleSelfTorqueResult(
        spin_torque_native=torque,
        point_torque_coefficient_native=_POINT_TORQUE_COEFFICIENT_NATIVE,
        four_velocity_dot_torque_native=minkowski_dot(velocity, torque),
        magnetic_moment_dot_torque_native=minkowski_dot(moment, torque),
        four_velocity_dot_magnetic_moment_native=velocity_dot_moment,
        inertial_worldline_only=True,
    )


__all__ = [
    "InertialMagneticDipoleSelfTorqueResult",
    "evaluate_inertial_point_magnetic_dipole_self_torque_native",
]
