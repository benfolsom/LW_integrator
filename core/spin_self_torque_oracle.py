"""Diagnostic pure-magnetic point-dipole self-torque comparators.

The finite neutral-shell oracle approaches the rest-frame reaction torque

``N_RR = mu_0 * mu x d^3(mu)/dt^3 / (6*pi*c^3)``.

This module gives that result its minimal Lorentz-covariant extension along an
inertial worldline.  It also provides a separately named translation of the
torque-relevant part of Unruh, Phys. Rev. A 59, 131 (1999), Eq. (60), for the
paper's planar-acceleration model.  Both are comparators, not production laws.
The accelerated result still requires an independent moving-source balance
test, a genuinely three-dimensional derivation, and reduction of order.
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


@dataclass(frozen=True)
class UnruhPlanarAcceleratedDipoleTorqueComparatorResult:
    """Shell-normalized translation of Unruh's accelerated reaction field.

    All torque four-vectors use native energy units.  The first and third
    magnetic-moment derivatives are Fermi--Walker derivatives with respect to
    proper time in nanoseconds.
    ``proper_acceleration_squared_over_c2_per_ns2`` is ``-A.A/c^2`` and is
    nonnegative for a physical four-acceleration.

    The result remains a comparator because Unruh's derivation restricts the
    acceleration to a plane and does not supply the finite moving-source
    balance test required by this project.
    """

    inertial_spin_torque_native: np.ndarray
    acceleration_spin_torque_native: np.ndarray
    total_spin_torque_native: np.ndarray
    torque_driver_native_per_ns3: np.ndarray
    proper_acceleration_squared_over_c2_per_ns2: float
    four_velocity_dot_total_torque_native: float
    magnetic_moment_dot_total_torque_native: float
    four_velocity_dot_acceleration_native: float
    planar_acceleration_derivation_only: bool
    reduction_of_order_performed: bool


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


def evaluate_unruh_planar_accelerated_dipole_torque_comparator_native(
    *,
    four_velocity_mm_ns: VectorLike,
    four_acceleration_mm_ns2: VectorLike,
    magnetic_moment_four_vector_native: VectorLike,
    magnetic_moment_first_fermi_walker_derivative_native: VectorLike,
    magnetic_moment_third_fermi_walker_derivative_native: VectorLike,
) -> UnruhPlanarAcceleratedDipoleTorqueComparatorResult:
    """Evaluate the torque-relevant part of Unruh's Eq. (60).

    Unruh parameterizes the worldline by length ``u=c*tau``.  Restoring proper
    time and normalizing the inertial term to the finite-shell point limit
    makes the reaction-field driver, up to a term parallel to the moment,

    ``mu'''_FW + (3/2) * (-A.A/c^2) * mu'_FW``.

    The omitted term is proportional to ``mu * f * df/du`` and therefore
    gives exactly zero in ``mu x B_RR``.  It may still matter for other field
    couplings and is not claimed to vanish from the complete self-field.

    Every supplied moment derivative must already be a Fermi--Walker spatial
    vector orthogonal to ``u``.  No high derivative is generated and no
    reduction of order is performed here.
    """

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    acceleration = _four_vector(
        four_acceleration_mm_ns2,
        name="four_acceleration_mm_ns2",
    )
    moment = _four_vector(
        magnetic_moment_four_vector_native,
        name="magnetic_moment_four_vector_native",
    )
    moment_first = _four_vector(
        magnetic_moment_first_fermi_walker_derivative_native,
        name="magnetic_moment_first_fermi_walker_derivative_native",
    )
    moment_third = _four_vector(
        magnetic_moment_third_fermi_walker_derivative_native,
        name="magnetic_moment_third_fermi_walker_derivative_native",
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

    velocity_dot_acceleration = minkowski_dot(velocity, acceleration)
    acceleration_scale = C_MMNS * max(float(np.linalg.norm(acceleration)), 1.0e-300)
    if abs(velocity_dot_acceleration) > 2.0e-12 * acceleration_scale:
        raise ValueError("four-acceleration must satisfy u.A = 0")

    for name, vector in (
        ("magnetic moment", moment),
        ("first Fermi-Walker moment derivative", moment_first),
        ("third Fermi-Walker moment derivative", moment_third),
    ):
        residual = minkowski_dot(velocity, vector)
        scale = C_MMNS * max(float(np.linalg.norm(vector)), 1.0e-300)
        if abs(residual) > 2.0e-12 * scale:
            raise ValueError(f"{name} must be orthogonal to four-velocity")

    acceleration_norm = minkowski_dot(acceleration, acceleration)
    if acceleration_norm > 2.0e-12 * max(float(acceleration @ acceleration), 1.0e-300):
        raise ValueError("four-acceleration must be spacelike or zero")
    acceleration_rate_squared = max(0.0, -acceleration_norm / C_MMNS**2)
    acceleration_driver = 1.5 * acceleration_rate_squared * moment_first
    driver = moment_third + acceleration_driver
    normalized_velocity = velocity / C_MMNS
    inertial_torque = _POINT_TORQUE_COEFFICIENT_NATIVE * body_frame_cross(
        moment,
        moment_third,
        normalized_velocity,
    )
    acceleration_torque = _POINT_TORQUE_COEFFICIENT_NATIVE * body_frame_cross(
        moment,
        acceleration_driver,
        normalized_velocity,
    )
    total_torque = inertial_torque + acceleration_torque
    arrays = (inertial_torque, acceleration_torque, total_torque, driver)
    for array in arrays:
        array.setflags(write=False)

    return UnruhPlanarAcceleratedDipoleTorqueComparatorResult(
        inertial_spin_torque_native=arrays[0],
        acceleration_spin_torque_native=arrays[1],
        total_spin_torque_native=arrays[2],
        torque_driver_native_per_ns3=arrays[3],
        proper_acceleration_squared_over_c2_per_ns2=acceleration_rate_squared,
        four_velocity_dot_total_torque_native=minkowski_dot(velocity, total_torque),
        magnetic_moment_dot_total_torque_native=minkowski_dot(moment, total_torque),
        four_velocity_dot_acceleration_native=velocity_dot_acceleration,
        planar_acceleration_derivation_only=True,
        reduction_of_order_performed=False,
    )


__all__ = [
    "InertialMagneticDipoleSelfTorqueResult",
    "UnruhPlanarAcceleratedDipoleTorqueComparatorResult",
    "evaluate_inertial_point_magnetic_dipole_self_torque_native",
    "evaluate_unruh_planar_accelerated_dipole_torque_comparator_native",
]
