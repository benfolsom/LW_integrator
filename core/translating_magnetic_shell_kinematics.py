"""Field-free kinematics for a translating finite magnetic source.

The diagnostic source consists of two distinct, uniformly charged spherical
shells. Opposite charges and opposite angular velocities give zero net charge
while their magnetic moments add. Surface events lie on one Fermi--Walker
rest-space slice of a center moving along the x axis.

This module evaluates no electromagnetic fields and applies no force. It is a
kinematic oracle for a later finite-source radiation-reaction ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import C_MMNS, ELEMENTARY_CHARGE
from .external_fields import ELEMENTARY_CHARGE_COULOMB
from .magnetic_dipole import magnetic_moment_j_per_t_to_native
from .radiation_flux_oracle import gauss_legendre_sphere_quadrature

_FOUR_PI = 4.0 * np.pi


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("finite-source state contains a nonfinite value")
    result.flags.writeable = False
    return result


def _vector3(value: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _rotate_about_axis(
    vectors: np.ndarray, *, axis: np.ndarray, angle_rad: float
) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return (
        cosine * vectors
        + sine * np.cross(axis[np.newaxis, :], vectors)
        + (1.0 - cosine) * np.outer(vectors @ axis, axis)
    )


@dataclass(frozen=True)
class TranslatingMagneticShellSurfaceState:
    """Discrete shell events on one central comoving hypersurface.

    The surface weights are quadrature weights, not equal-area assumptions.
    ``rest_*`` arrays use the center's instantaneous inertial frame. Lab-frame
    events on this hypersurface are generally not simultaneous.
    """

    center_time_ns: float
    center_position_mm: np.ndarray
    center_beta_x: float
    center_proper_acceleration_mm_ns2: float
    event_time_ns: np.ndarray
    position_mm: np.ndarray
    beta: np.ndarray
    four_velocity_mm_ns: np.ndarray
    rest_position_mm: np.ndarray
    rest_velocity_mm_ns: np.ndarray
    charge_native: np.ndarray
    solid_angle_weight: np.ndarray
    shell_index: np.ndarray
    born_lapse: np.ndarray
    net_charge_native: float
    rest_electric_dipole_native_mm: np.ndarray
    rest_magnetic_moment_native: np.ndarray
    expected_rest_magnetic_moment_native: np.ndarray
    maximum_internal_beta: float
    maximum_born_rigidity_parameter: float

    def __post_init__(self) -> None:
        count = np.asarray(self.event_time_ns).size
        shapes = {
            "event_time_ns": (count,),
            "position_mm": (count, 3),
            "beta": (count, 3),
            "four_velocity_mm_ns": (count, 4),
            "rest_position_mm": (count, 3),
            "rest_velocity_mm_ns": (count, 3),
            "charge_native": (count,),
            "solid_angle_weight": (count,),
            "shell_index": (count,),
            "born_lapse": (count,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if name == "shell_index":
                indices = np.asarray(value, dtype=int).copy()
                indices.flags.writeable = False
                object.__setattr__(self, name, indices)
            else:
                object.__setattr__(self, name, _readonly(value))
        for name in (
            "center_position_mm",
            "rest_electric_dipole_native_mm",
            "rest_magnetic_moment_native",
            "expected_rest_magnetic_moment_native",
        ):
            value = _readonly(np.asarray(getattr(self, name), dtype=float))
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")
            object.__setattr__(self, name, value)


def build_counterrotating_shell_surface_state_native(
    *,
    center_time_ns: float,
    center_position_mm: Sequence[float],
    center_beta_x: float,
    center_proper_acceleration_mm_ns2: float,
    shell_radii_mm: Sequence[float],
    shell_charges_native: Sequence[float],
    shell_angular_velocities_per_ns: Sequence[float],
    rotation_axis_rest: Sequence[float],
    polar_order: int,
    azimuthal_order: int,
    shell_rotation_phases_rad: Sequence[float] = (0.0, 0.0),
) -> TranslatingMagneticShellSurfaceState:
    """Return two rotating shells on one collinear Fermi rest-space slice.

    Angular velocities are defined in the center's instantaneous inertial
    frame. The construction is exact for uniform translation. Under
    acceleration it is a declared small-source Born-rigid slice whose control
    parameter is ``abs(a) R / c**2``.
    """

    time = float(center_time_ns)
    center = _vector3(center_position_mm, name="center_position_mm")
    beta_x = float(center_beta_x)
    acceleration = float(center_proper_acceleration_mm_ns2)
    if not np.isfinite(time) or not np.isfinite(acceleration):
        raise ValueError("center time and acceleration must be finite")
    if not np.isfinite(beta_x) or abs(beta_x) >= 1.0:
        raise ValueError("center_beta_x must have magnitude below one")
    radii = np.asarray(shell_radii_mm, dtype=float)
    charges = np.asarray(shell_charges_native, dtype=float)
    angular_velocities = np.asarray(shell_angular_velocities_per_ns, dtype=float)
    phases = np.asarray(shell_rotation_phases_rad, dtype=float)
    for name, value in (
        ("shell_radii_mm", radii),
        ("shell_charges_native", charges),
        ("shell_angular_velocities_per_ns", angular_velocities),
        ("shell_rotation_phases_rad", phases),
    ):
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain two finite values")
    if np.any(radii <= 0.0) or radii[0] == radii[1]:
        raise ValueError("shell radii must be positive and distinct")
    charge_scale = float(np.max(np.abs(charges)))
    if charge_scale == 0.0 or abs(float(np.sum(charges))) > 2.0e-15 * charge_scale:
        raise ValueError("shell charges must be nonzero, opposite, and neutral")
    if charges[0] * charges[1] >= 0.0:
        raise ValueError("shell charges must have opposite signs")
    if (
        angular_velocities[0] == 0.0
        or angular_velocities[1] == 0.0
        or angular_velocities[0] * angular_velocities[1] >= 0.0
    ):
        raise ValueError("shell angular velocities must be nonzero and opposite")
    axis = _vector3(rotation_axis_rest, name="rotation_axis_rest")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.0:
        raise ValueError("rotation_axis_rest must be nonzero")
    axis = axis / axis_norm

    quadrature = gauss_legendre_sphere_quadrature(
        polar_order=polar_order, azimuthal_order=azimuthal_order
    )
    gamma = 1.0 / np.sqrt(1.0 - beta_x**2)
    rest_positions = []
    rest_velocities = []
    node_charges = []
    weights = []
    shell_indices = []
    for shell_index in range(2):
        directions = _rotate_about_axis(
            quadrature.directions,
            axis=axis,
            angle_rad=float(phases[shell_index]),
        )
        positions = radii[shell_index] * directions
        velocities = angular_velocities[shell_index] * np.cross(
            axis[np.newaxis, :], positions
        )
        rest_positions.append(positions)
        rest_velocities.append(velocities)
        weights.append(quadrature.solid_angle_weights)
        node_charges.append(
            charges[shell_index] * quadrature.solid_angle_weights / _FOUR_PI
        )
        shell_indices.append(np.full(quadrature.sample_count, shell_index, dtype=int))

    rest_position = np.concatenate(rest_positions)
    rest_velocity = np.concatenate(rest_velocities)
    charge = np.concatenate(node_charges)
    solid_angle_weight = np.concatenate(weights)
    shell_index_array = np.concatenate(shell_indices)
    internal_beta = rest_velocity / C_MMNS
    internal_beta_squared = np.sum(internal_beta**2, axis=1)
    if np.any(internal_beta_squared >= 1.0):
        raise ValueError("a shell surface element reaches or exceeds light speed")

    denominator = 1.0 + beta_x * internal_beta[:, 0]
    lab_beta = np.empty_like(internal_beta)
    lab_beta[:, 0] = (beta_x + internal_beta[:, 0]) / denominator
    lab_beta[:, 1:] = internal_beta[:, 1:] / (gamma * denominator[:, np.newaxis])
    lab_gamma = 1.0 / np.sqrt(1.0 - np.sum(lab_beta**2, axis=1))
    four_velocity = np.column_stack(
        (lab_gamma * C_MMNS, lab_gamma[:, None] * lab_beta * C_MMNS)
    )

    event_time = time + gamma * beta_x * rest_position[:, 0] / C_MMNS
    position = center[np.newaxis, :] + rest_position
    position[:, 0] = center[0] + gamma * rest_position[:, 0]
    born_lapse = 1.0 + acceleration * rest_position[:, 0] / C_MMNS**2
    if np.any(born_lapse <= 0.0):
        raise ValueError("shell crosses the collinear Fermi-coordinate horizon")

    electric_dipole = np.sum(charge[:, None] * rest_position, axis=0)
    charge_coulomb = charge * ELEMENTARY_CHARGE_COULOMB / ELEMENTARY_CHARGE
    moment_si = 0.5 * np.sum(
        charge_coulomb[:, None]
        * np.cross(rest_position * 1.0e-3, rest_velocity * 1.0e6),
        axis=0,
    )
    moment_native = np.array(
        [magnetic_moment_j_per_t_to_native(value) for value in moment_si]
    )
    expected_moment_si = np.sum(
        (
            charges
            * ELEMENTARY_CHARGE_COULOMB
            / ELEMENTARY_CHARGE
            * angular_velocities
            * 1.0e9
            * (radii * 1.0e-3) ** 2
            / 3.0
        )[:, None]
        * axis[np.newaxis, :],
        axis=0,
    )
    expected_moment_native = np.array(
        [magnetic_moment_j_per_t_to_native(value) for value in expected_moment_si]
    )

    return TranslatingMagneticShellSurfaceState(
        center_time_ns=time,
        center_position_mm=center,
        center_beta_x=beta_x,
        center_proper_acceleration_mm_ns2=acceleration,
        event_time_ns=event_time,
        position_mm=position,
        beta=lab_beta,
        four_velocity_mm_ns=four_velocity,
        rest_position_mm=rest_position,
        rest_velocity_mm_ns=rest_velocity,
        charge_native=charge,
        solid_angle_weight=solid_angle_weight,
        shell_index=shell_index_array,
        born_lapse=born_lapse,
        net_charge_native=float(np.sum(charge)),
        rest_electric_dipole_native_mm=electric_dipole,
        rest_magnetic_moment_native=moment_native,
        expected_rest_magnetic_moment_native=expected_moment_native,
        maximum_internal_beta=float(np.sqrt(np.max(internal_beta_squared))),
        maximum_born_rigidity_parameter=float(
            abs(acceleration) * np.max(radii) / C_MMNS**2
        ),
    )


__all__ = [
    "TranslatingMagneticShellSurfaceState",
    "build_counterrotating_shell_surface_state_native",
]
