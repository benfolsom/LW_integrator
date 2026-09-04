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
    material_proper_time_lapse: np.ndarray
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
            "material_proper_time_lapse": (count,),
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
        if np.any(self.material_proper_time_lapse <= 0.0):
            raise ValueError("material proper-time lapse must be positive")


@dataclass(frozen=True)
class TranslatingMagneticShellHistory:
    """Continuous material-node histories assembled from Fermi slices.

    ``beta_prime_per_mm`` contains instantaneous knot derivatives. Consumers
    must request ``source_acceleration_semantics="instantaneous"`` when using
    those values. The default charge provider reconstructs derivatives from
    positions and velocities and therefore remains an independent route.
    """

    center_proper_time_ns: np.ndarray
    event_time_ns: np.ndarray
    position_mm: np.ndarray
    beta: np.ndarray
    beta_prime_per_mm: np.ndarray
    material_proper_time_lapse: np.ndarray
    charge_native: np.ndarray
    shell_index: np.ndarray
    maximum_velocity_derivative_residual_mm_ns: float
    relative_velocity_derivative_residual: float

    def __post_init__(self) -> None:
        proper_time = np.asarray(self.center_proper_time_ns, dtype=float)
        event_time = np.asarray(self.event_time_ns, dtype=float)
        if proper_time.ndim != 1 or proper_time.size < 3:
            raise ValueError("center_proper_time_ns must contain at least three values")
        step_count = proper_time.size
        if event_time.ndim != 2 or event_time.shape[0] != step_count:
            raise ValueError("event_time_ns must have shape (steps, nodes)")
        node_count = event_time.shape[1]
        shapes = {
            "position_mm": (step_count, node_count, 3),
            "beta": (step_count, node_count, 3),
            "beta_prime_per_mm": (step_count, node_count, 3),
            "material_proper_time_lapse": (step_count, node_count),
            "charge_native": (node_count,),
            "shell_index": (node_count,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if np.any(np.diff(proper_time) <= 0.0):
            raise ValueError("center proper times must increase strictly")
        if np.any(np.diff(event_time, axis=0) <= 0.0):
            raise ValueError("every material-node laboratory time must increase")
        if np.any(np.asarray(self.material_proper_time_lapse) <= 0.0):
            raise ValueError("material proper-time lapse must be positive")
        residuals = (
            self.maximum_velocity_derivative_residual_mm_ns,
            self.relative_velocity_derivative_residual,
        )
        if not np.all(np.isfinite(residuals)) or np.any(np.asarray(residuals) < 0.0):
            raise ValueError(
                "velocity derivative residuals must be finite and nonnegative"
            )
        object.__setattr__(self, "center_proper_time_ns", _readonly(proper_time))
        object.__setattr__(self, "event_time_ns", _readonly(event_time))
        for name in (
            "position_mm",
            "beta",
            "beta_prime_per_mm",
            "material_proper_time_lapse",
            "charge_native",
        ):
            object.__setattr__(self, name, _readonly(np.asarray(getattr(self, name))))
        indices = np.asarray(self.shell_index, dtype=int).copy()
        indices.flags.writeable = False
        object.__setattr__(self, "shell_index", indices)

    def as_charge_provider_history(self) -> list[dict[str, np.ndarray]]:
        """Return the sampled worldlines in the charge-provider input format."""

        result = []
        for step in range(self.center_proper_time_ns.size):
            result.append(
                {
                    "t": self.event_time_ns[step],
                    "x": self.position_mm[step, :, 0],
                    "y": self.position_mm[step, :, 1],
                    "z": self.position_mm[step, :, 2],
                    "bx": self.beta[step, :, 0],
                    "by": self.beta[step, :, 1],
                    "bz": self.beta[step, :, 2],
                    "bdotx": self.beta_prime_per_mm[step, :, 0],
                    "bdoty": self.beta_prime_per_mm[step, :, 1],
                    "bdotz": self.beta_prime_per_mm[step, :, 2],
                    "q": self.charge_native,
                    "q_source": self.charge_native,
                    "_dead_particles": np.zeros(self.charge_native.size, dtype=bool),
                }
            )
        return result


@dataclass(frozen=True)
class TranslatingMagneticShellFourKinematics:
    """Material four-velocity and its first two node-proper-time derivatives."""

    four_velocity_mm_ns: np.ndarray
    four_acceleration_mm_ns2: np.ndarray
    four_jerk_mm_ns3: np.ndarray


def _differentiate_on_node_times(
    values: np.ndarray, event_time_ns: np.ndarray, *, coordinate_scale: float
) -> np.ndarray:
    """Differentiate each material history with a five-knot local polynomial."""

    sample_count = event_time_ns.shape[0]
    if sample_count < 5:
        raise ValueError("material histories require at least five time samples")
    knot = np.arange(sample_count)
    starts = np.clip(knot - 2, 0, sample_count - 5)
    stencil_indices = starts[:, np.newaxis] + np.arange(5)
    powers = np.arange(5)[np.newaxis, :, np.newaxis]
    derivative = np.empty_like(values)
    for node in range(event_time_ns.shape[1]):
        coordinates = coordinate_scale * event_time_ns[:, node]
        offsets = coordinates[stencil_indices] - coordinates[:, np.newaxis]
        scale = np.max(np.abs(offsets), axis=1)
        normalized = offsets / scale[:, np.newaxis]
        system = normalized[:, np.newaxis, :] ** powers
        right_hand_side = np.zeros((sample_count, 5))
        right_hand_side[:, 1] = 1.0 / scale
        weights = np.linalg.solve(system, right_hand_side)
        centered_values = values[stencil_indices, node] - values[:, node, np.newaxis]
        derivative[:, node] = np.einsum("ki,kic->kc", weights, centered_values)
    return derivative


def evaluate_shell_history_four_kinematics_native(
    history: TranslatingMagneticShellHistory,
) -> TranslatingMagneticShellFourKinematics:
    """Differentiate material four-velocity with respect to node proper time."""

    beta = history.beta
    gamma = 1.0 / np.sqrt(1.0 - np.sum(beta**2, axis=2))
    four_velocity = (
        gamma[..., np.newaxis]
        * C_MMNS
        * np.concatenate((np.ones((*beta.shape[:2], 1)), beta), axis=2)
    )
    central_times = np.broadcast_to(
        history.center_proper_time_ns[:, np.newaxis], history.event_time_ns.shape
    )
    velocity_derivative_per_center_time = _differentiate_on_node_times(
        four_velocity,
        central_times,
        coordinate_scale=1.0,
    )
    four_acceleration = velocity_derivative_per_center_time / (
        history.material_proper_time_lapse[..., np.newaxis]
    )
    acceleration_derivative_per_center_time = _differentiate_on_node_times(
        four_acceleration,
        central_times,
        coordinate_scale=1.0,
    )
    four_jerk = acceleration_derivative_per_center_time / (
        history.material_proper_time_lapse[..., np.newaxis]
    )
    return TranslatingMagneticShellFourKinematics(
        four_velocity_mm_ns=_readonly(four_velocity),
        four_acceleration_mm_ns2=_readonly(four_acceleration),
        four_jerk_mm_ns3=_readonly(four_jerk),
    )


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

    Angular velocities are phase rates with respect to the center's proper
    time. The local physical tangential velocity is divided by the Fermi lapse
    ``1 + a xi / c**2``. The construction is exact for uniform translation.
    Under acceleration it is a declared small-source Born-rigid slice whose
    control parameter is ``abs(a) R / c**2``.
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
    both_stationary = bool(np.all(angular_velocities == 0.0))
    counterrotating = bool(
        np.all(angular_velocities != 0.0)
        and angular_velocities[0] * angular_velocities[1] < 0.0
    )
    if not (both_stationary or counterrotating):
        raise ValueError(
            "shell angular velocities must both vanish or be nonzero and opposite"
        )
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
    born_lapses = []
    for shell_index in range(2):
        directions = _rotate_about_axis(
            quadrature.directions,
            axis=axis,
            angle_rad=float(phases[shell_index]),
        )
        positions = radii[shell_index] * directions
        lapse = 1.0 + acceleration * positions[:, 0] / C_MMNS**2
        if np.any(lapse <= 0.0):
            raise ValueError("shell crosses the collinear Fermi-coordinate horizon")
        coordinate_velocities = angular_velocities[shell_index] * np.cross(
            axis[np.newaxis, :], positions
        )
        velocities = coordinate_velocities / lapse[:, np.newaxis]
        rest_positions.append(positions)
        rest_velocities.append(velocities)
        born_lapses.append(lapse)
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
    born_lapse = np.concatenate(born_lapses)
    internal_beta = rest_velocity / C_MMNS
    internal_beta_squared = np.sum(internal_beta**2, axis=1)
    if np.any(internal_beta_squared >= 1.0):
        raise ValueError("a shell surface element reaches or exceeds light speed")
    material_proper_time_lapse = born_lapse * np.sqrt(1.0 - internal_beta_squared)

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
        material_proper_time_lapse=material_proper_time_lapse,
        net_charge_native=float(np.sum(charge)),
        rest_electric_dipole_native_mm=electric_dipole,
        rest_magnetic_moment_native=moment_native,
        expected_rest_magnetic_moment_native=expected_moment_native,
        maximum_internal_beta=float(np.sqrt(np.max(internal_beta_squared))),
        maximum_born_rigidity_parameter=float(
            abs(acceleration) * np.max(radii) / C_MMNS**2
        ),
    )


def build_constant_rotation_shell_history_native(
    *,
    center_proper_times_ns: Sequence[float],
    center_times_ns: Sequence[float],
    center_positions_mm: Sequence[Sequence[float]],
    center_beta_x: Sequence[float],
    center_proper_accelerations_mm_ns2: Sequence[float],
    shell_radii_mm: Sequence[float],
    shell_charges_native: Sequence[float],
    shell_angular_velocities_per_ns: Sequence[float],
    rotation_axis_rest: Sequence[float],
    polar_order: int,
    azimuthal_order: int,
    initial_shell_rotation_phases_rad: Sequence[float] = (0.0, 0.0),
) -> TranslatingMagneticShellHistory:
    """Build per-node histories for constant rotation in central proper time."""

    proper_times = np.asarray(center_proper_times_ns, dtype=float)
    center_times = np.asarray(center_times_ns, dtype=float)
    center_positions = np.asarray(center_positions_mm, dtype=float)
    center_betas = np.asarray(center_beta_x, dtype=float)
    accelerations = np.asarray(center_proper_accelerations_mm_ns2, dtype=float)
    if proper_times.ndim != 1 or proper_times.size < 5:
        raise ValueError("center_proper_times_ns must contain at least five values")
    count = proper_times.size
    for name, value, shape in (
        ("center_times_ns", center_times, (count,)),
        ("center_positions_mm", center_positions, (count, 3)),
        ("center_beta_x", center_betas, (count,)),
        ("center_proper_accelerations_mm_ns2", accelerations, (count,)),
    ):
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must have finite shape {shape}")
    if not np.all(np.isfinite(proper_times)) or np.any(np.diff(proper_times) <= 0.0):
        raise ValueError("center proper times must be finite and increasing")
    angular_velocities = np.asarray(shell_angular_velocities_per_ns, dtype=float)
    initial_phases = np.asarray(initial_shell_rotation_phases_rad, dtype=float)
    if angular_velocities.shape != (2,) or initial_phases.shape != (2,):
        raise ValueError("shell angular velocities and phases must contain two values")

    states = []
    for index, proper_time in enumerate(proper_times):
        phases = initial_phases + angular_velocities * (proper_time - proper_times[0])
        states.append(
            build_counterrotating_shell_surface_state_native(
                center_time_ns=float(center_times[index]),
                center_position_mm=center_positions[index],
                center_beta_x=float(center_betas[index]),
                center_proper_acceleration_mm_ns2=float(accelerations[index]),
                shell_radii_mm=shell_radii_mm,
                shell_charges_native=shell_charges_native,
                shell_angular_velocities_per_ns=angular_velocities,
                rotation_axis_rest=rotation_axis_rest,
                polar_order=polar_order,
                azimuthal_order=azimuthal_order,
                shell_rotation_phases_rad=phases,
            )
        )

    event_time = np.stack([state.event_time_ns for state in states])
    position = np.stack([state.position_mm for state in states])
    beta = np.stack([state.beta for state in states])
    material_proper_time_lapse = np.stack(
        [state.material_proper_time_lapse for state in states]
    )
    first = states[0]
    for state in states[1:]:
        if not np.array_equal(state.charge_native, first.charge_native):
            raise ArithmeticError("surface-node charges changed between slices")
        if not np.array_equal(state.shell_index, first.shell_index):
            raise ArithmeticError("surface-node identities changed between slices")

    beta_prime = _differentiate_on_node_times(beta, event_time, coordinate_scale=C_MMNS)
    numerical_velocity = _differentiate_on_node_times(
        position, event_time, coordinate_scale=1.0
    )
    velocity_residual = numerical_velocity - C_MMNS * beta
    maximum_residual = float(np.max(np.linalg.norm(velocity_residual, axis=2)))
    velocity_scale = max(
        float(np.max(np.linalg.norm(C_MMNS * beta, axis=2))),
        np.finfo(float).tiny,
    )
    return TranslatingMagneticShellHistory(
        center_proper_time_ns=proper_times,
        event_time_ns=event_time,
        position_mm=position,
        beta=beta,
        beta_prime_per_mm=beta_prime,
        material_proper_time_lapse=material_proper_time_lapse,
        charge_native=first.charge_native,
        shell_index=first.shell_index,
        maximum_velocity_derivative_residual_mm_ns=maximum_residual,
        relative_velocity_derivative_residual=maximum_residual / velocity_scale,
    )


__all__ = [
    "TranslatingMagneticShellFourKinematics",
    "TranslatingMagneticShellHistory",
    "TranslatingMagneticShellSurfaceState",
    "build_constant_rotation_shell_history_native",
    "build_counterrotating_shell_surface_state_native",
    "evaluate_shell_history_four_kinematics_native",
]
