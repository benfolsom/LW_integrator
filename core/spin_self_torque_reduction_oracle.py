"""Sampled reduction reference for the pure-magnetic self-torque.

The accelerated point comparator needs first and third Fermi--Walker
derivatives of the magnetic moment.  This module reconstructs them from a
short stencil of leading-order, non-self trajectory samples.  It is an
independent convergence oracle, not an online production implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import C_MMNS
from .magnetic_dipole import minkowski_dot
from .spin_self_force_reduction_oracle import (
    ArrayLike,
    _finite_difference_weights,
    _sample_matrix,
)
from .spin_self_torque_oracle import (
    UnruhPlanarAcceleratedDipoleTorqueComparatorResult,
    evaluate_unruh_planar_accelerated_dipole_torque_comparator_native,
)

_MINKOWSKI_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0))


def _lower(vector: np.ndarray) -> np.ndarray:
    return _MINKOWSKI_SIGNS * vector


def _rest_projector(normalized_velocity: np.ndarray) -> np.ndarray:
    return np.eye(4) - np.outer(normalized_velocity, _lower(normalized_velocity))


@dataclass(frozen=True)
class SampledFermiWalkerMagneticTorqueReductionResult:
    """Seven-or-more-sample Fermi--Walker reconstruction and torque.

    The reconstructed acceleration is projected into the rest space, and the
    reconstructed second velocity derivative is corrected only along the
    four-velocity so that it obeys the twice-differentiated normalization
    constraint.  The two residual fields report the uncorrected violations.
    """

    center_index: int
    evaluation_proper_time_ns: float
    sample_time_span_ns: float
    stencil_kind: str
    uses_future_samples: bool
    scaled_vandermonde_condition_number: float
    first_derivative_weights_per_ns: np.ndarray
    second_derivative_weights_per_ns2: np.ndarray
    third_derivative_weights_per_ns3: np.ndarray
    reconstructed_four_acceleration_mm_ns2: np.ndarray
    reconstructed_four_jerk_mm_ns3: np.ndarray
    magnetic_moment_first_fermi_walker_derivative_native: np.ndarray
    magnetic_moment_third_fermi_walker_derivative_native: np.ndarray
    velocity_normalization_first_derivative_residual_per_ns: float
    velocity_normalization_second_derivative_residual_per_ns2: float
    maximum_sample_velocity_norm_residual_mm2_ns2: float
    maximum_sample_velocity_moment_residual_native_mm_ns: float
    torque_comparator: UnruhPlanarAcceleratedDipoleTorqueComparatorResult
    leading_non_self_samples_required: bool
    reduction_of_order_reference: bool


def _evaluate_sampled_fermi_walker_magnetic_torque_reduction_native(
    *,
    proper_times_ns: Sequence[float] | np.ndarray,
    four_velocity_samples_mm_ns: ArrayLike,
    magnetic_moment_four_samples_native: ArrayLike,
    center_index: int,
    stencil_kind: str,
    require_samples_on_both_sides: bool,
) -> SampledFermiWalkerMagneticTorqueReductionResult:
    times = np.asarray(proper_times_ns, dtype=float)
    if times.ndim != 1 or times.size < 7:
        raise ValueError("proper_times_ns must contain at least seven values")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("proper_times_ns must be finite and strictly increasing")
    center = int(center_index)
    if center < 0:
        center += times.size
    if center < 0 or center >= times.size:
        raise ValueError("center_index must select one supplied sample")
    if require_samples_on_both_sides and (center <= 0 or center >= times.size - 1):
        raise ValueError("center_index must have samples on both sides")

    velocities = _sample_matrix(
        four_velocity_samples_mm_ns,
        sample_count=times.size,
        name="four_velocity_samples_mm_ns",
    )
    moments = _sample_matrix(
        magnetic_moment_four_samples_native,
        sample_count=times.size,
        name="magnetic_moment_four_samples_native",
    )
    if velocities[center, 0] <= 0.0:
        raise ValueError("center four-velocity must be future-directed")

    weights = tuple(
        _finite_difference_weights(
            times,
            center_index=center,
            derivative_order=order,
        )
        for order in (1, 2, 3)
    )
    offsets = times - times[center]
    scale = float(np.max(np.abs(offsets)))
    normalized_offsets = offsets / scale
    powers = np.arange(times.size, dtype=float)[:, np.newaxis]
    condition = float(np.linalg.cond(normalized_offsets[np.newaxis, :] ** powers))

    velocity_deltas = velocities - velocities[center]
    moment_deltas = moments - moments[center]
    raw_u1 = weights[0] @ velocity_deltas
    raw_u2 = weights[1] @ velocity_deltas
    moment_derivatives = tuple(weight @ moment_deltas for weight in weights)

    w0 = velocities[center] / C_MMNS
    velocity_norm = minkowski_dot(w0, w0)
    if not np.isclose(velocity_norm, 1.0, rtol=2.0e-12, atol=2.0e-12):
        raise ValueError("center four-velocity must satisfy u.u = c^2")
    projector = _rest_projector(w0)

    raw_w1 = raw_u1 / C_MMNS
    raw_w2 = raw_u2 / C_MMNS
    first_norm_residual = minkowski_dot(w0, raw_w1)
    w1 = projector @ raw_w1
    second_norm_residual = minkowski_dot(w0, raw_w2) + minkowski_dot(w1, w1)
    w2 = projector @ raw_w2 - w0 * minkowski_dot(w1, w1)

    projector_first = -(np.outer(w1, _lower(w0)) + np.outer(w0, _lower(w1)))
    projector_second = -(
        np.outer(w2, _lower(w0))
        + 2.0 * np.outer(w1, _lower(w1))
        + np.outer(w0, _lower(w2))
    )
    moment_first, moment_second, moment_third = moment_derivatives
    fw_first = projector @ moment_first
    fw_first_derivative = projector_first @ moment_first + projector @ moment_second
    fw_first_second_derivative = (
        projector_second @ moment_first
        + 2.0 * projector_first @ moment_second
        + projector @ moment_third
    )
    fw_second_derivative_derivative = (
        projector_first @ fw_first_derivative + projector @ fw_first_second_derivative
    )
    fw_third = projector @ fw_second_derivative_derivative

    acceleration = C_MMNS * w1
    jerk = C_MMNS * w2
    comparator = evaluate_unruh_planar_accelerated_dipole_torque_comparator_native(
        four_velocity_mm_ns=velocities[center],
        four_acceleration_mm_ns2=acceleration,
        magnetic_moment_four_vector_native=moments[center],
        magnetic_moment_first_fermi_walker_derivative_native=fw_first,
        magnetic_moment_third_fermi_walker_derivative_native=fw_third,
    )

    velocity_norm_residuals = np.asarray(
        [minkowski_dot(value, value) - C_MMNS**2 for value in velocities]
    )
    velocity_moment_residuals = np.asarray(
        [minkowski_dot(u, moment) for u, moment in zip(velocities, moments)]
    )
    arrays = (*weights, acceleration, jerk, fw_first, fw_third)
    for array in arrays:
        array.setflags(write=False)

    return SampledFermiWalkerMagneticTorqueReductionResult(
        center_index=center,
        evaluation_proper_time_ns=float(times[center]),
        sample_time_span_ns=float(times[-1] - times[0]),
        stencil_kind=stencil_kind,
        uses_future_samples=bool(center < times.size - 1),
        scaled_vandermonde_condition_number=condition,
        first_derivative_weights_per_ns=weights[0],
        second_derivative_weights_per_ns2=weights[1],
        third_derivative_weights_per_ns3=weights[2],
        reconstructed_four_acceleration_mm_ns2=acceleration,
        reconstructed_four_jerk_mm_ns3=jerk,
        magnetic_moment_first_fermi_walker_derivative_native=fw_first,
        magnetic_moment_third_fermi_walker_derivative_native=fw_third,
        velocity_normalization_first_derivative_residual_per_ns=float(
            first_norm_residual
        ),
        velocity_normalization_second_derivative_residual_per_ns2=float(
            second_norm_residual
        ),
        maximum_sample_velocity_norm_residual_mm2_ns2=float(
            np.max(np.abs(velocity_norm_residuals))
        ),
        maximum_sample_velocity_moment_residual_native_mm_ns=float(
            np.max(np.abs(velocity_moment_residuals))
        ),
        torque_comparator=comparator,
        leading_non_self_samples_required=True,
        reduction_of_order_reference=True,
    )


def evaluate_sampled_fermi_walker_magnetic_torque_reduction_native(
    *,
    proper_times_ns: Sequence[float] | np.ndarray,
    four_velocity_samples_mm_ns: ArrayLike,
    magnetic_moment_four_samples_native: ArrayLike,
    center_index: int | None = None,
) -> SampledFermiWalkerMagneticTorqueReductionResult:
    """Evaluate a centered, noncausal reduction-of-order reference."""

    times = np.asarray(proper_times_ns, dtype=float)
    center = times.size // 2 if center_index is None else int(center_index)
    return _evaluate_sampled_fermi_walker_magnetic_torque_reduction_native(
        proper_times_ns=times,
        four_velocity_samples_mm_ns=four_velocity_samples_mm_ns,
        magnetic_moment_four_samples_native=magnetic_moment_four_samples_native,
        center_index=center,
        stencil_kind="centered_reference",
        require_samples_on_both_sides=True,
    )


def evaluate_causal_sampled_fermi_walker_magnetic_torque_reduction_native(
    *,
    proper_times_ns: Sequence[float] | np.ndarray,
    four_velocity_samples_mm_ns: ArrayLike,
    magnetic_moment_four_samples_native: ArrayLike,
) -> SampledFermiWalkerMagneticTorqueReductionResult:
    """Evaluate the reference at the newest accepted non-self sample."""

    return _evaluate_sampled_fermi_walker_magnetic_torque_reduction_native(
        proper_times_ns=proper_times_ns,
        four_velocity_samples_mm_ns=four_velocity_samples_mm_ns,
        magnetic_moment_four_samples_native=magnetic_moment_four_samples_native,
        center_index=-1,
        stencil_kind="backward_accepted_history",
        require_samples_on_both_sides=False,
    )


__all__ = [
    "SampledFermiWalkerMagneticTorqueReductionResult",
    "evaluate_causal_sampled_fermi_walker_magnetic_torque_reduction_native",
    "evaluate_sampled_fermi_walker_magnetic_torque_reduction_native",
]
