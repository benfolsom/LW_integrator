"""Reduction references for the pure-magnetic self-torque.

The accelerated point comparator needs first and third Fermi--Walker
derivatives of the magnetic moment.  This module reconstructs them from a
short stencil of leading-order, non-self trajectory samples.  A second route
derives them analytically from the sparse potential derivatives already used
by the RFS reduction bridge.  Both remain passive diagnostic comparators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import C_MMNS
from .magnetic_dipole import minkowski_dot
from .spin_self_force_reduction_oracle import (
    ArrayLike,
    PotentialDirectionalIntrinsicSpinReductionResult,
    _finite_difference_weights,
    _sample_matrix,
)
from .spin_self_torque_oracle import (
    UnruhPlanarAcceleratedDipoleTorqueComparatorResult,
    evaluate_unruh_planar_accelerated_dipole_torque_comparator_native,
)
from .potential_jet_rfs import (
    MatrixLike,
    PotentialDirectionalRFSReductionJet,
    Tensor3Like,
    VectorLike,
    potential_directional_rfs_reduction_jet_native,
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


@dataclass(frozen=True)
class LocalFermiWalkerMagneticMomentDerivativeJet:
    """First and third rest-space derivatives from one local worldline jet."""

    first_fermi_walker_derivative_native: np.ndarray
    third_fermi_walker_derivative_native: np.ndarray
    projected_four_acceleration_mm_ns2: np.ndarray
    normalized_four_jerk_mm_ns3: np.ndarray
    velocity_normalization_first_derivative_residual_per_ns: float
    velocity_normalization_second_derivative_residual_per_ns2: float


@dataclass(frozen=True)
class PotentialDirectionalMagneticTorqueReductionResult:
    """Potential-only leading dynamics and reduced magnetic self-torque."""

    leading_dynamics: PotentialDirectionalRFSReductionJet
    fermi_walker_derivatives: LocalFermiWalkerMagneticMomentDerivativeJet
    torque_comparator: UnruhPlanarAcceleratedDipoleTorqueComparatorResult
    analytical_potential_derivatives_only: bool
    reduction_of_order_performed: bool


def fermi_walker_magnetic_moment_derivatives_from_local_jet_native(
    *,
    four_velocity_mm_ns: VectorLike,
    four_acceleration_mm_ns2: VectorLike,
    four_jerk_mm_ns3: VectorLike,
    magnetic_moment_first_derivative_native: VectorLike,
    magnetic_moment_second_derivative_native: VectorLike,
    magnetic_moment_third_derivative_native: VectorLike,
) -> LocalFermiWalkerMagneticMomentDerivativeJet:
    """Apply three successive rest-space derivatives to a local moment jet."""

    vectors = tuple(
        np.asarray(value, dtype=float)
        for value in (
            four_velocity_mm_ns,
            four_acceleration_mm_ns2,
            four_jerk_mm_ns3,
            magnetic_moment_first_derivative_native,
            magnetic_moment_second_derivative_native,
            magnetic_moment_third_derivative_native,
        )
    )
    if any(value.shape != (4,) or not np.all(np.isfinite(value)) for value in vectors):
        raise ValueError(
            "every local worldline and moment jet value must be a finite four-vector"
        )
    velocity, acceleration, jerk, moment_first, moment_second, moment_third = vectors
    w0 = velocity / C_MMNS
    if not np.isclose(minkowski_dot(w0, w0), 1.0, rtol=2.0e-12, atol=2.0e-12):
        raise ValueError("four_velocity_mm_ns must satisfy u.u = c^2")
    projector = _rest_projector(w0)
    raw_w1 = acceleration / C_MMNS
    raw_w2 = jerk / C_MMNS
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
    fw_first = projector @ moment_first
    fw_first_derivative = projector_first @ moment_first + projector @ moment_second
    fw_first_second_derivative = (
        projector_second @ moment_first
        + 2.0 * projector_first @ moment_second
        + projector @ moment_third
    )
    fw_third = projector @ (
        projector_first @ fw_first_derivative + projector @ fw_first_second_derivative
    )
    projected_acceleration = C_MMNS * w1
    normalized_jerk = C_MMNS * w2
    arrays = (fw_first, fw_third, projected_acceleration, normalized_jerk)
    for array in arrays:
        array.setflags(write=False)
    return LocalFermiWalkerMagneticMomentDerivativeJet(
        first_fermi_walker_derivative_native=fw_first,
        third_fermi_walker_derivative_native=fw_third,
        projected_four_acceleration_mm_ns2=projected_acceleration,
        normalized_four_jerk_mm_ns3=normalized_jerk,
        velocity_normalization_first_derivative_residual_per_ns=float(
            first_norm_residual
        ),
        velocity_normalization_second_derivative_residual_per_ns2=float(
            second_norm_residual
        ),
    )


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

    moment_first, moment_second, moment_third = moment_derivatives
    local = fermi_walker_magnetic_moment_derivatives_from_local_jet_native(
        four_velocity_mm_ns=velocities[center],
        four_acceleration_mm_ns2=raw_u1,
        four_jerk_mm_ns3=raw_u2,
        magnetic_moment_first_derivative_native=moment_first,
        magnetic_moment_second_derivative_native=moment_second,
        magnetic_moment_third_derivative_native=moment_third,
    )
    fw_first = local.first_fermi_walker_derivative_native
    fw_third = local.third_fermi_walker_derivative_native
    acceleration = local.projected_four_acceleration_mm_ns2
    jerk = local.normalized_four_jerk_mm_ns3
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
            local.velocity_normalization_first_derivative_residual_per_ns
        ),
        velocity_normalization_second_derivative_residual_per_ns2=float(
            local.velocity_normalization_second_derivative_residual_per_ns2
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


def _evaluate_magnetic_torque_from_leading_rfs_jet_native(
    *,
    four_velocity_mm_ns: VectorLike,
    normalized_spin_four_vector: VectorLike,
    magnetic_moment_native: float,
    leading: PotentialDirectionalRFSReductionJet,
) -> PotentialDirectionalMagneticTorqueReductionResult:
    velocity = np.asarray(four_velocity_mm_ns, dtype=float)
    spin = np.asarray(normalized_spin_four_vector, dtype=float)
    moment = float(magnetic_moment_native)
    if velocity.shape != (4,) or not np.all(np.isfinite(velocity)):
        raise ValueError("four_velocity_mm_ns must be a finite four-vector")
    if spin.shape != (4,) or not np.all(np.isfinite(spin)):
        raise ValueError("normalized_spin_four_vector must be a finite four-vector")
    if not np.isfinite(moment):
        raise ValueError("magnetic_moment_native must be finite")
    moment_four = moment * spin
    derivatives = fermi_walker_magnetic_moment_derivatives_from_local_jet_native(
        four_velocity_mm_ns=velocity,
        four_acceleration_mm_ns2=leading.four_acceleration,
        four_jerk_mm_ns3=leading.four_jerk,
        magnetic_moment_first_derivative_native=(
            moment * leading.normalized_spin_first_derivative
        ),
        magnetic_moment_second_derivative_native=(
            moment * leading.normalized_spin_second_derivative
        ),
        magnetic_moment_third_derivative_native=(
            moment * leading.normalized_spin_third_derivative
        ),
    )
    comparator = evaluate_unruh_planar_accelerated_dipole_torque_comparator_native(
        four_velocity_mm_ns=velocity,
        four_acceleration_mm_ns2=(derivatives.projected_four_acceleration_mm_ns2),
        magnetic_moment_four_vector_native=moment_four,
        magnetic_moment_first_fermi_walker_derivative_native=(
            derivatives.first_fermi_walker_derivative_native
        ),
        magnetic_moment_third_fermi_walker_derivative_native=(
            derivatives.third_fermi_walker_derivative_native
        ),
    )
    return PotentialDirectionalMagneticTorqueReductionResult(
        leading_dynamics=leading,
        fermi_walker_derivatives=derivatives,
        torque_comparator=comparator,
        analytical_potential_derivatives_only=True,
        reduction_of_order_performed=True,
    )


def evaluate_potential_directional_magnetic_torque_reduction_native(
    *,
    four_velocity_mm_ns: VectorLike,
    normalized_spin_four_vector: VectorLike,
    partial_a: MatrixLike,
    partial2_a: Tensor3Like,
    partial3_a_along_velocity: Tensor3Like,
    partial3_a_along_acceleration: Tensor3Like,
    partial4_a_along_velocity_twice: Tensor3Like,
    charge_native: float,
    mass_amu: float,
    magnetic_moment_native: float,
    invariant_spin_native: float,
) -> PotentialDirectionalMagneticTorqueReductionResult:
    """Reduce the planar torque comparator from analytical potential jets.

    The leading RFS motion contains no self-reaction.  Its acceleration,
    jerk, and first three normalized-spin derivatives supply the repeated
    Fermi--Walker derivatives without differencing trajectory samples.
    """

    velocity = np.asarray(four_velocity_mm_ns, dtype=float)
    spin = np.asarray(normalized_spin_four_vector, dtype=float)
    moment = float(magnetic_moment_native)
    if velocity.shape != (4,) or not np.all(np.isfinite(velocity)):
        raise ValueError("four_velocity_mm_ns must be a finite four-vector")
    if spin.shape != (4,) or not np.all(np.isfinite(spin)):
        raise ValueError("normalized_spin_four_vector must be a finite four-vector")
    if not np.isfinite(moment):
        raise ValueError("magnetic_moment_native must be finite")

    leading = potential_directional_rfs_reduction_jet_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        partial_a=partial_a,
        partial2_a=partial2_a,
        partial3_a_along_velocity=partial3_a_along_velocity,
        partial3_a_along_acceleration=partial3_a_along_acceleration,
        partial4_a_along_velocity_twice=partial4_a_along_velocity_twice,
        charge_native=charge_native,
        mass_amu=mass_amu,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin_native,
    )
    return _evaluate_magnetic_torque_from_leading_rfs_jet_native(
        four_velocity_mm_ns=velocity,
        normalized_spin_four_vector=spin,
        magnetic_moment_native=moment,
        leading=leading,
    )


def evaluate_magnetic_torque_from_intrinsic_spin_reduction_native(
    *,
    four_velocity_mm_ns: VectorLike,
    normalized_spin_four_vector: VectorLike,
    intrinsic_spin_reduction: PotentialDirectionalIntrinsicSpinReductionResult,
) -> PotentialDirectionalMagneticTorqueReductionResult:
    """Attach the passive magnetic torque to an existing retarded reduction.

    The linear-in-spin self-force and pure-magnetic torque use the same
    leading non-self RFS jet.  This adapter reuses that jet, so a diagnostic
    caller does not repeat either retarded-provider pass.
    """

    return _evaluate_magnetic_torque_from_leading_rfs_jet_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        normalized_spin_four_vector=normalized_spin_four_vector,
        magnetic_moment_native=(
            intrinsic_spin_reduction.intrinsic_magnetic_moment_native
        ),
        leading=intrinsic_spin_reduction.leading_dynamics,
    )


__all__ = [
    "LocalFermiWalkerMagneticMomentDerivativeJet",
    "PotentialDirectionalMagneticTorqueReductionResult",
    "SampledFermiWalkerMagneticTorqueReductionResult",
    "evaluate_causal_sampled_fermi_walker_magnetic_torque_reduction_native",
    "evaluate_magnetic_torque_from_intrinsic_spin_reduction_native",
    "evaluate_potential_directional_magnetic_torque_reduction_native",
    "evaluate_sampled_fermi_walker_magnetic_torque_reduction_native",
    "fermi_walker_magnetic_moment_derivatives_from_local_jet_native",
]
