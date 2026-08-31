"""Sampled reduction-of-order oracle for intrinsic-spin self-reaction.

The covariant point-particle result in :mod:`core.spin_self_force_oracle`
contains four-jerk, four-snap, and spin derivatives.  Those derivatives must
not be promoted to independent production state variables: doing so would
reintroduce the runaway-solution problem of the unreduced ALD equation.

This diagnostic module instead differentiates a short *leading-order,
non-self* trajectory stencil.  It is a convergence oracle, not the eventual
production implementation.  Its centered stencil uses future samples and is
therefore intentionally unsuitable for online causal stepping.  A production
version should obtain the same contractions from analytical external-field
jets or a separately validated causal stencil.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from .spin_self_force_oracle import (
    JakobsenIntrinsicSpinRadiationBalanceResult,
    evaluate_jakobsen_intrinsic_spin_radiation_balance_native,
)


ArrayLike = Union[Sequence[float], Sequence[Sequence[float]], np.ndarray]


def _sample_matrix(value: ArrayLike, *, sample_count: int, name: str) -> np.ndarray:
    samples = np.asarray(value, dtype=float)
    if samples.shape != (sample_count, 4):
        raise ValueError(f"{name} must have shape ({sample_count}, 4)")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} must contain only finite values")
    return samples


def _finite_difference_weights(
    proper_times_ns: np.ndarray,
    *,
    center_index: int,
    derivative_order: int,
) -> np.ndarray:
    """Return arbitrary-node derivative weights with scaled coordinates."""

    offsets = proper_times_ns - proper_times_ns[center_index]
    scale = float(np.max(np.abs(offsets)))
    if scale <= 0.0:
        raise ValueError("proper_times_ns must span a nonzero interval")
    normalized = offsets / scale
    powers = np.arange(proper_times_ns.size, dtype=float)[:, np.newaxis]
    system = normalized[np.newaxis, :] ** powers
    right_hand_side = np.zeros(proper_times_ns.size, dtype=float)
    right_hand_side[derivative_order] = float(math.factorial(derivative_order))
    return np.linalg.solve(system, right_hand_side) / scale**derivative_order


@dataclass(frozen=True)
class SampledIntrinsicSpinReductionResult:
    """Five-or-more-sample leading-order reconstruction and local balance."""

    center_index: int
    evaluation_proper_time_ns: float
    sample_time_span_ns: float
    stencil_kind: str
    uses_future_samples: bool
    scaled_vandermonde_condition_number: float
    first_derivative_weights_per_ns: np.ndarray
    second_derivative_weights_per_ns2: np.ndarray
    reconstructed_four_jerk_mm_ns3: np.ndarray
    reconstructed_four_snap_mm_ns4: np.ndarray
    reconstructed_spin_four_derivative_native: np.ndarray
    reconstructed_spin_four_second_derivative_native: np.ndarray
    velocity_derivative_residual_mm_ns2: np.ndarray
    radiation_balance: JakobsenIntrinsicSpinRadiationBalanceResult


def _evaluate_sampled_intrinsic_spin_reduction_native(
    *,
    proper_times_ns: Sequence[float] | np.ndarray,
    four_velocity_samples_mm_ns: ArrayLike,
    non_self_four_acceleration_samples_mm_ns2: ArrayLike,
    physical_spin_four_samples_native: ArrayLike,
    charge_native: float,
    mass_amu: float,
    g_factor: float,
    center_index: int,
    minimum_sample_count: int,
    stencil_kind: str,
    require_samples_on_both_sides: bool,
) -> SampledIntrinsicSpinReductionResult:
    times = np.asarray(proper_times_ns, dtype=float)
    if times.ndim != 1 or times.size < minimum_sample_count:
        raise ValueError(
            "proper_times_ns must contain at least " f"{minimum_sample_count} values"
        )
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
    accelerations = _sample_matrix(
        non_self_four_acceleration_samples_mm_ns2,
        sample_count=times.size,
        name="non_self_four_acceleration_samples_mm_ns2",
    )
    spins = _sample_matrix(
        physical_spin_four_samples_native,
        sample_count=times.size,
        name="physical_spin_four_samples_native",
    )

    first_weights = _finite_difference_weights(
        times,
        center_index=center,
        derivative_order=1,
    )
    second_weights = _finite_difference_weights(
        times,
        center_index=center,
        derivative_order=2,
    )
    offsets = times - times[center]
    normalized_offsets = offsets / float(np.max(np.abs(offsets)))
    powers = np.arange(times.size, dtype=float)[:, np.newaxis]
    scaled_vandermonde_condition = float(
        np.linalg.cond(normalized_offsets[np.newaxis, :] ** powers)
    )
    # Every derivative annihilates a constant.  Subtract the center value
    # explicitly so a nearly constant temporal component of four-velocity
    # does not lose precision through weighted cancellation of numbers near c.
    velocity_deltas = velocities - velocities[center]
    acceleration_deltas = accelerations - accelerations[center]
    spin_deltas = spins - spins[center]
    reconstructed_velocity_derivative = first_weights @ velocity_deltas
    reconstructed_jerk = first_weights @ acceleration_deltas
    reconstructed_snap = second_weights @ acceleration_deltas
    reconstructed_spin_derivative = first_weights @ spin_deltas
    reconstructed_spin_second_derivative = second_weights @ spin_deltas
    velocity_derivative_residual = (
        reconstructed_velocity_derivative - accelerations[center]
    )

    balance = evaluate_jakobsen_intrinsic_spin_radiation_balance_native(
        charge_native=charge_native,
        mass_amu=mass_amu,
        g_factor=g_factor,
        four_velocity_mm_ns=velocities[center],
        four_acceleration_mm_ns2=accelerations[center],
        four_jerk_mm_ns3=reconstructed_jerk,
        four_snap_mm_ns4=reconstructed_snap,
        spin_four_vector_native=spins[center],
        spin_four_derivative_native=reconstructed_spin_derivative,
        spin_four_second_derivative_native=reconstructed_spin_second_derivative,
    )

    for array in (
        first_weights,
        second_weights,
        reconstructed_jerk,
        reconstructed_snap,
        reconstructed_spin_derivative,
        reconstructed_spin_second_derivative,
        velocity_derivative_residual,
    ):
        array.setflags(write=False)

    return SampledIntrinsicSpinReductionResult(
        center_index=center,
        evaluation_proper_time_ns=float(times[center]),
        sample_time_span_ns=float(times[-1] - times[0]),
        stencil_kind=stencil_kind,
        uses_future_samples=bool(center < times.size - 1),
        scaled_vandermonde_condition_number=scaled_vandermonde_condition,
        first_derivative_weights_per_ns=first_weights,
        second_derivative_weights_per_ns2=second_weights,
        reconstructed_four_jerk_mm_ns3=reconstructed_jerk,
        reconstructed_four_snap_mm_ns4=reconstructed_snap,
        reconstructed_spin_four_derivative_native=reconstructed_spin_derivative,
        reconstructed_spin_four_second_derivative_native=(
            reconstructed_spin_second_derivative
        ),
        velocity_derivative_residual_mm_ns2=velocity_derivative_residual,
        radiation_balance=balance,
    )


def evaluate_sampled_intrinsic_spin_reduction_native(
    *,
    proper_times_ns: Sequence[float] | np.ndarray,
    four_velocity_samples_mm_ns: ArrayLike,
    non_self_four_acceleration_samples_mm_ns2: ArrayLike,
    physical_spin_four_samples_native: ArrayLike,
    charge_native: float,
    mass_amu: float,
    g_factor: float,
    center_index: int | None = None,
) -> SampledIntrinsicSpinReductionResult:
    """Evaluate a centered linear-spin reduction reference.

    The samples must describe the leading ordinary motion: prescribed forces,
    charge Lorentz response, and RFS dipole response may be included, but no
    Medina or magnetic self-reaction contribution may be present.  This is the
    order-reduction rule: every derivative inside the already-small
    self-reaction term is evaluated on the lower-order dynamics.

    At least five strictly increasing proper-time samples are required.  The
    default center is the middle sample.  Arbitrary spacing is supported, but
    the center must have samples on both sides.  The returned
    ``velocity_derivative_residual`` compares the derivative reconstructed
    from the velocity samples with the supplied center acceleration; it is a
    diagnostic of an inconsistent stencil rather than an automatic repair.
    """

    times = np.asarray(proper_times_ns, dtype=float)
    if times.ndim != 1 or times.size < 5:
        raise ValueError("proper_times_ns must contain at least five values")
    center = times.size // 2 if center_index is None else int(center_index)
    return _evaluate_sampled_intrinsic_spin_reduction_native(
        proper_times_ns=times,
        four_velocity_samples_mm_ns=four_velocity_samples_mm_ns,
        non_self_four_acceleration_samples_mm_ns2=(
            non_self_four_acceleration_samples_mm_ns2
        ),
        physical_spin_four_samples_native=physical_spin_four_samples_native,
        charge_native=charge_native,
        mass_amu=mass_amu,
        g_factor=g_factor,
        center_index=center,
        minimum_sample_count=5,
        stencil_kind="centered_reference",
        require_samples_on_both_sides=True,
    )


def evaluate_causal_sampled_intrinsic_spin_reduction_native(
    *,
    proper_times_ns: Sequence[float] | np.ndarray,
    four_velocity_samples_mm_ns: ArrayLike,
    non_self_four_acceleration_samples_mm_ns2: ArrayLike,
    physical_spin_four_samples_native: ArrayLike,
    charge_native: float,
    mass_amu: float,
    g_factor: float,
) -> SampledIntrinsicSpinReductionResult:
    """Evaluate the reduced force at the newest accepted leading-order sample.

    This one-sided reference uses six or more strictly increasing proper-time
    samples and evaluates all derivatives at the last sample.  It therefore
    reads no future state.  Six samples make the second-derivative
    reconstruction fourth-order on a uniformly refined smooth trajectory;
    arbitrary accepted-step spacing is supported.

    The function is still diagnostic.  A production caller must prove that
    every sample is an accepted non-self state, persist the required history
    through checkpoints, and ensure rejected nonlinear or adaptive trials can
    never enter the stencil.  The returned scaled-Vandermonde condition number
    flags clustered or irregular sample times that may amplify roundoff.
    """

    return _evaluate_sampled_intrinsic_spin_reduction_native(
        proper_times_ns=proper_times_ns,
        four_velocity_samples_mm_ns=four_velocity_samples_mm_ns,
        non_self_four_acceleration_samples_mm_ns2=(
            non_self_four_acceleration_samples_mm_ns2
        ),
        physical_spin_four_samples_native=physical_spin_four_samples_native,
        charge_native=charge_native,
        mass_amu=mass_amu,
        g_factor=g_factor,
        center_index=-1,
        minimum_sample_count=6,
        stencil_kind="backward_accepted_history",
        require_samples_on_both_sides=False,
    )


__all__ = [
    "SampledIntrinsicSpinReductionResult",
    "evaluate_causal_sampled_intrinsic_spin_reduction_native",
    "evaluate_sampled_intrinsic_spin_reduction_native",
]
