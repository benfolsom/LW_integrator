"""Strict serial kernels shared by exact retarded charge and dipole fields.

The module intentionally contains no automatic or platform-specific dispatch.
Every kernel uses binary64 arithmetic with ``fastmath=False`` and preserves the
input event order. Source addition and finite-difference reductions remain in
the provider's Python reference order.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .retarded_dipole_numba_roots import (
    NUMBA_AVAILABLE,
    NUMBA_COMPILATION_ERRORS,
    _STATUS_MINIMUM_SEPARATION,
    _STATUS_MISSING_HISTORY,
    _STATUS_SINGULAR_KAPPA,
    _STATUS_SPIN_INTERPOLATION_ZERO,
    _STATUS_SUPERLUMINAL_SOURCE,
    _STATUS_TERMINATED_SOURCE,
    _STATUS_VALID,
    _solve_retarded_sample,
    evaluate_source_events_full_strict_serial,
    evaluate_source_events_full_strict_from_segments_serial,
    evaluate_source_roots_exact_serial,
    jit,
)

_STATUS_CHARGE_ZERO_SEPARATION = 20
_STATUS_CHARGE_SUPERLUMINAL_SOURCE = 21
_STATUS_CHARGE_SINGULAR_KAPPA = 22


@jit(nopython=True, fastmath=False, nogil=True, cache=True)
def evaluate_charge_source_events_full_strict_serial(
    time_ns: np.ndarray,
    position_mm: np.ndarray,
    segment_duration_ns: np.ndarray,
    position_coefficients_mm: np.ndarray,
    charge_native: float,
    ended_by_loss: bool,
    observer_time_ns: np.ndarray,
    observer_position_mm: np.ndarray,
    root_tolerance_mm: float,
    max_root_iterations: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate one charge source at every event with strict serial arithmetic."""

    event_count = observer_time_ns.size
    status = np.empty(event_count, dtype=np.int64)
    electric = np.zeros((event_count, 3), dtype=np.float64)
    magnetic = np.zeros((event_count, 3), dtype=np.float64)
    four_potential = np.zeros((event_count, 4), dtype=np.float64)
    retarded_time_ns = np.full(event_count, np.nan, dtype=np.float64)
    residual_mm = np.full(event_count, np.nan, dtype=np.float64)
    separation_mm = np.full(event_count, np.nan, dtype=np.float64)
    valid = np.zeros(event_count, dtype=np.bool_)

    for event_index in range(event_count):
        (
            event_status,
            source_time_ns,
            source_x_mm,
            source_y_mm,
            source_z_mm,
            source_beta_x,
            source_beta_y,
            source_beta_z,
            source_beta_prime_x,
            source_beta_prime_y,
            source_beta_prime_z,
            source_residual_mm,
            source_separation_mm,
        ) = _solve_retarded_sample(
            time_ns,
            position_mm,
            segment_duration_ns,
            position_coefficients_mm,
            observer_time_ns[event_index],
            observer_position_mm[event_index, 0],
            observer_position_mm[event_index, 1],
            observer_position_mm[event_index, 2],
            root_tolerance_mm,
            max_root_iterations,
            ended_by_loss,
        )
        status[event_index] = event_status
        if event_status == _STATUS_TERMINATED_SOURCE:
            continue
        if event_status != _STATUS_VALID:
            status[event_index] = _STATUS_MISSING_HISTORY
            continue

        retarded_time_ns[event_index] = source_time_ns
        residual_mm[event_index] = source_residual_mm
        separation_mm[event_index] = source_separation_mm
        if source_separation_mm <= 0.0:
            status[event_index] = _STATUS_CHARGE_ZERO_SEPARATION
            continue

        beta_squared = (
            source_beta_x * source_beta_x
            + source_beta_y * source_beta_y
            + source_beta_z * source_beta_z
        )
        if beta_squared >= 1.0:
            status[event_index] = _STATUS_CHARGE_SUPERLUMINAL_SOURCE
            continue

        separation_x = observer_position_mm[event_index, 0] - source_x_mm
        separation_y = observer_position_mm[event_index, 1] - source_y_mm
        separation_z = observer_position_mm[event_index, 2] - source_z_mm
        direction_x = separation_x / source_separation_mm
        direction_y = separation_y / source_separation_mm
        direction_z = separation_z / source_separation_mm
        kappa = 1.0 - (
            direction_x * source_beta_x
            + direction_y * source_beta_y
            + direction_z * source_beta_z
        )
        if kappa <= 1.0e-14:
            status[event_index] = _STATUS_CHARGE_SINGULAR_KAPPA
            continue

        difference_x = direction_x - source_beta_x
        difference_y = direction_y - source_beta_y
        difference_z = direction_z - source_beta_z
        inner_cross_x = (
            difference_y * source_beta_prime_z - difference_z * source_beta_prime_y
        )
        inner_cross_y = (
            difference_z * source_beta_prime_x - difference_x * source_beta_prime_z
        )
        inner_cross_z = (
            difference_x * source_beta_prime_y - difference_y * source_beta_prime_x
        )
        radiation_cross_x = direction_y * inner_cross_z - direction_z * inner_cross_y
        radiation_cross_y = direction_z * inner_cross_x - direction_x * inner_cross_z
        radiation_cross_z = direction_x * inner_cross_y - direction_y * inner_cross_x
        kappa_cubed = kappa**3
        velocity_scale = (1.0 - beta_squared) / (kappa_cubed * source_separation_mm**2)
        radiation_scale = 1.0 / (kappa_cubed * source_separation_mm)
        electric_x = charge_native * (
            velocity_scale * difference_x + radiation_scale * radiation_cross_x
        )
        electric_y = charge_native * (
            velocity_scale * difference_y + radiation_scale * radiation_cross_y
        )
        electric_z = charge_native * (
            velocity_scale * difference_z + radiation_scale * radiation_cross_z
        )
        electric[event_index, 0] = electric_x
        electric[event_index, 1] = electric_y
        electric[event_index, 2] = electric_z
        magnetic[event_index, 0] = direction_y * electric_z - direction_z * electric_y
        magnetic[event_index, 1] = direction_z * electric_x - direction_x * electric_z
        magnetic[event_index, 2] = direction_x * electric_y - direction_y * electric_x

        scalar_potential = charge_native / (kappa * source_separation_mm)
        four_potential[event_index, 0] = scalar_potential
        four_potential[event_index, 1] = scalar_potential * source_beta_x
        four_potential[event_index, 2] = scalar_potential * source_beta_y
        four_potential[event_index, 3] = scalar_potential * source_beta_z
        valid[event_index] = True

    return (
        status,
        electric,
        magnetic,
        four_potential,
        retarded_time_ns,
        residual_mm,
        separation_mm,
        valid,
    )


__all__ = [
    "NUMBA_AVAILABLE",
    "NUMBA_COMPILATION_ERRORS",
    "_STATUS_CHARGE_SINGULAR_KAPPA",
    "_STATUS_CHARGE_SUPERLUMINAL_SOURCE",
    "_STATUS_CHARGE_ZERO_SEPARATION",
    "_STATUS_MINIMUM_SEPARATION",
    "_STATUS_MISSING_HISTORY",
    "_STATUS_SINGULAR_KAPPA",
    "_STATUS_SPIN_INTERPOLATION_ZERO",
    "_STATUS_SUPERLUMINAL_SOURCE",
    "_STATUS_TERMINATED_SOURCE",
    "_STATUS_VALID",
    "evaluate_charge_source_events_full_strict_serial",
    "evaluate_source_events_full_strict_serial",
    "evaluate_source_events_full_strict_from_segments_serial",
    "evaluate_source_roots_exact_serial",
]
