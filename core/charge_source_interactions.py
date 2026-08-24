"""Exact retarded charge fields and ordinary observer responses.

The exact light-cone provider in :mod:`core.retarded_fields` supplies one
ordinary Maxwell potential, field, and complete spacetime derivatives.  This
module applies an observer charge to that potential using the maintained
canonical convention

``P^mu = p^mu + (q_observer / c) A^mu``.

Both the canonical derivative oracle and the gauge-invariant mechanical
Lorentz force are returned.  The maintained exact integration path advances
the latter, then reconstructs canonical momentum from the accepted endpoint
potential.  A neutral observer still receives the field for RFS response while
both ordinary charge responses are exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .canonical_momentum import (
    canonical_four_force_from_potential_gradient_native,
    canonical_four_impulse_from_potential_gradient_native,
    canonical_potential_momentum_native,
    mechanical_lorentz_four_force_native,
    mechanical_lorentz_four_impulse_native,
)
from .retarded_fields import (
    ObserverEvent,
    RetardedChargeFieldGradientResult,
    TrajectoryHistory,
    evaluate_retarded_charge_field_gradient_native,
)


@dataclass(frozen=True)
class RetardedChargeSourceInteraction:
    """Exact charge field plus canonical and mechanical charge responses."""

    field: RetardedChargeFieldGradientResult
    canonical_potential_momentum: np.ndarray
    canonical_four_force: np.ndarray
    canonical_four_impulse: np.ndarray
    mechanical_four_force: np.ndarray
    mechanical_four_impulse: np.ndarray


def charge_source_interaction_from_field_native(
    field: RetardedChargeFieldGradientResult,
    *,
    four_velocity_mm_ns: Sequence[float] | np.ndarray,
    observer_charge_native: float,
    proper_time_step_ns: float,
) -> RetardedChargeSourceInteraction:
    """Contract one already-evaluated field with the current observer state.

    Exact fields depend on the observer event and source history, but the
    canonical force also depends on the trial four-velocity.  Keeping this
    contraction separate lets fixed-geometry nonlinear iterations reuse the
    expensive light-cone/stencil result while still recomputing the part that
    changes with velocity.
    """

    potential_momentum = canonical_potential_momentum_native(
        field.field.four_potential,
        charge_native=observer_charge_native,
    )
    canonical_force = canonical_four_force_from_potential_gradient_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        partial_a=field.partial_a,
        charge_native=observer_charge_native,
    )
    canonical_impulse = canonical_four_impulse_from_potential_gradient_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        partial_a=field.partial_a,
        charge_native=observer_charge_native,
        proper_time_step_ns=proper_time_step_ns,
    )
    mechanical_force = mechanical_lorentz_four_force_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        field_tensor=field.field.field_tensor,
        charge_native=observer_charge_native,
    )
    mechanical_impulse = mechanical_lorentz_four_impulse_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        field_tensor=field.field.field_tensor,
        charge_native=observer_charge_native,
        proper_time_step_ns=proper_time_step_ns,
    )
    return RetardedChargeSourceInteraction(
        field=field,
        canonical_potential_momentum=potential_momentum,
        canonical_four_force=canonical_force,
        canonical_four_impulse=canonical_impulse,
        mechanical_four_force=mechanical_force,
        mechanical_four_impulse=mechanical_impulse,
    )


def evaluate_retarded_charge_source_interaction_native(
    history: TrajectoryHistory,
    observer_event: ObserverEvent,
    *,
    four_velocity_mm_ns: Sequence[float] | np.ndarray,
    observer_charge_native: float,
    proper_time_step_ns: float,
    excluded_source_indices: Sequence[int] = (),
    require_complete_history: bool = True,
    relative_step: float = 1.0e-4,
    minimum_step_mm: float = 1.0e-15,
    root_tolerance_mm: float = 1.0e-21,
    max_root_iterations: int = 96,
) -> RetardedChargeSourceInteraction:
    """Evaluate exact non-self charge fields and ordinary responses once.

    ``canonical_four_force`` is ``dP^mu/dtau`` and remains a convention oracle.
    ``mechanical_four_force`` is ``dp^mu/dtau=(q/c)F.u`` and is the production
    exact-path translation response.  No RFS moment force, dipole source field,
    or radiation reaction is added here.
    """

    field = evaluate_retarded_charge_field_gradient_native(
        history,
        observer_event,
        excluded_source_indices=excluded_source_indices,
        require_complete_history=require_complete_history,
        relative_step=relative_step,
        minimum_step_mm=minimum_step_mm,
        root_tolerance_mm=root_tolerance_mm,
        max_root_iterations=max_root_iterations,
    )
    return charge_source_interaction_from_field_native(
        field=field,
        four_velocity_mm_ns=four_velocity_mm_ns,
        observer_charge_native=observer_charge_native,
        proper_time_step_ns=proper_time_step_ns,
    )


__all__ = [
    "RetardedChargeSourceInteraction",
    "charge_source_interaction_from_field_native",
    "evaluate_retarded_charge_source_interaction_native",
]
