"""Exact retarded charge fields and their canonical observer response.

The exact light-cone provider in :mod:`core.retarded_fields` supplies one
ordinary Maxwell potential, field, and complete spacetime derivatives.  This
module applies an observer charge to that potential using the maintained
canonical convention

``P^mu = p^mu + (q_observer / c) A^mu``.

The field result remains independent of the observer charge.  A neutral
observer therefore receives the same field for RFS magnetic-moment response,
while its ordinary canonical momentum, force, and impulse are exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .canonical_momentum import (
    canonical_four_force_from_potential_gradient_native,
    canonical_four_impulse_from_potential_gradient_native,
    canonical_potential_momentum_native,
)
from .retarded_fields import (
    ObserverEvent,
    RetardedChargeFieldGradientResult,
    TrajectoryHistory,
    evaluate_retarded_charge_field_gradient_native,
)


@dataclass(frozen=True)
class RetardedChargeSourceInteraction:
    """Exact charge field plus one observer's canonical charge response."""

    field: RetardedChargeFieldGradientResult
    canonical_potential_momentum: np.ndarray
    canonical_four_force: np.ndarray
    canonical_four_impulse: np.ndarray


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
    """Evaluate exact non-self charge fields and canonical response once.

    ``canonical_four_force`` is ``dP^mu/dtau``.  Subtracting the convective
    derivative of ``(q/c) A^mu`` recovers the mechanical Lorentz four-force.
    No RFS force, dipole source field, or radiation reaction is added here.
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
    return RetardedChargeSourceInteraction(
        field=field,
        canonical_potential_momentum=potential_momentum,
        canonical_four_force=canonical_force,
        canonical_four_impulse=canonical_impulse,
    )


__all__ = [
    "RetardedChargeSourceInteraction",
    "evaluate_retarded_charge_source_interaction_native",
]
