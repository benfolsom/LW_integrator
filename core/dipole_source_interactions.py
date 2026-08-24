"""Canonical response to ordinary fields sourced by intrinsic dipoles.

This module joins two independently tested pieces without adding another
pair-force law:

* :mod:`core.retarded_dipole_fields` supplies the ordinary Maxwell
  ``A^mu``, ``partial A``, ``F``, and ``partial F`` of all non-self dipoles;
* :mod:`core.canonical_momentum` applies the observer's ordinary charge to
  that potential using the same canonical equation as the charge field.

The returned ``field_tensor`` and ``partial_f`` are still consumed separately
by the RFS response.  A caller must pass them into RFS exactly once and keep
``charge_native=0`` in that added RFS translational call, because the charge
response is already represented by the canonical impulse here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

from .canonical_momentum import (
    canonical_four_force_from_potential_gradient_native,
    canonical_four_impulse_from_potential_gradient_native,
    canonical_potential_momentum_native,
)
from .retarded_dipole_fields import (
    RetardedDipoleFieldGradientResult,
    evaluate_retarded_dipole_field_gradient_native,
)
from .retarded_fields import ObserverEvent, TrajectoryHistory


@dataclass(frozen=True)
class RetardedDipoleSourceInteraction:
    """Ordinary dipole field plus one observer's canonical charge response."""

    field: RetardedDipoleFieldGradientResult
    canonical_potential_momentum: np.ndarray
    canonical_four_force: np.ndarray
    canonical_four_impulse: np.ndarray


def evaluate_retarded_dipole_source_interaction_native(
    history: TrajectoryHistory,
    observer_event: ObserverEvent,
    *,
    four_velocity_mm_ns: Sequence[float] | np.ndarray,
    observer_charge_native: float,
    proper_time_step_ns: float,
    source_identities: Sequence[Hashable] | None = None,
    observer_source_identity: Hashable | None = None,
    excluded_source_identities: Sequence[Hashable] = (),
    require_complete_history: bool = True,
    relative_step: float = 1.0e-3,
    minimum_step_mm: float = 1.0e-15,
    stencil_step_mm: float | None = None,
    minimum_separation_mm: float = 1.0e-15,
    root_tolerance_mm: float = 1.0e-21,
    max_root_iterations: int = 96,
) -> RetardedDipoleSourceInteraction:
    """Evaluate the non-self dipole field and canonical charge impulse.

    Source evaluation is independent of the observer charge.  Consequently a
    neutral observer receives the same field and RFS input while its ordinary
    canonical response is exactly zero.
    """

    field = evaluate_retarded_dipole_field_gradient_native(
        history,
        observer_event,
        source_identities=source_identities,
        observer_source_identity=observer_source_identity,
        excluded_source_identities=excluded_source_identities,
        require_complete_history=require_complete_history,
        relative_step=relative_step,
        minimum_step_mm=minimum_step_mm,
        stencil_step_mm=stencil_step_mm,
        minimum_separation_mm=minimum_separation_mm,
        root_tolerance_mm=root_tolerance_mm,
        max_root_iterations=max_root_iterations,
    )
    potential_momentum = canonical_potential_momentum_native(
        field.four_potential,
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
    return RetardedDipoleSourceInteraction(
        field=field,
        canonical_potential_momentum=potential_momentum,
        canonical_four_force=canonical_force,
        canonical_four_impulse=canonical_impulse,
    )


__all__ = [
    "RetardedDipoleSourceInteraction",
    "evaluate_retarded_dipole_source_interaction_native",
]
