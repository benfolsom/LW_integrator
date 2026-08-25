"""Ordinary response to Maxwell fields sourced by intrinsic dipoles.

This module joins two independently tested pieces without adding another
pair-force law:

* :mod:`core.retarded_dipole_fields` supplies the ordinary Maxwell
  ``A^mu``, ``partial A``, ``F``, and ``partial F`` of all non-self dipoles;
* :mod:`core.canonical_momentum` supplies both the canonical derivative oracle
  and the equivalent gauge-invariant mechanical Lorentz response.

The returned ``field_tensor`` and ``partial_f`` are consumed separately by the
RFS moment response.  A caller must pass them into that response exactly once
with ``charge_native=0``, because the ordinary charge response is already
represented by either the canonical or mechanical result here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

from .canonical_momentum import (
    canonical_four_force_from_potential_gradient_native,
    canonical_four_impulse_from_potential_gradient_native,
    canonical_potential_momentum_native,
    mechanical_lorentz_four_force_native,
    mechanical_lorentz_four_impulse_native,
)
from .retarded_dipole_fields import (
    RetardedDipoleFieldGradientResult,
    evaluate_retarded_dipole_field_gradient_native,
)
from .retarded_fields import ObserverEvent, TrajectoryHistory


@dataclass(frozen=True)
class RetardedDipoleSourceInteraction:
    """Dipole field plus canonical and mechanical observer-charge responses."""

    field: RetardedDipoleFieldGradientResult
    canonical_potential_momentum: np.ndarray
    canonical_four_force: np.ndarray
    canonical_four_impulse: np.ndarray
    mechanical_four_force: np.ndarray
    mechanical_four_impulse: np.ndarray


def dipole_source_interaction_from_field_native(
    field: RetardedDipoleFieldGradientResult,
    *,
    four_velocity_mm_ns: Sequence[float] | np.ndarray,
    observer_charge_native: float,
    proper_time_step_ns: float,
) -> RetardedDipoleSourceInteraction:
    """Contract one cached dipole field with the current trial velocity."""

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
    mechanical_force = mechanical_lorentz_four_force_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        field_tensor=field.field_tensor,
        charge_native=observer_charge_native,
    )
    mechanical_impulse = mechanical_lorentz_four_impulse_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        field_tensor=field.field_tensor,
        charge_native=observer_charge_native,
        proper_time_step_ns=proper_time_step_ns,
    )
    return RetardedDipoleSourceInteraction(
        field=field,
        canonical_potential_momentum=potential_momentum,
        canonical_four_force=canonical_force,
        canonical_four_impulse=canonical_impulse,
        mechanical_four_force=mechanical_force,
        mechanical_four_impulse=mechanical_impulse,
    )


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
    backend: str = "python",
) -> RetardedDipoleSourceInteraction:
    """Evaluate the non-self dipole field and ordinary charge responses.

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
        backend=backend,
    )
    return dipole_source_interaction_from_field_native(
        field=field,
        four_velocity_mm_ns=four_velocity_mm_ns,
        observer_charge_native=observer_charge_native,
        proper_time_step_ns=proper_time_step_ns,
    )


__all__ = [
    "RetardedDipoleSourceInteraction",
    "dipole_source_interaction_from_field_native",
    "evaluate_retarded_dipole_source_interaction_native",
]
