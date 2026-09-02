"""Ordinary response to Maxwell fields sourced by intrinsic dipoles.

This module joins two independently tested pieces without adding another
pair-force law:

* :mod:`core.retarded_dipole_fields` supplies either the ordinary Maxwell
  ``A^mu``, ``partial A``, ``F``, and ``partial F`` oracle or the compact
  ``A^mu`` plus packed ``F``/``partial F`` response of all non-self dipoles;
* :mod:`core.canonical_momentum` supplies both the canonical derivative oracle
  and the equivalent gauge-invariant mechanical Lorentz response.

The field response is consumed separately by the RFS moment response, either
as tensors or by direct packed contraction.  A caller must add it exactly once
without repeating the ordinary charge response represented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable, Sequence

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
    RetardedDipoleResponseGradientResult,
    evaluate_retarded_dipole_field_gradient_native,
)
from .retarded_fields import ObserverEvent, TrajectoryHistory

if TYPE_CHECKING:
    from .causal_c5_dipole_provider import CausalC5DipoleProviderResult


@dataclass(frozen=True)
class RetardedDipoleSourceInteraction:
    """Dipole field plus canonical and mechanical observer-charge responses."""

    field: RetardedDipoleFieldGradientResult | CausalC5DipoleProviderResult | None
    canonical_potential_momentum: np.ndarray
    canonical_four_force: np.ndarray | None
    canonical_four_impulse: np.ndarray | None
    mechanical_four_force: np.ndarray
    mechanical_four_impulse: np.ndarray
    response: RetardedDipoleResponseGradientResult | None = None

    @property
    def four_potential(self) -> np.ndarray:
        """Return the ordinary source potential for canonical bookkeeping."""

        if self.response is not None:
            return self.response.four_potential
        if self.field is None:
            raise RuntimeError("dipole interaction has no field or response payload")
        return self.field.four_potential


def dipole_source_interaction_from_field_native(
    field: RetardedDipoleFieldGradientResult | CausalC5DipoleProviderResult,
    *,
    four_velocity_mm_ns: Sequence[float] | np.ndarray,
    observer_charge_native: float,
    proper_time_step_ns: float,
) -> RetardedDipoleSourceInteraction:
    """Contract one materialized dipole field with the trial velocity.

    Both the legacy retarded-field result and the causally frozen analytical
    $C^5$ result expose the same physical payload: $A$, $\\partial A$, $F$, and
    $\\partial F$.  Keeping the contraction here ensures that changing the
    source-history representation does not introduce a second force law.
    """

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


def dipole_source_interaction_from_response_native(
    response: RetardedDipoleResponseGradientResult,
    *,
    four_velocity_mm_ns: Sequence[float] | np.ndarray,
    observer_charge_native: float,
    proper_time_step_ns: float,
    contraction_backend: str = "python",
) -> RetardedDipoleSourceInteraction:
    """Contract the compact response without materializing ``F`` or ``dF``.

    The compact jet deliberately omits ``partial A``.  It is therefore valid
    for the maintained exact endpoint scheme, which advances mechanical
    momentum with ``q F`` and reconstructs canonical momentum from ``A`` at
    the accepted endpoint.  The legacy COLD_START canonical-force path keeps
    using :func:`dipole_source_interaction_from_field_native`.
    """

    from .antisymmetric_response_rfs import (
        antisymmetric_response_charge_force_native,
    )

    potential_momentum = canonical_potential_momentum_native(
        response.four_potential,
        charge_native=observer_charge_native,
    )
    if contraction_backend == "python":
        mechanical_force = antisymmetric_response_charge_force_native(
            four_velocity_mm_ns=four_velocity_mm_ns,
            antisymmetric_response=response.antisymmetric_response,
            charge_native=observer_charge_native,
        )
    elif contraction_backend == "numba_strict_serial":
        from .contracted_antisymmetric_response_numba import (
            antisymmetric_response_charge_force_strict_serial,
        )

        mechanical_force = antisymmetric_response_charge_force_strict_serial(
            np.asarray(four_velocity_mm_ns, dtype=float),
            response.antisymmetric_response,
            float(observer_charge_native),
        )
    else:
        raise ValueError(
            "contraction_backend must be 'python' or 'numba_strict_serial'"
        )
    return RetardedDipoleSourceInteraction(
        field=None,
        canonical_potential_momentum=potential_momentum,
        canonical_four_force=None,
        canonical_four_impulse=None,
        mechanical_four_force=mechanical_force,
        mechanical_four_impulse=(mechanical_force * float(proper_time_step_ns)),
        response=response,
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
    spin_interpolation_model: str = "centered_c1",
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
        spin_interpolation_model=spin_interpolation_model,
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
    "dipole_source_interaction_from_response_native",
    "evaluate_retarded_dipole_source_interaction_native",
]
