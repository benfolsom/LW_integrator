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
    canonical_potential_momentum_native,
    mechanical_lorentz_four_force_native,
)
from .retarded_fields import (
    ObserverEvent,
    RetardedChargeFieldGradientResult,
    RetardedChargeResponseGradientResult,
    TrajectoryHistory,
    evaluate_retarded_charge_field_gradient_native,
    evaluate_retarded_charge_response_gradient_native,
)


@dataclass(frozen=True)
class RetardedChargeSourceInteraction:
    """Exact charge field plus canonical and mechanical charge responses."""

    field: RetardedChargeFieldGradientResult | None
    canonical_potential_momentum: np.ndarray
    canonical_four_force: np.ndarray
    canonical_four_impulse: np.ndarray
    mechanical_four_force: np.ndarray
    mechanical_four_impulse: np.ndarray
    response: RetardedChargeResponseGradientResult | None = None

    @property
    def four_potential(self) -> np.ndarray:
        """Return the ordinary source potential for canonical bookkeeping."""

        if self.response is not None:
            return self.response.four_potential
        if self.field is None:
            raise RuntimeError("charge interaction has no field or response payload")
        return self.field.field.four_potential

    @property
    def analytical_fallback_used(self) -> bool:
        return bool(self.response is not None and self.response.fallback_used)


def cached_charge_source_interaction(
    field: RetardedChargeFieldGradientResult | RetardedChargeResponseGradientResult,
    *,
    four_velocity_mm_ns: np.ndarray,
    observer_charge_native: float,
    proper_time_step_ns: float,
    cache: dict | None,
    key: tuple[str, int],
) -> RetardedChargeSourceInteraction:
    """Reuse a contraction only for the identical field, velocity, and charge.

    The two impulses retain their original multiplication order when h changes.
    Field identity prevents reuse after continuation or provisional replacement.
    Byte snapshots distinguish signed zeros and changed observer velocities.
    """
    signature = (four_velocity_mm_ns.tobytes(), float(observer_charge_native).hex())
    saved = None if cache is None else cache.get(key)
    if saved is not None and saved[0] is field and saved[1] == signature:
        step = float(proper_time_step_ns)
        if not np.isfinite(step):
            raise ValueError("proper_time_step_ns must be finite")
        previous = saved[2]
        return RetardedChargeSourceInteraction(
            field=previous.field,
            response=previous.response,
            canonical_potential_momentum=previous.canonical_potential_momentum,
            canonical_four_force=previous.canonical_four_force,
            canonical_four_impulse=step * previous.canonical_four_force,
            mechanical_four_force=previous.mechanical_four_force,
            mechanical_four_impulse=step * previous.mechanical_four_force,
        )
    if isinstance(field, RetardedChargeResponseGradientResult):
        interaction = charge_source_interaction_from_response_native(
            field,
            four_velocity_mm_ns=four_velocity_mm_ns,
            observer_charge_native=observer_charge_native,
            proper_time_step_ns=proper_time_step_ns,
            contraction_backend="numba_strict_serial",
        )
    else:
        interaction = charge_source_interaction_from_field_native(
            field,
            four_velocity_mm_ns=four_velocity_mm_ns,
            observer_charge_native=observer_charge_native,
            proper_time_step_ns=proper_time_step_ns,
        )
    if cache is not None:
        cache[key] = (field, signature, interaction)
    return interaction


def cached_ordinary_force_derivative(
    interaction,
    *,
    four_velocity_mm_ns: np.ndarray,
    four_acceleration_mm_ns2: np.ndarray,
    charge_native: float,
    cache: dict | None,
    key: tuple[str, int],
) -> np.ndarray:
    """Retain a start derivative while its complete contraction inputs agree."""
    from .antisymmetric_response_rfs import (
        antisymmetric_response_charge_force_derivative_native,
    )
    from .canonical_momentum import mechanical_lorentz_four_force_derivative_native

    payload = (
        interaction.response if interaction.response is not None else interaction.field
    )
    signature = (
        four_velocity_mm_ns.tobytes(),
        four_acceleration_mm_ns2.tobytes(),
        float(charge_native).hex(),
    )
    saved = None if cache is None else cache.get(key)
    if saved is not None and saved[0] is payload and saved[1] == signature:
        return saved[2]
    if interaction.response is not None:
        derivative = antisymmetric_response_charge_force_derivative_native(
            four_velocity_mm_ns=four_velocity_mm_ns,
            four_acceleration_mm_ns2=four_acceleration_mm_ns2,
            antisymmetric_response=payload.antisymmetric_response,
            partial_antisymmetric_response=payload.partial_antisymmetric_response,
            charge_native=charge_native,
        )
    else:
        field_tensor = (
            payload.field.field_tensor
            if hasattr(payload, "field")
            else payload.field_tensor
        )
        derivative = mechanical_lorentz_four_force_derivative_native(
            four_velocity_mm_ns=four_velocity_mm_ns,
            four_acceleration_mm_ns2=four_acceleration_mm_ns2,
            field_tensor=field_tensor,
            partial_f=payload.partial_f,
            charge_native=charge_native,
        )
    if cache is not None:
        cache[key] = (payload, signature, derivative)
    return derivative


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
    step = float(proper_time_step_ns)
    if not np.isfinite(step):
        raise ValueError("proper_time_step_ns must be finite")
    canonical_impulse = step * canonical_force
    mechanical_force = mechanical_lorentz_four_force_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        field_tensor=field.field.field_tensor,
        charge_native=observer_charge_native,
    )
    mechanical_impulse = step * mechanical_force
    return RetardedChargeSourceInteraction(
        field=field,
        canonical_potential_momentum=potential_momentum,
        canonical_four_force=canonical_force,
        canonical_four_impulse=canonical_impulse,
        mechanical_four_force=mechanical_force,
        mechanical_four_impulse=mechanical_impulse,
    )


def charge_source_interaction_from_response_native(
    response: RetardedChargeResponseGradientResult,
    *,
    four_velocity_mm_ns: Sequence[float] | np.ndarray,
    observer_charge_native: float,
    proper_time_step_ns: float,
    contraction_backend: str = "python",
) -> RetardedChargeSourceInteraction:
    """Contract one analytical response without materializing ``F`` or ``dF``."""

    from .antisymmetric_response_rfs import (
        antisymmetric_response_charge_force_native,
    )

    potential_momentum = canonical_potential_momentum_native(
        response.four_potential,
        charge_native=observer_charge_native,
    )
    canonical_force = canonical_four_force_from_potential_gradient_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        partial_a=response.partial_a,
        charge_native=observer_charge_native,
    )
    canonical_impulse = canonical_force * float(proper_time_step_ns)
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
    mechanical_impulse = mechanical_force * float(proper_time_step_ns)
    return RetardedChargeSourceInteraction(
        field=None,
        canonical_potential_momentum=potential_momentum,
        canonical_four_force=canonical_force,
        canonical_four_impulse=canonical_impulse,
        mechanical_four_force=mechanical_force,
        mechanical_four_impulse=mechanical_impulse,
        response=response,
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
    backend: str = "python",
) -> RetardedChargeSourceInteraction:
    """Evaluate exact non-self charge fields and ordinary responses once.

    ``canonical_four_force`` is ``dP^mu/dtau`` and remains a convention oracle.
    ``mechanical_four_force`` is ``dp^mu/dtau=(q/c)F.u`` and is the production
    exact-path translation response.  No RFS moment force, dipole source field,
    or radiation reaction is added here.
    """

    if str(backend).strip().lower() in {
        "numba_analytic_charge_response_serial",
        "numba_analytic_charge_dipole_response_serial",
    }:
        response = evaluate_retarded_charge_response_gradient_native(
            history,
            observer_event,
            excluded_source_indices=excluded_source_indices,
            require_complete_history=require_complete_history,
            relative_step=relative_step,
            minimum_step_mm=minimum_step_mm,
            root_tolerance_mm=root_tolerance_mm,
            max_root_iterations=max_root_iterations,
        )
        return charge_source_interaction_from_response_native(
            response=response,
            four_velocity_mm_ns=four_velocity_mm_ns,
            observer_charge_native=observer_charge_native,
            proper_time_step_ns=proper_time_step_ns,
            contraction_backend="numba_strict_serial",
        )
    field = evaluate_retarded_charge_field_gradient_native(
        history,
        observer_event,
        excluded_source_indices=excluded_source_indices,
        require_complete_history=require_complete_history,
        relative_step=relative_step,
        minimum_step_mm=minimum_step_mm,
        root_tolerance_mm=root_tolerance_mm,
        max_root_iterations=max_root_iterations,
        backend=backend,
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
    "charge_source_interaction_from_response_native",
    "evaluate_retarded_charge_source_interaction_native",
]
