"""Auditable step-doubling error norms for the future return integrator.

The module compares a single joint slab with the authoritative two-half-slab
path.  It contains no trajectory mutation or provider calls; rejected trials
therefore cannot alter accepted history through this layer.  Physics adapters
must supply mechanical momentum rather than canonical momentum and must sum
per-half diagnostic increments before constructing the refined sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorScale:
    """Absolute and relative scale for one physical error group."""

    absolute: float
    relative: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.absolute)
            or not np.isfinite(self.relative)
            or self.absolute < 0.0
            or self.relative < 0.0
            or (self.absolute == 0.0 and self.relative == 0.0)
        ):
            raise ValueError(
                "error scales must be finite, non-negative, and not both zero"
            )


@dataclass(frozen=True)
class StepDoublingTolerances:
    """Independent scales for the state groups used by adaptive acceptance."""

    position_mm: ErrorScale
    mechanical_momentum_native: ErrorScale
    rest_spin: ErrorScale
    diagnostics_native: ErrorScale


@dataclass(frozen=True)
class StepDoublingState:
    """One pair endpoint reduced to quantities relevant to local error."""

    position_mm: np.ndarray
    mechanical_momentum_native: np.ndarray
    rest_spin: np.ndarray
    diagnostics_native: np.ndarray


@dataclass(frozen=True)
class StepDoublingAssessment:
    """Scaled group errors and the resulting accept/reject decision."""

    accepted: bool
    normalized_error: float
    position_error: float
    mechanical_momentum_error: float
    rest_spin_error: float
    diagnostics_error: float


@dataclass(frozen=True)
class StepControllerConfig:
    """Bounded scalar controller for the next shared lab-time slab."""

    method_order: int
    safety_factor: float = 0.9
    minimum_factor: float = 0.2
    maximum_growth_factor: float = 2.0

    def __post_init__(self) -> None:
        if int(self.method_order) < 1:
            raise ValueError("method_order must be positive")
        values = (
            self.safety_factor,
            self.minimum_factor,
            self.maximum_growth_factor,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("step-controller factors must be finite and positive")
        if self.minimum_factor > 1.0:
            raise ValueError("minimum_factor must not exceed one")
        if self.maximum_growth_factor < 1.0:
            raise ValueError("maximum_growth_factor must be at least one")


def _validated_array(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _scaled_max_error(
    full_values: np.ndarray,
    refined_values: np.ndarray,
    *,
    scale: ErrorScale,
    richardson_denominator: float,
    name: str,
) -> float:
    full = _validated_array(full_values, f"full {name}")
    refined = _validated_array(refined_values, f"refined {name}")
    if full.shape != refined.shape:
        raise ValueError(f"full and refined {name} shapes must match")
    if full.size == 0:
        return 0.0
    local_error = np.abs(refined - full) / richardson_denominator
    denominator = scale.absolute + scale.relative * np.maximum(
        np.abs(full), np.abs(refined)
    )
    normalized = np.divide(
        local_error,
        denominator,
        out=np.zeros_like(local_error),
        where=denominator > 0.0,
    )
    return float(np.max(normalized))


def assess_step_doubling(
    full: StepDoublingState,
    refined: StepDoublingState,
    *,
    method_order: int,
    tolerances: StepDoublingTolerances,
) -> StepDoublingAssessment:
    """Compare one full slab with two half slabs using Richardson scaling.

    ``method_order`` is the observed order $p$ of the complete coupled path,
    not merely the order of one translational sub-kernel.  For the initial
    RFS-plus-Medina adapter this should conservatively remain $p=1$ until an
    end-to-end refinement study establishes otherwise.
    """

    method_order = int(method_order)
    if method_order < 1:
        raise ValueError("method_order must be positive")
    richardson_denominator = float(2**method_order - 1)
    position_error = _scaled_max_error(
        full.position_mm,
        refined.position_mm,
        scale=tolerances.position_mm,
        richardson_denominator=richardson_denominator,
        name="position",
    )
    momentum_error = _scaled_max_error(
        full.mechanical_momentum_native,
        refined.mechanical_momentum_native,
        scale=tolerances.mechanical_momentum_native,
        richardson_denominator=richardson_denominator,
        name="mechanical momentum",
    )
    spin_error = _scaled_max_error(
        full.rest_spin,
        refined.rest_spin,
        scale=tolerances.rest_spin,
        richardson_denominator=richardson_denominator,
        name="rest spin",
    )
    diagnostics_error = _scaled_max_error(
        full.diagnostics_native,
        refined.diagnostics_native,
        scale=tolerances.diagnostics_native,
        richardson_denominator=richardson_denominator,
        name="diagnostics",
    )
    normalized_error = max(
        position_error,
        momentum_error,
        spin_error,
        diagnostics_error,
    )
    return StepDoublingAssessment(
        accepted=bool(normalized_error <= 1.0),
        normalized_error=normalized_error,
        position_error=position_error,
        mechanical_momentum_error=momentum_error,
        rest_spin_error=spin_error,
        diagnostics_error=diagnostics_error,
    )


def propose_next_step_ns(
    current_step_ns: float,
    normalized_error: float,
    *,
    accepted: bool,
    config: StepControllerConfig,
    minimum_step_ns: float,
    maximum_step_ns: float,
) -> float:
    """Return a bounded next slab width without mutating controller state."""

    current_step_ns = float(current_step_ns)
    normalized_error = float(normalized_error)
    minimum_step_ns = float(minimum_step_ns)
    maximum_step_ns = float(maximum_step_ns)
    values = (current_step_ns, normalized_error, minimum_step_ns, maximum_step_ns)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("step-controller inputs must be finite")
    if current_step_ns <= 0.0 or minimum_step_ns <= 0.0:
        raise ValueError("step sizes must be positive")
    if maximum_step_ns < minimum_step_ns:
        raise ValueError("maximum_step_ns must not be below minimum_step_ns")
    if normalized_error < 0.0:
        raise ValueError("normalized_error must be non-negative")

    if normalized_error == 0.0:
        factor = config.maximum_growth_factor
    else:
        exponent = -1.0 / float(config.method_order + 1)
        factor = config.safety_factor * normalized_error**exponent
        factor = min(config.maximum_growth_factor, factor)
        factor = max(config.minimum_factor, factor)
    if not accepted:
        factor = min(1.0, factor)
    proposed = current_step_ns * factor
    return float(np.clip(proposed, minimum_step_ns, maximum_step_ns))


__all__ = [
    "ErrorScale",
    "StepControllerConfig",
    "StepDoublingAssessment",
    "StepDoublingState",
    "StepDoublingTolerances",
    "assess_step_doubling",
    "propose_next_step_ns",
]
