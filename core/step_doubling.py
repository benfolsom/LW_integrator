"""Auditable step-doubling error norms for the future return integrator.

The module compares a single joint slab with the authoritative two-half-slab
path.  It contains no trajectory mutation or provider calls; rejected trials
therefore cannot alter accepted history through this layer.  Physics adapters
must supply mechanical momentum rather than canonical momentum and must sum
per-half diagnostic increments before constructing the refined sample.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, cast

import numpy as np

from .constants import C_MMNS
from .types import ParticleState


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
    experimental_projection_ulp_floor: bool = False
    projection_ulp_multiplier: float = 4.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.projection_ulp_multiplier)
            or self.projection_ulp_multiplier <= 0.0
        ):
            raise ValueError("projection_ulp_multiplier must be finite and positive")


@dataclass(frozen=True)
class StepDoublingState:
    """One pair endpoint reduced to quantities relevant to local error."""

    position_mm: np.ndarray
    mechanical_momentum_native: np.ndarray
    rest_spin: np.ndarray
    diagnostics_native: np.ndarray
    particle_vector_scales: bool = False
    projection_reference_energy_native: np.ndarray | None = None


@dataclass(frozen=True)
class StepDoublingAssessment:
    """Scaled group errors and the resulting accept/reject decision."""

    accepted: bool
    normalized_error: float
    position_error: float
    mechanical_momentum_error: float
    rest_spin_error: float
    diagnostics_error: float
    position_error_index: tuple[int, ...]
    mechanical_momentum_error_index: tuple[int, ...]
    rest_spin_error_index: tuple[int, ...]
    diagnostics_error_index: tuple[int, ...]
    projection_floor_applied: bool = False
    projection_floor_changed_acceptance: bool = False
    diagnostics_error_without_floor: float | None = None
    maximum_projection_floor_native: float = 0.0


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


_PAIR_INCREMENT_DIAGNOSTICS = (
    "radiation_energy",
    "radiation_reaction_work",
    "medina_cross_field_energy_change",
    "mass_shell_projection_energy",
)


def _single_particle_vector(
    state: ParticleState,
    field_names: tuple[str, str, str],
    *,
    role: str,
) -> np.ndarray:
    values = np.asarray(
        [float(np.asarray(state[name], dtype=np.float64)[0]) for name in field_names],
        dtype=np.float64,
    )
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{role} {field_names} must contain finite scalars")
    return cast(np.ndarray, values)


def _pair_role_step_doubling_values(
    states: Sequence[ParticleState],
    *,
    role: str,
    include_spin_feedback: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not states:
        raise ValueError(f"{role} step-doubling path must contain a state")
    endpoint = states[-1]
    position = _single_particle_vector(endpoint, ("x", "y", "z"), role=role)
    beta = _single_particle_vector(endpoint, ("bx", "by", "bz"), role=role)
    gamma_values = np.asarray(endpoint.get("gamma", []), dtype=np.float64)
    mass_values = np.asarray(
        endpoint.get("m_species", endpoint.get("m", [])), dtype=np.float64
    )
    if (
        gamma_values.shape != (1,)
        or mass_values.shape != (1,)
        or not np.all(np.isfinite(gamma_values))
        or not np.all(np.isfinite(mass_values))
        or gamma_values[0] < 1.0
        or mass_values[0] <= 0.0
    ):
        raise ValueError(f"{role} endpoint gamma and mass must be physical scalars")
    mechanical_momentum = gamma_values[0] * mass_values[0] * C_MMNS * beta

    spin_names = ("spin_x", "spin_y", "spin_z")
    if not any(name in endpoint for name in spin_names):
        spin: np.ndarray = np.zeros(3, dtype=np.float64)
    elif not all(name in endpoint for name in spin_names):
        raise ValueError(f"{role} endpoint spin components must be all present")
    else:
        spin = _single_particle_vector(endpoint, spin_names, role=role)
    diagnostics: np.ndarray = np.zeros(
        len(_PAIR_INCREMENT_DIAGNOSTICS) + (2 if include_spin_feedback else 0),
        dtype=np.float64,
    )
    for state in states:
        for index, name in enumerate(_PAIR_INCREMENT_DIAGNOSTICS):
            values = np.asarray(state.get(name, np.zeros(1)), dtype=np.float64)
            if values.shape != (1,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{role} {name} must contain one finite increment")
            diagnostics[index] += float(values[0])
        if include_spin_feedback:
            from .experimental_spin_reaction import LinearSpinFeedbackRecord

            record = state.get("_linear_spin_feedback_record")
            if not isinstance(record, LinearSpinFeedbackRecord):
                raise ValueError(f"{role} experimental path is missing a recoil record")
            diagnostics[-2] += record.work_native
            diagnostics[-1] += (
                record.work_native - record.temporal_impulse_energy_native
            )
    return position, mechanical_momentum, spin, diagnostics


def build_pair_step_doubling_state(
    *,
    rider_states: Sequence[ParticleState],
    driver_states: Sequence[ParticleState],
    experimental_projection_ulp_floor: bool = False,
) -> StepDoublingState:
    """Reduce one full or two-half $1+1$ path to acceptance quantities.

    Spatial mechanical momentum is reconstructed from the endpoint
    ``gamma*m*c*beta`` so the comparison is independent of ordinary-potential
    gauge bookkeeping. Energy-like diagnostics are increments and are summed
    over both accepted half steps on the refined path.
    """

    counts = [len(states[-1]["x"]) for states in (rider_states, driver_states)]
    if max(counts) > 1:
        # Reuse the scalar reduction and its validation for every observer.
        values = []
        for role, states, count in zip(
            ("rider", "driver"), (rider_states, driver_states), counts
        ):
            for index in range(count):
                particle_states = [
                    {
                        key: (
                            value[index : index + 1]
                            if isinstance(value, np.ndarray)
                            and value.ndim
                            and len(value) == count
                            else value
                        )
                        for key, value in state.items()
                    }
                    for state in states
                ]
                values.append(
                    _pair_role_step_doubling_values(particle_states, role=role)
                )
        return StepDoublingState(
            position_mm=np.stack([value[0] for value in values]),
            mechanical_momentum_native=np.stack([value[1] for value in values]),
            rest_spin=np.stack([value[2] for value in values]),
            diagnostics_native=np.stack([value[3] for value in values]),
            particle_vector_scales=True,
            projection_reference_energy_native=(
                _projection_reference_energies(rider_states, driver_states)
                if experimental_projection_ulp_floor
                else None
            ),
        )

    include_spin_feedback = any(
        "_linear_spin_feedback_record" in state
        for states in (rider_states, driver_states)
        for state in states
    )
    rider = _pair_role_step_doubling_values(
        rider_states, role="rider", include_spin_feedback=include_spin_feedback
    )
    driver = _pair_role_step_doubling_values(
        driver_states, role="driver", include_spin_feedback=include_spin_feedback
    )
    return StepDoublingState(
        position_mm=np.stack((rider[0], driver[0])),
        mechanical_momentum_native=np.stack((rider[1], driver[1])),
        rest_spin=np.stack((rider[2], driver[2])),
        diagnostics_native=np.stack((rider[3], driver[3])),
        projection_reference_energy_native=(
            _projection_reference_energies(rider_states, driver_states)
            if experimental_projection_ulp_floor
            else None
        ),
    )


def _projection_reference_energies(
    *paths: Sequence[ParticleState],
) -> np.ndarray:
    """Endpoint kinetic energies, computed without subtracting rest energy.

    K = c |p| (|p| / (hypot(mc, |p|) + mc)). The reference is per physical
    observer, never population weighted, and does not change state bookkeeping.
    """
    energies = []
    for states in paths:
        state = states[-1]
        masses = np.asarray(state.get("m_species", state["m"]), dtype=float)
        gamma = np.asarray(state["gamma"], dtype=float)
        for index, mass in enumerate(masses):
            p = math.hypot(
                *(gamma[index] * mass * C_MMNS * state[f"b{a}"][index] for a in "xyz")
            )
            mc = mass * C_MMNS
            energies.append(C_MMNS * p * (p / (math.hypot(mc, p) + mc)))
    return np.asarray(energies, dtype=np.float64)


def _projection_floor_error(
    full: StepDoublingState,
    refined: StepDoublingState,
    tolerances: StepDoublingTolerances,
    richardson_denominator: float,
) -> tuple[float, tuple[int, ...], bool, float]:
    """Experimental per-observer floor for projection energy column only."""
    f = _validated_array(full.diagnostics_native, "full diagnostics")
    r = _validated_array(refined.diagnostics_native, "refined diagnostics")
    column = _PAIR_INCREMENT_DIAGNOSTICS.index("mass_shell_projection_energy")
    if f.ndim != 2 or f.shape[1] < len(_PAIR_INCREMENT_DIAGNOSTICS):
        raise ValueError("projection floor requires named pair diagnostic columns")
    references = []
    for state in (full, refined):
        if state.projection_reference_energy_native is None:
            raise ValueError("projection floor requires endpoint kinetic energies")
        reference = _validated_array(
            state.projection_reference_energy_native, "projection reference energy"
        )
        if reference.shape != (f.shape[0],) or np.any(reference < 0.0):
            raise ValueError("projection reference energy shape or sign is invalid")
        references.append(reference)
    reference = np.maximum(*references)
    floor = tolerances.projection_ulp_multiplier * np.spacing(reference)
    if not np.all(np.isfinite(floor)):
        raise ValueError("projection floor must be finite")
    scale = tolerances.diagnostics_native
    absolute = np.full_like(f, scale.absolute)
    absolute[:, column] = np.maximum(scale.absolute, floor)
    denominator = absolute + scale.relative * np.maximum(np.abs(f), np.abs(r))
    error = np.abs(r - f) / richardson_denominator
    normalized = np.divide(
        error, denominator, out=np.zeros_like(error), where=denominator > 0.0
    )
    index = tuple(int(v) for v in np.unravel_index(np.argmax(normalized), f.shape))
    return (
        float(normalized[index]),
        index,
        bool(np.any(floor > scale.absolute)),
        float(np.max(floor)),
    )


def _validated_array(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(np.ndarray, result)


def _scaled_max_error(
    full_values: np.ndarray,
    refined_values: np.ndarray,
    *,
    scale: ErrorScale,
    richardson_denominator: float,
    name: str,
    particle_vector_scales: bool = False,
) -> tuple[float, tuple[int, ...]]:
    full = _validated_array(full_values, f"full {name}")
    refined = _validated_array(refined_values, f"refined {name}")
    if full.shape != refined.shape:
        raise ValueError(f"full and refined {name} shapes must match")
    if full.size == 0:
        return 0.0, ()
    local_error = np.abs(refined - full) / richardson_denominator
    if particle_vector_scales:
        # Relative vector accuracy must not depend on the coordinate axes or
        # collapse when an orbit's momentum component crosses zero.
        denominator = scale.absolute + scale.relative * np.maximum(
            np.linalg.norm(full, axis=-1), np.linalg.norm(refined, axis=-1)
        )
        errors = np.linalg.norm(local_error, axis=-1) / denominator
        index = int(np.argmax(errors))
        return float(errors[index]), (index, int(np.argmax(local_error[index])))
    denominator = scale.absolute + scale.relative * np.maximum(
        np.abs(full), np.abs(refined)
    )
    normalized = np.divide(
        local_error,
        denominator,
        out=np.zeros_like(local_error),
        where=denominator > 0.0,
    )
    flat_index = int(np.argmax(normalized))
    maximum_index = tuple(
        int(value) for value in np.unravel_index(flat_index, full.shape)
    )
    return float(normalized[maximum_index]), maximum_index


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

    if full.particle_vector_scales != refined.particle_vector_scales:
        raise ValueError("full and refined vector error scales must match")
    method_order = int(method_order)
    if method_order < 1:
        raise ValueError("method_order must be positive")
    richardson_denominator = float(2**method_order - 1)
    position_error, position_error_index = _scaled_max_error(
        full.position_mm,
        refined.position_mm,
        scale=tolerances.position_mm,
        richardson_denominator=richardson_denominator,
        name="position",
        particle_vector_scales=full.particle_vector_scales,
    )
    momentum_error, momentum_error_index = _scaled_max_error(
        full.mechanical_momentum_native,
        refined.mechanical_momentum_native,
        scale=tolerances.mechanical_momentum_native,
        richardson_denominator=richardson_denominator,
        name="mechanical momentum",
        particle_vector_scales=full.particle_vector_scales,
    )
    spin_error, spin_error_index = _scaled_max_error(
        full.rest_spin,
        refined.rest_spin,
        scale=tolerances.rest_spin,
        richardson_denominator=richardson_denominator,
        name="rest spin",
        particle_vector_scales=full.particle_vector_scales,
    )
    diagnostics_error, diagnostics_error_index = _scaled_max_error(
        full.diagnostics_native,
        refined.diagnostics_native,
        scale=tolerances.diagnostics_native,
        richardson_denominator=richardson_denominator,
        name="diagnostics",
    )
    without_floor = None
    floor_applied = False
    maximum_floor = 0.0
    if tolerances.experimental_projection_ulp_floor:
        without_floor = diagnostics_error
        diagnostics_error, diagnostics_error_index, floor_applied, maximum_floor = (
            _projection_floor_error(full, refined, tolerances, richardson_denominator)
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
        position_error_index=position_error_index,
        mechanical_momentum_error_index=momentum_error_index,
        rest_spin_error_index=spin_error_index,
        diagnostics_error_index=diagnostics_error_index,
        projection_floor_applied=floor_applied,
        projection_floor_changed_acceptance=bool(
            normalized_error <= 1.0
            and without_floor is not None
            and without_floor > 1.0
        ),
        diagnostics_error_without_floor=without_floor,
        maximum_projection_floor_native=maximum_floor,
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
    "build_pair_step_doubling_state",
    "propose_next_step_ns",
]
