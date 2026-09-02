"""Transactional one-slab trials for the exact-retarded $1+1$ mode.

This module composes the shared-lab-time solver, immutable provisional source
histories, and pure endpoint canonical recomposition.  It deliberately does
not append accepted history or write checkpoints/public output.  A caller may
therefore discard the returned path without rollback work.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np

from .exact_pair_endpoint import finalize_exact_source_canonical_pair_states
from .self_consistency import SelfConsistencyConfig
from .shared_lab_time import (
    DEFAULT_PROPER_TIME_ROOT_MAX_ITERATIONS,
    SharedLabTimeError,
    SharedLabTimePair,
    solve_shared_lab_time_pair,
)
from .step_doubling import (
    StepDoublingAssessment,
    StepDoublingTolerances,
    assess_step_doubling,
    build_pair_step_doubling_state,
)
from .types import (
    ChronoMatchingMode,
    ExternalFieldConfig,
    GrowableTrajectoryBuilder,
    MagneticDipoleConfig,
    ParticleState,
    SimulationType,
    StartupMode,
    TrajectoryArrays,
    TrialTrajectoryHistory,
)

AdvanceRoleTrial = Callable[[float, ParticleState, ParticleState, Any], ParticleState]

if TYPE_CHECKING:
    from .causal_c5_dipole_provider import AcceptedPairCausalC5SourceHistory


@dataclass(frozen=True)
class ExactRoleSourceHistory:
    """Charge chronology plus an optional independent dipole history."""

    charge_history: Any
    dipole_source_collection: Any = None


@dataclass(frozen=True)
class ExactPairSlabTrial:
    """One finalized but unpublished pair slab and its source-history views."""

    pair: SharedLabTimePair
    rider_history: TrialTrajectoryHistory
    driver_history: TrialTrajectoryHistory


@dataclass(frozen=True)
class ExactPairEOMOptions:
    """Maintained equations-of-motion settings for a transactional pair trial."""

    aperture_radius_mm: float
    magnetic_dipole: MagneticDipoleConfig
    self_consistency: SelfConsistencyConfig | None = None
    chrono_mode: ChronoMatchingMode = ChronoMatchingMode.FAST
    radiation_reaction_mode: str = "off"
    external_field: ExternalFieldConfig | None = None
    step_idx: int | None = None
    cancel_callback: Any = None
    spin_interpolation_model: str = "causal_frozen_c1"

    def __post_init__(self) -> None:
        if not np.isfinite(self.aperture_radius_mm) or self.aperture_radius_mm <= 0.0:
            raise ValueError("aperture_radius_mm must be finite and positive")
        if not self.magnetic_dipole.enabled:
            raise ValueError("exact pair trials require magnetic-dipole/RFS mode")
        if self.magnetic_dipole.spin_model != "rfs_minimal_2021":
            raise ValueError("exact pair trials require rfs_minimal_2021")
        if (
            self.self_consistency is not None
            and self.self_consistency.enabled
            and self.self_consistency.convergence_mode != "fixed_geometry"
        ):
            raise ValueError("exact pair trials require fixed_geometry convergence")
        if self.spin_interpolation_model != "causal_frozen_c1":
            raise ValueError("exact pair trials require causal_frozen_c1 spin history")


@dataclass(frozen=True)
class ExactPairStepDoublingTrial:
    """One full path and the authoritative two-half trial path."""

    full: ExactPairSlabTrial
    midpoint: ExactPairSlabTrial
    refined: ExactPairSlabTrial
    assessment: StepDoublingAssessment
    health_failures: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        """Whether both the error norm and non-negotiable health gates pass."""

        return bool(self.assessment.accepted and not self.health_failures)


def make_exact_role_eom_advance(options: ExactPairEOMOptions) -> AdvanceRoleTrial:
    """Bind the maintained EOM to the transactional role-callback contract."""

    from .equations import retarded_equations_of_motion
    from .self_consistency import self_consistent_step

    def advance(
        proper_step_ns: float,
        observer_start: ParticleState,
        source_start: ParticleState,
        exact_source_history: Any,
    ) -> ParticleState:
        charge_history = exact_source_history
        dipole_source_collection = None
        if isinstance(exact_source_history, ExactRoleSourceHistory):
            charge_history = exact_source_history.charge_history
            dipole_source_collection = exact_source_history.dipole_source_collection
        return cast(
            ParticleState,
            self_consistent_step(
                retarded_equations_of_motion,
                proper_step_ns,
                [observer_start],
                [source_start],
                0,
                options.aperture_radius_mm,
                SimulationType.BUNCH_TO_BUNCH,
                options.self_consistency,
                options.chrono_mode,
                StartupMode.INERTIAL_PREHISTORY,
                step_idx=options.step_idx,
                cancel_callback=options.cancel_callback,
                radiation_reaction_mode=options.radiation_reaction_mode,
                external_field=options.external_field,
                magnetic_dipole=options.magnetic_dipole,
                exact_source_history=charge_history,
                exact_dipole_source_collection=dipole_source_collection,
                exact_source_spin_interpolation_model=(
                    options.spin_interpolation_model
                ),
            ),
        )

    return advance


def _history_tail_state(
    base: TrajectoryArrays,
    tail: tuple[ParticleState, ...],
    *,
    role: str,
) -> ParticleState:
    if base.n_particles != 1:
        raise SharedLabTimeError(
            f"{role} exact pair trial currently requires exactly one particle"
        )
    if tail:
        return copy.deepcopy(tail[-1])
    return copy.deepcopy(base.state_at(-1))


def _single_state_time(state: ParticleState, *, role: str) -> float:
    values = np.asarray(state.get("t", []), dtype=np.float64)
    if values.shape != (1,) or not np.all(np.isfinite(values)):
        raise SharedLabTimeError(f"{role} trial start must have one finite time")
    return float(values[0])


def _mark_accepted_canonical_offsets_ready(
    state: ParticleState,
    *,
    include_dipole_source: bool,
) -> None:
    """Restore readiness metadata omitted from public trajectory arrays."""

    particle_count = len(np.asarray(state.get("x", [])))
    state["charge_source_canonical_ready"] = np.ones(particle_count, dtype=bool)
    if include_dipole_source:
        state["dipole_source_canonical_ready"] = np.ones(particle_count, dtype=bool)


def _source_history(
    base: TrajectoryArrays,
    tail: tuple[ParticleState, ...],
) -> TrajectoryArrays | TrialTrajectoryHistory:
    return base if not tail else TrialTrajectoryHistory(base, tail)


def solve_exact_pair_slab_trial(
    *,
    accepted_rider_history: TrajectoryArrays,
    accepted_driver_history: TrajectoryArrays,
    advance_rider: AdvanceRoleTrial,
    advance_driver: AdvanceRoleTrial,
    delta_time_ns: float,
    rider_initial_proper_step_ns: float,
    driver_initial_proper_step_ns: float,
    magnetic_dipole: MagneticDipoleConfig,
    include_dipole_source: bool,
    rider_prior_tail: tuple[ParticleState, ...] = (),
    driver_prior_tail: tuple[ParticleState, ...] = (),
    causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None,
    spin_interpolation_model: str = "causal_frozen_c1",
    absolute_tolerance_ns: float = 1.0e-18,
    relative_tolerance: float = 1.0e-12,
    max_iterations: int = DEFAULT_PROPER_TIME_ROOT_MAX_ITERATIONS,
    max_bracket_expansions: int = 20,
    maximum_proper_step_ns: float = np.inf,
) -> ExactPairSlabTrial:
    """Return one endpoint-canonical pair slab without publishing history.

    ``advance_rider`` and ``advance_driver`` receive the proper step, detached
    observer/source states at the accepted slab boundary, and the exact source
    history view.  For a second half-step that view contains the accepted
    prefix plus the first provisional midpoint.
    """

    rider_prior_tail = tuple(rider_prior_tail)
    driver_prior_tail = tuple(driver_prior_tail)
    if len(rider_prior_tail) != len(driver_prior_tail):
        raise SharedLabTimeError("rider and driver trial tails must be aligned")
    if len(rider_prior_tail) > 1:
        raise SharedLabTimeError("one slab may begin after at most one trial midpoint")

    rider_start = _history_tail_state(
        accepted_rider_history, rider_prior_tail, role="rider"
    )
    driver_start = _history_tail_state(
        accepted_driver_history, driver_prior_tail, role="driver"
    )
    _mark_accepted_canonical_offsets_ready(
        rider_start,
        include_dipole_source=include_dipole_source,
    )
    _mark_accepted_canonical_offsets_ready(
        driver_start,
        include_dipole_source=include_dipole_source,
    )
    rider_start_time = _single_state_time(rider_start, role="rider")
    driver_start_time = _single_state_time(driver_start, role="driver")
    # Each endpoint root may lie one solver tolerance to either side of the
    # shared target.  The pair commit consequently accepts a two-tolerance
    # rider/driver separation.  Apply that same envelope at the next slab
    # boundary so a valid committed pair cannot become an invalid start.
    time_tolerance = 2.0 * (
        float(absolute_tolerance_ns)
        + float(relative_tolerance) * max(abs(rider_start_time), abs(driver_start_time))
    )
    if abs(rider_start_time - driver_start_time) > time_tolerance:
        raise SharedLabTimeError("rider and driver trial starts are not synchronized")
    start_time_ns = 0.5 * (rider_start_time + driver_start_time)

    rider_charge_history = _source_history(
        accepted_driver_history,
        driver_prior_tail,
    )
    driver_charge_history = _source_history(
        accepted_rider_history,
        rider_prior_tail,
    )
    rider_source_history: Any = rider_charge_history
    driver_source_history: Any = driver_charge_history
    if include_dipole_source and causal_c5_source_history is not None:
        rider_source_history = ExactRoleSourceHistory(
            charge_history=rider_charge_history,
            dipole_source_collection=causal_c5_source_history.driver,
        )
        driver_source_history = ExactRoleSourceHistory(
            charge_history=driver_charge_history,
            dipole_source_collection=causal_c5_source_history.rider,
        )
    provisional = solve_shared_lab_time_pair(
        advance_rider=lambda h: advance_rider(
            h,
            copy.deepcopy(rider_start),
            copy.deepcopy(driver_start),
            rider_source_history,
        ),
        advance_driver=lambda h: advance_driver(
            h,
            copy.deepcopy(driver_start),
            copy.deepcopy(rider_start),
            driver_source_history,
        ),
        start_time_ns=start_time_ns,
        delta_time_ns=delta_time_ns,
        rider_initial_proper_step_ns=rider_initial_proper_step_ns,
        driver_initial_proper_step_ns=driver_initial_proper_step_ns,
        absolute_tolerance_ns=absolute_tolerance_ns,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        max_bracket_expansions=max_bracket_expansions,
        maximum_proper_step_ns=maximum_proper_step_ns,
    )

    provisional_rider_history = TrialTrajectoryHistory(
        accepted_rider_history,
        rider_prior_tail + (provisional.rider.state,),
    )
    provisional_driver_history = TrialTrajectoryHistory(
        accepted_driver_history,
        driver_prior_tail + (provisional.driver.state,),
    )
    rider_state, driver_state = finalize_exact_source_canonical_pair_states(
        rider_state=provisional.rider.state,
        driver_state=provisional.driver.state,
        rider_endpoint_history=provisional_rider_history,
        driver_endpoint_history=provisional_driver_history,
        magnetic_dipole=magnetic_dipole,
        include_dipole_source=include_dipole_source,
        rider_dipole_source_collection=(
            None
            if causal_c5_source_history is None or not include_dipole_source
            else causal_c5_source_history.rider
        ),
        driver_dipole_source_collection=(
            None
            if causal_c5_source_history is None or not include_dipole_source
            else causal_c5_source_history.driver
        ),
        spin_interpolation_model=spin_interpolation_model,
    )
    finalized = replace(
        provisional,
        rider=replace(provisional.rider, state=rider_state),
        driver=replace(provisional.driver, state=driver_state),
    )
    return ExactPairSlabTrial(
        pair=finalized,
        rider_history=TrialTrajectoryHistory(
            accepted_rider_history,
            rider_prior_tail + (rider_state,),
        ),
        driver_history=TrialTrajectoryHistory(
            accepted_driver_history,
            driver_prior_tail + (driver_state,),
        ),
    )


def solve_exact_pair_step_doubling_trial(
    *,
    accepted_rider_history: TrajectoryArrays,
    accepted_driver_history: TrajectoryArrays,
    advance_rider: AdvanceRoleTrial,
    advance_driver: AdvanceRoleTrial,
    delta_time_ns: float,
    rider_initial_proper_step_ns: float,
    driver_initial_proper_step_ns: float,
    magnetic_dipole: MagneticDipoleConfig,
    include_dipole_source: bool,
    tolerances: StepDoublingTolerances,
    method_order: int = 1,
    causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None,
    build_causal_c5_midpoint_candidate: (
        Callable[
            [ExactPairSlabTrial, AcceptedPairCausalC5SourceHistory],
            AcceptedPairCausalC5SourceHistory,
        ]
        | None
    ) = None,
    spin_interpolation_model: str = "causal_frozen_c1",
    absolute_time_tolerance_ns: float = 1.0e-18,
    relative_time_tolerance: float = 1.0e-12,
    max_iterations: int = DEFAULT_PROPER_TIME_ROOT_MAX_ITERATIONS,
    max_bracket_expansions: int = 20,
    maximum_proper_step_ns: float = np.inf,
) -> ExactPairStepDoublingTrial:
    """Evaluate full and two-half paths without mutating accepted state."""

    def solve_slab(
        *,
        slab_time_ns: float,
        rider_proper_step_ns: float,
        driver_proper_step_ns: float,
        rider_tail: tuple[ParticleState, ...] = (),
        driver_tail: tuple[ParticleState, ...] = (),
        slab_causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None,
    ) -> ExactPairSlabTrial:
        return solve_exact_pair_slab_trial(
            accepted_rider_history=accepted_rider_history,
            accepted_driver_history=accepted_driver_history,
            advance_rider=advance_rider,
            advance_driver=advance_driver,
            delta_time_ns=slab_time_ns,
            rider_initial_proper_step_ns=rider_proper_step_ns,
            driver_initial_proper_step_ns=driver_proper_step_ns,
            magnetic_dipole=magnetic_dipole,
            include_dipole_source=include_dipole_source,
            rider_prior_tail=rider_tail,
            driver_prior_tail=driver_tail,
            causal_c5_source_history=slab_causal_c5_source_history,
            spin_interpolation_model=spin_interpolation_model,
            absolute_tolerance_ns=absolute_time_tolerance_ns,
            relative_tolerance=relative_time_tolerance,
            max_iterations=max_iterations,
            max_bracket_expansions=max_bracket_expansions,
            maximum_proper_step_ns=maximum_proper_step_ns,
        )

    full = solve_slab(
        slab_time_ns=delta_time_ns,
        rider_proper_step_ns=rider_initial_proper_step_ns,
        driver_proper_step_ns=driver_initial_proper_step_ns,
        slab_causal_c5_source_history=causal_c5_source_history,
    )
    half_time_ns = 0.5 * float(delta_time_ns)
    midpoint = solve_slab(
        slab_time_ns=half_time_ns,
        rider_proper_step_ns=0.5 * float(rider_initial_proper_step_ns),
        driver_proper_step_ns=0.5 * float(driver_initial_proper_step_ns),
        slab_causal_c5_source_history=causal_c5_source_history,
    )
    refined_c5_source_history = causal_c5_source_history
    if causal_c5_source_history is not None:
        if build_causal_c5_midpoint_candidate is None:
            from .causal_c5_dipole_provider import (
                AcceptedPairCausalC5SourceHistory,
            )

            refined_c5_source_history = AcceptedPairCausalC5SourceHistory(
                rider=causal_c5_source_history.rider.append_accepted_state(
                    midpoint.pair.rider.state
                ),
                driver=causal_c5_source_history.driver.append_accepted_state(
                    midpoint.pair.driver.state
                ),
            )
        else:
            refined_c5_source_history = build_causal_c5_midpoint_candidate(
                midpoint,
                causal_c5_source_history,
            )
    refined = solve_slab(
        slab_time_ns=half_time_ns,
        rider_proper_step_ns=midpoint.pair.rider.proper_step_ns,
        driver_proper_step_ns=midpoint.pair.driver.proper_step_ns,
        rider_tail=(midpoint.pair.rider.state,),
        driver_tail=(midpoint.pair.driver.state,),
        slab_causal_c5_source_history=refined_c5_source_history,
    )
    full_state = build_pair_step_doubling_state(
        rider_states=(full.pair.rider.state,),
        driver_states=(full.pair.driver.state,),
    )
    refined_state = build_pair_step_doubling_state(
        rider_states=(midpoint.pair.rider.state, refined.pair.rider.state),
        driver_states=(midpoint.pair.driver.state, refined.pair.driver.state),
    )
    assessment = assess_step_doubling(
        full_state,
        refined_state,
        method_order=method_order,
        tolerances=tolerances,
    )
    health_failures = _step_doubling_health_failures(
        accepted_rider_history=accepted_rider_history,
        accepted_driver_history=accepted_driver_history,
        full=full,
        midpoint=midpoint,
        refined=refined,
    )
    return ExactPairStepDoublingTrial(
        full=full,
        midpoint=midpoint,
        refined=refined,
        assessment=assessment,
        health_failures=health_failures,
    )


def _state_has_finite_medina_sample(state: ParticleState) -> bool:
    values = np.asarray(
        state.get("medina_external_force_sample_time", np.array([np.nan])),
        dtype=np.float64,
    )
    return bool(values.shape == (1,) and np.isfinite(values[0]))


def _state_observer_is_charged(state: ParticleState) -> bool:
    values = np.asarray(
        state.get("q_observer", state.get("q", np.zeros(1))),
        dtype=np.float64,
    )
    return bool(values.shape == (1,) and np.isfinite(values[0]) and values[0] != 0.0)


def _trial_state_health_failures(
    state: ParticleState,
    *,
    label: str,
    expected_medina_ready: bool | None,
) -> list[str]:
    failures: list[str] = []
    dead = np.asarray(state.get("_dead_particles", np.zeros(1, dtype=bool)), dtype=bool)
    if dead.shape != (1,) or bool(dead[0]):
        failures.append(f"{label}: particle death")
    capped = np.asarray(
        state.get("medina_impulse_capped", np.zeros(1, dtype=bool)), dtype=bool
    )
    if capped.shape != (1,) or bool(capped[0]):
        failures.append(f"{label}: Medina impulse cap")
    far_energy = np.asarray(
        state.get("radiation_energy", np.zeros(1)), dtype=np.float64
    )
    if far_energy.shape != (1,) or not np.isfinite(far_energy[0]):
        failures.append(f"{label}: invalid far-radiated energy")
    elif far_energy[0] < 0.0:
        failures.append(f"{label}: negative far-radiated energy")
    if expected_medina_ready is not None:
        ready = np.asarray(
            state.get("medina_force_derivative_ready", np.zeros(1, dtype=bool)),
            dtype=bool,
        )
        if ready.shape != (1,) or bool(ready[0]) is not expected_medina_ready:
            failures.append(f"{label}: unexpected Medina derivative readiness")
    return failures


def _step_doubling_health_failures(
    *,
    accepted_rider_history: TrajectoryArrays,
    accepted_driver_history: TrajectoryArrays,
    full: ExactPairSlabTrial,
    midpoint: ExactPairSlabTrial,
    refined: ExactPairSlabTrial,
) -> tuple[str, ...]:
    failures: list[str] = []
    paths = (
        (
            "rider",
            accepted_rider_history,
            full.pair.rider.state,
            midpoint.pair.rider.state,
            refined.pair.rider.state,
        ),
        (
            "driver",
            accepted_driver_history,
            full.pair.driver.state,
            midpoint.pair.driver.state,
            refined.pair.driver.state,
        ),
    )
    for role, accepted, full_state, midpoint_state, refined_state in paths:
        start = accepted.state_at(-1)
        medina_present = any(
            "medina_external_force_sample_time" in state
            for state in (full_state, midpoint_state, refined_state)
        )
        charged = _state_observer_is_charged(start)
        start_primed = _state_has_finite_medina_sample(start)
        first_ready = bool(start_primed) if medina_present and charged else None
        refined_ready = True if medina_present and charged else None
        failures.extend(
            _trial_state_health_failures(
                full_state,
                label=f"{role} full",
                expected_medina_ready=first_ready,
            )
        )
        failures.extend(
            _trial_state_health_failures(
                midpoint_state,
                label=f"{role} midpoint",
                expected_medina_ready=first_ready,
            )
        )
        failures.extend(
            _trial_state_health_failures(
                refined_state,
                label=f"{role} refined endpoint",
                expected_medina_ready=refined_ready,
            )
        )
    return tuple(failures)


def commit_accepted_exact_pair_step_doubling_trial(
    trial: ExactPairStepDoublingTrial,
    *,
    rider_builder: GrowableTrajectoryBuilder,
    driver_builder: GrowableTrajectoryBuilder,
) -> tuple[int, int]:
    """Jointly publish the authoritative midpoint and endpoint after acceptance.

    All four rows and both two-row capacity reservations are validated before
    the first append. Ordinary validation/allocation failures therefore leave
    both accepted histories unchanged. As with the one-row pair commit, a
    process-level interruption is recovered from the last atomic checkpoint.
    """

    if not trial.accepted:
        detail = "; ".join(trial.health_failures)
        suffix = f": {detail}" if detail else ""
        raise SharedLabTimeError(
            f"rejected step-doubling trial cannot be committed{suffix}"
        )
    if rider_builder.accepted_steps != driver_builder.accepted_steps:
        raise SharedLabTimeError("accepted rider and driver histories are misaligned")
    rider_states = (
        trial.midpoint.pair.rider.state,
        trial.refined.pair.rider.state,
    )
    driver_states = (
        trial.midpoint.pair.driver.state,
        trial.refined.pair.driver.state,
    )
    rider_builder.validate_append_steps(rider_states)
    driver_builder.validate_append_steps(driver_states)
    rider_builder.reserve_append_capacity(2)
    driver_builder.reserve_append_capacity(2)

    midpoint_rider_row = rider_builder.append_step(rider_states[0])
    midpoint_driver_row = driver_builder.append_step(driver_states[0])
    endpoint_rider_row = rider_builder.append_step(rider_states[1])
    endpoint_driver_row = driver_builder.append_step(driver_states[1])
    if midpoint_rider_row != midpoint_driver_row:
        raise RuntimeError("joint midpoint row indices diverged")
    if endpoint_rider_row != endpoint_driver_row:
        raise RuntimeError("joint endpoint row indices diverged")
    return midpoint_rider_row, endpoint_rider_row


__all__ = [
    "AdvanceRoleTrial",
    "ExactRoleSourceHistory",
    "ExactPairEOMOptions",
    "ExactPairSlabTrial",
    "ExactPairStepDoublingTrial",
    "commit_accepted_exact_pair_step_doubling_trial",
    "make_exact_role_eom_advance",
    "solve_exact_pair_slab_trial",
    "solve_exact_pair_step_doubling_trial",
]
