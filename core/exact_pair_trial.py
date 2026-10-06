"""Transactional one-slab trials for the exact-retarded $1+1$ mode.

This module composes the shared-lab-time solver, immutable provisional source
histories, and pure endpoint canonical recomposition.  It deliberately does
not append accepted history or write checkpoints/public output.  A caller may
therefore discard the returned path without rollback work.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np

from .particle_state_copy import copy_particle_state

from .exact_pair_endpoint import finalize_exact_source_canonical_pair_states
from .self_consistency import SelfConsistencyConfig
from .shared_lab_time import (
    DEFAULT_PROPER_TIME_ROOT_MAX_ITERATIONS,
    SharedLabTimeError,
    SharedLabTimePair,
    solve_shared_lab_time_bunches,
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

AdvanceRoleTrial = Callable[
    [float | np.ndarray, ParticleState, ParticleState, Any], ParticleState
]

if TYPE_CHECKING:
    from .causal_c5_dipole_provider import AcceptedPairCausalC5SourceHistory
    from .causal_local_source_history import AcceptedPairCausalLocalSourceHistory


@dataclass(frozen=True)
class ExactRoleSourceHistory:
    """Charge chronology plus an optional independent dipole history."""

    charge_history: Any
    observer_history: Any = None
    dipole_source_collection: Any = None
    observer_spin_reduction_history: Any = None
    observer_proper_time_ns: float | None = None


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
    space_charge: Any = None
    chrono_mode: ChronoMatchingMode = ChronoMatchingMode.FAST
    radiation_reaction_mode: str = "off"
    external_field: ExternalFieldConfig | None = None
    step_idx: int | None = None
    cancel_callback: Any = None
    spin_interpolation_model: str = "causal_frozen_c1"
    moment_impulse_diagnostic: Any = None
    cache_start_charge_fields: bool = True
    beamline_geometry: Any = None

    def __post_init__(self) -> None:
        if (
            self.magnetic_dipole.intrinsic_spin_self_reaction_mode
            == "experimental_linear_spin"
            and self.radiation_reaction_mode
            not in {"off", "medina_lad", "medina_lad_validity"}
        ):
            raise ValueError("experimental spin recoil supports only off or medina_lad")
        if self.moment_impulse_diagnostic is not None:
            if not callable(self.moment_impulse_diagnostic):
                raise ValueError("moment impulse diagnostic must be callable")
            if self.radiation_reaction_mode not in (
                "off",
                "medina_lad",
                "medina_lad_validity",
            ):
                raise ValueError(
                    "moment impulse diagnostic supports only off or medina_lad"
                )
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

    eom = retarded_equations_of_motion
    experimental = (
        options.magnetic_dipole.intrinsic_spin_self_reaction_mode
        == "experimental_linear_spin"
    )
    if experimental:
        from functools import partial

        eom = partial(eom, _experimental_linear_spin_adapter=True)
    if options.moment_impulse_diagnostic is not None:
        from functools import partial

        eom = partial(
            eom,
            moment_impulse_diagnostic=options.moment_impulse_diagnostic,
        )

    cached_contexts: list[
        tuple[
            tuple[Any, ...],
            tuple[np.ndarray, ...],
            dict[tuple[str, int], Any],
            tuple[Any, ...],
        ]
    ] = []
    cached_fields: dict[tuple[str, int], Any] = {}

    def advance(
        proper_step_ns: float | np.ndarray,
        observer_start: ParticleState,
        source_start: ParticleState,
        exact_source_history: Any,
        *,
        _skip_endpoint_diagnostic: bool = False,
        _clock_observer_index: int | None = None,
    ) -> ParticleState:
        nonlocal cached_fields
        cache_fields = bool(
            options.cache_start_charge_fields
            and len(observer_start.get("x", np.empty(0))) > 1
            and not options.magnetic_dipole.source.active
        )
        if cache_fields:
            # Full and first-half slabs wrap the same immutable histories in
            # separate role objects. Their start fields have identical inputs.
            histories = (
                (
                    exact_source_history.charge_history,
                    exact_source_history.observer_history,
                )
                if isinstance(exact_source_history, ExactRoleSourceHistory)
                else (exact_source_history,)
            )
            geometry = tuple(
                np.asarray(
                    observer_start[key]
                    if key in observer_start
                    else np.zeros_like(observer_start["x"])
                )
                for key in (
                    "t",
                    "x",
                    "y",
                    "z",
                    "source_time_low_ns",
                    "source_time_tail_ns",
                    *(
                        f"source_position_{part}_{axis}"
                        for part in ("low", "tail")
                        for axis in "xyz"
                    ),
                )
            )
            revisions = []
            for history in histories:
                base = (
                    history.base
                    if isinstance(history, TrialTrajectoryHistory)
                    else history
                )
                if isinstance(base, TrajectoryArrays):
                    base.require_current_storage()
                revisions.append(getattr(base, "storage_generation", None))
            revision_key = tuple(revisions)
            for index, context in enumerate(cached_contexts):
                if (
                    len(context[0]) == len(histories)
                    and all(a is b for a, b in zip(context[0], histories))
                    and context[3] == revision_key
                    and all(
                        np.array_equal(value, saved)
                        for value, saved in zip(geometry, context[1])
                    )
                ):
                    cached_contexts.append(cached_contexts.pop(index))
                    cached_fields = context[2]
                    break
            else:
                # One adapter serves both roles. Retain their two contexts across
                # the full and first-half paths; provisional midpoints evict them.
                cached_fields = {}
                cached_contexts.append(
                    (
                        histories,
                        tuple(value.copy() for value in geometry),
                        cached_fields,
                        revision_key,
                    )
                )
                if len(cached_contexts) > 2:
                    cached_contexts.pop(0)
        charge_history = exact_source_history
        observer_history = None
        dipole_source_collection = None
        if isinstance(exact_source_history, ExactRoleSourceHistory):
            charge_history = exact_source_history.charge_history
            observer_history = exact_source_history.observer_history
            dipole_source_collection = exact_source_history.dipole_source_collection
        if experimental and (
            not isinstance(exact_source_history, ExactRoleSourceHistory)
            or exact_source_history.observer_spin_reduction_history is None
            or exact_source_history.observer_proper_time_ns is None
        ):
            raise ValueError(
                "experimental spin recoil requires accepted observer reduction history"
            )

        eom_for_step: Callable[..., ParticleState]
        if np.ndim(proper_step_ns) != 0:
            from functools import partial

            eom_for_step = partial(
                eom, _particle_proper_steps_ns=np.asarray(proper_step_ns)
            )
            scalar_step = float(np.asarray(proper_step_ns)[0])
        else:
            eom_for_step = eom
            scalar_step = cast(float, proper_step_ns)

        def run(bound_eom: Callable[..., ParticleState]) -> ParticleState:
            if cache_fields:
                from functools import partial

                bound_eom = partial(bound_eom, _exact_trial_charge_fields=cached_fields)
                if _skip_endpoint_diagnostic:
                    bound_eom = partial(
                        bound_eom, _skip_exact_endpoint_field_diagnostic=True
                    )
                    if _clock_observer_index is not None:
                        bound_eom = partial(
                            bound_eom,
                            _exact_clock_observer_index=_clock_observer_index,
                        )
            return cast(
                ParticleState,
                self_consistent_step(
                    bound_eom,
                    scalar_step,
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
                    beamline_geometry=options.beamline_geometry,
                    exact_source_history=charge_history,
                    traj_soa=observer_history,
                    space_charge=options.space_charge,
                    exact_dipole_source_collection=dipole_source_collection,
                    exact_source_spin_interpolation_model=(
                        options.spin_interpolation_model
                    ),
                ),
            )

        result: ParticleState
        if (
            options.moment_impulse_diagnostic is not None
            and options.radiation_reaction_mode in {"medina_lad", "medina_lad_validity"}
        ):
            from functools import partial

            from .moment_medina_diagnostic import match_medina_force

            result = match_medina_force(
                lambda force: run(partial(eom, moment_radiation_force_native=force)),
                particle_count=len(observer_start["x"]),
            )
        else:
            result = run(eom_for_step)
        if experimental:
            from .experimental_spin_reaction import (
                apply_experimental_linear_spin_feedback,
            )

            result = apply_experimental_linear_spin_feedback(
                result=result,
                start=observer_start,
                accepted_history=exact_source_history.observer_spin_reduction_history,
                proper_time_ns=exact_source_history.observer_proper_time_ns,
                proper_step_ns=scalar_step,
            )
        return result

    if (
        options.cache_start_charge_fields
        and not options.magnetic_dipole.source.active
        and options.magnetic_dipole.intrinsic_spin_self_reaction_mode == "off"
    ):
        from functools import partial

        # Ordinary calls return complete diagnostics. Bunch clock queries are
        # followed by one complete vector evaluation at the solved steps.
        setattr(
            advance, "_clock_query", partial(advance, _skip_endpoint_diagnostic=True)
        )
        setattr(
            advance,
            "_clock_observer_query",
            lambda steps, index, *args: advance(
                steps,
                *args,
                _skip_endpoint_diagnostic=True,
                _clock_observer_index=index,
            ),
        )
    return advance


def _history_tail_state(
    base: TrajectoryArrays,
    tail: tuple[ParticleState, ...],
    *,
    role: str,
) -> ParticleState:
    if tail:
        return copy_particle_state(tail[-1])
    return copy_particle_state(base.state_at(-1))


def _single_state_time(state: ParticleState, *, role: str) -> float:
    values = np.asarray(state.get("t", []), dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise SharedLabTimeError(f"{role} trial start must have finite times")
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
    same_bunch_fields: bool = False,
    beamline_geometry: Any = None,
    rider_prior_tail: tuple[ParticleState, ...] = (),
    driver_prior_tail: tuple[ParticleState, ...] = (),
    causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None,
    causal_local_source_history: AcceptedPairCausalLocalSourceHistory | None = None,
    build_causal_local_endpoint_candidate: (
        Callable[
            [
                tuple[ParticleState, ...],
                tuple[ParticleState, ...],
                AcceptedPairCausalLocalSourceHistory,
            ],
            AcceptedPairCausalLocalSourceHistory,
        ]
        | None
    ) = None,
    spin_interpolation_model: str = "causal_frozen_c1",
    absolute_tolerance_ns: float = 1.0e-18,
    relative_tolerance: float = 1.0e-12,
    max_iterations: int = DEFAULT_PROPER_TIME_ROOT_MAX_ITERATIONS,
    max_bracket_expansions: int = 20,
    maximum_proper_step_ns: float = np.inf,
    intrinsic_spin_reduction_history: Any = None,
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
    if causal_c5_source_history is not None and causal_local_source_history is not None:
        raise ValueError("causal C5 and causal local histories are mutually exclusive")

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
    bunch_mode = (
        accepted_rider_history.n_particles > 1
        or accepted_driver_history.n_particles > 1
    )
    if bunch_mode and (
        include_dipole_source
        or magnetic_dipole.intrinsic_spin_self_reaction_mode != "off"
    ):
        raise NotImplementedError(
            "adaptive bunch trials support charge sources with spin self-reaction off"
        )
    rider_source_history: Any = rider_charge_history
    driver_source_history: Any = driver_charge_history
    if bunch_mode:
        rider_source_history = ExactRoleSourceHistory(
            charge_history=rider_charge_history,
            observer_history=_source_history(accepted_rider_history, rider_prior_tail),
        )
        driver_source_history = ExactRoleSourceHistory(
            charge_history=driver_charge_history,
            observer_history=_source_history(
                accepted_driver_history, driver_prior_tail
            ),
        )
    dipole_history = causal_c5_source_history or causal_local_source_history
    if include_dipole_source and dipole_history is not None:
        rider_source_history = ExactRoleSourceHistory(
            charge_history=rider_charge_history,
            dipole_source_collection=dipole_history.driver,
        )
        driver_source_history = ExactRoleSourceHistory(
            charge_history=driver_charge_history,
            dipole_source_collection=dipole_history.rider,
        )
    if magnetic_dipole.intrinsic_spin_self_reaction_mode == "experimental_linear_spin":
        if intrinsic_spin_reduction_history is None:
            raise ValueError("experimental spin recoil requires pair reduction history")
        rider_source_history = ExactRoleSourceHistory(
            charge_history=rider_charge_history,
            dipole_source_collection=(
                None
                if dipole_history is None or not include_dipole_source
                else dipole_history.driver
            ),
            observer_spin_reduction_history=intrinsic_spin_reduction_history.rider,
            observer_proper_time_ns=intrinsic_spin_reduction_history.rider_endpoint_proper_time_ns,
        )
        driver_source_history = ExactRoleSourceHistory(
            charge_history=driver_charge_history,
            dipole_source_collection=(
                None
                if dipole_history is None or not include_dipole_source
                else dipole_history.rider
            ),
            observer_spin_reduction_history=intrinsic_spin_reduction_history.driver,
            observer_proper_time_ns=intrinsic_spin_reduction_history.driver_endpoint_proper_time_ns,
        )
    solve_pair = (
        solve_shared_lab_time_bunches if bunch_mode else solve_shared_lab_time_pair
    )
    bunch_options: dict[str, Any] = (
        {"rider_start": rider_start, "driver_start": driver_start} if bunch_mode else {}
    )
    rider_query = advance_rider
    driver_query = advance_driver
    if bunch_mode:
        rider_query = getattr(advance_rider, "_clock_query", advance_rider)
        driver_query = getattr(advance_driver, "_clock_query", advance_driver)
        if rider_query is not advance_rider:
            bunch_options["finalize_rider"] = lambda h: advance_rider(
                h,
                copy_particle_state(rider_start),
                copy_particle_state(driver_start),
                rider_source_history,
            )
        if driver_query is not advance_driver:
            bunch_options["finalize_driver"] = lambda h: advance_driver(
                h,
                copy_particle_state(driver_start),
                copy_particle_state(rider_start),
                driver_source_history,
            )
        for role, advance, observer, source, history in (
            ("rider", advance_rider, rider_start, driver_start, rider_source_history),
            (
                "driver",
                advance_driver,
                driver_start,
                rider_start,
                driver_source_history,
            ),
        ):
            observer_query = getattr(advance, "_clock_observer_query", None)
            if observer_query is not None:

                def query(
                    h,
                    index,
                    query=observer_query,
                    observer=observer,
                    source=source,
                    history=history,
                ):
                    return query(
                        h,
                        index,
                        copy_particle_state(observer),
                        copy_particle_state(source),
                        history,
                    )

                bunch_options[f"query_{role}_observer"] = query
    provisional = solve_pair(
        **bunch_options,
        advance_rider=lambda h: rider_query(
            h,
            copy_particle_state(rider_start),
            copy_particle_state(driver_start),
            rider_source_history,
        ),
        advance_driver=lambda h: driver_query(
            h,
            copy_particle_state(driver_start),
            copy_particle_state(rider_start),
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
    endpoint_dipole_history = dipole_history
    if include_dipole_source and causal_local_source_history is not None:
        # Endpoint canonical momentum is evaluated at the new simultaneous
        # pair event. At close separation its retarded source event can lie
        # inside this provisional slab rather than in the accepted prefix.
        # Supply the unpublished endpoint kinematics to the local provider;
        # the candidate remains detached and disappears if the slab or outer
        # step-doubling trial is rejected.
        if build_causal_local_endpoint_candidate is None:
            from .causal_local_source_history import (
                AcceptedPairCausalLocalSourceHistory,
            )

            endpoint_dipole_history = AcceptedPairCausalLocalSourceHistory(
                rider=causal_local_source_history.rider.append_accepted_state(
                    provisional.rider.state
                ),
                driver=causal_local_source_history.driver.append_accepted_state(
                    provisional.driver.state
                ),
            )
        else:
            endpoint_dipole_history = build_causal_local_endpoint_candidate(
                rider_prior_tail + (provisional.rider.state,),
                driver_prior_tail + (provisional.driver.state,),
                causal_local_source_history,
            )
    rider_state, driver_state = finalize_exact_source_canonical_pair_states(
        rider_state=provisional.rider.state,
        driver_state=provisional.driver.state,
        same_bunch_fields=same_bunch_fields,
        beamline_geometry=beamline_geometry,
        rider_endpoint_history=provisional_rider_history,
        driver_endpoint_history=provisional_driver_history,
        magnetic_dipole=magnetic_dipole,
        include_dipole_source=include_dipole_source,
        rider_dipole_source_collection=(
            None
            if endpoint_dipole_history is None or not include_dipole_source
            else endpoint_dipole_history.rider
        ),
        driver_dipole_source_collection=(
            None
            if endpoint_dipole_history is None or not include_dipole_source
            else endpoint_dipole_history.driver
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
    same_bunch_fields: bool = False,
    beamline_geometry: Any = None,
    tolerances: StepDoublingTolerances,
    method_order: int = 1,
    causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None,
    causal_local_source_history: AcceptedPairCausalLocalSourceHistory | None = None,
    build_causal_c5_midpoint_candidate: (
        Callable[
            [ExactPairSlabTrial, AcceptedPairCausalC5SourceHistory],
            AcceptedPairCausalC5SourceHistory,
        ]
        | None
    ) = None,
    build_causal_local_midpoint_candidate: (
        Callable[
            [ExactPairSlabTrial, AcceptedPairCausalLocalSourceHistory],
            AcceptedPairCausalLocalSourceHistory,
        ]
        | None
    ) = None,
    build_causal_local_endpoint_candidate: (
        Callable[
            [
                tuple[ParticleState, ...],
                tuple[ParticleState, ...],
                AcceptedPairCausalLocalSourceHistory,
            ],
            AcceptedPairCausalLocalSourceHistory,
        ]
        | None
    ) = None,
    spin_interpolation_model: str = "causal_frozen_c1",
    absolute_time_tolerance_ns: float = 1.0e-18,
    relative_time_tolerance: float = 1.0e-12,
    max_iterations: int = DEFAULT_PROPER_TIME_ROOT_MAX_ITERATIONS,
    max_bracket_expansions: int = 20,
    maximum_proper_step_ns: float = np.inf,
    intrinsic_spin_reduction_history: Any = None,
) -> ExactPairStepDoublingTrial:
    """Evaluate full and two-half paths without mutating accepted state."""

    if causal_c5_source_history is not None and causal_local_source_history is not None:
        raise ValueError("causal C5 and causal local histories are mutually exclusive")

    def solve_slab(
        *,
        slab_time_ns: float,
        rider_proper_step_ns: float,
        driver_proper_step_ns: float,
        rider_tail: tuple[ParticleState, ...] = (),
        driver_tail: tuple[ParticleState, ...] = (),
        slab_causal_c5_source_history: AcceptedPairCausalC5SourceHistory | None = None,
        slab_causal_local_source_history: (
            AcceptedPairCausalLocalSourceHistory | None
        ) = None,
        slab_reduction_history: Any = None,
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
            same_bunch_fields=same_bunch_fields,
            beamline_geometry=beamline_geometry,
            rider_prior_tail=rider_tail,
            driver_prior_tail=driver_tail,
            causal_c5_source_history=slab_causal_c5_source_history,
            causal_local_source_history=slab_causal_local_source_history,
            build_causal_local_endpoint_candidate=(
                build_causal_local_endpoint_candidate
            ),
            spin_interpolation_model=spin_interpolation_model,
            absolute_tolerance_ns=absolute_time_tolerance_ns,
            relative_tolerance=relative_time_tolerance,
            max_iterations=max_iterations,
            max_bracket_expansions=max_bracket_expansions,
            maximum_proper_step_ns=maximum_proper_step_ns,
            intrinsic_spin_reduction_history=slab_reduction_history,
        )

    full = solve_slab(
        slab_reduction_history=intrinsic_spin_reduction_history,
        slab_time_ns=delta_time_ns,
        rider_proper_step_ns=rider_initial_proper_step_ns,
        driver_proper_step_ns=driver_initial_proper_step_ns,
        slab_causal_c5_source_history=causal_c5_source_history,
        slab_causal_local_source_history=causal_local_source_history,
    )
    half_time_ns = 0.5 * float(delta_time_ns)
    midpoint = solve_slab(
        slab_reduction_history=intrinsic_spin_reduction_history,
        slab_time_ns=half_time_ns,
        rider_proper_step_ns=0.5 * float(rider_initial_proper_step_ns),
        driver_proper_step_ns=0.5 * float(driver_initial_proper_step_ns),
        slab_causal_c5_source_history=causal_c5_source_history,
        slab_causal_local_source_history=causal_local_source_history,
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
    refined_local_source_history = causal_local_source_history
    if causal_local_source_history is not None:
        if build_causal_local_midpoint_candidate is None:
            from .causal_local_source_history import (
                AcceptedPairCausalLocalSourceHistory,
            )

            refined_local_source_history = AcceptedPairCausalLocalSourceHistory(
                rider=causal_local_source_history.rider.append_accepted_state(
                    midpoint.pair.rider.state
                ),
                driver=causal_local_source_history.driver.append_accepted_state(
                    midpoint.pair.driver.state
                ),
            )
        else:
            refined_local_source_history = build_causal_local_midpoint_candidate(
                midpoint,
                causal_local_source_history,
            )
    refined_reduction_history = intrinsic_spin_reduction_history
    if magnetic_dipole.intrinsic_spin_self_reaction_mode == "experimental_linear_spin":
        from .spin_self_force_reduction_history import (
            build_midpoint_intrinsic_spin_reduction_candidate,
        )

        refined_reduction_history = build_midpoint_intrinsic_spin_reduction_candidate(
            midpoint, intrinsic_spin_reduction_history
        )
    refined = solve_slab(
        slab_reduction_history=refined_reduction_history,
        slab_time_ns=half_time_ns,
        rider_proper_step_ns=midpoint.pair.rider.proper_step_ns,
        driver_proper_step_ns=midpoint.pair.driver.proper_step_ns,
        rider_tail=(midpoint.pair.rider.state,),
        driver_tail=(midpoint.pair.driver.state,),
        slab_causal_c5_source_history=refined_c5_source_history,
        slab_causal_local_source_history=refined_local_source_history,
    )
    full_state = build_pair_step_doubling_state(
        rider_states=(full.pair.rider.state,),
        driver_states=(full.pair.driver.state,),
        diagnostic_ulp_floor=tolerances.diagnostic_ulp_floor,
    )
    refined_state = build_pair_step_doubling_state(
        rider_states=(midpoint.pair.rider.state, refined.pair.rider.state),
        driver_states=(midpoint.pair.driver.state, refined.pair.driver.state),
        diagnostic_ulp_floor=tolerances.diagnostic_ulp_floor,
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
    return bool(values.ndim == 1 and values.size and np.all(np.isfinite(values)))


def _state_observer_is_charged(state: ParticleState) -> bool:
    values = np.asarray(
        state.get("q_observer", state.get("q", np.zeros(1))),
        dtype=np.float64,
    )
    return bool(values.ndim == 1 and np.any(np.isfinite(values) & (values != 0.0)))


def _trial_state_health_failures(
    state: ParticleState,
    *,
    label: str,
    expected_medina_ready: bool | None,
) -> list[str]:
    failures: list[str] = []
    count = len(state["x"])
    dead = np.asarray(
        state.get("_dead_particles", np.zeros(count, dtype=bool)), dtype=bool
    )
    if dead.shape != (count,) or bool(np.any(dead)):
        failures.append(f"{label}: particle death")
    capped = np.asarray(
        state.get("medina_impulse_capped", np.zeros(count, dtype=bool)), dtype=bool
    )
    if capped.shape != (count,) or bool(np.any(capped)):
        failures.append(f"{label}: Medina impulse cap")
    far_energy = np.asarray(
        state.get("radiation_energy", np.zeros(count)), dtype=np.float64
    )
    if far_energy.shape != (count,) or not np.all(np.isfinite(far_energy)):
        failures.append(f"{label}: invalid far-radiated energy")
    elif np.any(far_energy < 0.0):
        failures.append(f"{label}: negative far-radiated energy")
    if expected_medina_ready is not None:
        expected = np.full(count, expected_medina_ready, dtype=bool)
        restarted = np.asarray(
            state.get("_exact_visibility_force_history_restarted", np.zeros(count)),
            dtype=bool,
        )
        if restarted.shape != (count,):
            failures.append(f"{label}: invalid gate force-history restart mask")
        else:
            # A switch seeds its new force at the boundary. A positive final
            # piece must use that one-sided derivative; an endpoint switch
            # only seeds the following step.
            expected[restarted] = (
                np.asarray(state["medina_external_force_sample_time"])[restarted]
                < np.asarray(state["t"])[restarted]
            )
        ready = np.asarray(
            state.get("medina_force_derivative_ready", np.zeros(count, dtype=bool)),
            dtype=bool,
        )
        if ready.shape != (count,) or not np.all(ready == expected):
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
        if "_exact_visibility_force_history_restarted" in midpoint_state:
            refined_ready = (
                _state_has_finite_medina_sample(midpoint_state)
                if medina_present and charged
                else None
            )
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
