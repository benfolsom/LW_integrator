"""Pure accepted-endpoint canonical recomposition for exact source pairs.

Exact charge and dipole forces advance mechanical momentum.  During a trial,
the equations temporarily retain the start-event ordinary potential offset so
that the existing state representation remains canonical.  Once both
provisional endpoints are available, this module evaluates both endpoint
potentials before changing either state and replaces the two start offsets.

The helpers do not publish trajectory rows.  Production fixed-step code may
write the returned states into builders, while adaptive trial code may keep
them in immutable overlays until a joint acceptance decision is made.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .canonical_momentum import replace_canonical_potential_native
from .retarded_fields import ObserverEvent, RetardedHistoryError
from .types import MagneticDipoleConfig, ParticleState


def evaluate_exact_endpoint_four_potential(
    observer_state: ParticleState,
    source_history: Any,
    *,
    magnetic_dipole: MagneticDipoleConfig,
    include_dipole_source: bool,
    own_history: Any = None,
    require_complete_history: bool = True,
    dipole_source_collection: Any = None,
    spin_interpolation_model: str = "centered_c1",
    cross_extrapolate_ns: float = 0.0,
    beamline_geometry: Any = None,
    visibility_overrides: dict[int, frozenset[int]] | None = None,
    dipole_visibility_overrides: dict[int, frozenset[int]] | None = None,
    own_bunch_ranges: tuple[slice, ...] = (),
) -> np.ndarray:
    """Evaluate cross-bunch and non-self own-bunch potentials at endpoints.

    ``cross_extrapolate_ns`` is a fallback for the cross-bunch sources: the two
    bunches advance in proper time, so their final lab times differ; when an
    observer is closer to a source than c times that spread, the source's
    retarded point lies after its last accepted knot and its last segment is
    continued with the selected strict charge provider instead of failing.
    """

    from .retarded_fields import evaluate_retarded_charge_field_native
    from .exact_source_cloud import exact_cloud_history, same_bunch_exclusions
    from .exact_visibility import boundary_visibility_overrides

    if visibility_overrides is None:
        visibility_overrides = boundary_visibility_overrides(observer_state)
    if dipole_visibility_overrides is None:
        dipole_visibility_overrides = boundary_visibility_overrides(
            observer_state, dipole=True
        )

    dipole_history = source_history
    source_history = exact_cloud_history(
        source_history, magnetic_dipole.exact_charge_cloud
    )
    if own_history is not None:
        own_history = exact_cloud_history(
            own_history, magnetic_dipole.exact_charge_cloud
        )

    if include_dipole_source and dipole_source_collection is None:
        from .retarded_dipole_fields import (
            evaluate_retarded_dipole_potential_native,
        )
    if dipole_source_collection is not None and not include_dipole_source:
        raise ValueError("causal dipole history requires an active dipole source")

    particle_count = len(np.asarray(observer_state.get("x", [])))
    potentials: np.ndarray = np.zeros((particle_count, 4), dtype=float)
    required = np.asarray(
        observer_state.get(
            "_exact_source_endpoint_rebase_required",
            np.zeros(particle_count, dtype=bool),
        ),
        dtype=bool,
    )
    if required.shape != (particle_count,):
        raise ValueError("exact endpoint rebase mask must match particle count")

    source_options = magnetic_dipole.source
    continuation_backend = (
        "python"
        if magnetic_dipole.exact_retarded_backend == "python"
        else "numba_full_strict_serial"
    )
    charge_root_tolerance_mm = (
        float(source_options.root_tolerance_mm) if include_dipole_source else 1.0e-21
    )
    charge_max_root_iterations = (
        int(source_options.max_root_iterations) if include_dipole_source else 96
    )
    for particle_idx in np.flatnonzero(required):
        from .exact_visibility import visibility_history

        visible_history = visibility_history(
            source_history,
            beamline_geometry,
            tuple(
                observer_state.get(f"b{axis}", np.zeros(particle_count))[particle_idx]
                for axis in "xyz"
            ),
            (
                None
                if visibility_overrides is None
                else visibility_overrides.get(int(particle_idx))
            ),
        )
        visible_dipole_history = visibility_history(
            dipole_history,
            beamline_geometry,
            tuple(
                observer_state.get(f"b{axis}", np.zeros(particle_count))[particle_idx]
                for axis in "xyz"
            ),
            dipole_visibility_overrides.get(int(particle_idx)),
        )
        event = ObserverEvent(
            time_ns=float(observer_state["t"][particle_idx]),
            position_mm=(
                float(observer_state["x"][particle_idx]),
                float(observer_state["y"][particle_idx]),
                float(observer_state["z"][particle_idx]),
            ),
            time_low_ns=float(
                observer_state.get("source_time_low_ns", np.zeros(particle_count))[
                    particle_idx
                ]
            ),
            position_low_mm=tuple(
                float(
                    observer_state.get(
                        f"source_position_low_{axis}", np.zeros(particle_count)
                    )[particle_idx]
                )
                for axis in "xyz"
            ),
            time_tail_ns=float(
                observer_state.get("source_time_tail_ns", np.zeros(particle_count))[
                    particle_idx
                ]
            ),
            position_tail_mm=tuple(
                float(
                    observer_state.get(
                        f"source_position_tail_{axis}", np.zeros(particle_count)
                    )[particle_idx]
                )
                for axis in "xyz"
            ),
        )
        try:
            charge_field = evaluate_retarded_charge_field_native(
                visible_history,
                event,
                require_complete_history=require_complete_history,
                root_tolerance_mm=charge_root_tolerance_mm,
                max_root_iterations=charge_max_root_iterations,
                backend=magnetic_dipole.exact_retarded_backend,
            )
        except RetardedHistoryError:
            if cross_extrapolate_ns <= 0.0:
                raise
            charge_field = evaluate_retarded_charge_field_native(
                visible_history,
                event,
                require_complete_history=require_complete_history,
                root_tolerance_mm=charge_root_tolerance_mm,
                max_root_iterations=charge_max_root_iterations,
                backend=continuation_backend,
                extrapolate_ns=cross_extrapolate_ns,
            )
        potentials[particle_idx] += charge_field.four_potential
        if own_history is not None and particle_count > 1:
            # Bunch-mates advance in proper time, so their latest lab times can
            # differ slightly; continue their last segment across that spread.
            own_times = np.asarray(observer_state["t"], dtype=float)
            own_field = evaluate_retarded_charge_field_native(
                own_history,
                event,
                excluded_source_indices=same_bunch_exclusions(
                    particle_idx, particle_count, own_bunch_ranges
                ),
                require_complete_history=require_complete_history,
                root_tolerance_mm=charge_root_tolerance_mm,
                max_root_iterations=charge_max_root_iterations,
                backend=continuation_backend,
                extrapolate_ns=2.0 * float(np.ptp(own_times)),
            )
            potentials[particle_idx] += own_field.four_potential
        if include_dipole_source:
            if dipole_source_collection is not None:
                if source_options.history_model == "causal_c5":
                    from .causal_c5_dipole_provider import (
                        evaluate_causal_c5_dipole_source_collection_native,
                    )

                    dipole_response = (
                        evaluate_causal_c5_dipole_source_collection_native(
                            dipole_source_collection,
                            event,
                            minimum_separation_mm=float(
                                source_options.minimum_separation_mm
                            ),
                            root_tolerance_mm=float(source_options.root_tolerance_mm),
                            max_root_iterations=int(source_options.max_root_iterations),
                        )
                    )
                elif source_options.history_model == "causal_local_jet":
                    from .causal_local_source_jet import (
                        evaluate_configured_causal_local_source_jet_collection_native,
                    )

                    dipole_response = (
                        evaluate_configured_causal_local_source_jet_collection_native(
                            dipole_source_collection,
                            event,
                            source_options=source_options,
                        )
                    )
                else:
                    raise ValueError(
                        "an independent exact dipole history requires causal_c5 or "
                        "causal_local_jet selection"
                    )
                potentials[particle_idx] += dipole_response.four_potential
            elif (
                magnetic_dipole.exact_retarded_backend
                == "numba_analytic_charge_dipole_response_serial"
            ):
                # The accepted endpoint and the following step start must use
                # the same ordinary potential representation.  Mixing this
                # analytical A with the nine-event finite-difference endpoint
                # A creates a q*Delta(A)/c mechanical-momentum jump whenever
                # an adaptive step is split into two accepted half steps.
                from .dipole_hertz_jet import (
                    evaluate_retarded_dipole_field_gradient_hertz_jet_native,
                )

                dipole_response = (
                    evaluate_retarded_dipole_field_gradient_hertz_jet_native(
                        visible_dipole_history,
                        event,
                        require_complete_history=True,
                        fallback_relative_step=float(
                            source_options.relative_stencil_step
                        ),
                        fallback_minimum_step_mm=float(
                            source_options.minimum_stencil_step_mm
                        ),
                        minimum_separation_mm=float(
                            source_options.minimum_separation_mm
                        ),
                        root_tolerance_mm=float(source_options.root_tolerance_mm),
                        max_root_iterations=int(source_options.max_root_iterations),
                        response_kernel="numba_sparse_strict_serial",
                        fallback_backend="numba_full_strict_serial",
                        spin_interpolation_model=spin_interpolation_model,
                    )
                )
                potentials[particle_idx] += dipole_response.response.four_potential
            else:
                dipole_potential = evaluate_retarded_dipole_potential_native(
                    visible_dipole_history,
                    event,
                    require_complete_history=True,
                    relative_step=float(source_options.relative_stencil_step),
                    minimum_step_mm=float(source_options.minimum_stencil_step_mm),
                    minimum_separation_mm=float(source_options.minimum_separation_mm),
                    root_tolerance_mm=float(source_options.root_tolerance_mm),
                    max_root_iterations=int(source_options.max_root_iterations),
                    backend=magnetic_dipole.exact_retarded_backend,
                    spin_interpolation_model=spin_interpolation_model,
                )
                potentials[particle_idx] += dipole_potential.four_potential
    if not np.all(np.isfinite(potentials)):
        raise ValueError("exact endpoint four-potential must be finite")
    return potentials


def replace_exact_source_endpoint_potential(
    state: ParticleState,
    endpoint_four_potential: np.ndarray,
) -> None:
    """Replace the saved start-event ``qA/c`` offset by the endpoint offset."""

    particle_count = len(np.asarray(state.get("x", [])))
    start = np.asarray(
        state.get("_exact_source_start_four_potential", np.empty((0, 4))),
        dtype=float,
    )
    required = np.asarray(
        state.get("_exact_source_endpoint_rebase_required", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    endpoint = np.asarray(endpoint_four_potential, dtype=float)
    if start.shape != (particle_count, 4):
        raise ValueError("exact start four-potential must have shape [particles, 4]")
    if required.shape != (particle_count,):
        raise ValueError("exact endpoint rebase mask must match particle count")
    if endpoint.shape != (particle_count, 4):
        raise ValueError("exact endpoint four-potential must have shape [particles, 4]")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(endpoint)):
        raise ValueError("exact canonical endpoint potentials must be finite")

    charges = np.asarray(
        state.get("q_observer", state.get("q", np.zeros(particle_count))),
        dtype=float,
    )
    if charges.shape != (particle_count,) or not np.all(np.isfinite(charges)):
        raise ValueError("observer charge must be finite and match particle count")
    component_keys = ("Pt", "Px", "Py", "Pz")
    for particle_idx in np.flatnonzero(required):
        temporary = np.asarray(
            [state[key][particle_idx] for key in component_keys], dtype=float
        )
        finalized = replace_canonical_potential_native(
            temporary,
            start[particle_idx],
            endpoint[particle_idx],
            charge_native=float(charges[particle_idx]),
        )
        for component_index, key in enumerate(component_keys):
            state[key][particle_idx] = finalized[component_index]

    discard_exact_source_endpoint_scratch(state)


def discard_exact_source_endpoint_scratch(state: ParticleState) -> None:
    """Remove private endpoint handoff data from a state."""

    state.pop("_exact_source_start_four_potential", None)
    state.pop("_exact_source_endpoint_rebase_required", None)


def finalize_exact_source_canonical_pair_states(
    *,
    rider_state: ParticleState,
    driver_state: ParticleState,
    rider_endpoint_history: Any,
    driver_endpoint_history: Any,
    magnetic_dipole: MagneticDipoleConfig,
    include_dipole_source: bool,
    same_bunch_fields: bool = False,
    require_complete_history: bool = True,
    rider_dipole_source_collection: Any = None,
    driver_dipole_source_collection: Any = None,
    spin_interpolation_model: str = "centered_c1",
    beamline_geometry: Any = None,
    driver_bunch_ranges: tuple[slice, ...] = (),
) -> tuple[ParticleState, ParticleState]:
    """Return detached endpoint-canonical states without publishing either row.

    Both endpoint potentials are evaluated from the same provisional pair
    histories before either canonical state changes.  This retains the Jacobi
    pair ordering and makes a failed/rejected adaptive trial side-effect free.
    """

    rider = copy.deepcopy(rider_state)
    driver = copy.deepcopy(driver_state)
    endpoint_times = np.concatenate(
        (
            np.asarray(rider.get("t", []), dtype=float),
            np.asarray(driver.get("t", []), dtype=float),
        )
    )
    cross_spread = float(np.ptp(endpoint_times)) if endpoint_times.size else 0.0
    rider_endpoint = evaluate_exact_endpoint_four_potential(
        rider,
        driver_endpoint_history,
        magnetic_dipole=magnetic_dipole,
        include_dipole_source=include_dipole_source,
        own_history=rider_endpoint_history if same_bunch_fields else None,
        require_complete_history=require_complete_history,
        dipole_source_collection=driver_dipole_source_collection,
        spin_interpolation_model=spin_interpolation_model,
        cross_extrapolate_ns=2.0 * cross_spread,
        beamline_geometry=beamline_geometry,
    )
    driver_endpoint = evaluate_exact_endpoint_four_potential(
        driver,
        rider_endpoint_history,
        magnetic_dipole=magnetic_dipole,
        include_dipole_source=include_dipole_source,
        own_history=driver_endpoint_history if same_bunch_fields else None,
        require_complete_history=require_complete_history,
        dipole_source_collection=rider_dipole_source_collection,
        spin_interpolation_model=spin_interpolation_model,
        cross_extrapolate_ns=2.0 * cross_spread,
        beamline_geometry=beamline_geometry,
        own_bunch_ranges=driver_bunch_ranges,
    )
    replace_exact_source_endpoint_potential(rider, rider_endpoint)
    replace_exact_source_endpoint_potential(driver, driver_endpoint)
    rider.pop("_exact_visibility_endpoint_overrides", None)
    driver.pop("_exact_visibility_endpoint_overrides", None)
    return rider, driver


__all__ = [
    "discard_exact_source_endpoint_scratch",
    "evaluate_exact_endpoint_four_potential",
    "finalize_exact_source_canonical_pair_states",
    "replace_exact_source_endpoint_potential",
]
