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
from .retarded_fields import ObserverEvent
from .types import MagneticDipoleConfig, ParticleState


def evaluate_exact_endpoint_four_potential(
    observer_state: ParticleState,
    source_history: Any,
    *,
    magnetic_dipole: MagneticDipoleConfig,
    include_dipole_source: bool,
    spin_interpolation_model: str = "centered_c1",
) -> np.ndarray:
    """Evaluate ``A_charge + A_dipole`` at provisional observer endpoints."""

    from .retarded_fields import evaluate_retarded_charge_field_native

    if include_dipole_source:
        from .retarded_dipole_fields import (
            evaluate_retarded_dipole_potential_native,
        )

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
    charge_root_tolerance_mm = (
        float(source_options.root_tolerance_mm) if include_dipole_source else 1.0e-21
    )
    charge_max_root_iterations = (
        int(source_options.max_root_iterations) if include_dipole_source else 96
    )
    for particle_idx in np.flatnonzero(required):
        event = ObserverEvent(
            time_ns=float(observer_state["t"][particle_idx]),
            position_mm=(
                float(observer_state["x"][particle_idx]),
                float(observer_state["y"][particle_idx]),
                float(observer_state["z"][particle_idx]),
            ),
        )
        charge_field = evaluate_retarded_charge_field_native(
            source_history,
            event,
            require_complete_history=True,
            root_tolerance_mm=charge_root_tolerance_mm,
            max_root_iterations=charge_max_root_iterations,
            backend=magnetic_dipole.exact_retarded_backend,
        )
        potentials[particle_idx] += charge_field.four_potential
        if include_dipole_source:
            dipole_potential = evaluate_retarded_dipole_potential_native(
                source_history,
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
    spin_interpolation_model: str = "centered_c1",
) -> tuple[ParticleState, ParticleState]:
    """Return detached endpoint-canonical states without publishing either row.

    Both endpoint potentials are evaluated from the same provisional pair
    histories before either canonical state changes.  This retains the Jacobi
    pair ordering and makes a failed/rejected adaptive trial side-effect free.
    """

    rider = copy.deepcopy(rider_state)
    driver = copy.deepcopy(driver_state)
    rider_endpoint = evaluate_exact_endpoint_four_potential(
        rider,
        driver_endpoint_history,
        magnetic_dipole=magnetic_dipole,
        include_dipole_source=include_dipole_source,
        spin_interpolation_model=spin_interpolation_model,
    )
    driver_endpoint = evaluate_exact_endpoint_four_potential(
        driver,
        rider_endpoint_history,
        magnetic_dipole=magnetic_dipole,
        include_dipole_source=include_dipole_source,
        spin_interpolation_model=spin_interpolation_model,
    )
    replace_exact_source_endpoint_potential(rider, rider_endpoint)
    replace_exact_source_endpoint_potential(driver, driver_endpoint)
    return rider, driver


__all__ = [
    "discard_exact_source_endpoint_scratch",
    "evaluate_exact_endpoint_four_potential",
    "finalize_exact_source_canonical_pair_states",
    "replace_exact_source_endpoint_potential",
]
