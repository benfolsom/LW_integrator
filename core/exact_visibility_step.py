"""Split exact force steps at arriving ideal-gate boundary events."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .constants import C_MMNS
from .exact_pair_endpoint import (
    evaluate_exact_endpoint_four_potential,
    replace_exact_source_endpoint_potential,
)
from .exact_source_cloud import exact_cloud_history
from .exact_visibility_history import save_visibility_knots
from .exact_visibility import (
    boundary_visibility_overrides,
    save_boundary_visibility,
    selected_occluder,
    source_visibility_switches,
    visibility_history,
    visible_prepared,
)
from .potential_inclusion import LEDGER_FIELDS
from .retarded_fields import (
    ObserverEvent,
    RetardedHistoryError,
    _prepare_history,
    _solve_retarded_sample,
)
from .types import ParticleState


def _event(state: ParticleState, index: int) -> ObserverEvent:
    return ObserverEvent(
        float(state["t"][index]),
        tuple(float(state[axis][index]) for axis in "xyz"),
        float(state.get("source_time_low_ns", np.zeros(len(state["t"])))[index]),
        tuple(
            float(
                state.get(f"source_position_low_{axis}", np.zeros(len(state["t"])))[
                    index
                ]
            )
            for axis in "xyz"
        ),
        float(state.get("source_time_tail_ns", np.zeros(len(state["t"])))[index]),
        tuple(
            float(
                state.get(f"source_position_tail_{axis}", np.zeros(len(state["t"])))[
                    index
                ]
            )
            for axis in "xyz"
        ),
    )


def _sample(source: Any, event: ObserverEvent) -> Any:
    return _solve_retarded_sample(
        source,
        observer_time_ns=event.time_ns,
        observer_position_mm=np.asarray(event.position_mm),
        observer_time_low_ns=event.time_low_ns,
        observer_position_low_mm=np.asarray(event.position_low_mm),
        observer_time_tail_ns=event.time_tail_ns,
        observer_position_tail_mm=np.asarray(event.position_tail_mm),
        root_tolerance_mm=1.0e-21,
        max_root_iterations=96,
    )


def _arrival_residual(state: ParticleState, index: int, switch: Any) -> float:
    event = _event(state, index)
    position = (
        np.asarray(event.position_mm)
        + event.position_low_mm
        + np.asarray(event.position_tail_mm)
    )
    time = event.time_ns + event.time_low_ns + event.time_tail_ns
    return C_MMNS * (time - switch.time_ns) - np.linalg.norm(
        position - switch.position_mm
    )


def _arrival_roundoff(state: ParticleState, index: int, switch: Any) -> float:
    """Floating-point tie band, independent of the integration timestep."""
    position = np.array([state[axis][index] for axis in "xyz"])
    return (
        32.0
        * np.finfo(float).eps
        * max(
            np.linalg.norm(position),
            np.linalg.norm(switch.position_mm),
            np.linalg.norm(position - switch.position_mm),
            1.0,
        )
    )


def advance_exact_visibility_step(
    equation: Callable[..., ParticleState],
    parameters: dict[str, Any],
) -> ParticleState:
    """Advance smooth branches and rebase canonical P at the exact switch.

    Boundary emission times come from polynomial/aperture intersections, not
    changes in sampled masks. Their arriving light cones are solved against
    the actual numerical observer trial, rather than its coasting prediction.
    All observers split at the earliest proper-step fraction, preserving the
    existing bunch update and same-bunch force contract.
    """
    options = dict(parameters)
    options["_exact_visibility_internal"] = True
    options["_exact_trial_charge_fields"] = None
    source_history = options["exact_source_history"]
    if source_history is None:
        source_history = options["traj_ext_soa"]
    if source_history is None:
        source_history = options["trajectory_ext"][: options["index_traj"] + 1]
    magnetic = options["magnetic_dipole"]
    cloud = exact_cloud_history(source_history, magnetic.exact_charge_cloud)
    prepared = _prepare_history(cloud, ())
    dipole_offset = prepared.arrays.n_sources
    if magnetic.source.active:
        from dataclasses import replace
        from .retarded_dipole_fields import _prepare_dipole_history

        dipoles = _prepare_dipole_history(
            source_history,
            source_identities=None,
            observer_source_identity=None,
            excluded_source_identities=(),
            spin_interpolation_model=options["exact_source_spin_interpolation_model"],
        )
        prepared = replace(
            prepared,
            sources={
                **prepared.sources,
                **{
                    dipole_offset + i: source.worldline
                    for i, source in dipoles.sources.items()
                },
            },
        )
    geometry = options["beamline_geometry"]
    start = options["trajectory"][options["index_traj"]]
    count = len(start["x"])
    active = np.flatnonzero(
        ~np.asarray(
            start.get("_dead_particles", np.zeros(count, dtype=bool)), dtype=bool
        )
    )
    if options["_observer_trial_index"] is not None:
        active = np.asarray([options["_observer_trial_index"]])
    masks = {}
    for particle in active:
        direction = tuple(start[f"b{axis}"][particle] for axis in "xyz")
        selected = visible_prepared(
            prepared,
            visibility_history(cloud, geometry, direction),
            _event(start, particle),
            1.0e-21,
            96,
        )
        masks[int(particle)] = frozenset(selected.sources)
    for particle, saved in boundary_visibility_overrides(start).items():
        masks[particle] = saved | frozenset(
            i for i in masks.get(particle, ()) if i >= dipole_offset
        )
    for particle, saved in boundary_visibility_overrides(start, dipole=True).items():
        masks[particle] = frozenset(
            i for i in masks.get(particle, ()) if i < dipole_offset
        ) | frozenset(dipole_offset + i for i in saved)
    consumed = set()
    remaining = 1.0
    original_h = float(options["h"])
    original_steps = options["_particle_proper_steps_ns"]
    prefix = list(options["trajectory"][: options["index_traj"] + 1])
    integrated_diagnostics = {}
    restarted = np.zeros(count, dtype=bool)
    knots = []
    integral_fields = (
        "radiation_energy",
        "radiation_energy_applied",
        "radiation_reaction_work",
        "mass_shell_projection_energy",
        "medina_cross_field_energy_change",
    )

    def sector_masks(branch_masks):
        charge = {
            p: frozenset(i for i in indices if i < dipole_offset)
            for p, indices in branch_masks.items()
        }
        dipole = {
            p: frozenset(i - dipole_offset for i in indices if i >= dipole_offset)
            for p, indices in branch_masks.items()
        }
        return charge, dipole

    def publish_boundary(state):
        charge, dipole = sector_masks(masks)
        save_boundary_visibility(state, charge, dipole_masks=dipole)

    def finish(state):
        # These are interval integrals, while force samples and powers retain
        # their final-branch meaning. No canonical gate offset enters work.
        for name, total in integrated_diagnostics.items():
            state[name] += total
        if np.any(restarted):
            state["_exact_visibility_force_history_restarted"] = restarted.copy()
        else:
            state.pop("_exact_visibility_force_history_restarted", None)
        save_visibility_knots(state, knots)
        state.pop("_gate_prime_before", None)
        state.pop("_gate_prime_after", None)
        return state

    def advance(fraction):
        trial_options = dict(options)
        trial_options["h"] = original_h * fraction
        if original_steps is not None:
            trial_options["_particle_proper_steps_ns"] = original_steps * fraction
        charge_masks, dipole_masks = sector_masks(masks)
        trial_options["_exact_visibility_overrides"] = charge_masks
        trial_options["_exact_visibility_dipole_overrides"] = dipole_masks
        return equation(**trial_options)

    def branch_force(state, branch_masks, potential):
        """Seed a new branch with its own start force, never the ideal jump.

        The Taylor trial exposes the complete non-RR four-acceleration at
        its start. The first-order diagnostic uses a forward force secant.
        This probe is unpublished and cannot update accepted force history.
        """
        from .particle_state_copy import copy_particle_state

        probe_start = copy_particle_state(state)
        charge = probe_start.get("q_observer", probe_start["q"])
        for component, key in enumerate(("Pt", "Px", "Py", "Pz")):
            velocity = (
                np.ones(count)
                if component == 0
                else probe_start[f"b{'xyz'[component - 1]}"]
            )
            probe_start[key] = (
                probe_start["m"] * C_MMNS * probe_start["gamma"] * velocity
                + charge / C_MMNS * potential[:, component]
            )
        probe_options = dict(options)
        charge_masks, dipole_masks = sector_masks(branch_masks)
        probe_prefix = (
            prefix[:-1] + [probe_start]
            if np.array_equal(prefix[-1]["t"], probe_start["t"])
            else prefix + [probe_start]
        )
        probe_options.update(
            trajectory=probe_prefix,
            index_traj=len(probe_prefix) - 1,
            traj_soa=None,
            h=original_h / 8,
            radiation_reaction_mode="off",
            _exact_visibility_overrides=charge_masks,
            _exact_visibility_dipole_overrides=dipole_masks,
            _skip_exact_endpoint_field_diagnostic=True,
        )
        if original_steps is not None:
            probe_options["_particle_proper_steps_ns"] = original_steps / 8
        probe = equation(**probe_options)
        if "_intrinsic_spin_start_non_self_four_acceleration" in probe:
            acceleration = probe["_intrinsic_spin_start_non_self_four_acceleration"]
            force = state["m"][:, None] * acceleration[:, 1:] / state["gamma"][:, None]
            prime = np.column_stack(
                [probe[f"source_start_beta_prime_{axis}_per_mm"] for axis in "xyz"]
            )
        else:
            dt = probe["t"] - state["t"]
            inactive = np.ones(count, dtype=bool)
            inactive[active] = False
            dt[inactive] = 1.0
            before = state["gamma"][:, None] * np.column_stack(
                [state[f"b{a}"] for a in "xyz"]
            )
            after = probe["gamma"][:, None] * np.column_stack(
                [probe[f"b{a}"] for a in "xyz"]
            )
            force = state["m"][:, None] * C_MMNS * (after - before) / dt[:, None]
            prime = np.column_stack(
                [(probe[f"b{a}"] - state[f"b{a}"]) / (C_MMNS * dt) for a in "xyz"]
            )
        if not np.all(np.isfinite(force[active])) or not np.all(
            np.isfinite(prime[active])
        ):
            raise ValueError("gate branch force and acceleration must be finite")
        inactive = np.ones(count, dtype=bool)
        inactive[active] = False
        force[inactive] = 0.0
        prime[inactive] = 0.0
        return force, prime

    def check_endpoint_masks(state):
        # A changing source-history reconstruction must not silently change
        # the next step's branch. An exact boundary permits its saved side.
        for particle in active:
            direction = tuple(state[f"b{a}"][particle] for a in "xyz")
            actual = visible_prepared(
                prepared,
                visibility_history(cloud, geometry, direction),
                _event(state, particle),
                1.0e-21,
                96,
            )
            for index in masks[int(particle)] ^ frozenset(actual.sources):
                source = prepared.sources[index]
                sample = _sample(source, _event(state, particle))
                padding = 64 * np.finfo(float).eps * max(1.0, abs(sample.time_ns))
                switches = source_visibility_switches(
                    source,
                    index,
                    geometry,
                    direction,
                    sample.time_ns - padding,
                    sample.time_ns + padding,
                )
                if not any(
                    abs(_arrival_residual(state, particle, switch))
                    <= _arrival_roundoff(state, particle, switch)
                    for switch in switches
                ):
                    raise RetardedHistoryError(
                        "exact visibility mask changed without a located gate event "
                        "and potential ledger entry"
                    )

    for _ in range(256):
        end = advance(remaining)
        candidates = []
        for particle in active:
            direction = tuple(start[f"b{axis}"][particle] for axis in "xyz")
            endpoint_direction = tuple(end[f"b{axis}"][particle] for axis in "xyz")
            if selected_occluder(geometry, direction) is not selected_occluder(
                geometry, endpoint_direction
            ):
                raise NotImplementedError(
                    "Exact visibility event splitting requires a fixed observer "
                    "pipe during a force step; a directional pipe change needs "
                    "an observer-direction boundary event"
                )
            beginning, ending = _event(start, particle), _event(end, particle)
            for index, source in prepared.sources.items():
                first, last = _sample(source, beginning), _sample(source, ending)
                if first is None or last is None:
                    raise RetardedHistoryError(
                        "Exact visibility event search needs complete source "
                        "history at both observer endpoints; source continuation "
                        "requires matching gate boundary events"
                    )
                # An event exactly on a public endpoint can fall one ulp
                # outside either independently solved retarded-time bracket.
                # Include only a floating-point tie band, never a timestep band.
                padding = (
                    32.0
                    * np.finfo(float).eps
                    * max(
                        abs(first.time_ns),
                        abs(last.time_ns),
                        abs(ending.time_ns),
                        np.linalg.norm(ending.position_mm) / C_MMNS,
                    )
                )
                for switch in source_visibility_switches(
                    source,
                    index,
                    geometry,
                    direction,
                    first.time_ns - padding,
                    last.time_ns + padding,
                ):
                    key = (int(particle), index, switch.time_ns)
                    changes = (index in masks[int(particle)]) != switch.visible_after
                    if (
                        changes
                        and key not in consumed
                        and _arrival_residual(end, particle, switch)
                        >= -_arrival_roundoff(end, particle, switch)
                    ):
                        candidates.append((int(particle), switch, key))
        if not candidates:
            end.pop("_exact_visibility_endpoint_overrides", None)
            check_endpoint_masks(end)
            if consumed or boundary_visibility_overrides(start):
                publish_boundary(end)
            return finish(end)

        roots = []
        for particle, switch, key in candidates:
            lower, upper = 0.0, remaining
            # Timelike observer motion makes arrival residual monotonic. Solve
            # on the numerical branch, retaining an explicit post-switch mask.
            for _ in range(52):
                middle = 0.5 * (lower + upper)
                trial = advance(middle)
                if _arrival_residual(trial, particle, switch) < 0.0:
                    lower = middle
                else:
                    upper = middle
                if upper - lower <= 8.0 * np.finfo(float).eps * remaining:
                    break
            roots.append((0.5 * (lower + upper), particle, switch, key))
        fraction = min(item[0] for item in roots)
        zero_step = fraction <= 16.0 * np.finfo(float).eps * remaining
        if zero_step:
            from .particle_state_copy import copy_particle_state

            fraction = 0.0
            at_switch = copy_particle_state(start)
            required = np.zeros(count, bool)
            required[active] = True
            at_switch["_exact_source_endpoint_rebase_required"] = required
            for name in LEDGER_FIELDS:
                at_switch[name] = np.array(start.get(name, np.zeros(count)), copy=True)
            for name in integral_fields:
                if name in at_switch:
                    at_switch[name] = np.zeros(count)
        else:
            at_switch = advance(fraction)
        after_masks = dict(masks)
        for root, particle, switch, key in roots:
            if abs(root - fraction) <= 16.0 * np.finfo(float).eps * remaining:
                selected = set(after_masks[particle])
                if switch.visible_after:
                    selected.add(switch.source_index)
                else:
                    selected.discard(switch.source_index)
                after_masks[particle] = frozenset(selected)
                consumed.add(key)
        # Both branches use the same observer event and source chronology.
        common = dict(
            magnetic_dipole=magnetic,
            include_dipole_source=magnetic.source.active,
            beamline_geometry=geometry,
            own_history=(
                (prefix if zero_step else prefix + [at_switch])
                if options["space_charge"] is not None
                and options["space_charge"].enabled
                else None
            ),
            spin_interpolation_model=options["exact_source_spin_interpolation_model"],
            own_bunch_ranges=options["exact_same_bunch_ranges"],
        )
        charge_before, dipole_before = sector_masks(masks)
        before_A = evaluate_exact_endpoint_four_potential(
            at_switch,
            source_history,
            visibility_overrides=charge_before,
            dipole_visibility_overrides=dipole_before,
            **common,
        )
        if zero_step:
            at_switch["_exact_source_start_four_potential"] = before_A
        charge_after, dipole_after = sector_masks(after_masks)
        after_A = evaluate_exact_endpoint_four_potential(
            at_switch,
            source_history,
            visibility_overrides=charge_after,
            dipole_visibility_overrides=dipole_after,
            **common,
        )
        _, prime_before = branch_force(at_switch, masks, before_A)
        force_after, prime_after = branch_force(at_switch, after_masks, after_A)
        required = at_switch["_exact_source_endpoint_rebase_required"].copy()
        replace_exact_source_endpoint_potential(at_switch, after_A)
        for component, name in enumerate(LEDGER_FIELDS):
            axis = (component + 1) % 4  # ledger order x,y,z,t
            at_switch[name] += after_A[:, axis] - before_A[:, axis]
        # Seed the post-switch force at the boundary. The next midpoint then
        # supplies a one-sided derivative and applies RR within this step.
        changed = [p for p in after_masks if after_masks[p] != masks[p]]
        restarted[changed] = True
        if "medina_external_force_sample_time" in at_switch:
            for column, axis in enumerate("xyz"):
                at_switch[f"medina_external_force_{axis}"][changed] = force_after[
                    changed, column
                ]
            at_switch["medina_external_force_sample_time"][changed] = at_switch["t"][
                changed
            ]
            at_switch["medina_force_derivative_ready"][changed] = False
        at_switch["_gate_prime_before"] = prime_before
        at_switch["_gate_prime_after"] = prime_after
        knots.append(at_switch)
        masks = after_masks
        remaining -= fraction
        if remaining <= 16.0 * np.finfo(float).eps:
            at_switch["_exact_source_start_four_potential"] = after_A
            at_switch["_exact_source_endpoint_rebase_required"] = required
            at_switch["_exact_visibility_endpoint_overrides"] = charge_after
            publish_boundary(at_switch)
            check_endpoint_masks(at_switch)
            return finish(at_switch)
        for name in integral_fields:
            if name in at_switch:
                integrated_diagnostics.setdefault(name, np.zeros(count))
                integrated_diagnostics[name] += at_switch[name]
        start = at_switch
        if zero_step:
            prefix[-1] = start
        else:
            prefix.append(start)
        options["trajectory"] = prefix
        options["index_traj"] = len(prefix) - 1
        options["traj_soa"] = None
    raise RuntimeError("more than 256 exact visibility switches in one force step")
