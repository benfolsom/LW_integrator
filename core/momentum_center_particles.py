"""Many-particle full-spin stepping and opt-in subset scheduling.

Checkpoint index is persistent particle identity. All particles advance against
the same accepted past; histories publish together after successful steps.
"""

import copy

import numpy as np

from .momentum_center_pair import (
    MULTIPARTICLE_MODEL,
    _advance_particles,
    _initialize_particles,
)

DEFAULT_PASSIVE_CONSTRAINT_BUDGET_RELATIVE = 1e-6


def initialize_particles(particles, states, histories, **options):
    """Prepare two or more particles using the pair solver's physical options."""
    return _initialize_particles(
        particles, states, histories, checkpoint_model=MULTIPARTICLE_MODEL, **options
    )


def _physical_schedule(payload, spacing_ns):
    if (
        not np.isscalar(spacing_ns)
        or isinstance(spacing_ns, (bool, str))
        or not np.isfinite(spacing_ns)
        or spacing_ns <= 0
    ):
        raise ValueError("Positive finite pseudogrid selection spacing required")
    states = np.asarray(payload["states"], dtype=float)
    if states.ndim != 2 or states.shape[1] != 14 or not np.isfinite(states).all():
        raise ValueError("Finite many-particle states required")
    return dict(
        selection_spacing_ns=float(spacing_ns),
        selection_origin_ns=float(states[0, 0]),
        selection_count=0,
        active_indices=[],
    )


def configure_pseudogrid_reference(payload, active_count, selection_spacing_ns=None):
    """Record subset selections while retaining exact dynamics for everyone.

    This mode offers no speedup. Passive spin/history approximation is not yet
    enabled; this establishes its exact reference and checkpointed schedule.
    """
    _validate_model(payload)
    count = len(payload["particles"])
    if type(active_count) is not int or not 1 <= active_count <= count:
        raise ValueError("Active count must be an integer from one to particle count")
    if "pseudogrid_reference" in payload or "pseudogrid_reduced" in payload:
        raise ValueError("Resume preserves the existing pseudogrid schedule")
    result = copy.deepcopy(payload)
    result["pseudogrid_reference"] = dict(
        mode="exact_reference",
        active_count=active_count,
        last_active_step=[-1] * count,
        activation_count=[0] * count,
    )
    if selection_spacing_ns is not None:
        result["pseudogrid_reference"].update(
            _physical_schedule(payload, selection_spacing_ns)
        )
    return result


def _configure_reduced(
    payload,
    active_count,
    selection_spacing_ns,
    constraint_budget_relative,
    mode,
):
    """Use RK4 for selected particles and a lower-order passive update.

    Every particle remains a distinct retarded source; the passive accepted
    motion and moment are approximate. This is an opt-in reduced-motion
    experiment, not a source-reduction method.
    """
    _validate_model(payload)
    count = len(payload["particles"])
    if type(active_count) is not int or not 1 <= active_count < count:
        raise ValueError(
            "Reduced active count must be from one to particle count minus one"
        )
    if "pseudogrid_reference" in payload or "pseudogrid_reduced" in payload:
        raise ValueError("Resume preserves the existing pseudogrid schedule")
    if (
        payload.get("integration_method", "rk4") != "rk4"
        or "source_precision" in payload
        or "internal_step_control" in payload
        or any(p.get("reaction_mode") != "off" for p in payload["particles"])
    ):
        raise ValueError(
            "Reduced passive updates require reaction-off, unpreserved RK4 without internal steps"
        )
    if (
        not np.isscalar(constraint_budget_relative)
        or isinstance(constraint_budget_relative, (bool, str))
        or not np.isfinite(constraint_budget_relative)
        or constraint_budget_relative <= 0
    ):
        raise ValueError("Positive finite relative constraint budget required")
    result = copy.deepcopy(payload)
    result["pseudogrid_reduced"] = dict(
        mode=mode,
        active_count=active_count,
        last_active_step=[-1] * count,
        activation_count=[0] * count,
        constraint_budget_relative=float(constraint_budget_relative),
        fallback_count=[0] * count,
        reduced_accepted_count=[0] * count,
        **_physical_schedule(payload, selection_spacing_ns),
    )
    return result


def configure_pseudogrid_midpoint(
    payload,
    active_count,
    selection_spacing_ns,
    constraint_budget_relative=DEFAULT_PASSIVE_CONSTRAINT_BUDGET_RELATIVE,
):
    """Configure second-order passive motion with a recorded RK4 safety fallback."""
    return _configure_reduced(
        payload,
        active_count,
        selection_spacing_ns,
        constraint_budget_relative,
        "passive_midpoint",
    )


def configure_pseudogrid_rk3(
    payload,
    active_count,
    selection_spacing_ns,
    constraint_budget_relative=DEFAULT_PASSIVE_CONSTRAINT_BUDGET_RELATIVE,
):
    """Configure third-order passive motion with a recorded RK4 safety fallback."""
    return _configure_reduced(
        payload,
        active_count,
        selection_spacing_ns,
        constraint_budget_relative,
        "passive_rk3",
    )


def _validate_model(payload):
    if payload.get("model") != MULTIPARTICLE_MODEL:
        raise ValueError("A many-particle full-spin checkpoint is required")
    if "pseudo_grid" in payload:
        raise ValueError("Legacy reduced pseudogrid is not supported by this solver")


def _schedule(config, states, step, mode):
    from .pseudo_grid import select_active_indices

    config = copy.deepcopy(config)
    count = len(states)
    base_keys = {"mode", "active_count", "last_active_step", "activation_count"}
    clock_keys = {
        "selection_spacing_ns",
        "selection_origin_ns",
        "selection_count",
        "active_indices",
    }
    guard_keys = {
        "constraint_budget_relative",
        "fallback_count",
        "reduced_accepted_count",
    }
    clocked = isinstance(config, dict) and clock_keys <= set(config)
    guarded = isinstance(config, dict) and guard_keys <= set(config)
    if (
        not isinstance(config, dict)
        or set(config)
        != (
            base_keys
            | (clock_keys if clocked else set())
            | (guard_keys if guarded else set())
        )
        or config["mode"] != mode
        or type(config["active_count"]) is not int
        or not 1
        <= config["active_count"]
        <= (count if mode == "exact_reference" else count - 1)
        or (mode in ("passive_midpoint", "passive_rk3") and not clocked)
    ):
        raise ValueError("Invalid pseudogrid schedule configuration")
    if type(step) is not int or step < 0:
        raise ValueError("Invalid accepted step count")
    for key, lower, upper in (
        ("last_active_step", -1, step - 1),
        ("activation_count", 0, step),
    ):
        values = config[key]
        if (
            not isinstance(values, list)
            or len(values) != count
            or any(
                type(value) is not int or not lower <= value <= upper
                for value in values
            )
        ):
            raise ValueError("Invalid pseudogrid reference schedule counters")
    states = np.asarray(states, dtype=float)
    if states.shape != (count, 14) or not np.isfinite(states).all():
        raise ValueError("Invalid many-particle states")
    if guarded:
        budget = config["constraint_budget_relative"]
        if (
            not np.isscalar(budget)
            or isinstance(budget, (bool, str))
            or not np.isfinite(budget)
            or budget <= 0
            or any(
                not isinstance(config[key], list)
                or len(config[key]) != count
                or any(
                    type(value) is not int or not 0 <= value <= step
                    for value in config[key]
                )
                for key in ("fallback_count", "reduced_accepted_count")
            )
        ):
            raise ValueError("Invalid passive constraint policy")
    if clocked:
        spacing = config["selection_spacing_ns"]
        origin = config["selection_origin_ns"]
        selections = config["selection_count"]
        stored = config["active_indices"]
        if (
            not np.isscalar(spacing)
            or isinstance(spacing, (bool, str))
            or not np.isfinite(spacing)
            or spacing <= 0
            or not np.isscalar(origin)
            or isinstance(origin, (bool, str))
            or not np.isfinite(origin)
            or type(selections) is not int
            or not 0 <= selections <= step
            or not isinstance(stored, list)
            or len(stored) != (config["active_count"] if selections else 0)
            or any(type(i) is not int or not 0 <= i < count for i in stored)
            or len(set(stored)) != len(stored)
            or sum(config["activation_count"]) != config["active_count"] * selections
        ):
            raise ValueError("Invalid physical pseudogrid selection clock")
        next_time = origin + selections * spacing
        tolerance = (
            64 * np.finfo(float).eps * max(abs(next_time), abs(states[0, 0]), 1.0)
        )
        if states[0, 0] + tolerance < origin:
            raise ValueError("Pseudogrid selection origin is after the accepted state")
        if states[0, 0] + tolerance < next_time:
            return config, stored, False
    active = select_active_indices(
        {axis: states[:, index + 1] for index, axis in enumerate(("x", "y", "z"))},
        np.arange(count),
        active_count=config["active_count"],
        step_index=step,
        last_active_step=np.asarray(config["last_active_step"]),
        activation_count=np.asarray(config["activation_count"]),
    )
    for index in active:
        config["last_active_step"][index] = step
        config["activation_count"][index] += 1
    if clocked:
        config["selection_count"] += 1
        config["active_indices"] = active.tolist()
    return config, active.tolist(), True


def advance_particles(payload, width_ns, steps=1):
    """Advance on the shared recording clock with optional subset selection."""
    _validate_model(payload)
    reference = "pseudogrid_reference" in payload
    reduced = "pseudogrid_reduced" in payload
    if reference and reduced:
        raise ValueError("Only one pseudogrid schedule may be present")
    if not reference and not reduced:
        return _advance_particles(payload, width_ns, steps)
    if type(steps) is not int or steps < 1:
        raise ValueError("Positive integer step count required")
    field = "pseudogrid_reference" if reference else "pseudogrid_reduced"
    config = payload[field]
    mode = (
        "exact_reference"
        if reference
        else (config.get("mode") if isinstance(config, dict) else None)
    )
    if mode not in ("exact_reference", "passive_midpoint", "passive_rk3"):
        raise ValueError("Unknown pseudogrid schedule mode")
    budget = (
        config.get("constraint_budget_relative")
        if reduced and isinstance(config, dict)
        else None
    )
    if isinstance(config, dict) and "selection_spacing_ns" in config:
        if (
            not np.isscalar(config["selection_spacing_ns"])
            or isinstance(config["selection_spacing_ns"], (bool, str))
            or not np.isfinite(config["selection_spacing_ns"])
            or not np.isscalar(config.get("selection_origin_ns"))
            or isinstance(config.get("selection_origin_ns"), (bool, str))
            or not np.isfinite(config["selection_origin_ns"])
        ):
            raise ValueError("Invalid physical pseudogrid selection clock")
        ratio = config["selection_spacing_ns"] / width_ns
        phase = (payload["states"][0][0] - config["selection_origin_ns"]) / width_ns
        if (
            not np.isfinite([ratio, phase]).all()
            or ratio < 1
            or not np.isclose(ratio, round(ratio), atol=1e-8, rtol=0)
            or not np.isclose(phase, round(phase), atol=1e-8, rtol=0)
        ):
            raise ValueError(
                "Pseudogrid selection spacing must align with recording clock"
            )
    selections = []

    def select(states, step):
        nonlocal config
        config, active, updated = _schedule(config, states, step, mode)
        selections.append((active, updated))
        return (
            frozenset(set(range(len(states))) - set(active)) if reduced else frozenset()
        )

    current, records = _advance_particles(
        payload,
        width_ns,
        steps,
        passive_selector=select,
        passive_constraint_budget_relative=budget,
        passive_method="rk3" if mode == "passive_rk3" else "midpoint",
    )
    for record, (active, updated) in zip(records, selections):
        fallback_indices = [
            i
            for i, diagnostic in enumerate(record["particles"])
            if diagnostic.get("passive_reduced", {}).get("fallback_to_rk4", False)
        ]
        if budget is not None:
            for i in fallback_indices:
                config["fallback_count"][i] += 1
            for i, diagnostic in enumerate(record["particles"]):
                if "passive_reduced" in diagnostic and i not in fallback_indices:
                    config["reduced_accepted_count"][i] += 1
        record[field] = dict(
            selected_indices=active,
            selection_updated=updated,
            fallback_indices=fallback_indices,
            exact_particle_count=(
                (len(active) + len(fallback_indices))
                if reduced
                else len(current["particles"])
            ),
            approximate_particle_count=(
                (len(current["particles"]) - len(active) - len(fallback_indices))
                if reduced
                else 0
            ),
        )
    current[field] = config
    return current, records
