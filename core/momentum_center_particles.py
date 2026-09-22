"""Exact many-particle full-spin stepping and pseudogrid reference scheduling.

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


def initialize_particles(particles, states, histories, **options):
    """Prepare two or more particles using the pair solver's physical options."""
    return _initialize_particles(
        particles, states, histories, checkpoint_model=MULTIPARTICLE_MODEL, **options
    )


def configure_pseudogrid_reference(payload, active_count):
    """Record subset selections while retaining exact dynamics for everyone.

    This mode offers no speedup. Passive spin/history approximation is not yet
    enabled; this establishes its exact reference and checkpointed schedule.
    """
    _validate_model(payload)
    count = len(payload["particles"])
    if type(active_count) is not int or not 1 <= active_count <= count:
        raise ValueError("Active count must be an integer from one to particle count")
    if "pseudogrid_reference" in payload:
        raise ValueError("Resume preserves the existing pseudogrid reference schedule")
    result = copy.deepcopy(payload)
    result["pseudogrid_reference"] = dict(
        mode="exact_reference",
        active_count=active_count,
        last_active_step=[-1] * count,
        activation_count=[0] * count,
    )
    return result


def _validate_model(payload):
    if payload.get("model") != MULTIPARTICLE_MODEL:
        raise ValueError("A many-particle full-spin checkpoint is required")
    if "pseudo_grid" in payload:
        raise ValueError("Legacy reduced pseudogrid is not supported by this solver")


def _reference_schedule(payload):
    from .pseudo_grid import select_active_indices

    config = copy.deepcopy(payload["pseudogrid_reference"])
    count = len(payload["particles"])
    if (
        not isinstance(config, dict)
        or set(config)
        != {"mode", "active_count", "last_active_step", "activation_count"}
        or config["mode"] != "exact_reference"
        or type(config["active_count"]) is not int
        or not 1 <= config["active_count"] <= count
    ):
        raise ValueError("Invalid pseudogrid reference configuration")
    step = payload["accepted_steps"]
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
    states = np.asarray(payload["states"], dtype=float)
    if states.shape != (count, 14) or not np.isfinite(states).all():
        raise ValueError("Invalid many-particle states")
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
    return config, active.tolist()


def advance_particles(payload, width_ns, steps=1):
    """Advance all particles; optionally record a pseudogrid reference schedule."""
    _validate_model(payload)
    if "pseudogrid_reference" not in payload:
        return _advance_particles(payload, width_ns, steps)
    if type(steps) is not int or steps < 1:
        raise ValueError("Positive integer step count required")
    current, records = payload, []
    for _ in range(steps):
        config, active = _reference_schedule(current)
        current, batch = _advance_particles(current, width_ns)
        current["pseudogrid_reference"] = config
        batch[0]["pseudogrid_reference"] = dict(
            selected_indices=active,
            exact_particle_count=len(current["particles"]),
            approximate_particle_count=0,
        )
        records.extend(batch)
    return current, records
