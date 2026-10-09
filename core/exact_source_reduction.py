"""Conserved, fixed initial deposition for affordable exact cloud references.

The reduced macros replace the initial ensemble once, before prehistory is
built. There is no rotation, passive reconstruction, or charge override on
past knots. Each resulting macro has its own accepted centre history and
persistent transverse children, used by all exact providers.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .constants import C_MMNS
from .types import MacroparticleSmearingConfig, ParticleState


def fixed_spatial_partition(
    position: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Farthest-point seeds and nearest-seed cells in the supplied metric."""
    position = np.asarray(position, dtype=float)
    if position.ndim != 2 or position.shape[1] != 3 or not np.isfinite(position).all():
        raise ValueError("source partition requires finite positions of shape (n, 3)")
    if count < 1 or count > len(position):
        raise ValueError("source partition count must be within the original count")
    centre = np.mean(position, axis=0)
    first = int(np.argmin(np.sum((position - centre) ** 2, axis=1)))
    seeds = [first]
    distances = np.sum((position - position[first]) ** 2, axis=1)
    while len(seeds) < count:
        distances[seeds] = -1.0
        index = int(np.argmax(distances))
        seeds.append(index)
        distances = np.minimum(
            distances, np.sum((position - position[index]) ** 2, axis=1)
        )
    seeds = np.asarray(seeds, dtype=int)
    distances = np.sum((position[:, None] - position[seeds][None]) ** 2, axis=2)
    cells = np.argmin(distances, axis=1)
    # Distinct identities at coincident positions still own their seed cell.
    cells[seeds] = np.arange(count)
    return seeds, cells


def reduce_exact_initial_state(
    state: ParticleState,
    count: int,
    cloud: MacroparticleSmearingConfig | None = None,
) -> tuple[ParticleState, dict[str, Any]]:
    """Deposit homogeneous representative particles into fixed spatial cells.

    Source charge, diagnostic population, population-weighted position, and
    population-weighted spatial mechanical momentum are conserved initially.
    Species charge and inertia stay physical. Group internal velocity spread
    and self-fields are unresolved; refining the count recovers the ensemble.
    No closed-energy accuracy is implied by these initial moment identities.
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("exact source count must be a non-negative integer")
    total = len(state["x"])
    mapping: dict[str, Any] = {
        "original_count": total,
        "reduced_count": total,
        "identity": True,
    }
    if count == 0 or count >= total:
        return state, mapping
    if np.any(state.get("_dead_particles", False)):
        raise ValueError("initial source reduction does not support dead particles")
    for name in ("m", "m_species", "q_species", "q_observer", "char_time"):
        values = np.asarray(state[name], dtype=float)
        if (
            values.shape != (total,)
            or not np.isfinite(values).all()
            or not np.all(values == values[0])
        ):
            raise ValueError(f"source reduction requires homogeneous {name}")
    if not np.array_equal(state["m"], state["m_species"]) or not np.array_equal(
        state["q_observer"], state["q_species"]
    ):
        raise ValueError(
            "source reduction requires physical representative charge and inertia"
        )
    population = np.asarray(state["macro_population"], dtype=float)
    if (
        population.shape != (total,)
        or not np.isfinite(population).all()
        or np.any(population <= 0)
    ):
        raise ValueError("source reduction requires positive finite populations")
    charge = np.asarray(state["q_source"], dtype=float)
    # Neutral controls retain their populations and physical observer charges.
    if not (
        np.all(charge == 0) or np.array_equal(charge, population * state["q_species"])
    ):
        raise ValueError(
            "source charge must equal population times species charge, or zero"
        )
    if np.any(state.get("magnetic_moment_native", 0)):
        raise NotImplementedError(
            "source reduction currently supports zero intrinsic moments"
        )
    if np.any(state["t"] != state["t"][0]):
        raise ValueError("initial source reduction requires a common initial lab time")
    position = np.column_stack([state[axis] for axis in "xyz"])
    beta = np.column_stack([state["b" + axis] for axis in "xyz"])
    # One fixed bunch metric: stretch the longitudinal coordinate by the
    # gamma of the population-weighted mean proper velocity. This is a
    # geometric rest-frame proxy, not a Lorentz boost of asynchronous events.
    mean_u = np.average(state["gamma"][:, None] * beta, axis=0, weights=population)
    speed = np.linalg.norm(mean_u)
    direction = mean_u / speed if speed else np.array([0.0, 0.0, 1.0])
    metric_gamma = float(np.sqrt(1 + speed**2))
    centred = position - np.average(position, axis=0, weights=population)
    metric_position = (
        centred + (metric_gamma - 1) * (centred @ direction)[:, None] * direction
    )
    seeds, cells = fixed_spatial_partition(metric_position, count)
    result = {
        key: (
            value[seeds].copy()
            if isinstance(value, np.ndarray) and value.shape[:1] == (total,)
            else copy.deepcopy(value)
        )
        for key, value in state.items()
    }
    group_population = np.bincount(cells, weights=population, minlength=count)
    source_charge = np.bincount(cells, weights=charge, minlength=count)
    result["macro_population"] = group_population
    result["q_source"] = source_charge
    if not np.array_equal(state["q"], state["q_species"]):
        result["q"] = np.bincount(cells, weights=state["q"], minlength=count)
    momentum = (
        state["m_species"][:, None]
        * C_MMNS
        * state["gamma"][:, None]
        * np.column_stack([state["b" + axis] for axis in "xyz"])
    )
    group_momentum = np.zeros((count, 3))
    for axis_index, axis in enumerate("xyz"):
        result[axis] = (
            np.bincount(
                cells, weights=population * position[:, axis_index], minlength=count
            )
            / group_population
        )
        group_momentum[:, axis_index] = (
            np.bincount(
                cells, weights=population * momentum[:, axis_index], minlength=count
            )
            / group_population
        )
    proper_speed = group_momentum / (result["m_species"][:, None] * C_MMNS)
    result["gamma"] = np.sqrt(1 + np.sum(proper_speed**2, axis=1))
    for axis_index, axis in enumerate("xyz"):
        result["b" + axis] = proper_speed[:, axis_index] / result["gamma"]
        result["P" + axis] = group_momentum[:, axis_index]
    result["Pt"] = result["m_species"] * C_MMNS * result["gamma"]
    centres = np.column_stack([result[a] for a in "xyz"])
    transverse_covariance = np.zeros((count, 3, 3))
    cell_xv = np.zeros(count)
    cell_vv = np.zeros(count)
    for cell in range(count):
        members = cells == cell
        delta = position[members] - centres[cell]
        u = proper_speed[cell]
        norm = np.linalg.norm(u)
        normal = u / norm if norm else np.array([0.0, 0.0, 1.0])
        transverse = delta - (delta @ normal)[:, None] * normal
        transverse_covariance[cell] = (
            (transverse * population[members, None]).T
            @ transverse
            / group_population[cell]
        )
        if cloud is not None and cloud.enabled and cloud.breathing_enabled:
            # Unresolved members share the accepted centre translation and
            # retain their initial relative lab velocities (ballistic closure).
            velocity = beta[members] * C_MMNS
            velocity -= np.average(velocity, axis=0, weights=population[members])
            velocity -= (velocity @ normal)[:, None] * normal
            weights = population[members] / group_population[cell]
            cell_xv[cell] = np.sum(weights * np.sum(transverse * velocity, axis=1)) / 2
            cell_vv[cell] = np.sum(weights * np.sum(velocity**2, axis=1)) / 2
    # Isotropic per-axis RMS in the initial plane normal to the cell velocity.
    # The cloud provider floors this at the requested original effective width.
    cell_sigma = np.sqrt(
        np.maximum(0, np.trace(transverse_covariance, axis1=1, axis2=2) / 2)
    )
    result["exact_source_cell_sigma_mm"] = cell_sigma
    result["exact_source_original_count"] = np.full(count, total, dtype=float)
    if cloud is not None and cloud.enabled and cloud.breathing_enabled:
        result["exact_source_cell_xv_mm2_ns"] = cell_xv
        result["exact_source_cell_vv_mm2_ns2"] = cell_vv
        result["exact_source_cell_epoch_ns"] = np.full(count, state["t"][0])
        result["exact_source_breathing_ready"] = np.ones(count)
        mapping.update(
            breathing_covariance_model="ballistic_initial_cell_velocity_spread",
            cell_xv_mm2_ns=cell_xv.tolist(),
            cell_vv_mm2_ns2=cell_vv.tolist(),
            cell_epoch_ns=float(state["t"][0]),
        )
    # A coarse macro has new kinematics and a new centre. Seed caches from
    # those values, never from the selected parent's old resolved knot.
    for axis in "xyz":
        for part in ("low", "tail"):
            result[f"source_position_{part}_{axis}"] = np.zeros(count)
    for part in ("low", "tail"):
        result[f"source_time_{part}_ns"] = np.zeros(count)
    from .resolved_knot import initialize_mechanical_knots

    initialize_mechanical_knots(result)
    mapping.update(
        reduced_count=count,
        identity=False,
        seeds=seeds.tolist(),
        parent_cells=cells.tolist(),
        group_population=group_population.tolist(),
        group_source_charge=source_charge.tolist(),
        centres_mm=centres.tolist(),
        partition_metric="population_mean_proper_velocity_gamma_scaled",
        partition_gamma=metric_gamma,
        partition_direction=direction.tolist(),
        transverse_covariance_mm2=transverse_covariance.tolist(),
        cell_sigma_mm=cell_sigma.tolist(),
    )
    if cloud is not None and cloud.enabled:
        original_width = cloud.position_sigma_mm * cloud.sigma_multiplier
        mapping.update(
            original_effective_width_mm=original_width,
            group_effective_width_mm=np.maximum(cell_sigma, original_width).tolist(),
        )
    return result, mapping
