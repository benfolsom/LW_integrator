"""Physical particle-loss predicates for fixed-size trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .particle_status import mark_particle_dead
from .types import ParticleLossConfig, ParticleState, SimulationType


@dataclass(slots=True)
class ParticleLossContext:
    """Per-bunch loss thresholds resolved from the initial state."""

    initial_radial_loss_radius_mm: float | None = None


def build_particle_loss_context(
    config: ParticleLossConfig,
    initial_state: ParticleState,
) -> ParticleLossContext:
    """Resolve optional initial-state-dependent loss thresholds."""
    if not config.enabled or config.initial_radial_quantile is None:
        return ParticleLossContext()

    radius = _radial_distance(initial_state)
    if radius.size == 0:
        return ParticleLossContext()

    quantile_radius = float(np.quantile(radius, config.initial_radial_quantile))
    return ParticleLossContext(
        initial_radial_loss_radius_mm=(
            quantile_radius * float(config.initial_radial_multiplier)
            + float(config.initial_radial_margin_mm)
        )
    )


def mark_particle_losses(
    current_state: ParticleState,
    previous_state: ParticleState,
    *,
    step: int,
    config: ParticleLossConfig,
    context: ParticleLossContext | None = None,
    sim_type: SimulationType,
    wall_z: float,
    aperture_radius: float,
    logger: Any | None = None,
) -> int:
    """Apply enabled physical loss predicates to one accepted step.

    Returns the number of newly marked particles.
    """
    if not config.enabled:
        return 0

    particle_count = len(np.asarray(current_state.get("x", [])))
    if particle_count == 0:
        return 0

    previous_count = len(np.asarray(previous_state.get("x", [])))
    if previous_count != particle_count:
        return 0

    dead_mask = np.asarray(
        current_state.get("_dead_particles", np.zeros(particle_count, dtype=bool)),
        dtype=bool,
    )
    newly_lost = 0

    explicit_radius = config.loss_radius_mm
    if explicit_radius is not None:
        newly_lost += _mark_radial_losses(
            current_state,
            dead_mask,
            step=step,
            radius_limit_mm=float(explicit_radius),
            reason="loss_radius_exceeded",
        )
        dead_mask = np.asarray(
            current_state.get("_dead_particles", dead_mask),
            dtype=bool,
        )

    if context is not None and context.initial_radial_loss_radius_mm is not None:
        newly_lost += _mark_radial_losses(
            current_state,
            dead_mask,
            step=step,
            radius_limit_mm=float(context.initial_radial_loss_radius_mm),
            reason="initial_radial_envelope_exceeded",
        )
        dead_mask = np.asarray(
            current_state.get("_dead_particles", dead_mask),
            dtype=bool,
        )

    if (
        config.conducting_wall_aperture_loss_enabled
        and sim_type == SimulationType.CONDUCTING_WALL
        and aperture_radius > 0.0
    ):
        newly_lost += _mark_conducting_wall_aperture_losses(
            current_state,
            previous_state,
            dead_mask,
            step=step,
            wall_z=float(wall_z),
            aperture_radius=float(aperture_radius),
        )

    if newly_lost > 0:
        message = f"  [STATUS] Step {step}: {newly_lost} particle loss(es) marked"
        if logger:
            if callable(logger):
                logger(message)
            else:
                logger.info(message)
        else:
            print(message)

    return newly_lost


def _radial_distance(state: ParticleState) -> np.ndarray:
    x = np.asarray(state.get("x", []), dtype=float)
    y = np.asarray(state.get("y", np.zeros_like(x)), dtype=float)
    return np.hypot(x, y)


def _mark_radial_losses(
    state: ParticleState,
    dead_mask: np.ndarray,
    *,
    step: int,
    radius_limit_mm: float,
    reason: str,
) -> int:
    radius = _radial_distance(state)
    lost_indices = np.flatnonzero((radius > radius_limit_mm) & ~dead_mask)
    for particle_idx in lost_indices.tolist():
        mark_particle_dead(
            state,
            int(particle_idx),
            step,
            reason,
            details={
                "radius_mm": float(radius[particle_idx]),
                "radius_limit_mm": float(radius_limit_mm),
            },
        )
    return int(lost_indices.size)


def _mark_conducting_wall_aperture_losses(
    current_state: ParticleState,
    previous_state: ParticleState,
    dead_mask: np.ndarray,
    *,
    step: int,
    wall_z: float,
    aperture_radius: float,
) -> int:
    prev_z = np.asarray(previous_state["z"], dtype=float)
    curr_z = np.asarray(current_state["z"], dtype=float)
    dz = curr_z - prev_z
    crosses = (
        (np.abs(dz) > 0.0) & ((prev_z - wall_z) * (curr_z - wall_z) <= 0.0) & ~dead_mask
    )
    candidate_indices = np.flatnonzero(crosses)
    if candidate_indices.size == 0:
        return 0

    prev_x = np.asarray(previous_state["x"], dtype=float)
    prev_y = np.asarray(previous_state["y"], dtype=float)
    curr_x = np.asarray(current_state["x"], dtype=float)
    curr_y = np.asarray(current_state["y"], dtype=float)

    lost_count = 0
    for particle_idx in candidate_indices.tolist():
        alpha = float((wall_z - prev_z[particle_idx]) / dz[particle_idx])
        alpha = float(np.clip(alpha, 0.0, 1.0))
        x_cross = float(
            prev_x[particle_idx] + alpha * (curr_x[particle_idx] - prev_x[particle_idx])
        )
        y_cross = float(
            prev_y[particle_idx] + alpha * (curr_y[particle_idx] - prev_y[particle_idx])
        )
        radius_cross = float(np.hypot(x_cross, y_cross))
        if radius_cross <= aperture_radius:
            continue
        mark_particle_dead(
            current_state,
            int(particle_idx),
            step,
            "aperture_plane_loss",
            details={
                "wall_z_mm": float(wall_z),
                "aperture_radius_mm": float(aperture_radius),
                "radius_at_wall_mm": radius_cross,
                "x_at_wall_mm": x_cross,
                "y_at_wall_mm": y_cross,
            },
        )
        lost_count += 1
    return lost_count


__all__ = [
    "ParticleLossContext",
    "build_particle_loss_context",
    "mark_particle_losses",
]
