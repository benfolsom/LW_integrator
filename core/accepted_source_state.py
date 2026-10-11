"""Shared extraction of accepted source states and interval acceleration."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .source_kinematics import coordinate_beta_prime_from_four_kinematics


def _state_vector(
    state: Mapping[str, object],
    names: tuple[str, str, str],
    particle_index: int,
) -> np.ndarray:
    values = np.asarray(
        [np.asarray(state[name], dtype=np.float64)[particle_index] for name in names],
        dtype=np.float64,
    )
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"accepted source state has invalid components {names!r}")
    return values


def _state_scalar(
    state: Mapping[str, object],
    name: str,
    particle_index: int,
) -> float:
    values = np.asarray(state[name], dtype=np.float64)
    if values.ndim != 1 or particle_index >= values.size:
        raise ValueError(f"accepted source state has no {name} value")
    result = float(values[particle_index])
    if not np.isfinite(result):
        raise ValueError(f"accepted source state has non-finite {name}")
    return result


def _accepted_step_start_beta_prime_per_mm(
    state: Mapping[str, object],
    particle_index: int,
) -> np.ndarray | None:
    """Return the complete accepted step-start acceleration when available."""

    ready_values = np.asarray(
        state.get("source_start_beta_prime_ready", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    if (
        ready_values.ndim == 1
        and particle_index < ready_values.size
        and bool(ready_values[particle_index])
    ):
        return _state_vector(
            state,
            (
                "source_start_beta_prime_x_per_mm",
                "source_start_beta_prime_y_per_mm",
                "source_start_beta_prime_z_per_mm",
            ),
            particle_index,
        )

    complete_values = np.asarray(
        state.get("_source_start_acceleration_complete", np.zeros(0, dtype=bool)),
        dtype=bool,
    )
    if (
        complete_values.ndim != 1
        or particle_index >= complete_values.size
        or not bool(complete_values[particle_index])
    ):
        return None
    velocity_values = np.asarray(
        state.get("_intrinsic_spin_start_four_velocity"),
        dtype=np.float64,
    )
    acceleration_values = np.asarray(
        state.get("_intrinsic_spin_start_non_self_four_acceleration"),
        dtype=np.float64,
    )
    if (
        velocity_values.ndim != 2
        or acceleration_values.ndim != 2
        or velocity_values.shape[1:] != (4,)
        or acceleration_values.shape != velocity_values.shape
        or particle_index >= velocity_values.shape[0]
    ):
        raise ValueError("accepted source state has invalid start four-kinematics")
    return coordinate_beta_prime_from_four_kinematics(
        velocity_values[particle_index],
        acceleration_values[particle_index],
    )
