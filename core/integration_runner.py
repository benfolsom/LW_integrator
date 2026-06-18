"""High-level orchestration for retarded-field trajectory integration.

This module coordinates the low-level physics kernels, image-charge
construction, and optional self-consistency loops.  It provides the primary
programmatic entry points for running the modern Liénard–Wiechert integrator."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Callable, Optional, Tuple, cast

import numpy as np

from .constants import C_MMNS
from .equations import GammaBlowupError, retarded_equations_of_motion
from .images import generate_conducting_image, generate_switching_image
from .particle_status import (
    all_particles_dead,
    format_failure_summary,
    get_particle_failure_summary,
    mark_particle_dead,
    propagate_dead_particle_status,
)
from .particle_loss import build_particle_loss_context, mark_particle_losses
from .pseudo_grid import (
    ActiveTrajectoryView,
    PseudoGridPlannerState,
    PseudoGridStepSchedule,
    build_pseudo_grid_step_schedule,
    build_self_excluded_space_charge_source_charges,
    commit_pseudo_grid_step_schedule,
    initialize_pseudo_grid_planner_state,
    reconstruct_full_state_from_active_result,
    record_pseudo_grid_history_times,
    slice_trajectory_particle_history,
)
from .self_consistency import SelfConsistencyConfig, self_consistent_step
from .types import (
    ChronoMatchingMode,
    CavityExitConfig,
    DriverTrainConfig,
    IndexedTrajectoryArrays,
    IntegratorConfig,
    MacroparticleSmearingConfig,
    ParticleLossConfig,
    ParticleState,
    PseudoGridConfig,
    SimulationType,
    StartupMode,
    Trajectory,
    TrajectoryArrays,
    TrajectoryBuilder,
)


class IntegrationCancelled(RuntimeError):
    """Raised when an integration is cancelled by an external caller."""


def _build_partial_soa(
    trajectory: Trajectory, up_to_step: int
) -> "TrajectoryArrays | None":
    """Build a TrajectoryArrays from trajectory[0:up_to_step]."""
    if up_to_step < 1 or not trajectory[0]:
        return None
    try:
        n_p = len(trajectory[0]["x"])
        builder = TrajectoryBuilder(up_to_step, n_p)
        for idx in range(up_to_step):
            builder.set_step(idx, trajectory[idx])
        return builder.build()
    except Exception:
        return None


def _indexed_active_history(
    trajectory_soa: TrajectoryArrays | None,
    particle_indices: np.ndarray,
    *,
    start_step: int = 0,
    q_override: np.ndarray | None = None,
) -> IndexedTrajectoryArrays | None:
    if trajectory_soa is None:
        return None
    try:
        return IndexedTrajectoryArrays(
            trajectory_soa,
            np.asarray(particle_indices, dtype=int),
            start_step=int(start_step),
            q_override=q_override,
        )
    except ValueError:
        return None


@lru_cache(maxsize=None)
def _call_accepts_kw(callable_obj: Callable, name: str) -> bool:
    """Return whether *callable_obj* accepts a keyword argument."""
    parameters = inspect.signature(callable_obj).parameters
    return name in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


class EnergyJumpDetected(RuntimeError):
    """Raised when an energy jump exceeds the configured threshold."""


@dataclass
class EnergyMonitorConfig:
    """Configuration for optional energy jump detection during integration."""

    enabled: bool = False
    relative_threshold: float = (
        10.0  # Relative energy change threshold (e.g., 10.0 = 1000%)
    )
    check_interval: int = 1  # Check every N steps
    halt_on_jump: bool = False  # If True, raise exception; if False, just warn
    debug: bool = False  # Print energy changes


@dataclass
class AdaptiveTimestepConfig:
    """Configuration for adaptive timestep refinement on energy jumps.

    When an energy jump is detected, the timestep is reduced. The hysteresis
    parameters control how long we stay at the reduced timestep before attempting
    to return to the normal timestep.

    Proximity-based refinement: Automatically refine timesteps when approaching
    walls/apertures where image charge interactions become significant.
    """

    enabled: bool = False
    energy_jump_threshold: float = (
        0.1  # Relative energy change to trigger refinement (e.g., 0.1 = 10%)
    )
    timestep_reduction_factor: int = (
        3  # Reduce timestep by this factor when jump detected (2, 3, or 10 typical)
    )
    min_timestep_factor: float = 1e-4  # Minimum timestep as fraction of original

    @property
    def max_refinement_attempts(self) -> int:
        """Maximum refinement attempts, auto-calculated from reduction_factor and min_timestep_factor.

        This ensures consistency: the maximum number of reductions needed to reach
        the minimum timestep is automatically determined.

        Formula: max_attempts = ceil(log(1/min_timestep_factor) / log(reduction_factor))

        Examples:
            reduction_factor=3, min_timestep_factor=1e-4 → max_attempts = 9
            reduction_factor=10, min_timestep_factor=1e-4 → max_attempts = 4
        """
        import math

        if self.timestep_reduction_factor <= 1:
            return 1  # Safety: avoid division by zero or invalid log

        # Calculate how many reductions needed to reach min_timestep_factor
        # h / (factor^n) = h * min_factor  →  factor^n = 1/min_factor
        # n = log(1/min_factor) / log(factor)
        attempts = math.ceil(
            math.log(1.0 / self.min_timestep_factor)
            / math.log(self.timestep_reduction_factor)
        )
        return max(1, attempts)  # At least 1 attempt

    # Hysteresis parameters: stay on reduced timestep for stability
    cooldown_steps: int = 10  # Minimum steps at reduced timestep before probing return
    probe_threshold: float = 0.01  # Energy stability threshold for safe return (1%)
    max_probe_steps: int = 3  # Number of consecutive stable steps needed to return

    # Impractical timestep handling
    skip_cooldown_on_particle_death: bool = (
        False  # Keep survivors in cooldown mode for safer recovery
    )

    @property
    def max_substeps_per_step(self) -> int:
        """Maximum substeps per step, automatically calculated from min_timestep_factor.

        This ensures that even at minimum timestep, we can cover the full base timestep
        interval without time discontinuities.

        Returns ceil(1 / min_timestep_factor) with a 10% safety margin.
        """
        import math

        theoretical_max = math.ceil(1.0 / self.min_timestep_factor)
        # Add 10% safety margin
        return int(theoretical_max * 1.1)

    # Proximity-based refinement: refine timesteps near walls/apertures
    proximity_refinement_enabled: bool = True  # Enable proximity-based refinement
    proximity_distance_aperture_radii: float = (
        10.0  # Refine within this many aperture radii
    )
    proximity_reduction_factor: int = 5  # Timestep reduction factor in proximity region
    proximity_transition_zone: float = 2.0  # Smooth transition zone (in aperture radii)

    # Bunch-separation proximity refinement (BUNCH_TO_BUNCH mode only)
    bunch_proximity_enabled: bool = False
    bunch_proximity_sigma_mm: float = 5.0  # characteristic bunch length scale (mm)
    bunch_proximity_n_sigma: float = 5.0  # engage when separation < n_sigma * sigma_mm
    bunch_proximity_reduction_factor: float = 10.0  # divisor at full engagement
    bunch_proximity_transition_n_sigma: float = 2.0  # ramp-in width (sigma units)

    debug: bool = False  # Print adaptive timestep actions


def _compute_total_energy(state: ParticleState) -> float:
    """Compute total energy of particles in a state.

    Energy is calculated as E = γmc² summed over all particles.

    Parameters
    ----------
    state:
        Particle state containing gamma and mass arrays.

    Returns
    -------
    float
        Total energy in MeV.
    """
    gamma = np.asarray(state["gamma"])
    mass = np.asarray(state["m"])
    return float(np.sum(gamma * mass * C_MMNS * C_MMNS))


def _calculate_gamma(state: ParticleState) -> float:
    """Calculate the maximum gamma (Lorentz factor) from a particle state.

    Parameters
    ----------
    state:
        Particle state containing gamma array.

    Returns
    -------
    float
        Maximum gamma value among all particles in the state.
    """
    gamma = np.asarray(state["gamma"])
    return float(np.max(gamma))


def _ensure_startup_metadata(state: Optional[ParticleState]) -> None:
    if state is None:
        return

    required_length = len(state.get("x", []))
    if required_length == 0:
        return

    def _ensure(
        name: str, source_key: str | None = None, *, fill_value: float = 0.0
    ) -> None:
        if name not in state or state[name].shape != state["x"].shape:
            if source_key is not None:
                state[name] = np.copy(state[source_key])
            else:
                state[name] = np.full_like(state["x"], fill_value, dtype=float)

    _ensure("origin_x", "x")
    _ensure("origin_y", "y")
    _ensure("origin_z", "z")
    _ensure("beta_avg_x", "bx")
    _ensure("beta_avg_y", "by")
    _ensure("beta_avg_z", "bz")
    _ensure("beta_samples", None, fill_value=1.0)
    _ensure("radiation_power")
    _ensure("radiation_energy")
    _ensure("radiation_energy_applied")


def _set_pseudo_grid_schedule_metadata(
    state: Optional[ParticleState], schedule: object | None
) -> None:
    if state is None:
        return
    if schedule is None:
        state.pop("_pseudo_grid_schedule", None)
    else:
        state["_pseudo_grid_schedule"] = schedule


def _adaptive_timestep_enabled(
    adaptive_timestep: Optional[AdaptiveTimestepConfig],
) -> bool:
    return bool(adaptive_timestep is not None and adaptive_timestep.enabled)


def _space_charge_enabled(space_charge: Optional[Any]) -> bool:
    return bool(space_charge is not None and getattr(space_charge, "enabled", False))


def _mark_post_step_gamma_blowups(
    state: ParticleState,
    *,
    step: int,
    logger: Optional[Any] = None,
) -> int:
    gamma_array = state.get("gamma")
    if gamma_array is None:
        return 0

    marked = 0
    for particle_idx, gamma_val in enumerate(np.asarray(gamma_array, dtype=float)):
        dead_mask = state.get("_dead_particles")
        if dead_mask is not None and dead_mask[particle_idx]:
            continue

        if gamma_val > 1e8 or np.isnan(gamma_val) or np.isinf(gamma_val):
            message = (
                f"[WARNING] Step {step}: Particle {particle_idx} gamma blowup "
                f"detected (γ={gamma_val:.2e}). Marking particle as dead."
            )
            if logger:
                if callable(logger):
                    logger(message)
                else:
                    logger.warning(message)
            else:
                print(message)
            mark_particle_dead(
                state,
                particle_idx,
                step,
                "gamma_blowup_post_step",
                gamma_value=float(gamma_val),
            )
            marked += 1
    return marked


def _copy_particle_state(state: ParticleState) -> ParticleState:
    return {
        key: (value.copy() if isinstance(value, (dict, np.ndarray)) else value)
        for key, value in state.items()
    }


def _centroid_z(state: ParticleState) -> float:
    return float(np.mean(np.asarray(state["z"], dtype=float)))


def _motion_direction_sign(state: ParticleState) -> float:
    beta_z = np.asarray(state.get("bz", []), dtype=float)
    if beta_z.size:
        mean_beta_z = float(np.mean(beta_z))
        if mean_beta_z > 0.0:
            return 1.0
        if mean_beta_z < 0.0:
            return -1.0

    pz = np.asarray(state.get("Pz", []), dtype=float)
    if pz.size:
        mean_pz = float(np.mean(pz))
        if mean_pz > 0.0:
            return 1.0
        if mean_pz < 0.0:
            return -1.0

    return 0.0


def _leading_edge_z(state: ParticleState) -> float:
    return _leading_edge_z_for_slice(state, slice(None))


def _leading_edge_z_for_slice(state: ParticleState, particle_slice: slice) -> float:
    z_values = np.asarray(state["z"], dtype=float)[particle_slice]
    if z_values.size == 0:
        return 0.0

    beta_z = np.asarray(state.get("bz", []), dtype=float)
    if beta_z.size:
        beta_z = beta_z[particle_slice]
        if beta_z.size:
            mean_beta_z = float(np.mean(beta_z))
            if mean_beta_z < 0.0:
                return float(np.min(z_values))
            if mean_beta_z > 0.0:
                return float(np.max(z_values))

    pz = np.asarray(state.get("Pz", []), dtype=float)
    if pz.size:
        pz = pz[particle_slice]
        if pz.size:
            mean_pz = float(np.mean(pz))
            if mean_pz < 0.0:
                return float(np.min(z_values))
            if mean_pz > 0.0:
                return float(np.max(z_values))

    return float(np.mean(z_values))


def _crossed_exit(previous_z: float, current_z: float, exit_z: float) -> bool:
    if previous_z == exit_z or current_z == exit_z:
        return True
    return (previous_z - exit_z) * (current_z - exit_z) < 0.0


def _latest_populated_state(trajectory: Trajectory, before_index: int) -> ParticleState:
    for state in reversed(trajectory[:before_index]):
        if state:
            return state
    raise ValueError("trajectory has no populated state before cavity-exit check")


def _state_mean(state: ParticleState, key: str, default: float = 0.0) -> float:
    value = state.get(key)
    if value is None:
        return default
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return default
    return float(np.mean(array))


def _driver_train_bunch_slices(
    driver_train: DriverTrainConfig,
    particles_per_bunch: int,
) -> tuple[slice, ...]:
    if not driver_train.enabled:
        return ()
    return tuple(
        slice(index * particles_per_bunch, (index + 1) * particles_per_bunch)
        for index in range(driver_train.bunch_count)
    )


def _build_driver_train_initial_state(
    init_driver: ParticleState,
    driver_train: DriverTrainConfig,
) -> ParticleState:
    offsets = driver_train.resolved_z_offsets_mm()
    if not driver_train.enabled and len(offsets) == 1 and offsets[0] == 0.0:
        return _copy_particle_state(init_driver)

    particle_count = len(np.asarray(init_driver["x"]))
    train_state: ParticleState = {}
    for key, value in init_driver.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (particle_count,):
            pieces = []
            for z_offset in offsets:
                piece = np.copy(value)
                if key == "z":
                    piece = piece + float(z_offset)
                pieces.append(piece)
            train_state[key] = np.concatenate(pieces, axis=0)
        elif isinstance(value, np.ndarray):
            train_state[key] = np.copy(value)
        elif isinstance(value, dict):
            train_state[key] = dict(value)
        else:
            train_state[key] = value
    return train_state


def _coast_state_by_proper_steps(
    state: ParticleState,
    h_step: float,
    step_offset: int,
) -> ParticleState:
    result = _copy_particle_state(state)
    gamma = np.asarray(state["gamma"], dtype=float)
    dt_lab = gamma * float(h_step) * float(step_offset)
    for axis, beta_key in (("x", "bx"), ("y", "by"), ("z", "bz")):
        if axis in state and beta_key in state:
            result[axis] = np.asarray(state[axis], dtype=float) + (
                np.asarray(state[beta_key], dtype=float) * C_MMNS * dt_lab
            )
    if "t" in state:
        result["t"] = np.asarray(state["t"], dtype=float) + dt_lab
    return result


def _build_coasting_history(
    active_state: ParticleState,
    h_step: float,
    prehistory_steps: int,
) -> Trajectory:
    history = [
        _coast_state_by_proper_steps(active_state, h_step, step_offset)
        for step_offset in range(-int(prehistory_steps), 1)
    ]
    oldest = history[0]
    for state in history:
        for axis in ("x", "y", "z"):
            state[f"origin_{axis}"] = np.copy(oldest[axis])
        state["beta_avg_x"] = np.copy(state.get("bx", oldest["x"] * 0.0))
        state["beta_avg_y"] = np.copy(state.get("by", oldest["y"] * 0.0))
        state["beta_avg_z"] = np.copy(state.get("bz", oldest["z"] * 0.0))
        state["beta_samples"] = np.ones_like(oldest["x"], dtype=float)
    return history


def _slice_trajectory_arrays(
    arrays: TrajectoryArrays | None,
    start: int,
    stop: int,
) -> TrajectoryArrays | None:
    if arrays is None:
        return None
    return TrajectoryArrays(
        x=arrays.x[start:stop],
        y=arrays.y[start:stop],
        z=arrays.z[start:stop],
        t=arrays.t[start:stop],
        Px=arrays.Px[start:stop],
        Py=arrays.Py[start:stop],
        Pz=arrays.Pz[start:stop],
        Pt=arrays.Pt[start:stop],
        gamma=arrays.gamma[start:stop],
        bx=arrays.bx[start:stop],
        by=arrays.by[start:stop],
        bz=arrays.bz[start:stop],
        bdotx=arrays.bdotx[start:stop],
        bdoty=arrays.bdoty[start:stop],
        bdotz=arrays.bdotz[start:stop],
        radiation_power=arrays.radiation_power[start:stop],
        radiation_energy=arrays.radiation_energy[start:stop],
        radiation_energy_applied=arrays.radiation_energy_applied[start:stop],
        origin_x=arrays.origin_x[start:stop],
        origin_y=arrays.origin_y[start:stop],
        origin_z=arrays.origin_z[start:stop],
        beta_avg_x=arrays.beta_avg_x[start:stop],
        beta_avg_y=arrays.beta_avg_y[start:stop],
        beta_avg_z=arrays.beta_avg_z[start:stop],
        beta_samples=arrays.beta_samples[start:stop],
        dead=arrays.dead[start:stop],
        q=arrays.q,
        q_species=arrays.q_species,
        q_observer=arrays.q_observer,
        q_source=arrays.q_source,
        macro_population=arrays.macro_population,
        m=arrays.m,
        m_species=arrays.m_species,
        char_time=arrays.char_time,
        halted_early=arrays.halted_early[start:stop],
        halt_step=arrays.halt_step[start:stop],
        halt_reason=arrays.halt_reason[start:stop],
        particle_failure_info=arrays.particle_failure_info,
        pseudo_grid_schedule=arrays.pseudo_grid_schedule[start:stop],
    )


def _resolve_history_start_index(
    history: Trajectory,
    start_index: int | None,
    *,
    history_base_index: int = 0,
) -> int:
    if start_index is None:
        return 0
    if not history:
        return 0
    local_start_index = int(start_index) - int(history_base_index)
    if local_start_index <= 0:
        return 0
    return min(local_start_index, len(history) - 1)


def _history_retention_enabled(
    pseudo_grid: PseudoGridConfig,
    *,
    force_reduction_enabled: bool,
) -> bool:
    return bool(
        pseudo_grid.enabled
        and force_reduction_enabled
        and pseudo_grid.causal_history_pruning_enabled
    )


def _drop_retained_history_prefix(
    history: Trajectory,
    current_base_index: int,
    requested_start_index: int | None,
) -> tuple[int, int]:
    if requested_start_index is None or requested_start_index <= current_base_index:
        return current_base_index, 0
    if not history:
        return current_base_index, 0

    new_base_index = min(
        int(requested_start_index), current_base_index + len(history) - 1
    )
    drop_count = new_base_index - current_base_index
    if drop_count > 0:
        del history[:drop_count]
    return new_base_index, drop_count


def _clear_legacy_history_prefix(
    trajectory: Trajectory,
    *,
    start_index: int,
    end_index: int,
) -> int:
    if end_index <= start_index:
        return start_index
    bounded_end_index = min(end_index, len(trajectory))
    for history_index in range(start_index, bounded_end_index):
        trajectory[history_index] = {}
    return bounded_end_index


def _run_pseudo_grid_reduced_step(
    *,
    h_step: float,
    observer_history: Trajectory,
    source_history: Trajectory,
    observer_active_indices: np.ndarray,
    source_active_indices: np.ndarray,
    source_effective_charges: np.ndarray,
    source_history_start_index: int | None,
    passive_map: Any,
    aperture_radius: float,
    sim_type: SimulationType,
    self_consistency: Optional[SelfConsistencyConfig],
    chrono_mode: ChronoMatchingMode,
    startup_mode: StartupMode,
    step_idx: int,
    cancel_callback: Optional[Callable],
    logger: Optional[Any],
    radiation_reaction_mode: str,
    external_field: Optional[Any],
    space_charge: Optional[Any],
    pseudo_grid_weighting_mode: str,
    loss_tracking_enabled: bool,
    source_history_base_index: int = 0,
    observer_history_base_index: int = 0,
    observer_soa: TrajectoryArrays | None = None,
    source_soa: TrajectoryArrays | None = None,
    raise_gamma_blowup: bool = False,
    macroparticle_smearing: MacroparticleSmearingConfig | None = None,
) -> ParticleState:
    """Advance one pseudo-grid half-step via active-only observer/source solves."""
    if not observer_history:
        raise ValueError("observer_history must contain at least one state")

    if np.asarray(observer_active_indices, dtype=int).size == 0:
        return _copy_particle_state(observer_history[-1])

    observer_active = np.asarray(observer_active_indices, dtype=int)
    source_active = np.asarray(source_active_indices, dtype=int)
    source_local_start = _resolve_history_start_index(
        source_history,
        source_history_start_index,
        history_base_index=source_history_base_index,
    )
    observer_global_start = int(observer_history_base_index)
    source_global_start = int(source_history_base_index) + source_local_start

    observer_active_soa = _indexed_active_history(
        observer_soa,
        observer_active,
        start_step=observer_global_start,
    )
    source_active_soa = _indexed_active_history(
        source_soa,
        source_active,
        start_step=source_global_start,
        q_override=source_effective_charges,
    )

    if observer_active_soa is not None and source_active_soa is not None:
        observer_active_history = ActiveTrajectoryView(observer_active_soa)
        source_active_history = ActiveTrajectoryView(source_active_soa)
    else:
        observer_active_history = slice_trajectory_particle_history(
            observer_history,
            observer_active,
        )
        source_active_history = slice_trajectory_particle_history(
            source_history,
            source_active,
            start_index=source_local_start,
            q_override=source_effective_charges,
        )
        observer_active_soa = _build_partial_soa(
            observer_active_history,
            len(observer_active_history),
        )
        source_active_soa = _build_partial_soa(
            source_active_history,
            len(source_active_history),
        )

    local_index = len(observer_active_history) - 1
    pseudo_grid_space_charge_source_charges = None
    if _space_charge_enabled(space_charge):
        pseudo_grid_space_charge_source_charges = (
            build_self_excluded_space_charge_source_charges(
                observer_history[-1],
                observer_active_indices,
                passive_map,
                weighting_mode=pseudo_grid_weighting_mode,
            )
        )

    try:
        active_result_state = self_consistent_step(
            retarded_equations_of_motion,
            h_step,
            cast(Trajectory, observer_active_history),
            cast(Trajectory, source_active_history),
            local_index,
            aperture_radius,
            sim_type,
            self_consistency,
            chrono_mode,
            startup_mode,
            step_idx=step_idx,
            cancel_callback=cancel_callback,
            space_charge=space_charge,
            radiation_reaction_mode=radiation_reaction_mode,
            pseudo_grid_space_charge_source_charges=(
                pseudo_grid_space_charge_source_charges
            ),
            macroparticle_smearing=macroparticle_smearing,
            traj_soa=observer_active_soa,
            traj_ext_soa=source_active_soa,
            **(
                {"external_field": external_field} if external_field is not None else {}
            ),
        )
    except GammaBlowupError as exc:
        local_particle_idx = int(exc.particle_idx)
        if (
            0
            <= local_particle_idx
            < np.asarray(observer_active_indices, dtype=int).size
        ):
            global_particle_idx = int(observer_active_indices[local_particle_idx])
        else:
            global_particle_idx = local_particle_idx

        if raise_gamma_blowup:
            raise GammaBlowupError(
                step_idx=exc.step_idx,
                particle_idx=global_particle_idx,
                gamma_value=exc.gamma_value,
                iteration=exc.iteration,
                is_hard_blowup=exc.is_hard_blowup,
            ) from exc

        message = (
            f"    [CRITICAL] Step {step_idx}, Particle {global_particle_idx}: "
            f"Gamma blowup (γ={exc.gamma_value:.2e}) during pseudo-grid active solve. "
            f"Marking particle as dead."
        )
        if logger:
            if callable(logger):
                logger(message)
            else:
                logger.warning(message)
        else:
            print(message)

        active_result_state = _copy_particle_state(
            cast(ParticleState, observer_active_history[-1])
        )
        mark_particle_dead(
            active_result_state,
            local_particle_idx,
            step_idx,
            "gamma_blowup_no_adaptive",
            gamma_value=exc.gamma_value,
            iteration=exc.iteration,
        )

    return reconstruct_full_state_from_active_result(
        observer_history[-1],
        observer_active_indices,
        active_result_state,
        passive_map,
        loss_tracking_enabled=loss_tracking_enabled,
    )


@dataclass
class _AdaptiveStepState:
    """Mutable cross-step state for adaptive timestep logic."""

    reduced_timestep_mode: bool = False
    reduced_h_step: float = 0.0
    cooldown_counter: int = 0
    stable_steps_counter: int = 0
    last_particle_death_step: int = -1
    previous_energy: Optional[float] = None
    current_h_step: float = 0.0


def _run_adaptive_step(
    *,
    i: int,
    steps: int,
    h_step: float,
    wall_z: float,
    aperture_radius: float,
    sim_type: SimulationType,
    chrono_mode: ChronoMatchingMode,
    startup_mode: StartupMode,
    self_consistency: Optional[SelfConsistencyConfig],
    adaptive_timestep: Optional[AdaptiveTimestepConfig],
    space_charge: Optional[Any],
    external_field: Optional[Any],
    image_subcharge_count: int,
    use_conducting_image_weighting: bool,
    macroparticle_charge_multiplier: float,
    macroparticle_sigma_multiplier: float,
    macroparticle_use_momentum_errors: bool,
    bunch_transv_dist: float,
    bunch_transv_mom: float,
    z_cutoff: float,
    trajectory: Trajectory,
    trajectory_drv: Trajectory,
    _traj_drv_builder: Optional[TrajectoryBuilder],
    cancel_callback: Optional[Callable],
    logger: Optional[Any],
    adaptive_state: "_AdaptiveStepState",
    radiation_reaction_mode: str,
    pseudo_grid_schedule: PseudoGridStepSchedule | None = None,
    pseudo_grid_force_reduction_enabled: bool = False,
    pseudo_grid_weighting_mode: str = "inverse_distance",
    pseudo_grid_loss_tracking_enabled: bool = True,
    pseudo_grid_observer_history: Trajectory | None = None,
    pseudo_grid_source_history: Trajectory | None = None,
    pseudo_grid_observer_history_base_index: int = 0,
    pseudo_grid_source_history_base_index: int = 0,
    pseudo_grid_observer_soa: TrajectoryArrays | None = None,
    pseudo_grid_source_soa: TrajectoryArrays | None = None,
    use_full_history: bool = False,
    macroparticle_smearing: MacroparticleSmearingConfig | None = None,
) -> ParticleState:
    """Run one adaptive step, updating adaptive_state in-place.

    Returns the new ParticleState for step i.
    Reads trajectory[i-1] and trajectory_drv[i-1] as the previous step.
    """
    reduced_timestep_mode = adaptive_state.reduced_timestep_mode
    reduced_h_step = adaptive_state.reduced_h_step
    cooldown_counter = adaptive_state.cooldown_counter
    stable_steps_counter = adaptive_state.stable_steps_counter
    last_particle_death_step = adaptive_state.last_particle_death_step
    previous_energy = adaptive_state.previous_energy
    current_h_step = adaptive_state.current_h_step

    step_accepted = False
    refinement_attempt = 0

    temp_trajectory: Trajectory = []
    temp_driver: Trajectory = []

    bunch_proximity_reduction_active = False
    bunch_proximity_factor = 1.0
    proximity_reduction_active = False
    proximity_factor = 1.0
    if (
        adaptive_timestep is not None
        and adaptive_timestep.proximity_refinement_enabled
        and aperture_radius is not None
        and wall_z is not None
        and sim_type != SimulationType.BUNCH_TO_BUNCH
    ):
        current_z = float(np.mean(trajectory[i - 1]["z"]))
        distance_to_wall = abs(wall_z - current_z)
        interaction_distance = (
            aperture_radius * adaptive_timestep.proximity_distance_aperture_radii
        )
        transition_distance = (
            aperture_radius * adaptive_timestep.proximity_transition_zone
        )

        if distance_to_wall <= interaction_distance:
            proximity_reduction_active = True
            if distance_to_wall < (interaction_distance - transition_distance):
                proximity_factor = adaptive_timestep.proximity_reduction_factor
            else:
                ramp = (interaction_distance - distance_to_wall) / transition_distance
                proximity_factor = (
                    1.0 + (adaptive_timestep.proximity_reduction_factor - 1.0) * ramp
                )

            if adaptive_timestep.debug:
                in_transition = distance_to_wall >= (
                    interaction_distance - transition_distance
                )
                zone_name = "transition" if in_transition else "full reduction"
                msg = (
                    f"Step {i}: Proximity refinement active ({zone_name} zone). "
                    f"Distance to wall: {distance_to_wall:.6e} mm "
                    f"({distance_to_wall / aperture_radius:.1f} aperture radii). "
                    f"Reduction factor: {proximity_factor:.4f}x"
                )
                if logger:
                    logger(msg)
                else:
                    print(msg)

    if (
        adaptive_timestep is not None
        and adaptive_timestep.bunch_proximity_enabled
        and sim_type == SimulationType.BUNCH_TO_BUNCH
        and len(trajectory_drv) >= i
    ):
        sigma_mm = adaptive_timestep.bunch_proximity_sigma_mm
        n_sigma = adaptive_timestep.bunch_proximity_n_sigma
        transition_n_sigma = adaptive_timestep.bunch_proximity_transition_n_sigma
        engage_dist = n_sigma * sigma_mm
        full_dist = max(0.0, (n_sigma - transition_n_sigma) * sigma_mm)

        rider_z = float(np.mean(trajectory[i - 1]["z"]))
        driver_z = float(np.mean(trajectory_drv[i - 1]["z"]))
        separation = abs(driver_z - rider_z)

        if separation <= engage_dist:
            bunch_proximity_reduction_active = True
            if separation <= full_dist:
                bunch_proximity_factor = (
                    adaptive_timestep.bunch_proximity_reduction_factor
                )
                zone_name = "full reduction"
            else:
                transition_width = engage_dist - full_dist
                ramp = (engage_dist - separation) / transition_width
                bunch_proximity_factor = (
                    1.0
                    + (adaptive_timestep.bunch_proximity_reduction_factor - 1.0) * ramp
                )
                zone_name = "transition"

            if adaptive_timestep.debug:
                msg = (
                    f"Step {i}: Bunch proximity refinement active ({zone_name} zone). "
                    f"Separation: {separation:.4f} mm ({separation / sigma_mm:.2f} sigma). "
                    f"Reduction factor: {bunch_proximity_factor:.4f}x"
                )
                if logger:
                    logger(msg)
                else:
                    print(msg)

    if cancel_callback is not None and cancel_callback():
        raise IntegrationCancelled("Integration cancelled by caller.")

    if reduced_timestep_mode and adaptive_timestep is not None:
        expected_substeps = int(np.ceil(h_step / reduced_h_step))
        impractical_timestep = (
            expected_substeps > adaptive_timestep.max_substeps_per_step
        )
        skip_cooldown = (
            adaptive_timestep.skip_cooldown_on_particle_death
            and last_particle_death_step == i - 1
        )

        if impractical_timestep:
            if adaptive_timestep.debug:
                msg = (
                    f"Step {i}: Impractical timestep detected ({expected_substeps} sub-steps required). "
                    f"Skipping cooldown and attempting recovery to {h_step:.6e} ns"
                )
                if logger:
                    logger(msg)
                else:
                    print(msg)
            cooldown_counter = adaptive_timestep.cooldown_steps
        elif skip_cooldown:
            if adaptive_timestep.debug:
                msg = (
                    f"Step {i}: Particle died last step. Skipping cooldown for survivors, "
                    f"attempting recovery to {h_step:.6e} ns"
                )
                if logger:
                    logger(msg)
                else:
                    print(msg)
            cooldown_counter = adaptive_timestep.cooldown_steps
        elif cooldown_counter < adaptive_timestep.cooldown_steps:
            current_h_step = reduced_h_step
            cooldown_counter += 1
            if adaptive_timestep.debug:
                msg = (
                    f"Step {i}: Cooldown mode ({cooldown_counter}/{adaptive_timestep.cooldown_steps}), "
                    f"using reduced timestep {current_h_step:.6e} ns"
                )
                if logger:
                    logger(msg)
                else:
                    print(msg)
        else:
            current_h_step = reduced_h_step
            if adaptive_timestep.debug:
                msg = (
                    f"Step {i}: Probing stability with reduced timestep "
                    f"({stable_steps_counter}/{adaptive_timestep.max_probe_steps} stable)"
                )
                if logger:
                    logger(msg)
                else:
                    print(msg)
    else:
        current_h_step = h_step
        combined_factor = max(
            proximity_factor if proximity_reduction_active else 1.0,
            bunch_proximity_factor if bunch_proximity_reduction_active else 1.0,
        )
        if combined_factor > 1.0 and adaptive_timestep is not None:
            current_h_step = h_step / combined_factor
            if adaptive_timestep.debug:
                active_parts = []
                if proximity_reduction_active:
                    active_parts.append(f"wall({proximity_factor:.2f}x)")
                if bunch_proximity_reduction_active:
                    active_parts.append(f"bunch({bunch_proximity_factor:.2f}x)")
                msg = (
                    f"Step {i}: Applying proximity refinement "
                    f"[{', '.join(active_parts)}]: "
                    f"{h_step:.6e} \u2192 {current_h_step:.6e} ns"
                )
                if logger:
                    logger(msg)
                else:
                    print(msg)

    if (
        pseudo_grid_force_reduction_enabled
        and pseudo_grid_schedule is not None
        and not _adaptive_timestep_enabled(adaptive_timestep)
    ):
        result_state = _run_pseudo_grid_reduced_step(
            h_step=current_h_step,
            observer_history=pseudo_grid_observer_history or trajectory[:i],
            source_history=pseudo_grid_source_history or trajectory_drv[:i],
            observer_active_indices=pseudo_grid_schedule.rider_active_indices,
            source_active_indices=pseudo_grid_schedule.driver_active_indices,
            source_effective_charges=pseudo_grid_schedule.driver_effective_source_charges,
            source_history_start_index=pseudo_grid_schedule.driver_history_start_index,
            passive_map=pseudo_grid_schedule.rider_passive_map,
            aperture_radius=aperture_radius,
            sim_type=sim_type,
            self_consistency=self_consistency,
            chrono_mode=chrono_mode,
            startup_mode=startup_mode,
            step_idx=i,
            cancel_callback=cancel_callback,
            logger=logger,
            radiation_reaction_mode=radiation_reaction_mode,
            external_field=external_field,
            space_charge=space_charge,
            pseudo_grid_weighting_mode=pseudo_grid_weighting_mode,
            loss_tracking_enabled=pseudo_grid_loss_tracking_enabled,
            source_history_base_index=pseudo_grid_source_history_base_index,
            observer_history_base_index=pseudo_grid_observer_history_base_index,
            observer_soa=pseudo_grid_observer_soa,
            source_soa=pseudo_grid_source_soa,
            macroparticle_smearing=macroparticle_smearing,
        )
        adaptive_state.reduced_timestep_mode = reduced_timestep_mode
        adaptive_state.reduced_h_step = reduced_h_step
        adaptive_state.cooldown_counter = cooldown_counter
        adaptive_state.stable_steps_counter = stable_steps_counter
        adaptive_state.last_particle_death_step = last_particle_death_step
        adaptive_state.previous_energy = previous_energy
        adaptive_state.current_h_step = current_h_step
        return result_state

    temp_trajectory_base = {
        k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
        for k, v in trajectory[i - 1].items()
    }
    probe_reference_energy = previous_energy

    if i > 0:
        propagate_dead_particle_status(temp_trajectory_base, trajectory[i - 1])

    def propagate_deaths_to_base(state_with_deaths):
        if "_dead_particles" in state_with_deaths:
            if "_dead_particles" not in temp_trajectory_base:
                num_particles = len(state_with_deaths.get("gamma", []))
                temp_trajectory_base["_dead_particles"] = np.zeros(
                    num_particles, dtype=bool
                )
                temp_trajectory_base["_particle_failure_info"] = {}
            temp_trajectory_base["_dead_particles"] |= state_with_deaths[
                "_dead_particles"
            ]
            if "_particle_failure_info" in state_with_deaths:
                temp_trajectory_base["_particle_failure_info"].update(
                    state_with_deaths["_particle_failure_info"]
                )

    while not step_accepted:
        if cancel_callback is not None and cancel_callback():
            raise IntegrationCancelled("Integration cancelled by caller.")

        num_substeps_raw = int(np.round(h_step / current_h_step))

        if adaptive_timestep is not None:
            num_substeps = min(
                num_substeps_raw, adaptive_timestep.max_substeps_per_step
            )
            if (
                num_substeps_raw > adaptive_timestep.max_substeps_per_step
                and adaptive_timestep.debug
            ):
                msg = (
                    f"Step {i}: Sub-step limit reached ({adaptive_timestep.max_substeps_per_step}). "
                    f"Timestep {current_h_step:.6e} ns would require {num_substeps_raw} sub-steps. "
                    f"This step may not fully cover the base timestep interval."
                )
                if logger:
                    logger(msg)
                else:
                    print(msg)
        else:
            num_substeps = num_substeps_raw

        if num_substeps < 1:
            num_substeps = 1

        temp_trajectory = []
        if use_full_history and i > 1:
            temp_trajectory.extend(trajectory[: i - 1])
        temp_trajectory.append(
            {
                k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
                for k, v in temp_trajectory_base.items()
            }
        )
        temp_driver = (
            trajectory_drv[:i] if use_full_history else [trajectory_drv[i - 1]]
        )

        _n_p_rider = len(temp_trajectory[0]["x"])
        _temp_traj_builder = TrajectoryBuilder(
            len(temp_trajectory) + num_substeps,
            _n_p_rider,
        )
        for step_index, state in enumerate(temp_trajectory):
            _temp_traj_builder.set_step(step_index, state)
        _n_p_driver = len(temp_driver[0]["x"])
        _temp_drv_builder = TrajectoryBuilder(
            len(temp_driver) + num_substeps,
            _n_p_driver,
        )
        for step_index, state in enumerate(temp_driver):
            _temp_drv_builder.set_step(step_index, state)
        _scs_accepts_soa = _call_accepts_kw(self_consistent_step, "traj_soa")
        _scs_accepts_radiation = _call_accepts_kw(
            self_consistent_step, "radiation_reaction_mode"
        )
        _scs_accepts_external_field = _call_accepts_kw(
            self_consistent_step, "external_field"
        )

        energy_jump_detected = False
        gamma_blowup_detected = False
        max_refinement_reached = False
        min_timestep_reached = False

        for substep_idx in range(num_substeps):
            if cancel_callback is not None and cancel_callback():
                raise IntegrationCancelled("Integration cancelled by caller.")

            current_observer_index = len(temp_trajectory) - 1

            try:
                if (
                    pseudo_grid_force_reduction_enabled
                    and pseudo_grid_schedule is not None
                    and sim_type == SimulationType.BUNCH_TO_BUNCH
                ):
                    trial_state = _run_pseudo_grid_reduced_step(
                        h_step=current_h_step,
                        observer_history=temp_trajectory,
                        source_history=temp_driver,
                        observer_active_indices=pseudo_grid_schedule.rider_active_indices,
                        source_active_indices=pseudo_grid_schedule.driver_active_indices,
                        source_effective_charges=pseudo_grid_schedule.driver_effective_source_charges,
                        source_history_start_index=(
                            pseudo_grid_schedule.driver_history_start_index
                        ),
                        passive_map=pseudo_grid_schedule.rider_passive_map,
                        aperture_radius=aperture_radius,
                        sim_type=sim_type,
                        self_consistency=self_consistency,
                        chrono_mode=chrono_mode,
                        startup_mode=startup_mode,
                        step_idx=i,
                        cancel_callback=cancel_callback,
                        logger=logger,
                        radiation_reaction_mode=radiation_reaction_mode,
                        external_field=external_field,
                        space_charge=space_charge,
                        pseudo_grid_weighting_mode=pseudo_grid_weighting_mode,
                        loss_tracking_enabled=pseudo_grid_loss_tracking_enabled,
                        source_history_base_index=i - 1,
                        observer_history_base_index=i - 1,
                        observer_soa=None,
                        source_soa=None,
                        raise_gamma_blowup=_adaptive_timestep_enabled(
                            adaptive_timestep
                        ),
                        macroparticle_smearing=macroparticle_smearing,
                    )
                else:
                    trial_state = self_consistent_step(
                        retarded_equations_of_motion,
                        current_h_step,
                        temp_trajectory,
                        temp_driver,
                        current_observer_index,
                        aperture_radius,
                        sim_type,
                        self_consistency,
                        chrono_mode,
                        startup_mode,
                        step_idx=i,
                        cancel_callback=cancel_callback,
                        **(
                            {"radiation_reaction_mode": radiation_reaction_mode}
                            if _scs_accepts_radiation
                            else {}
                        ),
                        **(
                            {"space_charge": space_charge}
                            if space_charge is not None
                            else {}
                        ),
                        **(
                            {"external_field": external_field}
                            if external_field is not None
                            and _scs_accepts_external_field
                            else {}
                        ),
                        **(
                            {
                                "traj_soa": _temp_traj_builder.build_partial(
                                    len(temp_trajectory)
                                )
                            }
                            if _scs_accepts_soa
                            else {}
                        ),
                        **(
                            {
                                "traj_ext_soa": _temp_drv_builder.build_partial(
                                    len(temp_driver)
                                )
                            }
                            if _scs_accepts_soa
                            else {}
                        ),
                        macroparticle_smearing=macroparticle_smearing,
                    )
            except GammaBlowupError as e:
                if adaptive_timestep is None or not adaptive_timestep.enabled:
                    msg = (
                        f"    [CRITICAL] Step {i}, Particle {e.particle_idx}: "
                        f"Gamma blowup (\u03b3={e.gamma_value:.2e}) with no adaptive timestep available. "
                        f"Marking particle as dead."
                    )
                    if logger:
                        logger(msg)
                    else:
                        print(msg)
                    if len(temp_trajectory) > 0:
                        trial_state = {
                            k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
                            for k, v in temp_trajectory[-1].items()
                        }
                    else:
                        trial_state = {
                            k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
                            for k, v in trajectory[i - 1].items()
                        }

                    mark_particle_dead(
                        trial_state,
                        e.particle_idx,
                        i,
                        "gamma_blowup_no_adaptive",
                        gamma_value=e.gamma_value,
                        iteration=e.iteration,
                    )

                    temp_trajectory.append(trial_state)
                    _temp_traj_builder.set_step(len(temp_trajectory) - 1, trial_state)
                    temp_driver.append(
                        temp_driver[-1] if temp_driver else trajectory_drv[i - 1]
                    )
                    _temp_drv_builder.set_step(len(temp_driver) - 1, temp_driver[-1])
                    last_particle_death_step = i
                    gamma_blowup_detected = False
                    break
                else:
                    gamma_blowup_detected = True
                    refinement_attempt += 1

                    if refinement_attempt > adaptive_timestep.max_refinement_attempts:
                        msg = (
                            f"    [CRITICAL] Step {i}, Particle {e.particle_idx}: "
                            f"Max refinement attempts reached after gamma blowup (\u03b3={e.gamma_value:.6e}). "
                            f"Marking particle as dead."
                        )
                        if logger:
                            logger(msg)
                        else:
                            print(msg)
                        if len(temp_trajectory) > 0:
                            trial_state = {
                                k: (
                                    v.copy() if isinstance(v, (dict, np.ndarray)) else v
                                )
                                for k, v in temp_trajectory[-1].items()
                            }
                        else:
                            trial_state = {
                                k: (
                                    v.copy() if isinstance(v, (dict, np.ndarray)) else v
                                )
                                for k, v in trajectory[i - 1].items()
                            }

                        mark_particle_dead(
                            trial_state,
                            e.particle_idx,
                            i,
                            "gamma_blowup_max_retries",
                            gamma_value=e.gamma_value,
                            iteration=e.iteration,
                        )

                        temp_trajectory.append(trial_state)
                        _temp_traj_builder.set_step(
                            len(temp_trajectory) - 1, trial_state
                        )
                        temp_driver.append(
                            temp_driver[-1] if temp_driver else trajectory_drv[i - 1]
                        )
                        _temp_drv_builder.set_step(
                            len(temp_driver) - 1, temp_driver[-1]
                        )
                        last_particle_death_step = i
                        gamma_blowup_detected = False
                        break
                    else:
                        min_h = h_step * adaptive_timestep.min_timestep_factor
                        new_h_step = (
                            current_h_step / adaptive_timestep.timestep_reduction_factor
                        )

                        if new_h_step < min_h:
                            msg = (
                                f"    [CRITICAL] Step {i}, Particle {e.particle_idx}: "
                                f"Minimum timestep reached after gamma blowup (\u03b3={e.gamma_value:.6e}). "
                                f"Marking particle as dead."
                            )
                            if logger:
                                logger(msg)
                            else:
                                print(msg)
                            if len(temp_trajectory) > 0:
                                trial_state = {
                                    k: (
                                        v.copy()
                                        if isinstance(v, (dict, np.ndarray))
                                        else v
                                    )
                                    for k, v in temp_trajectory[-1].items()
                                }
                            else:
                                trial_state = {
                                    k: (
                                        v.copy()
                                        if isinstance(v, (dict, np.ndarray))
                                        else v
                                    )
                                    for k, v in trajectory[i - 1].items()
                                }

                            mark_particle_dead(
                                trial_state,
                                e.particle_idx,
                                i,
                                "gamma_blowup_min_timestep",
                                gamma_value=e.gamma_value,
                                iteration=e.iteration,
                            )

                            temp_trajectory.append(trial_state)
                            _temp_traj_builder.set_step(
                                len(temp_trajectory) - 1, trial_state
                            )
                            temp_driver.append(
                                temp_driver[-1]
                                if temp_driver
                                else trajectory_drv[i - 1]
                            )
                            _temp_drv_builder.set_step(
                                len(temp_driver) - 1, temp_driver[-1]
                            )
                            last_particle_death_step = i
                            gamma_blowup_detected = False
                            break
                        else:
                            if hasattr(e, "is_hard_blowup") and e.is_hard_blowup:
                                reduction_factor = (
                                    adaptive_timestep.timestep_reduction_factor**2
                                )
                                severity = "HARD"
                            else:
                                reduction_factor = (
                                    adaptive_timestep.timestep_reduction_factor
                                )
                                severity = "soft"

                            new_h_step = current_h_step / reduction_factor
                            if new_h_step < min_h:
                                new_h_step = min_h

                            current_h_step = new_h_step
                            if adaptive_timestep.debug:
                                msg = (
                                    f"Step {i}.{substep_idx}: {severity} gamma blowup detected (\u03b3={e.gamma_value:.6e}). "
                                    f"Reducing timestep by {reduction_factor}x "
                                    f"to {current_h_step:.6e} ns (attempt {refinement_attempt})"
                                )
                                if logger:
                                    logger(msg)
                                else:
                                    print(msg)

                            reduced_timestep_mode = True
                            reduced_h_step = current_h_step
                            cooldown_counter = 0
                            stable_steps_counter = 0
                            break

            propagate_deaths_to_base(trial_state)

            if sim_type == SimulationType.SWITCHING_WALL:
                trial_driver = generate_switching_image(
                    trial_state, wall_z, aperture_radius, z_cutoff
                )
            elif sim_type == SimulationType.CONDUCTING_WALL:
                trial_driver = generate_conducting_image(
                    trial_state,
                    wall_z,
                    aperture_radius,
                    subcharge_count=image_subcharge_count,
                    use_weighting=use_conducting_image_weighting,
                    macroparticle_charge_multiplier=macroparticle_charge_multiplier,
                    macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
                    macroparticle_use_momentum_errors=macroparticle_use_momentum_errors,
                    bunch_transv_dist=bunch_transv_dist,
                    bunch_transv_mom=bunch_transv_mom,
                    timestep=current_h_step,
                    step_number=i,
                )
            else:
                trial_driver = temp_driver[-1]

            if adaptive_timestep is not None and adaptive_timestep.enabled:
                current_energy = _compute_total_energy(trial_state)

                if previous_energy is not None and previous_energy > 0:
                    relative_change = (
                        abs(current_energy - previous_energy) / previous_energy
                    )

                    if relative_change > adaptive_timestep.energy_jump_threshold:
                        energy_jump_detected = True
                        refinement_attempt += 1

                        if (
                            refinement_attempt
                            > adaptive_timestep.max_refinement_attempts
                        ):
                            if adaptive_timestep.debug and not max_refinement_reached:
                                msg = (
                                    f"Step {i}: Max refinement attempts ({adaptive_timestep.max_refinement_attempts}) reached. "
                                    f"Accepting remaining substeps (\u0394E/E = {relative_change:.6e})"
                                )
                                if logger:
                                    logger(msg)
                                else:
                                    print(msg)
                                max_refinement_reached = True
                            energy_jump_detected = False
                        else:
                            min_h = h_step * adaptive_timestep.min_timestep_factor
                            new_h_step = (
                                current_h_step
                                / adaptive_timestep.timestep_reduction_factor
                            )

                            if new_h_step < min_h:
                                if adaptive_timestep.debug and not min_timestep_reached:
                                    msg = (
                                        f"Step {i}: Minimum timestep reached. "
                                        f"Accepting remaining substeps (\u0394E/E = {relative_change:.6e})"
                                    )
                                    if logger:
                                        logger(msg)
                                    else:
                                        print(msg)
                                    min_timestep_reached = True
                                energy_jump_detected = False
                            else:
                                current_h_step = new_h_step
                                if adaptive_timestep.debug:
                                    msg = (
                                        f"Step {i}: Energy jump (\u0394E/E = {relative_change:.6e}). "
                                        f"Reducing timestep by {adaptive_timestep.timestep_reduction_factor}x "
                                        f"to {current_h_step:.6e} ns (attempt {refinement_attempt})"
                                    )
                                    if logger:
                                        logger(msg)
                                    else:
                                        print(msg)

                                reduced_timestep_mode = True
                                reduced_h_step = current_h_step
                                cooldown_counter = 0
                                stable_steps_counter = 0

                                propagate_deaths_to_base(temp_trajectory[-1])
                                break

                previous_energy = current_energy

            temp_trajectory.append(trial_state)
            _temp_traj_builder.set_step(len(temp_trajectory) - 1, trial_state)
            temp_driver.append(trial_driver)
            _temp_drv_builder.set_step(len(temp_driver) - 1, trial_driver)

        if not energy_jump_detected and not gamma_blowup_detected:
            step_accepted = True
            if (
                adaptive_timestep is not None
                and adaptive_timestep.debug
                and refinement_attempt > 0
            ):
                msg = f"Step {i}: Completed {num_substeps} sub-step(s) with timestep {current_h_step:.6e} ns"
                if logger:
                    logger(msg)
                else:
                    print(msg)

            if (
                adaptive_timestep is not None
                and adaptive_timestep.enabled
                and reduced_timestep_mode
                and cooldown_counter >= adaptive_timestep.cooldown_steps
            ):
                if probe_reference_energy is not None:
                    current_energy = _compute_total_energy(temp_trajectory[-1])
                    relative_change = (
                        abs(current_energy - probe_reference_energy)
                        / probe_reference_energy
                    )

                    if relative_change < adaptive_timestep.probe_threshold:
                        stable_steps_counter += 1
                        if adaptive_timestep.debug:
                            msg = (
                                f"Step {i}: Stable (\u0394E/E = {relative_change:.6e} < {adaptive_timestep.probe_threshold:.6e}), "
                                f"count = {stable_steps_counter}/{adaptive_timestep.max_probe_steps}"
                            )
                            if logger:
                                logger(msg)
                            else:
                                print(msg)

                        if stable_steps_counter >= adaptive_timestep.max_probe_steps:
                            reduced_timestep_mode = False
                            stable_steps_counter = 0
                            cooldown_counter = 0
                            if adaptive_timestep.debug:
                                msg = (
                                    f"Step {i}: Returning to normal timestep {h_step:.6e} ns "
                                    f"after {adaptive_timestep.max_probe_steps} stable steps"
                                )
                                if logger:
                                    logger(msg)
                                else:
                                    print(msg)
                    else:
                        stable_steps_counter = 0
                        cooldown_counter = 0
                        if adaptive_timestep.debug:
                            msg = (
                                f"Step {i}: Unstable during probing (\u0394E/E = {relative_change:.6e}), "
                                f"restarting cooldown"
                            )
                            if logger:
                                logger(msg)
                            else:
                                print(msg)

                        propagate_deaths_to_base(temp_trajectory[-1])

    adaptive_state.reduced_timestep_mode = reduced_timestep_mode
    adaptive_state.reduced_h_step = reduced_h_step
    adaptive_state.cooldown_counter = cooldown_counter
    adaptive_state.stable_steps_counter = stable_steps_counter
    adaptive_state.last_particle_death_step = last_particle_death_step
    adaptive_state.previous_energy = previous_energy
    adaptive_state.current_h_step = current_h_step

    return temp_trajectory[-1]


def retarded_integrator(
    steps: int,
    h_step: float,
    wall_z: float,
    aperture_radius: float,
    sim_type: SimulationType,
    init_rider: ParticleState,
    init_driver: Optional[ParticleState],
    mean: float,
    cav_spacing: float,
    z_cutoff: float,
    z_cutoff_mode: str = "absolute",
    self_consistency: Optional[SelfConsistencyConfig] = None,
    chrono_mode: ChronoMatchingMode = ChronoMatchingMode.AVERAGED,
    startup_mode: StartupMode = StartupMode.COLD_START,
    image_subcharge_count: int = 12,
    use_conducting_image_weighting: bool = True,
    macroparticle_charge_multiplier: float = 1.0,
    macroparticle_sigma_multiplier: float = 1.0,
    macroparticle_use_momentum_errors: bool = True,
    bunch_transv_dist: float = 0.0,
    bunch_transv_mom: float = 0.0,
    energy_monitor: Optional[EnergyMonitorConfig] = None,
    adaptive_timestep: Optional[AdaptiveTimestepConfig] = None,
    space_charge: Optional[Any] = None,
    external_field: Optional[Any] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    logger: Optional[Any] = None,
    use_numba: bool = True,
    radiation_reaction_mode: str = "off",
    pseudo_grid: Optional[PseudoGridConfig] = None,
    driver_train: Optional[DriverTrainConfig] = None,
    cavity_exit: Optional[CavityExitConfig] = None,
    particle_loss: Optional[ParticleLossConfig] = None,
    macroparticle_smearing: Optional[MacroparticleSmearingConfig] = None,
) -> Tuple[
    Trajectory, Trajectory, "TrajectoryArrays | None", "TrajectoryArrays | None"
]:
    """Run the retarded-field integrator for rider and driver trajectories.

    Parameters
    ----------
    steps:
        Total number of integration updates to compute.
    h_step:
        Temporal step between states (``Δτ`` in the covariant formulation).
    wall_z:
        Conducting wall location for boundary-condition simulations.
    aperture_radius:
        Aperture radius used by the wall/image generators.
    sim_type:
        Boundaries and interaction type encoded as :class:`SimulationType`.
    init_rider:
        Initial state of the primary bunch.
    init_driver:
        Optional initial state of the opposing bunch (for ``BUNCH_TO_BUNCH``).
    mean:
        Historical bunch separation parameter retained for compatibility.
    cav_spacing:
        Longitudinal spacing between cavities when using switching walls.
    z_cutoff:
        Threshold beyond which the switching wall no longer mirrors charges.
        For BUNCH_TO_BUNCH with z_cutoff_mode='relative', this is the distance
        from starting position to stop integration.
    z_cutoff_mode:
        Interpretation of z_cutoff. 'absolute' (default) treats z_cutoff as
        absolute z position. 'relative' treats it as distance from start
        (useful for BUNCH_TO_BUNCH mode to stop after traveling distance X).
    self_consistency:
        Optional :class:`SelfConsistencyConfig` to iterate each step until the
        Lorentz factor converges.
    chrono_mode:
        Retardation sampling strategy; ``FAST`` reproduces the historical
        solver, ``AVERAGED`` blends ``R / c`` and ``2R / c`` emission times.
    startup_mode:
        Strategy for handling the lack of retarded history at the beginning of
        a simulation.
    image_subcharge_count:
        Number of subcharges used when constructing conducting-wall image
        charges. Must remain within the bounds accepted by
        :func:`generate_conducting_image`.
    use_conducting_image_weighting:
        Whether to apply radial weighting to conducting-wall image subcharges.
    macroparticle_charge_multiplier:
        Multiplier for particle and image charges in macroparticle simulations.
        Only applies to CONDUCTING_WALL simulations. Default 1.0.
    macroparticle_sigma_multiplier:
        Multiplier applied to bunch spread parameters when computing image charge errors.
        Default 1.0 (errors = bunch spread). Position errors derived from rider transv_dist,
        momentum errors derived from rider transv_mom. Only applies to CONDUCTING_WALL.
    macroparticle_use_momentum_errors:
        Whether to include momentum-based cumulative errors in image charge positions.
        If False, only constant position errors are applied. Default True (both types).
    bunch_transv_dist:
        Transverse distribution half-width (mm) from particle bunch initialization.
        Used to compute position spread for image charge errors. Default 0.0.
    bunch_transv_mom:
        Transverse momentum spread (amu*mm/ns) from particle bunch initialization.
        Used to compute cumulative displacement errors. Default 0.0.
    energy_monitor:
        Optional :class:`EnergyMonitorConfig` to detect sudden energy jumps
        during integration. Can warn or halt on excessive energy changes.
    adaptive_timestep:
        Optional :class:`AdaptiveTimestepConfig` to enable adaptive timestep
        refinement. When an energy jump is detected, the step is discarded
        and retried with a smaller timestep.
    external_field:
        Optional prescribed external field configuration. The first supported
        implementation is a uniform field in native solver units with simple
        spatial/temporal gates.
    progress_callback:
        Optional callable invoked as ``progress_callback(current, steps)`` after
        each successful step. Used for progress bars or cancellation checks.
    cancel_callback:
        Optional callable invoked periodically to check for cancellation.
        If it returns True, integration is aborted with IntegrationCancelled.
    logger:
        Optional logger instance for integration diagnostics.
    use_numba:
        Compatibility flag retained after removal of the legacy alternate
        integrator path. The canonical path is always used; this flag controls
        only logging intent, while actual kernel availability follows
        ``core.vectorized_interactions.NUMBA_AVAILABLE``.
    radiation_reaction_mode:
        Radiation reaction mode. ``off`` and ``diagnostic_only`` record
        Liénard radiated power without changing momentum.
        ``power_matched_damping`` removes the radiated energy from mechanical
        momentum after the normal LW update. ``medina_lad`` applies the
        experimental Medina/LAD candidate force to mechanical momentum before
        recomposing canonical momentum.
    pseudo_grid:
        Experimental pseudo-grid configuration surface. When enabled for
        ``BUNCH_TO_BUNCH`` runs, the integrator builds per-step active/passive
        schedules and, in supported configurations, advances active observers
        against active-source reduced histories with effective source charges
        before passive particles receive weighted delta updates. Adaptive
        timestep refinement participates in the reduced active-set path.
        Intra-bunch space charge also uses the reduced path when each bunch has
        at least two active particles; otherwise the canonical full-history
        fallback remains in place.
    driver_train:
        Optional flat driver-train configuration for ``BUNCH_TO_BUNCH`` runs.
        It expands the driver bunch into fixed longitudinally offset copies and
        can seed inertial prehistory before the active output window.
    particle_loss:
        Optional fixed-size particle-loss predicates. Lost particles are marked
        dead, retain their trajectory slot, and stop contributing source charge
        after the loss step.

    Returns
    -------
    Tuple[Trajectory, Trajectory, TrajectoryArrays | None, TrajectoryArrays | None]
        Rider and driver trajectories plus optional SOA views:
        ``(rider_trajectory, driver_trajectory, rider_soa, driver_soa)``.

    Raises
    ------
    IntegrationCancelled
        If ``cancel_callback()`` returns ``True`` during integration.
    EnergyJumpDetected
        If adaptive timestep is disabled and an energy jump exceeds the
        threshold (with ``halt_on_jump=True``).
    """

    from . import vectorized_interactions as _vectorized_interactions

    pseudo_grid = pseudo_grid or PseudoGridConfig()
    driver_train = driver_train or DriverTrainConfig()
    cavity_exit = cavity_exit or CavityExitConfig()
    particle_loss = particle_loss or ParticleLossConfig()
    macroparticle_smearing = macroparticle_smearing or MacroparticleSmearingConfig()
    driver_train_enabled = bool(driver_train.enabled)
    driver_train_bunch_ranges: tuple[slice, ...] = ()
    if driver_train_enabled and init_driver is not None:
        driver_train_bunch_ranges = _driver_train_bunch_slices(
            driver_train,
            len(np.asarray(init_driver["x"])),
        )
    if pseudo_grid.enabled and sim_type != SimulationType.BUNCH_TO_BUNCH:
        raise NotImplementedError(
            "Pseudo-grid schedule mode is currently implemented only for "
            "SimulationType.BUNCH_TO_BUNCH."
        )
    if driver_train_enabled:
        if sim_type != SimulationType.BUNCH_TO_BUNCH:
            raise NotImplementedError(
                "Driver-train mode is implemented only for "
                "SimulationType.BUNCH_TO_BUNCH."
            )
        if init_driver is None:
            raise ValueError("Driver-train mode requires init_driver state")
        init_driver = _build_driver_train_initial_state(init_driver, driver_train)

    cavity_exit_enabled = bool(cavity_exit.enabled)
    if cavity_exit_enabled:
        if sim_type != SimulationType.BUNCH_TO_BUNCH:
            raise NotImplementedError(
                "Cavity-exit cutoff is implemented only for SimulationType.BUNCH_TO_BUNCH."
            )
        if init_driver is None:
            raise ValueError("Cavity-exit cutoff requires init_driver state")

    active_start = int(driver_train.prehistory_steps) if driver_train_enabled else 0
    requested_steps = int(steps)
    total_steps = requested_steps + active_start

    pseudo_grid_space_charge_reduction_supported = not _space_charge_enabled(
        space_charge
    ) or (pseudo_grid.active_rider_count >= 2 and pseudo_grid.active_driver_count >= 2)
    pseudo_grid_force_reduction_enabled = (
        pseudo_grid.enabled
        and sim_type == SimulationType.BUNCH_TO_BUNCH
        and pseudo_grid_space_charge_reduction_supported
    )

    numba_kernels_enabled = bool(use_numba and _vectorized_interactions.NUMBA_AVAILABLE)

    if logger:
        if numba_kernels_enabled:
            message = "Using Numba-optimized kernels in canonical integrator path"
            if callable(logger):
                logger(message)
            else:
                logger.info(message)
        elif use_numba:
            message = "Numba not available, using pure Python kernels"
            if callable(logger):
                logger(message)
            else:
                logger.warning(message)
        else:
            message = "use_numba=False requested; running canonical path without forcing kernel changes"
            if callable(logger):
                logger(message)
            else:
                logger.info(message)

    if pseudo_grid.enabled:
        if pseudo_grid_force_reduction_enabled:
            retention_text = (
                "causal-history retention will compact live histories"
                if pseudo_grid.causal_history_pruning_enabled
                else "full history storage remains active"
            )
            message = (
                "Pseudo-grid reduced active-set force evaluation enabled for "
                f"BUNCH_TO_BUNCH; {retention_text}"
            )
        else:
            reasons = []
            if (
                _space_charge_enabled(space_charge)
                and not pseudo_grid_space_charge_reduction_supported
            ):
                reasons.append(
                    "intra-bunch space charge requires at least two active particles per bunch"
                )
            reason_text = (
                ", and ".join(reasons)
                if reasons
                else "current configuration requires fallback"
            )
            message = (
                "Pseudo-grid schedule construction enabled, but reduced "
                "active-set force evaluation is falling back to the canonical "
                f"full-history solve because {reason_text}"
            )
        if logger:
            if callable(logger):
                logger(message)
            else:
                logger.info(message)
        else:
            print(message)

    # Canonical integration implementation

    rider_seed_history = (
        _build_coasting_history(init_rider, h_step, active_start)
        if driver_train_enabled
        else [init_rider]
    )
    driver_seed_history = (
        _build_coasting_history(cast(ParticleState, init_driver), h_step, active_start)
        if driver_train_enabled and init_driver is not None
        else None
    )

    trajectory: Trajectory = [{} for _ in range(total_steps)]
    trajectory_drv: Trajectory = [{} for _ in range(total_steps)]
    _n_particles_rider = len(init_rider["x"])
    _n_particles_drv: int | None = None
    _traj_builder = TrajectoryBuilder(total_steps, _n_particles_rider)
    _traj_drv_builder: TrajectoryBuilder | None = None
    _pseudo_grid_planner_state: PseudoGridPlannerState | None = None
    _pseudo_grid_retention_active = _history_retention_enabled(
        pseudo_grid,
        force_reduction_enabled=pseudo_grid_force_reduction_enabled,
    )
    _rider_retained_history: Trajectory = []
    _driver_retained_history: Trajectory = []
    _rider_retained_history_start_index = 0
    _driver_retained_history_start_index = 0
    _rider_legacy_history_cleared_until = 0
    _driver_legacy_history_cleared_until = 0
    _legacy_history_compacted = False
    _rider_loss_context = build_particle_loss_context(particle_loss, init_rider)
    _driver_loss_context = (
        build_particle_loss_context(particle_loss, init_driver)
        if init_driver is not None
        else None
    )

    # Initialize energy monitoring
    previous_energy: Optional[float] = None

    # Store initial z position for relative cutoff mode
    z_initial: Optional[float] = None
    if z_cutoff_mode == "relative" and sim_type == SimulationType.BUNCH_TO_BUNCH:
        z_initial = _centroid_z(init_rider)
        if adaptive_timestep is not None and adaptive_timestep.debug:
            msg = (
                f"BUNCH_TO_BUNCH relative cutoff mode: z_initial = {z_initial:.6f} mm, "
                f"will stop after traveling {z_cutoff:.2f} mm"
            )
            if logger:
                logger(msg)
            else:
                print(msg)

    cavity_rider_exit_z: Optional[float] = None
    cavity_driver_exit_z: Optional[float] = None
    cavity_length_mm: Optional[float] = None
    cavity_exit_triggered = False
    cavity_exit_mode = str(cavity_exit.mode)
    rider_exit_controls_global_halt = cavity_exit_mode == "rider_exit_with_driver_tail"
    cavity_exit_exit_index: int | None = None
    cavity_exit_exit_species: str | None = None
    cavity_exit_exit_z: float | None = None
    cavity_exit_exit_time_ns: float | None = None
    cavity_exit_tail_steps_planned = 0
    cavity_exit_tail_steps_executed = 0
    cavity_exit_tail_time_ns = 0.0
    cavity_exit_tail_stop_index: int | None = None
    driver_bunch_exit_steps: list[int | None] = [
        None for _ in driver_train_bunch_ranges
    ]
    driver_bunch_tail_steps_planned: list[int] = [0 for _ in driver_train_bunch_ranges]
    driver_bunch_muted: list[bool] = [False for _ in driver_train_bunch_ranges]
    if cavity_exit_enabled and init_driver is not None:
        rider_z0 = _leading_edge_z(init_rider)
        driver_z0 = _leading_edge_z(init_driver)
        axis_sign = 1.0 if driver_z0 >= rider_z0 else -1.0
        cavity_length_mm = (
            float(cavity_exit.cavity_length_mm)
            if cavity_exit.cavity_length_mm is not None
            else abs(driver_z0 - rider_z0)
        )
        cavity_rider_exit_z = rider_z0 + axis_sign * cavity_length_mm
        cavity_driver_exit_z = driver_z0 - axis_sign * cavity_length_mm
        if adaptive_timestep is not None and adaptive_timestep.debug:
            msg = (
                "BUNCH_TO_BUNCH cavity-exit cutoff: "
                f"length={cavity_length_mm:.6f} mm, "
                f"rider_exit_z={cavity_rider_exit_z:.6f} mm, "
                f"driver_exit_z={cavity_driver_exit_z:.6f} mm"
            )
            if logger:
                logger(msg)
            else:
                print(msg)

    def _finalize_cavity_exit_halt(
        *,
        step_index: int,
        exit_species: str,
        exit_z: float,
        exit_time_ns: float,
        tail_steps_planned: int,
        tail_steps_executed: int,
        tail_time_ns: float,
    ) -> tuple[
        Trajectory, Trajectory, TrajectoryArrays | None, TrajectoryArrays | None
    ]:
        active_halt_step = max(0, step_index - active_start)
        reason_parts = [
            f"cavity_exit_reached species={exit_species}",
            f"exit_z={exit_z:.6f} mm",
            f"length={float(cavity_length_mm or 0.0):.6f} mm",
            f"at step {active_halt_step}/{requested_steps}",
        ]
        if tail_steps_planned > 0:
            reason_parts.append(
                f"tail_steps={tail_steps_executed}/{tail_steps_planned}"
            )
        reason = " ".join(reason_parts)

        trajectory_truncated = trajectory[: step_index + 1]
        trajectory_drv_truncated = trajectory_drv[: step_index + 1]
        _traj_builder.set_halt_metadata(
            step=step_index,
            reason=reason,
            halt_step=active_halt_step,
            requested_steps=requested_steps,
        )
        _traj_soa = _traj_builder.build()
        _traj_drv_soa = (
            _traj_drv_builder.build() if _traj_drv_builder is not None else None
        )
        if _legacy_history_compacted:
            trajectory_truncated = _traj_soa.to_legacy()[: step_index + 1]
            trajectory_drv_truncated = (
                _traj_drv_soa.to_legacy()[: step_index + 1]
                if _traj_drv_soa is not None
                else []
            )

        last_state = trajectory_truncated[-1]
        last_state["_halted_early"] = True
        last_state["_halt_reason"] = reason
        last_state["_halt_step"] = active_halt_step
        last_state["_requested_steps"] = requested_steps
        last_state["_termination_reason"] = "cavity_exit_reached"
        last_state["_exit_species"] = exit_species
        last_state["_exit_step"] = active_halt_step
        last_state["_exit_time_ns"] = exit_time_ns
        last_state["_cavity_length_mm"] = cavity_length_mm
        last_state["_rider_exit_z"] = cavity_rider_exit_z
        last_state["_driver_exit_z"] = cavity_driver_exit_z
        last_state["_residual_tail_steps"] = tail_steps_executed
        last_state["_residual_tail_steps_planned"] = tail_steps_planned
        last_state["_residual_tail_time_ns"] = tail_time_ns
        if driver_train_bunch_ranges:
            last_state["_driver_train_bunch_count"] = len(driver_train_bunch_ranges)
            last_state["_driver_train_muted_bunch_count"] = int(sum(driver_bunch_muted))
            last_state["_driver_train_active_bunch_count"] = int(
                len(driver_train_bunch_ranges) - sum(driver_bunch_muted)
            )

        if trajectory_drv_truncated:
            drv_last = trajectory_drv_truncated[-1]
            drv_last["_halted_early"] = True
            drv_last["_halt_reason"] = reason
            drv_last["_halt_step"] = active_halt_step
            drv_last["_requested_steps"] = requested_steps
            drv_last["_termination_reason"] = "cavity_exit_reached"
            drv_last["_exit_species"] = exit_species
            drv_last["_exit_step"] = active_halt_step
            drv_last["_exit_time_ns"] = exit_time_ns
            drv_last["_cavity_length_mm"] = cavity_length_mm
            drv_last["_rider_exit_z"] = cavity_rider_exit_z
            drv_last["_driver_exit_z"] = cavity_driver_exit_z
            drv_last["_residual_tail_steps"] = tail_steps_executed
            drv_last["_residual_tail_steps_planned"] = tail_steps_planned
            drv_last["_residual_tail_time_ns"] = tail_time_ns
            if driver_train_bunch_ranges:
                drv_last["_driver_train_bunch_count"] = len(driver_train_bunch_ranges)
                drv_last["_driver_train_muted_bunch_count"] = int(
                    sum(driver_bunch_muted)
                )
                drv_last["_driver_train_active_bunch_count"] = int(
                    len(driver_train_bunch_ranges) - sum(driver_bunch_muted)
                )
            drv_last["_cavity_exit_tail_mode"] = (
                "coasted" if tail_steps_planned > 0 else "none"
            )

        return _to_public_return(
            trajectory_truncated,
            trajectory_drv_truncated,
            _traj_soa,
            _traj_drv_soa,
        )

    def _estimate_driver_residual_tail_steps(rider_state: ParticleState) -> int:
        if (
            cavity_exit.residual_tail_factor <= 0.0
            or cavity_exit.max_residual_tail_steps <= 0
            or cavity_length_mm is None
        ):
            return 0
        rider_gamma = max(1.0, _state_mean(rider_state, "gamma", 1.0))
        rider_beta = min(0.999999, abs(_state_mean(rider_state, "bz", 0.0)))
        tail_time_ns = (
            float(cavity_exit.residual_tail_factor)
            * float(cavity_length_mm)
            / C_MMNS
            / max(1e-6, 1.0 - rider_beta)
        )
        tail_step_estimate = int(
            np.ceil(tail_time_ns / max(1e-12, rider_gamma * h_step))
        )
        return max(
            1,
            min(int(cavity_exit.max_residual_tail_steps), tail_step_estimate),
        )

    def _apply_driver_bunch_tail_cutoff(step_index: int) -> bool:
        if (
            not rider_exit_controls_global_halt
            or not driver_train_bunch_ranges
            or cavity_driver_exit_z is None
            or step_index <= active_start
        ):
            return False

        previous_driver_state = _latest_populated_state(trajectory_drv, step_index)
        current_driver_state = trajectory_drv[step_index]
        current_rider_state = trajectory[step_index]
        changed = False
        for bunch_index, bunch_slice in enumerate(driver_train_bunch_ranges):
            if driver_bunch_muted[bunch_index]:
                continue
            if driver_bunch_exit_steps[bunch_index] is None:
                previous_z = _leading_edge_z_for_slice(
                    previous_driver_state,
                    bunch_slice,
                )
                current_z = _leading_edge_z_for_slice(current_driver_state, bunch_slice)
                if _crossed_exit(previous_z, current_z, cavity_driver_exit_z):
                    driver_bunch_exit_steps[bunch_index] = step_index
                    driver_bunch_tail_steps_planned[bunch_index] = (
                        _estimate_driver_residual_tail_steps(current_rider_state)
                    )
            exit_step = driver_bunch_exit_steps[bunch_index]
            if exit_step is None:
                continue
            if step_index - exit_step < driver_bunch_tail_steps_planned[bunch_index]:
                continue

            particle_indices = range(
                int(bunch_slice.start or 0),
                int(bunch_slice.stop or 0),
            )
            if "_dead_particles" not in current_driver_state:
                current_driver_state["_dead_particles"] = np.zeros(
                    len(current_driver_state.get("gamma", [])),
                    dtype=bool,
                )
            dead_mask = current_driver_state["_dead_particles"]
            for particle_idx in particle_indices:
                dead_mask[particle_idx] = True
            if "q" in current_driver_state:
                current_driver_state["q"][bunch_slice] = 0.0
                if "q_source" in current_driver_state:
                    current_driver_state["q_source"][bunch_slice] = 0.0
            elif "stripped_ions" in current_driver_state:
                current_driver_state["stripped_ions"][bunch_slice] = 0.0
            current_driver_state["_driver_train_muted_bunch_count"] = int(
                sum(driver_bunch_muted) + 1
            )
            current_driver_state["_driver_train_bunch_count"] = len(
                driver_train_bunch_ranges
            )
            current_driver_state["_driver_train_last_muted_bunch_index"] = bunch_index
            current_driver_state["_driver_train_last_muted_exit_step"] = int(exit_step)
            current_driver_state["_driver_train_last_muted_tail_steps"] = int(
                driver_bunch_tail_steps_planned[bunch_index]
            )
            driver_bunch_muted[bunch_index] = True
            changed = True
        if changed:
            current_driver_state["_driver_train_muted_bunch_count"] = int(
                sum(driver_bunch_muted)
            )
            current_driver_state["_driver_train_active_bunch_count"] = int(
                len(driver_train_bunch_ranges) - sum(driver_bunch_muted)
            )
        return changed

    if progress_callback is not None:
        progress_callback(0, requested_steps)

    def _to_public_return(
        rider_traj: Trajectory,
        driver_traj: Trajectory,
        rider_soa: TrajectoryArrays | None,
        driver_soa: TrajectoryArrays | None,
    ) -> Tuple[
        Trajectory, Trajectory, TrajectoryArrays | None, TrajectoryArrays | None
    ]:
        if not driver_train_enabled or driver_train.preserve_prehistory_in_output:
            return rider_traj, driver_traj, rider_soa, driver_soa
        start = min(active_start, len(rider_traj))
        stop = len(rider_traj)
        return (
            rider_traj[start:stop],
            driver_traj[start:stop],
            _slice_trajectory_arrays(rider_soa, start, stop),
            _slice_trajectory_arrays(driver_soa, start, stop),
        )

    _adaptive_state = _AdaptiveStepState(current_h_step=h_step, reduced_h_step=h_step)
    for i in range(total_steps):
        if cancel_callback is not None and cancel_callback():
            raise IntegrationCancelled("Integration cancelled by caller.")
        if i <= active_start:
            trajectory[i] = (
                rider_seed_history[i] if driver_train_enabled else init_rider
            )
            _ensure_startup_metadata(trajectory[i])
            _set_pseudo_grid_schedule_metadata(trajectory[i], None)
            _traj_builder.set_step(i, trajectory[i])
            if sim_type == SimulationType.CONDUCTING_WALL:
                trajectory_drv[i] = generate_conducting_image(
                    trajectory[i],
                    wall_z,
                    aperture_radius,
                    subcharge_count=image_subcharge_count,
                    use_weighting=use_conducting_image_weighting,
                    macroparticle_charge_multiplier=macroparticle_charge_multiplier,
                    macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
                    macroparticle_use_momentum_errors=macroparticle_use_momentum_errors,
                    bunch_transv_dist=bunch_transv_dist,
                    bunch_transv_mom=bunch_transv_mom,
                    timestep=h_step,
                    step_number=i,
                )
            elif sim_type == SimulationType.SWITCHING_WALL:
                trajectory_drv[i] = generate_switching_image(
                    trajectory[i], wall_z, aperture_radius, z_cutoff
                )
            elif sim_type == SimulationType.BUNCH_TO_BUNCH:
                if init_driver is None:
                    raise ValueError(
                        "SimulationType.BUNCH_TO_BUNCH requires init_driver state"
                    )
                if driver_train_enabled and driver_seed_history is not None:
                    trajectory_drv[i] = driver_seed_history[i]
                else:
                    trajectory_drv[i] = init_driver
            _ensure_startup_metadata(trajectory_drv[i])
            _set_pseudo_grid_schedule_metadata(trajectory_drv[i], None)
            _n_particles_drv = len(trajectory_drv[i]["x"])
            if _traj_drv_builder is None:
                _traj_drv_builder = TrajectoryBuilder(total_steps, _n_particles_drv)
            _traj_drv_builder.set_step(i, trajectory_drv[i])
            if pseudo_grid.enabled and i == active_start:
                _pseudo_grid_planner_state = initialize_pseudo_grid_planner_state(
                    rider_particle_count=_n_particles_rider,
                    driver_particle_count=_n_particles_drv,
                    pair_reuse_window=pseudo_grid.pair_reuse_window,
                )
                record_pseudo_grid_history_times(
                    _pseudo_grid_planner_state,
                    trajectory[i],
                    trajectory_drv[i],
                )
                if _pseudo_grid_retention_active:
                    _rider_retained_history = [trajectory[i]]
                    _driver_retained_history = [trajectory_drv[i]]
            if i < active_start:
                continue
        else:
            _current_pseudo_grid_schedule = None
            if pseudo_grid.enabled:
                if _pseudo_grid_planner_state is None:
                    raise RuntimeError(
                        "Pseudo-grid planner state was not initialized at step 0."
                    )
                _current_pseudo_grid_schedule = build_pseudo_grid_step_schedule(
                    trajectory[i - 1],
                    trajectory_drv[i - 1],
                    step_index=i,
                    config=pseudo_grid,
                    planner_state=_pseudo_grid_planner_state,
                )

            trajectory[i] = _run_adaptive_step(
                i=i,
                steps=total_steps,
                h_step=h_step,
                wall_z=wall_z,
                aperture_radius=aperture_radius,
                sim_type=sim_type,
                chrono_mode=chrono_mode,
                startup_mode=startup_mode,
                self_consistency=self_consistency,
                adaptive_timestep=adaptive_timestep,
                space_charge=space_charge,
                external_field=external_field,
                image_subcharge_count=image_subcharge_count,
                use_conducting_image_weighting=use_conducting_image_weighting,
                macroparticle_charge_multiplier=macroparticle_charge_multiplier,
                macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
                macroparticle_use_momentum_errors=macroparticle_use_momentum_errors,
                bunch_transv_dist=bunch_transv_dist,
                bunch_transv_mom=bunch_transv_mom,
                z_cutoff=z_cutoff,
                trajectory=trajectory,
                trajectory_drv=trajectory_drv,
                _traj_drv_builder=_traj_drv_builder,
                cancel_callback=cancel_callback,
                logger=logger,
                adaptive_state=_adaptive_state,
                radiation_reaction_mode=radiation_reaction_mode,
                pseudo_grid_schedule=_current_pseudo_grid_schedule,
                pseudo_grid_force_reduction_enabled=pseudo_grid_force_reduction_enabled,
                pseudo_grid_weighting_mode=pseudo_grid.source_weighting_mode,
                pseudo_grid_loss_tracking_enabled=pseudo_grid.loss_tracking_enabled,
                pseudo_grid_observer_history=(
                    _rider_retained_history if _pseudo_grid_retention_active else None
                ),
                pseudo_grid_source_history=(
                    _driver_retained_history if _pseudo_grid_retention_active else None
                ),
                pseudo_grid_observer_history_base_index=(
                    _rider_retained_history_start_index
                    if _pseudo_grid_retention_active
                    else 0
                ),
                pseudo_grid_source_history_base_index=(
                    _driver_retained_history_start_index
                    if _pseudo_grid_retention_active
                    else 0
                ),
                pseudo_grid_observer_soa=_traj_builder.build_partial(i),
                pseudo_grid_source_soa=(
                    _traj_drv_builder.build_partial(i)
                    if _traj_drv_builder is not None
                    else None
                ),
                use_full_history=(
                    driver_train_enabled or sim_type == SimulationType.BUNCH_TO_BUNCH
                ),
                macroparticle_smearing=macroparticle_smearing,
            )
            _ensure_startup_metadata(trajectory[i])
            _set_pseudo_grid_schedule_metadata(
                trajectory[i], _current_pseudo_grid_schedule
            )
            if particle_loss.enabled:
                mark_particle_losses(
                    trajectory[i],
                    trajectory[i - 1],
                    step=i,
                    config=particle_loss,
                    context=_rider_loss_context,
                    sim_type=sim_type,
                    wall_z=wall_z,
                    aperture_radius=aperture_radius,
                    logger=logger,
                )
            _mark_post_step_gamma_blowups(trajectory[i], step=i, logger=logger)
            _traj_builder.set_step(i, trajectory[i])

            # Log alive/dead particle counts after post-step checks
            dead_mask = trajectory[i].get("_dead_particles")
            if dead_mask is not None and np.any(dead_mask):
                num_alive = np.sum(~dead_mask)
                num_dead = np.sum(dead_mask)
                num_total = len(dead_mask)
                if num_dead > 0:
                    print(
                        f"  [STATUS] Step {i}: {num_alive}/{num_total} particles alive, "
                        f"{num_dead}/{num_total} dead"
                    )

            # Check if all particles are dead
            if all_particles_dead(trajectory[i]):
                # Get failure summary
                failure_info = get_particle_failure_summary(trajectory)
                failure_summary = format_failure_summary(failure_info)
                num_dead = len(failure_info)

                msg1 = (
                    f"[CRITICAL] Step {i}: All {num_dead} particles have failed. "
                    f"Halting integration."
                )
                msg2 = f"  {failure_summary}"
                if logger:
                    logger(msg1)
                    logger(msg2)
                else:
                    print(msg1)
                    print(msg2)

                # Truncate trajectory and mark as halted
                trajectory = trajectory[: i + 1]
                trajectory_drv = trajectory_drv[: i + 1]

                # Store halt information
                trajectory[-1]["_halted_early"] = True
                active_halt_step = max(0, i - active_start)
                trajectory[-1][
                    "_halt_reason"
                ] = f"all_particles_dead at step {active_halt_step}/{requested_steps}. {failure_summary}"
                trajectory[-1]["_halt_step"] = active_halt_step
                trajectory[-1]["_requested_steps"] = requested_steps
                _traj_builder.set_halt_metadata(
                    step=i,
                    reason=(
                        f"all_particles_dead at step {active_halt_step}/{requested_steps}. "
                        f"{failure_summary}"
                    ),
                    halt_step=active_halt_step,
                    requested_steps=requested_steps,
                )
                _traj_soa = _traj_builder.build()
                _traj_drv_soa = (
                    _traj_drv_builder.build() if _traj_drv_builder is not None else None
                )
                if _legacy_history_compacted:
                    trajectory = _traj_soa.to_legacy()[: i + 1]
                    trajectory_drv = (
                        _traj_drv_soa.to_legacy()[: i + 1]
                        if _traj_drv_soa is not None
                        else []
                    )
                return _to_public_return(
                    trajectory,
                    trajectory_drv,
                    _traj_soa,
                    _traj_drv_soa,
                )

            if sim_type == SimulationType.SWITCHING_WALL:
                trajectory_drv[i] = generate_switching_image(
                    trajectory[i], wall_z, aperture_radius, z_cutoff
                )
                if np.mean(trajectory[i]["z"]) > z_cutoff:
                    z_cutoff += cav_spacing
                    wall_z += cav_spacing
            elif sim_type == SimulationType.CONDUCTING_WALL:
                trajectory_drv[i] = generate_conducting_image(
                    trajectory[i],
                    wall_z,
                    aperture_radius,
                    subcharge_count=image_subcharge_count,
                    use_weighting=use_conducting_image_weighting,
                    macroparticle_charge_multiplier=macroparticle_charge_multiplier,
                    macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
                    macroparticle_use_momentum_errors=macroparticle_use_momentum_errors,
                    bunch_transv_dist=bunch_transv_dist,
                    bunch_transv_mom=bunch_transv_mom,
                    timestep=_adaptive_state.current_h_step,
                    step_number=i,
                )
            elif sim_type == SimulationType.BUNCH_TO_BUNCH:
                if init_driver is None:
                    raise ValueError(
                        "SimulationType.BUNCH_TO_BUNCH requires init_driver state"
                    )
                if cavity_exit_triggered and cavity_exit_exit_species == "driver":
                    trajectory_drv[i] = _coast_state_by_proper_steps(
                        trajectory_drv[i - 1],
                        h_step,
                        1,
                    )
                    trajectory_drv[i]["_cavity_exit_tail_mode"] = "coasted"
                    trajectory_drv[i][
                        "_cavity_exit_tail_steps_planned"
                    ] = cavity_exit_tail_steps_planned
                    trajectory_drv[i]["_cavity_exit_tail_step"] = i - int(
                        cavity_exit_exit_index or i
                    )
                elif (
                    pseudo_grid_force_reduction_enabled
                    and _current_pseudo_grid_schedule is not None
                ):
                    trajectory_drv[i] = _run_pseudo_grid_reduced_step(
                        h_step=h_step,
                        observer_history=(
                            _driver_retained_history
                            if _pseudo_grid_retention_active
                            else trajectory_drv[:i]
                        ),
                        source_history=(
                            _rider_retained_history
                            if _pseudo_grid_retention_active
                            else trajectory[:i]
                        ),
                        observer_active_indices=_current_pseudo_grid_schedule.driver_active_indices,
                        source_active_indices=_current_pseudo_grid_schedule.rider_active_indices,
                        source_effective_charges=_current_pseudo_grid_schedule.rider_effective_source_charges,
                        source_history_start_index=(
                            _current_pseudo_grid_schedule.rider_history_start_index
                        ),
                        passive_map=_current_pseudo_grid_schedule.driver_passive_map,
                        aperture_radius=aperture_radius,
                        sim_type=sim_type,
                        self_consistency=self_consistency,
                        chrono_mode=chrono_mode,
                        startup_mode=startup_mode,
                        step_idx=i,
                        cancel_callback=cancel_callback,
                        logger=logger,
                        radiation_reaction_mode=radiation_reaction_mode,
                        external_field=external_field,
                        space_charge=space_charge,
                        pseudo_grid_weighting_mode=pseudo_grid.source_weighting_mode,
                        loss_tracking_enabled=pseudo_grid.loss_tracking_enabled,
                        source_history_base_index=(
                            _rider_retained_history_start_index
                            if _pseudo_grid_retention_active
                            else 0
                        ),
                        observer_history_base_index=(
                            _driver_retained_history_start_index
                            if _pseudo_grid_retention_active
                            else 0
                        ),
                        observer_soa=(
                            _traj_drv_builder.build_partial(i)
                            if _traj_drv_builder is not None
                            else None
                        ),
                        source_soa=_traj_builder.build_partial(i),
                        macroparticle_smearing=macroparticle_smearing,
                    )
                else:
                    _b2b_scs_accepts_soa = _call_accepts_kw(
                        self_consistent_step, "traj_soa"
                    )
                    _b2b_scs_accepts_radiation = _call_accepts_kw(
                        self_consistent_step, "radiation_reaction_mode"
                    )
                    _b2b_scs_accepts_external_field = _call_accepts_kw(
                        self_consistent_step, "external_field"
                    )
                    trajectory_drv[i] = self_consistent_step(
                        retarded_equations_of_motion,
                        h_step,
                        trajectory_drv,
                        trajectory,
                        i - 1,
                        aperture_radius,
                        sim_type,
                        self_consistency,
                        chrono_mode,
                        startup_mode,
                        step_idx=i,
                        **(
                            {"radiation_reaction_mode": radiation_reaction_mode}
                            if _b2b_scs_accepts_radiation
                            else {}
                        ),
                        **(
                            {"space_charge": space_charge}
                            if space_charge is not None and not driver_train_enabled
                            else {}
                        ),
                        **(
                            {"external_field": external_field}
                            if external_field is not None
                            and _b2b_scs_accepts_external_field
                            else {}
                        ),
                        **(
                            {"traj_soa": _traj_drv_builder.build_partial(i)}
                            if _b2b_scs_accepts_soa and _traj_drv_builder is not None
                            else {}
                        ),
                        **(
                            {"traj_ext_soa": _traj_builder.build_partial(i)}
                            if _b2b_scs_accepts_soa
                            else {}
                        ),
                        macroparticle_smearing=macroparticle_smearing,
                    )
            _ensure_startup_metadata(trajectory_drv[i])
            _set_pseudo_grid_schedule_metadata(trajectory_drv[i], None)
            if (
                particle_loss.enabled
                and sim_type == SimulationType.BUNCH_TO_BUNCH
                and _driver_loss_context is not None
            ):
                mark_particle_losses(
                    trajectory_drv[i],
                    trajectory_drv[i - 1],
                    step=i,
                    config=particle_loss,
                    context=_driver_loss_context,
                    sim_type=sim_type,
                    wall_z=wall_z,
                    aperture_radius=aperture_radius,
                    logger=logger,
                )
            if _traj_drv_builder is not None:
                _traj_drv_builder.set_step(i, trajectory_drv[i])
            if (
                pseudo_grid.enabled
                and _pseudo_grid_planner_state is not None
                and _current_pseudo_grid_schedule is not None
            ):
                commit_pseudo_grid_step_schedule(
                    _pseudo_grid_planner_state,
                    _current_pseudo_grid_schedule,
                )
                record_pseudo_grid_history_times(
                    _pseudo_grid_planner_state,
                    trajectory[i],
                    trajectory_drv[i],
                )
                if _pseudo_grid_retention_active:
                    _rider_retained_history.append(trajectory[i])
                    _driver_retained_history.append(trajectory_drv[i])
                    (
                        _driver_retained_history_start_index,
                        driver_dropped_samples,
                    ) = _drop_retained_history_prefix(
                        _driver_retained_history,
                        _driver_retained_history_start_index,
                        _current_pseudo_grid_schedule.driver_history_start_index,
                    )
                    (
                        _rider_retained_history_start_index,
                        rider_dropped_samples,
                    ) = _drop_retained_history_prefix(
                        _rider_retained_history,
                        _rider_retained_history_start_index,
                        _current_pseudo_grid_schedule.rider_history_start_index,
                    )
                    _current_pseudo_grid_schedule = replace(
                        _current_pseudo_grid_schedule,
                        driver_retained_history_start_index=_driver_retained_history_start_index,
                        rider_retained_history_start_index=_rider_retained_history_start_index,
                        driver_dropped_history_samples=driver_dropped_samples,
                        rider_dropped_history_samples=rider_dropped_samples,
                    )
                    _set_pseudo_grid_schedule_metadata(
                        trajectory[i],
                        _current_pseudo_grid_schedule,
                    )
                    _traj_builder.set_step(i, trajectory[i])
                    previous_rider_cleared_until = _rider_legacy_history_cleared_until
                    previous_driver_cleared_until = _driver_legacy_history_cleared_until
                    _rider_legacy_history_cleared_until = _clear_legacy_history_prefix(
                        trajectory,
                        start_index=_rider_legacy_history_cleared_until,
                        end_index=_rider_retained_history_start_index,
                    )
                    _driver_legacy_history_cleared_until = _clear_legacy_history_prefix(
                        trajectory_drv,
                        start_index=_driver_legacy_history_cleared_until,
                        end_index=_driver_retained_history_start_index,
                    )
                    _legacy_history_compacted = _legacy_history_compacted or (
                        _rider_legacy_history_cleared_until
                        > previous_rider_cleared_until
                        or _driver_legacy_history_cleared_until
                        > previous_driver_cleared_until
                    )

            if _apply_driver_bunch_tail_cutoff(i) and _traj_drv_builder is not None:
                _traj_drv_builder.set_step(i, trajectory_drv[i])

        # Check for early termination when a configured cavity exit is reached.
        if (
            cavity_exit_enabled
            and not cavity_exit_triggered
            and sim_type == SimulationType.BUNCH_TO_BUNCH
            and i > active_start
            and init_driver is not None
            and cavity_rider_exit_z is not None
            and cavity_driver_exit_z is not None
            and cavity_length_mm is not None
        ):
            rider_previous_z = _leading_edge_z(_latest_populated_state(trajectory, i))
            rider_current_z = _leading_edge_z(trajectory[i])
            driver_previous_z = _leading_edge_z(
                _latest_populated_state(trajectory_drv, i)
            )
            driver_current_z = _leading_edge_z(trajectory_drv[i])
            rider_exited = _crossed_exit(
                rider_previous_z, rider_current_z, cavity_rider_exit_z
            )
            driver_exited = _crossed_exit(
                driver_previous_z, driver_current_z, cavity_driver_exit_z
            )
            if rider_exited or (driver_exited and not rider_exit_controls_global_halt):
                if rider_exit_controls_global_halt:
                    exit_species = "rider"
                elif rider_exited and driver_exited:
                    rider_overshoot = abs(rider_current_z - cavity_rider_exit_z)
                    driver_overshoot = abs(driver_current_z - cavity_driver_exit_z)
                    exit_species = (
                        "rider" if rider_overshoot <= driver_overshoot else "driver"
                    )
                else:
                    exit_species = "rider" if rider_exited else "driver"
                exit_z = (
                    cavity_rider_exit_z
                    if exit_species == "rider"
                    else cavity_driver_exit_z
                )
                exit_time_ns = _state_mean(trajectory[i], "t")
                active_halt_step = max(0, i - active_start)
                if (
                    exit_species == "driver"
                    and cavity_exit.residual_tail_factor > 0.0
                    and cavity_exit.max_residual_tail_steps > 0
                    and cavity_length_mm is not None
                ):
                    cavity_exit_tail_steps_planned = (
                        _estimate_driver_residual_tail_steps(trajectory[i])
                    )
                    rider_beta = min(
                        0.999999, abs(_state_mean(trajectory[i], "bz", 0.0))
                    )
                    cavity_exit_tail_time_ns = (
                        float(cavity_exit.residual_tail_factor)
                        * float(cavity_length_mm)
                        / C_MMNS
                        / max(1e-6, 1.0 - rider_beta)
                    )
                    cavity_exit_triggered = True
                    cavity_exit_exit_index = i
                    cavity_exit_exit_species = exit_species
                    cavity_exit_exit_z = exit_z
                    cavity_exit_exit_time_ns = exit_time_ns
                    cavity_exit_tail_stop_index = i + cavity_exit_tail_steps_planned
                    trajectory[i]["_cavity_exit_tail_mode"] = "coasted"
                    trajectory[i][
                        "_cavity_exit_tail_steps_planned"
                    ] = cavity_exit_tail_steps_planned
                    trajectory_drv[i]["_cavity_exit_tail_mode"] = "coasted"
                    trajectory_drv[i][
                        "_cavity_exit_tail_steps_planned"
                    ] = cavity_exit_tail_steps_planned
                else:
                    return _finalize_cavity_exit_halt(
                        step_index=i,
                        exit_species=exit_species,
                        exit_z=exit_z,
                        exit_time_ns=exit_time_ns,
                        tail_steps_planned=0,
                        tail_steps_executed=0,
                        tail_time_ns=0.0,
                    )

        # Check for early termination in BUNCH_TO_BUNCH relative mode
        if (
            z_cutoff_mode == "relative"
            and sim_type == SimulationType.BUNCH_TO_BUNCH
            and i >= active_start
            and z_initial is not None
            and z_cutoff > 0
        ):
            z_current = float(np.mean(trajectory[i]["z"]))
            distance_traveled = abs(z_current - z_initial)
            if distance_traveled > z_cutoff:
                # Truncate trajectory and mark as halted
                if adaptive_timestep is not None and adaptive_timestep.debug:
                    print(
                        f"Step {i}: BUNCH_TO_BUNCH relative cutoff reached. "
                        f"Traveled {distance_traveled:.2f} mm > {z_cutoff:.2f} mm cutoff. "
                        f"Stopping integration early."
                    )
                trajectory_truncated = trajectory[: i + 1]
                trajectory_drv_truncated = trajectory_drv[: i + 1]
                # Store halt information in the last particle's metadata
                trajectory_truncated[-1]["_halted_early"] = True
                active_halt_step = max(0, i - active_start)
                trajectory_truncated[-1][
                    "_halt_reason"
                ] = f"distance_reached ({distance_traveled:.2f} mm > {z_cutoff:.2f} mm at step {active_halt_step}/{requested_steps})"
                trajectory_truncated[-1]["_halt_step"] = active_halt_step
                trajectory_truncated[-1]["_requested_steps"] = requested_steps
                _traj_builder.set_halt_metadata(
                    step=i,
                    reason=(
                        f"distance_reached ({distance_traveled:.2f} mm > {z_cutoff:.2f} mm "
                        f"at step {active_halt_step}/{requested_steps})"
                    ),
                    halt_step=active_halt_step,
                    requested_steps=requested_steps,
                )
                _traj_soa = _traj_builder.build()
                _traj_drv_soa = (
                    _traj_drv_builder.build() if _traj_drv_builder is not None else None
                )
                if _legacy_history_compacted:
                    trajectory_truncated = _traj_soa.to_legacy()[: i + 1]
                    trajectory_drv_truncated = (
                        _traj_drv_soa.to_legacy()[: i + 1]
                        if _traj_drv_soa is not None
                        else []
                    )
                return _to_public_return(
                    trajectory_truncated,
                    trajectory_drv_truncated,
                    _traj_soa,
                    _traj_drv_soa,
                )

        # Energy monitoring (for warning/halting, separate from adaptive timestep)
        if (
            energy_monitor is not None
            and energy_monitor.enabled
            and i > 0
            and i % energy_monitor.check_interval == 0
        ):
            current_energy = _compute_total_energy(trajectory[i])
            if previous_energy is not None and previous_energy > 0:
                relative_change = (
                    abs(current_energy - previous_energy) / previous_energy
                )
                if relative_change > energy_monitor.relative_threshold:
                    msg = (
                        f"Energy jump detected at step {i}/{requested_steps}: "
                        f"ΔE/E = {relative_change:.2e} "
                        f"(threshold = {energy_monitor.relative_threshold:.2e})"
                    )
                    if energy_monitor.halt_on_jump:
                        raise EnergyJumpDetected(msg)
                    else:
                        print(f"WARNING: {msg}")
                elif energy_monitor.debug:
                    print(
                        f"Step {i}: Energy = {current_energy:.6e} MeV, ΔE/E = {relative_change:.6e}"
                    )
            previous_energy = current_energy

        if progress_callback is not None and i >= active_start:
            progress_callback(
                min(i - active_start + 1, requested_steps),
                requested_steps,
            )

        if (
            cavity_exit_triggered
            and cavity_exit_exit_species == "driver"
            and cavity_exit_tail_stop_index is not None
            and i >= cavity_exit_tail_stop_index
        ):
            cavity_exit_tail_steps_executed = max(
                0, i - int(cavity_exit_exit_index or i)
            )
            return _finalize_cavity_exit_halt(
                step_index=i,
                exit_species=str(cavity_exit_exit_species or "driver"),
                exit_z=float(cavity_exit_exit_z or 0.0),
                exit_time_ns=float(cavity_exit_exit_time_ns or 0.0),
                tail_steps_planned=cavity_exit_tail_steps_planned,
                tail_steps_executed=cavity_exit_tail_steps_executed,
                tail_time_ns=cavity_exit_tail_time_ns,
            )

    if cavity_exit_triggered and cavity_exit_exit_species == "driver":
        final_step_index = len(trajectory) - 1
        cavity_exit_tail_steps_executed = max(
            0, final_step_index - int(cavity_exit_exit_index or final_step_index)
        )
        return _finalize_cavity_exit_halt(
            step_index=final_step_index,
            exit_species=str(cavity_exit_exit_species or "driver"),
            exit_z=float(cavity_exit_exit_z or 0.0),
            exit_time_ns=float(cavity_exit_exit_time_ns or 0.0),
            tail_steps_planned=cavity_exit_tail_steps_planned,
            tail_steps_executed=cavity_exit_tail_steps_executed,
            tail_time_ns=cavity_exit_tail_time_ns,
        )

    _traj_soa = _traj_builder.build()
    _traj_drv_soa = _traj_drv_builder.build() if _traj_drv_builder is not None else None
    if _legacy_history_compacted:
        trajectory = _traj_soa.to_legacy()
        trajectory_drv = _traj_drv_soa.to_legacy() if _traj_drv_soa is not None else []
    return _to_public_return(trajectory, trajectory_drv, _traj_soa, _traj_drv_soa)


def run_integrator(
    config: IntegratorConfig,
    init_rider: ParticleState,
    init_driver: Optional[ParticleState],
    self_consistency: Optional[SelfConsistencyConfig] = None,
    energy_monitor: Optional[EnergyMonitorConfig] = None,
    adaptive_timestep: Optional[AdaptiveTimestepConfig] = None,
    space_charge: Optional[Any] = None,
    external_field: Optional[Any] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Tuple[
    Trajectory, Trajectory, "TrajectoryArrays | None", "TrajectoryArrays | None"
]:
    """Convenience wrapper using :class:`IntegratorConfig`.

    All parameters are supplied via ``config`` which mirrors the keyword
    arguments accepted by :func:`retarded_integrator`.

    Parameters
    ----------
    config : IntegratorConfig
        Structured configuration for the integrator
    init_rider : ParticleState
        Initial state of the primary bunch
    init_driver : Optional[ParticleState]
        Optional initial state of the opposing bunch
    self_consistency : Optional[SelfConsistencyConfig]
        Optional self-consistency configuration
    energy_monitor : Optional[EnergyMonitorConfig]
        Optional energy monitoring configuration
    adaptive_timestep : Optional[AdaptiveTimestepConfig]
        Optional adaptive timestep configuration
    progress_callback : Optional[Callable[[int, int], None]]
        Optional progress callback
    cancel_callback : Optional[Callable[[], bool]]
        Optional cancellation callback

    Returns
    -------
    Tuple[Trajectory, Trajectory, TrajectoryArrays | None, TrajectoryArrays | None]
        Rider and driver trajectories plus optional SOA views.
    """

    return retarded_integrator(
        steps=config.steps,
        h_step=config.time_step,
        wall_z=config.wall_position,
        aperture_radius=config.aperture_radius,
        sim_type=config.simulation_type,
        init_rider=init_rider,
        init_driver=init_driver,
        mean=config.bunch_mean,
        cav_spacing=config.cavity_spacing,
        z_cutoff=config.z_cutoff,
        z_cutoff_mode=config.z_cutoff_mode,
        self_consistency=self_consistency,
        chrono_mode=config.chrono_mode,
        startup_mode=config.startup_mode,
        image_subcharge_count=config.image_subcharge_count,
        use_conducting_image_weighting=config.use_image_weighting,
        radiation_reaction_mode=config.radiation_reaction_mode,
        macroparticle_charge_multiplier=config.macroparticle_charge_multiplier,
        macroparticle_sigma_multiplier=config.macroparticle_sigma_multiplier,
        macroparticle_use_momentum_errors=config.macroparticle_use_momentum_errors,
        bunch_transv_dist=config.bunch_transv_dist,
        bunch_transv_mom=config.bunch_transv_mom,
        energy_monitor=energy_monitor,
        adaptive_timestep=adaptive_timestep,
        space_charge=space_charge,
        external_field=external_field,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        pseudo_grid=config.pseudo_grid,
        driver_train=config.driver_train,
        cavity_exit=config.cavity_exit,
        particle_loss=config.particle_loss,
        macroparticle_smearing=config.macroparticle_smearing,
    )


__all__ = [
    "IntegrationCancelled",
    "EnergyJumpDetected",
    "EnergyMonitorConfig",
    "AdaptiveTimestepConfig",
    "retarded_integrator",
    "run_integrator",
]
