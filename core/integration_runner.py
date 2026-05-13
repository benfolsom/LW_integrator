"""High-level orchestration for retarded-field trajectory integration.

This module coordinates the low-level physics kernels, image-charge
construction, and optional self-consistency loops.  It provides the primary
programmatic entry points for running the modern Liénard–Wiechert integrator."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

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
from .self_consistency import SelfConsistencyConfig, self_consistent_step
from .types import (
    ChronoMatchingMode,
    IntegratorConfig,
    ParticleState,
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
        if proximity_reduction_active and adaptive_timestep is not None:
            current_h_step = h_step / proximity_factor
            if adaptive_timestep.debug:
                msg = (
                    f"Step {i}: Applying proximity refinement: "
                    f"{h_step:.6e} \u2192 {current_h_step:.6e} ns"
                )
                if logger:
                    logger(msg)
                else:
                    print(msg)

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

        temp_trajectory = [
            {
                k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
                for k, v in temp_trajectory_base.items()
            }
        ]
        temp_driver = [trajectory_drv[i - 1]]

        _n_p_rider = len(temp_trajectory[0]["x"])
        _temp_traj_builder = TrajectoryBuilder(num_substeps + 1, _n_p_rider)
        _temp_traj_builder.set_step(0, temp_trajectory[0])
        _temp_drv_soa = _traj_drv_builder.build_partial(i) if _traj_drv_builder is not None else None
        _scs_accepts_soa = "traj_soa" in inspect.signature(self_consistent_step).parameters

        energy_jump_detected = False
        gamma_blowup_detected = False
        max_refinement_reached = False
        min_timestep_reached = False

        for substep_idx in range(num_substeps):
            if cancel_callback is not None and cancel_callback():
                raise IntegrationCancelled("Integration cancelled by caller.")

            try:
                trial_state = self_consistent_step(
                    retarded_equations_of_motion,
                    current_h_step,
                    temp_trajectory,
                    temp_driver,
                    substep_idx,
                    aperture_radius,
                    sim_type,
                    self_consistency,
                    chrono_mode,
                    startup_mode,
                    step_idx=i,
                    cancel_callback=cancel_callback,
                    radiation_reaction_mode=radiation_reaction_mode,
                    **({"space_charge": space_charge} if space_charge is not None else {}),
                    **({"traj_soa": _temp_traj_builder.build_partial(substep_idx + 1)} if _scs_accepts_soa else {}),
                    **({"traj_ext_soa": _temp_drv_soa} if _scs_accepts_soa else {}),
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
                        trial_state, e.particle_idx, i, "gamma_blowup_no_adaptive",
                        gamma_value=e.gamma_value, iteration=e.iteration,
                    )

                    temp_trajectory.append(trial_state)
                    _temp_traj_builder.set_step(len(temp_trajectory) - 1, trial_state)
                    temp_driver.append(temp_driver[-1] if temp_driver else trajectory_drv[i - 1])
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
                                k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
                                for k, v in temp_trajectory[-1].items()
                            }
                        else:
                            trial_state = {
                                k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
                                for k, v in trajectory[i - 1].items()
                            }

                        mark_particle_dead(
                            trial_state, e.particle_idx, i, "gamma_blowup_max_retries",
                            gamma_value=e.gamma_value, iteration=e.iteration,
                        )

                        temp_trajectory.append(trial_state)
                        _temp_traj_builder.set_step(len(temp_trajectory) - 1, trial_state)
                        temp_driver.append(temp_driver[-1] if temp_driver else trajectory_drv[i - 1])
                        last_particle_death_step = i
                        gamma_blowup_detected = False
                        break
                    else:
                        min_h = h_step * adaptive_timestep.min_timestep_factor
                        new_h_step = current_h_step / adaptive_timestep.timestep_reduction_factor

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
                                    k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
                                    for k, v in temp_trajectory[-1].items()
                                }
                            else:
                                trial_state = {
                                    k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
                                    for k, v in trajectory[i - 1].items()
                                }

                            mark_particle_dead(
                                trial_state, e.particle_idx, i, "gamma_blowup_min_timestep",
                                gamma_value=e.gamma_value, iteration=e.iteration,
                            )

                            temp_trajectory.append(trial_state)
                            _temp_traj_builder.set_step(len(temp_trajectory) - 1, trial_state)
                            temp_driver.append(temp_driver[-1] if temp_driver else trajectory_drv[i - 1])
                            last_particle_death_step = i
                            gamma_blowup_detected = False
                            break
                        else:
                            if hasattr(e, "is_hard_blowup") and e.is_hard_blowup:
                                reduction_factor = adaptive_timestep.timestep_reduction_factor ** 2
                                severity = "HARD"
                            else:
                                reduction_factor = adaptive_timestep.timestep_reduction_factor
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

                        if refinement_attempt > adaptive_timestep.max_refinement_attempts:
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
                            new_h_step = current_h_step / adaptive_timestep.timestep_reduction_factor

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
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
    logger: Optional[Any] = None,
    use_numba: bool = True,
    radiation_reaction_mode: str = "off",
) -> Tuple[Trajectory, Trajectory, "TrajectoryArrays | None", "TrajectoryArrays | None"]:
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
        momentum after the normal LW update.

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

    # Canonical integration implementation

    trajectory: Trajectory = [{} for _ in range(steps)]
    trajectory_drv: Trajectory = [{} for _ in range(steps)]
    _n_particles_rider = len(init_rider["x"])
    _n_particles_drv: int | None = None
    _traj_builder = TrajectoryBuilder(steps, _n_particles_rider)
    _traj_drv_builder: TrajectoryBuilder | None = None

    # Initialize energy monitoring
    previous_energy: Optional[float] = None

    # Store initial z position for relative cutoff mode
    z_initial: Optional[float] = None
    if z_cutoff_mode == "relative" and sim_type == SimulationType.BUNCH_TO_BUNCH:
        z_initial = float(np.mean(init_rider["z"]))
        if adaptive_timestep is not None and adaptive_timestep.debug:
            msg = (
                f"BUNCH_TO_BUNCH relative cutoff mode: z_initial = {z_initial:.6f} mm, "
                f"will stop after traveling {z_cutoff:.2f} mm"
            )
            if logger:
                logger(msg)
            else:
                print(msg)

    if progress_callback is not None:
        progress_callback(0, steps)

    _adaptive_state = _AdaptiveStepState(current_h_step=h_step, reduced_h_step=h_step)
    for i in range(steps):
        if cancel_callback is not None and cancel_callback():
            raise IntegrationCancelled("Integration cancelled by caller.")
        if i == 0:
            trajectory[i] = init_rider
            _ensure_startup_metadata(trajectory[i])
            _traj_builder.set_step(i, trajectory[i])
            if sim_type == SimulationType.CONDUCTING_WALL:
                trajectory_drv[i] = generate_conducting_image(
                    init_rider,
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
                    init_rider, wall_z, aperture_radius, z_cutoff
                )
            elif sim_type == SimulationType.BUNCH_TO_BUNCH:
                if init_driver is None:
                    raise ValueError(
                        "SimulationType.BUNCH_TO_BUNCH requires init_driver state"
                    )
                trajectory_drv[i] = init_driver
            _ensure_startup_metadata(trajectory_drv[i])
            _n_particles_drv = len(trajectory_drv[i]["x"])
            _traj_drv_builder = TrajectoryBuilder(steps, _n_particles_drv)
            _traj_drv_builder.set_step(i, trajectory_drv[i])
        else:
            trajectory[i] = _run_adaptive_step(
                i=i,
                steps=steps,
                h_step=h_step,
                wall_z=wall_z,
                aperture_radius=aperture_radius,
                sim_type=sim_type,
                chrono_mode=chrono_mode,
                startup_mode=startup_mode,
                self_consistency=self_consistency,
                adaptive_timestep=adaptive_timestep,
                space_charge=space_charge,
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
            )
            _ensure_startup_metadata(trajectory[i])
            _traj_builder.set_step(i, trajectory[i])

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
                trajectory[-1]["_halt_reason"] = (
                    f"all_particles_dead at step {i}/{steps}. {failure_summary}"
                )
                trajectory[-1]["_halt_step"] = i
                trajectory[-1]["_requested_steps"] = steps
                _traj_builder.set_halt_metadata(
                    step=i,
                    reason=f"all_particles_dead at step {i}/{steps}. {failure_summary}",
                    halt_step=i,
                    requested_steps=steps,
                )
                _traj_soa = _traj_builder.build()
                _traj_drv_soa = _traj_drv_builder.build() if _traj_drv_builder is not None else None
                return trajectory, trajectory_drv, _traj_soa, _traj_drv_soa

            # Post-step gamma check for individual particles
            # Mark any particle with extreme gamma as dead
            gamma_array = trajectory[i].get("gamma")
            if gamma_array is not None:
                for particle_idx in range(len(gamma_array)):
                    gamma_val = gamma_array[particle_idx]
                    # Skip if already dead
                    dead_mask = trajectory[i].get("_dead_particles")
                    if dead_mask is not None and dead_mask[particle_idx]:
                        continue

                    # Check for gamma blowup
                    if gamma_val > 1e8 or np.isnan(gamma_val) or np.isinf(gamma_val):
                        print(
                            f"[WARNING] Step {i}: Particle {particle_idx} gamma blowup "
                            f"detected (γ={gamma_val:.2e}). Marking particle as dead."
                        )
                        mark_particle_dead(
                            trajectory[i],
                            particle_idx,
                            i,
                            "gamma_blowup_post_step",
                            gamma_value=gamma_val,
                        )

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
                _b2b_scs_accepts_soa = "traj_soa" in inspect.signature(
                    self_consistent_step
                ).parameters
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
                    radiation_reaction_mode=radiation_reaction_mode,
                    **({
                        "space_charge": space_charge
                    } if space_charge is not None else {}),
                    **({
                        "traj_soa": _traj_drv_builder.build_partial(i)
                    } if _b2b_scs_accepts_soa and _traj_drv_builder is not None else {}),
                    **({
                        "traj_ext_soa": _traj_builder.build_partial(i)
                    } if _b2b_scs_accepts_soa else {}),
                )
            _ensure_startup_metadata(trajectory_drv[i])
            if _traj_drv_builder is not None:
                _traj_drv_builder.set_step(i, trajectory_drv[i])

        # Check for early termination in BUNCH_TO_BUNCH relative mode
        if (
            z_cutoff_mode == "relative"
            and sim_type == SimulationType.BUNCH_TO_BUNCH
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
                trajectory_truncated[-1]["_halt_reason"] = (
                    f"distance_reached ({distance_traveled:.2f} mm > {z_cutoff:.2f} mm at step {i}/{steps})"
                )
                trajectory_truncated[-1]["_halt_step"] = i
                trajectory_truncated[-1]["_requested_steps"] = steps
                _traj_builder.set_halt_metadata(
                    step=i,
                    reason=f"distance_reached ({distance_traveled:.2f} mm > {z_cutoff:.2f} mm at step {i}/{steps})",
                    halt_step=i,
                    requested_steps=steps,
                )
                _traj_soa = _traj_builder.build()
                _traj_drv_soa = _traj_drv_builder.build() if _traj_drv_builder is not None else None
                return trajectory_truncated, trajectory_drv_truncated, _traj_soa, _traj_drv_soa

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
                        f"Energy jump detected at step {i}/{steps}: "
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

        if progress_callback is not None:
            progress_callback(i + 1, steps)

    _traj_soa = _traj_builder.build()
    _traj_drv_soa = _traj_drv_builder.build() if _traj_drv_builder is not None else None
    return trajectory, trajectory_drv, _traj_soa, _traj_drv_soa



def run_integrator(
    config: IntegratorConfig,
    init_rider: ParticleState,
    init_driver: Optional[ParticleState],
    self_consistency: Optional[SelfConsistencyConfig] = None,
    energy_monitor: Optional[EnergyMonitorConfig] = None,
    adaptive_timestep: Optional[AdaptiveTimestepConfig] = None,
    space_charge: Optional[Any] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Tuple[Trajectory, Trajectory, "TrajectoryArrays | None", "TrajectoryArrays | None"]:
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
        macroparticle_charge_multiplier=config.macroparticle_charge_multiplier,
        macroparticle_sigma_multiplier=config.macroparticle_sigma_multiplier,
        macroparticle_use_momentum_errors=config.macroparticle_use_momentum_errors,
        bunch_transv_dist=config.bunch_transv_dist,
        bunch_transv_mom=config.bunch_transv_mom,
        energy_monitor=energy_monitor,
        adaptive_timestep=adaptive_timestep,
        space_charge=space_charge,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


__all__ = [
    "IntegrationCancelled",
    "EnergyJumpDetected",
    "EnergyMonitorConfig",
    "AdaptiveTimestepConfig",
    "retarded_integrator",
    "run_integrator",
]
