"""High-level orchestration for retarded-field trajectory integration.

This module coordinates the low-level physics kernels, image-charge
construction, and optional self-consistency loops.  It provides the primary
programmatic entry points for running the modern Liénard–Wiechert integrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

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
)


class IntegrationCancelled(RuntimeError):
    """Raised when an integration is cancelled by an external caller."""


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
        10  # Reduce timestep by this factor when jump detected
    )
    max_refinement_attempts: int = 5  # Maximum number of timestep refinements per step
    min_timestep_factor: float = 1e-4  # Minimum timestep as fraction of original

    # Hysteresis parameters: stay on reduced timestep for stability
    cooldown_steps: int = 10  # Minimum steps at reduced timestep before probing return
    probe_threshold: float = 0.01  # Energy stability threshold for safe return (1%)
    max_probe_steps: int = 3  # Number of consecutive stable steps needed to return

    # Impractical timestep handling and sub-step limit
    max_substeps_per_step: int = (
        1000  # Hard cap on substeps per step (derived from 1/min_timestep_factor)
    )
    skip_cooldown_on_particle_death: bool = (
        False  # Keep survivors in cooldown mode for safer recovery
    )

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
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Tuple[Trajectory, Trajectory]:
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
        each integration step completes. ``current`` counts completed steps.
    cancel_callback:
        Optional predicate evaluated before each step. If it returns ``True``
        the integration stops early by raising :class:`IntegrationCancelled`.


    Returns
    -------
    tuple[Trajectory, Trajectory]
        Two trajectories: the rider (primary bunch) and the driver (image or
        opposing bunch), each represented as a list of particle states.
    """

    trajectory: Trajectory = [{} for _ in range(steps)]
    trajectory_drv: Trajectory = [{} for _ in range(steps)]

    # Initialize energy monitoring
    previous_energy: Optional[float] = None

    # Track actual timestep (may be modified by adaptive refinement)
    current_h_step = h_step

    # Hysteresis tracking for adaptive timestep
    # When timestep is reduced, we stay reduced for cooldown_steps before probing
    reduced_timestep_mode = False
    reduced_h_step = h_step
    cooldown_counter = 0
    stable_steps_counter = 0  # Count consecutive stable steps during probing
    last_particle_death_step = (
        -1
    )  # Track when particles die to handle timestep recovery

    # Store initial z position for relative cutoff mode
    z_initial: Optional[float] = None
    if z_cutoff_mode == "relative" and sim_type == SimulationType.BUNCH_TO_BUNCH:
        z_initial = float(np.mean(init_rider["z"]))
        if adaptive_timestep is not None and adaptive_timestep.debug:
            print(
                f"BUNCH_TO_BUNCH relative cutoff mode: z_initial = {z_initial:.6f} mm, "
                f"will stop after traveling {z_cutoff:.2f} mm"
            )

    if progress_callback is not None:
        progress_callback(0, steps)

    for i in range(steps):
        if cancel_callback is not None and cancel_callback():
            raise IntegrationCancelled("Integration cancelled by caller.")
        if i == 0:
            trajectory[i] = init_rider
            _ensure_startup_metadata(trajectory[i])
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
        else:
            # Adaptive timestep refinement with sub-stepping and hysteresis:
            # If timestep is reduced, we stay reduced for several steps before probing return
            step_accepted = False
            refinement_attempt = 0  # Track energy jump detections for this step

            # Initialize to satisfy type checker (will be assigned in loop)
            temp_trajectory: Trajectory = []
            temp_driver: Trajectory = []

            # Proximity-based timestep refinement: detect nearness to walls/apertures
            proximity_reduction_active = False
            if (
                adaptive_timestep is not None
                and adaptive_timestep.proximity_refinement_enabled
                and aperture_radius is not None
                and wall_z is not None
            ):
                # Get current particle position (use mean z for bunch)
                current_z = float(np.mean(trajectory[i - 1]["z"]))
                distance_to_wall = abs(wall_z - current_z)

                # Define interaction region in terms of aperture radii
                interaction_distance = (
                    aperture_radius
                    * adaptive_timestep.proximity_distance_aperture_radii
                )
                transition_distance = (
                    aperture_radius * adaptive_timestep.proximity_transition_zone
                )

                # Check if we're in or approaching the interaction region
                if distance_to_wall <= interaction_distance:
                    proximity_reduction_active = True

                    # Smooth transition: full reduction close to wall, gradual outside
                    if distance_to_wall < (interaction_distance - transition_distance):
                        # Full reduction in strong interaction region
                        proximity_factor = adaptive_timestep.proximity_reduction_factor
                    else:
                        # Linear ramp in transition zone
                        ramp = (
                            interaction_distance - distance_to_wall
                        ) / transition_distance
                        proximity_factor = (
                            1.0
                            + (adaptive_timestep.proximity_reduction_factor - 1.0)
                            * ramp
                        )

                    if adaptive_timestep.debug:
                        # Determine which zone we're in for clearer logging
                        in_transition = distance_to_wall >= (
                            interaction_distance - transition_distance
                        )
                        zone_name = "transition" if in_transition else "full reduction"

                        print(
                            f"Step {i}: Proximity refinement active ({zone_name} zone). "
                            f"Distance to wall: {distance_to_wall:.6e} mm "
                            f"({distance_to_wall / aperture_radius:.1f} aperture radii). "
                            f"Reduction factor: {proximity_factor:.4f}x"
                        )

            # Hysteresis logic: decide starting timestep for this step
            if reduced_timestep_mode and adaptive_timestep is not None:
                # Check if timestep is impractically small (would require too many sub-steps)
                expected_substeps = int(np.ceil(h_step / reduced_h_step))
                # Note: max_substeps_per_step also serves as hard cap to prevent discontinuities
                impractical_timestep = (
                    expected_substeps > adaptive_timestep.max_substeps_per_step
                )

                # Skip cooldown if a particle just died and we have the skip flag enabled
                skip_cooldown = (
                    adaptive_timestep.skip_cooldown_on_particle_death
                    and last_particle_death_step == i - 1
                )

                if impractical_timestep:
                    # Timestep is too small - skip cooldown and try to recover immediately
                    if adaptive_timestep.debug:
                        print(
                            f"Step {i}: Impractical timestep detected ({expected_substeps} sub-steps required). "
                            f"Skipping cooldown and attempting recovery to {h_step:.6e} ns"
                        )
                    cooldown_counter = (
                        adaptive_timestep.cooldown_steps
                    )  # Jump to probing phase
                elif skip_cooldown:
                    # Particle just died - skip cooldown for survivors
                    if adaptive_timestep.debug:
                        print(
                            f"Step {i}: Particle died last step. Skipping cooldown for survivors, "
                            f"attempting recovery to {h_step:.6e} ns"
                        )
                    cooldown_counter = (
                        adaptive_timestep.cooldown_steps
                    )  # Jump to probing phase
                elif cooldown_counter < adaptive_timestep.cooldown_steps:
                    # Still in cooldown - use reduced timestep
                    current_h_step = reduced_h_step
                    cooldown_counter += 1
                    if adaptive_timestep.debug:
                        print(
                            f"Step {i}: Cooldown mode ({cooldown_counter}/{adaptive_timestep.cooldown_steps}), "
                            f"using reduced timestep {current_h_step:.6e} ns"
                        )
                else:
                    # Cooldown complete - probe whether we can return to normal
                    current_h_step = reduced_h_step  # Still use reduced for now
                    if adaptive_timestep.debug:
                        print(
                            f"Step {i}: Probing stability with reduced timestep "
                            f"({stable_steps_counter}/{adaptive_timestep.max_probe_steps} stable)"
                        )
            else:
                # Normal mode or adaptive disabled
                current_h_step = h_step

                # Apply proximity-based reduction if active
                if proximity_reduction_active and adaptive_timestep is not None:
                    current_h_step = h_step / proximity_factor
                    if adaptive_timestep.debug:
                        print(
                            f"Step {i}: Applying proximity refinement: "
                            f"{h_step:.6e} → {current_h_step:.6e} ns"
                        )

            # Initialize temp_trajectory base ONCE before retry loop
            # This preserves dead particle status across retry attempts within this step
            temp_trajectory_base = {
                k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
                for k, v in trajectory[i - 1].items()
            }

            # Propagate dead particle status from previous step ONCE
            if i > 0:
                propagate_dead_particle_status(temp_trajectory_base, trajectory[i - 1])

            def propagate_deaths_to_base(state_with_deaths):
                """Helper to propagate dead particles from a state to temp_trajectory_base."""
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
                # Check for cancellation inside retry loop
                if cancel_callback is not None and cancel_callback():
                    raise IntegrationCancelled("Integration cancelled by caller.")

                # Determine number of sub-steps needed to cover base timestep interval
                num_substeps_raw = int(np.round(h_step / current_h_step))

                # Apply cap to prevent runaway subdivision (uses max_substeps_per_step)
                if adaptive_timestep is not None:
                    num_substeps = min(
                        num_substeps_raw, adaptive_timestep.max_substeps_per_step
                    )

                    # Warn if cap is hit (indicates pathological region)
                    if (
                        num_substeps_raw > adaptive_timestep.max_substeps_per_step
                        and adaptive_timestep.debug
                    ):
                        print(
                            f"Step {i}: Sub-step limit reached ({adaptive_timestep.max_substeps_per_step}). "
                            f"Timestep {current_h_step:.6e} ns would require {num_substeps_raw} sub-steps. "
                            f"This step may not fully cover the base timestep interval."
                        )
                else:
                    # No adaptive timestep - use raw count
                    num_substeps = num_substeps_raw

                if num_substeps < 1:
                    num_substeps = 1

                # Build temporary trajectory for sub-stepping
                # Start with a COPY of the base (which already has dead particle status)
                # This ensures dead particles from previous retry attempts are preserved
                temp_trajectory = [
                    {
                        k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
                        for k, v in temp_trajectory_base.items()
                    }
                ]
                temp_driver = [trajectory_drv[i - 1]]

                energy_jump_detected = False
                gamma_blowup_detected = False
                max_refinement_reached = (
                    False  # Track if we've already warned about max refinement
                )
                min_timestep_reached = (
                    False  # Track if we've already warned about minimum timestep
                )

                for substep_idx in range(num_substeps):
                    # Check for cancellation inside substep loop
                    if cancel_callback is not None and cancel_callback():
                        raise IntegrationCancelled("Integration cancelled by caller.")

                    # Compute one sub-step, catching soft gamma blowups
                    try:
                        trial_state = self_consistent_step(
                            retarded_equations_of_motion,
                            current_h_step,
                            temp_trajectory,
                            temp_driver,
                            substep_idx,  # Use correct index in temp trajectory
                            aperture_radius,
                            sim_type,
                            self_consistency,
                            chrono_mode,
                            startup_mode,
                            step_idx=i,  # Pass main step index for error messages
                            cancel_callback=cancel_callback,  # Pass cancellation check through
                        )
                    except GammaBlowupError as e:
                        # Soft gamma blowup detected - reduce timestep and retry
                        # Only handle this if adaptive timestep is enabled
                        if adaptive_timestep is None or not adaptive_timestep.enabled:
                            # No adaptive timestep available - treat as hard blowup
                            print(
                                f"    [CRITICAL] Step {i}, Particle {e.particle_idx}: "
                                f"Gamma blowup (γ={e.gamma_value:.2e}) with no adaptive timestep available. "
                                f"Marking particle as dead."
                            )
                            # Get or create trial_state from previous step to mark particle dead
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
                                "gamma_blowup_no_adaptive",
                                gamma_value=e.gamma_value,
                                iteration=e.iteration,
                            )

                            # Append the state with dead particle to trajectory
                            temp_trajectory.append(trial_state)
                            temp_driver.append(
                                temp_driver[-1]
                                if temp_driver
                                else trajectory_drv[i - 1]
                            )

                            # Track particle death for timestep recovery logic
                            last_particle_death_step = i

                            # Don't retry - accept step with dead particle
                            gamma_blowup_detected = False
                            break  # Exit substep loop, step will be accepted below
                        else:
                            # Adaptive timestep available - try to reduce and retry
                            gamma_blowup_detected = True

                            if (
                                refinement_attempt
                                > adaptive_timestep.max_refinement_attempts
                            ):
                                # Exhausted retry attempts - convert to hard blowup
                                print(
                                    f"    [CRITICAL] Step {i}, Particle {e.particle_idx}: "
                                    f"Max refinement attempts reached after gamma blowup (γ={e.gamma_value:.6e}). "
                                    f"Marking particle as dead."
                                )
                                # Get or create trial_state from previous step
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
                                    "gamma_blowup_max_retries",
                                    gamma_value=e.gamma_value,
                                    iteration=e.iteration,
                                )

                                # Append the state with dead particle to trajectory
                                temp_trajectory.append(trial_state)
                                temp_driver.append(
                                    temp_driver[-1]
                                    if temp_driver
                                    else trajectory_drv[i - 1]
                                )

                                # Track particle death for timestep recovery logic
                                last_particle_death_step = i

                                # Don't retry - accept step with dead particle
                                gamma_blowup_detected = False
                                break  # Exit substep loop, step will be accepted below
                            else:
                                # Check minimum timestep limit
                                min_h = h_step * adaptive_timestep.min_timestep_factor
                                new_h_step = (
                                    current_h_step
                                    / adaptive_timestep.timestep_reduction_factor
                                )

                                if new_h_step < min_h:
                                    # Minimum timestep reached - convert to hard blowup
                                    print(
                                        f"    [CRITICAL] Step {i}, Particle {e.particle_idx}: "
                                        f"Minimum timestep reached after gamma blowup (γ={e.gamma_value:.6e}). "
                                        f"Marking particle as dead."
                                    )
                                    # Get or create trial_state from previous step
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

                                    # Append the state with dead particle to trajectory
                                    temp_trajectory.append(trial_state)
                                    temp_driver.append(
                                        temp_driver[-1]
                                        if temp_driver
                                        else trajectory_drv[i - 1]
                                    )

                                    # Track particle death for timestep recovery logic
                                    last_particle_death_step = i

                                    # Don't retry - accept step with dead particle
                                    gamma_blowup_detected = False
                                    break  # Exit substep loop, step will be accepted below
                                else:
                                    # Reduce timestep and retry
                                    # For hard blowups (NaN/Inf or gamma > 1e20), use more aggressive reduction
                                    if (
                                        hasattr(e, "is_hard_blowup")
                                        and e.is_hard_blowup
                                    ):
                                        # Hard blowup: reduce by factor squared for more aggressive stepping
                                        reduction_factor = (
                                            adaptive_timestep.timestep_reduction_factor
                                            ** 2
                                        )
                                        severity = "HARD"
                                    else:
                                        # Soft blowup: use normal reduction factor
                                        reduction_factor = (
                                            adaptive_timestep.timestep_reduction_factor
                                        )
                                        severity = "soft"

                                    new_h_step = current_h_step / reduction_factor

                                    # Ensure we don't go below minimum
                                    if new_h_step < min_h:
                                        new_h_step = min_h

                                    current_h_step = new_h_step
                                    if adaptive_timestep.debug:
                                        print(
                                            f"Step {i}.{substep_idx}: {severity} gamma blowup detected (γ={e.gamma_value:.6e}). "
                                            f"Reducing timestep by {reduction_factor}x "
                                            f"to {current_h_step:.6e} ns (attempt {refinement_attempt})"
                                        )

                                    # Enter or stay in reduced timestep mode
                                    reduced_timestep_mode = True
                                    reduced_h_step = current_h_step
                                    cooldown_counter = 0
                                    stable_steps_counter = 0

                                    # Propagate dead particles to base before breaking
                                    propagate_deaths_to_base(trial_state)

                                    break  # Exit sub-step loop to retry with smaller timestep

                    # After substep completes, propagate any newly-dead particles back to base
                    # so subsequent retries will skip them
                    propagate_deaths_to_base(trial_state)

                    # Update driver state for this substep
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
                    else:  # BUNCH_TO_BUNCH - driver doesn't change during substeps
                        trial_driver = temp_driver[-1]

                    # Check for energy jump if adaptive timestep is enabled
                    if adaptive_timestep is not None and adaptive_timestep.enabled:
                        current_energy = _compute_total_energy(trial_state)

                        if previous_energy is not None and previous_energy > 0:
                            relative_change = (
                                abs(current_energy - previous_energy) / previous_energy
                            )

                            # If energy jump exceeds threshold, abort and refine
                            if (
                                relative_change
                                > adaptive_timestep.energy_jump_threshold
                            ):
                                energy_jump_detected = True
                                refinement_attempt += 1

                                if (
                                    refinement_attempt
                                    > adaptive_timestep.max_refinement_attempts
                                ):
                                    # Max refinement attempts reached - accept step and continue substeps
                                    if (
                                        adaptive_timestep.debug
                                        and not max_refinement_reached
                                    ):
                                        print(
                                            f"Step {i}: Max refinement attempts ({adaptive_timestep.max_refinement_attempts}) reached. "
                                            f"Accepting remaining substeps (ΔE/E = {relative_change:.6e})"
                                        )
                                        max_refinement_reached = (
                                            True  # Only print once per step
                                        )
                                    energy_jump_detected = (
                                        False  # Don't retry, accept this substep
                                    )
                                else:
                                    # Check minimum timestep limit
                                    min_h = (
                                        h_step * adaptive_timestep.min_timestep_factor
                                    )
                                    new_h_step = (
                                        current_h_step
                                        / adaptive_timestep.timestep_reduction_factor
                                    )

                                    if new_h_step < min_h:
                                        if (
                                            adaptive_timestep.debug
                                            and not min_timestep_reached
                                        ):
                                            print(
                                                f"Step {i}: Minimum timestep reached. "
                                                f"Accepting remaining substeps (ΔE/E = {relative_change:.6e})"
                                            )
                                            min_timestep_reached = (
                                                True  # Only print once per retry
                                            )
                                        energy_jump_detected = False  # Accept anyway
                                    else:
                                        current_h_step = new_h_step
                                        if adaptive_timestep.debug:
                                            print(
                                                f"Step {i}: Energy jump (ΔE/E = {relative_change:.6e}). "
                                                f"Reducing timestep by {adaptive_timestep.timestep_reduction_factor}x "
                                                f"to {current_h_step:.6e} ns (attempt {refinement_attempt})"
                                            )

                                        # Enter or stay in reduced timestep mode
                                        reduced_timestep_mode = True
                                        reduced_h_step = current_h_step
                                        cooldown_counter = 0
                                        stable_steps_counter = 0

                                        # Propagate dead particles to base before breaking
                                        propagate_deaths_to_base(temp_trajectory[-1])

                                        break  # Exit sub-step loop to retry with smaller timestep

                        # Update previous energy for next substep
                        previous_energy = current_energy

                    # Append this substep to the temporary trajectory
                    temp_trajectory.append(trial_state)
                    temp_driver.append(trial_driver)

                # If no energy jump, gamma blowup, or we're accepting anyway, we're done
                if not energy_jump_detected and not gamma_blowup_detected:
                    step_accepted = True
                    if (
                        adaptive_timestep is not None
                        and adaptive_timestep.debug
                        and refinement_attempt > 0
                    ):
                        print(
                            f"Step {i}: Completed {num_substeps} sub-step(s) with timestep {current_h_step:.6e} ns"
                        )

                    # Hysteresis: check if we're in probing phase and can return to normal
                    if (
                        adaptive_timestep is not None
                        and adaptive_timestep.enabled
                        and reduced_timestep_mode
                        and cooldown_counter >= adaptive_timestep.cooldown_steps
                    ):
                        # We're in probing phase - check energy stability
                        if previous_energy is not None:
                            # Use temp_trajectory[-1] which contains the current step result
                            # (trajectory[i] hasn't been assigned yet at this point)
                            current_energy = _compute_total_energy(temp_trajectory[-1])
                            relative_change = (
                                abs(current_energy - previous_energy) / previous_energy
                            )

                            if relative_change < adaptive_timestep.probe_threshold:
                                # This step was stable
                                stable_steps_counter += 1
                                if adaptive_timestep.debug:
                                    print(
                                        f"Step {i}: Stable (ΔE/E = {relative_change:.6e} < {adaptive_timestep.probe_threshold:.6e}), "
                                        f"count = {stable_steps_counter}/{adaptive_timestep.max_probe_steps}"
                                    )

                                if (
                                    stable_steps_counter
                                    >= adaptive_timestep.max_probe_steps
                                ):
                                    # Safe to return to normal timestep
                                    reduced_timestep_mode = False
                                    stable_steps_counter = 0
                                    cooldown_counter = 0
                                    if adaptive_timestep.debug:
                                        print(
                                            f"Step {i}: Returning to normal timestep {h_step:.6e} ns "
                                            f"after {adaptive_timestep.max_probe_steps} stable steps"
                                        )
                            else:
                                # Energy jump during probing - reset and stay reduced
                                stable_steps_counter = 0
                                cooldown_counter = 0  # Restart cooldown
                                if adaptive_timestep.debug:
                                    print(
                                        f"Step {i}: Unstable during probing (ΔE/E = {relative_change:.6e}), "
                                        f"restarting cooldown"
                                    )

                                # Propagate dead particles to base before restarting cooldown
                                propagate_deaths_to_base(temp_trajectory[-1])

            # Accept the final sub-step state as the step result
            trajectory[i] = temp_trajectory[-1]
            _ensure_startup_metadata(trajectory[i])

            # Check if all particles are dead
            if all_particles_dead(trajectory[i]):
                # Get failure summary
                failure_info = get_particle_failure_summary(trajectory)
                failure_summary = format_failure_summary(failure_info)
                num_dead = len(failure_info)

                print(
                    f"[CRITICAL] Step {i}: All {num_dead} particles have failed. "
                    f"Halting integration."
                )
                print(f"  {failure_summary}")

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
                return trajectory, trajectory_drv

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
                    timestep=current_h_step,
                    step_number=i,
                )
            elif sim_type == SimulationType.BUNCH_TO_BUNCH:
                if init_driver is None:
                    raise ValueError(
                        "SimulationType.BUNCH_TO_BUNCH requires init_driver state"
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
                    step_idx=i,  # Pass step index for error messages
                )
            _ensure_startup_metadata(trajectory_drv[i])

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
                return trajectory_truncated, trajectory_drv_truncated

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

        # Update previous energy for adaptive timestep (even if energy_monitor is disabled)
        if adaptive_timestep is not None and adaptive_timestep.enabled and i > 0:
            previous_energy = _compute_total_energy(trajectory[i])

        if progress_callback is not None:
            progress_callback(i + 1, steps)

    return trajectory, trajectory_drv


def run_integrator(
    config: IntegratorConfig,
    init_rider: ParticleState,
    init_driver: Optional[ParticleState],
    self_consistency: Optional[SelfConsistencyConfig] = None,
    energy_monitor: Optional[EnergyMonitorConfig] = None,
    adaptive_timestep: Optional[AdaptiveTimestepConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Tuple[Trajectory, Trajectory]:
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
    Tuple[Trajectory, Trajectory]
        Rider and driver trajectories
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
