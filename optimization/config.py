"""Shared configuration and timestep utilities for optimization workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.constants import C_MMNS  # type: ignore[import]
from core.types import SimulationType  # type: ignore[import]
from optimization.simulation_type_helpers import is_bunch_to_bunch

_ELECTRON_MASS_AMU = 0.00054857990907
_PROTON_MASS_AMU = 1.0
_ELECTRON_ENERGY_THRESHOLD_GEV = 120.0
_PROTON_ENERGY_THRESHOLD_GEV = 1.0e4
_ENERGY_THRESHOLD_EXPONENT = math.log(
    _PROTON_ENERGY_THRESHOLD_GEV / _ELECTRON_ENERGY_THRESHOLD_GEV
) / math.log(_PROTON_MASS_AMU / _ELECTRON_MASS_AMU)
_ENERGY_THRESHOLD_SCALE = _ELECTRON_ENERGY_THRESHOLD_GEV / (
    _ELECTRON_MASS_AMU**_ENERGY_THRESHOLD_EXPONENT
)


@dataclass
class OptimizationConfig:
    """Configuration for parameter sweep or optimization run."""

    # Simulation type
    simulation_type: SimulationType = SimulationType.CONDUCTING_WALL

    # Mode selection
    mode: str = "blind_sweep"  # "blind_sweep" or "optimization"

    # Optimization settings (only used when mode="optimization")
    optimization_method: str = (
        "differential_evolution"  # "differential_evolution", "genetic_algorithm", "multi_start", "adaptive_grid"
    )
    optimization_maxiter: int = 50  # Max iterations/generations
    optimization_population_size: int = (
        20  # For genetic algorithm and differential evolution
    )
    optimization_mutation_rate: float = 0.1  # For genetic algorithm
    optimization_crossover_rate: float = 0.7  # For genetic algorithm
    optimization_n_starts: int = 5  # For multi_start method
    optimization_save_top_n: int = 3  # Save trajectories from top N results
    optimization_convergence_tol: float = 1e-6  # Convergence tolerance (relative)
    optimization_convergence_patience: int = 10  # Generations for plateau detection

    # Penalty settings
    particle_death_penalty_fraction: float = (
        1.0  # Penalty multiplier for particle deaths (1.0 = 1:1 scaling)
        # Example with default 1.0: 10% lost → 10% penalty, 50% lost → 50% penalty
        # Set to 0.5 for gentler: 10% lost → 5% penalty
        # Set to 2.0 for stricter: 10% lost → 20% penalty
    )

    # Parameter ranges
    aperture_range: Tuple[float, float] = (1e-5, 1e-3)  # mm (10 μm to 1 mm)
    aperture_points: int = 10
    aperture_log_scale: bool = True

    energy_range: Tuple[float, float] = (1.0, 1000.0)  # GeV
    energy_points: int = 10
    energy_log_scale: bool = True

    transverse_offset_fractions: List[float] = None  # Fraction of aperture radius
    starting_z_positions: List[float] = None  # mm (particle starting z-coordinates)

    # Sweepable parameters (can be added to grid)
    transverse_momentum_range: Optional[Tuple[float, float]] = None  # amu·mm/ns
    transverse_momentum_points: int = 1
    transverse_momentum_log_scale: bool = False
    transverse_spread_range: Optional[Tuple[float, float]] = None  # mm (transv_dist)
    transverse_spread_points: int = 1
    transverse_spread_log_scale: bool = False
    timestep_range: Optional[Tuple[float, float]] = None  # ns (proper time)
    timestep_points: int = 1
    starting_z_range: Optional[Tuple[float, float]] = None  # mm
    starting_z_points: int = 1
    wall_z_range: Optional[Tuple[float, float]] = None  # mm
    wall_z_points: int = 1
    particle_mass_range: Optional[Tuple[float, float]] = None  # amu
    particle_mass_points: int = 1
    particle_charge_range: Optional[Tuple[float, float]] = None  # charge_sign
    particle_charge_points: int = 1
    cavity_spacing_range: Optional[Tuple[float, float]] = None  # mm (SWITCHING_WALL)
    cavity_spacing_points: int = 1
    macroparticle_charge_range: Optional[Tuple[float, float]] = (
        None  # charge multiplier
    )
    macroparticle_charge_points: int = 1
    macroparticle_sigma_range: Optional[Tuple[float, float]] = None  # sigma multiplier
    macroparticle_sigma_points: int = 1
    rider_stripped_ions_range: Optional[Tuple[float, float]] = None  # charge state
    rider_stripped_ions_points: int = 1
    driver_stripped_ions_range: Optional[Tuple[float, float]] = (
        None  # charge state (BUNCH_TO_BUNCH)
    )
    driver_stripped_ions_points: int = 1
    particle_count_range: Optional[Tuple[int, int]] = None  # rider pcount
    particle_count_points: int = 1
    driver_mass_range: Optional[Tuple[float, float]] = None  # amu (BUNCH_TO_BUNCH)
    driver_mass_points: int = 1
    driver_charge_sign_range: Optional[Tuple[float, float]] = None  # BUNCH_TO_BUNCH
    driver_charge_sign_points: int = 1
    driver_pcount_range: Optional[Tuple[int, int]] = None  # BUNCH_TO_BUNCH
    driver_pcount_points: int = 1
    driver_transv_mom_range: Optional[Tuple[float, float]] = (
        None  # amu·mm/ns (BUNCH_TO_BUNCH)
    )
    driver_transv_mom_points: int = 1
    driver_transv_mom_log_scale: bool = False
    driver_transv_dist_range: Optional[Tuple[float, float]] = (
        None  # mm (BUNCH_TO_BUNCH)
    )
    driver_transv_dist_points: int = 1
    driver_transv_dist_log_scale: bool = False
    driver_starting_distance_range: Optional[Tuple[float, float]] = (
        None  # mm (BUNCH_TO_BUNCH)
    )
    driver_starting_distance_points: int = 1
    driver_starting_distance_log_scale: bool = (
        False  # Use log scale for driver starting distance sweep
    )
    driver_starting_Pz_range: Optional[Tuple[float, float]] = (
        None  # amu·mm/ns (BUNCH_TO_BUNCH)
    )
    driver_starting_Pz_points: int = 1
    driver_energy_range: Optional[Tuple[float, float]] = None  # GeV (BUNCH_TO_BUNCH)
    driver_energy_points: int = 1
    driver_energy_log_scale: bool = False  # Use log scale for driver energy sweep

    # Linked energy sweep: when True, driver energy is locked to rider energy values
    # This results in a 1D sweep where both particles have the same kinetic energy
    linked_energy_sweep: bool = False

    # Fixed parameters
    wall_z: float = 100.0  # mm
    cavity_spacing: float = 1e5  # mm (for SWITCHING_WALL)
    steps: int = 2000
    timestep: float = 3e-7  # ns (proper time) - default from main GUI
    auto_steps: bool = False  # Automatically calculate steps based on distance
    auto_steps_target: int = (
        500  # Target number of steps when auto-calculating timestep
    )
    auto_steps_distance_past_wall: float = 10.0  # mm past wall to stop integration
    seed: int = 12345

    # Timestep strategy for energy sweeps
    timestep_strategy: str = (
        "auto_distance"  # "fixed", "energy_scaled", or "auto_distance"
    )
    energy_scale_exponent: float = 1.0  # For energy_scaled: h ∝ γ^-α
    target_distance_mm: float = 100.0  # For auto_distance: distance to reach
    z_cutoff_mode: str = "absolute"  # "absolute" or "relative" (for BUNCH_TO_BUNCH)
    cavity_exit_enabled: bool = False  # Stop BUNCH_TO_BUNCH runs at first cavity exit
    cavity_exit_length_mm: Optional[float] = None  # None uses initial rider-driver separation
    cavity_exit_residual_tail_factor: float = 0.0  # Reserved for residual-field tail handling
    cavity_exit_max_residual_tail_steps: int = 0  # Reserved for residual-field tail handling
    startup_mode: str = "COLD_START"  # "COLD_START" or "APPROXIMATE_BACK_HISTORY"

    # Fixed particle parameters (not swept)
    transv_mom: float = 1.2e-05  # amu·mm/ns
    transv_dist: float = 2e-06  # mm - transverse spread/radius
    long_dist: float = 0.0  # mm - longitudinal Gaussian sigma for rider bunch (0 = point slice)
    driver_long_dist: float = 0.0  # mm - longitudinal Gaussian sigma for driver bunch
    transverse_geometry: str = "square"  # point, square, gaussian, or ring
    transv_offset_x: float = 0.0  # mm - rider x-offset of bunch center from axis
    transv_offset_y: float = 0.0  # mm - rider y-offset of bunch center from axis
    driver_transv_offset_x: float = (
        0.0  # mm - driver x-offset of bunch center from axis
    )
    driver_transv_offset_y: float = (
        0.0  # mm - driver y-offset of bunch center from axis
    )
    m_particle: float = 0.00054857990907  # amu (electron mass)
    pcount: int = 1
    charge_sign: float = -1.0
    stripped_ions: float = 1.0  # Rider stripped ions (charge state)
    driver_m_particle: float = 207.2  # amu (driver particle mass, for BUNCH_TO_BUNCH)
    driver_charge_sign: float = 1.0  # Driver charge sign (for BUNCH_TO_BUNCH)
    driver_pcount: int = 5  # Driver particle count (for BUNCH_TO_BUNCH)
    driver_transv_mom: float = 0.0  # amu·mm/ns (driver transverse momentum)
    driver_transv_dist: float = -0.07998  # mm (driver transverse distance)
    driver_transverse_geometry: str = "square"  # point, square, gaussian, or ring
    driver_starting_distance: float = 1000.0  # mm (driver starting distance)
    driver_stripped_ions: float = 54.0  # Driver stripped ions (for BUNCH_TO_BUNCH)
    driver_starting_Pz: float = -4925.0  # Fixed driver Pz (amu·mm/ns)
    driver_energy_gev: float = (
        0.6057  # Fixed driver kinetic energy (GeV), consistent with Pz=-4925 for Xe
    )
    driver_direction: str = (
        "-z"  # Driver momentum direction: "-z" (toward rider) or "+z" (away from rider)
    )

    # Macroparticle simulation options (CONDUCTING_WALL only)
    macroparticle_enabled: bool = False
    macroparticle_charge_multiplier: float = 1.0
    macroparticle_sigma_multiplier: float = 1.0  # Multiplier for bunch spread params
    macroparticle_use_momentum_errors: bool = (
        True  # Include momentum-based cumulative errors
    )

    # Bounded macroparticle source smearing options
    macroparticle_smearing_enabled: bool = False
    macroparticle_smearing_subcharge_count: int = 8
    macroparticle_smearing_sigma_multiplier: float = 1.0
    macroparticle_smearing_position_sigma_mm: Optional[float] = None
    macroparticle_smearing_longitudinal_sigma_mm: Optional[float] = None
    macroparticle_smearing_momentum_sigma_amu_mm_ns: Optional[float] = None
    macroparticle_smearing_use_position_errors: bool = True
    macroparticle_smearing_use_momentum_errors: bool = True
    macroparticle_smearing_use_centroid_errors: bool = True
    macroparticle_smearing_use_internal_cloud: bool = True
    macroparticle_smearing_apply_to_active_observers: bool = True
    macroparticle_smearing_apply_to_active_sources: bool = True
    macroparticle_smearing_apply_to_passive_sources: bool = True
    macroparticle_smearing_apply_to_passive_updates: bool = False
    macroparticle_smearing_seed: int = 12345
    macroparticle_smearing_refresh_policy: str = "fixed_per_particle"

    # Conducting wall image parameters
    image_subcharge_count: int = 12  # Number of subcharges for conducting wall images
    use_image_weighting: bool = True  # Apply radial weighting to image subcharges

    # Optimization objective
    objective: str = "max_energy_gain"  # Primary objective to optimize

    # Multi-objective weighting (for future use)
    objective_weights: Dict[str, float] = (
        None  # e.g., {"max_energy_gain": 1.0, "min_transverse_spread": 0.5}
    )

    # Output
    output_dir: str = "results/sweeps"
    save_results: bool = True
    save_plots: bool = True

    # Trajectory saving options
    save_top_n_trajectories: bool = False  # Save trajectories for top N results
    save_all_trajectories: bool = False  # Save ALL evaluation trajectories
    save_failed_trajectories: bool = False  # Save only failed trajectories
    trajectory_stride: int = (
        1  # Save every Nth point to reduce file size (only used with "All")
    )

    # Metrics export options
    metrics_export_format: str = "both"  # Options: "json", "csv", "both", "none"
    metrics_export_scope: str = "all"  # Options: "all", "top_n"

    # Log saving options
    log_verbosity: str = "truncated"  # "none", "truncated", "full", "top_n_only"
    # none = no debug logs saved
    # truncated = 1-2 lines per run with parameters + metrics + errors/warnings only
    # full = complete debug output with SC iterations and adaptive timestep refinements
    # top_n_only = compact sweep logs; detailed output is limited to Top-N reruns

    # Stability and robustness options (from SimulationOptions)
    self_consistency_enabled: bool = True
    self_consistency_tolerance: float = 1e-4
    self_consistency_convergence_mode: str = "fixed_geometry"
    self_consistency_target_ms_tolerance: float = 1e-6
    self_consistency_max_iterations: int = 5
    self_consistency_mass_shell_tolerance: float = 1e-2
    self_consistency_mass_shell_relaxation: float = 0.7
    self_consistency_verbosity: int = 2  # 0=silent, 1=summary, 2=failures, 3=full
    self_consistency_chrono_interpolate: bool = False
    self_consistency_chrono_tolerance: float = 1e-3  # ns
    self_consistency_chrono_matching_mode: str = "FAST"
    self_consistency_chrono_high_precision: bool = False
    self_consistency_chrono_adaptive_tolerance: bool = False

    # Gamma reconciliation parameters
    self_consistency_gamma_reconciliation_method: str = "DISABLED"
    self_consistency_gamma_reconciliation_low_beta_threshold: float = 0.9
    self_consistency_gamma_reconciliation_high_beta_threshold: float = 0.99
    self_consistency_gamma_reconciliation_low_beta_weight: float = 0.8
    self_consistency_gamma_reconciliation_high_beta_weight: float = 0.2
    self_consistency_gamma_reconciliation_mid_beta_weight: float = 0.5
    self_consistency_gamma_reconciliation_fixed_weight: float = 0.5

    # Energy monitoring removed - functionality integrated into adaptive timestep
    energy_monitor_enabled: bool = False
    energy_monitor_threshold: float = 2.0
    energy_monitor_check_interval: int = 10
    energy_monitor_halt_on_jump: bool = False  # Now in adaptive_timestep
    energy_monitor_debug: bool = False

    adaptive_timestep_enabled: bool = True
    adaptive_timestep_threshold: float = 0.10
    adaptive_timestep_reduction_factor: int = 3
    adaptive_timestep_min_factor: float = 1e-4
    adaptive_timestep_cooldown_steps: int = 10
    adaptive_timestep_probe_threshold: float = 0.01
    adaptive_timestep_max_probe_steps: int = 3
    adaptive_timestep_debug: bool = False

    # Bunch-separation proximity refinement (BUNCH_TO_BUNCH mode only)
    adaptive_timestep_bunch_proximity_enabled: bool = False
    adaptive_timestep_bunch_proximity_sigma_mm: float = 5.0
    adaptive_timestep_bunch_proximity_n_sigma: float = 5.0
    adaptive_timestep_bunch_proximity_reduction_factor: float = 10.0
    adaptive_timestep_bunch_proximity_transition_n_sigma: float = 2.0

    # Intra-bunch space-charge options
    space_charge_enabled: bool = False
    space_charge_retarded: bool = True
    space_charge_softening_mm: float = 0.0
    space_charge_bunch_sigma_mm: float = 0.01
    space_charge_min_retarded_steps: Optional[int] = None

    # Prescribed external uniform field options
    external_field_enabled: bool = False
    external_electric_field_native: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    external_electric_field_v_per_m: Optional[Tuple[float, float, float]] = None
    external_magnetic_field_native: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    external_field_x_min: Optional[float] = None
    external_field_x_max: Optional[float] = None
    external_field_y_min: Optional[float] = None
    external_field_y_max: Optional[float] = None
    external_field_z_min: Optional[float] = None
    external_field_z_max: Optional[float] = None
    external_field_t_min: Optional[float] = None
    external_field_t_max: Optional[float] = None

    # Radiation-reaction handling
    radiation_reaction_mode: str = "medina_lad"

    # Fixed-size particle-loss options
    particle_loss_enabled: bool = True
    particle_loss_radius_mm: Optional[float] = 500.0
    particle_loss_conducting_wall_aperture_loss_enabled: bool = True
    particle_loss_initial_radial_quantile: Optional[float] = None
    particle_loss_initial_radial_multiplier: float = 1.0
    particle_loss_initial_radial_margin_mm: float = 0.0

    # Experimental pseudo-grid options
    pseudo_grid_enabled: bool = False
    pseudo_grid_active_rider_count: int = 4
    pseudo_grid_active_driver_count: int = 4
    pseudo_grid_passive_neighbor_count: int = 4
    pseudo_grid_coverage_strategy: str = "farthest_point_staleness"
    pseudo_grid_coverage_space: str = "position"
    pseudo_grid_pair_reuse_window: int = 16
    pseudo_grid_source_weighting_mode: str = "inverse_distance"
    pseudo_grid_loss_tracking_enabled: bool = True
    pseudo_grid_causal_history_pruning_enabled: bool = False
    pseudo_grid_causal_history_safety_margin_steps: int = 2

    # Driver-train options (BUNCH_TO_BUNCH only)
    driver_train_enabled: bool = False
    driver_train_bunch_count: int = 1
    driver_train_z_spacing_mm: float = 0.0
    driver_train_z_offsets_mm: Tuple[float, ...] = ()
    driver_train_prehistory_steps: int = 0
    driver_train_preserve_prehistory_in_output: bool = False

    # Sweep execution / robustness options
    workers: int = 1
    per_run_timeout: float = 300.0  # seconds (0 = no timeout, default 5 minutes)
    skip_failed_runs: bool = True  # Continue sweep even if individual runs fail
    failed_run_retry_attempts: int = (
        1  # Number of retry attempts for failed runs with new random seeds (0 = no retries)
    )

    # Trajectory stability checking (multi-step numerical validation)
    smoothness_enabled: bool = True  # Enable trajectory stability analysis
    smoothness_window_size: int = 20  # Steps for moving-window analysis
    smoothness_oscillation_threshold: float = (
        0.5  # Max oscillation score (sign-change rate)
    )
    smoothness_trend_threshold: float = 0.30  # Max polynomial fit residual
    smoothness_reject_on_violation: bool = True  # Reject numerically unstable runs
    smoothness_max_violations: int = 3  # Max violations before rejection

    def __post_init__(self):
        """Set defaults for list fields."""
        if self.transverse_offset_fractions is None:
            self.transverse_offset_fractions = [0.0]
        if self.starting_z_positions is None:
            self.starting_z_positions = [0.0]  # Default: start at origin
        if self.objective_weights is None:
            self.objective_weights = {}

    def calculate_timestep_for_energy(
        self,
        energy_gev: float,
        m_particle_amu: float = 0.00054857990907,
        wall_z: float = None,
        start_z: float = 0.0,
        driver_start_z: float = 1000.0,
        driver_energy_gev: float | None = None,
        driver_m_particle_amu: float | None = None,
    ) -> float:
        """Calculate appropriate timestep for given energy based on strategy.

        Parameters
        ----------
        energy_gev : float
            Particle energy in GeV
        m_particle_amu : float
            Particle mass in amu
        wall_z : float
            Wall position in mm (for CONDUCTING_WALL/SWITCHING_WALL)
        start_z : float
            Rider starting position in mm
        driver_start_z : float
            Driver starting position in mm (for BUNCH_TO_BUNCH mode)

        Returns
        -------
        float
            Timestep in ns (proper time)
        """
        if self.timestep_strategy == "fixed":
            return self.timestep

        rest_energy_mev = m_particle_amu * 931.494  # amu to MeV

        # For BUNCH_TO_BUNCH, energy is kinetic energy; for others, it's total energy
        if is_bunch_to_bunch(self.simulation_type):
            # Kinetic energy: γ = (KE / E_rest) + 1
            gamma = (energy_gev * 1e3) / rest_energy_mev + 1.0
        else:
            # Total energy: γ = E_total / E_rest
            gamma = (energy_gev * 1e3) / rest_energy_mev

        beta = np.sqrt(1.0 - 1.0 / gamma**2)
        rider_gamma_beta = gamma * beta

        if self.timestep_strategy == "energy_scaled":
            # Scale timestep inversely with gamma
            return self.timestep / (gamma**self.energy_scale_exponent)

        if self.timestep_strategy == "auto_distance":
            if wall_z is None:
                wall_z = self.wall_z

            # Calculate target distance based on simulation type
            if is_bunch_to_bunch(self.simulation_type):
                driver_mass = (
                    self.driver_m_particle
                    if driver_m_particle_amu is None
                    else driver_m_particle_amu
                )
                driver_energy = abs(
                    self.driver_energy_gev
                    if driver_energy_gev is None
                    else driver_energy_gev
                )
                driver_rest_mev = driver_mass * 931.494
                driver_gamma = (driver_energy * 1e3) / driver_rest_mev + 1.0
                driver_beta = (
                    np.sqrt(1.0 - 1.0 / driver_gamma**2)
                    if driver_gamma > 1.0
                    else 0.0
                )
                driver_gamma_beta = driver_gamma * driver_beta
                if getattr(self, "driver_direction", "-z") == "-z":
                    solver_closing_scale = rider_gamma_beta + driver_gamma_beta
                else:
                    solver_closing_scale = abs(rider_gamma_beta - driver_gamma_beta)
                if solver_closing_scale <= 0.0:
                    solver_closing_scale = max(rider_gamma_beta, 1e-12)

                # For BUNCH_TO_BUNCH the shared step is proper-time-like: each
                # bunch advances by gamma*beta*c*h. Size h from the solver
                # closing distance, not from the rider distance alone.
                total_distance = abs(driver_start_z - start_z) + self.target_distance_mm
            else:
                # For CONDUCTING_WALL/SWITCHING_WALL: travel to wall + target_distance
                total_distance = abs(wall_z - start_z) + self.target_distance_mm
                solver_closing_scale = rider_gamma_beta

            c_mmns = 299.792458  # mm/ns
            h_calculated = total_distance / (
                self.steps * c_mmns * solver_closing_scale
            )
            return h_calculated

        raise ValueError(f"Unknown timestep_strategy: {self.timestep_strategy}")

    @classmethod
    def from_simulation_options(cls, options: Any) -> "OptimizationConfig":
        """Create OptimizationConfig from SimulationOptions (main GUI config)."""
        rider = options.rider_params
        core = options.core_params
        timestep = core.get("time_step", 3e-7)

        return cls(
            simulation_type=options.simulation_type,
            wall_z=core.get("wall_z", 100.0),
            steps=options.steps,
            timestep=timestep,
            seed=options.seed,
            m_particle=rider.get("m_particle", 0.00054857990907),
            charge_sign=rider.get("charge_sign", -1.0),
            pcount=rider.get("pcount", 1),
            stripped_ions=rider.get("stripped_ions", 1.0),
            transv_mom=rider.get("transv_mom", 1.2e-05),
            transv_dist=rider.get("transv_dist", 2e-06),
            transverse_geometry=rider.get("transverse_geometry", "square"),
            driver_transverse_geometry=(
                options.driver_params.get("transverse_geometry", "square")
                if options.driver_params is not None
                else "square"
            ),
            output_dir=str(options.output_dir.parent / "optimization_results"),
            # Preserve stability options from main config
            self_consistency_enabled=options.self_consistency_enabled,
            self_consistency_tolerance=options.self_consistency_tolerance,
            self_consistency_convergence_mode=getattr(
                options, "self_consistency_convergence_mode", "fixed_geometry"
            ),
            self_consistency_target_ms_tolerance=getattr(
                options, "self_consistency_target_ms_tolerance", 1e-6
            ),
            self_consistency_max_iterations=options.self_consistency_max_iterations,
            self_consistency_mass_shell_tolerance=getattr(
                options, "self_consistency_mass_shell_tolerance", 1e-2
            ),
            self_consistency_mass_shell_relaxation=getattr(
                options, "self_consistency_mass_shell_relaxation", 0.7
            ),
            self_consistency_verbosity=options.self_consistency_verbosity,
            self_consistency_chrono_interpolate=getattr(
                options, "self_consistency_chrono_interpolate", False
            ),
            self_consistency_chrono_tolerance=getattr(
                options, "self_consistency_chrono_tolerance", 1e-3
            ),
            self_consistency_chrono_matching_mode=getattr(
                options, "self_consistency_chrono_matching_mode", "FAST"
            ),
            self_consistency_chrono_high_precision=getattr(
                options, "self_consistency_chrono_high_precision", False
            ),
            self_consistency_chrono_adaptive_tolerance=getattr(
                options, "self_consistency_chrono_adaptive_tolerance", False
            ),
            energy_monitor_enabled=False,  # Removed - integrated into adaptive timestep
            energy_monitor_threshold=2.0,
            energy_monitor_check_interval=10,
            energy_monitor_halt_on_jump=options.energy_monitor_halt_on_jump,
            energy_monitor_debug=False,
            adaptive_timestep_enabled=options.adaptive_timestep_enabled,
            adaptive_timestep_threshold=options.adaptive_timestep_threshold,
            adaptive_timestep_reduction_factor=options.adaptive_timestep_reduction_factor,
            adaptive_timestep_min_factor=options.adaptive_timestep_min_factor,
            adaptive_timestep_cooldown_steps=options.adaptive_timestep_cooldown_steps,
            adaptive_timestep_probe_threshold=options.adaptive_timestep_probe_threshold,
            adaptive_timestep_max_probe_steps=options.adaptive_timestep_max_probe_steps,
            adaptive_timestep_debug=options.adaptive_timestep_debug,
            adaptive_timestep_bunch_proximity_enabled=getattr(
                options, "adaptive_timestep_bunch_proximity_enabled", False
            ),
            adaptive_timestep_bunch_proximity_sigma_mm=getattr(
                options, "adaptive_timestep_bunch_proximity_sigma_mm", 5.0
            ),
            adaptive_timestep_bunch_proximity_n_sigma=getattr(
                options, "adaptive_timestep_bunch_proximity_n_sigma", 5.0
            ),
            adaptive_timestep_bunch_proximity_reduction_factor=getattr(
                options, "adaptive_timestep_bunch_proximity_reduction_factor", 10.0
            ),
            adaptive_timestep_bunch_proximity_transition_n_sigma=getattr(
                options,
                "adaptive_timestep_bunch_proximity_transition_n_sigma",
                2.0,
            ),
            space_charge_enabled=getattr(options, "space_charge_enabled", False),
            space_charge_retarded=getattr(options, "space_charge_retarded", True),
            space_charge_softening_mm=getattr(
                options, "space_charge_softening_mm", 0.0
            ),
            space_charge_bunch_sigma_mm=getattr(
                options, "space_charge_bunch_sigma_mm", 0.01
            ),
            space_charge_min_retarded_steps=getattr(
                options, "space_charge_min_retarded_steps", None
            ),
            external_field_enabled=getattr(options, "external_field_enabled", False),
            external_electric_field_native=tuple(
                float(v)
                for v in getattr(
                    options, "external_electric_field_native", (0.0, 0.0, 0.0)
                )
            ),
            external_electric_field_v_per_m=(
                tuple(float(v) for v in options.external_electric_field_v_per_m)
                if getattr(options, "external_electric_field_v_per_m", None) is not None
                else None
            ),
            external_magnetic_field_native=tuple(
                float(v)
                for v in getattr(
                    options, "external_magnetic_field_native", (0.0, 0.0, 0.0)
                )
            ),
            external_field_x_min=getattr(options, "external_field_x_min", None),
            external_field_x_max=getattr(options, "external_field_x_max", None),
            external_field_y_min=getattr(options, "external_field_y_min", None),
            external_field_y_max=getattr(options, "external_field_y_max", None),
            external_field_z_min=getattr(options, "external_field_z_min", None),
            external_field_z_max=getattr(options, "external_field_z_max", None),
            external_field_t_min=getattr(options, "external_field_t_min", None),
            external_field_t_max=getattr(options, "external_field_t_max", None),
            radiation_reaction_mode=getattr(
                options,
                "radiation_reaction_mode",
                "medina_lad",
            ),
            particle_loss_enabled=getattr(options, "particle_loss_enabled", True),
            particle_loss_radius_mm=getattr(
                options,
                "particle_loss_radius_mm",
                500.0,
            ),
            particle_loss_conducting_wall_aperture_loss_enabled=getattr(
                options,
                "particle_loss_conducting_wall_aperture_loss_enabled",
                True,
            ),
            particle_loss_initial_radial_quantile=getattr(
                options,
                "particle_loss_initial_radial_quantile",
                None,
            ),
            particle_loss_initial_radial_multiplier=getattr(
                options,
                "particle_loss_initial_radial_multiplier",
                1.0,
            ),
            particle_loss_initial_radial_margin_mm=getattr(
                options,
                "particle_loss_initial_radial_margin_mm",
                0.0,
            ),
            pseudo_grid_enabled=getattr(options, "pseudo_grid_enabled", False),
            pseudo_grid_active_rider_count=getattr(
                options,
                "pseudo_grid_active_rider_count",
                4,
            ),
            pseudo_grid_active_driver_count=getattr(
                options,
                "pseudo_grid_active_driver_count",
                4,
            ),
            pseudo_grid_passive_neighbor_count=getattr(
                options,
                "pseudo_grid_passive_neighbor_count",
                4,
            ),
            pseudo_grid_coverage_strategy=getattr(
                options,
                "pseudo_grid_coverage_strategy",
                "farthest_point_staleness",
            ),
            pseudo_grid_coverage_space=getattr(
                options,
                "pseudo_grid_coverage_space",
                "position",
            ),
            pseudo_grid_pair_reuse_window=getattr(
                options,
                "pseudo_grid_pair_reuse_window",
                16,
            ),
            pseudo_grid_source_weighting_mode=getattr(
                options,
                "pseudo_grid_source_weighting_mode",
                "inverse_distance",
            ),
            pseudo_grid_loss_tracking_enabled=getattr(
                options,
                "pseudo_grid_loss_tracking_enabled",
                True,
            ),
            pseudo_grid_causal_history_pruning_enabled=getattr(
                options,
                "pseudo_grid_causal_history_pruning_enabled",
                False,
            ),
            pseudo_grid_causal_history_safety_margin_steps=getattr(
                options,
                "pseudo_grid_causal_history_safety_margin_steps",
                2,
            ),
            cavity_exit_enabled=getattr(options, "cavity_exit_enabled", False),
            cavity_exit_length_mm=getattr(options, "cavity_exit_length_mm", None),
            cavity_exit_residual_tail_factor=getattr(
                options, "cavity_exit_residual_tail_factor", 0.0
            ),
            cavity_exit_max_residual_tail_steps=getattr(
                options, "cavity_exit_max_residual_tail_steps", 0
            ),
            driver_train_enabled=getattr(options, "driver_train_enabled", False),
            driver_train_bunch_count=getattr(options, "driver_train_bunch_count", 1),
            driver_train_z_spacing_mm=getattr(
                options,
                "driver_train_z_spacing_mm",
                0.0,
            ),
            driver_train_z_offsets_mm=tuple(
                getattr(options, "driver_train_z_offsets_mm", ())
            ),
            driver_train_prehistory_steps=getattr(
                options,
                "driver_train_prehistory_steps",
                0,
            ),
            driver_train_preserve_prehistory_in_output=getattr(
                options,
                "driver_train_preserve_prehistory_in_output",
                False,
            ),
            # Startup mode from core params
            startup_mode=core.get("startup_mode", "COLD_START"),
            # Default timeout and skip settings for sweeps
            workers=1,
            per_run_timeout=300.0,
            skip_failed_runs=True,
            # Default stability checking for sweeps
            smoothness_enabled=True,
            smoothness_reject_on_violation=True,
            smoothness_max_violations=3,
        )


def calculate_auto_timestep(
    start_z: float,
    wall_z: float,
    distance_past_wall: float,
    particle_energy_gev: float,
    particle_mass_amu: float = 0.00054857990907,
    target_steps: int = 500,
) -> float:
    """Calculate appropriate timestep to achieve target number of steps."""
    total_distance = abs(wall_z - start_z) + distance_past_wall
    AMU_TO_MEV = 931.494
    rest_energy_mev = particle_mass_amu * AMU_TO_MEV
    gamma = (particle_energy_gev * 1e3) / rest_energy_mev + 1.0
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.999
    timestep = total_distance / (target_steps * beta * gamma * C_MMNS)
    return timestep


def calculate_auto_steps(
    start_z: float,
    wall_z: float,
    distance_past_wall: float,
    timestep: float,
    particle_energy_gev: float,
    particle_mass_amu: float = 0.00054857990907,
) -> int:
    """Calculate number of integration steps automatically."""
    total_distance = abs(wall_z - start_z) + distance_past_wall
    AMU_TO_MEV = 931.494
    rest_energy_mev = particle_mass_amu * AMU_TO_MEV
    gamma = (particle_energy_gev * 1e3) / rest_energy_mev + 1.0
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.999
    distance_per_step = beta * gamma * C_MMNS * timestep
    steps = int(np.ceil(total_distance / distance_per_step * 1.1))
    min_steps = 20
    return max(steps, min_steps)


def calculate_steps_from_duration(
    total_duration_ns: float,
    particle_energy_gev: float,
    particle_mass_amu: float = 0.00054857990907,
) -> tuple[int, float]:
    """Calculate timestep and number of steps from total duration."""
    min_steps = 20
    timestep = total_duration_ns / min_steps
    return min_steps, timestep


__all__ = [
    "OptimizationConfig",
    "calculate_auto_timestep",
    "calculate_auto_steps",
    "calculate_steps_from_duration",
]
