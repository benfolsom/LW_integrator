"""Shared helpers for optimization plugin config persistence."""

from __future__ import annotations

from typing import Any, Dict


_PERSISTED_CONFIG_DEFAULTS: dict[str, Any] = {
    "self_consistency_enabled": True,
    "self_consistency_tolerance": 1e-4,
    "self_consistency_convergence_mode": "fixed_geometry",
    "self_consistency_target_ms_tolerance": 1e-6,
    "self_consistency_max_iterations": 5,
    "self_consistency_mass_shell_tolerance": 1e-2,
    "self_consistency_mass_shell_relaxation": 0.7,
    "self_consistency_verbosity": 0,
    "self_consistency_chrono_interpolate": False,
    "self_consistency_chrono_tolerance": 1e-3,
    "self_consistency_chrono_matching_mode": "FAST",
    "self_consistency_chrono_high_precision": False,
    "self_consistency_chrono_adaptive_tolerance": False,
    "energy_monitor_halt_on_jump": False,
    "adaptive_timestep_enabled": True,
    "adaptive_timestep_threshold": 0.10,
    "adaptive_timestep_reduction_factor": 10,
    "adaptive_timestep_min_factor": 1e-4,
    "adaptive_timestep_cooldown_steps": 10,
    "adaptive_timestep_probe_threshold": 0.01,
    "adaptive_timestep_max_probe_steps": 3,
    "adaptive_timestep_debug": False,
    "space_charge_enabled": False,
    "space_charge_retarded": True,
    "space_charge_softening_mm": 0.0,
    "space_charge_bunch_sigma_mm": 0.01,
    "space_charge_min_retarded_steps": None,
    "external_field_enabled": False,
    "external_electric_field_native": (0.0, 0.0, 0.0),
    "external_electric_field_v_per_m": None,
    "external_magnetic_field_native": (0.0, 0.0, 0.0),
    "external_field_x_min": None,
    "external_field_x_max": None,
    "external_field_y_min": None,
    "external_field_y_max": None,
    "external_field_z_min": None,
    "external_field_z_max": None,
    "external_field_t_min": None,
    "external_field_t_max": None,
    "radiation_reaction_mode": "medina_lad",
    "particle_loss_enabled": True,
    "particle_loss_radius_mm": 500.0,
    "particle_loss_conducting_wall_aperture_loss_enabled": True,
    "particle_loss_initial_radial_quantile": None,
    "particle_loss_initial_radial_multiplier": 1.0,
    "particle_loss_initial_radial_margin_mm": 0.0,
    "pseudo_grid_enabled": False,
    "pseudo_grid_active_rider_count": 4,
    "pseudo_grid_active_driver_count": 4,
    "pseudo_grid_passive_neighbor_count": 4,
    "pseudo_grid_coverage_strategy": "farthest_point_staleness",
    "pseudo_grid_coverage_space": "position",
    "pseudo_grid_pair_reuse_window": 16,
    "pseudo_grid_source_weighting_mode": "inverse_distance",
    "pseudo_grid_loss_tracking_enabled": True,
    "pseudo_grid_causal_history_pruning_enabled": False,
    "pseudo_grid_causal_history_safety_margin_steps": 2,
    "self_consistency_gamma_reconciliation_method": "DISABLED",
    "self_consistency_gamma_reconciliation_low_beta_threshold": 0.9,
    "self_consistency_gamma_reconciliation_high_beta_threshold": 0.99,
    "self_consistency_gamma_reconciliation_low_beta_weight": 0.8,
    "self_consistency_gamma_reconciliation_high_beta_weight": 0.2,
    "self_consistency_gamma_reconciliation_mid_beta_weight": 0.5,
    "self_consistency_gamma_reconciliation_fixed_weight": 0.5,
    "workers": 1,
    "per_run_timeout": 300.0,
    "skip_failed_runs": True,
    "failed_run_retry_attempts": 1,
    "smoothness_enabled": True,
    "smoothness_window_size": 20,
    "smoothness_oscillation_threshold": 0.5,
    "smoothness_trend_threshold": 0.30,
    "smoothness_reject_on_violation": True,
    "smoothness_max_violations": 3,
    "macroparticle_enabled": False,
    "macroparticle_charge_multiplier": 1.0,
    "macroparticle_sigma_multiplier": 1.0,
    "macroparticle_use_momentum_errors": True,
    "image_subcharge_count": 12,
    "use_image_weighting": True,
    "timestep_strategy": "auto_distance",
    "z_cutoff_mode": "absolute",
    "startup_mode": "COLD_START",
    "target_distance_mm": 100.0,
    "timestep": 3e-7,
    "energy_scale_exponent": 1.0,
    "transverse_geometry": "square",
    "driver_transverse_geometry": "square",
}


def metrics_export_settings_from_data(data: Dict[str, Any]) -> tuple[str, str]:
    """Resolve persisted metrics export settings."""
    return (
        data.get("metrics_export_format", "both"),
        data.get("metrics_export_scope", "all"),
    )


def apply_persisted_config_overrides(config: Any, data: Dict[str, Any]) -> Any:
    """Apply persisted config values and defaults onto an OptimizationConfig."""
    for attr_name, default in _PERSISTED_CONFIG_DEFAULTS.items():
        setattr(config, attr_name, data.get(attr_name, default))
    config.transverse_geometry = data.get(
        "rider_transverse_geometry", data.get("transverse_geometry", "square")
    )
    config.driver_transverse_geometry = data.get("driver_transverse_geometry", "square")

    # Energy monitor behavior was removed; keep internal defaults explicit.
    config.energy_monitor_enabled = False
    config.energy_monitor_threshold = 2.0
    config.energy_monitor_check_interval = 10
    config.energy_monitor_debug = False
    return config


def build_saved_config_payload(
    config: Any,
    *,
    timestep_mode: str,
    auto_steps_distance: float,
    rider_stripped_ions: float,
    driver_stripped_ions: float,
    driver_direction: str,
    sweep_state: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the persisted config payload from config and UI-only fields."""
    return {
        "simulation_type": config.simulation_type.name,
        "mode": config.mode,
        "aperture_min": config.aperture_range[0],
        "aperture_max": config.aperture_range[1],
        "aperture_points": config.aperture_points,
        "aperture_log_scale": config.aperture_log_scale,
        "energy_min": config.energy_range[0],
        "energy_max": config.energy_range[1],
        "energy_points": config.energy_points,
        "energy_log_scale": config.energy_log_scale,
        "transverse_offset_fractions": config.transverse_offset_fractions,
        "starting_z_positions": config.starting_z_positions,
        "wall_z": config.wall_z,
        "wall_z_range": config.wall_z_range,
        "wall_z_points": config.wall_z_points,
        "cavity_spacing": config.cavity_spacing,
        "steps": config.steps,
        "objective": config.objective,
        "save_top_n_trajectories": config.save_top_n_trajectories,
        "save_all_trajectories": config.save_all_trajectories,
        "save_failed_trajectories": config.save_failed_trajectories,
        "trajectory_stride": config.trajectory_stride,
        "metrics_export_format": config.metrics_export_format,
        "metrics_export_scope": config.metrics_export_scope,
        "log_verbosity": config.log_verbosity,
        "optimization_method": config.optimization_method,
        "optimization_maxiter": config.optimization_maxiter,
        "optimization_population_size": config.optimization_population_size,
        "optimization_mutation_rate": config.optimization_mutation_rate,
        "optimization_crossover_rate": config.optimization_crossover_rate,
        "optimization_n_starts": config.optimization_n_starts,
        "optimization_save_top_n": config.optimization_save_top_n,
        "optimization_convergence_tol": config.optimization_convergence_tol,
        "optimization_convergence_patience": (config.optimization_convergence_patience),
        "self_consistency_enabled": config.self_consistency_enabled,
        "self_consistency_tolerance": config.self_consistency_tolerance,
        "self_consistency_convergence_mode": config.self_consistency_convergence_mode,
        "self_consistency_target_ms_tolerance": config.self_consistency_target_ms_tolerance,
        "self_consistency_max_iterations": config.self_consistency_max_iterations,
        "self_consistency_mass_shell_tolerance": config.self_consistency_mass_shell_tolerance,
        "self_consistency_mass_shell_relaxation": config.self_consistency_mass_shell_relaxation,
        "self_consistency_verbosity": config.self_consistency_verbosity,
        "self_consistency_chrono_interpolate": (
            config.self_consistency_chrono_interpolate
        ),
        "self_consistency_chrono_tolerance": config.self_consistency_chrono_tolerance,
        "self_consistency_chrono_matching_mode": config.self_consistency_chrono_matching_mode,
        "self_consistency_chrono_high_precision": (
            config.self_consistency_chrono_high_precision
        ),
        "self_consistency_chrono_adaptive_tolerance": (
            config.self_consistency_chrono_adaptive_tolerance
        ),
        "self_consistency_gamma_reconciliation_method": (
            config.self_consistency_gamma_reconciliation_method
        ),
        "self_consistency_gamma_reconciliation_low_beta_threshold": (
            config.self_consistency_gamma_reconciliation_low_beta_threshold
        ),
        "self_consistency_gamma_reconciliation_high_beta_threshold": (
            config.self_consistency_gamma_reconciliation_high_beta_threshold
        ),
        "self_consistency_gamma_reconciliation_low_beta_weight": (
            config.self_consistency_gamma_reconciliation_low_beta_weight
        ),
        "self_consistency_gamma_reconciliation_high_beta_weight": (
            config.self_consistency_gamma_reconciliation_high_beta_weight
        ),
        "self_consistency_gamma_reconciliation_mid_beta_weight": (
            config.self_consistency_gamma_reconciliation_mid_beta_weight
        ),
        "self_consistency_gamma_reconciliation_fixed_weight": (
            config.self_consistency_gamma_reconciliation_fixed_weight
        ),
        "energy_monitor_halt_on_jump": config.energy_monitor_halt_on_jump,
        "adaptive_timestep_enabled": config.adaptive_timestep_enabled,
        "adaptive_timestep_threshold": config.adaptive_timestep_threshold,
        "adaptive_timestep_reduction_factor": (
            config.adaptive_timestep_reduction_factor
        ),
        "adaptive_timestep_min_factor": config.adaptive_timestep_min_factor,
        "adaptive_timestep_cooldown_steps": config.adaptive_timestep_cooldown_steps,
        "adaptive_timestep_probe_threshold": config.adaptive_timestep_probe_threshold,
        "adaptive_timestep_max_probe_steps": config.adaptive_timestep_max_probe_steps,
        "adaptive_timestep_debug": config.adaptive_timestep_debug,
        "space_charge_enabled": config.space_charge_enabled,
        "space_charge_retarded": config.space_charge_retarded,
        "space_charge_softening_mm": config.space_charge_softening_mm,
        "space_charge_bunch_sigma_mm": config.space_charge_bunch_sigma_mm,
        "space_charge_min_retarded_steps": config.space_charge_min_retarded_steps,
        "external_field_enabled": config.external_field_enabled,
        "external_electric_field_native": list(config.external_electric_field_native),
        "external_electric_field_v_per_m": (
            list(config.external_electric_field_v_per_m)
            if config.external_electric_field_v_per_m is not None
            else None
        ),
        "external_magnetic_field_native": list(config.external_magnetic_field_native),
        "external_field_x_min": config.external_field_x_min,
        "external_field_x_max": config.external_field_x_max,
        "external_field_y_min": config.external_field_y_min,
        "external_field_y_max": config.external_field_y_max,
        "external_field_z_min": config.external_field_z_min,
        "external_field_z_max": config.external_field_z_max,
        "external_field_t_min": config.external_field_t_min,
        "external_field_t_max": config.external_field_t_max,
        "radiation_reaction_mode": config.radiation_reaction_mode,
        "particle_loss": {
            "enabled": config.particle_loss_enabled,
            "loss_radius_mm": config.particle_loss_radius_mm,
            "conducting_wall_aperture_loss_enabled": (
                config.particle_loss_conducting_wall_aperture_loss_enabled
            ),
            "initial_radial_quantile": config.particle_loss_initial_radial_quantile,
            "initial_radial_multiplier": config.particle_loss_initial_radial_multiplier,
            "initial_radial_margin_mm": config.particle_loss_initial_radial_margin_mm,
        },
        "pseudo_grid": {
            "enabled": config.pseudo_grid_enabled,
            "active_rider_count": config.pseudo_grid_active_rider_count,
            "active_driver_count": config.pseudo_grid_active_driver_count,
            "passive_neighbor_count": config.pseudo_grid_passive_neighbor_count,
            "coverage_strategy": config.pseudo_grid_coverage_strategy,
            "coverage_space": config.pseudo_grid_coverage_space,
            "pair_reuse_window": config.pseudo_grid_pair_reuse_window,
            "source_weighting_mode": config.pseudo_grid_source_weighting_mode,
            "loss_tracking_enabled": config.pseudo_grid_loss_tracking_enabled,
            "causal_history_pruning_enabled": (
                config.pseudo_grid_causal_history_pruning_enabled
            ),
            "causal_history_safety_margin_steps": (
                config.pseudo_grid_causal_history_safety_margin_steps
            ),
        },
        "workers": config.workers,
        "per_run_timeout": config.per_run_timeout,
        "skip_failed_runs": config.skip_failed_runs,
        "failed_run_retry_attempts": config.failed_run_retry_attempts,
        "smoothness_enabled": config.smoothness_enabled,
        "smoothness_window_size": config.smoothness_window_size,
        "smoothness_oscillation_threshold": (config.smoothness_oscillation_threshold),
        "smoothness_trend_threshold": config.smoothness_trend_threshold,
        "smoothness_reject_on_violation": config.smoothness_reject_on_violation,
        "smoothness_max_violations": config.smoothness_max_violations,
        "macroparticle_enabled": config.macroparticle_enabled,
        "macroparticle_charge_multiplier": config.macroparticle_charge_multiplier,
        "macroparticle_sigma_multiplier": config.macroparticle_sigma_multiplier,
        "macroparticle_use_momentum_errors": (config.macroparticle_use_momentum_errors),
        "image_subcharge_count": config.image_subcharge_count,
        "use_image_weighting": config.use_image_weighting,
        "timestep_strategy": config.timestep_strategy,
        "target_distance_mm": config.target_distance_mm,
        "timestep": config.timestep,
        "energy_scale_exponent": config.energy_scale_exponent,
        "z_cutoff_mode": config.z_cutoff_mode,
        "startup_mode": config.startup_mode,
        "timestep_mode": timestep_mode,
        "auto_steps_distance": auto_steps_distance,
        "rider_stripped_ions": rider_stripped_ions,
        "driver_stripped_ions": driver_stripped_ions,
        "transverse_geometry": getattr(config, "transverse_geometry", "square"),
        "rider_transverse_geometry": getattr(config, "transverse_geometry", "square"),
        "driver_transverse_geometry": getattr(
            config, "driver_transverse_geometry", "square"
        ),
        "rider_offset_x": config.transv_offset_x,
        "rider_offset_y": config.transv_offset_y,
        "driver_offset_x": config.driver_transv_offset_x,
        "driver_offset_y": config.driver_transv_offset_y,
        "driver_direction": driver_direction,
        "linked_energy_sweep": config.linked_energy_sweep,
        "sweep_parameters": sweep_state,
    }


def resolve_loaded_sweep_state(
    sweep_state: Dict[str, Dict[str, Any]], param_name: str
) -> Dict[str, Any] | None:
    """Return saved sweep state, including legacy driver Pz conversion."""
    if param_name in sweep_state:
        return sweep_state[param_name]

    if param_name == "driver_energy_gev" and "driver_starting_Pz" in sweep_state:
        old_state = sweep_state["driver_starting_Pz"]
        return {
            "enabled": old_state.get("enabled", False),
            "min": "50.0",
            "max": "200.0",
            "points": old_state.get("points", "3"),
            "log": old_state.get("log", False),
            "fixed_value": "112.5",
        }

    return None


__all__ = [
    "apply_persisted_config_overrides",
    "build_saved_config_payload",
    "metrics_export_settings_from_data",
    "resolve_loaded_sweep_state",
]
