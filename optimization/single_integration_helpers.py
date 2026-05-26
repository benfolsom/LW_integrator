"""Pure helpers for per-run integration result processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from core.constants import C_MMNS
from core.smoothness_analyzer import SmoothnessConfig, analyze_trajectory_smoothness
from lw_integrator.testbed_runner import SimulationOptions
from optimization.simulation_type_helpers import is_bunch_to_bunch
from optimization.sweep_helpers import AMU_TO_MEV


@dataclass(frozen=True)
class SingleIntegrationSetup:
    """Resolved options and parameters for one sweep/optimization integration."""

    options: SimulationOptions
    rider_m_particle: float
    rider_charge_sign: float
    rider_pcount: int
    rider_transv_mom: float
    rider_transv_dist: float
    rider_stripped_ions: float
    wall_z: float
    macroparticle_charge_multiplier: float
    macroparticle_sigma_multiplier: float


@dataclass(frozen=True)
class IntegrationMetricsOutcome:
    """Computed metrics plus log lines that should be emitted by the caller."""

    metrics: dict[str, Any]
    log_lines: list[str]


@dataclass(frozen=True)
class HaltedIntegrationOutput:
    """Output payload plus log lines for a halted integration."""

    output: dict[str, Any]
    log_lines: list[str]


@dataclass(frozen=True)
class IntegrationTrajectoryOutput:
    """Output updates and logs from trajectory post-processing."""

    output_updates: dict[str, Any]
    log_lines: list[str]
    debug_print_lines: list[str]


def build_single_integration_setup(
    config: Any,
    *,
    aperture: float,
    energy_gev: float,
    start_z: float,
    transv_offset: float,
    timestep: float,
    steps: int,
    run_output_dir: Path,
    run_num: int,
    driver_params: dict[str, Any] | None,
    rider_m_particle: float | None = None,
    rider_charge_sign: float | None = None,
    rider_pcount: int | None = None,
    rider_transv_mom: float | None = None,
    rider_transv_dist: float | None = None,
    rider_stripped_ions: float | None = None,
    macroparticle_charge_multiplier: float | None = None,
    macroparticle_sigma_multiplier: float | None = None,
    wall_z: float | None = None,
    seed_override: int | None = None,
    simulation_options_cls: type = SimulationOptions,
) -> SingleIntegrationSetup:
    """Resolve integration parameters and build the testbed options object."""
    rider_m_particle = cast(
        float, _override_or_config(rider_m_particle, config, "m_particle")
    )
    rider_charge_sign = cast(
        float, _override_or_config(rider_charge_sign, config, "charge_sign")
    )
    rider_pcount = int(_override_or_config(rider_pcount, config, "pcount"))
    rider_transv_mom = cast(
        float, _override_or_config(rider_transv_mom, config, "transv_mom")
    )
    rider_transv_dist = cast(
        float, _override_or_config(rider_transv_dist, config, "transv_dist")
    )
    rider_stripped_ions = cast(
        float, _override_or_config(rider_stripped_ions, config, "stripped_ions")
    )
    wall_z = cast(float, _override_or_config(wall_z, config, "wall_z"))
    macroparticle_charge_multiplier = cast(
        float,
        (
            macroparticle_charge_multiplier
            if macroparticle_charge_multiplier is not None
            else config.macroparticle_charge_multiplier
        ),
    )
    macroparticle_sigma_multiplier = cast(
        float,
        (
            macroparticle_sigma_multiplier
            if macroparticle_sigma_multiplier is not None
            else config.macroparticle_sigma_multiplier
        ),
    )

    rider_params = {
        "starting_distance": start_z,
        "transv_mom": rider_transv_mom,
        "transv_dist": rider_transv_dist,
        "transverse_geometry": getattr(config, "transverse_geometry", "square"),
        "transv_offset_x": transv_offset,
        "transv_offset_y": 0.0,
        "m_particle": rider_m_particle,
        "charge_sign": rider_charge_sign,
        "pcount": rider_pcount,
        "stripped_ions": rider_stripped_ions,
        "starting_Pz": calculate_rider_starting_pz(
            energy_gev, rider_m_particle, config.simulation_type
        ),
    }
    core_params = {
        "time_step": timestep,
        "wall_z": wall_z,
        "aperture_radius": aperture,
        "mean": 1.0e5,
        "cav_spacing": getattr(config, "cavity_spacing", 1.0e5),
        "z_cutoff": (
            config.target_distance_mm if config.z_cutoff_mode == "relative" else 0.0
        ),
        "z_cutoff_mode": config.z_cutoff_mode,
        "startup_mode": config.startup_mode,
    }

    actual_seed = seed_override if seed_override is not None else config.seed + run_num
    external_electric_native = _coerce_vector3(
        getattr(config, "external_electric_field_native", (0.0, 0.0, 0.0)),
        default=(0.0, 0.0, 0.0),
    )
    external_electric_v_per_m = _coerce_optional_vector3(
        getattr(config, "external_electric_field_v_per_m", None)
    )
    external_magnetic_native = _coerce_vector3(
        getattr(config, "external_magnetic_field_native", (0.0, 0.0, 0.0)),
        default=(0.0, 0.0, 0.0),
    )

    options = simulation_options_cls(
        steps=steps,
        seed=actual_seed,
        simulation_type=config.simulation_type,
        rider_params=rider_params,
        driver_params=driver_params,
        core_params=core_params,
        trajectory_save=False,
        trajectory_interval=config.trajectory_stride,
        energy_display=False,
        energy_save=False,
        transverse_display=False,
        transverse_save=True,
        beta_display=False,
        beta_save=False,
        momentum_display=False,
        momentum_save=False,
        gamma_display=False,
        gamma_save=False,
        zposition_display=False,
        zposition_save=False,
        macroparticle_enabled=config.macroparticle_enabled,
        macroparticle_charge_multiplier=macroparticle_charge_multiplier,
        macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
        macroparticle_use_momentum_errors=config.macroparticle_use_momentum_errors,
        macroparticle_smearing_enabled=getattr(
            config, "macroparticle_smearing_enabled", False
        ),
        macroparticle_smearing_subcharge_count=getattr(
            config, "macroparticle_smearing_subcharge_count", 8
        ),
        macroparticle_smearing_sigma_multiplier=getattr(
            config, "macroparticle_smearing_sigma_multiplier", 1.0
        ),
        macroparticle_smearing_position_sigma_mm=getattr(
            config, "macroparticle_smearing_position_sigma_mm", None
        ),
        macroparticle_smearing_longitudinal_sigma_mm=getattr(
            config, "macroparticle_smearing_longitudinal_sigma_mm", None
        ),
        macroparticle_smearing_momentum_sigma_amu_mm_ns=getattr(
            config, "macroparticle_smearing_momentum_sigma_amu_mm_ns", None
        ),
        macroparticle_smearing_use_position_errors=getattr(
            config, "macroparticle_smearing_use_position_errors", True
        ),
        macroparticle_smearing_use_momentum_errors=getattr(
            config, "macroparticle_smearing_use_momentum_errors", True
        ),
        macroparticle_smearing_use_centroid_errors=getattr(
            config, "macroparticle_smearing_use_centroid_errors", True
        ),
        macroparticle_smearing_use_internal_cloud=getattr(
            config, "macroparticle_smearing_use_internal_cloud", True
        ),
        macroparticle_smearing_apply_to_active_observers=getattr(
            config, "macroparticle_smearing_apply_to_active_observers", False
        ),
        macroparticle_smearing_apply_to_active_sources=getattr(
            config, "macroparticle_smearing_apply_to_active_sources", True
        ),
        macroparticle_smearing_apply_to_passive_sources=getattr(
            config, "macroparticle_smearing_apply_to_passive_sources", True
        ),
        macroparticle_smearing_apply_to_passive_updates=getattr(
            config, "macroparticle_smearing_apply_to_passive_updates", False
        ),
        macroparticle_smearing_seed=getattr(
            config, "macroparticle_smearing_seed", 12345
        ),
        macroparticle_smearing_refresh_policy=getattr(
            config, "macroparticle_smearing_refresh_policy", "fixed_per_particle"
        ),
        image_subcharge_count=config.image_subcharge_count,
        use_image_weighting=config.use_image_weighting,
        output_dir=run_output_dir,
        self_consistency_enabled=config.self_consistency_enabled,
        self_consistency_tolerance=config.self_consistency_tolerance,
        self_consistency_convergence_mode=getattr(
            config, "self_consistency_convergence_mode", "fixed_geometry"
        ),
        self_consistency_target_ms_tolerance=getattr(
            config, "self_consistency_target_ms_tolerance", 1e-6
        ),
        self_consistency_max_iterations=config.self_consistency_max_iterations,
        self_consistency_mass_shell_tolerance=getattr(
            config, "self_consistency_mass_shell_tolerance", 1e-2
        ),
        self_consistency_mass_shell_relaxation=getattr(
            config, "self_consistency_mass_shell_relaxation", 0.7
        ),
        self_consistency_verbosity=config.self_consistency_verbosity,
        self_consistency_chrono_interpolate=getattr(
            config, "self_consistency_chrono_interpolate", False
        ),
        self_consistency_chrono_tolerance=getattr(
            config, "self_consistency_chrono_tolerance", 1e-3
        ),
        self_consistency_chrono_matching_mode=getattr(
            config, "self_consistency_chrono_matching_mode", "FAST"
        ),
        self_consistency_chrono_high_precision=getattr(
            config, "self_consistency_chrono_high_precision", False
        ),
        self_consistency_chrono_adaptive_tolerance=getattr(
            config, "self_consistency_chrono_adaptive_tolerance", False
        ),
        self_consistency_gamma_reconciliation_method=getattr(
            config, "self_consistency_gamma_reconciliation_method", "DISABLED"
        ),
        self_consistency_gamma_reconciliation_low_beta_threshold=getattr(
            config,
            "self_consistency_gamma_reconciliation_low_beta_threshold",
            0.9,
        ),
        self_consistency_gamma_reconciliation_high_beta_threshold=getattr(
            config,
            "self_consistency_gamma_reconciliation_high_beta_threshold",
            0.99,
        ),
        self_consistency_gamma_reconciliation_low_beta_weight=getattr(
            config,
            "self_consistency_gamma_reconciliation_low_beta_weight",
            0.8,
        ),
        self_consistency_gamma_reconciliation_high_beta_weight=getattr(
            config,
            "self_consistency_gamma_reconciliation_high_beta_weight",
            0.2,
        ),
        self_consistency_gamma_reconciliation_mid_beta_weight=getattr(
            config,
            "self_consistency_gamma_reconciliation_mid_beta_weight",
            0.5,
        ),
        self_consistency_gamma_reconciliation_fixed_weight=getattr(
            config, "self_consistency_gamma_reconciliation_fixed_weight", 0.5
        ),
        energy_monitor_enabled=False,
        energy_monitor_threshold=2.0,
        energy_monitor_check_interval=10,
        energy_monitor_halt_on_jump=config.energy_monitor_halt_on_jump,
        energy_monitor_debug=False,
        adaptive_timestep_enabled=config.adaptive_timestep_enabled,
        adaptive_timestep_threshold=config.adaptive_timestep_threshold,
        adaptive_timestep_reduction_factor=config.adaptive_timestep_reduction_factor,
        adaptive_timestep_min_factor=config.adaptive_timestep_min_factor,
        adaptive_timestep_cooldown_steps=config.adaptive_timestep_cooldown_steps,
        adaptive_timestep_probe_threshold=config.adaptive_timestep_probe_threshold,
        adaptive_timestep_max_probe_steps=config.adaptive_timestep_max_probe_steps,
        adaptive_timestep_debug=config.adaptive_timestep_debug,
        space_charge_enabled=getattr(config, "space_charge_enabled", False),
        space_charge_retarded=getattr(config, "space_charge_retarded", True),
        space_charge_softening_mm=getattr(config, "space_charge_softening_mm", 0.0),
        space_charge_bunch_sigma_mm=getattr(
            config, "space_charge_bunch_sigma_mm", 0.01
        ),
        space_charge_min_retarded_steps=getattr(
            config, "space_charge_min_retarded_steps", None
        ),
        external_field_enabled=getattr(config, "external_field_enabled", False),
        external_electric_field_native=external_electric_native,
        external_electric_field_v_per_m=external_electric_v_per_m,
        external_magnetic_field_native=external_magnetic_native,
        external_field_x_min=getattr(config, "external_field_x_min", None),
        external_field_x_max=getattr(config, "external_field_x_max", None),
        external_field_y_min=getattr(config, "external_field_y_min", None),
        external_field_y_max=getattr(config, "external_field_y_max", None),
        external_field_z_min=getattr(config, "external_field_z_min", None),
        external_field_z_max=getattr(config, "external_field_z_max", None),
        external_field_t_min=getattr(config, "external_field_t_min", None),
        external_field_t_max=getattr(config, "external_field_t_max", None),
        radiation_reaction_mode=getattr(
            config,
            "radiation_reaction_mode",
            "medina_lad",
        ),
        particle_loss_enabled=getattr(config, "particle_loss_enabled", True),
        particle_loss_radius_mm=getattr(config, "particle_loss_radius_mm", 500.0),
        particle_loss_conducting_wall_aperture_loss_enabled=getattr(
            config,
            "particle_loss_conducting_wall_aperture_loss_enabled",
            True,
        ),
        particle_loss_initial_radial_quantile=getattr(
            config,
            "particle_loss_initial_radial_quantile",
            None,
        ),
        particle_loss_initial_radial_multiplier=getattr(
            config,
            "particle_loss_initial_radial_multiplier",
            1.0,
        ),
        particle_loss_initial_radial_margin_mm=getattr(
            config,
            "particle_loss_initial_radial_margin_mm",
            0.0,
        ),
        pseudo_grid_enabled=getattr(config, "pseudo_grid_enabled", False),
        pseudo_grid_active_rider_count=getattr(
            config,
            "pseudo_grid_active_rider_count",
            4,
        ),
        pseudo_grid_active_driver_count=getattr(
            config,
            "pseudo_grid_active_driver_count",
            4,
        ),
        pseudo_grid_passive_neighbor_count=getattr(
            config,
            "pseudo_grid_passive_neighbor_count",
            4,
        ),
        pseudo_grid_coverage_strategy=getattr(
            config,
            "pseudo_grid_coverage_strategy",
            "farthest_point_staleness",
        ),
        pseudo_grid_coverage_space=getattr(
            config,
            "pseudo_grid_coverage_space",
            "position",
        ),
        pseudo_grid_pair_reuse_window=getattr(
            config,
            "pseudo_grid_pair_reuse_window",
            16,
        ),
        pseudo_grid_source_weighting_mode=getattr(
            config,
            "pseudo_grid_source_weighting_mode",
            "inverse_distance",
        ),
        pseudo_grid_loss_tracking_enabled=getattr(
            config,
            "pseudo_grid_loss_tracking_enabled",
            True,
        ),
        pseudo_grid_causal_history_pruning_enabled=getattr(
            config,
            "pseudo_grid_causal_history_pruning_enabled",
            False,
        ),
        pseudo_grid_causal_history_safety_margin_steps=getattr(
            config,
            "pseudo_grid_causal_history_safety_margin_steps",
            2,
        ),
        driver_train_enabled=getattr(config, "driver_train_enabled", False),
        driver_train_bunch_count=getattr(config, "driver_train_bunch_count", 1),
        driver_train_z_spacing_mm=getattr(config, "driver_train_z_spacing_mm", 0.0),
        driver_train_z_offsets_mm=tuple(
            getattr(config, "driver_train_z_offsets_mm", ())
        ),
        driver_train_prehistory_steps=getattr(
            config,
            "driver_train_prehistory_steps",
            0,
        ),
        driver_train_preserve_prehistory_in_output=getattr(
            config,
            "driver_train_preserve_prehistory_in_output",
            False,
        ),
    )

    return SingleIntegrationSetup(
        options=options,
        rider_m_particle=rider_m_particle,
        rider_charge_sign=rider_charge_sign,
        rider_pcount=rider_pcount,
        rider_transv_mom=rider_transv_mom,
        rider_transv_dist=rider_transv_dist,
        rider_stripped_ions=rider_stripped_ions,
        wall_z=wall_z,
        macroparticle_charge_multiplier=macroparticle_charge_multiplier,
        macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
    )


def _override_or_config(explicit_value: Any | None, config: Any, attr: str) -> Any:
    return explicit_value if explicit_value is not None else getattr(config, attr)


def _coerce_vector3(
    value: Any,
    *,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return default
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return default


def _coerce_optional_vector3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return _coerce_vector3(value, default=(0.0, 0.0, 0.0))


def calculate_rider_starting_pz(
    energy_gev: float, rider_m_particle: float, simulation_type: Any
) -> float:
    """Convert rider energy to the specific starting Pz expected by init_bunch."""
    rest_energy_mev = rider_m_particle * AMU_TO_MEV
    if is_bunch_to_bunch(simulation_type):
        gamma = (energy_gev * 1e3) / rest_energy_mev + 1.0
    else:
        gamma = (energy_gev * 1e3) / rest_energy_mev
    return C_MMNS * np.sqrt(gamma * gamma - 1.0)


def build_integration_metrics(
    result: Any,
    *,
    rider_m_particle: float,
    run_num: int,
    optimization_mode: bool = False,
) -> IntegrationMetricsOutcome:
    """Build rider metrics from a single integration result."""
    metrics: dict[str, Any] = {}
    log_lines = [
        f"  [RESULT] Run {run_num} metrics:",
        f"    rider_gamma_initial: {result.rider_gamma_initial}",
        f"    rider_gamma_final: {result.rider_gamma_final}",
    ]

    if result.rider_delta_e is not None:
        metrics["rider_delta_e_mev"] = result.rider_delta_e
    if result.rider_gamma_initial is not None:
        metrics["rider_gamma_initial"] = result.rider_gamma_initial
        metrics["initial_gamma_mean"] = result.rider_gamma_initial
    if result.rider_gamma_final is not None:
        metrics["rider_gamma_final"] = result.rider_gamma_final
        metrics["final_gamma_mean"] = result.rider_gamma_final

    driver_gamma_initial = getattr(result, "driver_gamma_initial", None)
    if driver_gamma_initial is not None:
        metrics["driver_gamma_initial"] = driver_gamma_initial
    driver_gamma_final = getattr(result, "driver_gamma_final", None)
    if driver_gamma_final is not None:
        metrics["driver_gamma_final"] = driver_gamma_final

    gamma_initial = result.rider_gamma_initial
    gamma_final = result.rider_gamma_final
    if gamma_initial is not None and gamma_final is not None and gamma_initial > 0:
        _add_energy_gain_metrics(
            metrics,
            log_lines,
            gamma_initial=gamma_initial,
            gamma_final=gamma_final,
            rider_m_particle=rider_m_particle,
            source_prefix="",
        )
        _add_rider_peak_energy_metrics(
            metrics,
            log_lines,
            result.rider_trajectory,
            gamma_initial=gamma_initial,
            rider_m_particle=rider_m_particle,
        )
        if optimization_mode:
            log_lines.append(
                f"    optimizer_objective: {-metrics['max_percent_energy_gain']:.12e}"
            )
    else:
        _add_fallback_energy_gain_metrics(
            result,
            metrics,
            log_lines,
            rider_m_particle=rider_m_particle,
            run_num=run_num,
        )

    _add_beam_optics_metrics(result, metrics)

    metrics["num_particles_dead"] = result.num_particles_dead
    if result.halted_early:
        metrics["halted_early"] = True
        if result.halt_reason:
            metrics["halt_reason"] = result.halt_reason

    return IntegrationMetricsOutcome(metrics=metrics, log_lines=log_lines)


def build_halted_integration_output(
    result: Any,
    *,
    run_num: int,
    save_trajectory: bool,
    trajectory_stride: int,
) -> HaltedIntegrationOutput:
    """Build output and log lines for a run halted before metrics extraction."""
    log_lines = [
        f"  [INFO] Run {run_num} was halted early - skipping metrics calculation",
        "    Only trajectory and logs will be saved (if enabled)",
    ]
    output = {
        "metrics": {},
        "halted_early": True,
        "halt_reason": result.halt_reason,
    }

    if result.rider_trajectory is not None and save_trajectory:
        traj = result.rider_trajectory
        try:
            output["trajectory"] = sample_trajectory_arrays(traj, trajectory_stride)
            log_lines.append(
                f"    Halted trajectory saved ({len(traj['z'])} points, "
                f"stride={trajectory_stride})"
            )
        except Exception as exc:
            log_lines.append(f"    [WARNING] Failed to save halted trajectory: {exc}")

    driver_trajectory = getattr(result, "driver_trajectory", None)
    if driver_trajectory is not None and save_trajectory:
        try:
            output["driver_trajectory"] = sample_trajectory_arrays(
                driver_trajectory,
                trajectory_stride,
            )
            log_lines.append(
                "    Halted driver trajectory saved "
                f"({len(driver_trajectory['z'])} points, "
                f"stride={trajectory_stride})"
            )
        except Exception as exc:
            log_lines.append(
                f"    [WARNING] Failed to save halted driver trajectory: {exc}"
            )

    log_lines.append(
        f"  [DEBUG] _run_single_integration returning for halted Run {run_num}"
    )
    return HaltedIntegrationOutput(output=output, log_lines=log_lines)


def build_integration_trajectory_output(
    result: Any,
    config: Any,
    *,
    run_num: int,
    rider_m_particle: float,
    metrics: dict[str, Any],
    save_trajectory: bool,
    trajectory_stride: int,
) -> IntegrationTrajectoryOutput:
    """Build output updates and log lines from trajectory post-processing."""
    output_updates: dict[str, Any] = {}
    log_lines = [f"  [DEBUG] Processing trajectory data for Run {run_num}..."]
    debug_print_lines: list[str] = []

    if result.rider_trajectory is None:
        log_lines.append(f"  [WARNING] No trajectory data available for Run {run_num}")
        if config.smoothness_enabled:
            log_lines.extend(
                [
                    (
                        "  [WARNING] Stability analysis SKIPPED - no trajectory "
                        "data returned from integration"
                    ),
                    "    Check that transverse_save=True in SimulationOptions",
                ]
            )
        return IntegrationTrajectoryOutput(
            output_updates=output_updates,
            log_lines=log_lines,
            debug_print_lines=debug_print_lines,
        )

    traj = result.rider_trajectory
    try:
        distance_info = distance_info_from_trajectory(traj)
        if distance_info is not None:
            output_updates["_distance_info"] = distance_info
    except Exception as exc:
        debug_print_lines.append(f"[DEBUG] Failed to extract distance info: {exc}")

    try:
        rider_position_metrics = _position_metrics_from_trajectory(
            traj,
            prefix="rider",
            include_radial_toward_driver=True,
        )
        if rider_position_metrics is not None:
            metrics.update(rider_position_metrics)
        _add_loss_metrics_from_alive_fraction(
            metrics,
            prefix="rider",
            total_count=getattr(config, "pcount", None),
        )
    except Exception as exc:
        debug_print_lines.append(
            f"[DEBUG] Failed to extract rider position metrics: {exc}"
        )

    driver_traj = getattr(result, "driver_trajectory", None)
    if driver_traj is not None:
        try:
            driver_distance_info = distance_info_from_trajectory(driver_traj)
            if driver_distance_info is not None:
                output_updates["_driver_distance_info"] = driver_distance_info
        except Exception as exc:
            debug_print_lines.append(
                f"[DEBUG] Failed to extract driver distance info: {exc}"
            )

        try:
            driver_position_metrics = _position_metrics_from_trajectory(
                driver_traj,
                prefix="driver",
                include_radial_toward_driver=False,
            )
            if driver_position_metrics is not None:
                metrics.update(driver_position_metrics)
            _add_loss_metrics_from_alive_fraction(
                metrics,
                prefix="driver",
                total_count=getattr(config, "driver_pcount", None),
            )
        except Exception as exc:
            debug_print_lines.append(
                f"[DEBUG] Failed to extract driver position metrics: {exc}"
            )

    if config.smoothness_enabled:
        log_lines.append(
            f"  [DEBUG] Performing stability analysis for Run {run_num}..."
        )
        smoothness_config = SmoothnessConfig(
            enabled=True,
            window_size=config.smoothness_window_size,
            oscillation_threshold=config.smoothness_oscillation_threshold,
            trend_smoothness_threshold=config.smoothness_trend_threshold,
            reject_on_violation=config.smoothness_reject_on_violation,
            max_allowed_violations=config.smoothness_max_violations,
        )
        smoothness_result = analyze_trajectory_smoothness(
            traj,
            smoothness_config,
            particle_mass_amu=rider_m_particle,
        )
        output_updates["stability_analysis"] = {
            "passed": smoothness_result.passed,
            "num_violations": len(smoothness_result.violations),
            "oscillation_score": smoothness_result.oscillation_score,
            "trend_smoothness_score": smoothness_result.trend_smoothness_score,
            "quality": smoothness_result.quality_summary,
        }

        if not smoothness_result.passed:
            log_lines.extend(
                [
                    f"  [WARNING] Stability check FAILED for Run {run_num}",
                    f"    Quality: {smoothness_result.quality_summary}",
                ]
            )
            if len(smoothness_result.violations) > 0:
                log_lines.append(f"    Violations: {len(smoothness_result.violations)}")
                for violation in smoothness_result.violations[:2]:
                    log_lines.append(f"      - {violation.description}")

            if config.smoothness_reject_on_violation:
                log_lines.append(
                    f"  [REJECT] Run {run_num} rejected due to numerical instability"
                )
                metrics["max_percent_energy_gain"] = np.nan
                output_updates["stability_rejected"] = True
        else:
            log_lines.append(
                f"  [OK] Stability check PASSED for Run {run_num}: "
                f"{smoothness_result.quality_summary}"
            )
    else:
        log_lines.append(
            f"  [INFO] Stability analysis DISABLED for Run {run_num} "
            "(smoothness_enabled=False)"
        )

    if save_trajectory:
        try:
            output_updates["trajectory"] = sample_trajectory_arrays(
                traj,
                trajectory_stride,
            )
        except Exception as exc:
            log_lines.append(f"    [WARNING] Failed to save trajectory arrays: {exc}")

        if driver_traj is not None:
            try:
                output_updates["driver_trajectory"] = sample_trajectory_arrays(
                    driver_traj,
                    trajectory_stride,
                )
            except Exception as exc:
                log_lines.append(
                    f"    [WARNING] Failed to save driver trajectory arrays: {exc}"
                )

    if result.halted_early:
        output_updates["halted_early"] = True
        output_updates["halt_reason"] = result.halt_reason

    return IntegrationTrajectoryOutput(
        output_updates=output_updates,
        log_lines=log_lines,
        debug_print_lines=debug_print_lines,
    )


def sample_trajectory_arrays(
    trajectory: Mapping[str, Any], stride: int
) -> dict[str, list]:
    """Return a stride-sampled trajectory payload suitable for JSON output."""
    z_array = np.asarray(trajectory["z"])
    series_length = len(z_array)
    if series_length == 0:
        return {
            key: []
            for key, value in trajectory.items()
            if np.asarray(value).ndim == 1 and len(np.asarray(value)) == 0
        }

    stride = max(int(stride), 1)
    sample_indices = list(range(0, series_length, stride))
    last_index = series_length - 1
    if sample_indices[-1] != last_index:
        sample_indices.append(last_index)
    sample_indices = sorted(set(sample_indices))

    sampled: dict[str, list] = {}
    for key, value in trajectory.items():
        array = np.asarray(value)
        if array.ndim != 1 or len(array) != series_length:
            continue
        sampled[key] = array[sample_indices].tolist()
    return sampled


def distance_info_from_trajectory(
    trajectory: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return compact distance metadata from a trajectory, if z data exists."""
    z_array = np.asarray(trajectory["z"])
    if len(z_array) == 0:
        return None
    return {
        "z_start": float(z_array[0]),
        "z_end": float(z_array[-1]),
        "num_steps": len(z_array),
    }


def _add_series_metrics(
    metrics: dict[str, float],
    trajectory: Mapping[str, Any],
    *,
    source_key: str,
    metric_base: str,
    prefix: str,
    lower_is_better: bool = False,
) -> None:
    values = np.asarray(trajectory.get(source_key, []), dtype=float)
    if len(values) == 0:
        return
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return

    start = float(values[0])
    end = float(values[-1])
    min_val = float(np.min(finite_values))
    max_val = float(np.max(finite_values))
    metrics[f"{prefix}_{metric_base}_initial"] = start
    metrics[f"{prefix}_{metric_base}_final"] = end
    metrics[f"{prefix}_{metric_base}_min"] = min_val
    metrics[f"{prefix}_{metric_base}_max"] = max_val
    metrics[f"{prefix}_{metric_base}_delta"] = end - start
    if lower_is_better:
        metrics[f"{prefix}_{metric_base}_reduction"] = start - end
        metrics[f"{prefix}_{metric_base}_peak_reduction"] = start - min_val


def _position_metrics_from_trajectory(
    trajectory: Mapping[str, Any],
    *,
    prefix: str,
    include_radial_toward_driver: bool,
) -> dict[str, float] | None:
    """Return position metrics from a trajectory, if data exists."""
    metrics: dict[str, float] = {}

    for component in ("x", "y", "z"):
        component_array = np.asarray(trajectory.get(component, []))
        if len(component_array) == 0:
            continue
        start = float(component_array[0])
        end = float(component_array[-1])
        metrics[f"{prefix}_{component}_initial_mm"] = start
        metrics[f"{prefix}_{component}_final_mm"] = end
        metrics[f"{prefix}_{component}_delta_mm"] = end - start

    r_array = np.asarray(trajectory.get("r", []))
    if len(r_array) > 0:
        r_start = float(r_array[0])
        r_end = float(r_array[-1])
        r_min = float(np.min(r_array))
        r_max = float(np.max(r_array))
        metrics[f"{prefix}_radial_initial_mm"] = r_start
        metrics[f"{prefix}_radial_final_mm"] = r_end
        metrics[f"{prefix}_radial_min_mm"] = r_min
        metrics[f"{prefix}_radial_max_mm"] = r_max
        metrics[f"{prefix}_radial_delta_mm"] = r_end - r_start
        if include_radial_toward_driver:
            metrics[f"{prefix}_radial_toward_driver_mm"] = r_start - r_end
            metrics[f"{prefix}_radial_peak_inward_mm"] = r_start - r_min

    r_rms_array = np.asarray(trajectory.get("r_rms_particle", []))
    if len(r_rms_array) > 0:
        r_rms_start = float(r_rms_array[0])
        r_rms_end = float(r_rms_array[-1])
        r_rms_min = float(np.min(r_rms_array))
        r_rms_max = float(np.max(r_rms_array))
        metrics[f"{prefix}_radial_rms_initial_mm"] = r_rms_start
        metrics[f"{prefix}_radial_rms_final_mm"] = r_rms_end
        metrics[f"{prefix}_radial_rms_min_mm"] = r_rms_min
        metrics[f"{prefix}_radial_rms_max_mm"] = r_rms_max
        metrics[f"{prefix}_radial_rms_delta_mm"] = r_rms_end - r_rms_start
        if include_radial_toward_driver:
            metrics[f"{prefix}_radial_rms_toward_driver_mm"] = r_rms_start - r_rms_end
            metrics[f"{prefix}_radial_rms_peak_inward_mm"] = r_rms_start - r_rms_min

    for percentile in (50, 68, 90, 95, 99):
        _add_series_metrics(
            metrics,
            trajectory,
            source_key=f"r_p{percentile}_particle",
            metric_base=f"radial_p{percentile}_mm",
            prefix=prefix,
            lower_is_better=True,
        )

    for multiplier in (2, 3, 5):
        _add_series_metrics(
            metrics,
            trajectory,
            source_key=f"halo_gt_{multiplier}_initial_rms_fraction",
            metric_base=f"halo_gt_{multiplier}_initial_rms_fraction",
            prefix=prefix,
            lower_is_better=True,
        )

    _add_series_metrics(
        metrics,
        trajectory,
        source_key="alive_fraction",
        metric_base="alive_fraction",
        prefix=prefix,
    )
    for width_name in ("z_width_p90_particle", "z_width_p98_particle"):
        metric_base = width_name.replace("z_width", "longitudinal_width").replace(
            "_particle", "_mm"
        )
        _add_series_metrics(
            metrics,
            trajectory,
            source_key=width_name,
            metric_base=metric_base,
            prefix=prefix,
            lower_is_better=True,
        )
    _add_series_metrics(
        metrics,
        trajectory,
        source_key="z_std_particle",
        metric_base="longitudinal_std_mm",
        prefix=prefix,
        lower_is_better=True,
    )
    _add_series_metrics(
        metrics,
        trajectory,
        source_key="gamma_std_particle",
        metric_base="gamma_std",
        prefix=prefix,
        lower_is_better=True,
    )
    _add_series_metrics(
        metrics,
        trajectory,
        source_key="pz_std_particle",
        metric_base="normalized_pz_std",
        prefix=prefix,
        lower_is_better=True,
    )

    return metrics or None


def build_final_z_check_log_lines(
    *,
    trajectory: Mapping[str, Any] | None,
    simulation_type: Any,
    driver_params: Mapping[str, Any] | None,
    target_distance_mm: float,
    wall_z: float,
    run_num: int,
) -> list[str]:
    """Return final-z diagnostic log lines for auto-distance runs."""
    if trajectory is None:
        return []

    try:
        z_array = np.asarray(trajectory.get("z", []))
        if len(z_array) == 0:
            return []

        final_z = float(z_array[-1])
        if is_bunch_to_bunch(simulation_type):
            driver_start_z = (
                driver_params.get("starting_distance", 1000.0)
                if driver_params is not None
                else 1000.0
            )
            expected_max_z = abs(driver_start_z) + target_distance_mm
            expected_line = (
                f"    Expected max z: {expected_max_z:.2f} mm "
                f"(driver_start + target={target_distance_mm:.2f})"
            )
        else:
            expected_max_z = wall_z + target_distance_mm
            expected_line = (
                f"    Expected max z: {expected_max_z:.2f} mm "
                f"(wall_z={wall_z:.2f} + target={target_distance_mm:.2f})"
            )

        if final_z > expected_max_z:
            excess = final_z - expected_max_z
            return [
                f"  [WARNING] Run {run_num}: Final z position EXCEEDED expected distance!",
                f"    Final z: {final_z:.2f} mm",
                expected_line,
                (
                    f"    Exceeded by: {excess:.2f} mm "
                    f"({excess / expected_max_z * 100:.1f}%)"
                ),
            ]

        under = expected_max_z - final_z
        return [
            f"  [DEBUG] Run {run_num}: Final z check OK",
            f"    Final z: {final_z:.2f} mm (under by {under:.2f} mm)",
        ]
    except Exception as exc:
        return [f"  [WARNING] Run {run_num}: Failed to check final z position: {exc}"]


def _add_energy_gain_metrics(
    metrics: dict[str, Any],
    log_lines: list[str],
    *,
    gamma_initial: float,
    gamma_final: float,
    rider_m_particle: float,
    source_prefix: str,
) -> None:
    delta_gamma = gamma_final - gamma_initial
    energy_gain_percent = delta_gamma / gamma_initial * 100.0
    energy_gain_ppm = delta_gamma / gamma_initial * 1e6
    rest_energy_mev = rider_m_particle * AMU_TO_MEV
    delta_e_mev = delta_gamma * rest_energy_mev
    initial_total_energy_mev = gamma_initial * rest_energy_mev
    initial_kinetic_energy_mev = max((gamma_initial - 1.0) * rest_energy_mev, 0.0)
    delta_e_fraction_initial_total = delta_e_mev / initial_total_energy_mev
    delta_e_fraction_initial_kinetic = (
        delta_e_mev / initial_kinetic_energy_mev
        if initial_kinetic_energy_mev > 0.0
        else np.nan
    )
    final_percent_energy_gain = 100.0 * delta_e_fraction_initial_kinetic

    metrics["rider_final_energy_gain_mev"] = delta_e_mev
    metrics["rider_final_energy_gain_gev"] = delta_e_mev / 1e3
    metrics["rider_final_percent_energy_gain"] = final_percent_energy_gain
    metrics["rider_final_percent_total_energy_gain"] = energy_gain_percent
    metrics["rider_max_energy_gain_mev"] = delta_e_mev
    metrics["rider_max_energy_gain_gev"] = delta_e_mev / 1e3
    metrics["rider_max_percent_energy_gain"] = final_percent_energy_gain
    metrics["rider_max_percent_total_energy_gain"] = energy_gain_percent

    metrics["max_percent_energy_gain"] = energy_gain_percent
    metrics["percent_delta_e"] = energy_gain_percent
    metrics["delta_gamma"] = delta_gamma
    metrics["delta_e_mev"] = delta_e_mev
    metrics["rider_initial_total_energy_mev"] = initial_total_energy_mev
    metrics["rider_initial_kinetic_energy_mev"] = initial_kinetic_energy_mev
    metrics["rider_delta_e_fraction_initial_total"] = delta_e_fraction_initial_total
    metrics["rider_delta_e_fraction_initial_kinetic"] = delta_e_fraction_initial_kinetic
    metrics["rider_delta_e_percent_initial_total"] = (
        100.0 * delta_e_fraction_initial_total
    )
    metrics["rider_delta_e_percent_initial_kinetic"] = (
        100.0 * delta_e_fraction_initial_kinetic
    )
    metrics["energy_gain_ppm"] = energy_gain_ppm
    metrics["max_energy_gain_gev"] = delta_e_mev / 1e3
    metrics["max_relative_gain"] = delta_gamma / gamma_initial

    if source_prefix:
        log_lines.extend(
            [
                f"    gamma_initial ({source_prefix}): {gamma_initial:.12e}",
                f"    gamma_final ({source_prefix}): {gamma_final:.12e}",
            ]
        )
    log_lines.extend(
        [
            f"    delta_gamma: {delta_gamma:.12e}",
            f"    delta_e_mev: {delta_e_mev:.12e} MeV",
            f"    rider_final_percent_energy_gain: {final_percent_energy_gain:.12e}%",
            f"    max_percent_energy_gain: {energy_gain_percent:.12e}%",
            f"    percent_delta_e: {energy_gain_percent:.12e}%",
            f"    energy_gain_ppm: {energy_gain_ppm:.6f} ppm",
        ]
    )


def _add_rider_peak_energy_metrics(
    metrics: dict[str, Any],
    log_lines: list[str],
    trajectory: Mapping[str, Any] | None,
    *,
    gamma_initial: float,
    rider_m_particle: float,
) -> None:
    if trajectory is None:
        return

    gamma_values = np.asarray(trajectory.get("gamma", []), dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(gamma_values))
    if finite_indices.size == 0:
        return

    finite_gamma_values = gamma_values[finite_indices]
    local_max_index = int(np.argmax(finite_gamma_values))
    max_step = int(finite_indices[local_max_index])
    gamma_max = float(finite_gamma_values[local_max_index])
    rest_energy_mev = rider_m_particle * AMU_TO_MEV
    initial_kinetic_energy_mev = max((gamma_initial - 1.0) * rest_energy_mev, 0.0)
    max_delta_e_mev = (gamma_max - gamma_initial) * rest_energy_mev
    max_total_energy_gain_percent = (gamma_max - gamma_initial) / gamma_initial * 100.0
    max_kinetic_energy_gain_percent = (
        100.0 * max_delta_e_mev / initial_kinetic_energy_mev
        if initial_kinetic_energy_mev > 0.0
        else np.nan
    )

    metrics["rider_max_gamma"] = gamma_max
    metrics["rider_max_energy_gain_step"] = max_step
    metrics["rider_max_energy_gain_mev"] = max_delta_e_mev
    metrics["rider_max_energy_gain_gev"] = max_delta_e_mev / 1e3
    metrics["rider_max_percent_energy_gain"] = max_kinetic_energy_gain_percent
    metrics["rider_max_percent_total_energy_gain"] = max_total_energy_gain_percent

    log_lines.extend(
        [
            f"    rider_max_gamma: {gamma_max:.12e}",
            f"    rider_max_energy_gain_mev: {max_delta_e_mev:.12e} MeV",
            (
                "    rider_max_percent_energy_gain: "
                f"{max_kinetic_energy_gain_percent:.12e}%"
            ),
            f"    rider_max_energy_gain_step: {max_step}",
        ]
    )


def _add_fallback_energy_gain_metrics(
    result: Any,
    metrics: dict[str, Any],
    log_lines: list[str],
    *,
    rider_m_particle: float,
    run_num: int,
) -> None:
    log_lines.append(
        "  [WARNING] Gamma values missing, attempting trajectory fallback..."
    )
    if result.rider_trajectory is None:
        log_lines.append("  [ERROR] No trajectory data available for fallback")
        _log_missing_energy_gain(log_lines, run_num)
        return

    try:
        gamma_array = np.asarray(result.rider_trajectory.get("gamma", []))
        if len(gamma_array) == 0:
            log_lines.append("  [ERROR] Trajectory gamma array is empty")
            _log_missing_energy_gain(log_lines, run_num)
            return

        gamma_initial = float(gamma_array[0])
        gamma_final = float(gamma_array[-1])
        if gamma_initial <= 0:
            log_lines.append("  [ERROR] Fallback gamma_initial <= 0")
            _log_missing_energy_gain(log_lines, run_num)
            return

        log_lines.append("  [OK] Fallback calculation successful:")
        _add_energy_gain_metrics(
            metrics,
            log_lines,
            gamma_initial=gamma_initial,
            gamma_final=gamma_final,
            rider_m_particle=rider_m_particle,
            source_prefix="from traj",
        )
        _add_rider_peak_energy_metrics(
            metrics,
            log_lines,
            result.rider_trajectory,
            gamma_initial=gamma_initial,
            rider_m_particle=rider_m_particle,
        )
    except Exception as exc:
        log_lines.append(f"  [ERROR] Fallback calculation failed: {exc}")
        _log_missing_energy_gain(log_lines, run_num)


def _log_missing_energy_gain(log_lines: list[str], run_num: int) -> None:
    log_lines.extend(
        [
            f"  [CRITICAL] max_percent_energy_gain could not be calculated for Run {run_num}",
            "  [CRITICAL] This will result in NaN/inf for optimization objective",
        ]
    )


def _add_loss_metrics_from_alive_fraction(
    metrics: dict[str, Any],
    *,
    prefix: str,
    total_count: Any,
) -> None:
    alive_fraction = metrics.get(f"{prefix}_alive_fraction_final")
    if alive_fraction is None:
        return
    try:
        total = int(total_count)
    except (TypeError, ValueError):
        return
    if total < 0:
        return

    loss_fraction = max(0.0, min(1.0, 1.0 - float(alive_fraction)))
    metrics[f"{prefix}_loss_fraction"] = loss_fraction
    metrics[f"{prefix}_loss_count"] = int(round(loss_fraction * total))


def _add_beam_optics_metrics(result: Any, metrics: dict[str, Any]) -> None:
    if getattr(result, "rider_emittance_x_mm_mrad", None) is not None:
        metrics["rider_emittance_x_mm_mrad"] = result.rider_emittance_x_mm_mrad
    if getattr(result, "rider_emittance_y_mm_mrad", None) is not None:
        metrics["rider_emittance_y_mm_mrad"] = result.rider_emittance_y_mm_mrad
    if getattr(result, "rider_norm_emittance_x_mm_mrad", None) is not None:
        metrics["rider_norm_emittance_x_mm_mrad"] = (
            result.rider_norm_emittance_x_mm_mrad
        )
    if getattr(result, "rider_norm_emittance_y_mm_mrad", None) is not None:
        metrics["rider_norm_emittance_y_mm_mrad"] = (
            result.rider_norm_emittance_y_mm_mrad
        )
    if getattr(result, "rider_beta_x_m", None) is not None:
        metrics["rider_beta_x_m"] = result.rider_beta_x_m
    if getattr(result, "rider_beta_y_m", None) is not None:
        metrics["rider_beta_y_m"] = result.rider_beta_y_m

    if getattr(result, "driver_emittance_x_mm_mrad", None) is not None:
        metrics["driver_emittance_x_mm_mrad"] = result.driver_emittance_x_mm_mrad
    if getattr(result, "driver_emittance_y_mm_mrad", None) is not None:
        metrics["driver_emittance_y_mm_mrad"] = result.driver_emittance_y_mm_mrad
    if getattr(result, "driver_norm_emittance_x_mm_mrad", None) is not None:
        metrics["driver_norm_emittance_x_mm_mrad"] = (
            result.driver_norm_emittance_x_mm_mrad
        )
    if getattr(result, "driver_norm_emittance_y_mm_mrad", None) is not None:
        metrics["driver_norm_emittance_y_mm_mrad"] = (
            result.driver_norm_emittance_y_mm_mrad
        )
    if getattr(result, "driver_beta_x_m", None) is not None:
        metrics["driver_beta_x_m"] = result.driver_beta_x_m
    if getattr(result, "driver_beta_y_m", None) is not None:
        metrics["driver_beta_y_m"] = result.driver_beta_y_m


__all__ = [
    "HaltedIntegrationOutput",
    "IntegrationMetricsOutcome",
    "IntegrationTrajectoryOutput",
    "SingleIntegrationSetup",
    "build_final_z_check_log_lines",
    "build_halted_integration_output",
    "build_integration_metrics",
    "build_integration_trajectory_output",
    "build_single_integration_setup",
    "calculate_rider_starting_pz",
    "distance_info_from_trajectory",
    "sample_trajectory_arrays",
]
