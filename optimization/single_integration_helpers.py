"""Pure helpers for per-run integration result processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
    rider_m_particle = _override_or_config(rider_m_particle, config, "m_particle")
    rider_charge_sign = _override_or_config(rider_charge_sign, config, "charge_sign")
    rider_pcount = int(_override_or_config(rider_pcount, config, "pcount"))
    rider_transv_mom = _override_or_config(rider_transv_mom, config, "transv_mom")
    rider_transv_dist = _override_or_config(rider_transv_dist, config, "transv_dist")
    rider_stripped_ions = _override_or_config(
        rider_stripped_ions, config, "stripped_ions"
    )
    wall_z = _override_or_config(wall_z, config, "wall_z")
    macroparticle_charge_multiplier = (
        macroparticle_charge_multiplier
        if macroparticle_charge_multiplier is not None
        else config.macroparticle_charge_multiplier
    )
    macroparticle_sigma_multiplier = (
        macroparticle_sigma_multiplier
        if macroparticle_sigma_multiplier is not None
        else config.macroparticle_sigma_multiplier
    )

    rider_params = {
        "starting_distance": start_z,
        "transv_mom": rider_transv_mom,
        "transv_dist": rider_transv_dist,
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
    return {
        "z": np.asarray(trajectory["z"])[::stride].tolist(),
        "r": np.asarray(trajectory["r"])[::stride].tolist(),
        "pz": np.asarray(trajectory["pz"])[::stride].tolist(),
        "pr": np.asarray(trajectory["pr"])[::stride].tolist(),
        "t": np.asarray(trajectory["t"])[::stride].tolist(),
        "gamma": np.asarray(trajectory["gamma"])[::stride].tolist(),
    }


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
    delta_e_mev = delta_gamma * rider_m_particle * AMU_TO_MEV

    metrics["max_percent_energy_gain"] = energy_gain_percent
    metrics["percent_delta_e"] = energy_gain_percent
    metrics["delta_gamma"] = delta_gamma
    metrics["delta_e_mev"] = delta_e_mev
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
            f"    max_percent_energy_gain: {energy_gain_percent:.12e}%",
            f"    percent_delta_e: {energy_gain_percent:.12e}%",
            f"    energy_gain_ppm: {energy_gain_ppm:.6f} ppm",
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


def _add_beam_optics_metrics(result: Any, metrics: dict[str, Any]) -> None:
    if result.rider_emittance_x_mm_mrad is not None:
        metrics["rider_emittance_x_mm_mrad"] = result.rider_emittance_x_mm_mrad
    if result.rider_emittance_y_mm_mrad is not None:
        metrics["rider_emittance_y_mm_mrad"] = result.rider_emittance_y_mm_mrad
    if result.rider_norm_emittance_x_mm_mrad is not None:
        metrics["rider_norm_emittance_x_mm_mrad"] = (
            result.rider_norm_emittance_x_mm_mrad
        )
    if result.rider_norm_emittance_y_mm_mrad is not None:
        metrics["rider_norm_emittance_y_mm_mrad"] = (
            result.rider_norm_emittance_y_mm_mrad
        )
    if result.rider_beta_x_m is not None:
        metrics["rider_beta_x_m"] = result.rider_beta_x_m
    if result.rider_beta_y_m is not None:
        metrics["rider_beta_y_m"] = result.rider_beta_y_m


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
