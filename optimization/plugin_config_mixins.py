"""Configuration load/save helpers for the optimization plugin."""

from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any

from optimization.config import OptimizationConfig
from optimization.plugin_persistence_helpers import (
    apply_persisted_config_overrides,
    build_saved_config_payload,
    metrics_export_settings_from_data,
    resolve_loaded_sweep_state,
)
from optimization.simulation_type_helpers import is_bunch_to_bunch
from optimization.sweep_helpers import calculate_energy_from_pz
from optimization.ui_helpers import (
    show_error_dialog as _show_error_dialog,
)
from optimization.ui_helpers import (
    show_warning_dialog as _show_warning_dialog,
)


class OptimizationPluginConfigMixin:
    """Load, save, and sync optimization plugin configuration state."""

    def _sync_stability_to_main_gui(self, config):
        """Sync stability settings from config to main GUI's stability tab."""
        if not self.gui_controller:
            return

        try:
            if hasattr(self.gui_controller, "self_consistency_enabled_var"):
                self.gui_controller.self_consistency_enabled_var.set(
                    config.self_consistency_enabled
                )
            if hasattr(self.gui_controller, "self_consistency_target_ms_tolerance_var"):
                self.gui_controller.self_consistency_target_ms_tolerance_var.set(
                    f"{config.self_consistency_tolerance:.1e}"
                )
            if hasattr(self.gui_controller, "self_consistency_max_iterations_var"):
                self.gui_controller.self_consistency_max_iterations_var.set(
                    str(config.self_consistency_max_iterations)
                )
            if hasattr(self.gui_controller, "self_consistency_verbosity_var"):
                self.gui_controller.self_consistency_verbosity_var.set(
                    str(config.self_consistency_verbosity)
                )
            if hasattr(self.gui_controller, "self_consistency_chrono_interpolate_var"):
                self.gui_controller.self_consistency_chrono_interpolate_var.set(
                    config.self_consistency_chrono_interpolate
                )
            if hasattr(self.gui_controller, "self_consistency_chrono_tolerance_var"):
                self.gui_controller.self_consistency_chrono_tolerance_var.set(
                    f"{config.self_consistency_chrono_tolerance:.1e}"
                )
            if hasattr(
                self.gui_controller, "self_consistency_chrono_high_precision_var"
            ):
                self.gui_controller.self_consistency_chrono_high_precision_var.set(
                    config.self_consistency_chrono_high_precision
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_chrono_adaptive_tolerance_var",
            ):
                self.gui_controller.self_consistency_chrono_adaptive_tolerance_var.set(
                    config.self_consistency_chrono_adaptive_tolerance
                )

            if hasattr(self.gui_controller, "adaptive_timestep_enabled_var"):
                self.gui_controller.adaptive_timestep_enabled_var.set(
                    config.adaptive_timestep_enabled
                )
            if hasattr(self.gui_controller, "adaptive_timestep_threshold_var"):
                self.gui_controller.adaptive_timestep_threshold_var.set(
                    f"{config.adaptive_timestep_threshold:.2f}"
                )
            if hasattr(self.gui_controller, "adaptive_timestep_reduction_factor_var"):
                self.gui_controller.adaptive_timestep_reduction_factor_var.set(
                    str(config.adaptive_timestep_reduction_factor)
                )
            if hasattr(self.gui_controller, "adaptive_timestep_min_factor_var"):
                self.gui_controller.adaptive_timestep_min_factor_var.set(
                    f"{config.adaptive_timestep_min_factor:.1e}"
                )
            if hasattr(self.gui_controller, "adaptive_timestep_cooldown_steps_var"):
                self.gui_controller.adaptive_timestep_cooldown_steps_var.set(
                    str(config.adaptive_timestep_cooldown_steps)
                )
            if hasattr(self.gui_controller, "adaptive_timestep_probe_threshold_var"):
                self.gui_controller.adaptive_timestep_probe_threshold_var.set(
                    f"{config.adaptive_timestep_probe_threshold:.6g}"
                )
            if hasattr(self.gui_controller, "adaptive_timestep_max_probe_steps_var"):
                self.gui_controller.adaptive_timestep_max_probe_steps_var.set(
                    str(config.adaptive_timestep_max_probe_steps)
                )
            if hasattr(self.gui_controller, "adaptive_timestep_debug_var"):
                self.gui_controller.adaptive_timestep_debug_var.set(
                    config.adaptive_timestep_debug
                )

            if hasattr(
                self.gui_controller, "self_consistency_gamma_reconciliation_method_var"
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_method_var.set(
                    config.self_consistency_gamma_reconciliation_method
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_low_beta_threshold_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_low_beta_threshold_var.set(
                    f"{config.self_consistency_gamma_reconciliation_low_beta_threshold:.2f}"
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_high_beta_threshold_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_high_beta_threshold_var.set(
                    f"{config.self_consistency_gamma_reconciliation_high_beta_threshold:.2f}"
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_low_beta_weight_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_low_beta_weight_var.set(
                    f"{config.self_consistency_gamma_reconciliation_low_beta_weight:.2f}"
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_high_beta_weight_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_high_beta_weight_var.set(
                    f"{config.self_consistency_gamma_reconciliation_high_beta_weight:.2f}"
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_mid_beta_weight_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_mid_beta_weight_var.set(
                    f"{config.self_consistency_gamma_reconciliation_mid_beta_weight:.2f}"
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_fixed_weight_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_fixed_weight_var.set(
                    f"{config.self_consistency_gamma_reconciliation_fixed_weight:.2f}"
                )

            if hasattr(self.gui_controller, "_toggle_self_consistency_controls"):
                self.gui_controller._toggle_self_consistency_controls()
            if hasattr(self.gui_controller, "_toggle_adaptive_timestep_controls"):
                self.gui_controller._toggle_adaptive_timestep_controls()
            if hasattr(self.gui_controller, "_toggle_gamma_reconciliation_params"):
                self.gui_controller._toggle_gamma_reconciliation_params()

            self._log_result(
                "[OK] Stability settings synced to main GUI's Stability tab"
            )

        except Exception as e:
            self._log_result(
                f"[WARNING] Failed to sync some stability settings to main GUI: {e}"
            )

    def _on_load_from_main_config(self):
        """Load parameters from the main GUI configuration."""
        if not self.gui_controller:
            _show_warning_dialog(
                self,
                "Load Config",
                "No main GUI controller available. Cannot load configuration.",
            )
            return

        try:
            main_options = self.gui_controller._build_options_from_ui()
            opt_config = OptimizationConfig.from_simulation_options(main_options)

            self.sim_type_var.set(opt_config.simulation_type.name)
            self._sync_main_gui_simulation_type(opt_config.simulation_type.name)
            self.wall_z_var.set(str(opt_config.wall_z))
            self.cavity_spacing_var.set(str(opt_config.cavity_spacing))

            self.timestep_mode_var.set("duration")
            self.steps_var.set(str(opt_config.steps))
            self.duration_var.set(f"{opt_config.timestep:.2e}")
            self._toggle_timestep_mode()
            self._set_fixed_sweep_value(
                "rider_m_particle", f"{opt_config.m_particle:.14e}"
            )
            self._set_fixed_sweep_value(
                "rider_charge_sign", str(opt_config.charge_sign)
            )
            self._set_fixed_sweep_value("rider_pcount", str(opt_config.pcount))
            self._set_fixed_sweep_value(
                "rider_stripped_ions", str(opt_config.stripped_ions)
            )
            self._set_fixed_sweep_value(
                "rider_transv_mom", f"{opt_config.transv_mom:.2e}"
            )
            self._set_fixed_sweep_value(
                "rider_transv_dist", f"{opt_config.transv_dist:.2e}"
            )
            self._apply_macroparticle_ui_state(
                enabled=getattr(opt_config, "macroparticle_enabled", False),
                charge_multiplier=f"{getattr(opt_config, 'macroparticle_charge_multiplier', 1.0):.2e}",
                sigma_multiplier=f"{getattr(opt_config, 'macroparticle_sigma_multiplier', 1.0):.2e}",
                momentum_errors=getattr(
                    opt_config, "macroparticle_use_momentum_errors", True
                ),
                refresh_state=True,
            )
            self.main_timestep_display_var.set(f"{opt_config.timestep:.2e}")

            rider_start_z = main_options.rider_params.get("starting_distance", 0.0)
            if is_bunch_to_bunch(main_options.simulation_type):
                if self._apply_driver_sweep_values(main_options.driver_params):
                    self._log_result("[INFO] Loaded driver parameters from main GUI")
                self.start_z_var.set(f"{rider_start_z}")
            else:
                self.start_z_var.set(f"{rider_start_z}")

            if hasattr(opt_config, "smoothness_enabled"):
                self._apply_smoothness_ui_state(
                    enabled=opt_config.smoothness_enabled,
                    window_size=str(opt_config.smoothness_window_size),
                    oscillation_threshold=str(
                        opt_config.smoothness_oscillation_threshold
                    ),
                    reject_on_violation=opt_config.smoothness_reject_on_violation,
                )

            self._log_result("[OK] Loaded parameters from main GUI configuration")
            self._log_result(f"  Simulation type: {opt_config.simulation_type.name}")
            self._log_result(f"  Wall z: {opt_config.wall_z} mm")
            self._log_result(f"  Cavity spacing: {opt_config.cavity_spacing} mm")
            self._log_result(
                "  Timestep mode: auto-calc duration (user provides count)"
            )
            self._log_result(f"  Steps: {opt_config.steps}")
            self._log_result(f"  Duration: {opt_config.timestep:.2e} ns")
            self._log_result(f"  Particle mass: {opt_config.m_particle:.6e} amu")
            self._log_result(
                f"  Transverse momentum: {opt_config.transv_mom:.2e} amu·mm/ns"
            )
            self._log_result(f"  Transverse distance: {opt_config.transv_dist:.2e} mm")
            self._log_result("")

            if self.gui_controller:
                self._sync_main_gui_visibility_state()

            self._update_driver_visibility()

            self._log_result("[INFO] Stability options loaded from main config:")
            self._log_result(
                f"  Self-consistency: {opt_config.self_consistency_enabled} (tol={opt_config.self_consistency_tolerance:.1e})"
            )
            self._log_result(
                f"  Adaptive timestep: {opt_config.adaptive_timestep_enabled} (threshold={opt_config.adaptive_timestep_threshold * 100:.0f}%)"
            )
            self._log_result("")

            self.config = opt_config

        except Exception as e:
            _show_error_dialog(
                self,
                "Load Config Error",
                f"Failed to load configuration from main GUI:\n{e}",
            )
            import traceback

            self._log_result(f"[ERROR] Error loading main config: {e}")
            self._log_result(traceback.format_exc())

    def _apply_macroparticle_ui_state(
        self,
        *,
        enabled: bool,
        charge_multiplier: str,
        sigma_multiplier: str,
        momentum_errors: bool,
        refresh_state: bool = False,
    ):
        """Apply macroparticle-related UI state."""
        self.macroparticle_enabled_var.set(enabled)
        self._set_fixed_sweep_value(
            "macroparticle_charge_multiplier", charge_multiplier
        )
        self._set_fixed_sweep_value("macroparticle_sigma_multiplier", sigma_multiplier)
        self.macroparticle_momentum_errors_var.set(momentum_errors)
        if refresh_state:
            self._toggle_macroparticle_controls()
            self._update_macroparticle_state()

    def _apply_smoothness_ui_state(
        self,
        *,
        enabled: bool,
        window_size: str,
        oscillation_threshold: str,
        reject_on_violation: bool,
    ):
        """Apply smoothness-related UI state."""
        self.smoothness_enabled_var.set(enabled)
        self.smoothness_window_var.set(window_size)
        self.smoothness_oscillation_var.set(oscillation_threshold)
        self.smoothness_reject_var.set(reject_on_violation)
        self._toggle_smoothness_controls()

    def _apply_driver_sweep_values(self, driver_params: dict[str, Any] | None) -> bool:
        """Populate driver sweep controls from driver parameters."""
        if not driver_params:
            return False

        self._set_fixed_sweep_value(
            "driver_m_particle", f"{driver_params.get('m_particle', 207.2):.6e}"
        )
        self._set_fixed_sweep_value(
            "driver_charge_sign", str(driver_params.get("charge_sign", 1.0))
        )
        self._set_fixed_sweep_value(
            "driver_pcount", str(driver_params.get("pcount", 5))
        )
        self._set_fixed_sweep_value(
            "driver_transv_mom", f"{driver_params.get('transv_mom', 0.0):.2e}"
        )
        self._set_fixed_sweep_value(
            "driver_transv_dist", f"{driver_params.get('transv_dist', -0.07998):.6e}"
        )
        self._set_fixed_sweep_value(
            "driver_starting_distance",
            f"{driver_params.get('starting_distance', 1000.0):.2e}",
        )
        driver_pz = driver_params.get("starting_Pz", -4925.0)
        driver_mass = driver_params.get("m_particle", 207.2)
        driver_energy = calculate_energy_from_pz(driver_pz, driver_mass)
        self._set_fixed_sweep_value("driver_energy_gev", f"{driver_energy:.6e}")
        self._set_fixed_sweep_value(
            "driver_stripped_ions", str(driver_params.get("stripped_ions", 54.0))
        )
        return True

    def _sync_main_gui_simulation_type(self, sim_type_value: str):
        """Sync the selected simulation type back to the main GUI, if available."""
        if not (self.gui_controller and hasattr(self.gui_controller, "sim_type_var")):
            return

        self.gui_controller.sim_type_var.set(sim_type_value)
        if hasattr(self.gui_controller, "sim_type_combo"):
            try:
                values_list = list(self.gui_controller.sim_type_combo["values"])
                if sim_type_value in values_list:
                    idx = values_list.index(sim_type_value)
                    self.gui_controller.sim_type_combo.current(idx)
                    self.gui_controller.root.update_idletasks()
            except Exception:
                pass

    def _sync_main_gui_visibility_state(self):
        """Refresh main GUI visibility state affected by simulation type."""
        if not self.gui_controller:
            return

        if hasattr(self.gui_controller, "_update_driver_visibility"):
            self.gui_controller._update_driver_visibility()
        if hasattr(self.gui_controller, "_update_image_subcharge_state"):
            self.gui_controller._update_image_subcharge_state()

    def _load_config_from_path(self, filepath: str) -> None:
        """Load configuration from a specific file path."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self.last_loaded_config = filepath

            if self.gui_controller and hasattr(
                self.gui_controller, "sweep_config_name_var"
            ):
                self.gui_controller.sweep_config_name_var.set(Path(filepath).name)

            sim_type_value = data.get("simulation_type", "CONDUCTING_WALL")
            self.sim_type_var.set(sim_type_value)
            self._sync_main_gui_simulation_type(sim_type_value)

            self.mode_var.set(data.get("mode", "blind_sweep"))
            self.aperture_min_var.set(str(data.get("aperture_min", 1e-5)))
            self.aperture_max_var.set(str(data.get("aperture_max", 1e-3)))
            self.aperture_points_var.set(str(data.get("aperture_points", 10)))
            self.aperture_log_var.set(data.get("aperture_log_scale", True))
            self.energy_min_var.set(str(data.get("energy_min", 1.0)))
            self.energy_max_var.set(str(data.get("energy_max", 1000.0)))
            self.energy_points_var.set(str(data.get("energy_points", 10)))
            self.energy_log_var.set(data.get("energy_log_scale", True))
            self.offset_fractions_var.set(
                ", ".join(
                    map(str, data.get("transverse_offset_fractions", [0.1, 0.3, 0.5]))
                )
            )
            start_z_list = data.get("starting_z_positions", [0.0])
            self.start_z_var.set(str(start_z_list[0] if start_z_list else 0.0))
            self.wall_z_var.set(str(data.get("wall_z", 100.0)))

            if "wall_z_range" in data and data["wall_z_range"] is not None:
                wall_z_range = data["wall_z_range"]
                self.wall_z_min_var.set(str(wall_z_range[0]))
                self.wall_z_max_var.set(str(wall_z_range[1]))
                self.wall_z_points_var.set(str(data.get("wall_z_points", 3)))
                self.wall_z_sweep_var.set(True)
                self._toggle_wall_z_sweep()
            else:
                self.wall_z_sweep_var.set(False)
                self._toggle_wall_z_sweep()

            self.cavity_spacing_var.set(str(data.get("cavity_spacing", 1e5)))
            self.steps_var.set(str(data.get("steps", 2000)))
            self.objective_var.set(data.get("objective", "max_energy_gain"))

            self.save_top_n_traj_var.set(data.get("save_top_n_trajectories", False))
            self.save_all_traj_var.set(data.get("save_all_trajectories", False))
            self.save_failed_traj_var.set(data.get("save_failed_trajectories", False))
            self.trajectory_stride_var.set(str(data.get("trajectory_stride", 10)))

            metrics_format, metrics_scope = metrics_export_settings_from_data(data)
            self.metrics_format_var.set(metrics_format)
            self.metrics_scope_var.set(metrics_scope)
            self.log_verbosity_var.set(data.get("log_verbosity", "truncated"))

            self.optimization_method_var.set(
                data.get("optimization_method", "genetic_algorithm")
            )
            self.optimization_maxiter_var.set(str(data.get("optimization_maxiter", 50)))
            self.optimization_popsize_var.set(
                str(data.get("optimization_population_size", 20))
            )
            self.optimization_mutation_var.set(
                str(data.get("optimization_mutation_rate", 0.1))
            )
            self.optimization_crossover_var.set(
                str(data.get("optimization_crossover_rate", 0.7))
            )
            self.optimization_nstarts_var.set(str(data.get("optimization_n_starts", 5)))
            self.optimization_save_top_n_var.set(
                str(data.get("optimization_save_top_n", 3))
            )
            self.optimization_convergence_tol_var.set(
                str(data.get("optimization_convergence_tol", 1e-6))
            )
            self.optimization_convergence_patience_var.set(
                str(data.get("optimization_convergence_patience", 10))
            )

            self._update_mode_visibility()
            self._update_optimization_controls()

            loaded_config = self._gather_config()
            loaded_config = apply_persisted_config_overrides(loaded_config, data)

            print("[DEBUG] _load_config_from_path: Assigning loaded_config to self.config")
            print(f"  SC enabled: {loaded_config.self_consistency_enabled}")
            print(f"  SC tolerance: {loaded_config.self_consistency_tolerance}")
            print(f"  AT enabled: {loaded_config.adaptive_timestep_enabled}")
            print(f"  AT debug: {loaded_config.adaptive_timestep_debug}")
            self.config = loaded_config

            self.per_run_timeout_var.set(str(loaded_config.per_run_timeout))
            self.skip_failed_runs_var.set(loaded_config.skip_failed_runs)
            self.failed_run_retry_attempts_var.set(
                str(loaded_config.failed_run_retry_attempts)
            )

            self.timestep_mode_var.set(data.get("timestep_mode", "duration"))
            self.auto_steps_distance_var.set(str(data.get("auto_steps_distance", 10.0)))
            self.trajectory_stride_var.set(str(data.get("trajectory_stride", 10)))
            self.sweep_params["rider_stripped_ions"]["fixed_var"].set(
                str(data.get("rider_stripped_ions", 1.0))
            )
            rider_x = data.get("rider_offset_x", 0.0)
            rider_y = data.get("rider_offset_y", 0.0)
            self.offset_fractions_var.set(f"{rider_x}, {rider_y}")
            driver_x = data.get("driver_offset_x", 0.0)
            driver_y = data.get("driver_offset_y", 0.0)
            self.driver_offset_var.set(f"{driver_x}, {driver_y}")
            self.sweep_params["driver_stripped_ions"]["fixed_var"].set(
                str(data.get("driver_stripped_ions", 54.0))
            )
            self._toggle_timestep_mode()

            self._apply_smoothness_ui_state(
                enabled=loaded_config.smoothness_enabled,
                window_size=str(loaded_config.smoothness_window_size),
                oscillation_threshold=str(
                    loaded_config.smoothness_oscillation_threshold
                ),
                reject_on_violation=loaded_config.smoothness_reject_on_violation,
            )

            if self.gui_controller:
                self._sync_stability_to_main_gui(loaded_config)
                if hasattr(self.gui_controller, "image_subcharge_var"):
                    self.gui_controller.image_subcharge_var.set(
                        loaded_config.image_subcharge_count
                    )
                if hasattr(self.gui_controller, "image_weighting_var"):
                    self.gui_controller.image_weighting_var.set(
                        loaded_config.use_image_weighting
                    )

            self._log_result("[INFO] Additional stability settings loaded:")
            self._log_result(
                f"  Self-consistency max_iterations: {loaded_config.self_consistency_max_iterations}"
            )
            self._log_result(
                f"  Self-consistency verbosity: {loaded_config.self_consistency_verbosity}"
            )
            self._log_result(
                f"  Self-consistency chrono_interpolate: {loaded_config.self_consistency_chrono_interpolate}"
            )
            self._log_result(
                f"  Self-consistency chrono_tolerance: {loaded_config.self_consistency_chrono_tolerance:.1e} ns"
            )
            self._log_result(
                f"  Self-consistency chrono_high_precision: {loaded_config.self_consistency_chrono_high_precision}"
            )
            self._log_result(
                f"  Self-consistency chrono_adaptive_tolerance: {loaded_config.self_consistency_chrono_adaptive_tolerance}"
            )
            self._log_result(
                f"  Adaptive timestep reduction_factor: {loaded_config.adaptive_timestep_reduction_factor}"
            )
            self._log_result(
                f"  Adaptive timestep min_factor: {loaded_config.adaptive_timestep_min_factor}"
            )
            self._log_result(
                f"  Adaptive timestep min_factor: {loaded_config.adaptive_timestep_min_factor}"
            )
            self._log_result(
                f"  Adaptive timestep cooldown_steps: {loaded_config.adaptive_timestep_cooldown_steps}"
            )
            self._log_result(
                f"  Adaptive timestep probe_threshold: {loaded_config.adaptive_timestep_probe_threshold}"
            )
            self._log_result(
                f"  Adaptive timestep max_probe_steps: {loaded_config.adaptive_timestep_max_probe_steps}"
            )
            self._log_result(
                f"  Smoothness trend_threshold: {loaded_config.smoothness_trend_threshold}"
            )
            self._log_result(
                f"  Smoothness max_violations: {loaded_config.smoothness_max_violations}"
            )

            self._apply_macroparticle_ui_state(
                enabled=loaded_config.macroparticle_enabled,
                charge_multiplier=str(loaded_config.macroparticle_charge_multiplier),
                sigma_multiplier=str(loaded_config.macroparticle_sigma_multiplier),
                momentum_errors=loaded_config.macroparticle_use_momentum_errors,
                refresh_state=True,
            )

            sweep_state = data.get("sweep_parameters", {})
            if hasattr(self, "driver_direction_var"):
                direction = data.get("driver_direction", "-z")
                self.driver_direction_var.set(
                    direction if direction in ("-z", "+z") else "-z"
                )

            for param_name, controls in self.sweep_params.items():
                state = resolve_loaded_sweep_state(sweep_state, param_name)
                if state is None:
                    continue

                if state.get("enabled", False):
                    controls["sweep_var"].set(True)
                    controls["min_var"].set(str(state.get("min", "")))
                    controls["max_var"].set(str(state.get("max", "")))
                    controls["points_var"].set(str(state.get("points", "3")))
                    controls["log_var"].set(state.get("log", False))
                    self._toggle_sweep_controls(param_name)
                else:
                    controls["sweep_var"].set(False)
                    fixed_val = state.get("fixed_value", controls["fixed_var"].get())
                    controls["fixed_var"].set(str(fixed_val))
                    self._toggle_sweep_controls(param_name)

            if hasattr(self, "link_driver_rider_energy_var"):
                linked_energy = data.get("linked_energy_sweep", False)
                self.link_driver_rider_energy_var.set(linked_energy)
                self._on_link_energy_toggled()
                if linked_energy:
                    self._log_result(
                        "[INFO] Linked energy sweep mode enabled - driver energy follows rider energy"
                    )

            self._update_driver_visibility()
            self._update_rider_pz_helper()
            self._update_driver_pz_helper()

            self._log_result("[OK] Configuration loaded and synced to main GUI")
            self._log_result("")
            self._log_result("=" * 60)
            self._log_result("LOADED STABILITY OPTIONS SUMMARY")
            self._log_result("=" * 60)
            self._log_result("[Self-Consistency]")
            self._log_result(f"  Enabled: {self.config.self_consistency_enabled}")
            self._log_result(
                f"  Tolerance: {self.config.self_consistency_tolerance:.1e}"
            )
            self._log_result(
                f"  Max iterations: {self.config.self_consistency_max_iterations}"
            )
            self._log_result(f"  Verbosity: {self.config.self_consistency_verbosity}")
            self._log_result(
                f"  Chrono interpolate: {self.config.self_consistency_chrono_interpolate}"
            )
            self._log_result(
                f"  Chrono tolerance: {self.config.self_consistency_chrono_tolerance:.1e} ns"
            )
            self._log_result(
                f"  Chrono high precision: {self.config.self_consistency_chrono_high_precision}"
            )
            self._log_result(
                f"  Chrono adaptive tolerance: {self.config.self_consistency_chrono_adaptive_tolerance}"
            )
            self._log_result("")
            self._log_result("[Adaptive Timestep]")
            self._log_result(f"  Enabled: {self.config.adaptive_timestep_enabled}")
            self._log_result(
                f"  Threshold: {self.config.adaptive_timestep_threshold * 100:.0f}%"
            )
            self._log_result(
                f"  Reduction factor: {self.config.adaptive_timestep_reduction_factor}x"
            )
            self._log_result(
                f"  Min timestep factor: {self.config.adaptive_timestep_min_factor}"
            )
            self._log_result(
                f"  Min factor: {self.config.adaptive_timestep_min_factor}"
            )
            self._log_result(
                f"  Cooldown steps: {self.config.adaptive_timestep_cooldown_steps}"
            )
            self._log_result(
                f"  Probe threshold: {self.config.adaptive_timestep_probe_threshold}"
            )
            self._log_result(
                f"  Max probe steps: {self.config.adaptive_timestep_max_probe_steps}"
            )
            self._log_result(f"  Debug: {self.config.adaptive_timestep_debug}")
            self._log_result("")
            self._log_result("[Trajectory Smoothness Analysis]")
            self._log_result(f"  Enabled: {self.config.smoothness_enabled}")
            self._log_result(f"  Window size: {self.config.smoothness_window_size}")
            self._log_result(
                f"  Oscillation threshold: {self.config.smoothness_oscillation_threshold}"
            )
            self._log_result(
                f"  Trend threshold: {self.config.smoothness_trend_threshold}"
            )
            self._log_result(
                f"  Reject on violation: {self.config.smoothness_reject_on_violation}"
            )
            self._log_result(
                f"  Max violations: {self.config.smoothness_max_violations}"
            )
            self._log_result("")
            self._log_result("=" * 60)
            self._log_result("")
            self._log_result(
                "NOTE: Stability settings are synced to main GUI's Stability tab"
            )
            self._log_result("      View/edit them in the main GUI's Stability tab")
            self._log_result(
                "      Log verbosity setting will override debug flags during run"
            )
            self._log_result("")

            if self.gui_controller and hasattr(self.gui_controller, "run_mode_var"):
                self.gui_controller.run_mode_var.set("sweep")
                if hasattr(self.gui_controller, "_on_run_mode_changed"):
                    self.gui_controller._on_run_mode_changed()
                self._log_result(
                    "[INFO] Auto-switched main GUI to Sweep/Optim run mode"
                )
                self._sync_main_gui_visibility_state()

        except Exception as e:
            _show_error_dialog(self, "Load Error", f"Failed to load config: {e}")

    def _on_load_config(self):
        """Load configuration from JSON file via dialog."""
        os.makedirs(self.sweep_config_dir, exist_ok=True)

        filename = filedialog.askopenfilename(
            title="Load Optimization Config",
            initialdir=self.sweep_config_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return

        self._load_config_from_path(filename)

    def _save_config_to_path(self, filepath: str) -> bool:
        """Save configuration to specified path."""
        error = self._validate_inputs()
        if error:
            _show_error_dialog(self, "Invalid Input", f"Cannot save: {error}")
            return False

        try:
            print("[DEBUG] _save_config_to_path: Gathering config for save")
            config = self._gather_config()
            print("[DEBUG] After _gather_config:")
            print(f"  SC enabled: {config.self_consistency_enabled}")
            print(f"  SC tolerance: {config.self_consistency_tolerance}")
            print(
                f"  SC chrono interpolate: {config.self_consistency_chrono_interpolate}"
            )
            print(f"  SC chrono tolerance: {config.self_consistency_chrono_tolerance}")
            print(
                f"  SC chrono high precision: {config.self_consistency_chrono_high_precision}"
            )
            print(
                f"  SC chrono adaptive tolerance: {config.self_consistency_chrono_adaptive_tolerance}"
            )
            print(f"  AT enabled: {config.adaptive_timestep_enabled}")
            print(f"  AT debug: {config.adaptive_timestep_debug}")

            sweep_state = {}
            for param_name, controls in self.sweep_params.items():
                if controls["sweep_var"].get():
                    sweep_state[param_name] = {
                        "enabled": True,
                        "min": controls["min_var"].get(),
                        "max": controls["max_var"].get(),
                        "points": controls["points_var"].get(),
                        "log": controls["log_var"].get(),
                    }
                else:
                    sweep_state[param_name] = {
                        "enabled": False,
                        "fixed_value": controls["fixed_var"].get(),
                    }

            data = build_saved_config_payload(
                config,
                timestep_mode=self.timestep_mode_var.get(),
                auto_steps_distance=float(self.auto_steps_distance_var.get()),
                rider_stripped_ions=float(
                    self.sweep_params["rider_stripped_ions"]["fixed_var"].get()
                ),
                driver_stripped_ions=float(
                    self.sweep_params["driver_stripped_ions"]["fixed_var"].get()
                ),
                driver_direction=getattr(
                    self, "driver_direction_var", tk.StringVar(value="-z")
                ).get(),
                sweep_state=sweep_state,
            )

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            self.last_loaded_config = filepath
            self._log_result(f"[OK] Configuration saved to {filepath}")
            print("[DEBUG] Chrono settings saved to config:")
            print(f"  chrono_interpolate: {config.self_consistency_chrono_interpolate}")
            print(f"  chrono_tolerance: {config.self_consistency_chrono_tolerance}")
            print(
                f"  chrono_high_precision: {config.self_consistency_chrono_high_precision}"
            )
            print(
                f"  chrono_adaptive_tolerance: {config.self_consistency_chrono_adaptive_tolerance}"
            )
            return True
        except Exception as e:
            _show_error_dialog(self, "Save Error", f"Failed to save config: {e}")
            return False

    def _on_save_config(self):
        """Save configuration to JSON file using file dialog."""
        error = self._validate_inputs()
        if error:
            _show_error_dialog(self, "Invalid Input", f"Cannot save: {error}")
            return

        os.makedirs(self.sweep_config_dir, exist_ok=True)

        filename = filedialog.asksaveasfilename(
            title="Save Optimization Config",
            initialdir=self.sweep_config_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return

        success = self._save_config_to_path(filename)

        if (
            success
            and self.gui_controller
            and hasattr(self.gui_controller, "sweep_config_name_var")
        ):
            config_name = Path(filename).name
            self.gui_controller.sweep_config_name_var.set(config_name)
            self.gui_controller.current_sweep_config_label.config(
                text=config_name, foreground="black", font=("TkDefaultFont", 9)
            )
            self.gui_controller._refresh_sweep_config_list(selected=config_name)
