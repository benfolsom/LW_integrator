"""Run-preparation and control helpers for the optimization plugin."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox
from typing import Optional

from core.types import SimulationType
from optimization.config import OptimizationConfig
from optimization.plugin_config_helpers import (
    apply_sweep_parameter_overrides,
    parse_float_list,
    parse_offset_pair,
)
from optimization.run_control_helpers import (
    SweepParameterValidationInput,
    build_extreme_parameter_warning,
    validate_optimization_inputs,
)
from optimization.simulation_type_helpers import is_bunch_to_bunch
from optimization.ui_helpers import (
    show_error_dialog as _show_error_dialog,
)


def _existing_config_value(config: OptimizationConfig | None, attr: str, default):
    return getattr(config, attr) if config is not None else default


class OptimizationPluginControlMixin:
    """Validate inputs and prepare or control optimization runs."""

    def _validate_inputs(self) -> Optional[str]:
        """Validate user inputs. Returns error message or None."""
        sweep_parameters = [
            SweepParameterValidationInput(
                name=param_name,
                swept=controls["sweep_var"].get(),
                min_value=controls["min_var"].get(),
                max_value=controls["max_var"].get(),
                points=controls["points_var"].get(),
                fixed_value=controls["fixed_var"].get(),
            )
            for param_name, controls in self.sweep_params.items()
        ]
        return validate_optimization_inputs(
            simulation_type=self.sim_type_var.get(),
            aperture_min=self.aperture_min_var.get(),
            aperture_max=self.aperture_max_var.get(),
            aperture_points=self.aperture_points_var.get(),
            energy_min=self.energy_min_var.get(),
            energy_max=self.energy_max_var.get(),
            energy_points=self.energy_points_var.get(),
            mode=self.mode_var.get(),
            offset_fractions=self.offset_fractions_var.get(),
            start_z=self.start_z_var.get(),
            wall_z=self.wall_z_var.get(),
            steps=self.steps_var.get(),
            auto_steps_distance=self.auto_steps_distance_var.get(),
            workers=self.workers_var.get(),
            sweep_parameters=sweep_parameters,
        )

    def _get_gui_stability_setting(self, var_name: str, default_value):
        """Get a stability setting from the main GUI if available."""
        if self.gui_controller and hasattr(self.gui_controller, var_name):
            var = getattr(self.gui_controller, var_name)
            value = var.get()
            if isinstance(value, str):
                if (
                    "tolerance" in var_name
                    or "threshold" in var_name
                    or "factor" in var_name
                ):
                    try:
                        return float(value)
                    except ValueError:
                        return default_value
                if (
                    "iterations" in var_name
                    or "verbosity" in var_name
                    or "attempts" in var_name
                    or "steps" in var_name
                ):
                    try:
                        return int(value)
                    except ValueError:
                        return default_value
            return value
        return default_value

    def _gather_stability_config_kwargs(
        self, existing_config: OptimizationConfig | None
    ) -> dict:
        """Return stability-related ``OptimizationConfig`` keyword arguments."""
        kwargs = self._gather_image_and_self_consistency_kwargs(existing_config)
        kwargs.update(self._gather_adaptive_timestep_kwargs(existing_config))
        kwargs.update(self._gather_gamma_reconciliation_kwargs(existing_config))
        kwargs["radiation_reaction_mode"] = (
            str(self.radiation_reaction_mode_var.get())
            if hasattr(self, "radiation_reaction_mode_var")
            else _existing_config_value(
                existing_config,
                "radiation_reaction_mode",
                "medina_lad",
            )
        )
        return kwargs

    def _gather_image_and_self_consistency_kwargs(
        self, existing_config: OptimizationConfig | None
    ) -> dict:
        """Return image-charge and self-consistency config keyword arguments."""
        config_value = _existing_config_value

        def setting(var_name: str, attr: str, default):
            return self._get_gui_stability_setting(
                var_name, config_value(existing_config, attr, default)
            )

        return {
            "image_subcharge_count": setting(
                "image_subcharge_var", "image_subcharge_count", 12
            ),
            "use_image_weighting": setting(
                "image_weighting_var", "use_image_weighting", True
            ),
            "self_consistency_enabled": setting(
                "self_consistency_enabled_var", "self_consistency_enabled", True
            ),
            "self_consistency_tolerance": setting(
                "self_consistency_target_ms_tolerance_var",
                "self_consistency_tolerance",
                1e-4,
            ),
            "self_consistency_convergence_mode": setting(
                "self_consistency_convergence_mode_var",
                "self_consistency_convergence_mode",
                "fixed_geometry",
            ),
            "self_consistency_target_ms_tolerance": setting(
                "self_consistency_target_ms_tolerance_var",
                "self_consistency_target_ms_tolerance",
                1e-6,
            ),
            "self_consistency_max_iterations": setting(
                "self_consistency_max_iterations_var",
                "self_consistency_max_iterations",
                2,
            ),
            "self_consistency_mass_shell_tolerance": setting(
                "self_consistency_mass_shell_tolerance_var",
                "self_consistency_mass_shell_tolerance",
                1e-2,
            ),
            "self_consistency_mass_shell_relaxation": setting(
                "self_consistency_mass_shell_relaxation_var",
                "self_consistency_mass_shell_relaxation",
                0.7,
            ),
            "self_consistency_verbosity": setting(
                "self_consistency_verbosity_var", "self_consistency_verbosity", 0
            ),
            "self_consistency_chrono_interpolate": setting(
                "self_consistency_chrono_interpolate_var",
                "self_consistency_chrono_interpolate",
                False,
            ),
            "self_consistency_chrono_tolerance": setting(
                "self_consistency_chrono_tolerance_var",
                "self_consistency_chrono_tolerance",
                1e-3,
            ),
            "self_consistency_chrono_matching_mode": config_value(
                existing_config,
                "self_consistency_chrono_matching_mode",
                "FAST",
            ),
            "self_consistency_chrono_high_precision": setting(
                "self_consistency_chrono_high_precision_var",
                "self_consistency_chrono_high_precision",
                False,
            ),
            "self_consistency_chrono_adaptive_tolerance": setting(
                "self_consistency_chrono_adaptive_tolerance_var",
                "self_consistency_chrono_adaptive_tolerance",
                False,
            ),
        }

    def _gather_adaptive_timestep_kwargs(
        self, existing_config: OptimizationConfig | None
    ) -> dict:
        """Return adaptive-timestep config keyword arguments."""
        config_value = _existing_config_value

        def setting(var_name: str, attr: str, default):
            return self._get_gui_stability_setting(
                var_name, config_value(existing_config, attr, default)
            )

        return {
            "energy_monitor_halt_on_jump": setting(
                "adaptive_timestep_halt_on_jump_var",
                "energy_monitor_halt_on_jump",
                False,
            ),
            "adaptive_timestep_enabled": setting(
                "adaptive_timestep_enabled_var", "adaptive_timestep_enabled", True
            ),
            "adaptive_timestep_threshold": setting(
                "adaptive_timestep_threshold_var", "adaptive_timestep_threshold", 0.10
            ),
            "adaptive_timestep_reduction_factor": setting(
                "adaptive_timestep_reduction_factor_var",
                "adaptive_timestep_reduction_factor",
                10,
            ),
            "adaptive_timestep_min_factor": setting(
                "adaptive_timestep_min_factor_var", "adaptive_timestep_min_factor", 1e-4
            ),
            "adaptive_timestep_cooldown_steps": setting(
                "adaptive_timestep_cooldown_steps_var",
                "adaptive_timestep_cooldown_steps",
                10,
            ),
            "adaptive_timestep_probe_threshold": setting(
                "adaptive_timestep_probe_threshold_var",
                "adaptive_timestep_probe_threshold",
                0.01,
            ),
            "adaptive_timestep_max_probe_steps": setting(
                "adaptive_timestep_max_probe_steps_var",
                "adaptive_timestep_max_probe_steps",
                3,
            ),
            "adaptive_timestep_debug": setting(
                "adaptive_timestep_debug_var", "adaptive_timestep_debug", False
            ),
            "adaptive_timestep_bunch_proximity_enabled": setting(
                "adaptive_timestep_bunch_proximity_enabled_var",
                "adaptive_timestep_bunch_proximity_enabled",
                False,
            ),
            "adaptive_timestep_bunch_proximity_sigma_mm": setting(
                "adaptive_timestep_bunch_proximity_sigma_mm_var",
                "adaptive_timestep_bunch_proximity_sigma_mm",
                5.0,
            ),
            "adaptive_timestep_bunch_proximity_n_sigma": setting(
                "adaptive_timestep_bunch_proximity_n_sigma_var",
                "adaptive_timestep_bunch_proximity_n_sigma",
                5.0,
            ),
            "adaptive_timestep_bunch_proximity_reduction_factor": setting(
                "adaptive_timestep_bunch_proximity_reduction_factor_var",
                "adaptive_timestep_bunch_proximity_reduction_factor",
                10.0,
            ),
            "adaptive_timestep_bunch_proximity_transition_n_sigma": setting(
                "adaptive_timestep_bunch_proximity_transition_n_sigma_var",
                "adaptive_timestep_bunch_proximity_transition_n_sigma",
                2.0,
            ),
        }

    def _gather_gamma_reconciliation_kwargs(
        self, existing_config: OptimizationConfig | None
    ) -> dict:
        """Return self-consistency gamma-reconciliation keyword arguments."""
        config_value = _existing_config_value

        def setting(var_name: str, attr: str, default):
            return self._get_gui_stability_setting(
                var_name, config_value(existing_config, attr, default)
            )

        return {
            "self_consistency_gamma_reconciliation_method": setting(
                "self_consistency_gamma_reconciliation_method_var",
                "self_consistency_gamma_reconciliation_method",
                "DISABLED",
            ),
            "self_consistency_gamma_reconciliation_low_beta_threshold": setting(
                "self_consistency_gamma_reconciliation_low_beta_threshold_var",
                "self_consistency_gamma_reconciliation_low_beta_threshold",
                0.9,
            ),
            "self_consistency_gamma_reconciliation_high_beta_threshold": setting(
                "self_consistency_gamma_reconciliation_high_beta_threshold_var",
                "self_consistency_gamma_reconciliation_high_beta_threshold",
                0.99,
            ),
            "self_consistency_gamma_reconciliation_low_beta_weight": setting(
                "self_consistency_gamma_reconciliation_low_beta_weight_var",
                "self_consistency_gamma_reconciliation_low_beta_weight",
                0.8,
            ),
            "self_consistency_gamma_reconciliation_high_beta_weight": setting(
                "self_consistency_gamma_reconciliation_high_beta_weight_var",
                "self_consistency_gamma_reconciliation_high_beta_weight",
                0.2,
            ),
            "self_consistency_gamma_reconciliation_mid_beta_weight": setting(
                "self_consistency_gamma_reconciliation_mid_beta_weight_var",
                "self_consistency_gamma_reconciliation_mid_beta_weight",
                0.5,
            ),
            "self_consistency_gamma_reconciliation_fixed_weight": setting(
                "self_consistency_gamma_reconciliation_fixed_weight_var",
                "self_consistency_gamma_reconciliation_fixed_weight",
                0.5,
            ),
        }

    def _gather_search_config_kwargs(
        self, existing_config: OptimizationConfig | None
    ) -> dict:
        """Return search-space and integration-grid config keyword arguments."""
        kwargs = self._gather_optimization_algorithm_kwargs()
        kwargs.update(self._gather_sweep_grid_kwargs())
        kwargs.update(self._gather_integration_grid_kwargs(existing_config))
        return kwargs

    def _gather_optimization_algorithm_kwargs(self) -> dict:
        """Return optimization algorithm config keyword arguments."""
        return {
            "mode": self.mode_var.get(),
            "optimization_method": self.optimization_method_var.get(),
            "optimization_maxiter": int(self.optimization_maxiter_var.get()),
            "optimization_population_size": int(self.optimization_popsize_var.get()),
            "optimization_mutation_rate": float(self.optimization_mutation_var.get()),
            "optimization_crossover_rate": float(self.optimization_crossover_var.get()),
            "optimization_n_starts": int(self.optimization_nstarts_var.get()),
            "optimization_save_top_n": int(self.optimization_save_top_n_var.get()),
            "optimization_convergence_tol": float(
                self.optimization_convergence_tol_var.get()
            ),
            "optimization_convergence_patience": int(
                self.optimization_convergence_patience_var.get()
            ),
            "objective": self.objective_var.get(),
        }

    def _gather_sweep_grid_kwargs(self) -> dict:
        """Return sweep grid config keyword arguments."""
        return {
            "simulation_type": SimulationType[self.sim_type_var.get()],
            "aperture_range": (
                float(self.aperture_min_var.get()),
                float(self.aperture_max_var.get()),
            ),
            "aperture_points": (
                1
                if is_bunch_to_bunch(self.sim_type_var.get())
                else int(self.aperture_points_var.get())
            ),
            "aperture_log_scale": self.aperture_log_var.get(),
            "energy_range": (
                float(self.energy_min_var.get()),
                float(self.energy_max_var.get()),
            ),
            "energy_points": int(self.energy_points_var.get()),
            "energy_log_scale": self.energy_log_var.get(),
            "transverse_offset_fractions": parse_float_list(
                self.offset_fractions_var.get()
            ),
            "starting_z_positions": [float(self.start_z_var.get())],
            "wall_z": float(self.wall_z_var.get()),
            "wall_z_range": (
                (
                    float(self.wall_z_min_var.get()),
                    float(self.wall_z_max_var.get()),
                )
                if self.wall_z_sweep_var.get()
                else None
            ),
            "wall_z_points": (
                int(self.wall_z_points_var.get()) if self.wall_z_sweep_var.get() else 1
            ),
        }

    def _gather_integration_grid_kwargs(
        self, existing_config: OptimizationConfig | None
    ) -> dict:
        """Return integration timing and startup config keyword arguments."""
        config_value = _existing_config_value

        return {
            "cavity_spacing": float(self.cavity_spacing_var.get()),
            "timestep": (
                float(self.duration_var.get())
                if self.timestep_mode_var.get() == "count"
                else 3e-7
            ),
            "steps": (
                int(self.steps_var.get())
                if self.timestep_mode_var.get() == "duration"
                else 200
            ),
            "auto_steps": True,
            "auto_steps_target": (
                int(self.steps_var.get())
                if self.timestep_mode_var.get() == "duration"
                else 200
            ),
            "auto_steps_distance_past_wall": float(self.auto_steps_distance_var.get()),
            "timestep_strategy": "auto_distance",
            "target_distance_mm": config_value(
                existing_config, "target_distance_mm", 100.0
            ),
            "energy_scale_exponent": config_value(
                existing_config, "energy_scale_exponent", 1.0
            ),
            "startup_mode": (
                self.gui_controller.core_param_vars["startup_mode"].get()
                if self.gui_controller
                and hasattr(self.gui_controller, "core_param_vars")
                else config_value(existing_config, "startup_mode", "COLD_START")
            ),
        }

    def _gather_particle_config_kwargs(
        self, rider_offset: tuple[float, float], driver_offset: tuple[float, float]
    ) -> dict:
        """Return particle and macroparticle config keyword arguments."""
        rider_transverse_geometry_var = getattr(
            self, "rider_transverse_geometry_var", None
        )
        driver_transverse_geometry_var = getattr(
            self, "driver_transverse_geometry_var", None
        )
        return {
            "transv_mom": float(
                self.sweep_params["rider_transv_mom"]["fixed_var"].get()
            ),
            "transv_dist": float(
                self.sweep_params["rider_transv_dist"]["fixed_var"].get()
            ),
            "transverse_geometry": (
                rider_transverse_geometry_var.get()
                if rider_transverse_geometry_var is not None
                else "square"
            ),
            "transv_offset_x": rider_offset[0],
            "transv_offset_y": rider_offset[1],
            "driver_transv_offset_x": driver_offset[0],
            "driver_transv_offset_y": driver_offset[1],
            "macroparticle_enabled": bool(self.macroparticle_enabled_var.get()),
            "macroparticle_charge_multiplier": float(
                self.sweep_params["macroparticle_charge_multiplier"]["fixed_var"].get()
            ),
            "macroparticle_sigma_multiplier": float(
                self.sweep_params["macroparticle_sigma_multiplier"]["fixed_var"].get()
            ),
            "macroparticle_use_momentum_errors": bool(
                self.macroparticle_momentum_errors_var.get()
            ),
            "m_particle": float(
                self.sweep_params["rider_m_particle"]["fixed_var"].get()
            ),
            "pcount": int(self.sweep_params["rider_pcount"]["fixed_var"].get()),
            "charge_sign": float(
                self.sweep_params["rider_charge_sign"]["fixed_var"].get()
            ),
            "stripped_ions": float(
                self.sweep_params["rider_stripped_ions"]["fixed_var"].get()
            ),
            "driver_m_particle": float(
                self.sweep_params["driver_m_particle"]["fixed_var"].get()
            ),
            "driver_charge_sign": float(
                self.sweep_params["driver_charge_sign"]["fixed_var"].get()
            ),
            "driver_pcount": int(self.sweep_params["driver_pcount"]["fixed_var"].get()),
            "driver_transv_mom": float(
                self.sweep_params["driver_transv_mom"]["fixed_var"].get()
            ),
            "driver_transv_dist": float(
                self.sweep_params["driver_transv_dist"]["fixed_var"].get()
            ),
            "driver_transverse_geometry": (
                driver_transverse_geometry_var.get()
                if driver_transverse_geometry_var is not None
                else "square"
            ),
            "driver_starting_distance": float(
                self.sweep_params["driver_starting_distance"]["fixed_var"].get()
            ),
            "driver_stripped_ions": float(
                self.sweep_params["driver_stripped_ions"]["fixed_var"].get()
            ),
        }

    def _gather_output_and_failure_kwargs(
        self, existing_config: OptimizationConfig | None
    ) -> dict:
        """Return result-output, smoothness, and failure-policy keyword arguments."""
        config_value = _existing_config_value

        return {
            "save_top_n_trajectories": bool(self.save_top_n_traj_var.get()),
            "save_all_trajectories": bool(self.save_all_traj_var.get()),
            "save_failed_trajectories": bool(self.save_failed_traj_var.get()),
            "trajectory_stride": int(self.trajectory_stride_var.get()),
            "metrics_export_format": str(self.metrics_format_var.get()),
            "metrics_export_scope": str(self.metrics_scope_var.get()),
            "log_verbosity": str(self.log_verbosity_var.get()),
            "smoothness_enabled": self.smoothness_enabled_var.get(),
            "smoothness_window_size": int(self.smoothness_window_var.get()),
            "smoothness_oscillation_threshold": float(
                self.smoothness_oscillation_var.get()
            ),
            "smoothness_reject_on_violation": self.smoothness_reject_var.get(),
            "smoothness_trend_threshold": config_value(
                existing_config, "smoothness_trend_threshold", 0.30
            ),
            "smoothness_max_violations": config_value(
                existing_config, "smoothness_max_violations", 3
            ),
            "workers": int(self.workers_var.get()),
            "per_run_timeout": float(self.per_run_timeout_var.get()),
            "skip_failed_runs": self.skip_failed_runs_var.get(),
            "failed_run_retry_attempts": int(self.failed_run_retry_attempts_var.get()),
        }

    def _gather_config(self) -> OptimizationConfig:
        """Gather configuration from UI fields."""
        existing_config = getattr(self, "config", None)

        rider_offset = parse_offset_pair(self.offset_fractions_var.get())
        driver_offset = parse_offset_pair(self.driver_offset_var.get())

        config_obj = OptimizationConfig(
            **self._gather_search_config_kwargs(existing_config),
            **self._gather_particle_config_kwargs(rider_offset, driver_offset),
            **self._gather_output_and_failure_kwargs(existing_config),
            **self._gather_stability_config_kwargs(existing_config),
        )

        driver_negative = (
            getattr(self, "driver_direction_var", None) is None
            or getattr(self, "driver_direction_var").get() == "-z"
        )
        linked_energy_sweep = getattr(
            self, "link_driver_rider_energy_var", tk.BooleanVar(value=False)
        ).get()

        return apply_sweep_parameter_overrides(
            config_obj,
            self.sweep_params,
            driver_negative=driver_negative,
            linked_energy_sweep=linked_energy_sweep,
            debug=print,
        )

    def _check_extreme_parameters(self) -> Optional[str]:
        """Check for extreme parameter combinations that might cause issues."""
        return build_extreme_parameter_warning(self.config)

    def _on_run_sweep(self):
        """Handle run sweep button click."""
        if self.gui_controller and hasattr(self.gui_controller, "_running"):
            if self.gui_controller._running:
                messagebox.showwarning(
                    "Optimization",
                    "Please wait for current simulation to complete",
                )
                return

        error = self._validate_inputs()
        if error:
            _show_error_dialog(self, "Invalid Input", error)
            return

        try:
            self.config = self._gather_config()

            extreme_warning = self._check_extreme_parameters()
            if extreme_warning:
                response = messagebox.askyesno(
                    "Extreme Parameters Warning", extreme_warning, icon="warning"
                )
                if not response:
                    self._log_result(
                        "[INFO] Sweep cancelled by user (extreme parameters)"
                    )
                    return

            self._log_result(
                "[INFO] Using stability options from main GUI Stability tab"
            )

            self.config.workers = int(self.workers_var.get())
            self.config.per_run_timeout = float(self.per_run_timeout_var.get())
            self.config.skip_failed_runs = self.skip_failed_runs_var.get()
            self.config.failed_run_retry_attempts = int(
                self.failed_run_retry_attempts_var.get()
            )

            self.config.smoothness_enabled = self.smoothness_enabled_var.get()
            self.config.smoothness_window_size = int(self.smoothness_window_var.get())
            self.config.smoothness_oscillation_threshold = float(
                self.smoothness_oscillation_var.get()
            )
            self.config.smoothness_reject_on_violation = (
                self.smoothness_reject_var.get()
            )

        except Exception as e:
            _show_error_dialog(self, "Configuration Error", str(e))
            return

        self._log_result(f"[INFO] Sweep worker processes: {self.config.workers}")

        self._was_cancelled = False
        self.running = True
        self._update_progress(0, "Initializing sweep...")

        if self.gui_controller and hasattr(self.gui_controller, "_running"):
            self.gui_controller._running = True
            if hasattr(self.gui_controller, "_cancel_requested"):
                self.gui_controller._cancel_requested = False
            if hasattr(self.gui_controller, "_set_status"):
                self.gui_controller._set_status("Running Optimization Sweep...")
            if hasattr(self.gui_controller, "_run_button"):
                self.gui_controller._run_button.configure(state="disabled")
            if hasattr(self.gui_controller, "_cancel_button"):
                self.gui_controller._cancel_button.configure(state="normal")

        thread = threading.Thread(target=self._run_sweep_background, daemon=True)
        thread.start()

    def _on_stop(self):
        """Handle stop button click."""
        self.running = False
        self._was_cancelled = True
        self._update_progress_text("Stopping...")

        if self.gui_controller and hasattr(self.gui_controller, "_cancel_requested"):
            self.gui_controller._cancel_requested = True

    def _set_fixed_sweep_value(self, param_name: str, value: str):
        """Update a fixed-value sweep control."""
        self.sweep_params[param_name]["fixed_var"].set(value)
