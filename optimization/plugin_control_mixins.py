"""Run-preparation and control helpers for the optimization plugin."""

from __future__ import annotations

import math
import threading
import tkinter as tk
from tkinter import messagebox, ttk
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


def _stability_dialog_logging_defaults(config: OptimizationConfig) -> tuple[str, bool]:
    """Return logging defaults without silently enabling debug output."""
    return str(config.self_consistency_verbosity), bool(config.adaptive_timestep_debug)


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
                5,
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
        return {
            "transv_mom": float(
                self.sweep_params["rider_transv_mom"]["fixed_var"].get()
            ),
            "transv_dist": float(
                self.sweep_params["rider_transv_dist"]["fixed_var"].get()
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

    def _confirm_stability_options(self) -> bool:
        """Show the stability options confirmation dialog."""
        dialog = tk.Toplevel(self)
        dialog.title("Confirm Stability Options")
        dialog.transient(self)
        dialog.grab_set()

        result = [False]

        main_frame = ttk.Frame(dialog, padding=15)
        main_frame.pack(fill="both", expand=True)

        info_label = ttk.Label(
            main_frame,
            text="The following stability options will be used for all sweep runs.\n"
            "These settings affect convergence, energy monitoring, and timestep adaptation.",
            wraplength=500,
            justify="left",
        )
        info_label.pack(pady=(0, 10))

        use_single_run_var = tk.BooleanVar(value=True)
        use_single_run_frame = ttk.Frame(main_frame)
        use_single_run_frame.pack(fill="x", pady=(0, 10))

        use_single_run_cb = ttk.Checkbutton(
            use_single_run_frame,
            text="Use single-run stability settings (uncheck for safer sweep defaults)",
            variable=use_single_run_var,
        )
        use_single_run_cb.pack(anchor="w")

        canvas = tk.Canvas(main_frame, height=300, width=550)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)

        scrollable.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        all_widgets = []

        sc_frame = ttk.LabelFrame(scrollable, text="Self-Consistency", padding=10)
        sc_frame.pack(fill="x", pady=5, padx=5)

        sc_enabled_var = tk.BooleanVar(value=self.config.self_consistency_enabled)
        sc_enabled_cb = ttk.Checkbutton(
            sc_frame, text="Enabled", variable=sc_enabled_var
        )
        sc_enabled_cb.pack(anchor="w")
        all_widgets.append(sc_enabled_cb)

        ttk.Label(sc_frame, text="Tolerance:").pack(anchor="w", pady=(5, 0))
        sc_tol_var = tk.StringVar(value=f"{self.config.self_consistency_tolerance:.1e}")
        sc_tol_entry = ttk.Entry(sc_frame, textvariable=sc_tol_var, width=15)
        sc_tol_entry.pack(anchor="w")
        all_widgets.append(sc_tol_entry)

        ttk.Label(sc_frame, text="Max iterations:").pack(anchor="w", pady=(5, 0))
        sc_iter_var = tk.StringVar(
            value=str(self.config.self_consistency_max_iterations)
        )
        sc_iter_entry = ttk.Entry(sc_frame, textvariable=sc_iter_var, width=15)
        sc_iter_entry.pack(anchor="w")
        all_widgets.append(sc_iter_entry)

        ttk.Label(
            sc_frame, text="Verbosity (0=silent, 1=summary, 2=failures, 3=full):"
        ).pack(anchor="w", pady=(5, 0))
        ttk.Label(
            sc_frame,
            text="  Note: Full logging inherits this; truncated/none suppress it during sweep/optim",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        ).pack(anchor="w")
        sc_verbosity, adaptive_debug = _stability_dialog_logging_defaults(self.config)
        sc_verb_var = tk.StringVar(value=sc_verbosity)
        sc_verb_entry = ttk.Entry(sc_frame, textvariable=sc_verb_var, width=15)
        sc_verb_entry.pack(anchor="w")
        all_widgets.append(sc_verb_entry)

        at_frame = ttk.LabelFrame(scrollable, text="Adaptive Timestep", padding=10)
        at_frame.pack(fill="x", pady=5, padx=5)

        at_enabled_var = tk.BooleanVar(value=self.config.adaptive_timestep_enabled)
        at_enabled_cb = ttk.Checkbutton(
            at_frame, text="Enabled", variable=at_enabled_var
        )
        at_enabled_cb.pack(anchor="w")
        all_widgets.append(at_enabled_cb)

        ttk.Label(at_frame, text="Energy jump threshold:").pack(anchor="w", pady=(5, 0))
        at_thresh_var = tk.StringVar(value=str(self.config.adaptive_timestep_threshold))
        at_thresh_entry = ttk.Entry(at_frame, textvariable=at_thresh_var, width=15)
        at_thresh_entry.pack(anchor="w")
        all_widgets.append(at_thresh_entry)

        ttk.Label(at_frame, text="Reduction factor:").pack(anchor="w", pady=(5, 0))
        at_factor_var = tk.StringVar(
            value=str(self.config.adaptive_timestep_reduction_factor)
        )
        at_factor_entry = ttk.Entry(at_frame, textvariable=at_factor_var, width=15)
        at_factor_entry.pack(anchor="w")
        all_widgets.append(at_factor_entry)

        try:
            reduction_factor = self.config.adaptive_timestep_reduction_factor
            min_factor = self.config.adaptive_timestep_min_factor
            if reduction_factor > 1 and min_factor > 0:
                calculated_attempts = math.ceil(
                    math.log(1.0 / min_factor) / math.log(reduction_factor)
                )
                attempts_display = (
                    f"{max(1, calculated_attempts)} "
                    "(auto-calculated from reduction factor & min timestep)"
                )
            else:
                attempts_display = "N/A"
        except (ValueError, ZeroDivisionError):
            attempts_display = "N/A"

        ttk.Label(at_frame, text="Max reduction attempts:").pack(
            anchor="w", pady=(5, 0)
        )
        at_attempts_display = ttk.Label(
            at_frame,
            text=attempts_display,
            relief="sunken",
            background="#f0f0f0",
            foreground="#606060",
            padding=(5, 2),
            font=("TkDefaultFont", 9, "italic"),
        )
        at_attempts_display.pack(anchor="w")
        all_widgets.append(at_attempts_display)

        at_halt_var = tk.BooleanVar(value=self.config.energy_monitor_halt_on_jump)
        at_halt_cb = ttk.Checkbutton(
            at_frame, text="Halt simulation on energy jump", variable=at_halt_var
        )
        at_halt_cb.pack(anchor="w", pady=(5, 0))
        all_widgets.append(at_halt_cb)

        at_debug_var = tk.BooleanVar(value=adaptive_debug)
        at_debug_cb = ttk.Checkbutton(
            at_frame,
            text="Debug logging (single run only; sweep/optim uses Log verbosity)",
            variable=at_debug_var,
        )
        at_debug_cb.pack(anchor="w", pady=(5, 0))
        all_widgets.append(at_debug_cb)

        def apply_sweep_defaults():
            sc_verb_var.set("1")
            at_debug_var.set(True)
            at_halt_var.set(False)

        def on_checkbox_toggle():
            if use_single_run_var.get():
                for widget in all_widgets:
                    widget.configure(state="disabled")
            else:
                for widget in all_widgets:
                    widget.configure(state="normal")
                apply_sweep_defaults()

        use_single_run_cb.configure(command=on_checkbox_toggle)
        on_checkbox_toggle()

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        sweep_frame = ttk.LabelFrame(main_frame, text="Sweep Robustness", padding=10)
        sweep_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(sweep_frame, text="Per-run timeout (seconds, 0=unlimited):").pack(
            anchor="w"
        )
        timeout_var = tk.StringVar(value=str(self.config.per_run_timeout))
        timeout_entry = ttk.Entry(sweep_frame, textvariable=timeout_var, width=15)
        timeout_entry.pack(anchor="w", pady=(0, 5))

        skip_failed_var = tk.BooleanVar(value=self.config.skip_failed_runs)
        skip_failed_cb = ttk.Checkbutton(
            sweep_frame,
            text="Skip failed runs and continue sweep",
            variable=skip_failed_var,
        )
        skip_failed_cb.pack(anchor="w")

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(10, 0))

        def on_confirm():
            try:
                self.config.self_consistency_enabled = sc_enabled_var.get()
                sc_tolerance = float(sc_tol_var.get())
                self.config.self_consistency_tolerance = sc_tolerance
                self.config.self_consistency_target_ms_tolerance = sc_tolerance
                self.config.self_consistency_max_iterations = int(sc_iter_var.get())
                self.config.self_consistency_verbosity = int(sc_verb_var.get())

                self.config.energy_monitor_enabled = False
                self.config.energy_monitor_halt_on_jump = at_halt_var.get()

                self.config.adaptive_timestep_enabled = at_enabled_var.get()
                self.config.adaptive_timestep_threshold = float(at_thresh_var.get())
                self.config.adaptive_timestep_reduction_factor = int(
                    at_factor_var.get()
                )
                self.config.adaptive_timestep_debug = at_debug_var.get()

                self.config.per_run_timeout = float(timeout_var.get())
                self.config.skip_failed_runs = skip_failed_var.get()

                self.per_run_timeout_var.set(str(self.config.per_run_timeout))
                self.skip_failed_runs_var.set(self.config.skip_failed_runs)

                result[0] = True
                dialog.destroy()
            except ValueError as e:
                _show_error_dialog(
                    dialog, "Invalid Input", f"Please check your inputs: {e}"
                )

        def on_cancel():
            result[0] = False
            dialog.destroy()

        confirm_btn = ttk.Button(
            button_frame, text="Proceed with Sweep", command=on_confirm, width=20
        )
        confirm_btn.pack(side="left", padx=5)

        cancel_btn = ttk.Button(
            button_frame, text="Cancel", command=on_cancel, width=15
        )
        cancel_btn.pack(side="left", padx=5)

        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"+{x}+{y}")

        dialog.wait_window()

        if result[0]:
            self._log_result("[INFO] Stability options confirmed for sweep:")
            self._log_result(
                f"  Self-consistency: {self.config.self_consistency_enabled} (tol={self.config.self_consistency_tolerance:.1e}, max_iter={self.config.self_consistency_max_iterations}, verbosity={self.config.self_consistency_verbosity})"
            )
            self._log_result(
                f"  Adaptive timestep: {self.config.adaptive_timestep_enabled} (threshold={self.config.adaptive_timestep_threshold * 100:.0f}%, reduction={self.config.adaptive_timestep_reduction_factor}x, min_factor={self.config.adaptive_timestep_min_factor}, debug={self.config.adaptive_timestep_debug})"
            )
            self._log_result(
                f"  Per-run timeout: {self.config.per_run_timeout}s, Skip failed: {self.config.skip_failed_runs}"
            )
            if not use_single_run_var.get():
                self._log_result(
                    "  [NOTE] Using safer sweep defaults (single-run settings overridden)"
                )
            self._log_result("")

        return result[0]

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
