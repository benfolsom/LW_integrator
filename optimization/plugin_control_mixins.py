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
from optimization.simulation_type_helpers import is_bunch_to_bunch
from optimization.sweep_helpers import AMU_TO_MEV
from optimization.ui_helpers import (
    show_error_dialog as _show_error_dialog,
)


class OptimizationPluginControlMixin:
    """Validate inputs and prepare or control optimization runs."""

    def _validate_inputs(self) -> Optional[str]:
        """Validate user inputs. Returns error message or None."""
        try:
            sim_type = self.sim_type_var.get()
            is_bunch_to_bunch = sim_type == "BUNCH_TO_BUNCH"

            if not is_bunch_to_bunch:
                aperture_min = float(self.aperture_min_var.get())
                aperture_max = float(self.aperture_max_var.get())
                if aperture_min >= aperture_max:
                    return "Aperture min must be less than max"
                if aperture_min <= 0:
                    return "Aperture min must be positive"

            energy_min = float(self.energy_min_var.get())
            energy_max = float(self.energy_max_var.get())
            energy_points = int(self.energy_points_var.get())

            if is_bunch_to_bunch and energy_points == 1:
                if energy_min <= 0:
                    return "Rider energy must be positive"
            else:
                if energy_min >= energy_max:
                    return "Energy min must be less than max"
                if energy_min <= 0:
                    return "Energy min must be positive"

            mode = self.mode_var.get()

            if mode == "blind_sweep":
                has_swept_sub_param = any(
                    controls["sweep_var"].get()
                    for controls in self.sweep_params.values()
                )
                if not is_bunch_to_bunch:
                    aperture_points = int(self.aperture_points_var.get())
                    if aperture_points < 2:
                        return "Sweep mode: Aperture must have at least 2 points"
                if energy_points < 2 and not has_swept_sub_param:
                    return (
                        "Sweep mode: Energy must have at least 2 points "
                        "(or enable a swept sub-parameter)"
                    )
            else:
                if not is_bunch_to_bunch:
                    aperture_points = int(self.aperture_points_var.get())
                    if aperture_points < 1:
                        return "Aperture must have at least 1 point"
                if energy_points < 1:
                    return "Energy must have at least 1 point"

            parse_float_list(self.offset_fractions_var.get())
            float(self.start_z_var.get())
            float(self.wall_z_var.get())
            steps = int(self.steps_var.get())
            if steps < 100:
                return "Steps must be at least 100"

            distance_past_wall = float(self.auto_steps_distance_var.get())
            if distance_past_wall < 0:
                return "Distance past wall must be non-negative"

            for param_name, controls in self.sweep_params.items():
                if controls["sweep_var"].get():
                    min_val = float(controls["min_var"].get())
                    max_val = float(controls["max_var"].get())
                    points = int(controls["points_var"].get())

                    if min_val >= max_val:
                        return f"{param_name}: min must be less than max"
                    if points < 2:
                        return f"{param_name}: must have at least 2 points"
                else:
                    fixed_val = float(controls["fixed_var"].get())
                    if "m_particle" in param_name and fixed_val <= 0:
                        return f"{param_name}: Particle mass must be positive"
                    if "pcount" in param_name and int(fixed_val) < 1:
                        return f"{param_name}: Particle count must be at least 1"

            float(self.sweep_params["rider_stripped_ions"]["fixed_var"].get())
            if self.sim_type_var.get() == "BUNCH_TO_BUNCH":
                float(self.sweep_params["driver_stripped_ions"]["fixed_var"].get())

            return None
        except ValueError as e:
            return f"Invalid input: {e}"

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

    def _gather_config(self) -> OptimizationConfig:
        """Gather configuration from UI fields."""
        existing_config = getattr(self, "config", None)

        has_gui = self.gui_controller is not None
        print(f"[DEBUG] _gather_config: Main GUI available: {has_gui}")
        if existing_config:
            print(
                "[DEBUG] _gather_config: Existing config available (will be used as fallback)"
            )
        else:
            print(
                "[DEBUG] _gather_config: No existing config, using defaults as fallback"
            )

        if has_gui:
            print(
                "[DEBUG] _gather_config: Reading stability settings from main GUI Stability tab"
            )
        else:
            print(
                "[DEBUG] _gather_config: No GUI available, using existing config or defaults"
            )

        rider_offset = parse_offset_pair(self.offset_fractions_var.get())
        driver_offset = parse_offset_pair(self.driver_offset_var.get())

        config_obj = OptimizationConfig(
            simulation_type=SimulationType[self.sim_type_var.get()],
            mode=self.mode_var.get(),
            optimization_method=self.optimization_method_var.get(),
            optimization_maxiter=int(self.optimization_maxiter_var.get()),
            optimization_population_size=int(self.optimization_popsize_var.get()),
            optimization_mutation_rate=float(self.optimization_mutation_var.get()),
            optimization_crossover_rate=float(self.optimization_crossover_var.get()),
            optimization_n_starts=int(self.optimization_nstarts_var.get()),
            optimization_save_top_n=int(self.optimization_save_top_n_var.get()),
            optimization_convergence_tol=float(
                self.optimization_convergence_tol_var.get()
            ),
            optimization_convergence_patience=int(
                self.optimization_convergence_patience_var.get()
            ),
            aperture_range=(
                float(self.aperture_min_var.get()),
                float(self.aperture_max_var.get()),
            ),
            aperture_points=(
                1
                if is_bunch_to_bunch(self.sim_type_var.get())
                else int(self.aperture_points_var.get())
            ),
            aperture_log_scale=self.aperture_log_var.get(),
            energy_range=(
                float(self.energy_min_var.get()),
                float(self.energy_max_var.get()),
            ),
            energy_points=int(self.energy_points_var.get()),
            energy_log_scale=self.energy_log_var.get(),
            transverse_offset_fractions=parse_float_list(self.offset_fractions_var.get()),
            starting_z_positions=[float(self.start_z_var.get())],
            wall_z=float(self.wall_z_var.get()),
            wall_z_range=(
                (
                    float(self.wall_z_min_var.get()),
                    float(self.wall_z_max_var.get()),
                )
                if self.wall_z_sweep_var.get()
                else None
            ),
            wall_z_points=(
                int(self.wall_z_points_var.get()) if self.wall_z_sweep_var.get() else 1
            ),
            cavity_spacing=float(self.cavity_spacing_var.get()),
            timestep=(
                float(self.duration_var.get())
                if self.timestep_mode_var.get() == "count"
                else 3e-7
            ),
            steps=(
                int(self.steps_var.get())
                if self.timestep_mode_var.get() == "duration"
                else 200
            ),
            auto_steps=True,
            auto_steps_target=(
                int(self.steps_var.get())
                if self.timestep_mode_var.get() == "duration"
                else 200
            ),
            auto_steps_distance_past_wall=float(self.auto_steps_distance_var.get()),
            objective=self.objective_var.get(),
            transv_mom=float(self.sweep_params["rider_transv_mom"]["fixed_var"].get()),
            transv_dist=float(
                self.sweep_params["rider_transv_dist"]["fixed_var"].get()
            ),
            transv_offset_x=rider_offset[0],
            transv_offset_y=rider_offset[1],
            driver_transv_offset_x=driver_offset[0],
            driver_transv_offset_y=driver_offset[1],
            macroparticle_enabled=bool(self.macroparticle_enabled_var.get()),
            macroparticle_charge_multiplier=float(
                self.sweep_params["macroparticle_charge_multiplier"]["fixed_var"].get()
            ),
            macroparticle_sigma_multiplier=float(
                self.sweep_params["macroparticle_sigma_multiplier"]["fixed_var"].get()
            ),
            macroparticle_use_momentum_errors=bool(
                self.macroparticle_momentum_errors_var.get()
            ),
            m_particle=float(self.sweep_params["rider_m_particle"]["fixed_var"].get()),
            pcount=int(self.sweep_params["rider_pcount"]["fixed_var"].get()),
            charge_sign=float(
                self.sweep_params["rider_charge_sign"]["fixed_var"].get()
            ),
            stripped_ions=float(
                self.sweep_params["rider_stripped_ions"]["fixed_var"].get()
            ),
            driver_m_particle=float(
                self.sweep_params["driver_m_particle"]["fixed_var"].get()
            ),
            driver_charge_sign=float(
                self.sweep_params["driver_charge_sign"]["fixed_var"].get()
            ),
            driver_pcount=int(self.sweep_params["driver_pcount"]["fixed_var"].get()),
            driver_transv_mom=float(
                self.sweep_params["driver_transv_mom"]["fixed_var"].get()
            ),
            driver_transv_dist=float(
                self.sweep_params["driver_transv_dist"]["fixed_var"].get()
            ),
            driver_starting_distance=float(
                self.sweep_params["driver_starting_distance"]["fixed_var"].get()
            ),
            driver_stripped_ions=float(
                self.sweep_params["driver_stripped_ions"]["fixed_var"].get()
            ),
            save_top_n_trajectories=bool(self.save_top_n_traj_var.get()),
            save_all_trajectories=bool(self.save_all_traj_var.get()),
            save_failed_trajectories=bool(self.save_failed_traj_var.get()),
            trajectory_stride=int(self.trajectory_stride_var.get()),
            metrics_export_format=str(self.metrics_format_var.get()),
            metrics_export_scope=str(self.metrics_scope_var.get()),
            log_verbosity=str(self.log_verbosity_var.get()),
            smoothness_enabled=self.smoothness_enabled_var.get(),
            smoothness_window_size=int(self.smoothness_window_var.get()),
            smoothness_oscillation_threshold=float(
                self.smoothness_oscillation_var.get()
            ),
            smoothness_reject_on_violation=self.smoothness_reject_var.get(),
            per_run_timeout=float(self.per_run_timeout_var.get()),
            skip_failed_runs=self.skip_failed_runs_var.get(),
            failed_run_retry_attempts=int(self.failed_run_retry_attempts_var.get()),
            image_subcharge_count=self._get_gui_stability_setting(
                "image_subcharge_var",
                existing_config.image_subcharge_count if existing_config else 12,
            ),
            use_image_weighting=self._get_gui_stability_setting(
                "image_weighting_var",
                existing_config.use_image_weighting if existing_config else True,
            ),
            self_consistency_enabled=self._get_gui_stability_setting(
                "self_consistency_enabled_var",
                existing_config.self_consistency_enabled if existing_config else True,
            ),
            self_consistency_tolerance=self._get_gui_stability_setting(
                "self_consistency_target_ms_tolerance_var",
                existing_config.self_consistency_tolerance if existing_config else 1e-4,
            ),
            self_consistency_max_iterations=self._get_gui_stability_setting(
                "self_consistency_max_iterations_var",
                (
                    existing_config.self_consistency_max_iterations
                    if existing_config
                    else 5
                ),
            ),
            self_consistency_verbosity=self._get_gui_stability_setting(
                "self_consistency_verbosity_var",
                existing_config.self_consistency_verbosity if existing_config else 0,
            ),
            self_consistency_chrono_interpolate=self._get_gui_stability_setting(
                "self_consistency_chrono_interpolate_var",
                (
                    existing_config.self_consistency_chrono_interpolate
                    if existing_config
                    else False
                ),
            ),
            self_consistency_chrono_tolerance=self._get_gui_stability_setting(
                "self_consistency_chrono_tolerance_var",
                (
                    existing_config.self_consistency_chrono_tolerance
                    if existing_config
                    else 1e-3
                ),
            ),
            self_consistency_chrono_high_precision=self._get_gui_stability_setting(
                "self_consistency_chrono_high_precision_var",
                (
                    existing_config.self_consistency_chrono_high_precision
                    if existing_config
                    else False
                ),
            ),
            self_consistency_chrono_adaptive_tolerance=self._get_gui_stability_setting(
                "self_consistency_chrono_adaptive_tolerance_var",
                (
                    existing_config.self_consistency_chrono_adaptive_tolerance
                    if existing_config
                    else False
                ),
            ),
            energy_monitor_halt_on_jump=self._get_gui_stability_setting(
                "adaptive_timestep_halt_on_jump_var",
                (
                    existing_config.energy_monitor_halt_on_jump
                    if existing_config
                    else False
                ),
            ),
            adaptive_timestep_enabled=self._get_gui_stability_setting(
                "adaptive_timestep_enabled_var",
                existing_config.adaptive_timestep_enabled if existing_config else True,
            ),
            adaptive_timestep_threshold=self._get_gui_stability_setting(
                "adaptive_timestep_threshold_var",
                (
                    existing_config.adaptive_timestep_threshold
                    if existing_config
                    else 0.10
                ),
            ),
            adaptive_timestep_reduction_factor=self._get_gui_stability_setting(
                "adaptive_timestep_reduction_factor_var",
                (
                    existing_config.adaptive_timestep_reduction_factor
                    if existing_config
                    else 10
                ),
            ),
            adaptive_timestep_min_factor=self._get_gui_stability_setting(
                "adaptive_timestep_min_factor_var",
                (
                    existing_config.adaptive_timestep_min_factor
                    if existing_config
                    else 1e-4
                ),
            ),
            adaptive_timestep_cooldown_steps=self._get_gui_stability_setting(
                "adaptive_timestep_cooldown_steps_var",
                (
                    existing_config.adaptive_timestep_cooldown_steps
                    if existing_config
                    else 10
                ),
            ),
            adaptive_timestep_probe_threshold=self._get_gui_stability_setting(
                "adaptive_timestep_probe_threshold_var",
                (
                    existing_config.adaptive_timestep_probe_threshold
                    if existing_config
                    else 0.01
                ),
            ),
            adaptive_timestep_max_probe_steps=self._get_gui_stability_setting(
                "adaptive_timestep_max_probe_steps_var",
                (
                    existing_config.adaptive_timestep_max_probe_steps
                    if existing_config
                    else 3
                ),
            ),
            adaptive_timestep_debug=self._get_gui_stability_setting(
                "adaptive_timestep_debug_var",
                existing_config.adaptive_timestep_debug if existing_config else False,
            ),
            self_consistency_gamma_reconciliation_method=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_method_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_method
                    if existing_config
                    else "DISABLED"
                ),
            ),
            self_consistency_gamma_reconciliation_low_beta_threshold=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_low_beta_threshold_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_low_beta_threshold
                    if existing_config
                    else 0.9
                ),
            ),
            self_consistency_gamma_reconciliation_high_beta_threshold=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_high_beta_threshold_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_high_beta_threshold
                    if existing_config
                    else 0.99
                ),
            ),
            self_consistency_gamma_reconciliation_low_beta_weight=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_low_beta_weight_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_low_beta_weight
                    if existing_config
                    else 0.8
                ),
            ),
            self_consistency_gamma_reconciliation_high_beta_weight=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_high_beta_weight_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_high_beta_weight
                    if existing_config
                    else 0.2
                ),
            ),
            self_consistency_gamma_reconciliation_mid_beta_weight=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_mid_beta_weight_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_mid_beta_weight
                    if existing_config
                    else 0.5
                ),
            ),
            self_consistency_gamma_reconciliation_fixed_weight=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_fixed_weight_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_fixed_weight
                    if existing_config
                    else 0.5
                ),
            ),
            smoothness_trend_threshold=(
                existing_config.smoothness_trend_threshold if existing_config else 0.30
            ),
            smoothness_max_violations=(
                existing_config.smoothness_max_violations if existing_config else 3
            ),
            timestep_strategy="auto_distance",
            target_distance_mm=(
                existing_config.target_distance_mm if existing_config else 100.0
            ),
            energy_scale_exponent=(
                existing_config.energy_scale_exponent if existing_config else 1.0
            ),
            startup_mode=(
                self.gui_controller.core_param_vars["startup_mode"].get()
                if self.gui_controller
                and hasattr(self.gui_controller, "core_param_vars")
                else (existing_config.startup_mode if existing_config else "COLD_START")
            ),
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
            text="  Note: Sweep/Optim override this via Log verbosity setting",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        ).pack(anchor="w")
        sc_verb_var = tk.StringVar(
            value=str(max(self.config.self_consistency_verbosity, 1))
        )
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

        at_debug_var = tk.BooleanVar(value=self.config.adaptive_timestep_debug or True)
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
                self.config.self_consistency_tolerance = float(sc_tol_var.get())
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
        warnings = []

        aperture_min = self.config.aperture_range[0]
        energy_max = self.config.energy_range[1]

        m_electron = 0.00054857990907
        rest_energy_mev = self.config.m_particle * AMU_TO_MEV

        if is_bunch_to_bunch(self.config.simulation_type):
            gamma_max = (energy_max * 1e3) / rest_energy_mev + 1.0
        else:
            gamma_max = (energy_max * 1e3) / rest_energy_mev

        m_proton = 1.007276466621

        if abs(self.config.m_particle - m_electron) < 1e-6:
            extreme_gamma_threshold = 1_956_000
            particle_type = "electron"
            extreme_energy_tev = 1.0
        elif abs(self.config.m_particle - m_proton) < 1e-3:
            extreme_gamma_threshold = 21_300
            particle_type = "proton"
            extreme_energy_tev = 20.0
        else:
            extreme_gamma_threshold = int(21_300 * m_proton / self.config.m_particle)
            particle_type = "particle"
            extreme_energy_tev = extreme_gamma_threshold * rest_energy_mev / 1e6

        if aperture_min < 1e-5 and gamma_max > 10000:
            warnings.append(
                f"• Very small aperture ({aperture_min:.2e} mm) with high energy ({energy_max:.1f} GeV, γ≈{gamma_max:.0f})\n"
                f"  This may cause extreme fields, SC convergence issues, and very slow runs."
            )

        if aperture_min < 1e-6:
            warnings.append(
                f"• Aperture < 1 μm detected ({aperture_min:.2e} mm)\n"
                f"  Sub-micron apertures often cause numerical instabilities."
            )

        if gamma_max > extreme_gamma_threshold:
            warnings.append(
                f"• Very high energy detected ({energy_max:.1f} GeV, γ≈{gamma_max:.0f})\n"
                f"  Exceeds recommended threshold for {particle_type}s (~{extreme_energy_tev:.1f} TeV)\n"
                f"  Ultra-relativistic particles may require very fine timesteps."
            )

        if not self.config.auto_steps:
            timestep = self.config.timestep
            beta_approx = 1.0 if gamma_max > 2 else 0.9
            distance_per_step = beta_approx * gamma_max * 300.0 * timestep

            if distance_per_step > aperture_min * 0.1:
                warnings.append(
                    f"• Fixed timestep may be too large for small apertures\n"
                    f"  Distance/step ≈ {distance_per_step:.3f} mm vs aperture {aperture_min:.2e} mm\n"
                    f"  Consider enabling 'Auto timestep' or reducing timestep."
                )

        if warnings:
            warning_text = "Extreme parameter combinations detected:\n\n" + "\n\n".join(
                warnings
            )
            warning_text += "\n\nRecommendations:\n"
            warning_text += "• Enable 'Per-run timeout' to prevent hangs\n"
            warning_text += "• Enable 'Skip failed runs' to complete the sweep\n"
            warning_text += (
                "• Consider more moderate parameter ranges for initial sweeps\n"
            )
            warning_text += "\nDo you want to proceed anyway?"
            return warning_text

        return None

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

    def _compute_soft_penalty(
        self,
        *,
        aperture_radius: float,
        macroparticle_charge_multiplier: float,
        initial_energy_gev: float,
    ) -> float:
        """Estimate a soft penalty for risky parameter combinations."""
        penalty = 0.0

        aperture_threshold_mm = 0.01
        charge_threshold = 800.0
        energy_threshold = 120.0
        penalty_scale = 1.0e-3

        small_aperture_factor = max(
            0.0, (aperture_threshold_mm - aperture_radius) / aperture_threshold_mm
        )
        high_charge_factor = max(
            0.0,
            (macroparticle_charge_multiplier - charge_threshold) / charge_threshold,
        )

        if small_aperture_factor > 0 and high_charge_factor > 0:
            penalty += small_aperture_factor * high_charge_factor

        if high_charge_factor > 0 and initial_energy_gev > energy_threshold:
            energy_factor = (initial_energy_gev - energy_threshold) / energy_threshold
            tight_aperture_factor = max(0.0, (0.1 - aperture_radius) / 0.1)
            penalty += 0.5 * energy_factor * high_charge_factor * tight_aperture_factor

        return max(0.0, penalty * penalty_scale)

    def _set_fixed_sweep_value(self, param_name: str, value: str):
        """Update a fixed-value sweep control."""
        self.sweep_params[param_name]["fixed_var"].set(value)
