"""GUI plugin for optimization and parameter sweeps.

This module provides a Tkinter panel for running parameter sweeps and
optimization studies on the LW integrator. It integrates with the main
testbed GUI and provides controls for:
- Parameter range specification (aperture, energy, transverse offset, etc.)
- Optimization objective selection (max energy gain, etc.)
- Progress monitoring
- Results visualization (heatmaps, summary plots)
"""

from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from core.types import SimulationType  # type: ignore[import]
from optimization.config import OptimizationConfig
from optimization.plugin_config_helpers import (
    apply_sweep_parameter_overrides,
    parse_float_list,
    parse_offset_pair,
)
from optimization.plugin_config_mixins import OptimizationPluginConfigMixin
from optimization.plugin_form_mixins import OptimizationPluginFormMixin
from optimization.plugin_parameter_mixins import OptimizationPluginParameterMixin
from optimization.plugin_results_helpers import (
    build_summary_heatmap_grid,
    build_trajectory_plot_data,
    collect_summary_plot_data,
    parse_results_payload,
    summarize_result_row,
    UNKNOWN_RESULTS_FORMAT_MESSAGE,
)
from optimization.plugin_ui_mixins import OptimizationPluginUIMixin
from optimization.results_mixins import OptimizationResultsMixin
from optimization.run_mixins import OptimizationRunMixin
from optimization.sweep_helpers import AMU_TO_MEV
from optimization.ui_helpers import (
    show_error_dialog as _show_error_dialog,
)
from optimization.ui_helpers import (
    show_info_dialog as _show_info_dialog,
)


class OptimizationPlugin(
    OptimizationPluginConfigMixin,
    OptimizationPluginParameterMixin,
    OptimizationPluginFormMixin,
    OptimizationPluginUIMixin,
    OptimizationRunMixin,
    OptimizationResultsMixin,
    ttk.Frame,
):
    """Optimization plugin for LW Integrator GUI."""

    import time

    def __init__(
        self,
        parent: tk.Widget,
        gui_controller=None,
        sweep_config_dir=None,
        sweep_output_dir=None,
        **kwargs,
    ):
        """Initialize the optimization plugin.

        Parameters
        ----------
        parent : tk.Widget
            Parent widget (typically a notebook tab or frame)
        gui_controller : Optional
            Reference to main GUI controller for run state integration
        sweep_config_dir : str, optional
            Directory for sweep configuration files
        sweep_output_dir : str, optional
            Directory for sweep output/results files
        """
        super().__init__(parent, **kwargs)
        self.gui_controller = gui_controller
        self.config = OptimizationConfig()
        self.running = False
        self.progress_value = 0.0
        self.progress_text = ""
        self._was_cancelled = False

        # Store sweep directories
        self.sweep_config_dir = sweep_config_dir or "configs/sweep_configs"
        self.sweep_output_dir = sweep_output_dir or "results/sweeps"

        # Log file tracking
        self._log_file = None
        self._log_file_path = None

        # Clean up any orphaned temp directories from previous runs
        self._cleanup_orphaned_temp_dirs()

        self._build_ui()

        # Sync simulation type with main GUI if available
        if self.gui_controller and hasattr(self.gui_controller, "sim_type_var"):
            main_sim_type = self.gui_controller.sim_type_var.get()
            self.sim_type_var.set(main_sim_type)
            # Update driver visibility based on synced simulation type
            self._update_driver_visibility()


    def _log_truncated_run(
        self, run_num: int, params: dict, metrics: dict = None, error: str = None
    ):
        """Log a single run in truncated format (1-2 lines).

        Parameters
        ----------
        run_num : int
            Run number
        params : dict
            Parameter values for this run
        metrics : dict, optional
            Result metrics (energy gain, emittance, etc.)
        error : str, optional
            Error message if run failed
        """
        # Format parameters compactly
        param_parts = []
        for key, value in params.items():
            if isinstance(value, float):
                if abs(value) < 0.001 or abs(value) > 1000:
                    param_parts.append(f"{key}={value:.3e}")
                else:
                    param_parts.append(f"{key}={value:.3g}")
            else:
                param_parts.append(f"{key}={value}")
        param_str = " ".join(param_parts)

        if error:
            # Failed run
            status = f"FAILED: {error}"
            self._log_result(f"Run #{run_num:4d} | {param_str} | {status}")
        elif metrics:
            # Successful run - format metrics compactly
            metric_parts = []
            for key, value in metrics.items():
                if isinstance(value, float):
                    if abs(value) < 0.001 or abs(value) > 1000:
                        metric_parts.append(f"{key}={value:.3e}")
                    else:
                        metric_parts.append(f"{key}={value:.3g}")
                else:
                    metric_parts.append(f"{key}={value}")
            metric_str = " ".join(metric_parts)
            self._log_result(
                f"Run #{run_num:4d} | {param_str} | {metric_str} | SUCCESS"
            )
        else:
            self._log_result(f"Run #{run_num:4d} | {param_str} | RUNNING")

    def _should_save_trajectory(self, run_result: dict, rank: int = None) -> bool:
        """Determine if trajectory should be saved based on config.

        Parameters
        ----------
        run_result : dict
            Result dictionary from integration run
        rank : int, optional
            Rank of this result (1=best, 2=second best, etc.)

        Returns
        -------
        bool
            True if trajectory should be saved
        """
        if self.config.save_all_trajectories:
            return True

        if self.config.save_failed_trajectories:
            # Save failed runs AND halted runs
            return run_result.get("failed", False) or run_result.get(
                "halted_early", False
            )

        if self.config.save_top_n_trajectories and rank is not None:
            return rank <= self.config.optimization_save_top_n

        return False

    def _validate_inputs(self) -> Optional[str]:
        """Validate user inputs. Returns error message or None."""
        try:
            sim_type = self.sim_type_var.get()
            is_bunch_to_bunch = sim_type == "BUNCH_TO_BUNCH"

            # Aperture range - only validate for CONDUCTING_WALL modes
            if not is_bunch_to_bunch:
                aperture_min = float(self.aperture_min_var.get())
                aperture_max = float(self.aperture_max_var.get())
                if aperture_min >= aperture_max:
                    return "Aperture min must be less than max"
                if aperture_min <= 0:
                    return "Aperture min must be positive"

            # Energy range (rider kinetic energy)
            energy_min = float(self.energy_min_var.get())
            energy_max = float(self.energy_max_var.get())
            energy_points = int(self.energy_points_var.get())

            # For BUNCH_TO_BUNCH the rider energy can be fixed (1 point,
            # min==max) when the sweep is purely over driver parameters.
            if is_bunch_to_bunch and energy_points == 1:
                # Single-point rider energy: just needs to be positive
                if energy_min <= 0:
                    return "Rider energy must be positive"
            else:
                if energy_min >= energy_max:
                    return "Energy min must be less than max"
                if energy_min <= 0:
                    return "Energy min must be positive"

            mode = self.mode_var.get()

            if mode == "blind_sweep":
                # Sweep mode requires at least 2 points for main parameters
                # UNLESS there is at least one swept driver/rider sub-parameter
                has_swept_sub_param = any(
                    controls["sweep_var"].get()
                    for controls in self.sweep_params.values()
                )
                if not is_bunch_to_bunch:
                    aperture_points = int(self.aperture_points_var.get())
                    if aperture_points < 2:
                        return "Sweep mode: Aperture must have at least 2 points"
                if energy_points < 2 and not has_swept_sub_param:
                    return "Sweep mode: Energy must have at least 2 points (or enable a swept sub-parameter)"
            else:
                # Optimization mode allows 1 point (fixed) for any parameter
                if not is_bunch_to_bunch:
                    aperture_points = int(self.aperture_points_var.get())
                    if aperture_points < 1:
                        return "Aperture must have at least 1 point"
                if energy_points < 1:
                    return "Energy must have at least 1 point"

            # Lists
            parse_float_list(self.offset_fractions_var.get())
            # Single float for rider starting z
            float(self.start_z_var.get())

            # Wall and steps
            float(self.wall_z_var.get())
            steps = int(self.steps_var.get())
            if steps < 100:
                return "Steps must be at least 100"

            # Validate distance past wall (always used in auto-calculation)
            distance_past_wall = float(self.auto_steps_distance_var.get())
            if distance_past_wall < 0:
                return "Distance past wall must be non-negative"

            # Validate sweepable parameters
            for param_name, controls in self.sweep_params.items():
                if controls["sweep_var"].get():
                    # Validate range for swept parameters
                    min_val = float(controls["min_var"].get())
                    max_val = float(controls["max_var"].get())
                    points = int(controls["points_var"].get())

                    if min_val >= max_val:
                        return f"{param_name}: min must be less than max"
                    if points < 2:
                        return f"{param_name}: must have at least 2 points"
                else:
                    # Validate fixed value
                    fixed_val = float(controls["fixed_var"].get())
                    if "m_particle" in param_name and fixed_val <= 0:
                        return f"{param_name}: Particle mass must be positive"
                    if "pcount" in param_name and int(fixed_val) < 1:
                        return f"{param_name}: Particle count must be at least 1"

            # Stripped ions (always fixed)
            float(self.sweep_params["rider_stripped_ions"]["fixed_var"].get())
            if self.sim_type_var.get() == "BUNCH_TO_BUNCH":
                float(self.sweep_params["driver_stripped_ions"]["fixed_var"].get())

            return None
        except ValueError as e:
            return f"Invalid input: {e}"

    def _get_gui_stability_setting(self, var_name: str, default_value):
        """Get stability setting from main GUI if available, otherwise use default.

        Parameters
        ----------
        var_name : str
            Name of the GUI variable to read (e.g., 'self_consistency_enabled_var')
        default_value : any
            Default value if GUI is not available

        Returns
        -------
        any
            Value from GUI or default
        """
        if self.gui_controller and hasattr(self.gui_controller, var_name):
            var = getattr(self.gui_controller, var_name)
            value = var.get()
            # Convert string to appropriate types
            if isinstance(value, str):
                # Tolerance and numeric values
                if (
                    "tolerance" in var_name
                    or "threshold" in var_name
                    or "factor" in var_name
                ):
                    try:
                        return float(value)
                    except ValueError:
                        return default_value
                # Integer values
                elif (
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
        # Stability settings are read from main GUI if available, otherwise from existing config
        existing_config = getattr(self, "config", None)

        # Debug logging
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
            # Force aperture_points=1 for BUNCH_TO_BUNCH (aperture not applicable)
            aperture_points=(
                1
                if SimulationType[self.sim_type_var.get()]
                == SimulationType.BUNCH_TO_BUNCH
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
            auto_steps=True,  # Always use auto-calculation
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
            # Trajectory saving options
            save_top_n_trajectories=bool(self.save_top_n_traj_var.get()),
            save_all_trajectories=bool(self.save_all_traj_var.get()),
            save_failed_trajectories=bool(self.save_failed_traj_var.get()),
            trajectory_stride=int(self.trajectory_stride_var.get()),
            # Metrics export options
            metrics_export_format=str(self.metrics_format_var.get()),
            metrics_export_scope=str(self.metrics_scope_var.get()),
            # Log verbosity
            log_verbosity=str(self.log_verbosity_var.get()),
            # Stability checking options
            smoothness_enabled=self.smoothness_enabled_var.get(),
            smoothness_window_size=int(self.smoothness_window_var.get()),
            smoothness_oscillation_threshold=float(
                self.smoothness_oscillation_var.get()
            ),
            smoothness_reject_on_violation=self.smoothness_reject_var.get(),
            # Sweep robustness options
            per_run_timeout=float(self.per_run_timeout_var.get()),
            skip_failed_runs=self.skip_failed_runs_var.get(),
            failed_run_retry_attempts=int(self.failed_run_retry_attempts_var.get()),
            # Conducting wall image parameters - read from main GUI
            image_subcharge_count=self._get_gui_stability_setting(
                "image_subcharge_var",
                existing_config.image_subcharge_count if existing_config else 12,
            ),
            use_image_weighting=self._get_gui_stability_setting(
                "image_weighting_var",
                existing_config.use_image_weighting if existing_config else True,
            ),
            # Stability options - read from main GUI if available, otherwise use existing config or defaults
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
            # Gamma reconciliation parameters
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
            # Timestep strategy - use auto_distance for sweeps/optimizations
            # This ensures all runs travel to wall_z + target_distance regardless of energy
            timestep_strategy="auto_distance",
            target_distance_mm=(
                existing_config.target_distance_mm if existing_config else 100.0
            ),
            energy_scale_exponent=(
                existing_config.energy_scale_exponent if existing_config else 1.0
            ),
            # Startup mode - read from main GUI core params
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
        """Show stability options confirmation dialog with ability to adjust settings.

        Returns
        -------
        bool
            True if user confirms to proceed, False to cancel
        """
        dialog = tk.Toplevel(self)
        dialog.title("Confirm Stability Options")
        dialog.transient(self)
        dialog.grab_set()

        # Result container
        result = [False]

        # Main frame
        main_frame = ttk.Frame(dialog, padding=15)
        main_frame.pack(fill="both", expand=True)

        # Info label
        info_label = ttk.Label(
            main_frame,
            text="The following stability options will be used for all sweep runs.\n"
            "These settings affect convergence, energy monitoring, and timestep adaptation.",
            wraplength=500,
            justify="left",
        )
        info_label.pack(pady=(0, 10))

        # Checkbox for using single-run settings vs safer sweep defaults
        use_single_run_var = tk.BooleanVar(value=True)
        use_single_run_frame = ttk.Frame(main_frame)
        use_single_run_frame.pack(fill="x", pady=(0, 10))

        use_single_run_cb = ttk.Checkbutton(
            use_single_run_frame,
            text="Use single-run stability settings (uncheck for safer sweep defaults)",
            variable=use_single_run_var,
        )
        use_single_run_cb.pack(anchor="w")

        # Scrollable frame for options
        canvas = tk.Canvas(main_frame, height=300, width=550)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)

        scrollable.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Store widget variables for enabling/disabling
        all_widgets = []

        # Self-consistency section
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

        # Adaptive timestep section (Energy Monitoring functionality integrated here)
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

        # Max refinement attempts is now auto-calculated (read-only display)
        import math

        try:
            reduction_factor = self.config.adaptive_timestep_reduction_factor
            min_factor = self.config.adaptive_timestep_min_factor
            if reduction_factor > 1 and min_factor > 0:
                calculated_attempts = math.ceil(
                    math.log(1.0 / min_factor) / math.log(reduction_factor)
                )
                attempts_display = f"{max(1, calculated_attempts)} (auto-calculated from reduction factor & min timestep)"
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

        # Function to apply safer sweep defaults
        def apply_sweep_defaults():
            """Apply safer defaults for sweeps."""
            # Self-consistency: more verbose for debugging
            sc_verb_var.set("1")
            # Adaptive timestep: debug enabled, don't halt
            # Note: max_attempts is now auto-calculated from reduction_factor and min_timestep_factor
            at_debug_var.set(True)
            at_halt_var.set(False)

        # Function to toggle widgets based on checkbox
        def on_checkbox_toggle():
            if use_single_run_var.get():
                # Checkbox is checked: use single-run settings, disable widgets (greyed out)
                for widget in all_widgets:
                    widget.configure(state="disabled")
            else:
                # Checkbox is unchecked: enable widgets and apply safer sweep defaults
                for widget in all_widgets:
                    widget.configure(state="normal")
                apply_sweep_defaults()

        # Bind checkbox to toggle function
        use_single_run_cb.configure(command=on_checkbox_toggle)

        # Initial state: checkbox is checked by default, so fields should be disabled
        on_checkbox_toggle()

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Sweep robustness section
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

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(10, 0))

        def on_confirm():
            """Validate and save settings."""
            try:
                # Check if using single-run settings or sweep defaults
                if not use_single_run_var.get():
                    # Apply safer sweep defaults (already set in UI via apply_sweep_defaults)
                    # Just read the values from the (disabled) widgets
                    pass

                # Update config with dialog values
                self.config.self_consistency_enabled = sc_enabled_var.get()
                self.config.self_consistency_tolerance = float(sc_tol_var.get())
                self.config.self_consistency_max_iterations = int(sc_iter_var.get())
                self.config.self_consistency_verbosity = int(sc_verb_var.get())

                # Energy monitoring removed - halt option now in adaptive timestep
                self.config.energy_monitor_enabled = False
                self.config.energy_monitor_halt_on_jump = at_halt_var.get()

                self.config.adaptive_timestep_enabled = at_enabled_var.get()
                self.config.adaptive_timestep_threshold = float(at_thresh_var.get())
                self.config.adaptive_timestep_reduction_factor = int(
                    at_factor_var.get()
                )
                # max_refinement_attempts is now auto-calculated from reduction_factor and min_timestep_factor
                self.config.adaptive_timestep_debug = at_debug_var.get()

                # Sweep robustness options
                self.config.per_run_timeout = float(timeout_var.get())
                self.config.skip_failed_runs = skip_failed_var.get()

                # Update UI variables so changes persist when config is saved
                self.per_run_timeout_var.set(str(self.config.per_run_timeout))
                self.skip_failed_runs_var.set(self.config.skip_failed_runs)

                result[0] = True
                dialog.destroy()
            except ValueError as e:
                _show_error_dialog(
                    dialog, "Invalid Input", f"Please check your inputs: {e}"
                )

        def on_cancel():
            """Cancel and close."""
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

        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"+{x}+{y}")

        # Wait for dialog
        dialog.wait_window()

        # Log confirmed settings
        if result[0]:
            self._log_result("[INFO] Stability options confirmed for sweep:")
            self._log_result(
                f"  Self-consistency: {self.config.self_consistency_enabled} (tol={self.config.self_consistency_tolerance:.1e}, max_iter={self.config.self_consistency_max_iterations}, verbosity={self.config.self_consistency_verbosity})"
            )
            # Energy monitoring removed - halt option integrated into adaptive timestep
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
        """Check for extreme parameter combinations that might cause issues.

        Returns
        -------
        Optional[str]
            Warning message if extreme parameters detected, None otherwise
        """
        warnings = []

        # Check for very small apertures with high energies
        aperture_min = self.config.aperture_range[0]
        energy_max = self.config.energy_range[1]

        # Electron mass in amu
        m_electron = 0.00054857990907

        # Calculate gamma for max energy
        AMU_TO_MEV = 931.494
        rest_energy_mev = self.config.m_particle * AMU_TO_MEV

        # For BUNCH_TO_BUNCH, energy is kinetic; for others, it's total
        if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
            gamma_max = (energy_max * 1e3) / rest_energy_mev + 1.0
        else:
            gamma_max = (energy_max * 1e3) / rest_energy_mev

        # Determine extreme energy threshold based on particle type
        # Electron mass in AMU
        m_electron = 0.00054857990907
        # Proton mass in AMU
        m_proton = 1.007276466621

        # Set gamma threshold: ~1 TeV for electrons, ~20 TeV for protons
        if abs(self.config.m_particle - m_electron) < 1e-6:
            # Electron: 1 TeV / 0.511 MeV ≈ 1,956,947
            extreme_gamma_threshold = 1_956_000
            particle_type = "electron"
            extreme_energy_tev = 1.0
        elif abs(self.config.m_particle - m_proton) < 1e-3:
            # Proton: 20 TeV / 938.27 MeV ≈ 21,321
            extreme_gamma_threshold = 21_300
            particle_type = "proton"
            extreme_energy_tev = 20.0
        else:
            # Generic particle: scale based on rest mass relative to proton
            extreme_gamma_threshold = int(21_300 * m_proton / self.config.m_particle)
            particle_type = "particle"
            extreme_energy_tev = extreme_gamma_threshold * rest_energy_mev / 1e6

        # Warn if aperture < 10 μm and gamma > 10,000
        if aperture_min < 1e-5 and gamma_max > 10000:
            warnings.append(
                f"• Very small aperture ({aperture_min:.2e} mm) with high energy ({energy_max:.1f} GeV, γ≈{gamma_max:.0f})\n"
                f"  This may cause extreme fields, SC convergence issues, and very slow runs."
            )

        # Warn if aperture < 1 μm
        if aperture_min < 1e-6:
            warnings.append(
                f"• Aperture < 1 μm detected ({aperture_min:.2e} mm)\n"
                f"  Sub-micron apertures often cause numerical instabilities."
            )

        # Warn if gamma exceeds threshold (~1 TeV for electrons, ~20 TeV for protons)
        if gamma_max > extreme_gamma_threshold:
            warnings.append(
                f"• Very high energy detected ({energy_max:.1f} GeV, γ≈{gamma_max:.0f})\n"
                f"  Exceeds recommended threshold for {particle_type}s (~{extreme_energy_tev:.1f} TeV)\n"
                f"  Ultra-relativistic particles may require very fine timesteps."
            )

        # Check timestep if not auto
        if not self.config.auto_steps:
            timestep = self.config.timestep
            # For high gamma, check if timestep might be too large
            # Distance per step ≈ γ * c * h (for β ≈ 1)
            # For 300 mm/ns * γ * h, we want distance/step << aperture
            beta_approx = 1.0 if gamma_max > 2 else 0.9
            distance_per_step = beta_approx * gamma_max * 300.0 * timestep  # mm

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
        """Handle run sweep button click (called from main GUI)."""
        # Check if main GUI is already running
        if self.gui_controller and hasattr(self.gui_controller, "_running"):
            if self.gui_controller._running:
                messagebox.showwarning(
                    "Optimization",
                    "Please wait for current simulation to complete",
                )
                return

        # Validate inputs
        error = self._validate_inputs()
        if error:
            _show_error_dialog(self, "Invalid Input", error)
            return

        # Gather configuration
        try:
            self.config = self._gather_config()

            # Check for extreme parameters and warn user
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

            # Use stability options from main GUI tab (already loaded in self.config)
            self._log_result(
                "[INFO] Using stability options from main GUI Stability tab"
            )

            # Update robustness options from UI
            self.config.per_run_timeout = float(self.per_run_timeout_var.get())
            self.config.skip_failed_runs = self.skip_failed_runs_var.get()
            self.config.failed_run_retry_attempts = int(
                self.failed_run_retry_attempts_var.get()
            )

            # Update stability options from UI
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

        # Update UI state
        self._was_cancelled = False
        self.running = True
        self._update_progress(0, "Initializing sweep...")

        # Integrate with main GUI run state
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

        # Run in background thread
        thread = threading.Thread(target=self._run_sweep_background, daemon=True)
        thread.start()

    def _on_stop(self):
        """Handle stop button click."""
        self.running = False
        self._was_cancelled = True
        self._update_progress_text("Stopping...")

        # Signal main GUI cancellation
        if self.gui_controller and hasattr(self.gui_controller, "_cancel_requested"):
            self.gui_controller._cancel_requested = True

    def _compute_soft_penalty(
        self,
        *,
        aperture_radius: float,
        macroparticle_charge_multiplier: float,
        initial_energy_gev: float,
    ) -> float:
        """Estimate a soft penalty for risky parameter combinations.

        Small apertures combined with very high charge multipliers and beam energies
        almost always trigger gamma blow-ups. Rather than rejecting those points
        outright, apply a tunable penalty so the optimizer learns to avoid them
        while keeping the search numerically stable.
        """

        penalty = 0.0

        aperture_threshold_mm = 0.01  # 10 microns
        charge_threshold = 800.0
        energy_threshold = 120.0
        penalty_scale = 1.0e-3  # keeps penalty on the same order as metrics

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

    def _on_view_results(self):
        """Display pre-generated summary plots from the latest sweep/optimization run."""
        import glob
        import os

        # Use sweep output directory from GUI preferences
        default_results_dir = self.sweep_output_dir

        # Find all timestamped result directories
        if os.path.exists(default_results_dir):
            result_dirs = [
                d
                for d in glob.glob(os.path.join(default_results_dir, "*"))
                if os.path.isdir(d)
            ]
        else:
            result_dirs = []

        if result_dirs:
            # Sort by modification time, most recent first
            result_dirs.sort(key=os.path.getmtime, reverse=True)
            latest_dir = result_dirs[0]

            # Find PNG plots in the directory
            png_files = sorted(glob.glob(os.path.join(latest_dir, "*.png")))

            if png_files:
                self._display_summary_plots(latest_dir, png_files)
            else:
                # No plots found, offer to browse
                response = messagebox.askyesno(
                    "No Plots Found",
                    f"No summary plots found in:\n{os.path.basename(latest_dir)}\n\n"
                    "Would you like to browse for a different results directory?",
                    parent=self,
                )
                if response:
                    dir_path = filedialog.askdirectory(
                        title="Select Results Directory",
                        initialdir=default_results_dir,
                    )
                    if dir_path:
                        png_files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
                        if png_files:
                            self._display_summary_plots(dir_path, png_files)
                        else:
                            _show_info_dialog(
                                self,
                                "No Plots Found",
                                f"No PNG plot files found in:\n{dir_path}",
                            )
        else:
            # No result directories found, offer to browse
            response = messagebox.askyesno(
                "No Results Found",
                "No result directories found in the default location.\n\n"
                f"Default location: {default_results_dir}\n\n"
                "Would you like to browse for a results directory?",
                parent=self,
            )
            if response:
                dir_path = filedialog.askdirectory(
                    title="Select Results Directory",
                    initialdir=(
                        default_results_dir
                        if os.path.exists(default_results_dir)
                        else "."
                    ),
                )
                if dir_path:
                    png_files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
                    if png_files:
                        self._display_summary_plots(dir_path, png_files)
                    else:
                        _show_info_dialog(
                            self,
                            "No Plots Found",
                            f"No PNG plot files found in:\n{dir_path}",
                        )

    def _display_summary_plots(self, results_dir, png_files):
        """Display summary plots in a scrollable window.

        Parameters
        ----------
        results_dir : str
            Path to results directory
        png_files : list
            List of PNG file paths
        """
        from pathlib import Path

        try:
            from PIL import Image, ImageTk
        except ImportError as e:
            _show_error_dialog(
                self,
                "PIL/Pillow Not Installed",
                f"Cannot display images: PIL/Pillow is not installed.\n\n{e}\n\n"
                "Install with: pip install Pillow",
            )
            return

        dir_name = os.path.basename(results_dir)

        # Debug: Log what we're trying to load
        self._log_result(f"[INFO] Loading summary plots from: {results_dir}")
        self._log_result(f"[INFO] Found {len(png_files)} PNG files")

        # Create window
        plot_window = tk.Toplevel(self)
        plot_window.title(f"Summary Plots: {dir_name}")
        plot_window.geometry("1000x800")

        # Main frame
        main_frame = ttk.Frame(plot_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        ttk.Label(
            main_frame,
            text=f"Summary Plots: {dir_name}",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(pady=(0, 10))

        # Create canvas with scrollbar for plots
        canvas = tk.Canvas(main_frame, bg="white")
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Load and display each PNG
        # Store as window attribute to prevent garbage collection
        plot_window.photo_images = []

        for png_file in png_files:
            try:
                # Debug: Log each file
                self._log_result(f"[INFO] Loading: {Path(png_file).name}")

                # Load image
                img = Image.open(png_file)

                # Debug: Log image info
                self._log_result(
                    f"[INFO] Image size: {img.width}x{img.height}, mode: {img.mode}"
                )

                # Resize if too large (maintain aspect ratio)
                max_width = 950
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                    self._log_result(f"[INFO] Resized to: {img.width}x{img.height}")

                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                plot_window.photo_images.append(photo)

                # Plot name label
                plot_name = Path(png_file).stem.replace("_", " ").title()
                ttk.Label(
                    scrollable_frame,
                    text=plot_name,
                    font=("TkDefaultFont", 10, "bold"),
                ).pack(pady=(10, 5))

                # Image label
                img_label = tk.Label(scrollable_frame, image=photo, bg="white")
                img_label.pack(pady=(0, 20))

                self._log_result(
                    f"[INFO] Successfully displayed: {Path(png_file).name}"
                )

            except Exception as e:
                # If image loading fails, show error in both GUI and log
                import traceback

                error_msg = f"Error loading {Path(png_file).name}: {e}"
                self._log_result(f"[ERROR] {error_msg}")
                self._log_result(f"[ERROR] Traceback: {traceback.format_exc()}")

                error_label = ttk.Label(
                    scrollable_frame,
                    text=error_msg,
                    foreground="red",
                )
                error_label.pack(pady=5)

        # Debug: Final summary
        self._log_result(
            f"[INFO] Finished loading {len(plot_window.photo_images)} images successfully"
        )

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(10, 0))

        ttk.Button(
            button_frame,
            text="Close",
            command=plot_window.destroy,
        ).pack()

        # Bind mouse wheel to scroll
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Cleanup binding when window closes
        def on_close():
            canvas.unbind_all("<MouseWheel>")
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_close)

    def _load_and_plot_results(self, file_path: str):
        """Load results file and display trajectory viewer with plots."""
        try:
            # Only JSON files contain trajectory data
            # CSV files (all_evaluations.csv) only contain metrics
            with open(file_path, "r") as f:
                data = json.load(f)

            parsed = parse_results_payload(
                data,
                m_particle_amu=getattr(self.config, "m_particle", 0.00054857990907),
                amu_to_mev=AMU_TO_MEV,
            )

            if parsed["kind"] == "sweep":
                results = parsed["results"]
                if not results:
                    _show_info_dialog(self, "No Results", "No results found in file.")
                    return

                results_with_traj = parsed["results_with_trajectories"]
                if not results_with_traj:
                    self._show_results_summary(results, file_path)
                    return

            elif parsed["kind"] == "optimization":
                import os

                results_dir = os.path.dirname(file_path)
                self._view_npz_trajectories(results_dir)
                return

            else:
                results_with_traj = parsed["results_with_trajectories"]

            # Create trajectory viewer dialog and automatically plot
            self._show_trajectory_viewer(results_with_traj, file_path, auto_plot=True)

        except ValueError as e:
            if str(e) == UNKNOWN_RESULTS_FORMAT_MESSAGE:
                _show_info_dialog(self, "Unknown Format", UNKNOWN_RESULTS_FORMAT_MESSAGE)
                return
            raise
        except Exception as e:
            import traceback

            _show_error_dialog(
                self,
                "Error Loading File",
                f"Failed to load file:\n{e}\n\n{traceback.format_exc()}",
            )

    def _on_plot_trajectories(self):
        """Open trajectory plotting dialog to visualize saved results."""
        import glob
        import os

        # Start with the configured sweep output directory when it has results.
        if os.path.exists(self.sweep_output_dir) and os.listdir(self.sweep_output_dir):
            base_dir = self.sweep_output_dir
        else:
            base_dir = self.config.output_dir

        # Find most recent timestamped subdirectory if any exist
        initial_dir = base_dir
        if os.path.exists(base_dir):
            result_dirs = [
                d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)
            ]
            if result_dirs:
                # Sort by modification time, most recent first
                result_dirs.sort(key=os.path.getmtime, reverse=True)
                initial_dir = result_dirs[0]

        # Ask user to select results file or directory
        # Support JSON files (sweep_results.json or optimization_results.json)
        # CSV files only contain metrics, not trajectories
        # Show directory name in title for clarity
        import os

        dir_name = os.path.basename(initial_dir) if initial_dir else "results"
        file_path = filedialog.askopenfilename(
            title=f"Select Results File (JSON) - Starting in: {dir_name}",
            initialdir=initial_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        # If no file selected, offer to browse for NPZ directory
        if not file_path:
            response = messagebox.askyesno(
                "Browse for NPZ Trajectories?",
                "No file selected. Would you like to browse for a directory containing NPZ trajectory files?",
                parent=self,
            )
            if response:
                dir_path = filedialog.askdirectory(
                    title="Select Directory with NPZ Trajectory Files",
                    initialdir=initial_dir,
                )
                if dir_path:
                    self._view_npz_trajectories(dir_path)
            return

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            parsed = parse_results_payload(
                data,
                m_particle_amu=getattr(self.config, "m_particle", 0.00054857990907),
                amu_to_mev=AMU_TO_MEV,
            )

            if parsed["kind"] == "sweep":
                results = parsed["results"]
                if not results:
                    _show_info_dialog(self, "No Results", "No results found in file.")
                    return

                results_with_traj = parsed["results_with_trajectories"]

                if not results_with_traj:
                    _show_info_dialog(
                        self,
                        "No Trajectories",
                        "No trajectory data found in results.\n\n"
                        "Make sure 'Save trajectories' was enabled during the sweep.\n\n"
                        "Note: all_evaluations.csv only contains metrics, not trajectories.\n"
                        "For optimizations, trajectory data is in NPZ files.",
                    )
                    return

            elif parsed["kind"] == "optimization":
                import os

                results_dir = os.path.dirname(file_path)
                self._view_npz_trajectories(results_dir)
                return

            else:
                results_with_traj = parsed["results_with_trajectories"]

            # Create trajectory viewer dialog
            self._show_trajectory_viewer(results_with_traj, file_path)

        except ValueError as e:
            if str(e) == UNKNOWN_RESULTS_FORMAT_MESSAGE:
                _show_info_dialog(
                    self,
                    "Unknown Format",
                    f"{UNKNOWN_RESULTS_FORMAT_MESSAGE}\n\n"
                    "Note: CSV files only contain metrics, not trajectory data.",
                )
                return
            raise
        except Exception as e:
            import traceback

            _show_error_dialog(
                self,
                "Error Loading File",
                f"Failed to load file:\n{e}\n\n{traceback.format_exc()}",
            )

    def _show_results_summary(self, results, file_path):
        """Show metrics-first results summary (works without trajectory data).

        Args:
            results: List of result dictionaries (may or may not have trajectories)
            file_path: Path to the results file
        """
        dialog = tk.Toplevel(self)
        dialog.title(f"Results Summary - {Path(file_path).name}")
        dialog.geometry("1100x700")
        dialog.transient(self)

        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Title
        ttk.Label(
            main_frame,
            text="Sweep Results Summary",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        # Summary info
        num_runs = len(results)
        sweep_info = results[0].get("sweep_info", {}) if results else {}
        config_name = sweep_info.get("config_name", "Unknown")

        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(
            info_frame,
            text=f"Configuration: {config_name}  |  Total Runs: {num_runs}",
            font=("TkDefaultFont", 10),
        ).pack(anchor="w")

        # Notebook for different views
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True, pady=(5, 0))

        # Tab 1: Metrics Table
        metrics_frame = ttk.Frame(notebook, padding=10)
        notebook.add(metrics_frame, text="Metrics Table")

        # Create scrollable table
        table_container = ttk.Frame(metrics_frame)
        table_container.pack(fill="both", expand=True)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_container)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar = ttk.Scrollbar(table_container, orient="horizontal")
        h_scrollbar.pack(side="bottom", fill="x")

        # Text widget for table (easier than Treeview for variable columns)
        metrics_text = tk.Text(
            table_container,
            wrap="none",
            font=("Courier", 9),
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
        )
        metrics_text.pack(side="left", fill="both", expand=True)
        v_scrollbar.config(command=metrics_text.yview)
        h_scrollbar.config(command=metrics_text.xview)

        # Build table content
        if results:
            # Check if we have beam optics data in any result
            has_beam_optics = any(
                r.get("metrics", {}).get("rider_emittance_x_mm_mrad") is not None
                for r in results
            )

            # Header
            if has_beam_optics:
                header = f"{'Run':<5} {'Aperture (mm)':<15} {'Energy (GeV)':<15} {'Start_z (mm)':<15} {'ΔE (MeV)':<12} {'Traveled (mm)':<15} {'γ_initial':<12} {'εx (mm·mrad)':<15} {'εnx (mm·mrad)':<16} {'βx (m)':<12}\n"
                header += "-" * 157 + "\n"
            else:
                header = f"{'Run':<5} {'Aperture (mm)':<15} {'Energy (GeV)':<15} {'Start_z (mm)':<15} {'ΔE (MeV)':<12} {'Traveled (mm)':<15} {'γ_initial':<12}\n"
                header += "-" * 110 + "\n"
            metrics_text.insert("end", header)

            # Data rows
            for r in results:
                row_data = summarize_result_row(r)

                if has_beam_optics:
                    row = (
                        f"{row_data['run_num']:<5} {row_data['aperture']:<15.3e} "
                        f"{row_data['energy']:<15.2f} {row_data['start_z']:<15.1f} "
                        f"{row_data['delta_e']:<12.3f} {row_data['traveled']:<15.1f} "
                        f"{row_data['gamma_initial']:<12.1f} "
                        f"{row_data['emit_x']:<15.3e} {row_data['norm_emit_x']:<16.3e} "
                        f"{row_data['beta_x']:<12.3e}\n"
                    )
                else:
                    row = (
                        f"{row_data['run_num']:<5} {row_data['aperture']:<15.3e} "
                        f"{row_data['energy']:<15.2f} {row_data['start_z']:<15.1f} "
                        f"{row_data['delta_e']:<12.3f} {row_data['traveled']:<15.1f} "
                        f"{row_data['gamma_initial']:<12.1f}\n"
                    )
                metrics_text.insert("end", row)

        metrics_text.config(state="disabled")

        # Tab 2: Plots (if applicable)
        plots_frame = ttk.Frame(notebook, padding=10)
        notebook.add(plots_frame, text="Visualization")

        # Check if we can make plots
        has_trajectories = any("trajectory" in r for r in results)

        if has_trajectories:
            ttk.Label(
                plots_frame,
                text="Trajectory data available. Click below to view trajectory plots.",
                font=("TkDefaultFont", 10),
            ).pack(pady=20)

            ttk.Button(
                plots_frame,
                text="Open Trajectory Viewer",
                command=lambda: self._open_trajectory_viewer_from_summary(
                    dialog, results, file_path
                ),
                style="Accent.TButton",
            ).pack(pady=10)
        else:
            # Try to make parameter sweep plot if we have varied parameters
            self._create_summary_plots(plots_frame, results)

        # Bottom buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(
            btn_frame,
            text="Export to CSV",
            command=lambda: self._export_metrics_csv(results, file_path),
        ).pack(side="left", padx=5)

        ttk.Button(
            btn_frame,
            text="Close",
            command=dialog.destroy,
        ).pack(side="right", padx=5)

    def _create_summary_plots(self, parent_frame, results):
        """Create parameter sweep visualization plots."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )

            plot_data = collect_summary_plot_data(results)
            apertures = plot_data["apertures"]
            energies = plot_data["energies"]
            delta_es = plot_data["delta_es"]

            # Create figure
            fig = plt.figure(figsize=(10, 6))

            # Determine if we have 1D or 2D sweep
            unique_apertures = len(set(apertures))
            unique_energies = len(set(energies))

            if unique_apertures > 1 and unique_energies > 1:
                # 2D sweep - make heatmap
                ax = fig.add_subplot(111)
                unique_a, unique_e, grid = build_summary_heatmap_grid(results)

                im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn_r")
                ax.set_xticks(range(len(unique_a)))
                ax.set_xticklabels(
                    [f"{a:.1e}" for a in unique_a], rotation=45, ha="right"
                )
                ax.set_yticks(range(len(unique_e)))
                ax.set_yticklabels([f"{e:.1f}" for e in unique_e])
                ax.set_xlabel("Aperture Radius (mm)")
                ax.set_ylabel("Particle Energy (GeV)")
                ax.set_title("ΔE Heatmap (MeV)")
                plt.colorbar(im, ax=ax, label="ΔE (MeV)")

            elif unique_apertures > 1:
                # Vary aperture, fixed energy
                ax = fig.add_subplot(111)
                ax.plot(apertures, delta_es, "o-", markersize=8)
                ax.set_xlabel("Aperture Radius (mm)")
                ax.set_ylabel("ΔE (MeV)")
                ax.set_title(f"Energy Change vs Aperture (E={energies[0]:.1f} GeV)")
                ax.grid(True, alpha=0.3)

            elif unique_energies > 1:
                # Vary energy, fixed aperture
                ax = fig.add_subplot(111)
                ax.plot(energies, delta_es, "o-", markersize=8)
                ax.set_xlabel("Particle Energy (GeV)")
                ax.set_ylabel("ΔE (MeV)")
                ax.set_title(f"Energy Change vs Energy (a={apertures[0]:.2e} mm)")
                ax.grid(True, alpha=0.3)
            else:
                # Single point
                ax = fig.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "Single-point simulation\nNo parameter sweep to visualize",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")

            fig.tight_layout()

            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, parent_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            toolbar = NavigationToolbar2Tk(canvas, parent_frame)
            toolbar.update()

        except Exception as e:
            ttk.Label(
                parent_frame,
                text=f"Could not create plots: {e}",
                foreground="red",
            ).pack(pady=20)

    def _export_metrics_csv(self, results, file_path):
        """Export metrics to CSV file."""
        import csv
        from tkinter import filedialog

        # Suggest filename
        default_name = Path(file_path).stem + "_metrics.csv"
        output_file = filedialog.asksaveasfilename(
            title="Export Metrics to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_name,
            parent=self,
        )

        if not output_file:
            return

        try:
            with open(output_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Header
                writer.writerow(
                    [
                        "Run",
                        "Aperture_mm",
                        "Energy_GeV",
                        "Start_z_mm",
                        "Delta_E_MeV",
                        "Traveled_mm",
                        "Gamma_initial",
                        "Gamma_final",
                        "Emittance_x_mm_mrad",
                        "Emittance_y_mm_mrad",
                        "Norm_Emittance_x_mm_mrad",
                        "Norm_Emittance_y_mm_mrad",
                        "Beta_x_m",
                        "Beta_y_m",
                    ]
                )

                # Data
                for r in results:
                    row = summarize_result_row(r)

                    writer.writerow(
                        [
                            row["run_num"],
                            row["aperture"],
                            row["energy"],
                            row["start_z"],
                            row["delta_e"],
                            row["traveled"],
                            row["gamma_initial"],
                            row["gamma_final"],
                            row["emit_x"],
                            row["emit_y"],
                            row["norm_emit_x"],
                            row["norm_emit_y"],
                            row["beta_x"],
                            row["beta_y"],
                        ]
                    )

            _show_info_dialog(
                self, "Export Successful", f"Metrics exported to:\n{output_file}"
            )

        except Exception as e:
            _show_error_dialog(self, "Export Failed", f"Failed to export CSV:\n{e}")

    def _open_trajectory_viewer_from_summary(self, summary_dialog, results, file_path):
        """Open trajectory viewer from the summary dialog."""
        results_with_traj = [r for r in results if "trajectory" in r]
        if results_with_traj:
            self._show_trajectory_viewer(results_with_traj, file_path, auto_plot=True)
        else:
            _show_info_dialog(
                summary_dialog,
                "No Trajectories",
                "No trajectory data found in results.",
            )

    def _show_trajectory_viewer(self, results, file_path, auto_plot=False):
        """Show trajectory viewer dialog with run selection and plotting.

        Args:
            results: List of result dictionaries with trajectories
            file_path: Path to the results file
            auto_plot: If True, automatically select and plot results on open
        """
        dialog = tk.Toplevel(self)
        dialog.title(f"Trajectory Viewer - {Path(file_path).name}")
        dialog.geometry("1000x700")
        dialog.transient(self)

        # Main container
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Left panel: Run selection
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side="left", fill="both", expand=False, padx=(0, 5))

        ttk.Label(
            left_panel, text="Select Runs to Plot:", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))

        # Scrollable listbox for runs
        list_frame = ttk.Frame(left_panel)
        list_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        run_listbox = tk.Listbox(
            list_frame,
            selectmode="extended",
            width=40,
            height=20,
            yscrollcommand=scrollbar.set,
        )
        run_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=run_listbox.yview)

        # Populate listbox with run summaries
        for r in results:
            params = r.get("parameters", {})
            run_num = r.get("run_number", "?")
            aperture = params.get("aperture_radius", 0)
            energy = params.get("particle_energy_gev", 0)
            delta_e = r.get("metrics", {}).get("rider_delta_e_mev", 0)

            summary = (
                f"Run #{run_num}: "
                f"a={aperture:.2e}mm, E={energy:.1f}GeV, "
                f"ΔE={delta_e:.6f}MeV"
            )
            run_listbox.insert("end", summary)

        # Control buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=(10, 0))

        plot_button = ttk.Button(
            btn_frame,
            text="Plot Selected",
            command=lambda: self._plot_selected_trajectories(
                run_listbox, results, dialog
            ),
        )
        plot_button.pack(fill="x", pady=2)

        select_all_btn = ttk.Button(
            btn_frame,
            text="Select All",
            command=lambda: run_listbox.select_set(0, "end"),
        )
        select_all_btn.pack(fill="x", pady=2)

        clear_btn = ttk.Button(
            btn_frame,
            text="Clear Selection",
            command=lambda: run_listbox.selection_clear(0, "end"),
        )
        clear_btn.pack(fill="x", pady=2)

        # Right panel: Plot display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)

        ttk.Label(
            right_panel, text="Plot Area", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))

        # Placeholder for matplotlib canvas
        plot_info = ttk.Label(
            right_panel,
            text="Select runs and click 'Plot Selected' to visualize trajectories.\n\n"
            "Transverse plots will be shown as scatter plots.",
            justify="center",
            foreground="gray",
        )
        plot_info.pack(expand=True)

        # Store for later use
        dialog.plot_area = right_panel
        dialog.plot_info = plot_info

        # Auto-plot if requested (for View Results button)
        if auto_plot:
            # Select all runs (or up to 10 for performance)
            max_auto_plot = min(10, len(results))
            for i in range(max_auto_plot):
                run_listbox.select_set(i)

            # Force widget and window updates
            run_listbox.update_idletasks()
            dialog.update()

            # Schedule plotting with enough delay for window to fully initialize
            # Use a longer delay and check that selection is valid before plotting
            def safe_auto_plot():
                if run_listbox.curselection():
                    self._plot_selected_trajectories(
                        run_listbox, results, dialog, is_auto_plot=True
                    )
                else:
                    # Fallback: select again and plot
                    for i in range(max_auto_plot):
                        run_listbox.select_set(i)
                    run_listbox.update()
                    dialog.after(
                        100,
                        lambda: self._plot_selected_trajectories(
                            run_listbox, results, dialog, is_auto_plot=True
                        ),
                    )

            dialog.after(200, safe_auto_plot)

    def _plot_selected_trajectories(
        self, listbox, results, parent_dialog, is_auto_plot=False
    ):
        """Plot trajectories for selected runs.

        Args:
            listbox: The listbox containing run selections
            results: List of result dictionaries
            parent_dialog: Parent dialog window
            is_auto_plot: If True, suppress error dialogs on empty selection
        """
        # Force update to ensure selection is current
        listbox.update_idletasks()
        selection = listbox.curselection()
        if not selection:
            # Only show dialog if this is a user-initiated action (not auto-plot)
            if not is_auto_plot and listbox.size() > 0:
                _show_info_dialog(
                    parent_dialog,
                    "No Selection",
                    "Please select at least one run to plot.",
                )
            return

        selected_results = [results[i] for i in selection]

        # Clear previous plot
        for widget in parent_dialog.plot_area.winfo_children():
            widget.destroy()

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )

            # Create figure with 3 subplots as requested
            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 1, hspace=0.3)

            ax_delta_e = fig.add_subplot(gs[0])
            ax_transverse = fig.add_subplot(gs[1])
            ax_heatmap = fig.add_subplot(gs[2])

            fig.suptitle(
                f"Sweep Results: {len(selected_results)} run(s)",
                fontsize=12,
                fontweight="bold",
            )

            plot_data = build_trajectory_plot_data(
                selected_results,
                m_particle_amu=getattr(self.config, "m_particle", 0.00054857990907),
                amu_to_mev=AMU_TO_MEV,
            )
            heatmap = plot_data["heatmap"]

            # Plot each selected trajectory
            for idx, series in enumerate(plot_data["series"]):
                label = (
                    f"Run #{series['run_num']} "
                    f"(a={series['aperture']:.2e}mm, E={series['energy']:.1f}GeV)"
                )
                color = plt.cm.tab10(idx % 10)

                ax_delta_e.plot(
                    series["z"],
                    series["energy_delta"],
                    label=label,
                    alpha=0.7,
                    color=color,
                    linewidth=1.5,
                )

                # Plot 2: x and y positions versus z (need to extract from r)
                # Since we only have r (radial distance), we'll plot r and -r to show transverse extent
                # In a real case, you'd have separate x and y coordinates
                ax_transverse.plot(
                    series["z"],
                    series["r"],
                    label=f"{label} (+r)",
                    alpha=0.6,
                    color=color,
                    linewidth=1.5,
                )
                ax_transverse.plot(
                    series["z"],
                    -series["r"],
                    alpha=0.3,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                )

            # Set labels and styling for Plot 1
            ax_delta_e.set_xlabel("z position (mm)", fontsize=10)
            ax_delta_e.set_ylabel("ΔE (MeV)", fontsize=10)
            ax_delta_e.set_title(
                "Energy Gain vs Position", fontsize=11, fontweight="bold"
            )
            ax_delta_e.legend(fontsize=7, loc="best")
            ax_delta_e.grid(True, alpha=0.3)

            # Set labels and styling for Plot 2
            ax_transverse.set_xlabel("z position (mm)", fontsize=10)
            ax_transverse.set_ylabel("Transverse position (mm)", fontsize=10)
            ax_transverse.set_title(
                "Transverse Position (±r) vs z", fontsize=11, fontweight="bold"
            )
            ax_transverse.legend(fontsize=7, loc="best")
            ax_transverse.grid(True, alpha=0.3)
            ax_transverse.axhline(
                y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3
            )

            # Plot 3: Heatmap (aperture vs energy, colored by delta_e)
            # Only show heatmap if both aperture and energy were swept
            apertures = heatmap["apertures"]
            energies = heatmap["energies"]
            delta_es = heatmap["delta_es"]
            unique_apertures = len(set(apertures))
            unique_energies = len(set(energies))

            if len(apertures) > 0 and unique_apertures > 1 and unique_energies > 1:
                # Create scatter plot for heatmap
                scatter = ax_heatmap.scatter(
                    energies,
                    [
                        a * 1e3 for a in apertures
                    ],  # Convert mm to microns for readability
                    c=delta_es,
                    cmap="viridis",
                    s=100,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )

                cbar = plt.colorbar(scatter, ax=ax_heatmap)
                cbar.set_label("ΔE (MeV)", fontsize=10)

                ax_heatmap.set_xlabel("Particle Energy (GeV)", fontsize=10)
                ax_heatmap.set_ylabel("Aperture Radius (μm)", fontsize=10)
                ax_heatmap.set_title(
                    "Parameter Space: ΔE(Energy, Aperture)",
                    fontsize=11,
                    fontweight="bold",
                )
                ax_heatmap.grid(True, alpha=0.3)

                # Use log scale if appropriate
                if max(energies) / min(energies) > 10 if min(energies) > 0 else False:
                    ax_heatmap.set_xscale("log")
                if (
                    max(apertures) / min(apertures) > 10
                    if min(apertures) > 0
                    else False
                ):
                    ax_heatmap.set_yscale("log")
            else:
                # Hide heatmap or show message
                ax_heatmap.text(
                    0.5,
                    0.5,
                    "Heatmap requires sweep over both\naperture and energy parameters",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="gray",
                    transform=ax_heatmap.transAxes,
                )
                ax_heatmap.set_xticks([])
                ax_heatmap.set_yticks([])
                ax_heatmap.set_title(
                    "Parameter Space Heatmap (N/A)",
                    fontsize=11,
                    fontweight="bold",
                    color="gray",
                )

            plt.tight_layout()

            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=parent_dialog.plot_area)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, parent_dialog.plot_area)
            toolbar.update()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except ImportError:
            _show_error_dialog(
                self,
                "Missing Dependency",
                "Matplotlib is required for plotting.\n\nInstall with: pip install matplotlib",
            )
        except Exception as e:
            _show_error_dialog(
                self, "Plotting Error", f"Failed to plot trajectories:\n{e}"
            )

    def _update_progress(self, value: float, text: str):
        """Update progress bar and label (thread-safe)."""

        def update():
            self.progress_bar["value"] = value
            self.progress_label["text"] = text

        self.after(0, update)

    def _update_progress_text(self, text: str):
        """Update only the progress label text (thread-safe)."""
        self.after(0, lambda: self.progress_label.config(text=text))

    def _log_result(self, message: str):
        """Log message to main GUI logs window (thread-safe)."""
        # Log to console/terminal always
        print(f"[OPTIMIZATION] {message}", flush=True)

        # Write to log file if enabled
        if self._log_file is not None:
            try:
                self._log_file.write(f"[OPTIMIZATION] {message}\n")
                self._log_file.flush()  # Ensure it's written immediately
            except Exception as e:
                print(f"[WARNING] Failed to write to log file: {e}", flush=True)

        # If we have a gui_controller, log to its log window
        if self.gui_controller is not None and hasattr(
            self.gui_controller, "_append_log"
        ):
            try:
                gui = self.gui_controller  # Type guard
                self.after(
                    0,
                    lambda: gui._append_log(f"[OPTIMIZATION] {message}"),
                )
            except Exception:
                pass  # Fail silently if main GUI log isn't available

    def _open_log_file(self, output_dir):
        """Open a log file in the output directory."""
        from datetime import datetime
        from pathlib import Path

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"optimization_log_{timestamp}.txt"
            self._log_file_path = output_path / log_filename

            self._log_file = open(self._log_file_path, "w", encoding="utf-8")
            self._log_result(f"Log file opened: {self._log_file_path}")
            return True
        except Exception as e:
            print(f"[WARNING] Failed to open log file: {e}", flush=True)
            self._log_file = None
            self._log_file_path = None
            return False

    def _close_log_file(self):
        """Close the log file if it's open."""
        if self._log_file is not None:
            try:
                self._log_result("Closing log file")
                self._log_file.close()
            except Exception as e:
                print(f"[WARNING] Failed to close log file: {e}", flush=True)
            finally:
                self._log_file = None
                self._log_file_path = None

    def _reset_ui_state(self):
        """Reset UI to ready state after run completes."""
        # Reset main GUI state if integrated
        if self.gui_controller and hasattr(self.gui_controller, "_running"):
            self.gui_controller._running = False
            if hasattr(self.gui_controller, "_cancel_requested"):
                self.gui_controller._cancel_requested = False
            if hasattr(self.gui_controller, "_set_status"):
                self.gui_controller._set_status("Ready")
            if hasattr(self.gui_controller, "_run_button"):
                self.gui_controller._run_button.configure(state="normal")
            if hasattr(self.gui_controller, "_cancel_button"):
                self.gui_controller._cancel_button.configure(state="disabled")
        if not self.running:
            self._update_progress_text("Ready")
