"""Config load/save and UI<->options mapping helpers for the main GUI."""

from __future__ import annotations

import math
import os
from pathlib import Path
from tkinter import messagebox

from core.particle_config import DEFAULT_DRIVER_PARAMS
from core.types import SimulationType

from .testbed_runner import (
    CORE_PARAM_DEFAULTS,
    PARTICLE_PARAM_FIELDS,
    SimulationOptions,
    load_config,
    save_config,
    supports_driver,
)


class IntegratorGUIConfigMixin:
    """Translate between GUI state and ``SimulationOptions`` configs."""

    def _load_config(self) -> None:
        from .gui import _show_error_dialog

        filename = self._selected_config_filename()
        if not filename:
            messagebox.showinfo("Load config", "Select a configuration to load.")
            return

        path = Path(self.config_dir_var.get()) / filename
        try:
            options = load_config(path)
        except Exception as exc:
            _show_error_dialog(
                self.root, "Load config", f"Failed to load {filename}: {exc}"
            )
            return

        self._apply_options_to_ui(options, preserve_directories=True)
        self.config_name_var.set(filename)
        self.config_file_var.set(filename)

        self.run_mode_var.set("single")
        self._on_run_mode_changed()
        print("[INFO] Auto-switched to Single Run mode")

        self._refresh_config_list(selected=filename)
        self._refresh_initial_summary()
        self._update_driver_visibility()
        self._update_image_subcharge_state()
        self._update_cavity_spacing_state()
        self._toggle_z_cutoff_controls()
        self._toggle_macroparticle_controls()
        self._update_macroparticle_state()

        current_value = self.sim_type_var.get()
        try:
            values_list = list(self.sim_type_combo["values"])
            if current_value in values_list:
                idx = values_list.index(current_value)
                self.sim_type_combo.current(idx)
                self.root.update_idletasks()
        except Exception:
            pass

        self._set_status(f"Loaded config: {filename}")
        self.current_config_label.config(text=filename, foreground="black")

    def _apply_options_to_ui(
        self, options: SimulationOptions, preserve_directories: bool = False
    ) -> None:
        self.options = options
        self.sim_type_var.set(options.simulation_type.name)
        self.steps_var.set(options.steps)
        self.seed_var.set(options.seed)
        self.energy_display_var.set(options.energy_display)
        self.energy_save_var.set(options.energy_save)
        self.energy_xaxis_var.set(getattr(options, "energy_xaxis", "z"))
        self.energy_yaxis_var.set(getattr(options, "energy_yaxis", "delta_total"))
        self.transverse_display_var.set(options.transverse_display)
        self.transverse_save_var.set(options.transverse_save)
        self.transverse_xaxis_var.set(getattr(options, "transverse_xaxis", "t"))
        self.beta_display_var.set(options.beta_display)
        self.beta_save_var.set(options.beta_save)
        self.beta_xaxis_var.set(getattr(options, "beta_xaxis", "t"))
        self.momentum_display_var.set(options.momentum_display)
        self.momentum_save_var.set(options.momentum_save)
        self.momentum_xaxis_var.set(getattr(options, "momentum_xaxis", "t"))
        self.gamma_display_var.set(getattr(options, "gamma_display", False))
        self.gamma_save_var.set(getattr(options, "gamma_save", False))
        self.gamma_xaxis_var.set(getattr(options, "gamma_xaxis", "t"))
        self.zposition_display_var.set(getattr(options, "zposition_display", False))
        self.zposition_save_var.set(getattr(options, "zposition_save", False))
        self.trajectory_save_var.set(options.trajectory_save)
        self.trajectory_interval_var.set(options.trajectory_interval)
        self.dpi_var.set(options.plot_dpi)
        self.image_subcharge_var.set(options.image_subcharge_count)
        self.image_weighting_var.set(options.use_image_weighting)
        self.use_numba_var.set(getattr(options, "use_numba", True))
        self.macroparticle_enabled_var.set(
            getattr(options, "macroparticle_enabled", False)
        )
        self.macroparticle_charge_multiplier_var.set(
            getattr(options, "macroparticle_charge_multiplier", 1.0)
        )
        self.macroparticle_sigma_multiplier_var.set(
            getattr(options, "macroparticle_sigma_multiplier", 1.0)
        )
        self.macroparticle_use_momentum_errors_var.set(
            getattr(options, "macroparticle_use_momentum_errors", True)
        )
        self.self_consistency_enabled_var.set(options.self_consistency_enabled)
        self.self_consistency_convergence_mode_var.set(
            options.self_consistency_convergence_mode
        )
        self.self_consistency_mass_shell_relaxation_var.set(
            options.self_consistency_mass_shell_relaxation
        )
        self.self_consistency_target_ms_tolerance_var.set(
            options.self_consistency_target_ms_tolerance
        )
        self.self_consistency_max_iterations_var.set(
            options.self_consistency_max_iterations
        )
        self.self_consistency_mass_shell_tolerance_var.set(
            options.self_consistency_mass_shell_tolerance
        )
        self.self_consistency_verbosity_var.set(options.self_consistency_verbosity)
        self.self_consistency_chrono_interpolate_var.set(
            getattr(options, "self_consistency_chrono_interpolate", False)
        )
        self.self_consistency_chrono_tolerance_var.set(
            getattr(options, "self_consistency_chrono_tolerance", 1e-3)
        )
        self.self_consistency_chrono_high_precision_var.set(
            getattr(options, "self_consistency_chrono_high_precision", False)
        )
        self.self_consistency_chrono_adaptive_tolerance_var.set(
            getattr(options, "self_consistency_chrono_adaptive_tolerance", False)
        )
        self.self_consistency_gamma_reconciliation_method_var.set(
            getattr(
                options,
                "self_consistency_gamma_reconciliation_method",
                "ADAPTIVE_WEIGHTED",
            )
        )
        self.self_consistency_gamma_reconciliation_low_beta_threshold_var.set(
            getattr(
                options, "self_consistency_gamma_reconciliation_low_beta_threshold", 0.9
            )
        )
        self.self_consistency_gamma_reconciliation_high_beta_threshold_var.set(
            getattr(
                options,
                "self_consistency_gamma_reconciliation_high_beta_threshold",
                0.99,
            )
        )
        self.self_consistency_gamma_reconciliation_low_beta_weight_var.set(
            getattr(
                options, "self_consistency_gamma_reconciliation_low_beta_weight", 0.8
            )
        )
        self.self_consistency_gamma_reconciliation_high_beta_weight_var.set(
            getattr(
                options, "self_consistency_gamma_reconciliation_high_beta_weight", 0.2
            )
        )
        self.self_consistency_gamma_reconciliation_mid_beta_weight_var.set(
            getattr(
                options, "self_consistency_gamma_reconciliation_mid_beta_weight", 0.5
            )
        )
        self.self_consistency_gamma_reconciliation_fixed_weight_var.set(
            getattr(options, "self_consistency_gamma_reconciliation_fixed_weight", 0.5)
        )
        self.adaptive_timestep_enabled_var.set(options.adaptive_timestep_enabled)
        self.adaptive_timestep_halt_on_jump_var.set(options.energy_monitor_halt_on_jump)
        self.adaptive_timestep_threshold_var.set(options.adaptive_timestep_threshold)
        self.adaptive_timestep_reduction_factor_var.set(
            options.adaptive_timestep_reduction_factor
        )
        self._update_max_attempts_display()
        self.adaptive_timestep_min_factor_var.set(options.adaptive_timestep_min_factor)
        self.adaptive_timestep_cooldown_steps_var.set(
            options.adaptive_timestep_cooldown_steps
        )
        self.adaptive_timestep_probe_threshold_var.set(
            options.adaptive_timestep_probe_threshold
        )
        self.adaptive_timestep_max_probe_steps_var.set(
            options.adaptive_timestep_max_probe_steps
        )
        self.adaptive_timestep_debug_var.set(options.adaptive_timestep_debug)
        self._update_max_substeps_display()
        self.save_log_file_var.set(options.save_log_file)

        if not preserve_directories:
            self.output_dir_var.set(str(options.output_dir))
            self.config_dir_var.set(str(options.config_dir))

        self.config_name_var.set(options.config_name)

        default_species_label = self._species_label_by_key.get(
            "custom", next(iter(self._species_by_label))
        )
        self.rider_species_var.set(default_species_label)
        self.driver_species_var.set(default_species_label)

        for name in PARTICLE_PARAM_FIELDS:
            self.rider_param_vars[name].set(options.rider_params[name])
            driver_value = (
                options.driver_params[name]
                if options.driver_params is not None and name in options.driver_params
                else DEFAULT_DRIVER_PARAMS[name]
            )
            self.driver_param_vars[name].set(driver_value)
        for name in CORE_PARAM_DEFAULTS:
            self.core_param_vars[name].set(options.core_params[name])

        z_cutoff_val = options.core_params.get("z_cutoff", 0.0)
        self.z_cutoff_enabled_var.set(z_cutoff_val != 0.0)
        self._toggle_z_cutoff_controls()

    def _update_max_attempts_display(self):
        """Update derived adaptive-timestep attempt display."""
        try:
            reduction_factor = self.adaptive_timestep_reduction_factor_var.get()
            min_factor = self.adaptive_timestep_min_factor_var.get()

            if reduction_factor <= 1 or min_factor <= 0:
                self.adaptive_timestep_max_attempts_display_var.set("N/A")
                return

            attempts = math.ceil(
                math.log(1.0 / min_factor) / math.log(reduction_factor)
            )
            attempts = max(1, attempts)
            self.adaptive_timestep_max_attempts_display_var.set(
                f"{attempts} (from reduction & min factor)"
            )
        except (ValueError, ZeroDivisionError):
            self.adaptive_timestep_max_attempts_display_var.set("N/A")

    def _update_max_substeps_display(self):
        """Update derived adaptive-timestep substep display."""
        try:
            min_factor = self.adaptive_timestep_min_factor_var.get()
            theoretical_max = math.ceil(1.0 / min_factor)
            with_margin = int(theoretical_max * 1.1)
            self.adaptive_timestep_max_substeps_display_var.set(
                f"{with_margin} (from min factor)"
            )
        except (ValueError, ZeroDivisionError):
            self.adaptive_timestep_max_substeps_display_var.set("N/A")

    def _build_options_from_ui(self) -> SimulationOptions:
        sim_type = SimulationType[self.sim_type_var.get()]
        rider_params = {
            name: self.rider_param_vars[name].get() for name in PARTICLE_PARAM_FIELDS
        }
        driver_supported = supports_driver(sim_type)
        driver_params = (
            {name: self.driver_param_vars[name].get() for name in PARTICLE_PARAM_FIELDS}
            if driver_supported
            else None
        )
        core_params = {}
        for name in CORE_PARAM_DEFAULTS:
            value = self.core_param_vars[name].get()
            if isinstance(CORE_PARAM_DEFAULTS[name], str):
                core_params[name] = value
            else:
                core_params[name] = float(value)

        if not self.z_cutoff_enabled_var.get():
            core_params["z_cutoff"] = 0.0

        config_name = self.config_name_var.get().strip() or "testbed_config"
        if not config_name.endswith(".json"):
            config_name += ".json"

        if self.random_seed_var.get():
            import random

            seed = random.randint(1, 2**31 - 1)
        else:
            seed = int(self.seed_var.get())

        return SimulationOptions(
            simulation_type=sim_type,
            steps=int(self.steps_var.get()),
            seed=seed,
            rider_params=rider_params,
            driver_params=driver_params,
            core_params=core_params,
            energy_display=bool(self.energy_display_var.get()),
            energy_save=bool(self.energy_save_var.get()),
            energy_xaxis=str(self.energy_xaxis_var.get()),
            energy_yaxis=str(self.energy_yaxis_var.get()),
            transverse_display=bool(self.transverse_display_var.get()),
            transverse_save=bool(self.transverse_save_var.get()),
            transverse_xaxis=str(self.transverse_xaxis_var.get()),
            beta_display=bool(self.beta_display_var.get()),
            beta_save=bool(self.beta_save_var.get()),
            beta_xaxis=str(self.beta_xaxis_var.get()),
            momentum_display=bool(self.momentum_display_var.get()),
            momentum_save=bool(self.momentum_save_var.get()),
            momentum_xaxis=str(self.momentum_xaxis_var.get()),
            gamma_display=bool(self.gamma_display_var.get()),
            gamma_save=bool(self.gamma_save_var.get()),
            gamma_xaxis=str(self.gamma_xaxis_var.get()),
            zposition_display=bool(self.zposition_display_var.get()),
            zposition_save=bool(self.zposition_save_var.get()),
            trajectory_save=bool(self.trajectory_save_var.get()),
            trajectory_interval=int(self.trajectory_interval_var.get()),
            plot_dpi=int(self.dpi_var.get()),
            output_dir=Path(self.output_dir_var.get()),
            config_dir=Path(self.config_dir_var.get()),
            config_name=config_name,
            image_subcharge_count=int(self.image_subcharge_var.get()),
            use_image_weighting=bool(self.image_weighting_var.get()),
            macroparticle_enabled=bool(self.macroparticle_enabled_var.get()),
            macroparticle_charge_multiplier=float(
                self.macroparticle_charge_multiplier_var.get()
            ),
            macroparticle_sigma_multiplier=float(
                self.macroparticle_sigma_multiplier_var.get()
            ),
            macroparticle_use_momentum_errors=bool(
                self.macroparticle_use_momentum_errors_var.get()
            ),
            self_consistency_enabled=bool(self.self_consistency_enabled_var.get()),
            self_consistency_convergence_mode=str(
                self.self_consistency_convergence_mode_var.get()
            ),
            self_consistency_mass_shell_relaxation=float(
                self.self_consistency_mass_shell_relaxation_var.get()
            ),
            self_consistency_target_ms_tolerance=float(
                self.self_consistency_target_ms_tolerance_var.get()
            ),
            self_consistency_max_iterations=int(
                self.self_consistency_max_iterations_var.get()
            ),
            self_consistency_mass_shell_tolerance=float(
                self.self_consistency_mass_shell_tolerance_var.get()
            ),
            self_consistency_verbosity=int(self.self_consistency_verbosity_var.get()),
            self_consistency_chrono_interpolate=bool(
                self.self_consistency_chrono_interpolate_var.get()
            ),
            self_consistency_chrono_tolerance=float(
                self.self_consistency_chrono_tolerance_var.get()
            ),
            self_consistency_chrono_high_precision=bool(
                self.self_consistency_chrono_high_precision_var.get()
            ),
            self_consistency_chrono_adaptive_tolerance=bool(
                self.self_consistency_chrono_adaptive_tolerance_var.get()
            ),
            self_consistency_gamma_reconciliation_method=self.self_consistency_gamma_reconciliation_method_var.get(),
            self_consistency_gamma_reconciliation_low_beta_threshold=float(
                self.self_consistency_gamma_reconciliation_low_beta_threshold_var.get()
            ),
            self_consistency_gamma_reconciliation_high_beta_threshold=float(
                self.self_consistency_gamma_reconciliation_high_beta_threshold_var.get()
            ),
            self_consistency_gamma_reconciliation_low_beta_weight=float(
                self.self_consistency_gamma_reconciliation_low_beta_weight_var.get()
            ),
            self_consistency_gamma_reconciliation_high_beta_weight=float(
                self.self_consistency_gamma_reconciliation_high_beta_weight_var.get()
            ),
            self_consistency_gamma_reconciliation_mid_beta_weight=float(
                self.self_consistency_gamma_reconciliation_mid_beta_weight_var.get()
            ),
            self_consistency_gamma_reconciliation_fixed_weight=float(
                self.self_consistency_gamma_reconciliation_fixed_weight_var.get()
            ),
            self_consistency_chrono_matching_mode="FAST",
            energy_monitor_enabled=False,
            energy_monitor_threshold=2.0,
            energy_monitor_check_interval=10,
            energy_monitor_halt_on_jump=bool(
                self.adaptive_timestep_halt_on_jump_var.get()
            ),
            energy_monitor_debug=False,
            adaptive_timestep_enabled=bool(self.adaptive_timestep_enabled_var.get()),
            adaptive_timestep_threshold=float(
                self.adaptive_timestep_threshold_var.get()
            ),
            adaptive_timestep_reduction_factor=int(
                self.adaptive_timestep_reduction_factor_var.get()
            ),
            adaptive_timestep_min_factor=float(
                self.adaptive_timestep_min_factor_var.get()
            ),
            adaptive_timestep_cooldown_steps=int(
                self.adaptive_timestep_cooldown_steps_var.get()
            ),
            adaptive_timestep_probe_threshold=float(
                self.adaptive_timestep_probe_threshold_var.get()
            ),
            adaptive_timestep_max_probe_steps=int(
                self.adaptive_timestep_max_probe_steps_var.get()
            ),
            adaptive_timestep_debug=bool(self.adaptive_timestep_debug_var.get()),
            save_log_file=bool(self.save_log_file_var.get()),
        )

    def _save_config(self) -> None:
        from .gui import _show_error_dialog

        try:
            options = self._build_options_from_ui()
        except ValueError as exc:
            _show_error_dialog(self.root, "Invalid configuration", str(exc))
            return

        filename = self.config_name_var.get().strip()
        if not filename:
            messagebox.showinfo("Save Run Config", "Enter a config name to save.")
            return

        if not filename.endswith(".json"):
            filename += ".json"

        config_dir = self.config_dir_var.get()
        os.makedirs(config_dir, exist_ok=True)

        filepath = os.path.join(config_dir, filename)
        if not self._check_override_warning(Path(filepath), "run"):
            return

        options.config_name = filename

        try:
            save_config(options, Path(filepath))
        except Exception as exc:
            _show_error_dialog(
                self.root, "Save config", f"Failed to save configuration: {exc}"
            )
            return

        self.config_name_var.set(filename)
        self.config_file_var.set(filename)
        self._refresh_config_list(selected=filename)
        self.current_config_label.config(text=filename, foreground="black")
        messagebox.showinfo("Save Run Config", f"Configuration saved as {filename}")
        self._set_status(f"Saved config: {filename}")
