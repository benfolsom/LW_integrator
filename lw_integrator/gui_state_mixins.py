"""Widget state and visibility helpers for the main GUI."""

from __future__ import annotations

from tkinter import ttk

from core.types import SimulationType

from .testbed_runner import supports_driver


class IntegratorGUIStateMixin:
    """Handle widget enable/disable state derived from current GUI options."""

    def _on_sim_type_change(self) -> None:
        self._update_driver_visibility()
        self._update_cavity_spacing_state()
        self._update_image_subcharge_state()
        self._update_macroparticle_state()
        self._toggle_macroparticle_smearing_controls()
        self._update_pseudo_grid_state()
        self._update_driver_train_state()
        self._refresh_initial_summary()

        if hasattr(self, "optimization_tab") and self.optimization_tab:
            sim_type_value = self.sim_type_var.get()
            if hasattr(self.optimization_tab, "sim_type_var"):
                self.optimization_tab.sim_type_var.set(sim_type_value)

    def _update_driver_visibility(self) -> None:
        sim_type = SimulationType[self.sim_type_var.get()]
        enabled = supports_driver(sim_type)
        entry_state = "normal" if enabled else "disabled"
        combo_state = "readonly" if enabled else "disabled"
        if hasattr(self, "driver_species_combo"):
            self.driver_species_combo.configure(state=combo_state)
        for entry in getattr(self, "_driver_entries", []):
            entry.configure(state=entry_state)
        if not enabled:
            default_label = self._species_label_by_key.get(
                "custom", next(iter(self._species_by_label))
            )
            self.driver_species_var.set(default_label)

        offsets_enabled = sim_type == SimulationType.BUNCH_TO_BUNCH
        offset_state = "normal" if offsets_enabled else "disabled"

        for entry in getattr(self, "_rider_offset_entries", []):
            entry.configure(state=offset_state)
        for entry in getattr(self, "_driver_offset_entries", []):
            entry.configure(state=offset_state)

        label_color = "black" if offsets_enabled else "gray60"
        for label in getattr(self, "_rider_offset_labels", []):
            label.configure(foreground=label_color)
        for label in getattr(self, "_driver_offset_labels", []):
            label.configure(foreground=label_color)

    def _update_image_subcharge_state(self) -> None:
        sim_type = SimulationType[self.sim_type_var.get()]
        enabled = sim_type != SimulationType.BUNCH_TO_BUNCH
        entry_state = "normal" if enabled else "disabled"
        if hasattr(self, "image_subcharge_entry"):
            self.image_subcharge_entry.configure(state=entry_state)
        if hasattr(self, "image_weighting_check"):
            self.image_weighting_check.configure(state=entry_state)
        if hasattr(self, "core_param_widgets"):
            if "aperture_radius" in self.core_param_widgets:
                self.core_param_widgets["aperture_radius"].configure(state=entry_state)
            if "wall_z" in self.core_param_widgets:
                self.core_param_widgets["wall_z"].configure(state=entry_state)

    def _update_cavity_spacing_state(self) -> None:
        is_switching = self.sim_type_var.get() == "SWITCHING_WALL"
        state = "normal" if is_switching else "disabled"

        if (
            hasattr(self, "core_param_widgets")
            and "cav_spacing" in self.core_param_widgets
        ):
            self.core_param_widgets["cav_spacing"].configure(state=state)

    def _toggle_random_seed(self) -> None:
        random_enabled = self.random_seed_var.get()
        state = "disabled" if random_enabled else "normal"

        if hasattr(self, "seed_entry"):
            self.seed_entry.configure(state=state)

    def _toggle_z_cutoff_controls(self) -> None:
        enabled = self.z_cutoff_enabled_var.get()
        state = "normal" if enabled else "disabled"
        combo_state = "readonly" if enabled else "disabled"

        if hasattr(self, "z_cutoff_entry"):
            self.z_cutoff_entry.configure(state=state)
        if hasattr(self, "z_cutoff_mode_combo"):
            self.z_cutoff_mode_combo.configure(state=combo_state)

        if not enabled:
            self.core_param_vars["z_cutoff"].set(0.0)

    def _toggle_self_consistency_controls(self) -> None:
        if not hasattr(self, "sc_target_ms_tolerance_label"):
            return

        enabled = self.self_consistency_enabled_var.get()
        param_state = "normal" if enabled else "disabled"

        controls_to_toggle = [
            self.sc_mode_label,
            self.sc_mode_combo,
            self.sc_target_ms_tolerance_label,
            self.sc_target_ms_tolerance_entry,
            self.sc_max_iterations_label,
            self.sc_max_iterations_entry,
            self.sc_mass_shell_tolerance_label,
            self.sc_mass_shell_tolerance_entry,
            self.sc_relaxation_label,
            self.sc_relaxation_entry,
            self.sc_verbosity_label,
            self.sc_verbosity_entry,
            self.sc_gamma_reconciliation_method_combo,
            self.sc_gamma_low_beta_threshold_entry,
            self.sc_gamma_high_beta_threshold_entry,
            self.sc_gamma_low_beta_weight_entry,
            self.sc_gamma_high_beta_weight_entry,
            self.sc_gamma_mid_beta_weight_entry,
            self.sc_gamma_fixed_weight_entry,
        ]

        for control in controls_to_toggle:
            if isinstance(control, (ttk.Entry, ttk.Spinbox)):
                control.configure(state=param_state)
            elif isinstance(control, ttk.Combobox):
                control.configure(state="readonly" if enabled else "disabled")
            elif isinstance(control, ttk.Label):
                fg_color = "black" if enabled else "gray"
                control.configure(foreground=fg_color)

        if enabled:
            self._on_sc_mode_changed()
        self._toggle_chrono_controls()

    def _toggle_chrono_controls(self) -> None:
        if not hasattr(self, "sc_chrono_tolerance_label"):
            return

        chrono_enabled = self.chrono_interpolate_var.get()
        enable_chrono_options = chrono_enabled

        param_state = "normal" if enable_chrono_options else "disabled"
        label_color = "black" if enable_chrono_options else "gray"

        chrono_sub_controls = [
            (self.sc_chrono_tolerance_label, "label"),
            (self.sc_chrono_tolerance_entry, "entry"),
            (self.sc_chrono_high_precision_check, "checkbutton"),
            (self.sc_chrono_adaptive_check, "checkbutton"),
        ]

        for control, control_type in chrono_sub_controls:
            if control_type == "entry":
                control.configure(state=param_state)
            elif control_type == "checkbutton":
                control.configure(state=param_state)
            elif control_type == "label":
                control.configure(foreground=label_color)

    def _on_sc_mode_changed(self, event=None):
        """Handle convergence mode changes."""
        return None

    def _toggle_macroparticle_controls(self) -> None:
        enabled = self.macroparticle_enabled_var.get()
        state = "normal" if enabled else "disabled"

        if hasattr(self, "_macroparticle_widgets"):
            for widget in self._macroparticle_widgets:
                if isinstance(widget, ttk.Entry):
                    widget.configure(state=state)
                elif isinstance(widget, ttk.Checkbutton):
                    widget.configure(state=state)
                elif isinstance(widget, ttk.Label):
                    fg_color = "black" if enabled else "gray"
                    widget.configure(foreground=fg_color)

    def _update_macroparticle_state(self) -> None:
        if not hasattr(self, "macroparticle_enable_check"):
            return

        is_conducting_wall = self.sim_type_var.get() == "CONDUCTING_WALL"
        check_state = "normal" if is_conducting_wall else "disabled"
        self.macroparticle_enable_check.configure(state=check_state)

        if not is_conducting_wall:
            self.macroparticle_enabled_var.set(False)
            widget_state = "disabled"
            label_color = "gray"
        else:
            enabled = self.macroparticle_enabled_var.get()
            widget_state = "normal" if enabled else "disabled"
            label_color = "black" if enabled else "gray"

        if hasattr(self, "_macroparticle_widgets"):
            for widget in self._macroparticle_widgets:
                if isinstance(widget, ttk.Entry):
                    widget.configure(state=widget_state)
                elif isinstance(widget, ttk.Checkbutton):
                    widget.configure(state=widget_state)
                elif isinstance(widget, ttk.Label):
                    widget.configure(foreground=label_color)

    def _toggle_macroparticle_smearing_controls(self) -> None:
        if not hasattr(self, "macroparticle_smearing_enabled_var"):
            return

        enabled = bool(self.macroparticle_smearing_enabled_var.get())
        state = "normal" if enabled else "disabled"
        label_color = "black" if enabled else "gray"
        for widget in getattr(self, "_macroparticle_smearing_widgets", []):
            if isinstance(widget, ttk.Entry):
                widget.configure(state=state)
            elif isinstance(widget, ttk.Checkbutton):
                widget.configure(state=state)
            elif isinstance(widget, ttk.Label):
                widget.configure(foreground=label_color)

    def _toggle_pseudo_grid_controls(self) -> None:
        if not hasattr(self, "pseudo_grid_enabled_var"):
            return

        enabled = bool(self.pseudo_grid_enabled_var.get())
        causal_enabled = enabled and bool(
            self.pseudo_grid_causal_history_pruning_enabled_var.get()
        )

        for widget in getattr(self, "_pseudo_grid_widgets", []):
            if isinstance(widget, ttk.Entry):
                widget.configure(state="normal" if enabled else "disabled")
            elif isinstance(widget, ttk.Combobox):
                widget.configure(state="readonly" if enabled else "disabled")
            elif isinstance(widget, ttk.Checkbutton):
                widget.configure(state="normal" if enabled else "disabled")
            elif isinstance(widget, ttk.Label):
                widget.configure(foreground="black" if enabled else "gray")

        for widget in getattr(self, "_pseudo_grid_causal_widgets", []):
            if isinstance(widget, ttk.Entry):
                widget.configure(state="normal" if causal_enabled else "disabled")
            elif isinstance(widget, ttk.Label):
                widget.configure(foreground="black" if causal_enabled else "gray")

    def _update_pseudo_grid_state(self) -> None:
        if not hasattr(self, "pseudo_grid_enable_check"):
            return

        is_bunch_to_bunch = self.sim_type_var.get() == "BUNCH_TO_BUNCH"
        self.pseudo_grid_enable_check.configure(
            state="normal" if is_bunch_to_bunch else "disabled"
        )

        if not is_bunch_to_bunch:
            self.pseudo_grid_enabled_var.set(False)

        self._toggle_pseudo_grid_controls()

    def _toggle_driver_train_controls(self) -> None:
        if not hasattr(self, "driver_train_enabled_var"):
            return

        enabled = bool(self.driver_train_enabled_var.get())
        for widget in getattr(self, "_driver_train_widgets", []):
            if isinstance(widget, ttk.Entry):
                widget.configure(state="normal" if enabled else "disabled")
            elif isinstance(widget, ttk.Checkbutton):
                widget.configure(state="normal" if enabled else "disabled")
            elif isinstance(widget, ttk.Label):
                widget.configure(foreground="black" if enabled else "gray")

    def _update_driver_train_state(self) -> None:
        if not hasattr(self, "driver_train_enable_check"):
            return

        is_bunch_to_bunch = self.sim_type_var.get() == "BUNCH_TO_BUNCH"
        self.driver_train_enable_check.configure(
            state="normal" if is_bunch_to_bunch else "disabled"
        )

        if not is_bunch_to_bunch:
            self.driver_train_enabled_var.set(False)

        self._toggle_driver_train_controls()

    def _toggle_gamma_reconciliation_params(self) -> None:
        if not hasattr(self, "sc_gamma_reconciliation_adaptive_frame"):
            return

        method = self.self_consistency_gamma_reconciliation_method_var.get()

        if method == "ADAPTIVE_WEIGHTED":
            self.sc_gamma_reconciliation_adaptive_frame.grid()
            self.sc_gamma_reconciliation_fixed_frame.grid_remove()
        elif method == "FIXED_WEIGHTED":
            self.sc_gamma_reconciliation_adaptive_frame.grid_remove()
            self.sc_gamma_reconciliation_fixed_frame.grid()
        else:
            self.sc_gamma_reconciliation_adaptive_frame.grid_remove()
            self.sc_gamma_reconciliation_fixed_frame.grid_remove()

    def _toggle_adaptive_timestep_controls(self) -> None:
        if not hasattr(self, "adaptive_threshold_label"):
            return

        adaptive_enabled = self.adaptive_timestep_enabled_var.get()
        param_state = "normal" if adaptive_enabled else "disabled"

        all_controls = [
            self.adaptive_threshold_label,
            self.adaptive_threshold_entry,
            self.adaptive_reduction_label,
            self.adaptive_reduction_entry,
            self.adaptive_max_attempts_label,
            self.adaptive_max_attempts_display,
            self.adaptive_min_factor_label,
            self.adaptive_min_factor_entry,
            self.adaptive_cooldown_label,
            self.adaptive_cooldown_entry,
            self.adaptive_probe_threshold_label,
            self.adaptive_probe_threshold_entry,
            self.adaptive_max_probe_label,
            self.adaptive_max_probe_entry,
            self.adaptive_halt_check,
            self.adaptive_debug_check,
            self.adaptive_bunch_proximity_check,
            self.adaptive_bunch_proximity_sigma_label,
            self.adaptive_bunch_proximity_sigma_entry,
            self.adaptive_bunch_proximity_n_sigma_label,
            self.adaptive_bunch_proximity_n_sigma_entry,
            self.adaptive_bunch_proximity_reduction_label,
            self.adaptive_bunch_proximity_reduction_entry,
            self.adaptive_bunch_proximity_transition_label,
            self.adaptive_bunch_proximity_transition_entry,
            self.adaptive_max_substeps_label,
            self.adaptive_max_substeps_display,
        ]

        for control in all_controls:
            if isinstance(control, (ttk.Entry, ttk.Checkbutton)):
                control.configure(state=param_state)
            elif isinstance(control, ttk.Label):
                fg_color = "black" if adaptive_enabled else "gray"
                control.configure(foreground=fg_color)

    def _toggle_cavity_exit_controls(self) -> None:
        if not hasattr(self, "cavity_exit_enable_check"):
            return
        enabled = bool(self.cavity_exit_enabled_var.get())
        state = "normal" if enabled else "disabled"
        for widget in getattr(self, "_cavity_exit_sub_widgets", []):
            try:
                if isinstance(widget, ttk.Label):
                    widget.configure(foreground="black" if enabled else "gray")
                else:
                    widget.configure(state=state)
            except Exception:
                pass

    def _toggle_auto_duration_controls(self) -> None:
        if not hasattr(self, "auto_duration_enable_check"):
            return

        enabled = self.auto_duration_enabled_var.get()
        sub_state = "normal" if enabled else "disabled"
        core_state = "disabled" if enabled else "normal"

        # Grey / restore the section's own sub-widgets (crossing steps, post factor)
        for widget in getattr(self, "_auto_duration_sub_widgets", []):
            try:
                if isinstance(widget, ttk.Label):
                    widget.configure(foreground="black" if enabled else "gray")
                else:
                    widget.configure(state=sub_state)
            except Exception:
                pass

        # Grey / restore the Steps field in the Core tab
        if hasattr(self, "steps_entry"):
            self.steps_entry.configure(state=core_state)
        if hasattr(self, "steps_auto_hint"):
            if enabled:
                self.steps_auto_hint.grid()
            else:
                self.steps_auto_hint.grid_remove()

        # Grey / restore the Time step field in the Core tab
        if (
            hasattr(self, "core_param_widgets")
            and "time_step" in self.core_param_widgets
        ):
            self.core_param_widgets["time_step"].configure(state=core_state)
        if hasattr(self, "time_step_auto_hint"):
            if enabled:
                self.time_step_auto_hint.grid()
            else:
                self.time_step_auto_hint.grid_remove()

    def _on_trajectory_save_toggled(self) -> None:
        if not hasattr(self, "trajectory_stride_entry"):
            return

        save_enabled = self.trajectory_save_var.get()
        widget_state = "normal" if save_enabled else "disabled"
        label_color = "black" if save_enabled else "gray"

        self.trajectory_stride_entry.configure(state=widget_state)
        self.trajectory_stride_label.configure(foreground=label_color)
