"""Form-state helpers for the optimization plugin."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from optimization.sweep_helpers import calculate_starting_pz_from_energy


class OptimizationPluginFormMixin:
    """Shared particle/sweep form helpers for the optimization plugin."""

    _LINKED_DISABLED_ENTRY_STYLE = "LinkedDriverEnergyDisabled.TEntry"

    def _add_sweepable_param(
        self, parent, row, param_name, label, default_value, width=15
    ):
        """Add a parameter row with optional sweep controls."""
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=2)

        var = tk.StringVar(value=default_value)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="w", pady=2, padx=5)

        sweep_var = tk.BooleanVar(value=False)
        sweep_cb = ttk.Checkbutton(
            parent,
            text="Sweep:",
            variable=sweep_var,
            command=lambda: self._toggle_sweep_controls(param_name),
        )
        sweep_cb.grid(row=row, column=2, sticky="w", pady=2, padx=(10, 2))

        range_frame = ttk.Frame(parent)
        range_frame.grid(row=row, column=3, columnspan=3, sticky="w", pady=2)
        range_frame.grid_remove()

        ttk.Label(range_frame, text="Min:").pack(side="left", padx=(0, 2))
        min_var = tk.StringVar(value=default_value)
        ttk.Entry(range_frame, textvariable=min_var, width=10).pack(side="left", padx=2)

        ttk.Label(range_frame, text="Max:").pack(side="left", padx=(5, 2))
        max_var = tk.StringVar(value=default_value)
        ttk.Entry(range_frame, textvariable=max_var, width=10).pack(side="left", padx=2)

        ttk.Label(range_frame, text="Pts:").pack(side="left", padx=(5, 2))
        points_var = tk.StringVar(value="3")
        ttk.Entry(range_frame, textvariable=points_var, width=4).pack(
            side="left", padx=2
        )

        log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(range_frame, text="Log", variable=log_var).pack(
            side="left", padx=(5, 0)
        )

        self.sweep_params[param_name] = {
            "label_text": label,
            "label_widget": label_widget,
            "fixed_var": var,
            "fixed_entry": entry,
            "sweep_var": sweep_var,
            "range_frame": range_frame,
            "min_var": min_var,
            "max_var": max_var,
            "points_var": points_var,
            "log_var": log_var,
        }

    def _toggle_sweep_controls(self, param_name):
        """Show/hide sweep range controls based on checkbox state."""
        controls = self.sweep_params[param_name]
        if controls["sweep_var"].get():
            controls["range_frame"].grid()
            controls["fixed_entry"].config(state="disabled")
        else:
            controls["range_frame"].grid_remove()
            controls["fixed_entry"].config(state="normal")

    def _on_link_energy_toggled(self):
        """Handle toggling of the 'Link to Rider Energy' checkbox."""
        linked = self.link_driver_rider_energy_var.get()
        driver_energy_controls = self.sweep_params["driver_energy_gev"]

        if linked:
            driver_energy_controls["sweep_var"].set(False)
            self._toggle_sweep_controls("driver_energy_gev")
            self._set_driver_energy_entry_linked_state(True)

            for widget in self.driver_frame.winfo_children():
                if isinstance(widget, ttk.Checkbutton):
                    try:
                        if widget.cget("variable") == str(
                            driver_energy_controls["sweep_var"]
                        ):
                            widget.config(state="disabled")
                    except Exception:
                        pass

            self._update_linked_energy_presentation()
            self._update_driver_pz_helper()
        else:
            self._set_driver_energy_entry_linked_state(False)

            for widget in self.driver_frame.winfo_children():
                if isinstance(widget, ttk.Checkbutton):
                    try:
                        if widget.cget("variable") == str(
                            driver_energy_controls["sweep_var"]
                        ):
                            widget.config(state="normal")
                    except Exception:
                        pass

            self._update_linked_energy_presentation()
            self._update_driver_pz_helper()

    def _set_driver_energy_entry_linked_state(self, linked: bool) -> None:
        """Apply a visibly grayed-out disabled style to driver energy in linked mode."""
        controls = self.sweep_params.get("driver_energy_gev")
        if not controls:
            return

        entry = controls["fixed_entry"]
        if linked:
            self._ensure_linked_disabled_entry_style()
            entry.config(style=self._LINKED_DISABLED_ENTRY_STYLE, state="disabled")
        else:
            entry.config(style="TEntry", state="normal")

    def _ensure_linked_disabled_entry_style(self) -> None:
        """Create the linked-mode disabled style once."""
        if getattr(self, "_linked_disabled_entry_style_ready", False):
            return

        style = ttk.Style()
        style.configure(self._LINKED_DISABLED_ENTRY_STYLE, foreground="#7a7a7a")
        style.map(
            self._LINKED_DISABLED_ENTRY_STYLE,
            foreground=[("disabled", "#7a7a7a")],
            fieldbackground=[("disabled", "#f0f0f0")],
        )
        self._linked_disabled_entry_style_ready = True

    def _update_driver_energy_link_state(self):
        """Update driver energy controls based on link state (called during load)."""
        if hasattr(self, "link_driver_rider_energy_var"):
            self._on_link_energy_toggled()

    def _update_linked_energy_presentation(self):
        """Update driver-energy label/help text to reflect linked rider sweep mode."""
        if not hasattr(self, "link_driver_rider_energy_var"):
            return

        controls = self.sweep_params.get("driver_energy_gev")
        if not controls:
            return

        label_widget = controls.get("label_widget")
        base_label = controls.get("label_text", "Kinetic Energy (GeV):")
        linked = self.link_driver_rider_energy_var.get()

        if linked:
            if label_widget is not None:
                label_widget.config(text="Kinetic Energy (GeV, linked):")

            try:
                energy_min = float(self.energy_min_var.get())
                energy_max = float(self.energy_max_var.get())
                energy_points = int(self.energy_points_var.get())
                self.link_energy_help_label.config(
                    text=(
                        f"(Driver follows rider sweep: {energy_min:g} to "
                        f"{energy_max:g} GeV, {energy_points} pts)"
                    )
                )
            except (ValueError, TypeError):
                self.link_energy_help_label.config(
                    text="(Driver energy = Rider energy for each sweep point)"
                )
        else:
            if label_widget is not None:
                label_widget.config(text=base_label)
            self.link_energy_help_label.config(text="")

    def _update_rider_pz_helper(self):
        """Update helper text showing rider starting Pz calculated from energy."""
        try:
            mass_str = self.sweep_params["rider_m_particle"]["fixed_var"].get()
            mass_amu = float(mass_str) if mass_str else 0.00054857990907
            energy_min = float(self.energy_min_var.get())
            energy_max = float(self.energy_max_var.get())
            pz_min = calculate_starting_pz_from_energy(energy_min, mass_amu)
            pz_max = calculate_starting_pz_from_energy(energy_max, mass_amu)
            self.rider_pz_helper_var.set(
                f"→ Starting Pz range: [{pz_min:.2f}, {pz_max:.2f}] amu·mm/ns"
            )
        except (ValueError, ZeroDivisionError):
            self.rider_pz_helper_var.set("")

    def _update_driver_pz_helper(self):
        """Update helper text showing driver starting Pz calculated from energy."""
        try:
            mass_str = self.sweep_params["driver_m_particle"]["fixed_var"].get()
            mass_amu = float(mass_str) if mass_str else 207.2
            negative = (
                getattr(self, "driver_direction_var", None) is None
                or getattr(self, "driver_direction_var").get() == "-z"
            )
            sign_label = "−ẑ" if negative else "+ẑ"

            if (
                getattr(self, "link_driver_rider_energy_var", None)
                and self.link_driver_rider_energy_var.get()
            ):
                energy_min = float(self.energy_min_var.get())
                energy_max = float(self.energy_max_var.get())
                pz_min = calculate_starting_pz_from_energy(
                    energy_min, mass_amu, negative=negative
                )
                pz_max = calculate_starting_pz_from_energy(
                    energy_max, mass_amu, negative=negative
                )
                self.driver_pz_helper_var.set(
                    f"→ [LINKED] Pz range: [{pz_min:.2f}, {pz_max:.2f}] amu·mm/ns ({sign_label})"
                )
            elif self.sweep_params["driver_energy_gev"]["sweep_var"].get():
                energy_min = abs(
                    float(self.sweep_params["driver_energy_gev"]["min_var"].get())
                )
                energy_max = abs(
                    float(self.sweep_params["driver_energy_gev"]["max_var"].get())
                )
                pz_min = calculate_starting_pz_from_energy(
                    energy_min, mass_amu, negative=negative
                )
                pz_max = calculate_starting_pz_from_energy(
                    energy_max, mass_amu, negative=negative
                )
                self.driver_pz_helper_var.set(
                    f"→ Pz range: [{pz_min:.2f}, {pz_max:.2f}] amu·mm/ns ({sign_label})"
                )
            else:
                energy_gev = abs(
                    float(self.sweep_params["driver_energy_gev"]["fixed_var"].get())
                )
                pz = calculate_starting_pz_from_energy(
                    energy_gev, mass_amu, negative=negative
                )
                self.driver_pz_helper_var.set(
                    f"→ Pz = {pz:.2f} amu·mm/ns ({sign_label})"
                )
        except (ValueError, ZeroDivisionError):
            self.driver_pz_helper_var.set("")

    def _update_energy_label(self):
        """Update energy label to clarify it's rider-only in BUNCH_TO_BUNCH mode."""
        sim_type = self.sim_type_var.get()
        if sim_type == "BUNCH_TO_BUNCH":
            self.energy_label.config(text="Rider Particle Energy:")
        else:
            self.energy_label.config(text="Particle Energy:")

    def _toggle_wall_z_sweep(self):
        """Toggle wall_z sweep controls."""
        if self.wall_z_sweep_var.get():
            self.wall_z_entry.config(state="disabled")
            for widget in self.wall_z_sweep_widgets:
                widget.config(state="normal")
        else:
            self.wall_z_entry.config(state="normal")
            for widget in self.wall_z_sweep_widgets:
                widget.config(state="disabled")

    def _toggle_timestep_mode(self):
        """Toggle between duration/count auto-calculation modes."""
        mode = self.timestep_mode_var.get()
        if mode == "duration":
            self.steps_entry.config(state="normal")
            self.duration_entry.config(state="disabled")
        else:
            self.steps_entry.config(state="disabled")
            self.duration_entry.config(state="normal")

    def _update_driver_visibility(self):
        """Show/hide driver section based on simulation type."""
        if not hasattr(self, "driver_frame"):
            return

        sim_type = self.sim_type_var.get()
        if sim_type == "BUNCH_TO_BUNCH":
            self.driver_frame.pack(
                fill="x",
                padx=10,
                pady=5,
                after=self.driver_frame.master.winfo_children()[2],
            )
            self._update_driver_pz_helper()
        else:
            self.driver_frame.pack_forget()

        self._update_energy_label()
        self._update_rider_pz_helper()

    def _on_sim_type_changed(self):
        """Handle simulation type change."""
        self._update_driver_visibility()
        self._update_macroparticle_state()
        self._update_parameter_visibility()

        if self.gui_controller and hasattr(self.gui_controller, "sim_type_var"):
            sim_type_value = self.sim_type_var.get()
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

    def _toggle_macroparticle_controls(self):
        """Enable/disable macroparticle controls based on checkbox state."""
        if not hasattr(self, "_macroparticle_widgets"):
            return

        enabled = self.macroparticle_enabled_var.get()
        state = "normal" if enabled else "disabled"

        for widget in self._macroparticle_widgets:
            if isinstance(widget, ttk.Entry):
                widget.configure(state=state)
            elif isinstance(widget, ttk.Checkbutton):
                widget.configure(state=state)
            elif isinstance(widget, ttk.Label):
                fg_color = "black" if enabled else "gray"
                widget.configure(foreground=fg_color)

    def _update_macroparticle_state(self):
        """Enable/disable macroparticle controls based on simulation type."""
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

        if hasattr(self, "_macroparticle_sweep_controls"):
            for controls in self._macroparticle_sweep_controls:
                if "range_frame" in controls:
                    for child in controls["range_frame"].winfo_children():
                        if isinstance(child, ttk.Entry):
                            child.configure(state=widget_state)
        self._update_parameter_visibility()

    def _set_frame_state(self, frame, state):
        """Recursively set state for all widgets in a frame."""
        if frame is None:
            return

        label_color = "black" if state == "normal" else "gray"

        for child in frame.winfo_children():
            widget_type = child.winfo_class()
            try:
                if widget_type in ("TEntry", "Entry"):
                    child.configure(state=state)
                elif widget_type in ("TCheckbutton", "Checkbutton"):
                    child.configure(state=state)
                elif widget_type in ("TLabel", "Label"):
                    child.configure(foreground=label_color)
                elif widget_type in ("TFrame", "Frame"):
                    self._set_frame_state(child, state)
            except Exception:
                pass

    def _update_parameter_visibility(self):
        """Update parameter field states based on simulation type."""
        if not hasattr(self, "cavity_spacing_entry"):
            return

        sim_type = self.sim_type_var.get()
        is_bunch_to_bunch = sim_type == "BUNCH_TO_BUNCH"

        if sim_type == "SWITCHING_WALL":
            self.cavity_spacing_entry.config(state="normal")
            if "cavity_spacing_label" in self._param_widgets:
                self._param_widgets["cavity_spacing_label"].config(foreground="black")
            if "cavity_spacing_desc_label" in self._param_widgets:
                self._param_widgets["cavity_spacing_desc_label"].config(
                    foreground="gray40"
                )
        else:
            self.cavity_spacing_entry.config(state="disabled")
            if "cavity_spacing_label" in self._param_widgets:
                self._param_widgets["cavity_spacing_label"].config(foreground="gray")
            if "cavity_spacing_desc_label" in self._param_widgets:
                self._param_widgets["cavity_spacing_desc_label"].config(
                    foreground="gray"
                )

        if is_bunch_to_bunch:
            self._set_frame_state(self._param_widgets.get("aperture_frame"), "disabled")
            if "aperture_label" in self._param_widgets:
                self._param_widgets["aperture_label"].config(foreground="gray")

            self._set_frame_state(
                self._param_widgets.get("wall_z_fixed_frame"), "disabled"
            )
            self._set_frame_state(
                self._param_widgets.get("wall_z_sweep_frame"), "disabled"
            )
            if "wall_z_label" in self._param_widgets:
                self._param_widgets["wall_z_label"].config(foreground="gray")
        else:
            self._set_frame_state(self._param_widgets.get("aperture_frame"), "normal")
            if "aperture_label" in self._param_widgets:
                self._param_widgets["aperture_label"].config(foreground="black")

            self._set_frame_state(
                self._param_widgets.get("wall_z_fixed_frame"), "normal"
            )
            if "wall_z_label" in self._param_widgets:
                self._param_widgets["wall_z_label"].config(foreground="black")
            self._toggle_wall_z_sweep()

        if "offset_label" in self._param_widgets:
            self._param_widgets["offset_label"].config(foreground="black")
        if "offset_entry" in self._param_widgets:
            self._param_widgets["offset_entry"].config(state="normal")
        if "offset_desc_label" in self._param_widgets:
            self._param_widgets["offset_desc_label"].config(foreground="gray40")

        driver_offset_state = "normal" if is_bunch_to_bunch else "disabled"
        driver_offset_color = "black" if is_bunch_to_bunch else "gray"
        if "driver_offset_label" in self._param_widgets:
            self._param_widgets["driver_offset_label"].config(
                foreground=driver_offset_color
            )
        if "driver_offset_entry" in self._param_widgets:
            self._param_widgets["driver_offset_entry"].config(state=driver_offset_state)
        if "driver_offset_desc_label" in self._param_widgets:
            self._param_widgets["driver_offset_desc_label"].config(
                foreground="gray40" if is_bunch_to_bunch else "gray"
            )

        self._update_timestep_tooltip()
        self._update_distance_target_labels()

    def _update_timestep_tooltip(self):
        """Update timestep calculation tooltip based on simulation type."""
        if not hasattr(self, "timestep_calc_label"):
            return

        sim_type = self.sim_type_var.get()

        if sim_type == "BUNCH_TO_BUNCH":
            tooltip_text = (
                "BUNCH_TO_BUNCH Mode:\n"
                "• Rider travels to: driver_start_position + distance_target\n"
                "• This ensures rider reaches and passes driver interaction point\n"
                "• Step duration auto-calculated to reach target in specified steps\n"
                "• Or step count auto-calculated for specified duration"
            )
        else:
            tooltip_text = (
                "CONDUCTING_WALL / SWITCHING_WALL Mode:\n"
                "• Particle travels to: wall_z + distance_target\n"
                "• Ensures consistent trajectory length across energies\n"
                "• Step duration auto-calculated to reach target in specified steps\n"
                "• Or step count auto-calculated for specified duration"
            )

        self._add_tooltip(self.timestep_calc_label, tooltip_text)

    def _update_distance_target_labels(self):
        """Update distance target label text based on simulation type."""
        if not hasattr(self, "distance_target_prefix_label"):
            return

        sim_type = self.sim_type_var.get()

        if sim_type == "BUNCH_TO_BUNCH":
            self.distance_target_prefix_label.config(text="Extra distance:")
            self.distance_target_suffix_label.config(text="mm past driver_start")
        else:
            self.distance_target_prefix_label.config(text="Target: wall +")
            self.distance_target_suffix_label.config(
                text="mm (min 5% of steps enforced)"
            )
