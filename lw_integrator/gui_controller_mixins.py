"""Small controller helpers for the main GUI."""

from __future__ import annotations

from tkinter import messagebox

from .testbed_runner import PARTICLE_PARAM_FIELDS, apply_species_preset


class IntegratorGUIControllerMixin:
    """Handle lightweight controller actions that are not layout-specific."""

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _apply_species(self, target: str) -> None:
        label = (
            self.rider_species_var.get()
            if target == "rider"
            else self.driver_species_var.get()
        )
        preset_key = self._species_by_label.get(label, "custom")
        if preset_key == "custom":
            return
        var_map = self.rider_param_vars if target == "rider" else self.driver_param_vars
        params = {field: var_map[field].get() for field in PARTICLE_PARAM_FIELDS}
        apply_species_preset(params, preset_key)
        for field, value in params.items():
            var_map[field].set(value)

        magnetic_label_by_key = getattr(self, "_magnetic_species_label_by_key", {})
        magnetic_label = magnetic_label_by_key.get(preset_key)
        magnetic_var = getattr(self, f"{target}_magnetic_species_var", None)
        if magnetic_label is not None and magnetic_var is not None:
            magnetic_var.set(magnetic_label)
        self._refresh_initial_summary()

    def _on_tab_changed(self, event=None) -> None:
        self._refresh_initial_summary()

    def _open_optimization_tab(self) -> None:
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "Sweep/Optim":
                self.notebook.select(i)
                break

    def _trigger_sweep(self) -> None:
        if (
            not hasattr(self.optimization_tab, "last_loaded_config")
            or not self.optimization_tab.last_loaded_config
        ):
            response = messagebox.askyesno(
                "No Configuration",
                "No sweep configuration has been loaded or saved.\n\n"
                "It is recommended to save your sweep configuration first.\n\n"
                "Continue anyway?",
                icon="warning",
            )
            if not response:
                return

        if hasattr(self.optimization_tab, "_on_run_sweep"):
            self.optimization_tab._on_run_sweep()
        else:
            messagebox.showerror(
                "Error",
                "Optimization plugin not properly initialized.",
            )
