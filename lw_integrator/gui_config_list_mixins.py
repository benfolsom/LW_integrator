"""Config list, directory, and sweep-config browser helpers for the main GUI."""

from __future__ import annotations

import os
from pathlib import Path
from tkinter import filedialog, messagebox

import tkinter as tk

from .testbed_runner import list_config_files


class IntegratorGUIConfigListMixin:
    """Manage config lists, directory pickers, and sweep-config routing."""

    def _refresh_config_list(self, selected: str | None = None) -> None:
        configs = list_config_files(Path(self.config_dir_var.get()))
        self.config_list.delete(0, tk.END)
        self.config_list.selection_clear(0, tk.END)

        highlight = None
        if selected and selected in configs:
            highlight = configs.index(selected)
        elif self.config_file_var.get() in configs:
            highlight = configs.index(self.config_file_var.get())

        for config_name in configs:
            self.config_list.insert(tk.END, config_name)

        if highlight is not None:
            self.config_list.selection_set(highlight)
            self.config_list.see(highlight)

    def _refresh_sweep_config_list(self, selected: str | None = None) -> None:
        self.sweep_config_list.delete(0, tk.END)
        sweep_dir = self.sweep_config_dir_var.get()

        highlight = None

        if os.path.isdir(sweep_dir):
            configs = [f for f in os.listdir(sweep_dir) if f.endswith(".json")]
            configs.sort()

            if selected and selected in configs:
                highlight = configs.index(selected)

            for config_name in configs:
                self.sweep_config_list.insert(tk.END, config_name)

        if highlight is not None:
            self.sweep_config_list.selection_set(highlight)
            self.sweep_config_list.see(highlight)

    def _selected_config_filename(self) -> str | None:
        selection = self.config_list.curselection()
        if not selection:
            return None
        result = self.config_list.get(selection[0])
        return str(result) if result else None

    def _on_config_selected(self) -> None:
        filename = self._selected_config_filename()
        if filename:
            self.config_name_var.set(filename)
            self.config_file_var.set(filename)
            self.current_config_label.config(text=filename, foreground="black")
        else:
            self.current_config_label.config(text="<none>", foreground="gray")

    def _on_sweep_config_selected(self) -> None:
        selection = self.sweep_config_list.curselection()
        if selection:
            filename = self.sweep_config_list.get(selection[0])
            self.sweep_config_name_var.set(filename)
            self.current_sweep_config_label.config(
                text=filename, foreground="black", font=("TkDefaultFont", 9)
            )
        else:
            self.current_sweep_config_label.config(
                text="<none>", foreground="gray", font=("TkDefaultFont", 9, "italic")
            )

    def _select_config_dir(self) -> None:
        initial_dir = self.config_dir_var.get()
        if not os.path.exists(initial_dir):
            initial_dir = self._default_config_dir

        directory = filedialog.askdirectory(
            title="Select config directory", initialdir=initial_dir
        )
        if directory:
            self.config_dir_var.set(directory)
            self._last_config_dir = directory
            self._save_preferences()
            self._refresh_config_list()

    def _select_output_dir(self) -> None:
        initial_dir = self.output_dir_var.get()
        if not os.path.exists(initial_dir):
            initial_dir = self._default_output_dir

        directory = filedialog.askdirectory(
            title="Select output directory", initialdir=initial_dir
        )
        if directory:
            self.output_dir_var.set(directory)
            self._last_output_dir = directory
            self._save_preferences()

    def _select_sweep_config_dir(self) -> None:
        initial_dir = self.sweep_config_dir_var.get()
        if not os.path.exists(initial_dir):
            initial_dir = self._default_sweep_config_dir

        directory = filedialog.askdirectory(
            title="Select sweep config directory", initialdir=initial_dir
        )
        if directory:
            self.sweep_config_dir_var.set(directory)
            self._last_sweep_config_dir = directory
            self._save_preferences()
            self._refresh_sweep_config_list()
            if hasattr(self, "optimization_tab"):
                self.optimization_tab.sweep_config_dir = directory

    def _select_sweep_output_dir(self) -> None:
        initial_dir = self.sweep_output_dir_var.get()
        if not os.path.exists(initial_dir):
            initial_dir = self._default_sweep_output_dir

        directory = filedialog.askdirectory(
            title="Select sweep output directory", initialdir=initial_dir
        )
        if directory:
            self.sweep_output_dir_var.set(directory)
            self._last_sweep_output_dir = directory
            self._save_preferences()
            if hasattr(self, "optimization_tab"):
                self.optimization_tab.sweep_output_dir = directory

    def _load_sweep_config(self) -> None:
        filename = self.sweep_config_name_var.get().strip()

        if not filename:
            selection = self.sweep_config_list.curselection()
            if not selection:
                messagebox.showinfo(
                    "Load Sweep Config",
                    "Enter a config name or select one from the list.",
                )
                return
            filename = self.sweep_config_list.get(selection[0])

        if not filename.endswith(".json"):
            filename += ".json"

        if not hasattr(self, "optimization_tab"):
            return

        sweep_config_dir = self.sweep_config_dir_var.get()
        path = os.path.join(sweep_config_dir, filename)

        if not os.path.exists(path):
            messagebox.showerror(
                "Load Sweep Config", f"Configuration file not found: {filename}"
            )
            return

        self.optimization_tab._load_config_from_path(path)
        self.sweep_config_name_var.set(filename)
        self.current_sweep_config_label.config(
            text=filename, foreground="black", font=("TkDefaultFont", 9)
        )

    def _save_sweep_config(self) -> None:
        if not hasattr(self, "optimization_tab"):
            return

        filename = self.sweep_config_name_var.get().strip()

        if not filename:
            messagebox.showinfo("Save Sweep Config", "Enter a config name to save.")
            return

        if not filename.endswith(".json"):
            filename += ".json"

        sweep_config_dir = self.sweep_config_dir_var.get()
        os.makedirs(sweep_config_dir, exist_ok=True)

        filepath = os.path.join(sweep_config_dir, filename)

        if not self._check_override_warning(Path(filepath), "sweep"):
            return

        success = self.optimization_tab._save_config_to_path(filepath)

        if success:
            self.sweep_config_name_var.set(filename)
            self.current_sweep_config_label.config(
                text=filename, foreground="black", font=("TkDefaultFont", 9)
            )
            self._refresh_sweep_config_list(selected=filename)
            messagebox.showinfo(
                "Save Sweep Config", f"Configuration saved as {filename}"
            )
