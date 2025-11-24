"""Tkinter front-end exposing the full integrator testbed experience.

This window mirrors the functionality of ``examples/validation/integrator_testbed.ipynb``
so users can configure particle parameters, manage JSON configs, export
figures, and review logs without relying on Jupyter.  Simulation work runs in a
background thread to keep the UI responsive; any requested figures are rendered
in dedicated top-level windows using Matplotlib's TkAgg backend.
"""

from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List, Optional, Tuple

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from examples.validation.core_vs_legacy_benchmark import (  # type: ignore[import]
    DEFAULT_DRIVER_PARAMS,
    DEFAULT_RIDER_PARAMS,
    SimulationType,
)

from .testbed_runner import (
    AVAILABLE_DPI_CHOICES,
    CORE_PARAM_DEFAULTS,
    CORE_PARAM_LABELS,
    PARAM_LABELS,
    PARTICLE_PARAM_FIELDS,
    SPECIES_OPTIONS,
    InitialSummary,
    RunResult,
    SimulationOptions,
    apply_species_preset,
    compute_initial_summary,
    ensure_directory,
    list_config_files,
    load_config,
    run_testbed,
    save_config,
    supports_driver,
)

DISPLAY_MAX_WIDTH = 1600  # pixels
DISPLAY_MAX_HEIGHT = 900  # pixels


@dataclass
class _FigureHandle:
    name: str
    figure: object
    window: tk.Toplevel
    canvas: FigureCanvasTkAgg


class IntegratorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LW Integrator Testbed")

        self.options = SimulationOptions()
        self._figure_windows: List[_FigureHandle] = []
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._cancel_requested = False

        self._init_variables()
        self._build_layout()
        self._apply_options_to_ui(self.options)
        self._refresh_config_list()
        self._refresh_initial_summary()
        self._update_legacy_state()
        self._update_driver_visibility()

    # ------------------------------------------------------------------
    # Variable initialisation
    # ------------------------------------------------------------------

    def _init_variables(self) -> None:
        self.sim_type_var = tk.StringVar(value=self.options.simulation_type.name)
        self.steps_var = tk.IntVar(value=self.options.steps)
        self.seed_var = tk.IntVar(value=self.options.seed)
        self.legacy_var = tk.BooleanVar(value=self.options.legacy_enabled)

        self._species_by_label = {label: key for label, key in SPECIES_OPTIONS}
        self._species_label_by_key = {key: label for label, key in SPECIES_OPTIONS}
        default_species_label = self._species_label_by_key.get(
            "custom", next(iter(self._species_by_label))
        )
        self.rider_species_var = tk.StringVar(value=default_species_label)
        self.driver_species_var = tk.StringVar(value=default_species_label)

        self.rider_param_vars: Dict[str, tk.Variable] = {}
        self.driver_param_vars: Dict[str, tk.Variable] = {}
        for name, default in DEFAULT_RIDER_PARAMS.items():
            var: tk.Variable
            if isinstance(default, int):
                var = tk.IntVar(value=int(default))
            else:
                var = tk.DoubleVar(value=float(default))
            self.rider_param_vars[name] = var
        for name, default in DEFAULT_DRIVER_PARAMS.items():
            var = (
                tk.IntVar(value=int(default))
                if isinstance(default, int)
                else tk.DoubleVar(value=float(default))
            )
            self.driver_param_vars[name] = var

        self.core_param_vars: Dict[str, tk.Variable] = {
            name: tk.DoubleVar(value=float(value))
            for name, value in CORE_PARAM_DEFAULTS.items()
        }

        self.overlay_display_var = tk.BooleanVar(value=self.options.overlay_display)
        self.overlay_save_var = tk.BooleanVar(value=self.options.overlay_save)
        self.difference_display_var = tk.BooleanVar(
            value=self.options.difference_display
        )
        self.difference_save_var = tk.BooleanVar(value=self.options.difference_save)
        self.metrics_save_var = tk.BooleanVar(value=self.options.metrics_save)
        self.energy_display_var = tk.BooleanVar(value=self.options.energy_display)
        self.energy_save_var = tk.BooleanVar(value=self.options.energy_save)
        self.transverse_display_var = tk.BooleanVar(
            value=self.options.transverse_display
        )
        self.transverse_save_var = tk.BooleanVar(value=self.options.transverse_save)
        self.trajectory_save_var = tk.BooleanVar(value=self.options.trajectory_save)
        self.trajectory_interval_var = tk.IntVar(value=self.options.trajectory_interval)
        self.dpi_var = tk.IntVar(value=self.options.plot_dpi)
        self.image_subcharge_var = tk.IntVar(value=self.options.image_subcharge_count)
        self.image_weighting_var = tk.BooleanVar(value=self.options.use_image_weighting)

        self.output_dir_var = tk.StringVar(value=str(self.options.output_dir))
        self.config_dir_var = tk.StringVar(value=str(self.options.config_dir))
        self.config_name_var = tk.StringVar(value=self.options.config_name)
        self.config_file_var = tk.StringVar(value="")

        self.status_var = tk.StringVar(value="Idle")
        self.summary_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.sim_type_var.trace_add("write", lambda *_: self._on_sim_type_change())
        self.legacy_var.trace_add("write", lambda *_: self._update_legacy_state())

        for var in [self.seed_var, self.rider_species_var, self.driver_species_var]:
            var.trace_add("write", lambda *_: self._refresh_initial_summary())
        for name in PARTICLE_PARAM_FIELDS:
            self.rider_param_vars[name].trace_add(
                "write", lambda *_: self._refresh_initial_summary()
            )
            self.driver_param_vars[name].trace_add(
                "write", lambda *_: self._refresh_initial_summary()
            )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        header = ttk.Frame(self.root, padding=8)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        self._driver_entries: List[ttk.Entry] = []

        ttk.Label(header, text="Simulation type:").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        sim_type = ttk.Combobox(
            header,
            textvariable=self.sim_type_var,
            state="readonly",
            values=[opt.name for opt in SimulationType],
        )
        sim_type.grid(row=0, column=1, sticky="ew")

        ttk.Label(header, text="Steps:").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Entry(header, textvariable=self.steps_var, width=8).grid(
            row=0, column=3, sticky="w"
        )

        ttk.Label(header, text="Seed:").grid(row=0, column=4, sticky="w", padx=(12, 0))
        ttk.Entry(header, textvariable=self.seed_var, width=8).grid(
            row=0, column=5, sticky="w"
        )

        ttk.Checkbutton(
            header, text="Enable legacy comparison", variable=self.legacy_var
        ).grid(row=0, column=6, sticky="w", padx=(12, 0))

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, sticky="nsew")

        # Particles tab --------------------------------------------------
        particle_frame = ttk.Frame(notebook, padding=12)
        notebook.add(particle_frame, text="Particles")
        particle_frame.columnconfigure(1, weight=1)
        particle_frame.columnconfigure(3, weight=1)

        ttk.Label(particle_frame, text="Rider species preset:").grid(
            row=0, column=0, sticky="w"
        )
        rider_combo = ttk.Combobox(
            particle_frame,
            textvariable=self.rider_species_var,
            values=[label for label, _ in SPECIES_OPTIONS],
            state="readonly",
        )
        rider_combo.grid(row=0, column=1, sticky="ew")
        rider_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._apply_species("rider")
        )

        ttk.Label(particle_frame, text="Driver species preset:").grid(
            row=0, column=2, sticky="w", padx=(12, 0)
        )
        driver_combo = ttk.Combobox(
            particle_frame,
            textvariable=self.driver_species_var,
            values=[label for label, _ in SPECIES_OPTIONS],
            state="readonly",
        )
        driver_combo.grid(row=0, column=3, sticky="ew")
        driver_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._apply_species("driver")
        )
        self.driver_species_combo = driver_combo

        for row, name in enumerate(PARTICLE_PARAM_FIELDS, start=1):
            ttk.Label(particle_frame, text=PARAM_LABELS[name] + ":").grid(
                row=row, column=0, sticky="w", pady=2
            )
            ttk.Entry(
                particle_frame, textvariable=self.rider_param_vars[name], width=12
            ).grid(row=row, column=1, sticky="ew", pady=2)
            ttk.Label(particle_frame, text=PARAM_LABELS[name] + " (driver):").grid(
                row=row, column=2, sticky="w", pady=2, padx=(12, 0)
            )
            driver_entry = ttk.Entry(
                particle_frame, textvariable=self.driver_param_vars[name], width=12
            )
            driver_entry.grid(row=row, column=3, sticky="ew", pady=2)
            self._driver_entries.append(driver_entry)

        # Image subcharge controls
        next_row = len(PARTICLE_PARAM_FIELDS) + 1
        ttk.Label(particle_frame, text="Image subcharge count:").grid(
            row=next_row, column=0, sticky="w", pady=(12, 2)
        )
        ttk.Entry(particle_frame, textvariable=self.image_subcharge_var, width=12).grid(
            row=next_row, column=1, sticky="ew", pady=(12, 2)
        )
        ttk.Checkbutton(
            particle_frame,
            text="Enable image weighting",
            variable=self.image_weighting_var,
        ).grid(row=next_row + 1, column=0, columnspan=2, sticky="w", pady=2)

        # Core tab ------------------------------------------------------
        core_frame = ttk.Frame(notebook, padding=12)
        notebook.add(core_frame, text="Core params")
        core_frame.columnconfigure(1, weight=1)

        for row, name in enumerate(CORE_PARAM_LABELS, start=0):
            ttk.Label(core_frame, text=CORE_PARAM_LABELS[name] + ":").grid(
                row=row, column=0, sticky="w", pady=2
            )
            ttk.Entry(
                core_frame, textvariable=self.core_param_vars[name], width=16
            ).grid(row=row, column=1, sticky="ew", pady=2)

        # Outputs tab ---------------------------------------------------
        output_frame = ttk.Frame(notebook, padding=12)
        notebook.add(output_frame, text="Outputs")
        output_frame.columnconfigure(1, weight=1)

        # Trajectory comparison outputs (grouped and dependent on legacy)
        comparison_frame = ttk.LabelFrame(
            output_frame, text="Trajectory Comparison (requires legacy)", padding=8
        )
        comparison_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        comparison_frame.columnconfigure(1, weight=1)

        self._add_output_toggle(
            comparison_frame,
            "Overlay plot",
            self.overlay_display_var,
            self.overlay_save_var,
            row=0,
        )
        self._add_output_toggle(
            comparison_frame,
            "Difference plot",
            self.difference_display_var,
            self.difference_save_var,
            row=1,
        )
        ttk.Checkbutton(
            comparison_frame, text="Save metrics JSON", variable=self.metrics_save_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self._comparison_frame = comparison_frame

        # Other outputs
        self._add_output_toggle(
            output_frame,
            "Energy plot",
            self.energy_display_var,
            self.energy_save_var,
            row=1,
        )
        self._add_output_toggle(
            output_frame,
            "Transverse plot",
            self.transverse_display_var,
            self.transverse_save_var,
            row=2,
        )

        ttk.Checkbutton(
            output_frame, text="Save trajectory", variable=self.trajectory_save_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Label(output_frame, text="Trajectory stride:").grid(
            row=4, column=0, sticky="w"
        )
        ttk.Entry(
            output_frame, textvariable=self.trajectory_interval_var, width=8
        ).grid(row=4, column=1, sticky="w")

        ttk.Label(output_frame, text="Plot DPI:").grid(
            row=5, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.dpi_var,
            values=[str(dpi) for dpi in AVAILABLE_DPI_CHOICES],
            width=8,
            state="readonly",
        ).grid(row=5, column=1, sticky="w", pady=(12, 0))

        # Config tab ----------------------------------------------------
        config_frame = ttk.Frame(notebook, padding=12)
        notebook.add(config_frame, text="Configs")
        config_frame.columnconfigure(1, weight=1)

        ttk.Label(config_frame, text="Config directory:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(config_frame, textvariable=self.config_dir_var).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(config_frame, text="Browse", command=self._select_config_dir).grid(
            row=0, column=2, padx=(6, 0)
        )

        ttk.Label(config_frame, text="Output directory:").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(config_frame, textvariable=self.output_dir_var).grid(
            row=1, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Button(config_frame, text="Browse", command=self._select_output_dir).grid(
            row=1, column=2, padx=(6, 0), pady=(6, 0)
        )

        ttk.Label(config_frame, text="Config name:").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(config_frame, textvariable=self.config_name_var).grid(
            row=2, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Button(config_frame, text="Save config", command=self._save_config).grid(
            row=2, column=2, padx=(6, 0), pady=(6, 0)
        )

        ttk.Label(config_frame, text="Saved configs:").grid(
            row=3, column=0, sticky="w", pady=(12, 0)
        )
        self.config_list = tk.Listbox(
            config_frame, height=8, listvariable=tk.Variable(value=[])
        )
        self.config_list.grid(row=4, column=0, columnspan=2, sticky="nsew")
        config_frame.rowconfigure(4, weight=1)
        self.config_list.bind(
            "<<ListboxSelect>>", lambda _event: self._on_config_selected()
        )
        self.config_list.bind("<Double-1>", lambda _event: self._load_config())

        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=4, column=2, sticky="nsw", padx=(6, 0))
        button_frame.columnconfigure(0, weight=1)
        ttk.Button(button_frame, text="Load selected", command=self._load_config).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(
            button_frame, text="Refresh", command=self._refresh_config_list
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        # Footer --------------------------------------------------------
        footer = ttk.Frame(self.root, padding=8)
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(2, weight=1)

        self._run_button = ttk.Button(footer, text="Run", command=self._trigger_run)
        self._run_button.grid(row=0, column=0, sticky="w")

        self._cancel_button = ttk.Button(
            footer, text="Cancel", command=self._trigger_cancel, state="disabled"
        )
        self._cancel_button.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(footer, textvariable=self.status_var).grid(
            row=0, column=2, sticky="w", padx=(12, 0)
        )

        self._progress_bar = ttk.Progressbar(
            footer,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            length=200,
        )
        self._progress_bar.grid(row=0, column=3, sticky="w", padx=(12, 0))
        ttk.Button(footer, text="Close", command=self.root.destroy).grid(
            row=0, column=4, sticky="e", padx=(12, 0)
        )

        # Summary + logs ------------------------------------------------
        summary_frame = ttk.LabelFrame(self.root, text="Initial summary", padding=8)
        summary_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        summary_frame.columnconfigure(0, weight=1)
        ttk.Label(summary_frame, textvariable=self.summary_var, justify="left").grid(
            row=0, column=0, sticky="w"
        )

        log_frame = ttk.LabelFrame(self.root, text="Logs", padding=8)
        log_frame.grid(row=4, column=0, sticky="nsew", padx=8, pady=(0, 8))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_output = scrolledtext.ScrolledText(
            log_frame, height=10, state="disabled"
        )
        self.log_output.grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------
    # UI Helpers
    # ------------------------------------------------------------------

    def _add_output_toggle(
        self,
        parent: tk.Widget,
        label: str,
        display_var: tk.BooleanVar,
        save_var: tk.BooleanVar,
        *,
        row: int,
    ) -> None:
        ttk.Checkbutton(
            parent, text=f"Display {label.lower()}", variable=display_var
        ).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Checkbutton(parent, text=f"Save {label.lower()}", variable=save_var).grid(
            row=row, column=1, sticky="w", pady=2
        )

    def _append_log(self, text: str) -> None:
        self.log_output.configure(state="normal")
        self.log_output.insert(tk.END, text + "\n")
        self.log_output.see(tk.END)
        self.log_output.configure(state="disabled")

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
        self._refresh_initial_summary()

    def _refresh_config_list(self, selected: Optional[str] = None) -> None:
        configs = list_config_files(Path(self.config_dir_var.get()))
        self.config_list.delete(0, tk.END)
        self.config_list.selection_clear(0, tk.END)

        highlight: Optional[int] = None
        if selected and selected in configs:
            highlight = configs.index(selected)
        elif self.config_file_var.get() in configs:
            highlight = configs.index(self.config_file_var.get())

        for config in configs:
            self.config_list.insert(tk.END, config)

        if highlight is None and configs:
            highlight = 0

        if highlight is not None and configs:
            self.config_list.selection_set(highlight)
            self.config_list.see(highlight)
            filename = configs[highlight]
            self.config_file_var.set(filename)
        else:
            self.config_file_var.set("")

    def _selected_config_filename(self) -> Optional[str]:
        selection = self.config_list.curselection()
        if not selection:
            return None
        return self.config_list.get(selection[0])

    def _on_config_selected(self) -> None:
        filename = self._selected_config_filename()
        if filename:
            self.config_file_var.set(filename)
        else:
            self.config_file_var.set("")

    def _load_config(self) -> None:
        filename = self._selected_config_filename()
        if not filename:
            messagebox.showinfo("Load config", "Select a configuration to load.")
            return

        path = Path(self.config_dir_var.get()) / filename
        try:
            options = load_config(path)
        except Exception as exc:
            messagebox.showerror("Load config", f"Failed to load {filename}: {exc}")
            return

        self._apply_options_to_ui(options)
        self.config_name_var.set(filename)
        self.config_file_var.set(filename)
        self._refresh_initial_summary()
        self._update_legacy_state()
        self._update_driver_visibility()
        self._set_status(f"Loaded config: {filename}")

    def _apply_options_to_ui(self, options: SimulationOptions) -> None:
        self.options = options
        self.sim_type_var.set(options.simulation_type.name)
        self.steps_var.set(options.steps)
        self.seed_var.set(options.seed)
        self.legacy_var.set(options.legacy_enabled)
        self.overlay_display_var.set(options.overlay_display)
        self.overlay_save_var.set(options.overlay_save)
        self.difference_display_var.set(options.difference_display)
        self.difference_save_var.set(options.difference_save)
        self.metrics_save_var.set(options.metrics_save)
        self.energy_display_var.set(options.energy_display)
        self.energy_save_var.set(options.energy_save)
        self.transverse_display_var.set(options.transverse_display)
        self.transverse_save_var.set(options.transverse_save)
        self.trajectory_save_var.set(options.trajectory_save)
        self.trajectory_interval_var.set(options.trajectory_interval)
        self.dpi_var.set(options.plot_dpi)
        self.image_subcharge_var.set(options.image_subcharge_count)
        self.image_weighting_var.set(options.use_image_weighting)
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
        core_params = {
            name: float(self.core_param_vars[name].get())
            for name in CORE_PARAM_DEFAULTS
        }

        config_name = self.config_name_var.get().strip() or "testbed_config"
        if not config_name.endswith(".json"):
            config_name += ".json"

        options = SimulationOptions(
            simulation_type=sim_type,
            steps=int(self.steps_var.get()),
            seed=int(self.seed_var.get()),
            rider_params=rider_params,
            driver_params=driver_params,
            core_params=core_params,
            legacy_enabled=bool(self.legacy_var.get()),
            overlay_display=bool(self.overlay_display_var.get()),
            overlay_save=bool(self.overlay_save_var.get()),
            difference_display=bool(self.difference_display_var.get()),
            difference_save=bool(self.difference_save_var.get()),
            metrics_save=bool(self.metrics_save_var.get()),
            energy_display=bool(self.energy_display_var.get()),
            energy_save=bool(self.energy_save_var.get()),
            transverse_display=bool(self.transverse_display_var.get()),
            transverse_save=bool(self.transverse_save_var.get()),
            trajectory_save=bool(self.trajectory_save_var.get()),
            trajectory_interval=int(self.trajectory_interval_var.get()),
            plot_dpi=int(self.dpi_var.get()),
            image_subcharge_count=int(self.image_subcharge_var.get()),
            use_image_weighting=bool(self.image_weighting_var.get()),
            output_dir=Path(self.output_dir_var.get()),
            config_dir=Path(self.config_dir_var.get()),
            config_name=config_name,
        )
        return options

    def _refresh_initial_summary(self) -> None:
        try:
            options = self._build_options_from_ui()
        except ValueError:
            return
        except Exception as exc:
            self.summary_var.set(f"Summary unavailable: {exc}")
            return
        summary = compute_initial_summary(options)
        self.summary_var.set(self._format_summary(summary))

    def _format_summary(self, summary: InitialSummary) -> str:
        lines = [f"Seed: {summary.seed}"]
        lines.append(f"Rider gamma: {summary.rider_gamma:.4f}")
        lines.append(
            "Rider rest energy: "
            f"{summary.rider_rest_mev:.4f} MeV ({summary.rider_rest_gev:.4f} GeV)"
        )
        lines.append(f"Rider total energy: {summary.rider_total_gev:.4f} GeV")
        if summary.has_driver:
            lines.append("Driver present")
            lines.append(f"Driver gamma: {summary.driver_gamma:.4f}")
            if (
                summary.driver_rest_mev is not None
                and summary.driver_rest_gev is not None
            ):
                lines.append(
                    "Driver rest energy: "
                    f"{summary.driver_rest_mev:.4f} MeV ({summary.driver_rest_gev:.4f} GeV)"
                )
            if summary.driver_total_gev is not None:
                lines.append(f"Driver total energy: {summary.driver_total_gev:.4f} GeV")
        else:
            lines.append("Driver disabled for this mode")
        return "\n".join(lines)

    def _select_config_dir(self) -> None:
        directory = filedialog.askdirectory(title="Select config directory")
        if directory:
            self.config_dir_var.set(directory)
            self._refresh_config_list()

    def _select_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Select output directory")
        if directory:
            self.output_dir_var.set(directory)

    def _save_config(self) -> None:
        try:
            options = self._build_options_from_ui()
        except ValueError as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        ensure_directory(options.config_dir)
        config_path = options.config_dir / options.config_name
        try:
            save_config(options, config_path)
        except Exception as exc:
            messagebox.showerror("Save config", f"Failed to save configuration: {exc}")
            return

        self.config_name_var.set(options.config_name)
        self.config_file_var.set(options.config_name)
        self._refresh_config_list(selected=options.config_name)
        messagebox.showinfo("Save config", f"Configuration saved as {config_path.name}")
        self._set_status(f"Saved config: {config_path.name}")

    def _on_sim_type_change(self) -> None:
        self._update_driver_visibility()
        self._refresh_initial_summary()

    def _update_driver_visibility(self) -> None:
        enabled = supports_driver(SimulationType[self.sim_type_var.get()])
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

    def _update_legacy_state(self) -> None:
        enabled = self.legacy_var.get()
        state = "normal" if enabled else "disabled"

        # Update the comparison frame state
        if hasattr(self, "_comparison_frame"):
            for child in self._comparison_frame.winfo_children():
                if isinstance(child, (ttk.Checkbutton, ttk.Button)):
                    child.configure(state=state)

        if not enabled:
            self.overlay_display_var.set(False)
            self.overlay_save_var.set(False)
            self.difference_display_var.set(False)
            self.difference_save_var.set(False)
            self.metrics_save_var.set(False)

    # ------------------------------------------------------------------
    # Simulation execution
    # ------------------------------------------------------------------

    def _trigger_run(self) -> None:
        if self._running:
            messagebox.showinfo("LW Integrator", "Simulation already running")
            return
        try:
            options = self._build_options_from_ui()
        except ValueError as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        self.options = options
        ensure_directory(options.output_dir)
        for handle in list(self._figure_windows):
            self._close_figure(handle)

        self._cancel_requested = False
        self._set_status("Running...")
        self._append_log("Launching simulation...")
        self._running = True
        self.progress_var.set(0.0)
        self._run_button.configure(state="disabled")
        self._cancel_button.configure(state="normal")

        self._worker = threading.Thread(
            target=self._run_background, args=(options,), daemon=True
        )
        self._worker.start()

    def _trigger_cancel(self) -> None:
        if self._running:
            self._cancel_requested = True
            self._cancel_button.configure(state="disabled")
            self._append_log("Cancellation requested...")
            self._set_status("Cancelling...")

    def _run_background(self, options: SimulationOptions) -> None:
        from core.integration_runner import IntegrationCancelled

        def progress_callback(current: int, total: int) -> None:
            progress_pct = (current / total * 100.0) if total > 0 else 0.0
            self.root.after(0, lambda: self.progress_var.set(progress_pct))

        def cancel_callback() -> bool:
            return self._cancel_requested

        try:
            result = run_testbed(
                options,
                log=self._queue_log,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )
        except IntegrationCancelled:
            self.root.after(0, self._on_cancelled)
            return
        except Exception as exc:  # pragma: no cover - UI safeguard
            self.root.after(0, partial(self._on_failure, str(exc)))
            return
        self.root.after(0, partial(self._on_success, result))

    def _queue_log(self, text: str) -> None:
        self.root.after(0, partial(self._append_log, text))

    def _on_cancelled(self) -> None:
        self._running = False
        self._worker = None
        self._cancel_requested = False
        self._set_status("Cancelled")
        self._append_log("Simulation cancelled by user.")
        self._run_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")
        self.progress_var.set(0.0)

    def _on_failure(self, message: str) -> None:
        self._running = False
        self._worker = None
        self._cancel_requested = False
        self._set_status("Failed")
        self._append_log(message)
        self._run_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")
        self.progress_var.set(0.0)
        messagebox.showerror("LW Integrator", message)

    def _on_success(self, result: RunResult) -> None:
        self._running = False
        self._worker = None
        self._cancel_requested = False
        self._set_status("Completed")
        self._append_log("Simulation finished successfully.")
        self._append_log(f"Duration: {result.duration_s:.2f} s")
        self._run_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")
        self.progress_var.set(100.0)

        for name, figure in result.figures.items():
            title = (
                name.replace("_", " ").title() if isinstance(name, str) else str(name)
            )
            self._show_figure(title, figure)

    def _show_figure(self, title: str, figure) -> None:
        width_px, height_px = self._prepare_figure_for_display(figure)
        window = tk.Toplevel(self.root)
        window.title(title)
        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        if width_px and height_px:
            window.geometry(f"{width_px}x{height_px}")
        handle = _FigureHandle(name=title, figure=figure, window=window, canvas=canvas)
        self._figure_windows.append(handle)
        window.protocol("WM_DELETE_WINDOW", partial(self._close_figure, handle))

    def _close_figure(self, handle: _FigureHandle) -> None:
        if handle in self._figure_windows:
            self._figure_windows.remove(handle)
        handle.canvas.get_tk_widget().destroy()
        handle.window.destroy()

    def _prepare_figure_for_display(self, figure) -> Tuple[int, int]:
        try:
            current_dpi = float(figure.get_dpi())
            width_in, height_in = [float(v) for v in figure.get_size_inches()]
        except Exception:  # pragma: no cover - defensive fallback
            return 0, 0

        width_px = width_in * current_dpi
        height_px = height_in * current_dpi

        scale = min(
            DISPLAY_MAX_WIDTH / width_px if width_px else 1.0,
            DISPLAY_MAX_HEIGHT / height_px if height_px else 1.0,
            1.0,
        )

        if scale < 1.0:
            new_width_in = max(1e-3, width_in * scale)
            new_height_in = max(1e-3, height_in * scale)
            figure.set_size_inches(new_width_in, new_height_in, forward=False)
            self._scale_figure_visuals(figure, scale)
            width_px = new_width_in * current_dpi
            height_px = new_height_in * current_dpi

        return int(width_px), int(height_px)

    def _scale_figure_visuals(self, figure, scale: float) -> None:
        if scale >= 0.999:
            return
        try:
            from matplotlib.collections import PathCollection
            from matplotlib.lines import Line2D
            from matplotlib.text import Text
        except Exception:  # pragma: no cover - matplotlib internals missing
            return

        for text in figure.findobj(match=Text):
            text.set_fontsize(text.get_fontsize() * scale)

        for line in figure.findobj(match=Line2D):
            line.set_linewidth(line.get_linewidth() * scale)
            line.set_markersize(line.get_markersize() * scale)

        for collection in figure.findobj(match=PathCollection):
            sizes = collection.get_sizes()
            if sizes is not None and len(sizes):
                collection.set_sizes(sizes * scale * scale)


def main() -> None:
    root = tk.Tk()
    IntegratorGUI(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
