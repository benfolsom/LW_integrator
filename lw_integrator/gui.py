"""Tkinter front-end exposing the full integrator testbed experience.

This window mirrors the functionality of ``examples/validation/integrator_testbed.ipynb``
so users can configure particle parameters, manage JSON configs, export
figures, and review logs without relying on Jupyter.  Simulation work runs in a
background thread to keep the UI responsive; any requested figures are rendered
in dedicated top-level windows using Matplotlib's TkAgg backend.
"""

from __future__ import annotations

import re
import threading
import tkinter as tk
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional, Set, Tuple

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from examples.validation.core_vs_legacy_benchmark import (  # type: ignore[import]
    DEFAULT_DRIVER_PARAMS,
    DEFAULT_RIDER_PARAMS,
    SimulationType,
)

from .optimization_plugin import OptimizationPlugin
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


def _show_error_dialog(parent: tk.Tk | tk.Toplevel, title: str, message: str) -> None:
    """Show an error dialog with selectable text."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()

    # Icon and message frame
    frame = ttk.Frame(dialog, padding=10)
    frame.pack(fill="both", expand=True)

    # Message text (read-only but selectable)
    text = tk.Text(frame, wrap="word", height=8, width=60, relief="flat", borderwidth=0)
    text.insert("1.0", message)
    text.configure(state="disabled", bg=frame.cget("background"))
    text.pack(side="top", fill="both", expand=True, pady=(0, 10))

    # OK button
    button_frame = ttk.Frame(frame)
    button_frame.pack(side="bottom")
    ok_button = ttk.Button(button_frame, text="OK", command=dialog.destroy, width=10)
    ok_button.pack()
    ok_button.focus_set()

    # Center dialog
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f"+{x}+{y}")

    # Bind Enter and Escape to close
    dialog.bind("<Return>", lambda e: dialog.destroy())
    dialog.bind("<Escape>", lambda e: dialog.destroy())


def _show_warning_dialog(parent: tk.Tk | tk.Toplevel, title: str, message: str) -> None:
    """Show a warning dialog with selectable text."""
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()

    # Icon and message frame
    frame = ttk.Frame(dialog, padding=10)
    frame.pack(fill="both", expand=True)

    # Message text (read-only but selectable)
    text = tk.Text(frame, wrap="word", height=8, width=60, relief="flat", borderwidth=0)
    text.insert("1.0", message)
    text.configure(state="disabled", bg=frame.cget("background"))
    text.pack(side="top", fill="both", expand=True, pady=(0, 10))

    # OK button
    button_frame = ttk.Frame(frame)
    button_frame.pack(side="bottom")
    ok_button = ttk.Button(button_frame, text="OK", command=dialog.destroy, width=10)
    ok_button.pack()
    ok_button.focus_set()

    # Center dialog
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f"+{x}+{y}")

    # Bind Enter and Escape to close
    dialog.bind("<Return>", lambda e: dialog.destroy())
    dialog.bind("<Escape>", lambda e: dialog.destroy())


@dataclass
class _FigureHandle:
    name: str
    figure: Any
    window: tk.Toplevel
    canvas: FigureCanvasTkAgg


class _ScrollableNotebookPage:
    def __init__(self, notebook: ttk.Notebook, title: str, padding: int = 12) -> None:
        self.container = ttk.Frame(notebook)
        notebook.add(self.container, text=title)
        self.container.columnconfigure(0, weight=1)
        self.container.rowconfigure(0, weight=1)
        self._bound_widgets: Set[int] = set()

        self.canvas = tk.Canvas(self.container, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(
            self.container, orient="vertical", command=self.canvas.yview
        )
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.frame = ttk.Frame(self.canvas, padding=padding)
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.frame, anchor="nw"
        )

        self.frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self._mousewheel_bound = False
        self._bind_mousewheel(self.container)
        self._bind_mousewheel(self.canvas)
        self._bind_mousewheel(self.frame)

    def _on_frame_configure(self, _event: Any) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: Any) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _bind_mousewheel(self, widget: tk.Widget) -> None:
        widget_id = widget.winfo_id()
        if widget_id in self._bound_widgets:
            return
        self._bound_widgets.add(widget_id)
        widget.bind("<Enter>", lambda _event: self._activate_mousewheel(), add=True)
        widget.bind(
            "<Leave>", lambda _event: self._maybe_deactivate_mousewheel(), add=True
        )

    def refresh_mousewheel_bindings(self) -> None:
        self._register_descendants(self.frame)

    def _register_descendants(self, widget: tk.Widget) -> None:
        self._bind_mousewheel(widget)
        for child in widget.winfo_children():
            self._register_descendants(child)

    def _activate_mousewheel(self) -> None:
        if self._mousewheel_bound:
            return
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        self._mousewheel_bound = True

    def _maybe_deactivate_mousewheel(self) -> None:
        if not self._mousewheel_bound:
            return
        widget = self.container.winfo_containing(
            self.container.winfo_pointerx(), self.container.winfo_pointery()
        )
        if not self._is_descendant(widget):
            self._deactivate_mousewheel()

    def _deactivate_mousewheel(self) -> None:
        if not self._mousewheel_bound:
            return
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
        self._mousewheel_bound = False

    def _is_descendant(self, widget: Optional[tk.Widget]) -> bool:
        while widget is not None:
            if widget == self.container:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _on_mousewheel(self, event: Any) -> None:
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
        self.canvas.yview_scroll(delta, "units")


class IntegratorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LW Integrator Testbed")
        self.root.geometry("1400x1000")

        self.options = SimulationOptions()
        self._figure_windows: List[_FigureHandle] = []
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._cancel_requested = False
        self._scroll_pages: List[_ScrollableNotebookPage] = []

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
        self.energy_dual_plot_var = tk.BooleanVar(value=self.options.energy_dual_plot)
        self.transverse_display_var = tk.BooleanVar(
            value=self.options.transverse_display
        )
        self.transverse_save_var = tk.BooleanVar(value=self.options.transverse_save)
        self.trajectory_save_var = tk.BooleanVar(value=self.options.trajectory_save)
        self.trajectory_interval_var = tk.IntVar(value=self.options.trajectory_interval)
        self.dpi_var = tk.IntVar(value=self.options.plot_dpi)
        self.image_subcharge_var = tk.IntVar(value=self.options.image_subcharge_count)
        self.image_weighting_var = tk.BooleanVar(value=self.options.use_image_weighting)

        # Self-consistency options
        self.self_consistency_enabled_var = tk.BooleanVar(
            value=self.options.self_consistency_enabled
        )
        self.self_consistency_tolerance_var = tk.DoubleVar(
            value=self.options.self_consistency_tolerance
        )
        self.self_consistency_max_iterations_var = tk.IntVar(
            value=self.options.self_consistency_max_iterations
        )
        self.self_consistency_verbosity_var = tk.IntVar(
            value=self.options.self_consistency_verbosity
        )

        # Energy monitoring options
        self.energy_monitor_enabled_var = tk.BooleanVar(
            value=self.options.energy_monitor_enabled
        )
        self.energy_monitor_threshold_var = tk.DoubleVar(
            value=self.options.energy_monitor_threshold
        )
        self.energy_monitor_check_interval_var = tk.IntVar(
            value=self.options.energy_monitor_check_interval
        )
        self.energy_monitor_halt_on_jump_var = tk.BooleanVar(
            value=self.options.energy_monitor_halt_on_jump
        )
        self.energy_monitor_debug_var = tk.BooleanVar(
            value=self.options.energy_monitor_debug
        )

        # Adaptive timestep options
        self.adaptive_timestep_enabled_var = tk.BooleanVar(
            value=self.options.adaptive_timestep_enabled
        )
        self.adaptive_timestep_threshold_var = tk.DoubleVar(
            value=self.options.adaptive_timestep_threshold
        )
        self.adaptive_timestep_reduction_factor_var = tk.IntVar(
            value=self.options.adaptive_timestep_reduction_factor
        )
        self.adaptive_timestep_max_attempts_var = tk.IntVar(
            value=self.options.adaptive_timestep_max_attempts
        )
        self.adaptive_timestep_min_factor_var = tk.DoubleVar(
            value=self.options.adaptive_timestep_min_factor
        )
        self.adaptive_timestep_cooldown_steps_var = tk.IntVar(
            value=self.options.adaptive_timestep_cooldown_steps
        )
        self.adaptive_timestep_probe_threshold_var = tk.DoubleVar(
            value=self.options.adaptive_timestep_probe_threshold
        )
        self.adaptive_timestep_max_probe_steps_var = tk.IntVar(
            value=self.options.adaptive_timestep_max_probe_steps
        )
        self.adaptive_timestep_debug_var = tk.BooleanVar(
            value=self.options.adaptive_timestep_debug
        )

        self.output_dir_var = tk.StringVar(value=str(self.options.output_dir))
        self.config_dir_var = tk.StringVar(value=str(self.options.config_dir))
        self.config_name_var = tk.StringVar(value=self.options.config_name)
        self.config_file_var = tk.StringVar(value="")

        # Log file options
        self.save_log_file_var = tk.BooleanVar(value=self.options.save_log_file)

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

    def _create_scrollable_tab(
        self, notebook: ttk.Notebook, title: str, *, padding: int = 12
    ) -> ttk.Frame:
        page = _ScrollableNotebookPage(notebook, title, padding=padding)
        self._scroll_pages.append(page)
        return page.frame

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Build the complete GUI layout with all controls."""
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

        main_paned = ttk.Panedwindow(self.root, orient="vertical")
        main_paned.grid(row=1, column=0, sticky="nsew")
        notebook = ttk.Notebook(main_paned)
        main_paned.add(notebook, weight=15)

        bottom_container = ttk.Frame(main_paned)
        bottom_container.columnconfigure(0, weight=1)
        bottom_container.rowconfigure(1, weight=1)
        main_paned.add(bottom_container, weight=1)

        # Particles tab --------------------------------------------------
        particle_frame = self._create_scrollable_tab(notebook, "Particles", padding=12)
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
        core_frame = self._create_scrollable_tab(notebook, "Core params", padding=12)
        core_frame.columnconfigure(1, weight=1)

        for row, name in enumerate(CORE_PARAM_LABELS, start=0):
            ttk.Label(core_frame, text=CORE_PARAM_LABELS[name] + ":").grid(
                row=row, column=0, sticky="w", pady=2
            )
            ttk.Entry(
                core_frame, textvariable=self.core_param_vars[name], width=16
            ).grid(row=row, column=1, sticky="ew", pady=2)

        # Stability Settings tab ----------------------------------------
        stability_frame = self._create_scrollable_tab(notebook, "Stability", padding=12)
        stability_frame.columnconfigure(1, weight=1)

        # Self-consistency section
        sc_frame = ttk.LabelFrame(
            stability_frame, text="Self-Consistency Checks", padding=8
        )
        sc_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        sc_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            sc_frame,
            text="Enable self-consistency iterations (recommended)",
            variable=self.self_consistency_enabled_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(sc_frame, text="Convergence tolerance:").grid(
            row=1, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        ttk.Entry(
            sc_frame, textvariable=self.self_consistency_tolerance_var, width=16
        ).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(sc_frame, text="Max iterations:").grid(
            row=2, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        ttk.Entry(
            sc_frame, textvariable=self.self_consistency_max_iterations_var, width=16
        ).grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(sc_frame, text="Verbosity:").grid(row=3, column=0, sticky="w", pady=2)
        verbosity_frame = ttk.Frame(sc_frame)
        verbosity_frame.grid(row=3, column=1, sticky="w", pady=2)
        ttk.Spinbox(
            verbosity_frame,
            from_=0,
            to=2,
            textvariable=self.self_consistency_verbosity_var,
            width=5,
        ).pack(side="left")
        ttk.Label(
            verbosity_frame,
            text=" (0=silent, 1=basic, 2=detailed)",
            foreground="gray",
        ).pack(side="left")

        # Energy monitoring section
        em_frame = ttk.LabelFrame(
            stability_frame, text="Energy Jump Detection", padding=8
        )
        em_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        em_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            em_frame,
            text="Enable runtime energy monitoring",
            variable=self.energy_monitor_enabled_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(em_frame, text="Jump threshold (rel. change):").grid(
            row=1, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        ttk.Entry(
            em_frame, textvariable=self.energy_monitor_threshold_var, width=16
        ).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(em_frame, text="Check interval (steps):").grid(
            row=2, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        ttk.Entry(
            em_frame, textvariable=self.energy_monitor_check_interval_var, width=16
        ).grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Checkbutton(
            em_frame,
            text="Halt simulation on energy jump",
            variable=self.energy_monitor_halt_on_jump_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0))

        ttk.Checkbutton(
            em_frame,
            text="Debug output",
            variable=self.energy_monitor_debug_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0))

        # Adaptive timestep section
        at_frame = ttk.LabelFrame(
            stability_frame, text="Adaptive Timestep Refinement", padding=8
        )
        at_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        at_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            at_frame,
            text="Enable adaptive timestep (auto-refine on energy jumps)",
            variable=self.adaptive_timestep_enabled_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        ttk.Label(at_frame, text="Energy jump threshold:").grid(
            row=1, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_threshold_var, width=16
        ).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(at_frame, text="Timestep reduction factor:").grid(
            row=2, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_reduction_factor_var, width=16
        ).grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(at_frame, text="Max refinement attempts:").grid(
            row=3, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_max_attempts_var, width=16
        ).grid(row=3, column=1, sticky="ew", pady=2)

        ttk.Label(at_frame, text="Min timestep factor:").grid(
            row=4, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_min_factor_var, width=16
        ).grid(row=4, column=1, sticky="ew", pady=2)

        # Hysteresis parameters
        ttk.Label(at_frame, text="Cooldown steps:").grid(
            row=5, column=0, sticky="w", pady=2
        )
        ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_cooldown_steps_var, width=16
        ).grid(row=5, column=1, sticky="ew", pady=2)

        ttk.Label(at_frame, text="Probe threshold:").grid(
            row=6, column=0, sticky="w", pady=2
        )
        ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_probe_threshold_var, width=16
        ).grid(row=6, column=1, sticky="ew", pady=2)

        ttk.Label(at_frame, text="Max probe steps:").grid(
            row=7, column=0, sticky="w", pady=2
        )
        ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_max_probe_steps_var, width=16
        ).grid(row=7, column=1, sticky="ew", pady=2)

        ttk.Checkbutton(
            at_frame,
            text="Debug output (show refinement actions)",
            variable=self.adaptive_timestep_debug_var,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0))

        # Help text
        help_text = ttk.Label(
            stability_frame,
            text="These settings help prevent energy jumps and numerical instabilities.\n"
            "Self-consistency is recommended for all simulations and is enabled by default.\n"
            "Energy monitoring detects problems during runtime (threshold: 2.0 = 200% change).\n"
            "Adaptive timestep automatically reduces timestep when energy jumps are detected (enabled by default).",
            wraplength=450,
            justify="left",
            foreground="gray",
        )
        help_text.grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 0))

        # Outputs tab ---------------------------------------------------
        output_frame = self._create_scrollable_tab(notebook, "Outputs", padding=12)
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
        ttk.Checkbutton(
            output_frame,
            text="  ↳ Show ΔE_z (longitudinal) on energy plots",
            variable=self.energy_dual_plot_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=(20, 0))
        self._add_output_toggle(
            output_frame,
            "Transverse plot",
            self.transverse_display_var,
            self.transverse_save_var,
            row=3,
        )

        ttk.Checkbutton(
            output_frame, text="Save trajectory", variable=self.trajectory_save_var
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Label(output_frame, text="Trajectory stride:").grid(
            row=5, column=0, sticky="w"
        )
        ttk.Entry(
            output_frame, textvariable=self.trajectory_interval_var, width=8
        ).grid(row=5, column=1, sticky="w")

        ttk.Label(output_frame, text="Plot DPI:").grid(
            row=6, column=0, sticky="w", pady=(12, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.dpi_var,
            values=[str(dpi) for dpi in AVAILABLE_DPI_CHOICES],
            width=8,
            state="readonly",
        ).grid(row=6, column=1, sticky="w", pady=(12, 0))

        # Log file saving
        ttk.Checkbutton(
            output_frame,
            text="Save log file to test_outputs directory",
            variable=self.save_log_file_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(12, 0))

        # Config tab ----------------------------------------------------
        config_frame = self._create_scrollable_tab(notebook, "Configs", padding=12)
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

        # Optimization tab -----------------------------------------------
        self.optimization_tab = OptimizationPlugin(notebook, gui_controller=self)
        notebook.add(self.optimization_tab, text="Optimization")

        # Footer --------------------------------------------------------

        footer = ttk.Frame(bottom_container, padding=8)
        footer.grid(row=0, column=0, sticky="ew")

        footer.columnconfigure(3, weight=1)

        self._run_button = ttk.Button(footer, text="Run", command=self._trigger_run)

        self._run_button.grid(row=0, column=0, sticky="w")

        self._cancel_button = ttk.Button(
            footer, text="Cancel", command=self._trigger_cancel, state="disabled"
        )

        self._cancel_button.grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Label(footer, textvariable=self.status_var).grid(
            row=0, column=3, sticky="w", padx=(12, 0)
        )

        self._progress_bar = ttk.Progressbar(
            footer,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            length=200,
        )

        self._progress_bar.grid(row=0, column=4, sticky="w", padx=(12, 0))

        ttk.Button(footer, text="Close", command=self.root.destroy).grid(
            row=0, column=5, sticky="e", padx=(12, 0)
        )

        # Summary + logs ------------------------------------------------

        lower_paned = ttk.Panedwindow(bottom_container, orient="vertical")

        lower_paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        summary_frame = ttk.LabelFrame(lower_paned, text="Initial summary", padding=8)

        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)

        # Make summary scrollable with Text widget instead of Label
        summary_text_frame = ttk.Frame(summary_frame)
        summary_text_frame.grid(row=0, column=0, sticky="nsew")
        summary_text_frame.columnconfigure(0, weight=1)
        summary_text_frame.rowconfigure(0, weight=1)

        self.summary_text = tk.Text(
            summary_text_frame,
            height=3,
            width=70,
            wrap="word",
            state="disabled",
            relief="flat",
            borderwidth=0,
        )
        self.summary_text.grid(row=0, column=0, sticky="nsew")

        summary_scrollbar = ttk.Scrollbar(
            summary_text_frame, command=self.summary_text.yview
        )
        summary_scrollbar.grid(row=0, column=1, sticky="ns")
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)

        lower_paned.add(summary_frame, weight=1)

        log_frame = ttk.LabelFrame(lower_paned, text="Logs", padding=8)

        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=0)
        log_frame.rowconfigure(1, weight=1)

        # Log controls
        log_controls = ttk.Frame(log_frame)
        log_controls.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self.log_format_var = tk.StringVar(value="detailed")
        ttk.Radiobutton(
            log_controls,
            text="Summary",
            variable=self.log_format_var,
            value="summary",
            command=self._update_log_format,
        ).pack(side="left", padx=5)
        ttk.Radiobutton(
            log_controls,
            text="Detailed",
            variable=self.log_format_var,
            value="detailed",
            command=self._update_log_format,
        ).pack(side="left", padx=5)

        ttk.Button(log_controls, text="Clear", command=self._clear_log, width=8).pack(
            side="right", padx=5
        )

        self.log_output = scrolledtext.ScrolledText(
            log_frame, height=6, state="disabled", wrap="none"
        )

        self.log_output.grid(row=1, column=0, sticky="nsew")

        # Store raw and parsed logs
        self._raw_log_lines = []
        self._log_summary = []

        lower_paned.add(log_frame, weight=1)

        for page in self._scroll_pages:
            page.refresh_mousewheel_bindings()

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
        """Append text to log with parsing for summary view."""
        # Store raw line
        self._raw_log_lines.append(text)

        # Parse for summary
        self._parse_log_line(text)

        # Display based on current format
        if self.log_format_var.get() == "detailed":
            self.log_output.configure(state="normal")
            self.log_output.insert(tk.END, text + "\n")
            self.log_output.see(tk.END)
            self.log_output.configure(state="disabled")

    def _parse_log_line(self, text: str, auto_refresh: bool = True) -> None:
        """Parse log line and extract key events for summary.

        Args:
            text: Log line to parse
            auto_refresh: If True, refresh summary display after parsing.
                         Set to False during bulk loading to avoid slowdown.
        """
        # SC convergence
        if "converged in" in text:
            match = re.search(r"Particle (\d+): converged in (\d+) iter", text)
            if match:
                self._log_summary.append(
                    f"[SC] P{match.group(1)} converged in {match.group(2)} iterations"
                )

        # SC iteration details
        elif "Δγ/γ =" in text:
            match = re.search(r"Δγ/γ = ([\d.e+-]+)", text)
            if match:
                gamma_err = float(match.group(1))
                self._log_summary.append(f"     γ error: {gamma_err:.2e}")

        # Energy jumps
        elif "Energy jump detected" in text:
            match = re.search(r"Step ([\d.]+).*ΔE/E = ([\d.e+-]+)", text)
            if match:
                step = match.group(1)
                de = float(match.group(2))
                self._log_summary.append(
                    f"[ENERGY] Step {step}: ΔE/E = {de:.2%} - reducing timestep"
                )

        # Adaptive timestep events
        elif "Reducing timestep by" in text or "reducing timestep by" in text:
            match = re.search(r"by (\d+)x to ([\d.e+-]+)", text)
            if match:
                factor = match.group(1)
                new_h = float(match.group(2))
                self._log_summary.append(
                    f"     → h reduced {factor}x to {new_h:.2e} ns"
                )

        elif "Cooldown mode" in text:
            match = re.search(r"Step (\d+): Cooldown mode \((\d+)/(\d+)\)", text)
            if match:
                step, current, total = match.group(1), match.group(2), match.group(3)
                if current == "1":  # Only log start of cooldown
                    self._log_summary.append(
                        f"[COOL] Step {step}: Cooldown phase ({total} steps)"
                    )

        elif "Returning to normal timestep" in text:
            match = re.search(r"Step (\d+):.*to ([\d.e+-]+)", text)
            if match:
                step = match.group(1)
                h = float(match.group(2))
                self._log_summary.append(
                    f"[RESUME] Step {step}: Normal timestep {h:.2e} ns restored"
                )

        elif "Mass-shell projection" in text:
            match = re.search(
                r"Pt ([\d.e+-]+) → ([\d.e+-]+).*error was ([\d.e+-]+)", text
            )
            if match:
                pt_old = float(match.group(1))
                pt_new = float(match.group(2))
                error = float(match.group(3))
                self._log_summary.append(
                    f"[MASS-SHELL] Pt corrected (error={error:.2e})"
                )

        # Optimization sweep messages
        elif "[OPTIMIZATION]" in text:
            # Include all optimization messages in summary
            # They're already prefixed and formatted nicely
            self._log_summary.append(text.strip())

        # Update summary display if in summary mode (but only if auto_refresh enabled)
        if (
            auto_refresh
            and self.log_format_var.get() == "summary"
            and self._log_summary
        ):
            self._refresh_summary_display()

    def _refresh_summary_display(self) -> None:
        """Refresh the summary log display."""
        self.log_output.configure(state="normal")
        self.log_output.delete("1.0", tk.END)

        # Show last 100 summary lines
        display_lines = self._log_summary[-100:]
        self.log_output.insert("1.0", "\n".join(display_lines))
        self.log_output.see(tk.END)
        self.log_output.configure(state="disabled")

    def _update_log_format(self) -> None:
        """Switch between summary and detailed log views."""
        self.log_output.configure(state="normal")
        self.log_output.delete("1.0", tk.END)

        if self.log_format_var.get() == "summary":
            # Show summary
            display_lines = self._log_summary[-100:]
            if display_lines:
                self.log_output.insert("1.0", "\n".join(display_lines))
        else:
            # Show detailed (last 500 lines)
            display_lines = self._raw_log_lines[-500:]
            if display_lines:
                self.log_output.insert("1.0", "\n".join(display_lines))

        self.log_output.see(tk.END)
        self.log_output.configure(state="disabled")

    def _clear_log(self) -> None:
        """Clear all logs."""
        self._raw_log_lines = []
        self._log_summary = []
        self.log_output.configure(state="normal")
        self.log_output.delete("1.0", tk.END)
        self.log_output.configure(state="disabled")

    def _load_verbose_logs(self, verbose_logs: str) -> None:
        """Load verbose logs into the detailed view automatically after run.

        Args:
            verbose_logs: String containing all verbose log output from the run
        """
        if verbose_logs:
            try:
                # Parse verbose logs into raw lines (disable auto-refresh during bulk load)
                line_count = 0
                for line in verbose_logs.splitlines():
                    if line.strip():
                        self._raw_log_lines.append(line)
                        self._parse_log_line(line, auto_refresh=False)
                        line_count += 1

                # Refresh the current view to show loaded logs
                self._update_log_format()

                self._append_log(f"--- Loaded {line_count:,} verbose log lines ---")
            except Exception as e:
                self._append_log(f"Error loading verbose logs: {e}")
                import traceback

                traceback.print_exc()

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
        result = self.config_list.get(selection[0])
        return str(result) if result else None

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
            _show_error_dialog(
                self.root, "Load config", f"Failed to load {filename}: {exc}"
            )
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
        self.energy_dual_plot_var.set(options.energy_dual_plot)
        self.transverse_display_var.set(options.transverse_display)
        self.transverse_save_var.set(options.transverse_save)
        self.trajectory_save_var.set(options.trajectory_save)
        self.trajectory_interval_var.set(options.trajectory_interval)
        self.dpi_var.set(options.plot_dpi)
        self.image_subcharge_var.set(options.image_subcharge_count)
        self.image_weighting_var.set(options.use_image_weighting)
        self.self_consistency_enabled_var.set(options.self_consistency_enabled)
        self.self_consistency_tolerance_var.set(options.self_consistency_tolerance)
        self.self_consistency_max_iterations_var.set(
            options.self_consistency_max_iterations
        )
        self.self_consistency_verbosity_var.set(options.self_consistency_verbosity)
        self.energy_monitor_enabled_var.set(options.energy_monitor_enabled)
        self.energy_monitor_threshold_var.set(options.energy_monitor_threshold)
        self.energy_monitor_check_interval_var.set(
            options.energy_monitor_check_interval
        )
        self.energy_monitor_halt_on_jump_var.set(options.energy_monitor_halt_on_jump)
        self.energy_monitor_debug_var.set(options.energy_monitor_debug)
        self.adaptive_timestep_enabled_var.set(options.adaptive_timestep_enabled)
        self.adaptive_timestep_threshold_var.set(options.adaptive_timestep_threshold)
        self.adaptive_timestep_reduction_factor_var.set(
            options.adaptive_timestep_reduction_factor
        )
        self.adaptive_timestep_max_attempts_var.set(
            options.adaptive_timestep_max_attempts
        )
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
        self.save_log_file_var.set(options.save_log_file)
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
            energy_dual_plot=bool(self.energy_dual_plot_var.get()),
            transverse_display=bool(self.transverse_display_var.get()),
            transverse_save=bool(self.transverse_save_var.get()),
            trajectory_save=bool(self.trajectory_save_var.get()),
            trajectory_interval=int(self.trajectory_interval_var.get()),
            plot_dpi=int(self.dpi_var.get()),
            output_dir=Path(self.output_dir_var.get()),
            config_dir=Path(self.config_dir_var.get()),
            config_name=config_name,
            image_subcharge_count=int(self.image_subcharge_var.get()),
            use_image_weighting=bool(self.image_weighting_var.get()),
            self_consistency_enabled=bool(self.self_consistency_enabled_var.get()),
            self_consistency_tolerance=float(self.self_consistency_tolerance_var.get()),
            self_consistency_max_iterations=int(
                self.self_consistency_max_iterations_var.get()
            ),
            self_consistency_verbosity=int(self.self_consistency_verbosity_var.get()),
            energy_monitor_enabled=bool(self.energy_monitor_enabled_var.get()),
            energy_monitor_threshold=float(self.energy_monitor_threshold_var.get()),
            energy_monitor_check_interval=int(
                self.energy_monitor_check_interval_var.get()
            ),
            energy_monitor_halt_on_jump=bool(
                self.energy_monitor_halt_on_jump_var.get()
            ),
            energy_monitor_debug=bool(self.energy_monitor_debug_var.get()),
            adaptive_timestep_enabled=bool(self.adaptive_timestep_enabled_var.get()),
            adaptive_timestep_threshold=float(
                self.adaptive_timestep_threshold_var.get()
            ),
            adaptive_timestep_reduction_factor=int(
                self.adaptive_timestep_reduction_factor_var.get()
            ),
            adaptive_timestep_max_attempts=int(
                self.adaptive_timestep_max_attempts_var.get()
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
        formatted_summary = self._format_summary(summary)
        self.summary_var.set(formatted_summary)

        # Update the summary text widget
        if hasattr(self, "summary_text"):
            self.summary_text.config(state="normal")
            self.summary_text.delete("1.0", "end")
            self.summary_text.insert("1.0", formatted_summary)
            self.summary_text.config(state="disabled")

    def _format_summary(self, summary: InitialSummary) -> str:
        lines = [f"Seed: {summary.seed}"]
        lines.append(f"Rider gamma: {summary.rider_gamma:.4f}")
        lines.append(
            "Rider rest energy: "
            f"{summary.rider_rest_mev:.4f} MeV ({summary.rider_rest_gev:.4f} GeV)"
        )
        lines.append(f"Rider total energy: {summary.rider_total_gev:.4f} GeV")

        # Add rider beam optics if available
        if summary.rider_emittance_x_mm_mrad is not None:
            # Convert to picometer-radians for alternative display
            emit_x_pm = summary.rider_emittance_x_mm_mrad * 1e9  # pm·rad
            emit_y_pm = summary.rider_emittance_y_mm_mrad * 1e9  # pm·rad
            norm_emit_x_pm = summary.rider_norm_emittance_x_mm_mrad * 1e9  # pm·rad
            norm_emit_y_pm = summary.rider_norm_emittance_y_mm_mrad * 1e9  # pm·rad

            lines.append(
                f"Rider ε: "
                f"{summary.rider_emittance_x_mm_mrad:.2e} mm·mrad ({emit_x_pm:.2e} pm·rad), "
                f"{summary.rider_emittance_y_mm_mrad:.2e} mm·mrad ({emit_y_pm:.2e} pm·rad)"
            )
            lines.append(
                f"Rider εₙ: "
                f"{summary.rider_norm_emittance_x_mm_mrad:.2e} mm·mrad ({norm_emit_x_pm:.2e} pm·rad), "
                f"{summary.rider_norm_emittance_y_mm_mrad:.2e} mm·mrad ({norm_emit_y_pm:.2e} pm·rad)"
            )
            lines.append(
                f"Rider β: "
                f"{summary.rider_beta_x_m:.3f} m, "
                f"{summary.rider_beta_y_m:.3f} m"
            )

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

            # Add driver beam optics if available
            if summary.driver_emittance_x_mm_mrad is not None:
                driver_emit_x_pm = summary.driver_emittance_x_mm_mrad * 1e9
                driver_emit_y_pm = summary.driver_emittance_y_mm_mrad * 1e9
                driver_norm_emit_x_pm = summary.driver_norm_emittance_x_mm_mrad * 1e9
                driver_norm_emit_y_pm = summary.driver_norm_emittance_y_mm_mrad * 1e9

                lines.append(
                    f"Driver ε: "
                    f"{summary.driver_emittance_x_mm_mrad:.2e} mm·mrad ({driver_emit_x_pm:.2e} pm·rad), "
                    f"{summary.driver_emittance_y_mm_mrad:.2e} mm·mrad ({driver_emit_y_pm:.2e} pm·rad)"
                )
                lines.append(
                    f"Driver εₙ: "
                    f"{summary.driver_norm_emittance_x_mm_mrad:.2e} mm·mrad ({driver_norm_emit_x_pm:.2e} pm·rad), "
                    f"{summary.driver_norm_emittance_y_mm_mrad:.2e} mm·mrad ({driver_norm_emit_y_pm:.2e} pm·rad)"
                )
                lines.append(
                    f"Driver β: "
                    f"{summary.driver_beta_x_m:.3f} m, "
                    f"{summary.driver_beta_y_m:.3f} m"
                )
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
            _show_error_dialog(self.root, "Invalid configuration", str(exc))
            return

        ensure_directory(options.config_dir)
        config_path = options.config_dir / options.config_name
        try:
            save_config(options, config_path)
        except Exception as exc:
            _show_error_dialog(
                self.root, "Save config", f"Failed to save configuration: {exc}"
            )
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
            _show_error_dialog(self.root, "Invalid configuration", str(exc))
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
            # Log brief error to summary
            brief_error = str(exc)
            # Log full traceback to detailed logs
            full_traceback = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            # Store full details in raw log lines for detailed view
            for line in full_traceback.splitlines():
                if line.strip():
                    self._raw_log_lines.append(line)
            # Pass brief error to failure handler
            self.root.after(0, partial(self._on_failure, brief_error))
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
        # Add error to summary
        self._log_summary.append(f"[ERROR] {message}")
        # Log brief error message
        self._append_log(f"Error: {message}")
        self._append_log("(Full traceback available in Detailed view)")
        self._run_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")
        self.progress_var.set(0.0)
        _show_error_dialog(self.root, "LW Integrator", message)

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

        # Auto-load verbose logs into GUI for post-run analysis
        if hasattr(result, "verbose_logs") and result.verbose_logs:
            verbose_line_count = len(
                [l for l in result.verbose_logs.splitlines() if l.strip()]
            )
            self._append_log(
                f"Loading {verbose_line_count:,} verbose log lines into GUI..."
            )
            self._load_verbose_logs(result.verbose_logs)

        for name, figure in result.figures.items():
            title = (
                name.replace("_", " ").title() if isinstance(name, str) else str(name)
            )
            try:
                self._show_figure(title, figure)
            except Exception as e:
                error_msg = f"Error displaying {title} plot: {e}"
                self._append_log(error_msg)
                _show_warning_dialog(self.root, "Plot Display Error", error_msg)

    def _show_figure(self, title: str, figure: Any) -> None:
        try:
            width_px, height_px = self._prepare_figure_for_display(figure)
        except Exception as e:
            self._append_log(f"Warning: Could not prepare figure for display: {e}")
            # Use default size if preparation fails
            width_px, height_px = 800, 600

        window = tk.Toplevel(self.root)
        window.title(title)

        # Create main container frame
        main_frame = ttk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas for the figure
        canvas = FigureCanvasTkAgg(figure, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add matplotlib navigation toolbar
        toolbar_frame = ttk.Frame(main_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        # Create custom controls frame
        controls_frame = ttk.Frame(main_frame, padding=5)
        controls_frame.pack(side=tk.TOP, fill=tk.X)

        # Log scale controls
        ttk.Label(controls_frame, text="Log scale:").pack(side=tk.LEFT, padx=(0, 5))

        x_log_var = tk.BooleanVar(value=False)
        y_log_var = tk.BooleanVar(value=False)

        def toggle_log_scale() -> None:
            try:
                for ax in figure.get_axes():
                    if x_log_var.get():
                        try:
                            ax.set_xscale("log")
                        except (ValueError, RuntimeWarning):
                            x_log_var.set(False)
                            _show_warning_dialog(
                                window,
                                "Log Scale Warning",
                                "X-axis cannot be log-scaled (data may contain non-positive values)",
                            )
                    else:
                        ax.set_xscale("linear")
                    if y_log_var.get():
                        try:
                            ax.set_yscale("log")
                        except (ValueError, RuntimeWarning):
                            y_log_var.set(False)
                            _show_warning_dialog(
                                window,
                                "Log Scale Warning",
                                "Y-axis cannot be log-scaled (data may contain non-positive values)",
                            )
                    else:
                        ax.set_yscale("linear")
                canvas.draw()
            except Exception as e:
                self._append_log(f"Error toggling log scale: {e}")

        ttk.Checkbutton(
            controls_frame, text="X-axis", variable=x_log_var, command=toggle_log_scale
        ).pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(
            controls_frame, text="Y-axis", variable=y_log_var, command=toggle_log_scale
        ).pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(controls_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        # Save/Save As buttons
        def save_figure() -> None:
            try:
                # Use the current filename if it exists
                default_name = f"{title.replace(' ', '_').replace('/', '_')}.png"
                figure.savefig(default_name, dpi=150, bbox_inches="tight")
                self._append_log(f"Figure saved to: {default_name}")
            except Exception as e:
                _show_error_dialog(window, "Save Error", f"Failed to save figure: {e}")

        def save_figure_as() -> None:
            try:
                default_name = f"{title.replace(' ', '_').replace('/', '_')}.png"
                filename = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    initialfile=default_name,
                    filetypes=[
                        ("PNG files", "*.png"),
                        ("PDF files", "*.pdf"),
                        ("SVG files", "*.svg"),
                        ("All files", "*.*"),
                    ],
                )
                if filename:
                    figure.savefig(filename, dpi=150, bbox_inches="tight")
                    self._append_log(f"Figure saved to: {filename}")
            except Exception as e:
                _show_error_dialog(window, "Save Error", f"Failed to save figure: {e}")

        ttk.Button(controls_frame, text="Save", command=save_figure).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Button(controls_frame, text="Save As...", command=save_figure_as).pack(
            side=tk.LEFT, padx=5
        )

        if width_px and height_px:
            # Add extra height for toolbar and controls (~100px)
            window.geometry(f"{width_px}x{height_px + 100}")

        handle = _FigureHandle(name=title, figure=figure, window=window, canvas=canvas)
        self._figure_windows.append(handle)
        window.protocol("WM_DELETE_WINDOW", partial(self._close_figure, handle))

    def _close_figure(self, handle: _FigureHandle) -> None:
        if handle in self._figure_windows:
            self._figure_windows.remove(handle)
        handle.canvas.get_tk_widget().destroy()
        handle.window.destroy()

    def _prepare_figure_for_display(self, figure: Any) -> Tuple[int, int]:
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

    def _scale_figure_visuals(self, figure: Any, scale: float) -> None:
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
