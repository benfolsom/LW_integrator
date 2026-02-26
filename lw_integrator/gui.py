"""Tkinter front-end exposing the full integrator testbed experience.

This window mirrors the functionality of ``examples/validation/integrator_testbed.ipynb``
so users can configure particle parameters, manage JSON configs, export
figures, and review logs without relying on Jupyter.  Simulation work runs in a
background thread to keep the UI responsive; any requested figures are rendered
in dedicated top-level windows using Matplotlib's TkAgg backend.
"""

from __future__ import annotations

import json
import locale
import os
import re
import signal
import sys
import threading
import tkinter as tk
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional, Set, Tuple

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from core.batched_logger import BatchedLogger, ThrottledProgressCallback
from core.debug_logger import initialize_debug_logging
from core.particle_config import DEFAULT_DRIVER_PARAMS, DEFAULT_RIDER_PARAMS
from core.types import SimulationType
from lw_integrator.testbed_runner import (
    COLOR_DRIVER,
    COLOR_LEGACY_DRIVER,
    COLOR_LEGACY_RIDER,
    COLOR_RIDER,
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
CONTENT_PANEL_WEIGHT = 3  # ttk.Panedwindow weight for the tabbed content area
CONFIG_PANEL_WEIGHT = 2  # ttk.Panedwindow weight for the configuration panel
CONTENT_PANEL_MIN_WIDTH = 800  # pixels; ensures input fields in tabs remain accessible
CONFIG_PANEL_MIN_WIDTH = 450  # pixels; ensures right-side controls have breathing room
CONFIG_PANEL_CANVAS_PADDING = 24  # pixels; allows scrollable content to render fully


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
    # Use a light gray background that matches typical dialog backgrounds
    text.configure(state="disabled", bg="#f0f0f0")
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
    # Use system default background color instead of trying to get ttk frame background
    try:
        bg_color = dialog.tk.eval("ttk::style lookup TFrame -background")
        if not bg_color:
            bg_color = dialog.cget("background")
    except:
        bg_color = "white"
    text.configure(state="disabled", bg=bg_color)
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


class Tooltip:
    """Hover tooltip for widgets."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tooltip_window: Optional[tk.Toplevel] = None
        self.widget.bind("<Enter>", self._show_tooltip)
        self.widget.bind("<Leave>", self._hide_tooltip)

    def _show_tooltip(self, event: Any = None) -> None:
        """Display tooltip on hover."""
        if self.tooltip_window or not self.text:
            return

        # Position tooltip near widget
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        # Create tooltip window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # No window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        # Create tooltip label with wrapped text
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            justify="left",
            background="#ffffe0",
            foreground="black",
            relief="solid",
            borderwidth=1,
            wraplength=400,
            padx=8,
            pady=6,
            font=("TkDefaultFont", 9),
        )
        label.pack()

    def _hide_tooltip(self, event: Any = None) -> None:
        """Hide tooltip when mouse leaves."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


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

        # Vertical scrollbar
        self.scrollbar = ttk.Scrollbar(
            self.container, orient="vertical", command=self.canvas.yview
        )
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        # Horizontal scrollbar
        self.h_scrollbar = ttk.Scrollbar(
            self.container, orient="horizontal", command=self.canvas.xview
        )
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(
            yscrollcommand=self.scrollbar.set, xscrollcommand=self.h_scrollbar.set
        )

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
        # Update scroll region but don't force width - let horizontal scrollbar work
        canvas_width = event.width
        canvas_height = event.height

        # Get the actual content size
        bbox = self.canvas.bbox("all")
        if bbox:
            content_width = bbox[2] - bbox[0]
            content_height = bbox[3] - bbox[1]

            # Only set width if canvas is wider than content (prevents horizontal scroll when not needed)
            if canvas_width >= content_width:
                self.canvas.itemconfigure(self._window_id, width=canvas_width)
            else:
                # Let content maintain its natural width for horizontal scrolling
                self.canvas.itemconfigure(self._window_id, width=content_width)

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
        self.root.geometry("1800x1000")
        # Set minimum window size to prevent panels from becoming inaccessible
        self.root.minsize(CONTENT_PANEL_MIN_WIDTH + CONFIG_PANEL_MIN_WIDTH + 50, 600)

        # Initialize debug logging system
        initialize_debug_logging(context="gui")
        print("[LOGCACHE] Debug logging initialized in logcache/")

        self.options = SimulationOptions()
        self._figure_windows: List[_FigureHandle] = []
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._cancel_requested = False
        self._scroll_pages: List[_ScrollableNotebookPage] = []
        self._batched_logger: Optional[BatchedLogger] = None

        # Keyboard debugging mode (via environment variable)
        self._keyboard_debug = os.environ.get("LW_KEYBOARD_DEBUG", "0") == "1"
        if self._keyboard_debug:
            print("=" * 60)
            print("KEYBOARD DEBUG MODE ENABLED")
            print("All keyboard events will be logged to console")
            print("Set LW_KEYBOARD_DEBUG=0 to disable")
            print("=" * 60)

        # Preferences file for directory persistence
        self._prefs_file = Path.home() / ".lw_integrator_prefs.json"
        self._load_preferences()

        self._init_variables()
        self._build_layout()
        self._apply_options_to_ui(self.options, preserve_directories=True)
        self._refresh_config_list()
        self._refresh_initial_summary()
        self._update_legacy_state()
        self._update_driver_visibility()
        self._update_image_subcharge_state()

        # Set initial sash position for main horizontal pane (70/30 split)
        self.root.update_idletasks()  # Ensure window is laid out
        total_width = self.root.winfo_width()
        # Position sash to give config panel ~30% (remaining 70% goes to content/optimization)
        sash_position = int(total_width * 0.7)
        if hasattr(self, "_main_horizontal_paned"):
            self._main_horizontal_paned.sash_place(0, sash_position, 0)
            # Bind to enforce minimum panel sizes when sash is dragged
            self._main_horizontal_paned.bind(
                "<ButtonRelease-1>", self._enforce_panel_minimums
            )
            self._main_horizontal_paned.bind(
                "<B1-Motion>", self._enforce_panel_minimums
            )

        # Set up keyboard fix for non-US layouts (always enabled)
        self._setup_keyboard_fix()

        # Handle window close to save preferences
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Variable initialisation
    # ------------------------------------------------------------------

    def _init_variables(self) -> None:
        self.sim_type_var = tk.StringVar(value=self.options.simulation_type.name)
        self.steps_var = tk.IntVar(value=self.options.steps)
        self.seed_var = tk.IntVar(value=self.options.seed)
        self.random_seed_var = tk.BooleanVar(value=False)
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

        self.core_param_vars: Dict[str, tk.Variable] = {}
        for name, value in CORE_PARAM_DEFAULTS.items():
            if isinstance(value, str):
                self.core_param_vars[name] = tk.StringVar(value=value)
            elif isinstance(value, (int, float)):
                self.core_param_vars[name] = tk.DoubleVar(value=float(value))
            else:
                self.core_param_vars[name] = tk.StringVar(value=str(value))

        # z_cutoff enable checkbox variable
        self.z_cutoff_enabled_var = tk.BooleanVar(value=False)

        # Trace to update control states
        self.z_cutoff_enabled_var.trace_add(
            "write", lambda *_: self._toggle_z_cutoff_controls()
        )

        self.overlay_display_var = tk.BooleanVar(value=self.options.overlay_display)
        self.overlay_save_var = tk.BooleanVar(value=self.options.overlay_save)
        self.difference_display_var = tk.BooleanVar(
            value=self.options.difference_display
        )
        self.difference_save_var = tk.BooleanVar(value=self.options.difference_save)
        self.metrics_save_var = tk.BooleanVar(value=self.options.metrics_save)
        self.energy_display_var = tk.BooleanVar(value=self.options.energy_display)
        self.energy_save_var = tk.BooleanVar(value=self.options.energy_save)
        self.energy_xaxis_var = tk.StringVar(
            value=getattr(self.options, "energy_xaxis", "z")
        )
        self.energy_yaxis_var = tk.StringVar(
            value=getattr(self.options, "energy_yaxis", "delta_total")
        )
        self.transverse_display_var = tk.BooleanVar(
            value=self.options.transverse_display
        )
        self.transverse_save_var = tk.BooleanVar(value=self.options.transverse_save)
        self.transverse_xaxis_var = tk.StringVar(
            value=getattr(self.options, "transverse_xaxis", "t")
        )
        self.beta_display_var = tk.BooleanVar(value=self.options.beta_display)
        self.beta_save_var = tk.BooleanVar(value=self.options.beta_save)
        self.beta_xaxis_var = tk.StringVar(
            value=getattr(self.options, "beta_xaxis", "t")
        )
        self.momentum_display_var = tk.BooleanVar(value=self.options.momentum_display)
        self.momentum_save_var = tk.BooleanVar(value=self.options.momentum_save)
        self.momentum_xaxis_var = tk.StringVar(
            value=getattr(self.options, "momentum_xaxis", "t")
        )
        self.gamma_display_var = tk.BooleanVar(
            value=getattr(self.options, "gamma_display", False)
        )
        self.gamma_save_var = tk.BooleanVar(
            value=getattr(self.options, "gamma_save", False)
        )
        self.gamma_xaxis_var = tk.StringVar(
            value=getattr(self.options, "gamma_xaxis", "t")
        )
        self.zposition_display_var = tk.BooleanVar(
            value=getattr(self.options, "zposition_display", False)
        )
        self.zposition_save_var = tk.BooleanVar(
            value=getattr(self.options, "zposition_save", False)
        )
        self.trajectory_save_var = tk.BooleanVar(value=self.options.trajectory_save)
        self.trajectory_interval_var = tk.IntVar(value=self.options.trajectory_interval)
        self.dpi_var = tk.IntVar(value=self.options.plot_dpi)
        self.image_subcharge_var = tk.IntVar(value=self.options.image_subcharge_count)
        self.image_weighting_var = tk.BooleanVar(value=self.options.use_image_weighting)

        # Macroparticle simulation options
        self.macroparticle_enabled_var = tk.BooleanVar(value=False)
        self.macroparticle_charge_multiplier_var = tk.StringVar(value="1.0")
        self.macroparticle_sigma_multiplier_var = tk.StringVar(value="1.0")
        self.macroparticle_use_momentum_errors_var = tk.BooleanVar(value=True)

        # Self-consistency options
        self.self_consistency_enabled_var = tk.BooleanVar(
            value=self.options.self_consistency_enabled
        )
        self.self_consistency_convergence_mode_var = tk.StringVar(
            value=self.options.self_consistency_convergence_mode
        )
        self.self_consistency_mass_shell_relaxation_var = tk.DoubleVar(
            value=self.options.self_consistency_mass_shell_relaxation
        )
        self.self_consistency_target_ms_tolerance_var = tk.DoubleVar(
            value=self.options.self_consistency_target_ms_tolerance
        )
        self.self_consistency_max_iterations_var = tk.IntVar(
            value=self.options.self_consistency_max_iterations
        )
        self.self_consistency_mass_shell_tolerance_var = tk.DoubleVar(
            value=self.options.self_consistency_mass_shell_tolerance
        )
        self.self_consistency_verbosity_var = tk.IntVar(
            value=getattr(self.options, "self_consistency_verbosity", 2)
        )
        self.self_consistency_chrono_interpolate_var = tk.BooleanVar(
            value=getattr(self.options, "self_consistency_chrono_interpolate", False)
        )
        self.self_consistency_chrono_tolerance_var = tk.DoubleVar(
            value=getattr(self.options, "self_consistency_chrono_tolerance", 1e-3)
        )
        self.self_consistency_chrono_high_precision_var = tk.BooleanVar(
            value=getattr(self.options, "self_consistency_chrono_high_precision", False)
        )
        self.self_consistency_chrono_adaptive_tolerance_var = tk.BooleanVar(
            value=getattr(
                self.options, "self_consistency_chrono_adaptive_tolerance", False
            )
        )
        # Gamma reconciliation options
        self.self_consistency_gamma_reconciliation_method_var = tk.StringVar(
            value=getattr(
                self.options,
                "self_consistency_gamma_reconciliation_method",
                "ADAPTIVE_WEIGHTED",
            )
        )
        self.self_consistency_gamma_reconciliation_low_beta_threshold_var = (
            tk.DoubleVar(
                value=getattr(
                    self.options,
                    "self_consistency_gamma_reconciliation_low_beta_threshold",
                    0.9,
                )
            )
        )
        self.self_consistency_gamma_reconciliation_high_beta_threshold_var = (
            tk.DoubleVar(
                value=getattr(
                    self.options,
                    "self_consistency_gamma_reconciliation_high_beta_threshold",
                    0.99,
                )
            )
        )
        self.self_consistency_gamma_reconciliation_low_beta_weight_var = tk.DoubleVar(
            value=getattr(
                self.options,
                "self_consistency_gamma_reconciliation_low_beta_weight",
                0.8,
            )
        )
        self.self_consistency_gamma_reconciliation_high_beta_weight_var = tk.DoubleVar(
            value=getattr(
                self.options,
                "self_consistency_gamma_reconciliation_high_beta_weight",
                0.2,
            )
        )
        self.self_consistency_gamma_reconciliation_mid_beta_weight_var = tk.DoubleVar(
            value=getattr(
                self.options,
                "self_consistency_gamma_reconciliation_mid_beta_weight",
                0.5,
            )
        )
        self.self_consistency_gamma_reconciliation_fixed_weight_var = tk.DoubleVar(
            value=getattr(
                self.options,
                "self_consistency_gamma_reconciliation_fixed_weight",
                0.5,
            )
        )
        # chrono_matching_mode kept at FAST (internal only, not exposed in GUI)

        # Trace to update control states
        self.self_consistency_enabled_var.trace_add(
            "write", lambda *_: self._toggle_self_consistency_controls()
        )
        self.self_consistency_chrono_interpolate_var.trace_add(
            "write", lambda *_: self._toggle_chrono_controls()
        )

        # Adaptive timestep options (includes halt on jump from removed energy monitor)
        self.adaptive_timestep_enabled_var = tk.BooleanVar(
            value=self.options.adaptive_timestep_enabled
        )
        # Migrate halt_on_jump from removed energy monitor
        self.adaptive_timestep_halt_on_jump_var = tk.BooleanVar(
            value=self.options.energy_monitor_halt_on_jump
        )
        self.adaptive_timestep_threshold_var = tk.DoubleVar(
            value=self.options.adaptive_timestep_threshold
        )
        self.adaptive_timestep_reduction_factor_var = tk.IntVar(
            value=self.options.adaptive_timestep_reduction_factor
        )
        # max_refinement_attempts is now calculated from reduction_factor and min_timestep_factor (read-only display)
        self.adaptive_timestep_max_attempts_display_var = tk.StringVar(value="")
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
        # max_substeps is now calculated from min_timestep_factor (read-only display)
        self.adaptive_timestep_max_substeps_display_var = tk.StringVar(value="")

        # Use preferences for directories
        self.output_dir_var = tk.StringVar(value=self._last_output_dir)
        self.config_dir_var = tk.StringVar(value=self._last_config_dir)
        self.sweep_config_dir_var = tk.StringVar(value=self._last_sweep_config_dir)
        self.sweep_output_dir_var = tk.StringVar(value=self._last_sweep_output_dir)
        self.config_name_var = tk.StringVar(value=self.options.config_name)
        self.config_file_var = tk.StringVar(value="")
        self.sweep_config_name_var = tk.StringVar(value="sweep_config.json")

        # Session-based warning suppression flags
        self._suppress_override_warning = False

        # Performance options
        self.use_numba_var = tk.BooleanVar(
            value=getattr(self.options, "use_numba", True)
        )

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

    def _enforce_panel_minimums(self, event=None):
        """Enforce minimum panel sizes when sash is moved."""
        if not hasattr(self, "_main_horizontal_paned"):
            return

        try:
            # Get current sash position
            sash_pos = self._main_horizontal_paned.sash_coord(0)[0]
            total_width = self._main_horizontal_paned.winfo_width()

            # Calculate minimum and maximum allowed positions
            min_left = CONTENT_PANEL_MIN_WIDTH
            max_left = total_width - CONFIG_PANEL_MIN_WIDTH

            # Enforce limits
            if sash_pos < min_left:
                self._main_horizontal_paned.sash_place(0, min_left, 0)
            elif sash_pos > max_left:
                self._main_horizontal_paned.sash_place(0, max_left, 0)
        except:
            pass  # Ignore errors during layout

    def _create_scrollable_tab(
        self, notebook: ttk.Notebook, title: str, *, padding: int = 12
    ) -> ttk.Frame:
        page = _ScrollableNotebookPage(notebook, title, padding=padding)
        self._scroll_pages.append(page)
        return page.frame

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _load_preferences(self) -> None:
        """Load saved directory preferences or use defaults."""
        # Run defaults
        self._default_config_dir = "configs/run_configs"
        self._default_output_dir = "results/runs"

        # Sweep defaults
        self._default_sweep_config_dir = "configs/sweep_configs"
        self._default_sweep_output_dir = "results/sweeps"

        if self._prefs_file.exists():
            try:
                with open(self._prefs_file, "r") as f:
                    prefs = json.load(f)
                self._last_config_dir = prefs.get(
                    "last_config_dir", self._default_config_dir
                )
                self._last_output_dir = prefs.get(
                    "last_output_dir", self._default_output_dir
                )
                self._last_sweep_config_dir = prefs.get(
                    "last_sweep_config_dir", self._default_sweep_config_dir
                )
                self._last_sweep_output_dir = prefs.get(
                    "last_sweep_output_dir", self._default_sweep_output_dir
                )
            except Exception:
                # If preferences file is corrupted, use defaults
                self._last_config_dir = self._default_config_dir
                self._last_output_dir = self._default_output_dir
                self._last_sweep_config_dir = self._default_sweep_config_dir
                self._last_sweep_output_dir = self._default_sweep_output_dir
        else:
            # First run - use defaults
            self._last_config_dir = self._default_config_dir
            self._last_output_dir = self._default_output_dir
            self._last_sweep_config_dir = self._default_sweep_config_dir
            self._last_sweep_output_dir = self._default_sweep_output_dir

    def _save_preferences(self) -> None:
        """Save current directory preferences."""
        try:
            prefs = {
                "last_config_dir": self._last_config_dir,
                "last_output_dir": self._last_output_dir,
                "last_sweep_config_dir": self._last_sweep_config_dir,
                "last_sweep_output_dir": self._last_sweep_output_dir,
            }
            with open(self._prefs_file, "w") as f:
                json.dump(prefs, f, indent=2)
        except Exception:
            pass  # Silently fail if we can't save preferences

    def _reset_directories_to_defaults(self) -> None:
        """Reset directories to default values."""
        self.config_dir_var.set(self._default_config_dir)
        self.output_dir_var.set(self._default_output_dir)
        self._last_config_dir = self._default_config_dir
        self._last_output_dir = self._default_output_dir
        self._last_sweep_config_dir = self._default_sweep_config_dir
        self._last_sweep_output_dir = self._default_sweep_output_dir
        self._save_preferences()
        self._refresh_config_list()

        # Also update optimization plugin directories if it exists
        if hasattr(self, "optimization_tab"):
            self.optimization_tab.sweep_config_dir = self._default_sweep_config_dir
            self.optimization_tab.sweep_output_dir = self._default_sweep_output_dir

        messagebox.showinfo(
            "Reset Directories",
            "Directories reset to defaults:\n\n"
            f"Run Config: {self._default_config_dir}\n"
            f"Run Output: {self._default_output_dir}\n"
            f"Sweep Config: {self._default_sweep_config_dir}\n"
            f"Sweep Output: {self._default_sweep_output_dir}",
        )

    def _on_close(self) -> None:
        """Handle window close event."""
        self._save_preferences()
        self.root.destroy()

    def _setup_keyboard_fix(self) -> None:
        """Set up keyboard fix for non-US layouts (Swedish, German, etc.).

        This fixes issues where Windows doesn't properly send keyboard layout
        information to Tkinter. We use keycode remapping for Swedish ISO layout.

        Enable debug: LW_KEYBOARD_DEBUG=1 python -m lw_integrator.gui
        """

        # Swedish ISO keyboard keycode mapping (Windows)
        # Maps keycodes to (unshifted_char, shifted_char)
        SWEDISH_KEYMAP = {
            # Number row
            10: ("1", "!"),
            11: ("2", '"'),
            12: ("3", "#"),
            13: ("4", "¤"),  # Currency sign
            14: ("5", "%"),
            15: ("6", "&"),
            16: ("7", "/"),
            17: ("8", "("),
            18: ("9", ")"),
            19: ("0", "="),
            20: ("+", "?"),
            21: ("´", "`"),  # Acute/grave accent
            # Top row
            24: ("q", "Q"),
            25: ("w", "W"),
            26: ("e", "E"),
            27: ("r", "R"),
            28: ("t", "T"),
            29: ("y", "Y"),
            30: ("u", "U"),
            31: ("i", "I"),
            32: ("o", "O"),
            33: ("p", "P"),
            34: ("å", "Å"),
            35: ("¨", "^"),  # Diaeresis/circumflex (dead key)
            # Home row
            38: ("a", "A"),
            39: ("s", "S"),
            40: ("d", "D"),
            41: ("f", "F"),
            42: ("g", "G"),
            43: ("h", "H"),
            44: ("j", "J"),
            45: ("k", "K"),
            46: ("l", "L"),
            47: ("ö", "Ö"),
            48: ("ä", "Ä"),
            49: ("'", "*"),
            # Bottom row
            52: ("z", "Z"),
            53: ("x", "X"),
            54: ("c", "C"),
            55: ("v", "V"),
            56: ("b", "B"),
            57: ("n", "N"),
            58: ("m", "M"),
            59: (",", ";"),
            60: (".", ":"),
            61: ("-", "_"),  # THIS IS THE HYPHEN KEY!
            # Special keys
            65: (" ", " "),  # Space
        }

        def fixed_key_handler(event):
            """Handle keyboard input using keycode remapping for Swedish layout."""
            widget = event.widget
            char = event.char
            keysym = event.keysym
            keycode = event.keycode
            state = event.state

            if self._keyboard_debug:
                widget_name = widget.winfo_name()
                print(f"[KEY] Widget: {widget_name}")
                print(f"      keysym:  {keysym}")
                print(f"      keycode: {keycode}")
                print(f"      char:    {repr(char)} (from OS)")
                print(f"      state:   {state}")

            # Check for modifier keys (except Shift)
            # state & 0x4 = Control, state & 0x8 or 0x20000 = Alt
            has_ctrl = bool(state & 0x4)
            has_alt = bool(state & 0x8 or state & 0x20000)
            has_shift = bool(state & 0x1)

            if has_ctrl or has_alt:
                # Let Tkinter handle Ctrl+X, Alt+X shortcuts
                if self._keyboard_debug:
                    print("      → Passing through (has Ctrl/Alt modifier)")
                    print("-" * 60)
                return None

            # Try Swedish keycode mapping
            correct_char = None
            if keycode in SWEDISH_KEYMAP:
                unshifted, shifted = SWEDISH_KEYMAP[keycode]
                correct_char = shifted if has_shift else unshifted

                if self._keyboard_debug:
                    print(
                        f"      ✓ Swedish keymap: keycode {keycode} → {repr(correct_char)}"
                    )
                    if correct_char != char:
                        print(
                            f"      ⚠ FIXED: OS gave {repr(char)}, using {repr(correct_char)}"
                        )
            else:
                # Not in our mapping - check if it's a control character or letter
                if not char or not char.isprintable():
                    # Control character, navigation key, etc.
                    if self._keyboard_debug:
                        print("      → Passing through (control/special key)")
                        print("-" * 60)
                    return None

                # Unmapped but printable - use what the OS gave us
                correct_char = char
                if self._keyboard_debug:
                    print(f"      ℹ Not in Swedish keymap, using OS char: {repr(char)}")

            if self._keyboard_debug:
                print(f"      ✓ Inserting: {repr(correct_char)}")
                print("-" * 60)

            # Handle Entry widgets
            if isinstance(widget, tk.Entry):
                # Check if text is selected
                try:
                    if widget.selection_present():
                        widget.delete("sel.first", "sel.last")
                except tk.TclError:
                    pass

                # Insert correct character at cursor position
                insert_pos = widget.index("insert")
                widget.insert(insert_pos, correct_char)

                # Return "break" to prevent Tkinter's default (wrong) handler
                return "break"

            # Handle Text widgets
            elif isinstance(widget, tk.Text):
                # Check if text is selected
                try:
                    if widget.tag_ranges("sel"):
                        widget.delete("sel.first", "sel.last")
                except tk.TclError:
                    pass

                # Insert correct character at cursor position
                widget.insert("insert", correct_char)

                # Return "break" to prevent Tkinter's default (wrong) handler
                return "break"

            # For other widgets, let Tkinter handle it
            return None

        # Bind fix handler to all current Entry and Text widgets
        def bind_fix_recursive(widget):
            """Recursively bind keyboard fix to all Entry/Text widgets."""
            if isinstance(widget, (tk.Entry, tk.Text)):
                # Use bindtags to ensure our handler runs BEFORE the class binding
                # Default bindtags order: (widget_name, class_name, toplevel, 'all')
                # We insert a custom tag before the class to intercept events first
                current_tags = list(widget.bindtags())

                # Create a unique tag for this widget
                custom_tag = f"CustomKey{id(widget)}"

                # Insert custom tag before the class name (usually second position)
                if len(current_tags) >= 2:
                    current_tags.insert(1, custom_tag)
                else:
                    current_tags.insert(0, custom_tag)

                widget.bindtags(tuple(current_tags))

                # Bind our handler to the custom tag
                widget.bind_class(custom_tag, "<Key>", fixed_key_handler)

            for child in widget.winfo_children():
                bind_fix_recursive(child)

        bind_fix_recursive(self.root)
        if self._keyboard_debug:
            print(
                "[FIX] Swedish keyboard keycode remapping applied to all text widgets"
            )

    def _build_layout(self) -> None:
        """Build the complete GUI layout with all controls."""
        self.root.rowconfigure(0, weight=0)  # Header row, fixed height
        self.root.rowconfigure(1, weight=1)  # Main content area expands
        self.root.columnconfigure(0, weight=1)

        # Header at top of window (above everything)
        header = ttk.Frame(self.root, padding=8)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        self._driver_entries: List[ttk.Entry] = []

        ttk.Label(header, text="Simulation type:").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        self.sim_type_combo = ttk.Combobox(
            header,
            textvariable=self.sim_type_var,
            state="readonly",
            values=[opt.name for opt in SimulationType],
        )
        self.sim_type_combo.grid(row=0, column=1, sticky="ew")

        # Create main horizontal split: left (tabs) and right (config/control panel)
        self._main_horizontal_paned = tk.PanedWindow(
            self.root,
            orient="horizontal",
            sashrelief="raised",
            sashwidth=8,
            bg="gray70",
        )
        self._main_horizontal_paned.grid(row=1, column=0, sticky="nsew")

        # Left side container for all tabs
        left_container = ttk.Frame(self._main_horizontal_paned)
        left_container.rowconfigure(0, weight=1)
        left_container.columnconfigure(0, weight=1)
        self._main_horizontal_paned.add(left_container, minsize=CONTENT_PANEL_MIN_WIDTH)

        # Vertical split on left side for tabs and logs
        left_vertical_paned = ttk.Panedwindow(left_container, orient="vertical")
        left_vertical_paned.grid(row=0, column=0, sticky="nsew")

        # Notebook for tabs
        self.notebook = ttk.Notebook(left_vertical_paned)
        left_vertical_paned.add(self.notebook, weight=15)

        # Bottom container for logs
        bottom_container = ttk.Frame(left_vertical_paned)
        bottom_container.columnconfigure(0, weight=1)
        bottom_container.rowconfigure(1, weight=1)
        left_vertical_paned.add(bottom_container, weight=1)

        # Particles tab --------------------------------------------------
        particle_frame = self._create_scrollable_tab(
            self.notebook, "Particles", padding=12
        )
        particle_frame.columnconfigure(1, weight=1, minsize=150)
        particle_frame.columnconfigure(3, weight=1, minsize=150)

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

        # Add info note about bunch parameters
        ttk.Label(
            particle_frame,
            text="Note: Particle count, transverse spread, and transverse momentum define the bunch distribution.\n"
            "Transverse offsets (x/y) define bunch center positions and are only used in BUNCH_TO_BUNCH mode.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 2))

        # Track offset entries separately for bunch-to-bunch visibility control
        self._rider_offset_entries = []
        self._driver_offset_entries = []
        self._rider_offset_labels = []
        self._driver_offset_labels = []

        for row, name in enumerate(PARTICLE_PARAM_FIELDS, start=2):
            rider_label = ttk.Label(particle_frame, text=PARAM_LABELS[name] + ":")
            rider_label.grid(row=row, column=0, sticky="w", pady=2)

            rider_entry = ttk.Entry(
                particle_frame, textvariable=self.rider_param_vars[name], width=12
            )
            rider_entry.grid(row=row, column=1, sticky="ew", pady=2)

            driver_label = ttk.Label(
                particle_frame, text=PARAM_LABELS[name] + " (driver):"
            )
            driver_label.grid(row=row, column=2, sticky="w", pady=2, padx=(12, 0))

            driver_entry = ttk.Entry(
                particle_frame, textvariable=self.driver_param_vars[name], width=12
            )
            driver_entry.grid(row=row, column=3, sticky="ew", pady=2)
            self._driver_entries.append(driver_entry)

            # Store offset widget references for bunch-to-bunch visibility control
            if name in ("transv_offset_x", "transv_offset_y"):
                self._rider_offset_entries.append(rider_entry)
                self._rider_offset_labels.append(rider_label)
                self._driver_offset_entries.append(driver_entry)
                self._driver_offset_labels.append(driver_label)

                # Add tooltip for offset fields
                tooltip_text = (
                    "Transverse offset (bunch center position).\n"
                    "Only used in BUNCH_TO_BUNCH simulations.\n\n"
                    "Defines the (x, y) position of the bunch center.\n"
                    "Separation between rider and driver bunches is:\n"
                    "  √[(x_driver - x_rider)² + (y_driver - y_rider)²]"
                )
                Tooltip(rider_entry, tooltip_text)
                Tooltip(driver_entry, tooltip_text)

        # Image subcharge controls
        next_row = len(PARTICLE_PARAM_FIELDS) + 2
        ttk.Label(particle_frame, text="Image subcharge count:").grid(
            row=next_row, column=0, sticky="w", pady=(12, 2)
        )
        self.image_subcharge_entry = ttk.Entry(
            particle_frame, textvariable=self.image_subcharge_var, width=12
        )
        self.image_subcharge_entry.grid(
            row=next_row, column=1, sticky="ew", pady=(12, 2)
        )

        # Help text for image subcharge count
        help_text_subcharge = ttk.Label(
            particle_frame,
            text="(Number of virtual charges used to model conducting-wall images.\n"
            "Range: 4-128. Higher = more accurate but slower. Default: 12)",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        )
        help_text_subcharge.grid(
            row=next_row + 1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        self.image_weighting_check = ttk.Checkbutton(
            particle_frame,
            text="Enable image weighting",
            variable=self.image_weighting_var,
        )
        self.image_weighting_check.grid(
            row=next_row + 2, column=0, columnspan=2, sticky="w", pady=2
        )

        # Help text for image weighting
        help_text_weighting = ttk.Label(
            particle_frame,
            text="(Uses radial weighting when distributing subcharges along aperture.\n"
            "Improves accuracy for aperture geometry. Recommended: enabled)",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        )
        help_text_weighting.grid(
            row=next_row + 3, column=0, columnspan=2, sticky="w", pady=(0, 2)
        )

        # Macroparticle simulation section
        next_row += 4
        ttk.Separator(particle_frame, orient="horizontal").grid(
            row=next_row, column=0, columnspan=4, sticky="ew", pady=(12, 12)
        )
        next_row += 1

        ttk.Label(
            particle_frame,
            text="Macroparticle Simulation (Conducting Wall only):",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=next_row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        next_row += 1

        self.macroparticle_enable_check = ttk.Checkbutton(
            particle_frame,
            text="Enable macroparticle simulation (bunch spread inherited from above)",
            variable=self.macroparticle_enabled_var,
            command=self._toggle_macroparticle_controls,
        )
        self.macroparticle_enable_check.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=2
        )
        next_row += 1

        # Charge multiplier
        self.macroparticle_charge_label = ttk.Label(
            particle_frame, text="Charge multiplier:"
        )
        self.macroparticle_charge_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.macroparticle_charge_entry = ttk.Entry(
            particle_frame,
            textvariable=self.macroparticle_charge_multiplier_var,
            width=12,
        )
        self.macroparticle_charge_entry.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        # Sigma multiplier for image charge errors
        self.macroparticle_sigma_label = ttk.Label(
            particle_frame, text="Image error sigma multiplier:"
        )
        self.macroparticle_sigma_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.macroparticle_sigma_entry = ttk.Entry(
            particle_frame,
            textvariable=self.macroparticle_sigma_multiplier_var,
            width=12,
        )
        self.macroparticle_sigma_entry.grid(row=next_row, column=1, sticky="ew", pady=2)
        next_row += 1

        # Include momentum errors checkbox
        self.macroparticle_momentum_errors_check = ttk.Checkbutton(
            particle_frame,
            text="Include momentum errors (cumulative)",
            variable=self.macroparticle_use_momentum_errors_var,
        )
        self.macroparticle_momentum_errors_check.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )
        next_row += 1

        # Help text for macroparticle
        help_text_macroparticle = ttk.Label(
            particle_frame,
            text=(
                "Macroparticle mode scales particle charge and adds Gaussian errors to image subcharges.\n"
                "Image errors are derived from bunch spread parameters (transv_dist, transv_mom) × sigma multiplier.\n"
                "Position errors: constant σ from transv_dist. Momentum errors: cumulative from transv_mom.\n"
                "Uncheck 'Include momentum errors' to apply only constant position errors (no cumulative growth).\n"
                "Only active for CONDUCTING_WALL simulations."
            ),
            font=("TkDefaultFont", 8),
            foreground="gray40",
            justify="left",
        )
        help_text_macroparticle.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=(0, 2), padx=(20, 0)
        )

        # Store macroparticle widgets for enable/disable
        self._macroparticle_widgets = [
            self.macroparticle_charge_label,
            self.macroparticle_charge_entry,
            self.macroparticle_sigma_label,
            self.macroparticle_sigma_entry,
            self.macroparticle_momentum_errors_check,
        ]

        # Core tab ------------------------------------------------------
        core_frame = self._create_scrollable_tab(
            self.notebook, "Core params", padding=12
        )
        core_frame.columnconfigure(1, weight=1)

        # Store widgets for dynamic enable/disable
        self.core_param_widgets = {}

        row = 0

        # Steps and Seed at the top of Core params
        ttk.Label(core_frame, text="Steps:").grid(row=row, column=0, sticky="w", pady=2)
        steps_widget = ttk.Entry(core_frame, textvariable=self.steps_var, width=16)
        steps_widget.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(core_frame, text="Seed:").grid(row=row, column=0, sticky="w", pady=2)
        seed_frame = ttk.Frame(core_frame)
        seed_frame.grid(row=row, column=1, sticky="ew", pady=2)
        seed_frame.columnconfigure(0, weight=1)

        self.seed_entry = ttk.Entry(seed_frame, textvariable=self.seed_var, width=16)
        self.seed_entry.grid(row=0, column=0, sticky="ew")

        self.random_seed_var = tk.BooleanVar(value=False)
        self.random_seed_check = ttk.Checkbutton(
            seed_frame,
            text="Random",
            variable=self.random_seed_var,
            command=self._toggle_random_seed,
        )
        self.random_seed_check.grid(row=0, column=1, sticky="w", padx=(5, 0))
        row += 1

        # Separator after steps/seed
        ttk.Separator(core_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(10, 10)
        )
        row += 1

        for name in CORE_PARAM_LABELS:
            # Skip z_cutoff and z_cutoff_mode - handled separately below
            # Skip mean - deprecated parameter, not used in any simulation mode
            # Skip startup_mode - handled separately below with combobox
            if name in ["z_cutoff", "z_cutoff_mode", "mean", "startup_mode"]:
                continue

            ttk.Label(core_frame, text=CORE_PARAM_LABELS[name] + ":").grid(
                row=row, column=0, sticky="w", pady=2
            )

            # Grey out cavity_spacing unless SWITCHING_WALL mode
            widget = ttk.Entry(
                core_frame, textvariable=self.core_param_vars[name], width=16
            )
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            self.core_param_widgets[name] = widget
            row += 1

        # Startup mode section
        ttk.Separator(core_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(10, 10)
        )
        row += 1

        ttk.Label(core_frame, text="Startup mode:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        startup_mode_combo = ttk.Combobox(
            core_frame,
            textvariable=self.core_param_vars["startup_mode"],
            values=["COLD_START", "APPROXIMATE_BACK_HISTORY"],
            state="readonly",
            width=22,
        )
        startup_mode_combo.grid(row=row, column=1, sticky="ew", pady=2)
        self.core_param_widgets["startup_mode"] = startup_mode_combo

        # Add informative tooltip for startup_mode
        Tooltip(
            startup_mode_combo,
            "Startup mode controls retarded force calculation at early timesteps.\n\n"
            "COLD_START (default, recommended):\n"
            "  • Suppresses retarded forces until particles build causal history\n"
            "  • Physically realistic for transient events (beam turn-on)\n"
            "  • Avoids unphysical extrapolation errors\n"
            "  • Compatible with all features (adaptive timestep, energy monitor)\n"
            "  • May show startup transient in first ~100 steps\n\n"
            "APPROXIMATE_BACK_HISTORY (experimental, benchmarking only):\n"
            "  • Assumes particles had constant velocity since t = -∞\n"
            "  • Enables immediate force calculation (no gating)\n"
            "  • Use ONLY for comparison with legacy solvers\n"
            "  • Not validated for production physics\n"
            "  • May introduce unphysical initial conditions\n\n"
            "For production: use COLD_START\n"
            "For legacy benchmarking: use APPROXIMATE_BACK_HISTORY",
        )
        row += 1

        # Brief inline help text for startup_mode
        startup_help_label = ttk.Label(
            core_frame,
            text="COLD_START (recommended): Forces gated until causal history available\n"
            "APPROXIMATE_BACK_HISTORY: Constant-velocity extrapolation (benchmarking only)",
            foreground="gray",
            font=("TkDefaultFont", 8),
            justify="left",
            wraplength=450,
        )
        startup_help_label.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5)
        )
        row += 1

        # Z-cutoff section with enable checkbox
        ttk.Separator(core_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(10, 10)
        )
        row += 1

        ttk.Label(
            core_frame,
            text="Force Cutoff (optional):",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1

        self.z_cutoff_enable_check = ttk.Checkbutton(
            core_frame,
            text="Enable z-cutoff (stops applying external forces when particle z > cutoff)",
            variable=self.z_cutoff_enabled_var,
            command=self._toggle_z_cutoff_controls,
        )
        self.z_cutoff_enable_check.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1

        # z_cutoff value entry (indented)
        self.z_cutoff_label = ttk.Label(core_frame, text="z cutoff (mm):")
        self.z_cutoff_label.grid(row=row, column=0, sticky="w", pady=2, padx=(20, 0))
        self.z_cutoff_entry = ttk.Entry(
            core_frame, textvariable=self.core_param_vars["z_cutoff"], width=16
        )
        self.z_cutoff_entry.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        # z_cutoff_mode combobox (indented)
        self.z_cutoff_mode_label = ttk.Label(core_frame, text="Reference:")
        self.z_cutoff_mode_label.grid(
            row=row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.z_cutoff_mode_combo = ttk.Combobox(
            core_frame,
            textvariable=self.core_param_vars["z_cutoff_mode"],
            values=["absolute", "relative"],
            state="readonly",
            width=14,
        )
        self.z_cutoff_mode_combo.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        # Help text for z_cutoff
        help_label = ttk.Label(
            core_frame,
            text="'absolute': fixed z position in lab frame\n"
            "'relative': distance from particle starting position\n\n"
            "Works in all simulation modes (BUNCH_TO_BUNCH, CONDUCTING_WALL, SWITCHING_WALL).\n"
            "In SWITCHING_WALL mode with cavity_spacing > 0, cutoff advances by cavity_spacing\n"
            "when particle passes threshold, creating periodic cavity structure.",
            foreground="gray",
            font=("TkDefaultFont", 8),
            justify="left",
            wraplength=450,
        )
        help_label.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5), padx=(20, 0)
        )

        # Outputs tab ---------------------------------------------------
        output_frame = self._create_scrollable_tab(self.notebook, "Output", padding=12)
        output_frame.columnconfigure(1, weight=1)

        # Notice about single run vs sweep/optimization
        notice_frame = ttk.Frame(output_frame)
        notice_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 15))

        notice_label = ttk.Label(
            notice_frame,
            text="⚠ These settings apply to SINGLE RUNS only.\nFor sweep/optimization output configuration, see the 'Sweep/Optim' tab → 'Results & Output Configuration'.",
            font=("TkDefaultFont", 9, "bold"),
            foreground="blue",
            justify="left",
        )
        notice_label.pack(anchor="w")

        # Legacy comparison toggle (moved from header)
        ttk.Checkbutton(
            output_frame, text="Enable legacy comparison", variable=self.legacy_var
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        # Trajectory comparison outputs (grouped and dependent on legacy)
        comparison_frame = ttk.LabelFrame(
            output_frame, text="Trajectory Comparison (requires legacy)", padding=8
        )
        comparison_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
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
            row=2,
        )
        # Energy plot x-axis configuration
        ttk.Label(output_frame, text="  ↳ X-axis:").grid(
            row=4, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.energy_xaxis_var,
            values=["z", "t", "dual"],
            width=12,
            state="readonly",
        ).grid(row=3, column=1, sticky="w")

        # Energy plot y-axis configuration
        ttk.Label(output_frame, text="  ↳ Y-axis:").grid(
            row=5, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.energy_yaxis_var,
            values=["delta_total", "delta_z", "delta_x", "delta_y", "total"],
            width=12,
            state="readonly",
        ).grid(row=5, column=1, sticky="w")
        self._add_output_toggle(
            output_frame,
            "Transverse position (⟨x⟩, ⟨y⟩)",
            self.transverse_display_var,
            self.transverse_save_var,
            row=5,
        )
        # Transverse plot x-axis configuration
        ttk.Label(output_frame, text="  ↳ X-axis:").grid(
            row=7, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.transverse_xaxis_var,
            values=["t", "z"],
            width=5,
            state="readonly",
        ).grid(row=7, column=1, sticky="w")

        self._add_output_toggle(
            output_frame,
            "Velocity (β_x, β_y, β_z, |β|)",
            self.beta_display_var,
            self.beta_save_var,
            row=7,
        )
        # Beta plot x-axis configuration
        ttk.Label(output_frame, text="  ↳ X-axis:").grid(
            row=9, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.beta_xaxis_var,
            values=["t", "z"],
            width=5,
            state="readonly",
        ).grid(row=9, column=1, sticky="w")

        self._add_output_toggle(
            output_frame,
            "Conjugate momentum (Pˣ, Pʸ, Pᶻ, |P⊥|, Pᵗ, |P|)",
            self.momentum_display_var,
            self.momentum_save_var,
            row=9,
        )
        # Momentum plot x-axis configuration
        ttk.Label(output_frame, text="  ↳ X-axis:").grid(
            row=11, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.momentum_xaxis_var,
            values=["t", "z"],
            width=5,
            state="readonly",
        ).grid(row=11, column=1, sticky="w")

        # Gamma (Lorentz factor) plot
        self._add_output_toggle(
            output_frame,
            "Gamma (Lorentz factor γ)",
            self.gamma_display_var,
            self.gamma_save_var,
            row=12,
        )
        # Gamma plot x-axis configuration
        ttk.Label(output_frame, text="  ↳ X-axis:").grid(
            row=13, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.gamma_xaxis_var,
            values=["t", "z"],
            width=5,
            state="readonly",
        ).grid(row=13, column=1, sticky="w")

        # Separator for position plots
        ttk.Separator(output_frame, orient="horizontal").grid(
            row=14, column=0, columnspan=2, sticky="ew", pady=(10, 10)
        )

        # Z-position vs time plot
        ttk.Label(output_frame, text="Longitudinal trajectory:").grid(
            row=15, column=0, columnspan=2, sticky="w", pady=(0, 2)
        )
        self._add_output_toggle(
            output_frame,
            "  z vs time",
            self.zposition_display_var,
            self.zposition_save_var,
            row=16,
        )

        # Separator before trajectory/output options
        ttk.Separator(output_frame, orient="horizontal").grid(
            row=17, column=0, columnspan=2, sticky="ew", pady=(10, 10)
        )

        ttk.Label(output_frame, text="Plot DPI:").grid(row=18, column=0, sticky="w")
        ttk.Combobox(
            output_frame,
            textvariable=self.dpi_var,
            values=[str(dpi) for dpi in AVAILABLE_DPI_CHOICES],
            width=8,
            state="readonly",
        ).grid(row=18, column=1, sticky="w")

        # Trajectory data saving
        ttk.Label(
            output_frame, text="Trajectory Data:", font=("TkDefaultFont", 9, "bold")
        ).grid(row=19, column=0, columnspan=2, sticky="w", pady=(12, 2))

        ttk.Checkbutton(
            output_frame,
            text="Save trajectory data (NPZ + JSON formats)",
            variable=self.trajectory_save_var,
            command=self._on_trajectory_save_toggled,
        ).grid(row=20, column=0, columnspan=2, sticky="w", pady=(0, 2))

        self.trajectory_stride_label = ttk.Label(
            output_frame, text="Trajectory stride:"
        )
        self.trajectory_stride_label.grid(row=21, column=0, sticky="w", padx=(20, 0))
        self.trajectory_stride_entry = ttk.Entry(
            output_frame, textvariable=self.trajectory_interval_var, width=8
        )
        self.trajectory_stride_entry.grid(row=21, column=1, sticky="w")

        ttk.Label(
            output_frame,
            text="(Save every Nth point to reduce file size)",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray50",
        ).grid(row=22, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=(0, 10))

        # Initialize trajectory stride state
        self._on_trajectory_save_toggled()

        # Log file saving
        ttk.Label(
            output_frame, text="Debug Logs:", font=("TkDefaultFont", 9, "bold")
        ).grid(row=23, column=0, columnspan=2, sticky="w", pady=(5, 2))

        ttk.Checkbutton(
            output_frame,
            text="Save debug log file to output directory",
            variable=self.save_log_file_var,
        ).grid(row=24, column=0, columnspan=2, sticky="w", pady=(0, 2))

        ttk.Label(
            output_frame,
            text="(Captures console output, warnings, and diagnostic info)",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray50",
        ).grid(row=25, column=0, columnspan=2, sticky="w", padx=(20, 0))

        # Stability Settings tab ----------------------------------------
        stability_frame = self._create_scrollable_tab(
            self.notebook, "Stability", padding=12
        )
        stability_frame.columnconfigure(1, weight=1)

        # Notice about single run vs sweep/optimization
        stability_notice_frame = ttk.Frame(stability_frame)
        stability_notice_frame.grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 15)
        )

        stability_notice_label = ttk.Label(
            stability_notice_frame,
            text="⚠ These settings apply to BOTH single runs AND sweeps/optimizations.\nStability controls affect all simulation modes.",
            font=("TkDefaultFont", 9, "bold"),
            foreground="blue",
            justify="left",
        )
        stability_notice_label.pack(anchor="w")

        # Self-consistency section
        sc_frame = ttk.LabelFrame(
            stability_frame, text="Self-Consistency Checks", padding=8
        )
        sc_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        sc_frame.columnconfigure(1, weight=1)

        self.sc_enable_check = ttk.Checkbutton(
            sc_frame,
            text="Enable self-consistency iterations (recommended)",
            variable=self.self_consistency_enabled_var,
            command=self._toggle_self_consistency_controls,
        )
        self.sc_enable_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        # Convergence mode dropdown
        mode_frame = ttk.Frame(sc_frame)
        mode_frame.grid(row=1, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_mode_label = ttk.Label(mode_frame, text="Convergence mode:")
        self.sc_mode_label.pack(side="left")
        mode_help = ttk.Label(mode_frame, text="ⓘ", foreground="blue", cursor="hand2")
        mode_help.pack(side="left", padx=(3, 0))
        Tooltip(
            mode_help,
            "Self-consistency convergence mode.\n\n"
            "• Fixed Geometry (default, fastest):\n"
            "  Projects Pt onto mass shell each iteration\n"
            "  Geometry computed once per timestep\n"
            "  Use for most cases\n"
            "  Speed: 1× baseline, 2-5 iterations typical\n\n"
            "• Variable Geometry (accurate, slower):\n"
            "  Projects Pt onto mass shell each iteration\n"
            "  Recomputes geometry each SC iteration\n"
            "  Use when particle moves significantly: |Δx| ~ 0.1×R\n"
            "  Speed: 2-10× slower than fixed\n\n"
            "Default: Fixed Geometry",
        )
        self.sc_mode_combo = ttk.Combobox(
            sc_frame,
            textvariable=self.self_consistency_convergence_mode_var,
            values=[
                "fixed_geometry",
                "variable_geometry",
            ],
            state="readonly",
            width=20,
        )
        self.sc_mode_combo.grid(row=1, column=1, sticky="ew", pady=2)
        self.sc_mode_combo.bind("<<ComboboxSelected>>", self._on_sc_mode_changed)

        # Target mass-shell tolerance
        target_ms_frame = ttk.Frame(sc_frame)
        target_ms_frame.grid(row=2, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_target_ms_tolerance_label = ttk.Label(
            target_ms_frame, text="Target MS tolerance:"
        )
        self.sc_target_ms_tolerance_label.pack(side="left")
        target_ms_help = ttk.Label(
            target_ms_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        target_ms_help.pack(side="left", padx=(3, 0))
        Tooltip(
            target_ms_help,
            "TARGET mass-shell convergence criterion.\n\n"
            "Loop continues until:\n"
            "  |Pt² - P² - (mc)²|/(mc)² < target_ms_tolerance\n\n"
            "This ensures energy-momentum relation is satisfied.\n\n"
            "Default: 1e-6 (0.0001% relative error)\n"
            "Aggressive: 1e-8 for ultra-relativistic (γ > 1000)\n"
            "Minimum: 1e-10 (stricter = more iterations)",
        )
        self.sc_target_ms_tolerance_entry = ttk.Entry(
            sc_frame,
            textvariable=self.self_consistency_target_ms_tolerance_var,
            width=16,
        )
        self.sc_target_ms_tolerance_entry.grid(row=2, column=1, sticky="ew", pady=2)

        # Target gamma tolerance
        # Max iterations with help icon
        max_iter_frame = ttk.Frame(sc_frame)
        max_iter_frame.grid(row=4, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_max_iterations_label = ttk.Label(max_iter_frame, text="Max iterations:")
        self.sc_max_iterations_label.pack(side="left")
        iter_help = ttk.Label(
            max_iter_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        iter_help.pack(side="left", padx=(3, 0))
        Tooltip(
            iter_help,
            "Maximum self-consistency iterations per particle per step.\n\n"
            "Loop continues until:\n"
            "  • Mass-shell constraint satisfied (momentum-based check), OR\n"
            "  • Max iterations reached (applies projection as fallback)\n\n"
            "More iterations = better accuracy but slower.\n"
            "Typical convergence: 2-4 iterations\n\n"
            "Default: 10\n"
            "Aggressive: 20 for ultra-relativistic particles (γ > 1000)\n"
            "Increase if seeing 'max iterations reached' warnings",
        )
        self.sc_max_iterations_entry = ttk.Entry(
            sc_frame, textvariable=self.self_consistency_max_iterations_var, width=16
        )
        self.sc_max_iterations_entry.grid(row=4, column=1, sticky="ew", pady=2)

        # Mass-shell tolerance (safety net) with help icon
        ms_frame = ttk.Frame(sc_frame)
        ms_frame.grid(row=5, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_mass_shell_tolerance_label = ttk.Label(
            ms_frame, text="Mass-shell tolerance:"
        )
        self.sc_mass_shell_tolerance_label.pack(side="left")
        ms_help = ttk.Label(ms_frame, text="ⓘ", foreground="blue", cursor="hand2")
        ms_help.pack(side="left", padx=(3, 0))
        Tooltip(
            ms_help,
            "SAFETY NET threshold enforced after iteration loop.\n\n"
            "PHYSICS: Pt² - P² = (mc)² must hold exactly.\n"
            "Numerical errors can violate this slightly.\n\n"
            "When |Pt² - P² - (mc)²|/(mc)² > tolerance:\n"
            "  → Pt is clamped to √(P² + (mc)²) as fallback\n\n"
            "This acts as a final safety net AFTER the loop.\n"
            "Should be LOOSER (larger) than target_tolerance.\n\n"
            "Default: 0.01 (1% relative error)\n"
            "Stricter: 0.001 for ultra-relativistic (γ > 1000)\n"
            "Disable: 1e10 (not recommended)\n\n"
            "Typical errors after convergence: 1e-12 to 1e-8 (rarely triggers)",
        )
        self.sc_mass_shell_tolerance_entry = ttk.Entry(
            sc_frame,
            textvariable=self.self_consistency_mass_shell_tolerance_var,
            width=16,
        )
        self.sc_mass_shell_tolerance_entry.grid(row=5, column=1, sticky="ew", pady=2)

        # Mass-shell relaxation with help icon
        relaxation_frame = ttk.Frame(sc_frame)
        relaxation_frame.grid(row=6, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_relaxation_label = ttk.Label(
            relaxation_frame, text="Relaxation weight:"
        )
        self.sc_relaxation_label.pack(side="left")
        relaxation_help = ttk.Label(
            relaxation_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        relaxation_help.pack(side="left", padx=(3, 0))
        Tooltip(
            relaxation_help,
            "Relaxation weight applied after Pt correction (both modes).\n\n"
            "Prevents oscillations by damping the correction:\n"
            "  Pt_final = α*Pt_corrected + (1-α)*Pt_old\n"
            "  where α = relaxation weight\n\n"
            "Values:\n"
            "  • 1.0 = Full correction (fastest, may oscillate)\n"
            "  • 0.7 = Recommended (default, good balance)\n"
            "  • 0.5 = Conservative (more stable, slower)\n"
            "  • 0.0 = No correction (broken, testing only)\n\n"
            "Increase for ultra-relativistic (γ > 1000) particles.\n"
            "Decrease if seeing convergence oscillations.\n\n"
            "Default: 0.7",
        )
        self.sc_relaxation_entry = ttk.Entry(
            sc_frame,
            textvariable=self.self_consistency_mass_shell_relaxation_var,
            width=16,
        )
        self.sc_relaxation_entry.grid(row=6, column=1, sticky="ew", pady=2)

        # Verbosity with help icon
        verbosity_frame = ttk.Frame(sc_frame)
        verbosity_frame.grid(row=7, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_verbosity_label = ttk.Label(verbosity_frame, text="Verbosity:")
        self.sc_verbosity_label.pack(side="left")
        verbosity_help = ttk.Label(
            verbosity_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        verbosity_help.pack(side="left", padx=(3, 0))
        Tooltip(
            verbosity_help,
            "Self-consistency convergence diagnostic output level.\n\n"
            "⚠️ APPLIES TO ALL MODES: single runs AND sweeps/optimizations\n\n"
            "For Sweep/Optimization:\n"
            "  • This verbosity level is INHERITED when Log verbosity = 'full'\n"
            "  • Set 'Log verbosity' in Sweep/Optim tab to control overall logging\n"
            "  • 'full' mode: uses this verbosity level for SC diagnostics\n"
            "  • 'truncated' mode: minimal output regardless of this setting\n\n"
            "Output is printed to BOTH:\n"
            "  • Console (real-time during run)\n"
            "  • Saved verbose log file (*_verbose.txt)\n\n"
            "Levels:\n"
            "  • 0 = Silent (no convergence details)\n"
            "  • 1 = Summary (one line per step: converged/failed)\n"
            "  • 2 = Failures only (detailed output only for non-converged steps)\n"
            "  • 3 = Full detail (iteration-by-iteration for all steps)\n\n"
            "Recommended:\n"
            "  • Production runs: 0 or 1 (small logs)\n"
            "  • Debugging: 2 (shows only problems, moderate logs)\n"
            "  • Deep diagnostics: 3 (very large logs, 100k+ lines)\n\n"
            "Default: 0 (silent)",
        )
        self.sc_verbosity_entry = ttk.Spinbox(
            sc_frame,
            from_=0,
            to=3,
            textvariable=self.self_consistency_verbosity_var,
            width=5,
        )
        self.sc_verbosity_entry.grid(row=7, column=1, sticky="w", pady=2)

        # Chrono-match interpolation with help icon
        chrono_interp_frame = ttk.Frame(sc_frame)
        chrono_interp_frame.grid(
            row=8, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )
        self.sc_chrono_interpolate_check = ttk.Checkbutton(
            chrono_interp_frame,
            text="Enable chrono-match interpolation",
            variable=self.self_consistency_chrono_interpolate_var,
            command=self._toggle_chrono_controls,
        )
        self.sc_chrono_interpolate_check.pack(side="left")
        chrono_interp_help = ttk.Label(
            chrono_interp_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        chrono_interp_help.pack(side="left", padx=(3, 0))
        Tooltip(
            chrono_interp_help,
            "Interpolate source particle state when retarded-time residual exceeds tolerance.\n\n"
            "When computing Liénard-Wiechert fields, the code searches backward through\n"
            "the source particle trajectory to find t_ret = t_obs - R/c. With coarse\n"
            "timesteps, the 'nearest' match may have significant time residual.\n\n"
            "When enabled:\n"
            "  • Computes time residual |t_matched - t_target|\n"
            "  • If residual > tolerance, linearly interpolates source quantities\n"
            "    (velocity, acceleration, gamma) between bracketing trajectory points\n"
            "  • Provides sub-timestep accuracy for retarded fields\n\n"
            "When to enable:\n"
            "  • Large timesteps relative to 1/γ characteristic time\n"
            "  • Ultra-relativistic simulations (γ > 100)\n"
            "  • Self-consistency failures related to field discontinuities\n"
            "  • Image-charge singularities\n\n"
            "Performance impact: ~1-2% overhead (minimal)\n\n"
            "Default: OFF (preserves legacy behavior)",
        )

        # Chrono tolerance with help icon
        chrono_tol_frame = ttk.Frame(sc_frame)
        chrono_tol_frame.grid(row=9, column=0, sticky="w", pady=2, padx=(40, 0))
        self.sc_chrono_tolerance_label = ttk.Label(
            chrono_tol_frame, text="Chrono tolerance (ns):"
        )
        self.sc_chrono_tolerance_label.pack(side="left")
        chrono_tol_help = ttk.Label(
            chrono_tol_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        chrono_tol_help.pack(side="left", padx=(3, 0))
        Tooltip(
            chrono_tol_help,
            "Time residual tolerance for chrono-matching (nanoseconds).\n\n"
            "If |t_matched - t_target| > chrono_tolerance, interpolation is applied\n"
            "(if chrono_interpolate is enabled) or a warning is issued (if verbosity >= 2).\n\n"
            "Typical values:\n"
            "  • 1e-3 ns (1 ps): Default, good for most simulations\n"
            "  • 5e-4 ns (0.5 ps): Tighter tolerance for high-precision work\n"
            "  • 1e-4 ns (0.1 ps): Very tight, for ultra-relativistic particles\n\n"
            "Rule of thumb: Set to ~0.1 × average_timestep\n\n"
            "Default: 1e-3 ns (1 picosecond)",
        )
        self.sc_chrono_tolerance_entry = ttk.Entry(
            sc_frame,
            textvariable=self.self_consistency_chrono_tolerance_var,
            width=16,
        )
        self.sc_chrono_tolerance_entry.grid(row=9, column=1, sticky="w", pady=2)

        # Advanced chrono options (high-precision mode)
        chrono_highprec_frame = ttk.Frame(sc_frame)
        chrono_highprec_frame.grid(
            row=10, column=0, columnspan=2, sticky="w", pady=2, padx=(40, 0)
        )
        self.sc_chrono_high_precision_check = ttk.Checkbutton(
            chrono_highprec_frame,
            text="High-precision mode (cubic + position interpolation)",
            variable=self.self_consistency_chrono_high_precision_var,
        )
        self.sc_chrono_high_precision_check.pack(side="left")
        chrono_highprec_help = ttk.Label(
            chrono_highprec_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        chrono_highprec_help.pack(side="left", padx=(3, 0))
        Tooltip(
            chrono_highprec_help,
            "Enable high-precision chrono-matching features.\n\n"
            "When enabled:\n"
            "  • Uses cubic (Catmull-Rom) interpolation instead of linear\n"
            "  • Interpolates particle positions (x/y/z) in addition to velocities\n"
            "  • Provides smoother derivatives for acceleration terms\n"
            "  • Better accuracy for ultra-relativistic particles (γ > 1000)\n\n"
            "Performance impact:\n"
            "  • ~3-5% overhead vs linear interpolation\n"
            "  • Requires at least 4 trajectory points for cubic fit\n\n"
            "When to enable:\n"
            "  • γ > 1000 with coarse timesteps\n"
            "  • Need smooth βdot derivatives\n"
            "  • Critical accuracy requirements\n\n"
            "Default: OFF (linear interpolation is usually sufficient)",
        )

        # Adaptive tolerance
        chrono_adaptive_frame = ttk.Frame(sc_frame)
        chrono_adaptive_frame.grid(
            row=11, column=0, columnspan=2, sticky="w", pady=2, padx=(40, 0)
        )
        self.sc_chrono_adaptive_check = ttk.Checkbutton(
            chrono_adaptive_frame,
            text="Adaptive tolerance (auto-scale with timestep)",
            variable=self.self_consistency_chrono_adaptive_tolerance_var,
        )
        self.sc_chrono_adaptive_check.pack(side="left")
        chrono_adaptive_help = ttk.Label(
            chrono_adaptive_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        chrono_adaptive_help.pack(side="left", padx=(3, 0))
        Tooltip(
            chrono_adaptive_help,
            "Automatically set chrono tolerance based on timestep.\n\n"
            "Formula: tolerance = 0.1 × timestep_h\n\n"
            "When enabled:\n"
            "  • Overrides manual chrono_tolerance setting\n"
            "  • Scales tolerance with integration timestep\n"
            "  • Useful for variable-timestep simulations\n\n"
            "Example:\n"
            "  • h = 1e-3 ns → tolerance = 1e-4 ns\n"
            "  • h = 5e-4 ns → tolerance = 5e-5 ns\n\n"
            "Default: OFF (use fixed tolerance)",
        )

        # Note: chrono_matching_mode removed from GUI
        # Always uses FAST mode (legacy behavior)
        # AVERAGED mode reserved for future APPROXIMATE_BACK_HISTORY implementation

        # Gamma reconciliation
        gamma_recon_frame = ttk.LabelFrame(
            sc_frame, text="Gamma Reconciliation", padding=8
        )
        gamma_recon_frame.grid(
            row=12, column=0, columnspan=2, sticky="ew", pady=2, padx=(20, 0)
        )
        gamma_recon_frame.columnconfigure(1, weight=1)

        # Method selection
        method_frame = ttk.Frame(gamma_recon_frame)
        method_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(method_frame, text="Method:").pack(side="left")
        self.sc_gamma_reconciliation_method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.self_consistency_gamma_reconciliation_method_var,
            values=[
                "DISABLED",
                "ADAPTIVE_WEIGHTED",
                "USE_VELOCITY",
                "USE_ENERGY",
                "FIXED_WEIGHTED",
            ],
            state="readonly",
            width=20,
        )
        self.sc_gamma_reconciliation_method_combo.pack(side="left", padx=(5, 5))
        method_help = ttk.Label(
            method_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        method_help.pack(side="left")
        Tooltip(
            method_help,
            "Gamma Reconciliation Method:\n\n"
            "DISABLED - No reconciliation (legacy, may cause blowups)\n\n"
            "ADAPTIVE_WEIGHTED - Velocity-dependent weighting (recommended)\n"
            "  • β < 0.9: Trust energy (weight=0.8)\n"
            "  • β > 0.99: Trust velocity (weight=0.2)\n"
            "  • Mid-range: Balanced (weight=0.5)\n\n"
            "USE_VELOCITY - Always use γ from β (breaks energy)\n\n"
            "USE_ENERGY - Always use γ from Pt (legacy)\n\n"
            "FIXED_WEIGHTED - Fixed 50/50 blend\n\n"
            "Recommended: ADAPTIVE_WEIGHTED",
        )

        # Adaptive weighted parameters
        self.sc_gamma_reconciliation_adaptive_frame = ttk.LabelFrame(
            gamma_recon_frame, text="Adaptive Weighted Parameters", padding=6
        )
        self.sc_gamma_reconciliation_adaptive_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(5, 2), padx=(10, 0)
        )
        self.sc_gamma_reconciliation_adaptive_frame.columnconfigure(1, weight=1)

        # Low beta threshold
        ttk.Label(
            self.sc_gamma_reconciliation_adaptive_frame, text="Low β threshold:"
        ).grid(row=0, column=0, sticky="w", pady=2)
        self.sc_gamma_low_beta_threshold_entry = ttk.Entry(
            self.sc_gamma_reconciliation_adaptive_frame,
            textvariable=self.self_consistency_gamma_reconciliation_low_beta_threshold_var,
            width=10,
        )
        self.sc_gamma_low_beta_threshold_entry.grid(
            row=0, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_low_beta_threshold_entry,
            "Velocity below which energy is trusted more\nDefault: 0.9",
        )

        # High beta threshold
        ttk.Label(
            self.sc_gamma_reconciliation_adaptive_frame, text="High β threshold:"
        ).grid(row=1, column=0, sticky="w", pady=2)
        self.sc_gamma_high_beta_threshold_entry = ttk.Entry(
            self.sc_gamma_reconciliation_adaptive_frame,
            textvariable=self.self_consistency_gamma_reconciliation_high_beta_threshold_var,
            width=10,
        )
        self.sc_gamma_high_beta_threshold_entry.grid(
            row=1, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_high_beta_threshold_entry,
            "Velocity above which velocity is trusted more\nDefault: 0.99",
        )

        # Low beta weight
        ttk.Label(
            self.sc_gamma_reconciliation_adaptive_frame, text="Low β weight (α):"
        ).grid(row=2, column=0, sticky="w", pady=2)
        self.sc_gamma_low_beta_weight_entry = ttk.Entry(
            self.sc_gamma_reconciliation_adaptive_frame,
            textvariable=self.self_consistency_gamma_reconciliation_low_beta_weight_var,
            width=10,
        )
        self.sc_gamma_low_beta_weight_entry.grid(
            row=2, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_low_beta_weight_entry,
            "Weight for energy when β < low threshold\n"
            "γ = α·γ_energy + (1-α)·γ_velocity\nDefault: 0.8",
        )

        # High beta weight
        ttk.Label(
            self.sc_gamma_reconciliation_adaptive_frame, text="High β weight (α):"
        ).grid(row=3, column=0, sticky="w", pady=2)
        self.sc_gamma_high_beta_weight_entry = ttk.Entry(
            self.sc_gamma_reconciliation_adaptive_frame,
            textvariable=self.self_consistency_gamma_reconciliation_high_beta_weight_var,
            width=10,
        )
        self.sc_gamma_high_beta_weight_entry.grid(
            row=3, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_high_beta_weight_entry,
            "Weight for energy when β > high threshold\n"
            "γ = α·γ_energy + (1-α)·γ_velocity\nDefault: 0.2",
        )

        # Mid beta weight
        ttk.Label(
            self.sc_gamma_reconciliation_adaptive_frame, text="Mid β weight (α):"
        ).grid(row=4, column=0, sticky="w", pady=2)
        self.sc_gamma_mid_beta_weight_entry = ttk.Entry(
            self.sc_gamma_reconciliation_adaptive_frame,
            textvariable=self.self_consistency_gamma_reconciliation_mid_beta_weight_var,
            width=10,
        )
        self.sc_gamma_mid_beta_weight_entry.grid(
            row=4, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_mid_beta_weight_entry,
            "Weight for energy in mid β range\n"
            "γ = α·γ_energy + (1-α)·γ_velocity\nDefault: 0.5",
        )

        # Fixed weighted parameter
        self.sc_gamma_reconciliation_fixed_frame = ttk.LabelFrame(
            gamma_recon_frame, text="Fixed Weighted Parameter", padding=6
        )
        self.sc_gamma_reconciliation_fixed_frame.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(5, 2), padx=(10, 0)
        )
        self.sc_gamma_reconciliation_fixed_frame.columnconfigure(1, weight=1)

        ttk.Label(
            self.sc_gamma_reconciliation_fixed_frame, text="Fixed weight (α):"
        ).grid(row=0, column=0, sticky="w", pady=2)
        self.sc_gamma_fixed_weight_entry = ttk.Entry(
            self.sc_gamma_reconciliation_fixed_frame,
            textvariable=self.self_consistency_gamma_reconciliation_fixed_weight_var,
            width=10,
        )
        self.sc_gamma_fixed_weight_entry.grid(
            row=0, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_fixed_weight_entry,
            "Fixed weight for FIXED_WEIGHTED method\n"
            "γ = α·γ_energy + (1-α)·γ_velocity\nDefault: 0.5",
        )

        # Trace method change to toggle parameter visibility
        self.self_consistency_gamma_reconciliation_method_var.trace_add(
            "write", lambda *_: self._toggle_gamma_reconciliation_params()
        )
        self._toggle_gamma_reconciliation_params()

        # Adaptive timestep section (Energy Jump Detection functionality integrated here)
        at_frame = ttk.LabelFrame(
            stability_frame, text="Adaptive Timestep Refinement", padding=8
        )
        at_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        at_frame.columnconfigure(1, weight=1)

        self.adaptive_enable_check = ttk.Checkbutton(
            at_frame,
            text="Enable adaptive timestep (auto-refine on energy jumps)",
            variable=self.adaptive_timestep_enabled_var,
            command=self._toggle_adaptive_timestep_controls,
        )
        self.adaptive_enable_check.grid(
            row=0, column=0, columnspan=2, sticky="w", pady=2
        )

        # Energy jump threshold with help icon
        ejt_frame = ttk.Frame(at_frame)
        ejt_frame.grid(row=1, column=0, sticky="w", pady=2, padx=(20, 0))
        self.adaptive_threshold_label = ttk.Label(
            ejt_frame, text="Energy jump threshold:"
        )
        self.adaptive_threshold_label.pack(side="left")
        ejt_help = ttk.Label(ejt_frame, text="ⓘ", foreground="blue", cursor="hand2")
        ejt_help.pack(side="left", padx=(3, 0))
        Tooltip(
            ejt_help,
            "Fractional energy change triggering timestep reduction.\n\n"
            "When |ΔE/E| > threshold:\n"
            "  1. Reject step\n"
            "  2. Reduce timestep by reduction factor\n"
            "  3. Retry step with smaller timestep\n"
            "  4. Repeat until energy change < threshold\n\n"
            "Default: 0.10 (10% energy change)\n"
            "Stricter: 0.05 for smooth energy evolution\n"
            "Looser: 0.20 for exploratory runs\n\n"
            "Lower = more refinements = slower but smoother",
        )
        self.adaptive_threshold_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_threshold_var, width=16
        )
        self.adaptive_threshold_entry.grid(row=1, column=1, sticky="ew", pady=2)

        # Timestep reduction factor with help icon
        red_frame = ttk.Frame(at_frame)
        red_frame.grid(row=2, column=0, sticky="w", pady=2, padx=(20, 0))
        self.adaptive_reduction_label = ttk.Label(
            red_frame, text="Timestep reduction factor:"
        )
        self.adaptive_reduction_label.pack(side="left")
        red_help = ttk.Label(red_frame, text="ⓘ", foreground="blue", cursor="hand2")
        red_help.pack(side="left", padx=(3, 0))
        Tooltip(
            red_help,
            "Timestep reduction on energy jump detection.\n\n"
            "new_timestep = current_timestep / reduction_factor\n\n"
            "Default: 10 (reduce by 10x)\n"
            "Aggressive: 100 for severe jumps/narrow apertures\n"
            "Conservative: 5 for gentle refinement\n\n"
            "Higher = more aggressive reduction = finer resolution\n"
            "Example: h=1e-5, factor=10 → h_new=1e-6",
        )
        self.adaptive_reduction_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_reduction_factor_var, width=16
        )
        self.adaptive_reduction_entry.grid(row=2, column=1, sticky="ew", pady=2)

        # Add trace to update max_attempts display when reduction_factor changes
        self.adaptive_timestep_reduction_factor_var.trace_add(
            "write", lambda *args: self._update_max_attempts_display()
        )

        # Max refinement attempts with help icon (calculated, read-only)
        att_frame = ttk.Frame(at_frame)
        att_frame.grid(row=3, column=0, sticky="w", pady=2, padx=(20, 0))
        self.adaptive_max_attempts_label = ttk.Label(
            att_frame, text="Max refinement attempts (calculated):"
        )
        self.adaptive_max_attempts_label.pack(side="left")
        att_help = ttk.Label(att_frame, text="ⓘ", foreground="blue", cursor="hand2")
        att_help.pack(side="left", padx=(3, 0))
        Tooltip(
            att_help,
            "Maximum timestep reductions per step (READ-ONLY).\n\n"
            "Auto-calculated to be consistent with reduction_factor and min_timestep_factor:\n\n"
            "  max_attempts = ceil(log(1/min_factor) / log(reduction_factor))\n\n"
            "Examples:\n"
            "  • reduction_factor=3, min_factor=1e-4 → max_attempts = 9\n"
            "  • reduction_factor=10, min_factor=1e-4 → max_attempts = 4\n\n"
            "Why automatic?\n"
            "After n attempts: h_final = h_base / (reduction_factor^n)\n"
            "At minimum: h_min = h_base × min_timestep_factor\n"
            "These must be consistent!\n\n"
            "To change: adjust 'Reduction factor' or 'Min timestep factor' above.",
        )
        # Create read-only display label for max_attempts (calculated value)
        self.adaptive_max_attempts_display = ttk.Label(
            at_frame,
            textvariable=self.adaptive_timestep_max_attempts_display_var,
            relief="sunken",
            background="#f0f0f0",
            foreground="#606060",
            padding=(5, 2),
            font=("TkDefaultFont", 9, "italic"),
        )
        self.adaptive_max_attempts_display.grid(row=3, column=1, sticky="ew", pady=2)

        # Min timestep factor with help icon
        min_frame = ttk.Frame(at_frame)
        min_frame.grid(row=4, column=0, sticky="w", pady=2, padx=(20, 0))
        self.adaptive_min_factor_label = ttk.Label(
            min_frame, text="Min timestep factor:"
        )
        self.adaptive_min_factor_label.pack(side="left")
        min_help = ttk.Label(min_frame, text="ⓘ", foreground="blue", cursor="hand2")
        min_help.pack(side="left", padx=(3, 0))
        Tooltip(
            min_help,
            "Minimum timestep as fraction of original.\n\n"
            "h_min = h_initial × min_factor\n"
            "Prevents infinitesimal timesteps (runaway refinement).\n\n"
            "If h_min reached and energy jump persists → fail step.\n\n"
            "Default: 1e-4 (0.01% of original)\n"
            "Allow extreme: 1e-6 for ultra-narrow apertures\n"
            "Fail faster: 1e-3 to detect bad setups early\n\n"
            "Example: h_initial=1e-5, factor=1e-4\n"
            "  → h_min = 1e-9 (minimum allowed)",
        )
        self.adaptive_min_factor_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_min_factor_var, width=16
        )
        self.adaptive_min_factor_entry.grid(row=4, column=1, sticky="ew", pady=2)

        # Add traces to update calculated displays when min_factor changes
        self.adaptive_timestep_min_factor_var.trace_add(
            "write",
            lambda *args: (
                self._update_max_attempts_display(),
                self._update_max_substeps_display(),
            ),
        )

        # Hysteresis parameters
        cd_frame = ttk.Frame(at_frame)
        cd_frame.grid(row=5, column=0, sticky="w", pady=2)
        self.adaptive_cooldown_label = ttk.Label(cd_frame, text="Cooldown steps:")
        self.adaptive_cooldown_label.pack(side="left")
        cd_help = ttk.Label(cd_frame, text="ⓘ", foreground="blue", cursor="hand2")
        cd_help.pack(side="left", padx=(3, 0))
        Tooltip(
            cd_help,
            "Steps to remain on reduced timestep after refinement.\n\n"
            "HYSTERESIS: Prevents oscillation\n"
            "  refine → restore → refine → restore (bad!)\n\n"
            "After refinement:\n"
            "  1. Stay on small h for cooldown steps\n"
            "  2. Then probe if safe to restore\n\n"
            "Default: 10 steps\n"
            "Cautious: 20 for unstable/boundary regions\n"
            "Aggressive: 5 to restore quickly",
        )
        self.adaptive_cooldown_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_cooldown_steps_var, width=16
        )
        self.adaptive_cooldown_entry.grid(row=5, column=1, sticky="ew", pady=2)

        pt_frame = ttk.Frame(at_frame)
        pt_frame.grid(row=6, column=0, sticky="w", pady=2)
        self.adaptive_probe_threshold_label = ttk.Label(
            pt_frame, text="Probe threshold:"
        )
        self.adaptive_probe_threshold_label.pack(side="left")
        pt_help = ttk.Label(pt_frame, text="ⓘ", foreground="blue", cursor="hand2")
        pt_help.pack(side="left", padx=(3, 0))
        Tooltip(
            pt_help,
            "Energy threshold for testing timestep restoration.\n\n"
            "After cooldown, probe if safe to increase h:\n"
            "  If |ΔE/E| < probe_threshold for N consecutive steps\n"
            "  → begin restoring h toward original\n\n"
            "CRITICAL: Must be < energy jump threshold!\n"
            "  Otherwise: restore → jump → refine → restore (oscillation)\n\n"
            "Default: 0.01 (1% change, 10× below jump threshold)\n"
            "Cautious: 0.001 for slower restoration\n"
            "Faster: 0.05 (ensure jump threshold ≥ 0.10)",
        )
        self.adaptive_probe_threshold_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_probe_threshold_var, width=16
        )
        self.adaptive_probe_threshold_entry.grid(row=6, column=1, sticky="ew", pady=2)

        mp_frame = ttk.Frame(at_frame)
        mp_frame.grid(row=7, column=0, sticky="w", pady=2)
        self.adaptive_max_probe_label = ttk.Label(mp_frame, text="Max probe steps:")
        self.adaptive_max_probe_label.pack(side="left")
        mp_help = ttk.Label(mp_frame, text="ⓘ", foreground="blue", cursor="hand2")
        mp_help.pack(side="left", padx=(3, 0))
        Tooltip(
            mp_help,
            "Consecutive 'good' steps needed to restore timestep.\n\n"
            "After cooldown, requires N consecutive steps with\n"
            "|ΔE/E| < probe_threshold before increasing h.\n\n"
            "Prevents premature restoration from one lucky step.\n\n"
            "Default: 3 steps\n"
            "Very cautious: 5-10 (verify stability thoroughly)\n"
            "Quick restore: 1-2 (risky if region is unstable)",
        )
        self.adaptive_max_probe_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_max_probe_steps_var, width=16
        )
        self.adaptive_max_probe_entry.grid(row=7, column=1, sticky="ew", pady=2)

        # Halt on jump option (migrated from removed Energy Jump Detection)
        self.adaptive_halt_check = ttk.Checkbutton(
            at_frame,
            text="Halt simulation on energy jump",
            variable=self.adaptive_timestep_halt_on_jump_var,
        )
        self.adaptive_halt_check.grid(
            row=8, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )

        self.adaptive_debug_check = ttk.Checkbutton(
            at_frame,
            text="Verbose output (inherited by sweep/optim when Log verbosity = 'full')",
            variable=self.adaptive_timestep_debug_var,
        )
        self.adaptive_debug_check.grid(
            row=9, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )

        # Max sub-steps limit
        max_substeps_frame = ttk.Frame(at_frame)
        max_substeps_frame.grid(row=10, column=0, sticky="w", pady=2, padx=(20, 0))
        self.adaptive_max_substeps_label = ttk.Label(
            max_substeps_frame, text="Max sub-steps (calculated):"
        )
        self.adaptive_max_substeps_label.pack(side="left")
        max_substeps_help = ttk.Label(
            max_substeps_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        max_substeps_help.pack(side="left", padx=(3, 0))
        Tooltip(
            max_substeps_help,
            "Maximum number of sub-steps per main step (READ-ONLY).\n\n"
            "This value is automatically calculated from min_timestep_factor\n"
            "to prevent time discontinuities:\n\n"
            "  max_substeps = ceil(1 / min_timestep_factor) × 1.1\n\n"
            "The 1.1× safety margin ensures coverage even with rounding.\n\n"
            "Example:\n"
            "  • min_timestep_factor = 1e-4\n"
            "  • Theoretical max = ceil(1 / 1e-4) = 10,000\n"
            "  • With margin = 10,000 × 1.1 = 11,000 substeps\n\n"
            "Why automatic?\n"
            "If min_timestep_factor allows timestep to reduce to h × 1e-4,\n"
            "then at minimum timestep you need 1/1e-4 = 10,000 substeps\n"
            "to cover the full base timestep interval.\n\n"
            "Setting max_substeps lower than this would create time\n"
            "discontinuities where some time is skipped!\n\n"
            "To change this value: adjust 'Min timestep factor' above.",
        )

        # Create read-only display label for max_substeps (calculated value)
        self.adaptive_max_substeps_display = ttk.Label(
            at_frame,
            textvariable=self.adaptive_timestep_max_substeps_display_var,
            relief="sunken",
            background="#f0f0f0",
            foreground="#606060",
            padding=(5, 2),
            font=("TkDefaultFont", 9, "italic"),
        )
        self.adaptive_max_substeps_display.grid(
            row=10, column=1, sticky="w", pady=2, padx=(10, 0)
        )

        # Help text removed - was obscuring Adaptive Timestep Refinement section
        # All parameter help is now available via ⓘ tooltips

        # Initialize control states
        self._toggle_self_consistency_controls()
        self._toggle_chrono_controls()
        self._toggle_adaptive_timestep_controls()

        # Optimization/Sweep tab ----------------------------------------
        self.optimization_tab = OptimizationPlugin(
            self.notebook,
            gui_controller=self,
            sweep_config_dir=self._last_sweep_config_dir,
            sweep_output_dir=self._last_sweep_output_dir,
        )
        self.notebook.add(self.optimization_tab, text="Sweep/Optim")

        # Right side: Persistent Config/Control Panel -------------------
        right_container = ttk.Frame(self._main_horizontal_paned)
        right_container.columnconfigure(0, weight=1)
        right_container.rowconfigure(0, weight=0)
        self._main_horizontal_paned.add(right_container, minsize=CONFIG_PANEL_MIN_WIDTH)

        self._build_config_panel(right_container)

        # Summary + logs ------------------------------------------------
        # Horizontal layout: Logs on left, Summary on right

        lower_paned = ttk.Panedwindow(bottom_container, orient="horizontal")

        lower_paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        # Left pane: Logs
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

        lower_paned.add(log_frame, weight=3)

        # Right pane: Initial Summary
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
            width=40,
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

        for page in self._scroll_pages:
            page.refresh_mousewheel_bindings()

        # Set up notebook tab change callback to update run mode
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _build_config_panel(self, parent):
        """Build persistent config/control panel on right side."""
        panel = ttk.LabelFrame(parent, text="Configuration & Control", padding=10)
        panel.pack(fill="both", expand=True, padx=5, pady=5)

        # Create scrollable container for config sections
        # Use a Frame to limit expansion and ensure controls stay visible
        scroll_container = ttk.Frame(panel)
        scroll_container.pack(fill="both", expand=True, side="top", pady=(0, 5))

        canvas = tk.Canvas(
            scroll_container,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            scroll_container, orient="vertical", command=canvas.yview
        )
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Make scrollable_frame expand to fill canvas width
        def _on_canvas_resize(event):
            canvas.itemconfig(window_id, width=event.width)

        canvas.bind("<Configure>", _on_canvas_resize)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # === RUN CONFIG SECTION ===
        run_config_frame = ttk.LabelFrame(
            scrollable_frame, text="Single Run Configuration", padding=4
        )
        run_config_frame.pack(fill="x", pady=(0, 10))

        # Run config directory
        ttk.Label(run_config_frame, text="Config dir:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Entry(run_config_frame, textvariable=self.config_dir_var, width=20).grid(
            row=0, column=1, sticky="ew", pady=2, padx=(5, 2)
        )
        ttk.Button(
            run_config_frame, text="...", command=self._select_config_dir, width=3
        ).grid(row=0, column=2, sticky="w", pady=2)

        # Run output directory
        ttk.Label(run_config_frame, text="Output dir:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Entry(run_config_frame, textvariable=self.output_dir_var, width=20).grid(
            row=1, column=1, sticky="ew", pady=2, padx=(5, 2)
        )
        ttk.Button(
            run_config_frame, text="...", command=self._select_output_dir, width=3
        ).grid(row=1, column=2, sticky="w", pady=2)

        run_config_frame.columnconfigure(1, weight=1)

        # Run config name
        ttk.Label(run_config_frame, text="Config name:").grid(
            row=2, column=0, sticky="w", pady=(10, 2)
        )
        ttk.Entry(run_config_frame, textvariable=self.config_name_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(10, 2)
        )

        # Current run config display
        ttk.Label(run_config_frame, text="Current:").grid(
            row=3, column=0, sticky="w", pady=2
        )
        self.current_config_label = ttk.Label(
            run_config_frame,
            text="<unsaved>",
            foreground="gray",
            font=("TkDefaultFont", 9, "italic"),
        )
        self.current_config_label.grid(
            row=3, column=1, columnspan=2, sticky="w", pady=2
        )

        # Saved run configs list
        ttk.Label(run_config_frame, text="Saved configs:").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(10, 2)
        )

        run_list_frame = ttk.Frame(run_config_frame)
        run_list_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=2)
        run_list_frame.rowconfigure(0, weight=1)
        run_list_frame.columnconfigure(0, weight=1)

        self.config_list = tk.Listbox(run_list_frame, height=9)
        self.config_list.grid(row=0, column=0, sticky="nsew")
        self.config_list.bind(
            "<<ListboxSelect>>", lambda _event: self._on_config_selected()
        )
        self.config_list.bind("<Double-1>", lambda _event: self._load_config())

        run_scrollbar = ttk.Scrollbar(
            run_list_frame, orient="vertical", command=self.config_list.yview
        )
        run_scrollbar.grid(row=0, column=1, sticky="ns")
        self.config_list.configure(yscrollcommand=run_scrollbar.set)

        run_config_frame.rowconfigure(5, weight=1)

        # Run config buttons
        run_btn_frame = ttk.Frame(run_config_frame)
        run_btn_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(5, 0))

        ttk.Button(run_btn_frame, text="Load", command=self._load_config, width=8).pack(
            side="left", padx=2
        )
        ttk.Button(run_btn_frame, text="Save", command=self._save_config, width=8).pack(
            side="left", padx=2
        )
        ttk.Button(
            run_btn_frame, text="Refresh", command=self._refresh_config_list, width=8
        ).pack(side="left", padx=2)

        # === SWEEP CONFIG SECTION ===
        sweep_config_frame = ttk.LabelFrame(
            scrollable_frame, text="Sweep Configuration", padding=4
        )
        sweep_config_frame.pack(fill="x", pady=(0, 10))

        # Sweep config directory
        ttk.Label(sweep_config_frame, text="Config dir:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.sweep_config_dir_var = tk.StringVar(value=self._last_sweep_config_dir)
        ttk.Entry(
            sweep_config_frame, textvariable=self.sweep_config_dir_var, width=20
        ).grid(row=0, column=1, sticky="ew", pady=2, padx=(5, 2))
        ttk.Button(
            sweep_config_frame,
            text="...",
            command=self._select_sweep_config_dir,
            width=3,
        ).grid(row=0, column=2, sticky="w", pady=2)

        # Sweep output directory
        ttk.Label(sweep_config_frame, text="Output dir:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.sweep_output_dir_var = tk.StringVar(value=self._last_sweep_output_dir)
        ttk.Entry(
            sweep_config_frame, textvariable=self.sweep_output_dir_var, width=20
        ).grid(row=1, column=1, sticky="ew", pady=2, padx=(5, 2))
        ttk.Button(
            sweep_config_frame,
            text="...",
            command=self._select_sweep_output_dir,
            width=3,
        ).grid(row=1, column=2, sticky="w", pady=2)

        sweep_config_frame.columnconfigure(1, weight=1)

        # Sweep config name entry
        ttk.Label(sweep_config_frame, text="Config name:").grid(
            row=2, column=0, sticky="w", pady=(10, 2)
        )
        ttk.Entry(sweep_config_frame, textvariable=self.sweep_config_name_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(10, 2)
        )

        # Current sweep config display
        ttk.Label(sweep_config_frame, text="Current:").grid(
            row=3, column=0, sticky="w", pady=(5, 2)
        )
        self.current_sweep_config_label = ttk.Label(
            sweep_config_frame,
            text="<none>",
            foreground="gray",
            font=("TkDefaultFont", 9, "italic"),
        )
        self.current_sweep_config_label.grid(
            row=3, column=1, columnspan=2, sticky="w", pady=(5, 2)
        )

        # Saved sweep configs list
        ttk.Label(sweep_config_frame, text="Saved configs:").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(10, 2)
        )

        sweep_list_frame = ttk.Frame(sweep_config_frame)
        sweep_list_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=2)
        sweep_list_frame.rowconfigure(0, weight=1)
        sweep_list_frame.columnconfigure(0, weight=1)

        self.sweep_config_list = tk.Listbox(sweep_list_frame, height=9)
        self.sweep_config_list.grid(row=0, column=0, sticky="nsew")
        self.sweep_config_list.bind(
            "<Double-1>", lambda _event: self._load_sweep_config()
        )
        self.sweep_config_list.bind(
            "<<ListboxSelect>>", lambda _event: self._on_sweep_config_selected()
        )

        sweep_scrollbar = ttk.Scrollbar(
            sweep_list_frame, orient="vertical", command=self.sweep_config_list.yview
        )
        sweep_scrollbar.grid(row=0, column=1, sticky="ns")
        self.sweep_config_list.configure(yscrollcommand=sweep_scrollbar.set)

        sweep_config_frame.rowconfigure(5, weight=1)

        # Sweep config buttons
        sweep_btn_frame = ttk.Frame(sweep_config_frame)
        sweep_btn_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(5, 0))

        ttk.Button(
            sweep_btn_frame, text="Load", command=self._load_sweep_config, width=8
        ).pack(side="left", padx=2)
        ttk.Button(
            sweep_btn_frame, text="Save", command=self._save_sweep_config, width=8
        ).pack(side="left", padx=2)
        ttk.Button(
            sweep_btn_frame,
            text="Refresh",
            command=self._refresh_sweep_config_list,
            width=8,
        ).pack(side="left", padx=2)

        # Reset defaults button (applies to both)
        reset_frame = ttk.Frame(scrollable_frame)
        reset_frame.pack(fill="x", pady=(0, 10))
        ttk.Button(
            reset_frame,
            text="Reset All Directories to Defaults",
            command=self._reset_directories_to_defaults,
        ).pack(fill="x")

        # Status display - packed at bottom first (appears at true bottom)
        status_frame = ttk.LabelFrame(panel, text="Status", padding=4)
        status_frame.pack(side="bottom", fill="x")

        # Refresh sweep config list now that widget exists
        self._refresh_sweep_config_list()

        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w", pady=2)

        self._progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self._progress_bar.pack(fill="x", pady=5)

        # Control buttons - packed at bottom second (appears above status)
        control_frame = ttk.LabelFrame(panel, text="Controls", padding=4)
        control_frame.pack(side="bottom", fill="x", pady=(0, 5))

        # Buttons use minwidth to prevent shrinking below readable size
        # The CONFIG_PANEL_MIN_WIDTH on the paned window should prevent this,
        # but minwidth provides an extra safeguard
        self._run_button = ttk.Button(
            control_frame,
            text="▶ Run",
            command=self._trigger_run,
            style="Accent.TButton",
        )
        self._run_button.pack(fill="x", pady=2)
        self._run_button.configure(width=12)  # minimum character width

        self._cancel_button = ttk.Button(
            control_frame,
            text="⬛ Cancel",
            command=self._trigger_cancel,
            state="disabled",
        )
        self._cancel_button.pack(fill="x", pady=2)
        self._cancel_button.configure(width=12)  # minimum character width

        # Run mode selector - packed at bottom third (appears above controls)
        mode_frame = ttk.LabelFrame(panel, text="Run Mode", padding=4)
        mode_frame.pack(side="bottom", fill="x", pady=(0, 5))

        self.run_mode_var = tk.StringVar(value="single")

        ttk.Radiobutton(
            mode_frame,
            text="Single Run",
            variable=self.run_mode_var,
            value="single",
            command=self._on_run_mode_changed,
        ).pack(anchor="w", pady=2)

        ttk.Radiobutton(
            mode_frame,
            text="Sweep/Optim",
            variable=self.run_mode_var,
            value="sweep",
            command=self._on_run_mode_changed,
        ).pack(anchor="w", pady=2)

    def _on_run_mode_changed(self):
        """Handle run mode selection change."""
        mode = self.run_mode_var.get()
        if mode == "single":
            self._run_button.config(text="▶ Run", command=self._trigger_run)
        else:  # sweep
            self._run_button.config(text="▶ Run Sweep", command=self._trigger_sweep)

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

        for config_name in configs:
            self.config_list.insert(tk.END, config_name)

        if highlight is not None:
            self.config_list.selection_set(highlight)
            self.config_list.see(highlight)

    def _refresh_sweep_config_list(self, selected: Optional[str] = None) -> None:
        """Refresh the sweep config list."""
        import os

        self.sweep_config_list.delete(0, tk.END)
        sweep_dir = self.sweep_config_dir_var.get()

        highlight: Optional[int] = None

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
            self.current_config_label.config(text=filename, foreground="black")
        else:
            self.current_config_label.config(text="<none>", foreground="gray")

    def _on_sweep_config_selected(self) -> None:
        """Handle sweep config selection from list."""
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

        # Always preserve directories when loading a config - directories should only
        # change when explicitly set by the user via directory selection buttons
        self._apply_options_to_ui(options, preserve_directories=True)
        self.config_name_var.set(filename)
        self.config_file_var.set(filename)

        # Auto-switch to single run mode when loading a single run config
        self.run_mode_var.set("single")
        self._on_run_mode_changed()
        print("[INFO] Auto-switched to Single Run mode")

        # Refresh config list to update highlighting
        self._refresh_config_list(selected=filename)
        self._refresh_initial_summary()
        self._update_legacy_state()
        self._update_driver_visibility()
        self._update_image_subcharge_state()
        self._update_cavity_spacing_state()
        self._toggle_z_cutoff_controls()
        self._toggle_macroparticle_controls()
        self._update_macroparticle_state()

        # Force update of simulation type combobox display
        current_value = self.sim_type_var.get()
        # Use current() method to set by index instead of set() for readonly combobox
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
        self.legacy_var.set(options.legacy_enabled)
        self.overlay_display_var.set(options.overlay_display)
        self.overlay_save_var.set(options.overlay_save)
        self.difference_display_var.set(options.difference_display)
        self.difference_save_var.set(options.difference_save)
        self.metrics_save_var.set(options.metrics_save)
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
        # chrono_matching_mode not exposed in GUI, always FAST
        self.adaptive_timestep_enabled_var.set(options.adaptive_timestep_enabled)
        self.adaptive_timestep_halt_on_jump_var.set(options.energy_monitor_halt_on_jump)
        self.adaptive_timestep_threshold_var.set(options.adaptive_timestep_threshold)
        self.adaptive_timestep_reduction_factor_var.set(
            options.adaptive_timestep_reduction_factor
        )
        # Update calculated max_attempts display
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
        # Update calculated max_substeps display
        self._update_max_substeps_display()
        self.save_log_file_var.set(options.save_log_file)

        # Only override directories if not preserving loaded preferences
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

        # Set z_cutoff_enabled based on whether z_cutoff is non-zero
        z_cutoff_val = options.core_params.get("z_cutoff", 0.0)
        self.z_cutoff_enabled_var.set(z_cutoff_val != 0.0)
        self._toggle_z_cutoff_controls()

    def _update_max_attempts_display(self):
        """Update the calculated max_refinement_attempts display based on reduction_factor and min_timestep_factor."""
        import math

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
            # Handle invalid input gracefully
            self.adaptive_timestep_max_attempts_display_var.set("N/A")

    def _update_max_substeps_display(self):
        """Update the calculated max_substeps display based on min_timestep_factor."""
        import math

        try:
            min_factor = self.adaptive_timestep_min_factor_var.get()
            theoretical_max = math.ceil(1.0 / min_factor)
            with_margin = int(theoretical_max * 1.1)
            self.adaptive_timestep_max_substeps_display_var.set(
                f"{with_margin} (from min factor)"
            )
        except (ValueError, ZeroDivisionError):
            # Handle invalid input gracefully
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
            # Keep strings as strings, convert others to float
            if isinstance(CORE_PARAM_DEFAULTS[name], str):
                core_params[name] = value
            else:
                core_params[name] = float(value)

        # If z_cutoff is disabled, force it to 0 (or None equivalent)
        if not self.z_cutoff_enabled_var.get():
            core_params["z_cutoff"] = 0.0

        config_name = self.config_name_var.get().strip() or "testbed_config"
        if not config_name.endswith(".json"):
            config_name += ".json"

        # Handle random seed
        if self.random_seed_var.get():
            import random

            seed = random.randint(1, 2**31 - 1)
        else:
            seed = int(self.seed_var.get())

        options = SimulationOptions(
            simulation_type=sim_type,
            steps=int(self.steps_var.get()),
            seed=seed,
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
            self_consistency_chrono_matching_mode="FAST",  # Always FAST, not exposed in GUI
            energy_monitor_enabled=False,  # Removed, functionality in adaptive timestep
            energy_monitor_threshold=2.0,  # Default (unused)
            energy_monitor_check_interval=10,  # Default (unused)
            energy_monitor_halt_on_jump=bool(
                self.adaptive_timestep_halt_on_jump_var.get()
            ),
            energy_monitor_debug=False,  # Removed
            adaptive_timestep_enabled=bool(self.adaptive_timestep_enabled_var.get()),
            adaptive_timestep_threshold=float(
                self.adaptive_timestep_threshold_var.get()
            ),
            adaptive_timestep_reduction_factor=int(
                self.adaptive_timestep_reduction_factor_var.get()
            ),
            # max_refinement_attempts is now auto-calculated in AdaptiveTimestepConfig
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
            # max_substeps_per_step is now auto-calculated in AdaptiveTimestepConfig
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
        lines = ["(single run)", f"Seed: {summary.seed}"]
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

        # Only show driver info if this simulation type supports it
        if summary.supports_driver:
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
                    lines.append(
                        f"Driver total energy: {summary.driver_total_gev:.4f} GeV"
                    )

                # Add driver beam optics if available
                if summary.driver_emittance_x_mm_mrad is not None:
                    driver_emit_x_pm = summary.driver_emittance_x_mm_mrad * 1e9
                    driver_emit_y_pm = summary.driver_emittance_y_mm_mrad * 1e9
                    driver_norm_emit_x_pm = (
                        summary.driver_norm_emittance_x_mm_mrad * 1e9
                    )
                    driver_norm_emit_y_pm = (
                        summary.driver_norm_emittance_y_mm_mrad * 1e9
                    )

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
        return "\n".join(lines)

    def _select_config_dir(self) -> None:
        import os

        # Use last used directory or default
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
        import os

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
        """Select sweep config directory."""
        import os

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
            # Update optimization plugin
            if hasattr(self, "optimization_tab"):
                self.optimization_tab.sweep_config_dir = directory

    def _select_sweep_output_dir(self) -> None:
        """Select sweep output directory."""
        import os

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
            # Update optimization plugin
            if hasattr(self, "optimization_tab"):
                self.optimization_tab.sweep_output_dir = directory

    def _load_sweep_config(self) -> None:
        """Load sweep configuration from entry field or list selection."""
        import os

        # First try to get filename from entry field
        filename = self.sweep_config_name_var.get().strip()

        # If entry is empty, try to get from list selection
        if not filename:
            selection = self.sweep_config_list.curselection()
            if not selection:
                messagebox.showinfo(
                    "Load Sweep Config",
                    "Enter a config name or select one from the list.",
                )
                return
            filename = self.sweep_config_list.get(selection[0])

        # Ensure .json extension
        if not filename.endswith(".json"):
            filename += ".json"

        if not hasattr(self, "optimization_tab"):
            return

        # Build full path
        sweep_config_dir = self.sweep_config_dir_var.get()
        path = os.path.join(sweep_config_dir, filename)

        # Check if file exists
        if not os.path.exists(path):
            messagebox.showerror(
                "Load Sweep Config", f"Configuration file not found: {filename}"
            )
            return

        # Load the configuration
        self.optimization_tab._load_config_from_path(path)
        self.current_sweep_config_label.config(
            text=filename, foreground="black", font=("TkDefaultFont", 9)
        )

    def _save_sweep_config(self) -> None:
        """Save current sweep configuration using entered filename."""
        if not hasattr(self, "optimization_tab"):
            return

        # Get filename from entry field
        filename = self.sweep_config_name_var.get().strip()

        if not filename:
            messagebox.showinfo("Save Sweep Config", "Enter a config name to save.")
            return

        # Ensure .json extension
        if not filename.endswith(".json"):
            filename += ".json"

        import os

        sweep_config_dir = self.sweep_config_dir_var.get()
        os.makedirs(sweep_config_dir, exist_ok=True)

        filepath = os.path.join(sweep_config_dir, filename)

        # Check for override warning
        if not self._check_override_warning(Path(filepath), "sweep"):
            return

        # Delegate to optimization plugin with the filepath
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

    def _check_override_warning(self, filepath: Path, config_type: str = "run") -> bool:
        """Check if file exists and show override warning if needed.

        Parameters
        ----------
        filepath : Path
            Path to the file that will be saved
        config_type : str
            Type of config ("run" or "sweep") for the dialog title

        Returns
        -------
        bool
            True if save should proceed, False if cancelled
        """
        import os

        # If file doesn't exist, no warning needed
        if not os.path.exists(filepath):
            return True

        # If user has suppressed warnings for this session, proceed
        if self._suppress_override_warning:
            return True

        # Show override warning dialog with checkbox
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Override {config_type.capitalize()} Configuration")
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill="both", expand=True)

        # Warning message
        msg = f"Configuration file already exists:\n\n{filepath.name}\n\nDo you want to override it?"
        ttk.Label(frame, text=msg, wraplength=400).pack(pady=(0, 15))

        # Checkbox for suppressing future warnings
        suppress_var = tk.BooleanVar(value=False)
        check_frame = ttk.Frame(frame)
        check_frame.pack(fill="x", pady=(0, 15))
        ttk.Checkbutton(
            check_frame,
            text="Don't show this warning again (this session only)",
            variable=suppress_var,
        ).pack(anchor="w")

        # Info label
        info_label = ttk.Label(
            check_frame,
            text="Note: This setting resets when you restart the GUI",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        )
        info_label.pack(anchor="w", padx=(20, 0), pady=(5, 0))

        # Result container
        result = [False]

        def on_yes():
            if suppress_var.get():
                self._suppress_override_warning = True
            result[0] = True
            dialog.destroy()

        def on_no():
            result[0] = False
            dialog.destroy()

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack()
        ttk.Button(button_frame, text="Yes, Override", command=on_yes, width=15).pack(
            side="left", padx=5
        )
        ttk.Button(button_frame, text="No, Cancel", command=on_no, width=15).pack(
            side="left", padx=5
        )

        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"+{x}+{y}")

        # Wait for user response
        dialog.wait_window()

        return result[0]

    def _save_config(self) -> None:
        """Save current run configuration using entered filename."""
        try:
            options = self._build_options_from_ui()
        except ValueError as exc:
            _show_error_dialog(self.root, "Invalid configuration", str(exc))
            return

        # Get filename from entry field
        filename = self.config_name_var.get().strip()

        if not filename:
            messagebox.showinfo("Save Run Config", "Enter a config name to save.")
            return

        # Ensure .json extension
        if not filename.endswith(".json"):
            filename += ".json"

        import os

        config_dir = self.config_dir_var.get()
        os.makedirs(config_dir, exist_ok=True)

        filepath = os.path.join(config_dir, filename)

        # Check for override warning
        if not self._check_override_warning(Path(filepath), "run"):
            return

        # Update the config name to match the saved file
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

    def _on_sim_type_change(self) -> None:
        self._update_driver_visibility()
        self._update_cavity_spacing_state()
        self._update_image_subcharge_state()
        self._update_macroparticle_state()
        self._refresh_initial_summary()

        # Sync simulation type to optimization plugin if it exists
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

        # Transverse offsets are only used in BUNCH_TO_BUNCH mode
        offsets_enabled = sim_type == SimulationType.BUNCH_TO_BUNCH
        offset_state = "normal" if offsets_enabled else "disabled"

        for entry in getattr(self, "_rider_offset_entries", []):
            entry.configure(state=offset_state)
        for entry in getattr(self, "_driver_offset_entries", []):
            entry.configure(state=offset_state)

        # Also update label colors to indicate disabled state
        label_color = "black" if offsets_enabled else "gray60"
        for label in getattr(self, "_rider_offset_labels", []):
            label.configure(foreground=label_color)
        for label in getattr(self, "_driver_offset_labels", []):
            label.configure(foreground=label_color)

    def _update_image_subcharge_state(self) -> None:
        """Grey out image subcharge count, weighting, aperture radius, and wall_z when in BUNCH_TO_BUNCH mode."""
        sim_type = SimulationType[self.sim_type_var.get()]
        # Image subcharge, weighting, aperture, and wall_z are only used in CONDUCTING_WALL and SWITCHING_WALL modes
        enabled = sim_type != SimulationType.BUNCH_TO_BUNCH
        entry_state = "normal" if enabled else "disabled"
        if hasattr(self, "image_subcharge_entry"):
            self.image_subcharge_entry.configure(state=entry_state)
        if hasattr(self, "image_weighting_check"):
            self.image_weighting_check.configure(state=entry_state)
        # Also grey out aperture_radius and wall_z in core params
        if hasattr(self, "core_param_widgets"):
            if "aperture_radius" in self.core_param_widgets:
                self.core_param_widgets["aperture_radius"].configure(state=entry_state)
            if "wall_z" in self.core_param_widgets:
                self.core_param_widgets["wall_z"].configure(state=entry_state)

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

    def _update_cavity_spacing_state(self) -> None:
        """Grey out cavity_spacing unless simulation type is SWITCHING_WALL."""
        is_switching = self.sim_type_var.get() == "SWITCHING_WALL"
        state = "normal" if is_switching else "disabled"

        if (
            hasattr(self, "core_param_widgets")
            and "cav_spacing" in self.core_param_widgets
        ):
            self.core_param_widgets["cav_spacing"].configure(state=state)

    def _toggle_random_seed(self) -> None:
        """Enable/disable seed entry based on random seed checkbox."""
        random_enabled = self.random_seed_var.get()
        state = "disabled" if random_enabled else "normal"

        if hasattr(self, "seed_entry"):
            self.seed_entry.configure(state=state)

    def _toggle_z_cutoff_controls(self) -> None:
        """Enable/disable z_cutoff controls based on checkbox state."""
        enabled = self.z_cutoff_enabled_var.get()
        state = "normal" if enabled else "disabled"
        combo_state = "readonly" if enabled else "disabled"

        if hasattr(self, "z_cutoff_entry"):
            self.z_cutoff_entry.configure(state=state)
        if hasattr(self, "z_cutoff_mode_combo"):
            self.z_cutoff_mode_combo.configure(state=combo_state)

        # If disabled, set z_cutoff to None/0 to indicate it's not active
        if not enabled:
            self.core_param_vars["z_cutoff"].set(0.0)

    def _toggle_self_consistency_controls(self) -> None:
        """Enable/disable self-consistency controls based on enabled checkbox."""
        if not hasattr(self, "sc_target_ms_tolerance_label"):
            return  # Widgets not created yet

        enabled = self.self_consistency_enabled_var.get()
        param_state = "normal" if enabled else "disabled"

        # Gray out all sub-controls when disabled
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
            self.sc_chrono_interpolate_check,
            self.sc_chrono_tolerance_label,
            self.sc_chrono_tolerance_entry,
            self.sc_chrono_high_precision_check,
            self.sc_chrono_adaptive_check,
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

        # Also update mode-specific greying
        if enabled:
            self._on_sc_mode_changed()
            self._toggle_chrono_controls()

    def _toggle_chrono_controls(self) -> None:
        """Enable/disable chrono sub-controls based on chrono-match interpolation checkbox."""
        if not hasattr(self, "sc_chrono_tolerance_label"):
            return  # Widgets not created yet

        # Only enable chrono sub-options if both self-consistency AND chrono-match are enabled
        sc_enabled = self.self_consistency_enabled_var.get()
        chrono_enabled = self.self_consistency_chrono_interpolate_var.get()
        enable_chrono_options = sc_enabled and chrono_enabled

        param_state = "normal" if enable_chrono_options else "disabled"
        label_color = "black" if enable_chrono_options else "gray"

        # Chrono sub-controls that should only be enabled when chrono-match is enabled
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
        # No mode-specific UI updates needed
        pass

    def _toggle_macroparticle_controls(self) -> None:
        """Enable/disable macroparticle controls based on checkbox state."""
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
        """Enable/disable macroparticle controls based on simulation type."""
        if not hasattr(self, "macroparticle_enable_check"):
            return  # Widgets not created yet

        # Macroparticle simulation only available for CONDUCTING_WALL
        is_conducting_wall = self.sim_type_var.get() == "CONDUCTING_WALL"

        # Disable the entire macroparticle section if not conducting wall
        check_state = "normal" if is_conducting_wall else "disabled"
        self.macroparticle_enable_check.configure(state=check_state)

        # If not conducting wall, force it disabled and grey out all controls
        if not is_conducting_wall:
            self.macroparticle_enabled_var.set(False)
            widget_state = "disabled"
            label_color = "gray"
        else:
            # If conducting wall, respect the enabled checkbox state
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

    def _toggle_gamma_reconciliation_params(self) -> None:
        """Show/hide gamma reconciliation parameter groups based on selected method."""
        if not hasattr(self, "sc_gamma_reconciliation_adaptive_frame"):
            return  # Widgets not created yet

        method = self.self_consistency_gamma_reconciliation_method_var.get()

        # Show adaptive parameters only for ADAPTIVE_WEIGHTED method
        if method == "ADAPTIVE_WEIGHTED":
            self.sc_gamma_reconciliation_adaptive_frame.grid()
            self.sc_gamma_reconciliation_fixed_frame.grid_remove()
        # Show fixed weight only for FIXED_WEIGHTED method
        elif method == "FIXED_WEIGHTED":
            self.sc_gamma_reconciliation_adaptive_frame.grid_remove()
            self.sc_gamma_reconciliation_fixed_frame.grid()
        # Hide both for other methods (DISABLED, USE_VELOCITY, USE_ENERGY)
        else:
            self.sc_gamma_reconciliation_adaptive_frame.grid_remove()
            self.sc_gamma_reconciliation_fixed_frame.grid_remove()

    def _toggle_adaptive_timestep_controls(self) -> None:
        """Enable/disable adaptive timestep controls based on enabled checkbox.

        When adaptive timestep is disabled: gray out ALL controls
        """
        if not hasattr(self, "adaptive_threshold_label"):
            return  # Widgets not created yet

        adaptive_enabled = self.adaptive_timestep_enabled_var.get()
        param_state = "normal" if adaptive_enabled else "disabled"

        # All controls are grayed out when adaptive timestep is disabled
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
            self.adaptive_max_substeps_label,
            self.adaptive_max_substeps_display,
        ]

        for control in all_controls:
            if isinstance(control, (ttk.Entry, ttk.Checkbutton)):
                control.configure(state=param_state)
            elif isinstance(control, ttk.Label):
                fg_color = "black" if adaptive_enabled else "gray"
                control.configure(foreground=fg_color)

    def _on_trajectory_save_toggled(self) -> None:
        """Enable/disable trajectory stride controls based on save checkbox."""
        if not hasattr(self, "trajectory_stride_entry"):
            return  # Widgets not created yet

        save_enabled = self.trajectory_save_var.get()
        widget_state = "normal" if save_enabled else "disabled"
        label_color = "black" if save_enabled else "gray"

        self.trajectory_stride_entry.configure(state=widget_state)
        self.trajectory_stride_label.configure(foreground=label_color)

    # ------------------------------------------------------------------
    # Simulation execution
    # ------------------------------------------------------------------

    def _on_tab_changed(self, event=None) -> None:
        """Handle notebook tab change events.

        Note: We no longer auto-switch run mode when changing tabs.
        Users must explicitly select run mode via the radio buttons.
        """
        # Removed auto-switching behavior - run mode is now only changed
        # via explicit radio button selection in the control panel

        # Refresh initial summary when switching tabs to ensure it's up-to-date
        self._refresh_initial_summary()

    def _open_optimization_tab(self) -> None:
        """Switch to the Sweep/Optim tab."""
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "Sweep/Optim":
                self.notebook.select(i)
                break

    def _trigger_sweep(self) -> None:
        """Handle Run Sweep button click with validation."""
        # Check if a configuration is loaded/saved in optimization plugin
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

        # Delegate to optimization plugin's run sweep method
        if hasattr(self.optimization_tab, "_on_run_sweep"):
            self.optimization_tab._on_run_sweep()
        else:
            messagebox.showerror(
                "Error",
                "Optimization plugin not properly initialized.",
            )

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

        # Create timestamped output directory for single runs
        # Format: results/runs/YYYYMMDD_HHMMSS_configname/
        from datetime import datetime
        from pathlib import Path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(options.config_name).stem  # Remove .json extension
        timestamped_dir = Path("results/runs") / f"{timestamp}_{config_name}"

        # Update options to use timestamped directory
        options.output_dir = timestamped_dir
        ensure_directory(options.output_dir)

        # Log where results will be saved
        self._append_log(f"Output directory: {timestamped_dir}")
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

        # Create batched logger for this run (100 messages per batch, 500ms flush interval)
        def gui_log_callback(text: str) -> None:
            self.root.after(0, partial(self._append_log, text))

        self._batched_logger = BatchedLogger(
            gui_callback=gui_log_callback,
            batch_size=100,
            flush_interval_ms=500,
            max_queue_size=10000,
            enable_batching=True,  # Can be disabled for debugging if needed
        )

        # Throttled progress callback (max 10 updates/second)
        throttled_progress = ThrottledProgressCallback(
            gui_callback=lambda pct: self.root.after(
                0, lambda: self.progress_var.set(pct)
            ),
            min_interval_ms=100,
            force_final=True,
        )

        def cancel_callback() -> bool:
            return self._cancel_requested

        try:
            result = run_testbed(
                options,
                log=self._batched_logger.log,  # Use batched logger
                progress_callback=throttled_progress,  # Use throttled progress
                cancel_callback=cancel_callback,
            )
        except IntegrationCancelled:
            # Flush any remaining log messages before canceling
            if self._batched_logger:
                self._batched_logger.flush()
            self.root.after(0, self._on_cancelled)
            return
        except Exception as exc:  # pragma: no cover - UI safeguard
            # Flush any remaining log messages before error handling
            if self._batched_logger:
                self._batched_logger.flush()
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
        finally:
            # Clean shutdown of batched logger
            if self._batched_logger:
                stats = self._batched_logger.get_stats()
                if stats["total_messages"] > 0:
                    # Log batching statistics for monitoring
                    reduction = stats["reduction_factor"]
                    self._append_log(
                        f"[Batched Logging Stats] {stats['total_messages']} messages "
                        f"→ {stats['total_batches']} batches "
                        f"({reduction:.1f}× reduction, {stats['dropped_messages']} dropped)"
                    )
                self._batched_logger.shutdown()
                self._batched_logger = None

        self.root.after(0, partial(self._on_success, result))

    def _queue_log(self, text: str) -> None:
        """Legacy log queuing (kept for compatibility, but batched logger preferred)."""
        self.root.after(0, partial(self._append_log, text))

    def _replot_with_new_axis(
        self,
        figure: Any,
        plot_name: str,
        new_xaxis: str,
        canvas: Any,
        new_yaxis: str = None,
    ) -> None:
        """Regenerate plot with new x-axis (and optionally y-axis for energy) using stored data by updating artists in-place."""
        import numpy as np

        if not hasattr(figure, "_lw_plot_data"):
            return

        data = figure._lw_plot_data

        # Determine x-axis data
        if new_xaxis == "z":
            xdata = data["z_mm"]
            xlabel = "z position (mm)"
        else:
            xdata = data["times_ns"]
            xlabel = "Time (ns)"

        axes = figure.get_axes()

        # Store original font sizes to prevent explosion
        original_label_sizes = [ax.xaxis.label.get_fontsize() for ax in axes]
        original_title_sizes = [ax.title.get_fontsize() for ax in axes]
        original_xtick_sizes = []
        original_ytick_sizes = []
        for ax in axes:
            xtick_labels = ax.get_xticklabels()
            ytick_labels = ax.get_yticklabels()
            original_xtick_sizes.append(
                xtick_labels[0].get_fontsize() if xtick_labels else 9
            )
            original_ytick_sizes.append(
                ytick_labels[0].get_fontsize() if ytick_labels else 9
            )

        if plot_name == "energy":
            # Update energy plot artists in-place (avoid clearing to prevent font explosion)
            # Determine y-axis data based on new_yaxis parameter (if provided)
            if new_yaxis and data.get("energy_components"):
                if new_yaxis == "delta_total":
                    ydata_r = data["energy_components"]["delta_total_r"]
                    ydata_d = data["energy_components"].get("delta_total_d")
                    ylabel = "ΔE (GeV)"
                    title_suffix = "ΔE"
                elif new_yaxis == "delta_z":
                    ydata_r = data["energy_components"]["delta_z_r"]
                    ydata_d = data["energy_components"].get("delta_z_d")
                    ylabel = "ΔE_z (GeV)"
                    title_suffix = "ΔE_z"
                elif new_yaxis == "delta_x":
                    ydata_r = data["energy_components"]["delta_x_r"]
                    ydata_d = data["energy_components"].get("delta_x_d")
                    ylabel = "ΔE_x (GeV)"
                    title_suffix = "ΔE_x"
                elif new_yaxis == "delta_y":
                    ydata_r = data["energy_components"]["delta_y_r"]
                    ydata_d = data["energy_components"].get("delta_y_d")
                    ylabel = "ΔE_y (GeV)"
                    title_suffix = "ΔE_y"
                elif new_yaxis == "total":
                    ydata_r = data["energy_components"]["total_r"]
                    ydata_d = data["energy_components"].get("total_d")
                    ylabel = "E (GeV)"
                    title_suffix = "E"
                else:
                    ydata_r = data["core_r_energy_changes"]
                    ydata_d = data.get("core_d_energy_changes")
                    ylabel = "ΔE (GeV)"
                    title_suffix = "ΔE"
            else:
                ydata_r = data["core_r_energy_changes"]
                ydata_d = data.get("core_d_energy_changes")
                ylabel = "ΔE (GeV)"
                title_suffix = "ΔE"

            # Rider energy axis
            collections = axes[0].collections
            if len(collections) > 0:
                # Update core rider scatter
                collections[0].set_offsets(np.c_[xdata, ydata_r])
                if (
                    len(collections) > 1
                    and data["legacy_enabled"]
                    and data.get("legacy_r_energy_changes") is not None
                ):
                    # Update legacy rider scatter (use same y-data for now)
                    xdata_leg = (
                        data["z_mm_legacy"] if new_xaxis == "z" else data["times_ns"]
                    )
                    collections[1].set_offsets(
                        np.c_[xdata_leg, data["legacy_r_energy_changes"]]
                    )

            axes[0].set_xlabel(xlabel, fontsize=10)
            axes[0].set_ylabel(ylabel, fontsize=10)
            axes[0].set_title(
                f"Rider {title_suffix} vs " + ("z" if new_xaxis == "z" else "Time"),
                fontsize=12,
            )
            axes[0].relim()
            axes[0].autoscale_view(tight=True)
            # Force proper x-axis and y-axis limits with 10% buffer
            x_min, x_max = xdata.min(), xdata.max()
            x_range = x_max - x_min
            x_buffer = x_range * 0.1 if x_range > 0 else 0.1 * abs(x_max)
            axes[0].set_xlim(x_min - x_buffer, x_max + x_buffer)
            if len(ydata_r) > 0:
                y_min, y_max = ydata_r.min(), ydata_r.max()
                y_range = y_max - y_min
                y_buffer = y_range * 0.1 if y_range > 0 else 0.1 * abs(y_max)
                axes[0].set_ylim(y_min - y_buffer, y_max + y_buffer)

            # Driver energy axis (if present)
            if len(axes) > 1 and data["driver_allowed"] and ydata_d is not None:
                xdata_d = data["z_mm_driver"] if new_xaxis == "z" else data["times_ns"]
                collections_d = axes[1].collections
                if len(collections_d) > 0:
                    # Update core driver scatter
                    collections_d[0].set_offsets(np.c_[xdata_d, ydata_d])
                    if (
                        len(collections_d) > 1
                        and data["legacy_enabled"]
                        and data.get("legacy_d_energy_changes") is not None
                    ):
                        # Update legacy driver scatter
                        xdata_leg_d = (
                            data["z_mm_legacy_driver"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        collections_d[1].set_offsets(
                            np.c_[xdata_leg_d, data["legacy_d_energy_changes"]]
                        )

                axes[1].set_xlabel(xlabel, fontsize=10)
                axes[1].set_ylabel(ylabel, fontsize=10)
                axes[1].set_title(
                    f"Driver {title_suffix} vs "
                    + ("z" if new_xaxis == "z" else "Time"),
                    fontsize=12,
                )
                axes[1].relim()
                axes[1].autoscale_view(tight=True)
                # Force proper x-axis and y-axis limits with 10% buffer
                x_min_d, x_max_d = xdata_d.min(), xdata_d.max()
                x_range_d = x_max_d - x_min_d
                x_buffer_d = x_range_d * 0.1 if x_range_d > 0 else 0.1 * abs(x_max_d)
                axes[1].set_xlim(x_min_d - x_buffer_d, x_max_d + x_buffer_d)
                if len(ydata_d) > 0:
                    y_min_d, y_max_d = ydata_d.min(), ydata_d.max()
                    y_range_d = y_max_d - y_min_d
                    y_buffer_d = (
                        y_range_d * 0.1 if y_range_d > 0 else 0.1 * abs(y_max_d)
                    )
                    axes[1].set_ylim(y_min_d - y_buffer_d, y_max_d + y_buffer_d)

        elif plot_name == "transverse":
            # Update transverse plot artists in-place
            ax_x, ax_y = axes[0], axes[1]

            # Update line artists for x-axis subplot
            lines_x = ax_x.get_lines()
            if len(lines_x) > 0:
                lines_x[0].set_xdata(xdata)
                idx = 1
                if data["driver_allowed"] and data["core_d_hist"] is not None:
                    xdata_d = (
                        data["z_mm_driver"] if new_xaxis == "z" else data["times_ns"]
                    )
                    if len(lines_x) > idx:
                        lines_x[idx].set_xdata(xdata_d)
                        idx += 1
                if data["legacy_enabled"] and data["legacy_r_hist"] is not None:
                    xdata_leg = (
                        data["z_mm_legacy"] if new_xaxis == "z" else data["times_ns"]
                    )
                    if len(lines_x) > idx:
                        lines_x[idx].set_xdata(xdata_leg)
                        idx += 1
                    if data["driver_allowed"] and data["legacy_d_hist"] is not None:
                        xdata_leg_d = (
                            data["z_mm_legacy_driver"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        if len(lines_x) > idx:
                            lines_x[idx].set_xdata(xdata_leg_d)

            ax_x.set_xlabel(xlabel, fontsize=10)
            ax_x.relim()
            ax_x.autoscale_view(tight=True)

            # Update line artists for y-axis subplot
            lines_y = ax_y.get_lines()
            if len(lines_y) > 0:
                lines_y[0].set_xdata(xdata)
                idx = 1
                if data["driver_allowed"] and data["core_d_hist"] is not None:
                    xdata_d = (
                        data["z_mm_driver"] if new_xaxis == "z" else data["times_ns"]
                    )
                    if len(lines_y) > idx:
                        lines_y[idx].set_xdata(xdata_d)
                        idx += 1
                if data["legacy_enabled"] and data["legacy_r_hist"] is not None:
                    xdata_leg = (
                        data["z_mm_legacy"] if new_xaxis == "z" else data["times_ns"]
                    )
                    if len(lines_y) > idx:
                        lines_y[idx].set_xdata(xdata_leg)
                        idx += 1
                    if data["driver_allowed"] and data["legacy_d_hist"] is not None:
                        xdata_leg_d = (
                            data["z_mm_legacy_driver"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        if len(lines_y) > idx:
                            lines_y[idx].set_xdata(xdata_leg_d)

            ax_y.set_xlabel(xlabel, fontsize=10)
            ax_y.relim()
            ax_y.autoscale_view(tight=True)

        elif plot_name == "beta":
            # Update beta plot artists in-place for all 4 subplots
            for i, ax in enumerate(axes):
                lines = ax.get_lines()
                if len(lines) > 0:
                    lines[0].set_xdata(xdata)
                    idx = 1
                    if data["driver_allowed"] and data["core_d_beta"] is not None:
                        xdata_d = (
                            data["z_mm_driver"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        if len(lines) > idx:
                            lines[idx].set_xdata(xdata_d)
                            idx += 1
                    if data["legacy_enabled"] and data["legacy_r_beta"] is not None:
                        xdata_leg = (
                            data["z_mm_legacy"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        if len(lines) > idx:
                            lines[idx].set_xdata(xdata_leg)

                ax.set_xlabel(xlabel, fontsize=10)
                ax.relim()
                ax.autoscale_view(tight=True)

        elif plot_name == "momentum":
            # Update momentum plot artists in-place for all 6 subplots
            for i, ax in enumerate(axes[:6]):  # Process first 6 axes
                lines = ax.get_lines()
                if len(lines) > 0:
                    lines[0].set_xdata(xdata)
                    idx = 1
                    if data["driver_allowed"] and data["core_d_momentum"] is not None:
                        xdata_d = (
                            data["z_mm_driver"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        if len(lines) > idx:
                            lines[idx].set_xdata(xdata_d)
                            idx += 1
                    if data["legacy_enabled"] and data["legacy_r_momentum"] is not None:
                        xdata_leg = (
                            data["z_mm_legacy"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        if len(lines) > idx:
                            lines[idx].set_xdata(xdata_leg)

                ax.set_xlabel(xlabel, fontsize=10)
                ax.relim()
                ax.autoscale_view(tight=True)

        elif plot_name == "gamma":
            # Update gamma plot artists in-place
            for i, ax in enumerate(axes):
                collections = ax.collections
                if len(collections) > 0:
                    # Update rider core gamma scatter
                    collections[0].set_offsets(np.c_[xdata, data["core_r_gamma"]])
                    idx = 1
                    if data["driver_allowed"] and data.get("core_d_gamma") is not None:
                        xdata_d = (
                            data["z_mm_driver"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        if len(collections) > idx:
                            collections[idx].set_offsets(
                                np.c_[xdata_d, data["core_d_gamma"]]
                            )
                            idx += 1
                    if (
                        data["legacy_enabled"]
                        and data.get("legacy_r_gamma") is not None
                    ):
                        xdata_leg = (
                            data["z_mm_legacy"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        if len(collections) > idx:
                            collections[idx].set_offsets(
                                np.c_[xdata_leg, data["legacy_r_gamma"]]
                            )
                            idx += 1
                        if (
                            data["driver_allowed"]
                            and data.get("legacy_d_gamma") is not None
                            and i == 1
                        ):
                            xdata_leg_d = (
                                data["z_mm_legacy_driver"]
                                if new_xaxis == "z"
                                else data["times_ns"]
                            )
                            if len(collections) > idx:
                                collections[idx].set_offsets(
                                    np.c_[xdata_leg_d, data["legacy_d_gamma"]]
                                )

                ax.set_xlabel(xlabel, fontsize=10)
                ax.relim()
                ax.autoscale_view(tight=True)

                # Apply intelligent y-axis scaling for gamma to show small fluctuations
                try:
                    # Collect all gamma values for this subplot
                    all_gamma = []
                    if len(data["core_r_gamma"]) > 0:
                        all_gamma.extend(data["core_r_gamma"])
                    if (
                        i == 1
                        and data["driver_allowed"]
                        and data.get("core_d_gamma") is not None
                    ):
                        if len(data["core_d_gamma"]) > 0:
                            all_gamma.extend(data["core_d_gamma"])
                    if (
                        data["legacy_enabled"]
                        and data.get("legacy_r_gamma") is not None
                    ):
                        if len(data["legacy_r_gamma"]) > 0:
                            all_gamma.extend(data["legacy_r_gamma"])
                    if (
                        i == 1
                        and data["driver_allowed"]
                        and data.get("legacy_d_gamma") is not None
                    ):
                        if len(data["legacy_d_gamma"]) > 0:
                            all_gamma.extend(data["legacy_d_gamma"])

                    if len(all_gamma) > 0:
                        gamma_array = np.array(all_gamma)
                        gamma_min = np.min(gamma_array)
                        gamma_max = np.max(gamma_array)
                        gamma_mean = np.mean(gamma_array)
                        gamma_range = gamma_max - gamma_min

                        # Check if variation is small relative to mean (< 5% is considered small)
                        relative_variation = (
                            gamma_range / gamma_mean if gamma_mean > 0 else 0
                        )

                        if relative_variation < 0.05 and gamma_range > 0:
                            # Small variation: zoom in with 10% buffer around actual range
                            buffer = (
                                gamma_range * 0.1
                                if gamma_range > 0
                                else gamma_mean * 0.001
                            )
                            ax.set_ylim(gamma_min - buffer, gamma_max + buffer)
                except Exception as e:
                    # Silently ignore errors in y-axis scaling
                    pass

        # Restore original font sizes to prevent explosion
        for i, ax in enumerate(axes):
            if i < len(original_label_sizes):
                ax.xaxis.label.set_fontsize(original_label_sizes[i])
                ax.yaxis.label.set_fontsize(original_label_sizes[i])
            if i < len(original_title_sizes):
                ax.title.set_fontsize(original_title_sizes[i])
            # Restore tick label sizes
            if i < len(original_xtick_sizes):
                ax.tick_params(
                    axis="x", which="major", labelsize=original_xtick_sizes[i]
                )
            if i < len(original_ytick_sizes):
                ax.tick_params(
                    axis="y", which="major", labelsize=original_ytick_sizes[i]
                )

        # Force canvas redraw
        canvas.draw_idle()

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

        # Save config copy to output directory
        try:
            import json
            from pathlib import Path

            config_file = Path(self.options.output_dir) / "run_config.json"
            with open(config_file, "w") as f:
                json.dump(self.options.to_dict(), f, indent=2)
            self._append_log(f"Config saved to: {config_file}")
        except Exception as e:
            self._append_log(f"Warning: Could not save config: {e}")

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
                self._show_figure(title, figure, plot_name=name)
            except Exception as e:
                error_msg = f"Error displaying {title} plot: {e}"
                self._append_log(error_msg)
                _show_warning_dialog(self.root, "Plot Display Error", error_msg)

    def _show_figure(self, title: str, figure: Any, plot_name: str = "") -> None:
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

        # Log scale controls - check if data is suitable for log scaling
        ttk.Label(controls_frame, text="Log scale:").pack(side=tk.LEFT, padx=(0, 5))

        x_log_var = tk.BooleanVar(value=False)
        y_log_var = tk.BooleanVar(value=False)

        # Check if axes contain non-positive values
        def check_axis_data_for_log(ax, axis="x"):
            """Check if axis data is suitable for log scaling (all positive)."""
            import numpy as np

            try:
                if axis == "x":
                    # Check x-data from all artists
                    for line in ax.get_lines():
                        xdata = line.get_xdata()
                        if len(xdata) > 0 and np.any(xdata <= 0):
                            return False
                    for coll in ax.collections:
                        offsets = coll.get_offsets()
                        if len(offsets) > 0 and np.any(offsets[:, 0] <= 0):
                            return False
                else:  # y-axis
                    for line in ax.get_lines():
                        ydata = line.get_ydata()
                        if len(ydata) > 0 and np.any(ydata <= 0):
                            return False
                    for coll in ax.collections:
                        offsets = coll.get_offsets()
                        if len(offsets) > 0 and np.any(offsets[:, 1] <= 0):
                            return False
                return True
            except Exception:
                return False

        # Check all axes for log scale suitability
        x_log_suitable = all(
            check_axis_data_for_log(ax, "x") for ax in figure.get_axes()
        )
        y_log_suitable = all(
            check_axis_data_for_log(ax, "y") for ax in figure.get_axes()
        )

        def toggle_log_scale() -> None:
            try:
                import numpy as np
                from matplotlib.ticker import LogFormatterSciNotation, ScalarFormatter

                for ax in figure.get_axes():
                    if x_log_var.get():
                        if not check_axis_data_for_log(ax, "x"):
                            x_log_var.set(False)
                            _show_warning_dialog(
                                window,
                                "Log Scale Warning",
                                "X-axis cannot be log-scaled: data contains non-positive values",
                            )
                            return
                        try:
                            ax.set_xscale("log")
                            # Use scientific notation for log scale
                            formatter = LogFormatterSciNotation()
                            ax.xaxis.set_major_formatter(formatter)
                            ax.tick_params(axis="x", which="major", labelsize=9)
                            ax.tick_params(axis="x", which="minor", labelsize=8)
                        except Exception as e:
                            x_log_var.set(False)
                            _show_warning_dialog(
                                window,
                                "Log Scale Error",
                                f"Failed to set X-axis log scale: {e}",
                            )
                            return
                    else:
                        ax.set_xscale("linear")
                        formatter = ScalarFormatter()
                        formatter.set_scientific(False)
                        formatter.set_useOffset(False)
                        ax.xaxis.set_major_formatter(formatter)
                        ax.tick_params(axis="x", which="major", labelsize=9)

                    if y_log_var.get():
                        if not check_axis_data_for_log(ax, "y"):
                            y_log_var.set(False)
                            _show_warning_dialog(
                                window,
                                "Log Scale Warning",
                                "Y-axis cannot be log-scaled: data contains non-positive values",
                            )
                            return
                        try:
                            ax.set_yscale("log")
                            # Use scientific notation for log scale
                            formatter = LogFormatterSciNotation()
                            ax.yaxis.set_major_formatter(formatter)
                            ax.tick_params(axis="y", which="major", labelsize=9)
                            ax.tick_params(axis="y", which="minor", labelsize=8)
                        except Exception as e:
                            y_log_var.set(False)
                            _show_warning_dialog(
                                window,
                                "Log Scale Error",
                                f"Failed to set Y-axis log scale: {e}",
                            )
                            return
                    else:
                        ax.set_yscale("linear")
                        formatter = ScalarFormatter()
                        formatter.set_scientific(False)
                        formatter.set_useOffset(False)
                        ax.yaxis.set_major_formatter(formatter)
                        ax.tick_params(axis="y", which="major", labelsize=9)

                    # Force axis relimiting after scale change
                    ax.relim()
                    ax.autoscale_view(tight=False)

                canvas.draw_idle()
            except Exception as e:
                self._append_log(f"Error toggling log scale: {e}")

        x_log_check = ttk.Checkbutton(
            controls_frame, text="X-axis", variable=x_log_var, command=toggle_log_scale
        )
        x_log_check.pack(side=tk.LEFT, padx=5)
        if not x_log_suitable:
            x_log_check.configure(state="disabled")

        y_log_check = ttk.Checkbutton(
            controls_frame, text="Y-axis", variable=y_log_var, command=toggle_log_scale
        )
        y_log_check.pack(side=tk.LEFT, padx=5)
        if not y_log_suitable:
            y_log_check.configure(state="disabled")

        # Separator
        ttk.Separator(controls_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        # Add axis switching for applicable plots
        plot_supports_xaxis_switch = plot_name in [
            "beta",
            "momentum",
            "transverse",
            "zposition",
            "energy",
            "gamma",
        ]
        plot_supports_yaxis_switch = plot_name == "energy"

        if plot_supports_xaxis_switch and plot_name != "zposition":
            # Get current x-axis setting from the plot's x-label
            current_xaxis = "t"  # default
            try:
                first_ax = figure.get_axes()[0]
                xlabel = first_ax.get_xlabel().lower()
                if (
                    "z position" in xlabel
                    or "delta z" in xlabel
                    or "δz" in xlabel
                    or "z (mm)" in xlabel
                ):
                    current_xaxis = "z"
                elif "time" in xlabel:
                    current_xaxis = "t"
            except:
                pass

            xaxis_var = tk.StringVar(value=current_xaxis)

            # For energy plots, also set up Y-axis switching
            yaxis_var = None
            if plot_supports_yaxis_switch:
                current_yaxis = getattr(
                    self, "energy_yaxis_var", tk.StringVar(value="delta_total")
                ).get()
                yaxis_var = tk.StringVar(value=current_yaxis)

            def switch_xaxis() -> None:
                """Regenerate the plot with a different x-axis."""
                new_xaxis = xaxis_var.get()
                new_yaxis = yaxis_var.get() if yaxis_var else None
                try:
                    # Check if plot has data attached
                    if not hasattr(figure, "_lw_plot_data"):
                        # Update the corresponding option variable for next run
                        if plot_name == "beta":
                            self.beta_xaxis_var.set(new_xaxis)
                        elif plot_name == "momentum":
                            self.momentum_xaxis_var.set(new_xaxis)
                        elif plot_name == "transverse":
                            self.transverse_xaxis_var.set(new_xaxis)
                        elif plot_name == "energy":
                            self.energy_xaxis_var.set(new_xaxis)
                            if new_yaxis:
                                self.energy_yaxis_var.set(new_yaxis)
                        elif plot_name == "gamma":
                            self.gamma_xaxis_var.set(new_xaxis)

                        self._append_log(
                            f"Axis changed for {plot_name} plot. Re-run simulation to see changes."
                        )
                        window.after(
                            100,
                            lambda: _show_warning_dialog(
                                window,
                                "Axis Changed",
                                f"Axis preference saved. Please re-run the simulation to regenerate the {title} plot with the new axes.",
                            ),
                        )
                        return

                    # Regenerate plot with new axis
                    self._replot_with_new_axis(
                        figure, plot_name, new_xaxis, canvas, new_yaxis
                    )

                    # Update preference for future runs
                    if plot_name == "beta":
                        self.beta_xaxis_var.set(new_xaxis)
                    elif plot_name == "momentum":
                        self.momentum_xaxis_var.set(new_xaxis)
                    elif plot_name == "transverse":
                        self.transverse_xaxis_var.set(new_xaxis)
                    elif plot_name == "energy":
                        self.energy_xaxis_var.set(new_xaxis)
                        if new_yaxis:
                            self.energy_yaxis_var.set(new_yaxis)
                    elif plot_name == "gamma":
                        self.gamma_xaxis_var.set(new_xaxis)

                    self._append_log(f"Axis changed for {plot_name} plot.")

                except Exception as e:
                    self._append_log(f"Error switching axis: {e}")
                    import traceback

                    traceback.print_exc()

            ttk.Label(controls_frame, text="X-axis:").pack(side=tk.LEFT, padx=(10, 5))
            xaxis_combo = ttk.Combobox(
                controls_frame,
                textvariable=xaxis_var,
                values=["t", "z"],
                width=8,
                state="readonly",
            )
            xaxis_combo.pack(side=tk.LEFT, padx=5)
            xaxis_combo.bind("<<ComboboxSelected>>", lambda e: switch_xaxis())

            # Add Y-axis control for energy plots
            if plot_supports_yaxis_switch and yaxis_var:
                ttk.Label(controls_frame, text="Y-axis:").pack(
                    side=tk.LEFT, padx=(10, 5)
                )
                yaxis_combo = ttk.Combobox(
                    controls_frame,
                    textvariable=yaxis_var,
                    values=["delta_total", "delta_z", "delta_x", "delta_y", "total"],
                    width=12,
                    state="readonly",
                )
                yaxis_combo.pack(side=tk.LEFT, padx=5)
                yaxis_combo.bind("<<ComboboxSelected>>", lambda e: switch_xaxis())

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
                import os

                default_name = f"{title.replace(' ', '_').replace('/', '_')}.png"
                # Default to results/figures directory
                default_dir = "results/figures"
                if not os.path.exists(default_dir):
                    os.makedirs(default_dir, exist_ok=True)

                filename = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    initialfile=default_name,
                    initialdir=default_dir,
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
    # Set locale to system default for proper keyboard input (Swedish, etc.)
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        # Fall back to C locale if system locale unavailable
        try:
            locale.setlocale(locale.LC_ALL, "C")
        except locale.Error:
            pass  # Continue with default locale

    root = tk.Tk()

    # Set up signal handler for Ctrl-C to allow clean exit
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, closing GUI...")
        root.quit()
        root.destroy()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Schedule periodic check to allow signal processing
    def check_signals():
        root.after(100, check_signals)

    check_signals()

    IntegratorGUI(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, exiting...")
        root.quit()
        root.destroy()
        sys.exit(0)


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
