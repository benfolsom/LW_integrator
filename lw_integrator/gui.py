"""Tkinter front-end exposing the full integrator testbed experience.

This window mirrors the functionality of ``examples/validation/integrator_testbed.ipynb``
so users can configure particle parameters, manage JSON configs, export
figures, and review logs without relying on Jupyter.  Simulation work runs in a
background thread to keep the UI responsive; any requested figures are rendered
in dedicated top-level windows using Matplotlib's TkAgg backend.
"""

from __future__ import annotations

import locale
import os
import signal
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, List, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.batched_logger import BatchedLogger
from core.debug_logger import initialize_debug_logging
from core.particle_config import DEFAULT_DRIVER_PARAMS, DEFAULT_RIDER_PARAMS
from core.species import list_species
from core.types import SimulationType

from .gui_config_list_mixins import IntegratorGUIConfigListMixin
from .gui_config_mixins import IntegratorGUIConfigMixin
from .gui_controller_mixins import IntegratorGUIControllerMixin
from .gui_layout_mixins import (
    CONFIG_PANEL_MIN_WIDTH,
    CONTENT_PANEL_MIN_WIDTH,
    IntegratorGUILayoutMixin,
    _ScrollableNotebookPage,
)
from .gui_log_mixins import IntegratorGUILogMixin
from .gui_plot_mixins import IntegratorGUIPlotMixin
from .gui_runtime_mixins import IntegratorGUIRuntimeMixin
from .gui_shell_mixins import IntegratorGUIShellMixin
from .gui_state_mixins import IntegratorGUIStateMixin
from .gui_summary_mixins import IntegratorGUISummaryMixin
from .gui_tab_mixins import IntegratorGUITabMixin
from .optimization_plugin import OptimizationPlugin
from .testbed_runner import (
    CORE_PARAM_DEFAULTS,
    DIPOLE_SOURCE_BACKEND_OPTIONS,
    DIPOLE_SOURCE_MODEL_OPTIONS,
    PARTICLE_PARAM_FIELDS,
    SPECIES_OPTIONS,
    SimulationOptions,
)

DISPLAY_MAX_WIDTH = 1600  # pixels

DISPLAY_MAX_HEIGHT = 900  # pixels
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
    except Exception:
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


class IntegratorGUI(
    IntegratorGUILogMixin,
    IntegratorGUIShellMixin,
    IntegratorGUITabMixin,
    IntegratorGUILayoutMixin,
    IntegratorGUIControllerMixin,
    IntegratorGUIConfigListMixin,
    IntegratorGUIStateMixin,
    IntegratorGUISummaryMixin,
    IntegratorGUIPlotMixin,
    IntegratorGUIRuntimeMixin,
    IntegratorGUIConfigMixin,
):
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
        self._update_driver_visibility()
        self._update_image_subcharge_state()
        self._update_driver_train_state()

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

        self._species_by_label = {label: key for label, key in SPECIES_OPTIONS}
        self._species_label_by_key = {key: label for label, key in SPECIES_OPTIONS}
        default_species_label = self._species_label_by_key.get(
            "custom", next(iter(self._species_by_label))
        )
        self.rider_species_var = tk.StringVar(value=default_species_label)
        self.driver_species_var = tk.StringVar(value=default_species_label)

        magnetic_species_options = list_species()
        self._magnetic_species_by_label = {
            species.display_name: species.name for species in magnetic_species_options
        }
        self._magnetic_species_label_by_key = {
            species.name: species.display_name for species in magnetic_species_options
        }
        self.magnetic_dipole_enabled_var = tk.BooleanVar(
            value=getattr(self.options, "magnetic_dipole_enabled", False)
        )
        self.magnetic_dipole_spin_precession_enabled_var = tk.BooleanVar(
            value=getattr(
                self.options,
                "magnetic_dipole_spin_precession_enabled",
                True,
            )
        )
        self.magnetic_dipole_stern_gerlach_force_enabled_var = tk.BooleanVar(
            value=getattr(
                self.options,
                "magnetic_dipole_stern_gerlach_force_enabled",
                False,
            )
        )
        dipole_source_label_by_model = {
            model: label for label, model in DIPOLE_SOURCE_MODEL_OPTIONS
        }
        dipole_source_model = str(
            getattr(self.options, "magnetic_dipole_source_model", "off")
        )
        self.magnetic_dipole_source_model_var = tk.StringVar(
            value=dipole_source_label_by_model.get(dipole_source_model, "Off")
        )
        dipole_backend_label_by_name = {
            backend: label for label, backend in DIPOLE_SOURCE_BACKEND_OPTIONS
        }
        dipole_source_backend = str(
            getattr(self.options, "magnetic_dipole_source_backend", "python")
        )
        self.magnetic_dipole_source_backend_var = tk.StringVar(
            value=dipole_backend_label_by_name.get(
                dipole_source_backend, "Python reference"
            )
        )
        self.magnetic_dipole_source_minimum_separation_var = tk.StringVar(
            value=str(
                getattr(
                    self.options,
                    "magnetic_dipole_source_minimum_separation_mm",
                    2.0e-9,
                )
            )
        )
        self.rider_magnetic_species_var = tk.StringVar(
            value=self._magnetic_species_label_by_key.get(
                getattr(self.options, "rider_magnetic_species", "electron"),
                "Electron",
            )
        )
        self.driver_magnetic_species_var = tk.StringVar(
            value=self._magnetic_species_label_by_key.get(
                getattr(self.options, "driver_magnetic_species", "proton"),
                "Proton",
            )
        )
        self.rider_rest_spin_vars = [
            tk.StringVar(value=str(component))
            for component in getattr(self.options, "rider_rest_spin", (0.0, 0.0, 1.0))
        ]
        self.driver_rest_spin_vars = [
            tk.StringVar(value=str(component))
            for component in getattr(self.options, "driver_rest_spin", (0.0, 0.0, 1.0))
        ]

        self.rider_param_vars: Dict[str, tk.Variable] = {}
        self.driver_param_vars: Dict[str, tk.Variable] = {}
        for name, default in DEFAULT_RIDER_PARAMS.items():
            var: tk.Variable
            if isinstance(default, str):
                var = tk.StringVar(value=default)
            elif isinstance(default, int):
                var = tk.IntVar(value=int(default))
            else:
                var = tk.DoubleVar(value=float(default))
            self.rider_param_vars[name] = var
        for name, default in DEFAULT_DRIVER_PARAMS.items():
            if isinstance(default, str):
                var = tk.StringVar(value=default)
            elif isinstance(default, int):
                var = tk.IntVar(value=int(default))
            else:
                var = tk.DoubleVar(value=float(default))
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

        self.macroparticle_smearing_enabled_var = tk.BooleanVar(
            value=getattr(self.options, "macroparticle_smearing_enabled", False)
        )
        self.macroparticle_smearing_subcharge_count_var = tk.IntVar(
            value=getattr(self.options, "macroparticle_smearing_subcharge_count", 8)
        )
        self.macroparticle_smearing_sigma_multiplier_var = tk.StringVar(
            value=str(
                getattr(self.options, "macroparticle_smearing_sigma_multiplier", 1.0)
            )
        )
        self.macroparticle_smearing_position_sigma_var = tk.StringVar(
            value=(
                ""
                if getattr(
                    self.options, "macroparticle_smearing_position_sigma_mm", None
                )
                is None
                else str(
                    getattr(self.options, "macroparticle_smearing_position_sigma_mm")
                )
            )
        )
        self.macroparticle_smearing_longitudinal_sigma_var = tk.StringVar(
            value=(
                ""
                if getattr(
                    self.options, "macroparticle_smearing_longitudinal_sigma_mm", None
                )
                is None
                else str(
                    getattr(
                        self.options, "macroparticle_smearing_longitudinal_sigma_mm"
                    )
                )
            )
        )
        self.macroparticle_smearing_momentum_sigma_var = tk.StringVar(
            value=(
                ""
                if getattr(
                    self.options,
                    "macroparticle_smearing_momentum_sigma_amu_mm_ns",
                    None,
                )
                is None
                else str(
                    getattr(
                        self.options, "macroparticle_smearing_momentum_sigma_amu_mm_ns"
                    )
                )
            )
        )
        self.macroparticle_smearing_use_position_errors_var = tk.BooleanVar(
            value=getattr(
                self.options, "macroparticle_smearing_use_position_errors", True
            )
        )
        self.macroparticle_smearing_use_momentum_errors_var = tk.BooleanVar(
            value=getattr(
                self.options, "macroparticle_smearing_use_momentum_errors", True
            )
        )
        self.macroparticle_smearing_use_centroid_errors_var = tk.BooleanVar(
            value=getattr(
                self.options, "macroparticle_smearing_use_centroid_errors", True
            )
        )
        self.macroparticle_smearing_use_internal_cloud_var = tk.BooleanVar(
            value=getattr(
                self.options, "macroparticle_smearing_use_internal_cloud", True
            )
        )
        self.macroparticle_smearing_apply_to_active_observers_var = tk.BooleanVar(
            value=getattr(
                self.options, "macroparticle_smearing_apply_to_active_observers", True
            )
        )
        self.macroparticle_smearing_apply_to_active_sources_var = tk.BooleanVar(
            value=getattr(
                self.options, "macroparticle_smearing_apply_to_active_sources", True
            )
        )
        self.macroparticle_smearing_apply_to_passive_sources_var = tk.BooleanVar(
            value=getattr(
                self.options, "macroparticle_smearing_apply_to_passive_sources", True
            )
        )
        self.macroparticle_smearing_apply_to_passive_updates_var = tk.BooleanVar(
            value=getattr(
                self.options, "macroparticle_smearing_apply_to_passive_updates", False
            )
        )
        self.macroparticle_smearing_seed_var = tk.IntVar(
            value=getattr(self.options, "macroparticle_smearing_seed", 12345)
        )
        self.macroparticle_smearing_refresh_policy_var = tk.StringVar(
            value=str(
                getattr(
                    self.options,
                    "macroparticle_smearing_refresh_policy",
                    "fixed_per_particle",
                )
            ).replace("-", "_")
        )

        # Experimental pseudo-grid options
        self.pseudo_grid_enabled_var = tk.BooleanVar(
            value=getattr(self.options, "pseudo_grid_enabled", False)
        )
        self.pseudo_grid_active_rider_count_var = tk.IntVar(
            value=getattr(self.options, "pseudo_grid_active_rider_count", 4)
        )
        self.pseudo_grid_active_driver_count_var = tk.IntVar(
            value=getattr(self.options, "pseudo_grid_active_driver_count", 4)
        )
        self.pseudo_grid_field_rider_count_var = tk.IntVar(
            value=getattr(self.options, "pseudo_grid_field_rider_count", 0)
        )
        self.pseudo_grid_field_driver_count_var = tk.IntVar(
            value=getattr(self.options, "pseudo_grid_field_driver_count", 0)
        )
        self.pseudo_grid_field_deposition_neighbor_count_var = tk.IntVar(
            value=getattr(
                self.options,
                "pseudo_grid_field_deposition_neighbor_count",
                4,
            )
        )
        self.pseudo_grid_passive_neighbor_count_var = tk.IntVar(
            value=getattr(self.options, "pseudo_grid_passive_neighbor_count", 4)
        )
        self.pseudo_grid_coverage_strategy_var = tk.StringVar(
            value=getattr(
                self.options,
                "pseudo_grid_coverage_strategy",
                "farthest_point_staleness",
            )
        )
        self.pseudo_grid_coverage_space_var = tk.StringVar(
            value=getattr(self.options, "pseudo_grid_coverage_space", "position")
        )
        self.pseudo_grid_pair_reuse_window_var = tk.IntVar(
            value=getattr(self.options, "pseudo_grid_pair_reuse_window", 16)
        )
        self.pseudo_grid_source_weighting_mode_var = tk.StringVar(
            value=getattr(
                self.options,
                "pseudo_grid_source_weighting_mode",
                "inverse_distance",
            )
        )
        self.pseudo_grid_loss_tracking_enabled_var = tk.BooleanVar(
            value=getattr(self.options, "pseudo_grid_loss_tracking_enabled", True)
        )
        self.pseudo_grid_causal_history_pruning_enabled_var = tk.BooleanVar(
            value=getattr(
                self.options,
                "pseudo_grid_causal_history_pruning_enabled",
                False,
            )
        )
        self.pseudo_grid_causal_history_safety_margin_steps_var = tk.IntVar(
            value=getattr(
                self.options,
                "pseudo_grid_causal_history_safety_margin_steps",
                2,
            )
        )

        # Driver train / persistent prehistory options
        self.driver_train_enabled_var = tk.BooleanVar(
            value=getattr(self.options, "driver_train_enabled", False)
        )
        self.driver_train_bunch_count_var = tk.IntVar(
            value=getattr(self.options, "driver_train_bunch_count", 1)
        )
        self.driver_train_z_spacing_mm_var = tk.DoubleVar(
            value=getattr(self.options, "driver_train_z_spacing_mm", 0.0)
        )
        self.driver_train_z_offsets_mm_var = tk.StringVar(
            value=" ".join(
                str(value)
                for value in getattr(self.options, "driver_train_z_offsets_mm", ())
            )
        )
        self.driver_train_prehistory_steps_var = tk.IntVar(
            value=getattr(self.options, "driver_train_prehistory_steps", 0)
        )
        self.driver_train_preserve_prehistory_var = tk.BooleanVar(
            value=getattr(
                self.options,
                "driver_train_preserve_prehistory_in_output",
                False,
            )
        )

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
        self.chrono_interpolate_var = tk.BooleanVar(
            value=getattr(
                self.options,
                "chrono_interpolate",
                getattr(self.options, "self_consistency_chrono_interpolate", False),
            )
        )
        self.chrono_tolerance_var = tk.DoubleVar(
            value=getattr(
                self.options,
                "chrono_tolerance",
                getattr(self.options, "self_consistency_chrono_tolerance", 1e-3),
            )
        )
        self.chrono_high_precision_var = tk.BooleanVar(
            value=getattr(
                self.options,
                "chrono_high_precision",
                getattr(self.options, "self_consistency_chrono_high_precision", False),
            )
        )
        self.chrono_adaptive_tolerance_var = tk.BooleanVar(
            value=getattr(
                self.options,
                "chrono_adaptive_tolerance",
                getattr(
                    self.options, "self_consistency_chrono_adaptive_tolerance", False
                ),
            )
        )
        self.self_consistency_chrono_interpolate_var = self.chrono_interpolate_var
        self.self_consistency_chrono_tolerance_var = self.chrono_tolerance_var
        self.self_consistency_chrono_high_precision_var = self.chrono_high_precision_var
        self.self_consistency_chrono_adaptive_tolerance_var = (
            self.chrono_adaptive_tolerance_var
        )
        # Gamma reconciliation options
        self.self_consistency_gamma_reconciliation_method_var = tk.StringVar(
            value=getattr(
                self.options,
                "self_consistency_gamma_reconciliation_method",
                "DISABLED",
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
        self.chrono_interpolate_var.trace_add(
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
        self.adaptive_timestep_bunch_proximity_enabled_var = tk.BooleanVar(
            value=getattr(
                self.options, "adaptive_timestep_bunch_proximity_enabled", False
            )
        )
        self.adaptive_timestep_bunch_proximity_sigma_mm_var = tk.DoubleVar(
            value=getattr(
                self.options, "adaptive_timestep_bunch_proximity_sigma_mm", 5.0
            )
        )
        self.adaptive_timestep_bunch_proximity_n_sigma_var = tk.DoubleVar(
            value=getattr(
                self.options, "adaptive_timestep_bunch_proximity_n_sigma", 5.0
            )
        )
        self.adaptive_timestep_bunch_proximity_reduction_factor_var = tk.DoubleVar(
            value=getattr(
                self.options,
                "adaptive_timestep_bunch_proximity_reduction_factor",
                10.0,
            )
        )
        self.adaptive_timestep_bunch_proximity_transition_n_sigma_var = tk.DoubleVar(
            value=getattr(
                self.options,
                "adaptive_timestep_bunch_proximity_transition_n_sigma",
                2.0,
            )
        )
        # max_substeps is now calculated from min_timestep_factor (read-only display)
        self.adaptive_timestep_max_substeps_display_var = tk.StringVar(value="")

        self.radiation_reaction_mode_var = tk.StringVar(
            value=getattr(self.options, "radiation_reaction_mode", "medina_lad")
        )

        self.space_charge_enabled_var = tk.BooleanVar(value=False)
        self.space_charge_retarded_var = tk.BooleanVar(value=True)
        self.space_charge_softening_mm_var = tk.DoubleVar(value=0.0)
        self.space_charge_bunch_sigma_mm_var = tk.DoubleVar(value=0.01)
        self.space_charge_min_retarded_steps_var = tk.StringVar(value="")

        self.external_field_enabled_var = tk.BooleanVar(value=False)
        self.external_field_input_mode_var = tk.StringVar(value="SI V/m")
        self.external_electric_native_vars = [
            tk.StringVar(value="0.0") for _axis in range(3)
        ]
        self.external_electric_si_vars = [
            tk.StringVar(value="0.0") for _axis in range(3)
        ]
        self.external_magnetic_native_vars = [
            tk.StringVar(value="0.0") for _axis in range(3)
        ]
        self.external_magnetic_tesla_vars = [
            tk.StringVar(value="0.0") for _axis in range(3)
        ]
        self.external_magnetic_gradient_vars = [
            [tk.StringVar(value="0.0") for _coordinate in range(3)]
            for _component in range(3)
        ]
        self.external_field_window_vars = {
            f"{axis}_{bound}": tk.StringVar(value="")
            for axis in ("x", "y", "z", "t")
            for bound in ("min", "max")
        }

        self.cavity_exit_enabled_var = tk.BooleanVar(
            value=getattr(self.options, "cavity_exit_enabled", False)
        )
        self.cavity_exit_mode_var = tk.StringVar(
            value=getattr(self.options, "cavity_exit_mode", "first_exit")
        )
        self.cavity_exit_length_mm_var = tk.StringVar(
            value=(
                ""
                if getattr(self.options, "cavity_exit_length_mm", None) is None
                else str(getattr(self.options, "cavity_exit_length_mm"))
            )
        )

        self.beamline_geometry_enabled_var = tk.BooleanVar(
            value=getattr(self.options, "beamline_geometry_enabled", False)
        )
        self.manual_particle_config_enabled_var = tk.BooleanVar(
            value=getattr(self.options, "manual_particle_config_enabled", False)
        )

        self.auto_duration_enabled_var = tk.BooleanVar(value=False)
        self.auto_duration_crossing_steps_var = tk.IntVar(value=200)
        self.auto_duration_post_factor_var = tk.DoubleVar(value=2.0)

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

        self.driver_train_enabled_var.trace_add(
            "write", lambda *_: self._toggle_driver_train_controls()
        )
        self.cavity_exit_enabled_var.trace_add(
            "write", lambda *_: self._toggle_cavity_exit_controls()
        )
        self.beamline_geometry_enabled_var.trace_add(
            "write", lambda *_: self._toggle_beamline_geometry_controls()
        )
        self.manual_particle_config_enabled_var.trace_add(
            "write", lambda *_: self._toggle_manual_particle_config_controls()
        )
        self.sim_type_var.trace_add("write", lambda *_: self._on_sim_type_change())

        for var in [self.seed_var, self.rider_species_var, self.driver_species_var]:
            var.trace_add("write", lambda *_: self._refresh_initial_summary())
        for name in PARTICLE_PARAM_FIELDS:
            self.rider_param_vars[name].trace_add(
                "write", lambda *_: self._refresh_initial_summary()
            )
            self.driver_param_vars[name].trace_add(
                "write", lambda *_: self._refresh_initial_summary()
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

        self._build_particle_tab()

        self._build_manual_particle_config_tab()

        self._build_core_tab()

        self._build_output_tab()

        self._build_external_fields_tab()

        self._build_stability_tab()

        self._build_beamline_geometry_tab()

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

        self._build_log_summary_panel(bottom_container)

        for page in self._scroll_pages:
            page.refresh_mousewheel_bindings()

        # Set up notebook tab change callback to update run mode
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    # ------------------------------------------------------------------
    # Simulation execution
    # ------------------------------------------------------------------


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
