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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.constants import C_MMNS  # type: ignore[import]
from core.debug_logger import set_logging_context  # type: ignore[import]
from core.smoothness_analyzer import (  # type: ignore[import]
    SmoothnessConfig,
    analyze_trajectory_smoothness,
    filter_stable_trajectories,
)
from core.types import SimulationType  # type: ignore[import]
from lw_integrator.testbed_runner import (  # type: ignore[import]
    RunResult,
    SimulationOptions,
    run_testbed,
)
from optimization.config import (
    OptimizationConfig,
    calculate_auto_steps,
    calculate_auto_timestep,
    calculate_steps_from_duration,
)
from optimization.result_io import (
    generate_optimization_heatmap,
    generate_optimization_plots,
    generate_trajectory_comparison_plot,
    save_optimization_results,
    save_partial_optimization_results,
    save_top_n_optimization_trajectories,
    save_top_trajectories_summary_table,
)
from optimization.run_mixins import OptimizationRunMixin
from optimization.results_mixins import OptimizationResultsMixin
from optimization.ui_helpers import (
    ToolTip,
    show_error_dialog as _show_error_dialog,
    show_info_dialog as _show_info_dialog,
    show_warning_dialog as _show_warning_dialog,
)


class OptimizationPlugin(OptimizationRunMixin, OptimizationResultsMixin, ttk.Frame):
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

    def _build_ui(self):
        """Build the user interface."""
        # Main container with scrollbar
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

        def _bind_to_mousewheel(event):
            # Bind mousewheel for Windows/Mac
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
            # Bind mousewheel for Linux
            self.canvas.bind_all("<Button-4>", _on_mousewheel_linux)
            self.canvas.bind_all("<Button-5>", _on_mousewheel_linux)

        def _unbind_from_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

        # Bind/unbind on enter/leave
        self.canvas.bind("<Enter>", _bind_to_mousewheel)
        self.canvas.bind("<Leave>", _unbind_from_mousewheel)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Build sections
        self._build_simulation_section()
        self._build_mode_section()
        self._build_parameter_section()
        self._build_objective_section()
        self._build_optimization_section()
        self._build_control_section()
        self._build_results_output_section()
        self._build_progress_section()

        # Initialize mode visibility
        self._update_mode_visibility()

    def _add_tooltip(self, widget, text):
        """Add a tooltip to a widget.

        Args:
            widget: The tkinter widget to add tooltip to
            text: The tooltip text to display
        """
        ToolTip(widget, text)

    def _build_simulation_section(self):
        """Build simulation type selection section."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Simulation Type", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)

        self.sim_type_var = tk.StringVar(value="CONDUCTING_WALL")
        types = [
            ("Conducting Wall", "CONDUCTING_WALL"),
            ("Switching Wall", "SWITCHING_WALL"),
            ("Bunch to Bunch", "BUNCH_TO_BUNCH"),
        ]

        for i, (label, value) in enumerate(types):
            rb = ttk.Radiobutton(
                frame,
                text=label,
                variable=self.sim_type_var,
                value=value,
                command=self._on_sim_type_changed,
            )
            rb.grid(row=0, column=i, padx=5, sticky="w")

        # Store reference for updating visibility
        self.sim_type_frame = frame

    def _build_mode_section(self):
        """Build mode selection section (blind sweep vs optimization)."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Run Mode", padding=10)
        frame.pack(fill="x", padx=10, pady=5)
        self.mode_section_frame = frame

        # Mode selection
        self.mode_var = tk.StringVar(value="blind_sweep")

        mode_frame = ttk.Frame(frame)
        mode_frame.pack(fill="x", pady=5)

        ttk.Radiobutton(
            mode_frame,
            text="Blind Sweep (Grid Search)",
            variable=self.mode_var,
            value="blind_sweep",
            command=self._update_mode_visibility,
        ).grid(row=0, column=0, padx=5, sticky="w")

        ttk.Radiobutton(
            mode_frame,
            text="Optimization",
            variable=self.mode_var,
            value="optimization",
            command=self._update_mode_visibility,
        ).grid(row=0, column=1, padx=5, sticky="w")

        # Help text
        help_frame = ttk.Frame(frame)
        help_frame.pack(fill="x", pady=(5, 0))

        self.mode_help_label = ttk.Label(
            help_frame,
            text="Blind Sweep: Exhaustively evaluate all parameter combinations (good for exploring full space).\n"
            "Optimization: Use algorithms to find optimal parameters (faster, finds best configurations).",
            foreground="gray40",
            font=("TkDefaultFont", 8),
            wraplength=600,
        )
        self.mode_help_label.pack(anchor="w")

    def _build_parameter_section(self):
        """Build parameter range specification section."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Parameter Ranges", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)
        self.parameter_frame = frame

        # Add explanatory help text
        help_frame = ttk.Frame(frame)
        help_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        help_text = (
            "Coordinate system: Particles start at z-coordinate and travel toward the conducting wall.\n"
            "Example: Particle at z=0 travels to wall at z=2200 mm (distance = 2200 mm).\n"
            "Transverse offset: Fraction of aperture radius (0.0 = on-axis, 1.0 = at aperture edge)."
        )
        help_label = ttk.Label(
            help_frame, text=help_text, foreground="gray40", font=("TkDefaultFont", 8)
        )
        help_label.pack(anchor="w")

        # Aperture range
        ttk.Label(frame, text="Aperture Radius:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        aperture_frame = ttk.Frame(frame)
        aperture_frame.grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)

        ttk.Label(aperture_frame, text="Min (mm):").pack(side="left", padx=(0, 2))
        self.aperture_min_var = tk.StringVar(value="1e-5")
        ttk.Entry(aperture_frame, textvariable=self.aperture_min_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(aperture_frame, text="Max (mm):").pack(side="left", padx=(10, 2))
        self.aperture_max_var = tk.StringVar(value="1e-3")
        ttk.Entry(aperture_frame, textvariable=self.aperture_max_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(aperture_frame, text="Points:").pack(side="left", padx=(10, 2))
        self.aperture_points_var = tk.StringVar(value="10")
        ttk.Entry(aperture_frame, textvariable=self.aperture_points_var, width=5).pack(
            side="left", padx=2
        )

        self.aperture_log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            aperture_frame, text="Log scale", variable=self.aperture_log_var
        ).pack(side="left", padx=(10, 0))

        # Energy range
        ttk.Label(frame, text="Particle Energy:").grid(
            row=2, column=0, sticky="w", pady=2
        )
        energy_frame = ttk.Frame(frame)
        energy_frame.grid(row=2, column=1, columnspan=3, sticky="ew", pady=2)

        ttk.Label(energy_frame, text="Min (GeV):").pack(side="left", padx=(0, 2))
        self.energy_min_var = tk.StringVar(value="1.0")
        ttk.Entry(energy_frame, textvariable=self.energy_min_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(energy_frame, text="Max (GeV):").pack(side="left", padx=(10, 2))
        self.energy_max_var = tk.StringVar(value="1000.0")
        ttk.Entry(energy_frame, textvariable=self.energy_max_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(energy_frame, text="Points:").pack(side="left", padx=(10, 2))
        self.energy_points_var = tk.StringVar(value="10")
        ttk.Entry(energy_frame, textvariable=self.energy_points_var, width=5).pack(
            side="left", padx=2
        )

        self.energy_log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            energy_frame, text="Log scale", variable=self.energy_log_var
        ).pack(side="left", padx=(10, 0))

        # Transverse offset fractions
        ttk.Label(frame, text="Transverse Offset:").grid(
            row=3, column=0, sticky="w", pady=2
        )
        ttk.Label(frame, text="Fractions of aperture (comma-separated):").grid(
            row=3, column=1, sticky="w", pady=2
        )
        self.offset_fractions_var = tk.StringVar(value="0.0")
        ttk.Entry(frame, textvariable=self.offset_fractions_var, width=30).grid(
            row=3, column=2, columnspan=2, sticky="ew", pady=2, padx=5
        )

        # Starting z positions
        ttk.Label(frame, text="Starting Positions:").grid(
            row=4, column=0, sticky="w", pady=2
        )
        ttk.Label(frame, text="Particle z-coordinate (mm, comma-separated):").grid(
            row=4, column=1, sticky="w", pady=2
        )
        self.start_z_var = tk.StringVar(value="0.0")
        ttk.Entry(frame, textvariable=self.start_z_var, width=30).grid(
            row=4, column=2, columnspan=2, sticky="ew", pady=2, padx=5
        )
        # Wall Position (sweepable)
        ttk.Label(frame, text="Wall Position:").grid(
            row=5, column=0, sticky="w", pady=2
        )

        # Fixed value and sweep checkbox on same row
        wall_z_fixed_frame = ttk.Frame(frame)
        wall_z_fixed_frame.grid(row=5, column=1, columnspan=3, sticky="w", pady=2)

        self.wall_z_var = tk.StringVar(value="2200.0")
        self.wall_z_entry = ttk.Entry(
            wall_z_fixed_frame, textvariable=self.wall_z_var, width=10
        )
        self.wall_z_entry.pack(side="left", padx=(0, 10))

        self.wall_z_sweep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            wall_z_fixed_frame,
            text="Sweep",
            variable=self.wall_z_sweep_var,
            command=self._toggle_wall_z_sweep,
        ).pack(side="left")

        # Sweep controls on new row for better visibility
        wall_z_sweep_frame = ttk.Frame(frame)
        wall_z_sweep_frame.grid(
            row=6, column=1, columnspan=3, sticky="w", pady=2, padx=(20, 0)
        )

        ttk.Label(wall_z_sweep_frame, text="Min:").pack(side="left", padx=(0, 2))
        self.wall_z_min_var = tk.StringVar(value="2000.0")
        self.wall_z_min_entry = ttk.Entry(
            wall_z_sweep_frame, textvariable=self.wall_z_min_var, width=8
        )
        self.wall_z_min_entry.pack(side="left", padx=2)

        ttk.Label(wall_z_sweep_frame, text="Max:").pack(side="left", padx=(5, 2))
        self.wall_z_max_var = tk.StringVar(value="2400.0")
        self.wall_z_max_entry = ttk.Entry(
            wall_z_sweep_frame, textvariable=self.wall_z_max_var, width=8
        )
        self.wall_z_max_entry.pack(side="left", padx=2)

        ttk.Label(wall_z_sweep_frame, text="Pts:").pack(side="left", padx=(5, 2))
        self.wall_z_points_var = tk.StringVar(value="3")
        self.wall_z_points_entry = ttk.Entry(
            wall_z_sweep_frame, textvariable=self.wall_z_points_var, width=4
        )
        self.wall_z_points_entry.pack(side="left", padx=2)

        self.wall_z_log_var = tk.BooleanVar(value=False)
        self.wall_z_log_check = ttk.Checkbutton(
            wall_z_sweep_frame, text="Log", variable=self.wall_z_log_var
        )
        self.wall_z_log_check.pack(side="left", padx=(5, 0))

        # Store references for sweep control
        self.wall_z_sweep_frame = wall_z_sweep_frame
        self.wall_z_sweep_widgets = [
            self.wall_z_min_entry,
            self.wall_z_max_entry,
            self.wall_z_points_entry,
            self.wall_z_log_check,
        ]

        # Initially disable sweep controls
        self._toggle_wall_z_sweep()

        # Cavity Spacing (for SWITCHING_WALL)
        ttk.Label(frame, text="Cavity Spacing:").grid(
            row=7, column=0, sticky="w", pady=2
        )
        ttk.Label(frame, text="SWITCHING_WALL only (mm):").grid(
            row=7, column=1, sticky="w", pady=2
        )
        self.cavity_spacing_var = tk.StringVar(value="1e5")
        self.cavity_spacing_entry = ttk.Entry(
            frame, textvariable=self.cavity_spacing_var, width=10
        )
        self.cavity_spacing_entry.grid(row=7, column=2, sticky="w", pady=2, padx=5)

        # Timestep Auto-Calculation (always uses auto_distance strategy)
        timestep_label = ttk.Label(frame, text="Timestep Calculation:")
        timestep_label.grid(row=8, column=0, sticky="w", pady=2)

        # Add tooltip for explanatory note
        self._add_tooltip(
            timestep_label,
            "All runs travel to wall_z + target distance regardless of energy.\n"
            "This ensures consistent trajectory length across different energies.",
        )

        timestep_frame = ttk.Frame(frame)
        timestep_frame.grid(row=8, column=1, columnspan=3, sticky="ew", pady=2)

        self.timestep_mode_var = tk.StringVar(value="duration")
        ttk.Radiobutton(
            timestep_frame,
            text="Auto-calc duration, provide count:",
            variable=self.timestep_mode_var,
            value="duration",
            command=self._toggle_timestep_mode,
        ).pack(side="left", padx=(0, 5))

        self.steps_var = tk.StringVar(value="500")
        self.steps_entry = ttk.Entry(
            timestep_frame, textvariable=self.steps_var, width=8
        )
        self.steps_entry.pack(side="left", padx=2)
        ttk.Label(timestep_frame, text="steps").pack(side="left", padx=(2, 15))

        ttk.Radiobutton(
            timestep_frame,
            text="Auto-calc count, provide duration:",
            variable=self.timestep_mode_var,
            value="count",
            command=self._toggle_timestep_mode,
        ).pack(side="left", padx=(0, 5))

        self.duration_var = tk.StringVar(value="1e-3")
        self.duration_entry = ttk.Entry(
            timestep_frame, textvariable=self.duration_var, width=10
        )
        self.duration_entry.pack(side="left", padx=2)
        ttk.Label(timestep_frame, text="ns (proper time)").pack(side="left", padx=2)

        # Distance target
        ttk.Label(frame, text="Distance Target:").grid(
            row=9, column=0, sticky="w", pady=2
        )
        distance_frame = ttk.Frame(frame)
        distance_frame.grid(row=9, column=1, columnspan=3, sticky="ew", pady=2)
        ttk.Label(distance_frame, text="Target: wall +").pack(side="left", padx=(0, 2))
        self.auto_steps_distance_var = tk.StringVar(value="10.0")
        ttk.Entry(
            distance_frame, textvariable=self.auto_steps_distance_var, width=6
        ).pack(side="left", padx=2)
        ttk.Label(distance_frame, text="mm (min 5% of steps enforced)").pack(
            side="left", padx=2
        )

        # Note about trajectory and output configuration
        config_note = ttk.Label(
            frame,
            text="ℹ For trajectory saving and output options, see the 'Results & Output Configuration' section below",
            font=("TkDefaultFont", 8, "italic"),
            foreground="blue",
            justify="left",
        )
        config_note.grid(row=10, column=0, columnspan=4, sticky="w", pady=(10, 10))

        frame.columnconfigure(2, weight=1)

        # Initialize timestep mode state
        self._toggle_timestep_mode()

    def _build_particle_section(self):
        """Build rider and driver particle parameters sections with optional sweeping."""
        # Store sweep control variables (shared by rider and driver)
        self.sweep_params = {}

        # Rider particle parameters
        self._build_rider_particle_section()

        # Driver particle parameters (shown/hidden based on simulation type)
        self._build_driver_particle_section()

    def _build_rider_particle_section(self):
        """Build rider particle parameters section with optional sweeping."""
        frame = ttk.LabelFrame(
            self.scrollable_frame,
            text="Rider Particle Parameters (optionally sweepable)",
            padding=10,
        )
        frame.pack(fill="x", padx=10, pady=5)

        row = 0

        # Particle Mass
        self._add_sweepable_param(
            frame,
            row,
            "rider_m_particle",
            "Particle Mass (amu):",
            "0.00054857990907",
            width=15,
        )
        row += 1

        # Charge Sign
        self._add_sweepable_param(
            frame, row, "rider_charge_sign", "Charge Sign:", "-1.0", width=10
        )
        row += 1

        # Particle Count
        self._add_sweepable_param(
            frame, row, "rider_pcount", "Particle Count:", "1", width=10
        )
        row += 1

        # Transverse Momentum (spread, uniform ±)
        self._add_sweepable_param(
            frame,
            row,
            "rider_transv_mom",
            "Transverse Momentum (amu·mm/ns, spread ±):",
            "1.2e-05",
            width=15,
        )
        row += 1

        # Transverse Spread (bunch radius / half-width)
        self._add_sweepable_param(
            frame,
            row,
            "rider_transv_dist",
            "Transverse Spread (mm, half-width):",
            "2e-06",
            width=15,
        )
        row += 1

        # Stripped Ions (not sweepable, always fixed)
        ttk.Label(frame, text="Stripped Ions:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.rider_stripped_ions_var = tk.StringVar(value="1.0")
        ttk.Entry(frame, textvariable=self.rider_stripped_ions_var, width=10).grid(
            row=row, column=1, sticky="w", pady=2, padx=5
        )
        row += 1

        # Timestep from main config (display only)
        ttk.Label(frame, text="Timestep from main config (ns):").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.main_timestep_display_var = tk.StringVar(value="3e-7")
        ttk.Label(frame, textvariable=self.main_timestep_display_var).grid(
            row=row, column=1, sticky="w", pady=2, padx=5
        )
        row += 1

        # Info label
        info_label = ttk.Label(
            frame,
            text="Check 'Sweep' to enable range controls. Energy and position are swept by default.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        )
        info_label.grid(row=row, column=0, columnspan=6, sticky="w", pady=(5, 0))

        # Macroparticle simulation section
        row += 1
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=6, sticky="ew", pady=(10, 10)
        )
        row += 1

        ttk.Label(
            frame,
            text="Macroparticle Simulation (Conducting Wall only):",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=row, column=0, columnspan=6, sticky="w", pady=(0, 5))
        row += 1

        self.macroparticle_enabled_var = tk.BooleanVar(value=False)
        self.macroparticle_enable_check = ttk.Checkbutton(
            frame,
            text="Enable macroparticle simulation (bunch spread inherited from above)",
            variable=self.macroparticle_enabled_var,
            command=self._toggle_macroparticle_controls,
        )
        self.macroparticle_enable_check.grid(
            row=row, column=0, columnspan=6, sticky="w", pady=2
        )
        row += 1

        # Charge multiplier (sweepable)
        # Add with indented label text
        ttk.Label(frame, text="    Charge multiplier:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        charge_var = tk.StringVar(value="1.0")
        charge_entry = ttk.Entry(frame, textvariable=charge_var, width=15)
        charge_entry.grid(row=row, column=1, sticky="w", pady=2, padx=5)

        charge_sweep_var = tk.BooleanVar(value=False)
        charge_sweep_cb = ttk.Checkbutton(
            frame,
            text="Sweep:",
            variable=charge_sweep_var,
            command=lambda: self._toggle_sweep_controls(
                "macroparticle_charge_multiplier"
            ),
        )
        charge_sweep_cb.grid(row=row, column=2, sticky="w", pady=2, padx=(10, 2))

        charge_range_frame = ttk.Frame(frame)
        charge_range_frame.grid(row=row, column=3, columnspan=3, sticky="w", pady=2)
        charge_range_frame.grid_remove()

        ttk.Label(charge_range_frame, text="Min:").pack(side="left", padx=(0, 2))
        charge_min_var = tk.StringVar(value="1.0")
        ttk.Entry(charge_range_frame, textvariable=charge_min_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(charge_range_frame, text="Max:").pack(side="left", padx=(5, 2))
        charge_max_var = tk.StringVar(value="1.0")
        ttk.Entry(charge_range_frame, textvariable=charge_max_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(charge_range_frame, text="Pts:").pack(side="left", padx=(5, 2))
        charge_points_var = tk.StringVar(value="3")
        ttk.Entry(charge_range_frame, textvariable=charge_points_var, width=4).pack(
            side="left", padx=2
        )

        charge_log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(charge_range_frame, text="Log", variable=charge_log_var).pack(
            side="left", padx=(5, 0)
        )

        self.sweep_params["macroparticle_charge_multiplier"] = {
            "fixed_var": charge_var,
            "fixed_entry": charge_entry,
            "sweep_var": charge_sweep_var,
            "range_frame": charge_range_frame,
            "min_var": charge_min_var,
            "max_var": charge_max_var,
            "points_var": charge_points_var,
            "log_var": charge_log_var,
        }
        row += 1

        # Sigma multiplier for image charge errors (sweepable)
        # Add with indented label text
        ttk.Label(frame, text="    Image error sigma multiplier:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        sigma_var = tk.StringVar(value="1.0")
        sigma_entry = ttk.Entry(frame, textvariable=sigma_var, width=15)
        sigma_entry.grid(row=row, column=1, sticky="w", pady=2, padx=5)

        sigma_sweep_var = tk.BooleanVar(value=False)
        sigma_sweep_cb = ttk.Checkbutton(
            frame,
            text="Sweep:",
            variable=sigma_sweep_var,
            command=lambda: self._toggle_sweep_controls(
                "macroparticle_sigma_multiplier"
            ),
        )
        sigma_sweep_cb.grid(row=row, column=2, sticky="w", pady=2, padx=(10, 2))

        sigma_range_frame = ttk.Frame(frame)
        sigma_range_frame.grid(row=row, column=3, columnspan=3, sticky="w", pady=2)
        sigma_range_frame.grid_remove()

        ttk.Label(sigma_range_frame, text="Min:").pack(side="left", padx=(0, 2))
        sigma_min_var = tk.StringVar(value="1.0")
        ttk.Entry(sigma_range_frame, textvariable=sigma_min_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(sigma_range_frame, text="Max:").pack(side="left", padx=(5, 2))
        sigma_max_var = tk.StringVar(value="1.0")
        ttk.Entry(sigma_range_frame, textvariable=sigma_max_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(sigma_range_frame, text="Pts:").pack(side="left", padx=(5, 2))
        sigma_points_var = tk.StringVar(value="3")
        ttk.Entry(sigma_range_frame, textvariable=sigma_points_var, width=4).pack(
            side="left", padx=2
        )

        sigma_log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sigma_range_frame, text="Log", variable=sigma_log_var).pack(
            side="left", padx=(5, 0)
        )

        self.sweep_params["macroparticle_sigma_multiplier"] = {
            "fixed_var": sigma_var,
            "fixed_entry": sigma_entry,
            "sweep_var": sigma_sweep_var,
            "range_frame": sigma_range_frame,
            "min_var": sigma_min_var,
            "max_var": sigma_max_var,
            "points_var": sigma_points_var,
            "log_var": sigma_log_var,
        }
        row += 1

        # Include momentum errors checkbox
        self.macroparticle_momentum_errors_var = tk.BooleanVar(value=True)
        self.macroparticle_momentum_errors_check = ttk.Checkbutton(
            frame,
            text="Include momentum errors (cumulative)",
            variable=self.macroparticle_momentum_errors_var,
        )
        self.macroparticle_momentum_errors_check.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )
        row += 1

        # Help text
        help_label = ttk.Label(
            frame,
            text=(
                "Macroparticle mode scales test particle charge and adds Gaussian errors to image subcharges.\n"
                "Image errors derived from bunch spread (transv_dist, transv_mom) × sigma multiplier.\n"
                "Position errors: constant σ from transv_dist. Momentum errors: cumulative from transv_mom.\n"
                "Uncheck 'Include momentum errors' to apply only constant position errors (no cumulative growth).\n"
                "Only for CONDUCTING_WALL."
            ),
            font=("TkDefaultFont", 8),
            foreground="gray40",
            justify="left",
        )
        help_label.grid(
            row=row, column=0, columnspan=6, sticky="w", pady=(0, 2), padx=(20, 0)
        )

        # Store macroparticle widgets for enable/disable
        self._macroparticle_widgets = [
            self.macroparticle_momentum_errors_check,
            self.sweep_params["macroparticle_charge_multiplier"]["fixed_entry"],
            self.sweep_params["macroparticle_sigma_multiplier"]["fixed_entry"],
        ]
        # Store sweep control references separately for conditional disabling
        self._macroparticle_sweep_controls = [
            self.sweep_params["macroparticle_charge_multiplier"],
            self.sweep_params["macroparticle_sigma_multiplier"],
        ]

    def _build_driver_particle_section(self):
        """Build driver particle parameters section with optional sweeping."""
        self.driver_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text="Driver Particle Parameters (optionally sweepable)",
            padding=10,
        )
        self.driver_frame.pack(fill="x", padx=10, pady=5)

        row = 0

        # Particle Mass
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_m_particle",
            "Particle Mass (amu):",
            "207.2",
            width=15,
        )
        row += 1

        # Charge Sign
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_charge_sign",
            "Charge Sign:",
            "1.0",
            width=10,
        )
        row += 1

        # Particle Count
        self._add_sweepable_param(
            self.driver_frame, row, "driver_pcount", "Particle Count:", "5", width=10
        )
        row += 1

        # Transverse Momentum
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_transv_mom",
            "Transverse Momentum (amu·mm/ns):",
            "0.0",
            width=15,
        )
        row += 1

        # Transverse Distance
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_transv_dist",
            "Transverse Distance (mm):",
            "-0.07998",
            width=15,
        )
        row += 1

        # Starting Distance (driver-specific)
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_starting_distance",
            "Starting Distance (mm):",
            "1000.0",
            width=15,
        )
        row += 1

        # Starting Pz (driver-specific)
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_starting_Pz",
            "Starting Pz (amu·mm/ns):",
            "-4925.0",
            width=15,
        )
        row += 1

        # Stripped Ions (not sweepable, always fixed)
        ttk.Label(self.driver_frame, text="Stripped Ions:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.driver_stripped_ions_var = tk.StringVar(value="54.0")
        ttk.Entry(
            self.driver_frame, textvariable=self.driver_stripped_ions_var, width=10
        ).grid(row=row, column=1, sticky="w", pady=2, padx=5)
        row += 1

        # Info label
        info_label = ttk.Label(
            self.driver_frame,
            text="Driver parameters only used for BUNCH_TO_BUNCH simulations.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        )
        info_label.grid(row=row, column=0, columnspan=6, sticky="w", pady=(5, 0))

        # Hide driver section initially (shown for BUNCH_TO_BUNCH type)
        self._update_driver_visibility()

    def _add_sweepable_param(
        self, parent, row, param_name, label, default_value, width=15
    ):
        """Add a parameter row with optional sweep controls."""
        # Label
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)

        # Fixed value entry
        var = tk.StringVar(value=default_value)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="w", pady=2, padx=5)

        # Sweep checkbox
        sweep_var = tk.BooleanVar(value=False)
        sweep_cb = ttk.Checkbutton(
            parent,
            text="Sweep:",
            variable=sweep_var,
            command=lambda: self._toggle_sweep_controls(param_name),
        )
        sweep_cb.grid(row=row, column=2, sticky="w", pady=2, padx=(10, 2))

        # Range controls (initially hidden)
        range_frame = ttk.Frame(parent)
        range_frame.grid(row=row, column=3, columnspan=3, sticky="w", pady=2)
        range_frame.grid_remove()  # Hide initially

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

        # Store all controls
        self.sweep_params[param_name] = {
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

    def _toggle_wall_z_sweep(self):
        """Toggle wall_z sweep controls."""
        if self.wall_z_sweep_var.get():
            # Enable sweep controls, disable fixed value
            self.wall_z_entry.config(state="disabled")
            for widget in self.wall_z_sweep_widgets:
                widget.config(state="normal")
        else:
            # Disable sweep controls, enable fixed value
            self.wall_z_entry.config(state="normal")
            for widget in self.wall_z_sweep_widgets:
                widget.config(state="disabled")

    def _toggle_timestep_mode(self):
        """Toggle between duration/count auto-calculation modes."""
        mode = self.timestep_mode_var.get()
        if mode == "duration":
            # User provides count, we calculate duration
            self.steps_entry.config(state="normal")
            self.duration_entry.config(state="disabled")
        else:  # mode == "count"
            # User provides duration, we calculate count (min 200)
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
        else:
            self.driver_frame.pack_forget()

    def _on_sim_type_changed(self):
        """Handle simulation type change."""
        self._update_driver_visibility()
        self._update_macroparticle_state()

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

        # Also handle sweep controls (checkboxes and range frames)
        if hasattr(self, "_macroparticle_sweep_controls"):
            for controls in self._macroparticle_sweep_controls:
                # Disable sweep checkbox
                if "sweep_var" in controls:
                    # Get the checkbox widget by finding it in the parent
                    # We need to find it from the stored frame
                    pass  # The checkboxes are not directly stored, skip for now
                # Disable/hide range frame if not sweeping
                if "range_frame" in controls and not controls["sweep_var"].get():
                    # Range frames are already hidden when not sweeping
                    pass

    def _update_macroparticle_state(self):
        """Enable/disable macroparticle controls based on simulation type."""
        if not hasattr(self, "macroparticle_enable_check"):
            return

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

        # Also disable/enable macroparticle sweep parameter controls
        if hasattr(self, "_macroparticle_sweep_controls"):
            for controls in self._macroparticle_sweep_controls:
                # Entry widgets in the controls are already in _macroparticle_widgets
                # Just handle the range frame entries
                if "range_frame" in controls:
                    for child in controls["range_frame"].winfo_children():
                        if isinstance(child, ttk.Entry):
                            child.configure(state=widget_state)
        self._update_parameter_visibility()

    def _update_parameter_visibility(self):
        """Update parameter field states based on simulation type."""
        if not hasattr(self, "cavity_spacing_entry"):
            return

        sim_type = self.sim_type_var.get()

        # Grey out cavity_spacing unless SWITCHING_WALL mode
        if sim_type == "SWITCHING_WALL":
            self.cavity_spacing_entry.config(state="normal")
        else:
            self.cavity_spacing_entry.config(state="disabled")

    def _build_objective_section(self):
        """Build optimization objective selection section."""
        # First build particle section
        self._build_particle_section()

        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Optimization Objective", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)

        self.objective_var = tk.StringVar(value="max_energy_gain")
        objectives = [
            ("Maximize Energy Gain (GeV)", "max_energy_gain"),
            ("Maximize Energy Gain (%)", "max_percent_energy_gain"),
            ("Maximize Energy Efficiency", "max_energy_efficiency"),
            ("Minimize Transverse Deflection", "min_transverse_deflection"),
        ]

        for i, (label, value) in enumerate(objectives):
            rb = ttk.Radiobutton(
                frame, text=label, variable=self.objective_var, value=value
            )
            rb.grid(row=i, column=0, sticky="w", pady=2)

    def _build_optimization_section(self):
        """Build optimization method selection section."""
        self.optimization_frame = ttk.LabelFrame(
            self.scrollable_frame, text="Optimization Settings", padding=10
        )
        self.optimization_frame.pack(fill="x", padx=10, pady=5)

        # Method selection
        ttk.Label(self.optimization_frame, text="Optimization Method:").grid(
            row=0, column=0, sticky="w", pady=2
        )

        self.optimization_method_var = tk.StringVar(value="genetic_algorithm")
        method_combo = ttk.Combobox(
            self.optimization_frame,
            textvariable=self.optimization_method_var,
            values=[
                "genetic_algorithm",
                "differential_evolution",
                "nelder_mead",
                "multi_start",
                "adaptive_grid",
            ],
            state="readonly",
            width=25,
        )
        method_combo.grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=2, padx=(5, 0)
        )
        method_combo.bind("<<ComboboxSelected>>", self._update_optimization_controls)

        # Method descriptions
        method_descriptions = {
            "genetic_algorithm": "Evolutionary approach with selection, crossover, and mutation (robust, parallelizable)",
            "differential_evolution": "Global optimizer using vector differences (robust for rugged landscapes)",
            "nelder_mead": "Local simplex method (fast convergence, may find local optima)",
            "multi_start": "Multiple random starting points with local optimization (finds global optima)",
            "adaptive_grid": "Coarse-to-fine grid refinement (systematic, interpretable)",
        }

        self.method_desc_label = ttk.Label(
            self.optimization_frame,
            text=method_descriptions["genetic_algorithm"],
            foreground="gray40",
            font=("TkDefaultFont", 8),
            wraplength=500,
        )
        self.method_desc_label.grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        # Common parameters
        params_frame = ttk.Frame(self.optimization_frame)
        params_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)

        # Max iterations / generations
        ttk.Label(params_frame, text="Max Iterations/Generations:").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_maxiter_var = tk.StringVar(value="50")
        ttk.Entry(
            params_frame, textvariable=self.optimization_maxiter_var, width=10
        ).grid(row=0, column=1, sticky="w", pady=2)

        # Population size (for GA and DE)
        ttk.Label(params_frame, text="Population Size:").grid(
            row=0, column=2, sticky="w", pady=2, padx=(15, 5)
        )
        self.optimization_popsize_var = tk.StringVar(value="20")
        self.popsize_entry = ttk.Entry(
            params_frame, textvariable=self.optimization_popsize_var, width=10
        )
        self.popsize_entry.grid(row=0, column=3, sticky="w", pady=2)

        # GA-specific parameters
        self.ga_frame = ttk.LabelFrame(
            self.optimization_frame, text="Genetic Algorithm Parameters", padding=5
        )
        self.ga_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)

        ga_params_frame = ttk.Frame(self.ga_frame)
        ga_params_frame.pack(fill="x")

        ttk.Label(ga_params_frame, text="Mutation Rate:").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_mutation_var = tk.StringVar(value="0.1")
        ttk.Entry(
            ga_params_frame, textvariable=self.optimization_mutation_var, width=8
        ).grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(ga_params_frame, text="Crossover Rate:").grid(
            row=0, column=2, sticky="w", pady=2, padx=(15, 5)
        )
        self.optimization_crossover_var = tk.StringVar(value="0.7")
        ttk.Entry(
            ga_params_frame, textvariable=self.optimization_crossover_var, width=8
        ).grid(row=0, column=3, sticky="w", pady=2)

        # Multi-start parameter
        self.multistart_frame = ttk.Frame(self.optimization_frame)
        self.multistart_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)

        ttk.Label(self.multistart_frame, text="Number of Random Starts:").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_nstarts_var = tk.StringVar(value="5")
        ttk.Entry(
            self.multistart_frame, textvariable=self.optimization_nstarts_var, width=10
        ).grid(row=0, column=1, sticky="w", pady=2)

        # Output settings
        output_frame = ttk.LabelFrame(
            self.optimization_frame, text="Output Options", padding=5
        )
        output_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Label(output_frame, text="Save top N trajectories:").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_save_top_n_var = tk.StringVar(value="3")
        self.optimization_save_top_n_entry = ttk.Entry(
            output_frame, textvariable=self.optimization_save_top_n_var, width=8
        )
        self.optimization_save_top_n_entry.grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(
            output_frame,
            text="(Re-runs top N parameter sets to generate trajectories)",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray50",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 2))

        # Convergence settings
        convergence_frame = ttk.LabelFrame(
            self.optimization_frame, text="Convergence Settings", padding=5
        )
        convergence_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Label(convergence_frame, text="Tolerance (rel):").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_convergence_tol_var = tk.StringVar(value="1e-6")
        ttk.Entry(
            convergence_frame,
            textvariable=self.optimization_convergence_tol_var,
            width=10,
        ).grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(convergence_frame, text="Patience (generations):").grid(
            row=0, column=2, sticky="w", pady=2, padx=(15, 5)
        )
        self.optimization_convergence_patience_var = tk.StringVar(value="10")
        ttk.Entry(
            convergence_frame,
            textvariable=self.optimization_convergence_patience_var,
            width=8,
        ).grid(row=0, column=3, sticky="w", pady=2)

        ttk.Label(
            convergence_frame,
            text="Early stopping: stops if fitness doesn't improve by tolerance over patience generations",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray40",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))

        # Initialize visibility
        self._update_optimization_controls()

    def _update_mode_visibility(self):
        """Update visibility of sections based on selected mode."""
        mode = self.mode_var.get()

        if mode == "blind_sweep":
            # Hide optimization settings
            self.optimization_frame.pack_forget()
            # Grey out Top N controls (only relevant for optimization)
            self._set_top_n_controls_state("disabled")
        else:  # optimization
            # Show optimization settings
            self.optimization_frame.pack(fill="x", padx=10, pady=5)
            # Enable Top N controls
            self._set_top_n_controls_state("normal")

        # Update parameter visibility based on simulation type
        self._update_parameter_visibility()

    def _set_top_n_controls_state(self, state):
        """Enable or disable Top N related controls.

        Parameters
        ----------
        state : str
            "normal" or "disabled"
        """
        if not hasattr(self, "save_top_n_traj_var"):
            return  # Widgets not created yet
        if not hasattr(self, "results_output_frame"):
            return  # Results output frame not created yet

        # Optimization section: "Save top N trajectories" entry
        if hasattr(self, "optimization_save_top_n_entry"):
            self.optimization_save_top_n_entry.configure(state=state)

        # Top N trajectory checkbox
        for widget in self.results_output_frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Checkbutton):
                        # Find the "Top N trajectories" checkbox
                        if "top_n_traj_var" in str(child.cget("variable")):
                            child.configure(state=state)

        # Metrics scope "Top N only" radio button
        if hasattr(self, "metrics_scope_var"):
            for widget in self.results_output_frame.winfo_children():
                if isinstance(widget, ttk.LabelFrame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Frame):
                            for radio in child.winfo_children():
                                if isinstance(radio, ttk.Radiobutton):
                                    if radio.cget("value") == "top_n":
                                        radio.configure(state=state)

        # Log verbosity "Top N only" radio button
        if hasattr(self, "log_verbosity_var"):
            for widget in self.results_output_frame.winfo_children():
                if isinstance(widget, ttk.LabelFrame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Radiobutton):
                            if child.cget("value") == "top_n_only":
                                child.configure(state=state)

        # If disabling and Top N is selected, switch to default
        if state == "disabled":
            if hasattr(self, "save_top_n_traj_var") and self.save_top_n_traj_var.get():
                self.save_top_n_traj_var.set(False)
            if (
                hasattr(self, "metrics_scope_var")
                and self.metrics_scope_var.get() == "top_n"
            ):
                self.metrics_scope_var.set("all")
            if (
                hasattr(self, "log_verbosity_var")
                and self.log_verbosity_var.get() == "top_n_only"
            ):
                self.log_verbosity_var.set("truncated")

    def _update_optimization_controls(self, event=None):
        """Update visibility of optimization controls based on selected method."""
        method = self.optimization_method_var.get()

        # Update description
        method_descriptions = {
            "genetic_algorithm": "Evolutionary approach with selection, crossover, and mutation (robust, parallelizable)",
            "differential_evolution": "Global optimizer using vector differences (robust for rugged landscapes)",
            "nelder_mead": "Local simplex method (fast convergence, may find local optima)",
            "multi_start": "Multiple random starting points with local optimization (finds global optima)",
            "adaptive_grid": "Coarse-to-fine grid refinement (systematic, interpretable)",
        }
        self.method_desc_label.config(text=method_descriptions.get(method, ""))

        # Show/hide method-specific controls
        if method == "genetic_algorithm":
            self.ga_frame.grid()
            self.multistart_frame.grid_forget()
            self.popsize_entry.config(state="normal")
        elif method == "differential_evolution":
            self.ga_frame.grid_forget()
            self.multistart_frame.grid_forget()
            self.popsize_entry.config(state="normal")
        elif method == "multi_start":
            self.ga_frame.grid_forget()
            self.multistart_frame.grid()
            self.popsize_entry.config(state="disabled")
        else:  # nelder_mead, adaptive_grid
            self.ga_frame.grid_forget()
            self.multistart_frame.grid_forget()
            self.popsize_entry.config(state="disabled")

    def _build_control_section(self):
        """Build control buttons section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Sweep Tools", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        # Info label
        info_label = ttk.Label(
            frame,
            text="Use 'Run Mode' selector in right panel to choose Single Run or Parameter Sweep, then click Run button.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray40",
        )
        info_label.pack(anchor="w", pady=(0, 10))

        # Row 1: Load from main config helper
        helper_frame = ttk.Frame(frame)
        helper_frame.pack(fill="x", pady=2)

        ttk.Button(
            helper_frame,
            text="Load from Main GUI Config",
            command=self._on_load_from_main_config,
        ).pack(side="left", padx=5)

        ttk.Label(
            helper_frame,
            text="← Copy current single-run config to sweep parameters",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).pack(side="left", padx=5)

        # Robustness options
        robustness_frame = ttk.Frame(frame)
        robustness_frame.pack(fill="x", pady=(10, 2))

        ttk.Label(robustness_frame, text="Robustness:").pack(side="left", padx=(5, 10))

        ttk.Label(robustness_frame, text="Per-run timeout (s, 0=unlimited):").pack(
            side="left", padx=(0, 5)
        )
        self.per_run_timeout_var = tk.StringVar(value="300.0")
        ttk.Entry(
            robustness_frame, textvariable=self.per_run_timeout_var, width=8
        ).pack(side="left", padx=(0, 15))

        self.skip_failed_runs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            robustness_frame,
            text="Skip failed runs and continue sweep",
            variable=self.skip_failed_runs_var,
        ).pack(side="left", padx=5)

        # Row 4: Trajectory stability checking (multi-step analysis)
        smoothness_frame = ttk.LabelFrame(
            frame, text="Trajectory Stability Analysis", padding=8
        )
        smoothness_frame.pack(fill="x", pady=(5, 0))

        self.smoothness_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            smoothness_frame,
            text="Enable multi-step stability analysis (detects numerical instabilities)",
            variable=self.smoothness_enabled_var,
            command=self._toggle_smoothness_controls,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(smoothness_frame, text="Window size (steps):").grid(
            row=1, column=0, sticky="w", padx=(20, 5), pady=2
        )
        self.smoothness_window_var = tk.StringVar(value="20")
        window_entry = ttk.Entry(
            smoothness_frame, textvariable=self.smoothness_window_var, width=8
        )
        window_entry.grid(row=1, column=1, sticky="w", pady=2)
        self._add_tooltip(window_entry, "Moving window size for trend analysis")

        ttk.Label(smoothness_frame, text="Oscillation threshold:").grid(
            row=2, column=0, sticky="w", padx=(20, 5), pady=2
        )
        self.smoothness_oscillation_var = tk.StringVar(value="0.5")
        oscillation_entry = ttk.Entry(
            smoothness_frame, textvariable=self.smoothness_oscillation_var, width=8
        )
        oscillation_entry.grid(row=2, column=1, sticky="w", pady=2)
        self._add_tooltip(
            oscillation_entry, "Sign-change rate threshold (lower = stricter)"
        )

        self.smoothness_reject_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            smoothness_frame,
            text="Reject runs with numerical instabilities",
            variable=self.smoothness_reject_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=(20, 0), pady=2)

        # Store widgets for enable/disable toggle
        self.smoothness_widgets = [
            smoothness_frame.grid_slaves(row=1, column=0)[0],  # Window size label
            smoothness_frame.grid_slaves(row=1, column=1)[0],  # Window size entry
            smoothness_frame.grid_slaves(row=2, column=0)[0],  # Oscillation label
            smoothness_frame.grid_slaves(row=2, column=1)[0],  # Oscillation entry
            smoothness_frame.grid_slaves(row=3, column=0)[0],  # Reject checkbox
        ]

    def _build_results_output_section(self):
        """Build results viewing and output configuration section."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Results & Output Configuration", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)
        frame.columnconfigure(1, weight=1)

        # Results viewing buttons
        results_frame = ttk.Frame(frame)
        results_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 15))

        ttk.Label(results_frame, text="View Results:").pack(side="left", padx=(0, 10))

        ttk.Button(
            results_frame, text="View Results", command=self._on_view_results
        ).pack(side="left", padx=5)

        ttk.Button(
            results_frame,
            text="Plot Trajectories",
            command=self._on_plot_trajectories,
        ).pack(side="left", padx=5)

        # Trajectory saving options
        ttk.Label(frame, text="Trajectory Data:").grid(
            row=1, column=0, sticky="nw", pady=(5, 2)
        )

        traj_frame = ttk.Frame(frame)
        traj_frame.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(5, 2))

        self.save_top_n_traj_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            traj_frame,
            text="Top N trajectories (full detail)",
            variable=self.save_top_n_traj_var,
            command=self._on_top_n_traj_changed,
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.save_all_traj_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            traj_frame,
            text="All trajectories (with stride)",
            variable=self.save_all_traj_var,
            command=self._on_all_traj_changed,
        ).grid(row=0, column=1, sticky="w", padx=(0, 10))

        self.save_failed_traj_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            traj_frame,
            text="Failed only (full detail)",
            variable=self.save_failed_traj_var,
            command=self._on_failed_traj_changed,
        ).grid(row=0, column=2, sticky="w", padx=(0, 10))

        self.trajectory_stride_label = ttk.Label(traj_frame, text="Stride:")
        self.trajectory_stride_label.grid(row=0, column=3, sticky="w", padx=(10, 2))
        self.trajectory_stride_var = tk.StringVar(value="1")
        self.trajectory_stride_entry = ttk.Entry(
            traj_frame, textvariable=self.trajectory_stride_var, width=6
        )
        self.trajectory_stride_entry.grid(row=0, column=4, sticky="w", padx=2)

        # Initialize trajectory stride state
        self._update_stride_state()

        # Metrics export options
        ttk.Label(frame, text="Metrics Export:").grid(
            row=2, column=0, sticky="nw", pady=(10, 2)
        )

        metrics_frame = ttk.Frame(frame)
        metrics_frame.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(10, 2))

        # Format selection
        format_frame = ttk.Frame(metrics_frame)
        format_frame.grid(row=0, column=0, sticky="w", pady=2)

        ttk.Label(format_frame, text="Format:").pack(side="left", padx=(0, 5))

        self.metrics_format_var = tk.StringVar(value="both")
        format_options = [
            ("JSON + CSV", "both"),
            ("JSON only", "json"),
            ("CSV only", "csv"),
            ("None", "none"),
        ]
        for text, value in format_options:
            ttk.Radiobutton(
                format_frame,
                text=text,
                variable=self.metrics_format_var,
                value=value,
            ).pack(side="left", padx=5)

        # Scope selection
        scope_frame = ttk.Frame(metrics_frame)
        scope_frame.grid(row=1, column=0, sticky="w", pady=2)

        ttk.Label(scope_frame, text="Scope:").pack(side="left", padx=(0, 5))

        self.metrics_scope_var = tk.StringVar(value="all")
        ttk.Radiobutton(
            scope_frame,
            text="All evaluations",
            variable=self.metrics_scope_var,
            value="all",
        ).pack(side="left", padx=5)
        ttk.Radiobutton(
            scope_frame,
            text="Top N only",
            variable=self.metrics_scope_var,
            value="top_n",
        ).pack(side="left", padx=5)

        ttk.Label(
            frame,
            text="ℹ JSON contains metadata & structure; CSV is tabular with all parameters & metrics",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray50",
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=(0, 10))

        # Log saving options
        ttk.Label(frame, text="Debug Logs:").grid(
            row=4, column=0, sticky="nw", pady=(5, 2)
        )

        log_frame = ttk.Frame(frame)
        log_frame.grid(row=4, column=1, columnspan=2, sticky="ew", pady=(5, 2))

        self.log_verbosity_var = tk.StringVar(value="truncated")

        ttk.Radiobutton(
            log_frame,
            text="None (no debug logs saved)",
            variable=self.log_verbosity_var,
            value="none",
        ).grid(row=0, column=0, sticky="w", pady=2)

        ttk.Radiobutton(
            log_frame,
            text="Truncated (1-2 lines/run: parameters + metrics + errors only) — DEFAULT",
            variable=self.log_verbosity_var,
            value="truncated",
        ).grid(row=1, column=0, sticky="w", pady=2)

        ttk.Radiobutton(
            log_frame,
            text="Full debug (inherits SC verbosity & adaptive timestep debug from Stability tab)",
            variable=self.log_verbosity_var,
            value="full",
        ).grid(row=2, column=0, sticky="w", pady=2)

        ttk.Radiobutton(
            log_frame,
            text="Top N only (logs only for best N trajectories)",
            variable=self.log_verbosity_var,
            value="top_n_only",
        ).grid(row=3, column=0, sticky="w", pady=2)

        ttk.Label(
            frame,
            text="ℹ 'Truncated' is recommended for large sweeps.\n'Full debug' inherits verbosity settings from Stability tab and generates large log files.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="blue",
            justify="left",
        ).grid(row=5, column=1, columnspan=2, sticky="w", pady=(0, 5))

    def _on_top_n_traj_changed(self):
        """Handle Top N trajectory checkbox change."""
        # Top N can be combined with All or Failed, no exclusivity needed
        self._update_stride_state()

    def _on_all_traj_changed(self):
        """Handle All trajectories checkbox change."""
        if self.save_all_traj_var.get():
            # "All" was just checked - uncheck "Failed only"
            self.save_failed_traj_var.set(False)
        self._update_stride_state()

    def _on_failed_traj_changed(self):
        """Handle Failed only checkbox change."""
        if self.save_failed_traj_var.get():
            # "Failed only" was just checked - uncheck "All"
            self.save_all_traj_var.set(False)
        self._update_stride_state()

    def _update_stride_state(self):
        """Update stride field enabled/disabled state."""
        if not hasattr(self, "trajectory_stride_entry"):
            return  # Widgets not created yet

        # Stride is ONLY enabled when "All trajectories" is selected
        # (Top N and Failed only always save full detail with stride=1)
        stride_enabled = self.save_all_traj_var.get()

        widget_state = "normal" if stride_enabled else "disabled"
        label_color = "black" if stride_enabled else "gray"

        self.trajectory_stride_entry.configure(state=widget_state)
        self.trajectory_stride_label.configure(foreground=label_color)

    def _build_progress_section(self):
        """Build progress monitoring section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Sweep Progress", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            frame, mode="determinate", maximum=100, length=400
        )
        self.progress_bar.pack(fill="x", pady=5)

        # Progress label
        self.progress_label = ttk.Label(frame, text="Ready")
        self.progress_label.pack(anchor="w", pady=2)

        # Info label directing users to main GUI logs
        log_info = ttk.Label(
            frame,
            text="📋 Sweep progress and results are logged to the main GUI's LOGS window",
            font=("TkDefaultFont", 9),
            foreground="blue",
        )
        log_info.pack(anchor="w", pady=(10, 2))

    def _toggle_smoothness_controls(self):
        """Enable/disable smoothness controls based on checkbox."""
        state = "normal" if self.smoothness_enabled_var.get() else "disabled"
        for widget in self.smoothness_widgets:
            widget.configure(state=state)

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

    def _sync_stability_to_main_gui(self, config):
        """Sync stability settings from config to main GUI's stability tab.

        Parameters
        ----------
        config : OptimizationConfig
            Config with stability settings to sync
        """
        if not self.gui_controller:
            return

        try:
            # Self-consistency settings
            if hasattr(self.gui_controller, "self_consistency_enabled_var"):
                self.gui_controller.self_consistency_enabled_var.set(
                    config.self_consistency_enabled
                )
            if hasattr(self.gui_controller, "self_consistency_target_ms_tolerance_var"):
                self.gui_controller.self_consistency_target_ms_tolerance_var.set(
                    f"{config.self_consistency_tolerance:.1e}"
                )
            if hasattr(self.gui_controller, "self_consistency_max_iterations_var"):
                self.gui_controller.self_consistency_max_iterations_var.set(
                    str(config.self_consistency_max_iterations)
                )
            if hasattr(self.gui_controller, "self_consistency_verbosity_var"):
                self.gui_controller.self_consistency_verbosity_var.set(
                    str(config.self_consistency_verbosity)
                )
            if hasattr(self.gui_controller, "self_consistency_chrono_interpolate_var"):
                self.gui_controller.self_consistency_chrono_interpolate_var.set(
                    config.self_consistency_chrono_interpolate
                )
            if hasattr(self.gui_controller, "self_consistency_chrono_tolerance_var"):
                self.gui_controller.self_consistency_chrono_tolerance_var.set(
                    f"{config.self_consistency_chrono_tolerance:.1e}"
                )
            if hasattr(
                self.gui_controller, "self_consistency_chrono_high_precision_var"
            ):
                self.gui_controller.self_consistency_chrono_high_precision_var.set(
                    config.self_consistency_chrono_high_precision
                )
            if hasattr(
                self.gui_controller, "self_consistency_chrono_adaptive_tolerance_var"
            ):
                self.gui_controller.self_consistency_chrono_adaptive_tolerance_var.set(
                    config.self_consistency_chrono_adaptive_tolerance
                )

            # Adaptive timestep settings
            if hasattr(self.gui_controller, "adaptive_timestep_enabled_var"):
                self.gui_controller.adaptive_timestep_enabled_var.set(
                    config.adaptive_timestep_enabled
                )
            if hasattr(self.gui_controller, "adaptive_timestep_threshold_var"):
                self.gui_controller.adaptive_timestep_threshold_var.set(
                    f"{config.adaptive_timestep_threshold:.2f}"
                )
            if hasattr(self.gui_controller, "adaptive_timestep_reduction_factor_var"):
                self.gui_controller.adaptive_timestep_reduction_factor_var.set(
                    str(config.adaptive_timestep_reduction_factor)
                )
            if hasattr(self.gui_controller, "adaptive_timestep_max_attempts_var"):
                self.gui_controller.adaptive_timestep_max_attempts_var.set(
                    str(config.adaptive_timestep_max_attempts)
                )
            if hasattr(self.gui_controller, "adaptive_timestep_min_factor_var"):
                self.gui_controller.adaptive_timestep_min_factor_var.set(
                    f"{config.adaptive_timestep_min_factor:.1e}"
                )
            if hasattr(self.gui_controller, "adaptive_timestep_cooldown_steps_var"):
                self.gui_controller.adaptive_timestep_cooldown_steps_var.set(
                    str(config.adaptive_timestep_cooldown_steps)
                )
            if hasattr(self.gui_controller, "adaptive_timestep_probe_threshold_var"):
                self.gui_controller.adaptive_timestep_probe_threshold_var.set(
                    f"{config.adaptive_timestep_probe_threshold:.2f}"
                )
            if hasattr(self.gui_controller, "adaptive_timestep_max_probe_steps_var"):
                self.gui_controller.adaptive_timestep_max_probe_steps_var.set(
                    str(config.adaptive_timestep_max_probe_steps)
                )
            if hasattr(self.gui_controller, "adaptive_timestep_debug_var"):
                self.gui_controller.adaptive_timestep_debug_var.set(
                    config.adaptive_timestep_debug
                )

            # Toggle controls to match loaded state
            if hasattr(self.gui_controller, "_toggle_self_consistency_controls"):
                self.gui_controller._toggle_self_consistency_controls()
            if hasattr(self.gui_controller, "_toggle_adaptive_timestep_controls"):
                self.gui_controller._toggle_adaptive_timestep_controls()

            self._log_result(
                "[OK] Stability settings synced to main GUI's Stability tab"
            )

        except Exception as e:
            self._log_result(
                f"[WARNING] Failed to sync some stability settings to main GUI: {e}"
            )

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

    def _parse_list_field(self, value: str) -> List[float]:
        """Parse comma-separated list of floats."""
        try:
            return [float(x.strip()) for x in value.split(",") if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid list format: {value}")

    def _parse_range_field(self, value: str) -> Optional[Tuple[float, float]]:
        """Parse range field (min, max) or return None if empty."""
        if not value or not value.strip():
            return None
        try:
            parts = [float(x.strip()) for x in value.split(",") if x.strip()]
            if len(parts) != 2:
                raise ValueError(f"Range must have exactly 2 values (min, max)")
            if parts[0] >= parts[1]:
                raise ValueError(f"Range min must be less than max")
            return (parts[0], parts[1])
        except ValueError as e:
            raise ValueError(f"Invalid range format: {value} - {e}")

    def _validate_inputs(self) -> Optional[str]:
        """Validate user inputs. Returns error message or None."""
        try:
            # Aperture range
            aperture_min = float(self.aperture_min_var.get())
            aperture_max = float(self.aperture_max_var.get())
            if aperture_min >= aperture_max:
                return "Aperture min must be less than max"
            if aperture_min <= 0:
                return "Aperture min must be positive"

            # Energy range
            energy_min = float(self.energy_min_var.get())
            energy_max = float(self.energy_max_var.get())
            if energy_min >= energy_max:
                return "Energy min must be less than max"
            if energy_min <= 0:
                return "Energy min must be positive"

            # Points
            aperture_points = int(self.aperture_points_var.get())
            energy_points = int(self.energy_points_var.get())
            if aperture_points < 2 or energy_points < 2:
                return "Must have at least 2 points per parameter"

            # Lists
            self._parse_list_field(self.offset_fractions_var.get())
            self._parse_list_field(self.start_z_var.get())

            # Wall and steps
            wall_z = float(self.wall_z_var.get())
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
            rider_stripped = float(self.rider_stripped_ions_var.get())
            if self.sim_type_var.get() == "BUNCH_TO_BUNCH":
                driver_stripped = float(self.driver_stripped_ions_var.get())

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
                f"[DEBUG] _gather_config: Existing config available (will be used as fallback)"
            )
        else:
            print(
                f"[DEBUG] _gather_config: No existing config, using defaults as fallback"
            )

        if has_gui:
            print(
                f"[DEBUG] _gather_config: Reading stability settings from main GUI Stability tab"
            )
        else:
            print(
                f"[DEBUG] _gather_config: No GUI available, using existing config or defaults"
            )

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
            aperture_points=int(self.aperture_points_var.get()),
            aperture_log_scale=self.aperture_log_var.get(),
            energy_range=(
                float(self.energy_min_var.get()),
                float(self.energy_max_var.get()),
            ),
            energy_points=int(self.energy_points_var.get()),
            energy_log_scale=self.energy_log_var.get(),
            transverse_offset_fractions=self._parse_list_field(
                self.offset_fractions_var.get()
            ),
            starting_z_positions=self._parse_list_field(self.start_z_var.get()),
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
            stripped_ions=float(self.rider_stripped_ions_var.get()),
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
            adaptive_timestep_max_attempts=self._get_gui_stability_setting(
                "adaptive_timestep_max_attempts_var",
                (
                    existing_config.adaptive_timestep_max_attempts
                    if existing_config
                    else 5
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
        )

        # Dynamically add sweepable parameter ranges after config creation
        config = config_obj
        if self.sweep_params["rider_transv_mom"]["sweep_var"].get():
            config.transverse_momentum_range = (
                float(self.sweep_params["rider_transv_mom"]["min_var"].get()),
                float(self.sweep_params["rider_transv_mom"]["max_var"].get()),
            )
            config.transverse_momentum_points = int(
                self.sweep_params["rider_transv_mom"]["points_var"].get()
            )

        if self.sweep_params["rider_transv_dist"]["sweep_var"].get():
            config.transverse_spread_range = (
                float(self.sweep_params["rider_transv_dist"]["min_var"].get()),
                float(self.sweep_params["rider_transv_dist"]["max_var"].get()),
            )
            config.transverse_spread_points = int(
                self.sweep_params["rider_transv_dist"]["points_var"].get()
            )

        # Add macroparticle sweeps if enabled
        if self.sweep_params["macroparticle_charge_multiplier"]["sweep_var"].get():
            config.macroparticle_charge_range = (
                float(
                    self.sweep_params["macroparticle_charge_multiplier"][
                        "min_var"
                    ].get()
                ),
                float(
                    self.sweep_params["macroparticle_charge_multiplier"][
                        "max_var"
                    ].get()
                ),
            )
            config.macroparticle_charge_points = int(
                self.sweep_params["macroparticle_charge_multiplier"]["points_var"].get()
            )

        if self.sweep_params["macroparticle_sigma_multiplier"]["sweep_var"].get():
            config.macroparticle_sigma_range = (
                float(
                    self.sweep_params["macroparticle_sigma_multiplier"]["min_var"].get()
                ),
                float(
                    self.sweep_params["macroparticle_sigma_multiplier"]["max_var"].get()
                ),
            )
            config.macroparticle_sigma_points = int(
                self.sweep_params["macroparticle_sigma_multiplier"]["points_var"].get()
            )

        return config

    def _on_load_from_main_config(self):
        """Load parameters from main GUI configuration."""
        if not self.gui_controller:
            _show_warning_dialog(
                self,
                "Load Config",
                "No main GUI controller available. Cannot load configuration.",
            )
            return

        try:
            # Get options from main GUI
            main_options = self.gui_controller._build_options_from_ui()

            # Update optimization config from main options
            opt_config = OptimizationConfig.from_simulation_options(main_options)

            # Update UI fields
            self.sim_type_var.set(opt_config.simulation_type.name)
            self.wall_z_var.set(str(opt_config.wall_z))
            self.cavity_spacing_var.set(str(opt_config.cavity_spacing))

            # Set timestep mode and values based on loaded config
            # Default to "duration" mode (auto-calc duration, user provides count)
            self.timestep_mode_var.set("duration")
            self.steps_var.set(str(opt_config.steps))
            self.duration_var.set(f"{opt_config.timestep:.2e}")
            self._toggle_timestep_mode()  # Update UI state
            self.sweep_params["rider_m_particle"]["fixed_var"].set(
                f"{opt_config.m_particle:.14e}"
            )
            self.sweep_params["rider_charge_sign"]["fixed_var"].set(
                str(opt_config.charge_sign)
            )
            self.sweep_params["rider_pcount"]["fixed_var"].set(str(opt_config.pcount))
            self.rider_stripped_ions_var.set(str(opt_config.stripped_ions))
            self.sweep_params["rider_transv_mom"]["fixed_var"].set(
                f"{opt_config.transv_mom:.2e}"
            )
            self.sweep_params["rider_transv_dist"]["fixed_var"].set(
                f"{opt_config.transv_dist:.2e}"
            )
            self.macroparticle_enabled_var.set(
                getattr(opt_config, "macroparticle_enabled", False)
            )
            self.macroparticle_charge_var.set(
                f"{getattr(opt_config, 'macroparticle_charge_multiplier', 1.0):.2e}"
            )
            self.macroparticle_position_var.set(
                f"{getattr(opt_config, 'macroparticle_position_spread', 0.0):.2e}"
            )
            self.macroparticle_momentum_var.set(
                f"{getattr(opt_config, 'macroparticle_momentum_spread', 0.0):.2e}"
            )
            self._toggle_macroparticle_controls()
            self._update_macroparticle_state()
            self.main_timestep_display_var.set(f"{opt_config.timestep:.2e}")

            # Update stability options if they exist in config
            if hasattr(opt_config, "smoothness_enabled"):
                self.smoothness_enabled_var.set(opt_config.smoothness_enabled)
                self.smoothness_window_var.set(str(opt_config.smoothness_window_size))
                self.smoothness_oscillation_var.set(
                    str(opt_config.smoothness_oscillation_threshold)
                )
                self.smoothness_reject_var.set(
                    opt_config.smoothness_reject_on_violation
                )
                self._toggle_smoothness_controls()

            self._log_result("[OK] Loaded parameters from main GUI configuration")
            self._log_result(f"  Simulation type: {opt_config.simulation_type.name}")
            self._log_result(f"  Wall z: {opt_config.wall_z} mm")
            self._log_result(f"  Cavity spacing: {opt_config.cavity_spacing} mm")
            self._log_result(
                f"  Timestep mode: auto-calc duration (user provides count)"
            )
            self._log_result(f"  Steps: {opt_config.steps}")
            self._log_result(f"  Duration: {opt_config.timestep:.2e} ns")
            self._log_result(f"  Particle mass: {opt_config.m_particle:.6e} amu")
            self._log_result(
                f"  Transverse momentum: {opt_config.transv_mom:.2e} amu·mm/ns"
            )
            self._log_result(f"  Transverse distance: {opt_config.transv_dist:.2e} mm")
            self._log_result("")
            self._log_result("[INFO] Stability options loaded from main config:")
            self._log_result(
                f"  Self-consistency: {opt_config.self_consistency_enabled} (tol={opt_config.self_consistency_tolerance:.1e})"
            )
            # Energy monitoring removed - functionality in adaptive timestep
            self._log_result(
                f"  Adaptive timestep: {opt_config.adaptive_timestep_enabled} (threshold={opt_config.adaptive_timestep_threshold * 100:.0f}%)"
            )
            self._log_result("")

            # Update internal config with loaded stability settings
            self.config = opt_config

        except Exception as e:
            _show_error_dialog(
                self,
                "Load Config Error",
                f"Failed to load configuration from main GUI:\n{e}",
            )
            import traceback

            self._log_result(f"[ERROR] Error loading main config: {e}")
            self._log_result(traceback.format_exc())

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

        ttk.Label(at_frame, text="Max reduction attempts:").pack(
            anchor="w", pady=(5, 0)
        )
        at_attempts_var = tk.StringVar(
            value=str(self.config.adaptive_timestep_max_attempts)
        )
        at_attempts_entry = ttk.Entry(at_frame, textvariable=at_attempts_var, width=15)
        at_attempts_entry.pack(anchor="w")
        all_widgets.append(at_attempts_entry)

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
            # Adaptive timestep: debug enabled, don't halt, reduced max attempts to fail faster
            at_debug_var.set(True)
            at_halt_var.set(False)
            at_attempts_var.set("3")

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
                self.config.adaptive_timestep_max_attempts = int(at_attempts_var.get())
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
                f"  Adaptive timestep: {self.config.adaptive_timestep_enabled} (threshold={self.config.adaptive_timestep_threshold * 100:.0f}%, reduction={self.config.adaptive_timestep_reduction_factor}x, max_attempts={self.config.adaptive_timestep_max_attempts}, debug={self.config.adaptive_timestep_debug})"
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

    def _load_config_from_path(self, filepath: str) -> None:
        """Load configuration from a specific file path.

        Parameters
        ----------
        filepath : str
            Full path to the JSON configuration file
        """
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Store the loaded config name for later use in results naming
            self.last_loaded_config = filepath

            # Update GUI config name field if available
            if self.gui_controller and hasattr(
                self.gui_controller, "sweep_config_name_var"
            ):
                from pathlib import Path

                config_name = Path(filepath).name
                self.gui_controller.sweep_config_name_var.set(config_name)

            # Populate UI fields
            self.sim_type_var.set(data.get("simulation_type", "CONDUCTING_WALL"))
            self.mode_var.set(data.get("mode", "blind_sweep"))
            self.aperture_min_var.set(str(data.get("aperture_min", 1e-5)))
            self.aperture_max_var.set(str(data.get("aperture_max", 1e-3)))
            self.aperture_points_var.set(str(data.get("aperture_points", 10)))
            self.aperture_log_var.set(data.get("aperture_log_scale", True))
            self.energy_min_var.set(str(data.get("energy_min", 1.0)))
            self.energy_max_var.set(str(data.get("energy_max", 1000.0)))
            self.energy_points_var.set(str(data.get("energy_points", 10)))
            self.energy_log_var.set(data.get("energy_log_scale", True))
            self.offset_fractions_var.set(
                ", ".join(
                    map(str, data.get("transverse_offset_fractions", [0.1, 0.3, 0.5]))
                )
            )
            self.start_z_var.set(
                ", ".join(map(str, data.get("starting_z_positions", [0.0])))
            )
            self.wall_z_var.set(str(data.get("wall_z", 100.0)))

            # Load wall_z sweep config if present
            if "wall_z_range" in data and data["wall_z_range"] is not None:
                wall_z_range = data["wall_z_range"]
                self.wall_z_min_var.set(str(wall_z_range[0]))
                self.wall_z_max_var.set(str(wall_z_range[1]))
                self.wall_z_points_var.set(str(data.get("wall_z_points", 3)))
                self.wall_z_sweep_var.set(True)
                self._toggle_wall_z_sweep()
            else:
                self.wall_z_sweep_var.set(False)
                self._toggle_wall_z_sweep()

            self.cavity_spacing_var.set(str(data.get("cavity_spacing", 1e5)))
            self.steps_var.set(str(data.get("steps", 2000)))
            self.objective_var.set(data.get("objective", "max_energy_gain"))

            # Load trajectory options (with backward compatibility)
            self.save_top_n_traj_var.set(
                data.get(
                    "save_top_n_trajectories", data.get("save_trajectories", False)
                )
            )
            self.save_all_traj_var.set(
                data.get(
                    "save_all_trajectories",
                    data.get("save_all_evaluation_trajectories", False),
                )
            )
            self.save_failed_traj_var.set(data.get("save_failed_trajectories", False))
            self.trajectory_stride_var.set(str(data.get("trajectory_stride", 10)))

            # Load metrics export options (with backward compatibility)
            # Map old boolean settings to new format/scope settings
            if "metrics_export_format" in data:
                self.metrics_format_var.set(data.get("metrics_export_format", "both"))
            else:
                # Backward compatibility: convert old CSV checkboxes to new format
                export_full = data.get(
                    "export_full_metrics_csv",
                    data.get(
                        "export_evaluation_csv", data.get("export_eval_csv", True)
                    ),
                )
                if export_full:
                    self.metrics_format_var.set(
                        "both"
                    )  # Default to both for backward compat
                else:
                    self.metrics_format_var.set("none")

            if "metrics_export_scope" in data:
                self.metrics_scope_var.set(data.get("metrics_export_scope", "all"))
            else:
                # Backward compatibility: check old top_n setting
                export_top_n = data.get("export_top_n_metrics_csv", False)
                self.metrics_scope_var.set("top_n" if export_top_n else "all")

            # Load log verbosity
            self.log_verbosity_var.set(data.get("log_verbosity", "truncated"))

            # Load optimization parameters
            self.optimization_method_var.set(
                data.get("optimization_method", "genetic_algorithm")
            )
            self.optimization_maxiter_var.set(str(data.get("optimization_maxiter", 50)))
            self.optimization_popsize_var.set(
                str(data.get("optimization_population_size", 20))
            )
            self.optimization_mutation_var.set(
                str(data.get("optimization_mutation_rate", 0.1))
            )
            self.optimization_crossover_var.set(
                str(data.get("optimization_crossover_rate", 0.7))
            )
            self.optimization_nstarts_var.set(str(data.get("optimization_n_starts", 5)))
            self.optimization_save_top_n_var.set(
                str(data.get("optimization_save_top_n", 3))
            )
            self.optimization_convergence_tol_var.set(
                str(data.get("optimization_convergence_tol", 1e-6))
            )
            self.optimization_convergence_patience_var.set(
                str(data.get("optimization_convergence_patience", 10))
            )

            # Update mode visibility
            self._update_mode_visibility()
            self._update_optimization_controls()

            # Load stability options - create temp config with file values only
            # Don't use _gather_config() here because it copies from existing self.config
            loaded_config = self._gather_config()

            # Override ALL stability settings from file (don't use existing config)
            loaded_config.self_consistency_enabled = data.get(
                "self_consistency_enabled", True
            )
            loaded_config.self_consistency_tolerance = data.get(
                "self_consistency_tolerance", 1e-4
            )
            loaded_config.self_consistency_max_iterations = data.get(
                "self_consistency_max_iterations", 5
            )
            loaded_config.self_consistency_verbosity = data.get(
                "self_consistency_verbosity", 0
            )
            loaded_config.self_consistency_chrono_interpolate = data.get(
                "self_consistency_chrono_interpolate", False
            )
            loaded_config.self_consistency_chrono_tolerance = data.get(
                "self_consistency_chrono_tolerance", 1e-3
            )
            loaded_config.self_consistency_chrono_high_precision = data.get(
                "self_consistency_chrono_high_precision", False
            )
            loaded_config.self_consistency_chrono_adaptive_tolerance = data.get(
                "self_consistency_chrono_adaptive_tolerance", False
            )
            # Energy monitoring removed - functionality in adaptive timestep
            loaded_config.energy_monitor_enabled = False
            loaded_config.energy_monitor_threshold = 2.0
            loaded_config.energy_monitor_check_interval = 10
            loaded_config.energy_monitor_halt_on_jump = data.get(
                "energy_monitor_halt_on_jump", False
            )
            loaded_config.energy_monitor_debug = False
            loaded_config.adaptive_timestep_enabled = data.get(
                "adaptive_timestep_enabled", True
            )
            loaded_config.adaptive_timestep_threshold = data.get(
                "adaptive_timestep_threshold", 0.10
            )
            loaded_config.adaptive_timestep_reduction_factor = data.get(
                "adaptive_timestep_reduction_factor", 10
            )
            loaded_config.adaptive_timestep_max_attempts = data.get(
                "adaptive_timestep_max_attempts", 5
            )
            loaded_config.adaptive_timestep_min_factor = data.get(
                "adaptive_timestep_min_factor", 1e-4
            )
            loaded_config.adaptive_timestep_cooldown_steps = data.get(
                "adaptive_timestep_cooldown_steps", 10
            )
            loaded_config.adaptive_timestep_probe_threshold = data.get(
                "adaptive_timestep_probe_threshold", 0.01
            )
            loaded_config.adaptive_timestep_max_probe_steps = data.get(
                "adaptive_timestep_max_probe_steps", 3
            )
            loaded_config.adaptive_timestep_debug = data.get(
                "adaptive_timestep_debug", False
            )
            # Sweep robustness options
            loaded_config.per_run_timeout = data.get("per_run_timeout", 300.0)
            loaded_config.skip_failed_runs = data.get("skip_failed_runs", True)
            # Trajectory stability checking options
            loaded_config.smoothness_enabled = data.get("smoothness_enabled", True)
            loaded_config.smoothness_window_size = data.get(
                "smoothness_window_size", 20
            )
            loaded_config.smoothness_oscillation_threshold = data.get(
                "smoothness_oscillation_threshold", 0.5
            )
            loaded_config.smoothness_trend_threshold = data.get(
                "smoothness_trend_threshold", 0.30
            )
            loaded_config.smoothness_reject_on_violation = data.get(
                "smoothness_reject_on_violation", True
            )
            loaded_config.smoothness_max_violations = data.get(
                "smoothness_max_violations", 3
            )
            # Macroparticle parameters
            loaded_config.macroparticle_enabled = data.get(
                "macroparticle_enabled", False
            )
            loaded_config.macroparticle_charge_multiplier = data.get(
                "macroparticle_charge_multiplier", 1.0
            )
            loaded_config.macroparticle_sigma_multiplier = data.get(
                "macroparticle_sigma_multiplier", 1.0
            )
            loaded_config.macroparticle_use_momentum_errors = data.get(
                "macroparticle_use_momentum_errors", True
            )

            # Load timestep strategy and related parameters
            # Default to auto_distance for sweeps/optimizations
            loaded_config.timestep_strategy = data.get(
                "timestep_strategy", "auto_distance"
            )
            loaded_config.target_distance_mm = data.get("target_distance_mm", 100.0)
            loaded_config.timestep = data.get("timestep", 3e-7)
            loaded_config.energy_scale_exponent = data.get("energy_scale_exponent", 1.0)

            print(
                f"[DEBUG] _load_config_from_path: Assigning loaded_config to self.config"
            )
            print(f"  SC enabled: {loaded_config.self_consistency_enabled}")
            print(f"  SC tolerance: {loaded_config.self_consistency_tolerance}")
            print(f"  AT enabled: {loaded_config.adaptive_timestep_enabled}")
            print(f"  AT debug: {loaded_config.adaptive_timestep_debug}")
            self.config = loaded_config

            # Update UI controls
            self.per_run_timeout_var.set(str(loaded_config.per_run_timeout))
            self.skip_failed_runs_var.set(loaded_config.skip_failed_runs)

            # Load UI-specific fields
            self.timestep_mode_var.set(data.get("timestep_mode", "duration"))
            self.auto_steps_distance_var.set(str(data.get("auto_steps_distance", 10.0)))
            self.trajectory_stride_var.set(str(data.get("trajectory_stride", 10)))
            self.rider_stripped_ions_var.set(str(data.get("rider_stripped_ions", 1.0)))
            self.driver_stripped_ions_var.set(
                str(data.get("driver_stripped_ions", 54.0))
            )
            self._toggle_timestep_mode()

            # Update stability controls (smoothness has UI variables)
            self.smoothness_enabled_var.set(loaded_config.smoothness_enabled)
            self.smoothness_window_var.set(str(loaded_config.smoothness_window_size))
            self.smoothness_oscillation_var.set(
                str(loaded_config.smoothness_oscillation_threshold)
            )
            self.smoothness_reject_var.set(loaded_config.smoothness_reject_on_violation)
            self._toggle_smoothness_controls()

            # Update main GUI stability tab if available
            if self.gui_controller:
                self._sync_stability_to_main_gui(loaded_config)

            self._log_result("[INFO] Additional stability settings loaded:")
            self._log_result(
                f"  Self-consistency max_iterations: {loaded_config.self_consistency_max_iterations}"
            )
            self._log_result(
                f"  Self-consistency verbosity: {loaded_config.self_consistency_verbosity}"
            )
            self._log_result(
                f"  Self-consistency chrono_interpolate: {loaded_config.self_consistency_chrono_interpolate}"
            )
            self._log_result(
                f"  Self-consistency chrono_tolerance: {loaded_config.self_consistency_chrono_tolerance:.1e} ns"
            )
            self._log_result(
                f"  Self-consistency chrono_high_precision: {loaded_config.self_consistency_chrono_high_precision}"
            )
            self._log_result(
                f"  Self-consistency chrono_adaptive_tolerance: {loaded_config.self_consistency_chrono_adaptive_tolerance}"
            )
            self._log_result(
                f"  Adaptive timestep reduction_factor: {loaded_config.adaptive_timestep_reduction_factor}"
            )
            self._log_result(
                f"  Adaptive timestep max_attempts: {loaded_config.adaptive_timestep_max_attempts}"
            )
            self._log_result(
                f"  Adaptive timestep min_factor: {loaded_config.adaptive_timestep_min_factor}"
            )
            self._log_result(
                f"  Adaptive timestep cooldown_steps: {loaded_config.adaptive_timestep_cooldown_steps}"
            )
            self._log_result(
                f"  Adaptive timestep probe_threshold: {loaded_config.adaptive_timestep_probe_threshold}"
            )
            self._log_result(
                f"  Adaptive timestep max_probe_steps: {loaded_config.adaptive_timestep_max_probe_steps}"
            )
            self._log_result(
                f"  Smoothness trend_threshold: {loaded_config.smoothness_trend_threshold}"
            )
            self._log_result(
                f"  Smoothness max_violations: {loaded_config.smoothness_max_violations}"
            )

            # Update macroparticle controls
            self.macroparticle_enabled_var.set(loaded_config.macroparticle_enabled)
            self.sweep_params["macroparticle_charge_multiplier"]["fixed_var"].set(
                str(loaded_config.macroparticle_charge_multiplier)
            )
            self.sweep_params["macroparticle_sigma_multiplier"]["fixed_var"].set(
                str(loaded_config.macroparticle_sigma_multiplier)
            )
            self.macroparticle_momentum_errors_var.set(
                loaded_config.macroparticle_use_momentum_errors
            )
            self._toggle_macroparticle_controls()

            # Load sweep parameter states dynamically
            sweep_state = data.get("sweep_parameters", {})
            for param_name, controls in self.sweep_params.items():
                if param_name in sweep_state:
                    state = sweep_state[param_name]
                    if state.get("enabled", False):
                        controls["sweep_var"].set(True)
                        controls["min_var"].set(str(state.get("min", "")))
                        controls["max_var"].set(str(state.get("max", "")))
                        controls["points_var"].set(str(state.get("points", "3")))
                        controls["log_var"].set(state.get("log", False))
                        self._toggle_sweep_controls(param_name)
                    else:
                        controls["sweep_var"].set(False)
                        fixed_val = state.get(
                            "fixed_value", controls["fixed_var"].get()
                        )
                        controls["fixed_var"].set(str(fixed_val))
                        self._toggle_sweep_controls(param_name)

            self._log_result("[OK] Configuration loaded and synced to main GUI")
            self._log_result("")
            self._log_result("=" * 60)
            self._log_result("LOADED STABILITY OPTIONS SUMMARY")
            self._log_result("=" * 60)
            self._log_result("[Self-Consistency]")
            self._log_result(f"  Enabled: {self.config.self_consistency_enabled}")
            self._log_result(
                f"  Tolerance: {self.config.self_consistency_tolerance:.1e}"
            )
            self._log_result(
                f"  Max iterations: {self.config.self_consistency_max_iterations}"
            )
            self._log_result(f"  Verbosity: {self.config.self_consistency_verbosity}")
            self._log_result(
                f"  Chrono interpolate: {self.config.self_consistency_chrono_interpolate}"
            )
            self._log_result(
                f"  Chrono tolerance: {self.config.self_consistency_chrono_tolerance:.1e} ns"
            )
            self._log_result(
                f"  Chrono high precision: {self.config.self_consistency_chrono_high_precision}"
            )
            self._log_result(
                f"  Chrono adaptive tolerance: {self.config.self_consistency_chrono_adaptive_tolerance}"
            )
            self._log_result("")
            self._log_result("[Adaptive Timestep]")
            self._log_result(f"  Enabled: {self.config.adaptive_timestep_enabled}")
            self._log_result(
                f"  Threshold: {self.config.adaptive_timestep_threshold * 100:.0f}%"
            )
            self._log_result(
                f"  Reduction factor: {self.config.adaptive_timestep_reduction_factor}x"
            )
            self._log_result(
                f"  Max attempts: {self.config.adaptive_timestep_max_attempts}"
            )
            self._log_result(
                f"  Min factor: {self.config.adaptive_timestep_min_factor}"
            )
            self._log_result(
                f"  Cooldown steps: {self.config.adaptive_timestep_cooldown_steps}"
            )
            self._log_result(
                f"  Probe threshold: {self.config.adaptive_timestep_probe_threshold}"
            )
            self._log_result(
                f"  Max probe steps: {self.config.adaptive_timestep_max_probe_steps}"
            )
            self._log_result(f"  Debug: {self.config.adaptive_timestep_debug}")
            self._log_result("")
            self._log_result("[Trajectory Smoothness Analysis]")
            self._log_result(f"  Enabled: {self.config.smoothness_enabled}")
            self._log_result(f"  Window size: {self.config.smoothness_window_size}")
            self._log_result(
                f"  Oscillation threshold: {self.config.smoothness_oscillation_threshold}"
            )
            self._log_result(
                f"  Trend threshold: {self.config.smoothness_trend_threshold}"
            )
            self._log_result(
                f"  Reject on violation: {self.config.smoothness_reject_on_violation}"
            )
            self._log_result(
                f"  Max violations: {self.config.smoothness_max_violations}"
            )
            self._log_result("")
            self._log_result("=" * 60)
            self._log_result("")
            self._log_result(
                "NOTE: Stability settings are synced to main GUI's Stability tab"
            )
            self._log_result("      View/edit them in the main GUI's Stability tab")
            self._log_result(
                "      Log verbosity setting will override debug flags during run"
            )
            self._log_result("")

            # Auto-switch to sweep mode when loading a sweep/optimization config
            if self.gui_controller and hasattr(self.gui_controller, "run_mode_var"):
                self.gui_controller.run_mode_var.set("sweep")
                if hasattr(self.gui_controller, "_on_run_mode_changed"):
                    self.gui_controller._on_run_mode_changed()
                self._log_result(
                    "[INFO] Auto-switched main GUI to Sweep/Optim run mode"
                )

        except Exception as e:
            _show_error_dialog(self, "Load Error", f"Failed to load config: {e}")

    def _on_load_config(self):
        """Load configuration from JSON file via dialog."""
        import os

        # Use sweep config directory from GUI preferences
        os.makedirs(self.sweep_config_dir, exist_ok=True)

        filename = filedialog.askopenfilename(
            title="Load Optimization Config",
            initialdir=self.sweep_config_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return

        self._load_config_from_path(filename)

    def _save_config_to_path(self, filepath: str) -> bool:
        """Save configuration to specified path.

        Parameters
        ----------
        filepath : str
            Full path where config should be saved

        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        error = self._validate_inputs()
        if error:
            _show_error_dialog(self, "Invalid Input", f"Cannot save: {error}")
            return False

        try:
            print(f"[DEBUG] _save_config_to_path: Gathering config for save")
            config = self._gather_config()
            print(f"[DEBUG] After _gather_config:")
            print(f"  SC enabled: {config.self_consistency_enabled}")
            print(f"  SC tolerance: {config.self_consistency_tolerance}")
            print(
                f"  SC chrono interpolate: {config.self_consistency_chrono_interpolate}"
            )
            print(f"  SC chrono tolerance: {config.self_consistency_chrono_tolerance}")
            print(
                f"  SC chrono high precision: {config.self_consistency_chrono_high_precision}"
            )
            print(
                f"  SC chrono adaptive tolerance: {config.self_consistency_chrono_adaptive_tolerance}"
            )
            print(f"  AT enabled: {config.adaptive_timestep_enabled}")
            print(f"  AT debug: {config.adaptive_timestep_debug}")
            data = {
                "simulation_type": config.simulation_type.name,
                "mode": config.mode,
                "aperture_min": config.aperture_range[0],
                "aperture_max": config.aperture_range[1],
                "aperture_points": config.aperture_points,
                "aperture_log_scale": config.aperture_log_scale,
                "energy_min": config.energy_range[0],
                "energy_max": config.energy_range[1],
                "energy_points": config.energy_points,
                "energy_log_scale": config.energy_log_scale,
                "transverse_offset_fractions": config.transverse_offset_fractions,
                "starting_z_positions": config.starting_z_positions,
                "wall_z": config.wall_z,
                "wall_z_range": config.wall_z_range,
                "wall_z_points": config.wall_z_points,
                "cavity_spacing": config.cavity_spacing,
                "steps": config.steps,
                "objective": config.objective,
                # Trajectory saving options
                "save_top_n_trajectories": config.save_top_n_trajectories,
                "save_all_trajectories": config.save_all_trajectories,
                "save_failed_trajectories": config.save_failed_trajectories,
                "trajectory_stride": config.trajectory_stride,
                # Metrics export options
                "metrics_export_format": config.metrics_export_format,
                "metrics_export_scope": config.metrics_export_scope,
                # Log verbosity
                "log_verbosity": config.log_verbosity,
                # Optimization parameters
                "optimization_method": config.optimization_method,
                "optimization_maxiter": config.optimization_maxiter,
                "optimization_population_size": config.optimization_population_size,
                "optimization_mutation_rate": config.optimization_mutation_rate,
                "optimization_crossover_rate": config.optimization_crossover_rate,
                "optimization_n_starts": config.optimization_n_starts,
                "optimization_save_top_n": config.optimization_save_top_n,
                "optimization_convergence_tol": config.optimization_convergence_tol,
                "optimization_convergence_patience": config.optimization_convergence_patience,
                # Stability options
                "self_consistency_enabled": config.self_consistency_enabled,
                "self_consistency_tolerance": config.self_consistency_tolerance,
                "self_consistency_max_iterations": config.self_consistency_max_iterations,
                "self_consistency_verbosity": config.self_consistency_verbosity,
                "self_consistency_chrono_interpolate": config.self_consistency_chrono_interpolate,
                "self_consistency_chrono_tolerance": config.self_consistency_chrono_tolerance,
                "self_consistency_chrono_high_precision": config.self_consistency_chrono_high_precision,
                "self_consistency_chrono_adaptive_tolerance": config.self_consistency_chrono_adaptive_tolerance,
                # Energy monitoring removed - halt option in adaptive timestep
                "energy_monitor_halt_on_jump": config.energy_monitor_halt_on_jump,
                "adaptive_timestep_enabled": config.adaptive_timestep_enabled,
                "adaptive_timestep_threshold": config.adaptive_timestep_threshold,
                "adaptive_timestep_reduction_factor": config.adaptive_timestep_reduction_factor,
                "adaptive_timestep_max_attempts": config.adaptive_timestep_max_attempts,
                "adaptive_timestep_min_factor": config.adaptive_timestep_min_factor,
                "adaptive_timestep_cooldown_steps": config.adaptive_timestep_cooldown_steps,
                "adaptive_timestep_probe_threshold": config.adaptive_timestep_probe_threshold,
                "adaptive_timestep_max_probe_steps": config.adaptive_timestep_max_probe_steps,
                "adaptive_timestep_debug": config.adaptive_timestep_debug,
                # Sweep robustness options
                "per_run_timeout": config.per_run_timeout,
                "skip_failed_runs": config.skip_failed_runs,
                # Trajectory stability checking
                "smoothness_enabled": config.smoothness_enabled,
                "smoothness_window_size": config.smoothness_window_size,
                "smoothness_oscillation_threshold": config.smoothness_oscillation_threshold,
                "smoothness_trend_threshold": config.smoothness_trend_threshold,
                "smoothness_reject_on_violation": config.smoothness_reject_on_violation,
                "smoothness_max_violations": config.smoothness_max_violations,
                # Macroparticle parameters
                "macroparticle_enabled": config.macroparticle_enabled,
                "macroparticle_charge_multiplier": config.macroparticle_charge_multiplier,
                "macroparticle_sigma_multiplier": config.macroparticle_sigma_multiplier,
                "macroparticle_use_momentum_errors": config.macroparticle_use_momentum_errors,
                # Timestep strategy parameters
                "timestep_strategy": config.timestep_strategy,
                "target_distance_mm": config.target_distance_mm,
                "timestep": config.timestep,
                "energy_scale_exponent": config.energy_scale_exponent,
                # UI-specific fields
                "timestep_mode": self.timestep_mode_var.get(),
                "auto_steps_distance": float(self.auto_steps_distance_var.get()),
                "rider_stripped_ions": float(self.rider_stripped_ions_var.get()),
                "driver_stripped_ions": float(self.driver_stripped_ions_var.get()),
            }

            # Dynamically save all sweep parameter states
            sweep_state = {}
            for param_name, controls in self.sweep_params.items():
                if controls["sweep_var"].get():
                    sweep_state[param_name] = {
                        "enabled": True,
                        "min": controls["min_var"].get(),
                        "max": controls["max_var"].get(),
                        "points": controls["points_var"].get(),
                        "log": controls["log_var"].get(),
                    }
                else:
                    sweep_state[param_name] = {
                        "enabled": False,
                        "fixed_value": controls["fixed_var"].get(),
                    }
            data["sweep_parameters"] = sweep_state

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            # Update last_loaded_config so sweep results use correct name
            self.last_loaded_config = filepath

            self._log_result(f"[OK] Configuration saved to {filepath}")
            print(f"[DEBUG] Chrono settings saved to config:")
            print(f"  chrono_interpolate: {config.self_consistency_chrono_interpolate}")
            print(f"  chrono_tolerance: {config.self_consistency_chrono_tolerance}")
            print(
                f"  chrono_high_precision: {config.self_consistency_chrono_high_precision}"
            )
            print(
                f"  chrono_adaptive_tolerance: {config.self_consistency_chrono_adaptive_tolerance}"
            )
            return True
        except Exception as e:
            _show_error_dialog(self, "Save Error", f"Failed to save config: {e}")
            return False

    def _on_save_config(self):
        """Save configuration to JSON file using file dialog."""
        error = self._validate_inputs()
        if error:
            _show_error_dialog(self, "Invalid Input", f"Cannot save: {error}")
            return

        import os

        # Use sweep config directory from GUI preferences
        os.makedirs(self.sweep_config_dir, exist_ok=True)

        filename = filedialog.asksaveasfilename(
            title="Save Optimization Config",
            initialdir=self.sweep_config_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return

        # Use the new save method
        success = self._save_config_to_path(filename)

        # Update GUI if we have a controller and save was successful
        if (
            success
            and self.gui_controller
            and hasattr(self.gui_controller, "sweep_config_name_var")
        ):
            from pathlib import Path

            config_name = Path(filename).name
            self.gui_controller.sweep_config_name_var.set(config_name)
            self.gui_controller.current_sweep_config_label.config(
                text=config_name, foreground="black", font=("TkDefaultFont", 9)
            )
            self.gui_controller._refresh_sweep_config_list(selected=config_name)

    def _on_view_results(self):
        """Display pre-generated summary plots from the latest sweep/optimization run."""
        import glob
        import os
        from pathlib import Path

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

        # Also check legacy location
        legacy_dir = "optimization_results"
        if os.path.exists(legacy_dir):
            result_dirs.extend(
                [
                    d
                    for d in glob.glob(os.path.join(legacy_dir, "*"))
                    if os.path.isdir(d)
                ]
            )

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

            # Try to detect file format
            results = None

            if "results" in data:
                # Sweep format: sweep_results.json
                results = data.get("results", [])
                if not results:
                    _show_info_dialog(self, "No Results", "No results found in file.")
                    return

                # Check if we have trajectory data for plotting
                results_with_traj = [r for r in results if "trajectory" in r]

                if not results_with_traj:
                    # Show metrics summary even without trajectories
                    self._show_results_summary(results, file_path)
                    return

            elif "all_evaluations" in data or "best_parameters" in data:
                # Optimization format: optimization_results.json
                # Check for NPZ trajectory files in the same directory
                import os

                results_dir = os.path.dirname(file_path)
                self._view_npz_trajectories(results_dir)
                return

            elif "core" in data and "rider" in data["core"]:
                # Legacy format: single trajectory file
                results_with_traj = [self._convert_legacy_trajectory(data)]

            else:
                _show_info_dialog(
                    self,
                    "Unknown Format",
                    "Cannot parse this file format.\n\n"
                    "Expected either:\n"
                    "- sweep_results.json with 'results' array\n"
                    "- optimization_results.json with 'all_evaluations'\n"
                    "- Legacy trajectory file with 'core'/'rider' structure",
                )
                return

            # Create trajectory viewer dialog and automatically plot
            self._show_trajectory_viewer(results_with_traj, file_path, auto_plot=True)

        except Exception as e:
            import traceback

            _show_error_dialog(
                self,
                "Error Loading File",
                f"Failed to load file:\n{e}\n\n{traceback.format_exc()}",
            )

    def _on_plot_trajectories(self):
        """Open trajectory plotting dialog to visualize saved results."""
        # Default to optimization_results directory
        import glob
        import os

        # Use sweep output directory from GUI preferences, then fall back to legacy
        legacy_results_dir = "optimization_results"

        # Start with base directory
        if os.path.exists(self.sweep_output_dir) and os.listdir(self.sweep_output_dir):
            base_dir = self.sweep_output_dir
        elif os.path.exists(legacy_results_dir):
            base_dir = legacy_results_dir
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

            # Try to detect file format
            results = None

            if "results" in data:
                # Sweep format: sweep_results.json with "results" array
                results = data.get("results", [])
                if not results:
                    _show_info_dialog(self, "No Results", "No results found in file.")
                    return

                # Filter results with trajectories
                results_with_traj = [r for r in results if "trajectory" in r]

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

            elif "all_evaluations" in data or "best_parameters" in data:
                # Optimization format: optimization_results.json
                # This file contains metrics only, not trajectories
                # Load NPZ trajectory files from the same directory
                import os

                results_dir = os.path.dirname(file_path)
                self._view_npz_trajectories(results_dir)
                return

            elif "core" in data and "rider" in data["core"]:
                # Legacy format: single trajectory file
                results_with_traj = [self._convert_legacy_trajectory(data)]

            else:
                _show_info_dialog(
                    self,
                    "Unknown Format",
                    "Cannot parse this file format.\n\n"
                    "Expected either:\n"
                    "- sweep_results.json with 'results' array\n"
                    "- optimization_results.json with 'all_evaluations'\n"
                    "- Legacy trajectory file with 'core'/'rider' structure\n\n"
                    "Note: CSV files only contain metrics, not trajectory data.",
                )
                return

            # Create trajectory viewer dialog
            self._show_trajectory_viewer(results_with_traj, file_path)

        except Exception as e:
            import traceback

            _show_error_dialog(
                self,
                "Error Loading File",
                f"Failed to load file:\n{e}\n\n{traceback.format_exc()}",
            )

    def _convert_legacy_trajectory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert legacy trajectory format to sweep results format."""
        # Extract trajectory data from legacy format
        rider_data = data.get("core", {}).get("rider", {})

        # Get positions
        positions = rider_data.get("positions_mm", {})
        x = positions.get("x", [])
        y = positions.get("y", [])
        z_pos = positions.get("z", [])

        # Calculate r from x and y
        if x and y:
            r = [np.sqrt(xi**2 + yi**2) for xi, yi in zip(x, y)]
        else:
            r = []

        # Get momenta
        momenta = rider_data.get("conjugate_momenta", {})
        pz = momenta.get("Pz", [])
        px = momenta.get("Px", [])
        py = momenta.get("Py", [])

        # Calculate pr from px and py
        if px and py:
            pr = [np.sqrt(pxi**2 + pyi**2) for pxi, pyi in zip(px, py)]
        else:
            pr = []

        # Get time
        t = rider_data.get("time_ns", [])

        # Get gamma history for energy calculation
        gamma_hist = rider_data.get("gamma_hist", [])

        # Calculate metrics
        if gamma_hist:
            gamma_initial = gamma_hist[0] if len(gamma_hist) > 0 else 1.0
            gamma_final = gamma_hist[-1] if len(gamma_hist) > 0 else 1.0
            delta_e_mev = (gamma_final - gamma_initial) * 0.511  # For electrons
        else:
            gamma_initial = 1.0
            gamma_final = 1.0
            delta_e_mev = 0.0

        # Build result in sweep format
        result = {
            "run_number": 1,
            "parameters": {
                "aperture_radius": data.get("aperture_radius", 0),
                "particle_energy_gev": (gamma_initial - 1)
                * 0.511
                / 1000.0,  # Convert to GeV
                "start_z": z_pos[0] if z_pos else 0,
                "wall_z": data.get("wall_z", 0),
                "simulation_type": data.get("simulation_type", "UNKNOWN"),
            },
            "metrics": {
                "rider_delta_e_mev": delta_e_mev,
                "rider_gamma_initial": gamma_initial,
                "rider_gamma_final": gamma_final,
            },
            "trajectory": {
                "z": z_pos,
                "r": r,
                "pz": pz,
                "pr": pr,
                "t": t,
            },
        }

        return result

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
                params = r.get("parameters", {})
                metrics = r.get("metrics", {})
                dist_info = r.get("_distance_info", {})

                run_num = r.get("run_number", "?")
                aperture = params.get("aperture_radius", 0)
                energy = params.get("particle_energy_gev", 0)
                start_z = params.get("starting_z", 0)
                delta_e = metrics.get("rider_delta_e_mev", 0)

                # Calculate traveled distance
                z_start = dist_info.get("z_start", 0)
                z_end = dist_info.get("z_end", 0)
                traveled = abs(z_end - z_start)

                # Get gamma from metrics
                gamma = metrics.get("rider_gamma_initial", 0)

                if has_beam_optics:
                    # Include beam optics columns
                    emit_x = metrics.get("rider_emittance_x_mm_mrad", 0)
                    norm_emit_x = metrics.get("rider_norm_emittance_x_mm_mrad", 0)
                    beta_x = metrics.get("rider_beta_x_m", 0)
                    row = f"{run_num:<5} {aperture:<15.3e} {energy:<15.2f} {start_z:<15.1f} {delta_e:<12.3f} {traveled:<15.1f} {gamma:<12.1f} {emit_x:<15.3e} {norm_emit_x:<16.3e} {beta_x:<12.3e}\n"
                else:
                    row = f"{run_num:<5} {aperture:<15.3e} {energy:<15.2f} {start_z:<15.1f} {delta_e:<12.3f} {traveled:<15.1f} {gamma:<12.1f}\n"
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
            import numpy as np
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )

            # Extract data
            apertures = []
            energies = []
            delta_es = []

            for r in results:
                params = r.get("parameters", {})
                metrics = r.get("metrics", {})
                apertures.append(params.get("aperture_radius", 0))
                energies.append(params.get("particle_energy_gev", 0))
                delta_es.append(metrics.get("rider_delta_e_mev", 0))

            # Create figure
            fig = plt.figure(figsize=(10, 6))

            # Determine if we have 1D or 2D sweep
            unique_apertures = len(set(apertures))
            unique_energies = len(set(energies))

            if unique_apertures > 1 and unique_energies > 1:
                # 2D sweep - make heatmap
                ax = fig.add_subplot(111)

                # Reshape data
                apertures_arr = np.array(apertures)
                energies_arr = np.array(energies)
                delta_es_arr = np.array(delta_es)

                # Create grid
                unique_a = sorted(set(apertures))
                unique_e = sorted(set(energies))
                grid = np.zeros((len(unique_e), len(unique_a)))

                for i, r in enumerate(results):
                    params = r.get("parameters", {})
                    a_val = params.get("aperture_radius", 0)
                    e_val = params.get("particle_energy_gev", 0)
                    de_val = delta_es[i]

                    a_idx = unique_a.index(a_val)
                    e_idx = unique_e.index(e_val)
                    grid[e_idx, a_idx] = de_val

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
                    params = r.get("parameters", {})
                    metrics = r.get("metrics", {})
                    dist_info = r.get("_distance_info", {})

                    run_num = r.get("run_number", "")
                    aperture = params.get("aperture_radius", 0)
                    energy = params.get("particle_energy_gev", 0)
                    start_z = params.get("starting_z", 0)
                    delta_e = metrics.get("rider_delta_e_mev", 0)

                    z_start = dist_info.get("z_start", 0)
                    z_end = dist_info.get("z_end", 0)
                    traveled = abs(z_end - z_start)

                    gamma_i = metrics.get("rider_gamma_initial", 0)
                    gamma_f = metrics.get("rider_gamma_final", 0)

                    # Beam optics metrics
                    emit_x = metrics.get("rider_emittance_x_mm_mrad", "")
                    emit_y = metrics.get("rider_emittance_y_mm_mrad", "")
                    norm_emit_x = metrics.get("rider_norm_emittance_x_mm_mrad", "")
                    norm_emit_y = metrics.get("rider_norm_emittance_y_mm_mrad", "")
                    beta_x = metrics.get("rider_beta_x_m", "")
                    beta_y = metrics.get("rider_beta_y_m", "")

                    writer.writerow(
                        [
                            run_num,
                            aperture,
                            energy,
                            start_z,
                            delta_e,
                            traveled,
                            gamma_i,
                            gamma_f,
                            emit_x,
                            emit_y,
                            norm_emit_x,
                            norm_emit_y,
                            beta_x,
                            beta_y,
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

            # Collect data for heatmap
            apertures = []
            energies = []
            delta_es = []

            # Plot each selected trajectory
            for idx, result in enumerate(selected_results):
                traj = result.get("trajectory", {})
                params = result.get("parameters", {})
                metrics = result.get("metrics", {})
                run_num = result.get("run_number", "?")

                z = np.array(traj.get("z", []))
                r = np.array(traj.get("r", []))
                pz = np.array(traj.get("pz", []))
                t = np.array(traj.get("t", []))

                if len(z) == 0:
                    continue

                aperture = params.get("aperture_radius", 0)
                energy = params.get("particle_energy_gev", 0)
                delta_e_mev = metrics.get("rider_delta_e_mev", 0)
                gamma_initial = metrics.get("rider_gamma_initial", 1)
                gamma_final = metrics.get("rider_gamma_final", 1)

                label = f"Run #{run_num} (a={aperture:.2e}mm, E={energy:.1f}GeV)"
                color = plt.cm.tab10(idx % 10)

                # Calculate energy from gamma (E = (gamma - 1) * m * c^2)
                # For electrons: m*c^2 = 0.511 MeV
                energy_mev_initial = (gamma_initial - 1) * 0.511
                energy_mev_final = (gamma_final - 1) * 0.511

                # Calculate energy at each point along trajectory
                # Approximate: E(z) ≈ E_initial + ΔE * (z - z_0) / (z_final - z_0)
                if len(z) > 1:
                    z_range = z[-1] - z[0]
                    if abs(z_range) > 1e-6:
                        energy_mev = (
                            energy_mev_initial + delta_e_mev * (z - z[0]) / z_range
                        )
                    else:
                        energy_mev = np.full_like(z, energy_mev_initial)
                else:
                    energy_mev = np.array([energy_mev_initial])

                # Plot 1: Delta E versus z
                ax_delta_e.plot(
                    z,
                    energy_mev - energy_mev_initial,
                    label=label,
                    alpha=0.7,
                    color=color,
                    linewidth=1.5,
                )

                # Plot 2: x and y positions versus z (need to extract from r)
                # Since we only have r (radial distance), we'll plot r and -r to show transverse extent
                # In a real case, you'd have separate x and y coordinates
                ax_transverse.plot(
                    z, r, label=f"{label} (+r)", alpha=0.6, color=color, linewidth=1.5
                )
                ax_transverse.plot(
                    z, -r, alpha=0.3, color=color, linewidth=1.0, linestyle="--"
                )

                # Collect data for heatmap
                apertures.append(aperture)
                energies.append(energy)
                delta_es.append(delta_e_mev)

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

    def _run_optimization_background(self):
        """Delegate optimization background execution to mixin."""
        return super()._run_optimization_background()

    def _save_optimization_results(self, result, param_names):
        """Save optimization results to file via shared helper."""
        return save_optimization_results(self, result, param_names)

    def _save_top_trajectories_summary_table(self, result, param_names, output_dir):
        """Generate and save top trajectories summary via helper."""
        return save_top_trajectories_summary_table(
            self, result, param_names, output_dir
        )

    def _generate_optimization_plots(self, result, param_names, output_dir):
        """Generate optimization plots via shared helper."""
        return generate_optimization_plots(self, result, param_names, output_dir)

    def _generate_optimization_heatmap(self, all_evaluations, param_names, output_dir):
        """Generate optimization heatmap via shared helper."""
        return generate_optimization_heatmap(
            self, all_evaluations, param_names, output_dir
        )

    def _save_top_n_optimization_trajectories(self, result, param_names):
        """Re-run top N parameter sets and save trajectories via helper."""
        return save_top_n_optimization_trajectories(self, result, param_names)

    def _save_single_optimization_trajectory(
        self, params_dict, param_names, rank, fitness
    ):
        """Re-run a single parameter set and save its trajectory.

        Parameters
        ----------
        params_dict : dict
            Dictionary of parameter names to values
        param_names : list
            List of parameter names
        rank : int
            Rank of this parameter set (1 = best, 2 = second best, etc.)
        fitness : float
            Fitness value (objective function value to minimize)

        Returns
        -------
        dict or None
            Trajectory data dictionary if successful, None otherwise
        """
        from pathlib import Path

        import numpy as np

        try:
            # Set up run parameters (similar to evaluate_params)
            aperture = self.config.aperture_range[0]
            energy = self.config.energy_range[0]
            start_z = (
                self.config.starting_z_positions[0]
                if self.config.starting_z_positions
                else 0.0
            )
            offset_frac = (
                self.config.transverse_offset_fractions[0]
                if self.config.transverse_offset_fractions
                else 0.0
            )
            timestep = self.config.timestep
            steps = self.config.steps
            wall_z = self.config.wall_z

            # Map parameters
            for param_name, value in params_dict.items():
                if param_name == "aperture_radius":
                    aperture = value
                elif param_name == "initial_energy_gev":
                    energy = value
                elif param_name == "start_z":
                    start_z = value
                elif param_name == "transverse_offset":
                    offset_frac = value
                elif param_name == "timestep":
                    timestep = value
                elif param_name == "wall_z":
                    wall_z = value

            transv_offset = offset_frac * aperture

            # Temporarily enable trajectory saving
            save_all_backup = self.config.save_all_trajectories
            self.config.save_all_trajectories = True

            # Run integration
            result_data = self._run_single_integration(
                aperture=aperture,
                energy_gev=energy,
                start_z=start_z,
                transv_offset=transv_offset,
                timestep=timestep,
                steps=steps,
                rider_m_particle=self.config.m_particle,
                rider_charge_sign=self.config.charge_sign,
                rider_pcount=int(self.config.pcount),
                rider_transv_mom=self.config.transv_mom,
                driver_params=None,
                wall_z=wall_z,
                run_num=9999 + rank,  # Special run number for trajectory
            )

            # Restore trajectory setting
            self.config.save_all_trajectories = save_all_backup

            if result_data and "trajectory" in result_data:
                # Use the timestamped directory from _save_optimization_results
                output_dir = getattr(
                    self, "_last_optimization_dir", Path(self.config.output_dir)
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                # Plot trajectory
                import matplotlib.pyplot as plt

                traj = result_data["trajectory"]
                metrics = result_data.get("metrics", {})

                fig, axes = plt.subplots(3, 2, figsize=(14, 14))

                # Extract trajectory arrays
                z = np.array(traj["z"])
                t = np.array(traj["t"])
                r = np.array(traj["r"])
                gamma_arr = np.array(traj.get("gamma", []))
                pr = np.array(traj.get("pr", []))

                # Calculate delta_e and percent_delta_e from gamma
                if len(gamma_arr) > 0:
                    gamma_initial = gamma_arr[0]
                    delta_gamma = gamma_arr - gamma_initial
                    # Energy in MeV for electrons
                    delta_e_mev = delta_gamma * 0.511
                    percent_delta_e = (delta_gamma / gamma_initial) * 100.0
                else:
                    delta_e_mev = np.zeros_like(z)
                    percent_delta_e = np.zeros_like(z)

                # Row 1, Col 1: z vs t
                axes[0, 0].plot(t, z, "b-", linewidth=1.5)
                axes[0, 0].set_xlabel("Time (ns)", fontsize=10)
                axes[0, 0].set_ylabel("z (mm)", fontsize=10)
                axes[0, 0].set_title(
                    "Longitudinal Position", fontsize=11, fontweight="bold"
                )
                axes[0, 0].grid(True, alpha=0.3)

                # Row 1, Col 2: r vs z
                axes[0, 1].plot(z, r * 1e3, "r-", linewidth=1.5)
                axes[0, 1].set_xlabel("z (mm)", fontsize=10)
                axes[0, 1].set_ylabel("r (μm)", fontsize=10)
                axes[0, 1].set_title(
                    "Transverse Position (Radial)", fontsize=11, fontweight="bold"
                )
                axes[0, 1].grid(True, alpha=0.3)

                # Row 2, Col 1: gamma vs z (with adaptive scaling)
                if len(gamma_arr) > 0:
                    axes[1, 0].plot(z, gamma_arr, "g-", linewidth=1.5)
                    axes[1, 0].set_xlabel("z (mm)", fontsize=10)
                    axes[1, 0].set_ylabel("γ", fontsize=10)
                    axes[1, 0].set_title(
                        "Lorentz Factor", fontsize=11, fontweight="bold"
                    )
                    axes[1, 0].grid(True, alpha=0.3)
                    # Auto-scale y-axis to show variations
                    gamma_mean = np.mean(gamma_arr)
                    gamma_range = np.max(gamma_arr) - np.min(gamma_arr)
                    if gamma_range > 0:
                        margin = max(gamma_range * 0.1, gamma_mean * 0.001)
                        axes[1, 0].set_ylim(
                            [np.min(gamma_arr) - margin, np.max(gamma_arr) + margin]
                        )

                # Row 2, Col 2: Delta E (MeV) vs z
                axes[1, 1].plot(z, delta_e_mev, "orange", linewidth=1.5)
                axes[1, 1].set_xlabel("z (mm)", fontsize=10)
                axes[1, 1].set_ylabel("ΔE (MeV)", fontsize=10)
                axes[1, 1].set_title("Energy Change", fontsize=11, fontweight="bold")
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].axhline(
                    y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5
                )

                # Row 3, Col 1: Percent Delta E vs z
                axes[2, 0].plot(z, percent_delta_e, "purple", linewidth=1.5)
                axes[2, 0].set_xlabel("z (mm)", fontsize=10)
                axes[2, 0].set_ylabel("ΔE/E (%)", fontsize=10)
                axes[2, 0].set_title(
                    "Percent Energy Change", fontsize=11, fontweight="bold"
                )
                axes[2, 0].grid(True, alpha=0.3)
                axes[2, 0].axhline(
                    y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5
                )

                # Row 3, Col 2: pr vs z
                if len(pr) > 0:
                    axes[2, 1].plot(z, pr, "m-", linewidth=1.5)
                    axes[2, 1].set_xlabel("z (mm)", fontsize=10)
                    axes[2, 1].set_ylabel("pr (amu·mm/ns)", fontsize=10)
                    axes[2, 1].set_title(
                        "Transverse Momentum (Radial)", fontsize=11, fontweight="bold"
                    )
                    axes[2, 1].grid(True, alpha=0.3)

                # Create title with rank, fitness, and key metrics
                rank_str = f"Rank #{rank}" if rank > 1 else "Best"
                delta_e_final = delta_e_mev[-1] if len(delta_e_mev) > 0 else 0
                percent_final = percent_delta_e[-1] if len(percent_delta_e) > 0 else 0
                title = f"{rank_str} Trajectory (fitness={fitness:.6e})\n"
                title += f"ΔE={delta_e_final:.6f} MeV, ΔE/E={percent_final:.6f}%"
                plt.suptitle(title, fontsize=12, fontweight="bold")
                plt.tight_layout()

                # Save with rank in filename
                if rank == 1:
                    traj_plot = output_dir / "trajectory_rank1_best.png"
                    traj_data = output_dir / "trajectory_rank1_best.npz"
                else:
                    traj_plot = output_dir / f"trajectory_rank{rank}.png"
                    traj_data = output_dir / f"trajectory_rank{rank}.npz"

                plt.savefig(traj_plot, dpi=150, bbox_inches="tight")
                plt.close(fig)

                self._log_result(
                    f"  Rank #{rank} trajectory plot saved to: {traj_plot}"
                )

                # Also save trajectory data as numpy archive
                np.savez(traj_data, **traj)
                self._log_result(
                    f"  Rank #{rank} trajectory data saved to: {traj_data}"
                )

            else:
                self._log_result(
                    f"[WARNING] Could not generate rank #{rank} trajectory (integration failed)"
                )
                return None

            # Return trajectory data for comparison plot
            return result_data.get("trajectory")

        except Exception as e:
            import traceback

            self._log_result(f"[WARNING] Failed to save trajectory: {e}")
            self._log_result(f"[WARNING] Traceback: {traceback.format_exc()}")
            return None

    def _generate_trajectory_comparison_plot(self, trajectory_data_list):
        """Generate comparison plot for top trajectories via helper."""
        return generate_trajectory_comparison_plot(self, trajectory_data_list)

    def _save_partial_optimization_results(
        self, all_evaluations, param_names, status="PARTIAL"
    ):
        """Save partial optimization results when cancelled or failed.

        Parameters
        ----------
        all_evaluations : list
            List of completed evaluations
        param_names : list
            Parameter names
        status : str
            Status string ("CANCELLED", "FAILED", "PARTIAL")
        """
        import json
        from datetime import datetime
        from pathlib import Path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        if self.config.mode == "optimization":
            method = self.config.optimization_method
            output_dir = (
                Path(self.config.output_dir)
                / "optimizations"
                / f"{timestamp}_{method}_{status}"
            )
        else:
            output_dir = Path(self.config.output_dir) / f"{timestamp}_{status}"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save evaluations to CSV
        csv_path = output_dir / "all_evaluations.csv"
        successful_evals = [
            e
            for e in all_evaluations
            if not e.get("failed", False) and not e.get("halted_early", False)
        ]
        halted_evals = [e for e in all_evaluations if e.get("halted_early", False)]

        if len(all_evaluations) > 0:
            import csv

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["evaluation"]
                    + param_names
                    + ["objective_value", "failed", "halted_early", "halt_reason"],
                )
                writer.writeheader()
                for e in all_evaluations:
                    row = {
                        "evaluation": e["evaluation"],
                        "failed": e.get("failed", False),
                        "halted_early": e.get("halted_early", False),
                        "halt_reason": e.get("halt_reason", ""),
                    }
                    row.update(e["parameters"])
                    row["objective_value"] = e.get("objective_value", float("nan"))
                    writer.writerow(row)
            self._log_result(f"[OK] Partial results saved to: {csv_path}")

        # Save JSON summary
        summary = {
            "status": status,
            "timestamp": timestamp,
            "total_evaluations": len(all_evaluations),
            "successful_evaluations": len(successful_evals),
            "halted_evaluations": len(halted_evals),
            "failed_evaluations": len(all_evaluations)
            - len(successful_evals)
            - len(halted_evals),
            "parameters": param_names,
            "objective": self.config.objective,
        }

        if len(successful_evals) > 0:
            # Find best (filter out inf values)
            maximize = "max" in self.config.objective.lower()
            finite_evals = [
                e
                for e in successful_evals
                if np.isfinite(e.get("objective_value", np.inf))
            ]

            if len(finite_evals) > 0:
                if maximize:
                    best = max(
                        finite_evals,
                        key=lambda x: x.get("objective_value", -float("inf")),
                    )
                else:
                    best = min(
                        finite_evals,
                        key=lambda x: x.get("objective_value", float("inf")),
                    )
                summary["best_parameters"] = best["parameters"]
                summary["best_value"] = best["objective_value"]
            else:
                summary["note"] = "No finite objective values found"

        summary_path = output_dir / "partial_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self._log_result(f"[OK] Summary saved to: {summary_path}")

        # Move log file to results directory
        if self._log_file_path is not None and self._log_file_path.exists():
            import shutil

            dest_log = output_dir / self._log_file_path.name
            shutil.copy2(self._log_file_path, dest_log)
            self._log_result(f"[OK] Log file saved to: {dest_log}")

    def _run_sweep_background(self, is_finetune: bool = False, finetune_regions=None):
        """Delegate sweep background execution to mixin."""
        return super()._run_sweep_background(is_finetune=is_finetune, finetune_regions=finetune_regions)

    def _generate_parameter_grids(self):
        """Delegate parameter grid generation to mixin."""
        return super()._generate_parameter_grids()

    def _generate_range(self, min_val: float, max_val: float, points: int, log_scale: bool) -> List[float]:
        """Delegate range generation to mixin."""
        return super()._generate_range(min_val, max_val, points, log_scale)

    def _run_single_integration(self, aperture: float, energy_gev: float, start_z: float, transv_offset: float, timestep: float, steps: int, rider_m_particle: float = None, rider_charge_sign: float = None, rider_pcount: int = None, rider_transv_mom: float = None, rider_transv_dist: float = None, macroparticle_charge_multiplier: float = None, macroparticle_sigma_multiplier: float = None, driver_params: Dict[str, Any] = None, wall_z: float = None, run_num: int = 0, cancel_flag: Optional[List[bool]] = None) -> Dict[str, Any]:
        """Delegate single integration execution to mixin."""
        return super()._run_single_integration(
            aperture=aperture,
            energy_gev=energy_gev,
            start_z=start_z,
            transv_offset=transv_offset,
            timestep=timestep,
            steps=steps,
            rider_m_particle=rider_m_particle,
            rider_charge_sign=rider_charge_sign,
            rider_pcount=rider_pcount,
            rider_transv_mom=rider_transv_mom,
            rider_transv_dist=rider_transv_dist,
            macroparticle_charge_multiplier=macroparticle_charge_multiplier,
            macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
            driver_params=driver_params,
            wall_z=wall_z,
            run_num=run_num,
            cancel_flag=cancel_flag,
        )

    def _cleanup_orphaned_temp_dirs(self):
        """Delegate orphaned temp directory cleanup to mixin."""
        return super()._cleanup_orphaned_temp_dirs()

    def _save_sweep_results(self, results: List[Dict[str, Any]], failed_runs: List[Dict[str, Any]] = None) -> None:
        """Delegate sweep result persistence to mixin."""
        return super()._save_sweep_results(results, failed_runs)

    def _generate_summary_plots(self, results: List[Dict[str, Any]], output_dir: Path) -> None:
        """Delegate summary plot generation to mixin."""
        return super()._generate_summary_plots(results, output_dir)

    def _plot_single_trajectory(self, result: Dict[str, Any], output_file: Path) -> None:
        """Delegate single trajectory plotting to mixin."""
        return super()._plot_single_trajectory(result, output_file)

    def _update_progress(self, value: float, text: str):
        """Update progress bar and label (thread-safe)."""

        def update():
            self.progress_bar["value"] = value
            self.progress_label["text"] = text

        self.after(0, update)

    def _update_progress_text(self, text: str):
        """Update only the progress label text (thread-safe)."""
        self.after(0, lambda: self.progress_label.config(text=text))

    def _export_evaluations_csv(self, all_evaluations, param_names, output_dir):
        """Export all evaluations to CSV file.

        Parameters
        ----------
        all_evaluations : list
            List of evaluation records
        param_names : list
            List of parameter names
        output_dir : Path
            Output directory
        """
        import csv
        from pathlib import Path

        try:
            output_path = Path(output_dir)
            csv_file = output_path / "all_evaluations.csv"

            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                # Determine all possible metric names from evaluations
                metric_names = set()
                for eval_rec in all_evaluations:
                    if not eval_rec.get("failed", True) and "metrics" in eval_rec:
                        metric_names.update(eval_rec["metrics"].keys())

                metric_names = sorted(metric_names)

                # Create header
                header = (
                    ["evaluation", "failed", "halted_early", "halt_reason"]
                    + param_names
                    + metric_names
                    + ["objective_value", "fitness"]
                )
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()

                # Write rows
                for eval_rec in all_evaluations:
                    row = {
                        "evaluation": eval_rec["evaluation"],
                        "failed": eval_rec.get("failed", True),
                        "halted_early": eval_rec.get("halted_early", False),
                        "halt_reason": eval_rec.get("halt_reason", ""),
                        "objective_value": eval_rec.get("objective_value", ""),
                        "fitness": eval_rec.get("fitness", ""),
                    }

                    # Add parameters
                    for param_name in param_names:
                        row[param_name] = eval_rec.get("parameters", {}).get(
                            param_name, ""
                        )

                    # Add metrics
                    if not eval_rec.get("failed", True) and "metrics" in eval_rec:
                        for metric_name in metric_names:
                            row[metric_name] = eval_rec["metrics"].get(metric_name, "")

                    writer.writerow(row)

            self._log_result(f"Evaluation CSV exported to: {csv_file}")

        except Exception as e:
            self._log_result(f"[WARNING] Failed to export evaluations CSV: {e}")

    def _view_npz_trajectories(self, results_dir):
        """View NPZ trajectory files from an optimization run.

        Parameters
        ----------
        results_dir : str or Path
            Directory containing NPZ trajectory files
        """
        import glob
        import os
        from pathlib import Path

        try:
            results_path = Path(results_dir)

            # Find all NPZ trajectory files
            npz_pattern = str(results_path / "trajectory_rank*.npz")
            npz_files = sorted(glob.glob(npz_pattern))

            if not npz_files:
                # Try alternative pattern for evaluation trajectories
                npz_pattern = str(results_path / "evaluation_*_trajectory.npz")
                npz_files = sorted(glob.glob(npz_pattern))

            if not npz_files:
                _show_info_dialog(
                    self,
                    "No Trajectories Found",
                    f"No NPZ trajectory files found in:\n{results_dir}\n\n"
                    "Expected files like:\n"
                    "- trajectory_rank1_best.npz\n"
                    "- trajectory_rank2.npz\n"
                    "- evaluation_0001_trajectory.npz",
                )
                return

            # Create dialog to select and plot trajectories
            dialog = tk.Toplevel(self)
            dialog.title(f"NPZ Trajectories: {results_path.name}")
            dialog.geometry("600x500")
            dialog.transient(self)

            # Info label
            ttk.Label(
                dialog,
                text=f"Found {len(npz_files)} trajectory files",
                font=("TkDefaultFont", 10, "bold"),
            ).pack(pady=(10, 5))

            # Listbox with scrollbar
            list_frame = ttk.Frame(dialog)
            list_frame.pack(fill="both", expand=True, padx=10, pady=5)

            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side="right", fill="y")

            listbox = tk.Listbox(
                list_frame,
                selectmode="extended",
                yscrollcommand=scrollbar.set,
                height=15,
            )
            listbox.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=listbox.yview)

            # Populate listbox
            for npz_file in npz_files:
                filename = os.path.basename(npz_file)
                listbox.insert("end", filename)

            # Select first item by default
            if npz_files:
                listbox.selection_set(0)

            # Buttons
            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(pady=10)

            def plot_selected():
                selection = listbox.curselection()
                if not selection:
                    _show_info_dialog(
                        dialog,
                        "No Selection",
                        "Please select one or more trajectories to plot.",
                    )
                    return

                selected_files = [npz_files[i] for i in selection]
                self._plot_npz_trajectories(selected_files, results_path)

            ttk.Button(
                btn_frame,
                text="Plot Selected",
                command=plot_selected,
                style="Accent.TButton",
            ).pack(side="left", padx=5)

            ttk.Button(
                btn_frame,
                text="Close",
                command=dialog.destroy,
            ).pack(side="left", padx=5)

        except Exception as e:
            import traceback

            _show_error_dialog(
                self,
                "Error Viewing NPZ Trajectories",
                f"Failed to view NPZ trajectories:\n{e}\n\n{traceback.format_exc()}",
            )

    def _plot_npz_trajectories(self, npz_files, results_dir):
        """Plot NPZ trajectory files.

        Parameters
        ----------
        npz_files : list
            List of NPZ file paths
        results_dir : Path
            Results directory
        """
        try:
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )

            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

            ax_r = fig.add_subplot(gs[0, 0])
            ax_pz = fig.add_subplot(gs[0, 1])
            ax_pr = fig.add_subplot(gs[1, 0])
            ax_gamma = fig.add_subplot(gs[1, 1])
            ax_energy = fig.add_subplot(gs[2, :])

            # Color cycle
            colors = plt.cm.tab10(np.linspace(0, 1, len(npz_files)))

            for idx, npz_file in enumerate(npz_files):
                # Load NPZ
                data = np.load(npz_file)
                z = data["z"]
                r = data["r"]
                pz = data["pz"]
                pr = data["pr"]
                gamma = data["gamma"]

                # Get label from filename
                label = Path(npz_file).stem.replace("trajectory_", "").replace("_", " ")

                # Plot
                ax_r.plot(z, r * 1e3, label=label, color=colors[idx], alpha=0.7)
                ax_pz.plot(z, pz, color=colors[idx], alpha=0.7)
                ax_pr.plot(z, pr, color=colors[idx], alpha=0.7)
                ax_gamma.plot(z, gamma, color=colors[idx], alpha=0.7)

                # Energy in MeV (for electrons: E = (γ - 1) * 0.511 MeV)
                energy_mev = (gamma - 1) * 0.511
                ax_energy.plot(z, energy_mev, color=colors[idx], alpha=0.7, label=label)

            # Formatting
            ax_r.set_xlabel("z (mm)")
            ax_r.set_ylabel("r (μm)")
            ax_r.set_title("Transverse Position")
            ax_r.grid(True, alpha=0.3)
            ax_r.legend()

            ax_pz.set_xlabel("z (mm)")
            ax_pz.set_ylabel("Pz")
            ax_pz.set_title("Longitudinal Momentum")
            ax_pz.grid(True, alpha=0.3)

            ax_pr.set_xlabel("z (mm)")
            ax_pr.set_ylabel("Pr")
            ax_pr.set_title("Transverse Momentum")
            ax_pr.grid(True, alpha=0.3)

            ax_gamma.set_xlabel("z (mm)")
            ax_gamma.set_ylabel("γ")
            ax_gamma.set_title("Lorentz Factor")
            ax_gamma.grid(True, alpha=0.3)

            ax_energy.set_xlabel("z (mm)")
            ax_energy.set_ylabel("Energy (MeV)")
            ax_energy.set_title("Particle Energy")
            ax_energy.grid(True, alpha=0.3)
            ax_energy.legend()

            fig.suptitle(
                f"Optimization Trajectories: {results_dir.name}",
                fontsize=14,
                fontweight="bold",
            )

            plt.tight_layout()

            # Embed in Tkinter window instead of plt.show()
            plot_window = tk.Toplevel(self)
            plot_window.title(f"NPZ Trajectories: {results_dir.name}")
            plot_window.geometry("1200x900")

            # Create main container frame
            main_frame = ttk.Frame(plot_window)
            main_frame.pack(fill="both", expand=True)

            # Create canvas for the figure
            canvas = FigureCanvasTkAgg(fig, master=main_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

            # Add matplotlib navigation toolbar
            toolbar_frame = ttk.Frame(main_frame)
            toolbar_frame.pack(side="top", fill="x")
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()

            # Close button
            button_frame = ttk.Frame(main_frame, padding=5)
            button_frame.pack(side="top", fill="x")
            ttk.Button(
                button_frame,
                text="Close",
                command=plot_window.destroy,
            ).pack(side="right", padx=5)

        except Exception as e:
            import traceback

            _show_error_dialog(
                self,
                "Plotting Error",
                f"Failed to plot NPZ trajectories:\n{e}\n\n{traceback.format_exc()}",
            )

    def _save_evaluation_trajectory(self, eval_num, trajectory_data, output_dir):
        """Save a single evaluation trajectory to NPZ file.

        Parameters
        ----------
        eval_num : int
            Evaluation number
        trajectory_data : dict
            Dictionary containing trajectory arrays (z, r, pz, pr, t, gamma)
        output_dir : Path
            Directory to save the trajectory file

        Returns
        -------
        str or None
            Path to saved file, or None if save failed
        """
        try:
            from pathlib import Path

            import numpy as np

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            trajectory_file = output_path / f"evaluation_{eval_num:04d}_trajectory.npz"

            # Convert lists to numpy arrays and save
            np.savez(
                trajectory_file,
                z=np.array(trajectory_data["z"]),
                r=np.array(trajectory_data["r"]),
                pz=np.array(trajectory_data["pz"]),
                pr=np.array(trajectory_data["pr"]),
                t=np.array(trajectory_data["t"]),
                gamma=np.array(trajectory_data["gamma"]),
            )

            return str(trajectory_file)
        except Exception as e:
            self._log_result(
                f"  [WARNING] Failed to save evaluation {eval_num} trajectory: {e}"
            )
            return None

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
