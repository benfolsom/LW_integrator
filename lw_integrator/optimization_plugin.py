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
from typing import Any, Optional

from core.types import SimulationType  # type: ignore[import]
from optimization.config import OptimizationConfig
from optimization.plugin_config_helpers import (
    apply_sweep_parameter_overrides,
    parse_float_list,
    parse_offset_pair,
)
from optimization.plugin_persistence_helpers import (
    apply_persisted_config_overrides,
    build_saved_config_payload,
    metrics_export_settings_from_data,
    resolve_loaded_sweep_state,
)
from optimization.plugin_results_helpers import (
    build_summary_heatmap_grid,
    build_trajectory_plot_data,
    collect_summary_plot_data,
    parse_results_payload,
    summarize_result_row,
    UNKNOWN_RESULTS_FORMAT_MESSAGE,
)
from optimization.plugin_ui_mixins import OptimizationPluginUIMixin
from optimization.results_mixins import OptimizationResultsMixin
from optimization.run_mixins import OptimizationRunMixin
from optimization.sweep_helpers import (
    AMU_TO_MEV,
    calculate_energy_from_pz,
    calculate_starting_pz_from_energy,
)
from optimization.ui_helpers import (
    show_error_dialog as _show_error_dialog,
)
from optimization.ui_helpers import (
    show_info_dialog as _show_info_dialog,
)
from optimization.ui_helpers import (
    show_warning_dialog as _show_warning_dialog,
)


class OptimizationPlugin(
    OptimizationPluginUIMixin, OptimizationRunMixin, OptimizationResultsMixin, ttk.Frame
):
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

        # Sync simulation type with main GUI if available
        if self.gui_controller and hasattr(self.gui_controller, "sim_type_var"):
            main_sim_type = self.gui_controller.sim_type_var.get()
            self.sim_type_var.set(main_sim_type)
            # Update driver visibility based on synced simulation type
            self._update_driver_visibility()


    def _build_parameter_section(self):
        """Build parameter range specification section."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Parameter Ranges", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)
        self.parameter_frame = frame

        # Store parameter widgets for mode-based visibility control
        self._param_widgets = {}

        # Add explanatory help text
        help_frame = ttk.Frame(frame)
        help_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        help_text = (
            "Coordinate system: Particles start at z-coordinate and travel toward the conducting wall.\n"
            "Example: Particle at z=0 travels to wall at z=2200 mm (distance = 2200 mm).\n"
            "Transverse offset: BUNCH_TO_BUNCH mode uses absolute distance (mm), CONDUCTING_WALL uses fraction of aperture."
        )
        help_label = ttk.Label(
            help_frame, text=help_text, foreground="gray40", font=("TkDefaultFont", 8)
        )
        help_label.pack(anchor="w")

        # Aperture range
        self._param_widgets["aperture_label"] = ttk.Label(
            frame, text="Aperture Radius:"
        )
        self._param_widgets["aperture_label"].grid(row=1, column=0, sticky="w", pady=2)
        self._param_widgets["aperture_frame"] = ttk.Frame(frame)
        aperture_frame = self._param_widgets["aperture_frame"]
        aperture_frame.grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)

        ttk.Label(aperture_frame, text="Min (mm):").pack(side="left", padx=(0, 2))
        self.aperture_min_var = tk.StringVar(value="1e-5")
        self._param_widgets["aperture_min_entry"] = ttk.Entry(
            aperture_frame, textvariable=self.aperture_min_var, width=10
        )
        self._param_widgets["aperture_min_entry"].pack(side="left", padx=2)

        ttk.Label(aperture_frame, text="Max (mm):").pack(side="left", padx=(10, 2))
        self.aperture_max_var = tk.StringVar(value="1e-3")
        self._param_widgets["aperture_max_entry"] = ttk.Entry(
            aperture_frame, textvariable=self.aperture_max_var, width=10
        )
        self._param_widgets["aperture_max_entry"].pack(side="left", padx=2)

        ttk.Label(aperture_frame, text="Points:").pack(side="left", padx=(10, 2))
        self.aperture_points_var = tk.StringVar(value="10")
        self._param_widgets["aperture_points_entry"] = ttk.Entry(
            aperture_frame, textvariable=self.aperture_points_var, width=5
        )
        self._param_widgets["aperture_points_entry"].pack(side="left", padx=2)

        self.aperture_log_var = tk.BooleanVar(value=True)
        self._param_widgets["aperture_log_check"] = ttk.Checkbutton(
            aperture_frame, text="Log scale", variable=self.aperture_log_var
        )
        self._param_widgets["aperture_log_check"].pack(side="left", padx=(10, 0))

        # Energy range
        self.energy_label = ttk.Label(frame, text="Particle Energy:")
        self.energy_label.grid(row=2, column=0, sticky="w", pady=2)
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

        # Helper text showing starting Pz for rider
        self.rider_pz_helper_var = tk.StringVar(value="")
        self.rider_pz_helper_label = ttk.Label(
            frame,
            textvariable=self.rider_pz_helper_var,
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        )
        self.rider_pz_helper_label.grid(
            row=3, column=1, columnspan=3, sticky="w", pady=(0, 5)
        )

        # Trace energy and mass changes to update Pz helper
        self.energy_min_var.trace_add(
            "write", lambda *_: self._update_rider_pz_helper()
        )
        self.energy_max_var.trace_add(
            "write", lambda *_: self._update_rider_pz_helper()
        )
        # Note: rider mass trace is added in _build_rider_particle_section after widget creation

        # Rider Transverse offset (x, y)
        self._param_widgets["offset_label"] = ttk.Label(
            frame, text="Rider Transverse Offset (x, y):"
        )
        self._param_widgets["offset_label"].grid(row=4, column=0, sticky="w", pady=2)
        self.offset_fractions_var = tk.StringVar(value="0.0, 0.0")
        self._param_widgets["offset_entry"] = ttk.Entry(
            frame, textvariable=self.offset_fractions_var, width=15
        )
        self._param_widgets["offset_entry"].grid(
            row=4, column=1, sticky="w", pady=2, padx=(0, 5)
        )
        self._param_widgets["offset_desc_label"] = ttk.Label(
            frame,
            text="mm (BUNCH_TO_BUNCH: absolute offset | CONDUCTING_WALL: fraction of aperture)",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        )
        self._param_widgets["offset_desc_label"].grid(
            row=4, column=2, columnspan=2, sticky="w", pady=2
        )

        # Driver Transverse offset (x, y)
        self._param_widgets["driver_offset_label"] = ttk.Label(
            frame, text="Driver Transverse Offset (x, y):"
        )
        self._param_widgets["driver_offset_label"].grid(
            row=5, column=0, sticky="w", pady=2
        )
        self.driver_offset_var = tk.StringVar(value="0.0, 0.0")
        self._param_widgets["driver_offset_entry"] = ttk.Entry(
            frame, textvariable=self.driver_offset_var, width=15
        )
        self._param_widgets["driver_offset_entry"].grid(
            row=5, column=1, sticky="w", pady=2, padx=(0, 5)
        )
        self._param_widgets["driver_offset_desc_label"] = ttk.Label(
            frame,
            text="mm (BUNCH_TO_BUNCH only)",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        )
        self._param_widgets["driver_offset_desc_label"].grid(
            row=5, column=2, columnspan=2, sticky="w", pady=2
        )

        # Starting z position (rider only)
        ttk.Label(frame, text="Rider Starting z (mm):").grid(
            row=6, column=0, sticky="w", pady=2
        )
        self.start_z_var = tk.StringVar(value="0.0")
        ttk.Entry(frame, textvariable=self.start_z_var, width=15).grid(
            row=6, column=1, sticky="w", pady=2
        )
        ttk.Label(
            frame,
            text="Driver starting z uses sweepable param below.",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        ).grid(row=6, column=2, columnspan=2, sticky="w", pady=2, padx=(10, 0))

        # Wall Position (sweepable)
        self._param_widgets["wall_z_label"] = ttk.Label(frame, text="Wall Position:")
        self._param_widgets["wall_z_label"].grid(row=8, column=0, sticky="w", pady=2)

        # Fixed value and sweep checkbox on same row
        self._param_widgets["wall_z_fixed_frame"] = ttk.Frame(frame)
        wall_z_fixed_frame = self._param_widgets["wall_z_fixed_frame"]
        wall_z_fixed_frame.grid(row=8, column=1, columnspan=3, sticky="w", pady=2)

        self.wall_z_var = tk.StringVar(value="2200.0")
        self.wall_z_entry = ttk.Entry(
            wall_z_fixed_frame, textvariable=self.wall_z_var, width=10
        )
        self.wall_z_entry.pack(side="left", padx=(0, 10))

        self.wall_z_sweep_var = tk.BooleanVar(value=False)
        self._param_widgets["wall_z_sweep_check"] = ttk.Checkbutton(
            wall_z_fixed_frame,
            text="Sweep",
            variable=self.wall_z_sweep_var,
            command=self._toggle_wall_z_sweep,
        )
        self._param_widgets["wall_z_sweep_check"].pack(side="left")

        # Sweep controls on new row for better visibility
        self._param_widgets["wall_z_sweep_frame"] = ttk.Frame(frame)
        wall_z_sweep_frame = self._param_widgets["wall_z_sweep_frame"]
        wall_z_sweep_frame.grid(
            row=9, column=1, columnspan=3, sticky="w", pady=2, padx=(20, 0)
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

        # Cavity Spacing (for SWITCHING_WALL) - moved to row 10
        self._param_widgets["cavity_spacing_label"] = ttk.Label(
            frame, text="Cavity Spacing:"
        )
        self._param_widgets["cavity_spacing_label"].grid(
            row=10, column=0, sticky="w", pady=2
        )
        self.cavity_spacing_var = tk.StringVar(value="1e5")
        self.cavity_spacing_entry = ttk.Entry(
            frame, textvariable=self.cavity_spacing_var, width=15
        )
        self.cavity_spacing_entry.grid(
            row=10, column=1, sticky="w", pady=2, padx=(0, 5)
        )
        self._param_widgets["cavity_spacing_desc_label"] = ttk.Label(
            frame,
            text="mm (SWITCHING_WALL only)",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        )
        self._param_widgets["cavity_spacing_desc_label"].grid(
            row=10, column=2, columnspan=2, sticky="w", pady=2
        )

        # Timestep Auto-Calculation (always uses auto_distance strategy)
        timestep_label = ttk.Label(frame, text="Timestep Calculation:")
        timestep_label.grid(row=11, column=0, sticky="w", pady=2)

        # Store label for dynamic tooltip update
        self.timestep_calc_label = timestep_label

        # Initial tooltip - will be updated based on sim type
        self._update_timestep_tooltip()

        timestep_frame = ttk.Frame(frame)
        timestep_frame.grid(row=11, column=1, columnspan=3, sticky="ew", pady=2)

        self.timestep_mode_var = tk.StringVar(value="duration")
        ttk.Radiobutton(
            timestep_frame,
            text="Auto-calc step duration, provide count:",
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
            text="Auto-calc count, provide step duration:",
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
        self._param_widgets["distance_label"] = ttk.Label(
            frame, text="Distance Target:"
        )
        self._param_widgets["distance_label"].grid(row=12, column=0, sticky="w", pady=2)
        self._param_widgets["distance_frame"] = ttk.Frame(frame)
        distance_frame = self._param_widgets["distance_frame"]
        distance_frame.grid(row=12, column=1, columnspan=3, sticky="ew", pady=2)

        # Dynamic label that changes based on sim type
        self.distance_target_prefix_label = ttk.Label(distance_frame, text="Target:")
        self.distance_target_prefix_label.pack(side="left", padx=(0, 2))

        self.auto_steps_distance_var = tk.StringVar(value="10.0")
        self._param_widgets["distance_entry"] = ttk.Entry(
            distance_frame, textvariable=self.auto_steps_distance_var, width=6
        )
        self._param_widgets["distance_entry"].pack(side="left", padx=2)

        # Dynamic suffix label
        self.distance_target_suffix_label = ttk.Label(
            distance_frame, text="mm past wall (min 5% of driver position for B2B)"
        )
        self.distance_target_suffix_label.pack(side="left", padx=2)

        # Note about trajectory and output configuration
        config_note = ttk.Label(
            frame,
            text="ℹ For trajectory saving and output options, see the 'Results & Output Configuration' section below",
            font=("TkDefaultFont", 8, "italic"),
            foreground="blue",
            justify="left",
        )
        config_note.grid(row=13, column=0, columnspan=4, sticky="w", pady=(10, 10))

        frame.columnconfigure(2, weight=1)

        # Initialize timestep mode state
        self._toggle_timestep_mode()

        # Initialize parameter visibility based on simulation type
        self._update_parameter_visibility()

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

        # Add trace for rider mass to update Pz helper
        self.sweep_params["rider_m_particle"]["fixed_var"].trace_add(
            "write", lambda *_: self._update_rider_pz_helper()
        )

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

        # Stripped Ions (sweepable)
        self._add_sweepable_param(
            frame, row, "rider_stripped_ions", "Stripped Ions:", "1.0", width=10
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

        # Transverse Spread
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_transv_dist",
            "Transverse Spread (mm):",
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

        # Driver Energy (replaces Starting Pz)
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_energy_gev",
            "Kinetic Energy (GeV):",
            "112.5",
            width=15,
        )
        row += 1

        # Link to Rider Energy checkbox - when checked, driver energy follows rider energy
        link_frame = ttk.Frame(self.driver_frame)
        link_frame.grid(
            row=row, column=0, columnspan=6, sticky="w", pady=(0, 2), padx=(20, 0)
        )

        self.link_driver_rider_energy_var = tk.BooleanVar(value=False)
        self.link_energy_checkbox = ttk.Checkbutton(
            link_frame,
            text="Link to Rider Energy (sweep both at same value)",
            variable=self.link_driver_rider_energy_var,
            command=self._on_link_energy_toggled,
        )
        self.link_energy_checkbox.pack(side="left")

        # Help text for linked energy mode
        self.link_energy_help_label = ttk.Label(
            link_frame,
            text="",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        )
        self.link_energy_help_label.pack(side="left", padx=(10, 0))

        row += 1

        # Driver momentum direction selector (−ẑ or +ẑ)
        dir_frame = ttk.Frame(self.driver_frame)
        dir_frame.grid(
            row=row, column=0, columnspan=6, sticky="w", pady=(0, 2), padx=(20, 0)
        )

        ttk.Label(dir_frame, text="Momentum direction:").pack(side="left", padx=(0, 6))

        self.driver_direction_var = tk.StringVar(value="-z")

        rb_minus = ttk.Radiobutton(
            dir_frame,
            text="\u2212\u1e91  (toward rider)",
            variable=self.driver_direction_var,
            value="-z",
            command=self._update_driver_pz_helper,
        )
        rb_minus.pack(side="left", padx=(0, 12))

        rb_plus = ttk.Radiobutton(
            dir_frame,
            text="+\u1e91  (away from rider)",
            variable=self.driver_direction_var,
            value="+z",
            command=self._update_driver_pz_helper,
        )
        rb_plus.pack(side="left")

        row += 1

        # Helper text showing calculated starting Pz for driver
        self.driver_pz_helper_var = tk.StringVar(value="")
        self.driver_pz_helper_label = ttk.Label(
            self.driver_frame,
            textvariable=self.driver_pz_helper_var,
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        )
        self.driver_pz_helper_label.grid(
            row=row, column=0, columnspan=6, sticky="w", pady=(0, 5), padx=(20, 0)
        )
        row += 1

        # Trace driver energy and mass changes to update Pz helper
        self.sweep_params["driver_energy_gev"]["fixed_var"].trace_add(
            "write", lambda *_: self._update_driver_pz_helper()
        )
        self.sweep_params["driver_energy_gev"]["min_var"].trace_add(
            "write", lambda *_: self._update_driver_pz_helper()
        )
        self.sweep_params["driver_energy_gev"]["max_var"].trace_add(
            "write", lambda *_: self._update_driver_pz_helper()
        )
        self.sweep_params["driver_m_particle"]["fixed_var"].trace_add(
            "write", lambda *_: self._update_driver_pz_helper()
        )

        # Store references to driver energy widgets for enable/disable
        self._driver_energy_widgets = [
            self.sweep_params["driver_energy_gev"]["fixed_entry"],
            self.sweep_params["driver_energy_gev"]["range_frame"],
        ]

        # Stripped Ions (sweepable)
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_stripped_ions",
            "Stripped Ions:",
            "54.0",
            width=10,
        )
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

    def _on_link_energy_toggled(self):
        """Handle toggling of the 'Link to Rider Energy' checkbox."""
        linked = self.link_driver_rider_energy_var.get()

        # Get driver energy controls
        driver_energy_controls = self.sweep_params["driver_energy_gev"]

        if linked:
            # Disable driver energy input fields (rider energy will be used for both)
            driver_energy_controls["fixed_entry"].config(state="disabled")
            driver_energy_controls["sweep_var"].set(False)
            self._toggle_sweep_controls("driver_energy_gev")

            # Disable the sweep checkbox itself
            # Find the sweep checkbox widget in the driver frame
            for widget in self.driver_frame.winfo_children():
                if isinstance(widget, ttk.Checkbutton):
                    # Check if this is the driver_energy_gev sweep checkbox
                    try:
                        if widget.cget("variable") == str(
                            driver_energy_controls["sweep_var"]
                        ):
                            widget.config(state="disabled")
                    except Exception:
                        pass

            # Update help text
            self.link_energy_help_label.config(
                text="(Driver energy = Rider energy for each sweep point)"
            )
            self._update_driver_pz_helper()
        else:
            # Re-enable driver energy input fields
            driver_energy_controls["fixed_entry"].config(state="normal")

            # Re-enable the sweep checkbox
            for widget in self.driver_frame.winfo_children():
                if isinstance(widget, ttk.Checkbutton):
                    try:
                        if widget.cget("variable") == str(
                            driver_energy_controls["sweep_var"]
                        ):
                            widget.config(state="normal")
                    except Exception:
                        pass

            # Clear help text
            self.link_energy_help_label.config(text="")
            self._update_driver_pz_helper()

    def _update_driver_energy_link_state(self):
        """Update driver energy controls based on link state (called during load)."""
        if hasattr(self, "link_driver_rider_energy_var"):
            self._on_link_energy_toggled()

    def _update_rider_pz_helper(self):
        """Update helper text showing rider starting Pz calculated from energy."""
        try:
            # Get rider mass
            mass_str = self.sweep_params["rider_m_particle"]["fixed_var"].get()
            mass_amu = float(mass_str) if mass_str else 0.00054857990907

            # Get energy range
            energy_min = float(self.energy_min_var.get())
            energy_max = float(self.energy_max_var.get())

            # Calculate Pz values
            pz_min = calculate_starting_pz_from_energy(energy_min, mass_amu)
            pz_max = calculate_starting_pz_from_energy(energy_max, mass_amu)

            # Update helper text
            self.rider_pz_helper_var.set(
                f"→ Starting Pz range: [{pz_min:.2f}, {pz_max:.2f}] amu·mm/ns"
            )
        except (ValueError, ZeroDivisionError):
            self.rider_pz_helper_var.set("")

    def _update_driver_pz_helper(self):
        """Update helper text showing driver starting Pz calculated from energy."""
        try:
            # Get driver mass
            mass_str = self.sweep_params["driver_m_particle"]["fixed_var"].get()
            mass_amu = float(mass_str) if mass_str else 207.2

            # Determine sign from direction selector
            negative = (
                getattr(self, "driver_direction_var", None) is None
                or getattr(self, "driver_direction_var").get() == "-z"
            )
            sign_label = "\u2212\u1e91" if negative else "+\u1e91"

            # Check if linked energy mode is active
            if (
                getattr(self, "link_driver_rider_energy_var", None)
                and self.link_driver_rider_energy_var.get()
            ):
                # In linked mode, show that driver follows rider energy
                energy_min = float(self.energy_min_var.get())
                energy_max = float(self.energy_max_var.get())
                pz_min = calculate_starting_pz_from_energy(
                    energy_min, mass_amu, negative=negative
                )
                pz_max = calculate_starting_pz_from_energy(
                    energy_max, mass_amu, negative=negative
                )
                self.driver_pz_helper_var.set(
                    f"\u2192 [LINKED] Pz range: [{pz_min:.2f}, {pz_max:.2f}] amu\u00b7mm/ns ({sign_label})"
                )
            # Check if energy is being swept
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
                    f"\u2192 Pz range: [{pz_min:.2f}, {pz_max:.2f}] amu\u00b7mm/ns ({sign_label})"
                )
            else:
                energy_gev = abs(
                    float(self.sweep_params["driver_energy_gev"]["fixed_var"].get())
                )
                pz = calculate_starting_pz_from_energy(
                    energy_gev, mass_amu, negative=negative
                )
                self.driver_pz_helper_var.set(
                    f"\u2192 Pz = {pz:.2f} amu\u00b7mm/ns ({sign_label})"
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
            self._update_driver_pz_helper()
        else:
            self.driver_frame.pack_forget()

        # Update energy label
        self._update_energy_label()

        # Update rider Pz helper
        self._update_rider_pz_helper()

    def _on_sim_type_changed(self):
        """Handle simulation type change."""
        self._update_driver_visibility()
        self._update_macroparticle_state()
        self._update_parameter_visibility()

        # Sync simulation type to main GUI
        if self.gui_controller and hasattr(self.gui_controller, "sim_type_var"):
            sim_type_value = self.sim_type_var.get()
            self.gui_controller.sim_type_var.set(sim_type_value)
            # Force update of main GUI's simulation type combobox display
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

    def _set_frame_state(self, frame, state):
        """Recursively set state for all widgets in a frame.

        Args:
            frame: The frame widget containing children to update
            state: "normal" or "disabled"
        """
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
                    # Recursively process nested frames
                    self._set_frame_state(child, state)
            except Exception:
                # Some widgets might not support these configurations
                pass

    def _update_parameter_visibility(self):
        """Update parameter field states based on simulation type."""
        if not hasattr(self, "cavity_spacing_entry"):
            return

        sim_type = self.sim_type_var.get()
        is_bunch_to_bunch = sim_type == "BUNCH_TO_BUNCH"

        # Grey out cavity_spacing unless SWITCHING_WALL mode
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

        # Grey out aperture radius and wall position in BUNCH_TO_BUNCH mode
        if is_bunch_to_bunch:
            # Disable aperture widgets
            self._set_frame_state(self._param_widgets.get("aperture_frame"), "disabled")
            if "aperture_label" in self._param_widgets:
                self._param_widgets["aperture_label"].config(foreground="gray")

            # Disable wall_z widgets
            self._set_frame_state(
                self._param_widgets.get("wall_z_fixed_frame"), "disabled"
            )
            self._set_frame_state(
                self._param_widgets.get("wall_z_sweep_frame"), "disabled"
            )
            if "wall_z_label" in self._param_widgets:
                self._param_widgets["wall_z_label"].config(foreground="gray")
        else:
            # Enable aperture widgets
            self._set_frame_state(self._param_widgets.get("aperture_frame"), "normal")
            if "aperture_label" in self._param_widgets:
                self._param_widgets["aperture_label"].config(foreground="black")

            # Enable wall_z widgets (but sweep controls depend on sweep checkbox)
            self._set_frame_state(
                self._param_widgets.get("wall_z_fixed_frame"), "normal"
            )
            if "wall_z_label" in self._param_widgets:
                self._param_widgets["wall_z_label"].config(foreground="black")
            # Re-apply sweep control state
            self._toggle_wall_z_sweep()

        # Update dynamic help text based on simulation type
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
        else:  # CONDUCTING_WALL or SWITCHING_WALL
            tooltip_text = (
                "CONDUCTING_WALL / SWITCHING_WALL Mode:\n"
                "• Particle travels to: wall_z + distance_target\n"
                "• Ensures consistent trajectory length across energies\n"
                "• Step duration auto-calculated to reach target in specified steps\n"
                "• Or step count auto-calculated for specified duration"
            )

        # Remove old tooltip and add new one
        self._add_tooltip(self.timestep_calc_label, tooltip_text)

    def _update_distance_target_labels(self):
        """Update distance target label text based on simulation type."""
        if not hasattr(self, "distance_target_prefix_label"):
            return

        sim_type = self.sim_type_var.get()

        if sim_type == "BUNCH_TO_BUNCH":
            self.distance_target_prefix_label.config(text="Extra distance:")
            self.distance_target_suffix_label.config(text="mm past driver_start")
        else:  # CONDUCTING_WALL or SWITCHING_WALL
            self.distance_target_prefix_label.config(text="Target: wall +")
            self.distance_target_suffix_label.config(
                text="mm (min 5% of steps enforced)"
            )


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
            # max_refinement_attempts is now auto-calculated (read-only display in GUI)
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
                    f"{config.adaptive_timestep_probe_threshold:.6g}"
                )
            if hasattr(self.gui_controller, "adaptive_timestep_max_probe_steps_var"):
                self.gui_controller.adaptive_timestep_max_probe_steps_var.set(
                    str(config.adaptive_timestep_max_probe_steps)
                )
            if hasattr(self.gui_controller, "adaptive_timestep_debug_var"):
                self.gui_controller.adaptive_timestep_debug_var.set(
                    config.adaptive_timestep_debug
                )

            # Gamma reconciliation settings
            if hasattr(
                self.gui_controller, "self_consistency_gamma_reconciliation_method_var"
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_method_var.set(
                    config.self_consistency_gamma_reconciliation_method
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_low_beta_threshold_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_low_beta_threshold_var.set(
                    f"{config.self_consistency_gamma_reconciliation_low_beta_threshold:.2f}"
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_high_beta_threshold_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_high_beta_threshold_var.set(
                    f"{config.self_consistency_gamma_reconciliation_high_beta_threshold:.2f}"
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_low_beta_weight_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_low_beta_weight_var.set(
                    f"{config.self_consistency_gamma_reconciliation_low_beta_weight:.2f}"
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_high_beta_weight_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_high_beta_weight_var.set(
                    f"{config.self_consistency_gamma_reconciliation_high_beta_weight:.2f}"
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_mid_beta_weight_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_mid_beta_weight_var.set(
                    f"{config.self_consistency_gamma_reconciliation_mid_beta_weight:.2f}"
                )
            if hasattr(
                self.gui_controller,
                "self_consistency_gamma_reconciliation_fixed_weight_var",
            ):
                self.gui_controller.self_consistency_gamma_reconciliation_fixed_weight_var.set(
                    f"{config.self_consistency_gamma_reconciliation_fixed_weight:.2f}"
                )

            # Toggle controls to match loaded state
            if hasattr(self.gui_controller, "_toggle_self_consistency_controls"):
                self.gui_controller._toggle_self_consistency_controls()
            if hasattr(self.gui_controller, "_toggle_adaptive_timestep_controls"):
                self.gui_controller._toggle_adaptive_timestep_controls()
            if hasattr(self.gui_controller, "_toggle_gamma_reconciliation_params"):
                self.gui_controller._toggle_gamma_reconciliation_params()

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

    def _validate_inputs(self) -> Optional[str]:
        """Validate user inputs. Returns error message or None."""
        try:
            sim_type = self.sim_type_var.get()
            is_bunch_to_bunch = sim_type == "BUNCH_TO_BUNCH"

            # Aperture range - only validate for CONDUCTING_WALL modes
            if not is_bunch_to_bunch:
                aperture_min = float(self.aperture_min_var.get())
                aperture_max = float(self.aperture_max_var.get())
                if aperture_min >= aperture_max:
                    return "Aperture min must be less than max"
                if aperture_min <= 0:
                    return "Aperture min must be positive"

            # Energy range (rider kinetic energy)
            energy_min = float(self.energy_min_var.get())
            energy_max = float(self.energy_max_var.get())
            energy_points = int(self.energy_points_var.get())

            # For BUNCH_TO_BUNCH the rider energy can be fixed (1 point,
            # min==max) when the sweep is purely over driver parameters.
            if is_bunch_to_bunch and energy_points == 1:
                # Single-point rider energy: just needs to be positive
                if energy_min <= 0:
                    return "Rider energy must be positive"
            else:
                if energy_min >= energy_max:
                    return "Energy min must be less than max"
                if energy_min <= 0:
                    return "Energy min must be positive"

            mode = self.mode_var.get()

            if mode == "blind_sweep":
                # Sweep mode requires at least 2 points for main parameters
                # UNLESS there is at least one swept driver/rider sub-parameter
                has_swept_sub_param = any(
                    controls["sweep_var"].get()
                    for controls in self.sweep_params.values()
                )
                if not is_bunch_to_bunch:
                    aperture_points = int(self.aperture_points_var.get())
                    if aperture_points < 2:
                        return "Sweep mode: Aperture must have at least 2 points"
                if energy_points < 2 and not has_swept_sub_param:
                    return "Sweep mode: Energy must have at least 2 points (or enable a swept sub-parameter)"
            else:
                # Optimization mode allows 1 point (fixed) for any parameter
                if not is_bunch_to_bunch:
                    aperture_points = int(self.aperture_points_var.get())
                    if aperture_points < 1:
                        return "Aperture must have at least 1 point"
                if energy_points < 1:
                    return "Energy must have at least 1 point"

            # Lists
            parse_float_list(self.offset_fractions_var.get())
            # Single float for rider starting z
            float(self.start_z_var.get())

            # Wall and steps
            float(self.wall_z_var.get())
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
            float(self.sweep_params["rider_stripped_ions"]["fixed_var"].get())
            if self.sim_type_var.get() == "BUNCH_TO_BUNCH":
                float(self.sweep_params["driver_stripped_ions"]["fixed_var"].get())

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
                "[DEBUG] _gather_config: Existing config available (will be used as fallback)"
            )
        else:
            print(
                "[DEBUG] _gather_config: No existing config, using defaults as fallback"
            )

        if has_gui:
            print(
                "[DEBUG] _gather_config: Reading stability settings from main GUI Stability tab"
            )
        else:
            print(
                "[DEBUG] _gather_config: No GUI available, using existing config or defaults"
            )

        rider_offset = parse_offset_pair(self.offset_fractions_var.get())
        driver_offset = parse_offset_pair(self.driver_offset_var.get())

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
            # Force aperture_points=1 for BUNCH_TO_BUNCH (aperture not applicable)
            aperture_points=(
                1
                if SimulationType[self.sim_type_var.get()]
                == SimulationType.BUNCH_TO_BUNCH
                else int(self.aperture_points_var.get())
            ),
            aperture_log_scale=self.aperture_log_var.get(),
            energy_range=(
                float(self.energy_min_var.get()),
                float(self.energy_max_var.get()),
            ),
            energy_points=int(self.energy_points_var.get()),
            energy_log_scale=self.energy_log_var.get(),
            transverse_offset_fractions=parse_float_list(self.offset_fractions_var.get()),
            starting_z_positions=[float(self.start_z_var.get())],
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
            transv_offset_x=rider_offset[0],
            transv_offset_y=rider_offset[1],
            driver_transv_offset_x=driver_offset[0],
            driver_transv_offset_y=driver_offset[1],
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
            stripped_ions=float(
                self.sweep_params["rider_stripped_ions"]["fixed_var"].get()
            ),
            driver_m_particle=float(
                self.sweep_params["driver_m_particle"]["fixed_var"].get()
            ),
            driver_charge_sign=float(
                self.sweep_params["driver_charge_sign"]["fixed_var"].get()
            ),
            driver_pcount=int(self.sweep_params["driver_pcount"]["fixed_var"].get()),
            driver_transv_mom=float(
                self.sweep_params["driver_transv_mom"]["fixed_var"].get()
            ),
            driver_transv_dist=float(
                self.sweep_params["driver_transv_dist"]["fixed_var"].get()
            ),
            driver_starting_distance=float(
                self.sweep_params["driver_starting_distance"]["fixed_var"].get()
            ),
            driver_stripped_ions=float(
                self.sweep_params["driver_stripped_ions"]["fixed_var"].get()
            ),
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
            failed_run_retry_attempts=int(self.failed_run_retry_attempts_var.get()),
            # Conducting wall image parameters - read from main GUI
            image_subcharge_count=self._get_gui_stability_setting(
                "image_subcharge_var",
                existing_config.image_subcharge_count if existing_config else 12,
            ),
            use_image_weighting=self._get_gui_stability_setting(
                "image_weighting_var",
                existing_config.use_image_weighting if existing_config else True,
            ),
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
            # Gamma reconciliation parameters
            self_consistency_gamma_reconciliation_method=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_method_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_method
                    if existing_config
                    else "DISABLED"
                ),
            ),
            self_consistency_gamma_reconciliation_low_beta_threshold=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_low_beta_threshold_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_low_beta_threshold
                    if existing_config
                    else 0.9
                ),
            ),
            self_consistency_gamma_reconciliation_high_beta_threshold=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_high_beta_threshold_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_high_beta_threshold
                    if existing_config
                    else 0.99
                ),
            ),
            self_consistency_gamma_reconciliation_low_beta_weight=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_low_beta_weight_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_low_beta_weight
                    if existing_config
                    else 0.8
                ),
            ),
            self_consistency_gamma_reconciliation_high_beta_weight=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_high_beta_weight_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_high_beta_weight
                    if existing_config
                    else 0.2
                ),
            ),
            self_consistency_gamma_reconciliation_mid_beta_weight=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_mid_beta_weight_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_mid_beta_weight
                    if existing_config
                    else 0.5
                ),
            ),
            self_consistency_gamma_reconciliation_fixed_weight=self._get_gui_stability_setting(
                "self_consistency_gamma_reconciliation_fixed_weight_var",
                (
                    existing_config.self_consistency_gamma_reconciliation_fixed_weight
                    if existing_config
                    else 0.5
                ),
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
            # Startup mode - read from main GUI core params
            startup_mode=(
                self.gui_controller.core_param_vars["startup_mode"].get()
                if self.gui_controller
                and hasattr(self.gui_controller, "core_param_vars")
                else (existing_config.startup_mode if existing_config else "COLD_START")
            ),
        )

        driver_negative = (
            getattr(self, "driver_direction_var", None) is None
            or getattr(self, "driver_direction_var").get() == "-z"
        )
        linked_energy_sweep = getattr(
            self, "link_driver_rider_energy_var", tk.BooleanVar(value=False)
        ).get()

        return apply_sweep_parameter_overrides(
            config_obj,
            self.sweep_params,
            driver_negative=driver_negative,
            linked_energy_sweep=linked_energy_sweep,
            debug=print,
        )

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
            self._sync_main_gui_simulation_type(opt_config.simulation_type.name)
            self.wall_z_var.set(str(opt_config.wall_z))
            self.cavity_spacing_var.set(str(opt_config.cavity_spacing))

            # Set timestep mode and values based on loaded config
            # Default to "duration" mode (auto-calc duration, user provides count)
            self.timestep_mode_var.set("duration")
            self.steps_var.set(str(opt_config.steps))
            self.duration_var.set(f"{opt_config.timestep:.2e}")
            self._toggle_timestep_mode()  # Update UI state
            self._set_fixed_sweep_value(
                "rider_m_particle", f"{opt_config.m_particle:.14e}"
            )
            self._set_fixed_sweep_value(
                "rider_charge_sign", str(opt_config.charge_sign)
            )
            self._set_fixed_sweep_value("rider_pcount", str(opt_config.pcount))
            self._set_fixed_sweep_value(
                "rider_stripped_ions", str(opt_config.stripped_ions)
            )
            self._set_fixed_sweep_value(
                "rider_transv_mom", f"{opt_config.transv_mom:.2e}"
            )
            self._set_fixed_sweep_value(
                "rider_transv_dist", f"{opt_config.transv_dist:.2e}"
            )
            self._apply_macroparticle_ui_state(
                enabled=getattr(opt_config, "macroparticle_enabled", False),
                charge_multiplier=f"{getattr(opt_config, 'macroparticle_charge_multiplier', 1.0):.2e}",
                sigma_multiplier=f"{getattr(opt_config, 'macroparticle_sigma_multiplier', 1.0):.2e}",
                momentum_errors=getattr(
                    opt_config, "macroparticle_use_momentum_errors", True
                ),
                refresh_state=True,
            )
            self.main_timestep_display_var.set(f"{opt_config.timestep:.2e}")

            # Load driver parameters if BUNCH_TO_BUNCH
            if main_options.simulation_type == SimulationType.BUNCH_TO_BUNCH:
                if self._apply_driver_sweep_values(main_options.driver_params):
                    self._log_result("[INFO] Loaded driver parameters from main GUI")

                    # Update starting position field (rider only; driver uses sweep param)
                    rider_start_z = main_options.rider_params.get(
                        "starting_distance", 0.0
                    )
                    self.start_z_var.set(f"{rider_start_z}")
            else:
                # For non-BUNCH_TO_BUNCH modes, only set rider starting position
                rider_start_z = main_options.rider_params.get("starting_distance", 0.0)
                self.start_z_var.set(f"{rider_start_z}")

            # Update stability options if they exist in config
            if hasattr(opt_config, "smoothness_enabled"):
                self._apply_smoothness_ui_state(
                    enabled=opt_config.smoothness_enabled,
                    window_size=str(opt_config.smoothness_window_size),
                    oscillation_threshold=str(
                        opt_config.smoothness_oscillation_threshold
                    ),
                    reject_on_violation=opt_config.smoothness_reject_on_violation,
                )

            self._log_result("[OK] Loaded parameters from main GUI configuration")
            self._log_result(f"  Simulation type: {opt_config.simulation_type.name}")
            self._log_result(f"  Wall z: {opt_config.wall_z} mm")
            self._log_result(f"  Cavity spacing: {opt_config.cavity_spacing} mm")
            self._log_result(
                "  Timestep mode: auto-calc duration (user provides count)"
            )
            self._log_result(f"  Steps: {opt_config.steps}")
            self._log_result(f"  Duration: {opt_config.timestep:.2e} ns")
            self._log_result(f"  Particle mass: {opt_config.m_particle:.6e} amu")
            self._log_result(
                f"  Transverse momentum: {opt_config.transv_mom:.2e} amu·mm/ns"
            )
            self._log_result(f"  Transverse distance: {opt_config.transv_dist:.2e} mm")
            self._log_result("")

            # Update GUI controller's simulation type and driver visibility
            # This ensures the main GUI shows the correct simulation mode and driver parameters
            if self.gui_controller:
                self._sync_main_gui_visibility_state()

            # Update sweep optimizer's own driver visibility based on loaded simulation type
            self._update_driver_visibility()

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

        # Max refinement attempts is now auto-calculated (read-only display)
        import math

        try:
            reduction_factor = self.config.adaptive_timestep_reduction_factor
            min_factor = self.config.adaptive_timestep_min_factor
            if reduction_factor > 1 and min_factor > 0:
                calculated_attempts = math.ceil(
                    math.log(1.0 / min_factor) / math.log(reduction_factor)
                )
                attempts_display = f"{max(1, calculated_attempts)} (auto-calculated from reduction factor & min timestep)"
            else:
                attempts_display = "N/A"
        except (ValueError, ZeroDivisionError):
            attempts_display = "N/A"

        ttk.Label(at_frame, text="Max reduction attempts:").pack(
            anchor="w", pady=(5, 0)
        )
        at_attempts_display = ttk.Label(
            at_frame,
            text=attempts_display,
            relief="sunken",
            background="#f0f0f0",
            foreground="#606060",
            padding=(5, 2),
            font=("TkDefaultFont", 9, "italic"),
        )
        at_attempts_display.pack(anchor="w")
        all_widgets.append(at_attempts_display)

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
            # Adaptive timestep: debug enabled, don't halt
            # Note: max_attempts is now auto-calculated from reduction_factor and min_timestep_factor
            at_debug_var.set(True)
            at_halt_var.set(False)

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
                # max_refinement_attempts is now auto-calculated from reduction_factor and min_timestep_factor
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
                f"  Adaptive timestep: {self.config.adaptive_timestep_enabled} (threshold={self.config.adaptive_timestep_threshold * 100:.0f}%, reduction={self.config.adaptive_timestep_reduction_factor}x, min_factor={self.config.adaptive_timestep_min_factor}, debug={self.config.adaptive_timestep_debug})"
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

        # For BUNCH_TO_BUNCH, energy is kinetic; for others, it's total
        if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
            gamma_max = (energy_max * 1e3) / rest_energy_mev + 1.0
        else:
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
            self.config.failed_run_retry_attempts = int(
                self.failed_run_retry_attempts_var.get()
            )

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

    def _set_fixed_sweep_value(self, param_name: str, value: str):
        """Update a fixed-value sweep control."""
        self.sweep_params[param_name]["fixed_var"].set(value)

    def _apply_macroparticle_ui_state(
        self,
        *,
        enabled: bool,
        charge_multiplier: str,
        sigma_multiplier: str,
        momentum_errors: bool,
        refresh_state: bool = False,
    ):
        """Apply macroparticle-related UI state."""
        self.macroparticle_enabled_var.set(enabled)
        self._set_fixed_sweep_value(
            "macroparticle_charge_multiplier", charge_multiplier
        )
        self._set_fixed_sweep_value("macroparticle_sigma_multiplier", sigma_multiplier)
        self.macroparticle_momentum_errors_var.set(momentum_errors)
        if refresh_state:
            self._toggle_macroparticle_controls()
            self._update_macroparticle_state()

    def _apply_smoothness_ui_state(
        self,
        *,
        enabled: bool,
        window_size: str,
        oscillation_threshold: str,
        reject_on_violation: bool,
    ):
        """Apply smoothness-related UI state."""
        self.smoothness_enabled_var.set(enabled)
        self.smoothness_window_var.set(window_size)
        self.smoothness_oscillation_var.set(oscillation_threshold)
        self.smoothness_reject_var.set(reject_on_violation)
        self._toggle_smoothness_controls()

    def _apply_driver_sweep_values(self, driver_params: dict[str, Any] | None) -> bool:
        """Populate driver sweep controls from driver parameters."""
        if not driver_params:
            return False

        self._set_fixed_sweep_value(
            "driver_m_particle", f"{driver_params.get('m_particle', 207.2):.6e}"
        )
        self._set_fixed_sweep_value(
            "driver_charge_sign", str(driver_params.get("charge_sign", 1.0))
        )
        self._set_fixed_sweep_value(
            "driver_pcount", str(driver_params.get("pcount", 5))
        )
        self._set_fixed_sweep_value(
            "driver_transv_mom", f"{driver_params.get('transv_mom', 0.0):.2e}"
        )
        self._set_fixed_sweep_value(
            "driver_transv_dist", f"{driver_params.get('transv_dist', -0.07998):.6e}"
        )
        self._set_fixed_sweep_value(
            "driver_starting_distance",
            f"{driver_params.get('starting_distance', 1000.0):.2e}",
        )
        driver_pz = driver_params.get("starting_Pz", -4925.0)
        driver_mass = driver_params.get("m_particle", 207.2)
        driver_energy = calculate_energy_from_pz(driver_pz, driver_mass)
        self._set_fixed_sweep_value("driver_energy_gev", f"{driver_energy:.6e}")
        self._set_fixed_sweep_value(
            "driver_stripped_ions", str(driver_params.get("stripped_ions", 54.0))
        )
        return True

    def _sync_main_gui_simulation_type(self, sim_type_value: str):
        """Sync the selected simulation type back to the main GUI, if available."""
        if not (self.gui_controller and hasattr(self.gui_controller, "sim_type_var")):
            return

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

    def _sync_main_gui_visibility_state(self):
        """Refresh main GUI visibility state affected by simulation type."""
        if not self.gui_controller:
            return

        if hasattr(self.gui_controller, "_update_driver_visibility"):
            self.gui_controller._update_driver_visibility()
        if hasattr(self.gui_controller, "_update_image_subcharge_state"):
            self.gui_controller._update_image_subcharge_state()

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
            sim_type_value = data.get("simulation_type", "CONDUCTING_WALL")
            self.sim_type_var.set(sim_type_value)
            self._sync_main_gui_simulation_type(sim_type_value)

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
            # Load first value from starting_z_positions list (single float input)
            start_z_list = data.get("starting_z_positions", [0.0])
            self.start_z_var.set(str(start_z_list[0] if start_z_list else 0.0))
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

            # Load trajectory options
            self.save_top_n_traj_var.set(data.get("save_top_n_trajectories", False))
            self.save_all_traj_var.set(data.get("save_all_trajectories", False))
            self.save_failed_traj_var.set(data.get("save_failed_trajectories", False))
            self.trajectory_stride_var.set(str(data.get("trajectory_stride", 10)))

            metrics_format, metrics_scope = metrics_export_settings_from_data(data)
            self.metrics_format_var.set(metrics_format)
            self.metrics_scope_var.set(metrics_scope)

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

            loaded_config = apply_persisted_config_overrides(loaded_config, data)

            print("[DEBUG] _load_config_from_path: Assigning loaded_config to self.config")
            print(f"  SC enabled: {loaded_config.self_consistency_enabled}")
            print(f"  SC tolerance: {loaded_config.self_consistency_tolerance}")
            print(f"  AT enabled: {loaded_config.adaptive_timestep_enabled}")
            print(f"  AT debug: {loaded_config.adaptive_timestep_debug}")
            self.config = loaded_config

            # Update UI controls
            self.per_run_timeout_var.set(str(loaded_config.per_run_timeout))
            self.skip_failed_runs_var.set(loaded_config.skip_failed_runs)
            self.failed_run_retry_attempts_var.set(
                str(loaded_config.failed_run_retry_attempts)
            )

            # Load UI-specific fields
            self.timestep_mode_var.set(data.get("timestep_mode", "duration"))
            self.auto_steps_distance_var.set(str(data.get("auto_steps_distance", 10.0)))
            self.trajectory_stride_var.set(str(data.get("trajectory_stride", 10)))
            self.sweep_params["rider_stripped_ions"]["fixed_var"].set(
                str(data.get("rider_stripped_ions", 1.0))
            )
            # Load rider/driver offset pairs
            rider_x = data.get("rider_offset_x", 0.0)
            rider_y = data.get("rider_offset_y", 0.0)
            self.offset_fractions_var.set(f"{rider_x}, {rider_y}")
            driver_x = data.get("driver_offset_x", 0.0)
            driver_y = data.get("driver_offset_y", 0.0)
            self.driver_offset_var.set(f"{driver_x}, {driver_y}")
            self.sweep_params["driver_stripped_ions"]["fixed_var"].set(
                str(data.get("driver_stripped_ions", 54.0))
            )
            self._toggle_timestep_mode()

            # Update stability controls (smoothness has UI variables)
            self._apply_smoothness_ui_state(
                enabled=loaded_config.smoothness_enabled,
                window_size=str(loaded_config.smoothness_window_size),
                oscillation_threshold=str(
                    loaded_config.smoothness_oscillation_threshold
                ),
                reject_on_violation=loaded_config.smoothness_reject_on_violation,
            )

            # Update main GUI stability tab if available
            if self.gui_controller:
                self._sync_stability_to_main_gui(loaded_config)

                # Sync image parameters to main GUI Particles tab
                if hasattr(self.gui_controller, "image_subcharge_var"):
                    self.gui_controller.image_subcharge_var.set(
                        loaded_config.image_subcharge_count
                    )
                if hasattr(self.gui_controller, "image_weighting_var"):
                    self.gui_controller.image_weighting_var.set(
                        loaded_config.use_image_weighting
                    )

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
                f"  Adaptive timestep min_factor: {loaded_config.adaptive_timestep_min_factor}"
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
            self._apply_macroparticle_ui_state(
                enabled=loaded_config.macroparticle_enabled,
                charge_multiplier=str(loaded_config.macroparticle_charge_multiplier),
                sigma_multiplier=str(loaded_config.macroparticle_sigma_multiplier),
                momentum_errors=loaded_config.macroparticle_use_momentum_errors,
                refresh_state=True,
            )

            # Load sweep parameter states dynamically
            sweep_state = data.get("sweep_parameters", {})
            # Restore driver direction selector if present in config
            if hasattr(self, "driver_direction_var"):
                direction = data.get("driver_direction", "-z")
                self.driver_direction_var.set(
                    direction if direction in ("-z", "+z") else "-z"
                )

            for param_name, controls in self.sweep_params.items():
                state = resolve_loaded_sweep_state(sweep_state, param_name)
                if state is None:
                    continue

                if state.get("enabled", False):
                    controls["sweep_var"].set(True)
                    controls["min_var"].set(str(state.get("min", "")))
                    controls["max_var"].set(str(state.get("max", "")))
                    controls["points_var"].set(str(state.get("points", "3")))
                    controls["log_var"].set(state.get("log", False))
                    self._toggle_sweep_controls(param_name)
                else:
                    controls["sweep_var"].set(False)
                    fixed_val = state.get("fixed_value", controls["fixed_var"].get())
                    controls["fixed_var"].set(str(fixed_val))
                    self._toggle_sweep_controls(param_name)

            # Load linked energy sweep setting
            if hasattr(self, "link_driver_rider_energy_var"):
                linked_energy = data.get("linked_energy_sweep", False)
                self.link_driver_rider_energy_var.set(linked_energy)
                # Trigger the toggle handler to update UI state
                self._on_link_energy_toggled()
                if linked_energy:
                    self._log_result(
                        "[INFO] Linked energy sweep mode enabled - driver energy follows rider energy"
                    )

            # Update helper texts
            self._update_driver_visibility()
            self._update_rider_pz_helper()
            self._update_driver_pz_helper()

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
                f"  Min timestep factor: {self.config.adaptive_timestep_min_factor}"
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

                # Update driver visibility and image subcharge state based on simulation type
                # This ensures driver parameters are shown for BUNCH_TO_BUNCH and hidden otherwise
                self._sync_main_gui_visibility_state()

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
            print("[DEBUG] _save_config_to_path: Gathering config for save")
            config = self._gather_config()
            print("[DEBUG] After _gather_config:")
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

            data = build_saved_config_payload(
                config,
                timestep_mode=self.timestep_mode_var.get(),
                auto_steps_distance=float(self.auto_steps_distance_var.get()),
                rider_stripped_ions=float(
                    self.sweep_params["rider_stripped_ions"]["fixed_var"].get()
                ),
                driver_stripped_ions=float(
                    self.sweep_params["driver_stripped_ions"]["fixed_var"].get()
                ),
                driver_direction=getattr(
                    self, "driver_direction_var", tk.StringVar(value="-z")
                ).get(),
                sweep_state=sweep_state,
            )

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            # Update last_loaded_config so sweep results use correct name
            self.last_loaded_config = filepath

            self._log_result(f"[OK] Configuration saved to {filepath}")
            print("[DEBUG] Chrono settings saved to config:")
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

            parsed = parse_results_payload(
                data,
                m_particle_amu=getattr(self.config, "m_particle", 0.00054857990907),
                amu_to_mev=AMU_TO_MEV,
            )

            if parsed["kind"] == "sweep":
                results = parsed["results"]
                if not results:
                    _show_info_dialog(self, "No Results", "No results found in file.")
                    return

                results_with_traj = parsed["results_with_trajectories"]
                if not results_with_traj:
                    self._show_results_summary(results, file_path)
                    return

            elif parsed["kind"] == "optimization":
                import os

                results_dir = os.path.dirname(file_path)
                self._view_npz_trajectories(results_dir)
                return

            else:
                results_with_traj = parsed["results_with_trajectories"]

            # Create trajectory viewer dialog and automatically plot
            self._show_trajectory_viewer(results_with_traj, file_path, auto_plot=True)

        except ValueError as e:
            if str(e) == UNKNOWN_RESULTS_FORMAT_MESSAGE:
                _show_info_dialog(self, "Unknown Format", UNKNOWN_RESULTS_FORMAT_MESSAGE)
                return
            raise
        except Exception as e:
            import traceback

            _show_error_dialog(
                self,
                "Error Loading File",
                f"Failed to load file:\n{e}\n\n{traceback.format_exc()}",
            )

    def _on_plot_trajectories(self):
        """Open trajectory plotting dialog to visualize saved results."""
        import glob
        import os

        # Start with the configured sweep output directory when it has results.
        if os.path.exists(self.sweep_output_dir) and os.listdir(self.sweep_output_dir):
            base_dir = self.sweep_output_dir
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

            parsed = parse_results_payload(
                data,
                m_particle_amu=getattr(self.config, "m_particle", 0.00054857990907),
                amu_to_mev=AMU_TO_MEV,
            )

            if parsed["kind"] == "sweep":
                results = parsed["results"]
                if not results:
                    _show_info_dialog(self, "No Results", "No results found in file.")
                    return

                results_with_traj = parsed["results_with_trajectories"]

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

            elif parsed["kind"] == "optimization":
                import os

                results_dir = os.path.dirname(file_path)
                self._view_npz_trajectories(results_dir)
                return

            else:
                results_with_traj = parsed["results_with_trajectories"]

            # Create trajectory viewer dialog
            self._show_trajectory_viewer(results_with_traj, file_path)

        except ValueError as e:
            if str(e) == UNKNOWN_RESULTS_FORMAT_MESSAGE:
                _show_info_dialog(
                    self,
                    "Unknown Format",
                    f"{UNKNOWN_RESULTS_FORMAT_MESSAGE}\n\n"
                    "Note: CSV files only contain metrics, not trajectory data.",
                )
                return
            raise
        except Exception as e:
            import traceback

            _show_error_dialog(
                self,
                "Error Loading File",
                f"Failed to load file:\n{e}\n\n{traceback.format_exc()}",
            )

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
                row_data = summarize_result_row(r)

                if has_beam_optics:
                    row = (
                        f"{row_data['run_num']:<5} {row_data['aperture']:<15.3e} "
                        f"{row_data['energy']:<15.2f} {row_data['start_z']:<15.1f} "
                        f"{row_data['delta_e']:<12.3f} {row_data['traveled']:<15.1f} "
                        f"{row_data['gamma_initial']:<12.1f} "
                        f"{row_data['emit_x']:<15.3e} {row_data['norm_emit_x']:<16.3e} "
                        f"{row_data['beta_x']:<12.3e}\n"
                    )
                else:
                    row = (
                        f"{row_data['run_num']:<5} {row_data['aperture']:<15.3e} "
                        f"{row_data['energy']:<15.2f} {row_data['start_z']:<15.1f} "
                        f"{row_data['delta_e']:<12.3f} {row_data['traveled']:<15.1f} "
                        f"{row_data['gamma_initial']:<12.1f}\n"
                    )
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
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )

            plot_data = collect_summary_plot_data(results)
            apertures = plot_data["apertures"]
            energies = plot_data["energies"]
            delta_es = plot_data["delta_es"]

            # Create figure
            fig = plt.figure(figsize=(10, 6))

            # Determine if we have 1D or 2D sweep
            unique_apertures = len(set(apertures))
            unique_energies = len(set(energies))

            if unique_apertures > 1 and unique_energies > 1:
                # 2D sweep - make heatmap
                ax = fig.add_subplot(111)
                unique_a, unique_e, grid = build_summary_heatmap_grid(results)

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
                    row = summarize_result_row(r)

                    writer.writerow(
                        [
                            row["run_num"],
                            row["aperture"],
                            row["energy"],
                            row["start_z"],
                            row["delta_e"],
                            row["traveled"],
                            row["gamma_initial"],
                            row["gamma_final"],
                            row["emit_x"],
                            row["emit_y"],
                            row["norm_emit_x"],
                            row["norm_emit_y"],
                            row["beta_x"],
                            row["beta_y"],
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

            plot_data = build_trajectory_plot_data(
                selected_results,
                m_particle_amu=getattr(self.config, "m_particle", 0.00054857990907),
                amu_to_mev=AMU_TO_MEV,
            )
            heatmap = plot_data["heatmap"]

            # Plot each selected trajectory
            for idx, series in enumerate(plot_data["series"]):
                label = (
                    f"Run #{series['run_num']} "
                    f"(a={series['aperture']:.2e}mm, E={series['energy']:.1f}GeV)"
                )
                color = plt.cm.tab10(idx % 10)

                ax_delta_e.plot(
                    series["z"],
                    series["energy_delta"],
                    label=label,
                    alpha=0.7,
                    color=color,
                    linewidth=1.5,
                )

                # Plot 2: x and y positions versus z (need to extract from r)
                # Since we only have r (radial distance), we'll plot r and -r to show transverse extent
                # In a real case, you'd have separate x and y coordinates
                ax_transverse.plot(
                    series["z"],
                    series["r"],
                    label=f"{label} (+r)",
                    alpha=0.6,
                    color=color,
                    linewidth=1.5,
                )
                ax_transverse.plot(
                    series["z"],
                    -series["r"],
                    alpha=0.3,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                )

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
            apertures = heatmap["apertures"]
            energies = heatmap["energies"]
            delta_es = heatmap["delta_es"]
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

    def _update_progress(self, value: float, text: str):
        """Update progress bar and label (thread-safe)."""

        def update():
            self.progress_bar["value"] = value
            self.progress_label["text"] = text

        self.after(0, update)

    def _update_progress_text(self, text: str):
        """Update only the progress label text (thread-safe)."""
        self.after(0, lambda: self.progress_label.config(text=text))

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
