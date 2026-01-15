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
from dataclasses import dataclass, field
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
from optimization.config import OptimizationConfig
from optimization.result_io import (
    generate_optimization_heatmap,
    generate_optimization_plots,
    generate_trajectory_comparison_plot,
    save_optimization_results,
    save_partial_optimization_results,
    save_top_n_optimization_trajectories,
    save_top_trajectories_summary_table,
)


class ToolTip:
    """Simple tooltip widget for displaying help text on hover."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        """Display the tooltip."""
        if self.tip_window or not self.text:
            return
        x, y, _, _ = (
            self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        )
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("TkDefaultFont", 9),
        )
        label.pack(ipadx=5, ipady=3)

    def hide_tip(self, event=None):
        """Hide the tooltip."""
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


def _show_error_dialog(parent: tk.Widget, title: str, message: str) -> None:
    """Show an error dialog with selectable text."""
    # Log to console/terminal
    print(f"ERROR: {title}: {message}", flush=True)

    # Log to results text if parent is OptimizationPlugin
    if hasattr(parent, "_log_result"):
        parent._log_result(f"[ERROR] {title}: {message}")

    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()

    frame = ttk.Frame(dialog, padding=10)
    frame.pack(fill="both", expand=True)

    text = tk.Text(frame, wrap="word", height=8, width=60, relief="flat", borderwidth=0)
    text.insert("1.0", message)
    text.configure(state="disabled")
    text.pack(side="top", fill="both", expand=True, pady=(0, 10))

    button_frame = ttk.Frame(frame)
    button_frame.pack(side="bottom")
    ok_button = ttk.Button(button_frame, text="OK", command=dialog.destroy, width=10)
    ok_button.pack()
    ok_button.focus_set()

    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f"+{x}+{y}")

    dialog.bind("<Return>", lambda e: dialog.destroy())
    dialog.bind("<Escape>", lambda e: dialog.destroy())


def _show_info_dialog(parent: tk.Widget, title: str, message: str) -> None:
    """Show an info dialog with selectable text."""
    # Log to console/terminal
    print(f"INFO: {title}: {message}", flush=True)
    timestep_range: Optional[Tuple[float, float]] = None  # ns (proper time)
    timestep_points: int = 1
    starting_z_range: Optional[Tuple[float, float]] = None  # mm
    starting_z_points: int = 1
    wall_z_range: Optional[Tuple[float, float]] = None  # mm
    wall_z_points: int = 1
    particle_mass_range: Optional[Tuple[float, float]] = None  # amu
    particle_mass_points: int = 1
    particle_charge_range: Optional[Tuple[float, float]] = None  # charge_sign
    particle_charge_points: int = 1
    cavity_spacing_range: Optional[Tuple[float, float]] = None  # mm (SWITCHING_WALL)
    cavity_spacing_points: int = 1
    macroparticle_charge_range: Optional[Tuple[float, float]] = (
        None  # charge multiplier
    )
    macroparticle_charge_points: int = 1
    macroparticle_sigma_range: Optional[Tuple[float, float]] = None  # sigma multiplier
    macroparticle_sigma_points: int = 1

    # Fixed parameters
    wall_z: float = 100.0  # mm
    cavity_spacing: float = 1e5  # mm (for SWITCHING_WALL)
    steps: int = 2000
    timestep: float = 3e-7  # ns (proper time) - default from main GUI
    auto_steps: bool = False  # Automatically calculate steps based on distance
    auto_steps_target: int = (
        500  # Target number of steps when auto-calculating timestep
    )
    auto_steps_distance_past_wall: float = 10.0  # mm past wall to stop integration
    seed: int = 12345

    # Timestep strategy for energy sweeps
    timestep_strategy: str = (
        "auto_distance"  # "fixed", "energy_scaled", or "auto_distance"
    )
    energy_scale_exponent: float = 1.0  # For energy_scaled: h ∝ γ^-α
    target_distance_mm: float = 100.0  # For auto_distance: distance to reach
    z_cutoff_mode: str = "absolute"  # "absolute" or "relative" (for BUNCH_TO_BUNCH)

    # Fixed particle parameters (not swept)
    transv_mom: float = 1.2e-05  # amu·mm/ns
    transv_dist: float = 2e-06  # mm - transverse spread (half-width of distribution)
    transv_offset_x: float = 0.0  # mm - x-offset of bunch center from axis
    transv_offset_y: float = 0.0  # mm - y-offset of bunch center from axis
    m_particle: float = 0.00054857990907  # amu (electron mass)
    pcount: int = 1
    charge_sign: float = -1.0
    stripped_ions: float = 1.0

    # Macroparticle simulation options (CONDUCTING_WALL only)
    macroparticle_enabled: bool = False
    macroparticle_charge_multiplier: float = 1.0
    macroparticle_sigma_multiplier: float = 1.0  # Multiplier for bunch spread params
    macroparticle_use_momentum_errors: bool = (
        True  # Include momentum-based cumulative errors
    )

    # Optimization objective
    objective: str = "max_energy_gain"  # Primary objective to optimize

    # Multi-objective weighting (for future use)
    objective_weights: Dict[str, float] = (
        None  # e.g., {"max_energy_gain": 1.0, "min_transverse_spread": 0.5}
    )

    # Output
    output_dir: str = "results/sweeps"
    save_results: bool = True
    save_plots: bool = True

    # Trajectory saving options
    save_top_n_trajectories: bool = False  # Save trajectories for top N results
    save_all_trajectories: bool = False  # Save ALL evaluation trajectories
    save_failed_trajectories: bool = False  # Save only failed trajectories
    trajectory_stride: int = (
        1  # Save every Nth point to reduce file size (only used with "All")
    )

    # Metrics export options
    metrics_export_format: str = "both"  # Options: "json", "csv", "both", "none"
    metrics_export_scope: str = "all"  # Options: "all", "top_n"

    # Log saving options
    log_verbosity: str = "truncated"  # "none", "truncated", "full", "top_n_only"
    # none = no debug logs saved
    # truncated = 1-2 lines per run with parameters + metrics + errors/warnings only
    # full = complete debug output with SC iterations and adaptive timestep refinements
    # top_n_only = logs only for top N trajectories

    # Stability and robustness options (from SimulationOptions)
    self_consistency_enabled: bool = True
    self_consistency_tolerance: float = 1e-4
    self_consistency_max_iterations: int = 5
    self_consistency_verbosity: int = 2  # 0=silent, 1=summary, 2=failures, 3=full
    self_consistency_chrono_interpolate: bool = False
    self_consistency_chrono_tolerance: float = 1e-3  # ns
    self_consistency_chrono_high_precision: bool = False
    self_consistency_chrono_adaptive_tolerance: bool = False

    # Energy monitoring removed - functionality integrated into adaptive timestep
    energy_monitor_enabled: bool = False
    energy_monitor_threshold: float = 2.0
    energy_monitor_check_interval: int = 10
    energy_monitor_halt_on_jump: bool = False  # Now in adaptive_timestep
    energy_monitor_debug: bool = False

    adaptive_timestep_enabled: bool = True
    adaptive_timestep_threshold: float = 0.10
    adaptive_timestep_reduction_factor: int = 10
    adaptive_timestep_max_attempts: int = 5
    adaptive_timestep_min_factor: float = 1e-4
    adaptive_timestep_cooldown_steps: int = 10
    adaptive_timestep_probe_threshold: float = 0.01
    adaptive_timestep_max_probe_steps: int = 3
    adaptive_timestep_debug: bool = False

    # Sweep robustness options
    per_run_timeout: float = 300.0  # seconds (0 = no timeout, default 5 minutes)
    skip_failed_runs: bool = True  # Continue sweep even if individual runs fail

    # Trajectory stability checking (multi-step numerical validation)
    smoothness_enabled: bool = True  # Enable trajectory stability analysis
    smoothness_window_size: int = 20  # Steps for moving-window analysis
    smoothness_oscillation_threshold: float = (
        0.5  # Max oscillation score (sign-change rate)
    )
    smoothness_trend_threshold: float = 0.30  # Max polynomial fit residual
    smoothness_reject_on_violation: bool = True  # Reject numerically unstable runs
    smoothness_max_violations: int = 3  # Max violations before rejection

    def __post_init__(self):
        """Set defaults for list fields."""
        if self.transverse_offset_fractions is None:
            self.transverse_offset_fractions = [0.0]
        if self.starting_z_positions is None:
            self.starting_z_positions = [0.0]  # Default: start at origin
        if self.objective_weights is None:
            self.objective_weights = {}

    def calculate_timestep_for_energy(
        self,
        energy_gev: float,
        m_particle_amu: float = 0.00054857990907,
        wall_z: float = None,
        start_z: float = 0.0,
    ) -> float:
        """Calculate appropriate timestep for given energy based on strategy.

        Parameters
        ----------
        energy_gev : float
            Particle energy in GeV
        m_particle_amu : float
            Particle mass in amu (default: electron)
        wall_z : float, optional
            Wall position in mm (required for auto_distance strategy)
        start_z : float, optional
            Starting z position in mm (default: 0.0)

        Returns
        -------
        float
            Timestep in ns (proper time)
        """
        if self.timestep_strategy == "fixed":
            return self.timestep

        # Calculate gamma and beta
        rest_energy_mev = m_particle_amu * 931.494  # amu to MeV
        gamma = (energy_gev * 1e3) / rest_energy_mev
        beta = np.sqrt(1.0 - 1.0 / gamma**2)

        if self.timestep_strategy == "energy_scaled":
            # Scale timestep inversely with gamma
            # h_sweep = h_base / γ^α
            return self.timestep / (gamma**self.energy_scale_exponent)

        elif self.timestep_strategy == "auto_distance":
            # Calculate timestep to reach target distance in given steps
            # Total distance = from start_z to wall_z + target_distance_mm
            # Distance = N_steps × β × c × h × γ
            # Therefore: h = Distance / (N_steps × β × c × γ)
            if wall_z is None:
                wall_z = self.wall_z

            total_distance = abs(wall_z - start_z) + self.target_distance_mm
            c_mmns = 299.792458  # mm/ns
            h_calculated = total_distance / (self.steps * beta * c_mmns * gamma)
            return h_calculated

        else:
            raise ValueError(f"Unknown timestep_strategy: {self.timestep_strategy}")

    @classmethod
    def from_simulation_options(cls, options: Any) -> "OptimizationConfig":
        """Create OptimizationConfig from SimulationOptions (main GUI config).

        Parameters
        ----------
        options : SimulationOptions
            Main GUI simulation options

        Returns
        -------
        OptimizationConfig
            Optimization config with defaults from main GUI
        """
        # Extract particle parameters from rider_params
        rider = options.rider_params
        core = options.core_params

        # Calculate timestep from options if available
        # The main GUI stores time_step in core_params
        timestep = core.get("time_step", 3e-7)

        return cls(
            simulation_type=options.simulation_type,
            wall_z=core.get("wall_z", 100.0),
            steps=options.steps,
            timestep=timestep,
            seed=options.seed,
            m_particle=rider.get("m_particle", 0.00054857990907),
            charge_sign=rider.get("charge_sign", -1.0),
            pcount=rider.get("pcount", 1),
            stripped_ions=rider.get("stripped_ions", 1.0),
            transv_mom=rider.get("transv_mom", 1.2e-05),
            transv_dist=rider.get("transv_dist", 2e-06),
            output_dir=str(options.output_dir.parent / "optimization_results"),
            # Preserve stability options from main config
            self_consistency_enabled=options.self_consistency_enabled,
            self_consistency_tolerance=options.self_consistency_tolerance,
            self_consistency_max_iterations=options.self_consistency_max_iterations,
            self_consistency_verbosity=options.self_consistency_verbosity,
            self_consistency_chrono_interpolate=getattr(
                options, "self_consistency_chrono_interpolate", False
            ),
            self_consistency_chrono_tolerance=getattr(
                options, "self_consistency_chrono_tolerance", 1e-3
            ),
            self_consistency_chrono_high_precision=getattr(
                options, "self_consistency_chrono_high_precision", False
            ),
            self_consistency_chrono_adaptive_tolerance=getattr(
                options, "self_consistency_chrono_adaptive_tolerance", False
            ),
            energy_monitor_enabled=False,  # Removed - integrated into adaptive timestep
            energy_monitor_threshold=2.0,
            energy_monitor_check_interval=10,
            energy_monitor_halt_on_jump=options.energy_monitor_halt_on_jump,
            energy_monitor_debug=False,
            adaptive_timestep_enabled=options.adaptive_timestep_enabled,
            adaptive_timestep_threshold=options.adaptive_timestep_threshold,
            adaptive_timestep_reduction_factor=options.adaptive_timestep_reduction_factor,
            adaptive_timestep_max_attempts=options.adaptive_timestep_max_attempts,
            adaptive_timestep_min_factor=options.adaptive_timestep_min_factor,
            adaptive_timestep_cooldown_steps=options.adaptive_timestep_cooldown_steps,
            adaptive_timestep_probe_threshold=options.adaptive_timestep_probe_threshold,
            adaptive_timestep_max_probe_steps=options.adaptive_timestep_max_probe_steps,
            adaptive_timestep_debug=options.adaptive_timestep_debug,
            # Default timeout and skip settings for sweeps
            per_run_timeout=300.0,
            skip_failed_runs=True,
            # Default stability checking for sweeps
            smoothness_enabled=True,
            smoothness_reject_on_violation=True,
            smoothness_max_violations=3,
        )


def calculate_auto_timestep(
    start_z: float,
    wall_z: float,
    distance_past_wall: float,
    particle_energy_gev: float,
    particle_mass_amu: float = 0.00054857990907,
    target_steps: int = 500,
) -> float:
    """Calculate appropriate timestep to achieve target number of steps.

    Parameters
    ----------
    start_z : float
        Starting z position (mm)
    wall_z : float
        Wall z position (mm)
    distance_past_wall : float
        Additional distance past wall to integrate (mm)
    particle_energy_gev : float
        Particle energy (GeV)
    particle_mass_amu : float
        Particle rest mass (amu), default is electron mass
    target_steps : int
        Target number of integration steps (default: 500)

    Returns
    -------
    float
        Calculated timestep in proper time (ns)

    Notes
    -----
    The integrator uses proper time steps h = dτ (in ns).
    Coordinate time advance is Δt = γ·h. Distance per step is Δx = β·c·Δt = β·c·γ·h.
    We solve for h given the desired number of steps and total distance.
    """
    # Calculate total distance to travel
    total_distance = abs(wall_z - start_z) + distance_past_wall

    # Calculate particle velocity (beta) and gamma
    # E = gamma * m * c^2, where m*c^2 in MeV
    # 1 amu = 931.494 MeV/c^2 (standard conversion factor)
    AMU_TO_MEV = 931.494
    rest_energy_mev = particle_mass_amu * AMU_TO_MEV
    gamma = (particle_energy_gev * 1e3) / rest_energy_mev
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.999

    # We want: total_distance = target_steps * distance_per_step
    # distance_per_step = beta * gamma * C_MMNS * timestep (proper time h)
    # So: timestep = total_distance / (target_steps * beta * gamma * C_MMNS)
    timestep = total_distance / (target_steps * beta * gamma * C_MMNS)

    return timestep


def calculate_auto_steps(
    start_z: float,
    wall_z: float,
    distance_past_wall: float,
    timestep: float,
    particle_energy_gev: float,
    particle_mass_amu: float = 0.00054857990907,
) -> int:
    """Calculate number of integration steps automatically.

    Parameters
    ----------
    start_z : float
        Starting z position (mm)
    wall_z : float
        Wall z position (mm)
    distance_past_wall : float
        Additional distance past wall to integrate (mm)
    timestep : float
        Integration timestep in proper time (ns)
    particle_energy_gev : float
        Particle energy (GeV)
    particle_mass_amu : float
        Particle rest mass (amu), default is electron mass

    Returns
    -------
    int
        Number of integration steps needed

    Notes
    -----
    The integrator uses proper time steps h (in ns), but coordinate
    time advance is Δt = γ·h. Distance per step is β·c·Δt = β·c·γ·h.
    For ultra-relativistic particles (β ≈ 1), this becomes c·γ·h.
    """
    # Calculate total distance to travel
    total_distance = abs(wall_z - start_z) + distance_past_wall

    # Calculate particle velocity (beta) and gamma
    # E = gamma * m * c^2, where m*c^2 in MeV
    # 1 amu = 931.494 MeV/c^2 (standard conversion factor)
    AMU_TO_MEV = 931.494
    rest_energy_mev = particle_mass_amu * AMU_TO_MEV
    gamma = (particle_energy_gev * 1e3) / rest_energy_mev
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.999

    # Distance traveled per step = beta * c * coordinate_time_step
    # coordinate_time_step = gamma * timestep (proper time dilation)
    # distance_per_step = beta * c * gamma * h
    # For ultra-relativistic: β ≈ 1, so distance ≈ c * gamma * h
    distance_per_step = beta * gamma * C_MMNS * timestep

    # Calculate steps needed (add 10% margin for safety)
    steps = int(np.ceil(total_distance / distance_per_step * 1.1))

    # Ensure minimum reasonable value (absolute floor of 20)
    min_steps = 20
    return max(steps, min_steps)


def calculate_steps_from_duration(
    total_duration_ns: float,
    particle_energy_gev: float,
    particle_mass_amu: float = 0.00054857990907,
) -> tuple[int, float]:
    """Calculate timestep and number of steps from total duration.

    Auto-calculates timestep (h) given a desired total duration and step count.
    Enforces a minimum of 5% of requested steps (absolute floor of 20).

    Parameters
    ----------
    total_duration_ns : float
        Desired total duration in proper time (ns)
    particle_energy_gev : float
        Particle energy (GeV)
    particle_mass_amu : float
        Particle rest mass (amu), default is electron mass

    Returns
    -------
    tuple[int, float]
        (number_of_steps, timestep_in_ns) where steps >= max(20, requested_steps * 0.05)

    Notes
    -----
    Uses proper time formulation: h = dτ = dt/γ
    Total proper time = N_steps × h
    """
    # For duration mode, we don't have a target step count to base the minimum on,
    # so use an absolute minimum of 20 steps
    min_steps = 20

    # Calculate timestep from duration and minimum steps
    timestep = total_duration_ns / min_steps

    return min_steps, timestep


class OptimizationPlugin(ttk.Frame):
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
        """Run optimization in background using selected algorithm."""
        # Set logging context for this optimization run
        method = self.config.optimization_method
        set_logging_context(f"optimization_{method}")

        # Open log file in temporary location (will be moved when results are saved)
        import tempfile
        import time

        temp_dir = tempfile.mkdtemp(prefix="opt_log_")
        self._open_log_file(temp_dir)

        start_time = time.time()

        try:
            from optimization.optimizer import (
                adaptive_grid_search,
                genetic_algorithm,
                multi_start_optimize,
                optimize_parameters,
            )

            self._log_result("=" * 80)
            self._log_result(f"OPTIMIZATION MODE: {self.config.optimization_method}")
            self._log_result("=" * 80)
            self._log_result("")

            # Apply log verbosity settings (same as sweep mode)
            # Save original values to restore later
            original_sc_verbosity = self.config.self_consistency_verbosity
            original_adaptive_debug = self.config.adaptive_timestep_debug

            use_no_logging = self.config.log_verbosity == "none"
            use_truncated_logging = self.config.log_verbosity == "truncated"
            use_full_logging = self.config.log_verbosity == "full"

            # Apply log verbosity settings - control what gets logged
            # "full" mode INHERITS stability settings from config/GUI
            # Other modes override to reduce output
            if use_full_logging:
                # INHERIT stability verbosity settings from config (don't override)
                # Use whatever was set in Stability tab or loaded from config
                self._log_result(f"Log verbosity: {self.config.log_verbosity}")
                self._log_result(
                    "  Full debug logging enabled (inherits Stability tab settings)"
                )
                self._log_result(
                    f"    SC verbosity: {self.config.self_consistency_verbosity}"
                )
                self._log_result(
                    f"    Adaptive timestep debug: {self.config.adaptive_timestep_debug}"
                )
            elif use_truncated_logging:
                # Disable verbose logging for optimizations with many evaluations
                self.config.self_consistency_verbosity = 0
                self.config.adaptive_timestep_debug = False
                self._log_result(f"Log verbosity: {self.config.log_verbosity}")
                self._log_result(
                    "  Truncated logging (parameters + metrics + errors only)"
                )
            elif use_no_logging:
                # Completely disable all debug logging
                self.config.self_consistency_verbosity = 0
                self.config.adaptive_timestep_debug = False
                self._log_result(f"Log verbosity: {self.config.log_verbosity}")
                self._log_result("  Debug logging disabled")
            else:
                # Unknown log verbosity - use config file values
                self._log_result(
                    f"Log verbosity: {self.config.log_verbosity} (unknown, using config values)"
                )
                self._log_result(
                    f"  adaptive_timestep_debug: {self.config.adaptive_timestep_debug}"
                )
                self._log_result(
                    f"  self_consistency_verbosity: {self.config.self_consistency_verbosity}"
                )
            self._log_result("")

            # Build parameter names and bounds from config
            param_names = []
            param_bounds = []

            # Aperture
            if self.config.aperture_points > 1:
                param_names.append("aperture_radius")
                param_bounds.append(self.config.aperture_range)

            # Energy
            if self.config.energy_points > 1:
                param_names.append("initial_energy_gev")
                param_bounds.append(self.config.energy_range)

            # Transverse momentum (if enabled as sweep parameter)
            if (
                self.config.transverse_momentum_range is not None
                and self.config.transverse_momentum_points > 1
            ):
                param_names.append("transverse_momentum")
                param_bounds.append(self.config.transverse_momentum_range)

            # Timestep (if enabled as sweep parameter)
            if (
                self.config.timestep_range is not None
                and self.config.timestep_points > 1
            ):
                param_names.append("timestep")
                param_bounds.append(self.config.timestep_range)

            # Rider transverse distance (spread) - if enabled as sweep parameter
            if (
                self.config.transverse_spread_range is not None
                and self.config.transverse_spread_points > 1
            ):
                param_names.append("rider_transv_dist")
                param_bounds.append(self.config.transverse_spread_range)

            # Macroparticle charge multiplier - if enabled as sweep parameter
            if (
                self.config.macroparticle_charge_range is not None
                and self.config.macroparticle_charge_points > 1
            ):
                param_names.append("macroparticle_charge_multiplier")
                param_bounds.append(self.config.macroparticle_charge_range)

            # Macroparticle sigma multiplier - if enabled as sweep parameter
            if (
                self.config.macroparticle_sigma_range is not None
                and self.config.macroparticle_sigma_points > 1
            ):
                param_names.append("macroparticle_sigma_multiplier")
                param_bounds.append(self.config.macroparticle_sigma_range)

            # Wall z position - if enabled as sweep parameter
            if self.config.wall_z_range is not None and self.config.wall_z_points > 1:
                param_names.append("wall_z")
                param_bounds.append(self.config.wall_z_range)

            if len(param_names) == 0:
                self._log_result(
                    "[ERROR] No parameters to optimize! Enable at least 2 points for aperture or energy."
                )
                self.running = False
                return

            self._log_result(f"Optimizing parameters: {param_names}")
            self._log_result(f"Parameter bounds: {param_bounds}")
            self._log_result(f"Objective: {self.config.objective}")
            self._log_result("")

            # Create base config template (this would need proper implementation)
            # For now, create a minimal dict representation
            config_template = {
                "simulation_type": self.config.simulation_type,
                "wall_z": self.config.wall_z,
                "steps": self.config.steps,
                "timestep": self.config.timestep,
                "m_particle": self.config.m_particle,
                "charge_sign": self.config.charge_sign,
                "stripped_ions": self.config.stripped_ions,
                "transv_mom": self.config.transv_mom,
                # Add other fixed parameters
            }

            # Determine metric name from objective
            metric_name = "max_energy_gain_gev"
            maximize = True

            if self.config.objective == "max_percent_energy_gain":
                metric_name = "max_percent_energy_gain"
                maximize = True
            elif "min" in self.config.objective.lower():
                maximize = False

            # Run optimization based on selected method
            method = self.config.optimization_method
            self._log_result(f"Starting {method} optimization...")
            self._log_result("")

            result = None

            # Track evaluation count and all evaluations for heatmap
            eval_counter = [0]  # Use list for mutable closure
            all_evaluations = []  # Store all parameter sets and their results

            # Create custom objective function that uses our integration runner
            def evaluate_params(x):
                """Evaluate parameter vector and return value to minimize."""
                eval_num = eval_counter[0]
                eval_counter[0] += 1

                # Log evaluation start
                param_str = ", ".join(
                    [f"{name}={val:.4g}" for name, val in zip(param_names, x)]
                )
                self._log_result(f"[OPTIMIZATION] Evaluation {eval_num}: {param_str}")

                # Check for cancellation
                if not self.running:
                    self._log_result("[CANCELLED] Optimization cancelled by user")
                    return np.inf

                if self.gui_controller and hasattr(
                    self.gui_controller, "_cancel_requested"
                ):
                    if self.gui_controller._cancel_requested:
                        self._log_result("[CANCELLED] Optimization cancelled by user")
                        return np.inf

                try:
                    # Map parameters
                    aperture = self.config.aperture_range[0]  # default
                    energy = self.config.energy_range[0]  # default
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
                    rider_transv_dist = self.config.transv_dist  # default
                    macroparticle_charge_mult = (
                        self.config.macroparticle_charge_multiplier
                    )  # default
                    macroparticle_sigma_mult = (
                        self.config.macroparticle_sigma_multiplier
                    )  # default
                    wall_z = self.config.wall_z  # default

                    for i, param_name in enumerate(param_names):
                        if param_name == "aperture_radius":
                            aperture = x[i]
                        elif param_name == "initial_energy_gev":
                            energy = x[i]
                        elif param_name == "start_z":
                            start_z = x[i]
                        elif param_name == "transverse_offset":
                            offset_frac = x[i]
                        elif param_name == "timestep":
                            timestep = x[i]
                        elif param_name == "rider_transv_dist":
                            rider_transv_dist = x[i]
                        elif param_name == "macroparticle_charge_multiplier":
                            macroparticle_charge_mult = x[i]
                        elif param_name == "macroparticle_sigma_multiplier":
                            macroparticle_sigma_mult = x[i]
                        elif param_name == "wall_z":
                            wall_z = x[i]

                    # Calculate transverse offset in mm from fraction
                    transv_offset = offset_frac * aperture

                    # Calculate timestep if using auto_distance strategy
                    if self.config.timestep_strategy == "auto_distance":
                        timestep = self.config.calculate_timestep_for_energy(
                            energy,
                            self.config.m_particle,
                            wall_z=wall_z,
                            start_z=start_z,
                        )
                        steps = self.config.steps

                    # Run integration with timeout if enabled
                    result = None
                    timed_out = False

                    if self.config.per_run_timeout > 0:
                        import threading

                        result_container = [None]
                        error_container = [None]
                        cancel_flag = [False]

                        def run_integration():
                            try:
                                result_container[0] = self._run_single_integration(
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
                                    rider_transv_dist=rider_transv_dist,
                                    macroparticle_charge_multiplier=macroparticle_charge_mult,
                                    macroparticle_sigma_multiplier=macroparticle_sigma_mult,
                                    driver_params=None,
                                    wall_z=wall_z,
                                    run_num=eval_num,
                                    cancel_flag=cancel_flag,
                                )
                            except Exception as e:
                                error_container[0] = e

                        thread = threading.Thread(target=run_integration)
                        thread.daemon = True
                        thread.start()
                        thread.join(timeout=self.config.per_run_timeout)

                        if thread.is_alive():
                            timed_out = True
                            cancel_flag[0] = True
                            self._log_result(
                                f"[WARNING] Evaluation timed out for params {x} after {self.config.per_run_timeout}s"
                            )
                            self._log_result(
                                f"[WARNING] Signaling integration to cancel..."
                            )
                            # Give it a brief moment to respond
                            thread.join(timeout=2.0)
                            return np.inf
                        elif error_container[0] is not None:
                            raise error_container[0]
                        else:
                            result = result_container[0]
                    else:
                        # No timeout - run directly
                        result = self._run_single_integration(
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
                            rider_transv_dist=rider_transv_dist,
                            macroparticle_charge_multiplier=macroparticle_charge_mult,
                            macroparticle_sigma_multiplier=macroparticle_sigma_mult,
                            driver_params=None,
                            wall_z=wall_z,
                            run_num=eval_num,
                            cancel_flag=None,
                        )

                    if result is None or "metrics" not in result:
                        # Store failed evaluation
                        eval_record = {
                            "evaluation": eval_num,
                            "parameters": dict(zip(param_names, x)),
                            "failed": True,
                            "halted_early": (
                                result.get("halted_early", False) if result else False
                            ),
                            "halt_reason": (
                                result.get("halt_reason", None) if result else None
                            ),
                            "objective_value": float("inf"),
                        }
                        all_evaluations.append(eval_record)
                        return np.inf

                    # Check if run was halted early
                    if result.get("halted_early", False):
                        self._log_result(
                            f"[INFO] Evaluation {eval_num} halted early: {result.get('halt_reason', 'unknown')}"
                        )
                        self._log_result(
                            f"[INFO] Returning inf (rejecting halted evaluation)"
                        )
                        # Store halted evaluation
                        eval_record = {
                            "evaluation": eval_num,
                            "parameters": dict(zip(param_names, x)),
                            "failed": False,
                            "halted_early": True,
                            "halt_reason": result.get("halt_reason"),
                            "objective_value": float("inf"),
                        }
                        all_evaluations.append(eval_record)
                        return np.inf

                    # Extract metric value
                    metrics = result["metrics"]
                    value = metrics.get(metric_name, np.nan)

                    if np.isnan(value) or np.isinf(value):
                        self._log_result(
                            f"[WARNING] Evaluation {eval_num} returned {'NaN' if np.isnan(value) else 'inf'} for metric '{metric_name}'"
                        )
                        self._log_result(
                            f"[WARNING] Available metrics: {list(metrics.keys())}"
                        )
                        if len(metrics) > 0:
                            self._log_result(f"[WARNING] Metric values:")
                            for k, v in metrics.items():
                                self._log_result(f"  {k}: {v}")
                        self._log_result(
                            f"[WARNING] Returning inf (rejecting this evaluation)"
                        )
                        # Store failed evaluation
                        eval_record = {
                            "evaluation": eval_num,
                            "parameters": dict(zip(param_names, x)),
                            "failed": True,
                            "objective_value": float("inf"),
                            "metrics": result.get("metrics", {}),
                        }
                        all_evaluations.append(eval_record)
                        return np.inf

                    penalty = self._compute_soft_penalty(
                        aperture_radius=aperture,
                        macroparticle_charge_multiplier=macroparticle_charge_mult,
                        initial_energy_gev=energy,
                    )

                    adjusted_value = value
                    if penalty > 0:
                        if maximize:
                            adjusted_value = value - penalty
                        else:
                            adjusted_value = value + penalty
                        self._log_result(
                            "[INFO] Applied soft penalty of "
                            f"{penalty:.3e} to {self.config.objective} (risk-prone parameters)"
                        )

                    # Return value to minimize (negate if maximizing)
                    result_value = -adjusted_value if maximize else adjusted_value

                    # Store successful evaluation
                    eval_record = {
                        "evaluation": eval_num,
                        "parameters": dict(zip(param_names, x)),
                        "objective_value": adjusted_value,
                        "raw_objective_value": value,
                        "soft_penalty": penalty,
                        "fitness": result_value,  # Store fitness (for minimization)
                        "failed": False,
                        "halted_early": False,
                        "metrics": result.get("metrics", {}),
                    }

                    # Save trajectory if requested and available
                    if self.config.save_all_trajectories and "trajectory" in result:
                        # We'll save these after optimization dir is created
                        eval_record["trajectory"] = result["trajectory"]

                    all_evaluations.append(eval_record)

                    return result_value

                except Exception as e:
                    import traceback

                    self._log_result(
                        f"[ERROR] Evaluation {eval_num} failed for params {x}"
                    )
                    self._log_result(f"[ERROR] Exception: {type(e).__name__}: {e}")
                    self._log_result(f"[ERROR] Traceback:")
                    for line in traceback.format_exc().splitlines():
                        self._log_result(f"  {line}")

                    # Store failed evaluation
                    eval_record = {
                        "evaluation": eval_num,
                        "parameters": dict(zip(param_names, x)),
                        "failed": True,
                        "error": str(e),
                        "objective_value": float("inf"),
                    }
                    all_evaluations.append(eval_record)

                    return np.inf

            if method == "genetic_algorithm":
                # Define progress callback for convergence monitoring
                def log_convergence_progress(
                    generation,
                    best_value,
                    improvement,
                    tolerance,
                    patience_remaining,
                    converged,
                ):
                    """Log convergence progress after each generation."""
                    # Filter out inf values in logging
                    if np.isfinite(best_value):
                        self._log_result(
                            f"[OPTIMIZATION] Generation {generation}: best={best_value:.6e}, "
                            f"improvement={improvement:.6e}, tolerance={tolerance:.6e}"
                        )
                    else:
                        self._log_result(
                            f"[OPTIMIZATION] Generation {generation}: best=inf (no valid solutions yet), "
                            f"improvement={improvement:.6e}, tolerance={tolerance:.6e}"
                        )
                    if generation >= self.config.optimization_convergence_patience:
                        if converged:
                            self._log_result(
                                f"[CONVERGENCE] Converged! Improvement ({improvement:.6e}) "
                                f"< tolerance ({tolerance:.6e})"
                            )
                        else:
                            self._log_result(
                                f"[CONVERGENCE] Progress: {patience_remaining} generations "
                                f"remaining before early stop check"
                            )

                result = genetic_algorithm(
                    config_template=config_template,
                    parameter_names=param_names,
                    parameter_bounds=param_bounds,
                    metric_name=metric_name,
                    maximize=maximize,
                    population_size=self.config.optimization_population_size,
                    n_generations=self.config.optimization_maxiter,
                    mutation_rate=self.config.optimization_mutation_rate,
                    crossover_rate=self.config.optimization_crossover_rate,
                    seed=self.config.seed,
                    objective_function=evaluate_params,
                    convergence_tol=self.config.optimization_convergence_tol,
                    convergence_patience=self.config.optimization_convergence_patience,
                    progress_callback=log_convergence_progress,
                )

            elif method == "differential_evolution":
                result = optimize_parameters(
                    config_template=config_template,
                    parameter_names=param_names,
                    parameter_bounds=param_bounds,
                    metric_name=metric_name,
                    method="differential_evolution",
                    maximize=maximize,
                    maxiter=self.config.optimization_maxiter,
                    popsize=self.config.optimization_population_size,
                    objective_function=evaluate_params,
                    progress_callback=log_convergence_progress,
                )

            elif method == "nelder_mead":
                result = optimize_parameters(
                    config_template=config_template,
                    parameter_names=param_names,
                    parameter_bounds=param_bounds,
                    metric_name=metric_name,
                    method="nelder_mead",
                    maximize=maximize,
                    maxiter=self.config.optimization_maxiter,
                    objective_function=evaluate_params,
                    progress_callback=log_convergence_progress,
                )

            elif method == "multi_start":
                result = multi_start_optimize(
                    config_template=config_template,
                    parameter_names=param_names,
                    parameter_bounds=param_bounds,
                    metric_name=metric_name,
                    n_starts=self.config.optimization_n_starts,
                    maximize=maximize,
                    maxiter=self.config.optimization_maxiter,
                    objective_function=evaluate_params,
                    progress_callback=log_convergence_progress,
                )

            elif method == "adaptive_grid":
                best_params, best_value, history = adaptive_grid_search(
                    config_template=config_template,
                    parameter_names=param_names,
                    parameter_bounds=param_bounds,
                    metric_name=metric_name,
                    maximize=maximize,
                    initial_points_per_dim=5,
                    refinement_levels=2,
                    objective_function=evaluate_params,
                    progress_callback=log_convergence_progress,
                )
                # Convert to OptimizeResult format
                from scipy.optimize import OptimizeResult

                result = OptimizeResult()
                result.x = best_params
                result.fun = -best_value if maximize else best_value
                result.best_params_dict = dict(zip(param_names, best_params))
                result.success = True

            if result is None:
                self._log_result(f"[ERROR] Unknown optimization method: {method}")
                self.running = False
                return

            # Cache all evaluations for saving with results
            self._all_evaluations_cache = all_evaluations

            # Log results
            self._log_result("")
            self._log_result("=" * 80)
            self._log_result("OPTIMIZATION COMPLETE")
            self._log_result("=" * 80)
            self._log_result(f"Best {metric_name}: {result.fun:.12e}")
            self._log_result("Best parameters:")
            for param_name, value in result.best_params_dict.items():
                self._log_result(f"  {param_name}: {value:.12e}")
            self._log_result("")
            self._log_result(
                f"Function evaluations: {result.nfev if hasattr(result, 'nfev') else 'N/A'}"
            )
            self._log_result("")

            # Save results (this sets self._last_optimization_dir)
            self._save_optimization_results(result, param_names)

            # Re-run top N parameters to generate and save trajectories (only if enabled)
            if self.config.save_top_n_trajectories:
                self._save_top_n_optimization_trajectories(result, param_names)
            else:
                self._log_result("")
                self._log_result(
                    "[INFO] Top N trajectory saving disabled (save_top_n_trajectories=False)"
                )

            # Cache all evaluations for saving and generate heatmap
            if len(all_evaluations) > 0:
                self._all_evaluations_cache = all_evaluations
                self._generate_optimization_heatmap(
                    all_evaluations, param_names, self._last_optimization_dir
                )

            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = elapsed_time % 60

            self._log_result("[OK] Optimization complete!")
            if hours > 0:
                self._log_result(
                    f"  Total time: {hours}h {minutes}m {seconds:.1f}s ({elapsed_time:.1f}s)"
                )
            elif minutes > 0:
                self._log_result(
                    f"  Total time: {minutes}m {seconds:.1f}s ({elapsed_time:.1f}s)"
                )
            else:
                self._log_result(f"  Total time: {elapsed_time:.1f}s")

        except KeyboardInterrupt:
            self._log_result("")
            self._log_result("[CANCELLED] Optimization cancelled by user")
            self._log_result("")
            # Try to save partial results if we have any evaluations
            if "all_evaluations" in locals() and len(all_evaluations) > 0:
                self._log_result(
                    f"[INFO] Saving partial results ({len(all_evaluations)} evaluations completed)..."
                )
                try:
                    self._save_partial_optimization_results(
                        all_evaluations, param_names, "CANCELLED"
                    )
                except Exception as save_err:
                    self._log_result(
                        f"[WARNING] Failed to save partial results: {save_err}"
                    )
        except Exception as e:
            import traceback

            error_msg = f"Optimization failed: {e}\n{traceback.format_exc()}"
            self._log_result(f"[ERROR] {error_msg}")
            # Try to save partial results even on error
            if "all_evaluations" in locals() and len(all_evaluations) > 0:
                self._log_result(
                    f"[INFO] Saving partial results ({len(all_evaluations)} evaluations completed)..."
                )
                try:
                    self._save_partial_optimization_results(
                        all_evaluations, param_names, "FAILED"
                    )
                except Exception as save_err:
                    self._log_result(
                        f"[WARNING] Failed to save partial results: {save_err}"
                    )
        finally:
            # Restore original verbosity settings
            if "original_sc_verbosity" in locals():
                self.config.self_consistency_verbosity = original_sc_verbosity
            if "original_adaptive_debug" in locals():
                self.config.adaptive_timestep_debug = original_adaptive_debug

            self.running = False
            self._update_progress(100, "Done")
            # Ensure log file is closed
            if self._log_file is not None:
                self._close_log_file()

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

    def _run_sweep_background(self, is_finetune=False, finetune_regions=None):
        """Run parameter sweep in background with real integration.

        Args:
            is_finetune: If True, this is a fine-tuning sweep
            finetune_regions: List of parameter regions for fine-tuning
        """
        # Set logging context for this sweep run
        context = "sweep_finetune" if is_finetune else "sweep"
        set_logging_context(context)

        # Open log file in temporary location (will be moved when results are saved)
        import tempfile
        import time

        temp_dir = tempfile.mkdtemp(prefix="sweep_log_")
        self._open_log_file(temp_dir)

        start_time = time.time()

        try:
            # Check mode and route accordingly
            if self.config.mode == "optimization":
                self._run_optimization_background()
                return

            # Generate parameter grid including sweepable parameters
            param_grids = self._generate_parameter_grids()

            # Calculate total runs
            total_runs = 1
            for values in param_grids.values():
                total_runs *= len(values)

            # Determine verbosity level from config
            use_no_logging = self.config.log_verbosity == "none"
            use_truncated_logging = self.config.log_verbosity == "truncated"
            use_full_debug = self.config.log_verbosity == "full"

            # Override config verbosity settings based on log mode
            # Save original values to restore later
            original_sc_verbosity = self.config.self_consistency_verbosity
            original_adaptive_debug = self.config.adaptive_timestep_debug

            if use_no_logging or use_truncated_logging:
                # Suppress SC iteration output and adaptive timestep refinement output
                self.config.self_consistency_verbosity = 0
                self.config.adaptive_timestep_debug = False
            # else: full debug mode - INHERIT stability settings from config/GUI (don't override)

            self._log_result(
                f"Starting BLIND SWEEP (Grid Search): {total_runs} total runs"
            )
            self._log_result(f"Log verbosity: {self.config.log_verbosity}")

            # Log inherited stability settings in full debug mode
            if use_full_debug:
                self._log_result(
                    "  Full debug logging enabled (inherits Stability tab settings)"
                )
                self._log_result(
                    f"    SC verbosity: {self.config.self_consistency_verbosity}"
                )
                self._log_result(
                    f"    Adaptive timestep debug: {self.config.adaptive_timestep_debug}"
                )

            # Only log detailed config in full debug mode
            if use_full_debug:
                self._log_result(
                    f"Trajectory saving: Top N={self.config.save_top_n_trajectories}, All={self.config.save_all_trajectories}, Failed={self.config.save_failed_trajectories}"
                )

                # Log parameter grid info
                for param_name, values in param_grids.items():
                    if len(values) > 1:
                        self._log_result(
                            f"  {param_name}: {len(values)} points from {values[0]:.2e} to {values[-1]:.2e}"
                        )
                    else:
                        if param_name == "wall_z":
                            self._log_result(
                                f"  {param_name}: {values[0]:.2f} mm (fixed)"
                            )
                        else:
                            self._log_result(f"  {param_name}: {values[0]:.2e} (fixed)")
                self._log_result(
                    f"  Timestep strategy: {self.config.timestep_strategy}"
                )
                if self.config.timestep_strategy == "energy_scaled":
                    self._log_result(
                        f"    Energy scale exponent: {self.config.energy_scale_exponent} (h ∝ γ^-α)"
                    )
                elif self.config.timestep_strategy == "auto_distance":
                    self._log_result(
                        f"    Target distance: {self.config.target_distance_mm:.1f} mm (wall_z + target)"
                    )
                    self._log_result(
                        f"    All particles will travel to consistent z regardless of energy"
                    )
                elif self.config.auto_steps:
                    self._log_result(
                        f"    Legacy auto_steps: wall_z + {self.config.auto_steps_distance_past_wall:.1f} mm"
                    )
                self._log_result(f"  z_cutoff_mode: {self.config.z_cutoff_mode}")

            self._log_result("")

            # Use sweep output directory from GUI preferences
            self.config.output_dir = self.sweep_output_dir

            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            self._log_result(f"Output directory: {self.config.output_dir}")
            self._log_result("")

            # Store all results and failed runs
            all_results = []
            failed_runs = []
            run_num = 0

            # Create parameter combinations using itertools
            import itertools

            param_names = list(param_grids.keys())
            param_values_lists = [param_grids[name] for name in param_names]

            for param_combo in itertools.product(*param_values_lists):
                # Periodic cleanup of matplotlib figures to prevent memory leak
                if run_num > 0 and run_num % 10 == 0:
                    import matplotlib.pyplot as plt

                    plt.close("all")

                # Check for cancellation
                if not self.running:
                    self._log_result("[STOPPED] Sweep stopped by user")
                    break

                if self.gui_controller and hasattr(
                    self.gui_controller, "_cancel_requested"
                ):
                    if self.gui_controller._cancel_requested:
                        self._log_result("[CANCELLED] Sweep cancelled by user")
                        break

                run_num += 1
                progress = run_num / total_runs * 100
                self._update_progress(
                    progress,
                    f"Running simulation {run_num}/{total_runs}...",
                )

                # Extract parameters from combination
                params_dict = dict(zip(param_names, param_combo))

                aperture = params_dict["aperture"]
                energy = params_dict["energy"]
                start_z = params_dict["start_z"]
                offset_frac = params_dict["transverse_offset_fraction"]

                # Get rider particle parameters (either from sweep or fixed values)
                rider_m_particle = params_dict.get(
                    "rider_m_particle", self.config.m_particle
                )
                rider_charge_sign = params_dict.get(
                    "rider_charge_sign", self.config.charge_sign
                )
                rider_pcount = params_dict.get("rider_pcount", self.config.pcount)
                rider_transv_mom = params_dict.get(
                    "rider_transv_mom", self.config.transv_mom
                )
                rider_transv_dist = params_dict.get(
                    "rider_transv_dist", self.config.transv_dist
                )

                # Get macroparticle parameters (either from sweep or fixed values)
                macroparticle_charge_multiplier = params_dict.get(
                    "macroparticle_charge_multiplier",
                    self.config.macroparticle_charge_multiplier,
                )
                macroparticle_sigma_multiplier = params_dict.get(
                    "macroparticle_sigma_multiplier",
                    self.config.macroparticle_sigma_multiplier,
                )

                # Log parameter values based on verbosity
                if use_full_debug:
                    # Log ALL swept parameter values for this run
                    self._log_result(
                        f"  [PARAMS] Run {run_num}/{total_runs} - All parameters:"
                    )
                    self._log_result(f"    aperture: {aperture:.4e} mm")
                    self._log_result(f"    energy: {energy:.4f} GeV")
                    self._log_result(f"    start_z: {start_z:.4f} mm")
                    self._log_result(f"    transv_offset_frac: {offset_frac:.4f}")
                    self._log_result(
                        f"    rider_m_particle: {rider_m_particle:.4e} amu"
                    )
                    self._log_result(f"    rider_charge_sign: {rider_charge_sign:.1f}")
                    self._log_result(f"    rider_pcount: {rider_pcount}")
                    self._log_result(
                        f"    rider_transv_mom: {rider_transv_mom:.4e} amu·mm/ns"
                    )
                    self._log_result(
                        f"    rider_transv_dist: {rider_transv_dist:.4e} mm"
                    )
                    if self.config.macroparticle_enabled:
                        self._log_result(f"    macroparticle_enabled: True")
                        self._log_result(
                            f"    macroparticle_charge_multiplier: {macroparticle_charge_multiplier:.4f}"
                        )
                        self._log_result(
                            f"    macroparticle_sigma_multiplier: {macroparticle_sigma_multiplier:.4f}"
                        )
                        self._log_result(
                            f"    macroparticle_use_momentum_errors: {self.config.macroparticle_use_momentum_errors}"
                        )

                # Get driver particle parameters if BUNCH_TO_BUNCH
                driver_params_dict = None
                if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
                    driver_params_dict = {
                        "m_particle": params_dict.get("driver_m_particle", 207.2),
                        "charge_sign": params_dict.get("driver_charge_sign", 1.0),
                        "pcount": int(params_dict.get("driver_pcount", 5)),
                        "transv_mom": params_dict.get("driver_transv_mom", 0.0),
                        "transv_dist": params_dict.get("driver_transv_dist", -0.07998),
                        "starting_distance": params_dict.get(
                            "driver_starting_distance", 1000.0
                        ),
                        "starting_Pz": params_dict.get("driver_starting_Pz", -4925.0),
                        "stripped_ions": float(self.driver_stripped_ions_var.get()),
                    }

                # Calculate transverse offset
                transv_offset = offset_frac * aperture

                # Calculate timestep based on strategy
                if self.config.timestep_strategy != "fixed":
                    # Use energy-aware timestep calculation
                    # Get wall_z for this run (it may be swept)
                    wall_z_for_calc = params_dict.get("wall_z", self.config.wall_z)
                    timestep = self.config.calculate_timestep_for_energy(
                        energy,
                        rider_m_particle,
                        wall_z=wall_z_for_calc,
                        start_z=start_z,
                    )
                    steps = self.config.steps

                    # Calculate gamma for diagnostics (ALWAYS log for debugging)
                    AMU_TO_MEV = 931.494
                    rest_energy_mev = rider_m_particle * AMU_TO_MEV
                    gamma = (energy * 1e3) / rest_energy_mev
                    beta = (
                        np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.999
                    )
                    distance_per_step = beta * gamma * C_MMNS * timestep
                    expected_distance = distance_per_step * steps

                    if use_full_debug:
                        self._log_result(
                            f"  [TIMESTEP] Run {run_num} strategy '{self.config.timestep_strategy}':"
                        )
                        self._log_result(
                            f"    E={energy:.4f} GeV, m={rider_m_particle:.4e} amu"
                        )
                        self._log_result(f"    gamma={gamma:.2f}, beta={beta:.8f}")
                        self._log_result(
                            f"    timestep h={timestep:.4e} ns (proper time = dt/gamma)"
                        )
                        self._log_result(f"    steps={steps}")
                        self._log_result(
                            f"    distance_per_step = β·γ·c·h = {distance_per_step:.4f} mm"
                        )
                        self._log_result(
                            f"    expected_total_distance = {expected_distance:.2f} mm"
                        )
                        # Use wall_z from grid if available, otherwise use config default
                        current_wall_z = params_dict.get("wall_z", self.config.wall_z)
                        self._log_result(
                            f"    wall_z={current_wall_z:.2f} mm, start_z={start_z:.2f} mm"
                        )
                        self._log_result(
                            f"    distance_to_wall = {abs(current_wall_z - start_z):.2f} mm"
                        )
                        if self.config.timestep_strategy == "auto_distance":
                            self._log_result(
                                f"    target_distance={self.config.target_distance_mm:.2f} mm"
                            )
                elif self.config.auto_steps:
                    # Legacy auto_steps mode (deprecated, but keep for compatibility)
                    current_wall_z = params_dict.get("wall_z", self.config.wall_z)
                    distance_to_wall = abs(current_wall_z - start_z)
                    total_distance = (
                        distance_to_wall + self.config.auto_steps_distance_past_wall
                    )

                    timestep = calculate_auto_timestep(
                        start_z=start_z,
                        wall_z=current_wall_z,
                        distance_past_wall=self.config.auto_steps_distance_past_wall,
                        particle_energy_gev=energy,
                        particle_mass_amu=rider_m_particle,
                        target_steps=self.config.auto_steps_target,
                    )
                    steps = calculate_auto_steps(
                        start_z=start_z,
                        wall_z=current_wall_z,
                        distance_past_wall=self.config.auto_steps_distance_past_wall,
                        timestep=timestep,
                        particle_energy_gev=energy,
                        particle_mass_amu=rider_m_particle,
                    )
                else:
                    timestep = self.config.timestep
                    steps = self.config.steps

                # Enforce minimum of 5% of requested steps (absolute floor of 20)
                min_steps = max(20, int(self.config.steps * 0.05))
                if steps < min_steps:
                    if use_full_debug:
                        self._log_result(
                            f"  [WARNING] Steps adjusted from {steps} to {min_steps} (minimum floor)"
                        )
                    steps = min_steps

                # Log run start summary (only in full debug mode - truncated mode logs after completion)
                if use_full_debug:
                    self._log_result(
                        f"  [START] Run {run_num}/{total_runs}: "
                        f"a={aperture:.4e}mm, E={energy:.4f}GeV, z={start_z:.2f}mm, "
                        f"h={timestep:.4e}ns, N={steps}"
                    )

                # Run integration with timeout
                result = None
                run_error = None
                run_timed_out = False

                try:
                    # Check if timeout is enabled
                    if self.config.per_run_timeout > 0:
                        import threading

                        # Container for result (mutable for thread access)
                        result_container = [None]
                        error_container = [None]
                        cancel_flag = [False]  # Flag to signal cancellation

                        # Log warning for potentially problematic parameter combinations
                        if aperture < 0.1 and macroparticle_charge_multiplier > 1000:
                            self._log_result(
                                f"  [WARNING] Run {run_num}: Very small aperture ({aperture:.4f} mm) "
                                f"with large charge multiplier ({macroparticle_charge_multiplier:.0f})"
                            )
                            self._log_result(
                                f"    This may cause numerical instability or slow convergence"
                            )

                        def run_with_exception_handling():
                            """Wrapper to run integration and catch exceptions."""
                            try:
                                result_container[0] = self._run_single_integration(
                                    aperture=aperture,
                                    energy_gev=energy,
                                    start_z=start_z,
                                    transv_offset=transv_offset,
                                    timestep=timestep,
                                    steps=steps,
                                    rider_m_particle=rider_m_particle,
                                    rider_charge_sign=rider_charge_sign,
                                    rider_pcount=int(rider_pcount),
                                    rider_transv_mom=rider_transv_mom,
                                    macroparticle_charge_multiplier=macroparticle_charge_multiplier,
                                    macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
                                    driver_params=(
                                        driver_params_dict
                                        if self.config.simulation_type
                                        == SimulationType.BUNCH_TO_BUNCH
                                        else None
                                    ),
                                    wall_z=params_dict.get(
                                        "wall_z", self.config.wall_z
                                    ),
                                    run_num=run_num,
                                    cancel_flag=cancel_flag,
                                )
                            except Exception as e:
                                error_container[0] = e

                        # Start integration in separate thread
                        integration_thread = threading.Thread(
                            target=run_with_exception_handling
                        )
                        integration_thread.daemon = True
                        integration_thread.start()

                        # Wait for completion or timeout
                        integration_thread.join(timeout=self.config.per_run_timeout)

                        if integration_thread.is_alive():
                            # Timeout occurred - signal the integration to cancel
                            run_timed_out = True
                            cancel_flag[0] = True
                            self._log_result(
                                f"  [TIMEOUT] Run {run_num} exceeded timeout of {self.config.per_run_timeout}s"
                            )
                            self._log_result(
                                f"    Signaling integration to cancel (thread will terminate when it checks cancel flag)"
                            )
                            # Give it a brief moment to respond to cancellation
                            integration_thread.join(timeout=2.0)
                            if integration_thread.is_alive():
                                self._log_result(
                                    f"    Warning: Integration thread still running after cancel signal"
                                )
                                self._log_result(
                                    f"    Thread will be abandoned (daemon thread will terminate with main thread)"
                                )
                            elif error_container[0] is not None:
                                # Exception occurred in thread
                                raise error_container[0]
                        else:
                            # Success
                            result = result_container[0]
                    else:
                        # No timeout - run directly
                        result = self._run_single_integration(
                            aperture=aperture,
                            energy_gev=energy,
                            start_z=start_z,
                            transv_offset=transv_offset,
                            timestep=timestep,
                            steps=steps,
                            rider_m_particle=rider_m_particle,
                            rider_charge_sign=rider_charge_sign,
                            rider_pcount=int(rider_pcount),
                            rider_transv_mom=rider_transv_mom,
                            macroparticle_charge_multiplier=macroparticle_charge_multiplier,
                            macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
                            driver_params=(
                                driver_params_dict
                                if self.config.simulation_type
                                == SimulationType.BUNCH_TO_BUNCH
                                else None
                            ),
                            run_num=run_num,
                            cancel_flag=None,
                        )

                    if result is not None and use_full_debug:
                        self._log_result(
                            f"  [DEBUG] Run {run_num} integration completed"
                        )

                    if not run_timed_out and result is not None:
                        # Extract metrics
                        delta_e = result.get("metrics", {}).get(
                            "rider_delta_e_mev", 0.0
                        )
                        delta_gamma = result.get("metrics", {}).get(
                            "rider_delta_gamma", 0.0
                        )
                        gamma_initial = result.get("metrics", {}).get(
                            "rider_gamma_initial", 0.0
                        )
                        gamma_final = result.get("metrics", {}).get(
                            "rider_gamma_final", 0.0
                        )

                        # Create run_data structure (used regardless of logging mode)
                        run_data = {
                            "run_number": run_num,
                            "parameters": {
                                "aperture_radius": aperture,
                                "particle_energy_gev": energy,
                                "start_z": start_z,
                                "transverse_offset": transv_offset,
                                "transverse_offset_fraction": offset_frac,
                                "timestep": timestep,
                                "steps": steps,
                                "wall_z": params_dict.get("wall_z", self.config.wall_z),
                                "rider_m_particle": rider_m_particle,
                                "rider_charge_sign": rider_charge_sign,
                                "rider_pcount": int(rider_pcount),
                                "rider_transv_mom": rider_transv_mom,
                                "rider_transv_dist": rider_transv_dist,
                                "macroparticle_charge_multiplier": macroparticle_charge_multiplier,
                                "macroparticle_sigma_multiplier": macroparticle_sigma_multiplier,
                                "simulation_type": self.config.simulation_type.name,
                            },
                            "metrics": result.get("metrics", {}),
                        }

                        # Log based on verbosity mode
                        if use_no_logging:
                            # No logging mode: skip all run-level logs
                            pass
                        elif use_truncated_logging:
                            # Truncated mode: 1-2 lines with key info
                            self._log_truncated_run(
                                run_num,
                                params={
                                    "aperture": aperture,
                                    "energy": energy,
                                    "wall_z": params_dict.get(
                                        "wall_z", self.config.wall_z
                                    ),
                                },
                                metrics={
                                    "ΔE": delta_e,
                                    "Δγ": delta_gamma,
                                    "γ_i": gamma_initial,
                                    "γ_f": gamma_final,
                                },
                            )
                        elif use_full_debug:
                            # Full debug mode: all details
                            # Extract actual trajectory distance for diagnostics
                            actual_distance = 0.0
                            if "_distance_info" in result:
                                dist_info = result["_distance_info"]
                                actual_distance = abs(
                                    dist_info["z_end"] - dist_info["z_start"]
                                )
                            elif "trajectory" in result and result["trajectory"]:
                                # Fallback: try to extract from full trajectory if present
                                traj = result["trajectory"]
                                z_vals = traj.get("z", [])
                                if len(z_vals) > 1:
                                    # Safely handle both lists and numpy arrays
                                    z_start = float(np.asarray(z_vals[0]).flat[0])
                                    z_end = float(np.asarray(z_vals[-1]).flat[0])
                                    actual_distance = abs(z_end - z_start)

                            self._log_result(f"  [RESULT] Run {run_num}/{total_runs}:")
                            self._log_result(
                                f"    Distance: expected={expected_distance:.2f}mm, actual={actual_distance:.2f}mm"
                            )
                            self._log_result(
                                f"    Gamma: initial={gamma_initial:.6f}, final={gamma_final:.6f}, delta={delta_gamma:.6e}"
                            )
                            self._log_result(f"    Energy: ΔE={delta_e:.6f}MeV")
                            if actual_distance < 0.1:
                                self._log_result(
                                    f"  [WARNING] Particle barely moved! Check timestep calculation."
                                )

                        # Add trajectory if requested (check if any trajectory saving is enabled)
                        # Note: save_top_n_trajectories only applies to optimization mode, not sweeps
                        save_traj = (
                            self.config.save_all_trajectories
                            or self.config.save_failed_trajectories
                        )
                        if save_traj and "trajectory" in result:
                            run_data["trajectory"] = result["trajectory"]

                        # Add driver params to stored results if applicable
                        if driver_params_dict is not None:
                            run_data["parameters"].update(
                                {
                                    f"driver_{k}": v
                                    for k, v in driver_params_dict.items()
                                }
                            )

                        all_results.append(run_data)

                except Exception as e:
                    import traceback

                    error_details = traceback.format_exc()
                    run_error = str(e)

                    if self.config.skip_failed_runs:
                        self._log_result(f"[WARNING] Run {run_num} failed: {e}")
                        self._log_result(f"    Error details: {error_details}")
                        self._log_result(
                            f"    Skipping and continuing with next run..."
                        )

                        # Record failed run
                        failed_runs.append(
                            {
                                "run_number": run_num,
                                "parameters": {
                                    "aperture_radius": aperture,
                                    "particle_energy_gev": energy,
                                    "start_z": start_z,
                                    "transverse_offset": transv_offset,
                                    "timestep": timestep,
                                    "steps": steps,
                                    "wall_z": params_dict.get(
                                        "wall_z", self.config.wall_z
                                    ),
                                },
                                "error": run_error,
                                "error_details": error_details,
                            }
                        )
                    else:
                        # Don't skip - re-raise and stop sweep
                        self._log_result(f"[ERROR] Run {run_num} failed: {e}")
                        self._log_result(f"    Error details: {error_details}")
                        self._log_result(
                            f"    Stopping sweep (skip_failed_runs is disabled)"
                        )
                        raise

                # Handle timeout case
                if run_timed_out:
                    if self.config.skip_failed_runs:
                        self._log_result(
                            f"    Skipping and continuing with next run..."
                        )
                        failed_runs.append(
                            {
                                "run_number": run_num,
                                "parameters": {
                                    "aperture_radius": aperture,
                                    "particle_energy_gev": energy,
                                    "start_z": start_z,
                                    "transverse_offset": transv_offset,
                                    "timestep": timestep,
                                    "steps": steps,
                                },
                                "error": "TIMEOUT",
                                "timeout_seconds": self.config.per_run_timeout,
                            }
                        )
                    else:
                        self._log_result(
                            f"    Stopping sweep (skip_failed_runs is disabled)"
                        )
                        break

            # Save results
            if all_results and self.config.save_results:
                self._save_sweep_results(all_results, failed_runs)

            if self.running:
                elapsed_time = time.time() - start_time
                hours = int(elapsed_time // 3600)
                minutes = int((elapsed_time % 3600) // 60)
                seconds = elapsed_time % 60

                self._log_result("[OK] Sweep completed!")
                self._log_result(f"  Results saved to: {self.config.output_dir}")
                self._log_result(f"  Successful runs: {len(all_results)}")
                if failed_runs:
                    self._log_result(f"  Failed/timed-out runs: {len(failed_runs)}")
                if hours > 0:
                    self._log_result(
                        f"  Total time: {hours}h {minutes}m {seconds:.1f}s ({elapsed_time:.1f}s)"
                    )
                elif minutes > 0:
                    self._log_result(
                        f"  Total time: {minutes}m {seconds:.1f}s ({elapsed_time:.1f}s)"
                    )
                else:
                    self._log_result(f"  Total time: {elapsed_time:.1f}s")
                self._update_progress(100, "Complete!")
        except Exception as e:
            self._log_result(f"[ERROR] Error during sweep: {e}")
            import traceback

            self._log_result(traceback.format_exc())
        finally:
            # Restore original verbosity settings
            if "original_sc_verbosity" in locals():
                self.config.self_consistency_verbosity = original_sc_verbosity
            if "original_adaptive_debug" in locals():
                self.config.adaptive_timestep_debug = original_adaptive_debug

            self.running = False
            # Ensure log file is closed
            if self._log_file is not None:
                self._close_log_file()
            # Clean up any remaining matplotlib figures
            import matplotlib.pyplot as plt

            plt.close("all")
            # Update UI back to ready state
            self.after(100, self._reset_ui_state)

    def _generate_parameter_grids(self):
        """Generate all parameter grids including sweepable parameters."""
        grids = {}

        # Always swept: aperture and energy
        grids["aperture"] = self._generate_range(
            self.config.aperture_range[0],
            self.config.aperture_range[1],
            self.config.aperture_points,
            self.config.aperture_log_scale,
        )
        grids["energy"] = self._generate_range(
            self.config.energy_range[0],
            self.config.energy_range[1],
            self.config.energy_points,
            self.config.energy_log_scale,
        )

        # Lists (always swept if multiple values)
        grids["transverse_offset_fraction"] = self.config.transverse_offset_fractions
        grids["start_z"] = self.config.starting_z_positions

        # Wall z (optional sweep)
        if self.config.wall_z_range is not None and self.config.wall_z_points > 1:
            grids["wall_z"] = self._generate_range(
                self.config.wall_z_range[0],
                self.config.wall_z_range[1],
                self.config.wall_z_points,
                False,  # wall_z doesn't need log scale
            )

        # Optional sweeps for rider and driver particle parameters
        sim_type = self.config.simulation_type
        for param_name, controls in self.sweep_params.items():
            # Skip driver params if not BUNCH_TO_BUNCH
            if (
                param_name.startswith("driver_")
                and sim_type != SimulationType.BUNCH_TO_BUNCH
            ):
                continue

            if controls["sweep_var"].get():
                min_val = float(controls["min_var"].get())
                max_val = float(controls["max_var"].get())
                points = int(controls["points_var"].get())
                log_scale = controls["log_var"].get()
                grids[param_name] = self._generate_range(
                    min_val, max_val, points, log_scale
                )

        return grids

    def _generate_range(
        self, min_val: float, max_val: float, points: int, log_scale: bool
    ) -> List[float]:
        """Generate parameter range (linear or log scale)."""
        if points == 1:
            return [(min_val + max_val) / 2.0]
        if log_scale:
            return np.logspace(np.log10(min_val), np.log10(max_val), points).tolist()
        else:
            return np.linspace(min_val, max_val, points).tolist()

    def _run_single_integration(
        self,
        aperture: float,
        energy_gev: float,
        start_z: float,
        transv_offset: float,
        timestep: float,
        steps: int,
        rider_m_particle: float = None,
        rider_charge_sign: float = None,
        rider_pcount: int = None,
        rider_transv_mom: float = None,
        rider_transv_dist: float = None,
        macroparticle_charge_multiplier: float = None,
        macroparticle_sigma_multiplier: float = None,
        driver_params: Dict[str, Any] = None,
        wall_z: float = None,
        run_num: int = 0,
        cancel_flag: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """Run a single integration with given parameters."""
        # Log stability analysis configuration for debugging
        self._log_result(f"  [CONFIG] Run {run_num} stability settings:")
        self._log_result(f"    smoothness_enabled: {self.config.smoothness_enabled}")
        if self.config.smoothness_enabled:
            self._log_result(
                f"    smoothness_window_size: {self.config.smoothness_window_size}"
            )
            self._log_result(
                f"    smoothness_reject_on_violation: {self.config.smoothness_reject_on_violation}"
            )

        # Use provided rider values or fall back to config defaults
        rider_m_particle = (
            rider_m_particle if rider_m_particle is not None else self.config.m_particle
        )
        rider_charge_sign = (
            rider_charge_sign
            if rider_charge_sign is not None
            else self.config.charge_sign
        )
        rider_pcount = (
            rider_pcount if rider_pcount is not None else int(self.config.pcount)
        )
        rider_transv_mom = (
            rider_transv_mom if rider_transv_mom is not None else self.config.transv_mom
        )
        rider_transv_dist = (
            rider_transv_dist
            if rider_transv_dist is not None
            else self.config.transv_dist
        )
        wall_z = wall_z if wall_z is not None else self.config.wall_z
        macroparticle_charge_multiplier = (
            macroparticle_charge_multiplier
            if macroparticle_charge_multiplier is not None
            else self.config.macroparticle_charge_multiplier
        )
        macroparticle_sigma_multiplier = (
            macroparticle_sigma_multiplier
            if macroparticle_sigma_multiplier is not None
            else self.config.macroparticle_sigma_multiplier
        )

        # Build rider params
        # transv_offset is the radial offset from axis (in mm)
        # This is now properly used as an offset, not as spread
        rider_params = {
            "starting_distance": start_z,
            "transv_mom": rider_transv_mom,
            "transv_dist": rider_transv_dist,  # Use parameter, not config
            "transv_offset_x": transv_offset,  # Radial offset as x-offset
            "transv_offset_y": 0.0,  # Keep on x-axis (radial offset in x-direction)
            "m_particle": rider_m_particle,
            "charge_sign": rider_charge_sign,
            "pcount": rider_pcount,
            "stripped_ions": float(self.rider_stripped_ions_var.get()),
            "starting_Pz": 0.0,  # Will be calculated from energy
        }

        # Calculate initial Pz from energy
        # E = gamma * m * c^2, where m*c^2 in MeV
        AMU_TO_MEV = 931.494
        rest_energy_mev = rider_m_particle * AMU_TO_MEV
        gamma = (energy_gev * 1e3) / rest_energy_mev
        # Legacy init_bunch expects starting_Pz as specific momentum (momentum/mass)
        # It calculates: Pz = starting_Pz * mass, then γ = sqrt((Pz/(mc))² + 1)
        # Working backwards: γ² = (Pz/(mc))² + 1 = (starting_Pz/c)² + 1
        # Therefore: starting_Pz = c·sqrt(γ² - 1)
        rider_params["starting_Pz"] = C_MMNS * np.sqrt(gamma * gamma - 1.0)

        core_params = {
            "time_step": timestep,
            "wall_z": wall_z,
            "aperture_radius": aperture,
            "mean": 1.0e5,  # Large value (not used for CONDUCTING_WALL)
            "cav_spacing": 1.0e5,
            "z_cutoff": (
                self.config.target_distance_mm
                if self.config.z_cutoff_mode == "relative"
                else 0.0
            ),
            "z_cutoff_mode": self.config.z_cutoff_mode,
        }

        # Create a temporary subdirectory for this run's outputs (will be cleaned up)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        # Create a temporary subdirectory for this run's outputs (will be cleaned up)
        # IMPORTANT: This must live under the same base directory that the orphan-cleanup
        # routine scans (self.sweep_output_dir), otherwise temp dirs will only be cleaned
        # up when the GUI starts (or never, if output_dir differs).
        run_output_dir = (
            Path(self.sweep_output_dir) / f"_temp_run_{run_num}_{timestamp}"
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)

        options = SimulationOptions(
            steps=steps,
            seed=self.config.seed,
            simulation_type=self.config.simulation_type,
            rider_params=rider_params,
            driver_params=driver_params,  # Use provided driver params (None for CONDUCTING_WALL)
            core_params=core_params,
            legacy_enabled=False,
            trajectory_save=False,  # Don't save individual trajectory files to disk
            trajectory_interval=self.config.trajectory_stride,
            energy_display=False,  # Don't generate or display plots during sweep
            energy_save=False,
            transverse_display=False,
            transverse_save=True,  # Always return trajectory data for metrics calculation
            beta_display=False,  # Don't generate beta plots
            beta_save=False,
            momentum_display=False,  # Don't generate momentum plots
            momentum_save=False,
            gamma_display=False,  # Don't generate gamma plots
            gamma_save=False,
            zposition_display=False,  # Don't generate z-position plots
            zposition_save=False,
            macroparticle_enabled=self.config.macroparticle_enabled,
            macroparticle_charge_multiplier=macroparticle_charge_multiplier,
            macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
            macroparticle_use_momentum_errors=self.config.macroparticle_use_momentum_errors,
            overlay_display=False,
            overlay_save=False,
            difference_display=False,
            difference_save=False,
            metrics_save=False,
            output_dir=run_output_dir,
            # Use stability options from sweep config
            self_consistency_enabled=self.config.self_consistency_enabled,
            self_consistency_tolerance=self.config.self_consistency_tolerance,
            self_consistency_max_iterations=self.config.self_consistency_max_iterations,
            self_consistency_verbosity=self.config.self_consistency_verbosity,
            energy_monitor_enabled=False,  # Removed - functionality in adaptive timestep
            energy_monitor_threshold=2.0,
            energy_monitor_check_interval=10,
            energy_monitor_halt_on_jump=self.config.energy_monitor_halt_on_jump,
            energy_monitor_debug=False,  # Removed
            adaptive_timestep_enabled=self.config.adaptive_timestep_enabled,
            adaptive_timestep_threshold=self.config.adaptive_timestep_threshold,
            adaptive_timestep_reduction_factor=self.config.adaptive_timestep_reduction_factor,
            adaptive_timestep_max_attempts=self.config.adaptive_timestep_max_attempts,
            adaptive_timestep_min_factor=self.config.adaptive_timestep_min_factor,
            adaptive_timestep_cooldown_steps=self.config.adaptive_timestep_cooldown_steps,
            adaptive_timestep_probe_threshold=self.config.adaptive_timestep_probe_threshold,
            adaptive_timestep_max_probe_steps=self.config.adaptive_timestep_max_probe_steps,
            adaptive_timestep_debug=self.config.adaptive_timestep_debug,
        )

        # Create progress callback to track integration
        def progress_callback(current: int, total: int, run_id=run_num):
            """Log progress periodically."""
            # Log every 10% or every 100 steps for short runs
            if total <= 1000:
                log_interval = max(1, total // 10)
            else:
                log_interval = max(100, total // 20)

            if current % log_interval == 0 or current == total:
                self._log_result(
                    f"    [PROGRESS] Run {run_id}: step {current}/{total} "
                    f"({100 * current // total}%)"
                )

        # Run the integration with progress tracking
        #
        # NOTE: We must always clean up the per-run temp directory, even when returning
        # early (halted runs) or raising exceptions. We do that by wrapping the entire
        # run/analysis section in a try/finally.
        try:
            # Log diagnostic info for potentially problematic configurations
            if aperture < 0.1:
                self._log_result(
                    f"  [DIAGNOSTIC] Run {run_num}: Small aperture detected ({aperture:.6f} mm)"
                )
            if macroparticle_charge_multiplier > 1000:
                self._log_result(
                    f"  [DIAGNOSTIC] Run {run_num}: Large charge multiplier ({macroparticle_charge_multiplier:.0f})"
                )
                self._log_result(
                    f"    Note: This may significantly slow integration due to strong image forces"
                )

            self._log_result(f"  [DEBUG] Calling run_testbed for Run {run_num}...")

            # Create cancel callback if cancel_flag is provided
            cancel_callback = None
            if cancel_flag is not None:

                def check_cancel():
                    if cancel_flag[0]:
                        self._log_result(
                            f"  [CANCEL] Run {run_num}: Cancellation requested"
                        )
                    return cancel_flag[0] if cancel_flag else False

                cancel_callback = check_cancel

            # Create log callback to stream verbose SC/adaptive timestep output to GUI
            # This ensures logs are visible in real-time even when not saved to file
            log_callback = None
            if (
                self.config.self_consistency_verbosity > 0
                or self.config.adaptive_timestep_debug
            ):
                # Create callback that forwards verbose logs to GUI
                def verbose_log(message: str):
                    # Filter for SC and adaptive timestep related messages
                    if any(
                        keyword in message
                        for keyword in [
                            "Particle",  # SC convergence output
                            "converged",  # SC convergence status
                            "Mass-shell error",  # SC error metrics
                            "γ_velocity",  # SC gamma diagnostics
                            "γ_energy",  # SC gamma diagnostics
                            "γ_mass_shell",  # SC gamma diagnostics
                            "Energy jump detected",  # Adaptive timestep trigger
                            "Reducing timestep",  # Adaptive timestep action
                            "Proximity refinement",  # Adaptive timestep near walls
                            "Cooldown mode",  # Adaptive timestep state
                            "Probing stability",  # Adaptive timestep recovery
                            "Returning to normal timestep",  # Adaptive timestep recovery
                            "Stable",  # Adaptive timestep status
                            "Unstable",  # Adaptive timestep status
                            "Minimum timestep reached",  # Adaptive timestep limit
                            "Max refinement attempts",  # Adaptive timestep limit
                        ]
                    ):
                        self._log_result(f"    [VERBOSE] {message}")

                log_callback = verbose_log

            result = run_testbed(
                options,
                log=log_callback,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )
            self._log_result(f"  [DEBUG] run_testbed completed for Run {run_num}")

            # Check if integration was halted early
            if result.halted_early:
                self._log_result(
                    f"  [WARNING] Run {run_num} halted early: {result.halt_reason}"
                )
                self._log_result(
                    f"    Trajectory contains partial data and will still be analyzed"
                )

            # Sanity check: Verify final z position doesn't exceed expected distance
            if (
                result.rider_trajectory is not None
                and self.config.timestep_strategy == "auto_distance"
            ):
                try:
                    traj = result.rider_trajectory
                    z_array = np.asarray(traj.get("z", []))
                    if len(z_array) > 0:
                        final_z = float(z_array[-1])
                        expected_max_z = wall_z + self.config.target_distance_mm

                        if final_z > expected_max_z:
                            excess = final_z - expected_max_z
                            self._log_result(
                                f"  [WARNING] Run {run_num}: Final z position EXCEEDED expected distance!"
                            )
                            self._log_result(f"    Final z: {final_z:.2f} mm")
                            self._log_result(
                                f"    Expected max z: {expected_max_z:.2f} mm (wall_z={wall_z:.2f} + target={self.config.target_distance_mm:.2f})"
                            )
                            self._log_result(
                                f"    Exceeded by: {excess:.2f} mm ({excess / expected_max_z * 100:.1f}%)"
                            )
                        else:
                            under = expected_max_z - final_z
                            self._log_result(
                                f"  [DEBUG] Run {run_num}: Final z check OK"
                            )
                            self._log_result(
                                f"    Final z: {final_z:.2f} mm (under by {under:.2f} mm)"
                            )
                except Exception as e:
                    self._log_result(
                        f"  [WARNING] Run {run_num}: Failed to check final z position: {e}"
                    )

            # No figures should be generated during sweeps (all display/save flags set to False)
            # If any figures were created (shouldn't happen), close them as a safety measure
            if result.figures:
                self._log_result(
                    f"  [WARNING] Run {run_num}: Unexpected figures generated ({len(result.figures)}), closing them"
                )
                import matplotlib.pyplot as plt

                for fig_name, fig in result.figures.items():
                    try:
                        plt.close(fig)
                        self._log_result(f"    Closed unexpected figure: {fig_name}")
                    except Exception as e:
                        self._log_result(f"    Error closing figure {fig_name}: {e}")

            # Check if run was halted early - if so, skip metrics calculation
            if result.halted_early:
                self._log_result(
                    f"  [INFO] Run {run_num} was halted early - skipping metrics calculation"
                )
                self._log_result(
                    f"    Only trajectory and logs will be saved (if enabled)"
                )
                # Return minimal output with halt information
                output = {
                    "metrics": {},  # Empty metrics
                    "halted_early": True,
                    "halt_reason": result.halt_reason,
                }

                # Add trajectory if available and saving is enabled
                if result.rider_trajectory is not None:
                    save_traj = (
                        self.config.save_all_trajectories
                        or self.config.save_failed_trajectories
                    )
                    if save_traj:
                        traj = result.rider_trajectory
                        stride = self.config.trajectory_stride
                        try:
                            output["trajectory"] = {
                                "z": np.asarray(traj["z"])[::stride].tolist(),
                                "r": np.asarray(traj["r"])[::stride].tolist(),
                                "pz": np.asarray(traj["pz"])[::stride].tolist(),
                                "pr": np.asarray(traj["pr"])[::stride].tolist(),
                                "t": np.asarray(traj["t"])[::stride].tolist(),
                                "gamma": np.asarray(traj["gamma"])[::stride].tolist(),
                            }
                            self._log_result(
                                f"    Halted trajectory saved ({len(traj['z'])} points, stride={stride})"
                            )
                        except Exception as e:
                            self._log_result(
                                f"    [WARNING] Failed to save halted trajectory: {e}"
                            )

                self._log_result(
                    f"  [DEBUG] _run_single_integration returning for halted Run {run_num}"
                )
                return output

            # Extract metrics (only for non-halted runs)
            self._log_result(f"  [DEBUG] Extracting metrics for Run {run_num}...")
            metrics = {}
            if result.rider_delta_e is not None:
                metrics["rider_delta_e_mev"] = result.rider_delta_e
            if result.rider_gamma_initial is not None:
                metrics["rider_gamma_initial"] = result.rider_gamma_initial
            if result.rider_gamma_final is not None:
                metrics["rider_gamma_final"] = result.rider_gamma_final

            # Calculate max_percent_energy_gain from gamma values
            gamma_initial = result.rider_gamma_initial
            gamma_final = result.rider_gamma_final

            # Diagnostic logging
            self._log_result(f"  [RESULT] Run {run_num} metrics:")
            self._log_result(f"    rider_gamma_initial: {gamma_initial}")
            self._log_result(f"    rider_gamma_final: {gamma_final}")

            # Try to calculate from available gamma values
            if (
                gamma_initial is not None
                and gamma_final is not None
                and gamma_initial > 0
            ):
                delta_gamma = gamma_final - gamma_initial
                energy_gain_percent = delta_gamma / gamma_initial * 100.0
                energy_gain_ppm = delta_gamma / gamma_initial * 1e6  # parts per million
                # Calculate delta_e in MeV (for electrons: ΔE = Δγ * m_e*c^2 = Δγ * 0.511 MeV)
                delta_e_mev = delta_gamma * 0.511

                metrics["max_percent_energy_gain"] = energy_gain_percent
                metrics["percent_delta_e"] = energy_gain_percent
                metrics["delta_gamma"] = delta_gamma
                metrics["delta_e_mev"] = delta_e_mev
                metrics["energy_gain_ppm"] = energy_gain_ppm

                self._log_result(f"    delta_gamma: {delta_gamma:.12e}")
                self._log_result(f"    delta_e_mev: {delta_e_mev:.12e} MeV")
                self._log_result(
                    f"    max_percent_energy_gain: {energy_gain_percent:.12e}%"
                )
                self._log_result(f"    percent_delta_e: {energy_gain_percent:.12e}%")
                self._log_result(f"    energy_gain_ppm: {energy_gain_ppm:.6f} ppm")

                # For optimization runs, show what the optimizer sees
                if hasattr(self, "config") and hasattr(self.config, "mode"):
                    if self.config.mode == "optimization":
                        # Optimizer minimizes, so negate for maximization objectives
                        optimizer_value = (
                            -energy_gain_percent
                        )  # We maximize percent gain
                        self._log_result(
                            f"    optimizer_objective: {optimizer_value:.12e}"
                        )
            else:
                # Fallback: Try to calculate from trajectory if gamma values are missing
                self._log_result(
                    f"  [WARNING] Gamma values missing, attempting trajectory fallback..."
                )
                if result.rider_trajectory is not None:
                    try:
                        traj = result.rider_trajectory
                        gamma_array = np.asarray(traj.get("gamma", []))
                        if len(gamma_array) > 0:
                            gamma_initial_fallback = float(gamma_array[0])
                            gamma_final_fallback = float(gamma_array[-1])
                            if gamma_initial_fallback > 0:
                                delta_gamma_fallback = (
                                    gamma_final_fallback - gamma_initial_fallback
                                )
                                energy_gain_percent = (
                                    delta_gamma_fallback
                                    / gamma_initial_fallback
                                    * 100.0
                                )
                                energy_gain_ppm = (
                                    delta_gamma_fallback / gamma_initial_fallback * 1e6
                                )
                                delta_e_mev_fallback = delta_gamma_fallback * 0.511

                                metrics["max_percent_energy_gain"] = energy_gain_percent
                                metrics["percent_delta_e"] = energy_gain_percent
                                metrics["delta_gamma"] = delta_gamma_fallback
                                metrics["delta_e_mev"] = delta_e_mev_fallback
                                metrics["energy_gain_ppm"] = energy_gain_ppm

                                self._log_result(
                                    f"  [OK] Fallback calculation successful:"
                                )
                                self._log_result(
                                    f"    gamma_initial (from traj): {gamma_initial_fallback:.12e}"
                                )
                                self._log_result(
                                    f"    gamma_final (from traj): {gamma_final_fallback:.12e}"
                                )
                                self._log_result(
                                    f"    delta_gamma: {delta_gamma_fallback:.12e}"
                                )
                                self._log_result(
                                    f"    delta_e_mev: {delta_e_mev_fallback:.12e} MeV"
                                )
                                self._log_result(
                                    f"    max_percent_energy_gain: {energy_gain_percent:.12e}%"
                                )
                                self._log_result(
                                    f"    percent_delta_e: {energy_gain_percent:.12e}%"
                                )
                                self._log_result(
                                    f"    energy_gain_ppm: {energy_gain_ppm:.6f} ppm"
                                )
                            else:
                                self._log_result(
                                    f"  [ERROR] Fallback gamma_initial <= 0"
                                )
                        else:
                            self._log_result(
                                f"  [ERROR] Trajectory gamma array is empty"
                            )
                    except Exception as e:
                        self._log_result(f"  [ERROR] Fallback calculation failed: {e}")
                else:
                    self._log_result(
                        f"  [ERROR] No trajectory data available for fallback"
                    )

                # If still no metric calculated, warn explicitly
                if "max_percent_energy_gain" not in metrics:
                    self._log_result(
                        f"  [CRITICAL] max_percent_energy_gain could not be calculated for Run {run_num}"
                    )
                    self._log_result(
                        f"  [CRITICAL] This will result in NaN/inf for optimization objective"
                    )

            # Add beam optics metrics if available
            if result.rider_emittance_x_mm_mrad is not None:
                metrics["rider_emittance_x_mm_mrad"] = result.rider_emittance_x_mm_mrad
            if result.rider_emittance_y_mm_mrad is not None:
                metrics["rider_emittance_y_mm_mrad"] = result.rider_emittance_y_mm_mrad
            if result.rider_norm_emittance_x_mm_mrad is not None:
                metrics["rider_norm_emittance_x_mm_mrad"] = (
                    result.rider_norm_emittance_x_mm_mrad
                )
            if result.rider_norm_emittance_y_mm_mrad is not None:
                metrics["rider_norm_emittance_y_mm_mrad"] = (
                    result.rider_norm_emittance_y_mm_mrad
                )
            if result.rider_beta_x_m is not None:
                metrics["rider_beta_x_m"] = result.rider_beta_x_m
            if result.rider_beta_y_m is not None:
                metrics["rider_beta_y_m"] = result.rider_beta_y_m

            output = {"metrics": metrics}

            self._log_result(
                f"  [DEBUG] Processing trajectory data for Run {run_num}..."
            )
            # Add trajectory data for distance calculation even if not saving full arrays
            # (needed for diagnostics and basic metrics)
            if result.rider_trajectory is not None:
                traj = result.rider_trajectory

                # Always include minimal trajectory info for distance calculation
                try:
                    z_array = np.asarray(traj["z"])
                    if len(z_array) > 0:
                        output["_distance_info"] = {
                            "z_start": float(z_array[0]),
                            "z_end": float(z_array[-1]),
                            "num_steps": len(z_array),
                        }
                except Exception as e:
                    print(f"[DEBUG] Failed to extract distance info: {e}")

                # Perform stability analysis if enabled
                if self.config.smoothness_enabled:
                    self._log_result(
                        f"  [DEBUG] Performing stability analysis for Run {run_num}..."
                    )

                    # Create stability config from optimization config
                    smoothness_config = SmoothnessConfig(
                        enabled=True,
                        window_size=self.config.smoothness_window_size,
                        oscillation_threshold=self.config.smoothness_oscillation_threshold,
                        trend_smoothness_threshold=self.config.smoothness_trend_threshold,
                        reject_on_violation=self.config.smoothness_reject_on_violation,
                        max_allowed_violations=self.config.smoothness_max_violations,
                    )

                    # Analyze trajectory stability
                    smoothness_result = analyze_trajectory_smoothness(
                        traj, smoothness_config, particle_mass_amu=rider_m_particle
                    )

                    # Store stability analysis in output
                    output["stability_analysis"] = {
                        "passed": smoothness_result.passed,
                        "num_violations": len(smoothness_result.violations),
                        "oscillation_score": smoothness_result.oscillation_score,
                        "trend_smoothness_score": smoothness_result.trend_smoothness_score,
                        "quality": smoothness_result.quality_summary,
                    }

                    if not smoothness_result.passed:
                        self._log_result(
                            f"  [WARNING] Stability check FAILED for Run {run_num}"
                        )
                        self._log_result(
                            f"    Quality: {smoothness_result.quality_summary}"
                        )
                        if len(smoothness_result.violations) > 0:
                            self._log_result(
                                f"    Violations: {len(smoothness_result.violations)}"
                            )
                            for v in smoothness_result.violations[:2]:  # Show first 2
                                self._log_result(f"      - {v.description}")

                        if self.config.smoothness_reject_on_violation:
                            self._log_result(
                                f"  [REJECT] Run {run_num} rejected due to numerical instability"
                            )
                            # Mark metrics as invalid to trigger rejection in optimizer
                            output["metrics"]["max_percent_energy_gain"] = np.nan
                            output["stability_rejected"] = True
                    else:
                        self._log_result(
                            f"  [OK] Stability check PASSED for Run {run_num}: {smoothness_result.quality_summary}"
                        )
                else:
                    # Smoothness checking is disabled
                    self._log_result(
                        f"  [INFO] Stability analysis DISABLED for Run {run_num} (smoothness_enabled=False)"
                    )

                # Only save full trajectory arrays if explicitly requested
                # Check if any trajectory saving option is enabled
                # Note: save_top_n_trajectories is handled separately by re-running top N after optimization
                save_traj = (
                    self.config.save_all_trajectories
                    or self.config.save_failed_trajectories
                )
                if save_traj:
                    # Downsample trajectory
                    stride = self.config.trajectory_stride
                    try:
                        # Ensure we convert numpy arrays to flat lists
                        output["trajectory"] = {
                            "z": np.asarray(traj["z"])[::stride].tolist(),
                            "r": np.asarray(traj["r"])[::stride].tolist(),
                            "pz": np.asarray(traj["pz"])[::stride].tolist(),
                            "pr": np.asarray(traj["pr"])[::stride].tolist(),
                            "t": np.asarray(traj["t"])[::stride].tolist(),
                            "gamma": np.asarray(traj["gamma"])[::stride].tolist(),
                        }
                    except Exception as e:
                        self._log_result(
                            f"    [WARNING] Failed to save trajectory arrays: {e}"
                        )

                # Add halt information to output if present
                if result.halted_early:
                    output["halted_early"] = True
                    output["halt_reason"] = result.halt_reason
            else:
                # No trajectory data available - stability analysis cannot run
                self._log_result(
                    f"  [WARNING] No trajectory data available for Run {run_num}"
                )
                if self.config.smoothness_enabled:
                    self._log_result(
                        f"  [WARNING] Stability analysis SKIPPED - no trajectory data returned from integration"
                    )
                    self._log_result(
                        f"    Check that transverse_save=True in SimulationOptions"
                    )

            self._log_result(
                f"  [DEBUG] _run_single_integration returning for Run {run_num}"
            )

            return output
        finally:
            # Always clean up temporary run directory (success, halt, exception, cancel)
            try:
                import shutil

                if run_output_dir.exists():
                    shutil.rmtree(run_output_dir)
                    self._log_result(
                        f"  [DEBUG] Cleaned up temp directory: {run_output_dir.name}"
                    )
            except Exception as e:
                self._log_result(
                    f"  [WARNING] Failed to clean up temp directory {run_output_dir.name}: {e}"
                )

    def _cleanup_orphaned_temp_dirs(self):
        """Clean up any orphaned _temp_run directories from previous runs.

        This is called on plugin initialization to remove temp directories
        that weren't cleaned up due to crashes or interruptions.
        """
        import shutil
        from pathlib import Path

        try:
            output_dir = Path(self.sweep_output_dir)
            if not output_dir.exists():
                return

            # Find all _temp_run directories
            temp_dirs = list(output_dir.glob("_temp_run_*"))

            if temp_dirs:
                print(
                    f"[CLEANUP] Found {len(temp_dirs)} orphaned temp directories, removing..."
                )
                for temp_dir in temp_dirs:
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"[CLEANUP] Removed: {temp_dir.name}")
                    except Exception as e:
                        print(f"[WARNING] Failed to remove {temp_dir.name}: {e}")
        except Exception as e:
            print(f"[WARNING] Error during temp directory cleanup: {e}")

    def _save_sweep_results(
        self, results: List[Dict[str, Any]], failed_runs: List[Dict[str, Any]] = None
    ) -> None:
        """Save sweep results to JSON file with timestamp in dedicated folder.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            Successful run results
        failed_runs : List[Dict[str, Any]], optional
            Failed/skipped run information
        """
        from datetime import datetime

        # Close any open log file before creating new directory
        if self._log_file is not None:
            self._close_log_file()

        # Generate timestamp in sortable format: YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get config name if available (strip extension and path)
        config_name = "sweep"
        if hasattr(self, "last_loaded_config") and self.last_loaded_config:
            config_name = Path(self.last_loaded_config).stem

        # Create timestamped folder: YYYYMMDD_HHMMSS_configname
        sweep_dir = Path(self.sweep_output_dir) / f"{timestamp}_{config_name}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        # Create filename: sweep_results.json (inside timestamped folder)
        output_file = sweep_dir / "sweep_results.json"

        # Prepare data for JSON serialization
        output_data = {
            "sweep_name": f"Parameter Sweep {timestamp}",
            "timestamp": timestamp,
            "config": {
                "aperture_range": self.config.aperture_range,
                "aperture_points": self.config.aperture_points,
                "energy_range": self.config.energy_range,
                "energy_points": self.config.energy_points,
                "transverse_offset_fractions": self.config.transverse_offset_fractions,
                "starting_z_positions": self.config.starting_z_positions,
                "simulation_type": self.config.simulation_type.name,
                "wall_z": self.config.wall_z,
                "wall_z_range": self.config.wall_z_range,
                "wall_z_points": self.config.wall_z_points,
                "auto_steps": self.config.auto_steps,
            },
            "results": results,
            "total_runs": len(results),
        }

        # Add failed runs information if present
        if failed_runs:
            output_data["failed_runs"] = failed_runs
            output_data["num_failed"] = len(failed_runs)

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        self._log_result(f"Results saved to: {output_file}")
        if failed_runs:
            self._log_result(f"  (includes {len(failed_runs)} failed/timed-out runs)")

        # Reopen log file in the final output directory
        self._open_log_file(sweep_dir)

        # Generate and save summary plots
        if len(results) > 0:
            self._generate_summary_plots(results, sweep_dir)

    def _generate_summary_plots(
        self, results: List[Dict[str, Any]], output_dir: Path
    ) -> None:
        """Generate summary plots for the sweep results."""
        try:
            import matplotlib.pyplot as plt

            # Heatmap only needs metrics, not full trajectories
            # Collect all data from results with metrics
            apertures = []
            energies = []
            delta_es = []

            for result in results:
                params = result.get("parameters", {})
                metrics = result.get("metrics", {})

                # Only include if we have the necessary metrics
                if metrics:
                    apertures.append(params.get("aperture_radius", 0))
                    energies.append(params.get("particle_energy_gev", 0))
                    delta_es.append(metrics.get("rider_delta_e_mev", 0))

            if len(delta_es) == 0:
                self._log_result("[INFO] No results with metrics to plot")
                return

            # Count how many parameters were actually swept
            # (have more than 1 unique value across all results)
            # Collect all unique parameter values across all results
            all_param_values = {}
            for result in results:
                params = result.get("parameters", {})
                for key, value in params.items():
                    # Skip non-numeric parameters and internal bookkeeping
                    if key in ["simulation_type", "run_number", "timestep", "steps"]:
                        continue
                    if key not in all_param_values:
                        all_param_values[key] = []
                    all_param_values[key].append(value)

            # Count parameters with more than one unique value
            num_swept_params = 0
            for param_name, values in all_param_values.items():
                unique_values = set(v for v in values if v is not None)
                if len(unique_values) > 1:
                    num_swept_params += 1

            # Only generate heatmap if exactly 2 parameters were swept
            if num_swept_params == 2:
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))

                scatter = ax.scatter(
                    energies,
                    [a * 1e3 for a in apertures],  # Convert mm to microns
                    c=delta_es,
                    cmap="viridis",
                    s=150,
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=1,
                )

                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label("Energy Gain ΔE (MeV)", fontsize=12)

                ax.set_xlabel("Particle Energy (GeV)", fontsize=12)
                ax.set_ylabel("Aperture Radius (μm)", fontsize=12)
                ax.set_title(
                    "Parameter Space Exploration: Energy Gain",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.grid(True, alpha=0.3)

                # Use log scale if range is large
                if (
                    len(energies) > 0 and max(energies) / min(energies) > 10
                    if min(energies) > 0
                    else False
                ):
                    ax.set_xscale("log")
                if (
                    len(apertures) > 0 and max(apertures) / min(apertures) > 10
                    if min(apertures) > 0
                    else False
                ):
                    ax.set_yscale("log")

                plt.tight_layout()

                heatmap_file = output_dir / "sweep_heatmap.png"
                plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
                plt.close(fig)

                self._log_result(f"[OK] Heatmap saved to: {heatmap_file}")
            else:
                self._log_result(
                    f"[INFO] Skipping heatmap generation ({num_swept_params} parameters swept; heatmap only generated for 2-parameter sweeps)"
                )

            # Plot best trajectory (only if trajectories were saved)
            results_with_traj = [
                r
                for r in results
                if "trajectory" in r and len(r.get("trajectory", {}).get("z", [])) > 0
            ]

            if results_with_traj:
                best_result = max(
                    results_with_traj,
                    key=lambda r: r.get("metrics", {}).get("rider_delta_e_mev", -1e9),
                )
                self._plot_single_trajectory(
                    best_result, output_dir / "sweep_best_trajectory.png"
                )
            else:
                self._log_result(
                    "[INFO] No trajectories available for trajectory plot (enable 'Save trajectories' to generate)"
                )

        except Exception as e:
            self._log_result(f"[WARNING] Failed to generate summary plots: {e}")

    def _plot_single_trajectory(
        self, result: Dict[str, Any], output_file: Path
    ) -> None:
        """Plot trajectory for a single run."""
        try:
            import matplotlib.pyplot as plt

            traj = result.get("trajectory", {})
            params = result.get("parameters", {})
            metrics = result.get("metrics", {})

            z = np.array(traj.get("z", []))
            r = np.array(traj.get("r", []))

            if len(z) == 0:
                return

            aperture = params.get("aperture_radius", 0)
            energy = params.get("particle_energy_gev", 0)
            delta_e = metrics.get("rider_delta_e_mev", 0)
            gamma_initial = metrics.get("rider_gamma_initial", 1)
            gamma_final = metrics.get("rider_gamma_final", 1)

            # Calculate energy evolution
            energy_mev_initial = (gamma_initial - 1) * 0.511
            energy_mev_final = (gamma_final - 1) * 0.511

            if len(z) > 1 and abs(z[-1] - z[0]) > 1e-6:
                energy_mev = energy_mev_initial + delta_e * (z - z[0]) / (z[-1] - z[0])
            else:
                energy_mev = np.full_like(z, energy_mev_initial)

            # Create 3-panel plot
            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 1, hspace=0.3)

            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])

            fig.suptitle(
                f"Best Trajectory: a={aperture * 1e3:.1f}μm, E={energy:.1f}GeV, ΔE={delta_e:.6f}MeV",
                fontsize=12,
                fontweight="bold",
            )

            # Plot 1: Energy gain vs z
            ax1.plot(z, energy_mev - energy_mev_initial, "b-", linewidth=2)
            ax1.set_xlabel("z position (mm)", fontsize=10)
            ax1.set_ylabel("ΔE (MeV)", fontsize=10)
            ax1.set_title("Energy Gain vs Position", fontsize=11, fontweight="bold")
            ax1.grid(True, alpha=0.3)

            # Plot 2: Transverse position vs z
            ax2.plot(z, r, "r-", linewidth=2, label="+r")
            ax2.plot(z, -r, "r--", linewidth=1.5, alpha=0.6, label="-r")
            ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
            ax2.set_xlabel("z position (mm)", fontsize=10)
            ax2.set_ylabel("Transverse position (mm)", fontsize=10)
            ax2.set_title(
                "Transverse Position (±r) vs z", fontsize=11, fontweight="bold"
            )
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)

            # Plot 3: Phase space (r vs z)
            ax3.plot(z, r, "g-", linewidth=2)
            ax3.set_xlabel("z position (mm)", fontsize=10)
            ax3.set_ylabel("r (mm)", fontsize=10)
            ax3.set_title("Radial Position Evolution", fontsize=11, fontweight="bold")
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

            self._log_result(f"[OK] Best trajectory plot saved to: {output_file}")

        except Exception as e:
            self._log_result(f"[WARNING] Failed to plot trajectory: {e}")

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
