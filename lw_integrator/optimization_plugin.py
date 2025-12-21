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
from core.types import SimulationType  # type: ignore[import]
from lw_integrator.testbed_runner import (  # type: ignore[import]
    RunResult,
    SimulationOptions,
    run_testbed,
)


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

    # Log to results text if parent is OptimizationPlugin
    if hasattr(parent, "_log_result"):
        parent._log_result(f"[INFO] {title}: {message}")

    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()

    frame = ttk.Frame(dialog, padding=10)
    frame.pack(fill="both", expand=True)

    text = tk.Text(frame, wrap="word", height=8, width=60, relief="flat", borderwidth=0)
    text.insert("1.0", message)
    # Use system default background color instead of querying frame
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


def _show_warning_dialog(parent: tk.Widget, title: str, message: str) -> None:
    """Show a warning dialog with selectable text."""
    # Log to console/terminal
    print(f"WARNING: {title}: {message}", flush=True)

    # Log to results text if parent is OptimizationPlugin
    if hasattr(parent, "_log_result"):
        parent._log_result(f"[WARNING] {title}: {message}")

    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()

    frame = ttk.Frame(dialog, padding=10)
    frame.pack(fill="both", expand=True)

    text = tk.Text(frame, wrap="word", height=8, width=60, relief="flat", borderwidth=0)
    text.insert("1.0", message)
    text.configure(state="disabled", bg=frame.cget("background"))
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


@dataclass
class OptimizationConfig:
    """Configuration for parameter sweep or optimization run."""

    # Simulation type
    simulation_type: SimulationType = SimulationType.CONDUCTING_WALL

    # Mode selection
    mode: str = "blind_sweep"  # "blind_sweep" or "optimization"

    # Optimization settings (only used when mode="optimization")
    optimization_method: str = "genetic_algorithm"  # "genetic_algorithm", "differential_evolution", "nelder_mead", "multi_start", "adaptive_grid"
    optimization_maxiter: int = 50  # Max iterations/generations
    optimization_population_size: int = (
        20  # For genetic algorithm and differential evolution
    )
    optimization_mutation_rate: float = 0.1  # For genetic algorithm
    optimization_crossover_rate: float = 0.7  # For genetic algorithm
    optimization_n_starts: int = 5  # For multi_start method
    optimization_save_top_n: int = 3  # Save trajectories from top N results

    # Parameter ranges
    aperture_range: Tuple[float, float] = (1e-5, 1e-3)  # mm (10 μm to 1 mm)
    aperture_points: int = 10
    aperture_log_scale: bool = True

    energy_range: Tuple[float, float] = (1.0, 1000.0)  # GeV
    energy_points: int = 10
    energy_log_scale: bool = True

    transverse_offset_fractions: List[float] = None  # Fraction of aperture radius
    starting_z_positions: List[float] = None  # mm (particle starting z-coordinates)

    # Sweepable parameters (can be added to grid)
    transverse_momentum_range: Optional[Tuple[float, float]] = None  # amu·mm/ns
    transverse_momentum_points: int = 1
    transverse_spread_range: Optional[Tuple[float, float]] = None  # mm (transv_dist)
    transverse_spread_points: int = 1
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
    timestep_strategy: str = "fixed"  # "fixed", "energy_scaled", or "auto_distance"
    energy_scale_exponent: float = 1.0  # For energy_scaled: h ∝ γ^-α
    target_distance_mm: float = 100.0  # For auto_distance: distance to reach
    z_cutoff_mode: str = "absolute"  # "absolute" or "relative" (for BUNCH_TO_BUNCH)

    # Fixed particle parameters (not swept)
    transv_mom: float = 1.2e-05  # amu·mm/ns
    transv_dist: float = 2e-06  # mm - transverse distance from axis
    m_particle: float = 0.00054857990907  # amu (electron mass)
    pcount: int = 1
    charge_sign: float = -1.0
    stripped_ions: float = 1.0

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
    save_trajectories: bool = False  # Save trajectory data for each run
    trajectory_stride: int = 10  # Save every Nth point to reduce file size

    # Stability and robustness options (from SimulationOptions)
    self_consistency_enabled: bool = True
    self_consistency_tolerance: float = 1e-4
    self_consistency_max_iterations: int = 5
    self_consistency_verbosity: int = 0  # 0=silent, 1=basic, 2=detailed

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

    def __post_init__(self):
        """Set defaults for list fields."""
        if self.transverse_offset_fractions is None:
            self.transverse_offset_fractions = [0.0]
        if self.starting_z_positions is None:
            self.starting_z_positions = [0.0]  # Default: start at origin
        if self.objective_weights is None:
            self.objective_weights = {}

    def calculate_timestep_for_energy(
        self, energy_gev: float, m_particle_amu: float = 0.00054857990907
    ) -> float:
        """Calculate appropriate timestep for given energy based on strategy.

        Parameters
        ----------
        energy_gev : float
            Particle energy in GeV
        m_particle_amu : float
            Particle mass in amu (default: electron)

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
            # Distance = N_steps × β × c × h × γ
            # Therefore: h = Distance / (N_steps × β × c × γ)
            c_mmns = 299.792458  # mm/ns
            h_calculated = self.target_distance_mm / (
                self.steps * beta * c_mmns * gamma
            )
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

    # Ensure minimum reasonable value (hard minimum of 200 steps)
    return max(steps, 200)


def calculate_steps_from_duration(
    total_duration_ns: float,
    particle_energy_gev: float,
    particle_mass_amu: float = 0.00054857990907,
) -> tuple[int, float]:
    """Calculate timestep and number of steps from total duration.

    Auto-calculates timestep (h) given a desired total duration and step count.
    Enforces a hard minimum of 200 steps.

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
        (number_of_steps, timestep_in_ns) where steps >= 200

    Notes
    -----
    Uses proper time formulation: h = dτ = dt/γ
    Total proper time = N_steps × h
    """
    # Hard minimum of 200 steps
    min_steps = 200

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

        # Store sweep directories
        self.sweep_config_dir = sweep_config_dir or "configs/sweep_configs"
        self.sweep_output_dir = sweep_output_dir or "results/sweeps"

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
        self._build_progress_section()

        # Initialize mode visibility
        self._update_mode_visibility()

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
            text="Optimization (Intelligent Search)",
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
            "Optimization: Use intelligent algorithms to find optimal parameters (faster, finds best configurations).",
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
        ttk.Label(frame, text="Wall Position:").grid(
            row=5, column=0, sticky="w", pady=2
        )
        ttk.Label(frame, text="Conducting wall z-coordinate (mm):").grid(
            row=5, column=1, sticky="w", pady=2
        )
        self.wall_z_var = tk.StringVar(value="2200.0")
        ttk.Entry(frame, textvariable=self.wall_z_var, width=10).grid(
            row=5, column=2, sticky="w", pady=2, padx=5
        )

        # Cavity Spacing (for SWITCHING_WALL)
        ttk.Label(frame, text="Cavity Spacing:").grid(
            row=6, column=0, sticky="w", pady=2
        )
        ttk.Label(frame, text="Cavity spacing (mm, SWITCHING_WALL only):").grid(
            row=6, column=1, sticky="w", pady=2
        )
        self.cavity_spacing_var = tk.StringVar(value="1e5")
        self.cavity_spacing_entry = ttk.Entry(
            frame, textvariable=self.cavity_spacing_var, width=10
        )
        self.cavity_spacing_entry.grid(row=6, column=2, sticky="w", pady=2, padx=5)

        # Timestep Auto-Calculation (always enabled)
        ttk.Label(frame, text="Timestep Calculation:").grid(
            row=7, column=0, sticky="w", pady=2
        )
        timestep_frame = ttk.Frame(frame)
        timestep_frame.grid(row=7, column=1, columnspan=3, sticky="ew", pady=2)

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
        distance_frame = ttk.Frame(frame)
        distance_frame.grid(row=7, column=1, columnspan=3, sticky="ew", pady=2)
        ttk.Label(distance_frame, text="Target: wall +").pack(side="left", padx=(0, 2))
        self.auto_steps_distance_var = tk.StringVar(value="10.0")
        ttk.Entry(
            distance_frame, textvariable=self.auto_steps_distance_var, width=6
        ).pack(side="left", padx=2)
        ttk.Label(distance_frame, text="mm (min 200 steps enforced)").pack(
            side="left", padx=2
        )

        # Trajectory saving
        ttk.Label(frame, text="Trajectory Saving:").grid(
            row=8, column=0, sticky="w", pady=2
        )
        strategy_frame = ttk.Frame(frame)
        strategy_frame.grid(row=8, column=1, columnspan=3, sticky="ew", pady=(10, 2))

        self.save_trajectories_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            strategy_frame,
            text="Save trajectories (stride:",
            variable=self.save_trajectories_var,
        ).pack(side="left", padx=(0, 2))

        self.trajectory_stride_var = tk.StringVar(value="10")
        ttk.Entry(
            strategy_frame, textvariable=self.trajectory_stride_var, width=6
        ).pack(side="left", padx=2)

        ttk.Label(strategy_frame, text="points)").pack(side="left", padx=(0, 5))
        # Add info note about what gets saved
        traj_note = ttk.Label(
            frame,
            text="Note: If trajectories are not saved, only heatmap plot and optimization metrics will be saved.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray50",
        )
        traj_note.grid(row=9, column=1, columnspan=3, sticky="w", pady=(2, 0))

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

        # Transverse Momentum
        self._add_sweepable_param(
            frame,
            row,
            "rider_transv_mom",
            "Transverse Momentum (amu·mm/ns):",
            "1.2e-05",
            width=15,
        )
        row += 1

        # Transverse Spread (bunch radius)
        self._add_sweepable_param(
            frame,
            row,
            "rider_transv_dist",
            "Transverse Spread (mm):",
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
        output_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        ttk.Label(output_frame, text="Save trajectories from top N results:").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_save_top_n_var = tk.StringVar(value="3")
        ttk.Entry(
            output_frame, textvariable=self.optimization_save_top_n_var, width=8
        ).grid(row=0, column=1, sticky="w", pady=2)

        # Initialize visibility
        self._update_optimization_controls()

    def _update_mode_visibility(self):
        """Update visibility of sections based on selected mode."""
        mode = self.mode_var.get()

        if mode == "blind_sweep":
            # Hide optimization settings
            self.optimization_frame.pack_forget()
        else:  # optimization
            # Show optimization settings
            self.optimization_frame.pack(fill="x", padx=10, pady=5)

        # Update parameter visibility based on simulation type
        self._update_parameter_visibility()

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

        # Row 2: Results viewing
        results_frame = ttk.Frame(frame)
        results_frame.pack(fill="x", pady=(10, 2))

        ttk.Label(results_frame, text="Results:").pack(side="left", padx=(5, 10))

        ttk.Button(
            results_frame, text="View Results", command=self._on_view_results
        ).pack(side="left", padx=5)

        ttk.Button(
            results_frame,
            text="Plot Trajectories",
            command=self._on_plot_trajectories,
        ).pack(side="left", padx=5)

        # Plot display options
        plot_options_frame = ttk.LabelFrame(
            self.scrollable_frame, text="Sweep Display Options", padding=10
        )
        plot_options_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(plot_options_frame, text="Display every Nth run:").grid(
            row=0, column=1, sticky="w", padx=(20, 5), pady=2
        )
        self.plot_stride_var = tk.StringVar(value="10")
        ttk.Entry(plot_options_frame, textvariable=self.plot_stride_var, width=5).grid(
            row=0, column=2, sticky="w", pady=2
        )
        ttk.Label(plot_options_frame, text="(1=all, 10=every 10th)").grid(
            row=0, column=3, sticky="w", padx=(5, 0), pady=2
        )

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

    def _parse_list_field(self, value: str) -> List[float]:
        """Parse comma-separated list of floats."""
        try:
            return [float(x.strip()) for x in value.split(",") if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid list format: {value}")

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

    def _gather_config(self) -> OptimizationConfig:
        """Gather configuration from UI fields."""
        return OptimizationConfig(
            simulation_type=SimulationType[self.sim_type_var.get()],
            mode=self.mode_var.get(),
            optimization_method=self.optimization_method_var.get(),
            optimization_maxiter=int(self.optimization_maxiter_var.get()),
            optimization_population_size=int(self.optimization_popsize_var.get()),
            optimization_mutation_rate=float(self.optimization_mutation_var.get()),
            optimization_crossover_rate=float(self.optimization_crossover_var.get()),
            optimization_n_starts=int(self.optimization_nstarts_var.get()),
            optimization_save_top_n=int(self.optimization_save_top_n_var.get()),
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
            cavity_spacing=float(self.cavity_spacing_var.get()),
            timestep=float(self.duration_var.get())
            if self.timestep_mode_var.get() == "count"
            else 3e-7,
            steps=int(self.steps_var.get())
            if self.timestep_mode_var.get() == "duration"
            else 200,
            auto_steps=True,  # Always use auto-calculation
            auto_steps_target=int(self.steps_var.get())
            if self.timestep_mode_var.get() == "duration"
            else 200,
            auto_steps_distance_past_wall=float(self.auto_steps_distance_var.get()),
            objective=self.objective_var.get(),
            transv_mom=float(self.sweep_params["rider_transv_mom"]["fixed_var"].get()),
            transv_dist=float(
                self.sweep_params["rider_transv_dist"]["fixed_var"].get()
            ),
            m_particle=float(self.sweep_params["rider_m_particle"]["fixed_var"].get()),
            pcount=int(self.sweep_params["rider_pcount"]["fixed_var"].get()),
            charge_sign=float(
                self.sweep_params["rider_charge_sign"]["fixed_var"].get()
            ),
            stripped_ions=float(self.rider_stripped_ions_var.get()),
            save_trajectories=self.save_trajectories_var.get(),
            trajectory_stride=int(self.trajectory_stride_var.get()),
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
            self.main_timestep_display_var.set(f"{opt_config.timestep:.2e}")

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

        ttk.Label(sc_frame, text="Verbosity (0=silent, 1=basic, 2=detailed):").pack(
            anchor="w", pady=(5, 0)
        )
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
            text="Debug logging (recommended for sweeps)",
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

        # Warn if gamma > 50,000 (roughly > 25 GeV for electrons)
        if gamma_max > 50000:
            warnings.append(
                f"• Very high energy detected ({energy_max:.1f} GeV, γ≈{gamma_max:.0f})\n"
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

        except Exception as e:
            _show_error_dialog(self, "Configuration Error", str(e))
            return

        # Update UI state
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
        self._update_progress_text("Stopping...")

        # Signal main GUI cancellation
        if self.gui_controller and hasattr(self.gui_controller, "_cancel_requested"):
            self.gui_controller._cancel_requested = True

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
            self.cavity_spacing_var.set(str(data.get("cavity_spacing", 1e5)))
            self.steps_var.set(str(data.get("steps", 2000)))
            self.objective_var.set(data.get("objective", "max_energy_gain"))

            # Load trajectory options
            self.save_trajectories_var.set(data.get("save_trajectories", False))

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

            # Update mode visibility
            self._update_mode_visibility()
            self._update_optimization_controls()

            # Load stability options (with defaults from SimulationOptions)
            loaded_config = self._gather_config()
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
            self.config = loaded_config

            self._log_result("[OK] Configuration loaded successfully")
            self._log_result("[INFO] Stability options:")
            self._log_result(
                f"  Self-consistency: {self.config.self_consistency_enabled} (tol={self.config.self_consistency_tolerance:.1e})"
            )
            # Energy monitoring removed - functionality in adaptive timestep
            self._log_result(
                f"  Adaptive timestep: {self.config.adaptive_timestep_enabled} (threshold={self.config.adaptive_timestep_threshold * 100:.0f}%)"
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
            config = self._gather_config()
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
                "cavity_spacing": config.cavity_spacing,
                "steps": config.steps,
                "objective": config.objective,
                "save_trajectories": config.save_trajectories,
                # Optimization parameters
                "optimization_method": config.optimization_method,
                "optimization_maxiter": config.optimization_maxiter,
                "optimization_population_size": config.optimization_population_size,
                "optimization_mutation_rate": config.optimization_mutation_rate,
                "optimization_crossover_rate": config.optimization_crossover_rate,
                "optimization_n_starts": config.optimization_n_starts,
                "optimization_save_top_n": config.optimization_save_top_n,
                # Stability options
                "self_consistency_enabled": config.self_consistency_enabled,
                "self_consistency_tolerance": config.self_consistency_tolerance,
                "self_consistency_max_iterations": config.self_consistency_max_iterations,
                "self_consistency_verbosity": config.self_consistency_verbosity,
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
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            # Update last_loaded_config so sweep results use correct name
            self.last_loaded_config = filepath

            self._log_result(f"[OK] Configuration saved to {filepath}")
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
        """Open results viewer and automatically load latest results."""
        import glob
        import os

        # Use sweep output directory from GUI preferences
        default_results_dir = self.sweep_output_dir

        # Look for most recent sweep results
        search_patterns = [
            os.path.join(default_results_dir, "*", "sweep_results.json"),
            os.path.join(
                "optimization_results", "sweep_*", "sweep_results.json"
            ),  # Legacy location
            os.path.join(self.config.output_dir, "sweep_results.json"),
        ]

        result_files = []
        for pattern in search_patterns:
            result_files.extend(glob.glob(pattern))

        if result_files:
            # Sort by modification time, most recent first
            result_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = result_files[0]

            # Ask user if they want to view the latest results or choose a different file
            dialog = tk.Toplevel(self)
            dialog.title("View Results")
            dialog.geometry("500x300")
            dialog.transient(self)
            dialog.grab_set()

            msg_frame = ttk.Frame(dialog, padding=20)
            msg_frame.pack(fill="both", expand=True)

            ttk.Label(
                msg_frame,
                text="Results Found",
                font=("TkDefaultFont", 12, "bold"),
            ).pack(anchor="w", pady=(0, 10))

            info = ttk.Label(
                msg_frame,
                text=f"Most recent results file found:\n\n{os.path.basename(os.path.dirname(latest_file)) if 'sweep_' in latest_file else os.path.basename(latest_file)}",
                wraplength=450,
            )
            info.pack(anchor="w", pady=(0, 20))

            btn_frame = ttk.Frame(msg_frame)
            btn_frame.pack(fill="x", pady=10)

            def load_latest():
                dialog.destroy()
                self._load_and_plot_results(latest_file)

            def choose_file():
                dialog.destroy()
                self._on_plot_trajectories()

            ttk.Button(
                btn_frame,
                text="View Latest Results",
                command=load_latest,
                style="Accent.TButton",
            ).pack(side="left", padx=5, expand=True, fill="x")

            ttk.Button(
                btn_frame,
                text="Choose Different File...",
                command=choose_file,
            ).pack(side="left", padx=5, expand=True, fill="x")

            ttk.Button(
                msg_frame,
                text="Cancel",
                command=dialog.destroy,
            ).pack(pady=(10, 0))

            # Center on parent
            dialog.update_idletasks()
            x = self.winfo_rootx() + (self.winfo_width() - dialog.winfo_width()) // 2
            y = self.winfo_rooty() + (self.winfo_height() - dialog.winfo_height()) // 2
            dialog.geometry(f"+{x}+{y}")

        else:
            # No results found, offer to browse
            response = messagebox.askyesno(
                "No Results Found",
                "No recent sweep results found in the default directory.\n\n"
                f"Default location: {default_results_dir}\n\n"
                "Would you like to browse for a results file?",
                parent=self,
            )
            if response:
                self._on_plot_trajectories()

    def _load_and_plot_results(self, file_path: str):
        """Load results file and display trajectory viewer with plots."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Try to detect file format
            results = None

            if "results" in data:
                # New format: sweep_results.json
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
        import os

        # Use sweep output directory from GUI preferences, then fall back to legacy
        legacy_results_dir = "optimization_results"

        if os.path.exists(self.sweep_output_dir) and os.listdir(self.sweep_output_dir):
            initial_dir = self.sweep_output_dir
        elif os.path.exists(legacy_results_dir):
            initial_dir = legacy_results_dir
        else:
            initial_dir = self.config.output_dir

        # Ask user to select results file
        file_path = filedialog.askopenfilename(
            title="Select Results JSON File",
            initialdir=initial_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Try to detect file format
            # Format 1: sweep_results.json with "results" array
            # Format 2: Legacy trajectory file with "core" -> "rider" structure
            results = None

            if "results" in data:
                # New format: sweep_results.json
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
                        "Make sure 'Save trajectories' was enabled during the sweep.",
                    )
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
                    "- Legacy trajectory file with 'core'/'rider' structure",
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

            if method == "genetic_algorithm":
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

            # Log results
            self._log_result("")
            self._log_result("=" * 80)
            self._log_result("OPTIMIZATION COMPLETE")
            self._log_result("=" * 80)
            self._log_result(f"Best {metric_name}: {result.fun:.6e}")
            self._log_result("Best parameters:")
            for param_name, value in result.best_params_dict.items():
                self._log_result(f"  {param_name}: {value:.6e}")
            self._log_result("")
            self._log_result(
                f"Function evaluations: {result.nfev if hasattr(result, 'nfev') else 'N/A'}"
            )
            self._log_result("")

            # Save results
            self._save_optimization_results(result, param_names)

            self._log_result("[OK] Optimization complete!")

        except Exception as e:
            import traceback

            error_msg = f"Optimization failed: {e}\n{traceback.format_exc()}"
            self._log_result(f"[ERROR] {error_msg}")
        finally:
            self.running = False
            self._update_progress(100, "Done")

    def _save_optimization_results(self, result, param_names):
        """Save optimization results to file."""
        import json
        from pathlib import Path

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create results dictionary
        results_dict = {
            "optimization_method": self.config.optimization_method,
            "objective": self.config.objective,
            "best_parameters": result.best_params_dict,
            "best_value": float(result.fun),
            "function_evaluations": int(result.nfev)
            if hasattr(result, "nfev")
            else None,
            "success": bool(result.success),
            "message": str(result.message) if hasattr(result, "message") else None,
        }

        # Add convergence history if available
        if hasattr(result, "convergence_history"):
            results_dict["convergence_history"] = result.convergence_history

        # Save to JSON
        results_file = output_dir / "optimization_results.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        self._log_result(f"Results saved to: {results_file}")

    def _run_sweep_background(self, is_finetune=False, finetune_regions=None):
        """Run parameter sweep in background with real integration.

        Args:
            is_finetune: If True, this is a fine-tuning sweep
            finetune_regions: List of parameter regions for fine-tuning
        """
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

            self._log_result(
                f"Starting BLIND SWEEP (Grid Search): {total_runs} total runs"
            )
            self._log_result(
                f"Trajectory saving: {'ENABLED' if self.config.save_trajectories else 'DISABLED'}"
            )

            # Log parameter grid info
            for param_name, values in param_grids.items():
                if len(values) > 1:
                    self._log_result(
                        f"  {param_name}: {len(values)} points from {values[0]:.2e} to {values[-1]:.2e}"
                    )
                else:
                    self._log_result(f"  {param_name}: {values[0]:.2e} (fixed)")
            self._log_result(f"  Timestep strategy: {self.config.timestep_strategy}")
            if self.config.timestep_strategy == "energy_scaled":
                self._log_result(
                    f"    Energy scale exponent: {self.config.energy_scale_exponent} (h ∝ γ^-α)"
                )
            elif self.config.timestep_strategy == "auto_distance":
                self._log_result(
                    f"    Target distance: {self.config.target_distance_mm:.1f} mm"
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
                    timestep = self.config.calculate_timestep_for_energy(
                        energy, rider_m_particle
                    )
                    steps = self.config.steps

                    # Log diagnostic info for first run or every 50th run
                    if run_num == 1 or run_num % 50 == 0:
                        # Calculate gamma for diagnostics
                        AMU_TO_MEV = 931.494
                        rest_energy_mev = rider_m_particle * AMU_TO_MEV
                        gamma = (energy * 1e3) / rest_energy_mev
                        beta = (
                            np.sqrt(1.0 - 1.0 / (gamma * gamma))
                            if gamma > 1.0
                            else 0.999
                        )
                        distance_per_step = beta * gamma * C_MMNS * timestep
                        expected_distance = distance_per_step * steps

                        self._log_result(
                            f"  Run {run_num} timestep strategy '{self.config.timestep_strategy}': "
                            f"E={energy:.1f}GeV, gamma={gamma:.1f}, beta={beta:.6f}"
                        )
                        self._log_result(
                            f"    → timestep={timestep:.2e}ns, steps={steps}, "
                            f"dist/step={distance_per_step:.3f}mm, expected_travel={expected_distance:.1f}mm"
                        )
                elif self.config.auto_steps:
                    # Legacy auto_steps mode (deprecated, but keep for compatibility)
                    distance_to_wall = abs(self.config.wall_z - start_z)
                    total_distance = (
                        distance_to_wall + self.config.auto_steps_distance_past_wall
                    )

                    timestep = calculate_auto_timestep(
                        start_z=start_z,
                        wall_z=self.config.wall_z,
                        distance_past_wall=self.config.auto_steps_distance_past_wall,
                        particle_energy_gev=energy,
                        particle_mass_amu=rider_m_particle,
                        target_steps=self.config.auto_steps_target,
                    )
                    steps = calculate_auto_steps(
                        start_z=start_z,
                        wall_z=self.config.wall_z,
                        distance_past_wall=self.config.auto_steps_distance_past_wall,
                        timestep=timestep,
                        particle_energy_gev=energy,
                        particle_mass_amu=rider_m_particle,
                    )
                else:
                    timestep = self.config.timestep
                    steps = self.config.steps

                # Enforce hard minimum of 200 steps
                if steps < 200:
                    steps = 200

                # Log run start with full parameters for debugging
                self._log_result(
                    f"  [DEBUG] Starting Run {run_num}/{total_runs}: "
                    f"a={aperture:.2e}mm, E={energy:.1f}GeV, z={start_z:.1f}mm, "
                    f"timestep={timestep:.2e}ns, steps={steps}"
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
                                    driver_params=driver_params_dict
                                    if self.config.simulation_type
                                    == SimulationType.BUNCH_TO_BUNCH
                                    else None,
                                    run_num=run_num,
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
                            # Timeout occurred
                            run_timed_out = True
                            self._log_result(
                                f"  [TIMEOUT] Run {run_num} exceeded timeout of {self.config.per_run_timeout}s"
                            )
                            # Note: Thread will continue running but we move on
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
                            driver_params=driver_params_dict
                            if self.config.simulation_type
                            == SimulationType.BUNCH_TO_BUNCH
                            else None,
                            run_num=run_num,
                        )

                    if result is not None:
                        self._log_result(
                            f"  [DEBUG] Run {run_num} integration completed"
                        )

                    if not run_timed_out and result is not None:
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

                        # Log individual run result (every run or every 10th for large sweeps)
                        if (
                            total_runs <= 20
                            or run_num % 10 == 1
                            or run_num == total_runs
                        ):
                            delta_e = result.get("metrics", {}).get(
                                "rider_delta_e_mev", 0.0
                            )
                            log_msg = (
                                f"  Run {run_num}/{total_runs}: "
                                f"a={aperture:.2e}mm, E={energy:.1f}GeV, z={start_z:.1f}mm → "
                                f"ΔE={delta_e:.6f}MeV"
                            )
                            if actual_distance > 0:
                                log_msg += f", traveled={actual_distance:.1f}mm"
                            self._log_result(log_msg)

                            # Store result with all parameters
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
                                    "wall_z": self.config.wall_z,
                                    "rider_m_particle": rider_m_particle,
                                    "rider_charge_sign": rider_charge_sign,
                                    "rider_pcount": int(rider_pcount),
                                    "rider_transv_mom": rider_transv_mom,
                                    "rider_transv_dist": rider_transv_dist,
                                    "simulation_type": self.config.simulation_type.name,
                                },
                                "metrics": result.get("metrics", {}),
                            }

                            # Add trajectory if requested
                            if self.config.save_trajectories and "trajectory" in result:
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
                self._log_result("[OK] Sweep completed!")
                self._log_result(f"  Results saved to: {self.config.output_dir}")
                self._log_result(f"  Successful runs: {len(all_results)}")
                if failed_runs:
                    self._log_result(f"  Failed/timed-out runs: {len(failed_runs)}")
                self._update_progress(100, "Complete!")
        except Exception as e:
            self._log_result(f"[ERROR] Error during sweep: {e}")
            import traceback

            self._log_result(traceback.format_exc())
        finally:
            self.running = False
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
        driver_params: Dict[str, Any] = None,
        run_num: int = 0,
    ) -> Dict[str, Any]:
        """Run a single integration with given parameters."""
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

        # Build rider params
        rider_params = {
            "starting_distance": start_z,
            "transv_mom": rider_transv_mom,
            "transv_dist": transv_offset,  # Use calculated offset
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
            "wall_z": self.config.wall_z,
            "aperture_radius": aperture,
            "mean": 1.0e5,  # Large value (not used for CONDUCTING_WALL)
            "cav_spacing": 1.0e5,
            "z_cutoff": self.config.target_distance_mm
            if self.config.z_cutoff_mode == "relative"
            else 0.0,
            "z_cutoff_mode": self.config.z_cutoff_mode,
        }

        # Create a temporary subdirectory for this run's outputs (will be cleaned up)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        run_output_dir = (
            Path(self.config.output_dir) / f"_temp_run_{run_num}_{timestamp}"
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
            energy_display=False,  # Don't display plots during sweep
            energy_save=False,
            transverse_display=False,
            transverse_save=True,  # Always return trajectory data for metrics calculation
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
            self_consistency_verbosity=max(
                self.config.self_consistency_verbosity, 1
            ),  # At least basic for debugging
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
            adaptive_timestep_debug=self.config.adaptive_timestep_debug
            or True,  # Enable debug for sweeps
        )

        # Create progress callback to track integration and detect hangs
        import time

        last_progress_time = [time.time()]  # Mutable container for closure
        last_reported_step = [0]
        hang_warning_shown = [False]

        def progress_callback(current: int, total: int):
            """Log progress periodically to detect hangs."""
            current_time = time.time()
            last_progress_time[0] = current_time

            # Log every 10% or every 100 steps for short runs
            if total <= 1000:
                log_interval = max(1, total // 10)
            else:
                log_interval = max(100, total // 20)

            if current % log_interval == 0 or current == total:
                self._log_result(
                    f"    [PROGRESS] Run {run_num}: step {current}/{total} "
                    f"({100 * current // total}%)"
                )
                last_reported_step[0] = current

        def check_for_hang():
            """Check if integration appears to be hung."""
            elapsed = time.time() - last_progress_time[0]
            if (
                elapsed > 30.0 and not hang_warning_shown[0]
            ):  # 30 seconds without progress
                hang_warning_shown[0] = True
                self._log_result(
                    f"  [WARNING] Run {run_num} appears hung - no progress for {elapsed:.0f}s "
                    f"(last step: {last_reported_step[0]})"
                )
                self._log_result(
                    f"           Possible causes: energy monitor halt, SC iterations, or adaptive timestep refinement"
                )

        # Run the integration with progress tracking and hang detection
        self._log_result(f"  [DEBUG] Calling run_testbed for Run {run_num}...")

        # Start a timer thread to check for hangs
        import threading

        hang_check_timer = None

        def periodic_hang_check():
            check_for_hang()
            if self.running:
                hang_check_timer = threading.Timer(10.0, periodic_hang_check)
                hang_check_timer.daemon = True
                hang_check_timer.start()

        hang_check_timer = threading.Timer(10.0, periodic_hang_check)
        hang_check_timer.daemon = True
        hang_check_timer.start()

        try:
            result = run_testbed(options, progress_callback=progress_callback)
            self._log_result(f"  [DEBUG] run_testbed completed for Run {run_num}")
        finally:
            # Cancel hang check timer
            if hang_check_timer:
                hang_check_timer.cancel()

        # Display figures if requested, otherwise close them
        import matplotlib.pyplot as plt

        # Plot display during sweep removed - plots generated only at end
        if False:
            # Show the plots
            for fig in result.figures.values():
                fig.show()
                plt.pause(0.1)  # Allow GUI to update

        # Always close figures after displaying to prevent memory leak
        for fig in result.figures.values():
            plt.close(fig)

        # Clean up temporary run directory
        import shutil

        try:
            if run_output_dir.exists():
                shutil.rmtree(run_output_dir)
        except Exception:
            pass  # Ignore cleanup errors

        # Extract metrics
        metrics = {}
        if result.rider_delta_e is not None:
            metrics["rider_delta_e_mev"] = result.rider_delta_e
        if result.rider_gamma_initial is not None:
            metrics["rider_gamma_initial"] = result.rider_gamma_initial
        if result.rider_gamma_final is not None:
            metrics["rider_gamma_final"] = result.rider_gamma_final

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

            # Only save full trajectory arrays if explicitly requested
            if self.config.save_trajectories:
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
                    }
                except Exception as e:
                    self._log_result(
                        f"    [WARNING] Failed to save trajectory arrays: {e}"
                    )

        return output

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

            heatmap_file = output_dir / "energy_gain_heatmap.png"
            plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

            self._log_result(f"[OK] Heatmap saved to: {heatmap_file}")

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
                    best_result, output_dir / "best_trajectory.png"
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

    def _log_result(self, message: str):
        """Log message to main GUI logs window (thread-safe)."""
        # Log to console/terminal always
        print(f"[OPTIMIZATION] {message}", flush=True)

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
