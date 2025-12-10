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
from examples.validation.core_vs_legacy_benchmark import (
    SimulationType,  # type: ignore[import]
)
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
        parent._log_result(f"❌ {title}: {message}")

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


def _show_info_dialog(parent: tk.Widget, title: str, message: str) -> None:
    """Show an info dialog with selectable text."""
    # Log to console/terminal
    print(f"INFO: {title}: {message}", flush=True)

    # Log to results text if parent is OptimizationPlugin
    if hasattr(parent, "_log_result"):
        parent._log_result(f"ℹ️  {title}: {message}")

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


def _show_warning_dialog(parent: tk.Widget, title: str, message: str) -> None:
    """Show a warning dialog with selectable text."""
    # Log to console/terminal
    print(f"WARNING: {title}: {message}", flush=True)

    # Log to results text if parent is OptimizationPlugin
    if hasattr(parent, "_log_result"):
        parent._log_result(f"⚠️  {title}: {message}")

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

    # Parameter ranges
    aperture_range: Tuple[float, float] = (1e-5, 1e-3)  # mm (10 μm to 1 mm)
    aperture_points: int = 10
    aperture_log_scale: bool = True

    energy_range: Tuple[float, float] = (1.0, 1000.0)  # GeV
    energy_points: int = 10
    energy_log_scale: bool = True

    transverse_offset_fractions: List[float] = None  # Fraction of aperture radius
    starting_z_positions: List[float] = None  # mm before wall

    # Sweepable parameters (can be added to grid)
    transverse_momentum_range: Optional[Tuple[float, float]] = None  # amu·mm/ns
    transverse_momentum_points: int = 1
    timestep_range: Optional[Tuple[float, float]] = None  # ns (proper time)
    timestep_points: int = 1

    # Fixed parameters
    wall_z: float = 100.0  # mm
    steps: int = 2000
    timestep: float = 3e-7  # ns (proper time) - default from main GUI
    auto_steps: bool = False  # Automatically calculate steps based on distance
    auto_steps_target: int = (
        500  # Target number of steps when auto-calculating timestep
    )
    auto_steps_distance_past_wall: float = 10.0  # mm past wall to stop integration
    seed: int = 12345

    # Fixed particle parameters (not swept)
    transv_mom: float = 1.2e-05  # amu·mm/ns
    transv_dist: float = 2e-06  # mm - transverse distance from axis
    m_particle: float = 0.00054857990907  # amu (electron mass)
    pcount: int = 1
    charge_sign: float = -1.0
    stripped_ions: float = 1.0

    # Optimization objective
    objective: str = "max_energy_gain"  # or "max_energy_efficiency", etc.

    # Output
    output_dir: str = "configs/optimization"
    save_results: bool = True
    save_plots: bool = True
    save_trajectories: bool = False  # Save trajectory data for each run
    trajectory_stride: int = 10  # Save every Nth point to reduce file size

    def __post_init__(self):
        """Set defaults for list fields."""
        if self.transverse_offset_fractions is None:
            self.transverse_offset_fractions = [0.1, 0.3, 0.5]
        if self.starting_z_positions is None:
            self.starting_z_positions = [-10.0, -50.0, -100.0]

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
        # The main GUI stores h_step in rider_params
        timestep = rider.get("h_step", 3e-7)

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
            output_dir=str(options.output_dir.parent / "optimization"),
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
    The integrator uses proper time steps h (in ns), but coordinate
    time advance is Δt = γ·h. Distance per step is β·c·Δt = β·c·γ·h.
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
    # distance_per_step = beta * gamma * C_MMNS * timestep
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
    # For ultra-relativistic: beta ≈ 1, so distance ≈ c * gamma * h
    distance_per_step = beta * gamma * C_MMNS * timestep

    # Calculate steps needed (add 10% margin for safety)
    steps = int(np.ceil(total_distance / distance_per_step * 1.1))

    # Ensure minimum reasonable value
    return max(steps, 100)


class OptimizationPlugin(ttk.Frame):
    """Optimization plugin panel for the LW integrator GUI."""

    def __init__(self, parent: tk.Widget, gui_controller=None, **kwargs):
        """Initialize the optimization plugin.

        Parameters
        ----------
        parent : tk.Widget
            Parent widget (typically a notebook tab or frame)
        gui_controller : Optional
            Reference to main GUI controller for run state integration
        """
        super().__init__(parent, **kwargs)
        self.gui_controller = gui_controller
        self.config = OptimizationConfig()
        self.running = False
        self.progress_value = 0.0
        self.progress_text = ""

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
        self._build_parameter_section()
        self._build_objective_section()
        self._build_control_section()
        self._build_progress_section()

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
                command=self._update_driver_visibility,
            )
            rb.grid(row=0, column=i, padx=5, sticky="w")

    def _build_parameter_section(self):
        """Build parameter range specification section."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Parameter Ranges", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)

        # Aperture range
        ttk.Label(frame, text="Aperture Radius:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        aperture_frame = ttk.Frame(frame)
        aperture_frame.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)

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
            row=1, column=0, sticky="w", pady=2
        )
        energy_frame = ttk.Frame(frame)
        energy_frame.grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)

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
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Label(frame, text="Fractions of aperture (comma-separated):").grid(
            row=2, column=1, sticky="w", pady=2
        )
        self.offset_fractions_var = tk.StringVar(value="0.1, 0.3, 0.5")
        ttk.Entry(frame, textvariable=self.offset_fractions_var, width=30).grid(
            row=2, column=2, columnspan=2, sticky="ew", pady=2, padx=5
        )

        # Starting z positions
        ttk.Label(frame, text="Starting Positions:").grid(
            row=3, column=0, sticky="w", pady=2
        )
        ttk.Label(frame, text="z before wall (mm, comma-separated):").grid(
            row=3, column=1, sticky="w", pady=2
        )
        self.start_z_var = tk.StringVar(value="-10, -50, -100")
        ttk.Entry(frame, textvariable=self.start_z_var, width=30).grid(
            row=3, column=2, columnspan=2, sticky="ew", pady=2, padx=5
        )
        ttk.Label(frame, text="Wall Position:").grid(
            row=4, column=0, sticky="w", pady=2
        )
        ttk.Label(frame, text="z (mm):").grid(row=4, column=1, sticky="w", pady=2)
        self.wall_z_var = tk.StringVar(value="100.0")
        ttk.Entry(frame, textvariable=self.wall_z_var, width=10).grid(
            row=4, column=2, sticky="w", pady=2, padx=5
        )

        # Timestep
        ttk.Label(frame, text="Timestep:").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Label(frame, text="h (ns, proper time):").grid(
            row=5, column=1, sticky="w", pady=2
        )
        self.timestep_var = tk.StringVar(value="3e-7")
        ttk.Entry(frame, textvariable=self.timestep_var, width=10).grid(
            row=5, column=2, sticky="w", pady=2, padx=5
        )

        # Steps
        ttk.Label(frame, text="Integration Steps:").grid(
            row=6, column=0, sticky="w", pady=2
        )
        steps_frame = ttk.Frame(frame)
        steps_frame.grid(row=6, column=1, columnspan=3, sticky="ew", pady=2)

        self.steps_var = tk.StringVar(value="2000")
        self.steps_entry = ttk.Entry(steps_frame, textvariable=self.steps_var, width=10)
        self.steps_entry.pack(side="left", padx=(0, 10))

        self.auto_steps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            steps_frame,
            text="Auto-adjust timestep for ~",
            variable=self.auto_steps_var,
            command=self._toggle_auto_steps,
        ).pack(side="left", padx=(0, 2))

        self.auto_steps_target_var = tk.StringVar(value="500")
        ttk.Entry(steps_frame, textvariable=self.auto_steps_target_var, width=6).pack(
            side="left", padx=2
        )
        ttk.Label(steps_frame, text="steps (to wall +").pack(side="left", padx=(2, 2))

        self.auto_steps_distance_var = tk.StringVar(value="10.0")
        ttk.Entry(steps_frame, textvariable=self.auto_steps_distance_var, width=6).pack(
            side="left", padx=2
        )
        ttk.Label(steps_frame, text="mm)").pack(side="left")

        # Trajectory saving
        ttk.Label(frame, text="Trajectory Saving:").grid(
            row=7, column=0, sticky="w", pady=2
        )
        traj_frame = ttk.Frame(frame)
        traj_frame.grid(row=7, column=1, columnspan=3, sticky="ew", pady=2)

        self.save_trajectories_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            traj_frame,
            text="Save trajectories (stride:",
            variable=self.save_trajectories_var,
        ).pack(side="left", padx=(0, 2))

        self.trajectory_stride_var = tk.StringVar(value="10")
        ttk.Entry(traj_frame, textvariable=self.trajectory_stride_var, width=6).pack(
            side="left", padx=2
        )
        ttk.Label(traj_frame, text="points, conservative to reduce file size)").pack(
            side="left"
        )

        frame.columnconfigure(2, weight=1)

        # Initialize auto-steps state
        self._toggle_auto_steps()

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

        # Transverse Distance
        self._add_sweepable_param(
            frame,
            row,
            "rider_transv_dist",
            "Transverse Distance (mm):",
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

    def _toggle_auto_steps(self):
        """Enable/disable steps entry based on auto-adjust checkbox."""
        if self.auto_steps_var.get():
            self.steps_entry.config(state="disabled")
        else:
            self.steps_entry.config(state="normal")

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
            ("Maximize Energy Gain", "max_energy_gain"),
            ("Maximize Energy Efficiency", "max_energy_efficiency"),
            ("Minimize Transverse Deflection", "min_transverse_deflection"),
        ]

        for i, (label, value) in enumerate(objectives):
            rb = ttk.Radiobutton(
                frame, text=label, variable=self.objective_var, value=value
            )
            rb.grid(row=i, column=0, sticky="w", pady=2)

    def _build_control_section(self):
        """Build control buttons section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Controls", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x")

        # Button to load from main config
        self.load_main_config_button = ttk.Button(
            button_frame,
            text="Load from Main Config",
            command=self._on_load_from_main_config,
        )
        self.load_main_config_button.pack(side="left", padx=5)

        self.run_button = ttk.Button(
            button_frame,
            text="▶ Run Sweep",
            command=self._on_run_sweep,
            style="Accent.TButton",
        )
        self.run_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(
            button_frame,
            text="⬛ Stop",
            command=self._on_stop,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=5)

        ttk.Button(
            button_frame, text="📁 Load Config", command=self._on_load_config
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame, text="💾 Save Config", command=self._on_save_config
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame, text="📊 View Results", command=self._on_view_results
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame,
            text="📈 Plot Trajectories",
            command=self._on_plot_trajectories,
        ).pack(side="left", padx=5)

        # Fine-tune controls in a second row
        finetune_frame = ttk.Frame(frame)
        finetune_frame.pack(fill="x", pady=(5, 0))

        self.auto_finetune_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            finetune_frame,
            text="Auto-prompt for fine-tuning after coarse sweep",
            variable=self.auto_finetune_var,
        ).pack(side="left", padx=5)

    def _build_progress_section(self):
        """Build progress monitoring section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Progress", padding=10)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            frame, mode="determinate", maximum=100, length=400
        )
        self.progress_bar.pack(fill="x", pady=5)

        # Progress label
        self.progress_label = ttk.Label(frame, text="Ready")
        self.progress_label.pack(anchor="w", pady=2)

        # Results summary (text area with scrollbar)
        ttk.Label(frame, text="Results Summary:").pack(anchor="w", pady=(10, 2))

        # Create a frame to hold text and scrollbar
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill="both", expand=True, pady=2)

        self.results_text = tk.Text(
            text_frame, height=20, width=70, wrap="word", state="disabled"
        )
        self.results_text.pack(side="left", fill="both", expand=True)

        # Scrollbar for results
        scrollbar = ttk.Scrollbar(text_frame, command=self.results_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.results_text.configure(yscrollcommand=scrollbar.set)

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

            # Auto-steps parameters
            if self.auto_steps_var.get():
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
            timestep=float(self.timestep_var.get()),
            steps=int(self.steps_var.get()),
            auto_steps=self.auto_steps_var.get(),
            auto_steps_target=int(self.auto_steps_target_var.get()),
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
            self.timestep_var.set(f"{opt_config.timestep:.2e}")
            self.steps_var.set(str(opt_config.steps))
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

            self._log_result("✓ Loaded parameters from main GUI configuration")
            self._log_result(f"  Simulation type: {opt_config.simulation_type.name}")
            self._log_result(f"  Wall z: {opt_config.wall_z} mm")
            self._log_result(f"  Timestep: {opt_config.timestep:.2e} ns")
            self._log_result(f"  Steps: {opt_config.steps}")
            self._log_result(f"  Particle mass: {opt_config.m_particle:.6e} amu")
            self._log_result(
                f"  Transverse momentum: {opt_config.transv_mom:.2e} amu·mm/ns"
            )
            self._log_result(f"  Transverse distance: {opt_config.transv_dist:.2e} mm")
            self._log_result("")

        except Exception as e:
            _show_error_dialog(
                self,
                "Load Config Error",
                f"Failed to load configuration from main GUI:\n{e}",
            )
            import traceback

            self._log_result(f"❌ Error loading main config: {e}")
            self._log_result(traceback.format_exc())

    def _on_run_sweep(self):
        """Handle run sweep button click."""
        # Check if main GUI is already running
        if self.gui_controller and hasattr(self.gui_controller, "_running"):
            if self.gui_controller._running:
                _show_info_dialog(
                    self,
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
        except Exception as e:
            _show_error_dialog(self, "Configuration Error", str(e))
            return

        # Update UI state
        self.running = True
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
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

    def _on_load_config(self):
        """Load configuration from JSON file."""
        import os

        # Default to configs/optimization directory
        default_dir = "configs/optimization"
        os.makedirs(default_dir, exist_ok=True)

        filename = filedialog.askopenfilename(
            title="Load Optimization Config",
            initialdir=default_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return

        try:
            with open(filename, "r") as f:
                data = json.load(f)

            # Populate UI fields
            self.sim_type_var.set(data.get("simulation_type", "CONDUCTING_WALL"))
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
                ", ".join(map(str, data.get("starting_z_positions", [-10, -50, -100])))
            )
            self.wall_z_var.set(str(data.get("wall_z", 100.0)))
            self.steps_var.set(str(data.get("steps", 2000)))
            self.objective_var.set(data.get("objective", "max_energy_gain"))

            self._log_result("✓ Configuration loaded successfully")
        except Exception as e:
            _show_error_dialog(self, "Load Error", f"Failed to load config: {e}")

    def _on_save_config(self):
        """Save configuration to JSON file."""
        error = self._validate_inputs()
        if error:
            _show_error_dialog(self, "Invalid Input", f"Cannot save: {error}")
            return

        import os

        # Default to configs/optimization directory
        default_dir = "configs/optimization"
        os.makedirs(default_dir, exist_ok=True)

        filename = filedialog.asksaveasfilename(
            title="Save Optimization Config",
            initialdir=default_dir,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not filename:
            return

        try:
            config = self._gather_config()
            data = {
                "simulation_type": config.simulation_type.name,
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
                "steps": config.steps,
                "objective": config.objective,
            }

            with open(filename, "w") as f:
                json.dump(data, f, indent=2)

            self._log_result(f"✓ Configuration saved to {filename}")
        except Exception as e:
            _show_error_dialog(self, "Save Error", f"Failed to save config: {e}")

    def _on_view_results(self):
        """Open results viewer (placeholder for now)."""
        # Create custom dialog with selectable text
        dialog = tk.Toplevel(self)
        dialog.title("View Results")
        dialog.geometry("600x400")

        # Make it modal
        dialog.transient(self)
        dialog.grab_set()

        # Message
        msg_frame = ttk.Frame(dialog, padding=20)
        msg_frame.pack(fill="both", expand=True)

        ttk.Label(
            msg_frame,
            text="Results Visualization Coming Soon!",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        # Selectable text area
        text_widget = tk.Text(msg_frame, wrap="word", height=15, width=70)
        text_widget.pack(fill="both", expand=True, pady=10)

        info_text = """For now, check the output directory for:

• results.json - Numerical data from parameter sweep
• heatmap.png - Parameter sweep visualization
• summary_plots/ - Detailed analysis plots

Note: The optimization sweep backend is still in development.
Once connected, this viewer will show:
  - Interactive heatmaps (aperture vs energy)
  - Energy gain distributions
  - Pareto fronts for multi-objective optimization
  - Trajectory visualizations for optimal configurations
  - Export options (CSV, HDF5, matplotlib figures)

Output directory: """ + str(self.config.output_dir)

        text_widget.insert("1.0", info_text)
        text_widget.configure(state="disabled")

        # Close button
        ttk.Button(
            msg_frame,
            text="Close",
            command=dialog.destroy,
        ).pack(pady=(10, 0))

        # Center on parent
        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def _on_plot_trajectories(self):
        """Open trajectory plotting dialog to visualize saved results."""
        # Ask user to select sweep_results.json file
        file_path = filedialog.askopenfilename(
            title="Select sweep_results.json",
            initialdir=self.config.output_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

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

            # Create trajectory viewer dialog
            self._show_trajectory_viewer(results_with_traj, file_path)

        except Exception as e:
            _show_error_dialog(self, "Error Loading File", f"Failed to load file:\n{e}")

    def _show_trajectory_viewer(self, results, file_path):
        """Show trajectory viewer dialog with run selection and plotting."""
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
                f"ΔE={delta_e:.2f}MeV"
            )
            run_listbox.insert("end", summary)

        # Control buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=(10, 0))

        plot_button = ttk.Button(
            btn_frame,
            text="📈 Plot Selected",
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

    def _plot_selected_trajectories(self, listbox, results, parent_dialog):
        """Plot trajectories for selected runs."""
        selection = listbox.curselection()
        if not selection:
            _show_info_dialog(
                self, "No Selection", "Please select at least one run to plot."
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

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle(
                f"Trajectories for {len(selected_results)} run(s)", fontsize=12
            )

            ax_z = axes[0, 0]
            ax_r = axes[0, 1]
            ax_pz = axes[1, 0]
            ax_pr = axes[1, 1]

            # Plot each selected trajectory
            for idx, result in enumerate(selected_results):
                traj = result.get("trajectory", {})
                params = result.get("parameters", {})
                run_num = result.get("run_number", "?")

                z = np.array(traj.get("z", []))
                r = np.array(traj.get("r", []))
                pz = np.array(traj.get("pz", []))
                pr = np.array(traj.get("pr", []))
                t = np.array(traj.get("t", []))

                if len(z) == 0:
                    continue

                label = f"Run #{run_num}"
                color = plt.cm.tab10(idx % 10)

                # z vs t
                ax_z.plot(t, z, label=label, alpha=0.7, color=color)

                # r vs z (scatter for transverse)
                ax_r.scatter(z, r, label=label, alpha=0.5, s=10, color=color)

                # pz vs z
                ax_pz.plot(z, pz, label=label, alpha=0.7, color=color)

                # pr vs r (scatter for transverse)
                ax_pr.scatter(r, pr, label=label, alpha=0.5, s=10, color=color)

            # Set labels and legends
            ax_z.set_xlabel("t (ns)")
            ax_z.set_ylabel("z (mm)")
            ax_z.legend(fontsize=8)
            ax_z.grid(True, alpha=0.3)

            ax_r.set_xlabel("z (mm)")
            ax_r.set_ylabel("r (mm)")
            ax_r.legend(fontsize=8)
            ax_r.grid(True, alpha=0.3)

            ax_pz.set_xlabel("z (mm)")
            ax_pz.set_ylabel("pz (amu·mm/ns)")
            ax_pz.legend(fontsize=8)
            ax_pz.grid(True, alpha=0.3)

            ax_pr.set_xlabel("r (mm)")
            ax_pr.set_ylabel("pr (amu·mm/ns)")
            ax_pr.legend(fontsize=8)
            ax_pr.grid(True, alpha=0.3)

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

    def _run_sweep_background(self, is_finetune=False, finetune_regions=None):
        """Run parameter sweep in background with real integration.

        Args:
            is_finetune: If True, this is a fine-tuning sweep
            finetune_regions: List of parameter regions for fine-tuning
        """
        try:
            # Generate parameter grid including sweepable parameters
            param_grids = self._generate_parameter_grids()

            # Calculate total runs
            total_runs = 1
            for values in param_grids.values():
                total_runs *= len(values)

            self._log_result(f"Starting parameter sweep: {total_runs} total runs")

            # Log parameter grid info
            for param_name, values in param_grids.items():
                if len(values) > 1:
                    self._log_result(
                        f"  {param_name}: {len(values)} points from {values[0]:.2e} to {values[-1]:.2e}"
                    )
                else:
                    self._log_result(f"  {param_name}: {values[0]:.2e} (fixed)")
            self._log_result(
                f"  Timestep mode: {'Auto-adjust' if self.config.auto_steps else 'Fixed'}"
            )
            if self.config.auto_steps:
                self._log_result(
                    f"    Target ~{self.config.auto_steps_target} steps to wall + {self.config.auto_steps_distance_past_wall} mm"
                )

            self._log_result("")

            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            self._log_result(f"Output directory: {self.config.output_dir}")
            self._log_result("")

            # Store all results
            all_results = []
            run_num = 0

            # Create parameter combinations using itertools
            import itertools

            param_names = list(param_grids.keys())
            param_values_lists = [param_grids[name] for name in param_names]

            for param_combo in itertools.product(*param_values_lists):
                # Check for cancellation
                if not self.running:
                    self._log_result("❌ Sweep stopped by user")
                    break

                if self.gui_controller and hasattr(
                    self.gui_controller, "_cancel_requested"
                ):
                    if self.gui_controller._cancel_requested:
                        self._log_result("❌ Sweep cancelled by user")
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

                # Auto-adjust timestep if enabled
                if self.config.auto_steps:
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

                # Run integration
                try:
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
                        driver_params=driver_params_dict,
                    )

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
                            {f"driver_{k}": v for k, v in driver_params_dict.items()}
                        )

                    all_results.append(run_data)

                except Exception as e:
                    self._log_result(f"⚠️  Run {run_num} failed: {e}")
                    all_results.append(
                        {
                            "run_number": run_num,
                            "parameters": {
                                "aperture_radius": aperture,
                                "particle_energy_gev": energy,
                                "start_z": start_z,
                                "transverse_offset": transv_offset,
                            },
                            "error": str(e),
                        }
                    )

            # Save results
            if all_results and self.config.save_results:
                self._save_sweep_results(all_results)

            if self.running:
                self._log_result("✓ Sweep completed successfully!")
                self._log_result(f"  Results saved to: {self.config.output_dir}")
                self._log_result(f"  Total runs: {len(all_results)}")
                self._update_progress(100, "Complete!")

                # Offer fine-tuning if enabled and not already a fine-tune run
                if not is_finetune and self.auto_finetune_var.get() and all_results:
                    self.after(500, lambda: self._prompt_finetune(all_results))
        except Exception as e:
            self._log_result(f"❌ Error during sweep: {e}")
            import traceback

            self._log_result(traceback.format_exc())
        finally:
            self.running = False
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

    def _prompt_finetune(self, coarse_results):
        """Prompt user for fine-tuning after coarse sweep."""
        # Find top results
        results_with_energy = [
            (r, r.get("metrics", {}).get("rider_delta_e_mev", float("-inf")))
            for r in coarse_results
        ]
        results_with_energy.sort(key=lambda x: x[1], reverse=True)

        top_n = min(5, len(results_with_energy))
        top_results = [r[0] for r in results_with_energy[:top_n]]

        if not top_results:
            return

        # Create dialog
        response = messagebox.askyesno(
            "Fine-Tuning",
            f"Coarse sweep complete!\n\n"
            f"Found {top_n} promising configurations.\n"
            f"Best energy gain: {results_with_energy[0][1]:.2f} MeV\n\n"
            f"Would you like to run a fine-tuning sweep around these optima?",
        )

        if response:
            self._log_result("\n" + "=" * 60)
            self._log_result("Starting fine-tuning sweep...")
            self._log_result("=" * 60 + "\n")
            # TODO: Implement fine-tuning logic
            _show_info_dialog(
                self,
                "Fine-Tuning",
                "Fine-tuning feature coming soon!\n\n"
                "This will automatically refine the parameter space\n"
                "around the best configurations found.",
            )

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
        # Pz ≈ gamma * m * c for ultra-relativistic
        # In units of amu*mm/ns: Pz = gamma * m * c_mmns
        rider_params["starting_Pz"] = gamma * rider_m_particle * C_MMNS

        core_params = {
            "time_step": timestep,
            "wall_z": self.config.wall_z,
            "aperture_radius": aperture,
            "mean": 1.0e5,  # Large value (not used for CONDUCTING_WALL)
            "cav_spacing": 1.0e5,
            "z_cutoff": 0.0,
        }

        options = SimulationOptions(
            steps=steps,
            seed=self.config.seed,
            simulation_type=self.config.simulation_type,
            rider_params=rider_params,
            driver_params=driver_params,  # Use provided driver params (None for CONDUCTING_WALL)
            core_params=core_params,
            legacy_enabled=False,
            trajectory_save=self.config.save_trajectories,
            trajectory_interval=self.config.trajectory_stride,
            energy_display=False,
            energy_save=False,
            transverse_display=False,
            transverse_save=False,
            overlay_display=False,
            overlay_save=False,
            difference_display=False,
            difference_save=False,
            metrics_save=False,
            output_dir=Path(self.config.output_dir),
        )

        # Run the integration
        result = run_testbed(options)

        # Extract metrics
        metrics = {}
        if result.rider_delta_e is not None:
            metrics["rider_delta_e_mev"] = result.rider_delta_e
        if result.rider_gamma_initial is not None:
            metrics["rider_gamma_initial"] = result.rider_gamma_initial
        if result.rider_gamma_final is not None:
            metrics["rider_gamma_final"] = result.rider_gamma_final

        output = {"metrics": metrics}

        # Add trajectory if requested and available
        if self.config.save_trajectories and result.rider_trajectory is not None:
            traj = result.rider_trajectory
            # Downsample trajectory
            stride = self.config.trajectory_stride
            output["trajectory"] = {
                "z": traj.z[::stride].tolist(),
                "r": traj.r[::stride].tolist(),
                "pz": traj.pz[::stride].tolist(),
                "pr": traj.pr[::stride].tolist(),
                "t": traj.t[::stride].tolist(),
            }

        return output

    def _save_sweep_results(self, results: List[Dict[str, Any]]) -> None:
        """Save sweep results to JSON file."""
        output_file = Path(self.config.output_dir) / "sweep_results.json"

        # Prepare data for JSON serialization
        output_data = {
            "config": {
                "aperture_range": self.config.aperture_range,
                "aperture_points": self.config.aperture_points,
                "energy_range": self.config.energy_range,
                "energy_points": self.config.energy_points,
                "transverse_offset_fractions": self.config.transverse_offset_fractions,
                "starting_z_positions": self.config.starting_z_positions,
                "simulation_type": self.config.simulation_type.name,
            },
            "results": results,
            "total_runs": len(results),
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        self._log_result(f"Results saved to: {output_file}")

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
        """Append message to results text area (thread-safe)."""

    def _log_result(self, message: str):
        """Append message to results text area (thread-safe)."""

        def append():
            self.results_text.config(state="normal")
            self.results_text.insert("end", message + "\n")
            self.results_text.see("end")
            self.results_text.config(state="disabled")

        self.after(0, append)

    def _reset_ui_state(self):
        """Reset UI to ready state after run completes."""
        self.run_button.config(state="normal")
        self.stop_button.config(state="disabled")

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
