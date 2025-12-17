"""Helper utilities mirroring the notebook testbed inside a desktop GUI.

The original ``integrator_testbed.ipynb`` notebook wires dozens of ipywidgets
around the benchmarking helpers in ``examples/validation/core_vs_legacy_benchmark``.
This module repackages that behaviour behind plain Python data structures so a
Tkinter GUI (or any other front-end) can drive the same workflows: generating
energy plots, exporting metrics, saving down-sampled trajectories, and managing
JSON snapshot files.

All strings deliberately use ASCII to keep packaging simple when rendered in
terminals that do not default to UTF-8.
"""

from __future__ import annotations

import json
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from core.constants import C_MMNS
from core.particle_config import (
    DEFAULT_DRIVER_PARAMS,
    DEFAULT_RIDER_PARAMS,
    PARTICLE_PARAM_FIELDS,
)
from core.types import SimulationType
from examples.validation.core_vs_legacy_benchmark import (  # type: ignore[import]
    compute_delta_energy_components,
    compute_delta_energy_series,
    prepare_two_particle_demo,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Constants mirroring the notebook defaults
# ---------------------------------------------------------------------------

COLOR_RIDER = "#0072B2"
COLOR_DRIVER = "#D55E00"
COLOR_LEGACY_RIDER = "#56B4E9"
COLOR_LEGACY_DRIVER = "#E69F00"
COLOR_DIFF_RIDER = "#009E73"
COLOR_DIFF_DRIVER = "#CC79A7"
SCATTER_STYLE = {"s": 140, "alpha": 0.78, "linewidth": 0, "edgecolors": "none"}
AVAILABLE_DPI_CHOICES: Tuple[int, ...] = (150, 300, 450, 600)
DEFAULT_PLOT_DPI = 300

PARAM_LABELS: Dict[str, str] = {
    "starting_distance": "Start z (mm)",
    "transv_mom": "Transverse momentum (amu*mm/ns)",
    "starting_Pz": "Initial Pz (amu*mm/ns)",
    "stripped_ions": "Stripped ions",
    "m_particle": "Mass (amu)",
    "transv_dist": "Transverse spread (mm)",
    "pcount": "Particle count",
    "charge_sign": "Charge sign",
}

CORE_PARAM_LABELS: Dict[str, str] = {
    "time_step": "Time step (ns)",
    "wall_z": "Wall z (mm)",
    "aperture_radius": "Aperture radius (mm)",
    "mean": "Mean separation (mm)",
    "cav_spacing": "Cavity spacing (mm)",
    "z_cutoff": "z cutoff (mm)",
    "z_cutoff_mode": "z cutoff mode",
}

CORE_PARAM_DEFAULTS: Dict[str, Any] = {
    "time_step": 2.2e-7,
    "wall_z": 1.0e5,
    "aperture_radius": 1.0e5,
    "mean": 1.0e5,
    "cav_spacing": 1.0e5,
    "z_cutoff": 0.0,
    "z_cutoff_mode": "absolute",
}

CORE_REQUIRED_PARAMS: Dict[SimulationType, set[str]] = {
    SimulationType.CONDUCTING_WALL: {"time_step", "wall_z", "aperture_radius"},
    SimulationType.SWITCHING_WALL: {
        "time_step",
        "wall_z",
        "aperture_radius",
        "cav_spacing",
        "z_cutoff",
    },
    SimulationType.BUNCH_TO_BUNCH: {
        "time_step",
        "aperture_radius",
        "z_cutoff",
        "z_cutoff_mode",
    },
}

SPECIES_PRESETS: Dict[str, Optional[Dict[str, float]]] = {
    "custom": None,
    "electron": {
        "m_particle": 5.48579909070e-4,
        "charge_sign": -1.0,
        "stripped_ions": 1.0,
    },
    "positron": {
        "m_particle": 5.48579909070e-4,
        "charge_sign": 1.0,
        "stripped_ions": 1.0,
    },
    "proton": {
        "m_particle": 1.007276466621,
        "charge_sign": 1.0,
        "stripped_ions": 1.0,
    },
    "antiproton": {
        "m_particle": 1.007276466621,
        "charge_sign": -1.0,
        "stripped_ions": 1.0,
    },
    "lead": {
        "m_particle": 207.9766521,
        "charge_sign": 1.0,
        "stripped_ions": 82.0,
    },
    "gold": {
        "m_particle": 196.9665687,
        "charge_sign": 1.0,
        "stripped_ions": 79.0,
    },
}

SPECIES_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Custom / manual", "custom"),
    ("Electron (e-)", "electron"),
    ("Positron (e+)", "positron"),
    ("Proton (p+)", "proton"),
    ("Antiproton (pbar)", "antiproton"),
    ("Lead ion (Pb^82+)", "lead"),
    ("Gold ion (Au^79+)", "gold"),
)

_TIMESTAMP_TOKEN_LENGTH = 15

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }
)

# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


class TeeStringIO(StringIO):
    """StringIO that also writes to another stream (like sys.stdout).

    This allows capturing output while also displaying it in real-time.
    """

    def __init__(self, tee_stream=None):
        super().__init__()
        self.tee_stream = tee_stream

    def write(self, s):
        """Write to both StringIO buffer and tee stream."""
        result = super().write(s)
        if self.tee_stream is not None:
            try:
                self.tee_stream.write(s)
                self.tee_stream.flush()
            except Exception:
                # Ignore errors writing to tee stream
                pass
        return result


# ---------------------------------------------------------------------------
# Dataclasses for strongly typed options/results
# ---------------------------------------------------------------------------


@dataclass
class SimulationOptions:
    """Full parameter surface matching the notebook widget snapshot."""

    steps: int = 1000
    seed: int = 12345
    simulation_type: SimulationType = SimulationType.BUNCH_TO_BUNCH
    legacy_enabled: bool = False
    overlay_display: bool = False
    overlay_save: bool = False
    difference_display: bool = False
    difference_save: bool = False
    metrics_save: bool = False
    energy_display: bool = True
    energy_save: bool = True
    energy_xaxis: str = "z"  # "z", "t", or "dual"
    energy_yaxis: str = (
        "delta_total"  # "delta_total", "delta_z", "delta_x", "delta_y", "total"
    )
    transverse_display: bool = False
    transverse_save: bool = False
    transverse_xaxis: str = "t"  # "t" or "z"
    beta_display: bool = False
    beta_save: bool = False
    beta_xaxis: str = "t"  # "t" or "z"
    momentum_display: bool = False
    momentum_save: bool = False
    momentum_xaxis: str = "t"  # "t" or "z"
    zposition_display: bool = False
    zposition_save: bool = False
    trajectory_save: bool = False
    trajectory_interval: int = 10
    plot_dpi: int = DEFAULT_PLOT_DPI
    output_dir: Path = Path("test_outputs/testbed_runs")
    config_dir: Path = Path("configs/testbed_runs")
    config_name: str = "testbed_config.json"
    rider_params: Dict[str, float | int] = field(
        default_factory=lambda: dict(DEFAULT_RIDER_PARAMS)
    )
    driver_params: Optional[Dict[str, float | int]] = field(
        default_factory=lambda: dict(DEFAULT_DRIVER_PARAMS)
    )
    core_params: Dict[str, float | str] = field(
        default_factory=lambda: {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in CORE_PARAM_DEFAULTS.items()
        }
    )
    image_subcharge_count: int = 12
    use_image_weighting: bool = True

    # Self-consistency options
    self_consistency_enabled: bool = True
    self_consistency_tolerance: float = (
        1e-4  # Legacy parameter for backward compatibility
    )
    self_consistency_convergence_mode: str = "fixed_geometry"  # or "variable_geometry"
    self_consistency_target_ms_tolerance: float = 1e-6  # Mass-shell loop criterion
    self_consistency_max_iterations: int = (
        10  # Maximum SC iterations per particle per step
    )
    self_consistency_mass_shell_tolerance: float = (
        1e-2  # Safety net threshold enforced after loop
    )
    self_consistency_mass_shell_relaxation: float = (
        0.7  # Relaxation weight for Pt correction (0.0-1.0, default 0.7)
    )
    self_consistency_verbosity: int = (
        0  # 0=silent, 1=basic, 2=detailed (prints to console and saved logs)
    )

    # Energy monitoring options
    energy_monitor_enabled: bool = True
    energy_monitor_threshold: float = 2.0
    energy_monitor_check_interval: int = 10
    energy_monitor_halt_on_jump: bool = False
    energy_monitor_debug: bool = False

    # Adaptive timestep options
    adaptive_timestep_enabled: bool = True
    adaptive_timestep_threshold: float = 0.10
    adaptive_timestep_reduction_factor: int = 10
    adaptive_timestep_max_attempts: int = 5
    adaptive_timestep_min_factor: float = 1e-4

    # Adaptive timestep hysteresis (stay on reduced timestep for stability)
    adaptive_timestep_cooldown_steps: int = 10
    adaptive_timestep_probe_threshold: float = 0.01
    adaptive_timestep_max_probe_steps: int = 3

    adaptive_timestep_debug: bool = False

    # Logging options
    save_log_file: bool = False
    log_file_path: Optional[str] = None  # If None, auto-generate in output_dir

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "steps": self.steps,
            "seed": self.seed,
            "simulation_type": self.simulation_type.name,
            "legacy_enabled": self.legacy_enabled,
            "overlay_display": self.overlay_display,
            "overlay_save": self.overlay_save,
            "difference_display": self.difference_display,
            "difference_save": self.difference_save,
            "metrics_save": self.metrics_save,
            "energy_display": self.energy_display,
            "energy_save": self.energy_save,
            "energy_xaxis": self.energy_xaxis,
            "energy_yaxis": self.energy_yaxis,
            "transverse_display": self.transverse_display,
            "transverse_save": self.transverse_save,
            "transverse_xaxis": self.transverse_xaxis,
            "beta_display": self.beta_display,
            "beta_save": self.beta_save,
            "beta_xaxis": self.beta_xaxis,
            "momentum_display": self.momentum_display,
            "momentum_save": self.momentum_save,
            "momentum_xaxis": self.momentum_xaxis,
            "zposition_display": self.zposition_display,
            "zposition_save": self.zposition_save,
            "trajectory_save": self.trajectory_save,
            "trajectory_interval": self.trajectory_interval,
            "plot_dpi": self.plot_dpi,
            "output_dir": str(self.output_dir),
            "config_dir": str(self.config_dir),
            "config_name": self.config_name,
            "rider_params": dict(self.rider_params),
            "driver_params": dict(self.driver_params) if self.driver_params else None,
            "core_params": dict(self.core_params),
            "image_subcharge_count": self.image_subcharge_count,
            "use_image_weighting": self.use_image_weighting,
            "self_consistency_enabled": self.self_consistency_enabled,
            "self_consistency_tolerance": self.self_consistency_tolerance,
            "self_consistency_convergence_mode": self.self_consistency_convergence_mode,
            "self_consistency_target_ms_tolerance": self.self_consistency_target_ms_tolerance,
            "self_consistency_max_iterations": self.self_consistency_max_iterations,
            "self_consistency_mass_shell_tolerance": self.self_consistency_mass_shell_tolerance,
            "self_consistency_mass_shell_relaxation": self.self_consistency_mass_shell_relaxation,
            "self_consistency_verbosity": self.self_consistency_verbosity,
            "energy_monitor_enabled": self.energy_monitor_enabled,
            "energy_monitor_threshold": self.energy_monitor_threshold,
            "energy_monitor_check_interval": self.energy_monitor_check_interval,
            "energy_monitor_halt_on_jump": self.energy_monitor_halt_on_jump,
            "energy_monitor_debug": self.energy_monitor_debug,
            "adaptive_timestep_enabled": self.adaptive_timestep_enabled,
            "adaptive_timestep_threshold": self.adaptive_timestep_threshold,
            "adaptive_timestep_reduction_factor": self.adaptive_timestep_reduction_factor,
            "adaptive_timestep_max_attempts": self.adaptive_timestep_max_attempts,
            "adaptive_timestep_min_factor": self.adaptive_timestep_min_factor,
            "adaptive_timestep_cooldown_steps": self.adaptive_timestep_cooldown_steps,
            "adaptive_timestep_probe_threshold": self.adaptive_timestep_probe_threshold,
            "adaptive_timestep_max_probe_steps": self.adaptive_timestep_max_probe_steps,
            "adaptive_timestep_debug": self.adaptive_timestep_debug,
            "save_log_file": self.save_log_file,
            "log_file_path": self.log_file_path,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SimulationOptions":
        def _bool(name: str, default: bool) -> bool:
            return bool(payload.get(name, default))

        def _int(name: str, default: int) -> int:
            value = payload.get(name, default)
            try:
                return int(value)  # type: ignore[arg-type,no-any-return,call-overload]
            except (TypeError, ValueError):
                return default

        def _float(name: str, default: float) -> float:
            value = payload.get(name, default)
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        sim_value = payload.get("simulation_type", "BUNCH_TO_BUNCH")
        if isinstance(sim_value, SimulationType):
            simulation_type = sim_value
        elif isinstance(sim_value, int):
            simulation_type = SimulationType(sim_value)
        elif isinstance(sim_value, str) and hasattr(SimulationType, sim_value):
            simulation_type = getattr(SimulationType, sim_value)
        else:
            simulation_type = SimulationType.BUNCH_TO_BUNCH

        rider_params = dict(DEFAULT_RIDER_PARAMS)
        rider_payload = payload.get("rider_params")
        if isinstance(rider_payload, dict):
            rider_params.update(rider_payload)

        driver_params: Optional[Dict[str, float | int]]
        driver_payload = payload.get("driver_params")
        if isinstance(driver_payload, dict):
            driver_params = dict(driver_payload)
        else:
            driver_params = dict(DEFAULT_DRIVER_PARAMS)

        core_params = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in CORE_PARAM_DEFAULTS.items()
        }
        core_payload = payload.get("core_params")
        if isinstance(core_payload, dict):
            for key, val in core_payload.items():
                # Handle both numeric and string values (e.g., z_cutoff_mode)
                if isinstance(val, str):
                    core_params[key] = val
                else:
                    try:
                        core_params[key] = float(val)
                    except (TypeError, ValueError):
                        continue

        options = cls(
            steps=_int("steps", 1000),
            seed=_int("seed", 12345),
            simulation_type=simulation_type,
            legacy_enabled=_bool("legacy_enabled", False),
            overlay_display=_bool("overlay_display", False),
            overlay_save=_bool("overlay_save", False),
            difference_display=_bool("difference_display", False),
            difference_save=_bool("difference_save", False),
            metrics_save=_bool("metrics_save", False),
            energy_display=_bool("energy_display", True),
            energy_save=_bool("energy_save", True),
            energy_xaxis=str(payload.get("energy_xaxis", "z")),
            energy_yaxis=str(payload.get("energy_yaxis", "delta_total")),
            transverse_display=_bool("transverse_display", False),
            transverse_save=_bool("transverse_save", False),
            transverse_xaxis=str(payload.get("transverse_xaxis", "t")),
            beta_display=_bool("beta_display", False),
            beta_save=_bool("beta_save", False),
            beta_xaxis=str(payload.get("beta_xaxis", "t")),
            momentum_display=_bool("momentum_display", False),
            momentum_save=_bool("momentum_save", False),
            momentum_xaxis=str(payload.get("momentum_xaxis", "t")),
            zposition_display=_bool("zposition_display", False),
            zposition_save=_bool("zposition_save", False),
            trajectory_save=_bool("trajectory_save", False),
            trajectory_interval=_int("trajectory_interval", 10),
            plot_dpi=_int("plot_dpi", DEFAULT_PLOT_DPI),
            output_dir=Path(
                str(payload.get("output_dir", "test_outputs/testbed_runs"))
            ),
            config_dir=Path(str(payload.get("config_dir", "configs/testbed_runs"))),
            config_name=str(payload.get("config_name", "testbed_config.json")),
            rider_params=rider_params,
            driver_params=driver_params,
            core_params=core_params,
            image_subcharge_count=_int("image_subcharge_count", 12),
            use_image_weighting=_bool("use_image_weighting", True),
            self_consistency_enabled=_bool("self_consistency_enabled", True),
            self_consistency_tolerance=_float("self_consistency_tolerance", 1e-4),
            self_consistency_convergence_mode=str(
                payload.get("self_consistency_convergence_mode", "mass_shell_only")
            ),
            self_consistency_target_ms_tolerance=_float(
                "self_consistency_target_ms_tolerance", 1e-6
            ),
            self_consistency_max_iterations=_int("self_consistency_max_iterations", 10),
            self_consistency_mass_shell_tolerance=_float(
                "self_consistency_mass_shell_tolerance", 1e-2
            ),
            self_consistency_mass_shell_relaxation=_float(
                "self_consistency_mass_shell_relaxation", 0.7
            ),
            self_consistency_verbosity=_int("self_consistency_verbosity", 0),
            energy_monitor_enabled=_bool("energy_monitor_enabled", True),
            energy_monitor_threshold=_float("energy_monitor_threshold", 2.0),
            energy_monitor_check_interval=_int("energy_monitor_check_interval", 10),
            energy_monitor_halt_on_jump=_bool("energy_monitor_halt_on_jump", False),
            energy_monitor_debug=_bool("energy_monitor_debug", False),
            adaptive_timestep_enabled=_bool("adaptive_timestep_enabled", True),
            adaptive_timestep_threshold=_float("adaptive_timestep_threshold", 0.10),
            adaptive_timestep_reduction_factor=_int(
                "adaptive_timestep_reduction_factor", 10
            ),
            adaptive_timestep_max_attempts=_int("adaptive_timestep_max_attempts", 5),
            adaptive_timestep_min_factor=_float("adaptive_timestep_min_factor", 1e-4),
            adaptive_timestep_cooldown_steps=_int(
                "adaptive_timestep_cooldown_steps", 10
            ),
            adaptive_timestep_probe_threshold=_float(
                "adaptive_timestep_probe_threshold", 0.01
            ),
            adaptive_timestep_max_probe_steps=_int(
                "adaptive_timestep_max_probe_steps", 3
            ),
            adaptive_timestep_debug=_bool("adaptive_timestep_debug", False),
            save_log_file=_bool("save_log_file", False),
            log_file_path=str(payload.get("log_file_path"))
            if payload.get("log_file_path") is not None
            else None,
        )
        return options


@dataclass
class InitialSummary:
    seed: int
    rider_gamma: float
    rider_rest_mev: float
    rider_rest_gev: float
    rider_total_gev: float
    driver_gamma: Optional[float]
    driver_rest_mev: Optional[float]
    driver_rest_gev: Optional[float]
    driver_total_gev: Optional[float]
    supports_driver: bool
    # Beam optics parameters
    rider_emittance_x_mm_mrad: Optional[float] = None
    rider_emittance_y_mm_mrad: Optional[float] = None
    rider_norm_emittance_x_mm_mrad: Optional[float] = None
    rider_norm_emittance_y_mm_mrad: Optional[float] = None
    rider_beta_x_m: Optional[float] = None
    rider_beta_y_m: Optional[float] = None
    driver_emittance_x_mm_mrad: Optional[float] = None
    driver_emittance_y_mm_mrad: Optional[float] = None
    driver_norm_emittance_x_mm_mrad: Optional[float] = None
    driver_norm_emittance_y_mm_mrad: Optional[float] = None
    driver_beta_x_m: Optional[float] = None
    driver_beta_y_m: Optional[float] = None

    @property
    def has_driver(self) -> bool:
        return self.supports_driver and self.driver_gamma is not None


def compute_beam_optics(state: Dict[str, np.ndarray], gamma: float) -> Dict[str, float]:
    """Calculate emittance and Twiss beta from particle state.

    Parameters
    ----------
    state : dict
        Particle state with keys 'x', 'y', 'Px', 'Py', 'Pz', 'm', etc.
    gamma : float
        Lorentz factor

    Returns
    -------
    dict
        Dictionary with keys:
        - emittance_x_mm_mrad: geometric emittance in x (mm·mrad)
        - emittance_y_mm_mrad: geometric emittance in y (mm·mrad)
        - norm_emittance_x_mm_mrad: normalized emittance in x (mm·mrad)
        - norm_emittance_y_mm_mrad: normalized emittance in y (mm·mrad)
        - beta_x_m: Twiss beta in x (meters)
        - beta_y_m: Twiss beta in y (meters)

    Notes
    -----
    Units: amu·mm/ns system
    - Position: mm
    - Momentum: amu·mm/ns
    - Angle x' = Px/Pz (dimensionless for paraxial beams)
    - Geometric emittance: sqrt(<x²><x'²> - <xx'>²) in mm·rad
    - Normalized emittance: β·γ·ε_geo in mm·rad
    - Twiss beta: <x²>/ε_geo in mm/rad → convert to m/rad
    """
    x = state["x"]
    y = state["y"]
    Px = state["Px"]
    Py = state["Py"]
    Pz = state["Pz"]
    mass = state["m"]

    # For small-angle approximation: x' ≈ tan(θ) ≈ Px/Pz
    # This is exact for the divergence angle in the paraxial limit
    xp = Px / Pz  # dimensionless (mm/ns / mm/ns)
    yp = Py / Pz  # dimensionless

    # Calculate RMS quantities
    x_rms = np.sqrt(np.mean(x**2))  # mm
    y_rms = np.sqrt(np.mean(y**2))  # mm
    xp_rms = np.sqrt(np.mean(xp**2))  # rad
    yp_rms = np.sqrt(np.mean(yp**2))  # rad

    xxp_mean = np.mean(x * xp)  # mm·rad
    yyp_mean = np.mean(y * yp)  # mm·rad

    # Geometric emittance: ε = sqrt(<x²><x'²> - <xx'>²)
    # Units: mm·rad
    emittance_x = np.sqrt(np.mean(x**2) * np.mean(xp**2) - xxp_mean**2)  # mm·rad
    emittance_y = np.sqrt(np.mean(y**2) * np.mean(yp**2) - yyp_mean**2)  # mm·rad

    # Convert to mm·mrad (more common units)
    emittance_x_mm_mrad = emittance_x * 1000.0  # mm·mrad
    emittance_y_mm_mrad = emittance_y * 1000.0  # mm·mrad

    # Normalized emittance: ε_n = β·γ·ε_geo
    # For relativistic beams, β ≈ 1, so ε_n ≈ γ·ε_geo
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    norm_emittance_x = beta * gamma * emittance_x  # mm·rad
    norm_emittance_y = beta * gamma * emittance_y  # mm·rad

    norm_emittance_x_mm_mrad = norm_emittance_x * 1000.0  # mm·mrad
    norm_emittance_y_mm_mrad = norm_emittance_y * 1000.0  # mm·mrad

    # Twiss beta function: β_twiss = <x²>/ε at a waist (where <xx'> ≈ 0)
    # Units: mm² / (mm·rad) = mm/rad
    # Convert to m/rad for standard accelerator units
    beta_x_mm_per_rad = x_rms**2 / emittance_x if emittance_x > 0 else 0.0
    beta_y_mm_per_rad = y_rms**2 / emittance_y if emittance_y > 0 else 0.0

    beta_x_m = beta_x_mm_per_rad * 1e-3  # m/rad
    beta_y_m = beta_y_mm_per_rad * 1e-3  # m/rad

    return {
        "emittance_x_mm_mrad": float(emittance_x_mm_mrad),
        "emittance_y_mm_mrad": float(emittance_y_mm_mrad),
        "norm_emittance_x_mm_mrad": float(norm_emittance_x_mm_mrad),
        "norm_emittance_y_mm_mrad": float(norm_emittance_y_mm_mrad),
        "beta_x_m": float(beta_x_m),
        "beta_y_m": float(beta_y_m),
    }


@dataclass
class RunResult:
    metrics: Optional[Dict[str, Dict[str, float]]]
    saved_paths: Dict[str, Path]
    figures: Dict[str, Figure]
    logs: List[str]
    verbose_logs: str  # Captured stdout/stderr from verbose integration output
    duration_s: float
    filename_base: str
    # Additional computed values for optimization
    rider_delta_e: Optional[float] = None  # Final energy change in MeV
    rider_gamma_initial: Optional[float] = None
    rider_gamma_final: Optional[float] = None
    rider_trajectory: Optional[Dict[str, Any]] = None
    # Beam optics parameters (initial)
    rider_emittance_x_mm_mrad: Optional[float] = None
    rider_emittance_y_mm_mrad: Optional[float] = None
    rider_norm_emittance_x_mm_mrad: Optional[float] = None
    rider_norm_emittance_y_mm_mrad: Optional[float] = None
    rider_beta_x_m: Optional[float] = None
    rider_beta_y_m: Optional[float] = None


# ---------------------------------------------------------------------------
# Simple helpers used by both the runner and GUI
# ---------------------------------------------------------------------------


def supports_driver(simulation_type: SimulationType) -> bool:
    return simulation_type is SimulationType.BUNCH_TO_BUNCH


def apply_species_preset(params: Dict[str, float | int], preset_key: str) -> None:
    preset = SPECIES_PRESETS.get(preset_key)
    if not preset:
        return
    for param_name, value in preset.items():
        if param_name in params:
            current = params[param_name]
            params[param_name] = type(current)(value)


def list_config_files(directory: Path) -> List[str]:
    if not directory.exists():
        return []
    return sorted(str(path.name) for path in directory.glob("*.json") if path.is_file())


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_filename_base(config_name: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base = config_name.strip().replace(".json", "") or "testbed_config"
    return f"{base}_{timestamp}"


def _extract_scalar_series(
    states: Iterable[Dict[str, np.ndarray]], key: str
) -> np.ndarray:
    values: List[float] = []
    for state in states:
        data = state.get(key)
        if data is None or len(data) == 0:
            values.append(0.0)
        else:
            values.append(float(np.asarray(data)[0]))
    return np.asarray(values, dtype=float)


def _extract_vector_series(
    states: Iterable[Dict[str, np.ndarray]], keys: Tuple[str, ...]
) -> np.ndarray:
    components = [_extract_scalar_series(states, key) for key in keys]
    return np.stack(components, axis=-1)


def compute_initial_summary(options: SimulationOptions) -> InitialSummary:
    sim_type = options.simulation_type
    driver_allowed = supports_driver(sim_type)
    rider_params = dict(options.rider_params)
    driver_params = dict(options.driver_params) if options.driver_params else None
    if not driver_allowed:
        driver_params = None

    # Spoof pcount to 2 for Twiss parameter calculation if pcount=1
    # This allows us to always show Twiss parameters in the summary
    rider_pcount_actual = int(rider_params.get("pcount", 1))
    rider_params_for_twiss = dict(rider_params)
    if rider_pcount_actual == 1:
        rider_params_for_twiss["pcount"] = 2
        # Ensure non-zero transverse distribution for emittance calculation
        if abs(rider_params_for_twiss.get("transv_dist", 0.0)) < 1e-10:
            rider_params_for_twiss["transv_dist"] = 1e-4  # 0.1 micron default

    driver_pcount_actual = int(driver_params.get("pcount", 1)) if driver_params else 1
    driver_params_for_twiss = dict(driver_params) if driver_params else None
    if driver_params_for_twiss and driver_pcount_actual == 1:
        driver_params_for_twiss["pcount"] = 2
        # Ensure non-zero transverse distribution for emittance calculation
        if abs(driver_params_for_twiss.get("transv_dist", 0.0)) < 1e-10:
            driver_params_for_twiss["transv_dist"] = 1e-4  # 0.1 micron default

    rider_state, driver_state, rider_rest_mev, driver_rest_mev = (
        prepare_two_particle_demo(
            seed=options.seed,
            rider_params=rider_params_for_twiss,
            driver_params=driver_params_for_twiss,
        )
    )

    rider_gamma = float(rider_state["gamma"][0])
    rider_rest_gev = rider_rest_mev * 1e-3
    rider_total_gev = rider_gamma * rider_rest_gev

    # Always calculate rider beam optics (using spoofed pcount if necessary)
    rider_optics = compute_beam_optics(rider_state, rider_gamma)

    # Declare driver variables with explicit types
    driver_gamma: Optional[float]
    driver_rest_mev_opt: Optional[float]
    driver_rest_gev: Optional[float]
    driver_total_gev: Optional[float]

    if driver_allowed and driver_state is not None and driver_rest_mev is not None:
        driver_gamma = float(driver_state["gamma"][0])
        driver_rest_mev_opt = driver_rest_mev
        driver_rest_gev = driver_rest_mev * 1e-3
        driver_total_gev = driver_gamma * driver_rest_gev

        # Always calculate driver beam optics (using spoofed pcount if necessary)
        driver_optics_result = compute_beam_optics(driver_state, driver_gamma)
        driver_emit_x = driver_optics_result["emittance_x_mm_mrad"]
        driver_emit_y = driver_optics_result["emittance_y_mm_mrad"]
        driver_norm_emit_x = driver_optics_result["norm_emittance_x_mm_mrad"]
        driver_norm_emit_y = driver_optics_result["norm_emittance_y_mm_mrad"]
        driver_beta_x = driver_optics_result["beta_x_m"]
        driver_beta_y = driver_optics_result["beta_y_m"]
    else:
        driver_gamma = None
        driver_rest_mev_opt = None
        driver_rest_gev = None
        driver_total_gev = None
        driver_emit_x = None
        driver_emit_y = None
        driver_norm_emit_x = None
        driver_norm_emit_y = None
        driver_beta_x = None
        driver_beta_y = None

    return InitialSummary(
        seed=options.seed,
        rider_gamma=rider_gamma,
        rider_rest_mev=rider_rest_mev,
        rider_rest_gev=rider_rest_gev,
        rider_total_gev=rider_total_gev,
        driver_gamma=driver_gamma,
        driver_rest_mev=driver_rest_mev_opt,
        driver_rest_gev=driver_rest_gev,
        driver_total_gev=driver_total_gev,
        supports_driver=driver_allowed,
        rider_emittance_x_mm_mrad=rider_optics["emittance_x_mm_mrad"],
        rider_emittance_y_mm_mrad=rider_optics["emittance_y_mm_mrad"],
        rider_norm_emittance_x_mm_mrad=rider_optics["norm_emittance_x_mm_mrad"],
        rider_norm_emittance_y_mm_mrad=rider_optics["norm_emittance_y_mm_mrad"],
        rider_beta_x_m=rider_optics["beta_x_m"],
        rider_beta_y_m=rider_optics["beta_y_m"],
        driver_emittance_x_mm_mrad=driver_emit_x,
        driver_emittance_y_mm_mrad=driver_emit_y,
        driver_norm_emittance_x_mm_mrad=driver_norm_emit_x,
        driver_norm_emittance_y_mm_mrad=driver_norm_emit_y,
        driver_beta_x_m=driver_beta_x,
        driver_beta_y_m=driver_beta_y,
    )


# ---------------------------------------------------------------------------
# Core runner mirroring the widget logic
# ---------------------------------------------------------------------------


def run_testbed(
    options: SimulationOptions,
    *,
    log: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> RunResult:
    """Execute the integrator benchmark with plotting/export side effects.

    Parameters
    ----------
    options:
        Simulation configuration options.
    log:
        Optional callback for logging messages.
    progress_callback:
        Optional callback invoked as progress_callback(current, total) to report progress.
    cancel_callback:
        Optional predicate that returns True if cancellation is requested.
    """

    start = time.perf_counter()
    logs: List[str] = []

    def _log(message: str) -> None:
        logs.append(message)
        if log is not None:
            log(message)

    sim_type = options.simulation_type
    driver_allowed = supports_driver(sim_type)

    rider_params = dict(options.rider_params)
    driver_params = (
        dict(options.driver_params)
        if options.driver_params and driver_allowed
        else None
    )

    core_params = {}
    for k in CORE_PARAM_DEFAULTS:
        val = options.core_params.get(k, CORE_PARAM_DEFAULTS[k])
        # Keep string values as-is (e.g., z_cutoff_mode), convert numeric to float
        if isinstance(val, str):
            core_params[k] = val
        else:
            core_params[k] = float(val)
    required_params = CORE_REQUIRED_PARAMS.get(sim_type, set())
    filtered_core_params = {name: core_params[name] for name in required_params}

    legacy_enabled = bool(options.legacy_enabled)
    if not legacy_enabled:
        overlay_display = False
        overlay_save = False
        difference_display = False
        difference_save = False
        metrics_save = False
    else:
        overlay_display = bool(options.overlay_display)
        overlay_save = bool(options.overlay_save)
        difference_display = bool(options.difference_display)
        difference_save = bool(options.difference_save)
        metrics_save = bool(options.metrics_save)

    energy_display = bool(options.energy_display)
    energy_save = bool(options.energy_save)
    transverse_display = bool(options.transverse_display)
    transverse_save = bool(options.transverse_save)
    trajectory_save = bool(options.trajectory_save)

    should_save = any(
        [
            overlay_save,
            difference_save,
            metrics_save,
            energy_save,
            transverse_save,
            trajectory_save,
        ]
    )

    output_dir = Path(options.output_dir).expanduser()
    if should_save:
        ensure_directory(output_dir)

    filename_base = generate_filename_base(options.config_name)
    timestamp_token = filename_base[-_TIMESTAMP_TOKEN_LENGTH:]
    if len(filename_base) > (_TIMESTAMP_TOKEN_LENGTH + 1):
        config_label = filename_base[: -(_TIMESTAMP_TOKEN_LENGTH + 1)]
    else:
        config_label = filename_base

    _log(
        f"Running {sim_type.name.replace('_', ' ').title()} integrator for {options.steps} steps"
    )
    _log(f"  Steps: {options.steps}")
    _log(f"  Seed: {options.seed}")
    _log(f"  Core params: {filtered_core_params}")
    _log(f"  Legacy enabled: {legacy_enabled}")
    _log(f"  Image subcharges: {options.image_subcharge_count}")
    _log(f"  Image weighting: {options.use_image_weighting}")
    # Normalize mode name for display (handle legacy aliases)
    mode_aliases = {
        "mass_shell_only": "fixed_geometry",
        "full_iteration": "variable_geometry",
    }
    display_mode = mode_aliases.get(
        options.self_consistency_convergence_mode,
        options.self_consistency_convergence_mode,
    )
    _log(
        f"  Self-consistency: {options.self_consistency_enabled} (mode={display_mode}, "
        f"ms_tol={options.self_consistency_target_ms_tolerance:.1e}, "
        f"max_iter={options.self_consistency_max_iterations}, safety_net={options.self_consistency_mass_shell_tolerance:.1e}, "
        f"relaxation={options.self_consistency_mass_shell_relaxation:.1f})"
    )
    _log(
        f"  Energy monitoring: {options.energy_monitor_enabled} (threshold={options.energy_monitor_threshold * 100:.0f}%, halt={options.energy_monitor_halt_on_jump})"
    )
    _log(
        f"  Adaptive timestep: {options.adaptive_timestep_enabled} (threshold={options.adaptive_timestep_threshold * 100:.0f}%, reduction={options.adaptive_timestep_reduction_factor}x)"
    )
    _log("")

    return_traj_flag = any(
        [
            trajectory_save,
            transverse_display,
            transverse_save,
            energy_save,
            energy_display,
            overlay_display,
            overlay_save,
        ]
    )

    # Capture stdout/stderr to get verbose SC and adaptive timestep logs
    # Use TeeStringIO to also print to console in real-time
    import sys

    stdout_capture = TeeStringIO(sys.stdout)
    stderr_capture = TeeStringIO(sys.stderr)

    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        result = run_benchmark(
            steps=options.steps,
            simulation_type=sim_type,
            rider_params=rider_params,
            driver_params=driver_params,
            seed=options.seed,
            legacy_enabled=legacy_enabled,
            return_trajectories=return_traj_flag,
            image_subcharge_count=int(options.image_subcharge_count),
            use_image_weighting=bool(options.use_image_weighting),
            self_consistency_enabled=options.self_consistency_enabled,
            self_consistency_tolerance=options.self_consistency_target_ms_tolerance,  # Map to tolerance for backward compat
            self_consistency_convergence_mode=options.self_consistency_convergence_mode,
            self_consistency_target_ms_tolerance=options.self_consistency_target_ms_tolerance,
            self_consistency_max_iterations=options.self_consistency_max_iterations,
            self_consistency_mass_shell_tolerance=options.self_consistency_mass_shell_tolerance,
            self_consistency_mass_shell_relaxation=options.self_consistency_mass_shell_relaxation,
            self_consistency_verbosity=options.self_consistency_verbosity,
            energy_monitor_enabled=options.energy_monitor_enabled,
            energy_monitor_threshold=options.energy_monitor_threshold,
            energy_monitor_check_interval=options.energy_monitor_check_interval,
            energy_monitor_halt_on_jump=options.energy_monitor_halt_on_jump,
            energy_monitor_debug=options.energy_monitor_debug,
            adaptive_timestep_enabled=options.adaptive_timestep_enabled,
            adaptive_timestep_threshold=options.adaptive_timestep_threshold,
            adaptive_timestep_reduction_factor=options.adaptive_timestep_reduction_factor,
            adaptive_timestep_max_attempts=options.adaptive_timestep_max_attempts,
            adaptive_timestep_min_factor=options.adaptive_timestep_min_factor,
            adaptive_timestep_cooldown_steps=options.adaptive_timestep_cooldown_steps,
            adaptive_timestep_probe_threshold=options.adaptive_timestep_probe_threshold,
            adaptive_timestep_max_probe_steps=options.adaptive_timestep_max_probe_steps,
            adaptive_timestep_debug=options.adaptive_timestep_debug,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            **filtered_core_params,
        )

    # Store captured stdout/stderr separately for verbose logs button
    captured_stdout = stdout_capture.getvalue()
    captured_stderr = stderr_capture.getvalue()

    # Log a summary
    stdout_lines = len([l for l in captured_stdout.splitlines() if l.strip()])
    stderr_lines = len([l for l in captured_stderr.splitlines() if l.strip()])

    if stdout_lines > 0:
        _log(
            f"Verbose output: {stdout_lines:,} lines (displayed in console and available via 'Load Verbose Logs')"
        )
    if stderr_lines > 0:
        _log(f"Stderr: {stderr_lines} lines")

    if isinstance(result, tuple) and len(result) == 2:
        metrics, payload = result
    else:
        metrics = result
        payload = {}

    saved_paths: Dict[str, Path] = {}
    figures: Dict[str, plt.Figure] = {}

    if legacy_enabled and metrics_save and metrics is not None:
        metrics_path = output_dir / f"{filename_base}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, default=str)
        saved_paths["metrics"] = metrics_path
        _log(f"Saved metrics to: {metrics_path}")

    core_traj = payload.get("core")
    legacy_traj = payload.get("legacy") if legacy_enabled else None
    initial_states = payload.get("initial_states", {})
    rest_energies = payload.get("rest_energy_mev", {})

    # Initialize values for RunResult
    rider_delta_e_final = None
    rider_gamma_initial = None
    rider_gamma_final = None
    rider_trajectory_data = None
    rider_emittance_x = None
    rider_emittance_y = None
    rider_norm_emittance_x = None
    rider_norm_emittance_y = None
    rider_beta_x = None
    rider_beta_y = None

    # Compute gamma and beam optics from initial parameters even if trajectories aren't saved
    # This ensures metrics are available for optimization sweeps
    rider_initial = initial_states.get("rider")
    if rider_initial:
        # initial_state values are numpy arrays (normalized), extract scalars
        # Note: keys are capital P (Pz, Px, Py) not lowercase
        Pz_init = float(np.asarray(rider_initial.get("Pz", 0)).flat[0])
        Px_init = float(np.asarray(rider_initial.get("Px", 0)).flat[0])
        Py_init = float(np.asarray(rider_initial.get("Py", 0)).flat[0])
        P_init = np.sqrt(Pz_init**2 + Px_init**2 + Py_init**2)
        mass = float(np.asarray(rider_initial.get("m", 1)).flat[0])
        # P is in units of amu*mm/ns, divide by (m*c) to get dimensionless momentum p
        # Then gamma = sqrt(1 + p^2)
        p_init = P_init / (mass * C_MMNS)
        rider_gamma_initial = float(np.sqrt(1 + p_init**2))
        _log(f"[DEBUG] Initial state gamma calculation:")
        _log(
            f"  Pz={Pz_init:.3f}, Px={Px_init:.3f}, Py={Py_init:.3f}, P_total={P_init:.3f}, mass={mass:.6f}, p={p_init:.3f}, gamma={rider_gamma_initial:.1f}"
        )

        # Compute beam optics if multi-particle bunch (pcount > 1)
        rider_pcount = options.rider_params.get("pcount", 1)
        if rider_pcount > 1:
            try:
                beam_optics = compute_beam_optics(rider_initial, rider_gamma_initial)
                rider_emittance_x = beam_optics.get("emittance_x_mm_mrad")
                rider_emittance_y = beam_optics.get("emittance_y_mm_mrad")
                rider_norm_emittance_x = beam_optics.get("norm_emittance_x_mm_mrad")
                rider_norm_emittance_y = beam_optics.get("norm_emittance_y_mm_mrad")
                rider_beta_x = beam_optics.get("beta_x_m")
                rider_beta_y = beam_optics.get("beta_y_m")
                _log(f"[DEBUG] Initial beam optics:")
                _log(
                    f"  εx={rider_emittance_x:.3e} mm·mrad, εy={rider_emittance_y:.3e} mm·mrad"
                )
                _log(f"  βx={rider_beta_x:.3e} m, βy={rider_beta_y:.3e} m")
            except Exception as exc:
                _log(f"[WARNING] Failed to compute beam optics: {exc}")

    if core_traj:
        rider_states = core_traj.get("rider", [])
        driver_states = core_traj.get("driver") if driver_allowed else None

        try:
            rider_initial = initial_states.get("rider")
            rider_rest_mev = rest_energies.get("rider")

            # Compute energy series - always get total for now
            rider_delta_e, rider_z = compute_delta_energy_series(
                rider_states,
                rider_initial,
                rider_rest_mev,
            )
            rider_delta_e_z = None
            rider_z_rel = rider_z - rider_z[0]

            # Extract values for RunResult
            if rider_delta_e is not None and len(rider_delta_e) > 0:
                rider_delta_e_final = float(rider_delta_e[-1])

            # Compute gamma values from trajectory states (for final state)
            if rider_states and len(rider_states) > 0:
                # Override initial gamma with trajectory data if available (more accurate)
                initial_state = rider_states[0]
                Pz_init = float(np.asarray(initial_state.get("Pz", 0)).flat[0])
                Px_init = float(np.asarray(initial_state.get("Px", 0)).flat[0])
                Py_init = float(np.asarray(initial_state.get("Py", 0)).flat[0])
                P_init = np.sqrt(Pz_init**2 + Px_init**2 + Py_init**2)
                mass_init = float(np.asarray(initial_state.get("m", 1)).flat[0])
                p_init = P_init / (mass_init * C_MMNS)
                rider_gamma_initial = float(np.sqrt(1 + p_init**2))

                # Compute final gamma from trajectory
                final_state = rider_states[-1]
                Pz_final = float(np.asarray(final_state.get("Pz", 0)).flat[0])
                Px_final = float(np.asarray(final_state.get("Px", 0)).flat[0])
                Py_final = float(np.asarray(final_state.get("Py", 0)).flat[0])
                P_final = np.sqrt(Pz_final**2 + Px_final**2 + Py_final**2)
                mass_final = float(np.asarray(final_state.get("m", 1)).flat[0])
                p_final = P_final / (mass_final * C_MMNS)
                rider_gamma_final = float(np.sqrt(1 + p_final**2))

                # Store trajectory data (extract scalars from normalized arrays)
                # Compute r from x,y; compute normalized momentum components
                z_arr = np.array(
                    [float(np.asarray(s.get("z", 0)).flat[0]) for s in rider_states]
                )
                x_arr = np.array(
                    [float(np.asarray(s.get("x", 0)).flat[0]) for s in rider_states]
                )
                y_arr = np.array(
                    [float(np.asarray(s.get("y", 0)).flat[0]) for s in rider_states]
                )
                r_arr = np.sqrt(x_arr**2 + y_arr**2)

                # Extract momentum components (capital P) and normalize by m*c
                Pz_arr = np.array(
                    [float(np.asarray(s.get("Pz", 0)).flat[0]) for s in rider_states]
                )
                Px_arr = np.array(
                    [float(np.asarray(s.get("Px", 0)).flat[0]) for s in rider_states]
                )
                Py_arr = np.array(
                    [float(np.asarray(s.get("Py", 0)).flat[0]) for s in rider_states]
                )
                m_arr = np.array(
                    [float(np.asarray(s.get("m", 1)).flat[0]) for s in rider_states]
                )
                # Compute transverse momentum magnitude
                Pr_arr = np.sqrt(Px_arr**2 + Py_arr**2)

                rider_trajectory_data = {
                    "z": z_arr,
                    "r": r_arr,
                    "pz": Pz_arr / (m_arr * C_MMNS),  # Normalized longitudinal momentum
                    "pr": Pr_arr / (m_arr * C_MMNS),  # Normalized transverse momentum
                    "t": np.array(
                        [float(np.asarray(s.get("t", 0)).flat[0]) for s in rider_states]
                    ),
                }

        except Exception as exc:  # pragma: no cover - defensive guard
            _log(f"Failed to compute rider energy series: {exc}")
            _log(
                f"Traceback:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
            )
            rider_delta_e = None
            rider_z_rel = None

        if driver_allowed and driver_states is not None:
            try:
                driver_initial = initial_states.get("driver")
                driver_rest_mev = rest_energies.get("driver")

                # Compute energy series - always get total for now
                driver_delta_e, driver_z = compute_delta_energy_series(
                    driver_states,
                    driver_initial,
                    driver_rest_mev,
                )
                driver_delta_e_z = None
                driver_z_rel = driver_z - driver_z[0]
            except Exception as exc:  # pragma: no cover - defensive guard
                _log(f"Failed to compute driver energy series: {exc}")
                _log(
                    f"Traceback:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
                )
                driver_delta_e = None
                driver_z_rel = None
        else:
            driver_delta_e = None
            driver_z_rel = None

        if legacy_traj and legacy_enabled:
            legacy_rider_states = legacy_traj.get("rider", [])
            try:
                legacy_rider_delta_e, legacy_rider_z = compute_delta_energy_series(
                    legacy_rider_states,
                    initial_states.get("rider"),
                    rest_energies.get("rider"),
                )
                legacy_rider_z_rel = legacy_rider_z - legacy_rider_z[0]
            except Exception as exc:  # pragma: no cover - defensive guard
                _log(f"Failed to compute legacy rider energy series: {exc}")
                _log(
                    f"Traceback:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
                )
                legacy_rider_delta_e = None
                legacy_rider_z_rel = None

            if driver_allowed and legacy_traj.get("driver") is not None:
                try:
                    legacy_driver_delta_e, legacy_driver_z = (
                        compute_delta_energy_series(
                            legacy_traj["driver"],
                            initial_states.get("driver"),
                            rest_energies.get("driver"),
                        )
                    )
                    legacy_driver_z_rel = legacy_driver_z - legacy_driver_z[0]
                except Exception as exc:
                    _log(f"Failed to compute legacy driver energy series: {exc}")  # type: ignore[assignment]
                    _log(
                        f"Traceback:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
                    )
                    legacy_driver_delta_e = None
                    legacy_driver_z_rel = None
            else:
                legacy_driver_delta_e = None
                legacy_driver_z_rel = None
        else:
            legacy_rider_delta_e = None
            legacy_rider_z_rel = None
            legacy_driver_delta_e = None
            legacy_driver_z_rel = None

        if (
            (energy_save or energy_display)
            and rider_delta_e is not None
            and rider_z_rel is not None
        ):
            # Validate data before plotting to prevent matplotlib errors
            rider_delta_e_valid = np.isfinite(rider_delta_e)
            rider_z_rel_valid = np.isfinite(rider_z_rel)
            valid_mask = rider_delta_e_valid & rider_z_rel_valid

            if not np.any(valid_mask):
                _log(
                    "Warning: Energy plot skipped - all data points are invalid (NaN or Inf)"
                )
                fig_energy = None
            else:
                if not np.all(valid_mask):
                    invalid_count = np.sum(~valid_mask)
                    _log(
                        f"Warning: {invalid_count} invalid data points removed from energy plot"
                    )

                fig_energy, axes_energy = plt.subplots(
                    1,
                    2 if driver_allowed and driver_delta_e is not None else 1,
                    figsize=(16 if driver_delta_e is not None else 8, 6),
                    dpi=options.plot_dpi,
                )
                if not isinstance(axes_energy, np.ndarray):
                    axes = [axes_energy]
                else:
                    axes = list(axes_energy)

                show_legend = legacy_enabled

                # Plot total ΔE
                axes[0].scatter(
                    rider_z_rel[valid_mask],
                    rider_delta_e[valid_mask],
                    color=COLOR_RIDER,
                    label="Core" if show_legend else None,
                    **SCATTER_STYLE,
                )

                # Note: ΔE_z plotting removed - use energy_yaxis dropdown instead
                if legacy_rider_delta_e is not None and legacy_rider_z_rel is not None:
                    legacy_valid = np.isfinite(legacy_rider_delta_e) & np.isfinite(
                        legacy_rider_z_rel
                    )
                    if np.any(legacy_valid):
                        axes[0].scatter(
                            legacy_rider_z_rel[legacy_valid],
                            legacy_rider_delta_e[legacy_valid],
                            color=COLOR_LEGACY_RIDER,
                            label="Legacy",
                            **SCATTER_STYLE,
                        )
                axes[0].set_xlabel("Δz (mm)")
                axes[0].set_ylabel("ΔE (GeV)")
                axes[0].set_title("Rider ΔE vs Δz", pad=10)
                axes[0].grid(True, alpha=0.3)
                axes[0].tick_params(axis="both", which="major", labelsize=10)
                axes[0].tick_params(axis="both", which="minor", labelsize=8)
                # Fix log scale tick label formatting
                from matplotlib.ticker import ScalarFormatter

                axes[0].yaxis.set_major_formatter(ScalarFormatter())
                axes[0].yaxis.get_major_formatter().set_scientific(False)
                axes[0].yaxis.get_major_formatter().set_useOffset(False)
                if show_legend:
                    axes[0].legend()

                if (
                    driver_delta_e is not None
                    and driver_z_rel is not None
                    and len(axes) > 1
                ):
                    driver_valid = np.isfinite(driver_delta_e) & np.isfinite(
                        driver_z_rel
                    )
                    if np.any(driver_valid):
                        axes[1].scatter(
                            driver_z_rel[driver_valid],
                            driver_delta_e[driver_valid],
                            color=COLOR_DRIVER,
                            label="Core" if legacy_enabled else None,
                            **SCATTER_STYLE,
                        )

                        # Note: ΔE_z plotting removed - use energy_yaxis dropdown instead
                    else:
                        _log("Warning: All driver energy data points are invalid")
                    if (
                        legacy_driver_delta_e is not None
                        and legacy_driver_z_rel is not None
                    ):
                        legacy_driver_valid = np.isfinite(
                            legacy_driver_delta_e
                        ) & np.isfinite(legacy_driver_z_rel)
                        if np.any(legacy_driver_valid):
                            axes[1].scatter(
                                legacy_driver_z_rel[legacy_driver_valid],
                                legacy_driver_delta_e[legacy_driver_valid],
                                color=COLOR_LEGACY_DRIVER,
                                label="Legacy",
                                **SCATTER_STYLE,
                            )
                    axes[1].set_xlabel("Δz (mm)")
                    axes[1].set_ylabel("ΔE (GeV)")
                    axes[1].set_title("Driver ΔE vs Δz", pad=10)
                    axes[1].grid(True, alpha=0.3)
                    axes[1].tick_params(axis="both", which="major", labelsize=10)
                    axes[1].tick_params(axis="both", which="minor", labelsize=8)
                    # Fix log scale tick label formatting
                    from matplotlib.ticker import ScalarFormatter

                    axes[1].yaxis.set_major_formatter(ScalarFormatter())
                    axes[1].yaxis.get_major_formatter().set_scientific(False)
                    axes[1].yaxis.get_major_formatter().set_useOffset(False)
                    if legacy_enabled:
                        axes[1].legend()

                # Attach metadata for interactive replotting
                # Extract time data from rider states
                rider_times = np.array(
                    [float(np.asarray(s.get("t", 0)).flat[0]) for s in rider_states]
                )

                fig_energy._lw_plot_data = {
                    "plot_type": "energy",
                    "times_ns": rider_times[valid_mask]
                    if np.any(valid_mask)
                    else np.array([]),
                    "z_mm": rider_z_rel[valid_mask]
                    if np.any(valid_mask)
                    else np.array([]),
                    "z_mm_driver": driver_z_rel[driver_valid]
                    if driver_delta_e is not None and np.any(driver_valid)
                    else None,
                    "z_mm_legacy": legacy_rider_z_rel[legacy_valid]
                    if legacy_rider_z_rel is not None and np.any(legacy_valid)
                    else None,
                    "z_mm_legacy_driver": legacy_driver_z_rel[legacy_driver_valid]
                    if legacy_driver_z_rel is not None and np.any(legacy_driver_valid)
                    else None,
                    "core_r_energy_changes": rider_delta_e[valid_mask]
                    if np.any(valid_mask)
                    else np.array([]),
                    "core_d_energy_changes": driver_delta_e[driver_valid]
                    if driver_delta_e is not None and np.any(driver_valid)
                    else None,
                    "legacy_r_energy_changes": legacy_rider_delta_e[legacy_valid]
                    if legacy_rider_delta_e is not None and np.any(legacy_valid)
                    else None,
                    "legacy_d_energy_changes": legacy_driver_delta_e[
                        legacy_driver_valid
                    ]
                    if legacy_driver_delta_e is not None and np.any(legacy_driver_valid)
                    else None,
                    "driver_allowed": driver_allowed,
                    "legacy_enabled": legacy_enabled,
                }

                fig_energy.tight_layout(pad=2.5, w_pad=3.0, h_pad=2.5)
                if energy_save and should_save:
                    energy_path = output_dir / f"{filename_base}_energy.png"
                    fig_energy.savefig(energy_path)
                    saved_paths["energy"] = energy_path
                    _log(f"Saved energy plot to: {energy_path}")
                if energy_display:
                    figures["energy"] = fig_energy
                else:
                    plt.close(fig_energy)

        if (
            legacy_enabled
            and (overlay_display or overlay_save)
            and rider_delta_e is not None
            and rider_z_rel is not None
            and legacy_rider_delta_e is not None
            and legacy_rider_z_rel is not None
        ):
            fig_overlay, axes_overlay = plt.subplots(
                1,
                (
                    2
                    if driver_delta_e is not None and legacy_driver_delta_e is not None
                    else 1
                ),
                figsize=(
                    (
                        16
                        if driver_delta_e is not None
                        and legacy_driver_delta_e is not None
                        else 8
                    ),
                    6,
                ),
                dpi=options.plot_dpi,
            )
            if not isinstance(axes_overlay, np.ndarray):
                axes = [axes_overlay]
            else:
                axes = list(axes_overlay)

            axes[0].plot(
                rider_z_rel,
                rider_delta_e,
                color=COLOR_RIDER,
                label="Core",
                linewidth=2.0,
            )
            axes[0].plot(
                legacy_rider_z_rel,
                legacy_rider_delta_e,
                color=COLOR_LEGACY_RIDER,
                label="Legacy",
                linewidth=2.0,
                linestyle="--",
            )
            axes[0].set_xlabel("Δz (mm)")
            axes[0].set_ylabel("ΔE (GeV)")
            axes[0].set_title("Rider ΔE Comparison")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            if (
                driver_delta_e is not None
                and driver_z_rel is not None
                and legacy_driver_delta_e is not None
                and legacy_driver_z_rel is not None
                and len(axes) > 1
            ):
                axes[1].plot(
                    driver_z_rel,
                    driver_delta_e,
                    color=COLOR_DRIVER,
                    label="Core",
                    linewidth=2.0,
                )
                axes[1].plot(
                    legacy_driver_z_rel,
                    legacy_driver_delta_e,
                    color=COLOR_LEGACY_DRIVER,
                    label="Legacy",
                    linewidth=2.0,
                    linestyle="--",
                )
                axes[1].set_xlabel("Δz (mm)")
                axes[1].set_ylabel("ΔE (GeV)")
                axes[1].set_title("Driver ΔE Comparison")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            fig_overlay.tight_layout()
            if overlay_save and should_save:
                overlay_path = output_dir / f"{filename_base}_overlay.png"
                fig_overlay.savefig(overlay_path)
                saved_paths["overlay"] = overlay_path
                _log(f"Saved overlay plot to: {overlay_path}")
            if overlay_display:
                figures["overlay"] = fig_overlay
            else:
                plt.close(fig_overlay)

        core_r_hist = np.array(
            [[s["t"][0], s["x"][0], s["y"][0], s["z"][0]] for s in rider_states]
        )
        core_r_gamma = _extract_scalar_series(rider_states, "gamma")
        core_r_momentum = _extract_vector_series(rider_states, ("Px", "Py", "Pz"))
        core_r_pt = _extract_scalar_series(rider_states, "Pt")
        core_r_beta = _extract_vector_series(rider_states, ("bx", "by", "bz"))
        core_r_betadot = _extract_vector_series(
            rider_states, ("bdotx", "bdoty", "bdotz")
        )
        plot_times_ns = core_r_hist[:, 0]
        plot_z_mm = core_r_hist[:, 3]

        if driver_allowed and driver_states is not None:
            core_d_hist = np.array(
                [[s["t"][0], s["x"][0], s["y"][0], s["z"][0]] for s in driver_states]
            )
            core_d_gamma = _extract_scalar_series(driver_states, "gamma")
            core_d_momentum = _extract_vector_series(driver_states, ("Px", "Py", "Pz"))
            core_d_beta = _extract_vector_series(driver_states, ("bx", "by", "bz"))
            core_d_betadot = _extract_vector_series(
                driver_states, ("bdotx", "bdoty", "bdotz")
            )
            core_d_pt = _extract_scalar_series(driver_states, "Pt")
        else:
            core_d_hist = None
            core_d_gamma = None
            core_d_momentum = None
            core_d_pt = None
            core_d_beta = None
            core_d_betadot = None

        if legacy_enabled and legacy_traj:
            legacy_r_hist = np.array(
                [
                    [s["t"][0], s["x"][0], s["y"][0], s["z"][0]]
                    for s in legacy_traj.get("rider", [])
                ]
            )
            legacy_r_momentum = _extract_vector_series(
                legacy_traj.get("rider", []), ("Px", "Py", "Pz")
            )
            legacy_r_beta = _extract_vector_series(
                legacy_traj.get("rider", []), ("bx", "by", "bz")
            )
            legacy_r_betadot = _extract_vector_series(
                legacy_traj.get("rider", []), ("bdotx", "bdoty", "bdotz")
            )
            legacy_r_pt = _extract_scalar_series(legacy_traj.get("rider", []), "Pt")
            if driver_allowed and legacy_traj.get("driver") is not None:
                legacy_d_hist = np.array(
                    [
                        [s["t"][0], s["x"][0], s["y"][0], s["z"][0]]
                        for s in legacy_traj.get("driver", [])
                    ]
                )
                legacy_d_momentum = _extract_vector_series(
                    legacy_traj.get("driver", []), ("Px", "Py", "Pz")
                )
                legacy_d_beta = _extract_vector_series(
                    legacy_traj.get("driver", []), ("bx", "by", "bz")
                )
                legacy_d_betadot = _extract_vector_series(
                    legacy_traj.get("driver", []), ("bdotx", "bdoty", "bdotz")
                )
                legacy_d_pt = _extract_scalar_series(
                    legacy_traj.get("driver", []), "Pt"
                )
            else:
                legacy_d_hist = None
                legacy_d_momentum = None
                legacy_d_beta = None
                legacy_d_betadot = None
                legacy_d_pt = None
        else:
            legacy_r_hist = None
            legacy_r_momentum = None
            legacy_r_beta = None
            legacy_r_betadot = None
            legacy_r_pt = None
            legacy_d_hist = None
            legacy_d_momentum = None
            legacy_d_beta = None
            legacy_d_betadot = None
            legacy_d_pt = None

        if (
            legacy_enabled
            and (difference_save or difference_display)
            and legacy_r_hist is not None
        ):
            fig_diff, axes_diff = plt.subplots(
                1,
                (
                    2
                    if driver_allowed
                    and core_d_hist is not None
                    and legacy_d_hist is not None
                    else 1
                ),
                figsize=(
                    (
                        16
                        if driver_allowed
                        and core_d_hist is not None
                        and legacy_d_hist is not None
                        else 8
                    ),
                    6,
                ),
                dpi=options.plot_dpi,
            )
            if not isinstance(axes_diff, np.ndarray):
                axes = [axes_diff]
            else:
                axes = list(axes_diff)

            r_delta_x = (core_r_hist[:, 1] - legacy_r_hist[:, 1]) * 1e3
            r_delta_y = (core_r_hist[:, 2] - legacy_r_hist[:, 2]) * 1e3
            r_delta_z = (core_r_hist[:, 3] - legacy_r_hist[:, 3]) * 1e3
            axes[0].plot(
                plot_times_ns, r_delta_x, label="Delta x (mm)", color=COLOR_DIFF_RIDER
            )
            axes[0].plot(
                plot_times_ns,
                r_delta_y,
                label="Delta y (mm)",
                color=COLOR_DIFF_RIDER,
                linestyle="--",
            )
            axes[0].plot(
                plot_times_ns,
                r_delta_z,
                label="Delta z (mm)",
                color=COLOR_DIFF_RIDER,
                linestyle=":",
            )
            axes[0].set_xlabel("Time (ns)")
            axes[0].set_ylabel("Δ position (mm)")
            axes[0].set_title("Rider Δ (core - legacy)")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            if (
                driver_allowed
                and core_d_hist is not None
                and legacy_d_hist is not None
                and len(axes) > 1
            ):
                d_delta_x = (core_d_hist[:, 1] - legacy_d_hist[:, 1]) * 1e3
                d_delta_y = (core_d_hist[:, 2] - legacy_d_hist[:, 2]) * 1e3
                d_delta_z = (core_d_hist[:, 3] - legacy_d_hist[:, 3]) * 1e3
                axes[1].plot(
                    plot_times_ns,
                    d_delta_x,
                    label="Delta x (mm)",
                    color=COLOR_DIFF_DRIVER,
                )
                axes[1].plot(
                    plot_times_ns,
                    d_delta_y,
                    label="Delta y (mm)",
                    color=COLOR_DIFF_DRIVER,
                    linestyle="--",
                )
                axes[1].plot(
                    plot_times_ns,
                    d_delta_z,
                    label="Delta z (mm)",
                    color=COLOR_DIFF_DRIVER,
                    linestyle=":",
                )
                axes[1].set_xlabel("Time (ns)")
                axes[1].set_ylabel("Δ position (mm)")
                axes[1].set_title("Driver Δ (core - legacy)")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            fig_diff.tight_layout()
            if difference_save and should_save:
                diff_path = output_dir / f"{filename_base}_difference.png"
                fig_diff.savefig(diff_path)
                saved_paths["difference"] = diff_path
                _log(f"Saved difference plot to: {diff_path}")
            if difference_display:
                figures["difference"] = fig_diff
            else:
                plt.close(fig_diff)

        transverse_xaxis = getattr(options, "transverse_xaxis", "t")
        if transverse_display or transverse_save:
            fig_transverse, (ax_x, ax_y) = plt.subplots(
                1, 2, figsize=(16, 6), dpi=options.plot_dpi
            )

            # Attach metadata for interactive replotting
            fig_transverse._lw_plot_data = {
                "plot_type": "transverse",
                "times_ns": plot_times_ns,
                "z_mm": plot_z_mm,
                "z_mm_driver": core_d_hist[:, 3]
                if driver_allowed and core_d_hist is not None
                else None,
                "z_mm_legacy": legacy_r_hist[:, 3]
                if legacy_enabled and legacy_r_hist is not None
                else None,
                "z_mm_legacy_driver": legacy_d_hist[:, 3]
                if legacy_enabled and driver_allowed and legacy_d_hist is not None
                else None,
                "core_r_hist": core_r_hist,
                "core_d_hist": core_d_hist if driver_allowed else None,
                "legacy_r_hist": legacy_r_hist if legacy_enabled else None,
                "legacy_d_hist": legacy_d_hist
                if legacy_enabled and driver_allowed
                else None,
                "driver_allowed": driver_allowed,
                "legacy_enabled": legacy_enabled,
            }

            # Determine x-axis data
            if transverse_xaxis == "z":
                xdata = plot_z_mm
                xlabel = "z position (mm)"
            else:
                xdata = plot_times_ns
                xlabel = "Time (ns)"

            ax_x.plot(
                xdata,
                core_r_hist[:, 1] * 1e3,
                color=COLOR_RIDER,
                label="Rider (Core)",
            )
            ax_y.plot(
                xdata,
                core_r_hist[:, 2] * 1e3,
                color=COLOR_RIDER,
                label="Rider (Core)",
            )
            if driver_allowed and core_d_hist is not None:
                if transverse_xaxis == "z":
                    xdata_d = core_d_hist[:, 3]
                else:
                    xdata_d = plot_times_ns
                ax_x.plot(
                    xdata_d,
                    core_d_hist[:, 1] * 1e3,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
                ax_y.plot(
                    xdata_d,
                    core_d_hist[:, 2] * 1e3,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_hist is not None:
                if transverse_xaxis == "z":
                    xdata_leg = legacy_r_hist[:, 3]
                else:
                    xdata_leg = plot_times_ns
                ax_x.plot(
                    xdata_leg,
                    legacy_r_hist[:, 1] * 1e3,
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
                ax_y.plot(
                    xdata_leg,
                    legacy_r_hist[:, 2] * 1e3,
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
                if driver_allowed and legacy_d_hist is not None:
                    if transverse_xaxis == "z":
                        xdata_leg_d = legacy_d_hist[:, 3]
                    else:
                        xdata_leg_d = plot_times_ns
                    ax_x.plot(
                        xdata_leg_d,
                        legacy_d_hist[:, 1] * 1e3,
                        color=COLOR_LEGACY_DRIVER,
                        linestyle="--",
                        label="Driver (Legacy)",
                    )
                    ax_y.plot(
                        xdata_leg_d,
                        legacy_d_hist[:, 2] * 1e3,
                        color=COLOR_LEGACY_DRIVER,
                        linestyle="--",
                        label="Driver (Legacy)",
                    )
            ax_x.set_xlabel(xlabel)
            ax_x.set_ylabel("Average x (mm)")
            ax_x.set_title("Average X Position", pad=12)
            ax_x.legend()
            ax_x.grid(True, alpha=0.3)
            ax_x.tick_params(axis="both", which="major", labelsize=10)
            ax_y.set_xlabel(xlabel)
            ax_y.set_ylabel("Average y (mm)")
            ax_y.set_title("Average Y Position", pad=12)
            ax_y.legend()
            ax_y.grid(True, alpha=0.3)
            ax_y.tick_params(axis="both", which="major", labelsize=10)
            fig_transverse.tight_layout(pad=3.0, w_pad=4.0, h_pad=3.0)
            if transverse_save and should_save:
                transverse_path = output_dir / f"{filename_base}_transverse.png"
                fig_transverse.savefig(transverse_path)
                saved_paths["transverse"] = transverse_path
                _log(f"Saved transverse plot to: {transverse_path}")
            if transverse_display:
                figures["transverse"] = fig_transverse
            else:
                plt.close(fig_transverse)

        # Beta (velocity) plots
        beta_display = options.beta_display
        beta_save = options.beta_save
        beta_xaxis = getattr(options, "beta_xaxis", "t")
        if (beta_display or beta_save) and core_r_beta is not None:
            fig_beta, axes_beta = plt.subplots(
                2, 2, figsize=(16, 14), dpi=options.plot_dpi, constrained_layout=True
            )
            axes_beta = axes_beta.flatten()

            # Attach metadata for interactive replotting
            fig_beta._lw_plot_data = {
                "plot_type": "beta",
                "times_ns": plot_times_ns,
                "z_mm": plot_z_mm,
                "z_mm_driver": core_d_hist[:, 3]
                if driver_allowed and core_d_hist is not None
                else None,
                "z_mm_legacy": legacy_r_hist[:, 3]
                if legacy_enabled and legacy_r_hist is not None
                else None,
                "core_r_beta": core_r_beta,
                "core_d_beta": core_d_beta if driver_allowed else None,
                "legacy_r_beta": legacy_r_beta if legacy_enabled else None,
                "legacy_d_beta": legacy_d_beta
                if legacy_enabled and driver_allowed
                else None,
                "driver_allowed": driver_allowed,
                "legacy_enabled": legacy_enabled,
            }

            # Determine x-axis data for beta plots
            if beta_xaxis == "z":
                xdata_beta = plot_z_mm
                xlabel_beta = "z position (mm)"
            else:
                xdata_beta = plot_times_ns
                xlabel_beta = "Time (ns)"

            # β_x
            axes_beta[0].plot(
                xdata_beta,
                core_r_beta[:, 0],
                color=COLOR_RIDER,
                label="Rider (Core)",
            )
            if driver_allowed and core_d_beta is not None:
                if beta_xaxis == "z":
                    xdata_beta_d = (
                        core_d_hist[:, 3] if core_d_hist is not None else plot_z_mm
                    )
                else:
                    xdata_beta_d = plot_times_ns
                axes_beta[0].plot(
                    xdata_beta_d,
                    core_d_beta[:, 0],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_beta is not None:
                if beta_xaxis == "z":
                    xdata_beta_leg = (
                        legacy_r_hist[:, 3] if legacy_r_hist is not None else plot_z_mm
                    )
                else:
                    xdata_beta_leg = plot_times_ns
                axes_beta[0].plot(
                    xdata_beta_leg,
                    legacy_r_beta[:, 0],
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
            axes_beta[0].set_xlabel(xlabel_beta)
            axes_beta[0].set_ylabel("β⟨x⟩")
            axes_beta[0].set_title("Beta X Component", pad=10)
            axes_beta[0].legend()
            axes_beta[0].grid(True, alpha=0.3)

            # β_y
            axes_beta[1].plot(
                xdata_beta,
                core_r_beta[:, 1],
                color=COLOR_RIDER,
                label="Rider (Core)",
            )
            if driver_allowed and core_d_beta is not None:
                axes_beta[1].plot(
                    xdata_beta_d,
                    core_d_beta[:, 1],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_beta is not None:
                axes_beta[1].plot(
                    xdata_beta_leg,
                    legacy_r_beta[:, 1],
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
            axes_beta[1].set_xlabel(xlabel_beta)
            axes_beta[1].set_ylabel("β⟨y⟩")
            axes_beta[1].set_title("Beta Y Component", pad=10)
            axes_beta[1].legend()
            axes_beta[1].grid(True, alpha=0.3)

            # β_z
            axes_beta[2].plot(
                xdata_beta,
                core_r_beta[:, 2],
                color=COLOR_RIDER,
                label="Rider (Core)",
            )
            if driver_allowed and core_d_beta is not None:
                axes_beta[2].plot(
                    xdata_beta_d,
                    core_d_beta[:, 2],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_beta is not None:
                axes_beta[2].plot(
                    xdata_beta_leg,
                    legacy_r_beta[:, 2],
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
            axes_beta[2].set_xlabel(xlabel_beta)
            axes_beta[2].set_ylabel("β⟨z⟩")
            axes_beta[2].set_title("Beta Z Component", pad=10)
            axes_beta[2].legend()
            axes_beta[2].grid(True, alpha=0.3)

            # |β| (magnitude)
            core_beta_mag = np.sqrt(np.sum(core_r_beta**2, axis=1))
            axes_beta[3].plot(
                xdata_beta, core_beta_mag, color=COLOR_RIDER, label="Rider (Core)"
            )
            if driver_allowed and core_d_beta is not None:
                driver_beta_mag = np.sqrt(np.sum(core_d_beta**2, axis=1))
                axes_beta[3].plot(
                    xdata_beta_d,
                    driver_beta_mag,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_beta is not None:
                legacy_beta_mag = np.sqrt(np.sum(legacy_r_beta**2, axis=1))
                axes_beta[3].plot(
                    xdata_beta_leg,
                    legacy_beta_mag,
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
            axes_beta[3].set_xlabel(xlabel_beta)
            axes_beta[3].set_ylabel("|β|")
            axes_beta[3].set_title("Beta Magnitude")
            axes_beta[3].legend()
            axes_beta[3].grid(True, alpha=0.3)

            if beta_save and should_save:
                beta_path = output_dir / f"{filename_base}_beta.png"
                fig_beta.savefig(beta_path)
                saved_paths["beta"] = beta_path
                _log(f"Saved beta plot to: {beta_path}")
            if beta_display:
                figures["beta"] = fig_beta
            else:
                plt.close(fig_beta)

        # Momentum plots (conjugate four-momentum in amu·mm/ns)
        momentum_display = options.momentum_display
        momentum_save = options.momentum_save
        momentum_xaxis = getattr(options, "momentum_xaxis", "t")
        if (momentum_display or momentum_save) and core_r_momentum is not None:
            fig_momentum, axes_mom = plt.subplots(
                2, 3, figsize=(20, 14), dpi=options.plot_dpi, constrained_layout=True
            )
            axes_mom = axes_mom.flatten()

            # Attach metadata for interactive replotting
            fig_momentum._lw_plot_data = {
                "plot_type": "momentum",
                "times_ns": plot_times_ns,
                "z_mm": plot_z_mm,
                "z_mm_driver": core_d_hist[:, 3]
                if driver_allowed and core_d_hist is not None
                else None,
                "z_mm_legacy": legacy_r_hist[:, 3]
                if legacy_enabled and legacy_r_hist is not None
                else None,
                "core_r_momentum": core_r_momentum,
                "core_r_pt": core_r_pt,
                "core_d_momentum": core_d_momentum if driver_allowed else None,
                "core_d_pt": core_d_pt if driver_allowed else None,
                "legacy_r_momentum": legacy_r_momentum if legacy_enabled else None,
                "legacy_r_pt": legacy_r_pt if legacy_enabled else None,
                "legacy_d_momentum": legacy_d_momentum
                if legacy_enabled and driver_allowed
                else None,
                "legacy_d_pt": legacy_d_pt
                if legacy_enabled and driver_allowed
                else None,
                "driver_allowed": driver_allowed,
                "legacy_enabled": legacy_enabled,
            }

            # Determine x-axis data for momentum plots
            if momentum_xaxis == "z":
                xdata_mom = plot_z_mm
                xlabel_mom = "z position (mm)"
            else:
                xdata_mom = plot_times_ns
                xlabel_mom = "Time (ns)"

            # P_x (conjugate momentum)
            axes_mom[0].plot(
                xdata_mom,
                core_r_momentum[:, 0],
                color=COLOR_RIDER,
                label="Rider (Core)",
            )
            if driver_allowed and core_d_momentum is not None:
                if momentum_xaxis == "z":
                    xdata_mom_d = (
                        core_d_hist[:, 3] if core_d_hist is not None else plot_z_mm
                    )
                else:
                    xdata_mom_d = plot_times_ns
                axes_mom[0].plot(
                    xdata_mom_d,
                    core_d_momentum[:, 0],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_momentum is not None:
                if momentum_xaxis == "z":
                    xdata_mom_leg = (
                        legacy_r_hist[:, 3] if legacy_r_hist is not None else plot_z_mm
                    )
                else:
                    xdata_mom_leg = plot_times_ns
                axes_mom[0].plot(
                    xdata_mom_leg,
                    legacy_r_momentum[:, 0],
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
            axes_mom[0].set_xlabel(xlabel_mom)
            axes_mom[0].set_ylabel("Pˣ (amu·mm/ns)")
            axes_mom[0].set_title("Conjugate Momentum Pˣ", pad=10)
            axes_mom[0].legend()
            axes_mom[0].grid(True, alpha=0.3)

            # P_y
            axes_mom[1].plot(
                xdata_mom,
                core_r_momentum[:, 1],
                color=COLOR_RIDER,
                label="Rider (Core)",
            )
            if driver_allowed and core_d_momentum is not None:
                axes_mom[1].plot(
                    xdata_mom_d,
                    core_d_momentum[:, 1],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_momentum is not None:
                axes_mom[1].plot(
                    xdata_mom_leg,
                    legacy_r_momentum[:, 1],
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
            axes_mom[1].set_xlabel(xlabel_mom)
            axes_mom[1].set_ylabel("Pʸ (amu·mm/ns)")
            axes_mom[1].set_title("Conjugate Momentum Pʸ", pad=10)
            axes_mom[1].legend()
            axes_mom[1].grid(True, alpha=0.3)

            # P_z
            axes_mom[2].plot(
                xdata_mom,
                core_r_momentum[:, 2],
                color=COLOR_RIDER,
                label="Rider (Core)",
            )
            if driver_allowed and core_d_momentum is not None:
                axes_mom[2].plot(
                    xdata_mom_d,
                    core_d_momentum[:, 2],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_momentum is not None:
                axes_mom[2].plot(
                    xdata_mom_leg,
                    legacy_r_momentum[:, 2],
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
            axes_mom[2].set_xlabel(xlabel_mom)
            axes_mom[2].set_ylabel("Pᶻ (amu·mm/ns)")
            axes_mom[2].set_title("Conjugate Momentum Pᶻ", pad=10)
            axes_mom[2].legend()
            axes_mom[2].grid(True, alpha=0.3)

            # |P_t| (transverse magnitude)
            core_pt_mag = np.sqrt(
                core_r_momentum[:, 0] ** 2 + core_r_momentum[:, 1] ** 2
            )
            axes_mom[3].plot(
                xdata_mom, core_pt_mag, color=COLOR_RIDER, label="Rider (Core)"
            )
            if driver_allowed and core_d_momentum is not None:
                driver_pt_mag = np.sqrt(
                    core_d_momentum[:, 0] ** 2 + core_d_momentum[:, 1] ** 2
                )
                axes_mom[3].plot(
                    xdata_mom_d,
                    driver_pt_mag,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_momentum is not None:
                legacy_pt_mag = np.sqrt(
                    legacy_r_momentum[:, 0] ** 2 + legacy_r_momentum[:, 1] ** 2
                )
                axes_mom[3].plot(
                    xdata_mom_leg,
                    legacy_pt_mag,
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
            axes_mom[3].set_xlabel(xlabel_mom)
            axes_mom[3].set_ylabel("|P⊥| (amu·mm/ns)")
            axes_mom[3].set_title("Transverse Momentum |P⊥|", pad=10)
            axes_mom[3].legend()
            axes_mom[3].grid(True, alpha=0.3)

            # P_t (temporal/energy component)
            axes_mom[4].plot(
                xdata_mom,
                core_r_pt,
                color=COLOR_RIDER,
                label="Rider (Core)",
            )
            if driver_allowed and core_d_pt is not None:
                axes_mom[4].plot(
                    xdata_mom_d,
                    core_d_pt,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_pt is not None:
                axes_mom[4].plot(
                    xdata_mom_leg,
                    legacy_r_pt,
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
            axes_mom[4].set_xlabel(xlabel_mom)
            axes_mom[4].set_ylabel("Pᵗ (amu·mm/ns)")
            axes_mom[4].set_title("Temporal Momentum Pᵗ (Energy/c)", pad=10)
            axes_mom[4].legend()
            axes_mom[4].grid(True, alpha=0.3)

            # P magnitude (optional - fourth momentum invariant check)
            core_p_mag = np.sqrt(
                core_r_momentum[:, 0] ** 2
                + core_r_momentum[:, 1] ** 2
                + core_r_momentum[:, 2] ** 2
            )
            axes_mom[5].plot(
                xdata_mom, core_p_mag, color=COLOR_RIDER, label="Rider (Core)"
            )
            if driver_allowed and core_d_momentum is not None:
                driver_p_mag = np.sqrt(
                    core_d_momentum[:, 0] ** 2
                    + core_d_momentum[:, 1] ** 2
                    + core_d_momentum[:, 2] ** 2
                )
                axes_mom[5].plot(
                    xdata_mom_d,
                    driver_p_mag,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_momentum is not None:
                legacy_p_mag = np.sqrt(
                    legacy_r_momentum[:, 0] ** 2
                    + legacy_r_momentum[:, 1] ** 2
                    + legacy_r_momentum[:, 2] ** 2
                )
                axes_mom[5].plot(
                    xdata_mom_leg,
                    legacy_p_mag,
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
            axes_mom[5].set_xlabel(xlabel_mom)
            axes_mom[5].set_ylabel("|P| (amu·mm/ns)")
            axes_mom[5].set_title("Total Spatial Momentum |P|", pad=10)
            axes_mom[5].legend()
            axes_mom[5].grid(True, alpha=0.3)

            if momentum_save and should_save:
                momentum_path = output_dir / f"{filename_base}_momentum.png"
                fig_momentum.savefig(momentum_path)
                saved_paths["momentum"] = momentum_path
                _log(f"Saved momentum plot to: {momentum_path}")
            if momentum_display:
                figures["momentum"] = fig_momentum
            else:
                plt.close(fig_momentum)

        # Z-position vs time plot
        zposition_display = getattr(options, "zposition_display", False)
        zposition_save = getattr(options, "zposition_save", False)
        if zposition_display or zposition_save:
            _log(
                f"Generating z-position vs time plot (display={zposition_display}, save={zposition_save})"
            )
            fig_zpos = plt.figure(figsize=(12, 8), dpi=options.plot_dpi)
            ax_zpos = fig_zpos.add_subplot(111)

            ax_zpos.plot(
                plot_times_ns,
                plot_z_mm,
                color=COLOR_RIDER,
                label="Rider (Core)",
                linewidth=2.0,
            )

            if driver_allowed and core_d_hist is not None:
                ax_zpos.plot(
                    plot_times_ns,
                    core_d_hist[:, 3],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    linewidth=2.0,
                )

            if legacy_enabled and legacy_r_hist is not None:
                ax_zpos.plot(
                    plot_times_ns,
                    legacy_r_hist[:, 3],
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                    linewidth=2.0,
                )
                if driver_allowed and legacy_d_hist is not None:
                    ax_zpos.plot(
                        plot_times_ns,
                        legacy_d_hist[:, 3],
                        color=COLOR_LEGACY_DRIVER,
                        linestyle="--",
                        label="Driver (Legacy)",
                        linewidth=2.0,
                    )

            ax_zpos.set_xlabel("Time (ns)")
            ax_zpos.set_ylabel("z position (mm)")
            ax_zpos.set_title("Longitudinal Position vs Time")
            ax_zpos.legend()
            ax_zpos.grid(True, alpha=0.3)
            ax_zpos.tick_params(axis="both", which="major", labelsize=10)
            fig_zpos.tight_layout()

            if zposition_save and should_save:
                zposition_path = output_dir / f"{filename_base}_zposition.png"
                fig_zpos.savefig(zposition_path)
                saved_paths["zposition"] = zposition_path
                _log(f"Saved z-position plot to: {zposition_path}")
            if zposition_display:
                figures["zposition"] = fig_zpos
                _log("Z-position plot added to display figures")
            else:
                plt.close(fig_zpos)

        if trajectory_save and should_save:
            interval = max(1, int(options.trajectory_interval))

            def _build_particle_payload(
                hist: np.ndarray,
                gamma: np.ndarray,
                momentum: np.ndarray,
                beta: np.ndarray,
                betadot: np.ndarray,
                pt: np.ndarray,
            ) -> Dict[str, object]:
                return {
                    "r_hist": hist[::interval].tolist(),
                    "gamma_hist": gamma[::interval].tolist(),
                    "time_ns": hist[::interval, 0].tolist(),
                    "positions_mm": {
                        "x": hist[::interval, 1].tolist(),
                        "y": hist[::interval, 2].tolist(),
                        "z": hist[::interval, 3].tolist(),
                    },
                    "conjugate_momenta": {
                        "Px": momentum[::interval, 0].tolist(),
                        "Py": momentum[::interval, 1].tolist(),
                        "Pz": momentum[::interval, 2].tolist(),
                    },
                    "beta": {
                        "bx": beta[::interval, 0].tolist(),
                        "by": beta[::interval, 1].tolist(),
                        "bz": beta[::interval, 2].tolist(),
                    },
                    "betadot": {
                        "bdotx": betadot[::interval, 0].tolist(),
                        "bdoty": betadot[::interval, 1].tolist(),
                        "bdotz": betadot[::interval, 2].tolist(),
                    },
                    "pt_hist": pt[::interval].tolist(),
                }

            core_payload: Dict[str, object] = {
                "rider": _build_particle_payload(
                    core_r_hist,
                    core_r_gamma,
                    core_r_momentum,
                    core_r_beta,
                    core_r_betadot,
                    core_r_pt,
                )
            }
            if (
                driver_allowed
                and core_d_hist is not None
                and core_d_momentum is not None
            ):
                core_payload["driver"] = _build_particle_payload(
                    core_d_hist,
                    core_d_gamma,
                    core_d_momentum,
                    core_d_beta,
                    core_d_betadot,
                    core_d_pt,
                )

            traj_data: Dict[str, object] = {
                "config_name": options.config_name,
                "config_label": config_label,
                "seed": options.seed,
                "num_steps": options.steps,
                "simulation_type": sim_type.name,
                "step_interval": interval,
                "timestamp": timestamp_token,
                "core": core_payload,
                "image_subcharge_count": options.image_subcharge_count,
            }

            if (
                legacy_enabled
                and legacy_r_hist is not None
                and legacy_r_momentum is not None
            ):
                legacy_payload: Dict[str, object] = {
                    "rider": _build_particle_payload(
                        legacy_r_hist,
                        _extract_scalar_series(legacy_traj.get("rider", []), "gamma"),
                        legacy_r_momentum,
                        legacy_r_beta,
                        legacy_r_betadot,
                        legacy_r_pt,
                    )
                }
                if (
                    driver_allowed
                    and legacy_d_hist is not None
                    and legacy_d_momentum is not None
                ):
                    legacy_payload["driver"] = _build_particle_payload(
                        legacy_d_hist,
                        _extract_scalar_series(legacy_traj.get("driver", []), "gamma"),
                        legacy_d_momentum,
                        legacy_d_beta,
                        legacy_d_betadot,
                        legacy_d_pt,
                    )
                traj_data["legacy"] = legacy_payload

            label_prefix = config_label if config_label else "trajectory"
            traj_path = (
                output_dir / f"{label_prefix}_trajectory_data_{timestamp_token}.json"
            )
            with traj_path.open("w", encoding="utf-8") as handle:
                json.dump(traj_data, handle, indent=2)
            saved_paths["trajectory"] = traj_path
            _log(f"Saved trajectories to: {traj_path} (interval={interval})")

    duration = time.perf_counter() - start
    _log("")
    _log("Run complete")

    # Save log file if requested
    if options.save_log_file:
        if options.log_file_path:
            log_path = Path(options.log_file_path).expanduser()
        else:
            # Auto-generate in output_dir
            ensure_directory(output_dir)
            log_path = output_dir / f"{filename_base}_log.txt"

        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write("\n".join(logs))
            saved_paths["log"] = log_path
            _log(f"Saved log file to: {log_path}")
        except Exception as exc:
            _log(f"Failed to save log file: {exc}")

        # Save verbose logs (SC convergence details) to separate file
        if captured_stdout:
            verbose_log_path = output_dir / f"{filename_base}_verbose.txt"
            try:
                with verbose_log_path.open("w", encoding="utf-8") as vlog_file:
                    vlog_file.write(captured_stdout)
                saved_paths["verbose_log"] = verbose_log_path
                _log(f"Saved verbose log to: {verbose_log_path}")
            except Exception as exc:
                _log(f"Failed to save verbose log: {exc}")

    return RunResult(
        metrics=metrics,
        saved_paths=saved_paths,
        figures=figures,
        logs=logs,
        verbose_logs=captured_stdout,  # Include captured verbose output
        duration_s=duration,
        filename_base=filename_base,
        rider_delta_e=rider_delta_e_final,
        rider_gamma_initial=rider_gamma_initial,
        rider_gamma_final=rider_gamma_final,
        rider_trajectory=rider_trajectory_data,
        rider_emittance_x_mm_mrad=rider_emittance_x,
        rider_emittance_y_mm_mrad=rider_emittance_y,
        rider_norm_emittance_x_mm_mrad=rider_norm_emittance_x,
        rider_norm_emittance_y_mm_mrad=rider_norm_emittance_y,
        rider_beta_x_m=rider_beta_x,
        rider_beta_y_m=rider_beta_y,
    )


def save_config(options: SimulationOptions, path: Path) -> None:
    ensure_directory(path.parent)
    payload = options.to_dict()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_config(path: Path) -> SimulationOptions:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a JSON object")
    return SimulationOptions.from_dict(payload)


__all__ = [
    "AVAILABLE_DPI_CHOICES",
    "CORE_PARAM_DEFAULTS",
    "CORE_PARAM_LABELS",
    "CORE_REQUIRED_PARAMS",
    "PARAM_LABELS",
    "PARTICLE_PARAM_FIELDS",
    "SimulationOptions",
    "InitialSummary",
    "RunResult",
    "SPECIES_OPTIONS",
    "SPECIES_PRESETS",
    "apply_species_preset",
    "compute_initial_summary",
    "ensure_directory",
    "generate_filename_base",
    "list_config_files",
    "load_config",
    "run_testbed",
    "save_config",
    "supports_driver",
]
