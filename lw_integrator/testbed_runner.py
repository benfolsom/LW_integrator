"""Helper utilities mirroring the notebook testbed inside a desktop GUI.

The original ``integrator_testbed.ipynb`` notebook wires dozens of ipywidgets
around single-run simulation helpers. This module repackages that behaviour
behind plain Python data structures so a Tkinter GUI (or any other front-end)
can drive the same workflows: generating plots, saving down-sampled
trajectories, and managing JSON snapshot files.

All strings deliberately use ASCII to keep packaging simple when rendered in
terminals that do not default to UTF-8.
"""

from __future__ import annotations

import json
import math
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
from core.debug_logger import get_current_log_path, initialize_debug_logging
from core.particle_config import (
    DEFAULT_DRIVER_PARAMS,
    DEFAULT_RIDER_PARAMS,
    PARTICLE_PARAM_FIELDS,
)
from core.particle_status import (
    compute_alive_particle_average,
    format_failure_summary,
    get_alive_particle_values,
    get_particle_failure_summary,
)
from core.self_consistency import canonicalize_self_consistency_mode
from core.types import SimulationType
from input_output.bunch_initialization import create_bunch_from_params

from .trajectory_metrics import compute_delta_energy_components, normalize_state

# ---------------------------------------------------------------------------
# Constants mirroring the notebook defaults
# ---------------------------------------------------------------------------

COLOR_RIDER = "#0072B2"
COLOR_DRIVER = "#D55E00"
SCATTER_STYLE = {"s": 140, "alpha": 0.78, "linewidth": 0, "edgecolors": "none"}
AVAILABLE_DPI_CHOICES: Tuple[int, ...] = (150, 300, 450, 600)
DEFAULT_PLOT_DPI = 300
RADIATION_REACTION_MODE_CHOICES: Tuple[str, ...] = (
    "off",
    "diagnostic_only",
    "power_matched_damping",
    "medina_lad",
)

PARAM_LABELS: Dict[str, str] = {
    "starting_distance": "Start z (mm)",
    "transv_mom": "Transverse momentum spread (amu*mm/ns, ±)",
    "starting_Pz": "Initial Pz (amu*mm/ns)",
    "stripped_ions": "Stripped ions",
    "m_particle": "Mass (amu)",
    "transv_dist": "Transverse spread/radius (mm)",
    "transverse_geometry": "Transverse geometry",
    "transv_offset_x": "Transverse offset x (mm)",
    "transv_offset_y": "Transverse offset y (mm)",
    "pcount": "Particle count (bunch size)",
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
    "startup_mode": "Startup mode",
}

CORE_PARAM_DEFAULTS: Dict[str, Any] = {
    "time_step": 2.2e-7,
    "wall_z": 1.0e5,
    "aperture_radius": 1.0e5,
    "mean": 1.0e5,
    "cav_spacing": 1.0e5,
    "z_cutoff": 0.0,
    "z_cutoff_mode": "absolute",
    "startup_mode": "COLD_START",
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
    gamma_display: bool = False
    gamma_save: bool = False
    gamma_xaxis: str = "t"  # "t" or "z"
    zposition_display: bool = False
    zposition_save: bool = False
    trajectory_save: bool = False
    trajectory_interval: int = 1
    plot_dpi: int = DEFAULT_PLOT_DPI
    output_dir: Path = Path("test_outputs/testbed_runs")
    config_dir: Path = Path("configs/testbed_runs")
    config_name: str = "testbed_config.json"
    rider_params: Dict[str, float | int | str] = field(
        default_factory=lambda: dict(DEFAULT_RIDER_PARAMS)
    )
    driver_params: Optional[Dict[str, float | int | str]] = field(
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

    # Macroparticle simulation options (CONDUCTING_WALL only)
    macroparticle_enabled: bool = False
    macroparticle_charge_multiplier: float = 1.0
    macroparticle_sigma_multiplier: float = 1.0
    macroparticle_use_momentum_errors: bool = True

    # Bounded macroparticle source smearing options
    macroparticle_smearing_enabled: bool = False
    macroparticle_smearing_subcharge_count: int = 8
    macroparticle_smearing_sigma_multiplier: float = 1.0
    macroparticle_smearing_position_sigma_mm: Optional[float] = None
    macroparticle_smearing_longitudinal_sigma_mm: Optional[float] = None
    macroparticle_smearing_momentum_sigma_amu_mm_ns: Optional[float] = None
    macroparticle_smearing_use_position_errors: bool = True
    macroparticle_smearing_use_momentum_errors: bool = True
    macroparticle_smearing_use_centroid_errors: bool = True
    macroparticle_smearing_use_internal_cloud: bool = True
    macroparticle_smearing_apply_to_active_observers: bool = True
    macroparticle_smearing_apply_to_active_sources: bool = True
    macroparticle_smearing_apply_to_passive_sources: bool = True
    macroparticle_smearing_apply_to_passive_updates: bool = False
    macroparticle_smearing_seed: int = 12345
    macroparticle_smearing_refresh_policy: str = "fixed_per_particle"

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
        2  # 0=silent, 1=basic, 2=detailed (prints to console and saved logs)
    )
    self_consistency_chrono_interpolate: bool = (
        False  # Enable chrono-match interpolation for retarded fields
    )
    self_consistency_chrono_tolerance: float = (
        1e-3  # Time residual tolerance for chrono-matching (ns)
    )
    self_consistency_chrono_matching_mode: str = (
        "FAST"  # Chrono-matching mode: "FAST" (default) or "AVERAGED" (internal only)
    )
    self_consistency_chrono_high_precision: bool = (
        False  # Enable cubic interpolation + position interpolation
    )
    self_consistency_chrono_adaptive_tolerance: bool = (
        False  # Auto-set tolerance = 0.1 × timestep
    )
    # Gamma reconciliation options
    self_consistency_gamma_reconciliation_method: str = (
        "DISABLED"  # Method: DISABLED, ADAPTIVE_WEIGHTED, USE_VELOCITY, USE_ENERGY, FIXED_WEIGHTED (default DISABLED for v0.4.8 compatibility)
    )
    self_consistency_gamma_reconciliation_low_beta_threshold: float = (
        0.9  # Below this β: trust energy (for ADAPTIVE_WEIGHTED)
    )
    self_consistency_gamma_reconciliation_high_beta_threshold: float = (
        0.99  # Above this β: trust velocity (for ADAPTIVE_WEIGHTED)
    )
    self_consistency_gamma_reconciliation_low_beta_weight: float = (
        0.8  # Weight α for β < low threshold (for ADAPTIVE_WEIGHTED)
    )
    self_consistency_gamma_reconciliation_high_beta_weight: float = (
        0.2  # Weight α for β > high threshold (for ADAPTIVE_WEIGHTED)
    )
    self_consistency_gamma_reconciliation_mid_beta_weight: float = (
        0.5  # Weight α for mid β range (for ADAPTIVE_WEIGHTED)
    )
    self_consistency_gamma_reconciliation_fixed_weight: float = (
        0.5  # Weight α for FIXED_WEIGHTED method
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
    adaptive_timestep_reduction_factor: int = 3
    # Note: max_refinement_attempts is now auto-calculated in AdaptiveTimestepConfig
    # from reduction_factor and min_timestep_factor to prevent inconsistencies
    adaptive_timestep_min_factor: float = 1e-4

    # Adaptive timestep hysteresis (stay on reduced timestep for stability)
    adaptive_timestep_cooldown_steps: int = 10
    adaptive_timestep_probe_threshold: float = 0.01
    adaptive_timestep_max_probe_steps: int = 3

    adaptive_timestep_debug: bool = False
    # Note: max_substeps_per_step is now auto-calculated in AdaptiveTimestepConfig
    # from min_timestep_factor to prevent time discontinuities

    # Intra-bunch space-charge options
    space_charge_enabled: bool = False
    space_charge_retarded: bool = True
    space_charge_softening_mm: float = 0.0
    space_charge_bunch_sigma_mm: float = 0.01
    space_charge_min_retarded_steps: Optional[int] = None

    # Prescribed external uniform field options
    external_field_enabled: bool = False
    external_electric_field_native: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    external_electric_field_v_per_m: Optional[Tuple[float, float, float]] = None
    external_magnetic_field_native: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    external_field_x_min: Optional[float] = None
    external_field_x_max: Optional[float] = None
    external_field_y_min: Optional[float] = None
    external_field_y_max: Optional[float] = None
    external_field_z_min: Optional[float] = None
    external_field_z_max: Optional[float] = None
    external_field_t_min: Optional[float] = None
    external_field_t_max: Optional[float] = None

    radiation_reaction_mode: str = "medina_lad"

    # Fixed-size physical particle-loss options
    particle_loss_enabled: bool = True
    particle_loss_radius_mm: Optional[float] = 500.0
    particle_loss_conducting_wall_aperture_loss_enabled: bool = True
    particle_loss_initial_radial_quantile: Optional[float] = None
    particle_loss_initial_radial_multiplier: float = 1.0
    particle_loss_initial_radial_margin_mm: float = 0.0

    # Auto-duration crossing mode (BUNCH_TO_BUNCH only)
    auto_duration_enabled: bool = False
    auto_duration_crossing_steps: int = 200
    auto_duration_post_factor: float = 2.0

    # Experimental pseudo-grid options (BUNCH_TO_BUNCH only)
    pseudo_grid_enabled: bool = False
    pseudo_grid_active_rider_count: int = 4
    pseudo_grid_active_driver_count: int = 4
    pseudo_grid_passive_neighbor_count: int = 4
    pseudo_grid_coverage_strategy: str = "farthest_point_staleness"
    pseudo_grid_coverage_space: str = "position"
    pseudo_grid_pair_reuse_window: int = 16
    pseudo_grid_source_weighting_mode: str = "inverse_distance"
    pseudo_grid_loss_tracking_enabled: bool = True
    pseudo_grid_causal_history_pruning_enabled: bool = False
    pseudo_grid_causal_history_safety_margin_steps: int = 2

    # Driver-train options (BUNCH_TO_BUNCH only)
    driver_train_enabled: bool = False
    driver_train_bunch_count: int = 1
    driver_train_z_spacing_mm: float = 0.0
    driver_train_z_offsets_mm: Tuple[float, ...] = field(default_factory=tuple)
    driver_train_prehistory_steps: int = 0
    driver_train_preserve_prehistory_in_output: bool = False

    # Logging options
    save_log_file: bool = False
    log_file_path: Optional[str] = None  # If None, auto-generate in output_dir

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "steps": self.steps,
            "seed": self.seed,
            "simulation_type": self.simulation_type.name,
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
            "gamma_display": self.gamma_display,
            "gamma_save": self.gamma_save,
            "gamma_xaxis": self.gamma_xaxis,
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
            "macroparticle_enabled": self.macroparticle_enabled,
            "macroparticle_charge_multiplier": self.macroparticle_charge_multiplier,
            "macroparticle_sigma_multiplier": self.macroparticle_sigma_multiplier,
            "macroparticle_use_momentum_errors": self.macroparticle_use_momentum_errors,
            "macroparticle_smearing": {
                "enabled": self.macroparticle_smearing_enabled,
                "subcharge_count": self.macroparticle_smearing_subcharge_count,
                "sigma_multiplier": self.macroparticle_smearing_sigma_multiplier,
                "position_sigma_mm": self.macroparticle_smearing_position_sigma_mm,
                "longitudinal_sigma_mm": self.macroparticle_smearing_longitudinal_sigma_mm,
                "momentum_sigma_amu_mm_ns": self.macroparticle_smearing_momentum_sigma_amu_mm_ns,
                "use_position_errors": self.macroparticle_smearing_use_position_errors,
                "use_momentum_errors": self.macroparticle_smearing_use_momentum_errors,
                "use_centroid_errors": self.macroparticle_smearing_use_centroid_errors,
                "use_internal_cloud": self.macroparticle_smearing_use_internal_cloud,
                "apply_to_active_observers": self.macroparticle_smearing_apply_to_active_observers,
                "apply_to_active_sources": self.macroparticle_smearing_apply_to_active_sources,
                "apply_to_passive_sources": self.macroparticle_smearing_apply_to_passive_sources,
                "apply_to_passive_updates": self.macroparticle_smearing_apply_to_passive_updates,
                "seed": self.macroparticle_smearing_seed,
                "refresh_policy": self.macroparticle_smearing_refresh_policy,
            },
            "self_consistency_enabled": self.self_consistency_enabled,
            "self_consistency_tolerance": self.self_consistency_tolerance,
            "self_consistency_convergence_mode": self.self_consistency_convergence_mode,
            "self_consistency_target_ms_tolerance": self.self_consistency_target_ms_tolerance,
            "self_consistency_max_iterations": self.self_consistency_max_iterations,
            "self_consistency_mass_shell_tolerance": self.self_consistency_mass_shell_tolerance,
            "self_consistency_mass_shell_relaxation": self.self_consistency_mass_shell_relaxation,
            "self_consistency_verbosity": self.self_consistency_verbosity,
            "self_consistency_chrono_interpolate": self.self_consistency_chrono_interpolate,
            "self_consistency_chrono_tolerance": self.self_consistency_chrono_tolerance,
            "self_consistency_chrono_matching_mode": self.self_consistency_chrono_matching_mode,
            "self_consistency_chrono_high_precision": self.self_consistency_chrono_high_precision,
            "self_consistency_chrono_adaptive_tolerance": self.self_consistency_chrono_adaptive_tolerance,
            "self_consistency_gamma_reconciliation_method": self.self_consistency_gamma_reconciliation_method,
            "self_consistency_gamma_reconciliation_low_beta_threshold": self.self_consistency_gamma_reconciliation_low_beta_threshold,
            "self_consistency_gamma_reconciliation_high_beta_threshold": self.self_consistency_gamma_reconciliation_high_beta_threshold,
            "self_consistency_gamma_reconciliation_low_beta_weight": self.self_consistency_gamma_reconciliation_low_beta_weight,
            "self_consistency_gamma_reconciliation_high_beta_weight": self.self_consistency_gamma_reconciliation_high_beta_weight,
            "self_consistency_gamma_reconciliation_mid_beta_weight": self.self_consistency_gamma_reconciliation_mid_beta_weight,
            "self_consistency_gamma_reconciliation_fixed_weight": self.self_consistency_gamma_reconciliation_fixed_weight,
            "energy_monitor_enabled": self.energy_monitor_enabled,
            "energy_monitor_threshold": self.energy_monitor_threshold,
            "energy_monitor_check_interval": self.energy_monitor_check_interval,
            "energy_monitor_halt_on_jump": self.energy_monitor_halt_on_jump,
            "energy_monitor_debug": self.energy_monitor_debug,
            "adaptive_timestep_enabled": self.adaptive_timestep_enabled,
            "adaptive_timestep_threshold": self.adaptive_timestep_threshold,
            "adaptive_timestep_reduction_factor": self.adaptive_timestep_reduction_factor,
            # max_refinement_attempts no longer stored (calculated from reduction_factor & min_factor)
            "adaptive_timestep_min_factor": self.adaptive_timestep_min_factor,
            "adaptive_timestep_cooldown_steps": self.adaptive_timestep_cooldown_steps,
            "adaptive_timestep_probe_threshold": self.adaptive_timestep_probe_threshold,
            "adaptive_timestep_max_probe_steps": self.adaptive_timestep_max_probe_steps,
            "adaptive_timestep_debug": self.adaptive_timestep_debug,
            # max_substeps no longer stored - auto-calculated from min_timestep_factor
            "space_charge_enabled": self.space_charge_enabled,
            "space_charge_retarded": self.space_charge_retarded,
            "space_charge_softening_mm": self.space_charge_softening_mm,
            "space_charge_bunch_sigma_mm": self.space_charge_bunch_sigma_mm,
            "space_charge_min_retarded_steps": self.space_charge_min_retarded_steps,
            "external_field_enabled": self.external_field_enabled,
            "external_electric_field_native": list(self.external_electric_field_native),
            "external_electric_field_v_per_m": (
                list(self.external_electric_field_v_per_m)
                if self.external_electric_field_v_per_m is not None
                else None
            ),
            "external_magnetic_field_native": list(self.external_magnetic_field_native),
            "external_field_x_min": self.external_field_x_min,
            "external_field_x_max": self.external_field_x_max,
            "external_field_y_min": self.external_field_y_min,
            "external_field_y_max": self.external_field_y_max,
            "external_field_z_min": self.external_field_z_min,
            "external_field_z_max": self.external_field_z_max,
            "external_field_t_min": self.external_field_t_min,
            "external_field_t_max": self.external_field_t_max,
            "radiation_reaction_mode": self.radiation_reaction_mode,
            "particle_loss": {
                "enabled": self.particle_loss_enabled,
                "loss_radius_mm": self.particle_loss_radius_mm,
                "conducting_wall_aperture_loss_enabled": self.particle_loss_conducting_wall_aperture_loss_enabled,
                "initial_radial_quantile": self.particle_loss_initial_radial_quantile,
                "initial_radial_multiplier": self.particle_loss_initial_radial_multiplier,
                "initial_radial_margin_mm": self.particle_loss_initial_radial_margin_mm,
            },
            "auto_duration_enabled": self.auto_duration_enabled,
            "auto_duration_crossing_steps": self.auto_duration_crossing_steps,
            "auto_duration_post_factor": self.auto_duration_post_factor,
            "pseudo_grid": {
                "enabled": self.pseudo_grid_enabled,
                "active_rider_count": self.pseudo_grid_active_rider_count,
                "active_driver_count": self.pseudo_grid_active_driver_count,
                "passive_neighbor_count": self.pseudo_grid_passive_neighbor_count,
                "coverage_strategy": self.pseudo_grid_coverage_strategy,
                "coverage_space": self.pseudo_grid_coverage_space,
                "pair_reuse_window": self.pseudo_grid_pair_reuse_window,
                "source_weighting_mode": self.pseudo_grid_source_weighting_mode,
                "loss_tracking_enabled": self.pseudo_grid_loss_tracking_enabled,
                "causal_history_pruning_enabled": self.pseudo_grid_causal_history_pruning_enabled,
                "causal_history_safety_margin_steps": self.pseudo_grid_causal_history_safety_margin_steps,
            },
            "driver_train": {
                "enabled": self.driver_train_enabled,
                "bunch_count": self.driver_train_bunch_count,
                "z_spacing_mm": self.driver_train_z_spacing_mm,
                "z_offsets_mm": list(self.driver_train_z_offsets_mm),
                "prehistory_steps": self.driver_train_prehistory_steps,
                "preserve_prehistory_in_output": self.driver_train_preserve_prehistory_in_output,
            },
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

        def _str(name: str, default: str) -> str:
            value = payload.get(name, default)
            return str(value) if value is not None else default

        def _optional_float(name: str) -> Optional[float]:
            value = payload.get(name)
            if value is None:
                return None
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

        def _tuple3(name: str) -> Optional[Tuple[float, float, float]]:
            value = payload.get(name)
            if value is None:
                return None
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                return None
            try:
                return (float(value[0]), float(value[1]), float(value[2]))
            except (TypeError, ValueError):
                return None

        particle_loss_payload_raw = payload.get("particle_loss")
        particle_loss_payload = (
            particle_loss_payload_raw
            if isinstance(particle_loss_payload_raw, dict)
            else {}
        )

        def _particle_loss_value(name: str, default: object) -> object:
            flat_name = f"particle_loss_{name}"
            if flat_name in payload:
                return payload.get(flat_name, default)
            return particle_loss_payload.get(name, default)

        def _particle_loss_bool(name: str, default: bool) -> bool:
            return bool(_particle_loss_value(name, default))

        def _particle_loss_enabled(default: bool) -> bool:
            value = _particle_loss_value("enabled", None)
            if value is not None:
                return bool(value)
            return (
                any(
                    _particle_loss_value(key, None) is not None
                    for key in ("loss_radius_mm", "initial_radial_quantile")
                )
                or default
            )

        def _particle_loss_float(name: str, default: float) -> float:
            value = _particle_loss_value(name, default)
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        def _particle_loss_optional_float(
            name: str,
            default: Optional[float] = None,
        ) -> Optional[float]:
            value = _particle_loss_value(name, default)
            if value is None:
                return None
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        pseudo_grid_payload_raw = payload.get("pseudo_grid")
        pseudo_grid_payload = (
            pseudo_grid_payload_raw if isinstance(pseudo_grid_payload_raw, dict) else {}
        )

        def _pseudo_value(name: str, default: object) -> object:
            flat_name = f"pseudo_grid_{name}"
            if flat_name in payload:
                return payload.get(flat_name, default)
            return pseudo_grid_payload.get(name, default)

        def _pseudo_bool(name: str, default: bool) -> bool:
            return bool(_pseudo_value(name, default))

        def _pseudo_int(name: str, default: int) -> int:
            value = _pseudo_value(name, default)
            try:
                return int(value)  # type: ignore[arg-type,no-any-return,call-overload]
            except (TypeError, ValueError):
                return default

        def _pseudo_str(name: str, default: str) -> str:
            value = _pseudo_value(name, default)
            return str(value) if value is not None else default

        smearing_payload_raw = payload.get("macroparticle_smearing")
        smearing_payload = (
            smearing_payload_raw if isinstance(smearing_payload_raw, dict) else {}
        )

        def _smearing_value(name: str, default: object) -> object:
            flat_name = f"macroparticle_smearing_{name}"
            if flat_name in payload:
                return payload.get(flat_name, default)
            return smearing_payload.get(name, default)

        def _smearing_bool(name: str, default: bool) -> bool:
            return bool(_smearing_value(name, default))

        def _smearing_int(name: str, default: int) -> int:
            value = _smearing_value(name, default)
            try:
                return int(value)  # type: ignore[arg-type,no-any-return,call-overload]
            except (TypeError, ValueError):
                return default

        def _smearing_float(name: str, default: float) -> float:
            value = _smearing_value(name, default)
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        def _smearing_optional_float(name: str) -> Optional[float]:
            value = _smearing_value(name, None)
            if value in (None, ""):
                return None
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

        driver_train_payload_raw = payload.get("driver_train")
        driver_train_payload = (
            driver_train_payload_raw
            if isinstance(driver_train_payload_raw, dict)
            else {}
        )

        def _driver_train_value(name: str, default: object) -> object:
            flat_name = f"driver_train_{name}"
            if flat_name in payload:
                return payload.get(flat_name, default)
            return driver_train_payload.get(name, default)

        def _driver_train_bool(name: str, default: bool) -> bool:
            return bool(_driver_train_value(name, default))

        def _driver_train_int(name: str, default: int) -> int:
            value = _driver_train_value(name, default)
            try:
                return int(value)  # type: ignore[arg-type,no-any-return,call-overload]
            except (TypeError, ValueError):
                return default

        def _driver_train_float(name: str, default: float) -> float:
            value = _driver_train_value(name, default)
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        def _driver_train_offsets() -> Tuple[float, ...]:
            value = _driver_train_value("z_offsets_mm", ())
            if value in (None, ""):
                return ()
            if not isinstance(value, (list, tuple)):
                return ()
            try:
                return tuple(float(item) for item in value)
            except (TypeError, ValueError):
                return ()

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

        driver_params: Optional[Dict[str, float | int | str]]
        driver_payload = payload.get("driver_params")
        if isinstance(driver_payload, dict):
            driver_params = dict(DEFAULT_DRIVER_PARAMS)
            driver_params.update(driver_payload)
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
            gamma_display=_bool("gamma_display", False),
            gamma_save=_bool("gamma_save", False),
            gamma_xaxis=str(payload.get("gamma_xaxis", "t")),
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
            macroparticle_enabled=_bool("macroparticle_enabled", False),
            macroparticle_charge_multiplier=_float(
                "macroparticle_charge_multiplier", 1.0
            ),
            macroparticle_sigma_multiplier=_float(
                "macroparticle_sigma_multiplier", 1.0
            ),
            macroparticle_use_momentum_errors=_bool(
                "macroparticle_use_momentum_errors", True
            ),
            macroparticle_smearing_enabled=_smearing_bool("enabled", False),
            macroparticle_smearing_subcharge_count=_smearing_int("subcharge_count", 8),
            macroparticle_smearing_sigma_multiplier=_smearing_float(
                "sigma_multiplier", 1.0
            ),
            macroparticle_smearing_position_sigma_mm=_smearing_optional_float(
                "position_sigma_mm"
            ),
            macroparticle_smearing_longitudinal_sigma_mm=_smearing_optional_float(
                "longitudinal_sigma_mm"
            ),
            macroparticle_smearing_momentum_sigma_amu_mm_ns=_smearing_optional_float(
                "momentum_sigma_amu_mm_ns"
            ),
            macroparticle_smearing_use_position_errors=_smearing_bool(
                "use_position_errors", True
            ),
            macroparticle_smearing_use_momentum_errors=_smearing_bool(
                "use_momentum_errors", True
            ),
            macroparticle_smearing_use_centroid_errors=_smearing_bool(
                "use_centroid_errors", True
            ),
            macroparticle_smearing_use_internal_cloud=_smearing_bool(
                "use_internal_cloud", True
            ),
            macroparticle_smearing_apply_to_active_observers=_smearing_bool(
                "apply_to_active_observers", True
            ),
            macroparticle_smearing_apply_to_active_sources=_smearing_bool(
                "apply_to_active_sources", True
            ),
            macroparticle_smearing_apply_to_passive_sources=_smearing_bool(
                "apply_to_passive_sources", True
            ),
            macroparticle_smearing_apply_to_passive_updates=_smearing_bool(
                "apply_to_passive_updates", False
            ),
            macroparticle_smearing_seed=_smearing_int("seed", 12345),
            macroparticle_smearing_refresh_policy=str(
                _smearing_value("refresh_policy", "fixed_per_particle")
            ).replace("-", "_"),
            self_consistency_enabled=_bool("self_consistency_enabled", True),
            self_consistency_tolerance=_float("self_consistency_tolerance", 1e-4),
            self_consistency_convergence_mode=canonicalize_self_consistency_mode(
                payload.get("self_consistency_convergence_mode", "fixed_geometry")
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
            self_consistency_chrono_interpolate=_bool(
                "self_consistency_chrono_interpolate", False
            ),
            self_consistency_chrono_tolerance=_float(
                "self_consistency_chrono_tolerance", 1e-3
            ),
            self_consistency_chrono_matching_mode=_str(
                "self_consistency_chrono_matching_mode", "FAST"
            ),
            self_consistency_chrono_high_precision=_bool(
                "self_consistency_chrono_high_precision", False
            ),
            self_consistency_chrono_adaptive_tolerance=_bool(
                "self_consistency_chrono_adaptive_tolerance", False
            ),
            energy_monitor_enabled=_bool("energy_monitor_enabled", True),
            energy_monitor_threshold=_float("energy_monitor_threshold", 2.0),
            energy_monitor_check_interval=_int("energy_monitor_check_interval", 10),
            energy_monitor_halt_on_jump=_bool("energy_monitor_halt_on_jump", False),
            energy_monitor_debug=_bool("energy_monitor_debug", False),
            adaptive_timestep_enabled=_bool("adaptive_timestep_enabled", True),
            adaptive_timestep_threshold=_float("adaptive_timestep_threshold", 0.10),
            adaptive_timestep_reduction_factor=_int(
                "adaptive_timestep_reduction_factor", 3
            ),
            # max_refinement_attempts no longer loaded (calculated automatically)
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
            # max_substeps no longer loaded - auto-calculated from min_timestep_factor
            space_charge_enabled=_bool("space_charge_enabled", False),
            space_charge_retarded=_bool("space_charge_retarded", True),
            space_charge_softening_mm=_float("space_charge_softening_mm", 0.0),
            space_charge_bunch_sigma_mm=_float("space_charge_bunch_sigma_mm", 0.01),
            space_charge_min_retarded_steps=(
                _int("space_charge_min_retarded_steps", 0)
                if payload.get("space_charge_min_retarded_steps") is not None
                else None
            ),
            external_field_enabled=_bool("external_field_enabled", False),
            external_electric_field_native=_tuple3("external_electric_field_native")
            or (0.0, 0.0, 0.0),
            external_electric_field_v_per_m=_tuple3("external_electric_field_v_per_m"),
            external_magnetic_field_native=_tuple3("external_magnetic_field_native")
            or (0.0, 0.0, 0.0),
            external_field_x_min=_optional_float("external_field_x_min"),
            external_field_x_max=_optional_float("external_field_x_max"),
            external_field_y_min=_optional_float("external_field_y_min"),
            external_field_y_max=_optional_float("external_field_y_max"),
            external_field_z_min=_optional_float("external_field_z_min"),
            external_field_z_max=_optional_float("external_field_z_max"),
            external_field_t_min=_optional_float("external_field_t_min"),
            external_field_t_max=_optional_float("external_field_t_max"),
            radiation_reaction_mode=_str("radiation_reaction_mode", "medina_lad"),
            particle_loss_enabled=_particle_loss_enabled(True),
            particle_loss_radius_mm=_particle_loss_optional_float(
                "loss_radius_mm",
                500.0,
            ),
            particle_loss_conducting_wall_aperture_loss_enabled=_particle_loss_bool(
                "conducting_wall_aperture_loss_enabled",
                True,
            ),
            particle_loss_initial_radial_quantile=_particle_loss_optional_float(
                "initial_radial_quantile"
            ),
            particle_loss_initial_radial_multiplier=_particle_loss_float(
                "initial_radial_multiplier",
                1.0,
            ),
            particle_loss_initial_radial_margin_mm=_particle_loss_float(
                "initial_radial_margin_mm",
                0.0,
            ),
            auto_duration_enabled=_bool("auto_duration_enabled", False),
            auto_duration_crossing_steps=_int("auto_duration_crossing_steps", 200),
            auto_duration_post_factor=_float("auto_duration_post_factor", 2.0),
            pseudo_grid_enabled=_pseudo_bool("enabled", False),
            pseudo_grid_active_rider_count=_pseudo_int("active_rider_count", 4),
            pseudo_grid_active_driver_count=_pseudo_int("active_driver_count", 4),
            pseudo_grid_passive_neighbor_count=_pseudo_int("passive_neighbor_count", 4),
            pseudo_grid_coverage_strategy=_pseudo_str(
                "coverage_strategy", "farthest_point_staleness"
            ),
            pseudo_grid_coverage_space=_pseudo_str("coverage_space", "position"),
            pseudo_grid_pair_reuse_window=_pseudo_int("pair_reuse_window", 16),
            pseudo_grid_source_weighting_mode=_pseudo_str(
                "source_weighting_mode", "inverse_distance"
            ),
            pseudo_grid_loss_tracking_enabled=_pseudo_bool(
                "loss_tracking_enabled", True
            ),
            pseudo_grid_causal_history_pruning_enabled=_pseudo_bool(
                "causal_history_pruning_enabled", False
            ),
            pseudo_grid_causal_history_safety_margin_steps=_pseudo_int(
                "causal_history_safety_margin_steps", 2
            ),
            driver_train_enabled=_driver_train_bool("enabled", False),
            driver_train_bunch_count=_driver_train_int("bunch_count", 1),
            driver_train_z_spacing_mm=_driver_train_float("z_spacing_mm", 0.0),
            driver_train_z_offsets_mm=_driver_train_offsets(),
            driver_train_prehistory_steps=_driver_train_int("prehistory_steps", 0),
            driver_train_preserve_prehistory_in_output=_driver_train_bool(
                "preserve_prehistory_in_output",
                False,
            ),
            self_consistency_gamma_reconciliation_method=_str(
                "self_consistency_gamma_reconciliation_method", "DISABLED"
            ),
            self_consistency_gamma_reconciliation_low_beta_threshold=_float(
                "self_consistency_gamma_reconciliation_low_beta_threshold", 0.9
            ),
            self_consistency_gamma_reconciliation_high_beta_threshold=_float(
                "self_consistency_gamma_reconciliation_high_beta_threshold", 0.99
            ),
            self_consistency_gamma_reconciliation_low_beta_weight=_float(
                "self_consistency_gamma_reconciliation_low_beta_weight", 0.8
            ),
            self_consistency_gamma_reconciliation_high_beta_weight=_float(
                "self_consistency_gamma_reconciliation_high_beta_weight", 0.2
            ),
            self_consistency_gamma_reconciliation_mid_beta_weight=_float(
                "self_consistency_gamma_reconciliation_mid_beta_weight", 0.5
            ),
            self_consistency_gamma_reconciliation_fixed_weight=_float(
                "self_consistency_gamma_reconciliation_fixed_weight", 0.5
            ),
            save_log_file=_bool("save_log_file", False),
            log_file_path=(
                str(payload.get("log_file_path"))
                if payload.get("log_file_path") is not None
                else None
            ),
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
    # For small-angle approximation: x' ≈ tan(θ) ≈ Px/Pz
    # This is exact for the divergence angle in the paraxial limit
    xp = Px / Pz  # dimensionless (mm/ns / mm/ns)
    yp = Py / Pz  # dimensionless

    # Calculate RMS quantities
    x_rms = np.sqrt(np.mean(x**2))  # mm
    y_rms = np.sqrt(np.mean(y**2))  # mm
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


def _empty_distribution_summary() -> Dict[str, float]:
    return {
        "alive_count": 0.0,
        "total_count": 0.0,
        "alive_fraction": 0.0,
    }


def _summary_series(summaries: list[Dict[str, float]], key: str) -> np.ndarray:
    return np.array([summary.get(key, 0.0) for summary in summaries], dtype=float)


def _alive_average_series(
    states: list[Dict[str, Any]],
    field: str,
    *,
    default: float,
) -> np.ndarray:
    return np.array(
        [compute_alive_particle_average(state, field) or default for state in states],
        dtype=float,
    )


def _mean_alive_gamma(state: Dict[str, Any]) -> float | None:
    return compute_alive_particle_average(state, "gamma")


def _mean_alive_gamma_series(states: list[Dict[str, Any]]) -> np.ndarray:
    return _alive_average_series(states, "gamma", default=1.0)


def _compute_alive_particle_radial_summary(
    state: Dict[str, Any],
    *,
    initial_rms_radius_mm: float | None = None,
) -> Dict[str, float]:
    x_alive = get_alive_particle_values(state, "x")
    y_alive = get_alive_particle_values(state, "y")
    total_count = float(len(np.asarray(state.get("x", []))))
    if x_alive is None or y_alive is None or len(x_alive) == 0 or len(y_alive) == 0:
        summary = _empty_distribution_summary()
        summary["total_count"] = total_count
        return summary

    radii = np.sqrt(
        np.asarray(x_alive, dtype=float) ** 2 + np.asarray(y_alive, dtype=float) ** 2
    )
    if len(radii) == 0:
        summary = _empty_distribution_summary()
        summary["total_count"] = total_count
        return summary

    summary = {
        "alive_count": float(len(radii)),
        "total_count": total_count,
        "alive_fraction": float(len(radii) / total_count) if total_count > 0 else 0.0,
        "r_mean_particle": float(np.mean(radii)),
        "r_rms_particle": float(np.sqrt(np.mean(radii**2))),
    }
    for percentile in (50, 68, 90, 95, 99):
        summary[f"r_p{percentile}_particle"] = float(np.percentile(radii, percentile))

    if initial_rms_radius_mm is not None and initial_rms_radius_mm > 0.0:
        for multiplier in (2, 3, 5):
            threshold = multiplier * initial_rms_radius_mm
            summary[f"halo_gt_{multiplier}_initial_rms_fraction"] = float(
                np.mean(radii > threshold)
            )

    return summary


def _compute_alive_particle_radial_stats(state: Dict[str, Any]) -> tuple[float, float]:
    summary = _compute_alive_particle_radial_summary(state)
    return summary.get("r_mean_particle", 0.0), summary.get("r_rms_particle", 0.0)


def _compute_alive_particle_longitudinal_summary(
    state: Dict[str, Any],
) -> Dict[str, float]:
    z_alive = get_alive_particle_values(state, "z")
    total_count = float(len(np.asarray(state.get("z", []))))
    if z_alive is None or len(z_alive) == 0:
        summary = _empty_distribution_summary()
        summary["total_count"] = total_count
        return summary

    z_values = np.asarray(z_alive, dtype=float)
    summary = {
        "alive_count": float(len(z_values)),
        "total_count": total_count,
        "alive_fraction": (
            float(len(z_values) / total_count) if total_count > 0 else 0.0
        ),
        "z_std_particle": float(np.std(z_values)),
    }
    percentiles = {
        "p01": 1,
        "p05": 5,
        "p50": 50,
        "p95": 95,
        "p99": 99,
    }
    percentile_values = {
        label: float(np.percentile(z_values, percentile))
        for label, percentile in percentiles.items()
    }
    for label, value in percentile_values.items():
        summary[f"z_{label}_particle"] = value
    summary["z_width_p90_particle"] = (
        percentile_values["p95"] - percentile_values["p05"]
    )
    summary["z_width_p98_particle"] = (
        percentile_values["p99"] - percentile_values["p01"]
    )
    return summary


def _compute_alive_particle_momentum_summary(state: Dict[str, Any]) -> Dict[str, float]:
    gamma_alive = get_alive_particle_values(state, "gamma")
    pz_alive = get_alive_particle_values(state, "Pz")
    m_alive = get_alive_particle_values(state, "m")
    if gamma_alive is None or len(gamma_alive) == 0:
        return {"gamma_std_particle": 0.0, "pz_std_particle": 0.0}

    summary = {
        "gamma_std_particle": float(np.std(np.asarray(gamma_alive, dtype=float)))
    }
    if pz_alive is not None and m_alive is not None and len(pz_alive) > 0:
        normalized_pz = np.asarray(pz_alive, dtype=float) / (
            np.asarray(m_alive, dtype=float) * C_MMNS
        )
        summary["pz_std_particle"] = float(np.std(normalized_pz))
    else:
        summary["pz_std_particle"] = 0.0
    return summary


@dataclass
class RunResult:
    metrics: Optional[Dict[str, Dict[str, float]]]
    saved_paths: Dict[str, Path]
    figures: Dict[str, Figure]
    logs: List[str]
    verbose_logs: str  # Captured stdout/stderr from verbose integration output
    duration_s: float
    filename_base: str
    debug_log_path: Optional[Path] = None
    # Additional computed values for optimization
    rider_delta_e: Optional[float] = None  # Final energy change in MeV
    rider_gamma_initial: Optional[float] = None
    rider_gamma_final: Optional[float] = None
    rider_trajectory: Optional[Dict[str, Any]] = None
    driver_gamma_initial: Optional[float] = None
    driver_gamma_final: Optional[float] = None
    driver_trajectory: Optional[Dict[str, Any]] = None
    # Beam optics parameters (initial)
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
    # Early termination tracking
    halted_early: bool = False  # True if integration was halted before completion
    halt_reason: Optional[str] = (
        None  # Reason for early halt (e.g., "gamma_blowup", "distance_reached")
    )
    # Particle failure tracking
    num_particles_dead: int = 0  # Number of particles that failed during simulation
    particle_failure_info: Optional[Dict[int, Dict]] = (
        None  # Detailed failure info per particle
    )


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
    base = config_name.strip().removesuffix(".json") or "testbed_config"
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


def prepare_particle_bunches(
    seed: int,
    *,
    rider_params: Dict[str, Any],
    driver_params: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray] | None, float, float | None]:
    """Prepare rider and driver particle bunches.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    rider_params : dict
        Rider particle parameters
    driver_params : dict, optional
        Driver particle parameters (None for single-bunch modes)
    Returns
    -------
    rider_state : dict
        Rider particle state
    driver_state : dict or None
        Driver particle state (None if not provided)
    rider_rest_mev : float
        Rider rest energy in MeV
    driver_rest_mev : float or None
        Driver rest energy in MeV (None if not provided)
    """
    rider_state, rider_rest_mev = create_bunch_from_params(
        seed=seed,
        **rider_params,
    )

    if driver_params is not None:
        driver_state, driver_rest_mev = create_bunch_from_params(
            seed=seed + 1,  # Different seed for driver
            **driver_params,
        )
    else:
        driver_state = None
        driver_rest_mev = None

    return rider_state, driver_state, rider_rest_mev, driver_rest_mev


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

    # Always use the maintained core initialization path here.
    rider_state, driver_state, rider_rest_mev, driver_rest_mev = (
        prepare_particle_bunches(
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


def build_self_consistency_config(options: SimulationOptions) -> Optional[object]:
    """Build SelfConsistencyConfig from SimulationOptions.

    Returns None if self_consistency is disabled.
    """
    if not options.self_consistency_enabled:
        return None

    from core.self_consistency import SelfConsistencyConfig
    from core.types import GammaReconciliationMethod

    # Parse gamma reconciliation method string to enum
    method_str = options.self_consistency_gamma_reconciliation_method.upper()
    try:
        gamma_method = GammaReconciliationMethod[method_str]
    except KeyError:
        # Fallback to ADAPTIVE_WEIGHTED if invalid method specified
        gamma_method = GammaReconciliationMethod.ADAPTIVE_WEIGHTED

    return SelfConsistencyConfig(
        enabled=True,
        convergence_mode=options.self_consistency_convergence_mode,
        target_ms_tolerance=options.self_consistency_target_ms_tolerance,
        max_iterations=options.self_consistency_max_iterations,
        mass_shell_tolerance=options.self_consistency_mass_shell_tolerance,
        mass_shell_relaxation=options.self_consistency_mass_shell_relaxation,
        verbosity=options.self_consistency_verbosity,
        chrono_interpolate=options.self_consistency_chrono_interpolate,
        chrono_tolerance=options.self_consistency_chrono_tolerance,
        chrono_matching_mode=options.self_consistency_chrono_matching_mode,
        chrono_high_precision=options.self_consistency_chrono_high_precision,
        chrono_adaptive_tolerance=options.self_consistency_chrono_adaptive_tolerance,
        gamma_reconciliation_method=gamma_method,
        gamma_reconciliation_low_beta_threshold=options.self_consistency_gamma_reconciliation_low_beta_threshold,
        gamma_reconciliation_high_beta_threshold=options.self_consistency_gamma_reconciliation_high_beta_threshold,
        gamma_reconciliation_low_beta_weight=options.self_consistency_gamma_reconciliation_low_beta_weight,
        gamma_reconciliation_high_beta_weight=options.self_consistency_gamma_reconciliation_high_beta_weight,
        gamma_reconciliation_mid_beta_weight=options.self_consistency_gamma_reconciliation_mid_beta_weight,
        gamma_reconciliation_fixed_weight=options.self_consistency_gamma_reconciliation_fixed_weight,
    )


def build_energy_monitor_config(options: SimulationOptions) -> Optional[object]:
    """Build EnergyMonitorConfig from SimulationOptions.

    Returns None if energy_monitor is disabled.
    """
    if not options.energy_monitor_enabled:
        return None

    from core.integration_runner import EnergyMonitorConfig

    return EnergyMonitorConfig(
        enabled=True,
        relative_threshold=options.energy_monitor_threshold,
        check_interval=options.energy_monitor_check_interval,
        halt_on_jump=options.energy_monitor_halt_on_jump,
        debug=options.energy_monitor_debug,
    )


def build_adaptive_timestep_config(options: SimulationOptions) -> Optional[object]:
    """Build AdaptiveTimestepConfig from SimulationOptions.

    Returns None if adaptive_timestep is disabled.
    """
    if not options.adaptive_timestep_enabled:
        return None

    from core.integration_runner import AdaptiveTimestepConfig

    return AdaptiveTimestepConfig(
        enabled=True,
        energy_jump_threshold=options.adaptive_timestep_threshold,
        timestep_reduction_factor=options.adaptive_timestep_reduction_factor,
        # max_refinement_attempts is now a calculated property, not passed as parameter
        min_timestep_factor=options.adaptive_timestep_min_factor,
        cooldown_steps=options.adaptive_timestep_cooldown_steps,
        probe_threshold=options.adaptive_timestep_probe_threshold,
        max_probe_steps=options.adaptive_timestep_max_probe_steps,
        # max_substeps_per_step is now a calculated property, not passed as parameter
        debug=options.adaptive_timestep_debug,
    )


def build_particle_loss_config(options: SimulationOptions) -> object:
    """Build ParticleLossConfig from SimulationOptions."""
    from core.types import ParticleLossConfig

    return ParticleLossConfig(
        enabled=bool(options.particle_loss_enabled),
        loss_radius_mm=options.particle_loss_radius_mm,
        conducting_wall_aperture_loss_enabled=bool(
            options.particle_loss_conducting_wall_aperture_loss_enabled
        ),
        initial_radial_quantile=options.particle_loss_initial_radial_quantile,
        initial_radial_multiplier=float(
            options.particle_loss_initial_radial_multiplier
        ),
        initial_radial_margin_mm=float(options.particle_loss_initial_radial_margin_mm),
    )


def build_pseudo_grid_config(options: SimulationOptions) -> object:
    """Build PseudoGridConfig from SimulationOptions."""
    from core.types import PseudoGridConfig

    return PseudoGridConfig(
        enabled=bool(options.pseudo_grid_enabled),
        active_rider_count=int(options.pseudo_grid_active_rider_count),
        active_driver_count=int(options.pseudo_grid_active_driver_count),
        passive_neighbor_count=int(options.pseudo_grid_passive_neighbor_count),
        coverage_strategy=str(options.pseudo_grid_coverage_strategy),
        coverage_space=str(options.pseudo_grid_coverage_space),
        pair_reuse_window=int(options.pseudo_grid_pair_reuse_window),
        source_weighting_mode=str(options.pseudo_grid_source_weighting_mode),
        loss_tracking_enabled=bool(options.pseudo_grid_loss_tracking_enabled),
        causal_history_pruning_enabled=bool(
            options.pseudo_grid_causal_history_pruning_enabled
        ),
        causal_history_safety_margin_steps=int(
            options.pseudo_grid_causal_history_safety_margin_steps
        ),
    )


def build_macroparticle_smearing_config(options: SimulationOptions) -> object:
    """Build MacroparticleSmearingConfig from SimulationOptions."""
    from core.types import MacroparticleSmearingConfig

    return MacroparticleSmearingConfig(
        enabled=bool(options.macroparticle_smearing_enabled),
        subcharge_count=int(options.macroparticle_smearing_subcharge_count),
        sigma_multiplier=float(options.macroparticle_smearing_sigma_multiplier),
        position_sigma_mm=options.macroparticle_smearing_position_sigma_mm,
        longitudinal_sigma_mm=options.macroparticle_smearing_longitudinal_sigma_mm,
        momentum_sigma_amu_mm_ns=options.macroparticle_smearing_momentum_sigma_amu_mm_ns,
        use_position_errors=bool(options.macroparticle_smearing_use_position_errors),
        use_momentum_errors=bool(options.macroparticle_smearing_use_momentum_errors),
        use_centroid_errors=bool(options.macroparticle_smearing_use_centroid_errors),
        use_internal_cloud=bool(options.macroparticle_smearing_use_internal_cloud),
        apply_to_active_observers=bool(
            options.macroparticle_smearing_apply_to_active_observers
        ),
        apply_to_active_sources=bool(
            options.macroparticle_smearing_apply_to_active_sources
        ),
        apply_to_passive_sources=bool(
            options.macroparticle_smearing_apply_to_passive_sources
        ),
        apply_to_passive_updates=bool(
            options.macroparticle_smearing_apply_to_passive_updates
        ),
        seed=int(options.macroparticle_smearing_seed),
        refresh_policy=str(options.macroparticle_smearing_refresh_policy).replace(
            "-", "_"
        ),
    )


def build_driver_train_config(options: SimulationOptions) -> object:
    """Build DriverTrainConfig from SimulationOptions."""
    from core.types import DriverTrainConfig

    return DriverTrainConfig(
        enabled=bool(options.driver_train_enabled),
        bunch_count=int(options.driver_train_bunch_count),
        z_spacing_mm=float(options.driver_train_z_spacing_mm),
        z_offsets_mm=tuple(float(value) for value in options.driver_train_z_offsets_mm),
        prehistory_steps=int(options.driver_train_prehistory_steps),
        preserve_prehistory_in_output=bool(
            options.driver_train_preserve_prehistory_in_output
        ),
    )


def build_space_charge_config(options: SimulationOptions) -> Optional[object]:
    """Build SpaceChargeConfig from SimulationOptions.

    Returns None if space charge is disabled.
    """
    if not options.space_charge_enabled:
        return None

    from core.types import SpaceChargeConfig

    return SpaceChargeConfig(
        enabled=True,
        retarded=options.space_charge_retarded,
        softening_mm=options.space_charge_softening_mm,
        bunch_sigma_mm=options.space_charge_bunch_sigma_mm,
        min_retarded_steps=options.space_charge_min_retarded_steps,
    )


def build_external_field_config(options: SimulationOptions) -> Optional[object]:
    """Build ExternalFieldConfig from SimulationOptions.

    Returns None if prescribed external fields are disabled.
    """
    if not options.external_field_enabled:
        return None

    from core.external_fields import electric_field_v_per_m_to_native
    from core.types import ExternalFieldConfig

    electric_native = tuple(float(v) for v in options.external_electric_field_native)
    if options.external_electric_field_v_per_m is not None:
        electric_native = tuple(
            electric_field_v_per_m_to_native(float(v))
            for v in options.external_electric_field_v_per_m
        )

    return ExternalFieldConfig(
        enabled=True,
        electric_field_native=electric_native,
        magnetic_field_native=tuple(
            float(v) for v in options.external_magnetic_field_native
        ),
        x_min=options.external_field_x_min,
        x_max=options.external_field_x_max,
        y_min=options.external_field_y_min,
        y_max=options.external_field_y_max,
        z_min=options.external_field_z_min,
        z_max=options.external_field_z_max,
        t_min=options.external_field_t_min,
        t_max=options.external_field_t_max,
    )


def build_chrono_mode_enum(chrono_mode_str: str) -> object:
    """Convert chrono mode string to ChronoMatchingMode enum."""
    from core.types import ChronoMatchingMode

    chrono_mode_upper = chrono_mode_str.upper()
    if chrono_mode_upper == "FAST":
        return ChronoMatchingMode.FAST
    elif chrono_mode_upper == "AVERAGED":
        return ChronoMatchingMode.AVERAGED
    else:
        # Keep old configs running, but default invalid values to the maintained mode.
        return ChronoMatchingMode.FAST


def build_startup_mode_enum(startup_mode_str: str) -> object:
    """Convert startup mode string to StartupMode enum."""
    from core.types import StartupMode

    startup_mode_upper = startup_mode_str.upper()
    if startup_mode_upper == "COLD_START":
        return StartupMode.COLD_START
    elif startup_mode_upper == "APPROXIMATE_BACK_HISTORY":
        return StartupMode.APPROXIMATE_BACK_HISTORY
    else:
        # Default to COLD_START if invalid
        return StartupMode.COLD_START


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

    energy_display = bool(options.energy_display)
    energy_save = bool(options.energy_save)
    transverse_display = bool(options.transverse_display)
    transverse_save = bool(options.transverse_save)
    beta_display = bool(options.beta_display)
    beta_save = bool(options.beta_save)
    momentum_display = bool(options.momentum_display)
    momentum_save = bool(options.momentum_save)
    gamma_display = bool(options.gamma_display)
    gamma_save = bool(options.gamma_save)
    zposition_display = bool(options.zposition_display)
    zposition_save = bool(options.zposition_save)
    trajectory_save = bool(options.trajectory_save)

    should_save = any(
        [
            energy_save,
            transverse_save,
            beta_save,
            momentum_save,
            gamma_save,
            zposition_save,
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

    # Start a fresh testbed log for each run so GUI and CLI single runs can
    # save the exact debug session instead of guessing the latest file later.
    initialize_debug_logging(context="testbed", force_new_log=True)
    debug_log_path = get_current_log_path()

    _log(
        f"Running {sim_type.name.replace('_', ' ').title()} integrator for {options.steps} steps"
    )
    if debug_log_path is not None:
        _log(f"  Debug log: {debug_log_path}")
    _log(f"  Steps: {options.steps}")
    _log(f"  Seed: {options.seed}")
    _log(f"  Core params: {filtered_core_params}")
    _log(f"  Image subcharge count: {options.image_subcharge_count}")
    _log(f"  Image weighting: {options.use_image_weighting}")
    if options.macroparticle_enabled and sim_type == SimulationType.CONDUCTING_WALL:
        _log("  Macroparticle simulation: ENABLED")
        _log(f"    Charge multiplier: {options.macroparticle_charge_multiplier}")
        _log(f"    Sigma multiplier: {options.macroparticle_sigma_multiplier}")
        _log(f"    Use momentum errors: {options.macroparticle_use_momentum_errors}")
        _log(
            f"    Bunch transv_dist: {options.rider_params.get('transv_dist', 0.0)} mm"
        )
        _log(f"    Bunch transv_mom: {options.rider_params.get('transv_mom', 0.0)}")
    _log(
        f"  Self-consistency: {options.self_consistency_enabled} (mode={options.self_consistency_convergence_mode}, "
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
    _log(f"  Radiation reaction: {options.radiation_reaction_mode}")
    if options.driver_train_enabled and sim_type == SimulationType.BUNCH_TO_BUNCH:
        _log(
            "  Driver train: enabled "
            f"(bunches={options.driver_train_bunch_count}, "
            f"spacing={options.driver_train_z_spacing_mm} mm, "
            f"prehistory_steps={options.driver_train_prehistory_steps})"
        )
    _log("")

    # Capture stdout/stderr to get verbose SC and adaptive timestep logs
    # Use TeeStringIO to also print to console in real-time
    import sys

    stdout_capture = TeeStringIO(sys.stdout)
    stderr_capture = TeeStringIO(sys.stderr)

    # Build configuration objects using helper functions
    import copy

    from core.trajectory_integrator import retarded_integrator

    self_consistency_config = build_self_consistency_config(options)
    energy_monitor_config = build_energy_monitor_config(options)
    adaptive_timestep_config = build_adaptive_timestep_config(options)
    particle_loss_config = build_particle_loss_config(options)
    pseudo_grid_config = build_pseudo_grid_config(options)
    macroparticle_smearing_config = build_macroparticle_smearing_config(options)
    driver_train_config = build_driver_train_config(options)
    space_charge_config = build_space_charge_config(options)
    external_field_config = build_external_field_config(options)
    chrono_mode_enum = build_chrono_mode_enum(
        options.self_consistency_chrono_matching_mode
    )
    startup_mode_enum = build_startup_mode_enum(
        core_params.get("startup_mode", "COLD_START")
    )

    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        # Prepare initial states using the maintained bunch initialization path.
        rider_state, driver_state, rider_rest_mev, driver_rest_mev = (
            prepare_particle_bunches(
                seed=options.seed,
                rider_params=rider_params,
                driver_params=driver_params,
            )
        )

        # Apply macroparticle charge multiplier if enabled
        if options.macroparticle_enabled and sim_type == SimulationType.CONDUCTING_WALL:
            charge_mult = float(options.macroparticle_charge_multiplier)
            if charge_mult != 1.0:
                rider_state["q"] = rider_state["q"] * charge_mult
                if driver_state is not None:
                    driver_state["q"] = driver_state["q"] * charge_mult

        rider_initial = normalize_state(copy.deepcopy(rider_state))
        driver_initial = (
            normalize_state(copy.deepcopy(driver_state))
            if driver_state is not None
            else None
        )

        _actual_h_step = filtered_core_params.get("time_step", 2.2e-7)
        _actual_steps = options.steps
        if options.auto_duration_enabled and sim_type == SimulationType.BUNCH_TO_BUNCH:
            _rider_pz = float(np.asarray(rider_initial["Pz"]).mean())
            _rider_m = float(np.asarray(rider_initial["m"]).mean())
            _rider_gamma = float(np.asarray(rider_initial["gamma"]).mean())
            _rider_beta_z = abs(_rider_pz) / (_rider_gamma * _rider_m * C_MMNS)
            _driver_beta_z = 0.0
            if driver_initial is not None:
                _drv_pz = float(np.asarray(driver_initial["Pz"]).mean())
                _drv_m = float(np.asarray(driver_initial["m"]).mean())
                _drv_gamma = float(np.asarray(driver_initial["gamma"]).mean())
                _driver_beta_z = abs(_drv_pz) / (_drv_gamma * _drv_m * C_MMNS)
            _closing_speed = (_rider_beta_z + _driver_beta_z) * C_MMNS  # mm/ns
            _rider_z0 = float(np.asarray(rider_initial["z"]).mean())
            _driver_z0 = (
                float(np.asarray(driver_initial["z"]).mean())
                if driver_initial is not None
                else 0.0
            )
            _separation = abs(_driver_z0 - _rider_z0)
            if _closing_speed > 0 and _separation > 0:
                _actual_h_step = _separation / (
                    _closing_speed * options.auto_duration_crossing_steps
                )
                _actual_steps = max(
                    10,
                    int(
                        math.ceil(
                            options.auto_duration_crossing_steps
                            * options.auto_duration_post_factor
                        )
                    ),
                )
                _log(
                    f"  Auto-duration: sep={_separation:.4f} mm, "
                    f"closing={_closing_speed:.4f} mm/ns, "
                    f"h_step={_actual_h_step:.3e} ns, steps={_actual_steps}"
                )

        # Run core integrator directly
        core_traj_rider, core_traj_driver, *_soa_out = retarded_integrator(
            steps=_actual_steps,
            h_step=_actual_h_step,
            wall_z=filtered_core_params.get("wall_z", 1e5),
            aperture_radius=filtered_core_params.get("aperture_radius", 1e5),
            sim_type=sim_type,
            init_rider=copy.deepcopy(rider_initial),
            init_driver=copy.deepcopy(driver_initial),
            mean=filtered_core_params.get("mean", 1e5),
            cav_spacing=filtered_core_params.get("cav_spacing", 1e5),
            z_cutoff=filtered_core_params.get("z_cutoff", 0.0),
            z_cutoff_mode=filtered_core_params.get("z_cutoff_mode", "absolute"),
            self_consistency=self_consistency_config,
            chrono_mode=chrono_mode_enum,
            startup_mode=startup_mode_enum,
            energy_monitor=energy_monitor_config,
            adaptive_timestep=adaptive_timestep_config,
            space_charge=space_charge_config,
            external_field=external_field_config,
            image_subcharge_count=int(options.image_subcharge_count),
            use_conducting_image_weighting=bool(options.use_image_weighting),
            macroparticle_charge_multiplier=(
                float(options.macroparticle_charge_multiplier)
                if options.macroparticle_enabled
                else 1.0
            ),
            macroparticle_sigma_multiplier=(
                float(options.macroparticle_sigma_multiplier)
                if options.macroparticle_enabled
                else 1.0
            ),
            macroparticle_use_momentum_errors=(
                bool(options.macroparticle_use_momentum_errors)
                if options.macroparticle_enabled
                else True
            ),
            bunch_transv_dist=float(options.rider_params.get("transv_dist", 0.0)),
            bunch_transv_mom=float(options.rider_params.get("transv_mom", 0.0)),
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            logger=log,
            use_numba=getattr(options, "use_numba", True),
            radiation_reaction_mode=options.radiation_reaction_mode,
            pseudo_grid=pseudo_grid_config,
            driver_train=driver_train_config,
            particle_loss=particle_loss_config,
            macroparticle_smearing=macroparticle_smearing_config,
        )

        # Build a normalized payload shared by the GUI and CLI surfaces.
        payload = {
            "core": {
                "rider": [normalize_state(s) for s in core_traj_rider],
                "driver": [normalize_state(s) for s in core_traj_driver],
            },
            "initial_states": {
                "rider": rider_initial,
                "driver": driver_initial,
            },
            "rest_energy_mev": {
                "rider": rider_rest_mev,
                "driver": driver_rest_mev,
            },
        }

        result = ({}, payload)  # Empty metrics dict, payload with trajectories

    # Store captured stdout/stderr separately for verbose logs button
    captured_stdout = stdout_capture.getvalue()
    captured_stderr = stderr_capture.getvalue()

    # Log a summary
    stdout_lines = len([line for line in captured_stdout.splitlines() if line.strip()])
    stderr_lines = len([line for line in captured_stderr.splitlines() if line.strip()])

    if stdout_lines > 0:
        _log(
            f"Verbose output: {stdout_lines:,} lines (displayed in console and available via 'Load Verbose Logs')"
        )
    if stderr_lines > 0:
        _log(f"Stderr: {stderr_lines} lines")

    metrics: Optional[Dict[str, Dict[str, float]]] = None
    if isinstance(result, tuple) and len(result) == 2:
        _, payload = result
    else:
        payload = {}

    saved_paths: Dict[str, Path] = {}
    figures: Dict[str, plt.Figure] = {}

    core_traj = payload.get("core")
    initial_states = payload.get("initial_states", {})
    rest_energies = payload.get("rest_energy_mev", {})

    # Initialize values for RunResult
    rider_delta_e_final = None
    rider_gamma_initial = None
    rider_gamma_final = None
    rider_trajectory_data = None
    driver_gamma_initial = None
    driver_gamma_final = None
    driver_trajectory_data = None
    rider_emittance_x = None
    rider_emittance_y = None
    rider_norm_emittance_x = None
    rider_norm_emittance_y = None
    rider_beta_x = None
    rider_beta_y = None
    driver_emittance_x = None
    driver_emittance_y = None
    driver_norm_emittance_x = None
    driver_norm_emittance_y = None
    driver_beta_x = None
    driver_beta_y = None

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
        _log("[DEBUG] Initial state gamma calculation:")
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
                _log("[DEBUG] Initial beam optics:")
                _log(
                    f"  εx={rider_emittance_x:.3e} mm·mrad, εy={rider_emittance_y:.3e} mm·mrad"
                )
                _log(f"  βx={rider_beta_x:.3e} m, βy={rider_beta_y:.3e} m")
            except Exception as exc:
                _log(f"[WARNING] Failed to compute beam optics: {exc}")

    driver_initial = initial_states.get("driver") if driver_allowed else None
    if driver_initial:
        Pz_init = float(np.asarray(driver_initial.get("Pz", 0)).flat[0])
        Px_init = float(np.asarray(driver_initial.get("Px", 0)).flat[0])
        Py_init = float(np.asarray(driver_initial.get("Py", 0)).flat[0])
        P_init = np.sqrt(Pz_init**2 + Px_init**2 + Py_init**2)
        mass = float(np.asarray(driver_initial.get("m", 1)).flat[0])
        p_init = P_init / (mass * C_MMNS)
        driver_gamma_initial = float(np.sqrt(1 + p_init**2))

        driver_pcount = options.driver_params.get("pcount", 1)
        if driver_pcount > 1:
            try:
                beam_optics = compute_beam_optics(driver_initial, driver_gamma_initial)
                driver_emittance_x = beam_optics.get("emittance_x_mm_mrad")
                driver_emittance_y = beam_optics.get("emittance_y_mm_mrad")
                driver_norm_emittance_x = beam_optics.get("norm_emittance_x_mm_mrad")
                driver_norm_emittance_y = beam_optics.get("norm_emittance_y_mm_mrad")
                driver_beta_x = beam_optics.get("beta_x_m")
                driver_beta_y = beam_optics.get("beta_y_m")
            except Exception as exc:
                _log(f"[WARNING] Failed to compute driver beam optics: {exc}")

    if core_traj:
        rider_states = core_traj.get("rider", [])
        driver_states = core_traj.get("driver") if driver_allowed else None

        try:
            rider_initial = initial_states.get("rider")
            rider_rest_mev = rest_energies.get("rider")

            # Compute energy series with all components for plotting
            rider_delta_e_total, rider_delta_e_z, rider_z = (
                compute_delta_energy_components(
                    rider_states,
                    rider_initial,
                    rider_rest_mev,
                )
            )
            rider_delta_e = rider_delta_e_total  # For backward compatibility
            rider_z_rel = rider_z  # Use absolute z-positions for plotting

            # Compute transverse energy components (use alive particles only)
            rider_gamma_series = _mean_alive_gamma_series(rider_states)
            rider_z_rel = _alive_average_series(rider_states, "z", default=0.0)
            rider_bx_series = _alive_average_series(rider_states, "bx", default=0.0)
            rider_by_series = _alive_average_series(rider_states, "by", default=0.0)
            rider_initial_gamma = (
                compute_alive_particle_average(rider_initial, "gamma") or 1.0
            )
            rider_initial_bx = (
                compute_alive_particle_average(rider_initial, "bx") or 0.0
            )
            rider_initial_by = (
                compute_alive_particle_average(rider_initial, "by") or 0.0
            )
            rider_rest_gev = rider_rest_mev * 1e-3
            rider_delta_e_total = (
                rider_gamma_series - rider_initial_gamma
            ) * rider_rest_gev
            rider_delta_e = rider_delta_e_total
            rider_delta_e_final = (
                float(rider_delta_e[-1]) * 1e3 if len(rider_delta_e) > 0 else None
            )

            rider_delta_e_x = (
                rider_gamma_series * rider_bx_series
                - rider_initial_gamma * rider_initial_bx
            ) * rider_rest_gev
            rider_delta_e_y = (
                rider_gamma_series * rider_by_series
                - rider_initial_gamma * rider_initial_by
            ) * rider_rest_gev
            rider_e_total = rider_gamma_series * rider_rest_gev

            # Compute gamma values from trajectory states (for final state)
            if rider_states and len(rider_states) > 0:
                rider_gamma_initial = (
                    _mean_alive_gamma(rider_states[0]) or rider_gamma_initial
                )
                final_state = rider_states[-1]
                rider_gamma_final = _mean_alive_gamma(final_state)
                if rider_gamma_final is None:
                    rider_gamma_final = rider_gamma_initial
                    _log(
                        "[WARNING] All particles dead at final step - using initial gamma"
                    )

                # Store trajectory data (extract values from alive particles, averaged)
                # Compute r from x,y; compute normalized momentum components
                z_arr = np.array(
                    [
                        compute_alive_particle_average(s, "z") or 0.0
                        for s in rider_states
                    ]
                )
                x_arr = np.array(
                    [
                        compute_alive_particle_average(s, "x") or 0.0
                        for s in rider_states
                    ]
                )
                y_arr = np.array(
                    [
                        compute_alive_particle_average(s, "y") or 0.0
                        for s in rider_states
                    ]
                )
                r_arr = np.sqrt(x_arr**2 + y_arr**2)
                rider_initial_radial_summary = _compute_alive_particle_radial_summary(
                    rider_states[0]
                )
                rider_initial_rms = rider_initial_radial_summary.get(
                    "r_rms_particle", 0.0
                )
                rider_radial_summaries = [
                    _compute_alive_particle_radial_summary(
                        s,
                        initial_rms_radius_mm=rider_initial_rms,
                    )
                    for s in rider_states
                ]
                rider_longitudinal_summaries = [
                    _compute_alive_particle_longitudinal_summary(s)
                    for s in rider_states
                ]
                rider_momentum_summaries = [
                    _compute_alive_particle_momentum_summary(s) for s in rider_states
                ]
                r_mean_particle_arr = _summary_series(
                    rider_radial_summaries, "r_mean_particle"
                )
                r_rms_particle_arr = _summary_series(
                    rider_radial_summaries, "r_rms_particle"
                )

                # Extract momentum components (capital P) and normalize by m*c
                Pz_arr = np.array(
                    [
                        compute_alive_particle_average(s, "Pz") or 0.0
                        for s in rider_states
                    ]
                )
                Px_arr = np.array(
                    [
                        compute_alive_particle_average(s, "Px") or 0.0
                        for s in rider_states
                    ]
                )
                Py_arr = np.array(
                    [
                        compute_alive_particle_average(s, "Py") or 0.0
                        for s in rider_states
                    ]
                )
                m_arr = np.array(
                    [
                        compute_alive_particle_average(s, "m") or 1.0
                        for s in rider_states
                    ]
                )
                # Compute transverse momentum magnitude
                Pr_arr = np.sqrt(Px_arr**2 + Py_arr**2)
                gamma_arr = rider_gamma_series

                rider_trajectory_data = {
                    "z": z_arr,
                    "x": x_arr,
                    "y": y_arr,
                    "r": r_arr,
                    "r_mean_particle": r_mean_particle_arr,
                    "r_rms_particle": r_rms_particle_arr,
                    "r_p50_particle": _summary_series(
                        rider_radial_summaries, "r_p50_particle"
                    ),
                    "r_p68_particle": _summary_series(
                        rider_radial_summaries, "r_p68_particle"
                    ),
                    "r_p90_particle": _summary_series(
                        rider_radial_summaries, "r_p90_particle"
                    ),
                    "r_p95_particle": _summary_series(
                        rider_radial_summaries, "r_p95_particle"
                    ),
                    "r_p99_particle": _summary_series(
                        rider_radial_summaries, "r_p99_particle"
                    ),
                    "halo_gt_2_initial_rms_fraction": _summary_series(
                        rider_radial_summaries,
                        "halo_gt_2_initial_rms_fraction",
                    ),
                    "halo_gt_3_initial_rms_fraction": _summary_series(
                        rider_radial_summaries,
                        "halo_gt_3_initial_rms_fraction",
                    ),
                    "halo_gt_5_initial_rms_fraction": _summary_series(
                        rider_radial_summaries,
                        "halo_gt_5_initial_rms_fraction",
                    ),
                    "alive_fraction": _summary_series(
                        rider_radial_summaries, "alive_fraction"
                    ),
                    "z_p01_particle": _summary_series(
                        rider_longitudinal_summaries, "z_p01_particle"
                    ),
                    "z_p05_particle": _summary_series(
                        rider_longitudinal_summaries, "z_p05_particle"
                    ),
                    "z_p50_particle": _summary_series(
                        rider_longitudinal_summaries, "z_p50_particle"
                    ),
                    "z_p95_particle": _summary_series(
                        rider_longitudinal_summaries, "z_p95_particle"
                    ),
                    "z_p99_particle": _summary_series(
                        rider_longitudinal_summaries, "z_p99_particle"
                    ),
                    "z_width_p90_particle": _summary_series(
                        rider_longitudinal_summaries, "z_width_p90_particle"
                    ),
                    "z_width_p98_particle": _summary_series(
                        rider_longitudinal_summaries, "z_width_p98_particle"
                    ),
                    "z_std_particle": _summary_series(
                        rider_longitudinal_summaries, "z_std_particle"
                    ),
                    "gamma_std_particle": _summary_series(
                        rider_momentum_summaries, "gamma_std_particle"
                    ),
                    "pz_std_particle": _summary_series(
                        rider_momentum_summaries, "pz_std_particle"
                    ),
                    "pz": Pz_arr / (m_arr * C_MMNS),
                    "pr": Pr_arr / (m_arr * C_MMNS),
                    "gamma": gamma_arr,
                    "t": np.array(
                        [
                            compute_alive_particle_average(s, "t") or 0.0
                            for s in rider_states
                        ]
                    ),
                }

                # Check for early halt metadata and particle failures in the last trajectory state
                halted_early = False
                halt_reason = None
                num_particles_dead = 0
                particle_failure_info = None
                if len(rider_states) > 0:
                    last_state = rider_states[-1]
                    if "_halted_early" in last_state:
                        halted_early = bool(last_state["_halted_early"])
                    if "_halt_reason" in last_state:
                        halt_reason = str(last_state["_halt_reason"])
                        _log(f"[INFO] Integration halted early: {halt_reason}")

                    # Log particle failure summary if any particles failed
                    failure_info = get_particle_failure_summary(rider_states)
                    if failure_info:
                        num_particles_dead = len(failure_info)
                        particle_failure_info = failure_info
                        failure_summary = format_failure_summary(failure_info)
                        _log(f"[INFO] {failure_summary}")

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

                # Compute energy series with all components for plotting
                driver_delta_e_total, driver_delta_e_z, driver_z = (
                    compute_delta_energy_components(
                        driver_states,
                        driver_initial,
                        driver_rest_mev,
                    )
                )
                driver_delta_e = driver_delta_e_total  # For backward compatibility
                driver_z_rel = driver_z  # Use absolute z-positions for plotting

                # Compute transverse energy components
                driver_gamma_series = _mean_alive_gamma_series(driver_states)
                driver_z_rel = _alive_average_series(driver_states, "z", default=0.0)
                driver_bx_series = _alive_average_series(
                    driver_states, "bx", default=0.0
                )
                driver_by_series = _alive_average_series(
                    driver_states, "by", default=0.0
                )
                driver_initial_gamma = (
                    compute_alive_particle_average(driver_initial, "gamma") or 1.0
                )
                driver_initial_bx = (
                    compute_alive_particle_average(driver_initial, "bx") or 0.0
                )
                driver_initial_by = (
                    compute_alive_particle_average(driver_initial, "by") or 0.0
                )
                driver_rest_gev = driver_rest_mev * 1e-3

                driver_delta_e_x = (
                    driver_gamma_series * driver_bx_series
                    - driver_initial_gamma * driver_initial_bx
                ) * driver_rest_gev
                driver_delta_e_y = (
                    driver_gamma_series * driver_by_series
                    - driver_initial_gamma * driver_initial_by
                ) * driver_rest_gev
                driver_e_total = driver_gamma_series * driver_rest_gev

                if driver_states and len(driver_states) > 0:
                    driver_gamma_initial = (
                        _mean_alive_gamma(driver_states[0]) or driver_initial_gamma
                    )
                    final_state = driver_states[-1]
                    driver_gamma_final = _mean_alive_gamma(final_state)
                    if driver_gamma_final is None:
                        driver_gamma_final = driver_gamma_initial

                    z_arr = np.array(
                        [
                            compute_alive_particle_average(s, "z") or 0.0
                            for s in driver_states
                        ]
                    )
                    x_arr = np.array(
                        [
                            compute_alive_particle_average(s, "x") or 0.0
                            for s in driver_states
                        ]
                    )
                    y_arr = np.array(
                        [
                            compute_alive_particle_average(s, "y") or 0.0
                            for s in driver_states
                        ]
                    )
                    r_arr = np.sqrt(x_arr**2 + y_arr**2)
                    driver_initial_radial_summary = (
                        _compute_alive_particle_radial_summary(driver_states[0])
                    )
                    driver_initial_rms = driver_initial_radial_summary.get(
                        "r_rms_particle", 0.0
                    )
                    driver_radial_summaries = [
                        _compute_alive_particle_radial_summary(
                            s,
                            initial_rms_radius_mm=driver_initial_rms,
                        )
                        for s in driver_states
                    ]
                    driver_longitudinal_summaries = [
                        _compute_alive_particle_longitudinal_summary(s)
                        for s in driver_states
                    ]
                    driver_momentum_summaries = [
                        _compute_alive_particle_momentum_summary(s)
                        for s in driver_states
                    ]
                    r_mean_particle_arr = _summary_series(
                        driver_radial_summaries, "r_mean_particle"
                    )
                    r_rms_particle_arr = _summary_series(
                        driver_radial_summaries, "r_rms_particle"
                    )
                    Pz_arr = np.array(
                        [
                            compute_alive_particle_average(s, "Pz") or 0.0
                            for s in driver_states
                        ]
                    )
                    Px_arr = np.array(
                        [
                            compute_alive_particle_average(s, "Px") or 0.0
                            for s in driver_states
                        ]
                    )
                    Py_arr = np.array(
                        [
                            compute_alive_particle_average(s, "Py") or 0.0
                            for s in driver_states
                        ]
                    )
                    m_arr = np.array(
                        [
                            compute_alive_particle_average(s, "m") or 1.0
                            for s in driver_states
                        ]
                    )
                    Pr_arr = np.sqrt(Px_arr**2 + Py_arr**2)
                    gamma_arr = driver_gamma_series

                    driver_trajectory_data = {
                        "z": z_arr,
                        "x": x_arr,
                        "y": y_arr,
                        "r": r_arr,
                        "r_mean_particle": r_mean_particle_arr,
                        "r_rms_particle": r_rms_particle_arr,
                        "r_p50_particle": _summary_series(
                            driver_radial_summaries, "r_p50_particle"
                        ),
                        "r_p68_particle": _summary_series(
                            driver_radial_summaries, "r_p68_particle"
                        ),
                        "r_p90_particle": _summary_series(
                            driver_radial_summaries, "r_p90_particle"
                        ),
                        "r_p95_particle": _summary_series(
                            driver_radial_summaries, "r_p95_particle"
                        ),
                        "r_p99_particle": _summary_series(
                            driver_radial_summaries, "r_p99_particle"
                        ),
                        "halo_gt_2_initial_rms_fraction": _summary_series(
                            driver_radial_summaries,
                            "halo_gt_2_initial_rms_fraction",
                        ),
                        "halo_gt_3_initial_rms_fraction": _summary_series(
                            driver_radial_summaries,
                            "halo_gt_3_initial_rms_fraction",
                        ),
                        "halo_gt_5_initial_rms_fraction": _summary_series(
                            driver_radial_summaries,
                            "halo_gt_5_initial_rms_fraction",
                        ),
                        "alive_fraction": _summary_series(
                            driver_radial_summaries, "alive_fraction"
                        ),
                        "z_p01_particle": _summary_series(
                            driver_longitudinal_summaries, "z_p01_particle"
                        ),
                        "z_p05_particle": _summary_series(
                            driver_longitudinal_summaries, "z_p05_particle"
                        ),
                        "z_p50_particle": _summary_series(
                            driver_longitudinal_summaries, "z_p50_particle"
                        ),
                        "z_p95_particle": _summary_series(
                            driver_longitudinal_summaries, "z_p95_particle"
                        ),
                        "z_p99_particle": _summary_series(
                            driver_longitudinal_summaries, "z_p99_particle"
                        ),
                        "z_width_p90_particle": _summary_series(
                            driver_longitudinal_summaries, "z_width_p90_particle"
                        ),
                        "z_width_p98_particle": _summary_series(
                            driver_longitudinal_summaries, "z_width_p98_particle"
                        ),
                        "z_std_particle": _summary_series(
                            driver_longitudinal_summaries, "z_std_particle"
                        ),
                        "gamma_std_particle": _summary_series(
                            driver_momentum_summaries, "gamma_std_particle"
                        ),
                        "pz_std_particle": _summary_series(
                            driver_momentum_summaries, "pz_std_particle"
                        ),
                        "pz": Pz_arr / (m_arr * C_MMNS),
                        "pr": Pr_arr / (m_arr * C_MMNS),
                        "gamma": gamma_arr,
                        "t": np.array(
                            [
                                compute_alive_particle_average(s, "t") or 0.0
                                for s in driver_states
                            ]
                        ),
                    }
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

                # Plot total ΔE
                axes[0].scatter(
                    rider_z_rel[valid_mask],
                    rider_delta_e[valid_mask],
                    color=COLOR_RIDER,
                    label="Rider",
                    **SCATTER_STYLE,
                )
                axes[0].set_xlabel("z position (mm)")
                axes[0].set_ylabel("ΔE (GeV)")
                axes[0].set_title("Rider ΔE vs z", pad=10)
                axes[0].grid(True, alpha=0.3)
                axes[0].tick_params(axis="both", which="major", labelsize=10)
                axes[0].tick_params(axis="both", which="minor", labelsize=8)
                # Fix log scale tick label formatting
                from matplotlib.ticker import ScalarFormatter

                axes[0].yaxis.set_major_formatter(ScalarFormatter())
                axes[0].yaxis.get_major_formatter().set_scientific(False)
                axes[0].yaxis.get_major_formatter().set_useOffset(False)
                axes[0].legend()

                driver_valid = np.array([], dtype=bool)
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
                            label="Driver",
                            **SCATTER_STYLE,
                        )
                    else:
                        _log("Warning: All driver energy data points are invalid")
                    axes[1].set_xlabel("z position (mm)")
                    axes[1].set_ylabel("ΔE (GeV)")
                    axes[1].set_title("Driver ΔE vs z", pad=10)
                    axes[1].grid(True, alpha=0.3)
                    axes[1].tick_params(axis="both", which="major", labelsize=10)
                    axes[1].tick_params(axis="both", which="minor", labelsize=8)
                    # Fix log scale tick label formatting
                    from matplotlib.ticker import ScalarFormatter

                    axes[1].yaxis.set_major_formatter(ScalarFormatter())
                    axes[1].yaxis.get_major_formatter().set_scientific(False)
                    axes[1].yaxis.get_major_formatter().set_useOffset(False)
                    axes[1].legend()

                # Attach metadata for interactive replotting
                # Extract time data from rider states
                rider_times = np.array(
                    [float(np.asarray(s.get("t", 0)).flat[0]) for s in rider_states]
                )

                fig_energy._lw_plot_data = {
                    "plot_type": "energy",
                    "times_ns": (
                        rider_times[valid_mask] if np.any(valid_mask) else np.array([])
                    ),
                    "z_mm": (
                        rider_z_rel[valid_mask] if np.any(valid_mask) else np.array([])
                    ),
                    "z_mm_driver": (
                        driver_z_rel[driver_valid]
                        if driver_delta_e is not None and np.any(driver_valid)
                        else None
                    ),
                    "core_r_energy_changes": (
                        rider_delta_e[valid_mask]
                        if np.any(valid_mask)
                        else np.array([])
                    ),
                    "core_d_energy_changes": (
                        driver_delta_e[driver_valid]
                        if driver_delta_e is not None and np.any(driver_valid)
                        else None
                    ),
                    "driver_allowed": driver_allowed,
                    # Energy components for Y-axis switching
                    "energy_components": {
                        "delta_total_r": (
                            rider_delta_e_total[valid_mask]
                            if np.any(valid_mask)
                            else np.array([])
                        ),
                        "delta_z_r": (
                            rider_delta_e_z[valid_mask]
                            if np.any(valid_mask)
                            else np.array([])
                        ),
                        "delta_x_r": (
                            rider_delta_e_x[valid_mask]
                            if np.any(valid_mask)
                            else np.array([])
                        ),
                        "delta_y_r": (
                            rider_delta_e_y[valid_mask]
                            if np.any(valid_mask)
                            else np.array([])
                        ),
                        "total_r": (
                            rider_e_total[valid_mask]
                            if np.any(valid_mask)
                            else np.array([])
                        ),
                        "delta_total_d": (
                            driver_delta_e_total[driver_valid]
                            if driver_delta_e is not None and np.any(driver_valid)
                            else None
                        ),
                        "delta_z_d": (
                            driver_delta_e_z[driver_valid]
                            if driver_delta_e is not None and np.any(driver_valid)
                            else None
                        ),
                        "delta_x_d": (
                            driver_delta_e_x[driver_valid]
                            if driver_delta_e is not None and np.any(driver_valid)
                            else None
                        ),
                        "delta_y_d": (
                            driver_delta_e_y[driver_valid]
                            if driver_delta_e is not None and np.any(driver_valid)
                            else None
                        ),
                        "total_d": (
                            driver_e_total[driver_valid]
                            if driver_delta_e is not None and np.any(driver_valid)
                            else None
                        ),
                    },
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

        core_r_hist = np.column_stack(
            (
                _extract_scalar_series(rider_states, "t"),
                _extract_scalar_series(rider_states, "x"),
                _extract_scalar_series(rider_states, "y"),
                _extract_scalar_series(rider_states, "z"),
            )
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
            core_d_hist = np.column_stack(
                (
                    _extract_scalar_series(driver_states, "t"),
                    _extract_scalar_series(driver_states, "x"),
                    _extract_scalar_series(driver_states, "y"),
                    _extract_scalar_series(driver_states, "z"),
                )
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
                "z_mm_driver": (
                    core_d_hist[:, 3]
                    if driver_allowed and core_d_hist is not None
                    else None
                ),
                "core_r_hist": core_r_hist,
                "core_d_hist": core_d_hist if driver_allowed else None,
                "driver_allowed": driver_allowed,
            }

            # Determine x-axis data
            if transverse_xaxis == "z":
                xdata = plot_z_mm
                xlabel = "z position (mm)"
            else:
                xdata = plot_times_ns
                xlabel = "Time (ns)"

            ax_x.scatter(
                xdata,
                core_r_hist[:, 1] * 1e3,
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            ax_y.scatter(
                xdata,
                core_r_hist[:, 2] * 1e3,
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            if driver_allowed and core_d_hist is not None:
                if transverse_xaxis == "z":
                    xdata_d = core_d_hist[:, 3]
                else:
                    xdata_d = plot_times_ns
                ax_x.scatter(
                    xdata_d,
                    core_d_hist[:, 1] * 1e3,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
                )
                ax_y.scatter(
                    xdata_d,
                    core_d_hist[:, 2] * 1e3,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
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
                "z_mm_driver": (
                    core_d_hist[:, 3]
                    if driver_allowed and core_d_hist is not None
                    else None
                ),
                "core_r_beta": core_r_beta,
                "core_d_beta": core_d_beta if driver_allowed else None,
                "driver_allowed": driver_allowed,
            }

            # Determine x-axis data for beta plots
            if beta_xaxis == "z":
                xdata_beta = plot_z_mm
                xlabel_beta = "z position (mm)"
            else:
                xdata_beta = plot_times_ns
                xlabel_beta = "Time (ns)"

            # β_x
            axes_beta[0].scatter(
                xdata_beta,
                core_r_beta[:, 0],
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            if driver_allowed and core_d_beta is not None:
                if beta_xaxis == "z":
                    xdata_beta_d = (
                        core_d_hist[:, 3] if core_d_hist is not None else plot_z_mm
                    )
                else:
                    xdata_beta_d = plot_times_ns
                axes_beta[0].scatter(
                    xdata_beta_d,
                    core_d_beta[:, 0],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
                )
            axes_beta[0].set_xlabel(xlabel_beta)
            axes_beta[0].set_ylabel("β⟨x⟩")
            axes_beta[0].set_title("Beta X Component", pad=10)
            axes_beta[0].legend()
            axes_beta[0].grid(True, alpha=0.3)

            # β_y
            axes_beta[1].scatter(
                xdata_beta,
                core_r_beta[:, 1],
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            if driver_allowed and core_d_beta is not None:
                axes_beta[1].scatter(
                    xdata_beta_d,
                    core_d_beta[:, 1],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
                )
            axes_beta[1].set_xlabel(xlabel_beta)
            axes_beta[1].set_ylabel("β⟨y⟩")
            axes_beta[1].set_title("Beta Y Component", pad=10)
            axes_beta[1].legend()
            axes_beta[1].grid(True, alpha=0.3)

            # β_z
            axes_beta[2].scatter(
                xdata_beta,
                core_r_beta[:, 2],
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            if driver_allowed and core_d_beta is not None:
                axes_beta[2].scatter(
                    xdata_beta_d,
                    core_d_beta[:, 2],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
                )
            axes_beta[2].set_xlabel(xlabel_beta)
            axes_beta[2].set_ylabel("β⟨z⟩")
            axes_beta[2].set_title("Beta Z Component", pad=10)
            axes_beta[2].legend()
            axes_beta[2].grid(True, alpha=0.3)

            # |β| (magnitude)
            core_beta_mag = np.sqrt(np.sum(core_r_beta**2, axis=1))
            axes_beta[3].scatter(
                xdata_beta,
                core_beta_mag,
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            if driver_allowed and core_d_beta is not None:
                driver_beta_mag = np.sqrt(np.sum(core_d_beta**2, axis=1))
                axes_beta[3].scatter(
                    xdata_beta_d,
                    driver_beta_mag,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
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
                "z_mm_driver": (
                    core_d_hist[:, 3]
                    if driver_allowed and core_d_hist is not None
                    else None
                ),
                "core_r_momentum": core_r_momentum,
                "core_r_pt": core_r_pt,
                "core_d_momentum": core_d_momentum if driver_allowed else None,
                "core_d_pt": core_d_pt if driver_allowed else None,
                "driver_allowed": driver_allowed,
            }

            # Determine x-axis data for momentum plots
            if momentum_xaxis == "z":
                xdata_mom = plot_z_mm
                xlabel_mom = "z position (mm)"
            else:
                xdata_mom = plot_times_ns
                xlabel_mom = "Time (ns)"

            # P_x (conjugate momentum)
            axes_mom[0].scatter(
                xdata_mom,
                core_r_momentum[:, 0],
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            if driver_allowed and core_d_momentum is not None:
                if momentum_xaxis == "z":
                    xdata_mom_d = (
                        core_d_hist[:, 3] if core_d_hist is not None else plot_z_mm
                    )
                else:
                    xdata_mom_d = plot_times_ns
                axes_mom[0].scatter(
                    xdata_mom_d,
                    core_d_momentum[:, 0],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
                )
            axes_mom[0].set_xlabel(xlabel_mom)
            axes_mom[0].set_ylabel("Pˣ (amu·mm/ns)")
            axes_mom[0].set_title("Conjugate Momentum Pˣ", pad=10)
            axes_mom[0].legend()
            axes_mom[0].grid(True, alpha=0.3)

            # P_y
            axes_mom[1].scatter(
                xdata_mom,
                core_r_momentum[:, 1],
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            if driver_allowed and core_d_momentum is not None:
                axes_mom[1].scatter(
                    xdata_mom_d,
                    core_d_momentum[:, 1],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
                )
            axes_mom[1].set_xlabel(xlabel_mom)
            axes_mom[1].set_ylabel("Pʸ (amu·mm/ns)")
            axes_mom[1].set_title("Conjugate Momentum Pʸ", pad=10)
            axes_mom[1].legend()
            axes_mom[1].grid(True, alpha=0.3)

            # P_z
            axes_mom[2].scatter(
                xdata_mom,
                core_r_momentum[:, 2],
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            if driver_allowed and core_d_momentum is not None:
                axes_mom[2].scatter(
                    xdata_mom_d,
                    core_d_momentum[:, 2],
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
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
            axes_mom[3].scatter(
                xdata_mom,
                core_pt_mag,
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            if driver_allowed and core_d_momentum is not None:
                driver_pt_mag = np.sqrt(
                    core_d_momentum[:, 0] ** 2 + core_d_momentum[:, 1] ** 2
                )
                axes_mom[3].scatter(
                    xdata_mom_d,
                    driver_pt_mag,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
                )
            axes_mom[3].set_xlabel(xlabel_mom)
            axes_mom[3].set_ylabel("|P⊥| (amu·mm/ns)")
            axes_mom[3].set_title("Transverse Momentum |P⊥|", pad=10)
            axes_mom[3].legend()
            axes_mom[3].grid(True, alpha=0.3)

            # P_t (temporal/energy component)
            axes_mom[4].scatter(
                xdata_mom,
                core_r_pt,
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            if driver_allowed and core_d_pt is not None:
                axes_mom[4].scatter(
                    xdata_mom_d,
                    core_d_pt,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
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
            axes_mom[5].scatter(
                xdata_mom,
                core_p_mag,
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )
            if driver_allowed and core_d_momentum is not None:
                driver_p_mag = np.sqrt(
                    core_d_momentum[:, 0] ** 2
                    + core_d_momentum[:, 1] ** 2
                    + core_d_momentum[:, 2] ** 2
                )
                axes_mom[5].scatter(
                    xdata_mom_d,
                    driver_p_mag,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
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

        # Gamma (Lorentz factor) plot
        gamma_xaxis = getattr(options, "gamma_xaxis", "t")
        if (gamma_display or gamma_save) and core_r_hist is not None:
            _log(f"Generating gamma plot (display={gamma_display}, save={gamma_save})")

            # Extract gamma from states
            core_r_gamma = np.array([float(s["gamma"][0]) for s in rider_states])

            core_d_gamma = None
            if driver_allowed and driver_states is not None:
                core_d_gamma = np.array([float(s["gamma"][0]) for s in driver_states])

            fig_gamma, axes_gamma = plt.subplots(
                1,
                2 if driver_allowed else 1,
                figsize=(16 if driver_allowed else 8, 6),
                dpi=options.plot_dpi,
            )
            if not isinstance(axes_gamma, np.ndarray):
                axes_gamma = [axes_gamma]
            else:
                axes_gamma = list(axes_gamma)

            # Determine x-axis data
            if gamma_xaxis == "z":
                xdata_gamma = plot_z_mm
                xlabel_gamma = "z position (mm)"
            else:
                xdata_gamma = plot_times_ns
                xlabel_gamma = "Time (ns)"

            # Rider gamma
            axes_gamma[0].scatter(
                xdata_gamma,
                core_r_gamma,
                color=COLOR_RIDER,
                label="Rider (Core)",
                **SCATTER_STYLE,
            )

            if driver_allowed and core_d_gamma is not None:
                if gamma_xaxis == "z":
                    xdata_gamma_d = (
                        core_d_hist[:, 3] if core_d_hist is not None else plot_z_mm
                    )
                else:
                    xdata_gamma_d = plot_times_ns
                axes_gamma[0].scatter(
                    xdata_gamma_d,
                    core_d_gamma,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                    **SCATTER_STYLE,
                )

            axes_gamma[0].set_xlabel(xlabel_gamma)
            axes_gamma[0].set_ylabel("γ (Lorentz factor)")
            axes_gamma[0].set_title("Rider Lorentz Factor γ", pad=10)
            axes_gamma[0].legend()
            axes_gamma[0].grid(True, alpha=0.3)

            # Driver gamma (if present)
            if driver_allowed and len(axes_gamma) > 1:
                if core_d_gamma is not None:
                    axes_gamma[1].scatter(
                        xdata_gamma_d,
                        core_d_gamma,
                        color=COLOR_DRIVER,
                        label="Core",
                        **SCATTER_STYLE,
                    )

                axes_gamma[1].set_xlabel(xlabel_gamma)
                axes_gamma[1].set_ylabel("γ (Lorentz factor)")
                axes_gamma[1].set_title("Driver Lorentz Factor γ", pad=10)
                axes_gamma[1].legend()
                axes_gamma[1].grid(True, alpha=0.3)

            # Apply intelligent y-axis scaling for gamma to show small fluctuations
            for i, ax in enumerate(axes_gamma):
                try:
                    # Collect all gamma values for this subplot
                    all_gamma = []
                    if i == 0:  # Rider axis
                        if len(core_r_gamma) > 0:
                            all_gamma.extend(core_r_gamma)
                        if (
                            driver_allowed
                            and core_d_gamma is not None
                            and len(core_d_gamma) > 0
                        ):
                            all_gamma.extend(core_d_gamma)
                    elif i == 1 and driver_allowed:  # Driver axis
                        if core_d_gamma is not None and len(core_d_gamma) > 0:
                            all_gamma.extend(core_d_gamma)

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
                            _log(
                                f"Applied y-axis scaling for gamma subplot {i + 1} (Δγ/γ = {relative_variation * 100:.3f}%)"
                            )
                except Exception:
                    # Silently ignore errors in y-axis scaling
                    pass

            # Attach metadata for interactive replotting
            fig_gamma._lw_plot_data = {
                "plot_type": "gamma",
                "times_ns": plot_times_ns,
                "z_mm": plot_z_mm,
                "z_mm_driver": (
                    core_d_hist[:, 3]
                    if driver_allowed and core_d_hist is not None
                    else None
                ),
                "core_r_gamma": core_r_gamma,
                "core_d_gamma": core_d_gamma,
                "driver_allowed": driver_allowed,
            }

            fig_gamma.tight_layout(pad=2.5, w_pad=3.0, h_pad=2.5)
            if gamma_save and should_save:
                gamma_path = output_dir / f"{filename_base}_gamma.png"
                fig_gamma.savefig(gamma_path)
                saved_paths["gamma"] = gamma_path
                _log(f"Saved gamma plot to: {gamma_path}")
            if gamma_display:
                figures["gamma"] = fig_gamma
            else:
                plt.close(fig_gamma)

        # Z-position vs time plot
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

            label_prefix = config_label if config_label else "trajectory"

            # Save as JSON for the maintained single-run plotting toolchain.
            traj_path_json = (
                output_dir / f"{label_prefix}_trajectory_data_{timestamp_token}.json"
            )
            with traj_path_json.open("w", encoding="utf-8") as handle:
                json.dump(traj_data, handle, indent=2)
            saved_paths["trajectory_json"] = traj_path_json
            _log(f"Saved trajectory JSON to: {traj_path_json} (interval={interval})")

            # Also save as NPZ (standard format matching sweep/optimization)
            # This allows trajectory files to be loaded by the optimization plugin viewer
            traj_path_npz = (
                output_dir / f"{label_prefix}_trajectory_data_{timestamp_token}.npz"
            )

            # Extract core rider trajectory data in NPZ format
            rider_data = core_payload.get("rider", {})
            positions = rider_data.get("positions_mm", {})
            momenta = rider_data.get("conjugate_momenta", {})

            # Calculate r and pr from x, y components
            x_arr = np.array(positions.get("x", []))
            y_arr = np.array(positions.get("y", []))
            z_arr = np.array(positions.get("z", []))
            px_arr = np.array(momenta.get("Px", []))
            py_arr = np.array(momenta.get("Py", []))
            pz_arr = np.array(momenta.get("Pz", []))
            gamma_arr = np.array(rider_data.get("gamma_hist", []))
            t_arr = np.array(rider_data.get("time_ns", []))

            r_arr = np.sqrt(x_arr**2 + y_arr**2)
            pr_arr = np.sqrt(px_arr**2 + py_arr**2)

            # Save NPZ with standard format (z, r, pz, pr, t, gamma)
            np.savez(
                traj_path_npz,
                z=z_arr,
                r=r_arr,
                pz=pz_arr,
                pr=pr_arr,
                t=t_arr,
                gamma=gamma_arr,
            )
            saved_paths["trajectory_npz"] = traj_path_npz
            _log(f"Saved trajectory NPZ to: {traj_path_npz} (interval={interval})")

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

        # Copy the exact debug log session used by this run.
        if debug_log_path is not None and debug_log_path.exists():
            import shutil

            copied_debug_log_path = output_dir / debug_log_path.name
            try:
                shutil.copy2(debug_log_path, copied_debug_log_path)
                saved_paths["debug_log"] = copied_debug_log_path
                _log(f"Copied debug log to: {copied_debug_log_path}")
            except Exception as exc:
                _log(f"Failed to copy debug log: {exc}")

    return RunResult(
        metrics=metrics,
        saved_paths=saved_paths,
        figures=figures,
        logs=logs,
        verbose_logs=captured_stdout,  # Include captured verbose output
        duration_s=duration,
        filename_base=filename_base,
        debug_log_path=debug_log_path,
        rider_delta_e=rider_delta_e_final,
        rider_gamma_initial=rider_gamma_initial,
        rider_gamma_final=rider_gamma_final,
        rider_trajectory=rider_trajectory_data,
        driver_gamma_initial=driver_gamma_initial,
        driver_gamma_final=driver_gamma_final,
        driver_trajectory=driver_trajectory_data,
        rider_emittance_x_mm_mrad=rider_emittance_x,
        rider_emittance_y_mm_mrad=rider_emittance_y,
        rider_norm_emittance_x_mm_mrad=rider_norm_emittance_x,
        rider_norm_emittance_y_mm_mrad=rider_norm_emittance_y,
        rider_beta_x_m=rider_beta_x,
        rider_beta_y_m=rider_beta_y,
        driver_emittance_x_mm_mrad=driver_emittance_x,
        driver_emittance_y_mm_mrad=driver_emittance_y,
        driver_norm_emittance_x_mm_mrad=driver_norm_emittance_x,
        driver_norm_emittance_y_mm_mrad=driver_norm_emittance_y,
        driver_beta_x_m=driver_beta_x,
        driver_beta_y_m=driver_beta_y,
        halted_early=halted_early if "halted_early" in locals() else False,
        halt_reason=halt_reason if "halt_reason" in locals() else None,
        num_particles_dead=(
            num_particles_dead if "num_particles_dead" in locals() else 0
        ),
        particle_failure_info=(
            particle_failure_info if "particle_failure_info" in locals() else None
        ),
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
    "RADIATION_REACTION_MODE_CHOICES",
    "SimulationOptions",
    "InitialSummary",
    "RunResult",
    "SPECIES_OPTIONS",
    "SPECIES_PRESETS",
    "apply_species_preset",
    "build_external_field_config",
    "build_pseudo_grid_config",
    "build_macroparticle_smearing_config",
    "build_driver_train_config",
    "compute_initial_summary",
    "ensure_directory",
    "generate_filename_base",
    "list_config_files",
    "load_config",
    "run_testbed",
    "save_config",
    "supports_driver",
]
