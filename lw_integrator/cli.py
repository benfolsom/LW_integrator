"""Command-line interface for running LW Integrator simulations.

The ``lw-simulate`` console script and ``python -m lw_integrator`` entry point
both call :func:`main`.  Users can either rely on the built-in default scenario
(a 35 MeV electron approaching a conducting aperture) or provide a JSON
configuration that customises the simulation parameters and particle bunches.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from core.constants import C_MMNS, ELECTRON_MASS_AMU
from core.integration_runner import AdaptiveTimestepConfig, retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.types import (
    BeamlineGeometryConfig,
    ChronoMatchingMode,
    CavityExitConfig,
    DriverTrainConfig,
    ExternalFieldConfig,
    GammaReconciliationMethod,
    IntegratorConfig,
    MacroparticleSmearingConfig,
    Occluder,
    ParticleLossConfig,
    ParticleState,
    PseudoGridConfig,
    SimulationType,
    SpaceChargeConfig,
    StartupMode,
    Trajectory,
)
from input_output.bunch_initialization import create_bunch_from_energy
from optimization.plugin_results_helpers import (
    parse_results_payload,
    summarize_result_row,
    summarize_saved_results,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SIMULATION: Dict[str, Any] = {
    "steps": 1200,
    "time_step": 1e-3,
    "simulation_type": "conducting-wall",
    "wall_position": 0.0,
    "aperture_radius": 5e-4,
    "bunch_mean": 1000.0,
    "cavity_spacing": 0.0,
    "z_cutoff": 0.0,
    "z_cutoff_mode": "absolute",
    "chrono_mode": "fast",
    "startup_mode": "cold-start",
    "image_subcharge_count": 12,
    "use_image_weighting": True,
    "radiation_reaction_mode": "medina_lad",
}

RADIATION_REACTION_MODE_CHOICES: Tuple[str, ...] = (
    "off",
    "diagnostic_only",
    "power_matched_damping",
    "medina_lad",
)

DEFAULT_RIDER: Dict[str, Any] = {
    "kinetic_energy_mev": 35.0,
    "mass_amu": ELECTRON_MASS_AMU,
    "charge_sign": -1.0,
    "position_z": -300.0,
    "particle_count": 1,
    "transverse_radius": 0.0,
    "transverse_momentum": 0.0,
    "transverse_spread": 0.0,
    "transverse_geometry": "square",
    "longitudinal_spread": 0.0,
}

DEFAULT_PARTICLE_LOSS: Dict[str, Any] = {
    "enabled": True,
    "loss_radius_mm": 500.0,
    "conducting_wall_aperture_loss_enabled": True,
    "initial_radial_quantile": None,
    "initial_radial_multiplier": 1.0,
    "initial_radial_margin_mm": 0.0,
}

DEFAULT_PSEUDO_GRID: Dict[str, Any] = {
    "enabled": False,
    "active_rider_count": 4,
    "active_driver_count": 4,
    "passive_neighbor_count": 4,
    "coverage_strategy": "farthest_point_staleness",
    "coverage_space": "position",
    "pair_reuse_window": 16,
    "source_weighting_mode": "inverse_distance",
    "loss_tracking_enabled": True,
    "causal_history_pruning_enabled": False,
    "causal_history_safety_margin_steps": 2,
}

DEFAULT_CAVITY_EXIT: Dict[str, Any] = {
    "enabled": False,
    "mode": "first_exit",
    "cavity_length_mm": None,
    "residual_tail_factor": 0.0,
    "max_residual_tail_steps": 0,
}

DEFAULT_BEAMLINE_GEOMETRY: Dict[str, Any] = {
    "enabled": False,
    "occluders": [],
}

DEFAULT_MACROPARTICLE_SMEARING: Dict[str, Any] = {
    "enabled": False,
    "mode": "deterministic_subcharge",
    "subcharge_count": 8,
    "sigma_multiplier": 1.0,
    "position_sigma_mm": None,
    "longitudinal_sigma_mm": None,
    "momentum_sigma_amu_mm_ns": None,
    "use_position_errors": True,
    "use_momentum_errors": True,
    "use_centroid_errors": True,
    "use_internal_cloud": True,
    "apply_to_active_observers": True,
    "apply_to_active_sources": True,
    "apply_to_passive_sources": True,
    "apply_to_passive_updates": False,
    "seed": 12345,
    "refresh_policy": "fixed_per_particle",
}

DEFAULT_DRIVER_TRAIN: Dict[str, Any] = {
    "enabled": False,
    "bunch_count": 1,
    "z_spacing_mm": 0.0,
    "z_offsets_mm": [],
    "prehistory_steps": 0,
    "preserve_prehistory_in_output": False,
}

DEFAULT_ADAPTIVE_TIMESTEP: Dict[str, Any] = {
    "enabled": False,
    "energy_jump_threshold": 0.10,
    "timestep_reduction_factor": 3,
    "min_timestep_factor": 1e-4,
    "cooldown_steps": 10,
    "probe_threshold": 0.01,
    "max_probe_steps": 3,
    "debug": False,
    "bunch_proximity_enabled": False,
    "bunch_proximity_sigma_mm": 5.0,
    "bunch_proximity_n_sigma": 5.0,
    "bunch_proximity_reduction_factor": 10.0,
    "bunch_proximity_transition_n_sigma": 2.0,
}

SIMULATION_TYPE_ALIASES: Mapping[str, SimulationType] = {
    "conducting-wall": SimulationType.CONDUCTING_WALL,
    "conducting_wall": SimulationType.CONDUCTING_WALL,
    "wall": SimulationType.CONDUCTING_WALL,
    "switching-wall": SimulationType.SWITCHING_WALL,
    "switching_wall": SimulationType.SWITCHING_WALL,
    "switching": SimulationType.SWITCHING_WALL,
    "bunch-to-bunch": SimulationType.BUNCH_TO_BUNCH,
    "bunch_to_bunch": SimulationType.BUNCH_TO_BUNCH,
    "bunch": SimulationType.BUNCH_TO_BUNCH,
}

STARTUP_MODE_ALIASES: Mapping[str, StartupMode] = {
    "cold-start": StartupMode.COLD_START,
    "cold_start": StartupMode.COLD_START,
    "cold": StartupMode.COLD_START,
    "approximate-back-history": StartupMode.APPROXIMATE_BACK_HISTORY,
    "approximate_back_history": StartupMode.APPROXIMATE_BACK_HISTORY,
    "approximate": StartupMode.APPROXIMATE_BACK_HISTORY,
}

REQUIRED_PARTICLE_FIELDS: Iterable[str] = (
    "kinetic_energy_mev",
    "mass_amu",
    "charge_sign",
)


@dataclass(slots=True)
class SimulationRequest:
    """Container for the data required to run a simulation."""

    config: IntegratorConfig
    rider: ParticleState
    driver: Optional[ParticleState]
    external_field: Optional[ExternalFieldConfig] = None
    space_charge: Optional[SpaceChargeConfig] = None
    adaptive_timestep: Optional[AdaptiveTimestepConfig] = None
    self_consistency: Optional[SelfConsistencyConfig] = None
    auto_duration_enabled: bool = False
    auto_duration_crossing_steps: int = 200
    auto_duration_post_factor: float = 2.0


class SimulationConfigError(RuntimeError):
    """Raised when the CLI receives an invalid or incomplete configuration."""


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        prog="lw-simulate",
        description=(
            "Run Liénard–Wiechert retarded-field simulations using the modern "
            "core integrator. Provide overrides with flags or supply a JSON "
            "configuration file for advanced scenarios, or run parameter sweeps "
            "with --sweep-config."
        ),
    )
    # argparse's built-in negative-number matcher does not classify values like
    # ``-1.5e9`` as numeric, so nargs vector options otherwise reject common
    # scientific-notation field strengths.
    parser._negative_number_matcher = re.compile(  # noqa: SLF001
        r"^-\d+(\.\d*)?([eE][+-]?\d+)?$|^-\.\d+([eE][+-]?\d+)?$"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON file describing the simulation parameters and bunches.",
    )
    parser.add_argument(
        "--sweep-config",
        type=Path,
        dest="sweep_config",
        help="Path to a JSON sweep configuration file for parameter sweeps.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        dest="workers",
        help="Number of parallel worker processes for sweep execution. Default is sequential.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        dest="results_file",
        help="Path to a saved sweep or optimization results JSON file to summarize.",
    )
    parser.add_argument(
        "--log-verbosity",
        type=str,
        dest="log_verbosity",
        choices=["none", "truncated", "full"],
        help="Override log verbosity level for sweeps (none, truncated, or full). "
        "Default is from config. 'full' shows all SC iterations and adaptive timestep details.",
    )
    parser.add_argument(
        "--sc-verbosity",
        type=int,
        dest="sc_verbosity",
        choices=[0, 1, 2, 3],
        help="Override self-consistency verbosity (0=silent, 1=summary, 2=failures, 3=full detail).",
    )
    parser.add_argument(
        "--adaptive-debug",
        dest="adaptive_debug",
        action="store_true",
        help="Enable adaptive timestep debug output.",
    )
    parser.add_argument(
        "--no-adaptive-debug",
        dest="adaptive_debug",
        action="store_false",
        help="Disable adaptive timestep debug output.",
    )
    parser.set_defaults(adaptive_debug=None)
    parser.add_argument(
        "--adaptive-timestep",
        dest="adaptive_timestep_enabled",
        action="store_true",
        help="Enable adaptive timestep refinement.",
    )
    parser.add_argument(
        "--no-adaptive-timestep",
        dest="adaptive_timestep_enabled",
        action="store_false",
        help="Disable adaptive timestep refinement.",
    )
    parser.set_defaults(adaptive_timestep_enabled=None)
    parser.add_argument(
        "--adaptive-timestep-threshold",
        type=float,
        dest="adaptive_timestep_threshold",
        help="Fractional energy jump threshold for adaptive timestep refinement.",
    )
    parser.add_argument(
        "--adaptive-timestep-reduction-factor",
        type=int,
        dest="adaptive_timestep_reduction_factor",
        help="Timestep divisor used after an adaptive refinement trigger.",
    )
    parser.add_argument(
        "--adaptive-timestep-min-factor",
        type=float,
        dest="adaptive_timestep_min_factor",
        help="Minimum timestep as a fraction of the base timestep.",
    )
    parser.add_argument(
        "--adaptive-bunch-proximity",
        dest="adaptive_timestep_bunch_proximity_enabled",
        action="store_true",
        help="Enable BUNCH_TO_BUNCH timestep refinement as bunch centroids approach.",
    )
    parser.add_argument(
        "--no-adaptive-bunch-proximity",
        dest="adaptive_timestep_bunch_proximity_enabled",
        action="store_false",
        help="Disable BUNCH_TO_BUNCH bunch-proximity timestep refinement.",
    )
    parser.set_defaults(adaptive_timestep_bunch_proximity_enabled=None)
    parser.add_argument(
        "--adaptive-bunch-proximity-sigma-mm",
        type=float,
        dest="adaptive_timestep_bunch_proximity_sigma_mm",
        help="Characteristic bunch length sigma in mm for proximity refinement.",
    )
    parser.add_argument(
        "--adaptive-bunch-proximity-n-sigma",
        type=float,
        dest="adaptive_timestep_bunch_proximity_n_sigma",
        help="Start proximity refinement below this centroid separation in sigma.",
    )
    parser.add_argument(
        "--adaptive-bunch-proximity-reduction-factor",
        type=float,
        dest="adaptive_timestep_bunch_proximity_reduction_factor",
        help="Maximum timestep divisor for bunch-proximity refinement.",
    )
    parser.add_argument(
        "--adaptive-bunch-proximity-transition-n-sigma",
        type=float,
        dest="adaptive_timestep_bunch_proximity_transition_n_sigma",
        help="Ramp-in width in sigma for bunch-proximity refinement.",
    )
    parser.add_argument(
        "--space-charge",
        action="store_true",
        default=False,
        help="Enable intra-bunch space-charge forces (rider-rider retarded Liénard-Wiechert).",
    )
    parser.add_argument(
        "--space-charge-softening-mm",
        type=float,
        default=0.0,
        metavar="MM",
        help="Plummer softening length (mm) for space-charge interactions. Default: 0 (no softening).",
    )
    parser.add_argument(
        "--external-field",
        action="store_true",
        default=False,
        help="Enable a prescribed uniform external field.",
    )
    parser.add_argument(
        "--external-e-field-native",
        type=float,
        nargs=3,
        metavar=("EX", "EY", "EZ"),
        help="Uniform electric field vector in native solver units.",
    )
    parser.add_argument(
        "--external-e-field-v-per-m",
        type=float,
        nargs=3,
        metavar=("EX", "EY", "EZ"),
        help="Uniform electric field vector in V/m, converted to native units.",
    )
    parser.add_argument(
        "--external-b-field-native",
        type=float,
        nargs=3,
        metavar=("BX", "BY", "BZ"),
        help="Uniform magnetic field vector in native solver units.",
    )
    for axis in ("x", "y", "z", "t"):
        parser.add_argument(
            f"--external-field-{axis}-min",
            type=float,
            dest=f"external_field_{axis}_min",
            help=f"Lower {axis} bound for applying the external field.",
        )
        parser.add_argument(
            f"--external-field-{axis}-max",
            type=float,
            dest=f"external_field_{axis}_max",
            help=f"Upper {axis} bound for applying the external field.",
        )
    parser.add_argument(
        "--auto-duration",
        action="store_true",
        default=False,
        dest="auto_duration",
        help="Auto-compute timestep and steps to cover the BUNCH_TO_BUNCH crossing window.",
    )
    parser.add_argument(
        "--auto-duration-steps",
        type=int,
        default=None,
        dest="auto_duration_crossing_steps",
        help="Target steps to cover the approach (default: 200).",
    )
    parser.add_argument(
        "--auto-duration-post-factor",
        type=float,
        default=None,
        dest="auto_duration_post_factor",
        help="Total steps = crossing_steps * post_factor (default: 2.0).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Total number of integration steps (overrides configuration/default).",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        dest="time_step",
        help="Integrator time step in nanoseconds.",
    )
    parser.add_argument(
        "--simulation-type",
        choices=sorted(set(SIMULATION_TYPE_ALIASES.keys())),
        help="Simulation boundary condition (conducting-wall, switching-wall, bunch-to-bunch).",
    )
    parser.add_argument(
        "--wall-position",
        type=float,
        help="Position of the conducting wall in millimetres.",
    )
    parser.add_argument(
        "--aperture-radius",
        type=float,
        dest="aperture_radius",
        help="Radius of the aperture in millimetres.",
    )
    parser.add_argument(
        "--bunch-mean",
        type=float,
        dest="bunch_mean",
        help="Initial bunch separation parameter.",
    )
    parser.add_argument(
        "--cavity-spacing",
        type=float,
        dest="cavity_spacing",
        help="Cavity spacing for switching-wall simulations.",
    )
    parser.add_argument(
        "--z-cutoff",
        type=float,
        dest="z_cutoff",
        help="Longitudinal cutoff for switching-wall simulations.",
    )
    parser.add_argument(
        "--z-cutoff-mode",
        choices=("absolute", "relative"),
        dest="z_cutoff_mode",
        help="Interpret z-cutoff as an absolute z position or relative rider distance.",
    )
    parser.add_argument(
        "--cavity-exit",
        dest="cavity_exit_enabled",
        action="store_true",
        help="Enable BUNCH_TO_BUNCH cutoff when either bunch reaches the opposite cavity exit.",
    )
    parser.add_argument(
        "--no-cavity-exit",
        dest="cavity_exit_enabled",
        action="store_false",
        default=None,
        help="Disable cavity-exit cutoff explicitly.",
    )
    parser.add_argument(
        "--cavity-exit-length-mm",
        type=float,
        dest="cavity_exit_length_mm",
        help="Optional absolute cavity length; default uses initial rider-driver separation.",
    )
    parser.add_argument(
        "--cavity-exit-mode",
        choices=("first_exit", "rider_exit_with_driver_tail"),
        dest="cavity_exit_mode",
        help=(
            "Cavity-exit behavior. 'first_exit' stops on the first rider/driver "
            "exit; 'rider_exit_with_driver_tail' stops on rider exit while "
            "muting driver-train bunches after their exit-tail window."
        ),
    )
    parser.add_argument(
        "--beamline-geometry-enabled",
        dest="beamline_geometry_enabled",
        action="store_true",
        help="Enable beamline-geometry line-of-sight screening between bunches.",
    )
    parser.add_argument(
        "--no-beamline-geometry",
        dest="beamline_geometry_enabled",
        action="store_false",
        default=None,
        help="Disable beamline-geometry screening explicitly.",
    )
    parser.add_argument(
        "--beamline-geometry-file",
        type=str,
        dest="beamline_geometry_file",
        help="Path to a JSON file defining the beamline_geometry block (occluders list).",
    )
    parser.add_argument(
        "--chrono-mode",
        choices=("averaged", "fast"),
        help=(
            "Retardation sampling strategy. 'fast' is the maintained default; "
            "'averaged' is retained for diagnostics with approximate back-history."
        ),
    )
    parser.add_argument(
        "--chrono-interpolate",
        dest="chrono_interpolate",
        action="store_true",
        help="Enable retarded-time source-state interpolation independent of SC iterations.",
    )
    parser.add_argument(
        "--no-chrono-interpolate",
        dest="chrono_interpolate",
        action="store_false",
        help="Disable retarded-time source-state interpolation.",
    )
    parser.set_defaults(chrono_interpolate=None)
    parser.add_argument(
        "--chrono-tolerance",
        type=float,
        dest="chrono_tolerance",
        help="Chrono-match time residual tolerance in ns for interpolation/warnings.",
    )
    parser.add_argument(
        "--chrono-high-precision",
        dest="chrono_high_precision",
        action="store_true",
        help="Use cubic interpolation and interpolate source positions when chrono interpolation is active.",
    )
    parser.add_argument(
        "--no-chrono-high-precision",
        dest="chrono_high_precision",
        action="store_false",
        help="Disable high-precision chrono interpolation.",
    )
    parser.set_defaults(chrono_high_precision=None)
    parser.add_argument(
        "--chrono-adaptive-tolerance",
        dest="chrono_adaptive_tolerance",
        action="store_true",
        help="Scale chrono tolerance automatically with the integration timestep.",
    )
    parser.add_argument(
        "--no-chrono-adaptive-tolerance",
        dest="chrono_adaptive_tolerance",
        action="store_false",
        help="Use the fixed chrono tolerance instead of timestep-scaled tolerance.",
    )
    parser.set_defaults(chrono_adaptive_tolerance=None)
    parser.add_argument(
        "--startup-mode",
        choices=("cold-start", "approximate-back-history"),
        dest="startup_mode",
        help=(
            "Early-step strategy: 'cold-start' suppresses forces until the "
            "observer has travelled sufficiently, while "
            "'approximate-back-history' assumes constant source velocity."
        ),
    )
    parser.add_argument(
        "--radiation-reaction-mode",
        dest="radiation_reaction_mode",
        choices=RADIATION_REACTION_MODE_CHOICES,
        help=(
            "Radiation-reaction mode for single runs. Default: medina_lad. "
            "Choose off or diagnostic_only for baselines."
        ),
    )
    parser.add_argument(
        "--image-subcharge-count",
        type=int,
        dest="image_subcharge_count",
        help="Number of subcharges used when mirroring a conducting wall (4-128).",
    )
    parser.add_argument(
        "--image-weighting",
        dest="use_image_weighting",
        action="store_true",
        help="Enable radial weighting for conducting-wall image subcharges.",
    )
    parser.add_argument(
        "--no-image-weighting",
        dest="use_image_weighting",
        action="store_false",
        help="Disable radial weighting for conducting-wall image subcharges.",
    )
    parser.set_defaults(use_image_weighting=None)
    parser.add_argument(
        "--particle-loss",
        dest="particle_loss_enabled",
        action="store_true",
        help="Enable fixed-size physical particle-loss tracking.",
    )
    parser.add_argument(
        "--no-particle-loss",
        dest="particle_loss_enabled",
        action="store_false",
        help="Disable fixed-size physical particle-loss tracking.",
    )
    parser.set_defaults(particle_loss_enabled=None)
    parser.add_argument(
        "--loss-radius-mm",
        type=float,
        dest="particle_loss_radius_mm",
        help="Absolute cylindrical transverse loss radius in millimetres.",
    )
    parser.add_argument(
        "--no-conducting-wall-aperture-loss",
        dest="conducting_wall_aperture_loss_enabled",
        action="store_false",
        help="Disable conducting-wall aperture-plane loss checks.",
    )
    parser.set_defaults(conducting_wall_aperture_loss_enabled=None)
    parser.add_argument(
        "--particle-loss-initial-radial-quantile",
        type=float,
        dest="particle_loss_initial_radial_quantile",
        help="Optional initial radial quantile used to derive a robust envelope.",
    )
    parser.add_argument(
        "--particle-loss-initial-radial-multiplier",
        type=float,
        dest="particle_loss_initial_radial_multiplier",
        help="Multiplier for the initial radial quantile envelope.",
    )
    parser.add_argument(
        "--particle-loss-initial-radial-margin-mm",
        type=float,
        dest="particle_loss_initial_radial_margin_mm",
        help="Additive margin for the initial radial quantile envelope.",
    )
    parser.add_argument(
        "--pseudo-grid",
        dest="pseudo_grid_enabled",
        action="store_true",
        help=(
            "Enable the experimental pseudo-grid configuration surface for "
            "BUNCH_TO_BUNCH runs. The reduced solver path is still under development."
        ),
    )
    parser.add_argument(
        "--no-pseudo-grid",
        dest="pseudo_grid_enabled",
        action="store_false",
        help="Disable pseudo-grid mode explicitly.",
    )
    parser.set_defaults(pseudo_grid_enabled=None)
    parser.add_argument(
        "--pseudo-grid-active-rider-count",
        type=int,
        dest="pseudo_grid_active_rider_count",
        help="Number of active rider particles to solve directly each step.",
    )
    parser.add_argument(
        "--pseudo-grid-active-driver-count",
        type=int,
        dest="pseudo_grid_active_driver_count",
        help="Number of active driver particles to solve directly each step.",
    )
    parser.add_argument(
        "--pseudo-grid-passive-neighbor-count",
        type=int,
        dest="pseudo_grid_passive_neighbor_count",
        help="Nearest active neighbors used when advancing passive particles.",
    )
    parser.add_argument(
        "--pseudo-grid-coverage-strategy",
        choices=("farthest_point_staleness", "farthest_point"),
        dest="pseudo_grid_coverage_strategy",
        help="Coverage strategy used when selecting the active subset.",
    )
    parser.add_argument(
        "--pseudo-grid-coverage-space",
        choices=("position", "phase_space"),
        dest="pseudo_grid_coverage_space",
        help="Metric space used for pseudo-grid coverage and neighbor searches.",
    )
    parser.add_argument(
        "--pseudo-grid-pair-reuse-window",
        type=int,
        dest="pseudo_grid_pair_reuse_window",
        help="Recent-match window used to discourage repeated active pairings.",
    )
    parser.add_argument(
        "--pseudo-grid-source-weighting-mode",
        choices=("inverse_distance", "nearest"),
        dest="pseudo_grid_source_weighting_mode",
        help="Source weighting mode for represented passive charge.",
    )
    parser.add_argument(
        "--pseudo-grid-loss-tracking",
        dest="pseudo_grid_loss_tracking_enabled",
        action="store_true",
        help="Enable explicit pseudo-grid particle-loss tracking.",
    )
    parser.add_argument(
        "--no-pseudo-grid-loss-tracking",
        dest="pseudo_grid_loss_tracking_enabled",
        action="store_false",
        help="Disable pseudo-grid particle-loss tracking.",
    )
    parser.set_defaults(pseudo_grid_loss_tracking_enabled=None)
    parser.add_argument(
        "--pseudo-grid-causal-pruning",
        dest="pseudo_grid_causal_history_pruning_enabled",
        action="store_true",
        help="Enable causal-history pruning for the pseudo-grid history window.",
    )
    parser.add_argument(
        "--no-pseudo-grid-causal-pruning",
        dest="pseudo_grid_causal_history_pruning_enabled",
        action="store_false",
        help="Disable pseudo-grid causal-history pruning explicitly.",
    )
    parser.set_defaults(pseudo_grid_causal_history_pruning_enabled=None)
    parser.add_argument(
        "--pseudo-grid-causal-safety-margin-steps",
        type=int,
        dest="pseudo_grid_causal_history_safety_margin_steps",
        help="Safety margin, in steps, retained beyond the causal pruning bound.",
    )
    parser.add_argument(
        "--macroparticle-smearing",
        dest="macroparticle_smearing_enabled",
        action="store_true",
        help="Enable bounded deterministic macroparticle source smearing.",
    )
    parser.add_argument(
        "--no-macroparticle-smearing",
        dest="macroparticle_smearing_enabled",
        action="store_false",
        help="Disable macroparticle source smearing explicitly.",
    )
    parser.set_defaults(macroparticle_smearing_enabled=None)
    parser.add_argument(
        "--macroparticle-smearing-subcharge-count",
        type=int,
        dest="macroparticle_smearing_subcharge_count",
        help="Number of deterministic source subcharges per macroparticle.",
    )
    parser.add_argument(
        "--macroparticle-smearing-sigma-multiplier",
        type=float,
        dest="macroparticle_smearing_sigma_multiplier",
        help="Multiplier for automatically derived smearing widths.",
    )
    parser.add_argument(
        "--macroparticle-smearing-position-sigma-mm",
        type=float,
        dest="macroparticle_smearing_position_sigma_mm",
        help="Override transverse position smearing sigma in millimetres.",
    )
    parser.add_argument(
        "--macroparticle-smearing-longitudinal-sigma-mm",
        type=float,
        dest="macroparticle_smearing_longitudinal_sigma_mm",
        help="Override longitudinal position smearing sigma in millimetres.",
    )
    parser.add_argument(
        "--macroparticle-smearing-momentum-sigma",
        type=float,
        dest="macroparticle_smearing_momentum_sigma_amu_mm_ns",
        help="Momentum smearing sigma in amu*mm/ns.",
    )
    parser.add_argument(
        "--macroparticle-smearing-seed",
        type=int,
        dest="macroparticle_smearing_seed",
        help="Seed for deterministic macroparticle smearing offsets.",
    )
    parser.add_argument(
        "--macroparticle-smearing-refresh-policy",
        choices=("fixed-per-particle", "fixed_per_particle", "per-step", "per_step"),
        dest="macroparticle_smearing_refresh_policy",
        help="Use persistent per-particle offsets or refresh them each step.",
    )

    parser.add_argument(
        "--macroparticle-smearing-passive-updates",
        dest="macroparticle_smearing_apply_to_passive_updates",
        action="store_true",
        help="Enable experimental smearing for pseudo-grid passive updates.",
    )
    parser.set_defaults(macroparticle_smearing_apply_to_passive_updates=None)
    parser.add_argument(
        "--driver-train",
        dest="driver_train_enabled",
        action="store_true",
        help="Enable flat BUNCH_TO_BUNCH driver-train expansion.",
    )
    parser.add_argument(
        "--no-driver-train",
        dest="driver_train_enabled",
        action="store_false",
        help="Disable driver-train mode explicitly.",
    )
    parser.set_defaults(driver_train_enabled=None)
    parser.add_argument(
        "--driver-train-bunch-count",
        type=int,
        dest="driver_train_bunch_count",
        help="Number of longitudinal driver bunch copies in the train.",
    )
    parser.add_argument(
        "--driver-train-z-spacing-mm",
        type=float,
        dest="driver_train_z_spacing_mm",
        help="Longitudinal spacing between driver bunch copies in millimetres.",
    )
    parser.add_argument(
        "--driver-train-z-offsets-mm",
        type=float,
        nargs="+",
        dest="driver_train_z_offsets_mm",
        help="Explicit z offsets for each driver bunch copy in millimetres.",
    )
    parser.add_argument(
        "--driver-train-prehistory-steps",
        type=int,
        dest="driver_train_prehistory_steps",
        help="Number of inertial coasting history rows before the active window.",
    )
    parser.add_argument(
        "--driver-train-preserve-prehistory",
        dest="driver_train_preserve_prehistory_in_output",
        action="store_true",
        help="Keep prehistory rows in returned/saved trajectories.",
    )
    parser.add_argument(
        "--driver-train-trim-prehistory",
        dest="driver_train_preserve_prehistory_in_output",
        action="store_false",
        help="Trim prehistory rows from returned/saved trajectories.",
    )
    parser.set_defaults(driver_train_preserve_prehistory_in_output=None)
    parser.add_argument(
        "--driver-from-rider",
        action="store_true",
        help=(
            "For bunch-to-bunch simulations, clone the rider bunch to use as the "
            "driver when no driver configuration is supplied."
        ),
    )
    parser.add_argument(
        "--space-charge-bunch-sigma-mm",
        type=float,
        default=None,
        help=(
            "Bunch width used to auto-delay retarded intra-bunch space charge "
            "(default: 0.01 mm)."
        ),
    )
    parser.add_argument(
        "--space-charge-min-retarded-steps",
        type=int,
        default=None,
        help=(
            "Explicit step threshold before retarded intra-bunch space charge "
            "is used; omit to derive it from bunch sigma and timestep."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write a JSON summary report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary (still writes JSON if requested).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=_version_string(),
    )
    return parser.parse_args(argv)


def _version_string() -> str:
    from core._version import __version__

    return f"lw-integrator {__version__}"


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------


def build_request(args: argparse.Namespace) -> SimulationRequest:
    """Combine defaults, configuration file, and CLI overrides."""

    file_payload: Dict[str, Any] = {}
    if args.config is not None:
        file_payload = _load_config(args.config)

    simulation_payload = _merge_simulation_payload(file_payload, args)
    rider_payload = _merge_particle_payload(
        file_payload.get("rider", {}),
        overrides={},
        defaults=DEFAULT_RIDER,
    )

    driver_payload: Optional[Dict[str, Any]] = None
    if "driver" in file_payload:
        driver_payload = _merge_particle_payload(
            file_payload["driver"], overrides={}, defaults=DEFAULT_RIDER
        )

    config = _build_integrator_config(simulation_payload)
    external_field = _build_external_field_config(
        simulation_payload.get("external_field")
    )
    space_charge = _build_space_charge_config(simulation_payload)
    adaptive_timestep = _build_adaptive_timestep_config(simulation_payload)
    self_consistency = _build_self_consistency_config(simulation_payload)
    rider_state = _build_particle_state(rider_payload)

    driver_state: Optional[ParticleState] = None
    if config.simulation_type is SimulationType.BUNCH_TO_BUNCH:
        if driver_payload is not None:
            driver_state = _build_particle_state(driver_payload)
        elif args.driver_from_rider:
            driver_state = {key: np.copy(value) for key, value in rider_state.items()}
        else:
            raise SimulationConfigError(
                "BUNCH_TO_BUNCH simulations require a driver bunch. Provide "
                "one in the configuration file or pass --driver-from-rider."
            )
    elif driver_payload is not None:
        driver_state = _build_particle_state(driver_payload)

    auto_duration_enabled = bool(simulation_payload.get("auto_duration_enabled", False))
    auto_duration_crossing_steps = int(
        simulation_payload.get("auto_duration_crossing_steps", 200)
    )
    auto_duration_post_factor = float(
        simulation_payload.get("auto_duration_post_factor", 2.0)
    )

    if auto_duration_enabled:
        if config.simulation_type is not SimulationType.BUNCH_TO_BUNCH:
            raise SimulationConfigError(
                "auto_duration is only supported for BUNCH_TO_BUNCH simulations"
            )
        if auto_duration_crossing_steps <= 0:
            raise SimulationConfigError(
                "auto_duration_crossing_steps must be a positive integer"
            )
        if auto_duration_post_factor <= 0.0:
            raise SimulationConfigError("auto_duration_post_factor must be positive")

    return SimulationRequest(
        config=config,
        rider=rider_state,
        driver=driver_state,
        external_field=external_field,
        space_charge=space_charge,
        adaptive_timestep=adaptive_timestep,
        self_consistency=self_consistency,
        auto_duration_enabled=auto_duration_enabled,
        auto_duration_crossing_steps=auto_duration_crossing_steps,
        auto_duration_post_factor=auto_duration_post_factor,
    )


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SimulationConfigError(f"Configuration file not found: {path}") from exc
    except OSError as exc:  # pragma: no cover - filesystem errors
        raise SimulationConfigError(
            f"Unable to read configuration file {path}: {exc}"
        ) from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SimulationConfigError(
            f"Configuration file {path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, MutableMapping):
        raise SimulationConfigError(
            "Configuration file must contain a JSON object at the top level."
        )

    return dict(payload)


def _merge_simulation_payload(
    file_payload: Mapping[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    result = dict(DEFAULT_SIMULATION)
    result["particle_loss"] = dict(DEFAULT_PARTICLE_LOSS)
    result["pseudo_grid"] = dict(DEFAULT_PSEUDO_GRID)
    result["driver_train"] = dict(DEFAULT_DRIVER_TRAIN)
    result["macroparticle_smearing"] = dict(DEFAULT_MACROPARTICLE_SMEARING)
    result["adaptive_timestep"] = dict(DEFAULT_ADAPTIVE_TIMESTEP)
    for key in DEFAULT_SIMULATION:
        if key in file_payload:
            result[key] = file_payload[key]
    file_particle_loss = file_payload.get("particle_loss")
    if isinstance(file_particle_loss, Mapping):
        result["particle_loss"].update(file_particle_loss)
        if "enabled" not in file_particle_loss and any(
            file_particle_loss.get(key) is not None
            for key in ("loss_radius_mm", "initial_radial_quantile")
        ):
            result["particle_loss"]["enabled"] = True
    file_pseudo_grid = file_payload.get("pseudo_grid")
    if isinstance(file_pseudo_grid, Mapping):
        result["pseudo_grid"].update(file_pseudo_grid)
    file_driver_train = file_payload.get("driver_train")
    if isinstance(file_driver_train, Mapping):
        result["driver_train"].update(file_driver_train)
    file_smearing = file_payload.get("macroparticle_smearing")
    if isinstance(file_smearing, Mapping):
        result["macroparticle_smearing"].update(file_smearing)
    file_adaptive_timestep = file_payload.get("adaptive_timestep")
    if isinstance(file_adaptive_timestep, Mapping):
        result["adaptive_timestep"].update(file_adaptive_timestep)
    if "external_field" in file_payload:
        result["external_field"] = file_payload["external_field"]

    passthrough_keys = (
        "space_charge_enabled",
        "space_charge_retarded",
        "space_charge_softening_mm",
        "space_charge_bunch_sigma_mm",
        "space_charge_min_retarded_steps",
        "auto_duration_enabled",
        "auto_duration_crossing_steps",
        "auto_duration_post_factor",
        "self_consistency_enabled",
        "self_consistency_convergence_mode",
        "self_consistency_target_ms_tolerance",
        "self_consistency_max_iterations",
        "self_consistency_mass_shell_tolerance",
        "self_consistency_mass_shell_relaxation",
        "self_consistency_verbosity",
        "self_consistency_gamma_reconciliation_method",
        "self_consistency_gamma_reconciliation_fixed_weight",
        "chrono_interpolate",
        "chrono_tolerance",
        "chrono_high_precision",
        "chrono_adaptive_tolerance",
        "self_consistency_chrono_interpolate",
        "self_consistency_chrono_tolerance",
        "self_consistency_chrono_high_precision",
        "self_consistency_chrono_adaptive_tolerance",
        "self_consistency_chrono_matching_mode",
    )
    for key in passthrough_keys:
        if key in file_payload:
            result[key] = file_payload[key]
    legacy_adaptive_keys = {
        "adaptive_timestep_enabled": "enabled",
        "adaptive_timestep_threshold": "energy_jump_threshold",
        "adaptive_timestep_reduction_factor": "timestep_reduction_factor",
        "adaptive_timestep_min_factor": "min_timestep_factor",
        "adaptive_timestep_cooldown_steps": "cooldown_steps",
        "adaptive_timestep_probe_threshold": "probe_threshold",
        "adaptive_timestep_max_probe_steps": "max_probe_steps",
        "adaptive_timestep_debug": "debug",
        "adaptive_timestep_bunch_proximity_enabled": "bunch_proximity_enabled",
        "adaptive_timestep_bunch_proximity_sigma_mm": "bunch_proximity_sigma_mm",
        "adaptive_timestep_bunch_proximity_n_sigma": "bunch_proximity_n_sigma",
        "adaptive_timestep_bunch_proximity_reduction_factor": "bunch_proximity_reduction_factor",
        "adaptive_timestep_bunch_proximity_transition_n_sigma": "bunch_proximity_transition_n_sigma",
    }
    for source_key, target_key in legacy_adaptive_keys.items():
        if source_key in file_payload:
            result["adaptive_timestep"][target_key] = file_payload[source_key]

    override_keys = (
        "steps",
        "time_step",
        "simulation_type",
        "wall_position",
        "aperture_radius",
        "bunch_mean",
        "cavity_spacing",
        "z_cutoff",
        "z_cutoff_mode",
        "chrono_mode",
        "startup_mode",
        "radiation_reaction_mode",
        "image_subcharge_count",
        "use_image_weighting",
    )

    for key in override_keys:
        if getattr(args, key, None) is not None:
            result[key] = getattr(args, key)

    for key in (
        "chrono_interpolate",
        "chrono_tolerance",
        "chrono_high_precision",
        "chrono_adaptive_tolerance",
    ):
        if getattr(args, key, None) is not None:
            result[key] = getattr(args, key)
    if "chrono_matching_mode" not in result and "chrono_mode" in result:
        result["chrono_matching_mode"] = (
            str(result["chrono_mode"]).replace("-", "_").upper()
        )

    cavity_exit = result.get("cavity_exit")
    if not isinstance(cavity_exit, MutableMapping):
        cavity_exit = {}
        result["cavity_exit"] = cavity_exit
    if getattr(args, "cavity_exit_enabled", None) is not None:
        cavity_exit["enabled"] = bool(args.cavity_exit_enabled)
    if getattr(args, "cavity_exit_mode", None) is not None:
        cavity_exit["mode"] = args.cavity_exit_mode
    if getattr(args, "cavity_exit_length_mm", None) is not None:
        cavity_exit["cavity_length_mm"] = args.cavity_exit_length_mm

    beamline_geometry = result.get("beamline_geometry")
    if not isinstance(beamline_geometry, MutableMapping):
        beamline_geometry = {}
        result["beamline_geometry"] = beamline_geometry
    if getattr(args, "beamline_geometry_enabled", None) is not None:
        beamline_geometry["enabled"] = bool(args.beamline_geometry_enabled)
    if getattr(args, "beamline_geometry_file", None):
        with open(args.beamline_geometry_file) as f:
            file_block = json.load(f)
        if isinstance(file_block, dict):
            beamline_geometry.update(file_block)

    if getattr(args, "space_charge", False):
        result["space_charge_enabled"] = True
    if getattr(args, "space_charge_softening_mm", 0.0) != 0.0:
        result["space_charge_softening_mm"] = args.space_charge_softening_mm
    if getattr(args, "space_charge_bunch_sigma_mm", None) is not None:
        result["space_charge_bunch_sigma_mm"] = args.space_charge_bunch_sigma_mm
    if getattr(args, "space_charge_min_retarded_steps", None) is not None:
        result["space_charge_min_retarded_steps"] = args.space_charge_min_retarded_steps

    external_field = result.get("external_field")
    if not isinstance(external_field, MutableMapping):
        external_field = {}
    else:
        external_field = dict(external_field)

    if getattr(args, "external_field", False):
        external_field["enabled"] = True
    if getattr(args, "external_e_field_native", None) is not None:
        external_field["electric_field_native"] = args.external_e_field_native
        external_field["enabled"] = True
    if getattr(args, "external_e_field_v_per_m", None) is not None:
        external_field["electric_field_v_per_m"] = args.external_e_field_v_per_m
        external_field["enabled"] = True
    if getattr(args, "external_b_field_native", None) is not None:
        external_field["magnetic_field_native"] = args.external_b_field_native
        external_field["enabled"] = True
    for axis in ("x", "y", "z", "t"):
        for bound in ("min", "max"):
            arg_name = f"external_field_{axis}_{bound}"
            value = getattr(args, arg_name, None)
            if value is not None:
                external_field[f"{axis}_{bound}"] = value
                external_field["enabled"] = True
    if external_field:
        result["external_field"] = external_field

    if getattr(args, "auto_duration", False):
        result["auto_duration_enabled"] = True
    if getattr(args, "auto_duration_crossing_steps", None) is not None:
        result["auto_duration_crossing_steps"] = args.auto_duration_crossing_steps
    if getattr(args, "auto_duration_post_factor", None) is not None:
        result["auto_duration_post_factor"] = args.auto_duration_post_factor

    particle_loss = result["particle_loss"]
    particle_loss_overrides = {
        "enabled": getattr(args, "particle_loss_enabled", None),
        "loss_radius_mm": getattr(args, "particle_loss_radius_mm", None),
        "conducting_wall_aperture_loss_enabled": getattr(
            args,
            "conducting_wall_aperture_loss_enabled",
            None,
        ),
        "initial_radial_quantile": getattr(
            args,
            "particle_loss_initial_radial_quantile",
            None,
        ),
        "initial_radial_multiplier": getattr(
            args,
            "particle_loss_initial_radial_multiplier",
            None,
        ),
        "initial_radial_margin_mm": getattr(
            args,
            "particle_loss_initial_radial_margin_mm",
            None,
        ),
    }
    threshold_override_present = False
    for key, value in particle_loss_overrides.items():
        if value is not None:
            particle_loss[key] = value
            threshold_override_present = threshold_override_present or key in {
                "loss_radius_mm",
                "initial_radial_quantile",
            }
    if (
        threshold_override_present
        and getattr(args, "particle_loss_enabled", None) is None
    ):
        particle_loss["enabled"] = True

    pseudo_grid = result["pseudo_grid"]
    pseudo_grid_override_keys = (
        "enabled",
        "active_rider_count",
        "active_driver_count",
        "passive_neighbor_count",
        "coverage_strategy",
        "coverage_space",
        "pair_reuse_window",
        "source_weighting_mode",
        "loss_tracking_enabled",
        "causal_history_pruning_enabled",
        "causal_history_safety_margin_steps",
    )
    for key in pseudo_grid_override_keys:
        arg_name = f"pseudo_grid_{key}"
        value = getattr(args, arg_name, None)
        if value is not None:
            pseudo_grid[key] = value

    smearing = result["macroparticle_smearing"]
    smearing_override_keys = (
        "enabled",
        "subcharge_count",
        "sigma_multiplier",
        "position_sigma_mm",
        "longitudinal_sigma_mm",
        "momentum_sigma_amu_mm_ns",
        "use_position_errors",
        "use_momentum_errors",
        "use_centroid_errors",
        "use_internal_cloud",
        "apply_to_active_observers",
        "apply_to_active_sources",
        "apply_to_passive_sources",
        "apply_to_passive_updates",
        "seed",
        "refresh_policy",
    )
    for key in smearing_override_keys:
        arg_name = f"macroparticle_smearing_{key}"
        value = getattr(args, arg_name, None)
        if value is not None:
            if key == "refresh_policy" and isinstance(value, str):
                value = value.replace("-", "_")
            smearing[key] = value

    driver_train = result["driver_train"]
    driver_train_override_keys = (
        "enabled",
        "bunch_count",
        "z_spacing_mm",
        "z_offsets_mm",
        "prehistory_steps",
        "preserve_prehistory_in_output",
    )
    for key in driver_train_override_keys:
        arg_name = f"driver_train_{key}"
        value = getattr(args, arg_name, None)
        if value is not None:
            driver_train[key] = value

    adaptive_timestep = result["adaptive_timestep"]
    adaptive_timestep_override_keys = (
        "enabled",
        "threshold",
        "reduction_factor",
        "min_factor",
        "bunch_proximity_enabled",
        "bunch_proximity_sigma_mm",
        "bunch_proximity_n_sigma",
        "bunch_proximity_reduction_factor",
        "bunch_proximity_transition_n_sigma",
    )
    adaptive_key_map = {
        "threshold": "energy_jump_threshold",
        "reduction_factor": "timestep_reduction_factor",
        "min_factor": "min_timestep_factor",
    }
    for key in adaptive_timestep_override_keys:
        arg_name = f"adaptive_timestep_{key}"
        value = getattr(args, arg_name, None)
        if value is not None:
            adaptive_timestep[adaptive_key_map.get(key, key)] = value
            if key == "bunch_proximity_enabled" and value:
                adaptive_timestep["enabled"] = True
    if getattr(args, "adaptive_debug", None) is not None:
        adaptive_timestep["debug"] = args.adaptive_debug

    if getattr(args, "sc_verbosity", None) is not None:
        result["self_consistency_verbosity"] = args.sc_verbosity

    return result


def _merge_particle_payload(
    file_payload: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    result = dict(defaults)
    for key, value in file_payload.items():
        result[key] = value
    for key, value in overrides.items():
        result[key] = value
    return result


def _build_integrator_config(payload: Mapping[str, Any]) -> IntegratorConfig:
    try:
        simulation_type = _parse_simulation_type(payload["simulation_type"])
    except KeyError as exc:
        raise SimulationConfigError(
            "Simulation configuration missing 'simulation_type'."
        ) from exc

    missing = [
        key
        for key in ("steps", "time_step", "wall_position", "aperture_radius")
        if key not in payload
    ]
    if missing:
        raise SimulationConfigError(
            f"Simulation configuration missing required fields: {', '.join(missing)}"
        )

    chrono_mode = _parse_chrono_mode(
        payload.get("chrono_mode", DEFAULT_SIMULATION["chrono_mode"])
    )
    startup_mode = _parse_startup_mode(
        payload.get("startup_mode", StartupMode.COLD_START)
    )
    image_subcharge_count = _parse_image_subcharge_count(
        payload.get(
            "image_subcharge_count", DEFAULT_SIMULATION["image_subcharge_count"]
        )
    )
    use_image_weighting = _parse_image_weighting(
        payload.get("use_image_weighting", DEFAULT_SIMULATION["use_image_weighting"])
    )

    particle_loss = _build_particle_loss_config(payload.get("particle_loss"))
    pseudo_grid = _build_pseudo_grid_config(payload.get("pseudo_grid"))
    driver_train = _build_driver_train_config(payload.get("driver_train"))
    cavity_exit = _build_cavity_exit_config(payload.get("cavity_exit"))
    beamline_geometry = _build_beamline_geometry_config(payload.get("beamline_geometry"))
    macroparticle_smearing = _build_macroparticle_smearing_config(
        payload.get("macroparticle_smearing")
    )

    return IntegratorConfig(
        steps=int(payload["steps"]),
        time_step=float(payload["time_step"]),
        wall_position=float(payload["wall_position"]),
        aperture_radius=float(payload["aperture_radius"]),
        simulation_type=simulation_type,
        chrono_mode=chrono_mode,
        startup_mode=startup_mode,
        bunch_mean=float(payload.get("bunch_mean", DEFAULT_SIMULATION["bunch_mean"])),
        cavity_spacing=float(
            payload.get("cavity_spacing", DEFAULT_SIMULATION["cavity_spacing"])
        ),
        z_cutoff=float(payload.get("z_cutoff", DEFAULT_SIMULATION["z_cutoff"])),
        z_cutoff_mode=str(
            payload.get("z_cutoff_mode", DEFAULT_SIMULATION["z_cutoff_mode"])
        ),
        image_subcharge_count=image_subcharge_count,
        use_image_weighting=use_image_weighting,
        radiation_reaction_mode=str(
            payload.get(
                "radiation_reaction_mode",
                DEFAULT_SIMULATION["radiation_reaction_mode"],
            )
        ),
        pseudo_grid=pseudo_grid,
        macroparticle_smearing=macroparticle_smearing,
        driver_train=driver_train,
        cavity_exit=cavity_exit,
        particle_loss=particle_loss,
        beamline_geometry=beamline_geometry,
    )


def _parse_simulation_type(value: Any) -> SimulationType:
    if isinstance(value, SimulationType):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        try:
            return SimulationType(int(value))
        except ValueError as exc:
            raise SimulationConfigError(
                f"Unknown simulation type integer: {value}"
            ) from exc
    if isinstance(value, str):
        key = value.strip().lower()
        if key in SIMULATION_TYPE_ALIASES:
            return SIMULATION_TYPE_ALIASES[key]
    raise SimulationConfigError(f"Unknown simulation type: {value!r}")


def _parse_chrono_mode(value: Any) -> ChronoMatchingMode:
    if isinstance(value, ChronoMatchingMode):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"fast", "legacy"}:
            return ChronoMatchingMode.FAST
        if key in {"averaged", "average", "blended"}:
            return ChronoMatchingMode.AVERAGED
        raise SimulationConfigError(
            f"Unknown chrono_mode value: {value!r}. Expected 'fast' or 'averaged'."
        )
    raise SimulationConfigError(
        "chrono_mode must be a string or ChronoMatchingMode instance"
    )


def _parse_startup_mode(value: Any) -> StartupMode:
    if isinstance(value, StartupMode):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in STARTUP_MODE_ALIASES:
            return STARTUP_MODE_ALIASES[key]
        raise SimulationConfigError(
            f"Unknown startup_mode value: {value!r}. Expected 'cold-start' or 'approximate-back-history'."
        )
    raise SimulationConfigError("startup_mode must be a string or StartupMode instance")


def _parse_image_subcharge_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise SimulationConfigError("image_subcharge_count must be an integer") from exc

    if not 4 <= count <= 128:
        raise SimulationConfigError(
            "image_subcharge_count must be between 4 and 128 inclusive"
        )
    return count


def _parse_image_weighting(value: Any) -> bool:
    if value is None:
        return bool(DEFAULT_SIMULATION["use_image_weighting"])
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"1", "true", "yes", "on"}:
            return True
        if key in {"0", "false", "no", "off"}:
            return False
        raise SimulationConfigError(
            "use_image_weighting must be a boolean or truthy/falsey string"
        )
    return bool(value)


def _build_particle_loss_config(payload: Any) -> ParticleLossConfig:
    if payload is None:
        return ParticleLossConfig()
    if not isinstance(payload, Mapping):
        raise SimulationConfigError("particle_loss must be a JSON object")

    enabled_value = payload.get("enabled")
    enabled = (
        bool(enabled_value)
        if enabled_value is not None
        else bool(DEFAULT_PARTICLE_LOSS["enabled"])
    )
    loss_radius_mm = (
        _optional_float_field(payload, "loss_radius_mm")
        if "loss_radius_mm" in payload
        else float(DEFAULT_PARTICLE_LOSS["loss_radius_mm"])
    )

    try:
        return ParticleLossConfig(
            enabled=enabled,
            loss_radius_mm=loss_radius_mm,
            conducting_wall_aperture_loss_enabled=bool(
                payload.get(
                    "conducting_wall_aperture_loss_enabled",
                    DEFAULT_PARTICLE_LOSS["conducting_wall_aperture_loss_enabled"],
                )
            ),
            initial_radial_quantile=_optional_float_field(
                payload,
                "initial_radial_quantile",
            ),
            initial_radial_multiplier=float(
                payload.get(
                    "initial_radial_multiplier",
                    DEFAULT_PARTICLE_LOSS["initial_radial_multiplier"],
                )
            ),
            initial_radial_margin_mm=float(
                payload.get(
                    "initial_radial_margin_mm",
                    DEFAULT_PARTICLE_LOSS["initial_radial_margin_mm"],
                )
            ),
        )
    except (TypeError, ValueError) as exc:
        raise SimulationConfigError(str(exc)) from exc


def _build_pseudo_grid_config(payload: Any) -> PseudoGridConfig:
    if payload is None:
        return PseudoGridConfig()
    if not isinstance(payload, Mapping):
        raise SimulationConfigError("pseudo_grid must be a JSON object")

    def _as_int(name: str, default: int) -> int:
        value = payload.get(name, default)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise SimulationConfigError(
                f"pseudo_grid.{name} must be an integer"
            ) from exc

    try:
        return PseudoGridConfig(
            enabled=bool(payload.get("enabled", DEFAULT_PSEUDO_GRID["enabled"])),
            active_rider_count=_as_int(
                "active_rider_count", DEFAULT_PSEUDO_GRID["active_rider_count"]
            ),
            active_driver_count=_as_int(
                "active_driver_count", DEFAULT_PSEUDO_GRID["active_driver_count"]
            ),
            passive_neighbor_count=_as_int(
                "passive_neighbor_count",
                DEFAULT_PSEUDO_GRID["passive_neighbor_count"],
            ),
            coverage_strategy=str(
                payload.get(
                    "coverage_strategy", DEFAULT_PSEUDO_GRID["coverage_strategy"]
                )
            ),
            coverage_space=str(
                payload.get("coverage_space", DEFAULT_PSEUDO_GRID["coverage_space"])
            ),
            pair_reuse_window=_as_int(
                "pair_reuse_window", DEFAULT_PSEUDO_GRID["pair_reuse_window"]
            ),
            source_weighting_mode=str(
                payload.get(
                    "source_weighting_mode",
                    DEFAULT_PSEUDO_GRID["source_weighting_mode"],
                )
            ),
            loss_tracking_enabled=bool(
                payload.get(
                    "loss_tracking_enabled",
                    DEFAULT_PSEUDO_GRID["loss_tracking_enabled"],
                )
            ),
            causal_history_pruning_enabled=bool(
                payload.get(
                    "causal_history_pruning_enabled",
                    DEFAULT_PSEUDO_GRID["causal_history_pruning_enabled"],
                )
            ),
            causal_history_safety_margin_steps=_as_int(
                "causal_history_safety_margin_steps",
                DEFAULT_PSEUDO_GRID["causal_history_safety_margin_steps"],
            ),
        )
    except ValueError as exc:
        raise SimulationConfigError(str(exc)) from exc


def _build_cavity_exit_config(payload: Any) -> CavityExitConfig:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise SimulationConfigError("'cavity_exit' must be an object")
    merged = dict(DEFAULT_CAVITY_EXIT)
    merged.update(payload)
    return CavityExitConfig(
        enabled=bool(merged.get("enabled", False)),
        mode=str(merged.get("mode", "first_exit")),
        cavity_length_mm=(
            None
            if merged.get("cavity_length_mm") is None
            else float(merged["cavity_length_mm"])
        ),
        residual_tail_factor=float(merged.get("residual_tail_factor", 0.0)),
        max_residual_tail_steps=int(merged.get("max_residual_tail_steps", 0)),
    )


def _build_beamline_geometry_config(payload: Any) -> BeamlineGeometryConfig:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise SimulationConfigError("'beamline_geometry' must be an object")
    merged = dict(DEFAULT_BEAMLINE_GEOMETRY)
    merged.update(payload)
    occluders_payload = merged.get("occluders", [])
    occluders = []
    for item in occluders_payload:
        if not isinstance(item, Mapping):
            raise SimulationConfigError("each occluder must be an object")
        occluders.append(
            Occluder(
                axis=tuple(item.get("axis", (0.0, 0.0, 1.0))),
                center_mm=tuple(item.get("center_mm", (0.0, 0.0, 0.0))),
                radius_mm=float(item.get("radius_mm", 1.0)),
                length_mm=float(item.get("length_mm", 1.0)),
                label=str(item.get("label", "")),
            )
        )
    return BeamlineGeometryConfig(
        enabled=bool(merged.get("enabled", False)),
        occluders=occluders,
    )


def _build_macroparticle_smearing_config(payload: Any) -> MacroparticleSmearingConfig:
    if payload is None:
        return MacroparticleSmearingConfig()
    if not isinstance(payload, Mapping):
        raise SimulationConfigError("macroparticle_smearing must be a JSON object")

    def _as_int(name: str, default: int) -> int:
        value = payload.get(name, default)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise SimulationConfigError(
                f"macroparticle_smearing.{name} must be an integer"
            ) from exc

    def _as_float(name: str, default: float) -> float:
        value = payload.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise SimulationConfigError(
                f"macroparticle_smearing.{name} must be numeric"
            ) from exc

    def _as_optional_float(name: str) -> Optional[float]:
        value = payload.get(name)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise SimulationConfigError(
                f"macroparticle_smearing.{name} must be numeric"
            ) from exc

    refresh_policy = str(
        payload.get(
            "refresh_policy",
            DEFAULT_MACROPARTICLE_SMEARING["refresh_policy"],
        )
    ).replace("-", "_")

    try:
        return MacroparticleSmearingConfig(
            enabled=bool(
                payload.get("enabled", DEFAULT_MACROPARTICLE_SMEARING["enabled"])
            ),
            mode=str(payload.get("mode", DEFAULT_MACROPARTICLE_SMEARING["mode"])),
            subcharge_count=_as_int(
                "subcharge_count",
                DEFAULT_MACROPARTICLE_SMEARING["subcharge_count"],
            ),
            sigma_multiplier=_as_float(
                "sigma_multiplier",
                DEFAULT_MACROPARTICLE_SMEARING["sigma_multiplier"],
            ),
            position_sigma_mm=_as_optional_float("position_sigma_mm"),
            longitudinal_sigma_mm=_as_optional_float("longitudinal_sigma_mm"),
            momentum_sigma_amu_mm_ns=_as_optional_float("momentum_sigma_amu_mm_ns"),
            use_position_errors=bool(
                payload.get(
                    "use_position_errors",
                    DEFAULT_MACROPARTICLE_SMEARING["use_position_errors"],
                )
            ),
            use_momentum_errors=bool(
                payload.get(
                    "use_momentum_errors",
                    DEFAULT_MACROPARTICLE_SMEARING["use_momentum_errors"],
                )
            ),
            use_centroid_errors=bool(
                payload.get(
                    "use_centroid_errors",
                    DEFAULT_MACROPARTICLE_SMEARING["use_centroid_errors"],
                )
            ),
            use_internal_cloud=bool(
                payload.get(
                    "use_internal_cloud",
                    DEFAULT_MACROPARTICLE_SMEARING["use_internal_cloud"],
                )
            ),
            apply_to_active_observers=bool(
                payload.get(
                    "apply_to_active_observers",
                    DEFAULT_MACROPARTICLE_SMEARING["apply_to_active_observers"],
                )
            ),
            apply_to_active_sources=bool(
                payload.get(
                    "apply_to_active_sources",
                    DEFAULT_MACROPARTICLE_SMEARING["apply_to_active_sources"],
                )
            ),
            apply_to_passive_sources=bool(
                payload.get(
                    "apply_to_passive_sources",
                    DEFAULT_MACROPARTICLE_SMEARING["apply_to_passive_sources"],
                )
            ),
            apply_to_passive_updates=bool(
                payload.get(
                    "apply_to_passive_updates",
                    DEFAULT_MACROPARTICLE_SMEARING["apply_to_passive_updates"],
                )
            ),
            seed=_as_int("seed", DEFAULT_MACROPARTICLE_SMEARING["seed"]),
            refresh_policy=refresh_policy,
        )
    except ValueError as exc:
        raise SimulationConfigError(str(exc)) from exc


def _build_driver_train_config(payload: Any) -> DriverTrainConfig:
    if payload is None:
        return DriverTrainConfig()
    if not isinstance(payload, Mapping):
        raise SimulationConfigError("driver_train must be a JSON object")

    def _as_int(name: str, default: int) -> int:
        value = payload.get(name, default)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise SimulationConfigError(
                f"driver_train.{name} must be an integer"
            ) from exc

    def _as_float(name: str, default: float) -> float:
        value = payload.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise SimulationConfigError(f"driver_train.{name} must be numeric") from exc

    offsets_raw = payload.get("z_offsets_mm", DEFAULT_DRIVER_TRAIN["z_offsets_mm"])
    if offsets_raw in (None, ""):
        offsets: tuple[float, ...] = ()
    elif isinstance(offsets_raw, (list, tuple)):
        try:
            offsets = tuple(float(value) for value in offsets_raw)
        except (TypeError, ValueError) as exc:
            raise SimulationConfigError(
                "driver_train.z_offsets_mm must contain numeric values"
            ) from exc
    else:
        raise SimulationConfigError("driver_train.z_offsets_mm must be a list")

    try:
        return DriverTrainConfig(
            enabled=bool(payload.get("enabled", DEFAULT_DRIVER_TRAIN["enabled"])),
            bunch_count=_as_int("bunch_count", DEFAULT_DRIVER_TRAIN["bunch_count"]),
            z_spacing_mm=_as_float(
                "z_spacing_mm", DEFAULT_DRIVER_TRAIN["z_spacing_mm"]
            ),
            z_offsets_mm=offsets,
            prehistory_steps=_as_int(
                "prehistory_steps", DEFAULT_DRIVER_TRAIN["prehistory_steps"]
            ),
            preserve_prehistory_in_output=bool(
                payload.get(
                    "preserve_prehistory_in_output",
                    DEFAULT_DRIVER_TRAIN["preserve_prehistory_in_output"],
                )
            ),
        )
    except ValueError as exc:
        raise SimulationConfigError(str(exc)) from exc


def _optional_float_field(payload: Mapping[str, Any], name: str) -> Optional[float]:
    value = payload.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SimulationConfigError(f"external_field.{name} must be numeric") from exc


def _parse_field_vector(
    payload: Mapping[str, Any],
    name: str,
) -> Optional[Tuple[float, float, float]]:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SimulationConfigError(f"external_field.{name} must be a 3-vector")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise SimulationConfigError(
            f"external_field.{name} must contain numeric values"
        ) from exc


def _build_space_charge_config(
    payload: Mapping[str, Any],
) -> Optional[SpaceChargeConfig]:
    enabled = bool(payload.get("space_charge_enabled", False))
    if not enabled:
        return None

    try:
        softening_mm = float(payload.get("space_charge_softening_mm", 0.0))
        bunch_sigma_mm = float(payload.get("space_charge_bunch_sigma_mm", 0.01))
    except (TypeError, ValueError) as exc:
        raise SimulationConfigError(
            "space_charge_softening_mm and space_charge_bunch_sigma_mm must be numeric"
        ) from exc
    min_retarded_steps_raw = payload.get("space_charge_min_retarded_steps")
    try:
        min_retarded_steps = (
            int(min_retarded_steps_raw) if min_retarded_steps_raw is not None else None
        )
    except (TypeError, ValueError) as exc:
        raise SimulationConfigError(
            "space_charge_min_retarded_steps must be an integer or null"
        ) from exc

    return SpaceChargeConfig(
        enabled=True,
        retarded=bool(payload.get("space_charge_retarded", True)),
        softening_mm=softening_mm,
        bunch_sigma_mm=bunch_sigma_mm,
        min_retarded_steps=min_retarded_steps,
    )


def _build_self_consistency_config(
    payload: Mapping[str, Any],
) -> Optional[SelfConsistencyConfig]:
    chrono_interpolate = bool(
        payload.get(
            "chrono_interpolate",
            payload.get("self_consistency_chrono_interpolate", False),
        )
    )
    chrono_high_precision = bool(
        payload.get(
            "chrono_high_precision",
            payload.get("self_consistency_chrono_high_precision", False),
        )
    )
    chrono_adaptive_tolerance = bool(
        payload.get(
            "chrono_adaptive_tolerance",
            payload.get("self_consistency_chrono_adaptive_tolerance", False),
        )
    )
    sc_enabled = bool(payload.get("self_consistency_enabled", False))
    if not sc_enabled and not (
        chrono_interpolate or chrono_high_precision or chrono_adaptive_tolerance
    ):
        return None

    method_name = str(
        payload.get("self_consistency_gamma_reconciliation_method", "DISABLED")
    ).upper()
    try:
        gamma_method = GammaReconciliationMethod[method_name]
    except KeyError:
        raise SimulationConfigError(
            f"Unknown gamma reconciliation method: {method_name!r}"
        ) from None

    try:
        return SelfConsistencyConfig(
            enabled=sc_enabled,
            convergence_mode=str(
                payload.get("self_consistency_convergence_mode", "fixed_geometry")
            ),
            target_ms_tolerance=float(
                payload.get("self_consistency_target_ms_tolerance", 1e-6)
            ),
            mass_shell_tolerance=float(
                payload.get("self_consistency_mass_shell_tolerance", 1e-2)
            ),
            mass_shell_relaxation=float(
                payload.get("self_consistency_mass_shell_relaxation", 0.7)
            ),
            max_iterations=int(payload.get("self_consistency_max_iterations", 2)),
            verbosity=int(payload.get("self_consistency_verbosity", 0)),
            chrono_interpolate=chrono_interpolate,
            chrono_tolerance=float(
                payload.get(
                    "chrono_tolerance",
                    payload.get("self_consistency_chrono_tolerance", 1e-3),
                )
            ),
            chrono_matching_mode=str(
                payload.get(
                    "chrono_matching_mode",
                    payload.get("self_consistency_chrono_matching_mode", "FAST"),
                )
            ),
            chrono_high_precision=chrono_high_precision,
            chrono_adaptive_tolerance=chrono_adaptive_tolerance,
            gamma_reconciliation_method=gamma_method,
            gamma_reconciliation_fixed_weight=float(
                payload.get("self_consistency_gamma_reconciliation_fixed_weight", 0.5)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise SimulationConfigError(
            "Invalid self-consistency or chrono-matching option"
        ) from exc


def _build_adaptive_timestep_config(
    payload: Mapping[str, Any],
) -> Optional[AdaptiveTimestepConfig]:
    adaptive_payload = payload.get("adaptive_timestep")
    if adaptive_payload is None:
        return None
    if not isinstance(adaptive_payload, Mapping):
        raise SimulationConfigError("adaptive_timestep must be a JSON object")
    if not bool(adaptive_payload.get("enabled", DEFAULT_ADAPTIVE_TIMESTEP["enabled"])):
        return None

    try:
        return AdaptiveTimestepConfig(
            enabled=True,
            energy_jump_threshold=float(
                adaptive_payload.get(
                    "energy_jump_threshold",
                    DEFAULT_ADAPTIVE_TIMESTEP["energy_jump_threshold"],
                )
            ),
            timestep_reduction_factor=int(
                adaptive_payload.get(
                    "timestep_reduction_factor",
                    DEFAULT_ADAPTIVE_TIMESTEP["timestep_reduction_factor"],
                )
            ),
            min_timestep_factor=float(
                adaptive_payload.get(
                    "min_timestep_factor",
                    DEFAULT_ADAPTIVE_TIMESTEP["min_timestep_factor"],
                )
            ),
            cooldown_steps=int(
                adaptive_payload.get(
                    "cooldown_steps", DEFAULT_ADAPTIVE_TIMESTEP["cooldown_steps"]
                )
            ),
            probe_threshold=float(
                adaptive_payload.get(
                    "probe_threshold", DEFAULT_ADAPTIVE_TIMESTEP["probe_threshold"]
                )
            ),
            max_probe_steps=int(
                adaptive_payload.get(
                    "max_probe_steps", DEFAULT_ADAPTIVE_TIMESTEP["max_probe_steps"]
                )
            ),
            bunch_proximity_enabled=bool(
                adaptive_payload.get(
                    "bunch_proximity_enabled",
                    DEFAULT_ADAPTIVE_TIMESTEP["bunch_proximity_enabled"],
                )
            ),
            bunch_proximity_sigma_mm=float(
                adaptive_payload.get(
                    "bunch_proximity_sigma_mm",
                    DEFAULT_ADAPTIVE_TIMESTEP["bunch_proximity_sigma_mm"],
                )
            ),
            bunch_proximity_n_sigma=float(
                adaptive_payload.get(
                    "bunch_proximity_n_sigma",
                    DEFAULT_ADAPTIVE_TIMESTEP["bunch_proximity_n_sigma"],
                )
            ),
            bunch_proximity_reduction_factor=float(
                adaptive_payload.get(
                    "bunch_proximity_reduction_factor",
                    DEFAULT_ADAPTIVE_TIMESTEP["bunch_proximity_reduction_factor"],
                )
            ),
            bunch_proximity_transition_n_sigma=float(
                adaptive_payload.get(
                    "bunch_proximity_transition_n_sigma",
                    DEFAULT_ADAPTIVE_TIMESTEP["bunch_proximity_transition_n_sigma"],
                )
            ),
            debug=bool(
                adaptive_payload.get("debug", DEFAULT_ADAPTIVE_TIMESTEP["debug"])
            ),
        )
    except (TypeError, ValueError) as exc:
        raise SimulationConfigError(str(exc)) from exc


def _build_external_field_config(payload: Any) -> Optional[ExternalFieldConfig]:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise SimulationConfigError("external_field must be a JSON object")
    if not bool(payload.get("enabled", True)):
        return None

    electric_native = _parse_field_vector(payload, "electric_field_native") or (
        0.0,
        0.0,
        0.0,
    )
    electric_si = _parse_field_vector(payload, "electric_field_v_per_m")
    if electric_si is not None:
        from core.external_fields import electric_field_v_per_m_to_native

        electric_native = tuple(
            electric_field_v_per_m_to_native(component) for component in electric_si
        )

    magnetic_native = _parse_field_vector(payload, "magnetic_field_native") or (
        0.0,
        0.0,
        0.0,
    )

    return ExternalFieldConfig(
        enabled=True,
        electric_field_native=electric_native,
        magnetic_field_native=magnetic_native,
        x_min=_optional_float_field(payload, "x_min"),
        x_max=_optional_float_field(payload, "x_max"),
        y_min=_optional_float_field(payload, "y_min"),
        y_max=_optional_float_field(payload, "y_max"),
        z_min=_optional_float_field(payload, "z_min"),
        z_max=_optional_float_field(payload, "z_max"),
        t_min=_optional_float_field(payload, "t_min"),
        t_max=_optional_float_field(payload, "t_max"),
    )


def _build_particle_state(payload: Mapping[str, Any]) -> ParticleState:
    missing = [field for field in REQUIRED_PARTICLE_FIELDS if field not in payload]
    if missing:
        raise SimulationConfigError(
            "Particle configuration is missing required fields: " + ", ".join(missing)
        )

    if "momentum_axis" in payload:
        state = _build_particle_state_3d(payload)
    else:
        try:
            state, _rest_energy = create_bunch_from_energy(**payload)
        except TypeError as exc:
            raise SimulationConfigError(
                f"Particle configuration includes unsupported options: {exc}"
            ) from exc

    return state


def _build_particle_state_3d(payload: Mapping[str, Any]) -> ParticleState:
    """Build a 3D-oriented bunch when ``momentum_axis`` is present."""
    from core.particle_initialization import create_particle_state_3d

    try:
        axis = tuple(float(v) for v in payload["momentum_axis"])
    except (TypeError, ValueError) as exc:
        raise SimulationConfigError(
            f"momentum_axis must be a list of 3 numbers: {exc}"
        ) from exc

    starting_position = payload.get(
        "starting_position_mm", [0.0, 0.0, 0.0]
    )
    try:
        starting_position = tuple(float(v) for v in starting_position)
    except (TypeError, ValueError) as exc:
        raise SimulationConfigError(
            f"starting_position_mm must be a list of 3 numbers: {exc}"
        ) from exc

    transverse_axes = payload.get("transverse_axes")
    if transverse_axes is not None:
        try:
            transverse_axes = tuple(
                tuple(float(v) for v in ax) for ax in transverse_axes
            )
        except (TypeError, ValueError) as exc:
            raise SimulationConfigError(
                f"transverse_axes must be a list of axis lists: {exc}"
            ) from exc

    try:
        state, _rest_energy = create_particle_state_3d(
            starting_position_mm=starting_position,
            momentum_axis=axis,
            kinetic_energy_mev=float(payload["kinetic_energy_mev"]),
            stripped_ions=float(payload.get("stripped_ions", 1.0)),
            particle_mass_amu=float(payload["mass_amu"]),
            particle_count=int(payload.get("particle_count", 1)),
            charge_sign=float(payload["charge_sign"]),
            transverse_distance_mm=float(payload.get("transverse_distance_mm", 0.0)),
            transverse_momentum=float(payload.get("transverse_momentum", 0.0)),
            longitudinal_span_mm=float(payload.get("longitudinal_span_mm", 0.0)),
            transverse_axes=transverse_axes,
            charge_multiplier=float(payload.get("charge_multiplier", 1.0)),
        )
    except (TypeError, ValueError) as exc:
        raise SimulationConfigError(
            f"3D particle configuration error: {exc}"
        ) from exc

    return state


# ---------------------------------------------------------------------------
# Simulation execution
# ---------------------------------------------------------------------------


def _resolve_auto_duration(request: SimulationRequest) -> tuple[int, float]:
    steps = int(request.config.steps)
    h_step = float(request.config.time_step)

    if not request.auto_duration_enabled:
        return steps, h_step

    rider_pz = float(np.asarray(request.rider["Pz"]).mean())
    rider_m = float(np.asarray(request.rider["m"]).mean())
    rider_gamma = float(np.asarray(request.rider["gamma"]).mean())
    rider_beta_z = abs(rider_pz) / (rider_gamma * rider_m * C_MMNS)

    driver_beta_z = 0.0
    if request.driver is not None:
        driver_pz = float(np.asarray(request.driver["Pz"]).mean())
        driver_m = float(np.asarray(request.driver["m"]).mean())
        driver_gamma = float(np.asarray(request.driver["gamma"]).mean())
        driver_beta_z = abs(driver_pz) / (driver_gamma * driver_m * C_MMNS)

    closing_speed = (rider_beta_z + driver_beta_z) * C_MMNS
    rider_z0 = float(np.asarray(request.rider["z"]).mean())
    driver_z0 = (
        float(np.asarray(request.driver["z"]).mean())
        if request.driver is not None
        else 0.0
    )
    separation = abs(driver_z0 - rider_z0)

    if closing_speed <= 0.0 or separation <= 0.0:
        return steps, h_step

    h_step = separation / (closing_speed * request.auto_duration_crossing_steps)
    steps = max(
        10,
        int(
            math.ceil(
                request.auto_duration_crossing_steps * request.auto_duration_post_factor
            )
        ),
    )
    return steps, h_step


def run_simulation(request: SimulationRequest) -> tuple:
    steps, h_step = _resolve_auto_duration(request)
    return retarded_integrator(
        steps=steps,
        h_step=h_step,
        wall_z=request.config.wall_position,
        aperture_radius=request.config.aperture_radius,
        sim_type=request.config.simulation_type,
        init_rider=request.rider,
        init_driver=request.driver,
        mean=request.config.bunch_mean,
        cav_spacing=request.config.cavity_spacing,
        z_cutoff=request.config.z_cutoff,
        z_cutoff_mode=request.config.z_cutoff_mode,
        chrono_mode=request.config.chrono_mode,
        startup_mode=request.config.startup_mode,
        image_subcharge_count=request.config.image_subcharge_count,
        use_conducting_image_weighting=request.config.use_image_weighting,
        radiation_reaction_mode=request.config.radiation_reaction_mode,
        adaptive_timestep=request.adaptive_timestep,
        self_consistency=request.self_consistency,
        space_charge=request.space_charge,
        external_field=request.external_field,
        pseudo_grid=request.config.pseudo_grid,
        driver_train=request.config.driver_train,
        cavity_exit=request.config.cavity_exit,
        particle_loss=request.config.particle_loss,
        macroparticle_smearing=request.config.macroparticle_smearing,
        beamline_geometry=request.config.beamline_geometry,
    )


def summarise_trajectory(trajectory: Trajectory) -> Dict[str, Any]:
    initial = trajectory[0]
    final = trajectory[-1]

    def _mean(value: np.ndarray) -> float:
        return float(np.mean(np.asarray(value, dtype=float)))

    def _max_abs(value: np.ndarray) -> float:
        return float(np.max(np.abs(np.asarray(value, dtype=float))))

    initial_z = _mean(initial.get("z", np.array([0.0])))
    final_z = _mean(final.get("z", np.array([0.0])))
    initial_gamma = _mean(initial.get("gamma", np.array([1.0])))
    final_gamma = _mean(final.get("gamma", np.array([1.0])))
    summary_row = summarize_result_row(
        {
            "parameters": {"start_z": initial_z},
            "metrics": {
                "rider_gamma_initial": initial_gamma,
                "rider_gamma_final": final_gamma,
            },
            "_distance_info": {
                "z_start": initial_z,
                "z_end": final_z,
            },
        }
    )

    return {
        "steps_completed": len(trajectory),
        "initial_time_ns": _mean(initial.get("t", np.array([0.0]))),
        "final_time_ns": _mean(final.get("t", np.array([0.0]))),
        "initial_z_mm": initial_z,
        "final_z_mm": final_z,
        "traveled_distance_mm": summary_row["traveled"],
        "initial_gamma_mean": summary_row["gamma_initial"],
        "final_gamma_mean": summary_row["gamma_final"],
        "delta_gamma_mean": summary_row["gamma_final"] - summary_row["gamma_initial"],
        "max_absolute_velocity": _max_abs(final.get("bz", np.array([0.0]))),
    }


def print_summary(summary: Mapping[str, Any]) -> None:
    lines = ["LW Integrator simulation summary:"]
    for key in (
        "steps_completed",
        "initial_time_ns",
        "final_time_ns",
        "initial_z_mm",
        "final_z_mm",
        "traveled_distance_mm",
        "initial_gamma_mean",
        "final_gamma_mean",
        "delta_gamma_mean",
        "max_absolute_velocity",
    ):
        if key in summary:
            lines.append(
                f"  {key.replace('_', ' ').title()}: {_format_value(summary[key])}"
            )
    print("\n".join(lines))


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def build_report(
    trajectory: Trajectory, driver: Optional[Trajectory] = None
) -> Dict[str, Any]:
    """Build the CLI report payload for rider and optional driver trajectories."""
    report = dict(summarise_trajectory(trajectory))
    if driver is not None:
        report["driver_summary"] = summarise_trajectory(driver)
    return report


def _load_results_report(path: Path) -> Dict[str, Any]:
    """Load a saved results JSON file and build a normalized summary report."""
    payload = _load_config(path)
    parsed = parse_results_payload(
        payload, m_particle_amu=ELECTRON_MASS_AMU, amu_to_mev=931.494
    )
    report = summarize_saved_results(parsed)
    report["source"] = str(path)
    return report


def _print_results_report(report: Mapping[str, Any]) -> None:
    """Print a human-readable summary for a saved results file."""
    lines = ["LW Integrator saved-results summary:"]
    for key in (
        "result_type",
        "source",
        "config_name",
        "run_count",
        "trajectory_count",
        "optimization_method",
        "objective",
        "evaluation_count",
        "finite_evaluation_count",
        "successful_evaluation_count",
        "halted_evaluation_count",
        "failed_evaluation_count",
        "top_result_count",
        "success",
        "best_run_number",
        "best_delta_e_mev",
        "best_energy_gev",
        "best_aperture_mm",
        "best_value",
    ):
        if key in report:
            lines.append(
                f"  {key.replace('_', ' ').title()}: {_format_value(report[key])}"
            )

    top_results = report.get("top_results", [])
    if top_results:
        lines.append("  Top Results:")
        for result in top_results:
            parts = [f"rank={result.get('rank')}"]
            if result.get("metric_value") is not None:
                parts.append(f"metric={_format_value(result['metric_value'])}")
            if result.get("delta_e_mev") is not None:
                parts.append(f"delta_e_mev={_format_value(result['delta_e_mev'])}")
            if result.get("percent_energy_gain") is not None:
                parts.append(
                    f"percent_gain={_format_value(result['percent_energy_gain'])}"
                )
            lines.append("    " + ", ".join(parts))
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.results_file is not None:
        try:
            report = _load_results_report(args.results_file)
        except (SimulationConfigError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

        if not args.quiet:
            _print_results_report(report)
        return 0

    # Check if this is a sweep configuration
    if args.sweep_config is not None:
        return run_sweep(args)

    # Regular single simulation
    try:
        request = build_request(args)
    except SimulationConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    trajectory, driver, *_soa_out = run_simulation(request)
    report = build_report(trajectory, driver)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.quiet:
        print_summary(report)
        if driver is not None:
            print(f"Driver trajectory generated with {len(driver)} integration steps.")

    return 0


def run_sweep(args: argparse.Namespace) -> int:
    """Execute a parameter sweep from a sweep configuration file."""
    from lw_integrator.sweep_runner import run_sweep_from_config

    if not args.sweep_config.exists():
        print(
            f"Error: Sweep config file not found: {args.sweep_config}", file=sys.stderr
        )
        return 2

    quiet = getattr(args, "quiet", False)
    verbosity_overrides = _build_sweep_verbosity_overrides(args)

    # Run the sweep
    try:
        success = run_sweep_from_config(
            config_path=args.sweep_config,
            output_dir=None,
            verbose=not quiet,
            verbosity_overrides=verbosity_overrides,
            workers=getattr(args, "workers", None),
        )
        return 0 if success else 1
    except Exception as exc:
        print(f"Error running sweep: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2


def _build_sweep_verbosity_overrides(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Collect sweep verbosity-related overrides from CLI arguments."""
    overrides: Dict[str, Any] = {}
    if getattr(args, "log_verbosity", None) is not None:
        overrides["log_verbosity"] = args.log_verbosity
    if getattr(args, "sc_verbosity", None) is not None:
        overrides["self_consistency_verbosity"] = args.sc_verbosity
    if getattr(args, "adaptive_debug", None) is not None:
        overrides["adaptive_timestep_debug"] = args.adaptive_debug
    for key in (
        "chrono_interpolate",
        "chrono_tolerance",
        "chrono_high_precision",
        "chrono_adaptive_tolerance",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return overrides


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
