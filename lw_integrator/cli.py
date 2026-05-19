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
from core.integration_runner import retarded_integrator
from core.types import (
    ChronoMatchingMode,
    ExternalFieldConfig,
    IntegratorConfig,
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
        "--chrono-mode",
        choices=("averaged", "fast"),
        help=(
            "Retardation sampling strategy. 'fast' is the maintained default; "
            "'averaged' is retained for diagnostics with approximate back-history."
        ),
    )
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
    result["pseudo_grid"] = dict(DEFAULT_PSEUDO_GRID)
    for key in DEFAULT_SIMULATION:
        if key in file_payload:
            result[key] = file_payload[key]
    file_pseudo_grid = file_payload.get("pseudo_grid")
    if isinstance(file_pseudo_grid, Mapping):
        result["pseudo_grid"].update(file_pseudo_grid)
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
    )
    for key in passthrough_keys:
        if key in file_payload:
            result[key] = file_payload[key]

    override_keys = (
        "steps",
        "time_step",
        "simulation_type",
        "wall_position",
        "aperture_radius",
        "bunch_mean",
        "cavity_spacing",
        "z_cutoff",
        "chrono_mode",
        "startup_mode",
        "radiation_reaction_mode",
        "image_subcharge_count",
        "use_image_weighting",
    )

    for key in override_keys:
        if getattr(args, key, None) is not None:
            result[key] = getattr(args, key)

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

    pseudo_grid = _build_pseudo_grid_config(payload.get("pseudo_grid"))

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
        image_subcharge_count=image_subcharge_count,
        use_image_weighting=use_image_weighting,
        radiation_reaction_mode=str(
            payload.get(
                "radiation_reaction_mode",
                DEFAULT_SIMULATION["radiation_reaction_mode"],
            )
        ),
        pseudo_grid=pseudo_grid,
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

    try:
        state, _rest_energy = create_bunch_from_energy(**payload)
    except TypeError as exc:
        raise SimulationConfigError(
            f"Particle configuration includes unsupported options: {exc}"
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
        chrono_mode=request.config.chrono_mode,
        startup_mode=request.config.startup_mode,
        image_subcharge_count=request.config.image_subcharge_count,
        use_conducting_image_weighting=request.config.use_image_weighting,
        radiation_reaction_mode=request.config.radiation_reaction_mode,
        space_charge=request.space_charge,
        external_field=request.external_field,
        pseudo_grid=request.config.pseudo_grid,
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
    return overrides


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
