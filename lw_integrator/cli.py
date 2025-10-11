"""Command-line interface for running LW Integrator simulations.

The ``lw-simulate`` console script and ``python -m lw_integrator`` entry point
both call :func:`main`.  Users can either rely on the built-in default scenario
(a 35 MeV electron approaching a conducting aperture) or provide a JSON
configuration that customises the simulation parameters and particle bunches.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from core.constants import ELECTRON_MASS_AMU
from core.integration_runner import retarded_integrator
from core.types import (
    ChronoMatchingMode,
    IntegratorConfig,
    ParticleState,
    SimulationType,
    StartupMode,
    Trajectory,
)
from input_output.bunch_initialization import create_bunch_from_energy

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
    "chrono_mode": "averaged",
    "startup_mode": "cold-start",
}

DEFAULT_RIDER: Dict[str, Any] = {
    "kinetic_energy_mev": 35.0,
    "mass_amu": ELECTRON_MASS_AMU,
    "charge_sign": -1.0,
    "position_z": -300.0,
    "particle_count": 1,
    "transverse_radius": 0.0,
    "transverse_momentum": 0.0,
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
            "configuration file for advanced scenarios."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON file describing the simulation parameters and bunches.",
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
        help="Initial bunch separation parameter (legacy compatibility).",
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
            "Retardation sampling strategy: 'averaged' blends R/c and 2R/c, "
            "'fast' reproduces legacy single-sample behaviour."
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
        "--driver-from-rider",
        action="store_true",
        help=(
            "For bunch-to-bunch simulations, clone the rider bunch to use as the "
            "driver when no driver configuration is supplied."
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

    return SimulationRequest(config=config, rider=rider_state, driver=driver_state)


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
    for key in DEFAULT_SIMULATION:
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
    )

    for key in override_keys:
        if getattr(args, key, None) is not None:
            result[key] = getattr(args, key)

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

    chrono_mode_raw = payload.get("chrono_mode", ChronoMatchingMode.AVERAGED)
    if isinstance(chrono_mode_raw, str):
        key = chrono_mode_raw.strip().lower()
        if key in {"fast", "legacy"}:
            chrono_mode = ChronoMatchingMode.FAST
        elif key in {"averaged", "average", "blended"}:
            chrono_mode = ChronoMatchingMode.AVERAGED
        else:  # pragma: no cover - defensive parsing
            raise SimulationConfigError(
                f"Unknown chrono_mode value: {chrono_mode_raw!r}. Expected 'fast' or 'averaged'."
            )
    elif isinstance(chrono_mode_raw, ChronoMatchingMode):
        chrono_mode = chrono_mode_raw
    else:  # pragma: no cover - defensive parsing
        raise SimulationConfigError(
            "chrono_mode must be a string or ChronoMatchingMode instance"
        )

    startup_mode_raw = payload.get("startup_mode", StartupMode.COLD_START)
    if isinstance(startup_mode_raw, str):
        key = startup_mode_raw.strip().lower()
        if key in STARTUP_MODE_ALIASES:
            startup_mode = STARTUP_MODE_ALIASES[key]
        else:  # pragma: no cover - defensive parsing
            raise SimulationConfigError(
                f"Unknown startup_mode value: {startup_mode_raw!r}. Expected 'cold-start' or 'approximate-back-history'."
            )
    elif isinstance(startup_mode_raw, StartupMode):
        startup_mode = startup_mode_raw
    else:  # pragma: no cover - defensive parsing
        raise SimulationConfigError(
            "startup_mode must be a string or StartupMode instance"
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
    )


def _parse_simulation_type(value: Any) -> SimulationType:
    if isinstance(value, SimulationType):
        return value
    if isinstance(value, int):  # support legacy integer flags
        try:
            return SimulationType(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise SimulationConfigError(
                f"Unknown simulation type integer: {value}"
            ) from exc
    if isinstance(value, str):
        key = value.strip().lower()
        if key in SIMULATION_TYPE_ALIASES:
            return SIMULATION_TYPE_ALIASES[key]
    raise SimulationConfigError(f"Unknown simulation type: {value!r}")


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


def run_simulation(request: SimulationRequest) -> Tuple[Trajectory, Trajectory]:
    return retarded_integrator(
        steps=request.config.steps,
        h_step=request.config.time_step,
        wall_z=request.config.wall_position,
        aperture_radius=request.config.aperture_radius,
        sim_type=request.config.simulation_type,
        init_rider=request.rider,
        init_driver=request.driver,
        mean=request.config.bunch_mean,
        cav_spacing=request.config.cavity_spacing,
        z_cutoff=request.config.z_cutoff,
        chrono_mode=request.config.chrono_mode,
    )


def summarise_trajectory(trajectory: Trajectory) -> Dict[str, Any]:
    initial = trajectory[0]
    final = trajectory[-1]

    def _mean(value: np.ndarray) -> float:
        return float(np.mean(np.asarray(value, dtype=float)))

    def _max_abs(value: np.ndarray) -> float:
        return float(np.max(np.abs(np.asarray(value, dtype=float))))

    return {
        "steps_completed": len(trajectory),
        "initial_time_ns": _mean(initial.get("t", np.array([0.0]))),
        "final_time_ns": _mean(final.get("t", np.array([0.0]))),
        "initial_z_mm": _mean(initial.get("z", np.array([0.0]))),
        "final_z_mm": _mean(final.get("z", np.array([0.0]))),
        "initial_gamma_mean": _mean(initial.get("gamma", np.array([1.0]))),
        "final_gamma_mean": _mean(final.get("gamma", np.array([1.0]))),
        "delta_gamma_mean": _mean(final.get("gamma", np.array([1.0])))
        - _mean(initial.get("gamma", np.array([1.0]))),
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        request = build_request(args)
    except SimulationConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    trajectory, driver = run_simulation(request)
    summary = summarise_trajectory(trajectory)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.quiet:
        print_summary(summary)
        if driver is not None:
            print(
                "Driver trajectory generated with" f" {len(driver)} integration steps."
            )

    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
