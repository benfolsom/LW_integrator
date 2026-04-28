"""Pure helpers for sweep result records and compact run logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from optimization.simulation_type_helpers import is_bunch_to_bunch


@dataclass(frozen=True)
class SweepAttemptClassification:
    """Useful-metric classification for one sweep retry attempt."""

    succeeded: bool
    error: Exception | None
    log_lines: list[str]


def simulation_type_name(simulation_type: Any) -> str:
    """Return a stable serialized name for enum-backed or string-backed modes."""
    return str(getattr(simulation_type, "name", simulation_type))


def build_sweep_run_data(
    *,
    run_number: int,
    params_dict: Mapping[str, Any],
    simulation_type: Any,
    aperture: float,
    energy: float,
    start_z: float,
    transv_offset: float,
    offset_frac: float,
    timestep: float,
    steps: int,
    retry_attempts: int,
    default_wall_z: float,
    rider_m_particle: float,
    rider_charge_sign: float,
    rider_pcount: int,
    rider_transv_mom: float,
    rider_transv_dist: float,
    macroparticle_charge_multiplier: float,
    macroparticle_sigma_multiplier: float,
    metrics: Mapping[str, Any],
    driver_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the persisted sweep run record."""
    parameters = {
        "aperture_radius": aperture,
        "particle_energy_gev": energy,
        "start_z": start_z,
        "transverse_offset": transv_offset,
        "transverse_offset_fraction": offset_frac,
        "timestep": timestep,
        "steps": steps,
        "retry_attempts": retry_attempts,
        "wall_z": params_dict.get("wall_z", default_wall_z),
        "rider_m_particle": rider_m_particle,
        "rider_charge_sign": rider_charge_sign,
        "rider_pcount": int(rider_pcount),
        "rider_transv_mom": rider_transv_mom,
        "rider_transv_dist": rider_transv_dist,
        "macroparticle_charge_multiplier": macroparticle_charge_multiplier,
        "macroparticle_sigma_multiplier": macroparticle_sigma_multiplier,
        "simulation_type": simulation_type_name(simulation_type),
    }
    if driver_params is not None:
        parameters.update(
            {f"driver_{key}": value for key, value in driver_params.items()}
        )

    return {
        "run_number": run_number,
        "parameters": parameters,
        "metrics": dict(metrics),
    }


def build_truncated_sweep_log_params(
    *,
    param_grids: Mapping[str, list],
    params_dict: Mapping[str, Any],
    simulation_type: Any,
    aperture: float,
    energy: float,
    wall_z: float,
) -> dict[str, Any]:
    """Return compact parameter values for truncated sweep logging."""
    log_params = {
        param_name: params_dict[param_name]
        for param_name, grid in param_grids.items()
        if len(grid) > 1 and param_name in params_dict
    }
    if log_params:
        return log_params

    if is_bunch_to_bunch(simulation_type):
        if "initial_energy_gev" in params_dict:
            log_params["initial_energy_gev"] = params_dict["initial_energy_gev"]
        if "driver_starting_distance" in params_dict:
            log_params["driver_starting_distance"] = params_dict[
                "driver_starting_distance"
            ]
    else:
        log_params["aperture"] = aperture
        log_params["energy"] = energy

    if "wall_z" in params_dict:
        log_params["wall_z"] = params_dict["wall_z"]
    else:
        log_params["wall_z"] = wall_z

    return log_params


def extract_actual_distance(result: Mapping[str, Any]) -> float:
    """Extract traveled distance from a sweep integration result, if present."""
    if "_distance_info" in result:
        dist_info = result["_distance_info"]
        return abs(dist_info["z_end"] - dist_info["z_start"])

    trajectory = result.get("trajectory")
    if not trajectory:
        return 0.0

    z_values = trajectory.get("z", [])
    if len(z_values) <= 1:
        return 0.0

    z_start = float(np.asarray(z_values[0]).flat[0])
    z_end = float(np.asarray(z_values[-1]).flat[0])
    return abs(z_end - z_start)


def classify_sweep_attempt_result(
    attempt_result: Mapping[str, Any],
    *,
    run_num: int,
    retry_attempt: int,
    include_debug_logs: bool = False,
) -> SweepAttemptClassification:
    """Classify whether an integration attempt produced useful sweep metrics."""
    is_halted = attempt_result.get("halted_early", False)
    metrics = attempt_result.get("metrics", {})
    log_lines: list[str] = []

    if include_debug_logs:
        log_lines.append(
            f"  [DEBUG] Run {run_num} attempt {retry_attempt}: "
            f"is_halted={is_halted}, has_metrics={bool(metrics)}"
        )
        if metrics:
            log_lines.extend(
                [
                    (
                        "    max_percent_energy_gain="
                        f"{metrics.get('max_percent_energy_gain')}"
                    ),
                    f"    rider_gamma_final={metrics.get('rider_gamma_final')}",
                    f"    rider_delta_e_mev={metrics.get('rider_delta_e_mev')}",
                ]
            )

    has_useful_metrics = (
        not is_halted
        and bool(metrics)
        and (
            metrics.get("max_percent_energy_gain") is not None
            or (
                metrics.get("rider_gamma_final") is not None
                and metrics.get("rider_gamma_final") > 0
            )
            or metrics.get("rider_delta_e_mev") is not None
        )
    )
    if has_useful_metrics:
        return SweepAttemptClassification(
            succeeded=True,
            error=None,
            log_lines=log_lines,
        )

    halt_reason = attempt_result.get("halt_reason", "unknown")
    if include_debug_logs:
        log_lines.append(
            f"  [FAILED] Run {run_num} attempt {retry_attempt}: "
            f"halted={is_halted}, has_metrics={bool(metrics)}, "
            "has_useful_metrics=False"
        )

    return SweepAttemptClassification(
        succeeded=False,
        error=Exception(f"Run failed: halted_early={is_halted}, reason={halt_reason}"),
        log_lines=log_lines,
    )
