"""Pure helpers for sweep result records and compact run logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from optimization.simulation_type_helpers import is_bunch_to_bunch


@dataclass(frozen=True)
class SweepAttemptClassification:
    """Useful-metric classification for one sweep retry attempt."""

    succeeded: bool
    error: Exception | None
    log_lines: list[str]


@dataclass(frozen=True)
class SweepMetricSummary:
    """Common metric values used by sweep result logging."""

    delta_e: float
    delta_gamma: float
    gamma_initial: float
    gamma_final: float


@dataclass(frozen=True)
class SuccessfulSweepRunLog:
    """Formatted log output for one successful sweep run."""

    optimization_lines: list[str]
    detail_lines: list[str]
    compact_line: str
    metrics: SweepMetricSummary


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


def build_sweep_results_payload(
    *,
    config: Any,
    param_grids: Mapping[str, Sequence[Any]],
    total_runs: int,
    successful: int,
    failed: int,
    elapsed_time_seconds: float,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the persisted completed-sweep results payload."""
    return {
        "config": {
            "simulation_type": simulation_type_name(config.simulation_type),
            "aperture_range": list(config.aperture_range),
            "aperture_points": config.aperture_points,
            "energy_range": list(config.energy_range),
            "energy_points": config.energy_points,
            "param_grids": {key: values for key, values in param_grids.items()},
        },
        "total_runs": total_runs,
        "successful": successful,
        "failed": failed,
        "elapsed_time_seconds": elapsed_time_seconds,
        "results": list(results),
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


def extract_sweep_metric_summary(result: Mapping[str, Any]) -> SweepMetricSummary:
    """Extract common sweep result metrics with historical zero defaults."""
    metrics = result.get("metrics", {})
    return SweepMetricSummary(
        delta_e=metrics.get("rider_delta_e_mev", 0.0),
        delta_gamma=metrics.get("rider_delta_gamma", 0.0),
        gamma_initial=metrics.get("rider_gamma_initial", 0.0),
        gamma_final=metrics.get("rider_gamma_final", 0.0),
    )


def build_full_debug_sweep_result_log_lines(
    *,
    run_num: int,
    total_runs: int,
    expected_distance: float,
    actual_distance: float,
    metrics: SweepMetricSummary,
) -> list[str]:
    """Return full-debug sweep result log lines."""
    log_lines = [
        f"  [RESULT] Run {run_num}/{total_runs}:",
        (
            f"    Distance: expected={expected_distance:.2f}mm, "
            f"actual={actual_distance:.2f}mm"
        ),
        (
            f"    Gamma: initial={metrics.gamma_initial:.6f}, "
            f"final={metrics.gamma_final:.6f}, delta={metrics.delta_gamma:.6e}"
        ),
        f"    Energy: ΔE={metrics.delta_e:.6f}MeV",
    ]
    if actual_distance < 0.1:
        log_lines.append("  [WARNING] Particle barely moved! Check timestep calculation.")
    return log_lines


def build_failed_sweep_run_record(
    *,
    run_num: int,
    aperture: float,
    energy: float,
    start_z: float,
    transv_offset: float,
    timestep: float,
    steps: int,
    error: str,
    error_details: str,
    wall_z: float,
) -> dict[str, Any]:
    """Build a failed-run record for result processing exceptions."""
    return {
        "run_number": run_num,
        "parameters": {
            "aperture_radius": aperture,
            "particle_energy_gev": energy,
            "start_z": start_z,
            "transverse_offset": transv_offset,
            "timestep": timestep,
            "steps": steps,
            "wall_z": wall_z,
        },
        "error": error,
        "error_details": error_details,
    }


def build_timeout_sweep_run_record(
    *,
    run_num: int,
    aperture: float,
    energy: float,
    start_z: float,
    transv_offset: float,
    timestep: float,
    steps: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Build a failed-run record for timed-out sweep integrations."""
    return {
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
        "timeout_seconds": timeout_seconds,
    }


def build_sweep_completion_log_lines(
    *,
    output_dir: str,
    successful_runs: int,
    failed_runs: int,
    elapsed_time: float,
) -> list[str]:
    """Return final sweep completion log lines."""
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    log_lines = [
        "[OK] Sweep completed!",
        f"  Results saved to: {output_dir}",
        f"  Successful runs: {successful_runs}",
    ]
    if failed_runs:
        log_lines.append(f"  Failed/timed-out runs: {failed_runs}")
    if hours > 0:
        log_lines.append(
            f"  Total time: {hours}h {minutes}m {seconds:.1f}s ({elapsed_time:.1f}s)"
        )
    elif minutes > 0:
        log_lines.append(
            f"  Total time: {minutes}m {seconds:.1f}s ({elapsed_time:.1f}s)"
        )
    else:
        log_lines.append(f"  Total time: {elapsed_time:.1f}s")
    return log_lines


def build_successful_sweep_run_log(
    *,
    run_num: int,
    total_runs: int,
    metrics: Mapping[str, Any],
    rest_energy_mev: float,
    param_names: list[str],
    energy: float,
    rider_transv_dist: float,
    sweep_overrides: Mapping[str, Any],
    default_driver_energy_gev: float,
) -> SuccessfulSweepRunLog:
    """Return all formatted logging for one successful CLI sweep run."""
    gamma_initial = metrics.get(
        "initial_gamma_mean", metrics.get("rider_gamma_initial", 1.0)
    )
    gamma_final = metrics.get("final_gamma_mean", metrics.get("rider_gamma_final", 1.0))
    delta_gamma = gamma_final - gamma_initial
    delta_e_mev = delta_gamma * rest_energy_mev
    metric_summary = SweepMetricSummary(
        delta_e=delta_e_mev,
        delta_gamma=delta_gamma,
        gamma_initial=gamma_initial,
        gamma_final=gamma_final,
    )

    optimization_lines = [
        (
            "[OPTIMIZATION] max_percent_energy_gain: "
            f"{metrics.get('max_percent_energy_gain', 0):.12e}%"
        ),
        (
            "[OPTIMIZATION] max_energy_gain: "
            f"{metrics.get('max_energy_gain_gev', 0):.12e} GeV"
        ),
        (
            "[OPTIMIZATION] max_relative_gain: "
            f"{metrics.get('max_relative_gain', 0):.12e}"
        ),
        f"[OPTIMIZATION] delta_gamma: {delta_gamma:.12e}",
        f"[OPTIMIZATION] delta_e_mev: {delta_e_mev:.12e} MeV",
        f"[OPTIMIZATION] final_gamma: {gamma_final:.16f}",
        f"[OPTIMIZATION] initial_gamma: {gamma_initial:.16f}",
    ]

    detail_lines = [
        f"  [RESULT] Run {run_num}/{total_runs}:",
        f"    rider_gamma_initial: {gamma_initial:.16f}",
        f"    rider_gamma_final: {gamma_final:.16f}",
        f"    delta_gamma: {delta_gamma:.12e}",
        f"    delta_e_mev: {delta_e_mev:.12e} MeV",
        (
            "    max_percent_energy_gain: "
            f"{metrics.get('max_percent_energy_gain', 0):.12e}%"
        ),
        f"    max_energy_gain: {metrics.get('max_energy_gain_gev', 0):.12e} GeV",
        f"    max_relative_gain: {metrics.get('max_relative_gain', 0):.12e}",
    ]

    swept_params = []
    if "energy" in param_names:
        swept_params.append(f"initial_energy_gev={energy:.3g}")
    if "rider_transv_dist" in param_names:
        swept_params.append(f"rider_transv_dist={rider_transv_dist:.3e}")
    if "driver_energy_gev" in param_names:
        swept_params.append(
            "driver_energy_gev="
            f"{sweep_overrides.get('driver_energy_gev', default_driver_energy_gev):.3g}"
        )
    param_str = " ".join(swept_params) if swept_params else "fixed_params"

    compact_line = (
        f"Run #{run_num:4d} | {param_str} | "
        f"ΔE={delta_e_mev:.3e} Δγ={delta_gamma:.3e} "
        f"γ_i={gamma_initial:.2f} γ_f={gamma_final:.2f} | SUCCESS"
    )

    return SuccessfulSweepRunLog(
        optimization_lines=optimization_lines,
        detail_lines=detail_lines,
        compact_line=compact_line,
        metrics=metric_summary,
    )


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


__all__ = [
    "SweepAttemptClassification",
    "SweepMetricSummary",
    "SuccessfulSweepRunLog",
    "build_failed_sweep_run_record",
    "build_full_debug_sweep_result_log_lines",
    "build_sweep_completion_log_lines",
    "build_sweep_results_payload",
    "build_successful_sweep_run_log",
    "build_sweep_run_data",
    "build_timeout_sweep_run_record",
    "build_truncated_sweep_log_params",
    "classify_sweep_attempt_result",
    "extract_actual_distance",
    "extract_sweep_metric_summary",
    "simulation_type_name",
]
