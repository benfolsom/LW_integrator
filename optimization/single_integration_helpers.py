"""Pure helpers for per-run integration result processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from optimization.sweep_helpers import AMU_TO_MEV


@dataclass(frozen=True)
class IntegrationMetricsOutcome:
    """Computed metrics plus log lines that should be emitted by the caller."""

    metrics: dict[str, Any]
    log_lines: list[str]


def build_integration_metrics(
    result: Any,
    *,
    rider_m_particle: float,
    run_num: int,
    optimization_mode: bool = False,
) -> IntegrationMetricsOutcome:
    """Build rider metrics from a single integration result."""
    metrics: dict[str, Any] = {}
    log_lines = [
        f"  [RESULT] Run {run_num} metrics:",
        f"    rider_gamma_initial: {result.rider_gamma_initial}",
        f"    rider_gamma_final: {result.rider_gamma_final}",
    ]

    if result.rider_delta_e is not None:
        metrics["rider_delta_e_mev"] = result.rider_delta_e
    if result.rider_gamma_initial is not None:
        metrics["rider_gamma_initial"] = result.rider_gamma_initial
    if result.rider_gamma_final is not None:
        metrics["rider_gamma_final"] = result.rider_gamma_final

    gamma_initial = result.rider_gamma_initial
    gamma_final = result.rider_gamma_final
    if gamma_initial is not None and gamma_final is not None and gamma_initial > 0:
        _add_energy_gain_metrics(
            metrics,
            log_lines,
            gamma_initial=gamma_initial,
            gamma_final=gamma_final,
            rider_m_particle=rider_m_particle,
            source_prefix="",
        )
        if optimization_mode:
            log_lines.append(
                f"    optimizer_objective: {-metrics['max_percent_energy_gain']:.12e}"
            )
    else:
        _add_fallback_energy_gain_metrics(
            result,
            metrics,
            log_lines,
            rider_m_particle=rider_m_particle,
            run_num=run_num,
        )

    _add_beam_optics_metrics(result, metrics)

    metrics["num_particles_dead"] = result.num_particles_dead
    if result.halted_early:
        metrics["halted_early"] = True
        if result.halt_reason:
            metrics["halt_reason"] = result.halt_reason

    return IntegrationMetricsOutcome(metrics=metrics, log_lines=log_lines)


def sample_trajectory_arrays(
    trajectory: Mapping[str, Any], stride: int
) -> dict[str, list]:
    """Return a stride-sampled trajectory payload suitable for JSON output."""
    return {
        "z": np.asarray(trajectory["z"])[::stride].tolist(),
        "r": np.asarray(trajectory["r"])[::stride].tolist(),
        "pz": np.asarray(trajectory["pz"])[::stride].tolist(),
        "pr": np.asarray(trajectory["pr"])[::stride].tolist(),
        "t": np.asarray(trajectory["t"])[::stride].tolist(),
        "gamma": np.asarray(trajectory["gamma"])[::stride].tolist(),
    }


def distance_info_from_trajectory(
    trajectory: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return compact distance metadata from a trajectory, if z data exists."""
    z_array = np.asarray(trajectory["z"])
    if len(z_array) == 0:
        return None
    return {
        "z_start": float(z_array[0]),
        "z_end": float(z_array[-1]),
        "num_steps": len(z_array),
    }


def _add_energy_gain_metrics(
    metrics: dict[str, Any],
    log_lines: list[str],
    *,
    gamma_initial: float,
    gamma_final: float,
    rider_m_particle: float,
    source_prefix: str,
) -> None:
    delta_gamma = gamma_final - gamma_initial
    energy_gain_percent = delta_gamma / gamma_initial * 100.0
    energy_gain_ppm = delta_gamma / gamma_initial * 1e6
    delta_e_mev = delta_gamma * rider_m_particle * AMU_TO_MEV

    metrics["max_percent_energy_gain"] = energy_gain_percent
    metrics["percent_delta_e"] = energy_gain_percent
    metrics["delta_gamma"] = delta_gamma
    metrics["delta_e_mev"] = delta_e_mev
    metrics["energy_gain_ppm"] = energy_gain_ppm

    if source_prefix:
        log_lines.extend(
            [
                f"    gamma_initial ({source_prefix}): {gamma_initial:.12e}",
                f"    gamma_final ({source_prefix}): {gamma_final:.12e}",
            ]
        )
    log_lines.extend(
        [
            f"    delta_gamma: {delta_gamma:.12e}",
            f"    delta_e_mev: {delta_e_mev:.12e} MeV",
            f"    max_percent_energy_gain: {energy_gain_percent:.12e}%",
            f"    percent_delta_e: {energy_gain_percent:.12e}%",
            f"    energy_gain_ppm: {energy_gain_ppm:.6f} ppm",
        ]
    )


def _add_fallback_energy_gain_metrics(
    result: Any,
    metrics: dict[str, Any],
    log_lines: list[str],
    *,
    rider_m_particle: float,
    run_num: int,
) -> None:
    log_lines.append(
        "  [WARNING] Gamma values missing, attempting trajectory fallback..."
    )
    if result.rider_trajectory is None:
        log_lines.append("  [ERROR] No trajectory data available for fallback")
        _log_missing_energy_gain(log_lines, run_num)
        return

    try:
        gamma_array = np.asarray(result.rider_trajectory.get("gamma", []))
        if len(gamma_array) == 0:
            log_lines.append("  [ERROR] Trajectory gamma array is empty")
            _log_missing_energy_gain(log_lines, run_num)
            return

        gamma_initial = float(gamma_array[0])
        gamma_final = float(gamma_array[-1])
        if gamma_initial <= 0:
            log_lines.append("  [ERROR] Fallback gamma_initial <= 0")
            _log_missing_energy_gain(log_lines, run_num)
            return

        log_lines.append("  [OK] Fallback calculation successful:")
        _add_energy_gain_metrics(
            metrics,
            log_lines,
            gamma_initial=gamma_initial,
            gamma_final=gamma_final,
            rider_m_particle=rider_m_particle,
            source_prefix="from traj",
        )
    except Exception as exc:
        log_lines.append(f"  [ERROR] Fallback calculation failed: {exc}")
        _log_missing_energy_gain(log_lines, run_num)


def _log_missing_energy_gain(log_lines: list[str], run_num: int) -> None:
    log_lines.extend(
        [
            f"  [CRITICAL] max_percent_energy_gain could not be calculated for Run {run_num}",
            "  [CRITICAL] This will result in NaN/inf for optimization objective",
        ]
    )


def _add_beam_optics_metrics(result: Any, metrics: dict[str, Any]) -> None:
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
