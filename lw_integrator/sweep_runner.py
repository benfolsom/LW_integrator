"""Headless sweep runner for CLI execution of parameter sweeps.

This module provides a standalone interface to run parameter sweeps without
requiring the GUI. It can be invoked from the command-line interface or
used programmatically.

The CLI sweep runner now calls the SAME core code paths as the GUI:
  - run_testbed() for integration (same particle init, same integrator call)
  - SimulationOptions for configuration (same dataclass as GUI)
  - Same metric extraction from RunResult

Output Locations
----------------
When running sweeps via CLI, output is written to two locations:

1. **Results directory** (e.g., results/sweeps/YYYYMMDD_HHMMSS_configname/)
   - sweep_results.json: Parameter combinations and metrics
   - sweep.log: High-level progress summary

2. **logcache/** directory (same as GUI sweeps)
   - YYYYMMDD_HHMMSS_sweep_cli.log: Detailed debug output
   - Includes SC iterations, adaptive timestep details, etc.
   - Automatically rotated when files exceed 50 MB
   - Old logs purged when cache exceeds 500 MB

This matches the behavior of GUI sweeps, ensuring consistency in logging
and debugging workflows.
"""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import signal
import shutil
import time
import traceback as _traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import current_thread, main_thread
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

from core.constants import C_MMNS
from core.debug_logger import initialize_debug_logging
from core.smoothness_analyzer import SmoothnessConfig, analyze_trajectory_smoothness
from core.types import SimulationType
from lw_integrator.testbed_runner import run_testbed
from optimization.config import (
    OptimizationConfig,
    calculate_auto_steps,
    calculate_auto_timestep,
)
from optimization.logging_policy import (
    apply_run_logging_policy,
    describe_run_logging_policy,
    restore_run_logging_policy,
)
from optimization.mode_helpers import normalize_sweep_or_optimization_mode
from optimization.single_integration_helpers import (
    build_integration_metrics,
    build_integration_trajectory_output,
    build_single_integration_setup,
    calculate_rider_starting_pz,
)
from optimization.run_logging_helpers import (
    build_progress_log_line,
    build_small_aperture_diagnostic_line,
    build_stability_config_log_lines,
    should_emit_verbose_run_log,
)
from optimization.simulation_type_helpers import is_bunch_to_bunch
from optimization.sweep_helpers import build_config_parameter_grids
from optimization.sweep_result_helpers import (
    build_exception_sweep_run_log_lines,
    build_exception_sweep_run_record,
    build_interrupted_sweep_results_payload,
    build_successful_sweep_run_log,
    build_sweep_results_payload,
)
from optimization.sweep_run_helpers import (
    build_full_debug_parameter_log_lines,
    resolve_sweep_run_parameters,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMU_TO_MEV = 931.494


@dataclass(frozen=True)
class _ResolvedRiderOverrides:
    m_particle: float
    charge_sign: float
    pcount: int
    transv_mom: float
    transv_dist: float
    transverse_geometry: str
    stripped_ions: float
    macroparticle_charge_multiplier: float
    macroparticle_sigma_multiplier: float


@dataclass(frozen=True)
class _CliDriverSetup:
    params: dict[str, Any] | None
    log_line: str | None


@dataclass(frozen=True)
class _CliTimestepSetup:
    transv_offset: float
    steps: int
    timestep: float
    gamma: float
    beta: float


@dataclass(frozen=True)
class _CliStabilityOutcome:
    log_lines: list[str]
    metrics_updates: dict[str, Any]
    rejection_record: dict[str, Any] | None


class _PerRunTimeoutError(TimeoutError):
    """Raised when a CLI sweep integration exceeds its configured timeout."""

    def __init__(self, timeout_seconds: float):
        super().__init__(f"Run exceeded timeout of {timeout_seconds:.1f}s")
        self.timeout_seconds = timeout_seconds


@contextmanager
def _per_run_timeout(timeout_seconds: float):
    """Apply a wall-clock timeout to one integration attempt on Unix main threads."""
    if timeout_seconds <= 0 or current_thread() is not main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def _raise_timeout(_signum, _frame):
        raise _PerRunTimeoutError(timeout_seconds)

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _resolve_cli_rider_overrides(
    config: Any,
    sweep_overrides: Mapping[str, Any],
) -> _ResolvedRiderOverrides:
    return _ResolvedRiderOverrides(
        m_particle=sweep_overrides.get("rider_m_particle", config.m_particle),
        charge_sign=sweep_overrides.get("rider_charge_sign", config.charge_sign),
        pcount=int(sweep_overrides.get("rider_pcount", config.pcount)),
        transv_mom=sweep_overrides.get("rider_transv_mom", config.transv_mom),
        transv_dist=sweep_overrides.get("rider_transv_dist", config.transv_dist),
        transverse_geometry=getattr(config, "transverse_geometry", "square"),
        stripped_ions=sweep_overrides.get("rider_stripped_ions", config.stripped_ions),
        macroparticle_charge_multiplier=sweep_overrides.get(
            "macroparticle_charge_multiplier",
            config.macroparticle_charge_multiplier,
        ),
        macroparticle_sigma_multiplier=sweep_overrides.get(
            "macroparticle_sigma_multiplier",
            config.macroparticle_sigma_multiplier,
        ),
    )


def _resolve_cli_driver_setup(
    config: Any,
    sweep_overrides: Mapping[str, Any],
) -> _CliDriverSetup:
    if not is_bunch_to_bunch(config.simulation_type):
        return _CliDriverSetup(params=None, log_line=None)

    d_m = sweep_overrides.get("driver_m_particle", config.driver_m_particle)
    d_charge = sweep_overrides.get("driver_charge_sign", config.driver_charge_sign)
    d_pcount = int(sweep_overrides.get("driver_pcount", config.driver_pcount))
    d_transv_mom = sweep_overrides.get("driver_transv_mom", config.driver_transv_mom)
    d_transv_dist = sweep_overrides.get("driver_transv_dist", config.driver_transv_dist)
    d_long_dist = sweep_overrides.get("driver_long_dist", getattr(config, "driver_long_dist", 0.0))
    d_start_dist = sweep_overrides.get(
        "driver_starting_distance", config.driver_starting_distance
    )
    d_stripped = sweep_overrides.get(
        "driver_stripped_ions", config.driver_stripped_ions
    )
    d_energy_gev = sweep_overrides.get("driver_energy_gev", config.driver_energy_gev)

    driver_negative = getattr(config, "driver_direction", "-z") == "-z"
    pz_sign = -1.0 if driver_negative else 1.0
    driver_pz_magnitude = calculate_rider_starting_pz(
        abs(d_energy_gev), d_m, SimulationType.BUNCH_TO_BUNCH
    )
    params = {
        "starting_distance": d_start_dist,
        "transv_mom": d_transv_mom,
        "transv_dist": d_transv_dist,
        "long_dist": d_long_dist,
        "transverse_geometry": getattr(config, "driver_transverse_geometry", "square"),
        "transv_offset_x": getattr(config, "driver_transv_offset_x", 0.0),
        "transv_offset_y": getattr(config, "driver_transv_offset_y", 0.0),
        "m_particle": d_m,
        "charge_sign": d_charge,
        "pcount": d_pcount,
        "stripped_ions": d_stripped,
        "starting_Pz": pz_sign * driver_pz_magnitude,
    }

    dir_label = "\u2212z" if driver_negative else "+z"
    return _CliDriverSetup(
        params=params,
        log_line=(
            f"[OPTIMIZATION]   [DRIVER] energy={d_energy_gev:.4f} GeV, "
            f"m={d_m:.4e} amu, Pz={params['starting_Pz']:.4e} ({dir_label}), "
            f"stripped={d_stripped:.2e}, pcount={d_pcount}"
        ),
    )


def _resolve_cli_timestep_setup(
    config: Any,
    *,
    aperture: float,
    energy_gev: float,
    start_z: float,
    transv_offset_frac: float,
    rider_m_particle: float,
    sweep_overrides: Mapping[str, Any],
) -> _CliTimestepSetup:
    transv_offset = transv_offset_frac * aperture

    if config.auto_steps:
        if config.timestep_strategy == "auto_distance":
            preliminary_timestep = calculate_auto_timestep(
                start_z=start_z,
                wall_z=config.wall_z,
                distance_past_wall=config.auto_steps_distance_past_wall,
                particle_energy_gev=energy_gev,
                particle_mass_amu=rider_m_particle,
                target_steps=config.auto_steps_target,
            )
            steps = calculate_auto_steps(
                start_z=start_z,
                wall_z=config.wall_z,
                distance_past_wall=config.auto_steps_distance_past_wall,
                timestep=preliminary_timestep,
                particle_energy_gev=energy_gev,
                particle_mass_amu=rider_m_particle,
            )
        else:
            steps = calculate_auto_steps(
                start_z=start_z,
                wall_z=config.wall_z,
                distance_past_wall=config.auto_steps_distance_past_wall,
                timestep=config.timestep,
                particle_energy_gev=energy_gev,
                particle_mass_amu=rider_m_particle,
            )
    else:
        steps = config.steps

    driver_start_z = 1000.0
    if is_bunch_to_bunch(config.simulation_type):
        driver_start_z = sweep_overrides.get(
            "driver_starting_distance", config.driver_starting_distance
        )

    original_steps = config.steps
    try:
        config.steps = steps
        timestep = config.calculate_timestep_for_energy(
            energy_gev=energy_gev,
            start_z=start_z,
            wall_z=config.wall_z,
            driver_start_z=driver_start_z,
            m_particle_amu=rider_m_particle,
        )
    finally:
        config.steps = original_steps

    rest_energy_mev = rider_m_particle * AMU_TO_MEV
    if is_bunch_to_bunch(config.simulation_type):
        gamma = (energy_gev * 1e3) / rest_energy_mev + 1.0
    else:
        gamma = (energy_gev * 1e3) / rest_energy_mev
    if gamma < 1.0:
        gamma = 1.0
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.0

    return _CliTimestepSetup(
        transv_offset=transv_offset,
        steps=steps,
        timestep=timestep,
        gamma=gamma,
        beta=beta,
    )


def _format_aperture_for_start_log(aperture: float) -> str:
    if aperture >= 1.0:
        return f"{aperture:.1f}"
    if aperture >= 0.01:
        return f"{aperture:.4f}"
    return f"{aperture:.6f}"


def _build_cli_timestep_log_lines(
    *,
    run_num: int,
    timestep_strategy: str,
    energy_gev: float,
    rider_m_particle: float,
    gamma: float,
    beta: float,
    timestep: float,
    steps: int,
    start_z: float,
    wall_z: float,
    auto_steps_distance_past_wall: float,
    auto_steps_target: int,
) -> list[str]:
    lines = [
        (
            f"[OPTIMIZATION]   [TIMESTEP] Run {run_num} "
            f"strategy '{timestep_strategy}':"
        ),
        (
            f"[OPTIMIZATION]     E={energy_gev:.4f} GeV, "
            f"m={rider_m_particle:.4e} amu"
        ),
        f"[OPTIMIZATION]     gamma={gamma:.2f}, beta={beta:.8f}",
        (
            "[OPTIMIZATION]     timestep h="
            f"{timestep:.4e} ns (proper time = dt/gamma)"
        ),
        f"[OPTIMIZATION]     steps={steps}",
    ]
    if timestep_strategy == "auto_distance":
        distance_per_step = beta * gamma * C_MMNS * timestep
        expected_total = distance_per_step * steps
        lines.extend(
            [
                (
                    "[OPTIMIZATION]     distance_per_step = β·γ·c·h = "
                    f"{distance_per_step:.4f} mm"
                ),
                (
                    "[OPTIMIZATION]     expected_total_distance = "
                    f"{expected_total:.2f} mm"
                ),
                (
                    f"[OPTIMIZATION]     wall_z={wall_z:.2f} mm, "
                    f"start_z={start_z:.2f} mm"
                ),
                (
                    "[OPTIMIZATION]     distance_to_wall = "
                    f"{abs(wall_z - start_z):.2f} mm"
                ),
                (
                    "[OPTIMIZATION]     distance_past_wall="
                    f"{auto_steps_distance_past_wall:.2f} mm"
                ),
                f"[OPTIMIZATION]     target_steps={auto_steps_target}",
            ]
        )
    return lines


def _build_cli_start_log_lines(
    *,
    run_num: int,
    total_runs: int,
    aperture: float,
    energy_gev: float,
    start_z: float,
    timestep: float,
    steps: int,
) -> list[str]:
    aperture_str = _format_aperture_for_start_log(aperture)
    return [
        (
            f"[OPTIMIZATION] [START] Run {run_num}/{total_runs}: "
            f"a={aperture_str}mm, E={energy_gev:.2f}GeV"
        ),
        (
            f"[OPTIMIZATION]   [PARAMS] z={start_z:.2f}mm, "
            f"h={timestep:.4e}ns, N={steps}"
        ),
    ]


def _evaluate_cli_stability(
    config: Any,
    result: Any,
    metrics: Mapping[str, Any],
    *,
    rider_m_particle: float,
    run_num: int,
    aperture: float,
    energy_gev: float,
    start_z: float,
    transv_offset: float,
) -> _CliStabilityOutcome:
    log_lines = [
        f"[OPTIMIZATION]   [DEBUG] Processing trajectory data for Run {run_num}..."
    ]
    metrics_updates: dict[str, Any] = {}

    if result.rider_trajectory is not None and config.smoothness_enabled:
        log_lines.append(
            f"[OPTIMIZATION]   [DEBUG] Performing stability analysis for Run {run_num}..."
        )
        smoothness_config = SmoothnessConfig(
            enabled=True,
            window_size=config.smoothness_window_size,
            oscillation_threshold=config.smoothness_oscillation_threshold,
            trend_smoothness_threshold=config.smoothness_trend_threshold,
            reject_on_violation=config.smoothness_reject_on_violation,
            max_allowed_violations=config.smoothness_max_violations,
        )
        smoothness_result = analyze_trajectory_smoothness(
            result.rider_trajectory,
            smoothness_config,
            particle_mass_amu=rider_m_particle,
        )
        metrics_updates = {
            "smoothness_passed": smoothness_result.passed,
            "smoothness_violations": len(smoothness_result.violations),
        }

        if not smoothness_result.passed:
            log_lines.extend(
                [
                    (
                        f"[OPTIMIZATION]   [WARNING] Stability check FAILED "
                        f"for Run {run_num}"
                    ),
                    f"[OPTIMIZATION]     Quality: {smoothness_result.quality_summary}",
                ]
            )
            if config.smoothness_reject_on_violation:
                log_lines.append(
                    f"[OPTIMIZATION]   [REJECT] Run {run_num} rejected due to numerical instability"
                )
                rejected_metrics = {**metrics, **metrics_updates}
                return _CliStabilityOutcome(
                    log_lines=log_lines,
                    metrics_updates=metrics_updates,
                    rejection_record={
                        "success": False,
                        "error": (
                            "Smoothness violation: "
                            f"{len(smoothness_result.violations)} violations"
                        ),
                        "parameters": {
                            "aperture": aperture,
                            "energy_gev": energy_gev,
                            "start_z": start_z,
                            "transv_offset": transv_offset,
                        },
                        "metrics": rejected_metrics,
                    },
                )
        else:
            log_lines.append(
                f"[OPTIMIZATION]   [OK] Stability check PASSED for Run {run_num}: "
                f"{smoothness_result.quality_summary}"
            )
    elif result.rider_trajectory is None:
        log_lines.append(
            f"[OPTIMIZATION]   [WARNING] No trajectory data for Run {run_num}"
        )
    elif not config.smoothness_enabled:
        log_lines.append(
            f"[OPTIMIZATION]   [INFO] Stability analysis DISABLED for Run {run_num}"
        )

    return _CliStabilityOutcome(
        log_lines=log_lines,
        metrics_updates=metrics_updates,
        rejection_record=None,
    )


def _build_cli_sweep_start_log_lines(
    config: Any,
    *,
    param_grids: Mapping[str, list[Any]],
    total_runs: int,
) -> list[str]:
    lines = [
        f"Starting BLIND SWEEP (Grid Search): {total_runs} total runs",
        f"  Simulation type: {config.simulation_type}",
    ]
    for grid_key, grid_vals in param_grids.items():
        if len(grid_vals) > 1:
            lines.append(
                f"  {grid_key}: {len(grid_vals)} points from "
                f"{min(grid_vals):.4e} to {max(grid_vals):.4e}"
            )
        else:
            lines.append(f"  {grid_key}: {grid_vals[0]:.4e} (fixed)")

    lines.append(f"  Timestep strategy: {config.timestep_strategy}")
    if config.timestep_strategy == "auto_distance":
        lines.extend(
            [
                (
                    "    Distance past wall: "
                    f"{config.auto_steps_distance_past_wall} mm"
                ),
                (
                    "    Target steps for timestep calculation: "
                    f"{config.auto_steps_target}"
                ),
                "    All particles will travel to consistent z regardless of energy",
            ]
        )
    lines.append(f"  z_cutoff_mode: {config.z_cutoff_mode}")

    if is_bunch_to_bunch(config.simulation_type):
        lines.extend(
            [
                "",
                "  Fixed rider parameters:",
                f"    m_particle: {config.m_particle:.4e} amu",
                f"    charge_sign: {config.charge_sign}",
                f"    pcount: {config.pcount}",
                f"    stripped_ions: {config.stripped_ions:.2e}",
                f"    transv_mom: {config.transv_mom:.4e}",
                f"    transv_dist: {config.transv_dist:.4e}",
                f"    transverse_geometry: {getattr(config, 'transverse_geometry', 'square')}",
                "  Fixed driver parameters:",
                f"    m_particle: {config.driver_m_particle:.4e} amu",
                f"    charge_sign: {config.driver_charge_sign}",
                f"    pcount: {config.driver_pcount}",
                f"    stripped_ions: {config.driver_stripped_ions:.2e}",
                f"    energy_gev: {config.driver_energy_gev:.4f}",
                f"    transverse_geometry: {getattr(config, 'driver_transverse_geometry', 'square')}",
                ("    starting_distance: " f"{config.driver_starting_distance:.2f}"),
            ]
        )
    return lines


def _worker_run_combo(payload: dict) -> dict:
    """Top-level picklable worker for parallel sweep execution."""
    config: OptimizationConfig = payload["config"]
    output_dir: Path = payload["output_dir"]
    run_num: int = payload["run_num"]
    total_runs: int = payload["total_runs"]
    aperture: float = payload["aperture"]
    energy_gev: float = payload["energy_gev"]
    start_z: float = payload["start_z"]
    transv_offset_frac: float = payload["transv_offset_frac"]
    sweep_overrides: dict = payload["sweep_overrides"]
    params_dict: dict = payload["params_dict"]

    runner = SweepRunner(config=config, output_dir=output_dir, verbose=False)
    try:
        result = runner._run_single_integration(
            aperture=aperture,
            energy_gev=energy_gev,
            start_z=start_z,
            transv_offset_frac=transv_offset_frac,
            run_num=run_num,
            total_runs=total_runs,
            sweep_overrides=sweep_overrides,
            emit_run_diagnostics=False,
            emit_run_summary=False,
        )
    except Exception as exc:
        result = {
            "success": False,
            "error": str(exc),
            "error_details": _traceback.format_exc(),
        }

    result["run_number"] = run_num
    if result.get("parameters") is None:
        result["parameters"] = {}
    result["parameters"].update(params_dict)
    result["_params_dict"] = params_dict
    return result


class SweepRunner:
    """Execute parameter sweeps from configuration files without GUI.

    This runner delegates all integration work to ``run_testbed``, the same
    function used by the GUI.  This guarantees identical particle initialization,
    self-consistency configuration, adaptive timestep handling, metric extraction,
    and debug logging between the two interfaces.
    """

    def __init__(
        self,
        config: OptimizationConfig,
        output_dir: Path,
        verbose: bool = True,
        workers: Optional[int] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.workers = workers
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.results: List[Dict[str, Any]] = []
        self.log_file = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Log a message to stdout and log file with [OPTIMIZATION] prefix."""
        self._log_line(f"[OPTIMIZATION] {message}")

    def _log_line(self, line: str) -> None:
        """Log a preformatted line to stdout and the sweep log file."""
        if self.verbose:
            print(line, flush=True)
        if self.log_file is not None:
            self.log_file.write(f"{line}\n")
            self.log_file.flush()
        if self.log_callback is not None:
            self.log_callback(line)

    def _emit_progress(self, completed: int, total: int) -> None:
        """Emit sweep-progress updates when a callback is registered."""
        if self.progress_callback is not None:
            self.progress_callback(completed, total)

    # ------------------------------------------------------------------
    # Grid generation (unchanged from original)
    # ------------------------------------------------------------------

    def _generate_parameter_grids(self) -> Dict[str, List[float]]:
        """Generate parameter grids for sweep."""
        return build_config_parameter_grids(self.config)

    # ------------------------------------------------------------------
    # Single-integration runner (delegates to run_testbed)
    # ------------------------------------------------------------------

    def _run_single_integration(
        self,
        aperture: float,
        energy_gev: float,
        start_z: float,
        transv_offset_frac: float,
        run_num: int,
        total_runs: int = 1,
        sweep_overrides: Optional[Dict[str, float]] = None,
        emit_run_diagnostics: bool = True,
        emit_run_summary: bool = True,
    ) -> Dict[str, Any]:
        """Run a single integration via run_testbed (same path as GUI).

        This method constructs a SimulationOptions, calls run_testbed(),
        and extracts metrics from the RunResult — identical to the GUI's
        OptimizationPlugin._run_single_integration.
        """
        if sweep_overrides is None:
            sweep_overrides = {}

        rider = _resolve_cli_rider_overrides(self.config, sweep_overrides)

        timestep_setup = _resolve_cli_timestep_setup(
            self.config,
            aperture=aperture,
            energy_gev=energy_gev,
            start_z=start_z,
            transv_offset_frac=transv_offset_frac,
            rider_m_particle=rider.m_particle,
            sweep_overrides=sweep_overrides,
        )

        # ── Log timestep calculation ──
        if emit_run_diagnostics:
            for line in _build_cli_timestep_log_lines(
                run_num=run_num,
                timestep_strategy=self.config.timestep_strategy,
                energy_gev=energy_gev,
                rider_m_particle=rider.m_particle,
                gamma=timestep_setup.gamma,
                beta=timestep_setup.beta,
                timestep=timestep_setup.timestep,
                steps=timestep_setup.steps,
                start_z=start_z,
                wall_z=self.config.wall_z,
                auto_steps_distance_past_wall=(
                    self.config.auto_steps_distance_past_wall
                ),
                auto_steps_target=self.config.auto_steps_target,
            ):
                self._log_line(line)

        # Log [START] line
        if emit_run_summary:
            for line in _build_cli_start_log_lines(
                run_num=run_num,
                total_runs=total_runs,
                aperture=aperture,
                energy_gev=energy_gev,
                start_z=start_z,
                timestep=timestep_setup.timestep,
                steps=timestep_setup.steps,
            ):
                self._log_line(line)

        # ── Build driver_params dict if BUNCH_TO_BUNCH ──
        driver_setup = _resolve_cli_driver_setup(self.config, sweep_overrides)
        driver_params = driver_setup.params
        if emit_run_diagnostics and driver_setup.log_line is not None:
            self._log_line(driver_setup.log_line)

        # ── Build SimulationOptions (same dataclass the GUI uses) ──
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        run_output_dir = self.output_dir / f"_temp_run_{run_num}_{timestamp}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        setup = build_single_integration_setup(
            self.config,
            aperture=aperture,
            energy_gev=energy_gev,
            start_z=start_z,
            transv_offset=timestep_setup.transv_offset,
            timestep=timestep_setup.timestep,
            steps=timestep_setup.steps,
            run_output_dir=run_output_dir,
            run_num=run_num,
            driver_params=driver_params,
            rider_m_particle=rider.m_particle,
            rider_charge_sign=rider.charge_sign,
            rider_pcount=rider.pcount,
            rider_transv_mom=rider.transv_mom,
            rider_transv_dist=rider.transv_dist,
            rider_stripped_ions=rider.stripped_ions,
            macroparticle_charge_multiplier=rider.macroparticle_charge_multiplier,
            macroparticle_sigma_multiplier=rider.macroparticle_sigma_multiplier,
        )
        options = setup.options

        # ── Progress + log callbacks (same format as GUI) ──
        def progress_callback(current: int, total: int, _run_id=run_num):
            line = build_progress_log_line(
                run_num=_run_id,
                current=current,
                total=total,
                prefix="[OPTIMIZATION] ",
            )
            if line is not None:
                self._log_line(line)

        def _verbose_log(message: str) -> None:
            if should_emit_verbose_run_log(message):
                self._log_line(f"[OPTIMIZATION]     [VERBOSE] {message}")

        log_callback: Optional[Callable[[str], None]] = None
        if emit_run_diagnostics and (
            self.config.self_consistency_verbosity > 0
            or self.config.adaptive_timestep_debug
        ):
            log_callback = _verbose_log

        # ── Log stability settings (same as GUI) ──
        if emit_run_diagnostics:
            for line in build_stability_config_log_lines(
                self.config,
                run_num=run_num,
                prefix="[OPTIMIZATION] ",
            ):
                self._log_line(line)

        diagnostic_line = build_small_aperture_diagnostic_line(
            run_num=run_num,
            aperture=aperture,
            prefix="[OPTIMIZATION] ",
        )
        if emit_run_diagnostics and diagnostic_line is not None:
            self._log_line(diagnostic_line)

        if emit_run_diagnostics:
            self._log_line(
                f"[OPTIMIZATION]   [DEBUG] Calling run_testbed for Run {run_num}..."
            )

        # ── Call run_testbed (THE SAME function the GUI calls) ──
        try:
            with _per_run_timeout(float(self.config.per_run_timeout)):
                result = run_testbed(
                    options,
                    log=log_callback,
                    progress_callback=(
                        progress_callback if emit_run_diagnostics else None
                    ),
                )
            if emit_run_diagnostics:
                self._log_line(
                    f"[OPTIMIZATION]   [DEBUG] run_testbed completed for Run {run_num}"
                )
        except _PerRunTimeoutError as e:
            return {
                "success": False,
                "error": f"TIMEOUT after {e.timeout_seconds:.1f}s",
                "timed_out": True,
                "timeout_seconds": e.timeout_seconds,
                "parameters": {
                    "aperture": aperture,
                    "energy_gev": energy_gev,
                    "start_z": start_z,
                    "transv_offset": timestep_setup.transv_offset,
                    "timestep": timestep_setup.timestep,
                    "steps": timestep_setup.steps,
                },
            }
        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": f"run_testbed failed: {e}\n{traceback.format_exc()}",
                "parameters": {
                    "aperture": aperture,
                    "energy_gev": energy_gev,
                    "start_z": start_z,
                    "transv_offset": timestep_setup.transv_offset,
                },
            }
        finally:
            # Always clean up temp directory
            try:
                if run_output_dir.exists():
                    shutil.rmtree(run_output_dir)
            except Exception as cleanup_error:
                self._log(
                    f"[WARNING] Failed to remove temporary run directory "
                    f"{run_output_dir}: {cleanup_error}"
                )

        # ── Check for halted run ──
        # distance_reached means the relative z-cutoff fired as intended; treat as success.
        _distance_reached = (
            result.halted_early
            and isinstance(result.halt_reason, str)
            and result.halt_reason.startswith("distance_reached")
        )
        if result.halted_early and not _distance_reached:
            if emit_run_summary:
                self._log_line(
                    "[OPTIMIZATION]   [WARNING] "
                    f"Run {run_num} halted early: {result.halt_reason}"
                )
            return {
                "success": False,
                "error": f"Halted early: {result.halt_reason}",
                "halted_early": True,
                "halt_reason": result.halt_reason,
                "parameters": {
                    "aperture": aperture,
                    "energy_gev": energy_gev,
                    "start_z": start_z,
                    "transv_offset": timestep_setup.transv_offset,
                    "timestep": timestep_setup.timestep,
                    "steps": timestep_setup.steps,
                },
            }

        # ── Extract metrics from RunResult (same helper as GUI) ──
        if emit_run_diagnostics:
            self._log_line(
                f"[OPTIMIZATION]   [DEBUG] Extracting metrics for Run {run_num}..."
            )
        metrics_outcome = build_integration_metrics(
            result,
            rider_m_particle=rider.m_particle,
            run_num=run_num,
        )
        metrics = metrics_outcome.metrics
        if emit_run_diagnostics:
            for line in metrics_outcome.log_lines:
                self._log_line(f"[OPTIMIZATION] {line}")

        trajectory_outcome = build_integration_trajectory_output(
            result,
            self.config,
            run_num=run_num,
            rider_m_particle=rider.m_particle,
            metrics=metrics,
            save_trajectory=(
                self.config.save_all_trajectories
                or self.config.save_failed_trajectories
            ),
            trajectory_stride=self.config.trajectory_stride,
        )
        if emit_run_diagnostics:
            for line in trajectory_outcome.log_lines:
                self._log_line(f"[OPTIMIZATION] {line}")
            self._log_line(
                "[OPTIMIZATION]   [DEBUG] "
                f"_run_single_integration returning for Run {run_num}"
            )

        output = {
            "success": True,
            "parameters": {
                "aperture": aperture,
                "energy_gev": energy_gev,
                "start_z": start_z,
                "transv_offset": timestep_setup.transv_offset,
                "timestep": timestep_setup.timestep,
                "steps": timestep_setup.steps,
            },
            "metrics": metrics,
        }
        output.update(trajectory_outcome.output_updates)

        if output.get("stability_rejected"):
            output["success"] = False
            output["error"] = (
                "Smoothness violation: "
                f"{metrics.get('smoothness_violations', 0)} violations"
            )
        return output

    # ------------------------------------------------------------------
    # Sweep orchestration
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """Execute the parameter sweep.

        Returns True if sweep completed successfully.
        """
        start_time = None  # initialised early so KeyboardInterrupt handler can use it

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Open log file
        log_path = self.output_dir / "sweep.log"
        self.log_file = open(log_path, "w")

        # Initialize debug logging to logcache (like GUI sweeps)
        initialize_debug_logging(context="sweep_cli", force_new_log=True)
        logging_policy = apply_run_logging_policy(self.config)
        use_no_logging = logging_policy.suppress_run_logs
        use_truncated_logging = logging_policy.use_truncated_run_logs
        use_full_debug = logging_policy.use_full_run_logs

        try:
            self._log("")
            for line in describe_run_logging_policy(logging_policy):
                self._log(line)
            self._log(
                f"Trajectory saving: Top N={self.config.save_top_n_trajectories}, "
                f"All={self.config.save_all_trajectories}, "
                f"Failed={self.config.save_failed_trajectories}"
            )

            if self.config.mode == "optimization":
                self._log("[ERROR] Optimization mode not yet supported in headless CLI")
                self._log("Please use the GUI for optimization runs")
                return False

            # Generate parameter grids
            param_grids = self._generate_parameter_grids()

            # Calculate total runs
            total_runs = 1
            for values in param_grids.values():
                total_runs *= len(values)

            for line in _build_cli_sweep_start_log_lines(
                self.config,
                param_grids=param_grids,
                total_runs=total_runs,
            ):
                self._log(line)

            self._log("")
            self._log(f"Output directory: {self.output_dir}")
            self._log("")

            # ── Build iteration over all grid dimensions ──
            param_names = list(param_grids.keys())
            param_values_lists = [param_grids[name] for name in param_names]
            _positional_keys = {"aperture", "energy", "start_z", "transv_offset_frac"}

            # Pre-build the full list of combos so we can dispatch all at once
            all_combos = list(itertools.product(*param_values_lists))

            # Run sweep
            start_time = time.time()
            failed_count = 0

            use_parallel = self.workers is not None and self.workers > 1

            if use_parallel:
                # ── Build payloads for worker processes ──
                payloads = []
                for run_num, param_combo in enumerate(all_combos, start=1):
                    params_dict = dict(zip(param_names, param_combo))
                    aperture = params_dict.get("aperture", 0.001)
                    energy = params_dict["energy"]
                    start_z = params_dict["start_z"]
                    transv_offset_frac = params_dict.get("transv_offset_frac", 0.0)
                    sweep_overrides = {
                        k: v
                        for k, v in params_dict.items()
                        if k not in _positional_keys
                    }
                    payloads.append(
                        {
                            "config": self.config,
                            "output_dir": self.output_dir,
                            "run_num": run_num,
                            "total_runs": total_runs,
                            "aperture": aperture,
                            "energy_gev": energy,
                            "start_z": start_z,
                            "transv_offset_frac": transv_offset_frac,
                            "sweep_overrides": sweep_overrides,
                            "params_dict": params_dict,
                        }
                    )

                # ── Dispatch all combos in parallel ──
                raw_results: Dict[int, dict] = {}
                completed_count = 0
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.workers
                ) as executor:
                    future_to_run = {
                        executor.submit(_worker_run_combo, p): p["run_num"]
                        for p in payloads
                    }
                    try:
                        for future in concurrent.futures.as_completed(future_to_run):
                            rn = future_to_run[future]
                            try:
                                raw_results[rn] = future.result()
                            except Exception as exc:
                                raw_results[rn] = {
                                    "success": False,
                                    "run_number": rn,
                                    "error": str(exc),
                                    "error_details": _traceback.format_exc(),
                                    "parameters": payloads[rn - 1]["params_dict"],
                                    "_params_dict": payloads[rn - 1]["params_dict"],
                                }
                            completed_count += 1
                            self._emit_progress(completed_count, total_runs)
                    except KeyboardInterrupt:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

                # ── Replay results in run_num order for logging ──
                for run_num in range(1, total_runs + 1):
                    result = raw_results[run_num]
                    params_dict = result["_params_dict"]
                    energy = params_dict["energy"]
                    transv_offset_frac = params_dict.get("transv_offset_frac", 0.0)
                    sweep_overrides = {
                        k: v
                        for k, v in params_dict.items()
                        if k not in _positional_keys
                    }
                    helper_params = {
                        **params_dict,
                        "transverse_offset_fraction": transv_offset_frac,
                    }
                    run_params = resolve_sweep_run_parameters(
                        self.config, helper_params
                    )
                    if run_params is None:
                        raise ValueError("Sweep run parameters are missing energy")
                    rider_m_particle = run_params.rider_m_particle
                    rider_transv_dist = run_params.rider_transv_dist

                    self.results.append(result)

                    if not result.get("success"):
                        failed_count += 1
                        if "error_details" in result and not result.get("success"):
                            for line in build_exception_sweep_run_log_lines(
                                run_num=run_num,
                                total_runs=total_runs,
                                error=Exception(result.get("error", "unknown")),
                                error_detail=result.get("error_details", ""),
                            ):
                                self._log(line)
                        else:
                            error_msg = result.get("error", "Unknown error")
                            self._log(
                                f"  [FAILED] Run {run_num}/{total_runs}: {error_msg}"
                            )
                    else:
                        metrics = result.get("metrics", {})
                        log_output = build_successful_sweep_run_log(
                            run_num=run_num,
                            total_runs=total_runs,
                            metrics=metrics,
                            rest_energy_mev=rider_m_particle * AMU_TO_MEV,
                            param_names=param_names,
                            energy=energy,
                            rider_transv_dist=rider_transv_dist,
                            sweep_overrides=sweep_overrides,
                            default_driver_energy_gev=self.config.driver_energy_gev,
                        )
                        if use_truncated_logging or use_full_debug:
                            for line in log_output.optimization_lines:
                                self._log_line(line)
                            if use_full_debug:
                                for line in log_output.detail_lines:
                                    self._log(line)
                            self._log(log_output.compact_line)

            else:
                # ── Sequential path (unchanged behavior) ──
                run_num = 0
                for param_combo in all_combos:
                    run_num += 1
                    params_dict = dict(zip(param_names, param_combo))

                    aperture = params_dict.get("aperture", 0.001)
                    energy = params_dict["energy"]
                    start_z = params_dict["start_z"]
                    transv_offset_frac = params_dict.get("transv_offset_frac", 0.0)

                    sweep_overrides = {
                        k: v
                        for k, v in params_dict.items()
                        if k not in _positional_keys
                    }
                    helper_params = {
                        **params_dict,
                        "transverse_offset_fraction": transv_offset_frac,
                    }
                    run_params = resolve_sweep_run_parameters(
                        self.config, helper_params
                    )
                    if run_params is None:
                        raise ValueError("Sweep run parameters are missing energy")

                    rider_m_particle = run_params.rider_m_particle
                    rider_transv_dist = run_params.rider_transv_dist

                    if use_full_debug:
                        for line in build_full_debug_parameter_log_lines(
                            self.config,
                            run_params,
                            run_num=run_num,
                            total_runs=total_runs,
                            params_dict=helper_params,
                        ):
                            self._log(line)

                    # ── Run integration ──
                    try:
                        result = self._run_single_integration(
                            aperture=aperture,
                            energy_gev=energy,
                            start_z=start_z,
                            transv_offset_frac=transv_offset_frac,
                            run_num=run_num,
                            total_runs=total_runs,
                            sweep_overrides=sweep_overrides,
                            emit_run_diagnostics=use_full_debug,
                            emit_run_summary=not use_no_logging,
                        )

                        result["run_number"] = run_num
                        if result.get("parameters") is None:
                            result["parameters"] = {}
                        result["parameters"].update(params_dict)
                        self.results.append(result)

                        if not result["success"]:
                            failed_count += 1
                            error_msg = result.get("error", "Unknown error")
                            self._log(
                                f"  [FAILED] Run {run_num}/{total_runs}: {error_msg}"
                            )

                    except Exception as e:
                        failed_count += 1
                        error_detail = _traceback.format_exc()
                        for line in build_exception_sweep_run_log_lines(
                            run_num=run_num,
                            total_runs=total_runs,
                            error=e,
                            error_detail=error_detail,
                        ):
                            self._log(line)
                        self.results.append(
                            build_exception_sweep_run_record(
                                run_num=run_num,
                                error=e,
                                error_detail=error_detail,
                                params_dict=params_dict,
                            )
                        )
                        result = self.results[-1]

                    # ── Log results ──
                    if result.get("success"):
                        metrics = result.get("metrics", {})
                        log_output = build_successful_sweep_run_log(
                            run_num=run_num,
                            total_runs=total_runs,
                            metrics=metrics,
                            rest_energy_mev=rider_m_particle * AMU_TO_MEV,
                            param_names=param_names,
                            energy=energy,
                            rider_transv_dist=rider_transv_dist,
                            sweep_overrides=sweep_overrides,
                            default_driver_energy_gev=self.config.driver_energy_gev,
                        )

                        if use_truncated_logging or use_full_debug:
                            # Keep metric/start/compact lines for live plotting.
                            for line in log_output.optimization_lines:
                                self._log_line(line)
                            if use_full_debug:
                                for line in log_output.detail_lines:
                                    self._log(line)
                            self._log(log_output.compact_line)

                    self._emit_progress(run_num, total_runs)

            # ── Save results ──
            elapsed_time = (time.time() - start_time) if start_time is not None else 0.0

            self._log("")
            self._log("=" * 80)
            self._log("SWEEP COMPLETE")
            self._log("=" * 80)
            self._log(f"Total runs: {total_runs}")
            self._log(f"Successful: {total_runs - failed_count}")
            self._log(f"Failed: {failed_count}")
            self._log(f"Elapsed time: {elapsed_time:.1f}s ({elapsed_time / 60:.1f}min)")
            self._log("=" * 80)

            if self.config.save_results:
                results_path = self.output_dir / "sweep_results.json"
                with open(results_path, "w") as f:
                    json.dump(
                        build_sweep_results_payload(
                            config=self.config,
                            param_grids=param_grids,
                            total_runs=total_runs,
                            successful=total_runs - failed_count,
                            failed=failed_count,
                            elapsed_time_seconds=elapsed_time,
                            results=self.results,
                        ),
                        f,
                        indent=2,
                    )

                self._log("")
                self._log(f"Results saved to: {results_path}")

                from optimization.result_io import relocate_incomplete_sweep

                relocated = relocate_incomplete_sweep(
                    self.output_dir,
                    min_runs=100,
                    log_fn=self._log,
                )
                if relocated:
                    self.output_dir = relocated
            else:
                self._log("")
                self._log("Result saving disabled (save_results=False)")

            return True

        except KeyboardInterrupt:
            self._log("")
            self._log("[INFO] Sweep interrupted by user")

            # Save partial results before relocating
            partial_path = self.output_dir / "sweep_results.json"
            if not partial_path.exists() and self.results:
                try:
                    elapsed_time = (
                        (time.time() - start_time) if start_time is not None else 0.0
                    )
                    with open(partial_path, "w") as f:
                        json.dump(
                            build_interrupted_sweep_results_payload(
                                config=self.config,
                                total_runs=len(self.results),
                                elapsed_time_seconds=elapsed_time,
                                results=self.results,
                            ),
                            f,
                            indent=2,
                        )
                    self._log(f"[INFO] Partial results saved to: {partial_path}")
                except Exception as save_error:
                    self._log(
                        f"[WARNING] Failed to save partial results to "
                        f"{partial_path}: {save_error}"
                    )

            # Move to archive/incomplete if below minimum run threshold
            from optimization.result_io import relocate_incomplete_sweep

            relocate_incomplete_sweep(
                self.output_dir,
                min_runs=100,
                log_fn=self._log,
            )
            return False
        except Exception as e:
            self._log("")
            self._log(f"[ERROR] {e}")
            import traceback

            for line in traceback.format_exc().split("\n"):
                if line:
                    self._log(f"  {line}")
            return False
        finally:
            restore_run_logging_policy(self.config, logging_policy)
            if self.log_file is not None:
                self.log_file.close()


# ---------------------------------------------------------------------------
# Config conversion (unchanged)
# ---------------------------------------------------------------------------


def _convert_json_config_to_dataclass(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON config format to OptimizationConfig dataclass format.

    JSON uses separate min/max/points fields (e.g., aperture_min, aperture_max),
    while OptimizationConfig uses tuple ranges (e.g., aperture_range).

    This also maps *fixed* (non-swept) sweep_parameters to the corresponding
    OptimizationConfig scalar fields so that the CLI sweep runner uses the
    correct particle parameters instead of hard-coded defaults.
    """
    converted = dict(config_dict)
    if "mode" in converted:
        converted["mode"] = normalize_sweep_or_optimization_mode(converted["mode"])

    # Convert simulation_type string to SimulationType enum
    if "simulation_type" in converted and isinstance(converted["simulation_type"], str):
        sim_type_str = converted["simulation_type"].upper()
        if sim_type_str == "CONDUCTING_WALL":
            converted["simulation_type"] = SimulationType.CONDUCTING_WALL
        elif sim_type_str == "SWITCHING_WALL":
            converted["simulation_type"] = SimulationType.SWITCHING_WALL
        elif sim_type_str == "BUNCH_TO_BUNCH":
            converted["simulation_type"] = SimulationType.BUNCH_TO_BUNCH

    # Convert aperture fields
    if "aperture_min" in converted and "aperture_max" in converted:
        converted["aperture_range"] = (
            converted.pop("aperture_min"),
            converted.pop("aperture_max"),
        )

    # Convert energy fields
    if "energy_min" in converted and "energy_max" in converted:
        converted["energy_range"] = (
            converted.pop("energy_min"),
            converted.pop("energy_max"),
        )

    # Convert wall_z sweep fields
    if "wall_z_range" in converted and converted["wall_z_range"] is not None:
        wall_z_range = converted["wall_z_range"]
        if isinstance(wall_z_range, list) and len(wall_z_range) == 2:
            converted["wall_z_range"] = tuple(wall_z_range)

    # Map rider_stripped_ions top-level key -> stripped_ions (OptimizationConfig name)
    if "rider_stripped_ions" in converted and "stripped_ions" not in converted:
        converted["stripped_ions"] = float(converted.pop("rider_stripped_ions"))
    elif "rider_stripped_ions" in converted:
        converted.pop("rider_stripped_ions")

    # Map rider_offset / driver_offset top-level keys
    if "rider_offset_x" in converted:
        converted["transv_offset_x"] = float(converted.pop("rider_offset_x"))
    if "rider_offset_y" in converted:
        converted["transv_offset_y"] = float(converted.pop("rider_offset_y"))
    if "driver_offset_x" in converted:
        converted["driver_transv_offset_x"] = float(converted.pop("driver_offset_x"))
    if "driver_offset_y" in converted:
        converted["driver_transv_offset_y"] = float(converted.pop("driver_offset_y"))
    if "rider_transverse_geometry" in converted:
        converted["transverse_geometry"] = converted.pop("rider_transverse_geometry")

    # Map auto_steps_distance to auto_steps_distance_past_wall
    if (
        "auto_steps_distance" in converted
        and "auto_steps_distance_past_wall" not in converted
    ):
        converted["auto_steps_distance_past_wall"] = float(
            converted.pop("auto_steps_distance")
        )
    elif "auto_steps_distance" in converted:
        converted.pop("auto_steps_distance")

    particle_loss_payload = converted.pop("particle_loss", None)
    if isinstance(particle_loss_payload, dict):
        _particle_loss_field_map = {
            "enabled": "particle_loss_enabled",
            "loss_radius_mm": "particle_loss_radius_mm",
            "conducting_wall_aperture_loss_enabled": "particle_loss_conducting_wall_aperture_loss_enabled",
            "initial_radial_quantile": "particle_loss_initial_radial_quantile",
            "initial_radial_multiplier": "particle_loss_initial_radial_multiplier",
            "initial_radial_margin_mm": "particle_loss_initial_radial_margin_mm",
        }
        for source_key, target_key in _particle_loss_field_map.items():
            if source_key in particle_loss_payload:
                converted[target_key] = particle_loss_payload[source_key]

    pseudo_grid_payload = converted.pop("pseudo_grid", None)
    if isinstance(pseudo_grid_payload, dict):
        _pseudo_grid_field_map = {
            "enabled": "pseudo_grid_enabled",
            "active_rider_count": "pseudo_grid_active_rider_count",
            "active_driver_count": "pseudo_grid_active_driver_count",
            "passive_neighbor_count": "pseudo_grid_passive_neighbor_count",
            "coverage_strategy": "pseudo_grid_coverage_strategy",
            "coverage_space": "pseudo_grid_coverage_space",
            "pair_reuse_window": "pseudo_grid_pair_reuse_window",
            "source_weighting_mode": "pseudo_grid_source_weighting_mode",
            "loss_tracking_enabled": "pseudo_grid_loss_tracking_enabled",
            "causal_history_pruning_enabled": "pseudo_grid_causal_history_pruning_enabled",
            "causal_history_safety_margin_steps": "pseudo_grid_causal_history_safety_margin_steps",
        }
        for source_key, target_key in _pseudo_grid_field_map.items():
            if source_key in pseudo_grid_payload:
                converted[target_key] = pseudo_grid_payload[source_key]

    smearing_payload = converted.pop("macroparticle_smearing", None)
    if isinstance(smearing_payload, dict):
        _smearing_field_map = {
            "enabled": "macroparticle_smearing_enabled",
            "subcharge_count": "macroparticle_smearing_subcharge_count",
            "sigma_multiplier": "macroparticle_smearing_sigma_multiplier",
            "position_sigma_mm": "macroparticle_smearing_position_sigma_mm",
            "longitudinal_sigma_mm": "macroparticle_smearing_longitudinal_sigma_mm",
            "momentum_sigma_amu_mm_ns": "macroparticle_smearing_momentum_sigma_amu_mm_ns",
            "use_position_errors": "macroparticle_smearing_use_position_errors",
            "use_momentum_errors": "macroparticle_smearing_use_momentum_errors",
            "use_centroid_errors": "macroparticle_smearing_use_centroid_errors",
            "use_internal_cloud": "macroparticle_smearing_use_internal_cloud",
            "apply_to_active_observers": "macroparticle_smearing_apply_to_active_observers",
            "apply_to_active_sources": "macroparticle_smearing_apply_to_active_sources",
            "apply_to_passive_sources": "macroparticle_smearing_apply_to_passive_sources",
            "apply_to_passive_updates": "macroparticle_smearing_apply_to_passive_updates",
            "seed": "macroparticle_smearing_seed",
            "refresh_policy": "macroparticle_smearing_refresh_policy",
        }
        for source_key, target_key in _smearing_field_map.items():
            if source_key in smearing_payload:
                value = smearing_payload[source_key]
                if source_key == "refresh_policy" and isinstance(value, str):
                    value = value.replace("-", "_")
                converted[target_key] = value

    driver_train_payload = converted.pop("driver_train", None)
    if isinstance(driver_train_payload, dict):
        _driver_train_field_map = {
            "enabled": "driver_train_enabled",
            "bunch_count": "driver_train_bunch_count",
            "z_spacing_mm": "driver_train_z_spacing_mm",
            "z_offsets_mm": "driver_train_z_offsets_mm",
            "prehistory_steps": "driver_train_prehistory_steps",
            "preserve_prehistory_in_output": "driver_train_preserve_prehistory_in_output",
        }
        for source_key, target_key in _driver_train_field_map.items():
            if source_key in driver_train_payload:
                converted[target_key] = driver_train_payload[source_key]

    # Convert sweep_parameters to appropriate ranges and fixed values
    sweep_params = converted.get("sweep_parameters", {})

    _fixed_field_map_rider = {
        "rider_m_particle": "m_particle",
        "rider_charge_sign": "charge_sign",
        "rider_pcount": "pcount",
        "rider_transv_mom": "transv_mom",
        "rider_transv_dist": "transv_dist",
        "rider_long_dist": "long_dist",
        "rider_stripped_ions": "stripped_ions",
        "macroparticle_charge_multiplier": "macroparticle_charge_multiplier",
        "macroparticle_sigma_multiplier": "macroparticle_sigma_multiplier",
    }
    _fixed_field_map_driver = {
        "driver_m_particle": "driver_m_particle",
        "driver_charge_sign": "driver_charge_sign",
        "driver_pcount": "driver_pcount",
        "driver_transv_mom": "driver_transv_mom",
        "driver_transv_dist": "driver_transv_dist",
        "driver_long_dist": "driver_long_dist",
        "driver_transverse_geometry": "driver_transverse_geometry",
        "driver_starting_distance": "driver_starting_distance",
        "driver_energy_gev": "driver_energy_gev",
        "driver_stripped_ions": "driver_stripped_ions",
    }

    _range_field_map_rider = {
        "rider_m_particle": "particle_mass_range",
        "rider_charge_sign": "particle_charge_range",
        "rider_pcount": "particle_count_range",
        "rider_transv_mom": "transverse_momentum_range",
        "rider_transv_dist": "transverse_spread_range",
        "rider_stripped_ions": "rider_stripped_ions_range",
        "macroparticle_charge_multiplier": "macroparticle_charge_range",
        "macroparticle_sigma_multiplier": "macroparticle_sigma_range",
    }
    _range_field_map_driver = {
        "driver_m_particle": "driver_mass_range",
        "driver_charge_sign": "driver_charge_sign_range",
        "driver_pcount": "driver_pcount_range",
        "driver_transv_mom": "driver_transv_mom_range",
        "driver_transv_dist": "driver_transv_dist_range",
        "driver_starting_distance": "driver_starting_distance_range",
        "driver_energy_gev": "driver_energy_range",
        "driver_stripped_ions": "driver_stripped_ions_range",
    }

    all_param_names = list(_fixed_field_map_rider.keys()) + list(
        _fixed_field_map_driver.keys()
    )
    all_range_maps = {**_range_field_map_rider, **_range_field_map_driver}
    all_fixed_maps = {**_fixed_field_map_rider, **_fixed_field_map_driver}

    for param_name in all_param_names:
        if param_name not in sweep_params:
            continue
        param_config = sweep_params[param_name]

        if (
            param_config.get("enabled")
            and "min" in param_config
            and "max" in param_config
        ):
            # Swept parameter → range + points
            if param_name in all_range_maps:
                field_name = all_range_maps[param_name]
                min_val = float(param_config["min"])
                max_val = float(param_config["max"])
                if param_name == "driver_energy_gev":
                    min_val = abs(min_val)
                    max_val = abs(max_val)
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                converted[field_name] = (min_val, max_val)
                points_field = field_name.replace("_range", "_points")
                if "points" in param_config:
                    converted[points_field] = int(param_config["points"])
                if "log" in param_config:
                    log_field = field_name.replace("_range", "_log_scale")
                    converted[log_field] = bool(param_config["log"])
        else:
            # Fixed (disabled) parameter → scalar field
            if "fixed_value" in param_config and param_name in all_fixed_maps:
                scalar_field = all_fixed_maps[param_name]
                raw_val = param_config["fixed_value"]
                if param_name in ("rider_pcount", "driver_pcount"):
                    converted[scalar_field] = int(float(raw_val))
                else:
                    converted[scalar_field] = float(raw_val)

    # Remove sweep_parameters from converted dict as it's been processed
    converted.pop("sweep_parameters", None)

    # Remove fields that exist in JSON but not in OptimizationConfig dataclass
    fields_to_remove = [
        "timestep_mode",
        "auto_steps_distance",
    ]
    for field in fields_to_remove:
        converted.pop(field, None)

    return converted


def run_sweep_from_config(
    config_path: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
    verbosity_overrides: Optional[Dict[str, Any]] = None,
    workers: Optional[int] = None,
) -> bool:
    """Run a parameter sweep from a configuration file.

    Parameters
    ----------
    config_path : Path
        Path to sweep configuration JSON file
    output_dir : Path, optional
        Output directory. If None, auto-generated from config name and timestamp
    verbose : bool, optional
        Whether to print progress messages
    verbosity_overrides : Dict[str, Any], optional
        Dictionary of verbosity settings to override config values.
        Supported keys: 'log_verbosity', 'self_consistency_verbosity', 'adaptive_timestep_debug'
    workers : int, optional
        Number of parallel worker processes. None or 1 runs sequentially.

    Returns
    -------
    bool
        True if sweep completed successfully
    """
    # Load config
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Convert JSON format to dataclass format
    converted_dict = _convert_json_config_to_dataclass(config_dict)

    # Filter to only include valid OptimizationConfig fields
    from dataclasses import fields

    valid_fields = {f.name for f in fields(OptimizationConfig)}
    filtered_dict = {k: v for k, v in converted_dict.items() if k in valid_fields}

    # Create OptimizationConfig
    config = OptimizationConfig(**filtered_dict)

    # Apply verbosity overrides from CLI arguments
    if verbosity_overrides:
        for key, value in verbosity_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                if verbose:
                    print(f"[INFO] Overriding {key} from CLI: {value}", flush=True)

    if config.mode == "optimization":
        from lw_integrator.headless_optimization_runner import (
            run_headless_optimization_config,
        )

        optimization_output_dir = (
            output_dir if output_dir is not None else Path(config.output_dir)
        )
        return run_headless_optimization_config(
            config,
            output_dir=optimization_output_dir,
            config_path=config_path,
            verbose=verbose,
        )

    # Determine output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = config_path.stem
        output_dir = Path(config.output_dir) / f"{timestamp}_{config_name}"

    effective_workers = workers
    if effective_workers is None:
        effective_workers = getattr(config, "workers", 1)

    # Create and run sweep
    runner = SweepRunner(
        config,
        output_dir,
        verbose=verbose,
        workers=effective_workers,
    )
    return runner.run()
