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

import itertools
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
from optimization.single_integration_helpers import (
    build_integration_metrics,
    build_single_integration_setup,
    calculate_rider_starting_pz,
)
from optimization.simulation_type_helpers import is_bunch_to_bunch
from optimization.sweep_helpers import build_config_parameter_grids
from optimization.sweep_run_helpers import (
    build_full_debug_parameter_log_lines,
    resolve_sweep_run_parameters,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMU_TO_MEV = 931.494


class SweepRunner:
    """Execute parameter sweeps from configuration files without GUI.

    This runner delegates all integration work to ``run_testbed``, the same
    function used by the GUI.  This guarantees identical particle initialization,
    self-consistency configuration, adaptive timestep handling, metric extraction,
    and debug logging between the two interfaces.
    """

    def __init__(
        self, config: OptimizationConfig, output_dir: Path, verbose: bool = True
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.results: List[Dict[str, Any]] = []
        self.log_file = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Log a message to stdout and log file with [OPTIMIZATION] prefix."""
        print(f"[OPTIMIZATION] {message}", flush=True)
        if self.log_file is not None:
            self.log_file.write(f"[OPTIMIZATION] {message}\n")
            self.log_file.flush()

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
    ) -> Dict[str, Any]:
        """Run a single integration via run_testbed (same path as GUI).

        This method constructs a SimulationOptions, calls run_testbed(),
        and extracts metrics from the RunResult — identical to the GUI's
        OptimizationPlugin._run_single_integration.
        """
        if sweep_overrides is None:
            sweep_overrides = {}

        # ── Resolve rider parameters (sweep overrides > config) ──
        rider_m_particle = sweep_overrides.get(
            "rider_m_particle", self.config.m_particle
        )
        rider_charge_sign = sweep_overrides.get(
            "rider_charge_sign", self.config.charge_sign
        )
        rider_pcount = int(sweep_overrides.get("rider_pcount", self.config.pcount))
        rider_transv_mom = sweep_overrides.get(
            "rider_transv_mom", self.config.transv_mom
        )
        rider_transv_dist = sweep_overrides.get(
            "rider_transv_dist", self.config.transv_dist
        )
        rider_stripped_ions = sweep_overrides.get(
            "rider_stripped_ions", self.config.stripped_ions
        )
        macro_charge_mult = sweep_overrides.get(
            "macroparticle_charge_multiplier",
            self.config.macroparticle_charge_multiplier,
        )
        macro_sigma_mult = sweep_overrides.get(
            "macroparticle_sigma_multiplier", self.config.macroparticle_sigma_multiplier
        )

        # ── Compute timestep and steps (same logic as before) ──
        transv_offset = transv_offset_frac * aperture

        # Determine steps
        if self.config.auto_steps:
            if self.config.timestep_strategy == "auto_distance":
                preliminary_timestep = calculate_auto_timestep(
                    start_z=start_z,
                    wall_z=self.config.wall_z,
                    distance_past_wall=self.config.auto_steps_distance_past_wall,
                    particle_energy_gev=energy_gev,
                    particle_mass_amu=rider_m_particle,
                    target_steps=self.config.auto_steps_target,
                )
                steps = calculate_auto_steps(
                    start_z=start_z,
                    wall_z=self.config.wall_z,
                    distance_past_wall=self.config.auto_steps_distance_past_wall,
                    timestep=preliminary_timestep,
                    particle_energy_gev=energy_gev,
                    particle_mass_amu=rider_m_particle,
                )
            else:
                steps = calculate_auto_steps(
                    start_z=start_z,
                    wall_z=self.config.wall_z,
                    distance_past_wall=self.config.auto_steps_distance_past_wall,
                    timestep=self.config.timestep,
                    particle_energy_gev=energy_gev,
                    particle_mass_amu=rider_m_particle,
                )
        else:
            steps = self.config.steps

        # Calculate timestep
        driver_start_z = 1000.0  # default for non-BUNCH_TO_BUNCH
        if is_bunch_to_bunch(self.config.simulation_type):
            driver_start_z = sweep_overrides.get(
                "driver_starting_distance", self.config.driver_starting_distance
            )

        original_steps = self.config.steps
        self.config.steps = steps
        timestep = self.config.calculate_timestep_for_energy(
            energy_gev=energy_gev,
            start_z=start_z,
            wall_z=self.config.wall_z,
            driver_start_z=driver_start_z,
            m_particle_amu=rider_m_particle,
        )
        self.config.steps = original_steps

        # ── Log timestep calculation ──
        rest_energy_mev = rider_m_particle * AMU_TO_MEV
        if is_bunch_to_bunch(self.config.simulation_type):
            gamma = (energy_gev * 1e3) / rest_energy_mev + 1.0
        else:
            gamma = (energy_gev * 1e3) / rest_energy_mev
        if gamma < 1.0:
            gamma = 1.0
        beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.0

        print(
            f"[OPTIMIZATION]   [TIMESTEP] Run {run_num} strategy '{self.config.timestep_strategy}':",
            flush=True,
        )
        print(
            f"[OPTIMIZATION]     E={energy_gev:.4f} GeV, m={rider_m_particle:.4e} amu",
            flush=True,
        )
        print(f"[OPTIMIZATION]     gamma={gamma:.2f}, beta={beta:.8f}", flush=True)
        print(
            f"[OPTIMIZATION]     timestep h={timestep:.4e} ns (proper time = dt/gamma)",
            flush=True,
        )
        print(f"[OPTIMIZATION]     steps={steps}", flush=True)

        if self.config.timestep_strategy == "auto_distance":
            distance_per_step = beta * gamma * C_MMNS * timestep
            expected_total = distance_per_step * steps
            print(
                f"[OPTIMIZATION]     distance_per_step = β·γ·c·h = {distance_per_step:.4f} mm",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     expected_total_distance = {expected_total:.2f} mm",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     wall_z={self.config.wall_z:.2f} mm, start_z={start_z:.2f} mm",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     distance_to_wall = {abs(self.config.wall_z - start_z):.2f} mm",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     distance_past_wall={self.config.auto_steps_distance_past_wall:.2f} mm",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     target_steps={self.config.auto_steps_target}",
                flush=True,
            )

        # Log [START] line
        if aperture >= 1.0:
            aperture_str = f"{aperture:.1f}"
        elif aperture >= 0.01:
            aperture_str = f"{aperture:.4f}"
        else:
            aperture_str = f"{aperture:.6f}"

        print(
            f"[OPTIMIZATION] [START] Run {run_num}/{total_runs}: a={aperture_str}mm, E={energy_gev:.2f}GeV",
            flush=True,
        )
        print(
            f"[OPTIMIZATION]   [PARAMS] z={start_z:.2f}mm, h={timestep:.4e}ns, N={steps}",
            flush=True,
        )

        # ── Build driver_params dict if BUNCH_TO_BUNCH ──
        driver_params = None
        if is_bunch_to_bunch(self.config.simulation_type):
            d_m = sweep_overrides.get(
                "driver_m_particle", self.config.driver_m_particle
            )
            d_charge = sweep_overrides.get(
                "driver_charge_sign", self.config.driver_charge_sign
            )
            d_pcount = int(
                sweep_overrides.get("driver_pcount", self.config.driver_pcount)
            )
            d_transv_mom = sweep_overrides.get(
                "driver_transv_mom", self.config.driver_transv_mom
            )
            d_transv_dist = sweep_overrides.get(
                "driver_transv_dist", self.config.driver_transv_dist
            )
            d_start_dist = sweep_overrides.get(
                "driver_starting_distance", self.config.driver_starting_distance
            )
            d_stripped = sweep_overrides.get(
                "driver_stripped_ions", self.config.driver_stripped_ions
            )
            d_energy_gev = sweep_overrides.get(
                "driver_energy_gev", self.config.driver_energy_gev
            )

            # Determine Pz sign from driver direction
            driver_negative = getattr(self.config, "driver_direction", "-z") == "-z"
            pz_sign = -1.0 if driver_negative else 1.0
            driver_pz_magnitude = calculate_rider_starting_pz(
                abs(d_energy_gev), d_m, SimulationType.BUNCH_TO_BUNCH
            )

            driver_params = {
                "starting_distance": d_start_dist,
                "transv_mom": d_transv_mom,
                "transv_dist": d_transv_dist,
                "transv_offset_x": 0.0,
                "transv_offset_y": 0.0,
                "m_particle": d_m,
                "charge_sign": d_charge,
                "pcount": d_pcount,
                "stripped_ions": d_stripped,
                "starting_Pz": pz_sign * driver_pz_magnitude,
            }

            dir_label = "\u2212z" if driver_negative else "+z"
            print(
                f"[OPTIMIZATION]   [DRIVER] energy={d_energy_gev:.4f} GeV, "
                f"m={d_m:.4e} amu, Pz={driver_params['starting_Pz']:.4e} ({dir_label}), "
                f"stripped={d_stripped:.2e}, pcount={d_pcount}",
                flush=True,
            )

        # ── Build SimulationOptions (same dataclass the GUI uses) ──
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        run_output_dir = self.output_dir / f"_temp_run_{run_num}_{timestamp}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        setup = build_single_integration_setup(
            self.config,
            aperture=aperture,
            energy_gev=energy_gev,
            start_z=start_z,
            transv_offset=transv_offset,
            timestep=timestep,
            steps=steps,
            run_output_dir=run_output_dir,
            run_num=run_num,
            driver_params=driver_params,
            rider_m_particle=rider_m_particle,
            rider_charge_sign=rider_charge_sign,
            rider_pcount=rider_pcount,
            rider_transv_mom=rider_transv_mom,
            rider_transv_dist=rider_transv_dist,
            rider_stripped_ions=rider_stripped_ions,
            macroparticle_charge_multiplier=macro_charge_mult,
            macroparticle_sigma_multiplier=macro_sigma_mult,
        )
        options = setup.options

        # ── Progress + log callbacks (same format as GUI) ──
        def progress_callback(current: int, total: int, _run_id=run_num):
            if total <= 1000:
                log_interval = max(1, total // 10)
            else:
                log_interval = max(100, total // 20)
            if current % log_interval == 0 or current == total:
                print(
                    f"[OPTIMIZATION]     [PROGRESS] Run {_run_id}: step {current}/{total} "
                    f"({100 * current // total}%)",
                    flush=True,
                )

        _verbose_keywords = [
            "Particle",
            "converged",
            "Mass-shell error",
            "γ_velocity",
            "γ_energy",
            "γ_mass_shell",
            "Energy jump detected",
            "Reducing timestep",
            "Proximity refinement",
            "Cooldown mode",
            "Probing stability",
            "Returning to normal timestep",
            "Stable",
            "Unstable",
            "Minimum timestep reached",
            "Max refinement attempts",
        ]

        def _verbose_log(message: str) -> None:
            if any(kw in message for kw in _verbose_keywords):
                print(f"[OPTIMIZATION]     [VERBOSE] {message}", flush=True)

        log_callback: Optional[Callable[[str], None]] = None
        if (
            self.config.self_consistency_verbosity > 0
            or self.config.adaptive_timestep_debug
        ):
            log_callback = _verbose_log

        # ── Log stability settings (same as GUI) ──
        print(
            f"[OPTIMIZATION]   [CONFIG] Run {run_num} stability settings:", flush=True
        )
        print(
            f"[OPTIMIZATION]     smoothness_enabled: {self.config.smoothness_enabled}",
            flush=True,
        )
        if self.config.smoothness_enabled:
            print(
                f"[OPTIMIZATION]     smoothness_window_size: {self.config.smoothness_window_size}",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     smoothness_reject_on_violation: {self.config.smoothness_reject_on_violation}",
                flush=True,
            )

        if aperture < 0.1:
            print(
                f"[OPTIMIZATION]   [DIAGNOSTIC] Run {run_num}: Small aperture detected ({aperture:.6f} mm)",
                flush=True,
            )

        print(
            f"[OPTIMIZATION]   [DEBUG] Calling run_testbed for Run {run_num}...",
            flush=True,
        )

        # ── Call run_testbed (THE SAME function the GUI calls) ──
        try:
            result = run_testbed(
                options,
                log=log_callback,
                progress_callback=progress_callback,
            )
            print(
                f"[OPTIMIZATION]   [DEBUG] run_testbed completed for Run {run_num}",
                flush=True,
            )
        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": f"run_testbed failed: {e}\n{traceback.format_exc()}",
                "parameters": {
                    "aperture": aperture,
                    "energy_gev": energy_gev,
                    "start_z": start_z,
                    "transv_offset": transv_offset,
                },
            }
        finally:
            # Always clean up temp directory
            try:
                if run_output_dir.exists():
                    shutil.rmtree(run_output_dir)
            except Exception as cleanup_error:
                if self.verbose:
                    self._log(
                        f"[WARNING] Failed to remove temporary run directory "
                        f"{run_output_dir}: {cleanup_error}"
                    )

        # ── Check for halted run ──
        if result.halted_early:
            print(
                f"[OPTIMIZATION]   [WARNING] Run {run_num} halted early: {result.halt_reason}",
                flush=True,
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
                    "transv_offset": transv_offset,
                    "timestep": timestep,
                    "steps": steps,
                },
            }

        # ── Extract metrics from RunResult (same helper as GUI) ──
        print(
            f"[OPTIMIZATION]   [DEBUG] Extracting metrics for Run {run_num}...",
            flush=True,
        )
        metrics_outcome = build_integration_metrics(
            result,
            rider_m_particle=rider_m_particle,
            run_num=run_num,
        )
        metrics = metrics_outcome.metrics
        for line in metrics_outcome.log_lines:
            print(f"[OPTIMIZATION] {line}", flush=True)

        # ── Stability analysis (same as GUI) ──
        print(
            f"[OPTIMIZATION]   [DEBUG] Processing trajectory data for Run {run_num}...",
            flush=True,
        )

        if result.rider_trajectory is not None and self.config.smoothness_enabled:
            print(
                f"[OPTIMIZATION]   [DEBUG] Performing stability analysis for Run {run_num}...",
                flush=True,
            )
            traj = result.rider_trajectory
            smoothness_config = SmoothnessConfig(
                enabled=True,
                window_size=self.config.smoothness_window_size,
                oscillation_threshold=self.config.smoothness_oscillation_threshold,
                trend_smoothness_threshold=self.config.smoothness_trend_threshold,
                reject_on_violation=self.config.smoothness_reject_on_violation,
                max_allowed_violations=self.config.smoothness_max_violations,
            )
            smoothness_result = analyze_trajectory_smoothness(
                traj,
                smoothness_config,
                particle_mass_amu=rider_m_particle,
            )
            metrics["smoothness_passed"] = smoothness_result.passed
            metrics["smoothness_violations"] = len(smoothness_result.violations)

            if not smoothness_result.passed:
                print(
                    f"[OPTIMIZATION]   [WARNING] Stability check FAILED for Run {run_num}",
                    flush=True,
                )
                print(
                    f"[OPTIMIZATION]     Quality: {smoothness_result.quality_summary}",
                    flush=True,
                )
                if self.config.smoothness_reject_on_violation:
                    print(
                        f"[OPTIMIZATION]   [REJECT] Run {run_num} rejected due to numerical instability",
                        flush=True,
                    )
                    return {
                        "success": False,
                        "error": f"Smoothness violation: {len(smoothness_result.violations)} violations",
                        "parameters": {
                            "aperture": aperture,
                            "energy_gev": energy_gev,
                            "start_z": start_z,
                            "transv_offset": transv_offset,
                        },
                        "metrics": metrics,
                    }
            else:
                print(
                    f"[OPTIMIZATION]   [OK] Stability check PASSED for Run {run_num}: {smoothness_result.quality_summary}",
                    flush=True,
                )
        elif result.rider_trajectory is None:
            print(
                f"[OPTIMIZATION]   [WARNING] No trajectory data for Run {run_num}",
                flush=True,
            )
        elif not self.config.smoothness_enabled:
            print(
                f"[OPTIMIZATION]   [INFO] Stability analysis DISABLED for Run {run_num}",
                flush=True,
            )

        print(
            f"[OPTIMIZATION]   [DEBUG] _run_single_integration returning for Run {run_num}",
            flush=True,
        )

        return {
            "success": True,
            "parameters": {
                "aperture": aperture,
                "energy_gev": energy_gev,
                "start_z": start_z,
                "transv_offset": transv_offset,
                "timestep": timestep,
                "steps": steps,
            },
            "metrics": metrics,
        }

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

            self._log(f"Starting BLIND SWEEP (Grid Search): {total_runs} total runs")
            self._log(f"  Simulation type: {self.config.simulation_type}")

            for grid_key, grid_vals in param_grids.items():
                if len(grid_vals) > 1:
                    self._log(
                        f"  {grid_key}: {len(grid_vals)} points from {min(grid_vals):.4e} to {max(grid_vals):.4e}"
                    )
                else:
                    self._log(f"  {grid_key}: {grid_vals[0]:.4e} (fixed)")

            self._log(f"  Timestep strategy: {self.config.timestep_strategy}")
            if self.config.timestep_strategy == "auto_distance":
                self._log(
                    f"    Distance past wall: {self.config.auto_steps_distance_past_wall} mm"
                )
                self._log(
                    f"    Target steps for timestep calculation: {self.config.auto_steps_target}"
                )
                self._log(
                    "    All particles will travel to consistent z regardless of energy"
                )
            self._log(f"  z_cutoff_mode: {self.config.z_cutoff_mode}")

            # Log fixed particle parameters
            if is_bunch_to_bunch(self.config.simulation_type):
                self._log("")
                self._log("  Fixed rider parameters:")
                self._log(f"    m_particle: {self.config.m_particle:.4e} amu")
                self._log(f"    charge_sign: {self.config.charge_sign}")
                self._log(f"    pcount: {self.config.pcount}")
                self._log(f"    stripped_ions: {self.config.stripped_ions:.2e}")
                self._log(f"    transv_mom: {self.config.transv_mom:.4e}")
                self._log(f"    transv_dist: {self.config.transv_dist:.4e}")
                self._log("  Fixed driver parameters:")
                self._log(f"    m_particle: {self.config.driver_m_particle:.4e} amu")
                self._log(f"    charge_sign: {self.config.driver_charge_sign}")
                self._log(f"    pcount: {self.config.driver_pcount}")
                self._log(f"    stripped_ions: {self.config.driver_stripped_ions:.2e}")
                self._log(f"    energy_gev: {self.config.driver_energy_gev:.4f}")
                self._log(
                    f"    starting_distance: {self.config.driver_starting_distance:.2f}"
                )

            self._log("")
            self._log(f"Output directory: {self.output_dir}")
            self._log("")

            # ── Build iteration over all grid dimensions ──
            param_names = list(param_grids.keys())
            param_values_lists = [param_grids[name] for name in param_names]
            _positional_keys = {"aperture", "energy", "start_z", "transv_offset_frac"}

            # Run sweep
            start_time = time.time()
            run_num = 0
            failed_count = 0

            for param_combo in itertools.product(*param_values_lists):
                run_num += 1
                params_dict = dict(zip(param_names, param_combo))

                aperture = params_dict.get("aperture", 0.001)
                energy = params_dict["energy"]
                start_z = params_dict["start_z"]
                transv_offset_frac = params_dict.get("transv_offset_frac", 0.0)

                sweep_overrides = {
                    k: v for k, v in params_dict.items() if k not in _positional_keys
                }
                helper_params = {
                    **params_dict,
                    "transverse_offset_fraction": transv_offset_frac,
                }
                run_params = resolve_sweep_run_parameters(self.config, helper_params)
                if run_params is None:
                    raise ValueError("Sweep run parameters are missing energy")

                rider_m_particle = run_params.rider_m_particle
                rider_transv_dist = run_params.rider_transv_dist

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
                    )

                    result["run_number"] = run_num
                    if result.get("parameters") is None:
                        result["parameters"] = {}
                    result["parameters"].update(params_dict)
                    self.results.append(result)

                    if not result["success"]:
                        failed_count += 1
                        error_msg = result.get("error", "Unknown error")
                        self._log(f"  [FAILED] Run {run_num}/{total_runs}: {error_msg}")

                except Exception as e:
                    failed_count += 1
                    import traceback

                    error_detail = traceback.format_exc()
                    self._log(f"  [EXCEPTION] Run {run_num}/{total_runs}: {e}")
                    for line in error_detail.split("\n"):
                        if line:
                            self._log(f"    {line}")
                    self.results.append(
                        {
                            "run_number": run_num,
                            "success": False,
                            "error": f"{e}\n{error_detail}",
                            "parameters": dict(params_dict),
                        }
                    )
                    result = self.results[-1]

                # ── Log results ──
                if result.get("success"):
                    metrics = result.get("metrics", {})
                    gamma_initial = metrics.get(
                        "initial_gamma_mean", metrics.get("rider_gamma_initial", 1.0)
                    )
                    gamma_final = metrics.get(
                        "final_gamma_mean", metrics.get("rider_gamma_final", 1.0)
                    )
                    delta_gamma = gamma_final - gamma_initial
                    rest_energy_mev = rider_m_particle * AMU_TO_MEV
                    delta_e_mev = delta_gamma * rest_energy_mev

                    # Log metrics in format compatible with plotting script
                    print(
                        f"[OPTIMIZATION] max_percent_energy_gain: {metrics.get('max_percent_energy_gain', 0):.12e}%",
                        flush=True,
                    )
                    print(
                        f"[OPTIMIZATION] max_energy_gain: {metrics.get('max_energy_gain_gev', 0):.12e} GeV",
                        flush=True,
                    )
                    print(
                        f"[OPTIMIZATION] max_relative_gain: {metrics.get('max_relative_gain', 0):.12e}",
                        flush=True,
                    )
                    print(f"[OPTIMIZATION] delta_gamma: {delta_gamma:.12e}", flush=True)
                    print(
                        f"[OPTIMIZATION] delta_e_mev: {delta_e_mev:.12e} MeV",
                        flush=True,
                    )
                    print(f"[OPTIMIZATION] final_gamma: {gamma_final:.16f}", flush=True)
                    print(
                        f"[OPTIMIZATION] initial_gamma: {gamma_initial:.16f}",
                        flush=True,
                    )

                    self._log(f"  [RESULT] Run {run_num}/{total_runs}:")
                    self._log(f"    rider_gamma_initial: {gamma_initial:.16f}")
                    self._log(f"    rider_gamma_final: {gamma_final:.16f}")
                    self._log(f"    delta_gamma: {delta_gamma:.12e}")
                    self._log(f"    delta_e_mev: {delta_e_mev:.12e} MeV")
                    self._log(
                        f"    max_percent_energy_gain: {metrics.get('max_percent_energy_gain', 0):.12e}%"
                    )
                    self._log(
                        f"    max_energy_gain: {metrics.get('max_energy_gain_gev', 0):.12e} GeV"
                    )
                    self._log(
                        f"    max_relative_gain: {metrics.get('max_relative_gain', 0):.12e}"
                    )

                    # Compact summary line (like GUI format)
                    swept_params = []
                    if "energy" in param_names:
                        swept_params.append(f"initial_energy_gev={energy:.3g}")
                    if "rider_transv_dist" in param_names:
                        swept_params.append(
                            f"rider_transv_dist={rider_transv_dist:.3e}"
                        )
                    if "driver_energy_gev" in param_names:
                        swept_params.append(
                            f"driver_energy_gev={sweep_overrides.get('driver_energy_gev', self.config.driver_energy_gev):.3g}"
                        )
                    param_str = (
                        " ".join(swept_params) if swept_params else "fixed_params"
                    )

                    self._log(
                        f"Run #{run_num:4d} | {param_str} | "
                        f"ΔE={delta_e_mev:.3e} Δγ={delta_gamma:.3e} "
                        f"γ_i={gamma_initial:.2f} γ_f={gamma_final:.2f} | SUCCESS"
                    )

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

            # Save results to JSON
            results_path = self.output_dir / "sweep_results.json"
            with open(results_path, "w") as f:
                json.dump(
                    {
                        "config": {
                            "simulation_type": (
                                self.config.simulation_type.name
                                if hasattr(self.config.simulation_type, "name")
                                else str(self.config.simulation_type)
                            ),
                            "aperture_range": list(self.config.aperture_range),
                            "aperture_points": self.config.aperture_points,
                            "energy_range": list(self.config.energy_range),
                            "energy_points": self.config.energy_points,
                            "param_grids": {k: v for k, v in param_grids.items()},
                        },
                        "total_runs": total_runs,
                        "successful": total_runs - failed_count,
                        "failed": failed_count,
                        "elapsed_time_seconds": elapsed_time,
                        "results": self.results,
                    },
                    f,
                    indent=2,
                )

            self._log("")
            self._log(f"Results saved to: {results_path}")

            # Move to archive/incomplete if below minimum run threshold
            from optimization.result_io import relocate_incomplete_sweep

            relocated = relocate_incomplete_sweep(
                self.output_dir,
                min_runs=100,
                log_fn=self._log,
            )
            if relocated:
                self.output_dir = relocated

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
                            {
                                "config": {
                                    "simulation_type": (
                                        self.config.simulation_type.name
                                        if hasattr(self.config.simulation_type, "name")
                                        else str(self.config.simulation_type)
                                    ),
                                },
                                "total_runs": len(self.results),
                                "successful": len(self.results),
                                "failed": 0,
                                "elapsed_time_seconds": elapsed_time,
                                "interrupted": True,
                                "results": self.results,
                            },
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

    # Convert sweep_parameters to appropriate ranges and fixed values
    sweep_params = converted.get("sweep_parameters", {})

    _fixed_field_map_rider = {
        "rider_m_particle": "m_particle",
        "rider_charge_sign": "charge_sign",
        "rider_pcount": "pcount",
        "rider_transv_mom": "transv_mom",
        "rider_transv_dist": "transv_dist",
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
                print(f"[INFO] Overriding {key} from CLI: {value}", flush=True)

    # Determine output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = config_path.stem
        output_dir = Path(config.output_dir) / f"{timestamp}_{config_name}"

    # Create and run sweep
    runner = SweepRunner(config, output_dir, verbose=verbose)
    return runner.run()
