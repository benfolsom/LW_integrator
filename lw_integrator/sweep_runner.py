"""Headless sweep runner for CLI execution of parameter sweeps.

This module provides a standalone interface to run parameter sweeps without
requiring the GUI. It can be invoked from the command-line interface or
used programmatically.

Output Locations
----------------
When running sweeps via CLI, output is written to two locations:

1. **Results directory** (e.g., results/sweeps/YYYYMMDD_HHMMSS_configname/)
   - results.json: Parameter combinations and metrics
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

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.constants import C_MMNS
from core.debug_logger import initialize_debug_logging, set_logging_context
from core.integration_runner import retarded_integrator
from core.smoothness_analyzer import SmoothnessConfig, analyze_trajectory_smoothness
from core.types import (
    ChronoMatchingMode,
    SimulationType,
    StartupMode,
)
from input_output.bunch_initialization import create_bunch_from_energy
from optimization.config import (
    OptimizationConfig,
    calculate_auto_steps,
    calculate_auto_timestep,
)
from optimization.metrics import compute_trajectory_metrics


class SweepRunner:
    """Execute parameter sweeps from configuration files without GUI."""

    def __init__(
        self, config: OptimizationConfig, output_dir: Path, verbose: bool = True
    ):
        """Initialize sweep runner.

        Parameters
        ----------
        config : OptimizationConfig
            Sweep configuration
        output_dir : Path
            Directory for results output
        verbose : bool, optional
            Whether to print progress messages, by default True
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.results: List[Dict[str, Any]] = []
        self.log_file = None

    def _log(self, message: str) -> None:
        """Log a message to stdout and log file with [OPTIMIZATION] prefix."""
        # Always print to stdout with [OPTIMIZATION] prefix (captured by debug_logger to logcache)
        print(f"[OPTIMIZATION] {message}", flush=True)

        # Also write to the sweep.log file in the results directory
        if self.log_file is not None:
            self.log_file.write(f"[OPTIMIZATION] {message}\n")
            self.log_file.flush()

    def _generate_parameter_grids(self) -> Dict[str, List[float]]:
        """Generate parameter grids for sweep."""
        grids = {}

        # Aperture grid
        if self.config.aperture_points > 1:
            aper_min, aper_max = self.config.aperture_range
            if self.config.aperture_log_scale:
                grids["aperture"] = np.logspace(
                    np.log10(aper_min), np.log10(aper_max), self.config.aperture_points
                ).tolist()
            else:
                grids["aperture"] = np.linspace(
                    aper_min, aper_max, self.config.aperture_points
                ).tolist()
        else:
            grids["aperture"] = [self.config.aperture_range[0]]

        # Energy grid
        if self.config.energy_points > 1:
            e_min, e_max = self.config.energy_range
            if self.config.energy_log_scale:
                grids["energy"] = np.logspace(
                    np.log10(e_min), np.log10(e_max), self.config.energy_points
                ).tolist()
            else:
                grids["energy"] = np.linspace(
                    e_min, e_max, self.config.energy_points
                ).tolist()
        else:
            grids["energy"] = [self.config.energy_range[0]]

        # Starting z positions
        if (
            self.config.starting_z_positions
            and len(self.config.starting_z_positions) > 1
        ):
            grids["start_z"] = self.config.starting_z_positions
        elif (
            self.config.starting_z_range is not None
            and self.config.starting_z_points > 1
        ):
            grids["start_z"] = np.linspace(
                self.config.starting_z_range[0],
                self.config.starting_z_range[1],
                self.config.starting_z_points,
            ).tolist()
        else:
            # Default: particle starts before wall
            grids["start_z"] = [self.config.wall_z - 100.0]

        # Transverse offsets
        if (
            self.config.transverse_offset_fractions
            and len(self.config.transverse_offset_fractions) > 1
        ):
            grids["transv_offset_frac"] = self.config.transverse_offset_fractions
        else:
            grids["transv_offset_frac"] = [0.0]

        return grids

    def _run_single_integration(
        self,
        aperture: float,
        energy_gev: float,
        start_z: float,
        transv_offset_frac: float,
        run_num: int,
    ) -> Dict[str, Any]:
        """Run a single integration with given parameters.

        Parameters
        ----------
        aperture : float
            Aperture radius in mm
        energy_gev : float
            Particle energy in GeV
        start_z : float
            Starting z position in mm
        transv_offset_frac : float
            Transverse offset as fraction of aperture
        run_num : int
            Run number for tracking

        Returns
        -------
        Dict[str, Any]
            Result dictionary with metrics and trajectory info
        """
        # Calculate transverse offset
        transv_offset = transv_offset_frac * aperture

        # Calculate timestep based on strategy
        if self.config.timestep_strategy == "auto_distance":
            timestep = calculate_auto_timestep(
                start_z=start_z,
                wall_z=self.config.wall_z,
                distance_past_wall=self.config.target_distance_mm,
                particle_energy_gev=energy_gev,
                particle_mass_amu=self.config.m_particle,
            )
        else:
            timestep = self.config.timestep

        # Calculate steps if auto mode enabled
        if self.config.auto_steps:
            steps = calculate_auto_steps(
                start_z=start_z,
                wall_z=self.config.wall_z,
                distance_past_wall=self.config.auto_steps_distance_past_wall,
                timestep=timestep,
                particle_energy_gev=energy_gev,
                particle_mass_amu=self.config.m_particle,
            )
        else:
            steps = self.config.steps

        # Log timestep calculation details
        AMU_TO_MEV = 931.494
        rest_energy_mev = self.config.m_particle * AMU_TO_MEV
        gamma = (energy_gev * 1e3) / rest_energy_mev
        beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.0

        print(
            f"[OPTIMIZATION]   [TIMESTEP] Run {run_num} strategy '{self.config.timestep_strategy}':",
            flush=True,
        )
        print(
            f"[OPTIMIZATION]     E={energy_gev:.4f} GeV, m={self.config.m_particle:.4e} amu",
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
                f"[OPTIMIZATION]     target_distance={self.config.target_distance_mm:.2f} mm",
                flush=True,
            )

        print(
            f"[OPTIMIZATION]   [START] Run {run_num}/{run_num}: a={aperture:.4e}mm, E={energy_gev:.4f}GeV, z={start_z:.2f}mm, h={timestep:.4e}ns, N={steps}",
            flush=True,
        )
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

        # Build rider params
        AMU_TO_MEV = 931.494
        rest_energy_mev = self.config.m_particle * AMU_TO_MEV
        gamma = (energy_gev * 1e3) / rest_energy_mev

        rider_params = {
            "starting_distance": start_z,
            "transv_mom": self.config.transv_mom,
            "transv_dist": self.config.transv_dist,
            "m_particle": self.config.m_particle,
            "charge_sign": self.config.charge_sign,
            "pcount": int(self.config.pcount),
            "stripped_ions": self.config.stripped_ions,
            "starting_Pz": C_MMNS * np.sqrt(gamma * gamma - 1.0),
        }

        # Core params
        core_params = {
            "time_step": timestep,
            "wall_z": self.config.wall_z,
            "aperture_radius": aperture,
            "mean": 1.0e5,
            "cav_spacing": self.config.cavity_spacing,
            "z_cutoff": 0.0,
            "z_cutoff_mode": self.config.z_cutoff_mode,
        }

        # Driver params (for BUNCH_TO_BUNCH)
        driver_params = None
        driver_transv_offset = 0.0
        if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
            driver_gamma = (self.config.driver_energy_gev * 1e3) / (
                self.config.m_particle * AMU_TO_MEV
            )
            driver_params = {
                "starting_distance": self.config.wall_z + 1000.0,
                "transv_mom": 0.0,
                "transv_dist": -0.07998,
                "m_particle": 207.2,
                "charge_sign": 1.0,
                "pcount": 5,
                "stripped_ions": self.config.driver_stripped_ions,
                "starting_Pz": C_MMNS * np.sqrt(driver_gamma * driver_gamma - 1.0),
            }

        # Create particle states
        try:
            # Create rider bunch - use transverse_spread instead of transverse_radius
            rider_state, rest_energy_mev_rider = create_bunch_from_energy(
                kinetic_energy_mev=energy_gev * 1e3,
                mass_amu=rider_params["m_particle"],
                charge_sign=rider_params["charge_sign"],
                position_z=rider_params["starting_distance"],
                particle_count=rider_params["pcount"],
                transverse_spread=rider_params["transv_dist"],
                transverse_momentum=rider_params["transv_mom"],
                transverse_offset_x=transv_offset,
                transverse_offset_y=0.0,
            )

            # Set stripped ions
            rider_state["stripped_ions"] = np.full(
                rider_params["pcount"], rider_params["stripped_ions"]
            )

            # Create driver bunch if needed
            driver_state = None
            if driver_params is not None:
                driver_state, _ = create_bunch_from_energy(
                    kinetic_energy_mev=self.config.driver_energy_gev * 1e3,
                    mass_amu=driver_params["m_particle"],
                    charge_sign=driver_params["charge_sign"],
                    position_z=driver_params["starting_distance"],
                    particle_count=driver_params["pcount"],
                    transverse_spread=abs(driver_params["transv_dist"]),
                    transverse_momentum=driver_params["transv_mom"],
                    transverse_offset_x=driver_transv_offset,
                    transverse_offset_y=0.0,
                )
                driver_state["stripped_ions"] = np.full(
                    driver_params["pcount"], driver_params["stripped_ions"]
                )

        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": f"Failed to create particle states: {e}\n{traceback.format_exc()}",
                "parameters": {
                    "aperture": aperture,
                    "energy_gev": energy_gev,
                    "start_z": start_z,
                    "transv_offset": transv_offset,
                },
            }

        # Run core integration
        try:
            # Create progress callback for step-by-step logging
            def progress_callback(current_step: int, total_steps: int):
                if current_step % 100 == 0:
                    progress_pct = (current_step / total_steps) * 100
                    print(
                        f"[OPTIMIZATION]     [PROGRESS] Run {run_num}: step {current_step}/{total_steps} ({progress_pct:.0f}%)",
                        flush=True,
                    )

            # Create logger callback for verbose messages (only if adaptive debug enabled)
            def logger_callback(message: str):
                if self.config.adaptive_timestep_debug:
                    print(f"[OPTIMIZATION]     [VERBOSE] {message}", flush=True)

            # Build self-consistency config
            from core.self_consistency import SelfConsistencyConfig

            sc_config = None
            if self.config.self_consistency_enabled:
                sc_config = SelfConsistencyConfig(
                    enabled=True,
                    target_ms_tolerance=self.config.self_consistency_tolerance,
                    max_iterations=self.config.self_consistency_max_iterations,
                    verbosity=self.config.self_consistency_verbosity,
                    chrono_interpolate=self.config.self_consistency_chrono_interpolate,
                    chrono_tolerance=self.config.self_consistency_chrono_tolerance,
                    chrono_high_precision=self.config.self_consistency_chrono_high_precision,
                    chrono_adaptive_tolerance=self.config.self_consistency_chrono_adaptive_tolerance,
                )

            # Build adaptive timestep config
            from core.integration_runner import AdaptiveTimestepConfig

            adaptive_config = None
            if self.config.adaptive_timestep_enabled:
                adaptive_config = AdaptiveTimestepConfig(
                    enabled=True,
                    energy_jump_threshold=self.config.adaptive_timestep_threshold,
                    timestep_reduction_factor=self.config.adaptive_timestep_reduction_factor,
                    min_timestep_factor=self.config.adaptive_timestep_min_factor,
                )

            rider_trajectory, driver_trajectory = retarded_integrator(
                steps=steps,
                h_step=timestep,
                wall_z=core_params["wall_z"],
                aperture_radius=core_params["aperture_radius"],
                sim_type=self.config.simulation_type,
                init_rider=rider_state,
                init_driver=driver_state,
                mean=core_params["mean"],
                cav_spacing=core_params["cav_spacing"],
                z_cutoff=core_params["z_cutoff"],
                chrono_mode=ChronoMatchingMode.AVERAGED,
                startup_mode=StartupMode.COLD_START,
                image_subcharge_count=self.config.image_subcharge_count,
                use_conducting_image_weighting=self.config.use_image_weighting,
                self_consistency=sc_config,
                adaptive_timestep=adaptive_config,
                macroparticle_charge_multiplier=self.config.macroparticle_charge_multiplier,
                macroparticle_sigma_multiplier=self.config.macroparticle_sigma_multiplier,
                macroparticle_use_momentum_errors=self.config.macroparticle_use_momentum_errors,
                bunch_transv_dist=rider_params["transv_dist"],
                bunch_transv_mom=rider_params["transv_mom"],
                progress_callback=progress_callback
                if self.config.log_verbosity == "full"
                else None,
                logger=logger_callback if self.config.log_verbosity == "full" else None,
            )

            print(
                f"[OPTIMIZATION]   [DEBUG] run_testbed completed for Run {run_num}",
                flush=True,
            )

            # Check if trajectory is valid
            if rider_trajectory is None or len(rider_trajectory) == 0:
                return {
                    "success": False,
                    "error": "Empty trajectory",
                    "parameters": {
                        "aperture": aperture,
                        "energy_gev": energy_gev,
                        "start_z": start_z,
                        "transv_offset": transv_offset,
                    },
                }

            # Compute metrics
            try:
                metrics = compute_trajectory_metrics(
                    trajectory=rider_trajectory,
                    initial_state=rider_trajectory[0],
                    rest_energy_mev=rest_energy_mev,
                    aperture_z=self.config.wall_z,
                )
            except (KeyError, IndexError) as e:
                import traceback

                return {
                    "success": False,
                    "error": f"Failed to compute metrics: {e}. Trajectory length: {len(rider_trajectory)}, First state keys: {list(rider_trajectory[0].keys()) if rider_trajectory else 'empty'}\n{traceback.format_exc()}",
                    "parameters": {
                        "aperture": aperture,
                        "energy_gev": energy_gev,
                        "start_z": start_z,
                        "transv_offset": transv_offset,
                    },
                }

            # Check smoothness if enabled
            if self.config.smoothness_enabled:
                # Convert trajectory list to dict format for smoothness analysis
                try:
                    trajectory_dict = {
                        "z": np.array([s["z"][0] for s in rider_trajectory]),
                        "gamma": np.array([s["gamma"][0] for s in rider_trajectory]),
                        "t": np.array([s["t"][0] for s in rider_trajectory]),
                    }
                except (KeyError, IndexError) as e:
                    return {
                        "success": False,
                        "error": f"Failed to extract trajectory data for smoothness analysis: {e}. Trajectory keys: {rider_trajectory[0].keys() if rider_trajectory else 'empty'}",
                        "parameters": {
                            "aperture": aperture,
                            "energy_gev": energy_gev,
                            "start_z": start_z,
                            "transv_offset": transv_offset,
                        },
                    }

                smoothness_config = SmoothnessConfig(
                    window_size=self.config.smoothness_window_size,
                    oscillation_threshold=self.config.smoothness_oscillation_threshold,
                    trend_smoothness_threshold=self.config.smoothness_trend_threshold,
                    max_allowed_violations=self.config.smoothness_max_violations,
                )
                smoothness_result = analyze_trajectory_smoothness(
                    trajectory_dict,
                    smoothness_config,
                    particle_mass_amu=self.config.m_particle,
                )
                metrics["smoothness_passed"] = smoothness_result.passed
                metrics["smoothness_violations"] = len(smoothness_result.violations)

                if (
                    self.config.smoothness_reject_on_violation
                    and not smoothness_result.passed
                ):
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

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "parameters": {
                    "aperture": aperture,
                    "energy_gev": energy_gev,
                    "start_z": start_z,
                    "transv_offset": transv_offset,
                },
            }

    def run(self) -> bool:
        """Execute the parameter sweep.

        Returns
        -------
        bool
            True if sweep completed successfully, False otherwise
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Open log file
        log_path = self.output_dir / "sweep.log"
        self.log_file = open(log_path, "w")

        # Initialize debug logging to logcache (like GUI sweeps)
        initialize_debug_logging(context="sweep_cli")
        set_logging_context("sweep_cli")

        # Save original verbosity settings before any overrides
        original_sc_verbosity = self.config.self_consistency_verbosity
        original_adaptive_debug = self.config.adaptive_timestep_debug

        try:
            # Apply log verbosity settings (like GUI does)

            if (
                self.config.log_verbosity == "none"
                or self.config.log_verbosity == "truncated"
            ):
                # Suppress detailed logging for non-full modes
                self.config.self_consistency_verbosity = 0
                self.config.adaptive_timestep_debug = False
            # else: "full" mode inherits settings from config (don't override)

            self._log("")
            self._log(f"Log verbosity: {self.config.log_verbosity}")
            if self.config.log_verbosity == "full":
                self._log("  Full debug logging enabled (inherits config settings)")
                self._log(f"    SC verbosity: {self.config.self_consistency_verbosity}")
                self._log(
                    f"    Adaptive timestep debug: {self.config.adaptive_timestep_debug}"
                )
            elif self.config.log_verbosity == "truncated":
                self._log("  Truncated logging (parameters + metrics + errors only)")
                self._log("    SC verbosity: 0 (overridden)")
                self._log("    Adaptive timestep debug: False (overridden)")
            elif self.config.log_verbosity == "none":
                self._log("  Debug logging disabled")
                self._log("    SC verbosity: 0 (overridden)")
                self._log("    Adaptive timestep debug: False (overridden)")
            self._log(
                f"Trajectory saving: Top N={self.config.save_top_n_trajectories}, All={self.config.save_all_trajectories}, Failed={self.config.save_failed_trajectories}"
            )

            if self.config.mode == "optimization":
                self._log("[ERROR] Optimization mode not yet supported in headless CLI")
                self._log("Please use the GUI for optimization runs")
                return False

            # Generate parameter grids
            param_grids = self._generate_parameter_grids()

            # Calculate total runs
            total_runs = 1
            for key, values in param_grids.items():
                total_runs *= len(values)

            # Log sweep start with total runs
            self._log(f"Starting BLIND SWEEP (Grid Search): {total_runs} total runs")
            self._log(
                f"  aperture: {len(param_grids['aperture'])} points from {min(param_grids['aperture']):.2e} to {max(param_grids['aperture']):.2e}"
            )
            self._log(
                f"  energy: {len(param_grids['energy'])} points from {min(param_grids['energy']):.2e} to {max(param_grids['energy']):.2e}"
            )
            self._log(
                "  transverse_offset_fraction: {:.2e} (fixed)".format(
                    param_grids["transv_offset_frac"][0]
                )
                if len(param_grids["transv_offset_frac"]) == 1
                else f"  transverse_offset_fraction: {len(param_grids['transv_offset_frac'])} values"
            )
            self._log(
                "  start_z: {:.2e} (fixed)".format(param_grids["start_z"][0])
                if len(param_grids["start_z"]) == 1
                else f"  start_z: {len(param_grids['start_z'])} values"
            )
            self._log(f"  Timestep strategy: {self.config.timestep_strategy}")
            if self.config.timestep_strategy == "auto_distance":
                self._log(
                    f"    Target distance: {self.config.target_distance_mm} mm (wall_z + target)"
                )
                self._log(
                    "    All particles will travel to consistent z regardless of energy"
                )
            self._log(f"  z_cutoff_mode: {self.config.z_cutoff_mode}")
            self._log("")
            self._log(f"Output directory: {self.output_dir}")
            self._log("")

            # Run sweep
            start_time = time.time()
            run_num = 0
            failed_count = 0
            result = None  # Initialize result variable

            for aperture in param_grids["aperture"]:
                for energy in param_grids["energy"]:
                    for start_z in param_grids["start_z"]:
                        for transv_offset_frac in param_grids["transv_offset_frac"]:
                            run_num += 1

                            self._log(
                                f"  [PARAMS] Run {run_num}/{total_runs} - All parameters:"
                            )
                            self._log(f"    aperture: {aperture:.4e} mm")
                            self._log(f"    energy: {energy:.4f} GeV")
                            self._log(f"    start_z: {start_z:.4f} mm")
                            self._log(
                                f"    transv_offset_frac: {transv_offset_frac:.4f}"
                            )
                            self._log(
                                f"    rider_m_particle: {self.config.m_particle:.4e} amu"
                            )
                            self._log(
                                f"    rider_charge_sign: {self.config.charge_sign}"
                            )
                            self._log(f"    rider_pcount: {self.config.pcount}")
                            self._log(
                                f"    rider_transv_mom: {self.config.transv_mom:.4e} amu·mm/ns"
                            )
                            self._log(
                                f"    rider_transv_dist: {self.config.transv_dist:.4e} mm"
                            )
                            if self.config.macroparticle_enabled:
                                self._log(
                                    f"    macroparticle_enabled: {self.config.macroparticle_enabled}"
                                )
                                self._log(
                                    f"    macroparticle_charge_multiplier: {self.config.macroparticle_charge_multiplier:.4f}"
                                )
                                self._log(
                                    f"    macroparticle_sigma_multiplier: {self.config.macroparticle_sigma_multiplier:.4f}"
                                )
                                self._log(
                                    f"    macroparticle_use_momentum_errors: {self.config.macroparticle_use_momentum_errors}"
                                )

                            try:
                                result = self._run_single_integration(
                                    aperture=aperture,
                                    energy_gev=energy,
                                    start_z=start_z,
                                    transv_offset_frac=transv_offset_frac,
                                    run_num=run_num,
                                )

                                self.results.append(result)

                                if not result["success"]:
                                    failed_count += 1
                                    error_msg = result.get("error", "Unknown error")
                                    self._log(
                                        f"  [FAILED] Run {run_num}/{total_runs}: {error_msg}"
                                    )
                            except Exception as e:
                                failed_count += 1
                                import traceback

                                error_detail = traceback.format_exc()
                                self._log(
                                    f"  [EXCEPTION] Run {run_num}/{total_runs}: {e}"
                                )
                                self._log("  Traceback:")
                                for line in error_detail.split("\n"):
                                    if line:
                                        self._log(f"    {line}")
                                self.results.append(
                                    {
                                        "success": False,
                                        "error": f"{e}\n{error_detail}",
                                        "parameters": {
                                            "aperture": aperture,
                                            "energy_gev": energy,
                                            "start_z": start_z,
                                            "transv_offset_frac": transv_offset_frac,
                                        },
                                    }
                                )
                                result = self.results[
                                    -1
                                ]  # Set result to the error we just added

                            if result.get("success"):
                                metrics = result.get("metrics", {})
                                self._log(f"  [RESULT] Run {run_num}/{total_runs}:")
                                self._log(
                                    f"    max_energy_gain: {metrics.get('max_energy_gain_gev', 0):.6e} GeV"
                                )
                                self._log(
                                    f"    final_gamma: {metrics.get('final_gamma_mean', 1):.6f}"
                                )
                                self._log(
                                    f"    initial_gamma: {metrics.get('initial_gamma_mean', 1):.6f}"
                                )

            # Save results
            elapsed_time = time.time() - start_time

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
            results_path = self.output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(
                    {
                        "config": {
                            "simulation_type": self.config.simulation_type.name
                            if hasattr(self.config.simulation_type, "name")
                            else str(self.config.simulation_type),
                            "aperture_range": self.config.aperture_range,
                            "aperture_points": self.config.aperture_points,
                            "energy_range": self.config.energy_range,
                            "energy_points": self.config.energy_points,
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

            return True

        except KeyboardInterrupt:
            self._log("")
            self._log("")
            self._log("[INFO] Sweep interrupted by user")
            return False
        except Exception as e:
            self._log("")
            self._log("")
            self._log(f"[ERROR] {e}")
            import traceback

            for line in traceback.format_exc().split("\n"):
                if line:
                    self._log(f"  {line}")
            return False
        finally:
            # Restore original verbosity settings
            self.config.self_consistency_verbosity = original_sc_verbosity
            self.config.adaptive_timestep_debug = original_adaptive_debug

            if self.log_file is not None:
                self.log_file.close()


def _convert_json_config_to_dataclass(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON config format to OptimizationConfig dataclass format.

    JSON uses separate min/max/points fields (e.g., aperture_min, aperture_max),
    while OptimizationConfig uses tuple ranges (e.g., aperture_range).
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

    # Convert sweep_parameters to appropriate ranges
    sweep_params = converted.get("sweep_parameters", {})

    # Handle rider parameters
    for param_name in [
        "rider_m_particle",
        "rider_charge_sign",
        "rider_pcount",
        "rider_transv_mom",
        "rider_transv_dist",
        "rider_stripped_ions",
        "macroparticle_charge_multiplier",
        "macroparticle_sigma_multiplier",
    ]:
        if param_name in sweep_params:
            param_config = sweep_params[param_name]
            if (
                param_config.get("enabled")
                and "min" in param_config
                and "max" in param_config
            ):
                # Map to OptimizationConfig field names
                field_map = {
                    "rider_m_particle": "particle_mass_range",
                    "rider_charge_sign": "particle_charge_range",
                    "rider_pcount": "particle_count_range",
                    "rider_transv_mom": "transverse_momentum_range",
                    "rider_transv_dist": "transverse_spread_range",
                    "rider_stripped_ions": "rider_stripped_ions_range",
                    "macroparticle_charge_multiplier": "macroparticle_charge_range",
                    "macroparticle_sigma_multiplier": "macroparticle_sigma_range",
                }
                if param_name in field_map:
                    field_name = field_map[param_name]
                    converted[field_name] = (param_config["min"], param_config["max"])
                    points_field = field_name.replace("_range", "_points")
                    if "points" in param_config:
                        converted[points_field] = param_config["points"]

    # Handle driver parameters (for BUNCH_TO_BUNCH)
    for param_name in [
        "driver_m_particle",
        "driver_charge_sign",
        "driver_pcount",
        "driver_transv_mom",
        "driver_transv_dist",
        "driver_starting_distance",
        "driver_energy_gev",
        "driver_stripped_ions",
    ]:
        if param_name in sweep_params:
            param_config = sweep_params[param_name]
            if (
                param_config.get("enabled")
                and "min" in param_config
                and "max" in param_config
            ):
                field_map = {
                    "driver_m_particle": "driver_mass_range",
                    "driver_charge_sign": "driver_charge_sign_range",
                    "driver_pcount": "driver_pcount_range",
                    "driver_transv_mom": "driver_transv_mom_range",
                    "driver_transv_dist": "driver_transv_dist_range",
                    "driver_starting_distance": "driver_starting_distance_range",
                    "driver_energy_gev": "driver_energy_range",
                    "driver_stripped_ions": "driver_stripped_ions_range",
                }
                if param_name in field_map:
                    field_name = field_map[param_name]
                    converted[field_name] = (param_config["min"], param_config["max"])
                    points_field = field_name.replace("_range", "_points")
                    if "points" in param_config:
                        converted[points_field] = param_config["points"]

    # Remove sweep_parameters from converted dict as it's been processed
    converted.pop("sweep_parameters", None)

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
        Whether to print progress messages, by default True
    verbosity_overrides : Dict[str, Any], optional
        Dictionary of verbosity settings to override config values.
        Supported keys: 'log_verbosity', 'self_consistency_verbosity', 'adaptive_timestep_debug'

    Returns
    -------
    bool
        True if sweep completed successfully, False otherwise
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
                print(
                    f"[INFO] Overriding {key} from CLI: {value}",
                    flush=True,
                )

    # Determine output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = config_path.stem
        output_dir = Path(config.output_dir) / f"{timestamp}_{config_name}"

    # Create and run sweep
    runner = SweepRunner(config, output_dir, verbose=verbose)
    return runner.run()
