"""Backend run logic mixin for OptimizationPlugin."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.constants import C_MMNS  # type: ignore[import]
from core.debug_logger import set_logging_context  # type: ignore[import]
from core.smoothness_analyzer import (  # type: ignore[import]
    SmoothnessConfig,
    analyze_trajectory_smoothness,
)
from core.types import SimulationType  # type: ignore[import]
from lw_integrator.testbed_runner import (  # type: ignore[import]
    SimulationOptions,
    run_testbed,
)
from optimization.config import (  # type: ignore[import]
    calculate_auto_steps,
    calculate_auto_timestep,
)
from optimization.sweep_helpers import (
    build_parameter_grids,
    calculate_starting_pz_from_energy,
    generate_parameter_range,
)


class OptimizationRunMixin:
    """Encapsulates run queue, threading, and integration helpers."""

    def _run_optimization_background(self):
        """Run optimization in background using selected algorithm."""
        # Set logging context for this optimization run
        method = self.config.optimization_method
        set_logging_context(f"optimization_{method}")

        # Open log file in temporary location (will be moved when results are saved)
        import tempfile
        import time

        temp_dir = tempfile.mkdtemp(prefix="opt_log_")
        self._open_log_file(temp_dir)

        start_time = time.time()

        try:
            from optimization.optimizer import (
                adaptive_grid_search,
                genetic_algorithm,
                multi_start_optimize,
                optimize_parameters,
            )

            self._log_result("=" * 80)
            self._log_result(f"OPTIMIZATION MODE: {self.config.optimization_method}")
            self._log_result("=" * 80)
            self._log_result("")

            # Apply log verbosity settings (same as sweep mode)
            # Save original values to restore later
            original_sc_verbosity = self.config.self_consistency_verbosity
            original_adaptive_debug = self.config.adaptive_timestep_debug

            use_no_logging = self.config.log_verbosity == "none"
            use_truncated_logging = self.config.log_verbosity == "truncated"
            use_full_logging = self.config.log_verbosity == "full"

            # Apply log verbosity settings - control what gets logged
            # "full" mode INHERITS stability settings from config/GUI
            # Other modes override to reduce output
            if use_full_logging:
                # INHERIT stability verbosity settings from config (don't override)
                # Use whatever was set in Stability tab or loaded from config
                self._log_result(f"Log verbosity: {self.config.log_verbosity}")
                self._log_result(
                    "  Full debug logging enabled (inherits Stability tab settings)"
                )
                self._log_result(
                    f"    SC verbosity: {self.config.self_consistency_verbosity}"
                )
                self._log_result(
                    f"    Adaptive timestep debug: {self.config.adaptive_timestep_debug}"
                )
            elif use_truncated_logging:
                # Disable verbose logging for optimizations with many evaluations
                self.config.self_consistency_verbosity = 0
                self.config.adaptive_timestep_debug = False
                self._log_result(f"Log verbosity: {self.config.log_verbosity}")
                self._log_result(
                    "  Truncated logging (parameters + metrics + errors only)"
                )
            elif use_no_logging:
                # Completely disable all debug logging
                self.config.self_consistency_verbosity = 0
                self.config.adaptive_timestep_debug = False
                self._log_result(f"Log verbosity: {self.config.log_verbosity}")
                self._log_result("  Debug logging disabled")
            else:
                # Unknown log verbosity - use config file values
                self._log_result(
                    f"Log verbosity: {self.config.log_verbosity} (unknown, using config values)"
                )
                self._log_result(
                    f"  adaptive_timestep_debug: {self.config.adaptive_timestep_debug}"
                )
                self._log_result(
                    f"  self_consistency_verbosity: {self.config.self_consistency_verbosity}"
                )
            self._log_result("")

            # Build parameter names and bounds from config
            param_names = []
            param_bounds = []

            # Aperture
            if self.config.aperture_points > 1:
                param_names.append("aperture_radius")
                param_bounds.append(self.config.aperture_range)

            # Energy
            if self.config.energy_points > 1:
                param_names.append("initial_energy_gev")
                param_bounds.append(self.config.energy_range)
                self._log_result(
                    f"    Added: initial_energy_gev, range={self.config.energy_range}, points={self.config.energy_points}"
                )

            # Transverse momentum (if enabled as sweep parameter)
            if (
                self.config.transverse_momentum_range is not None
                and self.config.transverse_momentum_points > 1
            ):
                param_names.append("transverse_momentum")
                param_bounds.append(self.config.transverse_momentum_range)
                self._log_result(
                    f"    Added: transverse_momentum, range={self.config.transverse_momentum_range}, points={self.config.transverse_momentum_points}"
                )

            # Timestep (if enabled as sweep parameter)
            if (
                self.config.timestep_range is not None
                and self.config.timestep_points > 1
            ):
                param_names.append("timestep")
                param_bounds.append(self.config.timestep_range)

            # Rider transverse distance (spread) - if enabled as sweep parameter
            if (
                self.config.transverse_spread_range is not None
                and self.config.transverse_spread_points > 1
            ):
                param_names.append("rider_transv_dist")
                param_bounds.append(self.config.transverse_spread_range)
                self._log_result(
                    f"    Added: rider_transv_dist, range={self.config.transverse_spread_range}, points={self.config.transverse_spread_points}"
                )

            # Macroparticle charge multiplier - if enabled as sweep parameter
            if (
                self.config.macroparticle_charge_range is not None
                and self.config.macroparticle_charge_points > 1
            ):
                param_names.append("macroparticle_charge_multiplier")
                param_bounds.append(self.config.macroparticle_charge_range)

            # Macroparticle sigma multiplier - if enabled as sweep parameter
            if (
                self.config.macroparticle_sigma_range is not None
                and self.config.macroparticle_sigma_points > 1
            ):
                param_names.append("macroparticle_sigma_multiplier")
                param_bounds.append(self.config.macroparticle_sigma_range)

            # Wall z position - if enabled as sweep parameter
            if self.config.wall_z_range is not None and self.config.wall_z_points > 1:
                param_names.append("wall_z")
                param_bounds.append(self.config.wall_z_range)

            # Rider stripped ions - if enabled as sweep parameter
            if (
                self.config.rider_stripped_ions_range is not None
                and self.config.rider_stripped_ions_points > 1
            ):
                param_names.append("rider_stripped_ions")
                param_bounds.append(self.config.rider_stripped_ions_range)

            # Driver stripped ions - if enabled as sweep parameter (BUNCH_TO_BUNCH only)
            if (
                self.config.driver_stripped_ions_range is not None
                and self.config.driver_stripped_ions_points > 1
            ):
                param_names.append("driver_stripped_ions")
                param_bounds.append(self.config.driver_stripped_ions_range)

            # Rider particle mass - if enabled as sweep parameter
            if (
                self.config.particle_mass_range is not None
                and self.config.particle_mass_points > 1
            ):
                param_names.append("rider_m_particle")
                param_bounds.append(self.config.particle_mass_range)
                self._log_result(
                    f"    Added: rider_m_particle, range={self.config.particle_mass_range}, points={self.config.particle_mass_points}"
                )

            # Rider charge sign - if enabled as sweep parameter
            if (
                self.config.particle_charge_range is not None
                and self.config.particle_charge_points > 1
            ):
                param_names.append("rider_charge_sign")
                param_bounds.append(self.config.particle_charge_range)

            # Rider particle count - if enabled as sweep parameter
            if (
                self.config.particle_count_range is not None
                and self.config.particle_count_points > 1
            ):
                param_names.append("rider_pcount")
                param_bounds.append(self.config.particle_count_range)
                self._log_result(
                    f"    Added: rider_pcount, range={self.config.particle_count_range}, points={self.config.particle_count_points}"
                )

            # Driver particle mass - if enabled as sweep parameter (BUNCH_TO_BUNCH only)
            if (
                self.config.driver_mass_range is not None
                and self.config.driver_mass_points > 1
            ):
                param_names.append("driver_m_particle")
                param_bounds.append(self.config.driver_mass_range)
                self._log_result(
                    f"    Added: driver_m_particle, range={self.config.driver_mass_range}, points={self.config.driver_mass_points}"
                )

            # Driver charge sign - if enabled as sweep parameter (BUNCH_TO_BUNCH only)
            if (
                self.config.driver_charge_sign_range is not None
                and self.config.driver_charge_sign_points > 1
            ):
                param_names.append("driver_charge_sign")
                param_bounds.append(self.config.driver_charge_sign_range)

            # Driver particle count - if enabled as sweep parameter (BUNCH_TO_BUNCH only)
            if (
                self.config.driver_pcount_range is not None
                and self.config.driver_pcount_points > 1
            ):
                param_names.append("driver_pcount")
                param_bounds.append(self.config.driver_pcount_range)
                self._log_result(
                    f"    Added: driver_pcount, range={self.config.driver_pcount_range}, points={self.config.driver_pcount_points}"
                )

            # Driver transverse momentum - if enabled as sweep parameter (BUNCH_TO_BUNCH only)
            if (
                self.config.driver_transv_mom_range is not None
                and self.config.driver_transv_mom_points > 1
            ):
                param_names.append("driver_transv_mom")
                param_bounds.append(self.config.driver_transv_mom_range)
                self._log_result(
                    f"    Added: driver_transv_mom, range={self.config.driver_transv_mom_range}, points={self.config.driver_transv_mom_points}"
                )

            # Driver transverse distance - if enabled as sweep parameter (BUNCH_TO_BUNCH only)
            if (
                self.config.driver_transv_dist_range is not None
                and self.config.driver_transv_dist_points > 1
            ):
                param_names.append("driver_transv_dist")
                param_bounds.append(self.config.driver_transv_dist_range)
                self._log_result(
                    f"    Added: driver_transv_dist, range={self.config.driver_transv_dist_range}, points={self.config.driver_transv_dist_points}"
                )

            # Driver starting distance - if enabled as sweep parameter (BUNCH_TO_BUNCH only)
            if (
                self.config.driver_starting_distance_range is not None
                and self.config.driver_starting_distance_points > 1
            ):
                param_names.append("driver_starting_distance")
                param_bounds.append(self.config.driver_starting_distance_range)

            # Driver energy - if enabled as sweep parameter (BUNCH_TO_BUNCH only)
            # Note: Internally uses Pz, but optimizer varies energy for user convenience
            if (
                self.config.driver_energy_range is not None
                and self.config.driver_energy_points > 1
            ):
                param_names.append("driver_energy_gev")
                param_bounds.append(self.config.driver_energy_range)

            if len(param_names) == 0:
                self._log_result(
                    "[ERROR] No parameters to optimize! Enable at least 2 points for aperture or energy."
                )
                self.running = False
                return

            self._log_result(
                f"[DEBUG] Total parameters to optimize: {len(param_names)}"
            )
            self._log_result(f"Optimizing parameters: {param_names}")
            self._log_result(f"Parameter bounds: {param_bounds}")
            self._log_result(f"Objective: {self.config.objective}")
            self._log_result("")

            # Create base config template (this would need proper implementation)
            # For now, create a minimal dict representation
            config_template = {
                "simulation_type": self.config.simulation_type,
                "wall_z": self.config.wall_z,
                "steps": self.config.steps,
                "timestep": self.config.timestep,
                "m_particle": self.config.m_particle,
                "charge_sign": self.config.charge_sign,
                "stripped_ions": self.config.stripped_ions,
                "transv_mom": self.config.transv_mom,
                # Add other fixed parameters
            }

            # Determine metric name from objective
            metric_name = "max_energy_gain_gev"
            maximize = True

            if self.config.objective == "max_percent_energy_gain":
                metric_name = "max_percent_energy_gain"
                maximize = True
            elif "min" in self.config.objective.lower():
                maximize = False

            # Run optimization based on selected method
            method = self.config.optimization_method
            self._log_result(f"Starting {method} optimization...")
            self._log_result("")

            result = None

            # Track evaluation count and all evaluations for heatmap
            eval_counter = [0]  # Use list for mutable closure
            all_evaluations = []  # Store all parameter sets and their results

            # Create custom objective function that uses our integration runner
            def evaluate_params(x):
                """Evaluate parameter vector and return value to minimize."""
                eval_num = eval_counter[0]
                eval_counter[0] += 1

                # Log evaluation start
                param_str = ", ".join(
                    [f"{name}={val:.4g}" for name, val in zip(param_names, x)]
                )
                self._log_result(f"[OPTIMIZATION] Evaluation {eval_num}: {param_str}")

                # Check for cancellation
                if not self.running:
                    self._log_result("[CANCELLED] Optimization cancelled by user")
                    return np.inf

                if self.gui_controller and hasattr(
                    self.gui_controller, "_cancel_requested"
                ):
                    if self.gui_controller._cancel_requested:
                        self._log_result("[CANCELLED] Optimization cancelled by user")
                        return np.inf

                try:
                    # Map parameters
                    aperture = self.config.aperture_range[0]  # default
                    energy = self.config.energy_range[0]  # default
                    start_z = (
                        self.config.starting_z_positions[0]
                        if self.config.starting_z_positions
                        else 0.0
                    )
                    offset_frac = (
                        self.config.transverse_offset_fractions[0]
                        if self.config.transverse_offset_fractions
                        else 0.0
                    )
                    timestep = self.config.timestep
                    steps = self.config.steps
                    rider_transv_dist = self.config.transv_dist  # default
                    macroparticle_charge_mult = (
                        self.config.macroparticle_charge_multiplier
                    )  # default
                    macroparticle_sigma_mult = (
                        self.config.macroparticle_sigma_multiplier
                    )  # default
                    wall_z = self.config.wall_z  # default
                    rider_stripped_ions = self.config.stripped_ions  # default
                    driver_stripped_ions = self.config.driver_stripped_ions  # default
                    rider_m_particle = self.config.m_particle  # default
                    rider_charge_sign = self.config.charge_sign  # default
                    rider_pcount = self.config.pcount  # default
                    rider_transv_mom = self.config.transv_mom  # default
                    driver_m_particle = self.config.driver_m_particle  # default
                    driver_charge_sign = self.config.driver_charge_sign  # default
                    driver_pcount = self.config.driver_pcount  # default
                    driver_transv_mom = self.config.driver_transv_mom  # default
                    driver_transv_dist = self.config.driver_transv_dist  # default
                    driver_starting_distance = (
                        self.config.driver_starting_distance
                    )  # default
                    driver_starting_Pz = self.config.driver_starting_Pz  # default
                    driver_energy_gev = self.config.driver_energy_gev  # default

                    for i, param_name in enumerate(param_names):
                        if param_name == "aperture_radius":
                            aperture = x[i]
                        elif param_name == "initial_energy_gev":
                            energy = x[i]
                        elif param_name == "start_z":
                            start_z = x[i]
                        elif param_name == "transverse_offset":
                            offset_frac = x[i]
                        elif param_name == "timestep":
                            timestep = x[i]
                        elif param_name == "transverse_momentum":
                            rider_transv_mom = x[i]
                        elif param_name == "rider_transv_dist":
                            rider_transv_dist = x[i]
                        elif param_name == "macroparticle_charge_multiplier":
                            macroparticle_charge_mult = x[i]
                        elif param_name == "macroparticle_sigma_multiplier":
                            macroparticle_sigma_mult = x[i]
                        elif param_name == "wall_z":
                            wall_z = x[i]
                        elif param_name == "rider_stripped_ions":
                            rider_stripped_ions = x[i]
                        elif param_name == "driver_stripped_ions":
                            driver_stripped_ions = x[i]
                        elif param_name == "rider_m_particle":
                            rider_m_particle = x[i]
                        elif param_name == "rider_charge_sign":
                            rider_charge_sign = x[i]
                        elif param_name == "rider_pcount":
                            rider_pcount = int(x[i])
                        elif param_name == "driver_m_particle":
                            driver_m_particle = x[i]
                        elif param_name == "driver_charge_sign":
                            driver_charge_sign = x[i]
                        elif param_name == "driver_pcount":
                            driver_pcount = int(x[i])
                        elif param_name == "driver_transv_mom":
                            driver_transv_mom = x[i]
                        elif param_name == "driver_transv_dist":
                            driver_transv_dist = x[i]
                        elif param_name == "driver_starting_distance":
                            driver_starting_distance = x[i]
                        elif param_name == "driver_energy_gev":
                            driver_energy_gev = x[i]
                            # Convert energy to Pz using configured direction
                            _drv_neg = (
                                getattr(self.config, "driver_direction", "-z") == "-z"
                            )
                            driver_starting_Pz = calculate_starting_pz_from_energy(
                                driver_energy_gev, driver_m_particle, negative=_drv_neg
                            )
                        elif param_name == "driver_starting_Pz":
                            # Legacy support if old configs still use Pz
                            driver_starting_Pz = x[i]

                    # Calculate transverse offset in mm
                    # For CONDUCTING_WALL/SWITCHING_WALL: fraction of aperture
                    # For BUNCH_TO_BUNCH: absolute distance in mm
                    sim_type_str = self.config.simulation_type
                    if sim_type_str == "BUNCH_TO_BUNCH":
                        transv_offset = (
                            offset_frac  # Direct mm value for bunch-to-bunch
                        )
                    else:
                        transv_offset = (
                            offset_frac * aperture
                        )  # Fraction for conducting wall

                    # Get driver particle parameters if BUNCH_TO_BUNCH (needed before timestep calc)
                    driver_params_dict = None
                    if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
                        driver_params_dict = {
                            "m_particle": driver_m_particle,
                            "charge_sign": driver_charge_sign,
                            "pcount": int(driver_pcount),
                            "transv_mom": driver_transv_mom,
                            "transv_dist": driver_transv_dist,
                            "starting_distance": driver_starting_distance,
                            "starting_Pz": driver_starting_Pz,
                            "stripped_ions": driver_stripped_ions,
                            "transv_offset_x": self.config.driver_transv_offset_x,
                            "transv_offset_y": self.config.driver_transv_offset_y,
                        }

                    # Calculate timestep if using auto_distance strategy
                    if self.config.timestep_strategy == "auto_distance":
                        # Get driver starting position for BUNCH_TO_BUNCH mode
                        driver_start_z = 1000.0  # Default driver starting position
                        if driver_params_dict is not None:
                            driver_start_z = driver_params_dict.get(
                                "starting_distance", 1000.0
                            )

                        timestep = self.config.calculate_timestep_for_energy(
                            energy,
                            self.config.m_particle,
                            wall_z=wall_z,
                            start_z=start_z,
                            driver_start_z=driver_start_z,
                        )
                        steps = self.config.steps

                    # Run integration with timeout if enabled
                    result = None
                    timed_out = False

                    if self.config.per_run_timeout > 0:
                        result_container = [None]
                        error_container = [None]
                        cancel_flag = [False]

                        def run_integration():
                            try:
                                result_container[0] = self._run_single_integration(
                                    aperture=aperture,
                                    energy_gev=energy,
                                    start_z=start_z,
                                    transv_offset=transv_offset,
                                    timestep=timestep,
                                    steps=steps,
                                    rider_m_particle=rider_m_particle,
                                    rider_charge_sign=rider_charge_sign,
                                    rider_pcount=int(rider_pcount),
                                    rider_transv_mom=rider_transv_mom,
                                    rider_transv_dist=rider_transv_dist,
                                    rider_stripped_ions=rider_stripped_ions,
                                    macroparticle_charge_multiplier=macroparticle_charge_mult,
                                    macroparticle_sigma_multiplier=macroparticle_sigma_mult,
                                    driver_params=driver_params_dict,
                                    wall_z=wall_z,
                                    run_num=eval_num,
                                    cancel_flag=cancel_flag,
                                )
                            except Exception as e:
                                error_container[0] = e

                        thread = threading.Thread(target=run_integration)
                        thread.daemon = True
                        thread.start()
                        thread.join(timeout=self.config.per_run_timeout)

                        if thread.is_alive():
                            timed_out = True
                            cancel_flag[0] = True
                            self._log_result(
                                f"[WARNING] Evaluation timed out for params {x} after {self.config.per_run_timeout}s"
                            )
                            self._log_result(
                                f"[WARNING] Signaling integration to cancel..."
                            )
                            # Give it a brief moment to respond
                            thread.join(timeout=2.0)
                            return np.inf
                        elif error_container[0] is not None:
                            raise error_container[0]
                        else:
                            result = result_container[0]
                    else:
                        # No timeout - run directly
                        result = self._run_single_integration(
                            aperture=aperture,
                            energy_gev=energy,
                            start_z=start_z,
                            transv_offset=transv_offset,
                            timestep=timestep,
                            steps=steps,
                            rider_m_particle=rider_m_particle,
                            rider_charge_sign=rider_charge_sign,
                            rider_pcount=int(rider_pcount),
                            rider_transv_mom=rider_transv_mom,
                            rider_transv_dist=rider_transv_dist,
                            rider_stripped_ions=rider_stripped_ions,
                            macroparticle_charge_multiplier=macroparticle_charge_mult,
                            macroparticle_sigma_multiplier=macroparticle_sigma_mult,
                            driver_params=driver_params_dict,
                            wall_z=wall_z,
                            run_num=eval_num,
                            cancel_flag=None,
                        )

                    if result is None or "metrics" not in result:
                        # Store failed evaluation
                        eval_record = {
                            "evaluation": eval_num,
                            "parameters": dict(zip(param_names, x)),
                            "failed": True,
                            "halted_early": (
                                result.get("halted_early", False) if result else False
                            ),
                            "halt_reason": (
                                result.get("halt_reason", None) if result else None
                            ),
                            "objective_value": float("inf"),
                        }
                        all_evaluations.append(eval_record)
                        return np.inf

                    # Check if run was halted early
                    if result.get("halted_early", False):
                        self._log_result(
                            f"[INFO] Evaluation {eval_num} halted early: {result.get('halt_reason', 'unknown')}"
                        )
                        self._log_result(
                            f"[INFO] Returning inf (rejecting halted evaluation)"
                        )
                        # Store halted evaluation
                        eval_record = {
                            "evaluation": eval_num,
                            "parameters": dict(zip(param_names, x)),
                            "failed": False,
                            "halted_early": True,
                            "halt_reason": result.get("halt_reason"),
                            "objective_value": float("inf"),
                        }
                        all_evaluations.append(eval_record)
                        return np.inf

                    # Extract metric value
                    metrics = result["metrics"]
                    value = metrics.get(metric_name, np.nan)

                    if np.isnan(value) or np.isinf(value):
                        self._log_result(
                            f"[WARNING] Evaluation {eval_num} returned {'NaN' if np.isnan(value) else 'inf'} for metric '{metric_name}'"
                        )
                        self._log_result(
                            f"[WARNING] Available metrics: {list(metrics.keys())}"
                        )
                        if len(metrics) > 0:
                            self._log_result(f"[WARNING] Metric values:")
                            for k, v in metrics.items():
                                self._log_result(f"  {k}: {v}")
                        self._log_result(
                            f"[WARNING] Returning inf (rejecting this evaluation)"
                        )
                        # Store failed evaluation
                        eval_record = {
                            "evaluation": eval_num,
                            "parameters": dict(zip(param_names, x)),
                            "failed": True,
                            "objective_value": float("inf"),
                            "metrics": result.get("metrics", {}),
                        }
                        all_evaluations.append(eval_record)
                        return np.inf

                    penalty = self._compute_soft_penalty(
                        aperture_radius=aperture,
                        macroparticle_charge_multiplier=macroparticle_charge_mult,
                        initial_energy_gev=energy,
                    )

                    adjusted_value = value
                    if penalty > 0:
                        if maximize:
                            adjusted_value = value - penalty
                        else:
                            adjusted_value = value + penalty
                        self._log_result(
                            "[INFO] Applied soft penalty of "
                            f"{penalty:.3e} to {self.config.objective} (risk-prone parameters)"
                        )

                    # Return value to minimize (negate if maximizing)
                    result_value = -adjusted_value if maximize else adjusted_value

                    # Store successful evaluation
                    eval_record = {
                        "evaluation": eval_num,
                        "parameters": dict(zip(param_names, x)),
                        "objective_value": adjusted_value,
                        "raw_objective_value": value,
                        "soft_penalty": penalty,
                        "fitness": result_value,  # Store fitness (for minimization)
                        "failed": False,
                        "halted_early": False,
                        "metrics": result.get("metrics", {}),
                    }

                    # Save trajectory if requested and available
                    if self.config.save_all_trajectories and "trajectory" in result:
                        # We'll save these after optimization dir is created
                        eval_record["trajectory"] = result["trajectory"]

                    all_evaluations.append(eval_record)

                    return result_value

                except Exception as e:
                    import traceback

                    self._log_result(
                        f"[ERROR] Evaluation {eval_num} failed for params {x}"
                    )
                    self._log_result(f"[ERROR] Exception: {type(e).__name__}: {e}")
                    self._log_result(f"[ERROR] Traceback:")
                    for line in traceback.format_exc().splitlines():
                        self._log_result(f"  {line}")

                    # Store failed evaluation
                    eval_record = {
                        "evaluation": eval_num,
                        "parameters": dict(zip(param_names, x)),
                        "failed": True,
                        "error": str(e),
                        "objective_value": float("inf"),
                    }
                    all_evaluations.append(eval_record)

                    return np.inf

            # Define progress callback for convergence monitoring (used by all methods)
            def log_convergence_progress(
                generation,
                best_value,
                improvement,
                tolerance,
                patience_remaining,
                converged,
            ):
                """Log convergence progress after each generation."""
                # Filter out inf values in logging
                if np.isfinite(best_value):
                    self._log_result(
                        f"[OPTIMIZATION] Generation {generation}: best={best_value:.6e}, "
                        f"improvement={improvement:.6e}, tolerance={tolerance:.6e}"
                    )
                else:
                    self._log_result(
                        f"[OPTIMIZATION] Generation {generation}: best=inf (no valid solutions yet), "
                        f"improvement={improvement:.6e}, tolerance={tolerance:.6e}"
                    )
                if generation >= self.config.optimization_convergence_patience:
                    if converged:
                        self._log_result(
                            f"[CONVERGENCE] Converged! Improvement ({improvement:.6e}) "
                            f"< tolerance ({tolerance:.6e})"
                        )
                    else:
                        self._log_result(
                            f"[CONVERGENCE] Progress: {patience_remaining} generations "
                            f"remaining before early stop check"
                        )

            if method == "genetic_algorithm":

                result = genetic_algorithm(
                    config_template=config_template,
                    parameter_names=param_names,
                    parameter_bounds=param_bounds,
                    metric_name=metric_name,
                    maximize=maximize,
                    population_size=self.config.optimization_population_size,
                    n_generations=self.config.optimization_maxiter,
                    mutation_rate=self.config.optimization_mutation_rate,
                    crossover_rate=self.config.optimization_crossover_rate,
                    seed=self.config.seed,
                    objective_function=evaluate_params,
                    convergence_tol=self.config.optimization_convergence_tol,
                    convergence_patience=self.config.optimization_convergence_patience,
                    progress_callback=log_convergence_progress,
                )

            elif method == "differential_evolution":
                result = optimize_parameters(
                    config_template=config_template,
                    parameter_names=param_names,
                    parameter_bounds=param_bounds,
                    metric_name=metric_name,
                    method="differential_evolution",
                    maximize=maximize,
                    maxiter=self.config.optimization_maxiter,
                    popsize=self.config.optimization_population_size,
                    objective_function=evaluate_params,
                    progress_callback=log_convergence_progress,
                )

            elif method == "multi_start":
                result = multi_start_optimize(
                    config_template=config_template,
                    parameter_names=param_names,
                    parameter_bounds=param_bounds,
                    metric_name=metric_name,
                    n_starts=self.config.optimization_n_starts,
                    maximize=maximize,
                    maxiter=self.config.optimization_maxiter,
                    objective_function=evaluate_params,
                    progress_callback=log_convergence_progress,
                )

            elif method == "adaptive_grid":
                best_params, best_value, history = adaptive_grid_search(
                    config_template=config_template,
                    parameter_names=param_names,
                    parameter_bounds=param_bounds,
                    metric_name=metric_name,
                    maximize=maximize,
                    initial_points_per_dim=5,
                    refinement_levels=2,
                    objective_function=evaluate_params,
                    progress_callback=log_convergence_progress,
                )
                # Convert to OptimizeResult format
                from scipy.optimize import OptimizeResult

                result = OptimizeResult()
                result.x = best_params
                result.fun = -best_value if maximize else best_value
                result.best_params_dict = dict(zip(param_names, best_params))
                result.success = True

            if result is None:
                self._log_result(f"[ERROR] Unknown optimization method: {method}")
                self.running = False
                return

            # Cache all evaluations for saving with results
            self._all_evaluations_cache = all_evaluations

            # Log results
            self._log_result("")
            self._log_result("=" * 80)
            self._log_result("OPTIMIZATION COMPLETE")
            self._log_result("=" * 80)
            # Un-negate the result if we were maximizing (optimizer minimizes by negating)
            best_metric_value = -result.fun if maximize else result.fun
            self._log_result(f"Best {metric_name}: {best_metric_value:.12e}")
            self._log_result("Best parameters:")
            for param_name, value in result.best_params_dict.items():
                self._log_result(f"  {param_name}: {value:.12e}")
            self._log_result("")
            self._log_result(
                f"Function evaluations: {result.nfev if hasattr(result, 'nfev') else 'N/A'}"
            )
            self._log_result("")

            # Save results (this sets self._last_optimization_dir)
            self._save_optimization_results(result, param_names)

            # Re-run top N parameters to generate and save trajectories (only if enabled)
            if self.config.save_top_n_trajectories:
                self._save_top_n_optimization_trajectories(result, param_names)
            else:
                self._log_result("")
                self._log_result(
                    "[INFO] Top N trajectory saving disabled (save_top_n_trajectories=False)"
                )

            # Cache all evaluations for saving and generate heatmap
            if len(all_evaluations) > 0:
                self._all_evaluations_cache = all_evaluations
                self._generate_optimization_heatmap(
                    all_evaluations, param_names, self._last_optimization_dir
                )

            elapsed_time = time.time() - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = elapsed_time % 60

            self._log_result("[OK] Optimization complete!")
            if hours > 0:
                self._log_result(
                    f"  Total time: {hours}h {minutes}m {seconds:.1f}s ({elapsed_time:.1f}s)"
                )
            elif minutes > 0:
                self._log_result(
                    f"  Total time: {minutes}m {seconds:.1f}s ({elapsed_time:.1f}s)"
                )
            else:
                self._log_result(f"  Total time: {elapsed_time:.1f}s")

        except KeyboardInterrupt:
            self._log_result("")
            self._log_result("[CANCELLED] Optimization cancelled by user")
            self._log_result("")
            # Try to save partial results if we have any evaluations
            if "all_evaluations" in locals() and len(all_evaluations) > 0:
                self._log_result(
                    f"[INFO] Saving partial results ({len(all_evaluations)} evaluations completed)..."
                )
                try:
                    self._save_partial_optimization_results(
                        all_evaluations, param_names, "CANCELLED"
                    )
                except Exception as save_err:
                    self._log_result(
                        f"[WARNING] Failed to save partial results: {save_err}"
                    )
        except Exception as e:  # pragma: no cover - integration path
            import traceback

            error_msg = f"Optimization failed: {e}\n{traceback.format_exc()}"
            self._log_result(f"[ERROR] {error_msg}")
            # Try to save partial results even on error
            if "all_evaluations" in locals() and len(all_evaluations) > 0:
                self._log_result(
                    f"[INFO] Saving partial results ({len(all_evaluations)} evaluations completed)..."
                )
                try:
                    self._save_partial_optimization_results(
                        all_evaluations, param_names, "FAILED"
                    )
                except Exception as save_err:
                    self._log_result(
                        f"[WARNING] Failed to save partial results: {save_err}"
                    )
        finally:
            # Restore original verbosity settings
            if "original_sc_verbosity" in locals():
                self.config.self_consistency_verbosity = original_sc_verbosity
            if "original_adaptive_debug" in locals():
                self.config.adaptive_timestep_debug = original_adaptive_debug

            self.running = False
            self._update_progress(100, "Done")
            # Ensure log file is closed
            if self._log_file is not None:
                self._close_log_file()

    def _run_sweep_background(self, is_finetune: bool = False, finetune_regions=None):
        """Run parameter sweep in background with real integration.

        Args:
            is_finetune: If True, this is a fine-tuning sweep
            finetune_regions: List of parameter regions for fine-tuning
        """
        # Set logging context for this sweep run
        context = "sweep_finetune" if is_finetune else "sweep"
        set_logging_context(context)

        # Open log file in temporary location (will be moved when results are saved)
        import tempfile
        import time

        temp_dir = tempfile.mkdtemp(prefix="sweep_log_")
        self._open_log_file(temp_dir)

        start_time = time.time()

        try:
            # Check mode and route accordingly
            if self.config.mode == "optimization":
                self._run_optimization_background()
                return

            # Generate parameter grid including sweepable parameters
            param_grids = self._generate_parameter_grids()

            # Calculate total runs
            total_runs = 1
            for values in param_grids.values():
                total_runs *= len(values)

            # Determine verbosity level from config
            use_no_logging = self.config.log_verbosity == "none"
            use_truncated_logging = self.config.log_verbosity == "truncated"
            use_full_debug = self.config.log_verbosity == "full"

            # Override config verbosity settings based on log mode
            # Save original values to restore later
            original_sc_verbosity = self.config.self_consistency_verbosity
            original_adaptive_debug = self.config.adaptive_timestep_debug

            if use_no_logging or use_truncated_logging:
                # Suppress SC iteration output and adaptive timestep refinement output
                self.config.self_consistency_verbosity = 0
                self.config.adaptive_timestep_debug = False
            # else: full debug mode - INHERIT stability settings from config/GUI (don't override)

            self._log_result(
                f"Starting BLIND SWEEP (Grid Search): {total_runs} total runs"
            )
            self._log_result(f"Log verbosity: {self.config.log_verbosity}")

            # Log inherited stability settings in full debug mode
            if use_full_debug:
                self._log_result(
                    "  Full debug logging enabled (inherits Stability tab settings)"
                )
                self._log_result(
                    f"    SC verbosity: {self.config.self_consistency_verbosity}"
                )
                self._log_result(
                    f"    Adaptive timestep debug: {self.config.adaptive_timestep_debug}"
                )

            # Only log detailed config in full debug mode
            if use_full_debug:
                self._log_result(
                    f"Trajectory saving: Top N={self.config.save_top_n_trajectories}, All={self.config.save_all_trajectories}, Failed={self.config.save_failed_trajectories}"
                )

                # Log parameter grid info
                for param_name, values in param_grids.items():
                    if len(values) > 1:
                        self._log_result(
                            f"  {param_name}: {len(values)} points from {values[0]:.2e} to {values[-1]:.2e}"
                        )
                    else:
                        if param_name == "wall_z":
                            self._log_result(
                                f"  {param_name}: {values[0]:.2f} mm (fixed)"
                            )
                        else:
                            self._log_result(f"  {param_name}: {values[0]:.2e} (fixed)")
                self._log_result(
                    f"  Timestep strategy: {self.config.timestep_strategy}"
                )
                if self.config.timestep_strategy == "energy_scaled":
                    self._log_result(
                        f"    Energy scale exponent: {self.config.energy_scale_exponent} (h ∝ γ^-α)"
                    )
                elif self.config.timestep_strategy == "auto_distance":
                    self._log_result(
                        f"    Target distance: {self.config.target_distance_mm:.1f} mm (wall_z + target)"
                    )
                    self._log_result(
                        f"    All particles will travel to consistent z regardless of energy"
                    )
                elif self.config.auto_steps:
                    self._log_result(
                        f"    Legacy auto_steps: wall_z + {self.config.auto_steps_distance_past_wall:.1f} mm"
                    )
                self._log_result(f"  z_cutoff_mode: {self.config.z_cutoff_mode}")

            self._log_result("")

            # Use sweep output directory from GUI preferences
            self.config.output_dir = self.sweep_output_dir

            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            self._log_result(f"Output directory: {self.config.output_dir}")
            self._log_result("")

            # Store all results and failed runs
            all_results = []
            failed_runs = []
            run_num = 0

            # Create parameter combinations using itertools
            import itertools

            param_names = list(param_grids.keys())
            param_values_lists = [param_grids[name] for name in param_names]

            for param_combo in itertools.product(*param_values_lists):
                # Periodic cleanup of matplotlib figures to prevent memory leak
                if run_num > 0 and run_num % 10 == 0:
                    import matplotlib.pyplot as plt

                    plt.close("all")

                # Check for cancellation
                if not self.running:
                    self._log_result("[STOPPED] Sweep stopped by user")
                    break

                if self.gui_controller and hasattr(
                    self.gui_controller, "_cancel_requested"
                ):
                    if self.gui_controller._cancel_requested:
                        self._log_result("[CANCELLED] Sweep cancelled by user")
                        break

                run_num += 1
                progress = run_num / total_runs * 100
                self._update_progress(
                    progress,
                    f"Running simulation {run_num}/{total_runs}...",
                )

                # Extract parameters from combination
                params_dict = dict(zip(param_names, param_combo))

                # Get aperture (only for CONDUCTING_WALL modes)
                aperture = params_dict.get("aperture", 0.001)  # Default if not present

                # Get energy (named differently for BUNCH_TO_BUNCH)
                energy = params_dict.get("initial_energy_gev") or params_dict.get(
                    "energy"
                )

                if energy is None:
                    self._log_result(
                        f"[ERROR] Run {run_num}: No energy parameter found in params_dict!"
                    )
                    self._log_result(
                        f"  Available parameters: {list(params_dict.keys())}"
                    )
                    self._log_result(
                        f"  Simulation type: {self.config.simulation_type}"
                    )
                    continue  # Skip this run

                start_z = params_dict["start_z"]
                offset_frac = params_dict["transverse_offset_fraction"]

                # Get rider particle parameters (either from sweep or fixed values)
                rider_m_particle = params_dict.get(
                    "rider_m_particle", self.config.m_particle
                )
                rider_charge_sign = params_dict.get(
                    "rider_charge_sign", self.config.charge_sign
                )
                rider_pcount = params_dict.get("rider_pcount", self.config.pcount)
                rider_transv_mom = params_dict.get(
                    "rider_transv_mom", self.config.transv_mom
                )
                rider_transv_dist = params_dict.get(
                    "rider_transv_dist", self.config.transv_dist
                )
                rider_stripped_ions = params_dict.get(
                    "rider_stripped_ions", self.config.stripped_ions
                )

                # Get macroparticle parameters (either from sweep or fixed values)
                macroparticle_charge_multiplier = params_dict.get(
                    "macroparticle_charge_multiplier",
                    self.config.macroparticle_charge_multiplier,
                )
                macroparticle_sigma_multiplier = params_dict.get(
                    "macroparticle_sigma_multiplier",
                    self.config.macroparticle_sigma_multiplier,
                )

                # Log parameter values based on verbosity
                if use_full_debug:
                    # Log ALL swept parameter values for this run
                    self._log_result(
                        f"  [PARAMS] Run {run_num}/{total_runs} - All parameters:"
                    )
                    self._log_result(f"    aperture: {aperture:.4e} mm")
                    self._log_result(f"    energy: {energy:.4f} GeV")
                    self._log_result(f"    start_z: {start_z:.4f} mm")
                    self._log_result(f"    transv_offset_frac: {offset_frac:.4f}")
                    self._log_result(
                        f"    rider_m_particle: {rider_m_particle:.4e} amu"
                    )
                    self._log_result(f"    rider_charge_sign: {rider_charge_sign:.1f}")
                    self._log_result(f"    rider_pcount: {rider_pcount}")
                    self._log_result(
                        f"    rider_transv_mom: {rider_transv_mom:.4e} amu·mm/ns"
                    )
                    self._log_result(
                        f"    rider_transv_dist: {rider_transv_dist:.4e} mm"
                    )
                    if self.config.macroparticle_enabled:
                        self._log_result(f"    macroparticle_enabled: True")
                        self._log_result(
                            f"    macroparticle_charge_multiplier: {macroparticle_charge_multiplier:.4f}"
                        )
                        self._log_result(
                            f"    macroparticle_sigma_multiplier: {macroparticle_sigma_multiplier:.4f}"
                        )
                        self._log_result(
                            f"    macroparticle_use_momentum_errors: {self.config.macroparticle_use_momentum_errors}"
                        )

                # Get driver particle parameters if BUNCH_TO_BUNCH
                driver_params_dict = None
                if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
                    driver_m = params_dict.get(
                        "driver_m_particle", self.config.driver_m_particle
                    )

                    # Convert driver_energy_gev to starting_Pz if present,
                    # otherwise fall back to legacy driver_starting_Pz key.
                    driver_neg = getattr(self.config, "driver_direction", "-z") == "-z"
                    if "driver_energy_gev" in params_dict:
                        driver_pz = calculate_starting_pz_from_energy(
                            abs(params_dict["driver_energy_gev"]),
                            driver_m,
                            negative=driver_neg,
                        )
                    else:
                        driver_pz = params_dict.get(
                            "driver_starting_Pz", self.config.driver_starting_Pz
                        )

                    driver_params_dict = {
                        "m_particle": driver_m,
                        "charge_sign": params_dict.get(
                            "driver_charge_sign", self.config.driver_charge_sign
                        ),
                        "pcount": int(
                            params_dict.get("driver_pcount", self.config.driver_pcount)
                        ),
                        "transv_mom": params_dict.get(
                            "driver_transv_mom", self.config.driver_transv_mom
                        ),
                        "transv_dist": params_dict.get(
                            "driver_transv_dist", self.config.driver_transv_dist
                        ),
                        "starting_distance": params_dict.get(
                            "driver_starting_distance",
                            self.config.driver_starting_distance,
                        ),
                        "starting_Pz": driver_pz,
                        "stripped_ions": params_dict.get(
                            "driver_stripped_ions", self.config.driver_stripped_ions
                        ),
                    }

                # Calculate transverse offset
                # For CONDUCTING_WALL/SWITCHING_WALL: fraction of aperture
                # For BUNCH_TO_BUNCH: absolute distance in mm
                sim_type_str = self.config.simulation_type
                if sim_type_str == "BUNCH_TO_BUNCH":
                    transv_offset = offset_frac  # Direct mm value for bunch-to-bunch
                else:
                    transv_offset = (
                        offset_frac * aperture
                    )  # Fraction for conducting wall

                # Calculate timestep based on strategy
                if self.config.timestep_strategy != "fixed":
                    # Use energy-aware timestep calculation
                    # Get wall_z for this run (it may be swept)
                    wall_z_for_calc = params_dict.get("wall_z", self.config.wall_z)

                    # Get driver starting position for BUNCH_TO_BUNCH mode
                    driver_start_z = 1000.0  # Default
                    if driver_params_dict is not None:
                        driver_start_z = driver_params_dict.get(
                            "starting_distance", 1000.0
                        )

                    timestep = self.config.calculate_timestep_for_energy(
                        energy,
                        rider_m_particle,
                        wall_z=wall_z_for_calc,
                        start_z=start_z,
                        driver_start_z=driver_start_z,
                    )
                    steps = self.config.steps

                    # Calculate gamma for diagnostics (ALWAYS log for debugging)
                    AMU_TO_MEV = 931.494
                    rest_energy_mev = rider_m_particle * AMU_TO_MEV

                    # For BUNCH_TO_BUNCH, energy is kinetic; for others, it's total
                    if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
                        gamma = (energy * 1e3) / rest_energy_mev + 1.0
                    else:
                        gamma = (energy * 1e3) / rest_energy_mev

                    beta = (
                        np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.999
                    )
                    distance_per_step = beta * gamma * C_MMNS * timestep
                    expected_distance = distance_per_step * steps

                    if use_full_debug:
                        self._log_result(
                            f"  [TIMESTEP] Run {run_num} strategy '{self.config.timestep_strategy}':"
                        )
                        self._log_result(
                            f"    E={energy:.4f} GeV, m={rider_m_particle:.4e} amu"
                        )
                        self._log_result(f"    gamma={gamma:.2f}, beta={beta:.8f}")
                        self._log_result(
                            f"    timestep h={timestep:.4e} ns (proper time = dt/gamma)"
                        )
                        self._log_result(f"    steps={steps}")
                        self._log_result(
                            f"    distance_per_step = β·γ·c·h = {distance_per_step:.4f} mm"
                        )
                        self._log_result(
                            f"    expected_total_distance = {expected_distance:.2f} mm"
                        )
                        # Use wall_z from grid if available, otherwise use config default
                        current_wall_z = params_dict.get("wall_z", self.config.wall_z)
                        self._log_result(
                            f"    wall_z={current_wall_z:.2f} mm, start_z={start_z:.2f} mm"
                        )
                        self._log_result(
                            f"    distance_to_wall = {abs(current_wall_z - start_z):.2f} mm"
                        )
                        if self.config.timestep_strategy == "auto_distance":
                            self._log_result(
                                f"    target_distance={self.config.target_distance_mm:.2f} mm"
                            )
                elif self.config.auto_steps:
                    # Legacy auto_steps mode (deprecated, but keep for compatibility)
                    current_wall_z = params_dict.get("wall_z", self.config.wall_z)
                    distance_to_wall = abs(current_wall_z - start_z)
                    total_distance = (
                        distance_to_wall + self.config.auto_steps_distance_past_wall
                    )

                    timestep = calculate_auto_timestep(
                        start_z=start_z,
                        wall_z=current_wall_z,
                        distance_past_wall=self.config.auto_steps_distance_past_wall,
                        particle_energy_gev=energy,
                        particle_mass_amu=rider_m_particle,
                        target_steps=self.config.auto_steps_target,
                    )
                    steps = calculate_auto_steps(
                        start_z=start_z,
                        wall_z=current_wall_z,
                        distance_past_wall=self.config.auto_steps_distance_past_wall,
                        timestep=timestep,
                        particle_energy_gev=energy,
                        particle_mass_amu=rider_m_particle,
                    )
                else:
                    timestep = self.config.timestep
                    steps = self.config.steps

                # Enforce minimum of 5% of requested steps (absolute floor of 20)
                min_steps = max(20, int(self.config.steps * 0.05))
                if steps < min_steps:
                    if use_full_debug:
                        self._log_result(
                            f"  [WARNING] Steps adjusted from {steps} to {min_steps} (minimum floor)"
                        )
                    steps = min_steps

                # Log run start summary (only in full debug mode - truncated mode logs after completion)
                if use_full_debug:
                    self._log_result(
                        f"  [START] Run {run_num}/{total_runs}: "
                        f"a={aperture:.4e}mm, E={energy:.4f}GeV, z={start_z:.2f}mm, "
                        f"h={timestep:.4e}ns, N={steps}"
                    )

                # Run integration with timeout and retry logic
                result = None
                run_error = None
                run_timed_out = False
                retry_attempt = 0
                max_retries = self.config.failed_run_retry_attempts

                # Loop for retry attempts (1 original + max_retries additional attempts)
                while retry_attempt <= max_retries:
                    # Check for global cancellation before starting a retry
                    if not self.running:
                        self._log_result(
                            f"  [CANCEL] Run {run_num}: Cancellation requested"
                        )
                        break
                    if self.gui_controller and hasattr(
                        self.gui_controller, "_cancel_requested"
                    ):
                        if self.gui_controller._cancel_requested:
                            self._log_result(
                                f"  [CANCEL] Run {run_num}: Cancellation requested"
                            )
                            break

                    # Generate seed for this attempt
                    if retry_attempt == 0:
                        # First attempt uses config seed
                        current_seed = self.config.seed
                    else:
                        # Retry attempts use deterministic but different seed
                        current_seed = (
                            self.config.seed + run_num * 10000 + retry_attempt * 100
                        )
                        if use_full_debug or use_truncated_logging:
                            self._log_result(
                                f"  [RETRY] Run {run_num}, attempt {retry_attempt}/{max_retries} with new seed {current_seed}"
                            )

                    # Reset error/timeout flags for this attempt
                    attempt_result = None
                    attempt_error = None
                    attempt_timed_out = False

                    try:
                        # Check if timeout is enabled
                        if self.config.per_run_timeout > 0:
                            import threading

                            # Container for result (mutable for thread access)
                            result_container = [None]
                            error_container = [None]
                            cancel_flag = [False]  # Flag to signal cancellation

                            # Log warning for potentially problematic parameter combinations
                            if (
                                aperture < 0.1
                                and macroparticle_charge_multiplier > 1000
                            ):
                                self._log_result(
                                    f"  [WARNING] Run {run_num}: Very small aperture ({aperture:.4f} mm) "
                                    f"with large charge multiplier ({macroparticle_charge_multiplier:.0f})"
                                )
                                self._log_result(
                                    f"    This may cause numerical instability or slow convergence"
                                )

                            def run_with_exception_handling():
                                """Wrapper to run integration and catch exceptions."""
                                try:
                                    result_container[0] = self._run_single_integration(
                                        aperture=aperture,
                                        energy_gev=energy,
                                        start_z=start_z,
                                        transv_offset=transv_offset,
                                        timestep=timestep,
                                        steps=steps,
                                        rider_m_particle=rider_m_particle,
                                        rider_charge_sign=rider_charge_sign,
                                        rider_pcount=int(rider_pcount),
                                        rider_transv_mom=rider_transv_mom,
                                        rider_transv_dist=rider_transv_dist,
                                        rider_stripped_ions=rider_stripped_ions,
                                        macroparticle_charge_multiplier=macroparticle_charge_multiplier,
                                        macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
                                        driver_params=driver_params_dict,
                                        wall_z=params_dict.get(
                                            "wall_z", self.config.wall_z
                                        ),
                                        run_num=run_num,
                                        cancel_flag=cancel_flag,
                                        seed_override=current_seed,
                                    )
                                except Exception as e:
                                    error_container[0] = e

                            # Start integration in separate thread
                            integration_thread = threading.Thread(
                                target=run_with_exception_handling
                            )
                            integration_thread.daemon = True
                            integration_thread.start()

                            # Wait for completion or timeout
                            integration_thread.join(
                                timeout=self.config.per_run_timeout
                            )

                            if integration_thread.is_alive():
                                # Timeout occurred - signal the integration to cancel
                                attempt_timed_out = True
                                cancel_flag[0] = True
                                self._log_result(
                                    f"  [TIMEOUT] Run {run_num} exceeded timeout of {self.config.per_run_timeout}s"
                                )
                                self._log_result(
                                    f"    Signaling integration to cancel (thread will terminate when it checks cancel flag)"
                                )
                                # Give it a brief moment to respond to cancellation
                                integration_thread.join(timeout=2.0)
                                if integration_thread.is_alive():
                                    self._log_result(
                                        f"    Warning: Integration thread still running after cancel signal"
                                    )
                                    self._log_result(
                                        f"    Thread will be abandoned (daemon thread will terminate with main thread)"
                                    )

                            if error_container[0] is not None:
                                attempt_error = error_container[0]
                            else:
                                attempt_result = result_container[0]
                        else:
                            # No timeout - run directly
                            attempt_result = self._run_single_integration(
                                aperture=aperture,
                                energy_gev=energy,
                                start_z=start_z,
                                transv_offset=transv_offset,
                                timestep=timestep,
                                steps=steps,
                                rider_m_particle=rider_m_particle,
                                rider_charge_sign=rider_charge_sign,
                                rider_pcount=int(rider_pcount),
                                rider_transv_mom=rider_transv_mom,
                                rider_transv_dist=rider_transv_dist,
                                rider_stripped_ions=rider_stripped_ions,
                                macroparticle_charge_multiplier=macroparticle_charge_multiplier,
                                macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
                                driver_params=driver_params_dict,
                                wall_z=params_dict.get("wall_z", self.config.wall_z),
                                run_num=run_num,
                                cancel_flag=None,
                                seed_override=current_seed,
                            )

                    except Exception as e:
                        attempt_error = e
                        if use_full_debug:
                            import traceback

                            self._log_result(
                                f"  [ERROR] Run {run_num} attempt {retry_attempt} exception: {type(e).__name__}: {e}"
                            )
                            self._log_result(
                                f"    Traceback:\n{traceback.format_exc()}"
                            )

                    # Check if this attempt succeeded
                    attempt_succeeded = False
                    if (
                        not attempt_timed_out
                        and attempt_error is None
                        and attempt_result is not None
                    ):
                        # Check if result has valid metrics (not all particles dead)
                        is_halted = attempt_result.get("halted_early", False)
                        metrics = attempt_result.get("metrics", {})

                        # DEBUG: Log what we're checking
                        if use_full_debug:
                            self._log_result(
                                f"  [DEBUG] Run {run_num} attempt {retry_attempt}: is_halted={is_halted}, has_metrics={bool(metrics)}"
                            )
                            if metrics:
                                self._log_result(
                                    f"    max_percent_energy_gain={metrics.get('max_percent_energy_gain')}"
                                )
                                self._log_result(
                                    f"    rider_gamma_final={metrics.get('rider_gamma_final')}"
                                )
                                self._log_result(
                                    f"    rider_delta_e_mev={metrics.get('rider_delta_e_mev')}"
                                )

                        has_useful_metrics = False
                        if not is_halted and metrics:
                            # Check for key optimization metrics
                            if metrics.get("max_percent_energy_gain") is not None:
                                has_useful_metrics = True
                            elif (
                                metrics.get("rider_gamma_final") is not None
                                and metrics.get("rider_gamma_final") > 0
                            ):
                                has_useful_metrics = True
                            elif metrics.get("rider_delta_e_mev") is not None:
                                has_useful_metrics = True

                        if has_useful_metrics:
                            # Success! Use this result
                            result = attempt_result
                            run_error = None
                            run_timed_out = False
                            attempt_succeeded = True
                            if retry_attempt > 0:
                                self._log_result(
                                    f"  [SUCCESS] Run {run_num} succeeded on retry attempt {retry_attempt}"
                                )
                            break
                        else:
                            # No useful metrics - all particles died or halted early
                            halt_reason = attempt_result.get("halt_reason", "unknown")
                            attempt_error = Exception(
                                f"Run failed: halted_early={is_halted}, reason={halt_reason}"
                            )
                            if use_full_debug or use_truncated_logging:
                                self._log_result(
                                    f"  [FAILED] Run {run_num} attempt {retry_attempt}: halted={is_halted}, has_metrics={bool(metrics)}, has_useful_metrics=False"
                                )

                    # If we got here without breaking, the attempt failed
                    if not attempt_succeeded:
                        if use_full_debug:
                            error_msg = f"  [DEBUG] Run {run_num} attempt {retry_attempt} failed: timeout={attempt_timed_out}, error={attempt_error is not None}"
                            if attempt_error is not None:
                                error_msg += f" ({type(attempt_error).__name__}: {attempt_error})"
                            self._log_result(error_msg)

                        # Decide whether to retry
                        if retry_attempt < max_retries:
                            # Will retry
                            retry_attempt += 1
                            continue
                        else:
                            # No more retries - record the final failure
                            result = attempt_result
                            run_error = attempt_error
                            run_timed_out = attempt_timed_out
                            break

                # Handle results (after all retry attempts)
                try:
                    if result is not None and use_full_debug:
                        self._log_result(
                            f"  [DEBUG] Run {run_num} integration completed"
                        )

                    if not run_timed_out and result is not None:
                        # Extract metrics
                        delta_e = result.get("metrics", {}).get(
                            "rider_delta_e_mev", 0.0
                        )
                        delta_gamma = result.get("metrics", {}).get(
                            "rider_delta_gamma", 0.0
                        )
                        gamma_initial = result.get("metrics", {}).get(
                            "rider_gamma_initial", 0.0
                        )
                        gamma_final = result.get("metrics", {}).get(
                            "rider_gamma_final", 0.0
                        )

                        # Create run_data structure (used regardless of logging mode)
                        run_data = {
                            "run_number": run_num,
                            "parameters": {
                                "aperture_radius": aperture,
                                "particle_energy_gev": energy,
                                "start_z": start_z,
                                "transverse_offset": transv_offset,
                                "transverse_offset_fraction": offset_frac,
                                "timestep": timestep,
                                "steps": steps,
                                "retry_attempts": retry_attempt,
                                "wall_z": params_dict.get("wall_z", self.config.wall_z),
                                "rider_m_particle": rider_m_particle,
                                "rider_charge_sign": rider_charge_sign,
                                "rider_pcount": int(rider_pcount),
                                "rider_transv_mom": rider_transv_mom,
                                "rider_transv_dist": rider_transv_dist,
                                "macroparticle_charge_multiplier": macroparticle_charge_multiplier,
                                "macroparticle_sigma_multiplier": macroparticle_sigma_multiplier,
                                "simulation_type": self.config.simulation_type.name,
                            },
                            "metrics": result.get("metrics", {}),
                        }

                        # Log based on verbosity mode
                        if use_no_logging:
                            # No logging mode: skip all run-level logs
                            pass
                        elif use_truncated_logging:
                            # Truncated mode: 1-2 lines with key info
                            # Build log params from actual swept parameters
                            log_params = {}

                            # Include parameters that have multiple values (i.e., are being swept)
                            for param_name in param_grids.keys():
                                if len(param_grids[param_name]) > 1:
                                    # This parameter is being swept - include it
                                    if param_name in params_dict:
                                        log_params[param_name] = params_dict[param_name]

                            # If no parameters are being swept (all fixed), show key simulation params
                            if not log_params:
                                # For BUNCH_TO_BUNCH, show initial_energy_gev if present
                                if (
                                    self.config.simulation_type
                                    == SimulationType.BUNCH_TO_BUNCH
                                ):
                                    if "initial_energy_gev" in params_dict:
                                        log_params["initial_energy_gev"] = params_dict[
                                            "initial_energy_gev"
                                        ]
                                    if "driver_starting_distance" in params_dict:
                                        log_params["driver_starting_distance"] = (
                                            params_dict["driver_starting_distance"]
                                        )
                                else:
                                    # For CONDUCTING_WALL modes
                                    log_params["aperture"] = aperture
                                    log_params["energy"] = energy

                                # Always show wall_z if present
                                if "wall_z" in params_dict:
                                    log_params["wall_z"] = params_dict["wall_z"]
                                elif hasattr(self.config, "wall_z"):
                                    log_params["wall_z"] = self.config.wall_z

                            self._log_truncated_run(
                                run_num,
                                params=log_params,
                                metrics={
                                    "ΔE": delta_e,
                                    "Δγ": delta_gamma,
                                    "γ_i": gamma_initial,
                                    "γ_f": gamma_final,
                                },
                            )
                        elif use_full_debug:
                            # Full debug mode: all details
                            # Extract actual trajectory distance for diagnostics
                            actual_distance = 0.0
                            if "_distance_info" in result:
                                dist_info = result["_distance_info"]
                                actual_distance = abs(
                                    dist_info["z_end"] - dist_info["z_start"]
                                )
                            elif "trajectory" in result and result["trajectory"]:
                                # Fallback: try to extract from full trajectory if present
                                traj = result["trajectory"]
                                z_vals = traj.get("z", [])
                                if len(z_vals) > 1:
                                    # Safely handle both lists and numpy arrays
                                    z_start = float(np.asarray(z_vals[0]).flat[0])
                                    z_end = float(np.asarray(z_vals[-1]).flat[0])
                                    actual_distance = abs(z_end - z_start)

                            self._log_result(f"  [RESULT] Run {run_num}/{total_runs}:")
                            self._log_result(
                                f"    Distance: expected={expected_distance:.2f}mm, actual={actual_distance:.2f}mm"
                            )
                            self._log_result(
                                f"    Gamma: initial={gamma_initial:.6f}, final={gamma_final:.6f}, delta={delta_gamma:.6e}"
                            )
                            self._log_result(f"    Energy: ΔE={delta_e:.6f}MeV")
                            if actual_distance < 0.1:
                                self._log_result(
                                    f"  [WARNING] Particle barely moved! Check timestep calculation."
                                )

                        # Add trajectory if requested (check if any trajectory saving is enabled)
                        # Note: save_top_n_trajectories only applies to optimization mode, not sweeps
                        save_traj = (
                            self.config.save_all_trajectories
                            or self.config.save_failed_trajectories
                        )
                        if save_traj and "trajectory" in result:
                            run_data["trajectory"] = result["trajectory"]

                        # Add driver params to stored results if applicable
                        if driver_params_dict is not None:
                            run_data["parameters"].update(
                                {
                                    f"driver_{k}": v
                                    for k, v in driver_params_dict.items()
                                }
                            )

                        all_results.append(run_data)

                except Exception as e:
                    import traceback

                    error_details = traceback.format_exc()
                    run_error = str(e)

                    if self.config.skip_failed_runs:
                        self._log_result(f"[WARNING] Run {run_num} failed: {e}")
                        self._log_result(f"    Error details: {error_details}")
                        self._log_result(
                            f"    Skipping and continuing with next run..."
                        )

                        # Record failed run
                        failed_runs.append(
                            {
                                "run_number": run_num,
                                "parameters": {
                                    "aperture_radius": aperture,
                                    "particle_energy_gev": energy,
                                    "start_z": start_z,
                                    "transverse_offset": transv_offset,
                                    "timestep": timestep,
                                    "steps": steps,
                                    "wall_z": params_dict.get(
                                        "wall_z", self.config.wall_z
                                    ),
                                },
                                "error": run_error,
                                "error_details": error_details,
                            }
                        )
                    else:
                        # Don't skip - re-raise and stop sweep
                        self._log_result(f"[ERROR] Run {run_num} failed: {e}")
                        self._log_result(f"    Error details: {error_details}")
                        self._log_result(
                            f"    Stopping sweep (skip_failed_runs is disabled)"
                        )
                        raise

                # Handle timeout case
                if run_timed_out:
                    if self.config.skip_failed_runs:
                        self._log_result(
                            f"    Skipping and continuing with next run..."
                        )
                        failed_runs.append(
                            {
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
                                "timeout_seconds": self.config.per_run_timeout,
                            }
                        )
                    else:
                        self._log_result(
                            f"    Stopping sweep (skip_failed_runs is disabled)"
                        )
                        break

            # Save results
            if all_results and self.config.save_results:
                self._save_sweep_results(all_results, failed_runs)

            if self.running:
                elapsed_time = time.time() - start_time
                hours = int(elapsed_time // 3600)
                minutes = int((elapsed_time % 3600) // 60)
                seconds = elapsed_time % 60

                self._log_result("[OK] Sweep completed!")
                self._log_result(f"  Results saved to: {self.config.output_dir}")
                self._log_result(f"  Successful runs: {len(all_results)}")
                if failed_runs:
                    self._log_result(f"  Failed/timed-out runs: {len(failed_runs)}")
                if hours > 0:
                    self._log_result(
                        f"  Total time: {hours}h {minutes}m {seconds:.1f}s ({elapsed_time:.1f}s)"
                    )
                elif minutes > 0:
                    self._log_result(
                        f"  Total time: {minutes}m {seconds:.1f}s ({elapsed_time:.1f}s)"
                    )
                else:
                    self._log_result(f"  Total time: {elapsed_time:.1f}s")
                self._update_progress(100, "Complete!")
        except Exception as e:
            self._log_result(f"[ERROR] Error during sweep: {e}")
            import traceback

            self._log_result(traceback.format_exc())
        finally:
            # Restore original verbosity settings
            if "original_sc_verbosity" in locals():
                self.config.self_consistency_verbosity = original_sc_verbosity
            if "original_adaptive_debug" in locals():
                self.config.adaptive_timestep_debug = original_adaptive_debug

            self.running = False
            # Ensure log file is closed
            if self._log_file is not None:
                self._close_log_file()
            # Clean up any remaining matplotlib figures
            import matplotlib.pyplot as plt

            plt.close("all")
            # Update UI back to ready state
            self.after(100, self._reset_ui_state)

    def _generate_parameter_grids(self):
        """Generate all parameter grids including sweepable parameters."""
        return build_parameter_grids(self.config, self.sweep_params)

    def _generate_range(
        self, min_val: float, max_val: float, points: int, log_scale: bool
    ) -> List[float]:
        """Generate parameter range (linear or log scale)."""
        return generate_parameter_range(min_val, max_val, points, log_scale)

    def _run_single_integration(
        self,
        aperture: float,
        energy_gev: float,
        start_z: float,
        transv_offset: float,
        timestep: float,
        steps: int,
        rider_m_particle: float = None,
        rider_charge_sign: float = None,
        rider_pcount: int = None,
        rider_transv_mom: float = None,
        rider_transv_dist: float = None,
        rider_stripped_ions: float = None,
        macroparticle_charge_multiplier: float = None,
        macroparticle_sigma_multiplier: float = None,
        driver_params: Dict[str, Any] = None,
        wall_z: float = None,
        run_num: int = 0,
        cancel_flag: Optional[List[bool]] = None,
        seed_override: int = None,
    ) -> Dict[str, Any]:
        """Run a single integration with given parameters."""
        # Log stability analysis configuration for debugging
        self._log_result(f"  [CONFIG] Run {run_num} stability settings:")
        self._log_result(f"    smoothness_enabled: {self.config.smoothness_enabled}")
        if self.config.smoothness_enabled:
            self._log_result(
                f"    smoothness_window_size: {self.config.smoothness_window_size}"
            )
            self._log_result(
                f"    smoothness_reject_on_violation: {self.config.smoothness_reject_on_violation}"
            )

        # Use provided rider values or fall back to config defaults
        rider_m_particle = (
            rider_m_particle if rider_m_particle is not None else self.config.m_particle
        )
        rider_charge_sign = (
            rider_charge_sign
            if rider_charge_sign is not None
            else self.config.charge_sign
        )
        rider_pcount = (
            rider_pcount if rider_pcount is not None else int(self.config.pcount)
        )
        rider_transv_mom = (
            rider_transv_mom if rider_transv_mom is not None else self.config.transv_mom
        )
        rider_transv_dist = (
            rider_transv_dist
            if rider_transv_dist is not None
            else self.config.transv_dist
        )
        rider_stripped_ions = (
            rider_stripped_ions
            if rider_stripped_ions is not None
            else self.config.stripped_ions
        )
        wall_z = wall_z if wall_z is not None else self.config.wall_z
        macroparticle_charge_multiplier = (
            macroparticle_charge_multiplier
            if macroparticle_charge_multiplier is not None
            else self.config.macroparticle_charge_multiplier
        )
        macroparticle_sigma_multiplier = (
            macroparticle_sigma_multiplier
            if macroparticle_sigma_multiplier is not None
            else self.config.macroparticle_sigma_multiplier
        )

        # Build rider params
        # transv_offset is the radial offset from axis (in mm)
        # This is now properly used as an offset, not as spread
        rider_params = {
            "starting_distance": start_z,
            "transv_mom": rider_transv_mom,
            "transv_dist": rider_transv_dist,  # Use parameter, not config
            "transv_offset_x": transv_offset,  # Radial offset as x-offset
            "transv_offset_y": 0.0,  # Keep on x-axis (radial offset in x-direction)
            "m_particle": rider_m_particle,
            "charge_sign": rider_charge_sign,
            "pcount": rider_pcount,
            "stripped_ions": rider_stripped_ions,
            "starting_Pz": 0.0,  # Will be calculated from energy
        }

        # Calculate initial Pz from energy
        # E = gamma * m * c^2, where m*c^2 in MeV
        AMU_TO_MEV = 931.494
        rest_energy_mev = rider_m_particle * AMU_TO_MEV

        # For BUNCH_TO_BUNCH, energy is kinetic energy; for others, it's total energy
        if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
            # Kinetic energy: γ = (KE / E_rest) + 1
            gamma = (energy_gev * 1e3) / rest_energy_mev + 1.0
        else:
            # Total energy: γ = E_total / E_rest
            gamma = (energy_gev * 1e3) / rest_energy_mev

        # Legacy init_bunch expects starting_Pz as specific momentum (momentum/mass)
        # It calculates: Pz = starting_Pz * mass, then γ = sqrt((Pz/(mc))² + 1)
        # Working backwards: γ² = (Pz/(mc))² + 1 = (starting_Pz/c)² + 1
        # Therefore: starting_Pz = c·sqrt(γ² - 1)
        rider_params["starting_Pz"] = C_MMNS * np.sqrt(gamma * gamma - 1.0)

        core_params = {
            "time_step": timestep,
            "wall_z": wall_z,
            "aperture_radius": aperture,
            "mean": 1.0e5,  # Large value (not used for CONDUCTING_WALL)
            "cav_spacing": 1.0e5,
            "z_cutoff": (
                self.config.target_distance_mm
                if self.config.z_cutoff_mode == "relative"
                else 0.0
            ),
            "z_cutoff_mode": self.config.z_cutoff_mode,
            "startup_mode": self.config.startup_mode,
        }

        # Create a temporary subdirectory for this run's outputs (will be cleaned up)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        # Create a temporary subdirectory for this run's outputs (will be cleaned up)
        # IMPORTANT: This must live under the same base directory that the orphan-cleanup
        # routine scans (self.sweep_output_dir), otherwise temp dirs will only be cleaned
        # up when the GUI starts (or never, if output_dir differs).
        run_output_dir = (
            Path(self.sweep_output_dir) / f"_temp_run_{run_num}_{timestamp}"
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Use seed override if provided (for retries), otherwise use config seed + run_num
        actual_seed = (
            seed_override if seed_override is not None else (self.config.seed + run_num)
        )

        options = SimulationOptions(
            steps=steps,
            seed=actual_seed,  # Unique seed per run for varied particle distributions
            simulation_type=self.config.simulation_type,
            rider_params=rider_params,
            driver_params=driver_params,  # Use provided driver params (None for CONDUCTING_WALL)
            core_params=core_params,
            legacy_enabled=False,
            trajectory_save=False,  # Don't save individual trajectory files to disk
            trajectory_interval=self.config.trajectory_stride,
            energy_display=False,  # Don't generate or display plots during sweep
            energy_save=False,
            transverse_display=False,
            transverse_save=True,  # Always return trajectory data for metrics calculation
            beta_display=False,  # Don't generate beta plots
            beta_save=False,
            momentum_display=False,  # Don't generate momentum plots
            momentum_save=False,
            gamma_display=False,  # Don't generate gamma plots
            gamma_save=False,
            zposition_display=False,  # Don't generate z-position plots
            zposition_save=False,
            macroparticle_enabled=self.config.macroparticle_enabled,
            macroparticle_charge_multiplier=macroparticle_charge_multiplier,
            macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
            macroparticle_use_momentum_errors=self.config.macroparticle_use_momentum_errors,
            image_subcharge_count=self.config.image_subcharge_count,
            use_image_weighting=self.config.use_image_weighting,
            overlay_display=False,
            overlay_save=False,
            difference_display=False,
            difference_save=False,
            metrics_save=False,
            output_dir=run_output_dir,
            # Use stability options from sweep config
            self_consistency_enabled=self.config.self_consistency_enabled,
            self_consistency_tolerance=self.config.self_consistency_tolerance,
            self_consistency_max_iterations=self.config.self_consistency_max_iterations,
            self_consistency_verbosity=self.config.self_consistency_verbosity,
            energy_monitor_enabled=False,  # Removed - functionality in adaptive timestep
            energy_monitor_threshold=2.0,
            energy_monitor_check_interval=10,
            energy_monitor_halt_on_jump=self.config.energy_monitor_halt_on_jump,
            energy_monitor_debug=False,  # Removed
            adaptive_timestep_enabled=self.config.adaptive_timestep_enabled,
            adaptive_timestep_threshold=self.config.adaptive_timestep_threshold,
            adaptive_timestep_reduction_factor=self.config.adaptive_timestep_reduction_factor,
            adaptive_timestep_min_factor=self.config.adaptive_timestep_min_factor,
            adaptive_timestep_cooldown_steps=self.config.adaptive_timestep_cooldown_steps,
            adaptive_timestep_probe_threshold=self.config.adaptive_timestep_probe_threshold,
            adaptive_timestep_max_probe_steps=self.config.adaptive_timestep_max_probe_steps,
            adaptive_timestep_debug=self.config.adaptive_timestep_debug,
        )

        # Create progress callback to track integration
        def progress_callback(current: int, total: int, run_id=run_num):
            """Log progress periodically."""
            # Log every 10% or every 100 steps for short runs
            if total <= 1000:
                log_interval = max(1, total // 10)
            else:
                log_interval = max(100, total // 20)

            if current % log_interval == 0 or current == total:
                self._log_result(
                    f"    [PROGRESS] Run {run_id}: step {current}/{total} "
                    f"({100 * current // total}%)"
                )

        # Run the integration with progress tracking
        #
        # NOTE: We must always clean up the per-run temp directory, even when returning
        # early (halted runs) or raising exceptions. We do that by wrapping the entire
        # run/analysis section in a try/finally.
        try:
            # Log diagnostic info for potentially problematic configurations
            # Only check aperture for CONDUCTING_WALL modes
            if (
                self.config.simulation_type != SimulationType.BUNCH_TO_BUNCH
                and aperture < 0.1
            ):
                self._log_result(
                    f"  [DIAGNOSTIC] Run {run_num}: Small aperture detected ({aperture:.6f} mm)"
                )
            if macroparticle_charge_multiplier > 1000:
                self._log_result(
                    f"  [DIAGNOSTIC] Run {run_num}: Large charge multiplier ({macroparticle_charge_multiplier:.0f})"
                )
                self._log_result(
                    "    Note: This may significantly slow integration due to strong image forces"
                )

            self._log_result(f"  [DEBUG] Calling run_testbed for Run {run_num}...")

            # Create cancel callback if cancel_flag is provided
            cancel_callback = None
            if cancel_flag is not None:

                def check_cancel():
                    if cancel_flag[0]:
                        self._log_result(
                            f"  [CANCEL] Run {run_num}: Cancellation requested"
                        )
                    return cancel_flag[0] if cancel_flag else False

                cancel_callback = check_cancel

            # Create log callback to stream verbose SC/adaptive timestep output to GUI
            # This ensures logs are visible in real-time even when not saved to file
            log_callback = None
            if (
                self.config.self_consistency_verbosity > 0
                or self.config.adaptive_timestep_debug
            ):

                def verbose_log(message: str):
                    # Filter for SC and adaptive timestep related messages
                    if any(
                        keyword in message
                        for keyword in [
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
                    ):
                        self._log_result(f"    [VERBOSE] {message}")

                log_callback = verbose_log

            result = run_testbed(
                options,
                log=log_callback,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )
            self._log_result(f"  [DEBUG] run_testbed completed for Run {run_num}")

            # Check if integration was halted early
            if result.halted_early:
                self._log_result(
                    f"  [WARNING] Run {run_num} halted early: {result.halt_reason}"
                )
                self._log_result(
                    "    Trajectory contains partial data and will still be analyzed"
                )

            # Sanity check: Verify final z position doesn't exceed expected distance
            if (
                result.rider_trajectory is not None
                and self.config.timestep_strategy == "auto_distance"
            ):
                try:
                    traj = result.rider_trajectory
                    z_array = np.asarray(traj.get("z", []))
                    if len(z_array) > 0:
                        final_z = float(z_array[-1])

                        # Calculate expected distance based on simulation type
                        if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
                            # For BUNCH_TO_BUNCH: target is driver_start + target_distance
                            if driver_params is not None:
                                driver_start_z = driver_params.get(
                                    "starting_distance", 1000.0
                                )
                            else:
                                driver_start_z = 1000.0
                            expected_max_z = (
                                abs(driver_start_z) + self.config.target_distance_mm
                            )
                        else:
                            # For CONDUCTING_WALL/SWITCHING_WALL: target is wall + target_distance
                            expected_max_z = wall_z + self.config.target_distance_mm

                        if final_z > expected_max_z:
                            excess = final_z - expected_max_z
                            self._log_result(
                                f"  [WARNING] Run {run_num}: Final z position EXCEEDED expected distance!"
                            )
                            self._log_result(f"    Final z: {final_z:.2f} mm")
                            if (
                                self.config.simulation_type
                                == SimulationType.BUNCH_TO_BUNCH
                            ):
                                self._log_result(
                                    f"    Expected max z: {expected_max_z:.2f} mm (driver_start + target={self.config.target_distance_mm:.2f})"
                                )
                            else:
                                self._log_result(
                                    f"    Expected max z: {expected_max_z:.2f} mm (wall_z={wall_z:.2f} + target={self.config.target_distance_mm:.2f})"
                                )
                            self._log_result(
                                f"    Exceeded by: {excess:.2f} mm ({excess / expected_max_z * 100:.1f}%)"
                            )
                        else:
                            under = expected_max_z - final_z
                            self._log_result(
                                f"  [DEBUG] Run {run_num}: Final z check OK"
                            )
                            self._log_result(
                                f"    Final z: {final_z:.2f} mm (under by {under:.2f} mm)"
                            )
                except Exception as e:
                    self._log_result(
                        f"  [WARNING] Run {run_num}: Failed to check final z position: {e}"
                    )

            # No figures should be generated during sweeps (all display/save flags set to False)
            # If any figures were created (shouldn't happen), close them as a safety measure
            if result.figures:
                self._log_result(
                    f"  [WARNING] Run {run_num}: Unexpected figures generated ({len(result.figures)}), closing them"
                )
                import matplotlib.pyplot as plt

                for fig_name, fig in result.figures.items():
                    try:
                        plt.close(fig)
                        self._log_result(f"    Closed unexpected figure: {fig_name}")
                    except Exception as e:
                        self._log_result(f"    Error closing figure {fig_name}: {e}")

            # Check if run was halted early - if so, skip metrics calculation
            if result.halted_early:
                self._log_result(
                    f"  [INFO] Run {run_num} was halted early - skipping metrics calculation"
                )
                self._log_result(
                    "    Only trajectory and logs will be saved (if enabled)"
                )
                output = {
                    "metrics": {},
                    "halted_early": True,
                    "halt_reason": result.halt_reason,
                }

                # Add trajectory if available and saving is enabled
                if result.rider_trajectory is not None:
                    save_traj = (
                        self.config.save_all_trajectories
                        or self.config.save_failed_trajectories
                    )
                    if save_traj:
                        traj = result.rider_trajectory
                        stride = self.config.trajectory_stride
                        try:
                            output["trajectory"] = {
                                "z": np.asarray(traj["z"])[::stride].tolist(),
                                "r": np.asarray(traj["r"])[::stride].tolist(),
                                "pz": np.asarray(traj["pz"])[::stride].tolist(),
                                "pr": np.asarray(traj["pr"])[::stride].tolist(),
                                "t": np.asarray(traj["t"])[::stride].tolist(),
                                "gamma": np.asarray(traj["gamma"])[::stride].tolist(),
                            }
                            self._log_result(
                                f"    Halted trajectory saved ({len(traj['z'])} points, stride={stride})"
                            )
                        except Exception as e:
                            self._log_result(
                                f"    [WARNING] Failed to save halted trajectory: {e}"
                            )

                self._log_result(
                    f"  [DEBUG] _run_single_integration returning for halted Run {run_num}"
                )
                return output

            # Extract metrics (only for non-halted runs)
            self._log_result(f"  [DEBUG] Extracting metrics for Run {run_num}...")
            metrics = {}
            if result.rider_delta_e is not None:
                metrics["rider_delta_e_mev"] = result.rider_delta_e
            if result.rider_gamma_initial is not None:
                metrics["rider_gamma_initial"] = result.rider_gamma_initial
            if result.rider_gamma_final is not None:
                metrics["rider_gamma_final"] = result.rider_gamma_final

            # Calculate max_percent_energy_gain from gamma values
            gamma_initial = result.rider_gamma_initial
            gamma_final = result.rider_gamma_final

            # Diagnostic logging
            self._log_result(f"  [RESULT] Run {run_num} metrics:")
            self._log_result(f"    rider_gamma_initial: {gamma_initial}")
            self._log_result(f"    rider_gamma_final: {gamma_final}")

            # Try to calculate from available gamma values
            if (
                gamma_initial is not None
                and gamma_final is not None
                and gamma_initial > 0
            ):
                delta_gamma = gamma_final - gamma_initial
                energy_gain_percent = delta_gamma / gamma_initial * 100.0
                energy_gain_ppm = delta_gamma / gamma_initial * 1e6
                rest_energy_mev = rider_m_particle * AMU_TO_MEV
                delta_e_mev = delta_gamma * rest_energy_mev

                metrics["max_percent_energy_gain"] = energy_gain_percent
                metrics["percent_delta_e"] = energy_gain_percent
                metrics["delta_gamma"] = delta_gamma
                metrics["delta_e_mev"] = delta_e_mev
                metrics["energy_gain_ppm"] = energy_gain_ppm

                self._log_result(f"    delta_gamma: {delta_gamma:.12e}")
                self._log_result(f"    delta_e_mev: {delta_e_mev:.12e} MeV")
                self._log_result(
                    f"    max_percent_energy_gain: {energy_gain_percent:.12e}%"
                )
                self._log_result(f"    percent_delta_e: {energy_gain_percent:.12e}%")
                self._log_result(f"    energy_gain_ppm: {energy_gain_ppm:.6f} ppm")

                if hasattr(self, "config") and hasattr(self.config, "mode"):
                    if self.config.mode == "optimization":
                        optimizer_value = -energy_gain_percent
                        self._log_result(
                            f"    optimizer_objective: {optimizer_value:.12e}"
                        )
            else:
                # Fallback: Try to calculate from trajectory if gamma values are missing
                self._log_result(
                    "  [WARNING] Gamma values missing, attempting trajectory fallback..."
                )
                if result.rider_trajectory is not None:
                    try:
                        traj = result.rider_trajectory
                        gamma_array = np.asarray(traj.get("gamma", []))
                        if len(gamma_array) > 0:
                            gamma_initial_fallback = float(gamma_array[0])
                            gamma_final_fallback = float(gamma_array[-1])
                            if gamma_initial_fallback > 0:
                                delta_gamma_fallback = (
                                    gamma_final_fallback - gamma_initial_fallback
                                )
                                energy_gain_percent = (
                                    delta_gamma_fallback
                                    / gamma_initial_fallback
                                    * 100.0
                                )
                                energy_gain_ppm = (
                                    delta_gamma_fallback / gamma_initial_fallback * 1e6
                                )
                                delta_e_mev_fallback = delta_gamma_fallback * (
                                    rider_m_particle * AMU_TO_MEV
                                )

                                metrics["max_percent_energy_gain"] = energy_gain_percent
                                metrics["percent_delta_e"] = energy_gain_percent
                                metrics["delta_gamma"] = delta_gamma_fallback
                                metrics["delta_e_mev"] = delta_e_mev_fallback
                                metrics["energy_gain_ppm"] = energy_gain_ppm

                                self._log_result(
                                    "  [OK] Fallback calculation successful:"
                                )
                                self._log_result(
                                    f"    gamma_initial (from traj): {gamma_initial_fallback:.12e}"
                                )
                                self._log_result(
                                    f"    gamma_final (from traj): {gamma_final_fallback:.12e}"
                                )
                                self._log_result(
                                    f"    delta_gamma: {delta_gamma_fallback:.12e}"
                                )
                                self._log_result(
                                    f"    delta_e_mev: {delta_e_mev_fallback:.12e} MeV"
                                )
                                self._log_result(
                                    f"    max_percent_energy_gain: {energy_gain_percent:.12e}%"
                                )
                                self._log_result(
                                    f"    percent_delta_e: {energy_gain_percent:.12e}%"
                                )
                                self._log_result(
                                    f"    energy_gain_ppm: {energy_gain_ppm:.6f} ppm"
                                )
                            else:
                                self._log_result(
                                    "  [ERROR] Fallback gamma_initial <= 0"
                                )
                        else:
                            self._log_result("  [ERROR] Trajectory gamma array is empty")
                    except Exception as e:
                        self._log_result(f"  [ERROR] Fallback calculation failed: {e}")
                else:
                    self._log_result(
                        "  [ERROR] No trajectory data available for fallback"
                    )

                if "max_percent_energy_gain" not in metrics:
                    self._log_result(
                        f"  [CRITICAL] max_percent_energy_gain could not be calculated for Run {run_num}"
                    )
                    self._log_result(
                        "  [CRITICAL] This will result in NaN/inf for optimization objective"
                    )

            # Add beam optics metrics if available
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

            # Add particle failure tracking
            metrics["num_particles_dead"] = result.num_particles_dead
            if result.halted_early:
                metrics["halted_early"] = True
                if result.halt_reason:
                    metrics["halt_reason"] = result.halt_reason

            output = {"metrics": metrics}

            self._log_result(
                f"  [DEBUG] Processing trajectory data for Run {run_num}..."
            )
            if result.rider_trajectory is not None:
                traj = result.rider_trajectory

                # Always include minimal trajectory info for distance calculation
                try:
                    z_array = np.asarray(traj["z"])
                    if len(z_array) > 0:
                        output["_distance_info"] = {
                            "z_start": float(z_array[0]),
                            "z_end": float(z_array[-1]),
                            "num_steps": len(z_array),
                        }
                except Exception as e:
                    print(f"[DEBUG] Failed to extract distance info: {e}")

                # Perform stability analysis if enabled
                if self.config.smoothness_enabled:
                    self._log_result(
                        f"  [DEBUG] Performing stability analysis for Run {run_num}..."
                    )

                    smoothness_config = SmoothnessConfig(
                        enabled=True,
                        window_size=self.config.smoothness_window_size,
                        oscillation_threshold=self.config.smoothness_oscillation_threshold,
                        trend_smoothness_threshold=self.config.smoothness_trend_threshold,
                        reject_on_violation=self.config.smoothness_reject_on_violation,
                        max_allowed_violations=self.config.smoothness_max_violations,
                    )

                    smoothness_result = analyze_trajectory_smoothness(
                        traj, smoothness_config, particle_mass_amu=rider_m_particle
                    )

                    output["stability_analysis"] = {
                        "passed": smoothness_result.passed,
                        "num_violations": len(smoothness_result.violations),
                        "oscillation_score": smoothness_result.oscillation_score,
                        "trend_smoothness_score": smoothness_result.trend_smoothness_score,
                        "quality": smoothness_result.quality_summary,
                    }

                    if not smoothness_result.passed:
                        self._log_result(
                            f"  [WARNING] Stability check FAILED for Run {run_num}"
                        )
                        self._log_result(
                            f"    Quality: {smoothness_result.quality_summary}"
                        )
                        if len(smoothness_result.violations) > 0:
                            self._log_result(
                                f"    Violations: {len(smoothness_result.violations)}"
                            )
                            for violation in smoothness_result.violations[:2]:
                                self._log_result(f"      - {violation.description}")

                        if self.config.smoothness_reject_on_violation:
                            self._log_result(
                                f"  [REJECT] Run {run_num} rejected due to numerical instability"
                            )
                            output["metrics"]["max_percent_energy_gain"] = np.nan
                            output["stability_rejected"] = True
                    else:
                        self._log_result(
                            f"  [OK] Stability check PASSED for Run {run_num}: {smoothness_result.quality_summary}"
                        )
                else:
                    self._log_result(
                        f"  [INFO] Stability analysis DISABLED for Run {run_num} (smoothness_enabled=False)"
                    )

                save_traj = (
                    self.config.save_all_trajectories
                    or self.config.save_failed_trajectories
                )
                if save_traj:
                    stride = self.config.trajectory_stride
                    try:
                        output["trajectory"] = {
                            "z": np.asarray(traj["z"])[::stride].tolist(),
                            "r": np.asarray(traj["r"])[::stride].tolist(),
                            "pz": np.asarray(traj["pz"])[::stride].tolist(),
                            "pr": np.asarray(traj["pr"])[::stride].tolist(),
                            "t": np.asarray(traj["t"])[::stride].tolist(),
                            "gamma": np.asarray(traj["gamma"])[::stride].tolist(),
                        }
                    except Exception as e:
                        self._log_result(
                            f"    [WARNING] Failed to save trajectory arrays: {e}"
                        )

                if result.halted_early:
                    output["halted_early"] = True
                    output["halt_reason"] = result.halt_reason
            else:
                self._log_result(
                    f"  [WARNING] No trajectory data available for Run {run_num}"
                )
                if self.config.smoothness_enabled:
                    self._log_result(
                        "  [WARNING] Stability analysis SKIPPED - no trajectory data returned from integration"
                    )
                    self._log_result(
                        "    Check that transverse_save=True in SimulationOptions"
                    )

            self._log_result(
                f"  [DEBUG] _run_single_integration returning for Run {run_num}"
            )

            return output
        finally:
            # Always clean up temporary run directory (success, halt, exception, cancel)
            try:
                import shutil

                if run_output_dir.exists():
                    shutil.rmtree(run_output_dir)
                    self._log_result(
                        f"  [DEBUG] Cleaned up temp directory: {run_output_dir.name}"
                    )
            except Exception as e:  # pragma: no cover - cleanup
                self._log_result(
                    f"  [WARNING] Failed to clean up temp directory {run_output_dir.name}: {e}"
                )

    def _cleanup_orphaned_temp_dirs(self):
        """Clean up any orphaned _temp_run directories from previous runs.

        This is called on plugin initialization to remove temp directories
        that weren't cleaned up due to crashes or interruptions.
        """
        import shutil
        from pathlib import Path

        try:
            output_dir = Path(self.sweep_output_dir)
            if not output_dir.exists():
                return

            # Find all _temp_run directories
            temp_dirs = list(output_dir.glob("_temp_run_*"))

            if temp_dirs:
                print(
                    f"[CLEANUP] Found {len(temp_dirs)} orphaned temp directories, removing..."
                )
                for temp_dir in temp_dirs:
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"[CLEANUP] Removed: {temp_dir.name}")
                    except Exception as e:
                        print(f"[WARNING] Failed to remove {temp_dir.name}: {e}")
        except Exception as e:
            print(f"[WARNING] Error during temp directory cleanup: {e}")
