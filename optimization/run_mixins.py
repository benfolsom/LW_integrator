"""Backend run logic mixin for OptimizationPlugin."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.constants import C_MMNS  # type: ignore[import]

# Physical constants for energy-momentum conversion
AMU_TO_MEV = 931.494  # Conversion factor amu to MeV
from core.debug_logger import set_logging_context  # type: ignore[import]
from core.smoothness_analyzer import (  # type: ignore[import]
    SmoothnessConfig,
    analyze_trajectory_smoothness,
    filter_stable_trajectories,
)
from core.types import SimulationType  # type: ignore[import]
from lw_integrator.testbed_runner import (  # type: ignore[import]
    RunResult,
    SimulationOptions,
    run_testbed,
)
from optimization.config import (  # type: ignore[import]
    calculate_auto_steps,
    calculate_auto_timestep,
)


def _calculate_starting_pz_from_energy(energy_gev: float, mass_amu: float) -> float:
    """Calculate starting Pz from total energy and mass.

    Parameters
    ----------
    energy_gev : float
        Total energy in GeV
    mass_amu : float
        Particle mass in amu

    Returns
    -------
    float
        Starting Pz in amu·mm/ns (negative, moving in -z direction)
    """
    rest_energy_mev = mass_amu * AMU_TO_MEV
    gamma = (energy_gev * 1e3) / rest_energy_mev
    if gamma < 1.0:
        gamma = 1.0
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.0
    return gamma * mass_amu * C_MMNS * beta


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

            # Debug logging for aperture decision
            self._log_result("[DEBUG] Optimization parameter setup:")
            self._log_result(f"  simulation_type = {self.config.simulation_type}")
            self._log_result(f"  aperture_points = {self.config.aperture_points}")
            self._log_result(
                f"  Is BUNCH_TO_BUNCH? {self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH}"
            )

            # Aperture (not used in BUNCH_TO_BUNCH mode)
            if (
                self.config.aperture_points > 1
                and self.config.simulation_type != SimulationType.BUNCH_TO_BUNCH
            ):
                param_names.append("aperture_radius")
                param_bounds.append(self.config.aperture_range)
                self._log_result(f"  → Aperture INCLUDED in optimization")
                self._log_result(
                    f"    Added: aperture_radius, range={self.config.aperture_range}"
                )
            else:
                self._log_result(f"  → Aperture EXCLUDED from optimization")

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

            # Driver starting Pz - if enabled as sweep parameter (BUNCH_TO_BUNCH only)
            if (
                self.config.driver_starting_Pz_range is not None
                and self.config.driver_starting_Pz_points > 1
            ):
                param_names.append("driver_starting_Pz")
                param_bounds.append(self.config.driver_starting_Pz_range)

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
                    # For BUNCH_TO_BUNCH, aperture is not used (set dummy value)
                    aperture = (
                        1.0e-4
                        if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH
                        else self.config.aperture_range[0]
                    )
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
                    driver_m_particle = 207.2  # default
                    driver_charge_sign = 1.0  # default
                    driver_pcount = 5  # default
                    driver_transv_mom = 0.0  # default
                    driver_transv_dist = -0.07998  # default
                    driver_starting_distance = 1000.0  # default
                    driver_starting_Pz = -4925.0  # default

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
                            # Convert energy to Pz for internal use
                            driver_starting_Pz = _calculate_starting_pz_from_energy(
                                driver_energy_gev, driver_m_particle
                            )
                        elif param_name == "driver_starting_Pz":
                            # Legacy support if old configs still use Pz
                            driver_starting_Pz = x[i]

                    # Calculate transverse offset in mm from fraction
                    transv_offset = offset_frac * aperture

                    # Build driver_params if BUNCH_TO_BUNCH mode
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
                        }

                    # Calculate timestep if using auto_distance strategy
                    if self.config.timestep_strategy == "auto_distance":
                        timestep = self.config.calculate_timestep_for_energy(
                            energy,
                            self.config.m_particle,
                            wall_z=wall_z,
                            start_z=start_z,
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
                            except Exception as e:  # pragma: no cover - passthrough
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

                    # Hard rejection for unphysical energy gains > 100%
                    if "max_percent_energy_gain" in metrics:
                        pct_gain = metrics["max_percent_energy_gain"]
                        if pct_gain > 100.0:
                            self._log_result(
                                f"[REJECT] Evaluation {eval_num} DISQUALIFIED: "
                                f"energy gain {pct_gain:.1f}% > 100% (unphysical)"
                            )
                            # Store rejected evaluation
                            eval_record = {
                                "evaluation": eval_num,
                                "parameters": dict(zip(param_names, x)),
                                "failed": True,
                                "halted_early": False,
                                "reject_reason": f"Unphysical energy gain: {pct_gain:.1f}% > 100%",
                                "objective_value": float("inf"),
                                "metrics": result.get("metrics", {}),
                            }
                            all_evaluations.append(eval_record)
                            return np.inf

                        # Penalize negative energy gains (deceleration)
                        # Make them worse than minimal positive gains but better than failures
                        if pct_gain < 0.0:
                            # Map negative gains to large positive penalty values
                            # Worse deceleration → larger penalty, but still finite (unlike blowups)
                            # Scale: -0.001% → penalty ~1.0, -1.0% → penalty ~1000, -10% → penalty ~10000
                            penalty_magnitude = (
                                abs(pct_gain) * 1000.0
                            )  # Scale to reasonable range

                            self._log_result(
                                f"[PENALTY] Evaluation {eval_num}: Negative energy gain "
                                f"({pct_gain:.6f}%) → penalty {penalty_magnitude:.3e}"
                            )

                            # Store evaluation with large penalty
                            eval_record = {
                                "evaluation": eval_num,
                                "parameters": dict(zip(param_names, x)),
                                "failed": False,
                                "halted_early": False,
                                "negative_gain": True,
                                "objective_value": penalty_magnitude,  # Large positive = bad for minimization
                                "raw_objective_value": pct_gain,
                                "metrics": result.get("metrics", {}),
                            }
                            all_evaluations.append(eval_record)

                            # Return penalty that's worse than any positive gain but better than inf
                            # For maximization, we'll return large positive value (gets negated later)
                            return penalty_magnitude

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

                    # Add stability-based penalty using continuous sliding scale
                    stability_penalty = 0.0
                    if "smoothness_metrics" in result:
                        smoothness = result["smoothness_metrics"]
                        quality = smoothness.get("quality_summary", "")

                        # Get quantitative metrics for sliding scale
                        max_oscillation = smoothness.get("oscillation_score", 0.0)
                        max_trend_residual = smoothness.get(
                            "trend_smoothness_score", 0.0
                        )

                        # Compute penalty factor based on continuous scale (0.0 = no penalty, 1.0 = full penalty)
                        # Oscillation contribution (0.7+ is severe, 0.3-0.7 is concerning, <0.3 is acceptable)
                        osc_factor = 0.0
                        if max_oscillation > 0.7:
                            osc_factor = 1.0  # Severe
                        elif max_oscillation > 0.3:
                            # Linear scale from 0.3 (0.0 penalty) to 0.7 (1.0 penalty)
                            osc_factor = (max_oscillation - 0.3) / 0.4

                        # Trend residual contribution (0.5+ is highly erratic, 0.2-0.5 is concerning, <0.2 is acceptable)
                        trend_factor = 0.0
                        if max_trend_residual > 0.5:
                            trend_factor = 1.0  # Highly erratic
                        elif max_trend_residual > 0.2:
                            # Linear scale from 0.2 (0.0 penalty) to 0.5 (1.0 penalty)
                            trend_factor = (max_trend_residual - 0.2) / 0.3

                        # Combined penalty factor (take worst of the two)
                        penalty_factor = max(osc_factor, trend_factor)

                        # Apply penalty with scaling: 0% penalty at factor=0, 99% penalty at factor=1
                        if penalty_factor > 0.0:
                            # Exponential scaling for more aggressive penalties at higher factors
                            # penalty_factor^2 gives: 0.25 → 6.25%, 0.5 → 25%, 0.75 → 56%, 1.0 → 99%
                            scaled_penalty = penalty_factor**2
                            stability_penalty = value * min(0.99, scaled_penalty)

                            # Log penalty with details
                            if penalty_factor > 0.8:
                                self._log_result(
                                    f"[WARNING] Heavy stability penalty: {stability_penalty:.3e} "
                                    f"(osc={max_oscillation:.3f}, trend={max_trend_residual:.3f}, "
                                    f"factor={penalty_factor:.3f}, quality: {quality})"
                                )
                            elif penalty_factor > 0.4:
                                self._log_result(
                                    f"[INFO] Moderate stability penalty: {stability_penalty:.3e} "
                                    f"(osc={max_oscillation:.3f}, trend={max_trend_residual:.3f}, "
                                    f"factor={penalty_factor:.3f})"
                                )
                            elif penalty_factor > 0.1:
                                self._log_result(
                                    f"[INFO] Light stability penalty: {stability_penalty:.3e} "
                                    f"(osc={max_oscillation:.3f}, trend={max_trend_residual:.3f})"
                                )
                        # penalty_factor = 0 → no penalty (Good quality)

                    # Also add penalty for unphysically high energy gains
                    energy_gain_penalty = 0.0
                    if "max_percent_energy_gain" in metrics:
                        pct_gain = metrics["max_percent_energy_gain"]
                        if pct_gain > 500.0:  # >500% likely unphysical
                            # Scale penalty with how extreme the gain is
                            excess = (pct_gain - 500.0) / 500.0
                            energy_gain_penalty = (
                                value * min(0.95, excess * 0.5)
                                if maximize
                                else value * min(0.95, excess * 0.5)
                            )
                            self._log_result(
                                f"[WARNING] Unphysical energy gain penalty: {energy_gain_penalty:.3e} "
                                f"(gain: {pct_gain:.1f}% > 500% threshold)"
                            )

                    # Add penalty for particle deaths (scales by fraction lost)
                    particle_death_penalty = 0.0
                    if "num_particles_dead" in metrics:
                        num_dead = metrics["num_particles_dead"]
                        if num_dead > 0:
                            # Get total particle count from config
                            total_particles = int(self.config.pcount)
                            if total_particles > 0:
                                # Penalty scales 1:1 with fraction lost: 10% lost → 10% penalty
                                # particle_death_penalty_fraction is a multiplier (default 1.0)
                                # Examples with default 1.0:
                                #   10% particles lost → 10% penalty
                                #   50% particles lost → 50% penalty
                                #   100% particles lost → 100% penalty
                                # Set to 0.5 for gentler: 10% lost → 5% penalty
                                # Set to 2.0 for stricter: 10% lost → 20% penalty
                                penalty_multiplier = getattr(
                                    self.config, "particle_death_penalty_fraction", 1.0
                                )
                                fraction_lost = num_dead / total_particles
                                particle_death_penalty = (
                                    value * fraction_lost * penalty_multiplier
                                )
                                self._log_result(
                                    f"[INFO] Particle death penalty: {particle_death_penalty:.3e} "
                                    f"({num_dead}/{total_particles} = {fraction_lost * 100:.1f}% lost, "
                                    f"penalty = {fraction_lost * penalty_multiplier * 100:.1f}% of objective)"
                                )

                    total_penalty = (
                        penalty
                        + stability_penalty
                        + energy_gain_penalty
                        + particle_death_penalty
                    )
                    adjusted_value = value
                    if total_penalty > 0:
                        if maximize:
                            adjusted_value = value - total_penalty
                        else:
                            adjusted_value = value + total_penalty
                        if penalty > 0:
                            self._log_result(
                                "[INFO] Applied parameter soft penalty of "
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
                        "stability_penalty": stability_penalty,
                        "energy_gain_penalty": energy_gain_penalty,
                        "particle_death_penalty": particle_death_penalty,
                        "total_penalty": total_penalty,
                        "fitness": result_value,  # Store fitness (for minimization)
                        "failed": False,
                        "halted_early": False,
                        "metrics": result.get("metrics", {}),
                    }

                    # Store stability quality if available
                    if "smoothness_metrics" in result:
                        eval_record["stability_quality"] = result[
                            "smoothness_metrics"
                        ].get("quality_summary", "Unknown")

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

            if method == "genetic_algorithm":

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

            elif method == "nelder_mead":
                result = optimize_parameters(
                    config_template=config_template,
                    parameter_names=param_names,
                    parameter_bounds=param_bounds,
                    metric_name=metric_name,
                    method="nelder_mead",
                    maximize=maximize,
                    maxiter=self.config.optimization_maxiter,
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
        """Run parameter sweep in background with real integration."""
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

                # Aperture is not present in BUNCH_TO_BUNCH mode
                aperture = params_dict.get(
                    "aperture", 1.0e-4
                )  # dummy value for BUNCH_TO_BUNCH
                energy = params_dict["energy"]
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
                    self._log_result(
                        f"    rider_stripped_ions: {rider_stripped_ions:.2f}"
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
                    driver_m = params_dict.get("driver_m_particle", 207.2)

                    # Convert driver_energy_gev to starting_Pz if present,
                    # otherwise fall back to legacy driver_starting_Pz key
                    if "driver_energy_gev" in params_dict:
                        driver_pz = _calculate_starting_pz_from_energy(
                            params_dict["driver_energy_gev"], driver_m
                        )
                    else:
                        driver_pz = params_dict.get("driver_starting_Pz", -4925.0)

                    driver_params_dict = {
                        "m_particle": driver_m,
                        "charge_sign": params_dict.get("driver_charge_sign", 1.0),
                        "pcount": int(params_dict.get("driver_pcount", 5)),
                        "transv_mom": params_dict.get("driver_transv_mom", 0.0),
                        "transv_dist": params_dict.get("driver_transv_dist", -0.07998),
                        "starting_distance": params_dict.get(
                            "driver_starting_distance", 1000.0
                        ),
                        "starting_Pz": driver_pz,
                        "stripped_ions": params_dict.get(
                            "driver_stripped_ions", self.config.driver_stripped_ions
                        ),
                    }

                    # Log driver parameters if BUNCH_TO_BUNCH and full debug
                    if use_full_debug:
                        self._log_result(
                            f"    driver_m_particle: {driver_params_dict['m_particle']:.4e} amu"
                        )
                        self._log_result(
                            f"    driver_charge_sign: {driver_params_dict['charge_sign']:.1f}"
                        )
                        self._log_result(
                            f"    driver_pcount: {driver_params_dict['pcount']}"
                        )
                        self._log_result(
                            f"    driver_transv_mom: {driver_params_dict['transv_mom']:.4e} amu·mm/ns"
                        )
                        self._log_result(
                            f"    driver_transv_dist: {driver_params_dict['transv_dist']:.4e} mm"
                        )
                        self._log_result(
                            f"    driver_starting_distance: {driver_params_dict['starting_distance']:.4f} mm"
                        )
                        self._log_result(
                            f"    driver_starting_Pz: {driver_params_dict['starting_Pz']:.4e} amu·mm/ns"
                        )
                        self._log_result(
                            f"    driver_stripped_ions: {driver_params_dict['stripped_ions']:.2f}"
                        )

                # Calculate transverse offset
                transv_offset = offset_frac * aperture

                # Calculate timestep based on strategy
                if self.config.timestep_strategy != "fixed":
                    # Use energy-aware timestep calculation
                    # Get wall_z for this run (it may be swept)
                    wall_z_for_calc = params_dict.get("wall_z", self.config.wall_z)
                    timestep = self.config.calculate_timestep_for_energy(
                        energy,
                        rider_m_particle,
                        wall_z=wall_z_for_calc,
                        start_z=start_z,
                    )
                    steps = self.config.steps

                    # Calculate gamma for diagnostics (ALWAYS log for debugging)
                    AMU_TO_MEV = 931.494
                    rest_energy_mev = rider_m_particle * AMU_TO_MEV
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

                # Loop for retry attempts
                while retry_attempt <= max_retries:
                    # Generate new seed for retries (original run uses config seed)
                    if retry_attempt > 0:
                        # Use a deterministic but different seed based on run number and retry attempt
                        current_seed = (
                            self.config.seed + run_num * 10000 + retry_attempt * 100
                        )
                        if use_full_debug or use_truncated_logging:
                            self._log_result(
                                f"  [RETRY] Run {run_num}, attempt {retry_attempt}/{max_retries} with new seed {current_seed}"
                            )
                    else:
                        current_seed = self.config.seed

                    # Reset error/timeout flags for this attempt
                    attempt_result = None
                    attempt_error = None
                    attempt_timed_out = False

                    try:
                        # Check if timeout is enabled
                        if self.config.per_run_timeout > 0:
                            # Container for result (mutable for thread access)
                            result_container: List[Optional[RunResult]] = [None]
                            error_container: List[Optional[Exception]] = [None]
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
                                except Exception as e:  # pragma: no cover - passthrough
                                    error_container[0] = e

                            thread = threading.Thread(target=run_integration)
                            thread.daemon = True
                            thread.start()

                            # Wait for completion or timeout
                            thread.join(timeout=self.config.per_run_timeout)

                            if thread.is_alive():
                                attempt_timed_out = True
                                cancel_flag[0] = True
                                self._log_result(
                                    f"  [TIMEOUT] Run {run_num}: exceeded {self.config.per_run_timeout}s, requesting cancel..."
                                )
                                # Give integration a brief moment to stop
                                thread.join(timeout=2.0)

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
                                seed_override=current_seed,
                            )

                    except Exception as e:  # pragma: no cover - integration path
                        attempt_error = e

                    # Check if this attempt succeeded
                    if (
                        not attempt_timed_out
                        and attempt_error is None
                        and attempt_result is not None
                    ):
                        # Check if result has valid metrics (not all particles dead)
                        # A run is considered failed if:
                        # 1. It was halted early (all particles died)
                        # 2. Metrics dict is empty or missing key metrics

                        is_halted = attempt_result.get("halted_early", False)
                        metrics = attempt_result.get("metrics", {})

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
                            elif metrics.get("delta_e_mev") is not None:
                                has_useful_metrics = True

                        if has_useful_metrics:
                            # Success! Use this result
                            result = attempt_result
                            run_error = None
                            run_timed_out = False
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
                                    f"  [FAILED] Run {run_num} attempt {retry_attempt}: halted={is_halted}, no useful metrics"
                                )

                    # This attempt failed - decide whether to retry
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
                if run_timed_out:
                    # Treat timeout as failed run
                    failed_runs.append(
                        {
                            "run_number": run_num,
                            "parameters": params_dict,
                            "error": f"Timeout after {self.config.per_run_timeout}s (tried {retry_attempt + 1} time(s))",
                            "timed_out": True,
                            "retry_attempts": retry_attempt,
                        }
                    )
                    self._log_result(
                        f"  [TIMEOUT] Run {run_num}: Timeout after {self.config.per_run_timeout}s"
                    )
                elif run_error is not None:
                    failed_runs.append(
                        {
                            "run_number": run_num,
                            "parameters": params_dict,
                            "error": str(run_error),
                            "timed_out": False,
                            "retry_attempts": retry_attempt,
                        }
                    )
                    self._log_result(
                        f"  [ERROR] Run {run_num}: Error during integration: {run_error}"
                    )
                elif result is None:
                    failed_runs.append(
                        {
                            "run_number": run_num,
                            "parameters": params_dict,
                            "error": "Integration returned no result",
                            "timed_out": False,
                            "retry_attempts": retry_attempt,
                        }
                    )
                    self._log_result(
                        f"  [ERROR] Run {run_num}: No result returned from integration"
                    )
                else:
                    # Calculate metrics and store results
                    metrics = result.metrics

                    # Add run metadata
                    result_data = {
                        "run_number": run_num,
                        "parameters": {
                            "particle_energy_gev": energy,
                            "starting_z_mm": start_z,
                            "transverse_offset_fraction": offset_frac,
                            "timestep": timestep,
                            "steps": steps,
                            "retry_attempts": retry_attempt,
                        },
                        "metrics": metrics,
                    }

                    # Add aperture_radius only for non-BUNCH_TO_BUNCH modes
                    if self.config.simulation_type != SimulationType.BUNCH_TO_BUNCH:
                        result_data["parameters"]["aperture_radius"] = aperture

                    # Include additional swept parameters if present
                    if "wall_z" in params_dict:
                        result_data["parameters"]["wall_z"] = params_dict["wall_z"]

                    # Add rider particle parameters (always include, may be swept)
                    if "rider_m_particle" in params_dict:
                        result_data["parameters"]["rider_m_particle"] = rider_m_particle
                    if "rider_charge_sign" in params_dict:
                        result_data["parameters"]["rider_charge_sign"] = (
                            rider_charge_sign
                        )
                    if "rider_pcount" in params_dict:
                        result_data["parameters"]["rider_pcount"] = rider_pcount
                    if "rider_transv_mom" in params_dict:
                        result_data["parameters"]["rider_transv_mom"] = rider_transv_mom
                    if "rider_transv_dist" in params_dict:
                        result_data["parameters"]["rider_transv_dist"] = (
                            rider_transv_dist
                        )
                    if "rider_stripped_ions" in params_dict:
                        result_data["parameters"]["rider_stripped_ions"] = (
                            rider_stripped_ions
                        )

                    # Add driver particle parameters if BUNCH_TO_BUNCH (always include, may be swept)
                    if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
                        if driver_params_dict is not None:
                            if "driver_m_particle" in params_dict:
                                result_data["parameters"]["driver_m_particle"] = (
                                    driver_params_dict["m_particle"]
                                )
                            if "driver_charge_sign" in params_dict:
                                result_data["parameters"]["driver_charge_sign"] = (
                                    driver_params_dict["charge_sign"]
                                )
                            if "driver_pcount" in params_dict:
                                result_data["parameters"]["driver_pcount"] = (
                                    driver_params_dict["pcount"]
                                )
                            if "driver_transv_mom" in params_dict:
                                result_data["parameters"]["driver_transv_mom"] = (
                                    driver_params_dict["transv_mom"]
                                )
                            if "driver_transv_dist" in params_dict:
                                result_data["parameters"]["driver_transv_dist"] = (
                                    driver_params_dict["transv_dist"]
                                )
                            if "driver_starting_distance" in params_dict:
                                result_data["parameters"][
                                    "driver_starting_distance"
                                ] = driver_params_dict["starting_distance"]
                            if "driver_starting_Pz" in params_dict:
                                result_data["parameters"]["driver_starting_Pz"] = (
                                    driver_params_dict["starting_Pz"]
                                )
                            if "driver_stripped_ions" in params_dict:
                                result_data["parameters"]["driver_stripped_ions"] = (
                                    driver_params_dict["stripped_ions"]
                                )

                    if self.config.macroparticle_enabled:
                        result_data["parameters"]["macroparticle_charge_multiplier"] = (
                            macroparticle_charge_multiplier
                        )
                        result_data["parameters"]["macroparticle_sigma_multiplier"] = (
                            macroparticle_sigma_multiplier
                        )

                    # Stability analysis only if trajectory is present
                    if result.trajectory is not None:
                        traj = result.trajectory

                        # Compute smoothness metrics
                        smoothness_config = SmoothnessConfig(
                            enabled=self.config.smoothness_enabled,
                            window_size=int(self.config.smoothness_window_size),
                            reject_on_violation=self.config.smoothness_reject_on_violation,
                            gamma_threshold=self.config.smoothness_gamma_threshold,
                            radius_threshold=self.config.smoothness_radius_threshold,
                        )

                        smoothness_result = analyze_trajectory_smoothness(
                            trajectory=traj,
                            smoothness_config=smoothness_config,
                            enable_logging=True,
                        )

                        # Filter stable trajectories if enabled
                        if self.config.stability_filter_enabled:
                            stable = filter_stable_trajectories(
                                trajectories=[traj],
                                smoothness_config=smoothness_config,
                                enable_logging=True,
                            )
                            result_data["is_stable"] = len(stable) > 0
                        else:
                            result_data["is_stable"] = True

                        # Store smoothness metrics
                        result_data["smoothness_metrics"] = smoothness_result.to_dict()

                        # Store trajectory with stride (for saving)
                        # Only save trajectory arrays if trajectory saving is enabled or stability enabled
                        save_traj = (
                            self.config.save_all_trajectories
                            or self.config.save_failed_trajectories
                            or self.config.smoothness_enabled
                        )
                        if save_traj:
                            stride = self.config.trajectory_stride
                            try:
                                result_data["trajectory"] = {
                                    "z": np.asarray(traj["z"])[::stride].tolist(),
                                    "r": np.asarray(traj["r"])[::stride].tolist(),
                                    "pz": np.asarray(traj["pz"])[::stride].tolist(),
                                    "pr": np.asarray(traj["pr"])[::stride].tolist(),
                                    "t": np.asarray(traj["t"])[::stride].tolist(),
                                    "gamma": np.asarray(traj["gamma"])[
                                        ::stride
                                    ].tolist(),
                                }
                            except Exception as e:
                                self._log_result(
                                    f"    [WARNING] Failed to downsample trajectory: {e}"
                                )
                    else:
                        result_data["is_stable"] = True  # No trajectory to check

                    all_results.append(result_data)

                    # Log run completion summary (Truncated mode logs here)
                    if use_truncated_logging:
                        self._log_result(
                            f"  [RESULT] Run {run_num}/{total_runs}: ΔE={metrics.get('rider_delta_e_mev', 0):.3f} MeV, "
                            f"z_final={metrics.get('rider_z_final', 0):.3f} mm"
                        )

            # Save results to JSON file
            self._save_sweep_results(all_results, failed_runs)

            # Close log file before moving directory
            if self._log_file is not None:
                self._close_log_file()

            # Move log file to sweep directory
            if self._log_file_path is not None and self._log_file_path.exists():
                import shutil

                log_dest = Path(self.config.output_dir) / self._log_file_path.name
                try:
                    shutil.copy2(self._log_file_path, log_dest)
                    self._log_result(f"Log file saved to: {log_dest}")
                except Exception as e:
                    self._log_result(
                        f"[WARNING] Failed to move log file to sweep directory: {e}"
                    )

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
        except Exception as e:  # pragma: no cover - integration path
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
        grids = {}

        # Aperture: only for non-BUNCH_TO_BUNCH modes
        if self.config.simulation_type != SimulationType.BUNCH_TO_BUNCH:
            grids["aperture"] = self._generate_range(
                self.config.aperture_range[0],
                self.config.aperture_range[1],
                self.config.aperture_points,
                self.config.aperture_log_scale,
            )

        # Energy: always swept
        grids["energy"] = self._generate_range(
            self.config.energy_range[0],
            self.config.energy_range[1],
            self.config.energy_points,
            self.config.energy_log_scale,
        )

        # Transverse offset is ALWAYS a single (x,y) configuration, NOT a sweep parameter
        # For both CONDUCTING_WALL and BUNCH_TO_BUNCH:
        # - If config has [0.0, 0.0], this represents a single (x=0, y=0) configuration
        # - If config has [0.1], this represents (x=0.1, y=0) configuration
        # - We use only the first value as the scalar offset for this sweep configuration
        if len(self.config.transverse_offset_fractions) > 0:
            grids["transverse_offset_fraction"] = [
                self.config.transverse_offset_fractions[0]
            ]
        else:
            # No offset provided, default to 0.0
            grids["transverse_offset_fraction"] = [0.0]

        # Starting z positions: always swept if multiple values
        grids["start_z"] = self.config.starting_z_positions

        # Wall z (optional sweep)
        if self.config.wall_z_range is not None and self.config.wall_z_points > 1:
            grids["wall_z"] = self._generate_range(
                self.config.wall_z_range[0],
                self.config.wall_z_range[1],
                self.config.wall_z_points,
                False,  # wall_z doesn't need log scale
            )

        # Optional sweeps for rider and driver particle parameters
        sim_type = self.config.simulation_type
        for param_name, controls in self.sweep_params.items():
            # Skip driver params if not BUNCH_TO_BUNCH
            if (
                param_name.startswith("driver_")
                and sim_type != SimulationType.BUNCH_TO_BUNCH
            ):
                continue

            if controls["sweep_var"].get():
                min_val = float(controls["min_var"].get())
                max_val = float(controls["max_var"].get())
                points = int(controls["points_var"].get())
                log_scale = controls["log_var"].get()
                grids[param_name] = self._generate_range(
                    min_val, max_val, points, log_scale
                )

        return grids

    def _generate_range(
        self, min_val: float, max_val: float, points: int, log_scale: bool
    ) -> List[float]:
        """Generate parameter range (linear or log scale)."""
        if points == 1:
            return [(min_val + max_val) / 2.0]
        if log_scale:
            return np.logspace(np.log10(min_val), np.log10(max_val), points).tolist()
        return np.linspace(min_val, max_val, points).tolist()

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
        rider_stripped_ions = (
            rider_stripped_ions
            if rider_stripped_ions is not None
            else self.config.stripped_ions
        )

        # Build rider params
        rider_params = {
            "starting_distance": start_z,
            "transv_mom": rider_transv_mom,
            "transv_dist": rider_transv_dist,
            "transv_offset_x": transv_offset,
            "transv_offset_y": 0.0,
            "m_particle": rider_m_particle,
            "charge_sign": rider_charge_sign,
            "pcount": rider_pcount,
            "stripped_ions": rider_stripped_ions,
            "starting_Pz": 0.0,
        }

        # Calculate initial Pz from energy
        AMU_TO_MEV = 931.494
        rest_energy_mev = rider_m_particle * AMU_TO_MEV
        gamma = (energy_gev * 1e3) / rest_energy_mev
        rider_params["starting_Pz"] = C_MMNS * np.sqrt(gamma * gamma - 1.0)

        core_params = {
            "time_step": timestep,
            "wall_z": wall_z,
            "aperture_radius": aperture,
            "mean": 1.0e5,
            "cav_spacing": 1.0e5,
            "z_cutoff": (
                self.config.target_distance_mm
                if self.config.z_cutoff_mode == "relative"
                else 0.0
            ),
            "z_cutoff_mode": self.config.z_cutoff_mode,
        }

        # Create a temporary subdirectory for this run's outputs (will be cleaned up)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
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
            driver_params=driver_params,
            core_params=core_params,
            legacy_enabled=False,
            trajectory_save=False,
            trajectory_interval=self.config.trajectory_stride,
            energy_display=False,
            energy_save=False,
            transverse_display=False,
            transverse_save=True,
            beta_display=False,
            beta_save=False,
            momentum_display=False,
            momentum_save=False,
            gamma_display=False,
            gamma_save=False,
            zposition_display=False,
            zposition_save=False,
            macroparticle_enabled=self.config.macroparticle_enabled,
            macroparticle_charge_multiplier=macroparticle_charge_multiplier,
            macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
            macroparticle_use_momentum_errors=self.config.macroparticle_use_momentum_errors,
            overlay_display=False,
            overlay_save=False,
            difference_display=False,
            difference_save=False,
            use_adaptive_timestep=self.config.use_adaptive_timestep,
            adaptive_timestep_min=self.config.adaptive_timestep_min,
            adaptive_timestep_max=self.config.adaptive_timestep_max,
            adaptive_timestep_target=self.config.adaptive_timestep_target,
            adaptive_timestep_debug=self.config.adaptive_timestep_debug,
            per_run_timeout=self.config.per_run_timeout,
            cancel_flag=cancel_flag,
        )

        try:
            result: RunResult = run_testbed(options=options, output_dir=run_output_dir)
            self._log_result(f"  [DEBUG] Integration complete for Run {run_num}")

            output = {
                "metrics": result.metrics,
                "parameters": {
                    "aperture_radius": aperture,
                    "particle_energy_gev": energy_gev,
                    "starting_z_mm": start_z,
                    "transverse_offset_fraction": (
                        transv_offset / aperture if aperture != 0 else 0
                    ),
                    "timestep": timestep,
                    "steps": steps,
                },
            }

            # Attach trajectory if available
            if result.trajectory is not None:
                traj = result.trajectory
                output["trajectory"] = traj

                # Stability analysis (smoothness)
                if self.config.smoothness_enabled:
                    smoothness_config = SmoothnessConfig(
                        enabled=self.config.smoothness_enabled,
                        window_size=int(self.config.smoothness_window_size),
                        reject_on_violation=self.config.smoothness_reject_on_violation,
                        gamma_threshold=self.config.smoothness_gamma_threshold,
                        radius_threshold=self.config.smoothness_radius_threshold,
                    )

                    smoothness_result = analyze_trajectory_smoothness(
                        trajectory=traj,
                        smoothness_config=smoothness_config,
                        enable_logging=True,
                    )

                    # Filter stable trajectories if enabled
                    if self.config.stability_filter_enabled:
                        stable = filter_stable_trajectories(
                            trajectories=[traj],
                            smoothness_config=smoothness_config,
                            enable_logging=True,
                        )
                        output["is_stable"] = len(stable) > 0
                    else:
                        output["is_stable"] = True

                    # Store smoothness metrics
                    output["smoothness_metrics"] = smoothness_result.to_dict()

                # Only save full trajectory arrays if explicitly requested
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
                        f"  [WARNING] Stability analysis SKIPPED - no trajectory data returned from integration"
                    )
                    self._log_result(
                        f"    Check that transverse_save=True in SimulationOptions"
                    )

            self._log_result(
                f"  [DEBUG] _run_single_integration returning for Run {run_num}"
            )

            return output
        finally:
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
        """Clean up any orphaned _temp_run directories from previous runs."""
        import shutil

        try:
            output_dir = Path(self.sweep_output_dir)
            if not output_dir.exists():
                return

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
