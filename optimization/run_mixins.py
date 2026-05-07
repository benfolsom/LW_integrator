"""Backend run logic mixin for OptimizationPlugin."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.debug_logger import initialize_debug_logging  # type: ignore[import]
from lw_integrator.testbed_runner import (  # type: ignore[import]
    SimulationOptions,
    run_testbed,
)
from optimization.logging_policy import (
    apply_run_logging_policy,
    describe_run_logging_policy,
    restore_run_logging_policy,
)
from optimization.penalties import compute_soft_penalty
from optimization.run_parameter_helpers import (
    build_optimization_evaluation_outcome,
    collect_optimization_parameter_selection,
    resolve_optimization_run_parameters,
    resolve_objective_metric,
)
from optimization.run_logging_helpers import (
    build_progress_log_line,
    build_small_aperture_diagnostic_line,
    build_stability_config_log_lines,
    should_emit_verbose_run_log,
)
from optimization.simulation_type_helpers import is_bunch_to_bunch
from optimization.single_integration_helpers import (
    build_integration_metrics,
    build_final_z_check_log_lines,
    build_halted_integration_output,
    build_integration_trajectory_output,
    build_single_integration_setup,
)
from optimization.sweep_helpers import build_parameter_grids
from optimization.sweep_run_helpers import (
    build_full_debug_parameter_log_lines,
    resolve_sweep_run_parameters,
    resolve_sweep_timestep,
)
from optimization.sweep_result_helpers import (
    build_failed_sweep_run_record,
    build_full_debug_sweep_result_log_lines,
    build_sweep_completion_log_lines,
    build_sweep_run_data,
    build_timeout_sweep_run_record,
    build_truncated_sweep_log_params,
    classify_sweep_attempt_result,
    extract_actual_distance,
    extract_sweep_metric_summary,
)


class OptimizationRunMixin:
    """Encapsulates run queue, threading, and integration helpers."""

    def _run_optimization_background(self):
        """Run optimization in background using selected algorithm."""
        # Set logging context for this optimization run
        method = self.config.optimization_method
        initialize_debug_logging(
            context=f"optimization_{method}",
            force_new_log=True,
        )

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

            logging_policy = apply_run_logging_policy(self.config)
            for line in describe_run_logging_policy(logging_policy):
                self._log_result(line)
            self._log_result("")

            selection = collect_optimization_parameter_selection(self.config)
            param_names = selection.names
            param_bounds = selection.bounds
            for line in selection.log_lines:
                self._log_result(line)

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

            metric_name, maximize = resolve_objective_metric(self.config.objective)

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
                    run_params = resolve_optimization_run_parameters(
                        self.config, param_names, x
                    )
                    aperture = run_params.aperture
                    energy = run_params.energy_gev
                    macroparticle_charge_mult = (
                        run_params.macroparticle_charge_multiplier
                    )

                    result, timed_out = self._run_optimization_evaluation_integration(
                        run_params, eval_num, x
                    )
                    if timed_out:
                        return np.inf

                    penalty = compute_soft_penalty(
                        self.config,
                        aperture_radius=aperture,
                        macroparticle_charge_multiplier=macroparticle_charge_mult,
                        initial_energy_gev=energy,
                    )
                    outcome = build_optimization_evaluation_outcome(
                        result,
                        eval_num=eval_num,
                        param_names=param_names,
                        values=x,
                        metric_name=metric_name,
                        maximize=maximize,
                        penalty=penalty,
                        objective_name=self.config.objective,
                        save_trajectory=self.config.save_all_trajectories,
                    )
                    for line in outcome.log_lines:
                        self._log_result(line)
                    all_evaluations.append(outcome.record)
                    return outcome.fitness

                except Exception as e:
                    import traceback

                    self._log_result(
                        f"[ERROR] Evaluation {eval_num} failed for params {x}"
                    )
                    self._log_result(f"[ERROR] Exception: {type(e).__name__}: {e}")
                    self._log_result("[ERROR] Traceback:")
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
            if "logging_policy" in locals():
                restore_run_logging_policy(self.config, logging_policy)

            self.running = False
            self._update_progress(100, "Done")
            # Ensure log file is closed
            if self._log_file is not None:
                self._close_log_file()

    def _run_optimization_evaluation_integration(
        self, run_params, eval_num: int, original_params
    ):
        """Run one optimizer evaluation, optionally with a per-run timeout."""
        integration_kwargs = {
            "aperture": run_params.aperture,
            "energy_gev": run_params.energy_gev,
            "start_z": run_params.start_z,
            "transv_offset": run_params.transv_offset,
            "timestep": run_params.timestep,
            "steps": run_params.steps,
            "rider_m_particle": run_params.rider_m_particle,
            "rider_charge_sign": run_params.rider_charge_sign,
            "rider_pcount": int(run_params.rider_pcount),
            "rider_transv_mom": run_params.rider_transv_mom,
            "rider_transv_dist": run_params.rider_transv_dist,
            "rider_stripped_ions": run_params.rider_stripped_ions,
            "macroparticle_charge_multiplier": (
                run_params.macroparticle_charge_multiplier
            ),
            "macroparticle_sigma_multiplier": run_params.macroparticle_sigma_multiplier,
            "driver_params": run_params.driver_params,
            "wall_z": run_params.wall_z,
            "run_num": eval_num,
        }

        if self.config.per_run_timeout <= 0:
            result = self._run_single_integration(
                **integration_kwargs,
                cancel_flag=None,
            )
            return result, False

        result_container = [None]
        error_container = [None]
        cancel_flag = [False]

        def run_integration():
            try:
                result_container[0] = self._run_single_integration(
                    **integration_kwargs,
                    cancel_flag=cancel_flag,
                )
            except Exception as e:
                error_container[0] = e

        thread = threading.Thread(target=run_integration)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.config.per_run_timeout)

        if thread.is_alive():
            cancel_flag[0] = True
            self._log_result(
                f"[WARNING] Evaluation timed out for params {original_params} "
                f"after {self.config.per_run_timeout}s"
            )
            self._log_result("[WARNING] Signaling integration to cancel...")
            thread.join(timeout=2.0)
            return None, True

        if error_container[0] is not None:
            raise error_container[0]

        return result_container[0], False

    def _run_sweep_integration_attempt(
        self,
        run_params,
        params_dict: Dict[str, Any],
        *,
        timestep: float,
        steps: int,
        run_num: int,
        seed_override: int,
    ):
        """Run one sweep integration attempt, optionally under a timeout."""

        def run_single(cancel_flag):
            return self._run_single_integration(
                aperture=run_params.aperture,
                energy_gev=run_params.energy,
                start_z=run_params.start_z,
                transv_offset=run_params.transv_offset,
                timestep=timestep,
                steps=steps,
                rider_m_particle=run_params.rider_m_particle,
                rider_charge_sign=run_params.rider_charge_sign,
                rider_pcount=int(run_params.rider_pcount),
                rider_transv_mom=run_params.rider_transv_mom,
                rider_transv_dist=run_params.rider_transv_dist,
                rider_stripped_ions=run_params.rider_stripped_ions,
                macroparticle_charge_multiplier=(
                    run_params.macroparticle_charge_multiplier
                ),
                macroparticle_sigma_multiplier=(
                    run_params.macroparticle_sigma_multiplier
                ),
                driver_params=run_params.driver_params,
                wall_z=params_dict.get("wall_z", self.config.wall_z),
                run_num=run_num,
                cancel_flag=cancel_flag,
                seed_override=seed_override,
            )

        if self.config.per_run_timeout <= 0:
            return run_single(None), None, False

        result_container = [None]
        error_container = [None]
        cancel_flag = [False]

        if run_params.aperture < 0.1 and run_params.macroparticle_charge_multiplier > 1000:
            self._log_result(
                f"  [WARNING] Run {run_num}: Very small aperture ({run_params.aperture:.4f} mm) "
                f"with large charge multiplier ({run_params.macroparticle_charge_multiplier:.0f})"
            )
            self._log_result(
                "    This may cause numerical instability or slow convergence"
            )

        def run_with_exception_handling():
            try:
                result_container[0] = run_single(cancel_flag)
            except Exception as exc:
                error_container[0] = exc

        integration_thread = threading.Thread(target=run_with_exception_handling)
        integration_thread.daemon = True
        integration_thread.start()
        integration_thread.join(timeout=self.config.per_run_timeout)

        timed_out = False
        if integration_thread.is_alive():
            timed_out = True
            cancel_flag[0] = True
            self._log_result(
                f"  [TIMEOUT] Run {run_num} exceeded timeout of {self.config.per_run_timeout}s"
            )
            self._log_result(
                "    Signaling integration to cancel (thread will terminate when it checks cancel flag)"
            )
            integration_thread.join(timeout=2.0)
            if integration_thread.is_alive():
                self._log_result(
                    "    Warning: Integration thread still running after cancel signal"
                )
                self._log_result(
                    "    Thread will be abandoned (daemon thread will terminate with main thread)"
                )

        if error_container[0] is not None:
            return None, error_container[0], timed_out
        return result_container[0], None, timed_out

    def _run_sweep_background(self, is_finetune: bool = False, finetune_regions=None):
        """Run parameter sweep in background with real integration.

        Args:
            is_finetune: If True, this is a fine-tuning sweep
            finetune_regions: List of parameter regions for fine-tuning
        """
        # Set logging context for this sweep run
        context = "sweep_finetune" if is_finetune else "sweep"
        initialize_debug_logging(context=context, force_new_log=True)

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

            logging_policy = apply_run_logging_policy(self.config)
            use_no_logging = logging_policy.suppress_run_logs
            use_truncated_logging = logging_policy.use_truncated_run_logs
            use_full_debug = logging_policy.use_full_run_logs

            self._log_result(
                f"Starting BLIND SWEEP (Grid Search): {total_runs} total runs"
            )
            for line in describe_run_logging_policy(logging_policy):
                self._log_result(line)

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
                        "    All particles will travel to consistent z regardless of energy"
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
                run_params = resolve_sweep_run_parameters(self.config, params_dict)
                if run_params is None:
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

                aperture = run_params.aperture
                energy = run_params.energy
                start_z = run_params.start_z
                offset_frac = run_params.offset_frac
                rider_m_particle = run_params.rider_m_particle
                rider_charge_sign = run_params.rider_charge_sign
                rider_pcount = run_params.rider_pcount
                rider_transv_mom = run_params.rider_transv_mom
                rider_transv_dist = run_params.rider_transv_dist
                rider_stripped_ions = run_params.rider_stripped_ions
                macroparticle_charge_multiplier = (
                    run_params.macroparticle_charge_multiplier
                )
                macroparticle_sigma_multiplier = (
                    run_params.macroparticle_sigma_multiplier
                )
                driver_params_dict = run_params.driver_params
                transv_offset = run_params.transv_offset

                # Log parameter values based on verbosity
                if use_full_debug:
                    for line in build_full_debug_parameter_log_lines(
                        self.config,
                        run_params,
                        run_num=run_num,
                        total_runs=total_runs,
                        params_dict=params_dict,
                    ):
                        self._log_result(line)

                timestep_resolution = resolve_sweep_timestep(
                    self.config,
                    params_dict,
                    run_params,
                    run_num=run_num,
                    use_full_debug=use_full_debug,
                )
                timestep = timestep_resolution.timestep
                steps = timestep_resolution.steps
                expected_distance = timestep_resolution.expected_distance
                for line in timestep_resolution.log_lines:
                    self._log_result(line)

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
                        (
                            attempt_result,
                            attempt_error,
                            attempt_timed_out,
                        ) = self._run_sweep_integration_attempt(
                            run_params,
                            params_dict,
                            timestep=timestep,
                            steps=steps,
                            run_num=run_num,
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
                        attempt_classification = classify_sweep_attempt_result(
                            attempt_result,
                            run_num=run_num,
                            retry_attempt=retry_attempt,
                            include_debug_logs=(
                                use_full_debug or use_truncated_logging
                            ),
                        )
                        for line in attempt_classification.log_lines:
                            self._log_result(line)

                        if attempt_classification.succeeded:
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
                            attempt_error = attempt_classification.error

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
                        metric_summary = extract_sweep_metric_summary(result)

                        # Create run_data structure (used regardless of logging mode)
                        run_data = build_sweep_run_data(
                            run_number=run_num,
                            params_dict=params_dict,
                            simulation_type=self.config.simulation_type,
                            aperture=aperture,
                            energy=energy,
                            start_z=start_z,
                            transv_offset=transv_offset,
                            offset_frac=offset_frac,
                            timestep=timestep,
                            steps=steps,
                            retry_attempts=retry_attempt,
                            default_wall_z=self.config.wall_z,
                            rider_m_particle=rider_m_particle,
                            rider_charge_sign=rider_charge_sign,
                            rider_pcount=rider_pcount,
                            rider_transv_mom=rider_transv_mom,
                            rider_transv_dist=rider_transv_dist,
                            macroparticle_charge_multiplier=macroparticle_charge_multiplier,
                            macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
                            metrics=result.get("metrics", {}),
                            driver_params=driver_params_dict,
                        )

                        # Log based on verbosity mode
                        if use_no_logging:
                            # No logging mode: skip all run-level logs
                            pass
                        elif use_truncated_logging:
                            # Truncated mode: 1-2 lines with key info
                            log_params = build_truncated_sweep_log_params(
                                param_grids=param_grids,
                                params_dict=params_dict,
                                simulation_type=self.config.simulation_type,
                                aperture=aperture,
                                energy=energy,
                                wall_z=self.config.wall_z,
                            )
                            self._log_truncated_run(
                                run_num,
                                params=log_params,
                                metrics={
                                    "ΔE": metric_summary.delta_e,
                                    "Δγ": metric_summary.delta_gamma,
                                    "γ_i": metric_summary.gamma_initial,
                                    "γ_f": metric_summary.gamma_final,
                                },
                            )
                        elif use_full_debug:
                            # Full debug mode: all details
                            actual_distance = extract_actual_distance(result)
                            for line in build_full_debug_sweep_result_log_lines(
                                run_num=run_num,
                                total_runs=total_runs,
                                expected_distance=expected_distance,
                                actual_distance=actual_distance,
                                metrics=metric_summary,
                            ):
                                self._log_result(line)

                        # Add trajectory if requested (check if any trajectory saving is enabled)
                        # Note: save_top_n_trajectories only applies to optimization mode, not sweeps
                        save_traj = (
                            self.config.save_all_trajectories
                            or self.config.save_failed_trajectories
                        )
                        if save_traj and "trajectory" in result:
                            run_data["trajectory"] = result["trajectory"]

                        all_results.append(run_data)

                except Exception as e:
                    import traceback

                    error_details = traceback.format_exc()
                    run_error = str(e)

                    if self.config.skip_failed_runs:
                        self._log_result(f"[WARNING] Run {run_num} failed: {e}")
                        self._log_result(f"    Error details: {error_details}")
                        self._log_result(
                            "    Skipping and continuing with next run..."
                        )

                        # Record failed run
                        failed_runs.append(
                            build_failed_sweep_run_record(
                                run_num=run_num,
                                aperture=aperture,
                                energy=energy,
                                start_z=start_z,
                                transv_offset=transv_offset,
                                timestep=timestep,
                                steps=steps,
                                wall_z=params_dict.get("wall_z", self.config.wall_z),
                                error=run_error,
                                error_details=error_details,
                            )
                        )
                    else:
                        # Don't skip - re-raise and stop sweep
                        self._log_result(f"[ERROR] Run {run_num} failed: {e}")
                        self._log_result(f"    Error details: {error_details}")
                        self._log_result(
                            "    Stopping sweep (skip_failed_runs is disabled)"
                        )
                        raise

                # Handle timeout case
                if run_timed_out:
                    if self.config.skip_failed_runs:
                        self._log_result(
                            "    Skipping and continuing with next run..."
                        )
                        failed_runs.append(
                            build_timeout_sweep_run_record(
                                run_num=run_num,
                                aperture=aperture,
                                energy=energy,
                                start_z=start_z,
                                transv_offset=transv_offset,
                                timestep=timestep,
                                steps=steps,
                                timeout_seconds=self.config.per_run_timeout,
                            )
                        )
                    else:
                        self._log_result(
                            "    Stopping sweep (skip_failed_runs is disabled)"
                        )
                        break

            # Save results
            if all_results and self.config.save_results:
                self._save_sweep_results(all_results, failed_runs)

            if self.running:
                elapsed_time = time.time() - start_time
                for line in build_sweep_completion_log_lines(
                    output_dir=self.config.output_dir,
                    successful_runs=len(all_results),
                    failed_runs=len(failed_runs),
                    elapsed_time=elapsed_time,
                ):
                    self._log_result(line)
                self._update_progress(100, "Complete!")
        except Exception as e:
            self._log_result(f"[ERROR] Error during sweep: {e}")
            import traceback

            self._log_result(traceback.format_exc())
        finally:
            if "logging_policy" in locals():
                restore_run_logging_policy(self.config, logging_policy)

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
        for line in build_stability_config_log_lines(self.config, run_num=run_num):
            self._log_result(line)

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
            macroparticle_charge_multiplier=macroparticle_charge_multiplier,
            macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
            wall_z=wall_z,
            seed_override=seed_override,
            simulation_options_cls=SimulationOptions,
        )
        options = setup.options
        rider_m_particle = setup.rider_m_particle
        wall_z = setup.wall_z
        macroparticle_charge_multiplier = setup.macroparticle_charge_multiplier

        # Create progress callback to track integration
        def progress_callback(current: int, total: int, run_id=run_num):
            """Log progress periodically."""
            line = build_progress_log_line(
                run_num=run_id,
                current=current,
                total=total,
            )
            if line is not None:
                self._log_result(line)

        # Run the integration with progress tracking
        #
        # NOTE: We must always clean up the per-run temp directory, even when returning
        # early (halted runs) or raising exceptions. We do that by wrapping the entire
        # run/analysis section in a try/finally.
        try:
            # Log diagnostic info for potentially problematic configurations
            # Only check aperture for CONDUCTING_WALL modes
            if (
                not is_bunch_to_bunch(self.config.simulation_type)
                and aperture < 0.1
            ):
                diagnostic_line = build_small_aperture_diagnostic_line(
                    run_num=run_num,
                    aperture=aperture,
                )
                if diagnostic_line is not None:
                    self._log_result(diagnostic_line)
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
                    if should_emit_verbose_run_log(message):
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
            if self.config.timestep_strategy == "auto_distance":
                for line in build_final_z_check_log_lines(
                    trajectory=result.rider_trajectory,
                    simulation_type=self.config.simulation_type,
                    driver_params=driver_params,
                    target_distance_mm=self.config.target_distance_mm,
                    wall_z=wall_z,
                    run_num=run_num,
                ):
                    self._log_result(line)

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
                halted = build_halted_integration_output(
                    result,
                    run_num=run_num,
                    save_trajectory=(
                        self.config.save_all_trajectories
                        or self.config.save_failed_trajectories
                    ),
                    trajectory_stride=self.config.trajectory_stride,
                )
                for line in halted.log_lines:
                    self._log_result(line)
                return halted.output

            # Extract metrics (only for non-halted runs)
            self._log_result(f"  [DEBUG] Extracting metrics for Run {run_num}...")
            metrics_outcome = build_integration_metrics(
                result,
                rider_m_particle=rider_m_particle,
                run_num=run_num,
                optimization_mode=getattr(self.config, "mode", None) == "optimization",
            )
            for line in metrics_outcome.log_lines:
                self._log_result(line)

            output = {"metrics": metrics_outcome.metrics}
            trajectory_outcome = build_integration_trajectory_output(
                result,
                self.config,
                run_num=run_num,
                rider_m_particle=rider_m_particle,
                metrics=output["metrics"],
                save_trajectory=(
                    self.config.save_all_trajectories
                    or self.config.save_failed_trajectories
                ),
                trajectory_stride=self.config.trajectory_stride,
            )
            for line in trajectory_outcome.debug_print_lines:
                print(line)
            for line in trajectory_outcome.log_lines:
                self._log_result(line)
            output.update(trajectory_outcome.output_updates)

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
