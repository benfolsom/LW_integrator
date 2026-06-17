"""Result saving and plotting helpers for optimization flows."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

AMU_TO_MEV = 931.494  # Conversion factor amu to MeV


def save_optimization_results(plugin: Any, result: Any, param_names: List[str]):
    """Save optimization results to file in timestamped directory."""

    # Generate timestamp in sortable format: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Close any open log file before creating new directory
    if plugin._log_file is not None:
        plugin._close_log_file()

    maximize = "max" in plugin.config.objective.lower()
    best_metric_value = -result.fun if maximize else result.fun
    best_metric_serializable = (
        float(best_metric_value) if np.isfinite(best_metric_value) else None
    )

    config_name = "optimization"
    if hasattr(plugin, "last_loaded_config") and plugin.last_loaded_config:
        config_name = Path(plugin.last_loaded_config).stem

    method_suffix = plugin.config.optimization_method.replace("_", "")
    opt_dir = (
        Path(plugin.sweep_output_dir) / f"{timestamp}_{config_name}_{method_suffix}"
    )
    opt_dir.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "optimization_method": plugin.config.optimization_method,
        "objective": plugin.config.objective,
        "best_parameters": result.best_params_dict,
        "best_value": best_metric_serializable,
        "function_evaluations": int(result.nfev) if hasattr(result, "nfev") else None,
        "success": bool(result.success),
        "message": str(result.message) if hasattr(result, "message") else None,
        "timestamp": timestamp,
    }

    # Get all_evaluations early so we can use it for top_n_results
    all_evaluations = None
    if hasattr(plugin, "_all_evaluations_cache"):
        all_evaluations = plugin._all_evaluations_cache

    if hasattr(result, "final_population") and hasattr(result, "final_fitness"):
        fitness_array = np.asarray(result.final_fitness, dtype=float)
        finite_indices = np.where(np.isfinite(fitness_array))[0]
        if finite_indices.size == 0:
            plugin._log_result(
                "[INFO] Skipping top-N summary (no finite fitness values)."
            )
            results_dict["top_n_results"] = []
            results_dict["top_n_count"] = 0
        else:
            top_n = max(1, int(plugin.config.optimization_save_top_n))
            sorted_finite = np.argsort(fitness_array[finite_indices])
            sorted_indices = finite_indices[sorted_finite]
            n_available = min(top_n, len(sorted_indices))

            top_n_summary = []
            maximize = "max" in plugin.config.objective.lower()
            for i in range(n_available):
                idx = sorted_indices[i]
                params_array = result.final_population[idx]
                params_dict = dict(zip(param_names, params_array))
                fitness = fitness_array[idx]
                metric_value = -fitness if maximize else fitness

                top_n_entry = {
                    "rank": i + 1,
                    "parameters": params_dict,
                    "fitness": float(fitness),
                    "metric_value": float(metric_value),
                }

                # Try to find corresponding evaluation to include full metrics
                if all_evaluations:
                    # Find evaluation with matching fitness value
                    for eval_rec in all_evaluations:
                        if not eval_rec.get("failed", False):
                            eval_fitness = eval_rec.get("fitness")
                            if (
                                eval_fitness is not None
                                and abs(eval_fitness - fitness) < 1e-9
                            ):
                                # Found matching evaluation - include its metrics
                                if "metrics" in eval_rec:
                                    top_n_entry["metrics"] = eval_rec["metrics"]
                                if "objective_value" in eval_rec:
                                    top_n_entry["objective_value"] = eval_rec[
                                        "objective_value"
                                    ]
                                if "raw_objective_value" in eval_rec:
                                    top_n_entry["raw_objective_value"] = eval_rec[
                                        "raw_objective_value"
                                    ]
                                if "soft_penalty" in eval_rec:
                                    top_n_entry["soft_penalty"] = eval_rec[
                                        "soft_penalty"
                                    ]
                                break

                top_n_summary.append(top_n_entry)

            results_dict["top_n_results"] = top_n_summary
            results_dict["top_n_count"] = n_available

    if hasattr(result, "convergence_history"):
        results_dict["convergence_history"] = result.convergence_history

    # all_evaluations already retrieved above for top_n_results
    if all_evaluations:
        all_evaluations_for_json = []
        for eval_rec in all_evaluations:
            eval_rec_copy = dict(eval_rec)
            if "trajectory" in eval_rec_copy:
                del eval_rec_copy["trajectory"]
            all_evaluations_for_json.append(eval_rec_copy)

        results_dict["all_evaluations"] = all_evaluations_for_json
        results_dict["total_evaluations"] = len(all_evaluations)

    export_format = plugin.config.metrics_export_format
    export_scope = plugin.config.metrics_export_scope

    evaluations_to_export = None
    scope_desc = "0"
    finite_evals = (
        [
            e
            for e in all_evaluations
            if np.isfinite(e.get("objective_value", float("inf")))
        ]
        if all_evaluations
        else []
    )

    if export_scope == "top_n" and finite_evals:
        top_n = max(1, int(plugin.config.optimization_save_top_n))
        sorted_evals = sorted(
            finite_evals, key=lambda e: e.get("objective_value", float("inf"))
        )
        evaluations_to_export = sorted_evals[:top_n]
        scope_desc = f"top {len(evaluations_to_export)}"
    elif export_scope == "top_n" and not finite_evals and all_evaluations:
        plugin._log_result(
            "[INFO] Skipping top-N export (no finite evaluation metrics)."
        )
    elif all_evaluations:
        evaluations_to_export = all_evaluations
        scope_desc = "all"

    if export_format in ["json", "both"]:
        if export_scope == "top_n" and evaluations_to_export:
            results_dict["all_evaluations"] = [
                {k: v for k, v in e.items() if k != "trajectory"}
                for e in evaluations_to_export
            ]
            results_dict["total_evaluations"] = len(evaluations_to_export)
            results_dict["export_scope"] = "top_n"

        results_file = opt_dir / "optimization_results.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        plugin._log_result(
            f"Results saved to JSON ({scope_desc} evaluations): {results_file}"
        )

    if export_format in ["csv", "both"] and evaluations_to_export:
        plugin._export_evaluations_csv(evaluations_to_export, param_names, opt_dir)
        plugin._log_result(f"Metrics exported to CSV ({scope_desc} evaluations)")

    if plugin.config.save_all_trajectories and all_evaluations:
        plugin._log_result("")
        plugin._log_result("Saving all evaluation trajectories...")
        saved_count = 0
        for eval_rec in all_evaluations:
            if not eval_rec.get("failed", True) and "trajectory" in eval_rec:
                eval_num = eval_rec["evaluation"]
                traj_file = plugin._save_evaluation_trajectory(
                    eval_num, eval_rec["trajectory"], opt_dir
                )
                if traj_file:
                    saved_count += 1
        plugin._log_result(f"  Saved {saved_count} evaluation trajectories")

    plugin._generate_optimization_plots(result, param_names, opt_dir)
    plugin._last_optimization_dir = opt_dir

    if hasattr(result, "final_population") and hasattr(result, "final_fitness"):
        plugin._save_top_trajectories_summary_table(result, param_names, opt_dir)

    # Copy debug log from logcache to results directory
    if hasattr(plugin, "_log_file_path") and plugin._log_file_path is not None:
        if plugin._log_file_path.exists():
            dest_log = opt_dir / plugin._log_file_path.name
            try:
                shutil.copy2(plugin._log_file_path, dest_log)
                plugin._log_result(f"[OK] Debug log copied to: {dest_log}")
            except Exception as e:
                plugin._log_result(f"[WARNING] Failed to copy debug log: {e}")
    else:
        # Try to find the most recent logcache file for this context
        logcache_dir = Path(plugin.config.output_dir).parent.parent / "logcache"
        if logcache_dir.exists():
            # Find most recent optimization log
            log_files = sorted(
                logcache_dir.glob("*optimization*.log"), key=lambda p: p.stat().st_mtime
            )
            if log_files:
                most_recent_log = log_files[-1]

                dest_log = opt_dir / most_recent_log.name
                try:
                    shutil.copy2(most_recent_log, dest_log)
                    plugin._log_result(f"[OK] Debug log copied to: {dest_log}")
                except Exception as e:
                    plugin._log_result(f"[WARNING] Failed to copy debug log: {e}")

    plugin._open_log_file(opt_dir)


def save_top_trajectories_summary_table(
    plugin: Any, result: Any, param_names: List[str], output_dir: Path
):
    """Generate and save a human-readable table of top trajectories with optimization metrics."""
    try:
        fitness_array = np.asarray(result.final_fitness, dtype=float)
        finite_indices = np.where(np.isfinite(fitness_array))[0]
        if finite_indices.size == 0:
            plugin._log_result(
                "[INFO] Skipping top-N trajectory table (no finite fitness values)."
            )
            return

        top_n = max(1, int(plugin.config.optimization_save_top_n))
        sorted_finite = np.argsort(fitness_array[finite_indices])
        sorted_indices = finite_indices[sorted_finite]
        n_available = min(top_n, len(sorted_indices))

        maximize = "max" in plugin.config.objective.lower()

        table_lines = []
        table_lines.append("=" * 120)
        table_lines.append(f"TOP {n_available} OPTIMIZATION RESULTS")
        table_lines.append(f"Objective: {plugin.config.objective}")
        table_lines.append("=" * 120)
        table_lines.append("")

        header = f"{'Rank':<6} {'Fitness':<15} {'Metric Value':<15} {'ΔE (MeV)':<15} {'ΔE/E (%)':<15} {'Parameters'}"
        table_lines.append(header)
        table_lines.append("-" * 120)

        for i in range(n_available):
            idx = sorted_indices[i]
            params_array = result.final_population[idx]
            params_dict = dict(zip(param_names, params_array))
            fitness = fitness_array[idx]
            metric_value = -fitness if maximize else fitness

            if "percent" in plugin.config.objective.lower():
                percent_delta_e = metric_value
                initial_energy_gev = params_dict.get("initial_energy_gev", 100.0)
                _rest_mev = (
                    getattr(plugin.config, "m_particle", 0.00054857990907) * AMU_TO_MEV
                )
                gamma_initial = initial_energy_gev * 1e3 / _rest_mev
                delta_e_mev = (percent_delta_e / 100.0) * gamma_initial * _rest_mev
            elif (
                "energy_gain" in plugin.config.objective.lower()
                or "delta_e" in plugin.config.objective.lower()
            ):
                delta_e_mev = metric_value
                initial_energy_gev = params_dict.get("initial_energy_gev", 100.0)
                _rest_mev = (
                    getattr(plugin.config, "m_particle", 0.00054857990907) * AMU_TO_MEV
                )
                gamma_initial = initial_energy_gev * 1e3 / _rest_mev
                percent_delta_e = (delta_e_mev / (gamma_initial * _rest_mev)) * 100.0
            else:
                delta_e_mev = float("nan")
                percent_delta_e = float("nan")

            param_str = ", ".join([f"{k}={v:.4g}" for k, v in params_dict.items()])
            if len(param_str) > 50:
                param_str = param_str[:47] + "..."

            row = f"{i + 1:<6} {fitness:<15.6e} {metric_value:<15.6e} {delta_e_mev:<15.6f} {percent_delta_e:<15.6e} {param_str}"
            table_lines.append(row)

        table_lines.append("=" * 120)
        table_lines.append("")
        table_lines.append("Notes:")
        table_lines.append("  - Fitness: Internal optimizer value (lower is better)")
        table_lines.append("  - Metric Value: Actual objective value")
        table_lines.append("  - ΔE: Energy change in MeV")
        table_lines.append("  - ΔE/E: Percent energy change")
        table_lines.append("  - Inf/-Inf values indicate halted/failed runs")
        table_lines.append("")

        table_file = output_dir / "top_trajectories_summary.txt"
        with open(table_file, "w") as f:
            f.write("\n".join(table_lines))

        plugin._log_result("")
        plugin._log_result("=" * 80)
        for line in table_lines[:20]:
            plugin._log_result(line)
        if len(table_lines) > 20:
            plugin._log_result("... (see full table in top_trajectories_summary.txt)")

        plugin._log_result("")

    except Exception as e:
        plugin._log_result(
            f"[WARNING] Failed to write top trajectories summary table: {e}"
        )


def generate_optimization_plots(
    plugin: Any, result: Any, param_names: List[str], output_dir: Path
):
    """Generate optimization visualization plots."""
    import matplotlib.pyplot as plt

    try:
        if hasattr(result, "convergence_history") and result.convergence_history:
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            history = result.convergence_history
            generations = [h["generation"] for h in history]
            best_fitness = [h["best_fitness"] for h in history]
            mean_fitness = [h["mean_fitness"] for h in history]
            std_fitness = [h["std_fitness"] for h in history]

            maximize = "max" in plugin.config.objective.lower()
            if maximize:
                best_values = [-f for f in best_fitness]
                mean_values = [-f for f in mean_fitness]
            else:
                best_values = best_fitness
                mean_values = mean_fitness

            axes[0].plot(generations, best_values, "b-", linewidth=2, label="Best")
            axes[0].plot(generations, mean_values, "g--", linewidth=1.5, label="Mean")
            axes[0].fill_between(
                generations,
                [m - s for m, s in zip(mean_values, std_fitness)],
                [m + s for m, s in zip(mean_values, std_fitness)],
                alpha=0.2,
                color="green",
                label="±1 std",
            )
            axes[0].set_xlabel("Generation")
            axes[0].set_ylabel(f"{plugin.config.objective}")
            axes[0].set_title("Optimization Convergence")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(generations, std_fitness, "r-", linewidth=2)
            axes[1].set_xlabel("Generation")
            axes[1].set_ylabel("Population Std Dev")
            axes[1].set_title("Population Diversity")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            convergence_plot = output_dir / "optimization_convergence.png"
            plt.savefig(convergence_plot, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plugin._log_result(f"Convergence plot saved to: {convergence_plot}")

        if len(param_names) == 2 and hasattr(result, "final_population"):
            fig, ax = plt.subplots(figsize=(8, 6))

            population = np.asarray(result.final_population)
            fitness = np.asarray(result.final_fitness, dtype=float)
            finite_mask = np.isfinite(fitness)
            if not np.any(finite_mask):
                plugin._log_result(
                    "[INFO] Skipping parameter exploration plot (no finite final population)."
                )
                plt.close(fig)
                return

            population = population[finite_mask]
            fitness = fitness[finite_mask]
            maximize = "max" in plugin.config.objective.lower()
            plot_fitness = -fitness if maximize else fitness

            scatter = ax.scatter(
                population[:, 0],
                population[:, 1],
                c=plot_fitness,
                cmap="viridis",
                s=50,
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5,
            )

            best_idx = np.argmin(fitness)
            ax.scatter(
                population[best_idx, 0],
                population[best_idx, 1],
                c="red",
                s=200,
                marker="*",
                edgecolors="black",
                linewidth=1.5,
                label="Best",
                zorder=5,
            )

            ax.set_xlabel(param_names[0])
            ax.set_ylabel(param_names[1])
            ax.set_title("Parameter Space Exploration (Final Population)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(plugin.config.objective)

            plt.tight_layout()
            param_plot = output_dir / "parameter_exploration.png"
            plt.savefig(param_plot, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plugin._log_result(f"Parameter exploration plot saved to: {param_plot}")

        if hasattr(result, "best_params_dict"):
            fig, ax = plt.subplots(figsize=(8, max(4, len(param_names) * 0.5)))

            params = list(result.best_params_dict.keys())
            values = list(result.best_params_dict.values())

            ax.barh(params, values, color="steelblue", edgecolor="black")
            ax.set_xlabel("Value")
            ax.set_title("Best Parameters Found")
            ax.grid(True, alpha=0.3, axis="x")

            plt.tight_layout()
            params_plot = output_dir / "best_parameters.png"
            plt.savefig(params_plot, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plugin._log_result(f"Best parameters plot saved to: {params_plot}")

    except Exception as e:
        import traceback

        plugin._log_result(f"[WARNING] Failed to generate optimization plots: {e}")
        plugin._log_result(traceback.format_exc())


def generate_optimization_heatmap(
    plugin: Any, all_evaluations: List[dict], param_names: List[str], output_dir: Path
):
    """Generate sparse heatmap from optimization evaluations."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    try:
        if len(all_evaluations) < 3:
            plugin._log_result(
                "[INFO] Not enough evaluations for heatmap (need at least 3)"
            )
            return

        successful_evals = [
            e
            for e in all_evaluations
            if not e.get("failed", False)
            and np.isfinite(e.get("objective_value", np.inf))
        ]

        if len(successful_evals) < 3:
            plugin._log_result("[INFO] Not enough successful evaluations for heatmap")
            return

        maximize = "max" in plugin.config.objective.lower()

        if len(param_names) > 2:
            plugin._log_result(
                f"[INFO] Skipping heatmap generation for {len(param_names)}-parameter optimization (heatmaps only for 1-2 parameters)"
            )
            return

        if len(param_names) == 2:
            plugin._log_result(
                f"[INFO] Generating 2D optimization heatmap for {param_names[0]} vs {param_names[1]}"
            )

            fig, ax = plt.subplots(figsize=(10, 8))

            x_vals = [e["parameters"][param_names[0]] for e in successful_evals]
            y_vals = [e["parameters"][param_names[1]] for e in successful_evals]
            z_vals = [e["objective_value"] for e in successful_evals]

            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)

            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= x_range * 0.05
            x_max += x_range * 0.05
            y_min -= y_range * 0.05
            y_max += y_range * 0.05

            grid_x, grid_y = np.meshgrid(
                np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
            )

            grid_z = griddata(
                (x_vals, y_vals),
                z_vals,
                (grid_x, grid_y),
                method="linear",
                fill_value=np.nan,
            )

            im = ax.contourf(
                grid_x,
                grid_y,
                grid_z,
                levels=20,
                cmap="RdYlGn" if maximize else "RdYlGn_r",
                extend="both",
            )

            ax.scatter(
                x_vals,
                y_vals,
                c=z_vals,
                s=30,
                edgecolors="black",
                linewidth=0.5,
                cmap="RdYlGn" if maximize else "RdYlGn_r",
                zorder=5,
                alpha=0.7,
            )

            best_idx = np.argmax(z_vals) if maximize else np.argmin(z_vals)
            ax.scatter(
                x_vals[best_idx],
                y_vals[best_idx],
                c="red",
                s=300,
                marker="*",
                edgecolors="black",
                linewidth=2,
                label="Best",
                zorder=10,
            )

            ax.set_xlabel(param_names[0])
            ax.set_ylabel(param_names[1])
            ax.set_title(
                f"Optimization Landscape: {plugin.config.objective}\n({len(successful_evals)} evaluations)"
            )
            ax.legend()

            if x_max / (x_min + 1e-10) > 100:
                ax.set_xscale("log")
            if y_max / (y_min + 1e-10) > 100:
                ax.set_yscale("log")

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(plugin.config.objective)

            plt.tight_layout()
            heatmap_file = output_dir / "optimization_heatmap_2d.png"
            plt.savefig(heatmap_file, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plugin._log_result(f"[OK] 2D optimization heatmap saved to: {heatmap_file}")
        else:
            plugin._log_result(
                "[INFO] Single parameter optimization - no heatmap needed"
            )

    except Exception as e:
        import traceback

        plugin._log_result(f"[WARNING] Failed to generate optimization plots: {e}")
        plugin._log_result(f"[WARNING] Traceback: {traceback.format_exc()}")


def save_top_n_optimization_trajectories(
    plugin: Any, result: Any, param_names: List[str]
):
    """Re-run top N parameter sets and save trajectories to timestamped directory."""
    if getattr(plugin, "_was_cancelled", False):
        plugin._log_result("")
        plugin._log_result(
            "[INFO] Skipping top-N trajectory regeneration (optimization cancelled)."
        )
        return

    try:
        top_n = max(1, int(plugin.config.optimization_save_top_n))
        top_params_list = []

        if hasattr(result, "final_population") and hasattr(result, "final_fitness"):
            fitness_array = np.asarray(result.final_fitness, dtype=float)
            finite_indices = np.where(np.isfinite(fitness_array))[0]
            if finite_indices.size == 0:
                plugin._log_result(
                    "[INFO] No finite fitness values available; skipping top-N trajectory generation."
                )
                return

            sorted_finite = np.argsort(fitness_array[finite_indices])
            sorted_indices = finite_indices[sorted_finite]
            n_available = min(top_n, len(sorted_indices))

            plugin._log_result("")
            plugin._log_result(
                f"Generating top {n_available} trajectories from population..."
            )

            from optimization.run_parameter_helpers import (
                collect_optimization_parameter_selection,
                decode_optimization_parameter_values,
            )

            selection = collect_optimization_parameter_selection(plugin.config)
            param_log_scales = selection.log_scales

            for i in range(n_available):
                idx = sorted_indices[i]
                params_array = result.final_population[idx]
                params_values = decode_optimization_parameter_values(
                    params_array,
                    param_log_scales,
                )
                params_dict = dict(zip(param_names, params_values))
                fitness = fitness_array[idx]
                top_params_list.append(
                    {"params": params_dict, "fitness": fitness, "rank": i + 1}
                )
        else:
            plugin._log_result("")
            plugin._log_result("Generating best trajectory...")

            top_params_list.append(
                {"params": result.best_params_dict, "fitness": result.fun, "rank": 1}
            )

        trajectory_data_list = []
        for item in top_params_list:
            traj_data = plugin._save_single_optimization_trajectory(
                params_dict=item["params"],
                param_names=param_names,
                rank=item["rank"],
                fitness=item["fitness"],
            )
            if traj_data:
                trajectory_data_list.append(
                    {
                        "rank": item["rank"],
                        "params": item["params"],
                        "fitness": item["fitness"],
                        "trajectory": traj_data,
                    }
                )

        if len(trajectory_data_list) >= 2:
            plugin._generate_trajectory_comparison_plot(trajectory_data_list)

    except Exception as e:
        import traceback

        plugin._log_result(f"[WARNING] Failed to save top N trajectories: {e}")
        plugin._log_result(f"[WARNING] Traceback: {traceback.format_exc()}")


def generate_trajectory_comparison_plot(plugin: Any, trajectory_data_list: List[dict]):
    """Generate comparison plot showing all top N trajectories overlaid."""
    import matplotlib.pyplot as plt

    try:
        output_dir = getattr(
            plugin, "_last_optimization_dir", Path(plugin.config.output_dir)
        )
        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectory_data_list)))
        all_gamma_values = []

        for i, item in enumerate(trajectory_data_list):
            rank = item["rank"]
            traj = item["trajectory"]
            color = colors[i]

            z = np.array(traj.get("z", []))
            t = np.array(traj.get("t", []))
            r = np.array(traj.get("r", []))
            gamma = np.array(traj.get("gamma", []))
            pr = np.array(traj.get("pr", []))

            if len(z) == 0:
                continue

            label = f"Rank #{rank}" if rank > 1 else "Best"

            if len(gamma) > 0:
                gamma_initial = gamma[0]
                delta_gamma = gamma - gamma_initial
                # Use actual particle rest energy (not hardcoded electron mass)
                rest_energy_mev = (
                    getattr(plugin.config, "m_particle", 0.00054857990907) * AMU_TO_MEV
                )
                delta_e_mev = delta_gamma * rest_energy_mev
                percent_delta_e = (delta_gamma / gamma_initial) * 100.0
                all_gamma_values.extend(gamma.tolist())
            else:
                delta_e_mev = np.zeros_like(z)
                percent_delta_e = np.zeros_like(z)

            if len(t) > 0 and len(z) > 0:
                axes[0, 0].plot(t, z, color=color, linewidth=2, label=label, alpha=0.8)

            if len(r) > 0 and len(z) > 0:
                axes[0, 1].plot(
                    z, r * 1e3, color=color, linewidth=2, label=label, alpha=0.8
                )

            if len(gamma) > 0 and len(z) > 0:
                axes[1, 0].plot(
                    z, gamma, color=color, linewidth=2, label=label, alpha=0.8
                )

            if len(delta_e_mev) > 0 and len(z) > 0:
                axes[1, 1].plot(
                    z, delta_e_mev, color=color, linewidth=2, label=label, alpha=0.8
                )

            if len(percent_delta_e) > 0 and len(z) > 0:
                axes[2, 0].plot(
                    z, percent_delta_e, color=color, linewidth=2, label=label, alpha=0.8
                )

            if len(pr) > 0 and len(z) > 0:
                axes[2, 1].plot(z, pr, color=color, linewidth=2, label=label, alpha=0.8)

        axes[0, 0].set_xlabel("Time (ns)")
        axes[0, 0].set_ylabel("z (mm)")
        axes[0, 0].set_title("Longitudinal Position")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=9, loc="best")

        axes[0, 1].set_xlabel("z (mm)")
        axes[0, 1].set_ylabel("r (μm)")
        axes[0, 1].set_title("Transverse Position (Radial)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=9, loc="best")

        axes[1, 0].set_xlabel("z (mm)")
        axes[1, 0].set_ylabel("γ")
        axes[1, 0].set_title("Lorentz Factor")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=9, loc="best")
        if len(all_gamma_values) > 0:
            gamma_min = np.min(all_gamma_values)
            gamma_max = np.max(all_gamma_values)
            gamma_mean = np.mean(all_gamma_values)
            gamma_range = gamma_max - gamma_min
            if gamma_range > 0:
                margin = max(gamma_range * 0.1, gamma_mean * 0.001)
                axes[1, 0].set_ylim([gamma_min - margin, gamma_max + margin])

        axes[1, 1].set_xlabel("z (mm)")
        axes[1, 1].set_ylabel("ΔE (MeV)")
        axes[1, 1].set_title("Energy Change")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=9, loc="best")
        axes[1, 1].axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

        axes[2, 0].set_xlabel("z (mm)")
        axes[2, 0].set_ylabel("ΔE/E (%)")
        axes[2, 0].set_title("Percent Energy Change")
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend(fontsize=9, loc="best")
        axes[2, 0].axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

        axes[2, 1].set_xlabel("z (mm)")
        axes[2, 1].set_ylabel("pr")
        axes[2, 1].set_title("Transverse Momentum (Radial)")
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend(fontsize=9, loc="best")

        plt.suptitle(
            f"Top {len(trajectory_data_list)} Trajectory Comparison",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        comparison_plot = output_dir / "trajectory_comparison.png"
        plt.savefig(comparison_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)

        plugin._log_result(
            f"[OK] Trajectory comparison plot saved to: {comparison_plot}"
        )

    except Exception as e:
        import traceback

        plugin._log_result(f"[WARNING] Failed to generate comparison plot: {e}")
        plugin._log_result(f"[WARNING] Traceback: {traceback.format_exc()}")


def save_partial_optimization_results(
    plugin: Any,
    all_evaluations: List[dict],
    param_names: List[str],
    status: str = "PARTIAL",
):
    """Save partial optimization results when cancelled or failed."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if plugin.config.mode == "optimization":
        method = plugin.config.optimization_method
        output_dir = Path(plugin.sweep_output_dir) / f"partial_{method}_{timestamp}"
    else:
        output_dir = Path(plugin.sweep_output_dir) / f"partial_sweep_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "partial_results.csv"

    successful_evals = [
        e
        for e in all_evaluations
        if not e.get("failed", False) and not e.get("halted_early", False)
    ]
    halted_evals = [e for e in all_evaluations if e.get("halted_early", False)]

    if len(all_evaluations) > 0:
        import csv

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["evaluation"]
                + param_names
                + ["objective_value", "failed", "halted_early", "halt_reason"],
            )
            writer.writeheader()
            for e in all_evaluations:
                row = {
                    "evaluation": e["evaluation"],
                    "failed": e.get("failed", False),
                    "halted_early": e.get("halted_early", False),
                    "halt_reason": e.get("halt_reason", ""),
                }
                row.update(e["parameters"])
                row["objective_value"] = e.get("objective_value", float("nan"))
                writer.writerow(row)
        plugin._log_result(f"[OK] Partial results saved to: {csv_path}")

    summary = {
        "status": status,
        "timestamp": timestamp,
        "total_evaluations": len(all_evaluations),
        "successful_evaluations": len(successful_evals),
        "halted_evaluations": len(halted_evals),
        "failed_evaluations": len(all_evaluations)
        - len(successful_evals)
        - len(halted_evals),
        "parameters": param_names,
        "objective": plugin.config.objective,
    }

    if len(successful_evals) > 0:
        maximize = "max" in plugin.config.objective.lower()
        finite_evals = [
            e for e in successful_evals if np.isfinite(e.get("objective_value", np.inf))
        ]

        if len(finite_evals) > 0:
            if maximize:
                best = max(
                    finite_evals, key=lambda x: x.get("objective_value", -float("inf"))
                )
            else:
                best = min(
                    finite_evals, key=lambda x: x.get("objective_value", float("inf"))
                )
            summary["best_parameters"] = best["parameters"]
            summary["best_value"] = best["objective_value"]
        else:
            summary["note"] = "No finite objective values found"

    summary_path = output_dir / "partial_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    plugin._log_result(f"[OK] Summary saved to: {summary_path}")

    if plugin._log_file_path is not None and plugin._log_file_path.exists():
        dest_log = output_dir / plugin._log_file_path.name
        shutil.copy2(plugin._log_file_path, dest_log)
        plugin._log_result(f"[OK] Log file saved to: {dest_log}")


def relocate_incomplete_sweep(
    sweep_dir: Path,
    min_runs: int = 100,
    log_fn: Optional[Any] = None,
) -> Optional[Path]:
    """Move a sweep directory to archive/incomplete if it has fewer than *min_runs*.

    The incomplete directory is placed as a sibling of the sweep output root::

        results/sweeps/20260309_...  ->  results/archive/incomplete/20260309_...

    The function inspects ``sweep_results.json`` inside *sweep_dir*.  If the
    file is missing or the recorded ``total_runs`` (falling back to
    ``len(results)``) is below *min_runs*, the entire directory is relocated.

    Parameters
    ----------
    sweep_dir : Path
        Fully-resolved path to a single sweep output directory
        (e.g. ``results/sweeps/20260309_121938_config_name``).
    min_runs : int, optional
        Minimum number of completed runs to keep a sweep in its original
        location.  Defaults to 100.
    log_fn : callable, optional
        Logging callable (e.g. ``print`` or ``self._log``).  If *None*, uses
        the module-level logger at INFO level.

    Returns
    -------
    Path or None
        The new path if the directory was moved, or *None* if it was kept
        in place (i.e. it met the threshold).
    """
    if log_fn is None:
        log_fn = logger.info

    sweep_dir = Path(sweep_dir)
    results_file = sweep_dir / "sweep_results.json"

    # Determine run count
    num_runs = 0
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
            num_runs = data.get("total_runs", len(data.get("results", [])))
        except (json.JSONDecodeError, OSError) as exc:
            log_fn(f"[WARN] Could not read {results_file}: {exc}")
    else:
        log_fn(f"[WARN] No sweep_results.json in {sweep_dir}")

    if num_runs >= min_runs:
        return None

    # Build the incomplete destination:
    #   <sweep_root>/../archive/incomplete/<dir_name>
    # e.g. results/sweeps/ABC  ->  results/archive/incomplete/ABC
    sweep_root = sweep_dir.parent  # e.g. results/sweeps
    incomplete_dir = sweep_root.parent / "archive" / "incomplete"
    incomplete_dir.mkdir(parents=True, exist_ok=True)

    dest = incomplete_dir / sweep_dir.name
    if dest.exists():
        # Avoid collision — append a counter
        counter = 1
        while dest.exists():
            dest = incomplete_dir / f"{sweep_dir.name}_{counter}"
            counter += 1

    shutil.move(str(sweep_dir), str(dest))
    log_fn(f"[INFO] Sweep has {num_runs} runs (< {min_runs}); moved to {dest}")
    return dest


__all__ = [
    "save_optimization_results",
    "save_partial_optimization_results",
    "relocate_incomplete_sweep",
]
