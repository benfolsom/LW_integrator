"""Results/export helpers for OptimizationPlugin."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, List

import matplotlib.patheffects as PathEffects
import numpy as np
from matplotlib.colors import LogNorm

AMU_TO_MEV = 931.494  # Conversion factor amu to MeV

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

from optimization.result_io import (  # type: ignore[import]
    generate_optimization_heatmap,
    generate_optimization_plots,
    generate_trajectory_comparison_plot,
    save_optimization_results,
    save_partial_optimization_results,
    save_top_n_optimization_trajectories,
    save_top_trajectories_summary_table,
)
from optimization.ui_helpers import (  # type: ignore[import]
    show_error_dialog as _show_error_dialog,
)
from optimization.ui_helpers import (
    show_info_dialog as _show_info_dialog,
)


class OptimizationResultsMixin:
    """Encapsulates result export and plotting helpers."""

    def _save_optimization_results(self, result, param_names):
        """Save optimization results to file via shared helper."""
        return save_optimization_results(self, result, param_names)

    def _save_top_trajectories_summary_table(self, result, param_names, output_dir):
        """Generate and save top trajectories summary via helper."""
        return save_top_trajectories_summary_table(
            self, result, param_names, output_dir
        )

    def _generate_optimization_plots(self, result, param_names, output_dir):
        """Generate optimization plots via shared helper."""
        return generate_optimization_plots(self, result, param_names, output_dir)

    def _generate_optimization_heatmap(self, all_evaluations, param_names, output_dir):
        """Generate optimization heatmap via shared helper."""
        return generate_optimization_heatmap(
            self, all_evaluations, param_names, output_dir
        )

    def _save_top_n_optimization_trajectories(self, result, param_names):
        """Re-run top N parameter sets and save trajectories via helper."""
        return save_top_n_optimization_trajectories(self, result, param_names)

    def _save_single_optimization_trajectory(
        self, params_dict, param_names, rank, fitness
    ):
        """Re-run a single parameter set and save its trajectory."""
        from pathlib import Path

        import numpy as np

        try:
            # Set up run parameters (similar to evaluate_params)
            aperture = self.config.aperture_range[0]
            energy = self.config.energy_range[0]
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
            wall_z = self.config.wall_z
            rider_transv_mom = self.config.transv_mom  # default
            rider_transv_dist = self.config.transv_dist  # default
            macroparticle_charge_mult = (
                self.config.macroparticle_charge_multiplier
            )  # default
            macroparticle_sigma_mult = (
                self.config.macroparticle_sigma_multiplier
            )  # default

            # Map parameters
            for param_name, value in params_dict.items():
                if param_name == "aperture_radius":
                    aperture = value
                elif param_name == "initial_energy_gev":
                    energy = value
                elif param_name == "start_z":
                    start_z = value
                elif param_name == "transverse_offset":
                    offset_frac = value
                elif param_name == "timestep":
                    timestep = value
                elif param_name == "wall_z":
                    wall_z = value
                elif param_name == "transverse_momentum":
                    rider_transv_mom = value
                elif param_name == "rider_transv_dist":
                    rider_transv_dist = value
                elif param_name == "macroparticle_charge_multiplier":
                    macroparticle_charge_mult = value
                elif param_name == "macroparticle_sigma_multiplier":
                    macroparticle_sigma_mult = value

            transv_offset = offset_frac * aperture

            # Temporarily enable trajectory saving
            save_all_backup = self.config.save_all_trajectories
            self.config.save_all_trajectories = True

            # Run integration
            result_data = self._run_single_integration(
                aperture=aperture,
                energy_gev=energy,
                start_z=start_z,
                transv_offset=transv_offset,
                timestep=timestep,
                steps=steps,
                rider_m_particle=self.config.m_particle,
                rider_charge_sign=self.config.charge_sign,
                rider_pcount=int(self.config.pcount),
                rider_transv_mom=rider_transv_mom,
                rider_transv_dist=rider_transv_dist,
                macroparticle_charge_multiplier=macroparticle_charge_mult,
                macroparticle_sigma_multiplier=macroparticle_sigma_mult,
                driver_params=None,
                wall_z=wall_z,
                run_num=9999 + rank,
            )

            # Restore trajectory setting
            self.config.save_all_trajectories = save_all_backup

            if result_data and "trajectory" in result_data:
                output_dir = getattr(
                    self, "_last_optimization_dir", Path(self.config.output_dir)
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                import matplotlib.pyplot as plt

                traj = result_data["trajectory"]
                metrics = result_data.get("metrics", {})

                fig, axes = plt.subplots(3, 2, figsize=(14, 14))

                z = np.array(traj["z"])
                t = np.array(traj["t"])
                r = np.array(traj["r"])
                gamma_arr = np.array(traj.get("gamma", []))
                pr = np.array(traj.get("pr", []))

                # Use actual particle rest energy (not hardcoded electron mass)
                rest_energy_mev = (
                    getattr(self.config, "m_particle", 0.00054857990907) * AMU_TO_MEV
                )
                if len(gamma_arr) > 0:
                    gamma_initial = gamma_arr[0]
                    delta_gamma = gamma_arr - gamma_initial
                    delta_e_mev = delta_gamma * rest_energy_mev
                    percent_delta_e = (delta_gamma / gamma_initial) * 100.0
                else:
                    delta_e_mev = np.zeros_like(z)
                    percent_delta_e = np.zeros_like(z)

                axes[0, 0].plot(t, z, "b-", linewidth=1.5)
                axes[0, 0].set_xlabel("Time (ns)", fontsize=10)
                axes[0, 0].set_ylabel("z (mm)", fontsize=10)
                axes[0, 0].set_title(
                    "Longitudinal Position", fontsize=11, fontweight="bold"
                )
                axes[0, 0].grid(True, alpha=0.3)

                axes[0, 1].plot(z, r * 1e3, "r-", linewidth=1.5)
                axes[0, 1].set_xlabel("z (mm)", fontsize=10)
                axes[0, 1].set_ylabel("r (μm)", fontsize=10)
                axes[0, 1].set_title(
                    "Transverse Position (Radial)", fontsize=11, fontweight="bold"
                )
                axes[0, 1].grid(True, alpha=0.3)

                if len(gamma_arr) > 0:
                    axes[1, 0].plot(z, gamma_arr, "g-", linewidth=1.5)
                    axes[1, 0].set_xlabel("z (mm)", fontsize=10)
                    axes[1, 0].set_ylabel("γ", fontsize=10)
                    axes[1, 0].set_title(
                        "Lorentz Factor", fontsize=11, fontweight="bold"
                    )
                    axes[1, 0].grid(True, alpha=0.3)
                    gamma_mean = np.mean(gamma_arr)
                    gamma_range = np.max(gamma_arr) - np.min(gamma_arr)
                    if gamma_range > 0:
                        margin = max(gamma_range * 0.1, gamma_mean * 0.001)
                        axes[1, 0].set_ylim(
                            [np.min(gamma_arr) - margin, np.max(gamma_arr) + margin]
                        )

                axes[1, 1].plot(z, delta_e_mev, "orange", linewidth=1.5)
                axes[1, 1].set_xlabel("z (mm)", fontsize=10)
                axes[1, 1].set_ylabel("ΔE (MeV)", fontsize=10)
                axes[1, 1].set_title("Energy Change", fontsize=11, fontweight="bold")
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].axhline(
                    y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5
                )

                axes[2, 0].plot(z, percent_delta_e, "purple", linewidth=1.5)
                axes[2, 0].set_xlabel("z (mm)", fontsize=10)
                axes[2, 0].set_ylabel("ΔE/E (%)", fontsize=10)
                axes[2, 0].set_title(
                    "Percent Energy Change", fontsize=11, fontweight="bold"
                )
                axes[2, 0].grid(True, alpha=0.3)
                axes[2, 0].axhline(
                    y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5
                )

                if len(pr) > 0:
                    axes[2, 1].plot(z, pr, "m-", linewidth=1.5)
                    axes[2, 1].set_xlabel("z (mm)", fontsize=10)
                    axes[2, 1].set_ylabel("pr (amu·mm/ns)", fontsize=10)
                    axes[2, 1].set_title(
                        "Transverse Momentum (Radial)", fontsize=11, fontweight="bold"
                    )
                    axes[2, 1].grid(True, alpha=0.3)

                rank_str = f"Rank #{rank}" if rank > 1 else "Best"
                delta_e_final = delta_e_mev[-1] if len(delta_e_mev) > 0 else 0
                percent_final = percent_delta_e[-1] if len(percent_delta_e) > 0 else 0
                title = f"{rank_str} Trajectory (fitness={fitness:.6e})\n"
                title += f"ΔE={delta_e_final:.6f} MeV, ΔE/E={percent_final:.6f}%"
                plt.suptitle(title, fontsize=12, fontweight="bold")
                plt.tight_layout()

                if rank == 1:
                    traj_plot = output_dir / "trajectory_rank1_best.png"
                    traj_data = output_dir / "trajectory_rank1_best.npz"
                else:
                    traj_plot = output_dir / f"trajectory_rank{rank}.png"
                    traj_data = output_dir / f"trajectory_rank{rank}.npz"

                plt.savefig(traj_plot, dpi=150, bbox_inches="tight")
                plt.close(fig)

                self._log_result(
                    f"  Rank #{rank} trajectory plot saved to: {traj_plot}"
                )

                np.savez(traj_data, **traj)
                self._log_result(
                    f"  Rank #{rank} trajectory data saved to: {traj_data}"
                )

            else:
                self._log_result(
                    f"[WARNING] Could not generate rank #{rank} trajectory (integration failed)"
                )
                return None

            return result_data.get("trajectory")

        except Exception as e:  # pragma: no cover - plotting path
            import traceback

            self._log_result(f"[WARNING] Failed to save trajectory: {e}")
            self._log_result(f"[WARNING] Traceback: {traceback.format_exc()}")
            return None

    def _generate_trajectory_comparison_plot(self, trajectory_data_list):
        """Generate comparison plot for top trajectories via helper."""
        return generate_trajectory_comparison_plot(self, trajectory_data_list)

    def _save_partial_optimization_results(
        self, all_evaluations, param_names, status="PARTIAL"
    ):
        """Save partial optimization results when cancelled or failed."""
        import csv
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.config.mode == "optimization":
            method = self.config.optimization_method
            output_dir = (
                Path(self.config.output_dir)
                / "optimizations"
                / f"{timestamp}_{method}_{status}"
            )
        else:
            output_dir = Path(self.config.output_dir) / f"{timestamp}_{status}"

        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "all_evaluations.csv"
        successful_evals = [
            e
            for e in all_evaluations
            if not e.get("failed", False) and not e.get("halted_early", False)
        ]
        halted_evals = [e for e in all_evaluations if e.get("halted_early", False)]

        if len(all_evaluations) > 0:
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
            self._log_result(f"[OK] Partial results saved to: {csv_path}")

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
            "objective": self.config.objective,
        }

        if len(successful_evals) > 0:
            maximize = "max" in self.config.objective.lower()
            finite_evals = [
                e
                for e in successful_evals
                if np.isfinite(e.get("objective_value", np.inf))
            ]

            if len(finite_evals) > 0:
                if maximize:
                    best = max(
                        finite_evals,
                        key=lambda x: x.get("objective_value", -float("inf")),
                    )
                else:
                    best = min(
                        finite_evals,
                        key=lambda x: x.get("objective_value", float("inf")),
                    )
                summary["best_parameters"] = best["parameters"]
                summary["best_value"] = best["objective_value"]
            else:
                summary["note"] = "No finite objective values found"

        summary_path = output_dir / "partial_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self._log_result(f"[OK] Summary saved to: {summary_path}")

        if self._log_file_path is not None and self._log_file_path.exists():
            import shutil

            dest_log = output_dir / self._log_file_path.name
            shutil.copy2(self._log_file_path, dest_log)
            self._log_result(f"[OK] Log file saved to: {dest_log}")

    def _save_sweep_results(
        self, results: List[Dict[str, Any]], failed_runs: List[Dict[str, Any]] = None
    ) -> None:
        """Save sweep results to JSON file with timestamp."""
        from datetime import datetime

        if self._log_file is not None:
            self._close_log_file()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        config_name = "sweep"
        if hasattr(self, "last_loaded_config") and self.last_loaded_config:
            config_name = Path(self.last_loaded_config).stem

        sweep_dir = Path(self.sweep_output_dir) / f"{timestamp}_{config_name}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        output_file = sweep_dir / "sweep_results.json"

        output_data = {
            "sweep_name": f"Parameter Sweep {timestamp}",
            "timestamp": timestamp,
            "config": {
                "aperture_range": self.config.aperture_range,
                "aperture_points": self.config.aperture_points,
                "energy_range": self.config.energy_range,
                "energy_points": self.config.energy_points,
                "transverse_offset_fractions": self.config.transverse_offset_fractions,
                "starting_z_positions": self.config.starting_z_positions,
                "simulation_type": self.config.simulation_type.name,
                "wall_z": self.config.wall_z,
                "wall_z_range": self.config.wall_z_range,
                "wall_z_points": self.config.wall_z_points,
                "auto_steps": self.config.auto_steps,
            },
            "results": results,
            "total_runs": len(results),
        }

        if failed_runs:
            output_data["failed_runs"] = failed_runs
            output_data["num_failed"] = len(failed_runs)

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        self._log_result(f"Results saved to: {output_file}")
        if failed_runs:
            self._log_result(f"  (includes {len(failed_runs)} failed/timed-out runs)")

        self._open_log_file(sweep_dir)

        if len(results) > 0:
            self._generate_summary_plots(results, sweep_dir)

        # Move to archive/incomplete if below minimum run threshold
        from optimization.result_io import relocate_incomplete_sweep

        relocated = relocate_incomplete_sweep(
            sweep_dir,
            min_runs=100,
            log_fn=self._log_result,
        )
        if relocated:
            sweep_dir = relocated

    def _generate_summary_plots(
        self, results: List[Dict[str, Any]], output_dir: Path
    ) -> None:
        """Generate summary plots for the sweep results."""
        try:
            import subprocess
            import sys

            # Count how many parameters were actually swept
            # (have more than 1 unique value across all results)
            # Collect all unique parameter values across all results
            all_param_values = {}
            for result in results:
                params = result.get("parameters", {})
                for key, value in params.items():
                    # Skip non-numeric parameters and internal bookkeeping
                    if key in [
                        "simulation_type",
                        "run_number",
                        "timestep",
                        "steps",
                        "retry_attempts",
                    ]:
                        continue
                    if key not in all_param_values:
                        all_param_values[key] = []
                    all_param_values[key].append(value)

            # Count parameters with more than one unique value
            num_swept_params = 0
            for param_name, values in all_param_values.items():
                unique_values = set(v for v in values if v is not None)
                if len(unique_values) > 1:
                    num_swept_params += 1

            # Only generate heatmap if exactly 2 parameters were swept
            if num_swept_params == 2:
                # Use the new smooth heatmap generator script
                script_path = Path(__file__).parent.parent / "generate_sweep_heatmap.py"

                try:
                    # Call the script with --gain-filter all to show both positive and negative gains
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(script_path),
                            str(output_dir),
                            "--gain-filter",
                            "all",
                            "--output",
                            "sweep_heatmap.png",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,  # 2 minute timeout
                    )

                    if result.returncode == 0:
                        self._log_result(
                            f"[OK] Heatmap saved to: {output_dir / 'sweep_heatmap.png'}"
                        )
                    else:
                        self._log_result(
                            f"[WARNING] Heatmap generation failed: {result.stderr}"
                        )

                except subprocess.TimeoutExpired:
                    self._log_result("[WARNING] Heatmap generation timed out")
                except Exception as e:
                    self._log_result(f"[WARNING] Failed to generate heatmap: {e}")
            else:
                self._log_result(
                    f"[INFO] Skipping heatmap generation ({num_swept_params} parameters swept; heatmap only generated for 2-parameter sweeps)"
                )

            results_with_traj = [
                r
                for r in results
                if "trajectory" in r and len(r.get("trajectory", {}).get("z", [])) > 0
            ]

            if results_with_traj:
                best_result = max(
                    results_with_traj,
                    key=lambda r: r.get("metrics", {}).get("rider_delta_e_mev", -1e9),
                )
                self._plot_single_trajectory(
                    best_result, output_dir / "sweep_best_trajectory.png"
                )
            else:
                self._log_result(
                    "[INFO] No trajectories available for trajectory plot (enable 'Save trajectories' to generate)"
                )

        except Exception as e:  # pragma: no cover - plotting path
            self._log_result(f"[WARNING] Failed to generate summary plots: {e}")

    def _generate_smooth_sweep_heatmap(
        self,
        energies: List[float],
        apertures: List[float],
        percent_gains: List[float],
        output_dir: Path,
    ) -> None:
        """Generate smooth interpolated heatmap for sweep results."""
        import matplotlib.pyplot as plt

        # Filter out points without valid gains
        valid_data = [
            (e, a, g) for e, a, g in zip(energies, apertures, percent_gains) if g != 0
        ]

        if len(valid_data) < 10:
            self._log_result(
                "[INFO] Not enough valid data points for smooth heatmap (need at least 10)"
            )
            return

        energies_filt, apertures_filt, gains_filt = zip(*valid_data)

        self._log_result(
            f"Creating smooth heatmap with {len(gains_filt)} data points..."
        )

        # Determine if log scale is appropriate for energy
        log_energy = (
            max(energies_filt) / min(energies_filt) > 10
            if min(energies_filt) > 0
            else False
        )

        # Work in log space for energy if appropriate
        if log_energy:
            x_data = np.log10(energies_filt)
        else:
            x_data = np.array(energies_filt)

        y_data = np.array(apertures_filt)

        # Create fine grid
        grid_resolution = 800
        x_grid_1d = np.linspace(min(x_data), max(x_data), grid_resolution)
        y_grid_1d = np.linspace(min(y_data), max(y_data), grid_resolution)
        X_grid, Y_grid = np.meshgrid(x_grid_1d, y_grid_1d)

        # Interpolate gains onto grid
        points = np.array([x_data, y_data]).T
        values = np.array(gains_filt)

        # Cubic interpolation
        gain_grid = griddata(points, values, (X_grid, Y_grid), method="cubic")

        # Fill NaN values with nearest neighbor
        nan_mask = np.isnan(gain_grid)
        if nan_mask.any():
            gain_grid_nearest = griddata(
                points, values, (X_grid, Y_grid), method="nearest"
            )
            gain_grid[nan_mask] = gain_grid_nearest[nan_mask]

        # Build KDTree for density checking
        x_range = max(x_data) - min(x_data)
        y_range = max(y_data) - min(y_data)

        grid_points = np.array([X_grid.flatten(), Y_grid.flatten()]).T
        grid_points_normalized = grid_points.copy()
        grid_points_normalized[:, 0] = (grid_points[:, 0] - min(x_data)) / x_range
        grid_points_normalized[:, 1] = (grid_points[:, 1] - min(y_data)) / y_range

        points_normalized = points.copy()
        points_normalized[:, 0] = (points[:, 0] - min(x_data)) / x_range
        points_normalized[:, 1] = (points[:, 1] - min(y_data)) / y_range

        tree_normalized = KDTree(points_normalized)

        # Calculate distances and neighbor counts
        distances, _ = tree_normalized.query(grid_points_normalized)
        distances_2d = distances.reshape(X_grid.shape)

        neighbor_radius = 0.12
        neighbor_counts = tree_normalized.query_ball_point(
            grid_points_normalized, neighbor_radius, return_length=True
        )
        neighbor_counts_2d = neighbor_counts.reshape(X_grid.shape)

        # Create smooth alpha channel
        max_distance = 0.10
        alpha_dist = 1.0 - (distances_2d / max_distance)
        alpha_dist = np.clip(alpha_dist, 0, 1)

        min_neighbors = 2
        alpha_neighbors = neighbor_counts_2d / (min_neighbors * 2.0)
        alpha_neighbors = np.clip(alpha_neighbors, 0, 1)

        alpha = np.maximum(alpha_dist, alpha_neighbors)

        # Apply multiple blur passes for ultra-smooth edges
        for _ in range(5):
            alpha = gaussian_filter(alpha, sigma=4.0)

        # Smooth gain data
        gain_grid_smooth = gaussian_filter(gain_grid, sigma=3.0)

        # Apply alpha mask
        alpha_threshold = 0.02
        gain_grid_final = np.ma.masked_where(alpha < alpha_threshold, gain_grid_smooth)

        # Handle negative/zero values for log scale
        has_positive = np.any(np.array(gains_filt) > 0)
        has_negative = np.any(np.array(gains_filt) < 0)

        if has_positive and not has_negative:
            gain_grid_final = np.ma.masked_where(gain_grid_final <= 0, gain_grid_final)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Determine color scale
        if has_positive and not has_negative:
            vmin = max(np.min(gains_filt), 0.001)
            vmax = np.max(gains_filt)
            norm = LogNorm(vmin=vmin, vmax=vmax)
            color_label = "Energy Gain (%)"
        else:
            vmin = np.min(gains_filt)
            vmax = np.max(gains_filt)
            norm = None
            color_label = "Energy Gain (%)"

        # Convert back to linear energy for plotting if needed
        if log_energy:
            X_plot = 10**X_grid
        else:
            X_plot = X_grid

        # Plot with pcolormesh for smooth continuous colorbar
        im = ax.pcolormesh(
            X_plot,
            Y_grid,
            gain_grid_final,
            cmap="viridis",
            norm=norm,
            shading="gouraud",
            edgecolors="none",
            linewidth=0,
        )
        cbar = plt.colorbar(im, ax=ax, label=color_label)

        # Create contour levels
        if has_positive and not has_negative:
            contour_threshold = 1.0
            low_levels = np.logspace(np.log10(vmin), np.log10(contour_threshold), 4)
            high_levels = np.logspace(np.log10(contour_threshold), np.log10(vmax), 7)
            high_levels = high_levels[high_levels <= vmax]
            contour_levels = np.sort(
                np.unique(np.concatenate([low_levels, high_levels]))
            )
        else:
            contour_levels = np.linspace(vmin, vmax, 11)

        # Draw contours
        contours = ax.contour(
            X_plot,
            Y_grid,
            gain_grid_final,
            levels=contour_levels,
            colors="white",
            alpha=0.35,
            linewidths=0.5,
        )

        # Add contour labels with subtle outline
        labels = ax.clabel(
            contours, inline=True, fontsize=8, fmt="%.2f%%", inline_spacing=10
        )

        for label in labels:
            label.set_path_effects(
                [
                    PathEffects.withStroke(
                        linewidth=1.2, foreground="black", alpha=0.3
                    ),
                    PathEffects.Normal(),
                ]
            )
            label.set_color("#CCCCCC")

        # Set scales and labels
        if log_energy:
            ax.set_xscale("log")

        ax.set_xlabel("Initial Energy (GeV)", fontsize=12)
        ax.set_ylabel("Aperture Radius (mm)", fontsize=12)
        ax.set_title(
            "Parameter Space Exploration: Energy Gain", fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.2, which="both")

        plt.tight_layout()

        heatmap_file = output_dir / "sweep_heatmap.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self._log_result(f"[OK] Smooth heatmap saved to: {heatmap_file}")

    def _plot_single_trajectory(
        self, result: Dict[str, Any], output_file: Path
    ) -> None:
        """Plot trajectory for a single run."""
        try:
            import matplotlib.pyplot as plt

            traj = result.get("trajectory", {})
            params = result.get("parameters", {})
            metrics = result.get("metrics", {})

            z = np.array(traj.get("z", []))
            r = np.array(traj.get("r", []))

            if len(z) == 0:
                return

            aperture = params.get("aperture_radius", 0)
            energy = params.get("particle_energy_gev", 0)
            delta_e = metrics.get("rider_delta_e_mev", 0)
            gamma_initial = metrics.get("rider_gamma_initial", 1)
            gamma_final = metrics.get("rider_gamma_final", 1)

            # KE = (γ - 1) · mc² — use actual particle rest energy
            _rest_mev = (
                getattr(self.config, "m_particle", 0.00054857990907) * AMU_TO_MEV
            )
            energy_mev_initial = (gamma_initial - 1) * _rest_mev
            energy_mev_final = (gamma_final - 1) * _rest_mev

            if len(z) > 1 and abs(z[-1] - z[0]) > 1e-6:
                energy_mev = energy_mev_initial + delta_e * (z - z[0]) / (z[-1] - z[0])
            else:
                energy_mev = np.full_like(z, energy_mev_initial)

            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 1, hspace=0.3)

            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])

            fig.suptitle(
                f"Best Trajectory: a={aperture * 1e3:.1f}μm, E={energy:.1f}GeV, ΔE={delta_e:.6f}MeV",
                fontsize=12,
                fontweight="bold",
            )

            ax1.plot(z, energy_mev - energy_mev_initial, "b-", linewidth=2)
            ax1.set_xlabel("z position (mm)", fontsize=10)
            ax1.set_ylabel("ΔE (MeV)", fontsize=10)
            ax1.set_title("Energy Gain vs Position", fontsize=11, fontweight="bold")
            ax1.grid(True, alpha=0.3)

            ax2.plot(z, r, "r-", linewidth=2, label="+r")
            ax2.plot(z, -r, "r--", linewidth=1.5, alpha=0.6, label="-r")
            ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
            ax2.set_xlabel("z position (mm)", fontsize=10)
            ax2.set_ylabel("Transverse position (mm)", fontsize=10)
            ax2.set_title(
                "Transverse Position (±r) vs z", fontsize=11, fontweight="bold"
            )
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)

            ax3.plot(z, r, "g-", linewidth=2)
            ax3.set_xlabel("z position (mm)", fontsize=10)
            ax3.set_ylabel("r (mm)", fontsize=10)
            ax3.set_title("Radial Position Evolution", fontsize=11, fontweight="bold")
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

            self._log_result(f"[OK] Best trajectory plot saved to: {output_file}")

        except Exception as e:  # pragma: no cover - plotting path
            self._log_result(f"[WARNING] Failed to plot trajectory: {e}")

    def _export_evaluations_csv(self, all_evaluations, param_names, output_dir):
        """Export all evaluations to CSV file."""
        import csv

        try:
            output_path = Path(output_dir)
            csv_file = output_path / "all_evaluations.csv"

            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                metric_names = set()
                for eval_rec in all_evaluations:
                    if not eval_rec.get("failed", True) and "metrics" in eval_rec:
                        metric_names.update(eval_rec["metrics"].keys())

                metric_names = sorted(metric_names)

                header = (
                    ["evaluation", "failed", "halted_early", "halt_reason"]
                    + param_names
                    + metric_names
                    + ["objective_value", "fitness"]
                )
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()

                for eval_rec in all_evaluations:
                    row = {
                        "evaluation": eval_rec["evaluation"],
                        "failed": eval_rec.get("failed", True),
                        "halted_early": eval_rec.get("halted_early", False),
                        "halt_reason": eval_rec.get("halt_reason", ""),
                        "objective_value": eval_rec.get("objective_value", ""),
                        "fitness": eval_rec.get("fitness", ""),
                    }

                    for param_name in param_names:
                        row[param_name] = eval_rec.get("parameters", {}).get(
                            param_name, ""
                        )

                    if not eval_rec.get("failed", True) and "metrics" in eval_rec:
                        for metric_name in metric_names:
                            row[metric_name] = eval_rec["metrics"].get(metric_name, "")

                    writer.writerow(row)

            self._log_result(f"Evaluation CSV exported to: {csv_file}")

        except Exception as e:  # pragma: no cover - file path
            self._log_result(f"[WARNING] Failed to export evaluations CSV: {e}")

    def _view_npz_trajectories(self, results_dir):
        """View NPZ trajectory files from an optimization run."""
        import glob
        import os

        try:
            results_path = Path(results_dir)

            npz_pattern = str(results_path / "trajectory_rank*.npz")
            npz_files = sorted(glob.glob(npz_pattern))

            if not npz_files:
                npz_pattern = str(results_path / "evaluation_*_trajectory.npz")
                npz_files = sorted(glob.glob(npz_pattern))

            if not npz_files:
                _show_info_dialog(
                    self,
                    "No Trajectories Found",
                    f"No NPZ trajectory files found in:\n{results_dir}\n\n"
                    "Expected files like:\n"
                    "- trajectory_rank1_best.npz\n"
                    "- trajectory_rank2.npz\n"
                    "- evaluation_0001_trajectory.npz",
                )
                return

            dialog = tk.Toplevel(self)
            dialog.title(f"NPZ Trajectories: {results_path.name}")
            dialog.geometry("600x500")
            dialog.transient(self)

            ttk.Label(
                dialog,
                text=f"Found {len(npz_files)} trajectory files",
                font=("TkDefaultFont", 10, "bold"),
            ).pack(pady=(10, 5))

            list_frame = ttk.Frame(dialog)
            list_frame.pack(fill="both", expand=True, padx=10, pady=5)

            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side="right", fill="y")

            listbox = tk.Listbox(
                list_frame,
                selectmode="extended",
                yscrollcommand=scrollbar.set,
                height=15,
            )
            listbox.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=listbox.yview)

            for npz_file in npz_files:
                filename = os.path.basename(npz_file)
                listbox.insert("end", filename)

            if npz_files:
                listbox.selection_set(0)

            btn_frame = ttk.Frame(dialog)
            btn_frame.pack(pady=10)

            def plot_selected():
                selection = listbox.curselection()
                if not selection:
                    _show_info_dialog(
                        dialog,
                        "No Selection",
                        "Please select one or more trajectories to plot.",
                    )
                    return

                selected_files = [npz_files[i] for i in selection]
                self._plot_npz_trajectories(selected_files, results_path)

            ttk.Button(
                btn_frame,
                text="Plot Selected",
                command=plot_selected,
                style="Accent.TButton",
            ).pack(side="left", padx=5)

            ttk.Button(
                btn_frame,
                text="Close",
                command=dialog.destroy,
            ).pack(side="left", padx=5)

        except Exception as e:  # pragma: no cover - UI path
            import traceback

            _show_error_dialog(
                self,
                "Error Viewing NPZ Trajectories",
                f"Failed to view NPZ trajectories:\n{e}\n\n{traceback.format_exc()}",
            )

    def _plot_npz_trajectories(self, npz_files, results_dir):
        """Plot NPZ trajectory files."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )

            fig = plt.figure(figsize=(14, 10))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

            ax_r = fig.add_subplot(gs[0, 0])
            ax_pz = fig.add_subplot(gs[0, 1])
            ax_pr = fig.add_subplot(gs[1, 0])
            ax_gamma = fig.add_subplot(gs[1, 1])
            ax_energy = fig.add_subplot(gs[2, :])

            colors = plt.cm.tab10(np.linspace(0, 1, len(npz_files)))

            for idx, npz_file in enumerate(npz_files):
                data = np.load(npz_file)
                z = data["z"]
                r = data["r"]
                pz = data["pz"]
                pr = data["pr"]
                gamma = data["gamma"]

                label = Path(npz_file).stem.replace("trajectory_", "").replace("_", " ")

                ax_r.plot(z, r * 1e3, label=label, color=colors[idx], alpha=0.7)
                ax_pz.plot(z, pz, color=colors[idx], alpha=0.7)
                ax_pr.plot(z, pr, color=colors[idx], alpha=0.7)
                ax_gamma.plot(z, gamma, color=colors[idx], alpha=0.7)

                # KE = (γ - 1) · mc²
                _rest_mev = (
                    getattr(self.config, "m_particle", 0.00054857990907) * AMU_TO_MEV
                )
                energy_mev = (gamma - 1) * _rest_mev
                ax_energy.plot(z, energy_mev, color=colors[idx], alpha=0.7, label=label)

            ax_r.set_xlabel("z (mm)")
            ax_r.set_ylabel("r (μm)")
            ax_r.set_title("Transverse Position")
            ax_r.grid(True, alpha=0.3)
            ax_r.legend()

            ax_pz.set_xlabel("z (mm)")
            ax_pz.set_ylabel("Pz")
            ax_pz.set_title("Longitudinal Momentum")
            ax_pz.grid(True, alpha=0.3)

            ax_pr.set_xlabel("z (mm)")
            ax_pr.set_ylabel("Pr")
            ax_pr.set_title("Transverse Momentum")
            ax_pr.grid(True, alpha=0.3)

            ax_gamma.set_xlabel("z (mm)")
            ax_gamma.set_ylabel("γ")
            ax_gamma.set_title("Lorentz Factor")
            ax_gamma.grid(True, alpha=0.3)

            ax_energy.set_xlabel("z (mm)")
            ax_energy.set_ylabel("Energy (MeV)")
            ax_energy.set_title("Particle Energy")
            ax_energy.grid(True, alpha=0.3)
            ax_energy.legend()

            fig.suptitle(
                f"Optimization Trajectories: {results_dir.name}",
                fontsize=14,
                fontweight="bold",
            )

            plt.tight_layout()

            plot_window = tk.Toplevel(self)
            plot_window.title(f"NPZ Trajectories: {results_dir.name}")
            plot_window.geometry("1200x900")

            main_frame = ttk.Frame(plot_window)
            main_frame.pack(fill="both", expand=True)

            canvas = FigureCanvasTkAgg(fig, master=main_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

            toolbar_frame = ttk.Frame(main_frame)
            toolbar_frame.pack(side="top", fill="x")
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()

            button_frame = ttk.Frame(main_frame, padding=5)
            button_frame.pack(side="top", fill="x")
            ttk.Button(
                button_frame,
                text="Close",
                command=plot_window.destroy,
            ).pack(side="right", padx=5)

        except Exception as e:  # pragma: no cover - UI path
            import traceback

            _show_error_dialog(
                self,
                "Plotting Error",
                f"Failed to plot NPZ trajectories:\n{e}\n\n{traceback.format_exc()}",
            )

    def _save_evaluation_trajectory(self, eval_num, trajectory_data, output_dir):
        """Save a single evaluation trajectory to NPZ file."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            trajectory_file = output_path / f"evaluation_{eval_num:04d}_trajectory.npz"

            np.savez(
                trajectory_file,
                z=np.array(trajectory_data["z"]),
                r=np.array(trajectory_data["r"]),
                pz=np.array(trajectory_data["pz"]),
                pr=np.array(trajectory_data["pr"]),
                t=np.array(trajectory_data["t"]),
                gamma=np.array(trajectory_data["gamma"]),
            )

            return str(trajectory_file)
        except Exception as e:  # pragma: no cover - file path
            self._log_result(
                f"  [WARNING] Failed to save evaluation {eval_num} trajectory: {e}"
            )
            return None
