"""Results-view helpers for the optimization plugin."""

from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from optimization.plugin_results_helpers import (
    UNKNOWN_RESULTS_FORMAT_MESSAGE,
    build_summary_heatmap_grid,
    build_trajectory_plot_data,
    collect_summary_plot_data,
    parse_results_payload,
    summarize_result_row,
)
from optimization.sweep_helpers import AMU_TO_MEV
from optimization.ui_helpers import (
    show_error_dialog as _show_error_dialog,
)
from optimization.ui_helpers import (
    show_info_dialog as _show_info_dialog,
)


class OptimizationPluginViewMixin:
    """Display saved optimization and sweep results inside the GUI."""

    def _on_view_results(self):
        """Display pre-generated summary plots from the latest run."""
        import glob

        default_results_dir = self.sweep_output_dir

        if os.path.exists(default_results_dir):
            result_dirs = [
                d
                for d in glob.glob(os.path.join(default_results_dir, "*"))
                if os.path.isdir(d)
            ]
        else:
            result_dirs = []

        if result_dirs:
            result_dirs.sort(key=os.path.getmtime, reverse=True)
            latest_dir = result_dirs[0]
            png_files = sorted(glob.glob(os.path.join(latest_dir, "*.png")))

            if png_files:
                self._display_summary_plots(latest_dir, png_files)
            else:
                response = messagebox.askyesno(
                    "No Plots Found",
                    f"No summary plots found in:\n{os.path.basename(latest_dir)}\n\n"
                    "Would you like to browse for a different results directory?",
                    parent=self,
                )
                if response:
                    dir_path = filedialog.askdirectory(
                        title="Select Results Directory",
                        initialdir=default_results_dir,
                    )
                    if dir_path:
                        png_files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
                        if png_files:
                            self._display_summary_plots(dir_path, png_files)
                        else:
                            _show_info_dialog(
                                self,
                                "No Plots Found",
                                f"No PNG plot files found in:\n{dir_path}",
                            )
        else:
            response = messagebox.askyesno(
                "No Results Found",
                "No result directories found in the default location.\n\n"
                f"Default location: {default_results_dir}\n\n"
                "Would you like to browse for a results directory?",
                parent=self,
            )
            if response:
                dir_path = filedialog.askdirectory(
                    title="Select Results Directory",
                    initialdir=(
                        default_results_dir if os.path.exists(default_results_dir) else "."
                    ),
                )
                if dir_path:
                    png_files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
                    if png_files:
                        self._display_summary_plots(dir_path, png_files)
                    else:
                        _show_info_dialog(
                            self,
                            "No Plots Found",
                            f"No PNG plot files found in:\n{dir_path}",
                        )

    def _display_summary_plots(self, results_dir, png_files):
        """Display summary plots in a scrollable window."""
        try:
            from PIL import Image, ImageTk
        except ImportError as e:
            _show_error_dialog(
                self,
                "PIL/Pillow Not Installed",
                f"Cannot display images: PIL/Pillow is not installed.\n\n{e}\n\n"
                "Install with: pip install Pillow",
            )
            return

        dir_name = os.path.basename(results_dir)

        self._log_result(f"[INFO] Loading summary plots from: {results_dir}")
        self._log_result(f"[INFO] Found {len(png_files)} PNG files")

        plot_window = tk.Toplevel(self)
        plot_window.title(f"Summary Plots: {dir_name}")
        plot_window.geometry("1000x800")

        main_frame = ttk.Frame(plot_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(
            main_frame,
            text=f"Summary Plots: {dir_name}",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(pady=(0, 10))

        canvas = tk.Canvas(main_frame, bg="white")
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        plot_window.photo_images = []

        for png_file in png_files:
            try:
                self._log_result(f"[INFO] Loading: {Path(png_file).name}")
                img = Image.open(png_file)
                self._log_result(
                    f"[INFO] Image size: {img.width}x{img.height}, mode: {img.mode}"
                )

                max_width = 950
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                    self._log_result(f"[INFO] Resized to: {img.width}x{img.height}")

                photo = ImageTk.PhotoImage(img)
                plot_window.photo_images.append(photo)

                plot_name = Path(png_file).stem.replace("_", " ").title()
                ttk.Label(
                    scrollable_frame,
                    text=plot_name,
                    font=("TkDefaultFont", 10, "bold"),
                ).pack(pady=(10, 5))

                img_label = tk.Label(scrollable_frame, image=photo, bg="white")
                img_label.pack(pady=(0, 20))

                self._log_result(
                    f"[INFO] Successfully displayed: {Path(png_file).name}"
                )

            except Exception as e:
                import traceback

                error_msg = f"Error loading {Path(png_file).name}: {e}"
                self._log_result(f"[ERROR] {error_msg}")
                self._log_result(f"[ERROR] Traceback: {traceback.format_exc()}")

                error_label = ttk.Label(
                    scrollable_frame,
                    text=error_msg,
                    foreground="red",
                )
                error_label.pack(pady=5)

        self._log_result(
            f"[INFO] Finished loading {len(plot_window.photo_images)} images successfully"
        )

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(10, 0))

        ttk.Button(
            button_frame,
            text="Close",
            command=plot_window.destroy,
        ).pack()

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        def on_close():
            canvas.unbind_all("<MouseWheel>")
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_close)

    def _load_and_plot_results(self, file_path: str):
        """Load a results file and display the appropriate viewer."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            parsed = parse_results_payload(
                data,
                m_particle_amu=getattr(self.config, "m_particle", 0.00054857990907),
                amu_to_mev=AMU_TO_MEV,
            )

            if parsed["kind"] == "sweep":
                results = parsed["results"]
                if not results:
                    _show_info_dialog(self, "No Results", "No results found in file.")
                    return

                results_with_traj = parsed["results_with_trajectories"]
                if not results_with_traj:
                    self._show_results_summary(results, file_path)
                    return
            elif parsed["kind"] == "optimization":
                results_dir = os.path.dirname(file_path)
                self._view_npz_trajectories(results_dir)
                return
            else:
                results_with_traj = parsed["results_with_trajectories"]

            self._show_trajectory_viewer(results_with_traj, file_path, auto_plot=True)

        except ValueError as e:
            if str(e) == UNKNOWN_RESULTS_FORMAT_MESSAGE:
                _show_info_dialog(self, "Unknown Format", UNKNOWN_RESULTS_FORMAT_MESSAGE)
                return
            raise
        except Exception as e:
            import traceback

            _show_error_dialog(
                self,
                "Error Loading File",
                f"Failed to load file:\n{e}\n\n{traceback.format_exc()}",
            )

    def _on_plot_trajectories(self):
        """Open trajectory plotting dialog to visualize saved results."""
        import glob

        if os.path.exists(self.sweep_output_dir) and os.listdir(self.sweep_output_dir):
            base_dir = self.sweep_output_dir
        else:
            base_dir = self.config.output_dir

        initial_dir = base_dir
        if os.path.exists(base_dir):
            result_dirs = [
                d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)
            ]
            if result_dirs:
                result_dirs.sort(key=os.path.getmtime, reverse=True)
                initial_dir = result_dirs[0]

        dir_name = os.path.basename(initial_dir) if initial_dir else "results"
        file_path = filedialog.askopenfilename(
            title=f"Select Results File (JSON) - Starting in: {dir_name}",
            initialdir=initial_dir,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not file_path:
            response = messagebox.askyesno(
                "Browse for NPZ Trajectories?",
                "No file selected. Would you like to browse for a directory containing NPZ trajectory files?",
                parent=self,
            )
            if response:
                dir_path = filedialog.askdirectory(
                    title="Select Directory with NPZ Trajectory Files",
                    initialdir=initial_dir,
                )
                if dir_path:
                    self._view_npz_trajectories(dir_path)
            return

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            parsed = parse_results_payload(
                data,
                m_particle_amu=getattr(self.config, "m_particle", 0.00054857990907),
                amu_to_mev=AMU_TO_MEV,
            )

            if parsed["kind"] == "sweep":
                results = parsed["results"]
                if not results:
                    _show_info_dialog(self, "No Results", "No results found in file.")
                    return

                results_with_traj = parsed["results_with_trajectories"]
                if not results_with_traj:
                    _show_info_dialog(
                        self,
                        "No Trajectories",
                        "No trajectory data found in results.\n\n"
                        "Make sure 'Save trajectories' was enabled during the sweep.\n\n"
                        "Note: all_evaluations.csv only contains metrics, not trajectories.\n"
                        "For optimizations, trajectory data is in NPZ files.",
                    )
                    return
            elif parsed["kind"] == "optimization":
                results_dir = os.path.dirname(file_path)
                self._view_npz_trajectories(results_dir)
                return
            else:
                results_with_traj = parsed["results_with_trajectories"]

            self._show_trajectory_viewer(results_with_traj, file_path)

        except ValueError as e:
            if str(e) == UNKNOWN_RESULTS_FORMAT_MESSAGE:
                _show_info_dialog(
                    self,
                    "Unknown Format",
                    f"{UNKNOWN_RESULTS_FORMAT_MESSAGE}\n\n"
                    "Note: CSV files only contain metrics, not trajectory data.",
                )
                return
            raise
        except Exception as e:
            import traceback

            _show_error_dialog(
                self,
                "Error Loading File",
                f"Failed to load file:\n{e}\n\n{traceback.format_exc()}",
            )

    def _show_results_summary(self, results, file_path):
        """Show a metrics-first results summary."""
        dialog = tk.Toplevel(self)
        dialog.title(f"Results Summary - {Path(file_path).name}")
        dialog.geometry("1100x700")
        dialog.transient(self)

        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        ttk.Label(
            main_frame,
            text="Sweep Results Summary",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        num_runs = len(results)
        sweep_info = results[0].get("sweep_info", {}) if results else {}
        config_name = sweep_info.get("config_name", "Unknown")

        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(
            info_frame,
            text=f"Configuration: {config_name}  |  Total Runs: {num_runs}",
            font=("TkDefaultFont", 10),
        ).pack(anchor="w")

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True, pady=(5, 0))

        metrics_frame = ttk.Frame(notebook, padding=10)
        notebook.add(metrics_frame, text="Metrics Table")

        table_container = ttk.Frame(metrics_frame)
        table_container.pack(fill="both", expand=True)

        v_scrollbar = ttk.Scrollbar(table_container)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar = ttk.Scrollbar(table_container, orient="horizontal")
        h_scrollbar.pack(side="bottom", fill="x")

        metrics_text = tk.Text(
            table_container,
            wrap="none",
            font=("Courier", 9),
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set,
        )
        metrics_text.pack(side="left", fill="both", expand=True)
        v_scrollbar.config(command=metrics_text.yview)
        h_scrollbar.config(command=metrics_text.xview)

        if results:
            has_beam_optics = any(
                r.get("metrics", {}).get("rider_emittance_x_mm_mrad") is not None
                for r in results
            )

            if has_beam_optics:
                header = f"{'Run':<5} {'Aperture (mm)':<15} {'Energy (GeV)':<15} {'Start_z (mm)':<15} {'ΔE (MeV)':<12} {'Traveled (mm)':<15} {'γ_initial':<12} {'εx (mm·mrad)':<15} {'εnx (mm·mrad)':<16} {'βx (m)':<12}\n"
                header += "-" * 157 + "\n"
            else:
                header = f"{'Run':<5} {'Aperture (mm)':<15} {'Energy (GeV)':<15} {'Start_z (mm)':<15} {'ΔE (MeV)':<12} {'Traveled (mm)':<15} {'γ_initial':<12}\n"
                header += "-" * 110 + "\n"
            metrics_text.insert("end", header)

            for r in results:
                row_data = summarize_result_row(r)

                if has_beam_optics:
                    row = (
                        f"{row_data['run_num']:<5} {row_data['aperture']:<15.3e} "
                        f"{row_data['energy']:<15.2f} {row_data['start_z']:<15.1f} "
                        f"{row_data['delta_e']:<12.3f} {row_data['traveled']:<15.1f} "
                        f"{row_data['gamma_initial']:<12.1f} "
                        f"{row_data['emit_x']:<15.3e} {row_data['norm_emit_x']:<16.3e} "
                        f"{row_data['beta_x']:<12.3e}\n"
                    )
                else:
                    row = (
                        f"{row_data['run_num']:<5} {row_data['aperture']:<15.3e} "
                        f"{row_data['energy']:<15.2f} {row_data['start_z']:<15.1f} "
                        f"{row_data['delta_e']:<12.3f} {row_data['traveled']:<15.1f} "
                        f"{row_data['gamma_initial']:<12.1f}\n"
                    )
                metrics_text.insert("end", row)

        metrics_text.config(state="disabled")

        plots_frame = ttk.Frame(notebook, padding=10)
        notebook.add(plots_frame, text="Visualization")

        has_trajectories = any("trajectory" in r for r in results)

        if has_trajectories:
            ttk.Label(
                plots_frame,
                text="Trajectory data available. Click below to view trajectory plots.",
                font=("TkDefaultFont", 10),
            ).pack(pady=20)

            ttk.Button(
                plots_frame,
                text="Open Trajectory Viewer",
                command=lambda: self._open_trajectory_viewer_from_summary(
                    dialog, results, file_path
                ),
                style="Accent.TButton",
            ).pack(pady=10)
        else:
            self._create_summary_plots(plots_frame, results)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(
            btn_frame,
            text="Export to CSV",
            command=lambda: self._export_metrics_csv(results, file_path),
        ).pack(side="left", padx=5)

        ttk.Button(
            btn_frame,
            text="Close",
            command=dialog.destroy,
        ).pack(side="right", padx=5)

    def _create_summary_plots(self, parent_frame, results):
        """Create parameter sweep visualization plots."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )

            plot_data = collect_summary_plot_data(results)
            apertures = plot_data["apertures"]
            energies = plot_data["energies"]
            delta_es = plot_data["delta_es"]

            fig = plt.figure(figsize=(10, 6))

            unique_apertures = len(set(apertures))
            unique_energies = len(set(energies))

            if unique_apertures > 1 and unique_energies > 1:
                ax = fig.add_subplot(111)
                unique_a, unique_e, grid = build_summary_heatmap_grid(results)

                im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn_r")
                ax.set_xticks(range(len(unique_a)))
                ax.set_xticklabels(
                    [f"{a:.1e}" for a in unique_a], rotation=45, ha="right"
                )
                ax.set_yticks(range(len(unique_e)))
                ax.set_yticklabels([f"{e:.1f}" for e in unique_e])
                ax.set_xlabel("Aperture Radius (mm)")
                ax.set_ylabel("Particle Energy (GeV)")
                ax.set_title("ΔE Heatmap (MeV)")
                plt.colorbar(im, ax=ax, label="ΔE (MeV)")
            elif unique_apertures > 1:
                ax = fig.add_subplot(111)
                ax.plot(apertures, delta_es, "o-", markersize=8)
                ax.set_xlabel("Aperture Radius (mm)")
                ax.set_ylabel("ΔE (MeV)")
                ax.set_title(f"Energy Change vs Aperture (E={energies[0]:.1f} GeV)")
                ax.grid(True, alpha=0.3)
            elif unique_energies > 1:
                ax = fig.add_subplot(111)
                ax.plot(energies, delta_es, "o-", markersize=8)
                ax.set_xlabel("Particle Energy (GeV)")
                ax.set_ylabel("ΔE (MeV)")
                ax.set_title(f"Energy Change vs Energy (a={apertures[0]:.2e} mm)")
                ax.grid(True, alpha=0.3)
            else:
                ax = fig.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "Single-point simulation\nNo parameter sweep to visualize",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")

            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, parent_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            toolbar = NavigationToolbar2Tk(canvas, parent_frame)
            toolbar.update()

        except Exception as e:
            ttk.Label(
                parent_frame,
                text=f"Could not create plots: {e}",
                foreground="red",
            ).pack(pady=20)

    def _export_metrics_csv(self, results, file_path):
        """Export metrics to CSV file."""
        import csv

        default_name = Path(file_path).stem + "_metrics.csv"
        output_file = filedialog.asksaveasfilename(
            title="Export Metrics to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_name,
            parent=self,
        )

        if not output_file:
            return

        try:
            with open(output_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(
                    [
                        "Run",
                        "Aperture_mm",
                        "Energy_GeV",
                        "Start_z_mm",
                        "Delta_E_MeV",
                        "Traveled_mm",
                        "Gamma_initial",
                        "Gamma_final",
                        "Emittance_x_mm_mrad",
                        "Emittance_y_mm_mrad",
                        "Norm_Emittance_x_mm_mrad",
                        "Norm_Emittance_y_mm_mrad",
                        "Beta_x_m",
                        "Beta_y_m",
                    ]
                )

                for r in results:
                    row = summarize_result_row(r)
                    writer.writerow(
                        [
                            row["run_num"],
                            row["aperture"],
                            row["energy"],
                            row["start_z"],
                            row["delta_e"],
                            row["traveled"],
                            row["gamma_initial"],
                            row["gamma_final"],
                            row["emit_x"],
                            row["emit_y"],
                            row["norm_emit_x"],
                            row["norm_emit_y"],
                            row["beta_x"],
                            row["beta_y"],
                        ]
                    )

            _show_info_dialog(
                self, "Export Successful", f"Metrics exported to:\n{output_file}"
            )

        except Exception as e:
            _show_error_dialog(self, "Export Failed", f"Failed to export CSV:\n{e}")

    def _open_trajectory_viewer_from_summary(self, summary_dialog, results, file_path):
        """Open trajectory viewer from the summary dialog."""
        results_with_traj = [r for r in results if "trajectory" in r]
        if results_with_traj:
            self._show_trajectory_viewer(results_with_traj, file_path, auto_plot=True)
        else:
            _show_info_dialog(
                summary_dialog,
                "No Trajectories",
                "No trajectory data found in results.",
            )

    def _show_trajectory_viewer(self, results, file_path, auto_plot=False):
        """Show trajectory viewer dialog with run selection and plotting."""
        dialog = tk.Toplevel(self)
        dialog.title(f"Trajectory Viewer - {Path(file_path).name}")
        dialog.geometry("1000x700")
        dialog.transient(self)

        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill="both", expand=True)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side="left", fill="both", expand=False, padx=(0, 5))

        ttk.Label(
            left_panel, text="Select Runs to Plot:", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))

        list_frame = ttk.Frame(left_panel)
        list_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        run_listbox = tk.Listbox(
            list_frame,
            selectmode="extended",
            width=40,
            height=20,
            yscrollcommand=scrollbar.set,
        )
        run_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=run_listbox.yview)

        for r in results:
            params = r.get("parameters", {})
            run_num = r.get("run_number", "?")
            aperture = params.get("aperture_radius", 0)
            energy = params.get("particle_energy_gev", 0)
            delta_e = r.get("metrics", {}).get("rider_delta_e_mev", 0)

            summary = (
                f"Run #{run_num}: "
                f"a={aperture:.2e}mm, E={energy:.1f}GeV, "
                f"ΔE={delta_e:.6f}MeV"
            )
            run_listbox.insert("end", summary)

        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=(10, 0))

        plot_button = ttk.Button(
            btn_frame,
            text="Plot Selected",
            command=lambda: self._plot_selected_trajectories(
                run_listbox, results, dialog
            ),
        )
        plot_button.pack(fill="x", pady=2)

        select_all_btn = ttk.Button(
            btn_frame,
            text="Select All",
            command=lambda: run_listbox.select_set(0, "end"),
        )
        select_all_btn.pack(fill="x", pady=2)

        clear_btn = ttk.Button(
            btn_frame,
            text="Clear Selection",
            command=lambda: run_listbox.selection_clear(0, "end"),
        )
        clear_btn.pack(fill="x", pady=2)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)

        ttk.Label(
            right_panel, text="Plot Area", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", pady=(0, 5))

        plot_info = ttk.Label(
            right_panel,
            text="Select runs and click 'Plot Selected' to visualize trajectories.\n\n"
            "Transverse plots will be shown as scatter plots.",
            justify="center",
            foreground="gray",
        )
        plot_info.pack(expand=True)

        dialog.plot_area = right_panel
        dialog.plot_info = plot_info

        if auto_plot:
            max_auto_plot = min(10, len(results))
            for i in range(max_auto_plot):
                run_listbox.select_set(i)

            run_listbox.update_idletasks()
            dialog.update()

            def safe_auto_plot():
                if run_listbox.curselection():
                    self._plot_selected_trajectories(
                        run_listbox, results, dialog, is_auto_plot=True
                    )
                else:
                    for i in range(max_auto_plot):
                        run_listbox.select_set(i)
                    run_listbox.update()
                    dialog.after(
                        100,
                        lambda: self._plot_selected_trajectories(
                            run_listbox, results, dialog, is_auto_plot=True
                        ),
                    )

            dialog.after(200, safe_auto_plot)

    def _plot_selected_trajectories(
        self, listbox, results, parent_dialog, is_auto_plot=False
    ):
        """Plot trajectories for selected runs."""
        listbox.update_idletasks()
        selection = listbox.curselection()
        if not selection:
            if not is_auto_plot and listbox.size() > 0:
                _show_info_dialog(
                    parent_dialog,
                    "No Selection",
                    "Please select at least one run to plot.",
                )
            return

        selected_results = [results[i] for i in selection]

        for widget in parent_dialog.plot_area.winfo_children():
            widget.destroy()

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )

            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 1, hspace=0.3)

            ax_delta_e = fig.add_subplot(gs[0])
            ax_transverse = fig.add_subplot(gs[1])
            ax_heatmap = fig.add_subplot(gs[2])

            fig.suptitle(
                f"Sweep Results: {len(selected_results)} run(s)",
                fontsize=12,
                fontweight="bold",
            )

            plot_data = build_trajectory_plot_data(
                selected_results,
                m_particle_amu=getattr(self.config, "m_particle", 0.00054857990907),
                amu_to_mev=AMU_TO_MEV,
            )
            heatmap = plot_data["heatmap"]

            for idx, series in enumerate(plot_data["series"]):
                label = (
                    f"Run #{series['run_num']} "
                    f"(a={series['aperture']:.2e}mm, E={series['energy']:.1f}GeV)"
                )
                color = plt.cm.tab10(idx % 10)

                ax_delta_e.plot(
                    series["z"],
                    series["energy_delta"],
                    label=label,
                    alpha=0.7,
                    color=color,
                    linewidth=1.5,
                )

                ax_transverse.plot(
                    series["z"],
                    series["r"],
                    label=f"{label} (+r)",
                    alpha=0.6,
                    color=color,
                    linewidth=1.5,
                )
                ax_transverse.plot(
                    series["z"],
                    -series["r"],
                    alpha=0.3,
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                )

            ax_delta_e.set_xlabel("z position (mm)", fontsize=10)
            ax_delta_e.set_ylabel("ΔE (MeV)", fontsize=10)
            ax_delta_e.set_title(
                "Energy Gain vs Position", fontsize=11, fontweight="bold"
            )
            ax_delta_e.legend(fontsize=7, loc="best")
            ax_delta_e.grid(True, alpha=0.3)

            ax_transverse.set_xlabel("z position (mm)", fontsize=10)
            ax_transverse.set_ylabel("Transverse position (mm)", fontsize=10)
            ax_transverse.set_title(
                "Transverse Position (±r) vs z", fontsize=11, fontweight="bold"
            )
            ax_transverse.legend(fontsize=7, loc="best")
            ax_transverse.grid(True, alpha=0.3)
            ax_transverse.axhline(
                y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3
            )

            apertures = heatmap["apertures"]
            energies = heatmap["energies"]
            delta_es = heatmap["delta_es"]
            unique_apertures = len(set(apertures))
            unique_energies = len(set(energies))

            if len(apertures) > 0 and unique_apertures > 1 and unique_energies > 1:
                scatter = ax_heatmap.scatter(
                    energies,
                    [a * 1e3 for a in apertures],
                    c=delta_es,
                    cmap="viridis",
                    s=100,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )

                cbar = plt.colorbar(scatter, ax=ax_heatmap)
                cbar.set_label("ΔE (MeV)", fontsize=10)

                ax_heatmap.set_xlabel("Particle Energy (GeV)", fontsize=10)
                ax_heatmap.set_ylabel("Aperture Radius (μm)", fontsize=10)
                ax_heatmap.set_title(
                    "Parameter Space: ΔE(Energy, Aperture)",
                    fontsize=11,
                    fontweight="bold",
                )
                ax_heatmap.grid(True, alpha=0.3)

                if max(energies) / min(energies) > 10 if min(energies) > 0 else False:
                    ax_heatmap.set_xscale("log")
                if (
                    max(apertures) / min(apertures) > 10
                    if min(apertures) > 0
                    else False
                ):
                    ax_heatmap.set_yscale("log")
            else:
                ax_heatmap.text(
                    0.5,
                    0.5,
                    "Heatmap requires sweep over both\naperture and energy parameters",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="gray",
                    transform=ax_heatmap.transAxes,
                )
                ax_heatmap.set_xticks([])
                ax_heatmap.set_yticks([])
                ax_heatmap.set_title(
                    "Parameter Space Heatmap (N/A)",
                    fontsize=11,
                    fontweight="bold",
                    color="gray",
                )

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=parent_dialog.plot_area)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            toolbar = NavigationToolbar2Tk(canvas, parent_dialog.plot_area)
            toolbar.update()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except ImportError:
            _show_error_dialog(
                self,
                "Missing Dependency",
                "Matplotlib is required for plotting.\n\nInstall with: pip install matplotlib",
            )
        except Exception as e:
            _show_error_dialog(
                self, "Plotting Error", f"Failed to plot trajectories:\n{e}"
            )
