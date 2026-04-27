"""Plot display and replot helpers for the main GUI."""

from __future__ import annotations

from functools import partial
from tkinter import filedialog
from typing import Any, Tuple

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk


class IntegratorGUIPlotMixin:
    """Display generated figures and support lightweight replot controls."""

    def _queue_log(self, text: str) -> None:
        """Queue a UI log line; batched logging remains the preferred path."""
        self.root.after(0, partial(self._append_log, text))

    def _replot_with_new_axis(
        self,
        figure: Any,
        plot_name: str,
        new_xaxis: str,
        canvas: Any,
        new_yaxis: str = None,
    ) -> None:
        """Regenerate plot with new axes using stored plot data."""
        import numpy as np

        if not hasattr(figure, "_lw_plot_data"):
            return

        data = figure._lw_plot_data

        if new_xaxis == "z":
            xdata = data["z_mm"]
            xlabel = "z position (mm)"
        else:
            xdata = data["times_ns"]
            xlabel = "Time (ns)"

        axes = figure.get_axes()

        original_label_sizes = [ax.xaxis.label.get_fontsize() for ax in axes]
        original_title_sizes = [ax.title.get_fontsize() for ax in axes]
        original_xtick_sizes = []
        original_ytick_sizes = []
        for ax in axes:
            xtick_labels = ax.get_xticklabels()
            ytick_labels = ax.get_yticklabels()
            original_xtick_sizes.append(
                xtick_labels[0].get_fontsize() if xtick_labels else 9
            )
            original_ytick_sizes.append(
                ytick_labels[0].get_fontsize() if ytick_labels else 9
            )

        if plot_name == "energy":
            if new_yaxis and data.get("energy_components"):
                if new_yaxis == "delta_total":
                    ydata_r = data["energy_components"]["delta_total_r"]
                    ydata_d = data["energy_components"].get("delta_total_d")
                    ylabel = "ΔE (GeV)"
                    title_suffix = "ΔE"
                elif new_yaxis == "delta_z":
                    ydata_r = data["energy_components"]["delta_z_r"]
                    ydata_d = data["energy_components"].get("delta_z_d")
                    ylabel = "ΔE_z (GeV)"
                    title_suffix = "ΔE_z"
                elif new_yaxis == "delta_x":
                    ydata_r = data["energy_components"]["delta_x_r"]
                    ydata_d = data["energy_components"].get("delta_x_d")
                    ylabel = "ΔE_x (GeV)"
                    title_suffix = "ΔE_x"
                elif new_yaxis == "delta_y":
                    ydata_r = data["energy_components"]["delta_y_r"]
                    ydata_d = data["energy_components"].get("delta_y_d")
                    ylabel = "ΔE_y (GeV)"
                    title_suffix = "ΔE_y"
                elif new_yaxis == "total":
                    ydata_r = data["energy_components"]["total_r"]
                    ydata_d = data["energy_components"].get("total_d")
                    ylabel = "E (GeV)"
                    title_suffix = "E"
                else:
                    ydata_r = data["core_r_energy_changes"]
                    ydata_d = data.get("core_d_energy_changes")
                    ylabel = "ΔE (GeV)"
                    title_suffix = "ΔE"
            else:
                ydata_r = data["core_r_energy_changes"]
                ydata_d = data.get("core_d_energy_changes")
                ylabel = "ΔE (GeV)"
                title_suffix = "ΔE"

            collections = axes[0].collections
            if len(collections) > 0:
                collections[0].set_offsets(np.c_[xdata, ydata_r])

            axes[0].set_xlabel(xlabel, fontsize=10)
            axes[0].set_ylabel(ylabel, fontsize=10)
            axes[0].set_title(
                f"Rider {title_suffix} vs " + ("z" if new_xaxis == "z" else "Time"),
                fontsize=12,
            )
            axes[0].relim()
            axes[0].autoscale_view(tight=True)
            x_min, x_max = xdata.min(), xdata.max()
            x_range = x_max - x_min
            x_buffer = x_range * 0.1 if x_range > 0 else 0.1 * abs(x_max)
            axes[0].set_xlim(x_min - x_buffer, x_max + x_buffer)
            if len(ydata_r) > 0:
                y_min, y_max = ydata_r.min(), ydata_r.max()
                y_range = y_max - y_min
                y_buffer = y_range * 0.1 if y_range > 0 else 0.1 * abs(y_max)
                axes[0].set_ylim(y_min - y_buffer, y_max + y_buffer)

            if len(axes) > 1 and data["driver_allowed"] and ydata_d is not None:
                xdata_d = data["z_mm_driver"] if new_xaxis == "z" else data["times_ns"]
                collections_d = axes[1].collections
                if len(collections_d) > 0:
                    collections_d[0].set_offsets(np.c_[xdata_d, ydata_d])

                axes[1].set_xlabel(xlabel, fontsize=10)
                axes[1].set_ylabel(ylabel, fontsize=10)
                axes[1].set_title(
                    f"Driver {title_suffix} vs "
                    + ("z" if new_xaxis == "z" else "Time"),
                    fontsize=12,
                )
                axes[1].relim()
                axes[1].autoscale_view(tight=True)
                x_min_d, x_max_d = xdata_d.min(), xdata_d.max()
                x_range_d = x_max_d - x_min_d
                x_buffer_d = x_range_d * 0.1 if x_range_d > 0 else 0.1 * abs(x_max_d)
                axes[1].set_xlim(x_min_d - x_buffer_d, x_max_d + x_buffer_d)
                if len(ydata_d) > 0:
                    y_min_d, y_max_d = ydata_d.min(), ydata_d.max()
                    y_range_d = y_max_d - y_min_d
                    y_buffer_d = (
                        y_range_d * 0.1 if y_range_d > 0 else 0.1 * abs(y_max_d)
                    )
                    axes[1].set_ylim(y_min_d - y_buffer_d, y_max_d + y_buffer_d)

        elif plot_name == "transverse":
            ax_x, ax_y = axes[0], axes[1]

            lines_x = ax_x.get_lines()
            if len(lines_x) > 0:
                lines_x[0].set_xdata(xdata)
                idx = 1
                if data["driver_allowed"] and data["core_d_hist"] is not None:
                    xdata_d = (
                        data["z_mm_driver"] if new_xaxis == "z" else data["times_ns"]
                    )
                    if len(lines_x) > idx:
                        lines_x[idx].set_xdata(xdata_d)

            ax_x.set_xlabel(xlabel, fontsize=10)
            ax_x.relim()
            ax_x.autoscale_view(tight=True)

            lines_y = ax_y.get_lines()
            if len(lines_y) > 0:
                lines_y[0].set_xdata(xdata)
                idx = 1
                if data["driver_allowed"] and data["core_d_hist"] is not None:
                    xdata_d = (
                        data["z_mm_driver"] if new_xaxis == "z" else data["times_ns"]
                    )
                    if len(lines_y) > idx:
                        lines_y[idx].set_xdata(xdata_d)

            ax_y.set_xlabel(xlabel, fontsize=10)
            ax_y.relim()
            ax_y.autoscale_view(tight=True)

        elif plot_name == "beta":
            for ax in axes:
                lines = ax.get_lines()
                if len(lines) > 0:
                    lines[0].set_xdata(xdata)
                    idx = 1
                    if data["driver_allowed"] and data["core_d_beta"] is not None:
                        xdata_d = (
                            data["z_mm_driver"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        if len(lines) > idx:
                            lines[idx].set_xdata(xdata_d)

                ax.set_xlabel(xlabel, fontsize=10)
                ax.relim()
                ax.autoscale_view(tight=True)

        elif plot_name == "momentum":
            for ax in axes[:6]:
                lines = ax.get_lines()
                if len(lines) > 0:
                    lines[0].set_xdata(xdata)
                    idx = 1
                    if data["driver_allowed"] and data["core_d_momentum"] is not None:
                        xdata_d = (
                            data["z_mm_driver"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        if len(lines) > idx:
                            lines[idx].set_xdata(xdata_d)

                ax.set_xlabel(xlabel, fontsize=10)
                ax.relim()
                ax.autoscale_view(tight=True)

        elif plot_name == "gamma":
            for i, ax in enumerate(axes):
                collections = ax.collections
                if len(collections) > 0:
                    collections[0].set_offsets(np.c_[xdata, data["core_r_gamma"]])
                    idx = 1
                    if data["driver_allowed"] and data.get("core_d_gamma") is not None:
                        xdata_d = (
                            data["z_mm_driver"]
                            if new_xaxis == "z"
                            else data["times_ns"]
                        )
                        if len(collections) > idx:
                            collections[idx].set_offsets(
                                np.c_[xdata_d, data["core_d_gamma"]]
                            )

                ax.set_xlabel(xlabel, fontsize=10)
                ax.relim()
                ax.autoscale_view(tight=True)

                try:
                    all_gamma = []
                    if len(data["core_r_gamma"]) > 0:
                        all_gamma.extend(data["core_r_gamma"])
                    if (
                        i == 1
                        and data["driver_allowed"]
                        and data.get("core_d_gamma") is not None
                        and len(data["core_d_gamma"]) > 0
                    ):
                        all_gamma.extend(data["core_d_gamma"])

                    if len(all_gamma) > 0:
                        gamma_array = np.array(all_gamma)
                        gamma_min = np.min(gamma_array)
                        gamma_max = np.max(gamma_array)
                        gamma_mean = np.mean(gamma_array)
                        gamma_range = gamma_max - gamma_min
                        relative_variation = (
                            gamma_range / gamma_mean if gamma_mean > 0 else 0
                        )

                        if relative_variation < 0.05 and gamma_range > 0:
                            buffer = (
                                gamma_range * 0.1
                                if gamma_range > 0
                                else gamma_mean * 0.001
                            )
                            ax.set_ylim(gamma_min - buffer, gamma_max + buffer)
                except Exception:
                    pass

        for i, ax in enumerate(axes):
            if i < len(original_label_sizes):
                ax.xaxis.label.set_fontsize(original_label_sizes[i])
                ax.yaxis.label.set_fontsize(original_label_sizes[i])
            if i < len(original_title_sizes):
                ax.title.set_fontsize(original_title_sizes[i])
            if i < len(original_xtick_sizes):
                ax.tick_params(
                    axis="x", which="major", labelsize=original_xtick_sizes[i]
                )
            if i < len(original_ytick_sizes):
                ax.tick_params(
                    axis="y", which="major", labelsize=original_ytick_sizes[i]
                )

        canvas.draw_idle()

    def _show_figure(self, title: str, figure: Any, plot_name: str = "") -> None:
        from .gui import _FigureHandle, _show_error_dialog, _show_warning_dialog

        try:
            width_px, height_px = self._prepare_figure_for_display(figure)
        except Exception as e:
            self._append_log(f"Warning: Could not prepare figure for display: {e}")
            width_px, height_px = 800, 600

        window = tk.Toplevel(self.root)
        window.title(title)

        main_frame = ttk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = FigureCanvasTkAgg(figure, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(main_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        controls_frame = ttk.Frame(main_frame, padding=5)
        controls_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls_frame, text="Log scale:").pack(side=tk.LEFT, padx=(0, 5))

        x_log_var = tk.BooleanVar(value=False)
        y_log_var = tk.BooleanVar(value=False)

        def check_axis_data_for_log(ax, axis="x"):
            import numpy as np

            try:
                if axis == "x":
                    for line in ax.get_lines():
                        xdata = line.get_xdata()
                        if len(xdata) > 0 and np.any(xdata <= 0):
                            return False
                    for coll in ax.collections:
                        offsets = coll.get_offsets()
                        if len(offsets) > 0 and np.any(offsets[:, 0] <= 0):
                            return False
                else:
                    for line in ax.get_lines():
                        ydata = line.get_ydata()
                        if len(ydata) > 0 and np.any(ydata <= 0):
                            return False
                    for coll in ax.collections:
                        offsets = coll.get_offsets()
                        if len(offsets) > 0 and np.any(offsets[:, 1] <= 0):
                            return False
                return True
            except Exception:
                return False

        x_log_suitable = all(
            check_axis_data_for_log(ax, "x") for ax in figure.get_axes()
        )
        y_log_suitable = all(
            check_axis_data_for_log(ax, "y") for ax in figure.get_axes()
        )

        def toggle_log_scale() -> None:
            try:
                from matplotlib.ticker import LogFormatterSciNotation, ScalarFormatter

                for ax in figure.get_axes():
                    if x_log_var.get():
                        if not check_axis_data_for_log(ax, "x"):
                            x_log_var.set(False)
                            _show_warning_dialog(
                                window,
                                "Log Scale Warning",
                                "X-axis cannot be log-scaled: data contains non-positive values",
                            )
                            return
                        try:
                            ax.set_xscale("log")
                            formatter = LogFormatterSciNotation()
                            ax.xaxis.set_major_formatter(formatter)
                            ax.tick_params(axis="x", which="major", labelsize=9)
                            ax.tick_params(axis="x", which="minor", labelsize=8)
                        except Exception as e:
                            x_log_var.set(False)
                            _show_warning_dialog(
                                window,
                                "Log Scale Error",
                                f"Failed to set X-axis log scale: {e}",
                            )
                            return
                    else:
                        ax.set_xscale("linear")
                        formatter = ScalarFormatter()
                        formatter.set_scientific(False)
                        formatter.set_useOffset(False)
                        ax.xaxis.set_major_formatter(formatter)
                        ax.tick_params(axis="x", which="major", labelsize=9)

                    if y_log_var.get():
                        if not check_axis_data_for_log(ax, "y"):
                            y_log_var.set(False)
                            _show_warning_dialog(
                                window,
                                "Log Scale Warning",
                                "Y-axis cannot be log-scaled: data contains non-positive values",
                            )
                            return
                        try:
                            ax.set_yscale("log")
                            formatter = LogFormatterSciNotation()
                            ax.yaxis.set_major_formatter(formatter)
                            ax.tick_params(axis="y", which="major", labelsize=9)
                            ax.tick_params(axis="y", which="minor", labelsize=8)
                        except Exception as e:
                            y_log_var.set(False)
                            _show_warning_dialog(
                                window,
                                "Log Scale Error",
                                f"Failed to set Y-axis log scale: {e}",
                            )
                            return
                    else:
                        ax.set_yscale("linear")
                        formatter = ScalarFormatter()
                        formatter.set_scientific(False)
                        formatter.set_useOffset(False)
                        ax.yaxis.set_major_formatter(formatter)
                        ax.tick_params(axis="y", which="major", labelsize=9)

                    ax.relim()
                    ax.autoscale_view(tight=False)

                canvas.draw_idle()
            except Exception as e:
                self._append_log(f"Error toggling log scale: {e}")

        x_log_check = ttk.Checkbutton(
            controls_frame, text="X-axis", variable=x_log_var, command=toggle_log_scale
        )
        x_log_check.pack(side=tk.LEFT, padx=5)
        if not x_log_suitable:
            x_log_check.configure(state="disabled")

        y_log_check = ttk.Checkbutton(
            controls_frame, text="Y-axis", variable=y_log_var, command=toggle_log_scale
        )
        y_log_check.pack(side=tk.LEFT, padx=5)
        if not y_log_suitable:
            y_log_check.configure(state="disabled")

        ttk.Separator(controls_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        plot_supports_xaxis_switch = plot_name in [
            "beta",
            "momentum",
            "transverse",
            "zposition",
            "energy",
            "gamma",
        ]
        plot_supports_yaxis_switch = plot_name == "energy"

        if plot_supports_xaxis_switch and plot_name != "zposition":
            current_xaxis = "t"
            try:
                first_ax = figure.get_axes()[0]
                xlabel = first_ax.get_xlabel().lower()
                if (
                    "z position" in xlabel
                    or "delta z" in xlabel
                    or "δz" in xlabel
                    or "z (mm)" in xlabel
                ):
                    current_xaxis = "z"
                elif "time" in xlabel:
                    current_xaxis = "t"
            except Exception:
                pass

            xaxis_var = tk.StringVar(value=current_xaxis)

            yaxis_var = None
            if plot_supports_yaxis_switch:
                current_yaxis = getattr(
                    self, "energy_yaxis_var", tk.StringVar(value="delta_total")
                ).get()
                yaxis_var = tk.StringVar(value=current_yaxis)

            def switch_xaxis() -> None:
                new_xaxis = xaxis_var.get()
                new_yaxis = yaxis_var.get() if yaxis_var else None
                try:
                    if not hasattr(figure, "_lw_plot_data"):
                        if plot_name == "beta":
                            self.beta_xaxis_var.set(new_xaxis)
                        elif plot_name == "momentum":
                            self.momentum_xaxis_var.set(new_xaxis)
                        elif plot_name == "transverse":
                            self.transverse_xaxis_var.set(new_xaxis)
                        elif plot_name == "energy":
                            self.energy_xaxis_var.set(new_xaxis)
                            if new_yaxis:
                                self.energy_yaxis_var.set(new_yaxis)
                        elif plot_name == "gamma":
                            self.gamma_xaxis_var.set(new_xaxis)

                        self._append_log(
                            f"Axis changed for {plot_name} plot. Re-run simulation to see changes."
                        )
                        window.after(
                            100,
                            lambda: _show_warning_dialog(
                                window,
                                "Axis Changed",
                                f"Axis preference saved. Please re-run the simulation to regenerate the {title} plot with the new axes.",
                            ),
                        )
                        return

                    self._replot_with_new_axis(
                        figure, plot_name, new_xaxis, canvas, new_yaxis
                    )

                    if plot_name == "beta":
                        self.beta_xaxis_var.set(new_xaxis)
                    elif plot_name == "momentum":
                        self.momentum_xaxis_var.set(new_xaxis)
                    elif plot_name == "transverse":
                        self.transverse_xaxis_var.set(new_xaxis)
                    elif plot_name == "energy":
                        self.energy_xaxis_var.set(new_xaxis)
                        if new_yaxis:
                            self.energy_yaxis_var.set(new_yaxis)
                    elif plot_name == "gamma":
                        self.gamma_xaxis_var.set(new_xaxis)

                    self._append_log(f"Axis changed for {plot_name} plot.")

                except Exception as e:
                    self._append_log(f"Error switching axis: {e}")
                    import traceback

                    traceback.print_exc()

            ttk.Label(controls_frame, text="X-axis:").pack(side=tk.LEFT, padx=(10, 5))
            xaxis_combo = ttk.Combobox(
                controls_frame,
                textvariable=xaxis_var,
                values=["t", "z"],
                width=8,
                state="readonly",
            )
            xaxis_combo.pack(side=tk.LEFT, padx=5)
            xaxis_combo.bind("<<ComboboxSelected>>", lambda e: switch_xaxis())

            if plot_supports_yaxis_switch and yaxis_var:
                ttk.Label(controls_frame, text="Y-axis:").pack(
                    side=tk.LEFT, padx=(10, 5)
                )
                yaxis_combo = ttk.Combobox(
                    controls_frame,
                    textvariable=yaxis_var,
                    values=["delta_total", "delta_z", "delta_x", "delta_y", "total"],
                    width=12,
                    state="readonly",
                )
                yaxis_combo.pack(side=tk.LEFT, padx=5)
                yaxis_combo.bind("<<ComboboxSelected>>", lambda e: switch_xaxis())

        ttk.Separator(controls_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        def save_figure() -> None:
            try:
                default_name = f"{title.replace(' ', '_').replace('/', '_')}.png"
                figure.savefig(default_name, dpi=150, bbox_inches="tight")
                self._append_log(f"Figure saved to: {default_name}")
            except Exception as e:
                _show_error_dialog(window, "Save Error", f"Failed to save figure: {e}")

        def save_figure_as() -> None:
            try:
                import os

                default_name = f"{title.replace(' ', '_').replace('/', '_')}.png"
                default_dir = "results/figures"
                if not os.path.exists(default_dir):
                    os.makedirs(default_dir, exist_ok=True)

                filename = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    initialfile=default_name,
                    initialdir=default_dir,
                    filetypes=[
                        ("PNG files", "*.png"),
                        ("PDF files", "*.pdf"),
                        ("SVG files", "*.svg"),
                        ("All files", "*.*"),
                    ],
                )
                if filename:
                    figure.savefig(filename, dpi=150, bbox_inches="tight")
                    self._append_log(f"Figure saved to: {filename}")
            except Exception as e:
                _show_error_dialog(window, "Save Error", f"Failed to save figure: {e}")

        ttk.Button(controls_frame, text="Save", command=save_figure).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(controls_frame, text="Save As...", command=save_figure_as).pack(
            side=tk.LEFT, padx=5
        )

        if width_px and height_px:
            window.geometry(f"{width_px}x{height_px + 100}")

        handle = _FigureHandle(name=title, figure=figure, window=window, canvas=canvas)
        self._figure_windows.append(handle)
        window.protocol("WM_DELETE_WINDOW", partial(self._close_figure, handle))

    def _close_figure(self, handle: Any) -> None:
        if handle in self._figure_windows:
            self._figure_windows.remove(handle)
        handle.canvas.get_tk_widget().destroy()
        handle.window.destroy()

    def _prepare_figure_for_display(self, figure: Any) -> Tuple[int, int]:
        from .gui import DISPLAY_MAX_HEIGHT, DISPLAY_MAX_WIDTH

        try:
            current_dpi = float(figure.get_dpi())
            width_in, height_in = [float(v) for v in figure.get_size_inches()]
        except Exception:  # pragma: no cover - defensive fallback
            return 0, 0

        width_px = width_in * current_dpi
        height_px = height_in * current_dpi

        scale = min(
            DISPLAY_MAX_WIDTH / width_px if width_px else 1.0,
            DISPLAY_MAX_HEIGHT / height_px if height_px else 1.0,
            1.0,
        )

        if scale < 1.0:
            new_width_in = max(1e-3, width_in * scale)
            new_height_in = max(1e-3, height_in * scale)
            figure.set_size_inches(new_width_in, new_height_in, forward=False)
            self._scale_figure_visuals(figure, scale)
            width_px = new_width_in * current_dpi
            height_px = new_height_in * current_dpi

        return int(width_px), int(height_px)

    def _scale_figure_visuals(self, figure: Any, scale: float) -> None:
        if scale >= 0.999:
            return
        try:
            from matplotlib.collections import PathCollection
            from matplotlib.lines import Line2D
            from matplotlib.text import Text
        except Exception:  # pragma: no cover - matplotlib internals missing
            return

        for text in figure.findobj(match=Text):
            text.set_fontsize(text.get_fontsize() * scale)

        for line in figure.findobj(match=Line2D):
            line.set_linewidth(line.get_linewidth() * scale)
            line.set_markersize(line.get_markersize() * scale)

        for collection in figure.findobj(match=PathCollection):
            sizes = collection.get_sizes()
            if sizes is not None and len(sizes):
                collection.set_sizes(sizes * scale * scale)
