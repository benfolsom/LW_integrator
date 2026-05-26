"""UI helper mixins for :class:`lw_integrator.optimization_plugin.OptimizationPlugin`."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from lw_integrator.testbed_runner import RADIATION_REACTION_MODE_CHOICES
from optimization.ui_helpers import ToolTip


class OptimizationPluginUIMixin:
    """Build and manage the shared Tk UI sections for the optimization plugin."""

    def _build_ui(self):
        """Build the user interface."""
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

        def _bind_to_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.canvas.bind_all("<Button-4>", _on_mousewheel_linux)
            self.canvas.bind_all("<Button-5>", _on_mousewheel_linux)

        def _unbind_from_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

        self.canvas.bind("<Enter>", _bind_to_mousewheel)
        self.canvas.bind("<Leave>", _unbind_from_mousewheel)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._build_simulation_section()
        self._build_mode_section()
        self._build_parameter_section()
        self._build_objective_section()
        self._build_optimization_section()
        self._build_control_section()
        self._build_results_output_section()
        self._build_progress_section()

        self._update_mode_visibility()

    def _add_tooltip(self, widget, text):
        """Add a tooltip to a widget."""
        ToolTip(widget, text)

    def _build_simulation_section(self):
        """Build simulation type selection section."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Simulation Type", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)

        self.sim_type_var = tk.StringVar(value="CONDUCTING_WALL")
        types = [
            ("Conducting Wall", "CONDUCTING_WALL"),
            ("Switching Wall", "SWITCHING_WALL"),
            ("Bunch to Bunch", "BUNCH_TO_BUNCH"),
        ]

        for i, (label, value) in enumerate(types):
            rb = ttk.Radiobutton(
                frame,
                text=label,
                variable=self.sim_type_var,
                value=value,
                command=self._on_sim_type_changed,
            )
            rb.grid(row=0, column=i, padx=5, sticky="w")

        self.sim_type_frame = frame

    def _build_mode_section(self):
        """Build mode selection section (blind sweep vs optimization)."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Run Mode", padding=10)
        frame.pack(fill="x", padx=10, pady=5)
        self.mode_section_frame = frame

        self.mode_var = tk.StringVar(value="blind_sweep")

        mode_frame = ttk.Frame(frame)
        mode_frame.pack(fill="x", pady=5)

        ttk.Radiobutton(
            mode_frame,
            text="Blind Sweep (Grid Search)",
            variable=self.mode_var,
            value="blind_sweep",
            command=self._update_mode_visibility,
        ).grid(row=0, column=0, padx=5, sticky="w")

        ttk.Radiobutton(
            mode_frame,
            text="Optimization",
            variable=self.mode_var,
            value="optimization",
            command=self._update_mode_visibility,
        ).grid(row=0, column=1, padx=5, sticky="w")

        help_frame = ttk.Frame(frame)
        help_frame.pack(fill="x", pady=(5, 0))

        self.mode_help_label = ttk.Label(
            help_frame,
            text="Blind Sweep: Exhaustively evaluate all parameter combinations (good for exploring full space).\n"
            "Optimization: Use algorithms to find optimal parameters (faster, finds best configurations).",
            foreground="gray40",
            font=("TkDefaultFont", 8),
            wraplength=600,
        )
        self.mode_help_label.pack(anchor="w")

    def _build_objective_section(self):
        """Build optimization objective selection section."""
        self._build_particle_section()

        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Optimization Objective", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)

        self.objective_var = tk.StringVar(value="max_energy_gain")
        objectives = [
            ("Maximize Energy Gain (GeV)", "max_energy_gain"),
            ("Maximize Energy Gain (%)", "max_percent_energy_gain"),
            ("Maximize Energy Efficiency", "max_energy_efficiency"),
            (
                "Maximize Inward Radial Focusing (final, 0 < dE ≤ 20%)",
                "max_inward_rider_radial_focusing_constrained_energy",
            ),
            (
                "Maximize Peak Inward Radial Focusing (centroid, 0 < dE ≤ 20%)",
                "max_peak_inward_rider_radial_focusing_constrained_energy",
            ),
            (
                "Maximize Peak Ring RMS Collapse (0 < dE ≤ 20%)",
                "max_peak_rider_radial_rms_collapse_constrained_energy",
            ),
            ("Minimize Transverse Deflection", "min_transverse_deflection"),
        ]

        for i, (label, value) in enumerate(objectives):
            rb = ttk.Radiobutton(
                frame, text=label, variable=self.objective_var, value=value
            )
            rb.grid(row=i, column=0, sticky="w", pady=2)

    def _build_optimization_section(self):
        """Build optimization method selection section."""
        self.optimization_frame = ttk.LabelFrame(
            self.scrollable_frame, text="Optimization Settings", padding=10
        )
        self.optimization_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(self.optimization_frame, text="Optimization Method:").grid(
            row=0, column=0, sticky="w", pady=2
        )

        self.optimization_method_var = tk.StringVar(value="differential_evolution")
        method_combo = ttk.Combobox(
            self.optimization_frame,
            textvariable=self.optimization_method_var,
            values=[
                "differential_evolution",
                "genetic_algorithm",
                "multi_start",
                "adaptive_grid",
            ],
            state="readonly",
            width=25,
        )
        method_combo.grid(
            row=0, column=1, columnspan=2, sticky="ew", pady=2, padx=(5, 0)
        )
        method_combo.bind("<<ComboboxSelected>>", self._update_optimization_controls)

        method_descriptions = {
            "differential_evolution": "⭐ RECOMMENDED: Global optimizer, robust to noise, best overall choice",
            "genetic_algorithm": "Evolutionary approach with selection, crossover, and mutation (robust, parallelizable)",
            "multi_start": "Multiple random starting points with local optimization (finds global optima)",
            "adaptive_grid": "Coarse-to-fine grid refinement (systematic exploration, creates heatmaps)",
        }

        self.method_desc_label = ttk.Label(
            self.optimization_frame,
            text=method_descriptions["differential_evolution"],
            foreground="gray40",
            font=("TkDefaultFont", 8),
            wraplength=500,
        )
        self.method_desc_label.grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        params_frame = ttk.Frame(self.optimization_frame)
        params_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)

        ttk.Label(params_frame, text="Max Iterations/Generations:").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_maxiter_var = tk.StringVar(value="50")
        ttk.Entry(
            params_frame, textvariable=self.optimization_maxiter_var, width=10
        ).grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(params_frame, text="Population Size:").grid(
            row=0, column=2, sticky="w", pady=2, padx=(15, 5)
        )
        self.optimization_popsize_var = tk.StringVar(value="20")
        self.popsize_entry = ttk.Entry(
            params_frame, textvariable=self.optimization_popsize_var, width=10
        )
        self.popsize_entry.grid(row=0, column=3, sticky="w", pady=2)

        self.ga_frame = ttk.LabelFrame(
            self.optimization_frame, text="Genetic Algorithm Parameters", padding=5
        )
        self.ga_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)

        ga_params_frame = ttk.Frame(self.ga_frame)
        ga_params_frame.pack(fill="x")

        ttk.Label(ga_params_frame, text="Mutation Rate:").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_mutation_var = tk.StringVar(value="0.1")
        ttk.Entry(
            ga_params_frame, textvariable=self.optimization_mutation_var, width=8
        ).grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(ga_params_frame, text="Crossover Rate:").grid(
            row=0, column=2, sticky="w", pady=2, padx=(15, 5)
        )
        self.optimization_crossover_var = tk.StringVar(value="0.7")
        ttk.Entry(
            ga_params_frame, textvariable=self.optimization_crossover_var, width=8
        ).grid(row=0, column=3, sticky="w", pady=2)

        self.multistart_frame = ttk.Frame(self.optimization_frame)
        self.multistart_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)

        ttk.Label(self.multistart_frame, text="Number of Random Starts:").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_nstarts_var = tk.StringVar(value="5")
        ttk.Entry(
            self.multistart_frame, textvariable=self.optimization_nstarts_var, width=10
        ).grid(row=0, column=1, sticky="w", pady=2)

        output_frame = ttk.LabelFrame(
            self.optimization_frame, text="Output Options", padding=5
        )
        output_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Label(output_frame, text="Save top N trajectories:").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_save_top_n_var = tk.StringVar(value="3")
        self.optimization_save_top_n_entry = ttk.Entry(
            output_frame, textvariable=self.optimization_save_top_n_var, width=8
        )
        self.optimization_save_top_n_entry.grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(
            output_frame,
            text="(Re-runs top N parameter sets to generate trajectories)",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray50",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 2))

        convergence_frame = ttk.LabelFrame(
            self.optimization_frame, text="Convergence Settings", padding=5
        )
        convergence_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Label(convergence_frame, text="Tolerance (rel):").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_convergence_tol_var = tk.StringVar(value="1e-6")
        ttk.Entry(
            convergence_frame,
            textvariable=self.optimization_convergence_tol_var,
            width=10,
        ).grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(convergence_frame, text="Patience (generations):").grid(
            row=0, column=2, sticky="w", pady=2, padx=(15, 5)
        )
        self.optimization_convergence_patience_var = tk.StringVar(value="10")
        ttk.Entry(
            convergence_frame,
            textvariable=self.optimization_convergence_patience_var,
            width=8,
        ).grid(row=0, column=3, sticky="w", pady=2)

        ttk.Label(
            convergence_frame,
            text="Early stopping: stops if fitness doesn't improve by tolerance over patience generations",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray40",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))

        self._update_optimization_controls()

    def _update_mode_visibility(self):
        """Update visibility of sections based on selected mode."""
        mode = self.mode_var.get()

        if mode == "blind_sweep":
            self.optimization_frame.pack_forget()
            self._set_top_n_controls_state("disabled")
        else:
            self.optimization_frame.pack(fill="x", padx=10, pady=5)
            self._set_top_n_controls_state("normal")

        self._update_parameter_visibility()

    def _set_top_n_controls_state(self, state):
        """Enable or disable Top N related controls."""
        if not hasattr(self, "save_top_n_traj_var"):
            return

        for attr in (
            "optimization_save_top_n_entry",
            "save_top_n_traj_check",
            "metrics_scope_top_n_radio",
            "log_top_n_only_radio",
        ):
            if hasattr(self, attr):
                getattr(self, attr).configure(state=state)

        if state == "disabled":
            if hasattr(self, "save_top_n_traj_var") and self.save_top_n_traj_var.get():
                self.save_top_n_traj_var.set(False)
            if (
                hasattr(self, "metrics_scope_var")
                and self.metrics_scope_var.get() == "top_n"
            ):
                self.metrics_scope_var.set("all")
            if (
                hasattr(self, "log_verbosity_var")
                and self.log_verbosity_var.get() == "top_n_only"
            ):
                self.log_verbosity_var.set("truncated")

    def _update_optimization_controls(self, event=None):
        """Update visibility of optimization controls based on selected method."""
        method = self.optimization_method_var.get()

        method_descriptions = {
            "differential_evolution": "⭐ RECOMMENDED: Global optimizer, robust to noise, best overall choice",
            "genetic_algorithm": "Evolutionary approach with selection, crossover, and mutation (robust, parallelizable)",
            "multi_start": "Multiple random starting points with local optimization (finds global optima)",
            "adaptive_grid": "Coarse-to-fine grid refinement (systematic exploration, creates heatmaps)",
        }
        self.method_desc_label.config(text=method_descriptions.get(method, ""))

        if method == "genetic_algorithm":
            self.ga_frame.grid()
            self.multistart_frame.grid_forget()
            self.popsize_entry.config(state="normal")
        elif method == "differential_evolution":
            self.ga_frame.grid_forget()
            self.multistart_frame.grid_forget()
            self.popsize_entry.config(state="normal")
        elif method == "multi_start":
            self.ga_frame.grid_forget()
            self.multistart_frame.grid()
            self.popsize_entry.config(state="disabled")
        else:
            self.ga_frame.grid_forget()
            self.multistart_frame.grid_forget()
            self.popsize_entry.config(state="disabled")

    def _build_control_section(self):
        """Build control buttons section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Sweep Tools", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        info_label = ttk.Label(
            frame,
            text="Use 'Run Mode' selector in right panel to choose Single Run or Parameter Sweep, then click Run button.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray40",
        )
        info_label.pack(anchor="w", pady=(0, 10))

        helper_frame = ttk.Frame(frame)
        helper_frame.pack(fill="x", pady=2)

        ttk.Button(
            helper_frame,
            text="Load from Single Run Config",
            command=self._on_load_from_main_config,
        ).pack(side="left", padx=5)

        ttk.Label(
            helper_frame,
            text="← Copy current single-run config to sweep parameters",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).pack(side="left", padx=5)

        robustness_frame = ttk.Frame(frame)
        robustness_frame.pack(fill="x", pady=(10, 2))

        ttk.Label(robustness_frame, text="Robustness:").pack(side="left", padx=(5, 10))

        ttk.Label(robustness_frame, text="Per-run timeout (s, 0=unlimited):").pack(
            side="left", padx=(0, 5)
        )
        self.per_run_timeout_var = tk.StringVar(value="300.0")
        ttk.Entry(
            robustness_frame, textvariable=self.per_run_timeout_var, width=8
        ).pack(side="left", padx=(0, 15))

        ttk.Label(robustness_frame, text="Sweep workers:").pack(
            side="left", padx=(0, 5)
        )
        self.workers_var = tk.StringVar(value="1")
        ttk.Entry(robustness_frame, textvariable=self.workers_var, width=6).pack(
            side="left", padx=(0, 5)
        )
        ttk.Label(
            robustness_frame,
            text="← 1=sequential; use a modest count (e.g. 2-4) to avoid oversubscribing the machine",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).pack(side="left", padx=(0, 15))

        self.skip_failed_runs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            robustness_frame,
            text="Skip failed runs and continue sweep",
            variable=self.skip_failed_runs_var,
        ).pack(side="left", padx=5)

        rr_frame = ttk.Frame(frame)
        rr_frame.pack(fill="x", pady=(5, 2))

        ttk.Label(rr_frame, text="Radiation reaction mode:").pack(
            side="left", padx=(5, 10)
        )
        self.radiation_reaction_mode_var = tk.StringVar(
            value=getattr(self.config, "radiation_reaction_mode", "medina_lad")
        )
        rr_combo = ttk.Combobox(
            rr_frame,
            textvariable=self.radiation_reaction_mode_var,
            values=RADIATION_REACTION_MODE_CHOICES,
            state="readonly",
            width=24,
        )
        rr_combo.pack(side="left", padx=(0, 5))
        self._add_tooltip(
            rr_combo,
            "Radiation-reaction mode for sweep runs.\n\n"
            "off - no momentum change from self-radiation\n"
            "diagnostic_only - record radiated power without changing momentum\n"
            "power_matched_damping - post-update energy-matched damping\n"
            "medina_lad - recommended default for new study runs",
        )
        ttk.Label(
            rr_frame,
            text="← use medina_lad for normal study runs; compare against off/diagnostic_only when needed",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).pack(side="left", padx=5)

        retry_frame = ttk.Frame(frame)
        retry_frame.pack(fill="x", pady=(5, 2))

        ttk.Label(retry_frame, text="Failed run retries:").pack(
            side="left", padx=(5, 10)
        )

        ttk.Label(retry_frame, text="Retry attempts (with new random seeds):").pack(
            side="left", padx=(0, 5)
        )
        self.failed_run_retry_attempts_var = tk.StringVar(value="1")
        ttk.Entry(
            retry_frame, textvariable=self.failed_run_retry_attempts_var, width=8
        ).pack(side="left", padx=(0, 5))

        ttk.Label(
            retry_frame,
            text="← 0=no retries, 1=retry once (default), 2+=retry multiple times",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).pack(side="left", padx=5)

        smoothness_frame = ttk.LabelFrame(
            frame, text="Trajectory Stability Analysis", padding=8
        )
        smoothness_frame.pack(fill="x", pady=(5, 0))

        self.smoothness_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            smoothness_frame,
            text="Enable multi-step stability analysis (detects numerical instabilities)",
            variable=self.smoothness_enabled_var,
            command=self._toggle_smoothness_controls,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(smoothness_frame, text="Window size (steps):").grid(
            row=1, column=0, sticky="w", padx=(20, 5), pady=2
        )
        self.smoothness_window_var = tk.StringVar(value="20")
        window_entry = ttk.Entry(
            smoothness_frame, textvariable=self.smoothness_window_var, width=8
        )
        window_entry.grid(row=1, column=1, sticky="w", pady=2)
        self._add_tooltip(window_entry, "Moving window size for trend analysis")

        ttk.Label(smoothness_frame, text="Oscillation threshold:").grid(
            row=2, column=0, sticky="w", padx=(20, 5), pady=2
        )
        self.smoothness_oscillation_var = tk.StringVar(value="0.5")
        oscillation_entry = ttk.Entry(
            smoothness_frame, textvariable=self.smoothness_oscillation_var, width=8
        )
        oscillation_entry.grid(row=2, column=1, sticky="w", pady=2)
        self._add_tooltip(
            oscillation_entry, "Sign-change rate threshold (lower = stricter)"
        )

        self.smoothness_reject_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            smoothness_frame,
            text="Reject runs with numerical instabilities",
            variable=self.smoothness_reject_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", padx=(20, 0), pady=2)

        self.smoothness_widgets = [
            smoothness_frame.grid_slaves(row=1, column=0)[0],
            smoothness_frame.grid_slaves(row=1, column=1)[0],
            smoothness_frame.grid_slaves(row=2, column=0)[0],
            smoothness_frame.grid_slaves(row=2, column=1)[0],
            smoothness_frame.grid_slaves(row=3, column=0)[0],
        ]

    def _build_results_output_section(self):
        """Build results viewing and output configuration section."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Results & Output Configuration", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)
        frame.columnconfigure(1, weight=1)
        self.results_output_frame = frame

        results_frame = ttk.Frame(frame)
        results_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 15))

        ttk.Label(results_frame, text="View Results:").pack(side="left", padx=(0, 10))

        ttk.Button(
            results_frame, text="View Results", command=self._on_view_results
        ).pack(side="left", padx=5)

        ttk.Button(
            results_frame,
            text="Plot Trajectories",
            command=self._on_plot_trajectories,
        ).pack(side="left", padx=5)

        ttk.Label(frame, text="Trajectory Data:").grid(
            row=1, column=0, sticky="nw", pady=(5, 2)
        )

        traj_frame = ttk.Frame(frame)
        traj_frame.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(5, 2))

        self.save_top_n_traj_var = tk.BooleanVar(value=False)
        self.save_top_n_traj_check = ttk.Checkbutton(
            traj_frame,
            text="Top N trajectories (full detail)",
            variable=self.save_top_n_traj_var,
            command=self._on_top_n_traj_changed,
        )
        self.save_top_n_traj_check.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.save_all_traj_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            traj_frame,
            text="All trajectories (with stride)",
            variable=self.save_all_traj_var,
            command=self._on_all_traj_changed,
        ).grid(row=0, column=1, sticky="w", padx=(0, 10))

        self.save_failed_traj_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            traj_frame,
            text="Failed only (full detail)",
            variable=self.save_failed_traj_var,
            command=self._on_failed_traj_changed,
        ).grid(row=0, column=2, sticky="w", padx=(0, 10))

        self.trajectory_stride_label = ttk.Label(traj_frame, text="Stride:")
        self.trajectory_stride_label.grid(row=0, column=3, sticky="w", padx=(10, 2))
        self.trajectory_stride_var = tk.StringVar(value="1")
        self.trajectory_stride_entry = ttk.Entry(
            traj_frame, textvariable=self.trajectory_stride_var, width=6
        )
        self.trajectory_stride_entry.grid(row=0, column=4, sticky="w", padx=2)

        self._update_stride_state()

        ttk.Label(frame, text="Metrics Export:").grid(
            row=2, column=0, sticky="nw", pady=(10, 2)
        )

        metrics_frame = ttk.Frame(frame)
        metrics_frame.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(10, 2))

        format_frame = ttk.Frame(metrics_frame)
        format_frame.grid(row=0, column=0, sticky="w", pady=2)

        ttk.Label(format_frame, text="Format:").pack(side="left", padx=(0, 5))

        self.metrics_format_var = tk.StringVar(value="both")
        format_options = [
            ("JSON + CSV", "both"),
            ("JSON only", "json"),
            ("CSV only", "csv"),
            ("None", "none"),
        ]
        for text, value in format_options:
            ttk.Radiobutton(
                format_frame,
                text=text,
                variable=self.metrics_format_var,
                value=value,
            ).pack(side="left", padx=5)

        scope_frame = ttk.Frame(metrics_frame)
        scope_frame.grid(row=1, column=0, sticky="w", pady=2)

        ttk.Label(scope_frame, text="Scope:").pack(side="left", padx=(0, 5))

        self.metrics_scope_var = tk.StringVar(value="all")
        ttk.Radiobutton(
            scope_frame,
            text="All evaluations",
            variable=self.metrics_scope_var,
            value="all",
        ).pack(side="left", padx=5)
        self.metrics_scope_top_n_radio = ttk.Radiobutton(
            scope_frame,
            text="Top N only",
            variable=self.metrics_scope_var,
            value="top_n",
        )
        self.metrics_scope_top_n_radio.pack(side="left", padx=5)

        ttk.Label(
            frame,
            text="ℹ JSON contains metadata & structure; CSV is tabular with all parameters & metrics",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray50",
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(frame, text="Debug Logs:").grid(
            row=4, column=0, sticky="nw", pady=(5, 2)
        )

        log_frame = ttk.Frame(frame)
        log_frame.grid(row=4, column=1, columnspan=2, sticky="ew", pady=(5, 2))

        self.log_verbosity_var = tk.StringVar(value="truncated")

        ttk.Radiobutton(
            log_frame,
            text="None (no debug logs saved)",
            variable=self.log_verbosity_var,
            value="none",
        ).grid(row=0, column=0, sticky="w", pady=2)

        ttk.Radiobutton(
            log_frame,
            text="Truncated (1-2 lines/run: parameters + metrics + errors only) — DEFAULT",
            variable=self.log_verbosity_var,
            value="truncated",
        ).grid(row=1, column=0, sticky="w", pady=2)

        ttk.Radiobutton(
            log_frame,
            text="Full debug (inherits SC verbosity & adaptive timestep debug from Stability tab)",
            variable=self.log_verbosity_var,
            value="full",
        ).grid(row=2, column=0, sticky="w", pady=2)

        self.log_top_n_only_radio = ttk.Radiobutton(
            log_frame,
            text="Top-N compact (suppresses SC/adaptive debug during sweep)",
            variable=self.log_verbosity_var,
            value="top_n_only",
        )
        self.log_top_n_only_radio.grid(row=3, column=0, sticky="w", pady=2)

        ttk.Label(
            frame,
            text="ℹ 'Truncated' is recommended for large sweeps.\n'Full debug' inherits verbosity settings from Stability tab and generates large log files.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="blue",
            justify="left",
        ).grid(row=5, column=1, columnspan=2, sticky="w", pady=(0, 5))

    def _on_top_n_traj_changed(self):
        """Handle Top N trajectory checkbox change."""
        self._update_stride_state()

    def _on_all_traj_changed(self):
        """Handle All trajectories checkbox change."""
        if self.save_all_traj_var.get():
            self.save_failed_traj_var.set(False)
        self._update_stride_state()

    def _on_failed_traj_changed(self):
        """Handle Failed only checkbox change."""
        if self.save_failed_traj_var.get():
            self.save_all_traj_var.set(False)
        self._update_stride_state()

    def _update_stride_state(self):
        """Update stride field enabled/disabled state."""
        if not hasattr(self, "trajectory_stride_entry"):
            return

        stride_enabled = self.save_all_traj_var.get()
        widget_state = "normal" if stride_enabled else "disabled"
        label_color = "black" if stride_enabled else "gray"

        self.trajectory_stride_entry.configure(state=widget_state)
        self.trajectory_stride_label.configure(foreground=label_color)

    def _build_progress_section(self):
        """Build progress monitoring section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Sweep Progress", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        self.progress_bar = ttk.Progressbar(
            frame, mode="determinate", maximum=100, length=400
        )
        self.progress_bar.pack(fill="x", pady=5)

        self.progress_label = ttk.Label(frame, text="Ready")
        self.progress_label.pack(anchor="w", pady=2)

        log_info = ttk.Label(
            frame,
            text="📋 Sweep progress and results are logged to the main GUI's LOGS window",
            font=("TkDefaultFont", 9),
            foreground="blue",
        )
        log_info.pack(anchor="w", pady=(10, 2))

    def _toggle_smoothness_controls(self):
        """Enable/disable smoothness controls based on checkbox."""
        state = "normal" if self.smoothness_enabled_var.get() else "disabled"
        for widget in self.smoothness_widgets:
            widget.configure(state=state)
