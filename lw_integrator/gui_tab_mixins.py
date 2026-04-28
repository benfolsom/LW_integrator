"""Tab-builder helpers for the main GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .testbed_runner import (
    AVAILABLE_DPI_CHOICES,
    CORE_PARAM_LABELS,
    PARAM_LABELS,
    PARTICLE_PARAM_FIELDS,
    SPECIES_OPTIONS,
)


class IntegratorGUITabMixin:
    """Own construction of the main notebook tabs."""

    def _build_particle_tab(self) -> None:
        """Build rider/driver particle and bunch-distribution controls."""
        from .gui import Tooltip

        particle_frame = self._create_scrollable_tab(
            self.notebook, "Particles", padding=12
        )
        particle_frame.columnconfigure(1, weight=1, minsize=150)
        particle_frame.columnconfigure(3, weight=1, minsize=150)

        ttk.Label(particle_frame, text="Rider species preset:").grid(
            row=0, column=0, sticky="w"
        )
        rider_combo = ttk.Combobox(
            particle_frame,
            textvariable=self.rider_species_var,
            values=[label for label, _ in SPECIES_OPTIONS],
            state="readonly",
        )
        rider_combo.grid(row=0, column=1, sticky="ew")
        rider_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._apply_species("rider")
        )

        ttk.Label(particle_frame, text="Driver species preset:").grid(
            row=0, column=2, sticky="w", padx=(12, 0)
        )
        driver_combo = ttk.Combobox(
            particle_frame,
            textvariable=self.driver_species_var,
            values=[label for label, _ in SPECIES_OPTIONS],
            state="readonly",
        )
        driver_combo.grid(row=0, column=3, sticky="ew")
        driver_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._apply_species("driver")
        )
        self.driver_species_combo = driver_combo

        ttk.Label(
            particle_frame,
            text="Note: Particle count, transverse spread, and transverse momentum define the bunch distribution.\n"
            "Transverse offsets (x/y) define bunch center positions and are only used in BUNCH_TO_BUNCH mode.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 2))

        self._rider_offset_entries = []
        self._driver_offset_entries = []
        self._rider_offset_labels = []
        self._driver_offset_labels = []

        for row, name in enumerate(PARTICLE_PARAM_FIELDS, start=2):
            rider_label = ttk.Label(particle_frame, text=PARAM_LABELS[name] + ":")
            rider_label.grid(row=row, column=0, sticky="w", pady=2)

            rider_entry = ttk.Entry(
                particle_frame, textvariable=self.rider_param_vars[name], width=12
            )
            rider_entry.grid(row=row, column=1, sticky="ew", pady=2)

            driver_label = ttk.Label(
                particle_frame, text=PARAM_LABELS[name] + " (driver):"
            )
            driver_label.grid(row=row, column=2, sticky="w", pady=2, padx=(12, 0))

            driver_entry = ttk.Entry(
                particle_frame, textvariable=self.driver_param_vars[name], width=12
            )
            driver_entry.grid(row=row, column=3, sticky="ew", pady=2)
            self._driver_entries.append(driver_entry)

            if name in ("transv_offset_x", "transv_offset_y"):
                self._rider_offset_entries.append(rider_entry)
                self._rider_offset_labels.append(rider_label)
                self._driver_offset_entries.append(driver_entry)
                self._driver_offset_labels.append(driver_label)

                tooltip_text = (
                    "Transverse offset (bunch center position).\n"
                    "Only used in BUNCH_TO_BUNCH simulations.\n\n"
                    "Defines the (x, y) position of the bunch center.\n"
                    "Separation between rider and driver bunches is:\n"
                    "  √[(x_driver - x_rider)² + (y_driver - y_rider)²]"
                )
                Tooltip(rider_entry, tooltip_text)
                Tooltip(driver_entry, tooltip_text)

        next_row = len(PARTICLE_PARAM_FIELDS) + 2
        ttk.Label(particle_frame, text="Image subcharge count:").grid(
            row=next_row, column=0, sticky="w", pady=(12, 2)
        )
        self.image_subcharge_entry = ttk.Entry(
            particle_frame, textvariable=self.image_subcharge_var, width=12
        )
        self.image_subcharge_entry.grid(
            row=next_row, column=1, sticky="ew", pady=(12, 2)
        )

        help_text_subcharge = ttk.Label(
            particle_frame,
            text="(Number of virtual charges used to model conducting-wall images.\n"
            "Range: 4-128. Higher = more accurate but slower. Default: 12)",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        )
        help_text_subcharge.grid(
            row=next_row + 1, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        self.image_weighting_check = ttk.Checkbutton(
            particle_frame,
            text="Enable image weighting",
            variable=self.image_weighting_var,
        )
        self.image_weighting_check.grid(
            row=next_row + 2, column=0, columnspan=2, sticky="w", pady=2
        )

        help_text_weighting = ttk.Label(
            particle_frame,
            text="(Uses radial weighting when distributing subcharges along aperture.\n"
            "Improves accuracy for aperture geometry. Recommended: enabled)",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        )
        help_text_weighting.grid(
            row=next_row + 3, column=0, columnspan=2, sticky="w", pady=(0, 2)
        )

        next_row += 4
        ttk.Separator(particle_frame, orient="horizontal").grid(
            row=next_row, column=0, columnspan=4, sticky="ew", pady=(12, 12)
        )
        next_row += 1

        ttk.Label(
            particle_frame,
            text="Macroparticle Simulation (Conducting Wall only):",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=next_row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        next_row += 1

        self.macroparticle_enable_check = ttk.Checkbutton(
            particle_frame,
            text="Enable macroparticle simulation (bunch spread inherited from above)",
            variable=self.macroparticle_enabled_var,
            command=self._toggle_macroparticle_controls,
        )
        self.macroparticle_enable_check.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=2
        )
        next_row += 1

        self.macroparticle_charge_label = ttk.Label(
            particle_frame, text="Charge multiplier:"
        )
        self.macroparticle_charge_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.macroparticle_charge_entry = ttk.Entry(
            particle_frame,
            textvariable=self.macroparticle_charge_multiplier_var,
            width=12,
        )
        self.macroparticle_charge_entry.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.macroparticle_sigma_label = ttk.Label(
            particle_frame, text="Image error sigma multiplier:"
        )
        self.macroparticle_sigma_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.macroparticle_sigma_entry = ttk.Entry(
            particle_frame,
            textvariable=self.macroparticle_sigma_multiplier_var,
            width=12,
        )
        self.macroparticle_sigma_entry.grid(row=next_row, column=1, sticky="ew", pady=2)
        next_row += 1

        self.macroparticle_momentum_errors_check = ttk.Checkbutton(
            particle_frame,
            text="Include momentum errors (cumulative)",
            variable=self.macroparticle_use_momentum_errors_var,
        )
        self.macroparticle_momentum_errors_check.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )
        next_row += 1

        help_text_macroparticle = ttk.Label(
            particle_frame,
            text=(
                "Macroparticle mode scales particle charge and adds Gaussian errors to image subcharges.\n"
                "Image errors are derived from bunch spread parameters (transv_dist, transv_mom) × sigma multiplier.\n"
                "Position errors: constant σ from transv_dist. Momentum errors: cumulative from transv_mom.\n"
                "Uncheck 'Include momentum errors' to apply only constant position errors (no cumulative growth).\n"
                "Only active for CONDUCTING_WALL simulations."
            ),
            font=("TkDefaultFont", 8),
            foreground="gray40",
            justify="left",
        )
        help_text_macroparticle.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=(0, 2), padx=(20, 0)
        )

        self._macroparticle_widgets = [
            self.macroparticle_charge_label,
            self.macroparticle_charge_entry,
            self.macroparticle_sigma_label,
            self.macroparticle_sigma_entry,
            self.macroparticle_momentum_errors_check,
        ]

    def _build_core_tab(self) -> None:
        """Build integration and force-cutoff controls."""
        from .gui import Tooltip

        core_frame = self._create_scrollable_tab(
            self.notebook, "Core params", padding=12
        )
        core_frame.columnconfigure(1, weight=1)

        self.core_param_widgets = {}
        row = 0

        ttk.Label(core_frame, text="Steps:").grid(row=row, column=0, sticky="w", pady=2)
        steps_widget = ttk.Entry(core_frame, textvariable=self.steps_var, width=16)
        steps_widget.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(core_frame, text="Seed:").grid(row=row, column=0, sticky="w", pady=2)
        seed_frame = ttk.Frame(core_frame)
        seed_frame.grid(row=row, column=1, sticky="ew", pady=2)
        seed_frame.columnconfigure(0, weight=1)

        self.seed_entry = ttk.Entry(seed_frame, textvariable=self.seed_var, width=16)
        self.seed_entry.grid(row=0, column=0, sticky="ew")

        self.random_seed_var = tk.BooleanVar(value=False)
        self.random_seed_check = ttk.Checkbutton(
            seed_frame,
            text="Random",
            variable=self.random_seed_var,
            command=self._toggle_random_seed,
        )
        self.random_seed_check.grid(row=0, column=1, sticky="w", padx=(5, 0))
        row += 1

        ttk.Separator(core_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(10, 10)
        )
        row += 1

        for name in CORE_PARAM_LABELS:
            if name in ["z_cutoff", "z_cutoff_mode", "mean", "startup_mode"]:
                continue

            ttk.Label(core_frame, text=CORE_PARAM_LABELS[name] + ":").grid(
                row=row, column=0, sticky="w", pady=2
            )

            widget = ttk.Entry(
                core_frame, textvariable=self.core_param_vars[name], width=16
            )
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            self.core_param_widgets[name] = widget
            row += 1

        ttk.Separator(core_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(10, 10)
        )
        row += 1

        ttk.Label(core_frame, text="Startup mode:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        startup_mode_combo = ttk.Combobox(
            core_frame,
            textvariable=self.core_param_vars["startup_mode"],
            values=["COLD_START", "APPROXIMATE_BACK_HISTORY"],
            state="readonly",
            width=22,
        )
        startup_mode_combo.grid(row=row, column=1, sticky="ew", pady=2)
        self.core_param_widgets["startup_mode"] = startup_mode_combo

        Tooltip(
            startup_mode_combo,
            "Startup mode controls retarded force calculation at early timesteps.\n\n"
            "COLD_START (default, recommended):\n"
            "  • Suppresses retarded forces until particles build causal history\n"
            "  • Physically realistic for transient events (beam turn-on)\n"
            "  • Avoids unphysical extrapolation errors\n"
            "  • Compatible with all features (adaptive timestep, energy monitor)\n"
            "  • May show startup transient in first ~100 steps\n\n"
            "APPROXIMATE_BACK_HISTORY (experimental, benchmarking only):\n"
            "  • Assumes particles had constant velocity since t = -∞\n"
            "  • Enables immediate force calculation (no gating)\n"
            "  • Intended only for benchmark/reference studies\n"
            "  • Not validated for production physics\n"
            "  • May introduce unphysical initial conditions\n\n"
            "For production: use COLD_START\n"
            "For reference benchmarking: use APPROXIMATE_BACK_HISTORY",
        )
        row += 1

        startup_help_label = ttk.Label(
            core_frame,
            text="COLD_START (recommended): Forces gated until causal history available\n"
            "APPROXIMATE_BACK_HISTORY: Constant-velocity extrapolation (benchmarking only)",
            foreground="gray",
            font=("TkDefaultFont", 8),
            justify="left",
            wraplength=450,
        )
        startup_help_label.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5)
        )
        row += 1

        ttk.Separator(core_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(10, 10)
        )
        row += 1

        ttk.Label(
            core_frame,
            text="Force Cutoff (optional):",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1

        self.z_cutoff_enable_check = ttk.Checkbutton(
            core_frame,
            text="Enable z-cutoff (stops applying external forces when particle z > cutoff)",
            variable=self.z_cutoff_enabled_var,
            command=self._toggle_z_cutoff_controls,
        )
        self.z_cutoff_enable_check.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        row += 1

        self.z_cutoff_label = ttk.Label(core_frame, text="z cutoff (mm):")
        self.z_cutoff_label.grid(row=row, column=0, sticky="w", pady=2, padx=(20, 0))
        self.z_cutoff_entry = ttk.Entry(
            core_frame, textvariable=self.core_param_vars["z_cutoff"], width=16
        )
        self.z_cutoff_entry.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        self.z_cutoff_mode_label = ttk.Label(core_frame, text="Reference:")
        self.z_cutoff_mode_label.grid(
            row=row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.z_cutoff_mode_combo = ttk.Combobox(
            core_frame,
            textvariable=self.core_param_vars["z_cutoff_mode"],
            values=["absolute", "relative"],
            state="readonly",
            width=14,
        )
        self.z_cutoff_mode_combo.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        help_label = ttk.Label(
            core_frame,
            text="'absolute': fixed z position in lab frame\n"
            "'relative': distance from particle starting position\n\n"
            "Works in all simulation modes (BUNCH_TO_BUNCH, CONDUCTING_WALL, SWITCHING_WALL).\n"
            "In SWITCHING_WALL mode with cavity_spacing > 0, cutoff advances by cavity_spacing\n"
            "when particle passes threshold, creating periodic cavity structure.",
            foreground="gray",
            font=("TkDefaultFont", 8),
            justify="left",
            wraplength=450,
        )
        help_label.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5), padx=(20, 0)
        )

    def _add_output_toggle(
        self,
        parent: tk.Widget,
        label: str,
        display_var: tk.BooleanVar,
        save_var: tk.BooleanVar,
        *,
        row: int,
    ) -> None:
        ttk.Checkbutton(
            parent, text=f"Display {label.lower()}", variable=display_var
        ).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Checkbutton(parent, text=f"Save {label.lower()}", variable=save_var).grid(
            row=row, column=1, sticky="w", pady=2
        )

    def _build_output_tab(self) -> None:
        """Build the single-run output tab."""
        output_frame = self._create_scrollable_tab(self.notebook, "Output", padding=12)
        output_frame.columnconfigure(1, weight=1)

        notice_frame = ttk.Frame(output_frame)
        notice_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 15))

        notice_label = ttk.Label(
            notice_frame,
            text="⚠ These settings apply to SINGLE RUNS only.\nFor sweep/optimization output configuration, see the 'Sweep/Optim' tab → 'Results & Output Configuration'.",
            font=("TkDefaultFont", 9, "bold"),
            foreground="blue",
            justify="left",
        )
        notice_label.pack(anchor="w")

        self._add_output_toggle(
            output_frame,
            "Energy plot",
            self.energy_display_var,
            self.energy_save_var,
            row=1,
        )
        ttk.Label(output_frame, text="  ↳ X-axis:").grid(
            row=4, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.energy_xaxis_var,
            values=["z", "t", "dual"],
            width=12,
            state="readonly",
        ).grid(row=3, column=1, sticky="w")

        ttk.Label(output_frame, text="  ↳ Y-axis:").grid(
            row=5, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.energy_yaxis_var,
            values=["delta_total", "delta_z", "delta_x", "delta_y", "total"],
            width=12,
            state="readonly",
        ).grid(row=5, column=1, sticky="w")
        self._add_output_toggle(
            output_frame,
            "Transverse position (⟨x⟩, ⟨y⟩)",
            self.transverse_display_var,
            self.transverse_save_var,
            row=5,
        )
        ttk.Label(output_frame, text="  ↳ X-axis:").grid(
            row=7, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.transverse_xaxis_var,
            values=["t", "z"],
            width=5,
            state="readonly",
        ).grid(row=7, column=1, sticky="w")

        self._add_output_toggle(
            output_frame,
            "Velocity (β_x, β_y, β_z, |β|)",
            self.beta_display_var,
            self.beta_save_var,
            row=7,
        )
        ttk.Label(output_frame, text="  ↳ X-axis:").grid(
            row=9, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.beta_xaxis_var,
            values=["t", "z"],
            width=5,
            state="readonly",
        ).grid(row=9, column=1, sticky="w")

        self._add_output_toggle(
            output_frame,
            "Conjugate momentum (Pˣ, Pʸ, Pᶻ, |P⊥|, Pᵗ, |P|)",
            self.momentum_display_var,
            self.momentum_save_var,
            row=9,
        )
        ttk.Label(output_frame, text="  ↳ X-axis:").grid(
            row=11, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.momentum_xaxis_var,
            values=["t", "z"],
            width=5,
            state="readonly",
        ).grid(row=11, column=1, sticky="w")

        self._add_output_toggle(
            output_frame,
            "Gamma (Lorentz factor γ)",
            self.gamma_display_var,
            self.gamma_save_var,
            row=12,
        )
        ttk.Label(output_frame, text="  ↳ X-axis:").grid(
            row=13, column=0, sticky="w", padx=(20, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.gamma_xaxis_var,
            values=["t", "z"],
            width=5,
            state="readonly",
        ).grid(row=13, column=1, sticky="w")

        ttk.Separator(output_frame, orient="horizontal").grid(
            row=14, column=0, columnspan=2, sticky="ew", pady=(10, 10)
        )

        ttk.Label(output_frame, text="Longitudinal trajectory:").grid(
            row=15, column=0, columnspan=2, sticky="w", pady=(0, 2)
        )
        self._add_output_toggle(
            output_frame,
            "  z vs time",
            self.zposition_display_var,
            self.zposition_save_var,
            row=16,
        )

        ttk.Separator(output_frame, orient="horizontal").grid(
            row=17, column=0, columnspan=2, sticky="ew", pady=(10, 10)
        )

        ttk.Label(output_frame, text="Plot DPI:").grid(row=18, column=0, sticky="w")
        ttk.Combobox(
            output_frame,
            textvariable=self.dpi_var,
            values=[str(dpi) for dpi in AVAILABLE_DPI_CHOICES],
            width=8,
            state="readonly",
        ).grid(row=18, column=1, sticky="w")

        ttk.Label(
            output_frame, text="Trajectory Data:", font=("TkDefaultFont", 9, "bold")
        ).grid(row=19, column=0, columnspan=2, sticky="w", pady=(12, 2))

        ttk.Checkbutton(
            output_frame,
            text="Save trajectory data (NPZ + JSON formats)",
            variable=self.trajectory_save_var,
            command=self._on_trajectory_save_toggled,
        ).grid(row=20, column=0, columnspan=2, sticky="w", pady=(0, 2))

        self.trajectory_stride_label = ttk.Label(
            output_frame, text="Trajectory stride:"
        )
        self.trajectory_stride_label.grid(row=21, column=0, sticky="w", padx=(20, 0))
        self.trajectory_stride_entry = ttk.Entry(
            output_frame, textvariable=self.trajectory_interval_var, width=8
        )
        self.trajectory_stride_entry.grid(row=21, column=1, sticky="w")

        ttk.Label(
            output_frame,
            text="(Save every Nth point to reduce file size)",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray50",
        ).grid(row=22, column=0, columnspan=2, sticky="w", padx=(20, 0), pady=(0, 10))

        self._on_trajectory_save_toggled()

        ttk.Label(
            output_frame, text="Debug Logs:", font=("TkDefaultFont", 9, "bold")
        ).grid(row=23, column=0, columnspan=2, sticky="w", pady=(5, 2))

        ttk.Checkbutton(
            output_frame,
            text="Save debug log file to output directory",
            variable=self.save_log_file_var,
        ).grid(row=24, column=0, columnspan=2, sticky="w", pady=(0, 2))

        ttk.Label(
            output_frame,
            text="(Captures console output, warnings, and diagnostic info)",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray50",
        ).grid(row=25, column=0, columnspan=2, sticky="w", padx=(20, 0))

    def _build_stability_tab(self) -> None:
        """Build self-consistency and adaptive timestep controls."""
        from .gui import Tooltip

        # Stability Settings tab ----------------------------------------
        stability_frame = self._create_scrollable_tab(
            self.notebook, "Stability", padding=12
        )
        stability_frame.columnconfigure(1, weight=1)

        # Notice about single run vs sweep/optimization
        stability_notice_frame = ttk.Frame(stability_frame)
        stability_notice_frame.grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 15)
        )

        stability_notice_label = ttk.Label(
            stability_notice_frame,
            text="⚠ These settings apply to BOTH single runs AND sweeps/optimizations.\nStability controls affect all simulation modes.",
            font=("TkDefaultFont", 9, "bold"),
            foreground="blue",
            justify="left",
        )
        stability_notice_label.pack(anchor="w")

        # Self-consistency section
        sc_frame = ttk.LabelFrame(
            stability_frame, text="Self-Consistency Checks", padding=8
        )
        sc_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        sc_frame.columnconfigure(1, weight=1)

        self.sc_enable_check = ttk.Checkbutton(
            sc_frame,
            text="Enable self-consistency iterations (recommended)",
            variable=self.self_consistency_enabled_var,
            command=self._toggle_self_consistency_controls,
        )
        self.sc_enable_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)

        # Convergence mode dropdown
        mode_frame = ttk.Frame(sc_frame)
        mode_frame.grid(row=1, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_mode_label = ttk.Label(mode_frame, text="Convergence mode:")
        self.sc_mode_label.pack(side="left")
        mode_help = ttk.Label(mode_frame, text="ⓘ", foreground="blue", cursor="hand2")
        mode_help.pack(side="left", padx=(3, 0))
        Tooltip(
            mode_help,
            "Self-consistency convergence mode.\n\n"
            "• Fixed Geometry (default, fastest):\n"
            "  Projects Pt onto mass shell each iteration\n"
            "  Geometry computed once per timestep\n"
            "  Use for most cases\n"
            "  Speed: 1× baseline, 2-5 iterations typical\n\n"
            "• Variable Geometry (accurate, slower):\n"
            "  Projects Pt onto mass shell each iteration\n"
            "  Recomputes geometry each SC iteration\n"
            "  Use when particle moves significantly: |Δx| ~ 0.1×R\n"
            "  Speed: 2-10× slower than fixed\n\n"
            "Default: Fixed Geometry",
        )
        self.sc_mode_combo = ttk.Combobox(
            sc_frame,
            textvariable=self.self_consistency_convergence_mode_var,
            values=[
                "fixed_geometry",
                "variable_geometry",
            ],
            state="readonly",
            width=20,
        )
        self.sc_mode_combo.grid(row=1, column=1, sticky="ew", pady=2)
        self.sc_mode_combo.bind("<<ComboboxSelected>>", self._on_sc_mode_changed)

        # Target mass-shell tolerance
        target_ms_frame = ttk.Frame(sc_frame)
        target_ms_frame.grid(row=2, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_target_ms_tolerance_label = ttk.Label(
            target_ms_frame, text="Target MS tolerance:"
        )
        self.sc_target_ms_tolerance_label.pack(side="left")
        target_ms_help = ttk.Label(
            target_ms_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        target_ms_help.pack(side="left", padx=(3, 0))
        Tooltip(
            target_ms_help,
            "TARGET mass-shell convergence criterion.\n\n"
            "Loop continues until:\n"
            "  |Pt² - P² - (mc)²|/(mc)² < target_ms_tolerance\n\n"
            "This ensures energy-momentum relation is satisfied.\n\n"
            "Default: 1e-6 (0.0001% relative error)\n"
            "Aggressive: 1e-8 for ultra-relativistic (γ > 1000)\n"
            "Minimum: 1e-10 (stricter = more iterations)",
        )
        self.sc_target_ms_tolerance_entry = ttk.Entry(
            sc_frame,
            textvariable=self.self_consistency_target_ms_tolerance_var,
            width=16,
        )
        self.sc_target_ms_tolerance_entry.grid(row=2, column=1, sticky="ew", pady=2)

        # Target gamma tolerance
        # Max iterations with help icon
        max_iter_frame = ttk.Frame(sc_frame)
        max_iter_frame.grid(row=4, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_max_iterations_label = ttk.Label(max_iter_frame, text="Max iterations:")
        self.sc_max_iterations_label.pack(side="left")
        iter_help = ttk.Label(
            max_iter_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        iter_help.pack(side="left", padx=(3, 0))
        Tooltip(
            iter_help,
            "Maximum self-consistency iterations per particle per step.\n\n"
            "Loop continues until:\n"
            "  • Mass-shell constraint satisfied (momentum-based check), OR\n"
            "  • Max iterations reached (applies projection as fallback)\n\n"
            "More iterations = better accuracy but slower.\n"
            "Typical convergence: 2-4 iterations\n\n"
            "Default: 10\n"
            "Aggressive: 20 for ultra-relativistic particles (γ > 1000)\n"
            "Increase if seeing 'max iterations reached' warnings",
        )
        self.sc_max_iterations_entry = ttk.Entry(
            sc_frame, textvariable=self.self_consistency_max_iterations_var, width=16
        )
        self.sc_max_iterations_entry.grid(row=4, column=1, sticky="ew", pady=2)

        # Mass-shell tolerance (safety net) with help icon
        ms_frame = ttk.Frame(sc_frame)
        ms_frame.grid(row=5, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_mass_shell_tolerance_label = ttk.Label(
            ms_frame, text="Mass-shell tolerance:"
        )
        self.sc_mass_shell_tolerance_label.pack(side="left")
        ms_help = ttk.Label(ms_frame, text="ⓘ", foreground="blue", cursor="hand2")
        ms_help.pack(side="left", padx=(3, 0))
        Tooltip(
            ms_help,
            "SAFETY NET threshold enforced after iteration loop.\n\n"
            "PHYSICS: Pt² - P² = (mc)² must hold exactly.\n"
            "Numerical errors can violate this slightly.\n\n"
            "When |Pt² - P² - (mc)²|/(mc)² > tolerance:\n"
            "  → Pt is clamped to √(P² + (mc)²) as fallback\n\n"
            "This acts as a final safety net AFTER the loop.\n"
            "Should be LOOSER (larger) than target_tolerance.\n\n"
            "Default: 0.01 (1% relative error)\n"
            "Stricter: 0.001 for ultra-relativistic (γ > 1000)\n"
            "Disable: 1e10 (not recommended)\n\n"
            "Typical errors after convergence: 1e-12 to 1e-8 (rarely triggers)",
        )
        self.sc_mass_shell_tolerance_entry = ttk.Entry(
            sc_frame,
            textvariable=self.self_consistency_mass_shell_tolerance_var,
            width=16,
        )
        self.sc_mass_shell_tolerance_entry.grid(row=5, column=1, sticky="ew", pady=2)

        # Mass-shell relaxation with help icon
        relaxation_frame = ttk.Frame(sc_frame)
        relaxation_frame.grid(row=6, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_relaxation_label = ttk.Label(
            relaxation_frame, text="Relaxation weight:"
        )
        self.sc_relaxation_label.pack(side="left")
        relaxation_help = ttk.Label(
            relaxation_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        relaxation_help.pack(side="left", padx=(3, 0))
        Tooltip(
            relaxation_help,
            "Relaxation weight applied after Pt correction (both modes).\n\n"
            "Prevents oscillations by damping the correction:\n"
            "  Pt_final = α*Pt_corrected + (1-α)*Pt_old\n"
            "  where α = relaxation weight\n\n"
            "Values:\n"
            "  • 1.0 = Full correction (fastest, may oscillate)\n"
            "  • 0.7 = Recommended (default, good balance)\n"
            "  • 0.5 = Conservative (more stable, slower)\n"
            "  • 0.0 = No correction (broken, testing only)\n\n"
            "Increase for ultra-relativistic (γ > 1000) particles.\n"
            "Decrease if seeing convergence oscillations.\n\n"
            "Default: 0.7",
        )
        self.sc_relaxation_entry = ttk.Entry(
            sc_frame,
            textvariable=self.self_consistency_mass_shell_relaxation_var,
            width=16,
        )
        self.sc_relaxation_entry.grid(row=6, column=1, sticky="ew", pady=2)

        # Verbosity with help icon
        verbosity_frame = ttk.Frame(sc_frame)
        verbosity_frame.grid(row=7, column=0, sticky="w", pady=2, padx=(20, 0))
        self.sc_verbosity_label = ttk.Label(verbosity_frame, text="Verbosity:")
        self.sc_verbosity_label.pack(side="left")
        verbosity_help = ttk.Label(
            verbosity_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        verbosity_help.pack(side="left", padx=(3, 0))
        Tooltip(
            verbosity_help,
            "Self-consistency convergence diagnostic output level.\n\n"
            "⚠️ APPLIES TO ALL MODES: single runs AND sweeps/optimizations\n\n"
            "For Sweep/Optimization:\n"
            "  • This verbosity level is INHERITED when Log verbosity = 'full'\n"
            "  • Set 'Log verbosity' in Sweep/Optim tab to control overall logging\n"
            "  • 'full' mode: uses this verbosity level for SC diagnostics\n"
            "  • 'truncated' mode: minimal output regardless of this setting\n\n"
            "Output is printed to BOTH:\n"
            "  • Console (real-time during run)\n"
            "  • Saved verbose log file (*_verbose.txt)\n\n"
            "Levels:\n"
            "  • 0 = Silent (no convergence details)\n"
            "  • 1 = Summary (one line per step: converged/failed)\n"
            "  • 2 = Failures only (detailed output only for non-converged steps)\n"
            "  • 3 = Full detail (iteration-by-iteration for all steps)\n\n"
            "Recommended:\n"
            "  • Production runs: 0 or 1 (small logs)\n"
            "  • Debugging: 2 (shows only problems, moderate logs)\n"
            "  • Deep diagnostics: 3 (very large logs, 100k+ lines)\n\n"
            "Default: 0 (silent)",
        )
        self.sc_verbosity_entry = ttk.Spinbox(
            sc_frame,
            from_=0,
            to=3,
            textvariable=self.self_consistency_verbosity_var,
            width=5,
        )
        self.sc_verbosity_entry.grid(row=7, column=1, sticky="w", pady=2)

        # Chrono-match interpolation with help icon
        chrono_interp_frame = ttk.Frame(sc_frame)
        chrono_interp_frame.grid(
            row=8, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )
        self.sc_chrono_interpolate_check = ttk.Checkbutton(
            chrono_interp_frame,
            text="Enable chrono-match interpolation",
            variable=self.self_consistency_chrono_interpolate_var,
            command=self._toggle_chrono_controls,
        )
        self.sc_chrono_interpolate_check.pack(side="left")
        chrono_interp_help = ttk.Label(
            chrono_interp_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        chrono_interp_help.pack(side="left", padx=(3, 0))
        Tooltip(
            chrono_interp_help,
            "Interpolate source particle state when retarded-time residual exceeds tolerance.\n\n"
            "When computing Liénard-Wiechert fields, the code searches backward through\n"
            "the source particle trajectory to find t_ret = t_obs - R/c. With coarse\n"
            "timesteps, the 'nearest' match may have significant time residual.\n\n"
            "When enabled:\n"
            "  • Computes time residual |t_matched - t_target|\n"
            "  • If residual > tolerance, linearly interpolates source quantities\n"
            "    (velocity, acceleration, gamma) between bracketing trajectory points\n"
            "  • Provides sub-timestep accuracy for retarded fields\n\n"
            "When to enable:\n"
            "  • Large timesteps relative to 1/γ characteristic time\n"
            "  • Ultra-relativistic simulations (γ > 100)\n"
            "  • Self-consistency failures related to field discontinuities\n"
            "  • Image-charge singularities\n\n"
            "Performance impact: ~1-2% overhead (minimal)\n\n"
            "Default: OFF",
        )

        # Chrono tolerance with help icon
        chrono_tol_frame = ttk.Frame(sc_frame)
        chrono_tol_frame.grid(row=9, column=0, sticky="w", pady=2, padx=(40, 0))
        self.sc_chrono_tolerance_label = ttk.Label(
            chrono_tol_frame, text="Chrono tolerance (ns):"
        )
        self.sc_chrono_tolerance_label.pack(side="left")
        chrono_tol_help = ttk.Label(
            chrono_tol_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        chrono_tol_help.pack(side="left", padx=(3, 0))
        Tooltip(
            chrono_tol_help,
            "Time residual tolerance for chrono-matching (nanoseconds).\n\n"
            "If |t_matched - t_target| > chrono_tolerance, interpolation is applied\n"
            "(if chrono_interpolate is enabled) or a warning is issued (if verbosity >= 2).\n\n"
            "Typical values:\n"
            "  • 1e-3 ns (1 ps): Default, good for most simulations\n"
            "  • 5e-4 ns (0.5 ps): Tighter tolerance for high-precision work\n"
            "  • 1e-4 ns (0.1 ps): Very tight, for ultra-relativistic particles\n\n"
            "Rule of thumb: Set to ~0.1 × average_timestep\n\n"
            "Default: 1e-3 ns (1 picosecond)",
        )
        self.sc_chrono_tolerance_entry = ttk.Entry(
            sc_frame,
            textvariable=self.self_consistency_chrono_tolerance_var,
            width=16,
        )
        self.sc_chrono_tolerance_entry.grid(row=9, column=1, sticky="w", pady=2)

        # Advanced chrono options (high-precision mode)
        chrono_highprec_frame = ttk.Frame(sc_frame)
        chrono_highprec_frame.grid(
            row=10, column=0, columnspan=2, sticky="w", pady=2, padx=(40, 0)
        )
        self.sc_chrono_high_precision_check = ttk.Checkbutton(
            chrono_highprec_frame,
            text="High-precision mode (cubic + position interpolation)",
            variable=self.self_consistency_chrono_high_precision_var,
        )
        self.sc_chrono_high_precision_check.pack(side="left")
        chrono_highprec_help = ttk.Label(
            chrono_highprec_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        chrono_highprec_help.pack(side="left", padx=(3, 0))
        Tooltip(
            chrono_highprec_help,
            "Enable high-precision chrono-matching features.\n\n"
            "When enabled:\n"
            "  • Uses cubic (Catmull-Rom) interpolation instead of linear\n"
            "  • Interpolates particle positions (x/y/z) in addition to velocities\n"
            "  • Provides smoother derivatives for acceleration terms\n"
            "  • Better accuracy for ultra-relativistic particles (γ > 1000)\n\n"
            "Performance impact:\n"
            "  • ~3-5% overhead vs linear interpolation\n"
            "  • Requires at least 4 trajectory points for cubic fit\n\n"
            "When to enable:\n"
            "  • γ > 1000 with coarse timesteps\n"
            "  • Need smooth βdot derivatives\n"
            "  • Critical accuracy requirements\n\n"
            "Default: OFF (linear interpolation is usually sufficient)",
        )

        # Adaptive tolerance
        chrono_adaptive_frame = ttk.Frame(sc_frame)
        chrono_adaptive_frame.grid(
            row=11, column=0, columnspan=2, sticky="w", pady=2, padx=(40, 0)
        )
        self.sc_chrono_adaptive_check = ttk.Checkbutton(
            chrono_adaptive_frame,
            text="Adaptive tolerance (auto-scale with timestep)",
            variable=self.self_consistency_chrono_adaptive_tolerance_var,
        )
        self.sc_chrono_adaptive_check.pack(side="left")
        chrono_adaptive_help = ttk.Label(
            chrono_adaptive_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        chrono_adaptive_help.pack(side="left", padx=(3, 0))
        Tooltip(
            chrono_adaptive_help,
            "Automatically set chrono tolerance based on timestep.\n\n"
            "Formula: tolerance = 0.1 × timestep_h\n\n"
            "When enabled:\n"
            "  • Overrides manual chrono_tolerance setting\n"
            "  • Scales tolerance with integration timestep\n"
            "  • Useful for variable-timestep simulations\n\n"
            "Example:\n"
            "  • h = 1e-3 ns → tolerance = 1e-4 ns\n"
            "  • h = 5e-4 ns → tolerance = 5e-5 ns\n\n"
            "Default: OFF (use fixed tolerance)",
        )

        # Note: chrono_matching_mode removed from GUI
        # Always uses FAST mode for the maintained GUI path.
        # AVERAGED mode reserved for future APPROXIMATE_BACK_HISTORY implementation

        # Gamma reconciliation
        gamma_recon_frame = ttk.LabelFrame(
            sc_frame, text="Gamma Reconciliation", padding=8
        )
        gamma_recon_frame.grid(
            row=12, column=0, columnspan=2, sticky="ew", pady=2, padx=(20, 0)
        )
        gamma_recon_frame.columnconfigure(1, weight=1)

        # Method selection
        method_frame = ttk.Frame(gamma_recon_frame)
        method_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(method_frame, text="Method:").pack(side="left")
        self.sc_gamma_reconciliation_method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.self_consistency_gamma_reconciliation_method_var,
            values=[
                "DISABLED",
                "ADAPTIVE_WEIGHTED",
                "USE_VELOCITY",
                "USE_ENERGY",
                "FIXED_WEIGHTED",
            ],
            state="readonly",
            width=20,
        )
        self.sc_gamma_reconciliation_method_combo.pack(side="left", padx=(5, 5))
        method_help = ttk.Label(
            method_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        method_help.pack(side="left")
        Tooltip(
            method_help,
            "Gamma Reconciliation Method:\n\n"
            "DISABLED - No reconciliation (not recommended; may cause blowups)\n\n"
            "ADAPTIVE_WEIGHTED - Velocity-dependent weighting (recommended)\n"
            "  • β < 0.9: Trust energy (weight=0.8)\n"
            "  • β > 0.99: Trust velocity (weight=0.2)\n"
            "  • Mid-range: Balanced (weight=0.5)\n\n"
            "USE_VELOCITY - Always use γ from β (breaks energy)\n\n"
            "USE_ENERGY - Always use γ from Pt\n\n"
            "FIXED_WEIGHTED - Fixed 50/50 blend\n\n"
            "Recommended: ADAPTIVE_WEIGHTED",
        )

        # Adaptive weighted parameters
        self.sc_gamma_reconciliation_adaptive_frame = ttk.LabelFrame(
            gamma_recon_frame, text="Adaptive Weighted Parameters", padding=6
        )
        self.sc_gamma_reconciliation_adaptive_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(5, 2), padx=(10, 0)
        )
        self.sc_gamma_reconciliation_adaptive_frame.columnconfigure(1, weight=1)

        # Low beta threshold
        ttk.Label(
            self.sc_gamma_reconciliation_adaptive_frame, text="Low β threshold:"
        ).grid(row=0, column=0, sticky="w", pady=2)
        self.sc_gamma_low_beta_threshold_entry = ttk.Entry(
            self.sc_gamma_reconciliation_adaptive_frame,
            textvariable=self.self_consistency_gamma_reconciliation_low_beta_threshold_var,
            width=10,
        )
        self.sc_gamma_low_beta_threshold_entry.grid(
            row=0, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_low_beta_threshold_entry,
            "Velocity below which energy is trusted more\nDefault: 0.9",
        )

        # High beta threshold
        ttk.Label(
            self.sc_gamma_reconciliation_adaptive_frame, text="High β threshold:"
        ).grid(row=1, column=0, sticky="w", pady=2)
        self.sc_gamma_high_beta_threshold_entry = ttk.Entry(
            self.sc_gamma_reconciliation_adaptive_frame,
            textvariable=self.self_consistency_gamma_reconciliation_high_beta_threshold_var,
            width=10,
        )
        self.sc_gamma_high_beta_threshold_entry.grid(
            row=1, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_high_beta_threshold_entry,
            "Velocity above which velocity is trusted more\nDefault: 0.99",
        )

        # Low beta weight
        ttk.Label(
            self.sc_gamma_reconciliation_adaptive_frame, text="Low β weight (α):"
        ).grid(row=2, column=0, sticky="w", pady=2)
        self.sc_gamma_low_beta_weight_entry = ttk.Entry(
            self.sc_gamma_reconciliation_adaptive_frame,
            textvariable=self.self_consistency_gamma_reconciliation_low_beta_weight_var,
            width=10,
        )
        self.sc_gamma_low_beta_weight_entry.grid(
            row=2, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_low_beta_weight_entry,
            "Weight for energy when β < low threshold\n"
            "γ = α·γ_energy + (1-α)·γ_velocity\nDefault: 0.8",
        )

        # High beta weight
        ttk.Label(
            self.sc_gamma_reconciliation_adaptive_frame, text="High β weight (α):"
        ).grid(row=3, column=0, sticky="w", pady=2)
        self.sc_gamma_high_beta_weight_entry = ttk.Entry(
            self.sc_gamma_reconciliation_adaptive_frame,
            textvariable=self.self_consistency_gamma_reconciliation_high_beta_weight_var,
            width=10,
        )
        self.sc_gamma_high_beta_weight_entry.grid(
            row=3, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_high_beta_weight_entry,
            "Weight for energy when β > high threshold\n"
            "γ = α·γ_energy + (1-α)·γ_velocity\nDefault: 0.2",
        )

        # Mid beta weight
        ttk.Label(
            self.sc_gamma_reconciliation_adaptive_frame, text="Mid β weight (α):"
        ).grid(row=4, column=0, sticky="w", pady=2)
        self.sc_gamma_mid_beta_weight_entry = ttk.Entry(
            self.sc_gamma_reconciliation_adaptive_frame,
            textvariable=self.self_consistency_gamma_reconciliation_mid_beta_weight_var,
            width=10,
        )
        self.sc_gamma_mid_beta_weight_entry.grid(
            row=4, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_mid_beta_weight_entry,
            "Weight for energy in mid β range\n"
            "γ = α·γ_energy + (1-α)·γ_velocity\nDefault: 0.5",
        )

        # Fixed weighted parameter
        self.sc_gamma_reconciliation_fixed_frame = ttk.LabelFrame(
            gamma_recon_frame, text="Fixed Weighted Parameter", padding=6
        )
        self.sc_gamma_reconciliation_fixed_frame.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(5, 2), padx=(10, 0)
        )
        self.sc_gamma_reconciliation_fixed_frame.columnconfigure(1, weight=1)

        ttk.Label(
            self.sc_gamma_reconciliation_fixed_frame, text="Fixed weight (α):"
        ).grid(row=0, column=0, sticky="w", pady=2)
        self.sc_gamma_fixed_weight_entry = ttk.Entry(
            self.sc_gamma_reconciliation_fixed_frame,
            textvariable=self.self_consistency_gamma_reconciliation_fixed_weight_var,
            width=10,
        )
        self.sc_gamma_fixed_weight_entry.grid(
            row=0, column=1, sticky="w", padx=(5, 0), pady=2
        )
        Tooltip(
            self.sc_gamma_fixed_weight_entry,
            "Fixed weight for FIXED_WEIGHTED method\n"
            "γ = α·γ_energy + (1-α)·γ_velocity\nDefault: 0.5",
        )

        # Trace method change to toggle parameter visibility
        self.self_consistency_gamma_reconciliation_method_var.trace_add(
            "write", lambda *_: self._toggle_gamma_reconciliation_params()
        )
        self._toggle_gamma_reconciliation_params()

        # Adaptive timestep section (Energy Jump Detection functionality integrated here)
        at_frame = ttk.LabelFrame(
            stability_frame, text="Adaptive Timestep Refinement", padding=8
        )
        at_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        at_frame.columnconfigure(1, weight=1)

        self.adaptive_enable_check = ttk.Checkbutton(
            at_frame,
            text="Enable adaptive timestep (auto-refine on energy jumps)",
            variable=self.adaptive_timestep_enabled_var,
            command=self._toggle_adaptive_timestep_controls,
        )
        self.adaptive_enable_check.grid(
            row=0, column=0, columnspan=2, sticky="w", pady=2
        )

        # Energy jump threshold with help icon
        ejt_frame = ttk.Frame(at_frame)
        ejt_frame.grid(row=1, column=0, sticky="w", pady=2, padx=(20, 0))
        self.adaptive_threshold_label = ttk.Label(
            ejt_frame, text="Energy jump threshold:"
        )
        self.adaptive_threshold_label.pack(side="left")
        ejt_help = ttk.Label(ejt_frame, text="ⓘ", foreground="blue", cursor="hand2")
        ejt_help.pack(side="left", padx=(3, 0))
        Tooltip(
            ejt_help,
            "Fractional energy change triggering timestep reduction.\n\n"
            "When |ΔE/E| > threshold:\n"
            "  1. Reject step\n"
            "  2. Reduce timestep by reduction factor\n"
            "  3. Retry step with smaller timestep\n"
            "  4. Repeat until energy change < threshold\n\n"
            "Default: 0.10 (10% energy change)\n"
            "Stricter: 0.05 for smooth energy evolution\n"
            "Looser: 0.20 for exploratory runs\n\n"
            "Lower = more refinements = slower but smoother",
        )
        self.adaptive_threshold_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_threshold_var, width=16
        )
        self.adaptive_threshold_entry.grid(row=1, column=1, sticky="ew", pady=2)

        # Timestep reduction factor with help icon
        red_frame = ttk.Frame(at_frame)
        red_frame.grid(row=2, column=0, sticky="w", pady=2, padx=(20, 0))
        self.adaptive_reduction_label = ttk.Label(
            red_frame, text="Timestep reduction factor:"
        )
        self.adaptive_reduction_label.pack(side="left")
        red_help = ttk.Label(red_frame, text="ⓘ", foreground="blue", cursor="hand2")
        red_help.pack(side="left", padx=(3, 0))
        Tooltip(
            red_help,
            "Timestep reduction on energy jump detection.\n\n"
            "new_timestep = current_timestep / reduction_factor\n\n"
            "Default: 10 (reduce by 10x)\n"
            "Aggressive: 100 for severe jumps/narrow apertures\n"
            "Conservative: 5 for gentle refinement\n\n"
            "Higher = more aggressive reduction = finer resolution\n"
            "Example: h=1e-5, factor=10 → h_new=1e-6",
        )
        self.adaptive_reduction_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_reduction_factor_var, width=16
        )
        self.adaptive_reduction_entry.grid(row=2, column=1, sticky="ew", pady=2)

        # Add trace to update max_attempts display when reduction_factor changes
        self.adaptive_timestep_reduction_factor_var.trace_add(
            "write", lambda *args: self._update_max_attempts_display()
        )

        # Max refinement attempts with help icon (calculated, read-only)
        att_frame = ttk.Frame(at_frame)
        att_frame.grid(row=3, column=0, sticky="w", pady=2, padx=(20, 0))
        self.adaptive_max_attempts_label = ttk.Label(
            att_frame, text="Max refinement attempts (calculated):"
        )
        self.adaptive_max_attempts_label.pack(side="left")
        att_help = ttk.Label(att_frame, text="ⓘ", foreground="blue", cursor="hand2")
        att_help.pack(side="left", padx=(3, 0))
        Tooltip(
            att_help,
            "Maximum timestep reductions per step (READ-ONLY).\n\n"
            "Auto-calculated to be consistent with reduction_factor and min_timestep_factor:\n\n"
            "  max_attempts = ceil(log(1/min_factor) / log(reduction_factor))\n\n"
            "Examples:\n"
            "  • reduction_factor=3, min_factor=1e-4 → max_attempts = 9\n"
            "  • reduction_factor=10, min_factor=1e-4 → max_attempts = 4\n\n"
            "Why automatic?\n"
            "After n attempts: h_final = h_base / (reduction_factor^n)\n"
            "At minimum: h_min = h_base × min_timestep_factor\n"
            "These must be consistent!\n\n"
            "To change: adjust 'Reduction factor' or 'Min timestep factor' above.",
        )
        # Create read-only display label for max_attempts (calculated value)
        self.adaptive_max_attempts_display = ttk.Label(
            at_frame,
            textvariable=self.adaptive_timestep_max_attempts_display_var,
            relief="sunken",
            background="#f0f0f0",
            foreground="#606060",
            padding=(5, 2),
            font=("TkDefaultFont", 9, "italic"),
        )
        self.adaptive_max_attempts_display.grid(row=3, column=1, sticky="ew", pady=2)

        # Min timestep factor with help icon
        min_frame = ttk.Frame(at_frame)
        min_frame.grid(row=4, column=0, sticky="w", pady=2, padx=(20, 0))
        self.adaptive_min_factor_label = ttk.Label(
            min_frame, text="Min timestep factor:"
        )
        self.adaptive_min_factor_label.pack(side="left")
        min_help = ttk.Label(min_frame, text="ⓘ", foreground="blue", cursor="hand2")
        min_help.pack(side="left", padx=(3, 0))
        Tooltip(
            min_help,
            "Minimum timestep as fraction of original.\n\n"
            "h_min = h_initial × min_factor\n"
            "Prevents infinitesimal timesteps (runaway refinement).\n\n"
            "If h_min reached and energy jump persists → fail step.\n\n"
            "Default: 1e-4 (0.01% of original)\n"
            "Allow extreme: 1e-6 for ultra-narrow apertures\n"
            "Fail faster: 1e-3 to detect bad setups early\n\n"
            "Example: h_initial=1e-5, factor=1e-4\n"
            "  → h_min = 1e-9 (minimum allowed)",
        )
        self.adaptive_min_factor_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_min_factor_var, width=16
        )
        self.adaptive_min_factor_entry.grid(row=4, column=1, sticky="ew", pady=2)

        # Add traces to update calculated displays when min_factor changes
        self.adaptive_timestep_min_factor_var.trace_add(
            "write",
            lambda *args: (
                self._update_max_attempts_display(),
                self._update_max_substeps_display(),
            ),
        )

        # Hysteresis parameters
        cd_frame = ttk.Frame(at_frame)
        cd_frame.grid(row=5, column=0, sticky="w", pady=2)
        self.adaptive_cooldown_label = ttk.Label(cd_frame, text="Cooldown steps:")
        self.adaptive_cooldown_label.pack(side="left")
        cd_help = ttk.Label(cd_frame, text="ⓘ", foreground="blue", cursor="hand2")
        cd_help.pack(side="left", padx=(3, 0))
        Tooltip(
            cd_help,
            "Steps to remain on reduced timestep after refinement.\n\n"
            "HYSTERESIS: Prevents oscillation\n"
            "  refine → restore → refine → restore (bad!)\n\n"
            "After refinement:\n"
            "  1. Stay on small h for cooldown steps\n"
            "  2. Then probe if safe to restore\n\n"
            "Default: 10 steps\n"
            "Cautious: 20 for unstable/boundary regions\n"
            "Aggressive: 5 to restore quickly",
        )
        self.adaptive_cooldown_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_cooldown_steps_var, width=16
        )
        self.adaptive_cooldown_entry.grid(row=5, column=1, sticky="ew", pady=2)

        pt_frame = ttk.Frame(at_frame)
        pt_frame.grid(row=6, column=0, sticky="w", pady=2)
        self.adaptive_probe_threshold_label = ttk.Label(
            pt_frame, text="Probe threshold:"
        )
        self.adaptive_probe_threshold_label.pack(side="left")
        pt_help = ttk.Label(pt_frame, text="ⓘ", foreground="blue", cursor="hand2")
        pt_help.pack(side="left", padx=(3, 0))
        Tooltip(
            pt_help,
            "Energy threshold for testing timestep restoration.\n\n"
            "After cooldown, probe if safe to increase h:\n"
            "  If |ΔE/E| < probe_threshold for N consecutive steps\n"
            "  → begin restoring h toward original\n\n"
            "CRITICAL: Must be < energy jump threshold!\n"
            "  Otherwise: restore → jump → refine → restore (oscillation)\n\n"
            "Default: 0.01 (1% change, 10× below jump threshold)\n"
            "Cautious: 0.001 for slower restoration\n"
            "Faster: 0.05 (ensure jump threshold ≥ 0.10)",
        )
        self.adaptive_probe_threshold_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_probe_threshold_var, width=16
        )
        self.adaptive_probe_threshold_entry.grid(row=6, column=1, sticky="ew", pady=2)

        mp_frame = ttk.Frame(at_frame)
        mp_frame.grid(row=7, column=0, sticky="w", pady=2)
        self.adaptive_max_probe_label = ttk.Label(mp_frame, text="Max probe steps:")
        self.adaptive_max_probe_label.pack(side="left")
        mp_help = ttk.Label(mp_frame, text="ⓘ", foreground="blue", cursor="hand2")
        mp_help.pack(side="left", padx=(3, 0))
        Tooltip(
            mp_help,
            "Consecutive 'good' steps needed to restore timestep.\n\n"
            "After cooldown, requires N consecutive steps with\n"
            "|ΔE/E| < probe_threshold before increasing h.\n\n"
            "Prevents premature restoration from one lucky step.\n\n"
            "Default: 3 steps\n"
            "Very cautious: 5-10 (verify stability thoroughly)\n"
            "Quick restore: 1-2 (risky if region is unstable)",
        )
        self.adaptive_max_probe_entry = ttk.Entry(
            at_frame, textvariable=self.adaptive_timestep_max_probe_steps_var, width=16
        )
        self.adaptive_max_probe_entry.grid(row=7, column=1, sticky="ew", pady=2)

        # Halt on jump option (migrated from removed Energy Jump Detection)
        self.adaptive_halt_check = ttk.Checkbutton(
            at_frame,
            text="Halt simulation on energy jump",
            variable=self.adaptive_timestep_halt_on_jump_var,
        )
        self.adaptive_halt_check.grid(
            row=8, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )

        self.adaptive_debug_check = ttk.Checkbutton(
            at_frame,
            text="Verbose output (inherited by sweep/optim when Log verbosity = 'full')",
            variable=self.adaptive_timestep_debug_var,
        )
        self.adaptive_debug_check.grid(
            row=9, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )

        # Max sub-steps limit
        max_substeps_frame = ttk.Frame(at_frame)
        max_substeps_frame.grid(row=10, column=0, sticky="w", pady=2, padx=(20, 0))
        self.adaptive_max_substeps_label = ttk.Label(
            max_substeps_frame, text="Max sub-steps (calculated):"
        )
        self.adaptive_max_substeps_label.pack(side="left")
        max_substeps_help = ttk.Label(
            max_substeps_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        max_substeps_help.pack(side="left", padx=(3, 0))
        Tooltip(
            max_substeps_help,
            "Maximum number of sub-steps per main step (READ-ONLY).\n\n"
            "This value is automatically calculated from min_timestep_factor\n"
            "to prevent time discontinuities:\n\n"
            "  max_substeps = ceil(1 / min_timestep_factor) × 1.1\n\n"
            "The 1.1× safety margin ensures coverage even with rounding.\n\n"
            "Example:\n"
            "  • min_timestep_factor = 1e-4\n"
            "  • Theoretical max = ceil(1 / 1e-4) = 10,000\n"
            "  • With margin = 10,000 × 1.1 = 11,000 substeps\n\n"
            "Why automatic?\n"
            "If min_timestep_factor allows timestep to reduce to h × 1e-4,\n"
            "then at minimum timestep you need 1/1e-4 = 10,000 substeps\n"
            "to cover the full base timestep interval.\n\n"
            "Setting max_substeps lower than this would create time\n"
            "discontinuities where some time is skipped!\n\n"
            "To change this value: adjust 'Min timestep factor' above.",
        )

        # Create read-only display label for max_substeps (calculated value)
        self.adaptive_max_substeps_display = ttk.Label(
            at_frame,
            textvariable=self.adaptive_timestep_max_substeps_display_var,
            relief="sunken",
            background="#f0f0f0",
            foreground="#606060",
            padding=(5, 2),
            font=("TkDefaultFont", 9, "italic"),
        )
        self.adaptive_max_substeps_display.grid(
            row=10, column=1, sticky="w", pady=2, padx=(10, 0)
        )

        # Help text removed - was obscuring Adaptive Timestep Refinement section
        # All parameter help is now available via ⓘ tooltips

        # Initialize control states
        self._toggle_self_consistency_controls()
        self._toggle_chrono_controls()
        self._toggle_adaptive_timestep_controls()

