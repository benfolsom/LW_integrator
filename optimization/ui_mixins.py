"""UI helper mixins used by :class:`OptimizationPlugin`."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from optimization.ui_helpers import ToolTip


class OptimizationPluginUIMixin:
    """Builds and manages the Tkinter UI for the optimization plugin."""

    def _build_ui(self):
        """Build the user interface."""
        # Main container with scrollbar
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

        def _bind_to_mousewheel(event):
            # Bind mousewheel for Windows/Mac
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
            # Bind mousewheel for Linux
            self.canvas.bind_all("<Button-4>", _on_mousewheel_linux)
            self.canvas.bind_all("<Button-5>", _on_mousewheel_linux)

        def _unbind_from_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

        # Bind/unbind on enter/leave
        self.canvas.bind("<Enter>", _bind_to_mousewheel)
        self.canvas.bind("<Leave>", _unbind_from_mousewheel)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Build sections
        self._build_simulation_section()
        self._build_mode_section()
        self._build_parameter_section()
        self._build_objective_section()
        self._build_optimization_section()
        self._build_control_section()
        self._build_results_output_section()
        self._build_progress_section()

        # Initialize mode visibility
        self._update_mode_visibility()

    def _add_tooltip(self, widget, text):
        """Add a tooltip to a widget.

        Args:
            widget: The tkinter widget to add tooltip to
            text: The tooltip text to display
        """
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

        # Store reference for updating visibility
        self.sim_type_frame = frame

    def _build_mode_section(self):
        """Build mode selection section (blind sweep vs optimization)."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Run Mode", padding=10)
        frame.pack(fill="x", padx=10, pady=5)
        self.mode_section_frame = frame

        # Mode selection
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

        # Help text
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

    def _build_parameter_section(self):
        """Build parameter range specification section."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Parameter Ranges", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)
        self.parameter_frame = frame

        # Add explanatory help text
        help_frame = ttk.Frame(frame)
        help_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        help_text = (
            "Coordinate system: Particles start at z-coordinate and travel toward the conducting wall.\n"
            "Example: Particle at z=0 travels to wall at z=2200 mm (distance = 2200 mm).\n"
            "Transverse offset: Fraction of aperture radius (0.0 = on-axis, 1.0 = at aperture edge)."
        )
        help_label = ttk.Label(
            help_frame, text=help_text, foreground="gray40", font=("TkDefaultFont", 8)
        )
        help_label.pack(anchor="w")

        # Aperture range
        ttk.Label(frame, text="Aperture Radius:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        aperture_frame = ttk.Frame(frame)
        aperture_frame.grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)

        ttk.Label(aperture_frame, text="Min (mm):").pack(side="left", padx=(0, 2))
        self.aperture_min_var = tk.StringVar(value="1e-5")
        ttk.Entry(aperture_frame, textvariable=self.aperture_min_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(aperture_frame, text="Max (mm):").pack(side="left", padx=(10, 2))
        self.aperture_max_var = tk.StringVar(value="1e-3")
        ttk.Entry(aperture_frame, textvariable=self.aperture_max_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(aperture_frame, text="Points:").pack(side="left", padx=(10, 2))
        self.aperture_points_var = tk.StringVar(value="10")
        ttk.Entry(aperture_frame, textvariable=self.aperture_points_var, width=5).pack(
            side="left", padx=2
        )

        self.aperture_log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            aperture_frame, text="Log scale", variable=self.aperture_log_var
        ).pack(side="left", padx=(10, 0))

        # Energy range
        ttk.Label(frame, text="Particle Energy:").grid(
            row=2, column=0, sticky="w", pady=2
        )
        energy_frame = ttk.Frame(frame)
        energy_frame.grid(row=2, column=1, columnspan=3, sticky="ew", pady=2)

        ttk.Label(energy_frame, text="Min (GeV):").pack(side="left", padx=(0, 2))
        self.energy_min_var = tk.StringVar(value="1.0")
        ttk.Entry(energy_frame, textvariable=self.energy_min_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(energy_frame, text="Max (GeV):").pack(side="left", padx=(10, 2))
        self.energy_max_var = tk.StringVar(value="1000.0")
        ttk.Entry(energy_frame, textvariable=self.energy_max_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(energy_frame, text="Points:").pack(side="left", padx=(10, 2))
        self.energy_points_var = tk.StringVar(value="10")
        ttk.Entry(energy_frame, textvariable=self.energy_points_var, width=5).pack(
            side="left", padx=2
        )

        self.energy_log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            energy_frame, text="Log scale", variable=self.energy_log_var
        ).pack(side="left", padx=(10, 0))

        # Transverse offset fractions
        ttk.Label(frame, text="Transverse Offset:").grid(
            row=3, column=0, sticky="w", pady=2
        )
        ttk.Label(frame, text="Fractions of aperture (comma-separated):").grid(
            row=3, column=1, sticky="w", pady=2
        )
        self.offset_fractions_var = tk.StringVar(value="0.0")
        ttk.Entry(frame, textvariable=self.offset_fractions_var, width=30).grid(
            row=3, column=2, columnspan=2, sticky="ew", pady=2, padx=5
        )

        # Starting z positions
        ttk.Label(frame, text="Starting Positions:").grid(
            row=4, column=0, sticky="w", pady=2
        )
        ttk.Label(frame, text="Particle z-coordinate (mm, comma-separated):").grid(
            row=4, column=1, sticky="w", pady=2
        )
        self.start_z_var = tk.StringVar(value="0.0")
        ttk.Entry(frame, textvariable=self.start_z_var, width=30).grid(
            row=4, column=2, columnspan=2, sticky="ew", pady=2, padx=5
        )
        # Wall Position (sweepable)
        ttk.Label(frame, text="Wall Position:").grid(
            row=5, column=0, sticky="w", pady=2
        )

        # Fixed value and sweep checkbox on same row
        wall_z_fixed_frame = ttk.Frame(frame)
        wall_z_fixed_frame.grid(row=5, column=1, columnspan=3, sticky="w", pady=2)

        self.wall_z_var = tk.StringVar(value="2200.0")
        self.wall_z_entry = ttk.Entry(
            wall_z_fixed_frame, textvariable=self.wall_z_var, width=10
        )
        self.wall_z_entry.pack(side="left", padx=(0, 10))

        self.wall_z_sweep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            wall_z_fixed_frame,
            text="Sweep",
            variable=self.wall_z_sweep_var,
            command=self._toggle_wall_z_sweep,
        ).pack(side="left")

        # Sweep controls on new row for better visibility
        wall_z_sweep_frame = ttk.Frame(frame)
        wall_z_sweep_frame.grid(
            row=6, column=1, columnspan=3, sticky="w", pady=2, padx=(20, 0)
        )

        ttk.Label(wall_z_sweep_frame, text="Min:").pack(side="left", padx=(0, 2))
        self.wall_z_min_var = tk.StringVar(value="2000.0")
        self.wall_z_min_entry = ttk.Entry(
            wall_z_sweep_frame, textvariable=self.wall_z_min_var, width=8
        )
        self.wall_z_min_entry.pack(side="left", padx=2)

        ttk.Label(wall_z_sweep_frame, text="Max:").pack(side="left", padx=(5, 2))
        self.wall_z_max_var = tk.StringVar(value="2400.0")
        self.wall_z_max_entry = ttk.Entry(
            wall_z_sweep_frame, textvariable=self.wall_z_max_var, width=8
        )
        self.wall_z_max_entry.pack(side="left", padx=2)

        ttk.Label(wall_z_sweep_frame, text="Pts:").pack(side="left", padx=(5, 2))
        self.wall_z_points_var = tk.StringVar(value="3")
        self.wall_z_points_entry = ttk.Entry(
            wall_z_sweep_frame, textvariable=self.wall_z_points_var, width=4
        )
        self.wall_z_points_entry.pack(side="left", padx=2)

        self.wall_z_log_var = tk.BooleanVar(value=False)
        self.wall_z_log_check = ttk.Checkbutton(
            wall_z_sweep_frame, text="Log", variable=self.wall_z_log_var
        )
        self.wall_z_log_check.pack(side="left", padx=(5, 0))

        # Store references for sweep control
        self.wall_z_sweep_frame = wall_z_sweep_frame
        self.wall_z_sweep_widgets = [
            self.wall_z_min_entry,
            self.wall_z_max_entry,
            self.wall_z_points_entry,
            self.wall_z_log_check,
        ]

        # Initially disable sweep controls
        self._toggle_wall_z_sweep()

        # Cavity Spacing (for SWITCHING_WALL)
        ttk.Label(frame, text="Cavity Spacing:").grid(
            row=7, column=0, sticky="w", pady=2
        )
        ttk.Label(frame, text="SWITCHING_WALL only (mm):").grid(
            row=7, column=1, sticky="w", pady=2
        )
        self.cavity_spacing_var = tk.StringVar(value="1e5")
        self.cavity_spacing_entry = ttk.Entry(
            frame, textvariable=self.cavity_spacing_var, width=10
        )
        self.cavity_spacing_entry.grid(row=7, column=2, sticky="w", pady=2, padx=5)

        # Timestep Auto-Calculation (always uses auto_distance strategy)
        timestep_label = ttk.Label(frame, text="Timestep Calculation:")
        timestep_label.grid(row=8, column=0, sticky="w", pady=2)

        # Add tooltip for explanatory note
        self._add_tooltip(
            timestep_label,
            "All runs travel to wall_z + target distance regardless of energy.\n"
            "This ensures consistent trajectory length across different energies.",
        )

        timestep_frame = ttk.Frame(frame)
        timestep_frame.grid(row=8, column=1, columnspan=3, sticky="ew", pady=2)

        self.timestep_mode_var = tk.StringVar(value="duration")
        ttk.Radiobutton(
            timestep_frame,
            text="Auto-calc duration, provide count:",
            variable=self.timestep_mode_var,
            value="duration",
            command=self._toggle_timestep_mode,
        ).pack(side="left", padx=(0, 5))

        self.steps_var = tk.StringVar(value="500")
        self.steps_entry = ttk.Entry(
            timestep_frame, textvariable=self.steps_var, width=8
        )
        self.steps_entry.pack(side="left", padx=2)
        ttk.Label(timestep_frame, text="steps").pack(side="left", padx=(2, 15))

        ttk.Radiobutton(
            timestep_frame,
            text="Auto-calc count, provide duration:",
            variable=self.timestep_mode_var,
            value="count",
            command=self._toggle_timestep_mode,
        ).pack(side="left", padx=(0, 5))

        self.duration_var = tk.StringVar(value="1e-3")
        self.duration_entry = ttk.Entry(
            timestep_frame, textvariable=self.duration_var, width=10
        )
        self.duration_entry.pack(side="left", padx=2)
        ttk.Label(timestep_frame, text="ns (proper time)").pack(side="left", padx=2)

        # Distance target
        ttk.Label(frame, text="Distance Target:").grid(
            row=9, column=0, sticky="w", pady=2
        )
        distance_frame = ttk.Frame(frame)
        distance_frame.grid(row=9, column=1, columnspan=3, sticky="ew", pady=2)
        ttk.Label(distance_frame, text="Target: wall +").pack(side="left", padx=(0, 2))
        self.auto_steps_distance_var = tk.StringVar(value="10.0")
        ttk.Entry(
            distance_frame, textvariable=self.auto_steps_distance_var, width=6
        ).pack(side="left", padx=2)
        ttk.Label(distance_frame, text="mm (min 5% of steps enforced)").pack(
            side="left", padx=2
        )

        # Note about trajectory and output configuration
        config_note = ttk.Label(
            frame,
            text="ℹ For trajectory saving and output options, see the 'Results & Output Configuration' section below",
            font=("TkDefaultFont", 8, "italic"),
            foreground="blue",
            justify="left",
        )
        config_note.grid(row=10, column=0, columnspan=4, sticky="w", pady=(10, 10))

        frame.columnconfigure(2, weight=1)

        # Initialize timestep mode state
        self._toggle_timestep_mode()

    def _build_particle_section(self):
        """Build rider and driver particle parameters sections with optional sweeping."""
        # Store sweep control variables (shared by rider and driver)
        self.sweep_params = {}

        # Rider particle parameters
        self._build_rider_particle_section()

        # Driver particle parameters (shown/hidden based on simulation type)
        self._build_driver_particle_section()

    def _build_rider_particle_section(self):
        """Build rider particle parameters section with optional sweeping."""
        frame = ttk.LabelFrame(
            self.scrollable_frame,
            text="Rider Particle Parameters (optionally sweepable)",
            padding=10,
        )
        frame.pack(fill="x", padx=10, pady=5)

        row = 0

        # Particle Mass
        self._add_sweepable_param(
            frame,
            row,
            "rider_m_particle",
            "Particle Mass (amu):",
            "0.00054857990907",
            width=15,
        )
        row += 1

        # Charge Sign
        self._add_sweepable_param(
            frame, row, "rider_charge_sign", "Charge Sign:", "-1.0", width=10
        )
        row += 1

        # Particle Count
        self._add_sweepable_param(
            frame, row, "rider_pcount", "Particle Count:", "1", width=10
        )
        row += 1

        # Transverse Momentum (spread, uniform ±)
        self._add_sweepable_param(
            frame,
            row,
            "rider_transv_mom",
            "Transverse Momentum (amu·mm/ns, spread ±):",
            "1.2e-05",
            width=15,
        )
        row += 1

        # Transverse Spread (bunch radius / half-width)
        self._add_sweepable_param(
            frame,
            row,
            "rider_transv_dist",
            "Transverse Spread (mm, half-width):",
            "2e-06",
            width=15,
        )
        row += 1

        # Stripped Ions (not sweepable, always fixed)
        ttk.Label(frame, text="Stripped Ions:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.rider_stripped_ions_var = tk.StringVar(value="1.0")
        ttk.Entry(frame, textvariable=self.rider_stripped_ions_var, width=10).grid(
            row=row, column=1, sticky="w", pady=2, padx=5
        )
        row += 1

        # Timestep from main config (display only)
        ttk.Label(frame, text="Timestep from main config (ns):").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.main_timestep_display_var = tk.StringVar(value="3e-7")
        ttk.Label(frame, textvariable=self.main_timestep_display_var).grid(
            row=row, column=1, sticky="w", pady=2, padx=5
        )
        row += 1

        # Info label
        info_label = ttk.Label(
            frame,
            text="Check 'Sweep' to enable range controls. Energy and position are swept by default.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        )
        info_label.grid(row=row, column=0, columnspan=6, sticky="w", pady=(5, 0))

        # Macroparticle simulation section
        row += 1
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=6, sticky="ew", pady=(10, 10)
        )
        row += 1

        ttk.Label(
            frame,
            text="Macroparticle Simulation (Conducting Wall only):",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=row, column=0, columnspan=6, sticky="w", pady=(0, 5))
        row += 1

        self.macroparticle_enabled_var = tk.BooleanVar(value=False)
        self.macroparticle_enable_check = ttk.Checkbutton(
            frame,
            text="Enable macroparticle simulation (bunch spread inherited from above)",
            variable=self.macroparticle_enabled_var,
            command=self._toggle_macroparticle_controls,
        )
        self.macroparticle_enable_check.grid(
            row=row, column=0, columnspan=6, sticky="w", pady=2
        )
        row += 1

        # Charge multiplier (sweepable)
        # Add with indented label text
        ttk.Label(frame, text="    Charge multiplier:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        charge_var = tk.StringVar(value="1.0")
        charge_entry = ttk.Entry(frame, textvariable=charge_var, width=15)
        charge_entry.grid(row=row, column=1, sticky="w", pady=2, padx=5)

        charge_sweep_var = tk.BooleanVar(value=False)
        charge_sweep_cb = ttk.Checkbutton(
            frame,
            text="Sweep:",
            variable=charge_sweep_var,
            command=lambda: self._toggle_sweep_controls(
                "macroparticle_charge_multiplier"
            ),
        )
        charge_sweep_cb.grid(row=row, column=2, sticky="w", pady=2, padx=(10, 2))

        charge_range_frame = ttk.Frame(frame)
        charge_range_frame.grid(row=row, column=3, columnspan=3, sticky="w", pady=2)
        charge_range_frame.grid_remove()

        ttk.Label(charge_range_frame, text="Min:").pack(side="left", padx=(0, 2))
        charge_min_var = tk.StringVar(value="1.0")
        ttk.Entry(charge_range_frame, textvariable=charge_min_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(charge_range_frame, text="Max:").pack(side="left", padx=(5, 2))
        charge_max_var = tk.StringVar(value="1.0")
        ttk.Entry(charge_range_frame, textvariable=charge_max_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(charge_range_frame, text="Pts:").pack(side="left", padx=(5, 2))
        charge_points_var = tk.StringVar(value="3")
        ttk.Entry(charge_range_frame, textvariable=charge_points_var, width=4).pack(
            side="left", padx=2
        )

        charge_log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(charge_range_frame, text="Log", variable=charge_log_var).pack(
            side="left", padx=(5, 0)
        )

        self.sweep_params["macroparticle_charge_multiplier"] = {
            "fixed_var": charge_var,
            "fixed_entry": charge_entry,
            "sweep_var": charge_sweep_var,
            "range_frame": charge_range_frame,
            "min_var": charge_min_var,
            "max_var": charge_max_var,
            "points_var": charge_points_var,
            "log_var": charge_log_var,
        }
        row += 1

        # Sigma multiplier for image charge errors (sweepable)
        # Add with indented label text
        ttk.Label(frame, text="    Image error sigma multiplier:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        sigma_var = tk.StringVar(value="1.0")
        sigma_entry = ttk.Entry(frame, textvariable=sigma_var, width=15)
        sigma_entry.grid(row=row, column=1, sticky="w", pady=2, padx=5)

        sigma_sweep_var = tk.BooleanVar(value=False)
        sigma_sweep_cb = ttk.Checkbutton(
            frame,
            text="Sweep:",
            variable=sigma_sweep_var,
            command=lambda: self._toggle_sweep_controls(
                "macroparticle_sigma_multiplier"
            ),
        )
        sigma_sweep_cb.grid(row=row, column=2, sticky="w", pady=2, padx=(10, 2))

        sigma_range_frame = ttk.Frame(frame)
        sigma_range_frame.grid(row=row, column=3, columnspan=3, sticky="w", pady=2)
        sigma_range_frame.grid_remove()

        ttk.Label(sigma_range_frame, text="Min:").pack(side="left", padx=(0, 2))
        sigma_min_var = tk.StringVar(value="1.0")
        ttk.Entry(sigma_range_frame, textvariable=sigma_min_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(sigma_range_frame, text="Max:").pack(side="left", padx=(5, 2))
        sigma_max_var = tk.StringVar(value="1.0")
        ttk.Entry(sigma_range_frame, textvariable=sigma_max_var, width=10).pack(
            side="left", padx=2
        )

        ttk.Label(sigma_range_frame, text="Pts:").pack(side="left", padx=(5, 2))
        sigma_points_var = tk.StringVar(value="3")
        ttk.Entry(sigma_range_frame, textvariable=sigma_points_var, width=4).pack(
            side="left", padx=2
        )

        sigma_log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sigma_range_frame, text="Log", variable=sigma_log_var).pack(
            side="left", padx=(5, 0)
        )

        self.sweep_params["macroparticle_sigma_multiplier"] = {
            "fixed_var": sigma_var,
            "fixed_entry": sigma_entry,
            "sweep_var": sigma_sweep_var,
            "range_frame": sigma_range_frame,
            "min_var": sigma_min_var,
            "max_var": sigma_max_var,
            "points_var": sigma_points_var,
            "log_var": sigma_log_var,
        }
        row += 1

        # Include momentum errors checkbox
        self.macroparticle_momentum_errors_var = tk.BooleanVar(value=True)
        self.macroparticle_momentum_errors_check = ttk.Checkbutton(
            frame,
            text="Include momentum errors (cumulative)",
            variable=self.macroparticle_momentum_errors_var,
        )
        self.macroparticle_momentum_errors_check.grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )
        row += 1

        # Help text
        help_label = ttk.Label(
            frame,
            text=(
                "Macroparticle mode scales test particle charge and adds Gaussian errors to image subcharges.\n"
                "Image errors derived from bunch spread (transv_dist, transv_mom) × sigma multiplier.\n"
                "Position errors: constant σ from transv_dist. Momentum errors: cumulative from transv_mom.\n"
                "Uncheck 'Include momentum errors' to apply only constant position errors (no cumulative growth).\n"
                "Only for CONDUCTING_WALL."
            ),
            font=("TkDefaultFont", 8),
            foreground="gray40",
            justify="left",
        )
        help_label.grid(
            row=row, column=0, columnspan=6, sticky="w", pady=(0, 2), padx=(20, 0)
        )

        # Store macroparticle widgets for enable/disable
        self._macroparticle_widgets = [
            self.macroparticle_momentum_errors_check,
            self.sweep_params["macroparticle_charge_multiplier"]["fixed_entry"],
            self.sweep_params["macroparticle_sigma_multiplier"]["fixed_entry"],
        ]
        # Store sweep control references separately for conditional disabling
        self._macroparticle_sweep_controls = [
            self.sweep_params["macroparticle_charge_multiplier"],
            self.sweep_params["macroparticle_sigma_multiplier"],
        ]

    def _build_driver_particle_section(self):
        """Build driver particle parameters section with optional sweeping."""
        self.driver_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text="Driver Particle Parameters (optionally sweepable)",
            padding=10,
        )
        self.driver_frame.pack(fill="x", padx=10, pady=5)

        row = 0

        # Particle Mass
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_m_particle",
            "Particle Mass (amu):",
            "207.2",
            width=15,
        )
        row += 1

        # Charge Sign
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_charge_sign",
            "Charge Sign:",
            "1.0",
            width=10,
        )
        row += 1

        # Particle Count
        self._add_sweepable_param(
            self.driver_frame, row, "driver_pcount", "Particle Count:", "5", width=10
        )
        row += 1

        # Transverse Momentum
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_transv_mom",
            "Transverse Momentum (amu·mm/ns):",
            "0.0",
            width=15,
        )
        row += 1

        # Transverse Distance
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_transv_dist",
            "Transverse Distance (mm):",
            "-0.07998",
            width=15,
        )
        row += 1

        # Starting Distance (driver-specific)
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_starting_distance",
            "Starting Distance (mm):",
            "1000.0",
            width=15,
        )
        row += 1

        # Starting Pz (driver-specific)
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_starting_Pz",
            "Starting Pz (amu·mm/ns):",
            "-4925.0",
            width=15,
        )
        row += 1

        # Stripped Ions (not sweepable, always fixed)
        ttk.Label(self.driver_frame, text="Stripped Ions:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.driver_stripped_ions_var = tk.StringVar(value="54.0")
        ttk.Entry(
            self.driver_frame, textvariable=self.driver_stripped_ions_var, width=10
        ).grid(row=row, column=1, sticky="w", pady=2, padx=5)
        row += 1

        # Info label
        info_label = ttk.Label(
            self.driver_frame,
            text="Driver parameters only used for BUNCH_TO_BUNCH simulations.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        )
        info_label.grid(row=row, column=0, columnspan=6, sticky="w", pady=(5, 0))

        # Hide driver section initially (shown for BUNCH_TO_BUNCH type)
        self._update_driver_visibility()

    def _add_sweepable_param(
        self, parent, row, param_name, label, default_value, width=15
    ):
        """Add a parameter row with optional sweep controls."""
        # Label
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)

        # Fixed value entry
        var = tk.StringVar(value=default_value)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="w", pady=2, padx=5)

        # Sweep checkbox
        sweep_var = tk.BooleanVar(value=False)
        sweep_cb = ttk.Checkbutton(
            parent,
            text="Sweep:",
            variable=sweep_var,
            command=lambda: self._toggle_sweep_controls(param_name),
        )
        sweep_cb.grid(row=row, column=2, sticky="w", pady=2, padx=(10, 2))

        # Range controls (initially hidden)
        range_frame = ttk.Frame(parent)
        range_frame.grid(row=row, column=3, columnspan=3, sticky="w", pady=2)
        range_frame.grid_remove()  # Hide initially

        ttk.Label(range_frame, text="Min:").pack(side="left", padx=(0, 2))
        min_var = tk.StringVar(value=default_value)
        ttk.Entry(range_frame, textvariable=min_var, width=10).pack(side="left", padx=2)

        ttk.Label(range_frame, text="Max:").pack(side="left", padx=(5, 2))
        max_var = tk.StringVar(value=default_value)
        ttk.Entry(range_frame, textvariable=max_var, width=10).pack(side="left", padx=2)

        ttk.Label(range_frame, text="Pts:").pack(side="left", padx=(5, 2))
        points_var = tk.StringVar(value="3")
        ttk.Entry(range_frame, textvariable=points_var, width=4).pack(
            side="left", padx=2
        )

        log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(range_frame, text="Log", variable=log_var).pack(
            side="left", padx=(5, 0)
        )

        # Store all controls
        self.sweep_params[param_name] = {
            "fixed_var": var,
            "fixed_entry": entry,
            "sweep_var": sweep_var,
            "range_frame": range_frame,
            "min_var": min_var,
            "max_var": max_var,
            "points_var": points_var,
            "log_var": log_var,
        }

    def _toggle_sweep_controls(self, param_name):
        """Show/hide sweep range controls based on checkbox state."""
        controls = self.sweep_params[param_name]
        if controls["sweep_var"].get():
            controls["range_frame"].grid()
            controls["fixed_entry"].config(state="disabled")
        else:
            controls["range_frame"].grid_remove()
            controls["fixed_entry"].config(state="normal")

    def _toggle_wall_z_sweep(self):
        """Toggle wall_z sweep controls."""
        if self.wall_z_sweep_var.get():
            # Enable sweep controls, disable fixed value
            self.wall_z_entry.config(state="disabled")
            for widget in self.wall_z_sweep_widgets:
                widget.config(state="normal")
        else:
            # Disable sweep controls, enable fixed value
            self.wall_z_entry.config(state="normal")
            for widget in self.wall_z_sweep_widgets:
                widget.config(state="disabled")

    def _toggle_timestep_mode(self):
        """Toggle between duration/count auto-calculation modes."""
        mode = self.timestep_mode_var.get()
        if mode == "duration":
            # User provides count, we calculate duration
            self.steps_entry.config(state="normal")
            self.duration_entry.config(state="disabled")
        else:  # mode == "count"
            # User provides duration, we calculate count (min 200)
            self.steps_entry.config(state="disabled")
            self.duration_entry.config(state="normal")

    def _update_driver_visibility(self):
        """Show/hide driver section based on simulation type."""
        if not hasattr(self, "driver_frame"):
            return

        sim_type = self.sim_type_var.get()
        if sim_type == "BUNCH_TO_BUNCH":
            self.driver_frame.pack(
                fill="x",
                padx=10,
                pady=5,
                after=self.driver_frame.master.winfo_children()[2],
            )
        else:
            self.driver_frame.pack_forget()

    def _on_sim_type_changed(self):
        """Handle simulation type change."""
        self._update_driver_visibility()
        self._update_macroparticle_state()

    def _toggle_macroparticle_controls(self):
        """Enable/disable macroparticle controls based on checkbox state."""
        if not hasattr(self, "_macroparticle_widgets"):
            return

        enabled = self.macroparticle_enabled_var.get()
        state = "normal" if enabled else "disabled"

        for widget in self._macroparticle_widgets:
            if isinstance(widget, ttk.Entry):
                widget.configure(state=state)
            elif isinstance(widget, ttk.Checkbutton):
                widget.configure(state=state)
            elif isinstance(widget, ttk.Label):
                fg_color = "black" if enabled else "gray"
                widget.configure(foreground=fg_color)

        # Also handle sweep controls (checkboxes and range frames)
        if hasattr(self, "_macroparticle_sweep_controls"):
            for controls in self._macroparticle_sweep_controls:
                # Disable sweep checkbox
                if "sweep_var" in controls:
                    # Get the checkbox widget by finding it in the parent
                    # We need to find it from the stored frame
                    pass  # The checkboxes are not directly stored, skip for now
                # Disable/hide range frame if not sweeping
                if "range_frame" in controls and not controls["sweep_var"].get():
                    # Range frames are already hidden when not sweeping
                    pass

    def _update_macroparticle_state(self):
        """Enable/disable macroparticle controls based on simulation type."""
        if not hasattr(self, "macroparticle_enable_check"):
            return

        # Macroparticle simulation only available for CONDUCTING_WALL
        is_conducting_wall = self.sim_type_var.get() == "CONDUCTING_WALL"

        # Disable the entire macroparticle section if not conducting wall
        check_state = "normal" if is_conducting_wall else "disabled"
        self.macroparticle_enable_check.configure(state=check_state)

        # If not conducting wall, force it disabled and grey out all controls
        if not is_conducting_wall:
            self.macroparticle_enabled_var.set(False)
            widget_state = "disabled"
            label_color = "gray"
        else:
            # If conducting wall, respect the enabled checkbox state
            enabled = self.macroparticle_enabled_var.get()
            widget_state = "normal" if enabled else "disabled"
            label_color = "black" if enabled else "gray"

        if hasattr(self, "_macroparticle_widgets"):
            for widget in self._macroparticle_widgets:
                if isinstance(widget, ttk.Entry):
                    widget.configure(state=widget_state)
                elif isinstance(widget, ttk.Checkbutton):
                    widget.configure(state=widget_state)
                elif isinstance(widget, ttk.Label):
                    widget.configure(foreground=label_color)

        # Also disable/enable macroparticle sweep parameter controls
        if hasattr(self, "_macroparticle_sweep_controls"):
            for controls in self._macroparticle_sweep_controls:
                # Entry widgets in the controls are already in _macroparticle_widgets
                # Just handle the range frame entries
                if "range_frame" in controls:
                    for child in controls["range_frame"].winfo_children():
                        if isinstance(child, ttk.Entry):
                            child.configure(state=widget_state)
        self._update_parameter_visibility()

    def _update_parameter_visibility(self):
        """Update parameter field states based on simulation type."""
        if not hasattr(self, "cavity_spacing_entry"):
            return

        sim_type = self.sim_type_var.get()

        # Grey out cavity_spacing unless SWITCHING_WALL mode
        if sim_type == "SWITCHING_WALL":
            self.cavity_spacing_entry.config(state="normal")
        else:
            self.cavity_spacing_entry.config(state="disabled")

    def _build_objective_section(self):
        """Build optimization objective selection section."""
        # First build particle section
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

        # Method selection
        ttk.Label(self.optimization_frame, text="Optimization Method:").grid(
            row=0, column=0, sticky="w", pady=2
        )

        self.optimization_method_var = tk.StringVar(value="genetic_algorithm")
        method_combo = ttk.Combobox(
            self.optimization_frame,
            textvariable=self.optimization_method_var,
            values=[
                "genetic_algorithm",
                "differential_evolution",
                "nelder_mead",
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

        # Method descriptions
        method_descriptions = {
            "genetic_algorithm": "Evolutionary approach with selection, crossover, and mutation (robust, parallelizable)",
            "differential_evolution": "Global optimizer using vector differences (robust for rugged landscapes)",
            "nelder_mead": "Local simplex method (fast convergence, may find local optima)",
            "multi_start": "Multiple random starting points with local optimization (finds global optima)",
            "adaptive_grid": "Coarse-to-fine grid refinement (systematic, interpretable)",
        }

        self.method_desc_label = ttk.Label(
            self.optimization_frame,
            text=method_descriptions["genetic_algorithm"],
            foreground="gray40",
            font=("TkDefaultFont", 8),
            wraplength=500,
        )
        self.method_desc_label.grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        # Common parameters
        params_frame = ttk.Frame(self.optimization_frame)
        params_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)

        # Max iterations / generations
        ttk.Label(params_frame, text="Max Iterations/Generations:").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_maxiter_var = tk.StringVar(value="50")
        ttk.Entry(
            params_frame, textvariable=self.optimization_maxiter_var, width=10
        ).grid(row=0, column=1, sticky="w", pady=2)

        # Population size (for GA and DE)
        ttk.Label(params_frame, text="Population Size:").grid(
            row=0, column=2, sticky="w", pady=2, padx=(15, 5)
        )
        self.optimization_popsize_var = tk.StringVar(value="20")
        self.popsize_entry = ttk.Entry(
            params_frame, textvariable=self.optimization_popsize_var, width=10
        )
        self.popsize_entry.grid(row=0, column=3, sticky="w", pady=2)

        # GA-specific parameters
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

        # Multi-start parameter
        self.multistart_frame = ttk.Frame(self.optimization_frame)
        self.multistart_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)

        ttk.Label(self.multistart_frame, text="Number of Random Starts:").grid(
            row=0, column=0, sticky="w", pady=2, padx=(0, 5)
        )
        self.optimization_nstarts_var = tk.StringVar(value="5")
        ttk.Entry(
            self.multistart_frame, textvariable=self.optimization_nstarts_var, width=10
        ).grid(row=0, column=1, sticky="w", pady=2)

        # Output settings
        output_frame = ttk.LabelFrame(
            self.optimization_frame, text="Output Options", padding=5
        )
        output_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

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

        # Convergence settings
        convergence_frame = ttk.LabelFrame(
            self.optimization_frame, text="Convergence Settings", padding=5
        )
        convergence_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))

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

        # Initialize visibility
        self._update_optimization_controls()

    def _update_mode_visibility(self):
        """Update visibility of sections based on selected mode."""
        mode = self.mode_var.get()

        if mode == "blind_sweep":
            # Hide optimization settings
            self.optimization_frame.pack_forget()
            # Grey out Top N controls (only relevant for optimization)
            self._set_top_n_controls_state("disabled")
        else:  # optimization
            # Show optimization settings
            self.optimization_frame.pack(fill="x", padx=10, pady=5)
            # Enable Top N controls
            self._set_top_n_controls_state("normal")

        # Update parameter visibility based on simulation type
        self._update_parameter_visibility()

    def _set_top_n_controls_state(self, state):
        """Enable or disable Top N related controls.

        Parameters
        ----------
        state : str
            "normal" or "disabled"
        """
        if not hasattr(self, "save_top_n_traj_var"):
            return  # Widgets not created yet
        if not hasattr(self, "results_output_frame"):
            return  # Results output frame not created yet

        # Optimization section: "Save top N trajectories" entry
        if hasattr(self, "optimization_save_top_n_entry"):
            self.optimization_save_top_n_entry.configure(state=state)

        # Top N trajectory checkbox
        for widget in self.results_output_frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Checkbutton):
                        # Find the "Top N trajectories" checkbox
                        if "top_n_traj_var" in str(child.cget("variable")):
                            child.configure(state=state)

        # Metrics scope "Top N only" radio button
        if hasattr(self, "metrics_scope_var"):
            for widget in self.results_output_frame.winfo_children():
                if isinstance(widget, ttk.LabelFrame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Frame):
                            for radio in child.winfo_children():
                                if isinstance(radio, ttk.Radiobutton):
                                    if radio.cget("value") == "top_n":
                                        radio.configure(state=state)

        # Log verbosity "Top N only" radio button
        if hasattr(self, "log_verbosity_var"):
            for widget in self.results_output_frame.winfo_children():
                if isinstance(widget, ttk.LabelFrame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Radiobutton):
                            if child.cget("value") == "top_n_only":
                                child.configure(state=state)

        # If disabling and Top N is selected, switch to default
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

        # Update description
        method_descriptions = {
            "genetic_algorithm": "Evolutionary approach with selection, crossover, and mutation (robust, parallelizable)",
            "differential_evolution": "Global optimizer using vector differences (robust for rugged landscapes)",
            "nelder_mead": "Local simplex method (fast convergence, may find local optima)",
            "multi_start": "Multiple random starting points with local optimization (finds global optima)",
            "adaptive_grid": "Coarse-to-fine grid refinement (systematic, interpretable)",
        }
        self.method_desc_label.config(text=method_descriptions.get(method, ""))

        # Show/hide method-specific controls
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
        else:  # nelder_mead, adaptive_grid
            self.ga_frame.grid_forget()
            self.multistart_frame.grid_forget()
            self.popsize_entry.config(state="disabled")

    def _build_control_section(self):
        """Build control buttons section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Sweep Tools", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        # Info label
        info_label = ttk.Label(
            frame,
            text="Use 'Run Mode' selector in right panel to choose Single Run or Parameter Sweep, then click Run button.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray40",
        )
        info_label.pack(anchor="w", pady=(0, 10))

        # Row 1: Load from main config helper
        helper_frame = ttk.Frame(frame)
        helper_frame.pack(fill="x", pady=2)

        ttk.Button(
            helper_frame,
            text="Load from Main GUI Config",
            command=self._on_load_from_main_config,
        ).pack(side="left", padx=5)

        ttk.Label(
            helper_frame,
            text="← Copy current single-run config to sweep parameters",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).pack(side="left", padx=5)

        # Robustness options
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

        self.skip_failed_runs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            robustness_frame,
            text="Skip failed runs and continue sweep",
            variable=self.skip_failed_runs_var,
        ).pack(side="left", padx=5)

        # Row 2: Retry attempts for failed runs
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

        # Row 4: Trajectory stability checking (multi-step analysis)
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

        # Store widgets for enable/disable toggle
        self.smoothness_widgets = [
            smoothness_frame.grid_slaves(row=1, column=0)[0],  # Window size label
            smoothness_frame.grid_slaves(row=1, column=1)[0],  # Window size entry
            smoothness_frame.grid_slaves(row=2, column=0)[0],  # Oscillation label
            smoothness_frame.grid_slaves(row=2, column=1)[0],  # Oscillation entry
            smoothness_frame.grid_slaves(row=3, column=0)[0],  # Reject checkbox
        ]

    def _build_results_output_section(self):
        """Build results viewing and output configuration section."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Results & Output Configuration", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)
        frame.columnconfigure(1, weight=1)

        # Results viewing buttons
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

        # Trajectory saving options
        ttk.Label(frame, text="Trajectory Data:").grid(
            row=1, column=0, sticky="nw", pady=(5, 2)
        )

        traj_frame = ttk.Frame(frame)
        traj_frame.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(5, 2))

        self.save_top_n_traj_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            traj_frame,
            text="Top N trajectories (full detail)",
            variable=self.save_top_n_traj_var,
            command=self._on_top_n_traj_changed,
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))

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

        # Initialize trajectory stride state
        self._update_stride_state()

        # Metrics export options
        ttk.Label(frame, text="Metrics Export:").grid(
            row=2, column=0, sticky="nw", pady=(10, 2)
        )

        metrics_frame = ttk.Frame(frame)
        metrics_frame.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(10, 2))

        # Format selection
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

        # Scope selection
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
        ttk.Radiobutton(
            scope_frame,
            text="Top N only",
            variable=self.metrics_scope_var,
            value="top_n",
        ).pack(side="left", padx=5)

        ttk.Label(
            frame,
            text="ℹ JSON contains metadata & structure; CSV is tabular with all parameters & metrics",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray50",
        ).grid(row=3, column=1, columnspan=2, sticky="w", pady=(0, 10))

        # Log saving options
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

        ttk.Radiobutton(
            log_frame,
            text="Top N only (logs only for best N trajectories)",
            variable=self.log_verbosity_var,
            value="top_n_only",
        ).grid(row=3, column=0, sticky="w", pady=2)

        ttk.Label(
            frame,
            text="ℹ 'Truncated' is recommended for large sweeps.\n'Full debug' inherits verbosity settings from Stability tab and generates large log files.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="blue",
            justify="left",
        ).grid(row=5, column=1, columnspan=2, sticky="w", pady=(0, 5))

    def _on_top_n_traj_changed(self):
        """Handle Top N trajectory checkbox change."""
        # Top N can be combined with All or Failed, no exclusivity needed
        self._update_stride_state()

    def _on_all_traj_changed(self):
        """Handle All trajectories checkbox change."""
        if self.save_all_traj_var.get():
            # "All" was just checked - uncheck "Failed only"
            self.save_failed_traj_var.set(False)
        self._update_stride_state()

    def _on_failed_traj_changed(self):
        """Handle Failed only checkbox change."""
        if self.save_failed_traj_var.get():
            # "Failed only" was just checked - uncheck "All"
            self.save_all_traj_var.set(False)
        self._update_stride_state()

    def _update_stride_state(self):
        """Update stride field enabled/disabled state."""
        if not hasattr(self, "trajectory_stride_entry"):
            return  # Widgets not created yet

        # Stride is ONLY enabled when "All trajectories" is selected
        # (Top N and Failed only always save full detail with stride=1)
        stride_enabled = self.save_all_traj_var.get()

        widget_state = "normal" if stride_enabled else "disabled"
        label_color = "black" if stride_enabled else "gray"

        self.trajectory_stride_entry.configure(state=widget_state)
        self.trajectory_stride_label.configure(foreground=label_color)

    def _build_progress_section(self):
        """Build progress monitoring section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Sweep Progress", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            frame, mode="determinate", maximum=100, length=400
        )
        self.progress_bar.pack(fill="x", pady=5)

        # Progress label
        self.progress_label = ttk.Label(frame, text="Ready")
        self.progress_label.pack(anchor="w", pady=2)

        # Info label directing users to main GUI logs
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
