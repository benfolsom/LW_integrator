"""Parameter and particle section builders for the optimization plugin."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class OptimizationPluginParameterMixin:
    """Build the parameter and particle form sections for the optimization plugin."""

    def _build_parameter_section(self):
        """Build parameter range specification section."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Parameter Ranges", padding=10
        )
        frame.pack(fill="x", padx=10, pady=5)
        self.parameter_frame = frame

        # Store parameter widgets for mode-based visibility control
        self._param_widgets = {}

        # Add explanatory help text
        help_frame = ttk.Frame(frame)
        help_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        help_text = (
            "Coordinate system: Particles start at z-coordinate and travel toward the conducting wall.\n"
            "Example: Particle at z=0 travels to wall at z=2200 mm (distance = 2200 mm).\n"
            "Transverse offset: BUNCH_TO_BUNCH mode uses absolute distance (mm), CONDUCTING_WALL uses fraction of aperture."
        )
        help_label = ttk.Label(
            help_frame, text=help_text, foreground="gray40", font=("TkDefaultFont", 8)
        )
        help_label.pack(anchor="w")

        # Aperture range
        self._param_widgets["aperture_label"] = ttk.Label(
            frame, text="Aperture Radius:"
        )
        self._param_widgets["aperture_label"].grid(row=1, column=0, sticky="w", pady=2)
        self._param_widgets["aperture_frame"] = ttk.Frame(frame)
        aperture_frame = self._param_widgets["aperture_frame"]
        aperture_frame.grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)

        ttk.Label(aperture_frame, text="Min (mm):").pack(side="left", padx=(0, 2))
        self.aperture_min_var = tk.StringVar(value="1e-5")
        self._param_widgets["aperture_min_entry"] = ttk.Entry(
            aperture_frame, textvariable=self.aperture_min_var, width=10
        )
        self._param_widgets["aperture_min_entry"].pack(side="left", padx=2)

        ttk.Label(aperture_frame, text="Max (mm):").pack(side="left", padx=(10, 2))
        self.aperture_max_var = tk.StringVar(value="1e-3")
        self._param_widgets["aperture_max_entry"] = ttk.Entry(
            aperture_frame, textvariable=self.aperture_max_var, width=10
        )
        self._param_widgets["aperture_max_entry"].pack(side="left", padx=2)

        ttk.Label(aperture_frame, text="Points:").pack(side="left", padx=(10, 2))
        self.aperture_points_var = tk.StringVar(value="10")
        self._param_widgets["aperture_points_entry"] = ttk.Entry(
            aperture_frame, textvariable=self.aperture_points_var, width=5
        )
        self._param_widgets["aperture_points_entry"].pack(side="left", padx=2)

        self.aperture_log_var = tk.BooleanVar(value=True)
        self._param_widgets["aperture_log_check"] = ttk.Checkbutton(
            aperture_frame, text="Log scale", variable=self.aperture_log_var
        )
        self._param_widgets["aperture_log_check"].pack(side="left", padx=(10, 0))

        # Energy range
        self.energy_label = ttk.Label(frame, text="Particle Energy:")
        self.energy_label.grid(row=2, column=0, sticky="w", pady=2)
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

        # Helper text showing starting Pz for rider
        self.rider_pz_helper_var = tk.StringVar(value="")
        self.rider_pz_helper_label = ttk.Label(
            frame,
            textvariable=self.rider_pz_helper_var,
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        )
        self.rider_pz_helper_label.grid(
            row=3, column=1, columnspan=3, sticky="w", pady=(0, 5)
        )

        # Trace energy and mass changes to update Pz helper
        self.energy_min_var.trace_add(
            "write", lambda *_: self._update_rider_pz_helper()
        )
        self.energy_max_var.trace_add(
            "write", lambda *_: self._update_rider_pz_helper()
        )
        # Note: rider mass trace is added in _build_rider_particle_section after widget creation

        # Rider Transverse offset (x, y)
        self._param_widgets["offset_label"] = ttk.Label(
            frame, text="Rider Transverse Offset (x, y):"
        )
        self._param_widgets["offset_label"].grid(row=4, column=0, sticky="w", pady=2)
        self.offset_fractions_var = tk.StringVar(value="0.0, 0.0")
        self._param_widgets["offset_entry"] = ttk.Entry(
            frame, textvariable=self.offset_fractions_var, width=15
        )
        self._param_widgets["offset_entry"].grid(
            row=4, column=1, sticky="w", pady=2, padx=(0, 5)
        )
        self._param_widgets["offset_desc_label"] = ttk.Label(
            frame,
            text="mm (BUNCH_TO_BUNCH: absolute offset | CONDUCTING_WALL: fraction of aperture)",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        )
        self._param_widgets["offset_desc_label"].grid(
            row=4, column=2, columnspan=2, sticky="w", pady=2
        )

        # Driver Transverse offset (x, y)
        self._param_widgets["driver_offset_label"] = ttk.Label(
            frame, text="Driver Transverse Offset (x, y):"
        )
        self._param_widgets["driver_offset_label"].grid(
            row=5, column=0, sticky="w", pady=2
        )
        self.driver_offset_var = tk.StringVar(value="0.0, 0.0")
        self._param_widgets["driver_offset_entry"] = ttk.Entry(
            frame, textvariable=self.driver_offset_var, width=15
        )
        self._param_widgets["driver_offset_entry"].grid(
            row=5, column=1, sticky="w", pady=2, padx=(0, 5)
        )
        self._param_widgets["driver_offset_desc_label"] = ttk.Label(
            frame,
            text="mm (BUNCH_TO_BUNCH only)",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        )
        self._param_widgets["driver_offset_desc_label"].grid(
            row=5, column=2, columnspan=2, sticky="w", pady=2
        )

        # Starting z position (rider only)
        ttk.Label(frame, text="Rider Starting z (mm):").grid(
            row=6, column=0, sticky="w", pady=2
        )
        self.start_z_var = tk.StringVar(value="0.0")
        ttk.Entry(frame, textvariable=self.start_z_var, width=15).grid(
            row=6, column=1, sticky="w", pady=2
        )
        ttk.Label(
            frame,
            text="Driver starting z uses sweepable param below.",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        ).grid(row=6, column=2, columnspan=2, sticky="w", pady=2, padx=(10, 0))

        # Wall Position (sweepable)
        self._param_widgets["wall_z_label"] = ttk.Label(frame, text="Wall Position:")
        self._param_widgets["wall_z_label"].grid(row=8, column=0, sticky="w", pady=2)

        # Fixed value and sweep checkbox on same row
        self._param_widgets["wall_z_fixed_frame"] = ttk.Frame(frame)
        wall_z_fixed_frame = self._param_widgets["wall_z_fixed_frame"]
        wall_z_fixed_frame.grid(row=8, column=1, columnspan=3, sticky="w", pady=2)

        self.wall_z_var = tk.StringVar(value="2200.0")
        self.wall_z_entry = ttk.Entry(
            wall_z_fixed_frame, textvariable=self.wall_z_var, width=10
        )
        self.wall_z_entry.pack(side="left", padx=(0, 10))

        self.wall_z_sweep_var = tk.BooleanVar(value=False)
        self._param_widgets["wall_z_sweep_check"] = ttk.Checkbutton(
            wall_z_fixed_frame,
            text="Sweep",
            variable=self.wall_z_sweep_var,
            command=self._toggle_wall_z_sweep,
        )
        self._param_widgets["wall_z_sweep_check"].pack(side="left")

        # Sweep controls on new row for better visibility
        self._param_widgets["wall_z_sweep_frame"] = ttk.Frame(frame)
        wall_z_sweep_frame = self._param_widgets["wall_z_sweep_frame"]
        wall_z_sweep_frame.grid(
            row=9, column=1, columnspan=3, sticky="w", pady=2, padx=(20, 0)
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

        # Cavity Spacing (for SWITCHING_WALL) - moved to row 10
        self._param_widgets["cavity_spacing_label"] = ttk.Label(
            frame, text="Cavity Spacing:"
        )
        self._param_widgets["cavity_spacing_label"].grid(
            row=10, column=0, sticky="w", pady=2
        )
        self.cavity_spacing_var = tk.StringVar(value="1e5")
        self.cavity_spacing_entry = ttk.Entry(
            frame, textvariable=self.cavity_spacing_var, width=15
        )
        self.cavity_spacing_entry.grid(
            row=10, column=1, sticky="w", pady=2, padx=(0, 5)
        )
        self._param_widgets["cavity_spacing_desc_label"] = ttk.Label(
            frame,
            text="mm (SWITCHING_WALL only)",
            font=("TkDefaultFont", 8),
            foreground="gray40",
        )
        self._param_widgets["cavity_spacing_desc_label"].grid(
            row=10, column=2, columnspan=2, sticky="w", pady=2
        )

        # Timestep Auto-Calculation (always uses auto_distance strategy)
        timestep_label = ttk.Label(frame, text="Timestep Calculation:")
        timestep_label.grid(row=11, column=0, sticky="w", pady=2)

        # Store label for dynamic tooltip update
        self.timestep_calc_label = timestep_label

        # Initial tooltip - will be updated based on sim type
        self._update_timestep_tooltip()

        timestep_frame = ttk.Frame(frame)
        timestep_frame.grid(row=11, column=1, columnspan=3, sticky="ew", pady=2)

        self.timestep_mode_var = tk.StringVar(value="duration")
        ttk.Radiobutton(
            timestep_frame,
            text="Auto-calc step duration, provide count:",
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
            text="Auto-calc count, provide step duration:",
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
        self._param_widgets["distance_label"] = ttk.Label(
            frame, text="Distance Target:"
        )
        self._param_widgets["distance_label"].grid(row=12, column=0, sticky="w", pady=2)
        self._param_widgets["distance_frame"] = ttk.Frame(frame)
        distance_frame = self._param_widgets["distance_frame"]
        distance_frame.grid(row=12, column=1, columnspan=3, sticky="ew", pady=2)

        # Dynamic label that changes based on sim type
        self.distance_target_prefix_label = ttk.Label(distance_frame, text="Target:")
        self.distance_target_prefix_label.pack(side="left", padx=(0, 2))

        self.auto_steps_distance_var = tk.StringVar(value="10.0")
        self._param_widgets["distance_entry"] = ttk.Entry(
            distance_frame, textvariable=self.auto_steps_distance_var, width=6
        )
        self._param_widgets["distance_entry"].pack(side="left", padx=2)

        # Dynamic suffix label
        self.distance_target_suffix_label = ttk.Label(
            distance_frame, text="mm past wall (min 5% of driver position for B2B)"
        )
        self.distance_target_suffix_label.pack(side="left", padx=2)

        # Note about trajectory and output configuration
        config_note = ttk.Label(
            frame,
            text="ℹ For trajectory saving and output options, see the 'Results & Output Configuration' section below",
            font=("TkDefaultFont", 8, "italic"),
            foreground="blue",
            justify="left",
        )
        config_note.grid(row=13, column=0, columnspan=4, sticky="w", pady=(10, 10))

        frame.columnconfigure(2, weight=1)

        # Initialize timestep mode state
        self._toggle_timestep_mode()

        # Initialize parameter visibility based on simulation type
        self._update_parameter_visibility()

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

        # Add trace for rider mass to update Pz helper
        self.sweep_params["rider_m_particle"]["fixed_var"].trace_add(
            "write", lambda *_: self._update_rider_pz_helper()
        )

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
            "Transverse Spread/Radius (mm):",
            "2e-06",
            width=15,
        )
        row += 1

        ttk.Label(frame, text="Transverse Geometry:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.rider_transverse_geometry_var = tk.StringVar(value="square")
        ttk.Combobox(
            frame,
            textvariable=self.rider_transverse_geometry_var,
            values=("square", "point", "gaussian", "ring"),
            state="readonly",
            width=15,
        ).grid(row=row, column=1, sticky="w", pady=2, padx=5)
        row += 1

        # Stripped Ions (sweepable)
        self._add_sweepable_param(
            frame, row, "rider_stripped_ions", "Stripped Ions:", "1.0", width=10
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

        # Transverse Spread
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_transv_dist",
            "Transverse Spread/Radius (mm):",
            "-0.07998",
            width=15,
        )
        row += 1

        ttk.Label(self.driver_frame, text="Transverse Geometry:").grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.driver_transverse_geometry_var = tk.StringVar(value="square")
        ttk.Combobox(
            self.driver_frame,
            textvariable=self.driver_transverse_geometry_var,
            values=("square", "point", "gaussian", "ring"),
            state="readonly",
            width=15,
        ).grid(row=row, column=1, sticky="w", pady=2, padx=5)
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

        # Driver Energy (replaces Starting Pz)
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_energy_gev",
            "Kinetic Energy (GeV):",
            "112.5",
            width=15,
        )
        row += 1

        # Link to Rider Energy checkbox - when checked, driver energy follows rider energy
        link_frame = ttk.Frame(self.driver_frame)
        link_frame.grid(
            row=row, column=0, columnspan=6, sticky="w", pady=(0, 2), padx=(20, 0)
        )

        self.link_driver_rider_energy_var = tk.BooleanVar(value=False)
        self.link_energy_checkbox = ttk.Checkbutton(
            link_frame,
            text="Link to Rider Energy (sweep both at same value)",
            variable=self.link_driver_rider_energy_var,
            command=self._on_link_energy_toggled,
        )
        self.link_energy_checkbox.pack(side="left")

        # Help text for linked energy mode
        self.link_energy_help_label = ttk.Label(
            link_frame,
            text="",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        )
        self.link_energy_help_label.pack(side="left", padx=(10, 0))

        for var in (self.energy_min_var, self.energy_max_var, self.energy_points_var):
            var.trace_add("write", lambda *_: self._update_linked_energy_presentation())

        row += 1

        # Driver momentum direction selector (-z or +z)
        dir_frame = ttk.Frame(self.driver_frame)
        dir_frame.grid(
            row=row, column=0, columnspan=6, sticky="w", pady=(0, 2), padx=(20, 0)
        )

        ttk.Label(dir_frame, text="Momentum direction:").pack(side="left", padx=(0, 6))

        self.driver_direction_var = tk.StringVar(value="-z")

        rb_minus = ttk.Radiobutton(
            dir_frame,
            text="\u2212\u1e91  (toward rider)",
            variable=self.driver_direction_var,
            value="-z",
            command=self._update_driver_pz_helper,
        )
        rb_minus.pack(side="left", padx=(0, 12))

        rb_plus = ttk.Radiobutton(
            dir_frame,
            text="+\u1e91  (away from rider)",
            variable=self.driver_direction_var,
            value="+z",
            command=self._update_driver_pz_helper,
        )
        rb_plus.pack(side="left")

        row += 1

        # Helper text showing calculated starting Pz for driver
        self.driver_pz_helper_var = tk.StringVar(value="")
        self.driver_pz_helper_label = ttk.Label(
            self.driver_frame,
            textvariable=self.driver_pz_helper_var,
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray",
        )
        self.driver_pz_helper_label.grid(
            row=row, column=0, columnspan=6, sticky="w", pady=(0, 5), padx=(20, 0)
        )
        row += 1

        # Trace driver energy and mass changes to update Pz helper
        self.sweep_params["driver_energy_gev"]["fixed_var"].trace_add(
            "write", lambda *_: self._update_driver_pz_helper()
        )
        self.sweep_params["driver_energy_gev"]["min_var"].trace_add(
            "write", lambda *_: self._update_driver_pz_helper()
        )
        self.sweep_params["driver_energy_gev"]["max_var"].trace_add(
            "write", lambda *_: self._update_driver_pz_helper()
        )
        self.sweep_params["driver_m_particle"]["fixed_var"].trace_add(
            "write", lambda *_: self._update_driver_pz_helper()
        )

        # Store references to driver energy widgets for enable/disable
        self._driver_energy_widgets = [
            self.sweep_params["driver_energy_gev"]["fixed_entry"],
            self.sweep_params["driver_energy_gev"]["range_frame"],
        ]

        # Stripped Ions (sweepable)
        self._add_sweepable_param(
            self.driver_frame,
            row,
            "driver_stripped_ions",
            "Stripped Ions:",
            "54.0",
            width=10,
        )
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
