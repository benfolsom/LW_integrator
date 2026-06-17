"""Tab-builder helpers for the main GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .testbed_runner import (
    AVAILABLE_DPI_CHOICES,
    CORE_PARAM_LABELS,
    PARAM_LABELS,
    PARTICLE_PARAM_FIELDS,
    RADIATION_REACTION_MODE_CHOICES,
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

            if name == "transverse_geometry":
                rider_entry = ttk.Combobox(
                    particle_frame,
                    textvariable=self.rider_param_vars[name],
                    values=("square", "point", "gaussian", "ring"),
                    state="readonly",
                    width=12,
                )
            else:
                rider_entry = ttk.Entry(
                    particle_frame, textvariable=self.rider_param_vars[name], width=12
                )
            rider_entry.grid(row=row, column=1, sticky="ew", pady=2)

            driver_label = ttk.Label(
                particle_frame, text=PARAM_LABELS[name] + " (driver):"
            )
            driver_label.grid(row=row, column=2, sticky="w", pady=2, padx=(12, 0))

            if name == "transverse_geometry":
                driver_entry = ttk.Combobox(
                    particle_frame,
                    textvariable=self.driver_param_vars[name],
                    values=("square", "point", "gaussian", "ring"),
                    state="readonly",
                    width=12,
                )
            else:
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

        next_row += 1
        ttk.Separator(particle_frame, orient="horizontal").grid(
            row=next_row, column=0, columnspan=4, sticky="ew", pady=(12, 12)
        )
        next_row += 1

        ttk.Label(
            particle_frame,
            text="Macroparticle Smearing (all modes):",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=next_row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        next_row += 1

        self.macroparticle_smearing_enable_check = ttk.Checkbutton(
            particle_frame,
            text="Enable bounded source smearing",
            variable=self.macroparticle_smearing_enabled_var,
            command=self._toggle_macroparticle_smearing_controls,
        )
        self.macroparticle_smearing_enable_check.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=2
        )
        next_row += 1

        smearing_fields = [
            (
                "Subcharge count:",
                "macroparticle_smearing_subcharge_count_var",
                "macroparticle_smearing_subcharge_count_entry",
            ),
            (
                "Sigma multiplier:",
                "macroparticle_smearing_sigma_multiplier_var",
                "macroparticle_smearing_sigma_multiplier_entry",
            ),
            (
                "Position sigma mm (blank=auto):",
                "macroparticle_smearing_position_sigma_var",
                "macroparticle_smearing_position_sigma_entry",
            ),
            (
                "Longitudinal sigma mm (blank=0):",
                "macroparticle_smearing_longitudinal_sigma_var",
                "macroparticle_smearing_longitudinal_sigma_entry",
            ),
            (
                "Momentum sigma amu*mm/ns (blank=0):",
                "macroparticle_smearing_momentum_sigma_var",
                "macroparticle_smearing_momentum_sigma_entry",
            ),
            (
                "Smearing seed:",
                "macroparticle_smearing_seed_var",
                "macroparticle_smearing_seed_entry",
            ),
        ]
        self._macroparticle_smearing_widgets = []
        for label_text, var_name, entry_name in smearing_fields:
            label = ttk.Label(particle_frame, text=label_text)
            label.grid(row=next_row, column=0, sticky="w", pady=2, padx=(20, 0))
            entry = ttk.Entry(
                particle_frame,
                textvariable=getattr(self, var_name),
                width=12,
            )
            entry.grid(row=next_row, column=1, sticky="ew", pady=2)
            setattr(self, entry_name, entry)
            self._macroparticle_smearing_widgets.extend([label, entry])
            next_row += 1

        self.macroparticle_smearing_passive_updates_check = ttk.Checkbutton(
            particle_frame,
            text="Apply to pseudo-grid passive updates (experimental)",
            variable=self.macroparticle_smearing_apply_to_passive_updates_var,
        )
        self.macroparticle_smearing_passive_updates_check.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )
        self._macroparticle_smearing_widgets.append(
            self.macroparticle_smearing_passive_updates_check
        )
        next_row += 1

        help_text_smearing = ttk.Label(
            particle_frame,
            text=(
                "Smearing splits source macroparticles into deterministic random subcharges.\n"
                "Auto position width scales with macro population but is capped near half\n"
                "an estimated inter-macroparticle spacing. Leave sigmas blank for defaults."
            ),
            font=("TkDefaultFont", 8),
            foreground="gray40",
            justify="left",
        )
        help_text_smearing.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=(0, 2), padx=(20, 0)
        )
        self._macroparticle_smearing_widgets.append(help_text_smearing)

        next_row += 1
        ttk.Separator(particle_frame, orient="horizontal").grid(
            row=next_row, column=0, columnspan=4, sticky="ew", pady=(12, 12)
        )
        next_row += 1

        ttk.Label(
            particle_frame,
            text="Pseudo-grid Mode (Bunch-to-Bunch only):",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=next_row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        next_row += 1

        self.pseudo_grid_enable_check = ttk.Checkbutton(
            particle_frame,
            text="Enable experimental pseudo-grid scheduler",
            variable=self.pseudo_grid_enabled_var,
            command=self._toggle_pseudo_grid_controls,
        )
        self.pseudo_grid_enable_check.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=2
        )
        next_row += 1

        self.pseudo_grid_active_rider_label = ttk.Label(
            particle_frame, text="Active rider count:"
        )
        self.pseudo_grid_active_rider_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.pseudo_grid_active_rider_entry = ttk.Entry(
            particle_frame,
            textvariable=self.pseudo_grid_active_rider_count_var,
            width=12,
        )
        self.pseudo_grid_active_rider_entry.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.pseudo_grid_active_driver_label = ttk.Label(
            particle_frame, text="Active driver count:"
        )
        self.pseudo_grid_active_driver_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.pseudo_grid_active_driver_entry = ttk.Entry(
            particle_frame,
            textvariable=self.pseudo_grid_active_driver_count_var,
            width=12,
        )
        self.pseudo_grid_active_driver_entry.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.pseudo_grid_passive_neighbor_label = ttk.Label(
            particle_frame, text="Passive neighbor count:"
        )
        self.pseudo_grid_passive_neighbor_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.pseudo_grid_passive_neighbor_entry = ttk.Entry(
            particle_frame,
            textvariable=self.pseudo_grid_passive_neighbor_count_var,
            width=12,
        )
        self.pseudo_grid_passive_neighbor_entry.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.pseudo_grid_pair_reuse_label = ttk.Label(
            particle_frame, text="Pair reuse window:"
        )
        self.pseudo_grid_pair_reuse_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.pseudo_grid_pair_reuse_entry = ttk.Entry(
            particle_frame,
            textvariable=self.pseudo_grid_pair_reuse_window_var,
            width=12,
        )
        self.pseudo_grid_pair_reuse_entry.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.pseudo_grid_coverage_strategy_label = ttk.Label(
            particle_frame, text="Coverage strategy:"
        )
        self.pseudo_grid_coverage_strategy_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.pseudo_grid_coverage_strategy_combo = ttk.Combobox(
            particle_frame,
            textvariable=self.pseudo_grid_coverage_strategy_var,
            values=("farthest_point_staleness", "farthest_point"),
            state="readonly",
            width=24,
        )
        self.pseudo_grid_coverage_strategy_combo.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.pseudo_grid_coverage_space_label = ttk.Label(
            particle_frame, text="Coverage space:"
        )
        self.pseudo_grid_coverage_space_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.pseudo_grid_coverage_space_combo = ttk.Combobox(
            particle_frame,
            textvariable=self.pseudo_grid_coverage_space_var,
            values=("position", "phase_space"),
            state="readonly",
            width=24,
        )
        self.pseudo_grid_coverage_space_combo.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.pseudo_grid_source_weighting_label = ttk.Label(
            particle_frame, text="Source weighting mode:"
        )
        self.pseudo_grid_source_weighting_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.pseudo_grid_source_weighting_combo = ttk.Combobox(
            particle_frame,
            textvariable=self.pseudo_grid_source_weighting_mode_var,
            values=("inverse_distance", "nearest"),
            state="readonly",
            width=24,
        )
        self.pseudo_grid_source_weighting_combo.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.pseudo_grid_loss_tracking_check = ttk.Checkbutton(
            particle_frame,
            text="Track particle losses in pseudo-grid mode",
            variable=self.pseudo_grid_loss_tracking_enabled_var,
        )
        self.pseudo_grid_loss_tracking_check.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )
        next_row += 1

        self.pseudo_grid_causal_pruning_check = ttk.Checkbutton(
            particle_frame,
            text="Enable causal-history pruning",
            variable=self.pseudo_grid_causal_history_pruning_enabled_var,
            command=self._toggle_pseudo_grid_controls,
        )
        self.pseudo_grid_causal_pruning_check.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )
        next_row += 1

        self.pseudo_grid_causal_safety_label = ttk.Label(
            particle_frame, text="Causal safety margin (steps):"
        )
        self.pseudo_grid_causal_safety_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(40, 0)
        )
        self.pseudo_grid_causal_safety_entry = ttk.Entry(
            particle_frame,
            textvariable=self.pseudo_grid_causal_history_safety_margin_steps_var,
            width=12,
        )
        self.pseudo_grid_causal_safety_entry.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        help_text_pseudo_grid = ttk.Label(
            particle_frame,
            text=(
                "Pseudo-grid mode is currently an experimental configuration surface for BUNCH_TO_BUNCH runs.\n"
                "Plumbing is present in the GUI, CLI, and saved configs while the reduced active/passive solver path is built incrementally.\n"
                "Use the pair-reuse window to discourage repeated active matches, and causal-history pruning to prepare for bounded history retention."
            ),
            font=("TkDefaultFont", 8),
            foreground="gray40",
            justify="left",
        )
        help_text_pseudo_grid.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=(0, 2), padx=(20, 0)
        )

        self._pseudo_grid_widgets = [
            self.pseudo_grid_active_rider_label,
            self.pseudo_grid_active_rider_entry,
            self.pseudo_grid_active_driver_label,
            self.pseudo_grid_active_driver_entry,
            self.pseudo_grid_passive_neighbor_label,
            self.pseudo_grid_passive_neighbor_entry,
            self.pseudo_grid_pair_reuse_label,
            self.pseudo_grid_pair_reuse_entry,
            self.pseudo_grid_coverage_strategy_label,
            self.pseudo_grid_coverage_strategy_combo,
            self.pseudo_grid_coverage_space_label,
            self.pseudo_grid_coverage_space_combo,
            self.pseudo_grid_source_weighting_label,
            self.pseudo_grid_source_weighting_combo,
            self.pseudo_grid_loss_tracking_check,
            self.pseudo_grid_causal_pruning_check,
            self.pseudo_grid_causal_safety_label,
            self.pseudo_grid_causal_safety_entry,
        ]
        self._pseudo_grid_causal_widgets = [
            self.pseudo_grid_causal_safety_label,
            self.pseudo_grid_causal_safety_entry,
        ]
        self._toggle_pseudo_grid_controls()

        next_row += 1
        ttk.Separator(particle_frame, orient="horizontal").grid(
            row=next_row, column=0, columnspan=4, sticky="ew", pady=(12, 12)
        )
        next_row += 1

        ttk.Label(
            particle_frame,
            text="Driver Train / Persistent Prehistory (Bunch-to-Bunch only):",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=next_row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        next_row += 1

        self.driver_train_enable_check = ttk.Checkbutton(
            particle_frame,
            text="Enable flat driver-train source",
            variable=self.driver_train_enabled_var,
            command=self._toggle_driver_train_controls,
        )
        self.driver_train_enable_check.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=2
        )
        next_row += 1

        self.driver_train_bunch_count_label = ttk.Label(
            particle_frame, text="Driver bunch count:"
        )
        self.driver_train_bunch_count_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.driver_train_bunch_count_entry = ttk.Entry(
            particle_frame,
            textvariable=self.driver_train_bunch_count_var,
            width=12,
        )
        self.driver_train_bunch_count_entry.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.driver_train_z_spacing_label = ttk.Label(
            particle_frame, text="z spacing (mm):"
        )
        self.driver_train_z_spacing_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.driver_train_z_spacing_entry = ttk.Entry(
            particle_frame,
            textvariable=self.driver_train_z_spacing_mm_var,
            width=12,
        )
        self.driver_train_z_spacing_entry.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.driver_train_z_offsets_label = ttk.Label(
            particle_frame, text="Explicit z offsets (mm):"
        )
        self.driver_train_z_offsets_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.driver_train_z_offsets_entry = ttk.Entry(
            particle_frame,
            textvariable=self.driver_train_z_offsets_mm_var,
            width=24,
        )
        self.driver_train_z_offsets_entry.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.driver_train_prehistory_label = ttk.Label(
            particle_frame, text="Prehistory rows:"
        )
        self.driver_train_prehistory_label.grid(
            row=next_row, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.driver_train_prehistory_entry = ttk.Entry(
            particle_frame,
            textvariable=self.driver_train_prehistory_steps_var,
            width=12,
        )
        self.driver_train_prehistory_entry.grid(
            row=next_row, column=1, sticky="ew", pady=2
        )
        next_row += 1

        self.driver_train_preserve_check = ttk.Checkbutton(
            particle_frame,
            text="Preserve prehistory rows in output",
            variable=self.driver_train_preserve_prehistory_var,
        )
        self.driver_train_preserve_check.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )
        next_row += 1

        help_text_driver_train = ttk.Label(
            particle_frame,
            text=(
                "Expands the configured driver bunch into longitudinal copies and can seed inertial back-history.\n"
                "Leave explicit offsets blank to use count × spacing; pseudo-grid can still be enabled, "
                "with full-history fallback when the reduced schedule is not suitable."
            ),
            font=("TkDefaultFont", 8),
            foreground="gray40",
            justify="left",
        )
        help_text_driver_train.grid(
            row=next_row, column=0, columnspan=2, sticky="w", pady=(0, 2), padx=(20, 0)
        )

        self._driver_train_widgets = [
            self.driver_train_bunch_count_label,
            self.driver_train_bunch_count_entry,
            self.driver_train_z_spacing_label,
            self.driver_train_z_spacing_entry,
            self.driver_train_z_offsets_label,
            self.driver_train_z_offsets_entry,
            self.driver_train_prehistory_label,
            self.driver_train_prehistory_entry,
            self.driver_train_preserve_check,
        ]
        self._toggle_driver_train_controls()

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
        self.steps_entry = ttk.Entry(core_frame, textvariable=self.steps_var, width=16)
        self.steps_entry.grid(row=row, column=1, sticky="ew", pady=2)
        row += 1
        self.steps_auto_hint = ttk.Label(
            core_frame,
            text="(computed by auto-duration)",
            foreground="gray",
            font=("TkDefaultFont", 8, "italic"),
        )
        self.steps_auto_hint.grid(row=row, column=1, sticky="w", pady=(0, 2))
        self.steps_auto_hint.grid_remove()
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

            if name == "time_step":
                self.time_step_auto_hint = ttk.Label(
                    core_frame,
                    text="(computed by auto-duration)",
                    foreground="gray",
                    font=("TkDefaultFont", 8, "italic"),
                )
                self.time_step_auto_hint.grid(
                    row=row, column=1, sticky="w", pady=(0, 2)
                )
                self.time_step_auto_hint.grid_remove()
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

    def _build_external_fields_tab(self) -> None:
        """Build prescribed external-field controls."""
        from .gui import Tooltip

        field_frame = self._create_scrollable_tab(
            self.notebook, "External Fields", padding=12
        )
        field_frame.columnconfigure(1, weight=1)
        field_frame.columnconfigure(2, weight=1)
        field_frame.columnconfigure(3, weight=1)

        ttk.Label(
            field_frame,
            text=(
                "These settings apply to BOTH single runs and sweeps/optimizations. "
                "Sweep runs inherit these fixed field settings for every point."
            ),
            font=("TkDefaultFont", 9, "bold"),
            foreground="blue",
            justify="left",
            wraplength=720,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))

        enable_frame = ttk.Frame(field_frame)
        enable_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 8))
        self.external_field_enable_check = ttk.Checkbutton(
            enable_frame,
            text="Enable prescribed uniform external field",
            variable=self.external_field_enabled_var,
            command=self._toggle_external_field_controls,
        )
        self.external_field_enable_check.pack(side="left")
        enable_help = ttk.Label(
            enable_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        enable_help.pack(side="left", padx=(3, 0))
        Tooltip(
            enable_help,
            "Applies a prescribed uniform mechanical Lorentz-force field.\n\n"
            "Current implementation supports fixed E and B vectors with optional\n"
            "x/y/z/t windows. Time-dependent covariant potential providers are a\n"
            "future extension, not this panel.",
        )

        ttk.Label(field_frame, text="Electric field input:").grid(
            row=2, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.external_field_input_mode_combo = ttk.Combobox(
            field_frame,
            textvariable=self.external_field_input_mode_var,
            values=("SI V/m", "Native"),
            state="readonly",
            width=12,
        )
        self.external_field_input_mode_combo.grid(row=2, column=1, sticky="w", pady=2)
        self.external_field_input_mode_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._toggle_external_field_controls(),
        )

        for column, axis in enumerate(("x", "y", "z"), start=1):
            ttk.Label(field_frame, text=axis).grid(row=3, column=column, sticky="w")

        self.external_electric_si_labels = []
        self.external_electric_si_entries = []
        self.external_electric_native_labels = []
        self.external_electric_native_entries = []
        self.external_magnetic_labels = []
        self.external_magnetic_entries = []

        self.external_electric_si_label = ttk.Label(field_frame, text="E (V/m):")
        self.external_electric_si_label.grid(
            row=4, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.external_electric_si_labels.append(self.external_electric_si_label)
        for column, var in enumerate(self.external_electric_si_vars, start=1):
            entry = ttk.Entry(field_frame, textvariable=var, width=14)
            entry.grid(row=4, column=column, sticky="ew", pady=2, padx=(0, 6))
            self.external_electric_si_entries.append(entry)

        self.external_electric_native_label = ttk.Label(field_frame, text="E (native):")
        self.external_electric_native_label.grid(
            row=5, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.external_electric_native_labels.append(self.external_electric_native_label)
        for column, var in enumerate(self.external_electric_native_vars, start=1):
            entry = ttk.Entry(field_frame, textvariable=var, width=14)
            entry.grid(row=5, column=column, sticky="ew", pady=2, padx=(0, 6))
            self.external_electric_native_entries.append(entry)

        self.external_magnetic_label = ttk.Label(field_frame, text="B (native):")
        self.external_magnetic_label.grid(
            row=6, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.external_magnetic_labels.append(self.external_magnetic_label)
        for column, var in enumerate(self.external_magnetic_native_vars, start=1):
            entry = ttk.Entry(field_frame, textvariable=var, width=14)
            entry.grid(row=6, column=column, sticky="ew", pady=2, padx=(0, 6))
            self.external_magnetic_entries.append(entry)

        ttk.Label(
            field_frame,
            text="Optional field window bounds in native simulation coordinates. Leave blank for unbounded.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="gray50",
            wraplength=720,
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(12, 4), padx=(20, 0))

        window_frame = ttk.LabelFrame(field_frame, text="Field Window", padding=8)
        window_frame.grid(row=8, column=0, columnspan=4, sticky="ew", pady=(0, 12))
        window_frame.columnconfigure(1, weight=1)
        window_frame.columnconfigure(2, weight=1)
        ttk.Label(window_frame, text="min").grid(row=0, column=1, sticky="w")
        ttk.Label(window_frame, text="max").grid(row=0, column=2, sticky="w")

        self.external_field_window_entries = []
        for row, axis in enumerate(("x", "y", "z", "t"), start=1):
            ttk.Label(window_frame, text=f"{axis}:").grid(
                row=row, column=0, sticky="w", pady=2
            )
            for column, bound in enumerate(("min", "max"), start=1):
                entry = ttk.Entry(
                    window_frame,
                    textvariable=self.external_field_window_vars[f"{axis}_{bound}"],
                    width=16,
                )
                entry.grid(row=row, column=column, sticky="ew", pady=2, padx=(0, 8))
                self.external_field_window_entries.append(entry)

        self._external_field_sub_widgets = [
            self.external_field_input_mode_combo,
            self.external_electric_si_label,
            *self.external_electric_si_entries,
            self.external_electric_native_label,
            *self.external_electric_native_entries,
            self.external_magnetic_label,
            *self.external_magnetic_entries,
            *self.external_field_window_entries,
        ]

        self._toggle_external_field_controls()

    def _toggle_external_field_controls(self) -> None:
        enabled = self.external_field_enabled_var.get()
        base_state = "normal" if enabled else "disabled"
        electric_mode = self.external_field_input_mode_var.get()

        for widget in getattr(self, "_external_field_sub_widgets", []):
            try:
                widget.configure(state=base_state)
            except Exception:
                pass

        if not enabled:
            return

        native_state = "normal" if electric_mode == "Native" else "disabled"
        si_state = "normal" if electric_mode == "SI V/m" else "disabled"
        for widget in [
            self.external_electric_native_label,
            *self.external_electric_native_entries,
        ]:
            try:
                widget.configure(state=native_state)
            except Exception:
                pass
        for widget in [
            self.external_electric_si_label,
            *self.external_electric_si_entries,
        ]:
            try:
                widget.configure(state=si_state)
            except Exception:
                pass

    def _build_stability_tab(self) -> None:
        """Build self-consistency and adaptive timestep controls."""
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

        self._build_self_consistency_section(stability_frame)
        self._build_chrono_matching_section(stability_frame)
        self._build_adaptive_timestep_section(stability_frame)
        self._build_radiation_reaction_section(stability_frame)
        self._build_space_charge_section(stability_frame)
        self._build_auto_duration_section(stability_frame)

        # Help text removed - was obscuring Adaptive Timestep Refinement section
        # All parameter help is now available via ⓘ tooltips

        # Initialize control states
        self._toggle_self_consistency_controls()
        self._toggle_chrono_controls()
        self._toggle_adaptive_timestep_controls()
        self._toggle_space_charge_controls()
        self._toggle_auto_duration_controls()

    def _build_self_consistency_section(self, stability_frame: ttk.Frame) -> None:
        """Build self-consistency and gamma reconciliation controls."""
        from .gui import Tooltip

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
            "DISABLED - Default baseline. Keeps mass-shell projection without\n"
            "  additional γ blending.\n\n"
            "ADAPTIVE_WEIGHTED - Velocity-dependent blending.\n"
            "  • β < 0.9: Trust energy (weight=0.8)\n"
            "  • β > 0.99: Trust velocity (weight=0.2)\n"
            "  • Mid-range: Balanced (weight=0.5)\n\n"
            "USE_VELOCITY - Always use γ from β (can break energy consistency)\n\n"
            "USE_ENERGY - Always use γ from Pt\n\n"
            "FIXED_WEIGHTED - Fixed 50/50 blend\n\n"
            "If you need strict baseline behavior, keep DISABLED.",
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

    def _build_chrono_matching_section(self, stability_frame: ttk.Frame) -> None:
        """Build retarded-time chrono-matching controls."""
        from .gui import Tooltip

        chrono_frame = ttk.LabelFrame(
            stability_frame, text="Chrono Matching (Retarded-Time Sampling)", padding=8
        )
        chrono_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        chrono_frame.columnconfigure(1, weight=1)

        chrono_interp_frame = ttk.Frame(chrono_frame)
        chrono_interp_frame.grid(
            row=0, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
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
        chrono_tol_frame = ttk.Frame(chrono_frame)
        chrono_tol_frame.grid(row=1, column=0, sticky="w", pady=2, padx=(40, 0))
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
            chrono_frame,
            textvariable=self.chrono_tolerance_var,
            width=16,
        )
        self.sc_chrono_tolerance_entry.grid(row=1, column=1, sticky="w", pady=2)

        # Advanced chrono options (high-precision mode)
        chrono_highprec_frame = ttk.Frame(chrono_frame)
        chrono_highprec_frame.grid(
            row=2, column=0, columnspan=2, sticky="w", pady=2, padx=(40, 0)
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
        chrono_adaptive_frame = ttk.Frame(chrono_frame)
        chrono_adaptive_frame.grid(
            row=3, column=0, columnspan=2, sticky="w", pady=2, padx=(40, 0)
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

    def _build_adaptive_timestep_section(self, stability_frame: ttk.Frame) -> None:
        """Build adaptive timestep refinement controls."""
        from .gui import Tooltip

        # Adaptive timestep section (Energy Jump Detection functionality integrated here)
        at_frame = ttk.LabelFrame(
            stability_frame, text="Adaptive Timestep Refinement", padding=8
        )
        at_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 12))
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

        self.adaptive_bunch_proximity_check = ttk.Checkbutton(
            at_frame,
            text="Refine near BUNCH_TO_BUNCH encounter",
            variable=self.adaptive_timestep_bunch_proximity_enabled_var,
        )
        self.adaptive_bunch_proximity_check.grid(
            row=10, column=0, columnspan=2, sticky="w", pady=2, padx=(20, 0)
        )

        self.adaptive_bunch_proximity_sigma_label = ttk.Label(
            at_frame, text="Bunch proximity sigma (mm):"
        )
        self.adaptive_bunch_proximity_sigma_label.grid(
            row=11, column=0, sticky="w", pady=2, padx=(40, 0)
        )
        self.adaptive_bunch_proximity_sigma_entry = ttk.Entry(
            at_frame,
            textvariable=self.adaptive_timestep_bunch_proximity_sigma_mm_var,
            width=16,
        )
        self.adaptive_bunch_proximity_sigma_entry.grid(
            row=11, column=1, sticky="ew", pady=2
        )

        self.adaptive_bunch_proximity_n_sigma_label = ttk.Label(
            at_frame, text="Engage below n sigma:"
        )
        self.adaptive_bunch_proximity_n_sigma_label.grid(
            row=12, column=0, sticky="w", pady=2, padx=(40, 0)
        )
        self.adaptive_bunch_proximity_n_sigma_entry = ttk.Entry(
            at_frame,
            textvariable=self.adaptive_timestep_bunch_proximity_n_sigma_var,
            width=16,
        )
        self.adaptive_bunch_proximity_n_sigma_entry.grid(
            row=12, column=1, sticky="ew", pady=2
        )

        self.adaptive_bunch_proximity_reduction_label = ttk.Label(
            at_frame, text="Bunch proximity reduction factor:"
        )
        self.adaptive_bunch_proximity_reduction_label.grid(
            row=13, column=0, sticky="w", pady=2, padx=(40, 0)
        )
        self.adaptive_bunch_proximity_reduction_entry = ttk.Entry(
            at_frame,
            textvariable=self.adaptive_timestep_bunch_proximity_reduction_factor_var,
            width=16,
        )
        self.adaptive_bunch_proximity_reduction_entry.grid(
            row=13, column=1, sticky="ew", pady=2
        )

        self.adaptive_bunch_proximity_transition_label = ttk.Label(
            at_frame, text="Transition width (sigma):"
        )
        self.adaptive_bunch_proximity_transition_label.grid(
            row=14, column=0, sticky="w", pady=2, padx=(40, 0)
        )
        self.adaptive_bunch_proximity_transition_entry = ttk.Entry(
            at_frame,
            textvariable=(
                self.adaptive_timestep_bunch_proximity_transition_n_sigma_var
            ),
            width=16,
        )
        self.adaptive_bunch_proximity_transition_entry.grid(
            row=14, column=1, sticky="ew", pady=2
        )

        # Max sub-steps limit
        max_substeps_frame = ttk.Frame(at_frame)
        max_substeps_frame.grid(row=15, column=0, sticky="w", pady=2, padx=(20, 0))
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
            row=15, column=1, sticky="w", pady=2, padx=(10, 0)
        )

    def _build_radiation_reaction_section(self, stability_frame: ttk.Frame) -> None:
        """Build radiation-reaction mode controls."""
        from .gui import Tooltip

        rr_frame = ttk.LabelFrame(stability_frame, text="Radiation Reaction", padding=8)
        rr_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        rr_frame.columnconfigure(1, weight=1)

        mode_frame = ttk.Frame(rr_frame)
        mode_frame.grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(mode_frame, text="Mode:").pack(side="left")
        rr_help = ttk.Label(mode_frame, text="ⓘ", foreground="blue", cursor="hand2")
        rr_help.pack(side="left", padx=(3, 0))
        Tooltip(
            rr_help,
            "Radiation-reaction handling for the single-run integrator.\n\n"
            "Modes:\n"
            "  • off - No momentum change from self-radiation.\n"
            "  • diagnostic_only - Record radiated power without changing momentum.\n"
            "  • power_matched_damping - Remove radiated energy from mechanical momentum after the LW update.\n"
            "  • medina_lad - Experimental Medina/LAD candidate reaction force.\n\n"
            "Recommended default for new study runs: medina_lad.\n"
            "Use off or diagnostic_only for baselines/comparisons.",
        )
        self.radiation_reaction_mode_combo = ttk.Combobox(
            rr_frame,
            textvariable=self.radiation_reaction_mode_var,
            values=RADIATION_REACTION_MODE_CHOICES,
            state="readonly",
            width=24,
        )
        self.radiation_reaction_mode_combo.grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(
            rr_frame,
            text=(
                "Use medina_lad for normal study runs; "
                "use off/diagnostic_only for baselines and power_matched_damping for targeted comparisons."
            ),
            font=("TkDefaultFont", 8),
            foreground="gray40",
            justify="left",
            wraplength=700,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def _build_space_charge_section(self, stability_frame: ttk.Frame) -> None:
        """Build intra-bunch space-charge controls."""
        from .gui import Tooltip

        sc_frame = ttk.LabelFrame(
            stability_frame, text="Intra-Bunch Space Charge", padding=8
        )
        sc_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        sc_frame.columnconfigure(1, weight=1)

        self.space_charge_enable_check = ttk.Checkbutton(
            sc_frame,
            text="Enable intra-bunch space-charge forces (rider-rider)",
            variable=self.space_charge_enabled_var,
            command=self._toggle_space_charge_controls,
        )
        self.space_charge_enable_check.grid(
            row=0, column=0, columnspan=2, sticky="w", pady=2
        )

        ret_frame = ttk.Frame(sc_frame)
        ret_frame.grid(row=1, column=0, sticky="w", pady=2, padx=(20, 0))
        self.space_charge_retarded_label = ttk.Label(
            ret_frame, text="Use retarded fields:"
        )
        self.space_charge_retarded_label.pack(side="left")
        ret_help = ttk.Label(ret_frame, text="ⓘ", foreground="blue", cursor="hand2")
        ret_help.pack(side="left", padx=(3, 0))
        Tooltip(
            ret_help,
            "When enabled (default), rider-rider forces use full retarded\n"
            "Liénard-Wiechert fields (causal, relativistically correct).\n\n"
            "When disabled, instantaneous Coulomb forces are used instead\n"
            "(faster but not Lorentz-covariant).\n\n"
            "Recommended: keep enabled.",
        )
        self.space_charge_retarded_check = ttk.Checkbutton(
            sc_frame,
            variable=self.space_charge_retarded_var,
        )
        self.space_charge_retarded_check.grid(row=1, column=1, sticky="w", pady=2)

        soft_frame = ttk.Frame(sc_frame)
        soft_frame.grid(row=2, column=0, sticky="w", pady=2, padx=(20, 0))
        self.space_charge_softening_label = ttk.Label(
            soft_frame, text="Plummer softening (mm):"
        )
        self.space_charge_softening_label.pack(side="left")
        soft_help = ttk.Label(soft_frame, text="ⓘ", foreground="blue", cursor="hand2")
        soft_help.pack(side="left", padx=(3, 0))
        Tooltip(
            soft_help,
            "Plummer softening length ε (mm).\n\n"
            "Replaces R with sqrt(R² + ε²) in force computation,\n"
            "preventing divergence when two macroparticles are very close.\n\n"
            "0.0 = no softening (exact Coulomb/LW, default).\n"
            "Set to ~10% of typical inter-particle spacing for\n"
            "macroparticle runs with pcount > 4.",
        )
        self.space_charge_softening_entry = ttk.Entry(
            sc_frame,
            textvariable=self.space_charge_softening_mm_var,
            width=16,
        )
        self.space_charge_softening_entry.grid(row=2, column=1, sticky="ew", pady=2)

        sigma_frame = ttk.Frame(sc_frame)
        sigma_frame.grid(row=3, column=0, sticky="w", pady=2, padx=(20, 0))
        self.space_charge_sigma_label = ttk.Label(
            sigma_frame, text="Bunch sigma for retarded startup (mm):"
        )
        self.space_charge_sigma_label.pack(side="left")
        sigma_help = ttk.Label(sigma_frame, text="ⓘ", foreground="blue", cursor="hand2")
        sigma_help.pack(side="left", padx=(3, 0))
        Tooltip(
            sigma_help,
            "Characteristic bunch width used to delay retarded rider-rider\n"
            "space-charge fields until the trajectory contains at least one\n"
            "light-crossing time of intra-bunch history.\n\n"
            "Default: 0.01 mm.",
        )
        self.space_charge_sigma_entry = ttk.Entry(
            sc_frame,
            textvariable=self.space_charge_bunch_sigma_mm_var,
            width=16,
        )
        self.space_charge_sigma_entry.grid(row=3, column=1, sticky="ew", pady=2)

        min_ret_frame = ttk.Frame(sc_frame)
        min_ret_frame.grid(row=4, column=0, sticky="w", pady=2, padx=(20, 0))
        self.space_charge_min_retarded_steps_label = ttk.Label(
            min_ret_frame, text="Min retarded SC steps:"
        )
        self.space_charge_min_retarded_steps_label.pack(side="left")
        min_ret_help = ttk.Label(
            min_ret_frame, text="ⓘ", foreground="blue", cursor="hand2"
        )
        min_ret_help.pack(side="left", padx=(3, 0))
        Tooltip(
            min_ret_help,
            "Optional explicit step threshold before retarded intra-bunch\n"
            "space charge is used. Leave blank to compute it from bunch sigma\n"
            "and timestep. Set 0 only for controlled diagnostics.",
        )
        self.space_charge_min_retarded_steps_entry = ttk.Entry(
            sc_frame,
            textvariable=self.space_charge_min_retarded_steps_var,
            width=16,
        )
        self.space_charge_min_retarded_steps_entry.grid(
            row=4, column=1, sticky="ew", pady=2
        )

        self._space_charge_sub_widgets = [
            self.space_charge_retarded_label,
            self.space_charge_retarded_check,
            self.space_charge_softening_label,
            self.space_charge_softening_entry,
            self.space_charge_sigma_label,
            self.space_charge_sigma_entry,
            self.space_charge_min_retarded_steps_label,
            self.space_charge_min_retarded_steps_entry,
        ]

    def _toggle_space_charge_controls(self) -> None:
        enabled = self.space_charge_enabled_var.get()
        state = "normal" if enabled else "disabled"
        for widget in getattr(self, "_space_charge_sub_widgets", []):
            try:
                widget.configure(state=state)
            except Exception:
                pass

    def _build_auto_duration_section(self, stability_frame: ttk.Frame) -> None:
        """Build auto-duration crossing mode controls."""
        from .gui import Tooltip

        cavity_frame = ttk.LabelFrame(
            stability_frame,
            text="Cavity Exit Cutoff (BUNCH_TO_BUNCH)",
            padding=8,
        )
        cavity_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        cavity_frame.columnconfigure(1, weight=1)

        self.cavity_exit_enable_check = ttk.Checkbutton(
            cavity_frame,
            text="Stop at configured BUNCH_TO_BUNCH cavity exit",
            variable=self.cavity_exit_enabled_var,
            command=self._toggle_cavity_exit_controls,
        )
        self.cavity_exit_enable_check.grid(
            row=0, column=0, columnspan=2, sticky="w", pady=2
        )

        self.cavity_exit_mode_label = ttk.Label(cavity_frame, text="Exit mode:")
        self.cavity_exit_mode_label.grid(
            row=1, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.cavity_exit_mode_combo = ttk.Combobox(
            cavity_frame,
            textvariable=self.cavity_exit_mode_var,
            values=("first_exit", "rider_exit_with_driver_tail"),
            state="readonly",
            width=28,
        )
        self.cavity_exit_mode_combo.grid(row=1, column=1, sticky="ew", pady=2)

        self.cavity_exit_length_label = ttk.Label(
            cavity_frame, text="Cavity length override (mm):"
        )
        self.cavity_exit_length_label.grid(
            row=2, column=0, sticky="w", pady=2, padx=(20, 0)
        )
        self.cavity_exit_length_entry = ttk.Entry(
            cavity_frame, textvariable=self.cavity_exit_length_mm_var, width=12
        )
        self.cavity_exit_length_entry.grid(row=2, column=1, sticky="ew", pady=2)

        cavity_help = ttk.Label(
            cavity_frame,
            text="Blank length uses abs(initial driver z - rider z).",
            foreground="gray",
            font=("TkDefaultFont", 8),
        )
        cavity_help.grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(0, 5), padx=(20, 0)
        )
        self._cavity_exit_sub_widgets = [
            self.cavity_exit_mode_label,
            self.cavity_exit_mode_combo,
            self.cavity_exit_length_label,
            self.cavity_exit_length_entry,
            cavity_help,
        ]

        ad_frame = ttk.LabelFrame(
            stability_frame,
            text="Auto-Duration Crossing (BUNCH_TO_BUNCH)",
            padding=8,
        )
        ad_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        ad_frame.columnconfigure(1, weight=1)

        enable_frame = ttk.Frame(ad_frame)
        enable_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        self.auto_duration_enable_check = ttk.Checkbutton(
            enable_frame,
            text="Auto-compute timestep and steps from crossing geometry",
            variable=self.auto_duration_enabled_var,
            command=self._toggle_auto_duration_controls,
        )
        self.auto_duration_enable_check.pack(side="left")
        ad_help = ttk.Label(enable_frame, text="ⓘ", foreground="blue", cursor="hand2")
        ad_help.pack(side="left", padx=(3, 0))
        Tooltip(
            ad_help,
            "Derives h_step and total steps from the actual particle betas and separation\n"
            "so the simulation always covers the full crossing window.\n"
            "Overrides manual time_step and steps when enabled.",
        )

        steps_label_frame = ttk.Frame(ad_frame)
        steps_label_frame.grid(row=1, column=0, sticky="w", pady=2, padx=(20, 0))
        self.auto_duration_steps_label = ttk.Label(
            steps_label_frame, text="Crossing steps target:"
        )
        self.auto_duration_steps_label.pack(side="left")
        self.auto_duration_steps_spin = ttk.Spinbox(
            ad_frame,
            from_=10,
            to=5000,
            textvariable=self.auto_duration_crossing_steps_var,
            width=8,
        )
        self.auto_duration_steps_spin.grid(row=1, column=1, sticky="w", pady=2)

        factor_label_frame = ttk.Frame(ad_frame)
        factor_label_frame.grid(row=2, column=0, sticky="w", pady=2, padx=(20, 0))
        self.auto_duration_factor_label = ttk.Label(
            factor_label_frame, text="Post-crossing factor:"
        )
        self.auto_duration_factor_label.pack(side="left")
        self.auto_duration_factor_entry = ttk.Entry(
            ad_frame,
            textvariable=self.auto_duration_post_factor_var,
            width=8,
        )
        self.auto_duration_factor_entry.grid(row=2, column=1, sticky="w", pady=2)

        self._auto_duration_sub_widgets = [
            self.auto_duration_steps_label,
            self.auto_duration_steps_spin,
            self.auto_duration_factor_label,
            self.auto_duration_factor_entry,
        ]

    def _toggle_auto_duration_controls(self) -> None:
        # Implemented in gui_state_mixins.IntegratorGUIStateMixin
        pass
