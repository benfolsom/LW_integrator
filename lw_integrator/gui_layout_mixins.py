"""Layout support helpers for the main GUI."""

from __future__ import annotations

from typing import Any, Optional, Set

import tkinter as tk
from tkinter import scrolledtext, ttk

from .testbed_runner import (
    AVAILABLE_DPI_CHOICES,
    CORE_PARAM_LABELS,
    PARAM_LABELS,
    PARTICLE_PARAM_FIELDS,
    SPECIES_OPTIONS,
)


CONTENT_PANEL_MIN_WIDTH = 800  # pixels; keeps tab content usable
CONFIG_PANEL_MIN_WIDTH = 450  # pixels; keeps right-side controls readable


class _ScrollableNotebookPage:
    def __init__(self, notebook: ttk.Notebook, title: str, padding: int = 12) -> None:
        self.container = ttk.Frame(notebook)
        notebook.add(self.container, text=title)
        self.container.columnconfigure(0, weight=1)
        self.container.rowconfigure(0, weight=1)
        self._bound_widgets: Set[int] = set()

        self.canvas = tk.Canvas(self.container, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(
            self.container, orient="vertical", command=self.canvas.yview
        )
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.h_scrollbar = ttk.Scrollbar(
            self.container, orient="horizontal", command=self.canvas.xview
        )
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(
            yscrollcommand=self.scrollbar.set, xscrollcommand=self.h_scrollbar.set
        )

        self.frame = ttk.Frame(self.canvas, padding=padding)
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.frame, anchor="nw"
        )

        self.frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self._mousewheel_bound = False
        self._bind_mousewheel(self.container)
        self._bind_mousewheel(self.canvas)
        self._bind_mousewheel(self.frame)

    def _on_frame_configure(self, _event: Any) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: Any) -> None:
        canvas_width = event.width
        bbox = self.canvas.bbox("all")
        if not bbox:
            return

        content_width = bbox[2] - bbox[0]
        if canvas_width >= content_width:
            self.canvas.itemconfigure(self._window_id, width=canvas_width)
        else:
            self.canvas.itemconfigure(self._window_id, width=content_width)

    def _bind_mousewheel(self, widget: tk.Widget) -> None:
        widget_id = widget.winfo_id()
        if widget_id in self._bound_widgets:
            return
        self._bound_widgets.add(widget_id)
        widget.bind("<Enter>", lambda _event: self._activate_mousewheel(), add=True)
        widget.bind(
            "<Leave>", lambda _event: self._maybe_deactivate_mousewheel(), add=True
        )

    def refresh_mousewheel_bindings(self) -> None:
        self._register_descendants(self.frame)

    def _register_descendants(self, widget: tk.Widget) -> None:
        self._bind_mousewheel(widget)
        for child in widget.winfo_children():
            self._register_descendants(child)

    def _activate_mousewheel(self) -> None:
        if self._mousewheel_bound:
            return
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        self._mousewheel_bound = True

    def _maybe_deactivate_mousewheel(self) -> None:
        if not self._mousewheel_bound:
            return
        widget = self.container.winfo_containing(
            self.container.winfo_pointerx(), self.container.winfo_pointery()
        )
        if not self._is_descendant(widget):
            self._deactivate_mousewheel()

    def _deactivate_mousewheel(self) -> None:
        if not self._mousewheel_bound:
            return
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
        self._mousewheel_bound = False

    def _is_descendant(self, widget: Optional[tk.Widget]) -> bool:
        while widget is not None:
            if widget == self.container:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _on_mousewheel(self, event: Any) -> None:
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
        self.canvas.yview_scroll(delta, "units")


class IntegratorGUILayoutMixin:
    """Own the persistent config panel and layout utility methods."""

    def _enforce_panel_minimums(self, event=None):
        """Enforce minimum panel sizes when sash is moved."""
        if not hasattr(self, "_main_horizontal_paned"):
            return

        try:
            sash_pos = self._main_horizontal_paned.sash_coord(0)[0]
            total_width = self._main_horizontal_paned.winfo_width()
            min_left = CONTENT_PANEL_MIN_WIDTH
            max_left = total_width - CONFIG_PANEL_MIN_WIDTH
            if sash_pos < min_left:
                self._main_horizontal_paned.sash_place(0, min_left, 0)
            elif sash_pos > max_left:
                self._main_horizontal_paned.sash_place(0, max_left, 0)
        except Exception:
            pass

    def _create_scrollable_tab(
        self, notebook: ttk.Notebook, title: str, *, padding: int = 12
    ) -> ttk.Frame:
        page = _ScrollableNotebookPage(notebook, title, padding=padding)
        self._scroll_pages.append(page)
        return page.frame

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

    def _build_config_panel(self, parent):
        """Build persistent config/control panel on right side."""
        panel = ttk.LabelFrame(parent, text="Configuration & Control", padding=10)
        panel.pack(fill="both", expand=True, padx=5, pady=5)

        scroll_container = ttk.Frame(panel)
        scroll_container.pack(fill="both", expand=True, side="top", pady=(0, 5))

        canvas = tk.Canvas(
            scroll_container,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            scroll_container, orient="vertical", command=canvas.yview
        )
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_resize(event):
            canvas.itemconfig(window_id, width=event.width)

        canvas.bind("<Configure>", _on_canvas_resize)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        run_config_frame = ttk.LabelFrame(
            scrollable_frame, text="Single Run Configuration", padding=4
        )
        run_config_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(run_config_frame, text="Config dir:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Entry(run_config_frame, textvariable=self.config_dir_var, width=20).grid(
            row=0, column=1, sticky="ew", pady=2, padx=(5, 2)
        )
        ttk.Button(
            run_config_frame, text="...", command=self._select_config_dir, width=3
        ).grid(row=0, column=2, sticky="w", pady=2)

        ttk.Label(run_config_frame, text="Output dir:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Entry(run_config_frame, textvariable=self.output_dir_var, width=20).grid(
            row=1, column=1, sticky="ew", pady=2, padx=(5, 2)
        )
        ttk.Button(
            run_config_frame, text="...", command=self._select_output_dir, width=3
        ).grid(row=1, column=2, sticky="w", pady=2)

        run_config_frame.columnconfigure(1, weight=1)

        ttk.Label(run_config_frame, text="Config name:").grid(
            row=2, column=0, sticky="w", pady=(10, 2)
        )
        ttk.Entry(run_config_frame, textvariable=self.config_name_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(10, 2)
        )

        ttk.Label(run_config_frame, text="Current:").grid(
            row=3, column=0, sticky="w", pady=2
        )
        self.current_config_label = ttk.Label(
            run_config_frame,
            text="<unsaved>",
            foreground="gray",
            font=("TkDefaultFont", 9, "italic"),
        )
        self.current_config_label.grid(
            row=3, column=1, columnspan=2, sticky="w", pady=2
        )

        ttk.Label(run_config_frame, text="Saved configs:").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(10, 2)
        )

        run_list_frame = ttk.Frame(run_config_frame)
        run_list_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=2)
        run_list_frame.rowconfigure(0, weight=1)
        run_list_frame.columnconfigure(0, weight=1)

        self.config_list = tk.Listbox(run_list_frame, height=9)
        self.config_list.grid(row=0, column=0, sticky="nsew")
        self.config_list.bind(
            "<<ListboxSelect>>", lambda _event: self._on_config_selected()
        )
        self.config_list.bind("<Double-1>", lambda _event: self._load_config())

        run_scrollbar = ttk.Scrollbar(
            run_list_frame, orient="vertical", command=self.config_list.yview
        )
        run_scrollbar.grid(row=0, column=1, sticky="ns")
        self.config_list.configure(yscrollcommand=run_scrollbar.set)

        run_config_frame.rowconfigure(5, weight=1)

        run_btn_frame = ttk.Frame(run_config_frame)
        run_btn_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(5, 0))

        ttk.Button(run_btn_frame, text="Load", command=self._load_config, width=8).pack(
            side="left", padx=2
        )
        ttk.Button(run_btn_frame, text="Save", command=self._save_config, width=8).pack(
            side="left", padx=2
        )
        ttk.Button(
            run_btn_frame, text="Refresh", command=self._refresh_config_list, width=8
        ).pack(side="left", padx=2)

        sweep_config_frame = ttk.LabelFrame(
            scrollable_frame, text="Sweep Configuration", padding=4
        )
        sweep_config_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(sweep_config_frame, text="Config dir:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.sweep_config_dir_var = tk.StringVar(value=self._last_sweep_config_dir)
        ttk.Entry(
            sweep_config_frame, textvariable=self.sweep_config_dir_var, width=20
        ).grid(row=0, column=1, sticky="ew", pady=2, padx=(5, 2))
        ttk.Button(
            sweep_config_frame,
            text="...",
            command=self._select_sweep_config_dir,
            width=3,
        ).grid(row=0, column=2, sticky="w", pady=2)

        ttk.Label(sweep_config_frame, text="Output dir:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.sweep_output_dir_var = tk.StringVar(value=self._last_sweep_output_dir)
        ttk.Entry(
            sweep_config_frame, textvariable=self.sweep_output_dir_var, width=20
        ).grid(row=1, column=1, sticky="ew", pady=2, padx=(5, 2))
        ttk.Button(
            sweep_config_frame,
            text="...",
            command=self._select_sweep_output_dir,
            width=3,
        ).grid(row=1, column=2, sticky="w", pady=2)

        sweep_config_frame.columnconfigure(1, weight=1)

        ttk.Label(sweep_config_frame, text="Config name:").grid(
            row=2, column=0, sticky="w", pady=(10, 2)
        )
        ttk.Entry(sweep_config_frame, textvariable=self.sweep_config_name_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(10, 2)
        )

        ttk.Label(sweep_config_frame, text="Current:").grid(
            row=3, column=0, sticky="w", pady=(5, 2)
        )
        self.current_sweep_config_label = ttk.Label(
            sweep_config_frame,
            text="<none>",
            foreground="gray",
            font=("TkDefaultFont", 9, "italic"),
        )
        self.current_sweep_config_label.grid(
            row=3, column=1, columnspan=2, sticky="w", pady=(5, 2)
        )

        ttk.Label(sweep_config_frame, text="Saved configs:").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(10, 2)
        )

        sweep_list_frame = ttk.Frame(sweep_config_frame)
        sweep_list_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=2)
        sweep_list_frame.rowconfigure(0, weight=1)
        sweep_list_frame.columnconfigure(0, weight=1)

        self.sweep_config_list = tk.Listbox(sweep_list_frame, height=9)
        self.sweep_config_list.grid(row=0, column=0, sticky="nsew")
        self.sweep_config_list.bind(
            "<Double-1>", lambda _event: self._load_sweep_config()
        )
        self.sweep_config_list.bind(
            "<<ListboxSelect>>", lambda _event: self._on_sweep_config_selected()
        )

        sweep_scrollbar = ttk.Scrollbar(
            sweep_list_frame, orient="vertical", command=self.sweep_config_list.yview
        )
        sweep_scrollbar.grid(row=0, column=1, sticky="ns")
        self.sweep_config_list.configure(yscrollcommand=sweep_scrollbar.set)

        sweep_config_frame.rowconfigure(5, weight=1)

        sweep_btn_frame = ttk.Frame(sweep_config_frame)
        sweep_btn_frame.grid(
            row=6, column=0, columnspan=3, sticky="ew", pady=(5, 0)
        )

        ttk.Button(
            sweep_btn_frame, text="Load", command=self._load_sweep_config, width=8
        ).pack(side="left", padx=2)
        ttk.Button(
            sweep_btn_frame, text="Save", command=self._save_sweep_config, width=8
        ).pack(side="left", padx=2)
        ttk.Button(
            sweep_btn_frame,
            text="Refresh",
            command=self._refresh_sweep_config_list,
            width=8,
        ).pack(side="left", padx=2)

        reset_frame = ttk.Frame(scrollable_frame)
        reset_frame.pack(fill="x", pady=(0, 10))
        ttk.Button(
            reset_frame,
            text="Reset All Directories to Defaults",
            command=self._reset_directories_to_defaults,
        ).pack(fill="x")

        status_frame = ttk.LabelFrame(panel, text="Status", padding=4)
        status_frame.pack(side="bottom", fill="x")

        self._refresh_sweep_config_list()
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w", pady=2)

        self._progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self._progress_bar.pack(fill="x", pady=5)

        control_frame = ttk.LabelFrame(panel, text="Controls", padding=4)
        control_frame.pack(side="bottom", fill="x", pady=(0, 5))

        self._run_button = ttk.Button(
            control_frame,
            text="▶ Run",
            command=self._trigger_run,
            style="Accent.TButton",
        )
        self._run_button.pack(fill="x", pady=2)
        self._run_button.configure(width=12)

        self._cancel_button = ttk.Button(
            control_frame,
            text="⬛ Cancel",
            command=self._trigger_cancel,
            state="disabled",
        )
        self._cancel_button.pack(fill="x", pady=2)
        self._cancel_button.configure(width=12)

        mode_frame = ttk.LabelFrame(panel, text="Run Mode", padding=4)
        mode_frame.pack(side="bottom", fill="x", pady=(0, 5))

        self.run_mode_var = tk.StringVar(value="single")

        ttk.Radiobutton(
            mode_frame,
            text="Single Run",
            variable=self.run_mode_var,
            value="single",
            command=self._on_run_mode_changed,
        ).pack(anchor="w", pady=2)

        ttk.Radiobutton(
            mode_frame,
            text="Sweep/Optim",
            variable=self.run_mode_var,
            value="sweep",
            command=self._on_run_mode_changed,
        ).pack(anchor="w", pady=2)

    def _on_run_mode_changed(self):
        """Handle run mode selection change."""
        mode = self.run_mode_var.get()
        if mode == "single":
            self._run_button.config(text="▶ Run", command=self._trigger_run)
        else:
            self._run_button.config(text="▶ Run Sweep", command=self._trigger_sweep)

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

    def _build_log_summary_panel(self, bottom_container: ttk.Frame) -> None:
        """Build the lower split panel for logs and initial summary."""
        lower_paned = ttk.Panedwindow(bottom_container, orient="horizontal")
        lower_paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        log_frame = ttk.LabelFrame(lower_paned, text="Logs", padding=8)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=0)
        log_frame.rowconfigure(1, weight=1)

        log_controls = ttk.Frame(log_frame)
        log_controls.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self.log_format_var = tk.StringVar(value="detailed")
        ttk.Radiobutton(
            log_controls,
            text="Summary",
            variable=self.log_format_var,
            value="summary",
            command=self._update_log_format,
        ).pack(side="left", padx=5)
        ttk.Radiobutton(
            log_controls,
            text="Detailed",
            variable=self.log_format_var,
            value="detailed",
            command=self._update_log_format,
        ).pack(side="left", padx=5)

        ttk.Button(log_controls, text="Clear", command=self._clear_log, width=8).pack(
            side="right", padx=5
        )

        self.log_output = scrolledtext.ScrolledText(
            log_frame, height=6, state="disabled", wrap="none"
        )
        self.log_output.grid(row=1, column=0, sticky="nsew")

        self._raw_log_lines = []
        self._log_summary = []

        lower_paned.add(log_frame, weight=3)

        summary_frame = ttk.LabelFrame(lower_paned, text="Initial summary", padding=8)
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)

        summary_text_frame = ttk.Frame(summary_frame)
        summary_text_frame.grid(row=0, column=0, sticky="nsew")
        summary_text_frame.columnconfigure(0, weight=1)
        summary_text_frame.rowconfigure(0, weight=1)

        self.summary_text = tk.Text(
            summary_text_frame,
            height=3,
            width=40,
            wrap="word",
            state="disabled",
            relief="flat",
            borderwidth=0,
        )
        self.summary_text.grid(row=0, column=0, sticky="nsew")

        summary_scrollbar = ttk.Scrollbar(
            summary_text_frame, command=self.summary_text.yview
        )
        summary_scrollbar.grid(row=0, column=1, sticky="ns")
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)

        lower_paned.add(summary_frame, weight=1)
