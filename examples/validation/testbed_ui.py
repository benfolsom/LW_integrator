from __future__ import annotations

import html
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from core_vs_legacy_benchmark import (
    DEFAULT_DRIVER_PARAMS,
    DEFAULT_RIDER_PARAMS,
    FIELDS_TO_TRACK,
    PARTICLE_PARAM_FIELDS,
    SimulationType,
    compute_delta_energy_series,
    prepare_two_particle_demo,
    run_benchmark,
    summarise_metrics,
)

COLOR_RIDER = "#0072B2"
COLOR_DRIVER = "#D55E00"
COLOR_LEGACY_RIDER = "#56B4E9"
COLOR_LEGACY_DRIVER = "#E69F00"
COLOR_DIFF_RIDER = "#009E73"
COLOR_DIFF_DRIVER = "#CC79A7"
SCATTER_STYLE = {"s": 140, "alpha": 0.78, "linewidth": 0, "edgecolors": "none"}
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 12
AVAILABLE_DPI_CHOICES = (150, 300, 450, 600)
DEFAULT_PLOT_DPI = 300

PARTICLE_WIDGET_WIDTH = widgets.Layout(width="320px")
CORE_WIDGET_WIDTH = widgets.Layout(width="280px")
PARTICLE_WIDGET_STYLE = {"description_width": "175px"}
CORE_WIDGET_STYLE = {"description_width": "175px"}
ROW_LAYOUT = widgets.Layout(gap="14px")

PARAM_LABELS = {
    "starting_distance": "Start z (mm)",
    "transv_mom": "Transverse momentum (amu·mm/ns)",
    "starting_Pz": "Initial Pz (amu·mm/ns)",
    "stripped_ions": "Stripped ions",
    "m_particle": "Mass (amu)",
    "transv_dist": "Transverse spread (mm)",
    "pcount": "Particle count",
    "charge_sign": "Charge sign",
}

CORE_PARAM_LABELS = {
    "time_step": "Time step (ns)",
    "wall_z": "Wall z (mm)",
    "aperture_radius": "Aperture radius (mm)",
    "mean": "Mean separation (mm)",
    "cav_spacing": "Cavity spacing (mm)",
    "z_cutoff": "z cutoff (mm)",
}

SIMULATION_TYPE_OPTIONS = {
    "Conducting wall": SimulationType.CONDUCTING_WALL,
    "Switching wall": SimulationType.SWITCHING_WALL,
    "Bunch to bunch": SimulationType.BUNCH_TO_BUNCH,
}

SPECIES_PRESETS: Dict[str, Optional[Dict[str, float]]] = {
    "custom": None,
    "electron": {
        "m_particle": 5.48579909070e-4,
        "charge_sign": -1.0,
        "stripped_ions": 1.0,
    },
    "positron": {
        "m_particle": 5.48579909070e-4,
        "charge_sign": 1.0,
        "stripped_ions": 1.0,
    },
    "proton": {
        "m_particle": 1.007276466621,
        "charge_sign": 1.0,
        "stripped_ions": 1.0,
    },
    "antiproton": {
        "m_particle": 1.007276466621,
        "charge_sign": -1.0,
        "stripped_ions": 1.0,
    },
    "lead": {
        "m_particle": 207.9766521,
        "charge_sign": 1.0,
        "stripped_ions": 82.0,
    },
    "gold": {
        "m_particle": 196.9665687,
        "charge_sign": 1.0,
        "stripped_ions": 79.0,
    },
}

SPECIES_DROPDOWN_OPTIONS = [
    ("Custom / manual", "custom"),
    ("Electron (e⁻)", "electron"),
    ("Positron (e⁺)", "positron"),
    ("Proton (p⁺)", "proton"),
    ("Antiproton (p̄⁻)", "antiproton"),
    ("Lead ion (Pb⁸²⁺)", "lead"),
    ("Gold ion (Au⁷⁹⁺)", "gold"),
]

CORE_PARAM_DEFAULTS = {
    "time_step": 2.2e-7,
    "wall_z": 1e5,
    "aperture_radius": 1e5,
    "mean": 1e5,
    "cav_spacing": 1e5,
    "z_cutoff": 0.0,
}

CORE_REQUIRED_PARAMS = {
    SimulationType.CONDUCTING_WALL: {"time_step", "wall_z", "aperture_radius"},
    SimulationType.SWITCHING_WALL: {
        "time_step",
        "wall_z",
        "aperture_radius",
        "cav_spacing",
        "z_cutoff",
    },
    SimulationType.BUNCH_TO_BUNCH: {"time_step", "aperture_radius"},
}

TRAJECTORY_MODE_OPTIONS = [
    ("All particles", "all"),
    ("Average per step", "average"),
]

TRAJECTORY_INTERVAL_OPTIONS = [
    ("Every step", 1),
    ("Every 2nd step", 2),
    ("Every 3rd step", 3),
    ("Every 5th step", 5),
    ("Every 10th step", 10),
]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
    }
)


class IntegratorTestbedApp:
    """Encapsulates the interactive integrator testbed UI."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or self._detect_project_root()
        self.validation_root = self.project_root / "examples" / "validation"

        self.last_metrics: Optional[Dict[str, Dict[str, float]]] = None
        self.last_payload: Optional[dict] = None
        self.last_log_messages: List[str] = []

        self._build_widgets()
        self._wire_events()
        self._update_legacy_controls({"new": self.legacy_toggle.value})
        self._update_core_controls()
        self._refresh_config_file_options()
        self._refresh_initial_properties()
        self._update_trajectory_controls({"new": self.trajectory_save_checkbox.value})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def display(self) -> None:
        """Display the composed widget layout."""
        display(self.controls_layout)

    # ------------------------------------------------------------------
    # Widget construction helpers
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        self.steps_widget = widgets.IntSlider(
            value=1000,
            min=10,
            max=20000,
            step=10,
            description="Steps:",
            continuous_update=False,
        )
        self.seed_widget = widgets.IntText(value=12345, description="Seed:")
        self.legacy_toggle = widgets.Checkbox(
            value=False, description="Include legacy comparison"
        )
        self.simulation_widget = widgets.Dropdown(
            options=[(label, value) for label, value in SIMULATION_TYPE_OPTIONS.items()],
            value=SimulationType.BUNCH_TO_BUNCH,
            description="Core mode:",
        )
        self.overlay_display_widget = widgets.Checkbox(
            value=False, description="Show legacy overlay plot"
        )
        self.overlay_save_widget = widgets.Checkbox(
            value=False, description="Save legacy overlay plot"
        )
        self.difference_display_widget = widgets.Checkbox(
            value=False, description="Show Δ(core−legacy) plot"
        )
        self.difference_save_widget = widgets.Checkbox(
            value=False, description="Save Δ(core−legacy) plot"
        )
        self.metrics_save_widget = widgets.Checkbox(
            value=False, description="Save metrics JSON"
        )
        self.energy_save_widget = widgets.Checkbox(
            value=False, description="Save ΔE plots"
        )
        self.dpi_widget = widgets.Dropdown(
            options=[(f"{value} dpi", value) for value in AVAILABLE_DPI_CHOICES],
            value=DEFAULT_PLOT_DPI,
            description="Plot DPI:",
            layout=widgets.Layout(width="320px"),
            style=dict(CORE_WIDGET_STYLE, description_width="130px"),
        )
        self.output_dir_widget = widgets.Text(
            value="test_outputs/testbed_runs",
            description="Plot dir:",
            layout=widgets.Layout(width="360px"),
            style=CORE_WIDGET_STYLE,
        )
        self.config_dir_widget = widgets.Text(
            value="configs/testbed_runs",
            description="Config dir:",
            layout=widgets.Layout(width="360px"),
            style=CORE_WIDGET_STYLE,
        )
        self.config_name_widget = widgets.Text(
            value="testbed_config.json",
            description="Config name:",
            layout=widgets.Layout(width="320px"),
            style=CORE_WIDGET_STYLE,
        )
        self.config_file_dropdown = widgets.Dropdown(
            options=[("No saved configs", "")],
            value="",
            description="Existing:",
            layout=widgets.Layout(width="320px"),
            style=dict(CORE_WIDGET_STYLE, description_width="130px"),
            disabled=True,
        )
        self.config_refresh_button = widgets.Button(
            description="Refresh list", icon="refresh"
        )
        self.save_config_button = widgets.Button(
            description="Save config", icon="save"
        )
        self.load_config_button = widgets.Button(
            description="Load config", icon="folder-open"
        )
        self.trajectory_save_checkbox = widgets.Checkbox(
            value=False, description="Save trajectories"
        )
        self.trajectory_mode_dropdown = widgets.Dropdown(
            options=TRAJECTORY_MODE_OPTIONS,
            value="all",
            description="Particle mode:",
            layout=widgets.Layout(width="320px"),
            style=dict(CORE_WIDGET_STYLE, description_width="130px"),
        )
        self.trajectory_interval_dropdown = widgets.Dropdown(
            options=TRAJECTORY_INTERVAL_OPTIONS,
            value=1,
            description="Step interval:",
            layout=widgets.Layout(width="320px"),
            style=dict(CORE_WIDGET_STYLE, description_width="130px"),
        )
        self.run_button = widgets.Button(
            description="Run integrator", icon="play", button_style="primary"
        )
        self.output_area = widgets.Output()
        self.config_feedback_output = widgets.Output(
            layout=widgets.Layout(border="1px solid #dddddd", padding="8px", margin="4px 0")
        )
        self.initial_state_output = widgets.Output(
            layout=widgets.Layout(border="1px solid #dddddd", padding="8px", margin="6px 0")
        )

        self.rider_controls = self._make_particle_widgets(DEFAULT_RIDER_PARAMS)
        self.driver_controls = self._make_particle_widgets(DEFAULT_DRIVER_PARAMS)
        self.core_controls = self._make_core_widgets()

        self.rider_species_widget = widgets.Dropdown(
            options=SPECIES_DROPDOWN_OPTIONS,
            value="custom",
            description="Rider species:",
            layout=PARTICLE_WIDGET_WIDTH,
            style=PARTICLE_WIDGET_STYLE,
        )
        self.driver_species_widget = widgets.Dropdown(
            options=SPECIES_DROPDOWN_OPTIONS,
            value="custom",
            description="Driver species:",
            layout=PARTICLE_WIDGET_WIDTH,
            style=PARTICLE_WIDGET_STYLE,
        )

        self.rider_section = self._build_particle_section(
            "Rider particle", self.rider_controls, self.rider_species_widget
        )
        self.driver_section = self._build_particle_section(
            "Driver particle", self.driver_controls, self.driver_species_widget
        )
        self.core_section = self._build_core_section(self.core_controls)

        config_controls = widgets.VBox(
            [
                widgets.HBox(
                    [self.config_dir_widget, self.config_refresh_button],
                    layout=ROW_LAYOUT,
                ),
                widgets.HBox(
                    [self.config_file_dropdown, self.load_config_button],
                    layout=ROW_LAYOUT,
                ),
                widgets.HBox(
                    [self.config_name_widget, self.save_config_button],
                    layout=ROW_LAYOUT,
                ),
                self.config_feedback_output,
            ],
            layout=widgets.Layout(padding="4px 0"),
        )

        trajectory_controls = widgets.HBox(
            [
                self.trajectory_save_checkbox,
                self.trajectory_mode_dropdown,
                self.trajectory_interval_dropdown,
            ],
            layout=ROW_LAYOUT,
        )

        self.controls_layout = widgets.VBox(
            [
                widgets.HBox(
                    [self.steps_widget, self.seed_widget, self.simulation_widget],
                    layout=ROW_LAYOUT,
                ),
                widgets.HBox(
                    [
                        self.legacy_toggle,
                        self.overlay_display_widget,
                        self.overlay_save_widget,
                        self.difference_display_widget,
                        self.difference_save_widget,
                    ],
                    layout=ROW_LAYOUT,
                ),
                widgets.HBox(
                    [
                        self.metrics_save_widget,
                        self.energy_save_widget,
                        self.dpi_widget,
                    ],
                    layout=ROW_LAYOUT,
                ),
                trajectory_controls,
                self.output_dir_widget,
                config_controls,
                self.core_section,
                self.rider_section,
                self.driver_section,
                self.initial_state_output,
                self.run_button,
                self.output_area,
            ]
        )

    def _wire_events(self) -> None:
        self.rider_species_widget.observe(self._on_rider_species, names="value")
        self.driver_species_widget.observe(self._on_driver_species, names="value")
        self.legacy_toggle.observe(
            lambda change: self._update_legacy_controls(change), names="value"
        )
        self.simulation_widget.observe(self._on_simulation_change, names="value")
        self.seed_widget.observe(self._refresh_initial_properties, names="value")
        for control in self.rider_controls.values():
            control.observe(self._refresh_initial_properties, names="value")
        for control in self.driver_controls.values():
            control.observe(self._refresh_initial_properties, names="value")
        for control in self.core_controls.values():
            control.observe(self._refresh_initial_properties, names="value")
        self.config_dir_widget.observe(self._on_config_dir_change, names="value")
        self.config_file_dropdown.observe(
            self._on_config_file_selected, names="value"
        )
        self.config_refresh_button.on_click(
            lambda _: self._refresh_config_file_options()
        )
        self.save_config_button.on_click(self._handle_save_config)
        self.load_config_button.on_click(self._handle_load_config)
        self.run_button.on_click(self.handle_run)
        self.trajectory_save_checkbox.observe(
            self._update_trajectory_controls, names="value"
        )

    # ------------------------------------------------------------------
    # Event handlers and callbacks
    # ------------------------------------------------------------------
    def _update_legacy_controls(self, change) -> None:
        legacy_enabled = bool(change["new"] if isinstance(change, dict) else change)
        for checkbox in (
            self.overlay_display_widget,
            self.overlay_save_widget,
            self.difference_display_widget,
            self.difference_save_widget,
            self.metrics_save_widget,
        ):
            checkbox.disabled = not legacy_enabled
            if not legacy_enabled:
                checkbox.value = False

    def _on_simulation_change(self, change) -> None:
        self._update_core_controls(change)

    def _on_rider_species(self, change) -> None:
        new_value = change.get("new") if isinstance(change, dict) else change
        if not new_value or new_value == change.get("old"):
            return
        self._apply_species_preset(self.rider_controls, str(new_value))
        self._refresh_initial_properties()

    def _on_driver_species(self, change) -> None:
        new_value = change.get("new") if isinstance(change, dict) else change
        if not new_value or new_value == change.get("old"):
            return
        self._apply_species_preset(self.driver_controls, str(new_value))
        self._refresh_initial_properties()

    def _on_config_dir_change(self, change) -> None:
        self._refresh_config_file_options()

    def _on_config_file_selected(self, change) -> None:
        new_value = change.get("new")
        if not new_value:
            return
        self.config_name_widget.value = new_value

    def _update_trajectory_controls(self, change) -> None:
        enabled = bool(change.get("new")) if isinstance(change, dict) else bool(change)
        for control in (self.trajectory_mode_dropdown, self.trajectory_interval_dropdown):
            control.disabled = not enabled
            control.layout.opacity = 1.0 if enabled else 0.45

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def handle_run(self, _button) -> None:
        with self.output_area:
            self.output_area.clear_output(wait=True)
            legacy_enabled = self.legacy_toggle.value
            sim_type_value = self.simulation_widget.value
            supports_driver = self._supports_driver(sim_type_value)
            rider_params = self._collect_particle_values(self.rider_controls)
            driver_params = (
                self._collect_particle_values(self.driver_controls)
                if supports_driver
                else None
            )
            core_params = self._collect_core_values(self.core_controls)
            required_core = self._required_params_for(sim_type_value)
            needs_dir = any(
                [
                    self.overlay_save_widget.value,
                    self.difference_save_widget.value,
                    self.metrics_save_widget.value,
                    self.energy_save_widget.value,
                    self.trajectory_save_checkbox.value,
                ]
            )
            save_dir = self._save_dir_if_needed(needs_dir)
            save_json_path = (
                save_dir / "benchmark_metrics.json"
                if legacy_enabled and self.metrics_save_widget.value
                else None
            )
            save_overlay_path = (
                save_dir / "core_legacy_overlay.png"
                if legacy_enabled and self.overlay_save_widget.value
                else None
            )
            save_difference_path = (
                save_dir / "core_legacy_delta_energy_difference.png"
                if legacy_enabled and self.difference_save_widget.value
                else None
            )
            save_energy_path = (
                save_dir / "delta_energy_scatter.png"
                if self.energy_save_widget.value
                else None
            )
            plot_dpi_value = self.dpi_widget.value
            time_step_value = core_params["time_step"]
            wall_z_value = core_params.get("wall_z") if "wall_z" in required_core else None
            aperture_radius_value = (
                core_params.get("aperture_radius")
                if "aperture_radius" in required_core
                else None
            )
            mean_value = (
                core_params.get("mean") if "mean" in required_core else None
            )
            cav_spacing_value = (
                core_params.get("cav_spacing")
                if "cav_spacing" in required_core
                else None
            )
            z_cutoff_value = (
                core_params.get("z_cutoff")
                if "z_cutoff" in required_core
                else None
            )
            log_messages: List[str] = []
            if legacy_enabled and supports_driver and driver_params is not None:
                rider_count = int(rider_params.get("pcount", 1))
                driver_count = int(driver_params.get("pcount", rider_count))
                if rider_count != driver_count:
                    driver_params["pcount"] = rider_count
                    with self.driver_controls["pcount"].hold_trait_notifications():
                        self.driver_controls["pcount"].value = rider_count
                    log_messages.append(
                        "Driver particle count synchronised with rider count for the legacy integrator.",
                    )

            metrics, payload = run_benchmark(
                steps=self.steps_widget.value,
                seed=self.seed_widget.value,
                rider_params=rider_params,
                driver_params=driver_params,
                legacy_enabled=legacy_enabled,
                simulation_type=sim_type_value,
                time_step=time_step_value,
                wall_z=wall_z_value,
                aperture_radius=aperture_radius_value,
                mean=mean_value,
                cav_spacing=cav_spacing_value,
                z_cutoff=z_cutoff_value,
                save_json=save_json_path,
                save_fig=None,
                show=False,
                plot=False,
                return_trajectories=True,
                plot_dpi=plot_dpi_value,
                log_messages=log_messages,
            )

            self.last_metrics = metrics
            self.last_payload = payload
            self.last_log_messages = list(log_messages)

            if legacy_enabled:
                log_messages.append("Legacy comparison completed successfully.")

            rider_delta, rider_z = compute_delta_energy_series(
                payload["core"]["rider"],
                payload["initial_states"]["rider"],
                payload["rest_energy_mev"]["rider"],
            )
            rider_z_rel = rider_z - rider_z[0]
            axis_count = 2 if supports_driver else 1
            fig_width = 13 if supports_driver else 7.5
            fig, axes = plt.subplots(
                1,
                axis_count,
                figsize=(fig_width, 6),
                constrained_layout=True,
                dpi=plot_dpi_value,
            )
            if axis_count == 1:
                axes = [axes]
            fig.patch.set_facecolor("white")
            axes[0].scatter(
                rider_z_rel,
                rider_delta,
                color=COLOR_RIDER,
                label="Rider ΔE",
                **SCATTER_STYLE,
            )
            axes[0].set_title("Rider ΔE vs Δz")
            axes[0].set_xlabel("Δz (mm)")
            axes[0].set_ylabel("ΔE (GeV)")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            if supports_driver:
                driver_delta, driver_z = compute_delta_energy_series(
                    payload["core"]["driver"],
                    payload["initial_states"]["driver"],
                    payload["rest_energy_mev"]["driver"],
                )
                driver_z_rel = driver_z - driver_z[0]
                axes[1].scatter(
                    driver_z_rel,
                    driver_delta,
                    color=COLOR_DRIVER,
                    label="Driver ΔE",
                    **SCATTER_STYLE,
                )
                axes[1].set_title("Driver ΔE vs Δz")
                axes[1].set_xlabel("Δz (mm)")
                axes[1].set_ylabel("ΔE (GeV)")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
            else:
                log_messages.append("Driver outputs suppressed for wall-mode simulations.")
            for axis in axes:
                axis.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
            if save_energy_path is not None:
                fig.savefig(save_energy_path, dpi=plot_dpi_value, bbox_inches="tight")
                log_messages.append(f"ΔE scatter saved to {save_energy_path}")
            figure_buffer = io.BytesIO()
            fig.savefig(figure_buffer, format="png", dpi=plot_dpi_value, bbox_inches="tight")
            figure_buffer.seek(0)
            figure_widget = widgets.Image(
                value=figure_buffer.getvalue(),
                format="png",
                layout=widgets.Layout(max_width="100%"),
            )
            plt.close(fig)
            figure_widgets: List[widgets.Widget] = [figure_widget]

            legacy_data = payload.get("legacy") if legacy_enabled else None
            if legacy_enabled and legacy_data:
                legacy_rider_delta, legacy_rider_z = compute_delta_energy_series(
                    legacy_data["rider"],
                    payload["initial_states"]["rider"],
                    payload["rest_energy_mev"]["rider"],
                )
                legacy_rider_z_rel = legacy_rider_z - legacy_rider_z[0]
                if self.overlay_display_widget.value or self.overlay_save_widget.value:
                    overlay_axis_count = 2 if supports_driver else 1
                    overlay_fig, overlay_axes = plt.subplots(
                        1,
                        overlay_axis_count,
                        figsize=(fig_width, 6),
                        constrained_layout=True,
                        dpi=plot_dpi_value,
                    )
                    if overlay_axis_count == 1:
                        overlay_axes = [overlay_axes]
                    overlay_fig.patch.set_facecolor("white")
                    overlay_axes[0].plot(
                        rider_z_rel,
                        rider_delta,
                        color=COLOR_RIDER,
                        label="Core rider",
                        linewidth=2.0,
                    )
                    overlay_axes[0].plot(
                        legacy_rider_z_rel,
                        legacy_rider_delta,
                        color=COLOR_LEGACY_RIDER,
                        label="Legacy rider",
                        linewidth=2.0,
                        linestyle="--",
                    )
                    overlay_axes[0].set_title("Rider ΔE comparison")
                    overlay_axes[0].set_xlabel("Δz (mm)")
                    overlay_axes[0].set_ylabel("ΔE (GeV)")
                    overlay_axes[0].grid(True, alpha=0.3)
                    overlay_axes[0].legend()
                    if supports_driver and legacy_data.get("driver") is not None:
                        legacy_driver_delta, legacy_driver_z = compute_delta_energy_series(
                            legacy_data["driver"],
                            payload["initial_states"]["driver"],
                            payload["rest_energy_mev"]["driver"],
                        )
                        legacy_driver_z_rel = legacy_driver_z - legacy_driver_z[0]
                        overlay_axes[1].plot(
                            driver_z_rel,
                            driver_delta,
                            color=COLOR_DRIVER,
                            label="Core driver",
                            linewidth=2.0,
                        )
                        overlay_axes[1].plot(
                            legacy_driver_z_rel,
                            legacy_driver_delta,
                            color=COLOR_LEGACY_DRIVER,
                            label="Legacy driver",
                            linewidth=2.0,
                            linestyle="--",
                        )
                        overlay_axes[1].set_title("Driver ΔE comparison")
                        overlay_axes[1].set_xlabel("Δz (mm)")
                        overlay_axes[1].set_ylabel("ΔE (GeV)")
                        overlay_axes[1].grid(True, alpha=0.3)
                        overlay_axes[1].legend()
                    for axis in overlay_axes:
                        axis.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
                    if save_overlay_path is not None:
                        overlay_fig.savefig(
                            save_overlay_path, dpi=plot_dpi_value, bbox_inches="tight"
                        )
                        log_messages.append(
                            f"Legacy overlay plot saved to {save_overlay_path}"
                        )
                    overlay_buffer = io.BytesIO()
                    overlay_fig.savefig(
                        overlay_buffer, format="png", dpi=plot_dpi_value, bbox_inches="tight"
                    )
                    overlay_buffer.seek(0)
                    overlay_widget = widgets.Image(
                        value=overlay_buffer.getvalue(),
                        format="png",
                        layout=widgets.Layout(max_width="100%"),
                    )
                    figure_widgets.append(overlay_widget)
                    plt.close(overlay_fig)
                if (
                    self.difference_display_widget.value
                    or self.difference_save_widget.value
                ) and legacy_data.get("rider") is not None:
                    diff_axis_count = (
                        2
                        if supports_driver and legacy_data.get("driver") is not None
                        else 1
                    )
                    diff_fig, diff_axes = plt.subplots(
                        1,
                        diff_axis_count,
                        figsize=(fig_width, 6),
                        constrained_layout=True,
                        dpi=plot_dpi_value,
                    )
                    if diff_axis_count == 1:
                        diff_axes = [diff_axes]
                    step_axis = np.arange(len(rider_delta))
                    rider_diff = np.asarray(rider_delta) - np.asarray(legacy_rider_delta)
                    diff_axes[0].plot(
                        step_axis,
                        rider_diff,
                        color=COLOR_DIFF_RIDER,
                        label="Rider Δ(core−legacy) ΔE",
                        linewidth=2.0,
                    )
                    diff_axes[0].axhline(0.0, color="#333333", linewidth=1.0, alpha=0.4)
                    diff_axes[0].set_title("Rider ΔE difference")
                    diff_axes[0].set_xlabel("Step")
                    diff_axes[0].set_ylabel("ΔE difference (GeV)")
                    diff_axes[0].grid(True, alpha=0.3)
                    diff_axes[0].legend()
                    if diff_axis_count == 2 and supports_driver:
                        legacy_driver_delta, _legacy_driver_z = compute_delta_energy_series(
                            legacy_data["driver"],
                            payload["initial_states"]["driver"],
                            payload["rest_energy_mev"]["driver"],
                        )
                        driver_diff = np.asarray(driver_delta) - np.asarray(legacy_driver_delta)
                        diff_axes[1].plot(
                            step_axis,
                            driver_diff,
                            color=COLOR_DIFF_DRIVER,
                            label="Driver Δ(core−legacy) ΔE",
                            linewidth=2.0,
                        )
                        diff_axes[1].axhline(0.0, color="#333333", linewidth=1.0, alpha=0.4)
                        diff_axes[1].set_title("Driver ΔE difference")
                        diff_axes[1].set_xlabel("Step")
                        diff_axes[1].set_ylabel("ΔE difference (GeV)")
                        diff_axes[1].grid(True, alpha=0.3)
                        diff_axes[1].legend()
                    for axis in diff_axes:
                        axis.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
                    if save_difference_path is not None:
                        diff_fig.savefig(
                            save_difference_path, dpi=plot_dpi_value, bbox_inches="tight"
                        )
                        log_messages.append(
                            f"Δ(core−legacy) plot saved to {save_difference_path}"
                        )
                    diff_buffer = io.BytesIO()
                    diff_fig.savefig(
                        diff_buffer, format="png", dpi=plot_dpi_value, bbox_inches="tight"
                    )
                    diff_buffer.seek(0)
                    diff_widget = widgets.Image(
                        value=diff_buffer.getvalue(),
                        format="png",
                        layout=widgets.Layout(max_width="100%"),
                    )
                    figure_widgets.append(diff_widget)
                    plt.close(diff_fig)

            if self.trajectory_save_checkbox.value:
                trajectory_path = self._save_trajectories(
                    payload,
                    save_dir,
                    mode=self.trajectory_mode_dropdown.value,
                    interval=self.trajectory_interval_dropdown.value,
                )
                if trajectory_path is not None:
                    log_messages.append(f"Trajectories saved to {trajectory_path}")

            info_sections: List[widgets.Widget] = []
            section_titles: List[str] = []
            if metrics:
                summary_text = summarise_metrics(metrics)
                if not supports_driver:
                    summary_text = self._strip_driver_summary(summary_text)
                info_sections.append(
                    widgets.HTML(value=f"<pre style='margin:0'>{html.escape(summary_text)}</pre>")
                )
                section_titles.append("Metrics summary")
            else:
                info_sections.append(
                    widgets.HTML(
                        value="<p style='margin:0'>Legacy comparison skipped; no metrics computed.</p>"
                    )
                )
                section_titles.append("Metrics summary")

            config_payload = {
                "simulation_type": sim_type_value.name,
                "required_core_params": {k: core_params[k] for k in required_core},
                "optional_core_params": {
                    k: core_params[k] for k in core_params if k not in required_core
                },
                "plot_dpi": plot_dpi_value,
                "steps": self.steps_widget.value,
                "seed": self.seed_widget.value,
                "legacy_enabled": legacy_enabled,
            }
            info_sections.append(
                widgets.HTML(
                    value=(
                        "<pre style='margin:0'>"
                        f"{html.escape(json.dumps(config_payload, indent=2, sort_keys=True))}"
                        "</pre>"
                    )
                )
            )
            section_titles.append("Run configuration")

            if log_messages:
                messages_html = "<br>".join(html.escape(message) for message in log_messages)
                info_sections.append(
                    widgets.HTML(value=f"<p style='margin:0'>{messages_html}</p>")
                )
                section_titles.append("Run messages")

            info_accordion = widgets.Accordion(children=info_sections)
            with info_accordion.hold_trait_notifications():
                for index, title in enumerate(section_titles):
                    info_accordion.set_title(index, title)
                info_accordion.selected_index = 0 if info_sections else None
            display(widgets.VBox(figure_widgets + [info_accordion]))

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _collect_configuration_snapshot(self) -> dict:
        sim_type_value = self.simulation_widget.value
        supports_driver = self._supports_driver(sim_type_value)
        rider_params = self._collect_particle_values(self.rider_controls)
        driver_params = (
            self._collect_particle_values(self.driver_controls)
            if supports_driver
            else None
        )
        core_params = self._collect_core_values(self.core_controls)
        snapshot = {
            "steps": self.steps_widget.value,
            "seed": self.seed_widget.value,
            "simulation_type": sim_type_value.name,
            "legacy_enabled": self.legacy_toggle.value,
            "overlay_display": self.overlay_display_widget.value,
            "overlay_save": self.overlay_save_widget.value,
            "difference_display": self.difference_display_widget.value,
            "difference_save": self.difference_save_widget.value,
            "metrics_save": self.metrics_save_widget.value,
            "energy_save": self.energy_save_widget.value,
            "plot_dpi": self.dpi_widget.value,
            "output_dir": self.output_dir_widget.value,
            "config_dir": self.config_dir_widget.value,
            "config_name": self.config_name_widget.value,
            "save_trajectories": self.trajectory_save_checkbox.value,
            "trajectory_mode": self.trajectory_mode_dropdown.value,
            "trajectory_interval": self.trajectory_interval_dropdown.value,
            "rider_params": rider_params,
            "driver_params": driver_params,
            "core_params": core_params,
        }
        return snapshot

    def _apply_configuration_snapshot(self, snapshot: dict) -> None:
        with self.config_feedback_output:
            self.config_feedback_output.clear_output()
        try:
            if "simulation_type" in snapshot:
                sim_name = snapshot["simulation_type"]
                if isinstance(sim_name, str) and hasattr(SimulationType, sim_name):
                    self.simulation_widget.value = getattr(SimulationType, sim_name)
            if "steps" in snapshot:
                self.steps_widget.value = int(snapshot["steps"])
            if "seed" in snapshot:
                self.seed_widget.value = int(snapshot["seed"])
            if "legacy_enabled" in snapshot:
                self.legacy_toggle.value = bool(snapshot["legacy_enabled"])
            for widget, key in (
                (self.overlay_display_widget, "overlay_display"),
                (self.overlay_save_widget, "overlay_save"),
                (self.difference_display_widget, "difference_display"),
                (self.difference_save_widget, "difference_save"),
                (self.metrics_save_widget, "metrics_save"),
                (self.energy_save_widget, "energy_save"),
            ):
                if key in snapshot:
                    widget.value = bool(snapshot[key])
            if "plot_dpi" in snapshot:
                self.dpi_widget.value = int(snapshot["plot_dpi"])
            if "output_dir" in snapshot:
                self.output_dir_widget.value = str(snapshot["output_dir"])
            if "config_dir" in snapshot:
                self.config_dir_widget.value = str(snapshot["config_dir"])
            if "config_name" in snapshot:
                self.config_name_widget.value = str(snapshot["config_name"])
            if "save_trajectories" in snapshot:
                self.trajectory_save_checkbox.value = bool(snapshot["save_trajectories"])
            if "trajectory_mode" in snapshot and snapshot["trajectory_mode"] in {
                option for _, option in TRAJECTORY_MODE_OPTIONS
            }:
                self.trajectory_mode_dropdown.value = snapshot["trajectory_mode"]
            if "trajectory_interval" in snapshot:
                try:
                    interval = int(snapshot["trajectory_interval"])
                except (TypeError, ValueError):
                    interval = 1
                available = {value for _, value in TRAJECTORY_INTERVAL_OPTIONS}
                if interval in available:
                    self.trajectory_interval_dropdown.value = interval
            rider_params = snapshot.get("rider_params") or {}
            for name, control in self.rider_controls.items():
                if name in rider_params:
                    control.value = rider_params[name]
            driver_params = snapshot.get("driver_params") or {}
            for name, control in self.driver_controls.items():
                if name in driver_params:
                    control.value = driver_params[name]
            core_params = snapshot.get("core_params") or {}
            for name, control in self.core_controls.items():
                if name in core_params:
                    control.value = core_params[name]
            self._refresh_initial_properties()
            with self.config_feedback_output:
                print("Configuration loaded.")
        except Exception as exc:  # pragma: no cover - best effort feedback
            with self.config_feedback_output:
                print(f"Failed to load configuration: {exc}")

    def _handle_save_config(self, _button) -> None:
        snapshot = self._collect_configuration_snapshot()
        try:
            target_dir = self._ensure_config_dir()
            file_name = self._maybe_append_json(
                snapshot.get("config_name", "testbed_config.json")
            )
            target_path = target_dir / file_name
            payload = dict(snapshot)
            payload.pop("config_dir", None)
            payload.pop("config_name", None)
            with target_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            with self.config_feedback_output:
                self.config_feedback_output.clear_output()
                print(f"Configuration saved to {target_path}")
            self.config_name_widget.value = file_name
            self._refresh_config_file_options(selected=file_name)
        except Exception as exc:  # pragma: no cover - best effort feedback
            with self.config_feedback_output:
                self.config_feedback_output.clear_output()
                print(f"Failed to save configuration: {exc}")

    def _handle_load_config(self, _button) -> None:
        try:
            target_dir = self._resolved_config_dir()
            file_name = self._maybe_append_json(self.config_name_widget.value)
            target_path = target_dir / file_name
            if not target_path.exists():
                raise FileNotFoundError(f"No config found at {target_path}")
            with target_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            rebuilt_snapshot = dict(payload)
            rebuilt_snapshot["config_dir"] = str(self.config_dir_widget.value)
            rebuilt_snapshot["config_name"] = file_name
            self._apply_configuration_snapshot(rebuilt_snapshot)
            self._refresh_config_file_options(selected=file_name)
        except Exception as exc:  # pragma: no cover - best effort feedback
            with self.config_feedback_output:
                self.config_feedback_output.clear_output()
                print(f"Failed to load configuration: {exc}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _make_particle_widgets(self, defaults: Dict[str, float | int]) -> Dict[str, widgets.Widget]:
        controls: Dict[str, widgets.Widget] = {}
        for name in PARTICLE_PARAM_FIELDS:
            default_value = defaults[name]
            description = PARAM_LABELS.get(name, name.replace("_", " ").title())
            if isinstance(default_value, int):
                control = widgets.IntText(
                    value=int(default_value),
                    description=description,
                    layout=PARTICLE_WIDGET_WIDTH,
                    style=PARTICLE_WIDGET_STYLE,
                )
            else:
                control = widgets.FloatText(
                    value=float(default_value),
                    description=description,
                    layout=PARTICLE_WIDGET_WIDTH,
                    style=PARTICLE_WIDGET_STYLE,
                )
            controls[name] = control
        return controls

    def _make_core_widgets(self) -> Dict[str, widgets.Widget]:
        controls: Dict[str, widgets.Widget] = {}
        for name, default_value in CORE_PARAM_DEFAULTS.items():
            description = CORE_PARAM_LABELS.get(name, name.replace("_", " ").title())
            controls[name] = widgets.FloatText(
                value=float(default_value),
                description=description,
                layout=CORE_WIDGET_WIDTH,
                style=CORE_WIDGET_STYLE,
            )
        return controls

    def _apply_species_preset(
        self, controls: Dict[str, widgets.Widget], preset_key: str
    ) -> None:
        preset = SPECIES_PRESETS.get(preset_key)
        if not preset:
            return
        for field, value in preset.items():
            control = controls.get(field)
            if control is None:
                continue
            current_value = control.value
            if isinstance(current_value, int):
                control.value = int(round(value))
            else:
                control.value = float(value)

    def _build_particle_section(
        self, title: str, controls: Dict[str, widgets.Widget], preset_widget: widgets.Widget
    ) -> widgets.Accordion:
        rows: List[widgets.Widget] = [preset_widget]
        rows.extend(self._particle_rows(controls))
        accordion = widgets.Accordion(children=[widgets.VBox(rows)])
        accordion.set_title(0, title)
        return accordion

    def _build_core_section(self, controls: Dict[str, widgets.Widget]) -> widgets.Accordion:
        rows = self._core_rows(controls)
        accordion = widgets.Accordion(children=[widgets.VBox(rows)])
        accordion.set_title(0, "Core configuration")
        return accordion

    def _particle_rows(self, controls: Dict[str, widgets.Widget]) -> List[widgets.Widget]:
        rows: List[widgets.Widget] = []
        row: List[widgets.Widget] = []
        for name in PARTICLE_PARAM_FIELDS:
            row.append(controls[name])
            if len(row) == 2:
                rows.append(widgets.HBox(row, layout=ROW_LAYOUT))
                row = []
        if row:
            rows.append(widgets.HBox(row, layout=ROW_LAYOUT))
        return rows

    def _core_rows(self, controls: Dict[str, widgets.Widget]) -> List[widgets.Widget]:
        rows: List[widgets.Widget] = []
        row: List[widgets.Widget] = []
        for name in CORE_PARAM_DEFAULTS:
            row.append(controls[name])
            if len(row) == 3:
                rows.append(widgets.HBox(row, layout=ROW_LAYOUT))
                row = []
        if row:
            rows.append(widgets.HBox(row, layout=ROW_LAYOUT))
        return rows

    def _collect_particle_values(
        self, controls: Dict[str, widgets.Widget]
    ) -> Dict[str, float | int]:
        values: Dict[str, float | int] = {}
        for name, control in controls.items():
            values[name] = control.value
        return values

    def _collect_core_values(self, controls: Dict[str, widgets.Widget]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for name, control in controls.items():
            values[name] = float(control.value)
        return values

    def _strip_driver_summary(self, summary: str) -> str:
        lines: List[str] = []
        skipping = False
        for line in summary.splitlines():
            if line.startswith("- Driver"):
                skipping = True
                continue
            if skipping:
                if line.startswith("  driver"):
                    continue
                skipping = False
            lines.append(line)
        return "\n".join(lines)

    def _supports_driver(self, sim_type: SimulationType) -> bool:
        return sim_type == SimulationType.BUNCH_TO_BUNCH

    def _apply_driver_visibility(self, sim_value: SimulationType) -> None:
        supports_driver = self._supports_driver(sim_value)
        self.driver_section.layout.display = "" if supports_driver else "none"
        if not supports_driver:
            self.driver_section.selected_index = None

    def _refresh_initial_properties(self, _change=None) -> None:
        sim_type_value = self.simulation_widget.value
        supports_driver = self._supports_driver(sim_type_value)
        rider_params = self._collect_particle_values(self.rider_controls)
        driver_params = (
            self._collect_particle_values(self.driver_controls)
            if supports_driver
            else None
        )
        try:
            rider_state, driver_state, rider_rest_mev, driver_rest_mev = prepare_two_particle_demo(
                seed=self.seed_widget.value,
                rider_params=rider_params,
                driver_params=driver_params,
            )
        except Exception as exc:  # pragma: no cover - best effort feedback
            with self.initial_state_output:
                self.initial_state_output.clear_output()
                print(f"Failed to compute initial states: {exc}")
            return

        rider_gamma = float(rider_state["gamma"][0])
        rider_rest_gev = rider_rest_mev * 1e-3
        rider_total_gev = rider_gamma * rider_rest_gev

        rows = [
            ("Rider γ", f"{rider_gamma:.6f}"),
            (
                "Rider rest energy",
                f"{rider_rest_gev:.6f} GeV ({rider_rest_mev:.2f} MeV)",
            ),
            ("Rider total energy", f"{rider_total_gev:.6f} GeV"),
        ]

        if supports_driver and driver_params is not None:
            driver_gamma = float(driver_state["gamma"][0])
            driver_rest_gev = driver_rest_mev * 1e-3
            driver_total_gev = driver_gamma * driver_rest_gev
            rows.extend(
                [
                    ("Driver γ", f"{driver_gamma:.6f}"),
                    (
                        "Driver rest energy",
                        f"{driver_rest_gev:.6f} GeV ({driver_rest_mev:.2f} MeV)",
                    ),
                    ("Driver total energy", f"{driver_total_gev:.6f} GeV"),
                ]
            )
            driver_note = ""
        else:
            driver_note = "<p style='margin:4px 0 0'><em>Driver not active for this simulation type.</em></p>"

        table_rows = "".join(
            f"<tr><th style='text-align:left;padding-right:12px'>{label}</th><td>{value}</td></tr>"
            for label, value in rows
        )
        table_html = (
            "<p style='margin:0 0 6px'><strong>Seed:</strong> "
            f"{self.seed_widget.value}</p><table style='border-collapse:collapse'>"
            f"{table_rows}</table>{driver_note}"
        )
        with self.initial_state_output:
            self.initial_state_output.clear_output()
            display(widgets.HTML(value=table_html))

    def _update_core_controls(self, change=None) -> None:
        sim_value = change["new"] if isinstance(change, dict) else self.simulation_widget.value
        required = self._required_params_for(sim_value)
        self._apply_core_param_state(self.core_controls, required)
        self._apply_driver_visibility(sim_value)
        self._refresh_initial_properties()

    def _required_params_for(self, sim_type: SimulationType) -> set[str]:
        return CORE_REQUIRED_PARAMS.get(sim_type, set())

    def _apply_core_param_state(
        self, controls: Dict[str, widgets.Widget], required_params: set[str]
    ) -> None:
        for name, control in controls.items():
            control.disabled = name not in required_params
            control.layout.opacity = 1.0 if name in required_params else 0.45

    def _refresh_config_file_options(self, selected: Optional[str] = None) -> None:
        directory = self._resolved_config_dir()
        files = self._list_config_files(directory)
        if files:
            options = [(file_name, file_name) for file_name in files]
            self.config_file_dropdown.options = options
            self.config_file_dropdown.disabled = False
            if selected and selected in files:
                self.config_file_dropdown.value = selected
            elif self.config_file_dropdown.value in files:
                self.config_file_dropdown.value = self.config_file_dropdown.value
            else:
                self.config_file_dropdown.value = options[0][1]
        else:
            self.config_file_dropdown.options = [("No saved configs", "")]
            self.config_file_dropdown.value = ""
            self.config_file_dropdown.disabled = True

    def _save_trajectories(
        self,
        payload: dict,
        directory: Path,
        *,
        mode: str,
        interval: int,
    ) -> Optional[Path]:
        if interval <= 0:
            interval = 1
        core_data = payload.get("core")
        if core_data is None:
            return None
        trajectory_package: Dict[str, dict] = {
            "mode": mode,
            "step_interval": interval,
            "fields": list(FIELDS_TO_TRACK),
            "initial_states": {
                key: self._serialize_state(payload["initial_states"][key])
                for key in ("rider", "driver")
                if payload["initial_states"].get(key) is not None
            },
            "rest_energy_mev": payload.get("rest_energy_mev", {}),
            "core": {},
        }
        for species in ("rider", "driver"):
            states = core_data.get(species)
            if not states:
                continue
            trajectory_package["core"][species] = self._serialize_trajectories(
                states, mode=mode, interval=interval
            )
        legacy_data = payload.get("legacy")
        if legacy_data:
            trajectory_package["legacy"] = {}
            for species in ("rider", "driver"):
                states = legacy_data.get(species)
                if not states:
                    continue
                trajectory_package["legacy"][species] = self._serialize_trajectories(
                    states, mode=mode, interval=interval
                )
        directory.mkdir(parents=True, exist_ok=True)
        file_name = f"trajectories_{mode}_every{interval}.json"
        destination = directory / file_name
        with destination.open("w", encoding="utf-8") as fh:
            json.dump(trajectory_package, fh, indent=2)
        return destination

    def _serialize_state(self, state: dict) -> Dict[str, List[float]]:
        serialized: Dict[str, List[float]] = {}
        for field, value in state.items():
            if isinstance(value, (int, float)):
                serialized[field] = [float(value)]
            else:
                serialized[field] = np.asarray(value, dtype=float).tolist()
        return serialized

    def _serialize_trajectories(
        self, states: List[dict], *, mode: str, interval: int
    ) -> List[dict]:
        data: List[dict] = []
        for step_index, state in enumerate(states):
            if step_index % interval != 0:
                continue
            entry = {"step": step_index}
            for field in FIELDS_TO_TRACK:
                if field not in state:
                    continue
                array = np.asarray(state[field], dtype=float)
                if mode == "average":
                    entry[field] = float(np.mean(array))
                else:
                    entry[field] = array.tolist()
            data.append(entry)
        return data

    def _resolved_output_dir(self) -> Path:
        raw_value = self.output_dir_widget.value.strip() or "test_outputs/testbed_runs"
        return Path(raw_value).expanduser()

    def _save_dir_if_needed(self, should_create: bool) -> Path:
        directory = self._resolved_output_dir()
        if should_create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _resolved_config_dir(self) -> Path:
        raw_value = self.config_dir_widget.value.strip() or "configs/testbed_runs"
        return Path(raw_value).expanduser()

    def _ensure_config_dir(self) -> Path:
        directory = self._resolved_config_dir()
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _list_config_files(self, directory: Path) -> List[str]:
        if not directory.exists():
            return []
        return sorted(
            str(path.name) for path in directory.glob("*.json") if path.is_file()
        )

    def _maybe_append_json(self, name: str) -> str:
        candidate = name.strip() or "testbed_config.json"
        return candidate if candidate.lower().endswith(".json") else f"{candidate}.json"

    def _detect_project_root(self) -> Path:
        current = Path.cwd().resolve()
        for candidate in [current, *current.parents]:
            if (candidate / "pyproject.toml").exists() and candidate.name == "LW_windows":
                return candidate
        return current