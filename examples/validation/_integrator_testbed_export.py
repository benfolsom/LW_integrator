# In[ ]:
from __future__ import annotations

import io
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Ensure repository paths are importable when running in VS Code/notebooks
PROJECT_ROOT = Path.cwd().resolve()
while PROJECT_ROOT.name and PROJECT_ROOT.name != "LW_windows":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LEGACY_ROOT = PROJECT_ROOT / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

VALIDATION_ROOT = PROJECT_ROOT / "examples" / "validation"
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from core_vs_legacy_benchmark import (  # noqa: E402
    DEFAULT_DRIVER_PARAMS,
    DEFAULT_RIDER_PARAMS,
    PARTICLE_PARAM_FIELDS,
    SimulationType,
    compute_delta_energy_series,
    prepare_two_particle_demo,
    run_benchmark,
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

LAST_METRICS: Optional[Dict[str, Dict[str, float]]] | None = None
LAST_PAYLOAD: Optional[dict] = None
LAST_LOG_MESSAGES: List[str] = []

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

# In[ ]:
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
    "steps": "Steps",
    "seed": "Seed",
    "time_step": "Time step (ns)",
    "wall_z": "Wall z (mm)",
    "aperture_radius": "Aperture radius (mm)",
    "mean": "Mean separation (mm)",
    "cav_spacing": "Cavity spacing (mm)",
    "z_cutoff": "Force cutoff z (mm)",
}

PARTICLE_WIDGET_WIDTH = widgets.Layout(width="320px")
CORE_WIDGET_WIDTH = widgets.Layout(width="280px")
PARTICLE_WIDGET_STYLE = {"description_width": "175px"}
CORE_WIDGET_STYLE = {"description_width": "175px"}
ROW_LAYOUT = widgets.Layout(gap="14px")

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
    "steps": 1000,
    "seed": 12345,
    "time_step": 2.2e-7,
    "wall_z": 1e5,
    "aperture_radius": 1e5,
    "mean": 1e5,
    "cav_spacing": 1e5,
    "z_cutoff": 0.0,
}

CORE_REQUIRED_PARAMS = {
    SimulationType.CONDUCTING_WALL: {
        "steps",
        "seed",
        "time_step",
        "wall_z",
        "aperture_radius",
    },
    SimulationType.SWITCHING_WALL: {
        "steps",
        "seed",
        "time_step",
        "wall_z",
        "aperture_radius",
        "cav_spacing",
    },
    SimulationType.BUNCH_TO_BUNCH: {"steps", "seed", "time_step", "aperture_radius"},
}


def _make_particle_widgets(
    defaults: Dict[str, float | int],
) -> Dict[str, widgets.Widget]:
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


def _make_core_widgets() -> Dict[str, widgets.Widget]:
    controls: Dict[str, widgets.Widget] = {}
    for name, default_value in CORE_PARAM_DEFAULTS.items():
        description = CORE_PARAM_LABELS.get(name, name.replace("_", " ").title())
        # Use IntText for steps and seed, FloatText for others
        if name in ("steps", "seed"):
            controls[name] = widgets.IntText(
                value=int(default_value),
                description=description,
                layout=CORE_WIDGET_WIDTH,
                style=CORE_WIDGET_STYLE,
            )
        else:
            controls[name] = widgets.FloatText(
                value=float(default_value),
                description=description,
                layout=CORE_WIDGET_WIDTH,
                style=CORE_WIDGET_STYLE,
            )
    return controls


def _apply_species_preset(controls: Dict[str, widgets.Widget], preset_key: str) -> None:
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


def _particle_rows(controls: Dict[str, widgets.Widget]) -> List[widgets.Widget]:
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


def _core_rows(
    controls: Dict[str, widgets.Widget],
    z_cutoff_checkbox: widgets.Checkbox,
    random_seed_checkbox: widgets.Checkbox,
) -> List[widgets.Widget]:
    rows: List[widgets.Widget] = []
    row: List[widgets.Widget] = []
    for name in CORE_PARAM_DEFAULTS:
        # Add z_cutoff checkbox right after z_cutoff field
        if name == "z_cutoff":
            row.append(controls[name])
            row.append(z_cutoff_checkbox)
            if len(row) >= 3:
                rows.append(widgets.HBox(row, layout=ROW_LAYOUT))
                row = []
        # Add random_seed checkbox right after seed field
        elif name == "seed":
            row.append(controls[name])
            row.append(random_seed_checkbox)
            if len(row) >= 3:
                rows.append(widgets.HBox(row, layout=ROW_LAYOUT))
                row = []
        else:
            row.append(controls[name])
            if len(row) == 3:
                rows.append(widgets.HBox(row, layout=ROW_LAYOUT))
                row = []
    if row:
        rows.append(widgets.HBox(row, layout=ROW_LAYOUT))
    return rows


def _collect_particle_values(
    controls: Dict[str, widgets.Widget],
) -> Dict[str, float | int]:
    values: Dict[str, float | int] = {}
    for name, control in controls.items():
        values[name] = control.value
    return values


def _collect_core_values(controls: Dict[str, widgets.Widget]) -> Dict[str, float | int]:
    values: Dict[str, float | int] = {}
    for name, control in controls.items():
        # Keep steps and seed as int, others as float
        if name in ("steps", "seed"):
            values[name] = int(control.value)
        else:
            values[name] = float(control.value)
    return values


def _build_particle_section(
    title: str,
    controls: Dict[str, widgets.Widget],
    preset_widget: widgets.Widget | None,
) -> widgets.Accordion:
    rows = []
    if preset_widget is not None:
        rows.append(preset_widget)
    rows.extend(_particle_rows(controls))
    accordion = widgets.Accordion(children=[widgets.VBox(rows)])
    accordion.set_title(0, title)
    return accordion


def _build_core_section(
    controls: Dict[str, widgets.Widget],
    z_cutoff_checkbox: widgets.Checkbox,
    random_seed_checkbox: widgets.Checkbox,
) -> widgets.Accordion:
    rows = _core_rows(controls, z_cutoff_checkbox, random_seed_checkbox)
    accordion = widgets.Accordion(children=[widgets.VBox(rows)])
    accordion.set_title(0, "Core configuration")
    return accordion


def _strip_driver_summary(summary: str) -> str:
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


def _required_params_for(sim_type: SimulationType) -> set[str]:
    return CORE_REQUIRED_PARAMS.get(sim_type, set())


def _apply_core_param_state(
    controls: Dict[str, widgets.Widget],
    required_params: set[str],
    z_cutoff_checkbox: widgets.Checkbox,
    random_seed_checkbox: widgets.Checkbox,
    sim_type: SimulationType,
) -> None:
    for name, control in controls.items():
        control.disabled = name not in required_params
        control.layout.opacity = 1.0 if name in required_params else 0.45

    # z_cutoff checkbox is available for all modes
    z_cutoff_checkbox.disabled = False
    z_cutoff_checkbox.layout.opacity = 1.0

    # Disable z_cutoff field if checkbox is not checked
    if "z_cutoff" in controls:
        controls["z_cutoff"].disabled = not z_cutoff_checkbox.value
        controls["z_cutoff"].layout.opacity = 1.0 if z_cutoff_checkbox.value else 0.45

    # random_seed checkbox is available for all modes
    random_seed_checkbox.disabled = False
    random_seed_checkbox.layout.opacity = 1.0

    # Disable seed field if random_seed checkbox is checked
    if "seed" in controls:
        controls["seed"].disabled = random_seed_checkbox.value
        controls["seed"].layout.opacity = 0.45 if random_seed_checkbox.value else 1.0


def _sync_config_name_after_load(_button) -> None:
    """Ensure the config name widget mirrors the loaded filename."""
    selected_file = config_file_dropdown.value
    if selected_file:
        config_name_widget.value = str(selected_file)


# In[ ]:
legacy_toggle = widgets.Checkbox(value=False, description="Include legacy comparison")
z_cutoff_checkbox = widgets.Checkbox(
    value=False,
    description="Enable z cutoff",
    tooltip="Stop applying external forces when particle z > z_cutoff (all modes)",
)
random_seed_checkbox = widgets.Checkbox(
    value=False,
    description="Random seed",
    tooltip="Generate a new random seed for each run",
)
simulation_widget = widgets.Dropdown(
    options=[(label, value) for label, value in SIMULATION_TYPE_OPTIONS.items()],
    value=SimulationType.BUNCH_TO_BUNCH,
    description="Core mode:",
)
overlay_display_widget = widgets.Checkbox(
    value=False, description="Show legacy overlay plot"
)
overlay_save_widget = widgets.Checkbox(
    value=False, description="Save legacy overlay plot"
)
difference_display_widget = widgets.Checkbox(
    value=False, description="Show Δ(core−legacy) plot"
)
difference_save_widget = widgets.Checkbox(
    value=False, description="Save Δ(core−legacy) plot"
)
metrics_save_widget = widgets.Checkbox(value=False, description="Save metrics JSON")
energy_save_widget = widgets.Checkbox(value=True, description="Save ΔE plots")
energy_display_widget = widgets.Checkbox(value=True, description="Show ΔE plots")

# Energy plot configuration widgets
energy_xaxis_widget = widgets.Dropdown(
    options=[("z position", "z"), ("time", "t"), ("both (dual axis)", "dual")],
    value="z",
    description="Energy plot x-axis:",
    layout=widgets.Layout(width="320px"),
    style={"description_width": "120px"},
)
energy_yaxis_widget = widgets.Dropdown(
    options=[
        ("ΔE total", "delta_total"),
        ("ΔE_z (longitudinal)", "delta_z"),
        ("ΔE_x (transverse)", "delta_x"),
        ("ΔE_y (transverse)", "delta_y"),
        ("E total (absolute)", "total"),
    ],
    value="delta_total",
    description="Energy plot y-axis:",
    layout=widgets.Layout(width="320px"),
    style={"description_width": "120px"},
)

transverse_display_widget = widgets.Checkbox(
    value=False, description="Show ⟨x⟩, ⟨y⟩ plots"
)
transverse_save_widget = widgets.Checkbox(
    value=False, description="Save ⟨x⟩, ⟨y⟩ plots"
)
trajectory_save_widget = widgets.Checkbox(value=False, description="Save trajectories")
trajectory_interval_widget = widgets.IntText(
    value=10,
    description="Traj. interval:",
    layout=widgets.Layout(width="220px"),
    style=dict(CORE_WIDGET_STYLE, description_width="120px"),
)
dpi_widget = widgets.Dropdown(
    options=[(f"{value} dpi", value) for value in AVAILABLE_DPI_CHOICES],
    value=DEFAULT_PLOT_DPI,
    description="Plot DPI:",
    layout=widgets.Layout(width="260px"),
    style=dict(CORE_WIDGET_STYLE, description_width="120px"),
)
output_dir_widget = widgets.Text(
    value="test_outputs/testbed_runs",
    description="Plot dir:",
    layout=widgets.Layout(width="360px"),
    style=CORE_WIDGET_STYLE,
)
config_dir_widget = widgets.Text(
    value="configs/testbed_runs",
    description="Config dir:",
    layout=widgets.Layout(width="360px"),
    style=CORE_WIDGET_STYLE,
)
config_name_widget = widgets.Text(
    value="testbed_config.json",
    description="Config name:",
    layout=widgets.Layout(width="320px"),
    style=CORE_WIDGET_STYLE,
)
config_file_dropdown = widgets.Dropdown(
    options=[("No saved configs", "")],
    value="",
    description="Existing:",
    layout=widgets.Layout(width="320px"),
    style=dict(CORE_WIDGET_STYLE, description_width="120px"),
    disabled=True,
)
output_area = widgets.Output()
config_feedback_output = widgets.Output(
    layout=widgets.Layout(border="1px solid #dddddd", padding="8px", margin="4px 0")
)
initial_state_output = widgets.Output(
    layout=widgets.Layout(border="1px solid #dddddd", padding="8px", margin="6px 0")
)
rider_controls = _make_particle_widgets(DEFAULT_RIDER_PARAMS)
driver_controls = _make_particle_widgets(DEFAULT_DRIVER_PARAMS)
core_controls = _make_core_widgets()
rider_species_widget = widgets.Dropdown(
    options=SPECIES_DROPDOWN_OPTIONS,
    value="custom",
    description="Rider species:",
    layout=PARTICLE_WIDGET_WIDTH,
    style=PARTICLE_WIDGET_STYLE,
)
driver_species_widget = widgets.Dropdown(
    options=SPECIES_DROPDOWN_OPTIONS,
    value="custom",
    description="Driver species:",
    layout=PARTICLE_WIDGET_WIDTH,
    style=PARTICLE_WIDGET_STYLE,
)
rider_section = _build_particle_section(
    "Rider particle", rider_controls, rider_species_widget
)
driver_section = _build_particle_section(
    "Driver particle", driver_controls, driver_species_widget
)
core_section = _build_core_section(
    core_controls, z_cutoff_checkbox, random_seed_checkbox
)
image_subcharge_widget = widgets.IntSlider(
    value=12,
    min=4,
    max=128,
    step=1,
    description="Image subcharges:",
    continuous_update=False,
    layout=widgets.Layout(width="360px"),
    style=dict(CORE_WIDGET_STYLE, description_width="160px"),
)


def _resolved_output_dir() -> Path:
    raw_value = output_dir_widget.value.strip() or "test_outputs/testbed_runs"
    return Path(raw_value).expanduser()


def _resolved_config_dir() -> Path:
    raw_value = config_dir_widget.value.strip() or "configs/testbed_runs"
    return Path(raw_value).expanduser()


def _save_dir_if_needed(should_create: bool) -> Path:
    directory = _resolved_output_dir()
    if should_create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _ensure_config_dir() -> Path:
    directory = _resolved_config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _list_config_files(directory: Path) -> List[str]:
    if not directory.exists():
        return []
    return sorted(str(path.name) for path in directory.glob("*.json") if path.is_file())


def _maybe_append_json(name: str) -> str:
    candidate = name.strip() or "testbed_config.json"
    return candidate if candidate.lower().endswith(".json") else f"{candidate}.json"


def _refresh_config_file_options(selected: Optional[str] = None) -> None:
    directory = _resolved_config_dir()
    files = _list_config_files(directory)
    if files:
        options = [(file_name, file_name) for file_name in files]
        config_file_dropdown.options = options
        config_file_dropdown.disabled = False
        if selected and selected in files:
            config_file_dropdown.value = selected
        elif config_file_dropdown.value in files:
            config_file_dropdown.value = config_file_dropdown.value
        else:
            config_file_dropdown.value = options[0][1]
    else:
        config_file_dropdown.options = [("No saved configs", "")]
        config_file_dropdown.value = ""
        config_file_dropdown.disabled = True


def _update_legacy_controls(change) -> None:
    legacy_enabled = bool(
        change["new"] if isinstance(change, dict) else legacy_toggle.value
    )
    for checkbox in (
        overlay_display_widget,
        overlay_save_widget,
        difference_display_widget,
        difference_save_widget,
        metrics_save_widget,
    ):
        checkbox.disabled = not legacy_enabled
        if not legacy_enabled:
            checkbox.value = False


def _supports_driver(sim_type: SimulationType) -> bool:
    return sim_type == SimulationType.BUNCH_TO_BUNCH


def _apply_driver_visibility(sim_value: SimulationType) -> None:
    supports_driver = _supports_driver(sim_value)
    driver_section.layout.display = "" if supports_driver else "none"
    if not supports_driver:
        driver_section.selected_index = None


def _refresh_initial_properties(change=None) -> None:
    sim_type_value = simulation_widget.value
    supports_driver = _supports_driver(sim_type_value)
    rider_params = _collect_particle_values(rider_controls)
    driver_params = (
        _collect_particle_values(driver_controls) if supports_driver else None
    )
    try:
        rider_state, driver_state, rider_rest_mev, driver_rest_mev = (
            prepare_two_particle_demo(
                seed=int(core_controls.get("seed", widgets.IntText(value=12345)).value),
                rider_params=rider_params,
                driver_params=driver_params,
            )
        )
    except Exception as exc:
        with initial_state_output:
            initial_state_output.clear_output()
            print(f"Failed to compute initial states: {exc}")
        return

    rider_gamma = float(rider_state["gamma"][0])
    rider_rest_gev = rider_rest_mev * 1e-3
    rider_total_gev = rider_gamma * rider_rest_gev

    rows = [
        ("Rider γ", f"{rider_gamma:.6f}"),
        ("Rider rest energy", f"{rider_rest_gev:.6f} GeV ({rider_rest_mev:.2f} MeV)"),
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
        f"{int(core_controls.get('seed', widgets.IntText(value=12345)).value)}</p><table style='border-collapse:collapse'>"
        f"{table_rows}</table>{driver_note}"
    )
    with initial_state_output:
        initial_state_output.clear_output()
        display(widgets.HTML(value=table_html))


def _collect_configuration_snapshot() -> dict:
    sim_type_value = simulation_widget.value
    supports_driver = _supports_driver(sim_type_value)
    rider_params = _collect_particle_values(rider_controls)
    driver_params = (
        _collect_particle_values(driver_controls) if supports_driver else None
    )
    core_params = _collect_core_values(core_controls)
    snapshot = {
        "simulation_type": sim_type_value.name,
        "legacy_enabled": legacy_toggle.value,
        "overlay_display": overlay_display_widget.value,
        "overlay_save": overlay_save_widget.value,
        "difference_display": difference_display_widget.value,
        "difference_save": difference_save_widget.value,
        "metrics_save": metrics_save_widget.value,
        "energy_display": energy_display_widget.value,
        "energy_save": energy_save_widget.value,
        "energy_xaxis": energy_xaxis_widget.value,
        "energy_yaxis": energy_yaxis_widget.value,
        "transverse_display": transverse_display_widget.value,
        "transverse_save": transverse_save_widget.value,
        "trajectory_save": trajectory_save_widget.value,
        "trajectory_interval": trajectory_interval_widget.value,
        "plot_dpi": dpi_widget.value,
        "output_dir": output_dir_widget.value,
        "config_dir": config_dir_widget.value,
        "config_name": config_name_widget.value,
        "rider_params": rider_params,
        "driver_params": driver_params,
        "core_params": core_params,
        "image_subcharge_count": image_subcharge_widget.value,
        "z_cutoff_enabled": z_cutoff_checkbox.value,
        "random_seed_enabled": random_seed_checkbox.value,
    }
    return snapshot


def _apply_configuration_snapshot(snapshot: dict) -> None:
    with config_feedback_output:
        config_feedback_output.clear_output()
    try:
        if "simulation_type" in snapshot:
            sim_name = snapshot["simulation_type"]
            if isinstance(sim_name, str) and hasattr(SimulationType, sim_name):
                simulation_widget.value = getattr(SimulationType, sim_name)
        if "z_cutoff_enabled" in snapshot:
            z_cutoff_checkbox.value = bool(snapshot["z_cutoff_enabled"])
        if "random_seed_enabled" in snapshot:
            random_seed_checkbox.value = bool(snapshot["random_seed_enabled"])
        if "legacy_enabled" in snapshot:
            legacy_toggle.value = bool(snapshot["legacy_enabled"])
        for widget, key in [
            (overlay_display_widget, "overlay_display"),
            (overlay_save_widget, "overlay_save"),
            (difference_display_widget, "difference_display"),
            (difference_save_widget, "difference_save"),
            (metrics_save_widget, "metrics_save"),
            (energy_display_widget, "energy_display"),
            (energy_save_widget, "energy_save"),
            (energy_xaxis_widget, "energy_xaxis"),
            (energy_yaxis_widget, "energy_yaxis"),
            (transverse_display_widget, "transverse_display"),
            (transverse_save_widget, "transverse_save"),
            (trajectory_save_widget, "trajectory_save"),
        ]:
            if key in snapshot:
                widget.value = bool(snapshot[key])
        if "trajectory_interval" in snapshot:
            trajectory_interval_widget.value = int(snapshot["trajectory_interval"])
        if "plot_dpi" in snapshot:
            dpi_widget.value = int(snapshot["plot_dpi"])
        if "output_dir" in snapshot:
            output_dir_widget.value = str(snapshot["output_dir"])
        if "config_dir" in snapshot:
            config_dir_widget.value = str(snapshot["config_dir"])
        if "config_name" in snapshot:
            config_name_widget.value = str(snapshot["config_name"])
        if "rider_params" in snapshot:
            rider_params = snapshot["rider_params"]
            for name, value in rider_params.items():
                control = rider_controls.get(name)
                if control is not None:
                    control.value = value
        if "driver_params" in snapshot:
            driver_params = snapshot["driver_params"]
            if driver_params is not None:
                for name, value in driver_params.items():
                    control = driver_controls.get(name)
                    if control is not None:
                        control.value = value
        if "core_params" in snapshot:
            core_params = snapshot["core_params"]
            for name, value in core_params.items():
                control = core_controls.get(name)
                if control is not None:
                    control.value = value
        if "image_subcharge_count" in snapshot:
            try:
                value = int(snapshot["image_subcharge_count"])
            except (TypeError, ValueError):
                value = image_subcharge_widget.value
            value = max(
                image_subcharge_widget.min, min(image_subcharge_widget.max, value)
            )
            image_subcharge_widget.value = value
        with config_feedback_output:
            print("✓ Configuration loaded successfully")
    except Exception as exc:
        with config_feedback_output:
            print(f"Error loading configuration: {exc}")


def _handle_save_config(button) -> None:
    with config_feedback_output:
        config_feedback_output.clear_output()
    try:
        directory = _ensure_config_dir()
        filename = _maybe_append_json(config_name_widget.value)
        filepath = directory / filename
        snapshot = _collect_configuration_snapshot()
        with filepath.open("w") as handle:
            json.dump(snapshot, handle, indent=2, default=str)
        _refresh_config_file_options(selected=filename)
        with config_feedback_output:
            print(f"✓ Saved config to: {filepath}")
    except Exception as exc:
        with config_feedback_output:
            print(f"Error saving config: {exc}")


def _handle_load_config(button) -> None:
    with config_feedback_output:
        config_feedback_output.clear_output()
    try:
        directory = _resolved_config_dir()
        selected_file = config_file_dropdown.value
        if not selected_file:
            with config_feedback_output:
                print("No config file selected")
            return
        filepath = directory / selected_file
        if not filepath.exists():
            with config_feedback_output:
                print(f"Config file not found: {filepath}")
            return
        with filepath.open("r") as handle:
            snapshot = json.load(handle)
        _apply_configuration_snapshot(snapshot)
    except Exception as exc:
        with config_feedback_output:
            print(f"Error loading config: {exc}")


def _handle_refresh_config_list(button) -> None:
    with config_feedback_output:
        config_feedback_output.clear_output()
    _refresh_config_file_options()
    with config_feedback_output:
        print("✓ Config list refreshed")


def _update_core_controls(change) -> None:
    sim_value: SimulationType = (
        change["new"] if isinstance(change, dict) else simulation_widget.value
    )
    required_params = CORE_REQUIRED_PARAMS.get(sim_value, set())

    # Handle z_cutoff for all modes when checkbox is enabled
    if z_cutoff_checkbox.value:
        required_params = required_params | {"z_cutoff"}
    else:
        required_params = required_params - {"z_cutoff"}

    _apply_driver_visibility(sim_value)
    _apply_core_param_state(
        core_controls,
        required_params,
        z_cutoff_checkbox,
        random_seed_checkbox,
        sim_value,
    )
    image_subcharge_widget.disabled = sim_value != SimulationType.CONDUCTING_WALL
    image_subcharge_widget.layout.opacity = (
        1.0 if not image_subcharge_widget.disabled else 0.45
    )


# Recreate buttons to clear any existing callbacks from previous cell executions
config_refresh_button = widgets.Button(description="Refresh list", icon="refresh")
save_config_button = widgets.Button(description="Save config", icon="save")
load_config_button = widgets.Button(description="Load config", icon="folder-open")
run_button = widgets.Button(
    description="Run integrator", icon="play", button_style="primary"
)

if not hasattr(load_config_button, "_syncs_config_name"):
    load_config_button.on_click(_sync_config_name_after_load)
    load_config_button._syncs_config_name = True

# CRITICAL FIX: Clear all existing observers to prevent accumulation
# This handles the case where widgets persist across cell re-runs
if hasattr(simulation_widget, "_testbed_observers_attached"):
    # Clear observers from all widgets
    for widget in [
        simulation_widget,
        legacy_toggle,
        rider_species_widget,
        driver_species_widget,
    ]:
        if hasattr(widget, "unobserve_all"):
            widget.unobserve_all()
    for control in list(rider_controls.values()) + list(driver_controls.values()):
        if hasattr(control, "unobserve_all"):
            control.unobserve_all()
    delattr(simulation_widget, "_testbed_observers_attached")

# Attach observers
simulation_widget.observe(_update_core_controls, names="value")
legacy_toggle.observe(_update_legacy_controls, names="value")
simulation_widget.observe(_refresh_initial_properties, names="value")
z_cutoff_checkbox.observe(_update_core_controls, names="value")
random_seed_checkbox.observe(_update_core_controls, names="value")
for control in rider_controls.values():
    control.observe(_refresh_initial_properties, names="value")
for control in driver_controls.values():
    control.observe(_refresh_initial_properties, names="value")
# Observe seed changes in core_controls
if "seed" in core_controls:
    core_controls["seed"].observe(_refresh_initial_properties, names="value")
rider_species_widget.observe(
    lambda change: _apply_species_preset(rider_controls, change["new"]), names="value"
)
driver_species_widget.observe(
    lambda change: _apply_species_preset(driver_controls, change["new"]), names="value"
)
rider_species_widget.observe(_refresh_initial_properties, names="value")
driver_species_widget.observe(_refresh_initial_properties, names="value")

# Mark that observers have been attached
simulation_widget._testbed_observers_attached = True

# Button callbacks are always fresh since we recreated the buttons
config_refresh_button.on_click(_handle_refresh_config_list)
save_config_button.on_click(_handle_save_config)
load_config_button.on_click(_handle_load_config)


# In[ ]:
def _generate_filename_base() -> str:
    """Generate base filename from config name and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_base = config_name_widget.value.replace(".json", "")
    return f"{config_base}_{timestamp}"


_TIMESTAMP_TOKEN_LENGTH = 15


def _extract_scalar_series(states, key, default=0.0):
    values = []
    for state in states:
        data = state.get(key)
        if data is None or len(data) == 0:
            values.append(float(default))
        else:
            values.append(float(np.asarray(data)[0]))
    return np.asarray(values, dtype=float)


def _extract_vector_series(states, keys, default=0.0):
    components = [_extract_scalar_series(states, key, default) for key in keys]
    return np.stack(components, axis=-1)


# Use a lock so repeated button clicks cannot overlap runs
_MIN_RUN_INTERVAL_SECONDS = 0.5
_run_lock = threading.Lock()
_last_run_time = 0.0


def handle_run() -> None:
    global _last_run_time
    if not _run_lock.acquire(blocking=False):
        return

    try:
        now = time.monotonic()
        if now - _last_run_time < _MIN_RUN_INTERVAL_SECONDS:
            return
        _last_run_time = now

        sim_type_value = simulation_widget.value
        supports_driver = _supports_driver(sim_type_value)
        rider_params = _collect_particle_values(rider_controls)
        driver_params = (
            _collect_particle_values(driver_controls) if supports_driver else None
        )
        core_params = _collect_core_values(core_controls)
        required_params = CORE_REQUIRED_PARAMS.get(sim_type_value, set())

        # Handle z_cutoff for all modes when checkbox is enabled
        if z_cutoff_checkbox.value:
            required_params = required_params | {"z_cutoff"}
        else:
            required_params = required_params - {"z_cutoff"}

        # Extract steps and seed from core_params
        num_steps = int(core_params.get("steps", 1000))
        # Use random seed if checkbox is enabled
        if random_seed_checkbox.value:
            import random

            seed = random.randint(1, 2**31 - 1)
            log(f"  Generated random seed: {seed}")
        else:
            seed = int(core_params.get("seed", 12345))

        filtered_core_params = {
            name: value
            for name, value in core_params.items()
            if name in required_params and name not in ("steps", "seed")
        }
        legacy_enabled = legacy_toggle.value
        dpi_value = dpi_widget.value
        image_subcharge_count = int(image_subcharge_widget.value)
        filename_base = _generate_filename_base()
        timestamp_token = filename_base[-_TIMESTAMP_TOKEN_LENGTH:]
        if len(filename_base) > (_TIMESTAMP_TOKEN_LENGTH + 1):
            config_label = filename_base[: -(_TIMESTAMP_TOKEN_LENGTH + 1)]
        else:
            config_label = filename_base
        should_save_plots = any(
            [
                overlay_save_widget.value,
                difference_save_widget.value,
                metrics_save_widget.value,
                energy_save_widget.value,
                transverse_save_widget.value,
                trajectory_save_widget.value,
            ]
        )

        output_dir = _save_dir_if_needed(should_save_plots)
        with output_area:
            output_area.clear_output(wait=True)
            display(widgets.HTML(value="<p><strong>Running...</strong></p>"))

            try:
                log_buffer = io.StringIO()

                def log(message: str) -> None:
                    print(message, file=log_buffer)
                    print(message)

                log(
                    f"Running {sim_type_value.name.replace('_', ' ').title()} integrator for {num_steps} steps..."
                )
                log(f"  Steps: {num_steps}")
                log(f"  Seed: {seed}")
                log(f"  Core params: {filtered_core_params}")
                log(f"  Legacy enabled: {legacy_enabled}")
                log(f"  Image subcharges: {image_subcharge_count}")
                log("")

                return_traj_flag = any(
                    [
                        trajectory_save_widget.value,
                        transverse_display_widget.value,
                        transverse_save_widget.value,
                        energy_save_widget.value,
                        energy_display_widget.value,
                        overlay_display_widget.value,
                        overlay_save_widget.value,
                    ]
                )

                result = run_benchmark(
                    steps=num_steps,
                    simulation_type=sim_type_value,
                    rider_params=rider_params,
                    driver_params=driver_params,
                    seed=seed,
                    legacy_enabled=legacy_enabled,
                    return_trajectories=return_traj_flag,
                    image_subcharge_count=image_subcharge_count,
                    **filtered_core_params,
                )

                if isinstance(result, tuple) and len(result) == 2:
                    metrics, payload = result
                else:
                    metrics = result
                    payload = {}

                global LAST_METRICS, LAST_PAYLOAD, LAST_LOG_MESSAGES
                LAST_METRICS = metrics
                LAST_PAYLOAD = payload

                log("✓ Integration complete")

                if legacy_enabled and metrics_save_widget.value:
                    metrics_filename = f"{filename_base}_metrics.json"
                    metrics_path = output_dir / metrics_filename
                    with metrics_path.open("w") as handle:
                        json.dump(metrics, handle, indent=2, default=str)
                    log(f"Saved metrics to: {metrics_path}")

                core_traj = payload.get("core")
                legacy_traj = payload.get("legacy") if legacy_enabled else None
                initial_states = payload.get("initial_states", {})
                rest_energies = payload.get("rest_energy_mev", {})

                if core_traj is not None:
                    rider_delta_e, rider_z = compute_delta_energy_series(
                        core_traj["rider"],
                        initial_states.get("rider"),
                        rest_energies.get("rider"),
                    )
                    rider_z_rel = rider_z - rider_z[0]
                    # Extract time series for dual x-axis
                    rider_t = np.array([s["t"][0] for s in core_traj["rider"]])

                    if supports_driver:
                        driver_delta_e, driver_z = compute_delta_energy_series(
                            core_traj["driver"],
                            initial_states.get("driver"),
                            rest_energies.get("driver"),
                        )
                        driver_z_rel = driver_z - driver_z[0]
                        driver_t = np.array([s["t"][0] for s in core_traj["driver"]])
                    else:
                        driver_delta_e = None
                        driver_z_rel = None
                        driver_t = None

                    if legacy_enabled and legacy_traj:
                        legacy_rider_delta_e, legacy_rider_z = (
                            compute_delta_energy_series(
                                legacy_traj["rider"],
                                initial_states.get("rider"),
                                rest_energies.get("rider"),
                            )
                        )
                        legacy_rider_z_rel = legacy_rider_z - legacy_rider_z[0]
                        legacy_rider_t = np.array(
                            [s["t"][0] for s in legacy_traj["rider"]]
                        )

                        if supports_driver:
                            legacy_driver_delta_e, legacy_driver_z = (
                                compute_delta_energy_series(
                                    legacy_traj["driver"],
                                    initial_states.get("driver"),
                                    rest_energies.get("driver"),
                                )
                            )
                            legacy_driver_z_rel = legacy_driver_z - legacy_driver_z[0]
                            legacy_driver_t = np.array(
                                [s["t"][0] for s in legacy_traj["driver"]]
                            )
                        else:
                            legacy_driver_delta_e = None
                            legacy_driver_z_rel = None
                            legacy_driver_t = None
                    else:
                        legacy_rider_delta_e = None
                        legacy_rider_z_rel = None
                        legacy_rider_t = None
                        legacy_driver_delta_e = None
                        legacy_driver_z_rel = None
                        legacy_driver_t = None

                    if energy_save_widget.value or energy_display_widget.value:
                        # Extract energy plot configuration
                        xaxis_mode = energy_xaxis_widget.value  # "z", "t", or "dual"
                        yaxis_mode = (
                            energy_yaxis_widget.value
                        )  # "delta_total", "delta_z", etc.

                        # Prepare x-axis data
                        if xaxis_mode == "z":
                            rider_x = rider_z_rel
                            driver_x = driver_z_rel if supports_driver else None
                            xlabel = "z (mm)"
                        elif xaxis_mode == "t":
                            rider_x = rider_t
                            driver_x = driver_t if supports_driver else None
                            xlabel = "t (ns)"
                        else:  # dual
                            rider_x = rider_z_rel
                            driver_x = driver_z_rel if supports_driver else None
                            xlabel = "z (mm)"

                        # Prepare y-axis data
                        if yaxis_mode == "delta_total":
                            rider_y = rider_delta_e
                            driver_y = driver_delta_e if supports_driver else None
                            ylabel = "ΔE (GeV)"
                            title_suffix = "Energy Change"
                        elif yaxis_mode == "delta_z":
                            # Extract ΔE_z from trajectories
                            rider_Pz = np.array(
                                [s["Pz"][0] for s in core_traj["rider"]]
                            )
                            rider_Pz0 = rider_Pz[0]
                            rider_y = (
                                (rider_Pz - rider_Pz0) * 3e8 * 1e-9 / 1e9
                            )  # Convert to GeV
                            if supports_driver and driver_delta_e is not None:
                                driver_Pz = np.array(
                                    [s["Pz"][0] for s in core_traj["driver"]]
                                )
                                driver_Pz0 = driver_Pz[0]
                                driver_y = (driver_Pz - driver_Pz0) * 3e8 * 1e-9 / 1e9
                            else:
                                driver_y = None
                            ylabel = "ΔE_z (GeV)"
                            title_suffix = "Longitudinal Energy Change"
                        elif yaxis_mode == "delta_x":
                            rider_Px = np.array(
                                [s["Px"][0] for s in core_traj["rider"]]
                            )
                            rider_Px0 = rider_Px[0]
                            rider_y = (rider_Px - rider_Px0) * 3e8 * 1e-9 / 1e9
                            if supports_driver and driver_delta_e is not None:
                                driver_Px = np.array(
                                    [s["Px"][0] for s in core_traj["driver"]]
                                )
                                driver_Px0 = driver_Px[0]
                                driver_y = (driver_Px - driver_Px0) * 3e8 * 1e-9 / 1e9
                            else:
                                driver_y = None
                            ylabel = "ΔE_x (GeV)"
                            title_suffix = "Transverse-X Energy Change"
                        elif yaxis_mode == "delta_y":
                            rider_Py = np.array(
                                [s["Py"][0] for s in core_traj["rider"]]
                            )
                            rider_Py0 = rider_Py[0]
                            rider_y = (rider_Py - rider_Py0) * 3e8 * 1e-9 / 1e9
                            if supports_driver and driver_delta_e is not None:
                                driver_Py = np.array(
                                    [s["Py"][0] for s in core_traj["driver"]]
                                )
                                driver_Py0 = driver_Py[0]
                                driver_y = (driver_Py - driver_Py0) * 3e8 * 1e-9 / 1e9
                            else:
                                driver_y = None
                            ylabel = "ΔE_y (GeV)"
                            title_suffix = "Transverse-Y Energy Change"
                        else:  # total (absolute energy)
                            rider_E = np.array([s["E"][0] for s in core_traj["rider"]])
                            rider_y = rider_E * 3e8 * 1e-9 / 1e9  # Convert to GeV
                            if supports_driver and driver_delta_e is not None:
                                driver_E = np.array(
                                    [s["E"][0] for s in core_traj["driver"]]
                                )
                                driver_y = driver_E * 3e8 * 1e-9 / 1e9
                            else:
                                driver_y = None
                            ylabel = "E (GeV)"
                            title_suffix = "Total Energy"

                        fig_energy, axes_energy = plt.subplots(
                            1,
                            2 if supports_driver else 1,
                            figsize=(16 if supports_driver else 8, 6),
                            dpi=dpi_value,
                        )
                        if not supports_driver:
                            axes_energy = [axes_energy]

                        # Rider plot
                        axes_energy[0].scatter(
                            rider_x,
                            rider_y,
                            color=COLOR_RIDER,
                            label="Core",
                            **SCATTER_STYLE,
                        )
                        if (
                            legacy_enabled
                            and legacy_rider_delta_e is not None
                            and yaxis_mode == "delta_total"
                        ):
                            legacy_x = (
                                legacy_rider_z_rel
                                if xaxis_mode == "z"
                                else (
                                    legacy_rider_t
                                    if xaxis_mode == "t"
                                    else legacy_rider_z_rel
                                )
                            )
                            axes_energy[0].scatter(
                                legacy_x,
                                legacy_rider_delta_e,
                                color=COLOR_LEGACY_RIDER,
                                label="Legacy",
                                **SCATTER_STYLE,
                            )
                        axes_energy[0].set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
                        axes_energy[0].set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
                        axes_energy[0].set_title(
                            f"Rider {title_suffix}", fontsize=TITLE_FONTSIZE
                        )
                        axes_energy[0].legend(fontsize=LEGEND_FONTSIZE)
                        axes_energy[0].tick_params(labelsize=TICK_FONTSIZE)
                        axes_energy[0].grid(True, alpha=0.3)

                        # Add dual x-axis if requested
                        if xaxis_mode == "dual":
                            ax_time_0 = axes_energy[0].twiny()
                            ax_time_0.scatter(rider_t, rider_y, alpha=0)
                            ax_time_0.set_xlabel("t (ns)", fontsize=LABEL_FONTSIZE)
                            ax_time_0.tick_params(labelsize=TICK_FONTSIZE)

                        if supports_driver and driver_y is not None:
                            axes_energy[1].scatter(
                                driver_x,
                                driver_y,
                                color=COLOR_DRIVER,
                                label="Core",
                                **SCATTER_STYLE,
                            )
                            if (
                                legacy_enabled
                                and legacy_driver_delta_e is not None
                                and yaxis_mode == "delta_total"
                            ):
                                legacy_x = (
                                    legacy_driver_z_rel
                                    if xaxis_mode == "z"
                                    else (
                                        legacy_driver_t
                                        if xaxis_mode == "t"
                                        else legacy_driver_z_rel
                                    )
                                )
                                axes_energy[1].scatter(
                                    legacy_x,
                                    legacy_driver_delta_e,
                                    color=COLOR_LEGACY_DRIVER,
                                    label="Legacy",
                                    **SCATTER_STYLE,
                                )
                            axes_energy[1].set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
                            axes_energy[1].set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
                            axes_energy[1].set_title(
                                f"Driver {title_suffix}", fontsize=TITLE_FONTSIZE
                            )
                            axes_energy[1].legend(fontsize=LEGEND_FONTSIZE)
                            axes_energy[1].tick_params(labelsize=TICK_FONTSIZE)
                            axes_energy[1].grid(True, alpha=0.3)

                            # Add dual x-axis if requested
                            if xaxis_mode == "dual":
                                ax_time_1 = axes_energy[1].twiny()
                                ax_time_1.scatter(driver_t, driver_y, alpha=0)
                                ax_time_1.set_xlabel("t (ns)", fontsize=LABEL_FONTSIZE)
                                ax_time_1.tick_params(labelsize=TICK_FONTSIZE)

                        fig_energy.tight_layout()
                        if energy_save_widget.value:
                            energy_path = output_dir / f"{filename_base}_energy.png"
                            fig_energy.savefig(energy_path, bbox_inches="tight")
                            log(f"Saved energy plots to: {energy_path}")
                        if energy_display_widget.value:
                            display(fig_energy)
                        plt.close(fig_energy)

                    if (
                        legacy_enabled
                        and (overlay_display_widget.value or overlay_save_widget.value)
                        and legacy_rider_delta_e is not None
                    ):
                        fig_overlay, axes_overlay = plt.subplots(
                            1,
                            2 if supports_driver else 1,
                            figsize=(16 if supports_driver else 8, 6),
                            dpi=dpi_value,
                        )
                        if not supports_driver:
                            axes_overlay = [axes_overlay]

                        axes_overlay[0].plot(
                            rider_z_rel,
                            rider_delta_e,
                            color=COLOR_RIDER,
                            label="Core",
                            linewidth=2.0,
                        )
                        axes_overlay[0].plot(
                            legacy_rider_z_rel,
                            legacy_rider_delta_e,
                            color=COLOR_LEGACY_RIDER,
                            label="Legacy",
                            linewidth=2.0,
                            linestyle="--",
                        )
                        axes_overlay[0].set_xlabel("Δz (mm)", fontsize=LABEL_FONTSIZE)
                        axes_overlay[0].set_ylabel("ΔE (GeV)", fontsize=LABEL_FONTSIZE)
                        axes_overlay[0].set_title(
                            "Rider ΔE Comparison", fontsize=TITLE_FONTSIZE
                        )
                        axes_overlay[0].legend(fontsize=LEGEND_FONTSIZE)
                        axes_overlay[0].tick_params(labelsize=TICK_FONTSIZE)
                        axes_overlay[0].grid(True, alpha=0.3)

                        if (
                            supports_driver
                            and driver_delta_e is not None
                            and legacy_driver_delta_e is not None
                        ):
                            axes_overlay[1].plot(
                                driver_z_rel,
                                driver_delta_e,
                                color=COLOR_DRIVER,
                                label="Core",
                                linewidth=2.0,
                            )
                            axes_overlay[1].plot(
                                legacy_driver_z_rel,
                                legacy_driver_delta_e,
                                color=COLOR_LEGACY_DRIVER,
                                label="Legacy",
                                linewidth=2.0,
                                linestyle="--",
                            )
                            axes_overlay[1].set_xlabel(
                                "Δz (mm)", fontsize=LABEL_FONTSIZE
                            )
                            axes_overlay[1].set_ylabel(
                                "ΔE (GeV)", fontsize=LABEL_FONTSIZE
                            )
                            axes_overlay[1].set_title(
                                "Driver ΔE Comparison", fontsize=TITLE_FONTSIZE
                            )
                            axes_overlay[1].legend(fontsize=LEGEND_FONTSIZE)
                            axes_overlay[1].tick_params(labelsize=TICK_FONTSIZE)
                            axes_overlay[1].grid(True, alpha=0.3)

                        fig_overlay.tight_layout()
                        if overlay_save_widget.value:
                            overlay_path = output_dir / f"{filename_base}_overlay.png"
                            fig_overlay.savefig(overlay_path, bbox_inches="tight")
                            log(f"Saved overlay plots to: {overlay_path}")
                        if overlay_display_widget.value:
                            display(fig_overlay)
                        plt.close(fig_overlay)

                    rider_states = core_traj["rider"]
                    driver_states = core_traj["driver"] if supports_driver else None
                    core_r_hist = np.array(
                        [
                            [s["t"][0], s["x"][0], s["y"][0], s["z"][0]]
                            for s in rider_states
                        ]
                    )
                    core_r_gamma = _extract_scalar_series(rider_states, "gamma")
                    core_r_momentum = _extract_vector_series(
                        rider_states, ("Px", "Py", "Pz")
                    )
                    core_r_beta = _extract_vector_series(
                        rider_states, ("bx", "by", "bz")
                    )
                    core_r_betadot = _extract_vector_series(
                        rider_states, ("bdotx", "bdoty", "bdotz")
                    )
                    core_r_pt = _extract_scalar_series(rider_states, "Pt")
                    plot_times_ns = core_r_hist[:, 0]

                    if supports_driver and driver_states is not None:
                        core_d_hist = np.array(
                            [
                                [s["t"][0], s["x"][0], s["y"][0], s["z"][0]]
                                for s in driver_states
                            ]
                        )
                        core_d_gamma = _extract_scalar_series(driver_states, "gamma")
                        core_d_momentum = _extract_vector_series(
                            driver_states, ("Px", "Py", "Pz")
                        )
                        core_d_beta = _extract_vector_series(
                            driver_states, ("bx", "by", "bz")
                        )
                        core_d_betadot = _extract_vector_series(
                            driver_states, ("bdotx", "bdoty", "bdotz")
                        )
                        core_d_pt = _extract_scalar_series(driver_states, "Pt")
                    else:
                        core_d_hist = None
                        core_d_gamma = None
                        core_d_momentum = None
                        core_d_beta = None
                        core_d_betadot = None
                        core_d_pt = None

                    if legacy_enabled and legacy_traj:
                        legacy_rider_states = legacy_traj["rider"]
                        legacy_r_hist = np.array(
                            [
                                [s["t"][0], s["x"][0], s["y"][0], s["z"][0]]
                                for s in legacy_rider_states
                            ]
                        )
                        legacy_r_gamma = _extract_scalar_series(
                            legacy_rider_states, "gamma"
                        )
                        legacy_r_momentum = _extract_vector_series(
                            legacy_rider_states, ("Px", "Py", "Pz")
                        )
                        legacy_r_beta = _extract_vector_series(
                            legacy_rider_states, ("bx", "by", "bz")
                        )
                        legacy_r_betadot = _extract_vector_series(
                            legacy_rider_states, ("bdotx", "bdoty", "bdotz")
                        )
                        legacy_r_pt = _extract_scalar_series(legacy_rider_states, "Pt")
                        if supports_driver and legacy_traj.get("driver") is not None:
                            legacy_driver_states = legacy_traj["driver"]
                            legacy_d_hist = np.array(
                                [
                                    [s["t"][0], s["x"][0], s["y"][0], s["z"][0]]
                                    for s in legacy_driver_states
                                ]
                            )
                            legacy_d_gamma = _extract_scalar_series(
                                legacy_driver_states, "gamma"
                            )
                            legacy_d_momentum = _extract_vector_series(
                                legacy_driver_states, ("Px", "Py", "Pz")
                            )
                            legacy_d_beta = _extract_vector_series(
                                legacy_driver_states, ("bx", "by", "bz")
                            )
                            legacy_d_betadot = _extract_vector_series(
                                legacy_driver_states, ("bdotx", "bdoty", "bdotz")
                            )
                            legacy_d_pt = _extract_scalar_series(
                                legacy_driver_states, "Pt"
                            )
                        else:
                            legacy_d_hist = None
                            legacy_d_gamma = None
                            legacy_d_momentum = None
                            legacy_d_beta = None
                            legacy_d_betadot = None
                            legacy_d_pt = None
                    else:
                        legacy_r_hist = None
                        legacy_r_gamma = None
                        legacy_r_momentum = None
                        legacy_r_beta = None
                        legacy_r_betadot = None
                        legacy_r_pt = None
                        legacy_d_hist = None
                        legacy_d_gamma = None
                        legacy_d_momentum = None
                        legacy_d_beta = None
                        legacy_d_betadot = None
                        legacy_d_pt = None

                    if (
                        legacy_enabled
                        and difference_save_widget.value
                        and legacy_r_hist is not None
                    ):
                        diff_path = output_dir / f"{filename_base}_difference.png"
                        fig_diff, axes_diff = plt.subplots(
                            1,
                            2 if supports_driver else 1,
                            figsize=(16 if supports_driver else 8, 6),
                            dpi=dpi_value,
                        )
                        if not supports_driver:
                            axes_diff = [axes_diff]

                        r_delta_x = (core_r_hist[:, 1] - legacy_r_hist[:, 1]) * 1e3
                        r_delta_y = (core_r_hist[:, 2] - legacy_r_hist[:, 2]) * 1e3
                        r_delta_z = (core_r_hist[:, 3] - legacy_r_hist[:, 3]) * 1e3
                        axes_diff[0].plot(
                            plot_times_ns,
                            r_delta_x,
                            label="Δx (mm)",
                            color=COLOR_DIFF_RIDER,
                        )
                        axes_diff[0].plot(
                            plot_times_ns,
                            r_delta_y,
                            label="Δy (mm)",
                            color=COLOR_DIFF_RIDER,
                            linestyle="--",
                        )
                        axes_diff[0].plot(
                            plot_times_ns,
                            r_delta_z,
                            label="Δz (mm)",
                            color=COLOR_DIFF_RIDER,
                            linestyle=":",
                        )
                        axes_diff[0].set_xlabel("Time (ns)", fontsize=LABEL_FONTSIZE)
                        axes_diff[0].set_ylabel("Δ (mm)", fontsize=LABEL_FONTSIZE)
                        axes_diff[0].set_title(
                            "Rider: Δ(core − legacy)", fontsize=TITLE_FONTSIZE
                        )
                        axes_diff[0].legend(fontsize=LEGEND_FONTSIZE)
                        axes_diff[0].tick_params(labelsize=TICK_FONTSIZE)
                        axes_diff[0].grid(True, alpha=0.3)

                        if core_d_hist is not None and legacy_d_hist is not None:
                            d_delta_x = (core_d_hist[:, 1] - legacy_d_hist[:, 1]) * 1e3
                            d_delta_y = (core_d_hist[:, 2] - legacy_d_hist[:, 2]) * 1e3
                            d_delta_z = (core_d_hist[:, 3] - legacy_d_hist[:, 3]) * 1e3
                            axes_diff[1].plot(
                                plot_times_ns,
                                d_delta_x,
                                label="Δx (mm)",
                                color=COLOR_DIFF_DRIVER,
                            )
                            axes_diff[1].plot(
                                plot_times_ns,
                                d_delta_y,
                                label="Δy (mm)",
                                color=COLOR_DIFF_DRIVER,
                                linestyle="--",
                            )
                            axes_diff[1].plot(
                                plot_times_ns,
                                d_delta_z,
                                label="Δz (mm)",
                                color=COLOR_DIFF_DRIVER,
                                linestyle=":",
                            )
                            axes_diff[1].set_xlabel(
                                "Time (ns)", fontsize=LABEL_FONTSIZE
                            )
                            axes_diff[1].set_ylabel("Δ (mm)", fontsize=LABEL_FONTSIZE)
                            axes_diff[1].set_title(
                                "Driver: Δ(core − legacy)", fontsize=TITLE_FONTSIZE
                            )
                            axes_diff[1].legend(fontsize=LEGEND_FONTSIZE)
                            axes_diff[1].tick_params(labelsize=TICK_FONTSIZE)
                            axes_diff[1].grid(True, alpha=0.3)

                        fig_diff.tight_layout()
                        if difference_save_widget.value:
                            fig_diff.savefig(diff_path, bbox_inches="tight")
                            log(f"Saved difference plot to: {diff_path}")
                        if difference_display_widget.value:
                            display(fig_diff)
                        plt.close(fig_diff)

                    if transverse_display_widget.value or transverse_save_widget.value:
                        fig_transverse, (ax_x, ax_y) = plt.subplots(
                            1, 2, figsize=(16, 6), dpi=dpi_value
                        )
                        core_r_x = core_r_hist[:, 1] * 1e3
                        core_r_y = core_r_hist[:, 2] * 1e3
                        ax_x.plot(
                            plot_times_ns,
                            core_r_x,
                            color=COLOR_RIDER,
                            label="Rider (Core)",
                        )
                        ax_y.plot(
                            plot_times_ns,
                            core_r_y,
                            color=COLOR_RIDER,
                            label="Rider (Core)",
                        )
                        if supports_driver and core_d_hist is not None:
                            core_d_x = core_d_hist[:, 1] * 1e3
                            core_d_y = core_d_hist[:, 2] * 1e3
                            ax_x.plot(
                                plot_times_ns,
                                core_d_x,
                                color=COLOR_DRIVER,
                                label="Driver (Core)",
                            )
                            ax_y.plot(
                                plot_times_ns,
                                core_d_y,
                                color=COLOR_DRIVER,
                                label="Driver (Core)",
                            )
                        if legacy_enabled and legacy_r_hist is not None:
                            legacy_r_x = legacy_r_hist[:, 1] * 1e3
                            legacy_r_y = legacy_r_hist[:, 2] * 1e3
                            ax_x.plot(
                                plot_times_ns,
                                legacy_r_x,
                                color=COLOR_LEGACY_RIDER,
                                linestyle="--",
                                label="Rider (Legacy)",
                            )
                            ax_y.plot(
                                plot_times_ns,
                                legacy_r_y,
                                color=COLOR_LEGACY_RIDER,
                                linestyle="--",
                                label="Rider (Legacy)",
                            )
                            if supports_driver and legacy_d_hist is not None:
                                legacy_d_x = legacy_d_hist[:, 1] * 1e3
                                legacy_d_y = legacy_d_hist[:, 2] * 1e3
                                ax_x.plot(
                                    plot_times_ns,
                                    legacy_d_x,
                                    color=COLOR_LEGACY_DRIVER,
                                    linestyle="--",
                                    label="Driver (Legacy)",
                                )
                                ax_y.plot(
                                    plot_times_ns,
                                    legacy_d_y,
                                    color=COLOR_LEGACY_DRIVER,
                                    linestyle="--",
                                    label="Driver (Legacy)",
                                )
                        ax_x.set_xlabel("Time (ns)", fontsize=LABEL_FONTSIZE)
                        ax_x.set_ylabel("⟨x⟩ (mm)", fontsize=LABEL_FONTSIZE)
                        ax_x.set_title("Average X Position", fontsize=TITLE_FONTSIZE)
                        ax_x.legend(fontsize=LEGEND_FONTSIZE)
                        ax_x.tick_params(labelsize=TICK_FONTSIZE)
                        ax_x.grid(True, alpha=0.3)
                        ax_y.set_xlabel("Time (ns)", fontsize=LABEL_FONTSIZE)
                        ax_y.set_ylabel("⟨y⟩ (mm)", fontsize=LABEL_FONTSIZE)
                        ax_y.set_title("Average Y Position", fontsize=TITLE_FONTSIZE)
                        ax_y.legend(fontsize=LEGEND_FONTSIZE)
                        ax_y.tick_params(labelsize=TICK_FONTSIZE)
                        ax_y.grid(True, alpha=0.3)
                        fig_transverse.tight_layout()
                        if transverse_save_widget.value:
                            transverse_path = (
                                output_dir / f"{filename_base}_transverse.png"
                            )
                            fig_transverse.savefig(transverse_path, bbox_inches="tight")
                            log(f"Saved transverse plots to: {transverse_path}")
                        if transverse_display_widget.value:
                            display(fig_transverse)
                        plt.close(fig_transverse)

                    if trajectory_save_widget.value:
                        interval = max(1, int(trajectory_interval_widget.value))

                        def _build_particle_payload(
                            hist, gamma, momentum, beta, betadot, pt
                        ):
                            return {
                                "r_hist": hist[::interval].tolist(),
                                "gamma_hist": gamma[::interval].tolist(),
                                "time_ns": hist[::interval, 0].tolist(),
                                "positions_mm": {
                                    "x": hist[::interval, 1].tolist(),
                                    "y": hist[::interval, 2].tolist(),
                                    "z": hist[::interval, 3].tolist(),
                                },
                                "conjugate_momenta": {
                                    "Px": momentum[::interval, 0].tolist(),
                                    "Py": momentum[::interval, 1].tolist(),
                                    "Pz": momentum[::interval, 2].tolist(),
                                },
                                "beta": {
                                    "bx": beta[::interval, 0].tolist(),
                                    "by": beta[::interval, 1].tolist(),
                                    "bz": beta[::interval, 2].tolist(),
                                },
                                "betadot": {
                                    "bdotx": betadot[::interval, 0].tolist(),
                                    "bdoty": betadot[::interval, 1].tolist(),
                                    "bdotz": betadot[::interval, 2].tolist(),
                                },
                                "pt_hist": pt[::interval].tolist(),
                            }

                        core_payload = {
                            "rider": _build_particle_payload(
                                core_r_hist,
                                core_r_gamma,
                                core_r_momentum,
                                core_r_beta,
                                core_r_betadot,
                                core_r_pt,
                            )
                        }
                        if (
                            supports_driver
                            and core_d_hist is not None
                            and core_d_momentum is not None
                        ):
                            core_payload["driver"] = _build_particle_payload(
                                core_d_hist,
                                core_d_gamma,
                                core_d_momentum,
                                core_d_beta,
                                core_d_betadot,
                                core_d_pt,
                            )

                        traj_data = {
                            "config_name": config_name_widget.value,
                            "config_label": config_label,
                            "seed": seed,
                            "num_steps": num_steps,
                            "simulation_type": sim_type_value.name,
                            "step_interval": interval,
                            "timestamp": timestamp_token,
                            "core": core_payload,
                            "image_subcharge_count": image_subcharge_count,
                        }

                        if (
                            legacy_enabled
                            and legacy_r_hist is not None
                            and legacy_r_momentum is not None
                        ):
                            legacy_payload = {
                                "rider": _build_particle_payload(
                                    legacy_r_hist,
                                    legacy_r_gamma,
                                    legacy_r_momentum,
                                    legacy_r_beta,
                                    legacy_r_betadot,
                                    legacy_r_pt,
                                )
                            }
                            if (
                                supports_driver
                                and legacy_d_hist is not None
                                and legacy_d_momentum is not None
                            ):
                                legacy_payload["driver"] = _build_particle_payload(
                                    legacy_d_hist,
                                    legacy_d_gamma,
                                    legacy_d_momentum,
                                    legacy_d_beta,
                                    legacy_d_betadot,
                                    legacy_d_pt,
                                )
                            traj_data["legacy"] = legacy_payload

                        label_prefix = config_label if config_label else "trajectory"
                        traj_filename = (
                            f"{label_prefix}_trajectory_data_{timestamp_token}.json"
                        )
                        traj_path = output_dir / traj_filename
                        with traj_path.open("w") as handle:
                            json.dump(traj_data, handle, indent=2)
                        log(f"Saved trajectories to: {traj_path} (interval={interval})")

                log("\n✓ Run complete!")
                LAST_LOG_MESSAGES = log_buffer.getvalue().splitlines()

            except Exception as exc:
                output_area.clear_output(wait=True)
                display(
                    widgets.HTML(
                        value=f"<p><strong>Error during run:</strong> {exc}</p>"
                    )
                )
                import traceback

                traceback.print_exc()
    finally:
        _run_lock.release()


run_button.on_click(lambda _btn: handle_run())

# Initialize UI state after definitions so widgets refresh with latest callbacks
_update_legacy_controls(None)
_update_core_controls(None)
_refresh_config_file_options()
_refresh_initial_properties()

config_controls = widgets.VBox(
    [
        widgets.HTML("<h3 style='margin:8px 0'>Configuration Management</h3>"),
        widgets.HBox([config_dir_widget], layout=ROW_LAYOUT),
        widgets.HBox([config_name_widget, save_config_button], layout=ROW_LAYOUT),
        widgets.HBox(
            [config_file_dropdown, load_config_button, config_refresh_button],
            layout=ROW_LAYOUT,
        ),
        config_feedback_output,
    ]
)
controls_layout = widgets.VBox(
    [
        widgets.HTML("<h2 style='margin:12px 0 8px'>Integrator Testbed</h2>"),
        widgets.HBox(
            [simulation_widget, legacy_toggle],
            layout=ROW_LAYOUT,
        ),
        initial_state_output,
        rider_section,
        driver_section,
        core_section,
        widgets.HBox([image_subcharge_widget], layout=ROW_LAYOUT),
        widgets.HTML("<h3 style='margin:12px 0 8px'>Plot & Export Options</h3>"),
        widgets.HBox([overlay_display_widget, overlay_save_widget], layout=ROW_LAYOUT),
        widgets.HBox(
            [difference_display_widget, difference_save_widget], layout=ROW_LAYOUT
        ),
        widgets.HBox([energy_display_widget, energy_save_widget], layout=ROW_LAYOUT),
        widgets.HBox([energy_xaxis_widget, energy_yaxis_widget], layout=ROW_LAYOUT),
        widgets.HBox(
            [transverse_display_widget, transverse_save_widget], layout=ROW_LAYOUT
        ),
        widgets.HBox(
            [metrics_save_widget, trajectory_save_widget, trajectory_interval_widget],
            layout=ROW_LAYOUT,
        ),
        widgets.HBox([dpi_widget, output_dir_widget], layout=ROW_LAYOUT),
        config_controls,
        widgets.HBox([run_button], layout=ROW_LAYOUT),
        output_area,
    ],
    layout=widgets.Layout(
        width="98%",
        max_height="95vh",
        overflow_y="auto",
        overflow_x="hidden",
        border="1px solid #ddd",
        padding="10px",
    ),
)

display(controls_layout)


# In[ ]:


# In[ ]:
