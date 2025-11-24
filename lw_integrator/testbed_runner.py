"""Helper utilities mirroring the notebook testbed inside a desktop GUI.

The original ``integrator_testbed.ipynb`` notebook wires dozens of ipywidgets
around the benchmarking helpers in ``examples/validation/core_vs_legacy_benchmark``.
This module repackages that behaviour behind plain Python data structures so a
Tkinter GUI (or any other front-end) can drive the same workflows: generating
energy plots, exporting metrics, saving down-sampled trajectories, and managing
JSON snapshot files.

All strings deliberately use ASCII to keep packaging simple when rendered in
terminals that do not default to UTF-8.
"""

from __future__ import annotations

import json
import time
import traceback

import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from examples.validation.core_vs_legacy_benchmark import (  # type: ignore[import]
    DEFAULT_DRIVER_PARAMS,
    DEFAULT_RIDER_PARAMS,
    PARTICLE_PARAM_FIELDS,
    SimulationType,
    compute_delta_energy_series,
    prepare_two_particle_demo,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Constants mirroring the notebook defaults
# ---------------------------------------------------------------------------

COLOR_RIDER = "#0072B2"
COLOR_DRIVER = "#D55E00"
COLOR_LEGACY_RIDER = "#56B4E9"
COLOR_LEGACY_DRIVER = "#E69F00"
COLOR_DIFF_RIDER = "#009E73"
COLOR_DIFF_DRIVER = "#CC79A7"
SCATTER_STYLE = {"s": 140, "alpha": 0.78, "linewidth": 0, "edgecolors": "none"}
AVAILABLE_DPI_CHOICES: Tuple[int, ...] = (150, 300, 450, 600)
DEFAULT_PLOT_DPI = 300

PARAM_LABELS: Dict[str, str] = {
    "starting_distance": "Start z (mm)",
    "transv_mom": "Transverse momentum (amu*mm/ns)",
    "starting_Pz": "Initial Pz (amu*mm/ns)",
    "stripped_ions": "Stripped ions",
    "m_particle": "Mass (amu)",
    "transv_dist": "Transverse spread (mm)",
    "pcount": "Particle count",
    "charge_sign": "Charge sign",
}

CORE_PARAM_LABELS: Dict[str, str] = {
    "time_step": "Time step (ns)",
    "wall_z": "Wall z (mm)",
    "aperture_radius": "Aperture radius (mm)",
    "mean": "Mean separation (mm)",
    "cav_spacing": "Cavity spacing (mm)",
    "z_cutoff": "z cutoff (mm)",
}

CORE_PARAM_DEFAULTS: Dict[str, float] = {
    "time_step": 2.2e-7,
    "wall_z": 1.0e5,
    "aperture_radius": 1.0e5,
    "mean": 1.0e5,
    "cav_spacing": 1.0e5,
    "z_cutoff": 0.0,
}

CORE_REQUIRED_PARAMS: Dict[SimulationType, set[str]] = {
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

SPECIES_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Custom / manual", "custom"),
    ("Electron (e-)", "electron"),
    ("Positron (e+)", "positron"),
    ("Proton (p+)", "proton"),
    ("Antiproton (pbar)", "antiproton"),
    ("Lead ion (Pb^82+)", "lead"),
    ("Gold ion (Au^79+)", "gold"),
)

_TIMESTAMP_TOKEN_LENGTH = 15

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }
)

# ---------------------------------------------------------------------------
# Dataclasses for strongly typed options/results
# ---------------------------------------------------------------------------


@dataclass
class SimulationOptions:
    """Full parameter surface matching the notebook widget snapshot."""

    steps: int = 1000
    seed: int = 12345
    simulation_type: SimulationType = SimulationType.BUNCH_TO_BUNCH
    legacy_enabled: bool = False
    overlay_display: bool = False
    overlay_save: bool = False
    difference_display: bool = False
    difference_save: bool = False
    metrics_save: bool = False
    energy_display: bool = True
    energy_save: bool = True
    transverse_display: bool = False
    transverse_save: bool = False
    trajectory_save: bool = False
    trajectory_interval: int = 10
    plot_dpi: int = DEFAULT_PLOT_DPI
    output_dir: Path = Path("test_outputs/testbed_runs")
    config_dir: Path = Path("configs/testbed_runs")
    config_name: str = "testbed_config.json"
    rider_params: Dict[str, float | int] = field(
        default_factory=lambda: dict(DEFAULT_RIDER_PARAMS)
    )
    driver_params: Optional[Dict[str, float | int]] = field(
        default_factory=lambda: dict(DEFAULT_DRIVER_PARAMS)
    )
    core_params: Dict[str, float] = field(
        default_factory=lambda: {k: float(v) for k, v in CORE_PARAM_DEFAULTS.items()}
    )
    image_subcharge_count: int = 12
    use_image_weighting: bool = True

    # Self-consistency options
    self_consistency_enabled: bool = True
    self_consistency_tolerance: float = 1e-6
    self_consistency_max_iterations: int = 5
    self_consistency_debug: bool = False

    # Energy monitoring options
    energy_monitor_enabled: bool = True
    energy_monitor_threshold: float = 2.0
    energy_monitor_check_interval: int = 10
    energy_monitor_halt_on_jump: bool = False
    energy_monitor_debug: bool = False

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "steps": self.steps,
            "seed": self.seed,
            "simulation_type": self.simulation_type.name,
            "legacy_enabled": self.legacy_enabled,
            "overlay_display": self.overlay_display,
            "overlay_save": self.overlay_save,
            "difference_display": self.difference_display,
            "difference_save": self.difference_save,
            "metrics_save": self.metrics_save,
            "energy_display": self.energy_display,
            "energy_save": self.energy_save,
            "transverse_display": self.transverse_display,
            "transverse_save": self.transverse_save,
            "trajectory_save": self.trajectory_save,
            "trajectory_interval": self.trajectory_interval,
            "plot_dpi": self.plot_dpi,
            "output_dir": str(self.output_dir),
            "config_dir": str(self.config_dir),
            "config_name": self.config_name,
            "rider_params": dict(self.rider_params),
            "driver_params": dict(self.driver_params) if self.driver_params else None,
            "core_params": dict(self.core_params),
            "image_subcharge_count": self.image_subcharge_count,
            "use_image_weighting": self.use_image_weighting,
            "self_consistency_enabled": self.self_consistency_enabled,
            "self_consistency_tolerance": self.self_consistency_tolerance,
            "self_consistency_max_iterations": self.self_consistency_max_iterations,
            "self_consistency_debug": self.self_consistency_debug,
            "energy_monitor_enabled": self.energy_monitor_enabled,
            "energy_monitor_threshold": self.energy_monitor_threshold,
            "energy_monitor_check_interval": self.energy_monitor_check_interval,
            "energy_monitor_halt_on_jump": self.energy_monitor_halt_on_jump,
            "energy_monitor_debug": self.energy_monitor_debug,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SimulationOptions":
        def _bool(name: str, default: bool) -> bool:
            return bool(payload.get(name, default))

        def _int(name: str, default: int) -> int:
            value = payload.get(name, default)
            try:
                return int(value)  # type: ignore[arg-type,no-any-return,call-overload]
            except (TypeError, ValueError):
                return default

        def _float(name: str, default: float) -> float:
            value = payload.get(name, default)
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        sim_value = payload.get("simulation_type", "BUNCH_TO_BUNCH")
        if isinstance(sim_value, SimulationType):
            simulation_type = sim_value
        elif isinstance(sim_value, int):
            simulation_type = SimulationType(sim_value)
        elif isinstance(sim_value, str) and hasattr(SimulationType, sim_value):
            simulation_type = getattr(SimulationType, sim_value)
        else:
            simulation_type = SimulationType.BUNCH_TO_BUNCH

        rider_params = dict(DEFAULT_RIDER_PARAMS)
        rider_payload = payload.get("rider_params")
        if isinstance(rider_payload, dict):
            rider_params.update(rider_payload)

        driver_params: Optional[Dict[str, float | int]]
        driver_payload = payload.get("driver_params")
        if isinstance(driver_payload, dict):
            driver_params = dict(driver_payload)
        else:
            driver_params = dict(DEFAULT_DRIVER_PARAMS)

        core_params = {k: float(v) for k, v in CORE_PARAM_DEFAULTS.items()}
        core_payload = payload.get("core_params")
        if isinstance(core_payload, dict):
            for key, value in core_payload.items():
                try:
                    core_params[key] = float(value)
                except (TypeError, ValueError):
                    continue

        options = cls(
            steps=_int("steps", 1000),
            seed=_int("seed", 12345),
            simulation_type=simulation_type,
            legacy_enabled=_bool("legacy_enabled", False),
            overlay_display=_bool("overlay_display", False),
            overlay_save=_bool("overlay_save", False),
            difference_display=_bool("difference_display", False),
            difference_save=_bool("difference_save", False),
            metrics_save=_bool("metrics_save", False),
            energy_display=_bool("energy_display", True),
            energy_save=_bool("energy_save", True),
            transverse_display=_bool("transverse_display", False),
            transverse_save=_bool("transverse_save", False),
            trajectory_save=_bool("trajectory_save", False),
            trajectory_interval=_int("trajectory_interval", 10),
            plot_dpi=_int("plot_dpi", DEFAULT_PLOT_DPI),
            output_dir=Path(
                str(payload.get("output_dir", "test_outputs/testbed_runs"))
            ),
            config_dir=Path(str(payload.get("config_dir", "configs/testbed_runs"))),
            config_name=str(payload.get("config_name", "testbed_config.json")),
            rider_params=rider_params,
            driver_params=driver_params,
            core_params=core_params,
            image_subcharge_count=_int("image_subcharge_count", 12),
            use_image_weighting=_bool("use_image_weighting", True),
        )
        return options


@dataclass
class InitialSummary:
    seed: int
    rider_gamma: float
    rider_rest_mev: float
    rider_rest_gev: float
    rider_total_gev: float
    driver_gamma: Optional[float]
    driver_rest_mev: Optional[float]
    driver_rest_gev: Optional[float]
    driver_total_gev: Optional[float]
    supports_driver: bool

    @property
    def has_driver(self) -> bool:
        return self.supports_driver and self.driver_gamma is not None


@dataclass
class RunResult:
    metrics: Optional[Dict[str, Dict[str, float]]]
    saved_paths: Dict[str, Path]
    figures: Dict[str, plt.Figure]
    logs: List[str]
    duration_s: float
    filename_base: str


# ---------------------------------------------------------------------------
# Simple helpers used by both the runner and GUI
# ---------------------------------------------------------------------------


def supports_driver(simulation_type: SimulationType) -> bool:
    return simulation_type is SimulationType.BUNCH_TO_BUNCH


def apply_species_preset(params: Dict[str, float | int], preset_key: str) -> None:
    preset = SPECIES_PRESETS.get(preset_key)
    if not preset:
        return
    for param_name, value in preset.items():
        if param_name in params:
            current = params[param_name]
            params[param_name] = type(current)(value)


def list_config_files(directory: Path) -> List[str]:
    if not directory.exists():
        return []
    return sorted(str(path.name) for path in directory.glob("*.json") if path.is_file())


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_filename_base(config_name: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base = config_name.strip().replace(".json", "") or "testbed_config"
    return f"{base}_{timestamp}"


def _extract_scalar_series(
    states: Iterable[Dict[str, np.ndarray]], key: str
) -> np.ndarray:
    values: List[float] = []
    for state in states:
        data = state.get(key)
        if data is None or len(data) == 0:
            values.append(0.0)
        else:
            values.append(float(np.asarray(data)[0]))
    return np.asarray(values, dtype=float)


def _extract_vector_series(
    states: Iterable[Dict[str, np.ndarray]], keys: Tuple[str, ...]
) -> np.ndarray:
    components = [_extract_scalar_series(states, key) for key in keys]
    return np.stack(components, axis=-1)


def compute_initial_summary(options: SimulationOptions) -> InitialSummary:
    sim_type = options.simulation_type
    driver_allowed = supports_driver(sim_type)
    rider_params = dict(options.rider_params)
    driver_params = dict(options.driver_params) if options.driver_params else None
    if not driver_allowed:
        driver_params = None

    rider_state, driver_state, rider_rest_mev, driver_rest_mev = (
        prepare_two_particle_demo(
            seed=options.seed,
            rider_params=rider_params,
            driver_params=driver_params,
        )
    )

    rider_gamma = float(rider_state["gamma"][0])
    rider_rest_gev = rider_rest_mev * 1e-3
    rider_total_gev = rider_gamma * rider_rest_gev

    if driver_allowed and driver_state is not None:
        driver_gamma = float(driver_state["gamma"][0])
        driver_rest_gev = driver_rest_mev * 1e-3
        driver_total_gev = driver_gamma * driver_rest_gev
    else:
        driver_gamma = None
        driver_rest_mev = None
        driver_rest_gev = None
        driver_total_gev = None

    return InitialSummary(
        seed=options.seed,
        rider_gamma=rider_gamma,
        rider_rest_mev=rider_rest_mev,
        rider_rest_gev=rider_rest_gev,
        rider_total_gev=rider_total_gev,
        driver_gamma=driver_gamma,
        driver_rest_mev=driver_rest_mev,
        driver_rest_gev=driver_rest_gev,
        driver_total_gev=driver_total_gev,
        supports_driver=driver_allowed,
    )


# ---------------------------------------------------------------------------
# Core runner mirroring the widget logic
# ---------------------------------------------------------------------------


def run_testbed(
    options: SimulationOptions,
    *,
    log: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> RunResult:
    """Execute the integrator benchmark with plotting/export side effects.

    Parameters
    ----------
    options:
        Simulation configuration options.
    log:
        Optional callback for logging messages.
    progress_callback:
        Optional callback invoked as progress_callback(current, total) to report progress.
    cancel_callback:
        Optional predicate that returns True if cancellation is requested.
    """

    start = time.perf_counter()
    logs: List[str] = []

    def _log(message: str) -> None:
        logs.append(message)
        if log is not None:
            log(message)

    sim_type = options.simulation_type
    driver_allowed = supports_driver(sim_type)

    rider_params = dict(options.rider_params)
    driver_params = (
        dict(options.driver_params)
        if options.driver_params and driver_allowed
        else None
    )

    core_params = {
        k: float(options.core_params.get(k, CORE_PARAM_DEFAULTS[k]))
        for k in CORE_PARAM_DEFAULTS
    }
    required_params = CORE_REQUIRED_PARAMS.get(sim_type, set())
    filtered_core_params = {name: core_params[name] for name in required_params}

    legacy_enabled = bool(options.legacy_enabled)
    if not legacy_enabled:
        overlay_display = False
        overlay_save = False
        difference_display = False
        difference_save = False
        metrics_save = False
    else:
        overlay_display = bool(options.overlay_display)
        overlay_save = bool(options.overlay_save)
        difference_display = bool(options.difference_display)
        difference_save = bool(options.difference_save)
        metrics_save = bool(options.metrics_save)

    energy_display = bool(options.energy_display)
    energy_save = bool(options.energy_save)
    transverse_display = bool(options.transverse_display)
    transverse_save = bool(options.transverse_save)
    trajectory_save = bool(options.trajectory_save)

    should_save = any(
        [
            overlay_save,
            difference_save,
            metrics_save,
            energy_save,
            transverse_save,
            trajectory_save,
        ]
    )

    output_dir = Path(options.output_dir).expanduser()
    if should_save:
        ensure_directory(output_dir)

    filename_base = generate_filename_base(options.config_name)
    timestamp_token = filename_base[-_TIMESTAMP_TOKEN_LENGTH:]
    if len(filename_base) > (_TIMESTAMP_TOKEN_LENGTH + 1):
        config_label = filename_base[: -(_TIMESTAMP_TOKEN_LENGTH + 1)]
    else:
        config_label = filename_base

    _log(
        f"Running {sim_type.name.replace('_', ' ').title()} integrator for {options.steps} steps"
    )
    _log(f"  Steps: {options.steps}")
    _log(f"  Seed: {options.seed}")
    _log(f"  Core params: {filtered_core_params}")
    _log(f"  Legacy enabled: {legacy_enabled}")
    _log(f"  Image subcharges: {options.image_subcharge_count}")
    _log(f"  Image weighting: {options.use_image_weighting}")
    _log(
        f"  Self-consistency: {options.self_consistency_enabled} (tol={options.self_consistency_tolerance:.1e}, max_iter={options.self_consistency_max_iterations})"
    )
    _log(
        f"  Energy monitoring: {options.energy_monitor_enabled} (threshold={options.energy_monitor_threshold * 100:.0f}%, halt={options.energy_monitor_halt_on_jump})"
    )
    _log("")

    return_traj_flag = any(
        [
            trajectory_save,
            transverse_display,
            transverse_save,
            energy_save,
            energy_display,
            overlay_display,
            overlay_save,
        ]
    )

    result = run_benchmark(
        steps=options.steps,
        simulation_type=sim_type,
        rider_params=rider_params,
        driver_params=driver_params,
        seed=options.seed,
        legacy_enabled=legacy_enabled,
        return_trajectories=return_traj_flag,
        image_subcharge_count=int(options.image_subcharge_count),
        use_image_weighting=bool(options.use_image_weighting),
        self_consistency_enabled=options.self_consistency_enabled,
        self_consistency_tolerance=options.self_consistency_tolerance,
        self_consistency_max_iterations=options.self_consistency_max_iterations,
        self_consistency_debug=options.self_consistency_debug,
        energy_monitor_enabled=options.energy_monitor_enabled,
        energy_monitor_threshold=options.energy_monitor_threshold,
        energy_monitor_check_interval=options.energy_monitor_check_interval,
        energy_monitor_halt_on_jump=options.energy_monitor_halt_on_jump,
        energy_monitor_debug=options.energy_monitor_debug,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        **filtered_core_params,
    )

    if isinstance(result, tuple) and len(result) == 2:
        metrics, payload = result
    else:
        metrics = result
        payload = {}

    saved_paths: Dict[str, Path] = {}
    figures: Dict[str, plt.Figure] = {}

    if legacy_enabled and metrics_save and metrics is not None:
        metrics_path = output_dir / f"{filename_base}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, default=str)
        saved_paths["metrics"] = metrics_path
        _log(f"Saved metrics to: {metrics_path}")

    core_traj = payload.get("core")
    legacy_traj = payload.get("legacy") if legacy_enabled else None
    initial_states = payload.get("initial_states", {})
    rest_energies = payload.get("rest_energy_mev", {})

    if core_traj:
        rider_states = core_traj.get("rider", [])
        driver_states = core_traj.get("driver") if driver_allowed else None

        try:
            rider_initial = initial_states.get("rider")
            rider_rest_mev = rest_energies.get("rider")
            rider_delta_e, rider_z = compute_delta_energy_series(
                rider_states,
                rider_initial,
                rider_rest_mev,
            )
            rider_z_rel = rider_z - rider_z[0]
        except Exception as exc:  # pragma: no cover - defensive guard
            _log(f"Failed to compute rider energy series: {exc}")
            _log(
                f"Traceback:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
            )
            rider_delta_e = None
            rider_z_rel = None

        if driver_allowed and driver_states is not None:
            try:
                driver_initial = initial_states.get("driver")
                driver_rest_mev = rest_energies.get("driver")
                driver_delta_e, driver_z = compute_delta_energy_series(
                    driver_states,
                    driver_initial,
                    driver_rest_mev,
                )
                driver_z_rel = driver_z - driver_z[0]
            except Exception as exc:  # pragma: no cover - defensive guard
                _log(f"Failed to compute driver energy series: {exc}")
                _log(
                    f"Traceback:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
                )
                driver_delta_e = None
                driver_z_rel = None
        else:
            driver_delta_e = None
            driver_z_rel = None

        if legacy_traj and legacy_enabled:
            legacy_rider_states = legacy_traj.get("rider", [])
            try:
                legacy_rider_delta_e, legacy_rider_z = compute_delta_energy_series(
                    legacy_rider_states,
                    initial_states.get("rider"),
                    rest_energies.get("rider"),
                )
                legacy_rider_z_rel = legacy_rider_z - legacy_rider_z[0]
            except Exception as exc:  # pragma: no cover - defensive guard
                _log(f"Failed to compute legacy rider energy series: {exc}")
                _log(
                    f"Traceback:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
                )
                legacy_rider_delta_e = None
                legacy_rider_z_rel = None

            if driver_allowed and legacy_traj.get("driver") is not None:
                try:
                    legacy_driver_delta_e, legacy_driver_z = (
                        compute_delta_energy_series(
                            legacy_traj["driver"],
                            initial_states.get("driver"),
                            rest_energies.get("driver"),
                        )
                    )
                    legacy_driver_z_rel = legacy_driver_z - legacy_driver_z[0]
                except Exception as exc:
                    _log(f"Failed to compute legacy driver energy series: {exc}")  # type: ignore[assignment]
                    _log(
                        f"Traceback:\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
                    )
                    legacy_driver_delta_e = None
                    legacy_driver_z_rel = None
            else:
                legacy_driver_delta_e = None
                legacy_driver_z_rel = None
        else:
            legacy_rider_delta_e = None
            legacy_rider_z_rel = None
            legacy_driver_delta_e = None
            legacy_driver_z_rel = None

        if (
            (energy_save or energy_display)
            and rider_delta_e is not None
            and rider_z_rel is not None
        ):
            fig_energy, axes_energy = plt.subplots(
                1,
                2 if driver_allowed and driver_delta_e is not None else 1,
                figsize=(16 if driver_delta_e is not None else 8, 6),
                dpi=options.plot_dpi,
            )
            if not isinstance(axes_energy, np.ndarray):
                axes = [axes_energy]
            else:
                axes = list(axes_energy)

            show_legend = legacy_enabled

            axes[0].scatter(
                rider_z_rel,
                rider_delta_e,
                color=COLOR_RIDER,
                label="Core" if show_legend else None,
                **SCATTER_STYLE,
            )
            if legacy_rider_delta_e is not None and legacy_rider_z_rel is not None:
                axes[0].scatter(
                    legacy_rider_z_rel,
                    legacy_rider_delta_e,
                    color=COLOR_LEGACY_RIDER,
                    label="Legacy",
                    **SCATTER_STYLE,
                )
            axes[0].set_xlabel("Delta z (mm)")
            axes[0].set_ylabel("Delta E (GeV)")
            axes[0].set_title("Rider Delta E vs Delta z")
            axes[0].grid(True, alpha=0.3)
            if show_legend:
                axes[0].legend()

            if (
                driver_delta_e is not None
                and driver_z_rel is not None
                and len(axes) > 1
            ):
                axes[1].scatter(
                    driver_z_rel,
                    driver_delta_e,
                    color=COLOR_DRIVER,
                    label="Core" if legacy_enabled else None,
                    **SCATTER_STYLE,
                )
                if (
                    legacy_driver_delta_e is not None
                    and legacy_driver_z_rel is not None
                ):
                    axes[1].scatter(
                        legacy_driver_z_rel,
                        legacy_driver_delta_e,
                        color=COLOR_LEGACY_DRIVER,
                        label="Legacy",
                        **SCATTER_STYLE,
                    )
                axes[1].set_xlabel("Delta z (mm)")
                axes[1].set_ylabel("Delta E (GeV)")
                axes[1].set_title("Driver Delta E vs Delta z")
                axes[1].grid(True, alpha=0.3)
                if legacy_enabled:
                    axes[1].legend()

            fig_energy.tight_layout()
            if energy_save and should_save:
                energy_path = output_dir / f"{filename_base}_energy.png"
                fig_energy.savefig(energy_path)
                saved_paths["energy"] = energy_path
                _log(f"Saved energy plot to: {energy_path}")
            if energy_display:
                figures["energy"] = fig_energy
            else:
                plt.close(fig_energy)

        if (
            legacy_enabled
            and (overlay_display or overlay_save)
            and rider_delta_e is not None
            and rider_z_rel is not None
            and legacy_rider_delta_e is not None
            and legacy_rider_z_rel is not None
        ):
            fig_overlay, axes_overlay = plt.subplots(
                1,
                (
                    2
                    if driver_delta_e is not None and legacy_driver_delta_e is not None
                    else 1
                ),
                figsize=(
                    (
                        16
                        if driver_delta_e is not None
                        and legacy_driver_delta_e is not None
                        else 8
                    ),
                    6,
                ),
                dpi=options.plot_dpi,
            )
            if not isinstance(axes_overlay, np.ndarray):
                axes = [axes_overlay]
            else:
                axes = list(axes_overlay)

            axes[0].plot(
                rider_z_rel,
                rider_delta_e,
                color=COLOR_RIDER,
                label="Core",
                linewidth=2.0,
            )
            axes[0].plot(
                legacy_rider_z_rel,
                legacy_rider_delta_e,
                color=COLOR_LEGACY_RIDER,
                label="Legacy",
                linewidth=2.0,
                linestyle="--",
            )
            axes[0].set_xlabel("Delta z (mm)")
            axes[0].set_ylabel("Delta E (GeV)")
            axes[0].set_title("Rider Delta E Comparison")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            if (
                driver_delta_e is not None
                and driver_z_rel is not None
                and legacy_driver_delta_e is not None
                and legacy_driver_z_rel is not None
                and len(axes) > 1
            ):
                axes[1].plot(
                    driver_z_rel,
                    driver_delta_e,
                    color=COLOR_DRIVER,
                    label="Core",
                    linewidth=2.0,
                )
                axes[1].plot(
                    legacy_driver_z_rel,
                    legacy_driver_delta_e,
                    color=COLOR_LEGACY_DRIVER,
                    label="Legacy",
                    linewidth=2.0,
                    linestyle="--",
                )
                axes[1].set_xlabel("Delta z (mm)")
                axes[1].set_ylabel("Delta E (GeV)")
                axes[1].set_title("Driver Delta E Comparison")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            fig_overlay.tight_layout()
            if overlay_save and should_save:
                overlay_path = output_dir / f"{filename_base}_overlay.png"
                fig_overlay.savefig(overlay_path)
                saved_paths["overlay"] = overlay_path
                _log(f"Saved overlay plot to: {overlay_path}")
            if overlay_display:
                figures["overlay"] = fig_overlay
            else:
                plt.close(fig_overlay)

        core_r_hist = np.array(
            [[s["t"][0], s["x"][0], s["y"][0], s["z"][0]] for s in rider_states]
        )
        core_r_gamma = _extract_scalar_series(rider_states, "gamma")
        core_r_momentum = _extract_vector_series(rider_states, ("Px", "Py", "Pz"))
        core_r_beta = _extract_vector_series(rider_states, ("bx", "by", "bz"))
        core_r_betadot = _extract_vector_series(
            rider_states, ("bdotx", "bdoty", "bdotz")
        )
        core_r_pt = _extract_scalar_series(rider_states, "Pt")
        plot_times_ns = core_r_hist[:, 0]

        if driver_allowed and driver_states is not None:
            core_d_hist = np.array(
                [[s["t"][0], s["x"][0], s["y"][0], s["z"][0]] for s in driver_states]
            )
            core_d_gamma = _extract_scalar_series(driver_states, "gamma")
            core_d_momentum = _extract_vector_series(driver_states, ("Px", "Py", "Pz"))
            core_d_beta = _extract_vector_series(driver_states, ("bx", "by", "bz"))
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
            legacy_r_hist = np.array(
                [
                    [s["t"][0], s["x"][0], s["y"][0], s["z"][0]]
                    for s in legacy_traj.get("rider", [])
                ]
            )
            legacy_r_momentum = _extract_vector_series(
                legacy_traj.get("rider", []), ("Px", "Py", "Pz")
            )
            legacy_r_beta = _extract_vector_series(
                legacy_traj.get("rider", []), ("bx", "by", "bz")
            )
            legacy_r_betadot = _extract_vector_series(
                legacy_traj.get("rider", []), ("bdotx", "bdoty", "bdotz")
            )
            legacy_r_pt = _extract_scalar_series(legacy_traj.get("rider", []), "Pt")
            if driver_allowed and legacy_traj.get("driver") is not None:
                legacy_d_hist = np.array(
                    [
                        [s["t"][0], s["x"][0], s["y"][0], s["z"][0]]
                        for s in legacy_traj.get("driver", [])
                    ]
                )
                legacy_d_momentum = _extract_vector_series(
                    legacy_traj.get("driver", []), ("Px", "Py", "Pz")
                )
                legacy_d_beta = _extract_vector_series(
                    legacy_traj.get("driver", []), ("bx", "by", "bz")
                )
                legacy_d_betadot = _extract_vector_series(
                    legacy_traj.get("driver", []), ("bdotx", "bdoty", "bdotz")
                )
                legacy_d_pt = _extract_scalar_series(
                    legacy_traj.get("driver", []), "Pt"
                )
            else:
                legacy_d_hist = None
                legacy_d_momentum = None
                legacy_d_beta = None
                legacy_d_betadot = None
                legacy_d_pt = None
        else:
            legacy_r_hist = None
            legacy_r_momentum = None
            legacy_r_beta = None
            legacy_r_betadot = None
            legacy_r_pt = None
            legacy_d_hist = None
            legacy_d_momentum = None
            legacy_d_beta = None
            legacy_d_betadot = None
            legacy_d_pt = None

        if (
            legacy_enabled
            and (difference_save or difference_display)
            and legacy_r_hist is not None
        ):
            fig_diff, axes_diff = plt.subplots(
                1,
                (
                    2
                    if driver_allowed
                    and core_d_hist is not None
                    and legacy_d_hist is not None
                    else 1
                ),
                figsize=(
                    (
                        16
                        if driver_allowed
                        and core_d_hist is not None
                        and legacy_d_hist is not None
                        else 8
                    ),
                    6,
                ),
                dpi=options.plot_dpi,
            )
            if not isinstance(axes_diff, np.ndarray):
                axes = [axes_diff]
            else:
                axes = list(axes_diff)

            r_delta_x = (core_r_hist[:, 1] - legacy_r_hist[:, 1]) * 1e3
            r_delta_y = (core_r_hist[:, 2] - legacy_r_hist[:, 2]) * 1e3
            r_delta_z = (core_r_hist[:, 3] - legacy_r_hist[:, 3]) * 1e3
            axes[0].plot(
                plot_times_ns, r_delta_x, label="Delta x (mm)", color=COLOR_DIFF_RIDER
            )
            axes[0].plot(
                plot_times_ns,
                r_delta_y,
                label="Delta y (mm)",
                color=COLOR_DIFF_RIDER,
                linestyle="--",
            )
            axes[0].plot(
                plot_times_ns,
                r_delta_z,
                label="Delta z (mm)",
                color=COLOR_DIFF_RIDER,
                linestyle=":",
            )
            axes[0].set_xlabel("Time (ns)")
            axes[0].set_ylabel("Delta position (mm)")
            axes[0].set_title("Rider Delta (core - legacy)")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            if (
                driver_allowed
                and core_d_hist is not None
                and legacy_d_hist is not None
                and len(axes) > 1
            ):
                d_delta_x = (core_d_hist[:, 1] - legacy_d_hist[:, 1]) * 1e3
                d_delta_y = (core_d_hist[:, 2] - legacy_d_hist[:, 2]) * 1e3
                d_delta_z = (core_d_hist[:, 3] - legacy_d_hist[:, 3]) * 1e3
                axes[1].plot(
                    plot_times_ns,
                    d_delta_x,
                    label="Delta x (mm)",
                    color=COLOR_DIFF_DRIVER,
                )
                axes[1].plot(
                    plot_times_ns,
                    d_delta_y,
                    label="Delta y (mm)",
                    color=COLOR_DIFF_DRIVER,
                    linestyle="--",
                )
                axes[1].plot(
                    plot_times_ns,
                    d_delta_z,
                    label="Delta z (mm)",
                    color=COLOR_DIFF_DRIVER,
                    linestyle=":",
                )
                axes[1].set_xlabel("Time (ns)")
                axes[1].set_ylabel("Delta position (mm)")
                axes[1].set_title("Driver Delta (core - legacy)")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            fig_diff.tight_layout()
            if difference_save and should_save:
                diff_path = output_dir / f"{filename_base}_difference.png"
                fig_diff.savefig(diff_path)
                saved_paths["difference"] = diff_path
                _log(f"Saved difference plot to: {diff_path}")
            if difference_display:
                figures["difference"] = fig_diff
            else:
                plt.close(fig_diff)

        if transverse_display or transverse_save:
            fig_transverse, (ax_x, ax_y) = plt.subplots(
                1, 2, figsize=(16, 6), dpi=options.plot_dpi
            )
            ax_x.plot(
                plot_times_ns,
                core_r_hist[:, 1] * 1e3,
                color=COLOR_RIDER,
                label="Rider (Core)",
            )
            ax_y.plot(
                plot_times_ns,
                core_r_hist[:, 2] * 1e3,
                color=COLOR_RIDER,
                label="Rider (Core)",
            )
            if driver_allowed and core_d_hist is not None:
                ax_x.plot(
                    plot_times_ns,
                    core_d_hist[:, 1] * 1e3,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
                ax_y.plot(
                    plot_times_ns,
                    core_d_hist[:, 2] * 1e3,
                    color=COLOR_DRIVER,
                    label="Driver (Core)",
                )
            if legacy_enabled and legacy_r_hist is not None:
                ax_x.plot(
                    plot_times_ns,
                    legacy_r_hist[:, 1] * 1e3,
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
                ax_y.plot(
                    plot_times_ns,
                    legacy_r_hist[:, 2] * 1e3,
                    color=COLOR_LEGACY_RIDER,
                    linestyle="--",
                    label="Rider (Legacy)",
                )
                if driver_allowed and legacy_d_hist is not None:
                    ax_x.plot(
                        plot_times_ns,
                        legacy_d_hist[:, 1] * 1e3,
                        color=COLOR_LEGACY_DRIVER,
                        linestyle="--",
                        label="Driver (Legacy)",
                    )
                    ax_y.plot(
                        plot_times_ns,
                        legacy_d_hist[:, 2] * 1e3,
                        color=COLOR_LEGACY_DRIVER,
                        linestyle="--",
                        label="Driver (Legacy)",
                    )
            ax_x.set_xlabel("Time (ns)")
            ax_x.set_ylabel("Average x (mm)")
            ax_x.set_title("Average X Position")
            ax_x.legend()
            ax_x.grid(True, alpha=0.3)
            ax_y.set_xlabel("Time (ns)")
            ax_y.set_ylabel("Average y (mm)")
            ax_y.set_title("Average Y Position")
            ax_y.legend()
            ax_y.grid(True, alpha=0.3)
            fig_transverse.tight_layout()
            if transverse_save and should_save:
                transverse_path = output_dir / f"{filename_base}_transverse.png"
                fig_transverse.savefig(transverse_path)
                saved_paths["transverse"] = transverse_path
                _log(f"Saved transverse plot to: {transverse_path}")
            if transverse_display:
                figures["transverse"] = fig_transverse
            else:
                plt.close(fig_transverse)

        if trajectory_save and should_save:
            interval = max(1, int(options.trajectory_interval))

            def _build_particle_payload(
                hist: np.ndarray,
                gamma: np.ndarray,
                momentum: np.ndarray,
                beta: np.ndarray,
                betadot: np.ndarray,
                pt: np.ndarray,
            ) -> Dict[str, object]:
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

            core_payload: Dict[str, object] = {
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
                driver_allowed
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

            traj_data: Dict[str, object] = {
                "config_name": options.config_name,
                "config_label": config_label,
                "seed": options.seed,
                "num_steps": options.steps,
                "simulation_type": sim_type.name,
                "step_interval": interval,
                "timestamp": timestamp_token,
                "core": core_payload,
                "image_subcharge_count": options.image_subcharge_count,
            }

            if (
                legacy_enabled
                and legacy_r_hist is not None
                and legacy_r_momentum is not None
            ):
                legacy_payload: Dict[str, object] = {
                    "rider": _build_particle_payload(
                        legacy_r_hist,
                        _extract_scalar_series(legacy_traj.get("rider", []), "gamma"),
                        legacy_r_momentum,
                        legacy_r_beta,
                        legacy_r_betadot,
                        legacy_r_pt,
                    )
                }
                if (
                    driver_allowed
                    and legacy_d_hist is not None
                    and legacy_d_momentum is not None
                ):
                    legacy_payload["driver"] = _build_particle_payload(
                        legacy_d_hist,
                        _extract_scalar_series(legacy_traj.get("driver", []), "gamma"),
                        legacy_d_momentum,
                        legacy_d_beta,
                        legacy_d_betadot,
                        legacy_d_pt,
                    )
                traj_data["legacy"] = legacy_payload

            label_prefix = config_label if config_label else "trajectory"
            traj_path = (
                output_dir / f"{label_prefix}_trajectory_data_{timestamp_token}.json"
            )
            with traj_path.open("w", encoding="utf-8") as handle:
                json.dump(traj_data, handle, indent=2)
            saved_paths["trajectory"] = traj_path
            _log(f"Saved trajectories to: {traj_path} (interval={interval})")

    duration = time.perf_counter() - start
    _log("")
    _log("Run complete")

    return RunResult(
        metrics=metrics,
        saved_paths=saved_paths,
        figures=figures,
        logs=logs,
        duration_s=duration,
        filename_base=filename_base,
    )


def save_config(options: SimulationOptions, path: Path) -> None:
    ensure_directory(path.parent)
    payload = options.to_dict()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_config(path: Path) -> SimulationOptions:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a JSON object")
    return SimulationOptions.from_dict(payload)


__all__ = [
    "AVAILABLE_DPI_CHOICES",
    "CORE_PARAM_DEFAULTS",
    "CORE_PARAM_LABELS",
    "CORE_REQUIRED_PARAMS",
    "PARAM_LABELS",
    "PARTICLE_PARAM_FIELDS",
    "SimulationOptions",
    "InitialSummary",
    "RunResult",
    "SPECIES_OPTIONS",
    "SPECIES_PRESETS",
    "apply_species_preset",
    "compute_initial_summary",
    "ensure_directory",
    "generate_filename_base",
    "list_config_files",
    "load_config",
    "run_testbed",
    "save_config",
    "supports_driver",
]
