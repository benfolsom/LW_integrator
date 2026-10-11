"""The saved-option boundary shared by run, sweep, and GUI adapters.

Defaults come from dataclass fields. File policies retain their historical
precedence and coercion; constructor alias merging remains a separate policy.
Legacy keys are accepted and emitted with their existing meanings, including
inert self-consistency knobs. No numerical setting is migrated implicitly.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, dataclass, fields
from enum import Enum
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any, Optional, Tuple, TypeVar

from core.self_consistency import canonicalize_self_consistency_mode
from core.types import SimulationType
from optimization.mode_helpers import normalize_sweep_or_optimization_mode


@dataclass
class SharedOptions:
    """Identical run/sweep defaults, declared once; adapters keep their overrides."""

    seed: int = 12345
    image_subcharge_count: int = 12
    use_image_weighting: bool = True
    macroparticle_enabled: bool = False
    macroparticle_charge_multiplier: float = 1.0
    macroparticle_sigma_multiplier: float = 1.0
    macroparticle_use_momentum_errors: bool = True
    macroparticle_smearing_enabled: bool = False
    macroparticle_smearing_subcharge_count: int = 8
    macroparticle_smearing_sigma_multiplier: float = 1.0
    macroparticle_smearing_position_sigma_mm: Optional[float] = None
    macroparticle_smearing_longitudinal_sigma_mm: Optional[float] = None
    macroparticle_smearing_momentum_sigma_amu_mm_ns: Optional[float] = None
    macroparticle_smearing_use_position_errors: bool = True
    macroparticle_smearing_use_momentum_errors: bool = True
    macroparticle_smearing_use_centroid_errors: bool = True
    macroparticle_smearing_use_internal_cloud: bool = True
    macroparticle_smearing_apply_to_active_observers: bool = True
    macroparticle_smearing_apply_to_active_sources: bool = True
    macroparticle_smearing_apply_to_passive_sources: bool = True
    macroparticle_smearing_apply_to_passive_updates: bool = False
    macroparticle_smearing_seed: int = 12345
    macroparticle_smearing_refresh_policy: str = "fixed_per_particle"
    self_consistency_enabled: bool = True
    self_consistency_tolerance: float = 0.0001
    self_consistency_convergence_mode: str = "fixed_geometry"
    self_consistency_target_ms_tolerance: float = 1e-06
    self_consistency_max_iterations: int = 2
    self_consistency_mass_shell_tolerance: float = 0.01
    self_consistency_mass_shell_relaxation: float = 0.7
    self_consistency_verbosity: int = 0
    chrono_interpolate: bool = False
    chrono_tolerance: float = 0.001
    chrono_matching_mode: str = "FAST"
    chrono_high_precision: bool = False
    chrono_adaptive_tolerance: bool = False
    self_consistency_chrono_interpolate: bool = False
    self_consistency_chrono_tolerance: float = 0.001
    self_consistency_chrono_matching_mode: str = "FAST"
    self_consistency_chrono_high_precision: bool = False
    self_consistency_chrono_adaptive_tolerance: bool = False
    self_consistency_gamma_reconciliation_method: str = "DISABLED"
    self_consistency_gamma_reconciliation_low_beta_threshold: float = 0.9
    self_consistency_gamma_reconciliation_high_beta_threshold: float = 0.99
    self_consistency_gamma_reconciliation_low_beta_weight: float = 0.8
    self_consistency_gamma_reconciliation_high_beta_weight: float = 0.2
    self_consistency_gamma_reconciliation_mid_beta_weight: float = 0.5
    self_consistency_gamma_reconciliation_fixed_weight: float = 0.5
    energy_monitor_threshold: float = 2.0
    energy_monitor_check_interval: int = 10
    energy_monitor_halt_on_jump: bool = False
    energy_monitor_debug: bool = False
    adaptive_timestep_enabled: bool = True
    adaptive_timestep_threshold: float = 0.1
    adaptive_timestep_reduction_factor: int = 3
    adaptive_timestep_min_factor: float = 0.0001
    adaptive_timestep_cooldown_steps: int = 10
    adaptive_timestep_probe_threshold: float = 0.01
    adaptive_timestep_max_probe_steps: int = 3
    adaptive_timestep_debug: bool = False
    adaptive_timestep_bunch_proximity_enabled: bool = False
    adaptive_timestep_bunch_proximity_sigma_mm: float = 5.0
    adaptive_timestep_bunch_proximity_n_sigma: float = 5.0
    adaptive_timestep_bunch_proximity_reduction_factor: float = 10.0
    adaptive_timestep_bunch_proximity_transition_n_sigma: float = 2.0
    space_charge_enabled: bool = False
    space_charge_retarded: bool = True
    space_charge_softening_mm: float = 0.0
    space_charge_bunch_sigma_mm: float = 0.01
    space_charge_instantaneous_clock: str = "legacy_same_index"
    space_charge_min_retarded_steps: Optional[int] = None
    external_field_enabled: bool = False
    external_electric_field_native: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    external_electric_field_v_per_m: Optional[Tuple[float, float, float]] = None
    external_magnetic_field_native: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    external_field_x_min: Optional[float] = None
    external_field_x_max: Optional[float] = None
    external_field_y_min: Optional[float] = None
    external_field_y_max: Optional[float] = None
    external_field_z_min: Optional[float] = None
    external_field_z_max: Optional[float] = None
    external_field_t_min: Optional[float] = None
    external_field_t_max: Optional[float] = None
    source_history_representation: str = "light_cone_quintic"
    radiation_reaction_mode: str = "medina_lad"
    particle_loss_enabled: bool = True
    particle_loss_radius_mm: Optional[float] = 500.0
    particle_loss_conducting_wall_aperture_loss_enabled: bool = True
    particle_loss_initial_radial_quantile: Optional[float] = None
    particle_loss_initial_radial_multiplier: float = 1.0
    particle_loss_initial_radial_margin_mm: float = 0.0
    cavity_exit_enabled: bool = False
    cavity_exit_mode: str = "first_exit"
    cavity_exit_length_mm: Optional[float] = None
    cavity_exit_residual_tail_factor: float = 0.0
    cavity_exit_max_residual_tail_steps: int = 0
    pseudo_grid_enabled: bool = False
    pseudo_grid_active_rider_count: int = 4
    pseudo_grid_active_driver_count: int = 4
    pseudo_grid_passive_neighbor_count: int = 4
    pseudo_grid_coverage_strategy: str = "farthest_point_staleness"
    pseudo_grid_coverage_space: str = "position"
    pseudo_grid_pair_reuse_window: int = 16
    pseudo_grid_source_weighting_mode: str = "inverse_distance"
    pseudo_grid_loss_tracking_enabled: bool = True
    pseudo_grid_causal_history_pruning_enabled: bool = False
    pseudo_grid_causal_history_safety_margin_steps: int = 2
    driver_train_enabled: bool = False
    driver_train_bunch_count: int = 1
    driver_train_z_spacing_mm: float = 0.0
    driver_train_prehistory_steps: int = 0
    driver_train_preserve_prehistory_in_output: bool = False


OptionT = TypeVar("OptionT", bound=SharedOptions)


_NESTED_PREFIXES = {
    "checkpoint_": ("checkpoint",),
    "adaptive_pair_": ("adaptive_pair_return",),
    "macroparticle_smearing_": ("macroparticle_smearing",),
    "magnetic_dipole_source_": ("magnetic_dipole", "source"),
    "magnetic_dipole_": ("magnetic_dipole",),
    "particle_loss_": ("particle_loss",),
    "cavity_exit_": ("cavity_exit",),
    "beamline_geometry_": ("beamline_geometry",),
    "pseudo_grid_": ("pseudo_grid",),
    "driver_train_": ("driver_train",),
}

_WIRE_EXCEPTIONS = {
    "adaptive_pair_return_enabled": ("adaptive_pair_return", "enabled"),
    "rider_magnetic_species": ("magnetic_dipole", "rider", "species"),
    "rider_magnetic_moment_j_per_t": (
        "magnetic_dipole",
        "rider",
        "magnetic_moment_j_per_t",
    ),
    "rider_spin_quantum_number": ("magnetic_dipole", "rider", "spin_quantum_number"),
    "rider_rest_spin": ("magnetic_dipole", "rider", "rest_spin"),
    "rider_polarization": ("magnetic_dipole", "rider", "polarization"),
    "driver_magnetic_species": ("magnetic_dipole", "driver", "species"),
    "driver_magnetic_moment_j_per_t": (
        "magnetic_dipole",
        "driver",
        "magnetic_moment_j_per_t",
    ),
    "driver_spin_quantum_number": ("magnetic_dipole", "driver", "spin_quantum_number"),
    "driver_rest_spin": ("magnetic_dipole", "driver", "rest_spin"),
    "driver_polarization": ("magnetic_dipole", "driver", "polarization"),
    "particle_loss_radius_mm": ("particle_loss", "loss_radius_mm"),
    "cavity_exit_length_mm": ("cavity_exit", "cavity_length_mm"),
}

_STR_DEFAULT_ON_NONE = (
    "macroparticle_dynamics_mode",
    "chrono_matching_mode",
    "self_consistency_chrono_matching_mode",
    "source_history_representation",
    "radiation_reaction_mode",
    "cavity_exit_mode",
    "pseudo_grid_coverage_strategy",
    "pseudo_grid_coverage_space",
    "pseudo_grid_active_selection_mode",
    "pseudo_grid_passive_update_mode",
    "pseudo_grid_passive_remap_mode",
    "pseudo_grid_source_weighting_mode",
    "self_consistency_gamma_reconciliation_method",
)


def wire_path(name: str) -> tuple[str, ...]:
    """Return the established saved key, including historical irregular names."""
    if name in _WIRE_EXCEPTIONS:
        return _WIRE_EXCEPTIONS[name]
    for prefix, group in _NESTED_PREFIXES.items():
        if name.startswith(prefix):
            return group + (name[len(prefix) :],)
    return (name,)


def json_value(value: Any) -> Any:
    """Copy values into JSON-compatible containers without numerical coercion."""
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def serialize_run_options(options: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for option in fields(options):
        name = option.name
        source = (
            name.removeprefix("self_consistency_")
            if name.startswith("self_consistency_chrono_")
            else name
        )
        value = getattr(options, source)
        if name == "stochastic_emission":
            value = dict(value or {})
        target = payload
        path = wire_path(name)
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = json_value(value)
    return payload


def merge_constructor_chrono_aliases(options: Any) -> None:
    """Preserve direct-constructor OR/non-default merging, distinct from files."""
    for name, value in constructor_chrono_values(options).items():
        setattr(options, name, value)


def constructor_chrono_values(options: Any) -> dict[str, Any]:
    values = {}
    value: Any
    for suffix, default in CHRONO_DEFAULTS.items():
        canonical = f"chrono_{suffix}"
        alias = f"self_consistency_chrono_{suffix}"
        current = getattr(options, canonical, default)
        legacy = getattr(options, alias, default)
        if isinstance(default, bool):
            value = bool(current or legacy)
        else:
            value = legacy if current == default and legacy != default else current
        values[canonical] = values[alias] = value
    return values


CHRONO_DEFAULTS = {
    "interpolate": False,
    "tolerance": 1e-3,
    "matching_mode": "FAST",
    "high_precision": False,
    "adaptive_tolerance": False,
}


def nested_value(payload: dict, path: tuple[str, ...], default: Any) -> Any:
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def run_value(payload: dict, name: str, default: Any) -> Any:
    path = wire_path(name)
    if len(path) == 1:
        return payload.get(name, default)
    if path[0] == "checkpoint":
        return nested_value(payload, path, payload.get(name, default))
    if path[0] == "adaptive_pair_return":
        fallback = payload.get(name, default) if name.endswith("_enabled") else default
        return nested_value(payload, path, fallback)
    # The old run loader uses the nested leaf to spell flat compatibility keys.
    # In particular, these are rider_species and particle_loss_loss_radius_mm.
    prefix = {
        "rider_magnetic_species": "rider_",
        "driver_magnetic_species": "driver_",
        "particle_loss_radius_mm": "particle_loss_",
        "cavity_exit_length_mm": "cavity_exit_",
    }.get(name, name[: -len(path[-1])])
    flat = prefix + path[-1]
    return payload[flat] if flat in payload else nested_value(payload, path, default)


def coerce_scalar(value: Any, kind: type, default: Any, *, strict: bool = False) -> Any:
    if strict:
        return kind(value)
    try:
        return kind(value)
    except (TypeError, ValueError):
        return default


def strict_bool(value: Any, name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"{name} must be a JSON boolean (true/false), got {value!r} "
        f"({type(value).__name__})"
    )


def vector3(value: Any, default: Any) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return default
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return default


def parse_run_options(cls: type[OptionT], payload: dict[str, Any]) -> OptionT:
    """Parse saved run options while retaining legacy coercion and precedence."""
    sweep_keys = {"sweep_parameters", "energy_min", "energy_max", "energy_points"}
    if (
        "rider_params" not in payload
        and "driver_params" not in payload
        and (sweep_keys.intersection(payload))
    ):
        raise ValueError(
            "Flat sweep configuration cannot be loaded as a testbed config; "
            "use --sweep-config instead of --testbed-config. Testbed configs "
            "use nested rider_params, driver_params, and core_params."
        )
    # Preserve early backend conflict and checkpoint-path rejection.
    backend = resolve_exact_backend(payload)
    for name in ("checkpoint_directory", "checkpoint_resume_from"):
        value = run_value(payload, name, None)
        if value not in {None, ""}:
            Path(str(value))
    defaults = option_defaults(cls)
    # A saved run historically samples every ten steps; direct construction
    # samples every step. Keep this intentional adapter difference explicit.
    defaults["trajectory_interval"] = 10
    rfs = (
        bool(run_value(payload, "magnetic_dipole_enabled", False))
        and str(run_value(payload, "magnetic_dipole_spin_model", "rfs_minimal_2021"))
        .strip()
        .lower()
        == "rfs_minimal_2021"
    )
    defaults["radiation_reaction_mode"] = "off" if rfs else "medina_lad"
    defaults["adaptive_timestep_enabled"] = not rfs
    skip = {
        "simulation_type",
        "rider_params",
        "driver_params",
        "core_params",
        "beamline_geometry_occluders",
        "magnetic_dipole_source_local_jet_scales",
    }
    result = {}
    for item in fields(cls):
        name, kind = item.name, item.type
        if name in skip or name.startswith(("chrono_", "self_consistency_chrono_")):
            continue
        default = defaults[name]
        value = run_value(payload, name, default)
        strict = name.startswith(
            ("checkpoint_", "adaptive_pair_", "magnetic_dipole_")
        ) or name.endswith("_polarization")
        if kind == "bool":
            value = (
                strict_bool(value, name, default)
                if name == "self_consistency_enabled"
                else bool(value)
            )
        elif kind in {"int", "float"}:
            value = coerce_scalar(
                value, int if kind == "int" else float, default, strict=strict
            )
        elif kind == "str":
            value = (
                default
                if value is None and name in _STR_DEFAULT_ON_NONE
                else str(value)
            )
        elif kind == "Path":
            value = Path(str(value))
        elif kind == "Optional[Path]":
            value = Path(str(value)) if value not in {None, ""} else None
        elif kind == "Optional[float]":
            if name.startswith("adaptive_pair_") and value in {None, ""}:
                value = None
            value = (
                None
                if value is None
                else coerce_scalar(
                    value, float, default, strict=name.startswith("adaptive_pair_")
                )
            )
        elif kind == "Optional[int]":
            value = None if value is None else coerce_scalar(value, int, 0)
        elif kind == "Optional[str]":
            value = None if value is None else str(value)
        elif kind in {
            "Tuple[float, float, float]",
            "Optional[Tuple[float, float, float]]",
        }:
            value = vector3(value, default)
        elif name == "external_magnetic_field_gradient_t_per_m":
            rows = vector3_matrix(value)
            value = rows if rows is not None else default
        elif name == "driver_train_z_offsets_mm":
            try:
                value = (
                    tuple(float(v) for v in value)
                    if isinstance(value, (list, tuple))
                    else ()
                )
            except (TypeError, ValueError):
                value = ()
        if name == "particle_loss_enabled" and run_value(payload, name, None) is None:
            value = True
        result[name] = value
    for suffix, default in CHRONO_DEFAULTS.items():
        canonical, alias = f"chrono_{suffix}", f"self_consistency_chrono_{suffix}"
        # Evaluate both spellings, even if canonical wins: strict legacy booleans
        # historically reject strings in conflicting inputs as well.
        legacy = payload.get(alias, default)
        if isinstance(default, bool):
            legacy = strict_bool(legacy, alias, default)
            value = strict_bool(payload.get(canonical, legacy), canonical, legacy)
        elif isinstance(default, float):
            legacy = coerce_scalar(legacy, float, default)
            value = coerce_scalar(payload.get(canonical, legacy), float, legacy)
        else:
            legacy = str(legacy) if legacy is not None else default
            value = payload.get(canonical, legacy)
            value = str(value) if value is not None else legacy
        result[canonical] = result[alias] = value
    sim = payload.get("simulation_type", "BUNCH_TO_BUNCH")
    if isinstance(sim, SimulationType):
        result["simulation_type"] = sim
    elif isinstance(sim, int):
        result["simulation_type"] = SimulationType(sim)
    else:
        result["simulation_type"] = (
            getattr(SimulationType, sim, SimulationType.BUNCH_TO_BUNCH)
            if isinstance(sim, str)
            else SimulationType.BUNCH_TO_BUNCH
        )
    for name in ("rider_params", "driver_params"):
        result[name] = defaults[name]
        if isinstance(payload.get(name), dict):
            result[name].update(payload[name])
    core = defaults["core_params"]
    if isinstance(payload.get("core_params"), dict):
        for key, value in payload["core_params"].items():
            if isinstance(value, str):
                core[key] = value
            else:
                try:
                    core[key] = float(value)
                except (TypeError, ValueError):
                    continue
    result["core_params"] = core
    result["self_consistency_convergence_mode"] = canonicalize_self_consistency_mode(
        payload.get("self_consistency_convergence_mode", "fixed_geometry")
    )
    result["macroparticle_smearing_refresh_policy"] = result[
        "macroparticle_smearing_refresh_policy"
    ].replace("-", "_")
    result["magnetic_dipole_exact_retarded_backend"] = backend
    result["magnetic_dipole_source_local_jet_scales"] = parse_jet_scales(
        run_value(payload, "magnetic_dipole_source_local_jet_scales", [])
    )
    result["beamline_geometry_occluders"] = parse_occluders(
        run_value(payload, "beamline_geometry_occluders", [])
    )
    return cls(**result)


def vector3_matrix(value: Any) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        rows = tuple(tuple(float(item) for item in row) for row in value)
    except (TypeError, ValueError):
        return None
    return rows if all(len(row) == 3 for row in rows) else None


def resolve_exact_backend(payload: dict) -> Any:
    magnetic = payload.get("magnetic_dipole")
    magnetic = magnetic if isinstance(magnetic, dict) else {}
    source = magnetic.get("source")
    source = source if isinstance(source, dict) else {}
    canonical_present = (
        "magnetic_dipole_exact_retarded_backend" in payload
        or "exact_retarded_backend" in magnetic
    )
    canonical = payload.get(
        "magnetic_dipole_exact_retarded_backend", magnetic.get("exact_retarded_backend")
    )
    legacy = source.get("backend")
    if (
        canonical_present
        and "backend" in source
        and str(canonical).strip().lower() != str(legacy).strip().lower()
    ):
        raise ValueError(
            "magnetic_dipole.exact_retarded_backend conflicts with legacy "
            "magnetic_dipole.source.backend"
        )
    return str(
        canonical if canonical_present else legacy if "backend" in source else "python"
    )


def parse_jet_scales(value: Any) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("magnetic_dipole.source.local_jet_scales must be a list")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(
            "each magnetic_dipole.source.local_jet_scales item must be an object"
        )
    return [dict(item) for item in value]


def parse_occluders(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    occluders = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            occluders.append(
                {
                    "axis": [float(v) for v in item.get("axis", [0.0, 0.0, 1.0])],
                    "center_mm": [
                        float(v) for v in item.get("center_mm", [0.0, 0.0, 0.0])
                    ],
                    "radius_mm": float(item.get("radius_mm", 1.0)),
                    "length_mm": float(item.get("length_mm", 1.0)),
                    "label": str(item.get("label", "")),
                }
            )
        except (TypeError, ValueError):
            continue
    return occluders


_PERSISTED_FIELDS = """
self_consistency_enabled self_consistency_tolerance
self_consistency_convergence_mode self_consistency_target_ms_tolerance
self_consistency_max_iterations self_consistency_mass_shell_tolerance
self_consistency_mass_shell_relaxation self_consistency_verbosity chrono_interpolate
chrono_tolerance chrono_matching_mode chrono_high_precision
chrono_adaptive_tolerance self_consistency_chrono_interpolate
self_consistency_chrono_tolerance self_consistency_chrono_matching_mode
self_consistency_chrono_high_precision self_consistency_chrono_adaptive_tolerance
energy_monitor_halt_on_jump adaptive_timestep_enabled adaptive_timestep_threshold
adaptive_timestep_reduction_factor adaptive_timestep_min_factor
adaptive_timestep_cooldown_steps adaptive_timestep_probe_threshold
adaptive_timestep_max_probe_steps adaptive_timestep_debug space_charge_enabled
space_charge_retarded space_charge_softening_mm space_charge_bunch_sigma_mm
space_charge_instantaneous_clock space_charge_min_retarded_steps
external_field_enabled external_electric_field_native
external_electric_field_v_per_m external_magnetic_field_native external_field_x_min
external_field_x_max external_field_y_min external_field_y_max external_field_z_min
external_field_z_max external_field_t_min external_field_t_max
radiation_reaction_mode particle_loss_enabled particle_loss_radius_mm
particle_loss_conducting_wall_aperture_loss_enabled
particle_loss_initial_radial_quantile particle_loss_initial_radial_multiplier
particle_loss_initial_radial_margin_mm pseudo_grid_enabled
pseudo_grid_active_rider_count pseudo_grid_active_driver_count
pseudo_grid_passive_neighbor_count pseudo_grid_coverage_strategy
pseudo_grid_coverage_space pseudo_grid_pair_reuse_window
pseudo_grid_source_weighting_mode pseudo_grid_loss_tracking_enabled
pseudo_grid_causal_history_pruning_enabled
pseudo_grid_causal_history_safety_margin_steps driver_train_enabled
driver_train_bunch_count driver_train_z_spacing_mm driver_train_z_offsets_mm
driver_train_prehistory_steps driver_train_preserve_prehistory_in_output
self_consistency_gamma_reconciliation_method
self_consistency_gamma_reconciliation_low_beta_threshold
self_consistency_gamma_reconciliation_high_beta_threshold
self_consistency_gamma_reconciliation_low_beta_weight
self_consistency_gamma_reconciliation_high_beta_weight
self_consistency_gamma_reconciliation_mid_beta_weight
self_consistency_gamma_reconciliation_fixed_weight workers per_run_timeout
skip_failed_runs failed_run_retry_attempts smoothness_enabled smoothness_window_size
smoothness_oscillation_threshold smoothness_trend_threshold
smoothness_reject_on_violation smoothness_max_violations macroparticle_enabled
macroparticle_charge_multiplier macroparticle_sigma_multiplier
macroparticle_use_momentum_errors image_subcharge_count use_image_weighting
timestep_strategy z_cutoff_mode cavity_exit_enabled cavity_exit_mode
cavity_exit_length_mm cavity_exit_residual_tail_factor
cavity_exit_max_residual_tail_steps startup_mode target_distance_mm timestep
energy_scale_exponent transverse_geometry driver_transverse_geometry
""".split()

_PERSISTED_PROFILE_DEFAULTS = {"adaptive_timestep_reduction_factor": 10}


def apply_persisted_config_overrides(config: Any, data: dict[str, Any]) -> Any:
    """Apply persisted config values and defaults onto an OptimizationConfig."""
    nested_driver_train = data.get("driver_train")
    if isinstance(nested_driver_train, dict):
        data = dict(data)
        for source_key, target_key in {
            "enabled": "driver_train_enabled",
            "bunch_count": "driver_train_bunch_count",
            "z_spacing_mm": "driver_train_z_spacing_mm",
            "z_offsets_mm": "driver_train_z_offsets_mm",
            "prehistory_steps": "driver_train_prehistory_steps",
            "preserve_prehistory_in_output": "driver_train_preserve_prehistory_in_output",
        }.items():
            if source_key in nested_driver_train:
                data[target_key] = nested_driver_train[source_key]

    defaults = option_defaults(type(config))
    defaults.update(_PERSISTED_PROFILE_DEFAULTS)
    for attr_name in _PERSISTED_FIELDS:
        default = defaults[attr_name]
        setattr(config, attr_name, data.get(attr_name, default))
    # Chrono settings have a canonical chrono_* key and a legacy
    # self_consistency_chrono_* alias. The canonical key wins when both are
    # saved (matching the CLI and SimulationOptions.from_dict); otherwise the
    # alias fills both so downstream OR/non-default merging cannot diverge.
    for suffix in CHRONO_DEFAULTS:
        canonical_key = f"chrono_{suffix}"
        legacy_key = f"self_consistency_chrono_{suffix}"
        if canonical_key in data:
            value = data[canonical_key]
        elif legacy_key in data:
            value = data[legacy_key]
        else:
            continue
        setattr(config, canonical_key, value)
        setattr(config, legacy_key, value)
    config.driver_train_z_offsets_mm = tuple(config.driver_train_z_offsets_mm or ())
    config.transverse_geometry = data.get(
        "rider_transverse_geometry", data.get("transverse_geometry", "square")
    )
    config.driver_transverse_geometry = data.get("driver_transverse_geometry", "square")

    # Energy monitor behavior was removed; keep internal defaults explicit.
    config.energy_monitor_enabled = False
    config.energy_monitor_threshold = 2.0
    config.energy_monitor_check_interval = 10
    config.energy_monitor_debug = False
    return config


_SWEEP_SAVED_FIELDS = """
mode aperture_points aperture_log_scale energy_points energy_log_scale
transverse_offset_fractions starting_z_positions wall_z wall_z_range wall_z_points
cavity_spacing steps objective save_top_n_trajectories save_all_trajectories
save_failed_trajectories trajectory_stride metrics_export_format
metrics_export_scope log_verbosity optimization_method optimization_maxiter
optimization_population_size optimization_mutation_rate optimization_crossover_rate
optimization_n_starts optimization_save_top_n optimization_convergence_tol
optimization_convergence_patience self_consistency_enabled
self_consistency_tolerance self_consistency_convergence_mode
self_consistency_target_ms_tolerance self_consistency_max_iterations
self_consistency_mass_shell_tolerance self_consistency_mass_shell_relaxation
self_consistency_verbosity chrono_interpolate chrono_tolerance chrono_matching_mode
chrono_high_precision chrono_adaptive_tolerance self_consistency_chrono_interpolate
self_consistency_chrono_tolerance self_consistency_chrono_matching_mode
self_consistency_chrono_high_precision self_consistency_chrono_adaptive_tolerance
self_consistency_gamma_reconciliation_method
self_consistency_gamma_reconciliation_low_beta_threshold
self_consistency_gamma_reconciliation_high_beta_threshold
self_consistency_gamma_reconciliation_low_beta_weight
self_consistency_gamma_reconciliation_high_beta_weight
self_consistency_gamma_reconciliation_mid_beta_weight
self_consistency_gamma_reconciliation_fixed_weight energy_monitor_halt_on_jump
adaptive_timestep_enabled adaptive_timestep_threshold
adaptive_timestep_reduction_factor adaptive_timestep_min_factor
adaptive_timestep_cooldown_steps adaptive_timestep_probe_threshold
adaptive_timestep_max_probe_steps adaptive_timestep_debug
adaptive_timestep_bunch_proximity_enabled adaptive_timestep_bunch_proximity_sigma_mm
adaptive_timestep_bunch_proximity_n_sigma
adaptive_timestep_bunch_proximity_reduction_factor
adaptive_timestep_bunch_proximity_transition_n_sigma space_charge_enabled
space_charge_retarded space_charge_softening_mm space_charge_bunch_sigma_mm
space_charge_instantaneous_clock space_charge_min_retarded_steps
external_field_enabled external_electric_field_native
external_electric_field_v_per_m external_magnetic_field_native external_field_x_min
external_field_x_max external_field_y_min external_field_y_max external_field_z_min
external_field_z_max external_field_t_min external_field_t_max
radiation_reaction_mode workers per_run_timeout skip_failed_runs
failed_run_retry_attempts smoothness_enabled smoothness_window_size
smoothness_oscillation_threshold smoothness_trend_threshold
smoothness_reject_on_violation smoothness_max_violations macroparticle_enabled
macroparticle_charge_multiplier macroparticle_sigma_multiplier
macroparticle_use_momentum_errors image_subcharge_count use_image_weighting
timestep_strategy target_distance_mm timestep energy_scale_exponent z_cutoff_mode
cavity_exit_enabled cavity_exit_mode cavity_exit_length_mm
cavity_exit_residual_tail_factor cavity_exit_max_residual_tail_steps startup_mode
linked_energy_sweep
""".split()


def build_saved_config_payload(
    config: Any,
    *,
    timestep_mode: str,
    auto_steps_distance: float,
    rider_stripped_ions: float,
    driver_stripped_ions: float,
    driver_direction: str,
    sweep_state: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Serialize the GUI sweep format without dropping its UI-only keys."""
    run_payload = serialize_run_options(config)
    payload = {
        key: json_value(
            getattr(
                config,
                (
                    key.removeprefix("self_consistency_")
                    if key.startswith("self_consistency_chrono_")
                    else key
                ),
            )
        )
        for key in _SWEEP_SAVED_FIELDS
    }
    payload["particle_loss"] = run_payload["particle_loss"]
    payload["pseudo_grid"] = run_payload["pseudo_grid"]
    payload["driver_train"] = run_payload["driver_train"]
    payload.update(
        {
            "simulation_type": config.simulation_type.name,
            "aperture_min": config.aperture_range[0],
            "aperture_max": config.aperture_range[1],
            "energy_min": config.energy_range[0],
            "energy_max": config.energy_range[1],
            "timestep_mode": timestep_mode,
            "auto_steps_distance": auto_steps_distance,
            "rider_stripped_ions": rider_stripped_ions,
            "driver_stripped_ions": driver_stripped_ions,
            "transverse_geometry": getattr(config, "transverse_geometry", "square"),
            "rider_transverse_geometry": getattr(
                config, "transverse_geometry", "square"
            ),
            "driver_transverse_geometry": getattr(
                config, "driver_transverse_geometry", "square"
            ),
            "rider_offset_x": config.transv_offset_x,
            "rider_offset_y": config.transv_offset_y,
            "driver_offset_x": config.driver_transv_offset_x,
            "driver_offset_y": config.driver_transv_offset_y,
            "driver_direction": driver_direction,
            "sweep_parameters": sweep_state,
        }
    )
    return payload


def convert_sweep_options(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON config format to OptimizationConfig dataclass format.

    JSON uses separate min/max/points fields (e.g., aperture_min, aperture_max),
    while OptimizationConfig uses tuple ranges (e.g., aperture_range).

    This also maps *fixed* (non-swept) sweep_parameters to the corresponding
    OptimizationConfig scalar fields so that the CLI sweep runner uses the
    correct particle parameters instead of hard-coded defaults.
    """
    converted = dict(config_dict)
    if "mode" in converted:
        converted["mode"] = normalize_sweep_or_optimization_mode(converted["mode"])

    # Convert simulation_type string to SimulationType enum
    if "simulation_type" in converted and isinstance(converted["simulation_type"], str):
        sim_type_str = converted["simulation_type"].upper()
        if sim_type_str == "CONDUCTING_WALL":
            converted["simulation_type"] = SimulationType.CONDUCTING_WALL
        elif sim_type_str == "SWITCHING_WALL":
            converted["simulation_type"] = SimulationType.SWITCHING_WALL
        elif sim_type_str == "BUNCH_TO_BUNCH":
            converted["simulation_type"] = SimulationType.BUNCH_TO_BUNCH

    # Convert aperture fields
    if "aperture_min" in converted and "aperture_max" in converted:
        converted["aperture_range"] = (
            converted.pop("aperture_min"),
            converted.pop("aperture_max"),
        )

    # Convert energy fields
    if "energy_min" in converted and "energy_max" in converted:
        converted["energy_range"] = (
            converted.pop("energy_min"),
            converted.pop("energy_max"),
        )

    # Convert wall_z sweep fields
    if "wall_z_range" in converted and converted["wall_z_range"] is not None:
        wall_z_range = converted["wall_z_range"]
        if isinstance(wall_z_range, list) and len(wall_z_range) == 2:
            converted["wall_z_range"] = tuple(wall_z_range)

    # Map rider_stripped_ions top-level key -> stripped_ions (OptimizationConfig name)
    if "rider_stripped_ions" in converted and "stripped_ions" not in converted:
        converted["stripped_ions"] = float(converted.pop("rider_stripped_ions"))
    elif "rider_stripped_ions" in converted:
        converted.pop("rider_stripped_ions")

    # Map rider_offset / driver_offset top-level keys
    if "rider_offset_x" in converted:
        converted["transv_offset_x"] = float(converted.pop("rider_offset_x"))
    if "rider_offset_y" in converted:
        converted["transv_offset_y"] = float(converted.pop("rider_offset_y"))
    if "driver_offset_x" in converted:
        converted["driver_transv_offset_x"] = float(converted.pop("driver_offset_x"))
    if "driver_offset_y" in converted:
        converted["driver_transv_offset_y"] = float(converted.pop("driver_offset_y"))
    if "rider_transverse_geometry" in converted:
        converted["transverse_geometry"] = converted.pop("rider_transverse_geometry")

    # Map auto_steps_distance to auto_steps_distance_past_wall
    if (
        "auto_steps_distance" in converted
        and "auto_steps_distance_past_wall" not in converted
    ):
        converted["auto_steps_distance_past_wall"] = float(
            converted.pop("auto_steps_distance")
        )
    elif "auto_steps_distance" in converted:
        converted.pop("auto_steps_distance")

    flatten_sweep_blocks(converted)
    mirror_file_chrono_aliases(converted)

    # Convert sweep_parameters to appropriate ranges and fixed values
    sweep_params = converted.get("sweep_parameters", {})

    _fixed_field_map_rider = {
        "rider_m_particle": "m_particle",
        "rider_charge_sign": "charge_sign",
        "rider_pcount": "pcount",
        "rider_transv_mom": "transv_mom",
        "rider_transv_dist": "transv_dist",
        "rider_long_dist": "long_dist",
        "rider_stripped_ions": "stripped_ions",
        "macroparticle_charge_multiplier": "macroparticle_charge_multiplier",
        "macroparticle_sigma_multiplier": "macroparticle_sigma_multiplier",
    }
    _fixed_field_map_driver = {
        "driver_m_particle": "driver_m_particle",
        "driver_charge_sign": "driver_charge_sign",
        "driver_pcount": "driver_pcount",
        "driver_transv_mom": "driver_transv_mom",
        "driver_transv_dist": "driver_transv_dist",
        "driver_long_dist": "driver_long_dist",
        "driver_transverse_geometry": "driver_transverse_geometry",
        "driver_starting_distance": "driver_starting_distance",
        "driver_energy_gev": "driver_energy_gev",
        "driver_stripped_ions": "driver_stripped_ions",
    }

    _range_field_map_rider = {
        "rider_m_particle": "particle_mass_range",
        "rider_charge_sign": "particle_charge_range",
        "rider_pcount": "particle_count_range",
        "rider_transv_mom": "transverse_momentum_range",
        "rider_transv_dist": "transverse_spread_range",
        "rider_long_dist": "long_dist_range",
        "rider_stripped_ions": "rider_stripped_ions_range",
        "macroparticle_charge_multiplier": "macroparticle_charge_range",
        "macroparticle_sigma_multiplier": "macroparticle_sigma_range",
    }
    _range_field_map_driver = {
        "driver_m_particle": "driver_mass_range",
        "driver_charge_sign": "driver_charge_sign_range",
        "driver_pcount": "driver_pcount_range",
        "driver_transv_mom": "driver_transv_mom_range",
        "driver_transv_dist": "driver_transv_dist_range",
        "driver_long_dist": "driver_long_dist_range",
        "driver_starting_distance": "driver_starting_distance_range",
        "driver_energy_gev": "driver_energy_range",
        "driver_stripped_ions": "driver_stripped_ions_range",
    }

    all_param_names = list(_fixed_field_map_rider.keys()) + list(
        _fixed_field_map_driver.keys()
    )
    all_range_maps = {**_range_field_map_rider, **_range_field_map_driver}
    all_fixed_maps = {**_fixed_field_map_rider, **_fixed_field_map_driver}

    for param_name in all_param_names:
        if param_name not in sweep_params:
            continue
        param_config = sweep_params[param_name]

        if (
            param_config.get("enabled")
            and "min" in param_config
            and "max" in param_config
        ):
            # Swept parameter → range + points
            if param_name in all_range_maps:
                field_name = all_range_maps[param_name]
                min_val = float(param_config["min"])
                max_val = float(param_config["max"])
                if param_name == "driver_energy_gev":
                    min_val = abs(min_val)
                    max_val = abs(max_val)
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                converted[field_name] = (min_val, max_val)
                points_field = field_name.replace("_range", "_points")
                if "points" in param_config:
                    converted[points_field] = int(param_config["points"])
                if "log" in param_config:
                    log_field = field_name.replace("_range", "_log_scale")
                    converted[log_field] = bool(param_config["log"])
        else:
            # Fixed (disabled) parameter → scalar field
            if "fixed_value" in param_config and param_name in all_fixed_maps:
                scalar_field = all_fixed_maps[param_name]
                raw_val = param_config["fixed_value"]
                if param_name in ("rider_pcount", "driver_pcount"):
                    converted[scalar_field] = int(float(raw_val))
                else:
                    converted[scalar_field] = float(raw_val)

    # Remove sweep_parameters from converted dict as it's been processed
    converted.pop("sweep_parameters", None)

    # Remove fields that exist in JSON but not in OptimizationConfig dataclass
    fields_to_remove = [
        "timestep_mode",
        "auto_steps_distance",
    ]
    for field in fields_to_remove:
        converted.pop(field, None)

    return converted


def option_defaults(cls: Any) -> dict[str, Any]:
    defaults = {}
    for item in fields(cls):
        if item.default is not MISSING:
            defaults[item.name] = deepcopy(item.default)
        elif item.default_factory is not MISSING:
            defaults[item.name] = item.default_factory()
        else:
            raise ValueError(f"Option {item.name} must have a default")
    return defaults


def flatten_sweep_blocks(converted: dict[str, Any]) -> None:
    # Sweep files historically give nested blocks precedence over flat keys.
    # Only these four blocks are recognized by the CLI sweep conversion.
    for block in (
        "particle_loss",
        "pseudo_grid",
        "macroparticle_smearing",
        "driver_train",
    ):
        nested = converted.pop(block, None)
        if not isinstance(nested, dict):
            continue
        for name in SWEEP_BLOCK_FIELDS[block]:
            leaf = wire_path(name)[-1]
            if leaf in nested:
                value = nested[leaf]
                if leaf == "refresh_policy" and isinstance(value, str):
                    value = value.replace("-", "_")
                converted[name] = value


def mirror_file_chrono_aliases(converted: dict[str, Any]) -> None:
    for suffix in CHRONO_DEFAULTS:
        canonical, alias = f"chrono_{suffix}", f"self_consistency_chrono_{suffix}"
        if canonical in converted:
            if alias in converted:
                converted[alias] = converted[canonical]
        elif alias in converted:
            converted[canonical] = converted[alias]


SWEEP_BLOCK_FIELDS = {
    block: [
        item.name for item in fields(SharedOptions) if item.name.startswith(block + "_")
    ]
    for block in (
        "particle_loss",
        "pseudo_grid",
        "macroparticle_smearing",
        "driver_train",
    )
}
# This run default uses a factory; the sweep default is an immutable tuple.
SWEEP_BLOCK_FIELDS["driver_train"].append("driver_train_z_offsets_mm")


_RUN_TO_SWEEP_FIELDS = """
simulation_type steps seed self_consistency_enabled self_consistency_tolerance
self_consistency_convergence_mode self_consistency_target_ms_tolerance
self_consistency_max_iterations self_consistency_mass_shell_tolerance
self_consistency_mass_shell_relaxation self_consistency_verbosity chrono_interpolate
chrono_tolerance chrono_matching_mode chrono_high_precision
chrono_adaptive_tolerance energy_monitor_halt_on_jump adaptive_timestep_enabled
adaptive_timestep_threshold adaptive_timestep_reduction_factor
adaptive_timestep_min_factor adaptive_timestep_cooldown_steps
adaptive_timestep_probe_threshold adaptive_timestep_max_probe_steps
adaptive_timestep_debug adaptive_timestep_bunch_proximity_enabled
adaptive_timestep_bunch_proximity_sigma_mm adaptive_timestep_bunch_proximity_n_sigma
adaptive_timestep_bunch_proximity_reduction_factor
adaptive_timestep_bunch_proximity_transition_n_sigma space_charge_enabled
space_charge_retarded space_charge_softening_mm space_charge_bunch_sigma_mm
space_charge_instantaneous_clock space_charge_min_retarded_steps
external_field_enabled external_field_x_min external_field_x_max
external_field_y_min external_field_y_max external_field_z_min external_field_z_max
external_field_t_min external_field_t_max source_history_representation
radiation_reaction_mode particle_loss_enabled particle_loss_radius_mm
particle_loss_conducting_wall_aperture_loss_enabled
particle_loss_initial_radial_quantile particle_loss_initial_radial_multiplier
particle_loss_initial_radial_margin_mm pseudo_grid_enabled
pseudo_grid_active_rider_count pseudo_grid_active_driver_count
pseudo_grid_passive_neighbor_count pseudo_grid_coverage_strategy
pseudo_grid_coverage_space pseudo_grid_pair_reuse_window
pseudo_grid_source_weighting_mode pseudo_grid_loss_tracking_enabled
pseudo_grid_causal_history_pruning_enabled
pseudo_grid_causal_history_safety_margin_steps cavity_exit_enabled cavity_exit_mode
cavity_exit_length_mm cavity_exit_residual_tail_factor
cavity_exit_max_residual_tail_steps driver_train_enabled driver_train_bunch_count
driver_train_z_spacing_mm driver_train_prehistory_steps
driver_train_preserve_prehistory_in_output
""".split()

_RUN_TO_SWEEP_REQUIRED = (
    "simulation_type",
    "steps",
    "seed",
    "self_consistency_enabled",
    "self_consistency_tolerance",
    "self_consistency_max_iterations",
    "self_consistency_verbosity",
    "energy_monitor_halt_on_jump",
    "adaptive_timestep_enabled",
    "adaptive_timestep_threshold",
    "adaptive_timestep_reduction_factor",
    "adaptive_timestep_min_factor",
    "adaptive_timestep_cooldown_steps",
    "adaptive_timestep_probe_threshold",
    "adaptive_timestep_max_probe_steps",
    "adaptive_timestep_debug",
)


def copy_run_options_to_sweep(cls: Any, options: Any) -> dict[str, Any]:
    defaults = option_defaults(cls)
    copied = {
        name: getattr(options, name, defaults[name]) for name in _RUN_TO_SWEEP_FIELDS
    }
    for name in _RUN_TO_SWEEP_REQUIRED:
        copied[name] = getattr(options, name)
    for suffix, default in CHRONO_DEFAULTS.items():
        copied[f"self_consistency_chrono_{suffix}"] = getattr(
            options, f"chrono_{suffix}", default
        )
    return copied


def read_config_document(path: Path) -> Any:
    """Read a saved document losslessly, before selecting a domain parser."""
    import json

    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_config_document(
    payload: Any, path: Path, *, stringify_unknown: bool = False
) -> None:
    """Write all saved keys, including opaque metadata and deprecated keys."""
    import json

    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            json_value(payload),
            stream,
            indent=2,
            default=str if stringify_unknown else None,
        )


_SWEEP_TO_RUN_FIELDS = """
macroparticle_enabled macroparticle_use_momentum_errors
macroparticle_smearing_enabled macroparticle_smearing_subcharge_count
macroparticle_smearing_sigma_multiplier macroparticle_smearing_position_sigma_mm
macroparticle_smearing_longitudinal_sigma_mm
macroparticle_smearing_momentum_sigma_amu_mm_ns
macroparticle_smearing_use_position_errors
macroparticle_smearing_use_momentum_errors
macroparticle_smearing_use_centroid_errors macroparticle_smearing_use_internal_cloud
macroparticle_smearing_apply_to_active_observers
macroparticle_smearing_apply_to_active_sources
macroparticle_smearing_apply_to_passive_sources
macroparticle_smearing_apply_to_passive_updates macroparticle_smearing_seed
macroparticle_smearing_refresh_policy image_subcharge_count use_image_weighting
self_consistency_enabled self_consistency_tolerance
self_consistency_convergence_mode self_consistency_target_ms_tolerance
self_consistency_max_iterations self_consistency_mass_shell_tolerance
self_consistency_mass_shell_relaxation self_consistency_verbosity
self_consistency_gamma_reconciliation_method
self_consistency_gamma_reconciliation_low_beta_threshold
self_consistency_gamma_reconciliation_high_beta_threshold
self_consistency_gamma_reconciliation_low_beta_weight
self_consistency_gamma_reconciliation_high_beta_weight
self_consistency_gamma_reconciliation_mid_beta_weight
self_consistency_gamma_reconciliation_fixed_weight energy_monitor_halt_on_jump
adaptive_timestep_enabled adaptive_timestep_threshold
adaptive_timestep_reduction_factor adaptive_timestep_min_factor
adaptive_timestep_cooldown_steps adaptive_timestep_probe_threshold
adaptive_timestep_max_probe_steps adaptive_timestep_debug space_charge_enabled
space_charge_retarded space_charge_softening_mm space_charge_bunch_sigma_mm
space_charge_instantaneous_clock space_charge_min_retarded_steps
external_field_enabled external_field_x_min external_field_x_max
external_field_y_min external_field_y_max external_field_z_min external_field_z_max
external_field_t_min external_field_t_max source_history_representation
radiation_reaction_mode particle_loss_enabled particle_loss_radius_mm
particle_loss_conducting_wall_aperture_loss_enabled
particle_loss_initial_radial_quantile particle_loss_initial_radial_multiplier
particle_loss_initial_radial_margin_mm pseudo_grid_enabled
pseudo_grid_active_rider_count pseudo_grid_active_driver_count
pseudo_grid_passive_neighbor_count pseudo_grid_coverage_strategy
pseudo_grid_coverage_space pseudo_grid_pair_reuse_window
pseudo_grid_source_weighting_mode pseudo_grid_loss_tracking_enabled
pseudo_grid_causal_history_pruning_enabled
pseudo_grid_causal_history_safety_margin_steps cavity_exit_enabled cavity_exit_mode
cavity_exit_length_mm cavity_exit_residual_tail_factor
cavity_exit_max_residual_tail_steps driver_train_enabled driver_train_bunch_count
driver_train_z_spacing_mm driver_train_prehistory_steps
driver_train_preserve_prehistory_in_output
""".split()

_SWEEP_TO_RUN_REQUIRED = (
    "macroparticle_enabled",
    "macroparticle_use_momentum_errors",
    "image_subcharge_count",
    "use_image_weighting",
    "self_consistency_enabled",
    "self_consistency_tolerance",
    "self_consistency_max_iterations",
    "self_consistency_verbosity",
    "energy_monitor_halt_on_jump",
    "adaptive_timestep_enabled",
    "adaptive_timestep_threshold",
    "adaptive_timestep_reduction_factor",
    "adaptive_timestep_min_factor",
    "adaptive_timestep_cooldown_steps",
    "adaptive_timestep_probe_threshold",
    "adaptive_timestep_max_probe_steps",
    "adaptive_timestep_debug",
)


def copy_sweep_options_to_run(config: Any) -> dict[str, Any]:
    """Copy the per-integration subset with the existing constructor policy."""
    defaults = option_defaults(SharedOptions)
    copied = {
        name: getattr(config, name, defaults[name]) for name in _SWEEP_TO_RUN_FIELDS
    }
    for name in _SWEEP_TO_RUN_REQUIRED:
        copied[name] = getattr(config, name)
    copied.update(constructor_chrono_values(config))
    return copied


RUN_OPTION_ORDER = """
steps seed simulation_type energy_display energy_save energy_xaxis energy_yaxis
transverse_display transverse_save transverse_xaxis beta_display beta_save
beta_xaxis momentum_display momentum_save momentum_xaxis gamma_display gamma_save
gamma_xaxis zposition_display zposition_save trajectory_save trajectory_interval
plot_dpi output_dir config_dir config_name checkpoint_enabled checkpoint_directory
checkpoint_resume_from checkpoint_interval_steps checkpoint_interval_seconds
adaptive_pair_return_enabled adaptive_pair_target_lab_time_ns
adaptive_pair_tolerance_scale adaptive_pair_diagnostics_absolute_tolerance_native
adaptive_pair_minimum_step_factor adaptive_pair_maximum_step_factor
adaptive_pair_public_sample_interval_ns
adaptive_pair_shared_time_absolute_tolerance_ns
adaptive_pair_shared_time_relative_tolerance adaptive_pair_maximum_attempts
adaptive_pair_maximum_accepted_slabs manual_particle_config_enabled rider_params
driver_params core_params macroparticle_dynamics_mode image_subcharge_count
use_image_weighting macroparticle_enabled macroparticle_charge_multiplier
macroparticle_sigma_multiplier macroparticle_use_momentum_errors
macroparticle_smearing_enabled macroparticle_smearing_subcharge_count
macroparticle_smearing_sigma_multiplier macroparticle_smearing_position_sigma_mm
macroparticle_smearing_longitudinal_sigma_mm
macroparticle_smearing_momentum_sigma_amu_mm_ns
macroparticle_smearing_use_position_errors
macroparticle_smearing_use_momentum_errors
macroparticle_smearing_use_centroid_errors macroparticle_smearing_use_internal_cloud
macroparticle_smearing_apply_to_active_observers
macroparticle_smearing_apply_to_active_sources
macroparticle_smearing_apply_to_passive_sources
macroparticle_smearing_apply_to_passive_updates macroparticle_smearing_seed
macroparticle_smearing_refresh_policy self_consistency_enabled
self_consistency_tolerance self_consistency_convergence_mode
self_consistency_target_ms_tolerance self_consistency_max_iterations
self_consistency_mass_shell_tolerance self_consistency_mass_shell_relaxation
self_consistency_verbosity chrono_interpolate chrono_tolerance chrono_matching_mode
chrono_high_precision chrono_adaptive_tolerance self_consistency_chrono_interpolate
self_consistency_chrono_tolerance self_consistency_chrono_matching_mode
self_consistency_chrono_high_precision self_consistency_chrono_adaptive_tolerance
self_consistency_gamma_reconciliation_method
self_consistency_gamma_reconciliation_low_beta_threshold
self_consistency_gamma_reconciliation_high_beta_threshold
self_consistency_gamma_reconciliation_low_beta_weight
self_consistency_gamma_reconciliation_high_beta_weight
self_consistency_gamma_reconciliation_mid_beta_weight
self_consistency_gamma_reconciliation_fixed_weight energy_monitor_enabled
energy_monitor_threshold energy_monitor_check_interval energy_monitor_halt_on_jump
energy_monitor_debug adaptive_timestep_enabled adaptive_timestep_threshold
adaptive_timestep_reduction_factor adaptive_timestep_min_factor
adaptive_timestep_cooldown_steps adaptive_timestep_probe_threshold
adaptive_timestep_max_probe_steps adaptive_timestep_debug
adaptive_timestep_bunch_proximity_enabled adaptive_timestep_bunch_proximity_sigma_mm
adaptive_timestep_bunch_proximity_n_sigma
adaptive_timestep_bunch_proximity_reduction_factor
adaptive_timestep_bunch_proximity_transition_n_sigma space_charge_enabled
space_charge_retarded space_charge_softening_mm space_charge_bunch_sigma_mm
space_charge_instantaneous_clock space_charge_min_retarded_steps
external_field_enabled external_electric_field_native
external_electric_field_v_per_m external_magnetic_field_native
external_magnetic_field_gradient_t_per_m external_field_x_min external_field_x_max
external_field_y_min external_field_y_max external_field_z_min external_field_z_max
external_field_t_min external_field_t_max magnetic_dipole_enabled
magnetic_dipole_spin_precession_enabled magnetic_dipole_stern_gerlach_force_enabled
magnetic_dipole_spin_model magnetic_dipole_stern_gerlach_model
magnetic_dipole_exact_retarded_backend magnetic_dipole_exact_retarded_update
magnetic_dipole_intrinsic_spin_self_reaction_mode magnetic_dipole_source_model
magnetic_dipole_source_history_model magnetic_dipole_source_minimum_separation_mm
magnetic_dipole_source_relative_stencil_step
magnetic_dipole_source_minimum_stencil_step_mm
magnetic_dipole_source_root_tolerance_mm magnetic_dipole_source_max_root_iterations
magnetic_dipole_source_local_jet_primary_half_width_ns
magnetic_dipole_source_local_jet_narrow_half_width_ns
magnetic_dipole_source_local_jet_wide_half_width_ns
magnetic_dipole_source_local_jet_acceleration_degree
magnetic_dipole_source_local_jet_spin_degree
magnetic_dipole_source_local_jet_maximum_condition_number
magnetic_dipole_source_local_jet_maximum_relative_spread
magnetic_dipole_source_local_jet_scales
magnetic_dipole_source_local_jet_maximum_cross_scale_relative_spread
magnetic_dipole_source_local_jet_acceleration_samples
magnetic_dipole_source_local_jet_window_alignment
magnetic_dipole_source_local_jet_window_weighting
magnetic_dipole_source_local_jet_inertial_prehistory rider_magnetic_species
rider_magnetic_moment_j_per_t rider_spin_quantum_number rider_rest_spin
rider_polarization driver_magnetic_species driver_magnetic_moment_j_per_t
driver_spin_quantum_number driver_rest_spin driver_polarization
source_history_representation radiation_reaction_mode particle_loss_enabled
particle_loss_radius_mm particle_loss_conducting_wall_aperture_loss_enabled
particle_loss_initial_radial_quantile particle_loss_initial_radial_multiplier
particle_loss_initial_radial_margin_mm cavity_exit_enabled cavity_exit_mode
cavity_exit_length_mm cavity_exit_residual_tail_factor
cavity_exit_max_residual_tail_steps beamline_geometry_enabled
beamline_geometry_occluders auto_duration_enabled auto_duration_crossing_steps
auto_duration_post_factor pseudo_grid_enabled pseudo_grid_active_rider_count
pseudo_grid_active_driver_count pseudo_grid_field_rider_count
pseudo_grid_field_driver_count pseudo_grid_field_deposition_neighbor_count
pseudo_grid_space_charge_near_neighbor_count pseudo_grid_passive_neighbor_count
pseudo_grid_coverage_strategy pseudo_grid_coverage_space
pseudo_grid_active_selection_mode pseudo_grid_passive_update_mode
pseudo_grid_active_rotation_interval pseudo_grid_active_rotation_fraction
pseudo_grid_passive_remap_mode pseudo_grid_passive_remap_warning_sigma
pseudo_grid_passive_remap_trigger_sigma pseudo_grid_pair_reuse_window
pseudo_grid_source_weighting_mode pseudo_grid_loss_tracking_enabled
pseudo_grid_numerical_failure_tolerance_fraction
pseudo_grid_causal_history_pruning_enabled
pseudo_grid_causal_history_safety_margin_steps driver_train_enabled
driver_train_bunch_count driver_train_z_spacing_mm driver_train_z_offsets_mm
driver_train_prehistory_steps driver_train_preserve_prehistory_in_output
save_log_file log_file_path stochastic_emission
""".split()


SWEEP_OPTION_ORDER = """
simulation_type mode optimization_method optimization_maxiter
optimization_population_size optimization_mutation_rate optimization_crossover_rate
optimization_n_starts optimization_save_top_n optimization_convergence_tol
optimization_convergence_patience particle_death_penalty_fraction aperture_range
aperture_points aperture_log_scale energy_range energy_points energy_log_scale
transverse_offset_fractions starting_z_positions transverse_momentum_range
transverse_momentum_points transverse_momentum_log_scale transverse_spread_range
transverse_spread_points transverse_spread_log_scale long_dist_range
long_dist_points long_dist_log_scale timestep_range timestep_points starting_z_range
starting_z_points wall_z_range wall_z_points particle_mass_range
particle_mass_points particle_charge_range particle_charge_points
cavity_spacing_range cavity_spacing_points macroparticle_charge_range
macroparticle_charge_points macroparticle_sigma_range macroparticle_sigma_points
rider_stripped_ions_range rider_stripped_ions_points driver_stripped_ions_range
driver_stripped_ions_points particle_count_range particle_count_points
driver_mass_range driver_mass_points driver_charge_sign_range
driver_charge_sign_points driver_pcount_range driver_pcount_points
driver_transv_mom_range driver_transv_mom_points driver_transv_mom_log_scale
driver_transv_dist_range driver_transv_dist_points driver_transv_dist_log_scale
driver_long_dist_range driver_long_dist_points driver_long_dist_log_scale
driver_starting_distance_range driver_starting_distance_points
driver_starting_distance_log_scale driver_starting_Pz_range
driver_starting_Pz_points driver_energy_range driver_energy_points
driver_energy_log_scale linked_energy_sweep wall_z cavity_spacing steps timestep
auto_steps auto_steps_target auto_steps_distance_past_wall seed timestep_strategy
energy_scale_exponent target_distance_mm z_cutoff_mode cavity_exit_enabled
cavity_exit_mode cavity_exit_length_mm cavity_exit_residual_tail_factor
cavity_exit_max_residual_tail_steps startup_mode transv_mom transv_dist long_dist
driver_long_dist transverse_geometry transv_offset_x transv_offset_y
driver_transv_offset_x driver_transv_offset_y m_particle pcount charge_sign
stripped_ions driver_m_particle driver_charge_sign driver_pcount driver_transv_mom
driver_transv_dist driver_transverse_geometry driver_starting_distance
driver_stripped_ions driver_starting_Pz driver_energy_gev driver_direction
macroparticle_enabled macroparticle_charge_multiplier macroparticle_sigma_multiplier
macroparticle_use_momentum_errors macroparticle_smearing_enabled
macroparticle_smearing_subcharge_count macroparticle_smearing_sigma_multiplier
macroparticle_smearing_position_sigma_mm
macroparticle_smearing_longitudinal_sigma_mm
macroparticle_smearing_momentum_sigma_amu_mm_ns
macroparticle_smearing_use_position_errors
macroparticle_smearing_use_momentum_errors
macroparticle_smearing_use_centroid_errors macroparticle_smearing_use_internal_cloud
macroparticle_smearing_apply_to_active_observers
macroparticle_smearing_apply_to_active_sources
macroparticle_smearing_apply_to_passive_sources
macroparticle_smearing_apply_to_passive_updates macroparticle_smearing_seed
macroparticle_smearing_refresh_policy image_subcharge_count use_image_weighting
objective objective_weights output_dir save_results save_plots
save_top_n_trajectories save_all_trajectories save_failed_trajectories
trajectory_stride metrics_export_format metrics_export_scope log_verbosity
self_consistency_enabled self_consistency_tolerance
self_consistency_convergence_mode self_consistency_target_ms_tolerance
self_consistency_max_iterations self_consistency_mass_shell_tolerance
self_consistency_mass_shell_relaxation self_consistency_verbosity chrono_interpolate
chrono_tolerance chrono_matching_mode chrono_high_precision
chrono_adaptive_tolerance self_consistency_chrono_interpolate
self_consistency_chrono_tolerance self_consistency_chrono_matching_mode
self_consistency_chrono_high_precision self_consistency_chrono_adaptive_tolerance
self_consistency_gamma_reconciliation_method
self_consistency_gamma_reconciliation_low_beta_threshold
self_consistency_gamma_reconciliation_high_beta_threshold
self_consistency_gamma_reconciliation_low_beta_weight
self_consistency_gamma_reconciliation_high_beta_weight
self_consistency_gamma_reconciliation_mid_beta_weight
self_consistency_gamma_reconciliation_fixed_weight energy_monitor_enabled
energy_monitor_threshold energy_monitor_check_interval energy_monitor_halt_on_jump
energy_monitor_debug adaptive_timestep_enabled adaptive_timestep_threshold
adaptive_timestep_reduction_factor adaptive_timestep_min_factor
adaptive_timestep_cooldown_steps adaptive_timestep_probe_threshold
adaptive_timestep_max_probe_steps adaptive_timestep_debug
adaptive_timestep_bunch_proximity_enabled adaptive_timestep_bunch_proximity_sigma_mm
adaptive_timestep_bunch_proximity_n_sigma
adaptive_timestep_bunch_proximity_reduction_factor
adaptive_timestep_bunch_proximity_transition_n_sigma space_charge_enabled
space_charge_retarded space_charge_softening_mm space_charge_bunch_sigma_mm
space_charge_instantaneous_clock space_charge_min_retarded_steps
external_field_enabled external_electric_field_native
external_electric_field_v_per_m external_magnetic_field_native external_field_x_min
external_field_x_max external_field_y_min external_field_y_max external_field_z_min
external_field_z_max external_field_t_min external_field_t_max
source_history_representation radiation_reaction_mode particle_loss_enabled
particle_loss_radius_mm particle_loss_conducting_wall_aperture_loss_enabled
particle_loss_initial_radial_quantile particle_loss_initial_radial_multiplier
particle_loss_initial_radial_margin_mm pseudo_grid_enabled
pseudo_grid_active_rider_count pseudo_grid_active_driver_count
pseudo_grid_passive_neighbor_count pseudo_grid_coverage_strategy
pseudo_grid_coverage_space pseudo_grid_pair_reuse_window
pseudo_grid_source_weighting_mode pseudo_grid_loss_tracking_enabled
pseudo_grid_causal_history_pruning_enabled
pseudo_grid_causal_history_safety_margin_steps driver_train_enabled
driver_train_bunch_count driver_train_z_spacing_mm driver_train_z_offsets_mm
driver_train_prehistory_steps driver_train_preserve_prehistory_in_output workers
per_run_timeout skip_failed_runs failed_run_retry_attempts smoothness_enabled
smoothness_window_size smoothness_oscillation_threshold smoothness_trend_threshold
smoothness_reject_on_violation smoothness_max_violations
""".split()


def preserve_option_constructor(order: list[str]) -> Any:
    """Keep public positional calls and signatures stable after sharing fields."""

    def decorate(cls: Any) -> Any:
        initialize = cls.__init__
        current = signature(initialize)
        parameters = current.parameters
        previous = current.replace(
            parameters=[parameters["self"]] + [parameters[name] for name in order]
        )

        @wraps(initialize)
        def initialize_in_order(self: Any, *args: Any, **kwargs: Any) -> None:
            if args:
                bound = previous.bind(self, *args, **kwargs)
                kwargs = dict(bound.arguments)
                kwargs.pop("self")
            initialize(self, **kwargs)

        setattr(initialize_in_order, "__signature__", previous)
        cls.__init__ = initialize_in_order
        return cls

    return decorate
