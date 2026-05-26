"""Pure parameter-mapping helpers for optimization run control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from optimization.simulation_type_helpers import is_bunch_to_bunch
from optimization.sweep_helpers import calculate_starting_pz_from_energy


@dataclass(frozen=True)
class OptimizationParameterDefinition:
    """Configuration fields that expose one optimization parameter."""

    name: str
    range_attr: str
    points_attr: str
    log_when_added: bool = False


@dataclass(frozen=True)
class OptimizationParameterSelection:
    """Selected optimization parameters and bounds."""

    names: List[str]
    bounds: List[Tuple[float, float]]
    log_lines: List[str]


@dataclass(frozen=True)
class OptimizationRunParameters:
    """Resolved parameters for one optimization objective evaluation."""

    aperture: float
    energy_gev: float
    start_z: float
    transv_offset: float
    timestep: float
    steps: int
    rider_m_particle: float
    rider_charge_sign: float
    rider_pcount: int
    rider_transv_mom: float
    rider_transv_dist: float
    rider_stripped_ions: float
    macroparticle_charge_multiplier: float
    macroparticle_sigma_multiplier: float
    driver_params: dict[str, Any] | None
    wall_z: float


@dataclass(frozen=True)
class OptimizationEvaluationOutcome:
    """Classified optimizer evaluation result and persisted record."""

    fitness: float
    record: dict[str, Any]
    log_lines: List[str]


ENERGY_CONSTRAINED_RADIAL_FOCUS_OBJECTIVES = {
    "max_inward_rider_radial_focusing_constrained_energy",
    "max_radial_focusing_constrained_energy",
}
PEAK_ENERGY_CONSTRAINED_RADIAL_FOCUS_OBJECTIVES = {
    "max_peak_inward_rider_radial_focusing_constrained_energy",
    "max_peak_radial_focusing_constrained_energy",
}
RMS_PEAK_ENERGY_CONSTRAINED_RADIAL_FOCUS_OBJECTIVES = {
    "max_peak_rider_radial_rms_collapse_constrained_energy",
    "max_peak_ring_rms_collapse_constrained_energy",
}
PERCENTILE_ENERGY_CONSTRAINED_HALO_OBJECTIVES = {
    "max_rider_radial_p95_reduction_constrained_energy",
    "max_rider_radial_p99_reduction_constrained_energy",
}
HALO_FRACTION_ENERGY_CONSTRAINED_OBJECTIVES = {
    "max_rider_halo_2rms_reduction_constrained_energy",
    "max_rider_halo_3rms_reduction_constrained_energy",
    "max_rider_halo_5rms_reduction_constrained_energy",
}


OPTIMIZATION_PARAMETER_DEFINITIONS: tuple[OptimizationParameterDefinition, ...] = (
    OptimizationParameterDefinition(
        "aperture_radius", "aperture_range", "aperture_points"
    ),
    OptimizationParameterDefinition(
        "initial_energy_gev", "energy_range", "energy_points", log_when_added=True
    ),
    OptimizationParameterDefinition(
        "transverse_momentum",
        "transverse_momentum_range",
        "transverse_momentum_points",
        log_when_added=True,
    ),
    OptimizationParameterDefinition("timestep", "timestep_range", "timestep_points"),
    OptimizationParameterDefinition(
        "rider_transv_dist",
        "transverse_spread_range",
        "transverse_spread_points",
        log_when_added=True,
    ),
    OptimizationParameterDefinition(
        "macroparticle_charge_multiplier",
        "macroparticle_charge_range",
        "macroparticle_charge_points",
    ),
    OptimizationParameterDefinition(
        "macroparticle_sigma_multiplier",
        "macroparticle_sigma_range",
        "macroparticle_sigma_points",
    ),
    OptimizationParameterDefinition("wall_z", "wall_z_range", "wall_z_points"),
    OptimizationParameterDefinition(
        "rider_stripped_ions",
        "rider_stripped_ions_range",
        "rider_stripped_ions_points",
    ),
    OptimizationParameterDefinition(
        "driver_stripped_ions",
        "driver_stripped_ions_range",
        "driver_stripped_ions_points",
    ),
    OptimizationParameterDefinition(
        "rider_m_particle",
        "particle_mass_range",
        "particle_mass_points",
        log_when_added=True,
    ),
    OptimizationParameterDefinition(
        "rider_charge_sign", "particle_charge_range", "particle_charge_points"
    ),
    OptimizationParameterDefinition(
        "rider_pcount",
        "particle_count_range",
        "particle_count_points",
        log_when_added=True,
    ),
    OptimizationParameterDefinition(
        "driver_m_particle",
        "driver_mass_range",
        "driver_mass_points",
        log_when_added=True,
    ),
    OptimizationParameterDefinition(
        "driver_charge_sign", "driver_charge_sign_range", "driver_charge_sign_points"
    ),
    OptimizationParameterDefinition(
        "driver_pcount",
        "driver_pcount_range",
        "driver_pcount_points",
        log_when_added=True,
    ),
    OptimizationParameterDefinition(
        "driver_transv_mom",
        "driver_transv_mom_range",
        "driver_transv_mom_points",
        log_when_added=True,
    ),
    OptimizationParameterDefinition(
        "driver_transv_dist",
        "driver_transv_dist_range",
        "driver_transv_dist_points",
        log_when_added=True,
    ),
    OptimizationParameterDefinition(
        "driver_starting_distance",
        "driver_starting_distance_range",
        "driver_starting_distance_points",
    ),
    OptimizationParameterDefinition(
        "driver_energy_gev", "driver_energy_range", "driver_energy_points"
    ),
)


def collect_optimization_parameter_selection(
    config: Any,
) -> OptimizationParameterSelection:
    """Return enabled optimizer parameters in the historical declaration order."""
    names: List[str] = []
    bounds: List[Tuple[float, float]] = []
    log_lines: List[str] = []

    for definition in OPTIMIZATION_PARAMETER_DEFINITIONS:
        range_value = getattr(config, definition.range_attr)
        point_count = getattr(config, definition.points_attr)
        if range_value is None or point_count <= 1:
            continue

        names.append(definition.name)
        bounds.append(range_value)
        if definition.log_when_added:
            log_lines.append(
                f"    Added: {definition.name}, "
                f"range={range_value}, points={point_count}"
            )

    return OptimizationParameterSelection(
        names=names, bounds=bounds, log_lines=log_lines
    )


def resolve_objective_metric(objective: str) -> tuple[str, bool]:
    """Return ``(metric_name, maximize)`` for an optimization objective name."""
    if objective == "max_percent_energy_gain":
        return "max_percent_energy_gain", True
    if (
        objective in RMS_PEAK_ENERGY_CONSTRAINED_RADIAL_FOCUS_OBJECTIVES
        or objective
        in {
            "max_peak_rider_radial_rms_collapse",
            "max_peak_ring_rms_collapse",
        }
    ):
        return "rider_radial_rms_peak_inward_mm", True
    if objective in PEAK_ENERGY_CONSTRAINED_RADIAL_FOCUS_OBJECTIVES or objective in {
        "max_peak_inward_rider_radial_focusing",
        "max_peak_radial_focusing",
    }:
        return "rider_radial_peak_inward_mm", True
    if objective in ENERGY_CONSTRAINED_RADIAL_FOCUS_OBJECTIVES or objective in {
        "max_inward_rider_radial_focusing",
        "max_radial_focusing",
    }:
        return "rider_radial_toward_driver_mm", True
    if objective == "max_rider_radial_p95_reduction_constrained_energy":
        return "rider_radial_p95_mm_reduction", True
    if objective == "max_rider_radial_p99_reduction_constrained_energy":
        return "rider_radial_p99_mm_reduction", True
    if objective == "max_rider_halo_2rms_reduction_constrained_energy":
        return "rider_halo_gt_2_initial_rms_fraction_reduction", True
    if objective == "max_rider_halo_3rms_reduction_constrained_energy":
        return "rider_halo_gt_3_initial_rms_fraction_reduction", True
    if objective == "max_rider_halo_5rms_reduction_constrained_energy":
        return "rider_halo_gt_5_initial_rms_fraction_reduction", True
    if "min" in objective.lower():
        return "max_energy_gain_gev", False
    return "max_energy_gain_gev", True


def calculate_transverse_offset(
    simulation_type: Any, offset_value: float, aperture: float
) -> float:
    """Resolve offset semantics for wall and bunch-to-bunch simulations."""
    if is_bunch_to_bunch(simulation_type):
        return offset_value
    return offset_value * aperture


def resolve_optimization_run_parameters(
    config: Any, param_names: list[str], values: Any
) -> OptimizationRunParameters:
    """Map one optimizer parameter vector onto integration-run arguments."""
    aperture = config.aperture_range[0]
    energy = config.energy_range[0]
    start_z = config.starting_z_positions[0] if config.starting_z_positions else 0.0
    offset_value = (
        config.transverse_offset_fractions[0]
        if config.transverse_offset_fractions
        else 0.0
    )
    timestep = config.timestep
    steps = config.steps
    rider_transv_dist = config.transv_dist
    macroparticle_charge_multiplier = config.macroparticle_charge_multiplier
    macroparticle_sigma_multiplier = config.macroparticle_sigma_multiplier
    wall_z = config.wall_z
    rider_stripped_ions = config.stripped_ions
    driver_stripped_ions = config.driver_stripped_ions
    rider_m_particle = config.m_particle
    rider_charge_sign = config.charge_sign
    rider_pcount = config.pcount
    rider_transv_mom = config.transv_mom
    driver_m_particle = config.driver_m_particle
    driver_charge_sign = config.driver_charge_sign
    driver_pcount = config.driver_pcount
    driver_transv_mom = config.driver_transv_mom
    driver_transv_dist = config.driver_transv_dist
    driver_starting_distance = config.driver_starting_distance
    driver_starting_pz = config.driver_starting_Pz

    for index, param_name in enumerate(param_names):
        value = values[index]
        if param_name == "aperture_radius":
            aperture = value
        elif param_name == "initial_energy_gev":
            energy = value
        elif param_name == "start_z":
            start_z = value
        elif param_name == "transverse_offset":
            offset_value = value
        elif param_name == "timestep":
            timestep = value
        elif param_name == "transverse_momentum":
            rider_transv_mom = value
        elif param_name == "rider_transv_dist":
            rider_transv_dist = value
        elif param_name == "macroparticle_charge_multiplier":
            macroparticle_charge_multiplier = value
        elif param_name == "macroparticle_sigma_multiplier":
            macroparticle_sigma_multiplier = value
        elif param_name == "wall_z":
            wall_z = value
        elif param_name == "rider_stripped_ions":
            rider_stripped_ions = value
        elif param_name == "driver_stripped_ions":
            driver_stripped_ions = value
        elif param_name == "rider_m_particle":
            rider_m_particle = value
        elif param_name == "rider_charge_sign":
            rider_charge_sign = value
        elif param_name == "rider_pcount":
            rider_pcount = int(value)
        elif param_name == "driver_m_particle":
            driver_m_particle = value
        elif param_name == "driver_charge_sign":
            driver_charge_sign = value
        elif param_name == "driver_pcount":
            driver_pcount = int(value)
        elif param_name == "driver_transv_mom":
            driver_transv_mom = value
        elif param_name == "driver_transv_dist":
            driver_transv_dist = value
        elif param_name == "driver_starting_distance":
            driver_starting_distance = value
        elif param_name == "driver_energy_gev":
            driver_negative = getattr(config, "driver_direction", "-z") == "-z"
            driver_starting_pz = calculate_starting_pz_from_energy(
                value, driver_m_particle, negative=driver_negative
            )
        elif param_name == "driver_starting_Pz":
            driver_starting_pz = value

    transv_offset = calculate_transverse_offset(
        config.simulation_type, offset_value, aperture
    )

    driver_params = None
    if is_bunch_to_bunch(config.simulation_type):
        driver_params = {
            "m_particle": driver_m_particle,
            "charge_sign": driver_charge_sign,
            "pcount": int(driver_pcount),
            "transv_mom": driver_transv_mom,
            "transv_dist": driver_transv_dist,
            "transverse_geometry": getattr(
                config, "driver_transverse_geometry", "square"
            ),
            "starting_distance": driver_starting_distance,
            "starting_Pz": driver_starting_pz,
            "stripped_ions": driver_stripped_ions,
            "transv_offset_x": config.driver_transv_offset_x,
            "transv_offset_y": config.driver_transv_offset_y,
        }

    if config.timestep_strategy == "auto_distance":
        driver_start_z = 1000.0
        if driver_params is not None:
            driver_start_z = driver_params.get("starting_distance", 1000.0)

        timestep = config.calculate_timestep_for_energy(
            energy,
            config.m_particle,
            wall_z=wall_z,
            start_z=start_z,
            driver_start_z=driver_start_z,
        )
        steps = config.steps

    return OptimizationRunParameters(
        aperture=aperture,
        energy_gev=energy,
        start_z=start_z,
        transv_offset=transv_offset,
        timestep=timestep,
        steps=steps,
        rider_m_particle=rider_m_particle,
        rider_charge_sign=rider_charge_sign,
        rider_pcount=int(rider_pcount),
        rider_transv_mom=rider_transv_mom,
        rider_transv_dist=rider_transv_dist,
        rider_stripped_ions=rider_stripped_ions,
        macroparticle_charge_multiplier=macroparticle_charge_multiplier,
        macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
        driver_params=driver_params,
        wall_z=wall_z,
    )


def build_optimization_evaluation_outcome(
    result: dict[str, Any] | None,
    *,
    eval_num: int,
    param_names: list[str],
    values: Any,
    metric_name: str,
    maximize: bool,
    penalty: float = 0.0,
    objective_name: str = "objective",
    save_trajectory: bool = False,
) -> OptimizationEvaluationOutcome:
    """Classify one optimizer evaluation result for fitness and persistence."""
    parameters = dict(zip(param_names, values))

    if result is None or "metrics" not in result:
        record = {
            "evaluation": eval_num,
            "parameters": parameters,
            "failed": True,
            "halted_early": result.get("halted_early", False) if result else False,
            "halt_reason": result.get("halt_reason", None) if result else None,
            "objective_value": float("inf"),
        }
        return OptimizationEvaluationOutcome(
            fitness=np.inf,
            record=record,
            log_lines=[],
        )

    if result.get("halted_early", False):
        record = {
            "evaluation": eval_num,
            "parameters": parameters,
            "failed": False,
            "halted_early": True,
            "halt_reason": result.get("halt_reason"),
            "objective_value": float("inf"),
        }
        return OptimizationEvaluationOutcome(
            fitness=np.inf,
            record=record,
            log_lines=[
                (
                    f"[INFO] Evaluation {eval_num} halted early: "
                    f"{result.get('halt_reason', 'unknown')}"
                ),
                "[INFO] Returning inf (rejecting halted evaluation)",
            ],
        )

    metrics = result["metrics"]
    value = metrics.get(metric_name, np.nan)
    constraint_reason = _energy_constrained_radial_focus_failure(
        metrics,
        focus_value=value,
        objective_name=objective_name,
    )
    if constraint_reason is not None:
        record = {
            "evaluation": eval_num,
            "parameters": parameters,
            "failed": False,
            "constraint_failed": True,
            "constraint_reason": constraint_reason,
            "objective_value": float("inf"),
            "raw_objective_value": value,
            "fitness": float("inf"),
            "metrics": result.get("metrics", {}),
        }
        return OptimizationEvaluationOutcome(
            fitness=np.inf,
            record=record,
            log_lines=[
                (
                    f"[INFO] Evaluation {eval_num} rejected by "
                    f"{objective_name}: {constraint_reason}"
                )
            ],
        )

    if np.isnan(value) or np.isinf(value):
        kind = "NaN" if np.isnan(value) else "inf"
        log_lines = [
            (
                f"[WARNING] Evaluation {eval_num} returned {kind} "
                f"for metric '{metric_name}'"
            ),
            f"[WARNING] Available metrics: {list(metrics.keys())}",
        ]
        if metrics:
            log_lines.append("[WARNING] Metric values:")
            log_lines.extend(f"  {key}: {metric}" for key, metric in metrics.items())
        log_lines.append("[WARNING] Returning inf (rejecting this evaluation)")

        record = {
            "evaluation": eval_num,
            "parameters": parameters,
            "failed": True,
            "objective_value": float("inf"),
            "metrics": result.get("metrics", {}),
        }
        return OptimizationEvaluationOutcome(
            fitness=np.inf,
            record=record,
            log_lines=log_lines,
        )

    adjusted_value = value
    log_lines: List[str] = []
    if penalty > 0:
        adjusted_value = value - penalty if maximize else value + penalty
        log_lines.append(
            "[INFO] Applied soft penalty of "
            f"{penalty:.3e} to {objective_name} (risk-prone parameters)"
        )

    fitness = -adjusted_value if maximize else adjusted_value
    record = {
        "evaluation": eval_num,
        "parameters": parameters,
        "objective_value": adjusted_value,
        "raw_objective_value": value,
        "soft_penalty": penalty,
        "fitness": fitness,
        "failed": False,
        "halted_early": False,
        "metrics": result.get("metrics", {}),
    }

    if save_trajectory and "trajectory" in result:
        record["trajectory"] = result["trajectory"]

    return OptimizationEvaluationOutcome(
        fitness=fitness,
        record=record,
        log_lines=log_lines,
    )


def _energy_constrained_radial_focus_failure(
    metrics: dict[str, Any],
    *,
    focus_value: Any,
    objective_name: str,
) -> str | None:
    if objective_name in RMS_PEAK_ENERGY_CONSTRAINED_RADIAL_FOCUS_OBJECTIVES:
        focus_label = "peak inward radial rms collapse"
    elif objective_name in PEAK_ENERGY_CONSTRAINED_RADIAL_FOCUS_OBJECTIVES:
        focus_label = "peak inward radial focusing"
    elif objective_name in ENERGY_CONSTRAINED_RADIAL_FOCUS_OBJECTIVES:
        focus_label = "inward radial focusing"
    elif objective_name in PERCENTILE_ENERGY_CONSTRAINED_HALO_OBJECTIVES:
        focus_label = "persistent radial percentile reduction"
    elif objective_name in HALO_FRACTION_ENERGY_CONSTRAINED_OBJECTIVES:
        focus_label = "halo-fraction reduction"
    else:
        return None

    focus = _finite_float_or_none(focus_value)
    if focus is None:
        return f"missing or invalid {focus_label} metric"
    if focus <= 0.0:
        return f"rider {focus_label} is not positive"

    delta_e_mev = metrics.get("delta_e_mev", metrics.get("rider_delta_e_mev"))
    delta_e = _finite_float_or_none(delta_e_mev)
    if delta_e is None:
        return "missing or invalid rider energy-change metric"
    if delta_e <= 0.0:
        return "rider dE is not positive"

    energy_fraction = _finite_float_or_none(
        metrics.get("rider_delta_e_fraction_initial_kinetic")
    )
    if energy_fraction is None:
        energy_fraction = _finite_float_or_none(
            metrics.get("rider_delta_e_fraction_initial_total")
        )
    if energy_fraction is None:
        energy_fraction = _finite_float_or_none(metrics.get("max_relative_gain"))
    if energy_fraction is None:
        return "missing or invalid rider dE fraction metric"
    if energy_fraction > 0.20:
        return "rider dE exceeds 20% of initial rider energy"

    return None


def _finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None
