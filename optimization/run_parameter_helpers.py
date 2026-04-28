"""Pure parameter-mapping helpers for optimization run control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from core.types import SimulationType


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
    if "min" in objective.lower():
        return "max_energy_gain_gev", False
    return "max_energy_gain_gev", True


def is_bunch_to_bunch(simulation_type: Any) -> bool:
    """Return whether a simulation type value represents BUNCH_TO_BUNCH mode."""
    if simulation_type == SimulationType.BUNCH_TO_BUNCH:
        return True
    if simulation_type == "BUNCH_TO_BUNCH":
        return True
    return getattr(simulation_type, "name", None) == "BUNCH_TO_BUNCH"


def calculate_transverse_offset(
    simulation_type: Any, offset_value: float, aperture: float
) -> float:
    """Resolve offset semantics for wall and bunch-to-bunch simulations."""
    if is_bunch_to_bunch(simulation_type):
        return offset_value
    return offset_value * aperture
