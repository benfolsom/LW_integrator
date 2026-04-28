"""Pure parameter-mapping helpers for optimization run control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from core.types import SimulationType
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
