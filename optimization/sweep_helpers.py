"""Shared helper functions for optimization sweeps and UI state."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from core.constants import C_MMNS  # type: ignore[import]
from optimization.simulation_type_helpers import is_bunch_to_bunch

AMU_TO_MEV = 931.494


def calculate_starting_pz_from_energy(
    energy_gev: float, mass_amu: float, negative: bool = False
) -> float:
    """Convert kinetic energy in GeV to starting Pz in amu·mm/ns."""
    energy_gev = abs(energy_gev)
    rest_energy_mev = mass_amu * AMU_TO_MEV
    gamma = (energy_gev * 1e3) / rest_energy_mev + 1.0
    gamma = max(gamma, 1.0)
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.0
    pz = gamma * mass_amu * C_MMNS * beta
    return -pz if negative else pz


def calculate_energy_from_pz(pz: float, mass_amu: float) -> float:
    """Convert Pz in amu·mm/ns to kinetic energy in GeV."""
    if abs(pz) < 1e-12:
        return 0.0

    rest_energy_mev = mass_amu * AMU_TO_MEV
    gamma_beta = abs(pz) / (mass_amu * C_MMNS)
    gamma = np.sqrt(gamma_beta**2 + 1.0)
    kinetic_energy_mev = (gamma - 1.0) * rest_energy_mev
    return kinetic_energy_mev / 1e3


def generate_parameter_range(
    min_val: float, max_val: float, points: int, log_scale: bool
) -> list[float]:
    """Generate a linearly or logarithmically spaced parameter range."""
    if points == 1:
        return [(min_val + max_val) / 2.0]
    if log_scale and min_val > 0 and max_val > 0:
        return np.logspace(np.log10(min_val), np.log10(max_val), points).tolist()
    return np.linspace(min_val, max_val, points).tolist()


def build_parameter_grids(config: Any, sweep_params: Mapping[str, Any]) -> dict[str, list]:
    """Build sweep grids from config and UI sweep control state."""
    grids: dict[str, list] = {}
    sim_type = config.simulation_type

    if not is_bunch_to_bunch(sim_type):
        grids["aperture"] = generate_parameter_range(
            config.aperture_range[0],
            config.aperture_range[1],
            config.aperture_points,
            config.aperture_log_scale,
        )

    energy_key = "initial_energy_gev" if is_bunch_to_bunch(sim_type) else "energy"
    grids[energy_key] = generate_parameter_range(
        config.energy_range[0],
        config.energy_range[1],
        config.energy_points,
        config.energy_log_scale,
    )

    if config.transverse_offset_fractions:
        grids["transverse_offset_fraction"] = [config.transverse_offset_fractions[0]]
    else:
        grids["transverse_offset_fraction"] = [0.0]

    grids["start_z"] = config.starting_z_positions

    if config.wall_z_range is not None and config.wall_z_points > 1:
        grids["wall_z"] = generate_parameter_range(
            config.wall_z_range[0],
            config.wall_z_range[1],
            config.wall_z_points,
            False,
        )

    for param_name, controls in sweep_params.items():
        if param_name.startswith("driver_") and not is_bunch_to_bunch(sim_type):
            continue

        if not controls["sweep_var"].get():
            continue

        min_val = float(controls["min_var"].get())
        max_val = float(controls["max_var"].get())
        points = int(controls["points_var"].get())
        log_scale = controls["log_var"].get()

        if param_name == "driver_energy_gev":
            min_val = abs(min_val)
            max_val = abs(max_val)
            if min_val > max_val:
                min_val, max_val = max_val, min_val

        grids[param_name] = generate_parameter_range(
            min_val, max_val, points, log_scale
        )

    return grids


_CONFIG_SWEEP_PARAM_DEFS = [
    ("particle_mass_range", "particle_mass_points", "rider_m_particle", None),
    ("particle_charge_range", "particle_charge_points", "rider_charge_sign", None),
    ("particle_count_range", "particle_count_points", "rider_pcount", None),
    (
        "transverse_momentum_range",
        "transverse_momentum_points",
        "rider_transv_mom",
        "transverse_momentum_log_scale",
    ),
    (
        "transverse_spread_range",
        "transverse_spread_points",
        "rider_transv_dist",
        "transverse_spread_log_scale",
    ),
    (
        "rider_stripped_ions_range",
        "rider_stripped_ions_points",
        "rider_stripped_ions",
        None,
    ),
    (
        "macroparticle_charge_range",
        "macroparticle_charge_points",
        "macroparticle_charge_multiplier",
        None,
    ),
    (
        "macroparticle_sigma_range",
        "macroparticle_sigma_points",
        "macroparticle_sigma_multiplier",
        None,
    ),
    ("driver_mass_range", "driver_mass_points", "driver_m_particle", None),
    (
        "driver_charge_sign_range",
        "driver_charge_sign_points",
        "driver_charge_sign",
        None,
    ),
    ("driver_pcount_range", "driver_pcount_points", "driver_pcount", None),
    (
        "driver_transv_mom_range",
        "driver_transv_mom_points",
        "driver_transv_mom",
        "driver_transv_mom_log_scale",
    ),
    (
        "driver_transv_dist_range",
        "driver_transv_dist_points",
        "driver_transv_dist",
        "driver_transv_dist_log_scale",
    ),
    (
        "driver_starting_distance_range",
        "driver_starting_distance_points",
        "driver_starting_distance",
        "driver_starting_distance_log_scale",
    ),
    (
        "driver_energy_range",
        "driver_energy_points",
        "driver_energy_gev",
        "driver_energy_log_scale",
    ),
    (
        "driver_stripped_ions_range",
        "driver_stripped_ions_points",
        "driver_stripped_ions",
        None,
    ),
]


def build_config_parameter_grids(config: Any) -> dict[str, list[float]]:
    """Build sweep grids from a dataclass config without GUI sweep controls."""
    grids: dict[str, list[float]] = {}
    sim_type = config.simulation_type

    if not is_bunch_to_bunch(sim_type):
        if config.aperture_points > 1:
            aper_min, aper_max = config.aperture_range
            grids["aperture"] = generate_parameter_range(
                aper_min,
                aper_max,
                config.aperture_points,
                config.aperture_log_scale,
            )
        else:
            grids["aperture"] = [config.aperture_range[0]]

    if config.energy_points > 1:
        e_min, e_max = config.energy_range
        grids["energy"] = generate_parameter_range(
            e_min,
            e_max,
            config.energy_points,
            config.energy_log_scale,
        )
    else:
        grids["energy"] = [config.energy_range[0]]

    if config.starting_z_positions and len(config.starting_z_positions) >= 1:
        grids["start_z"] = config.starting_z_positions
    elif config.starting_z_range is not None and config.starting_z_points > 1:
        grids["start_z"] = np.linspace(
            config.starting_z_range[0],
            config.starting_z_range[1],
            config.starting_z_points,
        ).tolist()
    else:
        grids["start_z"] = [config.wall_z - 100.0]

    if (
        config.transverse_offset_fractions
        and len(config.transverse_offset_fractions) >= 1
    ):
        grids["transv_offset_frac"] = [config.transverse_offset_fractions[0]]
    else:
        grids["transv_offset_frac"] = [0.0]

    if config.wall_z_range is not None and config.wall_z_points > 1:
        grids["wall_z"] = generate_parameter_range(
            config.wall_z_range[0],
            config.wall_z_range[1],
            config.wall_z_points,
            False,
        )

    for range_attr, points_attr, grid_key, log_attr in _CONFIG_SWEEP_PARAM_DEFS:
        rng = getattr(config, range_attr, None)
        if rng is None:
            continue
        pts = getattr(config, points_attr, 1)
        if pts <= 1:
            continue
        if grid_key.startswith("driver_") and not is_bunch_to_bunch(sim_type):
            continue
        log_scale = getattr(config, log_attr, False) if log_attr else False
        min_val, max_val = float(rng[0]), float(rng[1])
        grids[grid_key] = generate_parameter_range(min_val, max_val, pts, log_scale)

    return grids
