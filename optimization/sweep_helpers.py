"""Shared helper functions for optimization sweeps and UI state."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from core.constants import C_MMNS  # type: ignore[import]
from core.types import SimulationType  # type: ignore[import]

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
    if log_scale:
        return np.logspace(np.log10(min_val), np.log10(max_val), points).tolist()
    return np.linspace(min_val, max_val, points).tolist()


def build_parameter_grids(config: Any, sweep_params: Mapping[str, Any]) -> dict[str, list]:
    """Build sweep grids from config and UI sweep control state."""
    grids: dict[str, list] = {}
    sim_type = config.simulation_type

    if sim_type != SimulationType.BUNCH_TO_BUNCH:
        grids["aperture"] = generate_parameter_range(
            config.aperture_range[0],
            config.aperture_range[1],
            config.aperture_points,
            config.aperture_log_scale,
        )

    energy_key = (
        "initial_energy_gev"
        if sim_type == SimulationType.BUNCH_TO_BUNCH
        else "energy"
    )
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
        if (
            param_name.startswith("driver_")
            and sim_type != SimulationType.BUNCH_TO_BUNCH
        ):
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
