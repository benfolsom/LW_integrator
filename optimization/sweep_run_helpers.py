"""Pure helpers for preparing one sweep integration run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from core.constants import C_MMNS
from optimization.config import calculate_auto_steps, calculate_auto_timestep
from optimization.run_parameter_helpers import calculate_transverse_offset
from optimization.simulation_type_helpers import is_bunch_to_bunch
from optimization.sweep_helpers import AMU_TO_MEV, calculate_starting_pz_from_energy


@dataclass(frozen=True)
class SweepRunParameters:
    """Resolved physical and control parameters for one sweep grid point."""

    aperture: float
    energy: float
    start_z: float
    offset_frac: float
    transv_offset: float
    rider_m_particle: float
    rider_charge_sign: float
    rider_pcount: int
    rider_transv_mom: float
    rider_transv_dist: float
    rider_stripped_ions: float
    macroparticle_charge_multiplier: float
    macroparticle_sigma_multiplier: float
    driver_params: dict[str, Any] | None


@dataclass(frozen=True)
class SweepTimestepResolution:
    """Resolved timestep/step count plus diagnostics for one run."""

    timestep: float
    steps: int
    expected_distance: float
    log_lines: list[str]


def resolve_sweep_run_parameters(
    config: Any,
    params_dict: Mapping[str, Any],
) -> SweepRunParameters | None:
    """Resolve physical parameters for one sweep combination."""
    energy = params_dict.get("initial_energy_gev") or params_dict.get("energy")
    if energy is None:
        return None

    aperture = params_dict.get("aperture", 0.001)
    start_z = params_dict["start_z"]
    offset_frac = params_dict["transverse_offset_fraction"]
    rider_m_particle = params_dict.get("rider_m_particle", config.m_particle)
    rider_charge_sign = params_dict.get("rider_charge_sign", config.charge_sign)
    rider_pcount = params_dict.get("rider_pcount", config.pcount)
    rider_transv_mom = params_dict.get("rider_transv_mom", config.transv_mom)
    rider_transv_dist = params_dict.get("rider_transv_dist", config.transv_dist)
    rider_stripped_ions = params_dict.get("rider_stripped_ions", config.stripped_ions)
    macroparticle_charge_multiplier = params_dict.get(
        "macroparticle_charge_multiplier",
        config.macroparticle_charge_multiplier,
    )
    macroparticle_sigma_multiplier = params_dict.get(
        "macroparticle_sigma_multiplier",
        config.macroparticle_sigma_multiplier,
    )
    driver_params = _resolve_driver_params(config, params_dict)
    transv_offset = calculate_transverse_offset(
        config.simulation_type, offset_frac, aperture
    )

    return SweepRunParameters(
        aperture=aperture,
        energy=energy,
        start_z=start_z,
        offset_frac=offset_frac,
        transv_offset=transv_offset,
        rider_m_particle=rider_m_particle,
        rider_charge_sign=rider_charge_sign,
        rider_pcount=int(rider_pcount),
        rider_transv_mom=rider_transv_mom,
        rider_transv_dist=rider_transv_dist,
        rider_stripped_ions=rider_stripped_ions,
        macroparticle_charge_multiplier=macroparticle_charge_multiplier,
        macroparticle_sigma_multiplier=macroparticle_sigma_multiplier,
        driver_params=driver_params,
    )


def resolve_sweep_timestep(
    config: Any,
    params_dict: Mapping[str, Any],
    run_params: SweepRunParameters,
    *,
    run_num: int,
    use_full_debug: bool,
) -> SweepTimestepResolution:
    """Resolve timestep and step count for one sweep run."""
    timestep = config.timestep
    steps = config.steps
    expected_distance = _expected_distance(
        config.simulation_type,
        energy=run_params.energy,
        mass_amu=run_params.rider_m_particle,
        timestep=timestep,
        steps=steps,
    )
    log_lines: list[str] = []

    if config.timestep_strategy != "fixed":
        current_wall_z = params_dict.get("wall_z", config.wall_z)
        driver_start_z = 1000.0
        if run_params.driver_params is not None:
            driver_start_z = run_params.driver_params.get("starting_distance", 1000.0)

        timestep = config.calculate_timestep_for_energy(
            run_params.energy,
            run_params.rider_m_particle,
            wall_z=current_wall_z,
            start_z=run_params.start_z,
            driver_start_z=driver_start_z,
        )
        steps = config.steps
        expected_distance = _expected_distance(
            config.simulation_type,
            energy=run_params.energy,
            mass_amu=run_params.rider_m_particle,
            timestep=timestep,
            steps=steps,
        )
        if use_full_debug:
            log_lines.extend(
                _timestep_log_lines(
                    config,
                    params_dict,
                    run_params,
                    run_num=run_num,
                    timestep=timestep,
                    steps=steps,
                    expected_distance=expected_distance,
                )
            )
    elif config.auto_steps:
        current_wall_z = params_dict.get("wall_z", config.wall_z)
        timestep = calculate_auto_timestep(
            start_z=run_params.start_z,
            wall_z=current_wall_z,
            distance_past_wall=config.auto_steps_distance_past_wall,
            particle_energy_gev=run_params.energy,
            particle_mass_amu=run_params.rider_m_particle,
            target_steps=config.auto_steps_target,
        )
        steps = calculate_auto_steps(
            start_z=run_params.start_z,
            wall_z=current_wall_z,
            distance_past_wall=config.auto_steps_distance_past_wall,
            timestep=timestep,
            particle_energy_gev=run_params.energy,
            particle_mass_amu=run_params.rider_m_particle,
        )
        expected_distance = _expected_distance(
            config.simulation_type,
            energy=run_params.energy,
            mass_amu=run_params.rider_m_particle,
            timestep=timestep,
            steps=steps,
        )

    min_steps = max(20, int(config.steps * 0.05))
    if steps < min_steps:
        if use_full_debug:
            log_lines.append(
                f"  [WARNING] Steps adjusted from {steps} to {min_steps} "
                "(minimum floor)"
            )
        steps = min_steps
        expected_distance = _expected_distance(
            config.simulation_type,
            energy=run_params.energy,
            mass_amu=run_params.rider_m_particle,
            timestep=timestep,
            steps=steps,
        )

    return SweepTimestepResolution(
        timestep=timestep,
        steps=steps,
        expected_distance=expected_distance,
        log_lines=log_lines,
    )


def build_full_debug_parameter_log_lines(
    config: Any,
    run_params: SweepRunParameters,
    *,
    run_num: int,
    total_runs: int,
    params_dict: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return verbose parameter logging for one sweep run."""
    params_dict = params_dict or {}
    log_lines = [
        f"  [PARAMS] Run {run_num}/{total_runs} - All parameters:",
        f"    aperture: {run_params.aperture:.4e} mm",
        f"    energy: {run_params.energy:.4f} GeV",
        f"    start_z: {run_params.start_z:.4f} mm",
        f"    transv_offset_frac: {run_params.offset_frac:.4f}",
        f"    rider_m_particle: {run_params.rider_m_particle:.4e} amu",
        f"    rider_charge_sign: {run_params.rider_charge_sign:.1f}",
        f"    rider_pcount: {run_params.rider_pcount}",
        f"    rider_transv_mom: {run_params.rider_transv_mom:.4e} amu·mm/ns",
        f"    rider_transv_dist: {run_params.rider_transv_dist:.4e} mm",
        f"    rider_stripped_ions: {run_params.rider_stripped_ions:.2e}",
    ]
    if config.macroparticle_enabled:
        log_lines.extend(
            [
                "    macroparticle_enabled: True",
                (
                    "    macroparticle_charge_multiplier: "
                    f"{run_params.macroparticle_charge_multiplier:.4f}"
                ),
                (
                    "    macroparticle_sigma_multiplier: "
                    f"{run_params.macroparticle_sigma_multiplier:.4f}"
                ),
                (
                    "    macroparticle_use_momentum_errors: "
                    f"{config.macroparticle_use_momentum_errors}"
                ),
            ]
        )
    if run_params.driver_params is not None:
        driver = run_params.driver_params
        driver_energy_gev = params_dict.get(
            "driver_energy_gev", getattr(config, "driver_energy_gev", 0.0)
        )
        log_lines.extend(
            [
                f"    driver_m_particle: {driver['m_particle']:.4e} amu",
                f"    driver_charge_sign: {driver['charge_sign']:.1f}",
                f"    driver_pcount: {driver['pcount']}",
                f"    driver_transv_mom: {driver['transv_mom']:.4e} amu·mm/ns",
                f"    driver_transv_dist: {driver['transv_dist']:.4e} mm",
                (
                    "    driver_starting_distance: "
                    f"{driver['starting_distance']:.4f} mm"
                ),
                f"    driver_energy_gev: {driver_energy_gev:.4f} GeV",
                f"    driver_stripped_ions: {driver['stripped_ions']:.2e}",
            ]
        )
    return log_lines


def _resolve_driver_params(
    config: Any,
    params_dict: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not is_bunch_to_bunch(config.simulation_type):
        return None

    driver_m = params_dict.get("driver_m_particle", config.driver_m_particle)
    driver_neg = getattr(config, "driver_direction", "-z") == "-z"
    if "driver_energy_gev" in params_dict:
        driver_pz = calculate_starting_pz_from_energy(
            abs(params_dict["driver_energy_gev"]),
            driver_m,
            negative=driver_neg,
        )
    else:
        driver_pz = params_dict.get("driver_starting_Pz", config.driver_starting_Pz)

    return {
        "m_particle": driver_m,
        "charge_sign": params_dict.get("driver_charge_sign", config.driver_charge_sign),
        "pcount": int(params_dict.get("driver_pcount", config.driver_pcount)),
        "transv_mom": params_dict.get("driver_transv_mom", config.driver_transv_mom),
        "transv_dist": params_dict.get("driver_transv_dist", config.driver_transv_dist),
        "starting_distance": params_dict.get(
            "driver_starting_distance",
            config.driver_starting_distance,
        ),
        "starting_Pz": driver_pz,
        "stripped_ions": params_dict.get(
            "driver_stripped_ions", config.driver_stripped_ions
        ),
    }


def _expected_distance(
    simulation_type: Any,
    *,
    energy: float,
    mass_amu: float,
    timestep: float,
    steps: int,
) -> float:
    gamma = _gamma_for_energy(simulation_type, energy, mass_amu)
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.999
    return beta * gamma * C_MMNS * timestep * steps


def _gamma_for_energy(simulation_type: Any, energy: float, mass_amu: float) -> float:
    rest_energy_mev = mass_amu * AMU_TO_MEV
    if is_bunch_to_bunch(simulation_type):
        return (energy * 1e3) / rest_energy_mev + 1.0
    return (energy * 1e3) / rest_energy_mev


def _timestep_log_lines(
    config: Any,
    params_dict: Mapping[str, Any],
    run_params: SweepRunParameters,
    *,
    run_num: int,
    timestep: float,
    steps: int,
    expected_distance: float,
) -> list[str]:
    gamma = _gamma_for_energy(
        config.simulation_type, run_params.energy, run_params.rider_m_particle
    )
    beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.999
    distance_per_step = beta * gamma * C_MMNS * timestep
    current_wall_z = params_dict.get("wall_z", config.wall_z)
    log_lines = [
        f"  [TIMESTEP] Run {run_num} strategy '{config.timestep_strategy}':",
        f"    E={run_params.energy:.4f} GeV, m={run_params.rider_m_particle:.4e} amu",
        f"    gamma={gamma:.2f}, beta={beta:.8f}",
        f"    timestep h={timestep:.4e} ns (proper time = dt/gamma)",
        f"    steps={steps}",
        f"    distance_per_step = β·γ·c·h = {distance_per_step:.4f} mm",
        f"    expected_total_distance = {expected_distance:.2f} mm",
        f"    wall_z={current_wall_z:.2f} mm, start_z={run_params.start_z:.2f} mm",
        f"    distance_to_wall = {abs(current_wall_z - run_params.start_z):.2f} mm",
    ]
    if config.timestep_strategy == "auto_distance":
        log_lines.append(f"    target_distance={config.target_distance_mm:.2f} mm")
    return log_lines


__all__ = [
    "SweepRunParameters",
    "SweepTimestepResolution",
    "build_full_debug_parameter_log_lines",
    "resolve_sweep_run_parameters",
    "resolve_sweep_timestep",
]
