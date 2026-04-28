"""Pure helpers for optimization run-preparation and safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization.plugin_config_helpers import parse_float_list
from optimization.simulation_type_helpers import is_bunch_to_bunch
from optimization.sweep_helpers import AMU_TO_MEV

_ELECTRON_MASS_AMU = 0.00054857990907
_PROTON_MASS_AMU = 1.007276466621


@dataclass(frozen=True)
class SweepParameterValidationInput:
    """UI-independent snapshot of one sweepable sub-parameter."""

    name: str
    swept: bool
    min_value: Any
    max_value: Any
    points: Any
    fixed_value: Any


def validate_optimization_inputs(
    *,
    simulation_type: Any,
    aperture_min: Any,
    aperture_max: Any,
    aperture_points: Any,
    energy_min: Any,
    energy_max: Any,
    energy_points: Any,
    mode: str,
    offset_fractions: str,
    start_z: Any,
    wall_z: Any,
    steps: Any,
    auto_steps_distance: Any,
    sweep_parameters: list[SweepParameterValidationInput],
) -> str | None:
    """Validate optimization run inputs without depending on Tk widgets."""

    try:
        b2b_mode = is_bunch_to_bunch(simulation_type)

        energy_points_int = int(energy_points)
        error = _validate_aperture_range(
            b2b_mode=b2b_mode,
            aperture_min=aperture_min,
            aperture_max=aperture_max,
        )
        if error is not None:
            return error

        error = _validate_energy_range(
            b2b_mode=b2b_mode,
            energy_min=energy_min,
            energy_max=energy_max,
            energy_points=energy_points_int,
        )
        if error is not None:
            return error

        error = _validate_grid_point_counts(
            b2b_mode=b2b_mode,
            mode=mode,
            aperture_points=aperture_points,
            energy_points=energy_points_int,
            sweep_parameters=sweep_parameters,
        )
        if error is not None:
            return error

        error = _validate_scalar_run_fields(
            offset_fractions=offset_fractions,
            start_z=start_z,
            wall_z=wall_z,
            steps=steps,
            auto_steps_distance=auto_steps_distance,
        )
        if error is not None:
            return error

        return _validate_sweep_parameters(sweep_parameters)
    except ValueError as exc:
        return f"Invalid input: {exc}"


def _validate_aperture_range(
    *, b2b_mode: bool, aperture_min: Any, aperture_max: Any
) -> str | None:
    if b2b_mode:
        return None

    aperture_min_float = float(aperture_min)
    aperture_max_float = float(aperture_max)
    if aperture_min_float >= aperture_max_float:
        return "Aperture min must be less than max"
    if aperture_min_float <= 0:
        return "Aperture min must be positive"
    return None


def _validate_energy_range(
    *, b2b_mode: bool, energy_min: Any, energy_max: Any, energy_points: int
) -> str | None:
    energy_min_float = float(energy_min)
    energy_max_float = float(energy_max)
    if b2b_mode and energy_points == 1:
        if energy_min_float <= 0:
            return "Rider energy must be positive"
        return None

    if energy_min_float >= energy_max_float:
        return "Energy min must be less than max"
    if energy_min_float <= 0:
        return "Energy min must be positive"
    return None


def _validate_grid_point_counts(
    *,
    b2b_mode: bool,
    mode: str,
    aperture_points: Any,
    energy_points: int,
    sweep_parameters: list[SweepParameterValidationInput],
) -> str | None:
    if mode == "blind_sweep":
        has_swept_sub_param = any(param.swept for param in sweep_parameters)
        if not b2b_mode:
            aperture_points_int = int(aperture_points)
            if aperture_points_int < 2:
                return "Sweep mode: Aperture must have at least 2 points"
        if energy_points < 2 and not has_swept_sub_param:
            return (
                "Sweep mode: Energy must have at least 2 points "
                "(or enable a swept sub-parameter)"
            )
        return None

    if not b2b_mode:
        aperture_points_int = int(aperture_points)
        if aperture_points_int < 1:
            return "Aperture must have at least 1 point"
    if energy_points < 1:
        return "Energy must have at least 1 point"
    return None


def _validate_scalar_run_fields(
    *,
    offset_fractions: str,
    start_z: Any,
    wall_z: Any,
    steps: Any,
    auto_steps_distance: Any,
) -> str | None:
    parse_float_list(offset_fractions)
    float(start_z)
    float(wall_z)
    steps_int = int(steps)
    if steps_int < 100:
        return "Steps must be at least 100"

    distance_past_wall = float(auto_steps_distance)
    if distance_past_wall < 0:
        return "Distance past wall must be non-negative"
    return None


def _validate_sweep_parameters(
    sweep_parameters: list[SweepParameterValidationInput],
) -> str | None:
    for param in sweep_parameters:
        if param.swept:
            min_val = float(param.min_value)
            max_val = float(param.max_value)
            points = int(param.points)

            if min_val >= max_val:
                return f"{param.name}: min must be less than max"
            if points < 2:
                return f"{param.name}: must have at least 2 points"
        else:
            fixed_val = float(param.fixed_value)
            if "m_particle" in param.name and fixed_val <= 0:
                return f"{param.name}: Particle mass must be positive"
            if "pcount" in param.name and int(fixed_val) < 1:
                return f"{param.name}: Particle count must be at least 1"
    return None


def build_extreme_parameter_warning(config: Any) -> str | None:
    """Return a user-facing warning for risky run-control settings, if any."""
    warnings = []

    aperture_min = config.aperture_range[0]
    energy_max = config.energy_range[1]
    rest_energy_mev = config.m_particle * AMU_TO_MEV
    gamma_max = _max_gamma_for_energy(
        config.simulation_type,
        energy_max=energy_max,
        rest_energy_mev=rest_energy_mev,
    )
    threshold = _extreme_threshold_for_mass(
        config.m_particle, rest_energy_mev=rest_energy_mev
    )

    if aperture_min < 1e-5 and gamma_max > 10000:
        warnings.append(
            f"• Very small aperture ({aperture_min:.2e} mm) with high energy ({energy_max:.1f} GeV, γ≈{gamma_max:.0f})\n"
            f"  This may cause extreme fields, SC convergence issues, and very slow runs."
        )

    if aperture_min < 1e-6:
        warnings.append(
            f"• Aperture < 1 μm detected ({aperture_min:.2e} mm)\n"
            f"  Sub-micron apertures often cause numerical instabilities."
        )

    if gamma_max > threshold.gamma:
        warnings.append(
            f"• Very high energy detected ({energy_max:.1f} GeV, γ≈{gamma_max:.0f})\n"
            f"  Exceeds recommended threshold for {threshold.particle_type}s (~{threshold.energy_tev:.1f} TeV)\n"
            f"  Ultra-relativistic particles may require very fine timesteps."
        )

    if not config.auto_steps:
        timestep_warning = _fixed_timestep_warning(
            aperture_min=aperture_min,
            gamma_max=gamma_max,
            timestep=config.timestep,
        )
        if timestep_warning is not None:
            warnings.append(timestep_warning)

    if not warnings:
        return None

    warning_text = "Extreme parameter combinations detected:\n\n" + "\n\n".join(
        warnings
    )
    warning_text += "\n\nRecommendations:\n"
    warning_text += "• Enable 'Per-run timeout' to prevent hangs\n"
    warning_text += "• Enable 'Skip failed runs' to complete the sweep\n"
    warning_text += "• Consider more moderate parameter ranges for initial sweeps\n"
    warning_text += "\nDo you want to proceed anyway?"
    return warning_text


def _max_gamma_for_energy(
    simulation_type: Any, *, energy_max: float, rest_energy_mev: float
) -> float:
    gamma = (energy_max * 1e3) / rest_energy_mev
    return gamma + 1.0 if is_bunch_to_bunch(simulation_type) else gamma


@dataclass(frozen=True)
class _ExtremeThreshold:
    gamma: int
    particle_type: str
    energy_tev: float


def _extreme_threshold_for_mass(
    particle_mass_amu: float, *, rest_energy_mev: float
) -> _ExtremeThreshold:
    if abs(particle_mass_amu - _ELECTRON_MASS_AMU) < 1e-6:
        return _ExtremeThreshold(1_956_000, "electron", 1.0)
    if abs(particle_mass_amu - _PROTON_MASS_AMU) < 1e-3:
        return _ExtremeThreshold(21_300, "proton", 20.0)

    gamma = int(21_300 * _PROTON_MASS_AMU / particle_mass_amu)
    return _ExtremeThreshold(gamma, "particle", gamma * rest_energy_mev / 1e6)


def _fixed_timestep_warning(
    *, aperture_min: float, gamma_max: float, timestep: float
) -> str | None:
    beta_approx = 1.0 if gamma_max > 2 else 0.9
    distance_per_step = beta_approx * gamma_max * 300.0 * timestep
    if distance_per_step <= aperture_min * 0.1:
        return None
    return (
        f"• Fixed timestep may be too large for small apertures\n"
        f"  Distance/step ≈ {distance_per_step:.3f} mm vs aperture {aperture_min:.2e} mm\n"
        f"  Consider enabling 'Auto timestep' or reducing timestep."
    )


__all__ = [
    "SweepParameterValidationInput",
    "build_extreme_parameter_warning",
    "validate_optimization_inputs",
]
