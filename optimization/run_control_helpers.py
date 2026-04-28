"""Pure helpers for optimization run-preparation and safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimization.simulation_type_helpers import is_bunch_to_bunch
from optimization.sweep_helpers import AMU_TO_MEV

_ELECTRON_MASS_AMU = 0.00054857990907
_PROTON_MASS_AMU = 1.007276466621


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


__all__ = ["build_extreme_parameter_warning"]
