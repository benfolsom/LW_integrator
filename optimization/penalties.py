"""Penalty heuristics for optimization sweeps."""

from __future__ import annotations

from typing import Any

from optimization.config import (
    _ELECTRON_ENERGY_THRESHOLD_GEV,
    _ELECTRON_MASS_AMU,
    _ENERGY_THRESHOLD_EXPONENT,
    _ENERGY_THRESHOLD_SCALE,
    _PROTON_ENERGY_THRESHOLD_GEV,
    _PROTON_MASS_AMU,
)


def compute_soft_penalty(
    config: Any,
    *,
    aperture_radius: float,
    macroparticle_charge_multiplier: float,
    initial_energy_gev: float,
) -> float:
    """Estimate a soft penalty for risky parameter combinations.

    Small apertures combined with very high charge multipliers and beam energies
    almost always trigger gamma blow-ups. Rather than rejecting those points
    outright, apply a tunable penalty so the optimizer learns to avoid them
    while keeping the search numerically stable. The energy threshold adapts
    to the current particle mass so protons (or heavier ions) are not
    penalized until multi-TeV energies, while electrons still start getting
    nudged away once they exceed roughly 120 GeV.
    """

    penalty = 0.0

    aperture_threshold_mm = 0.01  # 10 microns
    charge_threshold = 800.0
    penalty_scale = 1.0e-3  # keeps penalty on the same order as metrics

    particle_mass_amu = max(
        float(getattr(config, "m_particle", _ELECTRON_MASS_AMU)),
        _ELECTRON_MASS_AMU * 0.1,
    )
    energy_threshold = _ENERGY_THRESHOLD_SCALE * (
        particle_mass_amu**_ENERGY_THRESHOLD_EXPONENT
    )
    energy_threshold = max(_ELECTRON_ENERGY_THRESHOLD_GEV * 0.25, energy_threshold)

    if particle_mass_amu >= _PROTON_MASS_AMU:
        energy_threshold = min(energy_threshold, _PROTON_ENERGY_THRESHOLD_GEV)

    small_aperture_factor = max(
        0.0, (aperture_threshold_mm - aperture_radius) / aperture_threshold_mm
    )
    high_charge_factor = max(
        0.0,
        (macroparticle_charge_multiplier - charge_threshold) / charge_threshold,
    )

    if small_aperture_factor > 0 and high_charge_factor > 0:
        penalty += small_aperture_factor * high_charge_factor

    if high_charge_factor > 0 and initial_energy_gev > energy_threshold:
        energy_factor = (initial_energy_gev - energy_threshold) / energy_threshold
        tight_aperture_factor = max(0.0, (0.1 - aperture_radius) / 0.1)
        penalty += 0.5 * energy_factor * high_charge_factor * tight_aperture_factor

    return max(0.0, penalty * penalty_scale)


__all__ = ["compute_soft_penalty"]
