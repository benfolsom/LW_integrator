"""Discrete mechanical accounting for the CPU Boris correction push.

Cloud moments are projections of particles, not additional mechanical bodies.
This ledger accounts for the two electric half-kicks and magnetic rotation;
it does not supply the missing electromagnetic energy or momentum reservoir.
"""

from __future__ import annotations

import numpy as np

from .grid import Species
from .kernels import C


def kinetic_change(species: Species, before: np.ndarray, after: np.ndarray) -> float:
    """Population-weighted kinetic change, rationalized for small kicks."""
    gamma_sum = np.sqrt(1 + np.sum(before**2, axis=1)) + np.sqrt(
        1 + np.sum(after**2, axis=1)
    )
    return float(
        np.sum(
            species.population
            * species.mass_kg
            * C**2
            * np.sum((after - before) * (after + before), axis=1)
            / gamma_sum
        )
    )


def _stage_velocities(
    species: Species, after: np.ndarray, electric: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    u = species.momentum_mc
    half = species.charge_c * dt / (2 * species.mass_kg * C) * electric
    minus, plus = u + half, after - half

    def gamma(momentum: np.ndarray) -> np.ndarray:
        return np.sqrt(1 + np.sum(momentum**2, axis=1))[:, None]

    # Each electric kick's secant velocity gives its exact kinetic work.
    electric_velocity = (
        0.5
        * C
        * (
            (u + minus) / (gamma(u) + gamma(minus))
            + (plus + after) / (gamma(plus) + gamma(after))
        )
    )
    # Boris rotates at fixed gamma(minus). This is the rotation's force
    # velocity, which need not equal the electric-work secant velocity.
    rotation_velocity = 0.5 * C * (minus + plus) / gamma(minus)
    return electric_velocity, rotation_velocity


def coupled_kick_ledger(
    species: Species,
    baseline_u: np.ndarray,
    corrected_u: np.ndarray,
    electric: np.ndarray,
    magnetic: np.ndarray,
    correction_electric: np.ndarray,
    correction_magnetic: np.ndarray,
    dt: float,
) -> dict:
    """Close corrected-minus-baseline kicks, including the PIC response.

    Baseline fields also do different work/impulse when the correction changes
    the Boris stage velocities. Omitting this mixed contribution is incorrect
    at finite timestep. Both field integrals are evaluated independently of
    the measured momentum and kinetic increments.
    """
    vb_e, vb_b = _stage_velocities(species, baseline_u, electric, dt)
    vc_e, vc_b = _stage_velocities(
        species, corrected_u, electric + correction_electric, dt
    )
    charge = species.source_charge_c

    def work(field: np.ndarray, velocity: np.ndarray) -> float:
        return float(dt * np.sum(charge * np.sum(field * velocity, axis=1)))

    def impulse(force_per_charge: np.ndarray) -> np.ndarray:
        return dt * np.sum(charge[:, None] * force_per_charge, axis=0)

    correction_work = work(correction_electric, vc_e)
    response_work = work(electric, vc_e - vb_e)
    correction_impulse = impulse(
        correction_electric + np.cross(vc_b, correction_magnetic)
    )
    response_impulse = impulse(np.cross(vc_b - vb_b, magnetic))
    kick_energy = kinetic_change(species, baseline_u, corrected_u)
    kick_momentum = np.sum(
        species.population[:, None] * species.mass_kg * C * (corrected_u - baseline_u),
        axis=0,
    )
    total_work = work(electric + correction_electric, vc_e)
    total_impulse = impulse(
        electric + correction_electric + np.cross(vc_b, magnetic + correction_magnetic)
    )
    return dict(
        kick_energy_j=kick_energy,
        correction_work_j=correction_work,
        pic_response_work_j=response_work,
        work_residual_j=correction_work + response_work - kick_energy,
        kick_momentum_kg_m_s=kick_momentum.tolist(),
        correction_impulse_kg_m_s=correction_impulse.tolist(),
        pic_response_impulse_kg_m_s=response_impulse.tolist(),
        impulse_residual_kg_m_s=(
            correction_impulse + response_impulse - kick_momentum
        ).tolist(),
        total_work_j=total_work,
        total_kick_energy_j=kinetic_change(species, species.momentum_mc, corrected_u),
        total_impulse_kg_m_s=total_impulse.tolist(),
        total_kick_momentum_kg_m_s=np.sum(
            species.population[:, None]
            * species.mass_kg
            * C
            * (corrected_u - species.momentum_mc),
            axis=0,
        ).tolist(),
    )
