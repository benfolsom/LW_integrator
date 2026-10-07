"""CPU Medina adapter with accepted full-force history in lab time.

The force sample is the complete non-RR Boris impulse divided by lab dt.
Successive samples follow each material particle, so their causal secant
includes motion, evolving PIC frames, and the complete LW correction. This
is a first-order derivative estimate, like the maintained LW integrator's
accepted-force prescription; it is not an analytic field derivative.
"""

from __future__ import annotations

import numpy as np

from ..external_fields import AMU_KG, NATIVE_FORCE_UNIT_NEWTON
from ..medina_radiation_reaction import compute_medina_radiation_reaction
from .coupling import kinetic_change
from .grid import Species
from .kernels import C, COULOMB

NATIVE_ENERGY_J = AMU_KG * 1e12
NATIVE_MOMENTUM_KG_M_S = AMU_KG * 1e6


class MedinaPIC:
    """Accepted force samples; no synthetic prehistory, clipping, or macro q²."""

    def __init__(self) -> None:
        self.previous: dict[str, tuple[float, np.ndarray]] = {}
        self.last_samples: dict[str, dict] = {}

    def apply(
        self, species: Species, non_rr_u: np.ndarray, sample_time_s: float, dt: float
    ) -> tuple[np.ndarray, dict]:
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(sample_time_s):
            raise ValueError("Medina needs finite sample time and positive lab dt")
        force = species.mass_kg * C * (non_rr_u - species.momentum_mc) / dt
        previous = self.previous.get(species.name)
        ready = previous is not None
        derivative = np.zeros_like(force)
        if previous is not None:
            previous_time, previous_force = previous
            if sample_time_s <= previous_time or previous_force.shape != force.shape:
                raise ValueError(
                    "Medina samples must advance with fixed particle identities"
                )
            derivative = (force - previous_force) / (sample_time_s - previous_time)
        if not np.all(np.isfinite(force)) or not np.all(np.isfinite(derivative)):
            raise ValueError("Medina full-force history must be finite")
        gamma = np.sqrt(1 + np.sum(non_rr_u**2, axis=1))
        beta = non_rr_u / gamma[:, None]
        acceleration = (force - beta * np.sum(beta * force, axis=1)[:, None]) / (
            species.mass_kg * gamma[:, None]
        )
        rr_force = np.zeros_like(force)
        radiated_impulse = np.zeros_like(force)
        cross_impulse = np.zeros_like(force)
        model_work = np.zeros(len(force))
        radiation = np.zeros(len(force))
        cross_change = np.zeros(len(force))
        cross_energy = np.zeros(len(force))
        cross_momentum = np.zeros_like(force)
        # Match the SI Coulomb coefficient exactly in Gaussian native units.
        # This charge is physical q_s, never N*q_s or a K-cloud charge.
        charge_native = species.charge_c * np.sqrt(COULOMB / (AMU_KG * 1e9))
        for i in range(len(force)):
            result = compute_medina_radiation_reaction(
                external_force=force[i] / NATIVE_FORCE_UNIT_NEWTON,
                external_force_time_derivative=derivative[i]
                * 1e-9
                / NATIVE_FORCE_UNIT_NEWTON,
                beta=beta[i],
                acceleration=acceleration[i] * 1e-15,
                gamma=float(gamma[i]),
                mass=species.mass_kg / AMU_KG,
                charge=charge_native,
                coordinate_dt=dt * 1e9,
            )
            cross_energy[i] = result.cross_field_energy * NATIVE_ENERGY_J
            cross_momentum[i] = (
                np.array(result.cross_field_momentum) * NATIVE_MOMENTUM_KG_M_S
            )
            radiation[i] = result.far_radiated_energy * NATIVE_ENERGY_J
            if ready:
                rr_force[i] = (
                    np.array(result.radiation_reaction_force) * NATIVE_FORCE_UNIT_NEWTON
                )
                model_work[i] = result.reaction_work * NATIVE_ENERGY_J
                cross_change[i] = result.cross_field_energy_change * NATIVE_ENERGY_J
                radiated_impulse[i] = (
                    np.array(result.radiated_momentum_rate)
                    * NATIVE_FORCE_UNIT_NEWTON
                    * dt
                )
                cross_impulse[i] = (
                    -result.response_time
                    * np.array(result.gamma_force_time_derivative)
                    * NATIVE_FORCE_UNIT_NEWTON
                    * dt
                )
        after = non_rr_u + dt * rr_force / (species.mass_kg * C)
        if not np.all(np.isfinite(after)):
            raise ValueError(
                "Medina produced non-finite momentum; no impulse was clipped"
            )
        gamma_after = np.sqrt(1 + np.sum(after**2, axis=1))
        work_velocity = C * (non_rr_u + after) / (gamma + gamma_after)[:, None]
        weights = species.population

        def scalar(values: np.ndarray) -> float:
            return float(np.sum(weights * values))

        def vector(values: np.ndarray) -> np.ndarray:
            return np.sum(weights[:, None] * values, axis=0)

        applied_work = scalar(dt * np.sum(rr_force * work_velocity, axis=1))
        applied_impulse = vector(dt * rr_force)
        kinetic = kinetic_change(species, non_rr_u, after)
        kick_momentum = vector(species.mass_kg * C * (after - non_rr_u))
        model = scalar(model_work)
        far = scalar(radiation) if ready else 0.0
        cross = scalar(cross_change)
        far_impulse, cross_momentum_change = vector(radiated_impulse), vector(
            cross_impulse
        )
        ledger = dict(
            kick_energy_j=kinetic,
            applied_work_j=applied_work,
            work_residual_j=applied_work - kinetic,
            model_work_j=model,
            far_radiated_energy_j=far,
            cross_field_energy_change_j=cross,
            model_energy_residual_j=model + far + cross,
            work_quadrature_difference_j=applied_work - model,
            priming_far_radiated_energy_j=scalar(radiation) if not ready else 0.0,
            kick_momentum_kg_m_s=kick_momentum.tolist(),
            applied_impulse_kg_m_s=applied_impulse.tolist(),
            impulse_residual_kg_m_s=(applied_impulse - kick_momentum).tolist(),
            far_radiated_impulse_kg_m_s=far_impulse.tolist(),
            cross_field_momentum_change_kg_m_s=cross_momentum_change.tolist(),
            model_momentum_residual_kg_m_s=(
                applied_impulse + far_impulse + cross_momentum_change
            ).tolist(),
        )
        self.previous[species.name] = sample_time_s, force.copy()
        self.last_samples[species.name] = dict(
            time_s=sample_time_s,
            derivative_ready=ready,
            cross_field_energy_j=scalar(cross_energy),
            cross_field_momentum_kg_m_s=vector(cross_momentum).tolist(),
            force_norm_n=float(np.linalg.norm(force)),
            force_derivative_norm_n_s=float(np.linalg.norm(derivative)),
            response_time_s=result.response_time * 1e-9,
        )
        return after, ledger
