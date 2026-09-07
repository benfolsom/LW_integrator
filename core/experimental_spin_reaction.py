"""Opt-in first-order-in-spin recoil, separate from Medina and magnetic-squared terms.

The force is Jakobsen (2024), Eqs. (19),(20a), evaluated with the existing
RFS-based reduction. This is an experimental hybrid, not a claim that the
non-self RFS equations equal the paper's full non-self model. At this order
the added rest-frame self-torque is zero. A rotation-free velocity change
transports spin so the velocity/spin constraints remain consistent.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from .constants import C_MMNS
from .magnetic_dipole import boost_rest_polarization, rest_polarization_from_four_vector

if TYPE_CHECKING:
    from .spin_self_force_reduction_history import AcceptedIntrinsicSpinReductionHistory

VectorLike = Sequence[float] | np.ndarray
_METRIC = np.array([1.0, -1.0, -1.0, -1.0])


def transport_spin_between_velocities(
    spin: VectorLike, before: VectorLike, after: VectorLike
) -> np.ndarray:
    """Rotation-free Lorentz transport between two future unit four-velocities.

    Unlike component projection, this map preserves the spin norm as well
    as orthogonality to the new velocity. The infinitesimal limit is the
    acceleration-induced Fermi--Walker term used by the existing RFS helper.
    """
    spin, before, after = (np.asarray(v, dtype=float) for v in (spin, before, after))
    if any(
        v.shape != (4,) or not np.all(np.isfinite(v)) for v in (spin, before, after)
    ):
        raise ValueError("finite four-vectors required for spin transport")
    if before[0] <= 0 or after[0] <= 0:
        raise ValueError("future-directed velocities required")
    for velocity in (before, after):
        if abs(np.dot(velocity * _METRIC, velocity) - C_MMNS**2) > 1e-11 * np.dot(
            velocity, velocity
        ):
            raise ValueError("spin transport requires on-shell four-velocities")
    if abs(np.dot(spin * _METRIC, before)) > 1e-11 * max(
        np.linalg.norm(spin) * np.linalg.norm(before), np.finfo(float).tiny
    ):
        raise ValueError("initial spin is not orthogonal to velocity")
    if np.array_equal(before, after):
        return spin.copy()
    denominator = C_MMNS**2 + np.dot(before * _METRIC, after)
    if denominator <= 0:
        raise ValueError("invalid relative velocity in spin transport")
    return spin - np.dot(spin * _METRIC, after) / denominator * (before + after)


@dataclass(frozen=True)
class LinearSpinFeedbackRecord:
    """Separate applied recoil ledger; none of these terms is charge Medina."""

    route: str
    applied: bool
    four_force_native: tuple[float, float, float, float]
    proper_step_ns: float
    work_native: float
    temporal_impulse_energy_native: float
    force_ratio: float

    def __post_init__(self) -> None:
        vector = np.asarray(self.four_force_native, dtype=float)
        if vector.shape != (4,) or not np.all(np.isfinite(vector)):
            raise ValueError("feedback force must be a finite four-vector")
        if not self.route or not isinstance(self.applied, bool):
            raise ValueError("feedback route and boolean applied flag required")
        if (
            not all(
                np.isfinite(v)
                for v in (
                    self.proper_step_ns,
                    self.work_native,
                    self.temporal_impulse_energy_native,
                    self.force_ratio,
                )
            )
            or self.proper_step_ns <= 0
            or self.force_ratio < 0
        ):
            raise ValueError("invalid feedback record scalars")
        object.__setattr__(self, "four_force_native", tuple(float(v) for v in vector))
        if not self.applied and (
            np.any(vector) or self.work_native or self.temporal_impulse_energy_native
        ):
            raise ValueError("unapplied feedback must have zero force and work")

    def to_checkpoint_payload(self) -> dict[str, Any]:
        return dict(
            route=self.route,
            applied=self.applied,
            four_force_native=self.four_force_native,
            proper_step_ns=self.proper_step_ns,
            work_native=self.work_native,
            temporal_impulse_energy_native=self.temporal_impulse_energy_native,
            force_ratio=self.force_ratio,
        )

    @classmethod
    def from_checkpoint_payload(
        cls, payload: Mapping[str, Any]
    ) -> LinearSpinFeedbackRecord:
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("feedback checkpoint keys differ")
        return cls(**payload)


def apply_linear_spin_impulse(
    *,
    result: Mapping[str, Any],
    start: Mapping[str, Any],
    four_force_native: VectorLike,
    proper_step_ns: float,
    route: str,
    force_ratio: float,
) -> dict[str, Any]:
    """Add one first-order magnetic recoil after the existing Medina update.

    Preserve charge force memory and charge radiation bookkeeping. Recompose
    kinetic energy from momentum and keep a separate stable work calculation.
    The time and position correction follows the same endpoint-average drift
    used by exact second-order pair stepping. Endpoint potentials are still
    finalized by the maintained pair finalizer after this function returns.
    """
    h = float(proper_step_ns)
    force = np.asarray(four_force_native, dtype=float)
    if (
        not np.isfinite(h)
        or h <= 0
        or force.shape != (4,)
        or not np.all(np.isfinite(force))
    ):
        raise ValueError("positive step and finite four-force required")
    if len(result["x"]) != 1 or len(start["x"]) != 1:
        raise ValueError(
            "experimental spin recoil currently requires one particle per role"
        )
    out = copy.deepcopy(dict(result))
    if not np.any(force):
        out["_linear_spin_feedback_record"] = LinearSpinFeedbackRecord(
            route, False, (0.0,) * 4, h, 0.0, 0.0, force_ratio
        )
        return out
    mass = float(result.get("m_species", result["m"])[0])
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError("positive species mass required")
    beta_before = np.array([result[f"b{a}"][0] for a in "xyz"])
    gamma_before = float(result["gamma"][0])
    p = mass * C_MMNS * gamma_before * beta_before
    impulse = h * force[1:]
    new_p = p + impulse
    old_pt = float(np.hypot(mass * C_MMNS, np.linalg.norm(p)))
    new_pt = float(np.hypot(mass * C_MMNS, np.linalg.norm(new_p)))
    work = C_MMNS * float(np.dot(impulse, 2 * p + impulse)) / (new_pt + old_pt)
    new_gamma = new_pt / (mass * C_MMNS)
    new_beta = new_p / new_pt
    if np.dot(new_beta, new_beta) >= 1 or not np.all(np.isfinite(new_beta)):
        raise ValueError("invalid velocity after spin recoil")
    spin = np.array([result[f"spin_{a}"][0] for a in "xyz"])
    before = np.r_[gamma_before * C_MMNS, p / mass]
    after = np.r_[new_gamma * C_MMNS, new_p / mass]
    moved_spin = transport_spin_between_velocities(
        boost_rest_polarization(spin, beta_before), before, after
    )
    rest_spin = rest_polarization_from_four_vector(moved_spin, new_beta)
    # Work is computed stably even when new_gamma-gamma_before rounds to zero.
    delta_gamma = work / (mass * C_MMNS**2)
    out["t"][0] += 0.5 * h * delta_gamma
    out["Pt"][0] += new_pt - old_pt
    out["gamma"][0] = new_gamma
    time_factor = C_MMNS * float(out["t"][0] - start["t"][0])
    if time_factor <= 0:
        raise ValueError("nonpositive elapsed time after spin recoil")
    for i, a in enumerate("xyz"):
        out[f"P{a}"][0] += impulse[i]
        out[a][0] += 0.5 * h * impulse[i] / mass
        out[f"b{a}"][0] = new_beta[i]
        out[f"bdot{a}"][0] = (new_beta[i] - start[f"b{a}"][0]) / time_factor
        out[f"spin_{a}"][0] = rest_spin[i]
        # Replace this step's contribution to the running average, not its count.
        key = f"beta_avg_{a}"
        if key in out and "beta_samples" in out and out["beta_samples"][0] > 0:
            out[key][0] += (new_beta[i] - beta_before[i]) / out["beta_samples"][0]
    out["_source_start_acceleration_complete"] = np.array([False])
    if "source_start_beta_prime_ready" in out:
        out["source_start_beta_prime_ready"] = np.array([False])
    out["_linear_spin_feedback_record"] = LinearSpinFeedbackRecord(
        route, True, tuple(force), h, work, h * C_MMNS * float(force[0]), force_ratio
    )
    return out


def apply_experimental_linear_spin_feedback(
    *,
    result: Mapping[str, Any],
    start: Mapping[str, Any],
    accepted_history: AcceptedIntrinsicSpinReductionHistory,
    proper_time_ns: float,
    proper_step_ns: float,
) -> dict[str, Any]:
    """Select a current-event reduction and apply only its magnetic/spin force."""
    from .spin_self_force_reduction_history import (
        _private_start_sample,
        _private_route_inputs,
        select_intrinsic_spin_reduction_route_native,
    )

    sample = _private_start_sample(result)
    analytical, reason, charge, mass, g_factor = _private_route_inputs(result)
    if analytical is not None:
        leading = analytical.leading_dynamics.four_acceleration
        actual = np.asarray(sample["non_self_four_acceleration_mm_ns2"])
        scale = max(
            np.linalg.norm(leading), np.linalg.norm(actual), np.finfo(float).tiny
        )
        if np.linalg.norm(leading - actual) > 1e-8 * scale:
            raise ValueError(
                "analytical spin reduction differs from the applied non-self dynamics"
            )
    current = accepted_history.append_accepted(proper_time_ns=proper_time_ns, **sample)
    if charge == 0 or not np.any(sample["physical_spin_four_native"]):
        return apply_linear_spin_impulse(
            result=result,
            start=start,
            four_force_native=np.zeros(4),
            proper_step_ns=proper_step_ns,
            route="zero_charge_or_spin",
            force_ratio=0.0,
        )
    if g_factor == 0:
        raise ValueError("charged zero-g spin is outside this experimental reduction")
    selected = select_intrinsic_spin_reduction_route_native(
        analytical_reduction=analytical,
        analytical_unavailable_reason=reason,
        accepted_history=current,
        charge_native=charge,
        mass_amu=mass,
        g_factor=g_factor,
    )
    reduction = selected.analytical_reduction or selected.causal_reduction
    if reduction is None:
        if selected.route == "unavailable_insufficient_accepted_history":
            return apply_linear_spin_impulse(
                result=result,
                start=start,
                four_force_native=np.zeros(4),
                proper_step_ns=proper_step_ns,
                route="warmup_no_force",
                force_ratio=0.0,
            )
        raise ValueError(f"experimental spin recoil unavailable: {selected.route}")
    force = reduction.radiation_balance.self_force.linear_spin_self_force_native
    velocity = np.asarray(sample["four_velocity_mm_ns"])
    scale = max(np.linalg.norm(force) * np.linalg.norm(velocity), np.finfo(float).tiny)
    if abs(np.dot(force * _METRIC, velocity)) > 1e-10 * scale:
        raise ValueError("self-force violates velocity orthogonality")

    def rest_norm(vector: np.ndarray, u: np.ndarray) -> float:
        return float(np.linalg.norm(vector[1:] - vector[0] * u[1:] / (u[0] + C_MMNS)))

    reference = max(
        rest_norm(mass * a, u)
        for a, u in zip(
            current.non_self_four_acceleration_mm_ns2, current.four_velocity_mm_ns
        )
    )
    strength = rest_norm(force, velocity)
    ratio = strength / max(reference, np.finfo(float).tiny)
    if ratio > 0.1:
        raise ValueError(
            "experimental recoil exceeds 10 percent of recent non-self force"
        )
    return apply_linear_spin_impulse(
        result=result,
        start=start,
        four_force_native=force,
        proper_step_ns=proper_step_ns,
        route=selected.route,
        force_ratio=ratio,
    )
