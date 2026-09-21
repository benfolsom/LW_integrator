"""Auditable first-pass diagnostics for a physical two-particle flyby.

This module deliberately separates three quantities which are easy to blur:

* mechanical four-momentum, either read from stored ``gamma``/``beta`` or
  reconstructed as ``p = P - q A / c`` when the *ordinary* charge-plus-dipole
  Maxwell potential is supplied;
* a relativistic, instantaneous-Coulomb osculating energy used only to label
  an inbound/outbound two-body state; and
* Medina's per-step charge-radiation bookkeeping.

The osculating energy is not a conserved Hamiltonian for a retarded-field
calculation.  It excludes velocity-dependent near-field energy, intrinsic
dipole interaction energy, and radiation already in flight.  It is useful for
the narrower question "did an initially unbound flyby cross the same radius
outbound with negative energy in the instantaneous Coulomb reference model?"
It must not be reported as a particle-plus-field energy balance.

The helpers require one physical particle on each side.  Mean-field
macroparticle source/observer weights do not define a reciprocal two-body
potential and are rejected instead of being assigned an ambiguous energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np

from .constants import C_MMNS
from .external_fields import ELEMENTARY_CHARGE_COULOMB
from .magnetic_dipole import NATIVE_ENERGY_UNIT_J

_ENERGY_MODEL = (
    "relativistic invariant-COM kinetic energy plus instantaneous Gaussian "
    "Coulomb q1*q2/r; diagnostic only, not conserved for retarded fields"
)


@dataclass(frozen=True)
class ParticleCaptureTrace:
    """One physical particle's public trajectory and radiation diagnostics."""

    time_ns: np.ndarray
    position_mm: np.ndarray
    canonical_four_momentum_native: np.ndarray
    gamma: np.ndarray
    beta: np.ndarray
    mass_amu: float
    observer_charge_native: float
    source_charge_native: float
    macro_population: float
    radiation_reaction_work_native: np.ndarray
    far_radiated_energy_native: np.ndarray
    medina_cross_field_energy_native: np.ndarray
    medina_cross_field_energy_change_native: np.ndarray
    medina_force_derivative_ready: np.ndarray
    medina_impulse_capped: np.ndarray
    medina_external_force_sample_time_ns: np.ndarray
    mass_shell_projection_energy_native: np.ndarray
    dead: np.ndarray
    ordinary_four_potential_native: np.ndarray | None = None


@dataclass(frozen=True)
class CanonicalMomentumAudit:
    """Agreement of ``P-qA/c`` with stored mechanical kinematics."""

    checked: bool
    max_absolute_residual_native: float
    max_relative_residual: float
    max_relative_mass_shell_residual: float


@dataclass(frozen=True)
class MedinaCaptureAudit:
    """Signed, additive Medina diagnostics for one particle history."""

    force_sample_count: int
    derivative_ready_count: int
    unexpected_unready_count: int
    impulse_cap_count: int
    negative_far_energy_count: int
    signed_reaction_work_native: float
    far_radiated_energy_native: float
    cross_field_energy_change_native: float
    cross_field_endpoint_change_native: float
    balance_residual_native: float


@dataclass(frozen=True)
class MassShellProjectionAudit:
    """Signed energy inserted or removed by explicit on-shell projection."""

    nonzero_step_count: int
    signed_energy_native: float
    sum_absolute_energy_native: float
    max_absolute_energy_native: float


@dataclass(frozen=True)
class TwoBodyOsculatingSeries:
    """Lab-synchronized two-body kinematics and reference energy series."""

    time_ns: np.ndarray
    separation_vector_mm: np.ndarray
    separation_mm: np.ndarray
    radial_velocity_mm_per_ns: np.ndarray
    first_mechanical_four_momentum_native: np.ndarray
    second_mechanical_four_momentum_native: np.ndarray
    invariant_com_kinetic_energy_native: np.ndarray
    instantaneous_coulomb_energy_native: np.ndarray
    osculating_energy_native: np.ndarray


@dataclass(frozen=True)
class FirstPassCaptureAnalysis:
    """Result of a same-radius inbound/outbound first-pass classification."""

    energy_model: str
    series: TwoBodyOsculatingSeries
    first_canonical_audit: CanonicalMomentumAudit
    second_canonical_audit: CanonicalMomentumAudit
    first_medina_audit: MedinaCaptureAudit
    second_medina_audit: MedinaCaptureAudit
    first_mass_shell_projection_audit: MassShellProjectionAudit
    second_mass_shell_projection_audit: MassShellProjectionAudit
    initial_separation_mm: float
    periapsis_time_ns: float
    periapsis_separation_mm: float
    final_separation_mm: float
    initial_osculating_energy_native: float
    outbound_reference_time_ns: float
    outbound_reference_energy_native: float
    final_osculating_energy_native: float
    complete_same_radius_pass: bool
    diagnostics_valid: bool
    captured: bool
    invalid_reasons: tuple[str, ...]

    @property
    def initial_osculating_energy_ev(self) -> float:
        return native_energy_to_ev(self.initial_osculating_energy_native)

    @property
    def outbound_reference_energy_ev(self) -> float:
        return native_energy_to_ev(self.outbound_reference_energy_native)

    @property
    def total_far_radiated_energy_native(self) -> float:
        return float(
            self.first_medina_audit.far_radiated_energy_native
            + self.second_medina_audit.far_radiated_energy_native
        )

    @property
    def total_signed_reaction_work_native(self) -> float:
        return float(
            self.first_medina_audit.signed_reaction_work_native
            + self.second_medina_audit.signed_reaction_work_native
        )

    @property
    def total_signed_mass_shell_projection_energy_native(self) -> float:
        return float(
            self.first_mass_shell_projection_audit.signed_energy_native
            + self.second_mass_shell_projection_audit.signed_energy_native
        )

    @property
    def max_absolute_mass_shell_projection_energy_native(self) -> float:
        return max(
            self.first_mass_shell_projection_audit.max_absolute_energy_native,
            self.second_mass_shell_projection_audit.max_absolute_energy_native,
        )

    @property
    def total_absolute_mass_shell_projection_energy_native(self) -> float:
        return float(
            self.first_mass_shell_projection_audit.sum_absolute_energy_native
            + self.second_mass_shell_projection_audit.sum_absolute_energy_native
        )


def native_energy_to_ev(value_native: float) -> float:
    """Convert ``amu mm^2/ns^2`` to electronvolts."""

    return float(value_native) * NATIVE_ENERGY_UNIT_J / ELEMENTARY_CHARGE_COULOMB


def particle_capture_trace_from_soa(
    trajectory: Any,
    *,
    particle_index: int = 0,
    ordinary_four_potential_native: (
        Sequence[Sequence[float]] | np.ndarray | None
    ) = None,
) -> ParticleCaptureTrace:
    """Extract one particle from a :class:`~core.types.TrajectoryArrays` view.

    ``ordinary_four_potential_native`` must be the complete non-self ordinary
    Maxwell potential at each stored event: retarded charge potential plus
    retarded intrinsic-dipole potential.  The RFS response quantity
    ``B_mu = F*_(mu nu) a^nu`` is not a vector potential and must not be added.
    The integrator does not currently persist this series, so omission simply
    leaves the canonical consistency audit unchecked.
    """

    index = int(particle_index)
    n_particles = int(trajectory.x.shape[1])
    if index < 0:
        index += n_particles
    if index < 0 or index >= n_particles:
        raise IndexError("particle_index is outside the trajectory particle axis")

    potential = None
    if ordinary_four_potential_native is not None:
        potential = np.asarray(ordinary_four_potential_native, dtype=float)

    return ParticleCaptureTrace(
        time_ns=np.asarray(trajectory.t[:, index], dtype=float),
        position_mm=np.stack(
            [trajectory.x[:, index], trajectory.y[:, index], trajectory.z[:, index]],
            axis=-1,
        ).astype(float, copy=False),
        canonical_four_momentum_native=np.stack(
            [
                trajectory.Pt[:, index],
                trajectory.Px[:, index],
                trajectory.Py[:, index],
                trajectory.Pz[:, index],
            ],
            axis=-1,
        ).astype(float, copy=False),
        gamma=np.asarray(trajectory.gamma[:, index], dtype=float),
        beta=np.stack(
            [trajectory.bx[:, index], trajectory.by[:, index], trajectory.bz[:, index]],
            axis=-1,
        ).astype(float, copy=False),
        mass_amu=float(trajectory.m[index]),
        observer_charge_native=float(trajectory.q_observer[index]),
        source_charge_native=float(trajectory.q_source[index]),
        macro_population=float(trajectory.macro_population[index]),
        radiation_reaction_work_native=np.asarray(
            trajectory.radiation_reaction_work[:, index], dtype=float
        ),
        far_radiated_energy_native=np.asarray(
            trajectory.radiation_energy[:, index], dtype=float
        ),
        medina_cross_field_energy_native=np.asarray(
            trajectory.medina_cross_field_energy[:, index], dtype=float
        ),
        medina_cross_field_energy_change_native=np.asarray(
            trajectory.medina_cross_field_energy_change[:, index], dtype=float
        ),
        medina_force_derivative_ready=np.asarray(
            trajectory.medina_force_derivative_ready[:, index], dtype=bool
        ),
        medina_impulse_capped=np.asarray(
            trajectory.medina_impulse_capped[:, index], dtype=bool
        ),
        medina_external_force_sample_time_ns=np.asarray(
            trajectory.medina_external_force_sample_time[:, index], dtype=float
        ),
        mass_shell_projection_energy_native=np.asarray(
            trajectory.mass_shell_projection_energy[:, index], dtype=float
        ),
        dead=np.asarray(trajectory.dead[:, index], dtype=bool),
        ordinary_four_potential_native=potential,
    )


def reconstruct_mechanical_four_momentum_series_native(
    canonical_four_momentum_native: Sequence[Sequence[float]] | np.ndarray,
    ordinary_four_potential_native: Sequence[Sequence[float]] | np.ndarray,
    *,
    observer_charge_native: float,
) -> np.ndarray:
    """Return ``p^mu=P^mu-q A^mu/c`` for a series of observer events."""

    canonical = np.asarray(canonical_four_momentum_native, dtype=float)
    potential = np.asarray(ordinary_four_potential_native, dtype=float)
    if canonical.ndim != 2 or canonical.shape[1] != 4:
        raise ValueError("canonical_four_momentum_native must have shape [steps, 4]")
    if potential.shape != canonical.shape:
        raise ValueError(
            "ordinary_four_potential_native must match the canonical [steps, 4] shape"
        )
    if not np.all(np.isfinite(canonical)) or not np.all(np.isfinite(potential)):
        raise ValueError("canonical momentum and ordinary potential must be finite")
    charge = float(observer_charge_native)
    if not np.isfinite(charge):
        raise ValueError("observer_charge_native must be finite")
    return cast(np.ndarray, canonical - charge * potential / C_MMNS)


def stored_mechanical_four_momentum_series_native(
    gamma: Sequence[float] | np.ndarray,
    beta: Sequence[Sequence[float]] | np.ndarray,
    *,
    mass_amu: float,
) -> np.ndarray:
    """Build on-shell ``p^mu=(gamma*m*c, gamma*m*c*beta)``."""

    gamma_array = np.asarray(gamma, dtype=float)
    beta_array = np.asarray(beta, dtype=float)
    if gamma_array.ndim != 1 or beta_array.shape != (gamma_array.size, 3):
        raise ValueError("gamma and beta must have shapes [steps] and [steps, 3]")
    mass = float(mass_amu)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_amu must be finite and positive")
    if not np.all(np.isfinite(gamma_array)) or not np.all(np.isfinite(beta_array)):
        raise ValueError("gamma and beta must be finite")
    if np.any(gamma_array < 1.0):
        raise ValueError("gamma must be at least one")
    p0 = gamma_array * mass * C_MMNS
    return np.concatenate((p0[:, np.newaxis], p0[:, np.newaxis] * beta_array), axis=1)


def audit_canonical_mechanical_momentum(
    trace: ParticleCaptureTrace,
) -> CanonicalMomentumAudit:
    """Compare optional exact-potential reconstruction to stored kinematics."""

    if trace.ordinary_four_potential_native is None:
        return CanonicalMomentumAudit(False, np.nan, np.nan, np.nan)
    stored = stored_mechanical_four_momentum_series_native(
        trace.gamma,
        trace.beta,
        mass_amu=trace.mass_amu,
    )
    reconstructed = reconstruct_mechanical_four_momentum_series_native(
        trace.canonical_four_momentum_native,
        trace.ordinary_four_potential_native,
        observer_charge_native=trace.observer_charge_native,
    )
    residual = reconstructed - stored
    row_residual = np.linalg.norm(residual, axis=1)
    row_scale = np.maximum(np.linalg.norm(stored, axis=1), trace.mass_amu * C_MMNS)
    shell = (
        reconstructed[:, 0] ** 2
        - np.sum(reconstructed[:, 1:] ** 2, axis=1)
        - (trace.mass_amu * C_MMNS) ** 2
    )
    shell_scale = (trace.mass_amu * C_MMNS) ** 2
    return CanonicalMomentumAudit(
        True,
        float(np.max(row_residual, initial=0.0)),
        float(np.max(row_residual / row_scale, initial=0.0)),
        float(np.max(np.abs(shell) / shell_scale, initial=0.0)),
    )


def audit_medina_capture_trace(
    trace: ParticleCaptureTrace,
    *,
    negative_energy_tolerance_native: float = 0.0,
) -> MedinaCaptureAudit:
    """Sum Medina terms without changing their signs or reinterpreting them."""

    tolerance = float(negative_energy_tolerance_native)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("negative_energy_tolerance_native must be finite and >= 0")
    sample_indices = np.flatnonzero(
        np.isfinite(trace.medina_external_force_sample_time_ns)
    )
    ready = np.asarray(trace.medina_force_derivative_ready, dtype=bool)
    unexpected_unready = (
        int(np.count_nonzero(~ready[sample_indices[1:]]))
        if sample_indices.size > 1
        else 0
    )
    cross = np.asarray(trace.medina_cross_field_energy_native, dtype=float)
    if sample_indices.size:
        endpoint_change = float(cross[sample_indices[-1]] - cross[sample_indices[0]])
    else:
        endpoint_change = 0.0
    reaction_steps = np.asarray(trace.radiation_reaction_work_native, dtype=float)
    far_steps = np.asarray(trace.far_radiated_energy_native, dtype=float)
    cross_steps = np.asarray(trace.medina_cross_field_energy_change_native, dtype=float)
    reaction_work = float(np.sum(reaction_steps, dtype=float))
    far_energy = float(np.sum(far_steps, dtype=float))
    cross_change = float(np.sum(cross_steps, dtype=float))
    # The first physical force sample intentionally diagnoses far radiation
    # but applies no incomplete Medina derivative or reaction impulse.  Test
    # the signed balance only where that derivative was actually ready while
    # retaining the priming interval in the separately reported far total.
    balance_mask = np.zeros_like(ready, dtype=bool)
    balance_mask[sample_indices] = ready[sample_indices]
    balance_residual = float(
        np.sum(
            reaction_steps[balance_mask]
            + far_steps[balance_mask]
            + cross_steps[balance_mask],
            dtype=float,
        )
    )
    return MedinaCaptureAudit(
        force_sample_count=int(sample_indices.size),
        derivative_ready_count=int(np.count_nonzero(ready[sample_indices])),
        unexpected_unready_count=unexpected_unready,
        impulse_cap_count=int(np.count_nonzero(trace.medina_impulse_capped)),
        negative_far_energy_count=int(
            np.count_nonzero(
                np.asarray(trace.far_radiated_energy_native, dtype=float) < -tolerance
            )
        ),
        signed_reaction_work_native=reaction_work,
        far_radiated_energy_native=far_energy,
        cross_field_energy_change_native=cross_change,
        cross_field_endpoint_change_native=endpoint_change,
        balance_residual_native=balance_residual,
    )


def audit_mass_shell_projection(
    trace: ParticleCaptureTrace,
) -> MassShellProjectionAudit:
    """Summarize the explicit pre-RR mass-shell energy correction by step."""

    projection = np.asarray(
        trace.mass_shell_projection_energy_native,
        dtype=float,
    )
    if projection.ndim != 1 or not np.all(np.isfinite(projection)):
        raise ValueError(
            "mass_shell_projection_energy_native must be one-dimensional and finite"
        )
    return MassShellProjectionAudit(
        nonzero_step_count=int(np.count_nonzero(projection)),
        signed_energy_native=float(np.sum(projection, dtype=float)),
        sum_absolute_energy_native=float(np.sum(np.abs(projection), dtype=float)),
        max_absolute_energy_native=float(np.max(np.abs(projection), initial=0.0)),
    )


def relativistic_invariant_com_kinetic_energy_native(
    first_four_momentum_native: Sequence[Sequence[float]] | np.ndarray,
    second_four_momentum_native: Sequence[Sequence[float]] | np.ndarray,
    *,
    first_mass_amu: float,
    second_mass_amu: float,
) -> np.ndarray:
    """Return stable total kinetic energy in the instantaneous mechanical COM.

    The input order is ``(p0, px, py, pz)`` with ``p0=E/c``.  Each spatial
    momentum is Lorentz-boosted into the center-of-momentum frame.  Kinetic
    energy is then evaluated as ``p^2 c / (sqrt((mc)^2+p^2)+mc)``, avoiding a
    subtraction of electronvolt-scale motion from gigaelectronvolt rest energy.
    """

    first = np.asarray(first_four_momentum_native, dtype=float)
    second = np.asarray(second_four_momentum_native, dtype=float)
    if first.ndim != 2 or first.shape[1] != 4 or second.shape != first.shape:
        raise ValueError("both four-momentum arrays must have shape [samples, 4]")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("four-momentum arrays must be finite")
    first_mass = float(first_mass_amu)
    second_mass = float(second_mass_amu)
    if (
        not np.isfinite(first_mass)
        or not np.isfinite(second_mass)
        or first_mass <= 0.0
        or second_mass <= 0.0
    ):
        raise ValueError("particle masses must be finite and positive")

    total_p0 = first[:, 0] + second[:, 0]
    total_spatial = first[:, 1:] + second[:, 1:]
    if np.any(total_p0 <= 0.0):
        raise ValueError("total mechanical energy must be positive")
    com_beta = total_spatial / total_p0[:, np.newaxis]
    com_beta_squared = np.sum(com_beta * com_beta, axis=1)
    if np.any(com_beta_squared >= 1.0):
        raise ValueError("total mechanical four-momentum must be timelike")
    com_gamma = 1.0 / np.sqrt(1.0 - com_beta_squared)
    stable_boost_coefficient = com_gamma**2 / (com_gamma + 1.0)

    def boosted_spatial(momentum: np.ndarray) -> np.ndarray:
        beta_dot_p = np.sum(com_beta * momentum[:, 1:], axis=1)
        coefficient = stable_boost_coefficient * beta_dot_p - com_gamma * momentum[:, 0]
        return cast(
            np.ndarray,
            momentum[:, 1:] + coefficient[:, np.newaxis] * com_beta,
        )

    first_com = boosted_spatial(first)
    second_com = boosted_spatial(second)

    def kinetic(spatial: np.ndarray, mass: float) -> np.ndarray:
        momentum_squared = np.sum(spatial * spatial, axis=1)
        rest_momentum = mass * C_MMNS
        return cast(
            np.ndarray,
            C_MMNS
            * momentum_squared
            / (np.sqrt(rest_momentum**2 + momentum_squared) + rest_momentum),
        )

    return cast(
        np.ndarray,
        kinetic(first_com, first_mass) + kinetic(second_com, second_mass),
    )


def _validate_trace(trace: ParticleCaptureTrace, *, name: str) -> None:
    count = int(np.asarray(trace.time_ns).size)
    if count < 2:
        raise ValueError(f"{name} trace must contain at least two samples")
    expected_vector = (count, 3)
    expected_four = (count, 4)
    if np.asarray(trace.position_mm).shape != expected_vector:
        raise ValueError(f"{name} position_mm must have shape [steps, 3]")
    if np.asarray(trace.beta).shape != expected_vector:
        raise ValueError(f"{name} beta must have shape [steps, 3]")
    if np.asarray(trace.canonical_four_momentum_native).shape != expected_four:
        raise ValueError(
            f"{name} canonical_four_momentum_native must have shape [steps, 4]"
        )
    if (
        trace.ordinary_four_potential_native is not None
        and np.asarray(trace.ordinary_four_potential_native).shape != expected_four
    ):
        raise ValueError(
            f"{name} ordinary_four_potential_native must have shape [steps, 4]"
        )
    one_dimensional = (
        "gamma",
        "radiation_reaction_work_native",
        "far_radiated_energy_native",
        "medina_cross_field_energy_native",
        "medina_cross_field_energy_change_native",
        "medina_force_derivative_ready",
        "medina_impulse_capped",
        "medina_external_force_sample_time_ns",
        "mass_shell_projection_energy_native",
        "dead",
    )
    for field_name in one_dimensional:
        if np.asarray(getattr(trace, field_name)).shape != (count,):
            raise ValueError(f"{name} {field_name} must have shape [steps]")
    times = np.asarray(trace.time_ns, dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{name} time_ns must be finite and strictly increasing")
    finite_fields = (
        "position_mm",
        "canonical_four_momentum_native",
        "gamma",
        "beta",
        "radiation_reaction_work_native",
        "far_radiated_energy_native",
        "medina_cross_field_energy_native",
        "medina_cross_field_energy_change_native",
        "mass_shell_projection_energy_native",
    )
    for field_name in finite_fields:
        if not np.all(np.isfinite(np.asarray(getattr(trace, field_name), dtype=float))):
            raise ValueError(f"{name} {field_name} must contain only finite values")
    sample_times = np.asarray(trace.medina_external_force_sample_time_ns, dtype=float)
    if np.any(np.isinf(sample_times)):
        raise ValueError(
            f"{name} medina_external_force_sample_time_ns may be finite or NaN, "
            "not infinite"
        )
    if not np.isfinite(trace.mass_amu) or trace.mass_amu <= 0.0:
        raise ValueError(f"{name} mass_amu must be finite and positive")
    if not np.isfinite(trace.observer_charge_native) or not np.isfinite(
        trace.source_charge_native
    ):
        raise ValueError(f"{name} source and observer charges must be finite")
    if not np.isfinite(trace.macro_population) or trace.macro_population != 1.0:
        raise ValueError(
            f"{name} must represent one physical particle (macro_population=1)"
        )
    beta = np.asarray(trace.beta, dtype=float)
    beta_squared = np.sum(beta * beta, axis=1)
    if np.any(beta_squared >= 1.0):
        raise ValueError(f"{name} beta must remain subluminal")
    expected_gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    if not np.allclose(
        np.asarray(trace.gamma, dtype=float),
        expected_gamma,
        rtol=1.0e-9,
        atol=1.0e-12,
    ):
        raise ValueError(f"{name} gamma and beta do not describe the same velocity")
    if trace.ordinary_four_potential_native is not None and not np.all(
        np.isfinite(np.asarray(trace.ordinary_four_potential_native, dtype=float))
    ):
        raise ValueError(
            f"{name} ordinary_four_potential_native must contain only finite values"
        )


def _interpolate_columns(
    sample_time: np.ndarray,
    values: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            np.interp(target_time, sample_time, values[:, column])
            for column in range(values.shape[1])
        ],
        axis=-1,
    )


def _synchronized_series(
    first: ParticleCaptureTrace,
    second: ParticleCaptureTrace,
) -> TwoBodyOsculatingSeries:
    first_time = np.asarray(first.time_ns, dtype=float)
    second_time = np.asarray(second.time_ns, dtype=float)
    overlap_start = max(float(first_time[0]), float(second_time[0]))
    overlap_stop = min(float(first_time[-1]), float(second_time[-1]))
    if overlap_stop <= overlap_start:
        raise ValueError("particle histories do not have a nonzero lab-time overlap")
    common_time = np.unique(
        np.concatenate(
            (
                first_time[
                    (first_time >= overlap_start) & (first_time <= overlap_stop)
                ],
                second_time[
                    (second_time >= overlap_start) & (second_time <= overlap_stop)
                ],
            )
        )
    )
    if common_time.size < 3:
        raise ValueError("lab-time overlap must contain at least three samples")

    first_position = _interpolate_columns(
        first_time, np.asarray(first.position_mm, dtype=float), common_time
    )
    second_position = _interpolate_columns(
        second_time, np.asarray(second.position_mm, dtype=float), common_time
    )
    first_stored = stored_mechanical_four_momentum_series_native(
        first.gamma, first.beta, mass_amu=first.mass_amu
    )
    second_stored = stored_mechanical_four_momentum_series_native(
        second.gamma, second.beta, mass_amu=second.mass_amu
    )
    first_spatial = _interpolate_columns(first_time, first_stored[:, 1:], common_time)
    second_spatial = _interpolate_columns(
        second_time, second_stored[:, 1:], common_time
    )

    def on_shell(spatial: np.ndarray, mass: float) -> np.ndarray:
        p0 = np.sqrt((mass * C_MMNS) ** 2 + np.sum(spatial * spatial, axis=1))
        return np.concatenate((p0[:, np.newaxis], spatial), axis=1)

    first_momentum = on_shell(first_spatial, first.mass_amu)
    second_momentum = on_shell(second_spatial, second.mass_amu)
    separation_vector = first_position - second_position
    separation = np.linalg.norm(separation_vector, axis=1)
    if np.any(separation <= 0.0) or not np.all(np.isfinite(separation)):
        raise ValueError("two-body separation must remain finite and positive")

    first_velocity = C_MMNS * first_momentum[:, 1:] / first_momentum[:, :1]
    second_velocity = C_MMNS * second_momentum[:, 1:] / second_momentum[:, :1]
    relative_velocity = first_velocity - second_velocity
    radial_velocity = np.sum(
        relative_velocity * separation_vector / separation[:, np.newaxis], axis=1
    )
    kinetic = relativistic_invariant_com_kinetic_energy_native(
        first_momentum,
        second_momentum,
        first_mass_amu=first.mass_amu,
        second_mass_amu=second.mass_amu,
    )
    coupling_first_observer = first.observer_charge_native * second.source_charge_native
    coupling_second_observer = (
        second.observer_charge_native * first.source_charge_native
    )
    coupling_scale = max(
        abs(coupling_first_observer), abs(coupling_second_observer), 1.0e-300
    )
    if (
        abs(coupling_first_observer - coupling_second_observer) / coupling_scale
        > 1.0e-12
    ):
        raise ValueError(
            "source/observer charge weights are not reciprocal; no unique physical "
            "two-body Coulomb potential exists"
        )
    coupling = 0.5 * (coupling_first_observer + coupling_second_observer)
    coulomb = coupling / separation
    return TwoBodyOsculatingSeries(
        time_ns=common_time,
        separation_vector_mm=separation_vector,
        separation_mm=separation,
        radial_velocity_mm_per_ns=radial_velocity,
        first_mechanical_four_momentum_native=first_momentum,
        second_mechanical_four_momentum_native=second_momentum,
        invariant_com_kinetic_energy_native=kinetic,
        instantaneous_coulomb_energy_native=coulomb,
        osculating_energy_native=kinetic + coulomb,
    )


def _periapsis_from_radial_crossing(
    series: TwoBodyOsculatingSeries,
    *,
    radial_velocity_tolerance_mm_per_ns: float,
) -> tuple[int, float, float] | None:
    radial = series.radial_velocity_mm_per_ns
    tolerance = float(radial_velocity_tolerance_mm_per_ns)
    candidates = np.flatnonzero((radial[:-1] < -tolerance) & (radial[1:] > tolerance))
    if not candidates.size:
        candidates = np.flatnonzero((radial[:-1] <= 0.0) & (radial[1:] >= 0.0))
    if not candidates.size:
        return None
    index = int(
        candidates[
            np.argmin(
                np.minimum(
                    series.separation_mm[candidates],
                    series.separation_mm[candidates + 1],
                )
            )
        ]
    )
    denominator = radial[index + 1] - radial[index]
    fraction = 0.5 if denominator == 0.0 else -radial[index] / denominator
    fraction = float(np.clip(fraction, 0.0, 1.0))
    time = float(
        series.time_ns[index]
        + fraction * (series.time_ns[index + 1] - series.time_ns[index])
    )
    separation = float(
        series.separation_mm[index]
        + fraction * (series.separation_mm[index + 1] - series.separation_mm[index])
    )
    return index, time, separation


def _outbound_same_radius_reference(
    series: TwoBodyOsculatingSeries,
    *,
    after_index: int,
    initial_separation_mm: float,
    radial_velocity_tolerance_mm_per_ns: float,
) -> tuple[float, float] | None:
    radius = series.separation_mm
    radial = series.radial_velocity_mm_per_ns
    for index in range(max(0, after_index), radius.size - 1):
        if (
            radius[index] <= initial_separation_mm
            and radius[index + 1] >= initial_separation_mm
            and radial[index + 1] > radial_velocity_tolerance_mm_per_ns
        ):
            denominator = radius[index + 1] - radius[index]
            fraction = (
                1.0
                if denominator == 0.0
                else (initial_separation_mm - radius[index]) / denominator
            )
            fraction = float(np.clip(fraction, 0.0, 1.0))
            time = float(
                series.time_ns[index]
                + fraction * (series.time_ns[index + 1] - series.time_ns[index])
            )
            energy = float(
                series.osculating_energy_native[index]
                + fraction
                * (
                    series.osculating_energy_native[index + 1]
                    - series.osculating_energy_native[index]
                )
            )
            return time, energy
    return None


def analyze_first_pass_capture(
    first: ParticleCaptureTrace,
    second: ParticleCaptureTrace,
    *,
    capture_energy_tolerance_native: float = 0.0,
    radial_velocity_tolerance_mm_per_ns: float = 0.0,
    medina_balance_relative_tolerance: float = 1.0e-5,
    medina_balance_absolute_tolerance_native: float = 0.0,
    canonical_relative_tolerance: float = 1.0e-8,
    canonical_mass_shell_tolerance: float = 1.0e-8,
) -> FirstPassCaptureAnalysis:
    """Classify one complete same-radius electron--proton-style first pass.

    A positive-to-negative energy transition is evaluated at the outbound
    crossing of the *initial synchronized separation*, not at an arbitrary
    final timestep.  Capture is accepted only when the encounter brackets a
    periapsis, the same outbound radius is reached, Medina has no cap/readiness
    or balance failure, no particle is dead, and any supplied exact-potential
    canonical audit passes.

    This is a deterministic classification inside the named osculating model.
    It is not a proof of a stable orbit and is not an energy-conservation test.
    """

    _validate_trace(first, name="first")
    _validate_trace(second, name="second")
    energy_tolerance = float(capture_energy_tolerance_native)
    radial_tolerance = float(radial_velocity_tolerance_mm_per_ns)
    balance_rtol = float(medina_balance_relative_tolerance)
    balance_atol = float(medina_balance_absolute_tolerance_native)
    if not np.isfinite(energy_tolerance) or energy_tolerance < 0.0:
        raise ValueError("capture_energy_tolerance_native must be finite and >= 0")
    if not np.isfinite(radial_tolerance) or radial_tolerance < 0.0:
        raise ValueError("radial_velocity_tolerance_mm_per_ns must be finite and >= 0")
    if not np.isfinite(balance_rtol) or balance_rtol < 0.0:
        raise ValueError("medina_balance_relative_tolerance must be finite and >= 0")
    if not np.isfinite(balance_atol) or balance_atol < 0.0:
        raise ValueError(
            "medina_balance_absolute_tolerance_native must be finite and >= 0"
        )

    series = _synchronized_series(first, second)
    first_canonical = audit_canonical_mechanical_momentum(first)
    second_canonical = audit_canonical_mechanical_momentum(second)
    first_medina = audit_medina_capture_trace(
        first, negative_energy_tolerance_native=balance_atol
    )
    second_medina = audit_medina_capture_trace(
        second, negative_energy_tolerance_native=balance_atol
    )
    first_projection = audit_mass_shell_projection(first)
    second_projection = audit_mass_shell_projection(second)
    invalid: list[str] = []

    initial_energy = float(series.osculating_energy_native[0])
    if initial_energy < -energy_tolerance:
        invalid.append("initial osculating state is already bound")
    if series.radial_velocity_mm_per_ns[0] >= -radial_tolerance:
        invalid.append("initial synchronized state is not inbound")
    periapsis = _periapsis_from_radial_crossing(
        series, radial_velocity_tolerance_mm_per_ns=radial_tolerance
    )
    initial_separation = float(series.separation_mm[0])
    if periapsis is None:
        periapsis_index = 0
        periapsis_time = np.nan
        periapsis_separation = np.nan
        invalid.append("trajectory does not bracket an inbound-to-outbound periapsis")
        outbound = None
    else:
        periapsis_index, periapsis_time, periapsis_separation = periapsis
        outbound = _outbound_same_radius_reference(
            series,
            after_index=periapsis_index,
            initial_separation_mm=initial_separation,
            radial_velocity_tolerance_mm_per_ns=radial_tolerance,
        )
    if outbound is None:
        outbound_time = np.nan
        outbound_energy = np.nan
        invalid.append("trajectory does not reach the initial separation outbound")
    else:
        outbound_time, outbound_energy = outbound

    for name, trace, medina, canonical in (
        ("first", first, first_medina, first_canonical),
        ("second", second, second_medina, second_canonical),
    ):
        if np.any(trace.dead):
            invalid.append(f"{name} particle is marked dead")
        if medina.impulse_cap_count:
            invalid.append(f"{name} Medina impulse was capped")
        if medina.unexpected_unready_count:
            invalid.append(f"{name} Medina derivative has an unexpected unready gap")
        if medina.negative_far_energy_count:
            invalid.append(f"{name} Medina far-radiated energy is negative")
        balance_scale = max(
            abs(medina.signed_reaction_work_native),
            abs(medina.far_radiated_energy_native),
            abs(medina.cross_field_energy_change_native),
        )
        if (
            abs(medina.balance_residual_native)
            > balance_atol + balance_rtol * balance_scale
        ):
            invalid.append(f"{name} Medina signed energy balance does not close")
        if canonical.checked and (
            canonical.max_relative_residual > canonical_relative_tolerance
            or canonical.max_relative_mass_shell_residual
            > canonical_mass_shell_tolerance
        ):
            invalid.append(f"{name} canonical P-qA/c audit failed")

    complete = periapsis is not None and outbound is not None
    diagnostics_valid = not invalid
    captured = bool(
        complete
        and diagnostics_valid
        and initial_energy >= -energy_tolerance
        and outbound_energy < -energy_tolerance
    )
    return FirstPassCaptureAnalysis(
        energy_model=_ENERGY_MODEL,
        series=series,
        first_canonical_audit=first_canonical,
        second_canonical_audit=second_canonical,
        first_medina_audit=first_medina,
        second_medina_audit=second_medina,
        first_mass_shell_projection_audit=first_projection,
        second_mass_shell_projection_audit=second_projection,
        initial_separation_mm=initial_separation,
        periapsis_time_ns=float(periapsis_time),
        periapsis_separation_mm=float(periapsis_separation),
        final_separation_mm=float(series.separation_mm[-1]),
        initial_osculating_energy_native=initial_energy,
        outbound_reference_time_ns=float(outbound_time),
        outbound_reference_energy_native=float(outbound_energy),
        final_osculating_energy_native=float(series.osculating_energy_native[-1]),
        complete_same_radius_pass=complete,
        diagnostics_valid=diagnostics_valid,
        captured=captured,
        invalid_reasons=tuple(invalid),
    )


__all__ = [
    "CanonicalMomentumAudit",
    "FirstPassCaptureAnalysis",
    "MassShellProjectionAudit",
    "MedinaCaptureAudit",
    "ParticleCaptureTrace",
    "TwoBodyOsculatingSeries",
    "analyze_first_pass_capture",
    "audit_canonical_mechanical_momentum",
    "audit_medina_capture_trace",
    "audit_mass_shell_projection",
    "native_energy_to_ev",
    "particle_capture_trace_from_soa",
    "reconstruct_mechanical_four_momentum_series_native",
    "relativistic_invariant_com_kinetic_energy_native",
    "stored_mechanical_four_momentum_series_native",
]
