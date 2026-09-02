"""Retarded equations of motion for the Liénard–Wiechert solver.

The implementation preserves the validated reference behavior so historical
regression data remains applicable. The heavy lifting
is performed inside :func:`retarded_equations_of_motion`, which calculates the
covariant updates for momentum, position, and acceleration for each particle.

Physical Foundation
-------------------

The integrator evolves particles in coordinate time with step h = Δt, updating
conjugate momentum from retarded electromagnetic forces, then deriving positions
and velocities.

Conjugate vs. Kinetic Momentum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The conjugate (canonical) momentum includes electromagnetic potentials::

    P^μ = γ·m·V^μ + (e/c)·A^μ

For spatial components: P_i = γ·m·v_i + (e/c)·A_i

The kinetic (mechanical) momentum is::

    P_kinetic = P - (e/c)·A = γ·m·v

Position Updates
~~~~~~~~~~~~~~~~

Spatial positions are updated in coordinate time using the kinetic momentum::

    Δx = v·h = (P_kinetic / (γ·m))·h

The 1/γ factor is **essential**: it ensures that velocity v = P_kinetic/(γ·m)
remains subluminal even as momentum grows with γ.

Velocity Calculation
~~~~~~~~~~~~~~~~~~~~

Velocity (beta) is computed from the coordinate-time displacement::

    β = v/c = Δx/(c·h)

Note: This does **not** include a γ factor in the denominator. The time dilation
is already accounted for in the position update formula.

Self-Consistency Iterations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ultra-relativistic particles (γ ≫ 1), forces depend strongly on γ through
the retarded field geometry (k-factor, field Lorentz contraction). The integrator
resolves the circular dependency γ → forces → P → γ through iterations:

1. Use γ_n-1 to compute retarded forces
2. Update conjugate momentum P_n from those forces
3. Update positions using the **same** γ_n-1: Δx = (h/(γ_n-1·m))·P_kinetic
4. Compute velocity: β = Δx/(c·h)
5. Derive two independent γ estimates:

   - From energy: γ_E = (Pt - e·Φ)/(mc)
   - From velocity: γ_V = 1/√(1-β²)

6. Check convergence: |γ_E - γ_V|/γ_E < ε (typically ε = 10⁻⁶)

If not converged, the next iteration uses γ_n = γ_E and repeats. Using a
**consistent** γ throughout each iteration for both forces and positions ensures
the velocity extracted from Δx corresponds physically to the computed momentum.

See :class:`core.self_consistency.SelfConsistencyConfig` for configuration.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .constants import C_MMNS
from .distances import (
    ChronoMatchResult,
    chrono_match_indices,
    chrono_match_indices_soa,
    compute_instantaneous_distance,
    compute_retarded_distance,
    compute_retarded_distance_soa,
)
from .beamline_geometry import compute_directional_visibility_mask
from .external_fields import (
    compute_uniform_external_field_impulse,
    evaluate_external_field_native,
    evaluate_external_field_si,
)
from .macroparticle_smearing import smear_source_samples
from .medina_radiation_reaction import (
    MedinaRadiationReactionResult,
    compute_medina_radiation_reaction,
)
from .self_consistency import (
    SelfConsistencyConfig,
    canonicalize_self_consistency_mode,
)
from .types import (
    BeamlineGeometryConfig,
    ChronoMatchingMode,
    GammaReconciliationMethod,
    MacroparticleSmearingConfig,
    MagneticDipoleConfig,
    ParticleState,
    SimulationType,
    StartupMode,
    Trajectory,
    TrajectoryArrays,
)
from .vectorized_interactions import (
    compute_vectorized_contributions,
    gather_external_samples,
    gather_external_samples_soa,
)


class GammaBlowupError(Exception):
    """Exception raised when gamma exceeds threshold during integration.

    This exception signals to the integration runner that the timestep should
    be reduced and the step retried. This is raised for ALL gamma blowups,
    including extreme values (> 1e20, NaN, or Inf). The integration runner
    will attempt timestep reduction, and only after exhausting retry attempts
    will the particle be marked as dead.

    Attributes
    ----------
    step_idx : int
        Integration step number where the blowup occurred.
    particle_idx : int
        Index of the particle that experienced the blowup.
    gamma_value : float
        The gamma value that triggered the blowup.
    iteration : int
        Self-consistency iteration number where the blowup was detected.
    is_hard_blowup : bool
        True if this was a hard blowup (NaN/Inf or > 1e20), used for logging.
    """

    def __init__(
        self,
        step_idx: int,
        particle_idx: int,
        gamma_value: float,
        iteration: int,
        is_hard_blowup: bool = False,
    ):
        self.step_idx = step_idx
        self.particle_idx = particle_idx
        self.gamma_value = gamma_value
        self.iteration = iteration
        self.is_hard_blowup = is_hard_blowup
        severity = "Hard" if is_hard_blowup else "Soft"
        super().__init__(
            f"{severity} gamma blowup at step {step_idx}, particle {particle_idx}: "
            f"γ={gamma_value:.2e} (iteration {iteration})"
        )


class SelfConsistencyNonConvergenceError(RuntimeError):
    """Raised when a particle step exhausts self-consistency iterations."""

    def __init__(
        self,
        step_idx: int,
        particle_idx: int,
        max_iterations: int,
        mass_shell_error: float,
    ) -> None:
        self.step_idx = step_idx
        self.particle_idx = particle_idx
        self.max_iterations = max_iterations
        self.mass_shell_error = mass_shell_error
        super().__init__(
            "Self-consistency failed to converge at step "
            f"{step_idx}, particle {particle_idx} after {max_iterations} iterations "
            f"(mass-shell error={mass_shell_error:.3e})"
        )


def _ensure_startup_metadata(state: ParticleState) -> None:
    """Initialize origin positions and beta averaging metadata if not present."""
    if "origin_x" not in state:
        state["origin_x"] = np.copy(state.get("x", np.array([])))
    if "origin_y" not in state:
        state["origin_y"] = np.copy(state.get("y", np.array([])))
    if "origin_z" not in state:
        state["origin_z"] = np.copy(state.get("z", np.array([])))
    if "beta_avg_x" not in state:
        state["beta_avg_x"] = np.copy(state.get("bx", np.array([])))
    if "beta_avg_y" not in state:
        state["beta_avg_y"] = np.copy(state.get("by", np.array([])))
    if "beta_avg_z" not in state:
        state["beta_avg_z"] = np.copy(state.get("bz", np.array([])))
    if "beta_samples" not in state:
        state["beta_samples"] = np.ones_like(state.get("x", np.array([])), dtype=float)
    if "radiation_power" not in state:
        state["radiation_power"] = np.zeros_like(
            state.get("x", np.array([])), dtype=float
        )
    if "radiation_energy" not in state:
        state["radiation_energy"] = np.zeros_like(
            state.get("x", np.array([])), dtype=float
        )
    if "radiation_energy_applied" not in state:
        state["radiation_energy_applied"] = np.zeros_like(
            state.get("x", np.array([])), dtype=float
        )
    if "mass_shell_projection_energy" not in state:
        state["mass_shell_projection_energy"] = np.zeros_like(
            state.get("x", np.array([])), dtype=float
        )
    if np.any(np.asarray(state.get("magnetic_dipole_active", []), dtype=float)):
        particle_template = state.get("x", np.array([]))
        for name in (
            "spin_x",
            "spin_y",
            "spin_z",
            "local_magnetic_field_x_t",
            "local_magnetic_field_y_t",
            "local_magnetic_field_z_t",
        ):
            if name not in state:
                state[name] = np.zeros_like(particle_template, dtype=float)


def _extract_self_consistency_params(
    self_consistency: Optional[SelfConsistencyConfig],
) -> tuple[bool, str, float, float, float, int, int]:
    """Extract self-consistency configuration parameters.

    Returns
    -------
    tuple[bool, str, float, float, float, int, int]
        A tuple containing (enabled, convergence_mode, target_ms_tolerance,
        mass_shell_tolerance, mass_shell_relaxation, max_iterations, verbosity).
    """
    is_enabled = self_consistency is not None and self_consistency.enabled
    convergence_mode = (
        self_consistency.convergence_mode
        if self_consistency is not None
        else "fixed_geometry"
    )
    convergence_mode = canonicalize_self_consistency_mode(convergence_mode)
    target_ms_tolerance = (
        self_consistency.target_ms_tolerance if self_consistency is not None else 1e-6
    )
    mass_shell_tolerance = (
        self_consistency.mass_shell_tolerance if self_consistency is not None else 1e-2
    )
    mass_shell_relaxation = (
        self_consistency.mass_shell_relaxation if self_consistency is not None else 0.7
    )
    max_iterations = (
        self_consistency.max_iterations if self_consistency is not None else 10
    )
    verbosity = self_consistency.verbosity if self_consistency is not None else 0

    return (
        is_enabled,
        convergence_mode,
        target_ms_tolerance,
        mass_shell_tolerance,
        mass_shell_relaxation,
        max_iterations,
        verbosity,
    )


def _initialize_result_state(current_state: ParticleState) -> ParticleState:
    """Create a copy of the current particle state for the next time step.

    Parameters
    ----------
    current_state : ParticleState
        The current state at this time step.

    Returns
    -------
    ParticleState
        A deep copy with all arrays duplicated, including dead particle metadata.
    """

    def _float_copy(name: str) -> np.ndarray:
        return np.array(current_state[name], dtype=float, copy=True)

    result = {
        "x": _float_copy("x"),
        "y": _float_copy("y"),
        "z": _float_copy("z"),
        "t": _float_copy("t"),
        "Px": _float_copy("Px"),
        "Py": _float_copy("Py"),
        "Pz": _float_copy("Pz"),
        "Pt": _float_copy("Pt"),
        "gamma": _float_copy("gamma"),
        "bx": _float_copy("bx"),
        "by": _float_copy("by"),
        "bz": _float_copy("bz"),
        "bdotx": _float_copy("bdotx"),
        "bdoty": _float_copy("bdoty"),
        "bdotz": _float_copy("bdotz"),
        "q": np.array(current_state["q"], dtype=float, copy=True),
        "q_species": np.array(
            current_state.get("q_species", current_state["q"]),
            dtype=float,
            copy=True,
        ),
        "q_observer": np.array(
            current_state.get("q_observer", current_state["q"]),
            dtype=float,
            copy=True,
        ),
        "q_source": np.array(
            current_state.get("q_source", current_state["q"]),
            dtype=float,
            copy=True,
        ),
        "macro_population": np.array(
            current_state.get("macro_population", np.ones_like(current_state["x"])),
            dtype=float,
            copy=True,
        ),
        "char_time": np.array(
            current_state.get("char_time", np.zeros_like(current_state["x"])),
            dtype=float,
            copy=True,
        ),
        "m": np.array(
            current_state.get("m", np.ones_like(current_state["x"])),
            dtype=float,
            copy=True,
        ),
        "m_species": np.array(
            current_state.get(
                "m_species", current_state.get("m", np.ones_like(current_state["x"]))
            ),
            dtype=float,
            copy=True,
        ),
        "dummy": np.zeros_like(current_state["bdotz"], dtype=float),
        "origin_x": _float_copy("origin_x"),
        "origin_y": _float_copy("origin_y"),
        "origin_z": _float_copy("origin_z"),
        "beta_avg_x": _float_copy("beta_avg_x"),
        "beta_avg_y": _float_copy("beta_avg_y"),
        "beta_avg_z": _float_copy("beta_avg_z"),
        "beta_samples": _float_copy("beta_samples"),
        "radiation_power": np.zeros_like(current_state["x"], dtype=float),
        "radiation_energy": np.zeros_like(current_state["x"], dtype=float),
        "radiation_energy_applied": np.zeros_like(current_state["x"], dtype=float),
        "mass_shell_projection_energy": np.zeros_like(current_state["x"], dtype=float),
    }

    magnetic_fields = (
        "spin_x",
        "spin_y",
        "spin_z",
        "local_magnetic_field_x_t",
        "local_magnetic_field_y_t",
        "local_magnetic_field_z_t",
        "magnetic_moment_j_per_t",
        "magnetic_moment_native",
        "spin_quantum_number",
        "gyromagnetic_ratio_rad_s_t",
        "magnetic_dipole_active",
        "spin_precession_active",
        "stern_gerlach_active",
        "dipole_source_canonical_ready",
    )
    for name in magnetic_fields:
        if name in current_state:
            result[name] = np.array(current_state[name], dtype=float, copy=True)

    if "charge_source_canonical_ready" in current_state:
        result["charge_source_canonical_ready"] = np.array(
            current_state["charge_source_canonical_ready"],
            dtype=bool,
            copy=True,
        )

    # Preserve dead particle metadata to prevent redundant logging
    if "_dead_particles" in current_state:
        result["_dead_particles"] = np.copy(current_state["_dead_particles"])
    if "_particle_failure_info" in current_state:
        # Deep copy the failure info dict
        result["_particle_failure_info"] = {
            k: v.copy() if isinstance(v, dict) else v
            for k, v in current_state["_particle_failure_info"].items()
        }

    return result


def _initialize_medina_step_state(result: ParticleState) -> None:
    """Reset Medina diagnostics and accepted-force history for one trial step.

    The caller populates the force-history fields only after it has recomputed
    the complete non-radiation-reaction force for the current trial.  Starting
    from invalid history here is important: a rejected adaptive trial or a run
    that changes radiation-reaction mode cannot leak a stale force derivative
    into a later accepted step.
    """

    particle_template = result["x"]
    for name in (
        "radiation_reaction_work",
        "medina_cross_field_energy",
        "medina_cross_field_energy_change",
        "medina_external_force_x",
        "medina_external_force_y",
        "medina_external_force_z",
    ):
        result[name] = np.zeros_like(particle_template, dtype=float)
    result["medina_external_force_sample_time"] = np.full_like(
        particle_template,
        np.nan,
        dtype=float,
    )
    result["medina_force_derivative_ready"] = np.zeros_like(
        particle_template,
        dtype=bool,
    )
    result["medina_impulse_capped"] = np.zeros_like(
        particle_template,
        dtype=bool,
    )


def _get_state_scalar(
    state: ParticleState, key: str, particle_idx: int, fallback_key: str | None = None
):
    values = state.get(key)
    if values is None and fallback_key is not None:
        values = state[fallback_key]
    if values is None:
        raise KeyError(key)
    if hasattr(values, "__getitem__"):
        return values[particle_idx]
    return values


def _get_particle_source_charge(state: ParticleState, particle_idx: int):
    """Extract source charge for a single particle."""
    return _get_state_scalar(state, "q_source", particle_idx, "q")


def _get_particle_observer_charge(state: ParticleState, particle_idx: int):
    """Extract observer charge for a single particle."""
    return _get_state_scalar(state, "q_observer", particle_idx, "q")


def _get_particle_charge(state: ParticleState, particle_idx: int):
    """Backward-compatible alias for observer charge extraction."""
    return _get_particle_observer_charge(state, particle_idx)


def _get_particle_mass(state: ParticleState, particle_idx: int):
    """Extract observer/species mass for a single particle."""
    return _get_state_scalar(state, "m_species", particle_idx, "m")


def effective_observer_charge(charge_native: float) -> float:
    """Backward-compatible helper retained for legacy tests/imports.

    Production force paths now read explicit ``q_observer`` metadata directly.
    This shim preserves the historical equations-module contract for synthetic
    tests that pass plain scalar charges rather than native-unit macroparticle
    source charges.
    """
    return float(charge_native)


def _scalar_potential_momentum_contribution(
    charge: float, scalar_potential: float
) -> np.float64:
    """Return qΦ/c in the solver's momentum units."""
    return np.float64(charge * scalar_potential / C_MMNS)


def _mechanical_momentum_components(
    *,
    px: float,
    py: float,
    pz: float,
    particle_mass: float,
    field_x: float,
    field_y: float,
    field_z: float,
) -> tuple[float, float, float]:
    return (
        float(np.float64(px) - np.float64(field_x * particle_mass)),
        float(np.float64(py) - np.float64(field_y * particle_mass)),
        float(np.float64(pz) - np.float64(field_z * particle_mass)),
    )


def _stable_kinetic_energy_native(
    mechanical_momentum: Sequence[float] | np.ndarray,
    particle_mass: float,
) -> float:
    """Return kinetic energy without subtracting the rest energy."""

    momentum = np.asarray(mechanical_momentum, dtype=float)
    if momentum.shape != (3,) or not np.all(np.isfinite(momentum)):
        raise ValueError("mechanical_momentum must be a finite three-vector")
    mass = float(particle_mass)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("particle_mass must be finite and positive")
    magnitude = float(
        np.hypot(
            np.hypot(abs(float(momentum[0])), abs(float(momentum[1]))),
            abs(float(momentum[2])),
        )
    )
    rest_momentum = mass * C_MMNS
    total_momentum = float(np.hypot(rest_momentum, magnitude))
    return float(C_MMNS * magnitude**2 / (total_momentum + rest_momentum))


def _canonical_pt_from_mechanical_mass_shell(
    *,
    px: float,
    py: float,
    pz: float,
    particle_mass: float,
    scalar_potential_contribution: float,
    field_x: float = 0.0,
    field_y: float = 0.0,
    field_z: float = 0.0,
) -> tuple[float, float]:
    mechanical_px, mechanical_py, mechanical_pz = _mechanical_momentum_components(
        px=px,
        py=py,
        pz=pz,
        particle_mass=particle_mass,
        field_x=field_x,
        field_y=field_y,
        field_z=field_z,
    )
    p_spatial_sq = mechanical_px**2 + mechanical_py**2 + mechanical_pz**2
    kinetic_pt = np.sqrt(p_spatial_sq + np.float64(particle_mass * C_MMNS) ** 2)
    canonical_pt = kinetic_pt + np.float64(scalar_potential_contribution)
    return float(kinetic_pt), float(canonical_pt)


def _four_velocity_native(beta: np.ndarray) -> np.ndarray:
    """Return native contravariant ``u=(gamma*c, gamma*c*beta)``."""

    beta_squared = float(beta @ beta)
    if beta_squared >= 1.0:
        raise ValueError("beta magnitude must be less than one")
    gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    return np.concatenate(([gamma * C_MMNS], gamma * C_MMNS * beta))


def _project_normalized_spin(
    spin_four_vector: np.ndarray,
    four_velocity: np.ndarray,
    target_rest_magnitude: float,
) -> np.ndarray:
    """Project spin onto u-orthogonal space and restore its invariant norm."""

    from .rfs import minkowski_dot

    target = float(target_rest_magnitude)
    if target == 0.0:
        return np.zeros(4, dtype=float)
    projected = np.asarray(spin_four_vector, dtype=float) - np.asarray(
        four_velocity, dtype=float
    ) * (minkowski_dot(four_velocity, spin_four_vector) / C_MMNS**2)
    norm_squared = -minkowski_dot(projected, projected)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise FloatingPointError("RFS spin update produced a non-spacelike spin")
    return np.asarray(projected * (target / np.sqrt(norm_squared)), dtype=float)


def _advance_rfs_rest_spin(
    rest_spin: np.ndarray,
    *,
    beta_start: np.ndarray,
    beta_end: np.ndarray,
    field_tensor: np.ndarray,
    partial_f: np.ndarray,
    charge_native: float,
    mass_amu: float,
    magnetic_moment_native: float,
    spin_quantum_number: float,
    proper_time_step_ns: float,
    applied_radiation_reaction_force_native: np.ndarray | None = None,
    analytic_antisymmetric_response: np.ndarray | None = None,
    analytic_partial_antisymmetric_response: np.ndarray | None = None,
    analytic_response_contraction: str = "python",
) -> np.ndarray:
    """Advance normalized RFS four-spin and return rest-frame polarization.

    When supplied, ``applied_radiation_reaction_force_native`` is the Medina
    lab three-force that the momentum update actually received after any
    numerical impulse cap.  The same step-averaged force is converted at the
    start and midpoint four-velocity/spin states so both Runge--Kutta stages
    include the matching Fermi--Walker correction.
    """

    from .magnetic_dipole import (
        HBAR_NATIVE,
        boost_rest_polarization,
        rest_polarization_from_four_vector,
    )
    from .rfs import (
        rfs_charge_radiation_reaction_terms_native,
        rfs_spin_rhs_native,
    )
    from .antisymmetric_response_rfs import antisymmetric_response_rfs_native

    rest_spin = np.asarray(rest_spin, dtype=float)
    polarization = float(np.linalg.norm(rest_spin))
    if polarization == 0.0 or proper_time_step_ns == 0.0:
        return rest_spin.copy()
    invariant_spin = float(spin_quantum_number) * HBAR_NATIVE
    if invariant_spin <= 0.0:
        return rest_spin.copy()
    applied_rr_force = (
        np.zeros(3, dtype=float)
        if applied_radiation_reaction_force_native is None
        else np.asarray(applied_radiation_reaction_force_native, dtype=float)
    )
    if applied_rr_force.shape != (3,) or not np.all(np.isfinite(applied_rr_force)):
        raise ValueError(
            "applied_radiation_reaction_force_native must contain three finite "
            "components"
        )
    radiation_reaction_active = bool(np.any(applied_rr_force != 0.0))
    packed_analytic_response = (
        None
        if analytic_antisymmetric_response is None
        else np.asarray(analytic_antisymmetric_response, dtype=float)
    )
    partial_packed_analytic_response = (
        None
        if analytic_partial_antisymmetric_response is None
        else np.asarray(analytic_partial_antisymmetric_response, dtype=float)
    )
    if (packed_analytic_response is None) != (partial_packed_analytic_response is None):
        raise ValueError(
            "analytical response and its derivative must be supplied together"
        )
    if analytic_response_contraction not in ("python", "numba_strict_serial"):
        raise ValueError(
            "analytic_response_contraction must be 'python' or " "'numba_strict_serial'"
        )

    def analytical_spin_rhs(
        four_velocity: np.ndarray, spin_four_vector: np.ndarray
    ) -> np.ndarray:
        if packed_analytic_response is None or partial_packed_analytic_response is None:
            return np.zeros(4, dtype=float)
        if analytic_response_contraction == "numba_strict_serial":
            from .contracted_antisymmetric_response_numba import (
                antisymmetric_response_rfs_strict_serial,
            )

            return antisymmetric_response_rfs_strict_serial(
                four_velocity,
                spin_four_vector,
                packed_analytic_response,
                partial_packed_analytic_response,
                float(charge_native),
                float(mass_amu),
                float(magnetic_moment_native),
                float(invariant_spin),
            )[3]
        return antisymmetric_response_rfs_native(
            four_velocity_mm_ns=four_velocity,
            spin_four_vector=spin_four_vector,
            antisymmetric_response=packed_analytic_response,
            partial_antisymmetric_response=partial_packed_analytic_response,
            charge_native=charge_native,
            mass_amu=mass_amu,
            magnetic_moment_native=magnetic_moment_native,
            invariant_spin_native=invariant_spin,
        ).spin_rhs

    u_start = _four_velocity_native(beta_start)
    u_end = _four_velocity_native(beta_end)
    beta_midpoint = 0.5 * (beta_start + beta_end)
    if float(beta_midpoint @ beta_midpoint) >= 1.0:
        beta_midpoint *= np.nextafter(1.0, 0.0) / np.linalg.norm(beta_midpoint)
    u_midpoint = _four_velocity_native(beta_midpoint)
    spin_start = boost_rest_polarization(rest_spin, beta_start)
    delta_tau_ns = float(proper_time_step_ns)

    derivative_start = rfs_spin_rhs_native(
        four_velocity_mm_ns=u_start,
        spin_four_vector=spin_start,
        field_tensor=field_tensor,
        partial_f=partial_f,
        charge_native=charge_native,
        mass_amu=mass_amu,
        magnetic_moment_native=magnetic_moment_native,
        invariant_spin_native=invariant_spin,
    )
    derivative_start += analytical_spin_rhs(u_start, spin_start)
    if radiation_reaction_active:
        derivative_start += rfs_charge_radiation_reaction_terms_native(
            four_velocity_mm_ns=u_start,
            spin_four_vector=spin_start,
            applied_radiation_reaction_force_native=applied_rr_force,
            mass_amu=mass_amu,
        ).spin_rhs_correction
    spin_midpoint = _project_normalized_spin(
        spin_start + 0.5 * delta_tau_ns * derivative_start,
        u_midpoint,
        polarization,
    )
    derivative_midpoint = rfs_spin_rhs_native(
        four_velocity_mm_ns=u_midpoint,
        spin_four_vector=spin_midpoint,
        field_tensor=field_tensor,
        partial_f=partial_f,
        charge_native=charge_native,
        mass_amu=mass_amu,
        magnetic_moment_native=magnetic_moment_native,
        invariant_spin_native=invariant_spin,
    )
    derivative_midpoint += analytical_spin_rhs(u_midpoint, spin_midpoint)
    if radiation_reaction_active:
        derivative_midpoint += rfs_charge_radiation_reaction_terms_native(
            four_velocity_mm_ns=u_midpoint,
            spin_four_vector=spin_midpoint,
            applied_radiation_reaction_force_native=applied_rr_force,
            mass_amu=mass_amu,
        ).spin_rhs_correction
    spin_end = _project_normalized_spin(
        spin_start + delta_tau_ns * derivative_midpoint,
        u_end,
        polarization,
    )
    rest_end = rest_polarization_from_four_vector(spin_end, beta_end)
    rest_norm = float(np.linalg.norm(rest_end))
    if rest_norm == 0.0:
        raise FloatingPointError("RFS spin update collapsed a nonzero spin")
    return rest_end * (polarization / rest_norm)


def _external_tensor_gradient(
    electric_field_native: np.ndarray,
    magnetic_field_native: np.ndarray,
    magnetic_gradient_native_per_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build native ``F`` and ``partial_lambda F`` from native field values."""

    from .rfs import electromagnetic_field_tensor_native

    field_tensor = electromagnetic_field_tensor_native(
        tuple(float(value) for value in electric_field_native),
        tuple(float(value) for value in magnetic_field_native),
    )
    partial_f = np.zeros((4, 4, 4), dtype=float)
    for coordinate in range(3):
        partial_f[coordinate + 1] = electromagnetic_field_tensor_native(
            (0.0, 0.0, 0.0),
            tuple(
                float(value) for value in magnetic_gradient_native_per_mm[:, coordinate]
            ),
        )
    return field_tensor, partial_f


def _refresh_kinematics_from_canonical_momentum(
    result: ParticleState,
    current_state: ParticleState,
    particle_idx: int,
    h: float,
    particle_mass: float,
    field_x: float,
    field_y: float,
    field_z: float,
) -> tuple[float, float, float, float, float, float, float]:
    mechanical_px, mechanical_py, mechanical_pz = _mechanical_momentum_components(
        px=result["Px"][particle_idx],
        py=result["Py"][particle_idx],
        pz=result["Pz"][particle_idx],
        particle_mass=particle_mass,
        field_x=field_x,
        field_y=field_y,
        field_z=field_z,
    )

    gamma = float(result["gamma"][particle_idx])
    result["t"][particle_idx] = current_state["t"][particle_idx] + h * gamma
    result["x"][particle_idx] = (
        current_state["x"][particle_idx] + h / particle_mass * mechanical_px
    )
    result["y"][particle_idx] = (
        current_state["y"][particle_idx] + h / particle_mass * mechanical_py
    )
    result["z"][particle_idx] = (
        current_state["z"][particle_idx] + h / particle_mass * mechanical_pz
    )

    beta_denom = gamma * particle_mass * C_MMNS
    if beta_denom > 0.0:
        beta_x = float(mechanical_px / beta_denom)
        beta_y = float(mechanical_py / beta_denom)
        beta_z = float(mechanical_pz / beta_denom)
    else:
        beta_x = beta_y = beta_z = 0.0
    beta_x, beta_y, beta_z = _limit_beta_magnitude(beta_x, beta_y, beta_z)
    result["bx"][particle_idx] = beta_x
    result["by"][particle_idx] = beta_y
    result["bz"][particle_idx] = beta_z

    coordinate_dt = result["t"][particle_idx] - current_state["t"][particle_idx]
    time_factor = C_MMNS * coordinate_dt
    if time_factor != 0.0:
        result["bdotx"][particle_idx] = (
            beta_x - current_state["bx"][particle_idx]
        ) / time_factor
        result["bdoty"][particle_idx] = (
            beta_y - current_state["by"][particle_idx]
        ) / time_factor
        result["bdotz"][particle_idx] = (
            beta_z - current_state["bz"][particle_idx]
        ) / time_factor

    return (
        beta_x,
        beta_y,
        beta_z,
        float(coordinate_dt),
        float(mechanical_px),
        float(mechanical_py),
        float(mechanical_pz),
    )


def _get_particle_char_time(state: ParticleState, particle_idx: int):
    """Extract characteristic time for a single particle, handling scalar or array."""
    char_time = state["char_time"]
    if hasattr(char_time, "__getitem__"):
        return char_time[particle_idx]
    return char_time


def _compute_approximate_retarded_distance(
    current_state: ParticleState,
    external_state: ParticleState,
    particle_idx: int,
    time_step_idx: int,
) -> tuple[dict, np.ndarray]:
    """Compute approximate retarded distance using constant velocity assumption.

    This is used in APPROXIMATE_BACK_HISTORY startup mode to estimate retardation
    effects when full historical data is not yet available.

    The retarded distance accounts for source motion during light travel time using
    the Liénard-Wiechert formula: R_ret = R / (1 - β_source·n̂)

    For numerical stability at ultra-relativistic energies (γ > 10⁵), we use the
    algebraically equivalent factored form:
        R_ret = R × (1 + β·n̂) / (1 - (β·n̂)²)

    This formulation divides by a denominator that goes to zero ~2× more slowly,
    reducing catastrophic cancellation and providing better precision.

    Validated for:
        - 500 GeV electrons (γ ≈ 978,474, β ≈ 0.9999999999995)
        - 20 TeV protons (γ ≈ 21,321, β ≈ 0.999999999)

    Parameters
    ----------
    current_state : ParticleState
        Observer particle state (the bunch being updated).
    external_state : ParticleState
        Source particle state (the external bunch).
    particle_idx : int
        Index of the observer particle within current_state.
    time_step_idx : int
        Current trajectory index.

    Returns
    -------
    tuple[dict, np.ndarray]
        A tuple of (nhat dictionary with corrected R, bounded_indices array).
    """
    sample_count = len(external_state["x"])
    indices_bounded = np.full(sample_count, time_step_idx, dtype=int)

    # Compute instantaneous distance and direction vector n̂
    nhat = compute_instantaneous_distance(current_state, external_state, particle_idx)

    # Compute β_source · n̂ where n̂ points FROM source TO observer
    beta_ext_dot_nhat = (
        external_state["bx"] * nhat["nx"]
        + external_state["by"] * nhat["ny"]
        + external_state["bz"] * nhat["nz"]
    )

    # Use factored form for numerical stability: R_ret = R × (1+β·n̂) / (1-(β·n̂)²)
    # This is algebraically equivalent to R / (1 - β·n̂) but more stable because:
    #   (1 - (β·n̂)²) = (1 - β·n̂)(1 + β·n̂)
    # The factored denominator goes to zero ~2× more slowly as β·n̂ → 1
    numerator = 1.0 + beta_ext_dot_nhat
    denominator = 1.0 - beta_ext_dot_nhat**2

    # Clamp denominator to prevent division by zero
    # Physical interpretation: β·n̂ → 1 means source moving at light speed away
    # from observer. Light never catches up, so R_ret → ∞ (correct physics).
    # Clamping gives huge but finite R_ret; forces become negligible (as expected).
    #
    # k_threshold = 1e-12 supports particles up to γ ≈ 7×10⁵:
    #   - Covers 500 GeV electrons (γ ≈ 978,474)
    #   - Covers 20 TeV protons (γ ≈ 21,321)
    #   - R_ret saturates at ~10¹² mm (1000 km) in extreme cases
    #   - Such large R leads to negligible forces (correct behavior)
    #
    # Force calculation has additional safety: K_CUTOFF_HARD = 1e-20 filters
    # interactions where k = (1 - β·n̂) is extremely small.
    k_threshold = 1e-12
    denominator = np.where(
        np.abs(denominator) < k_threshold,
        np.copysign(k_threshold, denominator),  # Preserve sign for negative β·n̂
        denominator,
    )

    # Apply correction to retarded distance
    nhat["R"] = nhat["R"] * numerator / denominator

    return nhat, indices_bounded


def _compute_full_retarded_distance(
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    time_step_idx: int,
    particle_idx: int,
    chrono_mode: ChronoMatchingMode,
    self_consistency: Optional[SelfConsistencyConfig] = None,
    timestep_h: float = 1e-3,
    traj_soa: Optional[TrajectoryArrays] = None,
    traj_ext_soa: Optional[TrajectoryArrays] = None,
) -> tuple[dict, np.ndarray, Optional[ChronoMatchResult]]:
    """Compute retarded distance using full chronological matching.

    This uses the complete trajectory history to find the proper retarded time
    for each external source particle.

    Returns
    -------
    tuple[dict, np.ndarray, Optional[ChronoMatchResult]]
        A tuple of (nhat dictionary, bounded_indices array, chrono_match_result).
        chrono_match_result is None if interpolation is disabled.
    """
    # Check if chrono-match interpolation is enabled
    chrono_interpolate = False
    chrono_tolerance = 1e-3
    chrono_high_precision = False
    chrono_adaptive_tolerance = False
    verbosity = 0

    if self_consistency is not None:
        chrono_interpolate = self_consistency.chrono_interpolate
        chrono_tolerance = self_consistency.chrono_tolerance
        chrono_high_precision = getattr(
            self_consistency, "chrono_high_precision", False
        )
        chrono_adaptive_tolerance = getattr(
            self_consistency, "chrono_adaptive_tolerance", False
        )
        verbosity = self_consistency.verbosity

    if traj_soa is not None and traj_ext_soa is not None:
        retarded_result = chrono_match_indices_soa(
            traj_soa,
            traj_ext_soa,
            time_step_idx,
            particle_idx,
            mode=chrono_mode,
            interpolate=chrono_interpolate,
            tolerance=chrono_tolerance,
            verbosity=verbosity,
            high_precision=chrono_high_precision,
            adaptive_tolerance=chrono_adaptive_tolerance,
            timestep_h=timestep_h,
        )
    else:
        retarded_result = chrono_match_indices(
            trajectory,
            trajectory_ext,
            time_step_idx,
            particle_idx,
            mode=chrono_mode,
            interpolate=chrono_interpolate,
            tolerance=chrono_tolerance,
            verbosity=verbosity,
            high_precision=chrono_high_precision,
            adaptive_tolerance=chrono_adaptive_tolerance,
            timestep_h=timestep_h,
        )

    # Handle both plain index-array returns and ChronoMatchResult payloads.
    if isinstance(retarded_result, ChronoMatchResult):
        retarded_indices = retarded_result.indices
        chrono_match_result = retarded_result
    else:
        retarded_indices = retarded_result
        chrono_match_result = None

    max_external_idx = len(trajectory_ext) - 1
    indices_bounded = np.minimum(np.maximum(retarded_indices, 0), max_external_idx)

    if traj_soa is not None and traj_ext_soa is not None:
        nhat = compute_retarded_distance_soa(
            traj_soa, traj_ext_soa, time_step_idx, particle_idx, indices_bounded
        )
    else:
        nhat = compute_retarded_distance(
            trajectory,
            trajectory_ext,
            time_step_idx,
            particle_idx,
            indices_bounded,
        )

    return nhat, indices_bounded, chrono_match_result


def _calculate_travel_distance(
    origin_position: tuple[float, float, float],
    current_position: tuple[float, float, float],
) -> float:
    """Calculate Euclidean distance between origin and current position."""
    dx = current_position[0] - origin_position[0]
    dy = current_position[1] - origin_position[1]
    dz = current_position[2] - origin_position[2]
    return float(np.sqrt(dx**2 + dy**2 + dz**2))


def _compute_gating_threshold(
    nhat: dict,
    beta_avg_x: float,
    beta_avg_y: float,
    beta_avg_z: float,
) -> float:
    """Compute the minimum travel distance before external forces are applied.

    The threshold ensures the observer particle has traveled far enough that
    light from the external source's initial position could have reached it.

    Light propagates at speed c. For a particle moving at velocity β relative
    to the source, the relative closing speed is c(1 - β·n̂), where n̂ points
    from source to observer.

    Physical examples (for approaching particles, β·n̂ < 0):
    - Stationary (β=0): threshold = 0 (forces apply immediately)
    - Low velocity (β=0.1): threshold ≈ 0.091·R
    - Moderate (β=0.5): threshold ≈ 0.33·R
    - Relativistic (β=0.9): threshold ≈ 0.47·R
    - Ultra-relativistic (β→1): threshold → R/2 (approaches limit, never exceeds)

    Formula: threshold = β·R / (1 - β·n̂)

    Special handling for β·n̂ ≥ 1 (receding at or above light speed):
    Return very large threshold (effectively infinite) to suppress forces.
    """
    # Handle edge case: particles receding at or faster than light speed
    # For β·n̂ ≥ 1, light never catches the observer, so threshold = ∞
    # Use a very large but finite value to avoid numerical issues
    LARGE_THRESHOLD = 1e12  # effectively infinite for simulation purposes

    # For denominators ≤ 0 or very small, use large threshold
    # Also handle case where β·n̂ is very close to 1 (denominator near 0)
    MIN_DENOMINATOR = 1e-6  # corresponds to β·n̂ = 0.999999

    # Calculate particle speed magnitude
    beta_magnitude = np.sqrt(beta_avg_x**2 + beta_avg_y**2 + beta_avg_z**2)

    # Most sweep runs use only a few external particles. Avoid constructing
    # several tiny NumPy temporaries in this per-particle, per-iteration gate.
    R_values = nhat["R"]
    if R_values.size <= 8:
        max_threshold = 0.0
        for R, nx, ny, nz in zip(R_values, nhat["nx"], nhat["ny"], nhat["nz"]):
            beta_avg_dot_nhat = beta_avg_x * nx + beta_avg_y * ny + beta_avg_z * nz
            denominator = 1.0 - beta_avg_dot_nhat
            if denominator > MIN_DENOMINATOR:
                threshold = beta_magnitude * R / denominator
                if threshold > max_threshold:
                    max_threshold = float(threshold)
            elif LARGE_THRESHOLD > max_threshold:
                max_threshold = LARGE_THRESHOLD
        return max(max_threshold, 0.0)

    beta_avg_dot_nhat = (
        beta_avg_x * nhat["nx"] + beta_avg_y * nhat["ny"] + beta_avg_z * nhat["nz"]
    )

    # Compute denominator: (1 - β·n̂)
    # β·n̂ < 0: approaching → denominator > 1 → small threshold (meet quickly)
    # β·n̂ > 0: receding → denominator < 1 → large threshold (takes longer)
    # β·n̂ → 1: receding at c → denominator → 0 → threshold → ∞ (never meet)
    denominators = 1.0 - beta_avg_dot_nhat

    thresholds = np.where(
        denominators > MIN_DENOMINATOR,
        beta_magnitude * R_values / denominators,
        LARGE_THRESHOLD,
    )

    if thresholds.size > 0:
        return float(np.max(np.maximum(thresholds, 0.0)))
    return 0.0


def _should_apply_external_forces(
    startup_mode: StartupMode,
    sim_type: SimulationType,
    nhat: dict,
    current_state: ParticleState,
    particle_idx: int,
) -> bool:
    """Determine whether external forces should be applied to this particle.

    In COLD_START mode, wall/image-style startup suppresses external source
    forces until the observer has traveled far enough from its origin for
    retardation effects to be meaningful. BUNCH_TO_BUNCH cross-bunch sources
    are gated by the same startup rule; intra-bunch space-charge calculations
    are handled separately in the dedicated space-charge block below and are
    not affected by this helper.
    """
    if startup_mode is not StartupMode.COLD_START or nhat["R"].size == 0:
        return True

    origin_position = (
        current_state["origin_x"][particle_idx],
        current_state["origin_y"][particle_idx],
        current_state["origin_z"][particle_idx],
    )

    current_position = (
        current_state["x"][particle_idx],
        current_state["y"][particle_idx],
        current_state["z"][particle_idx],
    )

    travel_distance = _calculate_travel_distance(origin_position, current_position)

    beta_avg_x = current_state["beta_avg_x"][particle_idx]
    beta_avg_y = current_state["beta_avg_y"][particle_idx]
    beta_avg_z = current_state["beta_avg_z"][particle_idx]

    threshold = _compute_gating_threshold(nhat, beta_avg_x, beta_avg_y, beta_avg_z)

    return travel_distance >= threshold


def _get_current_particle_gamma_and_beta(
    current_state: ParticleState,
    result_state: ParticleState,
    particle_idx: int,
    sc_iteration: int,
    sc_enabled: bool,
) -> tuple[float, tuple[float, float, float]]:
    """Get gamma and beta values for the current self-consistency iteration.

    On the first iteration, use values from the input state.
    On subsequent iterations, use the updated values from the result state.
    """
    if sc_enabled and sc_iteration > 0:
        gamma = result_state["gamma"][particle_idx]
        beta_vector = (
            result_state["bx"][particle_idx],
            result_state["by"][particle_idx],
            result_state["bz"][particle_idx],
        )
    else:
        gamma = current_state["gamma"][particle_idx]
        beta_vector = (
            current_state["bx"][particle_idx],
            current_state["by"][particle_idx],
            current_state["bz"][particle_idx],
        )

    return gamma, beta_vector


def _limit_beta_magnitude(
    beta_x: float, beta_y: float, beta_z: float
) -> tuple[float, float, float]:
    """Ensure beta magnitude stays below the speed of light.

    Uses float64 precision to allow beta extremely close to c (1 - 1e-16).
    This allows gamma up to ~1e8 while maintaining numerical stability.

    Returns
    -------
    tuple[float, float, float]
        The (possibly scaled) beta components (βx, βy, βz).
    """
    # Use float64 for high precision in beta calculations
    bx64 = np.float64(beta_x)
    by64 = np.float64(beta_y)
    bz64 = np.float64(beta_z)

    beta_magnitude = np.sqrt(bx64**2 + by64**2 + bz64**2)

    # Allow beta extremely close to 1.0, limited only by float64 precision
    # 1 - 1e-16 corresponds to gamma ~ 1e8
    beta_max_allowed = np.float64(1.0) - np.float64(1e-16)

    if beta_magnitude >= beta_max_allowed:
        scale_factor = beta_max_allowed / beta_magnitude
        return (
            float(bx64 * scale_factor),
            float(by64 * scale_factor),
            float(bz64 * scale_factor),
        )

    return beta_x, beta_y, beta_z


def _calculate_one_minus_beta_squared(
    beta_x: float, beta_y: float, beta_z: float
) -> float:
    """Calculate 1 - β² using Kahan compensated summation for numerical stability.

    At ultra-relativistic speeds (β → 1), direct calculation of 1 - β² suffers from
    catastrophic cancellation. This function uses Kahan summation to accurately
    compute β² first, then returns 1 - β².

    Parameters
    ----------
    beta_x, beta_y, beta_z : float
        Velocity components normalized by c.

    Returns
    -------
    float
        1 - β² with improved numerical accuracy for β ≈ 1.
    """
    # Use float64 for high precision
    bx64 = np.float64(beta_x)
    by64 = np.float64(beta_y)
    bz64 = np.float64(beta_z)

    # Kahan compensated summation for β² = βx² + βy² + βz²
    # This reduces floating-point errors when summing squares
    sum_beta_sq = np.float64(0.0)
    compensation = np.float64(0.0)

    for beta_component in [bx64, by64, bz64]:
        term = beta_component**2 - compensation
        temp_sum = sum_beta_sq + term
        compensation = (temp_sum - sum_beta_sq) - term
        sum_beta_sq = temp_sum

    beta_squared = sum_beta_sq

    # Clamp beta_squared just below 1.0 to prevent infinity
    max_beta_squared = (np.float64(1.0) - np.float64(1e-16)) ** 2
    if beta_squared >= max_beta_squared:
        beta_squared = max_beta_squared

    one_minus_beta_sq = np.float64(1.0) - beta_squared

    # Safety check: ensure non-negative result
    if one_minus_beta_sq <= np.float64(0.0):
        one_minus_beta_sq = np.float64(1.0) - max_beta_squared

    return float(one_minus_beta_sq)


def _calculate_gamma_from_beta(beta_x: float, beta_y: float, beta_z: float) -> float:
    """Calculate Lorentz factor from velocity components.

    γ = 1 / √(1 - β²)

    Uses Kahan summation and float64 precision to handle extremely relativistic
    particles accurately, avoiding catastrophic cancellation at β → 1.
    """
    one_minus_beta_sq = _calculate_one_minus_beta_squared(beta_x, beta_y, beta_z)
    return float(1.0 / np.sqrt(np.float64(one_minus_beta_sq)))


def _canonicalize_radiation_reaction_mode(mode: Optional[str]) -> str:
    """Normalize radiation-reaction mode names."""
    if mode is None:
        return "off"

    normalized = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "none": "off",
        "disabled": "off",
        "diagnostic": "diagnostic_only",
        "diagnostics": "diagnostic_only",
        "power_damping": "power_matched_damping",
        "damping": "power_matched_damping",
        "medina": "medina_lad",
        "medina_rr": "medina_lad",
        "lad_medina": "medina_lad",
    }
    normalized = aliases.get(normalized, normalized)
    valid_modes = {
        "off",
        "diagnostic_only",
        "power_matched_damping",
        "medina_lad",
    }
    if normalized not in valid_modes:
        raise ValueError(
            f"Unknown radiation_reaction_mode {mode!r}; expected one of "
            f"{sorted(valid_modes)}"
        )
    return normalized


def _compute_lienard_radiated_power(
    charge: float,
    beta: tuple[float, float, float],
    beta_dot_t: tuple[float, float, float],
    gamma: float,
) -> float:
    """Return instantaneous Liénard radiated power in native energy/ns units.

    ``beta_dot_t`` is dβ/dt in coordinate time. The solver's stored ``bdot``
    fields are dβ/d(ct), so callers should multiply stored ``bdot`` by
    :data:`C_MMNS` before calling this helper.
    """
    b_x, b_y, b_z = beta
    bd_x, bd_y, bd_z = beta_dot_t
    accel_sq = bd_x * bd_x + bd_y * bd_y + bd_z * bd_z
    if accel_sq <= 0.0 or charge == 0.0 or gamma <= 0.0:
        return 0.0

    cross_x = b_y * bd_z - b_z * bd_y
    cross_y = b_z * bd_x - b_x * bd_z
    cross_z = b_x * bd_y - b_y * bd_x
    cross_sq = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
    transverse_term = accel_sq - cross_sq
    if transverse_term <= 0.0:
        return 0.0

    return float((2.0 * charge**2 / (3.0 * C_MMNS)) * gamma**6 * transverse_term)


def _accepted_medina_force_derivative(
    *,
    current_state: ParticleState,
    particle_idx: int,
    current_force: tuple[float, float, float],
    current_sample_time: float,
) -> tuple[tuple[float, float, float], bool]:
    """Backward-difference two accepted midpoint force samples.

    Each force sample is the average non-radiation-reaction mechanical force
    over one on-shell predictor interval and is timestamped at that interval's
    midpoint.  A later Medina endpoint kick may shift the accepted endpoint
    time by second-order terms; it does not retrospectively move the force
    sample.  The resulting derivative is first-order accurate at the current
    predictor midpoint.  A nonlinear iteration never writes ``current_state``;
    rejected adaptive trials are therefore excluded automatically.

    Missing history and history whose timestamp lies after the accepted state
    are treated as unprimed.  The caller must not apply a Medina impulse until
    this function reports readiness.
    """

    force_keys = (
        "medina_external_force_x",
        "medina_external_force_y",
        "medina_external_force_z",
    )
    if any(key not in current_state for key in force_keys) or (
        "medina_external_force_sample_time" not in current_state
    ):
        return (0.0, 0.0, 0.0), False

    try:
        previous_force = np.asarray(
            [current_state[key][particle_idx] for key in force_keys],
            dtype=float,
        )
        previous_sample_time = float(
            current_state["medina_external_force_sample_time"][particle_idx]
        )
        accepted_state_time = float(current_state["t"][particle_idx])
    except (IndexError, KeyError, TypeError, ValueError):
        return (0.0, 0.0, 0.0), False

    current_force_vector = np.asarray(current_force, dtype=float)
    values_are_finite = bool(
        np.all(np.isfinite(previous_force))
        and np.all(np.isfinite(current_force_vector))
        and np.isfinite(previous_sample_time)
        and np.isfinite(current_sample_time)
        and np.isfinite(accepted_state_time)
    )
    if not values_are_finite:
        return (0.0, 0.0, 0.0), False

    timestamp_tolerance = (
        64.0
        * np.finfo(float).eps
        * max(
            1.0,
            abs(previous_sample_time),
            abs(accepted_state_time),
        )
    )
    if previous_sample_time > accepted_state_time + timestamp_tolerance:
        return (0.0, 0.0, 0.0), False

    sample_interval = float(current_sample_time - previous_sample_time)
    if sample_interval <= 0.0 or not np.isfinite(sample_interval):
        return (0.0, 0.0, 0.0), False

    derivative = (current_force_vector - previous_force) / sample_interval
    if not np.all(np.isfinite(derivative)):
        return (0.0, 0.0, 0.0), False
    return (
        (float(derivative[0]), float(derivative[1]), float(derivative[2])),
        True,
    )


def _cap_medina_radiation_reaction_impulse(
    *,
    impulse: tuple[float, float, float],
    external_force: tuple[float, float, float],
    coordinate_dt: float,
    max_impulse_fraction: float = 0.25,
) -> tuple[tuple[float, float, float], bool]:
    """Apply the legacy Medina step guard and report whether it activated.

    The pure Medina kernel remains uncapped.  Production retains the historical
    guard at 25 percent of the non-RR external impulse, but publishes a flag so
    validation and capture studies can reject guarded steps rather than
    silently treating them as physical results.
    """

    impulse_vector = np.asarray(impulse, dtype=float)
    force_vector = np.asarray(external_force, dtype=float)
    if not (
        impulse_vector.shape == (3,)
        and force_vector.shape == (3,)
        and np.all(np.isfinite(impulse_vector))
        and np.all(np.isfinite(force_vector))
        and np.isfinite(coordinate_dt)
        and coordinate_dt > 0.0
    ):
        return (0.0, 0.0, 0.0), False

    if max_impulse_fraction <= 0.0:
        return (
            (
                float(impulse_vector[0]),
                float(impulse_vector[1]),
                float(impulse_vector[2]),
            ),
            False,
        )

    reference_impulse = float(np.linalg.norm(force_vector)) * coordinate_dt
    maximum_impulse = max_impulse_fraction * reference_impulse
    impulse_norm = float(np.linalg.norm(impulse_vector))
    if maximum_impulse > 0.0 and impulse_norm > maximum_impulse:
        impulse_vector *= maximum_impulse / impulse_norm
        return (
            (
                float(impulse_vector[0]),
                float(impulse_vector[1]),
                float(impulse_vector[2]),
            ),
            True,
        )
    return (
        (
            float(impulse_vector[0]),
            float(impulse_vector[1]),
            float(impulse_vector[2]),
        ),
        False,
    )


def _derive_relativistic_kinematics_from_force(
    external_force: tuple[float, float, float],
    beta: tuple[float, float, float],
    gamma: float,
    mass: float,
) -> tuple[tuple[float, float, float], float]:
    """Return coordinate-time ``dβ/dt`` and ``dγ/dt`` from mechanical force.

    ``external_force`` is ``dp/dt`` in native mechanical units. Inverting
    ``dp/dt = mc d(γβ)/dt`` gives
    ``dβ/dt = (F - β(β·F)) / (γmc)`` and ``dγ/dt = β·F / (mc)``.
    """
    if gamma <= 0.0 or mass <= 0.0:
        return (0.0, 0.0, 0.0), 0.0

    force_vec = np.asarray(external_force, dtype=float)
    beta_vec = np.asarray(beta, dtype=float)
    if not (np.all(np.isfinite(force_vec)) and np.all(np.isfinite(beta_vec))):
        return (0.0, 0.0, 0.0), 0.0

    beta_dot_force = float(np.dot(beta_vec, force_vec))
    beta_dot_t = (force_vec - beta_vec * beta_dot_force) / (
        float(gamma) * float(mass) * C_MMNS
    )
    dgamma_dt = beta_dot_force / (float(mass) * C_MMNS)

    return (
        (float(beta_dot_t[0]), float(beta_dot_t[1]), float(beta_dot_t[2])),
        float(dgamma_dt),
    )


def _apply_power_matched_radiation_damping(
    mechanical_momentum: tuple[float, float, float],
    mass: float,
    gamma: float,
    radiated_energy: float,
) -> tuple[tuple[float, float, float], float, float]:
    """Remove radiated energy from mechanical momentum by scaling its magnitude.

    Returns ``(new_mechanical_momentum, new_gamma, applied_energy)``. Energy is
    represented in native mechanical-energy units (``amu * mm^2 / ns^2``), while
    momentum is represented in the solver's ``amu * mm / ns`` units.
    """
    if radiated_energy <= 0.0 or mass <= 0.0 or gamma <= 1.0:
        return mechanical_momentum, gamma, 0.0

    rest_energy = mass * C_MMNS**2
    mechanical_energy = gamma * rest_energy
    available_energy = max(0.0, mechanical_energy - rest_energy)
    applied_energy = min(radiated_energy, available_energy)
    if applied_energy <= 0.0:
        return mechanical_momentum, gamma, 0.0

    new_gamma = max(1.0, (mechanical_energy - applied_energy) / rest_energy)
    old_momentum = np.asarray(mechanical_momentum, dtype=float)
    old_momentum_mag = float(np.linalg.norm(old_momentum))
    new_momentum_mag_sq = max(0.0, new_gamma**2 - 1.0) * (mass * C_MMNS) ** 2
    new_momentum_mag = float(np.sqrt(new_momentum_mag_sq))

    if old_momentum_mag > 0.0:
        new_momentum = tuple(old_momentum * (new_momentum_mag / old_momentum_mag))
    else:
        new_momentum = mechanical_momentum

    return new_momentum, new_gamma, applied_energy


def _update_beta_running_average(
    previous_avg: tuple[float, float, float],
    previous_sample_count: float,
    new_beta: tuple[float, float, float],
) -> tuple[tuple[float, float, float], float]:
    """Update running average of beta components with a new sample.

    Returns
    -------
    tuple[tuple[float, float, float], float]
        Updated (beta_avg_x, beta_avg_y, beta_avg_z) and new sample count.
    """
    new_sample_count = previous_sample_count + 1.0

    avg_x = (previous_avg[0] * previous_sample_count + new_beta[0]) / new_sample_count
    avg_y = (previous_avg[1] * previous_sample_count + new_beta[1]) / new_sample_count
    avg_z = (previous_avg[2] * previous_sample_count + new_beta[2]) / new_sample_count

    return (avg_x, avg_y, avg_z), new_sample_count


def _check_mass_shell_convergence(
    Pt: float,
    Px: float,
    Py: float,
    Pz: float,
    particle_mass: float,
    C_MMNS: float,
    tolerance: float,
    *,
    scalar_potential_contribution: float = 0.0,
    field_x: float = 0.0,
    field_y: float = 0.0,
    field_z: float = 0.0,
) -> tuple[bool, float]:
    """Check whether kinetic/mechanical momentum satisfies the mass shell."""
    kinetic_pt = np.float64(Pt) - np.float64(scalar_potential_contribution)
    mechanical_px, mechanical_py, mechanical_pz = _mechanical_momentum_components(
        px=Px,
        py=Py,
        pz=Pz,
        particle_mass=particle_mass,
        field_x=field_x,
        field_y=field_y,
        field_z=field_z,
    )
    P_spatial_sq = mechanical_px**2 + mechanical_py**2 + mechanical_pz**2
    mass_shell_rhs = (particle_mass * C_MMNS) ** 2
    mass_shell_lhs = kinetic_pt**2 - P_spatial_sq

    mass_shell_error_abs = abs(mass_shell_lhs - mass_shell_rhs)
    mass_shell_error_rel = mass_shell_error_abs / max(mass_shell_rhs, 1e-40)

    has_converged = bool(mass_shell_error_rel < tolerance)

    return has_converged, float(mass_shell_error_rel)


def _check_gamma_consistency(
    gamma_velocity: float,
    gamma_energy: float,
    tolerance: float,
) -> tuple[bool, float]:
    """Check gamma consistency (DIAGNOSTIC check after convergence).

    This verifies that gamma from velocity matches gamma from energy.
    If mass-shell is satisfied, these should match to machine precision.

    Returns
    -------
    tuple[bool, float]
        (is_consistent, relative_gamma_error)
    """
    gamma_abs_change = abs(gamma_velocity - gamma_energy)
    gamma_rel_change = gamma_abs_change / max(abs(gamma_velocity), 1e-12)
    is_consistent = gamma_rel_change < tolerance

    return is_consistent, gamma_rel_change


def _print_convergence_info(
    particle_idx: int,
    iteration: int,
    gamma_from_velocity: float,
    gamma_from_energy: float,
    gamma_mass_shell: float,
    mass_shell_error: float,
    gamma_consistency_error: float,
    converged: bool,
    max_iterations: int,
    verbosity: int = 1,
    step_idx: Optional[int] = None,
    convergence_mode: str = "fixed_geometry",
    particle_position: Optional[tuple[float, float, float]] = None,
    particle_time: Optional[float] = None,
) -> None:
    """Print debug information about self-consistency convergence.

    Shows mass-shell convergence status.

    Parameters
    ----------
    gamma_from_velocity : float
        Gamma computed from velocity: γ = 1/√(1-β²)
    gamma_from_energy : float
        Gamma computed from kinetic energy: γ = (Pt - q·Φ/c)/(mc)
    gamma_mass_shell : float
        Gamma computed from mass-shell constraint: γ = √(P²+(mc)²)/(mc)
    mass_shell_error : float
        Relative mass-shell error: |Pt² - P² - (mc)²|/(mc)²
    gamma_consistency_error : float
        Relative gamma consistency error: |γ_velocity - γ_energy| / γ
    verbosity : int
        0 = silent (no output)
        1 = summary (one line per step)
        2 = failures only (detailed only for non-converged)
        3 = full detail (all iterations)
    step_idx : Optional[int]
        Integration step number for context in error messages
    convergence_mode : str
        Convergence mode: "fixed_geometry" or "variable_geometry"
    """
    if verbosity == 0:
        return

    # Basic output (verbosity >= 1)
    if converged:
        status = f"converged in {iteration + 1} iter"
    else:
        status = f"max iter ({max_iterations}) reached"

    # Prepare step prefix if step_idx is provided
    step_prefix = f"Step {step_idx}, " if step_idx is not None else ""

    # Adjust output based on convergence mode
    if verbosity == 1:
        # Summary: one line per particle
        if convergence_mode == "fixed_geometry":
            print(
                f"    {step_prefix}P{particle_idx}: {status}, E_ms={mass_shell_error:.3e}"
            )
        else:
            print(
                f"    {step_prefix}P{particle_idx}: {status}, E_ms={mass_shell_error:.3e}, "
                f"E_gamma={gamma_consistency_error:.3e}"
            )
    elif verbosity == 2:
        # Failures only: detailed output only for non-converged steps
        if not converged:
            print(f"    {step_prefix}Particle {particle_idx}: {status}")
            print(f"      Mass-shell error = {mass_shell_error:.15e}")
            # Print position and time for failures
            if particle_position is not None and particle_time is not None:
                x, y, z = particle_position
                print(f"      Position: x={x:.6e} mm, y={y:.6e} mm, z={z:.6e} mm")
                print(f"      Time: t={particle_time:.6e} ns")
            # Only print gamma values here, not the convergence-criteria summary.
            print(f"      γ_velocity (from β)        = {gamma_from_velocity:.15e}")
            print(f"      γ_energy   (from Pt - q·Φ/c) = {gamma_from_energy:.15e}")
            print(f"      γ_mass_shell (√(P²+(mc)²)/(mc)) = {gamma_mass_shell:.15e}")
        else:
            # For converged steps at verbosity 2, just show summary
            if convergence_mode == "fixed_geometry":
                print(
                    f"    {step_prefix}P{particle_idx}: {status}, E_ms={mass_shell_error:.3e}"
                )
            else:
                print(
                    f"    {step_prefix}P{particle_idx}: {status}, E_ms={mass_shell_error:.3e}, "
                    f"E_gamma={gamma_consistency_error:.3e}"
                )
    else:  # verbosity >= 3
        # Full detail: multi-line output with full precision for all steps
        print(f"    {step_prefix}Particle {particle_idx}: {status}")
        print(f"      Mass-shell error = {mass_shell_error:.15e}")
        # Print position and time for verbosity 3 when showing failures
        if (
            not converged
            and particle_position is not None
            and particle_time is not None
        ):
            x, y, z = particle_position
            print(f"      Position: x={x:.6e} mm, y={y:.6e} mm, z={z:.6e} mm")
            print(f"      Time: t={particle_time:.6e} ns")
        # Only print gamma values here, not the convergence-criteria summary.
        print(f"      γ_velocity (from β)        = {gamma_from_velocity:.15e}")
        print(f"      γ_energy   (from Pt - q·Φ/c) = {gamma_from_energy:.15e}")
        print(f"      γ_mass_shell (√(P²+(mc)²)/(mc)) = {gamma_mass_shell:.15e}")


def retarded_equations_of_motion(
    h: float,
    trajectory: Trajectory,
    trajectory_ext: Trajectory,
    index_traj: int,
    aperture_radius: float,
    sim_type: SimulationType,
    chrono_mode: ChronoMatchingMode = ChronoMatchingMode.AVERAGED,
    startup_mode: StartupMode = StartupMode.COLD_START,
    self_consistency: Optional[SelfConsistencyConfig] = None,
    step_idx: Optional[int] = None,
    cancel_callback: Optional[Any] = None,
    space_charge: Optional[Any] = None,
    traj_soa: Optional[TrajectoryArrays] = None,
    traj_ext_soa: Optional[TrajectoryArrays] = None,
    radiation_reaction_mode: Optional[str] = "off",
    external_field: Optional[Any] = None,
    pseudo_grid_space_charge_source_charges: Optional[np.ndarray] = None,
    pseudo_grid_space_charge_source_trajectory: Optional[Trajectory] = None,
    pseudo_grid_space_charge_source_soa: Optional[TrajectoryArrays] = None,
    pseudo_grid_space_charge_source_radii_mm: Optional[np.ndarray] = None,
    macroparticle_smearing: Optional[MacroparticleSmearingConfig] = None,
    beamline_geometry: Optional[BeamlineGeometryConfig] = None,
    magnetic_dipole: Optional[MagneticDipoleConfig] = None,
    exact_source_history: Optional[Any] = None,
    exact_dipole_source_collection: Optional[Any] = None,
    exact_source_spin_interpolation_model: str = "centered_c1",
) -> ParticleState:
    """Core equations of motion preserving the validated reference behavior.

    Parameters
    ----------
    h:
        Time step between trajectory samples.
    trajectory:
        Mutable view over the rider bunch history.
    trajectory_ext:
        History of the external bunch (driver, image or opposing bunch).
    index_traj:
        Index of the current time step within ``trajectory``.
    aperture_radius:
        Aperture radius supplied to the image generators.
    sim_type:
        Simulation boundary type encoded as :class:`SimulationType`.
    chrono_mode:
        Retardation sampling strategy; ``FAST`` retains the validated single
        sample, whereas ``AVERAGED`` blends ``R / c`` and ``2R / c`` emission
        times for the external bunch.
    startup_mode:
        Early-step handling strategy; ``COLD_START`` suppresses external forces
        until sufficient observer travel has occurred, while
        ``APPROXIMATE_BACK_HISTORY`` assumes constant source velocity to
        reconstruct an analytic history.
    exact_source_history:
        Optional immutable source-history view used by exact charge and dipole
        providers without replacing the accepted chronology/gating history.
    exact_dipole_source_collection:
        Optional ordered causal-$C^5$ dipole history.  Charge providers continue
        to use ``exact_source_history``; only the intrinsic-dipole Maxwell source
        is replaced.  This separation prevents a spin-history experiment from
        changing the already validated charge chronology.
    exact_source_spin_interpolation_model:
        Spin interpolation contract for ``exact_source_history``. Trial overlays
        require ``"causal_frozen_c1"`` so accepted spin segments stay fixed.
    self_consistency:
        Optional configuration for self-consistency iterations. If provided and
        enabled, each particle's update will iterate until gamma converges,
        solving the circular dependency between forces and gamma.
    step_idx:
        Optional integration step number for context in error messages.
    cancel_callback:
        Optional predicate to check for cancellation. If provided and returns True,
        raises IntegrationCancelled to abort the integration.

    Returns
    -------
    ParticleState
        Updated particle state for the next time step.
    """
    # Ensure metadata for startup mode is initialized
    _ensure_startup_metadata(trajectory[index_traj])

    # Initialize result state as a copy of current state
    current_state = trajectory[index_traj]
    result = _initialize_result_state(current_state)
    radiation_mode = _canonicalize_radiation_reaction_mode(radiation_reaction_mode)
    if radiation_mode == "medina_lad":
        _initialize_medina_step_state(result)

    num_particles = len(current_state["x"])
    exact_endpoint_recomposition_selected = bool(
        magnetic_dipole is not None
        and magnetic_dipole.enabled
        and magnetic_dipole.spin_model == "rfs_minimal_2021"
        and startup_mode is StartupMode.INERTIAL_PREHISTORY
        and sim_type == SimulationType.BUNCH_TO_BUNCH
    )
    second_order_exact_source_selected = bool(
        exact_endpoint_recomposition_selected
        and magnetic_dipole is not None
        and magnetic_dipole.exact_retarded_update
        == "second_order_start_taylor_endpoint"
    )
    intrinsic_spin_diagnostic_selected = bool(
        second_order_exact_source_selected
        and magnetic_dipole is not None
        and magnetic_dipole.intrinsic_spin_self_reaction_mode == "diagnostic"
    )
    if exact_endpoint_recomposition_selected:
        # Private, step-local handoff to the pair-level accepted-endpoint
        # finalizer in integration_runner.  It is intentionally absent from
        # TrajectoryArrays and public output.
        result["_exact_source_start_four_potential"] = np.zeros(
            (num_particles, 4), dtype=float
        )
        result["_exact_source_endpoint_rebase_required"] = np.zeros(
            num_particles, dtype=bool
        )
    if exact_dipole_source_collection is not None and not (
        exact_endpoint_recomposition_selected
        and magnetic_dipole is not None
        and magnetic_dipole.source.active
    ):
        raise ValueError(
            "causal C5 dipole history requires the exact inertial dipole-source path"
        )
    if second_order_exact_source_selected:
        # Private accepted-trial metadata for the diagnostic intrinsic-spin
        # history.  These values belong to the step start and are calculated
        # before Medina adds its charge-radiation impulse.  They are consumed
        # transactionally by the exact-pair adaptive controller and are never
        # materialized in TrajectoryArrays or public output.
        result["_intrinsic_spin_start_four_velocity"] = np.full(
            (num_particles, 4), np.nan, dtype=float
        )
        result["_intrinsic_spin_start_non_self_four_acceleration"] = np.full(
            (num_particles, 4), np.nan, dtype=float
        )
        result["_intrinsic_spin_start_physical_four_spin"] = np.full(
            (num_particles, 4), np.nan, dtype=float
        )
        if intrinsic_spin_diagnostic_selected:
            result["_intrinsic_spin_start_analytical_reduction"] = [
                None
            ] * num_particles
            result["_intrinsic_spin_start_analytical_unavailable_reason"] = [
                "not evaluated"
            ] * num_particles
            result["_intrinsic_spin_charge_native"] = np.full(
                num_particles, np.nan, dtype=float
            )
            result["_intrinsic_spin_mass_amu"] = np.full(
                num_particles, np.nan, dtype=float
            )
            result["_intrinsic_spin_g_factor"] = np.full(
                num_particles, np.nan, dtype=float
            )
    pseudo_grid_sc_charge_matrix = None
    pseudo_grid_sc_source_radii = None
    if pseudo_grid_space_charge_source_radii_mm is not None:
        pseudo_grid_sc_source_radii = np.asarray(
            pseudo_grid_space_charge_source_radii_mm,
            dtype=float,
        )
        if pseudo_grid_sc_source_radii.ndim != 1:
            raise ValueError(
                "pseudo_grid_space_charge_source_radii_mm must be a 1-D vector"
            )
        if np.any(pseudo_grid_sc_source_radii < 0.0):
            raise ValueError(
                "pseudo_grid_space_charge_source_radii_mm must be non-negative"
            )
    if pseudo_grid_space_charge_source_charges is not None:
        pseudo_grid_sc_charge_matrix = np.asarray(
            pseudo_grid_space_charge_source_charges,
            dtype=float,
        )
        if pseudo_grid_sc_charge_matrix.ndim != 2:
            raise ValueError(
                "pseudo_grid_space_charge_source_charges must be a 2-D matrix"
            )
        if pseudo_grid_sc_charge_matrix.shape[0] != num_particles:
            raise ValueError(
                "pseudo_grid_space_charge_source_charges must have one row per "
                f"observer particle ({num_particles})"
            )

    # Track particles marked dead in this step
    particles_marked_dead_this_step = 0

    # Extract self-consistency configuration
    (
        sc_enabled,
        sc_convergence_mode,
        sc_target_ms_tolerance,
        sc_mass_shell_tolerance,
        sc_mass_shell_relaxation,
        sc_max_iterations,
        sc_verbosity,
    ) = _extract_self_consistency_params(self_consistency)

    # Extract chrono-match parameters (needed for interpolation later)
    chrono_interpolate = False
    chrono_tolerance = 1e-3
    chrono_high_precision = False
    chrono_adaptive_tolerance = False
    chrono_verbosity = 0
    if self_consistency is not None:
        chrono_interpolate = getattr(self_consistency, "chrono_interpolate", False)
        chrono_tolerance = getattr(self_consistency, "chrono_tolerance", 1e-3)
        chrono_high_precision = getattr(
            self_consistency, "chrono_high_precision", False
        )
        chrono_adaptive_tolerance = getattr(
            self_consistency, "chrono_adaptive_tolerance", False
        )
        chrono_verbosity = getattr(self_consistency, "verbosity", 0)

    # Import IntegrationCancelled at top of function for use in cancel checks
    from .integration_runner import IntegrationCancelled

    # Process each particle independently
    for particle_idx in range(num_particles):
        # Check for cancellation before processing each particle
        if cancel_callback is not None and cancel_callback():
            raise IntegrationCancelled("Integration cancelled by caller.")

        # Skip particles that are already marked dead
        if "_dead_particles" in result and result["_dead_particles"][particle_idx]:
            # Copy previous state for dead particle (don't recompute)
            for key in [
                "x",
                "y",
                "z",
                "t",
                "bx",
                "by",
                "bz",
                "gamma",
                "Px",
                "Py",
                "Pz",
                "Pt",
                "bdotx",
                "bdoty",
                "bdotz",
                "spin_x",
                "spin_y",
                "spin_z",
                "local_magnetic_field_x_t",
                "local_magnetic_field_y_t",
                "local_magnetic_field_z_t",
            ]:
                if key in current_state:
                    result[key][particle_idx] = current_state[key][particle_idx]
            continue

        # Working state for SC iterations - tracks evolving state
        # On iteration 0, use current_state values
        # On iteration k > 0, use values from previous iteration
        working_beta_x = current_state["bx"][particle_idx]
        working_beta_y = current_state["by"][particle_idx]
        working_beta_z = current_state["bz"][particle_idx]
        working_gamma = current_state["gamma"][particle_idx]
        working_x = current_state["x"][particle_idx]
        working_y = current_state["y"][particle_idx]
        working_z = current_state["z"][particle_idx]

        force_particle_charge: float = float(
            _get_particle_observer_charge(current_state, particle_idx)
        )
        particle_mass: float = float(_get_particle_mass(current_state, particle_idx))
        accumulated_field_x: float = 0.0
        accumulated_field_y: float = 0.0
        accumulated_field_z: float = 0.0
        accumulated_scalar_potential: float = 0.0
        local_electric_field_v_m = np.zeros(3, dtype=float)
        local_magnetic_field_t = np.zeros(3, dtype=float)
        local_magnetic_gradient_t_per_m = np.zeros((3, 3), dtype=float)
        local_electric_field_native = np.zeros(3, dtype=float)
        local_magnetic_field_native = np.zeros(3, dtype=float)
        local_magnetic_gradient_native_per_mm = np.zeros((3, 3), dtype=float)
        rfs_field_tensor = np.zeros((4, 4), dtype=float)
        rfs_partial_f = np.zeros((4, 4, 4), dtype=float)
        stern_gerlach_impulse_applied = False
        rfs_selected = bool(
            magnetic_dipole is not None
            and magnetic_dipole.enabled
            and magnetic_dipole.spin_model == "rfs_minimal_2021"
        )
        rfs_force_selected = bool(
            rfs_selected
            and magnetic_dipole is not None
            and magnetic_dipole.stern_gerlach_model == "rfs_full_g"
        )
        dipole_source_selected = bool(
            magnetic_dipole is not None
            and magnetic_dipole.enabled
            and magnetic_dipole.source.active
        )
        exact_charge_source_selected = exact_endpoint_recomposition_selected
        on_shell_kinematic_boundary_selected = bool(
            exact_charge_source_selected or radiation_mode == "medina_lad"
        )
        exact_charge_field_cache = None
        dipole_source_field_cache = None
        exact_charge_source_interaction = None
        exact_charge_analytic_response = None
        exact_dipole_analytic_response = None
        exact_analytic_antisymmetric_response = None
        exact_analytic_partial_antisymmetric_response = None
        dipole_source_interaction = None
        exact_source_start_four_potential = np.zeros(4, dtype=float)
        # This is the Medina three-force actually applied during the final
        # nonlinear trial.  It stays exactly zero for off, neutral, and
        # derivative-unprimed steps so their RFS spin update is unchanged.
        applied_medina_force_native = np.zeros(3, dtype=float)

        # Self-consistency loop: iterate until gamma converges
        converged = False
        last_mass_shell_error = float("inf")
        for sc_iteration in range(sc_max_iterations):
            # Only the final self-consistency trial feeds the once-per-step
            # spin update.  Resetting here prevents an earlier trial's Medina
            # force from leaking into it.
            applied_medina_force_native.fill(0.0)
            # Check for cancellation during self-consistency iterations
            if cancel_callback is not None and cancel_callback():
                raise IntegrationCancelled("Integration cancelled by caller.")

            if sc_verbosity >= 3 and sc_iteration > 0:
                print(
                    f"    Particle {particle_idx} iteration {sc_iteration}: "
                    f"Starting refinement"
                )
                print(
                    f"      Working state: βx={working_beta_x:.15e}, "
                    f"βy={working_beta_y:.15e}, βz={working_beta_z:.15e}, "
                    f"γ={working_gamma:.15e}"
                )
                # Also print what result[bx/bz] contains to verify it's from previous iteration
                print(
                    f"      result[bx]={result['bx'][particle_idx]:.15e}, "
                    f"result[bz]={result['bz'][particle_idx]:.15e}"
                )

            # ================================================================
            # STEP 1: Determine observer state for retarded distance calculation
            # ================================================================
            # In variable_geometry mode, use position from previous iteration
            # In fixed_geometry mode, use initial position for all iterations
            if sc_convergence_mode == "variable_geometry" and sc_iteration > 0:
                # Create temporary state with updated position for retarded distance calc
                observer_state = {
                    "x": np.array([working_x]),
                    "y": np.array([working_y]),
                    "z": np.array([working_z]),
                    "t": np.array([current_state["t"][particle_idx]]),
                    "bx": np.array([working_beta_x]),
                    "by": np.array([working_beta_y]),
                    "bz": np.array([working_beta_z]),
                    "gamma": np.array([working_gamma]),
                    "origin_x": current_state["origin_x"],
                    "origin_y": current_state["origin_y"],
                    "origin_z": current_state["origin_z"],
                    "beta_avg_x": current_state["beta_avg_x"],
                    "beta_avg_y": current_state["beta_avg_y"],
                    "beta_avg_z": current_state["beta_avg_z"],
                }
                observer_particle_idx = 0  # Using single-element arrays

                if sc_verbosity >= 3:
                    print(
                        f"      Full iteration: Using updated position "
                        f"x={working_x:.6e}, y={working_y:.6e}, z={working_z:.6e}"
                    )
            else:
                # Use current_state position (start of timestep)
                observer_state = current_state
                observer_particle_idx = particle_idx

            # ================================================================
            # STEP 2: Early check for COLD_START gating
            # ================================================================
            # For COLD_START, check if we should skip force computation entirely
            # This avoids expensive retarded distance calculations during startup phase
            skip_external_forces = False
            if startup_mode is StartupMode.COLD_START:
                # Check if particle has traveled far enough from origin
                # This is the same check done in _should_apply_external_forces
                # but done here to avoid computing retarded distances needlessly
                origin_position = (
                    current_state["origin_x"][particle_idx],
                    current_state["origin_y"][particle_idx],
                    current_state["origin_z"][particle_idx],
                )
                current_position = (
                    current_state["x"][particle_idx],
                    current_state["y"][particle_idx],
                    current_state["z"][particle_idx],
                )
                travel_distance = _calculate_travel_distance(
                    origin_position, current_position
                )

                # Estimate threshold without computing full retarded distances
                # Use maximum possible R from trajectory_ext bounds
                beta_avg_x = current_state["beta_avg_x"][particle_idx]
                beta_avg_y = current_state["beta_avg_y"][particle_idx]
                beta_avg_z = current_state["beta_avg_z"][particle_idx]
                beta_avg_mag = np.sqrt(beta_avg_x**2 + beta_avg_y**2 + beta_avg_z**2)

                # Estimate max R from external trajectory bounds.
                # In pseudo-grid reduced solves the source history may be
                # causally pruned, so its local current index need not match
                # the observer-history index.
                external_current_step_idx = (
                    min(index_traj, len(trajectory_ext) - 1) if trajectory_ext else None
                )
                external_current_state = (
                    trajectory_ext[external_current_step_idx]
                    if external_current_step_idx is not None
                    else None
                )
                if (
                    external_current_state is not None
                    and external_current_state["x"].size > 0
                ):
                    ext_x = external_current_state["x"]
                    ext_y = external_current_state["y"]
                    ext_z = external_current_state["z"]
                    dx = current_position[0] - ext_x
                    dy = current_position[1] - ext_y
                    dz = current_position[2] - ext_z
                    distances = np.sqrt(dx**2 + dy**2 + dz**2)
                    estimated_max_R = (
                        float(np.max(distances)) if distances.size > 0 else 1000.0
                    )
                else:
                    # Fallback if no external particles
                    estimated_max_R = 1000.0

                # Correct formula: threshold = β·R / (1 - β·n̂)
                # For early check, use conservative estimate assuming worst case
                # (minimum threshold = particle approaching head-on)
                # For β·n̂ = -1 (approaching): threshold = β·R/2
                # For β·n̂ = 0 (perpendicular): threshold = β·R
                # For β·n̂ = +1 (receding): threshold → ∞
                # Use worst case (approaching) for conservative early gating
                # Worst case: β·n̂ = -beta_avg_mag → denominator = 1 + beta_avg_mag
                estimated_threshold = (
                    beta_avg_mag * estimated_max_R / (1.0 + beta_avg_mag)
                )

                # Skip if travel distance is definitely below threshold
                if travel_distance < estimated_threshold:
                    skip_external_forces = True

            # ================================================================
            # STEP 3: Compute retarded distances to external sources
            # ================================================================
            # Only compute if forces will actually be applied
            chrono_result: Optional[ChronoMatchResult] = None
            nhat = None
            indices_bounded = None

            if not skip_external_forces and not exact_charge_source_selected:
                if startup_mode is StartupMode.APPROXIMATE_BACK_HISTORY:
                    external_current_step_idx = min(
                        index_traj,
                        len(trajectory_ext) - 1,
                    )
                    nhat, indices_bounded = _compute_approximate_retarded_distance(
                        observer_state,
                        trajectory_ext[external_current_step_idx],
                        observer_particle_idx,
                        external_current_step_idx,
                    )
                else:
                    # For variable geometry modes, need to create trajectory with observer_state
                    if sc_convergence_mode == "variable_geometry" and sc_iteration > 0:
                        # Create temporary trajectory for retarded distance calculation
                        temp_trajectory = trajectory.copy()
                        temp_trajectory[index_traj] = observer_state
                        nhat, indices_bounded, chrono_result = (
                            _compute_full_retarded_distance(
                                temp_trajectory,
                                trajectory_ext,
                                index_traj,
                                observer_particle_idx,
                                chrono_mode,
                                self_consistency,
                                timestep_h=h,
                                **(
                                    {"traj_soa": traj_soa}
                                    if traj_soa is not None
                                    else {}
                                ),
                                **(
                                    {"traj_ext_soa": traj_ext_soa}
                                    if traj_ext_soa is not None
                                    else {}
                                ),
                            )
                        )
                    else:
                        nhat, indices_bounded, chrono_result = (
                            _compute_full_retarded_distance(
                                trajectory,
                                trajectory_ext,
                                index_traj,
                                particle_idx,
                                chrono_mode,
                                self_consistency,
                                timestep_h=h,
                                **(
                                    {"traj_soa": traj_soa}
                                    if traj_soa is not None
                                    else {}
                                ),
                                **(
                                    {"traj_ext_soa": traj_ext_soa}
                                    if traj_ext_soa is not None
                                    else {}
                                ),
                            )
                        )

            # Initialize position and time from current_state
            # These will be updated after force calculation
            result["x"][particle_idx] = current_state["x"][particle_idx]
            result["y"][particle_idx] = current_state["y"][particle_idx]
            result["z"][particle_idx] = current_state["z"][particle_idx]
            result["t"][particle_idx] = current_state["t"][particle_idx]

            # Start accumulation from initial momentum
            # Always start from current_state - forces will be recomputed using updated beta
            accumulated_momentum_x = current_state["Px"][particle_idx]
            accumulated_momentum_y = current_state["Py"][particle_idx]
            accumulated_momentum_z = current_state["Pz"][particle_idx]
            accumulated_momentum_t = current_state["Pt"][particle_idx]
            exact_mechanical_temporal_impulse = 0.0

            # Accumulated field contributions (used in position update)
            accumulated_field_x = 0.0
            accumulated_field_y = 0.0
            accumulated_field_z = 0.0

            # Accumulated scalar potential (used in gamma calculation)
            accumulated_scalar_potential = 0.0
            # Each nonlinear iteration recomputes the physical step from the
            # accepted start state. Only the final iteration may trigger the
            # dipole-specific mass-shell projection below.
            stern_gerlach_impulse_applied = False
            rfs_field_tensor = np.zeros((4, 4), dtype=float)
            rfs_partial_f = np.zeros((4, 4, 4), dtype=float)
            exact_charge_source_interaction = None
            exact_charge_analytic_response = None
            exact_dipole_analytic_response = None
            exact_analytic_antisymmetric_response = None
            exact_analytic_partial_antisymmetric_response = None
            dipole_source_interaction = None
            rfs_dipole_force_native = np.zeros(4, dtype=float)
            additional_start_force_native = np.zeros(4, dtype=float)
            exact_ordinary_response_beta = (
                np.asarray(
                    (
                        current_state["bx"][particle_idx],
                        current_state["by"][particle_idx],
                        current_state["bz"][particle_idx],
                    ),
                    dtype=float,
                )
                if second_order_exact_source_selected
                else np.asarray(
                    (working_beta_x, working_beta_y, working_beta_z),
                    dtype=float,
                )
            )
            exact_source_start_four_potential.fill(0.0)

            # ================================================================
            # STEP 4: Determine if external forces should be applied
            # ================================================================
            # If we already determined to skip (COLD_START early exit), use that
            # Otherwise, do the full check with computed nhat values
            if exact_charge_source_selected:
                apply_forces = True
            elif skip_external_forces:
                apply_forces = False
            elif nhat is not None:
                # Do full gating check with actual retarded distances
                apply_forces = _should_apply_external_forces(
                    startup_mode, sim_type, nhat, current_state, particle_idx
                )
            else:
                # No nhat computed, no forces to apply
                apply_forces = False

            # Use working state values for force calculations
            # These evolve across SC iterations
            particle_gamma = working_gamma
            particle_beta = (working_beta_x, working_beta_y, working_beta_z)

            # ================================================================
            # STEP 4: Compute and accumulate external force contributions
            # ================================================================
            if (
                apply_forces
                and not exact_charge_source_selected
                and nhat is not None
                and nhat["R"].size > 0
            ):
                # Gather external particle data at retarded times (with interpolation if enabled)
                _external_include_positions = bool(
                    (
                        macroparticle_smearing
                        and macroparticle_smearing.enabled
                        and (
                            macroparticle_smearing.apply_to_active_sources
                            or macroparticle_smearing.apply_to_passive_sources
                        )
                    )
                    or (beamline_geometry is not None and beamline_geometry.enabled)
                )
                if traj_ext_soa is not None and chrono_result is not None:
                    external_samples = gather_external_samples_soa(
                        traj_ext_soa,
                        indices_bounded,
                        indices_next=chrono_result.indices_next,
                        weights=chrono_result.weights,
                        needs_interpolation=chrono_result.needs_interpolation,
                        include_positions=_external_include_positions,
                    )
                elif chrono_result is not None:
                    # Use interpolation (with cubic and position interpolation if high-precision)
                    external_samples = gather_external_samples(
                        trajectory_ext,
                        indices_bounded,
                        indices_next=chrono_result.indices_next,
                        weights=chrono_result.weights,
                        indices_prev=chrono_result.indices_prev,
                        indices_next2=chrono_result.indices_next2,
                        use_cubic=chrono_result.use_cubic,
                        interpolate_positions=chrono_high_precision,
                        include_positions=_external_include_positions,
                    )
                elif traj_ext_soa is not None:
                    external_samples = gather_external_samples_soa(
                        traj_ext_soa,
                        indices_bounded,
                        include_positions=_external_include_positions,
                    )
                else:
                    # Legacy path: no interpolation
                    external_samples = gather_external_samples(
                        trajectory_ext,
                        indices_bounded,
                        include_positions=_external_include_positions,
                    )

                external_samples, smeared_nhat = smear_source_samples(
                    samples=external_samples,
                    observer_position=(
                        float(working_x),
                        float(working_y),
                        float(working_z),
                    ),
                    config=macroparticle_smearing,
                    step_index=index_traj,
                )
                if smeared_nhat:
                    nhat = smeared_nhat

                if (
                    beamline_geometry is not None
                    and beamline_geometry.enabled
                    and external_samples.x is not None
                ):
                    src_positions = np.stack(
                        [
                            np.asarray(external_samples.x, dtype=float),
                            np.asarray(external_samples.y, dtype=float),
                            np.asarray(external_samples.z, dtype=float),
                        ],
                        axis=-1,
                    )
                    visibility = compute_directional_visibility_mask(
                        src_positions,
                        beamline_geometry,
                        observer_direction=(
                            float(working_beta_x),
                            float(working_beta_y),
                            float(working_beta_z),
                        ),
                    )
                    external_samples.valid_mask = (
                        external_samples.valid_mask & visibility
                    )

                # Compute electromagnetic force contributions
                (
                    delta_momentum_x,
                    delta_momentum_y,
                    delta_momentum_z,
                    delta_momentum_t,
                    delta_field_x,
                    delta_field_y,
                    delta_field_z,
                    delta_scalar_potential,
                ) = compute_vectorized_contributions(
                    h=h,
                    charge_i=float(force_particle_charge),
                    mass_i=float(particle_mass),
                    gamma_i=particle_gamma,
                    beta_vec=particle_beta,
                    nhat_nx=np.asarray(nhat["nx"], dtype=float),
                    nhat_ny=np.asarray(nhat["ny"], dtype=float),
                    nhat_nz=np.asarray(nhat["nz"], dtype=float),
                    R_separation=np.asarray(nhat["R"], dtype=float),
                    samples=external_samples,
                    apply_external=apply_forces,
                    verbosity=sc_verbosity,
                )

                # Debug: Log what forces were computed
                if sc_verbosity >= 3 and sc_enabled:
                    print(
                        f"      Force contributions: ΔPx={delta_momentum_x:.15e}, "
                        f"ΔPy={delta_momentum_y:.15e}, ΔPz={delta_momentum_z:.15e}, "
                        f"ΔPt={delta_momentum_t:.15e}"
                    )
                    print(
                        f"      Using particle_beta=({particle_beta[0]:.15e}, "
                        f"{particle_beta[1]:.15e}, {particle_beta[2]:.15e}), "
                        f"gamma={particle_gamma:.15e}"
                    )

                # Accumulate momentum changes
                accumulated_momentum_x += delta_momentum_x
                accumulated_momentum_y += delta_momentum_y
                accumulated_momentum_z += delta_momentum_z
                accumulated_momentum_t += delta_momentum_t

                # Accumulate field contributions
                accumulated_field_x += delta_field_x
                accumulated_field_y += delta_field_y
                accumulated_field_z += delta_field_z

                # Accumulate scalar potential
                accumulated_scalar_potential += delta_scalar_potential

                if sc_verbosity >= 3 and sc_enabled and sc_iteration > 0:
                    print(
                        f"      After forces: ΔPt={delta_momentum_t:.15e}, "
                        f"accumulated_pt={accumulated_momentum_t:.15e}"
                    )

            # ================================================================
            # STEP 4b: Intra-bunch space-charge forces (rider-rider, j ≠ i)
            # ================================================================
            if (
                space_charge is not None
                and space_charge.enabled
                and len(trajectory) >= 1
            ):
                n_particles = current_state["x"].shape[0]
                sc_source_trajectory = (
                    pseudo_grid_space_charge_source_trajectory
                    if pseudo_grid_space_charge_source_trajectory is not None
                    else trajectory
                )
                sc_source_soa = (
                    pseudo_grid_space_charge_source_soa
                    if pseudo_grid_space_charge_source_soa is not None
                    else traj_soa
                )
                sc_source_count = (
                    sc_source_soa.n_particles
                    if sc_source_soa is not None
                    else len(sc_source_trajectory[-1]["x"])
                )
                if n_particles > 1 and sc_source_count > 0:
                    sc_softening = float(space_charge.softening_mm)
                    observer_sc_charge_row = None
                    if pseudo_grid_sc_charge_matrix is not None:
                        observer_sc_charge_row = pseudo_grid_sc_charge_matrix[
                            particle_idx
                        ]
                    if (
                        pseudo_grid_sc_charge_matrix is not None
                        and pseudo_grid_sc_charge_matrix.shape[1] != sc_source_count
                    ):
                        raise ValueError(
                            "pseudo-grid space-charge source matrix column count "
                            f"({pseudo_grid_sc_charge_matrix.shape[1]}) must match "
                            f"the source particle count ({sc_source_count})"
                        )
                    if (
                        pseudo_grid_sc_source_radii is not None
                        and pseudo_grid_sc_source_radii.shape != (sc_source_count,)
                    ):
                        raise ValueError(
                            "pseudo-grid space-charge source radii must have shape "
                            f"({sc_source_count},)"
                        )
                    # Use retarded SC only once sufficient history has accumulated
                    # (at least one light-crossing time of the bunch width).
                    # resolve_min_retarded_steps returns the step threshold; below
                    # it we use instantaneous Coulomb as a physically motivated
                    # startup approximation.
                    _sc_threshold = space_charge.resolve_min_retarded_steps(h)
                    use_retarded_sc = len(trajectory) > _sc_threshold

                    sc_chrono_result = None
                    use_sc_soa = traj_soa is not None and sc_source_soa is not None
                    if use_retarded_sc:
                        if use_sc_soa:
                            sc_retarded_result = chrono_match_indices_soa(
                                traj_soa,
                                sc_source_soa,
                                index_traj,
                                particle_idx,
                                mode=ChronoMatchingMode.FAST,
                                interpolate=chrono_interpolate,
                                tolerance=chrono_tolerance,
                                verbosity=chrono_verbosity,
                                high_precision=chrono_high_precision,
                                adaptive_tolerance=chrono_adaptive_tolerance,
                                timestep_h=h,
                            )
                        else:
                            sc_retarded_result = chrono_match_indices(
                                trajectory,
                                sc_source_trajectory,
                                index_traj,
                                particle_idx,
                                mode=ChronoMatchingMode.FAST,
                                interpolate=chrono_interpolate,
                                tolerance=chrono_tolerance,
                                verbosity=chrono_verbosity,
                                high_precision=chrono_high_precision,
                                adaptive_tolerance=chrono_adaptive_tolerance,
                                timestep_h=h,
                            )
                        if isinstance(sc_retarded_result, ChronoMatchResult):
                            sc_indices = sc_retarded_result.indices
                            sc_chrono_result = sc_retarded_result
                        else:
                            sc_indices = sc_retarded_result
                    else:
                        current_sc_index = min(
                            index_traj, len(sc_source_trajectory) - 1
                        )
                        sc_indices = np.full(
                            sc_source_count, current_sc_index, dtype=int
                        )

                    sc_indices = np.minimum(
                        np.maximum(sc_indices, 0), len(sc_source_trajectory) - 1
                    )
                    if use_sc_soa:
                        sc_nhat = compute_retarded_distance_soa(
                            traj_soa,
                            sc_source_soa,
                            index_traj,
                            particle_idx,
                            sc_indices,
                        )
                    else:
                        sc_nhat = compute_retarded_distance(
                            trajectory,
                            sc_source_trajectory,
                            index_traj,
                            particle_idx,
                            sc_indices,
                        )
                    sc_R = np.asarray(sc_nhat["R"], dtype=float)
                    source_radius = (
                        pseudo_grid_sc_source_radii
                        if pseudo_grid_sc_source_radii is not None
                        else 0.0
                    )
                    if sc_softening > 0.0 or pseudo_grid_sc_source_radii is not None:
                        sc_R = np.sqrt(sc_R**2 + sc_softening**2 + source_radius**2)
                        sc_nhat = dict(sc_nhat)
                        sc_nhat["R"] = sc_R
                    if sc_chrono_result is not None:
                        if use_sc_soa:
                            sc_samples = gather_external_samples_soa(
                                sc_source_soa,
                                sc_indices,
                                indices_next=sc_chrono_result.indices_next,
                                weights=sc_chrono_result.weights,
                                needs_interpolation=sc_chrono_result.needs_interpolation,
                                include_positions=bool(
                                    macroparticle_smearing
                                    and macroparticle_smearing.enabled
                                    and (
                                        macroparticle_smearing.apply_to_active_sources
                                        or macroparticle_smearing.apply_to_passive_sources
                                    )
                                ),
                            )
                        else:
                            sc_samples = gather_external_samples(
                                sc_source_trajectory,
                                sc_indices,
                                indices_next=sc_chrono_result.indices_next,
                                weights=sc_chrono_result.weights,
                                indices_prev=sc_chrono_result.indices_prev,
                                indices_next2=sc_chrono_result.indices_next2,
                                use_cubic=sc_chrono_result.use_cubic,
                                interpolate_positions=chrono_high_precision,
                                include_positions=bool(
                                    macroparticle_smearing
                                    and macroparticle_smearing.enabled
                                    and (
                                        macroparticle_smearing.apply_to_active_sources
                                        or macroparticle_smearing.apply_to_passive_sources
                                    )
                                ),
                            )
                    else:
                        if use_sc_soa:
                            sc_samples = gather_external_samples_soa(
                                sc_source_soa,
                                sc_indices,
                                include_positions=bool(
                                    macroparticle_smearing
                                    and macroparticle_smearing.enabled
                                    and (
                                        macroparticle_smearing.apply_to_active_sources
                                        or macroparticle_smearing.apply_to_passive_sources
                                    )
                                ),
                            )
                        else:
                            sc_samples = gather_external_samples(
                                sc_source_trajectory,
                                sc_indices,
                                include_positions=bool(
                                    macroparticle_smearing
                                    and macroparticle_smearing.enabled
                                    and (
                                        macroparticle_smearing.apply_to_active_sources
                                        or macroparticle_smearing.apply_to_passive_sources
                                    )
                                ),
                            )

                    if not use_retarded_sc:
                        sc_samples.bx = np.zeros_like(sc_samples.bx)
                        sc_samples.by = np.zeros_like(sc_samples.by)
                        sc_samples.bz = np.zeros_like(sc_samples.bz)
                        sc_samples.bdotx = np.zeros_like(sc_samples.bdotx)
                        sc_samples.bdoty = np.zeros_like(sc_samples.bdoty)
                        sc_samples.bdotz = np.zeros_like(sc_samples.bdotz)
                        sc_samples.gamma = np.ones_like(sc_samples.gamma)

                    sc_source_charges = sc_samples.charge.copy()
                    if observer_sc_charge_row is not None:
                        sc_source_charges = np.asarray(
                            observer_sc_charge_row,
                            dtype=float,
                        ).copy()
                        if sc_source_charges.shape != (sc_source_count,):
                            raise ValueError(
                                "pseudo-grid space-charge row must have shape "
                                f"({sc_source_count},)"
                            )
                    if pseudo_grid_sc_charge_matrix is None:
                        sc_source_charges[particle_idx] = 0.0
                    sc_samples.charge[...] = sc_source_charges
                    sc_samples.valid_mask = sc_samples.valid_mask.copy()
                    if pseudo_grid_sc_charge_matrix is None:
                        sc_samples.valid_mask[particle_idx] = False

                    sc_samples, smeared_sc_nhat = smear_source_samples(
                        samples=sc_samples,
                        observer_position=(
                            float(working_x),
                            float(working_y),
                            float(working_z),
                        ),
                        config=macroparticle_smearing,
                        step_index=index_traj,
                    )
                    if smeared_sc_nhat:
                        sc_nhat = smeared_sc_nhat
                        sc_R = np.asarray(sc_nhat["R"], dtype=float)
                        if sc_softening > 0.0:
                            sc_R = np.sqrt(sc_R**2 + sc_softening**2)
                            sc_nhat = dict(sc_nhat)
                            sc_nhat["R"] = sc_R

                    (
                        sc_dp_x,
                        sc_dp_y,
                        sc_dp_z,
                        sc_dp_t,
                        sc_df_x,
                        sc_df_y,
                        sc_df_z,
                        sc_dscalar,
                    ) = compute_vectorized_contributions(
                        h=h,
                        charge_i=float(force_particle_charge),
                        mass_i=float(particle_mass),
                        gamma_i=particle_gamma,
                        beta_vec=particle_beta,
                        nhat_nx=np.asarray(sc_nhat["nx"], dtype=float),
                        nhat_ny=np.asarray(sc_nhat["ny"], dtype=float),
                        nhat_nz=np.asarray(sc_nhat["nz"], dtype=float),
                        R_separation=sc_R,
                        samples=sc_samples,
                        apply_external=True,
                        verbosity=0,
                    )
                    accumulated_momentum_x += sc_dp_x
                    accumulated_momentum_y += sc_dp_y
                    accumulated_momentum_z += sc_dp_z
                    accumulated_momentum_t += sc_dp_t
                    accumulated_field_x += sc_df_x
                    accumulated_field_y += sc_df_y
                    accumulated_field_z += sc_df_z
                    accumulated_scalar_potential += sc_dscalar

            # ================================================================
            # STEP 4c: Prescribed external uniform fields
            # ================================================================
            if external_field is not None and getattr(external_field, "enabled", False):
                if sc_convergence_mode == "variable_geometry" and sc_iteration > 0:
                    field_position = (
                        float(working_x),
                        float(working_y),
                        float(working_z),
                    )
                else:
                    field_position = (
                        float(current_state["x"][particle_idx]),
                        float(current_state["y"][particle_idx]),
                        float(current_state["z"][particle_idx]),
                    )
                if rfs_selected:
                    (
                        local_electric_field_native,
                        local_magnetic_field_native,
                        local_magnetic_gradient_native_per_mm,
                    ) = evaluate_external_field_native(
                        external_field,
                        position_mm=field_position,
                        time_ns=float(current_state["t"][particle_idx]),
                    )
                else:
                    (
                        local_electric_field_v_m,
                        local_magnetic_field_t,
                        local_magnetic_gradient_t_per_m,
                    ) = evaluate_external_field_si(
                        external_field,
                        position_mm=field_position,
                        time_ns=float(current_state["t"][particle_idx]),
                    )
                (
                    ext_dp_x,
                    ext_dp_y,
                    ext_dp_z,
                    ext_dp_t,
                ) = compute_uniform_external_field_impulse(
                    external_field,
                    charge=float(force_particle_charge),
                    gamma=(
                        float(current_state["gamma"][particle_idx])
                        if second_order_exact_source_selected
                        else float(particle_gamma)
                    ),
                    beta=(
                        float(exact_ordinary_response_beta[0]),
                        float(exact_ordinary_response_beta[1]),
                        float(exact_ordinary_response_beta[2]),
                    ),
                    h_step=float(h),
                    position=field_position,
                    time=float(current_state["t"][particle_idx]),
                )
                accumulated_momentum_x += ext_dp_x
                accumulated_momentum_y += ext_dp_y
                accumulated_momentum_z += ext_dp_z
                accumulated_momentum_t += ext_dp_t
                if exact_endpoint_recomposition_selected:
                    exact_mechanical_temporal_impulse += float(ext_dp_t)
                if second_order_exact_source_selected:
                    additional_start_force_native += np.asarray(
                        (ext_dp_t, ext_dp_x, ext_dp_y, ext_dp_z),
                        dtype=float,
                    ) / float(h)

                dipole_active = (
                    bool(
                        _get_state_scalar(
                            current_state,
                            "magnetic_dipole_active",
                            particle_idx,
                        )
                    )
                    if "magnetic_dipole_active" in current_state
                    else False
                )
                sg_active = (
                    bool(
                        _get_state_scalar(
                            current_state,
                            "stern_gerlach_active",
                            particle_idx,
                        )
                    )
                    if "stern_gerlach_active" in current_state
                    else False
                )
                if dipole_active and sg_active and not rfs_force_selected:
                    from .magnetic_dipole import (
                        STATIC_REST_GRADIENT_MAX_BETA,
                        stern_gerlach_rest_impulse_native,
                    )

                    spin_vector = np.asarray(
                        (
                            current_state["spin_x"][particle_idx],
                            current_state["spin_y"][particle_idx],
                            current_state["spin_z"][particle_idx],
                        ),
                        dtype=float,
                    )
                    signed_moment = float(
                        _get_state_scalar(
                            current_state,
                            "magnetic_moment_j_per_t",
                            particle_idx,
                        )
                    )
                    sg_impulse = np.asarray(
                        stern_gerlach_rest_impulse_native(
                            signed_moment * spin_vector,
                            local_magnetic_gradient_t_per_m,
                            float(h),
                        ),
                        dtype=float,
                    )
                    beta_magnitude = float(np.linalg.norm(exact_ordinary_response_beta))
                    trial_px = accumulated_momentum_x + float(sg_impulse[0])
                    trial_py = accumulated_momentum_y + float(sg_impulse[1])
                    trial_pz = accumulated_momentum_z + float(sg_impulse[2])
                    trial_mechanical_momentum = np.asarray(
                        _mechanical_momentum_components(
                            px=trial_px,
                            py=trial_py,
                            pz=trial_pz,
                            particle_mass=particle_mass,
                            field_x=accumulated_field_x,
                            field_y=accumulated_field_y,
                            field_z=accumulated_field_z,
                        ),
                        dtype=float,
                    )
                    trial_momentum_magnitude = float(
                        np.linalg.norm(trial_mechanical_momentum)
                    )
                    trial_beta_magnitude = trial_momentum_magnitude / float(
                        np.sqrt(
                            trial_momentum_magnitude**2 + (particle_mass * C_MMNS) ** 2
                        )
                    )
                    if (
                        np.any(sg_impulse)
                        and max(beta_magnitude, trial_beta_magnitude)
                        > STATIC_REST_GRADIENT_MAX_BETA
                    ):
                        raise NotImplementedError(
                            "static_rest_gradient is restricted to "
                            f"|beta| <= {STATIC_REST_GRADIENT_MAX_BETA:g}; "
                            "this step would span "
                            f"|beta|={beta_magnitude:.6g} to "
                            f"{trial_beta_magnitude:.6g}. "
                            "Disable the Stern-Gerlach force or choose a future "
                            "named covariant gradient model."
                        )
                    accumulated_momentum_x += float(sg_impulse[0])
                    accumulated_momentum_y += float(sg_impulse[1])
                    accumulated_momentum_z += float(sg_impulse[2])
                    accumulated_momentum_t += float(
                        np.dot(exact_ordinary_response_beta, sg_impulse)
                    )
                    if exact_endpoint_recomposition_selected:
                        exact_mechanical_temporal_impulse += float(
                            np.dot(exact_ordinary_response_beta, sg_impulse)
                        )
                    if second_order_exact_source_selected:
                        additional_start_force_native += np.asarray(
                            (
                                np.dot(exact_ordinary_response_beta, sg_impulse),
                                sg_impulse[0],
                                sg_impulse[1],
                                sg_impulse[2],
                            ),
                            dtype=float,
                        ) / float(h)
                    stern_gerlach_impulse_applied = bool(np.any(sg_impulse))

            # ================================================================
            # STEP 4d: Exact ordinary charge/dipole sources and RFS response
            # ================================================================
            dipole_active = bool(
                current_state.get("magnetic_dipole_active", np.zeros(num_particles))[
                    particle_idx
                ]
            )
            precession_active = bool(
                current_state.get("spin_precession_active", np.zeros(num_particles))[
                    particle_idx
                ]
            )
            sg_active = bool(
                current_state.get("stern_gerlach_active", np.zeros(num_particles))[
                    particle_idx
                ]
            )
            if (
                exact_charge_source_selected
                and sim_type == SimulationType.BUNCH_TO_BUNCH
                and len(trajectory_ext) > 0
            ):
                from .charge_source_interactions import (
                    charge_source_interaction_from_field_native,
                    charge_source_interaction_from_response_native,
                )
                from .retarded_fields import (
                    ObserverEvent,
                    RetardedChargeResponseGradientResult,
                    RetardedHistoryError,
                    evaluate_retarded_charge_field_gradient_native,
                    evaluate_retarded_charge_response_gradient_native,
                )

                if sc_convergence_mode == "variable_geometry" and sc_iteration > 0:
                    charge_source_position = (
                        float(working_x),
                        float(working_y),
                        float(working_z),
                    )
                else:
                    charge_source_position = (
                        float(current_state["x"][particle_idx]),
                        float(current_state["y"][particle_idx]),
                        float(current_state["z"][particle_idx]),
                    )
                try:
                    if (
                        sc_convergence_mode == "fixed_geometry"
                        and exact_charge_field_cache is not None
                    ):
                        exact_charge_field = exact_charge_field_cache
                    else:
                        charge_history = (
                            exact_source_history
                            if exact_source_history is not None
                            else (
                                traj_ext_soa
                                if traj_ext_soa is not None
                                else trajectory_ext
                            )
                        )
                        charge_event = ObserverEvent(
                            time_ns=float(current_state["t"][particle_idx]),
                            position_mm=charge_source_position,
                        )
                        charge_relative_step = max(
                            1.0e-4,
                            (
                                float(magnetic_dipole.source.relative_stencil_step)
                                if magnetic_dipole is not None
                                and magnetic_dipole.source.active
                                else 1.0e-4
                            ),
                        )
                        charge_minimum_step = max(
                            1.0e-15,
                            (
                                float(magnetic_dipole.source.minimum_stencil_step_mm)
                                if magnetic_dipole is not None
                                and magnetic_dipole.source.active
                                else 1.0e-15
                            ),
                        )
                        charge_root_tolerance = (
                            float(magnetic_dipole.source.root_tolerance_mm)
                            if magnetic_dipole is not None
                            and magnetic_dipole.source.active
                            else 1.0e-21
                        )
                        charge_root_iterations = (
                            int(magnetic_dipole.source.max_root_iterations)
                            if magnetic_dipole is not None
                            and magnetic_dipole.source.active
                            else 96
                        )
                        charge_backend = (
                            magnetic_dipole.exact_retarded_backend
                            if magnetic_dipole is not None
                            else "python"
                        )
                        if charge_backend in {
                            "numba_analytic_charge_response_serial",
                            "numba_analytic_charge_dipole_response_serial",
                        }:
                            exact_charge_field = (
                                evaluate_retarded_charge_response_gradient_native(
                                    charge_history,
                                    charge_event,
                                    relative_step=charge_relative_step,
                                    minimum_step_mm=charge_minimum_step,
                                    root_tolerance_mm=charge_root_tolerance,
                                    max_root_iterations=charge_root_iterations,
                                )
                            )
                        else:
                            exact_charge_field = (
                                evaluate_retarded_charge_field_gradient_native(
                                    charge_history,
                                    charge_event,
                                    relative_step=charge_relative_step,
                                    minimum_step_mm=charge_minimum_step,
                                    root_tolerance_mm=charge_root_tolerance,
                                    max_root_iterations=charge_root_iterations,
                                    backend=charge_backend,
                                )
                            )
                        if sc_convergence_mode == "fixed_geometry":
                            exact_charge_field_cache = exact_charge_field
                except RetardedHistoryError:
                    # INERTIAL_PREHISTORY preflights every displaced light cone.
                    # Missing history after that gate is a model failure, never a
                    # request to fall back to cold-start force suppression.
                    raise

                if isinstance(exact_charge_field, RetardedChargeResponseGradientResult):
                    exact_charge_analytic_response = exact_charge_field
                    exact_charge_source_interaction = (
                        charge_source_interaction_from_response_native(
                            exact_charge_field,
                            four_velocity_mm_ns=_four_velocity_native(
                                exact_ordinary_response_beta
                            ),
                            observer_charge_native=float(force_particle_charge),
                            proper_time_step_ns=float(h),
                            contraction_backend="numba_strict_serial",
                        )
                    )
                else:
                    exact_charge_source_interaction = (
                        charge_source_interaction_from_field_native(
                            exact_charge_field,
                            four_velocity_mm_ns=_four_velocity_native(
                                exact_ordinary_response_beta
                            ),
                            observer_charge_native=float(force_particle_charge),
                            proper_time_step_ns=float(h),
                        )
                    )

                # Evolve the gauge-invariant mechanical momentum.  The
                # accepted canonical state is rebuilt from A at the endpoint
                # after both bunch endpoints have been appended to history.
                charge_mechanical_impulse = (
                    exact_charge_source_interaction.mechanical_four_impulse
                )
                accumulated_momentum_t += float(charge_mechanical_impulse[0])
                exact_mechanical_temporal_impulse += float(charge_mechanical_impulse[0])
                accumulated_momentum_x += float(charge_mechanical_impulse[1])
                accumulated_momentum_y += float(charge_mechanical_impulse[2])
                accumulated_momentum_z += float(charge_mechanical_impulse[3])

                charge_potential_momentum = (
                    exact_charge_source_interaction.canonical_potential_momentum
                )
                charge_canonical_ready = bool(
                    current_state.get(
                        "charge_source_canonical_ready",
                        np.zeros(num_particles, dtype=bool),
                    )[particle_idx]
                )
                if not charge_canonical_ready:
                    # Public t=0 input is mechanical.  The inertial prefix makes
                    # the ordinary retarded potential available immediately, so
                    # initialize P=p+qA/c once without changing p or beta.
                    accumulated_momentum_t += float(charge_potential_momentum[0])
                    accumulated_momentum_x += float(charge_potential_momentum[1])
                    accumulated_momentum_y += float(charge_potential_momentum[2])
                    accumulated_momentum_z += float(charge_potential_momentum[3])
                result["charge_source_canonical_ready"][particle_idx] = True
                accumulated_field_x += float(
                    charge_potential_momentum[1] / particle_mass
                )
                accumulated_field_y += float(
                    charge_potential_momentum[2] / particle_mass
                )
                accumulated_field_z += float(
                    charge_potential_momentum[3] / particle_mass
                )
                accumulated_scalar_potential += float(
                    exact_charge_source_interaction.four_potential[0]
                )
                exact_source_start_four_potential += (
                    exact_charge_source_interaction.four_potential
                )
                if exact_charge_source_interaction.field is not None:
                    rfs_field_tensor += (
                        exact_charge_source_interaction.field.field.field_tensor
                    )
                    rfs_partial_f += exact_charge_source_interaction.field.partial_f

            if (
                dipole_source_selected
                and sim_type == SimulationType.BUNCH_TO_BUNCH
                and len(trajectory_ext) > 0
            ):
                assert magnetic_dipole is not None
                from .dipole_source_interactions import (
                    dipole_source_interaction_from_field_native,
                    dipole_source_interaction_from_response_native,
                )
                from .dipole_hertz_jet import (
                    evaluate_retarded_dipole_field_gradient_hertz_jet_native,
                )
                from .retarded_dipole_fields import (
                    RetardedDipoleResponseGradientResult,
                    evaluate_retarded_dipole_field_gradient_native,
                )
                from .retarded_fields import ObserverEvent, RetardedHistoryError

                if exact_dipole_source_collection is not None:
                    from .causal_c5_dipole_provider import (
                        evaluate_causal_c5_dipole_source_collection_native,
                    )

                if sc_convergence_mode == "variable_geometry" and sc_iteration > 0:
                    dipole_source_position = (
                        float(working_x),
                        float(working_y),
                        float(working_z),
                    )
                else:
                    dipole_source_position = (
                        float(current_state["x"][particle_idx]),
                        float(current_state["y"][particle_idx]),
                        float(current_state["z"][particle_idx]),
                    )
                try:
                    if (
                        sc_convergence_mode == "fixed_geometry"
                        and dipole_source_field_cache is not None
                    ):
                        dipole_source_field = dipole_source_field_cache
                    else:
                        if exact_dipole_source_collection is not None:
                            dipole_source_field = (
                                evaluate_causal_c5_dipole_source_collection_native(
                                    exact_dipole_source_collection,
                                    ObserverEvent(
                                        time_ns=float(current_state["t"][particle_idx]),
                                        position_mm=dipole_source_position,
                                    ),
                                    root_tolerance_mm=(
                                        magnetic_dipole.source.root_tolerance_mm
                                    ),
                                    max_root_iterations=(
                                        magnetic_dipole.source.max_root_iterations
                                    ),
                                    minimum_separation_mm=(
                                        magnetic_dipole.source.minimum_separation_mm
                                    ),
                                )
                            )
                        elif (
                            exact_endpoint_recomposition_selected
                            and magnetic_dipole.exact_retarded_backend
                            == "numba_analytic_charge_dipole_response_serial"
                        ):
                            dipole_source_field = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
                                (
                                    exact_source_history
                                    if exact_source_history is not None
                                    else (
                                        traj_ext_soa
                                        if traj_ext_soa is not None
                                        else trajectory_ext
                                    )
                                ),
                                ObserverEvent(
                                    time_ns=float(current_state["t"][particle_idx]),
                                    position_mm=dipole_source_position,
                                ),
                                require_complete_history=True,
                                fallback_relative_step=(
                                    magnetic_dipole.source.relative_stencil_step
                                ),
                                fallback_minimum_step_mm=(
                                    magnetic_dipole.source.minimum_stencil_step_mm
                                ),
                                minimum_separation_mm=(
                                    magnetic_dipole.source.minimum_separation_mm
                                ),
                                root_tolerance_mm=(
                                    magnetic_dipole.source.root_tolerance_mm
                                ),
                                max_root_iterations=(
                                    magnetic_dipole.source.max_root_iterations
                                ),
                                response_kernel="numba_sparse_strict_serial",
                                fallback_backend="numba_full_strict_serial",
                                spin_interpolation_model=(
                                    exact_source_spin_interpolation_model
                                ),
                            ).response
                        else:
                            dipole_source_field = (
                                evaluate_retarded_dipole_field_gradient_native(
                                    (
                                        exact_source_history
                                        if exact_source_history is not None
                                        else (
                                            traj_ext_soa
                                            if traj_ext_soa is not None
                                            else trajectory_ext
                                        )
                                    ),
                                    ObserverEvent(
                                        time_ns=float(current_state["t"][particle_idx]),
                                        position_mm=dipole_source_position,
                                    ),
                                    relative_step=(
                                        magnetic_dipole.source.relative_stencil_step
                                    ),
                                    minimum_step_mm=(
                                        magnetic_dipole.source.minimum_stencil_step_mm
                                    ),
                                    minimum_separation_mm=(
                                        magnetic_dipole.source.minimum_separation_mm
                                    ),
                                    root_tolerance_mm=(
                                        magnetic_dipole.source.root_tolerance_mm
                                    ),
                                    max_root_iterations=(
                                        magnetic_dipole.source.max_root_iterations
                                    ),
                                    backend=magnetic_dipole.exact_retarded_backend,
                                    spin_interpolation_model=(
                                        exact_source_spin_interpolation_model
                                    ),
                                )
                            )
                        if sc_convergence_mode == "fixed_geometry":
                            dipole_source_field_cache = dipole_source_field
                except RetardedHistoryError:
                    if startup_mode is StartupMode.INERTIAL_PREHISTORY:
                        raise
                    # As for the exact charge-field derivative, COLD_START
                    # contributes nothing until every nested stencil event has
                    # an explicitly bracketed source light cone.
                    dipole_source_field = None

                if dipole_source_field is not None:
                    if isinstance(
                        dipole_source_field, RetardedDipoleResponseGradientResult
                    ):
                        exact_dipole_analytic_response = dipole_source_field
                        dipole_source_interaction = (
                            dipole_source_interaction_from_response_native(
                                dipole_source_field,
                                four_velocity_mm_ns=_four_velocity_native(
                                    exact_ordinary_response_beta
                                ),
                                observer_charge_native=float(force_particle_charge),
                                proper_time_step_ns=float(h),
                                contraction_backend="numba_strict_serial",
                            )
                        )
                    else:
                        dipole_source_interaction = (
                            dipole_source_interaction_from_field_native(
                                dipole_source_field,
                                four_velocity_mm_ns=_four_velocity_native(
                                    exact_ordinary_response_beta
                                ),
                                observer_charge_native=float(force_particle_charge),
                                proper_time_step_ns=float(h),
                            )
                        )

                if dipole_source_interaction is not None:
                    if exact_endpoint_recomposition_selected:
                        ordinary_dipole_impulse = (
                            dipole_source_interaction.mechanical_four_impulse
                        )
                    else:
                        # Retain the older COLD_START diagnostic boundary until
                        # it receives the same pair-level endpoint finalizer.
                        ordinary_dipole_impulse = (
                            dipole_source_interaction.canonical_four_impulse
                        )
                        if ordinary_dipole_impulse is None:
                            raise RuntimeError(
                                "compact dipole response is unavailable for the "
                                "legacy canonical-force path"
                            )
                    accumulated_momentum_t += float(ordinary_dipole_impulse[0])
                    if exact_endpoint_recomposition_selected:
                        exact_mechanical_temporal_impulse += float(
                            ordinary_dipole_impulse[0]
                        )
                    accumulated_momentum_x += float(ordinary_dipole_impulse[1])
                    accumulated_momentum_y += float(ordinary_dipole_impulse[2])
                    accumulated_momentum_z += float(ordinary_dipole_impulse[3])

                    dipole_potential_momentum = (
                        dipole_source_interaction.canonical_potential_momentum
                    )
                    dipole_canonical_ready = bool(
                        current_state.get(
                            "dipole_source_canonical_ready",
                            np.zeros(num_particles, dtype=bool),
                        )[particle_idx]
                    )
                    if not dipole_canonical_ready:
                        # Public initial states specify mechanical momentum.  A
                        # COLD_START dipole field becomes available only after
                        # an explicit light cone can be bracketed.  Rebase the
                        # state once from p to P=p+qA/c when that happens so the
                        # newly available vector potential cannot create a
                        # spurious mechanical-momentum jump.
                        accumulated_momentum_t += float(dipole_potential_momentum[0])
                        accumulated_momentum_x += float(dipole_potential_momentum[1])
                        accumulated_momentum_y += float(dipole_potential_momentum[2])
                        accumulated_momentum_z += float(dipole_potential_momentum[3])
                    result["dipole_source_canonical_ready"][particle_idx] = True
                    accumulated_field_x += float(
                        dipole_potential_momentum[1] / particle_mass
                    )
                    accumulated_field_y += float(
                        dipole_potential_momentum[2] / particle_mass
                    )
                    accumulated_field_z += float(
                        dipole_potential_momentum[3] / particle_mass
                    )
                    # accumulated_scalar_potential stores raw phi; its q/c
                    # conversion remains centralized below with the charge
                    # source potential.
                    accumulated_scalar_potential += float(
                        dipole_source_interaction.four_potential[0]
                    )
                    if exact_endpoint_recomposition_selected:
                        exact_source_start_four_potential += (
                            dipole_source_interaction.four_potential
                        )

            analytic_response_payloads = tuple(
                payload
                for payload in (
                    exact_charge_analytic_response,
                    exact_dipole_analytic_response,
                )
                if payload is not None
            )
            if analytic_response_payloads:
                exact_analytic_antisymmetric_response = np.sum(
                    [
                        payload.antisymmetric_response
                        for payload in analytic_response_payloads
                    ],
                    axis=0,
                )
                exact_analytic_partial_antisymmetric_response = np.sum(
                    [
                        payload.partial_antisymmetric_response
                        for payload in analytic_response_payloads
                    ],
                    axis=0,
                )

            if rfs_selected and dipole_active and (precession_active or sg_active):
                from .magnetic_dipole import HBAR_NATIVE, boost_rest_polarization
                from .retarded_fields import (
                    ObserverEvent,
                    evaluate_retarded_charge_field_gradient_native,
                )
                from .rfs import rfs_four_force_native

                external_tensor, external_partial_f = _external_tensor_gradient(
                    local_electric_field_native,
                    local_magnetic_field_native,
                    local_magnetic_gradient_native_per_mm,
                )
                rfs_field_tensor += external_tensor
                rfs_partial_f += external_partial_f

                if (
                    dipole_source_interaction is not None
                    and dipole_source_interaction.field is not None
                ):
                    rfs_field_tensor += dipole_source_interaction.field.field_tensor
                    rfs_partial_f += dipole_source_interaction.field.partial_f

                if (
                    not exact_charge_source_selected
                    and sim_type == SimulationType.BUNCH_TO_BUNCH
                    and len(trajectory_ext) > 0
                ):
                    if sc_convergence_mode == "variable_geometry" and sc_iteration > 0:
                        rfs_position = (
                            float(working_x),
                            float(working_y),
                            float(working_z),
                        )
                    else:
                        rfs_position = (
                            float(current_state["x"][particle_idx]),
                            float(current_state["y"][particle_idx]),
                            float(current_state["z"][particle_idx]),
                        )
                    from .retarded_fields import RetardedHistoryError

                    try:
                        charge_field = evaluate_retarded_charge_field_gradient_native(
                            (
                                traj_ext_soa
                                if traj_ext_soa is not None
                                else trajectory_ext
                            ),
                            ObserverEvent(
                                time_ns=float(current_state["t"][particle_idx]),
                                position_mm=rfs_position,
                            ),
                            backend=magnetic_dipole.exact_retarded_backend,
                        )
                    except RetardedHistoryError:
                        if startup_mode is StartupMode.INERTIAL_PREHISTORY:
                            raise
                        # COLD_START intentionally supplies no extrapolated
                        # field until explicit source samples bracket the light
                        # cone at every finite-difference stencil event.
                        charge_field = None
                    if charge_field is not None:
                        rfs_field_tensor += charge_field.field.field_tensor
                        rfs_partial_f += charge_field.partial_f

                # The ordinary exact-source path already supplies q F u as a
                # mechanical impulse. Inject only the normalized-spin dipole
                # term (mu/c) G[a] u here, in all four components, to avoid
                # counting the Lorentz force twice.
                if rfs_force_selected and sg_active:
                    signed_moment_native = float(
                        current_state["magnetic_moment_native"][particle_idx]
                    )
                    spin_quantum_number = float(
                        current_state["spin_quantum_number"][particle_idx]
                    )
                    if spin_quantum_number > 0.0 and signed_moment_native != 0.0:
                        rest_spin = np.asarray(
                            (
                                current_state["spin_x"][particle_idx],
                                current_state["spin_y"][particle_idx],
                                current_state["spin_z"][particle_idx],
                            ),
                            dtype=float,
                        )
                        beta_vector = (
                            exact_ordinary_response_beta
                            if second_order_exact_source_selected
                            else np.asarray(particle_beta, dtype=float)
                        )
                        normalized_spin = boost_rest_polarization(
                            rest_spin, beta_vector
                        )
                        dipole_force_native = rfs_four_force_native(
                            four_velocity_mm_ns=_four_velocity_native(beta_vector),
                            spin_four_vector=normalized_spin,
                            field_tensor=rfs_field_tensor,
                            partial_f=rfs_partial_f,
                            charge_native=0.0,
                            magnetic_moment_native=signed_moment_native,
                        )
                        if exact_analytic_antisymmetric_response is not None:
                            from .contracted_antisymmetric_response_numba import (
                                antisymmetric_response_rfs_strict_serial,
                            )

                            dipole_force_native += (
                                antisymmetric_response_rfs_strict_serial(
                                    _four_velocity_native(beta_vector),
                                    normalized_spin,
                                    exact_analytic_antisymmetric_response,
                                    exact_analytic_partial_antisymmetric_response,
                                    float(force_particle_charge),
                                    float(particle_mass),
                                    signed_moment_native,
                                    spin_quantum_number * HBAR_NATIVE,
                                )[1]
                            )
                        rfs_dipole_force_native += dipole_force_native
                        dipole_impulse_native = dipole_force_native * float(h)
                        accumulated_momentum_t += float(dipole_impulse_native[0])
                        if exact_endpoint_recomposition_selected:
                            exact_mechanical_temporal_impulse += float(
                                dipole_impulse_native[0]
                            )
                        accumulated_momentum_x += float(dipole_impulse_native[1])
                        accumulated_momentum_y += float(dipole_impulse_native[2])
                        accumulated_momentum_z += float(dipole_impulse_native[3])

            if second_order_exact_source_selected:
                from .antisymmetric_response_rfs import (
                    antisymmetric_response_charge_force_derivative_native,
                )
                from .canonical_momentum import (
                    mechanical_lorentz_four_force_derivative_native,
                )

                start_four_velocity = _four_velocity_native(
                    exact_ordinary_response_beta
                )
                ordinary_force_native = np.zeros(4, dtype=float)
                for interaction in (
                    exact_charge_source_interaction,
                    dipole_source_interaction,
                ):
                    if interaction is not None:
                        ordinary_force_native += interaction.mechanical_four_force
                start_four_acceleration = (
                    ordinary_force_native
                    + rfs_dipole_force_native
                    + additional_start_force_native
                ) / particle_mass
                from .magnetic_dipole import HBAR_NATIVE, boost_rest_polarization

                start_rest_spin = np.asarray(
                    (
                        current_state["spin_x"][particle_idx],
                        current_state["spin_y"][particle_idx],
                        current_state["spin_z"][particle_idx],
                    ),
                    dtype=float,
                )
                invariant_spin_native = (
                    float(current_state["spin_quantum_number"][particle_idx])
                    * HBAR_NATIVE
                )
                result["_intrinsic_spin_start_four_velocity"][
                    particle_idx
                ] = start_four_velocity
                result["_intrinsic_spin_start_non_self_four_acceleration"][
                    particle_idx
                ] = start_four_acceleration
                result["_intrinsic_spin_start_physical_four_spin"][particle_idx] = (
                    invariant_spin_native
                    * boost_rest_polarization(
                        start_rest_spin,
                        exact_ordinary_response_beta,
                    )
                )
                if intrinsic_spin_diagnostic_selected:
                    charge_for_reduction = float(force_particle_charge)
                    signed_moment_for_reduction = float(
                        current_state["magnetic_moment_native"][particle_idx]
                    )
                    if charge_for_reduction == 0.0 or invariant_spin_native == 0.0:
                        g_factor = 0.0
                        unavailable_reason = (
                            "intrinsic q-mu reduction requires nonzero charge and spin"
                        )
                        analytical_reduction = None
                    else:
                        g_factor = (
                            2.0
                            * particle_mass
                            * C_MMNS
                            * signed_moment_for_reduction
                            / (charge_for_reduction * invariant_spin_native)
                        )
                        unavailable_reason = None
                        analytical_reduction = None
                        if external_field is not None and getattr(
                            external_field, "enabled", False
                        ):
                            unavailable_reason = (
                                "prescribed external-field potential jet is unavailable"
                            )
                        elif exact_source_history is None:
                            unavailable_reason = (
                                "exact retarded source history is unavailable"
                            )
                        else:
                            from .spin_self_force_reduction_oracle import (
                                evaluate_retarded_potential_intrinsic_spin_reduction_native,
                            )
                            from .retarded_fields import ObserverEvent

                            source_settings = magnetic_dipole.source
                            analytical_result = evaluate_retarded_potential_intrinsic_spin_reduction_native(
                                source_history=exact_source_history,
                                observer_event=ObserverEvent(
                                    time_ns=float(current_state["t"][particle_idx]),
                                    position_mm=(
                                        float(current_state["x"][particle_idx]),
                                        float(current_state["y"][particle_idx]),
                                        float(current_state["z"][particle_idx]),
                                    ),
                                ),
                                four_velocity_mm_ns=start_four_velocity,
                                normalized_spin_four_vector=(
                                    boost_rest_polarization(
                                        start_rest_spin,
                                        exact_ordinary_response_beta,
                                    )
                                ),
                                charge_native=charge_for_reduction,
                                mass_amu=particle_mass,
                                invariant_spin_native=invariant_spin_native,
                                g_factor=g_factor,
                                require_complete_history=True,
                                include_dipole_source=(source_settings.active),
                                minimum_separation_mm=(
                                    source_settings.minimum_separation_mm
                                    if source_settings.active
                                    else 1.0e-15
                                ),
                                root_tolerance_mm=(
                                    source_settings.root_tolerance_mm
                                    if source_settings.active
                                    else 1.0e-21
                                ),
                                max_root_iterations=(
                                    source_settings.max_root_iterations
                                    if source_settings.active
                                    else 96
                                ),
                                spin_interpolation_model=(
                                    exact_source_spin_interpolation_model
                                ),
                            )
                            analytical_reduction = analytical_result.reduction
                            unavailable_reason = analytical_result.unavailable_reason
                    result["_intrinsic_spin_start_analytical_reduction"][
                        particle_idx
                    ] = analytical_reduction
                    result["_intrinsic_spin_start_analytical_unavailable_reason"][
                        particle_idx
                    ] = unavailable_reason
                    result["_intrinsic_spin_charge_native"][
                        particle_idx
                    ] = charge_for_reduction
                    result["_intrinsic_spin_mass_amu"][particle_idx] = particle_mass
                    result["_intrinsic_spin_g_factor"][particle_idx] = g_factor
                ordinary_force_derivative = np.zeros(4, dtype=float)
                for interaction in (
                    exact_charge_source_interaction,
                    dipole_source_interaction,
                ):
                    if interaction is None:
                        continue
                    if interaction.response is not None:
                        ordinary_force_derivative += (
                            antisymmetric_response_charge_force_derivative_native(
                                four_velocity_mm_ns=start_four_velocity,
                                four_acceleration_mm_ns2=start_four_acceleration,
                                antisymmetric_response=(
                                    interaction.response.antisymmetric_response
                                ),
                                partial_antisymmetric_response=(
                                    interaction.response.partial_antisymmetric_response
                                ),
                                charge_native=float(force_particle_charge),
                            )
                        )
                    elif interaction.field is not None:
                        interaction_field = interaction.field
                        if hasattr(interaction_field, "field"):
                            field_tensor = interaction_field.field.field_tensor
                        else:
                            field_tensor = interaction_field.field_tensor
                        ordinary_force_derivative += (
                            mechanical_lorentz_four_force_derivative_native(
                                four_velocity_mm_ns=start_four_velocity,
                                four_acceleration_mm_ns2=start_four_acceleration,
                                field_tensor=field_tensor,
                                partial_f=interaction_field.partial_f,
                                charge_native=float(force_particle_charge),
                            )
                        )
                second_order_correction = (
                    0.5 * float(h) * float(h) * ordinary_force_derivative
                )
                accumulated_momentum_t += float(second_order_correction[0])
                accumulated_momentum_x += float(second_order_correction[1])
                accumulated_momentum_y += float(second_order_correction[2])
                accumulated_momentum_z += float(second_order_correction[3])
                exact_mechanical_temporal_impulse += float(second_order_correction[0])

            # ================================================================
            # STEP 4: Update momentum and derive gamma from Pt
            # ================================================================
            result["Px"][particle_idx] = accumulated_momentum_x
            result["Py"][particle_idx] = accumulated_momentum_y
            result["Pz"][particle_idx] = accumulated_momentum_z
            result["Pt"][particle_idx] = accumulated_momentum_t

            scalar_potential_contribution = _scalar_potential_momentum_contribution(
                force_particle_charge, accumulated_scalar_potential
            )

            # Preserve the unconstrained temporal canonical update before any
            # named mass-shell projection or nonlinear relaxation. Non-exact
            # paths ledger the correction from this raw value. The exact
            # accepted-on-shell path uses the stable mechanical energy balance
            # below so rest-scale subtraction cannot dominate atomic energies.
            raw_canonical_pt_before_constraints = float(result["Pt"][particle_idx])

            if stern_gerlach_impulse_applied:
                # The static-gradient helper supplies a mechanical spatial
                # impulse, not a complete canonical Hamiltonian. Put the
                # combined momentum back on the ordinary mass shell before
                # positions, acceleration, radiation, and running averages
                # are evaluated. This is deliberately part of the named
                # approximation and must be replaced with its energy model if
                # a covariant Stern--Gerlach Hamiltonian is added later.
                _, sg_projected_pt = _canonical_pt_from_mechanical_mass_shell(
                    px=result["Px"][particle_idx],
                    py=result["Py"][particle_idx],
                    pz=result["Pz"][particle_idx],
                    particle_mass=particle_mass,
                    scalar_potential_contribution=scalar_potential_contribution,
                    field_x=accumulated_field_x,
                    field_y=accumulated_field_y,
                    field_z=accumulated_field_z,
                )
                result["Pt"][particle_idx] = sg_projected_pt

            # ================================================================
            # STEP 4a: Correct Pt during SC iterations based on mode
            # ================================================================
            # CRITICAL: Enforce constraints at each iteration
            # Mode determines HOW we correct Pt, but both modes check both errors
            if sc_enabled and sc_iteration > 0:
                kinetic_pt_from_mass_shell, Pt_from_mass_shell = (
                    _canonical_pt_from_mechanical_mass_shell(
                        px=result["Px"][particle_idx],
                        py=result["Py"][particle_idx],
                        pz=result["Pz"][particle_idx],
                        particle_mass=particle_mass,
                        scalar_potential_contribution=scalar_potential_contribution,
                        field_x=accumulated_field_x,
                        field_y=accumulated_field_y,
                        field_z=accumulated_field_z,
                    )
                )

                Pt_before_correction = np.float64(result["Pt"][particle_idx])

                # Determine Pt and P correction based on mode
                if sc_convergence_mode in ("fixed_geometry", "variable_geometry"):
                    # Modes 1 & 2: Project Pt onto mass-shell (asymmetric relaxation)
                    Pt_corrected = Pt_from_mass_shell

                    if sc_verbosity >= 3:
                        print(
                            f"      Mode: {sc_convergence_mode}, Pt_ms={Pt_from_mass_shell:.6e}"
                        )

                    # Apply relaxation to Pt only (asymmetric)
                    relaxation_weight = sc_mass_shell_relaxation
                    Pt_final = (
                        relaxation_weight * Pt_corrected
                        + (1.0 - relaxation_weight) * Pt_before_correction
                    )

                    result["Pt"][particle_idx] = float(Pt_final)
                    # P_xyz unchanged (from forces)

                else:
                    raise ValueError(f"Unknown convergence_mode: {sc_convergence_mode}")

                # Log relaxation details
                if sc_verbosity >= 3:
                    correction_magnitude = abs(Pt_final - Pt_before_correction)
                    print(
                        f"      After relaxation (α={relaxation_weight}): "
                        f"Pt {Pt_before_correction:.6e} → {Pt_final:.6e} "
                        f"(Δ={correction_magnitude:.6e})"
                    )

                if sc_verbosity >= 3:
                    correction_magnitude = abs(Pt_final - Pt_before_correction)
                    print(
                        f"      After relaxation (α={relaxation_weight}): "
                        f"Pt {Pt_before_correction:.6e} → {Pt_final:.6e} "
                        f"(Δ={correction_magnitude:.6e})"
                    )

            # ================================================================
            # STEP 4b: Compute gamma from energy
            # ================================================================
            # Gamma from relativistic energy with scalar potential correction:
            # γ = (Pt - q·Φ/c) / (mc), where Φ = Σ(q_j / (R_sep_j * k_factor_j)).
            # Pt is energy-over-c, so qΦ is converted to momentum units.
            kinetic_energy = (
                np.float64(result["Pt"][particle_idx]) - scalar_potential_contribution
            )
            gamma_from_energy = kinetic_energy / np.float64(particle_mass * C_MMNS)
            result["gamma"][particle_idx] = gamma_from_energy

            kinetic_pt_from_mass_shell, Pt_from_mass_shell = (
                _canonical_pt_from_mechanical_mass_shell(
                    px=result["Px"][particle_idx],
                    py=result["Py"][particle_idx],
                    pz=result["Pz"][particle_idx],
                    particle_mass=particle_mass,
                    scalar_potential_contribution=scalar_potential_contribution,
                    field_x=accumulated_field_x,
                    field_y=accumulated_field_y,
                    field_z=accumulated_field_z,
                )
            )
            Pt_before_projection = np.float64(result["Pt"][particle_idx])

            gamma_mass_shell = kinetic_pt_from_mass_shell / (particle_mass * C_MMNS)

            spatial_momentum_authoritative = exact_charge_source_selected
            if radiation_mode == "medina_lad" and not exact_charge_source_selected:
                # The maintained non-exact Medina path historically treats
                # canonical temporal energy as authoritative.  Put its spatial
                # mechanical momentum on that shell *before* predictor drift,
                # beta, and force sampling.  The exact charge/RFS path instead
                # remains spatial-momentum-authoritative so its RR-off and
                # RR-on capture controls share one boundary.
                energy_boundary_momentum = np.asarray(
                    _mechanical_momentum_components(
                        px=result["Px"][particle_idx],
                        py=result["Py"][particle_idx],
                        pz=result["Pz"][particle_idx],
                        particle_mass=particle_mass,
                        field_x=accumulated_field_x,
                        field_y=accumulated_field_y,
                        field_z=accumulated_field_z,
                    ),
                    dtype=float,
                )
                gamma_energy_boundary = float(result["gamma"][particle_idx])
                momentum_is_finite = bool(np.all(np.isfinite(energy_boundary_momentum)))
                mechanical_magnitude = float(
                    np.hypot(
                        np.hypot(
                            abs(float(energy_boundary_momentum[0])),
                            abs(float(energy_boundary_momentum[1])),
                        ),
                        abs(float(energy_boundary_momentum[2])),
                    )
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    target_factor = float(
                        np.sqrt(
                            (gamma_energy_boundary - 1.0)
                            * (gamma_energy_boundary + 1.0)
                        )
                    )
                    target_mechanical_magnitude = float(
                        particle_mass * C_MMNS * target_factor
                    )
                valid_rest_boundary = bool(
                    gamma_energy_boundary == 1.0 and mechanical_magnitude == 0.0
                )
                valid_moving_boundary = bool(
                    gamma_energy_boundary > 1.0
                    and mechanical_magnitude > 0.0
                    and target_mechanical_magnitude > 0.0
                )
                can_use_energy_boundary = bool(
                    np.isfinite(gamma_energy_boundary)
                    and momentum_is_finite
                    and np.isfinite(mechanical_magnitude)
                    and np.isfinite(target_mechanical_magnitude)
                    and (valid_rest_boundary or valid_moving_boundary)
                )
                if can_use_energy_boundary and valid_moving_boundary:
                    scale = target_mechanical_magnitude / mechanical_magnitude
                    scaled_momentum = energy_boundary_momentum * scale
                    if not np.isfinite(scale) or not np.all(
                        np.isfinite(scaled_momentum)
                    ):
                        can_use_energy_boundary = False
                    else:
                        result["Px"][particle_idx] = float(
                            scaled_momentum[0] + accumulated_field_x * particle_mass
                        )
                        result["Py"][particle_idx] = float(
                            scaled_momentum[1] + accumulated_field_y * particle_mass
                        )
                        result["Pz"][particle_idx] = float(
                            scaled_momentum[2] + accumulated_field_z * particle_mass
                        )
                        gamma_mass_shell = gamma_energy_boundary
                if not can_use_energy_boundary:
                    # At the near-rest/roundoff boundary, an energy-derived
                    # target can be imaginary, zero for nonzero p, directionless
                    # for p=0, or nonfinite.  Falling back to finite spatial p
                    # prevents the old sqrt(max(gamma**2-1, 0)) zeroing bug.
                    spatial_momentum_authoritative = True

            if spatial_momentum_authoritative:
                # Reconstruct the complete physical state from spatial
                # p = P - q A / c.  Near rest this avoids cancellation in
                # Pt - q Phi / c, and exact RR-off/RR-on capture controls both
                # use this same projection.
                on_shell_mechanical_momentum = np.asarray(
                    _mechanical_momentum_components(
                        px=result["Px"][particle_idx],
                        py=result["Py"][particle_idx],
                        pz=result["Pz"][particle_idx],
                        particle_mass=particle_mass,
                        field_x=accumulated_field_x,
                        field_y=accumulated_field_y,
                        field_z=accumulated_field_z,
                    ),
                    dtype=float,
                )
                on_shell_mechanical_magnitude = float(
                    np.hypot(
                        np.hypot(
                            abs(float(on_shell_mechanical_momentum[0])),
                            abs(float(on_shell_mechanical_momentum[1])),
                        ),
                        abs(float(on_shell_mechanical_momentum[2])),
                    )
                )
                on_shell_kinetic_pt = float(
                    np.hypot(
                        particle_mass * C_MMNS,
                        on_shell_mechanical_magnitude,
                    )
                )
                gamma_mass_shell = on_shell_kinetic_pt / (particle_mass * C_MMNS)
                Pt_from_mass_shell = float(
                    on_shell_kinetic_pt + scalar_potential_contribution
                )
                result["Px"][particle_idx] = float(
                    on_shell_mechanical_momentum[0]
                    + accumulated_field_x * particle_mass
                )
                result["Py"][particle_idx] = float(
                    on_shell_mechanical_momentum[1]
                    + accumulated_field_y * particle_mass
                )
                result["Pz"][particle_idx] = float(
                    on_shell_mechanical_momentum[2]
                    + accumulated_field_z * particle_mass
                )
                result["Pt"][particle_idx] = Pt_from_mass_shell
                result["gamma"][particle_idx] = gamma_mass_shell
                if exact_endpoint_recomposition_selected:
                    start_mechanical_momentum = np.asarray(
                        _mechanical_momentum_components(
                            px=float(current_state["Px"][particle_idx]),
                            py=float(current_state["Py"][particle_idx]),
                            pz=float(current_state["Pz"][particle_idx]),
                            particle_mass=particle_mass,
                            field_x=accumulated_field_x,
                            field_y=accumulated_field_y,
                            field_z=accumulated_field_z,
                        ),
                        dtype=float,
                    )
                    start_kinetic_energy = _stable_kinetic_energy_native(
                        start_mechanical_momentum,
                        particle_mass,
                    )
                    end_kinetic_energy = _stable_kinetic_energy_native(
                        on_shell_mechanical_momentum,
                        particle_mass,
                    )
                    result["mass_shell_projection_energy"][particle_idx] = float(
                        end_kinetic_energy
                        - start_kinetic_energy
                        - C_MMNS * exact_mechanical_temporal_impulse
                    )
                else:
                    result["mass_shell_projection_energy"][particle_idx] = float(
                        C_MMNS
                        * (Pt_from_mass_shell - raw_canonical_pt_before_constraints)
                    )

            if sc_verbosity >= 3 and sc_enabled and sc_iteration > 0:
                # Use Pt BEFORE projection to show the actual difference
                gamma_from_conjugate_before = Pt_before_projection / (
                    particle_mass * C_MMNS
                )
                print(f"      γ_energy (Pt - q·Φ/c)/(mc) = {gamma_from_energy:.15e}")
                print(
                    f"      γ_conjugate (Pt/(mc), before projection) = {gamma_from_conjugate_before:.15e}"
                )
                print(
                    f"      γ_mass_shell (√(P²+(mc)²)/(mc)) = {gamma_mass_shell:.15e}"
                )
                print(
                    f"      Scalar potential term q·Φ/c = {scalar_potential_contribution:.15e}"
                )
                # Show the mass-shell violation
                mass_shell_violation = abs(
                    gamma_from_conjugate_before - gamma_mass_shell
                )
                print(
                    f"      Mass-shell violation |γ_conjugate - γ_mass_shell| = {mass_shell_violation:.15e}"
                )

            # Update x^0 = dt = dtau * gamma
            if second_order_exact_source_selected:
                proper_to_coordinate_dt = float(
                    0.5
                    * h
                    * (
                        float(current_state["gamma"][particle_idx])
                        + float(result["gamma"][particle_idx])
                    )
                )
            else:
                proper_to_coordinate_dt = float(h * result["gamma"][particle_idx])
            result["t"][particle_idx] = (
                current_state["t"][particle_idx] + proper_to_coordinate_dt
            )

            # ================================================================
            # STEP 5: Update spatial positions
            # ================================================================
            # Position update in proper time formulation: dx/dτ = v·γ
            # Since h = dτ = dt/γ, we have: dx = v·γ·dτ = (P/m)·h
            # where v = P/(γ·m) and γ cancels in the product v·γ = P/m
            if second_order_exact_source_selected:
                start_mechanical_momentum = (
                    float(current_state["gamma"][particle_idx])
                    * particle_mass
                    * C_MMNS
                    * np.asarray(
                        (
                            current_state["bx"][particle_idx],
                            current_state["by"][particle_idx],
                            current_state["bz"][particle_idx],
                        ),
                        dtype=float,
                    )
                )
                end_mechanical_momentum = np.asarray(
                    (
                        result["Px"][particle_idx]
                        - accumulated_field_x * particle_mass,
                        result["Py"][particle_idx]
                        - accumulated_field_y * particle_mass,
                        result["Pz"][particle_idx]
                        - accumulated_field_z * particle_mass,
                    ),
                    dtype=float,
                )
                displacement = (
                    0.5
                    * float(h)
                    * (start_mechanical_momentum + end_mechanical_momentum)
                    / particle_mass
                )
                result["x"][particle_idx] = (
                    current_state["x"][particle_idx] + displacement[0]
                )
                result["y"][particle_idx] = (
                    current_state["y"][particle_idx] + displacement[1]
                )
                result["z"][particle_idx] = (
                    current_state["z"][particle_idx] + displacement[2]
                )
            else:
                result["x"][particle_idx] = current_state["x"][particle_idx] + h / (
                    particle_mass
                ) * (result["Px"][particle_idx] - accumulated_field_x * particle_mass)
                result["y"][particle_idx] = current_state["y"][particle_idx] + h / (
                    particle_mass
                ) * (result["Py"][particle_idx] - accumulated_field_y * particle_mass)
                result["z"][particle_idx] = current_state["z"][particle_idx] + h / (
                    particle_mass
                ) * (result["Pz"][particle_idx] - accumulated_field_z * particle_mass)

            # ================================================================
            # STEP 6: Compute velocity (beta) from position changes
            # ================================================================
            position_change_x = (
                result["x"][particle_idx] - current_state["x"][particle_idx]
            )
            position_change_y = (
                result["y"][particle_idx] - current_state["y"][particle_idx]
            )
            position_change_z = (
                result["z"][particle_idx] - current_state["z"][particle_idx]
            )

            # β = Δx/(c·Δt) using the actual coordinate-time step
            # Since Δx = (P/m)·h and Δt = γ·h, we get β = P/(γ·m·c)
            coordinate_dt = (
                proper_to_coordinate_dt
                if on_shell_kinematic_boundary_selected
                else result["t"][particle_idx] - current_state["t"][particle_idx]
            )
            if coordinate_dt == 0.0:
                coordinate_dt = proper_to_coordinate_dt
            if on_shell_kinematic_boundary_selected:
                # Derive beta from the same on-shell mechanical momentum used
                # for gamma.  Recovering it from two close positions can erase
                # the motion of a nearly stationary massive particle.
                beta_denominator = (
                    result["gamma"][particle_idx] * particle_mass * C_MMNS
                )
                beta_x = (
                    result["Px"][particle_idx] - accumulated_field_x * particle_mass
                ) / beta_denominator
                beta_y = (
                    result["Py"][particle_idx] - accumulated_field_y * particle_mass
                ) / beta_denominator
                beta_z = (
                    result["Pz"][particle_idx] - accumulated_field_z * particle_mass
                ) / beta_denominator
            else:
                beta_x = position_change_x / (C_MMNS * coordinate_dt)
                beta_y = position_change_y / (C_MMNS * coordinate_dt)
                beta_z = position_change_z / (C_MMNS * coordinate_dt)

            # Enforce speed of light limit IMMEDIATELY after calculation
            beta_x_limited, beta_y_limited, beta_z_limited = _limit_beta_magnitude(
                beta_x,
                beta_y,
                beta_z,
            )

            result["bx"][particle_idx] = beta_x_limited
            result["by"][particle_idx] = beta_y_limited
            result["bz"][particle_idx] = beta_z_limited

            # Compute gamma from the (possibly limited) beta
            # This is compared against gamma_from_energy for self-consistency
            gamma_from_velocity = _calculate_gamma_from_beta(
                beta_x_limited, beta_y_limited, beta_z_limited
            )

            # Debug: Print newly computed beta on all iterations when verbosity >= 3
            if sc_verbosity >= 3:
                print(
                    f"      Newly computed β: βx={beta_x_limited:.15e}, "
                    f"βy={beta_y_limited:.15e}, βz={beta_z_limited:.15e}"
                )

            if sc_verbosity >= 3 and sc_enabled and sc_iteration > 0:
                beta_total = np.sqrt(
                    beta_x_limited**2 + beta_y_limited**2 + beta_z_limited**2
                )
                print(
                    f"      γ_velocity (from β over Δt={coordinate_dt:.15e}) = {gamma_from_velocity:.15e}, "
                    f"βtot={beta_total:.15e}"
                )

            physical_mechanical_px = (
                result["Px"][particle_idx] - accumulated_field_x * particle_mass
            )
            physical_mechanical_py = (
                result["Py"][particle_idx] - accumulated_field_y * particle_mass
            )
            physical_mechanical_pz = (
                result["Pz"][particle_idx] - accumulated_field_z * particle_mass
            )
            reconciled_gamma_for_iteration = float(result["gamma"][particle_idx])

            # ================================================================
            # GAMMA RECONCILIATION (configurable)
            # ================================================================
            # This influences the next self-consistency iteration seed only.
            # The persisted solver state remains energy/momentum based.
            if (
                sc_enabled
                and self_consistency is not None
                and self_consistency.gamma_reconciliation_method
                != GammaReconciliationMethod.DISABLED
            ):
                gamma_from_energy = result["gamma"][particle_idx]
                beta_total = np.sqrt(
                    beta_x_limited**2 + beta_y_limited**2 + beta_z_limited**2
                )

                method = self_consistency.gamma_reconciliation_method

                if method == GammaReconciliationMethod.USE_VELOCITY:
                    gamma_reconciled = gamma_from_velocity
                    alpha = 0.0
                elif method == GammaReconciliationMethod.USE_ENERGY:
                    gamma_reconciled = gamma_from_energy
                    alpha = 1.0
                elif method == GammaReconciliationMethod.FIXED_WEIGHTED:
                    alpha = self_consistency.gamma_reconciliation_fixed_weight
                    gamma_reconciled = (
                        alpha * gamma_from_energy + (1.0 - alpha) * gamma_from_velocity
                    )
                elif method == GammaReconciliationMethod.ADAPTIVE_WEIGHTED:
                    low_threshold = (
                        self_consistency.gamma_reconciliation_low_beta_threshold
                    )
                    high_threshold = (
                        self_consistency.gamma_reconciliation_high_beta_threshold
                    )
                    low_weight = self_consistency.gamma_reconciliation_low_beta_weight
                    high_weight = self_consistency.gamma_reconciliation_high_beta_weight
                    mid_weight = self_consistency.gamma_reconciliation_mid_beta_weight

                    if beta_total < low_threshold:
                        alpha = low_weight
                    elif beta_total > high_threshold:
                        alpha = high_weight
                    else:
                        alpha = mid_weight

                    gamma_reconciled = (
                        alpha * gamma_from_energy + (1.0 - alpha) * gamma_from_velocity
                    )
                else:
                    gamma_reconciled = gamma_from_energy
                    alpha = 1.0

                reconciled_gamma_for_iteration = float(gamma_reconciled)

                if sc_verbosity >= 3:
                    print(
                        f"      Gamma reconciliation ({method.name}): α={alpha:.2f}, β={beta_total:.6f}, "
                        f"γ_energy={gamma_from_energy:.6e}, γ_velocity={gamma_from_velocity:.6e}, "
                        f"γ_iter_seed={gamma_reconciled:.6e}"
                    )

            # ================================================================
            # STEP 7: Compute acceleration (beta-dot)
            # ================================================================
            beta_change_x = (
                result["bx"][particle_idx] - current_state["bx"][particle_idx]
            )
            beta_change_y = (
                result["by"][particle_idx] - current_state["by"][particle_idx]
            )
            beta_change_z = (
                result["bz"][particle_idx] - current_state["bz"][particle_idx]
            )

            # Stored bdot is dβ/d(ct), matching the historical LW force
            # expressions. Liénard power diagnostics convert it back to dβ/dt.
            time_factor = C_MMNS * coordinate_dt
            result["bdotx"][particle_idx] = beta_change_x / time_factor
            result["bdoty"][particle_idx] = beta_change_y / time_factor
            result["bdotz"][particle_idx] = beta_change_z / time_factor

            # ================================================================
            # STEP 8: Radiation diagnostics and optional reaction corrections
            # ================================================================
            beta_tuple = (
                float(result["bx"][particle_idx]),
                float(result["by"][particle_idx]),
                float(result["bz"][particle_idx]),
            )
            beta_dot_t = (
                float(result["bdotx"][particle_idx] * C_MMNS),
                float(result["bdoty"][particle_idx] * C_MMNS),
                float(result["bdotz"][particle_idx] * C_MMNS),
            )
            radiation_power = _compute_lienard_radiated_power(
                float(force_particle_charge),
                beta_tuple,
                beta_dot_t,
                float(result["gamma"][particle_idx]),
            )
            radiation_energy = radiation_power * max(float(coordinate_dt), 0.0)
            result["radiation_power"][particle_idx] = radiation_power
            result["radiation_energy"][particle_idx] = radiation_energy

            if radiation_mode == "power_matched_damping" and radiation_energy > 0.0:
                mechanical_px = (
                    result["Px"][particle_idx] - accumulated_field_x * particle_mass
                )
                mechanical_py = (
                    result["Py"][particle_idx] - accumulated_field_y * particle_mass
                )
                mechanical_pz = (
                    result["Pz"][particle_idx] - accumulated_field_z * particle_mass
                )
                (
                    damped_mechanical_momentum,
                    damped_gamma,
                    applied_radiation_energy,
                ) = _apply_power_matched_radiation_damping(
                    (
                        float(mechanical_px),
                        float(mechanical_py),
                        float(mechanical_pz),
                    ),
                    float(particle_mass),
                    float(result["gamma"][particle_idx]),
                    float(radiation_energy),
                )
                result["radiation_energy_applied"][
                    particle_idx
                ] = applied_radiation_energy
                if applied_radiation_energy > 0.0:
                    (
                        mechanical_px,
                        mechanical_py,
                        mechanical_pz,
                    ) = damped_mechanical_momentum
                    result["Px"][particle_idx] = (
                        mechanical_px + accumulated_field_x * particle_mass
                    )
                    result["Py"][particle_idx] = (
                        mechanical_py + accumulated_field_y * particle_mass
                    )
                    result["Pz"][particle_idx] = (
                        mechanical_pz + accumulated_field_z * particle_mass
                    )
                    result["gamma"][particle_idx] = damped_gamma
                    scalar_potential_contribution = (
                        _scalar_potential_momentum_contribution(
                            force_particle_charge, accumulated_scalar_potential
                        )
                    )
                    result["Pt"][particle_idx] = (
                        damped_gamma * particle_mass * C_MMNS
                        + scalar_potential_contribution
                    )

                    beta_denom = damped_gamma * particle_mass * C_MMNS
                    if beta_denom > 0.0:
                        beta_x_limited = float(mechanical_px / beta_denom)
                        beta_y_limited = float(mechanical_py / beta_denom)
                        beta_z_limited = float(mechanical_pz / beta_denom)
                        (
                            beta_x_limited,
                            beta_y_limited,
                            beta_z_limited,
                        ) = _limit_beta_magnitude(
                            beta_x_limited,
                            beta_y_limited,
                            beta_z_limited,
                        )
                        result["bx"][particle_idx] = beta_x_limited
                        result["by"][particle_idx] = beta_y_limited
                        result["bz"][particle_idx] = beta_z_limited
                        result["bdotx"][particle_idx] = (
                            beta_x_limited - current_state["bx"][particle_idx]
                        ) / time_factor
                        result["bdoty"][particle_idx] = (
                            beta_y_limited - current_state["by"][particle_idx]
                        ) / time_factor
                        result["bdotz"][particle_idx] = (
                            beta_z_limited - current_state["bz"][particle_idx]
                        ) / time_factor
            elif radiation_mode == "medina_lad":
                predictor_coordinate_dt = coordinate_dt
                # These are the already on-shell, non-RR momenta used for the
                # position and beta update.  Do not rescale them from an
                # independently accumulated gamma: at the near-rest boundary,
                # sqrt(max(gamma**2 - 1, 0)) can collapse real momentum to zero.
                mechanical_px = physical_mechanical_px
                mechanical_py = physical_mechanical_py
                mechanical_pz = physical_mechanical_pz
                previous_mechanical_px = (
                    current_state["gamma"][particle_idx]
                    * particle_mass
                    * C_MMNS
                    * current_state["bx"][particle_idx]
                )
                previous_mechanical_py = (
                    current_state["gamma"][particle_idx]
                    * particle_mass
                    * C_MMNS
                    * current_state["by"][particle_idx]
                )
                previous_mechanical_pz = (
                    current_state["gamma"][particle_idx]
                    * particle_mass
                    * C_MMNS
                    * current_state["bz"][particle_idx]
                )
                if predictor_coordinate_dt > 0.0 and force_particle_charge != 0.0:
                    external_force = (
                        float(
                            (physical_mechanical_px - previous_mechanical_px)
                            / predictor_coordinate_dt
                        ),
                        float(
                            (physical_mechanical_py - previous_mechanical_py)
                            / predictor_coordinate_dt
                        ),
                        float(
                            (physical_mechanical_pz - previous_mechanical_pz)
                            / predictor_coordinate_dt
                        ),
                    )
                    current_force_sample_time = float(
                        current_state["t"][particle_idx] + 0.5 * predictor_coordinate_dt
                    )
                    result["medina_external_force_x"][particle_idx] = external_force[0]
                    result["medina_external_force_y"][particle_idx] = external_force[1]
                    result["medina_external_force_z"][particle_idx] = external_force[2]
                    result["medina_external_force_sample_time"][
                        particle_idx
                    ] = current_force_sample_time

                    external_force_time_derivative, derivative_ready = (
                        _accepted_medina_force_derivative(
                            current_state=current_state,
                            particle_idx=particle_idx,
                            current_force=external_force,
                            current_sample_time=current_force_sample_time,
                        )
                    )
                    result["medina_force_derivative_ready"][
                        particle_idx
                    ] = derivative_ready

                    medina_beta_dot_t, _ = _derive_relativistic_kinematics_from_force(
                        external_force,
                        beta_tuple,
                        float(result["gamma"][particle_idx]),
                        float(particle_mass),
                    )
                    medina_acceleration = tuple(
                        C_MMNS * component for component in medina_beta_dot_t
                    )
                    medina_result: MedinaRadiationReactionResult = (
                        compute_medina_radiation_reaction(
                            external_force=external_force,
                            external_force_time_derivative=(
                                external_force_time_derivative
                                if derivative_ready
                                else (0.0, 0.0, 0.0)
                            ),
                            beta=beta_tuple,
                            acceleration=medina_acceleration,
                            gamma=float(result["gamma"][particle_idx]),
                            mass=float(particle_mass),
                            charge=float(force_particle_charge),
                            coordinate_dt=float(predictor_coordinate_dt),
                        )
                    )
                    # Far radiation and the instantaneous cross-field energy do
                    # not require dF_ext/dt, so they remain valid while the
                    # first accepted force sample primes the derivative.
                    result["radiation_power"][
                        particle_idx
                    ] = medina_result.far_radiated_power
                    result["radiation_energy"][
                        particle_idx
                    ] = medina_result.far_radiated_energy
                    result["medina_cross_field_energy"][
                        particle_idx
                    ] = medina_result.cross_field_energy

                    if not derivative_ready:
                        medina_impulse = (0.0, 0.0, 0.0)
                        medina_capped = False
                    else:
                        result["medina_cross_field_energy_change"][
                            particle_idx
                        ] = medina_result.cross_field_energy_change
                        medina_impulse, medina_capped = (
                            _cap_medina_radiation_reaction_impulse(
                                impulse=medina_result.radiation_reaction_impulse,
                                external_force=external_force,
                                coordinate_dt=float(predictor_coordinate_dt),
                            )
                        )
                        result["medina_impulse_capped"][particle_idx] = medina_capped
                        uncapped_impulse_norm = float(
                            np.linalg.norm(
                                np.asarray(
                                    medina_result.radiation_reaction_impulse,
                                    dtype=float,
                                )
                            )
                        )
                        applied_impulse_norm = float(
                            np.linalg.norm(np.asarray(medina_impulse, dtype=float))
                        )
                        impulse_scale = (
                            applied_impulse_norm / uncapped_impulse_norm
                            if uncapped_impulse_norm > 0.0
                            else 1.0
                        )
                        signed_reaction_work = float(
                            impulse_scale * medina_result.reaction_work
                        )
                        result["radiation_reaction_work"][
                            particle_idx
                        ] = signed_reaction_work
                        result["radiation_energy_applied"][particle_idx] = max(
                            0.0,
                            -signed_reaction_work,
                        )

                    if medina_capped and sc_verbosity >= 2:
                        print(
                            "      Medina radiation-reaction impulse capped "
                            "by numerical guard"
                        )
                    impulse_vec = np.asarray(medina_impulse, dtype=float)
                    if derivative_ready:
                        # Use the same coordinate-time interval that formed
                        # and capped the applied impulse.  The RFS helper later
                        # converts this lab three-force separately at its start
                        # and midpoint four-velocity/spin states.
                        applied_medina_force_native[:] = impulse_vec / float(
                            predictor_coordinate_dt
                        )
                    if derivative_ready and float(np.linalg.norm(impulse_vec)) > 0.0:
                        mechanical_px = float(mechanical_px + impulse_vec[0])
                        mechanical_py = float(mechanical_py + impulse_vec[1])
                        mechanical_pz = float(mechanical_pz + impulse_vec[2])
                        mechanical_p_magnitude = float(
                            np.hypot(
                                np.hypot(abs(mechanical_px), abs(mechanical_py)),
                                abs(mechanical_pz),
                            )
                        )
                        medina_kinetic_pt = float(
                            np.hypot(
                                particle_mass * C_MMNS,
                                mechanical_p_magnitude,
                            )
                        )
                        medina_gamma = medina_kinetic_pt / (particle_mass * C_MMNS)
                        result["Px"][particle_idx] = (
                            mechanical_px + accumulated_field_x * particle_mass
                        )
                        result["Py"][particle_idx] = (
                            mechanical_py + accumulated_field_y * particle_mass
                        )
                        result["Pz"][particle_idx] = (
                            mechanical_pz + accumulated_field_z * particle_mass
                        )
                        result["gamma"][particle_idx] = medina_gamma
                        # The next nonlinear trial must start from one
                        # post-kick state.  Keeping the pre-RR gamma beside the
                        # post-RR beta would make its trial four-velocity
                        # internally inconsistent.
                        reconciled_gamma_for_iteration = medina_gamma
                        scalar_potential_contribution = (
                            _scalar_potential_momentum_contribution(
                                force_particle_charge, accumulated_scalar_potential
                            )
                        )
                        result["Pt"][particle_idx] = float(
                            medina_kinetic_pt + scalar_potential_contribution
                        )
                        beta_denom = medina_gamma * particle_mass * C_MMNS
                        if beta_denom > 0.0:
                            beta_x_limited = float(mechanical_px / beta_denom)
                            beta_y_limited = float(mechanical_py / beta_denom)
                            beta_z_limited = float(mechanical_pz / beta_denom)
                            (
                                beta_x_limited,
                                beta_y_limited,
                                beta_z_limited,
                            ) = _limit_beta_magnitude(
                                beta_x_limited,
                                beta_y_limited,
                                beta_z_limited,
                            )
                            result["bx"][particle_idx] = beta_x_limited
                            result["by"][particle_idx] = beta_y_limited
                            result["bz"][particle_idx] = beta_z_limited
                            # Ordinary and RFS forces in this solver are kicks
                            # followed by an endpoint drift.  Medina is
                            # evaluated from the non-RR predictor above, then
                            # joins that same split at first order: rebuild the
                            # endpoint from the final mechanical momentum and
                            # its mass-shell gamma.  Pt is reconstructed from
                            # p and q Phi / c; there is no independent temporal
                            # radiation-reaction kick.
                            if second_order_exact_source_selected:
                                result["x"][particle_idx] = (
                                    current_state["x"][particle_idx]
                                    + 0.5
                                    * h
                                    * (previous_mechanical_px + mechanical_px)
                                    / particle_mass
                                )
                                result["y"][particle_idx] = (
                                    current_state["y"][particle_idx]
                                    + 0.5
                                    * h
                                    * (previous_mechanical_py + mechanical_py)
                                    / particle_mass
                                )
                                result["z"][particle_idx] = (
                                    current_state["z"][particle_idx]
                                    + 0.5
                                    * h
                                    * (previous_mechanical_pz + mechanical_pz)
                                    / particle_mass
                                )
                                coordinate_dt = float(
                                    0.5
                                    * h
                                    * (
                                        float(current_state["gamma"][particle_idx])
                                        + medina_gamma
                                    )
                                )
                            else:
                                result["x"][particle_idx] = (
                                    current_state["x"][particle_idx]
                                    + h * mechanical_px / particle_mass
                                )
                                result["y"][particle_idx] = (
                                    current_state["y"][particle_idx]
                                    + h * mechanical_py / particle_mass
                                )
                                result["z"][particle_idx] = (
                                    current_state["z"][particle_idx]
                                    + h * mechanical_pz / particle_mass
                                )
                                coordinate_dt = float(h * medina_gamma)
                            result["t"][particle_idx] = (
                                current_state["t"][particle_idx] + coordinate_dt
                            )
                            time_factor = C_MMNS * coordinate_dt
                            result["bdotx"][particle_idx] = (
                                beta_x_limited - current_state["bx"][particle_idx]
                            ) / time_factor
                            result["bdoty"][particle_idx] = (
                                beta_y_limited - current_state["by"][particle_idx]
                            ) / time_factor
                            result["bdotz"][particle_idx] = (
                                beta_z_limited - current_state["bz"][particle_idx]
                            ) / time_factor
                            gamma_from_velocity = medina_gamma

            # ================================================================
            # STEP 9: Update running average of beta
            # ================================================================
            previous_beta_avg = (
                current_state["beta_avg_x"][particle_idx],
                current_state["beta_avg_y"][particle_idx],
                current_state["beta_avg_z"][particle_idx],
            )
            previous_sample_count = float(current_state["beta_samples"][particle_idx])

            new_beta = (
                result["bx"][particle_idx],
                result["by"][particle_idx],
                result["bz"][particle_idx],
            )

            updated_beta_avg, updated_sample_count = _update_beta_running_average(
                previous_beta_avg,
                previous_sample_count,
                new_beta,
            )

            result["beta_samples"][particle_idx] = updated_sample_count
            result["beta_avg_x"][particle_idx] = updated_beta_avg[0]
            result["beta_avg_y"][particle_idx] = updated_beta_avg[1]
            result["beta_avg_z"][particle_idx] = updated_beta_avg[2]

            if not sc_enabled:
                break

            # ================================================================
            # STEP 10: Update working state and check convergence
            # ================================================================
            # Update working state with newly computed values for next iteration
            new_working_beta_x = result["bx"][particle_idx]
            new_working_beta_y = result["by"][particle_idx]
            new_working_beta_z = result["bz"][particle_idx]
            new_working_gamma = reconciled_gamma_for_iteration

            if sc_enabled and sc_iteration > 0:
                # Check mass-shell convergence
                (
                    converged,
                    mass_shell_error_rel,
                ) = _check_mass_shell_convergence(
                    result["Pt"][particle_idx],
                    result["Px"][particle_idx],
                    result["Py"][particle_idx],
                    result["Pz"][particle_idx],
                    particle_mass,
                    C_MMNS,
                    sc_target_ms_tolerance,
                    scalar_potential_contribution=scalar_potential_contribution,
                    field_x=accumulated_field_x,
                    field_y=accumulated_field_y,
                    field_z=accumulated_field_z,
                )

                # Set dummy gamma consistency error (not checked)
                gamma_consistency_error = 0.0

                if converged:
                    last_mass_shell_error = mass_shell_error_rel
                    if sc_verbosity > 0:
                        _print_convergence_info(
                            particle_idx,
                            sc_iteration,
                            gamma_from_velocity,
                            gamma_from_energy,
                            gamma_mass_shell,
                            mass_shell_error_rel,
                            gamma_consistency_error,
                            converged=True,
                            max_iterations=sc_max_iterations,
                            verbosity=sc_verbosity,
                            step_idx=step_idx,
                            convergence_mode=sc_convergence_mode,
                            particle_position=(
                                result["x"][particle_idx],
                                result["y"][particle_idx],
                                result["z"][particle_idx],
                            ),
                            particle_time=result["t"][particle_idx],
                        )
                    break
                elif sc_iteration == sc_max_iterations - 1:
                    last_mass_shell_error = mass_shell_error_rel
                    if sc_verbosity > 0:
                        _print_convergence_info(
                            particle_idx,
                            sc_iteration,
                            gamma_from_velocity,
                            gamma_from_energy,
                            gamma_mass_shell,
                            mass_shell_error_rel,
                            gamma_consistency_error,
                            converged=False,
                            max_iterations=sc_max_iterations,
                            verbosity=sc_verbosity,
                            step_idx=step_idx,
                            convergence_mode=sc_convergence_mode,
                            particle_position=(
                                result["x"][particle_idx],
                                result["y"][particle_idx],
                                result["z"][particle_idx],
                            ),
                            particle_time=result["t"][particle_idx],
                        )

            # Update working state for next iteration
            working_beta_x = new_working_beta_x
            working_beta_y = new_working_beta_y
            working_beta_z = new_working_beta_z
            working_gamma = new_working_gamma
            working_x = result["x"][particle_idx]
            working_y = result["y"][particle_idx]
            working_z = result["z"][particle_idx]

            # Gamma blowup detection: ALL blowups now trigger retry attempts
            # The integration runner will reduce timestep and retry, only marking
            # particle as dead after exhausting all retry attempts.
            # - Soft threshold (1e8): likely recoverable with smaller timestep
            # - Hard threshold (1e20 or NaN/Inf): less likely but still attempt recovery
            if sc_enabled:
                is_nan_or_inf = np.isnan(working_gamma) or np.isinf(working_gamma)
                gamma_soft_threshold = 1e8
                gamma_hard_threshold = 1e20

                # Check for any gamma blowup
                if is_nan_or_inf or working_gamma > gamma_soft_threshold:
                    # Skip if particle already dead (suppress redundant errors)
                    already_dead = (
                        "_dead_particles" in result
                        and result["_dead_particles"][particle_idx]
                    )
                    if already_dead:
                        break  # Exit self-consistency loop for this particle

                    # Determine if this is a hard or soft blowup (for logging/metrics)
                    is_hard = is_nan_or_inf or working_gamma > gamma_hard_threshold

                    if sc_verbosity >= 1:
                        severity = "Hard" if is_hard else "Soft"
                        print(
                            f"    [WARNING] Step {step_idx if step_idx is not None else '?'}, "
                            f"Particle {particle_idx}/{num_particles}, Iteration {sc_iteration}: "
                            f"{severity} gamma blowup (γ={working_gamma:.2e}), requesting timestep reduction"
                        )

                    # Raise exception to signal integration_runner to reduce timestep
                    # The runner will attempt recovery for ALL blowups (soft and hard)
                    raise GammaBlowupError(
                        step_idx if step_idx is not None else -1,
                        particle_idx,
                        working_gamma,
                        sc_iteration,
                        is_hard_blowup=is_hard,
                    )

        if sc_enabled and not converged:
            raise SelfConsistencyNonConvergenceError(
                step_idx if step_idx is not None else -1,
                particle_idx,
                sc_max_iterations,
                last_mass_shell_error,
            )

        if exact_endpoint_recomposition_selected:
            result["_exact_source_start_four_potential"][
                particle_idx
            ] = exact_source_start_four_potential
            result["_exact_source_endpoint_rebase_required"][particle_idx] = True

        # ================================================================
        # AFTER self-consistency loop: Apply mass-shell projection if needed
        # ================================================================
        if sc_enabled:
            # Check final mass-shell error
            Pt_64 = np.float64(result["Pt"][particle_idx])
            scalar_potential_contribution = _scalar_potential_momentum_contribution(
                force_particle_charge, accumulated_scalar_potential
            )
            (
                _final_converged,
                mass_shell_error_final,
            ) = _check_mass_shell_convergence(
                result["Pt"][particle_idx],
                result["Px"][particle_idx],
                result["Py"][particle_idx],
                result["Pz"][particle_idx],
                particle_mass,
                C_MMNS,
                sc_mass_shell_tolerance,
                scalar_potential_contribution=scalar_potential_contribution,
                field_x=accumulated_field_x,
                field_y=accumulated_field_y,
                field_z=accumulated_field_z,
            )
            kinetic_pt_from_mass_shell, Pt_from_mass_shell = (
                _canonical_pt_from_mechanical_mass_shell(
                    px=result["Px"][particle_idx],
                    py=result["Py"][particle_idx],
                    pz=result["Pz"][particle_idx],
                    particle_mass=particle_mass,
                    scalar_potential_contribution=scalar_potential_contribution,
                    field_x=accumulated_field_x,
                    field_y=accumulated_field_y,
                    field_z=accumulated_field_z,
                )
            )

            if mass_shell_error_final > sc_mass_shell_tolerance:
                if sc_verbosity >= 2:
                    print(
                        f"    ⚠️  Final mass-shell projection: Pt {Pt_64:.6e} → "
                        f"{Pt_from_mass_shell:.6e} (error was {mass_shell_error_final:.2e})"
                    )

                result["Pt"][particle_idx] = float(Pt_from_mass_shell)

                # Recalculate gamma with projected Pt
                kinetic_energy = (
                    result["Pt"][particle_idx] - scalar_potential_contribution
                )
                result["gamma"][particle_idx] = kinetic_energy / (
                    particle_mass * C_MMNS
                )
                _refresh_kinematics_from_canonical_momentum(
                    result,
                    current_state,
                    particle_idx,
                    h,
                    particle_mass,
                    accumulated_field_x,
                    accumulated_field_y,
                    accumulated_field_z,
                )

        # Spin is advanced exactly once per accepted physical step, after all
        # self-consistency iterations. Reusing the start-of-step spin avoids
        # accidentally applying one precession update per nonlinear iteration.
        if bool(
            current_state.get("magnetic_dipole_active", np.zeros(num_particles))[
                particle_idx
            ]
        ):
            diagnostic_magnetic_field_t = np.zeros(3, dtype=float)
            if external_field is not None and getattr(external_field, "enabled", False):
                diagnostic_position = (
                    float(result["x"][particle_idx]),
                    float(result["y"][particle_idx]),
                    float(result["z"][particle_idx]),
                )
                if rfs_selected:
                    from .magnetic_dipole import magnetic_field_native_to_tesla

                    _, diagnostic_b_native, _ = evaluate_external_field_native(
                        external_field,
                        position_mm=diagnostic_position,
                        time_ns=float(result["t"][particle_idx]),
                    )
                    diagnostic_magnetic_field_t = np.asarray(
                        [
                            magnetic_field_native_to_tesla(value)
                            for value in diagnostic_b_native
                        ],
                        dtype=float,
                    )
                else:
                    _, diagnostic_magnetic_field_t, _ = evaluate_external_field_si(
                        external_field,
                        position_mm=diagnostic_position,
                        time_ns=float(result["t"][particle_idx]),
                    )
            if (
                rfs_selected
                and sim_type == SimulationType.BUNCH_TO_BUNCH
                and len(trajectory_ext) > 0
            ):
                from .retarded_fields import (
                    ObserverEvent,
                    RetardedHistoryError,
                    evaluate_retarded_charge_field_native,
                )

                try:
                    diagnostic_charge_field = evaluate_retarded_charge_field_native(
                        (
                            exact_source_history
                            if exact_source_history is not None
                            else (
                                traj_ext_soa
                                if traj_ext_soa is not None
                                else trajectory_ext
                            )
                        ),
                        ObserverEvent(
                            time_ns=float(result["t"][particle_idx]),
                            position_mm=(
                                float(result["x"][particle_idx]),
                                float(result["y"][particle_idx]),
                                float(result["z"][particle_idx]),
                            ),
                        ),
                        backend=magnetic_dipole.exact_retarded_backend,
                    )
                except RetardedHistoryError:
                    diagnostic_charge_field = None
                if diagnostic_charge_field is not None:
                    from .magnetic_dipole import magnetic_field_native_to_tesla

                    diagnostic_magnetic_field_t += np.asarray(
                        [
                            magnetic_field_native_to_tesla(value)
                            for value in diagnostic_charge_field.magnetic_field_native
                        ],
                        dtype=float,
                    )
            result["local_magnetic_field_x_t"][particle_idx] = (
                diagnostic_magnetic_field_t[0]
            )
            result["local_magnetic_field_y_t"][particle_idx] = (
                diagnostic_magnetic_field_t[1]
            )
            result["local_magnetic_field_z_t"][particle_idx] = (
                diagnostic_magnetic_field_t[2]
            )
            precession_active = bool(
                current_state.get("spin_precession_active", np.zeros(num_particles))[
                    particle_idx
                ]
            )
            if precession_active:
                spin_start = np.asarray(
                    (
                        current_state["spin_x"][particle_idx],
                        current_state["spin_y"][particle_idx],
                        current_state["spin_z"][particle_idx],
                    ),
                    dtype=float,
                )
                beta_start = np.asarray(
                    (
                        current_state["bx"][particle_idx],
                        current_state["by"][particle_idx],
                        current_state["bz"][particle_idx],
                    ),
                    dtype=float,
                )
                beta_end = np.asarray(
                    (
                        result["bx"][particle_idx],
                        result["by"][particle_idx],
                        result["bz"][particle_idx],
                    ),
                    dtype=float,
                )
                if rfs_selected:
                    spin_next = _advance_rfs_rest_spin(
                        spin_start,
                        beta_start=beta_start,
                        beta_end=beta_end,
                        field_tensor=rfs_field_tensor,
                        partial_f=rfs_partial_f,
                        charge_native=float(force_particle_charge),
                        mass_amu=float(particle_mass),
                        magnetic_moment_native=float(
                            current_state["magnetic_moment_native"][particle_idx]
                        ),
                        spin_quantum_number=float(
                            current_state["spin_quantum_number"][particle_idx]
                        ),
                        proper_time_step_ns=float(h),
                        applied_radiation_reaction_force_native=(
                            applied_medina_force_native
                        ),
                        analytic_antisymmetric_response=(
                            exact_analytic_antisymmetric_response
                        ),
                        analytic_partial_antisymmetric_response=(
                            exact_analytic_partial_antisymmetric_response
                        ),
                        analytic_response_contraction=(
                            "numba_strict_serial"
                            if exact_analytic_antisymmetric_response is not None
                            else "python"
                        ),
                    )
                else:
                    from .constants import ELEMENTARY_CHARGE
                    from .external_fields import AMU_KG, ELEMENTARY_CHARGE_COULOMB
                    from .magnetic_dipole import advance_spin_uniform_fields

                    beta_midpoint = 0.5 * (beta_start + beta_end)
                    coordinate_step_s = (
                        max(
                            float(
                                result["t"][particle_idx]
                                - current_state["t"][particle_idx]
                            ),
                            0.0,
                        )
                        * 1.0e-9
                    )
                    spin_next = advance_spin_uniform_fields(
                        spin_start,
                        beta=beta_midpoint,
                        electric_field_v_m=local_electric_field_v_m,
                        magnetic_field_t=local_magnetic_field_t,
                        charge_coulomb=(
                            float(force_particle_charge)
                            / ELEMENTARY_CHARGE
                            * ELEMENTARY_CHARGE_COULOMB
                        ),
                        mass_kg=float(particle_mass) * AMU_KG,
                        gyromagnetic_ratio_rad_s_t=float(
                            current_state["gyromagnetic_ratio_rad_s_t"][particle_idx]
                        ),
                        delta_time_s=coordinate_step_s,
                    )
                result["spin_x"][particle_idx] = spin_next[0]
                result["spin_y"][particle_idx] = spin_next[1]
                result["spin_z"][particle_idx] = spin_next[2]

    # Log summary if any particles died in this step
    if particles_marked_dead_this_step > 0:
        print(
            f"  [SUMMARY] Step {step_idx if step_idx is not None else '?'}: "
            f"{particles_marked_dead_this_step}/{num_particles} particles marked dead in this step"
        )

    return result


__all__ = ["retarded_equations_of_motion"]
