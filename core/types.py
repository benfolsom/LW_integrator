"""Foundational types shared across integrator modules.

The modern LW core favours explicit type aliases and dataclasses so that both
runtime code and documentation stay readable.  Keeping the definitions in a
single module ensures a consistent contract between physics routines, tests,
and example notebooks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from enum import Enum, IntEnum, auto
from itertools import count
from typing import Dict, List, Sequence, cast

import numpy as np

from .constants import C_MMNS

ParticleState = Dict[str, np.ndarray]
Trajectory = List[ParticleState]
TrajectoryView = Sequence[ParticleState]


_TRAJECTORY_STORAGE_TOKENS = count(1)


@dataclass
class _TrajectoryStorageState:
    """Live mutation metadata shared by every view of one builder's arrays.

    The monotonic token identifies the allocation without relying on the
    identity of a short-lived :class:`TrajectoryArrays` wrapper. ``generation``
    advances for every builder write. ``rewrite_epoch`` advances whenever a
    row already exposed by :meth:`TrajectoryBuilder.build_partial` is changed,
    so append-aware consumers can distinguish a safe tail extension from a
    history rewrite.
    """

    token: int = field(default_factory=lambda: next(_TRAJECTORY_STORAGE_TOKENS))
    capacity: int = 0
    generation: int = 0
    rewrite_epoch: int = 0
    array_revision: int = 0

    def __deepcopy__(self, memo: dict[int, object]) -> "_TrajectoryStorageState":
        """Copy one storage graph while minting a collision-free token."""

        copied = _TrajectoryStorageState(
            capacity=self.capacity,
            generation=self.generation,
            rewrite_epoch=self.rewrite_epoch,
            array_revision=self.array_revision,
        )
        memo[id(self)] = copied
        return copied

    def __reduce__(self) -> tuple[object, tuple[int, int, int, int]]:
        """Never persist a process-local allocation token through pickle."""

        return (
            _restore_trajectory_storage_state,
            (
                self.capacity,
                self.generation,
                self.rewrite_epoch,
                self.array_revision,
            ),
        )


def _restore_trajectory_storage_state(
    capacity: int,
    generation: int,
    rewrite_epoch: int,
    array_revision: int,
) -> _TrajectoryStorageState:
    return _TrajectoryStorageState(
        capacity=capacity,
        generation=generation,
        rewrite_epoch=rewrite_epoch,
        array_revision=array_revision,
    )


class StaleTrajectoryViewError(RuntimeError):
    """Raised when a trajectory wrapper no longer names current backing arrays."""


class SimulationType(IntEnum):
    """Supported simulation modes.

    The enum inherits from :class:`int` so that existing code using literal
    values (``0``, ``1``, ``2``) continues to work.  When writing new code,
    prefer the descriptive enum members for readability.
    """

    CONDUCTING_WALL = 0
    SWITCHING_WALL = 1
    BUNCH_TO_BUNCH = 2


class ChronoMatchingMode(Enum):
    """Retardation sampling strategies used by chrono-matching.

    ``FAST`` (default) reproduces the historical implementation by evaluating
    the causal delay once using the instantaneous dot product of particle
    velocity and the line-of-sight unit vector (``Δt = R (1 + β·n̂) / c``).

    ``AVERAGED`` is reserved for internal use with ``APPROXIMATE_BACK_HISTORY``
    startup mode. It samples two limiting cases—first assuming the source
    particle is stationary (``R / c``) and then assuming it moves at the speed
    of light in the line-of-sight direction (``2R / c``). The averaged dot
    product from those two samples is used to compute the retardation interval.
    This mode should NOT be used in production until APPROXIMATE_BACK_HISTORY
    is fully implemented and validated.
    """

    FAST = auto()
    AVERAGED = auto()


class StartupMode(Enum):
    """Strategies for handling the lack of retarded history at early steps.

    ``COLD_START`` suppresses external forces until the observer has travelled
    far enough for the light-cone constraint to be satisfied using a running
    average of the observer velocity. ``APPROXIMATE_BACK_HISTORY`` assumes the
    source velocity remains constant between steps, enabling an analytic
    back-fill of the retarded separation. ``INERTIAL_PREHISTORY`` supplies a
    finite, synthetic constant-velocity source history before active time zero,
    allowing exact retarded light-cone solves without exposing the prefix as
    part of the requested trajectory.
    """

    COLD_START = auto()
    APPROXIMATE_BACK_HISTORY = auto()
    INERTIAL_PREHISTORY = auto()


class GammaReconciliationMethod(Enum):
    """Methods for reconciling dual gamma calculations (from energy vs velocity).

    The integrator computes gamma in two ways:
    - γ_energy from conjugate momentum: γ = (Pt - q·Φ/c)/(mc)
    - γ_velocity from velocity: γ = 1/√(1-β²)

    These should be identical in exact math but differ numerically due to
    discretization, potentially causing energy jumps and instabilities.

    Reconciliation methods:

    ``DISABLED`` - No reconciliation; use γ_energy directly.
        May cause dual-gamma inconsistency and energy blowups.

    ``ADAPTIVE_WEIGHTED`` - Weighted average with velocity-dependent α.
        α = 0.8 (trust energy) for β < 0.9
        α = 0.2 (trust velocity) for β > 0.99
        α = 0.5 (balanced) for 0.9 ≤ β ≤ 0.99
        Provides smooth transition across velocity regimes, but remains an
        opt-in stabilisation option rather than the default.

    ``USE_VELOCITY`` - Always use γ_velocity (γ = 1/√(1-β²)).
        Geometrically consistent but can break energy bookkeeping.
        Not recommended for production.

    ``USE_ENERGY`` - Always use γ_energy (γ = (Pt - q·Φ/c)/(mc)).
        Same as DISABLED; provided for symmetry/clarity.

    ``FIXED_WEIGHTED`` - Fixed 50/50 weighted average.
        γ = 0.5·γ_energy + 0.5·γ_velocity
        Simple but doesn't adapt to physics regime.

    The default is ``DISABLED``. Weighted reconciliation methods are kept for
    diagnostics and legacy studies; production runs should prefer direct
    energy bookkeeping unless a sweep explicitly validates the stabilisation.
    """

    DISABLED = auto()
    ADAPTIVE_WEIGHTED = auto()
    USE_VELOCITY = auto()
    USE_ENERGY = auto()
    FIXED_WEIGHTED = auto()


@dataclass
class ParticleLossConfig:
    """Configuration for fixed-size physical particle-loss tracking."""

    enabled: bool = True
    loss_radius_mm: float | None = 500.0
    conducting_wall_aperture_loss_enabled: bool = True
    initial_radial_quantile: float | None = None
    initial_radial_multiplier: float = 1.0
    initial_radial_margin_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.loss_radius_mm is not None and self.loss_radius_mm <= 0.0:
            raise ValueError("particle_loss loss_radius_mm must be positive")
        if self.initial_radial_quantile is not None and not (
            0.0 < self.initial_radial_quantile <= 1.0
        ):
            raise ValueError("particle_loss initial_radial_quantile must be in (0, 1]")
        if self.initial_radial_multiplier <= 0.0:
            raise ValueError("particle_loss initial_radial_multiplier must be positive")
        if self.initial_radial_margin_mm < 0.0:
            raise ValueError("particle_loss initial_radial_margin_mm must be >= 0")


@dataclass
class MacroparticleSmearingConfig:
    """Configuration for bounded macroparticle source smearing."""

    enabled: bool = False
    mode: str = "deterministic_subcharge"
    subcharge_count: int = 8
    sigma_multiplier: float = 1.0
    position_sigma_mm: float | None = None
    longitudinal_sigma_mm: float | None = None
    momentum_sigma_amu_mm_ns: float | None = None
    use_position_errors: bool = True
    use_momentum_errors: bool = True
    use_centroid_errors: bool = True
    use_internal_cloud: bool = True
    apply_to_active_observers: bool = True
    apply_to_active_sources: bool = True
    apply_to_passive_sources: bool = True
    apply_to_passive_updates: bool = False
    seed: int = 12345
    refresh_policy: str = "fixed_per_particle"

    def __post_init__(self) -> None:
        if self.mode != "deterministic_subcharge":
            raise ValueError(
                "macroparticle smearing mode must be deterministic_subcharge"
            )
        if self.subcharge_count <= 0:
            raise ValueError("macroparticle smearing subcharge_count must be positive")
        if self.subcharge_count > 128:
            raise ValueError("macroparticle smearing subcharge_count must be <= 128")
        if self.sigma_multiplier < 0.0:
            raise ValueError(
                "macroparticle smearing sigma_multiplier must be non-negative"
            )
        for name, value in (
            ("position_sigma_mm", self.position_sigma_mm),
            ("longitudinal_sigma_mm", self.longitudinal_sigma_mm),
            ("momentum_sigma_amu_mm_ns", self.momentum_sigma_amu_mm_ns),
        ):
            if value is not None and value < 0.0:
                raise ValueError(f"macroparticle smearing {name} must be non-negative")
        if self.refresh_policy not in {"fixed_per_particle", "per_step"}:
            raise ValueError(
                "macroparticle smearing refresh_policy must be fixed_per_particle or per_step"
            )
        if self.apply_to_passive_updates:
            raise ValueError(
                "macroparticle smearing apply_to_passive_updates is not implemented yet"
            )


@dataclass
class PseudoGridConfig:
    """Configuration surface for the experimental pseudo-grid solver mode.

    The first implementation targets ``BUNCH_TO_BUNCH`` studies where only a
    small active subset performs full retarded LW solves while the remaining
    particles are advanced by neighborhood-weighted updates.

    The configuration is threaded through the public API before the reduced
    solver path is fully implemented so the CLI, GUI, saved configs, and tests
    can evolve in lockstep.
    """

    enabled: bool = False
    active_rider_count: int = 4
    active_driver_count: int = 4
    field_rider_count: int = 0
    field_driver_count: int = 0
    field_deposition_neighbor_count: int = 4
    space_charge_near_neighbor_count: int = 8
    passive_neighbor_count: int = 4
    coverage_strategy: str = "farthest_point_staleness"
    coverage_space: str = "position"
    active_selection_mode: str = "rotating_live"
    passive_update_mode: str = "weighted_delta"
    active_rotation_interval: int = 20
    active_rotation_fraction: float = 0.25
    passive_remap_mode: str = "none"
    passive_remap_warning_sigma: float = 0.5
    passive_remap_trigger_sigma: float = 1.0
    pair_reuse_window: int = 16
    source_weighting_mode: str = "inverse_distance"
    loss_tracking_enabled: bool = True
    numerical_failure_tolerance_fraction: float = 0.001
    causal_history_pruning_enabled: bool = False
    causal_history_safety_margin_steps: int = 2

    def __post_init__(self) -> None:
        if self.active_rider_count <= 0:
            raise ValueError("pseudo-grid active_rider_count must be positive")
        if self.active_driver_count <= 0:
            raise ValueError("pseudo-grid active_driver_count must be positive")
        if self.field_rider_count < 0:
            raise ValueError("pseudo-grid field_rider_count must be non-negative")
        if self.field_driver_count < 0:
            raise ValueError("pseudo-grid field_driver_count must be non-negative")
        if self.field_deposition_neighbor_count <= 0:
            raise ValueError(
                "pseudo-grid field_deposition_neighbor_count must be positive"
            )
        if self.space_charge_near_neighbor_count < 0:
            raise ValueError(
                "pseudo-grid space_charge_near_neighbor_count must be non-negative"
            )
        if self.passive_neighbor_count <= 0:
            raise ValueError("pseudo-grid passive_neighbor_count must be positive")
        if self.active_selection_mode not in {
            "rotating_live",
            "slow_rotating_live",
            "fixed_prefix",
        }:
            raise ValueError(
                "pseudo-grid active_selection_mode must be rotating_live, "
                "slow_rotating_live, or fixed_prefix"
            )
        if self.passive_update_mode not in {
            "weighted_delta",
            "ballistic",
            "external_interbunch",
            "frozen",
        }:
            raise ValueError(
                "pseudo-grid passive_update_mode must be weighted_delta, ballistic, "
                "external_interbunch, or frozen"
            )
        if self.active_rotation_interval <= 0:
            raise ValueError("pseudo-grid active_rotation_interval must be positive")
        if not (0.0 < self.active_rotation_fraction <= 1.0):
            raise ValueError("pseudo-grid active_rotation_fraction must be in (0, 1]")
        if self.passive_remap_mode != "none":
            raise ValueError(
                "pseudo-grid passive_remap_mode currently supports only none"
            )
        if self.passive_remap_warning_sigma < 0.0:
            raise ValueError(
                "pseudo-grid passive_remap_warning_sigma must be non-negative"
            )
        if self.passive_remap_trigger_sigma < self.passive_remap_warning_sigma:
            raise ValueError(
                "pseudo-grid passive_remap_trigger_sigma must be >= warning threshold"
            )
        if self.pair_reuse_window < 0:
            raise ValueError("pseudo-grid pair_reuse_window must be non-negative")
        if not (0.0 <= self.numerical_failure_tolerance_fraction <= 1.0):
            raise ValueError(
                "pseudo-grid numerical_failure_tolerance_fraction must be in [0, 1]"
            )
        if self.causal_history_safety_margin_steps < 0:
            raise ValueError(
                "pseudo-grid causal_history_safety_margin_steps must be non-negative"
            )


@dataclass
class DriverTrainConfig:
    """Configuration for flat BUNCH_TO_BUNCH driver-train sources."""

    enabled: bool = False
    bunch_count: int = 1
    z_spacing_mm: float = 0.0
    z_offsets_mm: tuple[float, ...] = field(default_factory=tuple)
    prehistory_steps: int = 0
    preserve_prehistory_in_output: bool = False

    def __post_init__(self) -> None:
        self.bunch_count = int(self.bunch_count)
        self.z_spacing_mm = float(self.z_spacing_mm)
        self.prehistory_steps = int(self.prehistory_steps)
        self.z_offsets_mm = tuple(float(value) for value in self.z_offsets_mm)
        if self.bunch_count < 1:
            raise ValueError("driver_train bunch_count must be at least 1")
        if self.prehistory_steps < 0:
            raise ValueError("driver_train prehistory_steps must be non-negative")
        if self.z_offsets_mm and len(self.z_offsets_mm) != self.bunch_count:
            raise ValueError("driver_train z_offsets_mm length must match bunch_count")

    def resolved_z_offsets_mm(self) -> tuple[float, ...]:
        if self.z_offsets_mm:
            return self.z_offsets_mm
        return tuple(index * self.z_spacing_mm for index in range(self.bunch_count))


@dataclass
class CavityExitConfig:
    """Configuration for BUNCH_TO_BUNCH cavity-exit early termination."""

    enabled: bool = False
    mode: str = "first_exit"
    cavity_length_mm: float | None = None
    residual_tail_factor: float = 0.0
    max_residual_tail_steps: int = 0

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.mode = str(self.mode)
        allowed_modes = {"first_exit", "rider_exit_with_driver_tail"}
        if self.mode not in allowed_modes:
            raise ValueError(
                "cavity_exit mode must be 'first_exit' or "
                "'rider_exit_with_driver_tail'"
            )
        if self.cavity_length_mm is not None:
            self.cavity_length_mm = float(self.cavity_length_mm)
            if self.cavity_length_mm <= 0:
                raise ValueError("cavity_exit cavity_length_mm must be positive")
        self.residual_tail_factor = float(self.residual_tail_factor)
        self.max_residual_tail_steps = int(self.max_residual_tail_steps)
        if self.residual_tail_factor < 0:
            raise ValueError("cavity_exit residual_tail_factor must be non-negative")
        if self.max_residual_tail_steps < 0:
            raise ValueError("cavity_exit max_residual_tail_steps must be non-negative")


@dataclass
class Occluder:
    """A single beam-pipe-like line-of-sight occluder.

    The occluder is a finite cylinder of radius ``radius_mm`` centered at
    ``center_mm`` with its axis along ``axis`` (a unit vector). A source
    particle is "inside" (has line of sight down the axis) when its
    transverse distance from the axis is less than ``radius_mm`` and it is
    within the cylinder's axial extent. The cylinder is open at both ends.
    """

    axis: tuple[float, float, float]
    center_mm: tuple[float, float, float]
    radius_mm: float
    length_mm: float
    label: str = ""

    def __post_init__(self) -> None:
        axis_arr = np.asarray(self.axis, dtype=float)
        norm = float(np.linalg.norm(axis_arr))
        if norm < 1e-15:
            raise ValueError("Occluder axis must be non-zero")
        normalized_axis = axis_arr / norm
        self.axis = (
            float(normalized_axis[0]),
            float(normalized_axis[1]),
            float(normalized_axis[2]),
        )
        center = np.asarray(self.center_mm, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("Occluder center_mm must contain three finite values")
        self.center_mm = (float(center[0]), float(center[1]), float(center[2]))
        if self.radius_mm <= 0:
            raise ValueError("Occluder radius_mm must be positive")
        if self.length_mm <= 0:
            raise ValueError("Occluder length_mm must be positive")
        self.label = str(self.label)


@dataclass
class BeamlineGeometryConfig:
    """Configuration for geometry-based line-of-sight screening.

    When enabled, retarded field contributions between bunch particles are
    zeroed when the source particle (at its retarded position) is outside
    all occluders that bound line of sight to the other bunch.

    Each occluder represents an open pipe. A source particle inside an
    occluder's transverse aperture has line of sight down that pipe's axis.
    The occlusion test is applied at the retarded source position so that
    residual fields (emitted earlier while inside) still arrive after the
    source exits.
    """

    enabled: bool = False
    occluders: list[Occluder] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)


@dataclass
class MagneticDipoleParticleConfig:
    """Magnetic-moment and initial-polarization settings for one bunch.

    ``magnetic_moment_j_per_t`` is signed with respect to the configured spin
    direction.  A value of ``None`` selects the cited value from ``species``.
    The three-vector is a unit rest-frame spin direction expressed in the lab
    coordinate axes. ``polarization`` is retained for legacy diagnostics, but
    the coupled RFS model accepts only 0 or 1: partial polarization must be an
    ensemble of full-magnitude spin orientations rather than a shrunken
    individual spin.
    """

    species: str = "custom"
    magnetic_moment_j_per_t: float | None = None
    spin_quantum_number: float | None = None
    rest_spin: tuple[float, float, float] = (0.0, 0.0, 1.0)
    polarization: float = 1.0

    def __post_init__(self) -> None:
        self.species = str(self.species).strip().lower()
        if not self.species:
            raise ValueError("magnetic-dipole species must not be empty")
        if self.magnetic_moment_j_per_t is not None:
            self.magnetic_moment_j_per_t = float(self.magnetic_moment_j_per_t)
            if not np.isfinite(self.magnetic_moment_j_per_t):
                raise ValueError("magnetic moment must be finite")
        if self.spin_quantum_number is not None:
            self.spin_quantum_number = float(self.spin_quantum_number)
            if (
                not np.isfinite(self.spin_quantum_number)
                or self.spin_quantum_number <= 0.0
            ):
                raise ValueError("spin quantum number must be finite and positive")
        spin = np.asarray(self.rest_spin, dtype=float)
        if spin.shape != (3,) or not np.all(np.isfinite(spin)):
            raise ValueError("rest_spin must contain three finite values")
        norm = float(np.linalg.norm(spin))
        if norm <= 0.0:
            raise ValueError("rest_spin must be non-zero")
        normalized_spin = spin / norm
        self.rest_spin = (
            float(normalized_spin[0]),
            float(normalized_spin[1]),
            float(normalized_spin[2]),
        )
        self.polarization = float(self.polarization)
        if not 0.0 <= self.polarization <= 1.0:
            raise ValueError("polarization must be in [0, 1]")


@dataclass
class DipoleSourceConfig:
    """Ordinary Maxwell field sourced by an intrinsic magnetic moment.

    ``covariant_retarded_point`` evaluates all near, induction, and radiation
    zones from a conserved covariant point-dipole current.  The strict
    ``minimum_separation_mm`` is an abort boundary, not softening or a
    finite-size model.  The default 2 pm boundary is about 5.2 electron
    reduced-Compton wavelengths and remains below the first 10 pm capture
    benchmark.

    Stencil and light-cone controls are advanced convergence settings.  The
    normal user-facing choice is only ``model``; validation studies should
    rerun with half and twice ``relative_stencil_step``.
    """

    model: str = "off"
    minimum_separation_mm: float = 2.0e-9
    relative_stencil_step: float = 1.0e-3
    minimum_stencil_step_mm: float = 1.0e-15
    root_tolerance_mm: float = 1.0e-21
    max_root_iterations: int = 96

    def __post_init__(self) -> None:
        self.model = str(self.model).strip().lower().replace("-", "_")
        aliases = {
            "none": "off",
            "disabled": "off",
            "retarded_point": "covariant_retarded_point",
            "full_retarded_point": "covariant_retarded_point",
        }
        self.model = aliases.get(self.model, self.model)
        if self.model not in {"off", "covariant_retarded_point"}:
            raise ValueError(
                "dipole source model must be one of: off, " "covariant_retarded_point"
            )
        for name in (
            "minimum_separation_mm",
            "relative_stencil_step",
            "minimum_stencil_step_mm",
            "root_tolerance_mm",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"dipole source {name} must be finite and positive")
            setattr(self, name, value)
        if self.relative_stencil_step >= 0.05:
            raise ValueError("dipole source relative_stencil_step must be below 0.05")
        self.max_root_iterations = int(self.max_root_iterations)
        if self.max_root_iterations <= 0:
            raise ValueError("dipole source max_root_iterations must be positive")

    @property
    def active(self) -> bool:
        """Whether an intrinsic-dipole Maxwell source model is selected."""

        return self.model != "off"


@dataclass
class MagneticDipoleConfig:
    """Configuration for experimental intrinsic magnetic-moment dynamics.

    Spin transport and Stern--Gerlach translation are deliberately separate.
    ``rfs_minimal_2021`` with ``rfs_full_g`` is the selected covariant model.
    The ``bmt_frenkel`` and ``static_rest_gradient`` pair remains available as
    a legacy diagnostic; the latter consumes only an explicit prescribed
    magnetic-field gradient and is not a relativistic Stern--Gerlach law.
    """

    enabled: bool = False
    spin_precession_enabled: bool = True
    stern_gerlach_force_enabled: bool = False
    spin_model: str = "rfs_minimal_2021"
    stern_gerlach_model: str = "rfs_full_g"
    exact_retarded_backend: str = "python"
    exact_retarded_update: str = "first_order_endpoint"
    intrinsic_spin_self_reaction_mode: str = "off"
    source: DipoleSourceConfig = field(default_factory=DipoleSourceConfig)
    rider: MagneticDipoleParticleConfig = field(
        default_factory=lambda: MagneticDipoleParticleConfig(species="electron")
    )
    driver: MagneticDipoleParticleConfig = field(
        default_factory=lambda: MagneticDipoleParticleConfig(species="proton")
    )

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.spin_precession_enabled = bool(self.spin_precession_enabled)
        self.stern_gerlach_force_enabled = bool(self.stern_gerlach_force_enabled)
        self.spin_model = str(self.spin_model).strip().lower()
        self.stern_gerlach_model = str(self.stern_gerlach_model).strip().lower()
        self.exact_retarded_backend = str(self.exact_retarded_backend).strip().lower()
        self.exact_retarded_update = (
            str(self.exact_retarded_update).strip().lower().replace("-", "_")
        )
        self.intrinsic_spin_self_reaction_mode = (
            str(self.intrinsic_spin_self_reaction_mode)
            .strip()
            .lower()
            .replace("-", "_")
        )
        update_aliases = {
            "first_order": "first_order_endpoint",
            "second_order": "second_order_start_taylor_endpoint",
            "second_order_taylor": "second_order_start_taylor_endpoint",
            "second_order_taylor_endpoint": ("second_order_start_taylor_endpoint"),
            "second_order_start_taylor": ("second_order_start_taylor_endpoint"),
        }
        self.exact_retarded_update = update_aliases.get(
            self.exact_retarded_update,
            self.exact_retarded_update,
        )
        valid_spin_models = {"bmt_frenkel", "rfs_minimal_2021"}
        if self.spin_model not in valid_spin_models:
            raise ValueError(
                "magnetic-dipole spin_model must be one of: "
                "bmt_frenkel, rfs_minimal_2021"
            )
        required_spin_model = {
            "rfs_full_g": "rfs_minimal_2021",
            "static_rest_gradient": "bmt_frenkel",
        }.get(self.stern_gerlach_model)
        if required_spin_model is None:
            raise ValueError(
                "magnetic-dipole stern_gerlach_model must be one of: "
                "rfs_full_g, static_rest_gradient"
            )
        if self.spin_model != required_spin_model:
            raise ValueError(
                "magnetic-dipole stern_gerlach_model "
                f"'{self.stern_gerlach_model}' requires spin_model "
                f"'{required_spin_model}'"
            )
        if self.exact_retarded_backend not in {
            "python",
            "numba_roots_exact_serial",
            "numba_full_strict_serial",
            "numba_analytic_charge_response_serial",
            "numba_analytic_charge_dipole_response_serial",
            "metal_certified_full_strict",
        }:
            raise ValueError(
                "magnetic-dipole exact_retarded_backend must be one of: python, "
                "numba_roots_exact_serial, numba_full_strict_serial, "
                "numba_analytic_charge_response_serial, "
                "numba_analytic_charge_dipole_response_serial, "
                "metal_certified_full_strict"
            )
        if self.exact_retarded_update not in {
            "first_order_endpoint",
            "second_order_start_taylor_endpoint",
        }:
            raise ValueError(
                "magnetic-dipole exact_retarded_update must be one of: "
                "first_order_endpoint, second_order_start_taylor_endpoint"
            )
        if self.intrinsic_spin_self_reaction_mode not in {"off", "diagnostic"}:
            raise ValueError(
                "magnetic-dipole intrinsic_spin_self_reaction_mode must be one "
                "of: off, diagnostic"
            )
        if (
            self.intrinsic_spin_self_reaction_mode == "diagnostic"
            and self.exact_retarded_update != "second_order_start_taylor_endpoint"
        ):
            raise ValueError(
                "intrinsic-spin self-reaction diagnostics require "
                "second_order_start_taylor_endpoint"
            )
        if isinstance(self.source, dict):
            self.source = DipoleSourceConfig(**self.source)
        if isinstance(self.rider, dict):
            self.rider = MagneticDipoleParticleConfig(**self.rider)
        if isinstance(self.driver, dict):
            self.driver = MagneticDipoleParticleConfig(**self.driver)


@dataclass
class AdaptivePairReturnConfig:
    """Checkpointable shared-lab-time integration for one exact particle pair.

    The mode is deliberately narrow: one rider and one driver, exact inertial
    prehistory, causal-frozen spin history, step doubling, and joint pair
    commits. ``time_step`` remains the initial proper-time guess; the factors
    below bound the adaptive shared-lab-time slab relative to that value.
    """

    enabled: bool = False
    target_lab_time_ns: float | None = None
    tolerance_scale: float = 1.0
    minimum_step_factor: float = 1.0 / 64.0
    maximum_step_factor: float = 64.0
    public_sample_interval_ns: float | None = None
    shared_time_absolute_tolerance_ns: float = 1.0e-20
    shared_time_relative_tolerance: float = 1.0e-12
    maximum_attempts: int = 2_000_000
    maximum_accepted_slabs: int = 1_000_000

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        if self.target_lab_time_ns is not None:
            self.target_lab_time_ns = float(self.target_lab_time_ns)
        self.tolerance_scale = float(self.tolerance_scale)
        self.minimum_step_factor = float(self.minimum_step_factor)
        self.maximum_step_factor = float(self.maximum_step_factor)
        if self.public_sample_interval_ns is not None:
            self.public_sample_interval_ns = float(self.public_sample_interval_ns)
        self.shared_time_absolute_tolerance_ns = float(
            self.shared_time_absolute_tolerance_ns
        )
        self.shared_time_relative_tolerance = float(self.shared_time_relative_tolerance)
        self.maximum_attempts = int(self.maximum_attempts)
        self.maximum_accepted_slabs = int(self.maximum_accepted_slabs)

        positive = (
            ("tolerance_scale", self.tolerance_scale),
            ("minimum_step_factor", self.minimum_step_factor),
            ("maximum_step_factor", self.maximum_step_factor),
        )
        if any(not np.isfinite(value) or value <= 0.0 for _, value in positive):
            raise ValueError(
                "adaptive-pair tolerance and step factors must be finite and positive"
            )
        if self.maximum_step_factor < self.minimum_step_factor:
            raise ValueError(
                "adaptive-pair maximum_step_factor must not be below the minimum"
            )
        if self.enabled and (
            self.target_lab_time_ns is None
            or not np.isfinite(self.target_lab_time_ns)
            or self.target_lab_time_ns <= 0.0
        ):
            raise ValueError(
                "adaptive-pair target_lab_time_ns is required and must be positive"
            )
        if self.public_sample_interval_ns is not None and (
            not np.isfinite(self.public_sample_interval_ns)
            or self.public_sample_interval_ns <= 0.0
        ):
            raise ValueError("adaptive-pair public_sample_interval_ns must be positive")
        time_tolerances = (
            self.shared_time_absolute_tolerance_ns,
            self.shared_time_relative_tolerance,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in time_tolerances):
            raise ValueError(
                "adaptive-pair shared-time tolerances must be finite and non-negative"
            )
        if not any(value > 0.0 for value in time_tolerances):
            raise ValueError(
                "adaptive-pair needs a positive absolute or relative time tolerance"
            )
        if self.maximum_attempts < 1 or self.maximum_accepted_slabs < 1:
            raise ValueError("adaptive-pair run limits must be positive")


@dataclass
class IntegratorConfig:
    """Structured configuration for :func:`core.integration_runner.run_integrator`.

    Attributes
    ----------
    steps:
        Total number of integration iterations to perform.
    time_step:
        Temporal spacing between successive states (``h`` in the literature).
    wall_position:
        Reference position of the conducting wall in millimetres.
    aperture_radius:
        Radius of the conducting aperture in millimetres.
    simulation_type:
        Boundary condition / interaction model, expressed as
        :class:`SimulationType`.
    chrono_mode:
        Retardation-matching strategy expressed as :class:`ChronoMatchingMode`.
    startup_mode:
        Strategy for handling the initial lack of retarded history expressed as
        :class:`StartupMode`.
    bunch_mean:
        Optional mean bunch separation used by archived notebooks. Not every
        integration path consumes it, but the value is retained for API
        compatibility.
    cavity_spacing:
        Distance between cavities used by the switching-wall configuration.
    z_cutoff:
        Longitudinal position at which the switching-wall stops mirroring
        charges. For BUNCH_TO_BUNCH mode with z_cutoff_mode='relative',
        this is the distance from starting position. Defaults to ``0``
        which effectively disables the cutoff.
    z_cutoff_mode:
        Interpretation of z_cutoff parameter. 'absolute' (default) uses
        z_cutoff as absolute z position. 'relative' uses z_cutoff as
        distance from starting position (for BUNCH_TO_BUNCH simulations).
    image_subcharge_count:
        Number of virtual subcharges used when constructing conducting-wall
        image charges. Must lie between 4 and 128. Defaults to ``12``.
    use_image_weighting:
        Enables radial weighting when distributing conducting-wall subcharges.
        Defaults to ``True`` for improved agreement with the aperture geometry.
    radiation_reaction_mode:
        Radiation-reaction handling mode forwarded to the canonical integrator.
        Defaults to ``"medina_lad"`` for user-facing runs.
    macroparticle_charge_multiplier:
        Multiplier for particle and image charges in macroparticle simulations.
        Defaults to ``1.0`` (no scaling). Use > 1.0 for macroparticle mode.
        Only applies to CONDUCTING_WALL simulations.
    macroparticle_sigma_multiplier:
        Multiplier for bunch spread parameters when applying to image charge errors.
        Defaults to ``1.0`` (errors = bunch spread). Use > 1.0 to increase uncertainty.
        Position errors use transv_dist × multiplier, momentum errors use transv_mom × multiplier.
        Only applies to CONDUCTING_WALL simulations when macroparticle mode is enabled.
    macroparticle_use_momentum_errors:
        Whether to include momentum-based cumulative errors in image charge positions.
        If False, only constant position errors are applied (no cumulative momentum effects).
        Defaults to ``True`` (include both position and momentum errors).
    bunch_transv_dist:
        Transverse distribution half-width (mm) from particle bunch initialization.
        Used to compute position spread for image charge errors. Defaults to ``0.0``.
    bunch_transv_mom:
        Transverse momentum spread (amu*mm/ns) from particle bunch initialization.
        Used to compute cumulative displacement errors. Defaults to ``0.0``.
    pseudo_grid:
        Experimental pseudo-grid solver settings. The initial plumbing keeps the
        surface available across the API while the reduced-physics update path is
        implemented incrementally. Defaults to a disabled configuration.
    particle_loss:
        Optional fixed-size particle-loss predicates. Lost particles are marked
        dead, keep their trajectory slots, and stop contributing charge after
        the loss step.
    adaptive_pair_return:
        Guarded checkpointable shared-lab-time stepping for one exact rider and
        one exact driver. This is independent of the legacy adaptive-timestep
        controller.
    """

    steps: int
    time_step: float
    wall_position: float
    aperture_radius: float
    simulation_type: SimulationType
    chrono_mode: ChronoMatchingMode = ChronoMatchingMode.FAST
    startup_mode: StartupMode = StartupMode.COLD_START
    bunch_mean: float = 0.0
    cavity_spacing: float = 0.0
    z_cutoff: float = 0.0
    z_cutoff_mode: str = "absolute"
    image_subcharge_count: int = 12
    use_image_weighting: bool = True
    radiation_reaction_mode: str = "medina_lad"
    macroparticle_charge_multiplier: float = 1.0
    macroparticle_sigma_multiplier: float = 1.0
    macroparticle_use_momentum_errors: bool = True
    bunch_transv_dist: float = 0.0
    bunch_transv_mom: float = 0.0
    pseudo_grid: PseudoGridConfig = field(default_factory=PseudoGridConfig)
    macroparticle_smearing: MacroparticleSmearingConfig = field(
        default_factory=MacroparticleSmearingConfig
    )
    driver_train: DriverTrainConfig = field(default_factory=DriverTrainConfig)
    cavity_exit: CavityExitConfig = field(default_factory=CavityExitConfig)
    particle_loss: ParticleLossConfig = field(default_factory=ParticleLossConfig)
    beamline_geometry: BeamlineGeometryConfig = field(
        default_factory=BeamlineGeometryConfig
    )
    magnetic_dipole: MagneticDipoleConfig = field(default_factory=MagneticDipoleConfig)
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig())
    adaptive_pair_return: AdaptivePairReturnConfig = field(
        default_factory=AdaptivePairReturnConfig
    )


@dataclass
class CheckpointConfig:
    """Accepted-step checkpoint and restart controls.

    Checkpoints are append-only directories.  ``directory`` selects a new
    checkpoint; ``resume_from`` reopens an existing one and continues writing
    there.  A write is due when either accepted-step or wall-clock interval is
    reached.  Setting an interval to zero disables that trigger.
    """

    enabled: bool = False
    directory: str | None = None
    resume_from: str | None = None
    interval_steps: int = 1000
    interval_seconds: float = 900.0

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled or self.resume_from is not None)
        if self.directory is not None:
            self.directory = str(self.directory)
        if self.resume_from is not None:
            self.resume_from = str(self.resume_from)
        self.interval_steps = int(self.interval_steps)
        self.interval_seconds = float(self.interval_seconds)
        if self.interval_steps < 0:
            raise ValueError("checkpoint interval_steps must be non-negative")
        if not np.isfinite(self.interval_seconds) or self.interval_seconds < 0.0:
            raise ValueError(
                "checkpoint interval_seconds must be finite and non-negative"
            )
        if self.enabled and self.interval_steps == 0 and self.interval_seconds == 0.0:
            raise ValueError(
                "checkpointing needs a positive step or wall-clock interval"
            )
        if self.enabled and self.resume_from is None and not self.directory:
            raise ValueError("checkpoint directory is required when checkpointing")
        if (
            self.resume_from is not None
            and self.directory is not None
            and self.directory != self.resume_from
        ):
            raise ValueError(
                "checkpoint directory must match resume_from when both are supplied"
            )


@dataclass
class SpaceChargeConfig:
    """Configuration for intra-bunch space-charge forces.

    When enabled, each rider particle also receives retarded Liénard-Wiechert
    forces from all *other* rider particles (j ≠ i) in addition to the
    driver/image forces already computed.  The Numba hot-path is bypassed
    automatically; the feature uses the same vectorised Python kernel as
    self-consistency and adaptive-timestep modes.

    The transition from instantaneous Coulomb (startup) to full retarded fields
    is governed by ``min_retarded_steps``:

    - ``None`` (default): auto-compute as ``ceil(bunch_sigma_mm / (c · h_step))``
      i.e. the number of steps needed for light to cross the bunch width.  This
      requires ``bunch_sigma_mm`` and the integrator ``h_step`` to be known at
      runtime; the equations module resolves it on first use.
    - ``0``: use retarded fields from step 1 onward (original minimal behaviour).
    - ``N > 0``: hold instantaneous Coulomb for at least N steps regardless of
      bunch size.
    """

    enabled: bool = True
    retarded: bool = True  # full retarded fields; False → always instantaneous Coulomb
    softening_mm: float = 0.0  # Plummer softening ε (mm); 0 = no softening
    bunch_sigma_mm: float = (
        0.01  # transverse RMS of the bunch (mm); used for auto threshold
    )
    min_retarded_steps: int | None = (
        None  # None = auto from bunch_sigma_mm / (c * h_step)
    )

    def __post_init__(self) -> None:
        if self.softening_mm < 0.0:
            raise ValueError("space-charge softening_mm must be non-negative")
        if self.bunch_sigma_mm <= 0.0:
            raise ValueError("space-charge bunch_sigma_mm must be positive")
        if self.min_retarded_steps is not None and self.min_retarded_steps < 0:
            raise ValueError("space-charge min_retarded_steps must be non-negative")

    def resolve_min_retarded_steps(self, h_step: float) -> int:
        """Return the step threshold below which instantaneous Coulomb is used.

        If ``min_retarded_steps`` is explicitly set, return it directly.
        Otherwise compute ``ceil(bunch_sigma_mm / (C_MMNS * h_step))`` so that
        retarded fields are only used once the trajectory spans at least one
        light-crossing time of the bunch.
        """
        if not self.retarded:
            return 10**9  # retarded never requested — always instantaneous
        if self.min_retarded_steps is not None:
            return self.min_retarded_steps
        import math

        light_crossing_ns = self.bunch_sigma_mm / C_MMNS
        return max(1, math.ceil(light_crossing_ns / h_step))


@dataclass
class ExternalFieldConfig:
    """Configuration for prescribed external electromagnetic fields.

    Field components use the solver's native units. Electric field components
    are force per native charge, i.e. ``amu * mm / ns^2 / q_native``. Magnetic
    field components are expressed in the same force-per-charge convention and
    enter the Lorentz term as ``beta × B``.

    This first implementation supports uniform fields, plus an optional linear
    magnetic-field gradient in SI T/m, with simple spatial/temporal windows.
    More general field maps or callable field providers can build on the same
    integrator hook later.
    """

    enabled: bool = True
    electric_field_native: tuple[float, float, float] = (0.0, 0.0, 0.0)
    magnetic_field_native: tuple[float, float, float] = (0.0, 0.0, 0.0)
    magnetic_field_gradient_t_per_m: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None
    t_min: float | None = None
    t_max: float | None = None

    def __post_init__(self) -> None:
        gradient = np.asarray(self.magnetic_field_gradient_t_per_m, dtype=float)
        if gradient.shape != (3, 3) or not np.all(np.isfinite(gradient)):
            raise ValueError(
                "magnetic_field_gradient_t_per_m must be a finite 3x3 matrix"
            )
        divergence = float(np.trace(gradient))
        divergence_tolerance = 1.0e-12 + 1.0e-10 * float(np.max(np.abs(gradient)))
        if abs(divergence) > divergence_tolerance:
            raise ValueError(
                "magnetic_field_gradient_t_per_m must satisfy div(B)=0; "
                f"matrix trace is {divergence:.12g} T/m"
            )
        self.magnetic_field_gradient_t_per_m = (
            (
                float(gradient[0, 0]),
                float(gradient[0, 1]),
                float(gradient[0, 2]),
            ),
            (
                float(gradient[1, 0]),
                float(gradient[1, 1]),
                float(gradient[1, 2]),
            ),
            (
                float(gradient[2, 0]),
                float(gradient[2, 1]),
                float(gradient[2, 2]),
            ),
        )

    def is_active(self, x: float, y: float, z: float, t: float) -> bool:
        """Return whether the field should be applied at a particle location."""
        if not self.enabled:
            return False
        bounds = (
            (self.x_min, self.x_max, x),
            (self.y_min, self.y_max, y),
            (self.z_min, self.z_max, z),
            (self.t_min, self.t_max, t),
        )
        for lower, upper, value in bounds:
            if lower is not None and value < lower:
                return False
            if upper is not None and value > upper:
                return False
        return True


@dataclass
class TrajectoryArrays:
    """Struct-of-arrays trajectory representation.

    All kinematic fields have shape ``[n_steps, n_particles]``.
    Particle-constant fields (``q``, ``q_source``, ``q_observer``, ``q_species``,
    ``macro_population``, ``m``, ``m_species``, ``char_time``) have shape
    ``[n_particles]``.  Per-step scalar metadata has shape ``[n_steps]``.
    """

    # Kinematic — [n_steps, n_particles]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    t: np.ndarray
    Px: np.ndarray
    Py: np.ndarray
    Pz: np.ndarray
    Pt: np.ndarray
    gamma: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    bdotx: np.ndarray
    bdoty: np.ndarray
    bdotz: np.ndarray
    radiation_power: np.ndarray
    radiation_energy: np.ndarray
    radiation_energy_applied: np.ndarray
    mass_shell_projection_energy: np.ndarray
    radiation_reaction_work: np.ndarray
    medina_cross_field_energy: np.ndarray
    medina_cross_field_energy_change: np.ndarray
    medina_force_derivative_ready: np.ndarray
    medina_impulse_capped: np.ndarray
    medina_external_force_x: np.ndarray
    medina_external_force_y: np.ndarray
    medina_external_force_z: np.ndarray
    medina_external_force_sample_time: np.ndarray
    origin_x: np.ndarray
    origin_y: np.ndarray
    origin_z: np.ndarray
    beta_avg_x: np.ndarray
    beta_avg_y: np.ndarray
    beta_avg_z: np.ndarray
    beta_samples: np.ndarray
    spin_x: np.ndarray
    spin_y: np.ndarray
    spin_z: np.ndarray
    local_magnetic_field_x_t: np.ndarray
    local_magnetic_field_y_t: np.ndarray
    local_magnetic_field_z_t: np.ndarray

    # Dead-particle mask — [n_steps, n_particles], bool
    dead: np.ndarray

    # Particle constants — [n_particles]
    q: np.ndarray
    q_species: np.ndarray
    q_observer: np.ndarray
    q_source: np.ndarray
    macro_population: np.ndarray
    m: np.ndarray
    m_species: np.ndarray
    char_time: np.ndarray
    magnetic_moment_j_per_t: np.ndarray
    magnetic_moment_native: np.ndarray
    spin_quantum_number: np.ndarray
    gyromagnetic_ratio_rad_s_t: np.ndarray
    magnetic_dipole_active: np.ndarray
    spin_precession_active: np.ndarray
    stern_gerlach_active: np.ndarray

    # Per-step scalars — [n_steps]
    halted_early: np.ndarray  # dtype bool
    halt_step: np.ndarray  # dtype int64, -1 if not halted

    # Non-array side-channels
    halt_reason: list  # length n_steps, str or None
    particle_failure_info: dict  # keyed by (step, particle_idx)
    pseudo_grid_schedule: list  # length n_steps, object or None

    # Builder-owned live metadata. Manually constructed SOA objects deliberately
    # leave this unset and are therefore not eligible for persistent caches.
    _storage_state: _TrajectoryStorageState | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _storage_array_revision: int | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def n_steps(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_particles(self) -> int:
        return int(self.x.shape[1])

    @property
    def storage_token(self) -> int | None:
        """Stable allocation token, or ``None`` for unmanaged arrays."""

        return None if self._storage_state is None else self._storage_state.token

    @property
    def storage_generation(self) -> int | None:
        """Current builder-write generation shared by all allocation views."""

        return None if self._storage_state is None else self._storage_state.generation

    @property
    def storage_capacity(self) -> int | None:
        """Allocated row capacity, or ``None`` for unmanaged arrays."""

        return None if self._storage_state is None else self._storage_state.capacity

    @property
    def storage_rewrite_epoch(self) -> int | None:
        """Current epoch for rewrites of previously exposed rows."""

        return (
            None if self._storage_state is None else self._storage_state.rewrite_epoch
        )

    @property
    def storage_array_revision(self) -> int | None:
        """Backing-array revision captured when this wrapper was constructed."""

        return self._storage_array_revision

    def require_current_storage(self) -> None:
        """Reject a wrapper after its builder replaces a backing-array family."""

        state = self._storage_state
        if state is None:
            return
        if self._storage_array_revision != state.array_revision:
            raise StaleTrajectoryViewError(
                "trajectory view is stale because its builder replaced backing "
                "arrays; request a fresh build_partial() or build() view"
            )

    def _make_managed_arrays_read_only(self) -> "TrajectoryArrays":
        """Expose non-writeable views while the builder keeps writable bases."""

        for descriptor in fields(self):
            values = getattr(self, descriptor.name)
            if isinstance(values, np.ndarray):
                readonly = values.view()
                readonly.flags.writeable = False
                setattr(self, descriptor.name, readonly)
        return self

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore managed wrappers with their read-only publication contract."""

        self.__dict__.update(state)
        if self._storage_state is not None:
            self._make_managed_arrays_read_only()

    def state_at(self, step: int) -> ParticleState:
        """Return a legacy ``ParticleState`` dict for *step*."""
        self.require_current_storage()
        s: ParticleState = {
            "x": self.x[step],
            "y": self.y[step],
            "z": self.z[step],
            "t": self.t[step],
            "Px": self.Px[step],
            "Py": self.Py[step],
            "Pz": self.Pz[step],
            "Pt": self.Pt[step],
            "gamma": self.gamma[step],
            "bx": self.bx[step],
            "by": self.by[step],
            "bz": self.bz[step],
            "bdotx": self.bdotx[step],
            "bdoty": self.bdoty[step],
            "bdotz": self.bdotz[step],
            "radiation_power": self.radiation_power[step],
            "radiation_energy": self.radiation_energy[step],
            "radiation_energy_applied": self.radiation_energy_applied[step],
            "mass_shell_projection_energy": self.mass_shell_projection_energy[step],
            "q": self.q,
            "q_species": self.q_species,
            "q_observer": self.q_observer,
            "q_source": self.q_source,
            "macro_population": self.macro_population,
            "m": self.m,
            "m_species": self.m_species,
            "char_time": self.char_time,
            "origin_x": self.origin_x[step],
            "origin_y": self.origin_y[step],
            "origin_z": self.origin_z[step],
            "beta_avg_x": self.beta_avg_x[step],
            "beta_avg_y": self.beta_avg_y[step],
            "beta_avg_z": self.beta_avg_z[step],
            "beta_samples": self.beta_samples[step],
            "_dead_particles": self.dead[step],
        }
        if self.halted_early[step]:
            metadata = cast(Dict[str, object], s)
            metadata["_halted_early"] = bool(self.halted_early[step])
            metadata["_halt_step"] = int(self.halt_step[step])
            metadata["_halt_reason"] = self.halt_reason[step]
        pseudo_grid_schedule = self.pseudo_grid_schedule[step]
        if pseudo_grid_schedule is not None:
            s["_pseudo_grid_schedule"] = pseudo_grid_schedule
        if np.any(np.isfinite(self.medina_external_force_sample_time[step])):
            s.update(
                {
                    "radiation_reaction_work": self.radiation_reaction_work[step],
                    "medina_cross_field_energy": self.medina_cross_field_energy[step],
                    "medina_cross_field_energy_change": (
                        self.medina_cross_field_energy_change[step]
                    ),
                    "medina_force_derivative_ready": (
                        self.medina_force_derivative_ready[step]
                    ),
                    "medina_impulse_capped": self.medina_impulse_capped[step],
                    "medina_external_force_x": self.medina_external_force_x[step],
                    "medina_external_force_y": self.medina_external_force_y[step],
                    "medina_external_force_z": self.medina_external_force_z[step],
                    "medina_external_force_sample_time": (
                        self.medina_external_force_sample_time[step]
                    ),
                }
            )
        if np.any(self.magnetic_dipole_active):
            s.update(
                {
                    "spin_x": self.spin_x[step],
                    "spin_y": self.spin_y[step],
                    "spin_z": self.spin_z[step],
                    "local_magnetic_field_x_t": self.local_magnetic_field_x_t[step],
                    "local_magnetic_field_y_t": self.local_magnetic_field_y_t[step],
                    "local_magnetic_field_z_t": self.local_magnetic_field_z_t[step],
                    "magnetic_moment_j_per_t": self.magnetic_moment_j_per_t,
                    "magnetic_moment_native": self.magnetic_moment_native,
                    "spin_quantum_number": self.spin_quantum_number,
                    "gyromagnetic_ratio_rad_s_t": (self.gyromagnetic_ratio_rad_s_t),
                    "magnetic_dipole_active": self.magnetic_dipole_active,
                    "spin_precession_active": self.spin_precession_active,
                    "stern_gerlach_active": self.stern_gerlach_active,
                }
            )
        return s

    def to_legacy(self) -> "Trajectory":
        """Return a full ``List[ParticleState]`` compatible with legacy consumers."""
        return [self.state_at(i) for i in range(self.n_steps)]


@dataclass(frozen=True)
class TrialTrajectoryHistory:
    """Immutable one- or two-row tail over accepted managed history.

    Step-doubling trials need provisional endpoints to participate in exact
    light-cone interpolation without publishing rejected knots. Providers
    recognize this wrapper and normally share the prepared prefix buffers,
    copying them only when the tiny tail crosses a geometric-capacity
    boundary. The accepted base and its cache entry remain logically
    unchanged.

    The first return milestone deliberately permits at most two trial rows:
    one full endpoint, or one midpoint plus one refined endpoint.
    """

    base: TrajectoryArrays
    tail: tuple[ParticleState, ...]

    def __post_init__(self) -> None:
        self.base.require_current_storage()
        if len(self.tail) not in {1, 2}:
            raise ValueError("trial history tail must contain one or two rows")
        particle_count = self.base.n_particles
        previous_time = np.asarray(self.base.t[-1], dtype=np.float64)
        detached_rows: list[ParticleState] = []
        constant_fields = (
            "q",
            "q_source",
            "magnetic_moment_native",
            "magnetic_dipole_active",
        )
        for state in self.tail:
            if "t" not in state:
                raise ValueError("trial history rows require coordinate time t")
            next_time = np.asarray(state["t"], dtype=np.float64)
            if next_time.shape != (particle_count,) or not np.all(
                np.isfinite(next_time)
            ):
                raise ValueError(
                    "trial history coordinate time must contain one finite value "
                    "per source"
                )
            if np.any(next_time <= previous_time):
                raise ValueError(
                    "trial history coordinate time must increase for every source"
                )
            for field_name in constant_fields:
                if field_name not in state or not hasattr(self.base, field_name):
                    continue
                values = np.asarray(state[field_name])
                expected = np.asarray(getattr(self.base, field_name))
                if not np.array_equal(values, expected, equal_nan=True):
                    raise ValueError(
                        f"trial history source constant {field_name} changed"
                    )
            detached: ParticleState = {}
            for name, values in state.items():
                if isinstance(values, np.ndarray):
                    copied = np.array(values, copy=True)
                    copied.flags.writeable = False
                    detached[name] = copied
                else:
                    detached[name] = copy.deepcopy(values)
            detached_rows.append(detached)
            previous_time = next_time
        object.__setattr__(self, "tail", tuple(detached_rows))

    @property
    def n_steps(self) -> int:
        return self.base.n_steps + len(self.tail)


@dataclass
class IndexedTrajectoryArrays:
    """Particle-indexed view over a :class:`TrajectoryArrays` history."""

    base: TrajectoryArrays
    particle_indices: np.ndarray
    start_step: int = 0
    q_override: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.base.require_current_storage()
        indices = np.asarray(self.particle_indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("particle_indices must be a 1-D array")
        if np.any(indices < 0) or np.any(indices >= self.base.n_particles):
            raise ValueError("particle_indices are out of bounds for base trajectory")
        if self.start_step < 0 or self.start_step > self.base.n_steps:
            raise ValueError("start_step is out of bounds for base trajectory")
        if self.q_override is not None:
            q_override = np.asarray(self.q_override, dtype=float)
            if q_override.shape != (indices.size,):
                raise ValueError("q_override must match the indexed particle count")
            self.q_override = q_override
        self.particle_indices = indices

    @property
    def n_steps(self) -> int:
        return self.base.n_steps - int(self.start_step)

    @property
    def n_particles(self) -> int:
        return int(self.particle_indices.size)

    def global_step(self, step: int) -> int:
        local_step = int(step)
        if local_step < 0:
            local_step += self.n_steps
        if local_step < 0 or local_step >= self.n_steps:
            raise IndexError("trajectory step index out of range")
        return int(self.start_step) + local_step

    def row(self, field_name: str, step: int) -> np.ndarray:
        self.base.require_current_storage()
        values = np.asarray(getattr(self.base, field_name))[self.global_step(step), :][
            self.particle_indices
        ]
        return np.asarray(values)

    def scalar(self, field_name: str, step: int, particle_idx: int) -> float:
        self.base.require_current_storage()
        return float(
            np.asarray(getattr(self.base, field_name))[
                self.global_step(step),
                int(self.particle_indices[int(particle_idx)]),
            ]
        )

    def values_at_steps(
        self,
        field_name: str,
        steps: np.ndarray,
        particle_indices: np.ndarray,
    ) -> np.ndarray:
        self.base.require_current_storage()
        local_steps = np.asarray(steps, dtype=int)
        local_particles = np.asarray(particle_indices, dtype=int)
        values = np.asarray(getattr(self.base, field_name))[
            int(self.start_step) + local_steps,
            self.particle_indices[local_particles],
        ]
        return np.asarray(values)

    def time_columns(self, up_to_step: int) -> np.ndarray:
        self.base.require_current_storage()
        end_step = self.global_step(up_to_step) + 1
        return np.asarray(self.base.t)[int(self.start_step) : end_step, :][
            :, self.particle_indices
        ]

    def constant(self, field_name: str) -> np.ndarray:
        self.base.require_current_storage()
        if field_name in {"q", "q_source"} and self.q_override is not None:
            return np.asarray(self.q_override, dtype=float)
        values = np.asarray(getattr(self.base, field_name))[self.particle_indices]
        return np.asarray(values)

    def state_at(self, step: int) -> ParticleState:
        global_step = self.global_step(step)
        state: ParticleState = {
            "x": self.row("x", step),
            "y": self.row("y", step),
            "z": self.row("z", step),
            "t": self.row("t", step),
            "Px": self.row("Px", step),
            "Py": self.row("Py", step),
            "Pz": self.row("Pz", step),
            "Pt": self.row("Pt", step),
            "gamma": self.row("gamma", step),
            "bx": self.row("bx", step),
            "by": self.row("by", step),
            "bz": self.row("bz", step),
            "bdotx": self.row("bdotx", step),
            "bdoty": self.row("bdoty", step),
            "bdotz": self.row("bdotz", step),
            "radiation_power": self.row("radiation_power", step),
            "radiation_energy": self.row("radiation_energy", step),
            "radiation_energy_applied": self.row("radiation_energy_applied", step),
            "mass_shell_projection_energy": self.row(
                "mass_shell_projection_energy", step
            ),
            "q": self.constant("q"),
            "q_species": self.constant("q_species"),
            "q_observer": self.constant("q_observer"),
            "q_source": self.constant("q_source"),
            "macro_population": self.constant("macro_population"),
            "m": self.constant("m"),
            "m_species": self.constant("m_species"),
            "char_time": self.constant("char_time"),
            "origin_x": self.row("origin_x", step),
            "origin_y": self.row("origin_y", step),
            "origin_z": self.row("origin_z", step),
            "beta_avg_x": self.row("beta_avg_x", step),
            "beta_avg_y": self.row("beta_avg_y", step),
            "beta_avg_z": self.row("beta_avg_z", step),
            "beta_samples": self.row("beta_samples", step),
            "_dead_particles": np.asarray(self.base.dead)[
                global_step,
                self.particle_indices,
            ],
        }
        pseudo_grid_schedule = self.base.pseudo_grid_schedule[global_step]
        if pseudo_grid_schedule is not None:
            state["_pseudo_grid_schedule"] = pseudo_grid_schedule
        if np.any(np.isfinite(self.row("medina_external_force_sample_time", step))):
            state.update(
                {
                    "radiation_reaction_work": self.row(
                        "radiation_reaction_work", step
                    ),
                    "medina_cross_field_energy": self.row(
                        "medina_cross_field_energy", step
                    ),
                    "medina_cross_field_energy_change": self.row(
                        "medina_cross_field_energy_change", step
                    ),
                    "medina_force_derivative_ready": self.row(
                        "medina_force_derivative_ready", step
                    ),
                    "medina_impulse_capped": self.row("medina_impulse_capped", step),
                    "medina_external_force_x": self.row(
                        "medina_external_force_x", step
                    ),
                    "medina_external_force_y": self.row(
                        "medina_external_force_y", step
                    ),
                    "medina_external_force_z": self.row(
                        "medina_external_force_z", step
                    ),
                    "medina_external_force_sample_time": self.row(
                        "medina_external_force_sample_time", step
                    ),
                }
            )
        if self.base.halted_early[global_step]:
            metadata = cast(Dict[str, object], state)
            metadata["_halted_early"] = bool(self.base.halted_early[global_step])
            metadata["_halt_step"] = int(self.base.halt_step[global_step])
            metadata["_halt_reason"] = self.base.halt_reason[global_step]
        if np.any(self.constant("magnetic_dipole_active")):
            state.update(
                {
                    "spin_x": self.row("spin_x", step),
                    "spin_y": self.row("spin_y", step),
                    "spin_z": self.row("spin_z", step),
                    "local_magnetic_field_x_t": self.row(
                        "local_magnetic_field_x_t", step
                    ),
                    "local_magnetic_field_y_t": self.row(
                        "local_magnetic_field_y_t", step
                    ),
                    "local_magnetic_field_z_t": self.row(
                        "local_magnetic_field_z_t", step
                    ),
                    "magnetic_moment_j_per_t": self.constant("magnetic_moment_j_per_t"),
                    "magnetic_moment_native": self.constant("magnetic_moment_native"),
                    "spin_quantum_number": self.constant("spin_quantum_number"),
                    "gyromagnetic_ratio_rad_s_t": self.constant(
                        "gyromagnetic_ratio_rad_s_t"
                    ),
                    "magnetic_dipole_active": self.constant("magnetic_dipole_active"),
                    "spin_precession_active": self.constant("spin_precession_active"),
                    "stern_gerlach_active": self.constant("stern_gerlach_active"),
                }
            )
        return state

    def to_legacy(self) -> "Trajectory":
        return [self.state_at(i) for i in range(self.n_steps)]


class TrajectoryBuilder:
    """Incremental accumulator for building a :class:`TrajectoryArrays`.

    Pre-allocates all arrays at construction time; each integration step
    writes one row via :meth:`set_step`.
    """

    # Fields present in legacy state dicts that map to 2-D kinematic arrays
    _KINEMATIC_FIELDS: tuple = (
        "x",
        "y",
        "z",
        "t",
        "Px",
        "Py",
        "Pz",
        "Pt",
        "gamma",
        "bx",
        "by",
        "bz",
        "bdotx",
        "bdoty",
        "bdotz",
        "radiation_power",
        "radiation_energy",
        "radiation_energy_applied",
        "mass_shell_projection_energy",
        "origin_x",
        "origin_y",
        "origin_z",
        "beta_avg_x",
        "beta_avg_y",
        "beta_avg_z",
        "beta_samples",
    )
    _MAGNETIC_KINEMATIC_FIELDS: tuple = (
        "spin_x",
        "spin_y",
        "spin_z",
        "local_magnetic_field_x_t",
        "local_magnetic_field_y_t",
        "local_magnetic_field_z_t",
    )
    _MEDINA_FLOAT_FIELDS: tuple = (
        "radiation_reaction_work",
        "medina_cross_field_energy",
        "medina_cross_field_energy_change",
        "medina_external_force_x",
        "medina_external_force_y",
        "medina_external_force_z",
        "medina_external_force_sample_time",
    )
    _MEDINA_BOOL_FIELDS: tuple = (
        "medina_force_derivative_ready",
        "medina_impulse_capped",
    )
    _PARTICLE_CONST_FIELDS: tuple = (
        "q",
        "q_species",
        "q_observer",
        "q_source",
        "macro_population",
        "m",
        "m_species",
        "char_time",
        "magnetic_moment_j_per_t",
        "magnetic_moment_native",
        "spin_quantum_number",
        "gyromagnetic_ratio_rad_s_t",
        "magnetic_dipole_active",
        "spin_precession_active",
        "stern_gerlach_active",
    )

    def __init__(
        self, n_steps: int, n_particles: int, *, magnetic_dipole: bool = False
    ) -> None:
        self._n_steps = n_steps
        self._n_particles = n_particles
        self._magnetic_arrays_allocated = bool(magnetic_dipole)
        self._medina_arrays_allocated = False
        self._storage_state = _TrajectoryStorageState(capacity=int(n_steps))
        self._published_stop = 0

        self._arrays: dict = {
            field_name: np.zeros((n_steps, n_particles), dtype=np.float64)
            for field_name in self._KINEMATIC_FIELDS
        }
        for field_name in self._MAGNETIC_KINEMATIC_FIELDS:
            if self._magnetic_arrays_allocated:
                magnetic_array = np.zeros((n_steps, n_particles), dtype=np.float64)
            else:
                # Preserve the public SOA shape without paying one full array
                # per magnetic diagnostic in feature-off simulations.
                magnetic_array = np.broadcast_to(
                    np.array(0.0, dtype=np.float64), (n_steps, n_particles)
                )
            self._arrays[field_name] = magnetic_array
        for field_name in self._MEDINA_FLOAT_FIELDS:
            default = (
                np.nan if field_name == "medina_external_force_sample_time" else 0.0
            )
            self._arrays[field_name] = np.broadcast_to(
                np.array(default, dtype=np.float64),
                (n_steps, n_particles),
            )
        for field_name in self._MEDINA_BOOL_FIELDS:
            self._arrays[field_name] = np.broadcast_to(
                np.array(False, dtype=bool),
                (n_steps, n_particles),
            )
        self._arrays["dead"] = np.zeros((n_steps, n_particles), dtype=bool)

        for field_name in self._PARTICLE_CONST_FIELDS:
            self._arrays[field_name] = np.zeros(n_particles, dtype=np.float64)

        self._halted_early = np.zeros(n_steps, dtype=bool)
        self._halt_step_arr = np.full(n_steps, -1, dtype=np.int64)
        self._halt_reason: list = [None] * n_steps
        self._particle_failure_info: dict = {}
        self._pseudo_grid_schedule: list = [None] * n_steps

    def _replace_row_capacity(self, new_capacity: int) -> None:
        """Replace row-shaped storage while preserving every existing value.

        Fixed-size integration never calls this helper.  The growable accepted-
        history builder below uses it at geometric capacity boundaries.  Every
        previously published wrapper becomes stale because all row-array bases
        have changed; the live array revision makes that failure explicit.
        """

        new_capacity = int(new_capacity)
        if new_capacity <= self._n_steps:
            raise ValueError("new trajectory capacity must exceed the old capacity")

        old_capacity = self._n_steps
        always_allocated = set(self._KINEMATIC_FIELDS) | {"dead"}
        magnetic_fields = set(self._MAGNETIC_KINEMATIC_FIELDS)
        medina_float_fields = set(self._MEDINA_FLOAT_FIELDS)
        medina_bool_fields = set(self._MEDINA_BOOL_FIELDS)
        for field_name in (
            always_allocated
            | magnetic_fields
            | medina_float_fields
            | medina_bool_fields
        ):
            old = self._arrays[field_name]
            if field_name in magnetic_fields and not self._magnetic_arrays_allocated:
                replacement = np.broadcast_to(
                    np.array(0.0, dtype=np.float64),
                    (new_capacity, self._n_particles),
                )
            elif (
                field_name in medina_float_fields and not self._medina_arrays_allocated
            ):
                default = (
                    np.nan if field_name == "medina_external_force_sample_time" else 0.0
                )
                replacement = np.broadcast_to(
                    np.array(default, dtype=np.float64),
                    (new_capacity, self._n_particles),
                )
            elif field_name in medina_bool_fields and not self._medina_arrays_allocated:
                replacement = np.broadcast_to(
                    np.array(False, dtype=bool),
                    (new_capacity, self._n_particles),
                )
            else:
                replacement = np.zeros(
                    (new_capacity, self._n_particles), dtype=old.dtype
                )
                replacement[:old_capacity] = old
            self._arrays[field_name] = replacement

        halted_early = np.zeros(new_capacity, dtype=bool)
        halted_early[:old_capacity] = self._halted_early
        self._halted_early = halted_early
        halt_step = np.full(new_capacity, -1, dtype=np.int64)
        halt_step[:old_capacity] = self._halt_step_arr
        self._halt_step_arr = halt_step
        self._halt_reason.extend([None] * (new_capacity - old_capacity))
        self._pseudo_grid_schedule.extend([None] * (new_capacity - old_capacity))

        self._n_steps = new_capacity
        self._storage_state.capacity = new_capacity
        self._storage_state.rewrite_epoch += 1
        self._storage_state.array_revision += 1

    def set_step(self, step: int, state: ParticleState) -> None:
        """Copy *state* fields into row *step* of the pre-allocated arrays."""
        step = int(step)
        if step < 0:
            step += self._n_steps
        if step < 0 or step >= self._n_steps:
            raise IndexError("trajectory step index out of range")
        self._storage_state.generation += 1
        if step < self._published_stop:
            self._storage_state.rewrite_epoch += 1

        if not self._magnetic_arrays_allocated and any(
            field_name in state for field_name in self._MAGNETIC_KINEMATIC_FIELDS
        ):
            for field_name in self._MAGNETIC_KINEMATIC_FIELDS:
                self._arrays[field_name] = np.zeros(
                    (self._n_steps, self._n_particles), dtype=np.float64
                )
            self._magnetic_arrays_allocated = True
            # Replacing one family of backing arrays changes the storage seen
            # by any previously built view, even when the current row is new.
            self._storage_state.rewrite_epoch += 1
            self._storage_state.array_revision += 1

        medina_fields = self._MEDINA_FLOAT_FIELDS + self._MEDINA_BOOL_FIELDS
        if not self._medina_arrays_allocated and any(
            field_name in state for field_name in medina_fields
        ):
            for field_name in self._MEDINA_FLOAT_FIELDS:
                if field_name == "medina_external_force_sample_time":
                    self._arrays[field_name] = np.full(
                        (self._n_steps, self._n_particles),
                        np.nan,
                        dtype=np.float64,
                    )
                else:
                    self._arrays[field_name] = np.zeros(
                        (self._n_steps, self._n_particles),
                        dtype=np.float64,
                    )
            for field_name in self._MEDINA_BOOL_FIELDS:
                self._arrays[field_name] = np.zeros(
                    (self._n_steps, self._n_particles),
                    dtype=bool,
                )
            self._medina_arrays_allocated = True
            self._storage_state.rewrite_epoch += 1
            self._storage_state.array_revision += 1

        for field_name in (
            self._KINEMATIC_FIELDS + self._MAGNETIC_KINEMATIC_FIELDS + medina_fields
        ):
            if field_name in state:
                self._arrays[field_name][step] = state[field_name]
            # else leave as zero (already pre-allocated)

        dead = state.get("_dead_particles")
        if dead is not None:
            self._arrays["dead"][step] = dead

        self._pseudo_grid_schedule[step] = state.get("_pseudo_grid_schedule")

        if step == 0:
            for field_name in self._PARTICLE_CONST_FIELDS:
                if field_name in state:
                    self._arrays[field_name][:] = state[field_name]
            q_values = state.get("q")
            if q_values is not None:
                if "q_species" not in state:
                    self._arrays["q_species"][:] = q_values
                if "q_observer" not in state:
                    self._arrays["q_observer"][:] = q_values
                if "q_source" not in state:
                    self._arrays["q_source"][:] = q_values
            if "macro_population" not in state:
                self._arrays["macro_population"][:] = 1.0
            if "m_species" not in state:
                self._arrays["m_species"][:] = state.get("m", 1.0)

    def set_canonical_momentum_step(
        self,
        step: int,
        state: ParticleState,
    ) -> None:
        """Replace only accepted ``P^mu`` after endpoint-potential rebasing.

        Retarded charge and dipole providers consume worldline, velocity,
        acceleration, spin, activity, and source constants, but never canonical
        momentum.  The pair-level exact endpoint finalizer therefore uses this
        narrow update after it has already published the provisional source
        row.  Avoiding a general ``set_step`` rewrite preserves the prepared
        history's append-only generation contract.
        """

        step = int(step)
        if step < 0:
            step += self._n_steps
        if step < 0 or step >= self._published_stop:
            raise IndexError("canonical momentum row must already be published")
        for field_name in ("Pt", "Px", "Py", "Pz"):
            values = np.asarray(state[field_name], dtype=np.float64)
            if values.shape != (self._n_particles,):
                raise ValueError(f"{field_name} must have shape ({self._n_particles},)")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{field_name} must contain only finite values")
            self._arrays[field_name][step] = values

    def restore_checkpoint_rows(
        self,
        start: int,
        row_arrays: dict[str, np.ndarray],
        *,
        particle_constants: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Bulk-load one contiguous accepted checkpoint block.

        This is intentionally narrower than a general mutation API.  Restart
        creates a fresh builder, restores rows in increasing order, and only
        then publishes a trajectory view.  No prepared-history consumer can
        therefore observe a partially restored block.
        """

        start = int(start)
        if start < 0:
            raise ValueError("checkpoint row start must be non-negative")
        if not row_arrays:
            raise ValueError("checkpoint row block must not be empty")
        first_values = np.asarray(next(iter(row_arrays.values())))
        if first_values.ndim < 1:
            raise ValueError("checkpoint row arrays must have a leading row axis")
        row_count = int(first_values.shape[0])
        stop = start + row_count
        if row_count <= 0 or stop > self._n_steps:
            raise ValueError("checkpoint row block exceeds trajectory capacity")
        if start != self._published_stop:
            raise ValueError("checkpoint rows must be restored contiguously")

        magnetic_fields = set(self._MAGNETIC_KINEMATIC_FIELDS)
        medina_fields = set(self._MEDINA_FLOAT_FIELDS + self._MEDINA_BOOL_FIELDS)
        if not self._magnetic_arrays_allocated and magnetic_fields & row_arrays.keys():
            for field_name in self._MAGNETIC_KINEMATIC_FIELDS:
                self._arrays[field_name] = np.zeros(
                    (self._n_steps, self._n_particles), dtype=np.float64
                )
            self._magnetic_arrays_allocated = True
            self._storage_state.array_revision += 1
        if not self._medina_arrays_allocated and medina_fields & row_arrays.keys():
            for field_name in self._MEDINA_FLOAT_FIELDS:
                fill = (
                    np.nan if field_name == "medina_external_force_sample_time" else 0.0
                )
                self._arrays[field_name] = np.full(
                    (self._n_steps, self._n_particles), fill, dtype=np.float64
                )
            for field_name in self._MEDINA_BOOL_FIELDS:
                self._arrays[field_name] = np.zeros(
                    (self._n_steps, self._n_particles), dtype=bool
                )
            self._medina_arrays_allocated = True
            self._storage_state.array_revision += 1

        expected_row_fields = set(
            self._KINEMATIC_FIELDS
            + self._MAGNETIC_KINEMATIC_FIELDS
            + self._MEDINA_FLOAT_FIELDS
            + self._MEDINA_BOOL_FIELDS
            + ("dead",)
        )
        missing = expected_row_fields - row_arrays.keys()
        if missing:
            raise ValueError(
                "checkpoint row block is missing fields: " + ", ".join(sorted(missing))
            )
        for field_name in expected_row_fields:
            values = np.asarray(row_arrays[field_name])
            target = self._arrays[field_name][start:stop]
            if values.shape != target.shape:
                raise ValueError(
                    f"checkpoint field {field_name} has shape {values.shape}, "
                    f"expected {target.shape}"
                )
            target[...] = values

        for name, target in (
            ("halted_early", self._halted_early),
            ("halt_step", self._halt_step_arr),
        ):
            values = np.asarray(row_arrays[name])
            if values.shape != (row_count,):
                raise ValueError(
                    f"checkpoint field {name} has shape {values.shape}, "
                    f"expected {(row_count,)}"
                )
            target[start:stop] = values

        if particle_constants is not None:
            missing_constants = (
                set(self._PARTICLE_CONST_FIELDS) - particle_constants.keys()
            )
            if missing_constants:
                raise ValueError(
                    "checkpoint constants are missing fields: "
                    + ", ".join(sorted(missing_constants))
                )
            for field_name in self._PARTICLE_CONST_FIELDS:
                values = np.asarray(particle_constants[field_name])
                target = self._arrays[field_name]
                if values.shape != target.shape:
                    raise ValueError(
                        f"checkpoint constant {field_name} has shape {values.shape}, "
                        f"expected {target.shape}"
                    )
                target[...] = values

        self._published_stop = stop
        self._storage_state.generation += row_count

    def set_halt_metadata(
        self,
        step: int,
        reason: str,
        halt_step: int,
        requested_steps: int,  # noqa: ARG002 — retained for API symmetry
    ) -> None:
        """Record halt information for *step*."""
        self._halted_early[step] = True
        self._halt_step_arr[step] = halt_step
        self._halt_reason[step] = reason

    def set_particle_failure(self, step: int, particle_idx: int, info: dict) -> None:
        """Store per-particle failure info keyed by ``(step, particle_idx)``."""
        self._particle_failure_info[(step, particle_idx)] = info

    def build_partial(self, up_to_step: int) -> "TrajectoryArrays":
        """Return a zero-copy view of the first *up_to_step* rows.

        The returned TrajectoryArrays shares memory with this builder's
        pre-allocated arrays. up_to_step must be >= 1.
        """
        if up_to_step < 1:
            raise ValueError(f"up_to_step must be >= 1, got {up_to_step}")
        s = up_to_step
        if s > self._n_steps:
            raise ValueError(
                f"up_to_step must be <= the allocated step count {self._n_steps}"
            )
        self._published_stop = max(self._published_stop, int(s))
        return TrajectoryArrays(
            x=self._arrays["x"][:s],
            y=self._arrays["y"][:s],
            z=self._arrays["z"][:s],
            t=self._arrays["t"][:s],
            Px=self._arrays["Px"][:s],
            Py=self._arrays["Py"][:s],
            Pz=self._arrays["Pz"][:s],
            Pt=self._arrays["Pt"][:s],
            gamma=self._arrays["gamma"][:s],
            bx=self._arrays["bx"][:s],
            by=self._arrays["by"][:s],
            bz=self._arrays["bz"][:s],
            bdotx=self._arrays["bdotx"][:s],
            bdoty=self._arrays["bdoty"][:s],
            bdotz=self._arrays["bdotz"][:s],
            radiation_power=self._arrays["radiation_power"][:s],
            radiation_energy=self._arrays["radiation_energy"][:s],
            radiation_energy_applied=self._arrays["radiation_energy_applied"][:s],
            mass_shell_projection_energy=self._arrays["mass_shell_projection_energy"][
                :s
            ],
            radiation_reaction_work=self._arrays["radiation_reaction_work"][:s],
            medina_cross_field_energy=self._arrays["medina_cross_field_energy"][:s],
            medina_cross_field_energy_change=self._arrays[
                "medina_cross_field_energy_change"
            ][:s],
            medina_force_derivative_ready=self._arrays["medina_force_derivative_ready"][
                :s
            ],
            medina_impulse_capped=self._arrays["medina_impulse_capped"][:s],
            medina_external_force_x=self._arrays["medina_external_force_x"][:s],
            medina_external_force_y=self._arrays["medina_external_force_y"][:s],
            medina_external_force_z=self._arrays["medina_external_force_z"][:s],
            medina_external_force_sample_time=self._arrays[
                "medina_external_force_sample_time"
            ][:s],
            origin_x=self._arrays["origin_x"][:s],
            origin_y=self._arrays["origin_y"][:s],
            origin_z=self._arrays["origin_z"][:s],
            beta_avg_x=self._arrays["beta_avg_x"][:s],
            beta_avg_y=self._arrays["beta_avg_y"][:s],
            beta_avg_z=self._arrays["beta_avg_z"][:s],
            beta_samples=self._arrays["beta_samples"][:s],
            spin_x=self._arrays["spin_x"][:s],
            spin_y=self._arrays["spin_y"][:s],
            spin_z=self._arrays["spin_z"][:s],
            local_magnetic_field_x_t=self._arrays["local_magnetic_field_x_t"][:s],
            local_magnetic_field_y_t=self._arrays["local_magnetic_field_y_t"][:s],
            local_magnetic_field_z_t=self._arrays["local_magnetic_field_z_t"][:s],
            dead=self._arrays["dead"][:s],
            q=self._arrays["q"],
            q_species=self._arrays["q_species"],
            q_observer=self._arrays["q_observer"],
            q_source=self._arrays["q_source"],
            macro_population=self._arrays["macro_population"],
            m=self._arrays["m"],
            m_species=self._arrays["m_species"],
            char_time=self._arrays["char_time"],
            magnetic_moment_j_per_t=self._arrays["magnetic_moment_j_per_t"],
            magnetic_moment_native=self._arrays["magnetic_moment_native"],
            spin_quantum_number=self._arrays["spin_quantum_number"],
            gyromagnetic_ratio_rad_s_t=self._arrays["gyromagnetic_ratio_rad_s_t"],
            magnetic_dipole_active=self._arrays["magnetic_dipole_active"],
            spin_precession_active=self._arrays["spin_precession_active"],
            stern_gerlach_active=self._arrays["stern_gerlach_active"],
            halted_early=self._halted_early[:s],
            halt_step=self._halt_step_arr[:s],
            halt_reason=self._halt_reason,
            particle_failure_info=self._particle_failure_info,
            pseudo_grid_schedule=self._pseudo_grid_schedule[:s],
            _storage_state=self._storage_state,
            _storage_array_revision=self._storage_state.array_revision,
        )._make_managed_arrays_read_only()

    def build(self) -> TrajectoryArrays:
        """Finalise and return the accumulated :class:`TrajectoryArrays`."""
        self._published_stop = self._n_steps
        return TrajectoryArrays(
            x=self._arrays["x"],
            y=self._arrays["y"],
            z=self._arrays["z"],
            t=self._arrays["t"],
            Px=self._arrays["Px"],
            Py=self._arrays["Py"],
            Pz=self._arrays["Pz"],
            Pt=self._arrays["Pt"],
            gamma=self._arrays["gamma"],
            bx=self._arrays["bx"],
            by=self._arrays["by"],
            bz=self._arrays["bz"],
            bdotx=self._arrays["bdotx"],
            bdoty=self._arrays["bdoty"],
            bdotz=self._arrays["bdotz"],
            radiation_power=self._arrays["radiation_power"],
            radiation_energy=self._arrays["radiation_energy"],
            radiation_energy_applied=self._arrays["radiation_energy_applied"],
            mass_shell_projection_energy=self._arrays["mass_shell_projection_energy"],
            radiation_reaction_work=self._arrays["radiation_reaction_work"],
            medina_cross_field_energy=self._arrays["medina_cross_field_energy"],
            medina_cross_field_energy_change=self._arrays[
                "medina_cross_field_energy_change"
            ],
            medina_force_derivative_ready=self._arrays["medina_force_derivative_ready"],
            medina_impulse_capped=self._arrays["medina_impulse_capped"],
            medina_external_force_x=self._arrays["medina_external_force_x"],
            medina_external_force_y=self._arrays["medina_external_force_y"],
            medina_external_force_z=self._arrays["medina_external_force_z"],
            medina_external_force_sample_time=self._arrays[
                "medina_external_force_sample_time"
            ],
            origin_x=self._arrays["origin_x"],
            origin_y=self._arrays["origin_y"],
            origin_z=self._arrays["origin_z"],
            beta_avg_x=self._arrays["beta_avg_x"],
            beta_avg_y=self._arrays["beta_avg_y"],
            beta_avg_z=self._arrays["beta_avg_z"],
            beta_samples=self._arrays["beta_samples"],
            spin_x=self._arrays["spin_x"],
            spin_y=self._arrays["spin_y"],
            spin_z=self._arrays["spin_z"],
            local_magnetic_field_x_t=self._arrays["local_magnetic_field_x_t"],
            local_magnetic_field_y_t=self._arrays["local_magnetic_field_y_t"],
            local_magnetic_field_z_t=self._arrays["local_magnetic_field_z_t"],
            dead=self._arrays["dead"],
            q=self._arrays["q"],
            q_species=self._arrays["q_species"],
            q_observer=self._arrays["q_observer"],
            q_source=self._arrays["q_source"],
            macro_population=self._arrays["macro_population"],
            m=self._arrays["m"],
            m_species=self._arrays["m_species"],
            char_time=self._arrays["char_time"],
            magnetic_moment_j_per_t=self._arrays["magnetic_moment_j_per_t"],
            magnetic_moment_native=self._arrays["magnetic_moment_native"],
            spin_quantum_number=self._arrays["spin_quantum_number"],
            gyromagnetic_ratio_rad_s_t=self._arrays["gyromagnetic_ratio_rad_s_t"],
            magnetic_dipole_active=self._arrays["magnetic_dipole_active"],
            spin_precession_active=self._arrays["spin_precession_active"],
            stern_gerlach_active=self._arrays["stern_gerlach_active"],
            halted_early=self._halted_early,
            halt_step=self._halt_step_arr,
            halt_reason=self._halt_reason,
            particle_failure_info=self._particle_failure_info,
            pseudo_grid_schedule=self._pseudo_grid_schedule,
            _storage_state=self._storage_state,
            _storage_array_revision=self._storage_state.array_revision,
        )._make_managed_arrays_read_only()


class GrowableTrajectoryBuilder(TrajectoryBuilder):
    """Append-only trajectory storage for accepted retarded-history knots.

    ``TrajectoryBuilder`` remains the fixed-size public-output accumulator.
    This separate builder starts with a small capacity and grows geometrically,
    allowing accepted source history to have a cadence and final length that
    are independent of public output.  It intentionally exposes no truncate or
    arbitrary-row append operation.

    Growth replaces backing arrays and therefore invalidates older managed
    views.  Appends within the current capacity preserve the allocation and are
    eligible for the existing append-aware provider caches.
    """

    def __init__(
        self,
        initial_capacity: int,
        n_particles: int,
        *,
        magnetic_dipole: bool = False,
        growth_factor: float = 2.0,
    ) -> None:
        initial_capacity = int(initial_capacity)
        n_particles = int(n_particles)
        growth_factor = float(growth_factor)
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be positive")
        if n_particles < 1:
            raise ValueError("n_particles must be positive")
        if not np.isfinite(growth_factor) or growth_factor <= 1.0:
            raise ValueError("growth_factor must be finite and greater than one")
        super().__init__(
            initial_capacity,
            n_particles,
            magnetic_dipole=magnetic_dipole,
        )
        self._growth_factor = growth_factor
        self._accepted_steps = 0

    @property
    def accepted_steps(self) -> int:
        """Number of accepted rows currently stored."""

        return self._accepted_steps

    @property
    def capacity(self) -> int:
        """Current allocated row capacity."""

        return self._n_steps

    def _validate_append_state(self, state: ParticleState) -> None:
        row_fields = (
            self._KINEMATIC_FIELDS
            + self._MAGNETIC_KINEMATIC_FIELDS
            + self._MEDINA_FLOAT_FIELDS
            + self._MEDINA_BOOL_FIELDS
        )
        for field_name in row_fields:
            if field_name not in state:
                continue
            values = np.asarray(state[field_name])
            if values.shape != (self._n_particles,):
                raise ValueError(f"{field_name} must have shape ({self._n_particles},)")
            if field_name == "medina_external_force_sample_time":
                valid = not np.any(np.isinf(values))
            else:
                valid = bool(np.all(np.isfinite(values)))
            if not valid:
                if field_name == "medina_external_force_sample_time":
                    raise ValueError(
                        "medina_external_force_sample_time must not contain infinity"
                    )
                raise ValueError(f"{field_name} must contain only finite values")

        dead = state.get("_dead_particles")
        if dead is not None and np.asarray(dead).shape != (self._n_particles,):
            raise ValueError(f"_dead_particles must have shape ({self._n_particles},)")

        if "t" not in state:
            raise ValueError("accepted history rows require coordinate time t")
        if self._accepted_steps:
            previous_t = self._arrays["t"][self._accepted_steps - 1]
            next_t = np.asarray(state["t"], dtype=np.float64)
            if np.any(next_t <= previous_t):
                raise ValueError(
                    "accepted history coordinate time must increase for every particle"
                )

        if not self._accepted_steps:
            for field_name in self._PARTICLE_CONST_FIELDS:
                if field_name not in state:
                    continue
                values = np.asarray(state[field_name])
                if values.shape != (self._n_particles,):
                    raise ValueError(
                        f"{field_name} must have shape ({self._n_particles},)"
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"{field_name} must contain only finite values")

    def validate_append_step(self, state: ParticleState) -> None:
        """Validate a proposed accepted row without changing stored history."""

        self._validate_append_state(state)

    def validate_append_steps(self, states: Sequence[ParticleState]) -> None:
        """Preflight a contiguous row sequence without publishing any row."""

        states = tuple(states)
        previous_time = (
            np.asarray(self._arrays["t"][self._accepted_steps - 1], dtype=np.float64)
            if self._accepted_steps
            else None
        )
        for state in states:
            self._validate_append_state(state)
            next_time = np.asarray(state["t"], dtype=np.float64)
            if previous_time is not None and np.any(next_time <= previous_time):
                raise ValueError(
                    "accepted history coordinate time must increase for every particle"
                )
            previous_time = next_time

    def reserve_append_capacity(self, additional_rows: int = 1) -> None:
        """Ensure a row sequence fits without publishing a history knot."""

        self._ensure_append_capacity(additional_rows)

    def _ensure_append_capacity(self, additional_rows: int = 1) -> None:
        additional_rows = int(additional_rows)
        if additional_rows < 1:
            raise ValueError("additional_rows must be positive")
        required_capacity = self._accepted_steps + additional_rows
        if required_capacity <= self._n_steps:
            return
        grown = max(
            required_capacity,
            int(np.ceil(self._n_steps * self._growth_factor)),
        )
        self._replace_row_capacity(grown)

    def set_step(self, step: int, state: ParticleState) -> None:
        """Reject arbitrary row writes; accepted history is append-only."""

        raise TypeError("GrowableTrajectoryBuilder rows must use append_step()")

    def append_step(self, state: ParticleState) -> int:
        """Validate and append one accepted state, returning its row index."""

        self.validate_append_step(state)
        self.reserve_append_capacity()
        step = self._accepted_steps
        super().set_step(step, state)
        self._accepted_steps += 1
        self._published_stop = self._accepted_steps
        return step

    def build_partial(self, up_to_step: int) -> TrajectoryArrays:
        """Publish only a prefix of rows that were actually accepted."""

        if int(up_to_step) > self._accepted_steps:
            raise ValueError(
                "up_to_step must not exceed the accepted history length "
                f"{self._accepted_steps}"
            )
        return super().build_partial(up_to_step)

    def restore_checkpoint_rows(
        self,
        start: int,
        row_arrays: dict[str, np.ndarray],
        *,
        particle_constants: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Restore one contiguous accepted block, growing before publication."""

        start = int(start)
        if start != self._accepted_steps:
            raise ValueError(
                "checkpoint rows must continue the accepted history contiguously"
            )
        if not row_arrays:
            raise ValueError("checkpoint row block must not be empty")
        first_values = np.asarray(next(iter(row_arrays.values())))
        if first_values.ndim < 1 or int(first_values.shape[0]) < 1:
            raise ValueError("checkpoint row arrays need a non-empty row axis")
        stop = start + int(first_values.shape[0])
        while stop > self._n_steps:
            grown = max(stop, int(np.ceil(self._n_steps * self._growth_factor)))
            self._replace_row_capacity(grown)
        super().restore_checkpoint_rows(
            start,
            row_arrays,
            particle_constants=particle_constants,
        )
        self._accepted_steps = stop

    def build_current(self) -> TrajectoryArrays:
        """Return a managed view containing exactly the accepted rows."""

        if self._accepted_steps < 1:
            raise ValueError("accepted history is empty")
        return self.build_partial(self._accepted_steps)

    def build(self) -> TrajectoryArrays:
        """Return accepted rows rather than unused geometric capacity."""

        return self.build_current()


__all__ = [
    "ParticleState",
    "Trajectory",
    "TrajectoryView",
    "SimulationType",
    "ChronoMatchingMode",
    "StartupMode",
    "IntegratorConfig",
    "CheckpointConfig",
    "DriverTrainConfig",
    "CavityExitConfig",
    "SpaceChargeConfig",
    "ExternalFieldConfig",
    "C_MMNS",
    "TrajectoryArrays",
    "TrialTrajectoryHistory",
    "IndexedTrajectoryArrays",
    "TrajectoryBuilder",
    "GrowableTrajectoryBuilder",
    "StaleTrajectoryViewError",
    "Occluder",
    "BeamlineGeometryConfig",
    "DipoleSourceConfig",
    "MagneticDipoleConfig",
    "MagneticDipoleParticleConfig",
]
