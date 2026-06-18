"""Foundational types shared across integrator modules.

The modern LW core favours explicit type aliases and dataclasses so that both
runtime code and documentation stay readable.  Keeping the definitions in a
single module ensures a consistent contract between physics routines, tests,
and example notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Dict, List, Sequence

import numpy as np

from .constants import C_MMNS

ParticleState = Dict[str, np.ndarray]
Trajectory = List[ParticleState]
TrajectoryView = Sequence[ParticleState]


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
    back-fill of the retarded separation.
    """

    COLD_START = auto()
    APPROXIMATE_BACK_HISTORY = auto()


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
    passive_neighbor_count: int = 4
    coverage_strategy: str = "farthest_point_staleness"
    coverage_space: str = "position"
    pair_reuse_window: int = 16
    source_weighting_mode: str = "inverse_distance"
    loss_tracking_enabled: bool = True
    causal_history_pruning_enabled: bool = False
    causal_history_safety_margin_steps: int = 2

    def __post_init__(self) -> None:
        if self.active_rider_count <= 0:
            raise ValueError("pseudo-grid active_rider_count must be positive")
        if self.active_driver_count <= 0:
            raise ValueError("pseudo-grid active_driver_count must be positive")
        if self.passive_neighbor_count <= 0:
            raise ValueError("pseudo-grid passive_neighbor_count must be positive")
        if self.pair_reuse_window < 0:
            raise ValueError("pseudo-grid pair_reuse_window must be non-negative")
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
        self.axis = tuple(axis_arr / norm)
        self.center_mm = tuple(float(v) for v in self.center_mm)
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
    beamline_geometry: BeamlineGeometryConfig = field(default_factory=BeamlineGeometryConfig)


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
    """Configuration for prescribed uniform external electromagnetic fields.

    Field components use the solver's native units. Electric field components
    are force per native charge, i.e. ``amu * mm / ns^2 / q_native``. Magnetic
    field components are expressed in the same force-per-charge convention and
    enter the Lorentz term as ``beta × B``.

    This first implementation intentionally supports uniform fields with simple
    spatial/temporal windows. More general field maps or callable field
    providers can build on the same integrator hook later.
    """

    enabled: bool = True
    electric_field_native: tuple[float, float, float] = (0.0, 0.0, 0.0)
    magnetic_field_native: tuple[float, float, float] = (0.0, 0.0, 0.0)
    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None
    t_min: float | None = None
    t_max: float | None = None

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
    origin_x: np.ndarray
    origin_y: np.ndarray
    origin_z: np.ndarray
    beta_avg_x: np.ndarray
    beta_avg_y: np.ndarray
    beta_avg_z: np.ndarray
    beta_samples: np.ndarray

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

    # Per-step scalars — [n_steps]
    halted_early: np.ndarray  # dtype bool
    halt_step: np.ndarray  # dtype int64, -1 if not halted

    # Non-array side-channels
    halt_reason: list  # length n_steps, str or None
    particle_failure_info: dict  # keyed by (step, particle_idx)
    pseudo_grid_schedule: list  # length n_steps, object or None

    @property
    def n_steps(self) -> int:
        return self.x.shape[0]

    @property
    def n_particles(self) -> int:
        return self.x.shape[1]

    def state_at(self, step: int) -> ParticleState:
        """Return a legacy ``ParticleState`` dict for *step*."""
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
            s["_halted_early"] = bool(self.halted_early[step])
            s["_halt_step"] = int(self.halt_step[step])
            s["_halt_reason"] = self.halt_reason[step]
        pseudo_grid_schedule = self.pseudo_grid_schedule[step]
        if pseudo_grid_schedule is not None:
            s["_pseudo_grid_schedule"] = pseudo_grid_schedule
        return s

    def to_legacy(self) -> "Trajectory":
        """Return a full ``List[ParticleState]`` compatible with legacy consumers."""
        return [self.state_at(i) for i in range(self.n_steps)]


@dataclass
class IndexedTrajectoryArrays:
    """Particle-indexed view over a :class:`TrajectoryArrays` history."""

    base: TrajectoryArrays
    particle_indices: np.ndarray
    start_step: int = 0
    q_override: np.ndarray | None = None

    def __post_init__(self) -> None:
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
        return np.asarray(getattr(self.base, field_name))[self.global_step(step), :][
            self.particle_indices
        ]

    def scalar(self, field_name: str, step: int, particle_idx: int) -> float:
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
        local_steps = np.asarray(steps, dtype=int)
        local_particles = np.asarray(particle_indices, dtype=int)
        return np.asarray(getattr(self.base, field_name))[
            int(self.start_step) + local_steps,
            self.particle_indices[local_particles],
        ]

    def time_columns(self, up_to_step: int) -> np.ndarray:
        end_step = self.global_step(up_to_step) + 1
        return np.asarray(self.base.t)[int(self.start_step) : end_step, :][
            :, self.particle_indices
        ]

    def constant(self, field_name: str) -> np.ndarray:
        if field_name in {"q", "q_source"} and self.q_override is not None:
            return np.asarray(self.q_override, dtype=float)
        return np.asarray(getattr(self.base, field_name))[self.particle_indices]

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
        if self.base.halted_early[global_step]:
            state["_halted_early"] = bool(self.base.halted_early[global_step])
            state["_halt_step"] = int(self.base.halt_step[global_step])
            state["_halt_reason"] = self.base.halt_reason[global_step]
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
        "origin_x",
        "origin_y",
        "origin_z",
        "beta_avg_x",
        "beta_avg_y",
        "beta_avg_z",
        "beta_samples",
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
    )

    def __init__(self, n_steps: int, n_particles: int) -> None:
        self._n_steps = n_steps
        self._n_particles = n_particles

        self._arrays: dict = {
            field_name: np.zeros((n_steps, n_particles), dtype=np.float64)
            for field_name in self._KINEMATIC_FIELDS
        }
        self._arrays["dead"] = np.zeros((n_steps, n_particles), dtype=bool)

        for field_name in self._PARTICLE_CONST_FIELDS:
            self._arrays[field_name] = np.zeros(n_particles, dtype=np.float64)

        self._halted_early = np.zeros(n_steps, dtype=bool)
        self._halt_step_arr = np.full(n_steps, -1, dtype=np.int64)
        self._halt_reason: list = [None] * n_steps
        self._particle_failure_info: dict = {}
        self._pseudo_grid_schedule: list = [None] * n_steps

    def set_step(self, step: int, state: ParticleState) -> None:
        """Copy *state* fields into row *step* of the pre-allocated arrays."""
        for field_name in self._KINEMATIC_FIELDS:
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
            origin_x=self._arrays["origin_x"][:s],
            origin_y=self._arrays["origin_y"][:s],
            origin_z=self._arrays["origin_z"][:s],
            beta_avg_x=self._arrays["beta_avg_x"][:s],
            beta_avg_y=self._arrays["beta_avg_y"][:s],
            beta_avg_z=self._arrays["beta_avg_z"][:s],
            beta_samples=self._arrays["beta_samples"][:s],
            dead=self._arrays["dead"][:s],
            q=self._arrays["q"],
            q_species=self._arrays["q_species"],
            q_observer=self._arrays["q_observer"],
            q_source=self._arrays["q_source"],
            macro_population=self._arrays["macro_population"],
            m=self._arrays["m"],
            m_species=self._arrays["m_species"],
            char_time=self._arrays["char_time"],
            halted_early=self._halted_early[:s],
            halt_step=self._halt_step_arr[:s],
            halt_reason=self._halt_reason,
            particle_failure_info=self._particle_failure_info,
            pseudo_grid_schedule=self._pseudo_grid_schedule[:s],
        )

    def build(self) -> TrajectoryArrays:
        """Finalise and return the accumulated :class:`TrajectoryArrays`."""
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
            origin_x=self._arrays["origin_x"],
            origin_y=self._arrays["origin_y"],
            origin_z=self._arrays["origin_z"],
            beta_avg_x=self._arrays["beta_avg_x"],
            beta_avg_y=self._arrays["beta_avg_y"],
            beta_avg_z=self._arrays["beta_avg_z"],
            beta_samples=self._arrays["beta_samples"],
            dead=self._arrays["dead"],
            q=self._arrays["q"],
            q_species=self._arrays["q_species"],
            q_observer=self._arrays["q_observer"],
            q_source=self._arrays["q_source"],
            macro_population=self._arrays["macro_population"],
            m=self._arrays["m"],
            m_species=self._arrays["m_species"],
            char_time=self._arrays["char_time"],
            halted_early=self._halted_early,
            halt_step=self._halt_step_arr,
            halt_reason=self._halt_reason,
            particle_failure_info=self._particle_failure_info,
            pseudo_grid_schedule=self._pseudo_grid_schedule,
        )


__all__ = [
    "ParticleState",
    "Trajectory",
    "TrajectoryView",
    "SimulationType",
    "ChronoMatchingMode",
    "StartupMode",
    "IntegratorConfig",
    "DriverTrainConfig",
    "CavityExitConfig",
    "SpaceChargeConfig",
    "ExternalFieldConfig",
    "C_MMNS",
    "TrajectoryArrays",
    "IndexedTrajectoryArrays",
    "TrajectoryBuilder",
    "Occluder",
    "BeamlineGeometryConfig",
]
