"""Foundational types shared across integrator modules.

The modern LW core favours explicit type aliases and dataclasses so that both
runtime code and documentation stay readable.  Keeping the definitions in a
single module ensures a consistent contract between physics routines, tests,
and example notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    - γ_energy from conjugate momentum: γ = (Pt - q·Φ)/(mc)
    - γ_velocity from velocity: γ = 1/√(1-β²)

    These should be identical in exact math but differ numerically due to
    discretization, potentially causing energy jumps and instabilities.

    Reconciliation methods:

    ``DISABLED`` - No reconciliation; use γ_energy directly.
        May cause dual-gamma inconsistency and energy blowups.

    ``ADAPTIVE_WEIGHTED`` - Weighted average with velocity-dependent α (default).
        α = 0.8 (trust energy) for β < 0.9
        α = 0.2 (trust velocity) for β > 0.99
        α = 0.5 (balanced) for 0.9 ≤ β ≤ 0.99
        Provides smooth transition across velocity regimes.

    ``USE_VELOCITY`` - Always use γ_velocity (γ = 1/√(1-β²)).
        Geometrically consistent but can break energy bookkeeping.
        Not recommended for production.

    ``USE_ENERGY`` - Always use γ_energy (γ = (Pt - q·Φ)/(mc)).
        Same as DISABLED; provided for symmetry/clarity.

    ``FIXED_WEIGHTED`` - Fixed 50/50 weighted average.
        γ = 0.5·γ_energy + 0.5·γ_velocity
        Simple but doesn't adapt to physics regime.
    """

    DISABLED = auto()
    ADAPTIVE_WEIGHTED = auto()
    USE_VELOCITY = auto()
    USE_ENERGY = auto()
    FIXED_WEIGHTED = auto()


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
    macroparticle_charge_multiplier: float = 1.0
    macroparticle_sigma_multiplier: float = 1.0
    macroparticle_use_momentum_errors: bool = True
    bunch_transv_dist: float = 0.0
    bunch_transv_mom: float = 0.0


@dataclass
class SpaceChargeConfig:
    """Configuration for intra-bunch space-charge forces.

    When enabled, each rider particle also receives retarded Liénard-Wiechert
    forces from all *other* rider particles (j ≠ i) in addition to the
    driver/image forces already computed.  The Numba hot-path is bypassed
    automatically; the feature uses the same vectorised Python kernel as
    self-consistency and adaptive-timestep modes.
    """

    enabled: bool = True
    retarded: bool = True        # full retarded fields; False → instantaneous Coulomb
    softening_mm: float = 0.0    # Plummer softening ε (mm); 0 = no softening


@dataclass
class TrajectoryArrays:
    """Struct-of-arrays trajectory representation.

    All kinematic fields have shape ``[n_steps, n_particles]``.
    Particle-constant fields (``q``, ``m``, ``char_time``) have shape
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
    m: np.ndarray
    char_time: np.ndarray

    # Per-step scalars — [n_steps]
    halted_early: np.ndarray   # dtype bool
    halt_step: np.ndarray      # dtype int64, -1 if not halted

    # Non-array side-channels
    halt_reason: list           # length n_steps, str or None
    particle_failure_info: dict  # keyed by (step, particle_idx)

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
            "q": self.q,
            "m": self.m,
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
        return s

    def to_legacy(self) -> "Trajectory":
        """Return a full ``List[ParticleState]`` compatible with legacy consumers."""
        return [self.state_at(i) for i in range(self.n_steps)]


class TrajectoryBuilder:
    """Incremental accumulator for building a :class:`TrajectoryArrays`.

    Pre-allocates all arrays at construction time; each integration step
    writes one row via :meth:`set_step`.
    """

    # Fields present in legacy state dicts that map to 2-D kinematic arrays
    _KINEMATIC_FIELDS: tuple = (
        "x", "y", "z", "t",
        "Px", "Py", "Pz", "Pt",
        "gamma",
        "bx", "by", "bz",
        "bdotx", "bdoty", "bdotz",
        "origin_x", "origin_y", "origin_z",
        "beta_avg_x", "beta_avg_y", "beta_avg_z",
        "beta_samples",
    )
    _PARTICLE_CONST_FIELDS: tuple = ("q", "m", "char_time")

    def __init__(self, n_steps: int, n_particles: int) -> None:
        self._n_steps = n_steps
        self._n_particles = n_particles

        self._arrays: dict = {
            field: np.zeros((n_steps, n_particles), dtype=np.float64)
            for field in self._KINEMATIC_FIELDS
        }
        self._arrays["dead"] = np.zeros((n_steps, n_particles), dtype=bool)

        for field in self._PARTICLE_CONST_FIELDS:
            self._arrays[field] = np.zeros(n_particles, dtype=np.float64)

        self._halted_early = np.zeros(n_steps, dtype=bool)
        self._halt_step_arr = np.full(n_steps, -1, dtype=np.int64)
        self._halt_reason: list = [None] * n_steps
        self._particle_failure_info: dict = {}

    def set_step(self, step: int, state: ParticleState) -> None:
        """Copy *state* fields into row *step* of the pre-allocated arrays."""
        for field in self._KINEMATIC_FIELDS:
            if field in state:
                self._arrays[field][step] = state[field]
            # else leave as zero (already pre-allocated)

        dead = state.get("_dead_particles")
        if dead is not None:
            self._arrays["dead"][step] = dead

        if step == 0:
            for field in self._PARTICLE_CONST_FIELDS:
                if field in state:
                    self._arrays[field][:] = state[field]

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

    def set_particle_failure(
        self, step: int, particle_idx: int, info: dict
    ) -> None:
        """Store per-particle failure info keyed by ``(step, particle_idx)``."""
        self._particle_failure_info[(step, particle_idx)] = info

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
            origin_x=self._arrays["origin_x"],
            origin_y=self._arrays["origin_y"],
            origin_z=self._arrays["origin_z"],
            beta_avg_x=self._arrays["beta_avg_x"],
            beta_avg_y=self._arrays["beta_avg_y"],
            beta_avg_z=self._arrays["beta_avg_z"],
            beta_samples=self._arrays["beta_samples"],
            dead=self._arrays["dead"],
            q=self._arrays["q"],
            m=self._arrays["m"],
            char_time=self._arrays["char_time"],
            halted_early=self._halted_early,
            halt_step=self._halt_step_arr,
            halt_reason=self._halt_reason,
            particle_failure_info=self._particle_failure_info,
        )


__all__ = [
    "ParticleState",
    "Trajectory",
    "TrajectoryView",
    "SimulationType",
    "ChronoMatchingMode",
    "StartupMode",
    "IntegratorConfig",
    "SpaceChargeConfig",
    "C_MMNS",
    "TrajectoryArrays",
    "TrajectoryBuilder",
]
