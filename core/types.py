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
]
