"""Compatibility layer exposing the retarded integrator public API.

The legacy codebase imported functionality directly from
``core.trajectory_integrator``.  The implementation has since been modularised
into smaller, focused modules.  This wrapper re-exports the public symbols to
maintain import compatibility while pointing callers at the new code layout.
"""

from __future__ import annotations

from .constants import C_MMNS
from .distances import (
    DistanceResult,
    chrono_match_indices,
    compute_instantaneous_distance,
    compute_retarded_distance,
)
from .equations import retarded_equations_of_motion
from .images import generate_conducting_image, generate_switching_image
from .integrator import retarded_integrator, run_integrator
from .types import IntegratorConfig, ParticleState, SimulationType, Trajectory

__all__ = [
    "C_MMNS",
    "DistanceResult",
    "IntegratorConfig",
    "ParticleState",
    "SimulationType",
    "Trajectory",
    "chrono_match_indices",
    "compute_instantaneous_distance",
    "compute_retarded_distance",
    "generate_conducting_image",
    "generate_switching_image",
    "retarded_equations_of_motion",
    "retarded_integrator",
    "run_integrator",
]
