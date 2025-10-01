"""Top-level exports for the Liénard–Wiechert integrator core."""

from __future__ import annotations

from . import trajectory_integrator
from ._version import VERSION, __version__
from .constants import (
	C_MMNS,
	CONVERGENCE_TOLERANCE,
	ELECTRON_MASS_AMU,
	ELEMENTARY_CHARGE,
	NUMERICAL_EPSILON,
	PROTON_MASS_AMU,
)
from .integrator import retarded_integrator, run_integrator
from .types import IntegratorConfig, ParticleState, SimulationType, Trajectory

__all__ = [
	"trajectory_integrator",
	"retarded_integrator",
	"run_integrator",
	"IntegratorConfig",
	"SimulationType",
	"ParticleState",
	"Trajectory",
	"C_MMNS",
	"ELEMENTARY_CHARGE",
	"ELECTRON_MASS_AMU",
	"PROTON_MASS_AMU",
	"NUMERICAL_EPSILON",
	"CONVERGENCE_TOLERANCE",
	"__version__",
	"VERSION",
]
