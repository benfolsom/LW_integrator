"""Public entry point package for the LW Integrator.

This module re-exports the primary runtime functions provided by the ``core``
package so downstream users can rely on the ``lw_integrator`` namespace.  The
command-line interface lives in :mod:`lw_integrator.cli` and can be invoked via
``python -m lw_integrator`` or the ``lw-simulate`` console script.
"""

from __future__ import annotations

from core import (
    C_MMNS,
    CONVERGENCE_TOLERANCE,
    ELEMENTARY_CHARGE,
    ELECTRON_MASS_AMU,
    NUMERICAL_EPSILON,
    PROTON_MASS_AMU,
    IntegratorConfig,
    ParticleState,
    SimulationType,
    Trajectory,
    retarded_integrator,
    run_integrator,
    trajectory_integrator,
)
from core._version import VERSION, __version__
from .cli import main as cli_main

__all__ = [
    "C_MMNS",
    "CONVERGENCE_TOLERANCE",
    "ELEMENTARY_CHARGE",
    "ELECTRON_MASS_AMU",
    "NUMERICAL_EPSILON",
    "PROTON_MASS_AMU",
    "IntegratorConfig",
    "SimulationType",
    "ParticleState",
    "Trajectory",
    "retarded_integrator",
    "run_integrator",
    "trajectory_integrator",
    "cli_main",
    "__version__",
    "VERSION",
]
