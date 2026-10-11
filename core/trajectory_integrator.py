"""Compatibility facade exposing the retarded integrator public API.

Earlier project layouts imported functionality directly from
``core.trajectory_integrator``. The implementation has since been modularised
into smaller, focused modules. This wrapper re-exports the public symbols to
maintain function import compatibility. The historical mutable class is retired;
use the function runners with an explicit configuration instead.
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
from .integration_runner import (
    AdaptiveTimestepConfig,
    EnergyMonitorConfig,
    retarded_integrator,
    run_integrator,
)
from .types import (
    ChronoMatchingMode,
    DriverTrainConfig,
    IntegratorConfig,
    ParticleLossConfig,
    ParticleState,
    SimulationType,
    StartupMode,
    Trajectory,
)


def __getattr__(name: str):
    if name == "LienardWiechertIntegrator":
        import warnings

        warnings.warn(
            "LienardWiechertIntegrator has been removed; use retarded_integrator "
            "or run_integrator. The historical drift_step used lab time, whereas "
            "the modern runner uses proper time.",
            FutureWarning,
            stacklevel=2,
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "C_MMNS",
    "AdaptiveTimestepConfig",
    "ChronoMatchingMode",
    "DistanceResult",
    "DriverTrainConfig",
    "EnergyMonitorConfig",
    "IntegratorConfig",
    "ParticleLossConfig",
    "ParticleState",
    "SimulationType",
    "StartupMode",
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
