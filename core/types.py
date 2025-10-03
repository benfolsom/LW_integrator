"""Foundational types shared across integrator modules.

The modern LW core favours explicit type aliases and dataclasses so that both
runtime code and documentation stay readable.  Keeping the definitions in a
single module ensures a consistent contract between physics routines, tests,
and example notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Sequence

import numpy as np

from .constants import C_MMNS

ParticleState = Dict[str, np.ndarray]
Trajectory = List[ParticleState]
TrajectoryView = Sequence[ParticleState]


class SimulationType(IntEnum):
    """Supported simulation modes (mirrors the legacy integer flags).

    The enum inherits from :class:`int` so that existing code using literal
    values (``0``, ``1``, ``2``) continues to work.  When writing new code,
    prefer the descriptive enum members for readability.
    """

    CONDUCTING_WALL = 0
    SWITCHING_WALL = 1
    BUNCH_TO_BUNCH = 2


@dataclass
class IntegratorConfig:
    """Structured configuration for :func:`core.integrator.run_integrator`.

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
    bunch_mean:
        Optional mean bunch separation used by legacy notebooks. Not every
        integration path consumes it, but the value is retained for API
        compatibility.
    cavity_spacing:
        Distance between cavities used by the switching-wall configuration.
    z_cutoff:
        Longitudinal position at which the switching-wall stops mirroring
        charges. Defaults to ``0`` which effectively disables the cutoff.
    """

    steps: int
    time_step: float
    wall_position: float
    aperture_radius: float
    simulation_type: SimulationType
    bunch_mean: float = 0.0
    cavity_spacing: float = 0.0
    z_cutoff: float = 0.0


__all__ = [
    "ParticleState",
    "Trajectory",
    "TrajectoryView",
    "SimulationType",
    "IntegratorConfig",
    "C_MMNS",
]
