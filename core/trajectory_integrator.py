"""Compatibility facade exposing the retarded integrator public API.

The legacy codebase imported functionality directly from
``core.trajectory_integrator``. The implementation has since been modularised
into smaller, focused modules. This wrapper re-exports the public symbols to
maintain import compatibility while also providing a light-weight class based
API used by historical notebooks and regression tests.
"""

from __future__ import annotations

from typing import Any, MutableMapping, Optional

import numpy as np

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


class LienardWiechertIntegrator:
    """Compatibility wrapper mimicking the legacy integrator facade.

    The historic API exposed a mutable class with helper methods that several
    notebooks and regression tests relied upon. The present codebase is
    function-oriented, but a class wrapper keeps downstream integrations
    working without modification.
    """

    def __init__(self, config: Optional[IntegratorConfig] = None) -> None:
        self.c_mmns = C_MMNS
        self.config = config

    @staticmethod
    def _clone_state(state: ParticleState | None) -> ParticleState | None:
        """Return a deep copy of a particle state dictionary.

        ``None`` is propagated through to support optional driver states.
        """

        if state is None:
            return None

        return {key: np.copy(value) for key, value in state.items()}

    def equations_of_motion_static_internal(
        self, h_step: float, state: ParticleState, _index: int
    ) -> ParticleState:
        """Propagate particles forward assuming a field-free drift.

        This mirrors the legacy helper used for static warm-up steps: particles
        drift according to their current velocity while conjugate momentum and
        Lorentz factor remain constant.
        """

        result = self._clone_state(state)

        if "bx" in state and "x" in state:
            result["x"] = state["x"] + state["bx"] * C_MMNS * h_step
        if "by" in state and "y" in state:
            result["y"] = state["y"] + state["by"] * C_MMNS * h_step
        if "bz" in state and "z" in state:
            result["z"] = state["z"] + state["bz"] * C_MMNS * h_step

        if "t" in state:
            result["t"] = state["t"] + h_step

        for key in ("Px", "Py", "Pz", "Pt", "gamma"):
            if key in state:
                result[key] = np.copy(state[key])

        for key in ("bdotx", "bdoty", "bdotz"):
            if key in state:
                result[key] = np.zeros_like(state[key])

        return result

    def integrate_retarded_fields(
        self,
        static_steps: int,
        ret_steps: int,
        h_step: float,
        wall_Z: float,
        apt_R: float,
        sim_type: SimulationType | int,
        init_rider: ParticleState,
        init_driver: Optional[ParticleState],
        bunch_dist: float,
        z_cutoff: float,
        *,
        extra_config: Optional[MutableMapping[str, Any]] = None,
    ) -> tuple[Trajectory, Trajectory]:
        """Execute retarded-field integration using the modern core runner.

        Parameters mirror the legacy signature but are forwarded to
        :func:`core.integrator.retarded_integrator`. ``static_steps`` is
        preserved for backwards compatibility and contributes to the total step
        count.
        """

        sim_type_enum = (
            sim_type
            if isinstance(sim_type, SimulationType)
            else SimulationType(int(sim_type))
        )

        total_steps = max(int(static_steps) + int(ret_steps), 1)

        rider_state = self._clone_state(init_rider)
        driver_state = self._clone_state(init_driver)

        if extra_config:
            wall_Z = float(extra_config.get("wall_Z", wall_Z))
            apt_R = float(extra_config.get("apt_R", apt_R))
            bunch_dist = float(extra_config.get("bunch_dist", bunch_dist))
            z_cutoff = float(extra_config.get("z_cutoff", z_cutoff))

        trajectory, driver = retarded_integrator(
            steps=total_steps,
            h_step=float(h_step),
            wall_z=float(wall_Z),
            aperture_radius=float(apt_R),
            sim_type=sim_type_enum,
            init_rider=rider_state,
            init_driver=driver_state,
            mean=float(bunch_dist),
            cav_spacing=0.0,
            z_cutoff=float(z_cutoff),
        )

        return trajectory, driver


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
    "LienardWiechertIntegrator",
]
