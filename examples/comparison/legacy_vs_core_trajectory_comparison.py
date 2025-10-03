#!/usr/bin/env python3
"""Visual comparison between the legacy notebook scenario and the core integrator."""

from __future__ import annotations

import copy
import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LEGACY_ROOT = os.path.join(REPO_ROOT, "legacy")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if LEGACY_ROOT not in sys.path:
    sys.path.insert(0, LEGACY_ROOT)

from core.trajectory_integrator import SimulationType, retarded_integrator
from legacy.bunch_inits import init_bunch  # type: ignore
from legacy.covariant_integrator_library import (  # type: ignore
    retarded_integrator as legacy_retarded_integrator,
)


def _convert_legacy_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Cast a legacy particle dictionary into the layout expected by the core code."""

    converted: Dict[str, np.ndarray] = {}
    length = len(state["x"])
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            converted[key] = value
        elif key in {"q", "m", "char_time"}:
            converted[key] = np.full(length, value, dtype=float)
        else:
            converted[key] = np.asarray(value, dtype=float)
    return converted


def run_two_particle_demo() -> Tuple[
    Tuple[np.ndarray, ...],
    Tuple[np.ndarray, ...],
    Tuple[np.ndarray, ...],
    Tuple[np.ndarray, ...],
]:
    """Execute the canonical two-particle demo for legacy and core integrators."""

    np.random.seed(12345)

    rider_state, _ = init_bunch(
        starting_distance=1e-6,
        transv_mom=0.0,
        starting_Pz=1.01e6,
        stripped_ions=1.0,
        m_particle=1.007319468,
        transv_dist=2e-4,
        pcount=5,
        charge_sign=-1.0,
    )
    driver_state, _ = init_bunch(
        starting_distance=1000.0,
        transv_mom=0.0,
        starting_Pz=-1.01e6 / 207.2 * 1.007319468,
        stripped_ions=54.0,
        m_particle=207.2,
        transv_dist=2e-4 - 8e-2,
        pcount=5,
        charge_sign=1.0,
    )

    legacy_traj, legacy_drv = legacy_retarded_integrator(
        40,
        2.2e-7,
        1e5,
        1e5,
        2,
        rider_state,
        driver_state,
        1e5,
        1e5,
        0.0,
    )

    core_traj, core_drv = retarded_integrator(
        steps=40,
        h_step=2.2e-7,
        wall_z=1e5,
        aperture_radius=1e5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_convert_legacy_state(copy.deepcopy(rider_state)),
        init_driver=_convert_legacy_state(copy.deepcopy(driver_state)),
        mean=1e5,
        cav_spacing=1e5,
        z_cutoff=0.0,
    )

    return legacy_traj, legacy_drv, core_traj, core_drv


def _plot_position_series(states, label: str, style: str = "-") -> None:
    data = [state["z"][0] for state in states]
    plt.plot(data, style, label=label)


def _plot_gamma_series(states, reference, label: str) -> None:
    diff = [
        core["gamma"][0] - legacy["gamma"][0] for legacy, core in zip(reference, states)
    ]
    plt.plot(diff, label=label)


def main() -> None:
    legacy_traj, legacy_drv, core_traj, core_drv = run_two_particle_demo()

    plt.figure(figsize=(12, 6))
    _plot_position_series(legacy_traj, "Legacy rider", "--")
    _plot_position_series(core_traj, "Core rider")
    _plot_position_series(legacy_drv, "Legacy driver", "--")
    _plot_position_series(core_drv, "Core driver")
    plt.title("Legacy vs Core trajectory overlap (z position)")
    plt.xlabel("Step")
    plt.ylabel("z (mm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    _plot_gamma_series(core_traj, legacy_traj, "Δγ rider")
    _plot_gamma_series(core_drv, legacy_drv, "Δγ driver")
    plt.title("Gamma difference between legacy and core trajectories")
    plt.xlabel("Step")
    plt.ylabel("Δγ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
