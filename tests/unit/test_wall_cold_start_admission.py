"""COLD_START wall runs admit the image when light from its first event arrives."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.constants import C_MMNS
from core.integration_runner import retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.types import SimulationType, StartupMode
from tests.unit.test_potential_inclusion import state

WALL_Z = 200.0
STEPS = 300
DT_LAB = 0.005


def _run(sim_type, beta, startup=StartupMode.COLD_START):
    rider = state(x=0.5, z=0.0, beta=beta)
    gamma = float(rider["gamma"][0])
    config = SelfConsistencyConfig(
        enabled=True,
        convergence_mode="fixed_geometry",
        max_iterations=2,
        target_ms_tolerance=1e-6,
        mass_shell_tolerance=0.01,
        chrono_interpolate=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trajectory, *_ = retarded_integrator(
            steps=STEPS,
            h_step=DT_LAB / gamma,
            wall_z=WALL_Z,
            aperture_radius=1.0,
            mean=1000,
            cav_spacing=1e6,
            z_cutoff=150.0,
            sim_type=sim_type,
            init_rider=rider,
            init_driver=None,
            startup_mode=startup,
            radiation_reaction_mode="medina_lad",
            self_consistency=config,
            allow_incomplete_startup_history=True,
        )
    return {
        key: np.array([np.asarray(s[key], dtype=float)[0] for s in trajectory])
        for key in ("t", "Px", "Pz")
    }


@pytest.mark.parametrize("beta, reference_dpx", [(0.5, -1.197e-7), (0.9, None)])
def test_conducting_wall_image_force_starts_at_light_arrival(beta, reference_dpx):
    result = _run(SimulationType.CONDUCTING_WALL, beta)
    # The image of the first state sits at z = 2 WALL_Z. A rider moving toward
    # it meets the light from its creation at t = 2 WALL_Z / (c (1 + beta)).
    t_arrival = 2.0 * WALL_Z / (C_MMNS * (1.0 + beta))
    kicked = np.flatnonzero(
        (np.abs(np.diff(result["Px"])) > 0) | (np.abs(np.diff(result["Pz"])) > 0)
    )
    # Pz also changes from radiation only if a force acts, so any change counts.
    assert kicked.size > 0, "image force never applied: wall run is free flight"
    first_kick_time = result["t"][kicked[0]]
    assert first_kick_time >= t_arrival - 2 * DT_LAB
    assert first_kick_time <= t_arrival + 2 * DT_LAB
    assert np.all(result["Px"][: kicked[0] + 1] == result["Px"][0])
    if reference_dpx is not None:
        # The late-time image force matches the pre-change (ungated) run.
        dpx = result["Px"][-1] - result["Px"][0]
        assert dpx == pytest.approx(reference_dpx, rel=0.02)


def test_switching_wall_image_force_not_applied_before_light_arrival():
    result = _run(SimulationType.SWITCHING_WALL, 0.5)
    t_arrival = 2.0 * WALL_Z / (C_MMNS * 1.5)
    kicked = np.flatnonzero(np.abs(np.diff(result["Pz"])) > 0)
    assert kicked.size > 0
    assert result["t"][kicked[0]] >= t_arrival - 2 * DT_LAB
