"""Regression tests for the core Liénard–Wiechert integrator implementations."""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LEGACY_ROOT = PROJECT_ROOT / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from core.performance import NUMBA_AVAILABLE, retarded_integrator_numba
from core.self_consistency import SelfConsistencyConfig
from core.trajectory_integrator import (
    IntegratorConfig,
    ParticleState,
    SimulationType,
    retarded_integrator,
)
from input_output.bunch_initialization import create_bunch_from_energy

from legacy.bunch_inits import init_bunch as legacy_init_bunch  # type: ignore
from legacy.covariant_integrator_library import (  # type: ignore
    retarded_integrator as legacy_retarded_integrator,
)


def _normalize_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    normalised: Dict[str, np.ndarray] = {}
    length = len(state.get("x", np.atleast_1d(state.get("Px", [0]))))
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            normalised[key] = value
        elif np.isscalar(value):
            normalised[key] = np.full(length, value, dtype=float)
        else:
            normalised[key] = np.asarray(value, dtype=float)
    return normalised


def _convert_legacy_state(state: Dict[str, np.ndarray]) -> ParticleState:
    converted = {}
    length = len(state["x"])
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            converted[key] = value
        elif key in {"q", "m", "char_time"}:
            converted[key] = np.full(length, value, dtype=float)
        else:
            converted[key] = np.asarray(value, dtype=float)
    return converted  # type: ignore[return-value]


def _compare_states(
    reference: Dict[str, np.ndarray],
    candidate: Dict[str, np.ndarray],
    fields: Iterable[str],
    atol: float = 1e-6,
    rtol: float = 1e-6,
):
    ref_norm = _normalize_state(reference)
    cand_norm = _normalize_state(candidate)
    for field in fields:
        if field not in ref_norm or field not in cand_norm:
            continue
        np.testing.assert_allclose(
            cand_norm[field], ref_norm[field], atol=atol, rtol=rtol, err_msg=f"Mismatch for field {field}"
        )


@pytest.mark.parametrize(
    "steps",
    [40],
)
def test_two_particle_demo_precision(steps: int):
    np.random.seed(12345)
    rider_state, _ = legacy_init_bunch(
        starting_distance=1e-6,
        transv_mom=0.0,
        starting_Pz=1.01e6,
        stripped_ions=1.0,
        m_particle=1.007319468,
        transv_dist=2e-4,
        pcount=5,
        charge_sign=-1.0,
    )
    driver_state, _ = legacy_init_bunch(
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
        steps,
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
        steps=steps,
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

    fields = ["x", "y", "z", "Px", "Py", "Pz", "Pt", "gamma"]
    _compare_states(legacy_traj[-1], core_traj[-1], fields, atol=1e-5, rtol=1e-5)
    _compare_states(legacy_drv[-1], core_drv[-1], fields, atol=1e-5, rtol=1e-5)

    if NUMBA_AVAILABLE:
        numba_traj, numba_drv = retarded_integrator_numba(
            steps,
            2.2e-7,
            1e5,
            1e5,
            SimulationType.BUNCH_TO_BUNCH,
            _convert_legacy_state(copy.deepcopy(rider_state)),
            _convert_legacy_state(copy.deepcopy(driver_state)),
            1e5,
            1e5,
            0.0,
            SelfConsistencyConfig(enabled=False),
        )
    _compare_states(core_traj[-1], numba_traj[-1], fields, atol=1e-6, rtol=1e-6)
    _compare_states(core_drv[-1], numba_drv[-1], fields, atol=1e-6, rtol=1e-6)

    sc_traj, sc_drv = retarded_integrator(
        steps=steps,
        h_step=2.2e-7,
        wall_z=1e5,
        aperture_radius=1e5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_convert_legacy_state(copy.deepcopy(rider_state)),
        init_driver=_convert_legacy_state(copy.deepcopy(driver_state)),
        mean=1e5,
        cav_spacing=1e5,
        z_cutoff=0.0,
        self_consistency=SelfConsistencyConfig(enabled=True, tolerance=1e-9, max_iterations=4),
    )
    _compare_states(core_traj[-1], sc_traj[-1], fields, atol=1e-7, rtol=1e-7)
    _compare_states(core_drv[-1], sc_drv[-1], fields, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(
    "particle_mass, charge_sign, energy, sim_type",
    [
        (1.007276, 1, 200.0, SimulationType.CONDUCTING_WALL),
        (0.0005485, -1, 10.0, SimulationType.SWITCHING_WALL),
    ],
)
def test_integrator_modes_with_numba(particle_mass, charge_sign, energy, sim_type):
    rider_state, _ = create_bunch_from_energy(
        kinetic_energy_mev=energy,
        mass_amu=particle_mass,
        charge_sign=charge_sign,
        position_z=0.0,
        particle_count=1,
    )

    config = IntegratorConfig(
        steps=12,
        time_step=1e-5,
        wall_position=5.0,
        aperture_radius=1.0,
        simulation_type=sim_type,
        bunch_mean=0.0,
        cavity_spacing=10.0,
        z_cutoff=0.5,
    )

    random.seed(20250930)
    base_traj, base_drv = retarded_integrator(
        steps=config.steps,
        h_step=config.time_step,
        wall_z=config.wall_position,
        aperture_radius=config.aperture_radius,
        sim_type=config.simulation_type,
        init_rider=rider_state,
        init_driver=None,
        mean=config.bunch_mean,
        cav_spacing=config.cavity_spacing,
        z_cutoff=config.z_cutoff,
    )

    if NUMBA_AVAILABLE:
        random.seed(20250930)
        numba_traj, numba_drv = retarded_integrator_numba(
            config.steps,
            config.time_step,
            config.wall_position,
            config.aperture_radius,
            config.simulation_type,
            rider_state,
            None,
            config.bunch_mean,
            config.cavity_spacing,
            config.z_cutoff,
            SelfConsistencyConfig(enabled=False),
        )
        fields = ["x", "y", "z", "Px", "Py", "Pz", "Pt"]
        _compare_states(base_traj[-1], numba_traj[-1], fields, atol=1e-6, rtol=1e-6)
        _compare_states(base_drv[-1], numba_drv[-1], fields, atol=1e-6, rtol=1e-6)
    else:
        pytest.skip("Numba runtime unavailable")
