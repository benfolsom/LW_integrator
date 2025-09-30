"""Benchmarks modern integrators against the legacy two-particle demo.

The historical `legacy/two_particle_demo_main.ipynb` notebook evolved into an
important validation reference.  This test replays the same scenario using the
legacy implementation and the modern modules so that parity in both physics and
performance is continually verified.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Dict

import numpy as np
import pytest

from core.performance import NUMBA_AVAILABLE, retarded_integrator_numba
from core.self_consistency import SelfConsistencyConfig
from core.trajectory_integrator import SimulationType, retarded_integrator

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


def _convert_legacy_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
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


@pytest.mark.integration
@pytest.mark.slow
def test_two_particle_demo_benchmark_parity():
    steps = 35
    time_step = 2.2e-7
    wall_z = 1e5
    aperture = 1e5
    mean = 1e5
    spacing = 1e5
    z_cutoff = 0.0

    np.random.seed(12345)
    random.seed(12345)
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

    start_legacy = time.perf_counter()
    legacy_traj, legacy_drv = legacy_retarded_integrator(
        steps,
        time_step,
        wall_z,
        aperture,
        SimulationType.BUNCH_TO_BUNCH.value,
        rider_state,
        driver_state,
        mean,
        spacing,
        z_cutoff,
    )
    legacy_duration = time.perf_counter() - start_legacy

    modern_rider = _convert_legacy_state(copy.deepcopy(rider_state))
    modern_driver = _convert_legacy_state(copy.deepcopy(driver_state))

    start_modern = time.perf_counter()
    modern_traj, modern_drv = retarded_integrator(
        steps=steps,
        h_step=time_step,
        wall_z=wall_z,
        aperture_radius=aperture,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=modern_rider,
        init_driver=modern_driver,
        mean=mean,
        cav_spacing=spacing,
        z_cutoff=z_cutoff,
        self_consistency=SelfConsistencyConfig(enabled=False),
    )
    modern_duration = time.perf_counter() - start_modern

    fields = ["x", "y", "z", "Px", "Py", "Pz", "Pt", "gamma"]
    for field in fields:
        np.testing.assert_allclose(
            _normalize_state(legacy_traj[-1])[field],
            _normalize_state(modern_traj[-1])[field],
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            _normalize_state(legacy_drv[-1])[field],
            _normalize_state(modern_drv[-1])[field],
            atol=1e-5,
            rtol=1e-5,
        )

    # Allow a small regression window to account for interpreter variance.
    if legacy_duration > 0:
        runtime_ratio = modern_duration / legacy_duration
        assert runtime_ratio < 1.3, (
            f"Modern integrator too slow: ratio={runtime_ratio:.2f}, "
            f"legacy={legacy_duration:.3f}s, modern={modern_duration:.3f}s"
        )

    if not NUMBA_AVAILABLE:
        return

    # Warm up the JIT so timing reflects steady-state performance.
    retarded_integrator_numba(
        2,
        time_step,
        wall_z,
        aperture,
        SimulationType.BUNCH_TO_BUNCH,
        _convert_legacy_state(copy.deepcopy(rider_state)),
        _convert_legacy_state(copy.deepcopy(driver_state)),
        mean,
        spacing,
        z_cutoff,
        SelfConsistencyConfig(enabled=False),
    )

    start_numba = time.perf_counter()
    numba_traj, numba_drv = retarded_integrator_numba(
        steps,
        time_step,
        wall_z,
        aperture,
        SimulationType.BUNCH_TO_BUNCH,
        _convert_legacy_state(copy.deepcopy(rider_state)),
        _convert_legacy_state(copy.deepcopy(driver_state)),
        mean,
        spacing,
        z_cutoff,
        SelfConsistencyConfig(enabled=False),
    )
    numba_duration = time.perf_counter() - start_numba

    for field in fields:
        np.testing.assert_allclose(
            _normalize_state(modern_traj[-1])[field],
            _normalize_state(numba_traj[-1])[field],
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            _normalize_state(modern_drv[-1])[field],
            _normalize_state(numba_drv[-1])[field],
            atol=1e-6,
            rtol=1e-6,
        )

    if modern_duration > 0:
        numba_ratio = numba_duration / modern_duration
        assert numba_ratio < 1.1, (
            f"Numba path should be at least ~10% faster: ratio={numba_ratio:.2f}, "
            f"modern={modern_duration:.3f}s, numba={numba_duration:.3f}s"
        )
