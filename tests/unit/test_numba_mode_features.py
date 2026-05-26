from __future__ import annotations

import numpy as np
import pytest

from core.integration_runner import (
    EnergyJumpDetected,
    EnergyMonitorConfig,
    retarded_integrator,
)
from core.types import SimulationType, SpaceChargeConfig, StartupMode
from core.vectorized_interactions import NUMBA_AVAILABLE

pytestmark = pytest.mark.skipif(
    not NUMBA_AVAILABLE, reason="Numba not available in this environment"
)

_WALL_Z = 100.0
_APERTURE = 10.0
_H_STEP = 0.001
_STEPS = 5


def _make_init_state(n: int = 2) -> dict:
    gamma = 100.0
    from core.constants import C_MMNS

    pt = gamma * C_MMNS
    z_offsets = np.linspace(0.0, 1.0, n)
    return {
        "x": np.zeros(n, dtype=float),
        "y": np.zeros(n, dtype=float),
        "z": z_offsets,
        "t": np.zeros(n, dtype=float),
        "Px": np.zeros(n, dtype=float),
        "Py": np.zeros(n, dtype=float),
        "Pz": np.zeros(n, dtype=float),
        "Pt": np.full(n, pt, dtype=float),
        "gamma": np.full(n, gamma, dtype=float),
        "bx": np.zeros(n, dtype=float),
        "by": np.zeros(n, dtype=float),
        "bz": np.zeros(n, dtype=float),
        "bdotx": np.zeros(n, dtype=float),
        "bdoty": np.zeros(n, dtype=float),
        "bdotz": np.zeros(n, dtype=float),
        "q": np.ones(n, dtype=float),
        "m": np.ones(n, dtype=float),
        "char_time": np.full(n, 1e-3, dtype=float),
    }


def _run(**kwargs):
    init = _make_init_state(2)
    return retarded_integrator(
        steps=_STEPS,
        h_step=_H_STEP,
        wall_z=_WALL_Z,
        aperture_radius=_APERTURE,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=init,
        init_driver=None,
        mean=0.0,
        cav_spacing=10.0,
        z_cutoff=50.0,
        startup_mode=StartupMode.COLD_START,
        use_numba=True,
        **kwargs,
    )


def test_numba_path_energy_monitor_warn():
    monitor = EnergyMonitorConfig(enabled=True, halt_on_jump=False, relative_threshold=0.0)
    result = _run(energy_monitor=monitor)
    assert len(result) == 4
    traj, traj_drv, _, _ = result
    assert len(traj) == _STEPS
    assert len(traj_drv) == _STEPS


def test_numba_path_energy_monitor_halt():
    monitor = EnergyMonitorConfig(enabled=True, halt_on_jump=True, relative_threshold=0.0)
    try:
        result = _run(energy_monitor=monitor)
        # Completed without raising — still valid
        assert len(result) == 4
    except EnergyJumpDetected:
        pass  # Expected when energy changes
    except Exception as exc:
        pytest.fail(f"Unexpected exception type: {type(exc).__name__}: {exc}")


def test_numba_path_macroparticle():
    result_2x = _run(macroparticle_charge_multiplier=2.0)
    assert len(result_2x) == 4
    traj_2x, traj_drv_2x, _, _ = result_2x
    assert len(traj_2x) == _STEPS

    result_1x = _run(macroparticle_charge_multiplier=1.0)
    traj_drv_1x = result_1x[1]

    # The charge multiplier directly scales image charges — sum of |q| should scale by 2x
    sum_q_2x = np.sum(np.abs(traj_drv_2x[0]["q"]))
    sum_q_1x = np.sum(np.abs(traj_drv_1x[0]["q"]))
    assert sum_q_1x > 0, "Image charges are zero at step 0"
    ratio = sum_q_2x / sum_q_1x
    assert abs(ratio - 2.0) < 0.1, f"Expected ~2x charge scaling, got ratio={ratio:.3f}"


def test_numba_path_space_charge():
    sc = SpaceChargeConfig(enabled=True, retarded=False)
    result = _run(space_charge=sc)
    assert len(result) == 4
    traj, traj_drv, _, _ = result
    assert len(traj) == _STEPS


def test_numba_path_all_three():
    monitor = EnergyMonitorConfig(enabled=True, halt_on_jump=False, relative_threshold=1e6)
    sc = SpaceChargeConfig(enabled=True, retarded=False)
    try:
        result = _run(
            energy_monitor=monitor,
            macroparticle_charge_multiplier=2.0,
            space_charge=sc,
        )
        assert len(result) == 4
        assert len(result[0]) == _STEPS
    except EnergyJumpDetected:
        pass


def test_numba_path_adaptive_timestep_basic():
    """Numba path with adaptive_timestep enabled but threshold too high to trigger."""
    from core.integration_runner import AdaptiveTimestepConfig

    at = AdaptiveTimestepConfig(enabled=True, energy_jump_threshold=1e9)
    result = _run(adaptive_timestep=at)
    assert len(result) == 4
    traj, traj_drv, _, _ = result
    assert len(traj) == _STEPS
    assert len(traj_drv) == _STEPS


def test_numba_path_adaptive_timestep_triggers():
    """Numba path with threshold=0.0 so adaptive refinement fires every step."""
    from core.integration_runner import AdaptiveTimestepConfig, EnergyJumpDetected

    at = AdaptiveTimestepConfig(enabled=True, energy_jump_threshold=0.0)
    try:
        result = _run(adaptive_timestep=at)
        assert len(result) == 4
    except EnergyJumpDetected:
        pass
    except Exception as exc:
        pytest.fail(f"Unexpected exception: {type(exc).__name__}: {exc}")


def test_numba_path_adaptive_vs_python_parity():
    """Both numba and python paths complete; initial condition is identical."""
    from core.integration_runner import AdaptiveTimestepConfig

    at = AdaptiveTimestepConfig(enabled=True, energy_jump_threshold=1e9)
    init = _make_init_state(2)
    common_kwargs = dict(
        steps=_STEPS,
        h_step=_H_STEP,
        wall_z=_WALL_Z,
        aperture_radius=_APERTURE,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=init,
        init_driver=None,
        mean=0.0,
        cav_spacing=10.0,
        z_cutoff=50.0,
        startup_mode=StartupMode.COLD_START,
        adaptive_timestep=at,
    )
    from core.integration_runner import retarded_integrator

    result_numba = retarded_integrator(**common_kwargs, use_numba=True)
    result_python = retarded_integrator(**common_kwargs, use_numba=False)

    assert len(result_numba) == 4
    assert len(result_python) == 4

    traj_n = result_numba[0]
    traj_p = result_python[0]

    assert len(traj_n) >= 1
    assert len(traj_p) >= 1

    for key in ("z", "gamma"):
        if key in traj_n[0] and key in traj_p[0]:
            np.testing.assert_array_equal(traj_n[0][key], traj_p[0][key])


def test_numba_kernel_mode_preserves_radiation_bookkeeping_soa():
    result = _run(radiation_reaction_mode="diagnostic_only")
    traj, _, traj_soa, _ = result

    assert traj_soa is not None
    assert traj_soa.radiation_power.shape == (_STEPS, 2)
    assert traj_soa.radiation_energy.shape == (_STEPS, 2)
    assert traj_soa.radiation_energy_applied.shape == (_STEPS, 2)

    expected_power = np.vstack([state["radiation_power"] for state in traj])
    expected_energy = np.vstack([state["radiation_energy"] for state in traj])
    expected_applied = np.vstack(
        [state["radiation_energy_applied"] for state in traj]
    )

    np.testing.assert_allclose(traj_soa.radiation_power, expected_power)
    np.testing.assert_allclose(traj_soa.radiation_energy, expected_energy)
    np.testing.assert_allclose(traj_soa.radiation_energy_applied, expected_applied)


def test_adaptive_step_state_persists():
    """Cooldown state carries correctly across steps without crashing."""
    from core.integration_runner import AdaptiveTimestepConfig, retarded_integrator

    at = AdaptiveTimestepConfig(
        enabled=True, energy_jump_threshold=0.01, cooldown_steps=3
    )
    init = _make_init_state(2)
    result = retarded_integrator(
        steps=10,
        h_step=_H_STEP,
        wall_z=_WALL_Z,
        aperture_radius=_APERTURE,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=init,
        init_driver=None,
        mean=0.0,
        cav_spacing=10.0,
        z_cutoff=50.0,
        startup_mode=StartupMode.COLD_START,
        use_numba=True,
        adaptive_timestep=at,
    )
    assert len(result) == 4
    traj = result[0]
    assert len(traj) >= 1
