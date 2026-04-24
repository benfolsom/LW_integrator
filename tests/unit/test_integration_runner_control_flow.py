from __future__ import annotations

import copy

import numpy as np
import pytest

import core.integration_runner as integration_runner
from core.integration_runner import (
    AdaptiveTimestepConfig,
    EnergyJumpDetected,
    EnergyMonitorConfig,
    retarded_integrator,
)
from core.types import SimulationType


def _make_particle_state(
    *,
    z: float = 0.0,
    gamma: float = 1.0,
    charge: float = 1.0,
    mass: float = 1.0,
) -> dict[str, np.ndarray]:
    pt = gamma * mass * integration_runner.C_MMNS
    return {
        "x": np.array([0.0], dtype=float),
        "y": np.array([0.0], dtype=float),
        "z": np.array([z], dtype=float),
        "t": np.array([0.0], dtype=float),
        "Px": np.array([0.0], dtype=float),
        "Py": np.array([0.0], dtype=float),
        "Pz": np.array([0.0], dtype=float),
        "Pt": np.array([pt], dtype=float),
        "gamma": np.array([gamma], dtype=float),
        "bx": np.array([0.0], dtype=float),
        "by": np.array([0.0], dtype=float),
        "bz": np.array([0.0], dtype=float),
        "bdotx": np.array([0.0], dtype=float),
        "bdoty": np.array([0.0], dtype=float),
        "bdotz": np.array([0.0], dtype=float),
        "q": np.array([charge], dtype=float),
        "m": np.array([mass], dtype=float),
        "char_time": np.array([1e-3], dtype=float),
    }


def _clone_state(
    state: dict[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            result[key] = value.copy()
        elif isinstance(value, dict):
            result[key] = copy.deepcopy(value)
        else:
            result[key] = value
    return result


def test_retarded_integrator_energy_monitor_raises_on_large_jump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_step(
        step_function: object,
        h_step: float,
        trajectory: list[dict[str, object]],
        trajectory_ext: list[dict[str, object]],
        index_traj: int,
        aperture_radius: float,
        sim_type: object,
        config: object,
        chrono_mode: object,
        startup_mode: object,
        step_idx: int | None = None,
        cancel_callback: object = None,
    ) -> dict[str, object]:
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        gamma = 1.0 if step_idx == 1 else 3.0
        state["gamma"] = np.array([gamma], dtype=float)
        state["Pt"] = np.array([gamma * integration_runner.C_MMNS], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    with pytest.raises(EnergyJumpDetected):
        retarded_integrator(
            steps=3,
            h_step=1.0,
            wall_z=100.0,
            aperture_radius=1.0,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=_make_particle_state(z=-1.0, gamma=1.0),
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            energy_monitor=EnergyMonitorConfig(
                enabled=True,
                relative_threshold=0.5,
                halt_on_jump=True,
            ),
            use_numba=False,
        )


def test_retarded_integrator_adaptive_retry_uses_reduced_timestep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h_calls: list[tuple[int | None, float]] = []

    def fake_step(
        step_function: object,
        h_step: float,
        trajectory: list[dict[str, object]],
        trajectory_ext: list[dict[str, object]],
        index_traj: int,
        aperture_radius: float,
        sim_type: object,
        config: object,
        chrono_mode: object,
        startup_mode: object,
        step_idx: int | None = None,
        cancel_callback: object = None,
    ) -> dict[str, object]:
        h_calls.append((step_idx, h_step))
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)

        if step_idx == 1:
            gamma = 10.0
        elif h_step >= 1.0:
            gamma = 20.0
        else:
            gamma = 10.5

        state["gamma"] = np.array([gamma], dtype=float)
        state["Pt"] = np.array([gamma * integration_runner.C_MMNS], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, _ = retarded_integrator(
        steps=3,
        h_step=1.0,
        wall_z=100.0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-1.0, gamma=1.0),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        adaptive_timestep=AdaptiveTimestepConfig(
            enabled=True,
            energy_jump_threshold=0.2,
            timestep_reduction_factor=2,
            min_timestep_factor=0.25,
            proximity_refinement_enabled=False,
        ),
        use_numba=False,
    )

    step_2_calls = [h for step, h in h_calls if step == 2]

    assert step_2_calls == [1.0, 0.5, 0.5]
    assert trajectory[-1]["t"][0] == pytest.approx(2.0)
    assert trajectory[-1]["gamma"][0] == pytest.approx(10.5)


def test_retarded_integrator_marks_relative_cutoff_early_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_step(
        step_function: object,
        h_step: float,
        trajectory: list[dict[str, object]],
        trajectory_ext: list[dict[str, object]],
        index_traj: int,
        aperture_radius: float,
        sim_type: object,
        config: object,
        chrono_mode: object,
        startup_mode: object,
        step_idx: int | None = None,
        cancel_callback: object = None,
    ) -> dict[str, object]:
        source_state = trajectory[index_traj] if trajectory[index_traj] else trajectory[-1]
        state = _clone_state(source_state)
        state["z"] = np.array([float(source_state["z"][0]) + 2.0], dtype=float)
        state["t"] = np.array([float(source_state["t"][0]) + h_step], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    trajectory, driver = retarded_integrator(
        steps=5,
        h_step=1.0,
        wall_z=0.0,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_particle_state(z=0.0),
        init_driver=_make_particle_state(z=5.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.5,
        z_cutoff_mode="relative",
        use_numba=False,
    )

    assert len(trajectory) == 2
    assert len(driver) == 2
    assert trajectory[-1]["_halted_early"] is True
    assert trajectory[-1]["_halt_step"] == 1
    assert "distance_reached" in trajectory[-1]["_halt_reason"]


def test_retarded_integrator_halts_when_all_particles_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_step(
        step_function: object,
        h_step: float,
        trajectory: list[dict[str, object]],
        trajectory_ext: list[dict[str, object]],
        index_traj: int,
        aperture_radius: float,
        sim_type: object,
        config: object,
        chrono_mode: object,
        startup_mode: object,
        step_idx: int | None = None,
        cancel_callback: object = None,
    ) -> dict[str, object]:
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        state["q"] = np.array([0.0], dtype=float)
        state["_dead_particles"] = np.array([True], dtype=bool)
        state["_particle_failure_info"] = {
            0: {"step": step_idx, "reason": "synthetic_failure"}
        }
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, driver = retarded_integrator(
        steps=4,
        h_step=1.0,
        wall_z=100.0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-1.0),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        use_numba=False,
    )

    assert len(trajectory) == 2
    assert len(driver) == 2
    assert trajectory[-1]["_halted_early"] is True
    assert trajectory[-1]["_halt_step"] == 1
    assert "all_particles_dead" in trajectory[-1]["_halt_reason"]
