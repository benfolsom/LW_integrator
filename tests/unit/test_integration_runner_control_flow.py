from __future__ import annotations

import copy

import numpy as np
import pytest

import core.integration_runner as integration_runner
from core.equations import GammaBlowupError
from core.pseudo_grid import PseudoGridStepSchedule
from core.integration_runner import (
    AdaptiveTimestepConfig,
    EnergyJumpDetected,
    EnergyMonitorConfig,
    IntegratorConfig,
    retarded_integrator,
    run_integrator,
)
from core.types import CavityExitConfig, PseudoGridConfig, SimulationType, SpaceChargeConfig


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


def _make_bunch_state(
    *,
    x: list[float],
    z: float = 0.0,
    charge: float = 1.0,
    mass: float = 1.0,
) -> dict[str, np.ndarray]:
    n_particles = len(x)
    gamma = np.ones(n_particles, dtype=float)
    pt = gamma * mass * integration_runner.C_MMNS
    zeros = np.zeros(n_particles, dtype=float)
    return {
        "x": np.asarray(x, dtype=float),
        "y": zeros.copy(),
        "z": np.full(n_particles, z, dtype=float),
        "t": zeros.copy(),
        "Px": zeros.copy(),
        "Py": zeros.copy(),
        "Pz": zeros.copy(),
        "Pt": pt,
        "gamma": gamma,
        "bx": zeros.copy(),
        "by": zeros.copy(),
        "bz": zeros.copy(),
        "bdotx": zeros.copy(),
        "bdoty": zeros.copy(),
        "bdotz": zeros.copy(),
        "q": np.full(n_particles, charge, dtype=float),
        "m": np.full(n_particles, mass, dtype=float),
        "char_time": np.full(n_particles, 1e-3, dtype=float),
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


def _make_asymmetric_bunch_pair(
    *, charge_scale: float = 1.0
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rider = _make_bunch_state(
        x=[0.0, 0.55, 1.7, 3.1],
        z=-0.4,
        charge=0.05 * charge_scale,
    )
    driver = _make_bunch_state(
        x=[0.15, 0.9, 2.4, 4.0],
        z=0.65,
        charge=-0.04 * charge_scale,
    )
    rider["q"] = charge_scale * np.array([0.05, 0.02, -0.03, 0.04])
    driver["q"] = charge_scale * np.array([-0.04, 0.03, 0.02, -0.01])
    return rider, driver


def _clone_particle_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in state.items()}


def _assert_pseudo_grid_tracks_full_solver(
    actual: object,
    expected: object,
    *,
    label: str,
    atol: float,
) -> None:
    fields = (
        "x",
        "y",
        "z",
        "t",
        "Px",
        "Py",
        "Pz",
        "Pt",
        "gamma",
        "bx",
        "by",
        "bz",
    )
    for field in fields:
        actual_values = getattr(actual, field)
        expected_values = getattr(expected, field)
        assert np.all(np.isfinite(actual_values)), f"{label}.{field} is not finite"
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            rtol=0.0,
            atol=atol,
            err_msg=f"{label}.{field}",
        )


def _assert_schedule_has_nontrivial_passive_charge_aggregation(
    schedule: PseudoGridStepSchedule,
    rider_state: dict[str, np.ndarray],
    driver_state: dict[str, np.ndarray],
) -> None:
    assert schedule.rider_active_indices.size == 3
    assert schedule.driver_active_indices.size == 3
    assert schedule.rider_passive_map.passive_indices.size == 1
    assert schedule.driver_passive_map.passive_indices.size == 1

    rider_q = np.asarray(rider_state["q"], dtype=float)
    driver_q = np.asarray(driver_state["q"], dtype=float)
    np.testing.assert_allclose(
        np.sum(schedule.rider_effective_source_charges),
        np.sum(rider_q),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.sum(schedule.driver_effective_source_charges),
        np.sum(driver_q),
        atol=1e-15,
    )
    assert not np.allclose(
        schedule.rider_effective_source_charges,
        rider_q[schedule.rider_active_indices],
    )
    assert not np.allclose(
        schedule.driver_effective_source_charges,
        driver_q[schedule.driver_active_indices],
    )


def _assert_trajectory_arrays_match(actual: object, expected: object) -> None:
    float_fields = (
        "x",
        "y",
        "z",
        "t",
        "Px",
        "Py",
        "Pz",
        "Pt",
        "gamma",
        "bx",
        "by",
        "bz",
        "bdotx",
        "bdoty",
        "bdotz",
        "radiation_power",
        "radiation_energy",
        "radiation_energy_applied",
        "origin_x",
        "origin_y",
        "origin_z",
        "beta_avg_x",
        "beta_avg_y",
        "beta_avg_z",
        "beta_samples",
        "q",
        "m",
        "char_time",
    )
    for field in float_fields:
        np.testing.assert_allclose(
            getattr(actual, field),
            getattr(expected, field),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=field,
        )

    for field in ("dead", "halted_early", "halt_step"):
        np.testing.assert_array_equal(
            getattr(actual, field),
            getattr(expected, field),
            err_msg=field,
        )

    assert getattr(actual, "halt_reason") == getattr(expected, "halt_reason")


class _LoggerRecorder:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def __bool__(self) -> bool:
        return True


def test_retarded_integrator_logs_numba_kernel_usage_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.vectorized_interactions as vectorized_interactions

    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", True)

    messages: list[str] = []
    trajectory, driver, *_soa_out = retarded_integrator(
        steps=2,
        h_step=1e-3,
        wall_z=0.0,
        aperture_radius=0.5,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-1.0),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        image_subcharge_count=8,
        use_conducting_image_weighting=False,
        logger=messages.append,
        use_numba=True,
    )

    assert len(trajectory) == 2
    assert len(driver) == 2
    assert any(
        "Using Numba-optimized kernels in canonical integrator path" in message
        for message in messages
    )


def test_retarded_integrator_logs_proximity_transition_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h_calls: list[float] = []
    messages: list[str] = []

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
        **_kwargs: object,
    ) -> dict[str, object]:
        h_calls.append(h_step)
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, driver, *_soa_out = retarded_integrator(
        steps=2,
        h_step=1.0,
        wall_z=0.0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-0.75),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        adaptive_timestep=AdaptiveTimestepConfig(
            enabled=True,
            debug=True,
            proximity_refinement_enabled=True,
            proximity_distance_aperture_radii=1.0,
            proximity_transition_zone=0.5,
            proximity_reduction_factor=4.0,
            energy_jump_threshold=10.0,
        ),
        logger=messages.append,
        use_numba=False,
    )

    assert len(trajectory) == 2
    assert len(driver) == 2
    assert h_calls == pytest.approx([0.4, 0.4])
    assert any("transition zone" in message for message in messages)
    assert any("Reduction factor: 2.5000x" in message for message in messages)


def test_adaptive_b2b_step_passes_full_driver_soa_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_driver_x: list[tuple[int, float, float]] = []

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
        traj_soa: object = None,
        traj_ext_soa: object = None,
        **kwargs: object,
    ) -> dict[str, object]:
        if step_idx == 2 and traj_ext_soa is not None:
            observed_driver_x.append(
                (
                    len(trajectory_ext),
                    float(trajectory_ext[index_traj]["x"][0]),
                    float(traj_ext_soa.x[index_traj, 0]),
                )
            )

        state = _clone_state(trajectory[index_traj])
        state["t"] = np.array([float(state["t"][0]) + h_step], dtype=float)
        state["x"] = np.array([float(state["x"][0]) + 10.0], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    retarded_integrator(
        steps=3,
        h_step=1e-3,
        wall_z=0.0,
        aperture_radius=0.5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_particle_state(z=-1.0),
        init_driver=_make_particle_state(z=1.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        image_subcharge_count=8,
        use_conducting_image_weighting=False,
        use_numba=False,
    )

    assert observed_driver_x[0] == (2, 10.0, 10.0)


def test_retarded_integrator_logs_when_numba_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.vectorized_interactions as vectorized_interactions

    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", False)

    logger = _LoggerRecorder()
    trajectory, driver, *_soa_out = retarded_integrator(
        steps=1,
        h_step=1e-3,
        wall_z=0.0,
        aperture_radius=0.5,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-1.0),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        logger=logger,
        use_numba=True,
    )

    assert len(trajectory) == 1
    assert len(driver) == 1
    assert any(
        "Numba not available, using pure Python kernels" in message
        for message in logger.warnings
    )


def test_retarded_integrator_rejects_pseudo_grid_outside_bunch_to_bunch() -> None:
    with pytest.raises(NotImplementedError, match="BUNCH_TO_BUNCH"):
        retarded_integrator(
            steps=1,
            h_step=1e-3,
            wall_z=0.0,
            aperture_radius=0.5,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=_make_particle_state(z=-1.0),
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            pseudo_grid=PseudoGridConfig(enabled=True),
            use_numba=False,
        )


def test_retarded_integrator_records_pseudo_grid_schedule_metadata_without_changing_b2b_updates(
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
        **kwargs: object,
    ) -> dict[str, object]:
        state = _clone_state(trajectory[index_traj])
        state["t"] = np.asarray(state["t"], dtype=float) + h_step
        state["z"] = np.asarray(state["z"], dtype=float) + 0.25
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    base_rider, base_driver, *_base_soa = retarded_integrator(
        steps=2,
        h_step=0.5,
        wall_z=0.0,
        aperture_radius=0.5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_bunch_state(x=[0.0, 10.0, 20.0], z=-1.0, charge=1.0),
        init_driver=_make_bunch_state(x=[1.0, 11.0, 21.0], z=1.0, charge=2.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        use_numba=False,
    )
    pseudo_rider, pseudo_driver, pseudo_rider_soa, *_pseudo_soa = retarded_integrator(
        steps=2,
        h_step=0.5,
        wall_z=0.0,
        aperture_radius=0.5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_bunch_state(x=[0.0, 10.0, 20.0], z=-1.0, charge=1.0),
        init_driver=_make_bunch_state(x=[1.0, 11.0, 21.0], z=1.0, charge=2.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        pseudo_grid=PseudoGridConfig(
            enabled=True,
            active_rider_count=2,
            active_driver_count=2,
            passive_neighbor_count=2,
            causal_history_pruning_enabled=True,
        ),
        use_numba=False,
    )

    np.testing.assert_allclose(pseudo_rider[-1]["t"], base_rider[-1]["t"])
    np.testing.assert_allclose(pseudo_rider[-1]["z"], base_rider[-1]["z"])
    np.testing.assert_allclose(pseudo_driver[-1]["t"], base_driver[-1]["t"])
    np.testing.assert_allclose(pseudo_driver[-1]["z"], base_driver[-1]["z"])
    assert "_pseudo_grid_schedule" not in base_rider[-1]

    schedule = pseudo_rider[-1]["_pseudo_grid_schedule"]
    assert isinstance(schedule, PseudoGridStepSchedule)
    np.testing.assert_array_equal(schedule.rider_active_indices, np.array([0, 2]))
    np.testing.assert_array_equal(schedule.driver_active_indices, np.array([0, 2]))
    np.testing.assert_array_equal(
        schedule.rider_passive_map.passive_indices, np.array([1])
    )
    np.testing.assert_allclose(
        schedule.rider_effective_source_charges, np.array([1.5, 1.5])
    )
    np.testing.assert_allclose(
        schedule.driver_effective_source_charges, np.array([3.0, 3.0])
    )
    assert schedule.driver_history_start_index == 0
    assert schedule.rider_history_start_index == 0
    assert pseudo_rider_soa is not None
    assert pseudo_rider_soa.pseudo_grid_schedule[1] is schedule
    assert pseudo_rider_soa.state_at(1)["_pseudo_grid_schedule"] is schedule


def test_retarded_integrator_uses_active_only_histories_and_effective_source_charges_in_pseudo_grid_b2b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_calls: list[dict[str, np.ndarray | int]] = []

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
        **kwargs: object,
    ) -> dict[str, object]:
        observed_calls.append(
            {
                "observer_count": len(np.asarray(trajectory[index_traj]["x"])),
                "source_count": len(np.asarray(trajectory_ext[index_traj]["x"])),
                "observer_q": np.asarray(
                    trajectory[index_traj]["q"], dtype=float
                ).copy(),
                "source_q": np.asarray(
                    trajectory_ext[index_traj]["q"], dtype=float
                ).copy(),
            }
        )
        state = _clone_state(trajectory[index_traj])
        state["t"] = np.asarray(state["t"], dtype=float) + h_step
        state["z"] = np.asarray(state["z"], dtype=float) + 0.25
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    pseudo_rider, pseudo_driver, *_pseudo_soa = retarded_integrator(
        steps=2,
        h_step=0.5,
        wall_z=0.0,
        aperture_radius=0.5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_bunch_state(x=[0.0, 10.0, 20.0], z=-1.0, charge=1.0),
        init_driver=_make_bunch_state(x=[1.0, 11.0, 21.0], z=1.0, charge=2.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        pseudo_grid=PseudoGridConfig(
            enabled=True,
            active_rider_count=2,
            active_driver_count=2,
            passive_neighbor_count=2,
        ),
        use_numba=False,
    )

    assert len(observed_calls) == 2
    assert observed_calls[0]["observer_count"] == 2
    assert observed_calls[0]["source_count"] == 2
    np.testing.assert_allclose(observed_calls[0]["observer_q"], np.array([1.0, 1.0]))
    np.testing.assert_allclose(observed_calls[0]["source_q"], np.array([3.0, 3.0]))
    np.testing.assert_allclose(observed_calls[1]["observer_q"], np.array([2.0, 2.0]))
    np.testing.assert_allclose(observed_calls[1]["source_q"], np.array([1.5, 1.5]))
    np.testing.assert_allclose(pseudo_rider[-1]["z"], np.array([-0.75, -0.75, -0.75]))
    np.testing.assert_allclose(pseudo_driver[-1]["z"], np.array([1.25, 1.25, 1.25]))


def test_retarded_integrator_uses_reduced_histories_for_pseudo_grid_when_adaptive_timestep_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_counts: list[tuple[int, int, float]] = []

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
        **kwargs: object,
    ) -> dict[str, object]:
        observed_counts.append(
            (
                len(np.asarray(trajectory[index_traj]["x"])),
                len(np.asarray(trajectory_ext[index_traj]["x"])),
                h_step,
            )
        )
        state = _clone_state(trajectory[index_traj])
        state["t"] = np.asarray(state["t"], dtype=float) + h_step
        state["z"] = np.asarray(state["z"], dtype=float) + 0.25
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    pseudo_rider, *_pseudo_outputs = retarded_integrator(
        steps=2,
        h_step=0.5,
        wall_z=0.0,
        aperture_radius=0.5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_bunch_state(x=[0.0, 10.0, 20.0], z=-1.0, charge=1.0),
        init_driver=_make_bunch_state(x=[1.0, 11.0, 21.0], z=1.0, charge=2.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        adaptive_timestep=AdaptiveTimestepConfig(
            enabled=True,
            proximity_refinement_enabled=False,
            energy_jump_threshold=10.0,
        ),
        pseudo_grid=PseudoGridConfig(
            enabled=True,
            active_rider_count=2,
            active_driver_count=2,
            passive_neighbor_count=2,
        ),
        use_numba=False,
    )

    assert observed_counts == [(2, 2, 0.5), (2, 2, 0.5)]
    assert isinstance(pseudo_rider[-1]["_pseudo_grid_schedule"], PseudoGridStepSchedule)


def test_retarded_integrator_prunes_reduced_source_histories_when_causal_history_pruning_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_source_history_lengths: list[int] = []
    observed_observer_history_lengths: list[int] = []

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
        **kwargs: object,
    ) -> dict[str, object]:
        observed_observer_history_lengths.append(len(trajectory))
        observed_source_history_lengths.append(len(trajectory_ext))
        state = _clone_state(trajectory[index_traj])
        state["t"] = np.asarray(state["t"], dtype=float) + h_step
        state["z"] = np.asarray(state["z"], dtype=float) + 0.25
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    pseudo_rider, *_pseudo_outputs = retarded_integrator(
        steps=4,
        h_step=0.5,
        wall_z=0.0,
        aperture_radius=0.5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_bunch_state(x=[0.0, 10.0, 20.0], z=-1.0, charge=1.0),
        init_driver=_make_bunch_state(x=[1.0, 11.0, 21.0], z=1.0, charge=2.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        pseudo_grid=PseudoGridConfig(
            enabled=True,
            active_rider_count=2,
            active_driver_count=2,
            passive_neighbor_count=2,
            causal_history_pruning_enabled=True,
            causal_history_safety_margin_steps=0,
        ),
        use_numba=False,
    )

    assert observed_observer_history_lengths == [1, 1, 2, 2, 2, 2]
    assert observed_source_history_lengths == [1, 1, 1, 1, 1, 1]
    assert len(pseudo_rider) == 4
    assert all(state for state in pseudo_rider)
    schedule = pseudo_rider[-1]["_pseudo_grid_schedule"]
    assert isinstance(schedule, PseudoGridStepSchedule)
    assert schedule.driver_history_start_index == 2
    assert schedule.rider_history_start_index == 2
    assert schedule.driver_retained_history_start_index == 2
    assert schedule.rider_retained_history_start_index == 2
    assert schedule.driver_dropped_history_samples == 1
    assert schedule.rider_dropped_history_samples == 1


def test_retarded_integrator_uses_reduced_histories_and_self_excluded_space_charge_in_pseudo_grid_b2b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_calls: list[dict[str, object]] = []

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
        **kwargs: object,
    ) -> dict[str, object]:
        observed_calls.append(
            {
                "observer_count": len(np.asarray(trajectory[index_traj]["x"])),
                "source_count": len(np.asarray(trajectory_ext[index_traj]["x"])),
                "observer_q": np.asarray(
                    trajectory[index_traj]["q"],
                    dtype=float,
                ).copy(),
                "source_q": np.asarray(
                    trajectory_ext[index_traj]["q"],
                    dtype=float,
                ).copy(),
                "space_charge_matrix": np.asarray(
                    kwargs["pseudo_grid_space_charge_source_charges"],
                    dtype=float,
                ).copy(),
                "space_charge_enabled": bool(
                    getattr(kwargs.get("space_charge"), "enabled", False)
                ),
            }
        )
        state = _clone_state(trajectory[index_traj])
        state["t"] = np.asarray(state["t"], dtype=float) + h_step
        state["z"] = np.asarray(state["z"], dtype=float) + 0.25
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    pseudo_rider, pseudo_driver, *_pseudo_outputs = retarded_integrator(
        steps=2,
        h_step=0.5,
        wall_z=0.0,
        aperture_radius=0.5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_bunch_state(x=[0.0, 10.0, 20.0], z=-1.0, charge=1.0),
        init_driver=_make_bunch_state(x=[1.0, 11.0, 21.0], z=1.0, charge=2.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        space_charge=SpaceChargeConfig(enabled=True),
        pseudo_grid=PseudoGridConfig(
            enabled=True,
            active_rider_count=2,
            active_driver_count=2,
            passive_neighbor_count=2,
        ),
        use_numba=False,
    )

    assert len(observed_calls) == 2
    assert observed_calls[0]["space_charge_enabled"] is True
    assert observed_calls[1]["space_charge_enabled"] is True
    assert observed_calls[0]["observer_count"] == 2
    assert observed_calls[0]["source_count"] == 2
    np.testing.assert_allclose(observed_calls[0]["observer_q"], np.array([1.0, 1.0]))
    np.testing.assert_allclose(observed_calls[0]["source_q"], np.array([3.0, 3.0]))
    np.testing.assert_allclose(
        observed_calls[0]["space_charge_matrix"],
        np.array([[0.0, 2.0], [2.0, 0.0]]),
    )
    np.testing.assert_allclose(observed_calls[1]["observer_q"], np.array([2.0, 2.0]))
    np.testing.assert_allclose(observed_calls[1]["source_q"], np.array([1.5, 1.5]))
    np.testing.assert_allclose(
        observed_calls[1]["space_charge_matrix"],
        np.array([[0.0, 4.0], [4.0, 0.0]]),
    )
    np.testing.assert_allclose(pseudo_rider[-1]["z"], np.array([-0.75, -0.75, -0.75]))
    np.testing.assert_allclose(pseudo_driver[-1]["z"], np.array([1.25, 1.25, 1.25]))


def test_retarded_integrator_falls_back_to_full_histories_for_pseudo_grid_when_space_charge_active_counts_are_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_counts: list[tuple[int, int]] = []

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
        **kwargs: object,
    ) -> dict[str, object]:
        observed_counts.append(
            (
                len(np.asarray(trajectory[index_traj]["x"])),
                len(np.asarray(trajectory_ext[index_traj]["x"])),
            )
        )
        state = _clone_state(trajectory[index_traj])
        state["t"] = np.asarray(state["t"], dtype=float) + h_step
        state["z"] = np.asarray(state["z"], dtype=float) + 0.25
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    pseudo_rider, *_pseudo_outputs = retarded_integrator(
        steps=2,
        h_step=0.5,
        wall_z=0.0,
        aperture_radius=0.5,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_bunch_state(x=[0.0, 10.0, 20.0], z=-1.0, charge=1.0),
        init_driver=_make_bunch_state(x=[1.0, 11.0, 21.0], z=1.0, charge=2.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        space_charge=SpaceChargeConfig(enabled=True),
        pseudo_grid=PseudoGridConfig(
            enabled=True,
            active_rider_count=1,
            active_driver_count=1,
            passive_neighbor_count=1,
        ),
        use_numba=False,
    )

    assert observed_counts == [(3, 3), (3, 3)]
    assert isinstance(pseudo_rider[-1]["_pseudo_grid_schedule"], PseudoGridStepSchedule)


def test_retarded_integrator_matches_full_solver_for_pseudo_grid_with_space_charge_when_all_particles_are_active() -> (
    None
):
    def run_case(
        pseudo_grid: PseudoGridConfig | None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], object, object]:
        return retarded_integrator(
            steps=3,
            h_step=1.0e-3,
            wall_z=0.0,
            aperture_radius=1.0,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_make_bunch_state(
                x=[0.0, 0.8, 1.7],
                z=-0.5,
                charge=1.0,
            ),
            init_driver=_make_bunch_state(
                x=[0.2, 1.1, 2.1],
                z=0.5,
                charge=2.0,
            ),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            space_charge=SpaceChargeConfig(
                enabled=True,
                retarded=False,
                softening_mm=0.2,
            ),
            pseudo_grid=pseudo_grid,
            use_numba=False,
        )

    full_rider, full_driver, full_rider_soa, full_driver_soa = run_case(None)
    pseudo_rider, pseudo_driver, pseudo_rider_soa, pseudo_driver_soa = run_case(
        PseudoGridConfig(
            enabled=True,
            active_rider_count=3,
            active_driver_count=3,
            passive_neighbor_count=2,
        )
    )

    assert full_rider_soa is not None
    assert full_driver_soa is not None
    assert pseudo_rider_soa is not None
    assert pseudo_driver_soa is not None

    _assert_trajectory_arrays_match(pseudo_rider_soa, full_rider_soa)
    _assert_trajectory_arrays_match(pseudo_driver_soa, full_driver_soa)

    for step_idx in range(len(full_rider)):
        np.testing.assert_allclose(
            pseudo_rider[step_idx]["x"],
            full_rider[step_idx]["x"],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            pseudo_driver[step_idx]["x"],
            full_driver[step_idx]["x"],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    assert "_pseudo_grid_schedule" not in full_rider[-1]
    schedule = pseudo_rider[-1]["_pseudo_grid_schedule"]
    assert isinstance(schedule, PseudoGridStepSchedule)
    np.testing.assert_array_equal(schedule.rider_active_indices, np.array([0, 1, 2]))
    np.testing.assert_array_equal(schedule.driver_active_indices, np.array([0, 1, 2]))


def test_retarded_integrator_matches_full_solver_when_pseudo_grid_prunes_causally_irrelevant_source_history() -> (
    None
):
    def run_case(
        pseudo_grid: PseudoGridConfig | None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], object, object]:
        return retarded_integrator(
            steps=4,
            h_step=0.5,
            wall_z=0.0,
            aperture_radius=0.5,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_make_bunch_state(x=[0.0, 10.0, 20.0], z=-1.0, charge=1.0),
            init_driver=_make_bunch_state(x=[1.0, 11.0, 21.0], z=1.0, charge=2.0),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            pseudo_grid=pseudo_grid,
            use_numba=False,
        )

    full_rider, full_driver, full_rider_soa, full_driver_soa = run_case(None)
    pseudo_rider, pseudo_driver, pseudo_rider_soa, pseudo_driver_soa = run_case(
        PseudoGridConfig(
            enabled=True,
            active_rider_count=3,
            active_driver_count=3,
            passive_neighbor_count=2,
            causal_history_pruning_enabled=True,
            causal_history_safety_margin_steps=0,
        )
    )

    assert full_rider_soa is not None
    assert full_driver_soa is not None
    assert pseudo_rider_soa is not None
    assert pseudo_driver_soa is not None

    _assert_trajectory_arrays_match(pseudo_rider_soa, full_rider_soa)
    _assert_trajectory_arrays_match(pseudo_driver_soa, full_driver_soa)

    assert len(pseudo_rider) == len(full_rider)
    assert len(pseudo_driver) == len(full_driver)
    assert all(state for state in pseudo_rider)
    assert all(state for state in pseudo_driver)

    schedule = pseudo_rider[-1]["_pseudo_grid_schedule"]
    assert isinstance(schedule, PseudoGridStepSchedule)
    assert schedule.driver_history_start_index == 2
    assert schedule.rider_history_start_index == 2
    assert schedule.driver_retained_history_start_index == 2
    assert schedule.rider_retained_history_start_index == 2
    assert schedule.driver_dropped_history_samples == 1
    assert schedule.rider_dropped_history_samples == 1


def test_retarded_integrator_pseudo_grid_active_subset_tracks_full_solver_for_asymmetric_bunches() -> (
    None
):
    rider, driver = _make_asymmetric_bunch_pair(charge_scale=0.4)

    def run_case(pseudo_grid: PseudoGridConfig | None):
        return retarded_integrator(
            steps=5,
            h_step=1.0e-4,
            wall_z=0.0,
            aperture_radius=5.0,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_clone_particle_state(rider),
            init_driver=_clone_particle_state(driver),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            pseudo_grid=pseudo_grid,
            use_numba=False,
        )

    full_rider, full_driver, full_rider_soa, full_driver_soa = run_case(None)
    pseudo_rider, pseudo_driver, pseudo_rider_soa, pseudo_driver_soa = run_case(
        PseudoGridConfig(
            enabled=True,
            active_rider_count=3,
            active_driver_count=3,
            passive_neighbor_count=2,
        )
    )

    assert full_rider_soa is not None
    assert full_driver_soa is not None
    assert pseudo_rider_soa is not None
    assert pseudo_driver_soa is not None
    assert len(full_rider) == len(pseudo_rider)
    assert len(full_driver) == len(pseudo_driver)

    _assert_pseudo_grid_tracks_full_solver(
        pseudo_rider_soa,
        full_rider_soa,
        label="rider",
        atol=5.0e-6,
    )
    _assert_pseudo_grid_tracks_full_solver(
        pseudo_driver_soa,
        full_driver_soa,
        label="driver",
        atol=5.0e-6,
    )

    schedule = pseudo_rider[-1]["_pseudo_grid_schedule"]
    assert isinstance(schedule, PseudoGridStepSchedule)
    _assert_schedule_has_nontrivial_passive_charge_aggregation(
        schedule,
        rider,
        driver,
    )


def test_retarded_integrator_pseudo_grid_active_subset_tracks_full_solver_with_adaptive_timestep() -> (
    None
):
    rider, driver = _make_asymmetric_bunch_pair(charge_scale=0.4)

    def run_case(pseudo_grid: PseudoGridConfig | None):
        return retarded_integrator(
            steps=5,
            h_step=1.0e-4,
            wall_z=0.0,
            aperture_radius=5.0,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_clone_particle_state(rider),
            init_driver=_clone_particle_state(driver),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            adaptive_timestep=AdaptiveTimestepConfig(
                enabled=True,
                proximity_refinement_enabled=False,
                energy_jump_threshold=100.0,
            ),
            pseudo_grid=pseudo_grid,
            use_numba=False,
        )

    _full_rider, _full_driver, full_rider_soa, full_driver_soa = run_case(None)
    pseudo_rider, _pseudo_driver, pseudo_rider_soa, pseudo_driver_soa = run_case(
        PseudoGridConfig(
            enabled=True,
            active_rider_count=3,
            active_driver_count=3,
            passive_neighbor_count=2,
        )
    )

    assert full_rider_soa is not None
    assert full_driver_soa is not None
    assert pseudo_rider_soa is not None
    assert pseudo_driver_soa is not None

    _assert_pseudo_grid_tracks_full_solver(
        pseudo_rider_soa,
        full_rider_soa,
        label="adaptive rider",
        atol=5.0e-6,
    )
    _assert_pseudo_grid_tracks_full_solver(
        pseudo_driver_soa,
        full_driver_soa,
        label="adaptive driver",
        atol=5.0e-6,
    )

    schedule = pseudo_rider[-1]["_pseudo_grid_schedule"]
    assert isinstance(schedule, PseudoGridStepSchedule)
    _assert_schedule_has_nontrivial_passive_charge_aggregation(
        schedule,
        rider,
        driver,
    )


def test_retarded_integrator_pseudo_grid_active_subset_tracks_full_solver_with_space_charge() -> (
    None
):
    rider, driver = _make_asymmetric_bunch_pair(charge_scale=0.4)

    def run_case(pseudo_grid: PseudoGridConfig | None):
        return retarded_integrator(
            steps=5,
            h_step=1.0e-4,
            wall_z=0.0,
            aperture_radius=5.0,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_clone_particle_state(rider),
            init_driver=_clone_particle_state(driver),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            space_charge=SpaceChargeConfig(
                enabled=True,
                retarded=False,
                softening_mm=0.3,
            ),
            pseudo_grid=pseudo_grid,
            use_numba=False,
        )

    _full_rider, _full_driver, full_rider_soa, full_driver_soa = run_case(None)
    pseudo_rider, _pseudo_driver, pseudo_rider_soa, pseudo_driver_soa = run_case(
        PseudoGridConfig(
            enabled=True,
            active_rider_count=3,
            active_driver_count=3,
            passive_neighbor_count=2,
        )
    )

    assert full_rider_soa is not None
    assert full_driver_soa is not None
    assert pseudo_rider_soa is not None
    assert pseudo_driver_soa is not None

    _assert_pseudo_grid_tracks_full_solver(
        pseudo_rider_soa,
        full_rider_soa,
        label="space-charge rider",
        atol=5.0e-6,
    )
    _assert_pseudo_grid_tracks_full_solver(
        pseudo_driver_soa,
        full_driver_soa,
        label="space-charge driver",
        atol=5.0e-6,
    )

    schedule = pseudo_rider[-1]["_pseudo_grid_schedule"]
    assert isinstance(schedule, PseudoGridStepSchedule)
    _assert_schedule_has_nontrivial_passive_charge_aggregation(
        schedule,
        rider,
        driver,
    )


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
        **_kwargs: object,
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
        **_kwargs: object,
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

    trajectory, _, *_soa_out = retarded_integrator(
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


def test_retarded_integrator_accepts_energy_jump_after_max_refinement_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    gammas = iter([1.0, 2.0, 2.0, 2.0])

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
        **_kwargs: object,
    ) -> dict[str, object]:
        gamma = next(gammas)
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        state["gamma"] = np.array([gamma], dtype=float)
        state["Pt"] = np.array([gamma * integration_runner.C_MMNS], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, driver, *_soa_out = retarded_integrator(
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
            debug=True,
            energy_jump_threshold=0.5,
            timestep_reduction_factor=2,
            min_timestep_factor=0.5,
            proximity_refinement_enabled=False,
        ),
        logger=messages.append,
        use_numba=False,
    )

    assert len(trajectory) == 3
    assert len(driver) == 3
    assert trajectory[-1]["gamma"][0] == pytest.approx(2.0)
    assert any("Max refinement attempts (1) reached" in message for message in messages)


def test_retarded_integrator_accepts_energy_jump_at_minimum_timestep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    gammas = iter([1.0, 2.0])

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
        **_kwargs: object,
    ) -> dict[str, object]:
        gamma = next(gammas)
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        state["gamma"] = np.array([gamma], dtype=float)
        state["Pt"] = np.array([gamma * integration_runner.C_MMNS], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, driver, *_soa_out = retarded_integrator(
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
            debug=True,
            energy_jump_threshold=0.5,
            timestep_reduction_factor=2,
            min_timestep_factor=0.9,
            proximity_refinement_enabled=False,
        ),
        logger=messages.append,
        use_numba=False,
    )

    assert len(trajectory) == 3
    assert len(driver) == 3
    assert trajectory[-1]["gamma"][0] == pytest.approx(2.0)
    assert any(
        "Minimum timestep reached. Accepting remaining substeps" in message
        for message in messages
    )


def test_retarded_integrator_gamma_blowup_without_adaptive_marks_particle_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []

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
        **_kwargs: object,
    ) -> dict[str, object]:
        raise GammaBlowupError(
            step_idx=int(step_idx or 0),
            particle_idx=0,
            gamma_value=1e9,
            iteration=2,
        )

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, driver, *_soa_out = retarded_integrator(
        steps=3,
        h_step=1.0,
        wall_z=100.0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-1.0),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        logger=messages.append,
        use_numba=False,
    )

    assert len(trajectory) == 2
    assert len(driver) == 2
    assert trajectory[-1]["_dead_particles"][0]
    assert (
        trajectory[-1]["_particle_failure_info"][0]["reason"]
        == "gamma_blowup_no_adaptive"
    )
    assert any("Gamma blowup" in message for message in messages)


def test_retarded_integrator_gamma_blowup_at_min_timestep_marks_particle_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []

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
        **_kwargs: object,
    ) -> dict[str, object]:
        raise GammaBlowupError(
            step_idx=int(step_idx or 0),
            particle_idx=0,
            gamma_value=1e9,
            iteration=3,
        )

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, driver, *_soa_out = retarded_integrator(
        steps=3,
        h_step=1.0,
        wall_z=100.0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-1.0),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        adaptive_timestep=AdaptiveTimestepConfig(
            enabled=True,
            debug=True,
            timestep_reduction_factor=2,
            min_timestep_factor=0.9,
            proximity_refinement_enabled=False,
        ),
        logger=messages.append,
        use_numba=False,
    )

    assert len(trajectory) == 2
    assert len(driver) == 2
    assert trajectory[-1]["_dead_particles"][0]
    assert (
        trajectory[-1]["_particle_failure_info"][0]["reason"]
        == "gamma_blowup_min_timestep"
    )
    assert any(
        "Minimum timestep reached after gamma blowup" in message for message in messages
    )


def test_retarded_integrator_gamma_blowup_retries_with_reduced_timestep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h_calls: list[float] = []
    messages: list[str] = []

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
        **_kwargs: object,
    ) -> dict[str, object]:
        h_calls.append(h_step)
        if len(h_calls) == 1:
            raise GammaBlowupError(
                step_idx=int(step_idx or 0),
                particle_idx=0,
                gamma_value=1e9,
                iteration=4,
                is_hard_blowup=True,
            )

        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        state["gamma"] = np.array([1.0], dtype=float)
        state["Pt"] = np.array([integration_runner.C_MMNS], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, driver, *_soa_out = retarded_integrator(
        steps=2,
        h_step=1.0,
        wall_z=100.0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-1.0),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        adaptive_timestep=AdaptiveTimestepConfig(
            enabled=True,
            debug=True,
            timestep_reduction_factor=2,
            min_timestep_factor=0.1,
            proximity_refinement_enabled=False,
            energy_jump_threshold=10.0,
        ),
        logger=messages.append,
        use_numba=False,
    )

    assert len(trajectory) == 2
    assert len(driver) == 2
    assert h_calls[0] == pytest.approx(1.0)
    assert h_calls[1:] == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert trajectory[-1]["t"][0] == pytest.approx(1.0)
    assert any("HARD gamma blowup detected" in message for message in messages)


def test_retarded_integrator_gamma_blowup_hits_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []

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
        **_kwargs: object,
    ) -> dict[str, object]:
        raise GammaBlowupError(
            step_idx=int(step_idx or 0),
            particle_idx=0,
            gamma_value=1e9,
            iteration=5,
        )

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, driver, *_soa_out = retarded_integrator(
        steps=2,
        h_step=1.0,
        wall_z=100.0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-1.0),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        adaptive_timestep=AdaptiveTimestepConfig(
            enabled=True,
            debug=True,
            timestep_reduction_factor=2,
            min_timestep_factor=0.5,
            proximity_refinement_enabled=False,
        ),
        logger=messages.append,
        use_numba=False,
    )

    assert len(trajectory) == 2
    assert len(driver) == 2
    assert trajectory[-1]["_dead_particles"][0]
    assert (
        trajectory[-1]["_particle_failure_info"][0]["reason"]
        == "gamma_blowup_max_retries"
    )
    assert any(
        "Max refinement attempts reached after gamma blowup" in message
        for message in messages
    )


def test_retarded_integrator_returns_to_normal_timestep_after_stable_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h_calls: list[tuple[int | None, float]] = []
    messages: list[str] = []

    gammas = iter([1.0, 2.0, 1.05, 1.05, 1.05])

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
        **_kwargs: object,
    ) -> dict[str, object]:
        h_calls.append((step_idx, h_step))
        gamma = next(gammas)
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        state["gamma"] = np.array([gamma], dtype=float)
        state["Pt"] = np.array([gamma * integration_runner.C_MMNS], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, driver, *_soa_out = retarded_integrator(
        steps=4,
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
            debug=True,
            energy_jump_threshold=0.5,
            probe_threshold=0.1,
            cooldown_steps=0,
            max_probe_steps=1,
            timestep_reduction_factor=2,
            min_timestep_factor=0.1,
            proximity_refinement_enabled=False,
        ),
        logger=messages.append,
        use_numba=False,
    )

    assert len(trajectory) == 4
    assert len(driver) == 4
    assert h_calls == [
        (1, 1.0),
        (2, 1.0),
        (2, 0.5),
        (2, 0.5),
        (3, 1.0),
    ]
    assert any("Stable (" in message for message in messages)
    assert any("Returning to normal timestep" in message for message in messages)


def test_retarded_integrator_unstable_probe_restarts_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h_calls: list[tuple[int | None, float]] = []
    messages: list[str] = []

    gammas = iter([1.0, 2.0, 1.3, 1.3, 1.45, 1.45])

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
        **_kwargs: object,
    ) -> dict[str, object]:
        h_calls.append((step_idx, h_step))
        gamma = next(gammas)
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        state["gamma"] = np.array([gamma], dtype=float)
        state["Pt"] = np.array([gamma * integration_runner.C_MMNS], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, driver, *_soa_out = retarded_integrator(
        steps=4,
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
            debug=True,
            energy_jump_threshold=0.5,
            probe_threshold=0.1,
            cooldown_steps=1,
            max_probe_steps=1,
            timestep_reduction_factor=2,
            min_timestep_factor=0.1,
            proximity_refinement_enabled=False,
        ),
        logger=messages.append,
        use_numba=False,
    )

    assert len(trajectory) == 4
    assert len(driver) == 4
    step_3_calls = [h for step, h in h_calls if step == 3]
    assert step_3_calls == [0.5, 0.5]
    assert any("Unstable during probing" in message for message in messages)
    assert any("Cooldown mode" in message for message in messages)


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
        **_kwargs: object,
    ) -> dict[str, object]:
        source_state = (
            trajectory[index_traj] if trajectory[index_traj] else trajectory[-1]
        )
        state = _clone_state(source_state)
        state["z"] = np.array([float(source_state["z"][0]) + 2.0], dtype=float)
        state["t"] = np.array([float(source_state["t"][0]) + h_step], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    trajectory, driver, *_soa_out = retarded_integrator(
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



def test_retarded_integrator_halts_when_rider_reaches_cavity_exit(
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
        **_kwargs: object,
    ) -> dict[str, object]:
        source_state = _clone_state(trajectory[index_traj])
        z_value = float(source_state["z"][0])
        source_state["z"] = np.array([z_value + (3.0 if z_value < 5.0 else -1.0)])
        source_state["t"] = np.array([float(source_state["t"][0]) + h_step])
        return source_state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    trajectory, driver, *_soa_out = retarded_integrator(
        steps=5,
        h_step=1.0,
        wall_z=0.0,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_particle_state(z=0.0),
        init_driver=_make_particle_state(z=5.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        cavity_exit=CavityExitConfig(enabled=True),
        use_numba=False,
    )

    assert len(trajectory) == 3
    assert len(driver) == 3
    assert trajectory[-1]["_halted_early"] is True
    assert trajectory[-1]["_termination_reason"] == "cavity_exit_reached"
    assert trajectory[-1]["_exit_species"] == "rider"
    assert trajectory[-1]["_cavity_length_mm"] == pytest.approx(5.0)
    assert "cavity_exit_reached species=rider" in trajectory[-1]["_halt_reason"]


def test_retarded_integrator_halts_when_driver_reaches_cavity_exit(
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
        **_kwargs: object,
    ) -> dict[str, object]:
        source_state = _clone_state(trajectory[index_traj])
        z_value = float(source_state["z"][0])
        other_initial_z = float(trajectory_ext[0]["z"][0])
        is_rider_update = other_initial_z > z_value
        source_state["z"] = np.array([z_value + (1.0 if is_rider_update else -3.0)])
        source_state["t"] = np.array([float(source_state["t"][0]) + h_step])
        return source_state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    trajectory, driver, *_soa_out = retarded_integrator(
        steps=5,
        h_step=1.0,
        wall_z=0.0,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_particle_state(z=0.0),
        init_driver=_make_particle_state(z=5.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        cavity_exit=CavityExitConfig(enabled=True),
        use_numba=False,
    )

    assert len(trajectory) == 3
    assert len(driver) == 3
    assert trajectory[-1]["_termination_reason"] == "cavity_exit_reached"
    assert trajectory[-1]["_exit_species"] == "driver"
    assert trajectory[-1]["_driver_exit_z"] == pytest.approx(0.0)


def test_retarded_integrator_logs_relative_cutoff_debug_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []

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
        **_kwargs: object,
    ) -> dict[str, object]:
        source_state = next(state for state in reversed(trajectory) if state)
        state = _clone_state(source_state)
        state["z"] = np.array([float(source_state["z"][0])], dtype=float)
        state["t"] = np.array([float(source_state["t"][0]) + h_step], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)

    trajectory, driver, *_soa_out = retarded_integrator(
        steps=2,
        h_step=1.0,
        wall_z=0.0,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_make_particle_state(z=2.0),
        init_driver=_make_particle_state(z=5.0),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.5,
        z_cutoff_mode="relative",
        adaptive_timestep=AdaptiveTimestepConfig(enabled=True, debug=True),
        logger=messages.append,
        use_numba=False,
    )

    assert len(trajectory) == 2
    assert len(driver) == 2
    assert any("BUNCH_TO_BUNCH relative cutoff mode" in message for message in messages)


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
        **_kwargs: object,
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

    trajectory, driver, *_soa_out = retarded_integrator(
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


def test_retarded_integrator_applies_proximity_refinement_debug_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h_calls: list[float] = []
    messages: list[str] = []

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
        **_kwargs: object,
    ) -> dict[str, object]:
        h_calls.append(h_step)
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        state["z"] = np.array([float(trajectory[-1]["z"][0])], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    retarded_integrator(
        steps=2,
        h_step=1.0,
        wall_z=0.0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-0.2),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        adaptive_timestep=AdaptiveTimestepConfig(
            enabled=True,
            debug=True,
            energy_jump_threshold=10.0,
            proximity_refinement_enabled=True,
            proximity_distance_aperture_radii=1.0,
            proximity_transition_zone=0.5,
            proximity_reduction_factor=4.0,
        ),
        logger=messages.append,
        use_numba=False,
    )

    assert h_calls
    assert max(h_calls) < 1.0
    assert any("Proximity refinement active" in message for message in messages)
    assert any("Applying proximity refinement" in message for message in messages)


def test_retarded_integrator_switching_wall_advances_cutoff_and_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_calls: list[tuple[float, float]] = []

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
        **_kwargs: object,
    ) -> dict[str, object]:
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        state["z"] = np.array([1.0], dtype=float)
        return state

    def fake_switching_image(
        state: dict[str, object],
        wall_z: float,
        aperture_radius: float,
        cut_z: float,
    ) -> dict[str, object]:
        image_calls.append((wall_z, cut_z))
        return _clone_state(state)

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_switching_image",
        fake_switching_image,
    )

    trajectory, driver, *_soa_out = retarded_integrator(
        steps=3,
        h_step=1.0,
        wall_z=0.0,
        aperture_radius=1.0,
        sim_type=SimulationType.SWITCHING_WALL,
        init_rider=_make_particle_state(z=0.0),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.25,
        z_cutoff=0.5,
        use_numba=False,
    )

    assert len(trajectory) == 3
    assert len(driver) == 3
    assert image_calls[0] == (0.0, 0.5)
    assert (0.25, 0.75) in image_calls


def test_retarded_integrator_marks_post_step_gamma_blowup_and_reports_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
        **_kwargs: object,
    ) -> dict[str, object]:
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        state["gamma"] = np.array([1e9], dtype=float)
        state["Pt"] = np.array([1e9 * integration_runner.C_MMNS], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    trajectory, _, *_soa_out = retarded_integrator(
        steps=2,
        h_step=1.0,
        wall_z=0.0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-1.0),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        use_numba=False,
    )

    output = capsys.readouterr().out
    assert "gamma blowup detected" in output
    assert "[STATUS] Step 1: 0/1 particles alive, 1/1 dead" in output
    assert trajectory[-1]["_dead_particles"][0]


def test_retarded_integrator_energy_monitor_warns_and_debugs_without_halting(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
        **_kwargs: object,
    ) -> dict[str, object]:
        gamma = {1: 1.0, 2: 1.05, 3: 2.0}[int(step_idx or 0)]
        state = _clone_state(trajectory[-1])
        state["t"] = np.array([float(trajectory[-1]["t"][0]) + h_step], dtype=float)
        state["gamma"] = np.array([gamma], dtype=float)
        state["Pt"] = np.array([gamma * integration_runner.C_MMNS], dtype=float)
        return state

    monkeypatch.setattr(integration_runner, "self_consistent_step", fake_step)
    monkeypatch.setattr(
        integration_runner,
        "generate_conducting_image",
        lambda state, *args, **kwargs: _clone_state(state),
    )

    retarded_integrator(
        steps=4,
        h_step=1.0,
        wall_z=0.0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(z=-1.0),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        energy_monitor=EnergyMonitorConfig(
            enabled=True,
            relative_threshold=0.1,
            halt_on_jump=False,
            debug=True,
            check_interval=1,
        ),
        use_numba=False,
    )

    output = capsys.readouterr().out
    assert "Step 2: Energy =" in output
    assert "WARNING: Energy jump detected at step 3/4" in output


def test_run_integrator_forwards_config_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_retarded_integrator(**kwargs: object) -> tuple[list[str], list[str]]:
        captured.update(kwargs)
        return (["rider"], ["driver"])

    monkeypatch.setattr(
        integration_runner, "retarded_integrator", fake_retarded_integrator
    )

    config = IntegratorConfig(
        steps=5,
        time_step=0.25,
        wall_position=1.5,
        aperture_radius=0.75,
        simulation_type=SimulationType.SWITCHING_WALL,
        bunch_mean=2.0,
        cavity_spacing=0.5,
        z_cutoff=3.0,
        z_cutoff_mode="relative",
        use_image_weighting=False,
        radiation_reaction_mode="medina_lad",
        image_subcharge_count=16,
        macroparticle_charge_multiplier=3.0,
        macroparticle_sigma_multiplier=1.5,
        macroparticle_use_momentum_errors=False,
        bunch_transv_dist=0.2,
        bunch_transv_mom=0.3,
        pseudo_grid=PseudoGridConfig(
            enabled=True,
            active_rider_count=3,
            active_driver_count=4,
            passive_neighbor_count=2,
            pair_reuse_window=5,
        ),
        cavity_exit=CavityExitConfig(enabled=True, cavity_length_mm=12.0),
    )

    result = run_integrator(
        config=config,
        init_rider=_make_particle_state(),
        init_driver=None,
        energy_monitor=EnergyMonitorConfig(enabled=True),
        adaptive_timestep=AdaptiveTimestepConfig(enabled=True),
    )

    assert result == (["rider"], ["driver"])
    assert captured["steps"] == 5
    assert captured["h_step"] == pytest.approx(0.25)
    assert captured["wall_z"] == pytest.approx(1.5)
    assert captured["sim_type"] == SimulationType.SWITCHING_WALL
    assert captured["z_cutoff_mode"] == "relative"
    assert captured["image_subcharge_count"] == 16
    assert captured["use_conducting_image_weighting"] is False
    assert captured["radiation_reaction_mode"] == "medina_lad"
    assert captured["macroparticle_charge_multiplier"] == pytest.approx(3.0)
    assert captured["macroparticle_sigma_multiplier"] == pytest.approx(1.5)
    assert captured["macroparticle_use_momentum_errors"] is False
    assert captured["bunch_transv_dist"] == pytest.approx(0.2)
    assert captured["bunch_transv_mom"] == pytest.approx(0.3)
    assert captured["pseudo_grid"] == config.pseudo_grid
    assert captured["cavity_exit"] == config.cavity_exit
