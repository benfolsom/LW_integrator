"""Tests for BUNCH_TO_BUNCH driver-train prehistory plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from core.distances import _locate_retarded_index, _locate_retarded_index_soa
from core.integration_runner import (
    _build_coasting_history,
    _build_driver_train_initial_state,
    retarded_integrator,
)
from core.types import (
    ChronoMatchingMode,
    DriverTrainConfig,
    SimulationType,
    StartupMode,
)
from input_output.bunch_initialization import create_bunch_from_energy


def _single_particle_state() -> dict[str, np.ndarray]:
    state, _ = create_bunch_from_energy(
        kinetic_energy_mev=1.0,
        mass_amu=1.0,
        charge_sign=1.0,
        position_z=10.0,
        particle_count=1,
    )
    return state


def test_driver_train_builder_concatenates_z_offsets_without_mutating_source():
    source = _single_particle_state()
    original_z = np.copy(source["z"])

    train = _build_driver_train_initial_state(
        source,
        DriverTrainConfig(
            enabled=True,
            bunch_count=3,
            z_offsets_mm=(0.0, 100.0, 250.0),
        ),
    )

    assert np.array_equal(source["z"], original_z)
    assert train["z"].shape == (3,)
    assert train["z"] == pytest.approx([10.0, 110.0, 260.0])
    assert train["q"] == pytest.approx([source["q"][0]] * 3)


def test_coasting_history_uses_negative_lab_times_and_oldest_origin():
    state = _single_particle_state()
    state["gamma"] = np.array([2.0])
    state["bz"] = np.array([0.5])
    state["z"] = np.array([10.0])
    state["t"] = np.array([0.0])

    history = _build_coasting_history(state, h_step=0.1, prehistory_steps=2)

    assert len(history) == 3
    assert history[-1]["t"] == pytest.approx([0.0])
    assert history[0]["t"] == pytest.approx([-0.4])
    assert history[1]["t"] == pytest.approx([-0.2])
    assert history[0]["z"][0] < history[-1]["z"][0]
    assert history[-1]["origin_z"] == pytest.approx(history[0]["z"])


def test_retarded_index_helpers_search_negative_prehistory():
    trajectory = [
        {"t": np.array([-2.0])},
        {"t": np.array([-1.0])},
        {"t": np.array([0.0])},
    ]
    t_col = np.array([-2.0, -1.0, 0.0])

    assert _locate_retarded_index(trajectory, 2, 0, -1.5) == 1
    assert _locate_retarded_index_soa(t_col, 2, -1.5) == 1
    assert _locate_retarded_index(trajectory, 2, 0, -3.0) == 0
    assert _locate_retarded_index_soa(t_col, 2, -3.0) == 0


def test_driver_train_zero_charge_smoke_returns_trimmed_active_window():
    rider = _single_particle_state()
    driver = _single_particle_state()
    driver["z"] = np.array([100.0])
    driver["Pz"] = -driver["Pz"]
    driver["bz"] = -driver["bz"]
    rider["q"] = np.zeros_like(rider["q"])
    driver["q"] = np.zeros_like(driver["q"])

    rider_traj, driver_traj, rider_soa, driver_soa = retarded_integrator(
        steps=4,
        h_step=1e-4,
        wall_z=0.0,
        aperture_radius=1e6,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=rider,
        init_driver=driver,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        chrono_mode=ChronoMatchingMode.FAST,
        startup_mode=StartupMode.COLD_START,
        radiation_reaction_mode="off",
        driver_train=DriverTrainConfig(
            enabled=True,
            bunch_count=2,
            z_spacing_mm=50.0,
            prehistory_steps=3,
        ),
    )

    assert len(rider_traj) == 4
    assert len(driver_traj) == 4
    assert rider_soa is not None
    assert driver_soa is not None
    assert rider_soa.n_steps == 4
    assert driver_soa.n_steps == 4
    assert driver_traj[0]["z"].shape == (2,)
    assert driver_traj[0]["z"] == pytest.approx([100.0, 150.0])
    assert rider_traj[0]["t"] == pytest.approx([0.0])
