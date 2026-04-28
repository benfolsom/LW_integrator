"""Tests for pure sweep result/logging helpers."""

from __future__ import annotations

import numpy as np

from core.types import SimulationType
from optimization.sweep_result_helpers import (
    build_sweep_run_data,
    build_truncated_sweep_log_params,
    extract_actual_distance,
    simulation_type_name,
)


def test_simulation_type_name_accepts_enum_and_string_modes():
    assert simulation_type_name(SimulationType.BUNCH_TO_BUNCH) == "BUNCH_TO_BUNCH"
    assert simulation_type_name("BUNCH_TO_BUNCH") == "BUNCH_TO_BUNCH"


def test_build_sweep_run_data_serializes_string_mode_and_driver_params():
    record = build_sweep_run_data(
        run_number=12,
        params_dict={"wall_z": 250.0},
        simulation_type="BUNCH_TO_BUNCH",
        aperture=0.001,
        energy=5.0,
        start_z=10.0,
        transv_offset=0.25,
        offset_frac=0.25,
        timestep=1e-7,
        steps=100,
        retry_attempts=2,
        default_wall_z=200.0,
        rider_m_particle=1.0,
        rider_charge_sign=1.0,
        rider_pcount=3,
        rider_transv_mom=0.0,
        rider_transv_dist=1e-4,
        macroparticle_charge_multiplier=4.0,
        macroparticle_sigma_multiplier=2.0,
        metrics={"max_percent_energy_gain": 1.5},
        driver_params={"starting_distance": 1000.0, "pcount": 5},
    )

    assert record["run_number"] == 12
    assert record["parameters"]["simulation_type"] == "BUNCH_TO_BUNCH"
    assert record["parameters"]["wall_z"] == 250.0
    assert record["parameters"]["driver_starting_distance"] == 1000.0
    assert record["parameters"]["driver_pcount"] == 5
    assert record["metrics"] == {"max_percent_energy_gain": 1.5}


def test_build_truncated_sweep_log_params_prefers_swept_values():
    params = build_truncated_sweep_log_params(
        param_grids={"energy": [1.0, 2.0], "wall_z": [200.0]},
        params_dict={"energy": 2.0, "wall_z": 200.0},
        simulation_type=SimulationType.CONDUCTING_WALL,
        aperture=0.01,
        energy=2.0,
        wall_z=200.0,
    )

    assert params == {"energy": 2.0}


def test_build_truncated_sweep_log_params_falls_back_for_string_b2b():
    params = build_truncated_sweep_log_params(
        param_grids={"initial_energy_gev": [5.0], "driver_starting_distance": [900.0]},
        params_dict={
            "initial_energy_gev": 5.0,
            "driver_starting_distance": 900.0,
        },
        simulation_type="BUNCH_TO_BUNCH",
        aperture=0.01,
        energy=5.0,
        wall_z=200.0,
    )

    assert params == {
        "initial_energy_gev": 5.0,
        "driver_starting_distance": 900.0,
        "wall_z": 200.0,
    }


def test_extract_actual_distance_prefers_distance_info():
    assert (
        extract_actual_distance({"_distance_info": {"z_start": 10.0, "z_end": 25.0}})
        == 15.0
    )


def test_extract_actual_distance_falls_back_to_trajectory_arrays():
    distance = extract_actual_distance(
        {"trajectory": {"z": [np.array([5.0]), np.array([17.5])]}}
    )

    assert distance == 12.5
