"""Tests for pure single-integration result helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.types import SimulationType
from optimization.single_integration_helpers import (
    build_integration_metrics,
    build_final_z_check_log_lines,
    build_single_integration_setup,
    calculate_rider_starting_pz,
    distance_info_from_trajectory,
    sample_trajectory_arrays,
)


def _result(**overrides):
    defaults = {
        "rider_delta_e": None,
        "rider_gamma_initial": None,
        "rider_gamma_final": None,
        "rider_trajectory": None,
        "rider_emittance_x_mm_mrad": None,
        "rider_emittance_y_mm_mrad": None,
        "rider_norm_emittance_x_mm_mrad": None,
        "rider_norm_emittance_y_mm_mrad": None,
        "rider_beta_x_m": None,
        "rider_beta_y_m": None,
        "num_particles_dead": 0,
        "halted_early": False,
        "halt_reason": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _config(**overrides):
    defaults = {
        "m_particle": 1.0,
        "charge_sign": 1.0,
        "pcount": 3,
        "transv_mom": 0.0,
        "transv_dist": 1e-4,
        "stripped_ions": 1.0,
        "wall_z": 200.0,
        "macroparticle_charge_multiplier": 1.0,
        "macroparticle_sigma_multiplier": 1.0,
        "simulation_type": SimulationType.CONDUCTING_WALL,
        "target_distance_mm": 25.0,
        "z_cutoff_mode": "absolute",
        "startup_mode": "FAST",
        "seed": 123,
        "trajectory_stride": 2,
        "macroparticle_enabled": False,
        "macroparticle_use_momentum_errors": True,
        "image_subcharge_count": 10,
        "use_image_weighting": True,
        "self_consistency_enabled": False,
        "self_consistency_tolerance": 1e-3,
        "self_consistency_max_iterations": 5,
        "self_consistency_verbosity": 0,
        "energy_monitor_halt_on_jump": True,
        "adaptive_timestep_enabled": False,
        "adaptive_timestep_threshold": 0.1,
        "adaptive_timestep_reduction_factor": 2,
        "adaptive_timestep_min_factor": 1e-4,
        "adaptive_timestep_cooldown_steps": 5,
        "adaptive_timestep_probe_threshold": 0.01,
        "adaptive_timestep_max_probe_steps": 3,
        "adaptive_timestep_debug": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_calculate_rider_starting_pz_uses_b2b_kinetic_energy():
    wall_pz = calculate_rider_starting_pz(
        5.0, 1.0, SimulationType.CONDUCTING_WALL
    )
    b2b_pz = calculate_rider_starting_pz(5.0, 1.0, "BUNCH_TO_BUNCH")

    assert b2b_pz > wall_pz


def test_build_single_integration_setup_resolves_defaults_and_options(tmp_path):
    setup = build_single_integration_setup(
        _config(simulation_type="BUNCH_TO_BUNCH", z_cutoff_mode="relative"),
        aperture=0.01,
        energy_gev=5.0,
        start_z=10.0,
        transv_offset=0.25,
        timestep=1e-7,
        steps=100,
        run_output_dir=tmp_path,
        run_num=4,
        driver_params={"starting_distance": 900.0},
        rider_m_particle=None,
        rider_charge_sign=-1.0,
        seed_override=999,
    )

    assert setup.rider_m_particle == 1.0
    assert setup.rider_charge_sign == -1.0
    assert setup.options.seed == 999
    assert setup.options.driver_params == {"starting_distance": 900.0}
    assert setup.options.rider_params["transv_offset_x"] == 0.25
    assert setup.options.rider_params["starting_Pz"] > 0.0
    assert setup.options.core_params["z_cutoff"] == 25.0
    assert setup.options.output_dir == tmp_path


def test_build_integration_metrics_uses_direct_gamma_values():
    outcome = build_integration_metrics(
        _result(
            rider_delta_e=1.0,
            rider_gamma_initial=10.0,
            rider_gamma_final=12.0,
            rider_emittance_x_mm_mrad=0.5,
            num_particles_dead=2,
        ),
        rider_m_particle=1.0,
        run_num=4,
        optimization_mode=True,
    )

    assert outcome.metrics["rider_delta_e_mev"] == 1.0
    assert outcome.metrics["max_percent_energy_gain"] == pytest.approx(20.0)
    assert outcome.metrics["delta_e_mev"] == pytest.approx(2.0 * 931.494)
    assert outcome.metrics["rider_emittance_x_mm_mrad"] == 0.5
    assert outcome.metrics["num_particles_dead"] == 2
    assert any("optimizer_objective" in line for line in outcome.log_lines)


def test_build_integration_metrics_falls_back_to_trajectory_gamma():
    outcome = build_integration_metrics(
        _result(rider_trajectory={"gamma": [10.0, 11.0]}),
        rider_m_particle=2.0,
        run_num=5,
    )

    assert outcome.metrics["max_percent_energy_gain"] == pytest.approx(10.0)
    assert outcome.metrics["delta_e_mev"] == pytest.approx(1.0 * 2.0 * 931.494)
    assert any("Fallback calculation successful" in line for line in outcome.log_lines)
    assert any("gamma_initial (from traj)" in line for line in outcome.log_lines)


def test_build_integration_metrics_reports_missing_gamma():
    outcome = build_integration_metrics(
        _result(rider_trajectory=None),
        rider_m_particle=1.0,
        run_num=6,
    )

    assert "max_percent_energy_gain" not in outcome.metrics
    assert any("No trajectory data available" in line for line in outcome.log_lines)
    assert any("could not be calculated for Run 6" in line for line in outcome.log_lines)


def test_sample_trajectory_arrays_applies_stride():
    sampled = sample_trajectory_arrays(
        {
            "z": [0.0, 1.0, 2.0],
            "r": [0.0, 0.1, 0.2],
            "pz": [1.0, 2.0, 3.0],
            "pr": [0.1, 0.2, 0.3],
            "t": [0.0, 0.5, 1.0],
            "gamma": [10.0, 11.0, 12.0],
        },
        stride=2,
    )

    assert sampled == {
        "z": [0.0, 2.0],
        "r": [0.0, 0.2],
        "pz": [1.0, 3.0],
        "pr": [0.1, 0.3],
        "t": [0.0, 1.0],
        "gamma": [10.0, 12.0],
    }


def test_distance_info_from_trajectory_handles_empty_and_present_z():
    assert distance_info_from_trajectory({"z": []}) is None
    assert distance_info_from_trajectory({"z": [3.0, 7.0]}) == {
        "z_start": 3.0,
        "z_end": 7.0,
        "num_steps": 2,
    }


def test_build_final_z_check_log_lines_reports_b2b_excess():
    lines = build_final_z_check_log_lines(
        trajectory={"z": [0.0, 930.0]},
        simulation_type="BUNCH_TO_BUNCH",
        driver_params={"starting_distance": 900.0},
        target_distance_mm=25.0,
        wall_z=200.0,
        run_num=8,
    )

    assert any("EXCEEDED" in line for line in lines)
    assert any("driver_start + target=25.00" in line for line in lines)


def test_build_final_z_check_log_lines_reports_wall_ok():
    lines = build_final_z_check_log_lines(
        trajectory={"z": [0.0, 210.0]},
        simulation_type=SimulationType.CONDUCTING_WALL,
        driver_params=None,
        target_distance_mm=25.0,
        wall_z=200.0,
        run_num=9,
    )

    assert lines == [
        "  [DEBUG] Run 9: Final z check OK",
        "    Final z: 210.00 mm (under by 15.00 mm)",
    ]
