"""Tests for pure sweep result/logging helpers."""

from __future__ import annotations

import numpy as np

from core.types import SimulationType
import optimization.sweep_result_helpers as sweep_result_helpers
from optimization.sweep_result_helpers import (
    SweepMetricSummary,
    build_failed_sweep_run_record,
    build_full_debug_sweep_result_log_lines,
    build_sweep_completion_log_lines,
    build_sweep_run_data,
    build_timeout_sweep_run_record,
    build_truncated_sweep_log_params,
    classify_sweep_attempt_result,
    extract_actual_distance,
    extract_sweep_metric_summary,
    simulation_type_name,
)


def test_module_exposes_only_maintained_public_helpers():
    assert sweep_result_helpers.__all__ == [
        "SweepAttemptClassification",
        "SweepMetricSummary",
        "build_failed_sweep_run_record",
        "build_full_debug_sweep_result_log_lines",
        "build_sweep_completion_log_lines",
        "build_sweep_run_data",
        "build_timeout_sweep_run_record",
        "build_truncated_sweep_log_params",
        "classify_sweep_attempt_result",
        "extract_actual_distance",
        "extract_sweep_metric_summary",
        "simulation_type_name",
    ]


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


def test_classify_sweep_attempt_result_accepts_percent_gain():
    classification = classify_sweep_attempt_result(
        {"metrics": {"max_percent_energy_gain": 0.0}},
        run_num=3,
        retry_attempt=0,
        include_debug_logs=True,
    )

    assert classification.succeeded is True
    assert classification.error is None
    assert any("has_metrics=True" in line for line in classification.log_lines)


def test_classify_sweep_attempt_result_accepts_positive_gamma_fallback():
    classification = classify_sweep_attempt_result(
        {"metrics": {"rider_gamma_final": 2.0}},
        run_num=3,
        retry_attempt=1,
    )

    assert classification.succeeded is True


def test_classify_sweep_attempt_result_rejects_halted_or_empty_metrics():
    halted = classify_sweep_attempt_result(
        {"metrics": {"max_percent_energy_gain": 2.0}, "halted_early": True},
        run_num=4,
        retry_attempt=2,
        include_debug_logs=True,
    )
    empty = classify_sweep_attempt_result(
        {"metrics": {}, "halt_reason": "all dead"},
        run_num=5,
        retry_attempt=3,
    )

    assert halted.succeeded is False
    assert halted.error is not None
    assert "halted_early=True" in str(halted.error)
    assert any("has_useful_metrics=False" in line for line in halted.log_lines)
    assert empty.succeeded is False
    assert empty.error is not None
    assert "reason=all dead" in str(empty.error)


def test_extract_sweep_metric_summary_uses_zero_defaults():
    summary = extract_sweep_metric_summary(
        {"metrics": {"rider_delta_e_mev": 1.0, "rider_gamma_final": 12.0}}
    )

    assert summary.delta_e == 1.0
    assert summary.delta_gamma == 0.0
    assert summary.gamma_initial == 0.0
    assert summary.gamma_final == 12.0


def test_build_full_debug_sweep_result_log_lines_warns_on_no_motion():
    lines = build_full_debug_sweep_result_log_lines(
        run_num=2,
        total_runs=5,
        expected_distance=10.0,
        actual_distance=0.0,
        metrics=SweepMetricSummary(
            delta_e=1.0,
            delta_gamma=0.2,
            gamma_initial=10.0,
            gamma_final=10.2,
        ),
    )

    assert lines[0] == "  [RESULT] Run 2/5:"
    assert any("Particle barely moved" in line for line in lines)


def test_build_failed_and_timeout_run_records():
    failed = build_failed_sweep_run_record(
        run_num=3,
        aperture=0.01,
        energy=5.0,
        start_z=1.0,
        transv_offset=0.1,
        timestep=1e-7,
        steps=100,
        wall_z=200.0,
        error="bad",
        error_details="traceback",
    )
    timeout = build_timeout_sweep_run_record(
        run_num=4,
        aperture=0.02,
        energy=6.0,
        start_z=2.0,
        transv_offset=0.2,
        timestep=2e-7,
        steps=200,
        timeout_seconds=30.0,
    )

    assert failed["parameters"]["wall_z"] == 200.0
    assert failed["error_details"] == "traceback"
    assert timeout["error"] == "TIMEOUT"
    assert timeout["timeout_seconds"] == 30.0


def test_build_sweep_completion_log_lines_formats_time_branches():
    seconds = build_sweep_completion_log_lines(
        output_dir="out",
        successful_runs=2,
        failed_runs=0,
        elapsed_time=3.5,
    )
    minutes = build_sweep_completion_log_lines(
        output_dir="out",
        successful_runs=2,
        failed_runs=1,
        elapsed_time=65.0,
    )
    hours = build_sweep_completion_log_lines(
        output_dir="out",
        successful_runs=2,
        failed_runs=1,
        elapsed_time=3665.0,
    )

    assert seconds[-1] == "  Total time: 3.5s"
    assert "Failed/timed-out" not in "\n".join(seconds)
    assert minutes[-1] == "  Total time: 1m 5.0s (65.0s)"
    assert hours[-1] == "  Total time: 1h 1m 5.0s (3665.0s)"
