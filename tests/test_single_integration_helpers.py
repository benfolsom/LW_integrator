"""Tests for pure single-integration result helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.types import SimulationType
import optimization.single_integration_helpers as single_integration_helpers
from optimization.single_integration_helpers import (
    HaltedIntegrationOutput,
    IntegrationTrajectoryOutput,
    build_integration_metrics,
    build_final_z_check_log_lines,
    build_halted_integration_output,
    build_integration_trajectory_output,
    build_single_integration_setup,
    calculate_rider_starting_pz,
    distance_info_from_trajectory,
    sample_trajectory_arrays,
)


def test_module_exposes_only_supported_public_helpers():
    assert single_integration_helpers.__all__ == [
        "HaltedIntegrationOutput",
        "IntegrationMetricsOutcome",
        "IntegrationTrajectoryOutput",
        "SingleIntegrationSetup",
        "build_final_z_check_log_lines",
        "build_halted_integration_output",
        "build_integration_metrics",
        "build_integration_trajectory_output",
        "build_single_integration_setup",
        "calculate_rider_starting_pz",
        "distance_info_from_trajectory",
        "sample_trajectory_arrays",
    ]


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
        "cavity_spacing": 321.0,
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
        "self_consistency_convergence_mode": "variable_geometry",
        "self_consistency_target_ms_tolerance": 3e-7,
        "self_consistency_max_iterations": 5,
        "self_consistency_mass_shell_tolerance": 4e-3,
        "self_consistency_mass_shell_relaxation": 0.42,
        "self_consistency_verbosity": 0,
        "self_consistency_chrono_interpolate": True,
        "self_consistency_chrono_tolerance": 2e-3,
        "self_consistency_chrono_matching_mode": "AVERAGED",
        "self_consistency_chrono_high_precision": True,
        "self_consistency_chrono_adaptive_tolerance": True,
        "self_consistency_gamma_reconciliation_method": "FIXED_WEIGHTED",
        "self_consistency_gamma_reconciliation_fixed_weight": 0.25,
        "energy_monitor_halt_on_jump": True,
        "adaptive_timestep_enabled": False,
        "adaptive_timestep_threshold": 0.1,
        "adaptive_timestep_reduction_factor": 2,
        "adaptive_timestep_min_factor": 1e-4,
        "adaptive_timestep_cooldown_steps": 5,
        "adaptive_timestep_probe_threshold": 0.01,
        "adaptive_timestep_max_probe_steps": 3,
        "adaptive_timestep_debug": False,
        "space_charge_enabled": True,
        "space_charge_retarded": False,
        "space_charge_softening_mm": 0.012,
        "space_charge_bunch_sigma_mm": 0.034,
        "space_charge_min_retarded_steps": 6,
        "external_field_enabled": True,
        "external_electric_field_native": (1.0, 0.0, 0.0),
        "external_electric_field_v_per_m": (0.0, 2.0, 0.0),
        "external_magnetic_field_native": (0.0, 0.0, 3.0),
        "external_field_x_min": -1.0,
        "external_field_x_max": 1.0,
        "external_field_y_min": -2.0,
        "external_field_y_max": 2.0,
        "external_field_z_min": -3.0,
        "external_field_z_max": 3.0,
        "external_field_t_min": 1.0e-6,
        "external_field_t_max": 2.0e-6,
        "smoothness_enabled": True,
        "smoothness_window_size": 20,
        "smoothness_oscillation_threshold": 0.2,
        "smoothness_trend_threshold": 0.3,
        "smoothness_reject_on_violation": True,
        "smoothness_max_violations": 3,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_calculate_rider_starting_pz_uses_b2b_kinetic_energy():
    wall_pz = calculate_rider_starting_pz(5.0, 1.0, SimulationType.CONDUCTING_WALL)
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
    assert setup.options.core_params["cav_spacing"] == 321.0
    assert setup.options.core_params["z_cutoff"] == 25.0
    assert setup.options.self_consistency_convergence_mode == "variable_geometry"
    assert setup.options.self_consistency_target_ms_tolerance == pytest.approx(3e-7)
    assert setup.options.self_consistency_chrono_interpolate is True
    assert setup.options.self_consistency_chrono_tolerance == pytest.approx(2e-3)
    assert setup.options.self_consistency_chrono_matching_mode == "AVERAGED"
    assert setup.options.self_consistency_chrono_high_precision is True
    assert setup.options.self_consistency_chrono_adaptive_tolerance is True
    assert setup.options.self_consistency_mass_shell_tolerance == pytest.approx(4e-3)
    assert setup.options.self_consistency_mass_shell_relaxation == pytest.approx(0.42)
    assert (
        setup.options.self_consistency_gamma_reconciliation_method == "FIXED_WEIGHTED"
    )
    assert setup.options.self_consistency_gamma_reconciliation_fixed_weight == 0.25
    assert setup.options.space_charge_enabled is True
    assert setup.options.space_charge_retarded is False
    assert setup.options.space_charge_softening_mm == pytest.approx(0.012)
    assert setup.options.space_charge_bunch_sigma_mm == pytest.approx(0.034)
    assert setup.options.space_charge_min_retarded_steps == 6
    assert setup.options.external_field_enabled is True
    assert setup.options.external_electric_field_native == pytest.approx(
        (1.0, 0.0, 0.0)
    )
    assert setup.options.external_electric_field_v_per_m == pytest.approx(
        (0.0, 2.0, 0.0)
    )
    assert setup.options.external_magnetic_field_native == pytest.approx(
        (0.0, 0.0, 3.0)
    )
    assert setup.options.external_field_x_min == pytest.approx(-1.0)
    assert setup.options.external_field_x_max == pytest.approx(1.0)
    assert setup.options.external_field_y_min == pytest.approx(-2.0)
    assert setup.options.external_field_y_max == pytest.approx(2.0)
    assert setup.options.external_field_z_min == pytest.approx(-3.0)
    assert setup.options.external_field_z_max == pytest.approx(3.0)
    assert setup.options.external_field_t_min == pytest.approx(1.0e-6)
    assert setup.options.external_field_t_max == pytest.approx(2.0e-6)
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
    assert outcome.metrics["initial_gamma_mean"] == 10.0
    assert outcome.metrics["final_gamma_mean"] == 12.0
    assert outcome.metrics["max_percent_energy_gain"] == pytest.approx(20.0)
    assert outcome.metrics["delta_e_mev"] == pytest.approx(2.0 * 931.494)
    assert outcome.metrics["max_energy_gain_gev"] == pytest.approx(2.0 * 931.494 / 1e3)
    assert outcome.metrics["max_relative_gain"] == pytest.approx(0.2)
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
    assert any(
        "could not be calculated for Run 6" in line for line in outcome.log_lines
    )


def test_build_halted_integration_output_can_save_sampled_trajectory():
    outcome = build_halted_integration_output(
        _result(
            halted_early=True,
            halt_reason="cancelled",
            rider_trajectory={
                "z": [0.0, 1.0, 2.0],
                "r": [0.0, 0.1, 0.2],
                "pz": [1.0, 2.0, 3.0],
                "pr": [0.1, 0.2, 0.3],
                "t": [0.0, 0.5, 1.0],
                "gamma": [10.0, 11.0, 12.0],
            },
        ),
        run_num=7,
        save_trajectory=True,
        trajectory_stride=2,
    )

    assert isinstance(outcome, HaltedIntegrationOutput)
    assert outcome.output == {
        "metrics": {},
        "halted_early": True,
        "halt_reason": "cancelled",
        "trajectory": {
            "z": [0.0, 2.0],
            "r": [0.0, 0.2],
            "pz": [1.0, 3.0],
            "pr": [0.1, 0.3],
            "t": [0.0, 1.0],
            "gamma": [10.0, 12.0],
        },
    }
    assert "    Halted trajectory saved (3 points, stride=2)" in outcome.log_lines
    assert outcome.log_lines[-1] == (
        "  [DEBUG] _run_single_integration returning for halted Run 7"
    )


def test_build_halted_integration_output_warns_on_bad_trajectory_payload():
    outcome = build_halted_integration_output(
        _result(
            halted_early=True,
            halt_reason="cancelled",
            rider_trajectory={"z": [0.0]},
        ),
        run_num=7,
        save_trajectory=True,
        trajectory_stride=1,
    )

    assert outcome.output == {
        "metrics": {},
        "halted_early": True,
        "halt_reason": "cancelled",
    }
    assert any(
        line.startswith("    [WARNING] Failed to save halted trajectory:")
        for line in outcome.log_lines
    )


def test_build_integration_trajectory_output_reports_missing_trajectory():
    outcome = build_integration_trajectory_output(
        _result(rider_trajectory=None),
        _config(smoothness_enabled=True),
        run_num=8,
        rider_m_particle=1.0,
        metrics={},
        save_trajectory=False,
        trajectory_stride=1,
    )

    assert isinstance(outcome, IntegrationTrajectoryOutput)
    assert outcome.output_updates == {}
    assert outcome.debug_print_lines == []
    assert outcome.log_lines == [
        "  [DEBUG] Processing trajectory data for Run 8...",
        "  [WARNING] No trajectory data available for Run 8",
        (
            "  [WARNING] Stability analysis SKIPPED - no trajectory data returned "
            "from integration"
        ),
        "    Check that transverse_save=True in SimulationOptions",
    ]


def test_build_integration_trajectory_output_can_save_without_stability():
    outcome = build_integration_trajectory_output(
        _result(
            rider_trajectory={
                "z": [3.0, 7.0],
                "r": [0.0, 0.2],
                "pz": [1.0, 3.0],
                "pr": [0.1, 0.3],
                "t": [0.0, 1.0],
                "gamma": [10.0, 12.0],
            },
        ),
        _config(smoothness_enabled=False),
        run_num=8,
        rider_m_particle=1.0,
        metrics={},
        save_trajectory=True,
        trajectory_stride=1,
    )

    assert outcome.output_updates["_distance_info"] == {
        "z_start": 3.0,
        "z_end": 7.0,
        "num_steps": 2,
    }
    assert outcome.output_updates["trajectory"]["gamma"] == [10.0, 12.0]
    assert outcome.log_lines == [
        "  [DEBUG] Processing trajectory data for Run 8...",
        "  [INFO] Stability analysis DISABLED for Run 8 (smoothness_enabled=False)",
    ]


def test_build_integration_trajectory_output_rejects_failed_smoothness(monkeypatch):
    def fake_analyze_trajectory_smoothness(*_args, **_kwargs):
        return SimpleNamespace(
            passed=False,
            violations=[
                SimpleNamespace(description="oscillation"),
                SimpleNamespace(description="trend"),
                SimpleNamespace(description="extra"),
            ],
            oscillation_score=0.9,
            trend_smoothness_score=0.8,
            quality_summary="bad",
        )

    monkeypatch.setattr(
        "optimization.single_integration_helpers.analyze_trajectory_smoothness",
        fake_analyze_trajectory_smoothness,
    )
    metrics = {"max_percent_energy_gain": 1.0}

    outcome = build_integration_trajectory_output(
        _result(
            rider_trajectory={
                "z": [3.0, 7.0],
                "r": [0.0, 0.2],
                "pz": [1.0, 3.0],
                "pr": [0.1, 0.3],
                "t": [0.0, 1.0],
                "gamma": [10.0, 12.0],
            },
        ),
        _config(smoothness_enabled=True, smoothness_reject_on_violation=True),
        run_num=8,
        rider_m_particle=1.0,
        metrics=metrics,
        save_trajectory=False,
        trajectory_stride=1,
    )

    assert np.isnan(metrics["max_percent_energy_gain"])
    assert outcome.output_updates["stability_rejected"] is True
    assert outcome.output_updates["stability_analysis"] == {
        "passed": False,
        "num_violations": 3,
        "oscillation_score": 0.9,
        "trend_smoothness_score": 0.8,
        "quality": "bad",
    }
    assert "    Violations: 3" in outcome.log_lines
    assert "      - oscillation" in outcome.log_lines
    assert "      - trend" in outcome.log_lines
    assert "      - extra" not in outcome.log_lines


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
