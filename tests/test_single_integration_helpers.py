"""Tests for pure single-integration result helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from optimization.single_integration_helpers import (
    build_integration_metrics,
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
