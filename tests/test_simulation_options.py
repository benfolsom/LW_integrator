"""Tests for SimulationOptions serialization and fallback behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types import SimulationType
from lw_integrator.testbed_runner import (
    CORE_PARAM_DEFAULTS,
    DEFAULT_DRIVER_PARAMS,
    DEFAULT_RIDER_PARAMS,
    SimulationOptions,
)


def test_simulation_options_roundtrip_preserves_gamma_reconciliation_fields(
    tmp_path: Path,
):
    options = SimulationOptions(
        steps=321,
        seed=54321,
        simulation_type=SimulationType.CONDUCTING_WALL,
        output_dir=tmp_path / "outputs",
        config_dir=tmp_path / "configs",
        config_name="custom.json",
        self_consistency_convergence_mode="variable_geometry",
        self_consistency_gamma_reconciliation_method="FIXED_WEIGHTED",
        self_consistency_gamma_reconciliation_low_beta_threshold=0.85,
        self_consistency_gamma_reconciliation_high_beta_threshold=0.995,
        self_consistency_gamma_reconciliation_low_beta_weight=0.9,
        self_consistency_gamma_reconciliation_high_beta_weight=0.1,
        self_consistency_gamma_reconciliation_mid_beta_weight=0.6,
        self_consistency_gamma_reconciliation_fixed_weight=0.7,
        energy_monitor_halt_on_jump=True,
        adaptive_timestep_min_factor=1e-5,
        log_file_path="custom.log",
    )

    payload = options.to_dict()
    loaded = SimulationOptions.from_dict(payload)

    assert payload["simulation_type"] == "CONDUCTING_WALL"
    assert payload["output_dir"] == str(tmp_path / "outputs")
    assert loaded.steps == 321
    assert loaded.seed == 54321
    assert loaded.simulation_type == SimulationType.CONDUCTING_WALL
    assert loaded.output_dir == tmp_path / "outputs"
    assert loaded.config_dir == tmp_path / "configs"
    assert loaded.config_name == "custom.json"
    assert loaded.self_consistency_convergence_mode == "variable_geometry"
    assert loaded.self_consistency_gamma_reconciliation_method == "FIXED_WEIGHTED"
    assert loaded.self_consistency_gamma_reconciliation_low_beta_threshold == pytest.approx(
        0.85
    )
    assert loaded.self_consistency_gamma_reconciliation_high_beta_threshold == pytest.approx(
        0.995
    )
    assert loaded.self_consistency_gamma_reconciliation_low_beta_weight == pytest.approx(
        0.9
    )
    assert loaded.self_consistency_gamma_reconciliation_high_beta_weight == pytest.approx(
        0.1
    )
    assert loaded.self_consistency_gamma_reconciliation_mid_beta_weight == pytest.approx(
        0.6
    )
    assert loaded.self_consistency_gamma_reconciliation_fixed_weight == pytest.approx(
        0.7
    )
    assert loaded.energy_monitor_halt_on_jump is True
    assert loaded.adaptive_timestep_min_factor == pytest.approx(1e-5)
    assert loaded.log_file_path == "custom.log"


def test_simulation_options_from_dict_accepts_legacy_mode_alias_and_int_enum():
    payload = {
        "simulation_type": int(SimulationType.SWITCHING_WALL),
        "self_consistency_convergence_mode": "mass_shell_only",
    }

    options = SimulationOptions.from_dict(payload)

    assert options.simulation_type == SimulationType.SWITCHING_WALL
    assert options.self_consistency_convergence_mode == "fixed_geometry"


def test_simulation_options_from_dict_uses_defaults_for_missing_nested_payloads():
    options = SimulationOptions.from_dict({})

    assert options.rider_params == dict(DEFAULT_RIDER_PARAMS)
    assert options.driver_params == dict(DEFAULT_DRIVER_PARAMS)
    assert options.core_params == {
        key: (float(value) if isinstance(value, (int, float)) else value)
        for key, value in CORE_PARAM_DEFAULTS.items()
    }
    assert options.output_dir == Path("test_outputs/testbed_runs")


def test_simulation_options_from_dict_falls_back_on_invalid_numeric_values():
    payload = {
        "steps": "not-an-int",
        "seed": None,
        "adaptive_timestep_min_factor": "bad",
        "energy_monitor_threshold": "bad",
        "trajectory_interval": "bad",
    }

    options = SimulationOptions.from_dict(payload)

    assert options.steps == 1000
    assert options.seed == 12345
    assert options.adaptive_timestep_min_factor == pytest.approx(1e-4)
    assert options.energy_monitor_threshold == pytest.approx(2.0)
    assert options.trajectory_interval == 10

