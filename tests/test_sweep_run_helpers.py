"""Tests for pure sweep run-preparation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.types import SimulationType
import optimization.sweep_run_helpers as sweep_run_helpers
from optimization.sweep_run_helpers import (
    build_full_debug_parameter_log_lines,
    resolve_sweep_run_parameters,
    resolve_sweep_timestep,
)


def test_module_exposes_only_maintained_public_helpers():
    assert sweep_run_helpers.__all__ == [
        "SweepRunParameters",
        "SweepTimestepResolution",
        "build_full_debug_parameter_log_lines",
        "resolve_sweep_run_parameters",
        "resolve_sweep_timestep",
    ]


def _config(**overrides):
    defaults = {
        "simulation_type": SimulationType.CONDUCTING_WALL,
        "m_particle": 1.0,
        "charge_sign": 1.0,
        "pcount": 3,
        "transv_mom": 0.0,
        "transv_dist": 1e-4,
        "stripped_ions": 1.0,
        "macroparticle_charge_multiplier": 1.0,
        "macroparticle_sigma_multiplier": 1.0,
        "macroparticle_enabled": False,
        "macroparticle_use_momentum_errors": True,
        "driver_m_particle": 2.0,
        "driver_charge_sign": -1.0,
        "driver_pcount": 5,
        "driver_transv_mom": 2e-4,
        "driver_transv_dist": 3e-4,
        "driver_starting_distance": 900.0,
        "driver_starting_Pz": -10.0,
        "driver_stripped_ions": 2.0,
        "driver_direction": "-z",
        "timestep_strategy": "fixed",
        "timestep": 1e-7,
        "steps": 100,
        "wall_z": 200.0,
        "auto_steps": False,
        "auto_steps_distance_past_wall": 10.0,
        "auto_steps_target": 100,
        "target_distance_mm": 25.0,
    }
    defaults.update(overrides)

    def calculate_timestep_for_energy(
        energy_gev,
        m_particle_amu,
        *,
        wall_z,
        start_z,
        driver_start_z,
    ):
        assert energy_gev > 0
        assert m_particle_amu > 0
        return (abs(driver_start_z - start_z) + abs(wall_z)) * 1e-9

    defaults.setdefault("calculate_timestep_for_energy", calculate_timestep_for_energy)
    return SimpleNamespace(**defaults)


def test_resolve_sweep_run_parameters_returns_none_for_missing_energy():
    params = {
        "start_z": 10.0,
        "transverse_offset_fraction": 0.25,
    }

    assert resolve_sweep_run_parameters(_config(), params) is None


def test_resolve_sweep_run_parameters_handles_wall_mode_offsets():
    params = {
        "energy": 5.0,
        "aperture": 0.01,
        "start_z": 10.0,
        "transverse_offset_fraction": 0.25,
        "rider_pcount": 7,
    }

    resolved = resolve_sweep_run_parameters(_config(), params)

    assert resolved is not None
    assert resolved.energy == 5.0
    assert resolved.rider_pcount == 7
    assert resolved.transv_offset == pytest.approx(0.0025)
    assert resolved.driver_params is None


def test_resolve_sweep_run_parameters_handles_string_b2b_driver_energy():
    params = {
        "initial_energy_gev": 5.0,
        "start_z": 10.0,
        "transverse_offset_fraction": 0.25,
        "driver_energy_gev": 2.0,
        "driver_starting_distance": 700.0,
    }

    resolved = resolve_sweep_run_parameters(
        _config(simulation_type="BUNCH_TO_BUNCH"), params
    )

    assert resolved is not None
    assert resolved.transv_offset == 0.25
    assert resolved.driver_params is not None
    assert resolved.driver_params["starting_distance"] == 700.0
    assert resolved.driver_params["starting_Pz"] < 0.0


def test_resolve_sweep_timestep_fixed_strategy_keeps_config_values():
    params = {
        "energy": 5.0,
        "aperture": 0.01,
        "start_z": 10.0,
        "transverse_offset_fraction": 0.25,
    }
    run_params = resolve_sweep_run_parameters(_config(), params)

    resolution = resolve_sweep_timestep(
        _config(),
        params,
        run_params,
        run_num=1,
        use_full_debug=True,
    )

    assert resolution.timestep == pytest.approx(1e-7)
    assert resolution.steps == 100
    assert resolution.expected_distance > 0.0
    assert resolution.log_lines == []


def test_resolve_sweep_timestep_energy_strategy_logs_diagnostics():
    config = _config(timestep_strategy="auto_distance")
    params = {
        "initial_energy_gev": 5.0,
        "start_z": 10.0,
        "transverse_offset_fraction": 0.25,
        "driver_starting_distance": 700.0,
    }
    run_params = resolve_sweep_run_parameters(
        _config(simulation_type="BUNCH_TO_BUNCH"), params
    )

    resolution = resolve_sweep_timestep(
        SimpleNamespace(**{**config.__dict__, "simulation_type": "BUNCH_TO_BUNCH"}),
        params,
        run_params,
        run_num=2,
        use_full_debug=True,
    )

    assert resolution.timestep > 0.0
    assert resolution.steps == config.steps
    assert any("[TIMESTEP] Run 2" in line for line in resolution.log_lines)
    assert any("target_distance=25.00 mm" in line for line in resolution.log_lines)


def test_build_full_debug_parameter_log_lines_includes_macroparticle_fields():
    params = {
        "energy": 5.0,
        "aperture": 0.01,
        "start_z": 10.0,
        "transverse_offset_fraction": 0.25,
        "macroparticle_charge_multiplier": 4.0,
        "macroparticle_sigma_multiplier": 2.0,
    }
    config = _config(macroparticle_enabled=True)
    run_params = resolve_sweep_run_parameters(config, params)

    lines = build_full_debug_parameter_log_lines(
        config,
        run_params,
        run_num=3,
        total_runs=9,
    )

    assert lines[0] == "  [PARAMS] Run 3/9 - All parameters:"
    assert any("macroparticle_enabled: True" in line for line in lines)
    assert any("macroparticle_charge_multiplier: 4.0000" in line for line in lines)
