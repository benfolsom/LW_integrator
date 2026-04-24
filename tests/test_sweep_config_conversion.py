"""Tests for converting JSON sweep configs into OptimizationConfig fields."""

from __future__ import annotations

from dataclasses import fields

from core.types import SimulationType
from lw_integrator.sweep_runner import _convert_json_config_to_dataclass
from optimization.config import OptimizationConfig


def _build_optimization_config(converted: dict) -> OptimizationConfig:
    valid_fields = {field.name for field in fields(OptimizationConfig)}
    return OptimizationConfig(
        **{key: value for key, value in converted.items() if key in valid_fields}
    )


def test_convert_json_config_maps_fixed_driver_parameters():
    config_dict = {
        "simulation_type": "BUNCH_TO_BUNCH",
        "sweep_parameters": {
            "driver_m_particle": {"enabled": False, "fixed_value": 1.0},
            "driver_charge_sign": {"enabled": False, "fixed_value": -1.0},
            "driver_pcount": {"enabled": False, "fixed_value": 1},
            "driver_transv_mom": {"enabled": False, "fixed_value": 1.0e-5},
            "driver_transv_dist": {"enabled": False, "fixed_value": 1.0e-5},
            "driver_starting_distance": {
                "enabled": False,
                "fixed_value": 1000.0,
            },
            "driver_stripped_ions": {"enabled": False, "fixed_value": 1e9},
        },
    }

    converted = _convert_json_config_to_dataclass(config_dict)
    config = _build_optimization_config(converted)

    assert converted["simulation_type"] == SimulationType.BUNCH_TO_BUNCH
    assert config.driver_m_particle == 1.0
    assert config.driver_charge_sign == -1.0
    assert config.driver_pcount == 1
    assert config.driver_transv_mom == 1.0e-5
    assert config.driver_transv_dist == 1.0e-5
    assert config.driver_starting_distance == 1000.0
    assert config.driver_stripped_ions == 1e9


def test_convert_json_config_normalizes_driver_energy_sweep_range():
    config_dict = {
        "sweep_parameters": {
            "driver_energy_gev": {
                "enabled": True,
                "min": -10.0,
                "max": -1.0,
                "points": 4,
                "log": True,
            }
        }
    }

    converted = _convert_json_config_to_dataclass(config_dict)

    assert converted["driver_energy_range"] == (1.0, 10.0)
    assert converted["driver_energy_points"] == 4
    assert converted["driver_energy_log_scale"] is True

