"""Keep flat sweep files out of the nested testbed loader."""

import json
from pathlib import Path

import pytest

from lw_integrator.testbed_runner import SimulationOptions, load_config

FIXTURE = Path(__file__).parent / "fixtures" / "pseudo_grid_flat_sweep.json"


def test_testbed_loader_rejects_captured_flat_sweep():
    with pytest.raises(ValueError, match="use --sweep-config"):
        load_config(FIXTURE)


@pytest.mark.parametrize(
    "key", ["sweep_parameters", "energy_min", "energy_max", "energy_points"]
)
def test_testbed_options_reject_sweep_markers(key):
    with pytest.raises(ValueError, match="--sweep-config"):
        SimulationOptions.from_dict({key: {}})


def test_nested_testbed_loader_preserves_beams_and_timestep(tmp_path):
    rider = {"m_particle": 1.007276466812, "charge_sign": 1.0, "pcount": 42}
    driver = {"m_particle": 0.00054857990907, "charge_sign": -1.0, "pcount": 42}
    path = tmp_path / "nested.json"
    path.write_text(
        json.dumps(
            {
                "rider_params": rider,
                "driver_params": driver,
                "core_params": {"time_step": 0.026912043446720352},
            }
        )
    )
    options = load_config(path)
    for key, value in rider.items():
        assert options.rider_params[key] == value
    for key, value in driver.items():
        assert options.driver_params[key] == value
    assert options.core_params["time_step"] == 0.026912043446720352


def test_default_testbed_options_and_single_beam_remain_supported():
    assert SimulationOptions.from_dict({}).rider_params
    options = SimulationOptions.from_dict(
        {
            "rider_params": {"pcount": 7},
            "energy_min": 0.01,
        }
    )
    assert options.rider_params["pcount"] == 7
