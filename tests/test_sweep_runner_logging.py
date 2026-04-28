"""Tests for CLI sweep-runner diagnostic formatting helpers."""

from __future__ import annotations

from types import SimpleNamespace

from core.types import SimulationType
from lw_integrator.sweep_runner import (
    _build_cli_start_log_lines,
    _build_cli_timestep_log_lines,
    _format_aperture_for_start_log,
    _resolve_cli_driver_setup,
    _resolve_cli_rider_overrides,
)


def _config(**overrides):
    defaults = {
        "simulation_type": SimulationType.CONDUCTING_WALL,
        "m_particle": 1.0,
        "charge_sign": 1.0,
        "pcount": 3,
        "transv_mom": 0.0,
        "transv_dist": 1e-4,
        "stripped_ions": 1.0,
        "macroparticle_charge_multiplier": 2.0,
        "macroparticle_sigma_multiplier": 3.0,
        "driver_m_particle": 2.0,
        "driver_charge_sign": -1.0,
        "driver_pcount": 5,
        "driver_transv_mom": 2e-4,
        "driver_transv_dist": 3e-4,
        "driver_starting_distance": 900.0,
        "driver_stripped_ions": 4.0,
        "driver_energy_gev": 6.0,
        "driver_direction": "-z",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_resolve_cli_rider_overrides_prefers_sweep_values():
    rider = _resolve_cli_rider_overrides(
        _config(),
        {
            "rider_m_particle": 4.0,
            "rider_pcount": 9.0,
            "macroparticle_charge_multiplier": 8.0,
        },
    )

    assert rider.m_particle == 4.0
    assert rider.charge_sign == 1.0
    assert rider.pcount == 9
    assert rider.transv_dist == 1e-4
    assert rider.macroparticle_charge_multiplier == 8.0
    assert rider.macroparticle_sigma_multiplier == 3.0


def test_resolve_cli_driver_setup_returns_none_for_wall_mode():
    setup = _resolve_cli_driver_setup(_config(), {"driver_energy_gev": 10.0})

    assert setup.params is None
    assert setup.log_line is None


def test_resolve_cli_driver_setup_builds_b2b_driver_params_and_log_line():
    setup = _resolve_cli_driver_setup(
        _config(simulation_type="BUNCH_TO_BUNCH", driver_direction="+z"),
        {
            "driver_m_particle": 3.0,
            "driver_pcount": 7.0,
            "driver_energy_gev": 8.0,
            "driver_starting_distance": 700.0,
        },
    )

    assert setup.params is not None
    assert setup.params["m_particle"] == 3.0
    assert setup.params["pcount"] == 7
    assert setup.params["starting_distance"] == 700.0
    assert setup.params["starting_Pz"] > 0.0
    assert setup.log_line is not None
    assert "energy=8.0000 GeV" in setup.log_line
    assert "(+z)" in setup.log_line
    assert "pcount=7" in setup.log_line


def test_format_aperture_for_start_log_matches_existing_precision_branches():
    assert _format_aperture_for_start_log(1.25) == "1.2"
    assert _format_aperture_for_start_log(0.012345) == "0.0123"
    assert _format_aperture_for_start_log(0.0012345) == "0.001234"


def test_build_cli_timestep_log_lines_includes_auto_distance_details():
    lines = _build_cli_timestep_log_lines(
        run_num=4,
        timestep_strategy="auto_distance",
        energy_gev=5.0,
        rider_m_particle=1.0,
        gamma=2.0,
        beta=0.5,
        timestep=1e-7,
        steps=10,
        start_z=10.0,
        wall_z=200.0,
        auto_steps_distance_past_wall=25.0,
        auto_steps_target=500,
    )

    assert lines[0] == "[OPTIMIZATION]   [TIMESTEP] Run 4 strategy 'auto_distance':"
    assert "[OPTIMIZATION]     E=5.0000 GeV, m=1.0000e+00 amu" in lines
    assert "[OPTIMIZATION]     gamma=2.00, beta=0.50000000" in lines
    assert "[OPTIMIZATION]     distance_per_step = β·γ·c·h = 0.0000 mm" in lines
    assert "[OPTIMIZATION]     expected_total_distance = 0.00 mm" in lines
    assert "[OPTIMIZATION]     distance_to_wall = 190.00 mm" in lines
    assert lines[-1] == "[OPTIMIZATION]     target_steps=500"


def test_build_cli_timestep_log_lines_omits_distance_details_for_fixed_strategy():
    lines = _build_cli_timestep_log_lines(
        run_num=1,
        timestep_strategy="fixed",
        energy_gev=5.0,
        rider_m_particle=1.0,
        gamma=2.0,
        beta=0.5,
        timestep=1e-7,
        steps=10,
        start_z=10.0,
        wall_z=200.0,
        auto_steps_distance_past_wall=25.0,
        auto_steps_target=500,
    )

    assert lines == [
        "[OPTIMIZATION]   [TIMESTEP] Run 1 strategy 'fixed':",
        "[OPTIMIZATION]     E=5.0000 GeV, m=1.0000e+00 amu",
        "[OPTIMIZATION]     gamma=2.00, beta=0.50000000",
        (
            "[OPTIMIZATION]     timestep h=1.0000e-07 ns "
            "(proper time = dt/gamma)"
        ),
        "[OPTIMIZATION]     steps=10",
    ]


def test_build_cli_start_log_lines_formats_start_and_params_lines():
    assert _build_cli_start_log_lines(
        run_num=2,
        total_runs=9,
        aperture=0.012345,
        energy_gev=5.0,
        start_z=10.0,
        timestep=1e-7,
        steps=100,
    ) == [
        "[OPTIMIZATION] [START] Run 2/9: a=0.0123mm, E=5.00GeV",
        "[OPTIMIZATION]   [PARAMS] z=10.00mm, h=1.0000e-07ns, N=100",
    ]
