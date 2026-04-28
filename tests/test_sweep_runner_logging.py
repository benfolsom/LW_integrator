"""Tests for CLI sweep-runner diagnostic formatting helpers."""

from __future__ import annotations

from lw_integrator.sweep_runner import (
    _build_cli_start_log_lines,
    _build_cli_timestep_log_lines,
    _format_aperture_for_start_log,
)


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
