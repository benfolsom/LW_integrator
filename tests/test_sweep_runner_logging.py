"""Tests for CLI sweep-runner diagnostic formatting helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.types import SimulationType
from lw_integrator.sweep_runner import (
    SweepRunner,
    _build_cli_start_log_lines,
    _build_cli_sweep_start_log_lines,
    _build_cli_timestep_log_lines,
    _evaluate_cli_stability,
    _format_aperture_for_start_log,
    _resolve_cli_driver_setup,
    _resolve_cli_rider_overrides,
    _resolve_cli_timestep_setup,
    run_sweep_from_config,
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
        "smoothness_enabled": True,
        "smoothness_window_size": 20,
        "smoothness_oscillation_threshold": 0.2,
        "smoothness_trend_threshold": 0.3,
        "smoothness_reject_on_violation": True,
        "smoothness_max_violations": 3,
        "auto_steps": False,
        "timestep_strategy": "fixed",
        "wall_z": 200.0,
        "auto_steps_distance_past_wall": 25.0,
        "auto_steps_target": 100,
        "timestep": 1e-7,
        "steps": 50,
        "z_cutoff_mode": "absolute",
    }
    defaults["calculate_timestep_for_energy"] = lambda **_kwargs: 1e-7
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


def test_sweep_runner_quiet_mode_suppresses_stdout_but_keeps_log_file(capsys, tmp_path):
    runner = SweepRunner(_config(), tmp_path, verbose=False)
    log_path = tmp_path / "sweep.log"
    tmp_path.mkdir(exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        runner.log_file = log_file
        runner._log("quiet message")

    assert capsys.readouterr().out == ""
    assert log_path.read_text(encoding="utf-8") == "[OPTIMIZATION] quiet message\n"


def test_run_sweep_from_config_respects_quiet_verbosity_override(
    capsys, monkeypatch, tmp_path
):
    config_path = tmp_path / "sweep.json"
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr("lw_integrator.sweep_runner.SweepRunner.run", lambda self: True)

    assert run_sweep_from_config(
        config_path=config_path,
        output_dir=tmp_path / "out",
        verbose=False,
        verbosity_overrides={"log_verbosity": "full"},
    )
    assert capsys.readouterr().out == ""


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


def test_resolve_cli_timestep_setup_fixed_path_restores_steps():
    config = _config(steps=50)
    captured = {}

    def calculate_timestep_for_energy(**kwargs):
        captured["steps_seen"] = config.steps
        captured.update(kwargs)
        return 2e-7

    config.calculate_timestep_for_energy = calculate_timestep_for_energy

    setup = _resolve_cli_timestep_setup(
        config,
        aperture=0.01,
        energy_gev=5.0,
        start_z=10.0,
        transv_offset_frac=0.25,
        rider_m_particle=1.0,
        sweep_overrides={},
    )

    assert setup.transv_offset == pytest.approx(0.0025)
    assert setup.steps == 50
    assert setup.timestep == pytest.approx(2e-7)
    assert setup.gamma == pytest.approx(5000.0 / 931.494)
    assert setup.beta > 0.0
    assert config.steps == 50
    assert captured["steps_seen"] == 50
    assert captured["driver_start_z"] == 1000.0


def test_resolve_cli_timestep_setup_uses_b2b_driver_start_and_gamma_offset():
    config = _config(simulation_type="BUNCH_TO_BUNCH", steps=30)
    captured = {}

    def calculate_timestep_for_energy(**kwargs):
        captured.update(kwargs)
        return 3e-7

    config.calculate_timestep_for_energy = calculate_timestep_for_energy

    setup = _resolve_cli_timestep_setup(
        config,
        aperture=0.01,
        energy_gev=5.0,
        start_z=10.0,
        transv_offset_frac=0.25,
        rider_m_particle=1.0,
        sweep_overrides={"driver_starting_distance": 700.0},
    )

    assert captured["driver_start_z"] == 700.0
    assert setup.gamma == pytest.approx(5000.0 / 931.494 + 1.0)
    assert config.steps == 30


def test_resolve_cli_timestep_setup_restores_steps_after_error():
    config = _config(steps=50)

    def calculate_timestep_for_energy(**_kwargs):
        raise RuntimeError("bad timestep")

    config.calculate_timestep_for_energy = calculate_timestep_for_energy

    with pytest.raises(RuntimeError, match="bad timestep"):
        _resolve_cli_timestep_setup(
            config,
            aperture=0.01,
            energy_gev=5.0,
            start_z=10.0,
            transv_offset_frac=0.25,
            rider_m_particle=1.0,
            sweep_overrides={},
        )

    assert config.steps == 50


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
        ("[OPTIMIZATION]     timestep h=1.0000e-07 ns " "(proper time = dt/gamma)"),
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


def test_evaluate_cli_stability_reports_disabled_without_metric_updates():
    outcome = _evaluate_cli_stability(
        _config(smoothness_enabled=False),
        SimpleNamespace(rider_trajectory={"gamma": [1.0, 2.0]}),
        {"existing": 1.0},
        rider_m_particle=1.0,
        run_num=4,
        aperture=0.01,
        energy_gev=5.0,
        start_z=10.0,
        transv_offset=0.0,
    )

    assert outcome.metrics_updates == {}
    assert outcome.rejection_record is None
    assert outcome.log_lines == [
        "[OPTIMIZATION]   [DEBUG] Processing trajectory data for Run 4...",
        "[OPTIMIZATION]   [INFO] Stability analysis DISABLED for Run 4",
    ]


def test_evaluate_cli_stability_reports_missing_trajectory():
    outcome = _evaluate_cli_stability(
        _config(),
        SimpleNamespace(rider_trajectory=None),
        {},
        rider_m_particle=1.0,
        run_num=4,
        aperture=0.01,
        energy_gev=5.0,
        start_z=10.0,
        transv_offset=0.0,
    )

    assert outcome.metrics_updates == {}
    assert outcome.rejection_record is None
    assert outcome.log_lines[-1] == (
        "[OPTIMIZATION]   [WARNING] No trajectory data for Run 4"
    )


def test_evaluate_cli_stability_builds_rejection_record(monkeypatch):
    def fake_analyze_trajectory_smoothness(*_args, **_kwargs):
        return SimpleNamespace(
            passed=False,
            violations=["oscillation", "trend"],
            quality_summary="bad quality",
        )

    monkeypatch.setattr(
        "lw_integrator.sweep_runner.analyze_trajectory_smoothness",
        fake_analyze_trajectory_smoothness,
    )

    outcome = _evaluate_cli_stability(
        _config(),
        SimpleNamespace(rider_trajectory={"gamma": [1.0, 2.0]}),
        {"existing": 1.0},
        rider_m_particle=1.0,
        run_num=6,
        aperture=0.01,
        energy_gev=5.0,
        start_z=10.0,
        transv_offset=0.0025,
    )

    assert outcome.metrics_updates == {
        "smoothness_passed": False,
        "smoothness_violations": 2,
    }
    assert outcome.rejection_record == {
        "success": False,
        "error": "Smoothness violation: 2 violations",
        "parameters": {
            "aperture": 0.01,
            "energy_gev": 5.0,
            "start_z": 10.0,
            "transv_offset": 0.0025,
        },
        "metrics": {
            "existing": 1.0,
            "smoothness_passed": False,
            "smoothness_violations": 2,
        },
    }
    assert "[OPTIMIZATION]     Quality: bad quality" in outcome.log_lines
    assert outcome.log_lines[-1] == (
        "[OPTIMIZATION]   [REJECT] Run 6 rejected due to numerical instability"
    )


def test_build_cli_sweep_start_log_lines_formats_wall_mode_summary():
    lines = _build_cli_sweep_start_log_lines(
        _config(timestep_strategy="auto_distance"),
        param_grids={"energy": [5.0, 6.0], "start_z": [10.0]},
        total_runs=2,
    )

    assert lines == [
        "Starting BLIND SWEEP (Grid Search): 2 total runs",
        f"  Simulation type: {SimulationType.CONDUCTING_WALL}",
        "  energy: 2 points from 5.0000e+00 to 6.0000e+00",
        "  start_z: 1.0000e+01 (fixed)",
        "  Timestep strategy: auto_distance",
        "    Distance past wall: 25.0 mm",
        "    Target steps for timestep calculation: 100",
        "    All particles will travel to consistent z regardless of energy",
        "  z_cutoff_mode: absolute",
    ]


def test_build_cli_sweep_start_log_lines_includes_b2b_fixed_parameters():
    lines = _build_cli_sweep_start_log_lines(
        _config(simulation_type="BUNCH_TO_BUNCH"),
        param_grids={"energy": [5.0]},
        total_runs=1,
    )

    assert "  Fixed rider parameters:" in lines
    assert "    m_particle: 1.0000e+00 amu" in lines
    assert "  Fixed driver parameters:" in lines
    assert "    energy_gev: 6.0000" in lines
    assert "    starting_distance: 900.00" in lines
