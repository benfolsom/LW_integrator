"""Focused tests for plotting entry points and sweep plot integration."""

from __future__ import annotations

from pathlib import Path

import lw_integrator.sweep_heatmap as sweep_heatmap
import numpy as np
import pytest
from lw_integrator import logcache_plotter, plot_latest_live
from optimization.results_mixins import OptimizationResultsMixin


class _ResultsHarness(OptimizationResultsMixin):
    def __init__(self) -> None:
        self.logs = []

    def _log_result(self, message: str) -> None:
        self.logs.append(message)

    def _plot_single_trajectory(self, *_args, **_kwargs) -> None:
        raise AssertionError("trajectory plot should not be called in this test")


def _write_log(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _load_project_scripts() -> dict[str, str]:
    scripts = {}
    in_scripts = False
    for line in Path("pyproject.toml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts and stripped.startswith("["):
            break
        if in_scripts and "=" in stripped:
            name, target = stripped.split("=", 1)
            scripts[name.strip()] = target.strip().strip('"')
    return scripts


def test_generate_sweep_heatmap_main_accepts_argv(monkeypatch):
    captured = {}

    def fake_generate_heatmap(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(sweep_heatmap, "generate_heatmap", fake_generate_heatmap)

    exit_code = sweep_heatmap.main(
        [
            "results/sweeps/example",
            "--gain-filter",
            "all",
            "--num-contours",
            "20",
            "--grey-zero",
        ]
    )

    assert exit_code == 0
    assert captured["sweep_dir"] == "results/sweeps/example"
    assert captured["gain_filter"] == "all"
    assert captured["num_contours_low"] == 5
    assert captured["num_contours_high"] == 15


def test_generate_sweep_heatmap_resolves_contour_counts():
    assert sweep_heatmap.resolve_contour_counts(None, False, 4, 7) == (4, 7)
    assert sweep_heatmap.resolve_contour_counts(20, False, 4, 7) == (8, 12)
    assert sweep_heatmap.resolve_contour_counts(20, True, 4, 7) == (5, 15)


def test_sweep_heatmap_collapses_energy_aliases_when_detecting_parameters():
    data = {
        "results": [
            {
                "parameters": {
                    "initial_energy_gev": 1.0,
                    "energy_gev": 1.0,
                    "aperture_radius": 0.1,
                    "timestep": 1e-3,
                }
            },
            {
                "parameters": {
                    "initial_energy_gev": 2.0,
                    "energy_gev": 2.0,
                    "aperture_radius": 0.2,
                    "timestep": 2e-3,
                }
            },
        ]
    }

    swept_params, labels = sweep_heatmap.detect_swept_parameters(data)

    assert swept_params == ["initial_energy_gev", "aperture_radius"]
    assert labels["initial_energy_gev"] == "Initial Energy (GeV)"
    assert labels["energy_gev"] == "Initial Energy (GeV)"


def test_sweep_heatmap_extract_data_uses_energy_alias_fallback():
    data = {
        "results": [
            {
                "parameters": {"energy_gev": 1.0, "aperture_radius": 0.1},
                "metrics": {"percent_delta_e": 3.0},
            },
            {
                "parameters": {"energy_gev": 2.0, "aperture_radius": 0.2},
                "metrics": {"percent_delta_e": -4.0},
            },
        ]
    }

    param1, param2, gains = sweep_heatmap.extract_data(
        data,
        param1_name="initial_energy_gev",
        param2_name="aperture_radius",
        gain_filter="all",
    )

    np.testing.assert_allclose(param1, [1.0, 2.0])
    np.testing.assert_allclose(param2, [0.1, 0.2])
    np.testing.assert_allclose(gains, [3.0, -4.0])


def test_generate_sweep_heatmap_rejects_removed_legacy_aliases():
    with pytest.raises(SystemExit) as excinfo:
        sweep_heatmap.main(["results/sweeps/example", "--energy-min", "1.0"])

    assert excinfo.value.code == 2


def test_generate_sweep_heatmap_exposes_only_supported_public_helpers():
    assert not hasattr(sweep_heatmap, "parse_args")
    assert sweep_heatmap.__all__ == [
        "generate_heatmap",
        "load_sweep_results",
        "main",
        "resolve_contour_counts",
    ]


def test_plot_latest_live_resolves_newest_log(tmp_path: Path):
    logcache = tmp_path / "logcache"
    logcache.mkdir()
    older = logcache / "older_sweep.log"
    newer = logcache / "newer_sweep.log"
    older.write_text("", encoding="utf-8")
    newer.write_text("", encoding="utf-8")
    older.touch()
    newer.touch()
    newer.touch()

    latest = plot_latest_live.resolve_latest_sweep_log(logcache)

    assert latest == newer


def test_plot_latest_live_forwards_to_live_plotter(tmp_path: Path, monkeypatch):
    logcache = tmp_path / "logcache"
    logcache.mkdir()
    latest_log = logcache / "run_sweep.log"
    latest_log.write_text("", encoding="utf-8")

    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(plot_latest_live, "plot_from_logcache_main", fake_main)

    exit_code = plot_latest_live.main(
        ["--logcache", str(logcache), "--output", "live.png", "--interval", "7"]
    )

    assert exit_code == 0
    assert captured["argv"] == [
        "--live",
        str(latest_log),
        "--interval",
        "7",
        "--output",
        "live.png",
    ]


def test_plot_latest_live_reports_missing_logs(tmp_path: Path, capsys):
    logcache = tmp_path / "logcache"
    logcache.mkdir()

    exit_code = plot_latest_live.main(["--logcache", str(logcache)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No sweep log files found" in captured.out


def test_plot_latest_live_exposes_only_supported_public_helpers():
    assert not hasattr(plot_latest_live, "parse_args")
    assert plot_latest_live.__all__ == ["main", "resolve_latest_sweep_log"]


def test_project_scripts_expose_maintained_plotting_tools():
    scripts = _load_project_scripts()

    assert scripts["lw-generate-sweep-heatmap"] == "lw_integrator.sweep_heatmap:main"
    assert scripts["lw-plot-latest-live"] == "lw_integrator.plot_latest_live:main"
    assert (
        scripts["lw-plot-from-logcache-live"] == "lw_integrator.logcache_plotter:main"
    )
    assert scripts["lw-plot-trajectory"] == "lw_integrator.trajectory_plotter:main"


def test_logcache_plotter_exposes_only_supported_public_helpers():
    assert not hasattr(logcache_plotter, "parse_args")
    assert logcache_plotter.__all__ == [
        "find_latest_log",
        "main",
        "parse_sweep_log",
    ]


def test_logcache_plotter_static_main_calls_contour_plot(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "run_sweep.log"
    log_path.write_text("placeholder", encoding="utf-8")
    output_path = tmp_path / "plot.png"
    captured = {}

    def fake_parse(log_file, verbose=True, max_gain_percent=30.0):
        captured["parse"] = {
            "log_file": log_file,
            "verbose": verbose,
            "max_gain_percent": max_gain_percent,
        }
        return (
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
            np.array([3.0, 4.0]),
            np.array([]),
            np.array([]),
            np.array([]),
            {"total": 2, "completed": 2},
            {"x_param_name": "aperture"},
        )

    def fake_contour(*args, **kwargs):
        captured["contour"] = {"args": args, "kwargs": kwargs}

    monkeypatch.setattr(logcache_plotter, "parse_sweep_log", fake_parse)
    monkeypatch.setattr(logcache_plotter, "_create_contour_plot", fake_contour)

    exit_code = logcache_plotter.main(
        [str(log_path), "--output", str(output_path), "--max-gain", "12.5", "--log-x"]
    )

    assert exit_code == 0
    assert captured["parse"] == {
        "log_file": str(log_path),
        "verbose": True,
        "max_gain_percent": 12.5,
    }
    contour_args = captured["contour"]["args"]
    np.testing.assert_allclose(contour_args[0], [1.0, 2.0])
    np.testing.assert_allclose(contour_args[1], [0.1, 0.2])
    np.testing.assert_allclose(contour_args[2], [3.0, 4.0])
    assert contour_args[3] == str(output_path)
    assert captured["contour"]["kwargs"]["log_x"] is True
    assert captured["contour"]["kwargs"]["log_y"] is False


def test_logcache_plotter_interpolation_falls_back_between_methods(monkeypatch):
    calls = []

    def fake_griddata(_points, _values, grid, method):
        calls.append(method)
        if method == "nearest":
            raise RuntimeError("nearest failed")
        return np.ones_like(grid[0])

    monkeypatch.setattr(logcache_plotter, "griddata", fake_griddata)

    grid_x, grid_y = np.meshgrid(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
    interpolated, method = logcache_plotter._interpolate_gain_grid(
        np.array([1.0, 2.0]),
        np.array([0.1, 0.2]),
        np.array([3.0, 4.0]),
        grid_x,
        grid_y,
    )

    assert calls == ["nearest", "linear"]
    assert method == "linear"
    np.testing.assert_allclose(interpolated, np.ones_like(grid_x))


def test_parse_sweep_log_uses_only_most_recent_sweep(tmp_path: Path):
    log_path = _write_log(
        tmp_path / "multi_sweep.log",
        [
            "[OPTIMIZATION] Starting BLIND SWEEP (Grid Search): 4 total runs",
            "[START] Run 1/4: a=0.001mm, E=0.010GeV",
            "[METRICS] max_percent_energy_gain: 1.5%",
            "[START] Run 2/4: a=0.002mm, E=0.020GeV",
            "[METRICS] max_percent_energy_gain: 2.5%",
            "",
            "[OPTIMIZATION] Starting BLIND SWEEP (Grid Search): 4 total runs",
            "[START] Run 1/4: a=0.003mm, E=0.030GeV",
            "[METRICS] max_percent_energy_gain: 3.5%",
            "[START] Run 2/4: a=0.004mm, E=0.040GeV",
            "[METRICS] max_percent_energy_gain: 4.5%",
            "",
            "[OPTIMIZATION] Starting BLIND SWEEP (Grid Search): 4 total runs",
            "[START] Run 1/4: a=0.005mm, E=0.050GeV",
            "[METRICS] max_percent_energy_gain: 5.5%",
            "[START] Run 2/4: a=0.006mm, E=0.060GeV",
            "[METRICS] max_percent_energy_gain: 6.5%",
            "[START] Run 3/4: a=0.007mm, E=0.070GeV",
            "[METRICS] max_percent_energy_gain: -1.5%",
            "[START] Run 4/4: a=0.008mm, E=0.080GeV",
            "[METRICS] max_percent_energy_gain: 0.0%",
        ],
    )

    (
        energies_pos,
        x_values_pos,
        gains_pos,
        energies_neg,
        x_values_neg,
        gains_neg,
        stats,
        param_metadata,
    ) = logcache_plotter.parse_sweep_log(log_path, verbose=False)

    np.testing.assert_allclose(energies_pos, [0.05, 0.06])
    np.testing.assert_allclose(x_values_pos, [0.005, 0.006])
    np.testing.assert_allclose(gains_pos, [5.5, 6.5])
    np.testing.assert_allclose(energies_neg, [0.07, 0.08])
    np.testing.assert_allclose(x_values_neg, [0.007, 0.008])
    np.testing.assert_allclose(gains_neg, [-1.5, 0.0])
    assert stats == {
        "total": 4,
        "completed": 4,
        "positive_gains": 2,
        "negative_gains": 2,
        "last_run": 4,
        "sweep_count": 3,
    }
    assert param_metadata["sweep_type"] == "CONDUCTING_WALL"
    assert param_metadata["x_param_name"] == "aperture"


def test_parse_sweep_log_separates_positive_and_non_positive_gains(tmp_path: Path):
    log_path = _write_log(
        tmp_path / "gain_filter.log",
        [
            "[OPTIMIZATION] Starting BLIND SWEEP (Grid Search): 5 total runs",
            "[START] Run 1/5: a=0.001mm, E=0.010GeV",
            "[METRICS] max_percent_energy_gain: 1.5%",
            "[START] Run 2/5: a=0.002mm, E=0.020GeV",
            "[METRICS] max_percent_energy_gain: -2.5%",
            "[START] Run 3/5: a=0.003mm, E=0.030GeV",
            "[METRICS] max_percent_energy_gain: 0.0%",
            "[START] Run 4/5: a=0.004mm, E=0.040GeV",
            "[METRICS] max_percent_energy_gain: 3.5%",
            "[START] Run 5/5: a=0.005mm, E=0.050GeV",
            "[METRICS] max_percent_energy_gain: -0.1%",
        ],
    )

    (
        energies_pos,
        x_values_pos,
        gains_pos,
        energies_neg,
        x_values_neg,
        gains_neg,
        stats,
        _param_metadata,
    ) = logcache_plotter.parse_sweep_log(log_path, verbose=False)

    np.testing.assert_allclose(energies_pos, [0.01, 0.04])
    np.testing.assert_allclose(x_values_pos, [0.001, 0.004])
    np.testing.assert_allclose(gains_pos, [1.5, 3.5])
    np.testing.assert_allclose(energies_neg, [0.02, 0.03, 0.05])
    np.testing.assert_allclose(x_values_neg, [0.002, 0.003, 0.005])
    np.testing.assert_allclose(gains_neg, [-2.5, 0.0, -0.1])
    assert stats["completed"] == 5
    assert stats["positive_gains"] == 2
    assert stats["negative_gains"] == 3


def test_parse_sweep_log_reads_truncated_b2b_driver_energy_axis(tmp_path: Path):
    log_path = _write_log(
        tmp_path / "b2b_truncated.log",
        [
            "[OPTIMIZATION] Starting BLIND SWEEP (Grid Search): 1 total runs",
            (
                "Run #   1 | initial_energy_gev=5.0 driver_energy_gev=-2.0 "
                "| ΔE=0.50MeV γ_i=10.0 γ_f=11.0 | SUCCESS"
            ),
        ],
    )

    (
        energies_pos,
        x_values_pos,
        gains_pos,
        _energies_neg,
        _x_values_neg,
        _gains_neg,
        stats,
        param_metadata,
    ) = logcache_plotter.parse_sweep_log(log_path, verbose=False)

    np.testing.assert_allclose(energies_pos, [5.0])
    np.testing.assert_allclose(x_values_pos, [2.0])
    np.testing.assert_allclose(gains_pos, [10.0])
    assert stats["completed"] == 1
    assert param_metadata["sweep_type"] == "BUNCH_TO_BUNCH"
    assert param_metadata["x_param_name"] == "driver_energy_gev"


def test_parse_sweep_log_reads_cli_params_block_b2b_axis(tmp_path: Path):
    log_path = _write_log(
        tmp_path / "b2b_params_block.log",
        [
            "[OPTIMIZATION] Starting BLIND SWEEP (Grid Search): 1 total runs",
            "[OPTIMIZATION] [PARAMS] Run 1/1 - All parameters:",
            "[OPTIMIZATION]     energy: 5.0 GeV",
            "[OPTIMIZATION]     driver_starting_distance: 900.0 mm",
            "[OPTIMIZATION] [RESULT] Run 1 metrics:",
            "[OPTIMIZATION] max_percent_energy_gain: 1.25%",
        ],
    )

    (
        energies_pos,
        x_values_pos,
        gains_pos,
        _energies_neg,
        _x_values_neg,
        _gains_neg,
        stats,
        param_metadata,
    ) = logcache_plotter.parse_sweep_log(log_path, verbose=False)

    np.testing.assert_allclose(energies_pos, [5.0])
    np.testing.assert_allclose(x_values_pos, [900.0])
    np.testing.assert_allclose(gains_pos, [1.25])
    assert stats["completed"] == 1
    assert param_metadata["sweep_type"] == "BUNCH_TO_BUNCH"
    assert param_metadata["x_param_name"] == "driver_starting_distance"


def test_generate_summary_plots_logs_postprocessing_heatmap_command(
    tmp_path: Path, monkeypatch
):
    harness = _ResultsHarness()

    def fail_if_called(_argv):
        raise AssertionError("sweep heatmap should be a post-processing command")

    monkeypatch.setattr(sweep_heatmap, "main", fail_if_called)

    results = [
        {
            "parameters": {"particle_energy_gev": 1.0, "aperture_radius": 0.1},
            "metrics": {},
        },
        {
            "parameters": {"particle_energy_gev": 2.0, "aperture_radius": 0.2},
            "metrics": {},
        },
    ]

    harness._generate_summary_plots(results, tmp_path)

    assert any(
        f"lw-generate-sweep-heatmap {tmp_path} --gain-filter all" in message
        for message in harness.logs
    )


def test_logcache_plotter_hides_internal_plot_helpers():
    assert not hasattr(logcache_plotter, "create_1d_curves_plot")
    assert not hasattr(logcache_plotter, "create_combined_gains_plot")
    assert not hasattr(logcache_plotter, "create_contour_plot")
    assert not hasattr(logcache_plotter, "live_monitor")
