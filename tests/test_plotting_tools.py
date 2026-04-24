"""Focused tests for plotting entry points and sweep plot integration."""

from __future__ import annotations

from pathlib import Path

import generate_sweep_heatmap as sweep_heatmap
from lw_integrator import plot_latest_live
from optimization.results_mixins import OptimizationResultsMixin


class _ResultsHarness(OptimizationResultsMixin):
    def __init__(self) -> None:
        self.logs = []

    def _log_result(self, message: str) -> None:
        self.logs.append(message)

    def _plot_single_trajectory(self, *_args, **_kwargs) -> None:
        raise AssertionError("trajectory plot should not be called in this test")


def test_generate_sweep_heatmap_main_accepts_argv(monkeypatch):
    captured = {}

    def fake_generate_heatmap(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(sweep_heatmap, "generate_heatmap", fake_generate_heatmap)

    exit_code = sweep_heatmap.main(["results/sweeps/example", "--gain-filter", "all"])

    assert exit_code == 0
    assert captured["sweep_dir"] == "results/sweeps/example"
    assert captured["gain_filter"] == "all"


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


def test_generate_summary_plots_calls_heatmap_tool(tmp_path: Path, monkeypatch):
    harness = _ResultsHarness()
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(sweep_heatmap, "main", fake_main)

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

    assert captured["argv"] == [
        str(tmp_path),
        "--gain-filter",
        "all",
        "--output",
        "sweep_heatmap.png",
    ]
    assert any("Heatmap saved to" in message for message in harness.logs)
