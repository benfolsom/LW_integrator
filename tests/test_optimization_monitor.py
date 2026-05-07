"""Tests for packaged optimization monitor helpers and CLI entry points."""

from __future__ import annotations

from pathlib import Path

from lw_integrator import optimization_monitor
from optimization.log_monitor import (
    MONITORED_INSIGHT_PARAMETERS,
    analyze_optimization_logs,
    collect_varied_parameters,
    parse_optimization_log,
    select_optimization_log_files,
    summarize_parameter_ranges,
)


def _write_log(path: Path, content: str, *, mtime: int) -> None:
    path.write_text(content, encoding="utf-8")
    path.touch()
    path.chmod(0o644)
    Path(path).touch()
    import os

    os.utime(path, (mtime, mtime))


class TestOptimizationLogMonitorHelpers:
    def test_module_exposes_only_supported_public_helpers(self):
        import optimization.log_monitor as log_monitor

        assert not hasattr(log_monitor, "parse_log_parameters")
        assert log_monitor.__all__ == [
            "MONITORED_INSIGHT_PARAMETERS",
            "analyze_optimization_logs",
            "collect_varied_parameters",
            "parse_optimization_log",
            "select_optimization_log_files",
            "summarize_parameter_ranges",
        ]

    def test_parse_optimization_log_extracts_numeric_parameter_values(
        self, tmp_path: Path
    ):
        log_path = tmp_path / "run_optimization.log"
        _write_log(
            log_path,
            "Evaluation 1: initial_energy_gev=35 transverse_momentum=1.5e-3 aperture_radius=0.2\n"
            "max_percent_energy_gain: 1.25%",
            mtime=10,
        )

        results = parse_optimization_log(log_path)

        assert results[0]["params"] == {
            "initial_energy_gev": 35.0,
            "transverse_momentum": 1.5e-3,
            "aperture_radius": 0.2,
        }
        assert MONITORED_INSIGHT_PARAMETERS[0] == "initial_energy_gev"

    def test_parse_optimization_log_collects_evaluations(self, tmp_path: Path):
        log_path = tmp_path / "run_optimization.log"
        _write_log(
            log_path,
            "\n".join(
                [
                    "Evaluation 1: initial_energy_gev=35 transverse_momentum=1.0e-3",
                    "max_percent_energy_gain: 1.25%",
                    "Evaluation 2: initial_energy_gev=36 transverse_momentum=2.0e-3",
                    "max_percent_energy_gain: -0.5%",
                ]
            ),
            mtime=10,
        )

        results = parse_optimization_log(log_path)

        assert [result["eval_num"] for result in results] == [1, 2]
        assert results[0]["params"]["initial_energy_gev"] == 35.0
        assert results[0]["energy_gain"] == 1.25
        assert results[1]["energy_gain"] == -0.5

    def test_select_optimization_log_files_supports_latest_and_specific(
        self, tmp_path: Path
    ):
        old_log = tmp_path / "20260101_optimization.log"
        new_log = tmp_path / "20260102_optimization.log"
        _write_log(old_log, "", mtime=10)
        _write_log(new_log, "", mtime=20)

        assert select_optimization_log_files(tmp_path, latest_only=True) == [new_log]
        assert select_optimization_log_files(tmp_path, specific_run="20260101") == [
            old_log
        ]

    def test_analyze_optimization_logs_sorts_best_gain_first(self, tmp_path: Path):
        log_path = tmp_path / "run_optimization.log"
        _write_log(
            log_path,
            "\n".join(
                [
                    "Evaluation 1: initial_energy_gev=35",
                    "max_percent_energy_gain: 0.5%",
                    "Evaluation 2: initial_energy_gev=36",
                    "max_percent_energy_gain: 2.0%",
                ]
            ),
            mtime=10,
        )

        results, log_files = analyze_optimization_logs(tmp_path)

        assert log_files == [log_path]
        assert [result["eval_num"] for result in results] == [2, 1]

    def test_collect_varied_parameters_and_ranges(self):
        results = [
            {
                "params": {
                    "initial_energy_gev": 40.0,
                    "transverse_momentum": 0.001,
                    "fixed_value": 1.0,
                }
            },
            {
                "params": {
                    "initial_energy_gev": 42.0,
                    "transverse_momentum": 0.002,
                    "fixed_value": 1.0,
                }
            },
            {
                "params": {
                    "initial_energy_gev": 44.0,
                    "transverse_momentum": 0.004,
                    "fixed_value": 1.0,
                }
            },
        ]

        assert collect_varied_parameters(results) == [
            "initial_energy_gev",
            "transverse_momentum",
        ]

        summaries = summarize_parameter_ranges(
            results,
            parameter_names=("initial_energy_gev", "transverse_momentum"),
            top_fraction=0.5,
        )

        assert summaries["initial_energy_gev"] == {
            "average": 40.0,
            "min": 40.0,
            "max": 40.0,
        }
        assert summaries["transverse_momentum"] == {
            "average": 0.001,
            "min": 0.001,
            "max": 0.001,
        }


class TestOptimizationMonitorCli:
    def test_module_hides_parser_helper(self):
        assert not hasattr(optimization_monitor, "parse_args")
        assert optimization_monitor.__all__ == ["OptimizationMonitor", "main"]

    def test_main_runs_packaged_monitor_once(self, tmp_path: Path, capsys):
        log_path = tmp_path / "current_optimization.log"
        _write_log(
            log_path,
            "\n".join(
                [
                    "Evaluation 1: initial_energy_gev=35 transverse_momentum=1.0e-3",
                    "max_percent_energy_gain: 1.25%",
                ]
            ),
            mtime=10,
        )

        exit_code = optimization_monitor.main(
            ["--once", "--latest", "--logcache", str(tmp_path), "--top", "1"]
        )

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "LIVE OPTIMIZATION MONITOR - LATEST RUN ONLY" in captured.out
        assert "TOP 1 PARAMETER COMBINATIONS" in captured.out

    def test_main_reports_missing_logcache(self, tmp_path: Path, capsys):
        missing_dir = tmp_path / "missing"

        exit_code = optimization_monitor.main(
            ["--once", "--logcache", str(missing_dir)]
        )

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "logcache directory not found" in captured.out
