"""Tests for the packaged optimization results CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

from lw_integrator import optimization_results


class TestOptimizationResultsCli:
    def test_module_hides_internal_formatting_helper(self):
        assert not hasattr(optimization_results, "format_value")

    def test_main_prints_top_results(self, tmp_path: Path, capsys):
        results_path = tmp_path / "optimization_results.json"
        results_path.write_text(
            json.dumps(
                {
                    "objective": "maximize_energy_gain",
                    "all_evaluations": [
                        {
                            "evaluation": 1,
                            "objective_value": 1.5,
                            "fitness": 1.5,
                            "parameters": {"initial_energy_gev": 35.0},
                            "metrics": {
                                "rider_delta_e_mev": 0.75,
                                "max_percent_energy_gain": 1.2,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        exit_code = optimization_results.main([str(results_path), "--top", "1"])

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "TOP 1 OPTIMIZATION RUNS" in captured.out
        assert "Successful Evaluations: 1" in captured.out

    def test_main_rejects_missing_top_n_source(self, tmp_path: Path, capsys):
        results_path = tmp_path / "optimization_results.json"
        results_path.write_text(
            json.dumps({"objective": "maximize_energy_gain", "all_evaluations": []}),
            encoding="utf-8",
        )

        exit_code = optimization_results.main([str(results_path), "--source", "top_n"])

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "No 'top_n_results' found" in captured.out

    def test_main_rejects_non_optimization_payload(self, tmp_path: Path, capsys):
        results_path = tmp_path / "sweep_results.json"
        results_path.write_text(json.dumps({"results": []}), encoding="utf-8")

        exit_code = optimization_results.main([str(results_path)])

        captured = capsys.readouterr()
        assert exit_code == 1
        assert "only supports optimization_results.json files" in captured.out
