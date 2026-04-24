"""Tests for the packaged optimization results CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import plot_optimization_results
from lw_integrator import optimization_results


class TestOptimizationResultsCli:
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

    def test_legacy_wrapper_delegates_modern_payload(self, tmp_path: Path, monkeypatch):
        results_path = tmp_path / "optimization_results.json"
        results_path.write_text(
            json.dumps({"objective": "maximize_energy_gain", "all_evaluations": []}),
            encoding="utf-8",
        )
        captured = {}

        def fake_main(argv):
            captured["argv"] = argv
            return 0

        monkeypatch.setattr(optimization_results, "main", fake_main)

        exit_code = plot_optimization_results.main([str(results_path)])

        assert exit_code == 0
        assert captured["argv"] == [str(results_path)]

    def test_legacy_wrapper_supports_list_payload(self, tmp_path: Path):
        results_path = tmp_path / "legacy_results.json"
        results_path.write_text(
            json.dumps(
                [
                    {
                        "aperture_mm": 0.1,
                        "energy_gev": 5.0,
                        "start_z_mm": 1.0,
                        "max_energy_gain_gev": 0.2,
                        "max_relative_gain": 0.02,
                        "num_deflection_events": 0,
                    },
                    {
                        "aperture_mm": 0.2,
                        "energy_gev": 10.0,
                        "start_z_mm": 2.0,
                        "max_energy_gain_gev": 0.3,
                        "max_relative_gain": 0.03,
                        "num_deflection_events": 1,
                    },
                ]
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "heatmap.png"

        exit_code = plot_optimization_results.main(
            [str(results_path), "--output", str(output_path)]
        )

        assert exit_code == 0
        assert output_path.exists()
