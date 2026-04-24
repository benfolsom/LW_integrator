"""Focused tests for the packaged saved-trajectory plotter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lw_integrator import trajectory_plotter


def test_infer_rest_energy_mev_uses_pt_over_gamma():
    rest_energy = trajectory_plotter.infer_rest_energy_mev(
        {
            "gamma_hist": [2.0, 3.0, 4.0],
            "pt_hist": [1.0, 1.5, 2.0],  # implies mass=0.5 amu
        }
    )

    assert rest_energy == pytest.approx(0.5 * trajectory_plotter.AMU_TO_MEV)


def test_plot_saved_json_trajectory_writes_png(tmp_path: Path):
    input_path = tmp_path / "trajectory.json"
    input_path.write_text(
        json.dumps(
            {
                "config_label": "demo",
                "simulation_type": "CONDUCTING_WALL",
                "core": {
                    "rider": {
                        "gamma_hist": [2.0, 2.2],
                        "pt_hist": [1.0, 1.1],
                        "positions_mm": {"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 2.0]},
                        "conjugate_momenta": {"Px": [0.0, 0.1], "Py": [0.0, 0.0], "Pz": [1.0, 1.1]},
                        "time_ns": [0.0, 0.1],
                    },
                    "driver": {
                        "gamma_hist": [3.0, 3.1],
                        "pt_hist": [1.5, 1.55],
                        "positions_mm": {"x": [0.0, 0.5], "y": [0.0, 0.0], "z": [0.0, 2.0]},
                        "conjugate_momenta": {"Px": [0.0, 0.05], "Py": [0.0, 0.0], "Pz": [1.5, 1.6]},
                        "time_ns": [0.0, 0.1],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    output_path = trajectory_plotter.plot_saved_trajectory(input_path)

    assert output_path.exists()
    assert output_path.suffix == ".png"


def test_plot_saved_npz_trajectory_writes_png(tmp_path: Path):
    input_path = tmp_path / "trajectory.npz"
    np.savez(
        input_path,
        z=np.array([0.0, 1.0, 2.0]),
        r=np.array([0.0, 0.1, 0.2]),
        pz=np.array([1.0, 1.1, 1.2]),
        pr=np.array([0.0, 0.01, 0.02]),
        gamma=np.array([2.0, 2.1, 2.3]),
    )

    output_path = trajectory_plotter.plot_saved_trajectory(input_path, mass_amu=1.0)

    assert output_path.exists()
    assert output_path.suffix == ".png"


def test_main_reports_missing_file(tmp_path: Path, capsys):
    missing = tmp_path / "missing.json"

    exit_code = trajectory_plotter.main([str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Trajectory file not found" in captured.out
