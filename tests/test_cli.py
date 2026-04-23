"""Focused tests for the CLI request-building and config parsing path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from core.types import ChronoMatchingMode, SimulationType, StartupMode
from lw_integrator import cli


def _make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "config": None,
        "sweep_config": None,
        "results_file": None,
        "log_verbosity": None,
        "sc_verbosity": None,
        "adaptive_debug": None,
        "steps": None,
        "time_step": None,
        "simulation_type": None,
        "wall_position": None,
        "aperture_radius": None,
        "bunch_mean": None,
        "cavity_spacing": None,
        "z_cutoff": None,
        "chrono_mode": None,
        "startup_mode": None,
        "image_subcharge_count": None,
        "use_image_weighting": None,
        "driver_from_rider": False,
        "output": None,
        "quiet": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCliConfigParsing:
    def test_parse_args_accepts_results_file(self):
        args = cli.parse_args(["--results-file", "results/sweep_results.json"])

        assert args.results_file == Path("results/sweep_results.json")

    def test_parse_args_applies_boolean_flags(self):
        args = cli.parse_args(
            ["--adaptive-debug", "--image-weighting", "--simulation-type", "wall"]
        )

        assert args.adaptive_debug is True
        assert args.use_image_weighting is True
        assert args.simulation_type == "wall"

    def test_parse_args_allows_disabling_boolean_flags(self):
        args = cli.parse_args(["--no-adaptive-debug", "--no-image-weighting"])

        assert args.adaptive_debug is False
        assert args.use_image_weighting is False

    def test_parse_simulation_type_accepts_aliases(self):
        assert cli._parse_simulation_type("wall") == SimulationType.CONDUCTING_WALL
        assert cli._parse_simulation_type("bunch-to-bunch") == (
            SimulationType.BUNCH_TO_BUNCH
        )

    def test_parse_simulation_type_accepts_enum_and_integer_values(self):
        assert cli._parse_simulation_type(SimulationType.SWITCHING_WALL) == (
            SimulationType.SWITCHING_WALL
        )
        assert cli._parse_simulation_type(SimulationType.BUNCH_TO_BUNCH.value) == (
            SimulationType.BUNCH_TO_BUNCH
        )

    def test_parse_simulation_type_rejects_unknown_value(self):
        with pytest.raises(cli.SimulationConfigError, match="Unknown simulation type"):
            cli._parse_simulation_type("not-a-mode")

    def test_parse_chrono_mode_accepts_aliases(self):
        assert cli._parse_chrono_mode("legacy") == ChronoMatchingMode.FAST
        assert cli._parse_chrono_mode("blended") == ChronoMatchingMode.AVERAGED

    def test_parse_chrono_mode_accepts_enum_instance(self):
        assert cli._parse_chrono_mode(ChronoMatchingMode.AVERAGED) == (
            ChronoMatchingMode.AVERAGED
        )

    def test_parse_chrono_mode_rejects_invalid_values(self):
        with pytest.raises(cli.SimulationConfigError, match="Unknown chrono_mode"):
            cli._parse_chrono_mode("slow")
        with pytest.raises(
            cli.SimulationConfigError,
            match="chrono_mode must be a string or ChronoMatchingMode instance",
        ):
            cli._parse_chrono_mode(123)

    def test_parse_startup_mode_accepts_enum_or_alias(self):
        assert cli._parse_startup_mode("approximate") == (
            StartupMode.APPROXIMATE_BACK_HISTORY
        )
        assert cli._parse_startup_mode(StartupMode.COLD_START) == (
            StartupMode.COLD_START
        )

    def test_parse_startup_mode_rejects_invalid_values(self):
        with pytest.raises(cli.SimulationConfigError, match="Unknown startup_mode"):
            cli._parse_startup_mode("warm")
        with pytest.raises(
            cli.SimulationConfigError,
            match="startup_mode must be a string or StartupMode instance",
        ):
            cli._parse_startup_mode(123)

    def test_parse_image_subcharge_count_validates_range(self):
        assert cli._parse_image_subcharge_count("12") == 12
        with pytest.raises(cli.SimulationConfigError, match="between 4 and 128"):
            cli._parse_image_subcharge_count(3)

    def test_parse_image_subcharge_count_rejects_non_integer(self):
        with pytest.raises(cli.SimulationConfigError, match="must be an integer"):
            cli._parse_image_subcharge_count("many")

    def test_parse_image_weighting_accepts_truthy_and_falsey_strings(self):
        assert cli._parse_image_weighting("yes") is True
        assert cli._parse_image_weighting("off") is False
        with pytest.raises(cli.SimulationConfigError, match="truthy/falsey string"):
            cli._parse_image_weighting("maybe")

    def test_parse_image_weighting_defaults_when_none(self):
        assert (
            cli._parse_image_weighting(None)
            == cli.DEFAULT_SIMULATION["use_image_weighting"]
        )

    def test_build_integrator_config_uses_extracted_parsers(self):
        config = cli._build_integrator_config(
            {
                "steps": "12",
                "time_step": "0.25",
                "wall_position": "1.5",
                "aperture_radius": "0.002",
                "simulation_type": "switching-wall",
                "chrono_mode": "legacy",
                "startup_mode": "approximate",
                "image_subcharge_count": "16",
                "use_image_weighting": "no",
            }
        )

        assert config.steps == 12
        assert config.time_step == pytest.approx(0.25)
        assert config.simulation_type == SimulationType.SWITCHING_WALL
        assert config.chrono_mode == ChronoMatchingMode.FAST
        assert config.startup_mode == StartupMode.APPROXIMATE_BACK_HISTORY
        assert config.image_subcharge_count == 16
        assert config.use_image_weighting is False

    def test_build_integrator_config_requires_simulation_type(self):
        with pytest.raises(
            cli.SimulationConfigError, match="missing 'simulation_type'"
        ):
            cli._build_integrator_config(
                {"steps": 1, "time_step": 0.1, "wall_position": 0.0, "aperture_radius": 1.0}
            )

    def test_build_integrator_config_requires_core_fields(self):
        with pytest.raises(cli.SimulationConfigError, match="missing required fields"):
            cli._build_integrator_config({"simulation_type": "wall", "steps": 1})


class TestCliBuildRequest:
    def test_merge_particle_payload_applies_overrides(self):
        payload = cli._merge_particle_payload(
            {"position_z": 5.0},
            overrides={"position_z": 7.5, "particle_count": 4},
            defaults=cli.DEFAULT_RIDER,
        )

        assert payload["position_z"] == 7.5
        assert payload["particle_count"] == 4
        assert payload["kinetic_energy_mev"] == cli.DEFAULT_RIDER["kinetic_energy_mev"]

    def test_merge_simulation_payload_applies_cli_overrides(self):
        args = _make_args(steps=77, simulation_type="switching-wall", z_cutoff=4.5)

        payload = cli._merge_simulation_payload({"steps": 12, "z_cutoff": 1.0}, args)

        assert payload["steps"] == 77
        assert payload["simulation_type"] == "switching-wall"
        assert payload["z_cutoff"] == 4.5

    def test_build_request_clones_driver_from_rider(self):
        args = _make_args(
            simulation_type="bunch-to-bunch",
            driver_from_rider=True,
        )

        request = cli.build_request(args)

        assert request.driver is not None
        for key, rider_value in request.rider.items():
            assert np.array_equal(request.driver[key], rider_value)
            assert request.driver[key] is not rider_value

    def test_build_request_requires_driver_for_bunch_to_bunch(self):
        args = _make_args(simulation_type="bunch-to-bunch")

        with pytest.raises(cli.SimulationConfigError, match="require a driver bunch"):
            cli.build_request(args)

    def test_build_request_loads_driver_from_file(self, tmp_path: Path):
        config_path = tmp_path / "b2b.json"
        config_path.write_text(
            json.dumps(
                {
                    "simulation_type": "bunch-to-bunch",
                    "driver": {
                        "kinetic_energy_mev": 50.0,
                        "mass_amu": 1.0,
                        "charge_sign": 1.0,
                        "position_z": 10.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        args = _make_args(config=config_path)

        request = cli.build_request(args)

        assert request.config.simulation_type == SimulationType.BUNCH_TO_BUNCH
        assert request.driver is not None
        assert float(request.driver["z"][0]) == pytest.approx(10.0)

    def test_build_request_keeps_optional_driver_for_non_b2b(self, tmp_path: Path):
        config_path = tmp_path / "wall.json"
        config_path.write_text(
            json.dumps(
                {
                    "simulation_type": "wall",
                    "driver": {
                        "kinetic_energy_mev": 40.0,
                        "mass_amu": 1.0,
                        "charge_sign": -1.0,
                    },
                }
            ),
            encoding="utf-8",
        )

        request = cli.build_request(_make_args(config=config_path))

        assert request.config.simulation_type == SimulationType.CONDUCTING_WALL
        assert request.driver is not None

    def test_load_config_rejects_non_object_json(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        with pytest.raises(cli.SimulationConfigError, match="top level"):
            cli._load_config(path)

    def test_load_config_reports_missing_file(self, tmp_path: Path):
        with pytest.raises(cli.SimulationConfigError, match="Configuration file not found"):
            cli._load_config(tmp_path / "missing.json")

    def test_load_config_rejects_invalid_json(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{not-json", encoding="utf-8")

        with pytest.raises(cli.SimulationConfigError, match="not valid JSON"):
            cli._load_config(path)

    def test_build_particle_state_requires_core_fields(self):
        with pytest.raises(
            cli.SimulationConfigError, match="missing required fields"
        ):
            cli._build_particle_state({"kinetic_energy_mev": 10.0, "mass_amu": 1.0})

    def test_build_particle_state_rejects_unsupported_options(self):
        with pytest.raises(
            cli.SimulationConfigError, match="includes unsupported options"
        ):
            cli._build_particle_state(
                {
                    "kinetic_energy_mev": 10.0,
                    "mass_amu": 1.0,
                    "charge_sign": -1.0,
                    "unsupported_flag": True,
                }
            )


class TestCliSweepEntryPoint:
    def test_build_sweep_verbosity_overrides_ignores_none_values(self):
        overrides = cli._build_sweep_verbosity_overrides(
            _make_args(log_verbosity=None, sc_verbosity=2, adaptive_debug=None)
        )

        assert overrides == {"self_consistency_verbosity": 2}

    def test_run_sweep_forwards_verbosity_overrides(self, monkeypatch, tmp_path: Path):
        config_path = tmp_path / "sweep.json"
        config_path.write_text("{}", encoding="utf-8")
        captured = {}

        def fake_run_sweep_from_config(*, config_path, output_dir, verbose, verbosity_overrides):
            captured["config_path"] = config_path
            captured["output_dir"] = output_dir
            captured["verbose"] = verbose
            captured["verbosity_overrides"] = verbosity_overrides
            return True

        monkeypatch.setattr(
            "lw_integrator.sweep_runner.run_sweep_from_config",
            fake_run_sweep_from_config,
        )

        result = cli.run_sweep(
            _make_args(
                sweep_config=config_path,
                quiet=True,
                log_verbosity="full",
                sc_verbosity=3,
                adaptive_debug=True,
            )
        )

        assert result == 0
        assert captured["config_path"] == config_path
        assert captured["output_dir"] is None
        assert captured["verbose"] is False
        assert captured["verbosity_overrides"] == {
            "log_verbosity": "full",
            "self_consistency_verbosity": 3,
            "adaptive_timestep_debug": True,
        }

    def test_run_sweep_returns_2_for_missing_config(self, tmp_path: Path, capsys):
        result = cli.run_sweep(_make_args(sweep_config=tmp_path / "missing.json"))

        assert result == 2
        assert "Sweep config file not found" in capsys.readouterr().err

    def test_run_sweep_returns_2_on_exception(self, monkeypatch, tmp_path: Path, capsys):
        config_path = tmp_path / "sweep.json"
        config_path.write_text("{}", encoding="utf-8")

        def fake_run_sweep_from_config(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "lw_integrator.sweep_runner.run_sweep_from_config",
            fake_run_sweep_from_config,
        )

        result = cli.run_sweep(_make_args(sweep_config=config_path))

        assert result == 2
        assert "Error running sweep: boom" in capsys.readouterr().err


class TestCliMain:
    def test_main_prints_saved_sweep_results_summary(self, tmp_path: Path, capsys):
        results_path = tmp_path / "sweep_results.json"
        results_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "run_number": 2,
                            "parameters": {
                                "aperture_radius": 0.1,
                                "particle_energy_gev": 5.0,
                            },
                            "metrics": {"rider_delta_e_mev": 1.25},
                            "trajectory": {"z": [0.0, 1.0], "r": [0.0, 0.0]},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        result = cli.main(["--results-file", str(results_path)])

        output = capsys.readouterr().out
        assert result == 0
        assert "LW Integrator saved-results summary:" in output
        assert "Result Type: sweep" in output
        assert "Run Count: 1" in output
        assert "Best Delta E Mev: 1.25" in output

    def test_main_writes_saved_optimization_results_summary_json(
        self, tmp_path: Path
    ):
        results_path = tmp_path / "optimization_results.json"
        output_path = tmp_path / "summary.json"
        results_path.write_text(
            json.dumps(
                {
                    "optimization_method": "genetic_algorithm",
                    "objective": "max_energy_gain",
                    "best_value": 2.5,
                    "success": True,
                    "all_evaluations": [
                        {"objective_value": 1.0},
                        {"objective_value": float("inf")},
                    ],
                    "total_evaluations": 2,
                    "top_n_results": [
                        {
                            "metrics": {"rider_delta_e_mev": 3.0},
                        }
                    ],
                    "top_n_count": 1,
                    "best_parameters": {"initial_energy_gev": 5.0},
                }
            ),
            encoding="utf-8",
        )

        result = cli.main(
            ["--results-file", str(results_path), "--output", str(output_path), "--quiet"]
        )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert result == 0
        assert payload["result_type"] == "optimization"
        assert payload["optimization_method"] == "genetic_algorithm"
        assert payload["evaluation_count"] == 2
        assert payload["finite_evaluation_count"] == 1
        assert payload["best_delta_e_mev"] == pytest.approx(3.0)
        assert payload["best_parameters"] == {"initial_energy_gev": 5.0}
        assert payload["top_results"] == [
            {
                "rank": 1,
                "metric_value": None,
                "fitness": None,
                "parameters": {},
                "delta_e_mev": 3.0,
                "percent_energy_gain": None,
                "metrics": {"rider_delta_e_mev": 3.0},
            }
        ]

    def test_main_prints_saved_optimization_top_results(self, tmp_path: Path, capsys):
        results_path = tmp_path / "optimization_results.json"
        results_path.write_text(
            json.dumps(
                {
                    "objective": "max_energy_gain",
                    "all_evaluations": [
                        {
                            "evaluation": 8,
                            "failed": False,
                            "halted_early": False,
                            "raw_objective_value": 2.5,
                            "fitness": -2.5,
                            "metrics": {
                                "rider_delta_e_mev": 4.0,
                                "max_percent_energy_gain": 1.5,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        result = cli.main(["--results-file", str(results_path)])

        output = capsys.readouterr().out
        assert result == 0
        assert "Top Results:" in output
        assert "rank=1, metric=2.5, delta_e_mev=4, percent_gain=1.5" in output

    def test_main_returns_2_for_unknown_results_file_format(
        self, tmp_path: Path, capsys
    ):
        results_path = tmp_path / "unknown.json"
        results_path.write_text(json.dumps({"unexpected": []}), encoding="utf-8")

        result = cli.main(["--results-file", str(results_path)])

        assert result == 2
        assert "Cannot parse this file format." in capsys.readouterr().err

    def test_main_dispatches_to_sweep_runner(self, monkeypatch):
        captured = {}

        def fake_run_sweep(args):
            captured["args"] = args
            return 1

        monkeypatch.setattr(cli, "run_sweep", fake_run_sweep)

        result = cli.main(["--sweep-config", "configs/example.json"])

        assert result == 1
        assert captured["args"].sweep_config == Path("configs/example.json")

    def test_main_writes_output_json(self, monkeypatch, tmp_path: Path):
        output_path = tmp_path / "summary.json"
        fake_request = object()
        monkeypatch.setattr(cli, "build_request", lambda args: fake_request)

        def fake_run_simulation(request):
            trajectory = [
                {
                    "gamma": np.array([1.0]),
                    "z": np.array([0.0]),
                    "t": np.array([0.0]),
                    "bz": np.array([0.0]),
                }
            ]
            return trajectory, None

        monkeypatch.setattr(cli, "run_simulation", fake_run_simulation)
        monkeypatch.setattr(
            cli,
            "summarise_trajectory",
            lambda trajectory: {"steps_completed": 1, "delta_gamma_mean": 0.0},
        )

        result = cli.main(["--output", str(output_path), "--quiet"])

        assert result == 0
        assert json.loads(output_path.read_text(encoding="utf-8")) == {
            "steps_completed": 1,
            "delta_gamma_mean": 0.0,
        }

    def test_main_prints_driver_summary_when_present(self, monkeypatch, capsys):
        fake_request = object()
        monkeypatch.setattr(cli, "build_request", lambda args: fake_request)
        monkeypatch.setattr(
            cli,
            "run_simulation",
            lambda request: ([{"gamma": np.array([1.0]), "z": np.array([0.0]), "t": np.array([0.0]), "bz": np.array([0.0])}], [1, 2, 3]),
        )
        monkeypatch.setattr(
            cli,
            "summarise_trajectory",
            lambda trajectory: {"steps_completed": 1, "delta_gamma_mean": 0.0},
        )

        result = cli.main([])

        output = capsys.readouterr().out
        assert result == 0
        assert "LW Integrator simulation summary:" in output
        assert "Driver trajectory generated with 3 integration steps." in output

    def test_build_report_includes_driver_summary_when_present(self):
        trajectory = [
            {
                "t": np.array([0.0]),
                "z": np.array([1.0]),
                "gamma": np.array([2.0]),
                "bz": np.array([0.25]),
            }
        ]
        driver = [
            {
                "t": np.array([0.0]),
                "z": np.array([5.0]),
                "gamma": np.array([3.0]),
                "bz": np.array([0.5]),
            }
        ]

        report = cli.build_report(trajectory, driver)

        assert report["steps_completed"] == 1
        assert report["driver_summary"]["steps_completed"] == 1
        assert report["driver_summary"]["initial_z_mm"] == pytest.approx(5.0)

    def test_main_writes_driver_summary_to_output_json(
        self, monkeypatch, tmp_path: Path
    ):
        output_path = tmp_path / "summary.json"
        fake_request = object()
        monkeypatch.setattr(cli, "build_request", lambda args: fake_request)

        monkeypatch.setattr(
            cli,
            "run_simulation",
            lambda request: (
                [
                    {
                        "gamma": np.array([1.0]),
                        "z": np.array([0.0]),
                        "t": np.array([0.0]),
                        "bz": np.array([0.0]),
                    }
                ],
                [
                    {
                        "gamma": np.array([2.0]),
                        "z": np.array([3.0]),
                        "t": np.array([0.5]),
                        "bz": np.array([0.2]),
                    }
                ],
            ),
        )

        result = cli.main(["--output", str(output_path), "--quiet"])

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert result == 0
        assert payload["steps_completed"] == 1
        assert payload["driver_summary"]["steps_completed"] == 1
        assert payload["driver_summary"]["initial_z_mm"] == pytest.approx(3.0)

    def test_main_returns_2_for_invalid_config(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli,
            "build_request",
            lambda args: (_ for _ in ()).throw(cli.SimulationConfigError("bad config")),
        )

        result = cli.main([])

        assert result == 2
        assert "Error: bad config" in capsys.readouterr().err


class TestCliRuntimeHelpers:
    def test_run_simulation_forwards_request_to_retarded_integrator(self, monkeypatch):
        request = cli.build_request(_make_args())
        captured = {}

        def fake_retarded_integrator(**kwargs):
            captured.update(kwargs)
            return ["rider"], None

        monkeypatch.setattr(cli, "retarded_integrator", fake_retarded_integrator)

        rider, driver = cli.run_simulation(request)

        assert rider == ["rider"]
        assert driver is None
        assert captured["steps"] == request.config.steps
        assert captured["h_step"] == request.config.time_step
        assert captured["wall_z"] == request.config.wall_position
        assert captured["aperture_radius"] == request.config.aperture_radius
        assert captured["sim_type"] == request.config.simulation_type
        assert captured["init_rider"] is request.rider
        assert captured["init_driver"] is request.driver
        assert captured["image_subcharge_count"] == request.config.image_subcharge_count
        assert (
            captured["use_conducting_image_weighting"]
            == request.config.use_image_weighting
        )

    def test_summarise_trajectory_uses_means_and_max_abs(self):
        trajectory = [
            {
                "t": np.array([0.0, 1.0]),
                "z": np.array([-2.0, 2.0]),
                "gamma": np.array([2.0, 4.0]),
                "bz": np.array([-0.5, 0.25]),
            },
            {
                "t": np.array([2.0, 4.0]),
                "z": np.array([10.0, 14.0]),
                "gamma": np.array([5.0, 7.0]),
                "bz": np.array([-0.75, 0.6]),
            },
        ]

        summary = cli.summarise_trajectory(trajectory)

        assert summary == {
            "steps_completed": 2,
            "initial_time_ns": pytest.approx(0.5),
            "final_time_ns": pytest.approx(3.0),
            "initial_z_mm": pytest.approx(0.0),
            "final_z_mm": pytest.approx(12.0),
            "traveled_distance_mm": pytest.approx(12.0),
            "initial_gamma_mean": pytest.approx(3.0),
            "final_gamma_mean": pytest.approx(6.0),
            "delta_gamma_mean": pytest.approx(3.0),
            "max_absolute_velocity": pytest.approx(0.75),
        }

    def test_print_summary_formats_human_readable_output(self, capsys):
        cli.print_summary(
            {
                "steps_completed": 5,
                "traveled_distance_mm": 12.3456789,
                "delta_gamma_mean": 1.23456789,
            }
        )

        output = capsys.readouterr().out
        assert "LW Integrator simulation summary:" in output
        assert "Steps Completed: 5" in output
        assert "Traveled Distance Mm: 12.3457" in output
        assert "Delta Gamma Mean: 1.23457" in output
