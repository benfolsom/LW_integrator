"""Focused tests for the CLI request-building and config parsing path."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import core
import lw_integrator
from core.external_fields import electric_field_v_per_m_to_native
from core.types import ChronoMatchingMode, SimulationType, StartupMode
from lw_integrator import cli
from lw_integrator.testbed_runner import SimulationOptions


def test_cli_direct_checkpoint_flags_build_core_config(tmp_path: Path) -> None:
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "steps": 4,
                "time_step": 1.0e-3,
                "wall_position": 0.0,
                "aperture_radius": 1.0,
                "simulation_type": "bunch-to-bunch",
                "rider": {},
                "driver": {},
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "capture.checkpoint"

    request = cli.build_request(
        cli.parse_args(
            [
                "--config",
                str(config_path),
                "--checkpoint-dir",
                str(checkpoint_path),
                "--checkpoint-every-steps",
                "25",
                "--checkpoint-every-seconds",
                "60",
            ]
        )
    )

    assert request.config.checkpoint.enabled is True
    assert request.config.checkpoint.directory == str(checkpoint_path)
    assert request.config.checkpoint.resume_from is None
    assert request.config.checkpoint.interval_steps == 25
    assert request.config.checkpoint.interval_seconds == pytest.approx(60.0)


def test_cli_testbed_resume_flag_overrides_loaded_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "testbed.json"
    config_path.write_text(json.dumps({"steps": 4}), encoding="utf-8")
    checkpoint_path = tmp_path / "capture.checkpoint"
    captured: dict[str, SimulationOptions] = {}

    def fake_run(options: SimulationOptions):
        captured["options"] = options
        return SimpleNamespace(
            duration_s=0.0,
            filename_base="test",
            halted_early=False,
            halt_reason=None,
            num_particles_dead=0,
            rider_delta_e=0.0,
            rider_gamma_initial=1.0,
            rider_gamma_final=1.0,
            driver_gamma_initial=None,
            driver_gamma_final=None,
            energy_ledger_metrics={},
            saved_paths={},
        )

    monkeypatch.setattr("lw_integrator.testbed_runner.run_testbed", fake_run)

    result = cli.main(
        [
            "--testbed-config",
            str(config_path),
            "--resume-from",
            str(checkpoint_path),
            "--quiet",
        ]
    )

    assert result == 0
    assert captured["options"].checkpoint_enabled is True
    assert captured["options"].checkpoint_resume_from == checkpoint_path


def _make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "config": None,
        "sweep_config": None,
        "results_file": None,
        "log_verbosity": None,
        "sc_verbosity": None,
        "adaptive_debug": None,
        "adaptive_timestep_enabled": None,
        "adaptive_timestep_threshold": None,
        "adaptive_timestep_reduction_factor": None,
        "adaptive_timestep_min_factor": None,
        "adaptive_timestep_bunch_proximity_enabled": None,
        "adaptive_timestep_bunch_proximity_sigma_mm": None,
        "adaptive_timestep_bunch_proximity_n_sigma": None,
        "adaptive_timestep_bunch_proximity_reduction_factor": None,
        "adaptive_timestep_bunch_proximity_transition_n_sigma": None,
        "space_charge": False,
        "space_charge_softening_mm": 0.0,
        "space_charge_bunch_sigma_mm": None,
        "space_charge_min_retarded_steps": None,
        "auto_duration": False,
        "auto_duration_crossing_steps": None,
        "auto_duration_post_factor": None,
        "steps": None,
        "time_step": None,
        "simulation_type": None,
        "wall_position": None,
        "aperture_radius": None,
        "bunch_mean": None,
        "cavity_spacing": None,
        "z_cutoff": None,
        "z_cutoff_mode": None,
        "cavity_exit_enabled": None,
        "cavity_exit_mode": None,
        "cavity_exit_length_mm": None,
        "beamline_geometry_enabled": None,
        "beamline_geometry_file": None,
        "chrono_mode": None,
        "chrono_interpolate": None,
        "chrono_tolerance": None,
        "chrono_high_precision": None,
        "chrono_adaptive_tolerance": None,
        "startup_mode": None,
        "radiation_reaction_mode": None,
        "magnetic_dipole_enabled": None,
        "dipole_source_model": None,
        "exact_retarded_backend": None,
        "dipole_source_minimum_separation_mm": None,
        "rider_magnetic_species": None,
        "driver_magnetic_species": None,
        "rider_spin": None,
        "driver_spin": None,
        "stern_gerlach_force_enabled": None,
        "spin_precession_enabled": None,
        "image_subcharge_count": None,
        "use_image_weighting": None,
        "pseudo_grid_enabled": None,
        "pseudo_grid_active_rider_count": None,
        "pseudo_grid_active_driver_count": None,
        "pseudo_grid_field_rider_count": None,
        "pseudo_grid_field_driver_count": None,
        "pseudo_grid_field_deposition_neighbor_count": None,
        "pseudo_grid_passive_neighbor_count": None,
        "pseudo_grid_coverage_strategy": None,
        "pseudo_grid_coverage_space": None,
        "pseudo_grid_pair_reuse_window": None,
        "pseudo_grid_source_weighting_mode": None,
        "pseudo_grid_loss_tracking_enabled": None,
        "pseudo_grid_causal_history_pruning_enabled": None,
        "pseudo_grid_causal_history_safety_margin_steps": None,
        "macroparticle_smearing_enabled": None,
        "macroparticle_smearing_subcharge_count": None,
        "macroparticle_smearing_sigma_multiplier": None,
        "macroparticle_smearing_position_sigma_mm": None,
        "macroparticle_smearing_longitudinal_sigma_mm": None,
        "macroparticle_smearing_momentum_sigma_amu_mm_ns": None,
        "macroparticle_smearing_seed": None,
        "macroparticle_smearing_refresh_policy": None,
        "macroparticle_smearing_apply_to_passive_updates": None,
        "driver_train_enabled": None,
        "driver_train_bunch_count": None,
        "driver_train_z_spacing_mm": None,
        "driver_train_z_offsets_mm": None,
        "driver_train_prehistory_steps": None,
        "driver_train_preserve_prehistory_in_output": None,
        "driver_from_rider": False,
        "output": None,
        "quiet": False,
        "external_field": False,
        "external_e_field_native": None,
        "external_e_field_v_per_m": None,
        "external_b_field_native": None,
        "external_b_field_tesla": None,
    }
    for axis in ("x", "y", "z", "t"):
        defaults[f"external_field_{axis}_min"] = None
        defaults[f"external_field_{axis}_max"] = None
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCliConfigParsing:
    def test_python_module_entry_point_delegates_to_cli_main(self, monkeypatch):
        monkeypatch.setattr(cli, "main", lambda: 17)

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("lw_integrator", run_name="__main__")

        assert exc_info.value.code == 17

    def test_help_text_describes_maintained_results_and_chrono_modes(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            cli.parse_args(["--help"])

        help_text = capsys.readouterr().out
        assert "--results-file RESULTS_FILE" in help_text
        assert "saved sweep or optimization results JSON" in help_text
        assert "'fast' is the" in help_text
        assert "maintained default" in help_text

    def test_parse_args_accepts_results_file(self):
        args = cli.parse_args(["--results-file", "results/sweep_results.json"])

        assert args.results_file == Path("results/sweep_results.json")

    def test_parse_args_accepts_testbed_config(self):
        args = cli.parse_args(["--testbed-config", "configs/study_run.json"])

        assert args.testbed_config == Path("configs/study_run.json")

    def test_parse_args_accepts_chrono_sampling_flags(self):
        args = cli.parse_args(
            [
                "--chrono-interpolate",
                "--chrono-tolerance",
                "5e-4",
                "--chrono-high-precision",
                "--chrono-adaptive-tolerance",
            ]
        )

        assert args.chrono_interpolate is True
        assert args.chrono_tolerance == pytest.approx(5e-4)
        assert args.chrono_high_precision is True
        assert args.chrono_adaptive_tolerance is True

    def test_parse_args_accepts_inertial_prehistory(self):
        args = cli.parse_args(["--startup-mode", "inertial-prehistory"])

        assert args.startup_mode == "inertial-prehistory"

    def test_package_exports_only_maintained_entry_points(self):
        assert lw_integrator.__all__ == ["__version__", "VERSION"]
        for name in [
            "cli_main",
            "IntegratorConfig",
            "SimulationType",
            "ParticleState",
            "Trajectory",
            "retarded_integrator",
            "run_integrator",
            "trajectory_integrator",
            "C_MMNS",
        ]:
            with pytest.raises(AttributeError):
                getattr(lw_integrator, name)

    def test_core_package_exports_only_version_metadata(self):
        assert core.__all__ == ["__version__", "VERSION"]
        assert "trajectory_integrator" not in core.__all__
        for name in [
            "IntegratorConfig",
            "SimulationType",
            "ParticleState",
            "Trajectory",
            "retarded_integrator",
            "run_integrator",
            "C_MMNS",
        ]:
            with pytest.raises(AttributeError):
                getattr(core, name)

    def test_build_request_carries_chrono_options_without_sc_iterations(self):
        request = cli.build_request(
            _make_args(
                chrono_interpolate=True,
                chrono_tolerance=5e-4,
                chrono_high_precision=True,
            )
        )

        assert request.self_consistency is not None
        assert request.self_consistency.enabled is False
        assert request.self_consistency.chrono_interpolate is True
        assert request.self_consistency.chrono_tolerance == pytest.approx(5e-4)
        assert request.self_consistency.chrono_high_precision is True

    def test_parse_args_applies_boolean_flags(self):
        args = cli.parse_args(
            ["--adaptive-debug", "--image-weighting", "--simulation-type", "wall"]
        )

        assert args.adaptive_debug is True
        assert args.use_image_weighting is True
        assert args.simulation_type == "wall"

    def test_parse_args_accepts_external_field_options(self):
        args = cli.parse_args(
            [
                "--external-e-field-v-per-m",
                "0",
                "0",
                "-1.5e9",
                "--external-b-field-native",
                "0",
                "3",
                "0",
                "--external-b-field-tesla",
                "0.1",
                "0.2",
                "0.3",
                "--external-field-z-min",
                "-0.2",
                "--external-field-t-max",
                "1e-6",
            ]
        )

        assert args.external_e_field_v_per_m == [0.0, 0.0, -1.5e9]
        assert args.external_b_field_native == [0.0, 3.0, 0.0]
        assert args.external_b_field_tesla == [0.1, 0.2, 0.3]
        assert args.external_field_z_min == pytest.approx(-0.2)
        assert args.external_field_t_max == pytest.approx(1.0e-6)

    def test_parse_args_accepts_radiation_reaction_mode(self):
        args = cli.parse_args(["--radiation-reaction-mode", "off"])

        assert args.radiation_reaction_mode == "off"

    def test_parse_args_accepts_magnetic_dipole_options_and_species_aliases(self):
        args = cli.parse_args(
            [
                "--magnetic-dipoles",
                "--rider-magnetic-species",
                "n",
                "--driver-magnetic-species",
                "h-",
                "--rider-spin",
                "0",
                "3",
                "4",
                "--driver-spin",
                "1",
                "0",
                "0",
                "--stern-gerlach",
                "--dipole-source",
                "full-retarded-point",
                "--exact-retarded-backend",
                "numba_roots_exact_serial",
                "--dipole-source-cutoff-mm",
                "2e-9",
            ]
        )

        assert args.magnetic_dipole_enabled is True
        assert args.rider_magnetic_species == "neutron"
        assert args.driver_magnetic_species == "h_minus"
        assert args.rider_spin == [0.0, 3.0, 4.0]
        assert args.driver_spin == [1.0, 0.0, 0.0]
        assert args.stern_gerlach_force_enabled is True
        assert args.dipole_source_model == "full-retarded-point"
        assert args.exact_retarded_backend == "numba_roots_exact_serial"
        assert args.dipole_source_minimum_separation_mm == pytest.approx(2.0e-9)

    def test_parse_args_accepts_disabling_magnetic_dipole_options(self):
        args = cli.parse_args(
            ["--no-magnetic-dipoles", "--no-stern-gerlach", "--no-spin-precession"]
        )

        assert args.magnetic_dipole_enabled is False
        assert args.stern_gerlach_force_enabled is False
        assert args.spin_precession_enabled is False

    def test_parse_args_accepts_full_strict_exact_retarded_backend(self):
        args = cli.parse_args(["--exact-retarded-backend", "numba_full_strict_serial"])

        assert args.exact_retarded_backend == "numba_full_strict_serial"

    def test_parse_args_accepts_certified_metal_exact_retarded_backend(self):
        args = cli.parse_args(
            ["--exact-retarded-backend", "metal_certified_full_strict"]
        )

        assert args.exact_retarded_backend == "metal_certified_full_strict"

    def test_magnetic_dipole_help_names_rfs_defaults(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_args(["--help"])

        help_text = " ".join(capsys.readouterr().out.split())
        assert exc_info.value.code == 0
        assert "intrinsic magnetic-moment dynamics (RFS by default)" in help_text
        assert "full RFS G tensor by default" in help_text
        assert "RFS minimal 2021 by default" in help_text
        assert "full retarded point-dipole oracle (experimental)" in help_text
        assert "Crossing it aborts the run; it is not softening" in help_text

    def test_magnetic_species_choices_include_neutral_and_h_minus_presets(self):
        assert "neutron" in cli.MAGNETIC_SPECIES_CHOICES
        assert "h_minus" in cli.MAGNETIC_SPECIES_CHOICES

    def test_parse_args_accepts_pseudo_grid_options(self):
        args = cli.parse_args(
            [
                "--pseudo-grid",
                "--pseudo-grid-active-rider-count",
                "6",
                "--pseudo-grid-active-driver-count",
                "8",
                "--pseudo-grid-field-rider-count",
                "24",
                "--pseudo-grid-field-driver-count",
                "26",
                "--pseudo-grid-field-deposition-neighbor-count",
                "5",
                "--pseudo-grid-passive-neighbor-count",
                "3",
                "--pseudo-grid-coverage-strategy",
                "farthest_point",
                "--pseudo-grid-coverage-space",
                "phase_space",
                "--pseudo-grid-pair-reuse-window",
                "21",
                "--pseudo-grid-source-weighting-mode",
                "nearest",
                "--pseudo-grid-loss-tracking",
                "--pseudo-grid-causal-pruning",
                "--pseudo-grid-causal-safety-margin-steps",
                "5",
            ]
        )

        assert args.pseudo_grid_enabled is True
        assert args.pseudo_grid_active_rider_count == 6
        assert args.pseudo_grid_active_driver_count == 8
        assert args.pseudo_grid_field_rider_count == 24
        assert args.pseudo_grid_field_driver_count == 26
        assert args.pseudo_grid_field_deposition_neighbor_count == 5
        assert args.pseudo_grid_passive_neighbor_count == 3
        assert args.pseudo_grid_coverage_strategy == "farthest_point"
        assert args.pseudo_grid_coverage_space == "phase_space"
        assert args.pseudo_grid_pair_reuse_window == 21
        assert args.pseudo_grid_source_weighting_mode == "nearest"
        assert args.pseudo_grid_loss_tracking_enabled is True
        assert args.pseudo_grid_causal_history_pruning_enabled is True
        assert args.pseudo_grid_causal_history_safety_margin_steps == 5

    def test_parse_args_accepts_cavity_exit_mode(self):
        args = cli.parse_args(
            [
                "--cavity-exit",
                "--cavity-exit-mode",
                "rider_exit_with_driver_tail",
                "--cavity-exit-length-mm",
                "42",
            ]
        )

        assert args.cavity_exit_enabled is True
        assert args.cavity_exit_mode == "rider_exit_with_driver_tail"
        assert args.cavity_exit_length_mm == pytest.approx(42.0)

    def test_parse_args_accepts_driver_train_options(self):
        args = cli.parse_args(
            [
                "--driver-train",
                "--driver-train-bunch-count",
                "3",
                "--driver-train-z-spacing-mm",
                "2997.9",
                "--driver-train-z-offsets-mm",
                "0",
                "100",
                "250",
                "--driver-train-prehistory-steps",
                "12",
                "--driver-train-preserve-prehistory",
            ]
        )

        assert args.driver_train_enabled is True
        assert args.driver_train_bunch_count == 3
        assert args.driver_train_z_spacing_mm == pytest.approx(2997.9)
        assert args.driver_train_z_offsets_mm == [0.0, 100.0, 250.0]
        assert args.driver_train_prehistory_steps == 12
        assert args.driver_train_preserve_prehistory_in_output is True

    def test_parse_args_accepts_bunch_proximity_timestep_options(self):
        args = cli.parse_args(
            [
                "--adaptive-bunch-proximity",
                "--adaptive-bunch-proximity-sigma-mm",
                "2.5",
                "--adaptive-bunch-proximity-n-sigma",
                "4",
                "--adaptive-bunch-proximity-reduction-factor",
                "8",
                "--adaptive-bunch-proximity-transition-n-sigma",
                "1.5",
            ]
        )

        assert args.adaptive_timestep_bunch_proximity_enabled is True
        assert args.adaptive_timestep_bunch_proximity_sigma_mm == pytest.approx(2.5)
        assert args.adaptive_timestep_bunch_proximity_n_sigma == pytest.approx(4.0)
        assert args.adaptive_timestep_bunch_proximity_reduction_factor == pytest.approx(
            8.0
        )
        assert (
            args.adaptive_timestep_bunch_proximity_transition_n_sigma
            == pytest.approx(1.5)
        )

    def test_parse_args_allows_disabling_boolean_flags(self):
        args = cli.parse_args(
            ["--no-adaptive-debug", "--no-image-weighting", "--no-driver-train"]
        )

        assert args.adaptive_debug is False
        assert args.use_image_weighting is False
        assert args.driver_train_enabled is False

    def test_parse_simulation_type_accepts_aliases(self):
        assert cli._parse_simulation_type("wall") == SimulationType.CONDUCTING_WALL
        assert cli._parse_simulation_type("bunch-to-bunch") == (
            SimulationType.BUNCH_TO_BUNCH
        )

    def test_parse_simulation_type_accepts_enum_instance(self):
        assert cli._parse_simulation_type(SimulationType.SWITCHING_WALL) == (
            SimulationType.SWITCHING_WALL
        )

    def test_parse_simulation_type_accepts_integer_flags(self):
        assert cli._parse_simulation_type(SimulationType.CONDUCTING_WALL.value) == (
            SimulationType.CONDUCTING_WALL
        )
        assert cli._parse_simulation_type(SimulationType.SWITCHING_WALL.value) == (
            SimulationType.SWITCHING_WALL
        )
        assert cli._parse_simulation_type(SimulationType.BUNCH_TO_BUNCH.value) == (
            SimulationType.BUNCH_TO_BUNCH
        )

    def test_parse_simulation_type_rejects_unknown_value(self):
        with pytest.raises(cli.SimulationConfigError, match="Unknown simulation type"):
            cli._parse_simulation_type("not-a-mode")
        with pytest.raises(cli.SimulationConfigError, match="Unknown simulation type"):
            cli._parse_simulation_type(True)
        with pytest.raises(
            cli.SimulationConfigError, match="Unknown simulation type integer"
        ):
            cli._parse_simulation_type(99)

    def test_parse_chrono_mode_accepts_supported_values(self):
        assert cli._parse_chrono_mode("fast") == ChronoMatchingMode.FAST
        assert cli._parse_chrono_mode("averaged") == ChronoMatchingMode.AVERAGED
        assert cli._parse_chrono_mode("legacy") == ChronoMatchingMode.FAST
        assert cli._parse_chrono_mode("average") == ChronoMatchingMode.AVERAGED
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

    def test_parse_startup_mode_accepts_enum_or_supported_alias(self):
        assert cli._parse_startup_mode("cold") == StartupMode.COLD_START
        assert cli._parse_startup_mode("approximate-back-history") == (
            StartupMode.APPROXIMATE_BACK_HISTORY
        )
        assert cli._parse_startup_mode("approximate") == (
            StartupMode.APPROXIMATE_BACK_HISTORY
        )
        assert cli._parse_startup_mode("inertial-prehistory") == (
            StartupMode.INERTIAL_PREHISTORY
        )
        assert cli._parse_startup_mode("inertial_prehistory") == (
            StartupMode.INERTIAL_PREHISTORY
        )
        assert cli._parse_startup_mode("inertial") == (StartupMode.INERTIAL_PREHISTORY)
        assert cli._parse_startup_mode(StartupMode.COLD_START) == (
            StartupMode.COLD_START
        )

    def test_build_integrator_config_accepts_inertial_prehistory(self):
        config = cli._build_integrator_config(
            {
                "steps": 12,
                "time_step": 0.25,
                "wall_position": 1.5,
                "aperture_radius": 0.002,
                "simulation_type": "bunch-to-bunch",
                "startup_mode": "inertial-prehistory",
            }
        )

        assert config.startup_mode is StartupMode.INERTIAL_PREHISTORY

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
                "chrono_mode": "fast",
                "startup_mode": "approximate-back-history",
                "image_subcharge_count": "16",
                "use_image_weighting": "no",
                "z_cutoff_mode": "relative",
                "cavity_exit": {
                    "enabled": True,
                    "mode": "rider_exit_with_driver_tail",
                    "cavity_length_mm": 42.0,
                },
                "pseudo_grid": {
                    "enabled": True,
                    "active_rider_count": 6,
                    "active_driver_count": 7,
                    "field_rider_count": 24,
                    "field_driver_count": 25,
                    "field_deposition_neighbor_count": 6,
                    "passive_neighbor_count": 3,
                    "pair_reuse_window": 20,
                    "causal_history_pruning_enabled": True,
                    "causal_history_safety_margin_steps": 5,
                },
                "driver_train": {
                    "enabled": True,
                    "bunch_count": 3,
                    "z_spacing_mm": 100.0,
                    "z_offsets_mm": [0.0, 100.0, 250.0],
                    "prehistory_steps": 8,
                    "preserve_prehistory_in_output": True,
                },
            }
        )

        assert config.steps == 12
        assert config.time_step == pytest.approx(0.25)
        assert config.simulation_type == SimulationType.SWITCHING_WALL
        assert config.chrono_mode == ChronoMatchingMode.FAST
        assert config.startup_mode == StartupMode.APPROXIMATE_BACK_HISTORY
        assert config.image_subcharge_count == 16
        assert config.use_image_weighting is False
        assert config.z_cutoff_mode == "relative"
        assert config.cavity_exit.enabled is True
        assert config.cavity_exit.mode == "rider_exit_with_driver_tail"
        assert config.cavity_exit.cavity_length_mm == pytest.approx(42.0)
        assert config.pseudo_grid.enabled is True
        assert config.pseudo_grid.active_rider_count == 6
        assert config.pseudo_grid.active_driver_count == 7
        assert config.pseudo_grid.field_rider_count == 24
        assert config.pseudo_grid.field_driver_count == 25
        assert config.pseudo_grid.field_deposition_neighbor_count == 6
        assert config.pseudo_grid.passive_neighbor_count == 3
        assert config.pseudo_grid.pair_reuse_window == 20
        assert config.pseudo_grid.causal_history_pruning_enabled is True
        assert config.pseudo_grid.causal_history_safety_margin_steps == 5
        assert config.driver_train.enabled is True
        assert config.driver_train.bunch_count == 3
        assert config.driver_train.z_spacing_mm == pytest.approx(100.0)
        assert config.driver_train.z_offsets_mm == pytest.approx((0.0, 100.0, 250.0))
        assert config.driver_train.prehistory_steps == 8
        assert config.driver_train.preserve_prehistory_in_output is True

    def test_build_integrator_config_requires_simulation_type(self):
        with pytest.raises(
            cli.SimulationConfigError, match="missing 'simulation_type'"
        ):
            cli._build_integrator_config(
                {
                    "steps": 1,
                    "time_step": 0.1,
                    "wall_position": 0.0,
                    "aperture_radius": 1.0,
                }
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

    def test_merge_simulation_payload_applies_external_field_overrides(self):
        args = _make_args(
            external_e_field_v_per_m=[0.0, 0.0, -1.5e9],
            external_b_field_native=[0.0, 3.0, 0.0],
            external_b_field_tesla=[0.0, 0.0, 2.0],
            external_field_z_min=-0.2,
            external_field_t_max=1.0e-6,
        )

        payload = cli._merge_simulation_payload({}, args)

        assert payload["external_field"] == {
            "enabled": True,
            "electric_field_v_per_m": [0.0, 0.0, -1.5e9],
            "magnetic_field_native": [0.0, 3.0, 0.0],
            "magnetic_field_tesla": [0.0, 0.0, 2.0],
            "z_min": -0.2,
            "t_max": 1.0e-6,
        }

    def test_merge_simulation_payload_keeps_file_passthrough_fields(self):
        payload = cli._merge_simulation_payload(
            {
                "space_charge_enabled": True,
                "space_charge_retarded": False,
                "space_charge_softening_mm": 0.05,
                "auto_duration_enabled": True,
                "auto_duration_crossing_steps": 180,
                "auto_duration_post_factor": 2.25,
            },
            _make_args(),
        )

        assert payload["space_charge_enabled"] is True
        assert payload["space_charge_retarded"] is False
        assert payload["space_charge_softening_mm"] == pytest.approx(0.05)
        assert payload["auto_duration_enabled"] is True
        assert payload["auto_duration_crossing_steps"] == 180
        assert payload["auto_duration_post_factor"] == pytest.approx(2.25)

    def test_merge_simulation_payload_applies_pseudo_grid_overrides(self):
        payload = cli._merge_simulation_payload(
            {
                "pseudo_grid": {
                    "enabled": False,
                    "active_rider_count": 4,
                    "active_driver_count": 4,
                    "pair_reuse_window": 8,
                }
            },
            _make_args(
                pseudo_grid_enabled=True,
                pseudo_grid_active_rider_count=7,
                pseudo_grid_active_driver_count=9,
                pseudo_grid_field_rider_count=31,
                pseudo_grid_field_driver_count=33,
                pseudo_grid_field_deposition_neighbor_count=5,
                pseudo_grid_passive_neighbor_count=2,
                pseudo_grid_pair_reuse_window=30,
                pseudo_grid_causal_history_pruning_enabled=True,
            ),
        )

        assert payload["pseudo_grid"]["enabled"] is True
        assert payload["pseudo_grid"]["active_rider_count"] == 7
        assert payload["pseudo_grid"]["active_driver_count"] == 9
        assert payload["pseudo_grid"]["field_rider_count"] == 31
        assert payload["pseudo_grid"]["field_driver_count"] == 33
        assert payload["pseudo_grid"]["field_deposition_neighbor_count"] == 5
        assert payload["pseudo_grid"]["passive_neighbor_count"] == 2
        assert payload["pseudo_grid"]["pair_reuse_window"] == 30
        assert payload["pseudo_grid"]["causal_history_pruning_enabled"] is True

    def test_merge_simulation_payload_applies_macroparticle_smearing_overrides(self):
        payload = cli._merge_simulation_payload(
            {"macroparticle_smearing": {"enabled": False, "subcharge_count": 4}},
            _make_args(
                macroparticle_smearing_enabled=True,
                macroparticle_smearing_subcharge_count=6,
                macroparticle_smearing_sigma_multiplier=0.5,
                macroparticle_smearing_position_sigma_mm=0.1,
                macroparticle_smearing_refresh_policy="per-step",
            ),
        )

        smearing = payload["macroparticle_smearing"]
        assert smearing["enabled"] is True
        assert smearing["subcharge_count"] == 6
        assert smearing["sigma_multiplier"] == pytest.approx(0.5)
        assert smearing["position_sigma_mm"] == pytest.approx(0.1)
        assert smearing["refresh_policy"] == "per_step"

    def test_merge_simulation_payload_applies_magnetic_dipole_overrides(self):
        payload = cli._merge_simulation_payload(
            {
                "magnetic_dipole": {
                    "enabled": True,
                    "stern_gerlach_force_enabled": True,
                    "exact_retarded_backend": "python",
                    "source": {
                        "model": "covariant_retarded_point",
                        "minimum_separation_mm": 4.0e-9,
                        "relative_stencil_step": 2.0e-3,
                        "minimum_stencil_step_mm": 3.0e-15,
                        "root_tolerance_mm": 4.0e-21,
                        "max_root_iterations": 72,
                    },
                    "rider": {
                        "species": "electron",
                        "magnetic_moment_j_per_t": -1.0e-23,
                    },
                }
            },
            _make_args(
                magnetic_dipole_enabled=False,
                rider_magnetic_species="neutron",
                rider_spin=[0.0, 1.0, 0.0],
                driver_magnetic_species="antiproton",
                driver_spin=[1.0, 0.0, 0.0],
                stern_gerlach_force_enabled=False,
                spin_precession_enabled=False,
                dipole_source_model="off",
                exact_retarded_backend="numba_roots_exact_serial",
                dipole_source_minimum_separation_mm=8.0e-9,
            ),
        )

        magnetic = payload["magnetic_dipole"]
        assert magnetic["enabled"] is False
        assert magnetic["stern_gerlach_force_enabled"] is False
        assert magnetic["spin_precession_enabled"] is False
        assert magnetic["exact_retarded_backend"] == "numba_roots_exact_serial"
        assert magnetic["source"] == {
            "model": "off",
            "minimum_separation_mm": 8.0e-9,
            "relative_stencil_step": 2.0e-3,
            "minimum_stencil_step_mm": 3.0e-15,
            "root_tolerance_mm": 4.0e-21,
            "max_root_iterations": 72,
        }
        assert magnetic["rider"]["species"] == "neutron"
        assert magnetic["rider"]["rest_spin"] == [0.0, 1.0, 0.0]
        assert magnetic["rider"]["magnetic_moment_j_per_t"] == pytest.approx(-1.0e-23)
        assert magnetic["driver"]["species"] == "antiproton"
        assert magnetic["driver"]["rest_spin"] == [1.0, 0.0, 0.0]

    def test_merge_simulation_payload_canonicalizes_legacy_backend_alias(self):
        payload = cli._merge_simulation_payload(
            {"magnetic_dipole": {"source": {"backend": "numba_full_strict_serial"}}},
            _make_args(),
        )

        magnetic = payload["magnetic_dipole"]
        assert magnetic["exact_retarded_backend"] == "numba_full_strict_serial"
        assert "backend" not in magnetic["source"]

    def test_merge_simulation_payload_rejects_conflicting_backend_alias(self):
        with pytest.raises(cli.SimulationConfigError, match="conflicts with legacy"):
            cli._merge_simulation_payload(
                {
                    "magnetic_dipole": {
                        "exact_retarded_backend": "python",
                        "source": {"backend": "numba_full_strict_serial"},
                    }
                },
                _make_args(),
            )

    def test_merge_simulation_payload_applies_cavity_exit_mode_override(self):
        payload = cli._merge_simulation_payload(
            {"cavity_exit": {"enabled": False, "mode": "first_exit"}},
            _make_args(
                cavity_exit_enabled=True,
                cavity_exit_mode="rider_exit_with_driver_tail",
                cavity_exit_length_mm=42.0,
            ),
        )

        assert payload["cavity_exit"]["enabled"] is True
        assert payload["cavity_exit"]["mode"] == "rider_exit_with_driver_tail"
        assert payload["cavity_exit"]["cavity_length_mm"] == pytest.approx(42.0)

    def test_merge_simulation_payload_applies_driver_train_overrides(self):
        payload = cli._merge_simulation_payload(
            {"driver_train": {"enabled": False, "bunch_count": 1}},
            _make_args(
                driver_train_enabled=True,
                driver_train_bunch_count=4,
                driver_train_z_spacing_mm=2997.9,
                driver_train_z_offsets_mm=[0.0, 10.0, 20.0, 30.0],
                driver_train_prehistory_steps=16,
                driver_train_preserve_prehistory_in_output=True,
            ),
        )

        assert payload["driver_train"]["enabled"] is True
        assert payload["driver_train"]["bunch_count"] == 4
        assert payload["driver_train"]["z_spacing_mm"] == pytest.approx(2997.9)
        assert payload["driver_train"]["z_offsets_mm"] == [0.0, 10.0, 20.0, 30.0]
        assert payload["driver_train"]["prehistory_steps"] == 16
        assert payload["driver_train"]["preserve_prehistory_in_output"] is True

    def test_merge_simulation_payload_applies_adaptive_bunch_proximity_overrides(self):
        payload = cli._merge_simulation_payload(
            {"adaptive_timestep": {"enabled": False}},
            _make_args(
                adaptive_timestep_enabled=True,
                adaptive_timestep_bunch_proximity_enabled=True,
                adaptive_timestep_bunch_proximity_sigma_mm=3.0,
                adaptive_timestep_bunch_proximity_n_sigma=6.0,
                adaptive_timestep_bunch_proximity_reduction_factor=12.0,
                adaptive_timestep_bunch_proximity_transition_n_sigma=2.5,
            ),
        )

        adaptive = payload["adaptive_timestep"]
        assert adaptive["enabled"] is True
        assert adaptive["bunch_proximity_enabled"] is True
        assert adaptive["bunch_proximity_sigma_mm"] == pytest.approx(3.0)
        assert adaptive["bunch_proximity_n_sigma"] == pytest.approx(6.0)
        assert adaptive["bunch_proximity_reduction_factor"] == pytest.approx(12.0)
        assert adaptive["bunch_proximity_transition_n_sigma"] == pytest.approx(2.5)

    def test_build_request_defaults_to_medina_lad_rr(self):
        request = cli.build_request(_make_args())

        assert request.config.radiation_reaction_mode == "medina_lad"

    def test_build_request_keeps_magnetic_dipoles_off_by_default(self):
        request = cli.build_request(_make_args())

        assert request.config.magnetic_dipole.enabled is False
        assert request.config.magnetic_dipole.spin_model == "rfs_minimal_2021"
        assert request.config.magnetic_dipole.stern_gerlach_model == "rfs_full_g"
        assert request.config.magnetic_dipole.exact_retarded_backend == "python"
        assert request.config.magnetic_dipole.source.model == "off"
        assert request.config.magnetic_dipole.source.minimum_separation_mm == (
            pytest.approx(2.0e-9)
        )
        assert request.config.magnetic_dipole.rider.species == "electron"
        assert request.config.magnetic_dipole.driver.species == "proton"

    def test_direct_config_accepts_legacy_magnetic_diagnostic_pair(self):
        magnetic = cli._build_magnetic_dipole_config(
            {
                "spin_model": "bmt_frenkel",
                "stern_gerlach_model": "static_rest_gradient",
            }
        )

        assert magnetic.spin_model == "bmt_frenkel"
        assert magnetic.stern_gerlach_model == "static_rest_gradient"

    def test_direct_config_preserves_full_retarded_source_controls(self):
        magnetic = cli._build_magnetic_dipole_config(
            {
                "exact_retarded_backend": "numba_roots_exact_serial",
                "source": {
                    "model": "full_retarded_point",
                    "minimum_separation_mm": 7.0e-9,
                    "relative_stencil_step": 2.0e-3,
                    "minimum_stencil_step_mm": 3.0e-15,
                    "root_tolerance_mm": 4.0e-21,
                    "max_root_iterations": 80,
                },
            }
        )

        assert magnetic.source.model == "covariant_retarded_point"
        assert magnetic.exact_retarded_backend == "numba_roots_exact_serial"
        assert magnetic.source.minimum_separation_mm == pytest.approx(7.0e-9)
        assert magnetic.source.relative_stencil_step == pytest.approx(2.0e-3)
        assert magnetic.source.minimum_stencil_step_mm == pytest.approx(3.0e-15)
        assert magnetic.source.root_tolerance_mm == pytest.approx(4.0e-21)
        assert magnetic.source.max_root_iterations == 80

    def test_direct_config_accepts_legacy_source_backend_alias(self):
        magnetic = cli._build_magnetic_dipole_config(
            {"source": {"backend": "numba_full_strict_serial"}}
        )

        assert magnetic.exact_retarded_backend == "numba_full_strict_serial"

    def test_direct_config_accepts_matching_backend_alias(self):
        magnetic = cli._build_magnetic_dipole_config(
            {
                "exact_retarded_backend": "numba_roots_exact_serial",
                "source": {"backend": "numba_roots_exact_serial"},
            }
        )

        assert magnetic.exact_retarded_backend == "numba_roots_exact_serial"

    def test_direct_config_rejects_conflicting_backend_alias(self):
        with pytest.raises(cli.SimulationConfigError, match="conflicts with legacy"):
            cli._build_magnetic_dipole_config(
                {
                    "exact_retarded_backend": "python",
                    "source": {"backend": "numba_full_strict_serial"},
                }
            )

    def test_direct_config_rejects_non_object_dipole_source(self):
        with pytest.raises(
            cli.SimulationConfigError,
            match="magnetic_dipole.source must be a JSON object",
        ):
            cli._build_magnetic_dipole_config({"source": "full-retarded-point"})

    @pytest.mark.parametrize(
        ("spin_model", "stern_gerlach_model", "required_model"),
        (
            ("bmt_frenkel", "rfs_full_g", "rfs_minimal_2021"),
            ("rfs_minimal_2021", "static_rest_gradient", "bmt_frenkel"),
        ),
    )
    def test_direct_config_rejects_mismatched_magnetic_models(
        self,
        spin_model: str,
        stern_gerlach_model: str,
        required_model: str,
    ):
        with pytest.raises(
            cli.SimulationConfigError,
            match=f"requires spin_model '{required_model}'",
        ):
            cli._build_magnetic_dipole_config(
                {
                    "spin_model": spin_model,
                    "stern_gerlach_model": stern_gerlach_model,
                }
            )

    def test_build_request_applies_direct_magnetic_dipole_options(self):
        request = cli.build_request(
            _make_args(
                magnetic_dipole_enabled=True,
                rider_magnetic_species="neutron",
                driver_magnetic_species="antiproton",
                rider_spin=[0.0, 3.0, 4.0],
                driver_spin=[1.0, 0.0, 0.0],
                stern_gerlach_force_enabled=True,
                dipole_source_model="full-retarded-point",
                exact_retarded_backend="numba_roots_exact_serial",
                dipole_source_minimum_separation_mm=6.0e-9,
            )
        )

        magnetic = request.config.magnetic_dipole
        assert magnetic.enabled is True
        assert magnetic.stern_gerlach_force_enabled is True
        assert magnetic.rider.species == "neutron"
        assert magnetic.rider.rest_spin == pytest.approx((0.0, 0.6, 0.8))
        assert magnetic.driver.species == "antiproton"
        assert magnetic.driver.rest_spin == pytest.approx((1.0, 0.0, 0.0))
        assert magnetic.source.model == "covariant_retarded_point"
        assert magnetic.exact_retarded_backend == "numba_roots_exact_serial"
        assert magnetic.source.minimum_separation_mm == pytest.approx(6.0e-9)
        assert request.config.radiation_reaction_mode == "off"

    def test_build_request_preserves_explicit_rr_with_rfs(self):
        request = cli.build_request(
            _make_args(
                magnetic_dipole_enabled=True,
                radiation_reaction_mode="medina_lad",
            )
        )

        assert request.config.radiation_reaction_mode == "medina_lad"

    def test_driver_from_rider_inherits_magnetic_species_unless_overridden(self):
        request = cli.build_request(
            _make_args(
                simulation_type="bunch-to-bunch",
                driver_from_rider=True,
                magnetic_dipole_enabled=True,
                rider_magnetic_species="electron",
                rider_spin=[0.0, 1.0, 0.0],
            )
        )

        assert request.config.magnetic_dipole.rider.species == "electron"
        assert request.config.magnetic_dipole.driver.species == "electron"
        assert request.config.magnetic_dipole.driver.rest_spin == pytest.approx(
            (0.0, 1.0, 0.0)
        )

    def test_driver_from_rider_preserves_explicit_driver_spin(self):
        request = cli.build_request(
            _make_args(
                simulation_type="bunch-to-bunch",
                driver_from_rider=True,
                magnetic_dipole_enabled=True,
                rider_magnetic_species="electron",
                rider_spin=[0.0, 1.0, 0.0],
                driver_spin=[1.0, 0.0, 0.0],
            )
        )

        assert request.config.magnetic_dipole.driver.species == "electron"
        assert request.config.magnetic_dipole.driver.rest_spin == pytest.approx(
            (1.0, 0.0, 0.0)
        )

    def test_build_request_native_json_can_supply_custom_h_minus_moment(
        self, tmp_path: Path
    ):
        config_path = tmp_path / "h_minus.json"
        config_path.write_text(
            json.dumps(
                {
                    "magnetic_dipole": {
                        "enabled": True,
                        "rider": {
                            "species": "h_minus",
                            "magnetic_moment_j_per_t": -9.2e-24,
                            "spin_quantum_number": 0.5,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        request = cli.build_request(_make_args(config=config_path))

        rider = request.config.magnetic_dipole.rider
        assert rider.species == "h_minus"
        assert rider.magnetic_moment_j_per_t == pytest.approx(-9.2e-24)
        assert rider.spin_quantum_number == pytest.approx(0.5)

    def test_build_request_rejects_h_minus_without_custom_moment(self):
        with pytest.raises(
            cli.SimulationConfigError,
            match="h_minus.*no supported moment preset",
        ):
            cli.build_request(
                _make_args(
                    magnetic_dipole_enabled=True,
                    rider_magnetic_species="h_minus",
                )
            )

    def test_build_request_rejects_magnetic_dipoles_with_pseudo_grid(self):
        with pytest.raises(
            cli.SimulationConfigError,
            match="not compatible with pseudo-grid",
        ):
            cli.build_request(
                _make_args(
                    magnetic_dipole_enabled=True,
                    pseudo_grid_enabled=True,
                )
            )

    def test_build_request_supports_neutral_rider_from_native_particle_json(
        self, tmp_path: Path
    ):
        config_path = tmp_path / "neutron.json"
        config_path.write_text(
            json.dumps(
                {
                    "rider": {
                        "kinetic_energy_mev": 1.0,
                        "mass_amu": 1.00866491606,
                        "charge_sign": 0.0,
                    }
                }
            ),
            encoding="utf-8",
        )

        request = cli.build_request(
            _make_args(
                config=config_path,
                magnetic_dipole_enabled=True,
                rider_magnetic_species="neutron",
            )
        )

        assert request.config.magnetic_dipole.rider.species == "neutron"
        assert np.all(request.rider["q"] == 0.0)
        assert np.all(request.rider["q_source"] == 0.0)
        assert np.all(request.rider["q_observer"] == 0.0)

    def test_build_request_rr_flag_overrides_config_file(self, tmp_path: Path):
        config_path = tmp_path / "rr_mode.json"
        config_path.write_text(
            json.dumps({"radiation_reaction_mode": "off"}),
            encoding="utf-8",
        )

        request = cli.build_request(
            _make_args(
                config=config_path,
                radiation_reaction_mode="power_matched_damping",
            )
        )

        assert request.config.radiation_reaction_mode == "power_matched_damping"

    def test_build_request_parses_external_field_config(self, tmp_path: Path):
        config_path = tmp_path / "external_field.json"
        config_path.write_text(
            json.dumps(
                {
                    "external_field": {
                        "enabled": True,
                        "electric_field_v_per_m": [0.0, 0.0, -1.5e9],
                        "magnetic_field_native": [0.0, 3.0, 0.0],
                        "magnetic_field_gradient_t_per_m": [
                            [-1.0, 0.0, 0.0],
                            [0.0, -2.0, 0.0],
                            [0.0, 0.0, 3.0],
                        ],
                        "z_min": -0.2,
                        "z_max": 0.2,
                        "t_min": 1.0e-6,
                        "t_max": 2.0e-6,
                    }
                }
            ),
            encoding="utf-8",
        )

        request = cli.build_request(_make_args(config=config_path))

        assert request.external_field is not None
        assert request.external_field.electric_field_native[2] == pytest.approx(
            electric_field_v_per_m_to_native(-1.5e9)
        )
        assert request.external_field.magnetic_field_native == pytest.approx(
            (0.0, 3.0, 0.0)
        )
        np.testing.assert_allclose(
            request.external_field.magnetic_field_gradient_t_per_m,
            ((-1.0, 0.0, 0.0), (0.0, -2.0, 0.0), (0.0, 0.0, 3.0)),
        )
        assert request.external_field.z_min == pytest.approx(-0.2)
        assert request.external_field.z_max == pytest.approx(0.2)
        assert request.external_field.t_min == pytest.approx(1.0e-6)
        assert request.external_field.t_max == pytest.approx(2.0e-6)

    def test_build_external_field_accepts_magnetic_field_in_tesla(self):
        from core.external_fields import magnetic_field_native_to_tesla

        field = cli._build_external_field_config(
            {"enabled": True, "magnetic_field_tesla": [0.1, -0.2, 0.3]}
        )

        assert field is not None
        converted = tuple(
            magnetic_field_native_to_tesla(component)
            for component in field.magnetic_field_native
        )
        assert converted == pytest.approx((0.1, -0.2, 0.3))

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

    def test_build_request_parses_space_charge_config(self, tmp_path: Path):
        config_path = tmp_path / "space_charge.json"
        config_path.write_text(
            json.dumps(
                {
                    "space_charge_enabled": True,
                    "space_charge_retarded": False,
                    "space_charge_softening_mm": 0.125,
                    "space_charge_bunch_sigma_mm": 0.02,
                    "space_charge_min_retarded_steps": 7,
                }
            ),
            encoding="utf-8",
        )

        request = cli.build_request(_make_args(config=config_path))

        assert request.space_charge is not None
        assert request.space_charge.enabled is True
        assert request.space_charge.retarded is False
        assert request.space_charge.softening_mm == pytest.approx(0.125)
        assert request.space_charge.bunch_sigma_mm == pytest.approx(0.02)
        assert request.space_charge.min_retarded_steps == 7

    def test_build_request_rejects_auto_duration_for_non_b2b(self):
        with pytest.raises(
            cli.SimulationConfigError,
            match="auto_duration is only supported for BUNCH_TO_BUNCH simulations",
        ):
            cli.build_request(_make_args(auto_duration=True))

    def test_build_request_accepts_auto_duration_for_b2b(self):
        request = cli.build_request(
            _make_args(
                simulation_type="bunch-to-bunch",
                driver_from_rider=True,
                auto_duration=True,
                auto_duration_crossing_steps=150,
                auto_duration_post_factor=2.5,
            )
        )

        assert request.auto_duration_enabled is True
        assert request.auto_duration_crossing_steps == 150
        assert request.auto_duration_post_factor == pytest.approx(2.5)

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

    def test_build_request_loads_3d_particle_configs_from_file(self, tmp_path: Path):
        config_path = tmp_path / "b2b_3d.json"
        config_path.write_text(
            json.dumps(
                {
                    "simulation_type": "bunch-to-bunch",
                    "rider": {
                        "kinetic_energy_mev": 12.0,
                        "mass_amu": 1.007276466621,
                        "charge_sign": -1.0,
                        "particle_count": 3,
                        "seed": 123,
                        "starting_position_mm": [1.0, 2.0, 3.0],
                        "momentum_axis": [1.0, 0.0, 0.0],
                        "longitudinal_span_mm": 2.0,
                        "transverse_distance_mm": 0.0,
                    },
                    "driver": {
                        "kinetic_energy_mev": 18.0,
                        "mass_amu": 1.007276466621,
                        "charge_sign": 1.0,
                        "particle_count": 3,
                        "seed": 124,
                        "starting_position_mm": [4.0, 5.0, 6.0],
                        "momentum_axis": [0.0, -1.0, 0.0],
                        "longitudinal_span_mm": 4.0,
                        "transverse_distance_mm": 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )

        request = cli.build_request(_make_args(config=config_path))
        repeated = cli.build_request(_make_args(config=config_path))

        assert request.config.simulation_type == SimulationType.BUNCH_TO_BUNCH
        assert request.driver is not None
        assert repeated.driver is not None
        assert request.rider["x"] == pytest.approx(repeated.rider["x"])
        assert request.driver["y"] == pytest.approx(repeated.driver["y"])
        assert request.rider["y"] == pytest.approx([2.0, 2.0, 2.0])
        assert request.rider["z"] == pytest.approx([3.0, 3.0, 3.0])
        assert request.rider["py"] == pytest.approx([0.0, 0.0, 0.0])
        assert request.rider["pz"] == pytest.approx([0.0, 0.0, 0.0])
        assert np.std(request.rider["x"]) > 0.0
        assert request.driver["x"] == pytest.approx([4.0, 4.0, 4.0])
        assert request.driver["z"] == pytest.approx([6.0, 6.0, 6.0])
        assert request.driver["px"] == pytest.approx([0.0, 0.0, 0.0])
        assert np.all(request.driver["py"] < 0.0)
        assert request.driver["pz"] == pytest.approx([0.0, 0.0, 0.0])
        assert np.std(request.driver["y"]) > 0.0

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
        with pytest.raises(
            cli.SimulationConfigError, match="Configuration file not found"
        ):
            cli._load_config(tmp_path / "missing.json")

    def test_load_config_rejects_invalid_json(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{not-json", encoding="utf-8")

        with pytest.raises(cli.SimulationConfigError, match="not valid JSON"):
            cli._load_config(path)

    def test_build_particle_state_requires_core_fields(self):
        with pytest.raises(cli.SimulationConfigError, match="missing required fields"):
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

        def fake_run_sweep_from_config(
            *, config_path, output_dir, verbose, verbosity_overrides, workers=None
        ):
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

    def test_run_sweep_returns_2_on_exception(
        self, monkeypatch, tmp_path: Path, capsys
    ):
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

    def test_main_writes_saved_optimization_results_summary_json(self, tmp_path: Path):
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
            [
                "--results-file",
                str(results_path),
                "--output",
                str(output_path),
                "--quiet",
            ]
        )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert result == 0
        assert payload["result_type"] == "optimization"
        assert payload["optimization_method"] == "genetic_algorithm"
        assert payload["evaluation_count"] == 2
        assert payload["finite_evaluation_count"] == 1
        assert payload["successful_evaluation_count"] == 2
        assert payload["halted_evaluation_count"] == 0
        assert payload["failed_evaluation_count"] == 0
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
        assert "Successful Evaluation Count: 1" in output
        assert "rank=1, metric=2.5, delta_e_mev=4, percent_gain=1.5" in output

    def test_main_returns_2_for_unknown_results_file_format(
        self, tmp_path: Path, capsys
    ):
        results_path = tmp_path / "unknown.json"
        results_path.write_text(json.dumps({"unexpected": []}), encoding="utf-8")

        result = cli.main(["--results-file", str(results_path)])

        assert result == 2
        assert "Cannot parse this file format." in capsys.readouterr().err

    def test_main_returns_2_for_legacy_results_file_format(
        self, tmp_path: Path, capsys
    ):
        results_path = tmp_path / "legacy_results.json"
        results_path.write_text(
            json.dumps(
                {"core": {"rider": {"positions_mm": {}, "conjugate_momenta": {}}}}
            ),
            encoding="utf-8",
        )

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

    def test_main_runs_testbed_config_through_testbed_runner(
        self, monkeypatch, tmp_path: Path
    ):
        config_path = tmp_path / "study_config.json"
        output_path = tmp_path / "summary.json"
        source_occluders = [
            {
                "axis": [0.0, 0.0, 1.0],
                "center_mm": [0.0, 0.0, 0.0],
                "radius_mm": 2.0,
                "length_mm": 500.0,
                "label": "study_pipe",
            }
        ]
        testbed_options = SimulationOptions(
            config_name=config_path.name,
            config_dir=tmp_path,
            output_dir=tmp_path / "artifacts",
            macroparticle_smearing_enabled=True,
            macroparticle_smearing_use_momentum_errors=False,
            beamline_geometry_enabled=True,
            beamline_geometry_occluders=source_occluders,
        )
        config_path.write_text(
            json.dumps(testbed_options.to_dict()),
            encoding="utf-8",
        )
        captured = {}
        result = SimpleNamespace(
            duration_s=1.25,
            filename_base="study_config_20260822_120000",
            halted_early=False,
            halt_reason=None,
            num_particles_dead=0,
            rider_delta_e=0.125,
            rider_gamma_initial=2.0,
            rider_gamma_final=2.25,
            driver_gamma_initial=1.5,
            driver_gamma_final=1.5,
            energy_ledger_metrics={"rider_final_delta_kinetic_energy_mev": 0.125},
            saved_paths={"trajectory_json": tmp_path / "trajectory.json"},
        )

        def fake_run_testbed(received_options):
            captured["options"] = received_options
            return result

        monkeypatch.setattr(
            "lw_integrator.testbed_runner.run_testbed", fake_run_testbed
        )

        exit_code = cli.main(
            [
                "--testbed-config",
                str(config_path),
                "--output",
                str(output_path),
                "--quiet",
            ]
        )

        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert exit_code == 0
        loaded_options = captured["options"]
        assert loaded_options.config_name == config_path.name
        assert loaded_options.macroparticle_smearing_use_momentum_errors is False
        assert loaded_options.beamline_geometry_enabled is True
        assert loaded_options.beamline_geometry_occluders == source_occluders
        assert report["run_mode"] == "testbed_config"
        assert report["config_path"] == str(config_path)
        assert report["rider_delta_e_mev"] == pytest.approx(0.125)
        assert report["saved_paths"] == {
            "trajectory_json": str(tmp_path / "trajectory.json")
        }

    def test_main_rejects_testbed_config_mode_conflicts(self, tmp_path: Path, capsys):
        config_path = tmp_path / "study_config.json"

        exit_code = cli.main(
            [
                "--testbed-config",
                str(config_path),
                "--config",
                str(tmp_path / "native_config.json"),
            ]
        )

        assert exit_code == 2
        assert "cannot be combined with --config" in capsys.readouterr().err

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

    def test_main_runs_minimal_config_without_monkeypatch(self, tmp_path: Path):
        config_path = tmp_path / "minimal.json"
        output_path = tmp_path / "summary.json"
        config_path.write_text(
            json.dumps(
                {
                    "steps": 2,
                    "time_step": 1e-6,
                    "simulation_type": "wall",
                    "wall_position": 1000.0,
                    "aperture_radius": 1000.0,
                    "bunch_mean": 1000.0,
                    "rider": {
                        "kinetic_energy_mev": 1.0,
                        "mass_amu": 0.00054857990907,
                        "charge_sign": -1.0,
                        "position_z": -10.0,
                        "particle_count": 1,
                        "transverse_radius": 0.0,
                        "transverse_momentum": 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )

        result = cli.main(
            ["--config", str(config_path), "--output", str(output_path), "--quiet"]
        )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert result == 0
        assert payload["steps_completed"] == 2
        assert payload["initial_z_mm"] == pytest.approx(-10.0)
        assert payload["final_z_mm"] > payload["initial_z_mm"]
        assert payload["delta_gamma_mean"] == pytest.approx(0.0)

    def test_main_prints_driver_summary_when_present(self, monkeypatch, capsys):
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
                [1, 2, 3],
            ),
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

    def test_build_report_records_source_model_and_exact_retarded_backend(self):
        trajectory = [
            {
                "t": np.array([0.0]),
                "z": np.array([1.0]),
                "gamma": np.array([2.0]),
                "bz": np.array([0.25]),
            }
        ]
        magnetic = cli._build_magnetic_dipole_config(
            {
                "exact_retarded_backend": "numba_roots_exact_serial",
                "source": {
                    "model": "covariant_retarded_point",
                },
            }
        )

        report = cli.build_report(trajectory, magnetic_dipole=magnetic)

        assert report["magnetic_dipole_source"] == {
            "model": "covariant_retarded_point",
        }
        assert report["exact_retarded"] == {
            "backend": "numba_roots_exact_serial",
        }

    def test_testbed_report_records_source_model_and_exact_retarded_backend(
        self, tmp_path: Path
    ):
        result = SimpleNamespace(
            duration_s=1.25,
            filename_base="capture",
            halted_early=False,
            halt_reason=None,
            num_particles_dead=0,
            rider_delta_e=0.0,
            rider_gamma_initial=1.0,
            rider_gamma_final=1.0,
            driver_gamma_initial=1.0,
            driver_gamma_final=1.0,
            energy_ledger_metrics={},
            saved_paths={},
        )
        options = SimulationOptions(
            magnetic_dipole_source_model="covariant_retarded_point",
            magnetic_dipole_exact_retarded_backend="numba_full_strict_serial",
        )

        report = cli._build_testbed_report(result, tmp_path / "capture.json", options)

        assert report["magnetic_dipole_source"] == {
            "model": "covariant_retarded_point",
        }
        assert report["exact_retarded"] == {
            "backend": "numba_full_strict_serial",
        }

    def test_metal_report_records_certification_and_fallback_counts(self, monkeypatch):
        from core.metal_certified_roots import MetalCertifiedRootDiagnostics

        monkeypatch.setattr(
            "core.metal_certified_roots.metal_certified_root_diagnostics",
            lambda: MetalCertifiedRootDiagnostics(7, 5, 2, 2047, 1, 0),
        )

        report = cli._exact_retarded_report("metal_certified_full_strict")

        assert report == {
            "backend": "metal_certified_full_strict",
            "metal_certified_roots": {
                "calls": 7,
                "below_threshold_calls": 5,
                "dispatches": 2,
                "accepted_proposals": 2047,
                "cpu_fallbacks": 1,
                "dispatch_failures": 0,
            },
        }

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

    def test_main_presents_pseudo_grid_magnetic_incompatibility(self, capsys):
        result = cli.main(["--pseudo-grid", "--magnetic-dipoles", "--quiet"])

        assert result == 2
        error = capsys.readouterr().err
        assert "not compatible with pseudo-grid" in error
        assert "--no-magnetic-dipoles" in error


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
        assert (
            captured["radiation_reaction_mode"]
            == request.config.radiation_reaction_mode
        )
        assert captured["space_charge"] is request.space_charge
        assert captured["external_field"] is request.external_field
        assert captured["pseudo_grid"] is request.config.pseudo_grid
        assert captured["driver_train"] is request.config.driver_train
        assert captured["z_cutoff_mode"] == request.config.z_cutoff_mode
        assert captured["cavity_exit"] is request.config.cavity_exit
        assert (
            captured["macroparticle_smearing"] is request.config.macroparticle_smearing
        )
        assert captured["magnetic_dipole"] is request.config.magnetic_dipole
        assert captured["checkpoint"] is request.config.checkpoint

    def test_run_simulation_applies_auto_duration_when_enabled(self, monkeypatch):
        request = cli.build_request(
            _make_args(
                simulation_type="bunch-to-bunch",
                driver_from_rider=True,
                auto_duration=True,
                auto_duration_crossing_steps=120,
                auto_duration_post_factor=2.5,
            )
        )

        assert request.driver is not None
        request.driver["z"] = request.driver["z"] + 1000.0

        expected_steps, expected_h_step = cli._resolve_auto_duration(request)
        captured = {}

        def fake_retarded_integrator(**kwargs):
            captured.update(kwargs)
            return ["rider"], ["driver"]

        monkeypatch.setattr(cli, "retarded_integrator", fake_retarded_integrator)

        rider, driver = cli.run_simulation(request)

        assert rider == ["rider"]
        assert driver == ["driver"]
        assert captured["steps"] == expected_steps
        assert captured["h_step"] == pytest.approx(expected_h_step)
        assert captured["steps"] == 300
        assert captured["h_step"] != pytest.approx(request.config.time_step)

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


class TestCliBeamlineGeometry:
    def test_parse_args_accepts_beamline_geometry_flags(self):
        args = cli.parse_args(
            ["--beamline-geometry-enabled", "--beamline-geometry-file", "geom.json"]
        )
        assert args.beamline_geometry_enabled is True
        assert args.beamline_geometry_file == "geom.json"

    def test_parse_args_no_beamline_geometry_flag(self):
        args = cli.parse_args(["--no-beamline-geometry"])
        assert args.beamline_geometry_enabled is False

    def test_merge_simulation_payload_applies_beamline_geometry_enabled(self):
        payload = cli._merge_simulation_payload(
            {"beamline_geometry": {"enabled": False, "occluders": []}},
            _make_args(beamline_geometry_enabled=True),
        )
        assert payload["beamline_geometry"]["enabled"] is True

    def test_merge_simulation_payload_preserves_json_beamline_geometry(self):
        payload = cli._merge_simulation_payload(
            {
                "beamline_geometry": {
                    "enabled": True,
                    "occluders": [
                        {
                            "axis": [0.0, 0.0, 1.0],
                            "center_mm": [0.0, 0.0, 0.0],
                            "radius_mm": 2.0,
                            "length_mm": 500.0,
                            "label": "proton_straight_pipe",
                        },
                        {
                            "axis": [0.0, 1.0, 0.0],
                            "center_mm": [0.0, 0.0, 0.0],
                            "radius_mm": 2.0,
                            "length_mm": 500.0,
                            "label": "source_channel_pipe",
                        },
                    ],
                }
            },
            _make_args(),
        )

        config = cli._build_integrator_config(payload)

        assert config.beamline_geometry.enabled is True
        assert [occluder.label for occluder in config.beamline_geometry.occluders] == [
            "proton_straight_pipe",
            "source_channel_pipe",
        ]

    def test_merge_simulation_payload_loads_beamline_geometry_file(self, tmp_path):
        geom_file = tmp_path / "geom.json"
        geom_file.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "occluders": [
                        {
                            "axis": [0.0, 0.0, 1.0],
                            "center_mm": [0.0, 0.0, 0.0],
                            "radius_mm": 15.0,
                            "length_mm": 2000.0,
                            "label": "electron_pipe",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        payload = cli._merge_simulation_payload(
            {},
            _make_args(beamline_geometry_file=str(geom_file)),
        )
        block = payload["beamline_geometry"]
        assert block["enabled"] is True
        assert len(block["occluders"]) == 1
        assert block["occluders"][0]["label"] == "electron_pipe"

    def test_build_beamline_geometry_config_disabled_default(self):
        config = cli._build_beamline_geometry_config(None)
        assert config.enabled is False
        assert config.occluders == []

    def test_build_beamline_geometry_config_two_occluders(self):
        config = cli._build_beamline_geometry_config(
            {
                "enabled": True,
                "occluders": [
                    {
                        "axis": [0.0, 0.0, 2.0],
                        "center_mm": [1.0, 2.0, 3.0],
                        "radius_mm": 15.0,
                        "length_mm": 2000.0,
                        "label": "pipe_a",
                    },
                    {
                        "axis": [1.0, 0.0, 0.0],
                        "center_mm": [-1.0, -2.0, -3.0],
                        "radius_mm": 7.5,
                        "length_mm": 500.0,
                        "label": "pipe_b",
                    },
                ],
            }
        )
        assert config.enabled is True
        assert len(config.occluders) == 2
        first, second = config.occluders
        assert first.label == "pipe_a"
        assert first.radius_mm == pytest.approx(15.0)
        assert first.length_mm == pytest.approx(2000.0)
        assert first.center_mm == (1.0, 2.0, 3.0)
        # axis is normalized in Occluder.__post_init__
        assert first.axis == pytest.approx((0.0, 0.0, 1.0))
        assert second.label == "pipe_b"
        assert second.axis == pytest.approx((1.0, 0.0, 0.0))

    def test_build_integrator_config_carries_beamline_geometry(self):
        config = cli._build_integrator_config(
            {
                "steps": "12",
                "time_step": "0.25",
                "wall_position": "1.5",
                "aperture_radius": "0.002",
                "simulation_type": "switching-wall",
                "chrono_mode": "fast",
                "startup_mode": "approximate-back-history",
                "image_subcharge_count": "16",
                "use_image_weighting": "no",
                "z_cutoff_mode": "relative",
                "beamline_geometry": {
                    "enabled": True,
                    "occluders": [
                        {
                            "axis": [0.0, 0.0, 1.0],
                            "center_mm": [0.0, 0.0, 0.0],
                            "radius_mm": 12.0,
                            "length_mm": 800.0,
                            "label": "pipe",
                        }
                    ],
                },
            }
        )
        assert config.beamline_geometry.enabled is True
        assert len(config.beamline_geometry.occluders) == 1
        assert config.beamline_geometry.occluders[0].label == "pipe"

    def test_build_beamline_geometry_config_rejects_non_object(self):
        with pytest.raises(cli.SimulationConfigError, match="must be an object"):
            cli._build_beamline_geometry_config([1, 2, 3])

    def test_build_beamline_geometry_config_rejects_non_object_occluder(self):
        with pytest.raises(cli.SimulationConfigError, match="each occluder"):
            cli._build_beamline_geometry_config(
                {"enabled": True, "occluders": ["not_an_object"]}
            )
