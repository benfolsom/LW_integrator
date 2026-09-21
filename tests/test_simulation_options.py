"""Tests for SimulationOptions serialization and fallback behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.external_fields import electric_field_v_per_m_to_native
from core.types import SimulationType, StartupMode
from lw_integrator.testbed_runner import (
    CORE_PARAM_DEFAULTS,
    DEFAULT_DRIVER_PARAMS,
    DEFAULT_RIDER_PARAMS,
    STARTUP_MODE_OPTIONS,
    SimulationOptions,
    build_driver_train_config,
    build_external_field_config,
    build_pseudo_grid_config,
    build_self_consistency_config,
    build_startup_mode_enum,
)


def test_simulation_options_roundtrip_preserves_gamma_reconciliation_fields(
    tmp_path: Path,
):
    options = SimulationOptions(
        steps=321,
        seed=54321,
        simulation_type=SimulationType.CONDUCTING_WALL,
        output_dir=tmp_path / "outputs",
        config_dir=tmp_path / "configs",
        config_name="custom.json",
        self_consistency_convergence_mode="variable_geometry",
        self_consistency_gamma_reconciliation_method="FIXED_WEIGHTED",
        self_consistency_gamma_reconciliation_low_beta_threshold=0.85,
        self_consistency_gamma_reconciliation_high_beta_threshold=0.995,
        self_consistency_gamma_reconciliation_low_beta_weight=0.9,
        self_consistency_gamma_reconciliation_high_beta_weight=0.1,
        self_consistency_gamma_reconciliation_mid_beta_weight=0.6,
        self_consistency_gamma_reconciliation_fixed_weight=0.7,
        energy_monitor_halt_on_jump=True,
        adaptive_timestep_min_factor=1e-5,
        space_charge_enabled=True,
        space_charge_retarded=True,
        space_charge_softening_mm=0.004,
        space_charge_bunch_sigma_mm=0.025,
        space_charge_min_retarded_steps=5,
        external_field_enabled=True,
        external_electric_field_v_per_m=(0.0, 0.0, -1.5e9),
        external_magnetic_field_native=(0.0, 3.0, 0.0),
        external_field_z_min=-0.2,
        external_field_z_max=0.2,
        external_field_t_min=1.0e-6,
        external_field_t_max=2.0e-6,
        radiation_reaction_mode="power_matched_damping",
        macroparticle_dynamics_mode="macro_inertia",
        pseudo_grid_enabled=True,
        pseudo_grid_active_rider_count=6,
        pseudo_grid_active_driver_count=7,
        pseudo_grid_field_rider_count=24,
        pseudo_grid_field_driver_count=25,
        pseudo_grid_field_deposition_neighbor_count=6,
        pseudo_grid_passive_neighbor_count=3,
        pseudo_grid_coverage_strategy="farthest_point",
        pseudo_grid_coverage_space="phase_space",
        pseudo_grid_active_selection_mode="slow_rotating_live",
        pseudo_grid_passive_update_mode="external_interbunch",
        pseudo_grid_active_rotation_interval=12,
        pseudo_grid_active_rotation_fraction=0.5,
        pseudo_grid_passive_remap_mode="none",
        pseudo_grid_passive_remap_warning_sigma=0.25,
        pseudo_grid_passive_remap_trigger_sigma=0.75,
        pseudo_grid_pair_reuse_window=25,
        pseudo_grid_source_weighting_mode="nearest",
        pseudo_grid_loss_tracking_enabled=False,
        pseudo_grid_causal_history_pruning_enabled=True,
        pseudo_grid_causal_history_safety_margin_steps=5,
        cavity_exit_enabled=True,
        cavity_exit_mode="rider_exit_with_driver_tail",
        cavity_exit_length_mm=123.0,
        driver_train_enabled=True,
        driver_train_bunch_count=3,
        driver_train_z_spacing_mm=2997.92458,
        driver_train_z_offsets_mm=(0.0, 100.0, 250.0),
        driver_train_prehistory_steps=12,
        driver_train_preserve_prehistory_in_output=True,
        log_file_path="custom.log",
    )

    payload = options.to_dict()
    loaded = SimulationOptions.from_dict(payload)

    assert payload["simulation_type"] == "CONDUCTING_WALL"
    assert payload["output_dir"] == str(tmp_path / "outputs")
    assert loaded.steps == 321
    assert loaded.seed == 54321
    assert loaded.simulation_type == SimulationType.CONDUCTING_WALL
    assert loaded.output_dir == tmp_path / "outputs"
    assert loaded.config_dir == tmp_path / "configs"
    assert loaded.config_name == "custom.json"
    assert loaded.self_consistency_convergence_mode == "variable_geometry"
    assert loaded.self_consistency_gamma_reconciliation_method == "FIXED_WEIGHTED"
    assert (
        loaded.self_consistency_gamma_reconciliation_low_beta_threshold
        == pytest.approx(0.85)
    )
    assert (
        loaded.self_consistency_gamma_reconciliation_high_beta_threshold
        == pytest.approx(0.995)
    )
    assert (
        loaded.self_consistency_gamma_reconciliation_low_beta_weight
        == pytest.approx(0.9)
    )
    assert (
        loaded.self_consistency_gamma_reconciliation_high_beta_weight
        == pytest.approx(0.1)
    )
    assert (
        loaded.self_consistency_gamma_reconciliation_mid_beta_weight
        == pytest.approx(0.6)
    )
    assert loaded.self_consistency_gamma_reconciliation_fixed_weight == pytest.approx(
        0.7
    )
    assert loaded.energy_monitor_halt_on_jump is True
    assert loaded.adaptive_timestep_min_factor == pytest.approx(1e-5)
    assert loaded.space_charge_enabled is True
    assert loaded.space_charge_retarded is True
    assert loaded.space_charge_softening_mm == pytest.approx(0.004)
    assert loaded.space_charge_bunch_sigma_mm == pytest.approx(0.025)
    assert loaded.space_charge_min_retarded_steps == 5
    assert loaded.external_field_enabled is True
    assert loaded.external_electric_field_v_per_m == pytest.approx((0.0, 0.0, -1.5e9))
    assert loaded.external_magnetic_field_native == pytest.approx((0.0, 3.0, 0.0))
    assert loaded.external_field_z_min == pytest.approx(-0.2)
    assert loaded.external_field_z_max == pytest.approx(0.2)
    assert loaded.external_field_t_min == pytest.approx(1.0e-6)
    assert loaded.external_field_t_max == pytest.approx(2.0e-6)
    assert payload["radiation_reaction_mode"] == "power_matched_damping"
    assert payload["macroparticle_dynamics_mode"] == "macro_inertia"
    assert loaded.macroparticle_dynamics_mode == "macro_inertia"
    assert payload["pseudo_grid"] == {
        "enabled": True,
        "active_rider_count": 6,
        "active_driver_count": 7,
        "field_rider_count": 24,
        "field_driver_count": 25,
        "field_deposition_neighbor_count": 6,
        "space_charge_near_neighbor_count": 8,
        "passive_neighbor_count": 3,
        "coverage_strategy": "farthest_point",
        "coverage_space": "phase_space",
        "active_selection_mode": "slow_rotating_live",
        "passive_update_mode": "external_interbunch",
        "active_rotation_interval": 12,
        "active_rotation_fraction": 0.5,
        "passive_remap_mode": "none",
        "passive_remap_warning_sigma": 0.25,
        "passive_remap_trigger_sigma": 0.75,
        "pair_reuse_window": 25,
        "source_weighting_mode": "nearest",
        "loss_tracking_enabled": False,
        "numerical_failure_tolerance_fraction": 0.15,
        "causal_history_pruning_enabled": True,
        "causal_history_safety_margin_steps": 5,
    }
    assert loaded.radiation_reaction_mode == "power_matched_damping"
    assert loaded.pseudo_grid_enabled is True
    assert loaded.pseudo_grid_active_rider_count == 6
    assert loaded.pseudo_grid_active_driver_count == 7
    assert loaded.pseudo_grid_field_rider_count == 24
    assert loaded.pseudo_grid_field_driver_count == 25
    assert loaded.pseudo_grid_field_deposition_neighbor_count == 6
    assert loaded.pseudo_grid_passive_neighbor_count == 3
    assert loaded.pseudo_grid_coverage_strategy == "farthest_point"
    assert loaded.pseudo_grid_coverage_space == "phase_space"
    assert loaded.pseudo_grid_active_selection_mode == "slow_rotating_live"
    assert loaded.pseudo_grid_passive_update_mode == "external_interbunch"
    assert loaded.pseudo_grid_active_rotation_interval == 12
    assert loaded.pseudo_grid_active_rotation_fraction == pytest.approx(0.5)
    assert loaded.pseudo_grid_passive_remap_mode == "none"
    assert loaded.pseudo_grid_passive_remap_warning_sigma == pytest.approx(0.25)
    assert loaded.pseudo_grid_passive_remap_trigger_sigma == pytest.approx(0.75)
    assert loaded.pseudo_grid_pair_reuse_window == 25
    assert loaded.pseudo_grid_source_weighting_mode == "nearest"
    assert loaded.pseudo_grid_loss_tracking_enabled is False
    assert loaded.pseudo_grid_causal_history_pruning_enabled is True
    assert loaded.pseudo_grid_causal_history_safety_margin_steps == 5
    assert payload["cavity_exit"]["enabled"] is True
    assert payload["cavity_exit"]["mode"] == "rider_exit_with_driver_tail"
    assert payload["cavity_exit"]["cavity_length_mm"] == pytest.approx(123.0)
    assert loaded.cavity_exit_enabled is True
    assert loaded.cavity_exit_mode == "rider_exit_with_driver_tail"
    assert loaded.cavity_exit_length_mm == pytest.approx(123.0)
    assert payload["driver_train"] == {
        "enabled": True,
        "bunch_count": 3,
        "z_spacing_mm": 2997.92458,
        "z_offsets_mm": [0.0, 100.0, 250.0],
        "prehistory_steps": 12,
        "preserve_prehistory_in_output": True,
    }
    assert loaded.driver_train_enabled is True
    assert loaded.driver_train_bunch_count == 3
    assert loaded.driver_train_z_spacing_mm == pytest.approx(2997.92458)
    assert loaded.driver_train_z_offsets_mm == pytest.approx((0.0, 100.0, 250.0))
    assert loaded.driver_train_prehistory_steps == 12
    assert loaded.driver_train_preserve_prehistory_in_output is True
    assert loaded.log_file_path == "custom.log"


def test_simulation_options_roundtrip_preserves_checkpoint_controls(
    tmp_path: Path,
) -> None:
    options = SimulationOptions(
        checkpoint_enabled=True,
        checkpoint_directory=tmp_path / "capture.checkpoint",
        checkpoint_interval_steps=250,
        checkpoint_interval_seconds=120.0,
    )

    restored = SimulationOptions.from_dict(options.to_dict())

    assert restored.checkpoint_enabled is True
    assert restored.checkpoint_directory == tmp_path / "capture.checkpoint"
    assert restored.checkpoint_resume_from is None
    assert restored.checkpoint_interval_steps == 250
    assert restored.checkpoint_interval_seconds == pytest.approx(120.0)


def test_checkpoint_resume_path_enables_checkpointing(tmp_path: Path) -> None:
    restored = SimulationOptions.from_dict(
        {
            "checkpoint": {
                "resume_from": str(tmp_path / "capture.checkpoint"),
                "interval_steps": 100,
                "interval_seconds": 30.0,
            }
        }
    )

    assert restored.checkpoint_enabled is True
    assert restored.checkpoint_resume_from == tmp_path / "capture.checkpoint"


def test_adaptive_pair_return_roundtrip_preserves_production_controls() -> None:
    options = SimulationOptions(
        adaptive_pair_return_enabled=True,
        adaptive_pair_target_lab_time_ns=1.25,
        adaptive_pair_tolerance_scale=0.5,
        adaptive_pair_minimum_step_factor=1.0 / 128.0,
        adaptive_pair_maximum_step_factor=32.0,
        adaptive_pair_public_sample_interval_ns=0.025,
        adaptive_pair_shared_time_absolute_tolerance_ns=2.0e-20,
        adaptive_pair_shared_time_relative_tolerance=3.0e-13,
        adaptive_pair_maximum_attempts=1234,
        adaptive_pair_maximum_accepted_slabs=567,
    )

    restored = SimulationOptions.from_dict(options.to_dict())

    assert restored.adaptive_pair_return_enabled is True
    assert restored.adaptive_pair_target_lab_time_ns == pytest.approx(1.25)
    assert restored.adaptive_pair_tolerance_scale == pytest.approx(0.5)
    assert restored.adaptive_pair_minimum_step_factor == pytest.approx(1.0 / 128.0)
    assert restored.adaptive_pair_maximum_step_factor == pytest.approx(32.0)
    assert restored.adaptive_pair_public_sample_interval_ns == pytest.approx(0.025)
    assert restored.adaptive_pair_shared_time_absolute_tolerance_ns == pytest.approx(
        2.0e-20
    )
    assert restored.adaptive_pair_shared_time_relative_tolerance == pytest.approx(
        3.0e-13
    )
    assert restored.adaptive_pair_maximum_attempts == 1234
    assert restored.adaptive_pair_maximum_accepted_slabs == 567


def test_causal_c5_dipole_history_roundtrips_in_nested_config() -> None:
    options = SimulationOptions(
        magnetic_dipole_source_model="covariant_retarded_point",
        magnetic_dipole_source_history_model="causal_c5",
    )

    payload = options.to_dict()
    restored = SimulationOptions.from_dict(payload)

    assert payload["magnetic_dipole"]["source"]["history_model"] == "causal_c5"
    assert restored.magnetic_dipole_source_history_model == "causal_c5"


def test_flat_checkpoint_fields_remain_loadable(tmp_path: Path) -> None:
    restored = SimulationOptions.from_dict(
        {
            "checkpoint_enabled": True,
            "checkpoint_directory": str(tmp_path / "legacy.checkpoint"),
            "checkpoint_interval_steps": 75,
            "checkpoint_interval_seconds": 15.0,
        }
    )

    assert restored.checkpoint_enabled is True
    assert restored.checkpoint_directory == tmp_path / "legacy.checkpoint"
    assert restored.checkpoint_interval_steps == 75
    assert restored.checkpoint_interval_seconds == pytest.approx(15.0)


def test_simulation_options_roundtrip_preserves_manual_particle_config_and_3d_payloads():
    options = SimulationOptions(
        simulation_type=SimulationType.BUNCH_TO_BUNCH,
        manual_particle_config_enabled=True,
        rider_params={
            "kinetic_energy_mev": 12.0,
            "mass_amu": 1.007276466621,
            "charge_sign": -1.0,
            "particle_count": 3,
            "starting_position_mm": [1.0, 2.0, 3.0],
            "momentum_axis": [1.0, 0.0, 0.0],
            "longitudinal_span_mm": 4.0,
        },
        driver_params={
            "kinetic_energy_mev": 15.0,
            "mass_amu": 1.007276466621,
            "charge_sign": 1.0,
            "particle_count": 3,
            "starting_position_mm": [4.0, 5.0, 6.0],
            "momentum_axis": [0.0, -1.0, 0.0],
            "transverse_distance_mm": 0.1,
        },
    )

    payload = options.to_dict()
    loaded = SimulationOptions.from_dict(payload)

    assert payload["manual_particle_config_enabled"] is True
    assert loaded.manual_particle_config_enabled is True
    assert loaded.rider_params["momentum_axis"] == [1.0, 0.0, 0.0]
    assert loaded.rider_params["starting_position_mm"] == [1.0, 2.0, 3.0]
    assert loaded.driver_params is not None
    assert loaded.driver_params["momentum_axis"] == [0.0, -1.0, 0.0]


def test_chrono_options_roundtrip_as_independent_fields():
    options = SimulationOptions(
        self_consistency_enabled=False,
        chrono_interpolate=True,
        chrono_tolerance=5e-4,
        chrono_matching_mode="FAST",
        chrono_high_precision=True,
        chrono_adaptive_tolerance=True,
    )

    payload = options.to_dict()
    loaded = SimulationOptions.from_dict(payload)
    config = build_self_consistency_config(loaded)

    assert payload["chrono_interpolate"] is True
    assert payload["self_consistency_chrono_interpolate"] is True
    assert loaded.chrono_tolerance == pytest.approx(5e-4)
    assert loaded.self_consistency_chrono_tolerance == pytest.approx(5e-4)
    assert config is not None
    assert config.enabled is False
    assert config.chrono_interpolate is True
    assert config.chrono_high_precision is True


def test_legacy_self_consistency_chrono_keys_populate_new_fields():
    loaded = SimulationOptions.from_dict(
        {
            "self_consistency_enabled": False,
            "self_consistency_chrono_interpolate": True,
            "self_consistency_chrono_tolerance": 2e-4,
            "self_consistency_chrono_high_precision": True,
        }
    )

    assert loaded.chrono_interpolate is True
    assert loaded.chrono_tolerance == pytest.approx(2e-4)
    assert loaded.chrono_high_precision is True


def test_build_external_field_config_converts_si_electric_field():
    options = SimulationOptions(
        external_field_enabled=True,
        external_electric_field_v_per_m=(0.0, 0.0, -1.5e9),
        external_magnetic_field_native=(0.0, 3.0, 0.0),
        external_field_z_min=-0.2,
        external_field_z_max=0.2,
    )

    config = build_external_field_config(options)

    assert config is not None
    assert config.electric_field_native[2] == pytest.approx(
        electric_field_v_per_m_to_native(-1.5e9)
    )
    assert config.magnetic_field_native == pytest.approx((0.0, 3.0, 0.0))
    assert config.z_min == pytest.approx(-0.2)
    assert config.z_max == pytest.approx(0.2)


def test_simulation_options_from_dict_accepts_legacy_mode_alias_and_int_enum():
    payload = {
        "simulation_type": int(SimulationType.SWITCHING_WALL),
        "self_consistency_convergence_mode": "mass_shell_only",
    }

    options = SimulationOptions.from_dict(payload)

    assert options.simulation_type == SimulationType.SWITCHING_WALL
    assert options.self_consistency_convergence_mode == "fixed_geometry"


def test_simulation_options_from_dict_uses_defaults_for_missing_nested_payloads():
    options = SimulationOptions.from_dict({})

    assert options.rider_params == dict(DEFAULT_RIDER_PARAMS)
    assert options.driver_params == dict(DEFAULT_DRIVER_PARAMS)
    assert options.core_params == {
        key: (float(value) if isinstance(value, (int, float)) else value)
        for key, value in CORE_PARAM_DEFAULTS.items()
    }
    assert options.radiation_reaction_mode == "medina_lad"
    assert options.pseudo_grid_enabled is False
    assert options.driver_train_enabled is False
    assert options.output_dir == Path("test_outputs/testbed_runs")


def test_simulation_options_from_dict_falls_back_on_invalid_numeric_values():
    payload = {
        "steps": "not-an-int",
        "seed": None,
        "adaptive_timestep_min_factor": "bad",
        "energy_monitor_threshold": "bad",
        "trajectory_interval": "bad",
    }

    options = SimulationOptions.from_dict(payload)

    assert options.steps == 1000
    assert options.seed == 12345
    assert options.adaptive_timestep_min_factor == pytest.approx(1e-4)
    assert options.energy_monitor_threshold == pytest.approx(2.0)
    assert options.trajectory_interval == 10


@pytest.mark.parametrize(
    "value",
    ("INERTIAL_PREHISTORY", "inertial-prehistory", "inertial_prehistory"),
)
def test_testbed_accepts_inertial_prehistory_spellings(value: str):
    assert build_startup_mode_enum(value) is StartupMode.INERTIAL_PREHISTORY


def test_inertial_prehistory_roundtrips_through_testbed_config():
    core_params = dict(CORE_PARAM_DEFAULTS)
    core_params["startup_mode"] = "INERTIAL_PREHISTORY"
    options = SimulationOptions(core_params=core_params)

    restored = SimulationOptions.from_dict(options.to_dict())

    assert "INERTIAL_PREHISTORY" in STARTUP_MODE_OPTIONS
    assert restored.core_params["startup_mode"] == "INERTIAL_PREHISTORY"


def test_build_pseudo_grid_config_reflects_simulation_options():
    options = SimulationOptions(
        pseudo_grid_enabled=True,
        pseudo_grid_active_rider_count=9,
        pseudo_grid_active_driver_count=11,
        pseudo_grid_passive_neighbor_count=2,
        pseudo_grid_pair_reuse_window=14,
        pseudo_grid_causal_history_pruning_enabled=True,
        pseudo_grid_causal_history_safety_margin_steps=4,
    )

    config = build_pseudo_grid_config(options)

    assert config.enabled is True
    assert config.active_rider_count == 9
    assert config.active_driver_count == 11
    assert config.passive_neighbor_count == 2
    assert config.pair_reuse_window == 14
    assert config.causal_history_pruning_enabled is True
    assert config.causal_history_safety_margin_steps == 4


def test_build_driver_train_config_reflects_simulation_options():
    options = SimulationOptions(
        driver_train_enabled=True,
        driver_train_bunch_count=3,
        driver_train_z_spacing_mm=100.0,
        driver_train_z_offsets_mm=(0.0, 100.0, 250.0),
        driver_train_prehistory_steps=8,
        driver_train_preserve_prehistory_in_output=True,
    )

    config = build_driver_train_config(options)

    assert config.enabled is True
    assert config.bunch_count == 3
    assert config.z_spacing_mm == pytest.approx(100.0)
    assert config.z_offsets_mm == pytest.approx((0.0, 100.0, 250.0))
    assert config.prehistory_steps == 8
    assert config.preserve_prehistory_in_output is True
