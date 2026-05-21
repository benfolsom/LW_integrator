"""Tests for pure optimization run-control helpers."""

import numpy as np

from core.types import SimulationType
from optimization.config import OptimizationConfig
from optimization.run_parameter_helpers import (
    build_optimization_evaluation_outcome,
    calculate_transverse_offset,
    collect_optimization_parameter_selection,
    is_bunch_to_bunch,
    resolve_optimization_run_parameters,
    resolve_objective_metric,
)
from optimization.sweep_helpers import calculate_starting_pz_from_energy


def test_collect_optimization_parameter_selection_preserves_existing_order():
    config = OptimizationConfig(
        simulation_type=SimulationType.BUNCH_TO_BUNCH,
        aperture_points=2,
        energy_points=3,
        transverse_momentum_range=(0.1, 0.2),
        transverse_momentum_points=2,
        timestep_range=(1e-7, 2e-7),
        timestep_points=2,
        transverse_spread_range=(1e-6, 2e-6),
        transverse_spread_points=2,
        driver_energy_range=(0.5, 0.8),
        driver_energy_points=2,
    )

    selection = collect_optimization_parameter_selection(config)

    assert selection.names == [
        "aperture_radius",
        "initial_energy_gev",
        "transverse_momentum",
        "timestep",
        "rider_transv_dist",
        "driver_energy_gev",
    ]
    assert selection.bounds == [
        config.aperture_range,
        config.energy_range,
        config.transverse_momentum_range,
        config.timestep_range,
        config.transverse_spread_range,
        config.driver_energy_range,
    ]
    assert selection.log_lines == [
        f"    Added: initial_energy_gev, range={config.energy_range}, points=3",
        (
            "    Added: transverse_momentum, "
            f"range={config.transverse_momentum_range}, points=2"
        ),
        (
            "    Added: rider_transv_dist, "
            f"range={config.transverse_spread_range}, points=2"
        ),
    ]


def test_collect_optimization_parameter_selection_skips_disabled_parameters():
    config = OptimizationConfig(
        aperture_points=1,
        energy_points=1,
        transverse_momentum_range=(0.1, 0.2),
        transverse_momentum_points=1,
        driver_energy_range=None,
        driver_energy_points=3,
    )

    selection = collect_optimization_parameter_selection(config)

    assert selection.names == []
    assert selection.bounds == []
    assert selection.log_lines == []


def test_resolve_objective_metric_keeps_historical_defaults():
    assert resolve_objective_metric("max_energy_gain") == ("max_energy_gain_gev", True)
    assert resolve_objective_metric("max_percent_energy_gain") == (
        "max_percent_energy_gain",
        True,
    )
    assert resolve_objective_metric("min_transverse_spread") == (
        "max_energy_gain_gev",
        False,
    )
    assert resolve_objective_metric(
        "max_inward_rider_radial_focusing_constrained_energy"
    ) == ("rider_radial_toward_driver_mm", True)
    assert resolve_objective_metric(
        "max_peak_inward_rider_radial_focusing_constrained_energy"
    ) == ("rider_radial_peak_inward_mm", True)
    assert resolve_objective_metric(
        "max_peak_rider_radial_rms_collapse_constrained_energy"
    ) == ("rider_radial_rms_peak_inward_mm", True)
    assert resolve_objective_metric(
        "max_rider_radial_p95_reduction_constrained_energy"
    ) == ("rider_radial_p95_mm_reduction", True)
    assert resolve_objective_metric(
        "max_rider_halo_2rms_reduction_constrained_energy"
    ) == ("rider_halo_gt_2_initial_rms_fraction_reduction", True)


def test_calculate_transverse_offset_handles_enum_and_string_modes():
    assert is_bunch_to_bunch(SimulationType.BUNCH_TO_BUNCH)
    assert is_bunch_to_bunch("BUNCH_TO_BUNCH")
    assert not is_bunch_to_bunch(SimulationType.CONDUCTING_WALL)

    assert (
        calculate_transverse_offset(
            SimulationType.BUNCH_TO_BUNCH, offset_value=0.25, aperture=0.001
        )
        == 0.25
    )
    assert (
        calculate_transverse_offset("BUNCH_TO_BUNCH", offset_value=0.25, aperture=0.001)
        == 0.25
    )
    assert (
        calculate_transverse_offset(
            SimulationType.CONDUCTING_WALL, offset_value=0.25, aperture=0.001
        )
        == 0.00025
    )


def test_resolve_optimization_run_parameters_maps_wall_mode_values():
    config = OptimizationConfig(
        simulation_type=SimulationType.CONDUCTING_WALL,
        aperture_range=(0.01, 0.02),
        energy_range=(1.0, 2.0),
        starting_z_positions=[5.0],
        transverse_offset_fractions=[0.2],
        timestep=1e-6,
        timestep_strategy="fixed",
        steps=123,
        transv_dist=0.0003,
        macroparticle_charge_multiplier=4.0,
    )

    resolved = resolve_optimization_run_parameters(
        config,
        [
            "aperture_radius",
            "initial_energy_gev",
            "transverse_offset",
            "rider_pcount",
            "macroparticle_sigma_multiplier",
        ],
        [0.5, 3.0, 0.25, 7, 2.5],
    )

    assert resolved.aperture == 0.5
    assert resolved.energy_gev == 3.0
    assert resolved.start_z == 5.0
    assert resolved.transv_offset == 0.125
    assert resolved.timestep == 1e-6
    assert resolved.steps == 123
    assert resolved.rider_pcount == 7
    assert resolved.rider_transv_dist == 0.0003
    assert resolved.macroparticle_charge_multiplier == 4.0
    assert resolved.macroparticle_sigma_multiplier == 2.5
    assert resolved.driver_params is None


def test_resolve_optimization_run_parameters_builds_bunch_driver_params_for_enum():
    config = OptimizationConfig(
        simulation_type=SimulationType.BUNCH_TO_BUNCH,
        aperture_range=(0.01, 0.02),
        energy_range=(1.0, 2.0),
        transverse_offset_fractions=[0.2],
        driver_direction="+z",
        driver_transv_offset_x=0.01,
        driver_transv_offset_y=-0.02,
    )

    resolved = resolve_optimization_run_parameters(
        config,
        [
            "transverse_offset",
            "driver_m_particle",
            "driver_charge_sign",
            "driver_pcount",
            "driver_transv_mom",
            "driver_transv_dist",
            "driver_starting_distance",
            "driver_energy_gev",
            "driver_stripped_ions",
        ],
        [0.25, 207.2, 1.0, 9, 0.03, -0.04, 800.0, 0.6, 54.0],
    )

    assert resolved.transv_offset == 0.25
    assert resolved.driver_params == {
        "m_particle": 207.2,
        "charge_sign": 1.0,
        "pcount": 9,
        "transv_mom": 0.03,
        "transv_dist": -0.04,
        "transverse_geometry": "square",
        "starting_distance": 800.0,
        "starting_Pz": calculate_starting_pz_from_energy(0.6, 207.2, negative=False),
        "stripped_ions": 54.0,
        "transv_offset_x": 0.01,
        "transv_offset_y": -0.02,
    }


def test_resolve_optimization_run_parameters_accepts_string_bunch_mode():
    config = OptimizationConfig(simulation_type=SimulationType.CONDUCTING_WALL)
    config.simulation_type = "BUNCH_TO_BUNCH"

    resolved = resolve_optimization_run_parameters(
        config,
        ["transverse_offset"],
        [0.25],
    )

    assert resolved.transv_offset == 0.25
    assert resolved.driver_params is not None


def test_build_optimization_evaluation_outcome_records_missing_metrics():
    outcome = build_optimization_evaluation_outcome(
        None,
        eval_num=3,
        param_names=["aperture"],
        values=[0.1],
        metric_name="max_percent_energy_gain",
        maximize=True,
    )

    assert outcome.fitness == np.inf
    assert outcome.record == {
        "evaluation": 3,
        "parameters": {"aperture": 0.1},
        "failed": True,
        "halted_early": False,
        "halt_reason": None,
        "objective_value": float("inf"),
    }
    assert outcome.log_lines == []


def test_build_optimization_evaluation_outcome_records_halted_runs():
    outcome = build_optimization_evaluation_outcome(
        {"metrics": {}, "halted_early": True, "halt_reason": "gamma blowup"},
        eval_num=4,
        param_names=["energy"],
        values=[5.0],
        metric_name="max_percent_energy_gain",
        maximize=True,
    )

    assert outcome.fitness == np.inf
    assert outcome.record["failed"] is False
    assert outcome.record["halted_early"] is True
    assert outcome.record["halt_reason"] == "gamma blowup"
    assert "halted early" in outcome.log_lines[0]


def test_build_optimization_evaluation_outcome_records_invalid_metrics():
    outcome = build_optimization_evaluation_outcome(
        {"metrics": {"max_percent_energy_gain": np.nan, "other": 1.0}},
        eval_num=5,
        param_names=["energy"],
        values=[5.0],
        metric_name="max_percent_energy_gain",
        maximize=True,
    )

    assert outcome.fitness == np.inf
    assert outcome.record["failed"] is True
    assert np.isnan(outcome.record["metrics"]["max_percent_energy_gain"])
    assert outcome.record["metrics"]["other"] == 1.0
    assert any("returned NaN" in line for line in outcome.log_lines)
    assert any("other: 1.0" in line for line in outcome.log_lines)


def test_build_optimization_evaluation_outcome_applies_penalty_and_saves_trajectory():
    outcome = build_optimization_evaluation_outcome(
        {
            "metrics": {"max_percent_energy_gain": 2.0},
            "trajectory": {"z": [0.0, 1.0]},
        },
        eval_num=6,
        param_names=["energy"],
        values=[5.0],
        metric_name="max_percent_energy_gain",
        maximize=True,
        penalty=0.25,
        objective_name="max_percent_energy_gain",
        save_trajectory=True,
    )

    assert outcome.fitness == -1.75
    assert outcome.record["objective_value"] == 1.75
    assert outcome.record["raw_objective_value"] == 2.0
    assert outcome.record["soft_penalty"] == 0.25
    assert outcome.record["trajectory"] == {"z": [0.0, 1.0]}
    assert any("Applied soft penalty" in line for line in outcome.log_lines)


def test_build_optimization_evaluation_outcome_minimization_penalty_direction():
    outcome = build_optimization_evaluation_outcome(
        {"metrics": {"spread": 2.0}},
        eval_num=7,
        param_names=["energy"],
        values=[5.0],
        metric_name="spread",
        maximize=False,
        penalty=0.25,
    )

    assert outcome.fitness == 2.25
    assert outcome.record["objective_value"] == 2.25


def test_energy_constrained_radial_focus_objective_accepts_valid_window():
    outcome = build_optimization_evaluation_outcome(
        {
            "metrics": {
                "rider_radial_toward_driver_mm": 0.003,
                "delta_e_mev": 10.0,
                "rider_delta_e_fraction_initial_kinetic": 0.05,
            }
        },
        eval_num=8,
        param_names=["energy"],
        values=[5.0],
        metric_name="rider_radial_toward_driver_mm",
        maximize=True,
        objective_name="max_inward_rider_radial_focusing_constrained_energy",
    )

    assert outcome.fitness == -0.003
    assert outcome.record["objective_value"] == 0.003
    assert outcome.record["failed"] is False


def test_energy_constrained_radial_focus_objective_rejects_large_gain():
    outcome = build_optimization_evaluation_outcome(
        {
            "metrics": {
                "rider_radial_toward_driver_mm": 0.003,
                "delta_e_mev": 50.0,
                "rider_delta_e_fraction_initial_kinetic": 0.25,
            }
        },
        eval_num=9,
        param_names=["energy"],
        values=[5.0],
        metric_name="rider_radial_toward_driver_mm",
        maximize=True,
        objective_name="max_inward_rider_radial_focusing_constrained_energy",
    )

    assert outcome.fitness == np.inf
    assert outcome.record["constraint_failed"] is True
    assert "20%" in outcome.record["constraint_reason"]


def test_peak_energy_constrained_radial_focus_objective_accepts_valid_window():
    outcome = build_optimization_evaluation_outcome(
        {
            "metrics": {
                "rider_radial_peak_inward_mm": 0.007,
                "delta_e_mev": 10.0,
                "rider_delta_e_fraction_initial_kinetic": 0.05,
            }
        },
        eval_num=10,
        param_names=["energy"],
        values=[5.0],
        metric_name="rider_radial_peak_inward_mm",
        maximize=True,
        objective_name="max_peak_inward_rider_radial_focusing_constrained_energy",
    )

    assert outcome.fitness == -0.007
    assert outcome.record["objective_value"] == 0.007
    assert outcome.record["failed"] is False


def test_peak_energy_constrained_radial_focus_objective_rejects_nonpositive_peak():
    outcome = build_optimization_evaluation_outcome(
        {
            "metrics": {
                "rider_radial_peak_inward_mm": 0.0,
                "delta_e_mev": 10.0,
                "rider_delta_e_fraction_initial_kinetic": 0.05,
            }
        },
        eval_num=11,
        param_names=["energy"],
        values=[5.0],
        metric_name="rider_radial_peak_inward_mm",
        maximize=True,
        objective_name="max_peak_inward_rider_radial_focusing_constrained_energy",
    )

    assert outcome.fitness == np.inf
    assert outcome.record["constraint_failed"] is True
    assert "peak inward radial focusing" in outcome.record["constraint_reason"]


def test_rms_peak_energy_constrained_radial_focus_objective_accepts_valid_window():
    outcome = build_optimization_evaluation_outcome(
        {
            "metrics": {
                "rider_radial_rms_peak_inward_mm": 0.011,
                "delta_e_mev": 10.0,
                "rider_delta_e_fraction_initial_kinetic": 0.05,
            }
        },
        eval_num=12,
        param_names=["energy"],
        values=[5.0],
        metric_name="rider_radial_rms_peak_inward_mm",
        maximize=True,
        objective_name="max_peak_rider_radial_rms_collapse_constrained_energy",
    )

    assert outcome.fitness == -0.011
    assert outcome.record["objective_value"] == 0.011
    assert outcome.record["failed"] is False
