"""Edge-case coverage for optimization helper modules."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.types import SimulationType
from optimization.metrics import (
    compute_delta_energy_components,
    compute_energy_at_position,
    compute_energy_gain_near_aperture,
    compute_trajectory_metrics,
    detect_transverse_deflection,
)
from optimization.penalties import compute_soft_penalty
import optimization.run_control_helpers as run_control_helpers
from optimization.run_control_helpers import (
    SweepParameterValidationInput,
    build_extreme_parameter_warning,
    validate_optimization_inputs,
)
from optimization.plugin_results_helpers import (
    build_summary_heatmap_grid,
    build_trajectory_plot_data,
    summarize_optimization_top_results,
    summarize_saved_results,
)
from optimization.sweep_helpers import (
    build_config_parameter_grids,
    build_parameter_grids,
    calculate_energy_from_pz,
)


def _state(
    *,
    gamma: float,
    z: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    dead: bool = False,
) -> dict[str, np.ndarray]:
    return {
        "gamma": np.array([gamma], dtype=float),
        "z": np.array([z], dtype=float),
        "x": np.array([x], dtype=float),
        "y": np.array([y], dtype=float),
        "_dead_particles": np.array([dead]),
    }


class _MockVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def test_metrics_delta_energy_components_tracks_total_and_longitudinal_changes():
    delta_total, delta_z = compute_delta_energy_components(
        {
            "gamma": np.array([10.0, 12.5, 13.0]),
            "Pz": np.array([1000.0, 1150.0, 1300.0]),
        },
        rest_energy_gev=0.5,
    )

    assert delta_total == pytest.approx((13.0 - 10.0) * 0.5)
    assert delta_z == pytest.approx((1300.0 - 1000.0) * 1e-3)


def test_energy_gain_near_aperture_returns_best_match_and_fallback():
    trajectory = [
        _state(gamma=100.0, z=0.0),
        _state(gamma=105.0, z=42.0),
        _state(gamma=110.0, z=49.0),
        _state(gamma=102.0, z=120.0),
    ]

    gain, z_at_max, step = compute_energy_gain_near_aperture(
        trajectory,
        initial_gamma=100.0,
        rest_energy_mev=2.0,
        aperture_z=50.0,
        search_range_mm=10.0,
    )

    assert gain == pytest.approx((110.0 - 100.0) * 2e-3)
    assert z_at_max == pytest.approx(49.0)
    assert step == 2

    gain, z_at_max, step = compute_energy_gain_near_aperture(
        trajectory,
        initial_gamma=100.0,
        rest_energy_mev=2.0,
        aperture_z=500.0,
        search_range_mm=1.0,
    )

    assert gain == 0.0
    assert z_at_max == pytest.approx(500.0)
    assert step == -1


def test_detect_transverse_deflection_handles_dead_initial_and_skipped_steps():
    assert detect_transverse_deflection([_state(gamma=100.0)]) == []

    all_dead = [
        _state(gamma=100.0, dead=True),
        _state(gamma=120.0),
        _state(gamma=90.0),
    ]
    assert detect_transverse_deflection(all_dead) == []
    assert detect_transverse_deflection(all_dead, initial_gamma=100.0) == []

    skipped = [
        _state(gamma=100.0),
        _state(gamma=150.0, dead=True),
        _state(gamma=130.0),
        _state(gamma=110.0),
    ]
    events = detect_transverse_deflection(
        skipped,
        energy_jump_threshold=0.2,
        energy_dip_threshold=0.1,
        initial_gamma=100.0,
    )

    assert [event[1] for event in events] == ["jump", "dip", "deflection"]


def test_compute_trajectory_metrics_handles_empty_and_fully_dead_final_states():
    initial_state = {
        "gamma": np.array([100.0]),
        "x": np.array([0.0]),
        "y": np.array([0.0]),
    }

    empty_metrics = compute_trajectory_metrics([], initial_state, 2.0)
    assert empty_metrics["final_energy_gain_gev"] == 0.0
    assert empty_metrics["near_aperture_max_gev"] == 0.0
    assert empty_metrics["near_aperture_z_mm"] == 0.0
    assert empty_metrics["final_gamma_mean"] == 1.0

    trajectory = [
        _state(gamma=100.0, z=0.0, x=0.0, y=0.0),
        _state(gamma=110.0, z=45.0, x=0.5, y=0.5),
        _state(gamma=999.0, z=50.0, x=9.0, y=9.0, dead=True),
    ]
    metrics = compute_trajectory_metrics(
        trajectory,
        initial_state,
        2.0,
        aperture_z=50.0,
    )

    assert metrics["final_energy_gain_gev"] == 0.0
    assert metrics["near_aperture_max_gev"] == pytest.approx((110.0 - 100.0) * 2e-3)
    assert metrics["near_aperture_z_mm"] == pytest.approx(45.0)
    assert metrics["max_transverse_displacement_mm"] == pytest.approx(
        np.sqrt(0.5**2 + 0.5**2)
    )
    assert metrics["num_particles_dead"] == 1
    assert metrics["final_gamma_mean"] == 1.0


def test_compute_energy_at_position_skips_dead_matches():
    trajectory = [
        _state(gamma=150.0, z=10.0, dead=True),
        _state(gamma=125.0, z=10.4),
    ]

    energy = compute_energy_at_position(
        trajectory,
        target_z=10.0,
        initial_gamma=100.0,
        rest_energy_mev=4.0,
        tolerance_mm=1.0,
    )

    assert energy == pytest.approx((125.0 - 100.0) * 4e-3)


def test_compute_soft_penalty_is_zero_for_safe_region_and_positive_for_risky_cases():
    electron = SimpleNamespace(m_particle=0.00054857990907)
    proton = SimpleNamespace(m_particle=1.0)

    assert (
        compute_soft_penalty(
            electron,
            aperture_radius=0.2,
            macroparticle_charge_multiplier=10.0,
            initial_energy_gev=1.0,
        )
        == 0.0
    )

    assert compute_soft_penalty(
        electron,
        aperture_radius=0.005,
        macroparticle_charge_multiplier=1600.0,
        initial_energy_gev=500.0,
    ) > 0.0

    assert (
        compute_soft_penalty(
            proton,
            aperture_radius=0.05,
            macroparticle_charge_multiplier=1600.0,
            initial_energy_gev=5000.0,
        )
        == 0.0
    )


def test_run_control_helpers_expose_only_supported_public_helpers():
    assert run_control_helpers.__all__ == [
        "SweepParameterValidationInput",
        "build_extreme_parameter_warning",
        "validate_optimization_inputs",
    ]


def _validation_params(**overrides):
    params = {
        "simulation_type": SimulationType.CONDUCTING_WALL,
        "aperture_min": "0.01",
        "aperture_max": "0.02",
        "aperture_points": "2",
        "energy_min": "1.0",
        "energy_max": "2.0",
        "energy_points": "2",
        "mode": "blind_sweep",
        "offset_fractions": "0.0",
        "start_z": "0.0",
        "wall_z": "100.0",
        "steps": "100",
        "auto_steps_distance": "0.0",
        "sweep_parameters": [
            SweepParameterValidationInput(
                name="rider_m_particle",
                swept=False,
                min_value="1.0",
                max_value="2.0",
                points="2",
                fixed_value="1.0",
            ),
            SweepParameterValidationInput(
                name="rider_pcount",
                swept=False,
                min_value="1",
                max_value="2",
                points="2",
                fixed_value="1",
            ),
            SweepParameterValidationInput(
                name="rider_stripped_ions",
                swept=False,
                min_value="1.0",
                max_value="2.0",
                points="2",
                fixed_value="1.0",
            ),
        ],
    }
    params.update(overrides)
    return params


def test_validate_optimization_inputs_accepts_enum_backed_b2b_single_rider_energy():
    error = validate_optimization_inputs(
        **_validation_params(
            simulation_type=SimulationType.BUNCH_TO_BUNCH,
            aperture_min="-1.0",
            aperture_max="-1.0",
            aperture_points="1",
            energy_min="1.0",
            energy_max="1.0",
            energy_points="1",
            mode="optimization",
        )
    )

    assert error is None


def test_validate_optimization_inputs_allows_blind_sweep_with_swept_subparameter():
    swept_param = SweepParameterValidationInput(
        name="rider_transv_mom",
        swept=True,
        min_value="0.0",
        max_value="1.0",
        points="2",
        fixed_value="0.0",
    )

    error = validate_optimization_inputs(
        **_validation_params(
            energy_min="1.0",
            energy_max="2.0",
            energy_points="1",
            sweep_parameters=[swept_param],
        )
    )

    assert error is None


def test_validate_optimization_inputs_reports_fixed_particle_constraints():
    error = validate_optimization_inputs(
        **_validation_params(
            sweep_parameters=[
                SweepParameterValidationInput(
                    name="rider_m_particle",
                    swept=False,
                    min_value="1.0",
                    max_value="2.0",
                    points="2",
                    fixed_value="0.0",
                )
            ]
        )
    )

    assert error == "rider_m_particle: Particle mass must be positive"


def test_build_extreme_parameter_warning_returns_none_for_safe_config():
    config = SimpleNamespace(
        aperture_range=(0.01, 0.02),
        energy_range=(1.0, 5.0),
        m_particle=1.0,
        simulation_type=SimulationType.CONDUCTING_WALL,
        auto_steps=True,
        timestep=1e-7,
    )

    assert build_extreme_parameter_warning(config) is None


def test_build_extreme_parameter_warning_uses_b2b_kinetic_gamma_convention():
    config = SimpleNamespace(
        aperture_range=(1e-7, 1e-6),
        energy_range=(1.0, 0.0001),
        m_particle=1.0,
        simulation_type=SimulationType.BUNCH_TO_BUNCH,
        auto_steps=False,
        timestep=1e-3,
    )

    warning = build_extreme_parameter_warning(config)

    assert warning is not None
    assert "Aperture < 1 μm" in warning
    assert "Fixed timestep may be too large" in warning
    assert "Do you want to proceed anyway?" in warning


def test_build_extreme_parameter_warning_keeps_proton_thresholds_mass_aware():
    proton = SimpleNamespace(
        aperture_range=(0.1, 0.2),
        energy_range=(1000.0, 5000.0),
        m_particle=1.007276466621,
        simulation_type=SimulationType.CONDUCTING_WALL,
        auto_steps=True,
        timestep=1e-7,
    )
    electron = SimpleNamespace(
        aperture_range=(0.1, 0.2),
        energy_range=(1000.0, 5000.0),
        m_particle=0.00054857990907,
        simulation_type=SimulationType.CONDUCTING_WALL,
        auto_steps=True,
        timestep=1e-7,
    )

    assert build_extreme_parameter_warning(proton) is None
    electron_warning = build_extreme_parameter_warning(electron)
    assert electron_warning is not None
    assert "threshold for electrons" in electron_warning


def test_plugin_result_helpers_cover_single_dimension_and_sparse_payloads():
    assert (
        build_summary_heatmap_grid(
            [
                {
                    "parameters": {
                        "aperture_radius": 0.1,
                        "particle_energy_gev": 1.0,
                    },
                    "metrics": {"rider_delta_e_mev": 1.0},
                }
            ]
        )
        is None
    )

    plot_data = build_trajectory_plot_data(
        [
            {
                "run_number": 1,
                "parameters": {"aperture_radius": 0.1, "particle_energy_gev": 2.0},
                "metrics": {"rider_delta_e_mev": 5.0, "rider_gamma_initial": 2.0},
                "trajectory": {"z": [], "r": []},
            },
            {
                "run_number": 2,
                "parameters": {"aperture_radius": 0.2, "particle_energy_gev": 3.0},
                "metrics": {"rider_delta_e_mev": 4.0, "rider_gamma_initial": 3.0},
                "trajectory": {"z": [7.0], "r": [0.1]},
            },
            {
                "run_number": 3,
                "parameters": {"aperture_radius": 0.3, "particle_energy_gev": 4.0},
                "metrics": {"rider_delta_e_mev": 6.0, "rider_gamma_initial": 4.0},
                "trajectory": {"z": [5.0, 5.0], "r": [0.2, 0.3]},
            },
        ],
        m_particle_amu=1.0,
        amu_to_mev=931.494,
    )

    assert [series["run_num"] for series in plot_data["series"]] == [2, 3]
    assert plot_data["series"][0]["energy_delta"].tolist() == [0.0]
    assert plot_data["series"][1]["energy_delta"].tolist() == [0.0, 0.0]

    summary = summarize_saved_results(
        {
            "kind": "sweep",
            "results": [
                {
                    "run_number": 5,
                    "parameters": {
                        "aperture_radius": 0.2,
                        "particle_energy_gev": 9.0,
                    },
                    "metrics": {"rider_delta_e_mev": 1.2},
                    "sweep_info": {"config_name": "demo_sweep.json"},
                }
            ],
            "results_with_trajectories": [],
        }
    )
    assert summary["config_name"] == "demo_sweep.json"


def test_summarize_optimization_top_results_handles_empty_failed_and_minimize_modes():
    assert summarize_optimization_top_results({"all_evaluations": []}) == []
    assert (
        summarize_optimization_top_results(
            {
                "all_evaluations": [
                    {"evaluation": 1, "failed": True},
                    {"evaluation": 2, "halted_early": True},
                ]
            }
        )
        == []
    )

    top_results = summarize_optimization_top_results(
        {
            "objective": "min_beam_size",
            "all_evaluations": [
                {
                    "evaluation": 1,
                    "objective_value": 2.5,
                    "parameters": {"a": 1},
                    "metrics": {"percent_delta_e": 4.0},
                },
                {
                    "evaluation": 2,
                    "parameters": {"a": 2},
                    "metrics": {"delta_e_mev": 7.0},
                },
                {
                    "evaluation": 3,
                    "objective_value": 1.5,
                    "parameters": {"a": 3},
                    "metrics": {"delta_e_mev": 5.0},
                },
            ],
        },
        limit=3,
    )

    assert [entry["evaluation"] for entry in top_results] == [3, 1, 2]
    assert top_results[0]["metric_value"] == pytest.approx(1.5)
    assert top_results[1]["percent_energy_gain"] == pytest.approx(4.0)
    assert top_results[2]["metric_value"] is None
    assert top_results[2]["delta_e_mev"] == pytest.approx(7.0)


def test_sweep_helpers_cover_zero_pz_and_disabled_non_driver_controls():
    assert calculate_energy_from_pz(0.0, 1.0) == 0.0

    config = SimpleNamespace(
        simulation_type=SimulationType.CONDUCTING_WALL,
        aperture_range=(0.1, 0.1),
        aperture_points=1,
        aperture_log_scale=False,
        energy_range=(2.0, 2.0),
        energy_points=1,
        energy_log_scale=False,
        transverse_offset_fractions=[],
        starting_z_positions=[1.0],
        wall_z_range=None,
        wall_z_points=1,
    )
    sweep_params = {
        "rider_transv_mom": {
            "sweep_var": _MockVar(False),
            "min_var": _MockVar(1.0),
            "max_var": _MockVar(2.0),
            "points_var": _MockVar(3),
            "log_var": _MockVar(False),
        }
    }

    grids = build_parameter_grids(config, sweep_params)

    assert "rider_transv_mom" not in grids
    assert grids["transverse_offset_fraction"] == [0.0]


def test_build_config_parameter_grids_accepts_string_b2b_mode():
    config = SimpleNamespace(
        simulation_type="BUNCH_TO_BUNCH",
        aperture_range=(0.01, 0.02),
        aperture_points=3,
        aperture_log_scale=False,
        energy_range=(1.0, 1.0),
        energy_points=1,
        energy_log_scale=False,
        starting_z_positions=[0.0],
        starting_z_range=None,
        starting_z_points=1,
        transverse_offset_fractions=[0.0],
        wall_z=1000.0,
        wall_z_range=None,
        wall_z_points=1,
        driver_energy_range=(10.0, 20.0),
        driver_energy_points=2,
        driver_energy_log_scale=False,
    )

    grids = build_config_parameter_grids(config)

    assert "aperture" not in grids
    assert grids["energy"] == [1.0]
    assert grids["driver_energy_gev"] == pytest.approx([10.0, 20.0])
