"""Tests for optimization module.

Basic tests to verify optimization functionality works correctly.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.types import SimulationType
from optimization.config import OptimizationConfig
from optimization.metrics import (
    compute_energy_at_position,
    compute_max_energy_gain,
    compute_relative_energy_gain,
    detect_transverse_deflection,
)
from optimization.plugin_config_helpers import (
    apply_sweep_parameter_overrides,
    parse_float_list,
    parse_float_range,
    parse_offset_pair,
)
from optimization.plugin_persistence_helpers import (
    apply_persisted_config_overrides,
    build_saved_config_payload,
    metrics_export_settings_from_data,
    resolve_loaded_sweep_state,
)
from optimization.plugin_results_helpers import (
    build_summary_heatmap_grid,
    build_trajectory_plot_data,
    collect_summary_plot_data,
    convert_legacy_trajectory_data,
    parse_results_payload,
    summarize_result_row,
    summarize_optimization_top_results,
    summarize_saved_results,
)
from optimization.parameter_sweep import ParameterGrid, create_energy_aperture_grid
from optimization.results_mixins import OptimizationResultsMixin
from optimization.sweep_helpers import (
    build_parameter_grids,
    calculate_energy_from_pz,
    calculate_starting_pz_from_energy,
    generate_parameter_range,
)


def create_mock_trajectory(n_steps=10, gamma_values=None):
    """Create a mock trajectory for testing.

    Parameters
    ----------
    n_steps : int
        Number of steps in trajectory
    gamma_values : array-like, optional
        Gamma values for each step. If None, uses linear increase.

    Returns
    -------
    list
        Mock trajectory
    """
    if gamma_values is None:
        gamma_values = np.linspace(1000, 1010, n_steps)

    trajectory = []
    for i, gamma in enumerate(gamma_values):
        state = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([i * 10.0]),  # 10 mm per step
            "t": np.array([i * 0.01]),
            "Px": np.array([0.0]),
            "Py": np.array([0.0]),
            "Pz": np.array([gamma * 0.99]),  # Approximate ultra-relativistic
            "Pt": np.array([gamma]),
            "gamma": np.array([gamma]),
            "bx": np.array([0.0]),
            "by": np.array([0.0]),
            "bz": np.array([0.99]),
            "m": np.array([1.0]),
            "charge": np.array([-1.0]),
        }
        trajectory.append(state)

    return trajectory


class TestMetrics:
    """Test metric computation functions."""

    def test_compute_max_energy_gain(self):
        """Test maximum energy gain computation."""
        trajectory = create_mock_trajectory(
            n_steps=5, gamma_values=[1000, 1005, 1010, 1008, 1012]
        )
        initial_gamma = 1000.0
        rest_energy_mev = 0.511

        max_gain = compute_max_energy_gain(trajectory, initial_gamma, rest_energy_mev)

        # Max gamma is 1012, so delta = 12
        # Energy gain = 12 * 0.511 MeV = 12 * 0.000511 GeV = 0.006132 GeV
        expected = 12 * 0.511 * 1e-3
        assert abs(max_gain - expected) < 1e-6

    def test_compute_max_energy_gain_zero(self):
        """Test energy gain when gamma is constant."""
        trajectory = create_mock_trajectory(
            n_steps=5, gamma_values=[1000, 1000, 1000, 1000, 1000]
        )
        initial_gamma = 1000.0
        rest_energy_mev = 0.511

        max_gain = compute_max_energy_gain(trajectory, initial_gamma, rest_energy_mev)

        assert abs(max_gain) < 1e-9

    def test_compute_relative_energy_gain(self):
        """Test relative energy gain computation."""
        trajectory = create_mock_trajectory(n_steps=3, gamma_values=[1000, 1010, 1020])
        initial_gamma = 1000.0

        relative_gain = compute_relative_energy_gain(trajectory, initial_gamma)

        # Max gamma is 1020, relative gain = (1020 - 1000) / 1000 = 0.02
        expected = 0.02
        assert abs(relative_gain - expected) < 1e-9

    def test_detect_transverse_deflection(self):
        """Test transverse deflection detection."""
        # Create trajectory with jump followed by dip
        gamma_values = [1000, 1000, 1150, 1050, 1050]  # Jump at step 2, dip at step 3
        trajectory = create_mock_trajectory(n_steps=5, gamma_values=gamma_values)

        events = detect_transverse_deflection(
            trajectory, energy_jump_threshold=0.1, energy_dip_threshold=0.05
        )

        # Should detect jump, dip, and deflection
        event_types = [event[1] for event in events]
        assert "jump" in event_types
        assert "dip" in event_types
        assert "deflection" in event_types

    def test_compute_energy_at_position(self):
        """Test energy computation at specific position."""
        trajectory = create_mock_trajectory(
            n_steps=5, gamma_values=[1000, 1005, 1010, 1015, 1020]
        )
        initial_gamma = 1000.0
        rest_energy_mev = 0.511

        # Step 2 is at z=20mm with gamma=1010
        energy = compute_energy_at_position(
            trajectory,
            target_z=20.0,
            initial_gamma=initial_gamma,
            rest_energy_mev=rest_energy_mev,
            tolerance_mm=1.0,
        )

        # Delta gamma = 10, energy = 10 * 0.000511 GeV
        expected = 10 * 0.511 * 1e-3
        assert energy is not None
        assert abs(energy - expected) < 1e-6

    def test_compute_energy_at_position_not_found(self):
        """Test energy computation when position not in trajectory."""
        trajectory = create_mock_trajectory(n_steps=3)
        initial_gamma = 1000.0
        rest_energy_mev = 0.511

        # Request position far from trajectory
        energy = compute_energy_at_position(
            trajectory,
            target_z=1000.0,
            initial_gamma=initial_gamma,
            rest_energy_mev=rest_energy_mev,
            tolerance_mm=1.0,
        )

        assert energy is None


class TestParameterGrid:
    """Test parameter grid functionality."""

    def test_parameter_grid_creation(self):
        """Test creating a parameter grid."""
        params = {"aperture": [0.1, 0.2, 0.3], "energy": [1.0, 10.0]}

        grid = ParameterGrid(params)

        assert len(grid) == 6  # 3 * 2 = 6 combinations
        assert grid.get_grid_shape() == (3, 2)

    def test_parameter_grid_iteration(self):
        """Test iterating over parameter grid."""
        params = {"a": [1, 2], "b": [10, 20]}

        grid = ParameterGrid(params)
        configs = list(grid)

        assert len(configs) == 4
        assert {"a": 1, "b": 10} in configs
        assert {"a": 1, "b": 20} in configs
        assert {"a": 2, "b": 10} in configs
        assert {"a": 2, "b": 20} in configs

    def test_create_energy_aperture_grid(self):
        """Test creating standard energy-aperture grid."""
        apertures = [0.01, 0.1, 1.0]
        energies = [1.0, 10.0, 100.0]

        grid = create_energy_aperture_grid(
            aperture_sizes_mm=apertures, energies_gev=energies
        )

        assert len(grid) == 9  # 3 * 3
        assert grid.param_names == ["aperture_radius", "initial_energy_gev"]

    def test_create_energy_aperture_grid_defaults(self):
        """Test creating grid with default parameters."""
        grid = create_energy_aperture_grid()

        # Default should have 20 points per dimension
        assert len(grid) == 400  # 20 * 20
        assert grid.param_names == ["aperture_radius", "initial_energy_gev"]


class TestParameterMapping:
    """Test parameter name mapping utilities."""

    def test_aperture_radius_in_grid(self):
        """Test that aperture_radius is properly handled."""
        grid = create_energy_aperture_grid(aperture_sizes_mm=[0.1], energies_gev=[10.0])

        configs = list(grid)
        assert len(configs) == 1
        assert configs[0]["aperture_radius"] == 0.1

    def test_energy_in_grid(self):
        """Test that initial_energy_gev is properly handled."""
        grid = create_energy_aperture_grid(aperture_sizes_mm=[0.1], energies_gev=[10.0])

        configs = list(grid)
        assert len(configs) == 1
        assert configs[0]["initial_energy_gev"] == 10.0


class _MockVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class TestSweepHelpers:
    """Test extracted optimization sweep helpers."""

    def test_calculate_starting_pz_round_trip(self):
        """Energy and Pz conversions should round-trip for kinetic energy."""
        energy_gev = 2.5
        mass_amu = 1.0

        pz = calculate_starting_pz_from_energy(energy_gev, mass_amu)

        assert calculate_energy_from_pz(pz, mass_amu) == pytest.approx(energy_gev)
        assert calculate_energy_from_pz(-pz, mass_amu) == pytest.approx(energy_gev)

    def test_calculate_starting_pz_respects_negative_direction(self):
        """Negative direction should only flip the Pz sign."""
        positive = calculate_starting_pz_from_energy(1.0, 1.0, negative=False)
        negative = calculate_starting_pz_from_energy(1.0, 1.0, negative=True)

        assert positive == pytest.approx(-negative)

    def test_generate_parameter_range_linear_and_log(self):
        """Parameter range generation should support midpoint, linear, and log modes."""
        assert generate_parameter_range(2.0, 6.0, 1, False) == [4.0]
        assert generate_parameter_range(1.0, 3.0, 3, False) == pytest.approx(
            [1.0, 2.0, 3.0]
        )
        assert generate_parameter_range(1e-3, 1e1, 3, True) == pytest.approx(
            [1e-3, 1e-1, 1e1]
        )

    def test_build_parameter_grids_for_bunch_to_bunch(self):
        """Bunch-to-bunch sweeps should omit aperture and normalize driver energy."""
        config = SimpleNamespace(
            simulation_type=SimulationType.BUNCH_TO_BUNCH,
            aperture_range=(0.01, 0.1),
            aperture_points=3,
            aperture_log_scale=False,
            energy_range=(1.0, 3.0),
            energy_points=3,
            energy_log_scale=False,
            transverse_offset_fractions=[],
            starting_z_positions=[10.0, 20.0],
            wall_z_range=None,
            wall_z_points=1,
        )
        sweep_params = {
            "driver_energy_gev": {
                "sweep_var": _MockVar(True),
                "min_var": _MockVar(-4.0),
                "max_var": _MockVar(-1.0),
                "points_var": _MockVar(4),
                "log_var": _MockVar(False),
            },
            "driver_transv_mom": {
                "sweep_var": _MockVar(True),
                "min_var": _MockVar(0.0),
                "max_var": _MockVar(2.0),
                "points_var": _MockVar(3),
                "log_var": _MockVar(False),
            },
        }

        grids = build_parameter_grids(config, sweep_params)

        assert "aperture" not in grids
        assert grids["initial_energy_gev"] == pytest.approx([1.0, 2.0, 3.0])
        assert grids["transverse_offset_fraction"] == [0.0]
        assert grids["start_z"] == [10.0, 20.0]
        assert grids["driver_energy_gev"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert grids["driver_transv_mom"] == pytest.approx([0.0, 1.0, 2.0])

    def test_build_parameter_grids_skips_driver_controls_outside_b2b(self):
        """Driver-specific sweep controls should be ignored for wall simulations."""
        config = SimpleNamespace(
            simulation_type=SimulationType.CONDUCTING_WALL,
            aperture_range=(0.01, 0.1),
            aperture_points=2,
            aperture_log_scale=False,
            energy_range=(5.0, 10.0),
            energy_points=2,
            energy_log_scale=False,
            transverse_offset_fractions=[0.25],
            starting_z_positions=[0.0],
            wall_z_range=(50.0, 70.0),
            wall_z_points=3,
        )
        sweep_params = {
            "driver_energy_gev": {
                "sweep_var": _MockVar(True),
                "min_var": _MockVar(1.0),
                "max_var": _MockVar(2.0),
                "points_var": _MockVar(2),
                "log_var": _MockVar(False),
            }
        }

        grids = build_parameter_grids(config, sweep_params)

        assert grids["aperture"] == pytest.approx([0.01, 0.1])
        assert grids["energy"] == pytest.approx([5.0, 10.0])
        assert grids["transverse_offset_fraction"] == [0.25]
        assert grids["wall_z"] == pytest.approx([50.0, 60.0, 70.0])
        assert "driver_energy_gev" not in grids


class TestPluginConfigHelpers:
    """Test extracted optimization plugin config helpers."""

    def test_parse_float_list(self):
        assert parse_float_list("1, 2.5, -3") == [1.0, 2.5, -3.0]

    def test_parse_float_list_rejects_invalid_input(self):
        with pytest.raises(ValueError, match="Invalid list format"):
            parse_float_list("1, two")

    def test_parse_float_range(self):
        assert parse_float_range("1, 3") == (1.0, 3.0)
        assert parse_float_range("  ") is None

    def test_parse_offset_pair(self):
        assert parse_offset_pair("1.5, -2.0") == (1.5, -2.0)
        assert parse_offset_pair("4.0") == (4.0, 0.0)
        assert parse_offset_pair("bad") == (0.0, 0.0)

    def test_apply_sweep_parameter_overrides(self):
        config = OptimizationConfig()
        debug_messages = []
        sweep_params = {
            "rider_transv_mom": _make_sweep_controls(
                enabled=True, min_val=1.0, max_val=3.0, points=5
            ),
            "rider_transv_dist": _make_sweep_controls(enabled=False),
            "macroparticle_charge_multiplier": _make_sweep_controls(enabled=False),
            "macroparticle_sigma_multiplier": _make_sweep_controls(enabled=False),
            "rider_m_particle": _make_sweep_controls(enabled=False),
            "rider_charge_sign": _make_sweep_controls(enabled=False),
            "rider_pcount": _make_sweep_controls(
                enabled=True, min_val=2, max_val=8, points=4
            ),
            "rider_stripped_ions": _make_sweep_controls(enabled=False),
            "driver_m_particle": _make_sweep_controls(enabled=False, fixed_val=131.0),
            "driver_charge_sign": _make_sweep_controls(enabled=False),
            "driver_pcount": _make_sweep_controls(enabled=False),
            "driver_transv_mom": _make_sweep_controls(enabled=False),
            "driver_transv_dist": _make_sweep_controls(enabled=False),
            "driver_starting_distance": _make_sweep_controls(
                enabled=True, min_val=50.0, max_val=500.0, points=3, log=True
            ),
            "driver_energy_gev": _make_sweep_controls(
                enabled=True, min_val=-8.0, max_val=-2.0, points=4
            ),
            "driver_stripped_ions": _make_sweep_controls(enabled=False),
        }

        updated = apply_sweep_parameter_overrides(
            config,
            sweep_params,
            driver_negative=False,
            linked_energy_sweep=True,
            debug=debug_messages.append,
        )

        assert updated is config
        assert config.linked_energy_sweep is True
        assert config.transverse_momentum_range == (1.0, 3.0)
        assert config.transverse_momentum_points == 5
        assert config.particle_count_range == (2, 8)
        assert config.particle_count_points == 4
        assert config.driver_starting_distance_range == (50.0, 500.0)
        assert config.driver_starting_distance_points == 3
        assert config.driver_starting_distance_log_scale is True
        assert config.driver_direction == "+z"
        assert config.driver_energy_range == (2.0, 8.0)
        assert config.driver_energy_points == 4
        assert config.driver_starting_Pz_range[0] < 0
        assert config.driver_starting_Pz_range[1] < 0
        assert any("Linked energy sweep ENABLED" in msg for msg in debug_messages)

    def test_apply_sweep_parameter_overrides_fixed_driver_energy(self):
        config = OptimizationConfig()
        sweep_params = {
            "rider_transv_mom": _make_sweep_controls(enabled=False),
            "rider_transv_dist": _make_sweep_controls(enabled=False),
            "macroparticle_charge_multiplier": _make_sweep_controls(enabled=False),
            "macroparticle_sigma_multiplier": _make_sweep_controls(enabled=False),
            "rider_m_particle": _make_sweep_controls(enabled=False),
            "rider_charge_sign": _make_sweep_controls(enabled=False),
            "rider_pcount": _make_sweep_controls(enabled=False),
            "rider_stripped_ions": _make_sweep_controls(enabled=False),
            "driver_m_particle": _make_sweep_controls(enabled=False, fixed_val=10.0),
            "driver_charge_sign": _make_sweep_controls(enabled=False),
            "driver_pcount": _make_sweep_controls(enabled=False),
            "driver_transv_mom": _make_sweep_controls(enabled=False),
            "driver_transv_dist": _make_sweep_controls(enabled=False),
            "driver_starting_distance": _make_sweep_controls(enabled=False),
            "driver_energy_gev": _make_sweep_controls(
                enabled=False, fixed_val=-1.25
            ),
            "driver_stripped_ions": _make_sweep_controls(enabled=False),
        }

        apply_sweep_parameter_overrides(
            config,
            sweep_params,
            driver_negative=True,
            linked_energy_sweep=False,
        )

        assert config.driver_energy_gev == 1.25
        assert config.driver_starting_Pz < 0


class TestPluginPersistenceHelpers:
    """Test extracted optimization plugin persistence helpers."""

    def test_metrics_export_settings_handles_legacy_keys(self):
        export_format, export_scope = metrics_export_settings_from_data(
            {
                "export_evaluation_csv": False,
                "export_top_n_metrics_csv": True,
            }
        )

        assert export_format == "none"
        assert export_scope == "top_n"

    def test_apply_persisted_config_overrides_sets_saved_values_and_defaults(self):
        config = OptimizationConfig()

        apply_persisted_config_overrides(
            config,
            {
                "self_consistency_enabled": False,
                "adaptive_timestep_threshold": 0.25,
                "per_run_timeout": 12.5,
            },
        )

        assert config.self_consistency_enabled is False
        assert config.adaptive_timestep_threshold == pytest.approx(0.25)
        assert config.per_run_timeout == pytest.approx(12.5)
        assert config.smoothness_enabled is True
        assert config.energy_monitor_enabled is False
        assert config.energy_monitor_threshold == pytest.approx(2.0)

    def test_build_saved_config_payload_captures_config_and_ui_fields(self):
        config = OptimizationConfig(
            simulation_type=SimulationType.BUNCH_TO_BUNCH,
            mode="optimization",
            aperture_range=(0.1, 0.2),
            energy_range=(1.0, 2.0),
            transverse_offset_fractions=[0.3],
            starting_z_positions=[5.0],
            save_top_n_trajectories=True,
            linked_energy_sweep=True,
            transv_offset_x=1.2,
            transv_offset_y=-1.3,
            driver_transv_offset_x=2.4,
            driver_transv_offset_y=-2.5,
        )

        payload = build_saved_config_payload(
            config,
            timestep_mode="duration",
            auto_steps_distance=42.0,
            rider_stripped_ions=3.0,
            driver_stripped_ions=54.0,
            driver_direction="+z",
            sweep_state={"driver_energy_gev": {"enabled": False, "fixed_value": "7"}},
        )

        assert payload["simulation_type"] == "BUNCH_TO_BUNCH"
        assert payload["mode"] == "optimization"
        assert payload["save_top_n_trajectories"] is True
        assert payload["driver_direction"] == "+z"
        assert payload["linked_energy_sweep"] is True
        assert payload["auto_steps_distance"] == pytest.approx(42.0)
        assert payload["rider_offset_x"] == pytest.approx(1.2)
        assert payload["driver_offset_y"] == pytest.approx(-2.5)
        assert payload["sweep_parameters"]["driver_energy_gev"]["fixed_value"] == "7"

    def test_resolve_loaded_sweep_state_converts_legacy_driver_pz(self):
        state = resolve_loaded_sweep_state(
            {
                "driver_starting_Pz": {
                    "enabled": True,
                    "points": "5",
                    "log": True,
                }
            },
            "driver_energy_gev",
        )

        assert state == {
            "enabled": True,
            "min": "50.0",
            "max": "200.0",
            "points": "5",
            "log": True,
            "fixed_value": "112.5",
        }


class TestPluginResultsHelpers:
    """Test extracted optimization plugin result helpers."""

    def test_convert_legacy_trajectory_data_handles_missing_gamma_history(self):
        result = convert_legacy_trajectory_data(
            {
                "core": {
                    "rider": {
                        "positions_mm": {"x": [3.0], "y": [4.0], "z": [5.0]},
                        "conjugate_momenta": {"Px": [6.0], "Py": [8.0], "Pz": [9.0]},
                        "time_ns": [0.1],
                    }
                },
                "aperture_radius": 0.2,
                "wall_z": 10.0,
            },
            m_particle_amu=1.0,
            amu_to_mev=931.494,
        )

        assert result["parameters"]["particle_energy_gev"] == pytest.approx(0.0)
        assert result["trajectory"]["r"] == pytest.approx([5.0])
        assert result["trajectory"]["pr"] == pytest.approx([10.0])
        assert result["metrics"]["rider_delta_e_mev"] == pytest.approx(0.0)

    def test_summarize_result_row_normalizes_distance_and_metrics(self):
        row = summarize_result_row(
            {
                "run_number": 7,
                "parameters": {
                    "aperture_radius": 0.03,
                    "particle_energy_gev": 12.0,
                    "starting_z": 4.0,
                },
                "metrics": {
                    "rider_delta_e_mev": 1.5,
                    "rider_gamma_initial": 100.0,
                    "rider_gamma_final": 101.0,
                },
                "_distance_info": {"z_start": 20.0, "z_end": 5.0},
            }
        )

        assert row["run_num"] == 7
        assert row["aperture"] == pytest.approx(0.03)
        assert row["energy"] == pytest.approx(12.0)
        assert row["start_z"] == pytest.approx(4.0)
        assert row["delta_e"] == pytest.approx(1.5)
        assert row["traveled"] == pytest.approx(15.0)
        assert row["gamma_initial"] == pytest.approx(100.0)
        assert row["gamma_final"] == pytest.approx(101.0)

    def test_build_summary_heatmap_grid_maps_results_to_grid(self):
        results = [
            {
                "parameters": {"aperture_radius": 0.1, "particle_energy_gev": 1.0},
                "metrics": {"rider_delta_e_mev": 10.0},
            },
            {
                "parameters": {"aperture_radius": 0.2, "particle_energy_gev": 1.0},
                "metrics": {"rider_delta_e_mev": 20.0},
            },
            {
                "parameters": {"aperture_radius": 0.1, "particle_energy_gev": 2.0},
                "metrics": {"rider_delta_e_mev": 30.0},
            },
            {
                "parameters": {"aperture_radius": 0.2, "particle_energy_gev": 2.0},
                "metrics": {"rider_delta_e_mev": 40.0},
            },
        ]

        plot_data = collect_summary_plot_data(results)
        assert plot_data == {
            "apertures": [0.1, 0.2, 0.1, 0.2],
            "energies": [1.0, 1.0, 2.0, 2.0],
            "delta_es": [10.0, 20.0, 30.0, 40.0],
        }

        unique_a, unique_e, grid = build_summary_heatmap_grid(results)
        assert unique_a == [0.1, 0.2]
        assert unique_e == [1.0, 2.0]
        assert grid.tolist() == [[10.0, 20.0], [30.0, 40.0]]

    def test_build_trajectory_plot_data_prepares_energy_delta_series(self):
        plot_data = build_trajectory_plot_data(
            [
                {
                    "run_number": 3,
                    "parameters": {
                        "aperture_radius": 0.05,
                        "particle_energy_gev": 15.0,
                    },
                    "metrics": {
                        "rider_delta_e_mev": 2.0,
                        "rider_gamma_initial": 10.0,
                    },
                    "trajectory": {
                        "z": [0.0, 5.0, 10.0],
                        "r": [0.1, 0.2, 0.3],
                    },
                }
            ],
            m_particle_amu=1.0,
            amu_to_mev=931.494,
        )

        assert len(plot_data["series"]) == 1
        series = plot_data["series"][0]
        assert series["run_num"] == 3
        assert series["aperture"] == pytest.approx(0.05)
        assert series["energy"] == pytest.approx(15.0)
        assert series["energy_delta"].tolist() == pytest.approx([0.0, 1.0, 2.0])
        assert plot_data["heatmap"] == {
            "apertures": [0.05],
            "energies": [15.0],
            "delta_es": [2.0],
        }

    def test_parse_results_payload_classifies_saved_formats(self):
        sweep = parse_results_payload(
            {"results": [{"trajectory": {"z": [0.0]}}]},
            m_particle_amu=1.0,
            amu_to_mev=931.494,
        )
        optimization = parse_results_payload(
            {"all_evaluations": []},
            m_particle_amu=1.0,
            amu_to_mev=931.494,
        )
        legacy = parse_results_payload(
            {"core": {"rider": {"positions_mm": {}, "conjugate_momenta": {}}}},
            m_particle_amu=1.0,
            amu_to_mev=931.494,
        )

        assert sweep["kind"] == "sweep"
        assert len(sweep["results_with_trajectories"]) == 1
        assert optimization["kind"] == "optimization"
        assert legacy["kind"] == "legacy"
        assert len(legacy["results"]) == 1

    def test_summarize_saved_results_handles_sweep_and_optimization_payloads(self):
        sweep_summary = summarize_saved_results(
            {
                "kind": "sweep",
                "results": [
                    {
                        "run_number": 2,
                        "parameters": {
                            "aperture_radius": 0.1,
                            "particle_energy_gev": 5.0,
                        },
                        "metrics": {"rider_delta_e_mev": 1.5},
                        "trajectory": {"z": [0.0]},
                    }
                ],
                "results_with_trajectories": [{"trajectory": {"z": [0.0]}}],
            }
        )
        optimization_summary = summarize_saved_results(
            {
                "kind": "optimization",
                "payload": {
                    "objective": "max_energy_gain",
                    "optimization_method": "genetic_algorithm",
                    "best_value": 2.5,
                    "success": True,
                    "total_evaluations": 3,
                    "all_evaluations": [
                        {"objective_value": 1.0},
                        {"objective_value": float("inf")},
                    ],
                    "top_n_count": 1,
                    "top_n_results": [
                        {"metrics": {"rider_delta_e_mev": 4.0}},
                    ],
                    "best_parameters": {"initial_energy_gev": 6.0},
                },
                "results": [],
                "results_with_trajectories": [],
            }
        )

        assert sweep_summary == {
            "result_type": "sweep",
            "run_count": 1,
            "trajectory_count": 1,
            "best_run_number": 2,
            "best_delta_e_mev": 1.5,
            "best_energy_gev": 5.0,
            "best_aperture_mm": 0.1,
        }
        assert optimization_summary["result_type"] == "optimization"
        assert optimization_summary["evaluation_count"] == 3
        assert optimization_summary["finite_evaluation_count"] == 1
        assert optimization_summary["best_delta_e_mev"] == pytest.approx(4.0)
        assert optimization_summary["best_parameters"] == {"initial_energy_gev": 6.0}
        assert optimization_summary["top_results"][0]["rank"] == 1
        assert optimization_summary["top_results"][0]["delta_e_mev"] == pytest.approx(
            4.0
        )

    def test_summarize_optimization_top_results_falls_back_to_all_evaluations(self):
        top_results = summarize_optimization_top_results(
            {
                "objective": "max_energy_gain",
                "all_evaluations": [
                    {
                        "evaluation": 5,
                        "failed": False,
                        "halted_early": False,
                        "raw_objective_value": 2.5,
                        "fitness": -2.5,
                        "parameters": {"initial_energy_gev": 7.0},
                        "metrics": {
                            "rider_delta_e_mev": 3.0,
                            "max_percent_energy_gain": 1.2,
                        },
                    },
                    {
                        "evaluation": 6,
                        "failed": False,
                        "halted_early": False,
                        "raw_objective_value": 1.5,
                        "fitness": -1.5,
                        "parameters": {"initial_energy_gev": 6.0},
                        "metrics": {
                            "rider_delta_e_mev": 2.0,
                            "max_percent_energy_gain": 1.0,
                        },
                    },
                ],
            }
        )

        assert top_results == [
            {
                "rank": 1,
                "evaluation": 5,
                "metric_value": 2.5,
                "fitness": -2.5,
                "parameters": {"initial_energy_gev": 7.0},
                "delta_e_mev": 3.0,
                "percent_energy_gain": 1.2,
                "metrics": {
                    "rider_delta_e_mev": 3.0,
                    "max_percent_energy_gain": 1.2,
                },
            },
            {
                "rank": 2,
                "evaluation": 6,
                "metric_value": 1.5,
                "fitness": -1.5,
                "parameters": {"initial_energy_gev": 6.0},
                "delta_e_mev": 2.0,
                "percent_energy_gain": 1.0,
                "metrics": {
                    "rider_delta_e_mev": 2.0,
                    "max_percent_energy_gain": 1.0,
                },
            },
        ]


class TestOptimizationResultsMixin:
    """Test extracted optimization results helpers."""

    def test_save_single_trajectory_handles_bunch_to_bunch_driver_params(
        self, tmp_path
    ):
        config = OptimizationConfig(
            simulation_type=SimulationType.BUNCH_TO_BUNCH,
            output_dir=str(tmp_path),
            aperture_range=(0.1, 0.2),
            energy_range=(1.0, 2.0),
            starting_z_positions=[5.0],
            transverse_offset_fractions=[0.25],
            m_particle=1.0,
            charge_sign=-1.0,
            pcount=3,
            transv_mom=0.01,
            transv_dist=0.02,
            driver_m_particle=10.0,
            driver_charge_sign=1.0,
            driver_pcount=4,
            driver_transv_mom=0.03,
            driver_transv_dist=0.04,
            driver_starting_distance=123.0,
            driver_starting_Pz=-456.0,
            driver_stripped_ions=5.0,
            driver_transv_offset_x=0.6,
            driver_transv_offset_y=-0.7,
        )
        harness = _ResultsMixinHarness(config, tmp_path)

        trajectory = harness._save_single_optimization_trajectory(
            {"transverse_offset": 0.5, "initial_energy_gev": 1.5},
            ["transverse_offset", "initial_energy_gev"],
            rank=1,
            fitness=0.123,
        )

        assert trajectory is not None
        assert harness.captured_run["transv_offset"] == pytest.approx(0.5)
        assert harness.captured_run["driver_params"] == {
            "m_particle": 10.0,
            "charge_sign": 1.0,
            "pcount": 4,
            "transv_mom": 0.03,
            "transv_dist": 0.04,
            "starting_distance": 123.0,
            "starting_Pz": -456.0,
            "stripped_ions": 5.0,
            "transv_offset_x": 0.6,
            "transv_offset_y": -0.7,
        }
        assert (tmp_path / "trajectory_rank1_best.npz").exists()
        assert (tmp_path / "trajectory_rank1_best.png").exists()


def _make_sweep_controls(
    *,
    enabled: bool,
    min_val: float = 0.0,
    max_val: float = 1.0,
    points: int = 2,
    fixed_val: float = 0.0,
    log: bool = False,
):
    return {
        "sweep_var": _MockVar(enabled),
        "min_var": _MockVar(min_val),
        "max_var": _MockVar(max_val),
        "points_var": _MockVar(points),
        "fixed_var": _MockVar(fixed_val),
        "log_var": _MockVar(log),
    }


class _ResultsMixinHarness(OptimizationResultsMixin):
    def __init__(self, config, output_dir):
        self.config = config
        self._last_optimization_dir = output_dir
        self.captured_run = None
        self.logged_messages = []

    def _run_single_integration(self, **kwargs):
        self.captured_run = kwargs
        return {
            "trajectory": {
                "z": [0.0, 1.0],
                "t": [0.0, 0.1],
                "r": [0.0, 0.2],
                "gamma": [2.0, 2.5],
                "pr": [0.0, 0.01],
            },
            "metrics": {},
        }

    def _log_result(self, message: str):
        self.logged_messages.append(message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
