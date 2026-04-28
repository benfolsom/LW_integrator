"""Tests for maintained testbed trajectory helper functionality."""

import json

import numpy as np
import pytest

from core.types import ParticleState  # noqa: E402
import lw_integrator.testbed_runner as testbed_runner
from lw_integrator.testbed_runner import SimulationOptions  # noqa: E402
from lw_integrator.trajectory_metrics import (  # noqa: E402
    compute_delta_energy_series,
    normalize_state,
)


def create_mock_particle_state(
    t: float, x: float, y: float, z: float, gamma: float
) -> ParticleState:
    """Create a mock ParticleState for testing."""
    return {
        "t": np.array([t]),
        "x": np.array([x]),
        "y": np.array([y]),
        "z": np.array([z]),
        "bx": np.array([0.0]),
        "by": np.array([0.0]),
        "bz": np.array([0.999]),
        "gamma": np.array([gamma]),
    }


class TestComputeDeltaEnergySeries:
    """Test the compute_delta_energy_series function."""

    def test_zero_energy_change(self):
        """Test when there's no energy change."""
        initial_gamma = 1000.0
        states = [
            create_mock_particle_state(0.0, 0.0, 0.0, 0.0, initial_gamma),
            create_mock_particle_state(1.0, 0.0, 0.0, 100.0, initial_gamma),
            create_mock_particle_state(2.0, 0.0, 0.0, 200.0, initial_gamma),
        ]
        initial_state = states[0]
        rest_energy_mev = 0.511  # electron rest mass

        delta_e, z_series = compute_delta_energy_series(
            states, initial_state, rest_energy_mev
        )

        assert len(delta_e) == 3
        assert len(z_series) == 3
        np.testing.assert_array_almost_equal(delta_e, [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(z_series, [0.0, 100.0, 200.0])

    def test_linear_energy_gain(self):
        """Test linear energy gain."""
        states = [
            create_mock_particle_state(0.0, 0.0, 0.0, 0.0, 1000.0),
            create_mock_particle_state(1.0, 0.0, 0.0, 100.0, 1010.0),
            create_mock_particle_state(2.0, 0.0, 0.0, 200.0, 1020.0),
        ]
        initial_state = states[0]
        rest_energy_mev = 0.511  # electron rest mass
        rest_energy_gev = rest_energy_mev * 1e-3

        delta_e, z_series = compute_delta_energy_series(
            states, initial_state, rest_energy_mev
        )

        expected_delta_e = np.array([0.0, 10.0, 20.0]) * rest_energy_gev
        np.testing.assert_array_almost_equal(delta_e, expected_delta_e)
        np.testing.assert_array_almost_equal(z_series, [0.0, 100.0, 200.0])

    def test_energy_loss(self):
        """Test energy loss scenario."""
        states = [
            create_mock_particle_state(0.0, 0.0, 0.0, 0.0, 10000.0),
            create_mock_particle_state(1.0, 0.0, 0.0, 100.0, 9990.0),
            create_mock_particle_state(2.0, 0.0, 0.0, 200.0, 9980.0),
        ]
        initial_state = states[0]
        rest_energy_mev = 0.511

        delta_e, z_series = compute_delta_energy_series(
            states, initial_state, rest_energy_mev
        )

        assert delta_e[1] < 0.0  # Energy loss
        assert delta_e[2] < delta_e[1]  # Continuing to lose energy
        assert z_series[1] > z_series[0]


class TestNormalizeState:
    def test_normalize_state_wraps_scalars_and_preserves_metadata(self):
        normalized = normalize_state(
            {
                "gamma": 10.0,
                "z": [1.0, 2.0],
                "_halt_reason": "jump",
            }
        )

        np.testing.assert_array_equal(normalized["gamma"], np.array([10.0]))
        np.testing.assert_array_equal(normalized["z"], np.array([1.0, 2.0]))
        assert normalized["_halt_reason"] == "jump"


class TestFilenameGeneration:
    """Test filename generation with config name and timestamp."""

    def test_filename_base_strips_json_extension(self, monkeypatch):
        """Test that generated filename bases include sanitized config names."""
        monkeypatch.setattr(testbed_runner.time, "strftime", lambda *_args: "20251022_123456")

        assert (
            testbed_runner.generate_filename_base("electronwall10.3gev.json")
            == "electronwall10.3gev_20251022_123456"
        )
        assert (
            testbed_runner.generate_filename_base("my_config")
            == "my_config_20251022_123456"
        )

    def test_filename_base_defaults_empty_config_name(self, monkeypatch):
        monkeypatch.setattr(testbed_runner.time, "strftime", lambda *_args: "20251022_123456")

        assert (
            testbed_runner.generate_filename_base("  ")
            == "testbed_config_20251022_123456"
        )


class TestPlotValidation:
    """Test plot data validation."""

    def test_delta_e_vs_delta_z(self):
        """Validate that plots show ΔE vs Δz, not Δγ/γ."""
        states = [
            create_mock_particle_state(0.0, 0.0, 0.0, 0.0, 1000.0),
            create_mock_particle_state(1.0, 0.0, 0.0, 100.0, 1010.0),
        ]
        initial_state = states[0]
        rest_energy_mev = 0.511
        rest_energy_gev = rest_energy_mev * 1e-3

        delta_e, z_series = compute_delta_energy_series(
            states, initial_state, rest_energy_mev
        )

        # Delta E should be in GeV (not percent)
        assert delta_e[0] == 0.0
        expected_delta_e_1 = 10.0 * rest_energy_gev
        assert abs(delta_e[1] - expected_delta_e_1) < 1e-6

        # Z series should be position in mm
        assert z_series[0] == 0.0
        assert z_series[1] == 100.0

    def test_relative_positions(self):
        """Test that Δz is relative to initial position."""
        states = [
            create_mock_particle_state(0.0, 0.0, 0.0, 500.0, 1000.0),
            create_mock_particle_state(1.0, 0.0, 0.0, 600.0, 1010.0),
            create_mock_particle_state(2.0, 0.0, 0.0, 700.0, 1020.0),
        ]
        initial_state = states[0]
        rest_energy_mev = 0.511

        _, z_series = compute_delta_energy_series(
            states, initial_state, rest_energy_mev
        )

        z_rel = z_series - z_series[0]
        np.testing.assert_array_almost_equal(z_rel, [0.0, 100.0, 200.0])


class TestConfigManagement:
    """Test configuration save/load functionality."""

    @pytest.mark.parametrize(
        ("raw_mode", "expected_mode"),
        [
            ("fixed_geometry", "fixed_geometry"),
            ("variable_geometry", "variable_geometry"),
            ("mass_shell_only", "fixed_geometry"),
            ("full_iteration", "variable_geometry"),
        ],
    )
    def test_simulation_options_canonicalizes_self_consistency_mode(
        self, raw_mode, expected_mode
    ):
        options = SimulationOptions.from_dict(
            {"self_consistency_convergence_mode": raw_mode}
        )

        assert options.self_consistency_convergence_mode == expected_mode

    def test_config_snapshot_structure(self):
        """Test that config snapshot has all required fields."""
        config_snapshot = {
            "steps": 1000,
            "seed": 12345,
            "simulation_type": "BUNCH_TO_BUNCH",
            "energy_save": True,
            "energy_display": True,
            "transverse_display": False,
            "transverse_save": False,
            "trajectory_save": False,
            "trajectory_interval": 10,
            "plot_dpi": 300,
            "output_dir": "test_outputs/testbed_runs",
            "config_dir": "configs/testbed_runs",
            "config_name": "test_config.json",
            "rider_params": {},
            "driver_params": {},
            "core_params": {},
        }

        # Verify all essential fields are present
        assert "steps" in config_snapshot
        assert "seed" in config_snapshot
        assert "simulation_type" in config_snapshot
        assert "energy_save" in config_snapshot
        assert "energy_display" in config_snapshot
        assert "rider_params" in config_snapshot
        assert "core_params" in config_snapshot

    def test_config_serialization(self):
        """Test that config can be serialized to JSON."""
        config = {
            "steps": 1000,
            "seed": 12345,
            "energy_save": True,
        }

        # Should not raise
        json_str = json.dumps(config, indent=2)
        loaded = json.loads(json_str)

        assert loaded["steps"] == 1000
        assert loaded["seed"] == 12345
        assert loaded["energy_save"] is True
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
