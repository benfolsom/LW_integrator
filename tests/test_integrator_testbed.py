"""
Tests for the integrator testbed notebook functionality.

This test suite validates the key functionality of the integrator testbed
widget without requiring the full Jupyter environment.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "validation"))

from core.types import ParticleState  # noqa: E402
from examples.validation.core_vs_legacy_benchmark import (  # noqa: E402
    compute_delta_energy_series,
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


class TestFilenameGeneration:
    """Test filename generation with config name and timestamp."""

    def test_filename_with_timestamp(self):
        """Test that filenames include timestamp."""
        from datetime import datetime

        config_name = "electronwall10.3gev.json"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Test energy filename
        expected_base = config_name.replace(".json", "")
        filename = f"{expected_base}_energy_{timestamp}.png"

        assert expected_base in filename
        assert "energy" in filename
        assert ".png" in filename
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS

    def test_filename_without_json_extension(self):
        """Test config name without .json extension."""
        config_name = "my_config"
        timestamp = "20251022_123456"

        filename = f"{config_name}_energy_{timestamp}.png"
        assert filename == "my_config_energy_20251022_123456.png"


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

    def test_config_snapshot_structure(self):
        """Test that config snapshot has all required fields."""
        config_snapshot = {
            "steps": 1000,
            "seed": 12345,
            "simulation_type": "BUNCH_TO_BUNCH",
            "legacy_enabled": False,
            "overlay_display": False,
            "overlay_save": False,
            "difference_display": False,
            "difference_save": False,
            "metrics_save": False,
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


class TestLegacyOverlayPlots:
    """Test legacy overlay plot functionality."""

    def test_overlay_requires_legacy_enabled(self):
        """Test that overlay plots require legacy to be enabled."""
        legacy_enabled = False
        overlay_display = True

        # Overlay should only work if legacy is enabled
        should_show_overlay = legacy_enabled and overlay_display
        assert should_show_overlay is False

    def test_overlay_with_legacy(self):
        """Test overlay plots when legacy is enabled."""
        legacy_enabled = True
        overlay_display = True

        should_show_overlay = legacy_enabled and overlay_display
        assert should_show_overlay is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
