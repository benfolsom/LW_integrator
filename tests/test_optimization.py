"""Tests for optimization module.

Basic tests to verify optimization functionality works correctly.
"""

import numpy as np
import pytest

from optimization.metrics import (
    compute_energy_at_position,
    compute_max_energy_gain,
    compute_relative_energy_gain,
    detect_transverse_deflection,
)
from optimization.parameter_sweep import ParameterGrid, create_energy_aperture_grid


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
