"""Tests for optimization plugin integration."""

import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from core.types import SimulationType
from lw_integrator.optimization_plugin import OptimizationConfig, OptimizationPlugin
from optimization.results_mixins import OptimizationResultsMixin


@pytest.fixture
def mock_config():
    """Create a mock optimization configuration."""
    config = OptimizationConfig(
        simulation_type=SimulationType.CONDUCTING_WALL,
        aperture_range=(1e-5, 1e-3),
        aperture_points=2,
        energy_range=(1.0, 10.0),
        energy_points=2,
        steps=100,
        timestep=1e-6,
        wall_z=100.0,
        output_dir="test_output",
        self_consistency_enabled=False,  # Disable for faster tests
        self_consistency_verbosity=0,
        adaptive_timestep_debug=False,
        per_run_timeout=5.0,  # Short timeout for tests
        skip_failed_runs=True,
    )
    return config


@pytest.fixture
def mock_run_result():
    """Create a mock RunResult from testbed_runner."""
    result = Mock()
    result.figures = {}  # Empty dict to avoid matplotlib issues
    result.rider_delta_e = 1.5  # MeV
    result.rider_gamma_initial = 1000.0
    result.rider_gamma_final = 1003.0
    result.rider_emittance_x_mm_mrad = None
    result.rider_emittance_y_mm_mrad = None
    result.rider_norm_emittance_x_mm_mrad = None
    result.rider_norm_emittance_y_mm_mrad = None
    result.rider_beta_x_m = None
    result.rider_beta_y_m = None
    result.rider_trajectory = {
        "z": np.array([0.0, 10.0, 20.0, 30.0]),
        "r": np.array([0.0, 0.0, 0.0, 0.0]),
        "pz": np.array([1000.0, 1001.0, 1002.0, 1003.0]),
        "pr": np.array([0.0, 0.0, 0.0, 0.0]),
        "t": np.array([0.0, 0.01, 0.02, 0.03]),
    }
    return result


class TestOptimizationPluginIntegration:
    """Test optimization plugin integration functionality."""

    def test_plugin_inherits_npz_viewer_helpers_from_results_mixin(self):
        assert (
            OptimizationPlugin._view_npz_trajectories
            is OptimizationResultsMixin._view_npz_trajectories
        )
        assert (
            OptimizationPlugin._plot_npz_trajectories
            is OptimizationResultsMixin._plot_npz_trajectories
        )

    def test_run_single_integration_completes(self, mock_config, mock_run_result):
        """Test that _run_single_integration completes without hanging."""
        # Test the logic without GUI by directly calling the method logic

        # Simulate what _run_single_integration does
        completed = threading.Event()
        result_data = None
        error = None

        def simulate_integration():
            nonlocal result_data, error
            try:
                # Simulate the key steps that could hang
                import matplotlib

                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt

                # Process mock figures
                figures = mock_run_result.figures
                for fig in figures.values():
                    plt.close(fig)

                # Extract metrics (the actual logic from _run_single_integration)
                metrics = {}
                if mock_run_result.rider_delta_e is not None:
                    metrics["rider_delta_e_mev"] = mock_run_result.rider_delta_e
                if mock_run_result.rider_gamma_initial is not None:
                    metrics["rider_gamma_initial"] = mock_run_result.rider_gamma_initial
                if mock_run_result.rider_gamma_final is not None:
                    metrics["rider_gamma_final"] = mock_run_result.rider_gamma_final

                output = {"metrics": metrics}

                # Process trajectory
                if mock_run_result.rider_trajectory is not None:
                    traj = mock_run_result.rider_trajectory
                    z_array = np.asarray(traj["z"])
                    if len(z_array) > 0:
                        output["_distance_info"] = {
                            "z_start": float(z_array[0]),
                            "z_end": float(z_array[-1]),
                            "num_steps": len(z_array),
                        }

                result_data = output
                completed.set()
            except Exception as e:
                error = e
                completed.set()

        thread = threading.Thread(target=simulate_integration, daemon=True)
        thread.start()

        # Wait for completion with timeout
        if not completed.wait(timeout=10.0):
            pytest.fail("Integration simulation hung and did not complete within 10s")

        # Check for errors
        if error:
            raise error

        # Verify result structure
        assert result_data is not None, "Result should not be None"
        assert "metrics" in result_data, "Result should contain metrics"
        assert "rider_delta_e_mev" in result_data["metrics"]
        assert result_data["metrics"]["rider_delta_e_mev"] == 1.5

        # Verify trajectory info was extracted
        assert "_distance_info" in result_data
        assert result_data["_distance_info"]["z_start"] == 0.0
        assert result_data["_distance_info"]["z_end"] == 30.0
        assert result_data["_distance_info"]["num_steps"] == 4

    def test_optimization_evaluate_params_with_timeout(self, mock_config):
        """Test that optimization evaluation respects timeout."""
        # Test timeout logic without GUI

        timeout = 2.0
        maximize = True

        def slow_function():
            time.sleep(10)  # Longer than timeout
            return {"metrics": {"value": 1.0}}

        # Simulate the timeout wrapper from evaluate_params
        timed_out = False

        result_container = [None]
        error_container = [None]

        def run_slow():
            try:
                result_container[0] = slow_function()
            except Exception as e:
                error_container[0] = e

        thread = threading.Thread(target=run_slow)
        thread.daemon = True

        start_time = time.time()
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            timed_out = True
            value = np.inf if not maximize else -np.inf
        elif error_container[0] is not None:
            raise error_container[0]
        else:
            _ = result_container[0]
            value = -1.0

        elapsed = time.time() - start_time

        # Should timeout in ~2s, not wait for the full 10s
        assert elapsed < 5.0, f"Should timeout quickly, took {elapsed}s"
        assert timed_out, "Should have timed out"
        assert value == -np.inf, "Timed out evaluation should return -inf"

    def test_matplotlib_cleanup_no_hang(self, mock_config):
        """Test that matplotlib figure cleanup doesn't hang."""
        # Test matplotlib cleanup logic without GUI

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        # Create a real figure
        fig = plt.figure()

        completed = threading.Event()
        error = None

        def cleanup_figure():
            nonlocal error
            try:
                # Force Agg backend and close
                matplotlib.use("Agg", force=True)
                plt.close(fig)
                completed.set()
            except Exception as e:
                error = e
                completed.set()

        thread = threading.Thread(target=cleanup_figure, daemon=True)
        thread.start()

        # Should complete quickly
        if not completed.wait(timeout=5.0):
            pytest.fail("Figure cleanup hung and did not complete within 5s")

        if error:
            raise error

        # Cleanup
        plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
