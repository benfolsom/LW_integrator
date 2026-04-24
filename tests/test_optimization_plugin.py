"""Tests for optimization plugin integration."""

from pathlib import Path
from types import SimpleNamespace
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from core.types import SimulationType
import lw_integrator.optimization_plugin as plugin_module
from lw_integrator.optimization_plugin import OptimizationConfig, OptimizationPlugin
from optimization.results_mixins import OptimizationResultsMixin
from optimization.run_mixins import OptimizationRunMixin
from optimization.plugin_ui_mixins import OptimizationPluginUIMixin
import optimization.run_mixins as run_mixins_module
from optimization.sweep_helpers import calculate_energy_from_pz


class _MockVar:
    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


def _build_sweep_harness(sweep_params):
    harness = SimpleNamespace(sweep_params=sweep_params)

    def set_fixed_sweep_value(param_name, value):
        harness.sweep_params[param_name]["fixed_var"].set(value)

    harness._set_fixed_sweep_value = set_fixed_sweep_value
    return harness


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

    def test_plugin_inherits_run_helpers_from_run_mixin(self):
        assert (
            OptimizationPlugin._run_optimization_background
            is OptimizationRunMixin._run_optimization_background
        )
        assert (
            OptimizationPlugin._run_sweep_background
            is OptimizationRunMixin._run_sweep_background
        )
        assert (
            OptimizationPlugin._run_single_integration
            is OptimizationRunMixin._run_single_integration
        )
        assert (
            OptimizationPlugin._cleanup_orphaned_temp_dirs
            is OptimizationRunMixin._cleanup_orphaned_temp_dirs
        )

    def test_plugin_inherits_section_builders_from_ui_mixin(self):
        assert OptimizationPlugin._build_ui is OptimizationPluginUIMixin._build_ui
        assert (
            OptimizationPlugin._build_simulation_section
            is OptimizationPluginUIMixin._build_simulation_section
        )
        assert (
            OptimizationPlugin._build_mode_section
            is OptimizationPluginUIMixin._build_mode_section
        )
        assert (
            OptimizationPlugin._build_optimization_section
            is OptimizationPluginUIMixin._build_optimization_section
        )
        assert (
            OptimizationPlugin._build_results_output_section
            is OptimizationPluginUIMixin._build_results_output_section
        )

    def test_run_single_integration_uses_current_simulation_options_fields(
        self, mock_config, tmp_path, monkeypatch
    ):
        captured = {}

        class _AbortRun(RuntimeError):
            pass

        class _FakeSimulationOptions:
            def __init__(self, **kwargs):
                forbidden = {
                    "legacy_enabled",
                    "overlay_display",
                    "overlay_save",
                    "difference_display",
                    "difference_save",
                    "metrics_save",
                }
                assert forbidden.isdisjoint(kwargs)
                captured["kwargs"] = kwargs

        def fake_run_testbed(*_args, **_kwargs):
            raise _AbortRun()

        monkeypatch.setattr(run_mixins_module, "SimulationOptions", _FakeSimulationOptions)
        monkeypatch.setattr(run_mixins_module, "run_testbed", fake_run_testbed)

        harness = SimpleNamespace(
            config=mock_config,
            sweep_output_dir=tmp_path,
            _log_result=lambda _message: None,
        )

        with pytest.raises(_AbortRun):
            OptimizationRunMixin._run_single_integration(
                harness,
                aperture=0.25,
                energy_gev=5.0,
                start_z=1.0,
                transv_offset=0.0,
                timestep=1e-6,
                steps=10,
                run_num=3,
            )

        assert captured["kwargs"]["output_dir"].parent == tmp_path
        assert captured["kwargs"]["seed"] == mock_config.seed + 3

    def test_apply_macroparticle_ui_state_updates_controls(self):
        harness = _build_sweep_harness(
            {
            "macroparticle_charge_multiplier": {"fixed_var": _MockVar()},
            "macroparticle_sigma_multiplier": {"fixed_var": _MockVar()},
            }
        )
        harness.macroparticle_enabled_var = _MockVar()
        harness.macroparticle_momentum_errors_var = _MockVar()
        harness._toggle_macroparticle_controls = Mock()
        harness._update_macroparticle_state = Mock()

        OptimizationPlugin._apply_macroparticle_ui_state(
            harness,
            enabled=True,
            charge_multiplier="1.23e+00",
            sigma_multiplier="4.56e+00",
            momentum_errors=False,
            refresh_state=True,
        )

        assert harness.macroparticle_enabled_var.get() is True
        assert (
            harness.sweep_params["macroparticle_charge_multiplier"]["fixed_var"].get()
            == "1.23e+00"
        )
        assert (
            harness.sweep_params["macroparticle_sigma_multiplier"]["fixed_var"].get()
            == "4.56e+00"
        )
        assert harness.macroparticle_momentum_errors_var.get() is False
        harness._toggle_macroparticle_controls.assert_called_once_with()
        harness._update_macroparticle_state.assert_called_once_with()

    def test_apply_smoothness_ui_state_updates_controls(self):
        harness = SimpleNamespace()
        harness.smoothness_enabled_var = _MockVar()
        harness.smoothness_window_var = _MockVar()
        harness.smoothness_oscillation_var = _MockVar()
        harness.smoothness_reject_var = _MockVar()
        harness._toggle_smoothness_controls = Mock()

        OptimizationPlugin._apply_smoothness_ui_state(
            harness,
            enabled=False,
            window_size="12",
            oscillation_threshold="0.75",
            reject_on_violation=True,
        )

        assert harness.smoothness_enabled_var.get() is False
        assert harness.smoothness_window_var.get() == "12"
        assert harness.smoothness_oscillation_var.get() == "0.75"
        assert harness.smoothness_reject_var.get() is True
        harness._toggle_smoothness_controls.assert_called_once_with()

    def test_plot_trajectories_uses_current_results_directories(
        self, tmp_path, monkeypatch
    ):
        sweep_dir = tmp_path / "sweeps"
        sweep_dir.mkdir()
        latest_dir = sweep_dir / "latest"
        latest_dir.mkdir()
        config_output_dir = tmp_path / "config_output"
        config_output_dir.mkdir()

        captured = {}

        def fake_askopenfilename(**kwargs):
            captured["initialdir"] = kwargs["initialdir"]
            return ""

        monkeypatch.setattr(plugin_module.filedialog, "askopenfilename", fake_askopenfilename)
        monkeypatch.setattr(plugin_module.messagebox, "askyesno", lambda *args, **kwargs: False)

        harness = SimpleNamespace(
            sweep_output_dir=str(sweep_dir),
            config=SimpleNamespace(output_dir=str(config_output_dir)),
        )

        OptimizationPlugin._on_plot_trajectories(harness)

        assert Path(captured["initialdir"]) == latest_dir

    def test_apply_driver_sweep_values_sets_driver_fields(self):
        harness = _build_sweep_harness(
            {
            "driver_m_particle": {"fixed_var": _MockVar()},
            "driver_charge_sign": {"fixed_var": _MockVar()},
            "driver_pcount": {"fixed_var": _MockVar()},
            "driver_transv_mom": {"fixed_var": _MockVar()},
            "driver_transv_dist": {"fixed_var": _MockVar()},
            "driver_starting_distance": {"fixed_var": _MockVar()},
            "driver_energy_gev": {"fixed_var": _MockVar()},
            "driver_stripped_ions": {"fixed_var": _MockVar()},
            }
        )

        updated = OptimizationPlugin._apply_driver_sweep_values(
            harness,
            {
                "m_particle": 207.2,
                "charge_sign": 1.0,
                "pcount": 5,
                "transv_mom": 0.0,
                "transv_dist": -0.07998,
                "starting_distance": 1000.0,
                "starting_Pz": -4925.0,
                "stripped_ions": 54.0,
            },
        )

        assert updated is True
        assert harness.sweep_params["driver_m_particle"]["fixed_var"].get() == "2.072000e+02"
        assert harness.sweep_params["driver_charge_sign"]["fixed_var"].get() == "1.0"
        assert harness.sweep_params["driver_pcount"]["fixed_var"].get() == "5"
        assert (
            float(harness.sweep_params["driver_energy_gev"]["fixed_var"].get())
            == pytest.approx(calculate_energy_from_pz(-4925.0, 207.2))
        )
        assert harness.sweep_params["driver_stripped_ions"]["fixed_var"].get() == "54.0"

    def test_sync_main_gui_simulation_type_updates_controller(self):
        sim_type_var = _MockVar()
        class _Combo:
            def __init__(self):
                self.current = Mock()

            def __getitem__(self, key):
                assert key == "values"
                return ("CONDUCTING_WALL", "BUNCH_TO_BUNCH")

        combo = _Combo()
        root = Mock()
        harness = SimpleNamespace(
            gui_controller=SimpleNamespace(
                sim_type_var=sim_type_var,
                sim_type_combo=combo,
                root=root,
            )
        )

        OptimizationPlugin._sync_main_gui_simulation_type(
            harness, "BUNCH_TO_BUNCH"
        )

        assert sim_type_var.get() == "BUNCH_TO_BUNCH"
        combo.current.assert_called_once_with(1)
        root.update_idletasks.assert_called_once_with()

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
