"""Tests for optimization plugin integration."""

from pathlib import Path
from types import SimpleNamespace
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from core.types import SimulationType
from lw_integrator.optimization_plugin import OptimizationConfig, OptimizationPlugin
from optimization.plugin_config_mixins import OptimizationPluginConfigMixin
from optimization.plugin_control_mixins import (
    OptimizationPluginControlMixin,
    _stability_dialog_logging_defaults,
)
from optimization.plugin_form_mixins import OptimizationPluginFormMixin
from optimization.plugin_parameter_mixins import OptimizationPluginParameterMixin
from optimization.plugin_runtime_mixins import OptimizationPluginRuntimeMixin
from optimization.plugin_view_mixins import OptimizationPluginViewMixin
from optimization.results_mixins import OptimizationResultsMixin
from optimization.run_mixins import OptimizationRunMixin
from optimization.run_parameter_helpers import OptimizationRunParameters
from optimization.plugin_ui_mixins import OptimizationPluginUIMixin
import optimization.plugin_view_mixins as view_mixins_module
import optimization.run_mixins as run_mixins_module
from optimization.sweep_helpers import calculate_energy_from_pz


class _MockVar:
    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class _WidgetRecorder:
    def __init__(self):
        self.calls = []
        self.text = None

    def config(self, **kwargs):
        self.calls.append(kwargs)
        if "text" in kwargs:
            self.text = kwargs["text"]

    def configure(self, **kwargs):
        self.config(**kwargs)


def _build_sweep_harness(sweep_params):
    harness = SimpleNamespace(sweep_params=sweep_params)

    def set_fixed_sweep_value(param_name, value):
        harness.sweep_params[param_name]["fixed_var"].set(value)

    harness._set_fixed_sweep_value = set_fixed_sweep_value
    return harness


def _optimization_run_params() -> OptimizationRunParameters:
    return OptimizationRunParameters(
        aperture=0.25,
        energy_gev=5.0,
        start_z=1.0,
        transv_offset=0.1,
        timestep=1e-6,
        steps=10,
        rider_m_particle=0.00054857990907,
        rider_charge_sign=-1.0,
        rider_pcount=2,
        rider_transv_mom=0.0,
        rider_transv_dist=0.0,
        rider_stripped_ions=1.0,
        macroparticle_charge_multiplier=1.0,
        macroparticle_sigma_multiplier=1.0,
        driver_params=None,
        wall_z=100.0,
    )


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

    def test_plugin_inherits_form_helpers_from_form_mixin(self):
        assert (
            OptimizationPlugin._add_sweepable_param
            is OptimizationPluginFormMixin._add_sweepable_param
        )
        assert (
            OptimizationPlugin._update_rider_pz_helper
            is OptimizationPluginFormMixin._update_rider_pz_helper
        )
        assert (
            OptimizationPlugin._on_sim_type_changed
            is OptimizationPluginFormMixin._on_sim_type_changed
        )
        assert (
            OptimizationPlugin._update_parameter_visibility
            is OptimizationPluginFormMixin._update_parameter_visibility
        )

    def test_plugin_inherits_parameter_section_builders_from_parameter_mixin(self):
        assert (
            OptimizationPlugin._build_parameter_section
            is OptimizationPluginParameterMixin._build_parameter_section
        )
        assert (
            OptimizationPlugin._build_particle_section
            is OptimizationPluginParameterMixin._build_particle_section
        )
        assert (
            OptimizationPlugin._build_rider_particle_section
            is OptimizationPluginParameterMixin._build_rider_particle_section
        )
        assert (
            OptimizationPlugin._build_driver_particle_section
            is OptimizationPluginParameterMixin._build_driver_particle_section
        )

    def test_plugin_inherits_config_helpers_from_config_mixin(self):
        assert (
            OptimizationPlugin._sync_stability_to_main_gui
            is OptimizationPluginConfigMixin._sync_stability_to_main_gui
        )
        assert (
            OptimizationPlugin._on_load_from_main_config
            is OptimizationPluginConfigMixin._on_load_from_main_config
        )
        assert (
            OptimizationPlugin._load_config_from_path
            is OptimizationPluginConfigMixin._load_config_from_path
        )
        assert (
            OptimizationPlugin._save_config_to_path
            is OptimizationPluginConfigMixin._save_config_to_path
        )

    def test_plugin_inherits_results_view_helpers_from_view_mixin(self):
        assert (
            OptimizationPlugin._on_view_results
            is OptimizationPluginViewMixin._on_view_results
        )
        assert (
            OptimizationPlugin._show_results_summary
            is OptimizationPluginViewMixin._show_results_summary
        )
        assert (
            OptimizationPlugin._show_trajectory_viewer
            is OptimizationPluginViewMixin._show_trajectory_viewer
        )

    def test_plugin_inherits_control_helpers_from_control_mixin(self):
        assert (
            OptimizationPlugin._validate_inputs
            is OptimizationPluginControlMixin._validate_inputs
        )
        assert (
            OptimizationPlugin._gather_config
            is OptimizationPluginControlMixin._gather_config
        )
        assert (
            OptimizationPlugin._on_run_sweep
            is OptimizationPluginControlMixin._on_run_sweep
        )
        assert OptimizationPlugin._on_stop is OptimizationPluginControlMixin._on_stop
        assert not hasattr(OptimizationPluginControlMixin, "_compute_soft_penalty")
        assert not hasattr(OptimizationPlugin, "_compute_soft_penalty")

    @pytest.mark.parametrize(
        ("sim_type", "driver_state", "driver_color"),
        [
            ("CONDUCTING_WALL", "disabled", "gray"),
            ("SWITCHING_WALL", "disabled", "gray"),
            ("BUNCH_TO_BUNCH", "normal", "black"),
        ],
    )
    def test_parameter_visibility_toggles_driver_offset_by_simulation_type(
        self, sim_type, driver_state, driver_color
    ):
        widgets = {
            name: _WidgetRecorder()
            for name in (
                "cavity_spacing_label",
                "cavity_spacing_desc_label",
                "offset_label",
                "offset_entry",
                "offset_desc_label",
                "driver_offset_label",
                "driver_offset_entry",
                "driver_offset_desc_label",
            )
        }
        harness = SimpleNamespace(
            sim_type_var=_MockVar(sim_type),
            cavity_spacing_entry=_WidgetRecorder(),
            _param_widgets=widgets,
            _set_frame_state=lambda frame, state: None,
            _toggle_wall_z_sweep=lambda: None,
            _update_timestep_tooltip=lambda: None,
            _update_distance_target_labels=lambda: None,
        )

        OptimizationPlugin._update_parameter_visibility(harness)

        assert widgets["offset_entry"].calls[-1] == {"state": "normal"}
        assert widgets["offset_label"].calls[-1] == {"foreground": "black"}
        assert widgets["offset_desc_label"].calls[-1] == {"foreground": "gray40"}
        assert widgets["driver_offset_entry"].calls[-1] == {"state": driver_state}
        assert widgets["driver_offset_label"].calls[-1] == {"foreground": driver_color}
        assert widgets["driver_offset_desc_label"].calls[-1] == {
            "foreground": "gray40" if sim_type == "BUNCH_TO_BUNCH" else "gray"
        }

    def test_set_top_n_controls_state_updates_direct_widget_references(self):
        controls = {
            name: _WidgetRecorder()
            for name in (
                "optimization_save_top_n_entry",
                "save_top_n_traj_check",
                "metrics_scope_top_n_radio",
                "log_top_n_only_radio",
            )
        }
        harness = SimpleNamespace(
            save_top_n_traj_var=_MockVar(True),
            metrics_scope_var=_MockVar("top_n"),
            log_verbosity_var=_MockVar("top_n_only"),
            **controls,
        )

        OptimizationPlugin._set_top_n_controls_state(harness, "disabled")

        for control in controls.values():
            assert control.calls[-1] == {"state": "disabled"}
        assert harness.save_top_n_traj_var.get() is False
        assert harness.metrics_scope_var.get() == "all"
        assert harness.log_verbosity_var.get() == "truncated"

    def test_gather_stability_config_kwargs_prefers_gui_with_config_fallback(self):
        existing_config = OptimizationConfig(
            image_subcharge_count=24,
            use_image_weighting=False,
            self_consistency_enabled=False,
            self_consistency_tolerance=2e-4,
            adaptive_timestep_enabled=False,
            adaptive_timestep_threshold=0.25,
            self_consistency_gamma_reconciliation_method="FIXED_WEIGHTED",
            self_consistency_gamma_reconciliation_fixed_weight=0.7,
        )
        harness = OptimizationPluginControlMixin()
        harness.gui_controller = SimpleNamespace(
            image_subcharge_var=_MockVar(16),
            self_consistency_target_ms_tolerance_var=_MockVar("5e-4"),
            adaptive_timestep_enabled_var=_MockVar(True),
        )

        kwargs = harness._gather_stability_config_kwargs(existing_config)

        assert kwargs["image_subcharge_count"] == 16
        assert kwargs["self_consistency_tolerance"] == pytest.approx(5e-4)
        assert kwargs["adaptive_timestep_enabled"] is True
        assert kwargs["use_image_weighting"] is False
        assert kwargs["self_consistency_enabled"] is False
        assert kwargs["adaptive_timestep_threshold"] == pytest.approx(0.25)
        assert (
            kwargs["self_consistency_gamma_reconciliation_method"] == "FIXED_WEIGHTED"
        )
        assert kwargs["self_consistency_gamma_reconciliation_fixed_weight"] == 0.7

    def test_stability_dialog_logging_defaults_preserve_silent_config(self):
        config = OptimizationConfig(
            self_consistency_verbosity=0,
            adaptive_timestep_debug=False,
        )

        assert _stability_dialog_logging_defaults(config) == ("0", False)

    def test_gather_particle_config_kwargs_reads_fixed_sweep_values(self):
        harness = OptimizationPluginControlMixin()
        harness.macroparticle_enabled_var = _MockVar(True)
        harness.macroparticle_momentum_errors_var = _MockVar(False)
        harness.sweep_params = {
            "rider_transv_mom": {"fixed_var": _MockVar("0.01")},
            "rider_transv_dist": {"fixed_var": _MockVar("0.02")},
            "macroparticle_charge_multiplier": {"fixed_var": _MockVar("10")},
            "macroparticle_sigma_multiplier": {"fixed_var": _MockVar("2")},
            "rider_m_particle": {"fixed_var": _MockVar("1.0")},
            "rider_pcount": {"fixed_var": _MockVar("3")},
            "rider_charge_sign": {"fixed_var": _MockVar("-1")},
            "rider_stripped_ions": {"fixed_var": _MockVar("4")},
            "driver_m_particle": {"fixed_var": _MockVar("2.0")},
            "driver_charge_sign": {"fixed_var": _MockVar("1")},
            "driver_pcount": {"fixed_var": _MockVar("5")},
            "driver_transv_mom": {"fixed_var": _MockVar("0.03")},
            "driver_transv_dist": {"fixed_var": _MockVar("0.04")},
            "driver_starting_distance": {"fixed_var": _MockVar("100")},
            "driver_stripped_ions": {"fixed_var": _MockVar("6")},
        }

        kwargs = harness._gather_particle_config_kwargs((0.1, 0.2), (-0.3, -0.4))

        assert kwargs["transv_mom"] == pytest.approx(0.01)
        assert kwargs["transv_dist"] == pytest.approx(0.02)
        assert kwargs["transv_offset_x"] == pytest.approx(0.1)
        assert kwargs["transv_offset_y"] == pytest.approx(0.2)
        assert kwargs["driver_transv_offset_x"] == pytest.approx(-0.3)
        assert kwargs["driver_transv_offset_y"] == pytest.approx(-0.4)
        assert kwargs["macroparticle_enabled"] is True
        assert kwargs["macroparticle_use_momentum_errors"] is False
        assert kwargs["m_particle"] == pytest.approx(1.0)
        assert kwargs["pcount"] == 3
        assert kwargs["charge_sign"] == pytest.approx(-1.0)
        assert kwargs["stripped_ions"] == pytest.approx(4.0)
        assert kwargs["driver_m_particle"] == pytest.approx(2.0)
        assert kwargs["driver_pcount"] == 5
        assert kwargs["driver_stripped_ions"] == pytest.approx(6.0)

    def test_gather_sweep_grid_kwargs_normalizes_b2b_aperture_points(self):
        harness = OptimizationPluginControlMixin()
        harness.sim_type_var = _MockVar("BUNCH_TO_BUNCH")
        harness.aperture_min_var = _MockVar("0.01")
        harness.aperture_max_var = _MockVar("0.02")
        harness.aperture_points_var = _MockVar("9")
        harness.aperture_log_var = _MockVar(False)
        harness.energy_min_var = _MockVar("1.0")
        harness.energy_max_var = _MockVar("2.0")
        harness.energy_points_var = _MockVar("3")
        harness.energy_log_var = _MockVar(True)
        harness.offset_fractions_var = _MockVar("0.1, 0.2")
        harness.start_z_var = _MockVar("5.0")
        harness.wall_z_var = _MockVar("100.0")
        harness.wall_z_sweep_var = _MockVar(True)
        harness.wall_z_min_var = _MockVar("90.0")
        harness.wall_z_max_var = _MockVar("110.0")
        harness.wall_z_points_var = _MockVar("4")

        kwargs = harness._gather_sweep_grid_kwargs()

        assert kwargs["simulation_type"] is SimulationType.BUNCH_TO_BUNCH
        assert kwargs["aperture_range"] == pytest.approx((0.01, 0.02))
        assert kwargs["aperture_points"] == 1
        assert kwargs["energy_range"] == pytest.approx((1.0, 2.0))
        assert kwargs["energy_points"] == 3
        assert kwargs["energy_log_scale"] is True
        assert kwargs["transverse_offset_fractions"] == pytest.approx([0.1, 0.2])
        assert kwargs["starting_z_positions"] == pytest.approx([5.0])
        assert kwargs["wall_z_range"] == pytest.approx((90.0, 110.0))
        assert kwargs["wall_z_points"] == 4

    def test_plugin_inherits_runtime_helpers_from_runtime_mixin(self):
        assert (
            OptimizationPlugin._log_truncated_run
            is OptimizationPluginRuntimeMixin._log_truncated_run
        )
        assert (
            OptimizationPlugin._should_save_trajectory
            is OptimizationPluginRuntimeMixin._should_save_trajectory
        )
        assert (
            OptimizationPlugin._update_progress
            is OptimizationPluginRuntimeMixin._update_progress
        )
        assert (
            OptimizationPlugin._reset_ui_state
            is OptimizationPluginRuntimeMixin._reset_ui_state
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

        monkeypatch.setattr(
            run_mixins_module, "SimulationOptions", _FakeSimulationOptions
        )
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

    def test_run_optimization_evaluation_integration_runs_directly(self):
        captured = {}

        def fake_run_single_integration(**kwargs):
            captured["kwargs"] = kwargs
            return {"metrics": {"ok": True}}

        harness = SimpleNamespace(
            config=SimpleNamespace(per_run_timeout=0.0),
            _run_single_integration=fake_run_single_integration,
            _log_result=lambda _message: None,
        )

        result, timed_out = (
            OptimizationRunMixin._run_optimization_evaluation_integration(
                harness, _optimization_run_params(), eval_num=4, original_params=[1.0]
            )
        )

        assert timed_out is False
        assert result == {"metrics": {"ok": True}}
        assert captured["kwargs"]["run_num"] == 4
        assert captured["kwargs"]["cancel_flag"] is None
        assert captured["kwargs"]["aperture"] == 0.25
        assert captured["kwargs"]["rider_pcount"] == 2

    def test_run_optimization_evaluation_integration_signals_timeout(self):
        logs = []

        def fake_run_single_integration(**kwargs):
            cancel_flag = kwargs["cancel_flag"]
            deadline = time.time() + 1.0
            while not cancel_flag[0] and time.time() < deadline:
                time.sleep(0.001)
            return {"metrics": {"late": True}}

        harness = SimpleNamespace(
            config=SimpleNamespace(per_run_timeout=0.01),
            _run_single_integration=fake_run_single_integration,
            _log_result=logs.append,
        )

        result, timed_out = (
            OptimizationRunMixin._run_optimization_evaluation_integration(
                harness, _optimization_run_params(), eval_num=5, original_params=[2.0]
            )
        )

        assert result is None
        assert timed_out is True
        assert any("timed out" in message for message in logs)

    def test_run_sweep_integration_attempt_runs_directly(self):
        captured = {}

        def fake_run_single_integration(**kwargs):
            captured["kwargs"] = kwargs
            return {"metrics": {"ok": True}}

        harness = SimpleNamespace(
            config=SimpleNamespace(per_run_timeout=0.0, wall_z=200.0),
            _run_single_integration=fake_run_single_integration,
            _log_result=lambda _message: None,
        )
        run_params = SimpleNamespace(
            aperture=0.25,
            energy=5.0,
            start_z=1.0,
            transv_offset=0.1,
            rider_m_particle=1.0,
            rider_charge_sign=1.0,
            rider_pcount=2,
            rider_transv_mom=0.0,
            rider_transv_dist=1e-4,
            rider_stripped_ions=1.0,
            macroparticle_charge_multiplier=1.0,
            macroparticle_sigma_multiplier=1.0,
            driver_params=None,
        )

        result, error, timed_out = OptimizationRunMixin._run_sweep_integration_attempt(
            harness,
            run_params,
            {"wall_z": 250.0},
            timestep=1e-7,
            steps=100,
            run_num=6,
            seed_override=1234,
        )

        assert result == {"metrics": {"ok": True}}
        assert error is None
        assert timed_out is False
        assert captured["kwargs"]["cancel_flag"] is None
        assert captured["kwargs"]["wall_z"] == 250.0
        assert captured["kwargs"]["seed_override"] == 1234

    def test_run_sweep_integration_attempt_signals_timeout(self):
        logs = []

        def fake_run_single_integration(**kwargs):
            cancel_flag = kwargs["cancel_flag"]
            deadline = time.time() + 1.0
            while not cancel_flag[0] and time.time() < deadline:
                time.sleep(0.001)
            return {"metrics": {"late": True}}

        harness = SimpleNamespace(
            config=SimpleNamespace(per_run_timeout=0.01, wall_z=200.0),
            _run_single_integration=fake_run_single_integration,
            _log_result=logs.append,
        )
        run_params = SimpleNamespace(
            aperture=0.01,
            energy=5.0,
            start_z=1.0,
            transv_offset=0.1,
            rider_m_particle=1.0,
            rider_charge_sign=1.0,
            rider_pcount=2,
            rider_transv_mom=0.0,
            rider_transv_dist=1e-4,
            rider_stripped_ions=1.0,
            macroparticle_charge_multiplier=2000.0,
            macroparticle_sigma_multiplier=1.0,
            driver_params=None,
        )

        _result, error, timed_out = OptimizationRunMixin._run_sweep_integration_attempt(
            harness,
            run_params,
            {},
            timestep=1e-7,
            steps=100,
            run_num=7,
            seed_override=1234,
        )

        assert error is None
        assert timed_out is True
        assert any("exceeded timeout" in message for message in logs)
        assert any("Very small aperture" in message for message in logs)

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

        monkeypatch.setattr(
            view_mixins_module.filedialog, "askopenfilename", fake_askopenfilename
        )
        monkeypatch.setattr(
            view_mixins_module.messagebox, "askyesno", lambda *args, **kwargs: False
        )

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
        assert (
            harness.sweep_params["driver_m_particle"]["fixed_var"].get()
            == "2.072000e+02"
        )
        assert harness.sweep_params["driver_charge_sign"]["fixed_var"].get() == "1.0"
        assert harness.sweep_params["driver_pcount"]["fixed_var"].get() == "5"
        assert float(
            harness.sweep_params["driver_energy_gev"]["fixed_var"].get()
        ) == pytest.approx(calculate_energy_from_pz(-4925.0, 207.2))
        assert harness.sweep_params["driver_stripped_ions"]["fixed_var"].get() == "54.0"

    def test_linked_energy_presentation_shows_rider_sweep_range(self):
        label_widget = _WidgetRecorder()
        help_label = _WidgetRecorder()
        harness = SimpleNamespace(
            link_driver_rider_energy_var=_MockVar(True),
            energy_min_var=_MockVar("0.5"),
            energy_max_var=_MockVar("3000"),
            energy_points_var=_MockVar("80"),
            link_energy_help_label=help_label,
            sweep_params={
                "driver_energy_gev": {
                    "label_text": "Kinetic Energy (GeV):",
                    "label_widget": label_widget,
                }
            },
        )

        OptimizationPluginFormMixin._update_linked_energy_presentation(harness)

        assert label_widget.text == "Kinetic Energy (GeV, linked):"
        assert (
            help_label.text == "(Driver follows rider sweep: 0.5 to 3000 GeV, 80 pts)"
        )

    def test_on_link_energy_toggled_grays_out_driver_energy_entry(self):
        fixed_entry = _WidgetRecorder()
        harness = SimpleNamespace(
            link_driver_rider_energy_var=_MockVar(True),
            driver_frame=SimpleNamespace(winfo_children=lambda: []),
            sweep_params={
                "driver_energy_gev": {
                    "fixed_entry": fixed_entry,
                    "sweep_var": _MockVar(True),
                }
            },
            _toggle_sweep_controls=Mock(),
            _update_linked_energy_presentation=Mock(),
            _update_driver_pz_helper=Mock(),
            _ensure_linked_disabled_entry_style=Mock(),
            _LINKED_DISABLED_ENTRY_STYLE=(
                OptimizationPluginFormMixin._LINKED_DISABLED_ENTRY_STYLE
            ),
        )
        harness._set_driver_energy_entry_linked_state = lambda linked: OptimizationPluginFormMixin._set_driver_energy_entry_linked_state(
            harness, linked
        )

        OptimizationPluginFormMixin._on_link_energy_toggled(harness)

        assert harness.sweep_params["driver_energy_gev"]["sweep_var"].get() is False
        assert {
            "style": "LinkedDriverEnergyDisabled.TEntry",
            "state": "disabled",
        } in fixed_entry.calls

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

        OptimizationPlugin._sync_main_gui_simulation_type(harness, "BUNCH_TO_BUNCH")

        assert sim_type_var.get() == "BUNCH_TO_BUNCH"
        combo.current.assert_called_once_with(1)
        root.update_idletasks.assert_called_once_with()

    def test_on_stop_marks_cancelled_and_notifies_gui(self):
        cancel_var = SimpleNamespace(_cancel_requested=False)
        harness = SimpleNamespace(
            running=True,
            _was_cancelled=False,
            gui_controller=cancel_var,
            _update_progress_text=Mock(),
        )

        OptimizationPlugin._on_stop(harness)

        assert harness.running is False
        assert harness._was_cancelled is True
        assert harness.gui_controller._cancel_requested is True
        harness._update_progress_text.assert_called_once_with("Stopping...")

    def test_reset_ui_state_restores_gui_controls(self):
        run_button = Mock()
        cancel_button = Mock()
        set_status = Mock()
        harness = SimpleNamespace(
            running=False,
            gui_controller=SimpleNamespace(
                _running=True,
                _cancel_requested=True,
                _set_status=set_status,
                _run_button=run_button,
                _cancel_button=cancel_button,
            ),
            _update_progress_text=Mock(),
        )

        OptimizationPlugin._reset_ui_state(harness)

        assert harness.gui_controller._running is False
        assert harness.gui_controller._cancel_requested is False
        set_status.assert_called_once_with("Ready")
        run_button.configure.assert_called_once_with(state="normal")
        cancel_button.configure.assert_called_once_with(state="disabled")
        harness._update_progress_text.assert_called_once_with("Ready")

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
