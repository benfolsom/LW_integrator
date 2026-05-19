"""Focused GUI regression tests for lightweight controller paths."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from types import SimpleNamespace

import pytest

from lw_integrator import gui
from lw_integrator.gui_config_mixins import IntegratorGUIConfigMixin
from lw_integrator.testbed_runner import SimulationOptions
from lw_integrator.gui_config_list_mixins import IntegratorGUIConfigListMixin
from lw_integrator.gui_controller_mixins import IntegratorGUIControllerMixin
from lw_integrator.gui_layout_mixins import IntegratorGUILayoutMixin
from lw_integrator.gui_log_mixins import IntegratorGUILogMixin
from lw_integrator.gui_plot_mixins import IntegratorGUIPlotMixin
from lw_integrator.gui_runtime_mixins import IntegratorGUIRuntimeMixin
from lw_integrator.gui_shell_mixins import IntegratorGUIShellMixin
from lw_integrator.gui_state_mixins import IntegratorGUIStateMixin
from lw_integrator.gui_summary_mixins import IntegratorGUISummaryMixin
from lw_integrator.gui_tab_mixins import IntegratorGUITabMixin


class _Var:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _ComboBoxStub:
    def __init__(self, values):
        self._values = values
        self.current_calls = []

    def __getitem__(self, key):
        assert key == "values"
        return self._values

    def current(self, index):
        self.current_calls.append(index)


def test_integrator_gui_constructs_full_tk_window():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    root.withdraw()
    try:
        app = gui.IntegratorGUI(root)
        root.update_idletasks()
        assert isinstance(app, gui.IntegratorGUI)
    finally:
        root.destroy()


def test_gui_inherits_config_helpers_from_config_mixin():
    assert gui.IntegratorGUI._load_config is IntegratorGUIConfigMixin._load_config
    assert (
        gui.IntegratorGUI._apply_options_to_ui
        is IntegratorGUIConfigMixin._apply_options_to_ui
    )
    assert (
        gui.IntegratorGUI._build_options_from_ui
        is IntegratorGUIConfigMixin._build_options_from_ui
    )
    assert gui.IntegratorGUI._save_config is IntegratorGUIConfigMixin._save_config


def test_gui_inherits_runtime_helpers_from_runtime_mixin():
    assert gui.IntegratorGUI._trigger_run is IntegratorGUIRuntimeMixin._trigger_run
    assert (
        gui.IntegratorGUI._trigger_cancel is IntegratorGUIRuntimeMixin._trigger_cancel
    )
    assert (
        gui.IntegratorGUI._run_background is IntegratorGUIRuntimeMixin._run_background
    )
    assert gui.IntegratorGUI._on_success is IntegratorGUIRuntimeMixin._on_success


def test_gui_inherits_plot_helpers_from_plot_mixin():
    assert gui.IntegratorGUI._queue_log is IntegratorGUIPlotMixin._queue_log
    assert (
        gui.IntegratorGUI._replot_with_new_axis
        is IntegratorGUIPlotMixin._replot_with_new_axis
    )
    assert gui.IntegratorGUI._show_figure is IntegratorGUIPlotMixin._show_figure
    assert (
        gui.IntegratorGUI._prepare_figure_for_display
        is IntegratorGUIPlotMixin._prepare_figure_for_display
    )


def test_gui_inherits_summary_helpers_from_summary_mixin():
    assert (
        gui.IntegratorGUI._refresh_initial_summary
        is IntegratorGUISummaryMixin._refresh_initial_summary
    )
    assert (
        gui.IntegratorGUI._format_summary is IntegratorGUISummaryMixin._format_summary
    )


def test_gui_inherits_state_helpers_from_state_mixin():
    assert (
        gui.IntegratorGUI._on_sim_type_change
        is IntegratorGUIStateMixin._on_sim_type_change
    )
    assert (
        gui.IntegratorGUI._update_driver_visibility
        is IntegratorGUIStateMixin._update_driver_visibility
    )
    assert (
        gui.IntegratorGUI._toggle_z_cutoff_controls
        is IntegratorGUIStateMixin._toggle_z_cutoff_controls
    )
    assert (
        gui.IntegratorGUI._update_macroparticle_state
        is IntegratorGUIStateMixin._update_macroparticle_state
    )


def test_gui_inherits_config_list_helpers_from_config_list_mixin():
    assert (
        gui.IntegratorGUI._refresh_config_list
        is IntegratorGUIConfigListMixin._refresh_config_list
    )
    assert (
        gui.IntegratorGUI._selected_config_filename
        is IntegratorGUIConfigListMixin._selected_config_filename
    )
    assert (
        gui.IntegratorGUI._load_sweep_config
        is IntegratorGUIConfigListMixin._load_sweep_config
    )
    assert (
        gui.IntegratorGUI._save_sweep_config
        is IntegratorGUIConfigListMixin._save_sweep_config
    )


def test_gui_inherits_controller_helpers_from_controller_mixin():
    assert gui.IntegratorGUI._set_status is IntegratorGUIControllerMixin._set_status
    assert (
        gui.IntegratorGUI._apply_species is IntegratorGUIControllerMixin._apply_species
    )
    assert (
        gui.IntegratorGUI._trigger_sweep is IntegratorGUIControllerMixin._trigger_sweep
    )


def test_gui_inherits_layout_helpers_from_layout_mixin():
    assert gui.IntegratorGUI._enforce_panel_minimums is (
        IntegratorGUILayoutMixin._enforce_panel_minimums
    )
    assert gui.IntegratorGUI._create_scrollable_tab is (
        IntegratorGUILayoutMixin._create_scrollable_tab
    )
    assert gui.IntegratorGUI._build_config_panel is (
        IntegratorGUILayoutMixin._build_config_panel
    )
    assert gui.IntegratorGUI._on_run_mode_changed is (
        IntegratorGUILayoutMixin._on_run_mode_changed
    )
    assert gui.IntegratorGUI._build_log_summary_panel is (
        IntegratorGUILayoutMixin._build_log_summary_panel
    )


def test_gui_inherits_tab_builders_from_tab_mixin():
    assert gui.IntegratorGUI._build_output_tab is (
        IntegratorGUITabMixin._build_output_tab
    )
    assert gui.IntegratorGUI._build_particle_tab is (
        IntegratorGUITabMixin._build_particle_tab
    )
    assert gui.IntegratorGUI._build_core_tab is IntegratorGUITabMixin._build_core_tab
    assert gui.IntegratorGUI._build_stability_tab is (
        IntegratorGUITabMixin._build_stability_tab
    )
    assert gui.IntegratorGUI._build_external_fields_tab is (
        IntegratorGUITabMixin._build_external_fields_tab
    )
    assert gui.IntegratorGUI._build_self_consistency_section is (
        IntegratorGUITabMixin._build_self_consistency_section
    )
    assert gui.IntegratorGUI._build_adaptive_timestep_section is (
        IntegratorGUITabMixin._build_adaptive_timestep_section
    )
    assert gui.IntegratorGUI._build_radiation_reaction_section is (
        IntegratorGUITabMixin._build_radiation_reaction_section
    )
    assert gui.IntegratorGUI._add_output_toggle is (
        IntegratorGUITabMixin._add_output_toggle
    )


def test_gui_inherits_log_helpers_from_log_mixin():
    assert gui.IntegratorGUI._append_log is IntegratorGUILogMixin._append_log
    assert gui.IntegratorGUI._parse_log_line is IntegratorGUILogMixin._parse_log_line
    assert (
        gui.IntegratorGUI._refresh_summary_display
        is IntegratorGUILogMixin._refresh_summary_display
    )
    assert (
        gui.IntegratorGUI._update_log_format is IntegratorGUILogMixin._update_log_format
    )
    assert gui.IntegratorGUI._clear_log is IntegratorGUILogMixin._clear_log
    assert (
        gui.IntegratorGUI._load_verbose_logs is IntegratorGUILogMixin._load_verbose_logs
    )


def test_gui_inherits_shell_helpers_from_shell_mixin():
    assert gui.IntegratorGUI._load_preferences is (
        IntegratorGUIShellMixin._load_preferences
    )
    assert gui.IntegratorGUI._save_preferences is (
        IntegratorGUIShellMixin._save_preferences
    )
    assert gui.IntegratorGUI._reset_directories_to_defaults is (
        IntegratorGUIShellMixin._reset_directories_to_defaults
    )
    assert gui.IntegratorGUI._on_close is IntegratorGUIShellMixin._on_close
    assert gui.IntegratorGUI._check_override_warning is (
        IntegratorGUIShellMixin._check_override_warning
    )
    assert gui.IntegratorGUI._setup_keyboard_fix is (
        IntegratorGUIShellMixin._setup_keyboard_fix
    )


def test_load_config_no_longer_requires_removed_legacy_state(tmp_path, monkeypatch):
    filename = "example.json"
    (tmp_path / filename).write_text("{}", encoding="utf-8")
    loaded_options = object()
    calls = []
    combo = _ComboBoxStub(["CONDUCTING_WALL"])

    monkeypatch.setattr(
        "lw_integrator.gui_config_mixins.load_config", lambda path: loaded_options
    )

    harness = SimpleNamespace(
        _selected_config_filename=lambda: filename,
        config_dir_var=SimpleNamespace(get=lambda: str(tmp_path)),
        root=SimpleNamespace(update_idletasks=lambda: calls.append("idle")),
        _apply_options_to_ui=lambda options, preserve_directories=False: calls.append(
            ("apply", options, preserve_directories)
        ),
        config_name_var=SimpleNamespace(
            set=lambda value: calls.append(("config_name", value))
        ),
        config_file_var=SimpleNamespace(
            set=lambda value: calls.append(("config_file", value))
        ),
        run_mode_var=SimpleNamespace(
            set=lambda value: calls.append(("run_mode", value))
        ),
        _on_run_mode_changed=lambda: calls.append("run_mode_changed"),
        _refresh_config_list=lambda selected=None: calls.append(
            ("refresh_config_list", selected)
        ),
        _refresh_initial_summary=lambda: calls.append("refresh_initial_summary"),
        _update_driver_visibility=lambda: calls.append("update_driver_visibility"),
        _update_image_subcharge_state=lambda: calls.append(
            "update_image_subcharge_state"
        ),
        _update_cavity_spacing_state=lambda: calls.append(
            "update_cavity_spacing_state"
        ),
        _toggle_z_cutoff_controls=lambda: calls.append("toggle_z_cutoff_controls"),
        _toggle_macroparticle_controls=lambda: calls.append(
            "toggle_macroparticle_controls"
        ),
        _update_macroparticle_state=lambda: calls.append("update_macroparticle_state"),
        sim_type_var=SimpleNamespace(get=lambda: "CONDUCTING_WALL"),
        sim_type_combo=combo,
        _set_status=lambda message: calls.append(("status", message)),
        current_config_label=SimpleNamespace(
            config=lambda **kwargs: calls.append(("label", kwargs))
        ),
    )

    gui.IntegratorGUI._load_config(harness)

    assert ("apply", loaded_options, True) in calls
    assert ("refresh_config_list", filename) in calls
    assert ("run_mode", "single") in calls
    assert "run_mode_changed" in calls
    assert "update_driver_visibility" in calls
    assert combo.current_calls == [0]
    assert ("status", f"Loaded config: {filename}") in calls


@pytest.mark.parametrize("mode", ["blind_sweep", "sweep", "optimization"])
def test_load_config_routes_sweep_and_optimization_configs_to_sweep_tab(
    tmp_path, monkeypatch, mode
):
    filename = f"{mode}.json"
    (tmp_path / filename).write_text(f'{{"mode": "{mode}"}}', encoding="utf-8")
    calls = []
    loaded_paths = []

    def _single_run_loader(path):
        raise AssertionError(f"single-run loader should not receive {path}")

    monkeypatch.setattr(
        "lw_integrator.gui_config_mixins.load_config", _single_run_loader
    )

    harness = SimpleNamespace(
        _selected_config_filename=lambda: filename,
        config_dir_var=_Var(str(tmp_path)),
        root=object(),
        optimization_tab=SimpleNamespace(
            _load_config_from_path=lambda path: loaded_paths.append(path),
            sweep_config_dir=None,
        ),
        sweep_config_name_var=_Var(""),
        sweep_config_dir_var=_Var(""),
        run_mode_var=_Var("single"),
        _on_run_mode_changed=lambda: calls.append("run_mode_changed"),
        _refresh_sweep_config_list=lambda selected=None: calls.append(
            ("refresh_sweep_config_list", selected)
        ),
        current_sweep_config_label=SimpleNamespace(
            config=lambda **kwargs: calls.append(("sweep_label", kwargs))
        ),
        _set_status=lambda message: calls.append(("status", message)),
    )
    harness._load_sweep_or_optimization_config = (
        lambda path: gui.IntegratorGUI._load_sweep_or_optimization_config(harness, path)
    )

    gui.IntegratorGUI._load_config(harness)

    assert loaded_paths == [str(tmp_path / filename)]
    assert harness.sweep_config_name_var.get() == filename
    assert harness.sweep_config_dir_var.get() == str(tmp_path)
    assert harness.optimization_tab.sweep_config_dir == str(tmp_path)
    assert harness.run_mode_var.get() == "sweep"
    assert "run_mode_changed" in calls
    assert ("refresh_sweep_config_list", filename) in calls
    assert ("status", f"Loaded sweep/optimization config: {filename}") in calls


def test_save_config_normalizes_filename_and_updates_ui(tmp_path, monkeypatch):
    saved = {}
    info_calls = []
    options = SimpleNamespace(config_name="placeholder.json")

    monkeypatch.setattr(
        "lw_integrator.gui_config_mixins.save_config",
        lambda opts, path: saved.update({"options": opts, "path": path}),
    )
    monkeypatch.setattr(
        "lw_integrator.gui_config_mixins.messagebox.showinfo",
        lambda title, message: info_calls.append((title, message)),
    )

    calls = []
    harness = SimpleNamespace(
        _build_options_from_ui=lambda: options,
        root=object(),
        config_name_var=_Var("my_run"),
        config_file_var=_Var(""),
        config_dir_var=_Var(str(tmp_path)),
        _check_override_warning=lambda path, config_type="run": True,
        _refresh_config_list=lambda selected=None: calls.append(
            ("refresh_config_list", selected)
        ),
        current_config_label=SimpleNamespace(
            config=lambda **kwargs: calls.append(("label", kwargs))
        ),
        _set_status=lambda message: calls.append(("status", message)),
    )

    gui.IntegratorGUI._save_config(harness)

    assert options.config_name == "my_run.json"
    assert saved["options"] is options
    assert saved["path"] == Path(tmp_path) / "my_run.json"
    assert harness.config_name_var.get() == "my_run.json"
    assert harness.config_file_var.get() == "my_run.json"
    assert ("refresh_config_list", "my_run.json") in calls
    assert ("status", "Saved config: my_run.json") in calls
    assert ("Save Run Config", "Configuration saved as my_run.json") in info_calls


def test_trigger_cancel_updates_run_state_feedback():
    calls = []
    harness = SimpleNamespace(
        _running=True,
        _cancel_requested=False,
        _cancel_button=SimpleNamespace(
            configure=lambda **kwargs: calls.append(("cancel_button", kwargs))
        ),
        _append_log=lambda message: calls.append(("log", message)),
        _set_status=lambda message: calls.append(("status", message)),
    )

    gui.IntegratorGUI._trigger_cancel(harness)

    assert harness._cancel_requested is True
    assert ("cancel_button", {"state": "disabled"}) in calls
    assert ("log", "Cancellation requested...") in calls
    assert ("status", "Cancelling...") in calls


def test_on_cancelled_restores_ready_controls():
    run_button_calls = []
    cancel_button_calls = []
    status_calls = []
    progress_values = []
    harness = SimpleNamespace(
        _running=True,
        _worker=object(),
        _cancel_requested=True,
        _set_status=lambda message: status_calls.append(message),
        _append_log=lambda message: status_calls.append(f"log:{message}"),
        _run_button=SimpleNamespace(
            configure=lambda **kwargs: run_button_calls.append(kwargs)
        ),
        _cancel_button=SimpleNamespace(
            configure=lambda **kwargs: cancel_button_calls.append(kwargs)
        ),
        progress_var=SimpleNamespace(set=lambda value: progress_values.append(value)),
    )

    gui.IntegratorGUI._on_cancelled(harness)

    assert harness._running is False
    assert harness._worker is None
    assert harness._cancel_requested is False
    assert "Cancelled" in status_calls
    assert "log:Simulation cancelled by user." in status_calls
    assert run_button_calls == [{"state": "normal"}]
    assert cancel_button_calls == [{"state": "disabled"}]
    assert progress_values == [0.0]


def test_prepare_figure_for_display_scales_large_figures():
    calls = []

    class _FigureStub:
        def get_dpi(self):
            return 100.0

        def get_size_inches(self):
            return (40.0, 20.0)

        def set_size_inches(self, width, height, forward=False):
            calls.append(("size", width, height, forward))

    harness = SimpleNamespace(
        _scale_figure_visuals=lambda figure, scale: calls.append(("scale", scale))
    )

    width_px, height_px = gui.IntegratorGUI._prepare_figure_for_display(
        harness, _FigureStub()
    )

    assert (width_px, height_px) == (1600, 800)
    assert ("size", 16.0, 8.0, False) in calls
    assert ("scale", 0.4) in calls


def test_close_figure_removes_handle_and_destroys_widgets():
    destroyed = []

    class _CanvasWidget:
        def destroy(self):
            destroyed.append("canvas")

    handle = SimpleNamespace(
        canvas=SimpleNamespace(get_tk_widget=lambda: _CanvasWidget()),
        window=SimpleNamespace(destroy=lambda: destroyed.append("window")),
    )
    harness = SimpleNamespace(_figure_windows=[handle])

    gui.IntegratorGUI._close_figure(harness, handle)

    assert harness._figure_windows == []
    assert destroyed == ["canvas", "window"]


def test_format_summary_includes_driver_block_when_present():
    summary = SimpleNamespace(
        seed=123,
        rider_gamma=10.0,
        rider_rest_mev=1.0,
        rider_rest_gev=0.001,
        rider_total_gev=2.5,
        rider_emittance_x_mm_mrad=None,
        rider_emittance_y_mm_mrad=None,
        rider_norm_emittance_x_mm_mrad=None,
        rider_norm_emittance_y_mm_mrad=None,
        rider_beta_x_m=None,
        rider_beta_y_m=None,
        supports_driver=True,
        has_driver=True,
        driver_gamma=20.0,
        driver_rest_mev=2.0,
        driver_rest_gev=0.002,
        driver_total_gev=3.5,
        driver_emittance_x_mm_mrad=None,
        driver_emittance_y_mm_mrad=None,
        driver_norm_emittance_x_mm_mrad=None,
        driver_norm_emittance_y_mm_mrad=None,
        driver_beta_x_m=None,
        driver_beta_y_m=None,
    )

    formatted = gui.IntegratorGUI._format_summary(SimpleNamespace(), summary)

    assert "Seed: 123" in formatted
    assert "Rider gamma: 10.0000" in formatted
    assert "Driver present" in formatted
    assert "Driver gamma: 20.0000" in formatted


def test_update_driver_visibility_disables_driver_fields_for_non_driver_modes():
    driver_combo_calls = []
    driver_entry_calls = []
    rider_offset_calls = []
    driver_offset_calls = []
    rider_label_calls = []
    driver_label_calls = []
    driver_species_var = _Var("not-custom")

    harness = SimpleNamespace(
        sim_type_var=_Var("CONDUCTING_WALL"),
        driver_species_combo=SimpleNamespace(
            configure=lambda **kwargs: driver_combo_calls.append(kwargs)
        ),
        _driver_entries=[
            SimpleNamespace(
                configure=lambda **kwargs: driver_entry_calls.append(kwargs)
            )
        ],
        _species_label_by_key={"custom": "Custom"},
        _species_by_label={"Custom": object()},
        driver_species_var=driver_species_var,
        _rider_offset_entries=[
            SimpleNamespace(
                configure=lambda **kwargs: rider_offset_calls.append(kwargs)
            )
        ],
        _driver_offset_entries=[
            SimpleNamespace(
                configure=lambda **kwargs: driver_offset_calls.append(kwargs)
            )
        ],
        _rider_offset_labels=[
            SimpleNamespace(configure=lambda **kwargs: rider_label_calls.append(kwargs))
        ],
        _driver_offset_labels=[
            SimpleNamespace(
                configure=lambda **kwargs: driver_label_calls.append(kwargs)
            )
        ],
    )

    gui.IntegratorGUI._update_driver_visibility(harness)

    assert driver_combo_calls == [{"state": "disabled"}]
    assert driver_entry_calls == [{"state": "disabled"}]
    assert driver_species_var.get() == "Custom"
    assert rider_offset_calls == [{"state": "disabled"}]
    assert driver_offset_calls == [{"state": "disabled"}]
    assert rider_label_calls == [{"foreground": "gray60"}]
    assert driver_label_calls == [{"foreground": "gray60"}]


def test_enforce_panel_minimums_clamps_sash_position():
    placements = []

    class _PanedStub:
        def sash_coord(self, index):
            assert index == 0
            return (100, 0)

        def winfo_width(self):
            return 1000

        def sash_place(self, index, x, y):
            placements.append((index, x, y))

    harness = SimpleNamespace(_main_horizontal_paned=_PanedStub())

    gui.IntegratorGUI._enforce_panel_minimums(harness)

    assert placements == [(0, 800, 0)]


def test_on_run_mode_changed_updates_run_button_command():
    calls = []
    harness = SimpleNamespace(
        run_mode_var=_Var("sweep"),
        _trigger_run=object(),
        _trigger_sweep=object(),
        _run_button=SimpleNamespace(config=lambda **kwargs: calls.append(kwargs)),
    )

    gui.IntegratorGUI._on_run_mode_changed(harness)

    assert calls == [
        {"text": "▶ Run Sweep", "command": harness._trigger_sweep},
    ]


def test_clear_log_resets_buffers_and_widget():
    calls = []
    harness = SimpleNamespace(
        _raw_log_lines=["a"],
        _log_summary=["b"],
        log_output=SimpleNamespace(
            configure=lambda **kwargs: calls.append(("configure", kwargs)),
            delete=lambda start, end: calls.append(("delete", start, end)),
        ),
    )

    gui.IntegratorGUI._clear_log(harness)

    assert harness._raw_log_lines == []
    assert harness._log_summary == []
    assert calls == [
        ("configure", {"state": "normal"}),
        ("delete", "1.0", "end"),
        ("configure", {"state": "disabled"}),
    ]


def test_check_override_warning_allows_missing_file(tmp_path):
    harness = SimpleNamespace(_suppress_override_warning=False)

    assert gui.IntegratorGUI._check_override_warning(harness, tmp_path / "new.json")


def test_toggle_z_cutoff_controls_disables_widgets_and_resets_cutoff():
    entry_calls = []
    combo_calls = []
    z_cutoff_var = _Var(12.0)
    harness = SimpleNamespace(
        z_cutoff_enabled_var=_Var(False),
        z_cutoff_entry=SimpleNamespace(
            configure=lambda **kwargs: entry_calls.append(kwargs)
        ),
        z_cutoff_mode_combo=SimpleNamespace(
            configure=lambda **kwargs: combo_calls.append(kwargs)
        ),
        core_param_vars={"z_cutoff": z_cutoff_var},
    )

    gui.IntegratorGUI._toggle_z_cutoff_controls(harness)

    assert entry_calls == [{"state": "disabled"}]
    assert combo_calls == [{"state": "disabled"}]
    assert z_cutoff_var.get() == 0.0


def test_toggle_external_field_controls_respects_enable_and_input_mode():
    calls = {}

    class _Widget:
        def __init__(self, name):
            self.name = name

        def configure(self, **kwargs):
            calls.setdefault(self.name, []).append(kwargs)

    si_label = _Widget("si_label")
    si_entry = _Widget("si_entry")
    native_label = _Widget("native_label")
    native_entry = _Widget("native_entry")
    combo = _Widget("combo")
    magnetic_entry = _Widget("magnetic_entry")
    window_entry = _Widget("window_entry")
    harness = SimpleNamespace(
        external_field_enabled_var=_Var(True),
        external_field_input_mode_var=_Var("SI V/m"),
        external_field_input_mode_combo=combo,
        external_electric_si_label=si_label,
        external_electric_si_entries=[si_entry],
        external_electric_native_label=native_label,
        external_electric_native_entries=[native_entry],
        _external_field_sub_widgets=[
            combo,
            si_label,
            si_entry,
            native_label,
            native_entry,
            magnetic_entry,
            window_entry,
        ],
    )

    gui.IntegratorGUI._toggle_external_field_controls(harness)

    assert calls["combo"] == [{"state": "normal"}]
    assert calls["si_label"] == [{"state": "normal"}, {"state": "normal"}]
    assert calls["si_entry"] == [{"state": "normal"}, {"state": "normal"}]
    assert calls["native_label"] == [{"state": "normal"}, {"state": "disabled"}]
    assert calls["native_entry"] == [{"state": "normal"}, {"state": "disabled"}]
    assert calls["magnetic_entry"] == [{"state": "normal"}]
    assert calls["window_entry"] == [{"state": "normal"}]

    calls.clear()
    harness.external_field_enabled_var.set(False)

    gui.IntegratorGUI._toggle_external_field_controls(harness)

    assert calls["combo"] == [{"state": "disabled"}]
    assert calls["si_entry"] == [{"state": "disabled"}]
    assert calls["native_entry"] == [{"state": "disabled"}]


def test_build_options_tolerates_invalid_external_field_inputs_when_disabled():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    root.withdraw()
    try:
        app = gui.IntegratorGUI(root)
        app.external_field_enabled_var.set(False)
        app.external_field_input_mode_var.set("SI V/m")

        for var in (
            app.external_electric_native_vars
            + app.external_electric_si_vars
            + app.external_magnetic_native_vars
        ):
            var.set("not-a-number")

        for var in app.external_field_window_vars.values():
            var.set("bad-bound")

        options = app._build_options_from_ui()

        assert options.external_field_enabled is False
        assert options.external_electric_field_native == pytest.approx((0.0, 0.0, 0.0))
        assert options.external_electric_field_v_per_m == pytest.approx((0.0, 0.0, 0.0))
        assert options.external_magnetic_field_native == pytest.approx((0.0, 0.0, 0.0))
        assert options.external_field_x_min is None
        assert options.external_field_t_max is None
    finally:
        root.destroy()


def test_radiation_reaction_mode_round_trips_through_gui_options():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    root.withdraw()
    try:
        app = gui.IntegratorGUI(root)
        app._apply_options_to_ui(
            SimulationOptions(radiation_reaction_mode="medina_lad"),
            preserve_directories=True,
        )

        assert app.radiation_reaction_mode_var.get() == "medina_lad"

        app.radiation_reaction_mode_var.set("power_matched_damping")
        rebuilt = app._build_options_from_ui()

        assert rebuilt.radiation_reaction_mode == "power_matched_damping"
    finally:
        root.destroy()


def test_update_macroparticle_state_forces_disabled_outside_conducting_wall():
    check_calls = []
    entry_calls = []
    label_calls = []
    enabled_var = _Var(True)
    harness = SimpleNamespace(
        sim_type_var=_Var("BUNCH_TO_BUNCH"),
        macroparticle_enable_check=SimpleNamespace(
            configure=lambda **kwargs: check_calls.append(kwargs)
        ),
        macroparticle_enabled_var=enabled_var,
        _macroparticle_widgets=[
            SimpleNamespace(configure=lambda **kwargs: entry_calls.append(kwargs)),
            SimpleNamespace(configure=lambda **kwargs: label_calls.append(kwargs)),
        ],
    )

    # Attach concrete ttk-ish classes by monkeypatching isinstance checks through subclassing
    import tkinter.ttk as ttk

    class _Entry(ttk.Entry):
        pass

    class _Label(ttk.Label):
        pass

    harness._macroparticle_widgets = [
        _Entry.__new__(_Entry),
        _Label.__new__(_Label),
    ]
    harness._macroparticle_widgets[0].configure = lambda **kwargs: entry_calls.append(
        kwargs
    )
    harness._macroparticle_widgets[1].configure = lambda **kwargs: label_calls.append(
        kwargs
    )

    gui.IntegratorGUI._update_macroparticle_state(harness)

    assert check_calls == [{"state": "disabled"}]
    assert enabled_var.get() is False
    assert entry_calls == [{"state": "disabled"}]
    assert label_calls == [{"foreground": "gray"}]


def test_selected_config_filename_returns_selected_value():
    config_list = SimpleNamespace(
        curselection=lambda: (0,),
        get=lambda index: "demo.json" if index == 0 else None,
    )
    harness = SimpleNamespace(config_list=config_list)

    assert gui.IntegratorGUI._selected_config_filename(harness) == "demo.json"


def test_config_selection_populates_save_target_name():
    calls = []
    harness = SimpleNamespace(
        _selected_config_filename=lambda: "demo.json",
        config_name_var=_Var("old.json"),
        config_file_var=_Var("old.json"),
        current_config_label=SimpleNamespace(
            config=lambda **kwargs: calls.append(("label", kwargs))
        ),
    )

    gui.IntegratorGUI._on_config_selected(harness)

    assert harness.config_name_var.get() == "demo.json"
    assert harness.config_file_var.get() == "demo.json"
    assert calls == [("label", {"text": "demo.json", "foreground": "black"})]


def test_load_sweep_config_uses_entry_value_and_normalizes_extension(tmp_path):
    loaded_paths = []
    sweep_file = tmp_path / "example.json"
    sweep_file.write_text("{}", encoding="utf-8")
    label_calls = []
    run_mode_changes = []
    harness = SimpleNamespace(
        sweep_config_name_var=_Var("example"),
        optimization_tab=SimpleNamespace(
            _load_config_from_path=lambda path: loaded_paths.append(path)
        ),
        sweep_config_dir_var=_Var(str(tmp_path)),
        run_mode_var=_Var("single"),
        _on_run_mode_changed=lambda: run_mode_changes.append("changed"),
        current_sweep_config_label=SimpleNamespace(
            config=lambda **kwargs: label_calls.append(kwargs)
        ),
    )

    gui.IntegratorGUI._load_sweep_config(harness)

    assert loaded_paths == [str(sweep_file)]
    assert harness.sweep_config_name_var.get() == "example.json"
    assert harness.run_mode_var.get() == "sweep"
    assert run_mode_changes == ["changed"]
    assert label_calls == [
        {"text": "example.json", "foreground": "black", "font": ("TkDefaultFont", 9)}
    ]


def test_save_sweep_config_normalizes_name_and_refreshes_list(tmp_path, monkeypatch):
    info_calls = []
    saved_paths = []
    refresh_calls = []
    label_calls = []
    monkeypatch.setattr(
        "lw_integrator.gui_config_list_mixins.messagebox.showinfo",
        lambda title, message: info_calls.append((title, message)),
    )

    harness = SimpleNamespace(
        optimization_tab=SimpleNamespace(
            _save_config_to_path=lambda path: saved_paths.append(path) or True
        ),
        sweep_config_name_var=_Var("demo_sweep"),
        sweep_config_dir_var=_Var(str(tmp_path)),
        _check_override_warning=lambda path, config_type="sweep": True,
        current_sweep_config_label=SimpleNamespace(
            config=lambda **kwargs: label_calls.append(kwargs)
        ),
        _refresh_sweep_config_list=lambda selected=None: refresh_calls.append(selected),
    )

    gui.IntegratorGUI._save_sweep_config(harness)

    assert saved_paths == [str(tmp_path / "demo_sweep.json")]
    assert harness.sweep_config_name_var.get() == "demo_sweep.json"
    assert refresh_calls == ["demo_sweep.json"]
    assert label_calls == [
        {
            "text": "demo_sweep.json",
            "foreground": "black",
            "font": ("TkDefaultFont", 9),
        }
    ]
    assert info_calls == [
        ("Save Sweep Config", "Configuration saved as demo_sweep.json")
    ]


def test_apply_species_updates_particle_vars_and_refreshes_summary():
    rider_param_vars = {
        "m_particle": _Var(1.0),
        "charge_sign": _Var(-1.0),
        "pcount": _Var(1),
        "energy": _Var(2.0),
        "transv_mom": _Var(0.0),
        "transv_dist": _Var(0.0),
        "stripped_ions": _Var(0.0),
    }
    # ensure all expected fields exist without hardcoding the full set
    from lw_integrator.testbed_runner import PARTICLE_PARAM_FIELDS

    for field in PARTICLE_PARAM_FIELDS:
        rider_param_vars.setdefault(field, _Var(0.0))

    refresh_calls = []
    harness = SimpleNamespace(
        rider_species_var=_Var("Muon"),
        driver_species_var=_Var("Custom"),
        _species_by_label={"Muon": "muon", "Custom": "custom"},
        rider_param_vars=rider_param_vars,
        driver_param_vars={field: _Var(0.0) for field in PARTICLE_PARAM_FIELDS},
        _refresh_initial_summary=lambda: refresh_calls.append("refresh"),
    )

    gui.IntegratorGUI._apply_species(harness, "rider")

    assert refresh_calls == ["refresh"]
    assert any(var.get() != 0.0 for var in rider_param_vars.values())


def test_trigger_sweep_delegates_to_optimization_tab(monkeypatch):
    run_calls = []
    monkeypatch.setattr(
        "lw_integrator.gui_controller_mixins.messagebox.askyesno",
        lambda *args, **kwargs: True,
    )

    harness = SimpleNamespace(
        optimization_tab=SimpleNamespace(
            last_loaded_config=None,
            _on_run_sweep=lambda: run_calls.append("run"),
        )
    )

    gui.IntegratorGUI._trigger_sweep(harness)

    assert run_calls == ["run"]
