"""Focused GUI regression tests for lightweight controller paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lw_integrator import gui
from lw_integrator.gui_config_mixins import IntegratorGUIConfigMixin
from lw_integrator.gui_plot_mixins import IntegratorGUIPlotMixin
from lw_integrator.gui_runtime_mixins import IntegratorGUIRuntimeMixin


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


def test_load_config_no_longer_requires_removed_legacy_state(
    tmp_path, monkeypatch
):
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
        run_mode_var=SimpleNamespace(set=lambda value: calls.append(("run_mode", value))),
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
        _update_macroparticle_state=lambda: calls.append(
            "update_macroparticle_state"
        ),
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
    assert "update_driver_visibility" in calls
    assert combo.current_calls == [0]
    assert ("status", f"Loaded config: {filename}") in calls


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
