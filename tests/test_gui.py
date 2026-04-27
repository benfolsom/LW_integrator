"""Focused GUI regression tests for lightweight controller paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lw_integrator import gui
from lw_integrator.gui_config_mixins import IntegratorGUIConfigMixin


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
