from pathlib import Path
from types import SimpleNamespace

import pytest

from lw_integrator import nonlinear_pair_gui as gui


def test_gui_uses_same_cli_and_explicit_analytic_selection():
    command = gui.build_command(
        "in.json", "out.json", 4, 0.01, "full_dipole_coupled", "", "analytic"
    )
    assert command[1:3] == ["-m", "lw_integrator.nonlinear_pair"]
    assert command[-2:] == ["--reaction-derivatives", "analytic"]
    assert "full_dipole_coupled" in command
    assert "--radiation-reaction" not in gui.build_command("in", "out", 1, 0.1)


def test_history_options_use_cli_and_refuse_adaptive_override():
    command = gui.build_command(
        "in", "out", 1, 0.01, history_method="connected_direct", dipole_budget="1e-11"
    )
    assert command[-4:] == [
        "--history-method",
        "connected_direct",
        "--dipole-drift-budget",
        "1e-11",
    ]
    with pytest.raises(ValueError, match="keeps checkpoint settings"):
        gui.build_command(
            "in",
            "out",
            1,
            "",
            controller="adaptive_resume",
            history_method="connected_direct",
        )


def test_gui_higher_order_and_adaptive_resume_commands():
    command = gui.build_command("in", "out", 1, 0.01, integration_method="dop853")
    assert command[-2:] == ["--integration-method", "dop853"]
    command = gui.build_command(
        "adaptive.json", "next.json", 3, "", controller="adaptive_resume"
    )
    assert command[2] == "lw_integrator.nonlinear_pair_adaptive"
    assert command[-2:] == ["--intervals", "3"]
    assert "--step-ns" not in command
    with pytest.raises(ValueError, match="keeps checkpoint settings"):
        gui.build_command("in", "out", 1, 0.01, controller="adaptive_resume")


def test_gui_internal_settings_use_native_runner_without_stacked_controllers():
    command = gui.build_command(
        "in", "out", 2, 0.01, internal_settings="tolerances.json"
    )
    assert command[2] == "lw_integrator.nonlinear_pair"
    assert command[-2:] == ["--internal-error-settings", "tolerances.json"]
    with pytest.raises(ValueError, match="keeps checkpoint settings"):
        gui.build_command(
            "in",
            "out",
            2,
            "",
            controller="adaptive_resume",
            internal_settings="tolerances.json",
        )


def test_launcher_detaches_and_preserves_existing_files(tmp_path, monkeypatch):
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=42)

    monkeypatch.setattr(gui.subprocess, "Popen", popen)
    target = tmp_path / "out.json"
    process, log = gui.launch(["test"], target)
    assert process.pid == 42
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["cwd"] == Path(gui.__file__).resolve().parents[1]
    assert log.exists()
    with pytest.raises(FileExistsError):
        gui.launch(["test"], target)
    target.write_text("original")
    with pytest.raises(FileExistsError):
        gui.launch(["test"], target)
    assert target.read_text() == "original"


def test_tk_window_builds():
    try:
        root = gui.tk.Tk()
    except gui.tk.TclError as error:
        pytest.skip(f"Tk display unavailable: {error}")
    root.withdraw()
    try:
        window = gui.open_pair_window(root)
        window.withdraw()
        root.update_idletasks()
        assert window.winfo_exists()
    finally:
        root.destroy()


def test_single_fit_command_spacing():
    from lw_integrator.nonlinear_pair_gui import build_command

    command = build_command(
        "in",
        "out",
        1,
        0.01,
        history_method="connected_single_fit",
        dipole_budget="1e-11",
        history_spacing="0.02",
    )
    assert command[-2:] == ["--history-sample-spacing-ns", "0.02"]
