import json

import pytest

from core.momentum_center import VelocityDomainError
from lw_integrator import nonlinear_pair as cli


def test_capabilities_are_explicit(capsys):
    assert cli.main(["--capabilities"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["supported_radiation_reaction"] == [
        "off",
        "charge_ll",
        "full_dipole_coupled",
        "full_dipole_rr",
    ]
    assert data["production_default"] is False
    assert data["supported_integration_methods"] == ["rk4", "dop853"]
    assert data["adaptive_runner"]["module"] == "lw_integrator.nonlinear_pair_adaptive"
    assert "charge_ll" not in data["adaptive_runner"]["reaction_modes"]
    assert data["supported_history_methods"] == [
        "endpoint",
        "connected_direct",
        "connected_single_fit",
    ]


def test_dipole_drift_failure_has_specific_guidance():
    from core.connected_dipole_history import SourceDipoleError

    text = cli.failure_recommendation(SourceDipoleError("drift"), {})
    assert "native dipole units" in text
    assert "No clipping" in text


def test_rejected_step_preserves_latest_checkpoint(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text('{"accepted": 0}')

    def advance(payload, width, steps):
        if payload["accepted"] >= 1:
            raise VelocityDomainError(
                "No continuous timelike velocity branch", beta_squared=1.1
            )
        return {"accepted": 1}, [{"time_ns": 0.1}]

    monkeypatch.setattr(cli, "advance_pair", advance)
    assert (
        cli.main(
            [
                "--checkpoint",
                str(source),
                "--output",
                str(output),
                "--step-ns",
                ".1",
                "--steps",
                "2",
            ]
        )
        == 2
    )
    assert json.loads(source.read_text()) == {"accepted": 0}
    assert json.loads(output.read_text()) == {"accepted": 1}
    error = json.loads(capsys.readouterr().err)
    assert error["completed_steps"] == 1
    assert error["beta_squared"] == 1.1


def test_unimplemented_reaction_is_not_silently_enabled(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--checkpoint",
                str(tmp_path / "in.json"),
                "--output",
                str(tmp_path / "out.json"),
                "--step-ns",
                "1",
                "--radiation-reaction",
                "medina_lad",
            ]
        )
    assert not (tmp_path / "out.json").exists()


def test_failed_json_encoding_does_not_replace_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"accepted": 1}')
    with pytest.raises(ValueError):
        cli.write_checkpoint(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"accepted": 1}


@pytest.mark.parametrize("saved_steps", [0, 1])
def test_interrupt_reports_last_save_and_preserves_input(
    tmp_path, monkeypatch, capsys, saved_steps
):
    source, output = tmp_path / "in.json", tmp_path / "out.json"
    encoded = '{"accepted": 0}'
    source.write_text(encoded)

    def advance(payload, width, count):
        if payload["accepted"] == saved_steps:
            raise KeyboardInterrupt
        return {"accepted": 1}, [{"time_ns": width}]

    monkeypatch.setattr(cli, "advance_pair", advance)
    assert (
        cli.main(
            [
                "--checkpoint",
                str(source),
                "--output",
                str(output),
                "--step-ns",
                ".1",
                "--steps",
                "2",
            ]
        )
        == 130
    )
    result = json.loads(capsys.readouterr().err)
    assert result["status"] == "interrupted"
    assert result["completed_steps"] == saved_steps
    assert result["saved_checkpoint"] == (str(output) if saved_steps else None)
    assert source.read_text() == encoded
    if saved_steps:
        assert json.loads(output.read_text()) == {"accepted": 1}
    else:
        assert not output.exists()


@pytest.mark.parametrize(
    "selection", [None, "off", "charge_ll", "full_dipole_rr", "full_dipole_coupled"]
)
def test_reaction_selection_preserves_input_and_is_explicit(
    tmp_path, monkeypatch, selection
):
    source, output = tmp_path / "in.json", tmp_path / "out.json"
    original = {"particles": [{"reaction_mode": "charge_ll"}] * 2}
    source.write_text(json.dumps(original))

    def advance(payload, width, count):
        return payload, [{"time_ns": 0.1}]

    monkeypatch.setattr(cli, "advance_pair", advance)
    args = ["--checkpoint", str(source), "--output", str(output), "--step-ns", ".1"]
    if selection is not None:
        args.extend(["--radiation-reaction", selection])
    if selection in ("full_dipole_rr", "full_dipole_coupled"):
        args.extend(["--reaction-window-ns", ".01"])
    assert cli.main(args) == 0
    assert json.loads(source.read_text()) == original
    assert all(
        p["reaction_mode"] == (selection or "charge_ll")
        for p in json.loads(output.read_text())["particles"]
    )
    if selection in ("full_dipole_rr", "full_dipole_coupled"):
        assert all(
            p["reaction_window_ns"] == 0.01
            for p in json.loads(output.read_text())["particles"]
        )


def test_evolved_checkpoint_cannot_change_equations():
    original = dict(
        accepted_steps=1,
        particles=[dict(reaction_mode="full_dipole_rr", reaction_window_ns=0.01)],
    )
    with pytest.raises(ValueError, match="Cannot change"):
        cli.configure_checkpoint(original, "full_dipole_coupled")
    with pytest.raises(ValueError, match="Cannot change"):
        cli.configure_checkpoint(original, window_ns=0.02)
    assert cli.configure_checkpoint(original, "full_dipole_rr") == original
    assert original["particles"][0]["reaction_mode"] == "full_dipole_rr"


def test_velocity_failure_recommends_coupled_without_switching():
    error = VelocityDomainError(
        "No continuous timelike velocity branch", beta_squared=1.1
    )
    payload = dict(particles=[dict(reaction_mode="full_dipole_rr")])
    assert "full_dipole_coupled" in cli.failure_recommendation(error, payload)
    assert payload["particles"][0]["reaction_mode"] == "full_dipole_rr"


def test_study_schema_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="coupled-study checkpoint"):
        cli.configure_checkpoint(dict(model="study_local_linear_coupled_history_v1"))
