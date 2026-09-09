import json

import pytest

from core.momentum_center import VelocityDomainError
from lw_integrator import nonlinear_pair as cli


def test_capabilities_are_explicit(capsys):
    assert cli.main(["--capabilities"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["supported_radiation_reaction"] == ["off", "charge_ll"]
    assert data["production_default"] is False


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


@pytest.mark.parametrize("selection", [None, "off", "charge_ll"])
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
    assert cli.main(args) == 0
    assert json.loads(source.read_text()) == original
    assert all(
        p["reaction_mode"] == (selection or "charge_ll")
        for p in json.loads(output.read_text())["particles"]
    )
