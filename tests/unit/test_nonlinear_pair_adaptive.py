"""Physical comparison, interval rollback and controller checkpoint parity."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from core import nonlinear_pair_adaptive as adaptive
from core import momentum_center as model
from core.constants import C_MMNS as c
from core.step_doubling import ErrorScale
from tests.unit.test_preserved_source import fresh_pair


def settings(width, position=1e-8):
    return adaptive.AdaptivePairSettings(
        ErrorScale(position, 0.0),
        ErrorScale(1e-6, 1e-6),
        ErrorScale(1e-6, 1e-6),
        ErrorScale(1e-6, 1e-6),
        ErrorScale(1e-6, 1e-6),
        ErrorScale(1e-6, 1e-6),
        width / 8,
        width,
    )


@pytest.mark.parametrize("beta", [0.0, 0.8, 0.99])
def test_physical_spin_is_in_kinetic_momentum_rest_frame(beta):
    u = np.array([1.0, beta, 0, 0]) / np.sqrt(1 - beta**2)
    rest = np.array([0.2, -0.3, 0.4])
    spin0 = u[1:] @ rest
    spin = np.r_[spin0, rest + u[1:] * spin0 / (1 + u[0])]
    state = np.zeros(14)
    state[8:] = model.pack(model.spin_tensor(u, spin))
    state[4:8] = [100, 10, 20, 30]  # Deliberately unrelated stored P, not used as p.
    data = dict(states=[state.tolist(), state.tolist()])
    records = [
        dict(
            particles=[dict(kinetic_momentum_native=u), dict(kinetic_momentum_native=u)]
        )
    ]
    _, p, s = adaptive._physical_state(data, records)
    np.testing.assert_allclose(p, np.tile(u, (2, 1)), rtol=0, atol=0)
    np.testing.assert_allclose(s, np.tile(rest, (2, 1)), rtol=2e-13, atol=1e-14)


def test_real_coupled_adaptive_restart_and_no_input_mutation():
    payload = dict(fresh_pair("full_dipole_coupled"), integration_method="dop853")
    width = 0.01 / c
    initial = adaptive.initialize_adaptive(payload, settings(width), width)
    original = copy.deepcopy(initial)
    first, reports = adaptive.advance_adaptive_interval(initial)
    assert reports[-1]["assessment"]["accepted"]
    assert first["native_payload"]["accepted_steps"] == 2
    a, _ = adaptive.advance_adaptive_interval(first)
    b, _ = adaptive.advance_adaptive_interval(json.loads(json.dumps(first)))
    assert a == b and initial == original
    assert a["accepted_intervals"] == 2
    assert a["native_payload"]["integration_method"] == "dop853"


def test_rejected_candidates_are_discarded(monkeypatch):
    width = 0.02 / c
    initial = adaptive.initialize_adaptive(fresh_pair(), settings(width, 1e-9), width)
    original = copy.deepcopy(initial)
    actual = adaptive.advance_pair

    def perturbed(payload, step, steps):
        result, records = actual(payload, step, steps)
        if steps == 1:
            result["states"][0][1] += step**2
        return result, records

    monkeypatch.setattr(adaptive, "advance_pair", perturbed)
    accepted, reports = adaptive.advance_adaptive_interval(initial)
    assert len(reports) > 1 and not reports[0]["assessment"]["accepted"]
    assert accepted["rejected_trials"] == len(reports) - 1
    assert initial == original
    assert accepted["native_payload"]["accepted_steps"] == 2


@pytest.mark.parametrize("budget_only", [False, True])
def test_accuracy_refusal_keeps_trial_evidence_and_input(monkeypatch, budget_only):
    width = 0.02 / c
    config = replace(
        settings(width, 1e-9),
        minimum_interval_ns=width / 8 if budget_only else width,
        maximum_attempts=1,
    )
    initial = adaptive.initialize_adaptive(fresh_pair(), config, width)
    original = copy.deepcopy(initial)
    actual = adaptive.advance_pair

    def perturbed(payload, step, steps):
        result, records = actual(payload, step, steps)
        if steps == 1:
            result["states"][0][1] += 1e-4
        return result, records

    monkeypatch.setattr(adaptive, "advance_pair", perturbed)
    message = "trial budget" if budget_only else "minimum interval"
    with pytest.raises(adaptive.AdaptiveAccuracyError, match=message) as caught:
        adaptive.advance_adaptive_interval(initial)
    assert initial == original
    assert len(caught.value.reports) == 1
    assert caught.value.reports[0]["assessment"]["normalized_error"] > 1


def test_separate_impulse_torque_and_exchange_scales():
    def ledger(value):
        return dict(
            dipole_reaction_ledger=[
                dict(
                    applied_impulse_native=[value] * 4,
                    applied_torque_native=[value] * 6,
                    external_action_mass_exchange=value,
                    sectors={
                        key: dict(impulse_native=[value] * 4, torque_native=[value] * 6)
                        for key in ("q_squared", "qD", "D_squared")
                    },
                )
                for _ in range(2)
            ]
        )

    config = adaptive.AdaptivePairSettings(
        *[ErrorScale(v, 0) for v in (1, 1, 1, 2, 4, 8)], 0.01, 0.1
    )
    a, b = adaptive._diagnostics(ledger(0), ledger(1), config)
    np.testing.assert_array_equal(a, 0)
    np.testing.assert_array_equal(b[:32], 0.5)
    np.testing.assert_array_equal(b[32:80], 0.25)
    np.testing.assert_array_equal(b[80:], 0.125)


def test_adaptive_cli_matches_api_and_preserves_input(tmp_path):
    from lw_integrator.nonlinear_pair_adaptive import main

    width = 0.01 / c
    envelope = adaptive.initialize_adaptive(fresh_pair(), settings(width), width)
    source = tmp_path / "source.json"
    output = tmp_path / "accepted.json"
    encoded = json.dumps(envelope)
    source.write_text(encoded)
    expected, _ = adaptive.advance_adaptive_interval(envelope)
    assert main(["--checkpoint", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == expected
    assert source.read_text() == encoded
    with pytest.raises(SystemExit):
        main(["--checkpoint", str(source), "--output", str(source)])
    with pytest.raises(SystemExit):
        main(["--checkpoint", str(source), "--output", str(output)])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_reaction_increments_cannot_be_masked(monkeypatch, invalid):
    monkeypatch.setattr(
        adaptive, "_increments", lambda payload: (np.array([invalid]),) * 3
    )
    with pytest.raises(ValueError, match="must be finite"):
        adaptive._diagnostics({}, {}, settings(0.1))


@pytest.mark.parametrize("ledger", [[], [{}, {}], [np.nan, np.nan]])
def test_invalid_accumulated_ledger_is_not_discarded(ledger):
    width = 0.01 / c
    envelope = adaptive.initialize_adaptive(fresh_pair(), settings(width), width)
    envelope["native_payload"]["dipole_reaction_ledger"] = ledger
    with pytest.raises(ValueError, match="ledger"):
        adaptive.advance_adaptive_interval(envelope)


def test_adaptive_failure_preserves_last_checkpoint(tmp_path, monkeypatch):
    from lw_integrator import nonlinear_pair_adaptive as cli

    width = 0.01 / c
    envelope = adaptive.initialize_adaptive(fresh_pair(), settings(width), width)
    source = tmp_path / "source.json"
    output = tmp_path / "last.json"
    source.write_text(json.dumps(envelope))
    actual = cli.advance_adaptive_interval
    calls = 0

    def fail_second(payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("Controlled second-interval failure")
        return actual(payload)

    monkeypatch.setattr(cli, "advance_adaptive_interval", fail_second)
    assert (
        cli.main(
            ["--checkpoint", str(source), "--output", str(output), "--intervals", "2"]
        )
        == 1
    )
    assert json.loads(output.read_text())["accepted_intervals"] == 1
    assert json.loads(source.read_text()) == envelope
