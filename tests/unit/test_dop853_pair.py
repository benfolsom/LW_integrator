"""Higher-order state stepping must also integrate every reaction ledger."""

import copy
import json

import numpy as np
import pytest

from core import momentum_center_pair as pair
from core import momentum_center as model
from core.preserved_runge_kutta import dop853_step
from core.preserved_source import rounded_state
from lw_integrator.nonlinear_pair import configure_checkpoint
from tests.unit.test_preserved_source import fresh_pair


def test_preserved_quartic_integral():
    ref = np.array([0.0, 1.0])
    high, low = np.zeros(2), np.zeros(2)
    for i in range(10):
        high, low, weights = dop853_step(
            ref, high, low, 1 / 1024, lambda y: np.array([1.0, 5 * y[0] ** 4])
        )
        high[0], low[0] = (i + 1) / 1024, 0.0
    assert len(weights) == 12
    assert abs((high[1] + low[1]) / (10 / 1024) ** 5 - 1) < 1e-13
    assert rounded_state(ref, high, low)[1] > 1


@pytest.mark.parametrize("mode", ["off", "full_dipole_coupled"])
def test_dop853_restart_and_failure_atomicity(mode):
    payload = configure_checkpoint(fresh_pair(mode), integration_method="dop853")
    original = copy.deepcopy(payload)
    whole, _ = pair.advance_pair(payload, 0.02 / pair.c, 2)
    middle, _ = pair.advance_pair(payload, 0.02 / pair.c, 1)
    resumed, _ = pair.advance_pair(json.loads(json.dumps(middle)), 0.02 / pair.c, 1)
    assert resumed == whole and payload == original
    assert whole["integration_method"] == "dop853"
    with pytest.raises(ValueError, match="Cannot change integration method"):
        configure_checkpoint(middle, integration_method="rk4")
    with pytest.raises(ValueError):
        pair.advance_pair(payload, 4 / pair.c, 1)
    assert payload == original


def test_all_full_reaction_ledgers_use_dop853_weights(monkeypatch):
    payload = configure_checkpoint(
        fresh_pair("full_dipole_coupled"), integration_method="dop853"
    )
    ordinary = pair.dynamics_native

    def supplied(state, particle, provider):
        rate, data = ordinary(state, particle, provider)
        factor = data["length_time"]["proper_velocity"][0] * state[0] ** 4
        reaction = data["length_time"]["reaction"]
        reaction["force"] = 6 * factor * np.arange(1.0, 5.0)
        reaction["torque"] = model.unpack(6 * factor * np.arange(1.0, 7.0))
        reaction["external_action_mass_rate"] = factor
        for multiplier, key in enumerate(("q_squared", "qD", "D_squared"), 1):
            reaction["sectors"][key]["force"] = (
                multiplier * factor * np.arange(1.0, 5.0)
            )
            reaction["sectors"][key]["torque"] = model.unpack(
                multiplier * factor * np.arange(1.0, 7.0)
            )
        return rate, data

    monkeypatch.setattr(pair, "dynamics_native", supplied)
    width = 0.02 / pair.c
    final, _ = pair.advance_pair(payload, width, 1)
    scale = pair.c**2 * width**5 / 5
    for row in final["dipole_reaction_ledger"]:
        np.testing.assert_allclose(
            row["applied_impulse_native"],
            6 * scale * np.arange(1.0, 5.0),
            rtol=1e-13,
            atol=0,
        )
        np.testing.assert_allclose(
            row["applied_torque_native"],
            6 * scale * np.arange(1.0, 7.0),
            rtol=1e-13,
            atol=0,
        )
        assert (
            abs(row["external_action_mass_exchange"] / (pair.c * width**5 / 5) - 1)
            < 1e-13
        )
        for multiplier, key in enumerate(("q_squared", "qD", "D_squared"), 1):
            np.testing.assert_allclose(
                row["sectors"][key]["impulse_native"],
                multiplier * scale * np.arange(1.0, 5.0),
                rtol=1e-13,
                atol=0,
            )
            np.testing.assert_allclose(
                row["sectors"][key]["torque_native"],
                multiplier * scale * np.arange(1.0, 7.0),
                rtol=1e-13,
                atol=0,
            )


def test_missing_precision_and_unknown_method_rejected():
    payload = fresh_pair()
    with pytest.raises(ValueError, match="Integration method"):
        pair.advance_pair(dict(payload, integration_method="guess"), 0.01 / pair.c)
    del payload["source_precision"]
    with pytest.raises(ValueError, match="preserved source"):
        configure_checkpoint(payload, integration_method="dop853")
