import json

import numpy as np
import pytest

from core import coupled_dipole_reaction as coupled
from core import momentum_center as m
from core import momentum_center_pair as pair
from core.constants import C_MMNS as c
from core.full_dipole_history import FullDipoleHistory


def test_coupled_native_coasting_and_restart():
    particles = [
        pair.MomentumCenterParticle(
            0, 1, reaction_mode="full_dipole_coupled", reaction_window_ns=0.025 / c
        )
        for _ in range(2)
    ]
    past = np.linspace(-2, 0, 81) / c
    histories = [
        FullDipoleHistory(
            past,
            np.tile([i, 0, 0], (len(past), 1)),
            np.zeros((len(past), 3)),
            np.zeros((len(past), 4, 4)),
            c,
        ).completed()
        for i in (-0.5, 0.5)
    ]

    def zero(t, x):
        return np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4, 4))

    states = [
        pair.initial_state_native(
            [0, i, 0, 0], [1, 0, 0, 0], c * np.array([0.1, 0.2, 0.3]), p, zero
        )
        for i, p in zip((-0.5, 0.5), particles)
    ]
    payload = pair.initialize_pair(
        particles, states, histories, startup_duration_ns=0.5 / c
    )
    assert payload["startup_smoothness"] == 7
    first, records = pair.advance_pair(payload, 0.01 / c)
    resumed, _ = pair.advance_pair(json.loads(json.dumps(first)), 0.01 / c)
    together, _ = pair.advance_pair(payload, 0.01 / c, 2)
    assert resumed == together
    assert payload["accepted_steps"] == 0
    assert first["particles"][0]["reaction_mode"] == "full_dipole_coupled"
    assert (
        records[0]["particles"][0]["length_time"]["reaction"]["mode"]
        == "full_dipole_coupled"
    )
    assert "dipole_reaction_ledger" in first


def test_coupled_method_rejects_legacy_derivatives():
    with pytest.raises(ValueError, match="centered"):
        pair.MomentumCenterParticle(
            1,
            1,
            reaction_mode="full_dipole_coupled",
            reaction_window_ns=0.1,
            reaction_derivative_method="backward",
        )


def test_unconverged_root_never_supplies_force(monkeypatch):
    calls = []
    monkeypatch.setattr(
        coupled, "ordinary_source_coefficients", lambda *a, **k: (None, None, {})
    )

    def generated(*a, **k):
        calls.append(1)
        return dict(
            field=np.ones((4, 4)) * len(calls), gradient=np.ones((4, 4, 4)) * len(calls)
        )

    monkeypatch.setattr(coupled, "regular_self_field", generated)
    monkeypatch.setattr(
        coupled,
        "solve_point",
        lambda *a, **k: (
            dict(converged=False, evaluations=1, relative_residuals=[0.1, 0.2]),
            None,
            None,
        ),
    )
    with pytest.raises(ValueError, match="residual failed"):
        coupled.evaluate_coupled_reaction(
            np.zeros(14), m.Particle(), None, window=0.025
        )
