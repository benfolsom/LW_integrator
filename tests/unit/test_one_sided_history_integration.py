"""One-sided integration values do not make general join derivatives unique."""

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.full_dipole_history import FullDipoleHistory
from core.momentum_center_pair import FullDipoleProvider


def source():
    length_times = np.arange(-20, 21) * 0.02
    times = length_times / c
    position = np.tile([0.5, 0, 0], (len(times), 1))
    dipole = np.zeros((len(times), 4, 4))
    value = 1e-8 * np.maximum(length_times, 0) ** 4
    dipole[:, 1, 2], dipole[:, 2, 1] = value, -value
    return FullDipoleHistory(
        times,
        position,
        np.zeros_like(position),
        dipole,
        c,
        inertial_until=0.0,
        startup_fit="one_sided",
    ).completed()


def test_strict_default_rejects_join_but_integrating_right_limit_is_finite():
    history = source()
    ordinary = FullDipoleProvider(history, 0.0)
    integrating = FullDipoleProvider(history, 0.0, one_sided_derivatives=True)
    events = np.zeros((5, 4))
    events[0] = [1, -0.5, 0, 0]
    events[1, 0] = 1
    with pytest.raises(ValueError, match="smooth segment"):
        ordinary.taylor_response_length_time(events)
    exact = integrating.taylor_response_length_time(events)
    inside = events.copy()
    inside[0, 0] += 1e-8
    nearby = ordinary.taylor_response_length_time(inside)
    for a, b in zip(exact, nearby):
        assert np.isfinite(a).all()
        assert (
            np.linalg.norm(a - b)
            <= 1e-6 * max(np.linalg.norm(a), np.linalg.norm(b)) + 1e-20
        )


def test_nonpreserved_time_balanced_history_roundtrips():
    history = source().with_time_balanced_sampling()
    payload = history.to_checkpoint_payload()
    assert payload["format"] == "full-dipole-history-v5"
    restored = FullDipoleHistory.from_checkpoint_payload(payload)
    assert restored.to_checkpoint_payload() == payload
