"""One test for the startup-history preflight; no other test covers its decisions."""

import numpy as np
import pytest

from core.constants import C_MMNS
from core.startup_history_guard import StartupHistoryError, check_startup_history
from core.types import SimulationType, StartupMode

B2B = SimulationType.BUNCH_TO_BUNCH


def _state(pos, beta, charge):
    beta = np.asarray(beta, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - beta @ beta)
    return {
        "x": np.array([pos[0]]),
        "y": np.array([pos[1]]),
        "z": np.array([pos[2]]),
        "t": np.zeros(1),
        "Px": np.array([gamma * beta[0] * C_MMNS]),
        "Py": np.array([gamma * beta[1] * C_MMNS]),
        "Pz": np.array([gamma * beta[2] * C_MMNS]),
        "m": np.ones(1),
        "gamma": np.array([gamma]),
        "q": np.array([charge]),
    }


def test_startup_history_guard_flags_missing_history_and_passes_separated_pairs():
    # Head-on, run covers the 1000 mm approach and the encounter: passes.
    rider = _state((0, 0, 0), (0, 0, 0.5), 1.0)
    driver = _state((0, 0.01, 1000), (0, 0, -0.5), 1.0)
    h = 1000.0 / (2 * 0.5 * C_MMNS * 400) * np.sqrt(1 - 0.25)
    kwargs = dict(sim_type=B2B, startup_mode=StartupMode.COLD_START, steps=800)
    assert check_startup_history(rider, driver, h_step=h, **kwargs).passed

    # Slow pair 60 mm apart with a run barely longer than R/c: the interaction
    # starts inside the missing interval, so the default is an error naming the pair.
    slow = _state((0, 0, 0), (0, 0, 0.015), 1.0)
    src = _state((0, 60, 0), (0, 0.06, 0), 1.0)
    short = dict(sim_type=B2B, startup_mode=StartupMode.COLD_START, steps=3000)
    with pytest.raises(StartupHistoryError, match=r"driver\[0\] -> rider\[0\]"):
        check_startup_history(slow, src, h_step=1.9e-3, **short)

    # The explicit override warns loudly instead; INERTIAL_PREHISTORY is not checked.
    with pytest.warns(RuntimeWarning, match="STARTUP-HISTORY OVERRIDE"):
        check_startup_history(slow, src, h_step=1.9e-3, allow_incomplete=True, **short)
    inertial = dict(short, startup_mode=StartupMode.INERTIAL_PREHISTORY)
    assert not check_startup_history(slow, src, h_step=1.9e-3, **inertial).checked
