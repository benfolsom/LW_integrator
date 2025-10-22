"""Targeted unit tests for electromagnetic helper utilities."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.distances import compute_instantaneous_distance
from core.trajectory_integrator import LienardWiechertIntegrator


def _minimal_state() -> dict[str, np.ndarray]:
    arr = np.array
    return {
        "x": arr([0.0], dtype=float),
        "y": arr([0.0], dtype=float),
        "z": arr([0.0], dtype=float),
        "t": arr([0.0], dtype=float),
        "Px": arr([0.0], dtype=float),
        "Py": arr([0.0], dtype=float),
        "Pz": arr([100.0], dtype=float),
        "Pt": arr([100.0], dtype=float),
        "gamma": arr([1.0], dtype=float),
        "bx": arr([0.0], dtype=float),
        "by": arr([0.1], dtype=float),
        "bz": arr([0.9], dtype=float),
        "bdotx": arr([0.0], dtype=float),
        "bdoty": arr([0.0], dtype=float),
        "bdotz": arr([0.0], dtype=float),
        "q": arr([1.0], dtype=float),
        "m": arr([1.0], dtype=float),
        "char_time": arr([1e-3], dtype=float),
    }


@pytest.mark.unit
def test_compute_instantaneous_distance_returns_unit_direction() -> None:
    source = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}
    target = {"x": np.array([3.0]), "y": np.array([4.0]), "z": np.array([0.0])}

    result = compute_instantaneous_distance(source, target, 0)

    assert result["R"][0] == pytest.approx(5.0)
    length = np.sqrt(result["nx"][0] ** 2 + result["ny"][0] ** 2 + result["nz"][0] ** 2)
    assert length == pytest.approx(1.0)


@pytest.mark.unit
def test_static_step_does_not_mutate_original_state() -> None:
    integrator = LienardWiechertIntegrator()
    state = _minimal_state()

    updated = integrator.drift_step(1e-3, state, 0)

    # Original state remains unchanged
    assert np.array_equal(state["z"], np.array([0.0]))

    # The propagated state drifts according to the configured velocity
    expected_z = state["z"][0] + state["bz"][0] * C_MMNS * 1e-3
    assert updated["z"][0] == pytest.approx(expected_z)
    assert np.all(updated["bdotx"] == 0.0)
    assert np.all(updated["bdoty"] == 0.0)
    assert np.all(updated["bdotz"] == 0.0)
