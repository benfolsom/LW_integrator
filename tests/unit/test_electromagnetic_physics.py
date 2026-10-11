"""Targeted unit tests for electromagnetic helper utilities."""

from __future__ import annotations

import numpy as np
import pytest

from core.distances import compute_instantaneous_distance


@pytest.mark.unit
def test_compute_instantaneous_distance_returns_unit_direction() -> None:
    source = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}
    target = {"x": np.array([3.0]), "y": np.array([4.0]), "z": np.array([0.0])}

    result = compute_instantaneous_distance(source, target, 0)

    assert result["R"][0] == pytest.approx(5.0)
    length = np.sqrt(result["nx"][0] ** 2 + result["ny"][0] ** 2 + result["nz"][0] ** 2)
    assert length == pytest.approx(1.0)
