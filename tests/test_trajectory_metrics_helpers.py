"""Additional direct tests for maintained trajectory metric helpers."""

from __future__ import annotations

import numpy as np

import lw_integrator.trajectory_metrics as trajectory_metrics
from lw_integrator.trajectory_metrics import (
    compute_delta_energy_components,
    extract_series,
)


def _state(*, gamma: float, bz: float, z: float) -> dict[str, np.ndarray]:
    return {
        "gamma": np.array([gamma], dtype=float),
        "bz": np.array([bz], dtype=float),
        "z": np.array([z], dtype=float),
    }


def test_extract_series_returns_scalar_history():
    series = extract_series(
        [
            {"gamma": np.array([10.0])},
            {"gamma": np.array([12.5])},
            {"gamma": np.array([13.0])},
        ],
        "gamma",
    )

    np.testing.assert_allclose(series, np.array([10.0, 12.5, 13.0]))


def test_module_exposes_only_maintained_public_helpers():
    assert trajectory_metrics.__all__ == [
        "compute_delta_energy_components",
        "compute_delta_energy_series",
        "extract_series",
        "normalize_state",
    ]


def test_compute_delta_energy_components_returns_total_longitudinal_and_z_series():
    states = [
        _state(gamma=10.0, bz=0.50, z=0.0),
        _state(gamma=12.0, bz=0.60, z=5.0),
        _state(gamma=13.0, bz=0.70, z=9.0),
    ]
    initial_state = states[0]

    delta_total, delta_z, z_series = compute_delta_energy_components(
        states,
        initial_state,
        rest_energy_mev=2.0,
    )

    np.testing.assert_allclose(delta_total, np.array([0.0, 0.004, 0.006]))
    np.testing.assert_allclose(delta_z, np.array([0.0, 0.0044, 0.0082]))
    np.testing.assert_allclose(z_series, np.array([0.0, 5.0, 9.0]))
