"""Tests for the energy ledger series and scalar metrics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from lw_integrator.testbed_runner import (
    _compute_energy_ledger_series,
    _ledger_scalar_metrics,
)


def _fake_ledger(**overrides) -> dict:
    """Build a minimal ledger dict with all fields used by _ledger_scalar_metrics."""
    base = {
        "kinetic_energy_mev": np.array([10.0, 11.0, 12.0]),
        "delta_kinetic_energy_mev": np.array([0.0, 1.0, 2.0]),
        "longitudinal_kinetic_energy_mev": np.array([5.0, 5.5, 6.0]),
        "kinetic_energy_z_mev": np.array([5.0, 5.5, 6.0]),
        "delta_kinetic_energy_z_mev": np.array([0.0, 0.5, 1.0]),
        "kinetic_energy_x_mev": np.array([1.0, 1.5, 2.0]),
        "delta_kinetic_energy_x_mev": np.array([0.0, 0.5, 1.0]),
        "kinetic_energy_y_mev": np.array([2.0, 2.5, 3.0]),
        "delta_kinetic_energy_y_mev": np.array([0.0, 0.5, 1.0]),
        "initial_gamma": 2.0,
        "initial_bz": 0.5,
        "initial_bx": 0.1,
        "initial_by": 0.2,
        "initial_kinetic_energy_mev": 10.0,
        "initial_longitudinal_kinetic_energy_mev": 5.0,
        "initial_kinetic_energy_x_mev": 1.0,
        "initial_kinetic_energy_y_mev": 2.0,
    }
    base.update(overrides)
    return base


def test_scalar_metrics_produce_new_xy_keys_with_correct_values():
    ledger = _fake_ledger()
    metrics = _ledger_scalar_metrics("rider", ledger)

    assert metrics["rider_final_delta_kinetic_energy_x_mev"] == pytest.approx(1.0)
    assert metrics["rider_max_delta_kinetic_energy_x_mev"] == pytest.approx(1.0)
    assert metrics["rider_min_delta_kinetic_energy_x_mev"] == pytest.approx(0.0)

    assert metrics["rider_final_delta_kinetic_energy_y_mev"] == pytest.approx(1.0)
    assert metrics["rider_max_delta_kinetic_energy_y_mev"] == pytest.approx(1.0)
    assert metrics["rider_min_delta_kinetic_energy_y_mev"] == pytest.approx(0.0)


def test_percent_energy_gain_computation():
    ledger = _fake_ledger(
        initial_kinetic_energy_mev=10.0,
        delta_kinetic_energy_mev=np.array([0.0, 1.0, 1.0]),
    )
    metrics = _ledger_scalar_metrics("rider", ledger)

    assert metrics["rider_final_percent_energy_gain"] == pytest.approx(10.0)
    assert metrics["rider_max_percent_energy_gain"] == pytest.approx(10.0)


def test_percent_energy_gain_zero_initial_ke_guard():
    ledger = _fake_ledger(
        initial_kinetic_energy_mev=0.0,
        delta_kinetic_energy_mev=np.array([0.0, 1.0, 2.0]),
    )
    metrics = _ledger_scalar_metrics("rider", ledger)

    assert metrics["rider_final_percent_energy_gain"] == pytest.approx(0.0)
    assert metrics["rider_max_percent_energy_gain"] == pytest.approx(0.0)
    assert np.isfinite(metrics["rider_final_percent_energy_gain"])
    assert np.isfinite(metrics["rider_max_percent_energy_gain"])


def test_backward_compat_existing_metrics_unchanged():
    ledger = _fake_ledger()
    metrics = _ledger_scalar_metrics("rider", ledger)

    expected_existing = {
        "rider_initial_mean_kinetic_energy_mev": 10.0,
        "rider_final_mean_kinetic_energy_mev": 12.0,
        "rider_max_mean_kinetic_energy_mev": 12.0,
        "rider_final_delta_kinetic_energy_mev": 2.0,
        "rider_max_delta_kinetic_energy_mev": 2.0,
        "rider_final_delta_kinetic_energy_z_mev": 1.0,
        "rider_max_delta_kinetic_energy_z_mev": 1.0,
        "rider_min_delta_kinetic_energy_z_mev": 0.0,
    }
    for key, value in expected_existing.items():
        assert key in metrics, f"missing backward-compat key: {key}"
        assert metrics[key] == pytest.approx(value), f"wrong value for {key}"


def _make_state(gamma: float, bx: float, by: float, bz: float) -> dict:
    return {
        "gamma": np.array([gamma]),
        "bx": np.array([bx]),
        "by": np.array([by]),
        "bz": np.array([bz]),
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
    }


def test_compute_energy_ledger_series_includes_xy_and_alias_fields():
    rest_energy_mev = 938.272
    initial = _make_state(gamma=2.0, bx=0.1, by=0.2, bz=0.5)
    states = [
        _make_state(gamma=2.0, bx=0.1, by=0.2, bz=0.5),
        _make_state(gamma=2.1, bx=0.12, by=0.22, bz=0.52),
        _make_state(gamma=2.2, bx=0.14, by=0.24, bz=0.54),
    ]

    ledger = _compute_energy_ledger_series(states, initial, rest_energy_mev)

    # New series keys present
    for key in (
        "kinetic_energy_x_mev",
        "kinetic_energy_y_mev",
        "delta_kinetic_energy_x_mev",
        "delta_kinetic_energy_y_mev",
        "initial_kinetic_energy_x_mev",
        "initial_kinetic_energy_y_mev",
    ):
        assert key in ledger, f"missing series key: {key}"

    # z alias equals longitudinal
    assert np.allclose(
        ledger["kinetic_energy_z_mev"], ledger["longitudinal_kinetic_energy_mev"]
    )

    # Spot-check x series value at final step: gamma * bx * rest_energy
    expected_final_x = 2.2 * 0.14 * rest_energy_mev
    assert ledger["kinetic_energy_x_mev"][-1] == pytest.approx(expected_final_x)

    # delta_x at final step
    expected_initial_x = 2.0 * 0.1 * rest_energy_mev
    assert ledger["delta_kinetic_energy_x_mev"][-1] == pytest.approx(
        expected_final_x - expected_initial_x
    )

    # initial scalar fields
    assert ledger["initial_kinetic_energy_x_mev"] == pytest.approx(expected_initial_x)
    assert ledger["initial_kinetic_energy_y_mev"] == pytest.approx(
        2.0 * 0.2 * rest_energy_mev
    )
    assert ledger["initial_bx"] == pytest.approx(0.1)
    assert ledger["initial_by"] == pytest.approx(0.2)
