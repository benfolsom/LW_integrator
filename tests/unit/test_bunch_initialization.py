"""Unit tests for bunch initialization helpers."""

from __future__ import annotations

import numpy as np

from input_output.bunch_initialization import create_bunch_from_params


def _sample_params() -> dict[str, float | int]:
    return {
        "starting_distance": 0.0,
        "transv_mom": 2.2e-5,
        "starting_Pz": 1000.0,
        "stripped_ions": 1.0,
        "m_particle": 0.00054857990907,
        "transv_dist": 0.2,
        "transv_offset_x": 0.15,
        "transv_offset_y": -0.1,
        "pcount": 10,
        "charge_sign": -1.0,
    }


def test_create_bunch_from_params_is_reproducible_with_same_seed():
    params = _sample_params()

    state_a, rest_energy_a = create_bunch_from_params(**params, seed=12345)
    state_b, rest_energy_b = create_bunch_from_params(**params, seed=12345)

    assert rest_energy_a == rest_energy_b
    for key in ("x", "y", "Px", "Py", "Pz", "gamma"):
        np.testing.assert_allclose(state_a[key], state_b[key])


def test_create_bunch_from_params_changes_with_different_seed():
    params = _sample_params()

    state_a, _ = create_bunch_from_params(**params, seed=12345)
    state_b, _ = create_bunch_from_params(**params, seed=12346)

    assert any(
        not np.allclose(state_a[key], state_b[key])
        for key in ("x", "y", "Px", "Py", "Pz")
    )


def test_create_bunch_from_params_respects_requested_spreads():
    params = _sample_params()

    state, _ = create_bunch_from_params(**params, seed=12345)

    x_min = params["transv_offset_x"] - params["transv_dist"]
    x_max = params["transv_offset_x"] + params["transv_dist"]
    y_min = params["transv_offset_y"] - params["transv_dist"]
    y_max = params["transv_offset_y"] + params["transv_dist"]
    max_transverse_momentum = params["transv_mom"] * params["m_particle"]

    assert np.all(state["x"] >= x_min)
    assert np.all(state["x"] <= x_max)
    assert np.all(state["y"] >= y_min)
    assert np.all(state["y"] <= y_max)
    assert np.all(np.abs(state["Px"]) <= max_transverse_momentum)
    assert np.all(np.abs(state["Py"]) <= max_transverse_momentum)

