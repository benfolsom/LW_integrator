"""Unit tests for bunch initialization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from input_output.bunch_initialization import (
    create_bunch_from_energy,
    create_bunch_from_params,
)


def _sample_params() -> dict[str, Any]:
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


def test_create_bunch_from_params_ring_geometry_places_particles_on_radius():
    params = _sample_params()
    params.update({"transverse_geometry": "ring", "pcount": 8})

    state, _ = create_bunch_from_params(**params, seed=12345)

    radius = np.sqrt(
        (state["x"] - params["transv_offset_x"]) ** 2
        + (state["y"] - params["transv_offset_y"]) ** 2
    )
    np.testing.assert_allclose(radius, params["transv_dist"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        state["x"][0], params["transv_offset_x"] + params["transv_dist"]
    )
    np.testing.assert_allclose(state["y"][0], params["transv_offset_y"])


def test_create_bunch_from_energy_accepts_ring_geometry_alias():
    state, _ = create_bunch_from_energy(
        kinetic_energy_mev=35.0,
        mass_amu=0.00054857990907,
        charge_sign=-1.0,
        particle_count=6,
        transverse_spread=0.03,
        transverse_offset_x=0.01,
        transverse_offset_y=-0.02,
        transverse_geometry="circle",
    )

    radius = np.sqrt((state["x"] - 0.01) ** 2 + (state["y"] + 0.02) ** 2)
    np.testing.assert_allclose(radius, 0.03, rtol=0.0, atol=1e-12)
