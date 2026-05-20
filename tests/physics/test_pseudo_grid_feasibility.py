"""Physics-facing pseudo-grid sanity checks."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.integration_runner import retarded_integrator
from core.types import PseudoGridConfig, SimulationType


def _make_crossing_bunch(
    *,
    n_particles: int,
    z_mm: float,
    beta_z: float,
    charge_scale: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    gamma_value = 1.0 / np.sqrt(1.0 - beta_z**2)
    mass = np.ones(n_particles, dtype=float)
    charge_pattern = rng.normal(0.0, 1.0, n_particles)
    charge_pattern -= float(np.mean(charge_pattern))
    if np.max(np.abs(charge_pattern)) > 0.0:
        charge_pattern /= float(np.max(np.abs(charge_pattern)))
    zeros = np.zeros(n_particles, dtype=float)
    return {
        "x": np.linspace(-0.4, 0.4, n_particles) + rng.normal(0.0, 0.01, n_particles),
        "y": rng.normal(0.0, 0.01, n_particles),
        "z": np.full(n_particles, z_mm, dtype=float),
        "t": zeros.copy(),
        "Px": zeros.copy(),
        "Py": zeros.copy(),
        "Pz": gamma_value * mass * C_MMNS * beta_z,
        "Pt": gamma_value * mass * C_MMNS,
        "gamma": np.full(n_particles, gamma_value, dtype=float),
        "bx": zeros.copy(),
        "by": zeros.copy(),
        "bz": np.full(n_particles, beta_z, dtype=float),
        "bdotx": zeros.copy(),
        "bdoty": zeros.copy(),
        "bdotz": zeros.copy(),
        "q": charge_scale * charge_pattern,
        "m": mass,
        "char_time": np.full(n_particles, 1.0e-3, dtype=float),
    }


def _clone_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in state.items()}


def _run_crossing_case(
    *,
    charge_scale: float,
    pseudo_grid: PseudoGridConfig | None,
    n_particles: int = 24,
    steps: int = 24,
    h_step: float = 1.0e-4,
    beta_z: float = 0.12,
):
    rider = _make_crossing_bunch(
        n_particles=n_particles,
        z_mm=-0.03,
        beta_z=beta_z,
        charge_scale=charge_scale,
        seed=100 + n_particles,
    )
    driver = _make_crossing_bunch(
        n_particles=n_particles,
        z_mm=0.03,
        beta_z=-beta_z,
        charge_scale=-charge_scale,
        seed=200 + n_particles,
    )
    return retarded_integrator(
        steps=steps,
        h_step=h_step,
        wall_z=0.0,
        aperture_radius=10.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_clone_state(rider),
        init_driver=_clone_state(driver),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        pseudo_grid=pseudo_grid,
        use_numba=False,
        radiation_reaction_mode="power_matched_damping",
    )


@pytest.mark.physics
def test_pseudo_grid_zero_charge_crossing_matches_inertial_baseline() -> None:
    steps = 24
    h_step = 1.0e-4
    beta_z = 0.12
    gamma_value = 1.0 / np.sqrt(1.0 - beta_z**2)
    expected_rider_z = -0.03 + np.arange(steps) * h_step * gamma_value * C_MMNS * beta_z
    expected_driver_z = 0.03 - np.arange(steps) * h_step * gamma_value * C_MMNS * beta_z

    _rider, _driver, rider_soa, driver_soa = _run_crossing_case(
        charge_scale=0.0,
        n_particles=16,
        steps=steps,
        h_step=h_step,
        beta_z=beta_z,
        pseudo_grid=PseudoGridConfig(
            enabled=True,
            active_rider_count=8,
            active_driver_count=8,
            passive_neighbor_count=2,
            causal_history_pruning_enabled=True,
            causal_history_safety_margin_steps=0,
        ),
    )

    assert rider_soa is not None
    assert driver_soa is not None
    np.testing.assert_allclose(
        rider_soa.z,
        np.repeat(expected_rider_z[:, None], rider_soa.n_particles, axis=1),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        driver_soa.z,
        np.repeat(expected_driver_z[:, None], driver_soa.n_particles, axis=1),
        atol=1.0e-12,
    )
    assert float(np.mean(rider_soa.z[0])) < 0.0 < float(np.mean(rider_soa.z[-1]))
    assert float(np.mean(driver_soa.z[-1])) < 0.0 < float(np.mean(driver_soa.z[0]))
    assert np.all(np.isfinite(rider_soa.gamma))
    assert np.all(np.isfinite(driver_soa.gamma))


@pytest.mark.physics
def test_pseudo_grid_weak_charge_crossing_tracks_full_solver() -> None:
    full_rider, full_driver, full_rider_soa, full_driver_soa = _run_crossing_case(
        charge_scale=2.0e-2,
        pseudo_grid=None,
    )
    pseudo_rider, pseudo_driver, pseudo_rider_soa, pseudo_driver_soa = (
        _run_crossing_case(
            charge_scale=2.0e-2,
            pseudo_grid=PseudoGridConfig(
                enabled=True,
                active_rider_count=16,
                active_driver_count=16,
                passive_neighbor_count=4,
                causal_history_pruning_enabled=True,
                causal_history_safety_margin_steps=0,
            ),
        )
    )

    assert full_rider_soa is not None
    assert full_driver_soa is not None
    assert pseudo_rider_soa is not None
    assert pseudo_driver_soa is not None
    assert len(pseudo_rider) == len(full_rider)
    assert len(pseudo_driver) == len(full_driver)
    assert (
        float(np.mean(pseudo_rider_soa.z[0]))
        < 0.0
        < float(np.mean(pseudo_rider_soa.z[-1]))
    )
    assert (
        float(np.mean(pseudo_driver_soa.z[-1]))
        < 0.0
        < float(np.mean(pseudo_driver_soa.z[0]))
    )

    np.testing.assert_allclose(pseudo_rider_soa.x, full_rider_soa.x, atol=1.0e-4)
    np.testing.assert_allclose(pseudo_rider_soa.z, full_rider_soa.z, atol=1.0e-4)
    np.testing.assert_allclose(
        pseudo_rider_soa.gamma,
        full_rider_soa.gamma,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(pseudo_driver_soa.x, full_driver_soa.x, atol=1.0e-4)
    np.testing.assert_allclose(pseudo_driver_soa.z, full_driver_soa.z, atol=1.0e-4)
    np.testing.assert_allclose(
        pseudo_driver_soa.gamma,
        full_driver_soa.gamma,
        atol=2.0e-5,
    )
    assert np.all(np.isfinite(pseudo_rider_soa.gamma))
    assert np.all(np.isfinite(pseudo_driver_soa.gamma))
