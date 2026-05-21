"""Physics-facing pseudo-grid sanity checks."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.integration_runner import AdaptiveTimestepConfig, retarded_integrator
from core.types import PseudoGridConfig, SimulationType, SpaceChargeConfig


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


def _apply_irregular_layout(
    state: dict[str, np.ndarray],
    *,
    layout: str,
    radius_mm: float,
    z_span_mm: float = 0.0,
    angle_offset: float = 0.0,
) -> None:
    n_particles = len(state["x"])
    angles = np.linspace(0.0, 2.0 * np.pi, n_particles, endpoint=False) + angle_offset
    if layout == "ring":
        radii = np.full(n_particles, radius_mm, dtype=float)
    elif layout == "uniform":
        side = int(np.ceil(np.sqrt(n_particles)))
        grid = np.linspace(-radius_mm, radius_mm, side)
        xx, yy = np.meshgrid(grid, grid)
        state["x"] = xx.ravel()[:n_particles]
        state["y"] = yy.ravel()[:n_particles]
        return
    elif layout == "hollow_cylinder":
        inner_radius = 0.55 * radius_mm
        radii = np.where(
            np.arange(n_particles) % 2 == 0,
            inner_radius,
            radius_mm,
        )
        state["z"] = state["z"] + np.linspace(-0.5, 0.5, n_particles) * z_span_mm
    else:
        raise ValueError(f"Unsupported test layout {layout!r}")

    state["x"] = radii * np.cos(angles)
    state["y"] = radii * np.sin(angles)


def _run_irregular_layout_case(
    *,
    layout: str,
    pseudo_grid: PseudoGridConfig | None,
):
    rider = _make_crossing_bunch(
        n_particles=24,
        z_mm=-0.03,
        beta_z=0.12,
        charge_scale=5.0e-3,
        seed=310,
    )
    driver = _make_crossing_bunch(
        n_particles=24,
        z_mm=0.03,
        beta_z=-0.12,
        charge_scale=-5.0e-3,
        seed=410,
    )
    _apply_irregular_layout(
        rider,
        layout=layout,
        radius_mm=0.04,
        z_span_mm=0.004,
    )
    _apply_irregular_layout(
        driver,
        layout=layout,
        radius_mm=0.05,
        z_span_mm=0.004,
        angle_offset=np.pi / 24.0,
    )
    return retarded_integrator(
        steps=24,
        h_step=1.0e-4,
        wall_z=0.0,
        aperture_radius=10.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_clone_state(rider),
        init_driver=_clone_state(driver),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        pseudo_grid=pseudo_grid,
        space_charge=SpaceChargeConfig(
            enabled=True,
            retarded=True,
            softening_mm=0.3,
            min_retarded_steps=0,
        ),
        use_numba=False,
        radiation_reaction_mode="power_matched_damping",
    )


def _run_crossing_case(
    *,
    charge_scale: float,
    pseudo_grid: PseudoGridConfig | None,
    n_particles: int = 24,
    steps: int = 24,
    h_step: float = 1.0e-4,
    beta_z: float = 0.12,
    space_charge: SpaceChargeConfig | None = None,
    adaptive_timestep: AdaptiveTimestepConfig | None = None,
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
        space_charge=space_charge,
        adaptive_timestep=adaptive_timestep,
        use_numba=False,
        radiation_reaction_mode="power_matched_damping",
    )


def _assert_finite_crossing(rider_soa, driver_soa) -> None:
    assert rider_soa is not None
    assert driver_soa is not None
    assert float(np.mean(rider_soa.z[0])) < 0.0 < float(np.mean(rider_soa.z[-1]))
    assert float(np.mean(driver_soa.z[-1])) < 0.0 < float(np.mean(driver_soa.z[0]))
    assert np.all(np.isfinite(rider_soa.x))
    assert np.all(np.isfinite(rider_soa.z))
    assert np.all(np.isfinite(rider_soa.gamma))
    assert np.all(np.isfinite(driver_soa.x))
    assert np.all(np.isfinite(driver_soa.z))
    assert np.all(np.isfinite(driver_soa.gamma))


def _assert_tracks_full_solver(
    full_rider_soa,
    full_driver_soa,
    pseudo_rider_soa,
    pseudo_driver_soa,
    *,
    position_atol: float = 1.0e-4,
    gamma_atol: float = 2.0e-5,
) -> None:
    _assert_finite_crossing(full_rider_soa, full_driver_soa)
    _assert_finite_crossing(pseudo_rider_soa, pseudo_driver_soa)
    np.testing.assert_allclose(pseudo_rider_soa.x, full_rider_soa.x, atol=position_atol)
    np.testing.assert_allclose(pseudo_rider_soa.z, full_rider_soa.z, atol=position_atol)
    np.testing.assert_allclose(
        pseudo_rider_soa.gamma, full_rider_soa.gamma, atol=gamma_atol
    )
    np.testing.assert_allclose(
        pseudo_driver_soa.x, full_driver_soa.x, atol=position_atol
    )
    np.testing.assert_allclose(
        pseudo_driver_soa.z, full_driver_soa.z, atol=position_atol
    )
    np.testing.assert_allclose(
        pseudo_driver_soa.gamma, full_driver_soa.gamma, atol=gamma_atol
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


@pytest.mark.physics
def test_pseudo_grid_space_charge_crossing_tracks_full_solver() -> None:
    space_charge = SpaceChargeConfig(
        enabled=True,
        retarded=False,
        softening_mm=0.3,
    )
    full_rider, full_driver, full_rider_soa, full_driver_soa = _run_crossing_case(
        charge_scale=5.0e-3,
        pseudo_grid=None,
        space_charge=space_charge,
    )
    pseudo_rider, pseudo_driver, pseudo_rider_soa, pseudo_driver_soa = (
        _run_crossing_case(
            charge_scale=5.0e-3,
            pseudo_grid=PseudoGridConfig(
                enabled=True,
                active_rider_count=16,
                active_driver_count=16,
                passive_neighbor_count=4,
                causal_history_pruning_enabled=True,
                causal_history_safety_margin_steps=0,
            ),
            space_charge=space_charge,
        )
    )

    assert len(pseudo_rider) == len(full_rider)
    assert len(pseudo_driver) == len(full_driver)
    _assert_tracks_full_solver(
        full_rider_soa,
        full_driver_soa,
        pseudo_rider_soa,
        pseudo_driver_soa,
    )


@pytest.mark.physics
def test_pseudo_grid_retarded_space_charge_crossing_tracks_full_solver() -> None:
    space_charge = SpaceChargeConfig(
        enabled=True,
        retarded=True,
        softening_mm=0.3,
        min_retarded_steps=0,
    )
    full_rider, full_driver, full_rider_soa, full_driver_soa = _run_crossing_case(
        charge_scale=5.0e-3,
        pseudo_grid=None,
        space_charge=space_charge,
    )
    pseudo_rider, pseudo_driver, pseudo_rider_soa, pseudo_driver_soa = (
        _run_crossing_case(
            charge_scale=5.0e-3,
            pseudo_grid=PseudoGridConfig(
                enabled=True,
                active_rider_count=16,
                active_driver_count=16,
                passive_neighbor_count=4,
                causal_history_pruning_enabled=True,
                causal_history_safety_margin_steps=0,
            ),
            space_charge=space_charge,
        )
    )

    assert len(pseudo_rider) == len(full_rider)
    assert len(pseudo_driver) == len(full_driver)
    _assert_tracks_full_solver(
        full_rider_soa,
        full_driver_soa,
        pseudo_rider_soa,
        pseudo_driver_soa,
    )


@pytest.mark.physics
@pytest.mark.parametrize("layout", ["ring", "uniform", "hollow_cylinder"])
def test_pseudo_grid_irregular_layout_crossing_tracks_full_solver(layout: str) -> None:
    _full_rider, _full_driver, full_rider_soa, full_driver_soa = (
        _run_irregular_layout_case(layout=layout, pseudo_grid=None)
    )
    _pseudo_rider, _pseudo_driver, pseudo_rider_soa, pseudo_driver_soa = (
        _run_irregular_layout_case(
            layout=layout,
            pseudo_grid=PseudoGridConfig(
                enabled=True,
                active_rider_count=18,
                active_driver_count=18,
                passive_neighbor_count=4,
                causal_history_pruning_enabled=True,
                causal_history_safety_margin_steps=0,
            ),
        )
    )

    _assert_tracks_full_solver(
        full_rider_soa,
        full_driver_soa,
        pseudo_rider_soa,
        pseudo_driver_soa,
        position_atol=2.0e-4,
        gamma_atol=5.0e-5,
    )


@pytest.mark.physics
def test_pseudo_grid_adaptive_retarded_space_charge_crossing_remains_finite() -> None:
    space_charge = SpaceChargeConfig(
        enabled=True,
        retarded=True,
        softening_mm=0.3,
        min_retarded_steps=0,
    )
    adaptive_timestep = AdaptiveTimestepConfig(
        enabled=True,
        energy_jump_threshold=0.05,
        timestep_reduction_factor=3,
        min_timestep_factor=1.0e-3,
        proximity_refinement_enabled=True,
        proximity_reduction_factor=3,
    )

    _pseudo_rider, _pseudo_driver, pseudo_rider_soa, pseudo_driver_soa = (
        _run_crossing_case(
            charge_scale=5.0e-3,
            pseudo_grid=PseudoGridConfig(
                enabled=True,
                active_rider_count=16,
                active_driver_count=16,
                passive_neighbor_count=4,
                causal_history_pruning_enabled=True,
                causal_history_safety_margin_steps=0,
            ),
            space_charge=space_charge,
            adaptive_timestep=adaptive_timestep,
        )
    )

    _assert_finite_crossing(pseudo_rider_soa, pseudo_driver_soa)
    assert pseudo_rider_soa is not None
    assert pseudo_driver_soa is not None
    assert float(np.max(pseudo_rider_soa.gamma)) < 2.0
    assert float(np.max(pseudo_driver_soa.gamma)) < 2.0


@pytest.mark.physics
def test_pseudo_grid_stronger_charge_longer_crossing_window_remains_finite() -> None:
    _pseudo_rider, _pseudo_driver, pseudo_rider_soa, pseudo_driver_soa = (
        _run_crossing_case(
            charge_scale=5.0e-2,
            pseudo_grid=PseudoGridConfig(
                enabled=True,
                active_rider_count=16,
                active_driver_count=16,
                passive_neighbor_count=4,
                causal_history_pruning_enabled=True,
                causal_history_safety_margin_steps=0,
            ),
            steps=72,
        )
    )

    _assert_finite_crossing(pseudo_rider_soa, pseudo_driver_soa)
    assert pseudo_rider_soa is not None
    assert pseudo_driver_soa is not None
    assert float(np.max(pseudo_rider_soa.gamma)) < 2.0
    assert float(np.max(pseudo_driver_soa.gamma)) < 2.0
