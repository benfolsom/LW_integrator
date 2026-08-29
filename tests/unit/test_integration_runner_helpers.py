from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.integration_runner import (
    AdaptiveTimestepConfig,
    IntegrationCancelled,
    _calculate_gamma,
    _compute_total_energy,
    _ensure_startup_metadata,
    retarded_integrator,
)
from core.types import MagneticDipoleConfig, SimulationType
from core.species import resolve_species


def _make_particle_state(
    *,
    z: float = -1.0,
    charge: float = 0.0,
    mass: float = 1.0,
    gamma: float = 1.0,
) -> dict[str, np.ndarray]:
    return {
        "x": np.array([0.0], dtype=float),
        "y": np.array([0.0], dtype=float),
        "z": np.array([z], dtype=float),
        "t": np.array([0.0], dtype=float),
        "Px": np.array([0.0], dtype=float),
        "Py": np.array([0.0], dtype=float),
        "Pz": np.array([0.0], dtype=float),
        "Pt": np.array([gamma * mass * C_MMNS], dtype=float),
        "gamma": np.array([gamma], dtype=float),
        "bx": np.array([0.0], dtype=float),
        "by": np.array([0.0], dtype=float),
        "bz": np.array([0.0], dtype=float),
        "bdotx": np.array([0.0], dtype=float),
        "bdoty": np.array([0.0], dtype=float),
        "bdotz": np.array([0.0], dtype=float),
        "q": np.array([charge], dtype=float),
        "m": np.array([mass], dtype=float),
        "char_time": np.array([1e-3], dtype=float),
    }


def test_adaptive_timestep_config_derived_limits_follow_parameters() -> None:
    config = AdaptiveTimestepConfig(
        timestep_reduction_factor=10,
        min_timestep_factor=1e-4,
    )

    assert config.max_refinement_attempts == 4

    proximity_config = AdaptiveTimestepConfig(min_timestep_factor=0.09)
    assert proximity_config.max_substeps_per_step == 13


def test_adaptive_timestep_config_guards_invalid_reduction_factor() -> None:
    config = AdaptiveTimestepConfig(timestep_reduction_factor=1)

    assert config.max_refinement_attempts == 1


def test_compute_total_energy_and_gamma_use_particle_arrays() -> None:
    state = {
        "gamma": np.array([2.0, 3.0], dtype=float),
        "m": np.array([4.0, 5.0], dtype=float),
    }

    assert _compute_total_energy(state) == np.sum(
        state["gamma"] * state["m"] * C_MMNS**2
    )
    assert _calculate_gamma(state) == 3.0


def test_ensure_startup_metadata_initializes_and_repairs_arrays() -> None:
    state = {
        "x": np.array([1.0, 2.0], dtype=float),
        "y": np.array([3.0, 4.0], dtype=float),
        "z": np.array([5.0, 6.0], dtype=float),
        "bx": np.array([0.1, 0.2], dtype=float),
        "by": np.array([0.3, 0.4], dtype=float),
        "bz": np.array([0.5, 0.6], dtype=float),
        "origin_x": np.array([99.0], dtype=float),
        "beta_avg_x": np.array([88.0], dtype=float),
        "beta_samples": np.array([77.0], dtype=float),
    }

    _ensure_startup_metadata(state)

    np.testing.assert_array_equal(state["origin_x"], state["x"])
    np.testing.assert_array_equal(state["origin_y"], state["y"])
    np.testing.assert_array_equal(state["origin_z"], state["z"])
    np.testing.assert_array_equal(state["beta_avg_x"], state["bx"])
    np.testing.assert_array_equal(state["beta_avg_y"], state["by"])
    np.testing.assert_array_equal(state["beta_avg_z"], state["bz"])
    np.testing.assert_array_equal(state["beta_samples"], np.ones_like(state["x"]))


def test_ensure_startup_metadata_ignores_none_and_empty_states() -> None:
    _ensure_startup_metadata(None)

    empty_state = {"x": np.array([], dtype=float)}
    _ensure_startup_metadata(empty_state)

    assert list(empty_state.keys()) == ["x"]


def test_retarded_integrator_initializes_step_zero_and_reports_progress() -> None:
    calls: list[tuple[int, int]] = []

    trajectory, trajectory_drv, *_soa_out = retarded_integrator(
        steps=1,
        h_step=1e-3,
        wall_z=0.0,
        aperture_radius=0.5,
        sim_type=SimulationType.CONDUCTING_WALL,
        init_rider=_make_particle_state(),
        init_driver=None,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        progress_callback=lambda current, total: calls.append((current, total)),
        use_numba=False,
    )

    assert calls == [(0, 1), (1, 1)]
    assert len(trajectory) == 1
    assert len(trajectory_drv) == 1
    assert "origin_x" in trajectory[0]
    assert "origin_x" in trajectory_drv[0]


def test_retarded_integrator_honors_early_cancellation() -> None:
    with pytest.raises(IntegrationCancelled):
        retarded_integrator(
            steps=2,
            h_step=1e-3,
            wall_z=0.0,
            aperture_radius=0.5,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=_make_particle_state(),
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            cancel_callback=lambda: True,
            use_numba=False,
        )


def test_retarded_integrator_requires_driver_for_bunch_to_bunch_mode() -> None:
    with pytest.raises(ValueError, match="requires init_driver state"):
        retarded_integrator(
            steps=1,
            h_step=1e-3,
            wall_z=0.0,
            aperture_radius=0.5,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_make_particle_state(),
            init_driver=None,
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            use_numba=False,
        )


def test_second_order_exact_update_requires_inertial_prehistory() -> None:
    with pytest.raises(
        ValueError,
        match="second_order_start_taylor_endpoint requires",
    ):
        retarded_integrator(
            steps=1,
            h_step=1e-3,
            wall_z=0.0,
            aperture_radius=0.5,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_make_particle_state(
                charge=-ELEMENTARY_CHARGE,
                mass=resolve_species("electron").mass_amu,
            ),
            init_driver=_make_particle_state(
                z=1.0,
                charge=ELEMENTARY_CHARGE,
                mass=resolve_species("proton").mass_amu,
            ),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            magnetic_dipole=MagneticDipoleConfig(
                enabled=True,
                exact_retarded_update=("second_order_start_taylor_endpoint"),
            ),
            use_numba=False,
        )
