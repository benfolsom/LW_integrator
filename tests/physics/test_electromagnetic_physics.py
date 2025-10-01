"""Physics-level regression tests for the modular LW integrator core."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.distances import compute_instantaneous_distance
from core.equations import retarded_equations_of_motion
from core.images import generate_conducting_image
from core.integrator import retarded_integrator
from core.types import IntegratorConfig, SimulationType
from tests.test_config import (
    BASIC_TWO_PARTICLE,
    PROTON,
    create_bunch_uniform_distribution,
    validate_physics_conservation,
)


def _single_particle_state(
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = -1.0,
    t: float = 0.0,
    px: float = 0.0,
    py: float = 0.0,
    pz: float = 0.0,
    charge: float = ELEMENTARY_CHARGE,
    mass: float = 1.0,
    gamma: float = 1.0,
    bx: float = 0.0,
    by: float = 0.0,
    bz: float = 0.0,
    char_time: float = 1e-3,
) -> dict[str, np.ndarray]:
    arr = np.array
    return {
        "x": arr([x], dtype=float),
        "y": arr([y], dtype=float),
        "z": arr([z], dtype=float),
        "t": arr([t], dtype=float),
        "Px": arr([px], dtype=float),
        "Py": arr([py], dtype=float),
        "Pz": arr([pz], dtype=float),
        "Pt": arr([gamma * mass * C_MMNS], dtype=float),
        "gamma": arr([gamma], dtype=float),
        "bx": arr([bx], dtype=float),
        "by": arr([by], dtype=float),
        "bz": arr([bz], dtype=float),
        "bdotx": arr([0.0], dtype=float),
        "bdoty": arr([0.0], dtype=float),
        "bdotz": arr([0.0], dtype=float),
        "q": arr([charge], dtype=float),
        "m": arr([mass], dtype=float),
        "char_time": arr([char_time], dtype=float),
    }


@pytest.mark.physics
def test_compute_instantaneous_distance_direction_unit_vector() -> None:
    source = {"x": np.array([0.0]), "y": np.array([0.0]), "z": np.array([0.0])}
    target = {"x": np.array([1.0]), "y": np.array([2.0]), "z": np.array([2.0])}

    result = compute_instantaneous_distance(source, target, 0)

    length = np.sqrt(result["nx"][0] ** 2 + result["ny"][0] ** 2 + result["nz"][0] ** 2)
    assert result["R"][0] == pytest.approx(np.sqrt(9.0))
    assert length == pytest.approx(1.0)


@pytest.mark.physics
def test_generate_conducting_image_reflects_boundary_conditions() -> None:
    source = _single_particle_state(z=-2.0, pz=1.5, bz=0.2, charge=-ELEMENTARY_CHARGE)

    image = generate_conducting_image(source, wall_z=0.0, aperture_radius=0.5)

    assert image["z"][0] == pytest.approx(2.0)
    assert image["Pz"][0] == pytest.approx(-source["Pz"][0])
    assert image["bz"][0] == pytest.approx(-source["bz"][0])
    assert np.max(np.abs(image["q"])) <= np.abs(source["q"][0]) + 1e-12


@pytest.mark.physics
def test_retarded_equations_of_motion_preserves_finite_values() -> None:
    trajectory = [_single_particle_state(z=-1.0), _single_particle_state(z=-0.5, t=1e-3)]
    trajectory_ext = [copy.deepcopy(state) for state in trajectory]
    for ext in trajectory_ext:
        ext["x"] += 1e-4
        ext["z"] += 1e-4

    updated = retarded_equations_of_motion(
        h=1e-3,
        trajectory=trajectory,
        trajectory_ext=trajectory_ext,
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
    )

    for key, value in updated.items():
        if isinstance(value, np.ndarray):
            assert np.all(np.isfinite(value))

    assert updated["t"][0] == pytest.approx(trajectory[0]["t"][0] + 1e-3)
    assert updated["q"][0] == pytest.approx(trajectory[0]["q"][0])


@pytest.mark.physics
def test_retarded_integrator_conservation_metrics() -> None:
    config = IntegratorConfig(
        steps=3,
        time_step=BASIC_TWO_PARTICLE.step_size,
        wall_position=BASIC_TWO_PARTICLE.wall_z,
        aperture_radius=BASIC_TWO_PARTICLE.aperture_r,
        simulation_type=SimulationType.CONDUCTING_WALL,
        bunch_mean=0.0,
        cavity_spacing=0.0,
        z_cutoff=BASIC_TWO_PARTICLE.z_cutoff,
    )

    rider_state = create_bunch_uniform_distribution(BASIC_TWO_PARTICLE, PROTON, "line")

    trajectory, _ = retarded_integrator(
        steps=config.steps,
        h_step=config.time_step,
        wall_z=config.wall_position,
        aperture_radius=config.aperture_radius,
        sim_type=config.simulation_type,
        init_rider=rider_state,
        init_driver=None,
        mean=config.bunch_mean,
        cav_spacing=config.cavity_spacing,
        z_cutoff=config.z_cutoff,
    )

    metrics = validate_physics_conservation(trajectory[0], trajectory[-1], tolerance=1e-2)

    assert metrics["charge_conservation"]["passed"]
    assert metrics["energy_conservation"]["relative_change"] < 1e-2
