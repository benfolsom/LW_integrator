from __future__ import annotations

import numpy as np
import pytest

from core.particle_loss import build_particle_loss_context, mark_particle_losses
from core.types import ParticleLossConfig, SimulationType, TrajectoryBuilder
from core.vectorized_interactions import gather_external_samples_soa


def _state(
    *,
    x: list[float],
    y: list[float] | None = None,
    z: list[float] | None = None,
    q: list[float] | None = None,
    dead: list[bool] | None = None,
) -> dict[str, np.ndarray]:
    n_particles = len(x)
    y_vals = y if y is not None else [0.0] * n_particles
    z_vals = z if z is not None else [0.0] * n_particles
    q_vals = q if q is not None else [1.0] * n_particles
    dead_vals = dead if dead is not None else [False] * n_particles
    zeros = np.zeros(n_particles, dtype=float)
    return {
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y_vals, dtype=float),
        "z": np.asarray(z_vals, dtype=float),
        "t": zeros.copy(),
        "Px": zeros.copy(),
        "Py": zeros.copy(),
        "Pz": zeros.copy(),
        "Pt": np.ones(n_particles, dtype=float),
        "gamma": np.ones(n_particles, dtype=float),
        "bx": zeros.copy(),
        "by": zeros.copy(),
        "bz": zeros.copy(),
        "bdotx": zeros.copy(),
        "bdoty": zeros.copy(),
        "bdotz": zeros.copy(),
        "q": np.asarray(q_vals, dtype=float),
        "m": np.ones(n_particles, dtype=float),
        "char_time": np.ones(n_particles, dtype=float),
        "origin_x": np.asarray(x, dtype=float),
        "origin_y": np.asarray(y_vals, dtype=float),
        "origin_z": np.asarray(z_vals, dtype=float),
        "beta_avg_x": zeros.copy(),
        "beta_avg_y": zeros.copy(),
        "beta_avg_z": zeros.copy(),
        "beta_samples": np.ones(n_particles, dtype=float),
        "radiation_power": zeros.copy(),
        "radiation_energy": zeros.copy(),
        "radiation_energy_applied": zeros.copy(),
        "_dead_particles": np.asarray(dead_vals, dtype=bool),
    }


def test_particle_loss_config_defaults_to_broad_physical_acceptance() -> None:
    config = ParticleLossConfig()

    assert config.enabled is True
    assert config.loss_radius_mm == pytest.approx(500.0)
    assert config.conducting_wall_aperture_loss_enabled is True


def test_loss_radius_marks_only_particles_outside_explicit_radius() -> None:
    previous = _state(x=[0.5, 0.0], y=[0.0, 0.0], q=[2.0, 2.0])
    current = _state(x=[0.5, 2.0], y=[0.0, 0.0], q=[2.0, 2.0])

    marked = mark_particle_losses(
        current,
        previous,
        step=4,
        config=ParticleLossConfig(enabled=True, loss_radius_mm=1.0),
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        wall_z=0.0,
        aperture_radius=1.0,
    )

    assert marked == 1
    assert current["_dead_particles"].tolist() == [False, True]
    assert current["q"].tolist() == pytest.approx([2.0, 0.0])
    assert current["_particle_failure_info"][1]["reason"] == "loss_radius_exceeded"
    assert current["_particle_failure_info"][1]["radius_limit_mm"] == pytest.approx(1.0)


def test_conducting_wall_aperture_loss_uses_interpolated_crossing_radius() -> None:
    previous = _state(x=[0.5, 2.0], z=[-1.0, -1.0], q=[1.0, 1.0])
    current = _state(x=[0.5, 2.0], z=[1.0, 1.0], q=[1.0, 1.0])

    marked = mark_particle_losses(
        current,
        previous,
        step=3,
        config=ParticleLossConfig(enabled=True),
        sim_type=SimulationType.CONDUCTING_WALL,
        wall_z=0.0,
        aperture_radius=1.0,
    )

    assert marked == 1
    assert current["_dead_particles"].tolist() == [False, True]
    assert current["_particle_failure_info"][1]["reason"] == "aperture_plane_loss"
    assert current["_particle_failure_info"][1]["radius_at_wall_mm"] == pytest.approx(
        2.0
    )


def test_initial_radial_quantile_context_can_define_robust_envelope() -> None:
    initial = _state(x=[0.0, 1.0, 2.0])
    previous = _state(x=[0.0, 1.0, 2.0])
    current = _state(x=[0.0, 1.0, 3.0])
    config = ParticleLossConfig(
        enabled=True,
        initial_radial_quantile=0.5,
        initial_radial_multiplier=2.0,
        initial_radial_margin_mm=0.25,
    )
    context = build_particle_loss_context(config, initial)

    assert context.initial_radial_loss_radius_mm == pytest.approx(2.25)
    marked = mark_particle_losses(
        current,
        previous,
        step=7,
        config=config,
        context=context,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        wall_z=0.0,
        aperture_radius=1.0,
    )

    assert marked == 1
    assert (
        current["_particle_failure_info"][2]["reason"]
        == "initial_radial_envelope_exceeded"
    )


def test_soa_gather_zeroes_charge_only_after_retarded_sample_is_dead() -> None:
    trajectory = [
        _state(x=[0.0, 1.0], q=[3.0, 4.0], dead=[False, False]),
        _state(x=[0.0, 1.0], q=[3.0, 0.0], dead=[False, True]),
    ]
    builder = TrajectoryBuilder(2, 2)
    for step, state in enumerate(trajectory):
        builder.set_step(step, state)
    soa = builder.build()

    before_loss = gather_external_samples_soa(soa, np.array([0, 0], dtype=int))
    after_loss = gather_external_samples_soa(soa, np.array([0, 1], dtype=int))

    assert before_loss.charge.tolist() == pytest.approx([3.0, 4.0])
    assert before_loss.valid_mask.tolist() == [True, True]
    assert after_loss.charge.tolist() == pytest.approx([3.0, 0.0])
    assert after_loss.valid_mask.tolist() == [True, False]
