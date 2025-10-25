"""Unit coverage for helper utilities in `core.trajectory_integrator`.

These tests focus on deterministic behaviours that were previously exercised only
through the large regression runs.  The lightweight checks below ensure that the
helper utilities continue to operate correctly even when run in isolation.
"""

from __future__ import annotations

import copy
import random
from typing import Dict

import numpy as np
import pytest

from core.distances import _compute_delta_t
from core.self_consistency import SelfConsistencyConfig
from core.trajectory_integrator import (
    C_MMNS,
    IntegratorConfig,
    ParticleState,
    SimulationType,
    chrono_match_indices,
    generate_conducting_image,
    generate_switching_image,
    retarded_equations_of_motion,
    retarded_integrator,
    run_integrator,
)
from core.types import ChronoMatchingMode, StartupMode


def _make_single_particle_state(
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = -1.0,
    t: float = 0.0,
    px: float = 0.0,
    py: float = 0.0,
    pz: float = 0.0,
    charge: float = 0.0,
    mass: float = 1.0,
    gamma: float = 1.0,
    bx: float = 0.0,
    by: float = 0.0,
    bz: float = 0.0,
    char_time: float = 1e-3,
) -> ParticleState:
    """Construct a minimal, fully-populated `ParticleState` for tests."""

    arr = np.array
    state: Dict[str, np.ndarray] = {
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
    return state


def test_compute_delta_t_fast_matches_legacy_formula():
    trajectory = [_make_single_particle_state(t=1e-3, x=0.0, bx=0.2)]
    trajectory_ext = [_make_single_particle_state(t=1e-3, x=0.0, bx=0.2)]

    distance = 0.75
    b_nhat = 0.35

    delta_t = _compute_delta_t(
        mode=ChronoMatchingMode.FAST,
        distance=distance,
        b_nhat=b_nhat,
        sample_index=0,
        index_traj=0,
        index_part=0,
        trajectory=trajectory,
        trajectory_ext=trajectory_ext,
    )

    expected = distance * (1.0 + b_nhat) / C_MMNS
    assert delta_t == pytest.approx(expected)


def test_compute_delta_t_averaged_blends_stationary_and_relativistic_samples():
    trajectory = [
        _make_single_particle_state(t=0.0, x=0.0),
        _make_single_particle_state(t=4e-3, x=0.5),
        _make_single_particle_state(t=8e-3, x=1.0),
    ]
    ultrarelativistic_beta = 0.9
    trajectory_ext = [
        _make_single_particle_state(t=0.0, x=0.0, bx=ultrarelativistic_beta),
        _make_single_particle_state(t=4e-3, x=0.0, bx=ultrarelativistic_beta),
        _make_single_particle_state(t=8e-3, x=0.0, bx=0.0),
    ]

    distance = 1.0
    b_nhat_current = trajectory_ext[2]["bx"][0]

    fast_delta = _compute_delta_t(
        mode=ChronoMatchingMode.FAST,
        distance=distance,
        b_nhat=b_nhat_current,
        sample_index=0,
        index_traj=2,
        index_part=0,
        trajectory=trajectory,
        trajectory_ext=trajectory_ext,
    )

    averaged_delta = _compute_delta_t(
        mode=ChronoMatchingMode.AVERAGED,
        distance=distance,
        b_nhat=b_nhat_current,
        sample_index=0,
        index_traj=2,
        index_part=0,
        trajectory=trajectory,
        trajectory_ext=trajectory_ext,
    )

    expected_fast = distance * (1.0 + b_nhat_current) / C_MMNS
    expected_avg = (
        distance * (1.0 + (b_nhat_current + ultrarelativistic_beta) / 2.0) / C_MMNS
    )

    assert fast_delta == pytest.approx(expected_fast)
    assert averaged_delta == pytest.approx(expected_avg)
    assert averaged_delta > fast_delta


def test_cold_start_defers_external_forces_until_travelled_distance():
    trajectory = [_make_single_particle_state(t=0.0, z=0.0, charge=1.0)]
    trajectory_ext = [_make_single_particle_state(t=0.0, z=1.0, x=0.5, charge=1.0)]

    cold_result = retarded_equations_of_motion(
        h=1e-3,
        trajectory=trajectory,
        trajectory_ext=trajectory_ext,
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        chrono_mode=ChronoMatchingMode.AVERAGED,
        startup_mode=StartupMode.COLD_START,
    )

    np.testing.assert_allclose(cold_result["Px"], trajectory[0]["Px"])


def test_approximate_back_history_recovers_initial_force_estimate():
    trajectory = [_make_single_particle_state(t=0.0, z=0.0, charge=1.0)]
    trajectory_ext = [
        _make_single_particle_state(t=0.0, z=1.0, x=0.5, bx=0.2, charge=-1.0)
    ]

    approx_result = retarded_equations_of_motion(
        h=1e-3,
        trajectory=trajectory,
        trajectory_ext=trajectory_ext,
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        chrono_mode=ChronoMatchingMode.AVERAGED,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
    )

    assert not np.allclose(approx_result["Px"], trajectory[0]["Px"])
    assert np.all(np.isfinite(approx_result["Px"]))


def test_generate_conducting_image_reflects_momentum_and_direction():
    random.seed(1234)
    source = _make_single_particle_state(z=-2.0, pz=1.5, bz=0.25, charge=-1.0)

    image = generate_conducting_image(source, wall_z=0.0, aperture_radius=0.5)

    assert len(image["z"]) == 12
    assert image["z"][0] == pytest.approx(2.0)
    assert image["Pz"][0] == pytest.approx(-source["Pz"][0])
    assert image["bz"][0] == pytest.approx(-source["bz"][0])
    assert image["t"][0] == pytest.approx(source["t"][0])

    R_dist = abs(2.0 - source["z"][0])
    reduction = 1 - 2 * (0.5**2) / (R_dist**2) * 1 / (1 - np.cos(np.pi / 2))
    expected_total_charge = source["q"][0] * reduction
    assert image["q"].sum() == pytest.approx(expected_total_charge)


def test_generate_conducting_image_respects_custom_subcharge_count():
    source = _make_single_particle_state(z=-1.5, charge=2.0)

    image = generate_conducting_image(
        source, wall_z=0.0, aperture_radius=0.5, subcharge_count=8
    )

    assert len(image["x"]) == 8
    np.testing.assert_allclose(image["z"], 1.5)

    mirrored_z = 0.0 + abs(0.0 - source["z"][0])
    R_dist = abs(mirrored_z - source["z"][0])
    cos_argument = 1.0 - 2.0 * (0.5**2) / (R_dist**2)
    theta = np.arccos(np.clip(cos_argument, -1.0, 1.0))
    shift = 2 * R_dist * np.tan(theta)
    reduction = 1 - 2 * (0.5**2) / (R_dist**2) * 1 / (1 - np.cos(np.pi / 2))
    assert image["q"].sum() == pytest.approx(source["q"][0] * reduction)
    np.testing.assert_allclose(
        np.hypot(image["x"] - source["x"][0], image["y"] - source["y"][0]),
        shift,
    )


def test_generate_conducting_image_rejects_out_of_range_subcharge_count():
    source = _make_single_particle_state()

    with pytest.raises(ValueError):
        generate_conducting_image(source, wall_z=0.0, aperture_radius=0.5, subcharge_count=2)
    with pytest.raises(ValueError):
        generate_conducting_image(source, wall_z=0.0, aperture_radius=0.5, subcharge_count=256)


def test_generate_switching_image_respects_cutoff_and_reflection():
    base = _make_single_particle_state(z=-1.2, pz=-0.75, bz=-0.1, charge=1.0)

    reflected = generate_switching_image(
        base,
        wall_z=0.0,
        aperture_radius=0.25,
        cut_z=0.0,
    )

    assert reflected["z"][0] == pytest.approx(1.2)
    assert reflected["Pz"][0] == pytest.approx(0.75)
    assert reflected["bz"][0] == pytest.approx(0.1)

    beyond_cut = _make_single_particle_state(z=0.6, charge=1.0)
    truncated = generate_switching_image(
        beyond_cut,
        wall_z=0.0,
        aperture_radius=0.25,
        cut_z=0.5,
    )
    assert np.allclose(truncated["q"], 0.0)


def test_chrono_match_indices_returns_bounded_results():
    trajectory = []
    trajectory_ext = []
    for step, (time_value, position) in enumerate(
        [(0.0, -1.0), (2e-3, -0.5), (4e-3, 0.0)]
    ):
        state = _make_single_particle_state(z=position, t=time_value)
        # Give the external trajectory a slight offset to keep distances finite.
        ext_state = _make_single_particle_state(z=position + 0.1, t=time_value)
        state["x"][0] = step * 0.1
        ext_state["x"][0] = step * 0.1 + 0.05
        trajectory.append(state)
        trajectory_ext.append(ext_state)

    indices_averaged = chrono_match_indices(
        trajectory, trajectory_ext, index_traj=2, index_part=0
    )
    indices_fast = chrono_match_indices(
        trajectory,
        trajectory_ext,
        index_traj=2,
        index_part=0,
        mode=ChronoMatchingMode.FAST,
    )

    for indices in (indices_averaged, indices_fast):
        assert indices.shape == (1,)
        assert indices.dtype.kind == "i"
        assert 0 <= int(indices[0]) <= 2


def test_retarded_equations_of_motion_keeps_zero_charge_state_stable():
    trajectory = [
        _make_single_particle_state(t=0.0),
        _make_single_particle_state(t=1e-3),
    ]
    trajectory_ext = [copy.deepcopy(state) for state in trajectory]
    for ext_state in trajectory_ext:
        ext_state["x"] += 1e-4
        ext_state["z"] += 1e-4

    result = retarded_equations_of_motion(
        h=1e-3,
        trajectory=trajectory,
        trajectory_ext=trajectory_ext,
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        chrono_mode=ChronoMatchingMode.AVERAGED,
        startup_mode=StartupMode.COLD_START,
    )

    assert np.all(np.isfinite(result["x"]))
    assert np.all(np.isfinite(result["gamma"]))
    assert result["t"][0] == pytest.approx(trajectory[0]["t"][0] + 1e-3)
    assert result["z"][0] == pytest.approx(trajectory[0]["z"][0])


def test_run_integrator_matches_direct_invocation_for_simple_case():
    rider = _make_single_particle_state(z=-1.0)
    config = IntegratorConfig(
        steps=3,
        time_step=1e-3,
        wall_position=0.0,
        aperture_radius=0.5,
        simulation_type=SimulationType.CONDUCTING_WALL,
        bunch_mean=0.0,
        cavity_spacing=0.0,
        z_cutoff=0.0,
    )

    direct_traj, direct_drv = retarded_integrator(
        steps=config.steps,
        h_step=config.time_step,
        wall_z=config.wall_position,
        aperture_radius=config.aperture_radius,
        sim_type=config.simulation_type,
        init_rider=copy.deepcopy(rider),
        init_driver=None,
        mean=config.bunch_mean,
        cav_spacing=config.cavity_spacing,
        z_cutoff=config.z_cutoff,
        self_consistency=SelfConsistencyConfig(
            enabled=True, tolerance=1e-9, max_iterations=2
        ),
    )

    wrapped_traj, wrapped_drv = run_integrator(config, copy.deepcopy(rider), None)

    assert len(direct_traj) == config.steps
    assert len(wrapped_traj) == config.steps
    np.testing.assert_allclose(direct_traj[-1]["z"], wrapped_traj[-1]["z"])
    np.testing.assert_allclose(direct_drv[-1]["z"], wrapped_drv[-1]["z"])
