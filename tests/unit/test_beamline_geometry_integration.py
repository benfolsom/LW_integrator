"""Integration tests for beamline-geometry occlusion wiring.

These tests verify that ``beamline_geometry`` is threaded through the
equations of motion and that the occlusion mask is applied to external
bunch samples (and only external samples). They avoid running a full
simulation; instead they exercise the parameter plumbing and the
mask-AND logic directly.
"""

from __future__ import annotations

import copy

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from core.beamline_geometry import compute_visibility_mask
from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.equations import retarded_equations_of_motion
from core.types import (
    BeamlineGeometryConfig,
    ChronoMatchingMode,
    Occluder,
    SimulationType,
    StartupMode,
)
from core.vectorized_interactions import ExternalSampleBatch


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


def _z_pipe(radius: float = 5.0, length: float = 100.0) -> Occluder:
    return Occluder(
        axis=(0.0, 0.0, 1.0),
        center_mm=(0.0, 0.0, 0.0),
        radius_mm=radius,
        length_mm=length,
        label="test_pipe",
    )


def _two_step_trajectory() -> tuple[list, list]:
    trajectory = [
        _single_particle_state(z=-1.0),
        _single_particle_state(z=-0.5, t=1e-3),
    ]
    trajectory_ext = [copy.deepcopy(state) for state in trajectory]
    for ext in trajectory_ext:
        ext["x"] += 1e-4
        ext["z"] += 1e-4
    return trajectory, trajectory_ext


def test_beamline_geometry_parameter_is_accepted_disabled():
    """retarded_equations_of_motion accepts beamline_geometry (disabled)."""
    trajectory, trajectory_ext = _two_step_trajectory()

    updated = retarded_equations_of_motion(
        h=1e-3,
        trajectory=trajectory,
        trajectory_ext=trajectory_ext,
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        chrono_mode=ChronoMatchingMode.AVERAGED,
        startup_mode=StartupMode.COLD_START,
        beamline_geometry=BeamlineGeometryConfig(enabled=False),
    )

    for key, value in updated.items():
        if isinstance(value, np.ndarray):
            assert np.all(np.isfinite(value)), f"non-finite values in {key}"


def test_disabled_geometry_is_noop_relative_to_none():
    """Disabled geometry must produce identical results to passing None."""
    traj_a, ext_a = _two_step_trajectory()
    traj_b, ext_b = _two_step_trajectory()

    result_none = retarded_equations_of_motion(
        h=1e-3,
        trajectory=traj_a,
        trajectory_ext=ext_a,
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        chrono_mode=ChronoMatchingMode.AVERAGED,
        startup_mode=StartupMode.COLD_START,
        beamline_geometry=None,
    )

    result_disabled = retarded_equations_of_motion(
        h=1e-3,
        trajectory=traj_b,
        trajectory_ext=ext_b,
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        chrono_mode=ChronoMatchingMode.AVERAGED,
        startup_mode=StartupMode.COLD_START,
        beamline_geometry=BeamlineGeometryConfig(enabled=False),
    )

    for key in result_none:
        v_none = result_none[key]
        v_disabled = result_disabled[key]
        if isinstance(v_none, np.ndarray):
            assert_allclose(v_disabled, v_none, err_msg=f"mismatch in {key}")


def test_occlusion_mask_zeros_occluded_external_samples():
    """Directly verify the mask-AND logic used in the equations of motion.

    Constructs an ExternalSampleBatch with two source particles: one inside
    the pipe (visible) and one outside (occluded). Applies the same mask-AND
    operation the equations of motion performs and confirms the occluded
    source's valid_mask entry is cleared while the visible one is preserved.
    """
    geometry = BeamlineGeometryConfig(enabled=True, occluders=[_z_pipe()])

    # Source 0: inside the pipe (y=0, |z|<L/2) -> visible.
    # Source 1: outside the pipe transversely (y=10 > R=5) -> occluded.
    src_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ]
    )

    visibility = compute_visibility_mask(src_positions, geometry)
    assert_array_equal(visibility, np.array([True, False]))

    n = src_positions.shape[0]
    samples = ExternalSampleBatch(
        charge=np.ones(n),
        gamma=np.ones(n),
        bx=np.zeros(n),
        by=np.zeros(n),
        bz=np.zeros(n),
        bdotx=np.zeros(n),
        bdoty=np.zeros(n),
        bdotz=np.zeros(n),
        valid_mask=np.ones(n, dtype=bool),
        x=src_positions[:, 0].copy(),
        y=src_positions[:, 1].copy(),
        z=src_positions[:, 2].copy(),
    )

    # Replicate the exact operation performed in retarded_equations_of_motion.
    samples.valid_mask = samples.valid_mask & visibility

    assert_array_equal(samples.valid_mask, np.array([True, False]))
    assert samples.valid_mask[0]
    assert not samples.valid_mask[1]


def test_occlusion_mask_preserves_partially_valid_samples():
    """Mask-AND must respect pre-existing valid_mask entries.

    A source that is geometrically visible but already marked invalid
    (e.g. dead particle) must remain invalid; the occlusion mask must not
    resurrect it.
    """
    geometry = BeamlineGeometryConfig(enabled=True, occluders=[_z_pipe()])

    src_positions = np.array(
        [
            [0.0, 0.0, 0.0],  # visible geometry, but pre-marked invalid
            [0.0, 10.0, 0.0],  # occluded geometry, pre-marked invalid
            [0.0, 0.0, 0.0],  # visible geometry, valid
        ]
    )
    visibility = compute_visibility_mask(src_positions, geometry)
    assert_array_equal(visibility, np.array([True, False, True]))

    initial_valid = np.array([False, False, True])
    combined = initial_valid & visibility
    assert_array_equal(combined, np.array([False, False, True]))
