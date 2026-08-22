from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from core.beamline_geometry import compute_visibility_mask
from core.types import BeamlineGeometryConfig, Occluder


def _z_pipe(radius: float = 15.0, length: float = 2000.0) -> Occluder:
    return Occluder(
        axis=(0.0, 0.0, 1.0),
        center_mm=(0.0, 0.0, 0.0),
        radius_mm=radius,
        length_mm=length,
        label="electron_pipe",
    )


def _y_pipe(radius: float = 15.0, length: float = 2000.0) -> Occluder:
    return Occluder(
        axis=(0.0, 1.0, 0.0),
        center_mm=(0.0, 0.0, 0.0),
        radius_mm=radius,
        length_mm=length,
        label="driver_pipe",
    )


def test_source_inside_z_cylinder_is_visible():
    geometry = BeamlineGeometryConfig(enabled=True, occluders=[_z_pipe()])
    positions = np.array([[0.0, 5.0, 0.0]])
    mask = compute_visibility_mask(positions, geometry)
    assert_array_equal(mask, np.array([True]))


def test_source_outside_transverse_aperture_is_occluded():
    geometry = BeamlineGeometryConfig(enabled=True, occluders=[_z_pipe()])
    positions = np.array([[0.0, 20.0, 0.0]])
    mask = compute_visibility_mask(positions, geometry)
    assert_array_equal(mask, np.array([False]))


def test_source_beyond_axial_extent_is_occluded():
    geometry = BeamlineGeometryConfig(enabled=True, occluders=[_z_pipe()])
    # |z| = 1001 > L/2 = 1000
    positions = np.array([[0.0, 5.0, 1001.0]])
    mask = compute_visibility_mask(positions, geometry)
    assert_array_equal(mask, np.array([False]))


def test_source_inside_one_of_two_occluders_is_visible():
    # Source outside the z-axis cylinder (|y| > R) but inside the y-axis
    # cylinder (|z| < R, |y| < L/2).
    geometry = BeamlineGeometryConfig(
        enabled=True, occluders=[_z_pipe(), _y_pipe()]
    )
    positions = np.array([[0.0, 20.0, 5.0]])
    mask = compute_visibility_mask(positions, geometry)
    assert_array_equal(mask, np.array([True]))


def test_disabled_geometry_returns_all_visible():
    geometry = BeamlineGeometryConfig(
        enabled=False, occluders=[_z_pipe()]
    )
    positions = np.array([[0.0, 100.0, 0.0], [0.0, 5.0, 0.0]])
    mask = compute_visibility_mask(positions, geometry)
    assert_array_equal(mask, np.array([True, True]))


def test_empty_occluder_list_returns_all_visible():
    geometry = BeamlineGeometryConfig(enabled=True, occluders=[])
    positions = np.array([[0.0, 100.0, 0.0], [0.0, 5.0, 0.0]])
    mask = compute_visibility_mask(positions, geometry)
    assert_array_equal(mask, np.array([True, True]))


def test_study_plan_example_electron_pipe_visibility():
    # Study-plan example: z-axis cylinder (R=15mm, L=2000mm) and y-axis
    # cylinder (R=15mm, L=2000mm) centered at origin.
    #
    # A driver particle at (0, 5, 0) is inside the z-axis cylinder
    # (electron pipe) -> visible for the driver->rider line of sight.
    #
    # A driver particle at (0, 20, 0) is outside the z-axis cylinder but
    # inside the y-axis cylinder. For the driver->rider line of sight, only
    # the z-axis cylinder (electron pipe) matters, so this point should be
    # occluded when only the electron pipe is passed to the mask.
    #
    # NOTE: compute_visibility_mask as implemented checks ALL occluders with
    # OR. This is correct for the general "is this point inside any pipe"
    # question. The per-pair direction-specific gating (driver->rider uses
    # only the electron pipe; rider->driver uses only the driver pipe) is
    # handled at the call site, not here. This test therefore passes only
    # the electron pipe to emulate the driver->rider call site.
    electron_pipe = _z_pipe()
    geometry = BeamlineGeometryConfig(enabled=True, occluders=[electron_pipe])

    positions = np.array([[0.0, 5.0, 0.0], [0.0, 20.0, 0.0]])
    mask = compute_visibility_mask(positions, geometry)
    assert_array_equal(mask, np.array([True, False]))


def test_study_plan_example_or_semantics_with_both_pipes():
    # Companion to the above: when BOTH pipes are passed, the (0, 20, 0)
    # point is inside the y-axis driver pipe and therefore visible under
    # the OR semantics. This confirms that direction-specific gating must
    # be applied at the call site by selecting which occluders to pass.
    geometry = BeamlineGeometryConfig(
        enabled=True, occluders=[_z_pipe(), _y_pipe()]
    )
    positions = np.array([[0.0, 5.0, 0.0], [0.0, 20.0, 0.0]])
    mask = compute_visibility_mask(positions, geometry)
    assert_array_equal(mask, np.array([True, True]))


def test_occluder_validates_inputs():
    with pytest.raises(ValueError):
        Occluder(
            axis=(0.0, 0.0, 0.0),
            center_mm=(0.0, 0.0, 0.0),
            radius_mm=10.0,
            length_mm=100.0,
        )
    with pytest.raises(ValueError):
        Occluder(
            axis=(0.0, 0.0, 1.0),
            center_mm=(0.0, 0.0, 0.0),
            radius_mm=-1.0,
            length_mm=100.0,
        )
    with pytest.raises(ValueError):
        Occluder(
            axis=(0.0, 0.0, 1.0),
            center_mm=(0.0, 0.0, 0.0),
            radius_mm=10.0,
            length_mm=0.0,
        )


def test_occluder_axis_is_normalized():
    occluder = Occluder(
        axis=(0.0, 0.0, 5.0),
        center_mm=(0.0, 0.0, 0.0),
        radius_mm=10.0,
        length_mm=100.0,
    )
    assert occluder.axis == pytest.approx((0.0, 0.0, 1.0))


def test_visibility_mask_handles_single_1d_position():
    geometry = BeamlineGeometryConfig(enabled=True, occluders=[_z_pipe()])
    positions = np.array([0.0, 5.0, 0.0])
    mask = compute_visibility_mask(positions, geometry)
    assert_array_equal(mask, np.array([True]))
