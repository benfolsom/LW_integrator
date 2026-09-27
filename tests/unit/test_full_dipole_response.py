"""Retarded-root accuracy must follow source interval and coordinate scales."""

import numpy as np
import pytest

from core.full_dipole_response import response


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
@pytest.mark.parametrize("allow_boundary", [False, True])
def test_small_interval_interior_root_is_not_rounded_to_join(scale, allow_boundary):
    # The relative scales and root fraction reproduce the Task H rejected event.
    start = 1.8253167191560666e-6 * scale
    duration = 1.2936709272283963e-11 * scale
    fraction = 0.9996282150881548
    radius = 1e-9 * scale
    root = start + fraction * duration
    event = np.array([root + radius, radius, 0.0, 0.0])
    result = response(
        event,
        start,
        duration,
        np.zeros((2, 3)),
        np.zeros((1, 4, 4)),
        charge=1.0,
        allow_boundary=allow_boundary,
    )
    assert result["segment_fraction"] == pytest.approx(fraction, abs=1e-9)
    assert result["retarded_time"] == pytest.approx(root, rel=2e-15, abs=0.0)
    # The static Coulomb field remains correct under a change of length scale.
    assert result["field_tensor"][0, 1] * radius**2 == pytest.approx(-1.0, rel=1e-11)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_true_join_still_requires_explicit_boundary_policy(fraction):
    start, duration, radius = 2.0**-20, 2.0**-36, 2.0**-30
    event = [start + fraction * duration + radius, radius, 0.0, 0.0]
    args = (event, start, duration, np.zeros((2, 3)), np.zeros((1, 4, 4)))
    with pytest.raises(ValueError, match="smooth segment"):
        response(*args)
    result = response(*args, allow_boundary=True)
    assert result["segment_fraction"] == pytest.approx(fraction, abs=1e-9)
