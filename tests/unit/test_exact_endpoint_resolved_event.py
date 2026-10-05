"""Endpoint potentials must use the same resolved observer event as forces."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.exact_pair_endpoint import evaluate_exact_endpoint_four_potential
from core.types import MagneticDipoleConfig


def test_endpoint_potential_at_resolved_position_matches_static_charge():
    source = []
    for time_ns in (0.0, 3.0):
        row = {key: np.zeros(1) for key in ("x", "y", "z", "bx", "by", "bz")}
        row.update({key: np.zeros(1) for key in ("bdotx", "bdoty", "bdotz")})
        row.update(t=np.array([time_ns]), q=np.ones(1), q_source=np.ones(1))
        source.append(row)
    observer = {
        "t": np.array([2.0]),
        "x": np.array([1.0]),
        "y": np.zeros(1),
        "z": np.zeros(1),
        "source_time_low_ns": np.array([0.125]),
        "source_time_tail_ns": np.array([0.0625]),
        "source_position_low_x": np.array([0.125]),
        "source_position_tail_x": np.array([0.0625]),
        "_exact_source_endpoint_rebase_required": np.array([True]),
    }
    potential = evaluate_exact_endpoint_four_potential(
        observer,
        source,
        magnetic_dipole=MagneticDipoleConfig(enabled=True),
        include_dipole_source=False,
    )
    # A unit stationary native charge has phi = 1/r and zero vector potential.
    np.testing.assert_allclose(
        potential[0], [1.0 / 1.1875, 0.0, 0.0, 0.0], rtol=1e-13, atol=0.0
    )
