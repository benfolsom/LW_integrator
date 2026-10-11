"""Cross-bunch endpoint potentials across the bunches' lab-time spread."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.exact_pair_endpoint import evaluate_exact_endpoint_four_potential
from core.retarded_fields import RetardedHistoryError
from core.types import MagneticDipoleConfig

pytestmark = pytest.mark.unit

_BETA = np.array([0.05, -0.02, 0.08])


def _history(t_last_ns: float) -> list[dict[str, np.ndarray]]:
    rows = []
    for time_ns in np.linspace(-0.02, t_last_ns, 161):
        position = _BETA * C_MMNS * time_ns
        rows.append(
            {
                "t": np.array([time_ns]),
                "x": np.array([position[0]]),
                "y": np.array([position[1]]),
                "z": np.array([position[2]]),
                "bx": np.array([_BETA[0]]),
                "by": np.array([_BETA[1]]),
                "bz": np.array([_BETA[2]]),
                "bdotx": np.zeros(1),
                "bdoty": np.zeros(1),
                "bdotz": np.zeros(1),
                "q": np.array([ELEMENTARY_CHARGE]),
                "q_source": np.array([ELEMENTARY_CHARGE]),
                "_dead_particles": np.zeros(1, dtype=bool),
            }
        )
    return rows


def _observer(t_ns: float) -> dict[str, np.ndarray]:
    # A few nm from the source's position at t_ns: closer than c times the
    # 1e-6 ns by which the observer is ahead of the source's last knot.
    position = _BETA * C_MMNS * t_ns + np.array([2.0e-6, 1.0e-6, 0.0])
    return {
        "x": np.array([position[0]]),
        "y": np.array([position[1]]),
        "z": np.array([position[2]]),
        "t": np.array([t_ns]),
        "_exact_source_endpoint_rebase_required": np.array([True]),
    }


def test_cross_bunch_endpoint_extrapolates_across_lab_time_spread() -> None:
    t_last = 0.004
    observer = _observer(t_last + 1.0e-6)
    config = MagneticDipoleConfig()
    with pytest.raises(RetardedHistoryError):
        evaluate_exact_endpoint_four_potential(
            observer, _history(t_last), magnetic_dipole=config, include_dipole_source=False
        )
    extrapolated = evaluate_exact_endpoint_four_potential(
        observer,
        _history(t_last),
        magnetic_dipole=config,
        include_dipole_source=False,
        cross_extrapolate_ns=2.0e-6,
    )
    reference = evaluate_exact_endpoint_four_potential(
        observer, _history(t_last + 1.0e-3), magnetic_dipole=config, include_dipole_source=False
    )
    np.testing.assert_allclose(extrapolated, reference, rtol=1e-9)
