"""Constant-width breathing vs rigid cloud on an ACCELERATED centre.

A single driver macro has zero ensemble variance, so its breathing target is
constant (sigma' = sigma'' = 0) and its children should trace the same
worldlines as the rigid cloud. The rider is charged so the driver accelerates.
The repo test covers breathing only on an inertial centre.
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.unit.test_exact_source_cloud import cloud_config  # noqa: E402
from tests.unit.test_exact_source_reduction import magnetic  # noqa: E402
from tests.unit.test_exact_visibility_gates import crossing_run, gate  # noqa: E402
from tests.unit.test_inertial_prehistory import _state  # noqa: E402

for backend in ("python", "numba_analytic_charge_response_serial"):
    out = {}
    for breathing in (False, True):
        out[breathing] = crossing_run(
            16, gate(), backend=backend, duration=0.15,
            rider_state=_state(position_mm=(0, 0, 10), beta=(0, 0, 0.05), observer_charge=10.0, source_charge=10.0),
            driver_state=_state(position_mm=(0, 0.5, 0), beta=(0, 0.1, 0), observer_charge=5.0, source_charge=10.0),
            macroparticle_smearing=replace(cloud_config(4, 0.2), breathing_enabled=True) if breathing else cloud_config(4, 0.2),
            magnetic_dipole=replace(magnetic(0), exact_retarded_backend=backend),
        )
    for role, k in (("rider", 0), ("driver", 1)):
        a, b = out[False][k][-1], out[True][k][-1]
        p0 = out[False][k][0]
        dp = lambda s: np.array([s["P" + c] for c in "xyz"]) - np.array([p0["P" + c] for c in "xyz"])
        kick = np.linalg.norm(dp(a))
        print(f"{backend:40s} {role}: |kick| {kick:.6e}, |kick_breathing - kick_rigid|/|kick| = {np.linalg.norm(dp(b) - dp(a)) / kick:.2e}")
