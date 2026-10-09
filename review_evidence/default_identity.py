"""Hash default-path outputs; run once per checkout (PYTHONPATH selects it).

Usage: python default_identity.py <checkout> <out.json>
Compare the printed digests between development and the branch.
"""
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import core  # noqa: E402

assert Path(core.__file__).resolve().parents[1] == root, core.__file__


def _feed(h, value, key=""):
    if isinstance(value, dict):
        for k in sorted(value):
            _feed(h, value[k], f"{key}/{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _feed(h, v, f"{key}[{i}]")
    elif isinstance(value, np.ndarray) and value.dtype != object:
        h.update(key.encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (float, int, np.floating, np.integer)):
        h.update(f"{key}={float(value).hex()}".encode())


def digest(obj):
    h = hashlib.sha256()
    _feed(h, obj)
    return h.hexdigest()[:16]


from tests.unit.test_exact_source_cloud import cloud_config  # noqa: E402
from tests.unit.test_exact_visibility_gates import crossing_run, gate  # noqa: E402

out = {}
for backend in ("python", "numba_analytic_charge_response_serial"):
    r = crossing_run(8, gate(), backend=backend)
    out[f"exact_point_{backend}"] = digest(r[:2])
    try:
        r = crossing_run(8, gate(), backend=backend, macroparticle_smearing=cloud_config(4, 0.2))
        out[f"exact_rigid_cloud_{backend}"] = digest(r[:2])
    except NotImplementedError:
        out[f"exact_rigid_cloud_{backend}"] = "rejected"

from lw_integrator.testbed_runner import load_config, run_testbed  # noqa: E402

for name in ("example_b2b_counter_propagating_proton_bunches.json",
             "example_b2b_relativistic_proton_stationary_lead_ion.json"):
    opts = load_config(root / "configs/run_configs" / name)
    opts = replace(opts, steps=40, output_dir=tempfile.mkdtemp(), trajectory_save=False)
    res = run_testbed(opts)
    out[name] = digest([res.rider_trajectory, res.metrics, res.rider_gamma_final, res.driver_gamma_final])
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
