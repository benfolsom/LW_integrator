"""Synthetic coasting kick: field seen by a passive observer across the kick cone.

Usage: python probe_synthetic.py [--strip]   (--strip removes kick metadata,
emulating the pre-fix smooth reconstruction)
"""

import copy
import sys

import numpy as np

sys.path.insert(0, ".")
from core.constants import C_MMNS as C  # noqa: E402
from core.exact_visibility_history import save_visibility_knots  # noqa: E402
from core.resolved_knot import RESOLVED_KNOT_FIELDS  # noqa: E402
from core.retarded_fields import (  # noqa: E402
    ObserverEvent,
    evaluate_retarded_charge_field_native,
)
from tests.unit.test_inertial_prehistory import _state  # noqa: E402

STRIP = "--strip" in sys.argv
import os
BACKEND = os.environ.get("BACKEND", "python")
B1 = np.array([0.0, 0.0, 0.8])
B2 = np.array([0.0, 0.02, 0.7])  # recoil changes direction and speed
T_KICK = 1.0


def row_at(t, beta, x):
    s = _state(position_mm=tuple(x), beta=tuple(beta), observer_charge=1.0)
    s["t"] = np.array([t])
    return s


def history(dt, kick=True, branch=None):
    """Rows on [0, 2*T_KICK]; kick at T_KICK unless a pure branch is asked."""
    n = int(round(2 * T_KICK / dt))
    rows = []
    for k in range(n + 1):
        t = k * dt
        if branch == 1:
            beta, x = B1, B1 * C * t
        elif branch == 2:
            beta, x = B2, B1 * C * T_KICK + B2 * C * (t - T_KICK)
        elif t < T_KICK - 1e-12:
            beta, x = B1, B1 * C * t
        else:
            beta, x = B2, B1 * C * T_KICK + B2 * C * (t - T_KICK)
        rows.append(row_at(t, beta, x))
    if kick and branch is None:
        k = int(round(T_KICK / dt))
        st = rows[k]
        knot = dict(st)
        knot["_kick_mask"] = np.array([True])
        knot["_kick_beta_before"] = B1[None, :].copy()
        res = np.zeros((1, len(RESOLVED_KNOT_FIELDS)))
        g1 = 1 / np.sqrt(1 - B1 @ B1)
        res[0, :3] = g1 * B1
        res[0, 3] = 1.0
        res[0, 12] = 1 / (g1 * (g1 + np.linalg.norm(g1 * B1)))
        knot["_kick_resolved_before"] = res
        knot["_gate_prime_before"] = np.full((1, 3), np.nan)
        knot["_gate_prime_after"] = np.full((1, 3), np.nan)
        if not STRIP:
            save_visibility_knots(st, [knot], append=True)
    return rows


def field(rows, t_obs, x_obs):
    r = evaluate_retarded_charge_field_native(
        rows, ObserverEvent(time_ns=t_obs, position_mm=tuple(x_obs)), backend=BACKEND
    )
    return np.asarray(r.electric_field_native).reshape(-1)[:3], r


x_obs = np.array([30.0, 0.0, 120.0])
x_kick = B1 * C * T_KICK
t_cone = T_KICK + np.linalg.norm(x_obs - x_kick) / C
offsets = np.linspace(-0.3, 0.3, 241)
for dt in (0.1, 0.05, 0.025):
    rows = history(dt)
    ref1 = history(dt, branch=1)
    ref2 = history(dt, branch=2)
    worst = 0.0
    fluence = 0.0
    for off in offsets:
        t_obs = t_cone + off
        e, _ = field(rows, t_obs, x_obs)
        ref = field(ref1 if off < 0 else ref2, t_obs, x_obs)[0]
        d = np.linalg.norm(e - ref)
        worst = max(worst, d / np.linalg.norm(ref))
        fluence += d**2 * (offsets[1] - offsets[0])
    print(
        f"{'strip' if STRIP else 'knots'} dt={dt}: max |E-E_branch|/|E|={worst:.3e}"
        f"  sum|dE|^2 dt={fluence:.3e}"
    )
