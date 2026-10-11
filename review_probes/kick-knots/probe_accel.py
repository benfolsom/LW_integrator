"""Kick between two smoothly accelerated branches (constant lab force along z).

Compares the kicked history with each analytically extended smooth branch at
the same timestep. A correct split leaves only reconstruction error that
vanishes under refinement; a spurious pulse would grow as dt shrinks.
Also exercises the non-light-cone quintic path used by dipole sources.
"""

import os
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, ".")
from core.constants import C_MMNS as C  # noqa: E402
from core.exact_visibility_history import save_visibility_knots  # noqa: E402
from core.resolved_knot import RESOLVED_KNOT_FIELDS  # noqa: E402
from core.retarded_fields import (  # noqa: E402
    ObserverEvent,
    _extract_history,
    _prepare_source_history,
    _quintic_worldline_sample,
    evaluate_retarded_charge_field_native,
)
from core.exact_visibility_history import expanded_visibility_history  # noqa: E402
from tests.unit.test_inertial_prehistory import _state  # noqa: E402

T_KICK = 1.0
U0 = np.array([0.0, 0.0, 1.2])
A = np.array([0.0, 0.0, 0.6])  # d u / d t  (1/ns)
DU = np.array([0.0, 0.05, -0.3])  # recoil in proper velocity
BACKEND = os.environ.get("BACKEND", "python")


def u_of(t, branch):
    base = U0 + A * t
    return base + DU if branch == 2 else base


def x_of(t, branch):
    # x(t) = x(T_KICK) + int_{T_KICK}^t c u/gamma; both branches share x(T_KICK)
    def vel(s, k, b):
        u = u_of(s, b)
        return C * u[k] / np.sqrt(1 + u @ u)

    xk = np.array([quad(vel, 0, T_KICK, args=(k, 1), epsabs=1e-14)[0] for k in range(3)])
    return xk + np.array(
        [quad(vel, T_KICK, t, args=(k, branch), epsabs=1e-14)[0] for k in range(3)]
    )


def row(t, branch):
    u = u_of(t, branch)
    g = np.sqrt(1 + u @ u)
    s = _state(position_mm=tuple(x_of(t, branch)), beta=tuple(u / g), observer_charge=1.0)
    s["t"] = np.array([t])
    s["gamma"] = np.array([g])
    return s


def history(dt, mode):
    n = int(round(2 * T_KICK / dt))
    k = int(round(T_KICK / dt))
    rows = []
    for i in range(n + 1):
        t = i * dt
        b = mode if mode in (1, 2) else (1 if i < k else 2)
        rows.append(row(t, b))
    if mode == "kick":
        st = rows[k]
        u1 = u_of(T_KICK, 1)
        g1 = np.sqrt(1 + u1 @ u1)
        knot = dict(st)
        knot["_kick_mask"] = np.array([True])
        knot["_kick_beta_before"] = (u1 / g1)[None, :]
        res = np.zeros((1, len(RESOLVED_KNOT_FIELDS)))
        res[0, :3] = g1 * (u1 / g1)
        res[0, 3] = 1.0
        res[0, 12] = 1 / (g1 * (g1 + np.linalg.norm(res[0, :3])))
        knot["_kick_resolved_before"] = res
        knot["_gate_prime_before"] = np.full((1, 3), np.nan)
        knot["_gate_prime_after"] = np.full((1, 3), np.nan)
        save_visibility_knots(st, [knot], append=True)
    return rows


def field(rows, t, x):
    r = evaluate_retarded_charge_field_native(
        rows, ObserverEvent(time_ns=t, position_mm=tuple(x)), backend=BACKEND
    )
    return np.asarray(r.electric_field_native).reshape(-1)[:3]


xk = x_of(T_KICK, 1)
x_obs = xk + np.array([30.0, 0.0, -100.0])
t_cone = T_KICK + np.linalg.norm(x_obs - xk) / C
offsets = np.linspace(-0.25, 0.25, 101)
offsets = offsets[np.abs(offsets) > 1e-9]
for dt in (0.1, 0.05, 0.025):
    kick, r1, r2 = history(dt, "kick"), history(dt, 1), history(dt, 2)
    worst = fl = 0.0
    for off in offsets:
        e = field(kick, t_cone + off, x_obs)
        ref = field(r1 if off < 0 else r2, t_cone + off, x_obs)
        d = np.linalg.norm(e - ref)
        worst = max(worst, d / np.linalg.norm(ref))
        fl += d**2 * (offsets[1] - offsets[0])
    # Non-light-cone quintic (dipole worldline) velocity at the knot sides.
    k = int(round(T_KICK / dt))
    src = _prepare_source_history(
        _extract_history(expanded_visibility_history(kick)), 0, light_cone_enabled=False
    )
    _, b_in, a_in = _quintic_worldline_sample(src, k - 1, T_KICK)
    _, b_out, a_out = _quintic_worldline_sample(src, k, T_KICK)
    u1, u2 = u_of(T_KICK, 1), u_of(T_KICK, 2)
    print(
        f"dt={dt}: field max rel diff={worst:.3e} fluence={fl:.3e} | quintic "
        f"|b_in-b1|={np.linalg.norm(b_in - u1/np.sqrt(1+u1@u1)):.1e} "
        f"|b_out-b2|={np.linalg.norm(b_out - u2/np.sqrt(1+u2@u2)):.1e} "
        f"a_in={a_in} a_out={a_out}"
    )
