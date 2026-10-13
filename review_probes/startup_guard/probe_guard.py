"""Targeted probes of core.startup_history_guard behaviour."""

import time
import warnings

import numpy as np

from core.constants import C_MMNS
from core.startup_history_guard import (
    StartupHistoryError,
    _pair_integrals,
    _unpack,
    check_startup_history,
)
from core.types import BeamlineGeometryConfig, Occluder, SimulationType, StartupMode

B2B = SimulationType.BUNCH_TO_BUNCH
CS = StartupMode.COLD_START
ABH = StartupMode.APPROXIMATE_BACK_HISTORY


def bunch(pos, beta, q=1.0):
    pos = np.atleast_2d(np.asarray(pos, float))
    n = len(pos)
    beta = np.broadcast_to(np.asarray(beta, float), (n, 3))
    gam = 1 / np.sqrt(1 - np.sum(beta**2, 1))
    m = 938.272
    return {
        "x": pos[:, 0].copy(), "y": pos[:, 1].copy(), "z": pos[:, 2].copy(),
        "t": np.zeros(n), "m": np.full(n, m), "gamma": gam,
        "Px": gam * beta[:, 0] * m * C_MMNS, "Py": gam * beta[:, 1] * m * C_MMNS,
        "Pz": gam * beta[:, 2] * m * C_MMNS, "q": np.full(n, q),
        "bx": beta[:, 0].copy(), "by": beta[:, 1].copy(), "bz": beta[:, 2].copy(),
    }


def run(label, r, d, mode=CS, **kw):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rep = check_startup_history(r, d, sim_type=B2B, startup_mode=mode, **kw)
        out = f"PASS worst={rep.worst_ratio:.3g}"
    except StartupHistoryError as e:
        out = "FAIL " + str(e).split("t0 - R/c. ")[1][:260]
    print(f"[{label}] {out}")


def counter_h(sep, g1, g2, crossing_steps):
    b1 = np.sqrt(1 - 1 / g1**2)
    b2 = np.sqrt(1 - 1 / g2**2)
    return sep / ((g1 * b1 + g2 * b2) * C_MMNS * crossing_steps)


print("== A. AGENTS.md defaults: head-on, 1000 mm, 1200 steps, closing-speed h ==")
for gam in (1.05, 2.0, 100.0):
    b = np.sqrt(1 - 1 / gam**2)
    h = counter_h(1000.0, gam, gam, 600)
    rng = np.random.default_rng(0)
    rp = rng.normal(0, [0.01, 0.01, 0.1], (200, 3))
    dp = rng.normal(0, [0.01, 0.01, 0.1], (200, 3)) + [0, 0, 1000]
    run(f"gamma={gam} N=200", bunch(rp, (0, 0, b)), bunch(dp, (0, 0, -b)),
        steps=1200, h_step=h)

print("\n== B. Same physical config, run stopped earlier (h fixed, fewer steps) ==")
gam = 2.0
b = np.sqrt(1 - 1 / gam**2)
h = counter_h(1000.0, gam, gam, 600)
for steps in (1200, 800, 650, 500, 300):
    run(f"steps={steps}", bunch([[0, 0, 0]], (0, 0, b)),
        bunch([[0, 0.01, 1000]], (0, 0, -b)), steps=steps, h_step=h)

print("\n== C. Long source bunch: guard's per-pair tau_c vs integrator's all-source "
      "COLD_START gate ==")
# Rider single particle 50 mm ahead of the head of a 2000 mm driver train.
gam = 2.0
b = np.sqrt(1 - 1 / gam**2)
dz = np.linspace(50, 2050, 201)
dpos = np.stack([np.zeros_like(dz), np.full_like(dz, 0.01), dz], 1)
r = bunch([[0, 0, 0]], (0, 0, b))
d = bunch(dpos, (0, 0, -b))
h = counter_h(2050.0, gam, gam, 600)
run("long driver", r, d, steps=1200, h_step=h)
# Integrator gate: travel >= max_j beta R_j / (1 - beta.n_j) (all sources at once).
R = dz
gate_t = np.max(b * R / (1 + b)) / (b * C_MMNS)  # observer moving toward sources
tau_near = 50 / (C_MMNS * (1 + b))
print(f"    nearest-source tau_c = {tau_near:.4g} ns; integrator gate opens at "
      f"{gate_t:.4g} ns (max over sources)")
ro, dso = _unpack(r), _unpack(d)
horizon = 1199 * h * gam
num = den = num_gate = 0.0
for j in range(len(dz)):
    m_, t_, tc, _ = _pair_integrals(ro, 0, dso, j, horizon, None)
    num += m_
    den += t_
    # recompute with integrator window by sampling
    grid = np.linspace(0, min(gate_t, horizon), 4001)
    # integrate |F| on [0, gate_t] reusing helper via a tiny horizon trick
    m2, _, _, _ = _pair_integrals(ro, 0, dso, j, min(gate_t, horizon), None)
    num_gate += m2 if tc >= gate_t else _pair_integrals(
        ro, 0, dso, j, min(gate_t, horizon), None)[1]
print(f"    guard ratio (per-pair window)   = {num / den:.3g}")
print(f"    ratio with integrator gate window = {num_gate / den:.3g}")

print("\n== D. Visibility gate always open (pipe encloses both bunches) ==")
pipe = BeamlineGeometryConfig(
    enabled=True,
    occluders=[Occluder(axis=(0, 0, 1), center_mm=(0, 0, 500), radius_mm=50.0,
                        length_mm=1e6)],
)
gam = 2.0
b = np.sqrt(1 - 1 / gam**2)
h = counter_h(1000.0, gam, gam, 600)
r = bunch([[0, 0, 0]], (0, 0, b))
d = bunch([[0, 0.01, 1000]], (0, 0, -b))
run("no geometry", r, d, steps=1200, h_step=h)
run("geometry, always visible", r, d, steps=1200, h_step=h, beamline_geometry=pipe)
d_far = bunch([[0, 0.01, 1e5]], (0, 0, -b))
run("geometry, 100 m apart, run 10x longer", r, d_far, steps=12000,
    h_step=counter_h(1e5, gam, gam, 6000), beamline_geometry=pipe)

print("\n== E. Tolerance validation ==")
slow = bunch([[0, 0, 0]], (0, 0, 0.015))
src = bunch([[0, 60, 0]], (0, 0.06, 0))
for tol in (float("nan"), -1.0, float("inf")):
    run(f"tol={tol}", slow, src, steps=3000, h_step=1.9e-3, tolerance=tol)

print("\n== F. Co-moving pair, gamma=100, 1 mm apart, 1200 steps of 1 m ==")
gam = 100.0
b = np.sqrt(1 - 1 / gam**2)
h = 1000.0 / (gam * b * C_MMNS * 1199)
run("co-moving", bunch([[0, 0, 1]], (0, 0, b)), bunch([[0, 0, 0]], (0, 0, b)),
    steps=1200, h_step=h)

print("\n== G. Bound/near-static pair (beta ~ 1e-3), R0 = 1 mm, run 100 R0/c ==")
b = 1e-3
T = 100 * 1.0 / C_MMNS
run("slow pair", bunch([[0, 0, 0]], (b, 0, 0)), bunch([[0, 0, 1.0]], (-b, 0, 0)),
    steps=1000, h_step=T / 999)

print("\n== H. Observer subset: one halo observer close to the other bunch ==")
gam = 1.2
b = np.sqrt(1 - 1 / gam**2)
rng = np.random.default_rng(1)
rp = rng.normal(0, [0.1, 0.1, 1.0], (5000, 3))
dp = rng.normal(0, [0.1, 0.1, 1.0], (50, 3)) + [0, 0, 300]
rp[1] = [0.0, 0.0, 299.0]  # halo particle (index not on the 24-point stride)
h = counter_h(300.0, gam, gam, 600)
r = bunch(rp, (0, 0, b))
d = bunch(dp, (0, 0, -b))
run("full bunch", r, d, steps=1200, h_step=h)
single = {k: (v[1:2] if isinstance(v, np.ndarray) else v) for k, v in r.items()}
run("halo particle alone", single, d, steps=1200, h_step=h)

print("\n== I. APPROXIMATE_BACK_HISTORY rejected on the same criterion ==")
run("ABH slow pair (unit-test case)", slow, src, mode=ABH, steps=3000,
    h_step=1.9e-3)

print("\n== J. Timing ==")
gam = 2.0
b = np.sqrt(1 - 1 / gam**2)
h = counter_h(1000.0, gam, gam, 600)
for n in (100, 10_000, 200_000):
    rng = np.random.default_rng(2)
    r = bunch(rng.normal(0, [0.01, 0.01, 0.1], (n, 3)), (0, 0, b))
    d = bunch(rng.normal(0, [0.01, 0.01, 0.1], (n, 3)) + [0, 0, 1000], (0, 0, -b))
    for geo in (None, pipe):
        t0 = time.perf_counter()
        try:
            check_startup_history(r, d, sim_type=B2B, startup_mode=CS, steps=1200,
                                  h_step=h, beamline_geometry=geo)
        except StartupHistoryError:
            pass
        print(f"    N={n:>7} geometry={'on ' if geo else 'off'} "
              f"{time.perf_counter() - t0:.2f} s")
