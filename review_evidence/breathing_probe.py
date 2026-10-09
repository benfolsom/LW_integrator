"""Breathing exact clouds: closure checks and cost scaling (commit 1f4a456).

Part A exercises core.exact_source_cloud.breathing_width_knots directly on
prescribed (ballistic, force-free) histories, so the "truth" is analytic.
Part B times the integrator with breathing on/off.
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.constants import C_MMNS  # noqa: E402
from core.exact_source_cloud import ExactCloudHistory, breathing_width_knots  # noqa: E402
from core.exact_source_reduction import reduce_exact_initial_state  # noqa: E402
from core.types import MacroparticleSmearingConfig  # noqa: E402

TAU = 0.01


def cfg(width=0.1, tau=TAU):
    return MacroparticleSmearingConfig(
        enabled=True, subcharge_count=4, position_sigma_mm=width,
        longitudinal_sigma_mm=0, momentum_sigma_amu_mm_ns=0,
        use_momentum_errors=False, breathing_enabled=True,
        breathing_response_time_ns=tau,
    )


def ballistic(x0, v, times):
    """positions[k, i] = x0_i + v_i (t_k,i - 0)."""
    return x0[None] + (times[..., None]) * v[None]


def header(n, extra=None):
    state = {
        "inertial_charge_boundary_ready": np.ones(n),
        "inertial_charge_boundary_time_ns": np.zeros(n),
        "macro_population": np.ones(n),
    }
    state.update(extra or {})
    return [state]


def part_a():
    rng = np.random.default_rng(1)
    n, beta_z = 24, 0.6
    x0 = np.column_stack([rng.normal(0, 0.3, n), rng.normal(0, 0.3, n), rng.normal(0, 0.5, n)])
    vt = 30.0  # mm/ns transverse rms velocity (0.1 c)
    v = np.column_stack([rng.normal(0, vt, n), rng.normal(0, vt, n), np.full(n, beta_z * C_MMNS)])
    beta = v / C_MMNS
    times = np.linspace(0, 0.1, 201)
    T = np.repeat(times[:, None], n, 1)
    pos = ballistic(x0, v, T)
    B = np.broadcast_to(beta, pos.shape).copy()
    w, r, a = breathing_width_knots(ExactCloudHistory(header(n), cfg(), None), T, pos, B, np.zeros_like(T, bool))
    rms = lambda p: np.sqrt(np.mean(np.sum((p[..., :2] - p[..., :2].mean(1, keepdims=True)) ** 2, -1), 1) / 2)
    target = 0.1 * rms(pos) / rms(pos)[0]
    dt = times[1] - times[0]
    fd_rate = (w[2:, 0] - w[:-2, 0]) / (2 * dt)
    fd_acc = (r[2:, 0] - r[:-2, 0]) / (2 * dt)
    print("A1 unreduced, synchronous ballistic ensemble, tau=0.01 ns")
    print(f"   rate vs FD(width) max rel {np.max(np.abs(fd_rate - r[1:-1, 0])) / np.max(np.abs(r[:, 0])):.1e};"
          f" accel vs FD(rate) max rel {np.max(np.abs(fd_acc - a[1:-1, 0])) / np.max(np.abs(a[:, 0])):.1e}")
    print(f"   width(t=0.1)/target {w[-1, 0] / target[-1]:.4f}; lag-free expectation"
          f" target(t-2tau)/target(t) = {np.interp(0.1 - 2 * TAU, times, target) / target[-1]:.4f}")
    print(f"   child relative speed |sigma'|*max|node| / c at end = {r[-1, 0] * 2.0 / C_MMNS:.2e}")

    # A2: two coaxial bunches in one driver history (driver train). Bunch A
    # expands, bunch B is cold.  The represented particles of B do not expand.
    vB = v.copy()
    vB[:, :2] = 0
    x0B = x0 + np.array([0, 0, 50.0])
    pos2 = np.concatenate([pos, ballistic(x0B, vB, T)], 1)
    B2 = np.concatenate([B, np.broadcast_to(vB / C_MMNS, pos.shape)], 1)
    T2 = np.concatenate([T, T], 1)
    w2, _, _ = breathing_width_knots(ExactCloudHistory(header(2 * n), cfg(), None), T2, pos2, B2, np.zeros_like(T2, bool))
    print("A2 two-bunch train in one history (A expanding, B cold)")
    print(f"   B macro width t=0 -> 0.1 ns: {w2[0, n]:.3f} -> {w2[-1, n]:.3f} mm (own-bunch truth 0.100);"
          f" A macro {w2[-1, 0]:.3f} mm (own-bunch truth {w[-1, 0]:.3f})")

    # A3: reduced ballistic closure vs the actual members, free-streaming.
    state = {k: v_ for k, v_ in _bunch_state(x0, beta).items()}
    red, mp = reduce_exact_initial_state(state, 4, cfg())
    cells = np.asarray(mp["parent_cells"])
    ages = np.array([0.0, 0.02, 0.05, 0.1])
    print("A3 reduced (4 cells) ballistic closure vs members' transverse RMS about the macro centre")
    for c in range(4):
        m = cells == c
        u = np.array([red["b" + a][c] for a in "xyz"])
        nrm = u / np.linalg.norm(u)
        row = []
        for age in ages:
            centre = np.array([red[a][c] for a in "xyz"]) + u * C_MMNS * age
            d = x0[m] + v[m] * age - centre
            d -= (d @ nrm)[:, None] * nrm
            true = np.sqrt(np.mean(np.sum(d**2, 1)) / 2)
            model = np.sqrt(red["exact_source_cell_sigma_mm"][c] ** 2 + 2 * age * red["exact_source_cell_xv_mm2_ns"][c] + age**2 * red["exact_source_cell_vv_mm2_ns2"][c])
            row.append(f"{model / true:.4f}")
        print(f"   cell {c} (n={m.sum()}): model/true at ages {ages.tolist()}: {row}")

    # A4: cost of the asynchronous predictor (gamma spread -> per-particle lab times).
    print("A4 asynchronous width reconstruction cost (one call; the runner calls it per role per step and per trial)")
    for nn, K in ((8, 100), (8, 200), (16, 100), (16, 200)):
        g = 1.25 + 0.01 * np.arange(nn)
        TT = np.outer(np.arange(K) * 1e-4, g)
        xx = rng.normal(0, 0.3, (nn, 3))
        vv = np.column_stack([rng.normal(0, vt, nn), rng.normal(0, vt, nn), np.full(nn, beta_z * C_MMNS)])
        P = xx[None] + TT[..., None] * vv[None]
        BB = np.broadcast_to(vv / C_MMNS, P.shape).copy()
        t0 = time.perf_counter()
        breathing_width_knots(ExactCloudHistory(header(nn), cfg(), None), TT, P, BB, np.zeros_like(TT, bool))
        print(f"   N={nn:3d} K={K:4d}: {time.perf_counter() - t0:.2f} s")


def _bunch_state(x0, beta):
    n = len(x0)
    g = 1 / np.sqrt(1 - np.sum(beta**2, 1))
    s = {a: x0[:, i].copy() for i, a in enumerate("xyz")}
    s.update({"b" + a: beta[:, i].copy() for i, a in enumerate("xyz")})
    one = np.ones(n)
    s.update(t=np.zeros(n), gamma=g, m=one * 1.0, m_species=one * 1.0, q=one, q_species=one,
             q_observer=one, q_source=one, macro_population=one, char_time=one * 1e-9)
    s.update({"P" + a: C_MMNS * g * beta[:, i] for i, a in enumerate("xyz")})
    s["Pt"] = C_MMNS * g
    return s


def part_b():
    from tests.unit.test_exact_source_cloud import cloud_config
    from tests.unit.test_exact_same_bunch_fields import _merge
    from tests.unit.test_exact_source_reduction import magnetic
    from tests.unit.test_exact_visibility_gates import crossing_run, gate
    from tests.unit.test_inertial_prehistory import _state

    print("B  integrator wall time (2 driver macros, 4 children, unreduced, numba_analytic_charge_response_serial)")
    for breathing in (False, True):
        for intervals in (16, 32, 64):
            options = dict(
                backend="numba_analytic_charge_response_serial", duration=0.15,
                macroparticle_smearing=replace(cloud_config(4, 0.02), breathing_enabled=breathing),
                driver_state=_merge(*[_state(position_mm=(x, 0.5, 0), beta=(x * 0.01, 0.1, 0), observer_charge=5) for x in (-0.05, 0.05)]),
                magnetic_dipole=replace(magnetic(0), exact_retarded_backend="numba_analytic_charge_response_serial"),
            )
            t0 = time.perf_counter()
            crossing_run(intervals, gate(), **options)
            print(f"   breathing={breathing!s:5s} intervals={intervals:3d}: {time.perf_counter() - t0:6.1f} s", flush=True)
    from core.types import DriverTrainConfig

    # Unreduced breathing with a driver train is accepted: the covariance in
    # breathing_width_knots then pools every bunch of the train (see A2).
    r = crossing_run(
        8, gate(), backend="numba_analytic_charge_response_serial", duration=0.15,
        macroparticle_smearing=replace(cloud_config(4, 0.02), breathing_enabled=True),
        driver_state=_merge(*[_state(position_mm=(x, 0.5, 0), beta=(x * 0.01, 0.1, 0), observer_charge=5) for x in (-0.05, 0.05)]),
        magnetic_dipole=replace(magnetic(0), exact_retarded_backend="numba_analytic_charge_response_serial"),
        driver_train=DriverTrainConfig(enabled=True, bunch_count=2, z_offsets_mm=(0, 2)),
    )
    print(f"   unreduced breathing + 2-bunch train accepted: driver macros = {len(r[1][0]['x'])}", flush=True)


if __name__ == "__main__":
    part_a()
    if "--integrator" in sys.argv:
        part_b()
