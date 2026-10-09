"""Macro-count convergence of the reduced exact sources after 1f4a456.

Same set-up as the earlier review's reduction_fields.py (uniform-motion LW
fields at t=0 under INERTIAL_PREHISTORY), but each reduced macro now uses the
per-cell width the branch realises:
  branch : sigma = max(sigma_orig, sigma_cell)            (code as shipped)
  quad   : sigma = sqrt(sigma_orig^2 + sigma_cell^2)      (second-moment-preserving)
  fixed  : sigma = sigma_orig                             (previous behaviour)
Errors are population-weighted RMS force differences on the coarse run's own
parent_cells (the metric adopted in scripts/summarize_sources2.py), relative
to the full-ensemble RMS force.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.exact_source_cloud import transverse_offsets  # noqa: E402
from core.exact_source_reduction import reduce_exact_initial_state  # noqa: E402
from core.types import MacroparticleSmearingConfig  # noqa: E402
from reduction_fields import bunch, uniform_field  # noqa: E402

W, CH = 0.1, 16
CFG = MacroparticleSmearingConfig(
    enabled=True,
    subcharge_count=CH,
    position_sigma_mm=W,
    longitudinal_sigma_mm=0,
    momentum_sigma_amu_mm_ns=0,
    use_momentum_errors=False,
)


def widths(s, mode):
    n = len(s["x"])
    cell = np.asarray(s.get("exact_source_cell_sigma_mm", np.zeros(n)))
    if mode == "fixed":
        return np.full(n, W)
    if mode == "branch":
        return np.maximum(cell, W)
    return np.sqrt(cell**2 + W**2)


def sources(s, mode, skip=None):
    pos = np.column_stack([s[a] for a in "xyz"])
    beta = np.column_stack([s["b" + a] for a in "xyz"])
    q = s["q_source"]
    ws = widths(s, mode)
    P, B, Q = [], [], []
    for i in range(len(q)):
        if skip is not None and i == skip:
            continue
        off, frac = transverse_offsets(beta[i], CFG, width_mm=ws[i])
        P.append(pos[i] + off)
        B.append(np.repeat(beta[i][None], len(frac), 0))
        Q.append(q[i] * frac)
    return np.vstack(P), np.vstack(B), np.concatenate(Q)


def lorentz(obs_state, E, B):
    bo = np.column_stack([obs_state["b" + a] for a in "xyz"])
    return obs_state["q_observer"][:, None] * (E + np.cross(bo, B))


def cross_force(rider, driver, mode):
    obs = np.column_stack([rider[a] for a in "xyz"])
    return lorentz(rider, *uniform_field(obs, *sources(driver, mode)))


def self_force(s, mode):
    out = []
    for i in range(len(s["x"])):
        obs = np.array([[s[a][i] for a in "xyz"]])
        one = {k: (v[i : i + 1] if isinstance(v, np.ndarray) and v.shape == s["x"].shape else v) for k, v in s.items()}
        out.append(lorentz(one, *uniform_field(obs, *sources(s, mode, skip=i)))[0])
    return np.array(out)


def projected_error(F, F_full, w_full, w_red, cells, count, rms_full):
    proj = np.vstack(
        [np.sum(w_full[cells == k, None] * F_full[cells == k], 0) / w_full[cells == k].sum() for k in range(count)]
    )
    return np.sqrt(np.sum(w_red * np.sum((F - proj) ** 2, 1)) / w_red.sum()) / rms_full


def main():
    n = 48
    counts = (6, 12, 24, 36, 47)
    print("relative cell-projected rms force error vs macro count", counts)
    for label, gamma_d, sig_t, sig_l, gap in [
        ("cross g=3 st=0.2 sl=0.5 gap=2", 3.0, 0.2, 0.5, 2.0),
        ("cross g=10 st=0.2 sl=2 gap=5", 10.0, 0.2, 2.0, 5.0),
        ("cross g=3 st=0.5 sl=0.5 gap=1", 3.0, 0.5, 0.5, 1.0),
    ]:
        bd = np.sqrt(1 - 1 / gamma_d**2)
        tot = {m: [] for m in ("fixed", "branch", "quad")}
        errs = {m: [] for m in tot}
        for seed in range(3):
            driver = bunch(n, np.zeros(3), -bd, sig_t, sig_l, 1e-3, 10 + seed)
            rider = bunch(n, np.array([gap, 0, 0.0]), 0.3, 0.2, 0.5, 1e-3, 20 + seed, mass=0.0005)
            F_full = cross_force(rider, driver, "fixed")
            w = rider["macro_population"]
            rms = np.sqrt(np.sum(w * np.sum(F_full**2, 1)) / w.sum())
            for mode in tot:
                row = []
                for c in counts:
                    dr, _ = reduce_exact_initial_state(driver, c)
                    rr, rm = reduce_exact_initial_state(rider, c)
                    F = cross_force(rr, dr, mode)
                    row.append(projected_error(F, F_full, w, rr["macro_population"], np.asarray(rm["parent_cells"]), c, rms))
                errs[mode].append(row)
        print(f"\n== {label} (mean of 3 seeds, both roles reduced)")
        for mode, rows in errs.items():
            print(f"  {mode:6s}", " ".join(f"{c}:{e:.2e}" for c, e in zip(counts, np.mean(rows, 0))))

    for label, gamma, sig_t, sig_l in [
        ("same-bunch g=3 st=0.2 sl=0.5", 3.0, 0.2, 0.5),
        ("same-bunch g=1.05 st=0.3 sl=0.3", 1.05, 0.3, 0.3),
    ]:
        bd = np.sqrt(1 - 1 / gamma**2)
        errs = {m: [] for m in ("fixed", "branch", "quad")}
        vir = {m: [] for m in errs}

        def virial(s, F):
            x = np.column_stack([s["x"], s["y"]])
            x = x - np.average(x, axis=0, weights=s["macro_population"])
            return np.sum(s["macro_population"] * np.sum(x * F[:, :2], 1))

        for seed in range(3):
            d = bunch(n, np.zeros(3), bd, sig_t, sig_l, 1e-3, 30 + seed)
            F_full = self_force(d, "fixed")
            w = d["macro_population"]
            rms = np.sqrt(np.sum(w * np.sum(F_full**2, 1)) / w.sum())
            v_full = virial(d, F_full)
            for mode in errs:
                row, vrow = [], []
                for c in counts:
                    r, m = reduce_exact_initial_state(d, c)
                    F = self_force(r, mode)
                    row.append(projected_error(F, F_full, w, r["macro_population"], np.asarray(m["parent_cells"]), c, rms))
                    vrow.append(abs(virial(r, F) - v_full) / abs(v_full))
                errs[mode].append(row)
                vir[mode].append(vrow)
        print(f"\n== {label} (mean of 3 seeds)")
        for mode, rows in errs.items():
            print(f"  {mode:6s}", " ".join(f"{c}:{e:.2e}" for c, e in zip(counts, np.mean(rows, 0))))
        print("  transverse virial sum N x.F relative error:")
        for mode, rows in vir.items():
            print(f"  {mode:6s}", " ".join(f"{c}:{e:.2e}" for c, e in zip(counts, np.mean(rows, 0))))


if __name__ == "__main__":
    main()
