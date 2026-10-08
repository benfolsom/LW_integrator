"""Field convergence of fixed exact source reduction (uniform-motion LW fields).

At t=0 under INERTIAL_PREHISTORY every source has been coasting, so its exact
retarded field is the Heaviside uniform-velocity field of its present position.
We compare the full ensemble with reduce_exact_initial_state at several counts,
including the transverse Gaussian children the exact cloud wrapper attaches.
"""

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from core.constants import C_MMNS  # noqa: E402
from core.exact_source_cloud import transverse_offsets  # noqa: E402
from core.exact_source_reduction import reduce_exact_initial_state  # noqa: E402
from core.resolved_knot import initialize_mechanical_knots  # noqa: E402
from core.types import MacroparticleSmearingConfig  # noqa: E402
from tests.unit.test_exact_same_bunch_fields import _merge  # noqa: E402
from tests.unit.test_inertial_prehistory import _state  # noqa: E402


def bunch(n, centre, beta_z, sig_t, sig_l, du, seed, q=1.0, mass=1.0):
    rng = np.random.default_rng(seed)
    g0 = 1 / np.sqrt(1 - beta_z**2)
    u0 = g0 * beta_z
    parts = []
    for _ in range(n):
        pos = centre + rng.normal(0, [sig_t, sig_t, sig_l])
        u = np.array([0, 0, u0]) + rng.normal(0, du, 3) * abs(u0)
        g = np.sqrt(1 + u @ u)
        parts.append(
            _state(
                position_mm=tuple(pos),
                beta=tuple(u / g),
                mass_amu=mass,
                observer_charge=q,
                source_charge=q,
            )
        )
    s = _merge(*parts)
    s["macro_population"][:] = 1000.0
    s["q_source"] = s["macro_population"] * s["q_species"]
    s["q"] = s["q_species"].copy()
    initialize_mechanical_knots(s)
    return s


def cloud_sources(s, width, children):
    pos = np.column_stack([s[a] for a in "xyz"])
    beta = np.column_stack([s["b" + a] for a in "xyz"])
    q = s["q_source"]
    if width == 0 or children == 1:
        return pos, beta, q
    cfg = MacroparticleSmearingConfig(
        enabled=True,
        subcharge_count=children,
        position_sigma_mm=width,
        longitudinal_sigma_mm=0,
        momentum_sigma_amu_mm_ns=0,
        use_momentum_errors=False,
    )
    P, B, Q = [], [], []
    for i in range(len(q)):
        off, frac = transverse_offsets(beta[i], cfg)
        P.append(pos[i] + off)
        B.append(np.repeat(beta[i][None], len(frac), 0))
        Q.append(q[i] * frac)
    return np.vstack(P), np.vstack(B), np.concatenate(Q)


def uniform_field(obs, src, beta, q):
    """Heaviside field of uniformly moving charges at their present positions."""
    R = obs[:, None, :] - src[None, :, :]
    r2 = np.sum(R**2, -1)
    b2 = np.sum(beta**2, -1)[None]
    bxr = np.cross(beta[None], R)
    denom = (r2 - np.sum(bxr**2, -1)) ** 1.5  # R^2(1-b^2 sin^2)
    E = np.sum((q * (1 - b2))[..., None] * R / denom[..., None], axis=1)
    B = np.sum(np.cross(beta[None], (q * (1 - b2))[..., None] * R / denom[..., None]), 1)
    return E, B


def force_on(rider, driver, width, children):
    src, beta, q = cloud_sources(driver, width, children)
    obs = np.column_stack([rider[a] for a in "xyz"])
    E, B = uniform_field(obs, src, beta, q)
    bo = np.column_stack([rider["b" + a] for a in "xyz"])
    return rider["q_observer"][:, None] * (E + np.cross(bo, B))


def main():
    n = 48
    # Counter-propagating screened-source style geometry: relativistic driver,
    # slower rider, separated by an impact gap comparable to the bunch size.
    for label, gamma_d, sig_l, gap in [
        ("gamma_d=1.05 sig_l=0.5 gap=2", 1.05, 0.5, 2.0),
        ("gamma_d=3 sig_l=0.5 gap=2", 3.0, 0.5, 2.0),
        ("gamma_d=3 sig_l=2 gap=2", 3.0, 2.0, 2.0),
        ("gamma_d=10 sig_l=2 gap=5", 10.0, 2.0, 5.0),
    ]:
        bd = np.sqrt(1 - 1 / gamma_d**2)
        driver = bunch(n, np.array([0, 0, 0.0]), -bd, 0.2, sig_l, 1e-3, 1)
        rider = bunch(n, np.array([gap, 0, 0.0]), 0.3, 0.2, 0.5, 1e-3, 2, mass=0.0005)
        print(f"\n== {label}")
        for width, ch in [(0.0, 1), (0.1, 16)]:
            F_full = force_on(rider, driver, width, ch)
            ref = np.sqrt(np.mean(np.sum(F_full**2, 1)))
            row = []
            for count in (4, 8, 16, 24, 32, 47):
                red, mp = reduce_exact_initial_state(driver, count)
                F = force_on(rider, red, width, ch)
                err = np.sqrt(np.mean(np.sum((F - F_full) ** 2, 1))) / ref
                row.append(f"{count}:{err:.2e}")
            print(f" driver-only reduction, width={width} ch={ch}: rel rms force err", *row)
        # Both roles reduced (as check_sources2.py does): compare population-
        # weighted totals and the script's weighted rms kick metric.
        F_full = force_on(rider, driver, 0.1, 16)
        w_full = rider["macro_population"]
        tot_full = np.sum(w_full[:, None] * F_full, 0)
        rms_full = np.sqrt(np.sum(w_full * np.sum(F_full**2, 1)) / w_full.sum())
        for count in (8, 16, 24, 47):
            rr, rmap = reduce_exact_initial_state(rider, count)
            dr, _ = reduce_exact_initial_state(driver, count)
            F = force_on(rr, dr, 0.1, 16)
            w = rr["macro_population"]
            tot = np.sum(w[:, None] * F, 0)
            rms = np.sqrt(np.sum(w * np.sum(F**2, 1)) / w.sum())
            # Projection of the full-run force onto the same rider cells.
            cells = np.asarray(rmap["parent_cells"])
            proj = np.vstack(
                [
                    np.sum(w_full[cells == k, None] * F_full[cells == k], 0)
                    / w_full[cells == k].sum()
                    for k in range(count)
                ]
            )
            perr = np.sqrt(np.sum(w * np.sum((F - proj) ** 2, 1)) / w.sum()) / rms_full
            print(
                f" both reduced {count:2d}: total rel err "
                f"{np.linalg.norm(tot - tot_full) / np.linalg.norm(tot_full):.2e}, "
                f"rms-kick rel change {abs(rms - rms_full) / rms_full:.2e}, "
                f"cell-projected rel err {perr:.2e}"
            )

    # Same-bunch (space-charge) field on the reduced macros vs full.
    print("\n== same-bunch field, gamma=3, sig_l=0.5")
    bd = np.sqrt(1 - 1 / 9)
    d = bunch(n, np.zeros(3), bd, 0.2, 0.5, 1e-3, 3)

    def self_force(s, width, ch):
        out = []
        for i in range(len(s["x"])):
            others = {k: (np.delete(v, i) if isinstance(v, np.ndarray) and v.shape == s["x"].shape else v) for k, v in s.items()}
            one = {k: (v[i : i + 1] if isinstance(v, np.ndarray) and v.shape == s["x"].shape else v) for k, v in s.items()}
            out.append(force_on(one, others, width, ch)[0])
        return np.array(out)

    F_full = self_force(d, 0.1, 16)
    w = d["macro_population"]
    rms_full = np.sqrt(np.sum(w * np.sum(F_full**2, 1)) / w.sum())
    for count in (8, 16, 24, 47):
        r, m = reduce_exact_initial_state(d, count)
        F = self_force(r, 0.1, 16)
        cells = np.asarray(m["parent_cells"])
        proj = np.vstack([F_full[cells == k].mean(0) for k in range(count)])
        wr = r["macro_population"]
        perr = np.sqrt(np.sum(wr * np.sum((F - proj) ** 2, 1)) / wr.sum()) / rms_full
        rms = np.sqrt(np.sum(wr * np.sum(F**2, 1)) / wr.sum())
        print(f" {count:2d}: rms change {abs(rms-rms_full)/rms_full:.2e}, cell-projected err {perr:.2e}")

    # Moment identities and what is not conserved.
    print("\n== moment accounting, gamma=3 driver with 1e-2 relative proper-velocity spread")
    d = bunch(n, np.zeros(3), -bd, 0.2, 0.5, 1e-2, 4)
    N = d["macro_population"]
    u = np.column_stack([d["gamma"] * d["b" + a] for a in "xyz"])
    beta = np.column_stack([d["b" + a] for a in "xyz"])
    pos = np.column_stack([d[a] for a in "xyz"])
    for count in (8, 24):
        r, _ = reduce_exact_initial_state(d, count)
        Nr = r["macro_population"]
        ur = np.column_stack([r["gamma"] * r["b" + a] for a in "xyz"])
        br = np.column_stack([r["b" + a] for a in "xyz"])
        pr = np.column_stack([r[a] for a in "xyz"])
        print(
            f" {count}: dQ={Nr.sum()-N.sum():.1e}"
            f" ddipole={np.abs(Nr@pr-N@pos).max():.1e}"
            f" dP={np.abs(Nr@ur-N@u).max():.1e}"
            f" dcurrent_rel={np.linalg.norm(Nr@br-N@beta)/np.linalg.norm(N@beta):.2e}"
            f" dKE_rel={(Nr@(r['gamma']-1)-N@(d['gamma']-1))/(N@(d['gamma']-1)):.2e}"
            f" quad_rel={np.abs((Nr[:,None]*pr).T@pr-(N[:,None]*pos).T@pos).max()/np.abs((N[:,None]*pos).T@pos).max():.2e}"
        )


if __name__ == "__main__":
    main()
