"""Sequential prescribed-history temporal cadence controls; no coupled claims."""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.pic import CloudCorrection, CorrectionConfig, ElectrostaticPIC
from core.pic.correction import TemporalNodeFields
from core.pic.kernels import C
from scripts.study_pic_correction_k import CASES, quiet, species, trajectory, observers

OUT = Path.home() / "compute-data/pic_correction_k/3_cadence"
TITLES = {
    "acceleration": "Acceleration",
    "crossing_electron_source": "Crossing: electron source",
    "crossing_proton_source": "Crossing: proton source",
}


def initial(case, count, start):
    offsets = quiet(count)
    src = species(offsets.copy(), case)
    corr = CloudCorrection(
        [src],
        CorrectionConfig(
            clouds_per_species=64,
            subcharge_count=4,
            cloud_width_rule="bunch_rms_k",
            lattice_shape=(4,) * 3,
            half_extent_m=(0.15e-3,) * 3,
        ),
    )
    # Every accepted knot uses the actual particle refit, including prehistory.
    for t in np.linspace(0, start, 2401)[1:]:
        _, displacement, beta = trajectory(case, t / case["duration"])
        src.position_m[:] = offsets + displacement
        src.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
        corr.accept([src], t)
    return src, offsets, corr


def sequence(name, case, divisions):
    width = 12e-12 if name == "acceleration" else 24e-12
    times = np.linspace(
        case["duration"] - width / 2, case["duration"] + width / 2, divisions + 1
    )
    src, offsets, corr = initial(case, 2**14, times[0])
    obs, grid, nodes, center = observers(case, 2**14)
    obs_offsets = obs.position_m - center
    local = obs_offsets.copy()
    fields, queries, refits, kicks = [], [], [], []
    solver = ElectrostaticPIC()
    dt = width / divisions
    for j, t in enumerate(times):
        begin = time.perf_counter()
        if j:
            _, displacement, beta = trajectory(case, t / case["duration"])
            src.position_m[:] = offsets + displacement
            src.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
            corr.accept([src], t)
        refits.append(time.perf_counter() - begin)
        shift = np.array([0, 0, case["observer_beta"] * C * (t - case["duration"])])
        begin = time.perf_counter()
        e, b = corr.direct(nodes + shift)
        queries.append(time.perf_counter() - begin)
        fields.append(np.stack((e, C * b)))
        begin = time.perf_counter()
        eg, bg = (grid.gather(local, f.reshape(*grid.shape, 3)) for f in (e, b))
        kicks.append(solver.push(obs, eg, bg, dt) - obs.momentum_mc)
        # Include unchanged gather/kick cost in the baseline step cost.
        refits[-1] += time.perf_counter() - begin
    assert corr.refits == 2401 + divisions
    fields = np.array(fields)
    np.savez_compressed(
        OUT / f"{name}_{divisions}.npz",
        times=times,
        fields=fields,
        query_s=queries,
        unchanged_s=refits,
        kicks=kicks,
    )
    rows = []
    # Exclude the cold first query from speedup; retain warm startup in errors.
    for n in (1, 2, 4, 8):
        for mode in ("linear", "hold", "extrapolate"):
            indices = np.arange(0, len(times), n)
            candidate = []
            cache = TemporalNodeFields()
            begin = time.perf_counter()
            for j, t in enumerate(times):
                if mode == "linear":
                    lo = (j // n) * n
                    hi = min(lo + n, divisions)
                    f = (
                        fields[lo]
                        if lo == hi
                        else fields[lo]
                        + (t - times[lo])
                        / (times[hi] - times[lo])
                        * (fields[hi] - fields[lo])
                    )
                else:
                    f = np.array(cache.sample(j, t, n, mode, lambda: tuple(fields[j])))
                candidate.append(f)
            overhead = time.perf_counter() - begin
            candidate = np.array(candidate)
            delta = candidate - fields
            kick = []
            for f in candidate:
                eg, bg = (
                    grid.gather(local, a.reshape(*grid.shape, 3))
                    for a in (f[0], f[1] / C)
                )
                kick.append(solver.push(obs, eg, bg, dt) - obs.momentum_mc)
            kick = np.array(kick)
            exact = np.array(kicks)
            # Population is one per observer, so sums are population-weighted.
            impulse_error = np.linalg.norm(
                (kick - exact).sum(axis=(0, 1))
            ) / np.linalg.norm(exact.sum(axis=(0, 1)))
            field_cost = sum(queries[i] for i in indices if i > 0)
            same_cost = sum(refits[1:])
            row = dict(
                case=name,
                divisions=divisions,
                dt_s=dt,
                n=n,
                mode=mode,
                e_error=float(
                    np.linalg.norm(delta[:, 0]) / np.linalg.norm(fields[:, 0])
                ),
                cb_error=float(
                    np.linalg.norm(delta[:, 1]) / np.linalg.norm(fields[:, 1])
                ),
                peak_error=float(
                    np.max(np.linalg.norm(delta.reshape(len(times), -1), axis=1))
                    / np.max(np.linalg.norm(fields.reshape(len(times), -1), axis=1))
                ),
                kick_error=float(np.linalg.norm(kick - exact) / np.linalg.norm(exact)),
                impulse_error=float(impulse_error),
                speedup=(sum(queries[1:]) + same_cost)
                / (field_cost + same_cost + overhead),
                full_warm_s=(sum(queries[1:]) + same_cost) / divisions,
                candidate_warm_s=(field_cost + same_cost + overhead) / divisions,
            )
            rows.append(row)
    print(name, divisions, "finished", flush=True)
    return rows


def main():
    assert int(Path("/private/tmp/compute_cap").read_text()) >= 2
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    convergence = []
    for name, case in CASES.items():
        for divisions in (32, 64, 128):
            rows.extend(sequence(name, case, divisions))
            (OUT / "results.json").write_text(json.dumps(rows, indent=2))
        fine = np.load(OUT / f"{name}_128.npz")["fields"]
        for divisions in (32, 64):
            coarse = np.load(OUT / f"{name}_{divisions}.npz")["fields"]
            matched = fine[:: 128 // divisions]
            convergence.append(
                dict(
                    case=name,
                    divisions=divisions,
                    e_error=float(
                        np.linalg.norm(coarse[:, 0] - matched[:, 0])
                        / np.linalg.norm(matched[:, 0])
                    ),
                    cb_error=float(
                        np.linalg.norm(coarse[:, 1] - matched[:, 1])
                        / np.linalg.norm(matched[:, 1])
                    ),
                )
            )
    (OUT / "timestep.json").write_text(json.dumps(convergence, indent=2))
    plot(rows)


def plot(rows, timings=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for column, name in enumerate(CASES):
        for mode in ("linear", "hold", "extrapolate"):
            subset = [
                r
                for r in rows
                if r["case"] == name and r["mode"] == mode and r["n"] == 8
            ]
            axes[0, column].loglog(
                [r["dt_s"] * 1e12 for r in subset],
                [r["e_error"] for r in subset],
                "o-",
                label=mode,
            )
            if timings is None:
                subset = [
                    r
                    for r in rows
                    if r["case"] == name and r["mode"] == mode and r["divisions"] == 128
                ]
                axes[1, column].plot(
                    [r["n"] for r in subset], [r["speedup"] for r in subset], "o-"
                )
        if timings is not None:
            subset = [r for r in timings if r["case"] == name]
            axes[1, column].plot(
                [r["n"] for r in subset],
                [subset[0]["step_s"] / r["step_s"] for r in subset],
                "o-",
            )
        axes[0, column].set(
            title=TITLES[name],
            xlabel="Timestep (ps)",
            ylabel="Relative E cadence error (n = 8)",
        )
        axes[1, column].set(
            xlabel="Evaluation cadence n",
            ylabel=(
                "Estimated correction-step speedup"
                if timings is None
                else "Measured correction-step speedup"
            ),
        )
        axes[0, column].legend()
    fig.tight_layout()
    fig.savefig(OUT / "cadence.png", dpi=160)


if __name__ == "__main__":
    main()
