"""Independent all-pair crossing sampling/source-step controls, without a scan."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.pic import ElectrostaticPIC, NearFieldConfig, NearFieldCorrection
from core.pic.nearfield import _lw_cloud, _prepared
from study_pic_correction_k_nearfield import OUT, make, kicks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, choices=(8, 32), default=8)
    parser.add_argument("--steps", type=int, default=2401)
    parser.add_argument("--samples", type=int, default=129)
    args = parser.parse_args()
    species, encounter = make(args.count)
    dt = encounter / (2 * (args.steps - 1) // 3)
    baseline = np.load(OUT / f"n{args.count}_s1201_q129.npz")
    baseline_meta = json.loads((OUT / f"n{args.count}_s1201_q129.json").read_text())
    baseline_dt = baseline_meta["dt_s"]
    baseline_times = baseline["time_s"]
    solver = ElectrostaticPIC((32,) * 3, (4e-3,) * 3, 2)
    near = NearFieldCorrection(species, NearFieldConfig(0.8e-3))
    selected = np.unique(
        np.rint(
            np.linspace((args.steps - 1) / 2, 5 * (args.steps - 1) / 6, args.samples)
        ).astype(int)
    )
    if args.samples == len(baseline_times):
        selected = np.rint(baseline_times / dt).astype(int)
    times, reference, states, ledgers = [], [], [], []
    start = time.perf_counter()
    for step in range(args.steps):
        if step in selected:
            print(args.count, step, "elapsed", time.perf_counter() - start, flush=True)
            times.append(step * dt)
            states.append([s.momentum_mc.copy() for s in species])
            row, row_ledger = [], []
            for oi, obs in enumerate(species):
                si = 1 - oi
                h = near.histories[si]
                prepared = _prepared(h)
                le, lb = [], []
                for i in range(args.count):
                    e, b = _lw_cloud(
                        prepared, h, near.source_cloud[si][i], obs.position_m
                    )
                    le.append(e)
                    lb.append(b)
                momentum, ledger = kicks(
                    solver,
                    obs,
                    np.stack(le, axis=1).sum(axis=1),
                    np.stack(lb, axis=1).sum(axis=1),
                    dt,
                )
                row.append(momentum)
                row_ledger.append(ledger)
            reference.append(row)
            ledgers.append(row_ledger)
        if step == args.steps - 1:
            break
        for s in species:
            s.position_m += 0.5 * dt * s.velocity_m_s
        fields = solver.fields(species)
        updated = [
            solver.push(s, *solver.sample(fields, s.position_m), dt) for s in species
        ]
        for s, u in zip(species, updated):
            s.momentum_mc = u
            s.position_m += 0.5 * dt * s.velocity_m_s
        near.accept(species, (step + 1) * dt)
    reference, times = np.asarray(reference), np.asarray(times)
    indices = np.array([np.argmin(np.abs(times - t)) for t in baseline_times])
    assert np.allclose(times[indices], baseline_times, rtol=0, atol=1e-24)
    rows = []
    for oi, name in enumerate(("protons", "electrons")):
        old = baseline[f"m48_r8.0_{name}_kicks"][:, 2]
        actual = reference[indices, oi]
        matching_error = np.linalg.norm(
            actual / dt - old / baseline_dt
        ) / np.linalg.norm(old / baseline_dt)
        old_imp = np.trapezoid(old / baseline_dt, baseline_times, axis=0)
        common_imp = np.trapezoid(actual / dt, times[indices], axis=0)
        fine_imp = np.trapezoid(reference[:, oi] / dt, times, axis=0)
        controls = []
        for stride in (4, 2):
            subset = np.unique(np.r_[np.arange(0, len(times), stride), len(times) - 1])
            imp = np.trapezoid(reference[subset, oi] / dt, times[subset], axis=0)
            controls.append(
                dict(
                    samples=len(subset),
                    relative_impulse_change=float(
                        np.linalg.norm(imp - fine_imp) / np.linalg.norm(fine_imp)
                    ),
                )
            )
        rows.append(
            dict(
                observer=name,
                matching_force_change=float(matching_error),
                common_sampling_impulse_change=float(
                    np.linalg.norm(common_imp - old_imp) / np.linalg.norm(old_imp)
                ),
                denser_sampling_impulse_change=float(
                    np.linalg.norm(fine_imp - common_imp) / np.linalg.norm(fine_imp)
                ),
                controls=controls,
            )
        )
    base = OUT / f"control_n{args.count}_s{args.steps}_q{args.samples}"
    np.savez_compressed(
        base.with_suffix(".npz"),
        time_s=times,
        reference_kicks=reference,
        momentum_mc=np.asarray(states),
        ledgers_json=np.array(json.dumps(ledgers)),
    )
    summary = dict(
        count_per_bunch=args.count,
        steps=args.steps,
        samples=len(times),
        dt_s=dt,
        runtime_s=time.perf_counter() - start,
        rows=rows,
        finite=bool(np.all(np.isfinite(reference))),
    )
    base.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
