"""Stage-2 crossing geometry: frozen-history, all-pair exact-cloud controls.

Run serially with the requested external interpreter/cache environment. Pair
matrices reuse costly exact roots in the cutoff/grid scan; selected snapshots
independently replay the public correction, and time its complete field query.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.pic import (
    CloudCorrection,
    CorrectionConfig,
    ElectrostaticPIC,
    NearFieldConfig,
    NearFieldCorrection,
    Species,
)
from core.pic.kernels import C
from core.pic.nearfield import grid_pair_fields, _lw_cloud, _prepared, _qs_cloud

OUT = Path.home() / "compute-data/pic_correction_k/5_nearfield"
Q, MP, ME = 1.602176634e-19, 1.67262192369e-27, 9.1093837139e-31


def make(count):
    offsets = np.array(list(itertools.product((-1, 1), repeat=3))) * [0.65, 0.65, 0.08]
    if count == 32:
        shifts = np.array(list(itertools.product((-0.05, 0.05), repeat=2)))
        shifts = np.column_stack((shifts, np.zeros(4)))
        offsets = (offsets[:, None, :] + shifts).reshape(32, 3)
    bp = np.sqrt(1 - (1 + 216 / 938.27208816) ** -2)
    be = np.sqrt(1 - 10**-2)
    species = []
    for name, charge, mass, total, beta, z in (
        ("protons", Q, MP, 100e-12, bp, -5),
        ("electrons", -Q, ME, 20e-12, -be, 5),
    ):
        species.append(
            Species(
                name,
                (offsets + [0, 0, z]) * 1e-3,
                np.tile([0, 0, beta / np.sqrt(1 - beta * beta)], (count, 1)),
                charge,
                mass,
                np.full(count, total / (count * Q)),
            )
        )
    return species, 10e-3 / ((bp + be) * C)


def kinetic_delta(s, u):
    before = s.momentum_mc
    dg = np.sum((u - before) * (u + before), axis=-1) / (
        np.sqrt(1 + np.sum(u * u, axis=-1))
        + np.sqrt(1 + np.sum(before * before, axis=-1))
    )
    return s.population * s.mass_kg * C * C * dg


def kicks(solver, obs, e, b, dt):
    u = solver.push(obs, e, b, dt)
    p = obs.mass_kg * C * (u - obs.momentum_mc)
    g0 = np.sqrt(1 + np.sum(obs.momentum_mc**2, axis=1))
    g1 = np.sqrt(1 + np.sum(u * u, axis=1))
    v = C * (obs.momentum_mc + u) / (g0 + g1)[:, None]
    return p, dict(
        kinetic_j=float(kinetic_delta(obs, u).sum()),
        work_j=float(dt * np.sum(obs.population[:, None] * obs.charge_c * v * e)),
        momentum=(np.sum(obs.population[:, None] * p, axis=0)).tolist(),
        impulse=(
            dt * np.sum(obs.source_charge_c[:, None] * (e + np.cross(v, b)), axis=0)
        ).tolist(),
    )


def study(
    count,
    steps,
    samples,
    *,
    cloud_config=None,
    near_subcharges=16,
    replay_meshes=(24, 48, 96),
    replay_cutoffs=(0.8, 3.2, 8.0),
):
    species, encounter = make(count)
    dt = encounter / (2 * (steps - 1) // 3)
    baseline_solver = ElectrostaticPIC((32,) * 3, (4e-3,) * 3, 2)
    cfg = cloud_config or CorrectionConfig(
        clouds_per_species=8, cloud_width_m=0.16e-3, subcharge_count=16
    )
    k_count = cfg.clouds_per_species
    coarse = CloudCorrection(species, cfg)
    near = NearFieldCorrection(
        species, NearFieldConfig(0.8e-3, subcharge_count=near_subcharges), coarse
    )
    selected = np.unique(
        np.rint(np.linspace((steps - 1) / 2, 5 * (steps - 1) / 6, samples)).astype(int)
    )
    cutoffs = (0.0, 0.4, 0.8, 1.6, 3.2, 8.0)
    meshes = (24, 48, 96)
    results = {(m, r, oi): [] for m in meshes for r in cutoffs for oi in (0, 1)}
    ledgers = {(m, r): [] for m in meshes for r in cutoffs}
    timings, closure, histories = (
        [],
        [],
        {"time_s": [], "position_m": [], "momentum_mc": []},
    )
    checkpoints = set(
        selected[np.unique(np.rint(np.linspace(0, len(selected) - 1, 3)).astype(int))]
    )
    exact_checks = []
    started = time.perf_counter()
    for step in range(steps):
        if step in selected:
            elapsed = time.perf_counter() - started
            print(
                f"N={count} step={step}/{steps - 1} elapsed={elapsed:.1f}s", flush=True
            )
            histories["time_s"].append(step * dt)
            histories["position_m"].append(np.array([s.position_m for s in species]))
            histories["momentum_mc"].append(np.array([s.momentum_mc for s in species]))
            pairs, clouds = [], []
            for oi, obs in enumerate(species):
                si = 1 - oi
                h = near.histories[si]
                prepared = _prepared(h)
                le, lb = np.empty((count, count, 3)), np.empty((count, count, 3))
                for i in range(count):
                    le[:, i], lb[:, i] = _lw_cloud(
                        prepared, h, near.source_cloud[si][i], obs.position_m
                    )
                pairs.append((le, lb))
                cp = _prepared(coarse)
                de, db = np.zeros((count, k_count, 3)), np.zeros((count, k_count, 3))
                local = [
                    (j, idx)
                    for j, (source_si, idx) in enumerate(coarse.members)
                    if source_si == si
                ]
                for k, (j, idx) in enumerate(local):
                    ee, bb = _lw_cloud(cp, coarse, j, obs.position_m)
                    qe, qb = _qs_cloud(coarse, j, obs.position_m)
                    de[:, k], db[:, k] = ee - qe, bb - qb
                clouds.append((de, db, local))
            for mesh in meshes:
                solver = ElectrostaticPIC((mesh,) * 3, (4e-3,) * 3, 2)
                tick = time.perf_counter()
                fields = solver.fields(species)
                build_s = time.perf_counter() - tick
                bases, grid_pairs = [], []
                for oi, obs in enumerate(species):
                    si = 1 - oi
                    f = next(f for f in fields if f.species_index == si)
                    bases.append(f.sample(obs.position_m))
                    oi_index = np.repeat(np.arange(count), count)
                    ii_index = np.tile(np.arange(count), count)
                    ge, gb = grid_pair_fields(
                        f, species[si], obs.position_m[oi_index], ii_index
                    )
                    ge, gb = ge.reshape(count, count, 3), gb.reshape(count, count, 3)
                    grid_pairs.append((ge, gb))
                    closure.append(
                        float(
                            np.linalg.norm(ge.sum(axis=1) - bases[-1][0])
                            / np.linalg.norm(bases[-1][0])
                        )
                    )
                for radius in cutoffs:
                    total_ledger = dict(reference={}, corrected={})
                    for oi, obs in enumerate(species):
                        si = 1 - oi
                        le, lb = pairs[oi]
                        ge, gb = grid_pairs[oi]
                        e, b = bases[oi]
                        de, db, local = clouds[oi]
                        distance = np.linalg.norm(
                            obs.position_m[:, None, :] - species[si].position_m, axis=2
                        )
                        w = (
                            NearFieldConfig(radius * 1e-3).weight(distance)
                            if radius
                            else np.zeros_like(distance)
                        )
                        coeff = np.ones((count, k_count))
                        for k, (j, idx) in enumerate(local):
                            coeff[:, k] -= (
                                np.sum(w[:, idx] * species[si].population[idx], axis=1)
                                / coarse.weights[j]
                            )
                        ce = np.sum(coeff[..., None] * de, axis=1) + np.sum(
                            w[..., None] * (le - ge), axis=1
                        )
                        cb = np.sum(coeff[..., None] * db, axis=1) + np.sum(
                            w[..., None] * (lb - gb), axis=1
                        )
                        reference, ref_ledger = kicks(
                            solver, obs, le.sum(axis=1), lb.sum(axis=1), dt
                        )
                        corrected, fix_ledger = kicks(solver, obs, e + ce, b + cb, dt)
                        plain, _ = kicks(solver, obs, e, b, dt)
                        results[mesh, radius, oi].append((plain, corrected, reference))
                        total_ledger["reference"][obs.name] = ref_ledger
                        total_ledger["corrected"][obs.name] = fix_ledger
                        if (
                            step in checkpoints
                            and mesh in replay_meshes
                            and radius in replay_cutoffs
                        ):
                            near.config = replace(near.config, cutoff_m=radius * 1e-3)
                            tick = time.perf_counter()
                            actual = near.sample(fields, obs, (si,))
                            query_s = time.perf_counter() - tick
                            errors = [
                                np.linalg.norm(a - z) / max(np.linalg.norm(z), 1e-100)
                                for a, z in zip(actual, (ce, cb))
                            ]
                            exact_checks.append(max(errors))
                            timings.append(
                                dict(
                                    step=step,
                                    mesh=mesh,
                                    cutoff_mm=radius,
                                    observer=obs.name,
                                    field_query_s=query_s,
                                    mesh_build_s=build_s,
                                    **near.counts,
                                )
                            )
                    ledgers[mesh, radius].append(total_ledger)
        if step == steps - 1:
            break
        for s in species:
            s.position_m += 0.5 * dt * s.velocity_m_s
        fields = baseline_solver.fields(species)
        updated = [
            baseline_solver.push(s, *baseline_solver.sample(fields, s.position_m), dt)
            for s in species
        ]
        for s, u in zip(species, updated):
            s.momentum_mc = u
            s.position_m += 0.5 * dt * s.velocity_m_s
        coarse.accept(species, (step + 1) * dt)
        near.accept(species, (step + 1) * dt)
    times = np.array(histories["time_s"])
    arrays = {name: np.array(values) for name, values in histories.items()}
    rows = []
    for (mesh, radius, oi), values in results.items():
        plain, corrected, reference = np.moveaxis(np.asarray(values), 1, 0)
        denominator = np.linalg.norm(reference)
        ip, ic, ir = [
            np.trapezoid(a / dt, times, axis=0) for a in (plain, corrected, reference)
        ]
        row = dict(
            mesh=mesh,
            cutoff_mm=radius,
            observer=species[oi].name,
            pic_kick_error=float(np.linalg.norm(plain - reference) / denominator),
            kick_error=float(np.linalg.norm(corrected - reference) / denominator),
            impulse_error=float(np.linalg.norm(ic - ir) / np.linalg.norm(ir)),
            pic_impulse_error=float(np.linalg.norm(ip - ir) / np.linalg.norm(ir)),
        )
        row["sampling_controls"] = []
        for stride in (4, 2):
            indices = np.unique(np.r_[np.arange(0, len(times), stride), len(times) - 1])
            coarse_imp = np.trapezoid(reference[indices] / dt, times[indices], axis=0)
            row["sampling_controls"].append(
                dict(
                    samples=len(indices),
                    impulse_change=float(
                        np.linalg.norm(coarse_imp - ir) / np.linalg.norm(ir)
                    ),
                )
            )
        rows.append(row)
        arrays[f"m{mesh}_r{radius}_{species[oi].name}_kicks"] = np.array(values)
    for (mesh, radius), samples_ledger in ledgers.items():
        rows_ledger = {}
        for kind in ("reference", "corrected"):
            all_work = (
                np.array(
                    [
                        sum(x[kind][s.name]["work_j"] for s in species)
                        for x in samples_ledger
                    ]
                )
                / dt
            )
            all_energy = (
                np.array(
                    [
                        sum(x[kind][s.name]["kinetic_j"] for s in species)
                        for x in samples_ledger
                    ]
                )
                / dt
            )
            all_momentum = (
                np.array(
                    [
                        np.sum([x[kind][s.name]["momentum"] for s in species], axis=0)
                        for x in samples_ledger
                    ]
                )
                / dt
            )
            all_impulse = (
                np.array(
                    [
                        np.sum([x[kind][s.name]["impulse"] for s in species], axis=0)
                        for x in samples_ledger
                    ]
                )
                / dt
            )
            work, energy = [
                float(np.trapezoid(x, times)) for x in (all_work, all_energy)
            ]
            momentum, impulse = [
                np.trapezoid(x, times, axis=0) for x in (all_momentum, all_impulse)
            ]
            rows_ledger[kind] = dict(
                work_j=work,
                kinetic_j=energy,
                work_minus_kinetic_j=work - energy,
                momentum=momentum.tolist(),
                lorentz_impulse=impulse.tolist(),
                impulse_minus_momentum=(impulse - momentum).tolist(),
            )
        arrays[f"m{mesh}_r{radius}_ledger_json"] = np.array(json.dumps(rows_ledger))
        for row in rows:
            if row["mesh"] == mesh and row["cutoff_mm"] == radius:
                row["pair_ledger"] = rows_ledger
    path = OUT / f"n{count}_s{steps}_q{samples}"
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    summary = dict(
        count_per_bunch=count,
        steps=steps,
        sample_count=len(times),
        dt_s=dt,
        source_mesh=32,
        source_width_mm=near.config.source_width_m * 1e3,
        subcharges=near_subcharges,
        cloud_config=vars(cfg),
        coarse_width_mm=coarse.smearing.position_sigma_mm,
        max_grid_pair_closure=max(closure),
        max_public_replay_error=max(exact_checks),
        rows=rows,
        timing=timings,
        runtime_s=time.perf_counter() - started,
        radiation_reaction_mode="off",
        source="identical accepted uncorrected PIC trajectories",
        reference="all cross-bunch strict exact-path particle-cloud LW pairs",
        minimum_finite_check=bool(
            all(
                np.all(np.isfinite(a)) for a in arrays.values() if a.dtype.kind in "fci"
            )
        ),
    )
    path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(path, "saved", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, choices=(8, 32), default=8)
    parser.add_argument("--steps", type=int, default=1201)
    parser.add_argument("--samples", type=int, default=129)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    study(args.count, args.steps, args.samples)
    files = [
        Path(__file__),
        *Path("core/pic").glob("*.py"),
        Path("tests/test_pic_nearfield.py"),
    ]
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
