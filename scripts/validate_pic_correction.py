#!/usr/bin/env python3
"""Bounded accepted-history correction validation; all large artifacts external.

These are kicks on common source trajectories, not a self-consistent exact-LW
trajectory campaign. RR is explicitly off. The crossing uses 8+8 centred
samples and resolves the encounter plus a post-encounter interval.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri

from core.pic import CloudCorrection, CorrectionConfig, ElectrostaticPIC, Species
from core.pic.correction import correction_ledger
from core.pic.kernels import C

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "compute-data/pic_correction"
Q = 1.602176634e-19
MP = 1.67262192369e-27
ME = 9.1093837139e-31


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print(name, "saved", flush=True)


def make(
    count=8,
    charge=Q,
    mass=MP,
    total=1e-12,
    beta=(0, 0, 0.8),
    center=(0, 0, 0),
    rms=0.25e-3,
):
    points = qmc.Sobol(3, scramble=True, seed=20261005).random_base2(
        int(np.log2(count)) - 1
    )
    points = ndtri(np.clip(points, 1e-9, 1 - 1e-9)) * rms
    position = np.concatenate((points, -points)) + center
    beta = np.asarray(beta, dtype=float)
    u = beta / np.sqrt(1 - beta @ beta)
    return Species(
        "source",
        position,
        np.tile(u, (count, 1)),
        charge,
        mass,
        np.full(count, total / (count * abs(charge))),
    )


def accelerated_source(
    count=32, steps=301, width=0.16e-3, k=None, subcharges=16, refit=1, end_fraction=1.0
):
    s = make(count, beta=(0, 0, -0.85))
    corr = CloudCorrection(
        [s],
        CorrectionConfig(
            clouds_per_species=k or count,
            cloud_width_m=width,
            subcharge_count=subcharges,
            refit_every=refit,
        ),
    )
    duration = 30e-12
    dt = duration * end_fraction / (steps - 1)
    for j in range(1, steps):
        t = j * dt
        beta = np.array(
            [
                0.25 * np.sin(np.pi * t / duration / 2) ** 2,
                0,
                -0.85 + 0.12 * np.sin(np.pi * t / duration / 2) ** 2,
            ]
        )
        old = s.velocity_m_s.copy()
        s.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
        s.position_m += 0.5 * dt * (old + s.velocity_m_s)
        corr.accept([s], t)
    return s, corr


def uniform():
    rows = []
    for beta in ((0, 0, 0), (0.3, -0.2, 0.8), (0, 0, 0.995)):
        for sub in (4, 16, 64):
            s = make(beta=beta)
            corr = CloudCorrection(
                [s], CorrectionConfig(clouds_per_species=8, subcharge_count=sub)
            )
            for j in range(1, 5):
                s.position_m += 1e-12 * s.velocity_m_s
                corr.accept([s], j * 1e-12)
            probes = np.array([[2.4, 1.8, 0.7], [-3.2, 2.2, 1.1]]) * 1e-3
            qe, qb = corr.quasi_static(probes)
            de, db = corr.direct(probes)
            rows.append(
                dict(
                    beta=beta,
                    subcharges=sub,
                    relative_e=float(np.linalg.norm(de) / np.linalg.norm(qe)),
                    relative_c_b=float(C * np.linalg.norm(db) / np.linalg.norm(qe)),
                    max_delta_e_v_m=float(np.max(np.abs(de))),
                )
            )
    save("uniform", rows)


def convergence():
    probes = np.array([[3.1, 2.1, -4.2], [3.4, 1.8, -4.5], [2.9, 1.9, -4.4]]) * 1e-3
    reference = accelerated_source(128, k=128)[1].direct(probes)
    scale = np.linalg.norm(reference[0])
    k_rows = []
    for k in (2, 4, 8, 16, 32, 64, 128):
        _, corr = accelerated_source(128, k=k)
        e, b = corr.direct(probes)
        k_rows.append(
            dict(
                k=k,
                relative_e_error=float(np.linalg.norm(e - reference[0]) / scale),
                delta_e_v_m=e.tolist(),
            )
        )
    s, corr = accelerated_source(32)
    direct_e, direct_b = corr.direct(probes)
    obs = Species("probes", probes, np.zeros_like(probes), Q, MP, np.array([1, 2, 3]))
    mesh_rows = []
    for n in (4, 8, 16):
        cfg = replace(corr.config, lattice_shape=(n,) * 3, half_extent_m=(0.65e-3,) * 3)
        corr.config = cfg
        e, b = corr.sample(obs)
        mesh_rows.append(
            dict(
                nodes_per_axis=n,
                spacing_mm=1.3 / (n - 1),
                relative_e_error=float(
                    np.linalg.norm(e - direct_e) / np.linalg.norm(direct_e)
                ),
                relative_b_error=float(
                    np.linalg.norm(b - direct_b) / np.linalg.norm(direct_b)
                ),
            )
        )
    widths = []
    narrow = accelerated_source(32, width=1e-8)[1].direct(probes)[0]
    for w in (0.32e-3, 0.16e-3, 0.08e-3, 0.04e-3):
        _, c = accelerated_source(32, width=w)
        e, b = c.direct(probes)
        widths.append(
            dict(
                width_mm=w * 1e3,
                relative_e_to_narrow=float(
                    np.linalg.norm(e - narrow) / np.linalg.norm(narrow)
                ),
            )
        )
    time_rows = []
    for steps in (76, 151, 301, 601):
        _, c = accelerated_source(32, steps=steps)
        e, b = c.direct(probes)
        time_rows.append(dict(steps=steps, delta_e_v_m=e.tolist()))
    fine = np.array(time_rows[-1]["delta_e_v_m"])
    for row in time_rows:
        row["relative_e_to_finest"] = float(
            np.linalg.norm(np.array(row["delta_e_v_m"]) - fine) / np.linalg.norm(fine)
        )
    quadrature = []
    for sub in (4, 16, 64):
        _, c = accelerated_source(32, subcharges=sub)
        e, b = c.direct(probes)
        quadrature.append(dict(subcharges=sub, delta_e_v_m=e.tolist()))
    fineq = np.array(quadrature[-1]["delta_e_v_m"])
    for row in quadrature:
        row["relative_e_to_64"] = float(
            np.linalg.norm(np.array(row["delta_e_v_m"]) - fineq) / np.linalg.norm(fineq)
        )
    save(
        "convergence",
        dict(
            k=k_rows,
            lattice=mesh_rows,
            width=widths,
            timestep=time_rows,
            quadrature=quadrature,
            interpretation=(
                "Width changes the physical model; narrow-width limit "
                "is an exterior-probe control."
            ),
        ),
    )


def kick_comparison(solver, sources, obs, corr, dt):
    e, b = solver.sample(solver.fields(sources), obs.position_m)
    de, db = corr.sample(obs)
    re, rb = corr.direct(obs.position_m)
    qe, qb = corr.quasi_static(obs.position_m)
    reference_u = solver.push(obs, qe + re, qb + rb, dt)
    baseline_u = solver.push(obs, e, b, dt)
    corrected_u = solver.push(obs, e + de, b + db, dt)
    p0 = obs.mass_kg * C * (baseline_u - obs.momentum_mc)
    pc = obs.mass_kg * C * (corrected_u - obs.momentum_mc)
    pr = obs.mass_kg * C * (reference_u - obs.momentum_mc)
    # Centred transverse focusing is meaningful when the dipole sum vanishes.
    center = np.average(obs.position_m, weights=obs.population, axis=0)
    transverse = obs.position_m[:, :2] - center[:2]
    denominator = np.sum(obs.population * np.sum(transverse**2, axis=1))
    focus = [
        float(
            np.sum(obs.population * np.sum(transverse * p[:, :2], axis=1)) / denominator
        )
        for p in (p0, pc, pr)
    ]
    ledger = correction_ledger(obs, baseline_u, corrected_u, de, db, dt)
    return dict(
        pic=p0.tolist(),
        corrected=pc.tolist(),
        exact_cloud_lw=pr.tolist(),
        pic_relative_error=float(np.linalg.norm(p0 - pr) / np.linalg.norm(pr)),
        corrected_relative_error=float(np.linalg.norm(pc - pr) / np.linalg.norm(pr)),
        focus_pic_corrected_lw=focus,
        correction_ledger=ledger,
        delta_e_relative_to_lw=float(np.linalg.norm(re) / np.linalg.norm(qe + re)),
    )


def acceleration():
    s, corr = accelerated_source(8)
    center = np.average(s.position_m, axis=0, weights=s.population)
    # Eight centred probe offsets around an exterior observer centroid.
    offsets = np.array(list(itertools.product((-1, 1), repeat=3))) * 0.12e-3
    obs = Species(
        "probes",
        center + np.array([3e-3, 2e-3, 0]) + offsets,
        np.tile([0, 0, 0.7], (8, 1)),
        Q,
        MP,
        np.full(8, 1e5),
    )
    corr.config = replace(
        corr.config, lattice_shape=(8,) * 3, half_extent_m=(0.4e-3,) * 3
    )
    rows = []
    for mesh in (24, 48, 96):
        solver = ElectrostaticPIC((mesh,) * 3, (2e-3,) * 3, 2)
        rows.append(
            dict(pic_mesh=mesh, **kick_comparison(solver, [s], obs, corr, 1e-13))
        )
    save(
        "acceleration",
        dict(
            rows=rows,
            source=(
                "smooth prescribed acceleration, accepted lab-time knots, "
                "inertial prehistory"
            ),
            radiation_reaction_mode="off",
            reference=(
                "Exact retarded charge-cloud provider at particle events, "
                "same eight source clouds"
            ),
        ),
    )


def crossing(mesh=24, steps=1201, nodes=6):
    # Centred corners resolve the head and tail without a steering offset.
    offsets = (
        np.array(list(itertools.product((-1, 1), repeat=3)))
        * np.array([0.65, 0.65, 0.08])
        * 1e-3
    )
    bp = np.sqrt(1 - (1 + 216 / 938.27208816) ** -2)
    be = np.sqrt(1 - 10**-2)
    p = Species(
        "protons",
        offsets + np.array([0, 0, -5e-3]),
        np.tile([0, 0, bp / np.sqrt(1 - bp * bp)], (8, 1)),
        Q,
        MP,
        np.full(8, 100e-12 / (8 * Q)),
    )
    e = Species(
        "electrons",
        offsets + np.array([0, 0, 5e-3]),
        np.tile([0, 0, -be / np.sqrt(1 - be * be)], (8, 1)),
        -Q,
        ME,
        np.full(8, 20e-12 / (8 * Q)),
    )
    species = [p, e]
    dt = 10e-3 / ((bp + be) * C * (2 * (steps - 1) // 3))
    solver = ElectrostaticPIC((mesh,) * 3, (4e-3,) * 3, 2)
    config = CorrectionConfig(
        clouds_per_species=8,
        lattice_shape=(nodes,) * 3,
        half_extent_m=(1.4e-3, 1.4e-3, 0.4e-3),
        cloud_width_m=0.16e-3,
    )
    corr = [CloudCorrection([s], config) for s in species]
    samples = np.unique(
        np.linspace((steps - 1) // 2, 5 * (steps - 1) // 6, 17).astype(int)
    )
    rows = []
    for step in range(steps):
        if step in samples:
            results = [
                kick_comparison(solver, [species[1 - i]], species[i], corr[1 - i], dt)
                for i in (0, 1)
            ]
            rows.append(
                dict(
                    step=step,
                    time_s=step * dt,
                    proton_cross_kick=results[0],
                    electron_cross_kick=results[1],
                )
            )
        if step == steps - 1:
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
        for s, c in zip(species, corr):
            c.accept([s], (step + 1) * dt)

    def aggregate(label):
        kicks = np.array([row[label]["exact_cloud_lw"] for row in rows])
        pic = np.array([row[label]["pic"] for row in rows])
        corrected = np.array([row[label]["corrected"] for row in rows])
        times = np.array([row["time_s"] for row in rows])
        # The per-step kicks are converted to forces before time quadrature.
        ref = np.trapezoid(kicks / dt, times, axis=0)
        base = np.trapezoid(pic / dt, times, axis=0)
        fix = np.trapezoid(corrected / dt, times, axis=0)
        return dict(
            force_pic_relative_error=float(
                np.linalg.norm(pic - kicks) / np.linalg.norm(kicks)
            ),
            force_corrected_relative_error=float(
                np.linalg.norm(corrected - kicks) / np.linalg.norm(kicks)
            ),
            impulse_pic_relative_error=float(
                np.linalg.norm(base - ref) / np.linalg.norm(ref)
            ),
            impulse_corrected_relative_error=float(
                np.linalg.norm(fix - ref) / np.linalg.norm(ref)
            ),
            max_delta_e_relative_to_lw=max(
                row[label]["delta_e_relative_to_lw"] for row in rows
            ),
        )

    raw = {}
    for name, c in zip(("protons", "electrons"), corr):
        h = c.builder.build_current()
        raw[f"{name}_time_ns"] = h.t
        raw[f"{name}_position_mm"] = np.stack((h.x, h.y, h.z), axis=-1)
        raw[f"{name}_beta"] = np.stack((h.bx, h.by, h.bz), axis=-1)
        raw[f"{name}_charge_c"] = c.charge_c
    np.savez_compressed(OUT / f"crossing_m{mesh}_s{steps}_histories.npz", **raw)
    save(
        f"crossing_m{mesh}_s{steps}_n{nodes}",
        dict(
            rows=rows,
            protons=aggregate("proton_cross_kick"),
            electrons=aggregate("electron_cross_kick"),
            dt_s=dt,
            centred_initial_samples=8,
            accepted_steps=steps - 1,
            pic_mesh=mesh,
            correction_nodes=nodes,
            initial_proton_charge_c=100e-12,
            initial_electron_charge_c=-20e-12,
            reference=(
                "Direct exact-LW cloud cross-bunch kicks on identical accepted PIC "
                "source trajectories; source K=8, observer count=8"
            ),
            radiation_reaction_mode="off",
            limitation=(
                "Not self-consistent exact-LW trajectories; reference uses "
                "the same finite cloud width, and omits same-bunch force "
                "from the cross-kick observable."
            ),
        ),
    )


def cadence():
    probes = np.array([[3.1, 2.1, -4.2], [3.4, 1.8, -4.5]]) * 1e-3
    ref = accelerated_source(8, steps=321)[1].direct(probes)[0]
    fits = []
    for m in (1, 2, 4, 8):
        _, c = accelerated_source(8, steps=321, refit=m)
        e, _ = c.direct(probes)
        fits.append(
            dict(
                refit_every=m,
                relative_e_to_m1=float(np.linalg.norm(e - ref) / np.linalg.norm(ref)),
            )
        )
    lag = []
    for interval in (2e-12, 1e-12, 0.5e-12, 0.25e-12):
        _, c = accelerated_source(
            8, steps=321, end_fraction=1 - 0.5 * interval / 30e-12
        )
        e, _ = c.direct(probes)
        lag.append(
            dict(
                lab_step_s=interval,
                relative_e_error=float(np.linalg.norm(e - ref) / np.linalg.norm(ref)),
            )
        )
    save(
        "cadence",
        dict(
            refit=fits,
            accepted_time_lag=lag,
            interpretation="First-order accepted-time correction; "
            "refit cadence and temporal lag are independent controls.",
        ),
    )


def pulse_sampling():
    path = OUT / "crossing_m48_s1201_histories.npz"
    data = np.load(path)
    species = []
    names = ("protons", "electrons")
    for name, charge, mass, total in zip(names, (Q, -Q), (MP, ME), (100e-12, 20e-12)):
        beta = data[f"{name}_beta"][1]
        u = beta / np.sqrt(1 - np.sum(beta * beta, axis=1))[:, None]
        species.append(
            Species(
                name,
                data[f"{name}_position_mm"][1] * 1e-3,
                u,
                charge,
                mass,
                np.full(8, total / (8 * Q)),
            )
        )
    clouds = [
        CloudCorrection(
            [s], CorrectionConfig(clouds_per_species=8, cloud_width_m=0.16e-3)
        )
        for s in species
    ]
    times = data["protons_time_ns"][1:, 0] * 1e-9
    dt = times[1] - times[0]
    solver = ElectrostaticPIC()
    forces = [[], []]
    sample_times = []
    for step in range(1, len(times)):
        for name, s, c in zip(names, species, clouds):
            beta = data[f"{name}_beta"][step + 1]
            s.position_m[:] = data[f"{name}_position_mm"][step + 1] * 1e-3
            s.momentum_mc[:] = beta / np.sqrt(1 - np.sum(beta * beta, axis=1))[:, None]
            c.accept([s], times[step])
        if 600 <= step <= 1000:
            sample_times.append(times[step])
            for i, obs in enumerate(species):
                c = clouds[1 - i]
                de, db = c.direct(obs.position_m)
                qe, qb = c.quasi_static(obs.position_m)
                after = solver.push(obs, qe + de, qb + db, dt)
                forces[i].append(obs.mass_kg * C * (after - obs.momentum_mc) / dt)
        if step == 1000:
            break
    sample_times = np.asarray(sample_times)
    result = {}
    for name, force in zip(names, forces):
        force = np.asarray(force)
        fine = np.trapezoid(force, sample_times, axis=0)
        rows = []
        for count in (17, 33, 65, 129, 401):
            indices = np.unique(np.rint(np.linspace(0, 400, count)).astype(int))
            impulse = np.trapezoid(force[indices], sample_times[indices], axis=0)
            rows.append(
                dict(
                    samples=len(indices),
                    relative_impulse_to_401=float(
                        np.linalg.norm(impulse - fine) / np.linalg.norm(fine)
                    ),
                )
            )
        result[name] = rows
    save(
        "pulse_sampling",
        dict(
            results=result,
            source_artifact=path.name,
            interpretation="Exact-cloud reference quadrature on "
            "frozen accepted fine-mesh PIC histories; 401 is "
            "a sampling control, not a source-timestep certificate.",
        ),
    )


def manifest():
    paths = list((ROOT / "core/pic").glob("*.py")) + [
        ROOT / "core/retarded_fields.py",
        Path(__file__).resolve(),
        ROOT / "tests/test_pic_correction.py",
        ROOT / "configs/run_configs/example_native_pic_correction.json",
    ]
    save(
        "source_manifest",
        dict(
            files={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths
            },
            base_commit="e2df1e645cad2af368c9ba2a00a30d0ae9ddec3e",
            python=sys.version,
            machine=platform.machine(),
            numpy=np.__version__,
            artifact_root=str(OUT),
            max_concurrent_compute_processes=4,
            radiation_reaction_mode="off",
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parts",
        nargs="+",
        default=[
            "uniform",
            "convergence",
            "acceleration",
            "cadence",
            "crossing",
            "manifest",
        ],
    )
    parser.add_argument("--mesh", type=int, default=24)
    parser.add_argument("--steps", type=int, default=1201)
    parser.add_argument("--nodes", type=int, default=6)
    args = parser.parse_args()
    for part in args.parts:
        start = time.monotonic()
        if part == "crossing":
            crossing(args.mesh, args.steps, args.nodes)
        else:
            globals()[part]()
        print(part, "seconds", round(time.monotonic() - start, 3), flush=True)


if __name__ == "__main__":
    main()
