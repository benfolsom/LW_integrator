"""Reproducible stage 1 accuracy and timing matrix; artifacts outside ~/work.

Run from the checkout with PYTHONPATH=. and the specified existing interpreter.
This reads retained external benchmarks without modifying the study checkout.
"""

import argparse
import csv
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtri
from scipy.stats import qmc

from core.pic import ElectrostaticPIC, NumpyBackend, Species
from core.pic.kernels import C, COULOMB


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def error(reference, got):
    delta = np.linalg.norm(got - reference, axis=-1)
    return dict(
        max_relative=float(np.max(delta / np.linalg.norm(reference, axis=-1))),
        relative_norm=float(
            np.linalg.norm(got - reference) / np.linalg.norm(reference)
        ),
    )


def ellipsoid_field(position, axes, charge):
    scale = np.max(axes) ** 2
    a2 = axes * axes / scale
    result = []
    for p in position:
        x = p / np.sqrt(scale)
        lam = 0.0
        if np.sum(x * x / a2) > 1:
            lam = brentq(lambda s: np.sum(x * x / (a2 + s)) - 1, 0, np.sum(x * x))
        integrals = [
            quad(
                lambda y: np.exp(y)
                / ((a2[i] + np.expm1(y)) * np.sqrt(np.prod(a2 + np.expm1(y)))),
                np.log1p(lam),
                45,
                epsabs=1e-13,
                epsrel=1e-11,
            )[0]
            for i in range(3)
        ]
        result.append(1.5 * COULOMB * charge * p * np.asarray(integrals) / scale**1.5)
    return np.asarray(result)


def gaussian_field(position, sigma, charge):
    scale = np.max(sigma) ** 2
    a2 = sigma * sigma / scale
    result = []
    for p in position:
        x = p / np.sqrt(scale)
        integrals = [
            quad(
                lambda y: np.exp(y)
                * np.exp(-0.5 * np.sum(x * x / (a2 + np.expm1(y))))
                / ((a2[i] + np.expm1(y)) * np.sqrt(np.prod(a2 + np.expm1(y)))),
                0,
                45,
                epsabs=1e-13,
                epsrel=1e-11,
            )[0]
            for i in range(3)
        ]
        result.append(
            COULOMB
            * charge
            * p
            * np.asarray(integrals)
            / (np.sqrt(2 * np.pi) * scale**1.5)
        )
    return np.asarray(result)


def analytic_matrix(out, power=20):
    # Quiet reflection symmetry, fixed population, and fixed analytic density.
    from itertools import product

    sigma = np.array([0.002, 0.002, 0.001397])
    beta = 0.582446
    gamma = 1 / np.sqrt(1 - beta * beta)
    axis = np.array([0.3, 0.4, np.sqrt(0.75)])
    from core.pic.grid import rest_basis

    basis = rest_basis(axis)
    half = 6 * sigma
    u0 = beta * gamma
    probes = (
        np.array([[0.7, 0.4, 0.3], [1.4, 0.3, 0.6], [3.0, 0.3, 0.2], [0.2, 0.3, 4.5]])
        * sigma
    )
    rows = []
    for distribution in ("gaussian", "ellipsoid"):
        if distribution == "gaussian":
            a = qmc.Sobol(3, scramble=True, seed=20261005).random_base2(power - 3)
            base = np.abs(ndtri(a)) * sigma
        else:
            a = qmc.Sobol(3, scramble=True, seed=20261004).random_base2(power - 3)
            z = 2 * np.sin(np.arcsin(a[:, 0]) / 3)
            radius = np.sqrt((1 - z * z) * a[:, 1])
            phi = np.pi * a[:, 2] / 2
            base = (
                np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])
                * np.sqrt(5)
                * sigma
            )
        position = np.vstack(
            [base * np.array(sign) for sign in product((-1.0, 1.0), repeat=3)]
        )
        rest_probes = probes.copy()
        rest_probes[:, 2] *= gamma
        rest_axes = sigma.copy()
        rest_axes[2] *= gamma
        exact = (
            gaussian_field(rest_probes, rest_axes, 1e-10)
            if distribution == "gaussian"
            else ellipsoid_field(rest_probes, np.sqrt(5) * rest_axes, 1e-10)
        )
        exact[:, :2] *= gamma
        exact = exact @ basis.T
        mom = np.tile(u0 * axis, (len(position), 1))
        species = Species(
            distribution,
            position @ basis.T,
            mom,
            1.602176634e-19,
            1.67262192369e-27,
            np.full(len(position), 1e-10 / 1.602176634e-19 / len(position)),
        )
        for order in (1, 2):
            previous = None
            for n in (24, 48, 96, 192):
                start = time.monotonic()
                solver = ElectrostaticPIC((n,) * 3, half, order)
                fields = solver.fields([species])
                electric, magnetic = solver.sample(fields, probes @ basis.T)
                record = dict(
                    distribution=distribution,
                    shape_order=order,
                    mesh=n,
                    particles=len(position),
                    electric_error=error(exact, electric),
                    electric_V_m=electric.tolist(),
                    elapsed_s=time.monotonic() - start,
                    magnetic_boost_relative_norm=float(
                        np.linalg.norm(magnetic - np.cross(beta * axis, electric) / C)
                        / np.linalg.norm(magnetic)
                    ),
                )
                if previous:
                    record["observed_order"] = np.log2(
                        previous["electric_error"]["relative_norm"]
                        / record["electric_error"]["relative_norm"]
                    )
                previous = record
                rows.append(record)
                save(
                    out
                    / ("analytic.json" if power == 20 else f"analytic_p{power}.json"),
                    rows,
                )
                print("analytic", record, flush=True)
    return rows


def benchmark_matrix(study, out, meshes, events):
    sys.path.insert(0, str(study / "scripts"))
    from pic_common import beams, cloud, probes, E, Q

    b = beams()[0]
    reference_path = study / "results/pic_20261004/reference/c0_o16_n65.json"
    reference = json.loads(reference_path.read_text())
    exact = np.asarray(reference["force_N"]["exact"])
    fraction_ref = np.asarray(reference["fractions"])
    source, u = cloud(b, 20)
    observer, ou = probes(b)
    source_velocity = u / np.hypot(1, u)
    observer_velocity = ou / np.hypot(1, ou)
    mom = np.column_stack([u * 0, u * 0, u])
    retained = []
    with (study / "results/pic_20261004/field_summary.csv").open() as f:
        for r in csv.DictReader(f):
            if (
                r["case"] == "proton216"
                and r["mode"] == "es"
                and r["production"] == "True"
                and r["particles"] == "1048576"
                and r["bins"] == "4"
                and r["nx"] == "256"
            ):
                path = study / r["path"]
                record = json.loads(path.read_text())
                retained.append(
                    dict(
                        code=r["code"],
                        fraction=float(r["fraction"]),
                        force_N=record["force_N"],
                        path=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    rows = []
    for n in meshes:
        fractions = np.linspace(0, 1, events if n == meshes[0] else 3)
        forces = []
        for fraction in fractions:
            p = source.copy()
            p[:, 2] += (source_velocity - b.beta) * C * b.duration * fraction
            o = observer.copy()
            o[:, 2] += (observer_velocity - b.beta) * C * b.duration * fraction
            for groups in ((1, 4) if fraction == 0 else (4,)):
                species = Species(
                    "proton216",
                    p,
                    mom,
                    E,
                    b.mass,
                    np.full(len(u), Q / E / len(u)),
                    groups,
                )
                solver = ElectrostaticPIC((n,) * 3, 6 * np.asarray(b.rms), 2)
                start = time.monotonic()
                fields = solver.fields([species])
                electric, magnetic = solver.sample(fields, o)
                v = np.column_stack([ou * 0, ou * 0, C * observer_velocity])
                force = E * (electric + np.cross(v, magnetic))
                ref = np.array(
                    [
                        [
                            np.interp(fraction, fraction_ref, exact[:, j, k])
                            for k in range(3)
                        ]
                        for j in range(8)
                    ]
                )
                r = dict(
                    mesh=n,
                    groups=groups,
                    fraction=float(fraction),
                    shape_order=2,
                    particles=len(u),
                    force_N=force.tolist(),
                    exact_error=error(ref, force),
                    elapsed_s=time.monotonic() - start,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
                r["external_comparisons"] = [
                    dict(
                        code=x["code"],
                        path=x["path"],
                        sha256=x["sha256"],
                        error=error(np.asarray(x["force_N"]), force),
                    )
                    for x in retained
                    if x["fraction"] == fraction
                ]
                for code in ("warpx", "impact"):
                    for mesh in (128, 256):
                        name = code if mesh == 128 else f"{code}_g{mesh}"
                        live = out / "external" / name / "result.json"
                        if fraction == 0 and live.exists():
                            x = json.loads(live.read_text())
                            r["external_comparisons"].append(
                                dict(
                                    code=x["code"] + f" live {mesh}",
                                    path=str(live),
                                    sha256=hashlib.sha256(
                                        live.read_bytes()
                                    ).hexdigest(),
                                    error=error(np.asarray(x["force_N"]), force),
                                )
                            )
                rows.append(r)
                save(out / "proton_snapshots.json", rows)
                print(
                    "proton",
                    n,
                    groups,
                    fraction,
                    r["exact_error"],
                    r["elapsed_s"],
                    flush=True,
                )
                if groups == 4:
                    forces.append(force)
                del fields, solver
        impulse = np.trapezoid(forces, fractions * b.duration, axis=0)
        ref_impulse = np.asarray(reference["impulse_kg_m_s"]["exact"])
        result = dict(
            mesh=n,
            events=len(fractions),
            impulse_kg_m_s=impulse.tolist(),
            exact_error=error(ref_impulse, impulse),
            interpretation="prescribed ballistic snapshots, not self-consistent drift",
            time_resolved=len(fractions) >= 33,
        )
        if len(fractions) > 3:
            coarse = np.trapezoid(
                np.asarray(forces)[::2], fractions[::2] * b.duration, axis=0
            )
            result["nested_time_change"] = error(impulse, coarse)
        save(out / f"proton_impulse_{n}.json", result)
        print("impulse", result, flush=True)
    save(
        out / "reference_provenance.json",
        dict(
            path=str(reference_path),
            sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            retained_external_records=retained,
        ),
    )


def timings(out):
    from core.pic.kernels import (
        deposit_kernel,
        gather_kernel,
        boris_kernel,
        current_kernel,
    )

    rng = np.random.default_rng(20261005)
    rows = []
    for count in (65536, 1048576):
        coordinates = rng.uniform(2, 61, (count, 3))
        charge = np.full(count, 1e-10 / count)
        u = rng.normal(size=(count, 3))
        e = rng.normal(size=(count, 3)) * 1e4
        b = rng.normal(size=(count, 3)) * 1e-4
        qm = np.full(count, 9.5788e7)
        new = coordinates + rng.uniform(-0.2, 0.2, coordinates.shape)
        for order in (1, 2):
            backend = NumpyBackend()
            grid_charge = deposit_kernel(coordinates, charge, (64,) * 3, order)
            field, _ = backend.solve(grid_charge, np.full(3, 0.0002))
            jobs = dict(
                deposit=lambda: deposit_kernel(coordinates, charge, (64,) * 3, order),
                gather=lambda: gather_kernel(coordinates, field, order),
                push=lambda: boris_kernel(u, e, b, qm, 1e-12),
                current=lambda: current_kernel(
                    coordinates,
                    new,
                    charge,
                    (64,) * 3,
                    np.full(3, 0.0002),
                    1e-12,
                    order,
                ),
                solve=lambda: backend.solve(grid_charge, np.full(3, 0.0002)),
            )

            def pipeline():
                rho = backend.deposit(coordinates, charge, (64,) * 3, order)
                field, _ = backend.solve(rho, np.full(3, 0.0002))
                gathered = backend.gather(coordinates, field, order)
                backend.push(u, gathered, b, qm, 1e-12)
                backend.current(
                    coordinates,
                    new,
                    charge,
                    (64,) * 3,
                    np.full(3, 0.0002),
                    1e-12,
                    order,
                )

            jobs["deposit_solve_gather_push_current"] = pipeline
            for name, job in jobs.items():
                job()  # compilation, plans, and allocations warmed before timing
                samples = []
                for _ in range(3):
                    start = time.perf_counter()
                    job()
                    samples.append(time.perf_counter() - start)
                elapsed = float(np.median(samples))
                rows.append(
                    dict(
                        kernel=name,
                        particles=count,
                        shape_order=order,
                        mesh=[64] * 3,
                        median_s=elapsed,
                        particles_per_s=count / elapsed,
                        includes_compile=False,
                        includes_green_setup=False,
                    )
                )
    save(out / "timings.json", rows)
    save(
        out / "machine.json",
        dict(
            platform=platform.platform(),
            python=sys.version,
            numpy=np.__version__,
            compute_cap=Path("/private/tmp/compute_cap").read_text().strip(),
        ),
    )


def source_check(study, out):
    """Check the final frame reductions and the native source-count floor."""
    sys.path.insert(0, str(study / "scripts"))
    from pic_common import beams, cloud, probes, E, Q

    b = beams()[0]
    prior = json.loads((out / "proton_snapshots.json").read_text())
    records = []
    for power, fraction in ((20, 0.0), (20, 1.0), (22, 0.0)):
        p, u = cloud(b, power)
        o, ou = probes(b)
        p[:, 2] += (u / np.hypot(1, u) - b.beta) * C * b.duration * fraction
        o[:, 2] += (ou / np.hypot(1, ou) - b.beta) * C * b.duration * fraction
        s = Species(
            "proton216",
            p,
            np.column_stack([u * 0, u * 0, u]),
            E,
            b.mass,
            np.full(len(u), Q / E / len(u)),
            4,
        )
        solver = ElectrostaticPIC((128,) * 3, 6 * np.asarray(b.rms), 2)
        start = time.monotonic()
        fields = solver.fields([s])
        ef, bf = solver.sample(fields, o)
        v = np.column_stack([ou * 0, ou * 0, C * ou / np.hypot(1, ou)])
        force = E * (ef + np.cross(v, bf))
        baseline = next(
            x
            for x in prior
            if x["mesh"] == 128 and x["groups"] == 4 and x["fraction"] == fraction
        )
        record = dict(
            particles=len(u),
            fraction=fraction,
            force_N=force.tolist(),
            elapsed_s=time.monotonic() - start,
            change_from_original_million=error(np.asarray(baseline["force_N"]), force),
        )
        records.append(record)
        print("source-check", record, flush=True)
        save(out / "proton_source_check.json", records)


def summarize(study, out):
    rows = json.loads((out / "proton_snapshots.json").read_text())
    mesh = rows[0]["mesh"]
    primary = [r for r in rows if r["mesh"] == mesh and r["groups"] == 4]
    impulse = json.loads((out / f"proton_impulse_{mesh}.json").read_text())
    result = dict(
        target=0.001,
        primary_mesh=[mesh] * 3,
        groups=4,
        shape_order=2,
        events=len(primary),
        force_exact_max_relative=max(x["exact_error"]["max_relative"] for x in primary),
        impulse_exact_max_relative=impulse["exact_error"]["max_relative"],
        nested_time_change=impulse.get("nested_time_change", {}).get("max_relative"),
    )
    external = json.loads(
        (study / "results/pic_20261004/drift_impulses.json").read_text()
    )
    for code in ("WarpX", "IMPACT-T"):
        comparisons = [
            x["error"]["max_relative"]
            for r in primary
            for x in r["external_comparisons"]
            if x["code"] == code
        ]
        if comparisons:
            result[f"{code}_force_max_relative"] = max(comparisons)
        reference = next(
            x
            for x in external
            if x["code"] == code
            and x["case"] == "proton216"
            and x["mesh"] == [256] * 3
            and x["bins"] == 4
        )
        result[f"{code}_impulse_max_relative"] = error(
            np.asarray(reference["impulse_kg_m_s"]),
            np.asarray(impulse["impulse_kg_m_s"]),
        )["max_relative"]
    fine = next(
        (
            x
            for x in rows
            if x["mesh"] == 256 and x["groups"] == 4 and x["fraction"] == 0
        ),
        None,
    )
    if fine:
        for code in ("warpx", "impact"):
            live = out / "external" / f"{code}_g256" / "result.json"
            if live.exists():
                reference = json.loads(live.read_text())
                result[f"{code}_fresh256_max_relative"] = error(
                    np.asarray(reference["force_N"]), np.asarray(fine["force_N"])
                )["max_relative"]
    source = json.loads((out / "proton_source_check.json").read_text())
    result["source_count_change"] = source[-1]["change_from_original_million"][
        "max_relative"
    ]
    result["final_code_frame_reduction_change"] = max(
        x["change_from_original_million"]["max_relative"] for x in source[:2]
    )
    result["proton_gate_passed"] = len(primary) >= 33 and all(
        value < 0.001 for key, value in result.items() if key.endswith("max_relative")
    )
    gate = (
        study.parent
        / "space_charge_crossing_compensation/compensation_v2/convergence_gate.json"
    )
    if gate.exists():
        result["compensation_gate"] = json.loads(gate.read_text())
    result["fine_three_event_impulse_time_resolved"] = False
    save(out / "validation_summary.json", result)
    print("summary", result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        default=Path.home()
        / "work/extracurr/LW_feasibility_studies/studies"
        / "moderate_beta_space_charge_benchmarks",
    )
    parser.add_argument(
        "--output", type=Path, default=Path.home() / "compute-data/native_pic"
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=["analytic", "proton", "timings", "source-check", "summary"],
        default=["analytic", "proton", "timings"],
    )
    parser.add_argument("--meshes", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--events", type=int, default=33)
    parser.add_argument("--analytic-power", type=int, default=20)
    args = parser.parse_args()
    output = args.output.resolve()
    work = (Path.home() / "work").resolve()
    if output == work or work in output.parents:
        parser.error("bulky validation artifacts must be outside ~/work")
    if "analytic" in args.parts:
        analytic_matrix(output, args.analytic_power)
    if "proton" in args.parts:
        benchmark_matrix(args.study, output, args.meshes, args.events)
    if "timings" in args.parts:
        timings(output)
    if "source-check" in args.parts:
        source_check(args.study, output)
    if "summary" in args.parts:
        summarize(args.study, output)
