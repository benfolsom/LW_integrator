"""Continuum Gaussian controls for prescribed rigid source trajectories.

No sampled PIC particle appears in the independent tensor GH reference.
Rigid translation lets the maintained exact point provider be evaluated at
translated observers, avoiding thousands of identical history preparations.
All heavy artifacts live outside the repository. RR is explicitly off in
these prescribed-source field and matched-kick diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from core import retarded_fields as rf
from core.pic import CloudCorrection, CorrectionConfig, ElectrostaticPIC, Species
from core.pic.correction import E_NATIVE_TO_SI
from core.pic.correction_fields import resolved_node_fields
from core.pic.grid import Grid
from core.pic.kernels import C, COULOMB
from scripts.validate_pic_correction import Q, MP, ME

SIGMA = 0.25e-3
OUT = Path.home() / "compute-data/pic_correction_k"
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CASES = {
    "acceleration": dict(
        b0=[0, 0, -0.85],
        change=[0.25, 0, 0.12],
        duration=30e-12,
        displacement=[3e-3, 2e-3, 0.5e-3],
        observer_beta=0.8,
        charge=Q,
        mass=MP,
        total=1e-12,
    ),
    "crossing_electron_source": dict(
        b0=[0, 0, -0.94],
        change=[0.03, 0, -0.04],
        duration=1e-9,
        displacement=[0.7e-3, 0, -4e-3],
        observer_beta=0.85,
        charge=-Q,
        mass=ME,
        total=-20e-12,
    ),
    "crossing_proton_source": dict(
        b0=[0, 0, 0.8],
        change=[-0.03, 0, 0.05],
        duration=1e-9,
        displacement=[0.7e-3, 0, 4e-3],
        observer_beta=-0.98,
        charge=Q,
        mass=MP,
        total=100e-12,
    ),
}


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"
    )
    print(name, "saved", flush=True)


def quiet(n, seed=20261006, sigma=SIGMA):
    points = qmc.Sobol(3, scramble=True, seed=seed).random_base2(int(np.log2(n)) - 1)
    offsets = ndtri(points) * sigma
    return np.concatenate((offsets, -offsets))


def species(offsets, case):
    beta = np.array(case["b0"])
    return Species(
        "source",
        offsets,
        np.broadcast_to(beta / np.sqrt(1 - beta @ beta), offsets.shape),
        case["charge"],
        case["mass"],
        case["total"] / (len(offsets) * case["charge"]),
    )


def trajectory(case, fraction):
    duration = case["duration"]
    t = fraction * duration
    b0 = np.array(case["b0"])
    change = np.array(case["change"])
    beta = b0 + change * np.sin(np.pi * fraction / 2) ** 2
    displacement = C * (
        b0 * t + change * (t / 2 - duration * np.sin(np.pi * fraction) / (2 * np.pi))
    )
    return t, displacement, beta


def prescribe(correction, case, steps=601):
    """Rigid cohort centroids, equivalent to refitting translated particles.

    Cohorts, populations, widths, and initial first moments come from the
    actual large-N production fitter. Translation is applied to those first
    moments directly, avoiding N*steps repeated identical rigid reductions.
    """
    centers = correction.initial_position_m.copy()
    for fraction in np.linspace(0, 1, steps)[1:]:
        t, displacement, beta = trajectory(case, fraction)
        previous = correction.beta.copy()
        correction.velocity_beta[:] = beta
        correction.position_m = centers + displacement
        correction.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
        prime = (correction.beta - previous) / ((t - correction.time_s) * C * 1e3)
        correction.builder.append_step(
            correction._state(t, correction.position_m, prime)
        )
        correction.time_s = t
        correction.accepted_steps += 1
    return correction


def point_history(case, steps=601):
    source = species(np.zeros((1, 3)), case)
    correction = prescribe(
        CloudCorrection([source], CorrectionConfig(clouds_per_species=1)), case, steps
    )
    return correction, rf._prepare_history(correction.builder.build_current(), ())


def quasi(positions, source_center, beta, charge):
    r = positions - source_center
    speed = np.linalg.norm(beta)
    axis = beta / speed
    parallel = r @ axis
    transverse = r - parallel[:, None] * axis
    deficit = 1 - beta @ beta
    d2 = parallel**2 + deficit * np.sum(transverse**2, axis=1)
    e = COULOMB * charge * deficit * r / d2[:, None] ** 1.5
    return e, np.cross(beta, e) / C


def integrate(point, prepared, nodes, offsets, fractions):
    """Dense exact point-field quadrature, keeping each source's Δ in float64."""
    electric = np.zeros_like(nodes)
    magnetic = np.zeros_like(nodes)
    qe = np.zeros_like(nodes)
    qb = np.zeros_like(nodes)
    start = time.perf_counter()
    for first in range(0, len(offsets), 128):
        off = offsets[first : first + 128]
        weights = fractions[first : first + 128]
        translated = (nodes[None, :, :] - off[:, None, :]).reshape(-1, 3)
        e, b = resolved_node_fields(
            prepared, np.full(len(translated), point.time_s * 1e9), translated * 1e3
        )
        e *= E_NATIVE_TO_SI
        b *= E_NATIVE_TO_SI / C
        se, sb = quasi(
            translated, point.position_m[0], point.beta[0], point.charge_c[0]
        )
        electric += np.einsum(
            "i,ijk->jk", weights, (e - se).reshape(len(off), len(nodes), 3)
        )
        magnetic += np.einsum(
            "i,ijk->jk", weights, (b - sb).reshape(len(off), len(nodes), 3)
        )
        qe += np.einsum("i,ijk->jk", weights, se.reshape(len(off), len(nodes), 3))
        qb += np.einsum("i,ijk->jk", weights, sb.reshape(len(off), len(nodes), 3))
    return electric, magnetic, qe, qb, time.perf_counter() - start


def gaussian_rule(order):
    x, w = np.polynomial.hermite.hermgauss(order)
    offsets = (
        np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
        * np.sqrt(2)
        * SIGMA
    )
    weights = np.einsum("i,j,k->ijk", w, w, w).ravel() / np.pi**1.5
    return offsets, weights / weights.sum()


def observers(case, count):
    _, center, _ = trajectory(case, 1)
    center += np.array(case["displacement"])
    s = Species(
        "observers",
        quiet(count, seed=20261007, sigma=0.025e-3) + center,
        np.broadcast_to(
            [0, 0, case["observer_beta"] / np.sqrt(1 - case["observer_beta"] ** 2)],
            (count, 3),
        ),
        Q if case["observer_beta"] > 0 else -Q,
        MP if case["observer_beta"] > 0 else ME,
        np.ones(count),
    )
    grid = Grid((4,) * 3, np.full(3, 0.1e-3), np.full(3, -0.15e-3))
    local = grid.origin_m + np.indices(grid.shape).reshape(3, -1).T * grid.spacing_m
    return s, grid, local + center, center


def relative(value, reference):
    return float(np.linalg.norm(value - reference) / np.linalg.norm(reference))


def measure(result, reference, obs, grid, center):
    e, b = result[:2]
    re, rb = reference[:2]
    fields = [
        grid.gather(obs.position_m - center, f.reshape(*grid.shape, 3))
        for f in (e, b, re, rb)
    ]
    solver = ElectrostaticPIC()
    # Matched correction-only Boris kicks; an impulse error, not total u error.
    dt = 1e-13
    after = solver.push(obs, fields[0], fields[1], dt) - obs.momentum_mc
    exact = solver.push(obs, fields[2], fields[3], dt) - obs.momentum_mc
    return dict(
        e_error=relative(e, re),
        cb_error=relative(b, rb),
        combined_error=relative(
            np.concatenate((e, C * b)), np.concatenate((re, C * rb))
        ),
        kick_error=relative(after, exact),
        population_impulse_error=relative(after.sum(axis=0), exact.sum(axis=0)),
        field_s=result[-1],
        kick_particles=len(obs.position_m),
        kick_dt_s=dt,
    )


def cloud_offsets(corr):
    offsets = (corr.initial_position_m[:, None, :] + corr.offsets_m).reshape(-1, 3)
    weights = (corr.weights[:, None] / corr.weights.sum() * corr.fractions).ravel()
    return offsets, weights


def frozen_boxes(source, k):
    """Record the production bisection planes for a moment-only N control."""
    groups = [
        (np.arange(len(source.position_m)), np.full(3, -np.inf), np.full(3, np.inf))
    ]
    while len(groups) < k:
        i = max(
            range(len(groups)),
            key=lambda j: float(source.population[groups[j][0]].sum()),
        )
        idx, lo, hi = groups.pop(i)
        axis = int(np.argmax(np.ptp(source.position_m[idx], axis=0)))
        idx = idx[np.argsort(source.position_m[idx, axis], kind="stable")]
        weights = source.population[idx]
        cut = int(np.searchsorted(np.cumsum(weights), weights.sum() / 2)) + 1
        cut = min(max(1, cut), len(idx) - 1)
        plane = 0.5 * (
            source.position_m[idx[cut - 1], axis] + source.position_m[idx[cut], axis]
        )
        left_hi, right_lo = hi.copy(), lo.copy()
        left_hi[axis] = right_lo[axis] = plane
        groups.extend(((idx[:cut], lo, left_hi), (idx[cut:], right_lo, hi)))
    return groups


def frozen_fit(source, boxes, cloud):
    centers, weights = [], []
    for _, lo, hi in boxes:
        selected = np.all((source.position_m >= lo) & (source.position_m < hi), axis=1)
        w = source.population[selected]
        p = source.position_m[selected]
        centers.append(p[0] + np.average(p - p[0], weights=w, axis=0))
        weights.append(w.sum())
    offsets = (np.array(centers)[:, None, :] + cloud.offsets_m).reshape(-1, 3)
    fractions = (np.array(weights)[:, None] / sum(weights) * cloud.fractions).ravel()
    return offsets, fractions


def study_case(name, max_power):
    case = CASES[name]
    source_offsets = quiet(2**max_power)
    source = species(source_offsets, case)
    obs, grid, nodes, center = observers(case, 2**max_power)
    point, prepared = point_history(case)
    np.savez_compressed(
        OUT / f"{name}_geometry.npz",
        nodes=nodes,
        observer_center=center,
        source_covariance=np.cov(source_offsets.T),
        source_mean=source_offsets.mean(axis=0),
    )
    references = []
    previous = None
    reference = None
    for order in (8, 12, 16, 24, 32, 48):
        offsets, weights = gaussian_rule(order)
        values = integrate(point, prepared, nodes, offsets, weights)
        row = dict(
            order=order,
            points=order**3,
            seconds=values[-1],
            e_change=None if previous is None else relative(values[0], previous[0]),
            cb_change=None if previous is None else relative(values[1], previous[1]),
            qs_change=None if previous is None else relative(values[2], previous[2]),
        )
        references.append(row)
        previous = reference = values
        save(f"{name}_quadrature", references)
        if order >= 24 and all(
            max(r[k] for k in ("e_change", "cb_change", "qs_change")) < 1e-4
            for r in references[-2:]
        ):
            break
    assert (
        max(references[-1][k] for k in ("e_change", "cb_change", "qs_change")) < 1e-4
    ), references
    np.savez_compressed(
        OUT / f"{name}_reference.npz",
        electric=reference[0],
        magnetic=reference[1],
        quasi_e=reference[2],
        quasi_b=reference[3],
    )
    rows = []
    fits = {}
    for k in KS:
        for rule, width in (
            ("fixed", 0.00001e-3),
            ("fixed", 0.08e-3),
            ("fixed", 0.2e-3),
            ("bunch_rms_k", 0.2e-3),
        ):
            start = time.perf_counter()
            config = CorrectionConfig(
                clouds_per_species=k, cloud_width_m=width, cloud_width_rule=rule
            )
            corr = CloudCorrection([source], config)
            fit_s = time.perf_counter() - start
            result = integrate(point, prepared, nodes, *cloud_offsets(corr))
            metrics = measure(result, reference, obs, grid, center)
            metrics.update(
                k=k,
                n=len(source_offsets),
                rule=rule,
                width_mm=corr.smearing.position_sigma_mm,
                subcharges=config.subcharge_count,
                fit_s=fit_s,
                events=len(nodes) * k * config.subcharge_count,
            )
            narrow = result if width == 0.00001e-3 else fits[(k, "fixed", 0.00001e-3)]
            metrics["width_contribution"] = float(
                np.linalg.norm(result[0] - narrow[0]) / np.linalg.norm(reference[0])
            )
            metrics["k_contribution"] = relative(narrow[0], reference[0])
            rows.append(metrics)
            fits[(k, rule, width)] = result
            save(f"{name}_k_width", rows)
    noise = []
    anchor = fits[(64, "bunch_rms_k", 0.2e-3)]
    boxes = frozen_boxes(source, 64)
    anchor_cloud = CloudCorrection(
        [source],
        CorrectionConfig(clouds_per_species=64, cloud_width_rule="bunch_rms_k"),
    )
    for (idx, _, _), (_, fitted_idx) in zip(boxes, anchor_cloud.members):
        np.testing.assert_array_equal(idx, fitted_idx)
    for power in (17, 18, 19, max_power):
        for seed in (20261006, 20261008, 20261009):
            s = species(quiet(2**power, seed), case)
            start = time.perf_counter()
            corr = CloudCorrection(
                [s],
                CorrectionConfig(clouds_per_species=64, cloud_width_rule="bunch_rms_k"),
            )
            fit_s = time.perf_counter() - start
            result = integrate(point, prepared, nodes, *cloud_offsets(corr))
            fixed = integrate(
                point, prepared, nodes, *frozen_fit(s, boxes, anchor_cloud)
            )
            metrics = measure(result, reference, obs, grid, center)
            metrics.update(
                n=2**power,
                seed=seed,
                fit_s=fit_s,
                difference_to_large_n=relative(result[0], anchor[0]),
                fixed_partition_difference=relative(fixed[0], anchor[0]),
            )
            noise.append(metrics)
            save(f"{name}_n", noise)
    subrows = []
    subvalues = []
    for sub in (4, 16, 64):
        corr = CloudCorrection(
            [source],
            CorrectionConfig(
                clouds_per_species=64,
                cloud_width_rule="bunch_rms_k",
                subcharge_count=sub,
            ),
        )
        result = integrate(point, prepared, nodes, *cloud_offsets(corr))
        subvalues.append(result[0])
        subrows.append(
            dict(subcharges=sub, **measure(result, reference, obs, grid, center))
        )
    for row, values in zip(subrows, subvalues):
        row["change_to_64"] = float(
            np.linalg.norm(values - subvalues[-1]) / np.linalg.norm(reference[0])
        )
    save(f"{name}_subcharges", subrows)
    # Independent history refinement (same 3D quadrature, distinct knots).
    history_rows = []
    for steps in (1201, 2401):
        finepoint, fineprepared = point_history(case, steps)
        result = integrate(finepoint, fineprepared, nodes, *gaussian_rule(16))
        history_rows.append(
            dict(
                steps=steps,
                e_difference=relative(result[0], reference[0]),
                cb_difference=relative(result[1], reference[1]),
                seconds=result[-1],
            )
        )
    save(f"{name}_history", history_rows)
    # Reference at off-node particles, separate from K and source N error.
    subset = obs.position_m[:: len(obs.position_m) // 16][:16]
    direct = integrate(point, prepared, subset, *gaussian_rule(references[-1]["order"]))
    gathered = grid.gather(subset - center, reference[0].reshape(*grid.shape, 3))
    save(
        f"{name}_lattice",
        dict(
            e_error=relative(gathered, direct[0]),
            cb_error=relative(
                grid.gather(subset - center, reference[1].reshape(*grid.shape, 3)),
                direct[1],
            ),
            probes=16,
            spacing_mm=0.1,
        ),
    )
    # Verify the rigid translation shortcut against production cloud histories.
    cfg = CorrectionConfig(
        clouds_per_species=64, cloud_width_rule="bunch_rms_k", subcharge_count=4
    )
    production = prescribe(CloudCorrection([source], cfg), case)
    start = time.perf_counter()
    pe, pb = production.direct(nodes)
    translated = integrate(point, prepared, nodes, *cloud_offsets(production))
    save(
        f"{name}_production",
        dict(
            e_error=relative(pe, translated[0]),
            cb_error=relative(pb, translated[1]),
            seconds=time.perf_counter() - start,
            config=asdict(cfg),
        ),
    )
    # Grid accuracy is checked on the continuum's boosted electrostatic part.
    _, displacement, beta = trajectory(case, 1)
    source.position_m += displacement
    source.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
    pics = []
    for mesh in (32, 64, 96):
        solver = ElectrostaticPIC((mesh,) * 3, (0.002,) * 3, 2)
        start = time.perf_counter()
        fields = solver.fields([source])
        e, b = solver.sample(fields, nodes)
        pics.append(
            dict(
                mesh=mesh,
                e_error=relative(e, reference[2]),
                cb_error=relative(b, reference[3]),
                seconds=time.perf_counter() - start,
            )
        )
        save(f"{name}_pic", pics)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--case", choices=list(CASES) + ["all"], default="all")
    parser.add_argument("--max-power", type=int, default=20)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    save(
        "manifest",
        dict(
            base="9e898dc",
            cases=CASES,
            sigma_mm=SIGMA * 1e3,
            quiet_start="scrambled Sobol normal quantiles plus antithetic mirrors",
            n=2**args.max_power,
            quadrature="tensor Gauss-Hermite in 3 fixed lab axes",
            trajectory="beta=b0+change*sin(pi*t/(2*T))**2; analytic position integral",
            source_steps=601,
            radiation_reaction_mode="off",
            refit_every=1,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
    for name in CASES if args.case == "all" else [args.case]:
        study_case(name, args.max_power)


if __name__ == "__main__":
    main()
