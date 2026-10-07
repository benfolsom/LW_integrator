"""Serial spatial error budget and close-crossing controls on matched histories.

The exterior Gaussian reference is independent tensor Gauss–Hermite exact LW.
Close crossing reuses the checkpoint's generating PIC trajectory and matched
finite-particle-cloud reference. Radiation reaction is explicitly off.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.pic import CloudCorrection, CorrectionConfig, ElectrostaticPIC
from core.pic.grid import Grid
from core.pic.kernels import C
from scripts.study_pic_correction_k import (
    CASES,
    cloud_offsets,
    gaussian_rule,
    integrate,
    observers,
    point_history,
    prescribe,
    quiet,
    relative,
    species,
    trajectory,
)

OUT = Path.home() / "compute-data/pic_correction_k/7_convergence"


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"
    )


def combined(e, b):
    return np.stack((e, C * b))


def exterior():
    rows = []
    for name, case in CASES.items():
        offsets = quiet(2**14)
        src = species(offsets.copy(), case)
        obs, _, _, center = observers(case, 16)
        point, prepared = point_history(case, 2401)
        reference = None
        controls = []
        for order in (16, 24, 32):
            result = integrate(point, prepared, obs.position_m, *gaussian_rule(order))
            controls.append(
                dict(
                    order=order,
                    seconds=result[-1],
                    change=(
                        None
                        if reference is None
                        else relative(combined(*result[:2]), combined(*reference[:2]))
                    ),
                    quasi_change=(
                        None if reference is None else relative(result[2], reference[2])
                    ),
                )
            )
            reference = result
            save(f"{name}_reference_control", controls)
        assert max(controls[-1]["change"], controls[-1]["quasi_change"]) < 1e-4
        ref = combined(*reference[:2])
        total_ref = combined(reference[0] + reference[2], reference[1] + reference[3])
        np.savez_compressed(
            OUT / f"{name}_reference.npz",
            positions=obs.position_m,
            correction=ref,
            total=total_ref,
        )
        cfg = CorrectionConfig(
            clouds_per_species=64, subcharge_count=4, cloud_width_rule="bunch_rms_k"
        )
        base = CloudCorrection([src], cfg)
        direct = integrate(point, prepared, obs.position_m, *cloud_offsets(base))
        anchor = combined(*direct[:2])
        configs = []
        for k in (16, 32, 64, 128):
            for rule in ("fixed", "bunch_rms_k"):
                configs.append(
                    CorrectionConfig(
                        clouds_per_species=k, subcharge_count=4, cloud_width_rule=rule
                    )
                )
        for sub in (1, 16, 36):
            configs.append(
                CorrectionConfig(
                    clouds_per_species=64,
                    subcharge_count=sub,
                    cloud_width_rule="bunch_rms_k",
                )
            )
        for config in configs:
            tick = time.perf_counter()
            cloud = CloudCorrection([src], config)
            fit = time.perf_counter() - tick
            value = integrate(point, prepared, obs.position_m, *cloud_offsets(cloud))
            field = combined(*value[:2])
            rows.append(
                dict(
                    case=name,
                    axis="representation",
                    config=asdict(config),
                    width_mm=cloud.smearing.position_sigma_mm,
                    error=relative(field, ref),
                    increment=relative(field, anchor),
                    fit_s=fit,
                    field_s=value[-1],
                )
            )
        # Extent is varied at fixed spacing; spacing is varied at fixed extent.
        # Every gather point must remain inside the patch: no clamped extrapolation.
        for extent, shapes in (
            (0.15e-3, (4, 7, 13)),
            (0.30e-3, (7,)),
            (0.60e-3, (13,)),
        ):
            for n in shapes:
                grid = Grid(
                    (n,) * 3, np.full(3, 2 * extent / (n - 1)), np.full(3, -extent)
                )
                local = obs.position_m - center
                assert np.max(np.abs(local)) < extent
                nodes = (
                    center
                    + grid.origin_m
                    + (np.indices(grid.shape).reshape(3, -1).T * grid.spacing_m)
                )
                value = integrate(point, prepared, nodes, *cloud_offsets(base))
                gathered = combined(
                    *(grid.gather(local, f.reshape(*grid.shape, 3)) for f in value[:2])
                )
                rows.append(
                    dict(
                        case=name,
                        axis="lattice",
                        extent_mm=extent * 1e3,
                        n=n,
                        spacing_mm=grid.spacing_m[0] * 1e3,
                        error=relative(gathered, ref),
                        increment=relative(gathered, anchor),
                        field_s=value[-1],
                        events=len(nodes) * 64 * 4,
                    )
                )
        # One public query checks translation and production gathering together.
        public_cfg = CorrectionConfig(
            clouds_per_species=64,
            subcharge_count=4,
            cloud_width_rule="bunch_rms_k",
            lattice_shape=(4,) * 3,
            half_extent_m=(0.15e-3,) * 3,
        )
        production = prescribe(CloudCorrection([src], public_cfg), case, 2401)
        tick = time.perf_counter()
        pe, pb = production.sample(obs)
        query_s = time.perf_counter() - tick
        pe2, pb2 = production.sample(obs)
        repeat_s = time.perf_counter() - tick - query_s
        production_direct = combined(*production.direct(obs.position_m))
        rows.append(
            dict(
                case=name,
                axis="public",
                direct_parity=relative(production_direct, anchor),
                error=relative(combined(pe, pb), ref),
                cold_query_s=query_s,
                warm_query_s=repeat_s,
                repeat_parity=relative(combined(pe2, pb2), combined(pe, pb)),
            )
        )
        _, displacement, beta = trajectory(case, 1)
        src.position_m[:] = offsets + displacement
        src.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
        for mesh in (24, 48, 96):
            for extent in (2e-3, 4e-3):
                solver = ElectrostaticPIC((mesh,) * 3, (extent,) * 3, 2)
                tick = time.perf_counter()
                fields = solver.fields([src])
                e, b = solver.sample(fields, obs.position_m)
                elapsed = time.perf_counter() - tick
                total = combined(e, b) + anchor
                # Population-weighted physical impulse, at a matched small kick.
                dt = 1e-13
                reference_u = solver.push(
                    obs, reference[0] + reference[2], reference[1] + reference[3], dt
                )
                candidate_u = solver.push(obs, e + direct[0], b + direct[1], dt)
                w = obs.population[:, None]
                impulse_error = relative(
                    np.sum(w * (candidate_u - obs.momentum_mc), axis=0),
                    np.sum(w * (reference_u - obs.momentum_mc), axis=0),
                )
                rows.append(
                    dict(
                        case=name,
                        axis="pic",
                        mesh=mesh,
                        extent_mm=extent * 1e3,
                        rest_spacing_mm=2 * extent / mesh * 1e3,
                        quasi_error=relative(combined(e, b), combined(*reference[2:4])),
                        total_error=relative(total, total_ref),
                        impulse_error=impulse_error,
                        build_gather_s=elapsed,
                    )
                )
        save("exterior", rows)
        print(name, "finished", flush=True)
    return rows


def close():
    from scripts import study_pic_correction_k_nearfield as study

    for label, k, width, sub in (
        ("baseline", 8, 0.16e-3, 16),
        ("k4", 4, 0.16e-3, 16),
        ("k16", 16, 0.16e-3, 16),
        ("width08", 8, 0.08e-3, 16),
        ("sub4", 8, 0.16e-3, 4),
        ("sub36", 8, 0.16e-3, 36),
    ):
        study.OUT = OUT / f"close_{label}"
        study.OUT.mkdir(parents=True, exist_ok=True)
        study.study(
            32,
            1201,
            17,
            cloud_config=CorrectionConfig(
                clouds_per_species=k, cloud_width_m=width, subcharge_count=sub
            ),
            near_subcharges=sub,
        )
        print(label, "finished", flush=True)


def close_scaled():
    from scripts import study_pic_correction_k_nearfield as study

    study.OUT = OUT / "close_scaled"
    study.OUT.mkdir(parents=True, exist_ok=True)
    study.study(
        32,
        1201,
        17,
        cloud_config=CorrectionConfig(
            clouds_per_species=8, cloud_width_rule="bunch_rms_k"
        ),
        replay_meshes=(48,),
        replay_cutoffs=(3.2,),
    )


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    rows = json.loads((OUT / "exterior.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = {
        "acceleration": "Acceleration",
        "crossing_electron_source": "Electron source",
        "crossing_proton_source": "Proton source",
    }
    for name, color in zip(CASES, ("tab:blue", "tab:orange", "tab:green")):
        lattice = [
            r
            for r in rows
            if r["case"] == name and r["axis"] == "lattice" and r["extent_mm"] == 0.15
        ]
        axes[0].loglog(
            [r["spacing_mm"] for r in lattice],
            [r["increment"] for r in lattice],
            "o-",
            label=titles[name],
            color=color,
        )
        for rule, style in (("fixed", "--"), ("bunch_rms_k", "-")):
            subset = [
                r
                for r in rows
                if r["case"] == name
                and r["axis"] == "representation"
                and r["config"]["subcharge_count"] == 4
                and r["config"]["cloud_width_rule"] == rule
            ]
            rule_label = "fixed 0.2 mm" if rule == "fixed" else "scaled width"
            axes[1].loglog(
                [r["config"]["clouds_per_species"] for r in subset],
                [r["error"] for r in subset],
                "o" + style,
                label=f"{titles[name]}: {rule_label}",
                color=color,
            )
        subset = [
            r
            for r in rows
            if r["case"] == name and r["axis"] == "pic" and r["extent_mm"] == 4
        ]
        axes[2].loglog(
            [r["rest_spacing_mm"] for r in subset],
            [r["total_error"] for r in subset],
            "o-",
            label=titles[name],
            color=color,
        )
    for ax, x, y in zip(
        axes,
        ("Lattice spacing (mm)", "Cloud count K", "PIC transverse spacing (mm)"),
        (
            "Interpolation increment",
            "Correction relative error",
            "Total LW relative error",
        ),
    ):
        ax.set(xlabel=x, ylabel=y)
        ax.legend(fontsize=6)
    for ax, ticks, labels in (
        (axes[0], [0.025, 0.05, 0.1], ["0.025", "0.05", "0.1"]),
        (axes[1], [16, 32, 64, 128], ["16", "32", "64", "128"]),
        (axes[2], [1 / 12, 1 / 6, 1 / 3], ["0.0833", "0.1667", "0.3333"]),
    ):
        ax.set_xticks(ticks, labels)
        ax.xaxis.set_minor_locator(NullLocator())
    fig.tight_layout()
    fig.savefig(OUT / "spatial_budget.png", dpi=180)
    fig.savefig(OUT / "spatial_budget.pdf")


def temporal():
    from scripts.study_pic_correction_k_predictor import convergence

    save("midpoint_timestep", convergence())


def defaults():
    rows = []
    for name, case in CASES.items():
        src = species(quiet(2**14), case)
        obs, _, _, _ = observers(case, 16)
        ref = np.load(OUT / f"{name}_reference.npz")["correction"]
        cfg = CorrectionConfig()
        corr = prescribe(CloudCorrection([src], cfg), case, 2401)
        tick = time.perf_counter()
        direct = combined(*corr.direct(obs.position_m))
        direct_s = time.perf_counter() - tick
        tick = time.perf_counter()
        gathered = combined(*corr.sample(obs))
        rows.append(
            dict(
                case=name,
                config=asdict(cfg),
                direct_error=relative(direct, ref),
                gathered_error=relative(gathered, ref),
                direct_s=direct_s,
                query_s=time.perf_counter() - tick,
            )
        )
    save("existing_defaults", rows)


def joint():
    """Check the simultaneous candidate settings through both public gathers."""
    rows = []
    for name, case in CASES.items():
        src = species(quiet(2**14), case)
        obs, _, _, _ = observers(case, 16)
        cfg = CorrectionConfig(
            clouds_per_species=64,
            subcharge_count=4,
            cloud_width_rule="bunch_rms_k",
            lattice_shape=(4,) * 3,
            half_extent_m=(0.15e-3,) * 3,
        )
        corr = prescribe(CloudCorrection([src], cfg), case, 2401)
        _, displacement, beta = trajectory(case, 1)
        src.position_m += displacement
        src.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
        solver = ElectrostaticPIC((48,) * 3, (2e-3,) * 3, 2)
        tick = time.perf_counter()
        fields = solver.fields([src])
        e, b = solver.sample(fields, obs.position_m)
        de, db = corr.sample(obs)
        total = combined(e + de, b + db)
        elapsed = time.perf_counter() - tick
        reference = np.load(OUT / f"{name}_reference.npz")["total"]
        dt = 1e-13
        exact = solver.push(obs, reference[0], reference[1] / C, dt)
        candidate = solver.push(obs, e + de, b + db, dt)
        weight = obs.population[:, None]
        rows.append(
            dict(
                case=name,
                config=asdict(cfg),
                mesh=48,
                extent_mm=2,
                total_error=relative(total, reference),
                impulse_error=relative(
                    np.sum(weight * (candidate - obs.momentum_mc), axis=0),
                    np.sum(weight * (exact - obs.momentum_mc), axis=0),
                ),
                field_build_query_s=elapsed,
            )
        )
    save("joint_candidate", rows)


def sampling():
    rows = []
    for name, case in CASES.items():
        obs, _, _, _ = observers(case, 16)
        ref = np.load(OUT / f"{name}_reference.npz")["correction"]
        point, prepared = point_history(case, 2401)
        cfg = CorrectionConfig(
            clouds_per_species=64, subcharge_count=4, cloud_width_rule="bunch_rms_k"
        )
        fields = {}
        for power in (14, 16, 18):
            for seed in (20261006, 20261008, 20261009):
                corr = CloudCorrection([species(quiet(2**power, seed), case)], cfg)
                value = integrate(point, prepared, obs.position_m, *cloud_offsets(corr))
                fields[power, seed] = combined(*value[:2])
                rows.append(
                    dict(
                        case=name,
                        n=2**power,
                        seed=seed,
                        error=relative(fields[power, seed], ref),
                    )
                )
        anchor = fields[18, 20261006]
        for row in rows:
            if row["case"] == name:
                power = int(np.log2(row["n"]))
                row["change_from_largest_n_seed1"] = relative(
                    fields[power, row["seed"]], anchor
                )
        save("sampling", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "exterior",
            "close",
            "close_scaled",
            "temporal",
            "defaults",
            "joint",
            "sampling",
            "plot",
        ),
    )
    args = parser.parse_args()
    assert int(Path("/private/tmp/compute_cap").read_text()) >= 2
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "exterior":
        exterior()
    elif args.mode == "close":
        close()
    elif args.mode == "close_scaled":
        close_scaled()
    elif args.mode == "temporal":
        temporal()
    elif args.mode == "defaults":
        defaults()
    elif args.mode == "joint":
        joint()
    elif args.mode == "sampling":
        sampling()
    else:
        plot()
    save(
        "source_manifest",
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path("scripts/study_pic_correction_k_nearfield.py"),
                *Path("core/pic").glob("*.py"),
            ]
        },
    )
