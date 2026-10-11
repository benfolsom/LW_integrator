"""Assemble auditable current and inherited convergence budgets and figures."""

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.study_pic_correction_k_convergence import OUT, plot, save

BASE = OUT.parent


def inherited():
    paths = [
        BASE / "3_cadence/results.json",
        BASE / "3_cadence/production_timing.json",
        BASE / "3_cadence/timestep_with_impulse.json",
        BASE / "5_nearfield/n32_s1201_q129.json",
        BASE / "5_nearfield/control_n8_s2401_q129.json",
        BASE / "5_nearfield/control_n8_s1201_q401.json",
    ]
    paths.extend(
        BASE / f"{name}_{axis}.json"
        for name in (
            "acceleration",
            "crossing_electron_source",
            "crossing_proton_source",
        )
        for axis in ("history", "k_width", "n", "subcharges")
    )
    evidence = []
    for path in paths:
        data = path.read_bytes()
        evidence.append(
            dict(
                path=str(path),
                sha256=hashlib.sha256(data).hexdigest(),
                evidence=json.loads(data),
            )
        )
    save("inherited_evidence", evidence)
    save(
        "inherited_array_manifest",
        [
            dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in (
                BASE / "5_nearfield/n8_s1201_q129.npz",
                BASE / "5_nearfield/n32_s1201_q129.npz",
                BASE / "5_nearfield/control_n8_s2401_q129.npz",
            )
        ],
    )


def close_summary():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = "n32_s1201_q17"
    anchor = np.load(OUT / f"close_baseline/{name}.npz")
    legacy = np.load(BASE / "5_nearfield/n32_s1201_q129.npz")
    indices = np.array(
        [np.argmin(np.abs(legacy["time_s"] - t)) for t in anchor["time_s"]]
    )
    assert np.allclose(legacy["time_s"][indices], anchor["time_s"], rtol=0, atol=1e-24)
    legacy_changes = []
    for observer in ("protons", "electrons"):
        key = f"m48_r8.0_{observer}_kicks"
        current = anchor[key][:, 2]
        old = legacy[key][indices, 2]
        change = float(np.linalg.norm(current - old) / np.linalg.norm(old))
        assert change < 1e-8
        legacy_changes.append(dict(observer=observer, reference_kick_change=change))
    save("close_checkpoint_replay", legacy_changes)
    rows = []
    labels = ["baseline", "k4", "k16", "width08", "sub4", "sub36"]
    if (OUT / f"close_scaled/{name}.json").exists():
        labels.append("scaled")
    for label in labels:
        data = np.load(OUT / f"close_{label}/{name}.npz")
        meta = json.loads((OUT / f"close_{label}/{name}.json").read_text())
        assert np.array_equal(data["position_m"], anchor["position_m"])
        assert np.array_equal(data["momentum_mc"], anchor["momentum_mc"])
        for observer in ("protons", "electrons"):
            reference = data[f"m48_r8.0_{observer}_kicks"][:, 2]
            baseline = anchor[f"m48_r8.0_{observer}_kicks"][:, 2]
            rows.append(
                dict(
                    variant=label,
                    observer=observer,
                    reference_kick_change=float(
                        np.linalg.norm(reference - baseline) / np.linalg.norm(baseline)
                    ),
                    rows=[r for r in meta["rows"] if r["observer"] == observer],
                    public_replay_error=meta["max_public_replay_error"],
                    pair_closure=meta["max_grid_pair_closure"],
                    finite=meta["minimum_finite_check"],
                    runtime_s=meta["runtime_s"],
                    timing=meta["timing"],
                )
            )
    save("close_summary", rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    titles = {
        "baseline": "K=8, 16 subcharges",
        "k4": "K=4",
        "k16": "K=16",
        "width08": "Coarse width 0.08 mm",
        "sub4": "4 subcharges",
        "sub36": "36 subcharges",
        "scaled": "Scaled coarse width",
    }
    for ax, observer in zip(axes, ("protons", "electrons")):
        for label in labels:
            values = next(
                r["rows"]
                for r in rows
                if r["observer"] == observer and r["variant"] == label
            )
            values = [
                r for r in values if r["mesh"] == 48 and 0 < r["cutoff_mm"] <= 3.2
            ]
            ax.semilogy(
                [r["cutoff_mm"] for r in values],
                [r["kick_error"] for r in values],
                "o-",
                label=titles[label],
            )
        ax.set(
            title="Proton observers" if observer == "protons" else "Electron observers",
            xlabel="Near cutoff (mm)",
            ylabel="Matched reference kick relative error",
        )
        ax.legend(fontsize=7)
    fig.text(
        0.5,
        0.01,
        "Subcharge curves use different references: 4 to 16 changes kicks ~89%; "
        "16 to 36 changes kicks ~167 times the 16-point norm.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "close_budget.png", dpi=180)
    fig.savefig(OUT / "close_budget.pdf")


def source_steps(count=8):
    old = BASE / "5_nearfield"
    if count == 8:
        coarse_path = old / "n8_s1201_q129"
        middle_path = old / "control_n8_s2401_q129"
        fine_path = OUT / "control_n8_s4801_q129"
    else:
        coarse_path = OUT / "close_baseline/n32_s1201_q17"
        middle_path = OUT / "control_n32_s2401_q17"
        fine_path = OUT / "control_n32_s4801_q17"
    coarse = np.load(coarse_path.with_suffix(".npz"))
    middle = np.load(middle_path.with_suffix(".npz"))
    fine = np.load(fine_path.with_suffix(".npz"))
    cdt = json.loads(coarse_path.with_suffix(".json").read_text())["dt_s"]
    mdt = json.loads(middle_path.with_suffix(".json").read_text())["dt_s"]
    fdt = json.loads(fine_path.with_suffix(".json").read_text())["dt_s"]
    assert np.allclose(middle["time_s"], fine["time_s"], rtol=0, atol=1e-24)
    rows = []
    for oi, observer in enumerate(("protons", "electrons")):
        c = coarse[f"m48_r8.0_{observer}_kicks"][:, 2] / cdt
        m = middle["reference_kicks"][:, oi] / mdt
        f = fine["reference_kicks"][:, oi] / fdt
        first = np.linalg.norm(m - c) / np.linalg.norm(m)
        second = np.linalg.norm(f - m) / np.linalg.norm(f)
        rows.append(
            dict(
                observer=observer,
                count_per_bunch=count,
                dt_fs=[cdt * 1e15, mdt * 1e15, fdt * 1e15],
                first_change=float(first),
                second_change=float(second),
                observed_order=float(np.log2(first / second)),
                impulse_second_change=float(
                    np.linalg.norm(np.trapezoid(f - m, fine["time_s"], axis=0))
                    / np.linalg.norm(np.trapezoid(f, fine["time_s"], axis=0))
                ),
            )
        )
    save("close_source_timestep" if count == 8 else "close32_source_timestep", rows)


def exterior_budget():
    """Triangle bound for spatial errors, all normalized to the same total LW."""
    rows = json.loads((OUT / "exterior.json").read_text())
    budgets = []
    for name in ("acceleration", "crossing_electron_source", "crossing_proton_source"):
        ref = np.load(OUT / f"{name}_reference.npz")
        ratio = np.linalg.norm(ref["correction"]) / np.linalg.norm(ref["total"])
        qs_ratio = np.linalg.norm(ref["total"] - ref["correction"]) / np.linalg.norm(
            ref["total"]
        )
        rep = next(
            r
            for r in rows
            if r["case"] == name
            and r["axis"] == "representation"
            and r["config"]["clouds_per_species"] == 64
            and r["config"]["cloud_width_rule"] == "bunch_rms_k"
            and r["config"]["subcharge_count"] == 4
        )
        lattice = next(
            r
            for r in rows
            if r["case"] == name and r["axis"] == "lattice" and r["n"] == 4
        )
        pic = next(
            r
            for r in rows
            if r["case"] == name
            and r["axis"] == "pic"
            and r["mesh"] == 48
            and r["extent_mm"] == 2
        )
        representation = rep["error"] * ratio
        interpolation = lattice["increment"] * (1 + rep["error"]) * ratio
        grid = pic["quasi_error"] * qs_ratio
        budgets.append(
            dict(
                case=name,
                correction_to_total_ratio=float(ratio),
                representation_total_scale=float(representation),
                interpolation_total_scale_bound=float(interpolation),
                pic_total_scale=float(grid),
                spatial_triangle_bound=float(representation + interpolation + grid),
            )
        )
    save("exterior_budget", budgets)


def temporal_plot():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data = json.loads((OUT / "midpoint_timestep.json").read_text())
    for mode in ("accepted", "midpoint"):
        rows = [r for r in data["rows"] if r["mode"] == mode]
        axes[0].loglog(
            [r["dt_s"] * 1e12 for r in rows],
            [r["integrated_field_error"] for r in rows],
            "o-",
            label=mode,
        )
    rows = json.loads((OUT / "close_source_timestep.json").read_text())
    if (OUT / "close32_source_timestep.json").exists():
        rows.extend(json.loads((OUT / "close32_source_timestep.json").read_text()))
    for row in rows:
        axes[1].loglog(
            row["dt_fs"][1:],
            [row["first_change"], row["second_change"]],
            "o-",
            label=f'N={row["count_per_bunch"]} {row["observer"]}',
        )
    axes[0].set(
        xlabel="Lab timestep (ps)",
        ylabel="Integrated correction error",
        title="Prescribed acceleration",
    )
    axes[1].set(
        xlabel="Source timestep (fs)",
        ylabel="Successive reference force change",
        title="Close crossing",
    )
    for ax in axes:
        ax.legend()
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    axes[0].set_xticks(
        [0.09375, 0.1875, 0.375, 0.75], ["0.09375", "0.1875", "0.375", "0.75"]
    )
    axes[1].set_xticks([6.6081235415, 13.216247083], ["6.61", "13.22"])
    for ax in axes:
        ax.xaxis.set_minor_locator(NullLocator())
    fig.tight_layout()
    fig.savefig(OUT / "temporal_budget.png", dpi=180)
    fig.savefig(OUT / "temporal_budget.pdf")


if __name__ == "__main__":
    assert int(Path("/private/tmp/compute_cap").read_text()) >= 2
    inherited()
    plot()
    close_summary()
    source_steps()
    if (OUT / "control_n32_s4801_q17.json").exists():
        source_steps(32)
    exterior_budget()
    temporal_plot()
