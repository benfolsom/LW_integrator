"""Create compact evidence tables and standalone figures from study artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.study_pic_correction_k import CASES


def read(root, name):
    return json.loads((root / f"{name}.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for col, name in enumerate(CASES):
        rows = read(out, f"{name}_k_width")
        noise = read(out, f"{name}_n")
        orders = []
        for label, selected in (
            ("Narrow", [r for r in rows if r["width_mm"] < 0.001]),
            (
                "Fixed 0.08 mm",
                [
                    r
                    for r in rows
                    if r["rule"] == "fixed" and 0.079 < r["width_mm"] < 0.081
                ],
            ),
            (
                "Fixed 0.2 mm",
                [
                    r
                    for r in rows
                    if r["rule"] == "fixed" and 0.199 < r["width_mm"] < 0.201
                ],
            ),
            ("RMS / K^(1/3)", [r for r in rows if r["rule"] == "bunch_rms_k"]),
        ):
            ks = np.array([r["k"] for r in selected])
            errors = np.array([r["e_error"] for r in selected])
            axes[0, col].loglog(ks, errors * 100, "o-", label=label)
            orders.append(
                dict(
                    rule=label,
                    k_orders=(-np.diff(np.log(errors)) / np.diff(np.log(ks))).tolist(),
                    order_32_to_256=float(
                        -np.polyfit(np.log(ks[-4:]), np.log(errors[-4:]), 1)[0]
                    ),
                )
            )
        for label, key in (
            ("Production re-fit", "difference_to_large_n"),
            ("Frozen partition moments", "fixed_partition_difference"),
        ):
            ns = sorted(set(r["n"] for r in noise))
            means = [
                float(np.sqrt(np.mean([r[key] ** 2 for r in noise if r["n"] == n])))
                for n in ns
            ]
            axes[1, col].loglog(ns, np.array(means) * 100, "o-", label=label)
        selected = [r for r in rows if r["rule"] == "bunch_rms_k"]
        axes2[col].loglog(
            [r["k"] for r in selected],
            [100 * r["e_error"] for r in selected],
            "o-",
            label="Node E error",
        )
        axes2[col].loglog(
            [r["k"] for r in selected],
            [100 * r["kick_error"] for r in selected],
            "s--",
            label="Particle kick error",
        )
        axes2[col].loglog(
            [r["k"] for r in selected],
            [100 * r["width_contribution"] for r in selected],
            "^:",
            label="Width increment",
        )
        title = name.replace("_", " ")
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("Clouds per species K")
        axes[0, col].set_ylabel("Correction E error (%)")
        axes[1, col].set_xlabel("PIC particles per species N")
        axes[1, col].set_ylabel("Fit change / reference E (%)")
        axes2[col].set_title(title)
        axes2[col].set_xlabel("Clouds per species K")
        axes2[col].set_ylabel("Error or increment (%)")
        for ax in (axes[0, col], axes[1, col], axes2[col]):
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
        summary[name] = dict(
            orders=orders,
            recommended=[r for r in selected if r["k"] == 64][0],
            quadrature=read(out, f"{name}_quadrature"),
            history=read(out, f"{name}_history"),
            pic=read(out, f"{name}_pic"),
            lattice=read(out, f"{name}_lattice"),
            production=read(out, f"{name}_production"),
            subcharges=read(out, f"{name}_subcharges"),
        )
    for figure, name in ((fig, "convergence"), (fig2, "kicks_and_width")):
        figure.savefig(out / f"{name}.png", dpi=180)
        figure.savefig(out / f"{name}.pdf")
    cold = read(out / "profile_cold_head", "profiles")
    warm = read(out / "profile_warm", "profiles")
    profiles = []
    for a, b in zip(cold, warm):
        assert (a["case"], a["k"], a["nodes"]) == (b["case"], b["k"], b["nodes"])
        filename = f'{a["case"]}_k{a["k"]}_n{a["nodes"]}.npz'
        with np.load(out / "profile_cold_head" / filename) as aa, np.load(
            out / "profile_warm" / filename
        ) as bb:
            assert aa.files == bb.files
            for key in aa.files:
                assert aa[key].dtype == bb[key].dtype
                assert aa[key].tobytes() == bb[key].tobytes(), (filename, key)
        profiles.append(
            dict(
                case=a["case"],
                k=a["k"],
                nodes=a["nodes"],
                cold_s_step=a["elapsed_s"] / a["steps"],
                warm_s_step=b["elapsed_s"] / b["steps"],
                speedup=a["elapsed_s"] / b["elapsed_s"],
                byte_parity=True,
                hit_rate=b["warm_start"]["hits"] / b["warm_start"]["proposals"],
                cold_stages={k: v / a["steps"] for k, v in a["stages_s"].items()},
                warm_stages={k: v / b["steps"] for k, v in b["stages_s"].items()},
                warm_counts=b["warm_start"],
            )
        )
    summary["profiles"] = profiles
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
