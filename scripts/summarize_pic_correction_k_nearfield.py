"""Render external near-field artifacts and concise numerical tables."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path.home() / "compute-data/pic_correction_k/5_nearfield"


def main():
    paths = sorted(OUT.glob("n*_s1201_q129.json"))
    data = [json.loads(p.read_text()) for p in paths]
    fig, axes = plt.subplots(len(data), 2, figsize=(10, 3.6 * len(data)), squeeze=False)
    summary = []
    for row_axes, d, path in zip(axes, data, paths):
        for ax, observer in zip(row_axes, ("protons", "electrons")):
            for mesh in (24, 48, 96):
                rows = [
                    r
                    for r in d["rows"]
                    if r["mesh"] == mesh
                    and r["observer"] == observer
                    and r["cutoff_mm"] > 0
                ]
                ax.semilogy(
                    [r["cutoff_mm"] for r in rows],
                    [100 * r["kick_error"] for r in rows],
                    "o-",
                    label=f"{mesh}³ grid",
                )
            ax.set(
                title=f'{d["count_per_bunch"]}+{d["count_per_bunch"]}: {observer}',
                xlabel="Cutoff radius, mm",
                ylabel="Matched kick error, %",
            )
            ax.axhline(1, color="gray", linestyle=":", linewidth=1)
            ax.legend()
            ax.grid(alpha=0.2)
        print(
            path.name,
            "runtime",
            d["runtime_s"],
            "pair closure",
            d["max_grid_pair_closure"],
            "public replay",
            d["max_public_replay_error"],
        )
        for r in d["rows"]:
            if r["mesh"] == 48:
                print(
                    r["observer"],
                    r["cutoff_mm"],
                    "kick %",
                    100 * r["kick_error"],
                    "impulse %",
                    100 * r["impulse_error"],
                )
        summary.append(
            dict(
                artifact=path.name,
                max_pair_closure=d["max_grid_pair_closure"],
                max_replay_error=d["max_public_replay_error"],
                finite=d["minimum_finite_check"],
            )
        )
    fig.tight_layout()
    fig.savefig(OUT / "cutoff_grid_convergence.png", dpi=170)
    fig.savefig(OUT / "cutoff_grid_convergence.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for d in data:
        replay = json.loads(
            (OUT / f'n{d["count_per_bunch"]}_s1201_q3.json').read_text()
        )
        d = replay
        mid = sorted({t["step"] for t in d["timing"]})[1]
        timings = [t for t in d["timing"] if t["step"] == mid and t["mesh"] == 48]
        for observer, ax in zip(("protons", "electrons"), axes):
            rows = [t for t in timings if t["observer"] == observer]
            ax.plot(
                [t["cutoff_mm"] for t in rows],
                [t["field_query_s"] for t in rows],
                "o-",
                label=f'{d["count_per_bunch"]}+{d["count_per_bunch"]}',
            )
            ax.set(
                title=observer,
                xlabel="Cutoff radius, mm",
                ylabel="Complete additive field query, s",
            )
            ax.legend()
            ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "query_cost.png", dpi=170)
    fig.savefig(OUT / "query_cost.pdf")
    plt.close(fig)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    parity = []
    for d, path in zip(data, paths):
        count = d["count_per_bunch"]
        old = np.load(path.with_suffix(".npz"))
        replay_path = OUT / f"n{count}_s1201_q3"
        new = np.load(replay_path.with_suffix(".npz"))
        indices = [int(np.argmin(np.abs(old["time_s"] - t))) for t in new["time_s"]]
        errors = []
        for key in new.files:
            if key.endswith("_kicks"):
                a, b = old[key][indices], new[key]
                errors.append(float(np.linalg.norm(a - b) / np.linalg.norm(a)))
        meta = json.loads(replay_path.with_suffix(".json").read_text())
        parity.append(
            dict(
                count_per_bunch=count,
                max_original_scan_kick_change=max(errors),
                max_final_grid_closure=meta["max_grid_pair_closure"],
                max_final_public_replay_error=meta["max_public_replay_error"],
            )
        )
    (OUT / "final_replay_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    print("Final replay parity", parity)


if __name__ == "__main__":
    main()
