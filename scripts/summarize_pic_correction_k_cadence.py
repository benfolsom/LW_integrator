"""Render cadence artifacts and explicit one-percent acceptance tables."""

import json
from pathlib import Path

OUT = Path.home() / "compute-data/pic_correction_k/3_cadence"


def main():
    from study_pic_correction_k_cadence import plot, TITLES

    rows = json.loads((OUT / "results.json").read_text())
    timings = json.loads((OUT / "production_timing.json").read_text())
    plot(rows, timings)
    import numpy as np

    timestep = json.loads((OUT / "timestep.json").read_text())
    for name in dict.fromkeys(r["case"] for r in rows):
        reference = None
        for divisions in (128, 64, 32):
            with np.load(OUT / f"{name}_{divisions}.npz") as data:
                fields, kicks = data["fields"], data["kicks"]
                assert np.all(np.isfinite(fields)) and np.all(np.isfinite(kicks))
                # Endpoint-weighted, population-summed prescribed impulses.
                impulse = kicks[1:-1].sum(axis=(0, 1)) + 0.5 * (
                    kicks[0].sum(axis=0) + kicks[-1].sum(axis=0)
                )
                if reference is None:
                    reference = impulse
                else:
                    row = next(
                        r
                        for r in timestep
                        if r["case"] == name and r["divisions"] == divisions
                    )
                    row["impulse_error"] = float(
                        np.linalg.norm(impulse - reference) / np.linalg.norm(reference)
                    )
    (OUT / "timestep_with_impulse.json").write_text(json.dumps(timestep, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(r["case"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, name in zip(axes, names):
        for mode in ("linear", "hold", "extrapolate"):
            subset = [
                r
                for r in rows
                if r["case"] == name
                and r["mode"] == mode
                and r["divisions"] == 128
                and r["n"] > 1
            ]
            axis.semilogy(
                [r["n"] for r in subset],
                [100 * r["e_error"] for r in subset],
                "o-",
                label=mode,
            )
        axis.axhline(1, color="gray", linestyle="--", label="1% budget")
        axis.set(
            title=TITLES[name],
            xlabel="Evaluation cadence n",
            ylabel="E cadence error (%)",
        )
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_n.png", dpi=160)
    text = []
    for name in names:
        text.extend(
            [
                f"### {TITLES[name]}",
                "",
                "| dt, ps | Mode | n=2 E / cB, % | n=4 E / cB, % | n=8 E / cB, % |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for divisions in (32, 64, 128):
            for mode in ("linear", "hold", "extrapolate"):
                subset = [
                    r
                    for r in rows
                    if r["case"] == name
                    and r["mode"] == mode
                    and r["divisions"] == divisions
                    and r["n"] > 1
                ]
                values = " | ".join(
                    f'{100*r["e_error"]:.4g} / {100*r["cb_error"]:.4g}' for r in subset
                )
                text.append(f'| {subset[0]["dt_s"]*1e12:g} | {mode} | {values} |')
        text.extend(
            [
                "",
                "Finest-timestep kick, impulse, and peak combined field errors:",
                "",
                "| Mode | n | Kick, % | Summed impulse, % | Peak combined field, % | Meets 1% in all metrics |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for r in rows:
            if r["case"] != name or r["divisions"] != 128 or r["n"] == 1:
                continue
            metrics = (
                "e_error",
                "cb_error",
                "kick_error",
                "impulse_error",
                "peak_error",
            )
            safe = all(r[key] <= 0.01 for key in metrics)
            text.append(
                f'| {r["mode"]} | {r["n"]} | {100*r["kick_error"]:.4g} | '
                f'{100*r["impulse_error"]:.4g} | {100*r["peak_error"]:.4g} | {"yes" if safe else "no"} |'
            )
        text.append("")
    (OUT / "tables.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
