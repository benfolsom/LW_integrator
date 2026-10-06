"""Single-cloud distance/radius convergence of the matched monopole."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.pic import CloudCorrection, CorrectionConfig
from scripts.study_pic_correction_k import CASES, prescribe, species, relative

OUT = Path.home() / "compute-data/pic_correction_k/2_cost"


def main():
    assert int(Path("/private/tmp/compute_cap").read_text()) >= 2
    rows = []
    for name, case in CASES.items():
        cfg = CorrectionConfig(
            clouds_per_species=1, subcharge_count=4, cloud_width_m=0.25e-3 / 4
        )
        corr = prescribe(CloudCorrection([species(np.zeros((1, 3)), case)], cfg), case)
        radius = np.max(np.linalg.norm(corr.offsets_m[0], axis=1))
        for ratio in (4, 8, 16, 32, 64, 128):
            # All three trajectories lie in xz; y is transverse at every knot.
            nodes = corr.position_m + np.array([[0, radius * ratio, 0]])
            corr.config = cfg
            full = corr.direct(nodes)
            corr.config = replace(cfg, far_field_ratio=2)
            cheap = corr.direct(nodes)
            assert corr.cost_counts["compressed"] == 1
            rows.append(
                dict(
                    case=name,
                    ratio=ratio,
                    e_error=relative(cheap[0], full[0]),
                    b_error=relative(cheap[1], full[1]),
                )
            )
            print(rows[-1], flush=True)
    (OUT / "cutoff.json").write_text(json.dumps(rows, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in CASES:
        subset = [r for r in rows if r["case"] == name]
        ax.loglog(
            [r["ratio"] for r in subset],
            [max(r["e_error"], r["b_error"]) for r in subset],
            "o-",
            label=name,
        )
    ax.axhline(0.001, color="gray", linestyle="--", label="0.1% additional error")
    ax.set(
        xlabel="Distance / maximum subcharge radius",
        ylabel="Maximum relative E or B correction error",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "cutoff.png", dpi=160)


if __name__ == "__main__":
    main()
