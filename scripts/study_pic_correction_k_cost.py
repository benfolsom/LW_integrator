"""Sequential full-versus-compressed K=64, N=2**20 cost controls."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.pic import CloudCorrection, CorrectionConfig, ElectrostaticPIC
from scripts.study_pic_correction_k import (
    CASES,
    quiet,
    species,
    prescribe,
    observers,
    trajectory,
    relative,
)

OUT = Path.home() / "compute-data/pic_correction_k/2_cost"


def timed(corr, nodes):
    start = time.perf_counter()
    fields = corr.direct(nodes)
    return fields, time.perf_counter() - start


def main():
    assert int(Path("/private/tmp/compute_cap").read_text()) >= 2
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, case in CASES.items():
        src = species(quiet(2**20), case)
        offsets = src.position_m.copy()
        cfg = CorrectionConfig(
            clouds_per_species=64, subcharge_count=4, cloud_width_rule="bunch_rms_k"
        )
        corr = prescribe(CloudCorrection([src], cfg), case)
        obs, grid, nodes, center = observers(case, 2**20)
        baseline, cold = timed(corr, nodes)
        baseline, warm = timed(corr, nodes)
        # An explicit disabled configuration must retain every arithmetic bit.
        corr.config = replace(cfg, certified_inertial_skip=False, far_field_ratio=None)
        disabled, _ = timed(corr, nodes)
        assert all(np.array_equal(a, b) for a, b in zip(baseline, disabled))
        for scale in (1, 2, 4, 8):
            _, displacement, _ = trajectory(case, 1)
            probes = nodes + (scale - 1) * (center - displacement)
            corr.config = cfg
            full, seconds = timed(corr, probes)
            for cutoff in (8, 16, 32, 64):
                corr.config = replace(
                    cfg, certified_inertial_skip=True, far_field_ratio=cutoff
                )
                cheap, elapsed = timed(corr, probes)
                rows.append(
                    dict(
                        case=name,
                        scale=scale,
                        cutoff=cutoff,
                        e_error=relative(cheap[0], full[0]),
                        b_error=relative(cheap[1], full[1]),
                        full_s=seconds,
                        cheap_s=elapsed,
                        speedup=seconds / elapsed,
                        **corr.cost_counts,
                    )
                )
                print(rows[-1], flush=True)
        # Warm, complete prescribed steps: N-particle refit, field evaluation,
        # lattice gather, and correction kick. These are not coupled runs.
        endpoints = []
        solver = ElectrostaticPIC()
        for cutoff in (None, 32):
            trial = prescribe(CloudCorrection([src], cfg), case)
            trial.config = replace(
                cfg, certified_inertial_skip=cutoff is not None, far_field_ratio=cutoff
            )
            trial.direct(nodes)
            start = time.perf_counter()
            for step in range(3):
                t, displacement, beta = trajectory(case, 1 + (step + 1) / 600)
                src.position_m[:] = offsets + displacement
                src.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
                trial.accept([src], t)
                e, b = trial.direct(nodes + displacement - trajectory(case, 1)[1])
                local = obs.position_m - center
                eg, bg = (grid.gather(local, f.reshape(*grid.shape, 3)) for f in (e, b))
                solver.push(obs, eg, bg, 1e-13)
            endpoints.append((time.perf_counter() - start) / 3)
            # Restore initial particles before constructing the next history.
            src.position_m[:] = offsets
            src.momentum_mc[:] = np.array(case["b0"]) / np.sqrt(
                1 - np.dot(case["b0"], case["b0"])
            )
        (OUT / f"{name}_timing.json").write_text(
            json.dumps(
                dict(
                    cold_s=cold,
                    warm_s=warm,
                    full_step_s=endpoints[0],
                    cheap_step_s=endpoints[1],
                    speedup=endpoints[0] / endpoints[1],
                ),
                indent=2,
            )
        )
        (OUT / "convergence.json").write_text(json.dumps(rows, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in CASES:
        subset = [r for r in rows if r["case"] == name and r["cutoff"] == 32]
        axes[0].loglog(
            [r["scale"] for r in subset],
            [max(r["e_error"], 1e-16) for r in subset],
            "o-",
            label=name,
        )
        axes[1].plot([r["scale"] for r in subset], [r["speedup"] for r in subset], "o-")
    axes[0].set(
        xlabel="Observer distance multiplier", ylabel="Relative E correction error"
    )
    axes[1].set(
        xlabel="Observer distance multiplier", ylabel="Field evaluation speedup"
    )
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=160)


if __name__ == "__main__":
    main()
