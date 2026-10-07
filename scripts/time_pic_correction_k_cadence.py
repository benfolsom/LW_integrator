"""Measured million-particle correction steps, sequential cadence timings."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.pic import CloudCorrection, CorrectionConfig, ElectrostaticPIC
from core.pic.kernels import C
from scripts.study_pic_correction_k import (
    CASES,
    quiet,
    species,
    trajectory,
    observers,
    prescribe,
)

OUT = Path.home() / "compute-data/pic_correction_k/3_cadence"


def main():
    assert int(Path("/private/tmp/compute_cap").read_text()) >= 2
    rows = []
    for name, case in CASES.items():
        offsets = quiet(2**20)
        obs, _, _, center = observers(case, 2**20)
        observer_offsets = obs.position_m - center
        dt = (12e-12 if name == "acceleration" else 24e-12) / 128
        cfg = CorrectionConfig(
            clouds_per_species=64,
            subcharge_count=4,
            cloud_width_rule="bunch_rms_k",
            lattice_shape=(4,) * 3,
            half_extent_m=(0.15e-3,) * 3,
        )
        solver = ElectrostaticPIC()
        for n in (1, 2, 4, 8):
            src = species(offsets.copy(), case)
            corr = prescribe(
                CloudCorrection([src], replace(cfg, evaluation_every=n)), case
            )
            _, displacement, beta = trajectory(case, 1)
            src.position_m[:] = offsets + displacement
            src.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
            obs.position_m[:] = observer_offsets + center
            corr.sample(obs)  # Exclude first field preparation from warm timings.
            start = time.perf_counter()
            for j in range(1, 17):
                t = case["duration"] + j * dt
                _, displacement, beta = trajectory(case, t / case["duration"])
                src.position_m[:] = offsets + displacement
                src.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
                corr.accept([src], t)
                obs.position_m[:] = (
                    observer_offsets
                    + center
                    + np.array([0, 0, case["observer_beta"] * C * j * dt])
                )
                e, b = corr.sample(obs)
                solver.push(obs, e, b, dt)
            seconds = (time.perf_counter() - start) / 16
            assert corr.refits == 17
            row = dict(
                case=name,
                n=n,
                dt_s=dt,
                step_s=seconds,
                steps=16,
                particles_per_species=2**20,
                refits=corr.refits,
            )
            rows.append(row)
            print(row, flush=True)
            (OUT / "production_timing.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
