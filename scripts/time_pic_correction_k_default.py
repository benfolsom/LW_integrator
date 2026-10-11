"""Native correction cost at large N, including production moment refits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from core.pic import CloudCorrection, CorrectionConfig, ElectrostaticPIC
from scripts.profile_pic_correction_k import StageTimers
from scripts.study_pic_correction_k import (
    CASES,
    quiet,
    species,
    prescribe,
    trajectory,
    observers,
    measure,
    relative,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--subcharges", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timers = StageTimers()
    rows = []
    for name, case in CASES.items():
        source = species(quiet(2**20), case)
        initial_offsets = source.position_m.copy()
        start = time.perf_counter()
        cfg = CorrectionConfig(
            clouds_per_species=args.k,
            cloud_width_rule="bunch_rms_k",
            subcharge_count=args.subcharges,
        )
        corr = prescribe(CloudCorrection([source], cfg), case)
        setup = time.perf_counter() - start
        obs, grid, nodes, center = observers(case, 2**20)
        start = time.perf_counter()
        electric, magnetic = corr.direct(nodes)
        cold = time.perf_counter() - start
        with np.load(args.output.parent / f"{name}_reference.npz") as reference:
            re, rb = reference["electric"], reference["magnetic"]
            qe, qb = reference["quasi_e"], reference["quasi_b"]
            metrics = measure((electric, magnetic, cold), (re, rb), obs, grid, center)
        _, displacement, current_beta = trajectory(case, 1)
        source.position_m[:] = initial_offsets + displacement
        source.momentum_mc[:] = current_beta / np.sqrt(1 - current_beta @ current_beta)
        # Overlapping source grids keep every observer on the ordinary gather
        # path; the earlier exterior-node controls use gather_open instead.
        solver = ElectrostaticPIC((128,) * 3, (0.006,) * 3, 2)
        pic_start = time.perf_counter()
        pic_fields = solver.fields([source])
        inside = all(
            np.all(f.grid.inside(f.grid.coordinates(f.rest_positions(obs.position_m))))
            for f in pic_fields
        )
        assert inside
        pe, pb = solver.sample(pic_fields, nodes)
        particle_e, particle_b = solver.sample(pic_fields, obs.position_m)
        gathered = [
            grid.gather(obs.position_m - center, f.reshape(*grid.shape, 3))
            for f in (electric, magnetic, re + qe, rb + qb)
        ]
        reference_kick = (
            solver.push(obs, gathered[2], gathered[3], 1e-13) - obs.momentum_mc
        )
        pic_kick = solver.push(obs, particle_e, particle_b, 1e-13) - obs.momentum_mc
        corrected_kick = (
            solver.push(obs, particle_e + gathered[0], particle_b + gathered[1], 1e-13)
            - obs.momentum_mc
        )
        pic_metrics = dict(
            mesh=128,
            half_extent_mm=6,
            all_observers_inside=inside,
            qs_e_error=relative(pe, qe),
            qs_cb_error=relative(pb, qb),
            pic_kick_error=relative(pic_kick, reference_kick),
            corrected_kick_error=relative(corrected_kick, reference_kick),
            seconds=time.perf_counter() - pic_start,
        )
        timers.reset()
        corr._direct_warm_start.proposals = corr._direct_warm_start.hits = (
            corr._direct_warm_start.inertial_events
        ) = 0
        start = time.perf_counter()
        for step in range(3):
            fraction = 1 + (step + 1) / 600
            t, position, beta = trajectory(case, fraction)
            # Preserve initial offsets independently of absolute source time.
            source.position_m[:] = initial_offsets + position
            source.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
            corr.accept([source], t)
            corr.direct(nodes + position - trajectory(case, 1)[1])
        elapsed = time.perf_counter() - start
        cache = corr._direct_warm_start
        rows.append(
            dict(
                case=name,
                n=len(source.position_m),
                config=asdict(cfg),
                **metrics,
                pic=pic_metrics,
                setup_s=setup,
                first_direct_s=cold,
                seconds_per_step=elapsed / 3,
                stages_s_step={key: value / 3 for key, value in timers.seconds.items()},
                proposals=cache.proposals,
                hits=cache.hits,
                events_per_step=len(nodes) * args.k * args.subcharges,
            )
        )
        (args.output / "timings.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(name, rows[-1], flush=True)


if __name__ == "__main__":
    main()
