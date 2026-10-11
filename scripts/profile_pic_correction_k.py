"""Bounded synthetic correction profiles; outputs stay outside the checkout."""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict
import json
from pathlib import Path
import pstats
import time
import sys
import subprocess
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core import retarded_fields as rf  # noqa: E402
from core.pic import (  # noqa: E402
    CloudCorrection,
    CorrectionConfig,
    ElectrostaticPIC,
    Species,
)
from core.pic.correction import correction_ledger  # noqa: E402
from core.pic.kernels import C  # noqa: E402
from scripts.validate_pic_correction import accelerated_source, Q, MP, ME  # noqa: E402


class StageTimers:
    """Exclusive wall-clock stages, including opaque compiled kernels."""

    def __init__(self):
        self.stack = []
        self.seconds = {}
        self.calls = {}
        from core import resolved_charge_batch
        from core.pic import correction_fields

        self.patch(rf, "_prepare_history", "history_preparation")
        self.patch(rf, "_solve_retarded_sample", "retarded_roots_and_samples")
        self.patch(
            resolved_charge_batch,
            "resolved_charge_jets_strict_serial",
            "field_evaluation",
        )
        self.patch(
            correction_fields, "resolved_charge_jets_strict_serial", "field_evaluation"
        )
        self.patch(correction_fields, "node_segment_hints", "batch_bracket_proposals")
        self.patch(correction_fields.NodeWarmStart, "propose", "warm_proposals")
        self.patch(CloudCorrection, "quasi_static", "quasi_static")
        self.patch(CloudCorrection, "accept", "cloud_accept_and_refit")
        from core.pic.grid import Grid

        self.patch(Grid, "gather", "gather")
        for method in ("fields", "sample", "push"):
            self.patch(ElectrostaticPIC, method, "pic_fields_and_push")

    def patch(self, owner, name, stage):
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            frame = [0.0]
            self.stack.append(frame)
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                self.stack.pop()
                self.seconds[stage] = self.seconds.get(stage, 0.0) + elapsed - frame[0]
                self.calls[stage] = self.calls.get(stage, 0) + 1
                if self.stack:
                    self.stack[-1][0] += elapsed

        setattr(owner, name, measured)

    def reset(self):
        self.seconds.clear()
        self.calls.clear()


def cases(k, nodes):
    source, correction = accelerated_source(
        max(8, k), k=k, steps=301, width=CorrectionConfig().cloud_width_m
    )
    correction.config = CorrectionConfig(
        clouds_per_species=k, lattice_shape=(nodes,) * 3
    )
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    observer = Species(
        "probes",
        np.mean(source.position_m, axis=0) + [0.003, 0.002, 0] + corners * 0.12e-3,
        np.tile([0, 0, 0.7], (8, 1)),
        Q,
        MP,
        np.full(8, 1e5),
    )
    yield "acceleration", [source], correction, [observer], 1e-13
    # Same stage 2 energies, populations, and symmetric head/tail geometry.
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    offsets = corners * [0.00065, 0.00065, 0.00008]
    if k > 8:
        offsets = np.concatenate([offsets + [i * 1e-6, 0, 0] for i in range(k // 8)])
    bp, be = np.sqrt(1 - (1 + 216 / 938.27208816) ** -2), np.sqrt(0.99)
    species = [
        Species(
            name,
            offsets + [0, 0, z],
            np.tile([0, 0, beta / np.sqrt(1 - beta**2)], (len(offsets), 1)),
            charge,
            mass,
            np.full(len(offsets), total / (len(offsets) * Q)),
        )
        for name, z, beta, charge, mass, total in (
            ("protons", -0.005, bp, Q, MP, 100e-12),
            ("electrons", 0.005, -be, -Q, ME, 20e-12),
        )
    ]
    correction = CloudCorrection(
        species, CorrectionConfig(clouds_per_species=k, lattice_shape=(nodes,) * 3)
    )
    solver = ElectrostaticPIC((24,) * 3, (0.004,) * 3, 2)
    dt = 0.01 / ((bp + be) * C * 800)
    for step in range(795):
        pic_step(solver, species, dt)
        correction.accept(species, (step + 1) * dt)
    yield "crossing", species, correction, species, dt


def pic_step(solver, species, dt, correction=None):
    for s in species:
        s.position_m += 0.5 * dt * s.velocity_m_s
    fields = solver.fields(species)
    updated = []
    values = []
    for s in species:
        e, b = solver.sample(fields, s.position_m)
        if correction is not None:
            de, db = correction.sample(s)
            values.extend((de, db))
            baseline = solver.push(s, e, b, dt)
            pushed = solver.push(s, e + de, b + db, dt)
            correction_ledger(s, baseline, pushed, de, db, dt)
            updated.append(pushed)
        else:
            updated.append(solver.push(s, e, b, dt))
    for s, u in zip(species, updated):
        s.momentum_mc = u
        s.position_m += 0.5 * dt * s.velocity_m_s
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="8:5,32:9")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--coupled-crossing", action="store_true")
    parser.add_argument("--instrument", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timers = StageTimers() if args.instrument else None
    if args.reference:
        import core.pic.correction as correction_module

        baseline = types.ModuleType("core.pic._correction_fields_9e898dc")
        baseline.__package__ = "core.pic"
        code = subprocess.check_output(
            ["git", "show", "9e898dc:core/pic/correction_fields.py"], cwd=ROOT
        )
        (args.output / "correction_fields_9e898dc.py").write_bytes(code)
        exec(compile(code, "correction_fields_9e898dc.py", "exec"), baseline.__dict__)
        if timers:
            timers.patch(baseline, "node_segment_hints", "batch_bracket_proposals")

        def original_fields(history, time_ns, positions, **kwargs):
            # Development 9e898dc node batching, without any warm proposal.
            return baseline.correction_node_fields(history, time_ns, positions)

        correction_module.correction_node_fields = original_fields
    rows = []
    for k, nodes in (map(int, pair.split(":")) for pair in args.sizes.split(",")):
        for name, species, corr, observers, dt in cases(k, nodes):
            print("warming", name, k, nodes, flush=True)
            # Exclude compilation and initial full history preparation.
            for obs in observers:
                corr.sample(obs)
            # Warm all PIC kernel signatures without mutating the measured state.
            solver = ElectrostaticPIC((24,) * 3, (0.004,) * 3, 2)
            fields = solver.fields(species)
            for s in species:
                solver.push(s, *solver.sample(fields, s.position_m), dt)
            before = asdict(rf._CHARGE_PREPARED_HISTORY_CACHE.stats())
            caches = [corr._direct_warm_start] + [
                cache for _, cache in corr._lattice_warm_starts
            ]
            for cache in caches:
                cache.proposals = cache.hits = cache.inertial_events = 0
                cache.batch_fallbacks = 0
            if timers:
                timers.reset()
            prof = cProfile.Profile()
            values = []
            start = time.perf_counter()
            if not args.plain:
                prof.enable()
            for step in range(args.steps):
                solver = ElectrostaticPIC((24,) * 3, (0.004,) * 3, 2)
                if name == "crossing" and args.coupled_crossing:
                    values.extend(pic_step(solver, species, dt, corr))
                else:
                    for obs in observers:
                        values.extend(corr.sample(obs))
                    pic_step(solver, species, dt)
                corr.accept(species, corr.time_s + dt)
                if args.instrument:
                    # Charge each measured step for its translated history tail,
                    # including the final step rather than leaving it untimed.
                    from core.exact_source_cloud import exact_cloud_history

                    rf._prepare_history(
                        exact_cloud_history(
                            corr.builder.build_current(), corr.smearing
                        ),
                        (),
                    )
                print("step", name, k, step + 1, flush=True)
            prof.disable()
            elapsed = time.perf_counter() - start
            path = args.output / f"{name}_k{k}_n{nodes}"
            if not args.plain:
                prof.dump_stats(str(path) + ".prof")
                with Path(str(path) + ".txt").open("w") as stream:
                    pstats.Stats(prof, stream=stream).sort_stats(
                        "cumulative"
                    ).print_stats(65)
            particles = {
                f"species_{i}_{quantity}": getattr(s, quantity)
                for i, s in enumerate(species)
                for quantity in ("position_m", "momentum_mc")
            }
            np.savez_compressed(str(path) + ".npz", *values, **particles)
            after = asdict(rf._CHARGE_PREPARED_HISTORY_CACHE.stats())
            timings = {}
            if not args.plain:
                stats = pstats.Stats(prof)
                for (_, _, function), (cc, nc, tt, ct, _) in stats.stats.items():
                    timings[function] = dict(calls=nc, self_s=tt, cumulative_s=ct)
            row = dict(
                case=name,
                k=k,
                nodes=nodes,
                steps=args.steps,
                observer_species=len(observers),
                source_clouds=len(corr.members),
                correction_config=asdict(corr.config),
                actual_cloud_width_mm=corr.smearing.position_sigma_mm,
                source_observer_events_per_step=nodes**3
                * len(corr.members)
                * 16
                * len(observers),
                elapsed_s=elapsed,
                coupled_crossing=args.coupled_crossing,
                functions=timings,
                stages_s=None if timers is None else dict(timers.seconds),
                stage_calls=None if timers is None else dict(timers.calls),
                cache_delta={key: after[key] - before[key] for key in before},
                warm_start=dict(
                    proposals=sum(cache.proposals for cache in caches),
                    hits=sum(cache.hits for cache in caches),
                    inertial_events=sum(cache.inertial_events for cache in caches),
                    batch_fallbacks=sum(cache.batch_fallbacks for cache in caches),
                ),
            )
            rows.append(row)
            (args.output / "profiles.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
            print(name, k, nodes, round(elapsed, 3), row["cache_delta"], flush=True)


if __name__ == "__main__":
    main()
