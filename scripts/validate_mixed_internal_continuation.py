"""Compare short adaptive continuations of an existing many-particle history.

This deliberately forks numerical settings for a diagnostic, not a production
checkpoint resume. Existing history stays intact. Reaction totals are reset so
only newly accumulated contributions enter the comparison.
Run from the checkout with PYTHONPATH=.
"""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from core.momentum_center_particles import advance_particles
from core.nonlinear_pair_adaptive import _zero_ledger
from core.pair_step_error import InternalStepSettings
from core.step_doubling import ErrorScale
from lw_integrator.nonlinear_pair import write_checkpoint
from scripts.compare_multiparticle_postarrival import relative_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-ns", type=float, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    base = json.loads(args.checkpoint.read_text())
    if base.get("pseudogrid_reduced", {}).get("mode") != "passive_rk3":
        parser.error("Existing mixed RK4/RK3 checkpoint required")
    if "dipole_reaction_ledger" not in base:
        parser.error("Checkpoint must contain full-dipole reaction records")
    if args.steps < 1 or not np.isfinite(args.step_ns) or args.step_ns <= 0:
        parser.error("Positive step count and recording interval required")
    args.output.mkdir(parents=True, exist_ok=False)
    base["internal_step_control"] = asdict(
        InternalStepSettings(*[ErrorScale(1e-8, 1e-6) for _ in range(6)])
    )
    for key in ("reaction_ledger", "dipole_reaction_ledger"):
        if key in base:
            base[key] = _zero_ledger(base[key])
    write_checkpoint(args.output / "initial.json", base)
    results = []
    statistics = {}
    for name in ("rk4", "mixed"):
        state = copy.deepcopy(base)
        if name == "rk4":
            del state["pseudogrid_reduced"]
        start = time.monotonic()
        state, records = advance_particles(state, args.step_ns, args.steps)
        write_checkpoint(args.output / (name + ".json"), state)
        statistics[name] = dict(
            elapsed_seconds=time.monotonic() - start,
            evolved_samples=sum(
                s["evolved_count"] for r in records for s in r["source_sampling"]
            ),
            internal_steps=[
                p["internal_step_control"] for r in records for p in r["particles"]
            ],
        )
        results.append((state, records[-1]))
        print(name + " completed", flush=True)
    reference, candidate = results
    comparisons = dict(
        position_mm=relative_difference(
            np.asarray(candidate[0]["states"])[:, 1:4],
            np.asarray(reference[0]["states"])[:, 1:4],
        ),
        kinetic_spatial_momentum_native=relative_difference(
            [p["kinetic_momentum_native"][1:] for p in candidate[1]["particles"]],
            [p["kinetic_momentum_native"][1:] for p in reference[1]["particles"]],
        ),
        spin_native=relative_difference(
            np.asarray(candidate[0]["states"])[:, 8:],
            np.asarray(reference[0]["states"])[:, 8:],
        ),
        external_action_mass_exchange=relative_difference(
            [
                row["external_action_mass_exchange"]
                for row in candidate[0]["dipole_reaction_ledger"]
            ],
            [
                row["external_action_mass_exchange"]
                for row in reference[0]["dipole_reaction_ledger"]
            ],
        ),
    )
    for sector in ("total", "q_squared", "qD", "D_squared"):
        for quantity in ("impulse_native", "torque_native"):
            values = [
                [
                    (
                        row["applied_" + quantity]
                        if sector == "total"
                        else row["sectors"][sector][quantity]
                    )
                    for row in data[0]["dipole_reaction_ledger"]
                ]
                for data in (candidate, reference)
            ]
            comparisons[sector + "_" + quantity] = relative_difference(*values)
    for comparison in comparisons.values():
        comparison["reference_norm"] = comparison.pop("fine_norm")
    report = dict(
        source_checkpoint=str(args.checkpoint),
        steps=args.steps,
        step_ns=args.step_ns,
        comparisons=comparisons,
        statistics=statistics,
        fallback_counts_before=base["pseudogrid_reduced"]["fallback_count"],
        fallback_counts_after=candidate[0]["pseudogrid_reduced"]["fallback_count"],
        note="Short diagnostic continuation, not whole-trajectory convergence. Relative denominators are corresponding adaptive all-RK4 norms; reaction quantities cover only this continuation.",
    )
    write_checkpoint(args.output / "comparison.json", report)
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    main()
