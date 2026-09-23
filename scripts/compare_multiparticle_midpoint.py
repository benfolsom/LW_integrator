"""Short, reproducible post-arrival check of reduced-motion candidates.

This compares the same active-selection schedule with exact RK4 or a lower-order
passive update. It is a weak small-particle test, not a production accuracy
guarantee or a benchmark of many-particle scaling.
"""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from core.constants import C_MMNS as c
from core.momentum_center_particles import (
    advance_particles,
    configure_pseudogrid_midpoint,
    configure_pseudogrid_reference,
    configure_pseudogrid_rk3,
    initialize_particles,
)
from tests.unit.test_momentum_center_particles import fixture


def run(payload, width, steps):
    start = perf_counter()
    result, records = advance_particles(payload, width, steps)
    elapsed = perf_counter() - start
    return result, records, elapsed


def maximum_constraint(records, key):
    return max(
        float(np.linalg.norm(particle["length_time"][key]))
        for record in records
        for particle in record["particles"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--passive-method", choices=("midpoint", "rk3"), default="midpoint"
    )
    parser.add_argument("--interaction-scale", type=float, default=1.0)
    parser.add_argument("--beta-z", type=float, default=0.0)
    parser.add_argument(
        "--beta-pattern", choices=("common", "alternating"), default="common"
    )
    parser.add_argument(
        "--case",
        choices=("three_symmetric", "four_irregular", "eight_ring"),
        default="three_symmetric",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1:
        parser.error("--repeats and --steps must be positive")
    if not np.isfinite(args.interaction_scale) or args.interaction_scale <= 0:
        parser.error("--interaction-scale must be finite and positive")
    if not np.isfinite(args.beta_z) or not -1 < args.beta_z < 1:
        parser.error("--beta-z must be finite and strictly between -1 and 1")
    count = {"three_symmetric": 3, "four_irregular": 4, "eight_ring": 8}[args.case]
    active_count = 1 if count == 3 else 2
    particle_betas = (
        args.beta_z
        if args.beta_pattern == "common"
        else tuple(args.beta_z * (-1) ** index for index in range(count))
    )
    particles, states, histories, options = fixture(
        count=count,
        radius_mm=0.08 if count == 8 else 0.06,
        startup_ct_mm=0.05,
        past_ct_mm=0.4,
        angles=(0.0, 1.1, 2.9, 4.8) if count == 4 else None,
        charge_scale=args.interaction_scale,
        spin_scale=args.interaction_scale,
        beta_z=particle_betas,
    )
    initial = initialize_particles(particles, states, histories, **options)
    selection_spacing_ns = 0.005 / c
    exact_input = configure_pseudogrid_reference(
        initial, active_count, selection_spacing_ns
    )
    configure_reduced = (
        configure_pseudogrid_midpoint
        if args.passive_method == "midpoint"
        else configure_pseudogrid_rk3
    )
    reduced_input = configure_reduced(initial, active_count, selection_spacing_ns)
    reduced_name = f"passive_{args.passive_method}"
    steps = args.steps
    width = 0.25 / (c * steps)
    timings = {"exact_reference": [], reduced_name: []}
    outputs = {}
    for _ in range(args.repeats):
        for mode, payload in (
            ("exact_reference", exact_input),
            (reduced_name, reduced_input),
        ):
            result, records, elapsed = run(payload, width, steps)
            timings[mode].append(elapsed)
            outputs[mode] = result, records
    exact, exact_records = outputs["exact_reference"]
    reduced, reduced_records = outputs[reduced_name]
    schedules_match = all(
        left["pseudogrid_reference"]["selected_indices"]
        == right["pseudogrid_reduced"]["selected_indices"]
        and left["pseudogrid_reference"]["selection_updated"]
        == right["pseudogrid_reduced"]["selection_updated"]
        for left, right in zip(exact_records, reduced_records)
    )
    selection_trace = [
        dict(
            time_ns=record["time_ns"] - width,
            selected_indices=record["pseudogrid_reduced"]["selected_indices"],
        )
        for record in reduced_records
        if record["pseudogrid_reduced"]["selection_updated"]
    ]
    a, b = np.asarray(exact["states"]), np.asarray(reduced["states"])
    comparison = {}
    for key, indices in (
        ("position_mm", slice(1, 4)),
        ("canonical_spatial_momentum_native", slice(5, 8)),
        ("canonical_transverse_momentum_native", slice(5, 7)),
        ("spin_native", slice(8, 14)),
    ):
        absolute = float(np.linalg.norm(a[:, indices] - b[:, indices]))
        scale = float(np.linalg.norm(a[:, indices]))
        comparison[key] = dict(
            absolute=absolute,
            exact_norm=scale,
            relative=absolute / scale if scale else None,
        )
    for key, indices in (
        ("kinetic_spatial_momentum_native", slice(1, 4)),
        ("kinetic_transverse_momentum_native", slice(1, 3)),
    ):
        exact_momentum = np.asarray(
            [d["kinetic_momentum_native"] for d in exact_records[-1]["particles"]]
        )[:, indices]
        reduced_momentum = np.asarray(
            [d["kinetic_momentum_native"] for d in reduced_records[-1]["particles"]]
        )[:, indices]
        absolute = float(np.linalg.norm(exact_momentum - reduced_momentum))
        scale = float(np.linalg.norm(exact_momentum))
        comparison[key] = dict(
            absolute=absolute,
            exact_norm=scale,
            relative=absolute / scale if scale else None,
        )
    evidence = dict(
        case=f"weak {args.case}, reaction off, unpreserved RK4",
        passive_method=args.passive_method,
        charge_and_spin_scale=args.interaction_scale,
        beta_z=args.beta_z,
        beta_pattern=args.beta_pattern,
        particle_count=count,
        active_count=active_count,
        selection_spacing_ns=selection_spacing_ns,
        passive_constraint_budget_relative=reduced_input["pseudogrid_reduced"].get(
            "constraint_budget_relative"
        ),
        fallback_count_by_particle=reduced["pseudogrid_reduced"].get("fallback_count"),
        reduced_accepted_count_by_particle=reduced["pseudogrid_reduced"].get(
            "reduced_accepted_count"
        ),
        selection_trace=selection_trace,
        schedules_match=schedules_match,
        steps=steps,
        step_ns=width,
        end_ct_mm=steps * width * c,
        elapsed_seconds=timings,
        speedup_vs_scheduled_exact=median(timings["exact_reference"])
        / median(timings[reduced_name]),
        comparison=comparison,
        maximum_spin_constraint={
            mode: maximum_constraint(records, "spin_constraint")
            for mode, (_, records) in outputs.items()
        },
        maximum_mass_constraint={
            mode: maximum_constraint(records, "mass_constraint")
            for mode, (_, records) in outputs.items()
        },
        postarrival_steps={
            mode: sum(
                any(sample["evolved_count"] for sample in record["source_sampling"])
                for record in records
            )
            for mode, (_, records) in outputs.items()
        },
        note=(
            "The schedule equality is checked explicitly. All particles "
            "remain individually stored causal sources. Speed is measured "
            "against the scheduled exact control, not unscheduled batch RK4."
        ),
    )
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in evidence.items() if key != "selection_trace"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
