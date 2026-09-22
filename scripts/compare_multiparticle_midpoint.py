"""Short, reproducible post-arrival check of experimental passive midpoint.

This compares the same active-selection schedule with exact RK4 or midpoint
passive updates. It is a weak three-particle test, not a production accuracy
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
        "--case",
        choices=("three_symmetric", "four_irregular"),
        default="three_symmetric",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1:
        parser.error("--repeats and --steps must be positive")
    count = 3 if args.case == "three_symmetric" else 4
    active_count = 1 if count == 3 else 2
    particles, states, histories, options = fixture(
        count=count,
        radius_mm=0.06,
        startup_ct_mm=0.05,
        past_ct_mm=0.4,
        angles=None if count == 3 else (0.0, 1.1, 2.9, 4.8),
    )
    initial = initialize_particles(particles, states, histories, **options)
    selection_spacing_ns = 0.005 / c
    exact_input = configure_pseudogrid_reference(
        initial, active_count, selection_spacing_ns
    )
    reduced_input = configure_pseudogrid_midpoint(
        initial, active_count, selection_spacing_ns
    )
    steps = args.steps
    width = 0.25 / (c * steps)
    timings = {"exact_reference": [], "passive_midpoint": []}
    outputs = {}
    for _ in range(args.repeats):
        for mode, payload in (
            ("exact_reference", exact_input),
            ("passive_midpoint", reduced_input),
        ):
            result, records, elapsed = run(payload, width, steps)
            timings[mode].append(elapsed)
            outputs[mode] = result, records
    exact, exact_records = outputs["exact_reference"]
    reduced, reduced_records = outputs["passive_midpoint"]
    schedules_match = all(
        left["pseudogrid_reference"]["selected_indices"]
        == right["pseudogrid_reduced"]["selected_indices"]
        for left, right in zip(exact_records, reduced_records)
    )
    a, b = np.asarray(exact["states"]), np.asarray(reduced["states"])
    comparison = {}
    for key, indices in (
        ("position_mm", slice(1, 4)),
        ("canonical_spatial_momentum_native", slice(5, 8)),
        ("spin_native", slice(8, 14)),
    ):
        absolute = float(np.linalg.norm(a[:, indices] - b[:, indices]))
        scale = float(np.linalg.norm(a[:, indices]))
        comparison[key] = dict(
            absolute=absolute,
            exact_norm=scale,
            relative=absolute / scale if scale else None,
        )
    evidence = dict(
        case=f"weak {args.case}, reaction off, unpreserved RK4",
        particle_count=count,
        active_count=active_count,
        selection_spacing_ns=selection_spacing_ns,
        schedules_match=schedules_match,
        steps=steps,
        step_ns=width,
        end_ct_mm=steps * width * c,
        elapsed_seconds=timings,
        speedup_vs_scheduled_exact=median(timings["exact_reference"])
        / median(timings["passive_midpoint"]),
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
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
