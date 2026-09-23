"""Bounded three-particle recording-resolution check, not a conservation proof.

Run from the checkout with PYTHONPATH=. and select a new output directory.
Both grids must use the same physical fit spacing. Reaction off is the initial
many-source check; full_dipole_coupled is a separate radiation check.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from core.constants import C_MMNS as c
from core.momentum_center_particles import (
    advance_particles,
    initialize_particles,
    configure_pseudogrid_rk3,
)
from lw_integrator.nonlinear_pair import write_checkpoint
from lw_integrator.pair_history_options import (
    configure_run_history,
    configure_startup_dipole_fit,
)
from tests.unit.test_momentum_center_particles import fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--end-ct-mm", type=float, default=2.5)
    parser.add_argument("--fit-spacing-ct-mm", type=float, default=0.025)
    parser.add_argument(
        "--integration-method", choices=("rk4", "dop853"), default="dop853"
    )
    parser.add_argument("--rk3-active-count", type=int)
    parser.add_argument("--selection-spacing-ct-mm", type=float)
    parser.add_argument(
        "--startup-dipole-fit",
        choices=("interpolate", "constrained"),
        default="interpolate",
    )
    parser.add_argument(
        "--reaction", choices=("off", "full_dipole_coupled"), default="off"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.end_ct_mm) or args.end_ct_mm <= 0:
        parser.error("Positive steps and finite positive duration required")
    if args.rk3_active_count is not None and (
        args.integration_method != "rk4" or args.selection_spacing_ct_mm is None
    ):
        parser.error(
            "Mixed RK3 requires --integration-method rk4 and explicit selection spacing"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    width = args.end_ct_mm / (c * args.steps)
    particles, states, histories, options = fixture(
        reaction=args.reaction, preserved=True, method=args.integration_method
    )
    payload = initialize_particles(particles, states, histories, **options)
    payload = configure_run_history(
        payload, "auto", None, args.fit_spacing_ct_mm / c, width
    )
    payload = configure_startup_dipole_fit(payload, args.startup_dipole_fit)
    if args.rk3_active_count is not None:
        payload = configure_pseudogrid_rk3(
            payload, args.rk3_active_count, args.selection_spacing_ct_mm / c
        )
    write_checkpoint(args.output / "initial.json", payload)
    write_checkpoint(args.output / "checkpoint.json", payload)
    start = time.monotonic()
    summary = dict(
        status="running",
        requested_steps=args.steps,
        completed_steps=0,
        reaction=args.reaction,
        integration_method=args.integration_method,
        rk3_active_count=args.rk3_active_count,
        selection_spacing_ns=(
            args.selection_spacing_ct_mm / c
            if args.selection_spacing_ct_mm is not None
            else None
        ),
        selection_trace=[],
        startup_dipole_fit=args.startup_dipole_fit,
        step_ns=width,
        fit_spacing_ns=args.fit_spacing_ct_mm / c,
        end_ct_mm=args.end_ct_mm,
        postarrival_steps=0,
        first_arrival_time_ns=None,
        maximum_mass_constraint=0.0,
        maximum_spin_constraint=0.0,
    )
    try:
        for step in range(args.steps):
            payload, records = advance_particles(payload, width)
            record = records[-1]
            if "pseudogrid_reduced" in record:
                selection = record["pseudogrid_reduced"]
                if selection["selection_updated"]:
                    summary["selection_trace"].append(
                        dict(
                            time_ns=record["time_ns"] - width,
                            selected_indices=selection["selected_indices"],
                        )
                    )
                summary["fallback_count"] = payload["pseudogrid_reduced"][
                    "fallback_count"
                ]
            arrivals = sum(s["evolved_count"] for s in record["source_sampling"])
            if arrivals:
                summary["postarrival_steps"] += 1
                if summary["first_arrival_time_ns"] is None:
                    summary["first_arrival_time_ns"] = record["time_ns"]
            summary["completed_steps"] = step + 1
            summary["time_ns"] = record["time_ns"]
            summary["elapsed_seconds"] = time.monotonic() - start
            for data in record["particles"]:
                for target, key in (
                    ("maximum_mass_constraint", "mass_constraint"),
                    ("maximum_spin_constraint", "spin_constraint"),
                ):
                    summary[target] = max(
                        summary[target], float(np.linalg.norm(data["length_time"][key]))
                    )
            summary["kinetic_momentum_native"] = [
                data["kinetic_momentum_native"].tolist() for data in record["particles"]
            ]
            # Save every accepted step so a refusal still leaves useful evidence.
            write_checkpoint(args.output / "checkpoint.json", payload)
            write_checkpoint(args.output / "summary.json", summary)
            with (args.output / "accepted_steps.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            time_ns=record["time_ns"],
                            states=payload["states"],
                            kinetic_momentum_native=summary["kinetic_momentum_native"],
                            source_sampling=record["source_sampling"],
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
            if (step + 1) % 10 == 0:
                print(
                    json.dumps(
                        {
                            key: summary[key]
                            for key in (
                                "completed_steps",
                                "elapsed_seconds",
                                "postarrival_steps",
                            )
                        }
                    ),
                    flush=True,
                )
        summary["status"] = "complete"
    except Exception as error:
        summary.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        write_checkpoint(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
